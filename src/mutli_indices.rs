use std::ops::{Index, IndexMut};

use rayon::prelude::{
    IndexedParallelIterator,
    IntoParallelIterator,
    ParallelIterator,
};
use rand::thread_rng;
use rand::prelude::SliceRandom;
use serde::{Serialize, Deserialize};

// -------------------------------------- MultiIndices --------------------------------------- //

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct MultiIndices
{
    indices: Vec<Vec<usize>>,
}

// ---------------------------------- Traits implementations ---------------------------------- //

impl Default for MultiIndices {
    fn default() -> Self {
        Self { indices: vec![vec![]] }
    }
}

impl From<Vec<Vec<usize>>> for MultiIndices {
    fn from(value: Vec<Vec<usize>>) -> Self {
        Self { indices: value }
    }
}

impl From<MultiIndices> for Vec<Vec<usize>> {
    fn from(value: MultiIndices) -> Self {
        value.release()
    }
}

impl IntoIterator for MultiIndices {
    type Item = Vec<usize>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.indices.into_iter()
    }
}

impl<'item> IntoIterator for &'item MultiIndices {
    type Item = &'item Vec<usize>;
    type IntoIter = std::slice::Iter<'item, Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter()
    }
}

impl AsRef<[Vec<usize>]> for MultiIndices
{
    fn as_ref(&self) -> &[Vec<usize>] {
        &self.indices
    }
}

// ----------------------------------------- Methods ------------------------------------------ //

impl MultiIndices {

    pub(super) fn new() -> Self
    {
        Default::default()
    }

    pub(super) fn new_one_side(dim: usize) -> Self
    {
        Self { indices: (0..dim).map(|x| vec![x]).collect::<Vec<_>>() }
    }

    pub(super) fn product(&self, other: &Self) -> Self
    {
        let indices = other.into_iter().flat_map(|x| {
            self.into_iter().map(|y| {
                y.into_iter().chain(x.into_iter()).map(|e| *e).collect() 
            }).collect::<Vec<_>>()
        }).collect();
        Self { indices }
    }

    pub(super) fn random_subset(self, subset_size: usize) -> Self
    {
        let mut rng = thread_rng();
        let indices_num = self.indices.len();
        let mut order: Vec<_> = (0..indices_num).collect();
        order[1..].shuffle(&mut rng); // Preserves [0, 0, ..., 0] index
        if order.len() > subset_size {
            order.resize(subset_size, 0);
        }
        order.sort_unstable();
        let indices = order.into_iter().map(|i| unsafe { self.indices.get_unchecked(i).to_owned() } ).collect::<Vec<_>>();
        Self { indices }
    }

    pub(super) fn filter(self, n: usize) -> Self
    {
        let indices = self.indices
            .into_iter()
            .filter(|index| {
                let nonzeros = index.into_iter().filter(|x| **x != 0).count();
                nonzeros <= n
            })
            .collect::<Vec<_>>();
        Self { indices }
    }

    pub(super) fn len(&self) -> usize
    {
        self.indices.len()
    }

    pub(super) fn release(self) -> Vec<Vec<usize>>
    {
        self.indices
    }
}

// ------------------------------------------ MultiIndicesSet ----------------------------------------- //

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct MultiIndicesSet
{
    indices: Vec<(MultiIndices, MultiIndices)>,
}

// -------------------------------------- Traits implementations ------------------------------------- //

impl Index<usize> for MultiIndicesSet {
    type Output = (MultiIndices, MultiIndices);
    fn index(&self, index: usize) -> &Self::Output {
        &self.indices[index]
    }
}

impl IndexMut<usize> for MultiIndicesSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.indices[index]
    }
}

impl<'item> IntoIterator for &'item MultiIndicesSet {
    type Item = &'item (MultiIndices, MultiIndices);
    type IntoIter = std::slice::Iter<'item, (MultiIndices, MultiIndices)>;
    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter()
    }
}

impl<'item> IntoIterator for &'item mut MultiIndicesSet {
    type Item = &'item mut (MultiIndices, MultiIndices);
    type IntoIter = std::slice::IterMut<'item, (MultiIndices, MultiIndices)>;
    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter_mut()
    }
}

// -------------------------------------------- Methods ---------------------------------------------- //

impl MultiIndicesSet {

    pub(super) fn build_random(
        mode_dims: &[usize],
        max_rank: usize,
        n: usize,
    ) -> Self
    {
        let kers_num = mode_dims.len();
        let mut all_left_indices = Vec::with_capacity(kers_num);
        all_left_indices.push(MultiIndices::new());
        let iter = mode_dims
            .into_iter()
            .take(kers_num-1);
        for dim in iter {
            let left_indices_ref = all_left_indices.last().unwrap();
            let cur_indices = MultiIndices::new_one_side(*dim);
            let mut left_new_indices = left_indices_ref.product(&cur_indices);
            if n != usize::MAX {
              left_new_indices = left_new_indices.filter(n);
            }
            if left_new_indices.len() > max_rank {
                left_new_indices = left_new_indices.random_subset(max_rank);
            }
            all_left_indices.push(left_new_indices);
        }
        let mut all_right_indices = vec![Default::default(); kers_num];
        let iter = mode_dims
            .into_iter().rev()
            .enumerate()
            .take(kers_num-1);
        for (idx, dim) in iter {
            let right_indices_ref = &all_right_indices[kers_num - 1 - idx];
            let cur_indices = MultiIndices::new_one_side(*dim);
            let mut right_new_indices = cur_indices.product(right_indices_ref);
            if n != usize::MAX {
                right_new_indices = right_new_indices.filter(n);
            }
            if right_new_indices.len() > max_rank {
                right_new_indices = right_new_indices.random_subset(max_rank);
            }
            all_right_indices[kers_num - 2 - idx] = right_new_indices;
        }
        let len = all_right_indices.len() - 1;
        for (lhs, rhs) in all_left_indices[1..].iter_mut().zip(&mut all_right_indices[..len])
        {
            if lhs.len() > rhs.len() {
                let mut tmp = MultiIndices::new();
                std::mem::swap(&mut tmp, lhs);
                *lhs = tmp.random_subset(rhs.len());
            } else if rhs.len() > lhs.len() {
                let mut tmp = MultiIndices::new();
                std::mem::swap(&mut tmp, rhs);
                *rhs = tmp.random_subset(lhs.len());
            }
        }
        let indices = all_left_indices.into_iter().zip(all_right_indices).collect::<Vec<_>>();
        Self { indices }
    }

    fn get_indices(&self) -> &[(MultiIndices, MultiIndices)]
    {
        &self.indices
    }

    pub(super) fn len(&self) -> usize
    {
        self.indices.len()
    }

    pub(super) fn get_bonds(&self) -> (Vec<usize>, Vec<usize>)
    {
        let mut left_bonds = Vec::with_capacity(self.len());
        let mut right_bonds = Vec::with_capacity(self.len());
        for (lhs, rhs) in self.into_iter() {
            left_bonds.push(lhs.len());
            right_bonds.push(rhs.len());
        }
        (left_bonds, right_bonds)
    }
}

// ------------------------------------------------------------ ParIter ---------------------------------------------------------- //

pub(super) fn indices_par_iter(
    pair: (MultiIndices, MultiIndices),
    is_c_layout: bool,
) -> impl IndexedParallelIterator<Item = Vec<usize>>
{
    let rows: Vec<Vec<usize>> = pair.0.release();
    let cols: Vec<Vec<usize>> = pair.1.release();
    let rows_size = rows.len();
    let cols_size = cols.len();
    let max_stride = if is_c_layout { cols_size } else { rows_size };
    let size = cols_size * rows_size;
    (0..size).into_par_iter()
    .map(move |idx| {
        let (row_idx, col_idx) = if is_c_layout {
            (idx / max_stride, idx % max_stride)
        } else {
            (idx % max_stride, idx / max_stride)
        };
        unsafe {
            rows.get_unchecked(row_idx).into_iter().map(|x| *x)
                .chain(
                    cols.get_unchecked(col_idx).into_iter().map(|x| *x)
                )
                .collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use rayon::iter::ParallelIterator;

    use crate::mutli_indices::{MultiIndices, MultiIndicesSet, indices_par_iter};
    use crate::utils::build_bonds;

    #[test]
    fn test_indices_prod() {
      let lhs: MultiIndices = vec![
        vec![1, 2, 3, 4, 5],
        vec![2, 3, 4, 5, 6],
        vec![6, 5, 4, 3, 2],
        vec![1, 1, 2, 2, 3],
      ].into();
      let rhs: MultiIndices = vec![
        vec![11, 12, 13],
        vec![14, 15, 16],
      ].into();
      let result: MultiIndices = vec![
        vec![1, 2, 3, 4, 5, 11, 12, 13],
        vec![2, 3, 4, 5, 6, 11, 12, 13],
        vec![6, 5, 4, 3, 2, 11, 12, 13],
        vec![1, 1, 2, 2, 3, 11, 12, 13],
        vec![1, 2, 3, 4, 5, 14, 15, 16],
        vec![2, 3, 4, 5, 6, 14, 15, 16],
        vec![6, 5, 4, 3, 2, 14, 15, 16],
        vec![1, 1, 2, 2, 3, 14, 15, 16],
      ].into();
      assert_eq!(result, lhs.product(&rhs));
      let lhs: MultiIndices = vec![vec![]].into();
      let rhs: MultiIndices = vec![vec![]].into();
      assert_eq!(Into::<MultiIndices>::into(vec![vec![]]), lhs.product(&rhs));
      let lhs: MultiIndices = vec![vec![1, 2, 3], vec![4, 5, 6]].into();
      let rhs: MultiIndices = vec![vec![]].into();
      assert_eq!(Into::<MultiIndices>::into(vec![vec![1, 2, 3], vec![4, 5, 6]]), lhs.product(&rhs));
      let lhs: MultiIndices = vec![vec![]].into();
      let rhs: MultiIndices = vec![vec![1, 2, 3], vec![4, 5, 6]].into();
      assert_eq!(Into::<MultiIndices>::into(vec![vec![1, 2, 3], vec![4, 5, 6]]), lhs.product(&rhs));
    }

    fn test_random_indices_and_iter_(mode_dims: &[usize], n: usize) {
        let rank = 15;
        let (left_bonds, right_bonds) = build_bonds(mode_dims, rank);
        let mi_set = MultiIndicesSet::build_random(mode_dims, rank, n);
        if n == usize::MAX {
            let (left_bonds_from_indices, right_bonds_from_indices) = mi_set.get_bonds();
            assert_eq!(left_bonds_from_indices, left_bonds);
            assert_eq!(right_bonds_from_indices, right_bonds);
        }
        assert_eq!(mi_set.get_indices().len(), mode_dims.len());
        let mut right_size = 1;
        for ((left_bond, right_bond), (left_indices, right_indices)) in left_bonds.into_iter().zip(right_bonds).zip(&mi_set) {
            assert_eq!(left_indices.len(), right_size);
            right_size = right_indices.len();
            let indices_from_iter = indices_par_iter((left_indices.clone(), right_indices.clone()), false).collect::<Vec<_>>();
            let indices_from_product = left_indices.product(&right_indices).release();
            assert_eq!(indices_from_iter, indices_from_product);
            for (left_index, right_index) in (&left_indices).into_iter().zip(right_indices) {
                assert_eq!(left_index.len() + right_index.len() + 1, mode_dims.len());
            }
            left_indices.as_ref()[0].iter().for_each(|x| {
                assert_eq!(*x, 0);
            });
            right_indices.as_ref()[0].iter().for_each(|x| {
                assert_eq!(*x, 0);
            });
            for left_index in left_indices {
                let nonzeros = left_index.into_iter().filter(|x| **x != 0).count();
                assert!(nonzeros <= n);
            }
            for right_index in right_indices {
                let nonzeros = right_index.into_iter().filter(|x| **x != 0).count();
                assert!(nonzeros <= n);
            }
            if n == usize::MAX {
                assert_eq!(left_indices.len(), left_bond);
                assert_eq!(right_indices.len(), right_bond);
            }
        }
    }

    #[test]
    fn test_random_indices_and_iter() {
        test_random_indices_and_iter_(&[1], 0);
        test_random_indices_and_iter_(&[5], 0);
        test_random_indices_and_iter_(&[1, 1, 1, 1], 0);
        test_random_indices_and_iter_(&[2, 3, 1, 2, 3, 5, 6, 3, 2, 1, 2, 3, 4, 6, 7], 0);
        test_random_indices_and_iter_(&[20, 30, 10, 20, 30, 50, 60, 30, 20, 10, 20, 30, 40, 60, 70], 0);
        test_random_indices_and_iter_(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 0);
        test_random_indices_and_iter_(&[1], 1);
        test_random_indices_and_iter_(&[5], 1);
        test_random_indices_and_iter_(&[1, 1, 1, 1], 1);
        test_random_indices_and_iter_(&[2, 3, 1, 2, 3, 5, 6, 3, 2, 1, 2, 3, 4, 6, 7], 1);
        test_random_indices_and_iter_(&[20, 30, 10, 20, 30, 50, 60, 30, 20, 10, 20, 30, 40, 60, 70], 1);
        test_random_indices_and_iter_(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 1);
        test_random_indices_and_iter_(&[1], 2);
        test_random_indices_and_iter_(&[5], 2);
        test_random_indices_and_iter_(&[1, 1, 1, 1], 2);
        test_random_indices_and_iter_(&[2, 3, 1, 2, 3, 5, 6, 3, 2, 1, 2, 3, 4, 6, 7], 2);
        test_random_indices_and_iter_(&[20, 30, 10, 20, 30, 50, 60, 30, 20, 10, 20, 30, 40, 60, 70], 2);
        test_random_indices_and_iter_(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 2);
        test_random_indices_and_iter_(&[1], usize::MAX);
        test_random_indices_and_iter_(&[5], usize::MAX);
        test_random_indices_and_iter_(&[1, 1, 1, 1], usize::MAX);
        test_random_indices_and_iter_(&[2, 3, 1, 2, 3, 5, 6, 3, 2, 1, 2, 3, 4, 6, 7], usize::MAX);
        test_random_indices_and_iter_(&[20, 30, 10, 20, 30, 50, 60, 30, 20, 10, 20, 30, 40, 60, 70], usize::MAX);
        test_random_indices_and_iter_(&[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], usize::MAX);
    }
}
