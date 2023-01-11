use num_traits::Float;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::ParallelIterator;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator};

pub(super) fn get_trunc_dim<T: Float>(lmbd: &[T], delta_local: T) -> usize
{
  let mut acc = T::zero();
  let mut counter = 0;
  for l in lmbd.into_iter().rev() {
    acc = acc + l.powi(2);
    if acc.sqrt() > delta_local { break; }
    counter += 1;
  }
  lmbd.len() - counter
}

pub(super) fn build_bonds(mode_dims: &[usize], rank: usize) -> (Vec<usize>, Vec<usize>)
{
  let kers_num = mode_dims.len();
  let mut bonds = vec![rank; kers_num + 1];
  bonds[0] = 1;
  *bonds.last_mut().unwrap() = 1;
  let mut prev_bond = 1;
  for (dim, bond) in mode_dims.iter().zip((&mut bonds[1..kers_num]).into_iter()) {
    if prev_bond * *dim > rank {
      break;
    } else {
      *bond = prev_bond * *dim;
      prev_bond = *bond;
    }
  }
  prev_bond = 1;
  for (dim, bond) in mode_dims.iter().rev().zip((&mut bonds[1..kers_num]).into_iter().rev()) {
    if prev_bond * *dim > rank {
      break;
    } else {
      *bond = std::cmp::min(prev_bond * *dim, *bond);
      prev_bond = *bond;
    }
  }
  (bonds[..kers_num].to_owned(), bonds[1..].to_owned())
}

pub(super) fn indices_prod(
  lhs: &[Vec<usize>],
  rhs: &[Vec<usize>],
) -> Vec<Vec<usize>>
{
  if lhs.is_empty() {
    return rhs.to_owned();
  }
  if rhs.is_empty() {
    return lhs.to_owned();
  }
  rhs.into_iter().flat_map(|x| {
    lhs.into_iter().map(|y| { 
      y.into_iter().chain(x.into_iter()).map(|z| *z).collect() 
    })
  }).collect()
}

fn random_indices_subset(
  indices: &[Vec<usize>],
  subset_size: usize,
) -> Vec<Vec<usize>>
{
  let mut rng = thread_rng();
  let indices_num = indices.len();
  let mut order: Vec<_> = (0..indices_num).collect();
  order.shuffle(&mut rng);
  order.resize(subset_size, 0);
  order.sort_unstable();
  order.into_iter().map(|i| unsafe { indices.get_unchecked(i).to_owned() } ).collect()
}

pub(super) fn build_random_indices(
  mode_dims: &[usize],
  left_bonds: &[usize],
  right_bonds: &[usize],
) -> (Vec<Vec<Vec<usize>>>, Vec<Vec<Vec<usize>>>)
{
  let kers_num = mode_dims.len();
  let mut all_left_indices = Vec::with_capacity(kers_num);
  all_left_indices.push(vec![]);
  let iter = mode_dims
    .into_iter()
    .zip(right_bonds.into_iter())
    .take(kers_num-1);
  for (dim, right_bond) in iter {
    let left_indices_ref = all_left_indices.last_mut().unwrap();
    let cur_indices: Vec<Vec<_>> = (0..*dim).map(|x| vec![x]).collect();
    let left_new_indices = indices_prod(left_indices_ref, &cur_indices);
    let left_new_indices = random_indices_subset(&left_new_indices, *right_bond);
    all_left_indices.push(left_new_indices);
  }
  let mut all_right_indices = vec![vec![]; kers_num];
  all_right_indices[kers_num-1] = vec![];
  let iter = mode_dims
    .into_iter().rev()
    .enumerate()
    .zip(left_bonds.into_iter().rev())
    .take(kers_num-1);
  for ((length, dim), left_bond) in iter {
    let right_indices_ref = &mut all_right_indices[kers_num - 1 - length];
    let cur_indices: Vec<Vec<_>> = (0..*dim).map(|x| vec![x]).collect();
    let right_new_indices = indices_prod(&cur_indices, right_indices_ref);
    let right_new_indices = random_indices_subset(&right_new_indices, *left_bond);
    all_right_indices[kers_num - 2 - length] = right_new_indices;
  }
  (all_left_indices, all_right_indices)
}

#[inline]
pub(super) fn get_indices_iter(
  first_indices: &[Vec<usize>],
  last_indices: &[Vec<usize>],
  is_reverse:bool,
) -> impl IndexedParallelIterator<Item = Vec<usize>> 
{
  let first_indices = if first_indices.is_empty() { vec![vec![]] } else { first_indices.to_owned() };
  let last_indices = if last_indices.is_empty() { vec![vec![]] } else { last_indices.to_owned() };
  let first_size = first_indices.len();
  let last_size = last_indices.len();
  let size = first_size * last_size;
  (0..size).into_par_iter()
    .map(move |mut idx| {
      let first_i = idx % first_size;
      idx /= first_size;
      let last_i = idx % last_size;
      if !is_reverse {
        unsafe {
          first_indices.get_unchecked(first_i).into_iter().map(|x| *x)
            .chain(
              last_indices.get_unchecked(last_i).into_iter().map(|x| *x)
            )
            .collect()
        }
      } else {
        unsafe {
          last_indices.get_unchecked(last_i).into_iter().map(|x| *x)
          .chain(
            first_indices.get_unchecked(first_i).into_iter().map(|x| *x)
          )
          .collect()
        }
      }
    })
}

#[cfg(test)]
mod tests {
  use super::{get_trunc_dim, build_bonds, indices_prod, build_random_indices, get_indices_iter};
  use rayon::iter::ParallelIterator;
  #[test]
  fn test_get_trunc_dim() {
    let lmbd = [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.];
    assert_eq!(10, get_trunc_dim(&lmbd, 0.9999));
    assert_eq!(9, get_trunc_dim(&lmbd, 1.0001));
    assert_eq!(9, get_trunc_dim(&lmbd, 5f32.sqrt() - 0.0001));
    assert_eq!(8, get_trunc_dim(&lmbd, 5f32.sqrt() + 0.0001));
    assert_eq!(8, get_trunc_dim(&lmbd, 14f32.sqrt() - 0.0001));
    assert_eq!(7, get_trunc_dim(&lmbd, 14f32.sqrt() + 0.0001));
    assert_eq!(7, get_trunc_dim(&lmbd, 30f32.sqrt() - 0.0001));
    assert_eq!(6, get_trunc_dim(&lmbd, 30f32.sqrt() + 0.0001));
    assert_eq!(6, get_trunc_dim(&lmbd, 55f32.sqrt() - 0.0001));
    assert_eq!(5, get_trunc_dim(&lmbd, 55f32.sqrt() + 0.0001));
    assert_eq!(5, get_trunc_dim(&lmbd, 91f32.sqrt() - 0.0001));
    assert_eq!(4, get_trunc_dim(&lmbd, 91f32.sqrt() + 0.0001));
  }

  #[test]
  fn test_build_bonds() {
    let mode_dims = [2, 1, 3, 4, 2, 1, 3, 2, 5, 3, 2, 4, 2, 2];
    let rank = 30;
    let (left_bonds, right_bonds) = build_bonds(&mode_dims, rank);
    assert_eq!(&left_bonds[..], &[1, 2, 2, 6, 24, 30, 30, 30, 30, 30, 30, 16, 4, 2]);
    assert_eq!(&right_bonds[..], &[2, 2, 6, 24, 30, 30, 30, 30, 30, 30, 16, 4, 2, 1]);
  }

  #[test]
  fn test_indices_prod() {
    let lhs = vec![
      vec![1, 2, 3, 4, 5],
      vec![2, 3, 4, 5, 6],
      vec![6, 5, 4, 3, 2],
      vec![1, 1, 2, 2, 3],
    ];
    let rhs = vec![
      vec![11, 12, 13],
      vec![14, 15, 16],
    ];
    let result = vec![
      vec![1, 2, 3, 4, 5, 11, 12, 13],
      vec![2, 3, 4, 5, 6, 11, 12, 13],
      vec![6, 5, 4, 3, 2, 11, 12, 13],
      vec![1, 1, 2, 2, 3, 11, 12, 13],
      vec![1, 2, 3, 4, 5, 14, 15, 16],
      vec![2, 3, 4, 5, 6, 14, 15, 16],
      vec![6, 5, 4, 3, 2, 14, 15, 16],
      vec![1, 1, 2, 2, 3, 14, 15, 16],
    ];
    assert_eq!(result, indices_prod(&lhs, &rhs));
  }

  fn test_build_random_indices_(mode_dims: &[usize]) {
    let (left_bonds, right_bonds) = build_bonds(mode_dims, 15);
    let (all_left_indices, all_right_indices) = build_random_indices(mode_dims, &left_bonds, &right_bonds);
    assert_eq!(all_left_indices.len(), mode_dims.len());
    assert_eq!(all_right_indices.len(), mode_dims.len());
    let iter = mode_dims.into_iter().zip(&all_left_indices[1..]).zip(&left_bonds[1..]);
    for ((dim, left_indices), left_bond) in iter {
      assert_eq!(*left_bond, left_indices.len());
      for left_index in left_indices {
        assert!(*left_index.last().unwrap() < *dim);
      }
    }
    let iter = mode_dims.into_iter().rev().zip(all_right_indices.iter().rev().skip(1)).zip(right_bonds.iter().rev().skip(1));
    for ((dim, right_indices), right_bond) in iter {
      assert_eq!(*right_bond, right_indices.len());
      for right_index in right_indices {
        assert!(right_index[0] < *dim);
      }
    }
  }

  #[test]
  fn test_build_random_indices() {
    test_build_random_indices_(&[1]);
    test_build_random_indices_(&[5]);
    test_build_random_indices_(&[1, 1, 1, 1]);
    test_build_random_indices_(&[2, 3, 1, 2, 3, 5, 6, 3, 2, 1, 2, 3, 4, 6, 7]);
  }

  #[test]
  fn test_get_indices_iter() {
    let first = vec![
      vec![0, 1, 2, 3],
      vec![3, 2, 1, 0],
      vec![1, 1, 1, 1],
    ];
    let last = vec![
      vec![1, 1, 1],
      vec![2, 2, 2],
    ];
    let true_indices = vec![
      vec![0, 1, 2, 3, 1, 1, 1],
      vec![3, 2, 1, 0, 1, 1, 1],
      vec![1, 1, 1, 1, 1, 1, 1],
      vec![0, 1, 2, 3, 2, 2, 2],
      vec![3, 2, 1, 0, 2, 2, 2],
      vec![1, 1, 1, 1, 2, 2, 2],
    ];
    let indices: Vec<_> = get_indices_iter(&first, &last, false).collect();
    assert_eq!(&true_indices, &indices);
    let true_indices = vec![
      vec![1, 1, 1, 0, 1, 2, 3],
      vec![1, 1, 1, 3, 2, 1, 0],
      vec![1, 1, 1, 1, 1, 1, 1],
      vec![2, 2, 2, 0, 1, 2, 3],
      vec![2, 2, 2, 3, 2, 1, 0],
      vec![2, 2, 2, 1, 1, 1, 1],
    ];
    let indices: Vec<_> = get_indices_iter(&first, &last, true).collect();
    assert_eq!(&true_indices, &indices);
    let first = vec![
    ];
    let last = vec![
      vec![1, 1, 1, 5],
      vec![2, 2, 2, 5],
      vec![3, 2, 1, 4],
    ];
    let true_indices = last.clone();
    let indices: Vec<_> = get_indices_iter(&first, &last, true).collect();
    assert_eq!(&true_indices, &indices);
    let last = vec![
      ];
      let first = vec![
        vec![1, 1, 1, 5],
        vec![2, 2, 2, 5],
        vec![3, 2, 1, 4],
      ];
      let true_indices = first.clone();
      let indices: Vec<_> = get_indices_iter(&first, &last, true).collect();
      assert_eq!(&true_indices, &indices);
  }
}