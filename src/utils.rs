use num_traits::Float;
use num_complex::ComplexFloat;

use crate::mutli_indices::MultiIndices;

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

pub(super) fn argsort<T: ComplexFloat>(slice: &[T]) -> Vec<usize>
where
  T::Real: PartialOrd,
{
  let mut indices: Vec<_> = (0..(slice.len())).collect();
  indices.sort_by(|i, j| unsafe { slice.get_unchecked(*j).abs().partial_cmp(&slice.get_unchecked(*i).abs()).unwrap() });
  indices
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

pub(super) fn get_restrictions(
  indices: &MultiIndices,
  n: usize,
) -> (Vec<usize>, usize)
{
  let mut forbidden = Vec::new();
  let mut must_have = 0;
  for (i, index) in indices.into_iter().enumerate() {
    let nonzeros = index.into_iter().filter(|x| **x != 0 ).count();
    if nonzeros > n {
      forbidden.push(i);
    }
    if nonzeros == 0 {
      must_have = i;
    }
  }
  (forbidden, must_have)
}

#[cfg(test)]
mod tests {
  use super::{
    get_trunc_dim,
    build_bonds,
    argsort,
  };
  use linwrap::init_utils::BufferGenerator;
  use num_complex::{
    Complex64,
    ComplexFloat,
  };
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
  fn test_argsort()
  {
    let values: Vec<Complex64> = Complex64::random_normal(100);
    let args = argsort(&values);
    let mut prev = f64::MAX;
    assert!(args.into_iter().all(|i| {
      let curr = values[i].abs();
      let flag = prev > curr;
      prev = curr;
      flag
    }))
  }
}