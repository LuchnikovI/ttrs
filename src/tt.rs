use std::mem::swap;
use std::iter::DoubleEndedIterator;

use num_complex::{
  Complex32,
  Complex64,
  ComplexFloat,
};

use rayon::prelude::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use linwrap::{
  Matrix,
  MatrixError,
  init_utils::{
    random_normal_f32,
    random_normal_f64,
    random_normal_c32,
    random_normal_c64,
    uninit_buff_f32,
    uninit_buff_f64,
    uninit_buff_c32,
    uninit_buff_c64,
  },
};

use crate::utils::get_trunc_dim;
use crate::ttcross::CrossBuilder;

// ---------------------------------------------------------------------------------- //

#[derive(Debug)]
pub enum TTError {
  MatrixError(MatrixError),
  IncorrectIndexLength,
  OutOfBound,
}

pub type TTResult<T> = Result<T, TTError>;

impl From<MatrixError> for TTError {
  fn from(err: MatrixError) -> Self {
    Self::MatrixError(err)        
  }
}

// ---------------------------------------------------------------------------------- //

pub struct TTIter<'a, T: 'a>
{
  kernels_iter: std::slice::Iter<'a, Vec<T>>,
  right_bonds_iter: std::slice::Iter<'a, usize>,
  left_bonds_iter: std::slice::Iter<'a, usize>,
  mode_dims_iter: std::slice::Iter<'a, usize>,
}

impl<'a, T> Iterator for TTIter<'a, T> {
  type Item = (&'a Vec<T>, usize, usize, usize);
  fn next(&mut self) -> Option<Self::Item> {
    match (self.kernels_iter.next(), self.right_bonds_iter.next(), self.left_bonds_iter.next(), self.mode_dims_iter.next()) {
      (Some(ker), Some(right_bond), Some(left_bond), Some(mode_dim)) => {
          Some((ker, *right_bond, *left_bond, *mode_dim))
      },
      _ => { None },
    }
  }
}

impl<'a, T> DoubleEndedIterator for TTIter<'a, T> {
  fn next_back(&mut self) -> Option<Self::Item> {
    match (self.kernels_iter.next_back(), self.right_bonds_iter.next_back(), self.left_bonds_iter.next_back(), self.mode_dims_iter.next_back()) {
      (Some(ker), Some(right_bond), Some(left_bond), Some(mode_dim)) => {
          Some((ker, *right_bond, *left_bond, *mode_dim))
      },
      _ => { None },
    }
  }
}

impl<'a, T: 'a> IntoIterator for &'a TensorTrain<T> {
  type Item = (&'a Vec<T>, usize, usize, usize);
  type IntoIter = TTIter<'a, T>;
  fn into_iter(self) -> Self::IntoIter {
    Self::IntoIter {
      kernels_iter: self.kernels.iter(),
      right_bonds_iter: self.right_bonds.iter(),
      left_bonds_iter: self.left_bonds.iter(),
      mode_dims_iter: self.mode_dims.iter(),
    }
  }
}

pub struct TTIterMut<'a, T: 'a>
{
  kernels_iter: std::slice::IterMut<'a, Vec<T>>,
  right_bonds_iter: std::slice::IterMut<'a, usize>,
  left_bonds_iter: std::slice::IterMut<'a, usize>,
  mode_dims_iter: std::slice::IterMut<'a, usize>,
}

impl<'a, T> Iterator for TTIterMut<'a, T> {
  type Item = (&'a mut Vec<T>, &'a mut usize, &'a mut usize, &'a mut usize);
  fn next(&mut self) -> Option<Self::Item> {
    match (self.kernels_iter.next(), self.right_bonds_iter.next(), self.left_bonds_iter.next(), self.mode_dims_iter.next()) {
      (Some(ker), Some(right_bond), Some(left_bond), Some(mode_dim)) => {
        Some((ker, right_bond, left_bond, mode_dim))
      },
      _ => { None },
    }
  }
}

impl<'a, T> DoubleEndedIterator for TTIterMut<'a, T> {
  fn next_back(&mut self) -> Option<Self::Item> {
    match (self.kernels_iter.next_back(), self.right_bonds_iter.next_back(), self.left_bonds_iter.next_back(), self.mode_dims_iter.next_back()) {
      (Some(ker), Some(right_bond), Some(left_bond), Some(mode_dim)) => {
          Some((ker, right_bond, left_bond, mode_dim))
      },
      _ => { None },
    }
  }
}

impl <'a, T: 'a> IntoIterator for &'a mut TensorTrain<T> {
  type Item = (&'a mut Vec<T>, &'a mut usize, &'a mut usize, &'a mut usize);
  type IntoIter = TTIterMut<'a, T>;
  fn into_iter(self) -> Self::IntoIter {
    Self::IntoIter {
      kernels_iter: self.kernels.iter_mut(),
      right_bonds_iter: self.right_bonds.iter_mut(),
      left_bonds_iter: self.left_bonds.iter_mut(),
      mode_dims_iter: self.mode_dims.iter_mut(),
    }
  }
}

// ---------------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
pub struct TensorTrain<T> {
  pub(super) kernels:     Vec<Vec<T>>,
  pub(super) right_bonds: Vec<usize>,
  pub(super) left_bonds:  Vec<usize>,
  pub(super) mode_dims:   Vec<usize>,
}

macro_rules! tt_impl {
  (
    $complex_type:ident,
    $real_type:ident,
    $fn_gen:ident,
    $fn_uninit_buff:ident,
    $complex_zero:expr,
    $complex_one:expr,
    $real_zero:expr,
    $real_one:expr
  ) => {
    impl TensorTrain<$complex_type> {
      pub fn new_random_normal(
        mode_dims: &[usize],
        max_rank: usize,
      ) -> Self
      {
        let modes_num = mode_dims.len();
        let mut kernels = Vec::with_capacity(modes_num);
        kernels.push($fn_gen(mode_dims[0] * max_rank));
        for dim in &mode_dims[1..(modes_num - 1)] {
          kernels.push($fn_gen(dim * max_rank * max_rank));
        }
        kernels.push($fn_gen(mode_dims[modes_num - 1] * max_rank));
        let mut right_bonds = Vec::with_capacity(modes_num);
        for _ in 0..(modes_num - 1) {
          right_bonds.push(max_rank);
        }
        right_bonds.push(1);
        let mut left_bonds = Vec::with_capacity(modes_num);
        left_bonds.push(1);
        for _ in 0..(modes_num - 1) {
          left_bonds.push(max_rank);
        }
        let mode_dims = mode_dims.to_owned();
        Self { kernels, right_bonds, left_bonds, mode_dims }
      }

      pub fn get_bonds(&self) -> &[usize] {
        &self.left_bonds[1..]
      }

      pub fn conj(&mut self) {
        for ker in &mut self.kernels {
          let len = ker.len();
          let ker_matrix = Matrix::from_mut_slice(ker, 1, len).unwrap();
          unsafe { ker_matrix.conj() };
        }
      }

      pub fn iter<'a>(&'a self) -> TTIter<'a, $complex_type> {
          self.into_iter()
      }

      pub fn iter_mut<'a>(&'a mut self) -> TTIterMut<'a, $complex_type> {
        self.into_iter()
      }

      pub fn dot(&self, other: &TensorTrain<$complex_type>) -> TTResult<$complex_type> {
        let mut agr_buff = vec![$complex_one];
        let iter = self.iter().zip(other.iter());
        for (
            (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
            (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
        ) in iter {
          unsafe {
            let agr = Matrix::from_mut_slice(&mut agr_buff, rhs_left_bond, lhs_left_bond).unwrap();
            let mut new_agr_buff = $fn_uninit_buff(rhs_left_bond * lhs_mode_dim * lhs_right_bond);
            let new_agr = Matrix::from_mut_slice(&mut new_agr_buff, rhs_left_bond, lhs_mode_dim * lhs_right_bond)?;
            let lhs_ker = Matrix::from_slice(lhs_ker, lhs_left_bond, lhs_mode_dim * lhs_right_bond)?;
            new_agr.matmul_inplace(agr, lhs_ker, false, false)?;
            agr_buff = new_agr.gen_buffer();
            let mut new_agr_buff = $fn_uninit_buff(rhs_right_bond * lhs_right_bond);
            let new_agr = Matrix::from_mut_slice(&mut new_agr_buff, rhs_right_bond, lhs_right_bond)?;
            let agr = Matrix::from_mut_slice(&mut agr_buff, rhs_left_bond * lhs_mode_dim, lhs_right_bond)?;
            let rhs_ker = Matrix::from_slice(rhs_ker, rhs_left_bond * rhs_mode_dim, rhs_right_bond)?;
            new_agr.matmul_inplace(rhs_ker, agr, true, false)?;
            agr_buff = new_agr.gen_buffer();
          }
        };
        Ok(agr_buff[0])
      }

      pub fn set_into_left_canonical(&mut self) -> TTResult<$real_type> {
        let mut new_left_bond = 1;
        let mut orth_buff = unsafe { $fn_uninit_buff(1) };
        let mut lognorm = $real_zero;
        orth_buff[0] = $complex_one;
        for (ker_buff, right_bond, left_bond, mode_dim) in &mut *self {
          let orth = Matrix::from_mut_slice(&mut orth_buff, new_left_bond, *left_bond)?;
          let ker = Matrix::from_mut_slice(ker_buff, *left_bond, *mode_dim * *right_bond)?;
          let mut new_ker_buff = unsafe { $fn_uninit_buff(new_left_bond * *mode_dim * *right_bond) };
          let new_ker = Matrix::from_mut_slice(&mut new_ker_buff, new_left_bond, *mode_dim * *right_bond)?;
          unsafe { new_ker.matmul_inplace(orth, ker, false, false) }?;
          let nrows = new_left_bond * *mode_dim;
          let ncols = *right_bond;
          let min_dim = std::cmp::min(nrows, ncols);
          let mut aux_buff = unsafe { $fn_uninit_buff(min_dim * min_dim) };
          let aux = Matrix::from_mut_slice(&mut aux_buff, min_dim, min_dim)?;
          let new_ker = new_ker.reshape(nrows, ncols)?;
          unsafe { new_ker.qr(aux)? };
          if nrows > ncols {
            *left_bond = new_left_bond;
            new_left_bond = ncols;
            swap(&mut new_ker_buff, ker_buff);
            swap(&mut orth_buff, &mut aux_buff);
          } else {
            *left_bond = new_left_bond;
            *right_bond = nrows;
            new_left_bond = nrows;
            swap(&mut aux_buff, ker_buff);
            swap(&mut new_ker_buff, &mut orth_buff);
          }
          let norm: $real_type = $real_type::sqrt((&orth_buff).into_par_iter().map(|x| { (*x).abs().powi(2) }).sum());
          (&mut orth_buff).into_par_iter().for_each(|x| { *x /= norm});
          lognorm += norm.ln();
        }
        self.kernels.last_mut().unwrap().iter_mut().for_each(|x| { *x *= orth_buff[0] });
        Ok(lognorm)
      }

      pub fn eval_index(&self, index: &[usize]) -> TTResult<$complex_type> {
        let mut agr_buff = vec![$complex_one; 1];
        let mut agr = Matrix::from_slice(&agr_buff, 1, 1)?;
        if self.kernels.len() != index.len() {
          return Err(TTError::IncorrectIndexLength);
        }
        for (i, (ker_buff, right_bond, left_bond, mode_dim)) in index.into_iter().zip(self) {
          if *i >= mode_dim { return Err(TTError::OutOfBound); }
          let ker = Matrix::from_slice(ker_buff, left_bond * mode_dim, right_bond)?;
          let subker = ker.subview(((left_bond * i)..(left_bond * (i + 1)), 0..right_bond))?;
          let mut new_agr_buff = unsafe { $fn_uninit_buff(right_bond) };
          let new_agr = Matrix::from_mut_slice(&mut new_agr_buff, 1, right_bond)?;
          unsafe { new_agr.matmul_inplace(agr, subker, false, false) }?;
          swap(&mut new_agr_buff, &mut agr_buff);
          agr = new_agr.into();
        }
        Ok(agr_buff[0])
      }

      pub fn truncate_left_canonical(&mut self, delta: $real_type) -> TTResult<$real_type> {
        let mut lmbd_buff = vec![$complex_one; 1];
        let mut lmbd = Matrix::from_mut_slice(&mut lmbd_buff, 1, 1)?;
        let mut isom_buff = vec![$complex_one; 1];
        let mut isom = Matrix::from_mut_slice(&mut isom_buff, 1, 1)?;
        let mut trunc_dim = 1;
        for (ker_buff, right_bond, left_bond, mode_dim) in self.into_iter().rev() {
          let ker = Matrix::from_mut_slice(ker_buff, *left_bond * *mode_dim, *right_bond)?;
          let mut orth_buff = unsafe { $fn_uninit_buff(*left_bond * *mode_dim * trunc_dim) };
          let mut orth = Matrix::from_mut_slice(&mut orth_buff, *left_bond * *mode_dim, trunc_dim)?;
          unsafe {
            orth.matmul_inplace(ker, isom, false, false)?;
            orth.mul(lmbd)?;
          };
          orth = orth.reshape(*left_bond, *mode_dim * trunc_dim)?;
          let min_dim = std::cmp::min(*left_bond, *mode_dim * trunc_dim);
          let mut u_buff = unsafe { $fn_uninit_buff(*left_bond * min_dim) };
          let u = Matrix::from_mut_slice(&mut u_buff, *left_bond, min_dim)?;
          let mut v_dag_buff = unsafe { $fn_uninit_buff(min_dim * trunc_dim * *mode_dim) };
          let v_dag = Matrix::from_mut_slice(&mut v_dag_buff, min_dim, trunc_dim * *mode_dim)?;
          let new_lmbd_buff = unsafe { orth.svd(u, v_dag) }?;
          let new_trunc_dim = get_trunc_dim(&new_lmbd_buff, delta);
          lmbd_buff = new_lmbd_buff.into_iter().take(new_trunc_dim).map(|x| $complex_type::from(x)).collect();
          lmbd = Matrix::from_mut_slice(&mut lmbd_buff, 1, new_trunc_dim)?;
          isom_buff = unsafe { u.subview((0..(*left_bond), 0..new_trunc_dim))?.gen_buffer() };
          isom = Matrix::from_mut_slice(&mut isom_buff, *left_bond, new_trunc_dim)?;
          let mut new_ker_buff = unsafe { v_dag.subview((0..new_trunc_dim, 0..(*mode_dim * trunc_dim)))?.gen_buffer() };
          swap(ker_buff, &mut new_ker_buff);
          *left_bond = new_trunc_dim;
          *right_bond = trunc_dim;
          trunc_dim = new_trunc_dim;
        }
        Ok((lmbd_buff[0].abs()))
      }
    }
  }
}

tt_impl!(f32,       f32, random_normal_f32, uninit_buff_f32, 0.,                     1.,                     0., 1.);
tt_impl!(f64,       f64, random_normal_f64, uninit_buff_f64, 0.,                     1.,                     0., 1.);
tt_impl!(Complex32, f32, random_normal_c32, uninit_buff_c32, Complex32::new(0., 0.), Complex32::new(1., 0.), 0., 1.);
tt_impl!(Complex64, f64, random_normal_c64, uninit_buff_c64, Complex64::new(0., 0.), Complex64::new(1., 0.), 0., 1.);

macro_rules! impl_ttcross {
  ($fn_name:ident, $complex_type:ty, $real_type:ty) => {
    pub fn $fn_name(
      mode_dims: &[usize],
      max_rank: usize,
      delta: $real_type,
      f: impl Fn(&[usize]) -> $complex_type + Sync,
      sweeps_num: usize,
    ) -> TTResult<TensorTrain<$complex_type>>
    {
      let kers_num = mode_dims.len();
      let mut builder = CrossBuilder::<$complex_type>::new(max_rank, delta, mode_dims);
      for _ in 0..(kers_num * sweeps_num) {
        builder.next(&f)?;
      }
      Ok(builder.to_tt())
    }
  };
}

impl_ttcross!(ttcross_f32, f32,       f32);
impl_ttcross!(ttcross_f64, f64,       f64);
impl_ttcross!(ttcross_c32, Complex32, f32);
impl_ttcross!(ttcross_c64, Complex64, f64);

#[cfg(test)]
mod tests {
  use super::*;
  macro_rules! test_dot_and_canonical {
    ($complex_type:ident) => {
      let mut tt = TensorTrain::<$complex_type>::new_random_normal(&[2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      let tt_clone = tt.clone();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let log_dot = tt.dot(&tt_conj).unwrap().ln().abs();
      let log_norm = tt.set_into_left_canonical().unwrap();
      assert!((log_dot - 2. * log_norm).abs() < 1e-5);
      tt_conj = tt.clone();
      tt_conj.conj();
      let dot = tt.dot(&tt_conj).unwrap().abs();
      assert!((dot - 1.).abs() < 1e-5);
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      let diff = tt.dot(&tt_conj).unwrap() +
                 tt_clone.dot(&tt_clone_conj).unwrap() / (2. * log_norm).exp() -
                 tt.dot(&tt_clone_conj).unwrap() / log_norm.exp()-
                 tt_clone.dot(&tt_conj).unwrap() / log_norm.exp();
      assert!(diff.abs() < 1e-5);
      assert!((
        tt.eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap() -
        tt_clone.eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap() / log_norm.exp()
      ).abs() < 1e-5)
    };
  }

  #[test]
  fn test_dot_and_canonical()
  {
    test_dot_and_canonical!(f32);
    test_dot_and_canonical!(f64);
    test_dot_and_canonical!(Complex32);
    test_dot_and_canonical!(Complex64);
  }

  macro_rules! test_truncation {
    ($complex_type:ident) => {
      let mut tt = TensorTrain::<$complex_type>::new_random_normal(&[2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      tt.set_into_left_canonical().unwrap();
      let tt_clone = tt.clone();
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      tt.truncate_left_canonical(0.01).unwrap();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let diff = tt.dot(&tt_conj).unwrap() +
                 tt_clone.dot(&tt_clone_conj).unwrap() -
                 tt.dot(&tt_clone_conj).unwrap() -
                 tt_clone.dot(&tt_conj).unwrap();
      assert!(diff.abs() < 1e-5)
    };
  }
  #[test]
  fn test_truncation() {
    test_truncation!(f32);
    test_truncation!(f64);
    test_truncation!(Complex32);
    test_truncation!(Complex64);
  }
}
