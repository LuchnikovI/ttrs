use std::mem::swap;
use std::iter::DoubleEndedIterator;

use num_complex::{
  Complex,
  Complex32,
  Complex64,
  ComplexFloat,
};

/*use rayon::prelude::IntoParallelIterator;
use rayon::iter::ParallelIterator;*/

use linwrap::{
  NDArray,
  NDArrayError,
  init_utils::{
    uninit_buff_f32,
    uninit_buff_f64,
    uninit_buff_c32,
    uninit_buff_c64,
  },
};

use crate::utils::get_trunc_dim;

// ---------------------------------------------------------------------------------- //

#[derive(Debug)]
pub enum TTError {

  /// This error appears when error happens at the level of NDArray
  NDArrayError(NDArrayError),

  /// This error appears when two Tensor Trains have different length
  /// while it is required them to be equal.
  LengthsMismatch,

  /// This error appears when one sends an index of incorrect length to the
  /// eval method.
  IncorrectIndexLength,

  /// This error appears when one sends an index that lies out of bound
  /// of a Tensor Train to the eval method.
  OutOfBound,

  /// This error appears when local a mode dimensions of two Tensor Trains do
  /// not match each other, while it is required them to be.
  IncorrectLocalDim,
}

pub type TTResult<T> = Result<T, TTError>;

impl From<NDArrayError> for TTError {
  fn from(err: NDArrayError) -> Self {
    Self::NDArrayError(err)        
  }
}

// ---------------------------------------------------------------------------------- //

pub struct TTIter<'a, T: 'a>
{
  pub(super) kernels_iter: std::slice::Iter<'a, Vec<T>>,
  pub(super) right_bonds_iter: std::slice::Iter<'a, usize>,
  pub(super) left_bonds_iter: std::slice::Iter<'a, usize>,
  pub(super) mode_dims_iter: std::slice::Iter<'a, usize>,
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

pub struct TTIterMut<'a, T: 'a>
{
  pub(super) kernels_iter: std::slice::IterMut<'a, Vec<T>>,
  pub(super) right_bonds_iter: std::slice::IterMut<'a, usize>,
  pub(super) left_bonds_iter: std::slice::IterMut<'a, usize>,
  pub(super) mode_dims_iter: std::slice::IterMut<'a, usize>,
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

// ---------------------------------------------------------------------------------- //

macro_rules! tt_impl {
  (
    $trait_name:ident,
    $complex_type:ident,
    $real_type:ident,
    $fn_gen:ident,
    $fn_uninit_buff:ident,
    $fn_uninit_lmbd:ident,
    $complex_zero:expr,
    $complex_one:expr,
    $real_zero:expr,
    $real_one:expr
  ) => {
    pub trait $trait_name
    {
      type Buff: AsMut<[$complex_type]> + AsRef<[$complex_type]>;
      type Kers: AsMut<[Self::Buff]> + AsRef<[Self::Buff]>;

      /// This method generates a new Tensor Train with random kernels.
      /// As an input it takes a vector with dimensions of each mode and
      /// maximal TT rank.
      fn new_random(
        mode_dims: Vec<usize>,
        max_rank:  usize,
      ) -> Self;

      /// This method returns a slice with kernels.
      fn get_kernels(&self) -> &[Self::Buff];

      /// This method returns a slice with left bond dimensions.
      fn get_left_bonds(&self) -> &[usize];

      /// This method returns a slice with right bond dimensions.
      fn get_right_bonds(&self) -> &[usize];

      /// This method returns a slice with modes dimensions.
      fn get_mode_dims(&self) -> &[usize];
      
      /// This method returns number of modes.
      fn get_len(&self) -> usize;

      /// This method returns an iterator over components of a Tensor Train.
      fn iter<'a>(&'a self) -> TTIter<'a, $complex_type>;

      /// This method returns a mutable slice with kernels.
      /// Safety: sizes of kernels buffers and corresponding modes and bonds dimensions
      /// must be consistent with each other.
      unsafe fn get_kernels_mut(&mut self) -> &mut [Self::Buff];

      /// This method returns a mutable slice with left bond dimensions.
      /// Safety: sizes of kernels buffers and corresponding modes and bonds dimensions
      /// must be consistent with each other.
      unsafe fn get_left_bonds_mut(&mut self) -> &mut [usize];

      /// This method returns a mutable slice with right bond dimensions.
      /// Safety: sizes of kernels buffers and corresponding modes and bonds dimensions
      /// must be consistent with each other.
      unsafe fn get_right_bonds_mut(&mut self) -> &mut [usize];

      /// This method returns a mutable iterator over components of a Tensor Train.
      /// Safety: sizes of kernels buffers and corresponding modes and bonds dimensions
      /// must be consistent with each other.
      unsafe fn iter_mut<'a>(&'a mut self) -> TTIterMut<'a, $complex_type>;

      /// This method returns internal bond dimensions of a Tensor Train.
      fn get_bonds(&self) -> &[usize] {
        &self.get_left_bonds()[1..]
      }

      /// This method conjugates elements of a tensor inplace.
      fn conj(&mut self) {
        for ker in unsafe { self.get_kernels_mut() } {
          let len = ker.as_ref().len();
          let ker_arr = NDArray::from_mut_slice(ker.as_mut(), [len]).unwrap();
          unsafe { ker_arr.conj() };
        }
      }

      /// This method returns a natural logarithm of dot product of two Tensor Trains.
      /// The logarithm is necessary to make computation stable, when the value of the
      /// dot is exponentially big.
      /// It returns error when shapes of tensors do not match each other.
      fn log_dot(&self, other: &Self) -> TTResult<Complex<$real_type>> {
        if self.get_len() != other.get_len() { return Err(TTError::LengthsMismatch); }
        let mut agr_buff = vec![$complex_one];
        let mut agr_val = $complex_zero;
        let iter = self.iter().zip(other.iter());
        for (
            (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
            (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
        ) in iter {
          unsafe {
            let agr = NDArray::from_mut_slice(&mut agr_buff, [rhs_left_bond, lhs_left_bond])?;
            let mut new_agr_buff = $fn_uninit_buff(rhs_left_bond * lhs_mode_dim * lhs_right_bond);
            let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [rhs_left_bond, lhs_mode_dim * lhs_right_bond])?;
            let lhs_ker = NDArray::from_slice(lhs_ker, [lhs_left_bond, lhs_mode_dim * lhs_right_bond])?;
            new_agr.matmul_inplace(agr, lhs_ker)?;
            (agr_buff, _) = new_agr.gen_f_array();
            let mut new_agr_buff = $fn_uninit_buff(rhs_right_bond * lhs_right_bond);
            let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [rhs_right_bond, lhs_right_bond])?;
            let agr = NDArray::from_mut_slice(&mut agr_buff, [rhs_left_bond * lhs_mode_dim, lhs_right_bond])?;
            let rhs_ker = NDArray::from_slice(rhs_ker, [rhs_left_bond * rhs_mode_dim, rhs_right_bond])?;
            new_agr.matmul_inplace(rhs_ker.transpose([1, 0])?, agr)?;
            (agr_buff, _) = new_agr.gen_f_array();
            let agr_buff_len = agr_buff.len();
            let agr = NDArray::from_mut_slice(&mut agr_buff, [agr_buff_len])?;
            let norm_sq = agr.norm_n_pow_n(2);
            agr.mul_by_scalar($complex_type::from(norm_sq.powf(-0.5)));
            agr_val = agr_val + norm_sq.ln() / 2.;
          }
        };
        Ok(Complex::<$real_type>::from(agr_val) + Complex::<$real_type>::from(agr_buff[0]).ln())
      }

      /// This method returns natural logarithm of the sum of all elements of a tensor.
      /// The logarithm is necessary to make computation stable, when the value of the
      /// sum is exponentially big.
      fn log_sum(&self) -> TTResult<Complex<$real_type>> {
        let mut agr_val = $complex_zero;
        let mut agr_buff = vec![$complex_one];
        for (ker, right_bond, left_bond, mode_dim) in self.iter() {
          let agr = NDArray::from_mut_slice(&mut agr_buff, [1, left_bond])?;
          let mut new_agr_buff = unsafe { $fn_uninit_buff(right_bond) };
          let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [1, right_bond])?;
          let mut reduced_ker_buff = vec![$complex_zero; left_bond * right_bond];
          let reduced_ker = NDArray::from_mut_slice(&mut reduced_ker_buff, [left_bond, 1, right_bond])?;
          let ker_arr = NDArray::from_slice(&ker, [left_bond, mode_dim, right_bond])?;
          unsafe { ker_arr.reduce_add(reduced_ker)? };
          let reduced_ker = NDArray::from_mut_slice(&mut reduced_ker_buff, [left_bond, right_bond])?;
          unsafe { new_agr.matmul_inplace(agr, reduced_ker)? };
          agr_buff = new_agr_buff;
          let agr = NDArray::from_mut_slice(&mut agr_buff, [right_bond])?;
          let norm_sq = unsafe { agr.norm_n_pow_n(2) };
          unsafe { agr.mul_by_scalar($complex_type::from(norm_sq.powf(-0.5))) };
          agr_val = agr_val + norm_sq.ln() / 2.;
        }
        Ok(Complex::<$real_type>::from(agr_val) + Complex::<$real_type>::from(agr_buff[0]).ln())
      }

      /// This method sets a Tensor Train into the left canonical form inplace.
      /// The L2 norm of a Tensor Train after this operation is equal to 1.
      /// The natural logarithm of norm of the initial Tensor Train is returned.
      fn set_into_left_canonical(&mut self) -> TTResult<$real_type> {
        let mut new_left_bond = 1;
        let mut orth_buff = unsafe { $fn_uninit_buff(1) };
        let mut lognorm = $real_zero;
        orth_buff[0] = $complex_one;
        for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() } {
          let orth = NDArray::from_mut_slice(&mut orth_buff, [new_left_bond, *left_bond])?;
          let ker = NDArray::from_mut_slice(ker_buff, [*left_bond, *mode_dim * *right_bond])?;
          let mut new_ker_buff = unsafe { $fn_uninit_buff(new_left_bond * *mode_dim * *right_bond) };
          let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [new_left_bond, *mode_dim * *right_bond])?;
          unsafe { new_ker.matmul_inplace(orth, ker) }?;
          let nrows = new_left_bond * *mode_dim;
          let ncols = *right_bond;
          let min_dim = std::cmp::min(nrows, ncols);
          let mut aux_buff = unsafe { $fn_uninit_buff(min_dim * min_dim) };
          let aux = NDArray::from_mut_slice(&mut aux_buff, [min_dim, min_dim])?;
          let new_ker = new_ker.reshape([nrows, ncols])?;
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
          let norm: $real_type = $real_type::sqrt((&orth_buff).into_iter().map(|x| { (*x).abs().powi(2) }).sum());
          (&mut orth_buff).into_iter().for_each(|x| { *x /= norm});
          lognorm += norm.ln();
        }
        unsafe { self.get_kernels_mut() }.last_mut().unwrap().as_mut().iter_mut().for_each(|x| { *x *= orth_buff[0] });
        Ok(lognorm)
      }

      /// This method sets a Tensor Train into the right canonical form inplace.
      /// The L2 norm of a Tensor Train after this operation is equal to 1.
      /// The natural logarithm of norm of the initial Tensor Train is returned.
      fn set_into_right_canonical(&mut self) -> TTResult<$real_type> {
        let mut new_right_bond = 1;
        let mut orth_buff = unsafe { $fn_uninit_buff(1) };
        let mut lognorm = $real_zero;
        orth_buff[0] = $complex_one;
        for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() }.rev() {
          let orth = NDArray::from_mut_slice(&mut orth_buff, [*right_bond, new_right_bond])?;
          let ker = NDArray::from_mut_slice(ker_buff, [*mode_dim * *left_bond, *right_bond])?;
          let mut new_ker_buff = unsafe { $fn_uninit_buff(*mode_dim * *left_bond * new_right_bond) };
          let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [*mode_dim * *left_bond, new_right_bond])?;
          unsafe { new_ker.matmul_inplace(ker, orth) }?;
          let ncols = new_right_bond * *mode_dim;
          let nrows = *left_bond;
          let min_dim = std::cmp::min(nrows, ncols);
          let mut aux_buff = unsafe { $fn_uninit_buff(min_dim * min_dim) };
          let aux = NDArray::from_mut_slice(&mut aux_buff, [min_dim, min_dim])?;
          let new_ker = new_ker.reshape([nrows, ncols])?;
          unsafe { new_ker.rq(aux)? };
          if ncols > nrows {
            *right_bond = new_right_bond;
            new_right_bond = nrows;
            swap(&mut new_ker_buff, ker_buff);
            swap(&mut orth_buff, &mut aux_buff);
          } else {
            *right_bond = new_right_bond;
            *left_bond = ncols;
            new_right_bond = ncols;
            swap(&mut aux_buff, ker_buff);
            swap(&mut new_ker_buff, &mut orth_buff);
          }
          let norm: $real_type = $real_type::sqrt((&orth_buff).into_iter().map(|x| { (*x).abs().powi(2) }).sum());
          (&mut orth_buff).into_iter().for_each(|x| { *x /= norm});
          lognorm += norm.ln();
        }
        unsafe { self.get_kernels_mut() }.first_mut().unwrap().as_mut().iter_mut().for_each(|x| { *x *= orth_buff[0] });
        Ok(lognorm)
      }

      /// This method evaluates a natural logarithm of an element of a Tensor Train given the index.
      /// The logarithm is necessary to make computation stable, when the value of the
      /// element is exponentially big or small.
      fn log_eval_index(&self, index: &[usize]) -> TTResult<Complex<$real_type>> {
        let mut agr_buff = vec![$complex_one; 1];
        let mut agr_val = $complex_zero;
        let mut agr = NDArray::from_mut_slice(&mut agr_buff, [1, 1])?;
        if self.get_len() != index.len() {
          return Err(TTError::IncorrectIndexLength);
        }
        for (i, (ker_buff, right_bond, left_bond, mode_dim)) in index.into_iter().zip(self.iter()) {
          if *i >= mode_dim { return Err(TTError::OutOfBound); }
          let ker = NDArray::from_slice(ker_buff, [left_bond * mode_dim, right_bond])?;
          let subker = unsafe { ker.subarray([(left_bond * i)..(left_bond * (i + 1)), 0..right_bond])? };
          let mut new_agr_buff = unsafe { $fn_uninit_buff(right_bond) };
          let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [1, right_bond])?;
          unsafe { new_agr.matmul_inplace(agr, subker) }?;
          swap(&mut new_agr_buff, &mut agr_buff);
          agr = new_agr.into();
          let norm_sq = unsafe { agr.norm_n_pow_n(2) };
          unsafe { agr.mul_by_scalar($complex_type::from(norm_sq.powf(-0.5))) };
          agr_val = agr_val + norm_sq.ln() / 2.;
        }
        Ok(Complex::<$real_type>::from(agr_val) + Complex::<$real_type>::from(agr_buff[0]).ln())
      }

      /// This method truncates a left canonical form of a Tensor Train inplace, given an accuracy
      /// of a local truncation delta. The L2 norm of a Tensor Train after truncation is equal to 1.
      /// Method returns the norm of a Tensor Train as if it was not normalized by 1.
      fn truncate_left_canonical(&mut self, delta: $real_type) -> TTResult<$real_type> {
        let mut lmbd_buff = vec![$complex_one; 1];
        let mut lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, 1])?;
        let mut isom_buff = vec![$complex_one; 1];
        let mut isom = NDArray::from_mut_slice(&mut isom_buff, [1, 1])?;
        let mut trunc_dim = 1;
        let mut phase_mul = $complex_type::from(0.);
        for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() }.rev() {
          let ker = NDArray::from_mut_slice(ker_buff, [*left_bond * *mode_dim, *right_bond])?;
          let mut orth_buff = unsafe { $fn_uninit_buff(*left_bond * *mode_dim * trunc_dim) };
          let mut orth = NDArray::from_mut_slice(&mut orth_buff, [*left_bond * *mode_dim, trunc_dim])?;
          unsafe {
            orth.matmul_inplace(ker, isom)?;
            orth.mul_inpl(lmbd)?;
          };
          orth = orth.reshape([*left_bond, *mode_dim * trunc_dim])?;
          let min_dim = std::cmp::min(*left_bond, *mode_dim * trunc_dim);
          let mut u_buff = unsafe { $fn_uninit_buff(*left_bond * min_dim) };
          let u = NDArray::from_mut_slice(&mut u_buff, [*left_bond, min_dim])?;
          let mut v_dag_buff = unsafe { $fn_uninit_buff(min_dim * trunc_dim * *mode_dim) };
          let v_dag = NDArray::from_mut_slice(&mut v_dag_buff, [min_dim, trunc_dim * *mode_dim])?;
          let mut new_lmbd_buff = unsafe { $fn_uninit_lmbd(min_dim) };
          let new_lmbd = NDArray::from_mut_slice(&mut new_lmbd_buff, [min_dim])?;
          unsafe { orth.svd(u, new_lmbd, v_dag) }?;
          let new_trunc_dim = get_trunc_dim(&new_lmbd_buff, delta);
          lmbd_buff = new_lmbd_buff.into_iter().take(new_trunc_dim).map(|x| $complex_type::from(x)).collect();
          lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, new_trunc_dim])?;
          (isom_buff, _) = unsafe { u.subarray([0..(*left_bond), 0..new_trunc_dim])?.gen_f_array() };
          isom = NDArray::from_mut_slice(&mut isom_buff, [*left_bond, new_trunc_dim])?;
          let (mut new_ker_buff, _) = unsafe { v_dag.subarray([0..new_trunc_dim, 0..(*mode_dim * trunc_dim)])?.gen_f_array() };
          swap(ker_buff, &mut new_ker_buff);
          *left_bond = new_trunc_dim;
          *right_bond = trunc_dim;
          trunc_dim = new_trunc_dim;
          phase_mul = u_buff[0]
        }
        unsafe { self.get_kernels_mut() }.first_mut().unwrap().as_mut().iter_mut().for_each(|x| *x *= phase_mul);
        Ok((lmbd_buff[0].abs()))
      }

      /// This method truncates a right canonical form of a Tensor Train inplace, given an accuracy
      /// of a local truncation delta. The L2 norm of a Tensor Train after truncation is equal to 1.
      /// Method returns the norm of a Tensor Train as if it was not normalized by 1.
      fn truncate_right_canonical(&mut self, delta: $real_type) -> TTResult<$real_type> {
        let mut lmbd_buff = vec![$complex_one; 1];
        let mut lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, 1])?;
        let mut isom_buff = vec![$complex_one; 1];
        let mut isom = NDArray::from_mut_slice(&mut isom_buff, [1, 1])?;
        let mut trunc_dim = 1;
        let mut phase_mul = $complex_type::from(0.);
        for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() } {
          let ker = NDArray::from_mut_slice(ker_buff, [*left_bond, *mode_dim * *right_bond])?;
          let mut orth_buff = unsafe { $fn_uninit_buff(trunc_dim * *mode_dim * *right_bond) };
          let mut orth = NDArray::from_mut_slice(&mut orth_buff, [trunc_dim, *mode_dim * *right_bond])?;
          unsafe {
            orth.matmul_inplace(isom, ker)?;
            orth.mul_inpl(lmbd)?;
          };
          orth = orth.reshape([*mode_dim * trunc_dim, *right_bond])?;
          let min_dim = std::cmp::min(*mode_dim * trunc_dim, *right_bond);
          let mut u_buff = unsafe { $fn_uninit_buff(trunc_dim * min_dim * *mode_dim) };
          let u = NDArray::from_mut_slice(&mut u_buff, [trunc_dim * *mode_dim, min_dim])?;
          let mut v_dag_buff = unsafe { $fn_uninit_buff(*right_bond * min_dim) };
          let v_dag = NDArray::from_mut_slice(&mut v_dag_buff, [min_dim, *right_bond])?;
          let mut new_lmbd_buff = unsafe { $fn_uninit_lmbd(min_dim) };
          let new_lmbd = NDArray::from_mut_slice(&mut new_lmbd_buff, [min_dim])?;
          unsafe { orth.svd(u, new_lmbd, v_dag) }?;
          let new_trunc_dim = get_trunc_dim(&new_lmbd_buff, delta);
          lmbd_buff = new_lmbd_buff.into_iter().take(new_trunc_dim).map(|x| $complex_type::from(x)).collect();
          lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [new_trunc_dim, 1])?;
          (isom_buff, _) = unsafe { v_dag.subarray([0..new_trunc_dim, 0..(*right_bond)])?.gen_f_array() };
          isom = NDArray::from_mut_slice(&mut isom_buff, [new_trunc_dim, *right_bond])?;
          let (mut new_ker_buff, _) = unsafe { u.subarray([0..(*mode_dim * trunc_dim), 0..new_trunc_dim])?.gen_f_array() };
          swap(ker_buff, &mut new_ker_buff);
          *right_bond = new_trunc_dim;
          *left_bond = trunc_dim;
          trunc_dim = new_trunc_dim;
          phase_mul = v_dag_buff[0];
        }
        unsafe { self.get_kernels_mut() }.last_mut().unwrap().as_mut().iter_mut().for_each(|x| *x *= phase_mul);
        Ok((lmbd_buff[0].abs()))
      }

      /// This method multiply a given Tensor Train by another one element-wisely. 
      fn elementwise_prod(&mut self, other: &Self) -> TTResult<()> {
        if self.get_len() != other.get_len() { return Err(TTError::LengthsMismatch) }
        let iter = unsafe { self.iter_mut() }.zip(other.iter());
        for (
            (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
            (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
        ) in iter {
          if *lhs_mode_dim != rhs_mode_dim { return Err(TTError::IncorrectLocalDim) }
          let new_lhs_left_bond = *lhs_left_bond * rhs_left_bond;
          let new_lhs_right_bond = *lhs_right_bond * rhs_right_bond;
          let mut new_lhs_ker_buff = unsafe { $fn_uninit_buff(new_lhs_left_bond * new_lhs_right_bond * *lhs_mode_dim) };
          let new_lhs_ker = NDArray::from_mut_slice(
            &mut new_lhs_ker_buff,
            [*lhs_left_bond, rhs_left_bond, *lhs_mode_dim, *lhs_right_bond, rhs_right_bond]
          )?;
          let lhs_arr = NDArray::from_slice(&lhs_ker,
            [*lhs_left_bond, 1, *lhs_mode_dim, *lhs_right_bond, 1]
          )?;
          let rhs_arr = NDArray::from_slice(&rhs_ker,
            [1, rhs_left_bond, *lhs_mode_dim, 1, rhs_right_bond]
          )?;
          unsafe { new_lhs_ker.mul(lhs_arr, rhs_arr)? };
          *lhs_ker = new_lhs_ker_buff;
          *lhs_left_bond = new_lhs_left_bond;
          *lhs_right_bond = new_lhs_right_bond;
        }
        Ok(())
      }
    }
  }
}

tt_impl!(TTf32, f32,       f32, random_normal_f32, uninit_buff_f32, uninit_buff_f32, 0.,                     1.,                     0., 1.);
tt_impl!(TTf64, f64,       f64, random_normal_f64, uninit_buff_f64, uninit_buff_f64, 0.,                     1.,                     0., 1.);
tt_impl!(TTc32, Complex32, f32, random_normal_c32, uninit_buff_c32, uninit_buff_f32, Complex32::new(0., 0.), Complex32::new(1., 0.), 0., 1.);
tt_impl!(TTc64, Complex64, f64, random_normal_c64, uninit_buff_c64, uninit_buff_f64, Complex64::new(0., 0.), Complex64::new(1., 0.), 0., 1.);
