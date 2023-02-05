use std::mem::swap;
use std::iter::DoubleEndedIterator;

use num_traits::NumCast;

/*use rayon::prelude::IntoParallelIterator;
use rayon::iter::ParallelIterator;*/

use linwrap::{
  NDArray,
  NDArrayError,
  init_utils::BufferGenerator,
  LinalgComplex,
  LinalgReal,
};

use crate::utils::{
  get_trunc_dim,
  argsort,
  indices_prod,
};

// ---------------------------------------------------------------------------------- //

#[derive(Debug)]
pub enum TTError {

  /// Appears when error happens at the level of NDArray
  NDArrayError(NDArrayError),

  /// Appears when two Tensor Trains have different length
  /// while it is required them to be equal.
  LengthsMismatch,

  /// Appears when dims of all modes of two Tensor Trains do not match each other.
  ModesDimsMismatch,

  /// Appears when one sends an index of incorrect length to the
  /// eval method.
  IncorrectIndexLength,

  /// Appears when one sends an index that lies out of bound
  /// of a Tensor Train to the eval method.
  OutOfBound,

  /// Appears when local a mode dimensions of two Tensor Trains do
  /// not match each other, while it is required them to be.
  IncorrectLocalDim,

  /// Appears when one calls TTCross update with empty updates, while
  /// some updates are required.
  EmptyUpdate,
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

/// Normalize a given array by one. Returns the logarithm of the norm.
#[inline]
unsafe fn normalize<T, const N: usize>(arr: NDArray<*mut T, N>) -> T
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  let norm_sq = arr.norm_n_pow_n(2);
  arr.mul_by_scalar(norm_sq.powf(<T::Real as NumCast>::from(-0.5).unwrap()));
  assert!(norm_sq.abs() > <T::Real as NumCast>::from(0.).unwrap(), "ololo {:?}", norm_sq);
  norm_sq.ln() / T::from(2.).unwrap()
}

// ---------------------------------------------------------------------------------- //

pub trait TensorTrain<T>: Clone
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  type Buff: AsMut<[T]> + AsRef<[T]>;
  type Kers: AsMut<[Self::Buff]> + AsRef<[Self::Buff]>;

  /// Returns a random Tensor Train given the modes dimensions.
  /// 
  /// # Arguments
  /// 
  /// * 'mode_dims' - a vector of dimensions of each mode.
  /// * 'max_rank' - maximal TT rank.
  fn new_random(
    mode_dims: Vec<usize>,
    max_rank:  usize,
  ) -> Self;

  /// Returns a slice with kernels.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  fn get_kernels(&self) -> &[Self::Buff];

  /// Returns a slice with left bond dimensions.
  ///
  /// # Note
  ///
  /// This method is mostly for development needs.
  fn get_left_bonds(&self) -> &[usize];

  /// Returns a slice with right bond dimensions.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  fn get_right_bonds(&self) -> &[usize];

  /// Returns a slice with modes dimensions.
  fn get_mode_dims(&self) -> &[usize];
  
  /// Returns number of modes.
  fn get_len(&self) -> usize;

  /// Returns an iterator over components of a Tensor Train.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  fn iter<'a>(&'a self) -> TTIter<'a, T>;

  /// Returns a mutable slice with kernels.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  /// 
  /// #Safety
  /// 
  /// Sizes of buffers and corresponding modes and bonds dimensions
  /// must be consistent with each other.
  unsafe fn get_kernels_mut(&mut self) -> &mut [Self::Buff];

  /// Returns a mutable slice with left bond dimensions.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  /// 
  /// #Safety
  /// 
  /// Sizes of buffers and corresponding modes and bonds dimensions
  /// must be consistent with each other.
  unsafe fn get_left_bonds_mut(&mut self) -> &mut [usize];

  /// Returns a mutable slice with right bond dimensions.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  /// 
  /// #Safety
  /// 
  /// Sizes of buffers and corresponding modes and bonds dimensions
  /// must be consistent with each other.
  unsafe fn get_right_bonds_mut(&mut self) -> &mut [usize];

  /// Returns a mutable iterator over components of a Tensor Train.
  /// 
  /// # Note
  /// 
  /// This method is mostly for development needs.
  /// 
  /// #Safety
  /// 
  /// Sizes of buffers and corresponding modes and bonds dimensions
  /// must be consistent with each other.
  unsafe fn iter_mut<'a>(&'a mut self) -> TTIterMut<'a, T>;

  /// Returns internal bond dimensions of a Tensor Train.
  fn get_bonds(&self) -> &[usize] {
    &self.get_left_bonds()[1..]
  }

  /// Complex conjugates elements of a Tensor Train inplace.
  fn conj(&mut self) {
    for ker in unsafe { self.get_kernels_mut() } {
      let len = ker.as_ref().len();
      let ker_arr = NDArray::from_mut_slice(ker.as_mut(), [len]).unwrap();
      unsafe { ker_arr.conj() };
    }
  }

  /// Computes a dot product of two Tensor Trains.
  /// Returns the result as a tuple of two values: the first value v1 is the logarithm
  /// of the dot product modulo, the second value v2 is the phase multiplier that can be written
  /// as v2 = exp(i * phi). The dot product can be reconstructed from these two values as
  /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
  /// the dot product can be exponentially large in some cases.
  /// If modes dimensions of Tensor Trains do not mach each other, it
  /// returns an error.
  /// 
  /// # Arguments
  /// 
  /// * 'other' - other Tensor Train that must have the same modes dimensions.
  fn log_dot(&self, other: &Self) -> TTResult<(T, T)> {
    if self.get_mode_dims() != other.get_mode_dims() { return Err(TTError::ModesDimsMismatch) }
    let mut agr_buff = vec![T::one()];
    let mut agr_val = T::zero();
    let iter = self.iter().zip(other.iter());
    for (
        (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
        (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
    ) in iter {
      unsafe {
        let agr = NDArray::from_mut_slice(&mut agr_buff, [rhs_left_bond, lhs_left_bond])?;
        let mut new_agr_buff = T::uninit_buff(rhs_left_bond * lhs_mode_dim * lhs_right_bond);
        let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [rhs_left_bond, lhs_mode_dim * lhs_right_bond])?;
        let lhs_ker = NDArray::from_slice(lhs_ker, [lhs_left_bond, lhs_mode_dim * lhs_right_bond])?;
        new_agr.matmul_inplace(agr, lhs_ker)?;
        (agr_buff, _) = new_agr.gen_f_array();
        let mut new_agr_buff = T::uninit_buff(rhs_right_bond * lhs_right_bond);
        let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [rhs_right_bond, lhs_right_bond])?;
        let agr = NDArray::from_mut_slice(&mut agr_buff, [rhs_left_bond * lhs_mode_dim, lhs_right_bond])?;
        let rhs_ker = NDArray::from_slice(rhs_ker, [rhs_left_bond * rhs_mode_dim, rhs_right_bond])?;
        new_agr.matmul_inplace(rhs_ker.transpose([1, 0])?, agr)?;
        (agr_buff, _) = new_agr.gen_f_array();
        let agr_buff_len = agr_buff.len();
        let agr = NDArray::from_mut_slice(&mut agr_buff, [agr_buff_len])?;
        let log_norm = normalize(agr);
        agr_val = agr_val + log_norm;
      }
    };
    Ok((agr_val, agr_buff[0]))
  }

  /// Computes the sum of all elements of a tensor represented by a Tensor Train.
  /// Returns the result as a tuple of two values: the first value v1 is the logarithm
  /// of the sum, the second value v2 is the phase multiplier that can be written
  /// as v2 = exp(i * phi). The resulting sum can be reconstructed from these two values as
  /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
  /// the value of sum can be exponentially large in some cases.
  fn log_sum(&self) -> TTResult<(T, T)> {
    let mut agr_val = T::zero();
    let mut agr_buff = vec![T::one()];
    for (ker, right_bond, left_bond, mode_dim) in self.iter() {
      let agr = NDArray::from_mut_slice(&mut agr_buff, [1, left_bond])?;
      let mut new_agr_buff = unsafe { T::uninit_buff(right_bond) };
      let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [1, right_bond])?;
      let mut reduced_ker_buff = vec![T::zero(); left_bond * right_bond];
      let reduced_ker = NDArray::from_mut_slice(&mut reduced_ker_buff, [left_bond, 1, right_bond])?;
      let ker_arr = NDArray::from_slice(&ker, [left_bond, mode_dim, right_bond])?;
      unsafe { ker_arr.reduce_add(reduced_ker)? };
      let reduced_ker = NDArray::from_mut_slice(&mut reduced_ker_buff, [left_bond, right_bond])?;
      unsafe { new_agr.matmul_inplace(agr, reduced_ker)? };
      agr_buff = new_agr_buff;
      let agr = NDArray::from_mut_slice(&mut agr_buff, [right_bond])?;
      let log_norm = unsafe { normalize(agr) };
      agr_val = agr_val + log_norm;
    }
    Ok((agr_val, agr_buff[0]))
  }

  /// Sets a Tensor Train into the left canonical form inplace.
  /// The L2 norm of a Tensor Train after this operation is equal to 1.
  /// Returns logarithm of the L2 norm of the tensor before canonicalization.
  /// Logarithm is necessary to guaranty stability of the computation when the
  /// norm of the tensor exponentially large.
  fn set_into_left_canonical(&mut self) -> TTResult<T> {
    let mut new_left_bond = 1;
    let mut orth_buff = unsafe { T::uninit_buff(1) };
    let mut lognorm = T::zero();
    orth_buff[0] = T::one();
    for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() } {
      let orth = NDArray::from_mut_slice(&mut orth_buff, [new_left_bond, *left_bond])?;
      let ker = NDArray::from_mut_slice(ker_buff, [*left_bond, *mode_dim * *right_bond])?;
      let mut new_ker_buff = unsafe { T::uninit_buff(new_left_bond * *mode_dim * *right_bond) };
      let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [new_left_bond, *mode_dim * *right_bond])?;
      unsafe { new_ker.matmul_inplace(orth, ker) }?;
      let nrows = new_left_bond * *mode_dim;
      let ncols = *right_bond;
      let min_dim = std::cmp::min(nrows, ncols);
      let mut aux_buff = unsafe { T::uninit_buff(min_dim * min_dim) };
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
      let len = orth_buff.len();
      let orth_buff_arr = NDArray::from_mut_slice(&mut orth_buff, [len])?;
      lognorm = unsafe { lognorm + normalize(orth_buff_arr) };
    }
    unsafe { self.get_kernels_mut() }.last_mut().unwrap().as_mut().iter_mut().for_each(|x| { *x = *x * orth_buff[0] });
    Ok(lognorm)
  }

  /// Sets a Tensor Train into the right canonical form inplace.
  /// The L2 norm of a Tensor Train after this operation is equal to 1.
  /// Returns logarithm of the L2 norm of the tensor before canonicalization.
  /// Logarithm is necessary to guaranty stability of the computation when the
  /// norm of the tensor exponentially large.
  fn set_into_right_canonical(&mut self) -> TTResult<T> {
    let mut new_right_bond = 1;
    let mut orth_buff = unsafe { T::uninit_buff(1) };
    let mut lognorm = T::zero();
    orth_buff[0] = T::one();
    for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() }.rev() {
      let orth = NDArray::from_mut_slice(&mut orth_buff, [*right_bond, new_right_bond])?;
      let ker = NDArray::from_mut_slice(ker_buff, [*mode_dim * *left_bond, *right_bond])?;
      let mut new_ker_buff = unsafe { T::uninit_buff(*mode_dim * *left_bond * new_right_bond) };
      let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [*mode_dim * *left_bond, new_right_bond])?;
      unsafe { new_ker.matmul_inplace(ker, orth) }?;
      let ncols = new_right_bond * *mode_dim;
      let nrows = *left_bond;
      let min_dim = std::cmp::min(nrows, ncols);
      let mut aux_buff = unsafe { T::uninit_buff(min_dim * min_dim) };
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
      let len = orth_buff.len();
      let orth_buff_arr = NDArray::from_mut_slice(&mut orth_buff, [len])?;
      lognorm = lognorm + unsafe { normalize(orth_buff_arr) };
    }
    unsafe { self.get_kernels_mut() }.first_mut().unwrap().as_mut().iter_mut().for_each(|x| { *x = *x * orth_buff[0] });
    Ok(lognorm)
  }

  /// Computes an element of a Tensor Train given the index.
  /// Returns the result as a tuple of two values: the first value v1 is the logarithm
  /// of the element modulo, the second value v2 is the phase multiplier that can be written
  /// as v2 = exp(i * phi). The value of an element can be reconstructed from these two values as
  /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
  /// the value of an element can be exponentially large in some cases.
  /// 
  /// # Arguments
  /// 
  /// * 'index' - index for which one evaluates a value.
  fn log_eval_index(&self, index: &[usize]) -> TTResult<(T, T)> {
    let mut agr_buff = vec![T::one(); 1];
    let mut agr_val = T::zero();
    let mut agr = NDArray::from_mut_slice(&mut agr_buff, [1, 1])?;
    if self.get_len() != index.len() {
      return Err(TTError::IncorrectIndexLength);
    }
    for (i, (ker_buff, right_bond, left_bond, mode_dim)) in index.into_iter().zip(self.iter()) {
      if *i >= mode_dim { return Err(TTError::OutOfBound); }
      let ker = NDArray::from_slice(ker_buff, [left_bond * mode_dim, right_bond])?;
      let subker = unsafe { ker.subarray([(left_bond * i)..(left_bond * (i + 1)), 0..right_bond])? };
      let mut new_agr_buff = unsafe { T::uninit_buff(right_bond) };
      let new_agr = NDArray::from_mut_slice(&mut new_agr_buff, [1, right_bond])?;
      unsafe { new_agr.matmul_inplace(agr, subker) }?;
      swap(&mut new_agr_buff, &mut agr_buff);
      agr = new_agr.into();
      let log_norm = unsafe { normalize(agr) };
      agr_val = agr_val + log_norm;
    }
    Ok((agr_val, agr_buff[0]))
  }

  /// Truncates the left canonical form of a Tensor Train inplace
  /// and normalizes it by 1. Returns L2 norm of a Tensor Train
  /// after truncation but before normalization
  /// (Could be considered as the truncation error).
  /// 
  /// # Arguments
  /// 
  /// * 'delta' - local SVD based truncation accuracy.
  /// 
  /// # Note
  /// 
  /// This method must be run only on the left canonical form of a Tensor Train.
  /// Otherwise, the result is meaningless.
  fn truncate_left_canonical(&mut self, delta: T::Real) -> TTResult<T::Real> {
    let mut lmbd_buff = vec![T::one(); 1];
    let mut lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, 1])?;
    let mut isom_buff = vec![T::one(); 1];
    let mut isom = NDArray::from_mut_slice(&mut isom_buff, [1, 1])?;
    let mut trunc_dim = 1;
    let mut phase_mul = T::zero();
    for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() }.rev() {
      let ker = NDArray::from_mut_slice(ker_buff, [*left_bond * *mode_dim, *right_bond])?;
      let mut orth_buff = unsafe { T::uninit_buff(*left_bond * *mode_dim * trunc_dim) };
      let mut orth = NDArray::from_mut_slice(&mut orth_buff, [*left_bond * *mode_dim, trunc_dim])?;
      unsafe {
        orth.matmul_inplace(ker, isom)?;
        orth.mul_inpl(lmbd)?;
      };
      orth = orth.reshape([*left_bond, *mode_dim * trunc_dim])?;
      let min_dim = std::cmp::min(*left_bond, *mode_dim * trunc_dim);
      let mut u_buff = unsafe { T::uninit_buff(*left_bond * min_dim) };
      let u = NDArray::from_mut_slice(&mut u_buff, [*left_bond, min_dim])?;
      let mut v_dag_buff = unsafe { T::uninit_buff(min_dim * trunc_dim * *mode_dim) };
      let v_dag = NDArray::from_mut_slice(&mut v_dag_buff, [min_dim, trunc_dim * *mode_dim])?;
      let mut new_lmbd_buff = unsafe { T::Real::uninit_buff(min_dim) };
      let new_lmbd = NDArray::from_mut_slice(&mut new_lmbd_buff, [min_dim])?;
      unsafe { orth.svd(u, new_lmbd, v_dag) }?;
      let new_trunc_dim = get_trunc_dim(&new_lmbd_buff, delta);
      lmbd_buff = new_lmbd_buff.into_iter().take(new_trunc_dim).map(|x| T::from(x).unwrap()).collect();
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
    unsafe { self.get_kernels_mut() }.first_mut().unwrap().as_mut().iter_mut().for_each(|x| *x = *x * phase_mul);
    Ok(lmbd_buff[0].abs())
  }

  /// Truncates the right canonical form of a Tensor Train inplace
  /// and normalizes it by 1. Returns L2 norm of a Tensor Train
  /// after truncation but before normalization
  /// (Could be considered as the truncation error).
  /// 
  /// # Arguments
  /// 
  /// * 'delta' - local SVD based truncation accuracy.
  /// 
  /// # Note
  /// 
  /// This method must be run only on the right canonical form of a Tensor Train.
  /// Otherwise, the result is meaningless.
  fn truncate_right_canonical(&mut self, delta: T::Real) -> TTResult<T::Real> {
    let mut lmbd_buff = vec![T::one(); 1];
    let mut lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, 1])?;
    let mut isom_buff = vec![T::one(); 1];
    let mut isom = NDArray::from_mut_slice(&mut isom_buff, [1, 1])?;
    let mut trunc_dim = 1;
    let mut phase_mul = T::zero();
    for (ker_buff, right_bond, left_bond, mode_dim) in unsafe { self.iter_mut() } {
      let ker = NDArray::from_mut_slice(ker_buff, [*left_bond, *mode_dim * *right_bond])?;
      let mut orth_buff = unsafe { T::uninit_buff(trunc_dim * *mode_dim * *right_bond) };
      let mut orth = NDArray::from_mut_slice(&mut orth_buff, [trunc_dim, *mode_dim * *right_bond])?;
      unsafe {
        orth.matmul_inplace(isom, ker)?;
        orth.mul_inpl(lmbd)?;
      };
      orth = orth.reshape([*mode_dim * trunc_dim, *right_bond])?;
      let min_dim = std::cmp::min(*mode_dim * trunc_dim, *right_bond);
      let mut u_buff = unsafe { T::uninit_buff(trunc_dim * min_dim * *mode_dim) };
      let u = NDArray::from_mut_slice(&mut u_buff, [trunc_dim * *mode_dim, min_dim])?;
      let mut v_dag_buff = unsafe { T::uninit_buff(*right_bond * min_dim) };
      let v_dag = NDArray::from_mut_slice(&mut v_dag_buff, [min_dim, *right_bond])?;
      let mut new_lmbd_buff = unsafe { T::Real::uninit_buff(min_dim) };
      let new_lmbd = NDArray::from_mut_slice(&mut new_lmbd_buff, [min_dim])?;
      unsafe { orth.svd(u, new_lmbd, v_dag) }?;
      let new_trunc_dim = get_trunc_dim(&new_lmbd_buff, delta);
      lmbd_buff = new_lmbd_buff.into_iter().take(new_trunc_dim).map(|x| T::from(x).unwrap()).collect();
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
    unsafe { self.get_kernels_mut() }.last_mut().unwrap().as_mut().iter_mut().for_each(|x| *x = *x * phase_mul);
    Ok(lmbd_buff[0].abs())
  }

  /// Multiplies a given Tensor Train by another one element-wisely.
  /// If modes dimensions of Tensor Trains do not match each other, returns
  /// an error.
  /// 
  /// # Arguments 
  /// 
  /// 'other' - other Tensor Train with the same modes dimensions.
  fn elementwise_prod(&mut self, other: &Self) -> TTResult<()> 
  {
    if self.get_len() != other.get_len() { return Err(TTError::LengthsMismatch) }
    let iter = unsafe { self.iter_mut() }.zip(other.iter());
    for (
        (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
        (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
    ) in iter {
      if *lhs_mode_dim != rhs_mode_dim { return Err(TTError::IncorrectLocalDim) }
      let new_lhs_left_bond = *lhs_left_bond * rhs_left_bond;
      let new_lhs_right_bond = *lhs_right_bond * rhs_right_bond;
      let mut new_lhs_ker_buff = unsafe { T::uninit_buff(new_lhs_left_bond * new_lhs_right_bond * *lhs_mode_dim) };
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

  /// TODO: make it more stable?
  /// Multiplies a Tensor Train by a scalar inplace.
  fn mul_by_scalar(&mut self, scalar: T)
  {
    unsafe { self.get_kernels_mut()[0].as_mut().iter_mut() }.for_each(|x| {
      *x = *x * scalar;
    });
  }

  /// Returns maximal bond dimension.
  fn get_tt_rank(&self) -> usize {
    *self.get_bonds().into_iter().max().unwrap()
  }

  /// Adds an other Tensor Train to a given one element-wisely.
  /// 
  /// # Note
  /// 
  /// If modes dimensions of Tensor Trains do not match each other, returns
  /// an error.
  fn elementwise_sum(&mut self, other: &Self) -> TTResult<()>
  {
    if self.get_len() != other.get_len() { return Err(TTError::LengthsMismatch) }
    let len = self.get_len();
    let iter = unsafe { self.iter_mut() }.zip(other.iter()).enumerate();
    for (
      i, 
      ( 
        (lhs_ker, lhs_right_bond, lhs_left_bond, lhs_mode_dim),
        (rhs_ker, rhs_right_bond, rhs_left_bond, rhs_mode_dim),
      )
    ) in iter {
      if *lhs_mode_dim != rhs_mode_dim { return Err(TTError::IncorrectLocalDim) }
      let mode_dim = rhs_mode_dim;
      if i == 0 {
        let lhs_ker_arr = NDArray::from_slice(lhs_ker, [*lhs_left_bond, mode_dim, *lhs_right_bond])?;
        let rhs_ker_arr = NDArray::from_slice(rhs_ker, [rhs_left_bond, mode_dim, rhs_right_bond])?;
        let new_right_bond = *lhs_right_bond + rhs_right_bond;
        let new_size = *lhs_left_bond * new_right_bond * mode_dim;
        let mut new_ker_buff = unsafe { T::uninit_buff(new_size) };
        let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [*lhs_left_bond, mode_dim, new_right_bond])?;
        let (k0, k1) = unsafe { new_ker.split_across_axis(2, *lhs_right_bond)? };
        unsafe { k0.into_f_iter().zip(lhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        unsafe { k1.into_f_iter().zip(rhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        *lhs_ker = new_ker_buff;
        *lhs_right_bond = new_right_bond;
      } else if i == len - 1 {
        let lhs_ker_arr = NDArray::from_slice(lhs_ker, [*lhs_left_bond, mode_dim, *lhs_right_bond])?;
        let rhs_ker_arr = NDArray::from_slice(rhs_ker, [rhs_left_bond, mode_dim, rhs_right_bond])?;
        let new_left_bond = *lhs_left_bond + rhs_left_bond;
        let new_size = new_left_bond * *lhs_right_bond * mode_dim;
        let mut new_ker_buff = unsafe { T::uninit_buff(new_size) };
        let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [new_left_bond, mode_dim, *lhs_right_bond])?;
        let (k0, k1) = unsafe { new_ker.split_across_axis(0, *lhs_left_bond)? };
        unsafe { k0.into_f_iter().zip(lhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        unsafe { k1.into_f_iter().zip(rhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        *lhs_ker = new_ker_buff;
        *lhs_left_bond = new_left_bond;
      } else {
        let lhs_ker_arr = NDArray::from_slice(lhs_ker, [*lhs_left_bond, mode_dim, *lhs_right_bond])?;
        let rhs_ker_arr = NDArray::from_slice(rhs_ker, [rhs_left_bond, mode_dim, rhs_right_bond])?;
        let new_left_bond = *lhs_left_bond + rhs_left_bond;
        let new_right_bond = *lhs_right_bond + rhs_right_bond;
        let new_size = new_left_bond * new_right_bond * mode_dim;
        let mut new_ker_buff = unsafe { T::uninit_buff(new_size) };
        let new_ker = NDArray::from_mut_slice(&mut new_ker_buff, [new_left_bond, mode_dim, new_right_bond])?;
        let (k0, k1) = unsafe { new_ker.split_across_axis(0, *lhs_left_bond)? };
        let (k00, k01) = unsafe { k0.split_across_axis(2, *lhs_right_bond)? };
        let (k10, k11) = unsafe { k1.split_across_axis(2, *lhs_right_bond)? };
        unsafe { k00.into_f_iter().zip(lhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        unsafe { k11.into_f_iter().zip(rhs_ker_arr.into_f_iter()).for_each(|(dst, src)| *dst.0 = *src.0) };
        unsafe { k10.into_f_iter().for_each(|dst| *dst.0 = T::zero() ) };
        unsafe { k01.into_f_iter().for_each(|dst| *dst.0 = T::zero() ) };
        *lhs_ker = new_ker_buff;
        *lhs_left_bond = new_left_bond;
        *lhs_right_bond = new_right_bond;
      }
    }
    Ok(())
  }

  /// This method is the combination of the optimization methods
  /// (1) <https://arxiv.org/abs/2101.03377> and (2) <https://arxiv.org/abs/2209.14808>
  /// The method (1) is essentially a power iteration method. It is being run first.
  /// It takes at most power_iterations_max_num steps or being terminated earlier if the
  /// max_rank of the power of a Tensor Train is reached. Then one runs (2) method
  /// on the resulting power of a Tensor Train, k is the hyper parameter (for more
  /// details see (2)), typically it is set to be equal ~ 10.
  /// 
  /// # Arguments
  /// 
  /// * 'delta' - accuracy of local truncations used in the method.
  /// * 'power_iterations_max_num' - maximum number of power iterations.
  /// * 'max_rank' - maximal TT rank allowed during the method execution.
  /// * 'k' - hyperparameter of the method from (2) <https://arxiv.org/abs/2209.14808>.
  fn argmax_modulo(
    &self,
    delta: T::Real,
    power_iterations_max_num: usize,
    max_rank: usize,
    k: usize,
  ) -> TTResult<Vec<usize>>
  {
    let len = self.get_len();
    let mut prob = self.clone();
    prob.conj();
    prob.elementwise_prod(self)?;
    prob.set_into_left_canonical()?;
    prob.truncate_left_canonical(delta)?;
    for i in 0..power_iterations_max_num {
      if prob.get_tt_rank() > max_rank {
        println!("Power iteration reached the critical TT rank 
        (critical TT rank: {}, reached TT rank: {}) at iteration {},
        switching to the optima_tt_max.", max_rank, prob.get_tt_rank(), i);
        break;
      }
      prob.elementwise_prod(self)?;
      prob.set_into_left_canonical()?;
      prob.truncate_left_canonical(delta)?;
      prob.conj();
      prob.elementwise_prod(self)?;
      prob.set_into_left_canonical()?;
      prob.truncate_left_canonical(delta)?;
    }
    let mut plug_buff = vec![T::one()];
    let mut right_parts: Vec<Vec<T>> = vec![Vec::new(); len];
    for (i, (ker, right_bond, left_bond, mode_dim)) in prob.iter().rev().enumerate() {
      let plug_arr = NDArray::from_mut_slice(&mut plug_buff, [right_bond, 1])?;
      let ker_arr = NDArray::from_slice(ker, [left_bond * mode_dim, right_bond])?;
      let mut new_right_part_buff = unsafe { T::uninit_buff(left_bond * mode_dim) };
      let new_right_part_arr = NDArray::from_mut_slice(&mut new_right_part_buff, [left_bond * mode_dim, 1])?;
      unsafe { new_right_part_arr.matmul_inplace(ker_arr, plug_arr) }?;
      let mut new_plug_buff = vec![T::zero(); left_bond];
      let new_plug_arr = NDArray::from_mut_slice(&mut new_plug_buff, [left_bond, 1])?;
      let new_right_part_arr = NDArray::from_mut_slice(&mut new_right_part_buff, [left_bond, mode_dim])?;
      unsafe { new_right_part_arr.reduce_add(new_plug_arr)? };
      let norm_sq = unsafe { new_plug_arr.norm_n_pow_n(2) };
      unsafe { new_plug_arr.mul_by_scalar(T::one() / norm_sq.sqrt()) };
      right_parts[len - 1 - i] = new_right_part_buff;
      plug_buff = new_plug_buff;
    }
    let mut values_buff = Vec::new();
    let mut args: Vec<Vec<usize>> = vec![];
    let mut plug_buff = vec![T::one()];
    let mut samples_num = 1;
    for (right_part, (ker, right_bond, left_bond, mode_dim)) in right_parts.into_iter().zip(prob.iter()) {
      let right_part_arr = NDArray::from_slice(&right_part, [left_bond, mode_dim])?;
      let plug_arr = NDArray::from_slice(&plug_buff, [samples_num, left_bond])?;
      values_buff = unsafe { T::uninit_buff(samples_num * mode_dim) };
      let values_arr = NDArray::from_mut_slice(&mut values_buff, [samples_num, mode_dim])?;
      unsafe { values_arr.matmul_inplace(plug_arr, right_part_arr)? };
      let local_args: Vec<Vec<usize>> = (0..mode_dim).map(|x| vec![x]).collect();
      args = indices_prod(&args, &local_args);
      let indices = argsort(&values_buff);
      let mut plugker_buff = unsafe { T::uninit_buff(samples_num * mode_dim * right_bond) };
      let plugker_arr = NDArray::from_mut_slice(&mut plugker_buff, [samples_num, mode_dim * right_bond])?;
      let ker_arr = NDArray::from_slice(&ker, [left_bond, mode_dim * right_bond])?;
      unsafe { plugker_arr.matmul_inplace(plug_arr, ker_arr)? };
      let plugker_arr = NDArray::from_mut_slice(&mut plugker_buff, [samples_num * mode_dim, right_bond])?;
      samples_num = std::cmp::min(k, indices.len());
      (plug_buff, _) = unsafe { plugker_arr.gen_f_array_from_axis_order(&indices[..samples_num], 0) };
      let new_args: Vec<Vec<usize>> = indices.iter().take(k).map(|i| unsafe { args.get_unchecked(*i).clone() }).collect();
      args = new_args;
      let new_values = indices.iter().take(k).map(|i| unsafe { *values_buff.get_unchecked(*i) }).collect();
      values_buff = new_values;
    }
    let (argmax, _) = values_buff.into_iter().enumerate().max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap();
    Ok(args[argmax].clone())
  }
}
