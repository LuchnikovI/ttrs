use serde::{Serialize, Deserialize};
use rayon::iter::IntoParallelIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

use linwrap::{
  NDArray,
  LinalgComplex,
  LinalgReal,
};

use crate::TensorTrain;
use crate::mutli_indices::{
  MultiIndicesSet,
  MultiIndices,
  indices_par_iter,
};
use crate::tt_traits::{
  TTResult,
  TTError,
};
use crate::utils::get_restrictions;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
enum DMRGState {
  ToLeft,
  ToRight,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrossBuilder<T, TT: TensorTrain<T>>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  pub(super) tt: TT,
  indices: MultiIndicesSet,
  cur_ker: usize,
  dmrg_state: DMRGState,
  delta: T::Real,
  argabsmax: Option<Vec<usize>>,
  absmax: Option<T>,
  restriction: usize,
  rank: usize,
}

impl<T, TT: TensorTrain<T>> CrossBuilder<T, TT>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  /// Returns a TTCross based builder of a Tensor Train.
  ///
  /// # Arguments
  /// 
  /// * 'rank' - maximal TT rank allowed during the method execution
  /// * 'delta' - the accuracy of the maxvol algorithm (typically should be
  ///   somewhat small, e.g. 0.01)
  /// * 'mode_dims' - a vector with dimensions of each mode of a Tensor Train
  /// * 'tt_opt' - a boolean flag showing if one needs to track data for TTOpt
  ///   optimization method (see <https://arxiv.org/abs/2205.00293>)
  pub fn new(
    rank: usize,
    delta: T::Real,
    mode_dims: &[usize],
    tt_opt: bool,
  ) -> Self
  {
    let tt = TT::new_random(mode_dims.to_owned(), rank);
    let indices = MultiIndicesSet::build_random(mode_dims, rank, usize::MAX);
    let absmax = if tt_opt { Some(T::zero()) } else { None };
    let argabsmax = if tt_opt { Some(vec![0; mode_dims.len()]) } else { None };
    Self {
      tt,
      indices,
      cur_ker: 0,
      dmrg_state: DMRGState::ToRight,
      delta,
      argabsmax,
      absmax,
      restriction: usize::MAX,
      rank,
    }
  }

  /// Restricts the exploration indices by a class where for each index one has not more than n non-zero elements.
  /// E.g. if one sets n = 2, then [0, 1, 0, 2, 0] is accepted, while [0, 1, 3, 2, 0] is not.
  /// This is useful when one needs to reconstruct a quantum state from a set of correlation functions whit restricted
  /// number of points.
  pub fn restrict(&mut self, n: usize) -> TTResult<()>
  {
    if n % 2 != 1 || n < 3 { return Err(TTError::Other("n must be odd and >= 3.")); }
    self.restriction = n;
    let indices = MultiIndicesSet::build_random(self.tt.get_mode_dims(), self.rank, (n - 1) / 2);
    let (left_bonds, right_bonds) = indices.get_bonds();
    for ((ker, old_right_bond, old_left_bond, mode_dim), (new_left_bond, new_right_bond)) 
    in unsafe { self.tt.iter_mut() }.zip(left_bonds.into_iter().zip(right_bonds))
    {
      let mut new_ker = T::random_normal(*mode_dim * new_left_bond * new_right_bond);
      std::mem::swap(&mut new_ker, ker);
      *old_left_bond = new_left_bond;
      *old_right_bond = new_right_bond;
    }
    self.indices = indices;
    Ok(())
  }

  /// Turns a TTCross builder into the corresponding Tensor Train.
  #[inline]
  pub fn to_tt(self) -> TT {
    self.tt
  }
  /// Returns the modulo argmax index found by the TTOpt method
  /// (see <https://arxiv.org/abs/2205.00293>) if flag tt_opt was turned
  /// ON at the initialization. Note, that TTOpt method finds maximum
  /// modulo element only approximately.
  #[inline]
  pub fn get_tt_opt_argmax(&self) -> Option<Vec<usize>> {
    self.argabsmax.clone()
  }
}

impl<T, TT: TensorTrain<T>> CrossBuilder<T, TT>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  /// Returns either an iterator over indices that must be evaluated or None.
  /// In case of None, nothing should be evaluated at the current step.
  ///
  /// # Note
  /// This method is mostly for development needs.
  #[inline]
  pub(super) fn get_args(
    &self,
  ) -> Option<impl IndexedParallelIterator<Item = Vec<usize>>>
  {
    let cur_ker = self.cur_ker;
    let dim = self.tt.get_mode_dims()[cur_ker];
    let left_bond = self.tt.get_left_bonds()[cur_ker];
    let right_bond = self.tt.get_right_bonds()[cur_ker];
    match self.dmrg_state {
      DMRGState::ToRight => {
        let local_indices = MultiIndices::new_one_side(dim);
        let left_indices = self.indices[cur_ker].0.product(&local_indices);
        let left_indices_len = left_indices.len();
        let iter = indices_par_iter((left_indices, self.indices[cur_ker].1.clone()), false);
        if cur_ker == (self.tt.get_len() - 1) {
          Some(iter)
        } else {
          if left_indices_len == right_bond {
            None
          } else {
            Some(iter)
          }
        }
      },
      DMRGState::ToLeft => {
        let local_indices = MultiIndices::new_one_side(dim);
        let right_indices = local_indices.product(&self.indices[cur_ker].1);
        let right_indices_len = right_indices.len();
        let iter = indices_par_iter((self.indices[cur_ker].0.clone(), right_indices), true);
        if cur_ker == 0 {
          Some(iter)
        } else {
          if right_indices_len == left_bond {
            None
          } else {
            Some(iter)
          }
        }
      },
    }
  }

  /// Performs an update step according to the obtained evaluated tensor elements.
  /// 
  /// # Arguments
  /// 
  /// * 'measurements' - an optional iterator over elements which were evaluated
  /// 
  /// This method is mostly for development needs.
  #[inline]
  pub(super) fn update(
    &mut self,
    measurements: Option<impl IndexedParallelIterator<Item = T>>
  ) -> TTResult<()>
  {
    let cur_ker = self.cur_ker;
    let dim = self.tt.get_mode_dims()[cur_ker];
    let left_bond = self.tt.get_left_bonds()[cur_ker];
    let right_bond = self.tt.get_right_bonds()[cur_ker];
    match self.dmrg_state {
      DMRGState::ToRight => {
        if cur_ker == (self.tt.get_len() - 1) {
          let iter = measurements.ok_or(TTError::EmptyUpdate)?;
          unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().into_par_iter().zip(iter).for_each(|(dst, src)| {
            *dst = src;
          }) };
          if let (Some(absmax), Some(argabsmax)) = (&mut self.absmax, &mut self.argabsmax)  {
            let (mut idx, val) = self.tt.get_kernels()[cur_ker].as_ref().into_iter().enumerate().max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap();
            if absmax.abs() < val.abs() {
              *absmax = *val;
              let lhs_idx = idx % left_bond;
              idx /= left_bond;
              let mid_idx = idx % dim;
              let rhs_idx = idx / dim;
              let (left_indices, right_indices) = self.indices[cur_ker].clone();
              *argabsmax = left_indices.release()[lhs_idx].iter().chain(Some(&mid_idx)).chain(right_indices.release()[rhs_idx].iter()).map(|x| *x).collect();
            }
          }
          self.dmrg_state = DMRGState::ToLeft;
        } else {
          if left_bond * dim == right_bond {
            let local_indices = MultiIndices::new_one_side(dim);
            unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut T::eye(right_bond)[..]) };
            self.indices[cur_ker + 1].0 = self.indices[cur_ker ].0.product(&local_indices);
            self.cur_ker += 1;
          } else {
            let iter = measurements.ok_or(TTError::EmptyUpdate)?;
            let mut m_buff: Vec<_> = iter.collect();
            if let (Some(absmax), Some(argabsmax)) = (&mut self.absmax, &mut self.argabsmax)  {
              let (mut idx, val) = m_buff.iter().enumerate().max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap();
              if absmax.abs() < val.abs() {
                *absmax = *val;
                let lhs_idx = idx % left_bond;
                idx /= left_bond;
                let mid_idx = idx % dim;
                let rhs_idx = idx / dim;
                let (left_indices, right_indices) = self.indices[cur_ker].clone();
                *argabsmax = left_indices.release()[lhs_idx].iter().chain(Some(&mid_idx)).chain(right_indices.release()[rhs_idx].iter()).map(|x| *x).collect();
              }
            }
            let m = NDArray::from_mut_slice(&mut m_buff, [left_bond * dim, right_bond])?;
            let mut aux_buff = unsafe { T::uninit_buff(right_bond.pow(2)) };
            let aux = NDArray::from_mut_slice(&mut aux_buff, [right_bond, right_bond])?;
            unsafe { m.qr(aux)? };
            let mut order = if self.restriction != usize::MAX {
              let left_indices = self.indices[cur_ker].0.product(&MultiIndices::new_one_side(dim));
              let (forbidden, must_have) = get_restrictions(&left_indices, (self.restriction - 1) / 2);
              unsafe { m.maxvol(self.delta, Some(&forbidden), Some(&[must_have]))? }
            } else {
              unsafe { m.maxvol(self.delta, None, None)? }
            };
            let mut reverse_order = Vec::with_capacity(order.len());
            unsafe { reverse_order.set_len(order.len()) };
            order.iter().enumerate().for_each(|(i, x)| {
              reverse_order[*x] = i;
            });
            let mut new_ker = unsafe { m.gen_f_array_from_axis_order(&reverse_order, 0).0 };
            unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut new_ker) };
            order.resize(right_bond, 0);
            let local_indices = MultiIndices::new_one_side(dim);
            let left_indices = self.indices[cur_ker].0.product(&local_indices).release();
            self.indices[cur_ker + 1].0 = order.into_iter().map(|i| left_indices[i].clone()).collect::<Vec<Vec<_>>>().into();
            self.cur_ker += 1;
          }
        }
      },
      DMRGState::ToLeft => {
        if cur_ker == 0 {
          let iter = measurements.ok_or(TTError::EmptyUpdate)?;
          unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().into_par_iter().zip(iter).for_each(|(dst, src)| {
            *dst = src;
          }) };
          if let (Some(absmax), Some(argabsmax)) = (&mut self.absmax, &mut self.argabsmax)  {
            let (mut idx, val) = self.tt.get_kernels()[cur_ker].as_ref().into_iter().enumerate().max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap();
            if absmax.abs() < val.abs() {
              *absmax = *val;
              let mid_idx = idx % dim;
              idx /= dim;
              let rhs_idx = idx % right_bond;
              let lhs_idx = idx / right_bond;
              let (left_indices, right_indices) = self.indices[cur_ker].clone();
              *argabsmax = left_indices.release()[lhs_idx].iter().chain(Some(&mid_idx)).chain(right_indices.release()[rhs_idx].iter()).map(|x| *x).collect();
            }
          }
          self.dmrg_state = DMRGState::ToRight;
        } else {
          if right_bond * dim == left_bond {
            let local_indices = MultiIndices::new_one_side(dim);
            unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut T::eye(left_bond)[..]) };
            self.indices[cur_ker - 1].1 = local_indices.product(&self.indices[cur_ker].1);
            self.cur_ker -= 1;
          } else {
            let local_indices = MultiIndices::new_one_side(dim);
            let right_indices = local_indices.product(&self.indices[cur_ker].1);
            let iter = measurements.ok_or(TTError::EmptyUpdate)?;
            let mut m_buff: Vec<_> = iter.collect();
            if let (Some(absmax), Some(argabsmax)) = (&mut self.absmax, &mut self.argabsmax)  {
              let (mut idx, val) = m_buff.iter().enumerate().max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap();
              if absmax.abs() < val.abs() {
                *absmax = *val;
                let mid_idx = idx % dim;
                idx /= dim;
                let rhs_idx = idx % right_bond;
                let lhs_idx = idx / right_bond;
                let (left_indices, right_indices) = self.indices[cur_ker].clone();
                *argabsmax = left_indices.release()[lhs_idx].iter().chain(Some(&mid_idx)).chain(right_indices.release()[rhs_idx].iter()).map(|x| *x).collect();
              }
            }
            let m = NDArray::from_mut_slice(&mut m_buff, [right_bond * dim, left_bond])?;
            let mut aux_buff = unsafe { T::uninit_buff(left_bond.pow(2)) };
            let aux = NDArray::from_mut_slice(&mut aux_buff, [left_bond, left_bond])?;
            unsafe { m.qr(aux)? };
            let mut order = if self.restriction != usize::MAX {
              let right_indices = MultiIndices::new_one_side(dim).product(&self.indices[cur_ker].1);
              let (forbidden, must_have) = get_restrictions(&right_indices, (self.restriction - 1) / 2);
              unsafe { m.maxvol(self.delta, Some(&forbidden), Some(&[must_have]))? }
            } else {
              unsafe { m.maxvol(self.delta, None, None)? }
            };
            let mut reverse_order = Vec::with_capacity(order.len());
            unsafe { reverse_order.set_len(order.len()) };
            order.iter().enumerate().for_each(|(i, x)| {
              reverse_order[*x] = i;
            });
            let mut new_ker = unsafe { m.transpose([1, 0])?.gen_f_array_from_axis_order(&reverse_order, 1).0 };
            unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut new_ker) };
            order.resize(left_bond, 0);
            let right_indices = right_indices.release();
            self.indices[cur_ker - 1].1 = order.into_iter().map(|i| right_indices[i].clone()).collect::<Vec<Vec<_>>>().into();
            self.cur_ker -= 1;
          }
        }
      },
    }
    Ok(())
  }

  /// Performs a step of cross approximation.
  /// 
  /// # Arguments
  /// 
  /// * 'f' - function, which is being reconstructed
  ///
  /// # Note the result is meaningful only when number of steps n holds n % modes_number == 0.
  #[inline]
  pub(super) fn next(
    &mut self,
    f: impl Fn(&[usize]) -> T + Sync,
  ) -> TTResult<()>
  {
    let measurements_iter = self.get_args().map(|it| it.map(|x| f(&x[..])));
    self.update(measurements_iter)?;
    if self.absmax.is_some() {
      if self.absmax.unwrap() != T::zero() {
        debug_assert_eq!(self.absmax.unwrap(), f(&self.argabsmax.as_deref().unwrap()), "Curr. ker: {} out of {}, dmrg state: {:?}", self.cur_ker, self.tt.get_len(), self.dmrg_state);
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::CrossBuilder;
  use crate::tt_vec::TTVec;
  use crate::TensorTrain;
  use linwrap::{LinalgComplex, LinalgReal};
  use num_complex::{
    ComplexFloat,
    Complex32,
    Complex64,
  };
  use num_traits::{One, NumCast};

  #[inline]
  fn cos_sqrt<T>(x: &[usize]) -> T
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let total_val: T = x.into_iter().enumerate().map(|(i, val)| {
      (T::from(*val).unwrap() - T::from(0.5).unwrap()) / T::from(2.).unwrap().powi(i as i32)
    }).sum();
    T::cos(total_val).sqrt()
  }

  #[inline]
  fn _test_cross<T>(
    acc: T::Real,
    maxvol_termination: T::Real,
    tt_size: usize,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
      let mut builder = CrossBuilder::<T, TTVec<T>>::new(25, maxvol_termination, &vec![2; tt_size], false);
      for _ in 0..tt_size {
        builder.next(cos_sqrt).unwrap();
      }
      assert!(builder.tt.get_kernels()[..(tt_size-1)].iter().all(|x| { x.into_iter().all(|y| {
        (*y).abs() < maxvol_termination + T::Real::one()
      }) }));
      for _ in 0..tt_size {
        builder.next(cos_sqrt).unwrap();
      }
      assert!(builder.tt.get_kernels()[1..].iter().all(|x| { x.into_iter().all(|y| {
        (*y).abs() < maxvol_termination + T::Real::one()
      }) }));
      for _ in 0..(4 * tt_size) {
        builder.next(cos_sqrt).unwrap();
      }
      let mut tt = builder.to_tt();
      let log_norm = tt.set_into_left_canonical().unwrap();
      let tt_based = (T::from(2.).unwrap() * log_norm - T::from(19. * (2.).ln()).unwrap()).exp();
      let exact = T::from(2. * (1.).sin()).unwrap();
      assert!((tt_based - exact).abs() < acc, "tt_based: {:?}, exact: {:?}", tt_based, exact);
      tt.truncate_left_canonical(<T::Real as NumCast>::from(1e-6).unwrap()).unwrap();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      tt.elementwise_prod(&tt_conj).unwrap();
      let index1 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
      let index2 = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
      let (mut log_abs, phase) = tt.log_eval_index(&index1).unwrap();
      log_abs = log_abs + log_norm + log_norm;
      assert!((log_abs.exp() * phase - cos_sqrt::<T>(&index1).powi(2)).abs() < <T::Real as NumCast>::from(1e-3).unwrap());
      let (mut log_abs, phase) = tt.log_eval_index(&index2).unwrap();
      log_abs = log_abs + log_norm + log_norm;
      assert!((log_abs.exp() * phase - cos_sqrt::<T>(&index2).powi(2)).abs() < <T::Real as NumCast>::from(1e-3).unwrap());
  }

  #[test]
  fn test_cross() {
    _test_cross::<f32>(      1e-3 , 0.001, 20);
    _test_cross::<f64>(      1e-10, 0.001, 20);
    _test_cross::<Complex32>(1e-3 , 0.001, 20);
    _test_cross::<Complex64>(1e-10, 0.001, 20);
  }
}