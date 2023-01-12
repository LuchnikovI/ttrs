use num_complex::{
  Complex32,
  Complex64,
};

use rayon::iter::IntoParallelIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

use linwrap::init_utils::{
  eye_f32,
  eye_f64,
  eye_c32,
  eye_c64,
  uninit_buff_f32,
  uninit_buff_f64,
  uninit_buff_c32,
  uninit_buff_c64,
};
use linwrap::NDArray;


use crate::{TTf32, TTf64, TTc32, TTc64};
use crate::utils::{
  build_random_indices,
  indices_prod,
  get_indices_iter,
};

use crate::tt_traits::{
  TTResult,
  TTError,
};

#[derive(Debug, Clone, Copy)]
enum DMRGState {
  ToLeft,
  ToRight,
}

macro_rules! impl_cross_builder {
  ($cross_name:ident, $complex_type:ty, $real_type:ty, $tt_trait:ident, $fn_gen:ident) => {

    #[derive(Debug, Clone)]
    pub struct $cross_name<T: $tt_trait> {
      pub(super) tt: T,
      left_indices: Vec<Vec<Vec<usize>>>,
      right_indices: Vec<Vec<Vec<usize>>>,
      cur_ker: usize,
      dmrg_state: DMRGState,
      delta: $real_type,
    }

    impl<T: $tt_trait> $cross_name<T>
    {
      /// This method initialize a TTCross builder.
      /// As an input it takes a maximal TT rank, a parameter delta,
      /// that specifies a stopping criteria of Maxvol algorithm (it should
      /// be sufficiently small, for example 0.01) and dimensions of Tensor Train
      /// modes.
      pub fn new(
        rank: usize,
        delta: $real_type,
        mode_dims: &[usize]
      ) -> Self
      {
        let tt = T::new_random(mode_dims.to_owned(), rank);
        let (left_indices, right_indices) = build_random_indices(mode_dims, tt.get_left_bonds(), tt.get_right_bonds());
        Self {
          tt,
          left_indices,
          right_indices,
          cur_ker: 0,
          dmrg_state: DMRGState::ToRight,
          delta,
        }
      }
      /// This method turns TTCross builder into the corresponding Tensor Train.
      pub fn to_tt(self) -> T {
        self.tt
      }
    }
  };
}

impl_cross_builder!(CBf32, f32,       f32, TTf32, random_normal_f32);
impl_cross_builder!(CBf64, f64,       f64, TTf64, random_normal_f64);
impl_cross_builder!(CBc32, Complex32, f32, TTc32, random_normal_c32);
impl_cross_builder!(CBc64, Complex64, f64, TTc64, random_normal_c64);

macro_rules! impl_next {
  ($cross_name:ident, $tt_trait:ident, $complex_type:ty, $fn_uninit_buff:ident, $fn_eye:ident) => {
    impl<T: $tt_trait> $cross_name<T> {

      /// This method returns either an iterator over indices that must be evaluated or None.
      /// In case of None, nothing should be evaluated at the current step.
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
            let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
            let left_indices = indices_prod(&self.left_indices[cur_ker], &local_indices);
            let iter = get_indices_iter(
              &self.right_indices[cur_ker],
              &left_indices,
              true,
            );
            if cur_ker == (self.tt.get_len() - 1) {
              Some(iter)
            } else {
              if left_indices.len() == right_bond {
                None
              } else {
                Some(iter)
              }
            }
          },
          DMRGState::ToLeft => {
            let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
            let right_indices = indices_prod(&local_indices, &self.right_indices[cur_ker]);
            let iter = get_indices_iter(
              &self.left_indices[cur_ker],
              &right_indices,
              false,
            );
            if cur_ker == 0 {
              Some(iter)
            } else {
              if right_indices.len() == left_bond {
                None
              } else {
                Some(iter)
              }
            }
          },
        }
      }

      /// This method perform an update step according to the obtained evaluated tensor elements.
      pub(super) fn update(
        &mut self,
        measurements: Option<impl IndexedParallelIterator<Item = $complex_type>>
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
              self.dmrg_state = DMRGState::ToLeft;
            } else {
              if left_bond * dim == right_bond {
                let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
                unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut $fn_eye(right_bond)[..]) };
                self.left_indices[cur_ker + 1] = indices_prod(&self.left_indices[cur_ker], &local_indices);
                self.cur_ker += 1;
              } else {
                let iter = measurements.ok_or(TTError::EmptyUpdate)?;
                let mut m_buff: Vec<_> = iter.collect();
                let m = NDArray::from_mut_slice(&mut m_buff, [right_bond, left_bond * dim])?;
                let mut aux_buff = unsafe { $fn_uninit_buff(right_bond.pow(2)) };
                let aux = NDArray::from_mut_slice(&mut aux_buff, [right_bond, right_bond])?;
                unsafe { m.rq(aux)? };
                let mut order = unsafe { m.maxvol(self.delta)? };
                let mut reverse_order = Vec::with_capacity(order.len());
                unsafe { reverse_order.set_len(order.len()) };
                order.iter().enumerate().for_each(|(i, x)| {
                  reverse_order[*x] = i;
                });
                unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut m.transpose([1, 0])?.gen_f_array_from_axis_order(&reverse_order, 0).0) };
                order.resize(right_bond, 0);
                let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
                let left_indices = indices_prod(&self.left_indices[cur_ker], &local_indices);
                self.left_indices[cur_ker + 1] = order.into_iter().map(|i| left_indices[i].clone()).collect();
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
              self.dmrg_state = DMRGState::ToRight;
            } else {
              if right_bond * dim == left_bond {
                let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
                unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut $fn_eye(left_bond)[..]) };
                self.right_indices[cur_ker - 1] = indices_prod(&local_indices, &self.right_indices[cur_ker]);
                self.cur_ker -= 1;
              } else {
                let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
                let right_indices = indices_prod(&local_indices, &self.right_indices[cur_ker]);
                let iter = measurements.ok_or(TTError::EmptyUpdate)?;
                let mut m_buff: Vec<_> = iter.collect();
                let m = NDArray::from_mut_slice(&mut m_buff, [left_bond, right_bond * dim])?;
                let mut aux_buff = unsafe { $fn_uninit_buff(left_bond.pow(2)) };
                let aux = NDArray::from_mut_slice(&mut aux_buff, [left_bond, left_bond])?;
                unsafe { m.qr(aux)? };
                let mut order = unsafe { m.maxvol(self.delta)? };
                let mut reverse_order = Vec::with_capacity(order.len());
                unsafe { reverse_order.set_len(order.len()) };
                order.iter().enumerate().for_each(|(i, x)| {
                  reverse_order[*x] = i;
                });
                unsafe { self.tt.get_kernels_mut()[cur_ker].as_mut().swap_with_slice(&mut m.gen_f_array_from_axis_order(&reverse_order, 1).0) };
                order.resize(left_bond, 0);
                self.right_indices[cur_ker - 1] = order.into_iter().map(|i| right_indices[i].clone()).collect();
                self.cur_ker -= 1;
              }
            }
          },
        }
        Ok(())
      }

      /// This method performs a step of cross approximation procedure. As an input it takes function
      /// that one tries to represent in terms of a Tensor Train. It evaluates the function
      /// number of times under the hood and update an estimation of a Tensor Train.
      pub(super) fn next(
        &mut self,
        f: impl Fn(&[usize]) -> $complex_type + Sync,
      ) -> TTResult<()>
      {
        let measurements_iter = self.get_args().map(|it| it.map(|x| f(&x[..])));
        self.update(measurements_iter)
      }
    }        
  };
}

impl_next!(CBf32, TTf32, f32,       uninit_buff_f32, eye_f32);
impl_next!(CBf64, TTf64, f64,       uninit_buff_f64, eye_f64);
impl_next!(CBc32, TTc32, Complex32, uninit_buff_c32, eye_c32);
impl_next!(CBc64, TTc64, Complex64, uninit_buff_c64, eye_c64);

#[cfg(test)]
mod tests {
  use super::{
    CBf32,
    CBf64,
    CBc32,
    CBc64,
  };
  use crate::tt_vec::TTVec;
  use crate::{
    TTf32,
    TTf64,
    TTc32,
    TTc64,
  };
  use num_complex::{
    Complex32,
    Complex64,
    ComplexFloat,
  };

  macro_rules! test_cross {
    ($cross_type:ident, $complex_type:ty, $real_type:ty, $acc:expr) => {
      fn cos_sqrt(x: &[usize]) -> $complex_type {
        let total_val: $real_type = x.into_iter().enumerate().map(|(i, val)| {
          (*val as $real_type - 0.5) / (2i64.pow(i as u32) as $real_type)
        }).sum();
        <$complex_type>::cos(<$complex_type>::from(total_val)).sqrt()
      }
      let mut builder = $cross_type::<TTVec<_>>::new(25, 0.001, &[2; 20]);
      for _ in 0..20 {
        builder.next(cos_sqrt).unwrap();
      }
      assert!(builder.tt.get_kernels()[..19].iter().all(|x| { x.into_iter().all(|y| {
        y.abs() < 1.001
      }) }));
      for _ in 0..20 {
        builder.next(cos_sqrt).unwrap();
      }
      assert!(builder.tt.get_kernels()[1..].iter().all(|x| { x.into_iter().all(|y| {
        y.abs() < 1.001
      }) }));
      for _ in 0..(4 * 20) {
        builder.next(cos_sqrt).unwrap();
      }
      let mut tt = builder.to_tt();
      let log_norm = tt.set_into_left_canonical().unwrap();
      let tt_based = (2. * log_norm - 19. * (2 as $real_type).ln()).exp();
      let exact = 2. * (1 as $real_type).sin();
      assert!((tt_based - exact).abs() < $acc);
      tt.truncate_left_canonical(1e-6).unwrap();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      tt.elementwise_prod(&tt_conj).unwrap();
      let index1 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
      let index2 = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
      assert!(((tt.log_eval_index(&index1).unwrap() + 2. * log_norm).exp() - cos_sqrt(&index1).powi(2)).abs() < 1e-3);
      assert!(((tt.log_eval_index(&index2).unwrap() + 2. * log_norm).exp() - cos_sqrt(&index2).powi(2)).abs() < 1e-3);
    };
  }

  #[test]
  fn test_cross() {
    { test_cross!(CBf32, f32,       f32, 1e-3 ); }
    { test_cross!(CBf64, f64,       f64, 1e-10); }
    { test_cross!(CBc32, Complex32, f32, 1e-3 ); }
    { test_cross!(CBc64, Complex64, f64, 1e-10); }
  }
}