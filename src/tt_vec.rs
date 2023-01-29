use num_complex::{
    Complex32,
    Complex64,
};
use serde::{Serialize, Deserialize};
use linwrap::init_utils::{
    random_normal_f32,
    random_normal_f64,
    random_normal_c32,
    random_normal_c64,
};
use crate::tt_traits::{
    TTIter,
    TTIterMut,
    TTResult,
    TTf32,
    TTf64,
    TTc32,
    TTc64,
};

use crate::tt_cross::{
  CBf32,
  CBf64,
  CBc32,
  CBc64,
};

use crate::utils::build_bonds;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TTVec<T>
{
    pub(super) kernels:     Vec<Vec<T>>,
    pub(super) left_bonds:  Vec<usize>,
    pub(super) right_bonds: Vec<usize>,
    pub(super) mode_dims:   Vec<usize>,
}

macro_rules! impl_tt_trait {
    ($complex_type:ty, $trait_type:ty, $fn_gen:ident) => {
        impl $trait_type for TTVec<$complex_type>
        {
            type Buff = Vec<$complex_type>;
            type Kers = Vec<Self::Buff>;

            fn new_random(
              mode_dims: Vec<usize>,
              max_rank:  usize,
            ) -> Self {
              let (left_bonds, right_bonds) = build_bonds(&mode_dims, max_rank);
              let mut kernels: Vec<Vec<$complex_type>> = Vec::with_capacity(mode_dims.len());
              for (left_bond, (dim, right_bond)) in left_bonds.iter().zip(mode_dims.iter().zip(right_bonds.iter())) {
                kernels.push($fn_gen(*left_bond * *dim * *right_bond));
              }
              Self { kernels, left_bonds, right_bonds, mode_dims }
            }

            fn get_kernels(&self) ->  &[Vec<$complex_type>] {
                &self.kernels
            }
            fn get_left_bonds(&self) ->  &[usize] {
                &self.left_bonds
            }
            fn get_right_bonds(&self) ->  &[usize] {
                &self.right_bonds
            }
            fn get_mode_dims(&self) ->  &[usize] {
                &self.mode_dims
            }
            fn get_len(&self) -> usize {
                self.kernels.len()
            }
            fn iter<'a>(&'a self) -> TTIter<'a, $complex_type> {
                TTIter {
                    kernels_iter: self.kernels.iter(),
                    right_bonds_iter: self.right_bonds.iter(),
                    left_bonds_iter: self.left_bonds.iter(),
                    mode_dims_iter: self.mode_dims.iter(),
                }
            }
            unsafe fn get_kernels_mut(&mut self) ->  &mut[Vec<$complex_type>] {
                &mut self.kernels
            }
            unsafe fn get_left_bonds_mut(&mut self) ->  &mut[usize] {
                &mut self.left_bonds
            }
            unsafe fn get_right_bonds_mut(&mut self) ->  &mut[usize] {
                &mut self.right_bonds
            }
            unsafe fn iter_mut<'a>(&'a mut self) -> TTIterMut<'a, $complex_type> {
                TTIterMut {
                    kernels_iter: self.kernels.iter_mut(),
                    right_bonds_iter: self.right_bonds.iter_mut(),
                    left_bonds_iter: self.left_bonds.iter_mut(),
                    mode_dims_iter: self.mode_dims.iter_mut(),
                  }
            }
        }
    };
}

impl_tt_trait!(f32,       TTf32, random_normal_f32);
impl_tt_trait!(f64,       TTf64, random_normal_f64);
impl_tt_trait!(Complex32, TTc32, random_normal_c32);
impl_tt_trait!(Complex64, TTc64, random_normal_c64);

macro_rules! impl_random {
    ($fn_gen:ident, $complex_type:ty, $real_type:ty, $cross_type:ty, $complex_one:expr) => {
        impl TTVec<$complex_type> {

            /// This method runs the TTCross algorithm reconstructing a Tensor Train representation
            /// of a function f. As an input, it takes mode dimensions, maximal TT rank, parameter delta that determines
            /// a Maxvol algorithm stopping criteria (this parameter should be small, e.g. 0.01),
            /// function itself and number of DMRG sweeps that one needs to perform.
            pub fn ttcross(
                mode_dims: &[usize],
                max_rank: usize,
                delta: $real_type,
                f: impl Fn(&[usize]) -> $complex_type + Sync,
                sweeps_num: usize,
                tt_opt: bool,
            ) -> TTResult<(Self, Option<Vec<usize>>)> {
                let kers_num = mode_dims.len();
                let mut builder = <$cross_type>::new(max_rank, delta, mode_dims, tt_opt);
                for _ in 0..(kers_num * sweeps_num) {
                  builder.next(&f)?;
                }
                let argabsmax = builder.get_tt_opt_argmax();
                Ok((builder.to_tt(), argabsmax))
            }
          
            pub fn new_ones(
              mode_dims: Vec<usize>,
            ) -> Self
            {
              let len = mode_dims.len();
              let left_bonds = vec![1; len];
              let right_bonds = left_bonds.clone();
              let mut kernels = Vec::with_capacity(len);
              for dim in &mode_dims {
                kernels.push(vec![$complex_one; *dim]);
              }
              Self {
                kernels,
                left_bonds,
                right_bonds,
                mode_dims,
              }
            }
        }
    };
}

impl_random!(random_normal_f32, f32      , f32, CBf32<TTVec<_>>, 1.                    );
impl_random!(random_normal_f64, f64      , f64, CBf64<TTVec<_>>, 1.                    );
impl_random!(random_normal_c32, Complex32, f32, CBc32<TTVec<_>>, Complex32::new(1., 0.));
impl_random!(random_normal_c64, Complex64, f64, CBc64<TTVec<_>>, Complex64::new(1., 0.));

// ----------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use num_complex::{
    ComplexFloat,
    Complex,
  };
  use crate::tt_traits::{
    TTf32,
    TTf64,
    TTc32,
    TTc64,
  };
  use super::*;

  macro_rules! test_dot_and_canonical {
    ($complex_type:ident, $set_to_canonical_method:ident) => {
      let mut tt = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      let tt_clone = tt.clone();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let log_dot = tt.log_dot(&tt_conj).unwrap().abs();
      let log_norm = tt.$set_to_canonical_method().unwrap();
      assert!((log_dot - 2. * log_norm).abs() < 1e-5);
      tt_conj = tt.clone();
      tt_conj.conj();
      let log_dot = tt.log_dot(&tt_conj).unwrap().abs();
      assert!(log_dot.abs() < 1e-5);
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      let diff = tt.log_dot(&tt_conj).unwrap().exp() +
                 tt_clone.log_dot(&tt_clone_conj).unwrap().exp() / (2. * log_norm).exp() -
                 tt.log_dot(&tt_clone_conj).unwrap().exp() / log_norm.exp()-
                 tt_clone.log_dot(&tt_conj).unwrap().exp() / log_norm.exp();
      assert!(diff.abs() < 1e-5);
      assert!((
        tt.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap() -
        tt_clone.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap() + log_norm
      ).abs() < 1e-5);
    };
  }

  #[test]
  fn test_dot_and_canonical()
  {
    test_dot_and_canonical!(f32,       set_into_left_canonical );
    test_dot_and_canonical!(f64,       set_into_left_canonical );
    test_dot_and_canonical!(Complex32, set_into_left_canonical );
    test_dot_and_canonical!(Complex64, set_into_left_canonical );
    test_dot_and_canonical!(f32,       set_into_right_canonical);
    test_dot_and_canonical!(f64,       set_into_right_canonical);
    test_dot_and_canonical!(Complex32, set_into_right_canonical);
    test_dot_and_canonical!(Complex64, set_into_right_canonical);
  }

  macro_rules! test_log_sum {
    ($complex_type:ty) => {
      let mut tt = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 15);
      let tt_other = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 15);
      let log_dot_val = tt.log_dot(&tt_other).unwrap();
      tt.elementwise_prod(&tt_other).unwrap();
      let log_sum_val = tt.log_sum().unwrap();
      assert!((log_sum_val - log_dot_val).abs() < 1e-4);
    };
  }

  #[test]
  fn test_log_sum() {
    test_log_sum!(f32      );
    test_log_sum!(f64      );
    test_log_sum!(Complex32);
    test_log_sum!(Complex64);
  }

  macro_rules! test_truncation {
    ($complex_type:ident, $set_to_canonical_method:ident, $truncation_method:ident) => {
      let mut tt = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      tt.$set_to_canonical_method().unwrap();
      let tt_clone = tt.clone();
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      tt.$truncation_method(0.001).unwrap();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let diff = tt.log_dot(&tt_conj).unwrap().exp() +
                 tt_clone.log_dot(&tt_clone_conj).unwrap().exp() -
                 tt.log_dot(&tt_clone_conj).unwrap().exp() -
                 tt_clone.log_dot(&tt_conj).unwrap().exp();
      assert!(diff.abs() < 1e-5)
    };
  }

  #[test]
  fn test_truncation() {
    test_truncation!(f32,       set_into_left_canonical,  truncate_left_canonical );
    test_truncation!(f64,       set_into_left_canonical,  truncate_left_canonical );
    test_truncation!(Complex32, set_into_left_canonical,  truncate_left_canonical );
    test_truncation!(Complex64, set_into_left_canonical,  truncate_left_canonical );
    test_truncation!(f32,       set_into_right_canonical, truncate_right_canonical);
    test_truncation!(f64,       set_into_right_canonical, truncate_right_canonical);
    test_truncation!(Complex32, set_into_right_canonical, truncate_right_canonical);
    test_truncation!(Complex64, set_into_right_canonical, truncate_right_canonical);
  }

  macro_rules! test_mul_by_scalar {
    ($complex_type:ty, $scalar:expr) => {
      let mut tt = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      let tt_clone = tt.clone();
      tt.mul_by_scalar($scalar);
      let val1 = tt.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap();
      let val2 = tt_clone.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap() + $scalar.ln();
      assert!((val1 - val2).abs() < 1e-4);
    }
  }

  #[test]
  fn test_mul_by_scalar() {
    test_mul_by_scalar!(f32,       2.28                   );
    test_mul_by_scalar!(f64,       2.28                   );
    test_mul_by_scalar!(Complex32, Complex::new(4.2, 2.28));
    test_mul_by_scalar!(Complex64, Complex::new(4.2, 2.28));
  }

  macro_rules! test_elementwise_sum {
    ($complex_type:ty) => {
      let mut tt1 = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 35);
      tt1.set_into_left_canonical().unwrap();
      let mut tt2 = TTVec::<$complex_type>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 25);
      tt2.set_into_left_canonical().unwrap();
      let val1 = tt1.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().exp() +
        tt2.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().exp();
      tt1.elementwise_sum(&tt2).unwrap();
      let val2 = tt1.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().exp();
      assert!((val1 - val2).abs() / (val1.abs() + val2.abs()) < 1e-5);
    };
  }

  #[test]
  fn test_elementwise_sum()
  {
    test_elementwise_sum!(f32      );
    test_elementwise_sum!(f64      );
    test_elementwise_sum!(Complex32);
    test_elementwise_sum!(Complex64);
  }

  #[inline]
  fn idx_to_arg(x: &[usize]) -> f64 {
      x.into_iter().enumerate().map(|(i, x)| {
          2. * (*x as f64) / 2f64.powi(i as i32)
      }).sum()
  }
  #[inline]
  fn target_function(x: &[usize]) -> f64 {
      let arg = idx_to_arg(x);
      ((arg - 2.512345678).powi(2) + 1e-3).ln() * (15. * (arg - 2.512345678)).cos()
  }

  macro_rules! test_argmax_modulo {
    ($complex_type:ty, $real_type:ty) => {
      let modes = vec![2; 20];
      let (tt, _) = TTVec::<$complex_type>::ttcross(&modes, 20, 0.01, |x| <$complex_type>::from(target_function(x)), 4, false).unwrap();
      let argmax = tt.argmax_modulo(1e-8, 0, 0, 10).unwrap();
      assert!((idx_to_arg(&argmax) - 2.512345678).abs() < 1e-5)
    };
  }

  #[test]
  fn test_optima_tt_max() {
    test_argmax_modulo!(f64,       f64);
    test_argmax_modulo!(Complex64, f64);
  }
}
