use num_complex::{
    Complex32,
    Complex64,
};
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

use crate::ttcross::{
  CBf32,
  CBf64,
  CBc32,
  CBc64,
};

use crate::utils::build_bonds;

#[derive(Debug, Clone)]
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
    ($fn_gen:ident, $complex_type:ty, $real_type:ty, $cross_type:ty) => {
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
            ) -> TTResult<Self> {
                let kers_num = mode_dims.len();
                let mut builder = <$cross_type>::new(max_rank, delta, mode_dims);
                for _ in 0..(kers_num * sweeps_num) {
                  builder.next(&f)?;
                }
                Ok(builder.to_tt()) 
            }
        }
    };
}

impl_random!(random_normal_f32, f32      , f32, CBf32<TTVec<_>>);
impl_random!(random_normal_f64, f64      , f64, CBf64<TTVec<_>>);
impl_random!(random_normal_c32, Complex32, f32, CBc32<TTVec<_>>);
impl_random!(random_normal_c64, Complex64, f64, CBc64<TTVec<_>>);

// ----------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use num_complex::ComplexFloat;
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
      let log_dot = tt.dot(&tt_conj).unwrap().ln().abs();
      let log_norm = tt.$set_to_canonical_method().unwrap();
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
      let diff = tt.dot(&tt_conj).unwrap() +
                 tt_clone.dot(&tt_clone_conj).unwrap() -
                 tt.dot(&tt_clone_conj).unwrap() -
                 tt_clone.dot(&tt_conj).unwrap();
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
}
