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

use crate::ttcross::CrossBuilder;

#[derive(Debug, Clone)]
pub struct TTVec<T>
{
    pub(super) kernels:     Vec<Vec<T>>,
    pub(super) left_bonds:  Vec<usize>,
    pub(super) right_bonds: Vec<usize>,
    pub(super) mode_dims:   Vec<usize>,
}

macro_rules! impl_tt_trait {
    ($complex_type:ty, $trait_type:ty) => {
        impl $trait_type for TTVec<$complex_type>
        {
            type Buff = Vec<$complex_type>;
            type Kers = Vec<Self::Buff>;

            fn new(kernels: Vec<Vec<$complex_type>>, internal_bonds:Vec<usize>, mode_dims:Vec<usize>,) -> Self {
                let mut right_bonds = internal_bonds.clone();
                right_bonds.push(1);
                let left_bonds = [vec![1], internal_bonds].concat();
                debug_assert!(&left_bonds[1..] == &right_bonds[..(right_bonds.len() - 1)], "Left and right bonds incorrect intersection. Line: {}, File: {}", line!(), file!());
                debug_assert!(left_bonds[0] == 1, "The most left bond is not equal to 1.");
                debug_assert!(right_bonds[right_bonds.len() - 1] == 1, "The most right bond is not equal to 1. Line: {}, File: {}", line!(), file!());
                Self { kernels, left_bonds, right_bonds, mode_dims }
            }
            fn from_cross_builder(builder: CrossBuilder<$complex_type>) -> Self {
                let mut internal_bonds = builder.right_bonds;
                internal_bonds.resize(builder.kernels.len()-1, 0);
                debug_assert!(&internal_bonds[..] == &builder.left_bonds[1..], "Incorrect internal bond. Line: {}, File: {}", line!(), file!());
                Self::new(
                  builder.kernels,
                  internal_bonds,
                  builder.mode_dims,
                )
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

impl_tt_trait!(f32,       TTf32);
impl_tt_trait!(f64,       TTf64);
impl_tt_trait!(Complex32, TTc32);
impl_tt_trait!(Complex64, TTc64);

macro_rules! impl_random {
    ($fn_gen:ident, $complex_type:ty, $real_type:ty) => {
        impl TTVec<$complex_type> {
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
                let internal_bonds = vec![max_rank; modes_num - 1];
                let mode_dims = mode_dims.to_owned();
                Self::new(kernels, internal_bonds, mode_dims)
              }

            pub fn ttcross(
                mode_dims: &[usize],
                max_rank: usize,
                delta: $real_type,
                f: impl Fn(&[usize]) -> $complex_type + Sync,
                sweeps_num: usize,
            ) -> TTResult<Self> {
                let kers_num = mode_dims.len();
                let mut builder = CrossBuilder::<$complex_type>::new(max_rank, delta, mode_dims);
                for _ in 0..(kers_num * sweeps_num) {
                  builder.next(&f)?;
                }
                Ok(Self::from_cross_builder(builder)) 
            }
        }
    };
}

impl_random!(random_normal_f32, f32      , f32);
impl_random!(random_normal_f64, f64      , f64);
impl_random!(random_normal_c32, Complex32, f32);
impl_random!(random_normal_c64, Complex64, f64);

// ----------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use num_complex::ComplexFloat;
  use super::*;
  macro_rules! test_dot_and_canonical {
    ($complex_type:ident, $set_to_canonical_method:ident) => {
      let mut tt = TTVec::<$complex_type>::new_random_normal(&[2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
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
      let mut tt = TTVec::<$complex_type>::new_random_normal(&[2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
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
