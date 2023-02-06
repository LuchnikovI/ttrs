use linwrap::{
  LinalgComplex,
  LinalgReal,
};
use num_complex::ComplexFloat;
use serde::{Serialize, Deserialize};
use crate::tt_traits::{
    TTIter,
    TTIterMut,
    TTResult,
    TensorTrain,
    TTError,
};

use crate::tt_cross::CrossBuilder;

use crate::utils::build_bonds;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TTVec<T>
{
    pub(super) kernels:     Vec<Vec<T>>,
    pub(super) left_bonds:  Vec<usize>,
    pub(super) right_bonds: Vec<usize>,
    pub(super) mode_dims:   Vec<usize>,
    pub(super) orth_center: Option<usize>,
}

impl<T> TensorTrain<T> for TTVec<T>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
    type Buff = Vec<T>;
    type Kers = Vec<Self::Buff>;

    #[inline]
    fn new_random(
      mode_dims: Vec<usize>,
      max_rank:  usize,
    ) -> Self {
      let (left_bonds, right_bonds) = build_bonds(&mode_dims, max_rank);
      let mut kernels: Vec<Vec<T>> = Vec::with_capacity(mode_dims.len());
      for (left_bond, (dim, right_bond)) in left_bonds.iter().zip(mode_dims.iter().zip(right_bonds.iter())) {
        kernels.push(T::random_normal(*left_bond * *dim * *right_bond));
      }
      Self { kernels, left_bonds, right_bonds, mode_dims, orth_center: None }
    }
    #[inline]
    unsafe fn build_from_raw_parts(
        mode_dims: Vec<usize>,
        bond_dims: Vec<usize>,
        kernels:   Vec<Vec<T>>,
      ) -> Self {
        let mut left_bonds = vec![1usize];
        left_bonds.extend_from_slice(&bond_dims);
        let mut right_bonds = bond_dims;
        right_bonds.push(1);
        Self { kernels, left_bonds, right_bonds, mode_dims, orth_center: None }
    }
    #[inline]
    fn get_orth_center_coordinate(&self) -> TTResult<usize> {
        self.orth_center.ok_or(TTError::UndefinedOrthCenter)
    }
    #[inline]
    unsafe fn set_orth_center_coordinate(&mut self, coord: Option<usize>) {
        self.orth_center = coord;
    }
    #[inline]
    fn get_kernels(&self) ->  &[Vec<T>] {
        &self.kernels
    }
    #[inline]
    fn get_left_bonds(&self) ->  &[usize] {
        &self.left_bonds
    }
    #[inline]
    fn get_right_bonds(&self) ->  &[usize] {
        &self.right_bonds
    }
    fn get_mode_dims(&self) ->  &[usize] {
        &self.mode_dims
    }
    #[inline]
    fn get_len(&self) -> usize {
        self.kernels.len()
    }
    fn iter<'a>(&'a self) -> TTIter<'a, T> {
        TTIter {
            kernels_iter: self.kernels.iter(),
            right_bonds_iter: self.right_bonds.iter(),
            left_bonds_iter: self.left_bonds.iter(),
            mode_dims_iter: self.mode_dims.iter(),
        }
    }
    #[inline]
    unsafe fn get_kernels_mut(&mut self) ->  &mut[Vec<T>] {
        &mut self.kernels
    }
    #[inline]
    unsafe fn get_left_bonds_mut(&mut self) ->  &mut[usize] {
        &mut self.left_bonds
    }
    #[inline]
    unsafe fn get_right_bonds_mut(&mut self) ->  &mut[usize] {
        &mut self.right_bonds
    }
    #[inline]
    unsafe fn iter_mut<'a>(&'a mut self) -> TTIterMut<'a, T> {
        TTIterMut {
            kernels_iter: self.kernels.iter_mut(),
            right_bonds_iter: self.right_bonds.iter_mut(),
            left_bonds_iter: self.left_bonds.iter_mut(),
            mode_dims_iter: self.mode_dims.iter_mut(),
          }
    }
}

impl<T: ComplexFloat> TTVec<T>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{

    /// Returns a Tensor Train representation of a function and optionally 
    /// result of TTOpt algorithm execution (see <https://arxiv.org/abs/2205.00293>)
    /// that is an approximate modulo argmax index.
    /// 
    /// # Arguments
    /// 
    /// * 'mode_dims' - a vector with dimensions of each mode of a Tensor Train
    /// * 'max_rank' - maximal TT rank allowed during the method execution
    /// * 'delta' - the accuracy of the maxvol algorithm (typically should be
    ///   somewhat small, e.g. 0.01)
    /// * 'f' - function that is being reconstructed in the Tensor Train form
    /// * 'sweeps_num' - number of sweeps (the entire Tensor Train traversals)
    /// * 'tt_opt' - a boolean flag showing if one needs to track data for TTOpt
    ///   optimization method (see <https://arxiv.org/abs/2205.00293>)
    pub fn ttcross(
        mode_dims: &[usize],
        max_rank: usize,
        delta: T::Real,
        f: impl Fn(&[usize]) -> T + Sync,
        sweeps_num: usize,
        tt_opt: bool,
    ) -> TTResult<(Self, Option<Vec<usize>>)> {
        let kers_num = mode_dims.len();
        let mut builder = CrossBuilder::new(max_rank, delta, mode_dims, tt_opt);
        for _ in 0..(kers_num * sweeps_num) {
          builder.next(&f)?;
        }
        let argabsmax = builder.get_tt_opt_argmax();
        Ok((builder.to_tt(), argabsmax))
    }
  
  /// Return a Tensor Train that represents a tensor with
  /// all elements equal to 1.
    pub fn new_ones(
      mode_dims: Vec<usize>,
    ) -> Self
    {
      let len = mode_dims.len();
      let left_bonds = vec![1; len];
      let right_bonds = left_bonds.clone();
      let mut kernels = Vec::with_capacity(len);
      for dim in &mode_dims {
        kernels.push(vec![T::one(); *dim]);
      }
      Self {
        kernels,
        left_bonds,
        right_bonds,
        mode_dims,
        orth_center: None,
      }
    }
}

// ----------------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {

use num_complex::{
    ComplexFloat,
    Complex,
    Complex32,
    Complex64,
  };
  use crate::tt_traits::TensorTrain;
  use super::*;

  #[inline]
  fn exp<T: ComplexFloat>(x: (T, T)) -> T {
    x.0.exp() * x.1
  }

  #[inline]
  fn _test_dot_and_canonical<T>(
    is_left_canonical: bool,
    acc: T::Real,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
      let mut tt = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      let tt_clone = tt.clone();
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let (log_dot, phase) = tt.log_dot(&tt_conj).unwrap();
      assert!((phase - T::one()).abs() < acc);
      let log_norm = if is_left_canonical {
        tt.set_into_left_canonical().unwrap()
      } else {
        tt.set_into_right_canonical().unwrap()
      };
      assert!((log_dot - log_norm - log_norm).abs() < acc);
      tt_conj = tt.clone();
      tt_conj.conj();
      let log_dot = tt.log_dot(&tt_conj).unwrap().0;
      assert!(log_dot.abs() < acc);
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      let (log_norm_tt_tt_dot, phase_tt_tt_dot) = tt.log_dot(&tt_conj).unwrap();
      assert!((phase_tt_tt_dot - T::one()).abs() < acc);
      let (mut log_norm_tt_clone_tt_clone_dot, phase_tt_clone_tt_clone_dot) = tt_clone.log_dot(&tt_clone_conj).unwrap();
      log_norm_tt_clone_tt_clone_dot = log_norm_tt_clone_tt_clone_dot - log_norm - log_norm;
      assert!((phase_tt_clone_tt_clone_dot - T::one()).abs() < acc);
      let (mut log_norm_tt_tt_clone_dot, phase_tt_tt_clone_dot) = tt.log_dot(&tt_clone_conj).unwrap();
      log_norm_tt_tt_clone_dot = log_norm_tt_tt_clone_dot - log_norm;
      assert!((phase_tt_tt_clone_dot - T::one()).abs() < acc);
      let (mut log_norm_tt_clone_tt_dot, phase_tt_tt_clone_dot) = tt_clone.log_dot(&tt_conj).unwrap();
      log_norm_tt_clone_tt_dot = log_norm_tt_clone_tt_dot - log_norm;assert!((phase_tt_tt_clone_dot - T::one()).abs() < acc);
      assert!((phase_tt_tt_clone_dot - T::one()).abs() < acc);
      let diff = log_norm_tt_tt_dot.exp() +
                 log_norm_tt_clone_tt_clone_dot.exp() -
                 log_norm_tt_tt_clone_dot.exp() -
                 log_norm_tt_clone_tt_dot.exp();
      assert!(diff.abs() < acc);
      assert!((
        tt.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().0 -
        tt_clone.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().0 + log_norm
      ).abs() < acc);
      assert!((
        tt.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().1 -
        tt_clone.log_eval_index(&[1, 2, 0, 4, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap().1
      ).abs() < acc);
  }

  #[test]
  fn test_dot_and_canonical()
  {
    _test_dot_and_canonical::<f32>(      true , 1e-4);
    _test_dot_and_canonical::<f64>(      true , 1e-9);
    _test_dot_and_canonical::<Complex32>(true , 1e-4);
    _test_dot_and_canonical::<Complex64>(true , 1e-9);
    _test_dot_and_canonical::<f32>(      false, 1e-4);
    _test_dot_and_canonical::<f64>(      false, 1e-9);
    _test_dot_and_canonical::<Complex32>(false, 1e-4);
    _test_dot_and_canonical::<Complex64>(false, 1e-9);
  }

  #[inline]
  fn _test_log_sum<T>(acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let mut tt = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 15);
    let tt_other = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 15);
    let log_dot_val = tt.log_dot(&tt_other).unwrap();
    tt.elementwise_prod(&tt_other).unwrap();
    let log_sum_val = tt.log_sum().unwrap();
    assert!((log_sum_val.0 - log_dot_val.0).abs() < acc);
    assert!((log_sum_val.1 - log_dot_val.1).abs() < acc);
  }

  #[test]
  fn test_log_sum()
  {
    _test_log_sum::<f32>(      1e-5 );
    _test_log_sum::<f64>(      1e-10);
    _test_log_sum::<Complex32>(1e-5 );
    _test_log_sum::<Complex64>(1e-10);
  }

  #[inline]
  fn _test_truncation<T>(does_truncate_left_canonical: bool, acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
      let mut tt = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      if does_truncate_left_canonical {
        tt.set_into_left_canonical().unwrap();
      } else {
        tt.set_into_right_canonical().unwrap();
      };
      let tt_clone = tt.clone();
      let mut tt_clone_conj = tt_clone.clone();
      tt_clone_conj.conj();
      if does_truncate_left_canonical {
        tt.truncate_left_canonical(acc).unwrap();
      } else {
        tt.truncate_right_canonical(acc).unwrap();
      }
      let mut tt_conj = tt.clone();
      tt_conj.conj();
      let diff = exp(tt.log_dot(&tt_conj).unwrap()) +
                 exp(tt_clone.log_dot(&tt_clone_conj).unwrap()) -
                 exp(tt.log_dot(&tt_clone_conj).unwrap()) -
                 exp(tt_clone.log_dot(&tt_conj).unwrap());
      assert!(diff.abs() < acc)
  }

  #[test]
  fn test_truncation()
  {
    _test_truncation::<f32>(      true , 1e-5 );
    _test_truncation::<f64>(      true , 1e-10);
    _test_truncation::<Complex32>(true , 1e-5 );
    _test_truncation::<Complex64>(true , 1e-10);
    _test_truncation::<f32>(      false, 1e-5 );
    _test_truncation::<f64>(      false, 1e-10);
    _test_truncation::<Complex32>(false, 1e-5 );
    _test_truncation::<Complex64>(false, 1e-10);
  }

  #[inline]
  fn _test_mul_by_scalar<T>(scalar: T, acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
      let mut tt = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 64);
      let tt_clone = tt.clone();
      tt.mul_by_scalar(scalar);
      let (norm1, phase1) = tt.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap();
      let (norm2, phase2) = tt_clone.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap();
      assert!(((norm1 - norm2).exp() * (phase1 / phase2) - scalar).abs() < acc);
  }

  #[test]
  fn test_mul_by_scalar()
  {
    _test_mul_by_scalar::<f32>(      2.28                   , 1e-4 );
    _test_mul_by_scalar::<f64>(      2.28                   , 1e-9);
    _test_mul_by_scalar::<Complex32>(Complex::new(4.2, 2.28), 1e-4 );
    _test_mul_by_scalar::<Complex64>(Complex::new(4.2, 2.28), 1e-9);
  }

  #[inline]
  fn _test_elementwise_sum<T>(acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
      let mut tt1 = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 35);
      tt1.set_into_left_canonical().unwrap();
      let mut tt2 = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3], 25);
      tt2.set_into_left_canonical().unwrap();
      let val1 = exp(tt1.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap()) +
        exp(tt2.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap());
      tt1.elementwise_sum(&tt2).unwrap();
      let val2 = exp(tt1.log_eval_index(&[1, 1, 0, 1, 2, 3, 3, 2, 0, 0, 1, 1]).unwrap());
      assert!((val1 - val2).abs() / (val1.abs() + val2.abs()) < acc);
  }

  #[test]
  fn test_elementwise_sum()
  {
    _test_elementwise_sum::<f32      >(1e-5 );
    _test_elementwise_sum::<f64      >(1e-10);
    _test_elementwise_sum::<Complex32>(1e-5 );
    _test_elementwise_sum::<Complex64>(1e-10);
  }

  #[inline]
  fn idx_to_arg(x: &[usize]) -> f64 {
      x.into_iter().enumerate().map(|(i, x)| {
          2. * (*x as f64) / 2f64.powi(i as i32)
      }).sum()
  }
  #[inline]
  fn target_function<T: ComplexFloat>(x: &[usize]) -> T {
      let arg = idx_to_arg(x);
      T::from(((arg - 2.512345678).powi(2) + 1e-3).ln() * (15. * (arg - 2.512345678)).cos()).unwrap()
  }

  #[inline]
  fn _test_argmax_modulo<T>(
    trunc_acc: T::Real,
    maxvol_termination: T::Real,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let modes = vec![2; 20];
    let (tt, _) = TTVec::<T>::ttcross(&modes, 20, maxvol_termination, |x| target_function(x), 4, false).unwrap();
    let argmax = tt.argmax_modulo(trunc_acc, 0, 0, 10).unwrap();
    assert!((idx_to_arg(&argmax) - 2.512345678).abs() < 1e-5)
  }

  #[test]
  fn test_optima_tt_max() {
    _test_argmax_modulo::<f64>(1e-8, 1e-5);
    _test_argmax_modulo::<Complex64>(1e-8, 1e-5);
  }

  #[inline]
  fn _test_reduced_sum<T>(mask: [bool; 25], acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let mode_dims = vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 4, 5, 6, 7, 6, 5, 4];
    let reduced_mode_dims: Vec<_> = mode_dims.iter().zip(mask).filter(|(_, p)| *p).map(|(x, _)| *x).collect();
    let tt = TTVec::<T>::new_random(vec![2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 4, 5, 6, 7, 6, 5, 4], 35);
    let reduced_tt = tt.reduced_sum(&mask).unwrap();
    assert_eq!(reduced_tt.get_mode_dims(), &reduced_mode_dims);
    let (reduced_log_abs, reduced_phase) = reduced_tt.log_sum().unwrap();
    let (log_abs, phase) = tt.log_sum().unwrap();
    assert!((reduced_log_abs - log_abs).abs() < acc);
    assert!((phase - reduced_phase).abs() < acc);
  }

  #[test]
  fn test_reduced_sum() {
    let mask = [
      true, true, true, false, false, false, true, true, true, false,
      false, true, true, false, false, false, true, true, true, false,
      false, true, false, false, true,
    ];
    _test_reduced_sum::<f32>(      mask, 1e-4 );
    _test_reduced_sum::<f64>(      mask, 1e-10);
    _test_reduced_sum::<Complex32>(mask, 1e-4 );
    _test_reduced_sum::<Complex64>(mask, 1e-10);
    let mask = [
      false, false, true, false, false, false, true, true, true, false,
      false, true, true, false, false, false, true, true, true, false,
      false, true, false, false, true,
    ];
    _test_reduced_sum::<f32>(      mask, 1e-4 );
    _test_reduced_sum::<f64>(      mask, 1e-10);
    _test_reduced_sum::<Complex32>(mask, 1e-4 );
    _test_reduced_sum::<Complex64>(mask, 1e-10);
    let mask = [
      true, true, true, false, false, false, true, true, true, false,
      false, true, true, false, false, false, true, true, true, false,
      false, true, false, false, false,
    ];
    _test_reduced_sum::<f32>(      mask, 1e-4 );
    _test_reduced_sum::<f64>(      mask, 1e-10);
    _test_reduced_sum::<Complex32>(mask, 1e-4 );
    _test_reduced_sum::<Complex64>(mask, 1e-10);
    let mask = [
      false, false, true, false, false, false, true, true, true, false,
      false, true, true, false, false, false, true, true, true, false,
      false, true, false, false, false,
    ];
    _test_reduced_sum::<f32>(      mask, 1e-4 );
    _test_reduced_sum::<f64>(      mask, 1e-10);
    _test_reduced_sum::<Complex32>(mask, 1e-4 );
    _test_reduced_sum::<Complex64>(mask, 1e-10);
  }
}
