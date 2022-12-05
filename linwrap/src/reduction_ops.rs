use std::iter::Sum;
use num_complex::ComplexFloat;
use rayon::prelude::ParallelIterator;

use crate::Matrix;
use crate::par_ptr_wrapper::{
  PointerExtWithDerefAndSend,
};

// ---------------------------------------------------------------------- //
// TODO: add argmax, max wrt absolute value
// TODO: add prod

impl<'a, Ptr, Ref> Matrix<Ptr, Ref>
where
  Ptr: PointerExtWithDerefAndSend<'a>,
  Ptr::Target: ComplexFloat + Send + Sync + 'a,
  <Ptr::Target as ComplexFloat>::Real: Sum + Send + Sync,
{
  pub fn norm_n_pow_n(&self, n: usize) -> <Ptr::Target as ComplexFloat>::Real {
    self.into_par_iter()
      .map(|x| { x.abs().powi(n as i32) })
      .sum::<<Ptr::Target as ComplexFloat>::Real>()
  }
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::Matrix;
  #[test]
  fn test_norm() {
    let buff = (0..10000).map(|x| x as f64).collect::<Vec<_>>();
    let m = Matrix::from_slice(&buff, 20, 500).unwrap();
    let m_norm_2_pow_2 = m.norm_n_pow_n(2);
    let true_m_norm_2_pow_2: f64 = buff.into_iter().map(|x| x.abs().powi(2)).sum();
    assert_eq!(m_norm_2_pow_2, true_m_norm_2_pow_2);
  }
}