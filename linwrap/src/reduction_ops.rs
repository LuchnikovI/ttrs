use std::iter::Sum;
use num_complex::ComplexFloat;
use rayon::prelude::{ParallelIterator, IntoParallelIterator};

use crate::Matrix;

// ---------------------------------------------------------------------- //
// TODO: add argmax, max wrt absolute value
// TODO?: add prod

macro_rules! reduction_op {
  ($ptr_type:ident) => {
    impl<T> Matrix<*$ptr_type T>
    where
      T: ComplexFloat + Send + Sync,
      <T as ComplexFloat>::Real: Sum + Send + Sync + PartialOrd,
    {
      pub unsafe fn norm_n_pow_n(self, n: usize) -> <T as ComplexFloat>::Real {
        self.into_par_iter()
          .map(|x| { (*x.0).abs().powi(n as i32) })
          .sum::<<T as ComplexFloat>::Real>()
      }
      pub unsafe fn argmax(self) -> (T, usize, usize)
      {
        let (idx, val) = (0..(self.ncols * self.nrows), self.into_par_iter()).into_par_iter()
          .max_by(|(_, lhs), (_, rhs)| {
            (*lhs.0).abs().partial_cmp(&(*rhs.0).abs()).unwrap()
          }).unwrap();
        (*val.0, idx % self.nrows, idx / self.nrows)
      }
    }
  };
}

reduction_op!(mut);
reduction_op!(const);

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::Matrix;
  #[test]
  fn test_norm() {
    let buff = (0..10000).map(|x| x as f64).collect::<Vec<_>>();
    let m = Matrix::from_slice(&buff, 20, 500).unwrap();
    let m_norm_2_pow_2 = unsafe { m.norm_n_pow_n(2) };
    let true_m_norm_2_pow_2: f64 = buff.into_iter().map(|x| x.abs().powi(2)).sum();
    assert_eq!(m_norm_2_pow_2, true_m_norm_2_pow_2);
  }
  #[test]
  fn test_argmax() {
    let buff = vec![
      0., 1., 2., 4.,
      3., 4., 5., 2.,
      6., 1., 7., 8.,
      8., 8., 10.,7.,
      5., 4., 3., 1.,
      2., 1., 3., 0.,
    ];
    let m = Matrix::from_slice(&buff, 4, 6).unwrap();
    let (val, row_num, col_num) = unsafe { m.argmax() };
    assert_eq!(val, 10.);
    assert_eq!(row_num, 2);
    assert_eq!(col_num, 3);
  }
}