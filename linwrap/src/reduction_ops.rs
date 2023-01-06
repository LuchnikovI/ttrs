use std::iter::Sum;
use num_complex::ComplexFloat;
//use rayon::prelude::{ParallelIterator, IntoParallelIterator};

use crate::NDArray;

// ---------------------------------------------------------------------- //
// TODO: find a universal API for reduction operation and
// implement NumPy like reductions for all meaningful operations

macro_rules! reduction_op {
  ($ptr_type:ident) => {
    impl<T, const N: usize> NDArray<*$ptr_type T, N>
    where
      T: ComplexFloat + Send + Sync + 'static,
      <T as ComplexFloat>::Real: Sum + Send + Sync + PartialOrd,
    {
      pub unsafe fn norm_n_pow_n(self, n: usize) -> <T as ComplexFloat>::Real {
        self.into_cache_friendly_iter()
          .map(|x| { (*x.0).abs().powi(n as i32) })
          .sum::<<T as ComplexFloat>::Real>()
      }
      pub unsafe fn argmax(self) -> (T, [usize; N])
      {
        let (mut idx, val) = self.into_f_iter().enumerate()
          .max_by(|(_, lhs), (_, rhs)| {
            (*lhs.0).abs().partial_cmp(&(*rhs.0).abs()).unwrap()
          }).unwrap();
        let mut index = [0; N];
        for (i, s) in index.iter_mut().zip(self.shape) {
          *i = idx % s;
          idx /= s;
        }
        (*val.0, index)
      }
    }
  };
}

reduction_op!(mut);
reduction_op!(const);

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::NDArray;
  #[test]
  fn test_norm() {
    let buff = (0..10000).map(|x| x as f64).collect::<Vec<_>>();
    let m = NDArray::from_slice(&buff, [20, 25, 2, 10]).unwrap();
    let m_norm_2_pow_2 = unsafe { m.norm_n_pow_n(2) };
    let true_m_norm_2_pow_2: f64 = buff.into_iter().map(|x| x.abs().powi(2)).sum();
    assert_eq!(m_norm_2_pow_2, true_m_norm_2_pow_2);
  }

  fn _test_argmax(index: [usize; 4]) {
    let mut buff: Vec<_> = (0..256).map(|x| x as f64).collect();
    let m = NDArray::from_mut_slice(&mut buff, [8, 4, 2, 4]).unwrap();
    unsafe { *m.at(index).unwrap() = 1e5;}
    let (val, found_index) = unsafe { m.argmax() };
    assert_eq!(val, 1e5);
    assert_eq!(index, found_index)
  }

  #[test]
  fn test_argmax() {
    _test_argmax([0, 0, 0, 0]);
    _test_argmax([4, 2, 1, 3]);
    _test_argmax([7, 3, 1, 3]);
  }
}