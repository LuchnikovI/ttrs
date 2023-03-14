use std::iter::Sum;
use num_complex::ComplexFloat;
//use rayon::prelude::{ParallelIterator, IntoParallelIterator};

use crate::NDArray;
use crate::NDArrayError;
use crate::ndarray::NDArrayResult;

// ---------------------------------------------------------------------- //
// TODO: implement reductions for all meaningful operations

#[inline]
fn max_abs<T: ComplexFloat>(lhs: T, rhs: T) -> T {
  let lhs = lhs.abs();
  let rhs = rhs.abs();
  if lhs > rhs {
    T::from(lhs).unwrap()
  } else {
    T::from(rhs).unwrap()
  }
}

macro_rules! reduction_ops_to_scalar {
  ($ptr_type:ident) => {
    impl<T, const N: usize> NDArray<*$ptr_type T, N>
    where
      T: ComplexFloat + Sum + Send + Sync + 'static,
      <T as ComplexFloat>::Real: Send + Sync + PartialOrd,
    {
      pub unsafe fn norm_n_pow_n(self, n: usize) -> T {
        self.into_cache_friendly_iter()
          .map(|x| { (*x.0 * (*x.0).conj()).sqrt().powi(n as i32) })
          .sum::<T>()
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

reduction_ops_to_scalar!(mut);
reduction_ops_to_scalar!(const);

// TODO: make it more cache friendly?
macro_rules! reduction_ops_to_array {
  ($ptr_type:ident, $fn_name:ident, $body:expr) => {
    impl<T, const N: usize> NDArray<*$ptr_type T, N>
    where
      T: ComplexFloat + Send + Sync + 'static,
      <T as ComplexFloat>::Real: Sum + Send + Sync + PartialOrd,
    {
      /// This method perform reduction operation over self and write result to the dst array.
      /// It follows rules similar to those for broadcasting: if a mode of dst has dimension 1,
      /// then this mode is being reduced. It also returns a broadcasting error, when this rule is
      /// violated.
      /// Note, that dst must be initialized by 'zero' for a particular reduction operation.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn $fn_name(
        self,
        mut dst: NDArray<*mut T, N>,
      ) -> NDArrayResult<()>
      {
        let iter = self.shape.into_iter().zip(dst.strides.iter_mut().zip(dst.shape.iter_mut()));
        for (self_sh, (dst_st, dst_sh)) in iter {
          if *dst_sh == 1 {
            *dst_st = 0;
            *dst_sh = self_sh;
          } else if (*dst_sh != self_sh) {
            return Err(NDArrayError::BroadcastingError(Box::new(self.shape), Box::new(dst.shape)));
          }
        }
        dst.into_f_iter().zip(self.into_f_iter()).for_each(|(d, s)| {
          $body(d.0, s.0);
        });
        Ok(())
      }
    }
  };
}

reduction_ops_to_array!(mut,   reduce_add,     |x: *mut T, y: *const T| *x = *x + *y        );
reduction_ops_to_array!(const, reduce_add,     |x: *mut T, y: *const T| *x = *x + *y        );
reduction_ops_to_array!(mut,   reduce_prod,    |x: *mut T, y: *const T| *x = *x * *y        );
reduction_ops_to_array!(const, reduce_prod,    |x: *mut T, y: *const T| *x = *x * *y        );
reduction_ops_to_array!(mut,   reduce_abs_max, |x: *mut T, y: *const T| *x = max_abs(*x, *y));
reduction_ops_to_array!(const, reduce_abs_max, |x: *mut T, y: *const T| *x = max_abs(*x, *y));

macro_rules! trace {
  ($ptr_type:ident) => {
    impl<T> NDArray<*$ptr_type T, 2>
    where
      T: ComplexFloat + Sum + Send + Sync + 'static,
      <T as ComplexFloat>::Real: Send + Sync + PartialOrd,
    {
      pub unsafe fn trace(&self) -> T {
        let mut t = T::zero();
        let s1 = self.strides[0];
        let s2 = self.strides[1];
        let m = self.shape[0];
        let n = self.shape[1];
        for i in 0..(std::cmp::min(m, n)) {
          t = t + *self.ptr.add(s1 * i + s2 * i);
        }
        t
      }
    }
  }
}

trace!(mut  );
trace!(const);

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::NDArray;
  use crate::init_utils::BufferGenerator;
  use ndarray::Array4;
  use ndarray::Axis;

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

  #[test]
  fn test_reduce_add() {
    let buff = (0..256).map(|x| x as f64).collect::<Vec<_>>();
    let mut dst_buff = vec![0.; 16];
    let arr1 = NDArray::from_slice(&buff, [4, 8, 4, 2]).unwrap();
    let dst_arr = NDArray::from_mut_slice(&mut dst_buff, [4, 1, 4, 1]).unwrap();
    let arr2 = Array4::from_shape_vec((2, 4, 8, 4), buff).unwrap();
    let arr2 = arr2.sum_axis(Axis(0)).sum_axis(Axis(1));
    unsafe { arr1.reduce_add(dst_arr).unwrap() };
    let is_eq = arr2.iter().zip(unsafe { dst_arr.into_f_iter() }).all(|(lhs, rhs)| {
      *lhs == unsafe { *rhs.0 }
    });
    assert!(is_eq);
  }

  #[test]
  fn test_reduce_abs_max() {
    let buff = f64::random_normal(256);
    let mut dst_buff = vec![0.; 16];
    let arr1 = NDArray::from_slice(&buff, [4, 8, 4, 2]).unwrap();
    let dst_arr = NDArray::from_mut_slice(&mut dst_buff, [4, 1, 4, 1]).unwrap();
    let arr2 = Array4::from_shape_vec((2, 4, 8, 4), buff).unwrap();
    let arr2 = arr2
      .map_axis(Axis(0), |x| (*x.iter().max_by(|l, r| l.abs().partial_cmp(&r.abs()).unwrap()).unwrap()).abs())
      .map_axis(Axis(1), |x| (*x.iter().max_by(|l, r| l.abs().partial_cmp(&r.abs()).unwrap()).unwrap()).abs());
    unsafe { arr1.reduce_abs_max(dst_arr).unwrap() };
    let is_eq = arr2.iter().zip(unsafe { dst_arr.into_f_iter() }).all(|(lhs, rhs)| {
      *lhs == unsafe { *rhs.0 }
    });
    assert!(is_eq);
  }
}
