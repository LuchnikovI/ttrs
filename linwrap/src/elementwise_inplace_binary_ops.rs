use std::iter::Sum;
use num_complex::ComplexFloat;
/*use rayon::prelude::{
  ParallelIterator,
  IndexedParallelIterator,
};*/

use crate::{
  NDArray,
  ndarray::{
    NDArrayResult,
    NDArrayError,
  },
};

// ---------------------------------------------------------------------- //

// TODO: complete binary operations to the full list
// TODO: make testing universal

// ---------------------------------------------------------------------- //

macro_rules! elementwise_bin_fn {
  ($fn_name:ident, $body:expr) => {
    /// This method perform a binary operation and writes a result into self.
    /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
    /// as for raw pointers.
    #[inline]
    pub unsafe fn $fn_name(
      self,
      lhs: impl Into<NDArray<*const T, N>>,
      rhs: impl Into<NDArray<*const T, N>>,
    ) -> NDArrayResult<()>
    {
      let mut lhs: NDArray<*const T, N> = lhs.into();
      let mut rhs: NDArray<*const T, N> = rhs.into();
      let iter = lhs.strides.iter_mut().zip(lhs.shape.iter_mut())
        .zip(rhs.strides.iter_mut().zip(rhs.shape.iter_mut()))
        .zip(self.shape.into_iter());
      for (((lhs_st, lhs_sh), (rhs_st, rhs_sh)), self_sh) in iter {
        if *lhs_sh == 1 {
          *lhs_sh = self_sh;
          *lhs_st = 0;
        } else if *lhs_sh != self_sh {
          return Err(NDArrayError::BroadcastingError(Box::new(lhs.shape), Box::new(self.shape)));
        }
        if *rhs_sh == 1 {
          *rhs_sh = self_sh;
          *rhs_st = 0;
        } else if *rhs_sh != self_sh {
          return Err(NDArrayError::BroadcastingError(Box::new(rhs.shape), Box::new(self.shape)));
        }
      }
      self.into_f_iter().zip(rhs.into_f_iter().zip(lhs.into_f_iter())).for_each($body);
      Ok(())
    }
  };
}

impl<T, const N: usize> NDArray<*mut T, N>
where
  T: ComplexFloat + Send + Sync + 'static,
  <T as ComplexFloat>::Real: Sum + Send + Sync,
{
  elementwise_bin_fn!(add, |(dst, (x, y))| *dst.0 = *x.0 + *y.0);
  elementwise_bin_fn!(sub, |(dst, (x, y))| *dst.0 = *x.0 - *y.0);
  elementwise_bin_fn!(mul, |(dst, (x, y))| *dst.0 = *x.0 * *y.0);
  elementwise_bin_fn!(div, |(dst, (x, y))| *dst.0 = *x.0 / *y.0);
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::NDArray;
  #[test]
  fn test_bin_ops() {
    let buff1: Vec<_> = (0..27).map(|x| x as f64).collect();
    let buff2: Vec<_> = (20..36).map(|x| x as f64).collect();
    let mut buff3: Vec<f64> = Vec::with_capacity(27 * 36);
    unsafe { buff3.set_len(27 * 16) };
    let arr1 = NDArray::from_slice(&buff1, [9, 1, 3, 1]).unwrap();
    let arr2 = NDArray::from_slice(&buff2, [1, 4, 1, 4]).unwrap();
    let arr3 = NDArray::from_mut_slice(&mut buff3, [9, 4, 3, 4]).unwrap();
    unsafe { arr3.mul(arr1, arr2).unwrap() };
    let (buff3, _) = unsafe { arr3.transpose([0, 2, 1, 3]).unwrap().gen_f_array() };
    let buff4: Vec<_> = buff2.into_iter().flat_map(|x| buff1.iter().map(move |y| {x * *y})).collect();
    assert_eq!(buff4, buff3);
  }
}