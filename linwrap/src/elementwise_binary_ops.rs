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
    /// This method performs a binary operation inplace.
    /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
    /// as for raw pointers.
    #[inline]
    pub unsafe fn $fn_name(
      self,
      other: impl Into<NDArray<*const T, N>>,
    ) -> NDArrayResult<()>
    {
      let mut other: NDArray<*const T, N> = other.into();
      let iter = other.strides.iter_mut().zip(other.shape.iter_mut()).zip(self.shape.into_iter());
      for ((other_st, other_sh), self_sh) in iter {
        if *other_sh == 1 {
          *other_sh = self_sh;
          *other_st = 0;
        } else if *other_sh != self_sh {
          return Err(NDArrayError::BroadcastingError(Box::new(other.shape), Box::new(self.shape)));
        }
      }
      self.into_f_iter().zip(other.into_f_iter()).for_each($body);
      Ok(())
    }
  };
}

impl<T, const N: usize> NDArray<*mut T, N>
where
  T: ComplexFloat + Send + Sync + 'static,
  <T as ComplexFloat>::Real: Sum + Send + Sync,
{
  elementwise_bin_fn!(add_inpl, |(x, y)| *x.0 = *x.0 + *y.0);
  elementwise_bin_fn!(sub_inpl, |(x, y)| *x.0 = *x.0 - *y.0);
  elementwise_bin_fn!(mul_inpl, |(x, y)| *x.0 = *x.0 * *y.0);
  elementwise_bin_fn!(div_inpl, |(x, y)| *x.0 = *x.0 / *y.0);
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::NDArray;
  #[test]
  fn test_bin_ops() {
    let buff1: Vec<_> = (0..27).map(|x| x as f64).collect();
    let buff2: Vec<_> = (20..36).map(|x| x as f64).collect();
    let mut buff3 = vec![1f64; 27 * 16];
    let arr1 = NDArray::from_slice(&buff1, [9, 1, 3, 1]).unwrap();
    let arr2 = NDArray::from_slice(&buff2, [1, 4, 1, 4]).unwrap();
    let arr3 = NDArray::from_mut_slice(&mut buff3, [9, 4, 3, 4]).unwrap();
    unsafe { arr3.mul_inpl(arr1).unwrap() };
    unsafe { arr3.mul_inpl(arr2).unwrap() };
    let (buff3, _) = unsafe { arr3.transpose([0, 2, 1, 3]).unwrap().gen_f_array() };
    let buff4: Vec<_> = buff2.into_iter().flat_map(|x| buff1.iter().map(move |y| {x * *y})).collect();
    assert_eq!(buff4, buff3);
  }
}