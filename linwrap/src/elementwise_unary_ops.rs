use num_complex::ComplexFloat;
//use rayon::prelude::ParallelIterator;

use crate::NDArray;

// ---------------------------------------------------------------------- //

macro_rules! elementwise_fn {
  ($fn_name:ident, $body:expr) => {
    #[inline]
    pub unsafe fn $fn_name(self)
    {
      self.into_cache_friendly_iter().for_each($body);
    }
  };
}

impl<T, const N: usize> NDArray<*mut T, N>
where
  T: ComplexFloat + Send + Sync + 'static,
{
  elementwise_fn!(conj,  |x| *x.0 = (*x.0).conj());
  elementwise_fn!(sin,   |x| *x.0 = (*x.0).sin());
  elementwise_fn!(cos,   |x| *x.0 = (*x.0).cos());
  elementwise_fn!(sqrt,  |x| *x.0 = (*x.0).sqrt());
  elementwise_fn!(tan,   |x| *x.0 = (*x.0).tan());
  elementwise_fn!(acos,  |x| *x.0 = (*x.0).acos());
  elementwise_fn!(asin,  |x| *x.0 = (*x.0).asin());
  elementwise_fn!(atan,  |x| *x.0 = (*x.0).atan());
  elementwise_fn!(exp,   |x| *x.0 = (*x.0).exp());
  elementwise_fn!(abs,   |x| *x.0 = T::from((*x.0).abs()).unwrap());
  elementwise_fn!(log,   |x| *x.0 = (*x.0).ln());
  elementwise_fn!(log2,  |x| *x.0 = (*x.0).log2());
  elementwise_fn!(log10, |x| *x.0 = (*x.0).log10());
  pub unsafe fn mul_by_scalar(self, other: T)
  {
    self.into_cache_friendly_iter().for_each(|x| *x.0 = *x.0 * other);
  }
  pub unsafe fn add_scalar(self, other: T)
  {
    self.into_cache_friendly_iter().for_each(|x| *x.0 = *x.0 + other);
  }
  pub unsafe fn pow(self, other: T::Real)
  where
    T::Real: Send + Sync,
  {
    self.into_cache_friendly_iter().for_each(|x| *x.0 = (*x.0).powf(other));
  }
}

#[cfg(test)]
mod tests {
  use crate::{init_utils::random_normal_c64, NDArray};
  macro_rules! test_elementwise_ops {
    ($fn_name:ident) => {
      let mut buff = random_normal_c64(256);
      let mut buff_copy = buff.clone();
      let arr = NDArray::from_mut_slice(&mut buff, [8, 4, 8]).unwrap().transpose([1, 2, 0]).unwrap();
      unsafe { arr.$fn_name() };
      buff_copy.iter_mut().for_each(|x| { *x = (*x).$fn_name() });
      assert_eq!(buff, buff_copy);
    };
  }
  #[test]
  fn test_elementwise_ops() {
    test_elementwise_ops!(conj);
    test_elementwise_ops!(sin);
    test_elementwise_ops!(cos);
    test_elementwise_ops!(sqrt);
    test_elementwise_ops!(tan);
    test_elementwise_ops!(acos);
    test_elementwise_ops!(asin);
    test_elementwise_ops!(atan);
    test_elementwise_ops!(exp);
    test_elementwise_ops!(log2);
    test_elementwise_ops!(log10);
  }
}