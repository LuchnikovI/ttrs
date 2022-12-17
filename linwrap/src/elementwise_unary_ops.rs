use num_complex::ComplexFloat;
use rayon::prelude::ParallelIterator;

use crate::Matrix;

// ---------------------------------------------------------------------- //

macro_rules! elementwise_fn {
  ($fn_name:ident, $body:expr) => {
    #[inline]
    pub unsafe fn $fn_name(self)
    {
      self.into_par_iter().for_each($body);
    }
  };
}

impl<T> Matrix<*mut T>
where
  T: ComplexFloat + Send + Sync,
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
    self.into_par_iter().for_each(|x| *x.0 = *x.0 * other);
  }
  pub unsafe fn add_scalar(self, other: T)
  {
    self.into_par_iter().for_each(|x| *x.0 = *x.0 + other);
  }
  pub unsafe fn pow(self, other: T::Real)
  where
    T::Real: Send + Sync,
  {
    self.into_par_iter().for_each(|x| *x.0 = (*x.0).powf(other));
  }
}