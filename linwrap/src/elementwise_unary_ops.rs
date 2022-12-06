use num_complex::ComplexFloat;
use rayon::prelude::ParallelIterator;

use crate::Matrix;

// ---------------------------------------------------------------------- //

macro_rules! elementwise_fn {
  ($fn_name:ident, $body:expr) => {
    pub fn $fn_name(&mut self)
    {
      self.into_par_iter_mut().for_each($body);
    }
  };
}

impl<T> Matrix<*mut T, &mut [T]>
where
  T: ComplexFloat + Send + Sync,
{
  elementwise_fn!(conj,  |x| *x = x.conj());
  elementwise_fn!(sin,   |x| *x = x.sin());
  elementwise_fn!(cos,   |x| *x = x.cos());
  elementwise_fn!(sqrt,  |x| *x = x.sqrt());
  elementwise_fn!(tan,   |x| *x = x.tan());
  elementwise_fn!(acos,  |x| *x = x.acos());
  elementwise_fn!(asin,  |x| *x = x.asin());
  elementwise_fn!(atan,  |x| *x = x.atan());
  elementwise_fn!(exp,   |x| *x = x.exp());
  elementwise_fn!(abs,   |x| *x = T::from(x.abs()).unwrap());
  elementwise_fn!(log,   |x| *x = x.ln());
  elementwise_fn!(log2,  |x| *x = x.log2());
  elementwise_fn!(log10, |x| *x = x.log10());
  pub fn mul_by_scalar(&mut self, other: T)
  {
    self.into_par_iter_mut().for_each(|x| *x = *x * other);
  }
  pub fn add_scalar(&mut self, other: T)
  {
    self.into_par_iter_mut().for_each(|x| *x = *x + other);
  }
  pub fn pow(&mut self, other: T::Real)
  where
    T::Real: Send + Sync,
  {
    self.into_par_iter_mut().for_each(|x| *x = x.powf(other));
  }
}