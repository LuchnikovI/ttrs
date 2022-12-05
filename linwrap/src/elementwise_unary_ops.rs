use num_complex::ComplexFloat;
use rayon::prelude::ParallelIterator;

use crate::Matrix;

// ---------------------------------------------------------------------- //

// TODO: complete unary operations to the full list

// ---------------------------------------------------------------------- //

impl<T> Matrix<*mut T, &mut [T]>
where
  T: ComplexFloat + Send + Sync,
{
  pub fn conj(&mut self) {
    self.into_par_iter_mut().for_each(|x| *x = x.conj());
  }
  pub fn scale(&mut self, other: T)
  {
    self.into_par_iter_mut().for_each(|x| *x = *x * other);
  }
}