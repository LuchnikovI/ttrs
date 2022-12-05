use std::{
  marker::{
    PhantomData,
    Send,
  },
  fmt::Debug, ffi::c_int,
};

use rawpointer::PointerExt;
use rayon::prelude::{
  IntoParallelIterator,
  ParallelIterator,
  IndexedParallelIterator,
};

use crate::par_ptr_wrapper::{
  PointerExtWithDerefAndSend,
  PointerExtWithDerefMutAndSend,
};

// ---------------------------------------------------------------------- //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MatrixError {
  ImpossibleToReshape,
  IncorrectSize,
  OutOfBound,
  IncorrectShape,
  LapackError(c_int),
  FortranLayoutRequired,
}

pub type MatrixResult<T> = Result<T, MatrixError>;

// ---------------------------------------------------------------------- //

// TODO: documentation
// TODO: proper indexing
// TODO: proper Debug trait implementation (reduce number of extra allocations,
//  improve alignment and representation of large matrices)
// TODO: it allows to initialize matrix from temp. Vec ==> segfault; Fix it.

// ---------------------------------------------------------------------- //

#[derive(Clone, Copy)]
pub struct Matrix<Ptr, Ref>
where
  Ptr: PointerExt,
{
  pub(super) ptr: Ptr,
  pub(super) ncols: usize,
  pub(super) nrows: usize,
  pub(super) stride1: usize,
  pub(super) stride2: usize,
  pub(super) is_init: bool,
  pub(super) pref: PhantomData<Ref>,
}

impl<Ptr, Ref> Matrix<Ptr, Ref>
where
  Ptr: PointerExt,
{
  pub fn col_split(
    self,
    ncols: usize
  ) -> MatrixResult<(Matrix<Ptr, Ref>, Matrix<Ptr, Ref>)>
  {
    if ncols > self.ncols { return Err(MatrixError::OutOfBound); }
    let lhs_ptr = self.ptr;
    let rhs_ptr = unsafe { self.ptr.add(self.stride2 * ncols) };
    let lhs_ncols = ncols;
    let rhs_ncols = self.ncols - ncols;
    Ok((
      Matrix { ptr: lhs_ptr, ncols: lhs_ncols, nrows: self.nrows,
               stride1: self.stride1, stride2: self.stride2,
               is_init: self.is_init, pref: PhantomData },
      Matrix { ptr: rhs_ptr, ncols: rhs_ncols, nrows: self.nrows,
               stride1: self.stride1, stride2: self.stride2,
               is_init: self.is_init, pref: PhantomData },
    ))
  }

  pub fn row_split(
    self,
    nrows: usize
  ) -> MatrixResult<(Matrix<Ptr, Ref>, Matrix<Ptr, Ref>)>
  {
    if nrows > self.nrows { return Err(MatrixError::OutOfBound); }
    let lhs_ptr = self.ptr;
    let rhs_ptr = unsafe { self.ptr.add(self.stride1 * nrows) };
    let lhs_nrows = nrows;
    let rhs_nrows = self.nrows - nrows;
    Ok((
      Matrix { ptr: lhs_ptr, ncols: self.ncols, nrows: lhs_nrows,
               stride1: self.stride1, stride2: self.stride2,
               is_init: self.is_init, pref: PhantomData },
      Matrix { ptr: rhs_ptr, ncols: self.ncols, nrows: rhs_nrows,
               stride1: self.stride1, stride2: self.stride2,
               is_init: self.is_init, pref: PhantomData },
    ))
  }

  pub fn reshape(
    mut self,
    nrows: usize,
    ncols: usize,
  ) -> MatrixResult<Matrix<Ptr, Ref>>
  {
    let new_size = ncols * nrows;
    let old_size = self.ncols * self.nrows;
    if old_size != new_size { return Err(MatrixError::IncorrectSize); }
    if (self.stride1 == 1) && (self.stride2 == self.nrows) {
      self.stride2 = nrows;
      self.nrows = nrows;
      self.ncols = ncols;
      Ok(self)
    }
    else if (self.stride1 == self.ncols) && (self.stride2 == 1) {
      self.stride1 = ncols;
      self.nrows = nrows;
      self.ncols = ncols;
      Ok(self)
    }
    else
    {
      Err(MatrixError::ImpossibleToReshape)
    }
  }

  pub fn into_par_iter<'a, T>(&self) -> impl IndexedParallelIterator<Item = &'a T> + Clone
  where
    Ptr: PointerExtWithDerefAndSend<'a, Target = T>,
    T: Send + Sync + 'a,
  {
    let ptr = self.ptr.wrap();
    let ncols = self.ncols;
    let nrows = self.nrows;
    let stride1 = self.stride1;
    let stride2 = self.stride2;
    (0..(nrows * ncols)).into_par_iter().map(move |i|{
      unsafe { ptr.add((i / nrows) * stride2 + (i % nrows) * stride1).deref() }
    })
  }

  pub fn into_par_iter_mut<'a, T>(&mut self) -> impl IndexedParallelIterator<Item = &'a mut T> + Clone
  where
    Ptr: PointerExtWithDerefMutAndSend<'a, Target = T>,
    T: Send + Sync + 'a,
  {
    let ptr = self.ptr.wrap();
    let ncols = self.ncols;
    let nrows = self.nrows;
    let stride1 = self.stride1;
    let stride2 = self.stride2;
    (0..(nrows * ncols)).into_par_iter().map(move |i|{
      unsafe { ptr.add((i / nrows) * stride2 + (i % nrows) * stride1).deref_mut() }
    })
  }
}

impl<T> Matrix<*const T, &[T]>
{
  pub fn from_slice(slice: &[T], nrows: usize, ncols: usize) -> MatrixResult<Matrix<*const T, &[T]>> {
    let m: Matrix<_, _> = slice.into();
    m.reshape(nrows, ncols)
  }
}

impl<T> Matrix<*mut T, &mut [T]>
{
  pub fn from_mut_slice(slice: &mut [T], nrows: usize, ncols: usize) -> MatrixResult<Matrix<*mut T, &mut [T]>> {
    let m: Matrix<_, _> = slice.into();
    m.reshape(nrows, ncols)
  }
}

impl<'a, T> From<&'a[T]> for Matrix<*const T, &'a [T]>
{
  fn from(buff: &'a [T]) -> Self {
    let ptr = buff.as_ptr();
    let len = buff.as_ref().len();
    Self { ptr, ncols: 1, nrows: len, stride1: 1, stride2: len, is_init: true, pref: PhantomData }
  }
}

impl<'a, T> From<&'a mut [T]> for Matrix<*mut T, &'a mut[T]>
{

  fn from(buff: &'a mut [T]) -> Self {
    let ptr = buff.as_mut_ptr();
    let len = buff.as_ref().len();
    Self { ptr, ncols: 1, nrows: len, stride1: 1, stride2: len, is_init: true, pref: PhantomData }
  }
}

impl<'a, Ptr, Ref> Debug for Matrix<Ptr, Ref>
where
  Ptr: PointerExtWithDerefAndSend<'a> + 'a,
  Ptr::Target: Debug,
{

  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if !self.is_init { return write!(f, "Matrix is not initialized."); }
    let stride1 = self.stride1;
    let stride2 = self.stride2;
    let mut msg = String::with_capacity(((3 * self.ncols + 3) * self.nrows) + 2);
    for i in 0..self.nrows {
      msg.push('[');
      for j in 0..self.ncols{
        msg.push_str(&format!("{:?}, ", unsafe { self.ptr.add(j * stride2 + i * stride1).deref() }));
      }
      msg.push_str("]\n");
    }
    write!(f, "{}", msg)
  }
}

impl<'a, Ptr, Ref> PartialEq for Matrix<Ptr, Ref>
where
  Ptr: PointerExtWithDerefAndSend<'a> + 'a + Clone + Copy,
  Ptr::Target: Debug + Send + Sync + PartialEq,
  Ref: Clone + Copy,
{

  fn eq(&self, other: &Self) -> bool {
    let params_eq = (self.ncols == other.ncols) &&
                    (self.nrows == other.nrows) &&
                    self.is_init && other.is_init;
    self.clone().into_par_iter().zip(other.into_par_iter())
      .all(|(lhs, rhs)| {
        lhs == rhs
      }) && params_eq
  }
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_split() {
    let buff = (0..256).collect::<Vec<i32>>();
    let m = Matrix::from_slice(&buff, 16, 16).unwrap()
             .col_split(4).unwrap()
             .1
             .col_split(4).unwrap()
             .0
             .row_split(3).unwrap()
             .1
             .row_split(5).unwrap()
             .0;
    let m_true: Matrix<_, _> = [
      67, 68, 69, 70, 71,
      83, 84, 85, 86, 87,
      99, 100, 101, 102, 103,
      115, 116, 117, 118, 119,
    ].as_slice().into();
    let m_true = m_true.reshape(5, 4).unwrap();
    assert_eq!(m_true, m)
  }
}
