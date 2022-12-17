use std::{
  fmt::Debug,
  ffi::c_int,
  ops::Range,
};

use rawpointer::PointerExt;
use rayon::prelude::{
  IntoParallelIterator,
  ParallelIterator,
  IndexedParallelIterator,
};

use crate::par_ptr_wrapper::ParPtrWrapper;

// ---------------------------------------------------------------------- //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MatrixError {
  ImpossibleToReshape,
  IncorrectSize,
  OutOfBound,
  IncorrectShape,
  LapackError(c_int),
  FortranLayoutRequired,
  MutableElementsOverlapping,
  IncorrectRange,
}

pub type MatrixResult<T> = Result<T, MatrixError>;

// ---------------------------------------------------------------------- //

// TODO: documentation
// TODO: proper Debug trait implementation (reduce number of extra allocations,
//  improve alignment and representation of large matrices)
// TODO: it allows to initialize matrix from temp. Vec ==> segfault; Fix it.

// ---------------------------------------------------------------------- //

#[derive(Clone, Copy)]
pub struct Matrix<Ptr>
{
  pub(super) ptr:     Ptr,
  pub(super) ncols:   usize,
  pub(super) nrows:   usize,
  pub(super) stride1: usize,
  pub(super) stride2: usize,
}

// ---------------------------------------------------------------------- //

impl<T> From<Matrix<*mut T>> for Matrix<*const T>
{
  fn from(m: Matrix<*mut T>) -> Self {
    Self {
      ptr: m.ptr as *const T,
      ncols: m.ncols,
      nrows: m.nrows,
      stride1: m.stride1,
      stride2: m.stride2
    }
  }
}

// ---------------------------------------------------------------------- //

impl<Ptr> Matrix<Ptr>
where
  Ptr: PointerExt,
{
  pub fn col_split(
    self,
    ncols: usize,
  ) -> MatrixResult<(Self, Self)>
  {
    if ncols > self.ncols { return Err(MatrixError::OutOfBound); }
    let lhs_ptr = self.ptr;
    let rhs_ptr = unsafe { self.ptr.add(self.stride2 * ncols) };
    let lhs_ncols = ncols;
    let rhs_ncols = self.ncols - ncols;
    Ok((
      Matrix { ptr: lhs_ptr, ncols: lhs_ncols, nrows: self.nrows,
               stride1: self.stride1, stride2: self.stride2 },
      Matrix { ptr: rhs_ptr, ncols: rhs_ncols, nrows: self.nrows,
               stride1: self.stride1, stride2: self.stride2 },
    ))
  }

  pub fn row_split(
    self,
    nrows: usize
  ) -> MatrixResult<(Self, Self)>
  {
    if nrows > self.nrows { return Err(MatrixError::OutOfBound); }
    let lhs_ptr = self.ptr;
    let rhs_ptr = unsafe { self.ptr.add(self.stride1 * nrows) };
    let lhs_nrows = nrows;
    let rhs_nrows = self.nrows - nrows;
    Ok((
      Matrix { ptr: lhs_ptr, ncols: self.ncols, nrows: lhs_nrows,
               stride1: self.stride1, stride2: self.stride2 },
      Matrix { ptr: rhs_ptr, ncols: self.ncols, nrows: rhs_nrows,
               stride1: self.stride1, stride2: self.stride2 },
    ))
  }

  pub fn subview(mut self, index: (Range<usize>, Range<usize>)) -> MatrixResult<Self> {
    let (start_row, end_row) = (index.0.start, index.0.end);
    if start_row > end_row { return Err(MatrixError::IncorrectRange); }
    let (start_col, end_col) = (index.1.start, index.1.end);
    if start_col > end_col { return Err(MatrixError::IncorrectRange); }
    if end_col > self.ncols { return Err(MatrixError::OutOfBound); }
    if end_row > self.nrows { return Err(MatrixError::OutOfBound); }
    self.ptr = unsafe { self.ptr.add(start_row * self.stride1 + start_col * self.stride2) };
    self.nrows = end_row - start_row;
    self.ncols = end_col - start_col;
    Ok(self)
  }

  pub fn reshape(
    mut self,
    nrows: usize,
    ncols: usize,
  ) -> MatrixResult<Matrix<Ptr>>
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

  pub unsafe fn into_par_iter(self) -> impl IndexedParallelIterator<Item = ParPtrWrapper<Ptr>> + Clone
  {
    let ptr = ParPtrWrapper(self.ptr);
    let ncols = self.ncols;
    let nrows = self.nrows;
    let stride1 = self.stride1;
    let stride2 = self.stride2;
    (0..(nrows * ncols)).into_par_iter().map(move |i|{
      ptr.add((i / nrows) * stride2 + (i % nrows) * stride1)
    })
  }
}

// ---------------------------------------------------------------------- //

macro_rules! impl_with_deref {
  ($ptr_type:ident) => {
    impl<T: Clone + Copy + Debug + PartialEq + Send> Matrix<*$ptr_type T>
    {
      pub unsafe fn gen_buffer(self) -> Vec<T>
      {
        self.into_par_iter().map(|x| *x.0).collect()
      }

      pub unsafe fn write_to(self, other: Matrix<*mut T>) -> MatrixResult<()>
      {
        if (other.ncols != self.ncols) || (other.nrows != self.nrows) { return Err(MatrixError::IncorrectShape); }
        (other.into_par_iter(), self.into_par_iter()).into_par_iter().for_each(|(dst, src)| { *dst.0 = *src.0; });
        Ok(())
      }

      pub unsafe fn to_string(self) -> String
      {
        let stride1 = self.stride1;
        let stride2 = self.stride2;
        let mut msg = String::with_capacity(((3 * self.ncols + 3) * self.nrows) + 2);
        for i in 0..self.nrows {
          msg.push('[');
          for j in 0..self.ncols{
            msg.push_str(&format!("{:?}, ", *self.ptr.add(j * stride2 + i * stride1) ));
          }
          msg.push_str("]\n")
        }
        msg
      }

      pub unsafe fn eq(self, other: Self) -> bool
      {
        let params_eq = (self.ncols == other.ncols) &&
        (self.nrows == other.nrows);
        self.clone().into_par_iter().zip(other.into_par_iter())
          .all(|(lhs, rhs)| {
            *lhs.0 == *rhs.0
          }) && params_eq
      }

      pub unsafe fn at(self, index: (usize, usize)) -> MatrixResult<* $ptr_type T>
      {
        if index.1 > self.ncols { return Err(MatrixError::OutOfBound); }
        if index.0 > self.nrows { return Err(MatrixError::OutOfBound); }
        let ptr = unsafe { self.ptr.add(index.0 * self.stride1 + index.1 * self.stride2) };
        Ok(ptr)
      }

      pub unsafe fn gen_from_cols_order(self, order: &[usize]) -> Vec<T>
      {
        let ncols = order.len();
        let mut buff: Vec<T> = Vec::with_capacity(ncols * self.nrows);
        buff.set_len(ncols * self.nrows);
        let output_matrix_ptr = ParPtrWrapper(buff.as_mut_ptr());
        let input_matrix_ptr = ParPtrWrapper(self.ptr);
        order.into_par_iter().zip(0..ncols).for_each(|(o, i)| {
          let output_col_ptr = output_matrix_ptr.add(i * self.nrows);
          let input_col_ptr = input_matrix_ptr.add(o * self.nrows);
          for j in 0..self.nrows {
            *output_col_ptr.add(j).0 = *input_col_ptr.add(j).0;
          }
        });
        buff
      }
    }
  };
}

impl_with_deref!(mut);
impl_with_deref!(const);

// ---------------------------------------------------------------------- //

impl<T> Matrix<*const T>
{
  pub fn from_slice(slice: &[T], nrows: usize, ncols: usize) -> MatrixResult<Self> {
    let m: Matrix<_> = slice.into();
    m.reshape(nrows, ncols)
  }
}

impl<T> Matrix<*mut T>
{
  pub fn from_mut_slice(slice: &mut [T], nrows: usize, ncols: usize) -> MatrixResult<Self> {
    let m: Matrix<_> = slice.into();
    m.reshape(nrows, ncols)
  }
}

impl<T> From<&[T]> for Matrix<*const T>
{
  fn from(buff: &[T]) -> Self {
    let ptr = buff.as_ptr();
    let len = buff.as_ref().len();
    Self { ptr, ncols: 1, nrows: len, stride1: 1, stride2: len }
  }
}

impl<T> From<&mut [T]> for Matrix<*mut T>
{
  fn from(buff: &mut [T]) -> Self {
    let ptr = buff.as_mut_ptr();
    let len = buff.as_ref().len();
    Self { ptr, ncols: 1, nrows: len, stride1: 1, stride2: len }
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
    let m_true: Matrix<_> = [
      67, 68, 69, 70, 71,
      83, 84, 85, 86, 87,
      99, 100, 101, 102, 103,
      115, 116, 117, 118, 119,
    ].as_slice().into();
    let m_true = m_true.reshape(5, 4).unwrap();
    assert!(unsafe { m_true.eq(m) })
  }

  #[test]
  fn test_at() {
    let buff = (0..256).collect::<Vec<i32>>();
    let m = Matrix::from_slice(&buff, 16, 16).unwrap();
    let subm = m.subview((5..7, 1..4)).unwrap();
    let subbuff = vec![
      21, 22,
      37, 38,
      53, 54,
    ];
    let true_subm = Matrix::from_slice(&subbuff, 2, 3).unwrap();
    assert!(unsafe { true_subm.eq(subm) });
  }

  #[test]
  fn test_gen_from_cols_order() {
    let buff = (0..256).collect::<Vec<i32>>();
    let m = Matrix::from_slice(&buff, 8, 32).unwrap();
    let new_buff = unsafe { m.gen_from_cols_order(&[5, 2, 8, 1, 0, 4]) };
    let true_new_buff = [
      40, 41, 42, 43, 44, 45, 46, 47,
      16, 17, 18, 19, 20, 21, 22, 23,
      64, 65, 66, 67, 68, 69, 70, 71,
       8,  9, 10, 11, 12, 13, 14, 15,
       0,  1,  2,  3,  4,  5,  6,  7,
      32, 33, 34, 35, 36, 37, 38, 39,
    ];
    assert_eq!(&true_new_buff[..], &new_buff[..]);
  }
}
