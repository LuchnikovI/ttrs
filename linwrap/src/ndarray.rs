use std::{
  fmt::Debug,
  fmt::Display,
  ffi::c_int,
  ops::Range,
};

use rawpointer::PointerExt;
/*use rayon::prelude::{
  IntoParallelIterator,
  ParallelIterator,
  IndexedParallelIterator,
};*/

use crate::{
  par_ptr_wrapper::ParPtrWrapper,
  ndarray_utils::{
    shape_to_strides,
    get_cache_friendly_order,
  },
};

// ---------------------------------------------------------------------- //

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum Layout {
  Fortran,
  C,
  General,
}

// ---------------------------------------------------------------------- //

#[derive(PartialEq, Eq, Clone)]
pub enum NDArrayError {

  /// This error appears when the layout of an array in memory disallows
  /// reshaping without copying 
  ImpossibleToReshape,

  /// This error appears when sizes of two arrays are different while it is required them to be
  /// equal
  SizeMismatch(usize, usize),

  /// This error appears when one requires access an index that is out of array bounds
  OutOfBound(Box<[usize]>, usize, usize),

  /// This error appears when shapes of arrays do not match each other while it is required them to be
  ShapesMismatch(Box<[usize]>, Box<[usize]>),

  /// This error appears when a given shape does not match a required one
  IncorrectShape(Box<[usize]>, Box<[usize]>),

  /// This error appears when one requires a square matrix but a rectangular one is given
  SquareMatrixRequired(usize, usize),

  /// This error appears when one can not perform matrix multiplication
  /// due to the mismatch of dimensions
  MatmulDimMismatch(usize, usize),

  /// This error appears when one sends a matrix of incorrect size to the Maxvol algorithm
  MaxvolInputSizeMismatch(usize, usize),

  /// This error appears when broadcasting is impossible
  BroadcastingError(Box<[usize]>, Box<[usize]>),

  /// This error appears when it is required for array to be contiguous in memory
  /// but it is not
  NotContiguous,

  /// This error appears when one try to transpose an array with an incorrect indices order
  IncorrectIndicesOrder(Box<[usize]>),

  /// This error appears when the linear solve lapack routine ?gesv fails.
  ErrorGESV(c_int),

  /// This error appears when the lapack routine for SVD ?gesvd fails.
  ErrorGESVD(c_int),

  /// This error appears when the lapack routine for QR decomposition ?geqrf fails. 
  ErrorGEQRF(c_int),

  /// This error appears when the lapack routine for QR decomposition postprocessing ?orgqr fails. 
  ErrorORGQR(c_int),

  /// This error appears when the Fortran layout is required and 
  /// an array has a layout different to the Fortran layout
  FortranLayoutRequired,

  /// This error appears when strides of an array admit overlapping of pointers
  /// for different indices and this is forbidden in a called function (e.g. matmul)
  MutableElementsOverlapping,

  /// This error appears when one passes an incorrect range as a sub-array specification 
  IncorrectRange,
}

impl Debug for NDArrayError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      match self {
        NDArrayError::ImpossibleToReshape => { f.write_str("Layout of an array disallows reshape operation without copying.") },
        NDArrayError::SizeMismatch(lhs, rhs) => { f.write_str(&format!("Mismatch of arrays sizes: size1 = {}, size2 = {}.", lhs, rhs)) },
        NDArrayError::OutOfBound(shape, num, idx) => { f.write_str(&format!("Index {} of an array of shape {:?} at mode number {} is out of bound.", idx, &shape[..], num)) },
        NDArrayError::ShapesMismatch(shape1, shape2) => { f.write_str(&format!("Shape {:?} and shape {:?} are not equal.", &shape1[..], &shape2[..])) },
        NDArrayError::IncorrectShape(shape_given, shape_required) => { f.write_str(&format!("Given shape {:?} does not match the required one {:?}.", &shape_given[..], &shape_required[..])) },
        NDArrayError::SquareMatrixRequired(m, n) => { f.write_str(&format!("Square matrix is required, given matrix of shape {:?}.", [m, n])) },
        NDArrayError::MatmulDimMismatch(m, n) => { f.write_str(&format!("Impossible to perform matrix multiplication along indices of different dimensions. Given dimensions {}, {}.", m, n)) },
        NDArrayError::MaxvolInputSizeMismatch(m, n) => { f.write_str(&format!("Number of columns of a Maxvol algorithm input must be >= than number of rows. Given number of columns {}, number of rows {}.", n, m))},
        NDArrayError::BroadcastingError(shape1, shape2) => { f.write_str(&format!("Impossible to broadcast arrays of shapes {:?} and {:?}.", &shape1[..], &shape2[..]))},
        NDArrayError::NotContiguous => { f.write_str("Array is not contiguous in memory.")},
        NDArrayError::IncorrectIndicesOrder(order) => { f.write_str(&format!("Incorrect transposition specification {:?}.", order)) },
        NDArrayError::ErrorGESV(code) => { f.write_str(&format!("Lapack linear systems solver (?GESV) failed with code {}.", code)) },
        NDArrayError::ErrorGESVD(code) => { f.write_str(&format!("Lapack SVD routine (?GESVD) failed with code {}.", code)) },
        NDArrayError::ErrorGEQRF(code) => { f.write_str(&format!("Lapack QR decomposition routine (?GEQRF) failed with code {}.", code)) },
        NDArrayError::ErrorORGQR(code) => { f.write_str(&format!("Lapac routine for QR decomposition result postprocessing (?ORGQR) failed with code {}", code)) },
        NDArrayError::FortranLayoutRequired => { f.write_str("Array has non-Fortran layout.")},
        NDArrayError::MutableElementsOverlapping => { f.write_str("Strides and a shape allows mutable elements overlapping.") },
        NDArrayError::IncorrectRange => { f.write_str("Invalid range for sub-array specification.")},
      }
  }
}

pub type NDArrayResult<T> = Result<T, NDArrayError>;

// ---------------------------------------------------------------------- //

#[derive(Clone, Copy)]
pub struct NDArray<Ptr, const N: usize>
{
  pub(super) ptr:     Ptr,
  pub(super) shape:   [usize; N],
  pub(super) strides: [usize; N],
  pub(super) is_contiguous: bool,
  pub(super) layout: Layout,
}

// ---------------------------------------------------------------------- //

impl<T, const N: usize> From<NDArray<*mut T, N>> for NDArray<*const T, N>
{
  fn from(m: NDArray<*mut T, N>) -> Self {
    Self {
      ptr: m.ptr as *const T,
      shape: m.shape,
      strides: m.strides,
      is_contiguous: m.is_contiguous,
      layout: m.layout,
    }
  }
}

// ---------------------------------------------------------------------- //

impl<Ptr, const N: usize> NDArray<Ptr, N>
where
  Ptr: PointerExt + 'static,
{
  /// This method splits an array into two sub-arrays across a given axis.
  /// It returns two arrays: the first one is for witch the given axis range is from 0 to size,
  /// the second one is for witch the given axis range is from size to the end.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn split_across_axis(
    self,
    axis: usize,
    size: usize,
  ) -> NDArrayResult<(Self, Self)>
  {
    if size > self.shape[axis] { return Err(NDArrayError::OutOfBound(Box::new(self.shape), axis, size)); }
    let lhs_ptr = self.ptr;
    let rhs_ptr = self.ptr.add(self.strides[axis] * size);
    let mut lhs_shape = self.shape;
    lhs_shape[axis] = size;
    let mut rhs_shape = self.shape;
    rhs_shape[axis] = self.shape[axis] - size;
    Ok((
      NDArray { ptr: lhs_ptr, shape: lhs_shape, strides: self.strides, is_contiguous: false, layout: self.layout },
      NDArray { ptr: rhs_ptr, shape: rhs_shape, strides: self.strides, is_contiguous: false, layout: self.layout },
    ))
  }

  /// This method returns a sub-array whose bounds are defined by an array of ranges.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn subarray(mut self, bounds: [Range<usize>; N]) -> NDArrayResult<Self> {
    for (i, r) in bounds.into_iter().enumerate() {
      let (start, end) = (r.start, r.end);
      if start > end { return Err(NDArrayError::IncorrectRange); }
      if end > self.shape[i] { return Err(NDArrayError::OutOfBound(Box::new(self.shape), i, end)); }
      self.ptr = self.ptr.add(self.strides[i] * start);
      self.shape[i] = end - start;
    }
    self.is_contiguous = false;
    Ok(self)
  }

  /// This method returns a reshaped array.
  pub fn reshape<const M: usize>(
    self,
    shape: [usize; M],
  ) -> NDArrayResult<NDArray<Ptr, M>>
  {
    let new_size: usize = shape.into_iter().product();
    let old_size: usize = self.shape.into_iter().product();
    if old_size != new_size { return Err(NDArrayError::SizeMismatch(old_size, new_size)); }
    if !self.is_contiguous { return Err(NDArrayError::ImpossibleToReshape); }
    let strides = if let Some(s) = shape_to_strides(shape, self.layout) {
      s
    } else {
      return Err(NDArrayError::ImpossibleToReshape) 
    };
    Ok(NDArray {
      ptr: self.ptr,
      shape,
      strides,
      is_contiguous: true,
      layout: self.layout,
    })
  }

  /// This method transposes an array.
  pub fn transpose(mut self, axes_order: [usize; N]) -> NDArrayResult<Self> {
    if axes_order.into_iter().enumerate().all(|(o, i)| o == i) { return Ok(self); }
    let mut sorted_axes_order = axes_order;
    sorted_axes_order.sort();
    for (i, o) in sorted_axes_order.into_iter().enumerate() {
      if i != o { return Err(NDArrayError::IncorrectIndicesOrder(Box::new(axes_order))); }
    }
    let mut new_shape = self.shape;
    let mut new_strides = self.strides;
    for (i, o) in axes_order.into_iter().enumerate() {
      new_shape[i] = self.shape[o];
      new_strides[i] = self.strides[o];
    }
    self.shape = new_shape;
    self.strides = new_strides;
    self.layout = if axes_order.into_iter().rev().enumerate().all(|(o, i)| o == i) {
      match self.layout {
        Layout::Fortran => Layout::C,
        Layout::C => Layout::Fortran,
        Layout::General => Layout::General,
      }
    } else {
      Layout::General
    };
    Ok(self)
  }

  /// This method returns iterator over raw pointers pointing to elements of an array.
  /// It traverses an array in the Fortran (logical) order.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn into_f_iter(self) -> impl ExactSizeIterator<Item = ParPtrWrapper<Ptr>> + Clone
  {
    let ptr = ParPtrWrapper(self.ptr);
    let size: usize = self.shape.into_iter().product();
    (0..size).into_iter().map(move |mut x| {
      let mut cur_ptr = ptr;
      for i in 0..N {
        let dim = self.shape.get_unchecked(i);
        cur_ptr = cur_ptr.add(self.strides.get_unchecked(i) * (x % dim));
        x /= dim;
      }
    cur_ptr
    })
  }

  /// This method returns iterator over raw pointers pointing to elements of an array.
  /// If array is not contiguous, it returns an error. If it is continuous, it returns iterator
  /// that traverses array in memory order.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn into_mem_iter(self) -> NDArrayResult<impl Iterator<Item = ParPtrWrapper<Ptr>> + Clone>
  {
    if !self.is_contiguous { return Err(NDArrayError::NotContiguous); }
    let ptr = ParPtrWrapper(self.ptr);
    let size: usize = self.shape.into_iter().product();
    Ok((0..size).map(move |x| {
      ptr.add(x)
    }))
  }

  /// This method returns iterator over raw pointers pointing to elements of an array.
  /// It traverses an array in a cache friendly way, in contrast to into_f_iter method.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn into_cache_friendly_iter(mut self) -> impl ExactSizeIterator<Item = ParPtrWrapper<Ptr>> + Clone
  {
    (self.strides, self.shape) = get_cache_friendly_order(self.strides, self.shape);
    self.into_f_iter()
  }
}

// ---------------------------------------------------------------------- //

macro_rules! impl_with_deref {
  ($ptr_type:ident) => {
    impl<T: Clone + Copy + Display + PartialEq + Send + 'static, const N: usize> NDArray<*$ptr_type T, N>
    {
      /// This method writes data from one array to other.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn write_to(
        self,
        other: NDArray<*mut T, N>,
      ) -> NDArrayResult<()>
      {
        if other.shape != self.shape { return Err(NDArrayError::ShapesMismatch(Box::new(self.shape), Box::new(other.shape))); }
        other.into_f_iter().zip(self.into_f_iter()).for_each(|(lhs, rhs)| {
          *lhs.0 = *rhs.0;
        });
        Ok(())
      }

      /// This method generates a buffer and the corresponding array that is the same as
      /// a source array, but has Fortran layout.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn gen_f_array(self) -> (Vec<T>, NDArray<*$ptr_type T, N>)
      {
        let mut buff: Vec<T> = self.into_f_iter().map(|x| *x.0).collect();
        let arr = NDArray::from_mut_slice(&mut buff, self.shape).unwrap();
        (buff, arr.into())
      }

      /// This method convert an array to its text representation. It replaces an implementation of
      /// the Debug trait, since it is unsafe method.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn to_string(self) -> String
      {
        let ptr = ParPtrWrapper(self.ptr);
        let size: usize = self.shape.into_iter().product();
        let mut prev_indices = [0; N];
        let mut new_indices = [0; N];
        let mut msg = String::with_capacity(2 * size); // TODO: calculate the required size precisely
        let msg_ref = &mut msg;
        for _ in 0..N {
          msg_ref.push('[');
        }
        msg_ref.push('\n');
        (0..size).into_iter().for_each(move |mut x| {
          let mut cur_ptr = ptr;
          for i in (0..N).rev() {
            let dim = self.shape.get_unchecked(i);
            let new_index = x % dim;
            new_indices[i] = new_index;
            cur_ptr = cur_ptr.add(self.strides.get_unchecked(i) * new_index);
            x /= dim;
          }
          let mut counter = 0;
          for (i, (prev, new)) in prev_indices.into_iter().zip(new_indices).enumerate() {
            if prev > new {
              if i != (N-1) {
                if counter == 0 {
                  msg_ref.push('\n')
                }
                msg_ref.push(']');
                counter += 1;
              }
            }
          }
          counter = 0;
          for (i, (prev, new)) in prev_indices.into_iter().zip(new_indices).enumerate() {
            if prev > new {
              if i == (N-1) {
                msg_ref.push('\n');
              } else {
                if counter == 0 {
                  msg_ref.push('\n');
                }
                msg_ref.push_str("[");
                counter += 1;
              }
            }
          }
          let str_num = format!("{:.4} ", *cur_ptr.0); // TODO: reduce the number of allocation
          msg_ref.push_str(&str_num);
          prev_indices = new_indices;
        });
        msg.push('\n');
        for _ in 0..N {
          msg.push(']');
        }
        msg
      }

      ///This method checks the equality of two arrays.
      /// It is replacement of the PartialEq trait implementation,
      /// since this method is unsafe.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn eq(self, other: impl Into<Self> + Clone + Copy) -> bool
      {
        self.clone().into_f_iter().zip(other.into().into_f_iter())
          .all(|(lhs, rhs)| {
            *lhs.0 == *rhs.0
          }) && (self.shape == other.into().shape)
      }

      /// This method returns a pointer to an element of an array given the index of an element.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn at(self, index: [usize; N]) -> NDArrayResult<* $ptr_type T>
      {
        let mut ptr = self.ptr;
        for (i, idx) in index.into_iter().enumerate() {
          ptr = ptr.add(self.strides.get_unchecked(i) * idx);
        }
        Ok(ptr)
      }

      /// This method generates a buffer and the corresponding array of Fortran layout with
      /// a dedicated axis being permuted according to the given order.
      /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
      /// as for raw pointers.
      pub unsafe fn gen_f_array_from_axis_order(
        mut self,
        order: &[usize],
        axis: usize,
      ) -> (Vec<T>, NDArray<*$ptr_type T, N>)
      {
        self.shape[axis] = order.len();
        let ptr = ParPtrWrapper(self.ptr);
        let size = self.shape.into_iter().product::<usize>();
        let mut buff: Vec<T> = (0..size).into_iter().map(move |mut x| {
          let mut cur_ptr = ptr;
          for i in 0..N {
            let dim = *self.shape.get_unchecked(i);
            cur_ptr = if i != axis {
              cur_ptr.add(self.strides.get_unchecked(i) * (x % dim))
            } else {
              cur_ptr.add(self.strides.get_unchecked(axis) * order.get_unchecked(x % dim))
            };
            x /= dim;
          }
        *cur_ptr.0
        }).collect();
        let arr = NDArray::from_mut_slice(&mut buff, self.shape).unwrap();
        (buff, arr.into())
      }
    }
  };
}

impl_with_deref!(mut);
impl_with_deref!(const);

// ---------------------------------------------------------------------- //

impl<T: 'static, const N: usize> NDArray<*const T, N>
{
  /// This method generates an array from a const slice.
  pub fn from_slice(slice: &[T], shape: [usize; N]) -> NDArrayResult<Self> {
    let m: NDArray<_, 1> = slice.into();
    m.reshape(shape)
  }
}

/// This method generates an array from a mut slice.
impl<T: 'static, const N: usize> NDArray<*mut T, N>
{
  pub fn from_mut_slice(slice: &mut [T], shape: [usize; N]) -> NDArrayResult<Self> {
    let m: NDArray<_, 1> = slice.into();
    m.reshape(shape)
  }
}

impl<T> From<&[T]> for NDArray<*const T, 1>
{
  fn from(buff: &[T]) -> Self {
    let ptr = buff.as_ptr();
    let len = buff.as_ref().len();
    Self { ptr, shape: [len], strides: [1], is_contiguous: true, layout: Layout::Fortran}
  }
}

impl<T> From<&mut [T]> for NDArray<*mut T, 1>
{
  fn from(buff: &mut [T]) -> Self {
    let ptr = buff.as_mut_ptr();
    let len = buff.as_ref().len();
    Self { ptr, shape: [len], strides: [1], is_contiguous: true, layout: Layout::Fortran}
  }
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::prelude::*;

  #[test]
  fn test_split() {
    let buff = (0..256).collect::<Vec<i32>>();
    let arr1 = unsafe {
      NDArray::from_slice(&buff, [4, 8, 2, 4])
        .unwrap()
        .split_across_axis(1, 3)
        .unwrap()
        .1
        .split_across_axis(1, 2)
        .unwrap()
        .0
        .split_across_axis(3, 1)
        .unwrap()
        .1
        .split_across_axis(3, 2)
        .unwrap()
        .0
    };
    let arr2 = Array4::from_shape_vec((4, 2, 8, 4), buff).unwrap();
    let arr2_slice = arr2.slice(s![1..2, .., 3..5, ..]);
    let is_eq = arr2_slice.iter().zip(unsafe { arr1.into_f_iter() }).all(|(lhs, rhs)| {
      *lhs == unsafe { *rhs.0 }
    });
    assert!(is_eq);
  }

  #[test]
  fn test_subarray() {
    let buff = (0..256).collect::<Vec<i32>>();
    let arr1 = unsafe { NDArray::from_slice(&buff, [4, 8, 4, 2])
      .unwrap()
      .subarray([1..3, 3..6, 0..2, 0..2]) }
      .unwrap();
    let arr2 = Array4::from_shape_vec((2, 4, 8, 4), buff).unwrap();
    let arr2_slice = arr2.slice(s![0..2, 0..2, 3..6, 1..3]);
    let is_eq = arr2_slice.iter().zip(unsafe { arr1.into_f_iter() }).all(|(lhs, rhs)| {
      *lhs == unsafe { *rhs.0 }
    });
    assert!(is_eq);
  }

  #[test]
  fn test_reshape() {
    let buff = (0..256).collect::<Vec<i32>>();
    let arr1 = NDArray::from_slice(&buff, [4, 8, 4, 2]).unwrap();
    let resh_arr1 = arr1.reshape([32, 8]).unwrap();
    let arr2 = NDArray::from_slice(&buff, [32, 8]).unwrap();
    assert!(unsafe { resh_arr1.eq(arr2) })
  }

  #[test]
  fn test_transpose() {
    let buff = (0..256).collect::<Vec<i32>>();
    let arr1 = NDArray::from_slice(&buff, [4, 8, 4, 2]).unwrap();
    let transp_arr1 = arr1.transpose([2, 1, 3, 0]).unwrap();
    let arr2 = Array4::from_shape_vec((4, 2, 8, 4).strides((1, 128, 4, 32)), buff).unwrap();
    let is_eq = arr2.iter().zip(unsafe { transp_arr1.into_f_iter() }).all(|(lhs, rhs)| {
      *lhs == unsafe { *rhs.0 }
    });
    assert!(is_eq);
  }

  #[test]
  fn test_write_to_and_gen_f_array() {
    let buff1 = (0..256).collect::<Vec<i32>>();
    let mut buff2: Vec<i32> = Vec::with_capacity(256);
    unsafe { buff2.set_len(256) };
    let arr1 = NDArray::from_slice(&buff1, [8, 4, 4, 2]).unwrap();
    let transp_arr1 = arr1.transpose([1, 3, 0, 2]).unwrap();
    let arr2 = NDArray::from_mut_slice(&mut buff2, [4, 2, 8, 4]).unwrap();
    unsafe { transp_arr1.write_to(arr2).unwrap() };
    let is_eq = unsafe { arr2.into_mem_iter().unwrap().zip(transp_arr1.into_f_iter()).all(|(lhs, rhs)| {
      *lhs.0 == *rhs.0
    }) };
    assert!(is_eq);
    let (buff3, arr3) = unsafe { transp_arr1.gen_f_array() };
    assert!(unsafe { transp_arr1.eq(arr2) });
    assert!(unsafe { arr3.eq(arr2) });
    assert_eq!(buff2, buff3);
  }

  #[test]
  fn test_at() {
    use crate::ndarray_utils::shape_to_strides;
    let indices = [[5, 2, 1, 0], [2, 1, 1, 1], [0, 0, 0, 0], [7, 3, 3, 1]];
    let buff1 = (0..256).collect::<Vec<usize>>();
    let arr1 = NDArray::from_slice(&buff1, [8, 4, 4, 2]).unwrap();
    let strides = shape_to_strides(arr1.shape, Layout::Fortran).unwrap();
    for index in indices {
      let val = unsafe { arr1.at(index).unwrap() };
      let true_val = index.into_iter().zip(strides).fold(0, |acc, x| {
        acc + x.0 * x.1
      });
      assert_eq!(unsafe { *val }, true_val);
    }
  }

  #[test]
  fn test_gen_f_array_from_axis_order() {
    let buff1 = (0..256).collect::<Vec<i32>>();
    let arr1 = NDArray::from_slice(&buff1, [8, 4, 4, 2]).unwrap();
    let transp_arr1 = arr1.transpose([3, 2, 1, 0]).unwrap();
    let (_buff2, arr2) = unsafe { transp_arr1.gen_f_array_from_axis_order(&[3, 1, 0, 2], 1) };
    let transp_arr1 = transp_arr1.transpose([0, 2, 3, 1]).unwrap();
    let iter = unsafe { arr2
      .subarray([0..2, 2..3, 0..4, 0..8]).unwrap()
      .into_f_iter()
      .chain(
        arr2
        .subarray([0..2, 1..2, 0..4, 0..8]).unwrap()
        .into_f_iter()
      )
      .chain(
        arr2
        .subarray([0..2, 3..4, 0..4, 0..8]).unwrap()
        .into_f_iter()
      )
      .chain(
        arr2
        .subarray([0..2, 0..1, 0..4, 0..8]).unwrap()
        .into_f_iter()
      )
    };
    let is_eq = unsafe {
      iter.zip( transp_arr1.into_f_iter() ).all(|(lhs, rhs)| {
        *lhs.0 == *rhs.0
      })
    };
    assert!(is_eq);
  }
}

