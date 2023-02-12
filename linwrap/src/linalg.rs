use std::{
  ffi::{
    c_char,
    c_int,
  },
  fmt::Debug,
};

use num_traits::{
  ToPrimitive,
  One,
};

/*use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IndexedParallelIterator;*/

use crate::{
  NDArray,
  ndarray::{
    NDArrayError,
    NDArrayResult,
    Layout,
  },
  linalg_utils::triangular_split,
    LinalgComplex,
    LinalgReal,
};

// TODO: get advantage of the generalized storage (arbitrary strides).
// TODO: refactor maxvol
// TODO: refactor all tests
// TODO: add documentation

// ---------------------------------------------------------------------- //

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LapackError {
  /// This error appears when the linear solve lapack routine ?gesv fails.
  ErrorGESV(c_int),

  /// This error appears when the lapack routine for SVD ?gesvd fails.
  ErrorGESVD(c_int),

  /// This error appears when the lapack routine for QR decomposition ?geqrf fails. 
  ErrorGEQRF(c_int),

  /// This error appears when the lapack routine for QR decomposition postprocessing ?orgqr fails. 
  ErrorORGQR(c_int),

  /// This error appears when ?getrf lapack subroutine fails.
  ErrorGETRF(c_int),

  /// This error appears when ?getrs lapack subroutine fails.
  ErrorGETRS(c_int),

  /// This error appears when ?heev of ?syev lapack subroutine fails.
  ErrorHEEV(c_int)
}

impl Into<NDArrayError> for LapackError {
  fn into(self) -> NDArrayError {
      NDArrayError::LapackError(self)
  }
}

impl Debug for LapackError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::ErrorGESV(code) => { f.write_str(&format!("Lapack linear systems solver (?GESV) failed with code {}.", code)) },
      Self::ErrorGESVD(code) => { f.write_str(&format!("Lapack SVD routine (?GESVD) failed with code {}.", code)) },
      Self::ErrorGEQRF(code) => { f.write_str(&format!("Lapack QR decomposition routine (?GEQRF) failed with code {}.", code)) },
      Self::ErrorORGQR(code) => { f.write_str(&format!("Lapack routine for QR decomposition result postprocessing (?ORGQR) failed with code {}", code)) },
      Self::ErrorGETRF(code) => { f.write_str(&format!("Lapack routine ?GETRF failed with code {}", code)) },
      Self::ErrorGETRS(code) => { f.write_str(&format!("Lapack routine ?GETRS failed with code {}", code)) },
      Self::ErrorHEEV(code) => { f.write_str(&format!("Lapack routine ?HEEV or ?SYEV failed with code {}", code)) },
    }
  }
}

// ---------------------------------------------------------------------- //

impl<T> NDArray<*mut T, 2>
where
  T: LinalgComplex,
  T::Real: LinalgReal,
{
  /// This method performs the multiplication of matrices a and b and writes
  /// the result into self.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn matmul_inplace(
    self,
    a: impl Into<NDArray<*const T, 2>>,
    b: impl Into<NDArray<*const T, 2>>,
  ) -> NDArrayResult<()>
  {
    let mut a = a.into();
    let mut b = b.into();
    let is_a_transposed = if let Layout::C = a.layout {
      a = a.transpose([1, 0])?;
      true
    } else {
      false
    };
    let is_b_transposed = if let Layout::C = b.layout {
      b = b.transpose([1, 0])?;
      true
    } else {
      false
    };
    if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if self.strides[1] < self.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    if a.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if b.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    let (m, k) = if is_a_transposed {
      (a.shape[1] as c_int, a.shape[0] as c_int)
    } else {
      (a.shape[0] as c_int, a.shape[1] as c_int)
    };
    let n = if is_b_transposed  {
      if b.shape[1] as c_int != k { return Err(NDArrayError::MatmulDimMismatch(b.shape[1], k as usize)) }
      b.shape[0] as c_int
    } else {
      if b.shape[0] as c_int != k { return Err(NDArrayError::MatmulDimMismatch(b.shape[0], k as usize)) }
      b.shape[1] as c_int
    };
    let transa = if is_a_transposed { 'T' as c_char } else { 'N' as c_char };
    let transb = if is_b_transposed { 'T' as c_char } else { 'N' as c_char };
    if (m != self.shape[0] as c_int) || (n != self.shape[1] as c_int) { return Err(NDArrayError::IncorrectShape(Box::new(self.shape), Box::new([m as usize, n as usize]))); }
    let alpha = T::one();
    let beta = T::zero();
    let lda = a.strides[1] as c_int;
    let ldb = b.strides[1] as c_int;
    let ldc = self.strides[1] as c_int;
    unsafe {
      T::gemm(&transa, &transb, &m, &n, &k, &alpha, a.ptr,
            &lda, b.ptr, &ldb, &beta, self.ptr, &ldc)
    };
    Ok(())
  }

  /// This method solves a batch of systems of linear equations,
  /// i.e. AX = B, where B is rhs, A is self. The solution X is written
  /// to the rhs array. The array self is destroyed after a call of the method.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn solve(
    self,
    rhs: Self,
  ) -> NDArrayResult<()>
    {
      if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
      if self.strides[1] < self.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
      if rhs.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
      if rhs.strides[1] < rhs.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
      if self.shape[1] != self.shape[0] { return Err(NDArrayError::SquareMatrixRequired(self.shape[0], self.shape[1])); }
      let n = self.shape[1] as c_int;
      if rhs.shape[0] as c_int != n { return Err(NDArrayError::IncorrectShape(Box::new(rhs.shape), Box::new([n as usize, rhs.shape[1]]))); }
      let nrhs = rhs.shape[1] as c_int;
      let lda = self.strides[1] as c_int;
      let ldb = rhs.strides[1] as c_int;
      let mut info: c_int = 0;
      let mut ipiv_buff = Vec::with_capacity(self.shape[1]);
      unsafe { ipiv_buff.set_len(self.shape[1]); }
      let ipiv = ipiv_buff.as_mut_ptr();
      unsafe { T::gesv
        (
          &n,
          &nrhs,
          self.ptr,
          &lda,
          ipiv,
          rhs.ptr,
          &ldb,
          &mut info,
        );
      }
      if info != 0 { return Err(LapackError::ErrorGESV(info).into()); }
      Ok(())
    }
  
  pub unsafe fn eigh(
    self,
    lmbd: NDArray<*mut T::Real, 1>,
  ) -> NDArrayResult<()>
  {
    if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if self.strides[1] < self.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    if lmbd.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    let jobz = 'V' as c_char;
    let uplo = 'U' as c_char;
    let n = self.shape[1] as c_int;
    let lda = self.strides[1] as c_int;
    let lwork = -1  as c_int;
    let mut work = T::zero();
    let mut rwork = Vec::with_capacity(std::cmp::max(1, 3 * n as usize - 2));
    unsafe { rwork.set_len(std::cmp::max(1, 3 * n as usize - 2)); }
    let mut info = 0;
    unsafe {
      T::heev(
        &jobz,
        &uplo,
        &n,
        self.ptr,
        &lda,
        lmbd.ptr,
        &mut work,
        &lwork,
        rwork.as_mut_ptr(),
        &mut info,
      )
    }
    let lwork: c_int = ToPrimitive::to_i32(&work.re()).unwrap() as c_int;
    let mut work: Vec<T> = Vec::with_capacity(lwork as usize);
    unsafe {
      T::heev(
        &jobz,
        &uplo,
        &n,
        self.ptr,
        &lda,
        lmbd.ptr,
        work.as_mut_ptr(),
        &lwork,
        rwork.as_mut_ptr(),
        &mut info,
      )
    }
    if info != 0 { return Err(LapackError::ErrorHEEV(info).into()); }
    Ok(())
  }

  /// This method performs computation of the SVD of self matrix.
  /// The result is written in u, lmbd and vdag arrays.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn svd(
    self,
    u: Self,
    lmbd: NDArray<*mut T::Real, 1>,
    vdag: Self,
  ) -> NDArrayResult<()>
  {
    if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if self.strides[1] < self.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    if u.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if u.strides[1] < u.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    if vdag.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if vdag.strides[1] < vdag.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    if lmbd.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    let jobu = 'S' as c_char;
    let jobvt = 'S' as c_char;
    let m = self.shape[0] as c_int;
    let n = self.shape[1] as c_int;
    let lda = self.strides[1] as c_int;
    let ldu = u.strides[1] as c_int;
    let ldvt = vdag.strides[1] as c_int;
    let min_dim = std::cmp::min(n, m);
    if lmbd.shape[0] as c_int != min_dim { return Err(NDArrayError::IncorrectShape(Box::new(lmbd.shape), Box::new([min_dim as usize]))); }
    if (u.shape[0] as c_int != m) || (u.shape[1] as c_int != min_dim) { return Err(NDArrayError::IncorrectShape(Box::new(u.shape), Box::new([m as usize, min_dim as usize]))); }
    if (vdag.shape[1] as c_int != n) || (vdag.shape[0] as c_int != min_dim) { return Err(NDArrayError::IncorrectShape(Box::new(vdag.shape), Box::new([min_dim as usize, n as usize]))); }
    let lwork = -1  as c_int;
    let mut work = T::zero();
    let mut rwork = Vec::with_capacity(5 * min_dim as usize);
    unsafe { rwork.set_len(5 * min_dim as usize); }
    let mut info = 0;
    // worksapce query
    unsafe { T::gesvd(
      &jobu,
      &jobvt,
      &m,
      &n,
      self.ptr,
      &lda,
      lmbd.ptr,
      u.ptr,
      &ldu,
      vdag.ptr,
      &ldvt,
      &mut work,
      &lwork,
      rwork.as_mut_ptr(),
      &mut info) }
    let lwork: c_int = ToPrimitive::to_i32(&work.re()).unwrap() as c_int;
    let mut work: Vec<T> = Vec::with_capacity(lwork as usize);
    unsafe { T::gesvd(
      &jobu,
      &jobvt,
      &m,
      &n,
      self.ptr,
      &lda,
      lmbd.ptr,
      u.ptr,
      &ldu,
      vdag.ptr,
      &ldvt,
      work.as_mut_ptr(),
      &lwork,
      rwork.as_mut_ptr(),
      &mut info) }
    if info != 0 { return Err(LapackError::ErrorGESVD(info).into()); }
    Ok(())
  }

  unsafe fn householder_(&mut self, tau: *mut T) -> NDArrayResult<()> {
    let m = self.shape[0] as c_int;
    let n = self.shape[1] as c_int;
    let lda = self.strides[1] as c_int;
    let mut work = T::zero();
    let lwork = -1 as c_int;
    let mut info: c_int = 0;
    T::geqrf(
      &m,
      &n,
      self.ptr,
      &lda,
      tau,
      &mut work,
      &lwork,
      &mut info,
    );
    let lwork: c_int = ToPrimitive::to_i32(&work.re()).unwrap() as c_int;
    let mut work_buff: Vec<T> = Vec::with_capacity(lwork as usize);
    unsafe { work_buff.set_len(lwork as usize); }
    let work = work_buff.as_mut_ptr();
    T::geqrf(
      &m,
      &n,
      self.ptr,
      &lda,
      tau,
      work,
      &lwork,
      &mut info,
    );
    if info != 0 { return Err(LapackError::ErrorGEQRF(info).into()); }
    Ok(())
  }

  unsafe fn householder_to_q_(self, tau: *mut T) -> NDArrayResult<()> {
    let m = self.shape[0] as c_int;
    let n = self.shape[1] as c_int;
    let k = std::cmp::min(m, n);
    let lda = self.strides[1] as c_int;
    let mut work = T::zero();
    let lwork = -1 as c_int;
    let mut info: c_int = 0;
    T::ungqr(
      &m,
      &n,
      &k,
      self.ptr,
      &lda,
      tau,
      &mut work,
      &lwork,
      &mut info,
    );
    let lwork: c_int = ToPrimitive::to_i32(&work.re()).unwrap() as c_int;
    let mut work_buff: Vec<T> = Vec::with_capacity(lwork as usize);
    unsafe { work_buff.set_len(lwork as usize); }
    let work = work_buff.as_mut_ptr();
    T::ungqr(
      &m,
      &n,
      &k,
      self.ptr,
      &lda,
      tau,
      work,
      &lwork,
      &mut info,
    );
    if info != 0 { return Err(LapackError::ErrorORGQR(info).into()); }
    Ok(())
  }

  /// This method performes QR deomposition of a self matrix of size m x n.
  /// A matrix other is an auxiliary matrix of size min(m, n) x min(m, n).
  /// If m > n the resultin Q matrix is written to self array and R matrix is
  /// written to other array. Otherwise, Q is written to other array, R is
  /// written to self array.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn qr(
    mut self,
    other: Self,
  ) -> NDArrayResult<()>
  {
    let m = self.shape[0];
    let n = self.shape[1];
    let min_dim = std::cmp::min(n, m);
    if other.shape != [min_dim, min_dim] { return Err(NDArrayError::SquareMatrixRequired(other.shape[0], other.shape[1])); }
    let mut tau = Vec::with_capacity(min_dim);
    unsafe { tau.set_len(min_dim); }
    unsafe { self.householder_(tau.as_mut_ptr())?; }
    unsafe { triangular_split(self, other); }
    if m > n {
      unsafe { self.householder_to_q_(tau.as_mut_ptr())?; }
    } else {
      unsafe { other.householder_to_q_(tau.as_mut_ptr())?; }
    }
    Ok(())
  }

  /// This method performes RQ deomposition of a self matrix of size m x n.
  /// A matrix other is an auxiliary matrix of size min(m, n) x min(m, n).
  /// If m < n the resultin Q matrix is written to self array and R matrix is
  /// written to other array. Otherwise, Q is written to other array, R is
  /// written to self array.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn rq(
    self,
    other: Self,
  ) -> NDArrayResult<()>
  {
    // TODO: to avoid extra allocations and copying use dedicated routines for rq decomposition
    let (mut _self_transposed_buff, self_transposed_arr) = self.transpose([1, 0])?.gen_f_array();
    let (mut _other_transposed_buff, other_transposed_arr) = other.transpose([1, 0])?.gen_f_array();
    self_transposed_arr.qr(other_transposed_arr)?;
    self_transposed_arr.transpose([1, 0])?.write_to(self)?;
    other_transposed_arr.transpose([1, 0])?.write_to(other)?;
    Ok(())
  }

  unsafe fn rank1_update(
    self,
    col: impl Into<NDArray<*const T, 2>>,
    row: impl Into<NDArray<*const T, 2>>,
    alpha: T,
  ) -> NDArrayResult<()>
  {
    let col = col.into();
    let row = row.into();
    if col.shape[1] != 1 { panic!("col matrix in the rank1_update method has incorrect shape: given: {:?}, required: {:?}.", col.shape, [self.shape[0], 1]) }
    if row.shape[0] != 1 { panic!("row matrix in the rank1_update method has incorrect shape: given: {:?}, required: {:?}.", col.shape, [1, self.shape[1]]) }
    if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if self.strides[1] < self.shape[0] { return Err(NDArrayError::MutableElementsOverlapping); }
    let m = self.shape[0] as c_int;
    let n = self.shape[1] as c_int;
    if col.shape[0] != m as usize { panic!("col matrix in the rank1_update method has incorrect shape: given: {:?}, required: {:?}.", col.shape, [self.shape[0], 1]) }
    if row.shape[1] != n as usize { panic!("row matrix in the rank1_update method has incorrect shape: given: {:?}, required: {:?}.", col.shape, [1, self.shape[1]]) }
    let incx = col.strides[0] as c_int;
    let incy = row.strides[1] as c_int;
    let lda = self.strides[1] as c_int;
    T::ger(&m, &n, &alpha, col.ptr, &incx, row.ptr, &incy, self.ptr, &lda);
    Ok(())
  }

  unsafe fn maxvol_preprocess(self) -> NDArrayResult<Vec<usize>> {
    let mut ipiv = Vec::with_capacity(self.shape[1]);
    ipiv.set_len(self.shape[1]);
    let m = self.shape[0] as c_int;
    let n = self.shape[1] as c_int;
    let lda = self.strides[1] as c_int;
    let mut info: c_int = 0;
    T::getrf(&m, &n, self.ptr, &lda, ipiv.as_mut_ptr(), &mut info);
    let (a, b) = self.split_across_axis(0, self.shape[1])?;
    let side = 'R' as c_char;
    let uplo = 'L' as c_char;
    let transa = 'N' as c_char;
    let diag = 'U' as c_char;
    let m = b.shape[0] as c_int;
    let n = b.shape[1] as c_int;
    let alpha = T::one();
    let lda = a.strides[1] as c_int;
    let ldb = b.strides[1] as c_int;
    T::trsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a.ptr, &lda, b.ptr, &ldb);
    Ok(ipiv.into_iter().map(|x| x as usize).collect())
  }

  /// This method runs Maxvol algorithm for a matrix self of size m x n, where m >= n.
  /// It returns the order of rows that corrseponds to the upper most part of the matrix
  /// to be a dominant submatrix with accuracy delta. The self matrix is rewritten by the
  /// matrix that has reshaffeled rows according to the obtained rows order and
  /// multiplied by the inverse dominant part from the right.
  /// Safety: NDArray is a raw pointer with additional information. Thus, safety rules are the same
  /// as for raw pointers.
  pub unsafe fn maxvol(self, delta: T::Real) -> NDArrayResult<Vec<usize>>
  {
    let m = self.shape[0];
    let n = self.shape[1];
    if self.strides[0] != 1 { return Err(NDArrayError::FortranLayoutRequired); }
    if self.strides[1] < m { return Err(NDArrayError::MutableElementsOverlapping); }
    if m < n { return Err(NDArrayError::MaxvolInputSizeMismatch(m, n)); }
    let mut order: Vec<usize> = (0..m).collect();
    let mut x_buff: Vec<T> = Vec::with_capacity(m - n);
    unsafe { x_buff.set_len(m - n); }
    let x = NDArray::from_mut_slice(&mut x_buff, [m - n, 1])?;
    let mut y_buff: Vec<T> = Vec::with_capacity(n);
    unsafe { y_buff.set_len(n); }
    let y = NDArray::from_mut_slice(&mut y_buff, [1, n])?;
    let ipiv = self.maxvol_preprocess()?;
    for (i, j) in ipiv.into_iter().enumerate() {
      order.swap(i, j-1);
    }
    let (a, b) = self.split_across_axis(0, n)?;
    unsafe { (0..(n * n)).into_iter().zip(a.into_f_iter()).for_each(|(i, x)| {
      if i % (n + 1) == 0 { *x.0 = T::one() } else { *x.0 = T::zero() }
    }) };
    let mut val;
    let mut indices;
    loop {
      (val, indices) = b.argmax();
      let row_num = unsafe { *indices.get_unchecked(0) };
      let col_num = unsafe { *indices.get_unchecked(1) };
      if val.abs() < delta + T::Real::one() { break; }
      let bij = *b.at(indices)?;
      let col = b.subarray([0..(m - n), col_num..(col_num + 1)])?;
      let row = b.subarray([row_num..(row_num + 1), 0..n])?;
      col.write_to(x)?;
      row.write_to(y)?;
      let elem = &mut *x.at([row_num, 0])?;
      *elem = *elem + T::one();
      let elem = &mut *y.at([0, col_num])?;
      *elem = *elem - T::one();
      b.rank1_update(x, y, -T::one() / bij)?;
      order.swap(col_num, row_num + n);
    }
    Ok(order)
  }
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::init_utils::BufferGenerator;
  use num_complex::{
    Complex64,
    Complex32,
  };
  use num_traits::One;
  use crate::{
    NDArray,
    LinalgComplex,
    LinalgReal,
  };
  use ndarray::Array;
  use ndarray_einsum_beta::einsum;

  #[inline]
  unsafe fn _test_matmul_inplace<T>
  (
    size: (usize, usize, usize),
    einsum_str: &str,
    is_a_transposed: bool,
    is_b_transposed: bool,
    acc: T::Real,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, k, n) = size;
    let buff_a = T::random_normal(k * m);
    let buff_a = Array::from_shape_vec(if is_a_transposed { [m, k] } else { [k, m] }, buff_a).unwrap();
    let buff_b = T::random_normal(n * k);
    let buff_b = Array::from_shape_vec(if is_b_transposed { [k, n] } else { [n, k] }, buff_b).unwrap();
    let einsum_c = einsum(einsum_str, &[&buff_b, &buff_a]).unwrap();
    let einsum_c = einsum_c.iter().map(|x| *x).collect::<Vec<_>>();
    let einsum_c = NDArray::from_slice(&einsum_c, [m, n]).unwrap();

    let a = NDArray::from_slice(buff_a.as_slice_memory_order().unwrap(), [k, m]).unwrap();
    let a = if is_a_transposed { a.transpose([1, 0]).unwrap() } else { a.reshape([m, k]).unwrap() };
    let b = NDArray::from_slice(buff_b.as_slice_memory_order().unwrap(), [n, k]).unwrap();
    let b = if is_b_transposed { b.transpose([1, 0]).unwrap() } else { b.reshape([k, n]).unwrap() };
    let mut buff_c: Vec<T> = vec![T::zero(); m * n];
    let c = NDArray::from_mut_slice(buff_c.as_mut_slice(), [m, n]).unwrap();
    c.matmul_inplace(a, b).unwrap();
    c.sub_inpl(einsum_c).unwrap();
    assert!(c.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_matmul_inplace() {
    unsafe {
      _test_matmul_inplace::<f32>(      (4, 6, 5), "ik,kj->ij", false, false, 1e-5 );
      _test_matmul_inplace::<f64>(      (4, 6, 5), "ik,kj->ij", false, false, 1e-10);
      _test_matmul_inplace::<Complex32>((4, 6, 5), "ik,kj->ij", false, false, 1e-5 );
      _test_matmul_inplace::<Complex64>((4, 6, 5), "ik,kj->ij", false, false, 1e-10);
      _test_matmul_inplace::<f32>(      (4, 6, 5), "ik,jk->ij", true , false, 1e-5 );
      _test_matmul_inplace::<f64>(      (4, 6, 5), "ik,jk->ij", true , false, 1e-10);
      _test_matmul_inplace::<Complex32>((4, 6, 5), "ik,jk->ij", true , false, 1e-5 );
      _test_matmul_inplace::<Complex64>((4, 6, 5), "ik,jk->ij", true , false, 1e-10);
      _test_matmul_inplace::<f32>(      (4, 6, 5), "ki,kj->ij", false, true , 1e-5 );
      _test_matmul_inplace::<f64>(      (4, 6, 5), "ki,kj->ij", false, true , 1e-10);
      _test_matmul_inplace::<Complex32>((4, 6, 5), "ki,kj->ij", false, true , 1e-5 );
      _test_matmul_inplace::<Complex64>((4, 6, 5), "ki,kj->ij", false, true , 1e-10);
      _test_matmul_inplace::<f32>(      (4, 6, 5), "ki,jk->ij", true , true , 1e-5 );
      _test_matmul_inplace::<f64>(      (4, 6, 5), "ki,jk->ij", true , true , 1e-10);
      _test_matmul_inplace::<Complex32>((4, 6, 5), "ki,jk->ij", true , true , 1e-5 );
      _test_matmul_inplace::<Complex64>((4, 6, 5), "ki,jk->ij", true , true , 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_solve<T>(size: (usize, usize), acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (n, nrhs) = size;
    let mut buff_a = T::random_normal(n * n);
    let a = NDArray::from_slice(buff_a.as_slice(), [n, n]).unwrap();
    let buff_x = T::random_normal(n * nrhs);
    let x = NDArray::from_slice(buff_x.as_slice(), [n, nrhs]).unwrap();
    let mut buff_b = T::random_normal(n * nrhs);
    let b = NDArray::from_mut_slice(buff_b.as_mut_slice(), [n, nrhs]).unwrap();
    b.matmul_inplace(a, x).unwrap();
    let a = NDArray::from_mut_slice(buff_a.as_mut_slice(), [n, n]).unwrap();
    a.solve(b).unwrap();
    let b = NDArray::from_mut_slice(buff_b.as_mut_slice(), [n, nrhs]).unwrap();
    b.sub_inpl(x).unwrap();
    assert!(b.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_solv() {
    unsafe {
      _test_solve::<f32>(      (10, 15), 1e-5 );
      _test_solve::<f64>(      (10, 15), 1e-10);
      _test_solve::<Complex32>((10, 15), 1e-5 );
      _test_solve::<Complex64>((10, 15), 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_eigh<T>(m: usize, acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let mut buff_a = T::random_normal(m * m);
    let mut buff_a_conj = buff_a.clone();
    let mut buff_aux = T::random_normal(m * m);
    let a = NDArray::from_mut_slice(buff_a.as_mut_slice(), [m, m]).unwrap();
    let a_h = NDArray::from_mut_slice(buff_a_conj.as_mut_slice(), [m, m]).unwrap().transpose([1, 0]).unwrap();
    a_h.conj();
    let aux = NDArray::from_mut_slice(buff_aux.as_mut_slice(), [m, m]).unwrap();
    a.add_inpl(a_h).unwrap();
    a.write_to(a_h).unwrap();
    let mut lmbd_buff: Vec<T::Real> = T::Real::uninit_buff(m);
    let lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [m]).unwrap();
    a.eigh(lmbd).unwrap();
    let mut lmbd_buff: Vec<T> = lmbd_buff.into_iter().map(|x| T::from(x).unwrap()).collect();
    let lmbd = NDArray::from_mut_slice(&mut lmbd_buff, [1, m]).unwrap();
    aux.matmul_inplace(a_h, a).unwrap();
    a.mul_inpl(lmbd).unwrap();
    aux.sub_inpl(a).unwrap();
    assert!(aux.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_eigh() {
    unsafe {
      _test_eigh::<f32>(      20, 1e-5 );
      _test_eigh::<f64>(      20, 1e-10);
      _test_eigh::<Complex32>(20, 1e-5 );
      _test_eigh::<Complex64>(20, 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_svd<T>(size: (usize, usize), acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, n) = size;
    let min_dim = std::cmp::min(m, n);
    let mut buff_a = T::random_normal(m * n);
    let buff_a_copy = buff_a.clone();
    let a = NDArray::from_mut_slice(buff_a.as_mut_slice(), [m, n]).unwrap();
    let a_copy = NDArray::from_slice(buff_a_copy.as_slice(), [m, n]).unwrap();
    let mut buff_u = T::random_normal(m * min_dim);
    let u = NDArray::from_mut_slice(buff_u.as_mut_slice(), [m, min_dim]).unwrap();
    let mut buff_vdag = T::random_normal(min_dim * n);
    let vdag = NDArray::from_mut_slice(buff_vdag.as_mut_slice(), [min_dim, n]).unwrap();
    let mut s_buff: Vec<T::Real> = T::Real::uninit_buff(min_dim);
    let s = NDArray::from_mut_slice(&mut s_buff, [min_dim]).unwrap();
    a.svd(u, s, vdag).unwrap();
    // Here we check that s is non-negative
    assert!(s_buff.iter().all(|x| *x >= T::zero().re() ));
    // Here we check isometric property of u and v
    let buff_vdag_copy: Vec<_> = buff_vdag.iter().map(|x| { x.conj() }).collect();
    let buff_u_copy: Vec<_> = buff_u.iter().map(|x| { x.conj() }).collect();
    let mut buff_v_vdag= vec![T::zero(); min_dim * min_dim];
    let mut buff_udag_u= vec![T::zero(); min_dim * min_dim];
    let v_copy = NDArray::from_slice(buff_vdag_copy.as_slice(), [min_dim, n]).unwrap();
    let udag_copy = NDArray::from_slice(buff_u_copy.as_slice(), [m, min_dim]).unwrap();
    let v_vdag = NDArray::from_mut_slice(buff_v_vdag.as_mut_slice(), [min_dim, min_dim]).unwrap();
    let udag_u = NDArray::from_mut_slice(buff_udag_u.as_mut_slice(), [min_dim, min_dim]).unwrap();
    let u = NDArray::from_slice(buff_u.as_slice(), [m, min_dim]).unwrap();
    let vdag = NDArray::from_slice(buff_vdag.as_slice(), [min_dim, n]).unwrap();
    v_vdag.matmul_inplace(vdag, v_copy.transpose([1, 0]).unwrap()).unwrap();
    udag_u.matmul_inplace(udag_copy.transpose([1, 0]).unwrap(), u).unwrap();
    let mut eye_buff = vec![T::zero(); min_dim * min_dim];
    for i in 0..(min_dim) {
      eye_buff[i * (min_dim + 1)] = T::one();
    }
    let eye = NDArray::from_slice(&eye_buff, [min_dim, min_dim]).unwrap();
    v_vdag.sub_inpl(eye).unwrap();
    udag_u.sub_inpl(eye).unwrap();
    assert!(v_vdag.norm_n_pow_n(2).abs() < acc);
    assert!(udag_u.norm_n_pow_n(2).abs() < acc);
    // Here we check decomposition correctness
    let mut result_buff = vec![T::zero(); m * n];
    let result = NDArray::from_mut_slice(&mut result_buff, [m, n]).unwrap();
    let s_buff: Vec<_> = s_buff.into_iter().map(|x| T::from(x).unwrap()).collect();
    let s = NDArray::from_slice(&s_buff, [1, min_dim]).unwrap();
    let lhs = NDArray::from_mut_slice(&mut buff_u, [m, min_dim]).unwrap();
    let rhs = NDArray::from_slice(&buff_vdag, [min_dim, n]).unwrap();
    lhs.mul_inpl(s).unwrap();
    result.matmul_inplace(lhs, rhs).unwrap();
    result.sub_inpl(a_copy).unwrap();
    assert!(result.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_svd() {
    unsafe {
      _test_svd::<f32>(      (10, 15), 1e-5 );
      _test_svd::<f64>(      (10, 15), 1e-10);
      _test_svd::<Complex32>((10, 15), 1e-5 );
      _test_svd::<Complex64>((10, 15), 1e-10);
      _test_svd::<f32>(      (15, 10), 1e-5 );
      _test_svd::<f64>(      (15, 10), 1e-10);
      _test_svd::<Complex32>((15, 10), 1e-5 );
      _test_svd::<Complex64>((15, 10), 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_qr<T>(size: (usize, usize), acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, n) = size;
    let min_dim = std::cmp::min(m, n);
    let mut buff_a = T::random_normal(m * n);
    let buff_a_copy = buff_a.clone();
    let a = NDArray::from_mut_slice(&mut buff_a, [m, n]).unwrap();
    let a_copy = NDArray::from_slice(&buff_a_copy, [m, n]).unwrap();
    let mut buff_other = T::random_normal(min_dim * min_dim);
    let other = NDArray::from_mut_slice(&mut buff_other, [min_dim, min_dim]).unwrap();
    a.qr(other).unwrap();
    let (q, r) = if m > n { (a, other) } else { (other, a) };
    // Here we check the isometric property of q;
    let eye_buff = T::eye(min_dim);
    let eye = NDArray::from_slice(&eye_buff, [min_dim, min_dim]).unwrap();
    let (mut _buff_q_dag, q_dag) = q.gen_f_array();
    //let q_dag = NDArray::from_mut_slice(&mut buff_q_dag, [m, min_dim]).unwrap();
    q_dag.conj();
    let mut buff_result = T::random_normal(min_dim * min_dim);
    let result = NDArray::from_mut_slice(&mut buff_result, [min_dim, min_dim]).unwrap();
    result.matmul_inplace(q_dag.transpose([1, 0]).unwrap(), q).unwrap();
    result.sub_inpl(eye).unwrap();
    assert!(result.norm_n_pow_n(2).abs() < acc);
    // Here we check the correctness of the decomposition
    let mut result_buff = T::random_normal(m * n);
    let result = NDArray::from_mut_slice(&mut result_buff, [m, n]).unwrap();
    result.matmul_inplace(q, r).unwrap();
    result.sub_inpl(a_copy).unwrap();
    assert!(result.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_qr() 
  {
    unsafe {
      _test_qr::<f32>(      (5, 10), 1e-5 );
      _test_qr::<f64>(      (5, 10), 1e-10);
      _test_qr::<Complex32>((5, 10), 1e-5 );
      _test_qr::<Complex64>((5, 10), 1e-10);
      _test_qr::<f32>(      (10, 5), 1e-5 );
      _test_qr::<f64>(      (10, 5), 1e-10);
      _test_qr::<Complex32>((10, 5), 1e-5 );
      _test_qr::<Complex64>((10, 5), 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_rq<T>(size: (usize, usize), acc: T::Real)
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, n) = size;
    let min_dim = std::cmp::min(m, n);
    let mut buff_a = T::random_normal(m * n);
    let buff_a_copy = buff_a.clone();
    let a = NDArray::from_mut_slice(&mut buff_a, [m, n]).unwrap();
    let a_copy = NDArray::from_slice(&buff_a_copy, [m, n]).unwrap();
    let mut buff_other = T::random_normal(min_dim * min_dim);
    let other = NDArray::from_mut_slice(&mut buff_other, [min_dim, min_dim]).unwrap();
    a.rq(other).unwrap();
    let (r, q) = if m < n { (other, a) } else { (a, other) };
    // Here we check the isometric property of q;
    let eye_buff = T::eye(min_dim);
    let eye = NDArray::from_slice(&eye_buff, [min_dim, min_dim]).unwrap();
    let (mut _buff_q_dag, q_dag) = q.gen_f_array();
    //let q_dag = NDArray::from_mut_slice(&mut buff_q_dag, [m, min_dim]).unwrap();
    q_dag.conj();
    let mut buff_result = T::random_normal(min_dim * min_dim);
    let result = NDArray::from_mut_slice(&mut buff_result, [min_dim, min_dim]).unwrap();
    result.matmul_inplace(q, q_dag.transpose([1, 0]).unwrap()).unwrap();
    result.sub_inpl(eye).unwrap();
    assert!(result.norm_n_pow_n(2).abs() < acc);
    // Here we check the correctness of the decomposition
    let mut result_buff = T::random_normal(m * n);
    let result = NDArray::from_mut_slice(&mut result_buff, [m, n]).unwrap();
    result.matmul_inplace(r, q).unwrap();
    result.sub_inpl(a_copy).unwrap();
    assert!(result.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_rq() {
    unsafe {
      _test_rq::<f32>(      (5, 10), 1e-5);
      _test_rq::<f64>(      (5, 10), 1e-10);
      _test_rq::<Complex32>((5, 10), 1e-5);
      _test_rq::<Complex64>((5, 10), 1e-10);
      _test_rq::<f32>(      (10, 5), 1e-5);
      _test_rq::<f64>(      (10, 5), 1e-10);
      _test_rq::<Complex32>((10, 5), 1e-5);
      _test_rq::<Complex64>((10, 5), 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_rank1_update<T>(
    size: (usize, usize),
    alpha: T,
    acc: T::Real,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, n) = size;
    let mut a_buff = T::random_normal(m * n);
    let a = NDArray::from_mut_slice(&mut a_buff, [m, n]).unwrap();
    let mut a_buff_copy = a_buff.clone();
    let a_copy = NDArray::from_mut_slice(&mut a_buff_copy, [m, n]).unwrap();
    let col_buff = T::random_normal(m);
    let col = NDArray::from_slice(&col_buff, [m, 1]).unwrap();
    let row_buff = T::random_normal(n);
    let row = NDArray::from_slice(&row_buff, [1, n]).unwrap();
    a.rank1_update(col, row, alpha).unwrap();
    let mut aux_buff = vec![T::zero(); m * n];
    let aux = NDArray::from_mut_slice(&mut aux_buff, [m, n]).unwrap();
    aux.add_inpl(col).unwrap();
    aux.mul_inpl(row).unwrap();
    aux.mul_by_scalar(alpha);
    a_copy.add_inpl(aux).unwrap();
    a_copy.sub_inpl(a).unwrap();
    assert!(a_copy.norm_n_pow_n(2).abs() < acc);
  }
  #[test]
  fn test_rank1_update() {
    unsafe {
      _test_rank1_update::<f32>(      (5, 12), 1.2,                       1e-5 );
      _test_rank1_update::<f64>(      (5, 12), 1.2,                       1e-10);
      _test_rank1_update::<Complex32>((5, 12), Complex32::new(1.2, 2.28), 1e-5 );
      _test_rank1_update::<Complex64>((5, 12), Complex64::new(2.28, 4.2), 1e-10);
    }
  }

  #[inline]
  unsafe fn _test_maxvol<T>(
    size: (usize, usize),
    delta: T::Real,
    acc: T::Real,
  )
  where
    T: LinalgComplex,
    T::Real: LinalgReal,
  {
    let (m, n) = size;
    let mut a_buff = T::random_normal(m * n);
    let a_buff_copy = a_buff.clone();
    let a = NDArray::from_mut_slice(&mut a_buff, [m, n]).unwrap();
    let a_copy = NDArray::from_slice(&a_buff_copy, [m, n]).unwrap();
    let new_order = a.maxvol(delta).unwrap();
    let (mut reordered_a_buff, _) = a_copy.transpose([1, 0]).unwrap().gen_f_array_from_axis_order(&new_order[..], 1);
    let reordered_a = NDArray::from_mut_slice(&mut reordered_a_buff, [n, m]).unwrap();
    let (lhs, rhs) = reordered_a.split_across_axis(1, n).unwrap();
    lhs.solve(rhs).unwrap();
    let max_val = rhs.into_f_iter().max_by(|x, y| {
      (*x.0).abs().partial_cmp(&(*y.0).abs()).unwrap()
    });
    assert!((*max_val.unwrap().0).abs() < T::Real::one() + delta, "lhs: {:#?}, rhs: {:#?}", (*max_val.unwrap().0).abs(), T::Real::one() + delta);
    rhs.transpose([1, 0]).unwrap().sub_inpl(a.subarray([n..m, 0..n]).unwrap()).unwrap();
    assert!(rhs.norm_n_pow_n(2).abs() < acc);
  }

  #[test]
  fn test_maxvol() {
    unsafe {
      _test_maxvol::<f32>(      (120, 50), 0.1 , 1e-5 );
      _test_maxvol::<f64>(      (120, 50), 0.1 , 1e-10);
      _test_maxvol::<Complex32>((120, 50), 0.1 , 1e-5 );
      _test_maxvol::<Complex64>((120, 50), 0.1 , 1e-10);
      _test_maxvol::<f32>(      (120, 50), 0.01, 1e-5 );
      _test_maxvol::<f64>(      (120, 50), 0.01, 1e-10);
      _test_maxvol::<Complex32>((120, 50), 0.01, 1e-5 );
      _test_maxvol::<Complex64>((120, 50), 0.01, 1e-10);
      _test_maxvol::<f32>(      (120, 50), 0.  , 1e-5 );
      _test_maxvol::<f64>(      (120, 50), 0.  , 1e-10);
      _test_maxvol::<Complex32>((120, 50), 0.  , 1e-5 );
      _test_maxvol::<Complex64>((120, 50), 0.  , 1e-10);
    }
  }
}
