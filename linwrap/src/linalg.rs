use std::ffi::{
  c_char,
  c_int,
};

use num_complex::{
  Complex32,
  Complex64,
  Complex,
};

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IndexedParallelIterator;

use crate::{
  Matrix,
  matrix::{
    MatrixError,
    MatrixResult,
  },
  linalg_utils::triangular_split,
};

use num_complex::ComplexFloat;

// TODO: get advantage of the generalized storage (arbitrary strides).

use crate::blas_bind::{sgemm_, dgemm_, cgemm_, zgemm_};
use crate::blas_bind::{sger_, dger_, cgeru_, zgeru_};
use crate::lapack_bind::{sgesv_, dgesv_, cgesv_, zgesv_};
use crate::lapack_bind::{sgesvd_, dgesvd_, cgesvd_, zgesvd_};
use crate::lapack_bind::{sgeqrf_, dgeqrf_, cgeqrf_, zgeqrf_};
use crate::lapack_bind::{sorgqr_, dorgqr_, cungqr_, zungqr_};

macro_rules! impl_matmul {
  ($fn_name:ident, $type_name:ident, $alpha:expr, $beta:expr) => {
    impl Matrix<*mut $type_name>
    {
      pub unsafe fn matmul_inplace(
        self,
        a: impl Into<Matrix<*const $type_name>>,
        b: impl Into<Matrix<*const $type_name>>,
        is_a_transposed: bool,
        is_b_transposed: bool,
      ) -> MatrixResult<()>
      {
        let a = a.into();
        let b = b.into();
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if self.stride2 < self.nrows { return Err(MatrixError::MutableElementsOverlapping); }
        if a.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if b.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        let (m, k) = if is_a_transposed {
          (a.ncols as c_int, a.nrows as c_int)
        } else {
          (a.nrows as c_int, a.ncols as c_int)
        };
        let n = if is_b_transposed {
          if b.ncols as c_int != k { return Err(MatrixError::IncorrectShape) }
          b.nrows as c_int
        } else {
          if b.nrows as c_int != k { return Err(MatrixError::IncorrectShape) }
          b.ncols as c_int
        };
        let transa = if is_a_transposed { 'T' as c_char } else { 'N' as c_char };
        let transb = if is_b_transposed { 'T' as c_char } else { 'N' as c_char };
        if (m != self.nrows as c_int) && (n != self.ncols as c_int) { return Err(MatrixError::IncorrectShape); }
        let alpha = $alpha;
        let beta = $beta;
        let lda = a.stride2 as c_int;
        let ldb = b.stride2 as c_int;
        let ldc = self.stride2 as c_int;
        unsafe {
          $fn_name(&transa, &transb, &m, &n, &k, &alpha, a.ptr,
                &lda, b.ptr, &ldb, &beta, self.ptr, &ldc)
        };
        Ok(())
      }
    }
  };
}

impl_matmul!(sgemm_, f32, 1., 0.);
impl_matmul!(dgemm_, f64, 1., 0.);
impl_matmul!(cgemm_, Complex32, Complex::new(1., 0.), Complex::new(0., 0.));
impl_matmul!(zgemm_, Complex64, Complex::new(1., 0.), Complex::new(0., 0.));

macro_rules! impl_solve {
  ($fn_name:ident, $type_name:ident) => {
    impl Matrix<*mut $type_name>
    {
      pub unsafe fn solve(
        self,
        rhs: Matrix<*mut $type_name>,
      ) -> MatrixResult<()>
        {
          if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
          if self.stride2 < self.nrows { return Err(MatrixError::MutableElementsOverlapping); }
          if rhs.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
          if rhs.stride2 < rhs.nrows { return Err(MatrixError::MutableElementsOverlapping); }
          if self.ncols != self.nrows { return Err(MatrixError::IncorrectShape); }
          let n = self.ncols as c_int;
          if rhs.nrows as c_int != n { return Err(MatrixError::IncorrectShape); }
          let nrhs = rhs.ncols as c_int;
          let lda = self.stride2 as c_int;
          let ldb = rhs.stride2 as c_int;
          let mut info: c_int = 0;
          let mut ipiv_buff = Vec::with_capacity(self.ncols);
          unsafe { ipiv_buff.set_len(self.ncols); }
          let ipiv = ipiv_buff.as_mut_ptr();
          unsafe { $fn_name
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
          if info != 0 { return Err(MatrixError::LapackError(info)); }
          Ok(())
        }
    }
  }
}

impl_solve!(sgesv_, f32      );
impl_solve!(dgesv_, f64      );
impl_solve!(cgesv_, Complex32);
impl_solve!(zgesv_, Complex64);

macro_rules! impl_svd {
  ($fn_name:ident, $type_name:ident, $complex_type_name:ident, $complex_zero:expr, $complex_to_real_fn:expr) => {
    impl Matrix<*mut $complex_type_name> {
      pub unsafe fn svd(
        self,
        u: Matrix<*mut $complex_type_name>,
        vdag: Matrix<*mut $complex_type_name>,
      ) -> MatrixResult<Vec<$type_name>>
      {
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if self.stride2 < self.nrows { return Err(MatrixError::MutableElementsOverlapping); }
        if u.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if u.stride2 < u.nrows { return Err(MatrixError::MutableElementsOverlapping); }
        if vdag.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if vdag.stride2 < vdag.nrows { return Err(MatrixError::MutableElementsOverlapping); }
        let jobu = 'S' as c_char;
        let jobvt = 'S' as c_char;
        let m = self.nrows as c_int;
        let n = self.ncols as c_int;
        let lda = self.stride2 as c_int;
        let ldu = u.stride2 as c_int;
        let ldvt = vdag.stride2 as c_int;
        let min_dim = std::cmp::min(n, m);
        if (u.nrows as c_int != m) && (u.ncols as c_int != min_dim) { return Err(MatrixError::IncorrectShape); }
        if (vdag.ncols as c_int != n) && (vdag.nrows as c_int != min_dim) { return Err(MatrixError::IncorrectShape); }
        let mut s: Vec<$type_name> = Vec::with_capacity(min_dim as usize);
        unsafe { s.set_len(min_dim as usize); }
        let lwork = -1  as c_int;
        let mut work = $complex_zero;
        let mut rwork = Vec::with_capacity(5 * min_dim as usize);
        unsafe { rwork.set_len(5 * min_dim as usize); }
        let mut info = 0;
        // worksapce query
        unsafe { $fn_name(
          &jobu,
          &jobvt,
          &m,
          &n,
          self.ptr,
          &lda,
          s.as_mut_ptr(),
          u.ptr,
          &ldu,
          vdag.ptr,
          &ldvt,
          &mut work,
          &lwork,
          rwork.as_mut_ptr(),
          &mut info) }
        let lwork = $complex_to_real_fn(work) as c_int;
        let mut work: Vec<$complex_type_name> = Vec::with_capacity(lwork as usize);
        unsafe { $fn_name(
          &jobu,
          &jobvt,
          &m,
          &n,
          self.ptr,
          &lda,
          s.as_mut_ptr(),
          u.ptr,
          &ldu,
          vdag.ptr,
          &ldvt,
          work.as_mut_ptr(),
          &lwork,
          rwork.as_mut_ptr(),
          &mut info) }
        if info != 0 { return Err(MatrixError::LapackError(info)); }
        Ok(s)
      }
    }
  };
}

impl_svd!(sgesvd_, f32, f32      , 0.                    , |x| x              );
impl_svd!(dgesvd_, f64, f64      , 0.                    , |x| x              );
impl_svd!(cgesvd_, f32, Complex32, Complex32::new(0., 0.), |x: Complex32| x.re);
impl_svd!(zgesvd_, f64, Complex64, Complex64::new(0., 0.), |x: Complex64| x.re);

macro_rules! impl_householder {
  ($fn_name:ident, $type_name:ident, $complex_zero:expr, $complex_to_real_fn:expr) => {
    impl Matrix<*mut $type_name> {
      unsafe fn householder_(&mut self, tau: *mut $type_name) -> MatrixResult<()> {
        let m = self.nrows as c_int;
        let n = self.ncols as c_int;
        let lda = self.stride2 as c_int;
        let mut work = $complex_zero;
        let lwork = -1 as c_int;
        let mut info: c_int = 0;
        $fn_name(
          &m,
          &n,
          self.ptr,
          &lda,
          tau,
          &mut work,
          &lwork,
          &mut info,
        );
        let lwork = $complex_to_real_fn(work) as c_int;
        let mut work_buff: Vec<$type_name> = Vec::with_capacity(lwork as usize);
        unsafe { work_buff.set_len(lwork as usize); }
        let work = work_buff.as_mut_ptr();
        $fn_name(
          &m,
          &n,
          self.ptr,
          &lda,
          tau,
          work,
          &lwork,
          &mut info,
        );
        if info != 0 { return Err(MatrixError::LapackError(info)); }
        Ok(())
      }
    }
  };
}

impl_householder!(sgeqrf_, f32,       0.,                   |x| x              );
impl_householder!(dgeqrf_, f64,       0.,                   |x| x              );
impl_householder!(cgeqrf_, Complex32, Complex::new(0., 0.), |x: Complex32| x.re);
impl_householder!(zgeqrf_, Complex64, Complex::new(0., 0.), |x: Complex64| x.re);

macro_rules! impl_householder_to_q {
  ($fn_name:ident, $type_name:ident, $complex_zero:expr, $complex_to_real_fn:expr) => {
    impl Matrix<*mut $type_name> {
      unsafe fn householder_to_q_(self, tau: *mut $type_name) -> MatrixResult<()> {
        let m = self.nrows as c_int;
        let n = self.ncols as c_int;
        let k = std::cmp::min(m, n);
        let lda = self.stride2 as c_int;
        let mut work = $complex_zero;
        let lwork = -1 as c_int;
        let mut info: c_int = 0;
        $fn_name(
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
        let lwork = $complex_to_real_fn(work) as c_int;
        let mut work_buff: Vec<$type_name> = Vec::with_capacity(lwork as usize);
        unsafe { work_buff.set_len(lwork as usize); }
        let work = work_buff.as_mut_ptr();
        $fn_name(
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
        if info != 0 { return Err(MatrixError::LapackError(info)); }
        Ok(())
      }
    } 
  };
}

impl_householder_to_q!(sorgqr_, f32,       0.,                     |x| x              );
impl_householder_to_q!(dorgqr_, f64,       0.,                     |x| x              );
impl_householder_to_q!(cungqr_, Complex32, Complex32::new(0., 0.), |x: Complex32| x.re);
impl_householder_to_q!(zungqr_, Complex64, Complex64::new(0., 0.), |x: Complex64| x.re);

macro_rules! impl_qr {
  ($type_name:ident) => {
    impl Matrix<*mut $type_name> {
      pub unsafe fn qr(
        mut self,
        other: Self,
      ) -> MatrixResult<()>
      {
        let m = self.nrows;
        let n = self.ncols;
        let min_dim = std::cmp::min(n, m);
        if other.ncols != min_dim { return Err(MatrixError::IncorrectShape); }
        if other.nrows != min_dim { return Err(MatrixError::IncorrectShape); }
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
    }
  };
}

impl_qr!(f32      );
impl_qr!(f64      );
impl_qr!(Complex32);
impl_qr!(Complex64);

macro_rules! impl_rank1_update {
  ($fn_name:ident, $type_name:ident) => {
    impl Matrix<*mut $type_name> {
      unsafe fn rank1_update(
        self,
        col: impl Into<Matrix<*const $type_name>>,
        row: impl Into<Matrix<*const $type_name>>,
        alpha: $type_name,
      ) -> MatrixResult<()>
      {
        let col = col.into();
        let row = row.into();
        if col.ncols != 1 { return Err(MatrixError::IncorrectShape); }
        if row.nrows != 1 { return Err(MatrixError::IncorrectShape); }
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if self.stride2 < self.nrows { return Err(MatrixError::MutableElementsOverlapping); }
        let m = self.nrows as c_int;
        let n = self.ncols as c_int;
        if col.nrows != m as usize { return Err(MatrixError::IncorrectShape); }
        if row.ncols != n as usize { return Err(MatrixError::IncorrectShape); }
        let incx = col.stride1 as c_int;
        let incy = row.stride2 as c_int;
        let lda = self.stride2 as c_int;
        $fn_name(&m, &n, &alpha, col.ptr, &incx, row.ptr, &incy, self.ptr, &lda);
        Ok(())
      }
    }
  };
}

impl_rank1_update!(sger_,  f32      );
impl_rank1_update!(dger_,  f64      );
impl_rank1_update!(cgeru_, Complex32);
impl_rank1_update!(zgeru_, Complex64);

macro_rules! impl_maxvol {
  ($complex_type_name:ident, $type_name:ident, $complex_one:expr, $complex_zero:expr) => {
    impl Matrix<*mut $complex_type_name> {
      pub unsafe fn maxvol(self, delta: $type_name) -> MatrixResult<Vec<usize>> {
        let m = self.nrows;
        let n = self.ncols;
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if self.stride2 < m { return Err(MatrixError::MutableElementsOverlapping); }
        if m > n { return Err(MatrixError::IncorrectShape); }
        let mut order: Vec<usize> = (0..n).collect();
        let mut x_buff: Vec<$complex_type_name> = Vec::with_capacity(m);
        unsafe { x_buff.set_len(m); }
        let x = Matrix::from_mut_slice(&mut x_buff, m, 1)?;
        let mut y_buff: Vec<$complex_type_name> = Vec::with_capacity(n - m);
        unsafe { y_buff.set_len(n - m); }
        let y = Matrix::from_mut_slice(&mut y_buff, 1, n - m)?;
        let (a, b) = self.col_split(m).unwrap();
        unsafe { a.solve(b) }?;
        unsafe { (0..(m * m)).into_par_iter().zip(a.into_par_iter()).for_each(|(i, x)| {
          if i % (m + 1) == 0 { *x.0 = $complex_one } else { *x.0 = $complex_zero }
        }) };
        let mut val;
        let mut row_num;
        let mut col_num;
        loop {
          (val, row_num, col_num) = b.argmax();
          if val.abs() < delta + 1. { break; }
          let bij = *b.at((row_num, col_num))?;
          let col = b.subview((0..m, col_num..(col_num + 1)))?;
          let row = b.subview((row_num..(row_num + 1), 0..(n - m)))?;
          col.write_to(x, false)?;
          row.write_to(y, false)?;
          *x.at((row_num, 0))? -= $complex_one;
          *y.at((0, col_num))? += $complex_one;
          b.rank1_update(x, y, -$complex_one / bij)?;
          order.swap(row_num, col_num + m);
        }
        Ok(order)
      }
    }
  };
}

impl_maxvol!(f32,       f32, 1.                    ,                     0.);
impl_maxvol!(f64,       f64, 1.                    ,                     0.);
impl_maxvol!(Complex32, f32, Complex32::new(1., 0.), Complex32::new(0., 0.));
impl_maxvol!(Complex64, f64, Complex64::new(1., 0.), Complex64::new(0., 0.));

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
use num_complex::{
    Complex64,
    Complex32,
    ComplexFloat,
  };
  use crate::Matrix;
  use crate::init_utils::{
    random_normal_f32,
    random_normal_f64,
    random_normal_c32,
    random_normal_c64,
    eye_f32,
    eye_f64,
    eye_c32,
    eye_c64,
  };
  use ndarray::Array;
  use ndarray_einsum_beta::einsum;
  use rayon::iter::ParallelIterator;

  macro_rules! test_matmul_inplace {
      ($sizes:expr, $einsum_str:expr, $is_a_transposed:expr, $is_b_transposed:expr, $type_name:ident, $gen_fn:ident) => {
        let (m, k, n) = $sizes;
        let buff_a = $gen_fn(k * m);
        let buff_a = Array::from_shape_vec(if $is_a_transposed { [m, k] } else { [k, m] }, buff_a).unwrap();
        let buff_b = $gen_fn(n * k);
        let buff_b = Array::from_shape_vec(if $is_b_transposed { [k, n] } else { [n, k] }, buff_b).unwrap();
        let einsum_c = einsum($einsum_str, &[&buff_b, &buff_a]).unwrap();
        let einsum_c = einsum_c.iter().map(|x| *x).collect::<Vec<_>>();
        let einsum_c = Matrix::from_slice(&einsum_c, m, n).unwrap();

        let a = Matrix::from_slice(buff_a.as_slice_memory_order().unwrap(), k, m).unwrap();
        let a = if $is_a_transposed { a } else { a.reshape(m, k).unwrap() };
        let b = Matrix::from_slice(buff_b.as_slice_memory_order().unwrap(), n, k).unwrap();
        let b = if $is_b_transposed { b } else { b.reshape(k, n).unwrap() };
        let mut buff_c: Vec<$type_name> = vec![Default::default(); m * n];
        let c = Matrix::from_mut_slice(buff_c.as_mut_slice(), m, n).unwrap();
        c.matmul_inplace(a, b, $is_a_transposed, $is_b_transposed).unwrap();
        c.sub(einsum_c).unwrap();
        assert!(c.norm_n_pow_n(2) < 1e-5);
      };
  }

  #[test]
  fn test_matmul_inplace() {
    unsafe {
      test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, f32,       random_normal_f32 );
      test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, f64,       random_normal_f64 );
      test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, Complex32, random_normal_c32 );
      test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, Complex64, random_normal_c64);
      test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  f32,       random_normal_f32 );
      test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  f64,       random_normal_f64 );
      test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  Complex32, random_normal_c32 );
      test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  Complex64, random_normal_c64);
      test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, f32,       random_normal_f32 );
      test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, f64,       random_normal_f64 );
      test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, Complex32, random_normal_c32 );
      test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, Complex64, random_normal_c64);
      test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  f32,       random_normal_f32 );
      test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  f64,       random_normal_f64 );
      test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  Complex32, random_normal_c32 );
      test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  Complex64, random_normal_c64);
    }
  }

  macro_rules! test_solve {
    ($sizes:expr, $type_name:ident, $gen_fn:ident) => {
      let (n, nrhs) = $sizes;
      let mut buff_a = $gen_fn(n * n);
      let a = Matrix::from_slice(buff_a.as_slice(), n, n).unwrap();
      let buff_x = $gen_fn(n * nrhs);
      let x = Matrix::from_slice(buff_x.as_slice(), n, nrhs).unwrap();
      let mut buff_b = $gen_fn(n * nrhs);
      let b = Matrix::from_mut_slice(buff_b.as_mut_slice(), n, nrhs).unwrap();
      b.matmul_inplace(a, x, false, false).unwrap();
      let a = Matrix::from_mut_slice(buff_a.as_mut_slice(), n, n).unwrap();
      a.solve(b).unwrap();
      let b = Matrix::from_mut_slice(buff_b.as_mut_slice(), n, nrhs).unwrap();
      b.sub(x).unwrap();
      assert!(b.norm_n_pow_n(2) < 1e-5);
    };
  }

  #[test]
  fn test_inv() {
    unsafe {
      test_solve!((10, 15), f32,       random_normal_f32 );
      test_solve!((10, 15), f64,       random_normal_f64 );
      test_solve!((10, 15), Complex32, random_normal_c32 );
      test_solve!((10, 15), Complex64, random_normal_c64);
    }
  }

  macro_rules! test_svd {
    ($sizes:expr, $gen_fn:ident, $complex_init:expr) => {
      let (m, n) = $sizes;
      let min_dim = std::cmp::min(m, n);
      let mut buff_a = $gen_fn(m * n);
      let buff_a_copy = buff_a.clone();
      let a = Matrix::from_mut_slice(buff_a.as_mut_slice(), m, n).unwrap();
      let a_copy = Matrix::from_slice(buff_a_copy.as_slice(), m, n).unwrap();
      let mut buff_u = $gen_fn(m * min_dim);
      let u = Matrix::from_mut_slice(buff_u.as_mut_slice(), m, min_dim).unwrap();
      let mut buff_vdag = $gen_fn(min_dim * n);
      let vdag = Matrix::from_mut_slice(buff_vdag.as_mut_slice(), min_dim, n).unwrap();
      let s_buff = a.svd(u, vdag).unwrap();
      // Here we check that s is non-negative
      assert!(s_buff.iter().all(|x| *x >= 0. ));
      // Here we check isometric property of u and v
      let buff_vdag_copy: Vec<_> = buff_vdag.iter().map(|x| { x.conj() }).collect();
      let buff_u_copy: Vec<_> = buff_u.iter().map(|x| { x.conj() }).collect();
      let mut buff_v_vdag= vec![$complex_init(0., 0.); min_dim * min_dim];
      let mut buff_udag_u= vec![$complex_init(0., 0.); min_dim * min_dim];
      let v_copy = Matrix::from_slice(buff_vdag_copy.as_slice(), min_dim, n).unwrap();
      let udag_copy = Matrix::from_slice(buff_u_copy.as_slice(), m, min_dim).unwrap();
      let v_vdag = Matrix::from_mut_slice(buff_v_vdag.as_mut_slice(), min_dim, min_dim).unwrap();
      let udag_u = Matrix::from_mut_slice(buff_udag_u.as_mut_slice(), min_dim, min_dim).unwrap();
      let u = Matrix::from_slice(buff_u.as_slice(), m, min_dim).unwrap();
      let vdag = Matrix::from_slice(buff_vdag.as_slice(), min_dim, n).unwrap();
      v_vdag.matmul_inplace(vdag, v_copy, false, true).unwrap();
      udag_u.matmul_inplace(udag_copy, u, true, false).unwrap();
      let mut eye_buff = vec![$complex_init(0., 0.); min_dim * min_dim];
      for i in 0..(min_dim) {
        eye_buff[i * (min_dim + 1)] = $complex_init(1., 0.);
      }
      let eye = Matrix::from_slice(&eye_buff, min_dim, min_dim).unwrap();
      v_vdag.sub(eye).unwrap();
      udag_u.sub(eye).unwrap();
      assert!(v_vdag.norm_n_pow_n(2) < 1e-5);
      assert!(udag_u.norm_n_pow_n(2) < 1e-5);
      // Here we check decomposition correctness
      let mut result_buff = vec![$complex_init(0., 0.); m * n];
      let result = Matrix::from_mut_slice(&mut result_buff, m, n).unwrap();
      let s_buff: Vec<_> = s_buff.into_iter().map(|x| $complex_init(x, 0.)).collect();
      let s = Matrix::from_slice(&s_buff, 1, min_dim).unwrap();
      let lhs = Matrix::from_mut_slice(&mut buff_u, m, min_dim).unwrap();
      let rhs = Matrix::from_slice(&buff_vdag, min_dim, n).unwrap();
      lhs.mul(s).unwrap();
      result.matmul_inplace(lhs, rhs, false, false).unwrap();
      result.sub(a_copy).unwrap();
      assert!(result.norm_n_pow_n(2) < 1e-5);
    };
  }

  #[test]
  fn test_svd() {
    unsafe {
      test_svd!((10, 15), random_normal_f32,  |x, _| x as f32            );
      test_svd!((10, 15), random_normal_f64,  |x, _| x as f64            );
      test_svd!((10, 15), random_normal_c32,  |x, y| Complex32::new(x, y));
      test_svd!((10, 15), random_normal_c64, |x, y| Complex64::new(x, y));
      test_svd!((15, 10), random_normal_f32,  |x, _| x as f32            );
      test_svd!((15, 10), random_normal_f64,  |x, _| x as f64            );
      test_svd!((15, 10), random_normal_c32,  |x, y| Complex32::new(x, y));
      test_svd!((15, 10), random_normal_c64, |x, y| Complex64::new(x, y));
    }
  }

  macro_rules! test_qr {
    ($sizes:expr, $gen_fn:ident, $eye_fn:ident) => {
      let (m, n) = $sizes;
      let min_dim = std::cmp::min(m, n);
      let mut buff_a = $gen_fn(m * n);
      let buff_a_copy = buff_a.clone();
      let a = Matrix::from_mut_slice(&mut buff_a, m, n).unwrap();
      let a_copy = Matrix::from_slice(&buff_a_copy, m, n).unwrap();
      let mut buff_other = $gen_fn(min_dim * min_dim);
      let other = Matrix::from_mut_slice(&mut buff_other, min_dim, min_dim).unwrap();
      a.qr(other).unwrap();
      let (q, r) = if m > n { (a, other) } else { (other, a) };
      // Here we check the isometric property of q;
      let eye_buff = $eye_fn(min_dim);
      let eye = Matrix::from_slice(&eye_buff, min_dim, min_dim).unwrap();
      let mut buff_q_dag = q.gen_buffer();
      let q_dag = Matrix::from_mut_slice(&mut buff_q_dag, m, min_dim).unwrap();
      q_dag.conj();
      let mut buff_result = $gen_fn(min_dim * min_dim);
      let result = Matrix::from_mut_slice(&mut buff_result, min_dim, min_dim).unwrap();
      result.matmul_inplace(q_dag, q, true, false).unwrap();
      result.sub(eye).unwrap();
      assert!(result.norm_n_pow_n(2) < 1e-5);
      // Here we check the correctness of the decomposition
      let mut result_buff = $gen_fn(m * n);
      let result = Matrix::from_mut_slice(&mut result_buff, m, n).unwrap();
      result.matmul_inplace(q, r, false, false).unwrap();
      result.sub(a_copy).unwrap();
      assert!(result.norm_n_pow_n(2) < 1e-5);
    };
  }

  #[test]
  fn test_qr() {
    unsafe {
      test_qr!((5, 10), random_normal_f32, eye_f32);
      test_qr!((5, 10), random_normal_f64, eye_f64);
      test_qr!((5, 10), random_normal_c32, eye_c32);
      test_qr!((5, 10), random_normal_c64, eye_c64);
      test_qr!((10, 5), random_normal_f32, eye_f32);
      test_qr!((10, 5), random_normal_f64, eye_f64);
      test_qr!((10, 5), random_normal_c32, eye_c32);
      test_qr!((10, 5), random_normal_c64, eye_c64);
    }
  }

  macro_rules! test_rank1_update {
    ($sizes:expr, $gen_fn:ident, $alpha:expr, $zero:expr) => {
      let (m, n) = $sizes;
      let mut a_buff = $gen_fn(m * n);
      let a = Matrix::from_mut_slice(&mut a_buff, m, n).unwrap();
      let mut a_buff_copy = a_buff.clone();
      let a_copy = Matrix::from_mut_slice(&mut a_buff_copy, m, n).unwrap();
      let col_buff = $gen_fn(m);
      let col = Matrix::from_slice(&col_buff, m, 1).unwrap();
      let row_buff = $gen_fn(n);
      let row = Matrix::from_slice(&row_buff, 1, n).unwrap();
      let alpha = $alpha;
      a.rank1_update(col, row, alpha).unwrap();
      let mut aux_buff = vec![$zero; m * n];
      let aux = Matrix::from_mut_slice(&mut aux_buff, m, n).unwrap();
      aux.add(col).unwrap();
      aux.mul(row).unwrap();
      aux.mul_by_scalar(alpha);
      a_copy.add(aux).unwrap();
      a_copy.sub(a).unwrap();
      assert!(a_copy.norm_n_pow_n(2) < 1e-5);
    };
  }
  #[test]
  fn test_rank1_update() {
    unsafe {
      test_rank1_update!((5, 12), random_normal_f32, 1.2,                       0.                    );
      test_rank1_update!((5, 12), random_normal_f64, 1.2,                       0.                    );
      test_rank1_update!((5, 12), random_normal_c32, Complex32::new(1.2, 0.98), Complex32::new(0., 0.));
      test_rank1_update!((5, 12), random_normal_c64, Complex64::new(1.2, 0.98), Complex64::new(0., 0.));
    }
  }

  macro_rules! test_maxvol {
    ($sizes:expr, $gen_fn:ident, $delta:expr) => {
      let (m, n) = $sizes;
      let mut a_buff = $gen_fn(m * n);
      let a_buff_copy = a_buff.clone();
      let a = Matrix::from_mut_slice(&mut a_buff, m, n).unwrap();
      let a_copy = Matrix::from_slice(&a_buff_copy, m, n).unwrap();
      let new_order = a.maxvol($delta).unwrap();
      let mut reordered_a_buff = a_copy.gen_from_cols_order(&new_order[..], false);
      let reordered_a = Matrix::from_mut_slice(&mut reordered_a_buff, m, n).unwrap();
      let (lhs, rhs) = reordered_a.col_split(m).unwrap();
      lhs.solve(rhs).unwrap();
      let max_val = rhs.into_par_iter().max_by(|x, y| {
        (*x.0).abs().partial_cmp(&(*y.0).abs()).unwrap()
      });
      assert!((*max_val.unwrap().0).abs() < 1. + $delta);
    };
  }

  #[test]
  fn test_maxvol() {
    unsafe {
      test_maxvol!((50, 120), random_normal_f32, 0.1 );
      test_maxvol!((50, 120), random_normal_f64, 0.1 );
      test_maxvol!((50, 120), random_normal_c32, 0.1 );
      test_maxvol!((50, 120), random_normal_c64, 0.1 );
      test_maxvol!((50, 120), random_normal_f32, 0.01);
      test_maxvol!((50, 120), random_normal_f64, 0.01);
      test_maxvol!((50, 120), random_normal_c32, 0.01);
      test_maxvol!((50, 120), random_normal_c64, 0.01);
      test_maxvol!((50, 120), random_normal_f32, 0.  );
      test_maxvol!((50, 120), random_normal_f64, 0.  );
      test_maxvol!((50, 120), random_normal_c32, 0.  );
      test_maxvol!((50, 120), random_normal_c64, 0.  );
    }
  }
}
