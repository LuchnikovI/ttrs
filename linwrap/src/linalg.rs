use std::ffi::{
  c_char,
  c_int,
};

use num_complex::{
  Complex32,
  Complex64, Complex
};

use crate::{
  Matrix,
  matrix::{
    MatrixError,
    MatrixResult,
  },
};

// TODO: get advantage of the generalized storage (arbitrary strides).

use crate::par_ptr_wrapper::PointerExtWithDerefAndSend;
use crate::blas_bind::{sgemm_, dgemm_, cgemm_, zgemm_};
use crate::lapack_bind::{sgesv_, dgesv_, cgesv_, zgesv_};
use crate::lapack_bind::{sgesvd_, dgesvd_, cgesvd_, zgesvd_};

macro_rules! impl_matmul {
  ($fn_name:ident, $type_name:ident, $alpha:expr, $beta:expr) => {
    impl Matrix<*mut $type_name, &mut [$type_name]>
    {
      pub fn matmul_inplace<'a, Ptr1, Ref1, Ptr2, Ref2>(
        &mut self,
        a: &Matrix<Ptr1, Ref1>,
        b: &Matrix<Ptr2, Ref2>,
        is_a_transposed: bool,
        is_b_transposed: bool,
      ) -> MatrixResult<()>
      where
        Ptr1: PointerExtWithDerefAndSend<'a, Target = $type_name>,
        Ptr2: PointerExtWithDerefAndSend<'a, Target = $type_name>,
      {
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
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
          $fn_name(&transa, &transb, &m, &n, &k, &alpha, a.ptr.deref(),
                &lda, b.ptr.deref(), &ldb, &beta, self.ptr, &ldc)
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
    impl Matrix<*mut $type_name, &mut [$type_name]>
    {
      pub fn solve(
        &mut self,
        rhs: &mut Matrix<*mut $type_name, &mut [$type_name]>,
      ) -> MatrixResult<()>
        {
          if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
          if rhs.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
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
    impl Matrix<*mut $complex_type_name, &mut [$complex_type_name]> {
      pub fn svd(
        &mut self,
        u: &mut Matrix<*mut $complex_type_name, &mut [$complex_type_name]>,
        vdag: &mut Matrix<*mut $complex_type_name, &mut [$complex_type_name]>,
      ) -> MatrixResult<Vec<$type_name>>
      {
        if self.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if u.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
        if vdag.stride1 != 1 { return Err(MatrixError::FortranLayoutRequired); }
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

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use num_complex::{
    Complex64,
    Complex32,
  };
  use crate::Matrix;
  use crate::par_utils::{
    gen_random_normal_buff_c128,
    gen_random_normal_buff_c64,
    gen_random_normal_buff_f32,
    gen_random_normal_buff_f64,
  };
  use ndarray::Array;
  use ndarray_einsum_beta::einsum;

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
        let mut c = Matrix::from_mut_slice(buff_c.as_mut_slice(), m, n).unwrap();
        c.matmul_inplace(&a, &b, $is_a_transposed, $is_b_transposed).unwrap();
        c.sub(einsum_c).unwrap();
        assert!(c.norm_n_pow_n(2) < 1e-5);
      };
  }

  #[test]
  fn test_matmul_inplace() {
    test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, f32,       gen_random_normal_buff_f32 );
    test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, f64,       gen_random_normal_buff_f64 );
    test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, Complex32, gen_random_normal_buff_c64 );
    test_matmul_inplace!((4, 6, 5), "ik,kj->ij", false, false, Complex64, gen_random_normal_buff_c128);
    test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  f32,       gen_random_normal_buff_f32 );
    test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  f64,       gen_random_normal_buff_f64 );
    test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  Complex32, gen_random_normal_buff_c64 );
    test_matmul_inplace!((4, 6, 5), "ki,kj->ij", false, true,  Complex64, gen_random_normal_buff_c128);
    test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, f32,       gen_random_normal_buff_f32 );
    test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, f64,       gen_random_normal_buff_f64 );
    test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, Complex32, gen_random_normal_buff_c64 );
    test_matmul_inplace!((4, 6, 5), "ik,jk->ij", true,  false, Complex64, gen_random_normal_buff_c128);
    test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  f32,       gen_random_normal_buff_f32 );
    test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  f64,       gen_random_normal_buff_f64 );
    test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  Complex32, gen_random_normal_buff_c64 );
    test_matmul_inplace!((4, 6, 5), "ki,jk->ij", true,  true,  Complex64, gen_random_normal_buff_c128);
  }

  macro_rules! test_solve {
    ($sizes:expr, $type_name:ident, $gen_fn:ident) => {
      let (n, nrhs) = $sizes;
      let mut buff_a = $gen_fn(n * n);
      let a = Matrix::from_slice(buff_a.as_slice(), n, n).unwrap();
      let buff_x = $gen_fn(n * nrhs);
      let x = Matrix::from_slice(buff_x.as_slice(), n, nrhs).unwrap();
      let mut buff_b = $gen_fn(n * nrhs);
      let mut b = Matrix::from_mut_slice(buff_b.as_mut_slice(), n, nrhs).unwrap();
      b.matmul_inplace(&a, &x, false, false).unwrap();
      let mut a = Matrix::from_mut_slice(buff_a.as_mut_slice(), n, n).unwrap();
      a.solve(&mut b).unwrap();
      let mut b = Matrix::from_mut_slice(buff_b.as_mut_slice(), n, nrhs).unwrap();
      b.sub(x).unwrap();
      assert!(b.norm_n_pow_n(2) < 1e-5);
    };
  }

  #[test]
  fn test_inv() {
    test_solve!((10, 15), f32,       gen_random_normal_buff_f32 );
    test_solve!((10, 15), f64,       gen_random_normal_buff_f64 );
    test_solve!((10, 15), Complex32, gen_random_normal_buff_c64 );
    test_solve!((10, 15), Complex64, gen_random_normal_buff_c128);
  }

  macro_rules! test_svd {
    ($sizes:expr, $gen_fn:ident, $complex_init:expr) => {
      let (m, n) = $sizes;
      let min_dim = std::cmp::min(m, n);
      let mut buff_a = $gen_fn(m * n);
      let buff_a_copy = buff_a.clone();
      let mut a = Matrix::from_mut_slice(buff_a.as_mut_slice(), m, n).unwrap();
      let a_copy = Matrix::from_slice(buff_a_copy.as_slice(), m, n).unwrap();
      let mut buff_u = $gen_fn(m * min_dim);
      let mut u = Matrix::from_mut_slice(buff_u.as_mut_slice(), m, min_dim).unwrap();
      let mut buff_vdag = $gen_fn(min_dim * n);
      let mut vdag = Matrix::from_mut_slice(buff_vdag.as_mut_slice(), min_dim, n).unwrap();
      let s_buff = a.svd(&mut u, &mut vdag).unwrap();
      // Here we check that s is non-negative
      assert!(s_buff.iter().all(|x| *x >= 0. ));
      // Here we check isometric property of u and v
      let mut buff_vdag_copy = buff_vdag.clone();
      let mut buff_u_copy = buff_u.clone();
      let mut buff_v_vdag= vec![$complex_init(0., 0.); min_dim * min_dim];
      let mut buff_udag_u= vec![$complex_init(0., 0.); min_dim * min_dim];
      let mut v_copy = Matrix::from_mut_slice(buff_vdag_copy.as_mut_slice(), min_dim, n).unwrap();
      v_copy.conj();
      let mut udag_copy = Matrix::from_mut_slice(buff_u_copy.as_mut_slice(), m, min_dim).unwrap();
      udag_copy.conj();
      let mut v_vdag = Matrix::from_mut_slice(buff_v_vdag.as_mut_slice(), min_dim, min_dim).unwrap();
      let mut udag_u = Matrix::from_mut_slice(buff_udag_u.as_mut_slice(), min_dim, min_dim).unwrap();
      let u = Matrix::from_mut_slice(buff_u.as_mut_slice(), m, min_dim).unwrap();
      let vdag = Matrix::from_mut_slice(buff_vdag.as_mut_slice(), min_dim, n).unwrap();
      v_vdag.matmul_inplace(&vdag, &v_copy, false, true).unwrap();
      udag_u.matmul_inplace(&udag_copy, &u, true, false).unwrap();
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
      let mut result = Matrix::from_mut_slice(&mut result_buff, m, n).unwrap();
      let s_buff: Vec<_> = s_buff.into_iter().map(|x| $complex_init(x, 0.)).collect();
      let s = Matrix::from_slice(&s_buff, 1, min_dim).unwrap();
      let mut lhs = Matrix::from_mut_slice(&mut buff_u, m, min_dim).unwrap();
      let rhs = Matrix::from_slice(&buff_vdag, min_dim, n).unwrap();
      lhs.mul(s).unwrap();
      result.matmul_inplace(&lhs, &rhs, false, false).unwrap();
      result.sub(a_copy).unwrap();
      assert!(result.norm_n_pow_n(2) < 1e-5);
    };
  }

  #[test]
  fn test_svd() {
    test_svd!((10, 15), gen_random_normal_buff_f32,  |x, _| x as f32            );
    test_svd!((10, 15), gen_random_normal_buff_f64,  |x, _| x as f64            );
    test_svd!((10, 15), gen_random_normal_buff_c64,  |x, y| Complex32::new(x, y));
    test_svd!((10, 15), gen_random_normal_buff_c128, |x, y| Complex64::new(x, y));
    test_svd!((15, 10), gen_random_normal_buff_f32,  |x, _| x as f32            );
    test_svd!((15, 10), gen_random_normal_buff_f64,  |x, _| x as f64            );
    test_svd!((15, 10), gen_random_normal_buff_c64,  |x, y| Complex32::new(x, y));
    test_svd!((15, 10), gen_random_normal_buff_c128, |x, y| Complex64::new(x, y));
  }
}
