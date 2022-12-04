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
use crate::par_ptr_wrapper::PointerExtWithDerefAndSend;
use crate::blas_bind::{sgemm_, dgemm_, cgemm_, zgemm_};
use crate::lapack_bind::{sgesv_, dgesv_, cgesv_, zgesv_};

macro_rules! impl_matmul {
  ($fn_name:ident, $type_name:ident, $alpha:expr, $beta:expr) => {
    impl Matrix<*mut $type_name, &mut [$type_name]>
    {
      pub fn matmul_inplace<'a, Ptr1, Ref1, Ptr2, Ref2>(
        &mut self,
        a: Matrix<Ptr1, Ref1>,
        b: Matrix<Ptr2, Ref2>,
        is_a_transposed: bool,
        is_b_transposed: bool,
      ) -> MatrixResult<()>
      where
        Ptr1: PointerExtWithDerefAndSend<'a, Target = $type_name>,
        Ptr2: PointerExtWithDerefAndSend<'a, Target = $type_name>,
      {
        let transa = if is_a_transposed { 'T' as c_char } else { 'N' as c_char };
        let transb = if is_b_transposed { 'T' as c_char } else { 'N' as c_char };
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
        if (m != self.nrows as c_int) && (n != self.ncols as c_int) { return Err(MatrixError::IncorrectShape); }
        let alpha = $alpha;
        let beta = $beta;
        let lda = a.ld as c_int;
        let ldb = b.ld as c_int;
        let ldc = self.ld as c_int;
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
        rhs: Matrix<*mut $type_name, &mut [$type_name]>,
      ) -> MatrixResult<()>
        {
          if self.ncols != self.nrows { return Err(MatrixError::IncorrectShape); }
          let n = self.ncols as c_int;
          if rhs.nrows as c_int != n { return Err(MatrixError::IncorrectShape); }
          let nrhs = rhs.ncols as c_int;
          let lda = self.ld as c_int;
          let ldb = rhs.ld as c_int;
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
          Ok(())
        }
    }
  }
}

impl_solve!(sgesv_, f32      );
impl_solve!(dgesv_, f64      );
impl_solve!(cgesv_, Complex32);
impl_solve!(zgesv_, Complex64);

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
        let einsum_c: Matrix<_, _> = einsum_c.as_slice().into();
        let einsum_c = einsum_c.reshape(m, n).unwrap();

        let a: Matrix<_, _> = buff_a.as_slice_memory_order().unwrap().into();
        let a = if $is_a_transposed { a.reshape(k, m) } else { a.reshape(m, k) }.unwrap();
        let b: Matrix<_, _> = buff_b.as_slice_memory_order().unwrap().into();
        let b = if $is_b_transposed { b.reshape(n, k) } else { b.reshape(k, n) }.unwrap();
        let mut buff_c: Vec<$type_name> = vec![Default::default(); m * n];
        let c: Matrix<_, _> = buff_c.as_mut_slice().into();
        let mut c = c.reshape(m, n).unwrap();
        c.matmul_inplace(a, b, $is_a_transposed, $is_b_transposed).unwrap();
        c.sub(einsum_c).unwrap();
        assert!(c.norm_n_pow_n(2) < 1e-10);
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

  /*#[test]
  fn test_inv() {
    let n = 6;
    let nrhs = 15;
    let mut buff_a = gen_random_normal_buff_c128(n * n);
    let a: Matrix<_, _> = buff_a.as_slice().into();
    let a = a.reshape(n, n).unwrap();
    let buff_x = gen_random_normal_buff_c128(n * nrhs);
    let x: Matrix<_, _> = buff_x.as_slice().into();
    let x = x.reshape(n, nrhs).unwrap();
    let mut buff_b = gen_random_normal_buff_c128(n * nrhs);
    let b: Matrix<_, _> = buff_b.as_mut_slice().into();
    let mut b = b.reshape(n, nrhs).unwrap();
    b.matmul_inplace(a, x, false, false).unwrap();
    let a: Matrix<_, _> = buff_a.as_mut_slice().into();
    let mut a = a.reshape(n, n).unwrap();
    a.solve(b).unwrap();
  }*/
}
