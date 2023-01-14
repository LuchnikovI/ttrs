extern crate blas_src;

use std::ffi::{
  c_char,
  c_int,
};

use num_complex::{Complex64, Complex32};

macro_rules! gemm {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name(
      transa: *const c_char,
      transb: *const c_char,
      m:      *const c_int,
      n:      *const c_int,
      k:      *const c_int,
      alpha:  *const $type_name,
      a:      *const $type_name,
      lda:    *const c_int,
      b:      *const $type_name,
      ldb:    *const c_int,
      beta:   *const $type_name,
      c:      *mut   $type_name,
      ldc:    *const c_int,
    );
  };
}

extern "C" {
  gemm!(sgemm_, f32      );
  gemm!(dgemm_, f64      );
  gemm!(cgemm_, Complex32);
  gemm!(zgemm_, Complex64);
}

macro_rules! ger {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name(
      m:     *const c_int,
      n:     *const c_int,
      alpha: *const $type_name,
      x:     *const $type_name,
      incx:  *const c_int,
      y:     *const $type_name,
      incy:  *const c_int,
      a:     *mut   $type_name,
      lda:   *const c_int,
    );
  };
}

extern "C" {
  ger!(sger_,  f32      );
  ger!(dger_,  f64      );
  ger!(cgeru_, Complex32);
  ger!(zgeru_, Complex64);
}

macro_rules! trsm {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name(
      side:   *const c_char,
      uplo:   *const c_char,
      transa: *const c_char,
      diag:   *const c_char,
      m:      *const c_int,
      n:      *const c_int,
      alpha:  *const $type_name,
      a:      *mut   $type_name,
      lda:    *const c_int,
      b:      *mut   $type_name,
      ldb:    *const c_int,
    );
  };
}

extern "C" {
  trsm!(strsm_, f32      );
  trsm!(dtrsm_, f64      );
  trsm!(ctrsm_, Complex32);
  trsm!(ztrsm_, Complex64);
}
