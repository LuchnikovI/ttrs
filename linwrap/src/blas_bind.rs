extern crate blis_src;

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
      m: *const c_int,
      n: *const c_int,
      k: *const c_int,
      alpha: *const $type_name,
      a: *const $type_name,
      lda: *const c_int,
      b: *const $type_name,
      ldb: *const c_int,
      beta: *const $type_name,
      c: *mut $type_name,
      ldc: *const c_int,
    );
  };
}

extern "C" {
  gemm!(sgemm_, f32      );
  gemm!(dgemm_, f64      );
  gemm!(cgemm_, Complex32);
  gemm!(zgemm_, Complex64);
}
