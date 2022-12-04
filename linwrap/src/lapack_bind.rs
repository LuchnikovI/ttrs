extern crate blis_src;

use std::ffi::c_int;

use num_complex::{Complex64, Complex32};

// TODO?: switch to the expert mode (@gesvx)
macro_rules! gesv {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name (
      n: *const c_int,
      nrhs: *const c_int,
      a: *mut $type_name,
      lda: *const c_int,
      ipiv: *mut c_int,
      b: *mut $type_name,
      ldb: *const c_int,
      info: *mut c_int,
    );
  }
}

extern "C" {
  gesv!(sgesv_, f32      );
  gesv!(dgesv_, f64      );
  gesv!(cgesv_, Complex32);
  gesv!(zgesv_, Complex64);
}