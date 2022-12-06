extern crate lapack_src;

use std::ffi::{
  c_int,
  c_char,
};

use num_complex::{Complex64, Complex32};

// TODO?: switch to the expert mode (@gesvx)
macro_rules! gesv {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name (
      n:    *const c_int,
      nrhs: *const c_int,
      a:    *mut   $type_name,
      lda:  *const c_int,
      ipiv: *mut   c_int,
      b:    *mut   $type_name,
      ldb:  *const c_int,
      info: *mut   c_int,
    );
  }
}

extern "C" {
  gesv!(sgesv_, f32      );
  gesv!(dgesv_, f64      );
  gesv!(cgesv_, Complex32);
  gesv!(zgesv_, Complex64);
}

macro_rules! gesvd {
  ($fn_name:ident, $type_name:ident, $complex_type_name:ident) => {
    pub(super) fn $fn_name(
      jobu:  *const c_char,
      jobvt: *const c_char,
      m:     *const c_int,
      n:     *const c_int,
      a:     *mut   $complex_type_name,
      lda:   *const c_int,
      s:     *mut   $type_name,
      u:     *mut   $complex_type_name,
      ldu:   *const c_int,
      vt:    *mut   $complex_type_name,
      ldvt:  *const c_int,
      work:  *mut   $complex_type_name,
      lwork: *const c_int,
      rwork: *mut   $type_name,
      info:  *mut   c_int,
    );
  };
}

extern "C" {
  gesvd!(sgesvd_, f32, f32      );
  gesvd!(dgesvd_, f64, f64      );
  gesvd!(cgesvd_, f32, Complex32);
  gesvd!(zgesvd_, f64, Complex64);
}

macro_rules! geqrf {
    ($fn_name:ident, $type_name:ident) => {
      pub(super) fn $fn_name(
        m:     *const c_int,
        n:     *const c_int,
        a:     *mut   $type_name,
        lda:   *const c_int,
        tau:   *mut   $type_name,
        work:  *mut   $type_name,
        lwork: *const c_int,
        info:  *mut   c_int,
      );
    };
}

extern "C" {
  geqrf!(sgeqrf_, f32      );
  geqrf!(dgeqrf_, f64      );
  geqrf!(cgeqrf_, Complex32);
  geqrf!(zgeqrf_, Complex64);
}

macro_rules! ungqr {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name
    (
      m:     *const c_int,
      n:     *const c_int,
      k:     *const c_int,
      a:     *mut   $type_name,
      lda:   *const c_int,
      tau:   *mut   $type_name,
      work:  *mut   $type_name,
      lwork: *const c_int,
      info:  *mut   c_int,
    );
  };
}

extern "C" {
  ungqr!(cungqr_, Complex32);
  ungqr!(zungqr_, Complex64);
  ungqr!(dorgqr_, f64);
  ungqr!(sorgqr_, f32);
}

