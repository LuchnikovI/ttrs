extern crate lapack_src;

use std::ffi::{
  c_int,
  c_char,
};

use num_complex::{Complex64, Complex32, ComplexFloat};

// ----------------------------------------- Externs ----------------------------------------- //

macro_rules! getrf {
  ($fn_name:ident, $type_name:ident) => {
    fn $fn_name (
      m:    *const c_int,
      n:    *const c_int,
      a:    *mut   $type_name,
      lda:  *const c_int,
      ipiv: *mut   c_int,
      info: *mut   c_int,
    );
  }
}

extern "C" {
  getrf!(sgetrf_, f32      );
  getrf!(dgetrf_, f64      );
  getrf!(cgetrf_, Complex32);
  getrf!(zgetrf_, Complex64);
}

/*macro_rules! getrs {
  ($fn_name:ident, $type_name:ident) => {
    pub(super) fn $fn_name (
      trans: *const c_char,
      n:     *const c_int,
      nrhs:  *const c_int,
      a:     *mut   $type_name,
      lda:   *const c_int,
      ipiv:  *mut   c_int,
      b:     *mut   $type_name,
      ldb:   *const c_int,
      info:  *mut   c_int,
    );
  }
}

extern "C" {
  getrs!(sgetrs_, f32      );
  getrs!(dgetrs_, f64      );
  getrs!(cgetrs_, Complex32);
  getrs!(zgetrs_, Complex64);
}*/

// TODO?: switch to the expert mode (@gesvx)
macro_rules! gesv {
  ($fn_name:ident, $type_name:ident) => {
    fn $fn_name (
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
    fn $fn_name(
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
      fn $fn_name(
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

/*macro_rules! gerqf {
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
gerqf!(sgerqf_, f32      );
gerqf!(dgerqf_, f64      );
gerqf!(cgerqf_, Complex32);
gerqf!(zgerqf_, Complex64);
}*/

macro_rules! ungqr {
  ($fn_name:ident, $type_name:ident) => {
    fn $fn_name
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

macro_rules! heev {
  ($fn_name:ident, $complex_type:ty, $real_type:ty) => {
    fn $fn_name
    (
      jobz:  *const c_char,
      uplo:  *const c_char,
      n:     *const c_int,
      a:     *mut   $complex_type,
      lda:   *const c_int,
      w:     *mut   $real_type,
      work:  *mut   $complex_type,
      lwork: *const c_int,
      rwork: *mut   $real_type,
      info:  *mut   c_int,
    );
  };
}

extern "C" {
  heev!(cheev_, Complex32, f32);
  heev!(zheev_, Complex64, f64);
}

macro_rules! syev {
  ($fn_name:ident, $type:ty) => {
    fn $fn_name
    (
      jobz:  *const c_char,
      uplo:  *const c_char,
      n:     *const c_int,
      a:     *mut   $type,
      lda:   *const c_int,
      w:     *mut   $type,
      work:  *mut   $type,
      lwork: *const c_int,
      info:  *mut   c_int,
    );
  };
}

extern "C" {
  syev!(ssyev_, f32);
  syev!(dsyev_, f64);
}

#[inline]
unsafe fn sheev_(
  jobz:  *const c_char,
  uplo:  *const c_char,
  n:     *const c_int,
  a:     *mut   f32,
  lda:   *const c_int,
  w:     *mut   f32,
  work:  *mut   f32,
  lwork: *const c_int,
  _:     *mut   f32,
  info:  *mut   c_int,
)
{
  ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

#[inline]
unsafe fn dheev_(
  jobz:  *const c_char,
  uplo:  *const c_char,
  n:     *const c_int,
  a:     *mut   f64,
  lda:   *const c_int,
  w:     *mut   f64,
  work:  *mut   f64,
  lwork: *const c_int,
  _:     *mut   f64,
  info:  *mut   c_int,
)
{
  dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

/*macro_rules! ungrq {
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
  ungrq!(cungrq_, Complex32);
  ungrq!(zungrq_, Complex64);
  ungrq!(dorgrq_, f64);
  ungrq!(sorgrq_, f32);
}*/
// ----------------------------------------- Lapack ----------------------------------------- //

pub trait Lapack: ComplexFloat
{
  unsafe fn getrf(
    m:    *const c_int,
    n:    *const c_int,
    a:    *mut   Self ,
    lda:  *const c_int,
    ipiv: *mut   c_int,
    info: *mut   c_int,
  );
  unsafe fn gesv (
    n:    *const c_int,
    nrhs: *const c_int,
    a:    *mut   Self ,
    lda:  *const c_int,
    ipiv: *mut   c_int,
    b:    *mut   Self ,
    ldb:  *const c_int,
    info: *mut   c_int,
  );
  unsafe fn gesvd(
    jobu:  *const c_char    ,
    jobvt: *const c_char    ,
    m:     *const c_int     ,
    n:     *const c_int     ,
    a:     *mut   Self      ,
    lda:   *const c_int     ,
    s:     *mut   Self::Real,
    u:     *mut   Self      ,
    ldu:   *const c_int     ,
    vt:    *mut   Self      ,
    ldvt:  *const c_int     ,
    work:  *mut   Self      ,
    lwork: *const c_int     ,
    rwork: *mut   Self::Real,
    info:  *mut   c_int     ,
  );
  unsafe fn geqrf(
    m:     *const c_int,
    n:     *const c_int,
    a:     *mut   Self ,
    lda:   *const c_int,
    tau:   *mut   Self ,
    work:  *mut   Self ,
    lwork: *const c_int,
    info:  *mut   c_int,
  );
  unsafe fn ungqr(
    m:     *const c_int,
    n:     *const c_int,
    k:     *const c_int,
    a:     *mut   Self ,
    lda:   *const c_int,
    tau:   *mut   Self ,
    work:  *mut   Self ,
    lwork: *const c_int,
    info:  *mut   c_int,
  );
  unsafe fn heev(
    jobz:  *const c_char,
    uplo:  *const c_char,
    n:     *const c_int,
    a:     *mut   Self,
    lda:   *const c_int,
    w:     *mut   Self::Real,
    work:  *mut   Self,
    lwork: *const c_int,
    rwork: *mut   Self::Real,
    info:  *mut   c_int,
  );
}

// ----------------------------------------- Externs impls ----------------------------------------- //

macro_rules! impl_lapack {
    ($type:ty, $getrf:ident, $gesv:ident, $gesvd:ident, $geqrf:ident, $ungqr:ident, $heev:ident) => {
      impl Lapack for $type {
        #[inline]
        unsafe fn getrf(
            m:    *const c_int,
            n:    *const c_int,
            a:    *mut   Self ,
            lda:  *const c_int,
            ipiv: *mut   c_int,
            info: *mut   c_int,
        )
        {
          $getrf(m, n, a, lda, ipiv, info);
        }
        #[inline]
        unsafe fn gesv (
            n:    *const c_int,
            nrhs: *const c_int,
            a:    *mut   Self ,
            lda:  *const c_int,
            ipiv: *mut   c_int,
            b:    *mut   Self ,
            ldb:  *const c_int,
            info: *mut   c_int,
        )
        {
          $gesv(n, nrhs, a, lda, ipiv, b, ldb, info); 
        }
        #[inline]
        unsafe fn gesvd(
            jobu:  *const c_char    ,
            jobvt: *const c_char    ,
            m:     *const c_int     ,
            n:     *const c_int     ,
            a:     *mut   Self      ,
            lda:   *const c_int     ,
            s:     *mut   Self::Real,
            u:     *mut   Self      ,
            ldu:   *const c_int     ,
            vt:    *mut   Self      ,
            ldvt:  *const c_int     ,
            work:  *mut   Self      ,
            lwork: *const c_int     ,
            rwork: *mut   Self::Real,
            info:  *mut   c_int     ,
        )
        {
          $gesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
        }
        #[inline]
        unsafe fn geqrf(
            m:     *const c_int,
            n:     *const c_int,
            a:     *mut   Self ,
            lda:   *const c_int,
            tau:   *mut   Self ,
            work:  *mut   Self ,
            lwork: *const c_int,
            info:  *mut   c_int,
        )
        {
          $geqrf(m, n, a, lda, tau, work, lwork, info);
        }
        #[inline]
        unsafe fn ungqr(
            m:     *const c_int,
            n:     *const c_int,
            k:     *const c_int,
            a:     *mut   Self ,
            lda:   *const c_int,
            tau:   *mut   Self ,
            work:  *mut   Self ,
            lwork: *const c_int,
            info:  *mut   c_int,
        )
        {
          $ungqr(m, n, k, a, lda, tau, work, lwork, info);
        }
        #[inline]
        unsafe fn heev(
          jobz:  *const c_char,
          uplo:  *const c_char,
          n:     *const c_int,
          a:     *mut   Self,
          lda:   *const c_int,
          w:     *mut   Self::Real,
          work:  *mut   Self,
          lwork: *const c_int,
          rwork: *mut   Self::Real,
          info:  *mut   c_int,
        )
        {
          $heev(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
        }
      }
    };
}

impl_lapack!(f32,       sgetrf_, sgesv_, sgesvd_, sgeqrf_, sorgqr_, sheev_);
impl_lapack!(f64,       dgetrf_, dgesv_, dgesvd_, dgeqrf_, dorgqr_, dheev_);
impl_lapack!(Complex32, cgetrf_, cgesv_, cgesvd_, cgeqrf_, cungqr_, cheev_);
impl_lapack!(Complex64, zgetrf_, zgesv_, zgesvd_, zgeqrf_, zungqr_, zheev_);
