use num_complex::{
    Complex32,
    Complex64,
    ComplexFloat,
};
use std::ffi::{
    c_int,
    c_char,
};

// This is only for Complex valued tensors.
// TODO?: Add Lanczos method for real valued symmetric matrices.

macro_rules! impl_naupd {
    ($name:ident, $complex_type:ty, $real_type:ty) => {
        fn $name(
            ido:    *mut   c_int        ,
            bmat:   *const c_char       ,
            n:      *const c_int        ,
            which:  *const c_char       ,
            nev:    *const c_int        ,
            tol:    *const $real_type   ,
            resid:  *mut   $complex_type,
            ncv:    *const c_int        ,
            v:      *mut   $complex_type,
            ldv:    *const c_int        ,
            iparam: *mut   c_int        ,
            ipntr:  *mut   c_int        ,
            workd:  *mut   $complex_type,
            workl:  *mut   $complex_type,
            lworkl: *const c_int        ,
            rwork:  *mut   $real_type   ,
            info:   *mut   c_int        ,
        );
    };
}

extern "C" {
    impl_naupd!(cnaupd_, Complex32, f32);
    impl_naupd!(znaupd_, Complex64, f64);
}

macro_rules! impl_neupd {
    ($name:ident, $complex_type:ty, $real_type:ty) => {
        fn $name(
            rvec:   *const c_int        ,
            howmny: *const c_char       ,
            select: *const c_int        ,
            d:      *mut   $complex_type,
            z:      *mut   $complex_type,
            ldz:    *const c_int        ,
            sigma:  *const $complex_type,
            workev: *mut   $complex_type,
            bmat:   *const c_char       ,
            n:      *const c_int        ,
            which:  *const c_char       ,
            nev:    *const c_int        ,
            tol:    *const $real_type   ,
            resid:  *mut   $complex_type,
            ncv:    *const c_int        ,
            v:      *mut   $complex_type,
            ldv:    *const c_int        ,
            iparam: *mut   c_int        ,
            ipntr:  *mut   c_int        ,
            workd:  *mut   $complex_type,
            workl:  *mut   $complex_type,
            lworkl: *const c_int        ,
            rwork:  *mut   $real_type   ,
            info:   *mut   c_int        ,
        );
    };
}

extern "C" {
    impl_neupd!(cneupd_, Complex32, f32);
    impl_neupd!(zneupd_, Complex64, f64);
}

pub trait Arpack: ComplexFloat {
    unsafe fn naupd(
        ido:    *mut   c_int        ,
        bmat:   *const c_char       ,
        n:      *const c_int        ,
        which:  *const c_char       ,
        nev:    *const c_int        ,
        tol:    *const Self::Real   ,
        resid:  *mut   Self         ,
        ncv:    *const c_int        ,
        v:      *mut   Self         ,
        ldv:    *const c_int        ,
        iparam: *mut   c_int        ,
        ipntr:  *mut   c_int        ,
        workd:  *mut   Self         ,
        workl:  *mut   Self         ,
        lworkl: *const c_int        ,
        rwork:  *mut   Self::Real   ,
        info:   *mut   c_int        ,
    );
    unsafe fn neupd(
        rvec:   *const c_int        ,
        howmny: *const c_char       ,
        select: *const c_int        ,
        d:      *mut   Self         ,
        z:      *mut   Self         ,
        ldz:    *const c_int        ,
        sigma:  *const Self         ,
        workev: *mut   Self         ,
        bmat:   *const c_char       ,
        n:      *const c_int        ,
        which:  *const c_char       ,
        nev:    *const c_int        ,
        tol:    *const Self::Real   ,
        resid:  *mut   Self         ,
        ncv:    *const c_int        ,
        v:      *mut   Self         ,
        ldv:    *const c_int        ,
        iparam: *mut   c_int        ,
        ipntr:  *mut   c_int        ,
        workd:  *mut   Self         ,
        workl:  *mut   Self         ,
        lworkl: *const c_int        ,
        rwork:  *mut   Self::Real   ,
        info:   *mut   c_int        ,
    );
}

macro_rules! impl_arpack {
    ($type:ty, $naupd:ident, $neupd:ident) => {
        impl Arpack for $type {
            #[inline]
            unsafe fn naupd(
                ido:    *mut   c_int        ,
                bmat:   *const c_char       ,
                n:      *const c_int        ,
                which:  *const c_char       ,
                nev:    *const c_int        ,
                tol:    *const Self::Real   ,
                resid:  *mut   Self         ,
                ncv:    *const c_int        ,
                v:      *mut   Self         ,
                ldv:    *const c_int        ,
                iparam: *mut   c_int        ,
                ipntr:  *mut   c_int        ,
                workd:  *mut   Self         ,
                workl:  *mut   Self         ,
                lworkl: *const c_int        ,
                rwork:  *mut   Self::Real   ,
                info:   *mut   c_int        ,
            )
            {
                $naupd(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
            }
            #[inline]
            unsafe fn neupd(
                rvec:   *const c_int        ,
                howmny: *const c_char       ,
                select: *const c_int        ,
                d:      *mut   Self         ,
                z:      *mut   Self         ,
                ldz:    *const c_int        ,
                sigma:  *const Self         ,
                workev: *mut   Self         ,
                bmat:   *const c_char       ,
                n:      *const c_int        ,
                which:  *const c_char       ,
                nev:    *const c_int        ,
                tol:    *const Self::Real   ,
                resid:  *mut   Self         ,
                ncv:    *const c_int        ,
                v:      *mut   Self         ,
                ldv:    *const c_int        ,
                iparam: *mut   c_int        ,
                ipntr:  *mut   c_int        ,
                workd:  *mut   Self         ,
                workl:  *mut   Self         ,
                lworkl: *const c_int        ,
                rwork:  *mut   Self::Real   ,
                info:   *mut   c_int        ,
            )
            {
               $neupd(rvec, howmny, select, d, z, ldz, sigma, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
            }
        }
    };
}

impl_arpack!(Complex32, cnaupd_, cneupd_);
impl_arpack!(Complex64, znaupd_, zneupd_);
