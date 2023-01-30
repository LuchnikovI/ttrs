use std::{ffi::{
    c_char,
    c_int,
  }, fmt::Debug};
  
use num_complex::{
    Complex32,
    Complex64,
};

use crate::{
    NDArray,
    ndarray::{
        NDArrayError,
        NDArrayResult,
        Layout,
    },
    arpack_bind::{
        cnaupd_,
        cneupd_,
        znaupd_,
        zneupd_,
    },
    init_utils::{
        uninit_buff_f32,
        uninit_buff_f64,
        uninit_buff_c32,
        uninit_buff_c64,
    },
};

// ---------------------------------------------------------------------- //

#[derive(Debug, Clone, Copy)]
pub enum Which {
    LargestMagnitude,
    SmallestMagnitude,
    LargestRealPart,
    SmallestRealPart,
    LargestImaginaryPart,
    SmallestImaginaryPart,
}

impl Into<&'static str> for Which {
    fn into(self) -> &'static str {
        match self {
            Self::LargestMagnitude      => "LM",
            Self::SmallestMagnitude     => "SM",
            Self::LargestRealPart       => "LR",
            Self::SmallestRealPart      => "SR",
            Self::LargestImaginaryPart  => "LI",
            Self::SmallestImaginaryPart => "SI",
        }
    }
}

// ---------------------------------------------------------------------- //

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SparseLinalgError {
    EigenVectorsAndValuesNumberMismatch,
    WrongParameters(&'static str),
    ErrorWithCode(c_int),
}

impl Into<NDArrayError> for SparseLinalgError {
    fn into(self) -> NDArrayError {
        NDArrayError::SparseLinalgError(self)
    }
}

impl Debug for SparseLinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseLinalgError::EigenVectorsAndValuesNumberMismatch => { f.write_str("Sizes of buffers for eigenvectors and eigenvalues do not match.") },
            SparseLinalgError::WrongParameters(s) => { f.write_str(&format!("Incorrect input parameters: {:?}", s)) },
            SparseLinalgError::ErrorWithCode(c) => {f.write_str(&format!("Error with code {:?}, see Arpack documentation.", c))}
        }
    }
}

// ---------------------------------------------------------------------- //

macro_rules! impl_sparse_eigensolve {
    (
        $name:ident,
        $naupd:ident,
        $neupd:ident,
        $complex_type:ty,
        $real_type:ty,
        $complex_uninit_buff_fn:ident,
        $real_uninit_buff_fn:ident,
        $complex_zero:expr
    ) => {
        pub unsafe fn $name(
            op: &impl Fn(NDArray<*const $complex_type, 1>, NDArray<*mut $complex_type, 1>),
            mode: Which,
            tol: $real_type,
            maxiter: usize,
            vecs: NDArray<*mut $complex_type, 2>,
            vals: NDArray<*mut $complex_type, 1>,
        ) -> NDArrayResult<()>
        {
            if vecs.layout != Layout::Fortran { return Err(NDArrayError::FortranLayoutRequired); }
            if vals.layout != Layout::Fortran { return Err(NDArrayError::FortranLayoutRequired); }
            if vecs.shape[1] != vals.shape[0] { return  Err(SparseLinalgError::EigenVectorsAndValuesNumberMismatch.into()); }
            let mut ido: c_int = 0;
            let bmat = 'I' as c_char;
            let n = vecs.shape[0] as c_int;
            let which = Into::<&str>::into(mode).as_ptr() as *const c_char;
            let nev = vecs.shape[1] as c_int;
            let mut resid = $complex_uninit_buff_fn(n as usize);
            let ncv = std::cmp::min(std::cmp::max(2 * nev + 1, 20), n);
            let mut v = $complex_uninit_buff_fn(n as usize * ncv as usize);
            let ldv = n;
            let mut iparam = [0; 11];
            iparam[0] = 1;
            iparam[2] = maxiter as c_int;
            iparam[6] = 1;
            let mut ipntr = [0; 14];
            let mut workd = $complex_uninit_buff_fn(3 * n as usize);
            let lworkl = 3 * ncv.pow(2) + 5 * ncv;
            let mut workl = $complex_uninit_buff_fn(lworkl as usize);
            let mut rwork = $real_uninit_buff_fn(ncv as usize);
            let mut info = 0;
            while ido != 99 {
                $naupd(
                    &mut ido,
                    &bmat,
                    &n,
                    which,
                    &nev,
                    &tol,
                    resid.as_mut_ptr(),
                    &ncv,
                    v.as_mut_ptr(),
                    &ldv,
                    iparam.as_mut_ptr(),
                    ipntr.as_mut_ptr(),
                    workd.as_mut_ptr(),
                    workl.as_mut_ptr(),
                    &lworkl,
                    rwork.as_mut_ptr(),
                    &mut info,
                );
                if (ido == 1) || (ido == -1) {
                    let src = workd[(ipntr[0] as usize - 1)..(ipntr[0] as usize + n as usize - 1)].to_owned();
                    let dst = &mut workd[(ipntr[1] as usize - 1)..(ipntr[1] as usize + n as usize - 1)];
                    op(
                        NDArray::from_slice(&src, [n as usize]).unwrap(),
                        NDArray::from_mut_slice(dst, [n as usize]).unwrap(),
                    );
                }
            }
            match info {
                0 | 1 | 2 => {}
                -1 => return Err(SparseLinalgError::WrongParameters("N must be positive.").into()),
                -2 => return Err(SparseLinalgError::WrongParameters("NEV must be positive.").into()),
                -3 => return Err(SparseLinalgError::WrongParameters("The following condition must hold true: NCV-NEV >= 1 && NCV <= N.").into()),
                -4 => return Err(SparseLinalgError::WrongParameters("Maxiter must be greater than 0.").into()),
                -5 => return Err(SparseLinalgError::WrongParameters("Maximum iterations must be greater than 0.").into()),
                i => return Err(SparseLinalgError::ErrorWithCode(i).into()),
            }
            let rvec = 1;
            let howmny = 'A' as c_char;
            let select = vec![0; ncv as usize];
            let mut d = $complex_uninit_buff_fn(nev as usize + 1);
            let ldz = n;
            let sigma = $complex_zero;
            let mut workev = $complex_uninit_buff_fn(2 * ncv as usize);
            $neupd(
                &rvec,
                &howmny,
                select.as_ptr(),
                d.as_mut_ptr(),
                vecs.ptr,
                &ldz,
                &sigma,
                workev.as_mut_ptr(),
                &bmat,
                &n,
                which,
                &nev,
                &tol,
                resid.as_mut_ptr(),
                &ncv,
                v.as_mut_ptr(),
                &ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                &lworkl,
                rwork.as_mut_ptr(),
                &mut info,
            );
            for (i, v) in d.into_iter().enumerate().take(nev as usize) {
                *vals.ptr.add(i) = v;
            }
            Ok(())
        }
    };
}

impl_sparse_eigensolve!(sparse_eigensolve_c32, cnaupd_, cneupd_, Complex32, f32, uninit_buff_c32, uninit_buff_f32, Complex32::new(0f32, 0f32));
impl_sparse_eigensolve!(sparse_eigensolve_c64, znaupd_, zneupd_, Complex64, f64, uninit_buff_c64, uninit_buff_f64, Complex64::new(0f64, 0f64));

#[cfg(test)]
mod tests {
    use num_complex::{
        Complex32,
        Complex64,
    };
    use crate::{
        NDArray,
        init_utils::{
            random_normal_c32,
            random_normal_c64,
        },
        sparse_linalg::Which::{
            LargestMagnitude,
            LargestRealPart,
            LargestImaginaryPart,
            SmallestMagnitude,
            SmallestRealPart,
            SmallestImaginaryPart,
        },
    };

    use super::sparse_eigensolve_c32;
    use super::sparse_eigensolve_c64;

    macro_rules! test_sparse_eigsolver {
        (
            $name:ident,
            $random_init_fn:ident,
            $complex_type:ty,
            $complex_zero:expr,
            $which:expr,
            $accuracy:expr
        ) => {
            let n = 100;
            let nev = 3;
            let mut m_buff = $random_init_fn(n * n);
            let mut m_buff_conj = m_buff.clone();
            let m = NDArray::from_mut_slice(&mut m_buff, [n, n]).unwrap();
            let m_conj = NDArray::from_mut_slice(&mut m_buff_conj, [n, n]).unwrap();
            unsafe { m_conj.conj() };
            unsafe { m.add_inpl(m_conj.transpose([1, 0]).unwrap()).unwrap() }
            let mut eigvecs_buff = $random_init_fn(n * nev);
            let mut eigvals_buff = $random_init_fn(nev);
            let eigvecs = NDArray::from_mut_slice(&mut eigvecs_buff, [n, nev]).unwrap();
            let eigvals = NDArray::from_mut_slice(&mut eigvals_buff, [nev]).unwrap();
            let op = |src: NDArray<*const $complex_type, 1>, dst: NDArray<*mut $complex_type, 1>| {
                unsafe {
                    dst.reshape([n as usize, 1]).unwrap()
                    .matmul_inplace(
                        m,
                        src.reshape([n, 1]).unwrap(),
                    )
                    .unwrap();
                }
            };
            unsafe {
                $name(
                    &op,
                    $which,
                    $accuracy,
                    10000,
                    eigvecs,
                    eigvals,
                ).unwrap()
            }
            let (mut _egivecs_h_buff, eigvecs_h) = unsafe { eigvecs.transpose([1, 0]).unwrap().gen_f_array() };
            unsafe { eigvecs_h.conj() };
            let mut eigvecs_prod_buff = vec![$complex_zero; nev * nev];
            let eigvecs_prod = NDArray::from_mut_slice(&mut eigvecs_prod_buff, [nev, nev]).unwrap();
            unsafe { eigvecs_prod.matmul_inplace(eigvecs_h, eigvecs).unwrap() }
            unsafe {
                eigvecs_prod.into_f_iter().enumerate().for_each(|(i, x)| {
                    if i % (nev + 1) == 0 {
                        assert!((*x.0).im.abs() < $accuracy);
                        assert!(((*x.0).re - 1.).abs() < $accuracy);
                    } else {
                        assert!((*x.0).im.abs() < $accuracy);
                        assert!((*x.0).re.abs() < $accuracy);
                    }
                });
            }
            let mut m_eigvecs_buff = vec![$complex_zero; n * nev];
            let m_eigvecs = NDArray::from_mut_slice(&mut m_eigvecs_buff, [n, nev]).unwrap();
            unsafe { assert!(eigvals.into_mem_iter().unwrap().all(|x| (*x.0).im.abs() < $accuracy )); }
            unsafe { m_eigvecs.matmul_inplace(m, eigvecs).unwrap(); }
            unsafe { eigvecs.mul_inpl(eigvals.reshape([1, nev]).unwrap()).unwrap(); }
            unsafe { eigvecs.sub_inpl(m_eigvecs).unwrap(); }
            assert!(unsafe { eigvecs.norm_n_pow_n(2) } < $accuracy);
        };
    }

    #[test]
    fn test_sparse_eigsolver() {
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), LargestMagnitude     , 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), LargestMagnitude     , 1e-10);
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), SmallestMagnitude    , 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), SmallestMagnitude    , 1e-10);
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), LargestRealPart      , 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), LargestRealPart      , 1e-10);
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), SmallestRealPart     , 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), SmallestRealPart     , 1e-10);
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), LargestImaginaryPart , 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), LargestImaginaryPart , 1e-10);
        test_sparse_eigsolver!(sparse_eigensolve_c32, random_normal_c32, Complex32, Complex32::new(0f32, 0f32), SmallestImaginaryPart, 1e-4 );
        test_sparse_eigsolver!(sparse_eigensolve_c64, random_normal_c64, Complex64, Complex64::new(0f64, 0f64), SmallestImaginaryPart, 1e-10);
    }
}
