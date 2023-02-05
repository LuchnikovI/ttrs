use std::{ffi::{
    c_char,
    c_int,
  }, fmt::Debug};

use crate::{
    NDArray,
    ndarray::{
        NDArrayError,
        NDArrayResult,
        Layout,
    },
    init_utils::BufferGenerator,
    arpack_bind::Arpack,
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

pub unsafe fn sparse_eigensolve<T>(
    op: &impl Fn(NDArray<*const T, 1>, NDArray<*mut T, 1>),
    mode: Which,
    tol: T::Real,
    maxiter: usize,
    vecs: NDArray<*mut T, 2>,
    vals: NDArray<*mut T, 1>,
) -> NDArrayResult<()>
where
    T: Arpack + BufferGenerator + 'static,
    T::Real: BufferGenerator,
{
    if vecs.layout != Layout::Fortran { return Err(NDArrayError::FortranLayoutRequired); }
    if vals.layout != Layout::Fortran { return Err(NDArrayError::FortranLayoutRequired); }
    if vecs.shape[1] != vals.shape[0] { return  Err(SparseLinalgError::EigenVectorsAndValuesNumberMismatch.into()); }
    let mut ido: c_int = 0;
    let bmat = 'I' as c_char;
    let n = vecs.shape[0] as c_int;
    let which = Into::<&str>::into(mode).as_ptr() as *const c_char;
    let nev = vecs.shape[1] as c_int;
    let mut resid = T::uninit_buff(n as usize);
    let ncv = std::cmp::min(std::cmp::max(2 * nev + 1, 20), n);
    let mut v = T::uninit_buff(n as usize * ncv as usize);
    let ldv = n;
    let mut iparam = [0; 11];
    iparam[0] = 1;
    iparam[2] = maxiter as c_int;
    iparam[6] = 1;
    let mut ipntr = [0; 14];
    let mut workd = T::uninit_buff(3 * n as usize);
    let lworkl = 3 * ncv.pow(2) + 5 * ncv;
    let mut workl = T::uninit_buff(lworkl as usize);
    let mut rwork = T::Real::uninit_buff(ncv as usize);
    let mut info = 0;
    while ido != 99 {
        T::naupd(
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
    let mut d = T::uninit_buff(nev as usize + 1);
    let ldz = n;
    let sigma = T::zero();
    let mut workev = T::uninit_buff(2 * ncv as usize);
    T::neupd(
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

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};
    use crate::{
        SparseLinalgComplex,
        LinalgReal,
        NDArray,
        sparse_linalg::Which::{
            LargestMagnitude,
            LargestRealPart,
            LargestImaginaryPart,
            SmallestMagnitude,
            SmallestRealPart,
            SmallestImaginaryPart,
        },
    };
    use super::{sparse_eigensolve, Which};

    #[inline]
    unsafe fn _test_sparse_eigsolver<T>(
        which: Which,
        acc: T::Real,
    )
    where
        T: SparseLinalgComplex,
        T::Real: LinalgReal,
    {
        let n = 100;
        let nev = 3;
        let mut m_buff = T::random_normal(n * n);
        let mut m_buff_conj = m_buff.clone();
        let m = NDArray::from_mut_slice(&mut m_buff, [n, n]).unwrap();
        let m_conj = NDArray::from_mut_slice(&mut m_buff_conj, [n, n]).unwrap();
        unsafe { m_conj.conj() };
        unsafe { m.add_inpl(m_conj.transpose([1, 0]).unwrap()).unwrap() }
        let mut eigvecs_buff = T::random_normal(n * nev);
        let mut eigvals_buff = T::random_normal(nev);
        let eigvecs = NDArray::from_mut_slice(&mut eigvecs_buff, [n, nev]).unwrap();
        let eigvals = NDArray::from_mut_slice(&mut eigvals_buff, [nev]).unwrap();
        let op = |src: NDArray<*const T, 1>, dst: NDArray<*mut T, 1>| {
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
            sparse_eigensolve(
                &op,
                which,
                acc,
                10000,
                eigvecs,
                eigvals,
            ).unwrap()
        }
        let (mut _egivecs_h_buff, eigvecs_h) = unsafe { eigvecs.transpose([1, 0]).unwrap().gen_f_array() };
        unsafe { eigvecs_h.conj() };
        let mut eigvecs_prod_buff = vec![T::zero(); nev * nev];
        let eigvecs_prod = NDArray::from_mut_slice(&mut eigvecs_prod_buff, [nev, nev]).unwrap();
        unsafe { eigvecs_prod.matmul_inplace(eigvecs_h, eigvecs).unwrap() }
        unsafe {
            eigvecs_prod.into_f_iter().enumerate().for_each(|(i, x)| {
                if i % (nev + 1) == 0 {
                    assert!((*x.0 - T::one()).abs() < acc);
                } else {
                    assert!((*x.0).abs() < acc);
                }
            });
        }
        let mut m_eigvecs_buff = vec![T::zero(); n * nev];
        let m_eigvecs = NDArray::from_mut_slice(&mut m_eigvecs_buff, [n, nev]).unwrap();
        unsafe { assert!(eigvals.into_mem_iter().unwrap().all(|x| ((*x.0).im() < acc) && ((*x.0).im() > -acc))); }
        unsafe { m_eigvecs.matmul_inplace(m, eigvecs).unwrap(); }
        unsafe { eigvecs.mul_inpl(eigvals.reshape([1, nev]).unwrap()).unwrap(); }
        unsafe { eigvecs.sub_inpl(m_eigvecs).unwrap(); }
        assert!(unsafe { eigvecs.norm_n_pow_n(2).abs() } < acc);
    }

    #[test]
    fn test_sparse_eigsolver() {
        unsafe {
            _test_sparse_eigsolver::<Complex32>(LargestMagnitude     , 1e-4 );
            _test_sparse_eigsolver::<Complex64>(LargestMagnitude     , 1e-10);
            _test_sparse_eigsolver::<Complex32>(SmallestMagnitude    , 1e-4 );
            _test_sparse_eigsolver::<Complex64>(SmallestMagnitude    , 1e-10);
            _test_sparse_eigsolver::<Complex32>(LargestRealPart      , 1e-4 );
            _test_sparse_eigsolver::<Complex64>(LargestRealPart      , 1e-10);
            _test_sparse_eigsolver::<Complex32>(SmallestRealPart     , 1e-4 );
            _test_sparse_eigsolver::<Complex64>(SmallestRealPart     , 1e-10);
            _test_sparse_eigsolver::<Complex32>(LargestImaginaryPart , 1e-4 );
            _test_sparse_eigsolver::<Complex64>(LargestImaginaryPart , 1e-10);
            _test_sparse_eigsolver::<Complex32>(SmallestImaginaryPart, 1e-4 );
            _test_sparse_eigsolver::<Complex64>(SmallestImaginaryPart, 1e-10);
        }
    }
}
