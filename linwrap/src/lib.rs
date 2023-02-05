mod par_ptr_wrapper;
mod ndarray_utils;
mod ndarray;
mod elementwise_unary_ops;
mod elementwise_inplace_binary_ops;
mod elementwise_binary_ops;
mod reduction_ops;
mod concatenation;
mod blas_bind;
mod linalg;
mod lapack_bind;
mod linalg_utils;
mod arpack_bind;
pub mod sparse_linalg;
pub mod init_utils;
pub use crate::ndarray::NDArray;
pub use crate::ndarray::NDArrayError;
use arpack_bind::Arpack;
use init_utils::BufferGenerator;
use lapack_bind::Lapack;
use blas_bind::Blas;
use num_complex::ComplexFloat;
pub use par_ptr_wrapper::ParPtrWrapper;

// These traits serve for the aggregation of all properties necessary for all linalg operations
use std::iter::Sum;
use std::fmt::Debug;
use num_complex::{ Complex32, Complex64 };
pub trait LinalgReal: BufferGenerator + Send + Sync + Sum + Debug {}
pub trait LinalgComplex: ComplexFloat + BufferGenerator + Sum + Lapack + Blas + Send + Sync + Debug + 'static
where
    Self::Real: LinalgReal,
{}
pub trait SparseLinalgComplex: LinalgComplex + Arpack
where
    Self::Real: LinalgReal,
{}
impl LinalgReal for f32 {}
impl LinalgReal for f64 {}
impl LinalgComplex for f32 {}
impl LinalgComplex for f64 {}
impl LinalgComplex for Complex32 {}
impl LinalgComplex for Complex64 {}
impl SparseLinalgComplex for Complex32 {}
impl SparseLinalgComplex for Complex64 {}
