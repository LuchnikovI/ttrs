#![feature(concat_idents)]

mod matrix;
mod par_ptr_wrapper;
mod elementwise_binary_ops;
mod reduction_ops;
mod blas_bind;
mod linalg;
mod lapack_bind;
pub mod par_utils;
pub use matrix::Matrix;
