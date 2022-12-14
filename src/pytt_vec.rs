use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
    };
use numpy::{PyArray3, PyArray2, PyArray1, PyArray};
use num_complex::Complex64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    TTc64,
    CBc64,
    TTVec,
    tt_traits::TTError,
};

// ------------------------------------------------------------------------------- //

impl From<TTError> for PyErr {
    fn from(err: TTError) -> Self {
        PyRuntimeError::new_err(format!("{:?}", err))
    }
}

// ------------------------------------------------------------------------------- //

/// This function initializes a random Tensor Train whose kernels are random tensors
/// with i.i.d. elements distributed according to N(0, 1). It also initializes additional
/// meta-information necessary for cross approximation subroutine.
/// Args:
///     mode_dims (List[int]): with dimensions of tensor modes;
///     rank (int): is the maximal TT rank of a generated Tensor Train;
///     delta (double): a hyperparameter of the Maxvol algorithm that specifies a
///         stopping criteria. Typically should be small, e.g. 0.01. For
///         more information see "S. A. Goreinov et. al. How to Find a Good Submatrix
///         https://doi.org/10.1142/9789812836021_0015";
/// Returns:
///     [TTVc64]: Tensor train object.
#[pyclass(text_signature = "(mode_dims: List[int], rank: int, delta: double)")]
#[derive(Clone)]
pub struct TTVc64(CBc64<TTVec<Complex64>>);

#[pymethods]
impl TTVc64 {

    #[args(mode_dims, rank, delta="0.01")]
    #[new]
    fn new(mode_dims: Vec<usize>, rank: usize, delta: f64) -> Self {
        Self (CBc64::new(rank, delta, &mode_dims))
    }

    /// This method returns a list with dimensions of bonds
    /// connecting neighboring kernels.
    /// Returns:
    ///     [List[int]]: bond dimensions.
    fn get_bonds(&self) -> Vec<usize> {
        self.0.tt.get_bonds().to_owned()
    }

    /// This method conjugates a Tensor Train inplace.
    /// Note: meaningful only for complex valued tensors.
    fn conj(&mut self) {
        self.0.tt.conj()
    }

    /// This method returns a natural logarithm of the dot product of
    /// two Tensor Trains. The natural logarithm is necessary to ensure
    /// stability when the dot product is exponentially big.
    /// Args:
    ///     other (TTVc64): another Tensor Train.
    /// Returns:
    ///     [Complex<double>]: the dot product value.
    fn log_dot(&self, other: &TTVc64) -> PyResult<Complex64> {
        let val = self.0.tt.log_dot(&other.0.tt)?;
        Ok(val)
    }

    /// This method returns a natural logarithm of the sum of all the elements
    /// of a tensor encoded in Tensor Train format. The natural logarithm is necessary to ensure
    /// stability when the sum is exponentially big.
    /// Returns:
    ///     [Complex<double>]: the sum value.
    fn log_sum<'py>(&self) -> PyResult<Complex64> {
        let val = self.0.tt.log_sum()?;
        Ok(val)
    }

    /// This method sets a Tensor Train into the left-canonical form inplace.
    /// It returns the natural logarithm of the L2 norm of the initial tensor.
    /// The L2 norm of the tensor after this subroutine is equal to 1.
    /// Returns:
    ///     [double]: natural logarithm of the L2 norm.
    fn set_into_left_canonical(&mut self) -> PyResult<f64> {
        let val = self.0.tt.set_into_left_canonical()?;
        Ok(val)
    }

    /// This method sets a Tensor Train into the right-canonical form inplace.
    /// It returns the natural logarithm of the L2 norm of the initial tensor.
    /// The L2 norm of the tensor after this subroutine is equal to 1.
    /// Returns:
    ///     [double]: natural logarithm of the L2 norm.
    fn set_into_right_canonical(&mut self) -> PyResult<f64> {
        let val = self.0.tt.set_into_right_canonical()?;
        Ok(val)
    }


    /// This method evaluates a tensor at a given index. It returns the natural logarithm of the
    /// actual element value, it is necessary to ensure stability of the computation if an
    /// element is exponentially small / big.
    /// Args:
    ///     index (np.ndarray<int>): index specifying an element of a tensor.
    /// Returns:
    ///     [Complex<double>]: an element of a Tensor Train.
    fn log_eval_index(&self, index: &PyArray1<i64>) -> PyResult<Complex64> {
        let index: Vec<_> = unsafe { index.as_slice()? }.into_iter().map(|x| *x as usize).collect();
        let val = self.0.tt.log_eval_index(&index[..])?;
        Ok(val)
    }

    /// This method truncates the left-canonical form of a Tensor Train. A Tensor Train is normalized
    /// to 1 after truncation. The actual L2 norm is returned by this method.
    /// Args:
    ///     delta (f64): truncation accuracy.
    /// Returns:
    ///     [double]: L2 norm of a Tensor Train as if it is not normalized to 1.
    fn truncate_left_canonical(&mut self, delta: f64) -> PyResult<f64> {
        let val = self.0.tt.truncate_left_canonical(delta)?;
        Ok(val)
    }

    /// This method truncates the right-canonical form of a Tensor Train. A Tensor Train is normalized
    /// to 1 after truncation. The actual L2 norm is returned by this method.
    /// Args:
    ///     delta (f64): truncation accuracy.
    /// Returns:
    ///     [double]: L2 norm of a Tensor Train as if it is not normalized to 1.
    fn truncate_right_canonical(&mut self, delta: f64) -> PyResult<f64> {
        let val = self.0.tt.truncate_right_canonical(delta)?;
        Ok(val)
    }

    /// This method multiply a given Tensor Train by an another one element-wisely.
    /// The given tensor is being modified.
    /// Args:
    ///     other (TTVc64): an another Tensor Train.
    fn elementwise_prod(&mut self, other: &TTVc64) -> PyResult<()> {
        self.0.tt.elementwise_prod(&other.0.tt)?;
        Ok(())
    }

    /// This method add to a given Tensor Train an another one element-wisely.
    /// The given tensor is being modified.
    /// Args:
    ///     other (TTVc64): an another Tensor Train.
    fn elementwise_sum(&mut self, other: &TTVc64) -> PyResult<()> {
        self.0.tt.elementwise_sum(&other.0.tt)?;
        Ok(())
    }

    /// This method multiply a Tensor Train by a scalar inplace.
    /// Args:
    ///     scalar (Complex<double>): scalar.
    fn mul_by_scalar(&mut self, scalar: Complex64) -> PyResult<()> {
        self.0.tt.mul_by_scalar(scalar);
        Ok(())
    }

    /// This method returns a set of indices that needs to be evaluated at the
    /// current step of the TTCross interpolation.
    /// Returns:
    ///     [np.ndarray]: matrix whose rows are indices that need to be evaluated.
    fn get_args<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray2<usize>>> {
        match self.0.get_args() {
            Some(iter) => {
                let indices: Vec<Vec<_>> = iter.collect();
                let arr = PyArray::from_vec2(py, &indices[..])?;
                Ok(Some(arr))
            },
            None => {
                //let empty_indices = vec![vec![]];
                //let arr = PyArray2::from_vec2(py, &empty_indices)?;
                Ok(None)
            },
        }
    }

    /// This method updates a Tensor Train according to the obtained measurement results.
    /// Args:
    ///     measurements (np.ndarray<Complex<double>>): a vector with measurement results.
    /// Note, that for the TTCross output to be meaningful, the number of updates must be
    /// N * number_of_modes, where N is integer. 
    fn update(&mut self, measurements: Option<&PyArray1<Complex64>>) -> PyResult<()> {
        let mut iter = None;
        match measurements {
            None => {
                self.0.update(iter)?;
            },
            Some(m) => {
                let slice = unsafe { m.as_slice()? };
                iter = Some(slice.into_par_iter().map(|x| *x));
                self.0.update(iter)?;
            },
        }
        Ok(())
    }

    /// This method returns a copy of a Tensor Train
    fn get_clone(&self) -> Self
    {
        self.clone()
    }

    /// This method returns kernels of a Tensor Train.
    /// Returns:
    ///     [List[np.ndarray<double>]]: list with kernels.
    fn get_kernels<'py>(&self, py: Python<'py>) -> PyResult<Vec<&'py PyArray3<Complex64>>>
    {
        let mut output: Vec<&PyArray3<Complex64>> = Vec::with_capacity(self.0.tt.kernels.len());
        for (ker, right_bond, left_bond, mode_dim) in self.0.tt.iter() {
            let arr = PyArray1::from_slice(py, ker);
            let arr = arr.reshape_with_order([left_bond, mode_dim, right_bond], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)?;
            output.push(arr)
        }
        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!("{{ Bond dims: {:?}, Modes: {:?} }}", self.get_bonds(), self.0.tt.mode_dims)
    }
}
