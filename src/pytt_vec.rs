use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
    };
use numpy::{PyArray3, PyArray2, PyArray1, PyArray};
use num_complex::Complex64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    TTVec,
    tt_traits::TTError, CrossBuilder,
    TensorTrain,
};

// ------------------------------------------------------------------------------- //

impl From<TTError> for PyErr {
    fn from(err: TTError) -> Self {
        PyRuntimeError::new_err(format!("{:?}", err))
    }
}

// ------------------------------------------------------------------------------- //

/// Initializes a random Tensor Train whose kernels are random tensors
/// with i.i.d. elements distributed according to N(0, 1). It also initializes additional
/// meta-information necessary for cross approximation subroutine.
/// Args:
///     mode_dims (List[int]): with dimensions of tensor modes;
///     rank (int): is the maximal TT rank of a generated Tensor Train;
///     delta (double): a hyperparameter of the Maxvol algorithm that specifies a
///         stopping criteria. Typically should be small, e.g. 0.01. For
///         more information see "S. A. Goreinov et. al. How to Find a Good Submatrix
///         https://doi.org/10.1142/9789812836021_0015";
///     tt_opt (bool): the parameter showing if one needs to track an approximate maximum modulo element
///         as it is described in the paper https://arxiv.org/abs/2205.00293;
/// Returns:
///     [TTVc64]: Tensor train object.
#[pyclass(text_signature = "(mode_dims: List[int], rank: int, delta: double, tt_opt: bool)")]
#[derive(Clone)]
pub struct TTVc64(CrossBuilder<Complex64, TTVec<Complex64>>);

#[pymethods]
impl TTVc64 {

    #[args(mode_dims, rank, delta="0.01", tt_opt="false")]
    #[new]
    fn new(mode_dims: Vec<usize>, rank: usize, delta: f64, tt_opt: bool) -> Self {
        Self (CrossBuilder::new(rank, delta, &mode_dims, tt_opt))
    }

    /// Returns a list with dimensions of bonds
    /// connecting neighboring kernels.
    /// Returns:
    ///     [List[int]]: bond dimensions.
    fn get_bonds(&self) -> Vec<usize> {
        self.0.tt.get_bonds().to_owned()
    }

    /// Conjugates a Tensor Train inplace.
    /// Note: meaningful only for complex valued tensors.
    fn conj(&mut self) {
        self.0.tt.conj()
    }

    /// Computes a dot product of two Tensor Trains.
    /// Returns the result as a tuple of two values: the first value v1 is the logarithm
    /// of the dot product modulo, the second value v2 is the phase multiplier that can be written
    /// as v2 = exp(i * phi). The dot product can be reconstructed from these two values as
    /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
    /// the dot product can be exponentially large in some cases.
    /// Args:
    ///     other (TTVc64): another Tensor Train.
    /// Returns:
    ///     [Tuple[Complex<double>]]: tuple with two values v1 and v2 (see explanation above).
    fn log_dot(&self, other: &TTVc64) -> PyResult<(Complex64, Complex64)> {
        let val = self.0.tt.log_dot(&other.0.tt)?;
        Ok(val)
    }

    /// Computes the sum of all elements of a tensor represented by a Tensor Train.
    /// Returns the result as a tuple of two values: the first value v1 is the logarithm
    /// of the sum, the second value v2 is the phase multiplier that can be written
    /// as v2 = exp(i * phi). The resulting sum can be reconstructed from these two values as
    /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
    /// the value of sum can be exponentially large in some cases.
    /// Returns:
    ///     [Tuple[Complex<double>]]: tuple with two values v1 and v2 (see explanation above).
    fn log_sum<'py>(&self) -> PyResult<(Complex64, Complex64)> {
        let val = self.0.tt.log_sum()?;
        Ok(val)
    }

    /// Sets a Tensor Train into the left-canonical form inplace.
    /// It returns the natural logarithm of the L2 norm of the initial tensor.
    /// The L2 norm of the tensor after this subroutine is equal to 1.
    /// Returns:
    ///     [double]: natural logarithm of the L2 norm.
    fn set_into_left_canonical(&mut self) -> PyResult<Complex64> {
        let val = self.0.tt.set_into_left_canonical()?;
        Ok(val)
    }

    /// Sets a Tensor Train into the right-canonical form inplace.
    /// It returns the natural logarithm of the L2 norm of the initial tensor.
    /// The L2 norm of the tensor after this subroutine is equal to 1.
    /// Returns:
    ///     [double]: natural logarithm of the L2 norm.
    fn set_into_right_canonical(&mut self) -> PyResult<Complex64> {
        let val = self.0.tt.set_into_right_canonical()?;
        Ok(val)
    }

    /// Computes an element of a Tensor Train given the index.
    /// Returns the result as a tuple of two values: the first value v1 is the logarithm
    /// of the element modulo, the second value v2 is the phase multiplier that can be written
    /// as v2 = exp(i * phi). The value of an element can be reconstructed from these two values as
    /// exp(v1) * v2. Such overcomplicated output is motivated by the fact, that
    /// the value of an element can be exponentially large in some cases.
    /// Args:
    ///     index (np.ndarray<int>): index specifying an element of a tensor.
    /// Returns:
    ///     [Tuple[Complex<double>]]: tuple with two values v1 and v2 (see explanation above).
    fn log_eval_index(&self, index: &PyArray1<i64>) -> PyResult<(Complex64, Complex64)> {
        let index: Vec<_> = unsafe { index.as_slice()? }.into_iter().map(|x| *x as usize).collect();
        let val = self.0.tt.log_eval_index(&index[..])?;
        Ok(val)
    }

    /// Truncates the left-canonical form of a Tensor Train. A Tensor Train is normalized
    /// to 1 after truncation. The actual L2 norm is returned by this method.
    /// Args:
    ///     delta (f64): truncation accuracy.
    /// Returns:
    ///     [double]: L2 norm of a Tensor Train as if it is not normalized to 1.
    fn truncate_left_canonical(&mut self, delta: f64) -> PyResult<f64> {
        let val = self.0.tt.truncate_left_canonical(delta)?;
        Ok(val)
    }

    /// Truncates the right-canonical form of a Tensor Train. A Tensor Train is normalized
    /// to 1 after truncation. The actual L2 norm is returned by this method.
    /// Args:
    ///     delta (f64): truncation accuracy.
    /// Returns:
    ///     [double]: L2 norm of a Tensor Train as if it is not normalized to 1.
    fn truncate_right_canonical(&mut self, delta: f64) -> PyResult<f64> {
        let val = self.0.tt.truncate_right_canonical(delta)?;
        Ok(val)
    }

    /// Multiply a given Tensor Train by an another one element-wisely.
    /// The given tensor is being modified.
    /// Args:
    ///     other (TTVc64): an another Tensor Train.
    fn elementwise_prod(&mut self, other: &TTVc64) -> PyResult<()> {
        self.0.tt.elementwise_prod(&other.0.tt)?;
        Ok(())
    }

    /// Adds to a given Tensor Train an another one element-wisely.
    /// The given tensor is being modified.
    /// Args:
    ///     other (TTVc64): an another Tensor Train.
    fn elementwise_sum(&mut self, other: &TTVc64) -> PyResult<()> {
        self.0.tt.elementwise_sum(&other.0.tt)?;
        Ok(())
    }

    /// Multiplies a Tensor Train by a scalar inplace.
    /// Args:
    ///     scalar (Complex<double>): scalar.
    fn mul_by_scalar(&mut self, scalar: Complex64) -> PyResult<()> {
        self.0.tt.mul_by_scalar(scalar);
        Ok(())
    }

    /// Returns a set of indices that needs to be evaluated at the
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

    /// Updates a Tensor Train according to the obtained measurement results.
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

    /// Returns a copy of a Tensor Train
    fn get_clone(&self) -> Self
    {
        self.clone()
    }

    /// Returns kernels of a Tensor Train.
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

    /// This method is the combination of the optimization methods
    /// (1) https://arxiv.org/abs/2101.03377 and (2) https://arxiv.org/abs/2209.14808
    /// The method (1) is essentially a power iteration method. It is being run first.
    /// It takes at most power_iterations_max_num or being terminated earlier if the
    /// max_rank of the power of a tensor train is achieved. Then one runs (2) method
    /// on the resulting power of a tensor train, k is the hyper parameter (for more
    /// details see (2)), typically it is set to be equal ~ 10.
    /// Args:
    ///     delta (double): truncation accuracy;
    ///     power_iterations_max_num (int): maximum number of power iterations;
    ///     max_rank: the maximal acceptable rank during the power iteration;
    ///     k: hyperparameter (see (2) for more details);
    /// Returns:
    ///     [List[int]]: modulo argmax.
    fn argmax_modulo(
        &self,
        delta: f64,
        power_iterations_max_num: usize,
        max_rank: usize,
        k: usize,
    ) -> PyResult<Vec<usize>>
    {
        let val = self.0.tt.argmax_modulo(delta, power_iterations_max_num, max_rank, k)?;
        Ok(val)
    }

    /// Returns an approximate argument of the maximum modulo element if the flag
    /// "tt_opt" was set at the initialization. For more details see https://arxiv.org/abs/2205.00293.
    /// Returns:
    ///     [Union[None, List[int]]]: approximate argument of the maximum modulo element.
    fn tt_opt_argmax_module(
        &self,
    ) -> Option<Vec<usize>>
    {
        self.0.get_tt_opt_argmax()
    }

    fn __repr__(&self) -> String {
        format!("{{ Bond dims: {:?}, Modes: {:?} }}", self.get_bonds(), self.0.tt.mode_dims)
    }
}
