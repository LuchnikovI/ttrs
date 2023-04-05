mod tt_traits;
mod tt_cross;
mod utils;
mod tt_vec;
mod pytt_vec;
mod mutli_indices;

pub use tt_vec::TTVec;
pub use tt_cross::CrossBuilder;
pub use tt_traits::TensorTrain;

use pyo3::prelude::*;
use pytt_vec::TTVc64;

/// The module provides basic operations with the Tensor Train format.
#[pymodule]
fn ttrs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_class::<TTVc64>()?;
  Ok(())
}

#[cfg(test)]
mod tests {
    use linwrap::{LinalgComplex, LinalgReal};
    use num_complex::{
        Complex64,
        Complex32,
        ComplexFloat,
    };
    use crate::TTVec;
    use crate::tt_traits::TensorTrain;

    #[inline]
    fn exp<T: ComplexFloat>(x: (T, T)) -> T {
      x.0.exp() * x.1
    }

    #[inline]
    fn _test_random_cross<T>(
        acc: T::Real,
        maxvol_termination: T::Real,
    )
    where
        T: LinalgComplex,
        T::Real: LinalgReal,
    {
        let mode_dims = [2, 5, 3, 2, 45, 5, 2, 1, 4, 3, 2, 3, 27, 5, 6, 5, 1, 3, 2, 3, 4, 1, 2, 3, 4, 30, 4, 3, 2, 3, 4, 5, 4, 3];
        let mut random_tt = TTVec::<T>::new_random(mode_dims.to_vec(), 25);
        random_tt.set_into_left_canonical().unwrap();
        let (mut cross_result, _) = TTVec::<T>::ttcross(&mode_dims, 30, maxvol_termination, |x| exp(random_tt.log_eval_index(x).unwrap()), 5, true).unwrap();
        let log_norm_cross_result = cross_result.set_into_left_canonical().unwrap();
        assert!(log_norm_cross_result.abs() < acc);
        let mut random_tt_conj = random_tt.clone();
        random_tt_conj.conj();
        let mut cross_result_conj = cross_result.clone();
        cross_result_conj.conj();
        let diff = exp(random_tt.log_dot(&random_tt_conj).unwrap()) +
                    exp(cross_result.log_dot(&cross_result_conj).unwrap()) -
                    exp(random_tt.log_dot(&cross_result_conj).unwrap()) -
                    exp(cross_result.log_dot(&random_tt_conj).unwrap());
        assert!(diff.abs() < acc);
    }

    #[test]
    fn test_random_cross() {
        _test_random_cross::<f32>(      1e-3,  1e-5);
        _test_random_cross::<f64>(      1e-10, 1e-5);
        _test_random_cross::<Complex32>(1e-3,  1e-5);
        _test_random_cross::<Complex64>(1e-10, 1e-5);
    }
}
