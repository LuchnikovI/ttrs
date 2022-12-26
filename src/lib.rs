mod tt;
mod ttcross;
mod utils;

pub use tt::{
    TensorTrain,
    ttcross_c32,
    ttcross_c64,
    ttcross_f32,
    ttcross_f64,
};

#[cfg(test)]
mod tests {
    use num_complex::{
        Complex64,
        Complex32,
        ComplexFloat,
    };
    use super::{
        TensorTrain,
        ttcross_c32,
        ttcross_c64,
        ttcross_f32,
        ttcross_f64,
    };

    macro_rules! test_random_cross {
        ($fn_cross:ident, $complex_type:ty, $acc:expr) => {
            let mode_dims = [2, 5, 3, 2, 45, 5, 2, 1, 4, 3, 2, 3, 27, 5, 6, 5, 1, 3, 2, 3, 4, 1, 2, 3, 4, 30, 4, 3, 2, 3, 4, 5, 4, 3];
            let mut random_tt = TensorTrain::<$complex_type>::new_random_normal(&mode_dims, 25);
            random_tt.set_into_left_canonical().unwrap();
            let mut cross_result = $fn_cross(&mode_dims, 40, 0.01, |x| random_tt.eval_index(x).unwrap(), 4).unwrap();
            let log_norm_cross_result = cross_result.set_into_left_canonical().unwrap();
            assert!(log_norm_cross_result.abs() < $acc);
            let mut random_tt_conj = random_tt.clone();
            random_tt_conj.conj();
            let mut cross_result_conj = cross_result.clone();
            cross_result_conj.conj();
            let diff = random_tt.dot(&random_tt_conj).unwrap() +
                       cross_result.dot(&cross_result_conj).unwrap() -
                       random_tt.dot(&cross_result_conj).unwrap() -
                       cross_result.dot(&random_tt_conj).unwrap();
            assert!(diff.abs() < $acc);
        };
    }
    #[test]
    fn test_random_cross() {
        test_random_cross!(ttcross_f32, f32,       1e-3 );
        test_random_cross!(ttcross_f64, f64,       1e-10);
        test_random_cross!(ttcross_c32, Complex32, 1e-3 );
        test_random_cross!(ttcross_c64, Complex64, 1e-10);
    }
}