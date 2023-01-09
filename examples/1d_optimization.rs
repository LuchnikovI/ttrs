use ttrs::{
    TTVec,
    TTf64,
};

// This function converts a bit string to a value from 0 to 4
#[inline]
fn idx_to_arg(x: &[usize]) -> f64 {
    x.into_iter().enumerate().map(|(i, x)| {
        2. * (*x as f64) / 2f64.powi(i as i32)
    }).sum()
}

// This function is a target that we minimize
#[inline]
fn target_function(x: &[usize]) -> f64 {
    let arg = idx_to_arg(x);
    ((arg - 1.23456789).powi(2) + 1e-7).ln() * (15. * (arg - 1.23456789)).cos()
}

fn main() {
    let acc = 1e-9;    // Accuracy
    let alpha = -1e-1; // Time step
    // Function representation in the TT format
    let mut tt = TTVec::<f64>::ttcross(&[2; 40], 30, 0.001, target_function, 6).unwrap();
    tt.set_into_left_canonical().unwrap();
    tt.truncate_left_canonical(acc).unwrap();
    println!("Bond dimensions after TT representation reconstruction and truncation {:?}", tt.get_bonds());
    // A tensor that converts index to a function argument
    let tt_idx_to_arg = TTVec::<f64>::ttcross(&[2; 40], 2, 0.001, idx_to_arg, 4).unwrap();
    // A target tensor that is being evolved in imag. time
    let mut target = TTVec::<f64>::new_ones(vec![2; 40]);
    // An "evolution operator"
    let mut approx_exp = TTVec::<f64>::new_ones(vec![2; 40]);
    approx_exp.set_into_left_canonical().unwrap();
    tt.mul_by_scalar(alpha);
    approx_exp.elementwise_sum(&tt).unwrap();
    approx_exp.set_into_left_canonical().unwrap();
    approx_exp.truncate_left_canonical(acc).unwrap();
    // Evolution in imag. time
    let mut max_bond = 0;
    for i in 0..1000 {
        target.elementwise_prod(&approx_exp).unwrap();
        target.set_into_left_canonical().unwrap();
        target.truncate_left_canonical(acc).unwrap();
        target.set_into_right_canonical().unwrap();
        target.truncate_right_canonical(acc).unwrap();
        max_bond = std::cmp::max(max_bond, target.get_bonds().into_iter().map(|x| *x).max().unwrap());
        if max_bond > 100 {
            println!("Max bond dimension reached {} at iteration {}", max_bond, i);
        }
    }
    println!("Max bond during evolution: {}.", max_bond);
    target.set_into_left_canonical().unwrap();
    let target_clone = target.clone();
    target.elementwise_prod(&target_clone).unwrap(); // Now  a target tensor train is a probability distribution
    let argmin = target.log_dot(&tt_idx_to_arg).unwrap().exp(); // Argmin is an averaged argument over a probability distribution
    println!("Resulting argmin = {}", argmin.re);
    println!("Exact argmin = {}", 1.23456789)
}