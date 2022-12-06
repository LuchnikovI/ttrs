use std::time::Instant;

use linwrap::{
  Matrix,
  init_utils::random_normal_c32,
};

fn main() {
  let m = 50_000;
  let k = 10_000;
  let start = Instant::now();
  let mut buff1 = random_normal_c32(m * k);
  let m1: Matrix<_, _> = buff1.as_mut_slice().into();
  let mut m1 = m1.reshape(m, k).unwrap();
  let duration = start.elapsed();
  println!("Random initialization of a complex64 matrix of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
  let buff2 = random_normal_c32(m * k);
  let m2: Matrix<_, _> = buff2.as_slice().into();
  let m2 = m2.reshape(m, k).unwrap();
  let start = Instant::now();
  m1.add(m2).unwrap();
  let duration = start.elapsed();
  println!("Summation of two complex64 matrices of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
  let start = Instant::now();
  m1.mul(m2).unwrap();
  let duration = start.elapsed();
  println!("Multiplication of two complex64 matrices of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
  let start = Instant::now();
  m1.sub(m2).unwrap();
  let duration = start.elapsed();
  println!("Subtraction of two complex64 matrices of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
  let start = Instant::now();
  m1.div(m2).unwrap();
  let duration = start.elapsed();
  println!("Division of two complex64 matrices of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
  let start = Instant::now();
  let _ = m2.norm_n_pow_n(2);
  let duration = start.elapsed();
  println!("Calculation of the L2 norm of a complex64 matrix of size {}x{} takes: {} secs;", m, k, duration.as_nanos() as f64 / 1e9);
}