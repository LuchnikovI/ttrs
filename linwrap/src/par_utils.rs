use rayon::prelude::{ParallelIterator, IntoParallelIterator};
use rand::{
  thread_rng,
  Rng,
};
use rand_distr::StandardNormal;
use num_complex::Complex;

pub fn gen_random_normal_buff_f32(size: usize) -> Vec<f32> {
  (0..size).into_par_iter().map(|_| {
    let mut rng = thread_rng();
    rng.sample::<f32, _>(StandardNormal)
  }).collect()
}

pub fn gen_random_normal_buff_f64(size: usize) -> Vec<f64> {
  (0..size).into_par_iter().map(|_| {
    let mut rng = thread_rng();
    rng.sample::<f64, _>(StandardNormal)
  }).collect()
}

pub fn gen_random_normal_buff_c64(size: usize) -> Vec<Complex<f32>> {
  (0..size).into_par_iter().map(|_| {
    let mut rng = thread_rng();
    Complex::<f32>::new(
      rng.sample::<f32, _>(StandardNormal),
      rng.sample::<f32, _>(StandardNormal)
    )
  }).collect()
}

pub fn gen_random_normal_buff_c128(size: usize) -> Vec<Complex<f64>> {
  (0..size).into_par_iter().map(|_| {
    let mut rng = thread_rng();
    Complex::<f64>::new(
      rng.sample::<f64, _>(StandardNormal),
      rng.sample::<f64, _>(StandardNormal)
    )
  }).collect()
}