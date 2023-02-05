/*use rayon::prelude::{
  ParallelIterator,
  IntoParallelIterator,
};
use rayon::iter::{
  IndexedParallelIterator,
  IntoParallelRefMutIterator,
};*/
use rand::{
  thread_rng,
  Rng,
};
use rand_distr::StandardNormal;
use num_complex::{
  Complex32,
  Complex64,
  ComplexFloat,
};

pub trait BufferGenerator: ComplexFloat {
  fn random_normal(size: usize) -> Vec<Self>;
  fn eye(m: usize) -> Vec<Self>;
  unsafe fn uninit_buff(size: usize) -> Vec<Self>;
}

macro_rules! float_buffer_generator_impl
{
  ($type:ty, $zero:expr, $one:expr) => {
    impl BufferGenerator for $type
    {
      fn random_normal(size: usize) -> Vec<Self> {
        (0..size).into_iter().map(|_| {
          let mut rng = thread_rng();
          rng.sample::<$type, _>(StandardNormal)
        }).collect()
      }
      fn eye(m: usize) -> Vec<Self> {
        let mut buff = Vec::with_capacity(m * m);
        unsafe { buff.set_len(m * m); }
        let r = (0..(m * m)).into_iter();
        r.zip(buff.iter_mut()).for_each(|(i, x)| {
          *x = $zero;
          if i % (m + 1) == 0 {
            *x = $one;
          }
        });
        buff
      }
      unsafe fn uninit_buff(size: usize) -> Vec<Self> {
        let mut buff = Vec::with_capacity(size);
        buff.set_len(size);
        buff
      }
    }
  };
}

float_buffer_generator_impl!(f32, 0f32, 1f32);
float_buffer_generator_impl!(f64, 0f64, 1f64);

macro_rules! complex_buffer_generator_impl
{
  ($type:ty, $zero:expr, $one:expr) => {
    impl BufferGenerator for $type
    {
      fn random_normal(size: usize) -> Vec<Self> {
        (0..size).into_iter().map(|_| {
          let mut rng = thread_rng();
          <$type>::new(
            rng.sample::<<$type as ComplexFloat>::Real, _>(StandardNormal),
            rng.sample::<<$type as ComplexFloat>::Real, _>(StandardNormal),
          )
        }).collect()
      }
      fn eye(m: usize) -> Vec<Self> {
        let mut buff = Vec::with_capacity(m * m);
        unsafe { buff.set_len(m * m); }
        let r = (0..(m * m)).into_iter();
        r.zip(buff.iter_mut()).for_each(|(i, x)| {
          *x = $zero;
          if i % (m + 1) == 0 {
            *x = $one;
          }
        });
        buff
      }
      unsafe fn uninit_buff(size: usize) -> Vec<Self> {
        let mut buff = Vec::with_capacity(size);
        buff.set_len(size);
        buff
      }
    }
  };
}

complex_buffer_generator_impl!(Complex32, Complex32::new(0f32, 0f32), Complex32::new(1f32, 0f32));
complex_buffer_generator_impl!(Complex64, Complex64::new(0f64, 0f64), Complex64::new(1f64, 0f64));
