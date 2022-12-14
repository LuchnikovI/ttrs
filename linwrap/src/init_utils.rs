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
};

/// This function generates a vector of f32 numbers that are i.i.d. normal random values
/// with mu = 0 and std = 1.
pub fn random_normal_f32(size: usize) -> Vec<f32> {
  (0..size).into_iter().map(|_| {
    let mut rng = thread_rng();
    rng.sample::<f32, _>(StandardNormal)
  }).collect()
}

/// This function generates a vector of f64 numbers that are i.i.d. normal random values
/// with mu = 0 and std = 1.
pub fn random_normal_f64(size: usize) -> Vec<f64> {
  (0..size).into_iter().map(|_| {
    let mut rng = thread_rng();
    rng.sample::<f64, _>(StandardNormal)
  }).collect()
}

/// This function generates a vector of Complex32 numbers whose real and imag parts are i.i.d. normal random values
/// with mu = 0 and std = 1.
pub fn random_normal_c32(size: usize) -> Vec<Complex32> {
  (0..size).into_iter().map(|_| {
    let mut rng = thread_rng();
    Complex32::new(
      rng.sample::<f32, _>(StandardNormal),
      rng.sample::<f32, _>(StandardNormal)
    )
  }).collect()
}

/// This function generates a vector of Complex64 numbers whose real and imag parts are i.i.d. normal random values
/// with mu = 0 and std = 1.
pub fn random_normal_c64(size: usize) -> Vec<Complex64> {
  (0..size).into_iter().map(|_| {
    let mut rng = thread_rng();
    Complex64::new(
      rng.sample::<f64, _>(StandardNormal),
      rng.sample::<f64, _>(StandardNormal),
    )
  }).collect()
}

macro_rules! eye {
  ($fn_name:ident, $type_name:ident, $complex_one:expr, $complex_zero:expr) => {
    /// This function generates a vector that corresponds to an identity matrix
    /// of size m x m.
    pub fn $fn_name(m: usize) -> Vec<$type_name> {
      let mut buff = Vec::with_capacity(m * m);
      unsafe { buff.set_len(m * m); }
      let r = (0..(m * m)).into_iter();
      r.zip(buff.iter_mut()).for_each(|(i, x)| {
        *x = $complex_zero;
        if i % (m + 1) == 0 {
          *x = $complex_one;
        }
      });
      buff
    }
  };
}

eye!(eye_f32, f32,       1.                    , 0.                    );
eye!(eye_f64, f64,       1.                    , 0.                    );
eye!(eye_c32, Complex32, Complex32::new(1., 0.), Complex32::new(0., 0.));
eye!(eye_c64, Complex64, Complex64::new(1., 0.), Complex64::new(0., 0.));


macro_rules! uninit_buff {
  ($fn_name:ident, $complex_type:ident) => {
    /// This function generates an uninitialized vector of a given size.
    /// Safety: one can not read values from the vector before initialization.
    pub unsafe fn $fn_name(size: usize) -> Vec<$complex_type> {
      let mut buff = Vec::with_capacity(size);
      buff.set_len(size);
      buff
    }
  };
}

uninit_buff!(uninit_buff_f32, f32      );
uninit_buff!(uninit_buff_f64, f64      );
uninit_buff!(uninit_buff_c32, Complex32);
uninit_buff!(uninit_buff_c64, Complex64);