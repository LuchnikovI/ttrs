use std::iter::Sum;
use num_complex::ComplexFloat;
use rayon::prelude::{
  ParallelIterator,
  IndexedParallelIterator,
};

use crate::{
  Matrix,
  matrix::{
    MatrixResult,
    MatrixError
  },
};

// ---------------------------------------------------------------------- //

// TODO: complete binary operations to the full list
// TODO: make testing universal

// ---------------------------------------------------------------------- //

macro_rules! elementwise_bin_fn {
  ($fn_name:ident, $body:expr) => {
    #[inline]
    pub unsafe fn $fn_name(
      self,
      other: impl Into<Matrix<*const T>>,
    ) -> MatrixResult<()>
    {
      let mut other: Matrix<*const T> = other.into();
      match (other.nrows, other.ncols) {
        (1, n) => {
          if n != self.ncols { return Err(MatrixError::IncorrectShape); }
          other.nrows = self.nrows;
          other.stride1 = 0;
        },
        (m, 1) => {
          if m != self.nrows { return Err(MatrixError::IncorrectShape); }
          other.ncols = self.ncols;
          other.stride2 = 0;
        },
        (m, n) => {
          if (m != self.nrows) || (n != self.ncols) { return Err(MatrixError::IncorrectShape); }
        },
      }
      let other_iter = other.into_par_iter();
      self.into_par_iter().zip(other_iter).for_each($body);
      Ok(())
    }
  };
}

impl<T> Matrix<*mut T>
where
  T: ComplexFloat + Send + Sync,
  <T as ComplexFloat>::Real: Sum + Send + Sync,
{
  elementwise_bin_fn!(add, |(x, y)| *x.0 = *x.0 + *y.0);
  elementwise_bin_fn!(sub, |(x, y)| *x.0 = *x.0 - *y.0);
  elementwise_bin_fn!(mul, |(x, y)| *x.0 = *x.0 * *y.0);
  elementwise_bin_fn!(div, |(x, y)| *x.0 = *x.0 / *y.0);
}

// ---------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
  use crate::Matrix;
  #[test]
  fn test_bin_ops() {
    let mut buff1 = (0..100).map(|x| x as f64).collect::<Vec<_>>();
    let m_dst = Matrix::from_mut_slice(&mut buff1, 10, 10).unwrap();
    let (m_dst0, m_dst1) = m_dst.col_split(5).unwrap();
    let (m_dst00, m_dst01) = m_dst0.row_split(5).unwrap();
    let (m_dst10, m_dst11) = m_dst1.row_split(5).unwrap();
    let buff2 = (1..101).map(|x| x as f64).collect::<Vec<_>>();
    let m_src = Matrix::from_slice(&buff2, 10, 10).unwrap();
    let (m_src0, m_src1) = m_src.col_split(5).unwrap();
    let (m_src00, m_src01) = m_src0.row_split(5).unwrap();
    let (m_src10, m_src11) = m_src1.row_split(5).unwrap();
    unsafe {
      m_dst00.add(m_src00).unwrap();
      m_dst01.sub(m_src01).unwrap();
      m_dst10.mul(m_src10).unwrap();
      m_dst11.div(m_src11).unwrap();
    }
    let mut true_buff = Vec::with_capacity(100);
    unsafe { true_buff.set_len(100) };
    for i in 0..5 {
      for j in 0..5 {
        true_buff[j + 10 * i] = 1. + 2. * (j + 10 * i) as f64;
        true_buff[5 + j + 10 * i] = -1.;
        true_buff[j + 10 * (i + 5)] = ((j + 10 * (i + 5)) as f64) * ((1 + j + 10 * (i + 5)) as f64);
        true_buff[5 + j + 10 * (i + 5)] = ((5 + j + 10 * (i + 5)) as f64) / ((1 + 5 + j + 10 * (i + 5)) as f64);
      }
    }
    let res_true = Matrix::from_mut_slice(&mut true_buff, 10, 10).unwrap();
    let res = Matrix::from_slice(&buff1, 10, 10).unwrap();
    unsafe {
      res_true.sub(res).unwrap();
      assert_eq!(0., res_true.norm_n_pow_n(2));
    }
  }

  #[test]
  fn test_broadcasting() {
    let mut m_buff = (1..6).flat_map(|x| (1..21).map(move |y| x * y)).map(|x| x as f64).collect::<Vec<_>>();
    let m_dst = Matrix::from_mut_slice(&mut m_buff, 20, 5).unwrap();
    let col_buff = (1..21).map(|x| x as f64).collect::<Vec<_>>();
    let col = Matrix::from_slice(&col_buff, 20, 1).unwrap();
    let row_buff = (1..6).map(|x| x as f64).collect::<Vec<_>>();
    let row = Matrix::from_slice(&row_buff, 1, 5).unwrap();
    unsafe {
      m_dst.div(col).unwrap();
      m_dst.sub(row).unwrap();
      assert_eq!(m_dst.norm_n_pow_n(2), 0.)
    }
  }
}