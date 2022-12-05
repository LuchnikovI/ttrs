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
use crate::par_ptr_wrapper::{
  PointerExtWithDerefAndSend,
  PointerExtWithDerefMutAndSend,
};

// ---------------------------------------------------------------------- //

// TODO: complete binary operations to the full list
// TODO: make testing universal

// ---------------------------------------------------------------------- //

impl<'a, Ptr1, Ref1> Matrix<Ptr1, Ref1>
where
  Ptr1: PointerExtWithDerefMutAndSend<'a>,
  Ptr1::Target: ComplexFloat + Send + Sync + 'a,
  <Ptr1::Target as ComplexFloat>::Real: Sum + Send + Sync,
{
  fn bin_op<Ptr2, Ref2>(
    &mut self,
    mut other: Matrix<Ptr2, Ref2>,
    op: impl Fn(&mut Ptr1::Target, &Ptr2::Target) + Send + Sync,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
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
    self.into_par_iter_mut().zip(other_iter).for_each(|(lhs, rhs)| {
      op(lhs, rhs);
    });
    Ok(())
  }

  pub fn add<Ptr2, Ref2>(
    &mut self,
    other: Matrix<Ptr2, Ref2>,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
    self.bin_op(other, |x, y| *x = *x + *y )?;
    Ok(())
  }

  pub fn sub<Ptr2, Ref2>(
    &mut self,
    other: Matrix<Ptr2, Ref2>,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
    self.bin_op(other, |x, y| *x = *x - *y )?;
    Ok(())
  }

  pub fn mul<Ptr2, Ref2>(
    &mut self,
    other: Matrix<Ptr2, Ref2>,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
    self.bin_op(other, |x, y| *x = *x * *y )?;
    Ok(())
  }

  pub fn div<Ptr2, Ref2>(
    &mut self,
    other: Matrix<Ptr2, Ref2>,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
    self.bin_op(other, |x, y| *x = *x / *y )?;
    Ok(())
  }

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
    let (mut m_dst00, mut m_dst01) = m_dst0.row_split(5).unwrap();
    let (mut m_dst10, mut m_dst11) = m_dst1.row_split(5).unwrap();
    let buff2 = (1..101).map(|x| x as f64).collect::<Vec<_>>();
    let m_src = Matrix::from_slice(&buff2, 10, 10).unwrap();
    let (m_src0, m_src1) = m_src.col_split(5).unwrap();
    let (m_src00, m_src01) = m_src0.row_split(5).unwrap();
    let (m_src10, m_src11) = m_src1.row_split(5).unwrap();
    m_dst00.add(m_src00).unwrap();
    m_dst01.sub(m_src01).unwrap();
    m_dst10.mul(m_src10).unwrap();
    m_dst11.div(m_src11).unwrap();
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
    let mut res_true = Matrix::from_mut_slice(&mut true_buff, 10, 10).unwrap();
    let res = Matrix::from_slice(&buff1, 10, 10).unwrap();
    res_true.sub(res).unwrap();
    assert_eq!(0., res_true.norm_n_pow_n(2));
  }

  #[test]
  fn test_broadcasting() {
    let mut m_buff = (1..6).flat_map(|x| (1..21).map(move |y| x * y)).map(|x| x as f64).collect::<Vec<_>>();
    let mut m_dst = Matrix::from_mut_slice(&mut m_buff, 20, 5).unwrap();
    let col_buff = (1..21).map(|x| x as f64).collect::<Vec<_>>();
    let col = Matrix::from_slice(&col_buff, 20, 1).unwrap();
    let row_buff = (1..6).map(|x| x as f64).collect::<Vec<_>>();
    let row = Matrix::from_slice(&row_buff, 1, 5).unwrap();
    m_dst.div(col).unwrap();
    m_dst.sub(row).unwrap();
    assert_eq!(m_dst.norm_n_pow_n(2), 0.)
  }
}