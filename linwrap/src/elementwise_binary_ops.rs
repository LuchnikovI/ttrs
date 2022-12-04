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
    other: Matrix<Ptr2, Ref2>,
    op: impl Fn(&mut Ptr1::Target, &Ptr2::Target) + Send + Sync,
  ) -> MatrixResult<()>
  where
    Ptr2: PointerExtWithDerefAndSend<'a, Target = Ptr1::Target>,
  {
    if (other.nrows != self.nrows) || (other.ncols != self.ncols) { return Err(MatrixError::IncorrectShape); }
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
    let m_dst: Matrix<_, _> = buff1.as_mut_slice().into();
    let (m_dst0, m_dst1) = m_dst.reshape(10, 10).unwrap().col_split(5).unwrap();
    let (mut m_dst00, mut m_dst01) = m_dst0.row_split(5).unwrap();
    let (mut m_dst10, mut m_dst11) = m_dst1.row_split(5).unwrap();
    let buff2 = (1..101).map(|x| x as f64).collect::<Vec<_>>();
    let m_src: Matrix<_, _> = buff2.as_slice().into();
    let (m_src0, m_src1) = m_src.reshape(10, 10).unwrap().col_split(5).unwrap();
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
    let res_true: Matrix<_, _> = true_buff.as_mut_slice().into();
    let mut res_true = res_true.reshape(10, 10).unwrap();
    let res: Matrix<_, _> = buff1.as_slice().into();
    let res = res.reshape(10, 10).unwrap();
    res_true.sub(res).unwrap();
    assert_eq!(0., res_true.norm_n_pow_n(2));
  }
}