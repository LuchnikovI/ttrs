use num_complex::ComplexFloat;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::Matrix;

pub(super) unsafe fn triangular_split<T: ComplexFloat + Send + Sync>(
  mut to_split: Matrix<*mut T>,
  second_part: Matrix<*mut T>,
)
{
  let m = to_split.nrows;
  let n = to_split.ncols;
  if m > n {
    to_split.nrows = n;
    (
      0..(n * n),
      to_split.into_par_iter(),
      second_part.into_par_iter()
    ).into_par_iter()
     .for_each(|(counter, x, y)| {
       let i = counter % n;
       let j = counter / n;
       if i <= j {
         *y.0 = *x.0;
         *x.0 = T::zero();
       } else {
         *y.0 = T::zero();
       }
     });
    to_split.nrows = m;
  } else {
    to_split.ncols = m;
    (
      0..(m * m),
      to_split.into_par_iter(),
      second_part.into_par_iter()
    ).into_par_iter()
     .for_each(|(counter, x, y)| {
       let i = counter % m;
       let j = counter / m;
       if i > j {
         *y.0 = *x.0;
         *x.0 = T::zero();
       } else {
         *y.0 = T::zero();
       }
     });
     to_split.ncols = n;
  }
}

/*#[cfg(test)]
mod test {
  use super::triangular_split;
  use crate::init_utils::random_normal_f32;
  use crate::Matrix;
  #[test]
  fn test_triangular_split() {
    let mut buff_to_split = random_normal_f32(50);
    let to_split = Matrix::from_mut_slice(&mut buff_to_split, 10, 5).unwrap();
    let mut buff_other: Vec<f32> = Vec::with_capacity(50);
    unsafe { buff_other.set_len(25); }
    let other = Matrix::from_mut_slice(&mut buff_other, 5, 5).unwrap();
    unsafe { triangular_split(to_split, other); }
    unsafe {
      println!("{:?}", to_split.to_string());
      println!("{:?}", other.to_string());
    }
  }
}*/