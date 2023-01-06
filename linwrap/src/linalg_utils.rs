use num_complex::ComplexFloat;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::NDArray;

pub(super) unsafe fn triangular_split<T: ComplexFloat + Send + Sync + 'static>(
  mut to_split: NDArray<*mut T, 2>,
  second_part: NDArray<*mut T, 2>,
)
{
  let m = to_split.shape[0];
  let n = to_split.shape[1];
  if m > n {
    to_split.shape[0] = n;
    to_split.into_f_iter()
      .zip(second_part.into_f_iter())
      .enumerate()
      .for_each(|(counter, (x, y))| {
        let i = counter % n;
        let j = counter / n;
        if i <= j {
          *y.0 = *x.0;
          *x.0 = T::zero();
        } else {
          *y.0 = T::zero();
        }
     });
    to_split.shape[0] = m;
  } else {
    to_split.shape[1] = m;
    to_split.into_f_iter()
      .zip(second_part.into_f_iter())
      .enumerate()
      .for_each(|(counter, (x, y))| {
        let i = counter % m;
        let j = counter / m;
        if i > j {
          *y.0 = *x.0;
          *x.0 = T::zero();
        } else {
          *y.0 = T::zero();
        }
     });
     to_split.shape[1] = n;
  }
}

/*#[cfg(test)]
mod test {
  use super::triangular_split;
  use crate::init_utils::random_normal_f32;
  use crate::NDArray;
  #[test]
  fn test_triangular_split() {
    let mut buff_to_split = random_normal_f32(50);
    let to_split = NDArray::from_mut_slice(&mut buff_to_split, [10, 5]).unwrap();
    let mut buff_other: Vec<f32> = Vec::with_capacity(50);
    unsafe { buff_other.set_len(25); }
    let other = NDArray::from_mut_slice(&mut buff_other, [5, 5]).unwrap();
    unsafe { triangular_split(to_split, other); }
    unsafe {
      println!("{}", to_split.to_string());
      println!("{}", other.to_string());
    }
  }
}*/