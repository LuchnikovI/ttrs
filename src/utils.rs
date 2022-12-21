use num_traits::Float;

pub(super) fn get_trunc_dim<T: Float>(lmbd: &[T], delta_local: T) -> usize {
  let mut acc = T::zero();
  let mut counter = 0;
  for l in lmbd.into_iter().rev() {
    acc = acc + l.powi(2);
    if acc.sqrt() > delta_local { break; }
    counter += 1;
  }
  lmbd.len() - counter
}

#[cfg(test)]
mod tests {
    use super::get_trunc_dim;

  #[test]
  fn test_get_trunc_dim() {
    let lmbd = [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.];
    assert_eq!(10, get_trunc_dim(&lmbd, 0.9999));
    assert_eq!(9, get_trunc_dim(&lmbd, 1.0001));
    assert_eq!(9, get_trunc_dim(&lmbd, 5f32.sqrt() - 0.0001));
    assert_eq!(8, get_trunc_dim(&lmbd, 5f32.sqrt() + 0.0001));
    assert_eq!(8, get_trunc_dim(&lmbd, 14f32.sqrt() - 0.0001));
    assert_eq!(7, get_trunc_dim(&lmbd, 14f32.sqrt() + 0.0001));
    assert_eq!(7, get_trunc_dim(&lmbd, 30f32.sqrt() - 0.0001));
    assert_eq!(6, get_trunc_dim(&lmbd, 30f32.sqrt() + 0.0001));
    assert_eq!(6, get_trunc_dim(&lmbd, 55f32.sqrt() - 0.0001));
    assert_eq!(5, get_trunc_dim(&lmbd, 55f32.sqrt() + 0.0001));
    assert_eq!(5, get_trunc_dim(&lmbd, 91f32.sqrt() - 0.0001));
    assert_eq!(4, get_trunc_dim(&lmbd, 91f32.sqrt() + 0.0001));
  }
}