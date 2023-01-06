use crate::ndarray::Layout;

#[inline]
pub(super) fn shape_to_strides<const N: usize>(
    shape: [usize; N],
    layout: Layout
) -> Option<[usize; N]> {
    let mut strides = [1; N];
    match layout {
        Layout::Fortran => {
            for (i, s) in shape.into_iter().enumerate().take(N-1) {
                strides[i+1] = strides[i] * s;
            }
            Some(strides)
        },
        Layout::C       => {
            for (i, s) in shape.into_iter().enumerate().rev().take(N-1) {
                strides[i-1] = strides[i] * s;
            }
            Some(strides)
        },
        Layout::General => {
            None
        },
    }
}

#[inline]
pub(super) fn get_cache_friendly_order<const N: usize>(
    mut strides: [usize; N],
    mut shape: [usize; N]
) -> ([usize; N], [usize; N])
{
    let mut strides_and_shape = [(0, 0); N];
    for ((st, sh), stsh) in strides.into_iter().zip(shape).zip(strides_and_shape.iter_mut()) {
      *stsh = (st, sh)
    }
    strides_and_shape.sort_by_key(|(st, _)| {
      *st
    });
    for ((st, sh), stsh) in strides.iter_mut().zip(shape.iter_mut()).zip(strides_and_shape.into_iter()) {
      *st = stsh.0;
      *sh = stsh.1;
    }
    (strides, shape)
}

#[cfg(test)]
mod tests {
    use crate::ndarray::Layout;
    use super::{shape_to_strides, get_cache_friendly_order};

    #[test]
    fn test_shape_to_strides() {
        let shape = [2, 3, 1, 4];
        let strides = shape_to_strides(shape, Layout::Fortran);
        assert_eq!(strides, Some([1, 2, 6, 6]));
        let strides = shape_to_strides(shape, Layout::C);
        assert_eq!(strides, Some([12, 4, 4, 1]));
    }

    #[test]
    fn test_get_cache_friendly_order() {
        let mut shape = [2, 3, 2, 8, 6];
        let mut strides = [3, 1, 6, 72, 12];
        let friendly_shape = [3, 2, 2, 6, 8];
        let friendly_strides = [1, 3, 6, 12, 72];
        (strides, shape) = get_cache_friendly_order(strides, shape);
        assert_eq!(shape, friendly_shape);
        assert_eq!(strides, friendly_strides);
    }
}