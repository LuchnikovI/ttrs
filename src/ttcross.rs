use num_complex::ComplexFloat;
use num_complex::{
  Complex32,
  Complex64,
};

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use linwrap::init_utils::{
  random_normal_f32,
  random_normal_f64,
  random_normal_c32,
  random_normal_c64,
  eye_f32,
  eye_f64,
  eye_c32,
  eye_c64,
  uninit_buff_f32,
  uninit_buff_f64,
  uninit_buff_c32,
  uninit_buff_c64,
};
use linwrap::Matrix;


use crate::utils::{
  build_bonds,
  build_random_indices,
  indices_prod,
};

use crate::tt::{
  TTResult,
  TensorTrain,
};

#[derive(Debug, Clone, Copy)]
enum DMRGState {
  ToLeft,
  ToRight,
}

#[derive(Debug)]
pub struct CrossBuilder<T: ComplexFloat> {
  left_indices: Vec<Vec<Vec<usize>>>,
  right_indices: Vec<Vec<Vec<usize>>>,
  kernels: Vec<Vec<T>>,
  right_bonds: Vec<usize>,
  left_bonds:  Vec<usize>,
  mode_dims: Vec<usize>,
  cur_ker: usize,
  dmrg_state: DMRGState,
  delta: T::Real,
}

impl<T: ComplexFloat> CrossBuilder<T> {
  pub(super) fn to_tt(self) -> TensorTrain<T> {
    TensorTrain {
      kernels: self.kernels,
      right_bonds: self.right_bonds,
      left_bonds: self.left_bonds,
      mode_dims: self.mode_dims,
    }
  }
}

macro_rules! impl_cross_builder {
  ($complex_type:ty, $real_type:ty, $fn_gen:ident) => {
    impl CrossBuilder<$complex_type> {
      pub(super) fn new(
        rank: usize,
        delta: $real_type,
        mode_dims: &[usize]
      ) -> Self {
        let (left_bonds, right_bonds) = build_bonds(mode_dims, rank);
        let (left_indices, right_indices) = build_random_indices(mode_dims, &left_bonds, &right_bonds);
        let mut kernels = Vec::with_capacity(mode_dims.len());
        for (left_bond, (dim, right_bond)) in left_bonds.iter().zip(mode_dims.into_iter().zip(right_bonds.iter())) {
          kernels.push($fn_gen(left_bond * dim * right_bond));
        }
        let mode_dims = mode_dims.to_owned();
        Self {
          left_indices,
          right_indices,
          kernels,
          right_bonds,
          left_bonds,
          mode_dims,
          cur_ker: 0,
          dmrg_state: DMRGState::ToRight,
          delta,
        }
      }
    }
  };
}

impl_cross_builder!(f32,       f32, random_normal_f32);
impl_cross_builder!(f64,       f64, random_normal_f64);
impl_cross_builder!(Complex32, f32, random_normal_c32);
impl_cross_builder!(Complex64, f64, random_normal_c64);

macro_rules! impl_next {
  ($complex_type:ty, $fn_uninit_buff:ident, $fn_eye:ident) => {
    impl CrossBuilder<$complex_type> {
      pub(super) fn next(&mut self, f: impl Fn(&[usize]) -> $complex_type + Sync) -> TTResult<()> {
        let cur_ker = self.cur_ker;
        let dim = self.mode_dims[cur_ker];
        let left_bond = self.left_bonds[cur_ker];
        let right_bond = self.right_bonds[cur_ker];
        match self.dmrg_state {
          DMRGState::ToRight => {
            let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
            let left_indices = indices_prod(&self.left_indices[cur_ker], &local_indices);
            let right_indices_ref = &self.right_indices[cur_ker];
            if cur_ker == (self.kernels.len() - 1) {
              self.kernels[cur_ker] = left_indices.into_par_iter().map(|i| {
                f(&i[..])
              }).collect();
              self.dmrg_state = DMRGState::ToLeft;
            } else {
              if left_indices.len() == right_bond {
                self.kernels[cur_ker] = $fn_eye(right_bond);
                self.left_indices[cur_ker + 1] = indices_prod(&self.left_indices[cur_ker], &local_indices);
                self.cur_ker += 1;
              } else {
                let mut m_buff_trans: Vec<_> = right_indices_ref.into_par_iter().flat_map(|rhs| {
                  (&left_indices).into_par_iter().map(|lhs| {
                    let index: Vec<_> = lhs.into_iter().chain(rhs.into_iter()).map(|x| *x).collect();
                    f(&index[..])
                  })
                }).collect();
                let m_trans = Matrix::from_mut_slice(&mut m_buff_trans, left_bond * dim, right_bond)?;
                let mut aux_buff = unsafe { $fn_uninit_buff(right_bond.pow(2)) };
                let aux = Matrix::from_mut_slice(&mut aux_buff, right_bond, right_bond)?;
                unsafe { m_trans.qr(aux)? };  // TODO: turn to the RQ decomposition in order to avoid transposition and extra allocation
                let mut m_buff = unsafe { $fn_uninit_buff(m_buff_trans.len()) };
                let m = Matrix::from_mut_slice(&mut m_buff, right_indices_ref.len(), left_indices.len())?;
                unsafe { m_trans.write_to(m, true)? };
                let mut order = unsafe { m.maxvol(self.delta)? };
                let mut reverse_order = Vec::with_capacity(order.len());
                unsafe { reverse_order.set_len(order.len()) };
                order.iter().enumerate().for_each(|(i, x)| {
                  reverse_order[*x] = i;
                });
                self.kernels[cur_ker] = unsafe { m.gen_from_cols_order(&reverse_order, true) };
                order.resize(right_indices_ref.len(), 0);
                self.left_indices[cur_ker + 1] = order.into_iter().map(|i| left_indices[i].clone()).collect();
                self.cur_ker += 1;
              }
            }
          },
          DMRGState::ToLeft => {
            let local_indices: Vec<Vec<usize>> = (0..dim).map(|x| vec![x]).collect();
            let right_indices = indices_prod(&local_indices, &self.right_indices[cur_ker]);
            let left_indices_ref = &self.left_indices[cur_ker];
            if cur_ker == 0 {
              self.kernels[cur_ker] = right_indices.into_par_iter().map(|i| {
                f(&i[..])
              }).collect();
              self.dmrg_state = DMRGState::ToRight;
            } else {
              if right_indices.len() == left_bond {
                self.kernels[cur_ker] = $fn_eye(left_bond);
                self.right_indices[cur_ker - 1] = indices_prod(&local_indices, &self.right_indices[cur_ker]);
                self.cur_ker -= 1;
              } else {
                let mut m_buff_trans: Vec<_> = left_indices_ref.into_par_iter().flat_map(|lhs| {
                  (&right_indices).into_par_iter().map(|rhs| {
                    let index: Vec<_> = lhs.into_iter().chain(rhs.into_iter()).map(|x| *x).collect();
                    f(&index[..])
                  })
                }).collect();
                let m_trans = Matrix::from_mut_slice(&mut m_buff_trans, right_indices.len(), left_indices_ref.len())?;
                let mut aux_buff = unsafe { $fn_uninit_buff(left_indices_ref.len().pow(2)) };
                let aux = Matrix::from_mut_slice(&mut aux_buff, left_indices_ref.len(), left_indices_ref.len())?;
                unsafe { m_trans.qr(aux)? };  // TODO: turn to the RQ decomposition in order to avoid transposition and extra allocation
                let mut m_buff = unsafe { $fn_uninit_buff(m_buff_trans.len()) };
                let m = Matrix::from_mut_slice(&mut m_buff, left_indices_ref.len(), right_indices.len())?;
                unsafe { m_trans.write_to(m, true)? };
                let mut order = unsafe { m.maxvol(self.delta)? };
                let mut reverse_order = Vec::with_capacity(order.len());
                unsafe { reverse_order.set_len(order.len()) };
                order.iter().enumerate().for_each(|(i, x)| {
                  reverse_order[*x] = i;
                });
                self.kernels[cur_ker] = unsafe { m.gen_from_cols_order(&reverse_order, false) };
                order.resize(left_indices_ref.len(), 0);
                self.right_indices[cur_ker - 1] = order.into_iter().map(|i| right_indices[i].clone()).collect();
                self.cur_ker -= 1;
              }
            }
          },
        }
        Ok(())
      }
    }        
  };
}

impl_next!(f32,       uninit_buff_f32, eye_f32);
impl_next!(f64,       uninit_buff_f64, eye_f64);
impl_next!(Complex32, uninit_buff_c32, eye_c32);
impl_next!(Complex64, uninit_buff_c64, eye_c64);

#[cfg(test)]
mod tests {
  use super::CrossBuilder;
  use num_complex::{
    Complex32,
    Complex64,
  };

  macro_rules! test_cross {
    ($complex_type:ty, $real_type:ty, $acc:expr) => {
      fn cos_sqrt(x: &[usize]) -> $complex_type {
        let total_val: $real_type = x.into_iter().enumerate().map(|(i, val)| {
          (*val as $real_type - 0.5) / (2i64.pow(i as u32) as $real_type)
        }).sum();
        <$complex_type>::cos(<$complex_type>::from(total_val)).sqrt()
      }
  
      let mut builder = CrossBuilder::<$complex_type>::new(25, 0.01, &[2; 20]);
      for _ in 0..(20 * 6) {
        builder.next(cos_sqrt).unwrap();
      }
      let mut tt = builder.to_tt();
      let log_norm = tt.set_into_left_canonical().unwrap();
      let tt_based = (2. * log_norm - 19. * (2 as $real_type).ln()).exp();
      let exact = 2. * (1 as $real_type).sin();
      assert!((tt_based - exact).abs() < $acc);
    };
  }

  #[test]
  fn test_cross() {
    { test_cross!(f32,       f32, 1e-3); }
    { test_cross!(f64,       f64, 1e-10); }
    { test_cross!(Complex32, f32, 1e-3); }
    { test_cross!(Complex64, f64, 1e-10); }
  }
}