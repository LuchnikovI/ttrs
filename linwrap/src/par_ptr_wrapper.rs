use rawpointer::PointerExt;

#[derive(Clone, Copy)]
pub struct ParPtrWrapper<T: PointerExt + Clone + Copy> (T);

unsafe impl<T: PointerExt + Clone + Copy> Send for ParPtrWrapper<T> {}
unsafe impl<T: PointerExt + Clone + Copy> Sync for ParPtrWrapper<T> {}

impl<T: PointerExt + Clone + Copy> PointerExt for ParPtrWrapper<T> {
  #[inline(always)]
  unsafe fn add(self, i: usize) -> Self {
    Self(self.0.add(i))
  }
  #[inline(always)]
  unsafe fn dec(&mut self) {
    self.0.dec();
  }
  #[inline(always)]
  unsafe fn inc(&mut self) {
    self.0.inc();
  }
  #[inline(always)]
  unsafe fn offset(self, i: isize) -> Self {
      Self(self.0.offset(i))
  }
  #[inline(always)]
  unsafe fn post_dec(&mut self) -> Self {
    Self(self.0.post_dec())
  }
  #[inline(always)]
  unsafe fn post_inc(&mut self) -> Self {
    Self(self.0.post_inc())
  }
  #[inline(always)]
  unsafe fn pre_dec(&mut self) -> Self {
      Self(self.0.pre_dec())
  }
  #[inline(always)]
  unsafe fn pre_inc(&mut self) -> Self {
      Self(self.0.pre_inc())
  }
  #[inline(always)]
  unsafe fn stride_offset(self, s: isize, index: usize) -> Self {
      Self(self.0.stride_offset(s, index))
  }
  #[inline(always)]
  unsafe fn sub(self, i: usize) -> Self {
      Self(self.0.sub(i))
  }
}

pub trait PointerExtWithDerefAndSend<'a>: PointerExt {
  type Target;
  unsafe fn deref(self) -> &'a Self::Target;
  fn wrap(self) -> ParPtrWrapper<Self>;
}

pub trait PointerExtWithDerefMutAndSend<'a>: PointerExtWithDerefAndSend<'a> {
  unsafe fn deref_mut(self) -> &'a mut Self::Target;
}

impl<'a, T> PointerExtWithDerefAndSend<'a> for *const T
{
  type Target = T;
  #[inline(always)]
  unsafe fn deref(self) -> &'a Self::Target {
      &*self
  }
  fn wrap(self) -> ParPtrWrapper<Self> {
    ParPtrWrapper(self)
  }
}

impl<'a, T> PointerExtWithDerefAndSend<'a> for *mut T 
{
  type Target = T;
  #[inline(always)]
  unsafe fn deref(self) -> &'a Self::Target {
      &*self
  }
  fn wrap(self) -> ParPtrWrapper<Self> {
    ParPtrWrapper(self)
  }
}

impl<'a, T> PointerExtWithDerefMutAndSend<'a> for *mut T 
{
  #[inline(always)]
  unsafe fn deref_mut(self) -> &'a mut Self::Target {
      &mut *self
  }
}

impl<'a, T: PointerExtWithDerefAndSend<'a> + Clone + Copy> ParPtrWrapper<T> {
  pub unsafe fn deref(&self) -> &'a T::Target {
    self.0.deref()
  }
}

impl<'a, T: PointerExtWithDerefMutAndSend<'a> + Clone + Copy> ParPtrWrapper<T> {
  pub unsafe fn deref_mut(&mut self) -> &'a mut T::Target {
    self.0.deref_mut()
  }
}
