use rawpointer::PointerExt;

#[derive(Clone, Copy)]
pub struct ParPtrWrapper<T> (pub T);

unsafe impl<T> Send for ParPtrWrapper<T> {}
unsafe impl<T> Sync for ParPtrWrapper<T> {}

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
