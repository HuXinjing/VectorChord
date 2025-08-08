// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use crate::stack::StackIntoIter;
use std::ptr::NonNull;

pub struct HeapIntoIter<T: Copy> {
    pointer: NonNull<T>,
    off: u16,
    len: u16,
}

impl<T: Copy> HeapIntoIter<T> {
    #[cold]
    pub(crate) fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= 65535_usize);
        let c = box_into_non_null::<[T]>(Box::from(slice)).cast::<T>();
        Self {
            pointer: c,
            off: 0,
            len: slice.len() as u16,
        }
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                self.pointer.as_ptr().add(self.off as _),
                (self.len - self.off) as _,
            )
        }
    }
}

impl<T: Copy> Iterator for HeapIntoIter<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        #[allow(unsafe_code)]
        unsafe {
            if self.off < self.len {
                let r = self.pointer.as_ptr().add(self.off as _).read();
                self.off += 1;
                Some(r)
            } else {
                None
            }
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.len - self.off) as usize;
        (size, Some(size))
    }
}

impl<T: Copy> ExactSizeIterator for HeapIntoIter<T> {}

impl<T: Copy> Clone for HeapIntoIter<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        #[allow(unsafe_code)]
        unsafe {
            let p = std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len as _);
            let c = NonNull::new_unchecked(Box::<[T]>::into_raw(Box::from(p)).cast::<T>());
            Self {
                pointer: c,
                off: self.off,
                len: self.len,
            }
        }
    }
}

impl<T: Copy> Drop for HeapIntoIter<T> {
    #[inline(always)]
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            let p = std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len as _);
            let _ = Box::<[T]>::from_raw(p);
        }
    }
}

#[derive(Clone)]
pub enum IntoIter<T: Copy, const N: usize> {
    Stack(StackIntoIter<T, N>),
    Heap(HeapIntoIter<T>),
}

impl<T: Copy, const N: usize> IntoIter<T, N> {
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= 65535);
        if slice.len() <= N && N <= 65535 {
            IntoIter::Stack(StackIntoIter::from_slice(slice))
        } else {
            IntoIter::Heap(HeapIntoIter::from_slice(slice))
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        match self {
            IntoIter::Stack(x) => x.as_slice(),
            IntoIter::Heap(x) => x.as_slice(),
        }
    }
}

impl<T: Copy, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Stack(iter) => iter.next(),
            IntoIter::Heap(iter) => iter.next(),
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Stack(iter) => iter.size_hint(),
            IntoIter::Heap(iter) => iter.size_hint(),
        }
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<IntoIter::<u32, 4>>() == 24);
};

// Emulate unstable library feature `box_vec_non_null`.
// See https://github.com/rust-lang/rust/issues/130364.

#[allow(dead_code)]
#[must_use]
fn box_into_non_null<T: ?Sized>(b: Box<T>) -> NonNull<T> {
    #[allow(unsafe_code)]
    unsafe {
        NonNull::new_unchecked(Box::into_raw(b))
    }
}

#[test]
fn tests() {
    for x in IntoIter::<u32, 5>::from_slice(&[1; 0]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 1]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 2]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 3]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 4]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 5]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 6]).clone() {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 7]).clone() {
        assert_eq!(x, 1);
    }
}
