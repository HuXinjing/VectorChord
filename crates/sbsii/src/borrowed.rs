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
use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Clone)]
pub struct HeapIntoIter<'a, T: Copy> {
    pointer: NonNull<T>,
    off: u16,
    len: u16,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, T: Copy> HeapIntoIter<'a, T> {
    #[cold]
    pub(crate) fn from_slice(slice: &'a [T]) -> Self {
        assert!(slice.len() <= 65535_usize);
        let c = NonNull::from_ref(slice).cast::<T>();
        Self {
            pointer: c,
            off: 0,
            len: slice.len() as u16,
            _phantom: PhantomData,
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

impl<'a, T: Copy> Iterator for HeapIntoIter<'a, T> {
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

impl<'a, T: Copy> ExactSizeIterator for HeapIntoIter<'a, T> {}

#[derive(Clone)]
pub enum IntoIter<'a, T: Copy, const N: usize> {
    Stack(StackIntoIter<T, N>),
    Heap(HeapIntoIter<'a, T>),
}

impl<'a, T: Copy + 'a, const N: usize> IntoIter<'a, T, N> {
    #[inline(always)]
    pub fn from_slice(slice: &[T], alloc: impl Fn(&[T]) -> &'a [T]) -> Self {
        assert!(slice.len() <= 65535);
        if slice.len() <= N && N <= 65535 {
            IntoIter::Stack(StackIntoIter::from_slice(slice))
        } else {
            IntoIter::Heap(HeapIntoIter::from_slice(alloc(slice)))
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

impl<'a, T: Copy, const N: usize> Iterator for IntoIter<'a, T, N> {
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

impl<'a, T: Copy, const N: usize> ExactSizeIterator for IntoIter<'a, T, N> {}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<IntoIter::<'static, u32, 1>>() == 16);
};

#[test]
fn tests() {
    let alloc = |slice: &[u32]| -> &'static [u32] {
        // make miri happy
        static GLOBAL: std::sync::Mutex<Vec<&'static [u32]>> = std::sync::Mutex::new(Vec::new());
        let pointer = Vec::leak(slice.to_vec());
        GLOBAL.lock().expect("failed to lock").push(pointer);
        pointer
    };
    for x in IntoIter::<u32, 5>::from_slice(&[1; 0], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 1], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 2], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 3], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 4], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 5], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 6], alloc) {
        assert_eq!(x, 1);
    }
    for x in IntoIter::<u32, 5>::from_slice(&[1; 7], alloc) {
        assert_eq!(x, 1);
    }
}
