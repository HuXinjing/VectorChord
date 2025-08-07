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

#[derive(Clone)]
pub struct HeapIntoIter<'a, T> {
    inner: std::iter::Copied<std::slice::Iter<'a, T>>,
}

impl<'a, T: Copy> HeapIntoIter<'a, T> {
    #[cold]
    pub(crate) fn from_slice(slice: &[T], alloc: impl Fn(&[T]) -> &'a [T]) -> Self {
        assert!(slice.len() <= 65535_usize);
        Self {
            inner: alloc(slice).iter().copied(),
        }
    }
}

impl<'a, T: Copy> Iterator for HeapIntoIter<'a, T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[derive(Clone)]
pub enum IntoIter<'a, T: Copy, const N: usize> {
    Stack(StackIntoIter<T, N>),
    Heap(HeapIntoIter<'a, T>),
}

impl<'a, T: Copy, const N: usize> IntoIter<'a, T, N> {
    #[inline(always)]
    pub fn from_slice(slice: &[T], alloc: impl Fn(&[T]) -> &'a [T]) -> Self {
        assert!(slice.len() <= 65535);
        if slice.len() <= N && N <= 65535 {
            IntoIter::Stack(StackIntoIter::from_slice(slice))
        } else {
            IntoIter::Heap(HeapIntoIter::from_slice(slice, alloc))
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
