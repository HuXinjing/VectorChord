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

use std::mem::MaybeUninit;

#[derive(Clone)]
pub struct StackIntoIter<T: Copy, const N: usize> {
    start: u16,
    end: u16,
    buffer: [MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> StackIntoIter<T, N> {
    #[inline(always)]
    pub(crate) fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= N && N <= 65535);
        Self {
            start: 0,
            end: slice.len() as u16,
            buffer: {
                let mut buffer = [const { MaybeUninit::uninit() }; N];
                for i in 0..slice.len() {
                    buffer[i] = MaybeUninit::new(slice[i]);
                }
                buffer
            },
        }
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.as_ptr().add(self.start as _).cast(),
                (self.end - self.start) as _,
            )
        }
    }
}

impl<T: Copy, const N: usize> Iterator for StackIntoIter<T, N> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        #[allow(unsafe_code)]
        unsafe {
            if self.start < self.end {
                let r = self.buffer[self.start as usize].assume_init();
                self.start += 1;
                Some(r)
            } else {
                None
            }
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.end - self.start) as usize;
        (size, Some(size))
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for StackIntoIter<T, N> {}
