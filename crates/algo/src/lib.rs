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

pub mod accessor;
pub mod prefetcher;
pub mod tuples;

use always_equal::AlwaysEqual;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::Peekable;
use std::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, IntoBytes};

pub trait Page: Sized + 'static {
    type Opaque: Opaque;

    #[must_use]
    fn get_opaque(&self) -> &Self::Opaque;
    #[must_use]
    fn get_opaque_mut(&mut self) -> &mut Self::Opaque;
    #[must_use]
    fn len(&self) -> u16;
    #[must_use]
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[must_use]
    fn get(&self, i: u16) -> Option<&[u8]>;
    #[must_use]
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]>;
    #[must_use]
    fn alloc(&mut self, data: &[u8]) -> Option<u16>;
    fn free(&mut self, i: u16);
    #[must_use]
    fn freespace(&self) -> u16;
    fn clear(&mut self, opaque: Self::Opaque);
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait ReadStream<'r> {
    type Relation: RelationReadTypes;
    type Guards: ExactSizeIterator<Item = <Self::Relation as RelationReadTypes>::ReadGuard<'r>>;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next(&mut self) -> Option<(Self::Item, Self::Guards)>;
    fn next_if<P: FnOnce(&Self::Item) -> bool>(
        &mut self,
        predicate: P,
    ) -> Option<(Self::Item, Self::Guards)>;
    fn into_inner(self) -> Self::Inner;
}

pub trait Relation {
    type Page: Page;
}

pub trait RelationReadTypes: Relation {
    type ReadGuard<'r>: PageGuard + Deref<Target = Self::Page>;
}

pub trait RelationRead: RelationReadTypes {
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWriteTypes: Relation {
    type WriteGuard<'r>: PageGuard + DerefMut<Target = Self::Page>;
}

pub trait RelationWrite: RelationWriteTypes {
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

pub trait RelationLength: Relation {
    fn len(&self) -> u32;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait RelationPrefetch: Relation {
    fn prefetch(&self, id: u32);
}

#[derive(Debug, Default, Clone)]
pub struct Hints {
    pub full: bool,
}

impl Hints {
    #[allow(clippy::needless_update)]
    #[inline]
    pub fn full(self, full: bool) -> Self {
        Self { full, ..self }
    }
}

pub trait RelationReadStreamTypes: RelationReadTypes {
    type ReadStream<'r, I: Iterator>: ReadStream<'r, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch;
}

pub trait RelationReadStream: RelationReadStreamTypes {
    fn read_stream<'r, I: Iterator>(&'r self, iter: I, hints: Hints) -> Self::ReadStream<'r, I>
    where
        I::Item: Fetch;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub trait Bump: 'static {
    #[allow(clippy::mut_from_ref)]
    fn alloc<T>(&self, value: T) -> &mut T;
    #[allow(clippy::mut_from_ref)]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T];
    fn reset(&mut self);
}

impl Bump for bumpalo::Bump {
    #[inline]
    fn alloc<T>(&self, value: T) -> &mut T {
        self.alloc(value)
    }

    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        self.alloc_slice_copy(slice)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }
}

pub type BorrowedIter<'b> = sbsii::borrowed::IntoIter<'b, u32, 1>;
pub type OwnedIter = sbsii::owned::IntoIter<u32, 1>;

pub trait Fetch {
    #[must_use]
    fn fetch(&self) -> OwnedIter;
}

impl Fetch for u32 {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        OwnedIter::from_slice(std::slice::from_ref(self))
    }
}

impl<T, A, B> Fetch for (T, AlwaysEqual<&mut (A, B, OwnedIter)>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let (_, AlwaysEqual((.., list))) = self;
        list.clone()
    }
}

impl<T, A, B, C> Fetch for (T, AlwaysEqual<&mut (A, B, C, OwnedIter)>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let (_, AlwaysEqual((.., list))) = self;
        list.clone()
    }
}

impl<T> Fetch for (T, AlwaysEqual<(u32, u16)>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let (_, AlwaysEqual((x, _))) = self;
        OwnedIter::from_slice(std::slice::from_ref(x))
    }
}

impl Fetch for (u32, u16) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        OwnedIter::from_slice(std::slice::from_ref(&self.0))
    }
}

impl<T> Fetch for (T, AlwaysEqual<((u32, u16), (u32, u16))>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let (_, AlwaysEqual(((x, _), _))) = self;
        OwnedIter::from_slice(std::slice::from_ref(x))
    }
}

pub trait Fetch1 {
    fn fetch_1(&self) -> u32;
}

impl<T, F: Fetch1> Fetch for (T, AlwaysEqual<&mut [F]>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let vec = self.1.0.iter().map(|x| x.fetch_1()).collect::<Vec<_>>();
        OwnedIter::from_slice(vec.as_slice())
    }
}

impl<T, U, F: Fetch1> Fetch for (T, AlwaysEqual<(&mut [F], U)>) {
    #[inline(always)]
    fn fetch(&self) -> OwnedIter {
        let vec = self.1.0.0.iter().map(|x| x.fetch_1()).collect::<Vec<_>>();
        OwnedIter::from_slice(vec.as_slice())
    }
}

pub trait Sequence {
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    #[must_use]
    fn next(&mut self) -> Option<Self::Item>;
    #[must_use]
    fn peek(&mut self) -> Option<&Self::Item>;
    #[must_use]
    fn into_inner(self) -> Self::Inner;
}

impl<T: Ord> Sequence for BinaryHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<I: Iterator> Sequence for Peekable<I> {
    type Item = I::Item;
    type Inner = Peekable<I>;
    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        Iterator::next(self)
    }
    #[inline]
    fn peek(&mut self) -> Option<&I::Item> {
        self.peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl<T> Sequence for VecDeque<T> {
    type Item = T;
    type Inner = std::collections::vec_deque::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop_front()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        self.front()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_iter()
    }
}

/// # Safety
///
/// * `Opaque` must aligned to 8 bytes.
#[allow(unsafe_code)]
pub unsafe trait Opaque: Copy + Send + Sync + FromBytes + IntoBytes + 'static {}
