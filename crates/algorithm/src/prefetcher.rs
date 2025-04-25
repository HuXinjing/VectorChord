use crate::{Fetch, Heap, ReadStream, RelationPrefetch, RelationRead, RelationReadStream};
use std::collections::{BinaryHeap, VecDeque, binary_heap, vec_deque};
use std::iter::Chain;

pub const WINDOW_SIZE: usize = 32;
const _: () = assert!(WINDOW_SIZE > 0);

pub trait Prefetcher: IntoIterator
where
    Self::Item: Fetch,
{
    type R: RelationRead;
    fn pop_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(Self::Item, Vec<<Self::R as RelationRead>::ReadGuard<'_>>)>;
}

pub struct PlainPrefetcher<R, H> {
    relation: R,
    heap: H,
}

impl<R, H: Heap> PlainPrefetcher<R, H> {
    pub fn new(relation: R, vec: Vec<H::Item>) -> Self {
        Self {
            relation,
            heap: Heap::make(vec),
        }
    }
}

impl<R, H: Heap> IntoIterator for PlainPrefetcher<R, H> {
    type Item = H::Item;

    type IntoIter = H::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.heap.into_iter()
    }
}

impl<R: RelationRead, H: Heap> Prefetcher for PlainPrefetcher<R, H>
where
    H::Item: Fetch + Ord,
{
    type R = R;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(H::Item, Vec<R::ReadGuard<'s>>)> {
        let e = self.heap.pop_if(predicate)?;
        let list = e.fetch().iter().map(|&id| self.relation.read(id)).collect();
        Some((e, list))
    }
}

pub struct SimplePrefetcher<R, T> {
    relation: R,
    window: VecDeque<T>,
    heap: BinaryHeap<T>,
}

impl<R, T: Ord> SimplePrefetcher<R, T> {
    pub fn new(relation: R, vec: Vec<T>) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            heap: BinaryHeap::from(vec),
        }
    }
}

impl<R, T> IntoIterator for SimplePrefetcher<R, T> {
    type Item = T;

    type IntoIter = Chain<vec_deque::IntoIter<T>, binary_heap::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.heap)
    }
}

impl<R: RelationRead + RelationPrefetch, T: Fetch + Ord> Prefetcher for SimplePrefetcher<R, T> {
    type R = R;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>)> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.heap.pop()
        {
            for id in e.fetch().iter().copied() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = vec_deque_pop_front_if(&mut self.window, predicate)?;
        let list = e.fetch().iter().map(|&id| self.relation.read(id)).collect();
        Some((e, list))
    }
}

pub struct StreamPrefetcherHeap<T>(BinaryHeap<T>);

impl<T: Ord> Iterator for StreamPrefetcherHeap<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

pub struct StreamPrefetcher<'r, R, T>
where
    R: RelationReadStream + 'r,
    T: Fetch + Ord,
{
    stream: R::ReadStream<'r, StreamPrefetcherHeap<T>>,
}

impl<'r, R: RelationReadStream, T: Fetch + Ord> StreamPrefetcher<'r, R, T> {
    pub fn new(relation: &'r R, vec: Vec<T>) -> Self {
        let stream = relation.read_stream(StreamPrefetcherHeap(BinaryHeap::from(vec)));
        Self { stream }
    }
}

impl<'r, R: RelationReadStream, T: Fetch + Ord> IntoIterator for StreamPrefetcher<'r, R, T> {
    type Item = T;

    type IntoIter = <<R as RelationReadStream>::ReadStream<
        'r,
        StreamPrefetcherHeap<T>,
    > as ReadStream<T>>::Inner;

    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<'r, R: RelationReadStream, T: Fetch + Ord> Prefetcher for StreamPrefetcher<'r, R, T> {
    type R = R;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>)> {
        self.stream.next_if(predicate)
    }
}

// Emulate unstable library feature `vec_deque_pop_if`.
// See https://github.com/rust-lang/rust/issues/135889.

fn vec_deque_pop_front_if<T>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&T) -> bool,
) -> Option<T> {
    let first = this.front()?;
    if predicate(first) {
        this.pop_front()
    } else {
        None
    }
}
