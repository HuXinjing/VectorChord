use crate::operator::Accessor1;
use crate::tuples::*;
use crate::{Page, PageGuard, RelationRead, RelationWrite};
use rabitq::binary::BinaryCode;
use std::marker::PhantomData;
use std::num::NonZero;

pub struct TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    head: R::WriteGuard<'a>,
    first: u32,
    index: &'a R,
    tracking_freespace: bool,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        let mut head = index.extend(tracking_freespace);
        head.get_opaque_mut().skip = head.id();
        let first = head.id();
        Self {
            head,
            first,
            index,
            tracking_freespace,
            _phantom: PhantomData,
        }
    }
    pub fn first(&self) -> u32 {
        self.first
    }
    pub fn freespace(&self) -> u16 {
        self.head.freespace()
    }
    pub fn tape_move(&mut self) {
        if self.head.len() == 0 {
            panic!("implementation: a clear page cannot accommodate a single tuple");
        }
        let next = self.index.extend(self.tracking_freespace);
        self.head.get_opaque_mut().next = next.id();
        self.head = next;
    }
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    T: Tuple,
{
    pub fn push(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            let next = self.index.extend(self.tracking_freespace);
            self.head.get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.alloc(&bytes) {
                (self.head.id(), i)
            } else {
                panic!("implementation: a free page cannot accommodate a single tuple")
            }
        }
    }
    pub fn tape_put(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            panic!("implementation: a free page cannot accommodate a single tuple")
        }
    }
}

pub fn read_h1_tape<A, T>(
    index: impl RelationRead,
    first: u32,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, u32, &'a [u32]),
    mut step: impl FnMut(u32),
) where
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [T; 32],
        >,
{
    assert!(first != u32::MAX);
    let mut current = first;
    let mut x = None;
    while current != u32::MAX {
        step(current);
        let guard = index.read(current);
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = H1Tuple::deserialize_ref(bytes);
            match tuple {
                H1TupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish(tuple.metadata());
                    let prefetch: [_; 32] = fix_0(tuple.prefetch());
                    for (j, value) in values.into_iter().enumerate() {
                        if j < tuple.len() as usize {
                            callback(value, tuple.head()[j], tuple.first()[j], fix_1(prefetch[j]));
                        }
                    }
                }
                H1TupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
        current = guard.get_opaque().next;
    }
}

pub fn read_frozen_tape<A, T>(
    index: impl RelationRead,
    first: u32,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32]),
    mut step: impl FnMut(u32),
) where
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [T; 32],
        >,
{
    assert!(first != u32::MAX);
    let mut current = first;
    let mut x = None;
    while current != u32::MAX {
        step(current);
        let guard = index.read(current);
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = FrozenTuple::deserialize_ref(bytes);
            match tuple {
                FrozenTupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish(tuple.metadata());
                    let prefetch: [_; 32] = fix_0(tuple.prefetch());
                    for (j, value) in values.into_iter().enumerate() {
                        if let Some(payload) = tuple.payload()[j] {
                            callback(value, tuple.mean()[j], payload, fix_1(prefetch[j]));
                        }
                    }
                }
                FrozenTupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
        current = guard.get_opaque().next;
    }
}

pub fn read_appendable_tape<T>(
    index: impl RelationRead,
    first: u32,
    mut access: impl for<'a> FnMut(BinaryCode<'a>) -> T,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32]),
    mut step: impl FnMut(u32),
) {
    assert!(first != u32::MAX);
    let mut current = first;
    while current != u32::MAX {
        step(current);
        let guard = index.read(current);
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = AppendableTuple::deserialize_ref(bytes);
            if let Some(payload) = tuple.payload() {
                let value = access(tuple.code());
                callback(value, tuple.head(), payload, tuple.prefetch());
            }
        }
        current = guard.get_opaque().next;
    }
}

#[allow(clippy::collapsible_else_if)]
pub fn append(
    index: impl RelationRead + RelationWrite,
    first: u32,
    bytes: &[u8],
    tracking_freespace: bool,
) -> (u32, u16) {
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, tracking_freespace);
            if write.get_opaque().next == u32::MAX {
                if let Some(i) = write.alloc(bytes) {
                    return (current, i);
                }
                let mut extend = index.extend(tracking_freespace);
                write.get_opaque_mut().next = extend.id();
                drop(write);
                let fresh = extend.id();
                if let Some(i) = extend.alloc(bytes) {
                    drop(extend);
                    let mut past = index.write(first, tracking_freespace);
                    past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                    return (fresh, i);
                } else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
            }
            if current == first && write.get_opaque().skip != first {
                current = write.get_opaque().skip;
            } else {
                current = write.get_opaque().next;
            }
        } else {
            if current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}

fn fix_0<T>(x: &[[T; 32]]) -> [&[T]; 32] {
    let step = x.len();
    let flat = x.as_flattened();
    std::array::from_fn(|i| &flat[i * step..][..step])
}

fn fix_1(x: &[u32]) -> &[u32] {
    if let Some(i) = x.iter().position(|&x| x == u32::MAX) {
        &x[..i]
    } else {
        x
    }
}
