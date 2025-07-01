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

use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{CloneAccessor, Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::visited::Visited;
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Bump, Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::iter::{repeat, zip};
use std::num::{NonZero, Wrapping};
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    bump: &'b impl Bump,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let start = meta_tuple.start();
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha();
    let ef = meta_tuple.ef_construction();
    let beam = meta_tuple.beam_construction();
    drop(meta_guard);
    let (vector_pointers, vertex_pointer) = insert_tuples::<R, O>(index, vector, payload, m);
    let start = if start.into_inner().is_none() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.into_inner().is_none() {
            *dst = OptionPointer::some(vertex_pointer);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return;
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_s = bump.alloc_slice(vertex_tuple.pointers());
        let score_s = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
        candidates.push((Reverse(score_s), AlwaysEqual((pointers_s, s))));
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some(((_, AlwaysEqual((pointers_u, u))), guards)) = candidates.pop() {
            let Ok((dis_u, _payload_u, outs_u)) =
                rerank::<R, O>(guards, pointers_u, vector, &mut visited)
            else {
                // the link is broken
                continue;
            };
            let mut iterator = prefetch_vertices.prefetch(outs_u);
            while let Some((v, guards)) = iterator.next() {
                visited.insert(v);
                let vertex_guard = {
                    let mut guards = guards;
                    let r = guards.next().expect("internal");
                    assert!(guards.next().is_none(), "internal");
                    drop(guards);
                    r
                };
                let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                    // the link is broken
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let pointers_v = bump.alloc_slice(vertex_tuple.pointers());
                let score_v = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
                candidates.push((Reverse(score_v), AlwaysEqual((pointers_v, v))));
            }
            return Some((Reverse(dis_u), AlwaysEqual((pointers_u, u))));
        }
        None
    });
    let mut results = Results::new(ef as _);
    for element @ (Reverse(dis_c), _) in iter.by_ref() {
        results.push(element);
        if results
            .peek_ef_th()
            .map(|dis_e| dis_e < dis_c)
            .unwrap_or_default()
        {
            break;
        }
    }
    let trace = results
        .into_inner()
        .0
        .into_iter()
        .map(|(dis_v, x)| (Reverse(dis_v), x))
        .collect::<Vec<_>>();
    let outs = crate::prune::robust_prune(
        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
        (bump.alloc_slice(&vector_pointers), vertex_pointer),
        trace.into_iter().flat_map(|item| {
            use algo::accessor::Accessor1;
            let (Reverse(dis_u), AlwaysEqual((pointers_u, u))) = item;
            let m = strict_sub(pointers_u.len(), 1);
            let mut accessor = CloneAccessor::<O::Vector>::default();
            for i in 0..m {
                let vector_guard = index.read(pointers_u[i].into_inner().0);
                let Some(vector_bytes) = vector_guard.get(pointers_u[i].into_inner().1) else {
                    // the link is broken
                    return None;
                };
                let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                let VectorTupleReader::_1(segment) = vector_tuple else {
                    // the link is broken
                    return None;
                };
                if segment.index() as usize != i {
                    // the link is broken
                    return None;
                }
                accessor.push(segment.elements());
            }
            let vector_u;
            {
                let vector_guard = index.read(pointers_u[m].into_inner().0);
                let Some(vector_bytes) = vector_guard.get(pointers_u[m].into_inner().1) else {
                    // the link is broken
                    return None;
                };
                let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                let VectorTupleReader::_0(segment) = vector_tuple else {
                    // the link is broken
                    return None;
                };
                accessor.push(segment.elements());
                vector_u = accessor.finish(*segment.metadata());
            }
            Some(((Reverse(dis_u), AlwaysEqual((pointers_u, u))), vector_u))
        }),
        m,
        alpha,
        |(_, u)| *u,
    );
    {
        let mut vector_guard = index.write(vector_pointers.last().unwrap().into_inner().0, false);
        let Some(vector_bytes) =
            vector_guard.get_mut(vector_pointers.last().unwrap().into_inner().1)
        else {
            // the link is broken
            return;
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_mut(vector_bytes);
        let VectorTupleWriter::_0(mut vector_tuple) = vector_tuple else {
            // the link is broken
            return;
        };
        let iterator = outs
            .iter()
            .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| OptionNeighbour::some(v, dis_v))
            .chain(repeat(OptionNeighbour::NONE));
        for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), iterator) {
            *hole = fill;
        }
    }
    for (Reverse(dis_t), AlwaysEqual((pointers_u, u))) in outs {
        while add_link::<R, O>(
            index,
            (pointers_u, u),
            (vertex_pointer, dis_t),
            m,
            alpha,
            bump,
        ) == Ok(false)
        {}
    }
}

fn rerank<'r, R: RelationRead, O: Operator>(
    mut guards: impl Iterator<Item = R::ReadGuard<'r>>,
    pointers_u: &mut [Pointer],
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    visited: &mut Visited,
) -> Result<(Distance, Option<NonZero<u64>>, VecDeque<(u32, u16)>), ()> {
    use algo::accessor::{Accessor1, LAccess};
    let m = strict_sub(pointers_u.len(), 1);
    let mut accessor = LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default());
    for i in 0..m {
        let vector_guard = guards.next().expect("internal");
        let Some(vector_bytes) = vector_guard.get(pointers_u[i].into_inner().1) else {
            // the link is broken
            return Err(());
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        let VectorTupleReader::_1(segment) = vector_tuple else {
            // the link is broken
            return Err(());
        };
        if segment.index() as usize != i {
            // the link is broken
            return Err(());
        }
        accessor.push(segment.elements());
    }
    let dis_u;
    let payload_u;
    let neighbours_u;
    {
        let vector_guard = guards.next().expect("internal");
        let Some(vector_bytes) = vector_guard.get(pointers_u[m].into_inner().1) else {
            // the link is broken
            return Err(());
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        let VectorTupleReader::_0(segment) = vector_tuple else {
            // the link is broken
            return Err(());
        };
        accessor.push(segment.elements());
        dis_u = accessor.finish(*segment.metadata());
        payload_u = segment.payload();
        neighbours_u = segment
            .neighbours()
            .iter()
            .flat_map(|neighbour| neighbour.into_inner())
            .map(|(v, _)| v)
            .filter(|&v| !visited.contains(v))
            .collect::<VecDeque<_>>();
    }
    Ok((dis_u, payload_u, neighbours_u))
}

fn insert_tuples<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    m: u32,
) -> (Vec<Pointer>, (u32, u16))
where
    R::Page: Page<Opaque = Opaque>,
{
    let vector_bytes = {
        let (left, right) = O::Vector::split(vector, m as _);
        let left = left.into_iter().enumerate().map(|(index, elements)| {
            VectorTuple::serialize(&VectorTuple::<O::Vector>::_1 {
                payload: Some(payload),
                elements: elements.to_vec(),
                index: index as u32,
            })
        });
        let right = VectorTuple::serialize(&VectorTuple::<O::Vector>::_0 {
            payload: Some(payload),
            elements: right.0.to_vec(),
            metadata: right.1,
            neighbours: vec![OptionNeighbour::NONE; m as usize],
            version: Wrapping(rand::random()),
        });
        left.chain(std::iter::once(right)).collect::<Vec<_>>()
    };
    let mut vertex_bytes = {
        let code = O::Vector::code(vector);
        VertexTuple::serialize(&VertexTuple {
            metadata: code.0.into_array(),
            payload: Some(payload),
            elements: rabitq::original::binary::pack_code(&code.1),
            pointers: vec![Pointer::new((u32::MAX, 0)); vector_bytes.len()], // a sentinel value
        })
    };
    append_vertex_tuple(index, 1, vertex_bytes.len(), |mut vertex_guard| {
        let vector_pointers = vector_bytes
            .iter()
            .map(|vector_bytes| {
                append_vector_tuple(
                    index,
                    vertex_guard.get_opaque().link,
                    vector_bytes.len(),
                    |mut vector_guard| {
                        let i = vector_guard.alloc(vector_bytes).expect(
                            "implementation: a free page cannot accommodate a single tuple",
                        );
                        Pointer::new((vector_guard.id(), i))
                    },
                )
            })
            .collect::<Vec<_>>();
        {
            let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
            vertex_tuple.pointers().copy_from_slice(&vector_pointers);
        }
        let vertex_pointer = {
            let i = vertex_guard
                .alloc(&vertex_bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            (vertex_guard.id(), i)
        };
        (vector_pointers, vertex_pointer)
    })
}

#[allow(clippy::collapsible_else_if)]
fn append_vertex_tuple<'r, R: RelationRead + RelationWrite, T>(
    index: &'r R,
    first: u32,
    size: usize,
    f: impl FnOnce(R::WriteGuard<'r>) -> T,
) -> T
where
    R::Page: Page<Opaque = Opaque>,
{
    if let Some(guard) = index.search(size) {
        return f(guard);
    }
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, true);
            if write.freespace() as usize >= size {
                return f(write);
            }
            if write.get_opaque().next == u32::MAX {
                let link = index
                    .extend(
                        Opaque {
                            next: u32::MAX,
                            skip: u32::MAX,
                            link: u32::MAX,
                            _padding_0: Default::default(),
                        },
                        false,
                    )
                    .id();
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        skip: u32::MAX,
                        link,
                        _padding_0: Default::default(),
                    },
                    true,
                );
                { write }.get_opaque_mut().next = extend.id();
                if extend.freespace() as usize >= size {
                    let fresh = extend.id();
                    let result = f(extend);
                    let mut past = index.write(first, true);
                    past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                    return result;
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

fn append_vector_tuple<'r, R: RelationRead + RelationWrite, T>(
    index: &'r R,
    first: u32,
    size: usize,
    f: impl FnOnce(R::WriteGuard<'r>) -> T,
) -> T
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, false);
            if write.freespace() as usize >= size {
                return f(write);
            }
            if write.get_opaque().next == u32::MAX {
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        skip: u32::MAX,
                        link: u32::MAX,
                        _padding_0: Default::default(),
                    },
                    false,
                );
                { write }.get_opaque_mut().next = extend.id();
                if extend.freespace() as usize >= size {
                    return f(extend);
                } else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
            }
            current = write.get_opaque().next
        } else {
            current = read.get_opaque().next
        }
    }
}

fn add_link<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    (pointers_u, u): (&mut [Pointer], (u32, u16)),
    new: ((u32, u16), Distance),
    m: u32,
    alpha: f32,
    bump: &impl Bump,
) -> Result<bool, ()> {
    let vector_guard = index.read(pointers_u.last().unwrap().into_inner().0);
    let Some(vector_bytes) = vector_guard.get(pointers_u.last().unwrap().into_inner().1) else {
        // the link is broken
        return Err(());
    };
    let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
    let VectorTupleReader::_0(vector_tuple) = vector_tuple else {
        // the link is broken
        return Err(());
    };
    let check = vector_tuple
        .neighbours()
        .iter()
        .flat_map(|neighbour| neighbour.into_inner())
        .map(|(v, dis_v)| (Reverse(dis_v), AlwaysEqual(v)))
        .collect::<Vec<_>>();
    let trace = check
        .iter()
        .copied()
        .chain(std::iter::once((Reverse(new.1), AlwaysEqual(new.0))))
        .collect::<Vec<_>>();
    let version = vector_tuple.version();
    drop(vector_guard);
    let outs = crate::prune::robust_prune(
        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
        (bump.alloc_slice(pointers_u), u),
        trace
            .into_iter()
            .flat_map(|item| {
                let (Reverse(dis_u), AlwaysEqual(u)) = item;
                let vertex_guard = index.read(u.0);
                let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                    // the link is broken
                    return None;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let pointers_u = bump.alloc_slice(vertex_tuple.pointers());
                Some((Reverse(dis_u), AlwaysEqual((pointers_u, u))))
            })
            .flat_map(|item| {
                use algo::accessor::Accessor1;
                let (Reverse(dis_u), AlwaysEqual((pointers_u, u))) = item;
                let m = strict_sub(pointers_u.len(), 1);
                let mut accessor = CloneAccessor::<O::Vector>::default();
                for i in 0..m {
                    let vector_guard = index.read(pointers_u[i].into_inner().0);
                    let Some(vector_bytes) = vector_guard.get(pointers_u[i].into_inner().1) else {
                        // the link is broken
                        return None;
                    };
                    let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                    let VectorTupleReader::_1(segment) = vector_tuple else {
                        // the link is broken
                        return None;
                    };
                    if segment.index() as usize != i {
                        // the link is broken
                        return None;
                    }
                    accessor.push(segment.elements());
                }
                let vector_u;
                {
                    let vector_guard = index.read(pointers_u[m].into_inner().0);
                    let Some(vector_bytes) = vector_guard.get(pointers_u[m].into_inner().1) else {
                        // the link is broken
                        return None;
                    };
                    let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                    let VectorTupleReader::_0(segment) = vector_tuple else {
                        // the link is broken
                        return None;
                    };
                    accessor.push(segment.elements());
                    vector_u = accessor.finish(*segment.metadata());
                }
                Some(((Reverse(dis_u), AlwaysEqual((pointers_u, u))), vector_u))
            }),
        m,
        alpha,
        |(_, u)| *u,
    );
    // fast path
    if outs
        .iter()
        .map(|&(x, AlwaysEqual((_, v)))| (x, AlwaysEqual(v)))
        .eq(check)
    {
        return Ok(true);
    }
    let mut vector_guard = index.write(pointers_u.last().unwrap().into_inner().0, false);
    let Some(vector_bytes) = vector_guard.get_mut(pointers_u.last().unwrap().into_inner().1) else {
        // the link is broken
        return Err(());
    };
    let vector_tuple = VectorTuple::<O::Vector>::deserialize_mut(vector_bytes);
    let VectorTupleWriter::_0(mut vector_tuple) = vector_tuple else {
        // the link is broken
        return Err(());
    };
    if *vector_tuple.version() != version {
        return Ok(false);
    } else {
        *vector_tuple.version() += 1;
    }
    let filling = outs
        .iter()
        .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| OptionNeighbour::some(v, dis_v))
        .chain(repeat(OptionNeighbour::NONE));
    for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), filling) {
        *hole = fill;
    }
    Ok(true)
}

// Emulate unstable library feature `strict_overflow_ops`.
// See https://github.com/rust-lang/rust/issues/118260.

#[inline]
pub const fn strict_sub(lhs: usize, rhs: usize) -> usize {
    let (a, b) = lhs.overflowing_sub(rhs);
    if b { panic!() } else { a }
}
