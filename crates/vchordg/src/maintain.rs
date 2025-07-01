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
use crate::operator::{CloneAccessor, Operator};
use crate::tuples::{
    MetaTuple, OptionNeighbour, Pointer, VectorTuple, VectorTupleReader, VectorTupleWriter,
    VertexTuple, WithReader, WithWriter,
};
use algo::{Bump, Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::iter::{repeat, zip};
use std::num::Wrapping;
use vector::VectorOwned;

pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
    bump: &impl Bump,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha();
    let start = meta_tuple.start().into_inner();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    // do it's best to remove broken edges
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let vertex_guard = index.read(current);
            let mut members = Vec::new();
            for i in 1..=vertex_guard.len() {
                if let Some(vertex_bytes) = vertex_guard.get(i) {
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    if vertex_tuple.payload().is_some() {
                        let pointers = bump.alloc_slice(vertex_tuple.pointers());
                        members.push((pointers, (vertex_guard.id(), i)));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            for member in members {
                while fix_link::<R, O>(index, (member.0, member.1), m, alpha, bump) == Ok(false) {}
            }
            current = next;
        }
    }
    // remove vertices and vectors
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let mut vertex_guard = index.write(current, true);
            for i in 1..=vertex_guard.len() {
                if let Some(bytes) = vertex_guard.get(i) {
                    let tuple = VertexTuple::deserialize_ref(bytes);
                    let p = tuple.payload();
                    if p.is_none() && Some((current, i)) != start {
                        vertex_guard.free(i);
                    }
                };
            }
            current = vertex_guard.get_opaque().next;
            {
                let mut current = { vertex_guard }.get_opaque().link;
                while current != u32::MAX {
                    check();
                    let mut vector_guard = index.write(current, true);
                    for i in 1..=vector_guard.len() {
                        if let Some(bytes) = vector_guard.get(i) {
                            use crate::tuples::VectorTupleReader;
                            let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                            match tuple {
                                VectorTupleReader::_0(tuple) => {
                                    let p = tuple.payload();
                                    if p.is_none() && Some((current, i)) != start {
                                        vector_guard.free(i);
                                    }
                                }
                                VectorTupleReader::_1(tuple) => {
                                    let p = tuple.payload();
                                    if p.is_none() && Some((current, i)) != start {
                                        vector_guard.free(i);
                                    }
                                }
                            }
                        };
                    }
                    current = vector_guard.get_opaque().next;
                }
            }
        }
    }
}

fn read_tuple<R: RelationRead, O: Operator>(
    index: &R,
    pointers_u: &mut [Pointer],
) -> Result<(O::Vector, VecDeque<((u32, u16), Distance)>, Wrapping<u32>), ()> {
    use algo::accessor::Accessor1;
    let m = strict_sub(pointers_u.len(), 1);
    let mut accessor = CloneAccessor::<O::Vector>::default();
    for i in 0..m {
        let vector_guard = index.read(pointers_u[i].into_inner().0);
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
    let vector_u;
    let neighbours_u;
    let version;
    {
        let vector_guard = index.read(pointers_u[m].into_inner().0);
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
        vector_u = accessor.finish(*segment.metadata());
        neighbours_u = segment
            .neighbours()
            .iter()
            .flat_map(|neighbour| neighbour.into_inner())
            .collect::<VecDeque<_>>();
        version = segment.version();
    }
    Ok((vector_u, neighbours_u, version))
}

fn fix_link<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    (pointers_u, u): (&mut [Pointer], (u32, u16)),
    m: u32,
    alpha: f32,
    bump: &impl Bump,
) -> Result<bool, ()> {
    let Ok((vector_u, neighbours_u, version)) = read_tuple::<R, O>(index, pointers_u) else {
        // the link is broken
        return Err(());
    };
    let mut trace = Vec::new();
    let mut extend = Vec::new();
    for &(v, dis_v) in neighbours_u.iter() {
        let vertex_guard = index.read(v.0);
        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
            // the link is broken
            continue;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_v = bump.alloc_slice(vertex_tuple.pointers());
        let payload_v = vertex_tuple.payload();
        drop(vertex_guard);
        let Ok((vector_v, neighbours_v, _)) = read_tuple::<R, O>(index, pointers_v) else {
            // the link is broken
            continue;
        };
        if payload_v.is_some() {
            trace.push((v, dis_v, vector_v));
        } else {
            let outs_v = neighbours_v.iter().map(|(v, _)| *v).collect::<Vec<_>>();
            extend.extend(outs_v);
        }
    }
    // fast path
    if extend.is_empty() {
        return Ok(true);
    }
    for v in extend {
        let vertex_guard = index.read(v.0);
        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
            // the link is broken
            continue;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_v = bump.alloc_slice(vertex_tuple.pointers());
        let payload_v = vertex_tuple.payload();
        drop(vertex_guard);
        let Ok((vector_v, _, _)) = read_tuple::<R, O>(index, pointers_v) else {
            // the link is broken
            continue;
        };
        if payload_v.is_some() {
            let dis_v = O::distance(vector_u.as_borrowed(), vector_v.as_borrowed());
            trace.push((v, dis_v, vector_v));
        }
    }
    let outs = crate::prune::robust_prune(
        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
        (bump.alloc_slice(pointers_u), u),
        trace.into_iter().flat_map(|item| {
            let (u, dis_u, vector_u) = item;
            let vertex_guard = index.read(u.0);
            let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                // the link is broken
                return None;
            };
            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
            let pointer_u = bump.alloc_slice(vertex_tuple.pointers());
            Some(((Reverse(dis_u), AlwaysEqual((pointer_u, u))), vector_u))
        }),
        m,
        alpha,
        |(_, u)| *u,
    );
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
    let iterator = outs
        .iter()
        .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| OptionNeighbour::some(v, dis_v))
        .chain(repeat(OptionNeighbour::NONE));
    for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), iterator) {
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
