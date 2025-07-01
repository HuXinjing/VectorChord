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
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::visited::Visited;
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Bump, Page, RelationRead};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    ef_search: u32,
    beam_search: u32,
    bump: &'b impl Bump,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) -> Box<dyn Iterator<Item = (Distance, NonZero<u64>)> + 'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let _ = bump;
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let start = meta_tuple.start();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let ef = ef_search;
    let beam = beam_search;
    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return Box::new(std::iter::empty());
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return Box::new(std::iter::empty());
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_s = bump.alloc_slice(vertex_tuple.pointers());
        let score_s = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
        candidates.push((Reverse(score_s), AlwaysEqual(pointers_s)));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some(((_, AlwaysEqual(pointers_u)), guards)) = candidates.pop() {
            let Ok((dis_u, payload_u, outs_u)) =
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
                candidates.push((Reverse(score_v), AlwaysEqual(pointers_v)));
            }
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
        }
        None
    });
    let mut results = Results::new(ef as _);
    let search = std::iter::from_fn(move || {
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
        results.pop_min()
    });
    Box::new(search.filter_map(|(dis_u, AlwaysEqual(payload_u))| Some((dis_u, payload_u?))))
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

// Emulate unstable library feature `strict_overflow_ops`.
// See https://github.com/rust-lang/rust/issues/118260.

#[inline]
pub const fn strict_sub(lhs: usize, rhs: usize) -> usize {
    let (a, b) = lhs.overflowing_sub(rhs);
    if b { panic!() } else { a }
}
