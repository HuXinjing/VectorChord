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
// Copyright (c) 2025-2026 TensorChord Inc.

use crate::index::fetcher::{Fetcher, FilterableTuple, Tuple};
use crate::index::vchordrq::opclass::Opfamily;
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vchordrq::types::OwnedVector;

pub(super) type HeapKey = [u16; 3];

#[derive(Clone, Copy, Debug)]
pub(super) struct Candidate {
    pub distance: Distance,
    pub heap_key: HeapKey,
}

pub(super) struct CandidateTensor {
    pub candidate: Candidate,
    pub vectors: Vec<OwnedVector>,
}

/// Supplies the full tensor for a candidate selected by the index.
///
/// Keeping this boundary separate from scoring lets future backends consume a
/// different source without changing candidate generation. The reference
/// source below reads the indexed column from the PostgreSQL heap.
pub(super) trait CandidateTensorSource {
    fn fetch(&mut self, candidate: Candidate) -> Result<Option<CandidateTensor>, RerankError>;
}

pub(super) struct HeapTensorSource<'a, F> {
    fetcher: &'a mut F,
    opfamily: Opfamily,
}

impl<'a, F> HeapTensorSource<'a, F> {
    pub fn new(fetcher: &'a mut F, opfamily: Opfamily) -> Self {
        Self { fetcher, opfamily }
    }
}

impl<F: Fetcher> CandidateTensorSource for HeapTensorSource<'_, F> {
    fn fetch(&mut self, candidate: Candidate) -> Result<Option<CandidateTensor>, RerankError> {
        let Some(mut tuple) = self.fetcher.fetch(candidate.heap_key) else {
            return Ok(None);
        };
        if !tuple.filter() {
            return Ok(None);
        }
        let (values, is_nulls) = tuple.build();
        if is_nulls[0] {
            return Err(RerankError::TensorMismatch);
        }
        let vectors =
            unsafe { self.opfamily.input_vectors(values[0]) }.ok_or(RerankError::TensorMismatch)?;
        Ok(Some(CandidateTensor { candidate, vectors }))
    }
}

/// Exact MaxSim scorer independent of candidate generation and tensor storage.
pub(super) trait ExactMaxsimBackend {
    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = Candidate>,
        source: &mut S,
    ) -> Result<Vec<Candidate>, RerankError>;
}

#[derive(Debug)]
pub(super) enum RerankError {
    TensorMismatch,
}

impl std::fmt::Display for RerankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorMismatch => write!(f, "MaxSim tensor kind or dimension is not matched"),
        }
    }
}

#[derive(Default)]
pub(super) struct CpuExactMaxsimBackend;

impl ExactMaxsimBackend for CpuExactMaxsimBackend {
    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = Candidate>,
        source: &mut S,
    ) -> Result<Vec<Candidate>, RerankError> {
        let mut results = BinaryHeap::new();
        for candidate in candidates {
            let Some(tensor) = source.fetch(candidate)? else {
                continue;
            };
            let distance =
                exact_maxsim_distance(query, &tensor.vectors).ok_or(RerankError::TensorMismatch)?;
            results.push((Reverse(distance), AlwaysEqual(tensor.candidate.heap_key)));
        }
        Ok(results
            .into_iter_sorted_polyfill()
            .map(|(Reverse(distance), AlwaysEqual(heap_key))| Candidate { distance, heap_key })
            .collect())
    }
}

fn exact_maxsim_distance(query: &[OwnedVector], document: &[OwnedVector]) -> Option<Distance> {
    if query.is_empty() || document.is_empty() {
        return None;
    }
    let mut maxsim = 0.0f32;
    for query_vector in query {
        let mut best = Distance::INFINITY;
        for document_vector in document {
            best = std::cmp::min(best, document_vector.operator_dot(query_vector)?);
        }
        maxsim += best.to_f32();
    }
    Some(Distance::from_f32(maxsim))
}

// Emulate unstable library feature `binary_heap_into_iter_sorted`.
trait IntoIterSortedPolyfill<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T>;
}

impl<T> IntoIterSortedPolyfill<T> for BinaryHeap<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T> {
        IntoIterSorted(self)
    }
}

struct IntoIterSorted<T>(BinaryHeap<T>);

impl<T: Ord> Iterator for IntoIterSorted<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vector::vect::VectOwned;

    struct MockSource(BTreeMap<HeapKey, Vec<OwnedVector>>);

    impl CandidateTensorSource for MockSource {
        fn fetch(&mut self, candidate: Candidate) -> Result<Option<CandidateTensor>, RerankError> {
            Ok(Some(CandidateTensor {
                candidate,
                vectors: self
                    .0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            }))
        }
    }

    fn vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf32(VectOwned::new(values.to_vec()))
    }

    #[test]
    fn cpu_backend_orders_candidates_by_exact_maxsim() {
        let first = [0, 0, 1];
        let second = [0, 0, 2];
        let query = vec![vector(&[1.0, 0.0]), vector(&[0.0, 1.0])];
        let mut candidates = vec![
            Candidate {
                distance: Distance::from_f32(-2.0),
                heap_key: second,
            },
            Candidate {
                distance: Distance::from_f32(-1.0),
                heap_key: first,
            },
        ]
        .into_iter();
        let mut source = MockSource(BTreeMap::from([
            (first, vec![vector(&[1.0, 0.0]), vector(&[0.0, 1.0])]),
            (second, vec![vector(&[0.5, 0.5])]),
        ]));

        let results = CpuExactMaxsimBackend
            .rerank(&query, &mut candidates, &mut source)
            .unwrap();

        assert_eq!(
            results.iter().map(|x| x.heap_key).collect::<Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(results[0].distance.to_f32(), -2.0);
        assert_eq!(results[1].distance.to_f32(), -1.0);
    }
}
