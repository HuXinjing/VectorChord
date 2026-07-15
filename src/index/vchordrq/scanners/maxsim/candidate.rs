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

use super::profile;
use crate::index::fetcher::pointer_to_kv;
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::num::NonZero;

pub(super) type HeapKey = [u16; 3];
pub(super) type TokenCandidate = (Distance, NonZero<u64>);
pub(super) type TokenSearchResult = (Vec<TokenCandidate>, Vec<TokenCandidate>, Distance);

#[derive(Clone, Copy, Debug)]
pub(super) struct PageCandidate {
    pub approximate_distance: Distance,
    pub heap_key: HeapKey,
}

pub(super) trait PageCandidateGenerator {
    type Candidates: Iterator<Item = PageCandidate>;

    fn generate(
        &mut self,
        query_count: usize,
        token_searches: &mut dyn Iterator<Item = TokenSearchResult>,
        candidate_limit: usize,
    ) -> Self::Candidates;
}

#[derive(Default)]
pub(super) struct DensePageCandidateGenerator;

impl PageCandidateGenerator for DensePageCandidateGenerator {
    type Candidates = PageCandidates;

    fn generate(
        &mut self,
        query_count: usize,
        token_searches: &mut dyn Iterator<Item = TokenSearchResult>,
        candidate_limit: usize,
    ) -> Self::Candidates {
        let mut page_lookup = HashMap::new();
        let mut page_keys = Vec::new();
        let mut best_by_page_query = Vec::new();
        let mut estimations = Vec::with_capacity(query_count);
        let mut hit_updates = 0u64;
        let mut page_token_updates = 0u64;
        for (query_id, (accu_set, rough_set, estimation_by_threshold)) in token_searches.enumerate()
        {
            debug_assert!(query_id < query_count);
            let hit_collect_timer = profile::ProfileTimer::start();
            hit_updates += (accu_set.len() + rough_set.len()) as u64;
            let is_empty = accu_set.is_empty() && rough_set.is_empty();
            let mut estimation_by_scope = Distance::NEG_INFINITY;
            for (distance, payload) in accu_set {
                estimation_by_scope = std::cmp::max(estimation_by_scope, distance);
                let (key, _) = pointer_to_kv(payload);
                page_token_updates += u64::from(record_page_token_distance(
                    &mut page_lookup,
                    &mut page_keys,
                    &mut best_by_page_query,
                    query_count,
                    query_id,
                    key,
                    distance,
                ));
            }
            for (distance, payload) in rough_set {
                let (key, _) = pointer_to_kv(payload);
                page_token_updates += u64::from(record_page_token_distance(
                    &mut page_lookup,
                    &mut page_keys,
                    &mut best_by_page_query,
                    query_count,
                    query_id,
                    key,
                    distance,
                ));
            }
            estimations.push(if !is_empty {
                std::cmp::max(estimation_by_scope, estimation_by_threshold)
            } else {
                Distance::ZERO
            });
            let hit_collect_elapsed = hit_collect_timer.elapsed();
            profile::update(|profile| {
                profile.hit_collect_us += profile::duration_us(hit_collect_elapsed);
            });
        }
        debug_assert_eq!(estimations.len(), query_count);
        profile::update(|profile| {
            profile.hit_updates += hit_updates;
            profile.page_token_updates += page_token_updates;
        });
        let page_aggregate_timer = profile::ProfileTimer::start();
        let mut page_order = (0..page_keys.len()).collect::<Vec<_>>();
        page_order.sort_unstable_by_key(|&page_index| page_keys[page_index]);
        let inner = page_order
            .into_iter()
            .map(|page_index| {
                let key = page_keys[page_index];
                let start = page_index * query_count;
                let values = &best_by_page_query[start..start + query_count];
                let mut maxsim = 0.0f32;
                for (query_id, distance) in values.iter().copied().enumerate() {
                    let distance = distance.unwrap_or(estimations[query_id]);
                    maxsim += distance.to_f32();
                }
                (Reverse(Distance::from_f32(maxsim)), AlwaysEqual(key))
            })
            .collect::<BinaryHeap<_>>();
        let page_aggregate_elapsed = page_aggregate_timer.elapsed();
        profile::update(|profile| {
            profile.page_aggregate_us += profile::duration_us(page_aggregate_elapsed);
            profile.aggregated_pages += page_keys.len() as u64;
        });
        PageCandidates {
            inner,
            remaining: candidate_limit,
        }
    }
}

fn record_page_token_distance(
    page_lookup: &mut HashMap<HeapKey, usize>,
    page_keys: &mut Vec<HeapKey>,
    best_by_page_query: &mut Vec<Option<Distance>>,
    query_count: usize,
    query_id: usize,
    key: HeapKey,
    distance: Distance,
) -> bool {
    let page_index = match page_lookup.entry(key) {
        Entry::Occupied(entry) => *entry.get(),
        Entry::Vacant(entry) => {
            let page_index = page_keys.len();
            entry.insert(page_index);
            page_keys.push(key);
            best_by_page_query.resize(best_by_page_query.len() + query_count, None);
            page_index
        }
    };
    let slot = &mut best_by_page_query[page_index * query_count + query_id];
    let first_for_page_token = slot.is_none();
    let best = slot.get_or_insert(Distance::INFINITY);
    *best = std::cmp::min(*best, distance);
    first_for_page_token
}

pub(super) struct PageCandidates {
    inner: BinaryHeap<(Reverse<Distance>, AlwaysEqual<HeapKey>)>,
    remaining: usize,
}

impl Iterator for PageCandidates {
    type Item = PageCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let (Reverse(approximate_distance), AlwaysEqual(heap_key)) = self.inner.pop()?;
        self.remaining -= 1;
        Some(PageCandidate {
            approximate_distance,
            heap_key,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len().min(self.remaining);
        (exact, Some(exact))
    }
}

impl ExactSizeIterator for PageCandidates {}
impl std::iter::FusedIterator for PageCandidates {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetcher::kv_to_pointer;

    fn distance(value: f32) -> Distance {
        Distance::from_f32(value)
    }

    #[test]
    fn aggregates_tokens_by_page_and_preserves_distance_order() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let mut token_searches = vec![
            (
                vec![
                    (distance(-1.0), kv_to_pointer((page_1, 0))),
                    (distance(-0.5), kv_to_pointer((page_2, 0))),
                ],
                vec![(distance(-0.4), kv_to_pointer((page_2, 1)))],
                distance(-0.75),
            ),
            (
                vec![(distance(-2.0), kv_to_pointer((page_1, 1)))],
                vec![],
                distance(-1.0),
            ),
        ]
        .into_iter();
        let candidates = DensePageCandidateGenerator
            .generate(2, &mut token_searches, usize::MAX)
            .collect::<Vec<_>>();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].heap_key, page_1);
        assert_eq!(candidates[0].approximate_distance.to_f32(), -3.0);
        assert_eq!(candidates[1].heap_key, page_2);
        assert_eq!(candidates[1].approximate_distance.to_f32(), -1.5);
    }

    #[test]
    fn candidate_limit_is_applied_after_global_ordering() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let mut token_searches = vec![(
            vec![
                (distance(-1.0), kv_to_pointer((page_1, 0))),
                (distance(-2.0), kv_to_pointer((page_2, 0))),
            ],
            vec![],
            Distance::ZERO,
        )]
        .into_iter();
        let candidates = DensePageCandidateGenerator
            .generate(1, &mut token_searches, 1)
            .collect::<Vec<_>>();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].heap_key, page_2);
    }
}
