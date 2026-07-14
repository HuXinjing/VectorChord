// Copyright (c) 2026 HuXinjing

use crate::tuples::{MetaTuple, WithWriter};
use index::relation::{Page, RelationWrite};

/// Store the number of live vector nodes observed by the latest complete
/// build or vacuum pass.
///
/// This is deliberately refreshed in bulk instead of on every insert. A
/// per-insert update would serialize all writers on the metapage, which is a
/// poor tradeoff for a planner statistic. Like PostgreSQL's relation
/// statistics, the value may be stale between maintenance passes.
pub fn set_indexed_vectors<R: RelationWrite>(index: &R, indexed_vectors: u64) {
    let mut meta_guard = index.write(0, false);
    let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
    let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
    meta_tuple.set_indexed_vectors(indexed_vectors);
}
