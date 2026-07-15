// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use super::candidate::{HeapKey, PageCandidate};
use super::external::{
    CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorStorage,
    TileMaxsimSourceBinding, resolve_tilemaxsim_source, validate_descriptor,
};
use super::gpu::{GpuExternalTileMaxsimBackend, UnixSocketTransport};
use super::profile;
use super::rerank::RerankError;
use crate::datatype::memory_halfvec::HalfvecInput;
use crate::datatype::memory_vector::VectorInput;
use crate::index::gucs::{self, PostgresMaxsimBackend};
use distance::Distance;
use pgrx::datum::{Array, DatumWithOid, FromDatum, IntoDatum};
use pgrx::iter::TableIterator;
use pgrx::{name, pg_extern};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::time::Duration;
use vchordrq::types::OwnedVector;
use vector::VectorBorrowed;

/// Exact TileMaxSim for a caller-supplied candidate set of external tensors.
/// Candidate selection, tenancy, ACLs, graph traversal, and clustering are
/// intentionally outside this function.
#[pg_extern(sql = "")]
fn _vchordrq_tilemaxsim_rerank_vector(
    source_oid: pgrx::pg_sys::Oid,
    query: Array<'_, VectorInput<'_>>,
    candidate_ids: Array<'_, i64>,
    top_k: i32,
) -> TableIterator<'static, (name!(public_id, i64), name!(similarity, f32))> {
    let rows = collect_vector_query(query)
        .and_then(|query| {
            collect_candidate_ids(candidate_ids)
                .and_then(|candidate_ids| execute_rerank(source_oid, query, candidate_ids, top_k))
        })
        .unwrap_or_else(|error| pgrx::error!("{error}"));
    TableIterator::new(rows)
}

/// Half-precision overload of exact caller-scoped TileMaxSim.
#[pg_extern(sql = "")]
fn _vchordrq_tilemaxsim_rerank_halfvec(
    source_oid: pgrx::pg_sys::Oid,
    query: Array<'_, HalfvecInput<'_>>,
    candidate_ids: Array<'_, i64>,
    top_k: i32,
) -> TableIterator<'static, (name!(public_id, i64), name!(similarity, f32))> {
    let rows = collect_halfvec_query(query)
        .and_then(|query| {
            collect_candidate_ids(candidate_ids)
                .and_then(|candidate_ids| execute_rerank(source_oid, query, candidate_ids, top_k))
        })
        .unwrap_or_else(|error| pgrx::error!("{error}"));
    TableIterator::new(rows)
}

fn collect_vector_query(
    query: Array<'_, VectorInput<'_>>,
) -> Result<Vec<OwnedVector>, RerankError> {
    let mut vectors = Vec::with_capacity(query.len());
    for vector in query.iter() {
        let vector = vector.ok_or(RerankError::TensorMismatch)?;
        vectors.push(OwnedVector::Vecf32(vector.as_borrowed().own()));
    }
    validate_query(&vectors)?;
    Ok(vectors)
}

fn collect_halfvec_query(
    query: Array<'_, HalfvecInput<'_>>,
) -> Result<Vec<OwnedVector>, RerankError> {
    let mut vectors = Vec::with_capacity(query.len());
    for vector in query.iter() {
        let vector = vector.ok_or(RerankError::TensorMismatch)?;
        vectors.push(OwnedVector::Vecf16(vector.as_borrowed().own()));
    }
    validate_query(&vectors)?;
    Ok(vectors)
}

fn validate_query(query: &[OwnedVector]) -> Result<(), RerankError> {
    if query.is_empty() {
        return Err(RerankError::Configuration(
            "query tensor must contain at least one token",
        ));
    }
    Ok(())
}

fn collect_candidate_ids(candidate_ids: Array<'_, i64>) -> Result<Vec<i64>, RerankError> {
    let mut result = Vec::with_capacity(candidate_ids.len());
    let mut unique = BTreeSet::new();
    for public_id in candidate_ids.iter() {
        let public_id = public_id.ok_or(RerankError::Configuration(
            "candidate_ids must not contain NULL",
        ))?;
        if !unique.insert(public_id) {
            return Err(RerankError::Configuration(
                "candidate_ids must not contain duplicates",
            ));
        }
        result.push(public_id);
    }
    if result.is_empty() {
        return Err(RerankError::Configuration(
            "candidate_ids must contain at least one ID",
        ));
    }
    Ok(result)
}

fn execute_rerank(
    source_oid: pgrx::pg_sys::Oid,
    query: Vec<OwnedVector>,
    candidate_ids: Vec<i64>,
    top_k: i32,
) -> Result<std::vec::IntoIter<(i64, f32)>, RerankError> {
    validate_top_k(candidate_ids.len(), top_k)?;
    if !matches!(gucs::vchordrq_maxsim_backend(), PostgresMaxsimBackend::Gpu) {
        return Err(RerankError::Configuration(
            "external TileMaxSim rerank requires vchordrq.maxsim_backend = 'gpu'",
        ));
    }
    require_mvcc_snapshot()?;

    let profile_guard = profile::ProfileGuard::start(gucs::vchordrq_maxsim_profile());
    let total_timer = profile::ProfileTimer::start();
    profile::update(|profile| {
        profile.query_tokens = query.len() as u64;
        profile.generated_candidates = candidate_ids.len() as u64;
    });

    let preflight_timer = profile::ProfileTimer::start();
    let binding = resolve_tilemaxsim_source(source_oid)?;
    let source_lock = RelationLock::open(source_oid, pgrx::pg_sys::AccessShareLock as _)?;
    let descriptor_lock = binding
        .descriptor_oid
        .map(|oid| RelationLock::open(oid, pgrx::pg_sys::AccessShareLock as _))
        .transpose()?;
    if source_lock.oid() != binding.source_oid
        || descriptor_lock.as_ref().map(RelationLock::oid) != binding.descriptor_oid
    {
        return Err(RerankError::Registry(
            "registered TileMaxSim tensor source changed during execution".into(),
        ));
    }
    preflight_descriptor_access(&binding)?;
    profile::update(|profile| {
        profile.preflight_us += profile::duration_us(preflight_timer.elapsed());
    });

    let descriptor_timer = profile::ProfileTimer::start();
    let (mut source, public_ids) = load_visible_descriptors(&binding, &candidate_ids)?;
    let mut candidates = public_ids
        .keys()
        .copied()
        .map(|heap_key| PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key,
        })
        .collect::<Vec<_>>()
        .into_iter();
    profile::update(|profile| {
        profile.descriptor_us += profile::duration_us(descriptor_timer.elapsed());
        profile.visible_candidates = public_ids.len() as u64;
        profile.descriptors = public_ids.len() as u64;
    });

    let endpoint = gucs::vchordrq_maxsim_gpu_endpoint()
        .map(|endpoint| endpoint.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut backend = GpuExternalTileMaxsimBackend::new(
        UnixSocketTransport::new(endpoint),
        binding.model_contract_id,
        Duration::from_millis(gucs::vchordrq_maxsim_gpu_timeout_ms() as u64),
        gucs::vchordrq_maxsim_gpu_max_batch_tokens() as usize,
        gucs::vchordrq_maxsim_gpu_max_batch_bytes() as usize,
    );
    if let Some(tenant) = gucs::vchordrq_maxsim_tenant() {
        backend = backend.with_scheduling(tenant, gucs::vchordrq_maxsim_priority());
    }
    let sidecar_timer = profile::ProfileTimer::start();
    let result_limit = top_k as usize;
    let mut best = BinaryHeap::with_capacity(result_limit.saturating_add(1));
    backend.rerank_batches(&query, &mut candidates, &mut source, |batch| {
        for result in batch {
            let public_id = public_ids.get(&result.heap_key).copied().ok_or_else(|| {
                RerankError::Protocol("sidecar result has no visible public ID".into())
            })?;
            let row = (result.distance, public_id);
            super::retain_top_k(&mut best, result_limit, row);
        }
        Ok(())
    })?;
    let mut rows = best.into_vec();
    profile::update(|profile| {
        profile.sidecar_us += profile::duration_us(sidecar_timer.elapsed());
    });

    let result_timer = profile::ProfileTimer::start();
    rows.sort_unstable_by(|(left_distance, left_id), (right_distance, right_id)| {
        left_distance
            .cmp(right_distance)
            .then_with(|| left_id.cmp(right_id))
    });
    rows.truncate(top_k as usize);
    let output = rows
        .into_iter()
        .map(|(distance, public_id)| (public_id, -distance.to_f32()))
        .collect::<Vec<_>>();
    profile::update(|profile| {
        profile.result_finalize_us += profile::duration_us(result_timer.elapsed());
        profile.returned_rows = output.len() as u64;
    });
    profile_guard.finish(total_timer.elapsed());
    Ok(output.into_iter())
}

fn validate_top_k(candidate_count: usize, top_k: i32) -> Result<(), RerankError> {
    if top_k <= 0 || top_k as usize > candidate_count {
        return Err(RerankError::Configuration(
            "top_k must be positive and no greater than candidate_ids length",
        ));
    }
    Ok(())
}

fn require_mvcc_snapshot() -> Result<(), RerankError> {
    let snapshot = unsafe { pgrx::pg_sys::GetActiveSnapshot() };
    if snapshot.is_null()
        || unsafe { (*snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
    {
        return Err(RerankError::Configuration(
            "external TileMaxSim rerank requires an active MVCC snapshot",
        ));
    }
    Ok(())
}

fn preflight_descriptor_access(binding: &TileMaxsimSourceBinding) -> Result<(), RerankError> {
    let query = descriptor_query(binding, true)?;
    pgrx::spi::Spi::connect(|client| {
        let privilege = client
            .prepare(
                "SELECT pg_catalog.has_table_privilege($1, 'SELECT') AS allowed",
                &pgrx::oids_of![pgrx::pg_sys::Oid],
            )
            .map_err(registry_error)?;
        for relation_oid in std::iter::once(binding.source_oid).chain(binding.descriptor_oid) {
            let allowed = client
                .select(&privilege, Some(1), &[relation_oid.into()])
                .map_err(registry_error)?
                .first()
                .get_by_name::<bool, _>("allowed")
                .map_err(registry_error)?
                .unwrap_or(false);
            if !allowed {
                return Err(RerankError::Registry(
                    "table-level SELECT privilege is required for every TileMaxSim relation".into(),
                ));
            }
        }
        client
            .select(query.as_str(), Some(1), &[])
            .map(|_| ())
            .map_err(registry_error)
    })
}

fn load_visible_descriptors(
    binding: &TileMaxsimSourceBinding,
    candidate_ids: &[i64],
) -> Result<(MaterializedDescriptorSource, BTreeMap<HeapKey, i64>), RerankError> {
    let mut candidates_by_id = BTreeMap::new();
    for (ordinal, public_id) in candidate_ids.iter().copied().enumerate() {
        candidates_by_id.insert(public_id, candidate_for_ordinal(ordinal)?);
    }
    let query = descriptor_query(binding, false)?;
    let (descriptors, public_ids) = pgrx::spi::Spi::connect(|client| {
        let int8_array_oid = unsafe { pgrx::pg_sys::get_array_type(pgrx::pg_sys::INT8OID) };
        let prepared = client
            .prepare(
                query.as_str(),
                &[
                    pgrx::pg_sys::PgOid::from(int8_array_oid),
                    pgrx::pg_sys::PgOid::from(pgrx::pg_sys::TEXTOID),
                ],
            )
            .map_err(registry_error)?;
        let args: [DatumWithOid<'_>; 2] = [
            candidate_ids.to_vec().into(),
            binding.model_contract_id.clone().into(),
        ];
        let rows = client
            .select(&prepared, Some(candidate_ids.len() as _), &args)
            .map_err(registry_error)?;
        let mut descriptors = BTreeMap::new();
        let mut public_ids = BTreeMap::new();
        for row in rows {
            let public_id = required_heap_column::<i64>(&row, "public_id")?;
            let candidate = candidates_by_id.get(&public_id).copied().ok_or_else(|| {
                RerankError::Registry("descriptor query returned an unknown public ID".into())
            })?;
            let descriptor = validate_descriptor(
                candidate,
                public_id,
                required_heap_column::<String>(&row, "tensor_ref")?,
                required_heap_column::<i32>(&row, "tensor_rows")?,
                required_heap_column::<i32>(&row, "tensor_dimension")?,
                required_heap_column::<String>(&row, "tensor_dtype")?,
                required_heap_column::<String>(&row, "tensor_checksum")?,
            )?;
            if descriptors.insert(candidate.heap_key, descriptor).is_some()
                || public_ids.insert(candidate.heap_key, public_id).is_some()
            {
                return Err(RerankError::Registry(
                    "descriptor query returned a duplicate public ID".into(),
                ));
            }
        }
        Ok((descriptors, public_ids))
    })?;
    Ok((MaterializedDescriptorSource(descriptors), public_ids))
}

fn candidate_for_ordinal(ordinal: usize) -> Result<PageCandidate, RerankError> {
    let ordinal = u32::try_from(ordinal).map_err(|_| RerankError::RequestTooLarge)?;
    Ok(PageCandidate {
        approximate_distance: Distance::ZERO,
        heap_key: [0, (ordinal >> 16) as u16, ordinal as u16],
    })
}

fn descriptor_query(
    binding: &TileMaxsimSourceBinding,
    preflight: bool,
) -> Result<String, RerankError> {
    let source_relation = relation_name(binding.source_oid)?;
    let names = &binding.column_names;
    let model_contract = pgrx::spi::quote_identifier(&names.model_contract);
    let public_id = pgrx::spi::quote_identifier(&names.public_id);
    let tensor_ref = pgrx::spi::quote_identifier(&names.tensor_ref);
    let tensor_rows = pgrx::spi::quote_identifier(&names.tensor_rows);
    let tensor_dimension = pgrx::spi::quote_identifier(&names.tensor_dimension);
    let tensor_dtype = pgrx::spi::quote_identifier(&names.tensor_dtype);
    let tensor_checksum = pgrx::spi::quote_identifier(&names.tensor_checksum);
    let predicate = if preflight {
        "false".to_string()
    } else {
        format!("h.{public_id}::bigint = ANY($1) AND h.{model_contract} = $2")
    };
    match binding.storage {
        ExternalTensorStorage::SameHeap => Ok(format!(
            "SELECT h.{public_id}::bigint AS public_id,
                    h.{tensor_ref} AS tensor_ref,
                    h.{tensor_rows} AS tensor_rows,
                    h.{tensor_dimension} AS tensor_dimension,
                    h.{tensor_dtype} AS tensor_dtype,
                    h.{tensor_checksum} AS tensor_checksum
               FROM ONLY {source_relation} AS h
              WHERE {predicate}"
        )),
        ExternalTensorStorage::DescriptorRelation => {
            let descriptor_oid = binding.descriptor_oid.ok_or_else(|| {
                RerankError::Registry("registered descriptor relation is missing".into())
            })?;
            let descriptor_relation = relation_name(descriptor_oid)?;
            let descriptor_public_id = pgrx::spi::quote_identifier(
                names.descriptor_public_id.as_deref().ok_or_else(|| {
                    RerankError::Registry(
                        "registered descriptor public ID column is missing".into(),
                    )
                })?,
            );
            Ok(format!(
                "SELECT h.{public_id}::bigint AS public_id,
                        d.{tensor_ref} AS tensor_ref,
                        d.{tensor_rows} AS tensor_rows,
                        d.{tensor_dimension} AS tensor_dimension,
                        d.{tensor_dtype} AS tensor_dtype,
                        d.{tensor_checksum} AS tensor_checksum
                   FROM ONLY {source_relation} AS h
                   JOIN ONLY {descriptor_relation} AS d
                     ON d.{descriptor_public_id}::bigint = h.{public_id}::bigint
                  WHERE {predicate}"
            ))
        }
    }
}

fn relation_name(relation_oid: pgrx::pg_sys::Oid) -> Result<String, RerankError> {
    pgrx::spi::Spi::connect(|client| {
        let prepared = client
            .prepare(
                "SELECT n.nspname::text AS schema_name, c.relname::text AS relation_name
                   FROM pg_catalog.pg_class AS c
                   JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                  WHERE c.oid = $1",
                &pgrx::oids_of![pgrx::pg_sys::Oid],
            )
            .map_err(registry_error)?;
        let rows = client
            .select(&prepared, Some(1), &[relation_oid.into()])
            .map_err(registry_error)?;
        if rows.is_empty() {
            return Err(RerankError::Registry(
                "registered relation disappeared".into(),
            ));
        }
        let row = rows.first();
        Ok(pgrx::spi::quote_qualified_identifier(
            required_column::<String>(&row, "schema_name")?,
            required_column::<String>(&row, "relation_name")?,
        ))
    })
}

fn required_column<T>(row: &pgrx::spi::SpiTupleTable<'_>, name: &str) -> Result<T, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name)
        .map_err(registry_error)?
        .ok_or_else(|| RerankError::Registry(format!("query returned NULL {name}")))
}

fn required_heap_column<T>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Result<T, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name)
        .map_err(registry_error)?
        .ok_or(RerankError::InvalidDescriptor(
            "descriptor query returned NULL",
        ))
}

fn registry_error(error: impl std::fmt::Display) -> RerankError {
    RerankError::Registry(error.to_string())
}

#[derive(Default)]
struct MaterializedDescriptorSource(BTreeMap<HeapKey, ExternalTensorDescriptor>);

impl CandidateTensorDescriptorSource for MaterializedDescriptorSource {
    fn fetch(
        &mut self,
        candidate: PageCandidate,
    ) -> Result<Option<ExternalTensorDescriptor>, RerankError> {
        Ok(self.0.remove(&candidate.heap_key))
    }
}

struct RelationLock {
    raw: pgrx::pg_sys::Relation,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl RelationLock {
    fn open(oid: pgrx::pg_sys::Oid, lockmode: pgrx::pg_sys::LOCKMODE) -> Result<Self, RerankError> {
        let raw = unsafe { pgrx::pg_sys::relation_open(oid, lockmode) };
        if raw.is_null() {
            return Err(RerankError::Registry("relation open returned NULL".into()));
        }
        Ok(Self { raw, lockmode })
    }

    fn oid(&self) -> pgrx::pg_sys::Oid {
        unsafe { (*self.raw).rd_id }
    }
}

impl Drop for RelationLock {
    fn drop(&mut self) {
        unsafe { pgrx::pg_sys::relation_close(self.raw, self.lockmode) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_candidates_support_full_corpus_ordinals() {
        assert!(validate_top_k(256, 10).is_ok());
        assert!(validate_top_k(10, 0).is_err());
        assert!(validate_top_k(10, 11).is_err());
        let first = candidate_for_ordinal(0).unwrap().heap_key;
        let last = candidate_for_ordinal(1_000_000).unwrap().heap_key;
        assert_ne!(first, last);
    }
}
