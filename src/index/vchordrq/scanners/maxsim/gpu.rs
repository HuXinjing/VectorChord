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
    CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorDtype,
};
use super::profile;
use super::rerank::{CandidateTensorSource, ExactMaxsimBackend, RerankError, RerankResults};
use crate::index::gucs::PostgresMaxsimScoringProfile;
use distance::Distance;
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::io::{Read, Write};
use std::mem::size_of_val;
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use vchordrq::types::OwnedVector;

const MAGIC: &[u8; 4] = b"VCTM";
const VERSION: u16 = 1;
const EXTERNAL_VERSION: u16 = 2;
const SCHEDULED_EXTERNAL_VERSION: u16 = 3;
const PROFILED_EXTERNAL_VERSION: u16 = 4;
const QUANTIZED_EXTERNAL_VERSION: u16 = 5;
const LOGICAL_EXTERNAL_VERSION: u16 = 6;
const COMPACT_LOGICAL_EXTERNAL_VERSION: u16 = 7;
const TYPED_COMPACT_LOGICAL_EXTERNAL_VERSION: u16 = 8;
const MANIFEST_LOGICAL_EXTERNAL_VERSION: u16 = 9;
const CATALOG_LOGICAL_EXTERNAL_VERSION: u16 = 10;
const CATALOG_SELECTION_REFERENCE_VERSION: u16 = 11;
const SCOPED_CATALOG_SELECTION_REFERENCE_VERSION: u16 = 12;
const PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE_VERSION: u16 = 13;
const PERSISTENT_CATALOG_SELECTION_REFERENCE_VERSION: u16 = 14;
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;
const HEADER_LEN: usize = 24;
const MAX_REMOTE_ERROR_BYTES: usize = 64 * 1024;
const MAX_EXTERNAL_CANDIDATES_PER_BATCH: usize = 65_536;
const MAX_MODEL_CONTRACT_BYTES: usize = 512;
const MAX_TENSOR_REF_BYTES: usize = 4096;
const MAX_CHECKSUM_BYTES: usize = 512;
const MAX_TENANT_BYTES: usize = 256;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static LAST_FALLBACK_WARNING_SECONDS: AtomicU64 = AtomicU64::new(0);
const MAX_CATALOG_DTYPE_HINTS: usize = 64;
static CATALOG_DTYPE_HINTS: OnceLock<Mutex<VecDeque<(String, TensorDtype)>>> = OnceLock::new();
#[cfg(unix)]
const MAX_PERSISTENT_TRANSPORT_ENDPOINTS: usize = 16;
#[cfg(unix)]
static PERSISTENT_TRANSPORTS: OnceLock<Mutex<VecDeque<(String, TransportStream)>>> =
    OnceLock::new();
#[cfg(unix)]
static PERSISTENT_CAPABILITIES: OnceLock<Mutex<VecDeque<((String, u16), bool)>>> = OnceLock::new();

pub(super) fn report_gpu_fallback(error: &RerankError) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let should_warn = LAST_FALLBACK_WARNING_SECONDS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
            (last == 0 || now.saturating_sub(last) >= 60).then_some(now)
        })
        .is_ok();
    if should_warn {
        pgrx::warning!("GPU MaxSim failed; using cpu_exact: {error}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorDtype {
    F32 = 1,
    F16 = 2,
    Fp8E4m3 = 3,
}

pub(super) trait TileMaxsimTransport {
    fn round_trip(
        &mut self,
        request: &[u8],
        timeout: Duration,
        max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError>;
}

pub(super) struct GpuTileMaxsimBackend<T> {
    transport: T,
    timeout: Duration,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
}

impl<T> GpuTileMaxsimBackend<T> {
    pub fn new(
        transport: T,
        timeout: Duration,
        max_batch_tokens: usize,
        max_batch_bytes: usize,
    ) -> Self {
        Self {
            transport,
            timeout,
            max_batch_tokens,
            max_batch_bytes,
        }
    }
}

impl<T: TileMaxsimTransport> ExactMaxsimBackend for GpuTileMaxsimBackend<T> {
    type Results = RerankResults;

    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<Self::Results, RerankError> {
        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let encoded = encode_request(
            request_id,
            query,
            candidates,
            source,
            self.max_batch_tokens,
            self.max_batch_bytes,
        )?;
        if encoded.heap_keys.is_empty() {
            return Ok(RerankResults {
                inner: BinaryHeap::new(),
            });
        }
        let max_response_bytes = HEADER_LEN
            .checked_add(8)
            .and_then(|size| size.checked_add(encoded.heap_keys.len().checked_mul(8)?))
            .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
            .ok_or(RerankError::RequestTooLarge)?;
        let response =
            self.transport
                .round_trip(&encoded.frame, self.timeout, max_response_bytes)?;
        decode_response(&response, request_id, &encoded.heap_keys)
    }
}

/// GPU backend for Phase 3B external tensor descriptors.
///
/// This codec is intentionally separate from [`ExactMaxsimBackend`]: an
/// external full tensor may have different logical values from the indexed
/// sketch, so it cannot be substituted into ordinary `@#` execution.
pub(super) struct GpuExternalTileMaxsimBackend<T> {
    transport: T,
    model_contract_id: String,
    timeout: Duration,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<TileMaxsimScheduling>,
    scoring_profile: PostgresMaxsimScoringProfile,
    quantization_contract: Option<String>,
    catalog_revision: Option<String>,
    catalog_storage_dtype: Option<TensorDtype>,
}

#[derive(Clone, Debug)]
pub(super) struct TileMaxsimScheduling {
    pub tenant: String,
    pub priority: i32,
}

pub(super) enum CatalogSelectionOutcome {
    Hit(RerankResults),
    Miss,
    Unsupported,
}

pub(super) enum ScopedCatalogSelectionOutcome {
    Hit {
        global: RerankResults,
        scoped: RerankResults,
    },
    Miss,
    Unsupported,
}

fn catalog_dtype_hint(key: &str) -> Option<TensorDtype> {
    let hints = CATALOG_DTYPE_HINTS.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut hints = hints.lock().unwrap_or_else(|error| error.into_inner());
    let position = hints.iter().position(|(candidate, _)| candidate == key)?;
    let entry = hints.remove(position)?;
    let dtype = entry.1;
    hints.push_back(entry);
    Some(dtype)
}

fn remember_catalog_dtype(key: &str, dtype: TensorDtype) {
    let hints = CATALOG_DTYPE_HINTS.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut hints = hints.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(position) = hints.iter().position(|(candidate, _)| candidate == key) {
        hints.remove(position);
    }
    while hints.len() >= MAX_CATALOG_DTYPE_HINTS {
        hints.pop_front();
    }
    hints.push_back((key.to_string(), dtype));
}

impl<T> GpuExternalTileMaxsimBackend<T> {
    pub(super) fn new(
        transport: T,
        model_contract_id: String,
        timeout: Duration,
        max_batch_tokens: usize,
        max_batch_bytes: usize,
    ) -> Self {
        Self {
            transport,
            model_contract_id,
            timeout,
            max_batch_tokens,
            max_batch_bytes,
            scheduling: None,
            scoring_profile: PostgresMaxsimScoringProfile::ExactFp16,
            quantization_contract: None,
            catalog_revision: None,
            catalog_storage_dtype: None,
        }
    }

    pub(super) fn with_quantization_contract(mut self, contract: Option<String>) -> Self {
        self.quantization_contract = contract;
        self
    }

    pub(super) fn with_scoring_profile(
        mut self,
        scoring_profile: PostgresMaxsimScoringProfile,
    ) -> Self {
        self.scoring_profile = scoring_profile;
        self
    }

    pub(super) fn with_scheduling(mut self, tenant: String, priority: i32) -> Self {
        self.scheduling = Some(TileMaxsimScheduling { tenant, priority });
        self
    }

    pub(super) fn with_catalog_revision(mut self, revision: Option<String>) -> Self {
        self.catalog_revision = revision.filter(|value| {
            !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
        });
        self
    }

    pub(super) fn with_catalog_storage_dtype(mut self, dtype: Option<ExternalTensorDtype>) -> Self {
        self.catalog_storage_dtype = dtype.map(|dtype| match dtype {
            ExternalTensorDtype::F32 => TensorDtype::F32,
            ExternalTensorDtype::F16 => TensorDtype::F16,
            ExternalTensorDtype::Fp8E4m3 => TensorDtype::Fp8E4m3,
        });
        self
    }

    fn catalog_storage_dtypes(&self, hint_key: &str) -> Vec<TensorDtype> {
        let mut dtypes = vec![TensorDtype::Fp8E4m3, TensorDtype::F16, TensorDtype::F32];
        let preferred = self
            .catalog_storage_dtype
            .or_else(|| catalog_dtype_hint(hint_key));
        if let Some(preferred) = preferred {
            dtypes.retain(|dtype| *dtype != preferred);
            dtypes.insert(0, preferred);
        }
        dtypes
    }
}

impl<T: TileMaxsimTransport> GpuExternalTileMaxsimBackend<T> {
    /// Probe a warm raw-FP8 descriptor catalog without reading descriptor rows
    /// from PostgreSQL. A miss deliberately returns before materialization so
    /// the caller can load descriptors only for the registration slow path.
    pub(super) fn rerank_raw_fp8_catalog(
        &mut self,
        query: &[OwnedVector],
        public_ids: &[i64],
        top_k: usize,
    ) -> Result<CatalogSelectionOutcome, RerankError> {
        let Some(revision) = self.catalog_revision.as_deref() else {
            return Ok(CatalogSelectionOutcome::Unsupported);
        };
        if self.scoring_profile != PostgresMaxsimScoringProfile::RawFp8E4m3 {
            return Ok(CatalogSelectionOutcome::Unsupported);
        }
        if public_ids.is_empty() {
            return Ok(CatalogSelectionOutcome::Hit(RerankResults {
                inner: BinaryHeap::new(),
            }));
        }
        if top_k == 0 || public_ids.len() > MAX_EXTERNAL_CANDIDATES_PER_BATCH {
            return Err(RerankError::RequestTooLarge);
        }
        let mut sorted_ids = public_ids.to_vec();
        sorted_ids.sort_unstable();
        if sorted_ids[0] <= 0 || sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RerankError::InvalidDescriptor(
                "catalog public IDs must be positive and unique",
            ));
        }
        let max_response_bytes = HEADER_LEN
            .checked_add(8)
            .and_then(|size| size.checked_add(top_k.min(sorted_ids.len()).checked_mul(8)?))
            .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
            .ok_or(RerankError::RequestTooLarge)?;
        let (_, dimension) = tensor_metadata(query)?;
        let dtype_hint_key = format!(
            "{}\0{}\0{}\0{}\0{}",
            self.model_contract_id,
            self.scheduling
                .as_ref()
                .map_or("__default__", |value| value.tenant.as_str()),
            scoring_profile_code(self.scoring_profile),
            dimension,
            revision,
        );
        // Raw FP8 is the desired Hopper storage contract, but a rolling
        // migration may still have an otherwise identical FP16/F32 catalog.
        // Probe those bounded protocol-defined dtypes before touching the
        // descriptor relation. Mixed-dtype candidate sets fail all probes and
        // deliberately fall back to the materializing registration path.
        let storage_dtypes = self.catalog_storage_dtypes(&dtype_hint_key);
        for storage_dtype in storage_dtypes {
            let reference_request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
            let (reference_frame, reference_heap_keys) =
                encode_raw_fp8_catalog_selection_reference(
                    reference_request_id,
                    &self.model_contract_id,
                    query,
                    &sorted_ids,
                    self.max_batch_tokens,
                    self.max_batch_bytes,
                    self.scheduling.as_ref(),
                    revision,
                    self.timeout,
                    top_k.min(sorted_ids.len()),
                    storage_dtype,
                )?;
            let reference_response =
                self.transport
                    .round_trip(&reference_frame, self.timeout, max_response_bytes)?;
            match decode_response_for_version(
                &reference_response,
                CATALOG_SELECTION_REFERENCE_VERSION,
                reference_request_id,
                &reference_heap_keys,
                top_k.min(sorted_ids.len()),
            ) {
                Ok(results) => {
                    remember_catalog_dtype(&dtype_hint_key, storage_dtype);
                    return Ok(CatalogSelectionOutcome::Hit(results));
                }
                Err(RerankError::Remote(message)) if message == "descriptor catalog miss" => {}
                Err(RerankError::Protocol(message)) if message == "unsupported version" => {}
                Err(error) => return Err(error),
            }
            let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
            let (frame, heap_keys) = encode_raw_fp8_catalog_selection(
                request_id,
                &self.model_contract_id,
                query,
                &sorted_ids,
                self.max_batch_tokens,
                self.max_batch_bytes,
                self.scheduling.as_ref(),
                revision,
                self.timeout,
                top_k.min(sorted_ids.len()),
                storage_dtype,
            )?;
            let response = self
                .transport
                .round_trip(&frame, self.timeout, max_response_bytes)?;
            match decode_response_for_version(
                &response,
                CATALOG_LOGICAL_EXTERNAL_VERSION,
                request_id,
                &heap_keys,
                top_k.min(sorted_ids.len()),
            ) {
                Ok(results) => {
                    remember_catalog_dtype(&dtype_hint_key, storage_dtype);
                    return Ok(CatalogSelectionOutcome::Hit(results));
                }
                Err(RerankError::Remote(message)) if message == "descriptor catalog miss" => {}
                Err(RerankError::Protocol(message)) if message == "unsupported version" => {
                    return Ok(CatalogSelectionOutcome::Unsupported);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(CatalogSelectionOutcome::Miss)
    }

    /// Score one catalog selection once while retaining two independent
    /// windows in the daemon. This avoids both a second GPU request and the
    /// full-score response previously needed to derive an additive scope.
    pub(super) fn rerank_raw_fp8_catalog_scoped(
        &mut self,
        query: &[OwnedVector],
        public_ids: &[i64],
        scoped_public_ids: &[i64],
        top_k: usize,
    ) -> Result<ScopedCatalogSelectionOutcome, RerankError> {
        let Some(revision) = self.catalog_revision.as_deref() else {
            return Ok(ScopedCatalogSelectionOutcome::Unsupported);
        };
        if self.scoring_profile != PostgresMaxsimScoringProfile::RawFp8E4m3 {
            return Ok(ScopedCatalogSelectionOutcome::Unsupported);
        }
        if public_ids.is_empty() || scoped_public_ids.is_empty() || top_k == 0 {
            return Err(RerankError::Configuration(
                "scoped catalog rerank requires nonempty candidates, scope, and top-k",
            ));
        }
        let mut sorted_ids = public_ids.to_vec();
        sorted_ids.sort_unstable();
        if sorted_ids.len() > MAX_EXTERNAL_CANDIDATES_PER_BATCH
            || sorted_ids[0] <= 0
            || sorted_ids.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(RerankError::InvalidDescriptor(
                "catalog public IDs must be positive and unique",
            ));
        }
        let mut sorted_scope = scoped_public_ids.to_vec();
        sorted_scope.sort_unstable();
        sorted_scope.dedup();
        if sorted_scope
            .iter()
            .any(|id| sorted_ids.binary_search(id).is_err())
        {
            return Err(RerankError::InvalidDescriptor(
                "scoped catalog public IDs must be a subset of the candidate set",
            ));
        }
        let expected_global = top_k.min(sorted_ids.len());
        let expected_scoped = top_k.min(sorted_scope.len());
        let max_response_bytes = HEADER_LEN
            .checked_add(8)
            .and_then(|size| {
                size.checked_add(
                    expected_global
                        .checked_add(expected_scoped)?
                        .checked_mul(8)?,
                )
            })
            .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
            .ok_or(RerankError::RequestTooLarge)?;
        let (_, dimension) = tensor_metadata(query)?;
        let dtype_hint_key = format!(
            "{}\0{}\0{}\0{}\0{}",
            self.model_contract_id,
            self.scheduling
                .as_ref()
                .map_or("__default__", |value| value.tenant.as_str()),
            scoring_profile_code(self.scoring_profile),
            dimension,
            revision,
        );
        let storage_dtypes = self.catalog_storage_dtypes(&dtype_hint_key);
        for storage_dtype in storage_dtypes {
            let encode_timer = profile::ProfileTimer::start();
            let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
            let (frame, heap_keys) = encode_raw_fp8_scoped_catalog_selection_reference(
                request_id,
                &self.model_contract_id,
                query,
                &sorted_ids,
                &sorted_scope,
                self.max_batch_tokens,
                self.max_batch_bytes,
                self.scheduling.as_ref(),
                revision,
                self.timeout,
                expected_global,
                storage_dtype,
            )?;
            profile::update(|entry| {
                entry.sidecar_encode_us += profile::duration_us(encode_timer.elapsed());
            });
            let transport_timer = profile::ProfileTimer::start();
            let response = self
                .transport
                .round_trip(&frame, self.timeout, max_response_bytes)?;
            profile::update(|entry| {
                entry.sidecar_transport_us += profile::duration_us(transport_timer.elapsed());
            });
            let decode_timer = profile::ProfileTimer::start();
            let decoded = decode_scoped_response(
                &response,
                request_id,
                &heap_keys,
                expected_global,
                expected_scoped,
            );
            profile::update(|entry| {
                entry.sidecar_decode_us += profile::duration_us(decode_timer.elapsed());
            });
            match decoded {
                Ok((global, scoped)) => {
                    remember_catalog_dtype(&dtype_hint_key, storage_dtype);
                    return Ok(ScopedCatalogSelectionOutcome::Hit { global, scoped });
                }
                Err(RerankError::Remote(message)) if message == "descriptor catalog miss" => {}
                Err(RerankError::Protocol(message)) if message == "unsupported version" => {
                    return Ok(ScopedCatalogSelectionOutcome::Unsupported);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(ScopedCatalogSelectionOutcome::Miss)
    }

    /// Submit a complete external descriptor set as one logical request. The
    /// daemon performs bounded cooperative GPU slicing and global top-k, which
    /// avoids one TCP round trip per client-side tensor-token batch.
    pub(super) fn rerank_logical<S: CandidateTensorDescriptorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
        top_k: usize,
    ) -> Result<RerankResults, RerankError> {
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .ok_or_else(|| RerankError::Transport("logical request deadline overflow".into()))?;
        let mut descriptors = Vec::new();
        for candidate in candidates {
            if let Some(descriptor) = source.fetch(candidate)? {
                descriptors.push(descriptor);
            }
        }
        if descriptors.is_empty() {
            return Ok(RerankResults {
                inner: BinaryHeap::new(),
            });
        }
        if top_k == 0 {
            return Err(RerankError::Configuration(
                "logical top-k must be greater than zero",
            ));
        }
        // A descriptor source may legitimately omit stale or unavailable
        // candidates. Preserve the previous batched behavior by returning as
        // many results as remain instead of rejecting the whole rerank.
        // A rolling upgrade may mix canonical FP16 and raw FP8 objects. The
        // compact protocol has one request-wide dtype, so score homogeneous
        // groups independently and merge their bounded top-k results.
        let groups = if self.scoring_profile == PostgresMaxsimScoringProfile::RawFp8E4m3 {
            let (raw, legacy): (Vec<_>, Vec<_>) = descriptors
                .into_iter()
                .partition(|item| item.dtype == ExternalTensorDtype::Fp8E4m3);
            [legacy, raw]
                .into_iter()
                .filter(|group| !group.is_empty())
                .collect::<Vec<_>>()
        } else {
            vec![descriptors]
        };
        let mut merged = BinaryHeap::new();
        for mut group in groups {
            if self.catalog_revision.is_some() {
                group.sort_unstable_by_key(|descriptor| descriptor.public_id);
                if group.iter().any(|descriptor| descriptor.public_id <= 0)
                    || group
                        .windows(2)
                        .any(|pair| pair[0].public_id == pair[1].public_id)
                {
                    return Err(RerankError::InvalidDescriptor(
                        "catalog public IDs must be positive and unique",
                    ));
                }
            }
            let group_top_k = top_k.min(group.len());
            let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
            let encoded = encode_external_descriptors(
                request_id,
                &self.model_contract_id,
                query,
                &group,
                self.max_batch_tokens,
                self.max_batch_bytes,
                self.scheduling.as_ref(),
                self.scoring_profile,
                self.quantization_contract.as_deref(),
                remaining_logical_timeout(deadline)?,
                Some(group_top_k),
            )?;
            let max_response_bytes = HEADER_LEN
                .checked_add(8)
                .and_then(|size| size.checked_add(group_top_k.checked_mul(8)?))
                .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
                .ok_or(RerankError::RequestTooLarge)?;
            if let Some(revision) = self.catalog_revision.as_deref() {
                let (reference, registration) = catalog_frames(&encoded, &group, revision)?;
                let response = self.transport.round_trip(
                    &reference,
                    remaining_logical_timeout(deadline)?,
                    max_response_bytes,
                )?;
                let decoded = match decode_response_for_version(
                    &response,
                    CATALOG_LOGICAL_EXTERNAL_VERSION,
                    request_id,
                    &encoded.heap_keys,
                    group_top_k,
                ) {
                    Err(RerankError::Remote(message)) if message == "descriptor catalog miss" => {
                        let response = self.transport.round_trip(
                            &registration,
                            remaining_logical_timeout(deadline)?,
                            max_response_bytes,
                        )?;
                        decode_response_for_version(
                            &response,
                            CATALOG_LOGICAL_EXTERNAL_VERSION,
                            request_id,
                            &encoded.heap_keys,
                            group_top_k,
                        )?
                    }
                    Err(RerankError::Protocol(message)) if message == "unsupported version" => self
                        .round_trip_manifest_fallback(
                            &encoded,
                            request_id,
                            group_top_k,
                            deadline,
                            max_response_bytes,
                        )?,
                    other => other?,
                };
                merged.extend(decoded.inner);
                continue;
            }
            let decoded = self.round_trip_manifest_fallback(
                &encoded,
                request_id,
                group_top_k,
                deadline,
                max_response_bytes,
            )?;
            merged.extend(decoded.inner);
        }
        let mut retained = BinaryHeap::new();
        for _ in 0..top_k.min(merged.len()) {
            retained.push(merged.pop().expect("bounded merged heap length"));
        }
        Ok(RerankResults { inner: retained })
    }

    fn round_trip_manifest_fallback(
        &mut self,
        encoded: &EncodedRequest,
        request_id: u64,
        group_top_k: usize,
        deadline: Instant,
        max_response_bytes: usize,
    ) -> Result<RerankResults, RerankError> {
        let reference = manifest_reference(encoded)?;
        let response = self.transport.round_trip(
            &reference,
            remaining_logical_timeout(deadline)?,
            max_response_bytes,
        )?;
        let decoded = decode_response_for_version(
            &response,
            MANIFEST_LOGICAL_EXTERNAL_VERSION,
            request_id,
            &encoded.heap_keys,
            group_top_k,
        );
        let decoded = match decoded {
            Err(RerankError::Remote(message)) if message == "descriptor manifest miss" => {
                let response = self.transport.round_trip(
                    &encoded.frame,
                    remaining_logical_timeout(deadline)?,
                    max_response_bytes,
                )?;
                decode_response_for_version(
                    &response,
                    encoded.version,
                    request_id,
                    &encoded.heap_keys,
                    group_top_k,
                )?
            }
            Err(RerankError::Protocol(message)) if message == "unsupported version" => {
                // A v2-v8 daemon rejects the v9 reference before it can
                // report a manifest miss. Retry the already validated v8
                // registration frame so extension-first rolling upgrades
                // remain available.
                let response = self.transport.round_trip(
                    &encoded.frame,
                    remaining_logical_timeout(deadline)?,
                    max_response_bytes,
                )?;
                decode_response_for_version(
                    &response,
                    encoded.version,
                    request_id,
                    &encoded.heap_keys,
                    group_top_k,
                )?
            }
            other => other?,
        };
        Ok(decoded)
    }

    #[cfg(test)]
    pub(super) fn rerank<S: CandidateTensorDescriptorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<RerankResults, RerankError> {
        let mut inner = BinaryHeap::new();
        self.rerank_batches(query, candidates, source, |batch| {
            inner.extend(batch.inner);
            Ok(())
        })?;
        Ok(RerankResults { inner })
    }

    /// Execute one logical rerank as bounded sidecar requests.
    ///
    /// The callback is invoked after every complete batch so callers that only
    /// need a global top-k do not have to retain every score. `self.timeout` is
    /// a deadline for the whole logical query, rather than a fresh timeout for
    /// each IPC request.
    pub(super) fn rerank_batches<S, F>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
        mut consume: F,
    ) -> Result<(), RerankError>
    where
        S: CandidateTensorDescriptorSource,
        F: FnMut(RerankResults) -> Result<(), RerankError>,
    {
        let deadline = Instant::now()
            .checked_add(self.timeout)
            .ok_or_else(|| RerankError::Transport("logical request deadline overflow".into()))?;
        let mut pending = None;

        loop {
            let descriptors = collect_external_batch(
                &self.model_contract_id,
                query,
                candidates,
                source,
                self.max_batch_tokens,
                self.max_batch_bytes,
                self.scheduling.as_ref(),
                self.scoring_profile,
                self.quantization_contract.as_deref(),
                &mut pending,
            )?;
            if descriptors.is_empty() {
                return Ok(());
            }

            let remaining = remaining_logical_timeout(deadline)?;
            let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
            let encoded = encode_external_descriptors(
                request_id,
                &self.model_contract_id,
                query,
                &descriptors,
                self.max_batch_tokens,
                self.max_batch_bytes,
                self.scheduling.as_ref(),
                self.scoring_profile,
                self.quantization_contract.as_deref(),
                remaining,
                None,
            )?;
            let max_response_bytes = HEADER_LEN
                .checked_add(8)
                .and_then(|size| size.checked_add(encoded.heap_keys.len().checked_mul(8)?))
                .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
                .ok_or(RerankError::RequestTooLarge)?;
            let response = self.transport.round_trip(
                &encoded.frame,
                remaining_logical_timeout(deadline)?,
                max_response_bytes,
            )?;
            consume(decode_response_for_version(
                &response,
                encoded.version,
                request_id,
                &encoded.heap_keys,
                encoded.heap_keys.len(),
            )?)?;
        }
    }
}

fn remaining_logical_timeout(deadline: Instant) -> Result<Duration, RerankError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| RerankError::Transport("logical request timed out".into()))
}

struct EncodedRequest {
    frame: Vec<u8>,
    heap_keys: Vec<HeapKey>,
    version: u16,
    descriptor_offset: usize,
}

fn manifest_reference(encoded: &EncodedRequest) -> Result<Vec<u8>, RerankError> {
    if !matches!(
        encoded.version,
        COMPACT_LOGICAL_EXTERNAL_VERSION | TYPED_COMPACT_LOGICAL_EXTERNAL_VERSION
    ) {
        return Err(RerankError::Protocol(
            "manifest reference requires a compact logical request".into(),
        ));
    }
    let mut hash = Sha256::new();
    hash.update(&encoded.frame[encoded.descriptor_offset..]);
    let digest = hash.finalize();
    let mut frame = encoded.frame[..encoded.descriptor_offset].to_vec();
    frame.extend_from_slice(&digest);
    frame[4..6].copy_from_slice(&MANIFEST_LOGICAL_EXTERNAL_VERSION.to_le_bytes());
    let body_len = frame
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or(RerankError::RequestTooLarge)?;
    frame[16..24].copy_from_slice(
        &u64::try_from(body_len)
            .map_err(|_| RerankError::RequestTooLarge)?
            .to_le_bytes(),
    );
    Ok(frame)
}

#[allow(clippy::too_many_arguments)]
fn encode_raw_fp8_catalog_selection(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    public_ids: &[i64],
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    revision: &str,
    timeout: Duration,
    top_k: usize,
    storage_dtype: TensorDtype,
) -> Result<(Vec<u8>, Vec<HeapKey>), RerankError> {
    if model_contract_id.is_empty()
        || model_contract_id.len() > MAX_MODEL_CONTRACT_BYTES
        || model_contract_id.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    if let Some(scheduling) = scheduling {
        if scheduling.tenant.is_empty()
            || scheduling.tenant.len() > MAX_TENANT_BYTES
            || scheduling.tenant.chars().any(char::is_control)
            || !(-100..=100).contains(&scheduling.priority)
        {
            return Err(RerankError::Configuration(
                "TileMaxSim scheduler tenant or priority is invalid",
            ));
        }
    }
    let (query_dtype, dimension) = tensor_metadata(query)?;
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    if query.len() > max_batch_tokens
        || tensor_bytes(query_rows, dimension, query_dtype)? > max_batch_bytes
        || public_ids.len() > MAX_EXTERNAL_CANDIDATES_PER_BATCH
    {
        return Err(RerankError::RequestTooLarge);
    }
    let tenant = scheduling.map_or("__default__", |value| value.tenant.as_str());
    let priority = scheduling.map_or(0, |value| value.priority);
    let mut writer = BoundedWriter::new(max_batch_bytes);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    writer.u32(u32::try_from(public_ids.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u8(query_dtype as u8)?;
    writer.u8(1)?;
    writer.u8(scoring_profile_code(
        PostgresMaxsimScoringProfile::RawFp8E4m3,
    ))?;
    writer.u8(if query_dtype == storage_dtype {
        0
    } else {
        storage_dtype as u8
    })?;
    writer
        .u32(u32::try_from(model_contract_id.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u32(0)?;
    writer.i32(priority)?;
    writer.u32(
        u32::try_from(timeout.as_millis().clamp(1, 600_000))
            .map_err(|_| RerankError::RequestTooLarge)?,
    )?;
    writer.u32(u32::try_from(tenant.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u32(u32::try_from(top_k).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.bytes(model_contract_id.as_bytes())?;
    writer.bytes(tenant.as_bytes())?;
    encode_tensor_values(&mut writer, query, query_dtype)?;
    writer.u8(1)?;
    writer.bytes(&Sha256::digest(revision.as_bytes()))?;
    let mut previous = 0_u64;
    let mut heap_keys = Vec::with_capacity(public_ids.len());
    for (ordinal, public_id) in public_ids.iter().copied().enumerate() {
        let current = u64::try_from(public_id)
            .map_err(|_| RerankError::InvalidDescriptor("catalog public ID is not positive"))?;
        let delta = current
            .checked_sub(previous)
            .filter(|value| *value > 0)
            .ok_or(RerankError::InvalidDescriptor(
                "catalog public IDs are not strictly increasing",
            ))?;
        let mut remaining = delta;
        while remaining >= 0x80 {
            writer.u8((remaining as u8 & 0x7f) | 0x80)?;
            remaining >>= 7;
        }
        writer.u8(remaining as u8)?;
        let ordinal = u32::try_from(ordinal).map_err(|_| RerankError::RequestTooLarge)?;
        heap_keys.push([0, (ordinal >> 16) as u16, ordinal as u16]);
        previous = current;
    }
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, CATALOG_LOGICAL_EXTERNAL_VERSION);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(
        16,
        u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?,
    );
    Ok((writer.finish(), heap_keys))
}

#[allow(clippy::too_many_arguments)]
fn encode_raw_fp8_catalog_selection_reference(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    public_ids: &[i64],
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    revision: &str,
    timeout: Duration,
    top_k: usize,
    storage_dtype: TensorDtype,
) -> Result<(Vec<u8>, Vec<HeapKey>), RerankError> {
    if model_contract_id.is_empty()
        || model_contract_id.len() > MAX_MODEL_CONTRACT_BYTES
        || model_contract_id.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    if let Some(scheduling) = scheduling {
        if scheduling.tenant.is_empty()
            || scheduling.tenant.len() > MAX_TENANT_BYTES
            || scheduling.tenant.chars().any(char::is_control)
            || !(-100..=100).contains(&scheduling.priority)
        {
            return Err(RerankError::Configuration(
                "TileMaxSim scheduler tenant or priority is invalid",
            ));
        }
    }
    let (query_dtype, dimension) = tensor_metadata(query)?;
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    if query.len() > max_batch_tokens
        || tensor_bytes(query_rows, dimension, query_dtype)? > max_batch_bytes
        || public_ids.is_empty()
        || public_ids.len() > MAX_EXTERNAL_CANDIDATES_PER_BATCH
    {
        return Err(RerankError::RequestTooLarge);
    }
    let mut selection_digest = Sha256::new();
    let mut heap_keys = Vec::with_capacity(public_ids.len());
    let mut previous = 0_i64;
    for (ordinal, public_id) in public_ids.iter().copied().enumerate() {
        if public_id <= previous {
            return Err(RerankError::InvalidDescriptor(
                "catalog public IDs are not strictly increasing",
            ));
        }
        selection_digest.update(public_id.to_le_bytes());
        let ordinal = u32::try_from(ordinal).map_err(|_| RerankError::RequestTooLarge)?;
        heap_keys.push([0, (ordinal >> 16) as u16, ordinal as u16]);
        previous = public_id;
    }
    let tenant = scheduling.map_or("__default__", |value| value.tenant.as_str());
    let priority = scheduling.map_or(0, |value| value.priority);
    let mut writer = BoundedWriter::new(max_batch_bytes);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    writer.u32(u32::try_from(public_ids.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u8(query_dtype as u8)?;
    writer.u8(1)?;
    writer.u8(scoring_profile_code(
        PostgresMaxsimScoringProfile::RawFp8E4m3,
    ))?;
    writer.u8(if query_dtype == storage_dtype {
        0
    } else {
        storage_dtype as u8
    })?;
    writer
        .u32(u32::try_from(model_contract_id.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u32(0)?;
    writer.i32(priority)?;
    writer.u32(
        u32::try_from(timeout.as_millis().clamp(1, 600_000))
            .map_err(|_| RerankError::RequestTooLarge)?,
    )?;
    writer.u32(u32::try_from(tenant.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.u32(u32::try_from(top_k).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.bytes(model_contract_id.as_bytes())?;
    writer.bytes(tenant.as_bytes())?;
    encode_tensor_values(&mut writer, query, query_dtype)?;
    writer.u8(3)?;
    writer.bytes(&Sha256::digest(revision.as_bytes()))?;
    writer.bytes(&selection_digest.finalize())?;
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, CATALOG_SELECTION_REFERENCE_VERSION);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(
        16,
        u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?,
    );
    Ok((writer.finish(), heap_keys))
}

#[allow(clippy::too_many_arguments)]
fn encode_raw_fp8_scoped_catalog_selection_reference(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    public_ids: &[i64],
    scoped_public_ids: &[i64],
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    revision: &str,
    timeout: Duration,
    top_k: usize,
    storage_dtype: TensorDtype,
) -> Result<(Vec<u8>, Vec<HeapKey>), RerankError> {
    let (mut frame, heap_keys) = encode_raw_fp8_catalog_selection_reference(
        request_id,
        model_contract_id,
        query,
        public_ids,
        max_batch_tokens,
        max_batch_bytes,
        scheduling,
        revision,
        timeout,
        top_k,
        storage_dtype,
    )?;
    let mut ordinals = Vec::with_capacity(scoped_public_ids.len());
    for public_id in scoped_public_ids {
        let ordinal = public_ids.binary_search(public_id).map_err(|_| {
            RerankError::InvalidDescriptor(
                "scoped catalog public IDs must be a subset of the candidate set",
            )
        })?;
        ordinals.push(u32::try_from(ordinal).map_err(|_| RerankError::RequestTooLarge)?);
    }
    ordinals.sort_unstable();
    ordinals.dedup();
    if ordinals.is_empty() {
        return Err(RerankError::InvalidDescriptor(
            "scoped catalog public IDs must not be empty",
        ));
    }
    // v11 ends after the two catalog digests. v12 changes only the mode byte
    // and appends a compact, delta-varint ordinal subset.
    let mode_offset = frame
        .len()
        .checked_sub(65)
        .ok_or(RerankError::RequestTooLarge)?;
    if frame.get(mode_offset).copied() != Some(3) {
        return Err(RerankError::Protocol(
            "invalid catalog selection reference layout".into(),
        ));
    }
    frame[mode_offset] = 4;
    frame.extend_from_slice(
        &u32::try_from(ordinals.len())
            .map_err(|_| RerankError::RequestTooLarge)?
            .to_le_bytes(),
    );
    let mut previous_plus_one = 0_u64;
    for ordinal in ordinals {
        let current_plus_one = u64::from(ordinal) + 1;
        let mut delta = current_plus_one
            .checked_sub(previous_plus_one)
            .filter(|value| *value > 0)
            .ok_or(RerankError::InvalidDescriptor(
                "scoped catalog ordinals are not strictly increasing",
            ))?;
        while delta >= 0x80 {
            frame.push((delta as u8 & 0x7f) | 0x80);
            delta >>= 7;
        }
        frame.push(delta as u8);
        previous_plus_one = current_plus_one;
    }
    if frame.len() > max_batch_bytes {
        return Err(RerankError::RequestTooLarge);
    }
    frame[4..6].copy_from_slice(&SCOPED_CATALOG_SELECTION_REFERENCE_VERSION.to_le_bytes());
    let body_len =
        u64::try_from(frame.len() - HEADER_LEN).map_err(|_| RerankError::RequestTooLarge)?;
    frame[16..24].copy_from_slice(&body_len.to_le_bytes());
    Ok((frame, heap_keys))
}

fn catalog_frames(
    encoded: &EncodedRequest,
    descriptors: &[ExternalTensorDescriptor],
    revision: &str,
) -> Result<(Vec<u8>, Vec<u8>), RerankError> {
    if !matches!(
        encoded.version,
        COMPACT_LOGICAL_EXTERNAL_VERSION | TYPED_COMPACT_LOGICAL_EXTERNAL_VERSION
    ) || descriptors.len() != encoded.heap_keys.len()
    {
        return Err(RerankError::Protocol(
            "catalog frames require a compact logical request".into(),
        ));
    }
    let digest: [u8; 32] = Sha256::digest(revision.as_bytes()).into();
    let mut reference = encoded.frame[..encoded.descriptor_offset].to_vec();
    reference.push(1);
    reference.extend_from_slice(&digest);
    let mut previous = 0_u64;
    for descriptor in descriptors {
        let current = u64::try_from(descriptor.public_id)
            .map_err(|_| RerankError::InvalidDescriptor("catalog public ID is not positive"))?;
        let delta = current
            .checked_sub(previous)
            .filter(|value| *value > 0)
            .ok_or(RerankError::InvalidDescriptor(
                "catalog public IDs are not strictly increasing",
            ))?;
        encode_varint(&mut reference, delta);
        previous = current;
    }
    finalize_catalog_frame(&mut reference)?;

    let mut registration = encoded.frame[..encoded.descriptor_offset].to_vec();
    registration.push(2);
    registration.extend_from_slice(&digest);
    for descriptor in descriptors {
        registration.extend_from_slice(&descriptor.public_id.to_le_bytes());
        registration.extend_from_slice(&descriptor.rows.to_le_bytes());
        let digest = descriptor
            .tensor_ref
            .strip_prefix("sha256://")
            .and_then(|value| decode_lower_hex_sha256(value.as_bytes()))
            .ok_or(RerankError::InvalidDescriptor("invalid tensor digest"))?;
        registration.extend_from_slice(&digest);
    }
    finalize_catalog_frame(&mut registration)?;
    Ok((reference, registration))
}

fn encode_varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn finalize_catalog_frame(frame: &mut [u8]) -> Result<(), RerankError> {
    frame[4..6].copy_from_slice(&CATALOG_LOGICAL_EXTERNAL_VERSION.to_le_bytes());
    let body_len = frame
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or(RerankError::RequestTooLarge)?;
    frame[16..24].copy_from_slice(
        &u64::try_from(body_len)
            .map_err(|_| RerankError::RequestTooLarge)?
            .to_le_bytes(),
    );
    Ok(())
}

#[cfg(test)]
fn encode_external_request<S: CandidateTensorDescriptorSource>(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    candidates: &mut dyn Iterator<Item = PageCandidate>,
    source: &mut S,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    timeout: Duration,
) -> Result<EncodedRequest, RerankError> {
    let mut descriptors = Vec::new();
    for candidate in candidates {
        if let Some(descriptor) = source.fetch(candidate)? {
            descriptors.push(descriptor);
        }
    }
    encode_external_descriptors(
        request_id,
        model_contract_id,
        query,
        &descriptors,
        max_batch_tokens,
        max_batch_bytes,
        scheduling,
        PostgresMaxsimScoringProfile::ExactFp16,
        None,
        timeout,
        None,
    )
}

fn encode_external_descriptors(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    descriptors: &[ExternalTensorDescriptor],
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    scoring_profile: PostgresMaxsimScoringProfile,
    quantization_contract: Option<&str>,
    timeout: Duration,
    logical_top_k: Option<usize>,
) -> Result<EncodedRequest, RerankError> {
    if model_contract_id.is_empty()
        || model_contract_id.len() > MAX_MODEL_CONTRACT_BYTES
        || model_contract_id.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    if let Some(scheduling) = scheduling {
        if scheduling.tenant.is_empty()
            || scheduling.tenant.len() > MAX_TENANT_BYTES
            || scheduling.tenant.chars().any(char::is_control)
            || !(-100..=100).contains(&scheduling.priority)
        {
            return Err(RerankError::Configuration(
                "TileMaxSim scheduler tenant or priority is invalid",
            ));
        }
    }
    let (dtype, dimension) = tensor_metadata(query)?;
    let external_dtype = descriptors.first().map_or_else(
        || match dtype {
            TensorDtype::F32 => ExternalTensorDtype::F32,
            TensorDtype::F16 => ExternalTensorDtype::F16,
            TensorDtype::Fp8E4m3 => unreachable!("query tensors originate as vector or halfvec"),
        },
        |item| item.dtype,
    );
    let storage_dtype = match external_dtype {
        ExternalTensorDtype::F32 => TensorDtype::F32,
        ExternalTensorDtype::F16 => TensorDtype::F16,
        ExternalTensorDtype::Fp8E4m3 => TensorDtype::Fp8E4m3,
    };
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    let mut total_tokens = query.len();
    if total_tokens > max_batch_tokens {
        return Err(RerankError::RequestTooLarge);
    }
    let mut declared_tensor_bytes = tensor_bytes(query_rows, dimension, dtype)?;
    if declared_tensor_bytes > max_batch_bytes {
        return Err(RerankError::RequestTooLarge);
    }

    let mut writer = BoundedWriter::new(max_batch_bytes);
    let quantized = matches!(
        scoring_profile,
        PostgresMaxsimScoringProfile::Pq | PostgresMaxsimScoringProfile::OpqRpq
    );
    if quantized
        && !matches!(quantization_contract, Some(value) if value.starts_with("qtc1-") && value.len() == 69 && value[5..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(RerankError::Configuration(
            "PQ/OPQ/RPQ requires a canonical qtc1 quantization contract",
        ));
    }
    let profiled = logical_top_k.is_some()
        || scheduling.is_some()
        || scoring_profile != PostgresMaxsimScoringProfile::ExactFp16;
    let tenant = scheduling.map_or("__default__", |value| value.tenant.as_str());
    let priority = scheduling.map_or(0, |value| value.priority);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    let candidate_count_offset = writer.len();
    writer.u32(0)?;
    writer.u8(dtype as u8)?;
    writer.u8(1)?; // sum_query_max_document_dot
    if profiled {
        writer.u8(scoring_profile_code(scoring_profile))?;
        writer.u8(if storage_dtype == dtype {
            0
        } else {
            storage_dtype as u8
        })?;
    } else {
        writer.u16(0)?;
    }
    writer
        .u32(u32::try_from(model_contract_id.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    if quantized || logical_top_k.is_some() {
        writer.u32(if quantized { 69 } else { 0 })?;
    }
    let version = if profiled || logical_top_k.is_some() {
        writer.i32(priority)?;
        writer.u32(
            u32::try_from(timeout.as_millis().clamp(1, 600_000))
                .map_err(|_| RerankError::RequestTooLarge)?,
        )?;
        writer.u32(u32::try_from(tenant.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
        if let Some(top_k) = logical_top_k {
            writer.u32(u32::try_from(top_k).map_err(|_| RerankError::RequestTooLarge)?)?;
            if storage_dtype == dtype {
                COMPACT_LOGICAL_EXTERNAL_VERSION
            } else {
                TYPED_COMPACT_LOGICAL_EXTERNAL_VERSION
            }
        } else if quantized {
            QUANTIZED_EXTERNAL_VERSION
        } else {
            PROFILED_EXTERNAL_VERSION
        }
    } else {
        EXTERNAL_VERSION
    };
    writer.bytes(model_contract_id.as_bytes())?;
    if let Some(contract) = quantization_contract.filter(|_| quantized) {
        writer.bytes(contract.as_bytes())?;
    }
    if profiled {
        writer.bytes(tenant.as_bytes())?;
    }
    encode_tensor_values(&mut writer, query, dtype)?;
    let descriptor_offset = writer.len();

    if descriptors.len() > MAX_EXTERNAL_CANDIDATES_PER_BATCH {
        return Err(RerankError::RequestTooLarge);
    }
    let mut heap_keys = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        validate_external_for_request(
            descriptor,
            dimension,
            external_dtype,
            MAX_TENSOR_REF_BYTES,
            MAX_CHECKSUM_BYTES,
        )?;
        total_tokens = total_tokens
            .checked_add(descriptor.rows as usize)
            .ok_or(RerankError::RequestTooLarge)?;
        if logical_top_k.is_none() && total_tokens > max_batch_tokens {
            return Err(RerankError::RequestTooLarge);
        }
        declared_tensor_bytes = declared_tensor_bytes
            .checked_add(tensor_bytes(descriptor.rows, dimension, storage_dtype)?)
            .ok_or(RerankError::RequestTooLarge)?;
        if logical_top_k.is_none() && declared_tensor_bytes > max_batch_bytes {
            return Err(RerankError::RequestTooLarge);
        }

        let candidate_id =
            u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
        writer.u32(candidate_id)?;
        writer.u32(descriptor.rows)?;
        if logical_top_k.is_some() {
            let digest = descriptor.tensor_ref.strip_prefix("sha256://").ok_or(
                RerankError::InvalidDescriptor("unsupported tensor reference"),
            )?;
            let bytes = decode_lower_hex_sha256(digest.as_bytes())
                .ok_or(RerankError::InvalidDescriptor("invalid tensor digest"))?;
            writer.bytes(&bytes)?;
            heap_keys.push(descriptor.candidate.heap_key);
            continue;
        }
        writer.u32(
            u32::try_from(descriptor.tensor_ref.len()).map_err(|_| RerankError::RequestTooLarge)?,
        )?;
        writer.u32(
            u32::try_from(descriptor.checksum.len()).map_err(|_| RerankError::RequestTooLarge)?,
        )?;
        writer.bytes(descriptor.tensor_ref.as_bytes())?;
        writer.bytes(descriptor.checksum.as_bytes())?;
        heap_keys.push(descriptor.candidate.heap_key);
    }

    let candidate_count =
        u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_u32(candidate_count_offset, candidate_count);
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, version);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(
        16,
        u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?,
    );
    Ok(EncodedRequest {
        frame: writer.finish(),
        heap_keys,
        version,
        descriptor_offset,
    })
}

fn decode_lower_hex_sha256(value: &[u8]) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut decoded = [0_u8; 32];
    for (output, pair) in decoded.iter_mut().zip(value.chunks_exact(2)) {
        let high = match pair[0] {
            b'0'..=b'9' => pair[0] - b'0',
            b'a'..=b'f' => pair[0] - b'a' + 10,
            _ => return None,
        };
        let low = match pair[1] {
            b'0'..=b'9' => pair[1] - b'0',
            b'a'..=b'f' => pair[1] - b'a' + 10,
            _ => return None,
        };
        *output = high << 4 | low;
    }
    Some(decoded)
}

fn scoring_profile_code(profile: PostgresMaxsimScoringProfile) -> u8 {
    match profile {
        PostgresMaxsimScoringProfile::ExactFp16 => 1,
        PostgresMaxsimScoringProfile::Int8 => 2,
        PostgresMaxsimScoringProfile::Fp8E4m3 => 3,
        PostgresMaxsimScoringProfile::RawFp8E4m3 => 6,
        PostgresMaxsimScoringProfile::Pq => 4,
        PostgresMaxsimScoringProfile::OpqRpq => 5,
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_external_batch<S: CandidateTensorDescriptorSource>(
    model_contract_id: &str,
    query: &[OwnedVector],
    candidates: &mut dyn Iterator<Item = PageCandidate>,
    source: &mut S,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
    scheduling: Option<&TileMaxsimScheduling>,
    scoring_profile: PostgresMaxsimScoringProfile,
    quantization_contract: Option<&str>,
    pending: &mut Option<ExternalTensorDescriptor>,
) -> Result<Vec<ExternalTensorDescriptor>, RerankError> {
    validate_external_request_identity(model_contract_id, scheduling)?;
    let (dtype, dimension) = tensor_metadata(query)?;
    let external_dtype = match dtype {
        TensorDtype::F32 => ExternalTensorDtype::F32,
        TensorDtype::F16 => ExternalTensorDtype::F16,
        TensorDtype::Fp8E4m3 => unreachable!("query tensors originate as vector or halfvec"),
    };
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    let query_bytes = tensor_bytes(query_rows, dimension, dtype)?;
    let mut total_tokens = query.len();
    let mut declared_tensor_bytes = query_bytes;
    let mut frame_bytes = external_request_base_bytes(
        model_contract_id,
        scheduling,
        scoring_profile,
        quantization_contract,
        query_bytes,
    )?;
    if total_tokens > max_batch_tokens
        || declared_tensor_bytes > max_batch_bytes
        || frame_bytes > max_batch_bytes
    {
        return Err(RerankError::RequestTooLarge);
    }

    let mut batch = Vec::new();
    while batch.len() < MAX_EXTERNAL_CANDIDATES_PER_BATCH {
        let descriptor = if let Some(descriptor) = pending.take() {
            descriptor
        } else {
            let mut fetched = None;
            for candidate in &mut *candidates {
                if let Some(descriptor) = source.fetch(candidate)? {
                    fetched = Some(descriptor);
                    break;
                }
            }
            let Some(descriptor) = fetched else {
                break;
            };
            descriptor
        };

        validate_external_for_request(
            &descriptor,
            dimension,
            external_dtype,
            MAX_TENSOR_REF_BYTES,
            MAX_CHECKSUM_BYTES,
        )?;
        let next_tokens = total_tokens
            .checked_add(descriptor.rows as usize)
            .ok_or(RerankError::RequestTooLarge)?;
        let next_declared_bytes = declared_tensor_bytes
            .checked_add(tensor_bytes(descriptor.rows, dimension, dtype)?)
            .ok_or(RerankError::RequestTooLarge)?;
        let descriptor_frame_bytes = 16usize
            .checked_add(descriptor.tensor_ref.len())
            .and_then(|size| size.checked_add(descriptor.checksum.len()))
            .ok_or(RerankError::RequestTooLarge)?;
        let next_frame_bytes = frame_bytes
            .checked_add(descriptor_frame_bytes)
            .ok_or(RerankError::RequestTooLarge)?;
        if next_tokens > max_batch_tokens
            || next_declared_bytes > max_batch_bytes
            || next_frame_bytes > max_batch_bytes
        {
            if batch.is_empty() {
                return Err(RerankError::RequestTooLarge);
            }
            *pending = Some(descriptor);
            break;
        }

        total_tokens = next_tokens;
        declared_tensor_bytes = next_declared_bytes;
        frame_bytes = next_frame_bytes;
        batch.push(descriptor);
    }
    Ok(batch)
}

fn validate_external_request_identity(
    model_contract_id: &str,
    scheduling: Option<&TileMaxsimScheduling>,
) -> Result<(), RerankError> {
    if model_contract_id.is_empty()
        || model_contract_id.len() > MAX_MODEL_CONTRACT_BYTES
        || model_contract_id.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    if let Some(scheduling) = scheduling {
        if scheduling.tenant.is_empty()
            || scheduling.tenant.len() > MAX_TENANT_BYTES
            || scheduling.tenant.chars().any(char::is_control)
            || !(-100..=100).contains(&scheduling.priority)
        {
            return Err(RerankError::Configuration(
                "TileMaxSim scheduler tenant or priority is invalid",
            ));
        }
    }
    Ok(())
}

fn external_request_base_bytes(
    model_contract_id: &str,
    scheduling: Option<&TileMaxsimScheduling>,
    scoring_profile: PostgresMaxsimScoringProfile,
    quantization_contract: Option<&str>,
    query_bytes: usize,
) -> Result<usize, RerankError> {
    let quantized = matches!(
        scoring_profile,
        PostgresMaxsimScoringProfile::Pq | PostgresMaxsimScoringProfile::OpqRpq
    );
    let fixed = if scheduling.is_some() || quantized {
        56usize
    } else {
        44usize
    };
    fixed
        .checked_add(model_contract_id.len())
        .and_then(|size| {
            size.checked_add(if quantized {
                4 + quantization_contract.map_or(0, str::len)
            } else {
                0
            })
        })
        .and_then(|size| {
            size.checked_add(scheduling.map_or(0, |scheduling| scheduling.tenant.len()))
        })
        .and_then(|size| size.checked_add(query_bytes))
        .ok_or(RerankError::RequestTooLarge)
}

fn tensor_bytes(rows: u32, dimension: u32, dtype: TensorDtype) -> Result<usize, RerankError> {
    let scalar_bytes = match dtype {
        TensorDtype::F32 => 4usize,
        TensorDtype::F16 => 2usize,
        TensorDtype::Fp8E4m3 => 1usize,
    };
    usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(dimension as usize))
        .and_then(|elements| elements.checked_mul(scalar_bytes))
        .ok_or(RerankError::RequestTooLarge)
}

fn validate_external_for_request(
    descriptor: &ExternalTensorDescriptor,
    dimension: u32,
    dtype: ExternalTensorDtype,
    max_tensor_ref_bytes: usize,
    max_checksum_bytes: usize,
) -> Result<(), RerankError> {
    if descriptor.rows == 0
        || descriptor.rows > 65_536
        || dimension > 60_000
        || descriptor.dimension != dimension
        || descriptor.dtype != dtype
    {
        return Err(RerankError::TensorMismatch);
    }
    if descriptor.tensor_ref.is_empty()
        || descriptor.tensor_ref.len() > max_tensor_ref_bytes
        || descriptor.tensor_ref.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "tensor reference is empty, oversized, or contains control characters",
        ));
    }
    if descriptor.checksum.len() > max_checksum_bytes
        || !descriptor
            .checksum
            .strip_prefix("sha256:")
            .is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            })
    {
        return Err(RerankError::InvalidDescriptor(
            "tensor checksum must be a lowercase sha256 digest",
        ));
    }
    Ok(())
}

fn encode_request<S: CandidateTensorSource>(
    request_id: u64,
    query: &[OwnedVector],
    candidates: &mut dyn Iterator<Item = PageCandidate>,
    source: &mut S,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
) -> Result<EncodedRequest, RerankError> {
    let (dtype, dimension) = tensor_metadata(query)?;
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    let mut total_tokens = query.len();
    if total_tokens > max_batch_tokens {
        return Err(RerankError::RequestTooLarge);
    }

    let mut writer = BoundedWriter::new(max_batch_bytes);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    let candidate_count_offset = writer.len();
    writer.u32(0)?;
    writer.u8(dtype as u8)?;
    writer.u8(1)?; // sum_query_max_document_dot
    writer.u16(0)?;
    encode_tensor_values(&mut writer, query, dtype)?;

    let mut heap_keys = Vec::new();
    for candidate in candidates {
        let Some(tensor) = source.fetch(candidate)? else {
            continue;
        };
        let (candidate_dtype, candidate_dimension) = tensor_metadata(&tensor.vectors)?;
        if candidate_dtype != dtype || candidate_dimension != dimension {
            return Err(RerankError::TensorMismatch);
        }
        total_tokens = total_tokens
            .checked_add(tensor.vectors.len())
            .ok_or(RerankError::RequestTooLarge)?;
        if total_tokens > max_batch_tokens {
            return Err(RerankError::RequestTooLarge);
        }
        let candidate_id =
            u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
        let rows = u32::try_from(tensor.vectors.len()).map_err(|_| RerankError::RequestTooLarge)?;
        writer.u32(candidate_id)?;
        writer.u32(rows)?;
        encode_tensor_values(&mut writer, &tensor.vectors, dtype)?;
        heap_keys.push(tensor.candidate.heap_key);
    }
    let candidate_count =
        u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_u32(candidate_count_offset, candidate_count);
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    let body_len = u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, VERSION);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(16, body_len);
    Ok(EncodedRequest {
        frame: writer.finish(),
        heap_keys,
        version: VERSION,
        descriptor_offset: 0,
    })
}

fn tensor_metadata(vectors: &[OwnedVector]) -> Result<(TensorDtype, u32), RerankError> {
    let Some(first) = vectors.first() else {
        return Err(RerankError::TensorMismatch);
    };
    let dtype = match first {
        OwnedVector::Vecf32(_) => TensorDtype::F32,
        OwnedVector::Vecf16(_) => TensorDtype::F16,
        OwnedVector::Rabitq8(_) | OwnedVector::Rabitq4(_) => {
            return Err(RerankError::UnsupportedTensorKind);
        }
    };
    let dimension = first.dim();
    for vector in vectors {
        let this_dtype = match vector {
            OwnedVector::Vecf32(_) => TensorDtype::F32,
            OwnedVector::Vecf16(_) => TensorDtype::F16,
            OwnedVector::Rabitq8(_) | OwnedVector::Rabitq4(_) => {
                return Err(RerankError::UnsupportedTensorKind);
            }
        };
        if this_dtype != dtype || vector.dim() != dimension {
            return Err(RerankError::TensorMismatch);
        }
    }
    Ok((dtype, dimension))
}

fn encode_tensor_values(
    writer: &mut BoundedWriter,
    vectors: &[OwnedVector],
    dtype: TensorDtype,
) -> Result<(), RerankError> {
    for vector in vectors {
        match (dtype, vector) {
            (TensorDtype::F32, OwnedVector::Vecf32(vector)) => {
                for value in vector.slice() {
                    writer.bytes(&value.to_le_bytes())?;
                }
            }
            (TensorDtype::F16, OwnedVector::Vecf16(vector)) => {
                for value in vector.slice() {
                    writer.u16(value.to_bits())?;
                }
            }
            (TensorDtype::Fp8E4m3, OwnedVector::Vecf32(vector)) => {
                for value in vector.slice() {
                    writer.u8(f32_to_e4m3fn(*value))?;
                }
            }
            (TensorDtype::Fp8E4m3, OwnedVector::Vecf16(vector)) => {
                for value in vector.slice() {
                    writer.u8(f32_to_e4m3fn(value.to_f32()))?;
                }
            }
            _ => return Err(RerankError::TensorMismatch),
        }
    }
    Ok(())
}

fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let fraction = bits & 0x07;
    if exponent == 0 {
        sign * fraction as f32 * 2.0_f32.powi(-9)
    } else if exponent == 0x0f && fraction == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + fraction as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    }
}

fn f32_to_e4m3fn(value: f32) -> u8 {
    let negative = value.is_sign_negative();
    let magnitude = value.abs().min(448.0);
    let mut best = (f32::INFINITY, 0_u8);
    for bits in 0_u8..=0x7e {
        let candidate = e4m3fn_to_f32(bits);
        let distance = (candidate - magnitude).abs();
        if distance < best.0 {
            best = (distance, bits);
        }
    }
    best.1 | if negative { 0x80 } else { 0 }
}

fn decode_response(
    frame: &[u8],
    request_id: u64,
    heap_keys: &[HeapKey],
) -> Result<RerankResults, RerankError> {
    decode_response_for_version(frame, VERSION, request_id, heap_keys, heap_keys.len())
}

fn decode_response_for_version(
    frame: &[u8],
    expected_version: u16,
    request_id: u64,
    heap_keys: &[HeapKey],
    expected_result_count: usize,
) -> Result<RerankResults, RerankError> {
    let mut cursor = Cursor::new(frame);
    if cursor.bytes(4)? != MAGIC {
        return Err(RerankError::Protocol("invalid magic".into()));
    }
    if cursor.u16()? != expected_version {
        return Err(RerankError::Protocol("unsupported version".into()));
    }
    if cursor.u16()? != RESPONSE_KIND {
        return Err(RerankError::Protocol("unexpected message kind".into()));
    }
    if cursor.u64()? != request_id {
        return Err(RerankError::Protocol("request ID mismatch".into()));
    }
    let body_len = usize::try_from(cursor.u64()?)
        .map_err(|_| RerankError::Protocol("response body is too large".into()))?;
    if body_len != frame.len().saturating_sub(HEADER_LEN) {
        return Err(RerankError::Protocol("response length mismatch".into()));
    }
    let status = cursor.u32()?;
    if status != 0 {
        let length = usize::try_from(cursor.u32()?)
            .map_err(|_| RerankError::Protocol("remote error is too large".into()))?;
        if length > MAX_REMOTE_ERROR_BYTES {
            return Err(RerankError::Protocol("remote error is too large".into()));
        }
        let message = std::str::from_utf8(cursor.bytes(length)?)
            .map_err(|_| RerankError::Protocol("remote error is not UTF-8".into()))?;
        cursor.finish()?;
        return Err(RerankError::Remote(message.into()));
    }
    let result_count = usize::try_from(cursor.u32()?)
        .map_err(|_| RerankError::Protocol("result count is too large".into()))?;
    if result_count != expected_result_count || expected_result_count > heap_keys.len() {
        return Err(RerankError::Protocol("partial result set".into()));
    }
    let mut seen = vec![false; heap_keys.len()];
    let mut results = BinaryHeap::new();
    for _ in 0..result_count {
        let candidate_id = usize::try_from(cursor.u32()?)
            .map_err(|_| RerankError::Protocol("candidate ID is too large".into()))?;
        let Some(heap_key) = heap_keys.get(candidate_id).copied() else {
            return Err(RerankError::Protocol("unknown candidate ID".into()));
        };
        if std::mem::replace(&mut seen[candidate_id], true) {
            return Err(RerankError::Protocol("duplicate candidate ID".into()));
        }
        let similarity = f32::from_bits(cursor.u32()?);
        if !similarity.is_finite() {
            return Err(RerankError::Protocol("non-finite similarity".into()));
        }
        let distance = Distance::from_f32(-similarity);
        results.push((Reverse(distance), Reverse(heap_key)));
    }
    cursor.finish()?;
    Ok(RerankResults { inner: results })
}

fn decode_scoped_response(
    frame: &[u8],
    request_id: u64,
    heap_keys: &[HeapKey],
    expected_global: usize,
    expected_scoped: usize,
) -> Result<(RerankResults, RerankResults), RerankError> {
    const SCOPED_RESULT_TAG: u32 = 1 << 31;
    let mut cursor = Cursor::new(frame);
    if cursor.bytes(4)? != MAGIC
        || cursor.u16()? != SCOPED_CATALOG_SELECTION_REFERENCE_VERSION
        || cursor.u16()? != RESPONSE_KIND
    {
        return Err(RerankError::Protocol("unsupported version".into()));
    }
    if cursor.u64()? != request_id {
        return Err(RerankError::Protocol("request ID mismatch".into()));
    }
    let body_len = usize::try_from(cursor.u64()?)
        .map_err(|_| RerankError::Protocol("response body is too large".into()))?;
    if body_len != frame.len().saturating_sub(HEADER_LEN) {
        return Err(RerankError::Protocol("response length mismatch".into()));
    }
    let status = cursor.u32()?;
    if status != 0 {
        let length = usize::try_from(cursor.u32()?)
            .map_err(|_| RerankError::Protocol("remote error is too large".into()))?;
        if length > MAX_REMOTE_ERROR_BYTES {
            return Err(RerankError::Protocol("remote error is too large".into()));
        }
        let message = std::str::from_utf8(cursor.bytes(length)?)
            .map_err(|_| RerankError::Protocol("remote error is not UTF-8".into()))?;
        cursor.finish()?;
        return Err(RerankError::Remote(message.into()));
    }
    let result_count = usize::try_from(cursor.u32()?)
        .map_err(|_| RerankError::Protocol("result count is too large".into()))?;
    if result_count != expected_global.saturating_add(expected_scoped) {
        return Err(RerankError::Protocol("partial scoped result set".into()));
    }
    let mut seen_global = vec![false; heap_keys.len()];
    let mut seen_scoped = vec![false; heap_keys.len()];
    let mut global = BinaryHeap::new();
    let mut scoped = BinaryHeap::new();
    for _ in 0..result_count {
        let tagged_id = cursor.u32()?;
        let is_scoped = tagged_id & SCOPED_RESULT_TAG != 0;
        let candidate_id = usize::try_from(tagged_id & !SCOPED_RESULT_TAG)
            .map_err(|_| RerankError::Protocol("candidate ID is too large".into()))?;
        let Some(heap_key) = heap_keys.get(candidate_id).copied() else {
            return Err(RerankError::Protocol("unknown candidate ID".into()));
        };
        let seen = if is_scoped {
            &mut seen_scoped
        } else {
            &mut seen_global
        };
        if std::mem::replace(&mut seen[candidate_id], true) {
            return Err(RerankError::Protocol("duplicate candidate ID".into()));
        }
        let similarity = f32::from_bits(cursor.u32()?);
        if !similarity.is_finite() {
            return Err(RerankError::Protocol("non-finite similarity".into()));
        }
        let entry = (Reverse(Distance::from_f32(-similarity)), Reverse(heap_key));
        if is_scoped {
            scoped.push(entry);
        } else {
            global.push(entry);
        }
    }
    cursor.finish()?;
    if global.len() != expected_global || scoped.len() != expected_scoped {
        return Err(RerankError::Protocol("mis-tagged scoped result set".into()));
    }
    Ok((
        RerankResults { inner: global },
        RerankResults { inner: scoped },
    ))
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn ensure(&self, additional: usize) -> Result<(), RerankError> {
        let size = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or(RerankError::RequestTooLarge)?;
        if size > self.limit {
            return Err(RerankError::RequestTooLarge);
        }
        Ok(())
    }

    fn zeros(&mut self, count: usize) -> Result<(), RerankError> {
        self.ensure(count)?;
        self.bytes.resize(self.bytes.len() + count, 0);
        Ok(())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), RerankError> {
        self.ensure(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), RerankError> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), RerankError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), RerankError> {
        self.bytes(&value.to_le_bytes())
    }

    fn i32(&mut self, value: i32) -> Result<(), RerankError> {
        self.bytes(&value.to_le_bytes())
    }

    fn patch_bytes(&mut self, offset: usize, bytes: &[u8]) {
        self.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    fn patch_u16(&mut self, offset: usize, value: u16) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn patch_u32(&mut self, offset: usize, value: u32) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn patch_u64(&mut self, offset: usize, value: u64) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], RerankError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| RerankError::Protocol("message offset overflow".into()))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| RerankError::Protocol("truncated message".into()))?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, RerankError> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, RerankError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, RerankError> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn finish(self) -> Result<(), RerankError> {
        if self.offset != self.bytes.len() {
            return Err(RerankError::Protocol("trailing response bytes".into()));
        }
        Ok(())
    }
}

pub(super) struct UnixSocketTransport {
    endpoint: String,
}

impl UnixSocketTransport {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
}

#[cfg(unix)]
impl TileMaxsimTransport for UnixSocketTransport {
    fn round_trip(
        &mut self,
        request: &[u8],
        timeout: Duration,
        max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError> {
        if self.endpoint.is_empty() {
            return Err(RerankError::Transport("endpoint is empty".into()));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| RerankError::Transport("invalid timeout".into()))?;
        let request_version = request
            .get(4..6)
            .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()));
        let persistent_version = match request_version {
            Some(CATALOG_SELECTION_REFERENCE_VERSION) => {
                Some(PERSISTENT_CATALOG_SELECTION_REFERENCE_VERSION)
            }
            Some(SCOPED_CATALOG_SELECTION_REFERENCE_VERSION) => {
                Some(PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE_VERSION)
            }
            _ => None,
        };
        if let Some(persistent_version) = persistent_version
            && persistent_capability(&self.endpoint, persistent_version) != Some(false)
        {
            let mut persistent_request = request.to_vec();
            persistent_request[4..6].copy_from_slice(&persistent_version.to_le_bytes());
            let (mut stream, reused) = take_persistent_transport(&self.endpoint)
                .map(|stream| (stream, true))
                .unwrap_or((
                    connect_endpoint_interruptible(&self.endpoint, deadline)?,
                    false,
                ));
            let response = match exchange_frame(
                &mut stream,
                &persistent_request,
                deadline,
                max_response_bytes,
            ) {
                Ok(response) => response,
                Err(_) if reused => {
                    stream = connect_endpoint_interruptible(&self.endpoint, deadline)?;
                    exchange_frame(
                        &mut stream,
                        &persistent_request,
                        deadline,
                        max_response_bytes,
                    )?
                }
                Err(error) => return Err(error),
            };
            if response.get(4..6) == Some(&persistent_version.to_le_bytes()) {
                let valid_envelope = response_matches_request(&response, &persistent_request);
                remember_persistent_capability(&self.endpoint, persistent_version, true);
                if valid_envelope && response_status(&response) == Some(0) {
                    return_persistent_transport(&self.endpoint, stream);
                }
                let mut compatible = response;
                compatible[4..6].copy_from_slice(&request_version.unwrap().to_le_bytes());
                return Ok(compatible);
            }
            // An older daemon responds with its oldest error protocol. Record
            // this negotiated version once and retry the idempotent scoring
            // request in its original one-shot form on a fresh connection.
            remember_persistent_capability(&self.endpoint, persistent_version, false);
        }

        let mut stream = connect_endpoint_interruptible(&self.endpoint, deadline)?;
        exchange_frame(&mut stream, request, deadline, max_response_bytes)
    }
}

#[cfg(unix)]
fn exchange_frame(
    stream: &mut TransportStream,
    request: &[u8],
    deadline: Instant,
    max_response_bytes: usize,
) -> Result<Vec<u8>, RerankError> {
    let poll = remaining_until(deadline)?.min(Duration::from_millis(50));
    stream
        .set_read_timeout(Some(poll))
        .map_err(|error| RerankError::Transport(error.to_string()))?;
    stream
        .set_write_timeout(Some(poll))
        .map_err(|error| RerankError::Transport(error.to_string()))?;
    write_interruptible(stream, request, deadline)?;
    let mut header = [0u8; HEADER_LEN];
    read_interruptible(stream, &mut header, deadline)?;
    let body_len = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
        .map_err(|_| RerankError::Protocol("response body is too large".into()))?;
    let response_len = HEADER_LEN
        .checked_add(body_len)
        .ok_or_else(|| RerankError::Protocol("response length overflow".into()))?;
    if response_len > max_response_bytes {
        return Err(RerankError::Protocol(
            "response exceeds configured limit".into(),
        ));
    }
    let mut response = Vec::with_capacity(response_len);
    response.extend_from_slice(&header);
    response.resize(response_len, 0);
    read_interruptible(stream, &mut response[HEADER_LEN..], deadline)?;
    Ok(response)
}

#[cfg(unix)]
fn response_status(response: &[u8]) -> Option<u32> {
    response
        .get(HEADER_LEN..HEADER_LEN + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
}

#[cfg(unix)]
fn response_matches_request(response: &[u8], request: &[u8]) -> bool {
    response.get(..4) == Some(MAGIC)
        && response.get(6..8) == Some(&RESPONSE_KIND.to_le_bytes())
        && response.get(8..16) == request.get(8..16)
}

#[cfg(unix)]
fn persistent_capability(endpoint: &str, version: u16) -> Option<bool> {
    PERSISTENT_CAPABILITIES
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .ok()
        .and_then(|capabilities| {
            capabilities
                .iter()
                .find(|((candidate, candidate_version), _)| {
                    candidate == endpoint && *candidate_version == version
                })
                .map(|(_, supported)| *supported)
        })
}

#[cfg(unix)]
fn remember_persistent_capability(endpoint: &str, version: u16, supported: bool) {
    let Ok(mut capabilities) = PERSISTENT_CAPABILITIES
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
    else {
        return;
    };
    capabilities.retain(|((candidate, candidate_version), _)| {
        candidate != endpoint || *candidate_version != version
    });
    while capabilities.len() >= MAX_PERSISTENT_TRANSPORT_ENDPOINTS {
        capabilities.pop_front();
    }
    capabilities.push_back(((endpoint.to_owned(), version), supported));
}

#[cfg(unix)]
fn take_persistent_transport(endpoint: &str) -> Option<TransportStream> {
    let mut transports = PERSISTENT_TRANSPORTS
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .ok()?;
    let index = transports
        .iter()
        .position(|(candidate, _)| candidate == endpoint)?;
    transports.remove(index).map(|(_, stream)| stream)
}

#[cfg(unix)]
fn return_persistent_transport(endpoint: &str, stream: TransportStream) {
    let Ok(mut transports) = PERSISTENT_TRANSPORTS
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
    else {
        return;
    };
    transports.retain(|(candidate, _)| candidate != endpoint);
    while transports.len() >= MAX_PERSISTENT_TRANSPORT_ENDPOINTS {
        transports.pop_front();
    }
    transports.push_back((endpoint.to_owned(), stream));
}

#[cfg(unix)]
enum TransportStream {
    Unix(std::os::unix::net::UnixStream),
    Tcp(TcpStream),
}

#[cfg(unix)]
impl TransportStream {
    fn set_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        match self {
            Self::Unix(stream) => stream.set_read_timeout(timeout),
            Self::Tcp(stream) => stream.set_read_timeout(timeout),
        }
    }

    fn set_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        match self {
            Self::Unix(stream) => stream.set_write_timeout(timeout),
            Self::Tcp(stream) => stream.set_write_timeout(timeout),
        }
    }
}

#[cfg(unix)]
impl Read for TransportStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.read(buffer),
            Self::Tcp(stream) => stream.read(buffer),
        }
    }
}

#[cfg(unix)]
impl Write for TransportStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Unix(stream) => stream.write(buffer),
            Self::Tcp(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Unix(stream) => stream.flush(),
            Self::Tcp(stream) => stream.flush(),
        }
    }
}

#[cfg(unix)]
fn connect_endpoint_interruptible(
    endpoint: &str,
    deadline: Instant,
) -> Result<TransportStream, RerankError> {
    let Some(authority) = endpoint.strip_prefix("tcp://") else {
        return connect_interruptible(endpoint, deadline).map(TransportStream::Unix);
    };
    if authority.is_empty()
        || authority.contains('/')
        || authority.contains('@')
        || !authority.contains(':')
    {
        return Err(RerankError::Transport(
            "TCP endpoint must be tcp://HOST:PORT without credentials or a path".into(),
        ));
    }
    let remaining = remaining_until(deadline)?;
    let addresses = authority
        .to_socket_addrs()
        .map_err(|error| RerankError::Transport(error.to_string()))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(RerankError::Transport(
            "TCP endpoint resolved to no addresses".into(),
        ));
    }
    let mut last_error = None;
    for address in addresses {
        pgrx::check_for_interrupts!();
        let attempt = remaining_until(deadline)?.min(remaining);
        match TcpStream::connect_timeout(&address, attempt) {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                return Ok(TransportStream::Tcp(stream));
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(RerankError::Transport(
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "TCP connection failed".to_owned()),
    ))
}

#[cfg(unix)]
fn connect_interruptible(
    endpoint: &str,
    deadline: Instant,
) -> Result<std::os::unix::net::UnixStream, RerankError> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let (address, address_len) = unix_socket_address(endpoint)?;
    let raw_fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if raw_fd < 0 {
        return Err(last_transport_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFD,
        libc::F_SETFD,
        libc::FD_CLOEXEC,
        true,
    )?;
    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFL,
        libc::F_SETFL,
        libc::O_NONBLOCK,
        true,
    )?;

    let connected = unsafe {
        libc::connect(
            fd.as_raw_fd(),
            (&raw const address).cast::<libc::sockaddr>(),
            address_len,
        )
    } == 0;
    if !connected {
        let error = std::io::Error::last_os_error();
        let raw_error = error.raw_os_error();
        if raw_error != Some(libc::EINPROGRESS)
            && raw_error != Some(libc::EAGAIN)
            && raw_error != Some(libc::EWOULDBLOCK)
        {
            return Err(RerankError::Transport(error.to_string()));
        }
        wait_for_connect(fd.as_raw_fd(), deadline)?;
    }

    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFL,
        libc::F_SETFL,
        libc::O_NONBLOCK,
        false,
    )?;
    Ok(std::os::unix::net::UnixStream::from(fd))
}

#[cfg(unix)]
fn unix_socket_address(
    endpoint: &str,
) -> Result<(libc::sockaddr_un, libc::socklen_t), RerankError> {
    let path = endpoint.as_bytes();
    let mut address = unsafe { std::mem::zeroed::<libc::sockaddr_un>() };
    if path.contains(&0) {
        return Err(RerankError::Transport(
            "endpoint contains a NUL byte".into(),
        ));
    }
    if path.len() >= address.sun_path.len() {
        return Err(RerankError::Transport("endpoint path is too long".into()));
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    unsafe {
        std::ptr::copy_nonoverlapping(
            path.as_ptr().cast::<libc::c_char>(),
            address.sun_path.as_mut_ptr(),
            path.len(),
        );
    }
    let length = std::mem::offset_of!(libc::sockaddr_un, sun_path)
        .checked_add(path.len())
        .and_then(|length| length.checked_add(1))
        .and_then(|length| libc::socklen_t::try_from(length).ok())
        .ok_or_else(|| RerankError::Transport("endpoint path is too long".into()))?;
    #[cfg(any(
        target_os = "aix",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "hurd",
        target_os = "ios",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos"
    ))]
    {
        address.sun_len = u8::try_from(length)
            .map_err(|_| RerankError::Transport("endpoint path is too long".into()))?;
    }
    Ok((address, length))
}

#[cfg(unix)]
fn update_fd_flag(
    fd: std::os::fd::RawFd,
    get_command: libc::c_int,
    set_command: libc::c_int,
    flag: libc::c_int,
    enabled: bool,
) -> Result<(), RerankError> {
    let current = unsafe { libc::fcntl(fd, get_command) };
    if current < 0 {
        return Err(last_transport_error());
    }
    let updated = if enabled {
        current | flag
    } else {
        current & !flag
    };
    if unsafe { libc::fcntl(fd, set_command, updated) } < 0 {
        return Err(last_transport_error());
    }
    Ok(())
}

#[cfg(unix)]
fn wait_for_connect(fd: std::os::fd::RawFd, deadline: Instant) -> Result<(), RerankError> {
    loop {
        pgrx::check_for_interrupts!();
        let remaining = remaining_until(deadline)?;
        let timeout_ms = remaining.min(Duration::from_millis(50)).as_millis().max(1) as libc::c_int;
        let mut poll_fd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut poll_fd, 1, timeout_ms) };
        if result == 0 {
            continue;
        }
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(RerankError::Transport(error.to_string()));
        }
        let mut socket_error = 0;
        let mut socket_error_len = size_of_val(&socket_error) as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                (&raw mut socket_error).cast(),
                &raw mut socket_error_len,
            )
        } < 0
        {
            return Err(last_transport_error());
        }
        if socket_error != 0 {
            return Err(RerankError::Transport(
                std::io::Error::from_raw_os_error(socket_error).to_string(),
            ));
        }
        return Ok(());
    }
}

#[cfg(unix)]
fn remaining_until(deadline: Instant) -> Result<Duration, RerankError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| RerankError::Transport("request timed out".into()))
}

#[cfg(unix)]
fn last_transport_error() -> RerankError {
    RerankError::Transport(std::io::Error::last_os_error().to_string())
}

#[cfg(unix)]
fn write_interruptible(
    stream: &mut TransportStream,
    mut bytes: &[u8],
    deadline: Instant,
) -> Result<(), RerankError> {
    while !bytes.is_empty() {
        match stream.write(bytes) {
            Ok(0) => return Err(RerankError::Transport("connection closed".into())),
            Ok(count) => bytes = &bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(RerankError::Transport(error.to_string())),
        }
        pgrx::check_for_interrupts!();
        if Instant::now() >= deadline {
            return Err(RerankError::Transport("request timed out".into()));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_interruptible(
    stream: &mut TransportStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> Result<(), RerankError> {
    while !bytes.is_empty() {
        match stream.read(bytes) {
            Ok(0) => return Err(RerankError::Transport("connection closed".into())),
            Ok(count) => bytes = &mut bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(RerankError::Transport(error.to_string())),
        }
        pgrx::check_for_interrupts!();
        if Instant::now() >= deadline {
            return Err(RerankError::Transport("request timed out".into()));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
impl TileMaxsimTransport for UnixSocketTransport {
    fn round_trip(
        &mut self,
        _request: &[u8],
        _timeout: Duration,
        _max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError> {
        if self.endpoint.is_empty() {
            return Err(RerankError::Transport("endpoint is empty".into()));
        }
        Err(RerankError::Transport(
            "Unix sockets are not supported on this platform".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::external::{
        CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorDtype,
    };
    use super::super::rerank::CandidateTensor;
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::net::TcpListener;
    use std::rc::Rc;
    use std::thread;
    use vector::vect::VectOwned;

    #[test]
    fn compact_digest_decoder_accepts_only_canonical_lower_hex() {
        assert_eq!(
            decode_lower_hex_sha256("ab".repeat(32).as_bytes()),
            Some([0xab; 32])
        );
        assert_eq!(decode_lower_hex_sha256("AB".repeat(32).as_bytes()), None);
        assert_eq!(decode_lower_hex_sha256(b"abcd"), None);
    }

    #[cfg(unix)]
    fn transport_test_frame(version: u16, request_id: u64) -> Vec<u8> {
        let mut frame = Vec::with_capacity(HEADER_LEN);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&version.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&request_id.to_le_bytes());
        frame.extend_from_slice(&0_u64.to_le_bytes());
        frame
    }

    #[cfg(unix)]
    fn read_transport_test_frame(stream: &mut TcpStream) -> (u16, u64) {
        let mut header = [0_u8; HEADER_LEN];
        stream.read_exact(&mut header).unwrap();
        let body_len =
            usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap())).unwrap();
        let mut body = vec![0_u8; body_len];
        stream.read_exact(&mut body).unwrap();
        (
            u16::from_le_bytes(header[4..6].try_into().unwrap()),
            u64::from_le_bytes(header[8..16].try_into().unwrap()),
        )
    }

    #[cfg(unix)]
    fn write_transport_test_response(stream: &mut TcpStream, version: u16, request_id: u64) {
        let mut response = transport_test_frame(version, request_id);
        response[6..8].copy_from_slice(&RESPONSE_KIND.to_le_bytes());
        response[16..24].copy_from_slice(&8_u64.to_le_bytes());
        response.extend_from_slice(&0_u32.to_le_bytes());
        response.extend_from_slice(&0_u32.to_le_bytes());
        stream.write_all(&response).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn tcp_transport_reuses_a_negotiated_persistent_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut versions = Vec::new();
            for _ in 0..2 {
                let (version, request_id) = read_transport_test_frame(&mut stream);
                versions.push(version);
                write_transport_test_response(&mut stream, version, request_id);
            }
            versions
        });
        let mut transport = UnixSocketTransport::new(endpoint);
        for request_id in [41, 42] {
            let response = transport
                .round_trip(
                    &transport_test_frame(SCOPED_CATALOG_SELECTION_REFERENCE_VERSION, request_id),
                    Duration::from_secs(2),
                    HEADER_LEN + 8,
                )
                .unwrap();
            assert_eq!(
                u16::from_le_bytes(response[4..6].try_into().unwrap()),
                SCOPED_CATALOG_SELECTION_REFERENCE_VERSION
            );
        }
        assert_eq!(
            server.join().unwrap(),
            vec![
                PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE_VERSION,
                PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE_VERSION,
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn tcp_transport_reuses_a_global_catalog_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut versions = Vec::new();
            for _ in 0..2 {
                let (version, request_id) = read_transport_test_frame(&mut stream);
                versions.push(version);
                write_transport_test_response(&mut stream, version, request_id);
            }
            versions
        });
        let mut transport = UnixSocketTransport::new(endpoint);
        for request_id in [51, 52] {
            let response = transport
                .round_trip(
                    &transport_test_frame(CATALOG_SELECTION_REFERENCE_VERSION, request_id),
                    Duration::from_secs(2),
                    HEADER_LEN + 8,
                )
                .unwrap();
            assert_eq!(
                u16::from_le_bytes(response[4..6].try_into().unwrap()),
                CATALOG_SELECTION_REFERENCE_VERSION
            );
        }
        assert_eq!(
            server.join().unwrap(),
            vec![
                PERSISTENT_CATALOG_SELECTION_REFERENCE_VERSION,
                PERSISTENT_CATALOG_SELECTION_REFERENCE_VERSION,
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn tcp_transport_falls_back_once_for_a_legacy_daemon() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let mut versions = Vec::new();
            for response_version in [EXTERNAL_VERSION, SCOPED_CATALOG_SELECTION_REFERENCE_VERSION] {
                let (mut stream, _) = listener.accept().unwrap();
                let (version, request_id) = read_transport_test_frame(&mut stream);
                versions.push(version);
                write_transport_test_response(&mut stream, response_version, request_id);
            }
            versions
        });
        let mut transport = UnixSocketTransport::new(endpoint);
        let response = transport
            .round_trip(
                &transport_test_frame(SCOPED_CATALOG_SELECTION_REFERENCE_VERSION, 43),
                Duration::from_secs(2),
                HEADER_LEN + 8,
            )
            .unwrap();
        assert_eq!(
            u16::from_le_bytes(response[4..6].try_into().unwrap()),
            SCOPED_CATALOG_SELECTION_REFERENCE_VERSION
        );
        assert_eq!(
            server.join().unwrap(),
            vec![
                PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE_VERSION,
                SCOPED_CATALOG_SELECTION_REFERENCE_VERSION,
            ]
        );
    }

    struct MockTensorSource(BTreeMap<HeapKey, Vec<OwnedVector>>);

    impl CandidateTensorSource for MockTensorSource {
        fn fetch(
            &mut self,
            candidate: PageCandidate,
        ) -> Result<Option<CandidateTensor>, RerankError> {
            Ok(Some(CandidateTensor {
                candidate,
                vectors: self
                    .0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            }))
        }
    }

    struct MockDescriptorSource(BTreeMap<HeapKey, ExternalTensorDescriptor>);

    impl CandidateTensorDescriptorSource for MockDescriptorSource {
        fn fetch(
            &mut self,
            candidate: PageCandidate,
        ) -> Result<Option<ExternalTensorDescriptor>, RerankError> {
            Ok(Some(
                self.0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            ))
        }
    }

    struct MockTransport {
        similarities: Vec<(u32, f32)>,
    }

    impl TileMaxsimTransport for MockTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            Ok(success_response_with_version(
                version,
                request_id,
                &self.similarities,
            ))
        }
    }

    #[derive(Default)]
    struct BatchObservations {
        candidate_counts: Vec<u32>,
        transport_timeouts: Vec<Duration>,
        scheduled_timeouts_ms: Vec<u32>,
    }

    struct BatchingTransport {
        observations: Rc<RefCell<BatchObservations>>,
        delay: Duration,
    }

    struct LogicalTransport {
        observations: Rc<RefCell<Vec<(u16, u32, u32)>>>,
    }

    struct ManifestMissTransport {
        versions: Rc<RefCell<Vec<u16>>>,
        legacy_rejection: bool,
    }

    struct CatalogMissTransport {
        calls: Rc<RefCell<Vec<(u16, u8, usize)>>>,
    }

    struct CatalogProbeMissTransport {
        calls: Rc<RefCell<Vec<(u16, u8, usize)>>>,
    }

    fn catalog_frame_mode(request: &[u8]) -> u8 {
        let dimension = u32::from_le_bytes(request[24..28].try_into().unwrap()) as usize;
        let query_rows = u32::from_le_bytes(request[28..32].try_into().unwrap()) as usize;
        let dtype = request[36];
        let scalar_bytes = match dtype {
            1 => 4,
            2 => 2,
            3 => 1,
            _ => unreachable!(),
        };
        let contract_bytes = u32::from_le_bytes(request[40..44].try_into().unwrap()) as usize;
        let tenant_bytes = u32::from_le_bytes(request[56..60].try_into().unwrap()) as usize;
        request[64 + contract_bytes + tenant_bytes + query_rows * dimension * scalar_bytes]
    }

    impl TileMaxsimTransport for CatalogProbeMissTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let mode = catalog_frame_mode(request);
            self.calls.borrow_mut().push((version, mode, request.len()));
            if self.calls.borrow().len() <= 2 {
                Ok(error_response_with_version(
                    version,
                    request_id,
                    "descriptor catalog miss",
                ))
            } else {
                Ok(success_response_with_version(
                    version,
                    request_id,
                    &[(0, 0.75)],
                ))
            }
        }
    }

    impl TileMaxsimTransport for CatalogMissTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let mode = catalog_frame_mode(request);
            self.calls.borrow_mut().push((version, mode, request.len()));
            Ok(error_response_with_version(
                version,
                request_id,
                "descriptor catalog miss",
            ))
        }
    }

    impl TileMaxsimTransport for ManifestMissTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            self.versions.borrow_mut().push(version);
            if version == MANIFEST_LOGICAL_EXTERNAL_VERSION {
                if self.legacy_rejection {
                    return Ok(error_response_with_version(
                        EXTERNAL_VERSION,
                        request_id,
                        "unsupported external protocol version",
                    ));
                }
                return Ok(error_response_with_version(
                    version,
                    request_id,
                    "descriptor manifest miss",
                ));
            }
            Ok(success_response_with_version(
                version,
                request_id,
                &[(0, 0.75)],
            ))
        }
    }

    impl TileMaxsimTransport for LogicalTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let candidate_count = u32::from_le_bytes(request[32..36].try_into().unwrap());
            let top_k = u32::from_le_bytes(request[60..64].try_into().unwrap());
            self.observations
                .borrow_mut()
                .push((version, candidate_count, top_k));
            Ok(success_response_with_version(
                version,
                request_id,
                &[(candidate_count - 1, 0.75)],
            ))
        }
    }

    impl TileMaxsimTransport for BatchingTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            let candidate_count = u32::from_le_bytes(request[32..36].try_into().unwrap());
            let call = {
                let mut observations = self.observations.borrow_mut();
                let call = observations.candidate_counts.len();
                observations.candidate_counts.push(candidate_count);
                observations.transport_timeouts.push(timeout);
                if matches!(
                    version,
                    SCHEDULED_EXTERNAL_VERSION | PROFILED_EXTERNAL_VERSION
                ) {
                    observations
                        .scheduled_timeouts_ms
                        .push(u32::from_le_bytes(request[48..52].try_into().unwrap()));
                }
                call
            };
            std::thread::sleep(self.delay);
            let similarities = (0..candidate_count)
                .map(|candidate_id| (candidate_id, call as f32 * 10.0 + candidate_id as f32))
                .collect::<Vec<_>>();
            Ok(success_response_with_version(
                version,
                request_id,
                &similarities,
            ))
        }
    }

    fn vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf32(VectOwned::new(values.to_vec()))
    }

    fn half_vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf16(VectOwned::new(
            values.iter().copied().map(simd::f16::from_f32).collect(),
        ))
    }

    fn success_response(request_id: u64, similarities: &[(u32, f32)]) -> Vec<u8> {
        success_response_with_version(VERSION, request_id, similarities)
    }

    fn success_response_with_version(
        version: u16,
        request_id: u64,
        similarities: &[(u32, f32)],
    ) -> Vec<u8> {
        let body_len = 8 + similarities.len() * 8;
        let mut response = Vec::with_capacity(HEADER_LEN + body_len);
        response.extend_from_slice(MAGIC);
        response.extend_from_slice(&version.to_le_bytes());
        response.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
        response.extend_from_slice(&request_id.to_le_bytes());
        response.extend_from_slice(&(body_len as u64).to_le_bytes());
        response.extend_from_slice(&0u32.to_le_bytes());
        response.extend_from_slice(&(similarities.len() as u32).to_le_bytes());
        for (candidate_id, similarity) in similarities {
            response.extend_from_slice(&candidate_id.to_le_bytes());
            response.extend_from_slice(&similarity.to_bits().to_le_bytes());
        }
        response
    }

    fn external_descriptor(
        candidate: PageCandidate,
        public_id: i64,
        tensor_ref: &str,
        rows: u32,
        dimension: u32,
        dtype: ExternalTensorDtype,
    ) -> ExternalTensorDescriptor {
        ExternalTensorDescriptor {
            candidate,
            public_id,
            tensor_ref: tensor_ref.into(),
            rows,
            dimension,
            dtype,
            checksum: format!("sha256:{}", "a".repeat(64)),
        }
    }

    fn error_response(request_id: u64, message: &str) -> Vec<u8> {
        error_response_with_version(VERSION, request_id, message)
    }

    fn error_response_with_version(version: u16, request_id: u64, message: &str) -> Vec<u8> {
        let body_len = 8 + message.len();
        let mut response = Vec::with_capacity(HEADER_LEN + body_len);
        response.extend_from_slice(MAGIC);
        response.extend_from_slice(&version.to_le_bytes());
        response.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
        response.extend_from_slice(&request_id.to_le_bytes());
        response.extend_from_slice(&(body_len as u64).to_le_bytes());
        response.extend_from_slice(&1u32.to_le_bytes());
        response.extend_from_slice(&(message.len() as u32).to_le_bytes());
        response.extend_from_slice(message.as_bytes());
        response
    }

    #[test]
    fn gpu_backend_maps_positive_similarity_to_ascending_distance() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![
            PageCandidate {
                approximate_distance: Distance::ZERO,
                heap_key: page_1,
            },
            PageCandidate {
                approximate_distance: Distance::ZERO,
                heap_key: page_2,
            },
        ]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([
            (page_1, vec![vector(&[1.0, 0.0])]),
            (page_2, vec![vector(&[0.5, 0.0])]),
        ]));
        let transport = MockTransport {
            similarities: vec![(0, 1.0), (1, 2.0)],
        };
        let results = GpuTileMaxsimBackend::new(transport, Duration::from_secs(1), 100, 4096)
            .rerank(&query, &mut candidates, &mut source)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(results[0].heap_key, page_2);
        assert_eq!(results[0].distance.to_f32(), -2.0);
        assert_eq!(results[1].heap_key, page_1);
        assert_eq!(results[1].distance.to_f32(), -1.0);
    }

    #[test]
    fn response_rejects_partial_and_duplicate_results() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let partial = success_response(7, &[(0, 1.0)]);
        assert!(matches!(
            decode_response(&partial, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let duplicate = success_response(7, &[(0, 1.0), (0, 2.0)]);
        assert!(matches!(
            decode_response(&duplicate, 7, &keys),
            Err(RerankError::Protocol(_))
        ));
    }

    #[test]
    fn logical_response_accepts_any_requested_top_k_including_all_candidates() {
        let keys = [[0, 0, 1], [0, 0, 2], [0, 0, 3]];
        let subset =
            success_response_with_version(LOGICAL_EXTERNAL_VERSION, 7, &[(2, 3.0), (0, 1.0)]);
        let subset_results =
            decode_response_for_version(&subset, LOGICAL_EXTERNAL_VERSION, 7, &keys, 2)
                .unwrap()
                .collect::<Vec<_>>();
        assert_eq!(subset_results.len(), 2);

        let full = success_response_with_version(
            LOGICAL_EXTERNAL_VERSION,
            8,
            &[(0, 1.0), (1, 2.0), (2, 3.0)],
        );
        let full_results =
            decode_response_for_version(&full, LOGICAL_EXTERNAL_VERSION, 8, &keys, keys.len())
                .unwrap()
                .collect::<Vec<_>>();
        assert_eq!(full_results.len(), keys.len());

        assert!(matches!(
            decode_response_for_version(&subset, LOGICAL_EXTERNAL_VERSION, 7, &keys, 1,),
            Err(RerankError::Protocol(_))
        ));
    }

    #[test]
    fn response_ids_may_arrive_out_of_order() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let response = success_response(7, &[(1, 0.5), (0, 1.0)]);
        let results = decode_response(&response, 7, &keys)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(results[0].heap_key, keys[0]);
        assert_eq!(results[0].distance.to_f32(), -1.0);
        assert_eq!(results[1].heap_key, keys[1]);
        assert_eq!(results[1].distance.to_f32(), -0.5);
    }

    #[test]
    fn response_rejects_unknown_non_finite_and_trailing_results() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let unknown = success_response(7, &[(0, 1.0), (2, 2.0)]);
        assert!(matches!(
            decode_response(&unknown, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let non_finite = success_response(7, &[(0, f32::NAN), (1, 2.0)]);
        assert!(matches!(
            decode_response(&non_finite, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let mut trailing = success_response(7, &[(0, 1.0), (1, 2.0)]);
        trailing.push(0);
        assert!(matches!(
            decode_response(&trailing, 7, &keys),
            Err(RerankError::Protocol(_))
        ));
    }

    #[test]
    fn response_rejects_invalid_header_fields() {
        let keys = [[0, 0, 1]];
        let valid = success_response(7, &[(0, 1.0)]);

        for (offset, replacement) in [(0, 0u8), (4, 2), (6, 1), (8, 8), (16, 0)] {
            let mut invalid = valid.clone();
            invalid[offset] = replacement;
            assert!(matches!(
                decode_response(&invalid, 7, &keys),
                Err(RerankError::Protocol(_))
            ));
        }
    }

    #[test]
    fn response_surfaces_remote_error() {
        let response = error_response(7, "CUDA queue is unavailable");
        assert!(matches!(
            decode_response(&response, 7, &[]),
            Err(RerankError::Remote(message)) if message == "CUDA queue is unavailable"
        ));
    }

    #[test]
    fn request_frame_is_versioned_and_length_prefixed() {
        let page = [0, 0, 1];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([(page, vec![vector(&[0.5, 0.0])])]));
        let encoded = encode_request(9, &query, &mut candidates, &mut source, 100, 4096).unwrap();

        assert_eq!(&encoded.frame[0..4], MAGIC);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[4..6].try_into().unwrap()),
            VERSION
        );
        assert_eq!(
            u16::from_le_bytes(encoded.frame[6..8].try_into().unwrap()),
            REQUEST_KIND
        );
        assert_eq!(
            u64::from_le_bytes(encoded.frame[8..16].try_into().unwrap()),
            9
        );
        assert_eq!(
            u64::from_le_bytes(encoded.frame[16..24].try_into().unwrap()) as usize,
            encoded.frame.len() - HEADER_LEN
        );
        assert_eq!(
            u32::from_le_bytes(encoded.frame[32..36].try_into().unwrap()),
            1
        );
        assert_eq!(encoded.heap_keys, vec![page]);
    }

    #[test]
    fn external_request_encodes_contract_and_opaque_descriptor_ids() {
        let page = [0, 0, 7];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![half_vector(&[1.0, -0.5])];
        let tensor_ref = "s3://immutable/page-9001.tensor";
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(candidate, 9001, tensor_ref, 2, 2, ExternalTensorDtype::F16),
        )]));
        let contract = "colqwen@immutable-revision";
        let encoded = encode_external_request(
            19,
            contract,
            &query,
            &mut candidates,
            &mut source,
            100,
            4096,
            None,
            Duration::from_secs(2),
        )
        .unwrap();

        assert_eq!(&encoded.frame[0..4], MAGIC);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[4..6].try_into().unwrap()),
            EXTERNAL_VERSION
        );
        assert_eq!(
            u32::from_le_bytes(encoded.frame[32..36].try_into().unwrap()),
            1
        );
        let contract_len = u32::from_le_bytes(encoded.frame[40..44].try_into().unwrap()) as usize;
        assert_eq!(&encoded.frame[44..44 + contract_len], contract.as_bytes());
        let candidate_offset = 44 + contract_len + 4; // one 2-D f16 query row
        assert_eq!(
            u32::from_le_bytes(
                encoded.frame[candidate_offset..candidate_offset + 4]
                    .try_into()
                    .unwrap()
            ),
            0
        );
        assert_eq!(
            u32::from_le_bytes(
                encoded.frame[candidate_offset + 4..candidate_offset + 8]
                    .try_into()
                    .unwrap()
            ),
            2
        );
        let reference_len = u32::from_le_bytes(
            encoded.frame[candidate_offset + 8..candidate_offset + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        let reference_offset = candidate_offset + 16;
        assert_eq!(
            &encoded.frame[reference_offset..reference_offset + reference_len],
            tensor_ref.as_bytes()
        );
        assert_eq!(encoded.heap_keys, vec![page]);
    }

    #[test]
    fn pq_request_binds_canonical_contract_in_v5_frame() {
        let page = [0, 0, 8];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let descriptor = external_descriptor(
            candidate,
            8,
            "object://immutable/tensor-8",
            2,
            2,
            ExternalTensorDtype::F16,
        );
        let contract = format!("qtc1-{}", "a".repeat(64));
        let encoded = encode_external_descriptors(
            20,
            "model@1",
            &[half_vector(&[1.0, 0.0])],
            &[descriptor],
            100,
            4096,
            None,
            PostgresMaxsimScoringProfile::Pq,
            Some(&contract),
            Duration::from_secs(2),
            None,
        )
        .unwrap();
        assert_eq!(encoded.version, QUANTIZED_EXTERNAL_VERSION);
        assert_eq!(
            encoded.frame[38],
            scoring_profile_code(PostgresMaxsimScoringProfile::Pq)
        );
        assert!(
            encoded
                .frame
                .windows(contract.len())
                .any(|value| value == contract.as_bytes())
        );

        assert!(matches!(
            encode_external_descriptors(
                21,
                "model@1",
                &[half_vector(&[1.0, 0.0])],
                &[],
                100,
                4096,
                None,
                PostgresMaxsimScoringProfile::Pq,
                None,
                Duration::from_secs(2),
                None,
            ),
            Err(RerankError::Configuration(_))
        ));
    }

    #[test]
    fn scheduled_external_request_encodes_tenant_priority_and_timeout() {
        let page = [0, 0, 8];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![half_vector(&[1.0, -0.5])];
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(
                candidate,
                9002,
                "sha256://opaque",
                2,
                2,
                ExternalTensorDtype::F16,
            ),
        )]));
        let scheduling = TileMaxsimScheduling {
            tenant: "tenant-a".to_owned(),
            priority: 17,
        };
        let encoded = encode_external_request(
            20,
            "contract@1",
            &query,
            &mut candidates,
            &mut source,
            100,
            4096,
            Some(&scheduling),
            Duration::from_millis(4_000),
        )
        .unwrap();

        assert_eq!(encoded.version, PROFILED_EXTERNAL_VERSION);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[4..6].try_into().unwrap()),
            PROFILED_EXTERNAL_VERSION
        );
        assert_eq!(
            i32::from_le_bytes(encoded.frame[44..48].try_into().unwrap()),
            17
        );
        assert_eq!(
            u32::from_le_bytes(encoded.frame[48..52].try_into().unwrap()),
            4_000
        );
        let tenant_len = u32::from_le_bytes(encoded.frame[52..56].try_into().unwrap()) as usize;
        let contract_len = u32::from_le_bytes(encoded.frame[40..44].try_into().unwrap()) as usize;
        assert_eq!(
            &encoded.frame[56 + contract_len..56 + contract_len + tenant_len],
            b"tenant-a"
        );
    }

    #[test]
    fn external_backend_maps_scores_without_exposing_public_ids() {
        let page = [0, 0, 8];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(
                candidate,
                i64::MAX,
                "object://immutable/tensor",
                4,
                2,
                ExternalTensorDtype::F32,
            ),
        )]));
        let transport = MockTransport {
            similarities: vec![(0, 3.5)],
        };
        let results = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .rerank(&query, &mut candidates, &mut source)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].heap_key, page);
        assert_eq!(results[0].distance.to_f32(), -3.5);
    }

    #[test]
    fn external_backend_splits_one_logical_query_and_shares_its_deadline() {
        let pages = [[0, 0, 1], [0, 0, 2], [0, 0, 3]];
        let candidates = pages.map(|heap_key| PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key,
        });
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidate_iter = candidates.into_iter();
        let mut source =
            MockDescriptorSource(BTreeMap::from_iter(candidates.into_iter().enumerate().map(
                |(index, candidate)| {
                    (
                        candidate.heap_key,
                        external_descriptor(
                            candidate,
                            index as i64,
                            &format!("object://immutable/tensor-{index}"),
                            2,
                            2,
                            ExternalTensorDtype::F32,
                        ),
                    )
                },
            )));
        let observations = Rc::new(RefCell::new(BatchObservations::default()));
        let transport = BatchingTransport {
            observations: Rc::clone(&observations),
            delay: Duration::from_millis(5),
        };
        let results = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_millis(100),
            5,
            4096,
        )
        .with_scheduling("tenant-a".into(), 9)
        .rerank(&query, &mut candidate_iter, &mut source)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(results.len(), 3);
        let observations = observations.borrow();
        assert_eq!(observations.candidate_counts, vec![2, 1]);
        assert!(observations.transport_timeouts[1] < observations.transport_timeouts[0]);
        assert!(observations.scheduled_timeouts_ms[1] < observations.scheduled_timeouts_ms[0]);
    }

    #[test]
    fn external_backend_rejects_a_single_unsplittable_tensor() {
        let page = [0, 0, 1];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(
                candidate,
                1,
                "object://immutable/tensor",
                5,
                2,
                ExternalTensorDtype::F32,
            ),
        )]));
        let observations = Rc::new(RefCell::new(BatchObservations::default()));
        let transport = BatchingTransport {
            observations: Rc::clone(&observations),
            delay: Duration::ZERO,
        };
        let result = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_millis(100),
            5,
            4096,
        )
        .rerank(&query, &mut candidates, &mut source);

        assert!(matches!(result, Err(RerankError::RequestTooLarge)));
        assert!(observations.borrow().candidate_counts.is_empty());
    }

    #[test]
    fn external_backend_does_not_refresh_timeout_for_later_batches() {
        let pages = [[0, 0, 1], [0, 0, 2], [0, 0, 3]];
        let candidates = pages.map(|heap_key| PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key,
        });
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidate_iter = candidates.into_iter();
        let mut source =
            MockDescriptorSource(BTreeMap::from_iter(candidates.into_iter().enumerate().map(
                |(index, candidate)| {
                    (
                        candidate.heap_key,
                        external_descriptor(
                            candidate,
                            index as i64,
                            &format!("object://immutable/tensor-{index}"),
                            2,
                            2,
                            ExternalTensorDtype::F32,
                        ),
                    )
                },
            )));
        let observations = Rc::new(RefCell::new(BatchObservations::default()));
        let transport = BatchingTransport {
            observations: Rc::clone(&observations),
            delay: Duration::from_millis(20),
        };
        let result = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_millis(5),
            5,
            4096,
        )
        .rerank(&query, &mut candidate_iter, &mut source);

        assert!(matches!(
            result,
            Err(RerankError::Transport(message)) if message == "logical request timed out"
        ));
        assert_eq!(observations.borrow().candidate_counts, vec![2]);
    }

    #[test]
    fn logical_external_request_submits_all_descriptors_once_and_returns_top_k() {
        let pages = [[0, 0, 1], [0, 0, 2], [0, 0, 3]];
        let candidates = pages.map(|heap_key| PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key,
        });
        let mut candidate_iter = candidates.into_iter();
        let mut source =
            MockDescriptorSource(BTreeMap::from_iter(candidates.into_iter().enumerate().map(
                |(index, candidate)| {
                    (
                        candidate.heap_key,
                        external_descriptor(
                            candidate,
                            index as i64,
                            &format!("object://immutable/logical-{index}"),
                            900_000,
                            2,
                            ExternalTensorDtype::F32,
                        ),
                    )
                },
            )));
        let observations = Rc::new(RefCell::new(Vec::new()));
        let transport = LogicalTransport {
            observations: Rc::clone(&observations),
        };
        let results = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_secs(1),
            5,
            4096,
        )
        .rerank_logical(&[vector(&[1.0, 0.0])], &mut candidate_iter, &mut source, 1)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(
            observations.borrow().as_slice(),
            &[(MANIFEST_LOGICAL_EXTERNAL_VERSION, 3, 1)]
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].heap_key, pages[2]);
    }

    #[test]
    fn logical_external_request_registers_after_a_manifest_miss() {
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 1],
        };
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            candidate.heap_key,
            external_descriptor(
                candidate,
                7,
                &format!("sha256://{}", "a".repeat(64)),
                2,
                2,
                ExternalTensorDtype::F32,
            ),
        )]));
        let versions = Rc::new(RefCell::new(Vec::new()));
        let results = GpuExternalTileMaxsimBackend::new(
            ManifestMissTransport {
                versions: Rc::clone(&versions),
                legacy_rejection: false,
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .rerank_logical(&[vector(&[1.0, 0.0])], &mut candidates, &mut source, 1)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(
            versions.borrow().as_slice(),
            &[
                MANIFEST_LOGICAL_EXTERNAL_VERSION,
                COMPACT_LOGICAL_EXTERNAL_VERSION
            ]
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].heap_key, candidate.heap_key);
    }

    #[test]
    fn catalog_request_registers_once_then_uses_public_id_selection() {
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 7],
        };
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut backend = GpuExternalTileMaxsimBackend::new(
            CatalogProbeMissTransport {
                calls: Rc::clone(&calls),
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .with_catalog_revision(Some("[[\"brain-a\",\"17\"]]".into()));
        for _ in 0..2 {
            let mut candidates = vec![candidate].into_iter();
            let mut source = MockDescriptorSource(BTreeMap::from([(
                candidate.heap_key,
                external_descriptor(
                    candidate,
                    7,
                    &format!("sha256://{}", "a".repeat(64)),
                    2,
                    2,
                    ExternalTensorDtype::F32,
                ),
            )]));
            assert_eq!(
                backend
                    .rerank_logical(&[vector(&[1.0, 0.0])], &mut candidates, &mut source, 1,)
                    .unwrap()
                    .count(),
                1
            );
        }
        let calls = calls.borrow();
        assert_eq!(
            calls.iter().map(|call| call.0).collect::<Vec<_>>(),
            vec![11, 10, 10, 11]
        );
        assert_eq!(
            calls.iter().map(|call| call.1).collect::<Vec<_>>(),
            vec![3, 1, 2, 3]
        );
        assert!(calls[0].2 < calls[1].2 && calls[1].2 < calls[2].2);
    }

    #[test]
    fn raw_fp8_catalog_probe_does_not_require_descriptors() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut backend = GpuExternalTileMaxsimBackend::new(
            CatalogMissTransport {
                calls: Rc::clone(&calls),
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .with_scoring_profile(PostgresMaxsimScoringProfile::RawFp8E4m3)
        .with_catalog_revision(Some("revision-17".into()));
        assert!(matches!(
            backend
                .rerank_raw_fp8_catalog(&[vector(&[1.0, 0.0])], &[7, 10], 1)
                .unwrap(),
            CatalogSelectionOutcome::Miss
        ));
        let calls = calls.borrow();
        assert_eq!(calls.len(), 6);
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.0, call.1))
                .collect::<Vec<_>>(),
            vec![(11, 3), (10, 1), (11, 3), (10, 1), (11, 3), (10, 1)],
        );
    }

    #[test]
    fn explicit_catalog_storage_dtype_is_probed_first() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let backend = GpuExternalTileMaxsimBackend::new(
            CatalogMissTransport {
                calls: Rc::clone(&calls),
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .with_catalog_storage_dtype(Some(ExternalTensorDtype::F16));

        assert_eq!(
            backend.catalog_storage_dtypes("explicit-dtype-test"),
            vec![TensorDtype::F16, TensorDtype::Fp8E4m3, TensorDtype::F32]
        );
    }

    #[test]
    fn catalog_storage_dtype_keeps_bounded_fallbacks() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let backend = GpuExternalTileMaxsimBackend::new(
            CatalogMissTransport {
                calls: Rc::clone(&calls),
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .with_catalog_storage_dtype(Some(ExternalTensorDtype::Fp8E4m3));

        assert_eq!(
            backend.catalog_storage_dtypes("fallback-dtype-test"),
            vec![TensorDtype::Fp8E4m3, TensorDtype::F16, TensorDtype::F32]
        );
    }

    #[test]
    fn logical_external_request_falls_back_during_extension_first_upgrade() {
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 2],
        };
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            candidate.heap_key,
            external_descriptor(
                candidate,
                8,
                &format!("sha256://{}", "a".repeat(64)),
                2,
                2,
                ExternalTensorDtype::F32,
            ),
        )]));
        let versions = Rc::new(RefCell::new(Vec::new()));
        let results = GpuExternalTileMaxsimBackend::new(
            ManifestMissTransport {
                versions: Rc::clone(&versions),
                legacy_rejection: true,
            },
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .rerank_logical(&[vector(&[1.0, 0.0])], &mut candidates, &mut source, 1)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(
            versions.borrow().as_slice(),
            &[
                MANIFEST_LOGICAL_EXTERNAL_VERSION,
                COMPACT_LOGICAL_EXTERNAL_VERSION
            ]
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].heap_key, candidate.heap_key);
    }

    #[test]
    fn external_request_rejects_shape_and_declared_payload_overflow() {
        let page = [0, 0, 9];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![half_vector(&[1.0, 0.0])];
        let make_source = |dimension, rows| {
            MockDescriptorSource(BTreeMap::from([(
                page,
                external_descriptor(
                    candidate,
                    9,
                    "object://immutable/tensor",
                    rows,
                    dimension,
                    ExternalTensorDtype::F16,
                ),
            )]))
        };

        let mut candidates = vec![candidate].into_iter();
        assert!(matches!(
            encode_external_request(
                1,
                "contract@1",
                &query,
                &mut candidates,
                &mut make_source(3, 1),
                100,
                4096,
                None,
                Duration::from_secs(2),
            ),
            Err(RerankError::TensorMismatch)
        ));

        let mut candidates = vec![candidate].into_iter();
        assert!(matches!(
            encode_external_request(
                1,
                "contract@1",
                &query,
                &mut candidates,
                &mut make_source(2, 100),
                1000,
                64,
                None,
                Duration::from_secs(2),
            ),
            Err(RerankError::RequestTooLarge)
        ));
    }

    #[test]
    fn request_frame_encodes_f16_tensor_bits() {
        let page = [0, 0, 1];
        let query = vec![half_vector(&[1.0, -0.5])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source =
            MockTensorSource(BTreeMap::from([(page, vec![half_vector(&[0.25, 2.0])])]));
        let encoded = encode_request(9, &query, &mut candidates, &mut source, 100, 4096).unwrap();

        assert_eq!(encoded.frame[36], TensorDtype::F16 as u8);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[40..42].try_into().unwrap()),
            simd::f16::from_f32(1.0).to_bits()
        );
        assert_eq!(
            u16::from_le_bytes(encoded.frame[42..44].try_into().unwrap()),
            simd::f16::from_f32(-0.5).to_bits()
        );
    }

    #[test]
    fn request_limits_are_enforced_before_transport() {
        let page = [0, 0, 1];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([(page, vec![vector(&[1.0, 0.0])])]));
        assert!(matches!(
            encode_request(1, &query, &mut candidates, &mut source, 1, 4096),
            Err(RerankError::RequestTooLarge)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_address_is_length_bounded_and_nul_terminated() {
        let (address, length) = unix_socket_address("/tmp/vectorchord.sock").unwrap();
        let path_offset = std::mem::offset_of!(libc::sockaddr_un, sun_path);

        assert_eq!(address.sun_family, libc::AF_UNIX as libc::sa_family_t);
        assert_eq!(
            length as usize,
            path_offset + "/tmp/vectorchord.sock".len() + 1
        );
        assert_eq!(
            &address.sun_path[.."/tmp/vectorchord.sock".len()],
            "/tmp/vectorchord.sock"
                .as_bytes()
                .iter()
                .map(|byte| *byte as libc::c_char)
                .collect::<Vec<_>>()
        );
        assert_eq!(address.sun_path["/tmp/vectorchord.sock".len()], 0);

        let too_long = "x".repeat(address.sun_path.len());
        assert!(matches!(
            unix_socket_address(&too_long),
            Err(RerankError::Transport(message)) if message == "endpoint path is too long"
        ));
        assert!(matches!(
            unix_socket_address("invalid\0path"),
            Err(RerankError::Transport(message)) if message == "endpoint contains a NUL byte"
        ));
    }
}
