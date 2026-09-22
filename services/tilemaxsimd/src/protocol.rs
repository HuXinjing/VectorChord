// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use anyhow::{Result, anyhow, bail};
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

// Catalog protocol constants are owned by the shared client crate so the
// daemon and every in-tree producer compile against one source of truth.
pub use tilemaxsim_client::{
    HEADER_BYTES, VERSION_CATALOG_LOGICAL_EXTERNAL, VERSION_CATALOG_SELECTION_REFERENCE,
    VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE,
    VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE,
    VERSION_SCOPED_CATALOG_SELECTION_REFERENCE,
};
pub const VERSION_EXTERNAL: u16 = 2;
pub const VERSION_SCHEDULED_EXTERNAL: u16 = 3;
pub const VERSION_PROFILED_EXTERNAL: u16 = 4;
pub const VERSION_QUANTIZED_EXTERNAL: u16 = 5;
/// One logical rerank request. Candidate tensor payloads remain external, so
/// the wire limit applies to descriptor metadata rather than the sum of the
/// referenced tensor payloads. The daemon owns bounded GPU quantization and
/// returns only the requested global top-k.
pub const VERSION_LOGICAL_EXTERNAL: u16 = 6;
/// Logical top-k with fixed-width content-addressed descriptors. The model
/// contract, dimension, and dtype are request-wide, so repeating two textual
/// SHA-256 representations per candidate only wastes wire and parse time.
pub const VERSION_COMPACT_LOGICAL_EXTERNAL: u16 = 7;
/// Compact logical request with a request-wide canonical storage dtype in the
/// former reserved profile byte. Query and document tensors may differ.
pub const VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL: u16 = 8;
/// Descriptor-manifest reference. The request carries the query tensor and a
/// SHA-256 digest of the canonical compact descriptor list; the daemon resolves
/// the list from its bounded manifest cache.
pub const VERSION_MANIFEST_LOGICAL_EXTERNAL: u16 = 9;
const MAGIC: &[u8; 4] = b"VCTM";
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Descriptor {
    pub candidate_id: u32,
    pub contract: String,
    pub digest: String,
    pub rows: u32,
    pub dimension: u32,
    pub dtype: u8,
    #[serde(skip, default)]
    pub raw_fp8_cache_key: OnceLock<Arc<str>>,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub protocol_version: u16,
    pub request_id: u64,
    /// Logical scheduling domain. It is never used for authorization or
    /// tensor lookup; the caller must already have applied hard ACL filters.
    pub tenant: String,
    pub model_contract: String,
    /// Higher values run first within the scheduler's fairness policy.
    pub priority: i32,
    /// Client-supplied end-to-end budget. Zero is used only by legacy v2.
    pub timeout_ms: u32,
    pub query_rows: u32,
    pub dimension: u32,
    pub dtype: u8,
    pub candidate_dtype: u8,
    pub scoring_profile: ScoringProfile,
    pub quantization_contract: Option<String>,
    pub top_k: Option<usize>,
    pub query: Vec<u8>,
    pub candidates: Arc<Vec<Descriptor>>,
    pub candidate_start: usize,
    pub candidate_end: usize,
    pub manifest_digest: Option<[u8; 32]>,
    pub catalog_digest: Option<[u8; 32]>,
    pub catalog_selection_digest: Option<[u8; 32]>,
    pub catalog_public_ids: Vec<i64>,
    pub catalog_registration: bool,
    /// Sorted ordinals within the resolved catalog selection. Empty means the
    /// request has only the ordinary global result window.
    pub scoped_candidate_ordinals: Vec<u32>,
}

impl Request {
    pub fn candidate_slice(&self) -> &[Descriptor] {
        &self.candidates[self.candidate_start..self.candidate_end]
    }

    pub fn replace_candidates(&mut self, candidates: Arc<Vec<Descriptor>>) {
        self.candidate_start = 0;
        self.candidate_end = candidates.len();
        self.candidates = candidates;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringProfile {
    ExactFp16,
    Int8,
    Fp8E4m3,
    RawFp8E4m3,
    Pq,
    OpqRpq,
}

impl ScoringProfile {
    fn parse(value: u8) -> Result<Self> {
        Ok(match value {
            1 => Self::ExactFp16,
            2 => Self::Int8,
            3 => Self::Fp8E4m3,
            4 => Self::Pq,
            5 => Self::OpqRpq,
            6 => Self::RawFp8E4m3,
            _ => bail!("unsupported TileMaxSim scoring profile"),
        })
    }

    pub fn cache_tag(self) -> &'static str {
        match self {
            Self::ExactFp16 => "exact-fp16-v1",
            Self::Int8 => "int8-row-scale-v1",
            Self::Fp8E4m3 => "fp8-e4m3-row-scale-v1",
            Self::RawFp8E4m3 => "fp8-e4m3-raw-v1",
            Self::Pq => "pq-adc-v1",
            Self::OpqRpq => "opq-rpq-adc-v1",
        }
    }

    pub fn native_code(self) -> u8 {
        match self {
            Self::ExactFp16 => 1,
            Self::Int8 => 2,
            Self::Fp8E4m3 => 3,
            // The native kernel consumes the same byte-plus-scale layout as
            // row-scaled E4M3. Raw E4M3 stores an identity scale per row.
            Self::RawFp8E4m3 => 3,
            Self::Pq => 4,
            Self::OpqRpq => 5,
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| anyhow!("request overflow"))?;
        if end > self.bytes.len() {
            bail!("truncated request");
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn varint_u64(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        let mut bytes = 0_usize;
        for shift in (0..=63).step_by(7) {
            let byte = self.u8()?;
            bytes += 1;
            if shift == 63 && byte > 1 {
                bail!("catalog public ID varint overflow");
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                if bytes > 1 && byte == 0 {
                    bail!("non-canonical catalog public ID varint");
                }
                return Ok(value);
            }
        }
        bail!("catalog public ID varint overflow")
    }

    fn text(&mut self, count: usize, maximum: usize, name: &str) -> Result<String> {
        if count == 0 || count > maximum {
            bail!("invalid {name} length");
        }
        let value = std::str::from_utf8(self.take(count)?)?;
        if value.chars().any(|character| character.is_control()) {
            bail!("{name} contains control characters");
        }
        Ok(value.to_owned())
    }

    fn finish(self) -> Result<()> {
        if self.offset != self.bytes.len() {
            bail!("trailing request bytes");
        }
        Ok(())
    }
}

fn tensor_bytes(rows: u32, dimension: u32, dtype: u8) -> Result<usize> {
    let scalar = match dtype {
        1 => 4,
        2 => 2,
        3 => 1,
        _ => bail!("unsupported tensor dtype"),
    };
    if rows == 0 || dimension == 0 || dimension > 60_000 {
        bail!("invalid tensor shape");
    }
    (rows as usize)
        .checked_mul(dimension as usize)
        .and_then(|value| value.checked_mul(scalar))
        .ok_or_else(|| anyhow!("tensor shape is too large"))
}

fn validate_finite(payload: &[u8], dtype: u8) -> Result<()> {
    if dtype == 1 {
        for scalar in payload.chunks_exact(4) {
            if !f32::from_bits(u32::from_le_bytes(scalar.try_into().unwrap())).is_finite() {
                bail!("query contains a non-finite value");
            }
        }
    } else if dtype == 2 {
        for scalar in payload.chunks_exact(2) {
            let bits = u16::from_le_bytes(scalar.try_into().unwrap());
            if bits & 0x7c00 == 0x7c00 {
                bail!("query contains a non-finite value");
            }
        }
    } else if payload.iter().any(|value| value & 0x7f == 0x7f) {
        bail!("query contains a non-finite value");
    }
    Ok(())
}

pub fn parse(frame: &[u8]) -> Result<Request> {
    if frame.len() < HEADER_BYTES {
        bail!("truncated request header");
    }
    if &frame[..4] != MAGIC {
        bail!("invalid protocol magic");
    }
    let version = u16::from_le_bytes(frame[4..6].try_into().unwrap());
    let kind = u16::from_le_bytes(frame[6..8].try_into().unwrap());
    let request_id = u64::from_le_bytes(frame[8..16].try_into().unwrap());
    let body_bytes = u64::from_le_bytes(frame[16..24].try_into().unwrap());
    if !matches!(
        version,
        VERSION_EXTERNAL
            | VERSION_SCHEDULED_EXTERNAL
            | VERSION_PROFILED_EXTERNAL
            | VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) || kind != REQUEST_KIND
    {
        bail!("Rust daemon requires TileMaxSim external protocol v2 through v14");
    }
    if usize::try_from(body_bytes).ok() != Some(frame.len() - HEADER_BYTES) {
        bail!("request body length mismatch");
    }
    let mut reader = Reader::new(&frame[HEADER_BYTES..]);
    let dimension = reader.u32()?;
    let query_rows = reader.u32()?;
    let candidate_count = reader.u32()?;
    let dtype = reader.u8()?;
    let scoring = reader.u8()?;
    let scoring_profile = if matches!(
        version,
        VERSION_PROFILED_EXTERNAL
            | VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        let profile = ScoringProfile::parse(reader.u8()?)?;
        let storage_dtype = reader.u8()?;
        if !matches!(
            version,
            VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
                | VERSION_MANIFEST_LOGICAL_EXTERNAL
                | VERSION_CATALOG_LOGICAL_EXTERNAL
                | VERSION_CATALOG_SELECTION_REFERENCE
                | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
                | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
                | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
        ) && storage_dtype != 0
        {
            bail!("unsupported reserved bits");
        }
        (profile, storage_dtype)
    } else {
        if reader.u16()? != 0 {
            bail!("unsupported reserved bits");
        }
        (ScoringProfile::ExactFp16, 0)
    };
    let (scoring_profile, storage_dtype) = scoring_profile;
    let contract_bytes = reader.u32()? as usize;
    let quantization_contract_bytes = if matches!(
        version,
        VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        reader.u32()? as usize
    } else {
        0
    };
    if scoring != 1 {
        bail!("unsupported scoring function or reserved bits");
    }
    if candidate_count > 65_536 {
        bail!("too many candidates");
    }
    let (priority, timeout_ms, tenant_bytes) = if matches!(
        version,
        VERSION_SCHEDULED_EXTERNAL
            | VERSION_PROFILED_EXTERNAL
            | VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        (reader.i32()?, reader.u32()?, reader.u32()? as usize)
    } else {
        (0, 0, 0)
    };
    if !(-100..=100).contains(&priority) {
        bail!("scheduler priority must be between -100 and 100");
    }
    if matches!(
        version,
        VERSION_SCHEDULED_EXTERNAL
            | VERSION_PROFILED_EXTERNAL
            | VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) && !(1..=600_000).contains(&timeout_ms)
    {
        bail!("scheduler timeout must be between 1 and 600000 milliseconds");
    }
    let top_k = if matches!(
        version,
        VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        let value = reader.u32()? as usize;
        if value == 0 || value > candidate_count as usize {
            bail!("logical top-k must be between 1 and candidate count");
        }
        Some(value)
    } else {
        None
    };
    let contract = reader.text(contract_bytes, 512, "model contract")?;
    let quantization_contract = if matches!(
        version,
        VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) && quantization_contract_bytes > 0
    {
        let value = reader.text(quantization_contract_bytes, 128, "quantization contract")?;
        if value.len() != 69
            || !value.starts_with("qtc1-")
            || !value[5..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("invalid quantization contract ID");
        }
        Some(value)
    } else {
        None
    };
    if matches!(scoring_profile, ScoringProfile::Pq | ScoringProfile::OpqRpq)
        != quantization_contract.is_some()
    {
        bail!("PQ-family profile and quantization contract must be supplied together");
    }
    let tenant = if matches!(
        version,
        VERSION_SCHEDULED_EXTERNAL
            | VERSION_PROFILED_EXTERNAL
            | VERSION_QUANTIZED_EXTERNAL
            | VERSION_LOGICAL_EXTERNAL
            | VERSION_COMPACT_LOGICAL_EXTERNAL
            | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
            | VERSION_MANIFEST_LOGICAL_EXTERNAL
            | VERSION_CATALOG_LOGICAL_EXTERNAL
            | VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        reader.text(tenant_bytes, 256, "scheduler tenant")?
    } else {
        "__default__".to_owned()
    };
    let query_bytes = tensor_bytes(query_rows, dimension, dtype)?;
    let query = reader.take(query_bytes)?.to_vec();
    validate_finite(&query, dtype)?;
    let manifest_digest = if version == VERSION_MANIFEST_LOGICAL_EXTERNAL {
        Some(reader.take(32)?.try_into().unwrap())
    } else {
        None
    };
    let (
        catalog_digest,
        catalog_selection_digest,
        catalog_registration,
        mut catalog_public_ids,
        scoped_candidate_ordinals,
    ) = if version == VERSION_CATALOG_LOGICAL_EXTERNAL {
        let mode = reader.u8()?;
        if !matches!(mode, 1 | 2) {
            bail!("invalid descriptor catalog frame mode");
        }
        let digest = reader.take(32)?.try_into().unwrap();
        let mut public_ids = Vec::with_capacity(candidate_count as usize);
        if mode == 1 {
            let mut previous = 0_u64;
            for _ in 0..candidate_count {
                let delta = reader.varint_u64()?;
                if delta == 0 {
                    bail!("catalog public IDs must be strictly increasing");
                }
                let current = previous
                    .checked_add(delta)
                    .ok_or_else(|| anyhow!("catalog public ID overflow"))?;
                if current > i64::MAX as u64 {
                    bail!("catalog public ID overflow");
                }
                public_ids.push(current as i64);
                previous = current;
            }
        }
        (Some(digest), None, mode == 2, public_ids, Vec::new())
    } else if matches!(
        version,
        VERSION_CATALOG_SELECTION_REFERENCE
            | VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
            | VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
    ) {
        let expected_mode = if matches!(
            version,
            VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
                | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
        ) {
            4
        } else {
            3
        };
        if reader.u8()? != expected_mode {
            bail!("invalid descriptor catalog selection reference mode");
        }
        let catalog_digest = reader.take(32)?.try_into().unwrap();
        let selection_digest = reader.take(32)?.try_into().unwrap();
        let mut scoped_ordinals = Vec::new();
        if matches!(
            version,
            VERSION_SCOPED_CATALOG_SELECTION_REFERENCE
                | VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
        ) {
            let scoped_count = reader.u32()? as usize;
            if scoped_count == 0 || scoped_count > candidate_count as usize {
                bail!("scoped candidate count must be between 1 and candidate count");
            }
            let mut previous_plus_one = 0_u64;
            for _ in 0..scoped_count {
                let delta = reader.varint_u64()?;
                if delta == 0 {
                    bail!("scoped candidate ordinals must be strictly increasing");
                }
                let current_plus_one = previous_plus_one
                    .checked_add(delta)
                    .ok_or_else(|| anyhow!("scoped candidate ordinal overflow"))?;
                let ordinal = current_plus_one
                    .checked_sub(1)
                    .ok_or_else(|| anyhow!("scoped candidate ordinal overflow"))?;
                if ordinal >= u64::from(candidate_count) {
                    bail!("scoped candidate ordinal is outside the candidate set");
                }
                scoped_ordinals.push(ordinal as u32);
                previous_plus_one = current_plus_one;
            }
        }
        (
            Some(catalog_digest),
            Some(selection_digest),
            false,
            Vec::new(),
            scoped_ordinals,
        )
    } else {
        (None, None, false, Vec::new(), Vec::new())
    };
    let mut total_tokens = query_rows as usize;
    let mut total_bytes = query_bytes;
    let mut candidate_ids = HashSet::new();
    let mut candidates = Vec::with_capacity(candidate_count as usize);
    for ordinal in
        0..if manifest_digest.is_some() || (catalog_digest.is_some() && !catalog_registration) {
            0
        } else {
            candidate_count
        }
    {
        let (public_id, candidate_id) = if catalog_registration {
            let public_id = reader.i64()?;
            if public_id <= 0
                || catalog_public_ids
                    .last()
                    .is_some_and(|previous| *previous >= public_id)
            {
                bail!("catalog public IDs must be positive and strictly increasing");
            }
            catalog_public_ids.push(public_id);
            (Some(public_id), ordinal)
        } else {
            (None, reader.u32()?)
        };
        let rows = reader.u32()?;
        if catalog_registration {
            if !candidate_ids.insert(candidate_id) {
                bail!("duplicate candidate ID");
            }
            let digest = hex::encode(reader.take(32)?);
            total_tokens = total_tokens
                .checked_add(rows as usize)
                .ok_or_else(|| anyhow!("token overflow"))?;
            total_bytes = total_bytes
                .checked_add(tensor_bytes(
                    rows,
                    dimension,
                    if storage_dtype == 0 {
                        dtype
                    } else {
                        storage_dtype
                    },
                )?)
                .ok_or_else(|| anyhow!("byte overflow"))?;
            candidates.push(Descriptor {
                candidate_id,
                contract: contract.clone(),
                digest,
                rows,
                dimension,
                dtype: if storage_dtype == 0 {
                    dtype
                } else {
                    storage_dtype
                },
                raw_fp8_cache_key: OnceLock::new(),
            });
            debug_assert!(public_id.is_some());
            continue;
        }
        if matches!(
            version,
            VERSION_COMPACT_LOGICAL_EXTERNAL | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
        ) {
            if !candidate_ids.insert(candidate_id) {
                bail!("duplicate candidate ID");
            }
            let digest = hex::encode(reader.take(32)?);
            total_tokens = total_tokens
                .checked_add(rows as usize)
                .ok_or_else(|| anyhow!("token overflow"))?;
            total_bytes = total_bytes
                .checked_add(tensor_bytes(
                    rows,
                    dimension,
                    if storage_dtype == 0 {
                        dtype
                    } else {
                        storage_dtype
                    },
                )?)
                .ok_or_else(|| anyhow!("byte overflow"))?;
            candidates.push(Descriptor {
                candidate_id,
                contract: contract.clone(),
                digest,
                rows,
                dimension,
                dtype: if storage_dtype == 0 {
                    dtype
                } else {
                    storage_dtype
                },
                raw_fp8_cache_key: OnceLock::new(),
            });
            continue;
        }
        let reference_bytes = reader.u32()? as usize;
        let checksum_bytes = reader.u32()? as usize;
        if !candidate_ids.insert(candidate_id) {
            bail!("duplicate candidate ID");
        }
        let tensor_ref = reader.text(reference_bytes, 4096, "tensor reference")?;
        let checksum = reader.text(checksum_bytes, 512, "tensor checksum")?;
        let digest = tensor_ref
            .strip_prefix("sha256://")
            .ok_or_else(|| anyhow!("unsupported tensor reference"))?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || checksum != format!("sha256:{digest}")
        {
            bail!("invalid content-addressed tensor descriptor");
        }
        let bytes = tensor_bytes(rows, dimension, dtype)?;
        total_tokens = total_tokens
            .checked_add(rows as usize)
            .ok_or_else(|| anyhow!("token overflow"))?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("byte overflow"))?;
        if !matches!(
            version,
            VERSION_LOGICAL_EXTERNAL
                | VERSION_COMPACT_LOGICAL_EXTERNAL
                | VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL
        ) && (total_tokens > 1_000_000 || total_bytes > 1024 * 1024 * 1024)
        {
            bail!("request exceeds tensor limits");
        }
        candidates.push(Descriptor {
            candidate_id,
            contract: contract.clone(),
            digest: digest.to_owned(),
            rows,
            dimension,
            dtype,
            raw_fp8_cache_key: OnceLock::new(),
        });
    }
    reader.finish()?;
    let candidates = Arc::new(candidates);
    let candidate_end = candidates.len();
    Ok(Request {
        protocol_version: version,
        request_id,
        tenant,
        model_contract: contract,
        priority,
        timeout_ms,
        query_rows,
        dimension,
        dtype,
        candidate_dtype: if storage_dtype == 0 {
            dtype
        } else {
            storage_dtype
        },
        scoring_profile,
        quantization_contract,
        top_k,
        query,
        candidates,
        candidate_start: 0,
        candidate_end,
        manifest_digest,
        catalog_digest,
        catalog_selection_digest,
        catalog_public_ids,
        catalog_registration,
        scoped_candidate_ordinals,
    })
}

pub fn success(version: u16, request_id: u64, results: &[(u32, f32)]) -> Vec<u8> {
    let body_bytes = 8 + results.len() * 8;
    let mut frame = Vec::with_capacity(HEADER_BYTES + body_bytes);
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&version.to_le_bytes());
    frame.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
    frame.extend_from_slice(&request_id.to_le_bytes());
    frame.extend_from_slice(&(body_bytes as u64).to_le_bytes());
    frame.extend_from_slice(&0_u32.to_le_bytes());
    frame.extend_from_slice(&(results.len() as u32).to_le_bytes());
    for (candidate_id, score) in results {
        frame.extend_from_slice(&candidate_id.to_le_bytes());
        frame.extend_from_slice(&score.to_le_bytes());
    }
    frame
}

pub fn failure(version: u16, request_id: u64, status: u32, message: &str) -> Vec<u8> {
    let message = message.as_bytes();
    let message = &message[..message.len().min(64 * 1024)];
    let body_bytes = 8 + message.len();
    let mut frame = Vec::with_capacity(HEADER_BYTES + body_bytes);
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&version.to_le_bytes());
    frame.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
    frame.extend_from_slice(&request_id.to_le_bytes());
    frame.extend_from_slice(&(body_bytes as u64).to_le_bytes());
    frame.extend_from_slice(&status.max(1).to_le_bytes());
    frame.extend_from_slice(&(message.len() as u32).to_le_bytes());
    frame.extend_from_slice(message);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduled_frame(priority: i32, timeout_ms: u32, tenant: &str) -> Vec<u8> {
        profiled_frame(VERSION_SCHEDULED_EXTERNAL, 0, priority, timeout_ms, tenant)
    }

    fn profiled_frame(
        version: u16,
        profile: u8,
        priority: i32,
        timeout_ms: u32,
        tenant: &str,
    ) -> Vec<u8> {
        let contract = "model@1";
        let digest = "a".repeat(64);
        let reference = format!("sha256://{digest}");
        let checksum = format!("sha256:{digest}");
        let mut body = Vec::new();
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.push(2);
        body.push(1);
        if version == VERSION_PROFILED_EXTERNAL {
            body.push(profile);
            body.push(0);
        } else {
            body.extend_from_slice(&0_u16.to_le_bytes());
        }
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&priority.to_le_bytes());
        body.extend_from_slice(&timeout_ms.to_le_bytes());
        body.extend_from_slice(&(tenant.len() as u32).to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(tenant.as_bytes());
        body.extend_from_slice(&0_u16.to_le_bytes());
        body.extend_from_slice(&0_u16.to_le_bytes());
        body.extend_from_slice(&7_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&(reference.len() as u32).to_le_bytes());
        body.extend_from_slice(&(checksum.len() as u32).to_le_bytes());
        body.extend_from_slice(reference.as_bytes());
        body.extend_from_slice(checksum.as_bytes());
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&version.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);
        frame
    }

    fn quantized_frame(profile: u8, contract_id: &str) -> Vec<u8> {
        let mut frame = profiled_frame(VERSION_PROFILED_EXTERNAL, profile, 0, 4_000, "tenant-a");
        let body = frame.split_off(HEADER_BYTES);
        let model_len = u32::from_le_bytes(body[16..20].try_into().unwrap()) as usize;
        let mut upgraded = Vec::with_capacity(body.len() + 4 + contract_id.len());
        upgraded.extend_from_slice(&body[..20]);
        upgraded.extend_from_slice(&(contract_id.len() as u32).to_le_bytes());
        upgraded.extend_from_slice(&body[20..32]);
        upgraded.extend_from_slice(&body[32..32 + model_len]);
        upgraded.extend_from_slice(contract_id.as_bytes());
        upgraded.extend_from_slice(&body[32 + model_len..]);
        frame.truncate(0);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_QUANTIZED_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(upgraded.len() as u64).to_le_bytes());
        frame.extend_from_slice(&upgraded);
        frame
    }

    #[test]
    fn scheduled_protocol_carries_tenant_priority_and_deadline() {
        let request = parse(&scheduled_frame(17, 4_000, "tenant-a")).unwrap();
        assert_eq!(request.protocol_version, VERSION_SCHEDULED_EXTERNAL);
        assert_eq!(request.request_id, 42);
        assert_eq!(request.tenant, "tenant-a");
        assert_eq!(request.priority, 17);
        assert_eq!(request.timeout_ms, 4_000);
        assert_eq!(request.candidates[0].candidate_id, 7);
    }

    #[test]
    fn scheduled_protocol_rejects_priority_outside_the_public_contract() {
        assert!(parse(&scheduled_frame(101, 4_000, "tenant-a")).is_err());
    }

    #[test]
    fn profiled_protocol_carries_an_explicit_quantized_profile() {
        let request = parse(&profiled_frame(
            VERSION_PROFILED_EXTERNAL,
            2,
            17,
            4_000,
            "tenant-a",
        ))
        .unwrap();
        assert_eq!(request.scoring_profile, ScoringProfile::Int8);
        assert_eq!(request.tenant, "tenant-a");
    }

    #[test]
    fn profiled_protocol_rejects_unknown_profiles() {
        assert!(
            parse(&profiled_frame(
                VERSION_PROFILED_EXTERNAL,
                99,
                0,
                4_000,
                "tenant-a",
            ))
            .is_err()
        );
    }

    #[test]
    fn quantized_protocol_binds_the_immutable_contract() {
        let contract = format!("qtc1-{}", "a".repeat(64));
        let request = parse(&quantized_frame(4, &contract)).unwrap();
        assert_eq!(request.scoring_profile, ScoringProfile::Pq);
        assert_eq!(
            request.quantization_contract.as_deref(),
            Some(contract.as_str())
        );
    }

    #[test]
    fn logical_protocol_accepts_external_work_above_legacy_tensor_limits() {
        let contract = "model@1";
        let digest = "a".repeat(64);
        let reference = format!("sha256://{digest}");
        let checksum = format!("sha256:{digest}");
        let mut body = Vec::new();
        body.extend_from_slice(&320_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.push(2);
        body.push(1);
        body.push(1);
        body.push(0);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&vec![0_u8; 640]);
        for candidate_id in 0..2_u32 {
            body.extend_from_slice(&candidate_id.to_le_bytes());
            body.extend_from_slice(&900_000_u32.to_le_bytes());
            body.extend_from_slice(&(reference.len() as u32).to_le_bytes());
            body.extend_from_slice(&(checksum.len() as u32).to_le_bytes());
            body.extend_from_slice(reference.as_bytes());
            body.extend_from_slice(checksum.as_bytes());
        }
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.top_k, Some(1));
        assert_eq!(request.candidates.len(), 2);
    }

    #[test]
    fn compact_logical_protocol_expands_fixed_width_digests() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&320_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 1, 0]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&vec![0_u8; 640]);
        body.extend_from_slice(&7_u32.to_le_bytes());
        body.extend_from_slice(&32_u32.to_le_bytes());
        body.extend_from_slice(&[0xab; 32]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_COMPACT_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.top_k, Some(1));
        assert_eq!(request.candidates[0].candidate_id, 7);
        assert_eq!(request.candidates[0].digest, "ab".repeat(32));
    }

    #[test]
    fn typed_compact_protocol_separates_query_and_storage_dtype() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&320_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 6, 3]); // FP16 query, raw E4M3 storage
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&vec![0_u8; 640]);
        body.extend_from_slice(&7_u32.to_le_bytes());
        body.extend_from_slice(&32_u32.to_le_bytes());
        body.extend_from_slice(&[0xab; 32]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_TYPED_COMPACT_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.dtype, 2);
        assert_eq!(request.candidates[0].dtype, 3);
        assert_eq!(request.scoring_profile, ScoringProfile::RawFp8E4m3);
    }

    #[test]
    fn manifest_protocol_carries_only_the_descriptor_digest() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&320_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&45_601_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 6, 3]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&50_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&vec![0_u8; 640]);
        body.extend_from_slice(&[0xcd; 32]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_MANIFEST_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.top_k, Some(50));
        assert_eq!(request.candidate_dtype, 3);
        assert!(request.candidates.is_empty());
        assert_eq!(request.manifest_digest, Some([0xcd; 32]));
    }

    #[test]
    fn catalog_selection_decodes_delta_varint_public_ids() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&3_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 1, 0]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&[0_u8; 4]);
        body.push(1);
        body.extend_from_slice(&[0xcd; 32]);
        body.extend_from_slice(&[7, 3, 0xac, 0x02]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_CATALOG_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.catalog_digest, Some([0xcd; 32]));
        assert_eq!(request.catalog_public_ids, vec![7, 10, 310]);
        assert!(!request.catalog_registration);
        assert!(request.candidates.is_empty());
    }

    #[test]
    fn catalog_selection_reference_carries_only_bounded_digests() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&45_601_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 6, 3]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&50_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&[0_u8; 4]);
        body.push(3);
        body.extend_from_slice(&[0xcd; 32]);
        body.extend_from_slice(&[0xef; 32]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_CATALOG_SELECTION_REFERENCE.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.top_k, Some(50));
        assert_eq!(request.catalog_digest, Some([0xcd; 32]));
        assert_eq!(request.catalog_selection_digest, Some([0xef; 32]));
        assert!(request.catalog_public_ids.is_empty());
        assert!(request.candidates.is_empty());

        frame[4..6].copy_from_slice(&VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE.to_le_bytes());
        let persistent = parse(&frame).unwrap();
        assert_eq!(
            persistent.protocol_version,
            VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE
        );
        assert!(persistent.scoped_candidate_ordinals.is_empty());
    }

    #[test]
    fn scoped_catalog_reference_decodes_a_bounded_ordinal_subset() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&45_601_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 6, 3]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&50_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&[0_u8; 4]);
        body.push(4);
        body.extend_from_slice(&[0xcd; 32]);
        body.extend_from_slice(&[0xef; 32]);
        body.extend_from_slice(&3_u32.to_le_bytes());
        // Ordinals 0, 4, and 300 encoded as deltas over ordinal + 1.
        body.extend_from_slice(&[1, 4, 0xa8, 0x02]);
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_SCOPED_CATALOG_SELECTION_REFERENCE.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert_eq!(request.top_k, Some(50));
        assert_eq!(request.catalog_digest, Some([0xcd; 32]));
        assert_eq!(request.catalog_selection_digest, Some([0xef; 32]));
        assert_eq!(request.scoped_candidate_ordinals, vec![0, 4, 300]);
        assert!(request.candidates.is_empty());

        frame[4..6]
            .copy_from_slice(&VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE.to_le_bytes());
        let persistent = parse(&frame).unwrap();
        assert_eq!(
            persistent.protocol_version,
            VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
        );
        assert_eq!(persistent.scoped_candidate_ordinals, vec![0, 4, 300]);
    }

    #[test]
    fn catalog_registration_carries_public_id_descriptor_pairs() {
        let contract = "model@1";
        let mut body = Vec::new();
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&2_u32.to_le_bytes());
        body.extend_from_slice(&[2, 1, 1, 0]);
        body.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        body.extend_from_slice(&0_u32.to_le_bytes());
        body.extend_from_slice(&0_i32.to_le_bytes());
        body.extend_from_slice(&4_000_u32.to_le_bytes());
        body.extend_from_slice(&6_u32.to_le_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(contract.as_bytes());
        body.extend_from_slice(b"tenant");
        body.extend_from_slice(&[0_u8; 4]);
        body.push(2);
        body.extend_from_slice(&[0xef; 32]);
        for (public_id, rows, digest) in [(7_i64, 3_u32, 0xab), (10, 4, 0xbc)] {
            body.extend_from_slice(&public_id.to_le_bytes());
            body.extend_from_slice(&rows.to_le_bytes());
            body.extend_from_slice(&[digest; 32]);
        }
        let mut frame = Vec::new();
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_CATALOG_LOGICAL_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);

        let request = parse(&frame).unwrap();
        assert!(request.catalog_registration);
        assert_eq!(request.catalog_public_ids, vec![7, 10]);
        assert_eq!(request.candidates.len(), 2);
        assert_eq!(request.candidates[1].candidate_id, 1);
        assert_eq!(request.candidates[1].rows, 4);
    }

    #[test]
    fn pq_profile_without_a_contract_fails_closed() {
        assert!(
            parse(&profiled_frame(
                VERSION_PROFILED_EXTERNAL,
                4,
                0,
                4_000,
                "tenant-a"
            ))
            .is_err()
        );
    }

    #[test]
    fn response_uses_the_request_protocol_version() {
        let response = success(VERSION_SCHEDULED_EXTERNAL, 9, &[(1, 0.5)]);
        assert_eq!(
            u16::from_le_bytes(response[4..6].try_into().unwrap()),
            VERSION_SCHEDULED_EXTERNAL
        );
    }
}
