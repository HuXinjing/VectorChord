// This software is licensed under the repository's dual license model.

//! Stable client-side builders for the TileMaxSim wire contract.
//!
//! Applications should normally use VectorChord's SQL API. This module exists
//! for VectorChord components that must speak to `vchord-tilemaxsimd`; it keeps
//! frame layout, catalog digests, response validation, and error classification
//! out of callers.

use crate::protocol::{
    HEADER_BYTES, VERSION_CATALOG_LOGICAL_EXTERNAL, VERSION_CATALOG_SELECTION_REFERENCE,
    VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE,
    VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE,
    VERSION_SCOPED_CATALOG_SELECTION_REFERENCE,
};
use sha2::{Digest, Sha256};
use std::fmt;

const MAGIC: &[u8; 4] = b"VCTM";
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;
const MAX_CANDIDATES: usize = 65_536;
const MAX_MODEL_CONTRACT_BYTES: usize = 512;
const MAX_TENANT_BYTES: usize = 256;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDtype {
    Float32 = 1,
    Float16 = 2,
    Fp8E4m3 = 3,
}

impl TensorDtype {
    fn scalar_bytes(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Fp8E4m3 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ScoringProfile {
    ExactFp16 = 1,
    Int8RowScaled = 2,
    Fp8E4m3RowScaled = 3,
    Pq = 4,
    OpqRpq = 5,
    Fp8E4m3Raw = 6,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogDescriptor {
    pub public_id: i64,
    pub rows: u32,
    /// Raw SHA-256 digest bytes of the immutable tensor object.
    pub digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct CatalogRequest<'a> {
    pub request_id: u64,
    pub model_contract: &'a str,
    pub tenant: &'a str,
    pub priority: i32,
    pub timeout_ms: u32,
    pub query_rows: u32,
    pub dimension: u32,
    pub query_dtype: TensorDtype,
    pub candidate_dtype: TensorDtype,
    pub scoring_profile: ScoringProfile,
    pub quantization_contract: Option<&'a str>,
    pub top_k: usize,
    pub query: &'a [u8],
    /// Stable catalog revision. The wire carries its SHA-256 digest.
    pub catalog_revision: &'a str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub candidate_id: u32,
    pub similarity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoreResponse {
    pub protocol_version: u16,
    pub request_id: u64,
    pub scores: Vec<Score>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SdkError {
    InvalidRequest(&'static str),
    InvalidResponse(&'static str),
    CatalogMiss,
    ManifestMiss,
    Remote { status: u32, message: String },
}

impl fmt::Display for SdkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::InvalidResponse(message) => {
                formatter.write_str(message)
            }
            Self::CatalogMiss => formatter.write_str("descriptor catalog miss"),
            Self::ManifestMiss => formatter.write_str("descriptor manifest miss"),
            Self::Remote { status, message } => {
                write!(formatter, "TileMaxSim status {status}: {message}")
            }
        }
    }
}

impl std::error::Error for SdkError {}

pub fn catalog_digest(revision: &str) -> [u8; 32] {
    Sha256::digest(revision.as_bytes()).into()
}

pub fn selection_digest(public_ids: &[i64]) -> Result<[u8; 32], SdkError> {
    validate_public_ids(public_ids)?;
    let mut digest = Sha256::new();
    for public_id in public_ids {
        digest.update(public_id.to_le_bytes());
    }
    Ok(digest.finalize().into())
}

/// Encode a v10 catalog registration. Registration is additive for one
/// immutable catalog revision and is the fallback after a typed catalog miss.
pub fn encode_catalog_registration(
    request: &CatalogRequest<'_>,
    descriptors: &[CatalogDescriptor],
) -> Result<Vec<u8>, SdkError> {
    if descriptors.is_empty() || descriptors.len() > MAX_CANDIDATES {
        return Err(SdkError::InvalidRequest(
            "catalog descriptors must be nonempty and bounded",
        ));
    }
    let public_ids: Vec<i64> = descriptors.iter().map(|value| value.public_id).collect();
    validate_public_ids(&public_ids)?;
    validate_request(request, descriptors.len())?;
    let mut frame = common_prefix(request, descriptors.len())?;
    frame.push(2);
    frame.extend_from_slice(&catalog_digest(request.catalog_revision));
    for descriptor in descriptors {
        if descriptor.rows == 0 {
            return Err(SdkError::InvalidRequest(
                "catalog descriptor rows must be positive",
            ));
        }
        frame.extend_from_slice(&descriptor.public_id.to_le_bytes());
        frame.extend_from_slice(&descriptor.rows.to_le_bytes());
        frame.extend_from_slice(&descriptor.digest);
    }
    finish_frame(
        &mut frame,
        VERSION_CATALOG_LOGICAL_EXTERNAL,
        request.request_id,
    )?;
    Ok(frame)
}

/// Encode a v10 explicit selection. Use this to populate the daemon's bounded
/// selection cache after a v11/v14 reference miss.
pub fn encode_catalog_selection(
    request: &CatalogRequest<'_>,
    public_ids: &[i64],
) -> Result<Vec<u8>, SdkError> {
    validate_public_ids(public_ids)?;
    validate_request(request, public_ids.len())?;
    let mut frame = common_prefix(request, public_ids.len())?;
    frame.push(1);
    frame.extend_from_slice(&catalog_digest(request.catalog_revision));
    encode_delta_values(&mut frame, public_ids.iter().map(|value| *value as u64))?;
    finish_frame(
        &mut frame,
        VERSION_CATALOG_LOGICAL_EXTERNAL,
        request.request_id,
    )?;
    Ok(frame)
}

/// Encode the compact catalog-selection reference used on the hot path.
/// `scoped_public_ids`, when present, must be a nonempty subset of the sorted
/// candidate set. `persistent` selects v13/v14 so callers may reuse a socket.
pub fn encode_catalog_selection_reference(
    request: &CatalogRequest<'_>,
    public_ids: &[i64],
    scoped_public_ids: Option<&[i64]>,
    persistent: bool,
) -> Result<Vec<u8>, SdkError> {
    validate_public_ids(public_ids)?;
    validate_request(request, public_ids.len())?;
    let scoped = scoped_public_ids
        .map(|values| scoped_ordinals(public_ids, values))
        .transpose()?;
    let version = match (scoped.is_some(), persistent) {
        (false, false) => VERSION_CATALOG_SELECTION_REFERENCE,
        (true, false) => VERSION_SCOPED_CATALOG_SELECTION_REFERENCE,
        (true, true) => VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE,
        (false, true) => VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE,
    };
    let mut frame = common_prefix(request, public_ids.len())?;
    frame.push(if scoped.is_some() { 4 } else { 3 });
    frame.extend_from_slice(&catalog_digest(request.catalog_revision));
    frame.extend_from_slice(&selection_digest(public_ids)?);
    if let Some(ordinals) = scoped {
        frame.extend_from_slice(&(ordinals.len() as u32).to_le_bytes());
        encode_delta_values(
            &mut frame,
            ordinals.into_iter().map(|ordinal| u64::from(ordinal) + 1),
        )?;
    }
    finish_frame(&mut frame, version, request.request_id)?;
    Ok(frame)
}

pub fn decode_response(frame: &[u8]) -> Result<ScoreResponse, SdkError> {
    if frame.len() < HEADER_BYTES || &frame[..4] != MAGIC {
        return Err(SdkError::InvalidResponse(
            "invalid TileMaxSim response header",
        ));
    }
    let version = u16::from_le_bytes(frame[4..6].try_into().unwrap());
    let kind = u16::from_le_bytes(frame[6..8].try_into().unwrap());
    let request_id = u64::from_le_bytes(frame[8..16].try_into().unwrap());
    let body_len = usize::try_from(u64::from_le_bytes(frame[16..24].try_into().unwrap()))
        .map_err(|_| SdkError::InvalidResponse("TileMaxSim response is too large"))?;
    if kind != RESPONSE_KIND
        || body_len > MAX_RESPONSE_BYTES
        || body_len != frame.len() - HEADER_BYTES
    {
        return Err(SdkError::InvalidResponse(
            "invalid TileMaxSim response length or kind",
        ));
    }
    if body_len < 8 {
        return Err(SdkError::InvalidResponse("truncated TileMaxSim response"));
    }
    let status = u32::from_le_bytes(frame[24..28].try_into().unwrap());
    let count = u32::from_le_bytes(frame[28..32].try_into().unwrap()) as usize;
    if status != 0 {
        if count > body_len - 8 {
            return Err(SdkError::InvalidResponse(
                "truncated TileMaxSim error response",
            ));
        }
        let message = std::str::from_utf8(&frame[32..32 + count])
            .map_err(|_| SdkError::InvalidResponse("TileMaxSim error is not UTF-8"))?
            .to_owned();
        return Err(match message.as_str() {
            "descriptor catalog miss" => SdkError::CatalogMiss,
            "descriptor manifest miss" => SdkError::ManifestMiss,
            _ => SdkError::Remote { status, message },
        });
    }
    let expected = 8_usize
        .checked_add(
            count
                .checked_mul(8)
                .ok_or(SdkError::InvalidResponse("score count overflow"))?,
        )
        .ok_or(SdkError::InvalidResponse("score count overflow"))?;
    if body_len != expected {
        return Err(SdkError::InvalidResponse(
            "TileMaxSim score count disagrees with response length",
        ));
    }
    let mut scores = Vec::with_capacity(count);
    for bytes in frame[32..].chunks_exact(8) {
        let candidate_id = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let similarity = f32::from_le_bytes(bytes[4..].try_into().unwrap());
        if !similarity.is_finite() {
            return Err(SdkError::InvalidResponse(
                "TileMaxSim returned a non-finite score",
            ));
        }
        scores.push(Score {
            candidate_id,
            similarity,
        });
    }
    Ok(ScoreResponse {
        protocol_version: version,
        request_id,
        scores,
    })
}

fn common_prefix(
    request: &CatalogRequest<'_>,
    candidate_count: usize,
) -> Result<Vec<u8>, SdkError> {
    let quantization = request.quantization_contract.unwrap_or("");
    let capacity = HEADER_BYTES
        .checked_add(request.query.len())
        .and_then(|value| value.checked_add(request.model_contract.len()))
        .and_then(|value| value.checked_add(request.tenant.len()))
        .and_then(|value| value.checked_add(quantization.len()))
        .and_then(|value| value.checked_add(128))
        .ok_or(SdkError::InvalidRequest("request size overflow"))?;
    let mut frame = Vec::with_capacity(capacity);
    frame.resize(HEADER_BYTES, 0);
    frame.extend_from_slice(&request.dimension.to_le_bytes());
    frame.extend_from_slice(&request.query_rows.to_le_bytes());
    frame.extend_from_slice(&(candidate_count as u32).to_le_bytes());
    frame.push(request.query_dtype as u8);
    frame.push(1);
    frame.push(request.scoring_profile as u8);
    frame.push(if request.query_dtype == request.candidate_dtype {
        0
    } else {
        request.candidate_dtype as u8
    });
    frame.extend_from_slice(&(request.model_contract.len() as u32).to_le_bytes());
    frame.extend_from_slice(&(quantization.len() as u32).to_le_bytes());
    frame.extend_from_slice(&request.priority.to_le_bytes());
    frame.extend_from_slice(&request.timeout_ms.to_le_bytes());
    frame.extend_from_slice(&(request.tenant.len() as u32).to_le_bytes());
    frame.extend_from_slice(&(request.top_k as u32).to_le_bytes());
    frame.extend_from_slice(request.model_contract.as_bytes());
    frame.extend_from_slice(quantization.as_bytes());
    frame.extend_from_slice(request.tenant.as_bytes());
    frame.extend_from_slice(request.query);
    Ok(frame)
}

fn finish_frame(frame: &mut [u8], version: u16, request_id: u64) -> Result<(), SdkError> {
    let body_len = frame
        .len()
        .checked_sub(HEADER_BYTES)
        .ok_or(SdkError::InvalidRequest("request length underflow"))?;
    frame[..4].copy_from_slice(MAGIC);
    frame[4..6].copy_from_slice(&version.to_le_bytes());
    frame[6..8].copy_from_slice(&REQUEST_KIND.to_le_bytes());
    frame[8..16].copy_from_slice(&request_id.to_le_bytes());
    frame[16..24].copy_from_slice(&(body_len as u64).to_le_bytes());
    Ok(())
}

fn validate_request(request: &CatalogRequest<'_>, candidate_count: usize) -> Result<(), SdkError> {
    if candidate_count == 0 || candidate_count > MAX_CANDIDATES {
        return Err(SdkError::InvalidRequest(
            "candidate set must be nonempty and bounded",
        ));
    }
    validate_text(
        request.model_contract,
        MAX_MODEL_CONTRACT_BYTES,
        "model contract is invalid",
    )?;
    validate_text(request.tenant, MAX_TENANT_BYTES, "tenant is invalid")?;
    if !(-100..=100).contains(&request.priority) || !(1..=600_000).contains(&request.timeout_ms) {
        return Err(SdkError::InvalidRequest(
            "scheduler priority or timeout is invalid",
        ));
    }
    if request.query_rows == 0 || request.dimension == 0 || request.dimension > 60_000 {
        return Err(SdkError::InvalidRequest("query tensor shape is invalid"));
    }
    let expected = (request.query_rows as usize)
        .checked_mul(request.dimension as usize)
        .and_then(|value| value.checked_mul(request.query_dtype.scalar_bytes()))
        .ok_or(SdkError::InvalidRequest("query tensor size overflow"))?;
    if request.query.len() != expected {
        return Err(SdkError::InvalidRequest(
            "query payload disagrees with its shape and dtype",
        ));
    }
    if request.top_k == 0 || request.top_k > candidate_count || request.top_k > u32::MAX as usize {
        return Err(SdkError::InvalidRequest(
            "top_k must be between one and candidate count",
        ));
    }
    let pq = matches!(
        request.scoring_profile,
        ScoringProfile::Pq | ScoringProfile::OpqRpq
    );
    if pq != request.quantization_contract.is_some() {
        return Err(SdkError::InvalidRequest(
            "PQ profiles require exactly one quantization contract",
        ));
    }
    if let Some(contract) = request.quantization_contract
        && (contract.len() != 69
            || !contract.starts_with("qtc1-")
            || !contract[5..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(SdkError::InvalidRequest("quantization contract is invalid"));
    }
    if request.catalog_revision.is_empty() {
        return Err(SdkError::InvalidRequest(
            "catalog revision must not be empty",
        ));
    }
    Ok(())
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), SdkError> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        Err(SdkError::InvalidRequest(message))
    } else {
        Ok(())
    }
}

fn validate_public_ids(public_ids: &[i64]) -> Result<(), SdkError> {
    if public_ids.is_empty() || public_ids.len() > MAX_CANDIDATES {
        return Err(SdkError::InvalidRequest(
            "catalog public IDs must be nonempty and bounded",
        ));
    }
    if public_ids
        .iter()
        .copied()
        .try_fold(0_i64, |previous, current| {
            (current > previous).then_some(current)
        })
        .is_none()
    {
        return Err(SdkError::InvalidRequest(
            "catalog public IDs must be positive and strictly increasing",
        ));
    }
    Ok(())
}

fn scoped_ordinals(public_ids: &[i64], scoped: &[i64]) -> Result<Vec<u32>, SdkError> {
    validate_public_ids(scoped)?;
    let mut result = Vec::with_capacity(scoped.len());
    for public_id in scoped {
        let ordinal = public_ids.binary_search(public_id).map_err(|_| {
            SdkError::InvalidRequest("scoped public IDs must be a subset of candidates")
        })?;
        result.push(ordinal as u32);
    }
    Ok(result)
}

fn encode_delta_values(
    frame: &mut Vec<u8>,
    values: impl IntoIterator<Item = u64>,
) -> Result<(), SdkError> {
    let mut previous = 0_u64;
    for current in values {
        let mut delta = current
            .checked_sub(previous)
            .filter(|value| *value > 0)
            .ok_or(SdkError::InvalidRequest(
                "delta values must be positive and strictly increasing",
            ))?;
        while delta >= 0x80 {
            frame.push((delta as u8 & 0x7f) | 0x80);
            delta >>= 7;
        }
        frame.push(delta as u8);
        previous = current;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol;

    fn request<'a>(query: &'a [u8]) -> CatalogRequest<'a> {
        CatalogRequest {
            request_id: 41,
            model_contract: "colqwen@1",
            tenant: "tenant-7",
            priority: 5,
            timeout_ms: 2_000,
            query_rows: 2,
            dimension: 2,
            query_dtype: TensorDtype::Float16,
            candidate_dtype: TensorDtype::Fp8E4m3,
            scoring_profile: ScoringProfile::Fp8E4m3Raw,
            quantization_contract: None,
            top_k: 2,
            query,
            catalog_revision: "catalog-17",
        }
    }

    #[test]
    fn sdk_catalog_frames_are_accepted_by_the_daemon_parser() {
        let query = [0_u8; 8];
        let request = request(&query);
        let ids = [11, 20, 42];

        let explicit = encode_catalog_selection(&request, &ids).unwrap();
        let parsed = protocol::parse(&explicit).unwrap();
        assert_eq!(parsed.protocol_version, VERSION_CATALOG_LOGICAL_EXTERNAL);
        assert_eq!(parsed.catalog_public_ids, ids);

        let reference =
            encode_catalog_selection_reference(&request, &ids, Some(&[20, 42]), true).unwrap();
        let parsed = protocol::parse(&reference).unwrap();
        assert_eq!(
            parsed.protocol_version,
            VERSION_PERSISTENT_SCOPED_CATALOG_SELECTION_REFERENCE
        );
        assert_eq!(parsed.scoped_candidate_ordinals, [1, 2]);
        assert_eq!(
            parsed.catalog_selection_digest,
            Some(selection_digest(&ids).unwrap())
        );
    }

    #[test]
    fn sdk_registration_frame_is_accepted_by_the_daemon_parser() {
        let query = [0_u8; 8];
        let request = request(&query);
        let descriptors = [
            CatalogDescriptor {
                public_id: 11,
                rows: 3,
                digest: [0xaa; 32],
            },
            CatalogDescriptor {
                public_id: 20,
                rows: 4,
                digest: [0xbb; 32],
            },
        ];
        let parsed =
            protocol::parse(&encode_catalog_registration(&request, &descriptors).unwrap()).unwrap();
        assert!(parsed.catalog_registration);
        assert_eq!(parsed.catalog_public_ids, [11, 20]);
        assert_eq!(parsed.candidates.len(), 2);
        assert_eq!(parsed.candidates[0].digest, hex::encode([0xaa; 32]));
    }

    #[test]
    fn response_decoder_accepts_partial_top_k_and_classifies_catalog_miss() {
        let response = protocol::success(
            VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE,
            41,
            &[(7, 0.9), (3, 0.8)],
        );
        let decoded = decode_response(&response).unwrap();
        assert_eq!(decoded.scores.len(), 2);
        assert_eq!(decoded.scores[0].candidate_id, 7);

        let miss = protocol::failure(
            VERSION_PERSISTENT_CATALOG_SELECTION_REFERENCE,
            41,
            4,
            "descriptor catalog miss",
        );
        assert_eq!(decode_response(&miss), Err(SdkError::CatalogMiss));
    }

    #[test]
    fn sdk_rejects_unsorted_ids_and_wrong_query_lengths() {
        let query = [0_u8; 8];
        assert!(matches!(
            encode_catalog_selection(&request(&query), &[20, 11]),
            Err(SdkError::InvalidRequest(_))
        ));
        let short = [0_u8; 6];
        assert!(matches!(
            encode_catalog_selection(&request(&short), &[11, 20]),
            Err(SdkError::InvalidRequest(_))
        ));
    }
}
