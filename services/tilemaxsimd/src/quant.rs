// This software is licensed under a dual license model:
// GNU Affero General Public License v3 or Elastic License v2.
// Copyright (c) 2026 Hu Xinjing

//! Immutable production formats for PQ-family TileMaxSim artifacts.

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

const QUANTIZER_MAGIC: &[u8; 4] = b"VCTQ";
const CODES_MAGIC: &[u8; 4] = b"VCTC";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 128;

#[derive(Clone, Debug)]
pub struct QuantizerArtifact {
    pub contract_digest: [u8; 32],
    pub dimension: u32,
    pub stages: u16,
    pub subspaces: u16,
    pub centroids: u16,
    pub rotation_mask: u16,
    /// Stage-major rotations followed by stage/subspace/centroid vectors, all LE f32.
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CodesArtifact {
    pub contract_digest: [u8; 32],
    pub source_digest: [u8; 32],
    pub rows: u32,
    pub dimension: u32,
    pub stages: u16,
    pub subspaces: u16,
    /// `[row][stage][subspace]` uint8 codes.
    pub codes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ActiveQuantizer {
    pub generation: u64,
    /// All contracts active in the same immutable registry snapshot that
    /// authorized this request.  Keeping these together prevents a rollback
    /// between two state reads from reclaiming the request's quantizer.
    pub active_contract_ids: std::collections::HashSet<String>,
    pub contract_id: String,
    pub artifact_root: PathBuf,
    pub quantizer: QuantizerArtifact,
}

#[derive(Deserialize)]
struct RegistryState {
    version: u32,
    generation: u64,
    scopes: HashMap<String, ScopeState>,
}

#[derive(Deserialize)]
struct ScopeState {
    active: Option<String>,
}

#[derive(Deserialize)]
struct ContractRecord {
    version: u32,
    contract_id: String,
    contract: ContractBody,
    artifact_root: PathBuf,
}

#[derive(Deserialize)]
struct ContractBody {
    model_contract: String,
    encoding: String,
    dimension: u32,
    subspaces: u16,
    centroids: u16,
    residual_stages: u16,
    opq_iterations: u16,
    quantizer_checksum: String,
}

#[derive(Clone, Debug)]
pub struct QuantizationRegistry {
    root: PathBuf,
}

impl QuantizationRegistry {
    pub fn open(root: PathBuf) -> Result<Self> {
        if !root.is_absolute() || fs::symlink_metadata(&root)?.file_type().is_symlink() {
            bail!("quantization registry root must be an absolute, non-symlink directory");
        }
        Ok(Self { root })
    }

    pub fn resolve_active(
        &self,
        contract_id: &str,
        model_contract: &str,
        profile: crate::protocol::ScoringProfile,
    ) -> Result<ActiveQuantizer> {
        let encoding = match profile {
            crate::protocol::ScoringProfile::Pq => "pq",
            crate::protocol::ScoringProfile::OpqRpq => "pq",
            _ => bail!("quantization registry is only valid for PQ-family profiles"),
        };
        let state: RegistryState =
            serde_json::from_slice(&fs::read(self.root.join("state.json"))?)?;
        if state.version != 2 {
            bail!("quantization registry must be migrated to state schema v2");
        }
        let scope = format!("{model_contract}\u{1f}{encoding}");
        let active = state
            .scopes
            .get(&scope)
            .and_then(|scope| scope.active.as_deref());
        if active != Some(contract_id) {
            bail!("requested quantization contract is not active for this model and encoding");
        }
        let record_path = self
            .root
            .join("contracts")
            .join(format!("{contract_id}.json"));
        if fs::symlink_metadata(&record_path)?.file_type().is_symlink() {
            bail!("quantization contract record must not be a symlink");
        }
        let record: ContractRecord = serde_json::from_slice(&fs::read(record_path)?)?;
        if record.version != 1
            || record.contract_id != contract_id
            || record.contract.model_contract != model_contract
            || record.contract.encoding != encoding
        {
            bail!("quantization contract record disagrees with the active request");
        }
        let expects_opq_rpq =
            record.contract.opq_iterations > 0 || record.contract.residual_stages > 1;
        if (profile == crate::protocol::ScoringProfile::OpqRpq) != expects_opq_rpq {
            bail!("PQ scoring profile disagrees with contract rotation/residual stages");
        }
        let artifact_root = record.artifact_root;
        if !artifact_root.is_absolute()
            || fs::symlink_metadata(&artifact_root)?
                .file_type()
                .is_symlink()
        {
            bail!("quantization artifact root must be absolute and must not be a symlink");
        }
        let quantizer_path = artifact_root.join("quantizer.vctq");
        if fs::symlink_metadata(&quantizer_path)?
            .file_type()
            .is_symlink()
        {
            bail!("quantizer artifact must not be a symlink");
        }
        let quantizer = parse_quantizer(&fs::read(quantizer_path)?)?;
        let digest = hex::decode(
            contract_id
                .strip_prefix("qtc1-")
                .ok_or_else(|| anyhow!("invalid contract ID"))?,
        )?;
        if digest.as_slice() != quantizer.contract_digest
            || quantizer.dimension != record.contract.dimension
            || quantizer.subspaces != record.contract.subspaces
            || quantizer.centroids != record.contract.centroids
            || quantizer.stages != record.contract.residual_stages
        {
            bail!("quantizer artifact shape or identity disagrees with its contract");
        }
        if record.contract.quantizer_checksum.len() != 64
            || hex::encode(Sha256::digest(&quantizer.payload)) != record.contract.quantizer_checksum
        {
            bail!("quantizer payload checksum disagrees with its contract identity");
        }
        let active_contract_ids = state
            .scopes
            .values()
            .filter_map(|scope| scope.active.clone())
            .collect();
        Ok(ActiveQuantizer {
            generation: state.generation,
            active_contract_ids,
            contract_id: contract_id.to_owned(),
            artifact_root,
            quantizer,
        })
    }

    pub fn active_contract_ids(&self) -> Result<std::collections::HashSet<String>> {
        let state: RegistryState =
            serde_json::from_slice(&fs::read(self.root.join("state.json"))?)?;
        if state.version != 2 {
            bail!("quantization registry must use state schema v2");
        }
        Ok(state
            .scopes
            .into_values()
            .filter_map(|scope| scope.active)
            .collect())
    }

    pub fn load_codes(
        &self,
        active: &ActiveQuantizer,
        source_digest: &str,
    ) -> Result<CodesArtifact> {
        if source_digest.len() != 64
            || !source_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("invalid source tensor digest");
        }
        let path = active
            .artifact_root
            .join("codes")
            .join(&source_digest[..2])
            .join(format!("{source_digest}.vctc"));
        if fs::symlink_metadata(&path)?.file_type().is_symlink() {
            bail!("PQ code artifact must not be a symlink");
        }
        let codes = parse_codes(&fs::read(path)?)?;
        if codes.contract_digest != active.quantizer.contract_digest
            || hex::encode(codes.source_digest) != source_digest
            || codes.dimension != active.quantizer.dimension
            || codes.stages != active.quantizer.stages
            || codes.subspaces != active.quantizer.subspaces
        {
            bail!("PQ code artifact disagrees with the active quantizer or source tensor");
        }
        Ok(codes)
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn checked_payload<'a>(
    bytes: &'a [u8],
    magic: &[u8; 4],
    checksum_offset: usize,
    reserved: std::ops::Range<usize>,
) -> Result<&'a [u8]> {
    if bytes.len() < HEADER_BYTES || &bytes[..4] != magic || u16_at(bytes, 4) != VERSION {
        bail!("invalid quantization artifact header");
    }
    if bytes[6..8] != [0, 0] || bytes[reserved].iter().any(|byte| *byte != 0) {
        bail!("unsupported quantization artifact flags or reserved bytes");
    }
    let length = usize::try_from(u64_at(bytes, 16)).map_err(|_| anyhow!("payload too large"))?;
    if bytes.len()
        != HEADER_BYTES
            .checked_add(length)
            .ok_or_else(|| anyhow!("length overflow"))?
    {
        bail!("quantization artifact length mismatch");
    }
    let payload = &bytes[HEADER_BYTES..];
    let mut hasher = Sha256::new();
    hasher.update(&bytes[..checksum_offset]);
    hasher.update(payload);
    let actual: [u8; 32] = hasher.finalize().into();
    if actual != bytes[checksum_offset..checksum_offset + 32] {
        bail!("quantization artifact checksum mismatch");
    }
    Ok(payload)
}

pub fn parse_quantizer(bytes: &[u8]) -> Result<QuantizerArtifact> {
    let payload = checked_payload(bytes, QUANTIZER_MAGIC, 60, 92..HEADER_BYTES)?;
    let dimension = u32_at(bytes, 8);
    let stages = u16_at(bytes, 12);
    let subspaces = u16_at(bytes, 14);
    let centroids = u16_at(bytes, 24);
    let rotation_mask = u16_at(bytes, 26);
    if dimension == 0
        || stages == 0
        || subspaces == 0
        || !(2..=256).contains(&centroids)
        || !dimension.is_multiple_of(u32::from(subspaces))
        || stages > 16
        || rotation_mask >> stages != 0
    {
        bail!("invalid quantizer shape");
    }
    let rotation_values =
        rotation_mask.count_ones() as usize * dimension as usize * dimension as usize;
    let codebook_values = stages as usize
        * subspaces as usize
        * centroids as usize
        * (dimension as usize / subspaces as usize);
    let expected = rotation_values
        .checked_add(codebook_values)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| anyhow!("quantizer size overflow"))?;
    if payload.len() != expected
        || payload
            .chunks_exact(4)
            .any(|value| !f32::from_le_bytes(value.try_into().unwrap()).is_finite())
    {
        bail!("quantizer payload disagrees with its shape or contains non-finite values");
    }
    Ok(QuantizerArtifact {
        contract_digest: bytes[28..60].try_into().unwrap(),
        dimension,
        stages,
        subspaces,
        centroids,
        rotation_mask,
        payload: payload.to_vec(),
    })
}

pub fn parse_codes(bytes: &[u8]) -> Result<CodesArtifact> {
    let codes = checked_payload(bytes, CODES_MAGIC, 92, 124..HEADER_BYTES)?;
    let rows = u32_at(bytes, 8);
    let dimension = u32_at(bytes, 12);
    let stages = u16_at(bytes, 24);
    let subspaces = u16_at(bytes, 26);
    let expected = rows as usize * stages as usize * subspaces as usize;
    if rows == 0 || dimension == 0 || stages == 0 || subspaces == 0 || codes.len() != expected {
        bail!("invalid PQ code artifact shape");
    }
    Ok(CodesArtifact {
        contract_digest: bytes[28..60].try_into().unwrap(),
        source_digest: bytes[60..92].try_into().unwrap(),
        rows,
        dimension,
        stages,
        subspaces,
        codes: codes.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frame(magic: &[u8; 4], payload: &[u8], checksum_offset: usize) -> Vec<u8> {
        let mut result = vec![0_u8; HEADER_BYTES];
        result[..4].copy_from_slice(magic);
        result[4..6].copy_from_slice(&VERSION.to_le_bytes());
        result[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        result.extend_from_slice(payload);
        let mut hasher = Sha256::new();
        hasher.update(&result[..checksum_offset]);
        hasher.update(payload);
        result[checksum_offset..checksum_offset + 32].copy_from_slice(&hasher.finalize());
        result
    }

    fn reseal(value: &mut [u8], checksum_offset: usize) {
        let mut hasher = Sha256::new();
        hasher.update(&value[..checksum_offset]);
        hasher.update(&value[HEADER_BYTES..]);
        value[checksum_offset..checksum_offset + 32].copy_from_slice(&hasher.finalize());
    }

    #[test]
    fn parses_and_authenticates_pq_codes() {
        let mut value = frame(CODES_MAGIC, &[1, 2, 3, 4], 92);
        value[8..12].copy_from_slice(&2_u32.to_le_bytes());
        value[12..16].copy_from_slice(&8_u32.to_le_bytes());
        value[24..26].copy_from_slice(&1_u16.to_le_bytes());
        value[26..28].copy_from_slice(&2_u16.to_le_bytes());
        reseal(&mut value, 92);
        assert_eq!(parse_codes(&value).unwrap().codes, [1, 2, 3, 4]);
        *value.last_mut().unwrap() ^= 1;
        assert!(parse_codes(&value).is_err());
    }

    #[test]
    fn parses_quantizer_shape_and_rejects_non_finite_codebooks() {
        // dimension=2, one stage/subspace, two 2-D centroids.
        let payload = [0.0_f32, 0.0, 1.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut value = frame(QUANTIZER_MAGIC, &payload, 60);
        value[8..12].copy_from_slice(&2_u32.to_le_bytes());
        value[12..14].copy_from_slice(&1_u16.to_le_bytes());
        value[14..16].copy_from_slice(&1_u16.to_le_bytes());
        value[24..26].copy_from_slice(&2_u16.to_le_bytes());
        reseal(&mut value, 60);
        let parsed = parse_quantizer(&value).unwrap();
        assert_eq!(
            (parsed.stages, parsed.subspaces, parsed.centroids),
            (1, 1, 2)
        );
    }

    #[test]
    fn registry_activation_and_rollback_are_observed_atomically() {
        let root = std::env::temp_dir().join(format!("vctm-quant-registry-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("contracts")).unwrap();
        let artifact = root.join("artifact");
        fs::create_dir_all(&artifact).unwrap();
        let digest = "a".repeat(64);
        let contract_id = format!("qtc1-{digest}");
        let payload = [0.0_f32, 0.0, 1.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut quantizer = frame(QUANTIZER_MAGIC, &payload, 60);
        quantizer[8..12].copy_from_slice(&2_u32.to_le_bytes());
        quantizer[12..14].copy_from_slice(&1_u16.to_le_bytes());
        quantizer[14..16].copy_from_slice(&1_u16.to_le_bytes());
        quantizer[24..26].copy_from_slice(&2_u16.to_le_bytes());
        quantizer[28..60].copy_from_slice(&hex::decode(&digest).unwrap());
        reseal(&mut quantizer, 60);
        fs::write(artifact.join("quantizer.vctq"), quantizer).unwrap();
        fs::write(
            root.join("contracts").join(format!("{contract_id}.json")),
            serde_json::to_vec(&json!({
                "version": 1, "contract_id": contract_id, "artifact_root": artifact,
                "contract": {"model_contract":"model@1","encoding":"pq","dimension":2,
                    "subspaces":1,"centroids":2,"residual_stages":1,"opq_iterations":0,
                    "quantizer_checksum": hex::encode(Sha256::digest(&payload))}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("state.json"), serde_json::to_vec(&json!({
            "version":2,"generation":1,"scopes":{"model@1\u{1f}pq":{"active":contract_id,"previous":null}}
        })).unwrap()).unwrap();
        let registry = QuantizationRegistry::open(root.clone()).unwrap();
        let active = registry
            .resolve_active(&contract_id, "model@1", crate::protocol::ScoringProfile::Pq)
            .unwrap();
        assert_eq!(active.generation, 1);
        let source = "b".repeat(64);
        let mut codes = frame(CODES_MAGIC, &[0, 1], 92);
        codes[8..12].copy_from_slice(&2_u32.to_le_bytes());
        codes[12..16].copy_from_slice(&2_u32.to_le_bytes());
        codes[24..26].copy_from_slice(&1_u16.to_le_bytes());
        codes[26..28].copy_from_slice(&1_u16.to_le_bytes());
        codes[28..60].copy_from_slice(&hex::decode(&digest).unwrap());
        codes[60..92].copy_from_slice(&hex::decode(&source).unwrap());
        reseal(&mut codes, 92);
        fs::create_dir_all(artifact.join("codes").join("bb")).unwrap();
        fs::write(
            artifact
                .join("codes")
                .join("bb")
                .join(format!("{source}.vctc")),
            codes,
        )
        .unwrap();
        assert_eq!(registry.load_codes(&active, &source).unwrap().codes, [0, 1]);
        fs::write(root.join("state.json"), serde_json::to_vec(&json!({
            "version":2,"generation":2,"scopes":{"model@1\u{1f}pq":{"active":null,"previous":contract_id}}
        })).unwrap()).unwrap();
        assert!(
            registry
                .resolve_active(&contract_id, "model@1", crate::protocol::ScoringProfile::Pq)
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
