// This software is licensed under a dual license model:
// GNU Affero General Public License v3 or Elastic License v2.
// Copyright (c) 2026 Hu Xinjing

//! Immutable production formats for PQ-family TileMaxSim artifacts.

use anyhow::{Result, anyhow, bail};
use sha2::{Digest, Sha256};

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
    if bytes.len() != HEADER_BYTES.checked_add(length).ok_or_else(|| anyhow!("length overflow"))? {
        bail!("quantization artifact length mismatch");
    }
    let payload = &bytes[HEADER_BYTES..];
    let actual: [u8; 32] = Sha256::digest(payload).into();
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
    if dimension == 0 || stages == 0 || subspaces == 0 || centroids < 2
        || dimension % u32::from(subspaces) != 0 || stages > 16
        || rotation_mask >> stages != 0
    {
        bail!("invalid quantizer shape");
    }
    let rotation_values = rotation_mask.count_ones() as usize
        * dimension as usize * dimension as usize;
    let codebook_values = stages as usize * subspaces as usize * centroids as usize
        * (dimension as usize / subspaces as usize);
    let expected = rotation_values.checked_add(codebook_values)
        .and_then(|value| value.checked_mul(4)).ok_or_else(|| anyhow!("quantizer size overflow"))?;
    if payload.len() != expected || payload.chunks_exact(4).any(|value| {
        !f32::from_le_bytes(value.try_into().unwrap()).is_finite()
    }) {
        bail!("quantizer payload disagrees with its shape or contains non-finite values");
    }
    Ok(QuantizerArtifact {
        contract_digest: bytes[28..60].try_into().unwrap(), dimension, stages,
        subspaces, centroids, rotation_mask, payload: payload.to_vec(),
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
        source_digest: bytes[60..92].try_into().unwrap(), rows, dimension,
        stages, subspaces, codes: codes.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(magic: &[u8; 4], payload: &[u8], checksum_offset: usize) -> Vec<u8> {
        let mut result = vec![0_u8; HEADER_BYTES];
        result[..4].copy_from_slice(magic);
        result[4..6].copy_from_slice(&VERSION.to_le_bytes());
        result[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        result[checksum_offset..checksum_offset + 32].copy_from_slice(&Sha256::digest(payload));
        result.extend_from_slice(payload);
        result
    }

    #[test]
    fn parses_and_authenticates_pq_codes() {
        let mut value = frame(CODES_MAGIC, &[1, 2, 3, 4], 92);
        value[8..12].copy_from_slice(&2_u32.to_le_bytes());
        value[12..16].copy_from_slice(&8_u32.to_le_bytes());
        value[24..26].copy_from_slice(&1_u16.to_le_bytes());
        value[26..28].copy_from_slice(&2_u16.to_le_bytes());
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
        let parsed = parse_quantizer(&value).unwrap();
        assert_eq!((parsed.stages, parsed.subspaces, parsed.centroids), (1, 1, 2));
    }
}
