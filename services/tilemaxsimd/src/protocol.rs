// Copyright (c) 2026 HuXinjing

use anyhow::{Result, anyhow, bail};
use std::collections::HashSet;

pub const HEADER_BYTES: usize = 24;
pub const VERSION_EXTERNAL: u16 = 2;
const MAGIC: &[u8; 4] = b"VCTM";
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;

#[derive(Clone, Debug)]
pub struct Descriptor {
    pub candidate_id: u32,
    pub contract: String,
    pub digest: String,
    pub rows: u32,
    pub dimension: u32,
    pub dtype: u8,
}

#[derive(Debug)]
pub struct Request {
    pub request_id: u64,
    pub query_rows: u32,
    pub dimension: u32,
    pub dtype: u8,
    pub query: Vec<u8>,
    pub candidates: Vec<Descriptor>,
    pub tensor_tokens: usize,
    pub tensor_bytes: usize,
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
    } else {
        for scalar in payload.chunks_exact(2) {
            let bits = u16::from_le_bytes(scalar.try_into().unwrap());
            if bits & 0x7c00 == 0x7c00 {
                bail!("query contains a non-finite value");
            }
        }
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
    if version != VERSION_EXTERNAL || kind != REQUEST_KIND {
        bail!("Rust daemon requires TileMaxSim external protocol v2");
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
    let reserved = reader.u16()?;
    let contract_bytes = reader.u32()? as usize;
    if scoring != 1 || reserved != 0 {
        bail!("unsupported scoring function or reserved bits");
    }
    if candidate_count > 65_536 {
        bail!("too many candidates");
    }
    let contract = reader.text(contract_bytes, 512, "model contract")?;
    let query_bytes = tensor_bytes(query_rows, dimension, dtype)?;
    let query = reader.take(query_bytes)?.to_vec();
    validate_finite(&query, dtype)?;
    let mut total_tokens = query_rows as usize;
    let mut total_bytes = query_bytes;
    let mut candidate_ids = HashSet::new();
    let mut candidates = Vec::with_capacity(candidate_count as usize);
    for _ in 0..candidate_count {
        let candidate_id = reader.u32()?;
        let rows = reader.u32()?;
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
        if total_tokens > 1_000_000 || total_bytes > 1024 * 1024 * 1024 {
            bail!("request exceeds tensor limits");
        }
        candidates.push(Descriptor {
            candidate_id,
            contract: contract.clone(),
            digest: digest.to_owned(),
            rows,
            dimension,
            dtype,
        });
    }
    reader.finish()?;
    Ok(Request {
        request_id,
        query_rows,
        dimension,
        dtype,
        query,
        candidates,
        tensor_tokens: total_tokens,
        tensor_bytes: total_bytes,
    })
}

pub fn success(request_id: u64, results: &[(u32, f32)]) -> Vec<u8> {
    let body_bytes = 8 + results.len() * 8;
    let mut frame = Vec::with_capacity(HEADER_BYTES + body_bytes);
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION_EXTERNAL.to_le_bytes());
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

pub fn failure(request_id: u64, status: u32, message: &str) -> Vec<u8> {
    let message = message.as_bytes();
    let message = &message[..message.len().min(64 * 1024)];
    let body_bytes = 8 + message.len();
    let mut frame = Vec::with_capacity(HEADER_BYTES + body_bytes);
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION_EXTERNAL.to_le_bytes());
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

    fn empty_request() -> Vec<u8> {
        let contract = b"health";
        let body_bytes = 4 + 4 + 4 + 1 + 1 + 2 + 4 + contract.len() + 2;
        let mut frame = Vec::with_capacity(HEADER_BYTES + body_bytes);
        frame.extend_from_slice(MAGIC);
        frame.extend_from_slice(&VERSION_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&7_u64.to_le_bytes());
        frame.extend_from_slice(&(body_bytes as u64).to_le_bytes());
        frame.extend_from_slice(&1_u32.to_le_bytes());
        frame.extend_from_slice(&1_u32.to_le_bytes());
        frame.extend_from_slice(&0_u32.to_le_bytes());
        frame.push(2);
        frame.push(1);
        frame.extend_from_slice(&0_u16.to_le_bytes());
        frame.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        frame.extend_from_slice(contract);
        frame.extend_from_slice(&0_u16.to_le_bytes());
        frame
    }

    #[test]
    fn zero_candidate_probe_is_a_valid_bounded_request() {
        let request = parse(&empty_request()).unwrap();
        assert_eq!(request.request_id, 7);
        assert_eq!(request.query_rows, 1);
        assert_eq!(request.dimension, 1);
        assert_eq!(request.dtype, 2);
        assert!(request.candidates.is_empty());
        assert_eq!(request.tensor_tokens, 1);
        assert_eq!(request.tensor_bytes, 2);
    }

    #[test]
    fn arbitrary_short_frames_fail_without_panicking() {
        // Fixed seed and LCG constants make malformed-frame coverage reproducible.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for length in 0..512 {
            let mut frame = vec![0_u8; length];
            for byte in &mut frame {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *byte = (state >> 32) as u8;
            }
            let _ = parse(&frame);
        }
    }
}
