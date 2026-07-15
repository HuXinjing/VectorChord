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

pub const HEADER_BYTES: usize = 24;
pub const VERSION_EXTERNAL: u16 = 2;
pub const VERSION_SCHEDULED_EXTERNAL: u16 = 3;
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

#[derive(Clone, Debug)]
pub struct Request {
    pub protocol_version: u16,
    pub request_id: u64,
    /// Logical scheduling domain. It is never used for authorization or
    /// tensor lookup; the caller must already have applied hard ACL filters.
    pub tenant: String,
    /// Higher values run first within the scheduler's fairness policy.
    pub priority: i32,
    /// Client-supplied end-to-end budget. Zero is used only by legacy v2.
    pub timeout_ms: u32,
    pub query_rows: u32,
    pub dimension: u32,
    pub dtype: u8,
    pub query: Vec<u8>,
    pub candidates: Vec<Descriptor>,
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
    if !matches!(version, VERSION_EXTERNAL | VERSION_SCHEDULED_EXTERNAL) || kind != REQUEST_KIND {
        bail!("Rust daemon requires TileMaxSim external protocol v2 or v3");
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
    let (priority, timeout_ms, tenant_bytes) = if version == VERSION_SCHEDULED_EXTERNAL {
        (reader.i32()?, reader.u32()?, reader.u32()? as usize)
    } else {
        (0, 0, 0)
    };
    if !(-100..=100).contains(&priority) {
        bail!("scheduler priority must be between -100 and 100");
    }
    if version == VERSION_SCHEDULED_EXTERNAL && !(1..=600_000).contains(&timeout_ms) {
        bail!("scheduler timeout must be between 1 and 600000 milliseconds");
    }
    let contract = reader.text(contract_bytes, 512, "model contract")?;
    let tenant = if version == VERSION_SCHEDULED_EXTERNAL {
        reader.text(tenant_bytes, 256, "scheduler tenant")?
    } else {
        "__default__".to_owned()
    };
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
        protocol_version: version,
        request_id,
        tenant,
        priority,
        timeout_ms,
        query_rows,
        dimension,
        dtype,
        query,
        candidates,
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
        body.extend_from_slice(&0_u16.to_le_bytes());
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
        frame.extend_from_slice(&VERSION_SCHEDULED_EXTERNAL.to_le_bytes());
        frame.extend_from_slice(&REQUEST_KIND.to_le_bytes());
        frame.extend_from_slice(&42_u64.to_le_bytes());
        frame.extend_from_slice(&(body.len() as u64).to_le_bytes());
        frame.extend_from_slice(&body);
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
    fn response_uses_the_request_protocol_version() {
        let response = success(VERSION_SCHEDULED_EXTERNAL, 9, &[(1, 0.5)]);
        assert_eq!(
            u16::from_le_bytes(response[4..6].try_into().unwrap()),
            VERSION_SCHEDULED_EXTERNAL
        );
    }
}
