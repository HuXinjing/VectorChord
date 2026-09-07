// This software is licensed under the repository's dual license model.

use crate::backend::{
    AcceleratorBackend, AdaptiveStatus, BackendCapabilities, BackendKind, DeviceInfo,
};
use anyhow::{Result, bail};
use std::collections::HashSet;

pub struct CpuBackend {
    arena: Vec<u8>,
    tensor_bytes: usize,
    info: DeviceInfo,
}

impl CpuBackend {
    pub fn create(ordinal: i32, total_bytes: usize, workspace_bytes: usize) -> Result<Self> {
        if ordinal != 0 {
            bail!("the CPU backend exposes exactly one device with ordinal 0");
        }
        if workspace_bytes == 0 || workspace_bytes >= total_bytes {
            bail!("invalid CPU arena configuration");
        }
        let tensor_bytes = total_bytes - workspace_bytes;
        let parallelism = std::thread::available_parallelism()
            .map(|value| value.get() as u32)
            .unwrap_or(1);
        Ok(Self {
            arena: vec![0; total_bytes],
            tensor_bytes,
            info: DeviceInfo {
                backend: BackendKind::Cpu,
                ordinal,
                name: std::env::consts::ARCH.to_owned(),
                architecture: std::env::consts::ARCH.to_owned(),
                driver_version: None,
                runtime_version: None,
                library_version: None,
                total_memory_bytes: None,
                compute_units: Some(parallelism),
                warp_size: None,
                memory_bus_width_bits: None,
                memory_clock_khz: None,
                shared_memory_per_block_bytes: None,
                persisting_l2_bytes: None,
                matrix_engine_workspace_bytes: None,
                document_tile_query_rows: None,
                compute_queue_priority: None,
                tuning_profile: cpu_tuning_profile().to_owned(),
                capabilities: BackendCapabilities {
                    kind: BackendKind::Cpu,
                    exact_fp16: true,
                    exact_fp32: true,
                    int8: false,
                    fp8_e4m3: false,
                    pq: false,
                    opq_rpq: false,
                    fused_multiquery: false,
                    matrix_engine: false,
                    asynchronous_copy: false,
                    unified_memory: true,
                    persisting_l2: false,
                },
            },
        })
    }
}

impl AcceleratorBackend for CpuBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    fn adaptive_status(&self) -> AdaptiveStatus {
        AdaptiveStatus {
            tensor_threshold_rows: u32::MAX,
            calibration_complete: true,
            ..AdaptiveStatus::default()
        }
    }

    fn ensure_quantizer(
        &mut self,
        _contract_id: &str,
        _payload: &[u8],
        _dimension: u32,
        _stages: u16,
        _subspaces: u16,
        _centroids: u16,
        _rotation_mask: u16,
    ) -> Result<()> {
        bail!("CPU backend does not support PQ-family profiles")
    }

    fn retain_quantizers(&mut self, _active: &HashSet<String>) {}

    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        for (offset, payload) in items {
            let start = usize::try_from(*offset)
                .map_err(|_| anyhow::anyhow!("CPU arena offset overflow"))?;
            let end = start
                .checked_add(payload.len())
                .ok_or_else(|| anyhow::anyhow!("CPU arena range overflow"))?;
            let destination = self
                .arena
                .get_mut(start..end)
                .ok_or_else(|| anyhow::anyhow!("CPU arena upload exceeds allocation"))?;
            destination.copy_from_slice(payload);
        }
        Ok(())
    }

    fn score(
        &mut self,
        query: &[u8],
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        if scoring_profile != 1 {
            bail!("CPU reference backend supports exact_fp16 only")
        }
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid CPU document metadata")
        }
        let scalar_bytes = match dtype {
            1 => 4,
            2 => 2,
            _ => bail!("unsupported CPU query dtype"),
        };
        let expected = query_rows as usize * dimension as usize * scalar_bytes;
        if query.len() != expected {
            bail!("CPU query shape disagrees with payload")
        }
        document_offsets
            .iter()
            .zip(document_rows)
            .map(|(offset, rows)| {
                exact_maxsim(
                    query,
                    query_rows,
                    dimension,
                    dtype,
                    &self.arena,
                    *offset,
                    *rows,
                )
            })
            .collect()
    }

    fn score_pq(
        &mut self,
        _contract_id: &str,
        _query: &[u8],
        _query_rows: u32,
        _dtype: u8,
        _document_offsets: &[u64],
        _document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        bail!("CPU backend does not support PQ-family profiles")
    }

    fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if scoring_profile != 1 {
            bail!("CPU backend does not support quantized batch scoring");
        }
        let scalar_bytes = match dtype {
            1 => 4,
            2 => 2,
            _ => bail!("unsupported CPU query dtype"),
        };
        query_offsets
            .windows(2)
            .map(|range| {
                let start = range[0] as usize * dimension as usize * scalar_bytes;
                let end = range[1] as usize * dimension as usize * scalar_bytes;
                self.score(
                    &queries[start..end],
                    range[1] - range[0],
                    dimension,
                    dtype,
                    scoring_profile,
                    document_offsets,
                    document_rows,
                )
            })
            .collect()
    }
}

fn exact_maxsim(
    query: &[u8],
    query_rows: u32,
    dimension: u32,
    dtype: u8,
    arena: &[u8],
    offset: u64,
    document_rows: u32,
) -> Result<f32> {
    let scalar_bytes = if dtype == 1 { 4 } else { 2 };
    let start =
        usize::try_from(offset).map_err(|_| anyhow::anyhow!("CPU document offset overflow"))?;
    let bytes = document_rows as usize * dimension as usize * scalar_bytes;
    let document = arena
        .get(start..start + bytes)
        .ok_or_else(|| anyhow::anyhow!("CPU document exceeds arena"))?;
    let mut score = 0.0_f32;
    for q in 0..query_rows as usize {
        let mut maximum = f32::NEG_INFINITY;
        for d in 0..document_rows as usize {
            let query_offset = q * dimension as usize * scalar_bytes;
            let document_offset = d * dimension as usize * scalar_bytes;
            let dot = if dtype == 1 {
                dot_f32_bytes(
                    &query[query_offset..query_offset + dimension as usize * 4],
                    &document[document_offset..document_offset + dimension as usize * 4],
                )
            } else {
                let mut dot = 0.0_f32;
                for k in 0..dimension as usize {
                    dot += read_scalar(query, query_offset + k * scalar_bytes, dtype)
                        * read_scalar(document, document_offset + k * scalar_bytes, dtype);
                }
                dot
            };
            maximum = maximum.max(dot);
        }
        score += maximum;
    }
    Ok(score)
}

fn cpu_tuning_profile() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return "cpu-avx2-fma";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "cpu-neon";
    }
    #[allow(unreachable_code)]
    "cpu-scalar"
}

fn dot_f32_bytes(left: &[u8], right: &[u8]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % 4, 0);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection guards the target features and the helper
        // uses unaligned loads within both equally-sized slices.
        return unsafe { dot_f32_avx2(left, right) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory for AArch64.
        return unsafe { dot_f32_neon(left, right) };
    }
    #[allow(unreachable_code)]
    dot_f32_scalar(left, right)
}

fn dot_f32_scalar(left: &[u8], right: &[u8]) -> f32 {
    left.chunks_exact(4)
        .zip(right.chunks_exact(4))
        .fold(0.0, |sum, (left, right)| {
            sum + f32::from_le_bytes(left.try_into().unwrap())
                * f32::from_le_bytes(right.try_into().unwrap())
        })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(left: &[u8], right: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let values = left.len() / 4;
    let mut index = 0;
    let mut sums = _mm256_setzero_ps();
    while index + 8 <= values {
        let a = unsafe { _mm256_loadu_ps(left.as_ptr().add(index * 4).cast()) };
        let b = unsafe { _mm256_loadu_ps(right.as_ptr().add(index * 4).cast()) };
        sums = _mm256_fmadd_ps(a, b, sums);
        index += 8;
    }
    let high = _mm256_extractf128_ps(sums, 1);
    let low = _mm256_castps256_ps128(sums);
    let sum128 = _mm_add_ps(low, high);
    let pair = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let total = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    let mut result = _mm_cvtss_f32(total);
    while index < values {
        let a = f32::from_le_bytes(left[index * 4..index * 4 + 4].try_into().unwrap());
        let b = f32::from_le_bytes(right[index * 4..index * 4 + 4].try_into().unwrap());
        result = a.mul_add(b, result);
        index += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(left: &[u8], right: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let values = left.len() / 4;
    let mut index = 0;
    let mut sums = vdupq_n_f32(0.0);
    while index + 4 <= values {
        let a = unsafe { vld1q_f32(left.as_ptr().add(index * 4).cast()) };
        let b = unsafe { vld1q_f32(right.as_ptr().add(index * 4).cast()) };
        sums = vfmaq_f32(sums, a, b);
        index += 4;
    }
    let mut result = vaddvq_f32(sums);
    while index < values {
        let a = f32::from_le_bytes(left[index * 4..index * 4 + 4].try_into().unwrap());
        let b = f32::from_le_bytes(right[index * 4..index * 4 + 4].try_into().unwrap());
        result = a.mul_add(b, result);
        index += 1;
    }
    result
}

fn read_scalar(bytes: &[u8], offset: usize, dtype: u8) -> f32 {
    if dtype == 1 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    } else {
        half_to_f32(u16::from_le_bytes(
            bytes[offset..offset + 2].try_into().unwrap(),
        ))
    }
}

fn half_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = value & 0x03ff;
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = fraction.leading_zeros() - 6;
            let normalized = (fraction << (leading + 1)) & 0x03ff;
            sign | ((127 - 15 - leading) << 23) | ((normalized as u32) << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reference_scores_maxsim() {
        let mut cpu = CpuBackend::create(0, 4096, 1024).unwrap();
        let document = [1.0_f32, 0.0, 0.0, 1.0];
        let payload = document
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        cpu.upload_batch(&[(0, &payload)]).unwrap();
        let query = [1.0_f32, 0.0, 0.5, 0.5];
        let query = query
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            cpu.score(&query, 2, 2, 1, 1, &[0], &[2]).unwrap(),
            vec![1.5]
        );
        assert!(cpu.supports_profile(1));
        assert!(!cpu.supports_profile(2));
        assert!(!cpu.supports_profile(5));
    }

    #[test]
    fn passes_vendor_neutral_conformance_probe() {
        let mut cpu = CpuBackend::create(0, 4096, 1024).unwrap();
        let report = crate::backend::run_conformance_probe(&mut cpu).unwrap();
        assert_eq!(report.backend, BackendKind::Cpu);
        assert_eq!(report.exact_fp32_score, report.expected_score);
    }

    #[test]
    fn runtime_simd_dot_matches_scalar_with_a_tail() {
        let left = (0..37)
            .flat_map(|index| ((index as f32 - 11.0) / 13.0).to_le_bytes())
            .collect::<Vec<_>>();
        let right = (0..37)
            .flat_map(|index| ((19.0 - index as f32) / 17.0).to_le_bytes())
            .collect::<Vec<_>>();
        let scalar = dot_f32_scalar(&left, &right);
        let dispatched = dot_f32_bytes(&left, &right);
        assert!((scalar - dispatched).abs() <= 2.0e-5 * (1.0 + scalar.abs()));
        assert!(cpu_tuning_profile().starts_with("cpu-"));
    }
}
