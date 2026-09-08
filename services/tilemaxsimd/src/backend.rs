// This software is licensed under the repository's dual license model.

use crate::dispatch::TensorCalibrationBucket;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)] // Non-CUDA variants are enabled by their isolated builds.
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Ascend,
    Metax,
}

#[derive(Clone, Debug, Serialize)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub exact_fp16: bool,
    pub exact_fp32: bool,
    pub int8: bool,
    pub fp8_e4m3: bool,
    pub pq: bool,
    pub opq_rpq: bool,
    pub fused_multiquery: bool,
    pub matrix_engine: bool,
    pub asynchronous_copy: bool,
    pub unified_memory: bool,
    pub persisting_l2: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeviceInfo {
    pub backend: BackendKind,
    pub ordinal: i32,
    pub name: String,
    pub architecture: String,
    pub driver_version: Option<u32>,
    pub runtime_version: Option<u32>,
    pub library_version: Option<u32>,
    pub total_memory_bytes: Option<u64>,
    pub compute_units: Option<u32>,
    pub warp_size: Option<u32>,
    pub memory_bus_width_bits: Option<u32>,
    pub memory_clock_khz: Option<u32>,
    pub shared_memory_per_block_bytes: Option<u64>,
    pub persisting_l2_bytes: Option<u64>,
    pub matrix_engine_workspace_bytes: Option<u64>,
    /// Whether request control payloads are packed into persistent pinned host
    /// memory before H2D submission on this device.
    pub pinned_control_staging: Option<bool>,
    /// Direct-pageable median time divided by pinned-packed median time,
    /// expressed in thousandths. Values above 1000 favour pinned packing.
    pub control_staging_speedup_milli: Option<u32>,
    /// Whether exact document rows use two shared-memory buffers so the next
    /// asynchronous copy can overlap current-row dot products.
    pub double_buffered_tile: Option<bool>,
    /// Single-buffer median divided by double-buffer median, in thousandths.
    pub double_buffer_speedup_milli: Option<u32>,
    /// Query rows served by one document-tile load in the native fused path.
    pub document_tile_query_rows: Option<u32>,
    /// Largest PQ document row count assigned one independent task per warp.
    pub pq_warp_task_max_document_rows: Option<u32>,
    pub compute_queue_priority: Option<i32>,
    pub tuning_profile: String,
    pub capabilities: BackendCapabilities,
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveStatus {
    pub tensor_threshold_rows: u32,
    pub calibration_complete: bool,
    pub batch_vector_calls: u64,
    pub batch_matrix_calls: u64,
    pub calibration_runs: u64,
    pub calibration_failures: u64,
    pub tensor_calibration_buckets: Vec<TensorCalibrationBucket>,
    pub tensor_chunk_candidates: u32,
    pub tensor_circuit_open: bool,
    pub tensor_suppressed_bucket_count: u32,
    pub tensor_request_fallbacks: u64,
    pub tensor_capacity_fallbacks: u64,
    pub tensor_device_fallbacks: u64,
}

/// Vendor-neutral execution contract. Scheduling, caching, storage and wire
/// protocol code must depend on this interface rather than a vendor runtime.
/// Unsupported precision profiles fail explicitly; they are never substituted.
pub trait AcceleratorBackend: Send {
    fn info(&self) -> &DeviceInfo;
    fn tensor_bytes(&self) -> usize;
    fn adaptive_status(&self) -> AdaptiveStatus;

    fn supports_profile(&self, native_profile: u8) -> bool {
        let capabilities = &self.info().capabilities;
        match native_profile {
            1 => capabilities.exact_fp16,
            2 => capabilities.int8,
            3 => capabilities.fp8_e4m3,
            4 => capabilities.pq,
            5 => capabilities.opq_rpq,
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_quantizer(
        &mut self,
        contract_id: &str,
        payload: &[u8],
        dimension: u32,
        stages: u16,
        subspaces: u16,
        centroids: u16,
        rotation_mask: u16,
    ) -> Result<()>;
    fn retain_quantizers(&mut self, active: &HashSet<String>);
    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn score(
        &mut self,
        query: &[u8],
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>>;

    fn score_pq(
        &mut self,
        contract_id: &str,
        query: &[u8],
        query_rows: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>>;

    #[allow(clippy::too_many_arguments)]
    fn score_pq_batch(
        &mut self,
        contract_id: &str,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let scalar_bytes = match dtype {
            1 => 4,
            2 => 2,
            _ => anyhow::bail!("unsupported PQ query dtype"),
        };
        query_offsets
            .windows(2)
            .map(|range| {
                let start = range[0] as usize * dimension as usize * scalar_bytes;
                let end = range[1] as usize * dimension as usize * scalar_bytes;
                self.score_pq(
                    contract_id,
                    queries
                        .get(start..end)
                        .ok_or_else(|| anyhow::anyhow!("PQ batch offsets exceed query payload"))?,
                    range[1] - range[0],
                    dtype,
                    document_offsets,
                    document_rows,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>>;
}

#[derive(Clone, Debug, Serialize)]
pub struct ConformanceReport {
    pub backend: BackendKind,
    pub architecture: String,
    pub exact_fp32_score: f32,
    pub exact_fp16_score: f32,
    pub batched_fp16_scores: Option<Vec<f32>>,
    pub expected_score: f32,
}

/// Minimal vendor-backend acceptance probe. Run on a newly-created arena
/// before serving requests; it verifies actual upload and exact MaxSim
/// execution rather than trusting a capability bit.
pub fn run_conformance_probe(backend: &mut dyn AcceleratorBackend) -> Result<ConformanceReport> {
    if !backend.info().capabilities.exact_fp32 || !backend.info().capabilities.exact_fp16 {
        anyhow::bail!("backend conformance requires exact FP32 and FP16 support");
    }
    let document = [1.0_f32, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let query = [1.0_f32, 0.0, 0.5, 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    backend.upload_batch(&[(0, &document)])?;
    let score = backend.score(&query, 2, 2, 1, 1, &[0], &[2])?[0];
    let expected = 1.5_f32;
    if !score.is_finite() || (score - expected).abs() > 1.0e-5 {
        anyhow::bail!(
            "backend exact MaxSim conformance failed: expected {expected}, received {score}"
        );
    }
    let document_fp16 = [0x3c00_u16, 0, 0, 0x3c00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let query_fp16 = [0x3c00_u16, 0, 0x3800, 0x3800]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    backend.upload_batch(&[(256, &document_fp16)])?;
    let fp16_score = backend.score(&query_fp16, 2, 2, 2, 1, &[256], &[2])?[0];
    if !fp16_score.is_finite() || (fp16_score - expected).abs() > 1.0e-3 {
        anyhow::bail!(
            "backend FP16 MaxSim conformance failed: expected {expected}, received {fp16_score}"
        );
    }
    let batched_fp16_scores = if backend.info().capabilities.fused_multiquery {
        let values = backend.score_batch(&query_fp16, &[0, 1, 2], 2, 2, 1, &[256], &[2])?;
        let values = values
            .into_iter()
            .map(|request| {
                request
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("backend batch conformance returned no score"))
            })
            .collect::<Result<Vec<_>>>()?;
        let expected_batch = [1.0_f32, 0.5];
        if values.len() != expected_batch.len()
            || values
                .iter()
                .zip(expected_batch)
                .any(|(actual, expected)| (actual - expected).abs() > 1.0e-3)
        {
            anyhow::bail!(
                "backend fused FP16 batch conformance failed: expected {expected_batch:?}, received {values:?}"
            );
        }
        Some(values)
    } else {
        None
    };
    Ok(ConformanceReport {
        backend: backend.info().backend,
        architecture: backend.info().architecture.clone(),
        exact_fp32_score: score,
        exact_fp16_score: fp16_score,
        batched_fp16_scores,
        expected_score: expected,
    })
}
