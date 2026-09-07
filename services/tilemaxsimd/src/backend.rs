// This software is licensed under the repository's dual license model.

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
    pub capabilities: BackendCapabilities,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveStatus {
    pub tensor_threshold_rows: u32,
    pub calibration_complete: bool,
    pub batch_vector_calls: u64,
    pub batch_matrix_calls: u64,
    pub calibration_runs: u64,
    pub calibration_failures: u64,
}

/// Vendor-neutral execution contract. Scheduling, caching, storage and wire
/// protocol code must depend on this interface rather than a vendor runtime.
/// Unsupported precision profiles fail explicitly; they are never substituted.
pub trait AcceleratorBackend: Send {
    fn info(&self) -> &DeviceInfo;
    fn tensor_bytes(&self) -> usize;
    fn adaptive_status(&self) -> AdaptiveStatus;

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
    fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>>;
}
