// This software is licensed under the repository's dual license model.

use crate::backend::{
    AcceleratorBackend, AdaptiveStatus, BackendCapabilities, BackendKind, DeviceInfo,
};
use anyhow::{Result, anyhow, bail};
use std::collections::HashSet;
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;

#[repr(C)]
struct NativeMetal(c_void);

unsafe extern "C" {
    fn vctm_metal_create(
        device: c_int,
        total_bytes: usize,
        workspace_bytes: usize,
        output: *mut *mut NativeMetal,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_metal_destroy(backend: *mut NativeMetal);
    fn vctm_metal_tensor_bytes(backend: *const NativeMetal) -> usize;
    fn vctm_metal_device_info(
        backend: *const NativeMetal,
        name: *mut c_char,
        name_capacity: usize,
        recommended_working_set_bytes: *mut u64,
        max_buffer_bytes: *mut u64,
    ) -> c_int;
    fn vctm_metal_upload_batch(
        backend: *mut NativeMetal,
        offsets: *const u64,
        payloads: *const *const c_uchar,
        lengths: *const usize,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_metal_score(
        backend: *mut NativeMetal,
        query: *const c_uchar,
        query_bytes: usize,
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        document_offsets: *const u64,
        document_rows: *const u32,
        count: usize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_metal_score_batch(
        backend: *mut NativeMetal,
        queries: *const c_uchar,
        query_bytes: usize,
        query_offsets: *const u32,
        request_count: u32,
        total_query_rows: u32,
        dimension: u32,
        dtype: u8,
        document_offsets: *const u64,
        document_rows: *const u32,
        count: usize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

pub struct MetalBackend {
    native: NonNull<NativeMetal>,
    tensor_bytes: usize,
    info: DeviceInfo,
}

// SAFETY: the native object and its command queue are uniquely owned. All
// execution methods require `&mut self`, and no Metal object escapes.
unsafe impl Send for MetalBackend {}

impl MetalBackend {
    pub fn create(device: i32, total_bytes: usize, workspace_bytes: usize) -> Result<Self> {
        let mut native = std::ptr::null_mut();
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_metal_create(
                device,
                total_bytes,
                workspace_bytes,
                &mut native,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        let native = NonNull::new(native).ok_or_else(|| anyhow!("Metal returned a null arena"))?;
        let tensor_bytes = unsafe { vctm_metal_tensor_bytes(native.as_ptr()) };
        let mut name = [0_i8; 256];
        let mut recommended_working_set_bytes = 0_u64;
        let mut max_buffer_bytes = 0_u64;
        let info_status = unsafe {
            vctm_metal_device_info(
                native.as_ptr(),
                name.as_mut_ptr(),
                name.len(),
                &mut recommended_working_set_bytes,
                &mut max_buffer_bytes,
            )
        };
        if info_status != 0 {
            unsafe { vctm_metal_destroy(native.as_ptr()) };
            bail!("Metal device information is unavailable");
        }
        Ok(Self {
            native,
            tensor_bytes,
            info: DeviceInfo {
                backend: BackendKind::Metal,
                ordinal: device,
                name: native_error(&name),
                architecture: "apple-gpu".to_owned(),
                driver_version: None,
                runtime_version: None,
                library_version: None,
                total_memory_bytes: (recommended_working_set_bytes != 0)
                    .then_some(recommended_working_set_bytes),
                compute_units: None,
                warp_size: Some(32),
                memory_bus_width_bits: None,
                memory_clock_khz: None,
                shared_memory_per_block_bytes: None,
                persisting_l2_bytes: None,
                matrix_engine_workspace_bytes: None,
                document_tile_query_rows: Some(8),
                pq_warp_task_max_document_rows: None,
                compute_queue_priority: None,
                tuning_profile: "metal-apple-unified".to_owned(),
                capabilities: BackendCapabilities {
                    kind: BackendKind::Metal,
                    exact_fp16: true,
                    exact_fp32: true,
                    int8: false,
                    fp8_e4m3: false,
                    pq: false,
                    opq_rpq: false,
                    fused_multiquery: true,
                    matrix_engine: false,
                    asynchronous_copy: false,
                    unified_memory: true,
                    persisting_l2: false,
                },
            },
        })
    }
}

impl AcceleratorBackend for MetalBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    fn adaptive_status(&self) -> AdaptiveStatus {
        AdaptiveStatus::default()
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
        bail!("Metal backend does not support PQ-family profiles")
    }

    fn retain_quantizers(&mut self, _active: &HashSet<String>) {}

    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let offsets = items.iter().map(|item| item.0).collect::<Vec<_>>();
        let payloads = items.iter().map(|item| item.1.as_ptr()).collect::<Vec<_>>();
        let lengths = items.iter().map(|item| item.1.len()).collect::<Vec<_>>();
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_metal_upload_batch(
                self.native.as_ptr(),
                offsets.as_ptr(),
                payloads.as_ptr(),
                lengths.as_ptr(),
                items.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        Ok(())
    }

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
    ) -> Result<Vec<f32>> {
        if scoring_profile != 1 {
            bail!("Metal backend supports only exact FP16/FP32 scoring");
        }
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid Metal document metadata");
        }
        let mut output = vec![0.0_f32; document_offsets.len()];
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_metal_score(
                self.native.as_ptr(),
                query.as_ptr(),
                query.len(),
                query_rows,
                dimension,
                dtype,
                document_offsets.as_ptr(),
                document_rows.as_ptr(),
                document_offsets.len(),
                output.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        if output.iter().any(|score| !score.is_finite()) {
            bail!("Metal TileMaxSim returned a non-finite score");
        }
        Ok(output)
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
        bail!("Metal backend does not support PQ-family profiles")
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
    ) -> Result<Vec<Vec<f32>>> {
        if scoring_profile != 1 {
            bail!("Metal backend does not support quantized batch scoring");
        }
        if query_offsets.len() < 3 || query_offsets[0] != 0 {
            bail!("invalid Metal batch query offsets");
        }
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid Metal batch document metadata");
        }
        let request_count = query_offsets.len() - 1;
        let total_query_rows = *query_offsets.last().unwrap();
        let output_count = request_count
            .checked_mul(document_offsets.len())
            .ok_or_else(|| anyhow!("Metal batch output shape overflows address space"))?;
        let mut flat_output = vec![0.0_f32; output_count];
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_metal_score_batch(
                self.native.as_ptr(),
                queries.as_ptr(),
                queries.len(),
                query_offsets.as_ptr(),
                request_count as u32,
                total_query_rows,
                dimension,
                dtype,
                document_offsets.as_ptr(),
                document_rows.as_ptr(),
                document_offsets.len(),
                flat_output.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        if flat_output.iter().any(|score| !score.is_finite()) {
            bail!("Metal batch TileMaxSim returned a non-finite score");
        }
        Ok(flat_output
            .chunks_exact(document_offsets.len())
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        unsafe { vctm_metal_destroy(self.native.as_ptr()) };
    }
}

fn native_error(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an explicitly assigned Apple Metal device"]
    fn passes_vendor_neutral_conformance_probe() {
        let mut backend = MetalBackend::create(0, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let report = crate::backend::run_conformance_probe(&mut backend).unwrap();
        assert_eq!(report.backend, BackendKind::Metal);
        assert_eq!(report.exact_fp32_score, report.expected_score);
        assert_eq!(report.exact_fp16_score, report.expected_score);
        assert_eq!(report.batched_fp16_scores, Some(vec![1.0, 0.5]));
    }

    #[test]
    #[ignore = "requires an explicitly assigned Apple Metal device"]
    fn native_batch_matches_individual_requests() {
        let mut backend = MetalBackend::create(0, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let document = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.upload_batch(&[(0, &document)]).unwrap();
        let queries = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let scores = backend
            .score_batch(&queries, &[0, 2, 3], 2, 1, 1, &[0], &[2])
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0][0] - 2.0).abs() < 1e-5);
        assert!((scores[1][0] - 1.0).abs() < 1e-5);
    }
}
