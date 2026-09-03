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
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;

#[repr(C)]
struct NativeGpu(c_void);
#[repr(C)]
struct NativeQuantizer(c_void);

unsafe extern "C" {
    fn vctm_gpu_create(
        device: c_int,
        total_bytes: usize,
        workspace_bytes: usize,
        output: *mut *mut NativeGpu,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_gpu_destroy(gpu: *mut NativeGpu);
    fn vctm_gpu_tensor_bytes(gpu: *const NativeGpu) -> usize;
    fn vctm_quantizer_create(
        device: c_int,
        payload: *const c_uchar,
        payload_bytes: usize,
        dimension: u32,
        stages: u16,
        subspaces: u16,
        centroids: u16,
        rotation_mask: u16,
        output: *mut *mut NativeQuantizer,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_quantizer_destroy(quantizer: *mut NativeQuantizer);
    fn vctm_gpu_upload_batch(
        gpu: *mut NativeGpu,
        offsets: *const u64,
        payloads: *const *const c_uchar,
        lengths: *const usize,
        count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_gpu_score(
        gpu: *mut NativeGpu,
        query: *const c_uchar,
        query_bytes: usize,
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: *const u64,
        document_rows: *const u32,
        count: usize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn vctm_gpu_score_batch(
        gpu: *mut NativeGpu,
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
    fn vctm_gpu_score_pq(
        gpu: *mut NativeGpu,
        quantizer: *const NativeQuantizer,
        query: *const c_uchar,
        query_bytes: usize,
        query_rows: u32,
        dtype: u8,
        document_offsets: *const u64,
        document_rows: *const u32,
        count: usize,
        output: *mut f32,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

pub struct Gpu {
    native: NonNull<NativeGpu>,
    device: i32,
    tensor_bytes: usize,
    quantizers: HashMap<String, NonNull<NativeQuantizer>>,
}

// SAFETY: `Gpu` uniquely owns the native handle. It may move to a scoped
// worker, but no method exposes the pointer and all calls require `&mut self`.
unsafe impl Send for Gpu {}

impl Gpu {
    pub fn create(device: i32, total_bytes: usize, workspace_bytes: usize) -> Result<Self> {
        let mut native = std::ptr::null_mut();
        let mut error = [0_i8; 512];
        // SAFETY: the C API writes one opaque pointer and a bounded error string.
        let status = unsafe {
            vctm_gpu_create(
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
        let native = NonNull::new(native).ok_or_else(|| anyhow!("CUDA returned a null arena"))?;
        // SAFETY: `native` is live until Drop.
        let tensor_bytes = unsafe { vctm_gpu_tensor_bytes(native.as_ptr()) };
        Ok(Self {
            native,
            device,
            tensor_bytes,
            quantizers: HashMap::new(),
        })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_quantizer(
        &mut self,
        contract_id: &str,
        payload: &[u8],
        dimension: u32,
        stages: u16,
        subspaces: u16,
        centroids: u16,
        rotation_mask: u16,
    ) -> Result<()> {
        if self.quantizers.contains_key(contract_id) {
            return Ok(());
        }
        let mut native = std::ptr::null_mut();
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_quantizer_create(
                self.device,
                payload.as_ptr(),
                payload.len(),
                dimension,
                stages,
                subspaces,
                centroids,
                rotation_mask,
                &mut native,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            bail!(native_error(&error));
        }
        self.quantizers.insert(
            contract_id.to_owned(),
            NonNull::new(native).ok_or_else(|| anyhow!("CUDA returned a null quantizer"))?,
        );
        Ok(())
    }

    pub fn retain_quantizers(&mut self, active: &std::collections::HashSet<String>) {
        self.quantizers.retain(|contract, quantizer| {
            if active.contains(contract) {
                true
            } else {
                unsafe { vctm_quantizer_destroy(quantizer.as_ptr()) };
                false
            }
        });
    }

    pub fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let offsets = items.iter().map(|item| item.0).collect::<Vec<_>>();
        let payloads = items.iter().map(|item| item.1.as_ptr()).collect::<Vec<_>>();
        let lengths = items.iter().map(|item| item.1.len()).collect::<Vec<_>>();
        let mut error = [0_i8; 512];
        // SAFETY: all slices remain alive for the synchronous native batch call.
        let status = unsafe {
            vctm_gpu_upload_batch(
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

    pub fn score(
        &mut self,
        query: &[u8],
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid native document metadata");
        }
        let mut output = vec![0.0_f32; document_offsets.len()];
        let mut error = [0_i8; 512];
        // SAFETY: the native call is synchronous and receives valid slice pointers.
        let status = unsafe {
            vctm_gpu_score(
                self.native.as_ptr(),
                query.as_ptr(),
                query.len(),
                query_rows,
                dimension,
                dtype,
                scoring_profile,
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
            bail!("native TileMaxSim returned a non-finite score");
        }
        Ok(output)
    }

    pub fn score_pq(
        &mut self,
        contract_id: &str,
        query: &[u8],
        query_rows: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        let quantizer = self
            .quantizers
            .get(contract_id)
            .ok_or_else(|| anyhow!("PQ quantizer is not resident on this GPU"))?;
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid PQ document metadata");
        }
        let mut output = vec![0.0_f32; document_offsets.len()];
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_gpu_score_pq(
                self.native.as_ptr(),
                quantizer.as_ptr(),
                query.as_ptr(),
                query.len(),
                query_rows,
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
            bail!("native PQ TileMaxSim returned a non-finite score");
        }
        Ok(output)
    }

    pub fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if query_offsets.len() < 3 || query_offsets[0] != 0 {
            bail!("a native multi-query batch requires at least two queries");
        }
        let request_count = query_offsets.len() - 1;
        let total_query_rows = *query_offsets.last().unwrap();
        let mut output = vec![0.0_f32; request_count * document_offsets.len()];
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_gpu_score_batch(
                self.native.as_ptr(),
                queries.as_ptr(),
                queries.len(),
                query_offsets.as_ptr(),
                u32::try_from(request_count).map_err(|_| anyhow!("too many batched queries"))?,
                total_query_rows,
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
            bail!("native multi-query TileMaxSim returned a non-finite score");
        }
        Ok(output
            .chunks(document_offsets.len())
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        for (_, quantizer) in self.quantizers.drain() {
            unsafe { vctm_quantizer_destroy(quantizer.as_ptr()) };
        }
        // SAFETY: this is the unique owned native pointer.
        unsafe { vctm_gpu_destroy(self.native.as_ptr()) };
    }
}

fn native_error(buffer: &[c_char]) -> String {
    // SAFETY: the native helper always NUL-terminates a nonempty error buffer.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_multiquery_scores_shared_documents_once_per_tile() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let half_one = 0x3c00_u16.to_le_bytes();
        let half_zero = 0_u16.to_le_bytes();
        let document = [half_one, half_zero, half_zero, half_one].concat();
        gpu.upload_batch(&[(0, &document)]).unwrap();
        let queries = [half_one, half_zero, half_zero, half_one].concat();
        let scores = gpu
            .score_batch(&queries, &[0, 1, 2], 2, 2, &[0], &[2])
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
        assert!((scores[1][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_int8_profile_scores_all_candidates() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        // Two 2-D rows followed by aligned per-row FP32 scales.
        let mut document = vec![127_u8, 0, 0, 127];
        document.extend_from_slice(&(1.0_f32 / 127.0).to_le_bytes());
        document.extend_from_slice(&(1.0_f32 / 127.0).to_le_bytes());
        gpu.upload_batch(&[(0, &document)]).unwrap();
        let query = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let scores = gpu.score(&query, 2, 2, 1, 2, &[0], &[2]).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_fp8_profile_scores_all_candidates() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        // E4M3FN 0x38 is 1.0; two identity rows with unit row scales.
        let mut document = vec![0x38_u8, 0, 0, 0x38];
        document.extend_from_slice(&1.0_f32.to_le_bytes());
        document.extend_from_slice(&1.0_f32.to_le_bytes());
        gpu.upload_batch(&[(0, &document)]).unwrap();
        let query = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let scores = gpu.score(&query, 2, 2, 1, 3, &[0], &[2]).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-5, "scores={scores:?}");
    }

    fn pq_gpu() -> Gpu {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse()
            .unwrap();
        Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap()
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        values.iter().copied().flat_map(f32::to_le_bytes).collect()
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_pq_adc_maxsim_matches_identity_oracle() {
        let mut gpu = pq_gpu();
        gpu.ensure_quantizer("pq", &f32_payload(&[1.0, 0.0, 0.0, 1.0]), 2, 1, 1, 2, 0)
            .unwrap();
        gpu.upload_batch(&[(0, &[0_u8, 1])]).unwrap();
        let query = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        let scores = gpu.score_pq("pq", &query, 2, 1, &[0], &[2]).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_opq_rotation_is_applied_before_adc() {
        let mut gpu = pq_gpu();
        // Swap-coordinate rotation, followed by identity centroids.
        let payload = f32_payload(&[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        gpu.ensure_quantizer("opq", &payload, 2, 1, 1, 2, 1)
            .unwrap();
        gpu.upload_batch(&[(0, &[1_u8, 0])]).unwrap();
        let query = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        let scores = gpu.score_pq("opq", &query, 2, 1, &[0], &[2]).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_residual_pq_accumulates_all_stages_before_max() {
        let mut gpu = pq_gpu();
        // Stage 0 contributes identity; stage 1 contributes 0.5 * identity.
        let payload = f32_payload(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
        gpu.ensure_quantizer("rpq", &payload, 2, 2, 1, 2, 0)
            .unwrap();
        // row-major [row][stage][subspace]
        gpu.upload_batch(&[(0, &[0_u8, 0, 1, 1])]).unwrap();
        let query = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        let scores = gpu.score_pq("rpq", &query, 2, 1, &[0], &[2]).unwrap();
        assert!((scores[0] - 3.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn inactive_quantizer_allocations_are_reclaimed_at_a_safe_point() {
        let mut gpu = pq_gpu();
        let payload = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        gpu.ensure_quantizer("active", &payload, 2, 1, 1, 2, 0)
            .unwrap();
        gpu.ensure_quantizer("retired", &payload, 2, 1, 1, 2, 0)
            .unwrap();
        gpu.retain_quantizers(&std::collections::HashSet::from(["active".to_owned()]));
        assert!(gpu.quantizers.contains_key("active"));
        assert!(!gpu.quantizers.contains_key("retired"));
    }
}
