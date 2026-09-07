// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use crate::backend::{
    AcceleratorBackend, AdaptiveStatus, BackendCapabilities, BackendKind, DeviceInfo,
};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;
use std::time::Instant;

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
    fn vctm_gpu_compute_capability(
        gpu: *const NativeGpu,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> c_int;
    fn vctm_gpu_device_name(gpu: *const NativeGpu, name: *mut c_char, capacity: usize) -> c_int;
    fn vctm_gpu_device_info(
        gpu: *const NativeGpu,
        driver_version: *mut c_int,
        runtime_version: *mut c_int,
        cublas_version: *mut c_int,
        total_memory_bytes: *mut u64,
        multiprocessors: *mut c_int,
        warp_size: *mut c_int,
        memory_bus_width_bits: *mut c_int,
        memory_clock_khz: *mut c_int,
        shared_memory_per_block_bytes: *mut u64,
        compute_stream_priority: *mut c_int,
    ) -> c_int;
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
    fn vctm_gpu_score_batch_tensor(
        gpu: *mut NativeGpu,
        queries: *const c_uchar,
        query_bytes: usize,
        query_offsets: *const u32,
        request_count: u32,
        total_query_rows: u32,
        dimension: u32,
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
    tensor_threshold_rows: u32,
    calibration_complete: bool,
    batch_warp_calls: u64,
    batch_tensor_calls: u64,
    calibration_runs: u64,
    calibration_failures: u64,
    info: DeviceInfo,
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
        let mut major = 0;
        let mut minor = 0;
        let capability_status =
            unsafe { vctm_gpu_compute_capability(native.as_ptr(), &mut major, &mut minor) };
        let mut name = [0_i8; 256];
        let name_status =
            unsafe { vctm_gpu_device_name(native.as_ptr(), name.as_mut_ptr(), name.len()) };
        let name = if name_status == 0 {
            native_error(&name)
        } else {
            String::new()
        };
        let mut driver_version = 0;
        let mut runtime_version = 0;
        let mut cublas_version = 0;
        let mut total_memory_bytes = 0_u64;
        let mut multiprocessors = 0;
        let mut warp_size = 0;
        let mut memory_bus_width_bits = 0;
        let mut memory_clock_khz = 0;
        let mut shared_memory_per_block_bytes = 0_u64;
        let mut compute_stream_priority = 0;
        let info_status = unsafe {
            vctm_gpu_device_info(
                native.as_ptr(),
                &mut driver_version,
                &mut runtime_version,
                &mut cublas_version,
                &mut total_memory_bytes,
                &mut multiprocessors,
                &mut warp_size,
                &mut memory_bus_width_bits,
                &mut memory_clock_khz,
                &mut shared_memory_per_block_bytes,
                &mut compute_stream_priority,
            )
        };
        let tensor_threshold_rows = if capability_status == 0 {
            crate::dispatch::device_thresholds(&name, major, minor)
                .map(|thresholds| u32::try_from(thresholds.tensor_ridge).unwrap_or(u32::MAX))
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        let mut gpu = Self {
            native,
            device,
            tensor_bytes,
            quantizers: HashMap::new(),
            tensor_threshold_rows,
            calibration_complete: false,
            batch_warp_calls: 0,
            batch_tensor_calls: 0,
            calibration_runs: 0,
            calibration_failures: 0,
            info: DeviceInfo {
                backend: BackendKind::Cuda,
                ordinal: device,
                name,
                architecture: if capability_status == 0 {
                    format!("sm_{major}{minor}")
                } else {
                    "unknown".to_owned()
                },
                driver_version: (info_status == 0).then_some(driver_version as u32),
                runtime_version: (info_status == 0).then_some(runtime_version as u32),
                library_version: (info_status == 0).then_some(cublas_version as u32),
                total_memory_bytes: (info_status == 0).then_some(total_memory_bytes),
                compute_units: (info_status == 0).then_some(multiprocessors as u32),
                warp_size: (info_status == 0).then_some(warp_size as u32),
                memory_bus_width_bits: (info_status == 0).then_some(memory_bus_width_bits as u32),
                memory_clock_khz: (info_status == 0).then_some(memory_clock_khz as u32),
                shared_memory_per_block_bytes: (info_status == 0)
                    .then_some(shared_memory_per_block_bytes),
                compute_queue_priority: (info_status == 0).then_some(compute_stream_priority),
                tuning_profile: match (major, minor) {
                    (8, 9) => "cuda-ada".to_owned(),
                    (9, _) => "cuda-hopper".to_owned(),
                    (10 | 12, _) => "cuda-blackwell".to_owned(),
                    (8, _) => "cuda-ampere".to_owned(),
                    _ => "cuda-generic".to_owned(),
                },
                capabilities: BackendCapabilities {
                    kind: BackendKind::Cuda,
                    exact_fp16: true,
                    exact_fp32: true,
                    int8: true,
                    fp8_e4m3: true,
                    pq: true,
                    opq_rpq: true,
                    fused_multiquery: true,
                    matrix_engine: tensor_threshold_rows != u32::MAX,
                    asynchronous_copy: true,
                    unified_memory: false,
                },
            },
        };
        gpu.calibrate_kernel_thresholds();
        Ok(gpu)
    }

    pub fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    pub fn adaptive_status(&self) -> (u32, bool, u64, u64, u64, u64) {
        (
            self.tensor_threshold_rows,
            self.calibration_complete,
            self.batch_warp_calls,
            self.batch_tensor_calls,
            self.calibration_runs,
            self.calibration_failures,
        )
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
        let total_rows = *query_offsets.last().unwrap_or(&0);
        let tensor_eligible = dtype == 2 && self.tensor_threshold_rows != u32::MAX;
        if tensor_eligible && !self.calibration_complete {
            self.calibration_runs += 1;
            let warp_started = Instant::now();
            let warp = self.score_batch_native(
                false,
                queries,
                query_offsets,
                dimension,
                dtype,
                document_offsets,
                document_rows,
            )?;
            let warp_elapsed = warp_started.elapsed();
            let tensor_started = Instant::now();
            match self.score_batch_native(
                true,
                queries,
                query_offsets,
                dimension,
                dtype,
                document_offsets,
                document_rows,
            ) {
                Ok(tensor) if batch_scores_close(&warp, &tensor) => {
                    let tensor_elapsed = tensor_started.elapsed();
                    self.calibration_complete = true;
                    if tensor_elapsed < warp_elapsed {
                        self.tensor_threshold_rows = total_rows.max(1);
                        self.batch_tensor_calls += 1;
                        return Ok(tensor);
                    }
                    self.tensor_threshold_rows =
                        total_rows.saturating_mul(2).max(self.tensor_threshold_rows);
                    self.batch_warp_calls += 1;
                    return Ok(warp);
                }
                _ => {
                    self.calibration_failures += 1;
                    self.calibration_complete = true;
                    self.tensor_threshold_rows = u32::MAX;
                    self.batch_warp_calls += 1;
                    return Ok(warp);
                }
            }
        }
        if tensor_eligible && total_rows >= self.tensor_threshold_rows {
            match self.score_batch_native(
                true,
                queries,
                query_offsets,
                dimension,
                dtype,
                document_offsets,
                document_rows,
            ) {
                Ok(scores) => {
                    self.batch_tensor_calls += 1;
                    return Ok(scores);
                }
                Err(_) => {
                    self.calibration_failures += 1;
                    self.tensor_threshold_rows = u32::MAX;
                }
            }
        }
        self.batch_warp_calls += 1;
        self.score_batch_native(
            false,
            queries,
            query_offsets,
            dimension,
            dtype,
            document_offsets,
            document_rows,
        )
    }

    fn calibrate_kernel_thresholds(&mut self) {
        if self.tensor_threshold_rows == u32::MAX {
            return;
        }
        const DIMENSION: u32 = 128;
        const DOCUMENT_ROWS: u32 = 32;
        let one = 0x3c00_u16.to_le_bytes();
        let zero = 0_u16.to_le_bytes();
        let mut document = Vec::with_capacity(DIMENSION as usize * DOCUMENT_ROWS as usize * 2);
        for row in 0..DOCUMENT_ROWS {
            for column in 0..DIMENSION {
                document.extend_from_slice(if column == row % DIMENSION {
                    &one
                } else {
                    &zero
                });
            }
        }
        if self.upload_batch(&[(0, &document)]).is_err() {
            self.calibration_failures += 1;
            return;
        }
        let fallback = self.tensor_threshold_rows;
        let mut successful = 0_u64;
        let mut crossover = None;
        for total_rows in [32_u32, 96, 256, 512] {
            let mut queries = Vec::with_capacity(total_rows as usize * DIMENSION as usize * 2);
            for row in 0..total_rows {
                for column in 0..DIMENSION {
                    queries.extend_from_slice(if column == row % DOCUMENT_ROWS {
                        &one
                    } else {
                        &zero
                    });
                }
            }
            let offsets = [0, total_rows / 2, total_rows];
            self.calibration_runs += 1;
            let warp_started = Instant::now();
            let Ok(warp) = self.score_batch_native(
                false,
                &queries,
                &offsets,
                DIMENSION,
                2,
                &[0],
                &[DOCUMENT_ROWS],
            ) else {
                self.calibration_failures += 1;
                continue;
            };
            let warp_elapsed = warp_started.elapsed();
            let tensor_started = Instant::now();
            let Ok(tensor) = self.score_batch_native(
                true,
                &queries,
                &offsets,
                DIMENSION,
                2,
                &[0],
                &[DOCUMENT_ROWS],
            ) else {
                self.calibration_failures += 1;
                continue;
            };
            let tensor_elapsed = tensor_started.elapsed();
            if !batch_scores_close(&warp, &tensor) {
                self.calibration_failures += 1;
                continue;
            }
            successful += 1;
            if crossover.is_none() && tensor_elapsed < warp_elapsed {
                crossover = Some(total_rows);
            }
        }
        if successful == 0 {
            self.tensor_threshold_rows = fallback;
            self.calibration_complete = false;
        } else {
            self.tensor_threshold_rows = crossover.unwrap_or(u32::MAX);
            self.calibration_complete = true;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score_batch_native(
        &mut self,
        tensor: bool,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        if tensor && dtype != 2 {
            bail!("Tensor Core TileMaxSim currently requires FP16 queries");
        }
        if query_offsets.len() < 3 || query_offsets[0] != 0 {
            bail!("a native multi-query batch requires at least two queries");
        }
        let request_count = query_offsets.len() - 1;
        let total_query_rows = *query_offsets.last().unwrap();
        let mut output = vec![0.0_f32; request_count * document_offsets.len()];
        let mut error = [0_i8; 512];
        let status = unsafe {
            if tensor {
                vctm_gpu_score_batch_tensor(
                    self.native.as_ptr(),
                    queries.as_ptr(),
                    queries.len(),
                    query_offsets.as_ptr(),
                    u32::try_from(request_count)
                        .map_err(|_| anyhow!("too many batched queries"))?,
                    total_query_rows,
                    dimension,
                    document_offsets.as_ptr(),
                    document_rows.as_ptr(),
                    document_offsets.len(),
                    output.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            } else {
                vctm_gpu_score_batch(
                    self.native.as_ptr(),
                    queries.as_ptr(),
                    queries.len(),
                    query_offsets.as_ptr(),
                    u32::try_from(request_count)
                        .map_err(|_| anyhow!("too many batched queries"))?,
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
            }
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

impl AcceleratorBackend for Gpu {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn tensor_bytes(&self) -> usize {
        Gpu::tensor_bytes(self)
    }

    fn adaptive_status(&self) -> AdaptiveStatus {
        let status = Gpu::adaptive_status(self);
        AdaptiveStatus {
            tensor_threshold_rows: status.0,
            calibration_complete: status.1,
            batch_vector_calls: status.2,
            batch_matrix_calls: status.3,
            calibration_runs: status.4,
            calibration_failures: status.5,
        }
    }

    fn ensure_quantizer(
        &mut self,
        contract_id: &str,
        payload: &[u8],
        dimension: u32,
        stages: u16,
        subspaces: u16,
        centroids: u16,
        rotation_mask: u16,
    ) -> Result<()> {
        Gpu::ensure_quantizer(
            self,
            contract_id,
            payload,
            dimension,
            stages,
            subspaces,
            centroids,
            rotation_mask,
        )
    }

    fn retain_quantizers(&mut self, active: &std::collections::HashSet<String>) {
        Gpu::retain_quantizers(self, active)
    }

    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        Gpu::upload_batch(self, items)
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
        Gpu::score(
            self,
            query,
            query_rows,
            dimension,
            dtype,
            scoring_profile,
            document_offsets,
            document_rows,
        )
    }

    fn score_pq(
        &mut self,
        contract_id: &str,
        query: &[u8],
        query_rows: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<f32>> {
        Gpu::score_pq(
            self,
            contract_id,
            query,
            query_rows,
            dtype,
            document_offsets,
            document_rows,
        )
    }

    fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        Gpu::score_batch(
            self,
            queries,
            query_offsets,
            dimension,
            dtype,
            document_offsets,
            document_rows,
        )
    }
}

fn batch_scores_close(left: &[Vec<f32>], right: &[Vec<f32>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.len() == right.len()
                && left.iter().zip(right).all(|(left, right)| {
                    (left - right).abs() <= 2.0e-3 * (1.0 + left.abs().max(right.abs()))
                })
        })
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
    fn device_info_reports_runtime_fingerprint() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let info = AcceleratorBackend::info(&gpu);
        assert_eq!(info.backend, BackendKind::Cuda);
        assert!(info.name.contains("NVIDIA"));
        assert!(info.architecture.starts_with("sm_"));
        assert!(info.driver_version.unwrap_or_default() > 0);
        assert!(info.runtime_version.unwrap_or_default() > 0);
        assert!(info.library_version.unwrap_or_default() > 0);
        assert!(info.total_memory_bytes.unwrap_or_default() > 0);
        assert!(info.compute_units.unwrap_or_default() > 0);
        assert_eq!(info.warp_size, Some(32));
        assert!(info.compute_queue_priority.is_some());
        assert!(info.tuning_profile.starts_with("cuda-"));
    }

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
        let tensor = gpu
            .score_batch_native(true, &queries, &[0, 1, 2], 2, 2, &[0], &[2])
            .unwrap();
        assert!(
            batch_scores_close(&scores, &tensor),
            "tile={scores:?} tensor={tensor:?}"
        );
        assert!(gpu.calibration_runs > 0);
        assert!(
            gpu.calibration_complete,
            "startup calibration did not complete"
        );
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn tensor_core_matches_tile_for_320d_variable_queries() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 96 * 1024 * 1024, 48 * 1024 * 1024).unwrap();
        const DIM: usize = 320;
        const DOC_ROWS: usize = 11;
        let mut document = Vec::with_capacity(DIM * DOC_ROWS * 2);
        for row in 0..DOC_ROWS {
            for column in 0..DIM {
                let value = if (column + row * 7) % 31 == 0 {
                    0x3c00_u16
                } else {
                    0_u16
                };
                document.extend_from_slice(&value.to_le_bytes());
            }
        }
        gpu.upload_batch(&[(0, &document)]).unwrap();
        let query_offsets = [0_u32, 3, 8, 15];
        let mut queries = Vec::with_capacity(15 * DIM * 2);
        for row in 0..15 {
            for column in 0..DIM {
                let value = if (column * 3 + row * 5) % 29 == 0 {
                    0x3800_u16
                } else {
                    0_u16
                };
                queries.extend_from_slice(&value.to_le_bytes());
            }
        }
        let tile = gpu
            .score_batch_native(
                false,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &[0],
                &[DOC_ROWS as u32],
            )
            .unwrap();
        let tensor = gpu
            .score_batch_native(
                true,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &[0],
                &[DOC_ROWS as u32],
            )
            .unwrap();
        assert!(
            batch_scores_close(&tile, &tensor),
            "tile={tile:?} tensor={tensor:?}"
        );
    }

    #[test]
    fn adaptive_comparison_rejects_material_score_drift() {
        assert!(batch_scores_close(&[vec![1.0, 2.0]], &[vec![1.001, 2.001]]));
        assert!(!batch_scores_close(&[vec![1.0]], &[vec![1.1]]));
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
