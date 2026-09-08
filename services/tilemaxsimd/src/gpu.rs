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
use crate::dispatch::{TensorCalibrationBucket, TensorDispatchInput, choose_tensor};
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

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
    fn vctm_gpu_set_tile_queries_per_warp(gpu: *mut NativeGpu, value: u32) -> c_int;
    fn vctm_gpu_tile_queries_per_warp(gpu: *const NativeGpu) -> u32;
    fn vctm_gpu_set_pinned_control_staging(gpu: *mut NativeGpu, enabled: c_int) -> c_int;
    fn vctm_gpu_pinned_control_staging(gpu: *const NativeGpu) -> c_int;
    fn vctm_gpu_set_double_buffered_tile(gpu: *mut NativeGpu, enabled: c_int) -> c_int;
    fn vctm_gpu_double_buffered_tile(gpu: *const NativeGpu) -> c_int;
    fn vctm_gpu_set_pq_warp_task_max_document_rows(gpu: *mut NativeGpu, rows: u32) -> c_int;
    fn vctm_gpu_pq_warp_task_max_document_rows(gpu: *const NativeGpu) -> u32;
    fn vctm_gpu_set_tensor_chunk_candidates(gpu: *mut NativeGpu, candidates: usize) -> c_int;
    fn vctm_gpu_tensor_chunk_candidates(gpu: *const NativeGpu) -> usize;
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
        persisting_l2_bytes: *mut u64,
        matrix_engine_workspace_bytes: *mut u64,
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
        scoring_profile: u8,
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
    fn vctm_gpu_score_pq_batch(
        gpu: *mut NativeGpu,
        quantizer: *const NativeQuantizer,
        queries: *const c_uchar,
        query_bytes: usize,
        query_offsets: *const u32,
        request_count: u32,
        total_query_rows: u32,
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
    tensor_calibration_buckets: Vec<TensorCalibrationBucket>,
    tensor_runtime_disabled: bool,
    calibration_complete: bool,
    batch_warp_calls: u64,
    batch_tensor_calls: u64,
    calibration_runs: u64,
    calibration_failures: u64,
    info: DeviceInfo,
}

type TimedBatchScores = (Duration, Vec<Vec<f32>>);
type PairedBatchCalibration = (TimedBatchScores, TimedBatchScores);

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
        let mut persisting_l2_bytes = 0_u64;
        let mut matrix_engine_workspace_bytes = 0_u64;
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
                &mut persisting_l2_bytes,
                &mut matrix_engine_workspace_bytes,
            )
        };
        if name_status != 0 {
            unsafe { vctm_gpu_destroy(native.as_ptr()) };
            bail!("unable to determine CUDA device identity");
        }
        if let Err(reason) =
            validate_cuda_runtime(capability_status, info_status, major, runtime_version)
        {
            unsafe { vctm_gpu_destroy(native.as_ptr()) };
            bail!(reason);
        }
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
            tensor_calibration_buckets: Vec::new(),
            tensor_runtime_disabled: false,
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
                memory_clock_khz: (info_status == 0 && memory_clock_khz > 0)
                    .then_some(memory_clock_khz as u32),
                shared_memory_per_block_bytes: (info_status == 0)
                    .then_some(shared_memory_per_block_bytes),
                persisting_l2_bytes: (info_status == 0 && persisting_l2_bytes != 0)
                    .then_some(persisting_l2_bytes),
                matrix_engine_workspace_bytes: (info_status == 0
                    && matrix_engine_workspace_bytes != 0)
                    .then_some(matrix_engine_workspace_bytes),
                pinned_control_staging: Some(true),
                control_staging_speedup_milli: None,
                double_buffered_tile: Some(false),
                double_buffer_speedup_milli: None,
                document_tile_query_rows: match major {
                    9.. => Some(64),
                    8 => Some(32),
                    _ if capability_status == 0 => Some(8),
                    _ => None,
                },
                pq_warp_task_max_document_rows: Some(0),
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
                    asynchronous_copy: capability_status == 0 && major >= 8,
                    unified_memory: false,
                    persisting_l2: info_status == 0 && persisting_l2_bytes != 0,
                },
            },
        };
        gpu.calibrate_kernel_thresholds();
        gpu.calibrate_pq_adc_dispatch();
        // SAFETY: the native arena is still live and owns this tuning value.
        let queries_per_warp = unsafe { vctm_gpu_tile_queries_per_warp(gpu.native.as_ptr()) };
        gpu.info.document_tile_query_rows = queries_per_warp.checked_mul(8);
        gpu.info.pq_warp_task_max_document_rows =
            Some(unsafe { vctm_gpu_pq_warp_task_max_document_rows(gpu.native.as_ptr()) });
        gpu.info.pinned_control_staging =
            Some(unsafe { vctm_gpu_pinned_control_staging(gpu.native.as_ptr()) != 0 });
        gpu.info.double_buffered_tile =
            Some(unsafe { vctm_gpu_double_buffered_tile(gpu.native.as_ptr()) != 0 });
        Ok(gpu)
    }

    pub fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }

    pub fn adaptive_status(&self) -> AdaptiveStatus {
        AdaptiveStatus {
            tensor_threshold_rows: self.tensor_threshold_rows,
            calibration_complete: self.calibration_complete,
            batch_vector_calls: self.batch_warp_calls,
            batch_matrix_calls: self.batch_tensor_calls,
            calibration_runs: self.calibration_runs,
            calibration_failures: self.calibration_failures,
            tensor_calibration_buckets: self.tensor_calibration_buckets.clone(),
            tensor_chunk_candidates: u32::try_from(unsafe {
                vctm_gpu_tensor_chunk_candidates(self.native.as_ptr())
            })
            .unwrap_or(u32::MAX),
        }
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    pub fn score_pq_batch(
        &mut self,
        contract_id: &str,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let quantizer = self
            .quantizers
            .get(contract_id)
            .ok_or_else(|| anyhow!("PQ quantizer is not resident on this GPU"))?;
        if query_offsets.len() < 3 || query_offsets[0] != 0 {
            bail!("invalid PQ batch query offsets");
        }
        let total_query_rows = *query_offsets.last().unwrap();
        let scalar_bytes = match dtype {
            1 => 4_usize,
            2 => 2_usize,
            _ => bail!("unsupported PQ batch query dtype"),
        };
        let expected_bytes = (total_query_rows as usize)
            .checked_mul(dimension as usize)
            .and_then(|values| values.checked_mul(scalar_bytes))
            .ok_or_else(|| anyhow!("PQ batch query shape overflows address space"))?;
        if expected_bytes != queries.len() {
            bail!("PQ batch query byte length disagrees with its shape");
        }
        if document_offsets.len() != document_rows.len() || document_offsets.is_empty() {
            bail!("invalid PQ batch document metadata");
        }
        let request_count = query_offsets.len() - 1;
        let mut flat_output = vec![0.0_f32; request_count * document_offsets.len()];
        let mut error = [0_i8; 512];
        let status = unsafe {
            vctm_gpu_score_pq_batch(
                self.native.as_ptr(),
                quantizer.as_ptr(),
                queries.as_ptr(),
                queries.len(),
                query_offsets.as_ptr(),
                request_count as u32,
                total_query_rows,
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
            bail!("native PQ batch TileMaxSim returned a non-finite score");
        }
        Ok(flat_output
            .chunks_exact(document_offsets.len())
            .map(<[f32]>::to_vec)
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let total_rows = *query_offsets.last().unwrap_or(&0);
        let tensor_eligible = scoring_profile == 1
            && dtype == 2
            && self.info.capabilities.matrix_engine
            && !self.tensor_runtime_disabled;
        let fallback_bucket = TensorCalibrationBucket {
            candidate_count: 64,
            reference_document_rows: 32,
            reference_row_groups: 1,
            threshold_query_rows: self.tensor_threshold_rows,
            tensor_chunk_candidates: 64,
            ..TensorCalibrationBucket::default()
        };
        let buckets = if self.tensor_calibration_buckets.is_empty() {
            std::slice::from_ref(&fallback_bucket)
        } else {
            &self.tensor_calibration_buckets
        };
        let total_document_rows = document_rows
            .iter()
            .fold(0_u64, |total, rows| total.saturating_add(u64::from(*rows)));
        let document_row_groups =
            u32::try_from(document_rows.iter().copied().collect::<HashSet<_>>().len())
                .unwrap_or(u32::MAX);
        let tensor_decision = tensor_eligible
            .then(|| {
                choose_tensor(
                    TensorDispatchInput {
                        candidate_count: document_offsets.len(),
                        total_query_rows: total_rows,
                        total_document_rows,
                        document_row_groups,
                        dimension,
                    },
                    buckets,
                )
            })
            .flatten();
        if let Some(decision) = tensor_decision.filter(|decision| decision.use_tensor) {
            if let Some(bucket) = buckets.iter().find(|bucket| {
                bucket.candidate_count == decision.selected_candidate_bucket
                    && bucket.reference_document_rows == decision.selected_document_rows
                    && bucket.reference_row_groups == decision.selected_row_groups
            }) {
                let _ = unsafe {
                    vctm_gpu_set_tensor_chunk_candidates(
                        self.native.as_ptr(),
                        bucket.tensor_chunk_candidates.max(1) as usize,
                    )
                };
            }
            match self.score_batch_native(
                true,
                queries,
                query_offsets,
                dimension,
                dtype,
                document_offsets,
                document_rows,
                scoring_profile,
            ) {
                Ok(scores) => {
                    self.batch_tensor_calls += 1;
                    return Ok(scores);
                }
                Err(_) => {
                    self.calibration_failures += 1;
                    self.tensor_runtime_disabled = true;
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
            scoring_profile,
        )
    }

    fn calibrate_kernel_thresholds(&mut self) {
        if self.tensor_threshold_rows == u32::MAX {
            return;
        }
        const DIMENSION: u32 = 320;
        const DOCUMENT_ROWS: u32 = 32;
        const CANDIDATES: usize = 64;
        const REPETITIONS: usize = 3;
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
        let document_offsets = (0..CANDIDATES)
            .map(|index| index as u64 * document.len() as u64)
            .collect::<Vec<_>>();
        let uploads = document_offsets
            .iter()
            .map(|offset| (*offset, document.as_slice()))
            .collect::<Vec<_>>();
        let document_rows = vec![DOCUMENT_ROWS; CANDIDATES];
        if self.upload_batch(&uploads).is_err() {
            self.calibration_failures += 1;
            return;
        }
        self.calibrate_control_staging(
            DIMENSION,
            DOCUMENT_ROWS,
            &document_offsets,
            &document_rows,
            &one,
            &zero,
        );
        self.calibrate_double_buffered_tile(
            DIMENSION,
            DOCUMENT_ROWS,
            &document_offsets,
            &document_rows,
            &one,
            &zero,
        );
        self.calibrate_tile_reuse(
            DIMENSION,
            DOCUMENT_ROWS,
            REPETITIONS,
            &document_offsets,
            &document_rows,
            &one,
            &zero,
        );
        self.calibrate_tensor_dispatch_buckets(DIMENSION, &one, &zero);
    }

    fn calibrate_tensor_dispatch_buckets(&mut self, dimension: u32, one: &[u8; 2], zero: &[u8; 2]) {
        const UNIFORM_ROWS: [u32; 1] = [32];
        const MODERATE_ROW_GROUPS: [u32; 8] = [8, 16, 24, 32, 40, 48, 56, 64];
        const FRAGMENTED_ROW_GROUPS: [u32; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let fallback = self.tensor_threshold_rows;
        let mut buckets = Vec::new();
        for candidates in [64_usize, 512, 4096] {
            for row_pattern in [
                UNIFORM_ROWS.as_slice(),
                MODERATE_ROW_GROUPS.as_slice(),
                FRAGMENTED_ROW_GROUPS.as_slice(),
            ] {
                if let Some(bucket) =
                    self.calibrate_tensor_bucket(dimension, candidates, row_pattern, one, zero)
                {
                    buckets.push(bucket);
                }
            }
        }
        if buckets.is_empty() {
            self.tensor_threshold_rows = fallback;
            self.calibration_complete = false;
            return;
        }
        self.tensor_threshold_rows = buckets
            .iter()
            .map(|bucket| bucket.threshold_query_rows)
            .min()
            .unwrap_or(fallback);
        self.tensor_calibration_buckets = buckets;
        self.calibration_complete = true;
    }

    fn calibrate_tensor_bucket(
        &mut self,
        dimension: u32,
        candidates: usize,
        row_pattern: &[u32],
        one: &[u8; 2],
        zero: &[u8; 2],
    ) -> Option<TensorCalibrationBucket> {
        const REPETITIONS: usize = 3;
        let templates = row_pattern
            .iter()
            .map(|rows| {
                let mut document = Vec::with_capacity(*rows as usize * dimension as usize * 2);
                for row in 0..*rows {
                    for column in 0..dimension {
                        document.extend_from_slice(if column == row % dimension {
                            one
                        } else {
                            zero
                        });
                    }
                }
                document
            })
            .collect::<Vec<_>>();
        let mut cursor = 0_u64;
        let mut document_offsets = Vec::with_capacity(candidates);
        let mut document_rows = Vec::with_capacity(candidates);
        for candidate in 0..candidates {
            let template = candidate % templates.len();
            document_offsets.push(cursor);
            document_rows.push(row_pattern[template]);
            cursor = cursor.checked_add(templates[template].len() as u64)?;
        }
        if cursor > self.tensor_bytes as u64 {
            return None;
        }
        let uploads = document_offsets
            .iter()
            .enumerate()
            .map(|(candidate, offset)| (*offset, templates[candidate % templates.len()].as_slice()))
            .collect::<Vec<_>>();
        if self.upload_batch(&uploads).is_err() {
            self.calibration_failures += 1;
            return None;
        }
        let chunk_candidates = self.calibrate_tensor_chunk_candidates(
            dimension,
            candidates,
            &document_offsets,
            &document_rows,
            one,
            zero,
        )?;
        if unsafe { vctm_gpu_set_tensor_chunk_candidates(self.native.as_ptr(), chunk_candidates) }
            != 0
        {
            self.calibration_failures += 1;
            return None;
        }

        let mut threshold = u32::MAX;
        let mut threshold_times = (0_u64, 0_u64);
        let mut successful = false;
        for total_rows in [64_u32, 256, 512, 1024, 2048] {
            let (queries, offsets) = calibration_queries(total_rows, dimension, 32, one, zero);
            self.calibration_runs += 1;
            let paired = paired_kernel_measurement(
                self,
                &queries,
                &offsets,
                dimension,
                &document_offsets,
                &document_rows,
                REPETITIONS,
            );
            let Ok(((tile_elapsed, tile), (tensor_elapsed, tensor))) = paired else {
                self.calibration_failures += 1;
                continue;
            };
            if !batch_scores_close(&tile, &tensor) {
                self.calibration_failures += 1;
                continue;
            }
            successful = true;
            threshold_times = (
                duration_ns_u64(tile_elapsed),
                duration_ns_u64(tensor_elapsed),
            );
            if tensor_elapsed.as_nanos().saturating_mul(100)
                < tile_elapsed.as_nanos().saturating_mul(95)
            {
                threshold = total_rows;
                break;
            }
        }
        successful.then(|| TensorCalibrationBucket {
            candidate_count: u32::try_from(candidates).unwrap_or(u32::MAX),
            reference_document_rows: u32::try_from(
                document_rows
                    .iter()
                    .map(|rows| u64::from(*rows))
                    .sum::<u64>()
                    / candidates as u64,
            )
            .unwrap_or(u32::MAX),
            reference_row_groups: u32::try_from(row_pattern.len()).unwrap_or(u32::MAX),
            threshold_query_rows: threshold,
            tile_time_ns: threshold_times.0,
            tensor_time_ns: threshold_times.1,
            tensor_chunk_candidates: u32::try_from(chunk_candidates).unwrap_or(u32::MAX),
        })
    }

    fn calibrate_tensor_chunk_candidates(
        &mut self,
        dimension: u32,
        candidates: usize,
        document_offsets: &[u64],
        document_rows: &[u32],
        one: &[u8; 2],
        zero: &[u8; 2],
    ) -> Option<usize> {
        const REPETITIONS: usize = 3;
        let (queries, offsets) = calibration_queries(512, dimension, 32, one, zero);
        let mut choices = [64_usize, 256, 1024, 4096, candidates]
            .into_iter()
            .filter(|choice| *choice <= candidates)
            .collect::<Vec<_>>();
        choices.sort_unstable();
        choices.dedup();
        let mut best: Option<(usize, Duration)> = None;
        let mut oracle: Option<Vec<Vec<f32>>> = None;
        for chunk in choices {
            if unsafe { vctm_gpu_set_tensor_chunk_candidates(self.native.as_ptr(), chunk) } != 0 {
                continue;
            }
            let mut samples = Vec::with_capacity(REPETITIONS);
            let mut scores = None;
            if self
                .score_batch_native(
                    true,
                    &queries,
                    &offsets,
                    dimension,
                    2,
                    document_offsets,
                    document_rows,
                    1,
                )
                .is_err()
            {
                continue;
            }
            for _ in 0..REPETITIONS {
                let started = Instant::now();
                let Ok(output) = self.score_batch_native(
                    true,
                    &queries,
                    &offsets,
                    dimension,
                    2,
                    document_offsets,
                    document_rows,
                    1,
                ) else {
                    samples.clear();
                    break;
                };
                samples.push(started.elapsed());
                scores = Some(output);
            }
            if samples.len() != REPETITIONS {
                continue;
            }
            samples.sort_unstable();
            let scores = scores?;
            if let Some(expected) = &oracle {
                if !batch_scores_close(expected, &scores) {
                    self.calibration_failures += 1;
                    continue;
                }
            } else {
                oracle = Some(scores);
            }
            let elapsed = samples[REPETITIONS / 2];
            if best.is_none_or(|(_, current)| {
                elapsed.as_nanos().saturating_mul(100) < current.as_nanos().saturating_mul(98)
            }) {
                best = Some((chunk, elapsed));
            }
        }
        self.calibration_runs += 1;
        best.map(|(chunk, _)| chunk)
    }

    fn calibrate_control_staging(
        &mut self,
        dimension: u32,
        document_rows_per_candidate: u32,
        document_offsets: &[u64],
        document_rows: &[u32],
        one: &[u8; 2],
        zero: &[u8; 2],
    ) {
        const QUERY_ROWS: u32 = 32;
        const REPETITIONS: usize = 7;
        let mut queries = Vec::with_capacity(QUERY_ROWS as usize * dimension as usize * 2);
        for row in 0..QUERY_ROWS {
            for column in 0..dimension {
                queries.extend_from_slice(if column == row % document_rows_per_candidate {
                    one
                } else {
                    zero
                });
            }
        }
        let offsets = [0, QUERY_ROWS / 2, QUERY_ROWS];
        let run_once = |gpu: &mut Self, enabled: bool| -> Result<(Duration, Vec<Vec<f32>>)> {
            if unsafe {
                vctm_gpu_set_pinned_control_staging(gpu.native.as_ptr(), i32::from(enabled))
            } != 0
            {
                bail!("CUDA rejected control-staging calibration mode");
            }
            let started = Instant::now();
            let scores = gpu.score_batch_native(
                false,
                &queries,
                &offsets,
                dimension,
                2,
                document_offsets,
                document_rows,
                1,
            )?;
            Ok((started.elapsed(), scores))
        };
        self.calibration_runs += 1;
        // Warm both variants, then alternate AB/BA order. Measuring every A
        // sample before every B sample mistakes clock ramp and cache warmth
        // for a backend optimization on shared production GPUs.
        let calibration = (|| -> Result<PairedBatchCalibration> {
            run_once(self, false)?;
            run_once(self, true)?;
            let mut samples = [
                Vec::with_capacity(REPETITIONS),
                Vec::with_capacity(REPETITIONS),
            ];
            let mut scores = [Vec::new(), Vec::new()];
            for repetition in 0..REPETITIONS {
                let order = if repetition & 1 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for enabled in order {
                    let (elapsed, output) = run_once(self, enabled)?;
                    let index = usize::from(enabled);
                    samples[index].push(elapsed);
                    scores[index] = output;
                }
            }
            samples[0].sort_unstable();
            samples[1].sort_unstable();
            Ok((
                (samples[0][REPETITIONS / 2], std::mem::take(&mut scores[0])),
                (samples[1][REPETITIONS / 2], std::mem::take(&mut scores[1])),
            ))
        })();
        let (pageable, pinned) = calibration.map_or_else(
            |error| (Err(error), Err(anyhow!("paired calibration failed"))),
            |(pageable, pinned)| (Ok(pageable), Ok(pinned)),
        );
        let (use_pinned, speedup_milli) = match (pageable, pinned) {
            (Ok((pageable_elapsed, pageable_scores)), Ok((pinned_elapsed, pinned_scores)))
                if batch_scores_close(&pageable_scores, &pinned_scores) =>
            {
                let ratio = pageable_elapsed
                    .as_nanos()
                    .saturating_mul(1000)
                    .checked_div(pinned_elapsed.as_nanos().max(1))
                    .unwrap_or(0)
                    .min(u32::MAX as u128) as u32;
                // Require a small but material win so noise on a shared device
                // does not force an extra host memcpy on every request.
                (
                    pinned_elapsed.as_nanos().saturating_mul(100)
                        < pageable_elapsed.as_nanos().saturating_mul(98),
                    Some(ratio),
                )
            }
            _ => {
                self.calibration_failures += 1;
                (false, None)
            }
        };
        self.info.control_staging_speedup_milli = speedup_milli;
        unsafe {
            vctm_gpu_set_pinned_control_staging(self.native.as_ptr(), i32::from(use_pinned));
        }
    }

    fn calibrate_double_buffered_tile(
        &mut self,
        dimension: u32,
        document_rows_per_candidate: u32,
        document_offsets: &[u64],
        document_rows: &[u32],
        one: &[u8; 2],
        zero: &[u8; 2],
    ) {
        const QUERY_ROWS: u32 = 256;
        const REPETITIONS: usize = 7;
        let row_bytes = dimension as usize * size_of::<u16>();
        if row_bytes > self.info.shared_memory_per_block_bytes.unwrap_or(0) as usize / 2 {
            return;
        }
        let mut queries = Vec::with_capacity(QUERY_ROWS as usize * dimension as usize * 2);
        for row in 0..QUERY_ROWS {
            for column in 0..dimension {
                queries.extend_from_slice(if column == row % document_rows_per_candidate {
                    one
                } else {
                    zero
                });
            }
        }
        let offsets = [0, QUERY_ROWS / 2, QUERY_ROWS];
        let run_once = |gpu: &mut Self, enabled: bool| -> Result<(Duration, Vec<Vec<f32>>)> {
            if unsafe { vctm_gpu_set_double_buffered_tile(gpu.native.as_ptr(), i32::from(enabled)) }
                != 0
            {
                bail!("CUDA rejected tile-buffer calibration mode");
            }
            let started = Instant::now();
            let scores = gpu.score_batch_native(
                false,
                &queries,
                &offsets,
                dimension,
                2,
                document_offsets,
                document_rows,
                1,
            )?;
            Ok((started.elapsed(), scores))
        };
        self.calibration_runs += 1;
        let calibration = (|| -> Result<PairedBatchCalibration> {
            run_once(self, false)?;
            run_once(self, true)?;
            let mut samples = [
                Vec::with_capacity(REPETITIONS),
                Vec::with_capacity(REPETITIONS),
            ];
            let mut scores = [Vec::new(), Vec::new()];
            for repetition in 0..REPETITIONS {
                let order = if repetition & 1 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for enabled in order {
                    let (elapsed, output) = run_once(self, enabled)?;
                    let index = usize::from(enabled);
                    samples[index].push(elapsed);
                    scores[index] = output;
                }
            }
            samples[0].sort_unstable();
            samples[1].sort_unstable();
            Ok((
                (samples[0][REPETITIONS / 2], std::mem::take(&mut scores[0])),
                (samples[1][REPETITIONS / 2], std::mem::take(&mut scores[1])),
            ))
        })();
        let (single, double) = calibration.map_or_else(
            |error| (Err(error), Err(anyhow!("paired calibration failed"))),
            |(single, double)| (Ok(single), Ok(double)),
        );
        let (use_double, speedup_milli) = match (single, double) {
            (Ok((single_elapsed, single_scores)), Ok((double_elapsed, double_scores)))
                if batch_scores_close(&single_scores, &double_scores) =>
            {
                let ratio = single_elapsed
                    .as_nanos()
                    .saturating_mul(1000)
                    .checked_div(double_elapsed.as_nanos().max(1))
                    .unwrap_or(0)
                    .min(u32::MAX as u128) as u32;
                (
                    double_elapsed.as_nanos().saturating_mul(100)
                        < single_elapsed.as_nanos().saturating_mul(95),
                    Some(ratio),
                )
            }
            _ => {
                self.calibration_failures += 1;
                (false, None)
            }
        };
        self.info.double_buffer_speedup_milli = speedup_milli;
        unsafe {
            vctm_gpu_set_double_buffered_tile(self.native.as_ptr(), i32::from(use_double));
        }
    }

    fn calibrate_pq_adc_dispatch(&mut self) {
        const DIMENSION: u32 = 320;
        const SUBSPACES: u16 = 20;
        const CENTROIDS: u16 = 256;
        const QUERY_ROWS: u32 = 32;
        const REQUESTS: u32 = 8;
        // Match the resident rerank scale used by the production benchmark.
        // A small candidate set can hide occupancy losses and choose an
        // over-aggressive crossover for the common larger batch.
        const CANDIDATES: usize = 512;
        const REPETITIONS: usize = 5;
        const CONTRACT: &str = "__startup_pq_adc_calibration__";
        let subdimension = DIMENSION as usize / SUBSPACES as usize;
        let codebook_values = SUBSPACES as usize * CENTROIDS as usize * subdimension;
        let mut codebook = Vec::with_capacity(codebook_values * size_of::<f32>());
        for index in 0..codebook_values {
            codebook
                .extend_from_slice(&(if index % 97 == 0 { 0.125_f32 } else { 0.0 }).to_le_bytes());
        }
        if self
            .ensure_quantizer(CONTRACT, &codebook, DIMENSION, 1, SUBSPACES, CENTROIDS, 0)
            .is_err()
        {
            self.calibration_failures += 1;
            return;
        }
        let query = (0..QUERY_ROWS as usize * DIMENSION as usize)
            .flat_map(|index| (if index % 89 == 0 { 0.5_f32 } else { 0.0 }).to_le_bytes())
            .collect::<Vec<_>>();
        let queries = (0..REQUESTS)
            .flat_map(|_| query.iter().copied())
            .collect::<Vec<_>>();
        let query_offsets = (0..=REQUESTS)
            .map(|request| request * QUERY_ROWS)
            .collect::<Vec<_>>();
        let mut selected = 0_u32;
        for candidate_rows in [2_u32, 4, 8] {
            let document = (0..candidate_rows as usize * SUBSPACES as usize)
                .map(|index| (index % CENTROIDS as usize) as u8)
                .collect::<Vec<_>>();
            let document_offsets = (0..CANDIDATES)
                .map(|candidate| candidate as u64 * document.len() as u64)
                .collect::<Vec<_>>();
            let uploads = document_offsets
                .iter()
                .copied()
                .map(|offset| (offset, document.as_slice()))
                .collect::<Vec<_>>();
            let document_rows = vec![candidate_rows; CANDIDATES];
            if self.upload_batch(&uploads).is_err() {
                self.calibration_failures += 1;
                break;
            }
            let measure = |gpu: &mut Self, warp_task_rows| {
                unsafe {
                    vctm_gpu_set_pq_warp_task_max_document_rows(gpu.native.as_ptr(), warp_task_rows)
                };
                let mut samples = Vec::with_capacity(REPETITIONS);
                let mut scores = Vec::new();
                for _ in 0..REPETITIONS {
                    let started = Instant::now();
                    scores = gpu.score_pq_batch(
                        CONTRACT,
                        &queries,
                        &query_offsets,
                        DIMENSION,
                        1,
                        &document_offsets,
                        &document_rows,
                    )?;
                    samples.push(started.elapsed());
                }
                samples.sort_unstable();
                Ok::<_, anyhow::Error>((samples[REPETITIONS / 2], scores))
            };
            self.calibration_runs += 1;
            let Ok((cooperative_elapsed, cooperative)) = measure(self, 0) else {
                self.calibration_failures += 1;
                break;
            };
            let Ok((warp_elapsed, warp)) = measure(self, candidate_rows) else {
                self.calibration_failures += 1;
                break;
            };
            if !batch_scores_close(&cooperative, &warp) {
                self.calibration_failures += 1;
                break;
            }
            // Demand a material win so startup noise cannot select a fragile
            // architecture threshold that regresses production traffic.
            if warp_elapsed.as_nanos().saturating_mul(100)
                < cooperative_elapsed.as_nanos().saturating_mul(95)
            {
                selected = candidate_rows;
            } else {
                break;
            }
        }
        unsafe {
            vctm_gpu_set_pq_warp_task_max_document_rows(self.native.as_ptr(), selected);
        }
        self.retain_quantizers(&HashSet::new());
    }

    #[allow(clippy::too_many_arguments)]
    fn calibrate_tile_reuse(
        &mut self,
        dimension: u32,
        document_rows_per_candidate: u32,
        repetitions: usize,
        document_offsets: &[u64],
        document_rows: &[u32],
        one: &[u8; 2],
        zero: &[u8; 2],
    ) {
        // SM80+ has the async tiled implementation. Older devices retain the
        // scalar-compatible single-query-per-warp path.
        if !self.info.capabilities.asynchronous_copy {
            return;
        }
        const TOTAL_ROWS: u32 = 256;
        let mut queries = Vec::with_capacity(TOTAL_ROWS as usize * dimension as usize * 2);
        for row in 0..TOTAL_ROWS {
            for column in 0..dimension {
                queries.extend_from_slice(if column == row % document_rows_per_candidate {
                    one
                } else {
                    zero
                });
            }
        }
        let offsets = [0, TOTAL_ROWS / 2, TOTAL_ROWS];
        // SAFETY: the native handle is live and accepted values are fixed by
        // the C ABI. Restore the original value unless a numerically equivalent
        // faster variant completes all repetitions.
        let original = unsafe { vctm_gpu_tile_queries_per_warp(self.native.as_ptr()) };
        let mut oracle = None::<Vec<Vec<f32>>>;
        let mut best = None::<(Duration, u32)>;
        for candidate in [1_u32, 4, 8] {
            if unsafe { vctm_gpu_set_tile_queries_per_warp(self.native.as_ptr(), candidate) } != 0 {
                continue;
            }
            self.calibration_runs += 1;
            let started = Instant::now();
            let mut scores = None;
            for _ in 0..repetitions {
                match self.score_batch_native(
                    false,
                    &queries,
                    &offsets,
                    dimension,
                    2,
                    document_offsets,
                    document_rows,
                    1,
                ) {
                    Ok(value) => scores = Some(value),
                    Err(_) => {
                        self.calibration_failures += 1;
                        scores = None;
                        break;
                    }
                }
            }
            let elapsed = started.elapsed() / repetitions as u32;
            let Some(scores) = scores else { continue };
            if let Some(reference) = oracle.as_ref() {
                if !batch_scores_close(reference, &scores) {
                    self.calibration_failures += 1;
                    continue;
                }
            } else {
                oracle = Some(scores);
            }
            if best.is_none_or(|(best_elapsed, _)| elapsed < best_elapsed) {
                best = Some((elapsed, candidate));
            }
        }
        let selected = best.map_or(original, |(_, candidate)| candidate);
        // SAFETY: selected is either the original native value or one of the
        // three accepted candidates above.
        if unsafe { vctm_gpu_set_tile_queries_per_warp(self.native.as_ptr(), selected) } != 0 {
            self.calibration_failures += 1;
            let _ = unsafe { vctm_gpu_set_tile_queries_per_warp(self.native.as_ptr(), original) };
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
        scoring_profile: u8,
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
                    scoring_profile,
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

fn calibration_queries(
    total_rows: u32,
    dimension: u32,
    document_rows: u32,
    one: &[u8; 2],
    zero: &[u8; 2],
) -> (Vec<u8>, Vec<u32>) {
    let mut queries = Vec::with_capacity(total_rows as usize * dimension as usize * 2);
    for row in 0..total_rows {
        for column in 0..dimension {
            queries.extend_from_slice(if column == row % document_rows {
                one
            } else {
                zero
            });
        }
    }
    let request_rows = 32;
    let offsets = (0..=total_rows / request_rows)
        .map(|request| request * request_rows)
        .collect();
    (queries, offsets)
}

#[allow(clippy::too_many_arguments)]
fn paired_kernel_measurement(
    gpu: &mut Gpu,
    queries: &[u8],
    query_offsets: &[u32],
    dimension: u32,
    document_offsets: &[u64],
    document_rows: &[u32],
    repetitions: usize,
) -> Result<PairedBatchCalibration> {
    let run_once = |gpu: &mut Gpu, tensor| -> Result<TimedBatchScores> {
        let started = Instant::now();
        let scores = gpu.score_batch_native(
            tensor,
            queries,
            query_offsets,
            dimension,
            2,
            document_offsets,
            document_rows,
            1,
        )?;
        Ok((started.elapsed(), scores))
    };
    run_once(gpu, false)?;
    run_once(gpu, true)?;
    let mut samples = [
        Vec::with_capacity(repetitions),
        Vec::with_capacity(repetitions),
    ];
    let mut scores = [Vec::new(), Vec::new()];
    for repetition in 0..repetitions {
        let order = if repetition & 1 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for tensor in order {
            let (elapsed, output) = run_once(gpu, tensor)?;
            let index = usize::from(tensor);
            samples[index].push(elapsed);
            scores[index] = output;
        }
    }
    samples[0].sort_unstable();
    samples[1].sort_unstable();
    Ok((
        (samples[0][repetitions / 2], std::mem::take(&mut scores[0])),
        (samples[1][repetitions / 2], std::mem::take(&mut scores[1])),
    ))
}

fn duration_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn validate_cuda_runtime(
    capability_status: c_int,
    info_status: c_int,
    major: c_int,
    runtime_version: c_int,
) -> Result<()> {
    if capability_status != 0 {
        bail!("unable to determine CUDA compute capability");
    }
    if info_status != 0 {
        bail!("unable to determine CUDA driver, runtime, or device capabilities");
    }
    if major < 8 {
        bail!(
            "CUDA backend requires compute capability 8.0 or newer; use the CPU backend on this device"
        );
    }
    if major >= 10 && runtime_version < 13_000 {
        bail!(
            "Blackwell-class CUDA devices require the CUDA 13 tilemaxsimd image; the loaded runtime reports {runtime_version}"
        );
    }
    Ok(())
}

impl AcceleratorBackend for Gpu {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn tensor_bytes(&self) -> usize {
        Gpu::tensor_bytes(self)
    }

    fn adaptive_status(&self) -> AdaptiveStatus {
        Gpu::adaptive_status(self)
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
        Gpu::score_pq_batch(
            self,
            contract_id,
            queries,
            query_offsets,
            dimension,
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
        scoring_profile: u8,
        document_offsets: &[u64],
        document_rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        Gpu::score_batch(
            self,
            queries,
            query_offsets,
            dimension,
            dtype,
            scoring_profile,
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
    fn runtime_gate_matches_published_cuda_artifacts() {
        assert!(validate_cuda_runtime(0, 0, 8, 12_060).is_ok());
        assert!(validate_cuda_runtime(0, 0, 9, 12_060).is_ok());
        assert!(validate_cuda_runtime(0, 0, 12, 13_030).is_ok());
        assert!(validate_cuda_runtime(1, 0, 9, 13_030).is_err());
        assert!(validate_cuda_runtime(0, 1, 9, 13_030).is_err());
        assert!(validate_cuda_runtime(0, 0, 7, 12_060).is_err());
        assert!(validate_cuda_runtime(0, 0, 10, 12_060).is_err());
        assert!(validate_cuda_runtime(0, 0, 12, 12_060).is_err());
    }

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
        assert!(info.document_tile_query_rows.unwrap_or_default() >= 8);
        assert!(info.pq_warp_task_max_document_rows.unwrap_or(u32::MAX) <= 8);
        assert!(info.pinned_control_staging.is_some());
        assert!(info.control_staging_speedup_milli.is_some());
        assert!(info.double_buffered_tile.is_some());
        assert!(info.double_buffer_speedup_milli.is_some());
        let adaptive = gpu.adaptive_status();
        assert!(!adaptive.tensor_calibration_buckets.is_empty());
        eprintln!(
            "cuda_tuning architecture={} document_tile_query_rows={:?} pq_warp_task_max_document_rows={:?} pinned_control_staging={:?} control_staging_speedup_milli={:?} double_buffered_tile={:?} double_buffer_speedup_milli={:?} tensor_chunk_candidates={} tensor_calibration_buckets={:?}",
            info.architecture,
            info.document_tile_query_rows,
            info.pq_warp_task_max_document_rows,
            info.pinned_control_staging,
            info.control_staging_speedup_milli,
            info.double_buffered_tile,
            info.double_buffer_speedup_milli,
            adaptive.tensor_chunk_candidates,
            adaptive.tensor_calibration_buckets,
        );
        assert_eq!(
            info.capabilities.persisting_l2,
            info.persisting_l2_bytes.is_some()
        );
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn passes_vendor_neutral_conformance_probe() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let report = crate::backend::run_conformance_probe(&mut gpu).unwrap();
        assert_eq!(report.backend, BackendKind::Cuda);
        assert_eq!(report.exact_fp32_score, report.expected_score);
        assert_eq!(report.exact_fp16_score, report.expected_score);
        assert_eq!(report.batched_fp16_scores, Some(vec![1.0, 0.5]));
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
            .score_batch(&queries, &[0, 1, 2], 2, 2, 1, &[0], &[2])
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
        assert!((scores[1][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
        let tensor = gpu
            .score_batch_native(true, &queries, &[0, 1, 2], 2, 2, &[0], &[2], 1)
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
        let document_offsets = (0..4)
            .map(|index| index * document.len() as u64)
            .collect::<Vec<_>>();
        let uploads = document_offsets
            .iter()
            .map(|offset| (*offset, document.as_slice()))
            .collect::<Vec<_>>();
        gpu.upload_batch(&uploads).unwrap();
        let document_rows = vec![DOC_ROWS as u32; document_offsets.len()];
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
                &document_offsets,
                &document_rows,
                1,
            )
            .unwrap();
        let tensor = gpu
            .score_batch_native(
                true,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &document_offsets,
                &document_rows,
                1,
            )
            .unwrap();
        assert!(
            batch_scores_close(&tile, &tensor),
            "tile={tile:?} tensor={tensor:?}"
        );
        // Reuse the cached row plan once, then change one candidate shape and
        // prove the plan is invalidated rather than retaining stale grouping.
        let repeated = gpu
            .score_batch_native(
                true,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &document_offsets,
                &document_rows,
                1,
            )
            .unwrap();
        assert!(batch_scores_close(&tensor, &repeated));
        let mut changed_rows = document_rows.clone();
        changed_rows[0] -= 1;
        let changed_tile = gpu
            .score_batch_native(
                false,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &document_offsets,
                &changed_rows,
                1,
            )
            .unwrap();
        let changed_tensor = gpu
            .score_batch_native(
                true,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &document_offsets,
                &changed_rows,
                1,
            )
            .unwrap();
        assert!(batch_scores_close(&changed_tile, &changed_tensor));
    }

    #[test]
    #[ignore = "microbenchmark requires an explicitly assigned CUDA device"]
    fn benchmark_batched_tensor_core_for_320d_candidates() {
        let bench_size = |name: &str, default: usize, maximum: usize| {
            let value = std::env::var(name).map_or(default, |raw| raw.parse::<usize>().unwrap());
            assert!(
                value > 0 && value <= maximum,
                "{name} must be in 1..={maximum}"
            );
            value
        };
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        const DIM: usize = 320;
        let document_rows = bench_size("VCTM_BENCH_DOCUMENT_ROWS", 32, 256);
        let candidates = bench_size("VCTM_BENCH_CANDIDATES", 64, 100_000);
        let requests = bench_size("VCTM_BENCH_REQUESTS", 8, 256);
        assert!(requests >= 2, "VCTM_BENCH_REQUESTS must be at least 2");
        let query_rows = bench_size("VCTM_BENCH_QUERY_ROWS", 32, 256);
        let warmups = bench_size("VCTM_BENCH_WARMUPS", 3, 100);
        let iterations = bench_size("VCTM_BENCH_ITERATIONS", 20, 100);
        let document_bytes = document_rows * DIM * 2;
        let resident_bytes = candidates
            .checked_mul(document_bytes)
            .and_then(|bytes| bytes.checked_add(128 * 1024 * 1024))
            .expect("benchmark resident arena size overflow");
        let workspace_bytes = candidates
            .checked_mul(requests)
            .and_then(|values| values.checked_mul(query_rows))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_add(64 * 1024 * 1024))
            .expect("benchmark workspace size overflow");
        let total_bytes = resident_bytes
            .checked_add(workspace_bytes)
            .expect("benchmark GPU arena size overflow");
        let mut gpu = Gpu::create(device, total_bytes, workspace_bytes).unwrap();
        let document = (0..document_rows * DIM)
            .flat_map(|index| if index % 37 == 0 { 0x3c00_u16 } else { 0_u16 }.to_le_bytes())
            .collect::<Vec<_>>();
        let offsets = (0..candidates)
            .map(|index| index as u64 * document.len() as u64)
            .collect::<Vec<_>>();
        let uploads = offsets
            .iter()
            .map(|offset| (*offset, document.as_slice()))
            .collect::<Vec<_>>();
        gpu.upload_batch(&uploads).unwrap();
        let rows = vec![document_rows as u32; candidates];
        let queries = (0..requests * query_rows * DIM)
            .flat_map(|index| if index % 41 == 0 { 0x3800_u16 } else { 0_u16 }.to_le_bytes())
            .collect::<Vec<_>>();
        let query_offsets = (0..=requests)
            .map(|index| (index * query_rows) as u32)
            .collect::<Vec<_>>();
        for _ in 0..warmups {
            gpu.score_batch_native(
                true,
                &queries,
                &query_offsets,
                DIM as u32,
                2,
                &offsets,
                &rows,
                1,
            )
            .unwrap();
        }
        let measure = |gpu: &mut Gpu, tensor| {
            let started = Instant::now();
            let mut result = Vec::new();
            for _ in 0..iterations {
                result = gpu
                    .score_batch_native(
                        tensor,
                        &queries,
                        &query_offsets,
                        DIM as u32,
                        2,
                        &offsets,
                        &rows,
                        1,
                    )
                    .unwrap();
            }
            (
                started.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
                result,
            )
        };
        unsafe {
            assert_eq!(vctm_gpu_set_double_buffered_tile(gpu.native.as_ptr(), 0), 0);
        }
        let (tile_ms, tile) = measure(&mut gpu, false);
        unsafe {
            assert_eq!(vctm_gpu_set_double_buffered_tile(gpu.native.as_ptr(), 1), 0);
        }
        let (double_buffer_ms, double_buffer) = measure(&mut gpu, false);
        unsafe {
            assert_eq!(vctm_gpu_set_double_buffered_tile(gpu.native.as_ptr(), 0), 0);
        }
        let (tensor_ms, tensor) = measure(&mut gpu, true);
        assert!(batch_scores_close(&tile, &double_buffer));
        assert!(batch_scores_close(&tile, &tensor));
        let tensor_calls_before = gpu.batch_tensor_calls;
        let auto = gpu
            .score_batch(&queries, &query_offsets, DIM as u32, 2, 1, &offsets, &rows)
            .unwrap();
        assert!(batch_scores_close(&tile, &auto));
        let automatic_kernel = if gpu.batch_tensor_calls > tensor_calls_before {
            "tensor"
        } else {
            "tile"
        };
        eprintln!(
            "tilemaxsim_320d candidates={candidates} requests={requests} query_rows={query_rows} document_rows={document_rows} warmups={warmups} iterations={iterations} tile_query_rows={} calibrated_threshold_rows={} automatic_kernel={automatic_kernel} tensor_chunk_candidates={} single_buffer_ms={tile_ms:.4} double_buffer_ms={double_buffer_ms:.4} double_buffer_speedup={:.3} tensor_ms={tensor_ms:.4} tensor_speedup={:.3}",
            gpu.info.document_tile_query_rows.unwrap_or_default(),
            gpu.tensor_threshold_rows,
            unsafe { vctm_gpu_tensor_chunk_candidates(gpu.native.as_ptr()) },
            tile_ms / double_buffer_ms,
            tile_ms / tensor_ms
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

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn quantized_multiquery_decodes_each_document_tile_once() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 64 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
        let mut int8_document = vec![127_u8, 0, 0, 127];
        int8_document.extend_from_slice(&(1.0_f32 / 127.0).to_le_bytes());
        int8_document.extend_from_slice(&(1.0_f32 / 127.0).to_le_bytes());
        let mut fp8_document = vec![0x38_u8, 0, 0, 0x38];
        fp8_document.extend_from_slice(&1.0_f32.to_le_bytes());
        fp8_document.extend_from_slice(&1.0_f32.to_le_bytes());
        gpu.upload_batch(&[(0, &int8_document), (256, &fp8_document)])
            .unwrap();
        let queries = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for (profile, offset) in [(2, 0), (3, 256)] {
            let scores = gpu
                .score_batch(&queries, &[0, 1, 2], 2, 1, profile, &[offset], &[2])
                .unwrap();
            assert_eq!(scores.len(), 2);
            assert!((scores[0][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
            assert!((scores[1][0] - 1.0).abs() < 1e-5, "scores={scores:?}");
        }
    }

    #[test]
    #[ignore = "microbenchmark requires an explicitly assigned CUDA device"]
    fn benchmark_quantized_fused_scoring_for_320d_candidates() {
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<i32>()
            .unwrap();
        let mut gpu = Gpu::create(device, 96 * 1024 * 1024, 48 * 1024 * 1024).unwrap();
        const DIM: usize = 320;
        const ROWS: usize = 32;
        const CANDIDATES: usize = 512;
        const QUERY_ROWS: usize = 32;
        let mut exact = Vec::with_capacity(ROWS * DIM * 2);
        let mut int8 = Vec::with_capacity(ROWS * DIM + ROWS * 4);
        let mut fp8 = Vec::with_capacity(ROWS * DIM + ROWS * 4);
        for index in 0..ROWS * DIM {
            let nonzero = index % 37 == 0;
            exact.extend_from_slice(&(if nonzero { 0x3c00_u16 } else { 0 }).to_le_bytes());
            int8.push(if nonzero { 127_u8 } else { 0 });
            fp8.push(if nonzero { 0x38_u8 } else { 0 });
        }
        for _ in 0..ROWS {
            int8.extend_from_slice(&(1.0_f32 / 127.0).to_le_bytes());
            fp8.extend_from_slice(&1.0_f32.to_le_bytes());
        }
        let bases = [0_u64, 16 * 1024 * 1024, 24 * 1024 * 1024];
        let payloads = [&exact, &int8, &fp8];
        let mut uploads = Vec::new();
        let mut offsets: [Vec<u64>; 3] = std::array::from_fn(|_| Vec::new());
        for (profile, payload) in payloads.into_iter().enumerate() {
            for candidate in 0..CANDIDATES {
                let offset = bases[profile] + candidate as u64 * payload.len() as u64;
                offsets[profile].push(offset);
                uploads.push((offset, payload.as_slice()));
            }
        }
        gpu.upload_batch(&uploads).unwrap();
        let rows = vec![ROWS as u32; CANDIDATES];
        let query = (0..QUERY_ROWS * DIM)
            .flat_map(|index| (if index % 41 == 0 { 0x3800_u16 } else { 0 }).to_le_bytes())
            .collect::<Vec<_>>();
        let measure = |gpu: &mut Gpu, profile: u8, offsets: &[u64]| {
            for _ in 0..3 {
                gpu.score(
                    &query,
                    QUERY_ROWS as u32,
                    DIM as u32,
                    2,
                    profile,
                    offsets,
                    &rows,
                )
                .unwrap();
            }
            let started = Instant::now();
            for _ in 0..20 {
                gpu.score(
                    &query,
                    QUERY_ROWS as u32,
                    DIM as u32,
                    2,
                    profile,
                    offsets,
                    &rows,
                )
                .unwrap();
            }
            started.elapsed().as_secs_f64() * 1000.0 / 20.0
        };
        let exact_ms = measure(&mut gpu, 1, &offsets[0]);
        let int8_ms = measure(&mut gpu, 2, &offsets[1]);
        let fp8_ms = measure(&mut gpu, 3, &offsets[2]);
        const REQUESTS: usize = 8;
        let batched_queries = (0..REQUESTS)
            .flat_map(|_| query.iter().copied())
            .collect::<Vec<_>>();
        let query_offsets = (0..=REQUESTS)
            .map(|request| (request * QUERY_ROWS) as u32)
            .collect::<Vec<_>>();
        let measure_batch = |gpu: &mut Gpu, profile: u8, offsets: &[u64]| {
            for _ in 0..3 {
                gpu.score_batch(
                    &batched_queries,
                    &query_offsets,
                    DIM as u32,
                    2,
                    profile,
                    offsets,
                    &rows,
                )
                .unwrap();
            }
            let started = Instant::now();
            for _ in 0..10 {
                gpu.score_batch(
                    &batched_queries,
                    &query_offsets,
                    DIM as u32,
                    2,
                    profile,
                    offsets,
                    &rows,
                )
                .unwrap();
            }
            started.elapsed().as_secs_f64() * 1000.0 / 10.0
        };
        let exact_batch_ms = measure_batch(&mut gpu, 1, &offsets[0]);
        let int8_batch_ms = measure_batch(&mut gpu, 2, &offsets[1]);
        let fp8_batch_ms = measure_batch(&mut gpu, 3, &offsets[2]);
        eprintln!(
            "quantized_320d candidates={CANDIDATES} query_rows={QUERY_ROWS} exact_ms={exact_ms:.4} int8_ms={int8_ms:.4} fp8_ms={fp8_ms:.4} int8_vs_exact={:.3} fp8_vs_exact={:.3} requests={REQUESTS} exact_batch_ms={exact_batch_ms:.4} int8_batch_ms={int8_batch_ms:.4} fp8_batch_ms={fp8_batch_ms:.4} int8_batch_vs_exact={:.3} fp8_batch_vs_exact={:.3}",
            exact_ms / int8_ms,
            exact_ms / fp8_ms,
            exact_batch_ms / int8_batch_ms,
            exact_batch_ms / fp8_batch_ms,
        );
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn benchmark_pinned_control_staging_for_320d_candidates() {
        const DIMENSION: u32 = 320;
        const DOCUMENT_ROWS: u32 = 32;
        const QUERY_ROWS: u32 = 32;
        const CANDIDATES: usize = 512;
        const REPETITIONS: usize = 31;
        let device = std::env::var("VCTM_TEST_GPU")
            .unwrap_or_else(|_| "0".to_owned())
            .parse()
            .unwrap();
        let mut gpu = Gpu::create(device, 96 * 1024 * 1024, 48 * 1024 * 1024).unwrap();
        let one = 0x3c00_u16.to_le_bytes();
        let zero = 0_u16.to_le_bytes();
        let mut document = Vec::with_capacity(DOCUMENT_ROWS as usize * DIMENSION as usize * 2);
        for row in 0..DOCUMENT_ROWS {
            for column in 0..DIMENSION {
                document.extend_from_slice(if column == row { &one } else { &zero });
            }
        }
        let document_offsets = (0..CANDIDATES)
            .map(|candidate| candidate as u64 * document.len() as u64)
            .collect::<Vec<_>>();
        let uploads = document_offsets
            .iter()
            .copied()
            .map(|offset| (offset, document.as_slice()))
            .collect::<Vec<_>>();
        gpu.upload_batch(&uploads).unwrap();
        let document_rows = vec![DOCUMENT_ROWS; CANDIDATES];
        let mut query = Vec::with_capacity(QUERY_ROWS as usize * DIMENSION as usize * 2);
        for row in 0..QUERY_ROWS {
            for column in 0..DIMENSION {
                query.extend_from_slice(if column == row { &one } else { &zero });
            }
        }
        let query_offsets = [0, QUERY_ROWS / 2, QUERY_ROWS];
        let measure = |gpu: &mut Gpu, enabled: bool| {
            unsafe {
                assert_eq!(
                    vctm_gpu_set_pinned_control_staging(gpu.native.as_ptr(), i32::from(enabled)),
                    0
                );
            }
            let mut samples = Vec::with_capacity(REPETITIONS);
            let mut scores = Vec::new();
            for _ in 0..REPETITIONS {
                let started = Instant::now();
                scores = gpu
                    .score_batch_native(
                        false,
                        &query,
                        &query_offsets,
                        DIMENSION,
                        2,
                        &document_offsets,
                        &document_rows,
                        1,
                    )
                    .unwrap();
                samples.push(started.elapsed());
            }
            samples.sort_unstable();
            (samples[REPETITIONS / 2], scores)
        };
        let (pageable, pageable_scores) = measure(&mut gpu, false);
        let (pinned, pinned_scores) = measure(&mut gpu, true);
        assert!(batch_scores_close(&pageable_scores, &pinned_scores));
        eprintln!(
            "control_staging_320d candidates={CANDIDATES} query_rows={QUERY_ROWS} pageable_ms={:.4} pinned_ms={:.4} speedup={:.3}",
            pageable.as_secs_f64() * 1000.0,
            pinned.as_secs_f64() * 1000.0,
            pageable.as_secs_f64() / pinned.as_secs_f64(),
        );
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
    fn native_pq_cooperative_adc_matches_long_document_oracle() {
        let mut gpu = pq_gpu();
        gpu.ensure_quantizer("pq", &f32_payload(&[1.0, 0.0, 0.0, 1.0]), 2, 1, 1, 2, 0)
            .unwrap();
        // Eight rows force the cooperative ADC mapping; both identity
        // centroids occur repeatedly, so each query row has a unit maximum.
        gpu.upload_batch(&[(0, &[0_u8, 1, 0, 1, 0, 1, 0, 1])])
            .unwrap();
        let query = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        let scores = gpu.score_pq("pq", &query, 2, 1, &[0], &[8]).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-5, "scores={scores:?}");
    }

    #[test]
    #[ignore = "requires an explicitly assigned CUDA device"]
    fn native_pq_batch_matches_individual_requests() {
        let mut gpu = pq_gpu();
        gpu.ensure_quantizer("pq", &f32_payload(&[1.0, 0.0, 0.0, 1.0]), 2, 1, 1, 2, 0)
            .unwrap();
        gpu.upload_batch(&[(0, &[0_u8, 1])]).unwrap();
        let first = f32_payload(&[1.0, 0.0, 0.0, 1.0]);
        let second = f32_payload(&[1.0, 0.0]);
        let mut queries = first.clone();
        queries.extend_from_slice(&second);
        let batched = gpu
            .score_pq_batch("pq", &queries, &[0, 2, 3], 2, 1, &[0], &[2])
            .unwrap();
        let individual = [
            gpu.score_pq("pq", &first, 2, 1, &[0], &[2]).unwrap(),
            gpu.score_pq("pq", &second, 1, 1, &[0], &[2]).unwrap(),
        ];
        assert_eq!(batched.len(), individual.len());
        for (actual, expected) in batched.iter().zip(individual) {
            assert!(
                (actual[0] - expected[0]).abs() < 1e-5,
                "batched={batched:?}"
            );
        }
    }

    #[test]
    #[ignore = "microbenchmark requires an explicitly assigned CUDA device"]
    fn benchmark_pq_continuous_batch() {
        const DIMENSION: usize = 320;
        const SUBSPACES: usize = 20;
        const CENTROIDS: usize = 256;
        const QUERY_ROWS: usize = 32;
        const REQUESTS: usize = 8;
        const CANDIDATES: usize = 512;
        let document_rows = std::env::var("VCTM_PQ_BENCH_DOCUMENT_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(32);
        assert!(document_rows > 0);
        let mut gpu = pq_gpu();
        let subvector = DIMENSION / SUBSPACES;
        let codebook = (0..SUBSPACES * CENTROIDS * subvector)
            .map(|index| if index % 97 == 0 { 0.125_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        gpu.ensure_quantizer(
            "pq-benchmark",
            &f32_payload(&codebook),
            DIMENSION as u32,
            1,
            SUBSPACES as u16,
            CENTROIDS as u16,
            0,
        )
        .unwrap();
        let document = (0..document_rows * SUBSPACES)
            .map(|index| (index % CENTROIDS) as u8)
            .collect::<Vec<_>>();
        let offsets = (0..CANDIDATES)
            .map(|candidate| candidate as u64 * document.len() as u64)
            .collect::<Vec<_>>();
        let uploads = offsets
            .iter()
            .copied()
            .map(|offset| (offset, document.as_slice()))
            .collect::<Vec<_>>();
        gpu.upload_batch(&uploads).unwrap();
        let rows = vec![document_rows as u32; CANDIDATES];
        let query = (0..QUERY_ROWS * DIMENSION)
            .map(|index| if index % 89 == 0 { 0.5_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        let query = f32_payload(&query);
        let queries = (0..REQUESTS)
            .flat_map(|_| query.iter().copied())
            .collect::<Vec<_>>();
        let query_offsets = (0..=REQUESTS)
            .map(|request| (request * QUERY_ROWS) as u32)
            .collect::<Vec<_>>();
        for _ in 0..3 {
            gpu.score_pq_batch(
                "pq-benchmark",
                &queries,
                &query_offsets,
                DIMENSION as u32,
                1,
                &offsets,
                &rows,
            )
            .unwrap();
        }
        let started = Instant::now();
        let mut individual = Vec::new();
        for _ in 0..10 {
            individual.clear();
            for _ in 0..REQUESTS {
                individual.push(
                    gpu.score_pq(
                        "pq-benchmark",
                        &query,
                        QUERY_ROWS as u32,
                        1,
                        &offsets,
                        &rows,
                    )
                    .unwrap(),
                );
            }
        }
        let individual_ms = started.elapsed().as_secs_f64() * 1000.0 / 10.0;
        let started = Instant::now();
        let mut batched = Vec::new();
        for _ in 0..10 {
            batched = gpu
                .score_pq_batch(
                    "pq-benchmark",
                    &queries,
                    &query_offsets,
                    DIMENSION as u32,
                    1,
                    &offsets,
                    &rows,
                )
                .unwrap();
        }
        let batched_ms = started.elapsed().as_secs_f64() * 1000.0 / 10.0;
        assert_eq!(batched.len(), individual.len());
        for (actual, expected) in batched.iter().zip(&individual) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-4);
            }
        }
        eprintln!(
            "pq_continuous_batch candidates={CANDIDATES} requests={REQUESTS} query_rows={QUERY_ROWS} document_rows={document_rows} individual_ms={individual_ms:.4} batched_ms={batched_ms:.4} speedup={:.3}",
            individual_ms / batched_ms
        );
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
