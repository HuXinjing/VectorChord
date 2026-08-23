// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use crate::cache::{Admission, GpuCache};
use crate::gpu::Gpu;
use crate::protocol::{Descriptor, Request, ScoringProfile};
use crate::quant::QuantizationRegistry;
use crate::shard::{HostCacheStatus, ShardStore, cache_key};
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

struct MissingTensor {
    candidate_index: usize,
    descriptor: Descriptor,
    key: String,
    payload: Arc<[u8]>,
}

struct ResidentTensor {
    candidate_index: usize,
    device: usize,
    key: String,
    offset: u64,
    rows: u32,
    transient: bool,
    newly_admitted: bool,
}

struct DeviceState {
    gpu: Gpu,
    cache: GpuCache,
    h2d_batches: u64,
    h2d_bytes: u64,
}

pub struct Engine {
    devices: Vec<DeviceState>,
    store: ShardStore,
    next_device: usize,
    quantization_registry: Option<QuantizationRegistry>,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceStatus {
    pub slot: usize,
    pub device: i32,
    pub capacity_bytes: usize,
    pub block_bytes: usize,
    pub free_bytes: usize,
    pub largest_free_extent_bytes: usize,
    pub allocated_bytes: usize,
    pub payload_bytes: usize,
    pub internal_waste_bytes: usize,
    pub entries: usize,
    pub pinned_entries: usize,
    pub pinned_bytes: usize,
    pub tenants: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub admission_rejections: u64,
    pub h2d_batches: u64,
    pub h2d_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct EngineStatus {
    pub devices: Vec<DeviceStatus>,
    pub host: HostCacheStatus,
    pub batch_read_calls: u64,
    pub batch_read_bytes: u64,
}

impl Engine {
    pub fn new(
        gpus: Vec<Gpu>,
        block_bytes: usize,
        store: ShardStore,
        tenant_cache_max_percent: u8,
        pinned_cache_max_percent: u8,
        tenant_reservations: &HashMap<String, usize>,
        quantization_registry: Option<QuantizationRegistry>,
    ) -> Result<Self> {
        if gpus.is_empty() {
            bail!("at least one GPU is required");
        }
        let devices = gpus
            .into_iter()
            .map(|gpu| {
                let cache = GpuCache::new_with_limits(
                    gpu.tensor_bytes(),
                    block_bytes,
                    tenant_cache_max_percent,
                    pinned_cache_max_percent,
                    tenant_reservations.clone(),
                )
                .map_err(|message| anyhow!(message))?;
                Ok(DeviceState {
                    gpu,
                    cache,
                    h2d_batches: 0,
                    h2d_bytes: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            devices,
            store,
            next_device: 0,
            quantization_registry,
        })
    }

    pub fn reload_shards(&mut self) -> Result<()> {
        self.store.reload()
    }

    pub fn prewarm(&mut self, descriptors: &[Descriptor], batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            bail!("resident prewarm batch size must be positive");
        }
        // A content-addressed manifest may legitimately reference the same tensor
        // from more than one logical candidate. Upload each cache key once. Without
        // this normalization, two duplicates in one batch both try to admit the
        // same not-yet-ready entry and the second insert replaces the first arena
        // allocation.
        let descriptors = unique_descriptors(descriptors);
        for batch in descriptors.chunks(batch_size) {
            let payloads = self.store.resolve_many(batch, "__resident__")?;
            let mut uploads = (0..self.devices.len())
                .map(|_| Vec::<(u64, &[u8])>::new())
                .collect::<Vec<_>>();
            let mut acquired = Vec::<(usize, String, bool)>::new();
            for (descriptor, payload) in batch.iter().zip(&payloads) {
                let key = gpu_cache_key(descriptor, ScoringProfile::ExactFp16, None);
                if let Some((device, _)) =
                    self.devices
                        .iter_mut()
                        .enumerate()
                        .find_map(|(index, device)| {
                            device
                                .cache
                                .acquire_existing(&key)
                                .map(|entry| (index, entry))
                        })
                {
                    acquired.push((device, key, false));
                    continue;
                }
                self.devices[self.next_device]
                    .cache
                    .record_access_miss(&key);
                let mut admission = None;
                for step in 0..self.devices.len() {
                    let device = (self.next_device + step) % self.devices.len();
                    if let Admission::Admitted { offset, .. } =
                        self.devices[device].cache.admit_for_tenant(
                            "__resident__",
                            key.clone(),
                            payload.len(),
                            descriptor.rows,
                            descriptor.dimension,
                            descriptor.dtype,
                            true,
                            true,
                        )
                    {
                        admission = Some((device, offset));
                        self.next_device = (device + 1) % self.devices.len();
                        break;
                    }
                }
                let Some((device, offset)) = admission else {
                    bail!("resident manifest exceeds the configured Rust GPU block caches");
                };
                uploads[device].push((offset, payload.as_ref()));
                acquired.push((device, key, true));
            }
            let mut upload_succeeded = vec![true; self.devices.len()];
            let mut upload_error = None;
            for (device, items) in uploads.iter().enumerate() {
                if items.is_empty() {
                    continue;
                }
                match self.devices[device].gpu.upload_batch(items) {
                    Ok(()) => {
                        self.devices[device].h2d_batches += 1;
                        self.devices[device].h2d_bytes += items
                            .iter()
                            .map(|(_, payload)| payload.len() as u64)
                            .sum::<u64>();
                    }
                    Err(error) => {
                        upload_succeeded[device] = false;
                        upload_error.get_or_insert(error);
                    }
                }
            }
            for (device, key, newly_admitted) in acquired {
                if newly_admitted && upload_succeeded[device] {
                    self.devices[device]
                        .cache
                        .mark_ready(&key)
                        .map_err(|message| anyhow!(message))?;
                }
                if newly_admitted && !upload_succeeded[device] {
                    self.devices[device]
                        .cache
                        .remove(&key)
                        .map_err(|message| anyhow!(message))?;
                } else {
                    self.devices[device]
                        .cache
                        .release(&key)
                        .map_err(|message| anyhow!(message))?;
                }
            }
            if let Some(error) = upload_error {
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn score(&mut self, request: &Request) -> Result<Vec<(u32, f32)>> {
        if matches!(request.scoring_profile, ScoringProfile::Pq | ScoringProfile::OpqRpq) {
            let contract_id = request.quantization_contract.as_deref().ok_or_else(|| anyhow!("PQ-family request has no quantization contract"))?;
            let model_contract = request.candidates.first().ok_or_else(|| anyhow!("PQ-family request has no candidates"))?.contract.as_str();
            self.quantization_registry.as_ref().ok_or_else(|| anyhow!("PQ-family scoring requires --quantization-registry-root"))?
                .resolve_active(contract_id, model_contract, request.scoring_profile)?;
        }
        if !matches!(
            request.scoring_profile,
            ScoringProfile::ExactFp16 | ScoringProfile::Int8 | ScoringProfile::Fp8E4m3
        ) {
            anyhow::bail!(
                "requested TileMaxSim scoring profile {:?} is not enabled by the native daemon",
                request.scoring_profile
            );
        }
        if request.candidates.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = vec![None; request.candidates.len()];
        let mut hit_chunks = (0..self.devices.len())
            .map(|_| Vec::<ResidentTensor>::new())
            .collect::<Vec<_>>();
        let mut missing_descriptors = Vec::new();
        let mut missing_indices = Vec::new();
        let mut first_candidate_by_key = HashMap::<String, usize>::new();
        let mut duplicate_candidates = Vec::<(usize, usize)>::new();
        for (index, descriptor) in request.candidates.iter().enumerate() {
            let key = gpu_cache_key(
                descriptor,
                request.scoring_profile,
                request.quantization_contract.as_deref(),
            );
            if let Some(first_index) = first_candidate_by_key.get(&key) {
                duplicate_candidates.push((index, *first_index));
                continue;
            }
            first_candidate_by_key.insert(key.clone(), index);
            let hit_device = self
                .devices
                .iter()
                .position(|device| device.cache.contains(&key));
            if let Some(device_index) = hit_device {
                let entry = self.devices[device_index]
                    .cache
                    .get(&key)
                    .expect("cache hit disappeared");
                validate_entry(descriptor, &entry, request.scoring_profile)?;
                hit_chunks[device_index].push(ResidentTensor {
                    candidate_index: index,
                    device: device_index,
                    key,
                    offset: entry.offset,
                    rows: entry.rows,
                    transient: false,
                    newly_admitted: false,
                });
            } else {
                // Record one request-level miss on the device that will get the
                // first admission opportunity. Other devices are not polluted.
                self.devices[self.next_device]
                    .cache
                    .record_access_miss(&key);
                missing_indices.push(index);
                missing_descriptors.push(descriptor.clone());
            }
        }

        let hit_result = self.score_devices(request, &hit_chunks, &mut scores);
        let hit_cleanup = self.release_chunks(&hit_chunks);
        hit_result?;
        hit_cleanup?;

        let payloads = self
            .store
            .resolve_many(&missing_descriptors, &request.tenant)?;
        let mut pending = missing_indices
            .into_iter()
            .zip(missing_descriptors)
            .zip(payloads)
            .map(|((candidate_index, descriptor), payload)| {
                let payload = encode_for_profile(&descriptor, payload, request.scoring_profile)?;
                Ok(MissingTensor {
                    candidate_index,
                    key: gpu_cache_key(
                        &descriptor,
                        request.scoring_profile,
                        request.quantization_contract.as_deref(),
                    ),
                    descriptor,
                    payload,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        while !pending.is_empty() {
            let mut chunks = (0..self.devices.len())
                .map(|_| Vec::<ResidentTensor>::new())
                .collect::<Vec<_>>();
            let mut uploads = (0..self.devices.len())
                .map(|_| Vec::<(u64, &[u8])>::new())
                .collect::<Vec<_>>();
            let mut consumed = 0;
            for tensor in &pending {
                if let Some((device_index, entry)) =
                    self.devices
                        .iter_mut()
                        .enumerate()
                        .find_map(|(index, device)| {
                            device
                                .cache
                                .acquire_existing(&tensor.key)
                                .map(|entry| (index, entry))
                        })
                {
                    chunks[device_index].push(ResidentTensor {
                        candidate_index: tensor.candidate_index,
                        device: device_index,
                        key: tensor.key.clone(),
                        offset: entry.offset,
                        rows: entry.rows,
                        transient: false,
                        newly_admitted: false,
                    });
                    consumed += 1;
                    continue;
                }

                let mut admitted = None;
                for step in 0..self.devices.len() {
                    let device_index = (self.next_device + step) % self.devices.len();
                    let admission = self.devices[device_index].cache.admit_for_tenant(
                        &request.tenant,
                        tensor.key.clone(),
                        tensor.payload.len(),
                        tensor.descriptor.rows,
                        tensor.descriptor.dimension,
                        tensor.descriptor.dtype,
                        false,
                        false,
                    );
                    if let Admission::Admitted { offset, .. } = admission {
                        admitted = Some((device_index, offset, false));
                        self.next_device = (device_index + 1) % self.devices.len();
                        break;
                    }
                }

                if admitted.is_none() && chunks.iter().all(Vec::is_empty) {
                    // TinyLFU rejected the cold item on every device, but the
                    // request must still be computed. Use one transient slab
                    // and remove it after the chunk completes.
                    for step in 0..self.devices.len() {
                        let device_index = (self.next_device + step) % self.devices.len();
                        if let Admission::Admitted { offset, .. } =
                            self.devices[device_index].cache.admit_for_tenant(
                                &request.tenant,
                                tensor.key.clone(),
                                tensor.payload.len(),
                                tensor.descriptor.rows,
                                tensor.descriptor.dimension,
                                tensor.descriptor.dtype,
                                false,
                                true,
                            )
                        {
                            admitted = Some((device_index, offset, true));
                            self.next_device = (device_index + 1) % self.devices.len();
                            break;
                        }
                    }
                }

                let Some((device_index, offset, transient)) = admitted else {
                    if chunks.iter().any(|chunk| !chunk.is_empty()) {
                        break;
                    }
                    bail!("one tensor cannot be scheduled in any configured Rust GPU block cache");
                };
                uploads[device_index].push((offset, tensor.payload.as_ref()));
                chunks[device_index].push(ResidentTensor {
                    candidate_index: tensor.candidate_index,
                    device: device_index,
                    key: tensor.key.clone(),
                    offset,
                    rows: tensor.descriptor.rows,
                    transient,
                    newly_admitted: true,
                });
                consumed += 1;
            }

            if consumed == 0 {
                bail!("Rust multi-GPU scheduler made no progress");
            }
            let upload_succeeded = match self.upload_devices(&chunks, &uploads) {
                Ok(upload_succeeded) => upload_succeeded,
                Err((error, upload_succeeded)) => {
                    self.cleanup_after_upload_failure(&chunks, &upload_succeeded)?;
                    return Err(error);
                }
            };
            debug_assert!(upload_succeeded.iter().all(|succeeded| *succeeded));
            let score_result = self.score_devices(request, &chunks, &mut scores);
            let cleanup_result = self.release_chunks(&chunks);
            score_result?;
            cleanup_result?;
            pending.drain(..consumed);
        }

        // Equal content-addressed tensors have equal TileMaxSim scores. Preserve
        // every logical candidate id while avoiding duplicate cache acquisitions,
        // uploads, and kernel work inside one request.
        for (duplicate_index, first_index) in duplicate_candidates {
            scores[duplicate_index] = scores[first_index];
        }

        request
            .candidates
            .iter()
            .enumerate()
            .map(|(index, descriptor)| {
                scores[index]
                    .map(|score| (descriptor.candidate_id, score))
                    .ok_or_else(|| anyhow!("missing native TileMaxSim result"))
            })
            .collect()
    }

    fn upload_devices(
        &mut self,
        chunks: &[Vec<ResidentTensor>],
        uploads: &[Vec<(u64, &[u8])>],
    ) -> Result<Vec<bool>, (anyhow::Error, Vec<bool>)> {
        let results = std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (device_index, ((device, chunk), upload)) in
                self.devices.iter_mut().zip(chunks).zip(uploads).enumerate()
            {
                if upload.is_empty() {
                    continue;
                }
                workers.push((
                    device_index,
                    scope.spawn(move || -> Result<()> {
                        device.gpu.upload_batch(upload)?;
                        device.h2d_batches += 1;
                        device.h2d_bytes += upload
                            .iter()
                            .map(|(_, payload)| payload.len() as u64)
                            .sum::<u64>();
                        // Keep the chunk borrow in this worker so upload metadata and
                        // payload lifetimes remain tied to the scoped thread.
                        let _ = chunk;
                        Ok(())
                    }),
                ));
            }
            workers
                .into_iter()
                .map(|(device, worker)| {
                    let result = worker
                        .join()
                        .map_err(|_| anyhow!("GPU upload worker panicked"))
                        .and_then(|result| result);
                    (device, result)
                })
                .collect::<Vec<_>>()
        });

        let mut succeeded = vec![true; self.devices.len()];
        let mut first_error = None;
        for (device, result) in results {
            if let Err(error) = result {
                succeeded[device] = false;
                first_error.get_or_insert(error);
            }
        }
        for (device, chunk) in chunks.iter().enumerate() {
            if !succeeded[device] {
                continue;
            }
            for tensor in chunk.iter().filter(|tensor| tensor.newly_admitted) {
                if let Err(message) = self.devices[device].cache.mark_ready(&tensor.key) {
                    succeeded[device] = false;
                    first_error.get_or_insert_with(|| anyhow!(message));
                    break;
                }
            }
        }
        if let Some(error) = first_error {
            Err((error, succeeded))
        } else {
            Ok(succeeded)
        }
    }

    fn score_devices(
        &mut self,
        request: &Request,
        chunks: &[Vec<ResidentTensor>],
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let completed = std::thread::scope(|scope| -> Result<Vec<Vec<(usize, f32)>>> {
            let mut workers = Vec::new();
            for (device, chunk) in self.devices.iter_mut().zip(chunks) {
                if chunk.is_empty() {
                    continue;
                }
                workers.push(scope.spawn(move || -> Result<Vec<(usize, f32)>> {
                    let offsets = chunk.iter().map(|item| item.offset).collect::<Vec<_>>();
                    let rows = chunk.iter().map(|item| item.rows).collect::<Vec<_>>();
                    let computed = device.gpu.score(
                        &request.query,
                        request.query_rows,
                        request.dimension,
                        request.dtype,
                        request.scoring_profile.native_code(),
                        &offsets,
                        &rows,
                    )?;
                    Ok(chunk
                        .iter()
                        .zip(computed)
                        .map(|(tensor, score)| (tensor.candidate_index, score))
                        .collect())
                }));
            }
            let mut completed = Vec::new();
            for worker in workers {
                completed.push(
                    worker
                        .join()
                        .map_err(|_| anyhow!("GPU worker panicked"))??,
                );
            }
            Ok(completed)
        })?;
        for device_scores in completed {
            for (candidate_index, score) in device_scores {
                scores[candidate_index] = Some(score);
            }
        }
        Ok(())
    }

    fn release_chunks(&mut self, chunks: &[Vec<ResidentTensor>]) -> Result<()> {
        for chunk in chunks {
            for tensor in chunk {
                if tensor.transient {
                    self.devices[tensor.device]
                        .cache
                        .remove(&tensor.key)
                        .map_err(|message| anyhow!(message))?;
                } else {
                    self.devices[tensor.device]
                        .cache
                        .release(&tensor.key)
                        .map_err(|message| anyhow!(message))?;
                }
            }
        }
        Ok(())
    }

    fn cleanup_after_upload_failure(
        &mut self,
        chunks: &[Vec<ResidentTensor>],
        upload_succeeded: &[bool],
    ) -> Result<()> {
        for chunk in chunks {
            for tensor in chunk {
                if tensor.newly_admitted && (!upload_succeeded[tensor.device] || tensor.transient) {
                    self.devices[tensor.device]
                        .cache
                        .remove(&tensor.key)
                        .map_err(|message| anyhow!(message))?;
                } else {
                    self.devices[tensor.device]
                        .cache
                        .release(&tensor.key)
                        .map_err(|message| anyhow!(message))?;
                }
            }
        }
        Ok(())
    }

    pub fn status_snapshot(&self) -> EngineStatus {
        let devices = self
            .devices
            .iter()
            .enumerate()
            .map(|(slot, device)| DeviceStatus {
                slot,
                device: device.gpu.device(),
                capacity_bytes: device.cache.capacity(),
                block_bytes: device.cache.block_bytes(),
                free_bytes: device.cache.free_bytes(),
                largest_free_extent_bytes: device.cache.largest_free_extent(),
                allocated_bytes: device.cache.allocated_bytes(),
                payload_bytes: device.cache.payload_bytes(),
                internal_waste_bytes: device
                    .cache
                    .allocated_bytes()
                    .saturating_sub(device.cache.payload_bytes()),
                entries: device.cache.entry_count(),
                pinned_entries: device.cache.pinned_entries(),
                pinned_bytes: device.cache.pinned_bytes(),
                tenants: device.cache.tenant_count(),
                hits: device.cache.hits,
                misses: device.cache.misses,
                evictions: device.cache.evictions,
                admission_rejections: device.cache.admission_rejections,
                h2d_batches: device.h2d_batches,
                h2d_bytes: device.h2d_bytes,
            })
            .collect();
        EngineStatus {
            devices,
            host: self.store.host_status(),
            batch_read_calls: self.store.batch_read_calls,
            batch_read_bytes: self.store.batch_read_bytes,
        }
    }

    pub fn status_json(&self) -> serde_json::Value {
        let status = self.status_snapshot();
        let devices = status
            .devices
            .iter()
            .map(|device| {
                serde_json::json!({
                    "index": device.slot,
                    "device": device.device,
                    "gpu_allocator": "segregated-page-runs",
                    "gpu_tensor_bytes": device.capacity_bytes,
                    "gpu_block_bytes": device.block_bytes,
                    "gpu_free_bytes": device.free_bytes,
                    "gpu_largest_free_extent_bytes": device.largest_free_extent_bytes,
                    "gpu_allocated_bytes": device.allocated_bytes,
                    "gpu_payload_bytes": device.payload_bytes,
                    "gpu_internal_waste_bytes": device.internal_waste_bytes,
                    "gpu_entries": device.entries,
                    "gpu_pinned_entries": device.pinned_entries,
                    "gpu_pinned_bytes": device.pinned_bytes,
                    "gpu_tenant_count": device.tenants,
                    "gpu_hits": device.hits,
                    "gpu_misses": device.misses,
                    "gpu_evictions": device.evictions,
                    "gpu_admission_rejections": device.admission_rejections,
                    "h2d_batches": device.h2d_batches,
                    "h2d_bytes": device.h2d_bytes,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "devices": devices,
            "host_capacity_bytes": status.host.capacity_bytes,
            "host_used_bytes": status.host.used_bytes,
            "host_entries": status.host.entries,
            "host_tenant_count": status.host.tenants,
            "host_hits": status.host.hits,
            "host_misses": status.host.misses,
            "host_evictions": status.host.evictions,
            "host_admission_rejections": status.host.admission_rejections,
            "batch_read_calls": status.batch_read_calls,
            "batch_read_bytes": status.batch_read_bytes,
        })
    }
}

fn unique_descriptors(descriptors: &[Descriptor]) -> Vec<Descriptor> {
    let mut seen = HashSet::with_capacity(descriptors.len());
    descriptors
        .iter()
        .filter(|descriptor| seen.insert(cache_key(descriptor)))
        .cloned()
        .collect()
}

fn gpu_cache_key(
    descriptor: &Descriptor,
    profile: ScoringProfile,
    quantization_contract: Option<&str>,
) -> String {
    format!(
        "{}:{}:{}",
        profile.cache_tag(),
        quantization_contract.unwrap_or("-"),
        cache_key(descriptor)
    )
}

fn validate_entry(
    descriptor: &Descriptor,
    entry: &crate::cache::CacheEntry,
    profile: ScoringProfile,
) -> Result<()> {
    let scalar_bytes = if descriptor.dtype == 1 { 4 } else { 2 };
    let exact_bytes = descriptor.rows as usize * descriptor.dimension as usize * scalar_bytes;
    let expected_bytes = match profile {
        ScoringProfile::ExactFp16 => exact_bytes,
        ScoringProfile::Int8 | ScoringProfile::Fp8E4m3 => {
            scaled_payload_bytes(descriptor.rows, descriptor.dimension)?
        }
        _ => bail!("unsupported GPU cache scoring profile"),
    };
    if entry.rows != descriptor.rows
        || entry.dimension != descriptor.dimension
        || entry.dtype != descriptor.dtype
        || entry.payload_bytes != expected_bytes
        || entry.allocated_bytes < expected_bytes
    {
        bail!("GPU cache metadata disagrees with the tensor descriptor");
    }
    Ok(())
}

fn encode_for_profile(
    descriptor: &Descriptor,
    payload: Arc<[u8]>,
    profile: ScoringProfile,
) -> Result<Arc<[u8]>> {
    if profile == ScoringProfile::ExactFp16 {
        return Ok(payload);
    }
    if !matches!(profile, ScoringProfile::Int8 | ScoringProfile::Fp8E4m3) {
        bail!("unsupported TileMaxSim encoding profile");
    }
    let rows = descriptor.rows as usize;
    let dimension = descriptor.dimension as usize;
    let scalar_bytes = if descriptor.dtype == 1 { 4 } else { 2 };
    let expected = rows
        .checked_mul(dimension)
        .and_then(|count| count.checked_mul(scalar_bytes))
        .ok_or_else(|| anyhow!("tensor encoding size overflow"))?;
    if payload.len() != expected {
        bail!("tensor payload length disagrees with its descriptor");
    }
    let code_bytes = rows
        .checked_mul(dimension)
        .ok_or_else(|| anyhow!("INT8 tensor size overflow"))?;
    let scale_offset = align_up(code_bytes, size_of::<f32>())?;
    let mut encoded = vec![0_u8; scaled_payload_bytes(descriptor.rows, descriptor.dimension)?];
    for row in 0..rows {
        let mut maximum = 0.0_f32;
        for column in 0..dimension {
            maximum = maximum.max(read_scalar(&payload, row * dimension + column, descriptor.dtype)?.abs());
        }
        let bound = if profile == ScoringProfile::Int8 { 127.0 } else { 448.0 };
        let scale = if maximum == 0.0 { 1.0 } else { maximum / bound };
        for column in 0..dimension {
            let value = read_scalar(&payload, row * dimension + column, descriptor.dtype)?;
            encoded[row * dimension + column] = if profile == ScoringProfile::Int8 {
                (value / scale).round().clamp(-127.0, 127.0) as i8 as u8
            } else {
                f32_to_e4m3fn(value / scale)
            };
        }
        let offset = scale_offset + row * size_of::<f32>();
        encoded[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
    }
    Ok(Arc::from(encoded))
}

fn scaled_payload_bytes(rows: u32, dimension: u32) -> Result<usize> {
    let codes = (rows as usize)
        .checked_mul(dimension as usize)
        .ok_or_else(|| anyhow!("INT8 tensor size overflow"))?;
    align_up(codes, size_of::<f32>())?
        .checked_add(rows as usize * size_of::<f32>())
        .ok_or_else(|| anyhow!("INT8 tensor size overflow"))
}

fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let fraction = bits & 0x07;
    if exponent == 0 {
        sign * fraction as f32 * 2.0_f32.powi(-9)
    } else if exponent == 0x0f && fraction == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + fraction as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    }
}

fn f32_to_e4m3fn(value: f32) -> u8 {
    static POSITIVE: OnceLock<Vec<(f32, u8)>> = OnceLock::new();
    let table = POSITIVE.get_or_init(|| {
        let mut values = (0_u16..=0x7e)
            .map(|bits| (e4m3fn_to_f32(bits as u8), bits as u8))
            .collect::<Vec<_>>();
        values.sort_by(|left, right| left.0.total_cmp(&right.0));
        values
    });
    let negative = value.is_sign_negative();
    let magnitude = value.abs().min(448.0);
    let index = table.partition_point(|(candidate, _)| *candidate < magnitude);
    let selected = match index {
        0 => table[0],
        value if value == table.len() => table[table.len() - 1],
        value => {
            let lower = table[value - 1];
            let upper = table[value];
            if magnitude - lower.0 <= upper.0 - magnitude { lower } else { upper }
        }
    };
    selected.1 | if negative { 0x80 } else { 0 }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| anyhow!("tensor alignment overflow"))
}

fn read_scalar(payload: &[u8], index: usize, dtype: u8) -> Result<f32> {
    match dtype {
        1 => {
            let offset = index * 4;
            Ok(f32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()))
        }
        2 => {
            let offset = index * 2;
            Ok(half_to_f32(u16::from_le_bytes(
                payload[offset..offset + 2].try_into().unwrap(),
            )))
        }
        _ => bail!("unsupported source tensor dtype"),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            sign | ((113 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(candidate_id: u32, digest: &str, rows: u32) -> Descriptor {
        Descriptor {
            candidate_id,
            contract: "colqwen35@test".to_owned(),
            digest: digest.to_owned(),
            rows,
            dimension: 320,
            dtype: 2,
        }
    }

    #[test]
    fn resident_prewarm_deduplicates_content_references() {
        let descriptors = vec![
            descriptor(1, "a", 100),
            descriptor(2, "a", 100),
            descriptor(3, "b", 120),
            // Shape is part of the cache identity and must not be collapsed.
            descriptor(4, "a", 101),
        ];

        let unique = unique_descriptors(&descriptors);
        assert_eq!(unique.len(), 3);
        assert_eq!(unique[0].candidate_id, 1);
        assert_eq!(unique[1].candidate_id, 3);
        assert_eq!(unique[2].candidate_id, 4);
    }

    #[test]
    fn fp16_decoder_covers_normal_and_subnormal_values() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2.0_f32.powi(-24));
    }

    #[test]
    fn int8_encoding_is_row_scaled_and_profile_namespaced() {
        let descriptor = descriptor(1, "a", 1);
        let mut payload = Vec::with_capacity(640);
        for column in 0..320 {
            let bits = if column == 0 { 0x3c00_u16 } else { 0_u16 };
            payload.extend_from_slice(&bits.to_le_bytes());
        }
        let encoded = encode_for_profile(
            &descriptor,
            Arc::from(payload),
            ScoringProfile::Int8,
        )
        .unwrap();
        assert_eq!(encoded.len(), 320 + 4);
        assert_eq!(encoded[0] as i8, 127);
        assert_eq!(
            f32::from_le_bytes(encoded[320..324].try_into().unwrap()),
            1.0 / 127.0
        );
        assert_ne!(
            gpu_cache_key(&descriptor, ScoringProfile::ExactFp16, None),
            gpu_cache_key(&descriptor, ScoringProfile::Int8, None)
        );
    }

    #[test]
    fn fp8_encoding_uses_e4m3fn_and_a_distinct_cache_namespace() {
        for value in [0.0, 0.5, 1.0, 12.0, 448.0, -1.0] {
            let decoded = e4m3fn_to_f32(f32_to_e4m3fn(value));
            assert!((decoded - value).abs() <= value.abs().max(1.0) / 8.0);
        }
        let descriptor = descriptor(1, "a", 1);
        assert_ne!(
            gpu_cache_key(&descriptor, ScoringProfile::Fp8E4m3, None),
            gpu_cache_key(&descriptor, ScoringProfile::Int8, None)
        );
    }
}
