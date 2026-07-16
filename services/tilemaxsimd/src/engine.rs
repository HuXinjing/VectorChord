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
use crate::protocol::{Descriptor, Request};
use crate::shard::{HostCacheStatus, ShardStore, cache_key};
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct IoPipelineConfig {
    pub overlap: bool,
    pub batch_bytes: usize,
}

impl IoPipelineConfig {
    pub fn validate(self) -> Result<Self> {
        if self.batch_bytes == 0 {
            bail!("I/O pipeline batch size must be positive");
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct IoPipelineStatus {
    pub overlap_enabled: bool,
    pub batch_bytes: usize,
    pub resolve_batches: u64,
    pub resolve_bytes: u64,
    pub resolve_microseconds: u64,
    pub upload_microseconds: u64,
    pub compute_microseconds: u64,
    pub overlap_cycles: u64,
    pub overlap_microseconds: u64,
}

struct MissingTensor {
    candidate_index: usize,
    descriptor: Descriptor,
    key: String,
    payload: Arc<[u8]>,
}

struct UnresolvedBatch {
    indices: Vec<usize>,
    descriptors: Vec<Descriptor>,
    payload_bytes: usize,
}

struct ResolvedBatch {
    tensors: VecDeque<MissingTensor>,
    payload_bytes: usize,
    elapsed: Duration,
}

struct PreparedBatch {
    chunks: Vec<Vec<ResidentTensor>>,
    uploads: Vec<Vec<(u64, Arc<[u8]>)>>,
}

impl PreparedBatch {
    fn empty(device_count: usize) -> Self {
        Self {
            chunks: (0..device_count).map(|_| Vec::new()).collect(),
            uploads: (0..device_count).map(|_| Vec::new()).collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.chunks.iter().all(Vec::is_empty)
    }
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
    gpu: Arc<Gpu>,
    cache: GpuCache,
    h2d_batches: u64,
    h2d_bytes: u64,
}

pub struct Engine {
    devices: Vec<DeviceState>,
    store: Arc<Mutex<ShardStore>>,
    next_device: usize,
    io_pipeline: IoPipelineConfig,
    io_status: IoPipelineStatus,
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
    pub io_pipeline: IoPipelineStatus,
}

impl Engine {
    pub fn new(
        gpus: Vec<Gpu>,
        block_bytes: usize,
        store: ShardStore,
        tenant_cache_max_percent: u8,
        pinned_cache_max_percent: u8,
        tenant_reservations: &HashMap<String, usize>,
        io_pipeline: IoPipelineConfig,
    ) -> Result<Self> {
        if gpus.is_empty() {
            bail!("at least one GPU is required");
        }
        let io_pipeline = io_pipeline.validate()?;
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
                    gpu: Arc::new(gpu),
                    cache,
                    h2d_batches: 0,
                    h2d_bytes: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            devices,
            store: Arc::new(Mutex::new(store)),
            next_device: 0,
            io_pipeline,
            io_status: IoPipelineStatus {
                overlap_enabled: io_pipeline.overlap,
                batch_bytes: io_pipeline.batch_bytes,
                ..IoPipelineStatus::default()
            },
        })
    }

    pub fn reload_shards(&mut self) -> Result<()> {
        self.store
            .lock()
            .map_err(|_| anyhow!("immutable shard store lock is poisoned"))?
            .reload()
    }

    /// Return the H2D payload expected for a descriptor at the instant the
    /// scheduler builds its next cooperative quantum. Host-cache residency does
    /// not make this zero: every L0 miss still consumes upload bandwidth.
    pub fn gpu_miss_bytes(&self, descriptor: &Descriptor) -> u64 {
        let key = cache_key(descriptor);
        if self
            .devices
            .iter()
            .any(|device| device.cache.contains(&key))
        {
            return 0;
        }
        descriptor_payload_bytes(descriptor)
            .ok()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX)
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
            let payloads = self
                .store
                .lock()
                .map_err(|_| anyhow!("immutable shard store lock is poisoned"))?
                .resolve_many(batch, "__resident__")?;
            let mut uploads = (0..self.devices.len())
                .map(|_| Vec::<(u64, &[u8])>::new())
                .collect::<Vec<_>>();
            let mut acquired = Vec::<(usize, String, bool)>::new();
            for (descriptor, payload) in batch.iter().zip(&payloads) {
                let key = cache_key(descriptor);
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
            let key = cache_key(descriptor);
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
                validate_entry(descriptor, &entry)?;
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

        let unresolved = split_unresolved_batches(
            missing_indices,
            missing_descriptors,
            self.io_pipeline.batch_bytes,
        )?;
        let hits = PreparedBatch {
            chunks: hit_chunks,
            uploads: (0..self.devices.len()).map(|_| Vec::new()).collect(),
        };
        if self.io_pipeline.overlap {
            let stage_before = self.pipeline_stage_microseconds();
            let pipeline_started = Instant::now();
            self.score_overlapped(request, hits, unresolved, &mut scores)?;
            let stage_delta = self
                .pipeline_stage_microseconds()
                .saturating_sub(stage_before);
            let hidden =
                stage_delta.saturating_sub(duration_microseconds(pipeline_started.elapsed()));
            self.io_status.overlap_microseconds =
                self.io_status.overlap_microseconds.saturating_add(hidden);
        } else {
            self.score_serial(request, hits, unresolved, &mut scores)?;
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

    fn score_serial(
        &mut self,
        request: &Request,
        mut current: PreparedBatch,
        mut unresolved: VecDeque<UnresolvedBatch>,
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let mut pending = VecDeque::new();
        loop {
            if !current.is_empty() {
                let result = score_chunks(&self.gpus(), request, &current.chunks);
                let cleanup = self.release_chunks(&current.chunks);
                let (computed, elapsed) = result?;
                self.io_status.compute_microseconds = self
                    .io_status
                    .compute_microseconds
                    .saturating_add(duration_microseconds(elapsed));
                apply_scores(scores, computed);
                cleanup?;
            }
            if pending.is_empty() && unresolved.is_empty() {
                return Ok(());
            }
            current =
                self.fill_next_batch(&request.tenant, &mut unresolved, &mut pending, false)?;
        }
    }

    fn score_overlapped(
        &mut self,
        request: &Request,
        current: PreparedBatch,
        unresolved: VecDeque<UnresolvedBatch>,
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let (sender, receiver) = mpsc::sync_channel::<Result<ResolvedBatch>>(1);
        let store = Arc::clone(&self.store);
        let tenant = request.tenant.clone();
        std::thread::scope(|scope| {
            let resolver = scope.spawn(move || {
                for batch in unresolved {
                    let result = resolve_batch(&store, batch, &tenant);
                    let failed = result.is_err();
                    if sender.send(result).is_err() || failed {
                        break;
                    }
                }
            });
            let result = self.score_overlapped_inner(request, current, &receiver, scores);
            // An early CUDA or cache error must unblock a resolver waiting on the
            // bounded one-batch channel before the scoped worker can join.
            drop(receiver);
            let resolver_result = resolver
                .join()
                .map_err(|_| anyhow!("tensor resolver coordinator panicked"));
            result.and(resolver_result)
        })
    }

    fn score_overlapped_inner(
        &mut self,
        request: &Request,
        mut current: PreparedBatch,
        receiver: &mpsc::Receiver<Result<ResolvedBatch>>,
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let mut pending = VecDeque::new();
        let mut resolver_done = false;
        loop {
            if current.is_empty() {
                current = self.fill_next_resolved_batch(
                    &request.tenant,
                    receiver,
                    &mut pending,
                    &mut resolver_done,
                    false,
                )?;
                if current.is_empty() && resolver_done {
                    return Ok(());
                }
                continue;
            }

            if pending.is_empty() && resolver_done {
                let result = score_chunks(&self.gpus(), request, &current.chunks);
                let cleanup = self.release_chunks(&current.chunks);
                let (computed, elapsed) = result?;
                self.io_status.compute_microseconds = self
                    .io_status
                    .compute_microseconds
                    .saturating_add(duration_microseconds(elapsed));
                apply_scores(scores, computed);
                cleanup?;
                return Ok(());
            }

            let gpus = self.gpus();
            let (score_result, next_result) = std::thread::scope(|scope| {
                let score_worker = scope.spawn(|| score_chunks(&gpus, request, &current.chunks));
                let next = self.fill_next_resolved_batch(
                    &request.tenant,
                    receiver,
                    &mut pending,
                    &mut resolver_done,
                    true,
                );
                let score = score_worker
                    .join()
                    .map_err(|_| anyhow!("GPU score coordinator panicked"))
                    .and_then(|result| result);
                (score, next)
            });
            let cleanup_current = self.release_chunks(&current.chunks);

            let (computed, compute_elapsed) = match score_result {
                Ok(result) => result,
                Err(error) => {
                    if let Ok(next) = next_result {
                        self.release_chunks(&next.chunks)?;
                    }
                    cleanup_current?;
                    return Err(error);
                }
            };
            self.io_status.compute_microseconds = self
                .io_status
                .compute_microseconds
                .saturating_add(duration_microseconds(compute_elapsed));
            apply_scores(scores, computed);
            cleanup_current?;

            let next = next_result?;
            self.io_status.overlap_cycles = self.io_status.overlap_cycles.saturating_add(1);
            current = next;
        }
    }

    fn fill_next_resolved_batch(
        &mut self,
        tenant: &str,
        receiver: &mpsc::Receiver<Result<ResolvedBatch>>,
        pending: &mut VecDeque<MissingTensor>,
        resolver_done: &mut bool,
        allow_deferred: bool,
    ) -> Result<PreparedBatch> {
        if pending.is_empty() && !*resolver_done {
            match receiver.recv() {
                Ok(result) => {
                    let resolved = result?;
                    self.record_resolution(&resolved);
                    *pending = resolved.tensors;
                }
                Err(_) => *resolver_done = true,
            }
        }
        if pending.is_empty() {
            return Ok(PreparedBatch::empty(self.devices.len()));
        }
        let mut prepared = self.prepare_pending(pending, tenant, allow_deferred)?;
        if prepared.is_empty() {
            return Ok(prepared);
        }
        let upload_elapsed = self.upload_prepared(&mut prepared)?;
        self.io_status.upload_microseconds = self
            .io_status
            .upload_microseconds
            .saturating_add(duration_microseconds(upload_elapsed));
        Ok(prepared)
    }

    fn fill_next_batch(
        &mut self,
        tenant: &str,
        unresolved: &mut VecDeque<UnresolvedBatch>,
        pending: &mut VecDeque<MissingTensor>,
        allow_deferred: bool,
    ) -> Result<PreparedBatch> {
        if pending.is_empty()
            && let Some(batch) = unresolved.pop_front()
        {
            let resolved = resolve_batch(&self.store, batch, tenant)?;
            self.record_resolution(&resolved);
            *pending = resolved.tensors;
        }
        if pending.is_empty() {
            return Ok(PreparedBatch::empty(self.devices.len()));
        }
        let mut prepared = self.prepare_pending(pending, tenant, allow_deferred)?;
        if prepared.is_empty() {
            return Ok(prepared);
        }
        let upload_elapsed = self.upload_prepared(&mut prepared)?;
        self.io_status.upload_microseconds = self
            .io_status
            .upload_microseconds
            .saturating_add(duration_microseconds(upload_elapsed));
        Ok(prepared)
    }

    fn record_resolution(&mut self, resolved: &ResolvedBatch) {
        self.io_status.resolve_batches = self.io_status.resolve_batches.saturating_add(1);
        self.io_status.resolve_bytes = self
            .io_status
            .resolve_bytes
            .saturating_add(u64::try_from(resolved.payload_bytes).unwrap_or(u64::MAX));
        self.io_status.resolve_microseconds = self
            .io_status
            .resolve_microseconds
            .saturating_add(duration_microseconds(resolved.elapsed));
    }

    fn pipeline_stage_microseconds(&self) -> u64 {
        self.io_status
            .resolve_microseconds
            .saturating_add(self.io_status.upload_microseconds)
            .saturating_add(self.io_status.compute_microseconds)
    }

    fn prepare_pending(
        &mut self,
        pending: &mut VecDeque<MissingTensor>,
        tenant: &str,
        allow_deferred: bool,
    ) -> Result<PreparedBatch> {
        let mut prepared = PreparedBatch::empty(self.devices.len());
        let mut consumed = 0;
        for tensor in pending.iter() {
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
                prepared.chunks[device_index].push(ResidentTensor {
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
                    tenant,
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

            if admitted.is_none() && prepared.is_empty() {
                // TinyLFU may reject a cold item, but every requested tensor must
                // still be scored. A transient page run is removed after this
                // chunk completes. During overlap it may be temporarily deferred
                // because the current compute batch still owns all freeable pages.
                for step in 0..self.devices.len() {
                    let device_index = (self.next_device + step) % self.devices.len();
                    if let Admission::Admitted { offset, .. } =
                        self.devices[device_index].cache.admit_for_tenant(
                            tenant,
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
                if !prepared.is_empty() || allow_deferred {
                    break;
                }
                bail!("one tensor cannot be scheduled in any configured Rust GPU block cache");
            };
            prepared.uploads[device_index].push((offset, Arc::clone(&tensor.payload)));
            prepared.chunks[device_index].push(ResidentTensor {
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
            if allow_deferred {
                return Ok(prepared);
            }
            bail!("Rust multi-GPU scheduler made no progress");
        }
        pending.drain(..consumed);
        Ok(prepared)
    }

    fn upload_prepared(&mut self, prepared: &mut PreparedBatch) -> Result<Duration> {
        let started = Instant::now();
        let gpus = self.gpus();
        let results = std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (device_index, (gpu, upload)) in gpus.iter().zip(&prepared.uploads).enumerate() {
                if upload.is_empty() {
                    continue;
                }
                workers.push((
                    device_index,
                    scope.spawn(move || -> Result<()> {
                        let items = upload
                            .iter()
                            .map(|(offset, payload)| (*offset, payload.as_ref()))
                            .collect::<Vec<_>>();
                        gpu.upload_batch(&items)
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
                continue;
            }
            self.devices[device].h2d_batches = self.devices[device].h2d_batches.saturating_add(1);
            self.devices[device].h2d_bytes = self.devices[device].h2d_bytes.saturating_add(
                prepared.uploads[device]
                    .iter()
                    .map(|(_, payload)| payload.len() as u64)
                    .sum(),
            );
        }
        for (device, chunk) in prepared.chunks.iter().enumerate() {
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
        prepared.uploads.iter_mut().for_each(Vec::clear);
        if let Some(error) = first_error {
            self.cleanup_after_upload_failure(&prepared.chunks, &succeeded)?;
            Err(error)
        } else {
            Ok(started.elapsed())
        }
    }

    fn gpus(&self) -> Vec<Arc<Gpu>> {
        self.devices
            .iter()
            .map(|device| Arc::clone(&device.gpu))
            .collect()
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
        let store = self.store.lock().unwrap_or_else(|error| error.into_inner());
        EngineStatus {
            devices,
            host: store.host_status(),
            batch_read_calls: store.batch_read_calls,
            batch_read_bytes: store.batch_read_bytes,
            io_pipeline: self.io_status.clone(),
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
            "io_pipeline": {
                "mode": if status.io_pipeline.overlap_enabled { "overlap" } else { "serial" },
                "batch_bytes": status.io_pipeline.batch_bytes,
                "resolve_batches": status.io_pipeline.resolve_batches,
                "resolve_bytes": status.io_pipeline.resolve_bytes,
                "resolve_microseconds": status.io_pipeline.resolve_microseconds,
                "upload_microseconds": status.io_pipeline.upload_microseconds,
                "compute_microseconds": status.io_pipeline.compute_microseconds,
                "overlap_cycles": status.io_pipeline.overlap_cycles,
                "overlap_microseconds": status.io_pipeline.overlap_microseconds,
            },
        })
    }
}

fn resolve_batch(
    store: &Arc<Mutex<ShardStore>>,
    batch: UnresolvedBatch,
    tenant: &str,
) -> Result<ResolvedBatch> {
    let started = Instant::now();
    let payloads = store
        .lock()
        .map_err(|_| anyhow!("immutable shard store lock is poisoned"))?
        .resolve_many(&batch.descriptors, tenant)?;
    let elapsed = started.elapsed();
    let tensors = batch
        .indices
        .into_iter()
        .zip(batch.descriptors)
        .zip(payloads)
        .map(|((candidate_index, descriptor), payload)| MissingTensor {
            candidate_index,
            key: cache_key(&descriptor),
            descriptor,
            payload,
        })
        .collect();
    Ok(ResolvedBatch {
        tensors,
        payload_bytes: batch.payload_bytes,
        elapsed,
    })
}

fn split_unresolved_batches(
    indices: Vec<usize>,
    descriptors: Vec<Descriptor>,
    maximum_bytes: usize,
) -> Result<VecDeque<UnresolvedBatch>> {
    if indices.len() != descriptors.len() {
        bail!("internal missing tensor metadata disagrees");
    }
    let mut output = VecDeque::new();
    let mut batch_indices = Vec::new();
    let mut batch_descriptors = Vec::new();
    let mut batch_bytes = 0_usize;
    for (index, descriptor) in indices.into_iter().zip(descriptors) {
        let payload_bytes = descriptor_payload_bytes(&descriptor)?;
        if !batch_descriptors.is_empty()
            && batch_bytes.saturating_add(payload_bytes) > maximum_bytes
        {
            output.push_back(UnresolvedBatch {
                indices: std::mem::take(&mut batch_indices),
                descriptors: std::mem::take(&mut batch_descriptors),
                payload_bytes: batch_bytes,
            });
            batch_bytes = 0;
        }
        batch_indices.push(index);
        batch_descriptors.push(descriptor);
        batch_bytes = batch_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| anyhow!("I/O pipeline batch byte count overflow"))?;
    }
    if !batch_descriptors.is_empty() {
        output.push_back(UnresolvedBatch {
            indices: batch_indices,
            descriptors: batch_descriptors,
            payload_bytes: batch_bytes,
        });
    }
    Ok(output)
}

fn descriptor_payload_bytes(descriptor: &Descriptor) -> Result<usize> {
    let scalar_bytes = match descriptor.dtype {
        1 => 4_usize,
        2 => 2_usize,
        _ => bail!("unsupported tensor dtype"),
    };
    (descriptor.rows as usize)
        .checked_mul(descriptor.dimension as usize)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .ok_or_else(|| anyhow!("tensor payload byte count overflow"))
}

fn score_chunks(
    gpus: &[Arc<Gpu>],
    request: &Request,
    chunks: &[Vec<ResidentTensor>],
) -> Result<(Vec<(usize, f32)>, Duration)> {
    let started = Instant::now();
    let completed = std::thread::scope(|scope| -> Result<Vec<Vec<(usize, f32)>>> {
        let mut workers = Vec::new();
        for (gpu, chunk) in gpus.iter().zip(chunks) {
            if chunk.is_empty() {
                continue;
            }
            workers.push(scope.spawn(move || -> Result<Vec<(usize, f32)>> {
                let offsets = chunk.iter().map(|item| item.offset).collect::<Vec<_>>();
                let rows = chunk.iter().map(|item| item.rows).collect::<Vec<_>>();
                let computed = gpu.score(
                    &request.query,
                    request.query_rows,
                    request.dimension,
                    request.dtype,
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
    Ok((completed.into_iter().flatten().collect(), started.elapsed()))
}

fn apply_scores(scores: &mut [Option<f32>], computed: Vec<(usize, f32)>) {
    for (candidate_index, score) in computed {
        scores[candidate_index] = Some(score);
    }
}

fn duration_microseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn unique_descriptors(descriptors: &[Descriptor]) -> Vec<Descriptor> {
    let mut seen = HashSet::with_capacity(descriptors.len());
    descriptors
        .iter()
        .filter(|descriptor| seen.insert(cache_key(descriptor)))
        .cloned()
        .collect()
}

fn validate_entry(descriptor: &Descriptor, entry: &crate::cache::CacheEntry) -> Result<()> {
    let scalar_bytes = if descriptor.dtype == 1 { 4 } else { 2 };
    let expected_bytes = descriptor.rows as usize * descriptor.dimension as usize * scalar_bytes;
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
    fn io_batches_are_payload_bounded_and_keep_one_oversized_tensor() {
        let descriptors = vec![
            descriptor(1, "a", 1),
            descriptor(2, "b", 1),
            descriptor(3, "c", 3),
        ];
        // One row is 640 bytes. The first two fit exactly; the final tensor is
        // larger than the target but must remain a schedulable one-item batch.
        let batches = split_unresolved_batches(vec![0, 1, 2], descriptors, 1_280).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].indices, vec![0, 1]);
        assert_eq!(batches[0].payload_bytes, 1_280);
        assert_eq!(batches[1].indices, vec![2]);
        assert_eq!(batches[1].payload_bytes, 1_920);
    }
}
