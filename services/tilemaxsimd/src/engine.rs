// Copyright (c) 2026 HuXinjing

use crate::cache::{Admission, GpuCache};
use crate::gpu::Gpu;
use crate::protocol::{Descriptor, Request};
use crate::shard::{ShardStore, cache_key};
use anyhow::{Result, anyhow, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

struct MissingTensor {
    candidate_index: usize,
    descriptor: Descriptor,
    key: String,
    payload: Vec<u8>,
}

struct ResidentTensor {
    candidate_index: usize,
    device: usize,
    key: String,
    offset: u64,
    rows: u32,
    transient: bool,
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
}

impl Engine {
    pub fn new(gpus: Vec<Gpu>, block_bytes: usize, store: ShardStore) -> Result<Self> {
        if gpus.is_empty() {
            bail!("at least one GPU is required");
        }
        let devices = gpus
            .into_iter()
            .map(|gpu| {
                let cache = GpuCache::new(gpu.tensor_bytes(), block_bytes)
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
        })
    }

    pub fn prewarm(&mut self, descriptors: &[Descriptor], batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            bail!("resident prewarm batch size must be positive");
        }
        for batch in descriptors.chunks(batch_size) {
            let payloads = self.store.resolve_many(batch)?;
            let mut uploads = (0..self.devices.len())
                .map(|_| Vec::<(u64, &[u8])>::new())
                .collect::<Vec<_>>();
            let mut acquired = Vec::<(usize, String)>::new();
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
                    acquired.push((device, key));
                    continue;
                }
                self.devices[self.next_device]
                    .cache
                    .record_access_miss(&key);
                let mut admission = None;
                for step in 0..self.devices.len() {
                    let device = (self.next_device + step) % self.devices.len();
                    if let Admission::Admitted { offset, .. } = self.devices[device].cache.admit(
                        key.clone(),
                        payload.len(),
                        descriptor.rows,
                        descriptor.dimension,
                        descriptor.dtype,
                        true,
                        true,
                    ) {
                        admission = Some((device, offset));
                        self.next_device = (device + 1) % self.devices.len();
                        break;
                    }
                }
                let Some((device, offset)) = admission else {
                    bail!("resident manifest exceeds the configured Rust GPU block caches");
                };
                uploads[device].push((offset, payload.as_slice()));
                acquired.push((device, key));
            }
            for (device, items) in uploads.iter().enumerate() {
                if items.is_empty() {
                    continue;
                }
                self.devices[device].gpu.upload_batch(items)?;
                self.devices[device].h2d_batches += 1;
                self.devices[device].h2d_bytes += items
                    .iter()
                    .map(|(_, payload)| payload.len() as u64)
                    .sum::<u64>();
            }
            for (device, key) in acquired {
                self.devices[device]
                    .cache
                    .release(&key)
                    .map_err(|message| anyhow!(message))?;
            }
        }
        Ok(())
    }

    pub fn score_until(
        &mut self,
        request: &Request,
        deadline: Instant,
        canceled: &AtomicBool,
    ) -> Result<Vec<(u32, f32)>> {
        self.score_controlled(request, Some(deadline), Some(canceled))
    }

    fn score_controlled(
        &mut self,
        request: &Request,
        deadline: Option<Instant>,
        canceled: Option<&AtomicBool>,
    ) -> Result<Vec<(u32, f32)>> {
        check_request_control(deadline, canceled)?;
        if request.candidates.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = vec![None; request.candidates.len()];
        let mut hit_chunks = (0..self.devices.len())
            .map(|_| Vec::<ResidentTensor>::new())
            .collect::<Vec<_>>();
        let mut missing_descriptors = Vec::new();
        let mut missing_indices = Vec::new();
        for (index, descriptor) in request.candidates.iter().enumerate() {
            let key = cache_key(descriptor);
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

        let empty_uploads = (0..self.devices.len())
            .map(|_| Vec::<(u64, &[u8])>::new())
            .collect::<Vec<_>>();
        self.execute_devices(request, &hit_chunks, &empty_uploads, &mut scores)?;
        for chunk in &hit_chunks {
            self.release_chunk(chunk)?;
        }

        check_request_control(deadline, canceled)?;
        let payloads = self.store.resolve_many(&missing_descriptors)?;
        let mut pending = missing_indices
            .into_iter()
            .zip(missing_descriptors)
            .zip(payloads)
            .map(|((candidate_index, descriptor), payload)| MissingTensor {
                candidate_index,
                key: cache_key(&descriptor),
                descriptor,
                payload,
            })
            .collect::<Vec<_>>();

        while !pending.is_empty() {
            check_request_control(deadline, canceled)?;
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
                    });
                    consumed += 1;
                    continue;
                }

                let mut admitted = None;
                for step in 0..self.devices.len() {
                    let device_index = (self.next_device + step) % self.devices.len();
                    let admission = self.devices[device_index].cache.admit(
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
                            self.devices[device_index].cache.admit(
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
                uploads[device_index].push((offset, tensor.payload.as_slice()));
                chunks[device_index].push(ResidentTensor {
                    candidate_index: tensor.candidate_index,
                    device: device_index,
                    key: tensor.key.clone(),
                    offset,
                    rows: tensor.descriptor.rows,
                    transient,
                });
                consumed += 1;
            }

            if consumed == 0 {
                bail!("Rust multi-GPU scheduler made no progress");
            }
            self.execute_devices(request, &chunks, &uploads, &mut scores)?;
            for chunk in &chunks {
                self.release_chunk(chunk)?;
            }
            check_request_control(deadline, canceled)?;
            pending.drain(..consumed);
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

    fn execute_devices(
        &mut self,
        request: &Request,
        chunks: &[Vec<ResidentTensor>],
        uploads: &[Vec<(u64, &[u8])>],
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let completed = std::thread::scope(|scope| -> Result<Vec<Vec<(usize, f32)>>> {
            let mut workers = Vec::new();
            for ((device, chunk), upload) in self.devices.iter_mut().zip(chunks).zip(uploads) {
                if chunk.is_empty() {
                    continue;
                }
                workers.push(scope.spawn(move || -> Result<Vec<(usize, f32)>> {
                    if !upload.is_empty() {
                        device.gpu.upload_batch(upload)?;
                        device.h2d_batches += 1;
                        device.h2d_bytes += upload
                            .iter()
                            .map(|(_, payload)| payload.len() as u64)
                            .sum::<u64>();
                    }
                    let offsets = chunk.iter().map(|item| item.offset).collect::<Vec<_>>();
                    let rows = chunk.iter().map(|item| item.rows).collect::<Vec<_>>();
                    let computed = device.gpu.score(
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
        for device_scores in completed {
            for (candidate_index, score) in device_scores {
                scores[candidate_index] = Some(score);
            }
        }
        Ok(())
    }

    fn release_chunk(&mut self, chunk: &[ResidentTensor]) -> Result<()> {
        let mut transient = std::collections::HashSet::new();
        for tensor in chunk {
            self.devices[tensor.device]
                .cache
                .release(&tensor.key)
                .map_err(|message| anyhow!(message))?;
            if tensor.transient {
                transient.insert((tensor.device, tensor.key.clone()));
            }
        }
        for (device, key) in transient {
            self.devices[device]
                .cache
                .remove(&key)
                .map_err(|message| anyhow!(message))?;
        }
        Ok(())
    }

    pub fn status_json(&self) -> serde_json::Value {
        let (host_hits, host_misses, host_evictions, host_rejections) = self.store.host_status();
        let devices = self
            .devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                serde_json::json!({
                    "index": index,
                    "gpu_allocator": "segregated-page-runs",
                    "gpu_tensor_bytes": device.cache.capacity(),
                    "gpu_block_bytes": device.cache.block_bytes(),
                    "gpu_free_bytes": device.cache.free_bytes(),
                    "gpu_largest_free_extent_bytes": device.cache.largest_free_extent(),
                    "gpu_allocated_bytes": device.cache.allocated_bytes(),
                    "gpu_payload_bytes": device.cache.payload_bytes(),
                    "gpu_internal_waste_bytes": device.cache.allocated_bytes()
                        - device.cache.payload_bytes(),
                    "gpu_entries": device.cache.entry_count(),
                    "gpu_pinned_entries": device.cache.pinned_entries(),
                    "gpu_hits": device.cache.hits,
                    "gpu_misses": device.cache.misses,
                    "gpu_evictions": device.cache.evictions,
                    "gpu_admission_rejections": device.cache.admission_rejections,
                    "h2d_batches": device.h2d_batches,
                    "h2d_bytes": device.h2d_bytes,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "devices": devices,
            "host_hits": host_hits,
            "host_misses": host_misses,
            "host_evictions": host_evictions,
            "host_admission_rejections": host_rejections,
            "batch_read_calls": self.store.batch_read_calls,
            "batch_read_bytes": self.store.batch_read_bytes,
        })
    }
}

fn check_request_control(deadline: Option<Instant>, canceled: Option<&AtomicBool>) -> Result<()> {
    if canceled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        bail!("request canceled because the client disconnected");
    }
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        bail!("request deadline expired");
    }
    Ok(())
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
