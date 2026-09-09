// This software is licensed under the repository's dual license model.
//! Experimental Vulkan compute backend, including AMD through Mesa RADV.
//! WSL requires a working Vulkan ICD (e.g. Dozen); /dev/dxg alone is insufficient.

use crate::backend::{
    AcceleratorBackend, AdaptiveStatus, BackendCapabilities, BackendKind, DeviceInfo,
};
use anyhow::{Result, anyhow, bail, ensure};
use std::collections::HashSet;
use wgpu::util::DeviceExt;

pub struct VulkanBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    arena: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    tensor_bytes: usize,
    workspace_bytes: usize,
    info: DeviceInfo,
}

impl VulkanBackend {
    pub fn create(ordinal: i32, total_bytes: usize, workspace_bytes: usize) -> Result<Self> {
        Self::create_inner(ordinal, total_bytes, workspace_bytes, false)
    }

    fn create_inner(
        ordinal: i32,
        total_bytes: usize,
        workspace_bytes: usize,
        allow_software: bool,
    ) -> Result<Self> {
        ensure!(ordinal >= 0, "Vulkan device ordinal must be nonnegative");
        ensure!(
            workspace_bytes > 0 && workspace_bytes < total_bytes,
            "invalid Vulkan arena configuration"
        );
        let tensor_bytes = total_bytes - workspace_bytes;
        ensure!(
            tensor_bytes.is_multiple_of(4),
            "Vulkan tensor arena must be aligned to four bytes"
        );
        let mut flags = wgpu::InstanceFlags::default();
        // Dozen reports no Vulkan conformance certification. Opt in explicitly;
        // our numerical startup probe still runs and software remains rejected.
        if std::env::var("TILEMAXSIM_VULKAN_ALLOW_NONCONFORMANT").as_deref() == Ok("1") {
            flags |= wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER;
            #[cfg(target_os = "linux")]
            pin_wsl_runtime()?;
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags,
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance.enumerate_adapters(wgpu::Backends::VULKAN).into_iter()
            .filter(|adapter| allow_software || matches!(adapter.get_info().device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu))
            .nth(ordinal as usize)
            .ok_or_else(|| anyhow!("no hardware Vulkan adapter at ordinal {ordinal}; install a Vulkan compute ICD. WSL2 requires Dozen/D3D12; /dev/dxg alone is insufficient. Software adapters are not a GPU fallback"))?;
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        ensure!(
            tensor_bytes as u64 <= limits.max_buffer_size
                && tensor_bytes <= limits.max_storage_buffer_binding_size as usize,
            "Vulkan tensor arena exceeds adapter storage-buffer limit {} bytes; reduce --device-memory-gb",
            limits.max_storage_buffer_binding_size
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tilemaxsim-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact-maxsim"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exact-maxsim"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            bail!("Vulkan pipeline: {error}");
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor-arena"),
            size: tensor_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            bail!("Vulkan arena allocation: {error}");
        }
        let info = DeviceInfo {
            backend: BackendKind::Vulkan,
            ordinal,
            name: adapter_info.name,
            architecture: format!(
                "vulkan-vendor-{:04x}-device-{:04x}",
                adapter_info.vendor, adapter_info.device
            ),
            driver_version: None,
            runtime_version: None,
            library_version: None,
            total_memory_bytes: None,
            compute_units: None,
            warp_size: None,
            memory_bus_width_bits: None,
            memory_clock_khz: None,
            shared_memory_per_block_bytes: None,
            persisting_l2_bytes: None,
            matrix_engine_workspace_bytes: None,
            pinned_control_staging: None,
            control_staging_speedup_milli: None,
            double_buffered_tile: None,
            double_buffer_speedup_milli: None,
            document_tile_query_rows: Some(1),
            pq_warp_task_max_document_rows: None,
            compute_queue_priority: None,
            tuning_profile: format!(
                "experimental-vulkan-exact/{} / {}",
                adapter_info.driver, adapter_info.driver_info
            ),
            capabilities: BackendCapabilities {
                kind: BackendKind::Vulkan,
                exact_fp16: true,
                exact_fp32: true,
                int8: false,
                fp8_e4m3: false,
                pq: false,
                opq_rpq: false,
                fused_multiquery: false,
                matrix_engine: false,
                asynchronous_copy: false,
                unified_memory: adapter_info.device_type == wgpu::DeviceType::IntegratedGpu,
                persisting_l2: false,
            },
        };
        Ok(Self {
            device,
            queue,
            arena,
            pipeline,
            tensor_bytes,
            workspace_bytes,
            info,
        })
    }

    fn buffer(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            })
    }
}

// WSL D3D12 registers thread-exit callbacks. Unloading it when the last Vulkan
// instance drops can leave callbacks pointing into unmapped code. Keep only
// the runtime library mapped until process exit, not device allocations. LOCAL
// is essential: LD_PRELOAD interposes D3D12 symbols and is not equivalent.
#[cfg(target_os = "linux")]
fn pin_wsl_runtime() -> Result<()> {
    if !std::path::Path::new("/dev/dxg").exists() {
        return Ok(());
    }
    static PINNED: std::sync::OnceLock<std::result::Result<(), String>> =
        std::sync::OnceLock::new();
    let result = PINNED.get_or_init(|| {
        // SAFETY: static NUL-terminated name; NODELETE preserves driver TLS code
        // after dlclose balances our loader reference. No symbol is called here.
        unsafe {
            let handle = libc::dlopen(
                c"libd3d12.so".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            );
            if handle.is_null() {
                return Err(
                    "cannot load WSL libd3d12.so; include /usr/lib/wsl/lib in LD_LIBRARY_PATH"
                        .to_owned(),
                );
            }
            libc::dlclose(handle);
        }
        Ok(())
    });
    result
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))
        .copied()
}

fn shape_bytes(rows: u32, dimension: u32, dtype: u8) -> Result<usize> {
    let scalar = match dtype {
        1 => 4,
        2 => 2,
        _ => bail!("unsupported Vulkan tensor dtype"),
    };
    ensure!(
        rows > 0 && dimension > 0,
        "Vulkan tensor dimensions must be positive"
    );
    (rows as usize)
        .checked_mul(dimension as usize)
        .and_then(|n| n.checked_mul(scalar))
        .ok_or_else(|| anyhow!("Vulkan tensor shape overflow"))
}

impl AcceleratorBackend for VulkanBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }
    fn tensor_bytes(&self) -> usize {
        self.tensor_bytes
    }
    fn adaptive_status(&self) -> AdaptiveStatus {
        AdaptiveStatus {
            calibration_complete: true,
            tensor_threshold_rows: u32::MAX,
            ..Default::default()
        }
    }
    fn ensure_quantizer(
        &mut self,
        _: &str,
        _: &[u8],
        _: u32,
        _: u16,
        _: u16,
        _: u16,
        _: u16,
    ) -> Result<()> {
        bail!("Vulkan backend does not support PQ-family profiles")
    }
    fn retain_quantizers(&mut self, _: &HashSet<String>) {}
    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        // Validate every range before mutating the arena. Cache allocations are 256-byte aligned.
        for (offset, payload) in items {
            ensure!(
                offset.is_multiple_of(4),
                "Vulkan uploads require four-byte aligned offsets"
            );
            let padded = payload
                .len()
                .checked_add(3)
                .ok_or_else(|| anyhow!("upload size overflow"))?
                & !3;
            ensure!(
                offset
                    .checked_add(padded as u64)
                    .is_some_and(|end| end <= self.tensor_bytes as u64),
                "Vulkan upload exceeds tensor arena"
            );
        }
        for (offset, payload) in items {
            if payload.is_empty() {
                continue;
            }
            if payload.len().is_multiple_of(4) {
                self.queue.write_buffer(&self.arena, *offset, payload);
            } else {
                let mut padded = payload.to_vec();
                padded.resize((payload.len() + 3) & !3, 0);
                self.queue.write_buffer(&self.arena, *offset, &padded);
            }
        }
        // Flush staging allocations even when no scoring follows an upload.
        self.queue.submit([]);
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
    fn score(
        &mut self,
        query: &[u8],
        query_rows: u32,
        dimension: u32,
        dtype: u8,
        profile: u8,
        offsets: &[u64],
        rows: &[u32],
    ) -> Result<Vec<f32>> {
        ensure!(
            profile == 1,
            "Vulkan backend supports only exact FP16/FP32 scoring"
        );
        ensure!(
            !offsets.is_empty() && offsets.len() == rows.len(),
            "invalid Vulkan document metadata"
        );
        let query_bytes = shape_bytes(query_rows, dimension, dtype)?;
        ensure!(
            query.len() == query_bytes,
            "Vulkan query shape disagrees with payload"
        );
        let limits = self.device.limits();
        ensure!(
            offsets.len() <= limits.max_compute_workgroups_per_dimension as usize
                && query_rows <= limits.max_compute_workgroups_per_dimension,
            "Vulkan dispatch exceeds device limits"
        );
        let output_bytes = offsets
            .len()
            .checked_mul(query_rows as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow!("Vulkan output size overflow"))?;
        let metadata_bytes = offsets.len() * 8;
        let padded_query_bytes = query_bytes
            .checked_add(3)
            .ok_or_else(|| anyhow!("query size overflow"))?
            & !3;
        let needed = output_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(padded_query_bytes))
            .and_then(|n| n.checked_add(metadata_bytes + 16))
            .ok_or_else(|| anyhow!("Vulkan workspace size overflow"))?;
        ensure!(
            needed <= self.workspace_bytes,
            "Vulkan request exceeds configured workspace budget"
        );
        for size in [padded_query_bytes, metadata_bytes, output_bytes] {
            ensure!(
                size <= limits.max_storage_buffer_binding_size as usize
                    && size as u64 <= limits.max_buffer_size,
                "Vulkan request buffer exceeds device limits"
            );
        }
        let mut metadata = Vec::with_capacity(metadata_bytes);
        for (&offset, &count) in offsets.iter().zip(rows) {
            let bytes = shape_bytes(count, dimension, dtype)?;
            ensure!(
                offset.is_multiple_of(if dtype == 1 { 4 } else { 2 })
                    && offset <= u32::MAX as u64
                    && offset
                        .checked_add(bytes as u64)
                        .is_some_and(|end| end <= self.tensor_bytes as u64),
                "Vulkan document range exceeds tensor arena or is misaligned"
            );
            metadata.extend_from_slice(&(offset as u32).to_le_bytes());
            metadata.extend_from_slice(&count.to_le_bytes());
        }
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut padded = query.to_vec();
        padded.resize(padded_query_bytes, 0);
        let query_buffer = self.buffer("query", &padded, wgpu::BufferUsages::STORAGE);
        let meta_buffer = self.buffer("metadata", &metadata, wgpu::BufferUsages::STORAGE);
        let params: Vec<u8> = [query_rows, dimension, dtype as u32, offsets.len() as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let params_buffer = self.buffer("params", &params, wgpu::BufferUsages::UNIFORM);
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = [
            &self.arena,
            &query_buffer,
            &meta_buffer,
            &params_buffer,
            &output,
        ];
        let entries: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("maxsim"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(offsets.len() as u32, query_rows, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes as u64);
        self.queue.submit([encoder.finish()]);
        let validation = pollster::block_on(self.device.pop_error_scope());
        let allocation = pollster::block_on(self.device.pop_error_scope());
        if let Some(error) = validation.or(allocation) {
            bail!("Vulkan request failed: {error}");
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let mapped = readback.slice(..).get_mapped_range();
        let scores: Vec<f32> = mapped
            .chunks_exact(query_rows as usize * 4)
            .map(|candidate| {
                candidate
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .sum()
            })
            .collect();
        drop(mapped);
        readback.unmap();
        ensure!(
            scores.iter().all(|score| score.is_finite()),
            "Vulkan MaxSim returned a non-finite score"
        );
        Ok(scores)
    }
    fn score_pq(
        &mut self,
        _: &str,
        _: &[u8],
        _: u32,
        _: u8,
        _: &[u64],
        _: &[u32],
    ) -> Result<Vec<f32>> {
        bail!("Vulkan backend does not support PQ-family profiles")
    }
    fn score_batch(
        &mut self,
        queries: &[u8],
        query_offsets: &[u32],
        dimension: u32,
        dtype: u8,
        profile: u8,
        offsets: &[u64],
        rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            query_offsets.len() >= 2
                && query_offsets[0] == 0
                && query_offsets.windows(2).all(|r| r[0] < r[1]),
            "invalid Vulkan batch query offsets"
        );
        let row_bytes = shape_bytes(1, dimension, dtype)?;
        ensure!(
            shape_bytes(*query_offsets.last().unwrap(), dimension, dtype)? == queries.len(),
            "Vulkan batch shape disagrees with payload"
        );
        query_offsets
            .windows(2)
            .map(|r| {
                self.score(
                    &queries[r[0] as usize * row_bytes..r[1] as usize * row_bytes],
                    r[1] - r[0],
                    dimension,
                    dtype,
                    profile,
                    offsets,
                    rows,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_shapes_without_a_device() {
        assert!(shape_bytes(0, 320, 2).is_err());
        assert!(shape_bytes(1, 0, 2).is_err());
        assert!(shape_bytes(1, 320, 3).is_err());
        assert!(shape_bytes(u32::MAX, u32::MAX, 1).is_err());
        assert_eq!(shape_bytes(3, 5, 2).unwrap(), 30);
    }
    #[test]
    #[ignore = "requires explicitly assigned Vulkan GPU; software is allowed only with TILEMAXSIM_VULKAN_TEST_SOFTWARE=1"]
    fn vulkan_conformance_and_edge_cases() {
        let software = std::env::var("TILEMAXSIM_VULKAN_TEST_SOFTWARE").as_deref() == Ok("1");
        let mut backend = VulkanBackend::create_inner(0, 8 << 20, 4 << 20, software).unwrap();
        eprintln!(
            "Vulkan acceptance device: {} ({})",
            backend.info.name, backend.info.architecture
        );
        crate::backend::run_conformance_probe(&mut backend).unwrap();
        eprintln!("FP16/FP32 conformance passed");
        // Independent scalar oracle, including dimensions wider than a workgroup.
        for dimension in [1u32, 37, 320, 513] {
            let queries: Vec<f32> = (0..5 * dimension)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let docs: Vec<f32> = (0..7 * dimension)
                .map(|i| ((i * 13 % 37) as f32 - 18.0) / 16.0)
                .collect();
            let bytes: Vec<u8> = docs.iter().flat_map(|v| v.to_le_bytes()).collect();
            let qbytes: Vec<u8> = queries.iter().flat_map(|v| v.to_le_bytes()).collect();
            backend.upload_batch(&[(1024, &bytes)]).unwrap();
            let actual = backend
                .score(&qbytes, 5, dimension, 1, 1, &[1024, 1024], &[3, 7])
                .unwrap();
            for (candidate, count) in [3usize, 7].into_iter().enumerate() {
                let expected: f32 = queries
                    .chunks_exact(dimension as usize)
                    .map(|q| {
                        docs.chunks_exact(dimension as usize)
                            .take(count)
                            .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum();
                assert!(
                    (actual[candidate] - expected).abs() < 1e-4,
                    "dimension {dimension}: {} != {expected}",
                    actual[candidate]
                );
            }
        }
        eprintln!("Scalar reference comparisons passed");
        // Packed FP16 across odd row boundaries and multiple workgroup lanes.
        let values = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let half = [0xbc00u16, 0xb800, 0, 0x3800, 0x3c00];
        for dimension in [37usize, 320] {
            let qindices: Vec<usize> = (0..3 * dimension).map(|i| (i * 3 + 1) % 5).collect();
            let dindices: Vec<usize> = (0..7 * dimension).map(|i| (i * 7 + 2) % 5).collect();
            let query: Vec<u8> = qindices
                .iter()
                .flat_map(|i| half[*i].to_le_bytes())
                .collect();
            let doc: Vec<u8> = dindices
                .iter()
                .flat_map(|i| half[*i].to_le_bytes())
                .collect();
            backend.upload_batch(&[(1024, &doc)]).unwrap();
            let actual = backend
                .score(&query, 3, dimension as u32, 2, 1, &[1024], &[7])
                .unwrap()[0];
            let expected: f32 = qindices
                .chunks_exact(dimension)
                .map(|q| {
                    dindices
                        .chunks_exact(dimension)
                        .map(|d| {
                            q.iter()
                                .zip(d)
                                .map(|(a, b)| values[*a] * values[*b])
                                .sum::<f32>()
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .sum();
            assert!((actual - expected).abs() < 1e-4);
        }
        // Odd FP16 dimension/payload, negative maxima, multiple candidates and batch slices.
        let doc: Vec<u8> = [0xbc00u16, 0xc000, 0xc200]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let query: Vec<u8> = [0x3c00u16, 0x3c00, 0x3c00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        backend.upload_batch(&[(512, &doc), (768, &doc)]).unwrap();
        assert_eq!(
            backend
                .score(&query, 1, 3, 2, 1, &[512, 768], &[1, 1])
                .unwrap(),
            vec![-6.0, -6.0]
        );
        assert_eq!(
            backend
                .score_batch(&query.repeat(2), &[0, 1, 2], 3, 2, 1, &[512], &[1])
                .unwrap(),
            vec![vec![-6.0], vec![-6.0]]
        );
        assert!(backend.score(&query, 1, 3, 2, 2, &[512], &[1]).is_err());
        assert!(
            backend
                .score(&query, 1, 3, 2, 1, &[u64::MAX], &[1])
                .is_err()
        );
        assert!(backend.score(&query, 1, 3, 2, 1, &[512], &[0]).is_err());
        assert!(
            backend
                .score_batch(&query, &[0, 2, 1], 3, 2, 1, &[512], &[1])
                .is_err()
        );
        assert!(backend.upload_batch(&[(u64::MAX, &doc)]).is_err());
        eprintln!("Vulkan edge cases passed");
    }
}
