// This software is licensed under the repository's dual license model.

use super::*;
use std::sync::mpsc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IoPipelineMode {
    #[default]
    Disabled,
    Serial,
    Overlap,
}

impl Engine {
    pub fn configure_io_pipeline(
        &mut self,
        mode: IoPipelineMode,
        batch_bytes: usize,
    ) -> Result<()> {
        if batch_bytes == 0 || batch_bytes > 1024 * 1024 * 1024 {
            bail!("I/O batch bytes must be between 1 and 1073741824");
        }
        self.io_pipeline = mode;
        self.io_batch_bytes = batch_bytes;
        self.io_upload_batch_bytes = batch_bytes;
        Ok(())
    }

    pub fn configure_io_upload_batch(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 || bytes > 4096usize * 1024 * 1024 {
            bail!("upload batch bytes must be between 1 and 4294967296");
        }
        self.io_upload_batch_bytes = bytes;
        Ok(())
    }

    pub(super) fn score_missing_pipeline(
        &mut self,
        request: &Request,
        indices: Vec<usize>,
        descriptors: Vec<Descriptor>,
        scores: &mut [Option<f32>],
    ) -> Result<()> {
        let batches = descriptor_batches(indices, descriptors, self.io_batch_bytes)?;
        let store = Arc::clone(&self.store);
        let tenant = &request.tenant;
        let started = std::time::Instant::now();
        let mut resolve_us = 0u128;
        let mut consume_us = 0u128;
        let mut completed = 0usize;
        let mut upload_groups = 0usize;
        let mut grouping = UploadGroup::new(self.io_upload_batch_bytes);
        let overlap = self.io_pipeline == IoPipelineMode::Overlap;
        let result = {
            let mut consume = |pending: &mut Vec<MissingTensor>| {
                let started = std::time::Instant::now();
                let result = self.score_pending(request, None, pending, scores);
                consume_us += started.elapsed().as_micros();
                upload_groups += 1;
                result
            };
            run_batches(
                batches,
                overlap,
                move |batch| {
                    let started = std::time::Instant::now();
                    let pending = resolve(&store, batch, tenant)?;
                    Ok((pending, started.elapsed().as_micros()))
                },
                |(pending, elapsed)| {
                    resolve_us += elapsed;
                    completed += 1;
                    for tensor in pending {
                        grouping.push(tensor, &mut consume)?;
                    }
                    Ok(())
                },
            )
            .and_then(|()| grouping.flush(&mut consume))
        };
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "tilemaxsim_io_pipeline", "mode": self.io_pipeline,
                "batch_bytes": self.io_batch_bytes, "completed_batches": completed,
                "upload_batch_bytes": self.io_upload_batch_bytes, "upload_groups": upload_groups,
                "resolve_us": resolve_us, "consume_us": consume_us,
                "elapsed_us": started.elapsed().as_micros(), "success": result.is_ok()
            })
        );
        result
    }
}

// Coalesce immutable read results without changing their order or GPU policy.
// Payload bound: upload budget plus three read batches (incoming, queued and
// resolving). An indivisible oversized tensor remains subject to GPU capacity.
struct UploadGroup {
    pending: Vec<MissingTensor>,
    bytes: usize,
    budget: usize,
}
impl UploadGroup {
    fn new(budget: usize) -> Self {
        Self {
            pending: Vec::new(),
            bytes: 0,
            budget,
        }
    }
    fn push(
        &mut self,
        tensor: MissingTensor,
        consume: &mut impl FnMut(&mut Vec<MissingTensor>) -> Result<()>,
    ) -> Result<()> {
        let bytes = tensor.payload.len();
        if !self.pending.is_empty() && self.bytes.saturating_add(bytes) > self.budget {
            self.flush(consume)?;
        }
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("upload grouping size overflow"))?;
        self.pending.push(tensor);
        if self.bytes >= self.budget {
            self.flush(consume)?;
        }
        Ok(())
    }
    fn flush(
        &mut self,
        consume: &mut impl FnMut(&mut Vec<MissingTensor>) -> Result<()>,
    ) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        consume(&mut self.pending)?;
        self.pending.clear();
        self.bytes = 0;
        Ok(())
    }
}

// The consumer exclusively owns GPU/cache state. The worker only resolves
// immutable storage bytes, so this does not require concurrent GPU aliases.
fn run_batches<I, T, Load, Consume>(
    batches: Vec<I>,
    overlap: bool,
    mut load: Load,
    mut consume: Consume,
) -> Result<()>
where
    I: Send,
    T: Send,
    Load: FnMut(I) -> Result<T> + Send,
    Consume: FnMut(T) -> Result<()>,
{
    if !overlap {
        for batch in batches {
            consume(load(batch)?)?;
        }
        return Ok(());
    }
    std::thread::scope(|scope| {
        // One queued, one resolving, one consumed batch. Drop receiver before
        // joining on all error paths, unblocking a producer waiting in send.
        let (sender, receiver) = mpsc::sync_channel(1);
        let producer = scope.spawn(move || {
            for batch in batches {
                let resolved = load(batch);
                let failed = resolved.is_err();
                if sender.send(resolved).is_err() || failed {
                    break;
                }
            }
        });
        let result = (|| {
            for resolved in &receiver {
                consume(resolved?)?;
            }
            Ok(())
        })();
        drop(receiver);
        let joined = producer
            .join()
            .map_err(|_| anyhow!("I/O resolver worker panicked"));
        result.and(joined)
    })
}

type DescriptorBatch = Vec<(usize, Descriptor)>;

fn descriptor_batches(
    indices: Vec<usize>,
    descriptors: Vec<Descriptor>,
    budget: usize,
) -> Result<Vec<DescriptorBatch>> {
    if budget == 0 || indices.len() != descriptors.len() {
        bail!("invalid I/O batch budget or descriptor count");
    }
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut bytes = 0usize;
    for (index, descriptor) in indices.into_iter().zip(descriptors) {
        let size = descriptor_payload_bytes(&descriptor)?;
        if !current.is_empty() && bytes.saturating_add(size) > budget {
            batches.push(std::mem::take(&mut current));
            bytes = 0;
        }
        // One indivisible tensor may exceed the batch budget; existing
        // descriptor and GPU capacity checks still bound/reject that tensor.
        bytes = bytes
            .checked_add(size)
            .ok_or_else(|| anyhow!("I/O batch size overflow"))?;
        current.push((index, descriptor));
    }
    if !current.is_empty() {
        batches.push(current);
    }
    Ok(batches)
}

fn descriptor_payload_bytes(d: &Descriptor) -> Result<usize> {
    let element_bytes = match d.dtype {
        1 => 4,
        2 => 2,
        3 => 1,
        _ => bail!("unsupported tensor storage dtype"),
    };
    (d.rows as usize)
        .checked_mul(d.dimension as usize)
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or_else(|| anyhow!("tensor storage size overflow"))
}

fn resolve(
    store: &Mutex<ShardStore>,
    batch: DescriptorBatch,
    tenant: &str,
) -> Result<Vec<MissingTensor>> {
    let descriptors = batch.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>();
    let payloads = store
        .lock()
        .map_err(|_| anyhow!("tensor store lock poisoned"))?
        .resolve_many(&descriptors, tenant)?;
    batch
        .into_iter()
        .zip(payloads)
        .map(|((candidate_index, descriptor), payload)| {
            let payload = encode_for_profile(&descriptor, payload, ScoringProfile::ExactFp16)?;
            Ok(MissingTensor {
                candidate_index,
                key: gpu_cache_key(&descriptor, ScoringProfile::ExactFp16, None),
                descriptor,
                payload,
            })
        })
        .collect()
}

#[cfg(all(test, feature = "backend-cpu"))]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    struct Fixture {
        root: PathBuf,
        request: Request,
    }
    impl Fixture {
        fn new(count: usize) -> Self {
            let root = std::env::temp_dir().join(format!(
                "io-overlap-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let mut descriptors = Vec::new();
            for i in 0..count {
                let rows = 1 + (i % 7) as u32;
                let payload = (0..rows * 64)
                    .flat_map(|n| {
                        (if (n + i as u32).is_multiple_of(5) {
                            0x3c00u16
                        } else {
                            0u16
                        })
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                let digest = hex::encode(Sha256::digest(&payload));
                let file = root
                    .join("objects")
                    .join(&digest[..2])
                    .join(format!("{digest}.tensor"));
                std::fs::create_dir_all(file.parent().unwrap()).unwrap();
                std::fs::write(file, payload).unwrap();
                descriptors.push(Descriptor {
                    candidate_id: i as u32,
                    contract: "io-test@1".into(),
                    digest: digest.clone(),
                    rows,
                    dimension: 64,
                    dtype: 2,
                    raw_fp8_cache_key: Default::default(),
                });
            }
            // Preserve different public IDs for duplicate immutable content.
            let mut duplicate = descriptors[0].clone();
            duplicate.candidate_id = count as u32;
            descriptors.push(duplicate);
            Self {
                root,
                request: Request {
                    protocol_version: 5,
                    request_id: 1,
                    tenant: "owner-a".into(),
                    model_contract: "io-test@1".into(),
                    priority: 0,
                    timeout_ms: 1000,
                    query_rows: 3,
                    dimension: 64,
                    dtype: 2,
                    candidate_dtype: 2,
                    scoring_profile: ScoringProfile::ExactFp16,
                    quantization_contract: None,
                    top_k: None,
                    query: (0..192).flat_map(|_| 0x3c00u16.to_le_bytes()).collect(),
                    candidate_end: descriptors.len(),
                    candidates: Arc::new(descriptors),
                    candidate_start: 0,
                    manifest_digest: None,
                    catalog_digest: None,
                    catalog_selection_digest: None,
                    catalog_public_ids: Vec::new(),
                    catalog_registration: false,
                    scoped_candidate_ordinals: Vec::new(),
                },
            }
        }
        fn engine(&self, mode: IoPipelineMode, capacity: usize, host: usize) -> Engine {
            let mut store =
                ShardStore::open(&[("io-test@1".into(), self.root.clone())], host, 80, false)
                    .unwrap();
            store.set_read_workers(4).unwrap();
            let mut engine = Engine::new(
                vec![crate::cpu::CpuBackend::create(0, capacity + 8192, 8192).unwrap()],
                256,
                store,
                80,
                80,
                &HashMap::new(),
                None,
            )
            .unwrap();
            engine.configure_io_pipeline(mode, 1024).unwrap();
            engine.enable_recovery();
            engine
        }
        fn assert_released(&self, engine: &Engine) {
            for d in self.request.candidate_slice() {
                let key = gpu_cache_key(d, ScoringProfile::ExactFp16, None);
                for device in &engine.devices {
                    if let Some(entry) = device.cache.entry(&key) {
                        assert_eq!(entry.references, 0);
                        assert!(entry.ready);
                    }
                }
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
    #[test]
    fn next_read_runs_while_consumer_is_active_and_order_is_stable() {
        let (loaded, observed) = std::sync::mpsc::channel();
        let (consuming, active) = std::sync::mpsc::channel();
        let mut seen = Vec::new();
        run_batches(
            vec![0, 1, 2],
            true,
            move |index| {
                if index == 1 {
                    active
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    loaded.send(()).unwrap();
                }
                Ok(index)
            },
            |index| {
                seen.push(index);
                if index == 0 {
                    consuming.send(()).unwrap();
                    observed
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, [0, 1, 2]);
    }

    #[test]
    fn producer_panic_and_early_consumer_failure_are_joined() {
        assert!(
            run_batches(
                vec![0, 1, 2],
                true,
                |index| -> Result<usize> {
                    if index == 1 {
                        panic!("injected resolver failure");
                    }
                    Ok(index)
                },
                |_| Ok(())
            )
            .is_err()
        );
        assert!(
            run_batches((0..1000).collect(), true, Ok, |_| bail!(
                "injected consumer failure"
            ))
            .is_err()
        );
    }

    #[test]
    fn upload_groups_preserve_order_and_payload_budget() {
        let fixture = Fixture::new(37);
        for budget in [1, 128, 1024, 8192] {
            let mut engine = fixture.engine(IoPipelineMode::Overlap, 65536, 0);
            engine.configure_io_upload_batch(budget).unwrap();
            let mut grouping = UploadGroup::new(budget);
            let descriptors = fixture.request.candidates.as_ref().clone();
            let pending = resolve(
                &engine.store,
                (0..descriptors.len()).zip(descriptors).collect(),
                "owner-a",
            )
            .unwrap();
            let mut seen = Vec::new();
            let mut consume = |batch: &mut Vec<MissingTensor>| {
                let bytes = batch.iter().map(|t| t.payload.len()).sum::<usize>();
                assert!(bytes <= budget || batch.len() == 1);
                seen.extend(batch.iter().map(|t| t.candidate_index));
                Ok(())
            };
            for tensor in pending {
                grouping.push(tensor, &mut consume).unwrap();
            }
            grouping.flush(&mut consume).unwrap();
            assert!(grouping.pending.is_empty());
            assert_eq!(grouping.bytes, 0);
            assert_eq!(
                seen,
                (0..fixture.request.candidates.len()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn independent_upload_budget_keeps_scores_owners_and_releases_leases() {
        let fixture = Fixture::new(37);
        let baseline = fixture
            .engine(IoPipelineMode::Disabled, 65536, 0)
            .score(&fixture.request)
            .unwrap();
        for mode in [IoPipelineMode::Serial, IoPipelineMode::Overlap] {
            for upload in [128, 1024, 4096, 65536] {
                for capacity in [4096, 65536] {
                    let mut engine = fixture.engine(mode, capacity, 0);
                    engine.configure_io_upload_batch(upload).unwrap();
                    assert_eq!(engine.score(&fixture.request).unwrap(), baseline);
                    fixture.assert_released(&engine);
                    assert!(
                        engine
                            .recovery_snapshot()
                            .unwrap()
                            .iter()
                            .all(|e| e.tenant == "owner-a")
                    );
                }
            }
        }
        let mut old = fixture.engine(IoPipelineMode::Overlap, 65536, 0);
        old.score(&fixture.request).unwrap();
        let mut combined = fixture.engine(IoPipelineMode::Overlap, 65536, 0);
        combined.configure_io_upload_batch(65536).unwrap();
        combined.score(&fixture.request).unwrap();
        assert!(
            combined.status_snapshot().devices[0].h2d_batches
                < old.status_snapshot().devices[0].h2d_batches
        );
    }

    #[test]
    fn coalescing_consumer_error_stops_producer_and_drops_unscored_payloads() {
        let fixture = Fixture::new(37);
        let mut engine = fixture.engine(IoPipelineMode::Overlap, 256, 0);
        engine.configure_io_upload_batch(65536).unwrap();
        assert!(engine.score(&fixture.request).is_err());
        fixture.assert_released(&engine);
    }

    #[test]
    fn pipeline_scores_match_legacy_under_eviction_and_host_pressure() {
        let fixture = Fixture::new(37);
        let baseline = fixture
            .engine(IoPipelineMode::Disabled, 65536, 0)
            .score(&fixture.request)
            .unwrap();
        for mode in [IoPipelineMode::Serial, IoPipelineMode::Overlap] {
            for (capacity, host) in [(4096, 0), (4096, 2048), (65536, 65536)] {
                let mut engine = fixture.engine(mode, capacity, host);
                for _ in 0..3 {
                    assert_eq!(engine.score(&fixture.request).unwrap(), baseline);
                    fixture.assert_released(&engine);
                    let recovery = engine.recovery_snapshot().unwrap();
                    assert!(recovery.iter().all(|e| e.tenant == "owner-a"));
                }
                let mut other = fixture.request.clone();
                other.tenant = "owner-b".into();
                assert_eq!(engine.score(&other).unwrap(), baseline);
                fixture.assert_released(&engine);
            }
        }
    }
    #[test]
    fn recovery_keeps_committed_membership_and_owner_with_overlap() {
        let fixture = Fixture::new(37);
        let mut original = fixture.engine(IoPipelineMode::Disabled, 65536, 0);
        let expected = original.score(&fixture.request).unwrap();
        let snapshot = original.recovery_snapshot().unwrap();
        let mut recovered = fixture.engine(IoPipelineMode::Overlap, 65536, 0);
        recovered.restore_recovery(&snapshot, 256).unwrap();
        assert_eq!(recovered.score(&fixture.request).unwrap(), expected);
        let after = recovered.recovery_snapshot().unwrap();
        assert_eq!(after.len(), snapshot.len());
        assert!(after.iter().all(|e| e.tenant == "owner-a"));
        fixture.assert_released(&recovered);
    }

    #[test]
    fn fp32_shapes_use_the_existing_exact_storage_contract() {
        let mut fixture = Fixture::new(9);
        let descriptors = Arc::make_mut(&mut fixture.request.candidates);
        for d in descriptors {
            let payload = (0..d.rows * d.dimension)
                .flat_map(|i| ((i % 7) as f32 * 0.125).to_le_bytes())
                .collect::<Vec<_>>();
            d.dtype = 1;
            d.digest = hex::encode(Sha256::digest(&payload));
            let file = fixture
                .root
                .join("objects")
                .join(&d.digest[..2])
                .join(format!("{}.tensor", d.digest));
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, payload).unwrap();
        }
        fixture.request.dtype = 1;
        fixture.request.candidate_dtype = 1;
        fixture.request.query = (0..192)
            .flat_map(|i| ((i % 3) as f32 * 0.25).to_le_bytes())
            .collect();
        let baseline = fixture
            .engine(IoPipelineMode::Disabled, 65536, 0)
            .score(&fixture.request)
            .unwrap();
        for mode in [IoPipelineMode::Serial, IoPipelineMode::Overlap] {
            let mut engine = fixture.engine(mode, 8192, 0);
            assert_eq!(engine.score(&fixture.request).unwrap(), baseline);
            fixture.assert_released(&engine);
        }
    }

    #[test]
    fn pipeline_does_not_bypass_raw_fp8_profile_guard() {
        let mut fixture = Fixture::new(1);
        let mut d = fixture.request.candidates[0].clone();
        let payload = vec![0x38u8; 64];
        d.dtype = 3;
        d.digest = hex::encode(Sha256::digest(&payload));
        let file = fixture
            .root
            .join("objects")
            .join(&d.digest[..2])
            .join(format!("{}.tensor", d.digest));
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, payload).unwrap();
        fixture.request.candidates = Arc::new(vec![d]);
        fixture.request.candidate_end = 1;
        fixture.request.candidate_dtype = 3;
        for mode in [
            IoPipelineMode::Disabled,
            IoPipelineMode::Serial,
            IoPipelineMode::Overlap,
        ] {
            let mut engine = fixture.engine(mode, 4096, 0);
            let error = engine.score(&fixture.request).unwrap_err().to_string();
            assert!(
                error.contains("raw FP8 canonical tensors require"),
                "{error}"
            );
            fixture.assert_released(&engine);
        }
    }

    #[test]
    fn producer_failure_does_not_publish_partial_result_or_leave_leases() {
        let fixture = Fixture::new(37);
        let d = &fixture.request.candidates[19];
        let file = fixture
            .root
            .join("objects")
            .join(&d.digest[..2])
            .join(format!("{}.tensor", d.digest));
        std::fs::write(file, [0u8]).unwrap();
        for mode in [IoPipelineMode::Serial, IoPipelineMode::Overlap] {
            let mut engine = fixture.engine(mode, 4096, 0);
            assert!(engine.score(&fixture.request).is_err());
            fixture.assert_released(&engine);
        }
    }
    #[test]
    fn consumer_failure_unblocks_bounded_producer() {
        let fixture = Fixture::new(37);
        let mut engine = fixture.engine(IoPipelineMode::Overlap, 256, 0);
        assert!(engine.score(&fixture.request).is_err());
        fixture.assert_released(&engine);
    }
    #[test]
    fn batches_keep_original_shapes_and_budget_except_indivisible_tensor() {
        let fixture = Fixture::new(37);
        let descriptors = fixture.request.candidates.as_ref().clone();
        for budget in [1, 256, 1024, 1 << 20] {
            let batches = descriptor_batches(
                (0..descriptors.len()).collect(),
                descriptors.clone(),
                budget,
            )
            .unwrap();
            let mut flat = Vec::new();
            for batch in batches {
                assert!(
                    batch.len() == 1
                        || batch
                            .iter()
                            .map(|(_, d)| descriptor_payload_bytes(d).unwrap())
                            .sum::<usize>()
                            <= budget
                );
                flat.extend(batch);
            }
            for ((index, actual), expected) in flat.into_iter().zip(&descriptors) {
                assert_eq!(actual.rows, expected.rows);
                assert_eq!(actual.dimension, expected.dimension);
                assert_eq!(actual.dtype, expected.dtype);
                assert_eq!(actual.candidate_id as usize, index);
                assert_eq!(actual.digest, expected.digest);
            }
        }
    }
}
