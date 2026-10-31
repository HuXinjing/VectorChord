//! Deterministic scheduler lifecycle tests. CPU arithmetic here is not a GPU
//! conformance or performance result.

use super::*;
use crate::backend::{AdaptiveStatus, DeviceInfo};
use crate::cpu::CpuBackend;

type Events = Arc<Mutex<Vec<(usize, u16)>>>;

struct FusedCpu {
    cpu: CpuBackend,
    info: DeviceInfo,
    close_after_batch: Arc<Mutex<Option<UnixStream>>>,
    events: Events,
}

impl AcceleratorBackend for FusedCpu {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }
    fn tensor_bytes(&self) -> usize {
        self.cpu.tensor_bytes()
    }
    fn adaptive_status(&self) -> AdaptiveStatus {
        self.cpu.adaptive_status()
    }
    fn ensure_quantizer(
        &mut self,
        id: &str,
        p: &[u8],
        d: u32,
        s: u16,
        u: u16,
        c: u16,
        r: u16,
    ) -> Result<()> {
        self.cpu.ensure_quantizer(id, p, d, s, u, c, r)
    }
    fn retain_quantizers(&mut self, active: &HashSet<String>) {
        self.cpu.retain_quantizers(active);
    }
    fn upload_batch(&mut self, items: &[(u64, &[u8])]) -> Result<()> {
        self.cpu.upload_batch(items)
    }
    fn score(
        &mut self,
        q: &[u8],
        qr: u32,
        d: u32,
        dt: u8,
        p: u8,
        offsets: &[u64],
        rows: &[u32],
    ) -> Result<Vec<f32>> {
        self.events
            .lock()
            .unwrap()
            .push((1, u16::from_le_bytes(q[..2].try_into().unwrap())));
        self.cpu.score(q, qr, d, dt, p, offsets, rows)
    }
    fn score_pq(
        &mut self,
        id: &str,
        q: &[u8],
        qr: u32,
        dt: u8,
        offsets: &[u64],
        rows: &[u32],
    ) -> Result<Vec<f32>> {
        self.cpu.score_pq(id, q, qr, dt, offsets, rows)
    }
    fn score_batch(
        &mut self,
        q: &[u8],
        qo: &[u32],
        d: u32,
        dt: u8,
        p: u8,
        offsets: &[u64],
        rows: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        self.events
            .lock()
            .unwrap()
            .push((qo.len() - 1, u16::from_le_bytes(q[..2].try_into().unwrap())));
        let results = self.cpu.score_batch(q, qo, d, dt, p, offsets, rows)?;
        // Trigger cancellation after all batch slots exist but before any
        // response is published. Remaining slots must not be compacted.
        drop(self.close_after_batch.lock().unwrap().take());
        Ok(results)
    }
}

struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(
    close_after_batch: Arc<Mutex<Option<UnixStream>>>,
) -> (Engine, RequestFixture, Root, Events) {
    let root = std::env::temp_dir().join(format!(
        "exact-batch-lifecycle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let descriptors = [[0x3c00u16, 0u16], [0, 0x3c00], [0x3c00, 0x3c00]]
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let payload = values
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            let digest = hex::encode(Sha256::digest(&payload));
            let directory = root.join("objects").join(&digest[..2]);
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join(format!("{digest}.tensor")), &payload).unwrap();
            protocol::Descriptor {
                candidate_id: index as u32,
                contract: "model-v1".into(),
                digest,
                rows: 1,
                dimension: 2,
                dtype: 2,
                raw_fp8_cache_key: OnceLock::new(),
            }
        })
        .collect::<Vec<_>>();
    let cpu = CpuBackend::create(0, 65536, 32768).unwrap();
    let mut info = cpu.info().clone();
    info.capabilities.fused_multiquery = true;
    let events = Arc::new(Mutex::new(Vec::new()));
    let backend = FusedCpu {
        cpu,
        info,
        close_after_batch,
        events: Arc::clone(&events),
    };
    let store = ShardStore::open(&[("model-v1".into(), root.clone())], 16384, 100, false).unwrap();
    let mut engine =
        Engine::new(vec![backend], 256, store, 100, 100, &HashMap::new(), None).unwrap();
    engine.prewarm(&descriptors, 3).unwrap();
    (engine, RequestFixture(descriptors), Root(root), events)
}

struct RequestFixture(Vec<protocol::Descriptor>);
impl RequestFixture {
    fn request(&self, tenant: &str, rows: u32, public_base: u32, marker: u16) -> protocol::Request {
        let mut candidates = self.0.clone();
        for descriptor in &mut candidates {
            descriptor.candidate_id += public_base;
        }
        protocol::Request {
            protocol_version: protocol::VERSION_EXTERNAL,
            request_id: 7,
            tenant: tenant.into(),
            model_contract: "model-v1".into(),
            priority: 0,
            timeout_ms: 1000,
            query_rows: rows,
            dimension: 2,
            dtype: 2,
            candidate_dtype: 2,
            scoring_profile: protocol::ScoringProfile::ExactFp16,
            quantization_contract: None,
            top_k: Some(3),
            query: (0..rows)
                .flat_map(|_| [marker, 0u16].into_iter().flat_map(u16::to_le_bytes))
                .collect(),
            candidates: Arc::new(candidates),
            candidate_start: 0,
            candidate_end: 3,
            manifest_digest: None,
            catalog_digest: None,
            catalog_selection_digest: None,
            catalog_public_ids: vec![],
            catalog_registration: false,
            scoped_candidate_ordinals: vec![],
        }
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        policy: SchedulerPolicy::FairPriority,
        priority_aging: Duration::from_secs(60),
        priority_band: 200,
        batch_window: Duration::ZERO,
        quantum_candidates: 3,
        quantum_tokens: u64::MAX,
        quantum_fmas: 8,
        max_microbatch_requests: 8,
        max_ingress_drain: 16,
        prepared_batch_window: true,
        exact_quantum_batching: true,
        cache_log_interval: 64,
        min_shared_candidates_milli: 500,
        socket_io_timeout: Duration::from_millis(100),
        tenant_weights: HashMap::from([("a".into(), 2.0)]),
    }
}

fn work(
    request: protocol::Request,
    admission: &Arc<PendingAdmission>,
    bytes: &Arc<ByteAdmission>,
) -> (Work, UnixStream, mpsc::Receiver<()>) {
    let (local, remote) = UnixStream::pair().unwrap();
    remote
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let (completion, completed) = mpsc::sync_channel(1);
    let now = Instant::now();
    let pending = admission.try_acquire(&request.tenant).unwrap();
    (
        Work {
            request,
            connection: ClientStream::Unix(local),
            accepted_at: now,
            deadline: now + Duration::from_secs(1),
            ready_at: now,
            frame_read_elapsed: Duration::ZERO,
            parse_elapsed: Duration::ZERO,
            next_candidate: 0,
            results: vec![],
            gpu_elapsed: Duration::ZERO,
            completion,
            _pending_permit: pending,
            _frame_permit: bytes.try_acquire(1).unwrap(),
        },
        remote,
        completed,
    )
}

fn execute(
    engine: Engine,
    works: Vec<Work>,
    config: SchedulerConfig,
    metrics: Arc<RuntimeMetrics>,
) {
    let (sender, receiver) = mpsc::sync_channel(16);
    for work in works {
        sender.send(work).unwrap();
    }
    drop(sender);
    let (_admin, admin_receiver) = mpsc::sync_channel(1);
    run_scheduler(
        engine,
        receiver,
        config,
        Arc::new(AtomicBool::new(false)),
        metrics,
        admin_receiver,
        Arc::new(Mutex::new(OperationRegistry::default())),
    )
    .unwrap();
}

fn response(mut stream: UnixStream) -> Vec<(u32, f32)> {
    let mut header = [0u8; 24];
    stream.read_exact(&mut header).unwrap();
    assert_eq!(u64::from_le_bytes(header[8..16].try_into().unwrap()), 7);
    let mut body = vec![0; u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize];
    stream.read_exact(&mut body).unwrap();
    assert_eq!(u32::from_le_bytes(body[..4].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(body[4..8].try_into().unwrap()), 3);
    body[8..]
        .chunks_exact(8)
        .map(|r| {
            (
                u32::from_le_bytes(r[..4].try_into().unwrap()),
                f32::from_le_bytes(r[4..].try_into().unwrap()),
            )
        })
        .collect()
}

#[test]
fn canceling_middle_fused_slot_preserves_other_connection_ids_and_releases_permits() {
    let cancel = Arc::new(Mutex::new(None));
    let (engine, requests, _root, events) = fixture(Arc::clone(&cancel));
    let metrics = Arc::new(RuntimeMetrics::default());
    let pending = Arc::new(PendingAdmission::new(4, 4, Arc::clone(&metrics)));
    let bytes = Arc::new(ByteAdmission::new(8, Arc::clone(&metrics)));
    let (a, client_a, completed_a) = work(requests.request("a", 1, 10, 0x3c00), &pending, &bytes);
    let (b, client_b, completed_b) = work(requests.request("a", 2, 20, 0x3c00), &pending, &bytes);
    let (c, client_c, completed_c) = work(requests.request("a", 3, 30, 0x3c00), &pending, &bytes);
    *cancel.lock().unwrap() = Some(client_b);
    execute(engine, vec![a, b, c], config(), Arc::clone(&metrics));
    assert_eq!(response(client_a), vec![(10, 1.0), (12, 1.0), (11, 0.0)]);
    assert_eq!(response(client_c), vec![(30, 3.0), (32, 3.0), (31, 0.0)]);
    for receiver in [completed_a, completed_b, completed_c] {
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    assert_eq!(metrics.completed.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.disconnected.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.pending_requests.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight_request_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(events.lock().unwrap()[0].0, 3);
}

#[test]
fn another_tenant_policy_winner_prevents_compatible_tail_from_bypassing_it() {
    let (engine, requests, _root, events) = fixture(Arc::new(Mutex::new(None)));
    let metrics = Arc::new(RuntimeMetrics::default());
    let pending = Arc::new(PendingAdmission::new(4, 4, Arc::clone(&metrics)));
    let bytes = Arc::new(ByteAdmission::new(8, Arc::clone(&metrics)));
    let mut cfg = config();
    cfg.quantum_fmas = u64::MAX;
    let (a, ca, ra) = work(requests.request("a", 1, 10, 0x3c00), &pending, &bytes);
    let (a_tail, ct, rt) = work(requests.request("a", 1, 20, 0x3800), &pending, &bytes);
    let (b, cb, rb) = work(requests.request("b", 1, 30, 0x3400), &pending, &bytes);
    execute(engine, vec![a, a_tail, b], cfg, Arc::clone(&metrics));
    response(ca);
    response(ct);
    response(cb);
    for receiver in [ra, rt, rb] {
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    assert_eq!(
        *events.lock().unwrap(),
        vec![(1, 0x3c00), (1, 0x3400), (1, 0x3800)]
    );
    assert_eq!(metrics.gpu_fused_requests.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.exact_batch_pairs.load(Ordering::Relaxed), 0);
    assert!(metrics.exact_batch_blocked_next.load(Ordering::Relaxed) > 0);
    assert_eq!(metrics.pending_requests.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight_request_bytes.load(Ordering::Relaxed), 0);
}

#[test]
fn queued_expiry_and_reload_keep_surviving_slots_and_release_every_permit() {
    let (engine, requests, _root, _events) = fixture(Arc::new(Mutex::new(None)));
    let metrics = Arc::new(RuntimeMetrics::default());
    let pending = Arc::new(PendingAdmission::new(4, 4, Arc::clone(&metrics)));
    let bytes = Arc::new(ByteAdmission::new(8, Arc::clone(&metrics)));
    let (mut expired, mut ce, re) = work(requests.request("a", 1, 30, 0x3c00), &pending, &bytes);
    expired.deadline = Instant::now();
    let (a, ca, ra) = work(requests.request("a", 1, 10, 0x3c00), &pending, &bytes);
    let (b, cb, rb) = work(requests.request("a", 1, 20, 0x3c00), &pending, &bytes);
    let (sender, receiver) = mpsc::sync_channel(4);
    for work in [expired, a, b] {
        sender.send(work).unwrap();
    }
    drop(sender);
    let (_admin, admin_receiver) = mpsc::sync_channel(1);
    super::run_scheduler(
        engine,
        receiver,
        config(),
        Arc::new(AtomicBool::new(true)),
        Arc::clone(&metrics),
        admin_receiver,
        Arc::new(Mutex::new(OperationRegistry::default())),
    )
    .unwrap();
    assert_eq!(response(ca), vec![(10, 1.0), (12, 1.0), (11, 0.0)]);
    assert_eq!(response(cb), vec![(20, 1.0), (22, 1.0), (21, 0.0)]);
    let mut header = [0u8; 24];
    ce.read_exact(&mut header).unwrap();
    assert_eq!(u64::from_le_bytes(header[8..16].try_into().unwrap()), 7);
    let mut body = vec![0; u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize];
    ce.read_exact(&mut body).unwrap();
    assert_eq!(u32::from_le_bytes(body[..4].try_into().unwrap()), 2);
    for receiver in [re, ra, rb] {
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
    }
    assert_eq!(metrics.reload_succeeded.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.timeout_in_queue.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.completed.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.pending_requests.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight_request_bytes.load(Ordering::Relaxed), 0);
}
