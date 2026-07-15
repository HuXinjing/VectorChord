// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

mod cache;
mod engine;
mod gpu;
mod protocol;
mod scheduler;
mod shard;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use engine::{Engine, EngineStatus};
use gpu::Gpu;
use protocol::{HEADER_BYTES, VERSION_EXTERNAL, VERSION_SCHEDULED_EXTERNAL};
use scheduler::{RequestQueue, Scheduled, SchedulerPolicy};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use shard::ShardStore;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

const GIB: usize = 1024 * 1024 * 1024;
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
static RELOAD_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(signal: libc::c_int) {
    if signal == libc::SIGHUP {
        RELOAD_REQUESTED.store(true, Ordering::Relaxed);
    } else {
        SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
    }
}

fn install_signal_handlers() -> Result<()> {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        if unsafe { libc::signal(signal, handle_signal as *const () as libc::sighandler_t) }
            == libc::SIG_ERR
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

#[derive(Parser)]
#[command(about = "Native Rust/CUDA TileMaxSim shard, cache, and scheduling daemon")]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long, required = true, value_parser = parse_gpu_memory)]
    gpu_memory_gb: Vec<GpuMemory>,
    #[arg(long, default_value = "2", value_parser = parse_gb)]
    gpu_workspace_gb: usize,
    #[arg(long, default_value = "8", value_parser = parse_gb)]
    host_cache_gb: usize,
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u8).range(1..=100))]
    host_tenant_cache_max_percent: u8,
    #[arg(long = "contract-root", required = true, value_parser = parse_contract_root)]
    contract_roots: Vec<(String, PathBuf)>,
    #[arg(long, default_value_t = 32)]
    gpu_block_kib: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_request_bytes: usize,
    #[arg(long, default_value = "1", value_parser = parse_gb)]
    max_inflight_request_gb: usize,
    #[arg(long, default_value = "600", value_parser = parse_mode)]
    socket_mode: u32,
    #[arg(long)]
    status_socket: Option<PathBuf>,
    #[arg(long, default_value = "600", value_parser = parse_mode)]
    status_socket_mode: u32,
    #[arg(long)]
    once: bool,
    #[arg(long)]
    verify_full_shards: bool,
    #[arg(long, default_value = "lru", value_parser = ["lru", "resident"])]
    gpu_cache_mode: String,
    #[arg(long = "resident-manifest", value_parser = parse_contract_root)]
    resident_manifests: Vec<(String, PathBuf)>,
    #[arg(long, default_value_t = 256)]
    prewarm_batch_size: usize,
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u8).range(1..=100))]
    tenant_cache_max_percent: u8,
    #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u8).range(0..=100))]
    pinned_cache_max_percent: u8,
    #[arg(long = "tenant-cache-reservation", value_parser = parse_tenant_memory)]
    tenant_cache_reservations: Vec<(String, usize)>,
    #[arg(long, default_value_t = 256)]
    max_connections: usize,
    #[arg(long, default_value_t = 128)]
    max_queued_requests: usize,
    #[arg(long, default_value_t = 16)]
    max_tenant_queued_requests: usize,
    #[arg(long, default_value_t = 5000)]
    socket_io_timeout_ms: u64,
    #[arg(long, default_value_t = 8000)]
    request_timeout_ms: u64,
    #[arg(long, default_value = "fair-priority", value_parser = parse_scheduler_policy)]
    scheduler_policy: SchedulerPolicy,
    #[arg(long, default_value_t = 1000)]
    priority_aging_ms: u64,
    #[arg(long, default_value_t = 200)]
    priority_band: i32,
    #[arg(long, default_value_t = 2)]
    scheduler_batch_window_ms: u64,
    #[arg(long, default_value_t = 1024)]
    scheduler_quantum_candidates: usize,
    #[arg(long, default_value_t = 250_000)]
    scheduler_quantum_tokens: u64,
    #[arg(long, default_value_t = 4_000_000_000)]
    scheduler_quantum_fmas: u64,
    #[arg(long = "tenant-weight", value_parser = parse_tenant_weight)]
    tenant_weights: Vec<(String, f64)>,
}

#[derive(Clone)]
struct GpuMemory {
    device: i32,
    bytes: usize,
}

fn parse_gpu_memory(value: &str) -> Result<GpuMemory, String> {
    let (device, gb) = value
        .strip_prefix("cuda:")
        .unwrap_or(value)
        .split_once('=')
        .ok_or_else(|| "GPU memory must be GPU=GB, for example 1=20".to_owned())?;
    let device = device
        .parse::<i32>()
        .map_err(|_| "GPU index must be a nonnegative integer".to_owned())?;
    if device < 0 {
        return Err("GPU index must be nonnegative".to_owned());
    }
    let bytes = parse_gb(gb)?;
    Ok(GpuMemory { device, bytes })
}

fn parse_gb(value: &str) -> Result<usize, String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.')
    {
        return Err("memory size must be a positive number of GB".to_owned());
    }
    let gb = value
        .parse::<f64>()
        .map_err(|_| "memory size must be a positive number of GB".to_owned())?;
    if !gb.is_finite() || gb <= 0.0 || gb * GIB as f64 > usize::MAX as f64 {
        return Err("memory size must be a positive number of GB".to_owned());
    }
    Ok((gb * GIB as f64) as usize)
}

fn kib_to_bytes(kib: usize) -> Result<usize, &'static str> {
    kib.checked_mul(1024).ok_or("GPU block size overflow")
}

fn parse_contract_root(value: &str) -> Result<(String, PathBuf), String> {
    let (contract, path) = value
        .split_once('=')
        .ok_or_else(|| "contract root must be MODEL_CONTRACT_ID=/absolute/path".to_owned())?;
    let path = PathBuf::from(path);
    if contract.is_empty() || !path.is_absolute() {
        return Err("contract root must contain a nonempty ID and absolute path".to_owned());
    }
    Ok((contract.to_owned(), path))
}

fn parse_mode(value: &str) -> Result<u32, String> {
    let mode = u32::from_str_radix(value, 8).map_err(|_| "invalid octal socket mode".to_owned())?;
    if mode > 0o777 {
        return Err("socket mode must be between 000 and 777".to_owned());
    }
    Ok(mode)
}

fn parse_scheduler_policy(value: &str) -> Result<SchedulerPolicy, String> {
    SchedulerPolicy::parse(value)
}

fn parse_tenant_weight(value: &str) -> Result<(String, f64), String> {
    let (tenant, weight) = value
        .split_once('=')
        .ok_or_else(|| "tenant weight must be TENANT=WEIGHT".to_owned())?;
    if tenant.is_empty()
        || tenant.len() > 256
        || tenant.chars().any(|character| character.is_control())
    {
        return Err("tenant weight has an invalid tenant name".to_owned());
    }
    let weight = weight
        .parse::<f64>()
        .map_err(|_| "tenant weight must be a finite positive number".to_owned())?;
    if !weight.is_finite() || weight <= 0.0 || weight > 1000.0 {
        return Err("tenant weight must be between 0 and 1000".to_owned());
    }
    Ok((tenant.to_owned(), weight))
}

fn parse_tenant_memory(value: &str) -> Result<(String, usize), String> {
    let (tenant, gb) = value
        .split_once('=')
        .ok_or_else(|| "tenant cache reservation must be TENANT=GB".to_owned())?;
    if tenant.is_empty()
        || tenant.len() > 256
        || tenant.chars().any(|character| character.is_control())
    {
        return Err("tenant cache reservation has an invalid tenant name".to_owned());
    }
    Ok((tenant.to_owned(), parse_gb(gb)?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.max_connections == 0
        || args.max_queued_requests == 0
        || args.max_tenant_queued_requests == 0
        || args.socket_io_timeout_ms == 0
        || args.request_timeout_ms == 0
        || args.priority_aging_ms == 0
        || args.scheduler_quantum_candidates == 0
        || args.scheduler_quantum_tokens == 0
        || args.scheduler_quantum_fmas == 0
    {
        bail!("connection, queue, timeout, and priority-aging limits must be positive");
    }
    let mut seen_devices = std::collections::HashSet::new();
    for specification in &args.gpu_memory_gb {
        if args.gpu_workspace_gb >= specification.bytes {
            bail!("every configured GPU allocation must exceed its workspace");
        }
        if !seen_devices.insert(specification.device) {
            bail!("each CUDA device may be configured only once");
        }
    }
    let block_bytes = kib_to_bytes(args.gpu_block_kib).map_err(|message| anyhow!(message))?;
    if block_bytes == 0 || block_bytes % 256 != 0 {
        bail!("GPU block size must be positive and 256-byte aligned");
    }
    let store = ShardStore::open(
        &args.contract_roots,
        args.host_cache_gb,
        args.host_tenant_cache_max_percent,
        args.verify_full_shards,
    )?;
    let gpus = args
        .gpu_memory_gb
        .iter()
        .map(|specification| {
            Gpu::create(
                specification.device,
                specification.bytes,
                args.gpu_workspace_gb,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let tenant_cache_reservations = args.tenant_cache_reservations.iter().cloned().collect();
    let mut engine = Engine::new(
        gpus,
        block_bytes,
        store,
        args.tenant_cache_max_percent,
        args.pinned_cache_max_percent,
        &tenant_cache_reservations,
    )?;
    if args.gpu_cache_mode == "resident" && args.resident_manifests.is_empty() {
        bail!("resident GPU cache mode requires at least one resident manifest");
    }
    if args.gpu_cache_mode == "lru" && !args.resident_manifests.is_empty() {
        bail!("resident manifests are valid only in resident GPU cache mode");
    }
    if args.gpu_cache_mode == "resident" {
        let prewarm_started = Instant::now();
        let descriptors = load_resident_manifests(&args.resident_manifests)?;
        engine.prewarm(&descriptors, args.prewarm_batch_size)?;
        println!(
            "{}",
            serde_json::json!({
                "event": "tilemaxsim_rust_prewarm_complete",
                "entries": descriptors.len(),
                "elapsed_ms": prewarm_started.elapsed().as_secs_f64() * 1000.0,
                "cache": engine.status_json(),
            })
        );
    }
    let _instance_lock = acquire_instance_lock(&args.socket)?;
    remove_stale_socket(&args.socket)?;
    let listener = UnixListener::bind(&args.socket)
        .with_context(|| format!("cannot bind {}", args.socket.display()))?;
    fs::set_permissions(&args.socket, fs::Permissions::from_mode(args.socket_mode))?;
    listener.set_nonblocking(true)?;
    if args.status_socket.as_ref() == Some(&args.socket) {
        bail!("status socket must differ from the TileMaxSim protocol socket");
    }
    let reload = Arc::new(AtomicBool::new(false));
    let metrics = Arc::new(RuntimeMetrics::new(
        args.max_connections,
        args.max_queued_requests,
        args.max_tenant_queued_requests,
        args.max_inflight_request_gb,
    ));
    metrics.update_engine(engine.status_snapshot());
    install_signal_handlers()?;
    let ready_cache = engine.status_json();

    let (sender, receiver) = mpsc::sync_channel::<Work>(args.max_queued_requests);
    let frame_admission = Arc::new(ByteAdmission::new(
        args.max_inflight_request_gb,
        Arc::clone(&metrics),
    ));
    let pending_admission = Arc::new(PendingAdmission::new(
        args.max_queued_requests,
        args.max_tenant_queued_requests,
        Arc::clone(&metrics),
    ));
    let tenant_weights = args.tenant_weights.iter().cloned().collect();
    let scheduler_config = SchedulerConfig {
        policy: args.scheduler_policy,
        priority_aging: Duration::from_millis(args.priority_aging_ms),
        priority_band: args.priority_band,
        batch_window: Duration::from_millis(args.scheduler_batch_window_ms),
        quantum_candidates: args.scheduler_quantum_candidates,
        quantum_tokens: args.scheduler_quantum_tokens,
        quantum_fmas: args.scheduler_quantum_fmas,
        socket_io_timeout: Duration::from_millis(args.socket_io_timeout_ms),
        tenant_weights,
    };
    let scheduler_reload = Arc::clone(&reload);
    let scheduler_metrics = Arc::clone(&metrics);
    let scheduler = thread::Builder::new()
        .name("tilemaxsim-scheduler".to_owned())
        .spawn(move || {
            run_scheduler(
                engine,
                receiver,
                scheduler_config,
                scheduler_reload,
                scheduler_metrics,
            )
        })?;
    let status_server = if let Some(path) = args.status_socket.clone() {
        remove_stale_socket(&path)?;
        let status_listener = UnixListener::bind(&path)
            .with_context(|| format!("cannot bind status socket {}", path.display()))?;
        fs::set_permissions(&path, fs::Permissions::from_mode(args.status_socket_mode))?;
        status_listener.set_nonblocking(true)?;
        let status_metrics = Arc::clone(&metrics);
        Some(
            thread::Builder::new()
                .name("tilemaxsim-status".to_owned())
                .spawn(move || run_status_server(status_listener, path, status_metrics))?,
        )
    } else {
        None
    };
    metrics.ready.store(true, Ordering::Release);
    println!(
        "{}",
        serde_json::json!({
            "event": "tilemaxsim_rust_ready",
            "socket": args.socket,
            "devices": args.gpu_memory_gb.iter().map(|item| serde_json::json!({
                "device": item.device,
                "allocated_bytes": item.bytes,
            })).collect::<Vec<_>>(),
            "workspace_bytes": args.gpu_workspace_gb,
            "scheduler_policy": format!("{:?}", args.scheduler_policy),
            "max_connections": args.max_connections,
            "max_queued_requests": args.max_queued_requests,
            "max_tenant_queued_requests": args.max_tenant_queued_requests,
            "max_inflight_request_bytes": args.max_inflight_request_gb,
            "scheduler_quantum_fmas": args.scheduler_quantum_fmas,
            "status_socket": args.status_socket,
            "cache": ready_cache,
        })
    );

    let mut readers = Vec::new();
    let mut accepted = 0_usize;
    let mut status_failed = false;
    let mut scheduler_failed = false;
    let mut fatal_error = None;
    while !SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
        if scheduler.is_finished() {
            scheduler_failed = true;
            metrics.ready.store(false, Ordering::Release);
            SHUTDOWN_REQUESTED.store(true, Ordering::Release);
            break;
        }
        if status_server
            .as_ref()
            .is_some_and(thread::JoinHandle::is_finished)
        {
            status_failed = true;
            metrics.ready.store(false, Ordering::Release);
            SHUTDOWN_REQUESTED.store(true, Ordering::Release);
            break;
        }
        if RELOAD_REQUESTED.swap(false, Ordering::AcqRel) {
            reload.store(true, Ordering::Release);
        }
        reap_readers(&mut readers);
        match listener.accept() {
            Ok((connection, _)) => {
                if !try_acquire_reader(&metrics.active_connections, args.max_connections) {
                    // Closing immediately is deliberate: we have not read enough
                    // bytes to know whether the peer expects a v2 or v3 response.
                    metrics.rejected_connections.fetch_add(1, Ordering::Relaxed);
                    drop(connection);
                    continue;
                }
                accepted += 1;
                let reader_sender = sender.clone();
                let reader_metrics = Arc::clone(&metrics);
                let reader_admission = Arc::clone(&pending_admission);
                let request_metrics = Arc::clone(&metrics);
                let reader_frame_admission = Arc::clone(&frame_admission);
                let reader_config = ReaderConfig {
                    maximum: args.max_request_bytes,
                    io_timeout: Duration::from_millis(args.socket_io_timeout_ms),
                    server_timeout: Duration::from_millis(args.request_timeout_ms),
                    maximum_candidate_fmas: args.scheduler_quantum_fmas,
                };
                match thread::Builder::new()
                    .name("tilemaxsim-reader".to_owned())
                    .spawn(move || {
                        let _permit = ReaderPermit(reader_metrics);
                        read_and_enqueue(
                            connection,
                            &reader_sender,
                            reader_config,
                            reader_admission,
                            request_metrics,
                            reader_frame_admission,
                        );
                    }) {
                    Ok(reader) => readers.push(reader),
                    Err(error) => {
                        metrics.active_connections.fetch_sub(1, Ordering::Release);
                        metrics.ready.store(false, Ordering::Release);
                        SHUTDOWN_REQUESTED.store(true, Ordering::Release);
                        fatal_error = Some(error.into());
                        break;
                    }
                }
                if args.once && accepted == 1 {
                    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                metrics.ready.store(false, Ordering::Release);
                SHUTDOWN_REQUESTED.store(true, Ordering::Release);
                fatal_error = Some(error.into());
                break;
            }
        }
    }
    SHUTDOWN_REQUESTED.store(true, Ordering::Release);
    drop(listener);
    for reader in readers {
        if reader.join().is_err() {
            eprintln!("tilemaxsim reader thread panicked during shutdown");
        }
    }
    drop(sender);
    metrics.ready.store(false, Ordering::Release);
    let scheduler_result = scheduler
        .join()
        .map_err(|_| anyhow!("TileMaxSim scheduler thread panicked"))
        .and_then(|result| result);
    if status_server.is_some_and(|status_server| status_server.join().is_err()) {
        eprintln!("TileMaxSim status thread panicked during shutdown");
    }
    match fs::remove_file(&args.socket) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    if status_failed {
        bail!("TileMaxSim status server exited unexpectedly");
    }
    if scheduler_failed {
        scheduler_result?;
        bail!("TileMaxSim scheduler exited unexpectedly");
    }
    scheduler_result?;
    if let Some(error) = fatal_error {
        return Err(error);
    }
    Ok(())
}

struct Work {
    request: protocol::Request,
    connection: UnixStream,
    accepted_at: Instant,
    deadline: Instant,
    next_candidate: usize,
    results: Vec<(u32, f32)>,
    gpu_elapsed: Duration,
    _pending_permit: PendingPermit,
    _frame_permit: BytePermit,
}

struct SchedulerConfig {
    policy: SchedulerPolicy,
    priority_aging: Duration,
    priority_band: i32,
    batch_window: Duration,
    quantum_candidates: usize,
    quantum_tokens: u64,
    quantum_fmas: u64,
    socket_io_timeout: Duration,
    tenant_weights: std::collections::HashMap<String, f64>,
}

#[derive(Default)]
struct RuntimeMetrics {
    ready: AtomicBool,
    max_connections: usize,
    max_pending_requests: usize,
    max_tenant_pending_requests: usize,
    max_inflight_request_bytes: usize,
    active_connections: AtomicUsize,
    inflight_request_bytes: AtomicUsize,
    pending_requests: AtomicUsize,
    pending_tenants: AtomicUsize,
    scheduler_depth: AtomicUsize,
    scheduler_depth_high_water: AtomicUsize,
    gpu_active: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
    timed_out: AtomicU64,
    disconnected: AtomicU64,
    rejected_connections: AtomicU64,
    rejected_frame_bytes: AtomicU64,
    rejected_queue_global: AtomicU64,
    rejected_tenant: AtomicU64,
    frame_read_failures: AtomicU64,
    invalid_requests: AtomicU64,
    gpu_failures: AtomicU64,
    scheduler_failures: AtomicU64,
    reload_succeeded: AtomicU64,
    reload_failed: AtomicU64,
    timeout_before_enqueue: AtomicU64,
    timeout_in_queue: AtomicU64,
    timeout_before_execution: AtomicU64,
    timeout_during_execution: AtomicU64,
    gpu_quantums: AtomicU64,
    scheduler_requeues: AtomicU64,
    admitted_priority_negative: AtomicU64,
    admitted_priority_zero: AtomicU64,
    admitted_priority_positive: AtomicU64,
    candidates_scored: AtomicU64,
    document_rows_scored: AtomicU64,
    latency_observations: AtomicU64,
    total_latency_us: AtomicU64,
    gpu_latency_us: AtomicU64,
    engine: Mutex<EngineStatus>,
}

impl RuntimeMetrics {
    fn new(
        max_connections: usize,
        max_pending_requests: usize,
        max_tenant_pending_requests: usize,
        max_inflight_request_bytes: usize,
    ) -> Self {
        Self {
            max_connections,
            max_pending_requests,
            max_tenant_pending_requests,
            max_inflight_request_bytes,
            ..Self::default()
        }
    }

    fn update_engine(&self, status: EngineStatus) {
        *self
            .engine
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = status;
    }

    fn update_scheduler_depth(&self, depth: usize) {
        self.scheduler_depth.store(depth, Ordering::Relaxed);
        self.scheduler_depth_high_water
            .fetch_max(depth, Ordering::Relaxed);
    }

    fn observe_latency(&self, total: Duration, gpu: Duration) {
        self.latency_observations.fetch_add(1, Ordering::Relaxed);
        saturating_atomic_add(&self.total_latency_us, duration_micros(total));
        saturating_atomic_add(&self.gpu_latency_us, duration_micros(gpu));
    }
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn saturating_atomic_add(counter: &AtomicU64, value: u64) {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(value))
        })
        .ok();
}

struct ReaderPermit(Arc<RuntimeMetrics>);

impl Drop for ReaderPermit {
    fn drop(&mut self) {
        self.0.active_connections.fetch_sub(1, Ordering::Release);
    }
}

struct PendingState {
    total: usize,
    tenants: HashMap<String, usize>,
}

struct PendingAdmission {
    state: Mutex<PendingState>,
    max_total: usize,
    max_tenant: usize,
    metrics: Arc<RuntimeMetrics>,
}

struct ByteAdmission {
    used: AtomicUsize,
    maximum: usize,
    metrics: Arc<RuntimeMetrics>,
}

struct BytePermit {
    admission: Arc<ByteAdmission>,
    bytes: usize,
}

impl ByteAdmission {
    fn new(maximum: usize, metrics: Arc<RuntimeMetrics>) -> Self {
        Self {
            used: AtomicUsize::new(0),
            maximum,
            metrics,
        }
    }

    fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<BytePermit> {
        let result = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|next| *next <= self.maximum)
            })
            .ok();
        if result.is_none() {
            self.metrics
                .rejected_frame_bytes
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.metrics
            .inflight_request_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        Some(BytePermit {
            admission: Arc::clone(self),
            bytes,
        })
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        self.admission.used.fetch_sub(self.bytes, Ordering::Release);
        self.admission
            .metrics
            .inflight_request_bytes
            .fetch_sub(self.bytes, Ordering::Release);
    }
}

#[derive(Debug)]
enum AdmissionRejection {
    Global,
    Tenant,
}

struct PendingPermit {
    admission: Arc<PendingAdmission>,
    tenant: String,
}

impl PendingAdmission {
    fn new(max_total: usize, max_tenant: usize, metrics: Arc<RuntimeMetrics>) -> Self {
        Self {
            state: Mutex::new(PendingState {
                total: 0,
                tenants: HashMap::new(),
            }),
            max_total,
            max_tenant,
            metrics,
        }
    }

    fn try_acquire(self: &Arc<Self>, tenant: &str) -> Result<PendingPermit, AdmissionRejection> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.total >= self.max_total {
            return Err(AdmissionRejection::Global);
        }
        if state.tenants.get(tenant).copied().unwrap_or(0) >= self.max_tenant {
            return Err(AdmissionRejection::Tenant);
        }
        state.total += 1;
        *state.tenants.entry(tenant.to_owned()).or_default() += 1;
        self.metrics
            .pending_requests
            .store(state.total, Ordering::Relaxed);
        self.metrics
            .pending_tenants
            .store(state.tenants.len(), Ordering::Relaxed);
        Ok(PendingPermit {
            admission: Arc::clone(self),
            tenant: tenant.to_owned(),
        })
    }
}

impl Drop for PendingPermit {
    fn drop(&mut self) {
        let mut state = self
            .admission
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.total = state.total.saturating_sub(1);
        if let Some(count) = state.tenants.get_mut(&self.tenant) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.tenants.remove(&self.tenant);
            }
        }
        self.admission
            .metrics
            .pending_requests
            .store(state.total, Ordering::Relaxed);
        self.admission
            .metrics
            .pending_tenants
            .store(state.tenants.len(), Ordering::Relaxed);
    }
}

fn try_acquire_reader(counter: &AtomicUsize, maximum: usize) -> bool {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            (current < maximum).then_some(current + 1)
        })
        .is_ok()
}

fn reap_readers(readers: &mut Vec<thread::JoinHandle<()>>) {
    let mut index = 0;
    while index < readers.len() {
        if readers[index].is_finished() {
            let reader = readers.swap_remove(index);
            if reader.join().is_err() {
                eprintln!("tilemaxsim reader thread panicked");
            }
        } else {
            index += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct ReaderConfig {
    maximum: usize,
    io_timeout: Duration,
    server_timeout: Duration,
    maximum_candidate_fmas: u64,
}

fn read_and_enqueue(
    mut connection: UnixStream,
    sender: &mpsc::SyncSender<Work>,
    config: ReaderConfig,
    pending_admission: Arc<PendingAdmission>,
    metrics: Arc<RuntimeMetrics>,
    frame_admission: Arc<ByteAdmission>,
) {
    let accepted_at = Instant::now();
    if let Err(error) = connection.set_read_timeout(Some(config.io_timeout)) {
        eprintln!("cannot configure TileMaxSim socket read timeout: {error}");
        return;
    }
    if let Err(error) = connection.set_write_timeout(Some(config.io_timeout)) {
        eprintln!("cannot configure TileMaxSim socket write timeout: {error}");
        return;
    }
    let (frame, frame_permit) =
        match read_request(&mut connection, config.maximum, &frame_admission) {
            Ok(frame) => frame,
            Err(error) => {
                metrics.failed.fetch_add(1, Ordering::Relaxed);
                metrics.frame_read_failures.fetch_add(1, Ordering::Relaxed);
                write_response_nonfatal(
                    &mut connection,
                    &protocol::failure(VERSION_EXTERNAL, 0, 1, &format!("{error:#}")),
                );
                return;
            }
        };
    let version = header_version(&frame);
    let request_id = header_request_id(&frame);
    let request = match protocol::parse(&frame) {
        Ok(request) => request,
        Err(error) => {
            metrics.failed.fetch_add(1, Ordering::Relaxed);
            metrics.invalid_requests.fetch_add(1, Ordering::Relaxed);
            write_response_nonfatal(
                &mut connection,
                &protocol::failure(version, request_id, 1, &format!("{error:#}")),
            );
            return;
        }
    };
    if request.candidates.iter().any(|candidate| {
        candidate_fmas(request.query_rows, request.dimension, candidate.rows)
            > config.maximum_candidate_fmas
    }) {
        metrics.failed.fetch_add(1, Ordering::Relaxed);
        metrics.invalid_requests.fetch_add(1, Ordering::Relaxed);
        write_response_nonfatal(
            &mut connection,
            &protocol::failure(
                version,
                request_id,
                1,
                "one candidate exceeds the configured CUDA kernel work limit",
            ),
        );
        return;
    }
    let client_timeout = if request.timeout_ms == 0 {
        config.server_timeout
    } else {
        Duration::from_millis(u64::from(request.timeout_ms)).min(config.server_timeout)
    };
    let deadline = accepted_at
        .checked_add(client_timeout)
        .unwrap_or(accepted_at);
    if deadline <= Instant::now() {
        metrics.timed_out.fetch_add(1, Ordering::Relaxed);
        metrics
            .timeout_before_enqueue
            .fetch_add(1, Ordering::Relaxed);
        write_response_nonfatal(
            &mut connection,
            &protocol::failure(
                version,
                request_id,
                2,
                "request deadline expired before enqueue",
            ),
        );
        return;
    }
    let pending_permit = match pending_admission.try_acquire(&request.tenant) {
        Ok(permit) => permit,
        Err(AdmissionRejection::Global) => {
            metrics
                .rejected_queue_global
                .fetch_add(1, Ordering::Relaxed);
            write_response_nonfatal(
                &mut connection,
                &protocol::failure(version, request_id, 2, "TileMaxSim scheduler queue is full"),
            );
            return;
        }
        Err(AdmissionRejection::Tenant) => {
            metrics.rejected_tenant.fetch_add(1, Ordering::Relaxed);
            write_response_nonfatal(
                &mut connection,
                &protocol::failure(
                    version,
                    request_id,
                    2,
                    "tenant TileMaxSim queue limit exceeded",
                ),
            );
            return;
        }
    };
    match sender.try_send(Work {
        request,
        connection,
        accepted_at,
        deadline,
        next_candidate: 0,
        results: Vec::new(),
        gpu_elapsed: Duration::ZERO,
        _pending_permit: pending_permit,
        _frame_permit: frame_permit,
    }) {
        Ok(()) => {}
        Err(mpsc::TrySendError::Full(mut work)) => {
            metrics
                .rejected_queue_global
                .fetch_add(1, Ordering::Relaxed);
            write_response_nonfatal(
                &mut work.connection,
                &protocol::failure(
                    work.request.protocol_version,
                    work.request.request_id,
                    2,
                    "TileMaxSim scheduler queue is full",
                ),
            );
        }
        Err(mpsc::TrySendError::Disconnected(mut work)) => {
            metrics.failed.fetch_add(1, Ordering::Relaxed);
            metrics.scheduler_failures.fetch_add(1, Ordering::Relaxed);
            write_response_nonfatal(
                &mut work.connection,
                &protocol::failure(
                    work.request.protocol_version,
                    work.request.request_id,
                    3,
                    "TileMaxSim scheduler is unavailable",
                ),
            );
        }
    }
}

fn run_scheduler(
    mut engine: Engine,
    receiver: mpsc::Receiver<Work>,
    config: SchedulerConfig,
    reload: Arc<AtomicBool>,
    metrics: Arc<RuntimeMetrics>,
) -> Result<()> {
    let mut queue = RequestQueue::new(
        config.policy,
        config.priority_aging,
        config.priority_band,
        config.tenant_weights.clone(),
    );
    let mut channel_open = true;
    while channel_open || queue.len() > 0 {
        if reload.swap(false, Ordering::AcqRel) {
            match engine.reload_shards() {
                Ok(()) => {
                    metrics.reload_succeeded.fetch_add(1, Ordering::Relaxed);
                    metrics.update_engine(engine.status_snapshot());
                    println!(
                        "{}",
                        serde_json::json!({"event": "tilemaxsim_rust_shards_reloaded"})
                    );
                }
                Err(error) => {
                    metrics.reload_failed.fetch_add(1, Ordering::Relaxed);
                    eprintln!("TileMaxSim shard reload rejected: {error:#}");
                }
            }
        }

        if queue.len() == 0 && channel_open {
            match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(work) => enqueue_work(&mut queue, work, &config, &metrics),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => channel_open = false,
            }
        }
        if channel_open {
            let batching_deadline = Instant::now() + config.batch_window;
            loop {
                let remaining = batching_deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match receiver.recv_timeout(remaining) {
                    Ok(work) => enqueue_work(&mut queue, work, &config, &metrics),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        channel_open = false;
                        break;
                    }
                }
            }
        }

        let now = Instant::now();
        for mut expired in queue.drain_expired(now) {
            metrics.timed_out.fetch_add(1, Ordering::Relaxed);
            metrics.timeout_in_queue.fetch_add(1, Ordering::Relaxed);
            metrics.observe_latency(expired.payload.accepted_at.elapsed(), Duration::ZERO);
            write_response_nonfatal(
                &mut expired.payload.connection,
                &protocol::failure(
                    expired.payload.request.protocol_version,
                    expired.payload.request.request_id,
                    2,
                    "request deadline expired in scheduler queue",
                ),
            );
        }
        metrics.update_scheduler_depth(queue.len());
        let Some(scheduled) = queue.pop(Instant::now()) else {
            continue;
        };
        metrics.update_scheduler_depth(queue.len());
        if peer_disconnected(&scheduled.payload.connection) {
            metrics.disconnected.fetch_add(1, Ordering::Relaxed);
            metrics.observe_latency(scheduled.payload.accepted_at.elapsed(), Duration::ZERO);
            continue;
        }
        let started = Instant::now();
        let mut work = scheduled.payload;
        let request_id = work.request.request_id;
        let version = work.request.protocol_version;
        let tenant = work.request.tenant.clone();
        let priority = work.request.priority;
        let response = if work.deadline <= started {
            metrics.timed_out.fetch_add(1, Ordering::Relaxed);
            metrics
                .timeout_before_execution
                .fetch_add(1, Ordering::Relaxed);
            Some(protocol::failure(
                version,
                request_id,
                2,
                "request deadline expired before execution",
            ))
        } else {
            let end = next_quantum_end(&work, &config);
            let quantum = protocol::Request {
                protocol_version: work.request.protocol_version,
                request_id: work.request.request_id,
                tenant: work.request.tenant.clone(),
                priority: work.request.priority,
                timeout_ms: work.request.timeout_ms,
                query_rows: work.request.query_rows,
                dimension: work.request.dimension,
                dtype: work.request.dtype,
                query: work.request.query.clone(),
                candidates: work.request.candidates[work.next_candidate..end].to_vec(),
            };
            let quantum_started = Instant::now();
            metrics.gpu_quantums.fetch_add(1, Ordering::Relaxed);
            saturating_atomic_add(
                &metrics.candidates_scored,
                u64::try_from(end - work.next_candidate).unwrap_or(u64::MAX),
            );
            saturating_atomic_add(
                &metrics.document_rows_scored,
                quantum
                    .candidates
                    .iter()
                    .map(|candidate| u64::from(candidate.rows))
                    .sum(),
            );
            metrics.gpu_active.store(1, Ordering::Relaxed);
            let score_result = engine.score(&quantum);
            metrics.gpu_active.store(0, Ordering::Relaxed);
            metrics.update_engine(engine.status_snapshot());
            match score_result {
                Ok(results) => {
                    work.gpu_elapsed += quantum_started.elapsed();
                    work.results.extend(results);
                    work.next_candidate = end;
                    if work.deadline <= Instant::now() {
                        metrics.timed_out.fetch_add(1, Ordering::Relaxed);
                        metrics
                            .timeout_during_execution
                            .fetch_add(1, Ordering::Relaxed);
                        Some(protocol::failure(
                            version,
                            request_id,
                            2,
                            "request deadline expired during GPU execution",
                        ))
                    } else if end < work.request.candidates.len() {
                        metrics.scheduler_requeues.fetch_add(1, Ordering::Relaxed);
                        let cost = estimated_next_work(&work, &config);
                        queue.push(Scheduled::new(
                            tenant.clone(),
                            priority,
                            cost,
                            work.accepted_at,
                            work.deadline,
                            work,
                        ));
                        metrics.update_scheduler_depth(queue.len());
                        continue;
                    } else {
                        metrics.completed.fetch_add(1, Ordering::Relaxed);
                        Some(protocol::success(version, request_id, &work.results))
                    }
                }
                Err(error) => {
                    metrics.failed.fetch_add(1, Ordering::Relaxed);
                    metrics.gpu_failures.fetch_add(1, Ordering::Relaxed);
                    work.gpu_elapsed += quantum_started.elapsed();
                    let diagnostic = format!("{error:#}");
                    let failure = protocol::failure(version, request_id, 3, &diagnostic);
                    if is_fatal_cuda_diagnostic(&diagnostic) {
                        // CUDA execution/context failures are not ordinary bad
                        // requests. Stop advertising readiness and terminate
                        // the scheduler after best-effort notification so the
                        // service supervisor can recreate the CUDA context.
                        metrics.ready.store(false, Ordering::Release);
                        write_response_nonfatal(&mut work.connection, &failure);
                        metrics.observe_latency(work.accepted_at.elapsed(), work.gpu_elapsed);
                        return Err(anyhow!("fatal CUDA failure: {diagnostic}"));
                    }
                    Some(failure)
                }
            }
        };
        let Some(response) = response else {
            continue;
        };
        work.connection
            .set_write_timeout(Some(config.socket_io_timeout))
            .ok();
        write_response_nonfatal(&mut work.connection, &response);
        metrics.observe_latency(work.accepted_at.elapsed(), work.gpu_elapsed);
        println!(
            "{}",
            serde_json::json!({
                "event": "tilemaxsim_rust_request",
                "request_id": request_id,
                "tenant_hash": tenant_hash(&tenant),
                "priority": priority,
                "total_ms": work.accepted_at.elapsed().as_secs_f64() * 1000.0,
                "gpu_ms": work.gpu_elapsed.as_secs_f64() * 1000.0,
                "queue_ms": work.accepted_at.elapsed().saturating_sub(work.gpu_elapsed).as_secs_f64() * 1000.0,
                "queue_depth": queue.len(),
                "cache": engine.status_json(),
            })
        );
    }
    Ok(())
}

fn enqueue_work(
    queue: &mut RequestQueue<Work>,
    work: Work,
    config: &SchedulerConfig,
    metrics: &RuntimeMetrics,
) {
    match work.request.priority.cmp(&0) {
        std::cmp::Ordering::Less => metrics
            .admitted_priority_negative
            .fetch_add(1, Ordering::Relaxed),
        std::cmp::Ordering::Equal => metrics
            .admitted_priority_zero
            .fetch_add(1, Ordering::Relaxed),
        std::cmp::Ordering::Greater => metrics
            .admitted_priority_positive
            .fetch_add(1, Ordering::Relaxed),
    };
    let cost = estimated_next_work(&work, config);
    queue.push(Scheduled::new(
        work.request.tenant.clone(),
        work.request.priority,
        cost,
        work.accepted_at,
        work.deadline,
        work,
    ));
    metrics.update_scheduler_depth(queue.len());
}

fn tenant_hash(tenant: &str) -> String {
    let digest = Sha256::digest(tenant.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_fatal_cuda_diagnostic(diagnostic: &str) -> bool {
    [
        "cudaSetDevice",
        "cudaMemcpy",
        "cudaStream",
        "CUDA workspace initialization",
        "TileMaxSim CUDA execution",
        "GPU upload worker panicked",
        "GPU worker panicked",
    ]
    .iter()
    .any(|marker| diagnostic.contains(marker))
}

fn next_quantum_end(work: &Work, config: &SchedulerConfig) -> usize {
    quantum_end(
        &work.request.candidates,
        work.next_candidate,
        config.quantum_candidates,
        config.quantum_tokens,
        config.quantum_fmas,
        work.request.query_rows,
        work.request.dimension,
    )
}

fn quantum_end(
    candidates: &[protocol::Descriptor],
    start: usize,
    maximum_candidates: usize,
    maximum_tokens: u64,
    maximum_fmas: u64,
    query_rows: u32,
    dimension: u32,
) -> usize {
    let mut end = start;
    let mut tokens = 0_u64;
    let mut fmas = 0_u64;
    while end < candidates.len() && end - start < maximum_candidates {
        let rows = u64::from(candidates[end].rows);
        let candidate_fmas = candidate_fmas(query_rows, dimension, candidates[end].rows);
        if end > start
            && (tokens.saturating_add(rows) > maximum_tokens
                || fmas.saturating_add(candidate_fmas) > maximum_fmas)
        {
            break;
        }
        tokens = tokens.saturating_add(rows);
        fmas = fmas.saturating_add(candidate_fmas);
        end += 1;
    }
    end
}

fn candidate_fmas(query_rows: u32, dimension: u32, document_rows: u32) -> u64 {
    u64::from(query_rows)
        .saturating_mul(u64::from(document_rows))
        .saturating_mul(u64::from(dimension))
}

fn estimated_next_work(work: &Work, config: &SchedulerConfig) -> u64 {
    let end = next_quantum_end(work, config);
    work.request.candidates[work.next_candidate..end]
        .iter()
        .map(|candidate| {
            candidate_fmas(
                work.request.query_rows,
                work.request.dimension,
                candidate.rows,
            )
        })
        .fold(0_u64, u64::saturating_add)
        .max(1)
}

fn peer_disconnected(connection: &UnixStream) -> bool {
    let mut byte = 0_u8;
    let result = unsafe {
        libc::recv(
            connection.as_raw_fd(),
            (&raw mut byte).cast(),
            1,
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    if result == 0 {
        return true;
    }
    if result < 0 {
        let error = std::io::Error::last_os_error();
        return !matches!(
            error.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
        );
    }
    false
}

fn write_response_nonfatal(connection: &mut UnixStream, response: &[u8]) {
    if let Err(error) = connection.write_all(response) {
        eprintln!("TileMaxSim response write failed without stopping daemon: {error}");
    }
}

fn run_status_server(listener: UnixListener, path: PathBuf, metrics: Arc<RuntimeMetrics>) {
    while !SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((mut connection, _)) => handle_status_connection(&mut connection, &metrics),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                eprintln!("TileMaxSim status socket failed: {error}");
                break;
            }
        }
    }
    drop(listener);
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!("cannot remove TileMaxSim status socket: {error}"),
    }
}

fn handle_status_connection(connection: &mut UnixStream, metrics: &RuntimeMetrics) {
    connection
        .set_read_timeout(Some(Duration::from_millis(250)))
        .ok();
    connection
        .set_write_timeout(Some(Duration::from_millis(250)))
        .ok();
    let mut request = [0_u8; 1024];
    let Ok(count) = connection.read(&mut request) else {
        return;
    };
    let request = String::from_utf8_lossy(&request[..count]);
    let (status, content_type, body) = if request.starts_with("GET /livez ") {
        (
            "200 OK",
            "application/json",
            serde_json::json!({"live": true}).to_string(),
        )
    } else if request.starts_with("GET /healthz ") {
        let ready = metrics.ready.load(Ordering::Acquire);
        (
            if ready {
                "200 OK"
            } else {
                "503 Service Unavailable"
            },
            "application/json",
            serde_json::json!({"ready": ready}).to_string(),
        )
    } else if request.starts_with("GET /metrics ") {
        (
            "200 OK",
            "text/plain; version=0.0.4",
            render_metrics(metrics),
        )
    } else {
        ("404 Not Found", "text/plain", "not found\n".to_owned())
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    if let Err(error) = connection.write_all(response.as_bytes()) {
        eprintln!("TileMaxSim status response failed: {error}");
    }
}

fn render_metrics(metrics: &RuntimeMetrics) -> String {
    let mut output = String::with_capacity(8 * 1024);
    writeln!(
        output,
        "# HELP tilemaxsim_ready Whether the daemon is ready to accept work."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_ready gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_ready {}",
        usize::from(metrics.ready.load(Ordering::Relaxed))
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_connections Active reader connections and configured limit."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_connections gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_connections{{kind=\"active\"}} {}",
        metrics.active_connections.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_connections{{kind=\"limit\"}} {}",
        metrics.max_connections
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_pending_requests Requests admitted but not yet completed."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_pending_requests gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_pending_requests{{kind=\"current\"}} {}",
        metrics.pending_requests.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_pending_requests{{kind=\"global_limit\"}} {}",
        metrics.max_pending_requests
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_pending_requests{{kind=\"per_tenant_limit\"}} {}",
        metrics.max_tenant_pending_requests
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_pending_tenants Tenants with admitted requests."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_pending_tenants gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_pending_tenants {}",
        metrics.pending_tenants.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_inflight_request_bytes Encoded request frames retained in memory."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_inflight_request_bytes gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_inflight_request_bytes{{kind=\"current\"}} {}",
        metrics.inflight_request_bytes.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_inflight_request_bytes{{kind=\"limit\"}} {}",
        metrics.max_inflight_request_bytes
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_scheduler_queue_depth Requests waiting for a GPU quantum."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_scheduler_queue_depth gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_scheduler_queue_depth{{kind=\"current\"}} {}",
        metrics.scheduler_depth.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_scheduler_queue_depth{{kind=\"high_water\"}} {}",
        metrics.scheduler_depth_high_water.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_active Whether a CUDA quantum is executing."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_active gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_gpu_active {}",
        metrics.gpu_active.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_requests_total Terminal request outcomes."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_requests_total counter").unwrap();
    for (outcome, value) in [
        ("completed", metrics.completed.load(Ordering::Relaxed)),
        ("failed", metrics.failed.load(Ordering::Relaxed)),
        ("timeout", metrics.timed_out.load(Ordering::Relaxed)),
        ("disconnected", metrics.disconnected.load(Ordering::Relaxed)),
    ] {
        writeln!(
            output,
            "tilemaxsim_requests_total{{outcome=\"{outcome}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_admission_rejections_total Bounded admission rejection reasons."
    )
    .unwrap();
    writeln!(
        output,
        "# TYPE tilemaxsim_admission_rejections_total counter"
    )
    .unwrap();
    for (reason, value) in [
        (
            "connections",
            metrics.rejected_connections.load(Ordering::Relaxed),
        ),
        (
            "frame_bytes",
            metrics.rejected_frame_bytes.load(Ordering::Relaxed),
        ),
        (
            "queue_global",
            metrics.rejected_queue_global.load(Ordering::Relaxed),
        ),
        (
            "queue_tenant",
            metrics.rejected_tenant.load(Ordering::Relaxed),
        ),
    ] {
        writeln!(
            output,
            "tilemaxsim_admission_rejections_total{{reason=\"{reason}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_failures_total Internal failure categories."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_failures_total counter").unwrap();
    for (reason, value) in [
        (
            "frame_io_or_validation",
            metrics.frame_read_failures.load(Ordering::Relaxed),
        ),
        ("request", metrics.invalid_requests.load(Ordering::Relaxed)),
        ("gpu", metrics.gpu_failures.load(Ordering::Relaxed)),
        (
            "scheduler",
            metrics.scheduler_failures.load(Ordering::Relaxed),
        ),
    ] {
        writeln!(
            output,
            "tilemaxsim_failures_total{{reason=\"{reason}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_timeouts_total Request timeout phase."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_timeouts_total counter").unwrap();
    for (phase, value) in [
        (
            "before_enqueue",
            metrics.timeout_before_enqueue.load(Ordering::Relaxed),
        ),
        ("queue", metrics.timeout_in_queue.load(Ordering::Relaxed)),
        (
            "before_execution",
            metrics.timeout_before_execution.load(Ordering::Relaxed),
        ),
        (
            "gpu",
            metrics.timeout_during_execution.load(Ordering::Relaxed),
        ),
    ] {
        writeln!(
            output,
            "tilemaxsim_timeouts_total{{phase=\"{phase}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_requests_admitted_total Requests admitted by bounded priority class."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_requests_admitted_total counter").unwrap();
    for (priority_class, value) in [
        (
            "negative",
            metrics.admitted_priority_negative.load(Ordering::Relaxed),
        ),
        (
            "zero",
            metrics.admitted_priority_zero.load(Ordering::Relaxed),
        ),
        (
            "positive",
            metrics.admitted_priority_positive.load(Ordering::Relaxed),
        ),
    ] {
        writeln!(
            output,
            "tilemaxsim_requests_admitted_total{{priority_class=\"{priority_class}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_scheduler_quantums_total CUDA scheduling quanta executed."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_scheduler_quantums_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_scheduler_quantums_total {}",
        metrics.gpu_quantums.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_scheduler_requeues_total Cooperative quantum yields requeued."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_scheduler_requeues_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_scheduler_requeues_total {}",
        metrics.scheduler_requeues.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_candidates_scored_total Candidate tensors submitted to CUDA."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_candidates_scored_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_candidates_scored_total {}",
        metrics.candidates_scored.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_document_rows_scored_total Document token rows submitted to CUDA."
    )
    .unwrap();
    writeln!(
        output,
        "# TYPE tilemaxsim_document_rows_scored_total counter"
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_document_rows_scored_total {}",
        metrics.document_rows_scored.load(Ordering::Relaxed)
    )
    .unwrap();
    let observations = metrics.latency_observations.load(Ordering::Relaxed);
    let total_us = metrics.total_latency_us.load(Ordering::Relaxed);
    let gpu_us = metrics.gpu_latency_us.load(Ordering::Relaxed);
    writeln!(
        output,
        "# HELP tilemaxsim_request_duration_seconds Request wall-clock duration summary."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_request_duration_seconds summary").unwrap();
    writeln!(
        output,
        "tilemaxsim_request_duration_seconds_count {observations}"
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_request_duration_seconds_sum {}",
        total_us as f64 / 1_000_000.0
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_duration_seconds CUDA execution duration summary."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_duration_seconds summary").unwrap();
    writeln!(
        output,
        "tilemaxsim_gpu_duration_seconds_count {observations}"
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_gpu_duration_seconds_sum {}",
        gpu_us as f64 / 1_000_000.0
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_queue_duration_seconds Non-CUDA request duration summary."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_queue_duration_seconds summary").unwrap();
    writeln!(
        output,
        "tilemaxsim_queue_duration_seconds_count {observations}"
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_queue_duration_seconds_sum {}",
        total_us.saturating_sub(gpu_us) as f64 / 1_000_000.0
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_shard_reloads_total Immutable shard metadata reload outcomes."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_shard_reloads_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_shard_reloads_total{{outcome=\"success\"}} {}",
        metrics.reload_succeeded.load(Ordering::Relaxed)
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_shard_reloads_total{{outcome=\"failed\"}} {}",
        metrics.reload_failed.load(Ordering::Relaxed)
    )
    .unwrap();

    let engine = metrics
        .engine
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_cache_bytes GPU tensor-cache byte accounting."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_cache_bytes gauge").unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_cache_entries GPU tensor-cache entry accounting."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_cache_entries gauge").unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_cache_events_total GPU tensor-cache cumulative events."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_cache_events_total counter").unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_h2d_batches_total Host-to-device transfer batches."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_h2d_batches_total counter").unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_gpu_h2d_bytes_total Host-to-device transfer bytes."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_gpu_h2d_bytes_total counter").unwrap();
    for device in &engine.devices {
        for (kind, value) in [
            ("capacity", device.capacity_bytes),
            ("free", device.free_bytes),
            ("largest_free_extent", device.largest_free_extent_bytes),
            ("allocated", device.allocated_bytes),
            ("payload", device.payload_bytes),
            ("internal_waste", device.internal_waste_bytes),
            ("pinned", device.pinned_bytes),
            ("block", device.block_bytes),
        ] {
            writeln!(
                output,
                "tilemaxsim_gpu_cache_bytes{{slot=\"{}\",device=\"{}\",kind=\"{kind}\"}} {value}",
                device.slot, device.device
            )
            .unwrap();
        }
        for (kind, value) in [
            ("total", device.entries),
            ("pinned", device.pinned_entries),
            ("tenants", device.tenants),
        ] {
            writeln!(
                output,
                "tilemaxsim_gpu_cache_entries{{slot=\"{}\",device=\"{}\",kind=\"{kind}\"}} {value}",
                device.slot, device.device
            )
            .unwrap();
        }
        for (event, value) in [
            ("hit", device.hits),
            ("miss", device.misses),
            ("eviction", device.evictions),
            ("admission_rejection", device.admission_rejections),
        ] {
            writeln!(output, "tilemaxsim_gpu_cache_events_total{{slot=\"{}\",device=\"{}\",event=\"{event}\"}} {value}", device.slot, device.device).unwrap();
        }
        writeln!(
            output,
            "tilemaxsim_gpu_h2d_batches_total{{slot=\"{}\",device=\"{}\"}} {}",
            device.slot, device.device, device.h2d_batches
        )
        .unwrap();
        writeln!(
            output,
            "tilemaxsim_gpu_h2d_bytes_total{{slot=\"{}\",device=\"{}\"}} {}",
            device.slot, device.device, device.h2d_bytes
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_host_cache_bytes Host tensor-cache byte accounting."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_host_cache_bytes gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_host_cache_bytes{{kind=\"capacity\"}} {}",
        engine.host.capacity_bytes
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_host_cache_bytes{{kind=\"used\"}} {}",
        engine.host.used_bytes
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_host_cache_entries Host tensor-cache entries and active tenants."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_host_cache_entries gauge").unwrap();
    writeln!(
        output,
        "tilemaxsim_host_cache_entries{{kind=\"entries\"}} {}",
        engine.host.entries
    )
    .unwrap();
    writeln!(
        output,
        "tilemaxsim_host_cache_entries{{kind=\"tenants\"}} {}",
        engine.host.tenants
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_host_cache_events_total Host tensor-cache cumulative events."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_host_cache_events_total counter").unwrap();
    for (event, value) in [
        ("hit", engine.host.hits),
        ("miss", engine.host.misses),
        ("eviction", engine.host.evictions),
        ("admission_rejection", engine.host.admission_rejections),
    ] {
        writeln!(
            output,
            "tilemaxsim_host_cache_events_total{{event=\"{event}\"}} {value}"
        )
        .unwrap();
    }
    writeln!(
        output,
        "# HELP tilemaxsim_storage_read_calls_total Immutable tensor storage read calls."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_storage_read_calls_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_storage_read_calls_total {}",
        engine.batch_read_calls
    )
    .unwrap();
    writeln!(
        output,
        "# HELP tilemaxsim_storage_read_bytes_total Immutable tensor storage read bytes."
    )
    .unwrap();
    writeln!(output, "# TYPE tilemaxsim_storage_read_bytes_total counter").unwrap();
    writeln!(
        output,
        "tilemaxsim_storage_read_bytes_total {}",
        engine.batch_read_bytes
    )
    .unwrap();
    output
}

fn read_request(
    connection: &mut UnixStream,
    maximum: usize,
    frame_admission: &Arc<ByteAdmission>,
) -> Result<(Vec<u8>, BytePermit)> {
    let mut header = [0_u8; HEADER_BYTES];
    connection.read_exact(&mut header)?;
    let body_bytes = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
        .context("request body does not fit this host")?;
    let total = HEADER_BYTES
        .checked_add(body_bytes)
        .ok_or_else(|| anyhow!("request length overflow"))?;
    if total > maximum {
        bail!("request exceeds byte limit");
    }
    let permit = frame_admission
        .try_acquire(total)
        .ok_or_else(|| anyhow!("in-flight TileMaxSim request byte budget is exhausted"))?;
    let mut frame = Vec::with_capacity(total);
    frame.extend_from_slice(&header);
    frame.resize(total, 0);
    connection.read_exact(&mut frame[HEADER_BYTES..])?;
    Ok((frame, permit))
}

fn header_request_id(frame: &[u8]) -> u64 {
    if frame.len() < HEADER_BYTES {
        0
    } else {
        u64::from_le_bytes(frame[8..16].try_into().unwrap())
    }
}

fn header_version(frame: &[u8]) -> u16 {
    if frame.len() < HEADER_BYTES {
        VERSION_EXTERNAL
    } else {
        match u16::from_le_bytes(frame[4..6].try_into().unwrap()) {
            VERSION_SCHEDULED_EXTERNAL => VERSION_SCHEDULED_EXTERNAL,
            _ => VERSION_EXTERNAL,
        }
    }
}

fn acquire_instance_lock(socket: &Path) -> Result<fs::File> {
    let lock_path = PathBuf::from(format!("{}.lock", socket.display()));
    let mut lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("cannot open instance lock {}", lock_path.display()))?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        bail!(
            "another TileMaxSim daemon holds the instance lock {}",
            lock_path.display()
        );
    }
    lock.set_len(0)?;
    writeln!(lock, "{}", std::process::id())?;
    lock.flush()?;
    Ok(lock)
}

fn remove_stale_socket(path: &PathBuf) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !metadata.file_type().is_socket() {
        bail!("refusing to remove non-socket path {}", path.display());
    }
    if UnixStream::connect(path).is_ok() {
        bail!(
            "another TileMaxSim daemon is already listening on {}",
            path.display()
        );
    }
    fs::remove_file(path)?;
    Ok(())
}

#[derive(Deserialize)]
struct ResidentRecord {
    tensor_ref: String,
    tensor_rows: u32,
    tensor_dim: u32,
    tensor_dtype: String,
    tensor_checksum: String,
    canonical_bytes: Option<usize>,
}

fn load_resident_manifests(values: &[(String, PathBuf)]) -> Result<Vec<protocol::Descriptor>> {
    let mut descriptors = Vec::new();
    for (contract, path) in values {
        let file = fs::File::open(path)
            .with_context(|| format!("cannot open resident manifest {}", path.display()))?;
        for (line_number, line) in std::io::BufReader::new(file).lines().enumerate() {
            let line = line?;
            let record: ResidentRecord = serde_json::from_str(&line).with_context(|| {
                format!(
                    "invalid resident manifest {}:{}",
                    path.display(),
                    line_number + 1
                )
            })?;
            let digest = record
                .tensor_ref
                .strip_prefix("sha256://")
                .ok_or_else(|| anyhow!("resident tensor reference is not SHA-256"))?;
            if record.tensor_checksum != format!("sha256:{digest}") {
                bail!("resident tensor reference and checksum disagree");
            }
            let (dtype, scalar_bytes) = match record.tensor_dtype.as_str() {
                "float16" => (2, 2),
                "float32" => (1, 4),
                _ => bail!("resident manifest has an unsupported tensor dtype"),
            };
            let expected_bytes =
                record.tensor_rows as usize * record.tensor_dim as usize * scalar_bytes;
            if record
                .canonical_bytes
                .is_some_and(|bytes| bytes != expected_bytes)
            {
                bail!("resident canonical byte length disagrees with its shape");
            }
            descriptors.push(protocol::Descriptor {
                candidate_id: descriptors.len() as u32,
                contract: contract.clone(),
                digest: digest.to_owned(),
                rows: record.tensor_rows,
                dimension: record.tensor_dim,
                dtype,
            });
        }
    }
    if descriptors.is_empty() {
        bail!("resident manifests contain no tensor descriptors");
    }
    Ok(descriptors)
}

#[cfg(test)]
mod tests {
    use super::{
        ByteAdmission, PendingAdmission, RuntimeMetrics, candidate_fmas, is_fatal_cuda_diagnostic,
        kib_to_bytes, quantum_end, render_metrics, tenant_hash,
    };
    use crate::engine::{DeviceStatus, EngineStatus};
    use crate::protocol::Descriptor;
    use crate::shard::HostCacheStatus;
    use std::sync::Arc;

    #[test]
    fn gpu_block_kib_is_converted_once() {
        assert_eq!(kib_to_bytes(32), Ok(32 * 1024));
        assert!(kib_to_bytes(usize::MAX).is_err());
    }

    #[test]
    fn scheduler_quantum_always_makes_progress_and_honours_both_limits() {
        let descriptor = |rows| Descriptor {
            candidate_id: rows,
            contract: "model".to_owned(),
            digest: "a".repeat(64),
            rows,
            dimension: 2,
            dtype: 2,
        };
        let candidates = vec![descriptor(60), descriptor(60), descriptor(60)];
        assert_eq!(quantum_end(&candidates, 0, 8, 100, 1_000, 5, 2), 1);
        assert_eq!(quantum_end(&candidates, 0, 2, 1_000, 2_000, 5, 2), 2);
        assert_eq!(quantum_end(&candidates, 2, 8, 10, 1_000, 5, 2), 3);
        assert_eq!(candidate_fmas(5, 2, 60), 600);
        assert_eq!(candidate_fmas(u32::MAX, u32::MAX, u32::MAX), u64::MAX);
    }

    #[test]
    fn pending_admission_is_globally_and_per_tenant_bounded() {
        let metrics = Arc::new(RuntimeMetrics::default());
        let admission = Arc::new(PendingAdmission::new(2, 1, metrics));
        let first = admission.try_acquire("a").unwrap();
        assert!(admission.try_acquire("a").is_err());
        let second = admission.try_acquire("b").unwrap();
        assert!(admission.try_acquire("c").is_err());
        drop(first);
        assert!(admission.try_acquire("a").is_ok());
        drop(second);
    }

    #[test]
    fn in_flight_frame_bytes_are_bounded_until_the_permit_drops() {
        let metrics = Arc::new(RuntimeMetrics::default());
        let admission = Arc::new(ByteAdmission::new(100, metrics));
        let first = admission.try_acquire(60).unwrap();
        assert!(admission.try_acquire(41).is_none());
        let second = admission.try_acquire(40).unwrap();
        assert!(admission.try_acquire(1).is_none());
        drop(first);
        assert!(admission.try_acquire(60).is_some());
        drop(second);
    }

    #[test]
    fn tenant_labels_are_stable_and_do_not_expose_raw_ids() {
        assert_eq!(tenant_hash("tenant-a"), tenant_hash("tenant-a"));
        assert_ne!(tenant_hash("tenant-a"), tenant_hash("tenant-b"));
        assert!(!tenant_hash("tenant-a").contains("tenant"));
    }

    #[test]
    fn only_cuda_runtime_failures_force_supervised_restart() {
        assert!(is_fatal_cuda_diagnostic(
            "TileMaxSim CUDA execution: an illegal memory access was encountered"
        ));
        assert!(is_fatal_cuda_diagnostic(
            "cudaMemcpyAsync(H2D): unspecified launch failure"
        ));
        assert!(!is_fatal_cuda_diagnostic(
            "tensor is missing from immutable shards and objects"
        ));
        assert!(!is_fatal_cuda_diagnostic(
            "TileMaxSim request exceeds the configured GPU workspace"
        ));
    }

    #[test]
    fn prometheus_status_contains_only_bounded_scheduler_counters() {
        let metrics = RuntimeMetrics::default();
        metrics
            .ready
            .store(true, std::sync::atomic::Ordering::Relaxed);
        metrics
            .completed
            .store(7, std::sync::atomic::Ordering::Relaxed);
        metrics.update_engine(EngineStatus {
            devices: vec![DeviceStatus {
                slot: 0,
                device: 3,
                capacity_bytes: 1_024,
                free_bytes: 512,
                largest_free_extent_bytes: 256,
                ..DeviceStatus::default()
            }],
            host: HostCacheStatus {
                capacity_bytes: 2_048,
                used_bytes: 128,
                ..HostCacheStatus::default()
            },
            batch_read_calls: 4,
            batch_read_bytes: 256,
        });
        let output = render_metrics(&metrics);
        assert!(output.contains("tilemaxsim_ready 1"));
        assert!(output.contains("outcome=\"completed\"} 7"));
        assert!(output.contains(
            "tilemaxsim_gpu_cache_bytes{slot=\"0\",device=\"3\",kind=\"capacity\"} 1024"
        ));
        assert!(output.contains("tilemaxsim_host_cache_bytes{kind=\"used\"} 128"));
        assert!(output.contains("tilemaxsim_storage_read_bytes_total 256"));
        assert!(!output.contains("tenant-a"));
    }
}
