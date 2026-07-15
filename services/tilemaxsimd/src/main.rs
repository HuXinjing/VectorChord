mod cache;
mod engine;
mod gpu;
mod protocol;
mod scheduler;
mod shard;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use engine::Engine;
use gpu::Gpu;
use protocol::{HEADER_BYTES, VERSION_EXTERNAL, VERSION_SCHEDULED_EXTERNAL};
use scheduler::{RequestQueue, Scheduled, SchedulerPolicy};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use shard::ShardStore;
use std::collections::HashMap;
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
    let metrics = Arc::new(RuntimeMetrics::default());
    install_signal_handlers()?;
    let ready_cache = engine.status_json();

    let (sender, receiver) = mpsc::sync_channel::<Work>(args.max_queued_requests);
    let frame_admission = Arc::new(ByteAdmission::new(args.max_inflight_request_gb));
    let pending_admission = Arc::new(PendingAdmission::new(
        args.max_queued_requests,
        args.max_tenant_queued_requests,
    ));
    let tenant_weights = args.tenant_weights.iter().cloned().collect();
    let scheduler_config = SchedulerConfig {
        policy: args.scheduler_policy,
        priority_aging: Duration::from_millis(args.priority_aging_ms),
        priority_band: args.priority_band,
        batch_window: Duration::from_millis(args.scheduler_batch_window_ms),
        quantum_candidates: args.scheduler_quantum_candidates,
        quantum_tokens: args.scheduler_quantum_tokens,
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
            "status_socket": args.status_socket,
            "cache": ready_cache,
        })
    );

    let live_readers = Arc::new(AtomicUsize::new(0));
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
                if !try_acquire_reader(&live_readers, args.max_connections) {
                    // Closing immediately is deliberate: we have not read enough
                    // bytes to know whether the peer expects a v2 or v3 response.
                    metrics.rejected_global.fetch_add(1, Ordering::Relaxed);
                    drop(connection);
                    continue;
                }
                accepted += 1;
                let reader_sender = sender.clone();
                let reader_count = Arc::clone(&live_readers);
                let reader_admission = Arc::clone(&pending_admission);
                let reader_metrics = Arc::clone(&metrics);
                let reader_frame_admission = Arc::clone(&frame_admission);
                let reader_config = ReaderConfig {
                    maximum: args.max_request_bytes,
                    io_timeout: Duration::from_millis(args.socket_io_timeout_ms),
                    server_timeout: Duration::from_millis(args.request_timeout_ms),
                };
                match thread::Builder::new()
                    .name("tilemaxsim-reader".to_owned())
                    .spawn(move || {
                        let _permit = ReaderPermit(reader_count);
                        read_and_enqueue(
                            connection,
                            &reader_sender,
                            reader_config,
                            reader_admission,
                            reader_metrics,
                            reader_frame_admission,
                        );
                    }) {
                    Ok(reader) => readers.push(reader),
                    Err(error) => {
                        live_readers.fetch_sub(1, Ordering::Release);
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
    socket_io_timeout: Duration,
    tenant_weights: std::collections::HashMap<String, f64>,
}

#[derive(Default)]
struct RuntimeMetrics {
    ready: AtomicBool,
    scheduler_depth: AtomicUsize,
    gpu_active: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
    timed_out: AtomicU64,
    disconnected: AtomicU64,
    rejected_global: AtomicU64,
    rejected_tenant: AtomicU64,
}

struct ReaderPermit(Arc<AtomicUsize>);

impl Drop for ReaderPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
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
}

struct ByteAdmission {
    used: AtomicUsize,
    maximum: usize,
}

struct BytePermit {
    admission: Arc<ByteAdmission>,
    bytes: usize,
}

impl ByteAdmission {
    fn new(maximum: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            maximum,
        }
    }

    fn try_acquire(self: &Arc<Self>, bytes: usize) -> Option<BytePermit> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|next| *next <= self.maximum)
            })
            .ok()?;
        Some(BytePermit {
            admission: Arc::clone(self),
            bytes,
        })
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        self.admission.used.fetch_sub(self.bytes, Ordering::Release);
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
    fn new(max_total: usize, max_tenant: usize) -> Self {
        Self {
            state: Mutex::new(PendingState {
                total: 0,
                tenants: HashMap::new(),
            }),
            max_total,
            max_tenant,
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
            write_response_nonfatal(
                &mut connection,
                &protocol::failure(version, request_id, 1, &format!("{error:#}")),
            );
            return;
        }
    };
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
            metrics.rejected_global.fetch_add(1, Ordering::Relaxed);
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
            metrics.rejected_global.fetch_add(1, Ordering::Relaxed);
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
                Ok(()) => println!(
                    "{}",
                    serde_json::json!({"event": "tilemaxsim_rust_shards_reloaded"})
                ),
                Err(error) => eprintln!("TileMaxSim shard reload rejected: {error:#}"),
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
        metrics
            .scheduler_depth
            .store(queue.len(), Ordering::Relaxed);
        let Some(scheduled) = queue.pop(Instant::now()) else {
            continue;
        };
        metrics
            .scheduler_depth
            .store(queue.len(), Ordering::Relaxed);
        if peer_disconnected(&scheduled.payload.connection) {
            metrics.disconnected.fetch_add(1, Ordering::Relaxed);
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
            metrics.gpu_active.store(1, Ordering::Relaxed);
            match engine.score(&quantum) {
                Ok(results) => {
                    metrics.gpu_active.store(0, Ordering::Relaxed);
                    work.gpu_elapsed += quantum_started.elapsed();
                    work.results.extend(results);
                    work.next_candidate = end;
                    if work.deadline <= Instant::now() {
                        metrics.timed_out.fetch_add(1, Ordering::Relaxed);
                        Some(protocol::failure(
                            version,
                            request_id,
                            2,
                            "request deadline expired during GPU execution",
                        ))
                    } else if end < work.request.candidates.len() {
                        let cost = estimated_next_work(&work, &config);
                        queue.push(Scheduled::new(
                            tenant.clone(),
                            priority,
                            cost,
                            work.accepted_at,
                            work.deadline,
                            work,
                        ));
                        metrics
                            .scheduler_depth
                            .store(queue.len(), Ordering::Relaxed);
                        continue;
                    } else {
                        metrics.completed.fetch_add(1, Ordering::Relaxed);
                        Some(protocol::success(version, request_id, &work.results))
                    }
                }
                Err(error) => {
                    metrics.gpu_active.store(0, Ordering::Relaxed);
                    metrics.failed.fetch_add(1, Ordering::Relaxed);
                    work.gpu_elapsed += quantum_started.elapsed();
                    Some(protocol::failure(
                        version,
                        request_id,
                        3,
                        &format!("{error:#}"),
                    ))
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
    let cost = estimated_next_work(&work, config);
    queue.push(Scheduled::new(
        work.request.tenant.clone(),
        work.request.priority,
        cost,
        work.accepted_at,
        work.deadline,
        work,
    ));
    metrics
        .scheduler_depth
        .store(queue.len(), Ordering::Relaxed);
}

fn tenant_hash(tenant: &str) -> String {
    let digest = Sha256::digest(tenant.as_bytes());
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn next_quantum_end(work: &Work, config: &SchedulerConfig) -> usize {
    quantum_end(
        &work.request.candidates,
        work.next_candidate,
        config.quantum_candidates,
        config.quantum_tokens,
    )
}

fn quantum_end(
    candidates: &[protocol::Descriptor],
    start: usize,
    maximum_candidates: usize,
    maximum_tokens: u64,
) -> usize {
    let mut end = start;
    let mut tokens = 0_u64;
    while end < candidates.len() && end - start < maximum_candidates {
        let rows = u64::from(candidates[end].rows);
        if end > start && tokens.saturating_add(rows) > maximum_tokens {
            break;
        }
        tokens = tokens.saturating_add(rows);
        end += 1;
    }
    end
}

fn estimated_next_work(work: &Work, config: &SchedulerConfig) -> u64 {
    let end = next_quantum_end(work, config);
    let document_rows = work.request.candidates[work.next_candidate..end]
        .iter()
        .map(|candidate| u64::from(candidate.rows))
        .sum::<u64>();
    u64::from(work.request.query_rows)
        .saturating_mul(document_rows)
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
    format!(
        concat!(
            "# HELP tilemaxsim_ready Whether the daemon is ready to accept work.\n",
            "# TYPE tilemaxsim_ready gauge\n",
            "tilemaxsim_ready {}\n",
            "# HELP tilemaxsim_scheduler_queue_depth Requests waiting for a GPU quantum.\n",
            "# TYPE tilemaxsim_scheduler_queue_depth gauge\n",
            "tilemaxsim_scheduler_queue_depth {}\n",
            "# HELP tilemaxsim_gpu_active Whether a CUDA quantum is executing.\n",
            "# TYPE tilemaxsim_gpu_active gauge\n",
            "tilemaxsim_gpu_active {}\n",
            "# HELP tilemaxsim_requests_total Completed request outcomes.\n",
            "# TYPE tilemaxsim_requests_total counter\n",
            "tilemaxsim_requests_total{{outcome=\"completed\"}} {}\n",
            "tilemaxsim_requests_total{{outcome=\"failed\"}} {}\n",
            "tilemaxsim_requests_total{{outcome=\"timeout\"}} {}\n",
            "tilemaxsim_requests_total{{outcome=\"disconnected\"}} {}\n",
            "# HELP tilemaxsim_admission_rejections_total Admission rejections.\n",
            "# TYPE tilemaxsim_admission_rejections_total counter\n",
            "tilemaxsim_admission_rejections_total{{reason=\"global\"}} {}\n",
            "tilemaxsim_admission_rejections_total{{reason=\"tenant\"}} {}\n",
        ),
        usize::from(metrics.ready.load(Ordering::Relaxed)),
        metrics.scheduler_depth.load(Ordering::Relaxed),
        metrics.gpu_active.load(Ordering::Relaxed),
        metrics.completed.load(Ordering::Relaxed),
        metrics.failed.load(Ordering::Relaxed),
        metrics.timed_out.load(Ordering::Relaxed),
        metrics.disconnected.load(Ordering::Relaxed),
        metrics.rejected_global.load(Ordering::Relaxed),
        metrics.rejected_tenant.load(Ordering::Relaxed),
    )
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
        ByteAdmission, PendingAdmission, RuntimeMetrics, kib_to_bytes, quantum_end, render_metrics,
        tenant_hash,
    };
    use crate::protocol::Descriptor;
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
        assert_eq!(quantum_end(&candidates, 0, 8, 100), 1);
        assert_eq!(quantum_end(&candidates, 0, 2, 1_000), 2);
        assert_eq!(quantum_end(&candidates, 2, 8, 10), 3);
    }

    #[test]
    fn pending_admission_is_globally_and_per_tenant_bounded() {
        let admission = Arc::new(PendingAdmission::new(2, 1));
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
        let admission = Arc::new(ByteAdmission::new(100));
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
    fn prometheus_status_contains_only_bounded_scheduler_counters() {
        let metrics = RuntimeMetrics::default();
        metrics
            .ready
            .store(true, std::sync::atomic::Ordering::Relaxed);
        metrics
            .completed
            .store(7, std::sync::atomic::Ordering::Relaxed);
        let output = render_metrics(&metrics);
        assert!(output.contains("tilemaxsim_ready 1"));
        assert!(output.contains("outcome=\"completed\"} 7"));
        assert!(!output.contains("tenant-a"));
    }
}
