// Copyright (c) 2026 HuXinjing

mod cache;
mod engine;
mod gpu;
mod lifecycle;
mod protocol;
mod server;
mod shard;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use engine::Engine;
use gpu::Gpu;
use lifecycle::ManagedFile;
use serde::Deserialize;
use server::ServerConfig;
use shard::ShardStore;
use std::collections::HashSet;
use std::fs;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const GIB: usize = 1024 * 1024 * 1024;
const MIB: usize = 1024 * 1024;

const AFTER_HELP: &str = r#"EXAMPLES:
  Start an evictable 20 GB cache on GPU 1:
    vchord-tilemaxsimd --socket /run/vectorchord/tilemaxsim.sock \
      --gpu-memory-gb 1=20 --gpu-workspace-gb 2 \
      --contract-root colqwen@1=/var/lib/vectorchord/colqwen

  Pin a complete tensor manifest before accepting requests:
    vchord-tilemaxsimd --socket /run/vectorchord/tilemaxsim.sock \
      --gpu-memory-gb 1=20 --gpu-workspace-gb 2 \
      --contract-root colqwen@1=/var/lib/vectorchord/colqwen \
      --gpu-cache-mode resident \
      --resident-manifest colqwen@1=/var/lib/vectorchord/colqwen/descriptors.jsonl

Memory options named GB use GiB internally (1 GB = 1024^3 bytes). The daemon
allocates every configured GPU arena before creating the socket; startup fails
without leaving a ready socket if any allocation or prewarm step fails."#;

#[derive(Debug, Parser)]
#[command(
    name = "vchord-tilemaxsimd",
    version,
    about = "VectorChord TileMaxSim GPU cache and scoring service",
    long_about = "Run the native VectorChord TileMaxSim service over a local Unix-domain socket.\n\
The process owns the configured GPU memory, immutable tensor shards, cache\n\
admission policy, and CUDA scoring scheduler.",
    after_long_help = AFTER_HELP,
    next_line_help = true
)]
struct Args {
    /// Unix-domain socket used by the VectorChord PostgreSQL backend.
    #[arg(long, value_name = "PATH", help_heading = "Connection options")]
    socket: PathBuf,

    /// Socket permissions as an octal mode.
    #[arg(
        long,
        value_name = "MODE",
        default_value = "600",
        value_parser = parse_mode,
        help_heading = "Connection options"
    )]
    socket_mode: u32,

    /// Maximum time for socket I/O, queueing, and scoring.
    #[arg(
        long,
        value_name = "MILLISECONDS",
        default_value_t = 2_000,
        value_parser = parse_positive_usize,
        help_heading = "Connection options"
    )]
    request_timeout_ms: usize,

    /// Number of clients that may be read or waiting for a response.
    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 8,
        value_parser = parse_positive_usize,
        help_heading = "Connection options"
    )]
    max_inflight: usize,

    /// Kernel listen backlog for the Unix-domain socket.
    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 64,
        value_parser = parse_positive_usize,
        help_heading = "Connection options"
    )]
    backlog: usize,

    /// Maximum parsed requests waiting for the GPU scheduler.
    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 64,
        value_parser = parse_positive_usize,
        help_heading = "Connection options"
    )]
    max_queued_requests: usize,

    /// Permit a client Unix user ID in addition to the daemon's own UID.
    #[arg(
        long,
        value_name = "UID",
        action = clap::ArgAction::Append,
        help_heading = "Connection options"
    )]
    allow_peer_uid: Vec<u32>,

    /// Permit a client Unix group ID.
    #[arg(
        long,
        value_name = "GID",
        action = clap::ArgAction::Append,
        help_heading = "Connection options"
    )]
    allow_peer_gid: Vec<u32>,

    /// Strict process-owned allocation in GPU=GB form; repeat for more GPUs.
    #[arg(
        long,
        required = true,
        value_name = "GPU=GB",
        value_parser = parse_gpu_memory,
        help_heading = "GPU and cache options"
    )]
    gpu_memory_gb: Vec<GpuMemory>,

    /// Per-GPU portion reserved for queries and scoring output.
    #[arg(
        long,
        value_name = "GB",
        default_value = "2",
        value_parser = parse_gb,
        help_heading = "GPU and cache options"
    )]
    gpu_workspace_gb: usize,

    /// Decoded host-memory tensor cache shared by shard readers.
    #[arg(
        long,
        value_name = "GB",
        default_value = "8",
        value_parser = parse_gb,
        help_heading = "GPU and cache options"
    )]
    host_cache_gb: usize,

    /// GPU cache behavior: evict cold tensors or pin a complete manifest.
    #[arg(
        long,
        value_name = "MODE",
        default_value = "lru",
        value_enum,
        help_heading = "GPU and cache options"
    )]
    gpu_cache_mode: CacheMode,

    /// Base page size inside each preallocated GPU tensor arena.
    #[arg(
        long,
        value_name = "KIB",
        default_value_t = 32,
        value_parser = parse_gpu_block_kib,
        help_heading = "GPU and cache options"
    )]
    gpu_block_kib: usize,

    /// Immutable tensor shard root in MODEL_CONTRACT_ID=/absolute/path form.
    #[arg(
        long = "contract-root",
        required = true,
        value_name = "MODEL_CONTRACT_ID=PATH",
        value_parser = parse_contract_root,
        help_heading = "Tensor storage options"
    )]
    contract_roots: Vec<(String, PathBuf)>,

    /// Verify complete shard hashes lazily in addition to per-tensor hashes.
    #[arg(long, help_heading = "Tensor storage options")]
    verify_full_shards: bool,

    /// Descriptor manifest to pin before readiness; required in resident mode.
    #[arg(
        long = "resident-manifest",
        value_name = "MODEL_CONTRACT_ID=PATH",
        value_parser = parse_contract_root,
        help_heading = "Tensor storage options"
    )]
    resident_manifests: Vec<(String, PathBuf)>,

    /// Number of resident descriptors resolved and uploaded per prewarm batch.
    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 256,
        value_parser = parse_positive_usize,
        help_heading = "Tensor storage options"
    )]
    prewarm_batch_size: usize,

    /// Maximum accepted request frame size.
    #[arg(
        long,
        value_name = "MB",
        default_value_t = 64,
        value_parser = parse_positive_usize,
        help_heading = "Resource limits"
    )]
    max_request_mb: usize,

    /// Maximum query plus candidate tensor tokens accepted per request.
    #[arg(
        long,
        value_name = "COUNT",
        default_value_t = 1_000_000,
        value_parser = parse_positive_usize,
        help_heading = "Resource limits"
    )]
    max_batch_tokens: usize,

    /// Maximum decoded query plus candidate tensor bytes per request.
    #[arg(
        long,
        value_name = "MB",
        default_value_t = 1024,
        value_parser = parse_positive_usize,
        help_heading = "Resource limits"
    )]
    max_batch_mb: usize,

    /// Atomically create this file after the service socket is ready.
    #[arg(long, value_name = "PATH", help_heading = "Process control")]
    ready_file: Option<PathBuf>,

    /// Write the daemon PID to this file and remove it during clean shutdown.
    #[arg(long, value_name = "PATH", help_heading = "Process control")]
    pid_file: Option<PathBuf>,

    /// Maximum time to drain accepted work after SIGINT or SIGTERM.
    #[arg(
        long,
        value_name = "MILLISECONDS",
        default_value_t = 30_000,
        value_parser = parse_positive_usize,
        help_heading = "Process control"
    )]
    shutdown_grace_ms: usize,

    /// Exit after one accepted request. Intended for tests and smoke checks.
    #[arg(long, hide = true)]
    once: bool,
}

#[derive(Clone, Debug)]
struct GpuMemory {
    device: i32,
    bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CacheMode {
    Lru,
    Resident,
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

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "value must be a positive integer".to_owned())?;
    if parsed == 0 {
        return Err("value must be a positive integer".to_owned());
    }
    Ok(parsed)
}

fn parse_gpu_block_kib(value: &str) -> Result<usize, String> {
    let kib = parse_positive_usize(value)?;
    if !(4..=1024).contains(&kib) || !kib.is_power_of_two() {
        return Err("GPU block size must be a power of two from 4 to 1024 KiB".to_owned());
    }
    Ok(kib)
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let _pid_file = args
        .pid_file
        .as_deref()
        .map(ManagedFile::create_pid)
        .transpose()?;
    let max_request_bytes = args
        .max_request_mb
        .checked_mul(MIB)
        .ok_or_else(|| anyhow!("maximum request size overflow"))?;
    let max_batch_bytes = args
        .max_batch_mb
        .checked_mul(MIB)
        .ok_or_else(|| anyhow!("maximum tensor batch size overflow"))?;
    let mut seen_devices = std::collections::HashSet::new();
    for specification in &args.gpu_memory_gb {
        if args.gpu_workspace_gb >= specification.bytes {
            bail!("every configured GPU allocation must exceed its workspace");
        }
        if !seen_devices.insert(specification.device) {
            bail!("each CUDA device may be configured only once");
        }
    }
    let block_bytes = args
        .gpu_block_kib
        .checked_mul(1024)
        .ok_or_else(|| anyhow!("GPU block size overflow"))?;
    if block_bytes == 0 || block_bytes % 256 != 0 {
        bail!("GPU block size must be positive and 256-byte aligned");
    }
    let store = ShardStore::open(
        &args.contract_roots,
        args.host_cache_gb,
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
    let mut engine = Engine::new(gpus, block_bytes, store)?;
    if args.gpu_cache_mode == CacheMode::Resident && args.resident_manifests.is_empty() {
        bail!("resident GPU cache mode requires at least one resident manifest");
    }
    if args.gpu_cache_mode == CacheMode::Lru && !args.resident_manifests.is_empty() {
        bail!("resident manifests are valid only in resident GPU cache mode");
    }
    if args.gpu_cache_mode == CacheMode::Resident {
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
    let mut allowed_uids = args.allow_peer_uid.iter().copied().collect::<HashSet<_>>();
    // SAFETY: geteuid has no preconditions and cannot fail.
    allowed_uids.insert(unsafe { libc::geteuid() });
    let allowed_gids = args.allow_peer_gid.iter().copied().collect();
    let server_config = ServerConfig {
        socket: args.socket.clone(),
        socket_mode: args.socket_mode,
        request_timeout: Duration::from_millis(args.request_timeout_ms as u64),
        max_request_bytes,
        max_batch_tokens: args.max_batch_tokens,
        max_batch_bytes,
        max_inflight: args.max_inflight,
        backlog: args.backlog,
        max_queued_requests: args.max_queued_requests,
        allowed_uids,
        allowed_gids,
        shutdown_grace: Duration::from_millis(args.shutdown_grace_ms as u64),
        once: args.once,
    };
    server::install_signal_handlers()?;
    let bound = server::bind(&server_config)?;
    let ready = serde_json::json!({
        "event": "tilemaxsim_rust_ready",
        "schema_version": 1,
        "pid": std::process::id(),
        "version": env!("CARGO_PKG_VERSION"),
        "socket": bound.path(),
        "devices": args.gpu_memory_gb.iter().map(|item| serde_json::json!({
            "device": item.device,
            "allocated_bytes": item.bytes,
        })).collect::<Vec<_>>(),
        "workspace_bytes": args.gpu_workspace_gb,
        "limits": {
            "request_bytes": max_request_bytes,
            "batch_tokens": args.max_batch_tokens,
            "batch_bytes": max_batch_bytes,
            "max_inflight": args.max_inflight,
            "connection_backlog": args.backlog,
            "gpu_queue": args.max_queued_requests,
            "request_timeout_ms": args.request_timeout_ms,
        },
        "cache": engine.status_json(),
    });
    let ready_contents = serde_json::to_vec(&ready)?;
    let ready_file = args
        .ready_file
        .as_deref()
        .map(|path| ManagedFile::create_ready(path, &ready_contents))
        .transpose()?;
    println!("{ready}");
    server::serve(engine, bound, server_config, ready_file)
}

fn validate_args(args: &Args) -> Result<()> {
    if args.max_inflight > 1024 {
        bail!("--max-inflight cannot exceed 1024");
    }
    if args.backlog > 65_535 {
        bail!("--backlog cannot exceed 65535");
    }
    if args.max_queued_requests > 65_535 {
        bail!("--max-queued-requests cannot exceed 65535");
    }
    if args.max_request_mb > 1024 {
        bail!("--max-request-mb cannot exceed 1024");
    }
    if args.max_batch_tokens > 1_000_000 {
        bail!("--max-batch-tokens cannot exceed 1000000");
    }
    if args.max_batch_mb > 1024 {
        bail!("--max-batch-mb cannot exceed 1024");
    }
    if args.request_timeout_ms > u32::MAX as usize {
        bail!("--request-timeout-ms is too large");
    }
    if args.shutdown_grace_ms > u32::MAX as usize {
        bail!("--shutdown-grace-ms is too large");
    }
    let paths = [
        ("--socket", Some(&args.socket)),
        ("--pid-file", args.pid_file.as_ref()),
        ("--ready-file", args.ready_file.as_ref()),
    ];
    for (index, (left_name, left_path)) in paths.iter().enumerate() {
        let Some(left_path) = left_path else {
            continue;
        };
        for (right_name, right_path) in &paths[index + 1..] {
            if right_path.is_some_and(|right_path| right_path == *left_path) {
                bail!("{left_name} and {right_name} must use different paths");
            }
        }
    }
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
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_consistent_and_documents_operational_contract() {
        Args::command().debug_assert();
        let help = Args::command().render_long_help().to_string();
        for expected in [
            "vchord-tilemaxsimd",
            "--gpu-memory-gb <GPU=GB>",
            "--max-request-mb <MB>",
            "--max-batch-tokens <COUNT>",
            "--max-batch-mb <MB>",
            "--request-timeout-ms <MILLISECONDS>",
            "--allow-peer-uid <UID>",
            "--ready-file <PATH>",
            "allocates every configured GPU arena before creating the socket",
            "EXAMPLES:",
        ] {
            assert!(help.contains(expected), "long help is missing {expected:?}");
        }
        assert!(!help.contains("--max-request-bytes"));
    }

    #[test]
    fn memory_is_configured_in_gigabytes_and_block_size_is_bounded() {
        let memory = parse_gpu_memory("cuda:2=1.5").unwrap();
        assert_eq!(memory.device, 2);
        assert_eq!(memory.bytes, GIB + GIB / 2);
        assert_eq!(parse_gpu_block_kib("32").unwrap(), 32);
        assert!(parse_gpu_block_kib("3").is_err());
        assert!(parse_gpu_block_kib("48").is_err());
        assert!(parse_gpu_block_kib("2048").is_err());
    }

    #[test]
    fn process_control_paths_must_not_alias_the_socket() {
        let args = Args::try_parse_from([
            "vchord-tilemaxsimd",
            "--socket",
            "/tmp/vchord-test.sock",
            "--gpu-memory-gb",
            "0=1",
            "--contract-root",
            "model@1=/tmp",
            "--ready-file",
            "/tmp/vchord-test.sock",
        ])
        .unwrap();
        let error = validate_args(&args).unwrap_err();
        assert!(format!("{error:#}").contains("must use different paths"));
    }
}
