mod cache;
mod engine;
mod gpu;
mod protocol;
mod shard;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use engine::Engine;
use gpu::Gpu;
use protocol::HEADER_BYTES;
use serde::Deserialize;
use shard::ShardStore;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Instant;

const GIB: usize = 1024 * 1024 * 1024;

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
    #[arg(long = "contract-root", required = true, value_parser = parse_contract_root)]
    contract_roots: Vec<(String, PathBuf)>,
    #[arg(long, default_value_t = 32)]
    gpu_block_kib: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_request_bytes: usize,
    #[arg(long, default_value = "600", value_parser = parse_mode)]
    socket_mode: u32,
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

fn main() -> Result<()> {
    let args = Args::parse();
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
    remove_stale_socket(&args.socket)?;
    let listener = UnixListener::bind(&args.socket)
        .with_context(|| format!("cannot bind {}", args.socket.display()))?;
    fs::set_permissions(&args.socket, fs::Permissions::from_mode(args.socket_mode))?;
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
            "cache": engine.status_json(),
        })
    );
    let mut accepted = 0_usize;
    for connection in listener.incoming() {
        let mut connection = connection?;
        let started = Instant::now();
        let response = match read_request(&mut connection, args.max_request_bytes) {
            Ok(frame) => {
                let request_id = header_request_id(&frame);
                match protocol::parse(&frame).and_then(|request| {
                    let request_id = request.request_id;
                    engine
                        .score(&request)
                        .map(|results| protocol::success(request_id, &results))
                }) {
                    Ok(response) => response,
                    Err(error) => protocol::failure(request_id, 3, &format!("{error:#}")),
                }
            }
            Err(error) => protocol::failure(0, 1, &format!("{error:#}")),
        };
        connection.write_all(&response)?;
        println!(
            "{}",
            serde_json::json!({
                "event": "tilemaxsim_rust_request",
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                "cache": engine.status_json(),
            })
        );
        accepted += 1;
        if args.once && accepted == 1 {
            break;
        }
    }
    drop(listener);
    match fs::remove_file(&args.socket) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn read_request(connection: &mut UnixStream, maximum: usize) -> Result<Vec<u8>> {
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
    let mut frame = Vec::with_capacity(total);
    frame.extend_from_slice(&header);
    frame.resize(total, 0);
    connection.read_exact(&mut frame[HEADER_BYTES..])?;
    Ok(frame)
}

fn header_request_id(frame: &[u8]) -> u64 {
    if frame.len() < HEADER_BYTES {
        0
    } else {
        u64::from_le_bytes(frame[8..16].try_into().unwrap())
    }
}

fn remove_stale_socket(path: &PathBuf) -> Result<()> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if !metadata.file_type().is_socket() {
        bail!("refusing to remove non-socket path {}", path.display());
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
