// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[derive(Parser)]
#[command(about = "Probe the native TileMaxSim daemon readiness socket")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long, default_value = "/run/vectorchord/tilemaxsim-status.sock")]
    socket: PathBuf,
    #[arg(long, default_value_t = 500)]
    io_timeout_ms: u64,
    #[arg(long, default_value_t = 0)]
    wait_timeout_ms: u64,
    #[arg(long, default_value = "ready", value_parser = ["ready", "live"])]
    probe: String,
}

#[derive(Subcommand)]
enum Command {
    /// Publish one canonical tensor from stdin into the immutable object store.
    PublishObject {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        rows: u32,
        #[arg(long)]
        dimension: u32,
        #[arg(long, value_parser = ["float16", "float32"])]
        dtype: String,
        #[arg(long)]
        expected_sha256: Option<String>,
    },
    /// Remove old immutable objects not named by the sha256:// refs on stdin.
    GcObjects {
        #[arg(long)]
        root: PathBuf,
        #[arg(long, default_value_t = 86_400)]
        grace_seconds: u64,
        /// Actually remove eligible objects. The default is a dry run.
        #[arg(long)]
        delete: bool,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::PublishObject {
        root,
        rows,
        dimension,
        dtype,
        expected_sha256,
    }) = args.command.as_ref()
    {
        let expected_bytes = tensor_bytes(*rows, *dimension, dtype)?;
        let mut payload = Vec::with_capacity(expected_bytes);
        std::io::stdin()
            .take(expected_bytes.saturating_add(1) as u64)
            .read_to_end(&mut payload)?;
        if payload.len() != expected_bytes {
            bail!(
                "stdin tensor length {} does not match declared length {expected_bytes}",
                payload.len()
            );
        }
        let descriptor = publish_object(
            root,
            *rows,
            *dimension,
            dtype,
            &payload,
            expected_sha256.as_deref(),
        )?;
        println!("{}", serde_json::to_string(&descriptor)?);
        return Ok(());
    }
    if let Some(Command::GcObjects {
        root,
        grace_seconds,
        delete,
    }) = args.command.as_ref()
    {
        let live = read_live_refs(BufReader::new(std::io::stdin().lock()))?;
        let outcome = gc_objects(root, &live, *grace_seconds, *delete)?;
        println!("{}", serde_json::to_string(&outcome)?);
        return Ok(());
    }
    if args.io_timeout_ms == 0 {
        bail!("I/O timeout must be positive");
    }
    let io_timeout = Duration::from_millis(args.io_timeout_ms);
    let wait_timeout = Duration::from_millis(args.wait_timeout_ms);
    let deadline = Instant::now()
        .checked_add(wait_timeout)
        .ok_or_else(|| anyhow!("readiness deadline overflow"))?;

    loop {
        let error = match probe(&args.socket, io_timeout, &args.probe) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        if wait_timeout.is_zero() || Instant::now() >= deadline {
            return Err(error);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[derive(serde::Serialize)]
struct PublishedDescriptor {
    tensor_ref: String,
    tensor_rows: u32,
    tensor_dim: u32,
    tensor_dtype: String,
    tensor_checksum: String,
}

#[derive(serde::Serialize)]
struct GcOutcome {
    live_refs: usize,
    scanned_objects: u64,
    eligible_objects: u64,
    eligible_bytes: u64,
    deleted_objects: u64,
    deleted_bytes: u64,
    skipped_entries: u64,
    dry_run: bool,
}

struct ObjectStoreLock {
    _file: File,
}

fn prepare_store_root(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("tensor root must be a real directory");
    }
    Ok(())
}

fn lock_object_store(root: &Path, exclusive: bool) -> Result<ObjectStoreLock> {
    prepare_store_root(root)?;
    let path = root.join(".tilemaxsim-objects.lock");
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .mode(0o640)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&path)
        .with_context(|| format!("cannot open object-store lock {}", path.display()))?;
    let operation = if exclusive {
        libc::LOCK_EX
    } else {
        libc::LOCK_SH
    };
    loop {
        if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
            break;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error)
                .with_context(|| format!("cannot lock tensor object store {}", root.display()));
        }
    }
    Ok(ObjectStoreLock { _file: file })
}

fn read_live_refs(reader: impl BufRead) -> Result<HashSet<String>> {
    let mut live = HashSet::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let value = line.trim();
        if value.is_empty() {
            continue;
        }
        let Some(digest) = value.strip_prefix("sha256://") else {
            bail!("live ref on line {} is not a sha256:// URI", index + 1);
        };
        if !is_lower_hex_digest(digest) {
            bail!("live ref on line {} has an invalid digest", index + 1);
        }
        live.insert(digest.to_owned());
    }
    Ok(live)
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn gc_objects(
    root: &Path,
    live: &HashSet<String>,
    grace_seconds: u64,
    delete: bool,
) -> Result<GcOutcome> {
    let _lock = lock_object_store(root, true)?;
    let mut outcome = GcOutcome {
        live_refs: live.len(),
        scanned_objects: 0,
        eligible_objects: 0,
        eligible_bytes: 0,
        deleted_objects: 0,
        deleted_bytes: 0,
        skipped_entries: 0,
        dry_run: !delete,
    };
    let objects = root.join("objects");
    if !objects.try_exists()? {
        return Ok(outcome);
    }
    let objects_metadata = fs::symlink_metadata(&objects)?;
    if objects_metadata.file_type().is_symlink() || !objects_metadata.is_dir() {
        bail!("tensor objects path must be a real directory");
    }

    let now = SystemTime::now();
    let grace = Duration::from_secs(grace_seconds);
    for directory in fs::read_dir(&objects)? {
        let directory = directory?;
        let directory_metadata = fs::symlink_metadata(directory.path())?;
        if directory_metadata.file_type().is_symlink() {
            bail!("tensor objects path contains a symlink");
        }
        let Some(prefix) = directory.file_name().to_str().map(str::to_owned) else {
            outcome.skipped_entries += 1;
            continue;
        };
        if !directory_metadata.is_dir()
            || prefix.len() != 2
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            outcome.skipped_entries += 1;
            continue;
        }
        for object in fs::read_dir(directory.path())? {
            let object = object?;
            let path = object.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                bail!("tensor object path contains a symlink");
            }
            let Some(filename) = object.file_name().to_str().map(str::to_owned) else {
                outcome.skipped_entries += 1;
                continue;
            };
            let Some(digest) = filename.strip_suffix(".tensor") else {
                outcome.skipped_entries += 1;
                continue;
            };
            if !metadata.is_file() || !is_lower_hex_digest(digest) || !digest.starts_with(&prefix) {
                outcome.skipped_entries += 1;
                continue;
            }
            outcome.scanned_objects += 1;
            if live.contains(digest) {
                continue;
            }
            let modified = metadata.modified().with_context(|| {
                format!("cannot read tensor object timestamp {}", path.display())
            })?;
            if now.duration_since(modified).unwrap_or_default() < grace {
                continue;
            }
            outcome.eligible_objects += 1;
            outcome.eligible_bytes = outcome.eligible_bytes.saturating_add(metadata.len());
            if delete {
                fs::remove_file(&path)
                    .with_context(|| format!("cannot remove tensor object {}", path.display()))?;
                outcome.deleted_objects += 1;
                outcome.deleted_bytes = outcome.deleted_bytes.saturating_add(metadata.len());
            }
        }
    }
    Ok(outcome)
}

fn tensor_bytes(rows: u32, dimension: u32, dtype: &str) -> Result<usize> {
    if rows == 0 || rows > 65_536 || dimension == 0 || dimension > 60_000 {
        bail!("invalid tensor shape");
    }
    let scalar_bytes = match dtype {
        "float16" => 2usize,
        "float32" => 4usize,
        _ => bail!("unsupported tensor dtype"),
    };
    (rows as usize)
        .checked_mul(dimension as usize)
        .and_then(|elements| elements.checked_mul(scalar_bytes))
        .ok_or_else(|| anyhow!("tensor shape overflow"))
}

fn publish_object(
    root: &Path,
    rows: u32,
    dimension: u32,
    dtype: &str,
    payload: &[u8],
    expected_sha256: Option<&str>,
) -> Result<PublishedDescriptor> {
    if payload.len() != tensor_bytes(rows, dimension, dtype)? {
        bail!("tensor payload length disagrees with its shape");
    }
    let digest = hex::encode(Sha256::digest(payload));
    if expected_sha256.is_some_and(|expected| expected != digest) {
        bail!("tensor payload checksum disagrees with --expected-sha256");
    }
    let _lock = lock_object_store(root, false)?;
    let objects = root.join("objects");
    fs::create_dir_all(&objects)?;
    let objects_metadata = fs::symlink_metadata(&objects)?;
    if objects_metadata.file_type().is_symlink() || !objects_metadata.is_dir() {
        bail!("tensor objects path must be a real directory");
    }
    let directory = objects.join(&digest[..2]);
    fs::create_dir_all(&directory)?;
    let directory_metadata = fs::symlink_metadata(&directory)?;
    if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
        bail!("tensor object directory must be a real directory");
    }
    let destination = directory.join(format!("{digest}.tensor"));
    if destination.try_exists()? {
        verify_existing_object(&destination, payload.len(), &digest)?;
    } else {
        let temporary = directory.join(format!(".{digest}.{}.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o640)
            .open(&temporary)
            .with_context(|| format!("cannot create {}", temporary.display()))?;
        let publish = (|| -> Result<()> {
            file.write_all(payload)?;
            file.sync_all()?;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o440))?;
            match fs::hard_link(&temporary, &destination) {
                Ok(()) => File::open(&directory)?.sync_all()?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    verify_existing_object(&destination, payload.len(), &digest)?;
                }
                Err(error) => return Err(error.into()),
            }
            Ok(())
        })();
        drop(file);
        let remove_result = fs::remove_file(&temporary);
        if let Err(error) = remove_result
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(error.into());
        }
        publish?;
    }
    Ok(PublishedDescriptor {
        tensor_ref: format!("sha256://{digest}"),
        tensor_rows: rows,
        tensor_dim: dimension,
        tensor_dtype: dtype.to_owned(),
        tensor_checksum: format!("sha256:{digest}"),
    })
}

fn verify_existing_object(path: &Path, expected_bytes: usize, expected_digest: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != expected_bytes as u64
    {
        bail!("existing immutable tensor object is invalid");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if hex::encode(digest.finalize()) != expected_digest {
        bail!("existing immutable tensor object checksum mismatch");
    }
    Ok(())
}

fn probe(socket: &PathBuf, timeout: Duration, kind: &str) -> Result<()> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("cannot connect to status socket {}", socket.display()))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let path = if kind == "live" { "/livez" } else { "/healthz" };
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    if !is_probe_response(&response, kind) {
        bail!("TileMaxSim daemon failed its {kind} probe");
    }
    Ok(())
}

fn is_probe_response(response: &[u8], kind: &str) -> bool {
    let expected_body = if kind == "live" {
        b"{\"live\":true}".as_slice()
    } else {
        b"{\"ready\":true}".as_slice()
    };
    response.starts_with(b"HTTP/1.1 200 ")
        && response
            .windows(b"\r\n\r\n".len())
            .any(|window| window == b"\r\n\r\n")
        && response.ends_with(expected_body)
}

#[cfg(test)]
mod tests {
    use super::{gc_objects, is_probe_response, publish_object, read_live_refs};
    use std::collections::HashSet;
    use std::fs;
    use std::io::Cursor;

    #[test]
    fn readiness_requires_success_status_and_true_body() {
        assert!(is_probe_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}",
            "ready"
        ));
        assert!(!is_probe_response(
            b"HTTP/1.1 503 Service Unavailable\r\n\r\n{\"ready\":false}",
            "ready"
        ));
        assert!(!is_probe_response(b"HTTP/1.1 200 OK\r\n\r\n", "ready"));
    }

    #[test]
    fn liveness_is_distinct_from_readiness() {
        assert!(is_probe_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"live\":true}",
            "live",
        ));
        assert!(!is_probe_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}",
            "live",
        ));
    }

    #[test]
    fn publish_object_is_content_addressed_and_idempotent() {
        let root = std::env::temp_dir().join(format!(
            "tilemaxsimctl-publish-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        let payload = [0_u8, 60, 0, 0];
        let first = publish_object(&root, 1, 2, "float16", &payload, None).unwrap();
        let second = publish_object(&root, 1, 2, "float16", &payload, None).unwrap();
        assert_eq!(first.tensor_ref, second.tensor_ref);
        assert_eq!(first.tensor_checksum, second.tensor_checksum);
        assert!(
            root.join("objects")
                .join(&first.tensor_checksum[7..9])
                .exists()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gc_requires_valid_refs_and_preserves_live_objects() {
        let root = std::env::temp_dir().join(format!(
            "tilemaxsimctl-gc-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        let first = publish_object(&root, 1, 2, "float16", &[0, 60, 0, 0], None).unwrap();
        let second = publish_object(&root, 1, 2, "float16", &[0, 56, 0, 0], None).unwrap();
        let live = read_live_refs(Cursor::new(format!("{}\n", first.tensor_ref))).unwrap();
        let dry_run = gc_objects(&root, &live, 0, false).unwrap();
        assert_eq!(dry_run.scanned_objects, 2);
        assert_eq!(dry_run.eligible_objects, 1);
        assert_eq!(dry_run.deleted_objects, 0);

        let deleted = gc_objects(&root, &live, 0, true).unwrap();
        assert_eq!(deleted.deleted_objects, 1);
        let first_digest = &first.tensor_checksum[7..];
        let second_digest = &second.tensor_checksum[7..];
        assert!(
            root.join("objects")
                .join(&first_digest[..2])
                .join(format!("{first_digest}.tensor"))
                .exists()
        );
        assert!(
            !root
                .join("objects")
                .join(&second_digest[..2])
                .join(format!("{second_digest}.tensor"))
                .exists()
        );
        assert!(read_live_refs(Cursor::new("not-a-ref\n")).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_live_manifest_is_explicitly_supported_for_an_empty_database() {
        let live = read_live_refs(Cursor::new("\n")).unwrap();
        assert_eq!(live, HashSet::new());
    }
}
