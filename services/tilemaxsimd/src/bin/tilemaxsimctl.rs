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
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

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
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::PublishObject {
        root,
        rows,
        dimension,
        dtype,
        expected_sha256,
    }) = args.command
    {
        let expected_bytes = tensor_bytes(rows, dimension, &dtype)?;
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
            &root,
            rows,
            dimension,
            &dtype,
            &payload,
            expected_sha256.as_deref(),
        )?;
        println!("{}", serde_json::to_string(&descriptor)?);
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
        let error = match probe(&args.socket, io_timeout) {
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
    fs::create_dir_all(root)?;
    let root_metadata = fs::symlink_metadata(root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        bail!("tensor root must be a real directory");
    }
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

fn probe(socket: &PathBuf, timeout: Duration) -> Result<()> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("cannot connect to status socket {}", socket.display()))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    if !is_ready_response(&response) {
        bail!("TileMaxSim daemon is not ready");
    }
    Ok(())
}

fn is_ready_response(response: &[u8]) -> bool {
    response.starts_with(b"HTTP/1.1 200 ")
        && response
            .windows(b"\r\n\r\n".len())
            .any(|window| window == b"\r\n\r\n")
        && response.ends_with(b"{\"ready\":true}")
}

#[cfg(test)]
mod tests {
    use super::{is_ready_response, publish_object};
    use std::fs;

    #[test]
    fn readiness_requires_success_status_and_true_body() {
        assert!(is_ready_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}"
        ));
        assert!(!is_ready_response(
            b"HTTP/1.1 503 Service Unavailable\r\n\r\n{\"ready\":false}"
        ));
        assert!(!is_ready_response(b"HTTP/1.1 200 OK\r\n\r\n"));
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
}
