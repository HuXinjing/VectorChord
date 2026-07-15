use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "Probe the native TileMaxSim daemon readiness socket")]
struct Args {
    #[arg(
        long,
        default_value = "/run/vectorchord/tilemaxsim-status.sock"
    )]
    socket: PathBuf,
    #[arg(long, default_value_t = 500)]
    io_timeout_ms: u64,
    #[arg(long, default_value_t = 0)]
    wait_timeout_ms: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
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
    use super::is_ready_response;

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
}
