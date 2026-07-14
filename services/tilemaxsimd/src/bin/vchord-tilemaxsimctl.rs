// Copyright (c) 2026 HuXinjing

use clap::{Args, Parser, Subcommand};
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const AFTER_HELP: &str = r#"EXAMPLES:
  Check only that the service socket accepts connections:
    vchord-tilemaxsimctl status --socket /run/vectorchord/tilemaxsim.sock

  Also require a coherent daemon readiness file:
    vchord-tilemaxsimctl status --socket /run/vectorchord/tilemaxsim.sock \
      --ready-file /run/vectorchord/tilemaxsim.ready

STATUS EXIT CODES:
  0  The readiness file is valid (when requested) and the socket accepts.
  1  The daemon is not ready or cannot be reached.
  2  Command-line usage is invalid."#;

#[derive(Debug, Parser)]
#[command(
    name = "vchord-tilemaxsimctl",
    version,
    about = "Inspect a VectorChord TileMaxSim daemon",
    long_about = "Perform local operational checks against vchord-tilemaxsimd.\n\
The status command submits an empty v2 request through the I/O workers and GPU\n\
scheduler. It reads no tensor data and launches no scoring kernel.",
    after_long_help = AFTER_HELP,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Report whether the daemon is accepting local connections.
    Status(StatusArgs),
}

#[derive(Debug, Args)]
struct StatusArgs {
    /// Unix-domain socket configured on vchord-tilemaxsimd.
    #[arg(long, value_name = "PATH")]
    socket: PathBuf,

    /// Also validate this daemon-created readiness JSON file.
    #[arg(long, value_name = "PATH")]
    ready_file: Option<PathBuf>,

    /// Overall deadline for connecting and completing the empty probe.
    #[arg(
        long,
        value_name = "MILLISECONDS",
        default_value_t = 1_000,
        value_parser = parse_positive_u32
    )]
    timeout_ms: u32,

    /// Suppress normal status output; the exit code remains authoritative.
    #[arg(short, long)]
    quiet: bool,
}

fn main() {
    let Cli { command } = Cli::parse();
    let (socket, quiet, result) = match command {
        Command::Status(args) => {
            let result = status(&args);
            (args.socket, args.quiet, result)
        }
    };
    match result {
        Ok(()) => {
            if !quiet {
                println!("{} - accepting TileMaxSim connections", socket.display());
            }
        }
        Err(error) => {
            if !quiet {
                eprintln!("{} - no response: {error}", socket.display());
            }
            std::process::exit(1);
        }
    }
}

fn status(args: &StatusArgs) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.socket)
        .map_err(|error| format!("cannot inspect socket: {error}"))?;
    if !metadata.file_type().is_socket() {
        return Err("configured path is not a Unix-domain socket".to_owned());
    }
    if let Some(path) = &args.ready_file {
        validate_ready_file(path, &args.socket)?;
    }
    let deadline = Instant::now() + Duration::from_millis(args.timeout_ms.into());
    let mut connection = connect_with_timeout(&args.socket, remaining(deadline)?)?;
    probe_scheduler(&mut connection, deadline)
}

fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let value = value
        .parse::<u32>()
        .map_err(|_| "value must be a positive integer".to_owned())?;
    if value == 0 {
        Err("value must be a positive integer".to_owned())
    } else {
        Ok(value)
    }
}

fn validate_ready_file(path: &Path, socket: &Path) -> Result<(), String> {
    let contents =
        fs::read(path).map_err(|error| format!("cannot read readiness file: {error}"))?;
    let value: Value = serde_json::from_slice(&contents)
        .map_err(|error| format!("invalid readiness JSON: {error}"))?;
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("unsupported readiness schema".to_owned());
    }
    if value.get("socket").and_then(Value::as_str) != socket.to_str() {
        return Err("readiness file names a different socket".to_owned());
    }
    let pid = value
        .get("pid")
        .and_then(Value::as_u64)
        .and_then(|pid| libc::pid_t::try_from(pid).ok())
        .filter(|pid| *pid > 0)
        .ok_or_else(|| "readiness file contains an invalid PID".to_owned())?;
    // SAFETY: kill(pid, 0) only checks whether a process exists and is visible.
    if unsafe { libc::kill(pid, 0) } != 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EPERM) {
            return Err(format!("readiness PID is not running: {error}"));
        }
    }
    Ok(())
}

fn connect_with_timeout(path: &Path, timeout: Duration) -> Result<UnixStream, String> {
    let bytes = path.as_os_str().as_bytes();
    if bytes.contains(&0) {
        return Err("socket path contains a NUL byte".to_owned());
    }
    let mut address = unsafe { std::mem::zeroed::<libc::sockaddr_un>() };
    if bytes.len() >= address.sun_path.len() {
        return Err("socket path is too long".to_owned());
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (target, source) in address.sun_path.iter_mut().zip(bytes.iter().copied()) {
        *target = source as libc::c_char;
    }
    // SAFETY: socket has no pointer arguments; successful ownership is moved to OwnedFd.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return Err(format!(
            "cannot create health-check socket: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `raw` is a new, uniquely owned file descriptor.
    let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
    let address_length =
        (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1) as libc::socklen_t;
    // SAFETY: the address is initialized for AF_UNIX and its exact length is supplied.
    if unsafe {
        libc::connect(
            descriptor.as_raw_fd(),
            std::ptr::addr_of!(address).cast(),
            address_length,
        )
    } == 0
    {
        return Ok(UnixStream::from(descriptor));
    }
    let error = std::io::Error::last_os_error();
    if !matches!(error.raw_os_error(), Some(libc::EINPROGRESS | libc::EAGAIN)) {
        return Err(format!("cannot connect: {error}"));
    }
    let timeout_ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
    let mut poll = libc::pollfd {
        fd: descriptor.as_raw_fd(),
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY: exactly one valid pollfd is supplied for the bounded wait.
    let poll_result = unsafe { libc::poll(std::ptr::addr_of_mut!(poll), 1, timeout_ms) };
    if poll_result == 0 {
        return Err("connection timed out".to_owned());
    }
    if poll_result < 0 {
        return Err(format!(
            "connection poll failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut socket_error = 0;
    let mut error_length = std::mem::size_of_val(&socket_error) as libc::socklen_t;
    // SAFETY: both output pointers refer to initialized writable values of the required size.
    if unsafe {
        libc::getsockopt(
            descriptor.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_ERROR,
            std::ptr::addr_of_mut!(socket_error).cast(),
            &mut error_length,
        )
    } != 0
    {
        return Err(format!(
            "cannot inspect connection: {}",
            std::io::Error::last_os_error()
        ));
    }
    if socket_error != 0 {
        return Err(format!(
            "cannot connect: {}",
            std::io::Error::from_raw_os_error(socket_error)
        ));
    }
    Ok(UnixStream::from(descriptor))
}

fn probe_scheduler(connection: &mut UnixStream, deadline: Instant) -> Result<(), String> {
    const REQUEST_ID: u64 = 0x5643_4845_414c_5448;
    let contract = b"health";
    let body_bytes = 4 + 4 + 4 + 1 + 1 + 2 + 4 + contract.len() + 2;
    let mut request = Vec::with_capacity(24 + body_bytes);
    request.extend_from_slice(b"VCTM");
    request.extend_from_slice(&2_u16.to_le_bytes());
    request.extend_from_slice(&1_u16.to_le_bytes());
    request.extend_from_slice(&REQUEST_ID.to_le_bytes());
    request.extend_from_slice(&(body_bytes as u64).to_le_bytes());
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&0_u32.to_le_bytes());
    request.push(2);
    request.push(1);
    request.extend_from_slice(&0_u16.to_le_bytes());
    request.extend_from_slice(&(contract.len() as u32).to_le_bytes());
    request.extend_from_slice(contract);
    request.extend_from_slice(&0_u16.to_le_bytes());

    connection
        .set_nonblocking(false)
        .map_err(|error| format!("cannot configure probe socket: {error}"))?;
    set_deadlines(connection, deadline)?;
    connection
        .write_all(&request)
        .map_err(|error| format!("cannot write scheduler probe: {error}"))?;
    let mut header = [0_u8; 24];
    set_deadlines(connection, deadline)?;
    connection
        .read_exact(&mut header)
        .map_err(|error| format!("cannot read scheduler probe: {error}"))?;
    if &header[..4] != b"VCTM"
        || u16::from_le_bytes(header[4..6].try_into().unwrap()) != 2
        || u16::from_le_bytes(header[6..8].try_into().unwrap()) != 2
        || u64::from_le_bytes(header[8..16].try_into().unwrap()) != REQUEST_ID
    {
        return Err("daemon returned an invalid probe response header".to_owned());
    }
    let body_length = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
        .map_err(|_| "daemon probe response is too large".to_owned())?;
    if body_length > 64 * 1024 {
        return Err("daemon probe response is too large".to_owned());
    }
    let mut body = vec![0_u8; body_length];
    set_deadlines(connection, deadline)?;
    connection
        .read_exact(&mut body)
        .map_err(|error| format!("cannot read scheduler probe body: {error}"))?;
    if body.len() != 8
        || u32::from_le_bytes(body[..4].try_into().unwrap()) != 0
        || u32::from_le_bytes(body[4..8].try_into().unwrap()) != 0
    {
        return Err("daemon scheduler rejected the empty probe".to_owned());
    }
    Ok(())
}

fn set_deadlines(connection: &UnixStream, deadline: Instant) -> Result<(), String> {
    let timeout = remaining(deadline)?;
    connection
        .set_read_timeout(Some(timeout))
        .and_then(|()| connection.set_write_timeout(Some(timeout)))
        .map_err(|error| format!("cannot configure probe deadline: {error}"))
}

fn remaining(deadline: Instant) -> Result<Duration, String> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    if timeout.is_zero() {
        Err("probe deadline expired".to_owned())
    } else {
        Ok(timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_and_exit_codes_are_documented() {
        Cli::command().debug_assert();
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("vchord-tilemaxsimctl"));
        assert!(help.contains("STATUS EXIT CODES:"));
        assert!(help.contains("status --socket"));
        assert!(
            Cli::try_parse_from([
                "vchord-tilemaxsimctl",
                "status",
                "--socket",
                "/tmp/test.sock",
                "--timeout-ms",
                "0"
            ])
            .is_err()
        );
    }
}
