// Copyright (c) 2026 HuXinjing

use crate::engine::Engine;
use crate::lifecycle::ManagedFile;
use crate::protocol::{self, HEADER_BYTES};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const STATUS_INVALID_REQUEST: u32 = 1;
const STATUS_RESOURCE_LIMIT: u32 = 2;
const STATUS_COMPUTE_ERROR: u32 = 3;
const WAIT_QUANTUM: Duration = Duration::from_millis(25);
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub socket: PathBuf,
    pub socket_mode: u32,
    pub request_timeout: Duration,
    pub max_request_bytes: usize,
    pub max_batch_tokens: usize,
    pub max_batch_bytes: usize,
    pub max_inflight: usize,
    pub backlog: usize,
    pub max_queued_requests: usize,
    pub allowed_uids: HashSet<u32>,
    pub allowed_gids: HashSet<u32>,
    pub shutdown_grace: Duration,
    pub once: bool,
}

pub struct BoundServer {
    socket: SocketGuard,
    listener: UnixListener,
}

impl BoundServer {
    pub fn path(&self) -> &PathBuf {
        &self.socket.path
    }

    fn close(mut self) -> Result<()> {
        self.socket.remove()?;
        drop(self.listener);
        Ok(())
    }
}

struct SocketGuard {
    path: PathBuf,
    device: u64,
    inode: u64,
    removed: bool,
}

impl SocketGuard {
    fn remove(&mut self) -> Result<()> {
        if self.removed {
            return Ok(());
        }
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                self.removed = true;
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        if metadata.dev() != self.device || metadata.ino() != self.inode {
            bail!(
                "refusing to remove replaced service socket {}",
                self.path.display()
            );
        }
        fs::remove_file(&self.path)?;
        self.removed = true;
        Ok(())
    }
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}

#[derive(Clone, Copy, Debug)]
struct PeerCredentials {
    pid: i32,
    uid: u32,
    gid: u32,
}

struct ConnectionTask {
    stream: UnixStream,
    accepted_at: Instant,
    peer: PeerCredentials,
}

struct WorkItem {
    request: protocol::Request,
    accepted_at: Instant,
    queued_at: Instant,
    deadline: Instant,
    peer: PeerCredentials,
    canceled: Arc<AtomicBool>,
    reply: SyncSender<Vec<u8>>,
}

pub fn install_signal_handlers() -> Result<()> {
    SHUTDOWN_REQUESTED.store(false, Ordering::Relaxed);
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = shutdown_signal_handler as *const () as usize;
    // SAFETY: `action.sa_mask` is a valid signal set owned by this function.
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    action.sa_flags = 0;
    // SAFETY: the handler only performs an async-signal-safe atomic store.
    if unsafe { libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut()) } != 0
        || unsafe { libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) } != 0
    {
        return Err(std::io::Error::last_os_error()).context("cannot install signal handlers");
    }
    Ok(())
}

extern "C" fn shutdown_signal_handler(_signal: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
}

pub fn bind(config: &ServerConfig) -> Result<BoundServer> {
    remove_stale_socket(&config.socket)?;
    let listener = UnixListener::bind(&config.socket)
        .with_context(|| format!("cannot bind {}", config.socket.display()))?;
    let metadata = fs::symlink_metadata(&config.socket)?;
    let socket = SocketGuard {
        path: config.socket.clone(),
        device: metadata.dev(),
        inode: metadata.ino(),
        removed: false,
    };
    fs::set_permissions(
        &config.socket,
        fs::Permissions::from_mode(config.socket_mode),
    )?;
    let backlog = i32::try_from(config.backlog).context("socket backlog is too large")?;
    // SAFETY: the listener owns a valid listening Unix socket descriptor.
    if unsafe { libc::listen(listener.as_raw_fd(), backlog) } != 0 {
        return Err(std::io::Error::last_os_error()).context("cannot set socket backlog");
    }
    listener.set_nonblocking(true)?;
    Ok(BoundServer { socket, listener })
}

pub fn serve(
    mut engine: Engine,
    bound: BoundServer,
    config: ServerConfig,
    mut ready_file: Option<ManagedFile>,
) -> Result<()> {
    let (connection_tx, connection_rx) = sync_channel::<ConnectionTask>(config.backlog);
    let connection_rx = Arc::new(Mutex::new(connection_rx));
    let (work_tx, work_rx) = sync_channel::<WorkItem>(config.max_queued_requests);
    let scheduler = thread::Builder::new()
        .name("vchord-tilemaxsim-gpu".to_owned())
        .spawn(move || scheduler_loop(&mut engine, work_rx))?;

    let mut workers = Vec::with_capacity(config.max_inflight);
    for index in 0..config.max_inflight {
        let connections = Arc::clone(&connection_rx);
        let work = work_tx.clone();
        let worker_config = config.clone();
        workers.push(
            thread::Builder::new()
                .name(format!("vchord-tilemaxsim-io-{index}"))
                .spawn(move || connection_loop(connections, work, worker_config))?,
        );
    }
    drop(connection_rx);
    drop(work_tx);

    let mut accepted = 0_u64;
    let mut runtime_failure = None;
    while !SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
        if scheduler.is_finished() {
            runtime_failure = Some("GPU scheduler stopped unexpectedly");
            break;
        }
        if workers.iter().any(JoinHandle::is_finished) {
            runtime_failure = Some("connection worker stopped unexpectedly");
            break;
        }
        match bound.listener.accept() {
            Ok((stream, _)) => {
                let accepted_at = Instant::now();
                let peer = match peer_credentials(&stream) {
                    Ok(peer) => peer,
                    Err(error) => {
                        reject_connection(
                            stream,
                            config.request_timeout,
                            STATUS_INVALID_REQUEST,
                            "cannot read peer credentials",
                        );
                        log_rejection(None, "peer_credentials", &format!("{error:#}"));
                        continue;
                    }
                };
                if !peer_is_allowed(peer, &config) {
                    reject_connection(
                        stream,
                        config.request_timeout,
                        STATUS_INVALID_REQUEST,
                        "Unix peer credentials are not authorized",
                    );
                    log_rejection(Some(peer), "unauthorized_peer", "peer rejected");
                    continue;
                }
                let task = ConnectionTask {
                    stream,
                    accepted_at,
                    peer,
                };
                match connection_tx.try_send(task) {
                    Ok(()) => {
                        accepted += 1;
                        if config.once && accepted == 1 {
                            break;
                        }
                    }
                    Err(TrySendError::Full(task)) => {
                        reject_connection(
                            task.stream,
                            config.request_timeout,
                            STATUS_RESOURCE_LIMIT,
                            "connection backlog is full",
                        );
                        log_rejection(Some(task.peer), "connection_backlog_full", "busy");
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        bail!("connection worker pool stopped unexpectedly");
                    }
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                thread::sleep(WAIT_QUANTUM);
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error).context("service socket accept failed"),
        }
    }

    if let Some(file) = ready_file.take() {
        file.remove().context("cannot remove readiness file")?;
    }
    let socket_path = bound.path().clone();
    bound.close()?;
    drop(connection_tx);
    let deadline = Instant::now() + config.shutdown_grace;
    let workers_finished = join_until(workers, deadline, "connection workers");
    let scheduler_finished = join_one_until(scheduler, deadline, "GPU scheduler");
    let drained = workers_finished && scheduler_finished;
    println!(
        "{}",
        json!({
            "event": "tilemaxsim_rust_stopped",
            "schema_version": 1,
            "socket": socket_path,
            "accepted": accepted,
            "drained": drained,
            "runtime_failure": runtime_failure,
        })
    );
    if let Some(message) = runtime_failure {
        bail!(message);
    }
    if !drained {
        bail!("shutdown grace period expired before all work drained");
    }
    Ok(())
}

fn connection_loop(
    connections: Arc<Mutex<Receiver<ConnectionTask>>>,
    work: SyncSender<WorkItem>,
    config: ServerConfig,
) {
    loop {
        let task = match connections.lock() {
            Ok(receiver) => receiver.recv(),
            Err(_) => return,
        };
        let Ok(task) = task else {
            return;
        };
        handle_connection(task, &work, &config);
    }
}

fn handle_connection(task: ConnectionTask, work: &SyncSender<WorkItem>, config: &ServerConfig) {
    let deadline = task.accepted_at + config.request_timeout;
    let mut stream = task.stream;
    let result = (|| -> Result<()> {
        set_stream_deadline(&stream, deadline)?;
        let Some(frame) = read_request(&mut stream, config.max_request_bytes)? else {
            log_io_result(task.peer, 0, "readiness_probe", task.accepted_at, None);
            return Ok(());
        };
        let request_id = header_request_id(&frame);
        let request = match protocol::parse(&frame) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    &mut stream,
                    protocol::failure(request_id, STATUS_INVALID_REQUEST, &format!("{error:#}")),
                    deadline,
                )?;
                log_io_result(
                    task.peer,
                    request_id,
                    "invalid_request",
                    task.accepted_at,
                    None,
                );
                return Ok(());
            }
        };
        if request.tensor_tokens > config.max_batch_tokens
            || request.tensor_bytes > config.max_batch_bytes
        {
            write_response(
                &mut stream,
                protocol::failure(
                    request.request_id,
                    STATUS_RESOURCE_LIMIT,
                    "request exceeds configured tensor batch limits",
                ),
                deadline,
            )?;
            log_io_result(
                task.peer,
                request.request_id,
                "batch_limit",
                task.accepted_at,
                None,
            );
            return Ok(());
        }
        if Instant::now() >= deadline {
            write_response(
                &mut stream,
                protocol::failure(
                    request.request_id,
                    STATUS_COMPUTE_ERROR,
                    "request deadline expired before queueing",
                ),
                deadline,
            )?;
            log_io_result(
                task.peer,
                request.request_id,
                "timeout",
                task.accepted_at,
                None,
            );
            return Ok(());
        }
        let canceled = Arc::new(AtomicBool::new(false));
        let (reply_tx, reply_rx) = sync_channel(1);
        let request_id = request.request_id;
        let item = WorkItem {
            request,
            accepted_at: task.accepted_at,
            queued_at: Instant::now(),
            deadline,
            peer: task.peer,
            canceled: Arc::clone(&canceled),
            reply: reply_tx,
        };
        match work.try_send(item) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                write_response(
                    &mut stream,
                    protocol::failure(
                        request_id,
                        STATUS_RESOURCE_LIMIT,
                        "GPU request queue is full",
                    ),
                    deadline,
                )?;
                log_io_result(
                    task.peer,
                    request_id,
                    "gpu_queue_full",
                    task.accepted_at,
                    None,
                );
                return Ok(());
            }
            Err(TrySendError::Disconnected(_)) => {
                bail!("GPU scheduler stopped unexpectedly");
            }
        }
        loop {
            let now = Instant::now();
            if now >= deadline {
                canceled.store(true, Ordering::Relaxed);
                let response =
                    protocol::failure(request_id, STATUS_COMPUTE_ERROR, "request deadline expired");
                let _ = write_response(&mut stream, response, deadline);
                return Ok(());
            }
            let wait = (deadline - now).min(WAIT_QUANTUM);
            match reply_rx.recv_timeout(wait) {
                Ok(response) => {
                    write_response(&mut stream, response, deadline)?;
                    return Ok(());
                }
                Err(RecvTimeoutError::Timeout) => {
                    if peer_disconnected(&stream)? {
                        canceled.store(true, Ordering::Relaxed);
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("GPU scheduler dropped a request without a response");
                }
            }
        }
    })();
    if let Err(error) = result {
        let request_id = 0;
        let status = if is_timeout_error(&error) {
            "timeout"
        } else {
            "io_error"
        };
        let response = protocol::failure(request_id, STATUS_COMPUTE_ERROR, &format!("{error:#}"));
        let _ = write_response(&mut stream, response, deadline);
        log_io_result(
            task.peer,
            request_id,
            status,
            task.accepted_at,
            Some(&format!("{error:#}")),
        );
    }
}

fn scheduler_loop(engine: &mut Engine, work: Receiver<WorkItem>) {
    for item in work {
        let compute_started = Instant::now();
        let queue_ms = (compute_started - item.queued_at).as_secs_f64() * 1000.0;
        let candidate_count = item.request.candidates.len();
        let request_id = item.request.request_id;
        let result = if compute_started >= item.deadline {
            Err(anyhow!("request deadline expired in the GPU queue"))
        } else if item.canceled.load(Ordering::Relaxed) {
            Err(anyhow!("request canceled because the client disconnected"))
        } else {
            engine.score_until(&item.request, item.deadline, &item.canceled)
        };
        let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;
        let (status, error, response) = match result {
            Ok(results) => ("ok", None, protocol::success(request_id, &results)),
            Err(error) => {
                let message = format!("{error:#}");
                let status = if message.contains("deadline") {
                    "timeout"
                } else if message.contains("canceled") {
                    "canceled"
                } else {
                    "compute_error"
                };
                let response = protocol::failure(request_id, STATUS_COMPUTE_ERROR, &message);
                (status, Some(message), response)
            }
        };
        let client_present = item.reply.try_send(response).is_ok();
        println!(
            "{}",
            json!({
                "event": "tilemaxsim_rust_request",
                "schema_version": 1,
                "request_id": request_id,
                "status": status,
                "error": error,
                "peer_pid": item.peer.pid,
                "peer_uid": item.peer.uid,
                "peer_gid": item.peer.gid,
                "candidates": candidate_count,
                "queue_ms": queue_ms,
                "compute_ms": compute_ms,
                "total_ms": item.accepted_at.elapsed().as_secs_f64() * 1000.0,
                "client_present": client_present,
                "cache": engine.status_json(),
            })
        );
    }
}

fn peer_credentials(stream: &UnixStream) -> Result<PeerCredentials> {
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: `credentials` and `length` point to writable objects of the
    // exact type and size required by Linux SO_PEERCRED.
    let status = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            std::ptr::addr_of_mut!(credentials).cast(),
            &mut length,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error()).context("SO_PEERCRED failed");
    }
    if length as usize != std::mem::size_of::<libc::ucred>() {
        bail!("SO_PEERCRED returned an unexpected structure size");
    }
    Ok(PeerCredentials {
        pid: credentials.pid,
        uid: credentials.uid,
        gid: credentials.gid,
    })
}

fn peer_is_allowed(peer: PeerCredentials, config: &ServerConfig) -> bool {
    config.allowed_uids.contains(&peer.uid) || config.allowed_gids.contains(&peer.gid)
}

fn peer_disconnected(stream: &UnixStream) -> Result<bool> {
    let mut descriptor = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLHUP | libc::POLLERR,
        revents: 0,
    };
    // SAFETY: exactly one valid pollfd is supplied for a nonblocking poll.
    let result = unsafe { libc::poll(std::ptr::addr_of_mut!(descriptor), 1, 0) };
    if result < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == ErrorKind::Interrupted {
            return Ok(false);
        }
        return Err(error).context("cannot inspect client connection");
    }
    Ok(descriptor.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0)
}

fn set_stream_deadline(stream: &UnixStream, deadline: Instant) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        bail!("request deadline expired");
    }
    stream.set_read_timeout(Some(remaining))?;
    stream.set_write_timeout(Some(remaining))?;
    Ok(())
}

fn write_response(stream: &mut UnixStream, response: Vec<u8>, deadline: Instant) -> Result<()> {
    set_stream_deadline(stream, deadline)?;
    stream.write_all(&response)?;
    Ok(())
}

fn read_request(connection: &mut UnixStream, maximum: usize) -> Result<Option<Vec<u8>>> {
    let mut header = [0_u8; HEADER_BYTES];
    let mut read = 0;
    while read < header.len() {
        match connection.read(&mut header[read..])? {
            0 if read == 0 => return Ok(None),
            0 => bail!("client closed with an incomplete request header"),
            bytes => read += bytes,
        }
    }
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
    Ok(Some(frame))
}

fn header_request_id(frame: &[u8]) -> u64 {
    if frame.len() < HEADER_BYTES {
        0
    } else {
        u64::from_le_bytes(frame[8..16].try_into().unwrap())
    }
}

fn reject_connection(mut stream: UnixStream, timeout: Duration, status: u32, message: &str) {
    let deadline = Instant::now() + timeout;
    let _ = write_response(&mut stream, protocol::failure(0, status, message), deadline);
}

fn log_rejection(peer: Option<PeerCredentials>, status: &str, error: &str) {
    println!(
        "{}",
        json!({
            "event": "tilemaxsim_rust_request",
            "schema_version": 1,
            "request_id": 0,
            "status": status,
            "error": error,
            "peer_pid": peer.map(|item| item.pid),
            "peer_uid": peer.map(|item| item.uid),
            "peer_gid": peer.map(|item| item.gid),
        })
    );
}

fn log_io_result(
    peer: PeerCredentials,
    request_id: u64,
    status: &str,
    accepted_at: Instant,
    error: Option<&str>,
) {
    println!(
        "{}",
        json!({
            "event": "tilemaxsim_rust_request",
            "schema_version": 1,
            "request_id": request_id,
            "status": status,
            "error": error,
            "peer_pid": peer.pid,
            "peer_uid": peer.uid,
            "peer_gid": peer.gid,
            "total_ms": accepted_at.elapsed().as_secs_f64() * 1000.0,
        })
    );
}

fn is_timeout_error(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        source
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| {
                matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock)
            })
    }) || format!("{error:#}").contains("deadline")
}

fn remove_stale_socket(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_socket() {
        bail!("refusing to remove non-socket path {}", path.display());
    }
    if socket_is_active(path)? {
        bail!("another daemon is already accepting on {}", path.display());
    }
    let current = fs::symlink_metadata(path)?;
    if current.dev() != metadata.dev() || current.ino() != metadata.ino() {
        bail!("service socket changed while checking {}", path.display());
    }
    fs::remove_file(path)?;
    Ok(())
}

fn socket_is_active(path: &Path) -> Result<bool> {
    let bytes = path.as_os_str().as_bytes();
    let mut address = unsafe { std::mem::zeroed::<libc::sockaddr_un>() };
    if bytes.contains(&0) || bytes.len() >= address.sun_path.len() {
        bail!("invalid Unix socket path {}", path.display());
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (target, source) in address.sun_path.iter_mut().zip(bytes.iter().copied()) {
        *target = source as libc::c_char;
    }
    // SAFETY: socket has no pointer arguments; ownership moves to OwnedFd on success.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error()).context("cannot probe service socket");
    }
    // SAFETY: `raw` is a new and uniquely owned descriptor.
    let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
    let address_length =
        (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1) as libc::socklen_t;
    // SAFETY: the initialized AF_UNIX address and its exact length are supplied.
    if unsafe {
        libc::connect(
            descriptor.as_raw_fd(),
            std::ptr::addr_of!(address).cast(),
            address_length,
        )
    } == 0
    {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ECONNREFUSED | libc::ENOENT) => Ok(false),
        Some(libc::EAGAIN | libc::EINPROGRESS | libc::EALREADY) => Ok(true),
        _ => Err(error)
            .with_context(|| format!("cannot prove that socket {} is stale", path.display())),
    }
}

fn join_until(handles: Vec<JoinHandle<()>>, deadline: Instant, name: &str) -> bool {
    let mut pending = handles;
    while !pending.is_empty() && Instant::now() < deadline {
        let mut index = 0;
        while index < pending.len() {
            if pending[index].is_finished() {
                let handle = pending.swap_remove(index);
                if handle.join().is_err() {
                    eprintln!("{name} thread panicked");
                }
            } else {
                index += 1;
            }
        }
        if !pending.is_empty() {
            thread::sleep(WAIT_QUANTUM);
        }
    }
    if !pending.is_empty() {
        eprintln!("shutdown grace period expired while draining {name}");
        return false;
    }
    true
}

fn join_one_until(handle: JoinHandle<()>, deadline: Instant, name: &str) -> bool {
    while !handle.is_finished() && Instant::now() < deadline {
        thread::sleep(WAIT_QUANTUM);
    }
    if !handle.is_finished() {
        eprintln!("shutdown grace period expired while draining {name}");
        return false;
    }
    if handle.join().is_err() {
        eprintln!("{name} thread panicked");
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn socket_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("vchord-tilemaxsim-{name}-{}", std::process::id()))
    }

    #[test]
    fn stale_socket_is_removed_but_live_socket_is_preserved() {
        let path = socket_path("stale-socket");
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let error = remove_stale_socket(&path).unwrap_err();
        assert!(format!("{error:#}").contains("already accepting"));
        assert!(path.exists());
        drop(listener);
        remove_stale_socket(&path).unwrap();
        assert!(!path.exists());
    }
}
