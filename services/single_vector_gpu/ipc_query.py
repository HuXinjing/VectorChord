"""Single-use, lease-bound FP32 queries received from a local CUDA producer."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as c
import errno
import fcntl
from hashlib import sha256
from hmac import compare_digest
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time

import torch


FRAME = struct.Struct("!4s16sIII64s64s32s")
MAX_DIMENSIONS = 65536


class IpcUnavailable(RuntimeError):
    """Expected transport failure safe to fall back from."""

    def __init__(self, reason):
        super().__init__("query_transport_unavailable")
        self.reason = reason


class _IpcHandle(c.Structure):
    _fields_ = [("data", c.c_ubyte * 64)]


class IpcQuery:
    def __init__(self, dimension, memory, ready, lease, peer):
        self.dimension = dimension
        self.memory, self.ready = memory, ready
        self.lease, self.peer = lease, peer
        self.expires = time.monotonic() + 5
        self.can_release = True

    def copy_to(self, row):
        if row.shape != (self.dimension,) or row.dtype != torch.float32 or not row.is_cuda:
            raise ValueError("ipc_query_destination")
        driver = c.CDLL("libcuda.so.1")
        source, event = c.c_uint64(), c.c_void_p()

        def call(name, *args):
            if status := getattr(driver, name)(*args):
                raise RuntimeError(f"{name}: CUDA driver status {status}")

        with torch.cuda.device(row.device):
            stream = torch.cuda.current_stream(row.device)
            stream_ptr = c.c_void_p(stream.cuda_stream)
            call("cuIpcOpenMemHandle_v2", c.byref(source),
                 _IpcHandle.from_buffer_copy(self.memory), c.c_uint(1))
            self.can_release = False
            try:
                base, capacity = c.c_uint64(), c.c_size_t()
                call("cuMemGetAddressRange_v2", c.byref(base), c.byref(capacity), source)
                if base.value != source.value or capacity.value < self.dimension * 4:
                    raise ValueError("ipc_query_allocation")
                if any(self.ready):
                    call("cuIpcOpenEventHandle", c.byref(event),
                         _IpcHandle.from_buffer_copy(self.ready))
                    call("cuStreamWaitEvent", stream_ptr, event, c.c_uint(0))
                call("cuMemcpyDtoDAsync_v2", c.c_uint64(row.data_ptr()),
                     source, c.c_size_t(self.dimension * 4), stream_ptr)
            finally:
                # A failed synchronization leaves the producer lease pinned.
                # Subsequent GPU requests fall back to the existing CPU route.
                stream.synchronize()
                try:
                    if event.value:
                        call("cuEventDestroy_v2", event)
                finally:
                    call("cuIpcCloseMemHandle", source)
                    self.can_release = True

    def close(self):
        try:
            if self.can_release:
                self.peer.sendall(b"R")
        except OSError:
            pass
        finally:
            self.peer.close()


class IpcInbox:
    def __init__(self, socket_path: str):
        self.path = Path(socket_path)
        if not self.path.is_absolute():
            raise ValueError("ipc_socket_path")
        self.lock = threading.Condition()
        self.entries: dict[bytes, IpcQuery] = {}
        self.stop = threading.Event()
        self.lock_fd = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._recover_stale_socket()
            self.listener.bind(socket_path)
            bound = True
            self.path.chmod(0o600)
            self.listener.listen(16)
            self.listener.settimeout(.1)
        except BaseException:
            self.listener.close()
            if bound:
                self.path.unlink(missing_ok=True)
            os.close(self.lock_fd)
            raise
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _recover_stale_socket(self):
        try:
            node = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(node.st_mode) or node.st_uid != os.geteuid():
            raise ValueError("ipc_socket_owned_by_another_process")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(.1)
            try:
                probe.connect(str(self.path))
            except OSError as error:
                if error.errno != errno.ECONNREFUSED:
                    raise
            else:
                raise RuntimeError("ipc_socket_in_use")
        if self.path.lstat() != node:
            raise RuntimeError("ipc_socket_changed")
        self.path.unlink()

    def _read(self, peer):
        peer.settimeout(.1)
        with peer.makefile("rb") as wire:
            raw = wire.read(FRAME.size)
        magic, ref, rows, cols, nbytes, memory, ready, lease = FRAME.unpack(raw)
        if (magic != b"SVQ1" or not any(ref) or rows != 1 or not 0 < cols <= MAX_DIMENSIONS
                or nbytes != cols * 4 or not any(lease)):
            raise ValueError("ipc_query_frame")
        with self.lock:
            if len(self.entries) >= 64 or ref in self.entries:
                raise ValueError("ipc_query_capacity")
            self.entries[ref] = IpcQuery(cols, memory, ready, lease, peer)
            self.lock.notify_all()

    def _serve(self):
        while not self.stop.is_set():
            try:
                peer, _ = self.listener.accept()
            except OSError:
                pass
            else:
                try:
                    self._read(peer)
                except (OSError, struct.error, ValueError):
                    try:
                        peer.sendall(b"R")  # No CUDA mapping was opened.
                    except OSError:
                        pass
                    peer.close()
            with self.lock:
                now = time.monotonic()
                expired = [self.entries.pop(ref) for ref, item in list(self.entries.items())
                           if item.expires <= now]
            for item in expired:
                item.close()

    @contextmanager
    def claim(self, references, leases, dimension):
        if len(references) != len(set(references)):
            raise ValueError("duplicate_ipc_query")
        with self.lock:
            # The producer sends before returning its control-plane reference.
            self.lock.wait_for(lambda: all(ref in self.entries for ref in references), .05)
            items = [self.entries.get(ref) for ref in references]
            now = time.monotonic()
            if any(item is None or item.expires <= now for item in items):
                raise IpcUnavailable("reference_expired_or_missing")
            if any(not compare_digest(item.lease, sha256(lease.encode()).digest())
                   for item, lease in zip(items, leases, strict=True)):
                raise PermissionError("ipc_query_lease_mismatch")
            if any(item.dimension != dimension for item in items):
                raise ValueError("ipc_query_dimension_mismatch")
            for ref in references:
                self.entries.pop(ref)
        try:
            yield items
        finally:
            for item in items:
                item.close()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.listener.close()
        self.path.unlink(missing_ok=True)
        os.close(self.lock_fd)
        with self.lock:
            for item in self.entries.values():
                item.close()
            self.entries.clear()
