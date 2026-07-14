# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# We welcome any commercial collaboration or support. For inquiries
# regarding the licenses, please contact us at:
# vectorchord-inquiry@tensorchord.ai
#
# Copyright (c) 2025-2026 TensorChord Inc.

"""Bounded CUDA executor for VectorChord TileMaxSim IPC v1 and v2.

Version 1 consumes inline tensors. Version 2 resolves canonical tensor payloads
from an operations-configured, per-model content-addressed cache. Applications may
populate that cache from any object store; object-store credentials and routing
never enter PostgreSQL or this protocol.

The service is disabled unless the operator explicitly assigns at least one
``GPU=GB`` arena. It acquires every configured arena before binding its socket.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import select
import signal
import socket
import stat
import struct
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from devtools import tilemaxsim_reference_sidecar as protocol
from services.tilemaxsim_gpu_cache import (
    GpuArenaSpec,
    GpuResourcePool,
    GpuTensorCache,
    GpuTensorHandle,
    parse_gpu_memory_gb,
    parse_memory_gb,
)

try:
    from services.tilemaxsim_triton import ragged_tilemaxsim_fp16
except ImportError:
    ragged_tilemaxsim_fp16 = None


def validate_finite_payload(
    payload: bytes, rows: int, dimension: int, dtype: int
) -> None:
    expected = protocol.checked_tensor_bytes(rows, dimension, dtype)
    if len(payload) != expected:
        raise protocol.SidecarError(
            protocol.STATUS_INVALID_REQUEST,
            "tensor byte length does not match its shape",
        )
    scalar_dtype = "<f4" if dtype == protocol.DTYPE_F32 else "<f2"
    if not np.isfinite(np.frombuffer(payload, dtype=scalar_dtype)).all():
        raise protocol.SidecarError(
            protocol.STATUS_INVALID_REQUEST, "tensor contains non-finite value"
        )


@dataclass(frozen=True)
class ResolvedPayload:
    payload: bytes
    cache_hit: bool


class PayloadCache:
    """Thread-safe byte-bounded LRU for checksum-verified canonical payloads."""

    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.current_bytes = 0
        self.entries: OrderedDict[tuple[object, ...], bytes] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple[object, ...]) -> bytes | None:
        if self.maximum_bytes == 0:
            return None
        with self.lock:
            payload = self.entries.get(key)
            if payload is not None:
                self.entries.move_to_end(key)
            return payload

    def put(self, key: tuple[object, ...], payload: bytes) -> None:
        if self.maximum_bytes == 0 or len(payload) > self.maximum_bytes:
            return
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.current_bytes -= len(previous)
            self.entries[key] = payload
            self.current_bytes += len(payload)
            while self.current_bytes > self.maximum_bytes:
                _, evicted = self.entries.popitem(last=False)
                self.current_bytes -= len(evicted)


class ContentAddressedResolver:
    """Resolve ``sha256://<digest>`` inside an allowlisted model cache root.

    A payload with digest ``abcdef...`` is stored as
    ``<contract-root>/ab/abcdef....bin``. Directory and file symlinks are
    rejected with ``openat(O_NOFOLLOW)``. The digest in the reference, the
    registered checksum, the exact byte length, and the file content must all
    agree before a payload is returned.
    """

    def __init__(self, roots: dict[str, Path], cache_bytes: int) -> None:
        self.root_fds: dict[str, int] = {}
        try:
            for contract, path in roots.items():
                self.root_fds[contract] = os.open(
                    os.fspath(path),
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
        except Exception:
            self.close()
            raise
        self.cache = PayloadCache(cache_bytes)

    def close(self) -> None:
        for descriptor in self.root_fds.values():
            os.close(descriptor)
        self.root_fds.clear()

    @staticmethod
    def _digest(request: protocol.ExternalTensorRequest) -> str:
        prefix = "sha256://"
        if not request.tensor_ref.startswith(prefix):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "unsupported tensor reference; expected sha256://<digest>",
            )
        digest = request.tensor_ref[len(prefix) :]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "invalid content-addressed tensor reference",
            )
        if not hmac.compare_digest(request.checksum, f"sha256:{digest}"):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "tensor reference and checksum disagree",
            )
        return digest

    @staticmethod
    def _read_exact_file(root_fd: int, digest: str, expected_bytes: int) -> bytes:
        directory_fd = -1
        payload_fd = -1
        try:
            directory_fd = os.open(
                digest[:2],
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
            payload_fd = os.open(
                f"{digest}.bin",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            metadata = os.fstat(payload_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise protocol.SidecarError(
                    protocol.STATUS_INVALID_REQUEST,
                    "resolved tensor is not a regular file",
                )
            if metadata.st_size != expected_bytes:
                raise protocol.SidecarError(
                    protocol.STATUS_INVALID_REQUEST,
                    "resolved tensor byte length does not match descriptor",
                )
            chunks = bytearray()
            while len(chunks) < expected_bytes:
                chunk = os.read(
                    payload_fd, min(1024 * 1024, expected_bytes - len(chunks))
                )
                if not chunk:
                    raise protocol.SidecarError(
                        protocol.STATUS_INVALID_REQUEST,
                        "resolved tensor file ended early",
                    )
                chunks.extend(chunk)
            if os.read(payload_fd, 1):
                raise protocol.SidecarError(
                    protocol.STATUS_INVALID_REQUEST,
                    "resolved tensor file grew during read",
                )
            return bytes(chunks)
        except FileNotFoundError as error:
            raise protocol.SidecarError(
                protocol.STATUS_COMPUTE_ERROR, "content-addressed tensor is missing"
            ) from error
        except OSError as error:
            raise protocol.SidecarError(
                protocol.STATUS_COMPUTE_ERROR,
                f"content-addressed tensor read failed: {error.strerror}",
            ) from error
        finally:
            if payload_fd >= 0:
                os.close(payload_fd)
            if directory_fd >= 0:
                os.close(directory_fd)

    def key(self, request: protocol.ExternalTensorRequest) -> tuple[object, ...]:
        root_fd = self.root_fds.get(request.model_contract_id)
        if root_fd is None:
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "model contract has no configured tensor cache root",
            )
        digest = self._digest(request)
        key = (
            request.model_contract_id,
            digest,
            request.rows,
            request.dimension,
            request.dtype,
        )
        return key

    def resolve(self, request: protocol.ExternalTensorRequest) -> ResolvedPayload:
        key = self.key(request)
        digest = str(key[1])
        root_fd = self.root_fds[request.model_contract_id]
        expected_bytes = protocol.checked_tensor_bytes(
            request.rows, request.dimension, request.dtype
        )
        cached = self.cache.get(key)
        if cached is not None:
            return ResolvedPayload(cached, True)
        payload = self._read_exact_file(root_fd, digest, expected_bytes)
        actual = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual, digest):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST, "resolved tensor checksum mismatch"
            )
        validate_finite_payload(payload, request.rows, request.dimension, request.dtype)
        self.cache.put(key, payload)
        return ResolvedPayload(payload, False)


class TorchTileMaxsimEngine:
    def __init__(
        self,
        device_name: str,
        max_device_bytes: int,
        allow_tf32: bool,
        max_cuda_inflight: int,
    ) -> None:
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        if self.device.type not in ("cuda", "cpu"):
            raise RuntimeError("device must be CUDA or CPU")
        self.max_device_bytes = max_device_bytes
        self.compute_slots = threading.BoundedSemaphore(max_cuda_inflight)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            with torch.inference_mode():
                left = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
                _ = left @ left
                torch.cuda.synchronize(self.device)

    @staticmethod
    def _cpu_tensor(
        payload: bytes, rows: int, dimension: int, dtype: int
    ) -> torch.Tensor:
        scalar_dtype = torch.float32 if dtype == protocol.DTYPE_F32 else torch.float16
        # bytearray gives torch a writable, owned buffer; clone detaches the
        # resulting tensor before that temporary buffer leaves scope.
        tensor = torch.frombuffer(bytearray(payload), dtype=scalar_dtype).reshape(
            rows, dimension
        )
        if scalar_dtype == torch.float32:
            return tensor.clone()
        return tensor.to(dtype=torch.float32)

    def _groups(
        self,
        query_rows: int,
        dimension: int,
        documents: list[tuple[int, int, bytes]],
    ) -> Iterable[list[tuple[int, int, bytes]]]:
        query_bytes = query_rows * dimension * 4
        group: list[tuple[int, int, bytes]] = []
        group_rows = 0
        for document in documents:
            rows = document[1]
            next_rows = group_rows + rows
            # Device residency includes the f32 query, f32 documents, and the
            # q-by-total-document-token similarity matrix.
            required = (
                query_bytes + next_rows * dimension * 4 + query_rows * next_rows * 4
            )
            if required > self.max_device_bytes and group:
                yield group
                group = []
                group_rows = 0
                next_rows = rows
                required = query_bytes + rows * dimension * 4 + query_rows * rows * 4
            if required > self.max_device_bytes:
                raise protocol.SidecarError(
                    protocol.STATUS_RESOURCE_LIMIT,
                    "one candidate exceeds the CUDA device-byte limit",
                )
            group.append(document)
            group_rows = next_rows
        if group:
            yield group

    def score(
        self,
        query_payload: bytes,
        query_rows: int,
        dimension: int,
        dtype: int,
        documents: list[tuple[int, int, bytes]],
        deadline: float,
        cancelled: Callable[[], bool],
    ) -> tuple[list[tuple[int, float]], float, float]:
        if not documents:
            return [], 0.0, 0.0
        query_cpu = self._cpu_tensor(query_payload, query_rows, dimension, dtype)
        results: list[tuple[int, float]] = []
        queue_started = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.compute_slots.acquire(timeout=remaining):
            raise protocol.SidecarError(
                protocol.STATUS_COMPUTE_ERROR,
                "request deadline expired while waiting for CUDA capacity",
            )
        queue_ms = (time.monotonic() - queue_started) * 1000.0
        compute_started = time.monotonic()
        try:
            with torch.inference_mode():
                if time.monotonic() >= deadline:
                    raise protocol.SidecarError(
                        protocol.STATUS_COMPUTE_ERROR, "request deadline expired"
                    )
                query_device = query_cpu.to(self.device)
                for group in self._groups(query_rows, dimension, documents):
                    if cancelled():
                        raise protocol.SidecarError(
                            protocol.STATUS_COMPUTE_ERROR, "request peer disconnected"
                        )
                    if time.monotonic() >= deadline:
                        raise protocol.SidecarError(
                            protocol.STATUS_COMPUTE_ERROR, "request deadline expired"
                        )
                    cpu_documents = [
                        self._cpu_tensor(payload, rows, dimension, dtype)
                        for _, rows, payload in group
                    ]
                    document_device = torch.cat(cpu_documents).to(self.device)
                    similarities = query_device @ document_device.transpose(0, 1)
                    scores = []
                    offset = 0
                    for _, rows, _ in group:
                        scores.append(
                            similarities[:, offset : offset + rows]
                            .amax(dim=1)
                            .sum(dtype=torch.float32)
                        )
                        offset += rows
                    host_scores = torch.stack(scores).to(device="cpu").tolist()
                    for (candidate_id, _, _), score in zip(
                        group, host_scores, strict=True
                    ):
                        if not math.isfinite(score):
                            raise protocol.SidecarError(
                                protocol.STATUS_COMPUTE_ERROR,
                                "TileMaxSim result is non-finite",
                            )
                        results.append((candidate_id, score))
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
        finally:
            self.compute_slots.release()
        return (
            results,
            queue_ms,
            (time.monotonic() - compute_started) * 1000.0,
        )


class ResidentTorchTileMaxsimEngine:
    """Score tensors already owned by one or more process GPU arenas."""

    def __init__(
        self,
        pool: GpuResourcePool,
        max_workspace_bytes: int,
        allow_tf32: bool,
        max_cuda_inflight: int,
    ) -> None:
        self.pool = pool
        self.device = pool.primary_device
        self.max_workspace_bytes = max_workspace_bytes
        self.compute_slots = threading.BoundedSemaphore(max_cuda_inflight)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        with torch.inference_mode():
            for arena in pool.arenas:
                left = torch.zeros((1, 1), dtype=torch.float32, device=arena.device)
                _ = left @ left
            for arena in pool.arenas:
                torch.cuda.synchronize(arena.device)

    @staticmethod
    def _cpu_tensor(
        payload: bytes, rows: int, dimension: int, dtype: int
    ) -> torch.Tensor:
        scalar_dtype = torch.float32 if dtype == protocol.DTYPE_F32 else torch.float16
        return torch.frombuffer(bytearray(payload), dtype=scalar_dtype).reshape(
            rows, dimension
        )

    def _groups(
        self,
        query_rows: int,
        dimension: int,
        dtype: int,
        documents: list[tuple[int, GpuTensorHandle]],
    ) -> Iterable[list[tuple[int, GpuTensorHandle]]]:
        scalar_bytes = 4 if dtype == protocol.DTYPE_F32 else 2
        query_bytes = query_rows * dimension * scalar_bytes
        group: list[tuple[int, GpuTensorHandle]] = []
        group_rows = 0
        for document in documents:
            rows = document[1].rows
            next_rows = group_rows + rows
            # The resident document remains inside the arena. torch.cat makes
            # one device-local contiguous scoring view; the other temporaries
            # are the query and q-by-document-token similarity matrix.
            required = (
                query_bytes
                + next_rows * dimension * scalar_bytes
                + query_rows * next_rows * scalar_bytes
            )
            if required > self.max_workspace_bytes and group:
                yield group
                group = []
                group_rows = 0
                next_rows = rows
                required = (
                    query_bytes
                    + rows * dimension * scalar_bytes
                    + query_rows * rows * scalar_bytes
                )
            if required > self.max_workspace_bytes:
                raise protocol.SidecarError(
                    protocol.STATUS_RESOURCE_LIMIT,
                    "one resident candidate exceeds the configured GPU workspace",
                )
            group.append(document)
            group_rows = next_rows
        if group:
            yield group

    def score(
        self,
        query_payload: bytes,
        query_rows: int,
        dimension: int,
        dtype: int,
        documents: list[tuple[int, GpuTensorHandle]],
        deadline: float,
        cancelled: Callable[[], bool],
    ) -> tuple[list[tuple[int, float]], float, float]:
        if not documents:
            return [], 0.0, 0.0
        query_cpu = self._cpu_tensor(query_payload, query_rows, dimension, dtype)
        by_device: dict[str, list[tuple[int, GpuTensorHandle]]] = {}
        for document in documents:
            handle = document[1]
            if handle.dimension != dimension or handle.dtype != dtype:
                raise protocol.SidecarError(
                    protocol.STATUS_INVALID_REQUEST,
                    "resident tensor contract disagrees with the query",
                )
            by_device.setdefault(str(handle.device), []).append(document)

        queue_started = time.monotonic()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.compute_slots.acquire(timeout=remaining):
            raise protocol.SidecarError(
                protocol.STATUS_COMPUTE_ERROR,
                "request deadline expired while waiting for CUDA capacity",
            )
        queue_ms = (time.monotonic() - queue_started) * 1000.0
        compute_started = time.monotonic()
        pending: list[tuple[list[tuple[int, GpuTensorHandle]], torch.Tensor]] = []
        try:
            with torch.inference_mode():
                for arena in self.pool.arenas:
                    device_documents = by_device.get(str(arena.device), [])
                    if not device_documents:
                        continue
                    if cancelled():
                        raise protocol.SidecarError(
                            protocol.STATUS_COMPUTE_ERROR,
                            "request peer disconnected",
                        )
                    if time.monotonic() >= deadline:
                        raise protocol.SidecarError(
                            protocol.STATUS_COMPUTE_ERROR, "request deadline expired"
                        )
                    query_device = query_cpu.to(arena.device)
                    if dtype == protocol.DTYPE_F16 and ragged_tilemaxsim_fp16:
                        offsets = torch.tensor(
                            [
                                handle.offset_bytes // 2
                                for _, handle in device_documents
                            ],
                            dtype=torch.int64,
                            device=arena.device,
                        )
                        rows = torch.tensor(
                            [handle.rows for _, handle in device_documents],
                            dtype=torch.int32,
                            device=arena.device,
                        )
                        assert arena.storage is not None
                        device_scores = ragged_tilemaxsim_fp16(
                            query_device,
                            arena.storage.view(torch.float16),
                            offsets,
                            rows,
                            max(handle.rows for _, handle in device_documents),
                        )
                        pending.append((device_documents, device_scores))
                        continue
                    for group in self._groups(
                        query_rows, dimension, dtype, device_documents
                    ):
                        document_device = torch.cat(
                            [handle.tensor() for _, handle in group]
                        )
                        similarities = query_device @ document_device.transpose(0, 1)
                        scores = []
                        offset = 0
                        for _, handle in group:
                            scores.append(
                                similarities[:, offset : offset + handle.rows]
                                .amax(dim=1)
                                .sum(dtype=torch.float32)
                            )
                            offset += handle.rows
                        pending.append((group, torch.stack(scores)))

                results: list[tuple[int, float]] = []
                for group, device_scores in pending:
                    host_scores = device_scores.to(device="cpu", dtype=torch.float32)
                    for (candidate_id, _), score in zip(
                        group, host_scores.tolist(), strict=True
                    ):
                        if not math.isfinite(score):
                            raise protocol.SidecarError(
                                protocol.STATUS_COMPUTE_ERROR,
                                "TileMaxSim result is non-finite",
                            )
                        results.append((candidate_id, score))
                for arena in self.pool.arenas:
                    torch.cuda.synchronize(arena.device)
        finally:
            self.compute_slots.release()
        return results, queue_ms, (time.monotonic() - compute_started) * 1000.0


class JsonMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def emit(self, fields: dict[str, object]) -> None:
        with self.lock:
            print(json.dumps(fields, separators=(",", ":"), sort_keys=True), flush=True)


class TileMaxsimService:
    def __init__(
        self,
        limits: protocol.Limits,
        resolver: ContentAddressedResolver,
        engine: TorchTileMaxsimEngine,
        request_timeout_ms: int,
        metrics: JsonMetrics,
        gpu_cache: GpuTensorCache | None = None,
        resident_engine: ResidentTorchTileMaxsimEngine | None = None,
        pin_gpu_entries: bool = False,
    ) -> None:
        self.limits = limits
        self.resolver = resolver
        self.engine = engine
        self.request_timeout_seconds = request_timeout_ms / 1000.0
        self.metrics = metrics
        self.gpu_cache = gpu_cache
        self.resident_engine = resident_engine
        self.pin_gpu_entries = pin_gpu_entries
        if (gpu_cache is None) != (resident_engine is None):
            raise ValueError(
                "GPU cache and resident engine must be configured together"
            )

    @staticmethod
    def _peer_disconnected(connection: socket.socket) -> bool:
        poller = select.poll()
        poller.register(connection, select.POLLHUP | select.POLLERR | select.POLLNVAL)
        return bool(poller.poll(0))

    @staticmethod
    def _receive_exact_until(
        connection: socket.socket, count: int, deadline: float
    ) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired during socket read")
            connection.settimeout(remaining)
            chunk = connection.recv(count - len(chunks))
            if not chunk:
                raise protocol.SidecarError(
                    protocol.STATUS_INVALID_REQUEST,
                    "connection closed during request",
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def process_frame(
        self,
        frame: bytes,
        connection: socket.socket,
        deadline: float,
        peer_credentials: tuple[int, int, int] | None,
    ) -> bytes:
        request_id = 0
        version = protocol.VERSION
        started = time.monotonic()
        metrics: dict[str, object] = {"event": "tilemaxsim_request"}
        resident_documents: list[tuple[int, GpuTensorHandle]] = []
        if peer_credentials is not None:
            metrics["peer_pid"], metrics["peer_uid"], metrics["peer_gid"] = (
                peer_credentials
            )
        try:
            if len(frame) >= protocol.HEADER.size:
                _, wire_version, _, request_id, _ = protocol.HEADER.unpack_from(frame)
                if wire_version in (protocol.VERSION, protocol.EXTERNAL_VERSION):
                    version = wire_version
            request = protocol.parse_request_frame(
                frame, self.limits, validate_finite=False
            )
            metrics.update(
                request_id=request.request_id,
                protocol_version=version,
                query_rows=request.query_rows,
                dimension=request.dimension,
                candidate_count=len(request.candidates),
            )
            validate_finite_payload(
                request.query_payload,
                request.query_rows,
                request.dimension,
                request.dtype,
            )
            resolve_started = time.monotonic()
            cache_hits = 0
            gpu_cache_hits = 0
            gpu_cache_misses = 0
            gpu_chunks = 0
            resident_results: list[tuple[int, float]] = []
            resident_queue_ms = 0.0
            resident_compute_ms = 0.0
            document_tokens = 0
            documents: list[tuple[int, int, bytes]] = []

            def flush_resident_documents() -> None:
                nonlocal gpu_chunks, resident_queue_ms, resident_compute_ms
                if not resident_documents:
                    return
                assert self.resident_engine is not None
                try:
                    batch_results, batch_queue_ms, batch_compute_ms = (
                        self.resident_engine.score(
                            request.query_payload,
                            request.query_rows,
                            request.dimension,
                            request.dtype,
                            resident_documents,
                            deadline,
                            lambda: self._peer_disconnected(connection),
                        )
                    )
                    resident_results.extend(batch_results)
                    resident_queue_ms += batch_queue_ms
                    resident_compute_ms += batch_compute_ms
                    gpu_chunks += 1
                finally:
                    assert self.gpu_cache is not None
                    for _, resident_handle in resident_documents:
                        self.gpu_cache.release(resident_handle)
                    resident_documents.clear()

            if isinstance(request, protocol.InlineTensorRequest):
                for candidate in request.candidates:
                    validate_finite_payload(
                        candidate.payload,
                        candidate.rows,
                        request.dimension,
                        request.dtype,
                    )
                documents = [
                    (candidate.candidate_id, candidate.rows, candidate.payload)
                    for candidate in request.candidates
                ]
                document_tokens = sum(
                    candidate.rows for candidate in request.candidates
                )
                metrics["source"] = "inline"
            else:
                metrics["source"] = "content_addressed"
                for candidate in request.candidates:
                    document_tokens += candidate.descriptor.rows
                    if time.monotonic() >= deadline:
                        raise protocol.SidecarError(
                            protocol.STATUS_COMPUTE_ERROR,
                            "request deadline expired during tensor resolution",
                        )
                    if self.gpu_cache is None:
                        resolved = self.resolver.resolve(candidate.descriptor)
                        cache_hits += int(resolved.cache_hit)
                        documents.append(
                            (
                                candidate.candidate_id,
                                candidate.descriptor.rows,
                                resolved.payload,
                            )
                        )
                    else:
                        key = self.resolver.key(candidate.descriptor)
                        loaded_payload: bytes | None = None

                        def load_payload() -> bytes:
                            nonlocal cache_hits, loaded_payload
                            if loaded_payload is None:
                                resolved = self.resolver.resolve(candidate.descriptor)
                                cache_hits += int(resolved.cache_hit)
                                loaded_payload = resolved.payload
                            return loaded_payload

                        while True:
                            try:
                                handle, gpu_hit = self.gpu_cache.acquire(
                                    key,
                                    candidate.descriptor.rows,
                                    candidate.descriptor.dimension,
                                    candidate.descriptor.dtype,
                                    load_payload,
                                    pin=self.pin_gpu_entries,
                                )
                                break
                            except protocol.SidecarError as error:
                                if (
                                    error.status != protocol.STATUS_RESOURCE_LIMIT
                                    or not resident_documents
                                ):
                                    raise
                                # A request may be larger than the configured
                                # GPU cache. Score and release the current
                                # working set, then admit the remaining
                                # candidates through the same bounded arenas.
                                flush_resident_documents()
                        gpu_cache_hits += int(gpu_hit)
                        gpu_cache_misses += int(not gpu_hit)
                        resident_documents.append((candidate.candidate_id, handle))
                if self.gpu_cache is not None:
                    flush_resident_documents()
            metrics["cache_hits"] = cache_hits
            metrics["host_cache_hits"] = cache_hits
            metrics["gpu_cache_hits"] = gpu_cache_hits
            metrics["gpu_cache_misses"] = gpu_cache_misses
            metrics["gpu_chunks"] = gpu_chunks
            metrics["resolve_ms"] = round(
                max(
                    0.0,
                    (time.monotonic() - resolve_started) * 1000.0
                    - resident_queue_ms
                    - resident_compute_ms,
                ),
                3,
            )
            metrics["document_tokens"] = document_tokens
            if self.gpu_cache is not None and isinstance(
                request, protocol.ParsedExternalTensorRequest
            ):
                results = resident_results
                queue_ms = resident_queue_ms
                compute_ms = resident_compute_ms
            else:
                results, queue_ms, compute_ms = self.engine.score(
                    request.query_payload,
                    request.query_rows,
                    request.dimension,
                    request.dtype,
                    documents,
                    deadline,
                    lambda: self._peer_disconnected(connection),
                )
            metrics["queue_ms"] = round(queue_ms, 3)
            metrics["compute_ms"] = round(compute_ms, 3)
            metrics["status"] = "ok"
            return protocol.success_response(request.request_id, results, version)
        except protocol.SidecarError as error:
            metrics.update(status="error", error_class=type(error).__name__)
            return protocol.error_response(
                request_id, error.status, str(error), version
            )
        except torch.OutOfMemoryError:
            metrics.update(status="error", error_class="CudaOutOfMemory")
            return protocol.error_response(
                request_id,
                protocol.STATUS_RESOURCE_LIMIT,
                "CUDA out of memory",
                version,
            )
        except Exception as error:
            metrics.update(status="error", error_class=type(error).__name__)
            return protocol.error_response(
                request_id,
                protocol.STATUS_COMPUTE_ERROR,
                f"TileMaxSim compute failed: {error}",
                version,
            )
        finally:
            if self.gpu_cache is not None:
                for _, handle in resident_documents:
                    self.gpu_cache.release(handle)
            metrics["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
            self.metrics.emit(metrics)

    def handle(self, connection: socket.socket) -> None:
        deadline = time.monotonic() + self.request_timeout_seconds
        request_id = 0
        version = protocol.VERSION
        peer_credentials = None
        if hasattr(socket, "SO_PEERCRED"):
            try:
                raw_credentials = connection.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                )
                peer_credentials = struct.unpack("3i", raw_credentials)
            except OSError:
                pass
        try:
            header = self._receive_exact_until(
                connection, protocol.HEADER.size, deadline
            )
            _, wire_version, _, request_id, body_len = protocol.HEADER.unpack(header)
            if wire_version in (protocol.VERSION, protocol.EXTERNAL_VERSION):
                version = wire_version
            if body_len > self.limits.max_request_bytes - protocol.HEADER.size:
                response = protocol.error_response(
                    request_id,
                    protocol.STATUS_RESOURCE_LIMIT,
                    "request exceeds byte limit",
                    version,
                )
            else:
                body = self._receive_exact_until(connection, body_len, deadline)
                response = self.process_frame(
                    header + body, connection, deadline, peer_credentials
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired before socket write")
            connection.settimeout(remaining)
            connection.sendall(response)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (TimeoutError, socket.timeout):
            try:
                connection.sendall(
                    protocol.error_response(
                        request_id,
                        protocol.STATUS_COMPUTE_ERROR,
                        "request deadline expired during socket I/O",
                        version,
                    )
                )
            except OSError:
                pass
        except Exception as error:
            try:
                connection.sendall(
                    protocol.error_response(
                        request_id,
                        protocol.STATUS_COMPUTE_ERROR,
                        str(error),
                        version,
                    )
                )
            except OSError:
                pass


def serve(
    socket_path: Path,
    socket_mode: int,
    backlog: int,
    max_inflight: int,
    service: TileMaxsimService,
    stop: threading.Event,
    once: bool = False,
) -> None:
    protocol.remove_stale_socket(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    slots = threading.BoundedSemaphore(max_inflight)
    workers = ThreadPoolExecutor(
        max_workers=max_inflight, thread_name_prefix="tilemaxsim"
    )
    active = 0
    active_lock = threading.Lock()

    def handle(connection: socket.socket) -> None:
        nonlocal active
        try:
            with connection:
                service.handle(connection)
        finally:
            with active_lock:
                active -= 1
            slots.release()

    try:
        listener.bind(os.fspath(socket_path))
        os.chmod(socket_path, socket_mode)
        listener.listen(backlog)
        listener.settimeout(0.25)
        bound_identity = socket_path.lstat().st_dev, socket_path.lstat().st_ino
        ready: dict[str, object] = {
            "event": "tilemaxsim_ready",
            "device": str(service.engine.device),
            "max_inflight": max_inflight,
            "socket": os.fspath(socket_path),
        }
        if service.gpu_cache is not None:
            ready["gpu_cache"] = service.gpu_cache.status()
        service.metrics.emit(ready)
        accepted = 0
        while not stop.is_set():
            if not slots.acquire(timeout=0.25):
                continue
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                slots.release()
                continue
            with active_lock:
                active += 1
                current_active = active
            service.metrics.emit(
                {"event": "tilemaxsim_accept", "inflight": current_active}
            )
            workers.submit(handle, connection)
            accepted += 1
            if once and accepted == 1:
                break
    finally:
        listener.close()
        workers.shutdown(wait=True, cancel_futures=False)
        try:
            current = socket_path.lstat()
            if (current.st_dev, current.st_ino) == bound_identity:
                socket_path.unlink()
        except (FileNotFoundError, UnboundLocalError):
            pass


def parse_mode(value: str) -> int:
    mode = int(value, 8)
    if mode < 0 or mode > 0o777:
        raise argparse.ArgumentTypeError("socket mode must be between 000 and 777")
    return mode


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def memory_gb(value: str) -> int:
    try:
        return parse_memory_gb(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def gpu_memory_gb(value: str) -> GpuArenaSpec:
    try:
        return parse_gpu_memory_gb(value)
    except (ValueError, RuntimeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def contract_roots(
    values: list[str], parser: argparse.ArgumentParser
) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            parser.error("--contract-root must be MODEL_CONTRACT_ID=/absolute/path")
        contract, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not contract or not path.is_absolute():
            parser.error("--contract-root must contain a nonempty ID and absolute path")
        if contract in roots:
            parser.error(f"duplicate --contract-root for {contract!r}")
        roots[contract] = path
    return roots


def contract_manifests(
    values: list[str], parser: argparse.ArgumentParser
) -> list[tuple[str, Path]]:
    manifests = []
    for value in values:
        if "=" not in value:
            parser.error("--resident-manifest must be MODEL_CONTRACT_ID=/absolute/path")
        contract, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not contract or not path.is_absolute():
            parser.error(
                "--resident-manifest must contain a nonempty ID and absolute path"
            )
        manifests.append((contract, path))
    return manifests


def prewarm_resident_cache(
    manifests: list[tuple[str, Path]],
    resolver: ContentAddressedResolver,
    gpu_cache: GpuTensorCache,
    metrics: JsonMetrics,
) -> None:
    completed = 0
    loaded_bytes = 0
    started = time.monotonic()
    for contract, path in manifests:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                dtype_name = record.get("tensor_dtype")
                if dtype_name == "float16":
                    dtype = protocol.DTYPE_F16
                elif dtype_name == "float32":
                    dtype = protocol.DTYPE_F32
                else:
                    raise ValueError(f"{path}:{line_number}: unsupported tensor_dtype")
                tensor_ref = record.get("tensor_ref")
                checksum = record.get("tensor_checksum")
                rows = record.get("tensor_rows")
                dimension = record.get("tensor_dim")
                if not isinstance(tensor_ref, str) or not tensor_ref:
                    raise ValueError(f"{path}:{line_number}: invalid tensor_ref")
                if not isinstance(checksum, str) or not checksum:
                    raise ValueError(f"{path}:{line_number}: invalid tensor_checksum")
                if not isinstance(rows, int) or rows <= 0:
                    raise ValueError(f"{path}:{line_number}: invalid tensor_rows")
                if not isinstance(dimension, int) or dimension <= 0:
                    raise ValueError(f"{path}:{line_number}: invalid tensor_dim")
                request = protocol.ExternalTensorRequest(
                    contract, tensor_ref, rows, dimension, dtype, checksum
                )
                expected_bytes = protocol.checked_tensor_bytes(rows, dimension, dtype)
                declared_bytes = record.get("canonical_bytes")
                if declared_bytes is not None and declared_bytes != expected_bytes:
                    raise ValueError(
                        f"{path}:{line_number}: canonical_bytes disagrees with shape"
                    )
                key = resolver.key(request)
                handle, _ = gpu_cache.acquire(
                    key,
                    rows,
                    dimension,
                    dtype,
                    lambda request=request: resolver.resolve(request).payload,
                    pin=True,
                )
                gpu_cache.release(handle)
                completed += 1
                loaded_bytes += expected_bytes
                if completed % 1000 == 0:
                    metrics.emit(
                        {
                            "event": "tilemaxsim_prewarm_progress",
                            "entries": completed,
                            "logical_bytes": loaded_bytes,
                        }
                    )
    if completed == 0:
        raise ValueError("resident manifests contain no tensor descriptors")
    metrics.emit(
        {
            "event": "tilemaxsim_prewarm_complete",
            "entries": completed,
            "logical_bytes": loaded_bytes,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            "gpu_cache": gpu_cache.status(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--socket-mode", type=parse_mode, default=0o600)
    parser.add_argument(
        "--gpu-cache-mode",
        choices=("lru", "resident"),
        default="lru",
        help="use evictable GPU arenas or pin a full descriptor manifest",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        action="append",
        type=gpu_memory_gb,
        default=[],
        metavar="GPU=GB",
        help="repeatable strict allocation, for example 1=20; enables TileMaxSim",
    )
    parser.add_argument(
        "--gpu-workspace-gb",
        type=memory_gb,
        default=2 * 1024**3,
        help="per-GPU portion of the configured GB reserved for scoring temporaries",
    )
    parser.add_argument("--contract-root", action="append", default=[])
    parser.add_argument(
        "--resident-manifest",
        action="append",
        default=[],
        metavar="MODEL_CONTRACT_ID=/ABSOLUTE/PATH",
    )
    parser.add_argument(
        "--max-request-bytes", type=positive_int, default=64 * 1024 * 1024
    )
    parser.add_argument("--max-batch-tokens", type=positive_int, default=1_000_000)
    parser.add_argument(
        "--max-tensor-bytes", type=positive_int, default=1024 * 1024 * 1024
    )
    parser.add_argument("--max-candidates", type=positive_int, default=65_536)
    parser.add_argument(
        "--host-cache-gb",
        type=memory_gb,
        default=8 * 1024**3,
        help="decoded host-memory tensor cache size in GB",
    )
    parser.add_argument("--request-timeout-ms", type=positive_int, default=2000)
    parser.add_argument("--max-inflight", type=positive_int, default=8)
    parser.add_argument("--max-cuda-inflight", type=positive_int, default=1)
    parser.add_argument("--backlog", type=positive_int, default=64)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    roots = contract_roots(args.contract_root, parser)
    manifests = contract_manifests(args.resident_manifest, parser)
    if not args.gpu_memory_gb:
        parser.error(
            "TileMaxSim is disabled until at least one --gpu-memory-gb GPU=GB is configured"
        )
    if args.gpu_cache_mode == "resident" and not manifests:
        parser.error(
            "--gpu-cache-mode resident requires at least one --resident-manifest"
        )
    if args.gpu_cache_mode == "lru" and manifests:
        parser.error("--resident-manifest is valid only in resident mode")
    limits = protocol.Limits(
        max_request_bytes=args.max_request_bytes,
        max_batch_tokens=args.max_batch_tokens,
        max_tensor_bytes=args.max_tensor_bytes,
        max_candidates=args.max_candidates,
    )
    resolver = ContentAddressedResolver(roots, args.host_cache_gb)
    metrics = JsonMetrics()
    pool: GpuResourcePool | None = None
    try:
        pool = GpuResourcePool(args.gpu_memory_gb, args.gpu_workspace_gb)
        engine = TorchTileMaxsimEngine(
            str(pool.primary_device),
            args.gpu_workspace_gb,
            args.allow_tf32,
            args.max_cuda_inflight,
        )
        gpu_cache = GpuTensorCache(pool, allow_eviction=args.gpu_cache_mode == "lru")
        resident_engine = ResidentTorchTileMaxsimEngine(
            pool,
            args.gpu_workspace_gb,
            args.allow_tf32,
            args.max_cuda_inflight,
        )
        if args.gpu_cache_mode == "resident":
            prewarm_resident_cache(manifests, resolver, gpu_cache, metrics)
        service = TileMaxsimService(
            limits,
            resolver,
            engine,
            args.request_timeout_ms,
            metrics,
            gpu_cache,
            resident_engine,
            pin_gpu_entries=args.gpu_cache_mode == "resident",
        )
        stop = threading.Event()

        def request_stop(_signum: int, _frame: object) -> None:
            stop.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        serve(
            args.socket,
            args.socket_mode,
            args.backlog,
            args.max_inflight,
            service,
            stop,
            args.once,
        )
    finally:
        if pool is not None:
            pool.close()
        resolver.close()


if __name__ == "__main__":
    main()
