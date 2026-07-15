# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Run storage, cache, H2D, and native-daemon TileMaxSim ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import select
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import decode_response
from services.tilemaxsim_cuda_sidecar import ContentAddressedResolver, PayloadCache
from services.tilemaxsim_gpu_cache import (
    FixedBlockAllocator,
    FreeExtentAllocator,
    GpuArenaSpec,
    GpuResourcePool,
    GpuTensorCache,
    GpuTensorLoad,
)


class _LegacyBuddyAllocator:
    """Former power-of-two allocator retained only as an ablation baseline."""

    def __init__(self, capacity: int, block_bytes: int = 256 * 1024) -> None:
        self.block_bytes = block_bytes
        self.block_count = capacity // block_bytes
        self.capacity = self.block_count * block_bytes
        self._free: dict[int, set[tuple[int, int, int]]] = {}
        self._allocated: dict[int, tuple[int, int, int]] = {}
        start = 0
        remaining = self.block_count
        while remaining:
            order = remaining.bit_length() - 1
            size = 1 << order
            self._free.setdefault(order, set()).add((start, start, order))
            start += size
            remaining -= size

    @property
    def free_bytes(self) -> int:
        return sum(
            len(items) * (1 << order) * self.block_bytes
            for order, items in self._free.items()
        )

    def allocation_bytes(self, payload_bytes: int) -> int:
        raw = math.ceil(payload_bytes / self.block_bytes)
        return (1 << (raw - 1).bit_length()) * self.block_bytes

    def allocate(self, payload_bytes: int) -> tuple[int, ...] | None:
        required = self.allocation_bytes(payload_bytes) // self.block_bytes
        order = required.bit_length() - 1
        available_order = next(
            (
                candidate
                for candidate in range(order, self.block_count.bit_length())
                if self._free.get(candidate)
            ),
            None,
        )
        if available_order is None:
            return None
        start, root_start, root_order = self._free[available_order].pop()
        while available_order > order:
            available_order -= 1
            buddy = start + (1 << available_order)
            self._free.setdefault(available_order, set()).add(
                (buddy, root_start, root_order)
            )
        self._allocated[start] = (order, root_start, root_order)
        return tuple(range(start, start + required))

    def release(self, blocks: tuple[int, ...]) -> None:
        start = blocks[0]
        order, root_start, root_order = self._allocated.pop(start)
        while order < root_order:
            buddy = root_start + ((start - root_start) ^ (1 << order))
            item = (buddy, root_start, root_order)
            free = self._free.setdefault(order, set())
            if item not in free:
                break
            free.remove(item)
            start = min(start, buddy)
            order += 1
        self._free.setdefault(order, set()).add((start, root_start, root_order))


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def request(record: dict[str, object], contract: str) -> protocol.ExternalTensorRequest:
    dtype = (
        protocol.DTYPE_F16
        if record["tensor_dtype"] == "float16"
        else protocol.DTYPE_F32
    )
    return protocol.ExternalTensorRequest(
        contract,
        str(record["tensor_ref"]),
        int(record["tensor_rows"]),
        int(record["tensor_dim"]),
        dtype,
        str(record["tensor_checksum"]),
    )


def evict_paths(paths: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def storage_ablation(
    selected: list[dict[str, object]],
    contract: str,
    legacy_root: Path,
    shard_root: Path,
) -> dict[str, object]:
    requests = [request(record, contract) for record in selected]
    legacy_paths = [
        legacy_root
        / str(record["tensor_checksum"])[7:9]
        / f"{str(record['tensor_checksum'])[7:]}.bin"
        for record in selected
    ]
    shard_paths = sorted((shard_root / "shards").glob("*.vts"))

    evict_paths(legacy_paths)
    resolver = ContentAddressedResolver({contract: legacy_root}, 0)
    started = time.perf_counter()
    try:
        sequential = [resolver.resolve(item) for item in requests]
    finally:
        resolver.close()
    sequential_ms = (time.perf_counter() - started) * 1000

    evict_paths(legacy_paths)
    resolver = ContentAddressedResolver({contract: legacy_root}, 0)
    started = time.perf_counter()
    try:
        legacy_batch = resolver.resolve_many(requests)
    finally:
        resolver.close()
    legacy_batch_ms = (time.perf_counter() - started) * 1000

    evict_paths(shard_paths)
    resolver = ContentAddressedResolver({contract: shard_root}, 0)
    started = time.perf_counter()
    try:
        shard_batch = resolver.resolve_many(requests)
        shard_status = resolver.status()
    finally:
        resolver.close()
    shard_batch_ms = (time.perf_counter() - started) * 1000
    expected = [item.payload for item in sequential]
    if expected != [item.payload for item in legacy_batch] or expected != [
        item.payload for item in shard_batch
    ]:
        raise RuntimeError("storage ablation payloads disagree")
    return {
        "candidates": len(selected),
        "logical_bytes": sum(len(item) for item in expected),
        "legacy_sequential_ms": sequential_ms,
        "legacy_batch_ms": legacy_batch_ms,
        "shard_batch_ms": shard_batch_ms,
        "shard_batch_read_calls": shard_status["batch_read_calls"],
        "shard_batch_read_bytes": shard_status["batch_read_bytes"],
    }


def h2d_ablation(
    selected: list[dict[str, object]], contract: str, shard_root: Path, device: int
) -> dict[str, object]:
    requests = [request(record, contract) for record in selected]
    resolver = ContentAddressedResolver({contract: shard_root}, 0)
    try:
        payloads = resolver.resolve_many(requests)
        keys = [resolver.key(item) for item in requests]
    finally:
        resolver.close()
    total_bytes = 768 * 1024**2
    workspace_bytes = 256 * 1024**2

    pool = GpuResourcePool(
        [GpuArenaSpec(f"cuda:{device}", total_bytes)], workspace_bytes
    )
    try:
        cache = GpuTensorCache(pool, allow_eviction=True)
        started = time.perf_counter()
        handles = []
        for key, item, resolved in zip(keys, requests, payloads, strict=True):
            handle, _ = cache.acquire(
                key,
                item.rows,
                item.dimension,
                item.dtype,
                lambda payload=resolved.payload: payload,
            )
            handles.append(handle)
        torch.cuda.synchronize(device)
        sequential_ms = (time.perf_counter() - started) * 1000
        sequential_status = pool.status()[0]
        for handle in handles:
            cache.release(handle)
    finally:
        pool.close()

    pool = GpuResourcePool(
        [GpuArenaSpec(f"cuda:{device}", total_bytes)], workspace_bytes
    )
    try:
        cache = GpuTensorCache(pool, allow_eviction=True)
        loads = [
            GpuTensorLoad(key, item.rows, item.dimension, item.dtype, resolved.payload)
            for key, item, resolved in zip(keys, requests, payloads, strict=True)
        ]
        started = time.perf_counter()
        batch = cache.acquire_many(loads)
        torch.cuda.synchronize(device)
        batch_ms = (time.perf_counter() - started) * 1000
        batch_status = pool.status()[0]
        if batch.bypassed or batch.deferred:
            raise RuntimeError("H2D ablation cache is undersized")
        for handle in batch.handles:
            assert handle is not None
            cache.release(handle)
    finally:
        pool.close()
    return {
        "candidates": len(selected),
        "logical_bytes": sum(len(item.payload) for item in payloads),
        "per_tensor_h2d_ms": sequential_ms,
        "batch_h2d_ms": batch_ms,
        "per_tensor_copy_batches": sequential_status["h2d_batches"],
        "batch_copy_batches": batch_status["h2d_batches"],
        "batch_copy_calls": batch_status["h2d_copy_calls"],
    }


def allocator_ablation(records: list[dict[str, object]]) -> dict[str, object]:
    sizes = [int(record["canonical_bytes"]) for record in records]
    rng = random.Random(991)
    capacity = 256 * 1024**2
    events: list[tuple[str, int, int]] = []
    abstract_live: list[int] = []
    next_identifier = 0
    for _ in range(20_000):
        if abstract_live and rng.random() < 0.48:
            index = rng.randrange(len(abstract_live))
            identifier = abstract_live.pop(index)
            events.append(("release", identifier, 0))
        else:
            size = rng.choice(sizes)
            identifier = next_identifier
            next_identifier += 1
            abstract_live.append(identifier)
            events.append(("allocate", identifier, size))

    def run_trace(
        allocator: FreeExtentAllocator | FixedBlockAllocator | _LegacyBuddyAllocator,
    ) -> dict[str, object]:
        live: dict[int, tuple[int, ...] | tuple[int, int]] = {}
        failures = 0
        fragmentation_failures = 0
        requested_bytes = 0
        allocated_bytes = 0
        started = time.perf_counter()
        for operation, identifier, size in events:
            if operation == "release":
                allocation = live.pop(identifier, None)
                if allocation is None:
                    continue
                if isinstance(allocator, FreeExtentAllocator):
                    allocator.release(*allocation)
                else:
                    allocator.release(allocation)
                continue
            required = allocator.allocation_bytes(size)
            allocation = allocator.allocate(size)
            if allocation is None:
                failures += 1
                fragmentation_failures += int(allocator.free_bytes >= required)
            else:
                live[identifier] = allocation
                requested_bytes += size
                allocated_bytes += (
                    allocation[1]
                    if isinstance(allocator, FreeExtentAllocator)
                    else len(allocation) * allocator.block_bytes
                )
        return {
            "allocation_failures": failures,
            "fragmentation_failures": fragmentation_failures,
            "internal_waste_ratio": (
                (allocated_bytes - requested_bytes) / allocated_bytes
            ),
            "trace_ms": (time.perf_counter() - started) * 1000,
        }

    extent = FreeExtentAllocator(capacity)
    legacy = _LegacyBuddyAllocator(capacity)
    page_runs = FixedBlockAllocator(capacity)
    legacy_full_bytes = sum(legacy.allocation_bytes(size) for size in sizes)
    page_run_full_bytes = sum(page_runs.allocation_bytes(size) for size in sizes)
    return {
        "operations": 20_000,
        "capacity_bytes": capacity,
        "exact_byte_extents": run_trace(extent),
        "legacy_power_of_two_buddy": {
            "block_bytes": legacy.block_bytes,
            "full_corpus_allocated_bytes": legacy_full_bytes,
            **run_trace(legacy),
        },
        "segregated_page_runs": {
            "block_bytes": page_runs.block_bytes,
            "full_corpus_allocated_bytes": page_run_full_bytes,
            "full_corpus_space_saved_bytes": legacy_full_bytes - page_run_full_bytes,
            **run_trace(page_runs),
        },
    }


class LegacyLru:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.entries: OrderedDict[str, bytes] = OrderedDict()

    def access(self, key: str, size: int) -> bool:
        if key in self.entries:
            self.entries.move_to_end(key)
            return True
        payload = bytes(size)
        if size <= self.maximum_bytes:
            self.entries[key] = payload
            self.bytes += size
            while self.bytes > self.maximum_bytes:
                _, evicted = self.entries.popitem(last=False)
                self.bytes -= len(evicted)
        return False


def policy_ablation(records: list[dict[str, object]]) -> dict[str, object]:
    rng = random.Random(77)
    universe = records[:5000]
    sizes = {
        str(record["tensor_ref"]): int(record["canonical_bytes"]) for record in universe
    }
    hot = list(sizes)[:100]
    cold = list(sizes)[100:]
    # Warm every hot object before injecting scans so TinyLFU admission is
    # measured directly rather than starting from an empty cache.
    trace = hot * 5
    for cycle in range(30):
        trace.extend(rng.choices(hot, weights=range(len(hot), 0, -1), k=1000))
        start = cycle * 100 % len(cold)
        trace.extend(cold[start : start + 300])
    # The budget holds exactly the hot set but not the scan tail.
    budget = sum(sizes[key] for key in hot)
    lru = LegacyLru(budget)
    gdsf = PayloadCache(budget)
    lru_hits = 0
    gdsf_hits = 0
    for key in trace:
        size = sizes[key]
        lru_hits += int(lru.access(key, size))
        cached = gdsf.get((key,))
        gdsf_hits += int(cached is not None)
        if cached is None:
            gdsf.put((key,), bytes(size))
    return {
        "accesses": len(trace),
        "budget_bytes": budget,
        "lru_hit_ratio": lru_hits / len(trace),
        "tinylfu_gdsf_hit_ratio": gdsf_hits / len(trace),
        "tinylfu_gdsf_status": gdsf.status(),
    }


def encode_frame(
    records: list[dict[str, object]], contract: str, query: np.ndarray, request_id: int
) -> bytes:
    dtype = protocol.DTYPE_F16 if query.dtype == np.dtype("<f2") else protocol.DTYPE_F32
    body = bytearray(
        protocol.EXTERNAL_REQUEST_FIXED.pack(
            query.shape[1], query.shape[0], len(records), dtype, 1, 0, len(contract)
        )
    )
    body.extend(contract.encode())
    body.extend(query.tobytes())
    for candidate_id, record in enumerate(records):
        tensor_ref = str(record["tensor_ref"]).encode()
        checksum = str(record["tensor_checksum"]).encode()
        body.extend(
            protocol.EXTERNAL_CANDIDATE_FIXED.pack(
                candidate_id,
                int(record["tensor_rows"]),
                len(tensor_ref),
                len(checksum),
            )
        )
        body.extend(tensor_ref)
        body.extend(checksum)
    return (
        protocol.HEADER.pack(
            protocol.MAGIC,
            protocol.EXTERNAL_VERSION,
            protocol.REQUEST_KIND,
            request_id,
            len(body),
        )
        + body
    )


def request_round_trip(
    socket_path: Path, frame: bytes
) -> tuple[float, list[tuple[int, float]]]:
    started = time.perf_counter()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.fspath(socket_path))
        connection.sendall(frame)
        header = protocol.receive_exact(connection, protocol.HEADER.size)
        body_bytes = protocol.HEADER.unpack(header)[4]
        response = header + protocol.receive_exact(connection, body_bytes)
    elapsed = (time.perf_counter() - started) * 1000
    _, status, parsed = decode_response(response)
    if status != 0 or not isinstance(parsed, list):
        raise RuntimeError(f"daemon request failed: {parsed}")
    if len(parsed) != len(protocol.parse_request_frame(frame).candidates):
        raise RuntimeError("daemon returned an incomplete result")
    return elapsed, parsed


def daemon_ablation(
    records: list[dict[str, object]],
    contract: str,
    shard_root: Path,
    rust_binary: Path,
    device: int,
    repeats: int,
    gpu_block_kib: int = 32,
) -> dict[str, object]:
    rng = np.random.default_rng(31)
    query = rng.standard_normal((44, int(records[0]["tensor_dim"]))).astype("<f2")
    query /= np.linalg.norm(query.astype(np.float32), axis=1, keepdims=True).astype(
        "<f2"
    )
    frame = encode_frame(records, contract, query, 7001)
    shard_paths = sorted((shard_root / "shards").glob("*.vts"))
    results = {}
    scores_by_daemon: dict[str, list[tuple[int, float]]] = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, command in (
            (
                "python_triton",
                [
                    sys.executable,
                    "-m",
                    "services.tilemaxsim_cuda_sidecar",
                    "--socket",
                    f"{directory}/python.sock",
                    "--gpu-memory-gb",
                    f"{device}=0.75",
                    "--gpu-workspace-gb",
                    "0.25",
                    "--gpu-block-kib",
                    str(gpu_block_kib),
                    "--host-cache-gb",
                    "0.25",
                    "--contract-root",
                    f"{contract}={shard_root}",
                    "--request-timeout-ms",
                    "20000",
                ],
            ),
            (
                "rust_cuda",
                [
                    os.fspath(rust_binary),
                    "--socket",
                    f"{directory}/rust.sock",
                    "--gpu-memory-gb",
                    f"{device}=0.75",
                    "--gpu-workspace-gb",
                    "0.25",
                    "--gpu-block-kib",
                    str(gpu_block_kib),
                    "--host-cache-gb",
                    "0.25",
                    "--contract-root",
                    f"{contract}={shard_root}",
                ],
            ),
        ):
            evict_paths(shard_paths)
            socket_path = Path(command[command.index("--socket") + 1])
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(3000):
                    if socket_path.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                startup_ms = (time.perf_counter() - started) * 1000
                if process.poll() is not None or not socket_path.exists():
                    output, _ = process.communicate(timeout=5)
                    raise RuntimeError(f"{name} failed to start: {output}")
                responses = [
                    request_round_trip(socket_path, frame) for _ in range(repeats + 1)
                ]
                samples = [item[0] for item in responses]
                scores_by_daemon[name] = responses[-1][1]
                results[name] = {
                    "startup_ms": startup_ms,
                    "cold_request_ms": samples[0],
                    "warm_p50_ms": statistics.median(samples[1:]),
                    "warm_p95_ms": percentile(samples[1:], 0.95),
                }
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                process.communicate(timeout=5)
    python_scores = dict(scores_by_daemon["python_triton"])
    rust_scores = dict(scores_by_daemon["rust_cuda"])
    deltas = [abs(python_scores[key] - rust_scores[key]) for key in python_scores]
    top_count = min(10, len(python_scores))
    python_top = {
        key
        for key, _ in sorted(
            python_scores.items(), key=lambda item: item[1], reverse=True
        )[:top_count]
    }
    rust_top = {
        key
        for key, _ in sorted(
            rust_scores.items(), key=lambda item: item[1], reverse=True
        )[:top_count]
    }
    results["correctness"] = {
        "max_abs_score_delta": max(deltas, default=0.0),
        "mean_abs_score_delta": statistics.mean(deltas) if deltas else 0.0,
        "top10_overlap": len(python_top & rust_top) / max(1, top_count),
    }
    return results


def prewarm_ablation(
    descriptor_manifest: Path,
    contract: str,
    shard_root: Path,
    rust_binary: Path,
    device: int,
    gpu_block_kib: int = 32,
) -> dict[str, object]:
    shard_paths = sorted((shard_root / "shards").glob("*.vts"))
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        commands = {
            "python_triton": [
                sys.executable,
                "-m",
                "services.tilemaxsim_cuda_sidecar",
                "--socket",
                f"{directory}/python-resident.sock",
                "--gpu-memory-gb",
                f"{device}=20",
                "--gpu-workspace-gb",
                "2",
                "--gpu-block-kib",
                str(gpu_block_kib),
                "--host-cache-gb",
                "0.1",
                "--contract-root",
                f"{contract}={shard_root}",
                "--gpu-cache-mode",
                "resident",
                "--resident-manifest",
                f"{contract}={descriptor_manifest}",
                "--prewarm-batch-size",
                "256",
                "--request-timeout-ms",
                "20000",
            ],
            "rust_cuda": [
                os.fspath(rust_binary),
                "--socket",
                f"{directory}/rust-resident.sock",
                "--gpu-memory-gb",
                f"{device}=20",
                "--gpu-workspace-gb",
                "2",
                "--gpu-block-kib",
                str(gpu_block_kib),
                "--host-cache-gb",
                "0.1",
                "--contract-root",
                f"{contract}={shard_root}",
                "--gpu-cache-mode",
                "resident",
                "--resident-manifest",
                f"{contract}={descriptor_manifest}",
                "--prewarm-batch-size",
                "256",
            ],
        }
        for name, command in commands.items():
            os.sync()
            evict_paths(shard_paths)
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            lines = []
            prewarm_event = None
            ready_event = None
            deadline = time.monotonic() + 300
            try:
                assert process.stdout is not None
                while time.monotonic() < deadline:
                    readable, _, _ = select.select([process.stdout], [], [], 1.0)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            lines.append(line)
                            if line.startswith("{"):
                                event = json.loads(line)
                                if event.get("event") in (
                                    "tilemaxsim_prewarm_complete",
                                    "tilemaxsim_rust_prewarm_complete",
                                ):
                                    prewarm_event = event
                                if event.get("event") in (
                                    "tilemaxsim_ready",
                                    "tilemaxsim_rust_ready",
                                ):
                                    ready_event = event
                                    break
                    if process.poll() is not None:
                        break
                if ready_event is None:
                    remainder, _ = process.communicate(timeout=5)
                    raise RuntimeError(
                        f"{name} resident prewarm failed: {''.join(lines)}{remainder}"
                    )
                results[name] = {
                    "process_to_ready_ms": (time.perf_counter() - started) * 1000,
                    "prewarm_reported_ms": prewarm_event.get("elapsed_ms")
                    if prewarm_event
                    else None,
                    "cache": ready_event.get("gpu_cache", ready_event.get("cache")),
                }
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
                process.communicate(timeout=5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--contract", default="benchmark@1")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gpu-block-kib", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full-prewarm", action="store_true")
    args = parser.parse_args()
    records = load_records(args.descriptor_manifest)
    selected = random.Random(20260714).sample(records, args.sample_size)
    report = {
        "corpus": {
            "records": len(records),
            "logical_bytes": sum(int(record["canonical_bytes"]) for record in records),
            "sample_size": len(selected),
        },
        "storage": storage_ablation(
            selected, args.contract, args.legacy_root, args.shard_root
        ),
        "h2d": h2d_ablation(selected, args.contract, args.shard_root, args.device),
        "allocator": allocator_ablation(records),
        "policy": policy_ablation(records),
        "daemon": daemon_ablation(
            selected,
            args.contract,
            args.shard_root,
            args.rust_binary,
            args.device,
            args.repeats,
            args.gpu_block_kib,
        ),
    }
    if args.full_prewarm:
        report["full_resident_prewarm"] = prewarm_ablation(
            args.descriptor_manifest,
            args.contract,
            args.shard_root,
            args.rust_binary,
            args.device,
            args.gpu_block_kib,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
