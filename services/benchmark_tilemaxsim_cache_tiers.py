# This software is licensed under a dual license model:
# GNU Affero General Public License v3 (AGPLv3) or Elastic License v2 (ELv2).
# Copyright (c) 2026 Hu Xinjing

"""Isolate L2 cold, L0 resident, and L1 rehydrate TileMaxSim requests."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from services.benchmark_tilemaxsim_ablation import (
    encode_frame,
    evict_paths,
    load_records,
    request_round_trip,
)


def metric_snapshot(socket_path: Path) -> dict[str, float]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect(os.fspath(socket_path))
        connection.sendall(b"GET /metrics HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n")
        payload = bytearray()
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            payload.extend(chunk)
    body = bytes(payload).split(b"\r\n\r\n", 1)[1].decode("utf-8")
    result = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        result[name] = float(value)
    return result


def delta(before: dict[str, float], after: dict[str, float], prefix: str) -> float:
    keys = set(before) | set(after)
    return sum(after.get(key, 0.0) - before.get(key, 0.0)
               for key in keys if key.startswith(prefix))


def labeled_delta(before, after, prefix: str, label: str) -> float:
    return sum(after.get(key, 0.0) - before.get(key, 0.0) for key in set(before) | set(after)
               if key.startswith(prefix) and label in key)


def wait_for(path: Path, process: subprocess.Popen[object]) -> None:
    for _ in range(3000):
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError("tilemaxsimd exited before ready")
        time.sleep(0.01)
    raise TimeoutError("tilemaxsimd did not become ready")


def groups(records, count: int):
    return [records[index:index + count] for index in range(0, len(records), count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--contract", default="benchmark@1")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hot-candidates", type=int, default=50)
    parser.add_argument("--thrash-candidates", type=int, default=80)
    parser.add_argument("--thrash-groups", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.hot_candidates, args.thrash_candidates, args.thrash_groups) <= 0:
        parser.error("cache-tier sizes must be positive")
    records = load_records(args.descriptor_manifest)
    required = args.hot_candidates + args.thrash_candidates * args.thrash_groups
    if len(records) < required:
        raise ValueError(f"cache-tier benchmark needs at least {required} descriptors")
    hot = records[:args.hot_candidates]
    thrash = groups(records[args.hot_candidates:required], args.thrash_candidates)
    generator = np.random.default_rng(20260823)
    query = generator.standard_normal((44, int(records[0]["tensor_dim"]))).astype(np.float32)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    query = query.astype("<f2")
    hot_frame = encode_frame(hot, args.contract, query, 1)
    thrash_frames = [encode_frame(group, args.contract, query, 100 + index)
                     for index, group in enumerate(thrash)]
    shard_paths = sorted((args.shard_root / "shards").glob("*.vts"))
    evict_paths(shard_paths)
    with tempfile.TemporaryDirectory(prefix="tilemaxsim-tier-") as directory:
        root = Path(directory)
        scoring_socket = root / "score.sock"
        status_socket = root / "status.sock"
        log = (root / "daemon.log").open("w", encoding="utf-8")
        process = subprocess.Popen([
            os.fspath(args.binary), "--socket", os.fspath(scoring_socket),
            "--status-socket", os.fspath(status_socket),
            "--gpu-memory-gb", f"{args.device}=0.3", "--gpu-workspace-gb", "0.25",
            "--host-cache-gb", "0.5", "--contract-root", f"{args.contract}={args.shard_root}",
            "--request-timeout-ms", "120000", "--socket-io-timeout-ms", "120000",
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(status_socket, process)
            initial = metric_snapshot(status_socket)
            l2_ms, l2_scores = request_round_trip(scoring_socket, hot_frame, timeout_s=120)
            after_l2 = metric_snapshot(status_socket)
            l0_ms, l0_scores = request_round_trip(scoring_socket, hot_frame, timeout_s=120)
            after_l0 = metric_snapshot(status_socket)
            for _ in range(3):
                for frame in thrash_frames:
                    request_round_trip(scoring_socket, frame, timeout_s=120)
            before_rehydrate = metric_snapshot(status_socket)
            l1_ms, l1_scores = request_round_trip(scoring_socket, hot_frame, timeout_s=120)
            after_l1 = metric_snapshot(status_socket)
        finally:
            process.terminate()
            process.wait(timeout=30)
            log.close()
    if l2_scores != l0_scores or l2_scores != l1_scores:
        raise RuntimeError("cache tiers returned different exact scores")

    def phase(before, after, latency_ms):
        return {
            "latency_ms": latency_ms,
            "gpu_hits": labeled_delta(before, after,
                "tilemaxsim_gpu_cache_events_total", 'event="hit"'),
            "gpu_misses": labeled_delta(before, after,
                "tilemaxsim_gpu_cache_events_total", 'event="miss"'),
            "host_hits": delta(before, after,
                'tilemaxsim_host_cache_events_total{event="hit"}'),
            "host_misses": delta(before, after,
                'tilemaxsim_host_cache_events_total{event="miss"}'),
            "storage_read_calls": delta(before, after, "tilemaxsim_storage_read_calls_total"),
            "h2d_bytes": delta(before, after, "tilemaxsim_gpu_h2d_bytes_total"),
        }

    report = {
        "version": 1,
        "hot_candidates": len(hot),
        "hot_logical_bytes": sum(int(item["canonical_bytes"]) for item in hot),
        "l2_cold": phase(initial, after_l2, l2_ms),
        "l0_resident": phase(after_l2, after_l0, l0_ms),
        "l1_rehydrate": phase(before_rehydrate, after_l1, l1_ms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
