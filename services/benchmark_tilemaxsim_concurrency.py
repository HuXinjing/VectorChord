# This software is licensed under a dual license model:
# GNU Affero General Public License v3 (AGPLv3) or Elastic License v2 (ELv2).
# Copyright (c) 2026 Hu Xinjing

"""Measure multi-domain TileMaxSim fairness, priority, and tail latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from services.benchmark_tilemaxsim_ablation import encode_scheduled_frame, request_round_trip
from services.tilemaxsim_shard import ImmutableShardWriter


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency sample is empty")
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99), "max": max(values),
    }


def maximum_consecutive(values: list[str]) -> int:
    maximum = current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def crossed_priority(index: int, tenants: int, priorities: list[int]) -> int:
    if index < 0 or tenants <= 0 or not priorities:
        raise ValueError("invalid crossed-priority configuration")
    return priorities[(index // tenants) % len(priorities)]


def build_corpus(root: Path, documents: int, rows: int, dimension: int, seed: int):
    generator = np.random.default_rng(seed)
    records = []
    writer = ImmutableShardWriter(root, target_bytes=64 * 1024**2, fsync=False)
    try:
        for index in range(documents):
            tensor = generator.standard_normal((rows, dimension)).astype(np.float32)
            tensor /= np.maximum(np.linalg.norm(tensor, axis=1, keepdims=True), 1e-12)
            tensor = tensor.astype("<f2")
            payload = tensor.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer.add(digest, payload, rows, dimension, "float16")
            records.append({
                "candidate_id": index, "tensor_ref": f"sha256://{digest}",
                "tensor_checksum": f"sha256:{digest}", "tensor_rows": rows,
                "tensor_dim": dimension, "tensor_dtype": "float16",
                "canonical_bytes": len(payload),
            })
        writer.finish()
    finally:
        writer.close()
    return records


def wait_ready(socket_path: Path, process: subprocess.Popen[str]) -> None:
    for _ in range(3000):
        if socket_path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError("tilemaxsimd exited before ready")
        time.sleep(0.01)
    raise TimeoutError("tilemaxsimd did not create its socket")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--document-rows", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=320)
    parser.add_argument("--query-rows", type=int, default=32)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--tenants", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--priorities", default="-10,0,10,50")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    positive = (args.documents, args.document_rows, args.dimension, args.query_rows,
                args.requests, args.tenants, args.concurrency)
    if any(value <= 0 for value in positive):
        parser.error("corpus and concurrency dimensions must be positive")
    priorities = [int(value) for value in args.priorities.split(",")]
    if not priorities or any(not -100 <= value <= 100 for value in priorities):
        parser.error("priorities must be comma-separated integers from -100 to 100")

    with tempfile.TemporaryDirectory(prefix="tilemaxsim-concurrency-") as directory:
        root = Path(directory)
        shard_root = root / "shards"
        shard_root.mkdir()
        records = build_corpus(shard_root, args.documents, args.document_rows,
                               args.dimension, args.seed)
        generator = np.random.default_rng(args.seed + 1)
        query = generator.standard_normal((args.query_rows, args.dimension)).astype(np.float32)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        query = query.astype("<f2")
        socket_path = root / "tilemaxsimd.sock"
        daemon_log = root / "daemon.log"
        log_stream = daemon_log.open("w+", encoding="utf-8")
        process = subprocess.Popen([
                os.fspath(args.binary), "--socket", os.fspath(socket_path),
                "--gpu-memory-gb", f"{args.device}=1", "--gpu-workspace-gb", "0.25",
                "--host-cache-gb", "0.25", "--contract-root", f"benchmark@1={shard_root}",
                "--scheduler-policy", "fair-priority", "--scheduler-batch-window-ms", "25",
                "--scheduler-quantum-candidates", "8", "--max-queued-requests",
                str(max(args.requests, 128)), "--max-tenant-queued-requests",
                str(max(args.requests, 32)), "--request-timeout-ms", "300000",
                "--socket-io-timeout-ms", "300000",
            ], stdout=log_stream, stderr=subprocess.STDOUT, text=True)
        cases = []
        daemon_output = ""
        try:
            wait_ready(socket_path, process)
            request_round_trip(socket_path, encode_scheduled_frame(
                records, "benchmark@1", query, 1, "warmup", 0, 300_000), timeout_s=360)

            def call(index: int) -> dict[str, object]:
                tenant = f"tenant-{index % args.tenants}"
                # Cross priorities with domains instead of correlating the two
                # independent variables when their cardinalities happen to match.
                priority = crossed_priority(index, args.tenants, priorities)
                frame = encode_scheduled_frame(records, "benchmark@1", query,
                    10_000 + index, tenant, priority, 300_000)
                latency_ms, results = request_round_trip(socket_path, frame, timeout_s=360)
                return {"request_id": 10_000 + index, "tenant": tenant,
                        "priority": priority, "latency_ms": latency_ms,
                        "result_count": len(results)}

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                cases = list(executor.map(call, range(args.requests)))
            wall_ms = (time.perf_counter() - started) * 1000
        finally:
            process.terminate()
            process.wait(timeout=30)
            log_stream.flush()
            log_stream.seek(0)
            daemon_output = log_stream.read()
            log_stream.close()

    by_priority = defaultdict(list)
    by_tenant = defaultdict(list)
    for case in cases:
        by_priority[int(case["priority"])].append(float(case["latency_ms"]))
        by_tenant[str(case["tenant"])].append(float(case["latency_ms"]))
    events = [json.loads(line) for line in daemon_output.splitlines() if line.startswith("{")]
    completions = [event for event in events
                   if event.get("event") == "tilemaxsim_rust_request"
                   and int(event.get("request_id", 0)) >= 10_000]
    domains = [str(event["tenant_hash"]) for event in completions]
    report = {
        "version": 1,
        "configuration": {"documents": args.documents, "document_rows": args.document_rows,
            "dimension": args.dimension, "query_rows": args.query_rows,
            "requests": args.requests, "tenants": args.tenants,
            "concurrency": args.concurrency, "priorities": priorities},
        "wall_ms": wall_ms,
        "throughput_requests_per_second": args.requests / (wall_ms / 1000),
        "latency_ms": latency_summary([float(case["latency_ms"]) for case in cases]),
        "latency_by_priority_ms": {str(key): latency_summary(values)
                                   for key, values in sorted(by_priority.items())},
        "latency_by_tenant_ms": {key: latency_summary(values)
                                 for key, values in sorted(by_tenant.items())},
        "completion_order": [int(event["request_id"]) for event in completions],
        "maximum_consecutive_completion_domain": maximum_consecutive(domains),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
