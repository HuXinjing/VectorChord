# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Compare serial and overlapped tensor I/O on the native CUDA daemon."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from services.benchmark_tilemaxsim_ablation import (
    encode_frame,
    evict_paths,
    request_round_trip,
)


def load_records(path: Path, maximum_candidates: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    tokens = 0
    logical_bytes = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            rows = int(record["tensor_rows"])
            payload_bytes = int(record["canonical_bytes"])
            if records and (
                len(records) >= maximum_candidates
                or tokens + rows > 1_000_000
                or logical_bytes + payload_bytes > 1024**3
            ):
                break
            records.append(record)
            tokens += rows
            logical_bytes += payload_bytes
    if len(records) < 2:
        raise ValueError("the benchmark needs at least two protocol-valid tensors")
    return records


def run_mode(
    *,
    mode: str,
    trial: int,
    binary: Path,
    shard_root: Path,
    contract: str,
    device: int,
    gpu_memory_gb: str,
    workspace_gb: str,
    host_cache_gb: str,
    io_batch_gb: str,
    frame: bytes,
    shard_paths: list[Path],
    directory: Path,
) -> tuple[float, list[tuple[int, float]], dict[str, object]]:
    evict_paths(shard_paths)
    socket_path = directory / f"{mode}-{trial}.sock"
    command = [
        os.fspath(binary),
        "--socket",
        os.fspath(socket_path),
        "--gpu-memory-gb",
        f"{device}={gpu_memory_gb}",
        "--gpu-workspace-gb",
        workspace_gb,
        "--host-cache-gb",
        host_cache_gb,
        "--io-pipeline",
        mode,
        "--io-batch-gb",
        io_batch_gb,
        "--contract-root",
        f"{contract}={shard_root}",
        "--request-timeout-ms",
        "60000",
        "--socket-io-timeout-ms",
        "60000",
        "--once",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(3000):
            if socket_path.exists() or process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is not None or not socket_path.exists():
            output, _ = process.communicate(timeout=5)
            raise RuntimeError(f"{mode} daemon failed to start: {output}")
        latency_ms, scores = request_round_trip(socket_path, frame)
        output, _ = process.communicate(timeout=70)
        if process.returncode != 0:
            raise RuntimeError(f"{mode} daemon failed: {output}")
        events = [
            json.loads(line) for line in output.splitlines() if line.startswith("{")
        ]
        request_event = next(
            event for event in events if event.get("event") == "tilemaxsim_rust_request"
        )
        return latency_ms, scores, request_event["cache"]["io_pipeline"]
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--query-embedding", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-memory-gb", default="1")
    parser.add_argument("--workspace-gb", default="0.2")
    parser.add_argument("--host-cache-gb", default="0.1")
    parser.add_argument("--io-batch-gb", default="0.05")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidates < 2 or args.trials < 1:
        parser.error("candidates must be >= 2 and trials must be positive")

    records = load_records(args.descriptor_manifest, args.candidates)
    query = np.load(args.query_embedding, allow_pickle=False)
    if query.dtype != np.dtype("float16"):
        query = query.astype("<f2")
    frame = encode_frame(records, args.contract, query, 9_001)
    shard_paths = sorted((args.shard_root / "shards").glob("*.vts"))
    if not shard_paths:
        raise ValueError("shard root has no immutable .vts files")

    samples: dict[str, list[float]] = {"serial": [], "overlap": []}
    pipeline_status: dict[str, dict[str, object]] = {}
    reference_scores: list[tuple[int, float]] | None = None
    maximum_score_delta = 0.0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for trial in range(args.trials):
            # Alternate order so a systematic first-run effect is not assigned
            # to the same mode in every trial.
            modes = ("serial", "overlap") if trial % 2 == 0 else ("overlap", "serial")
            for mode in modes:
                latency, scores, status = run_mode(
                    mode=mode,
                    trial=trial,
                    binary=args.rust_binary,
                    shard_root=args.shard_root,
                    contract=args.contract,
                    device=args.device,
                    gpu_memory_gb=args.gpu_memory_gb,
                    workspace_gb=args.workspace_gb,
                    host_cache_gb=args.host_cache_gb,
                    io_batch_gb=args.io_batch_gb,
                    frame=frame,
                    shard_paths=shard_paths,
                    directory=directory,
                )
                samples[mode].append(latency)
                pipeline_status[mode] = status
                if reference_scores is None:
                    reference_scores = scores
                else:
                    if [item[0] for item in scores] != [
                        item[0] for item in reference_scores
                    ]:
                        raise RuntimeError("serial and overlap candidate IDs disagree")
                    maximum_score_delta = max(
                        maximum_score_delta,
                        max(
                            abs(actual[1] - expected[1])
                            for actual, expected in zip(
                                scores, reference_scores, strict=True
                            )
                        ),
                    )

    serial_mean = statistics.fmean(samples["serial"])
    overlap_mean = statistics.fmean(samples["overlap"])
    report = {
        "benchmark": "tilemaxsim_tutti_io_pipeline_v1",
        "corpus": {
            "candidates": len(records),
            "tokens": sum(int(record["tensor_rows"]) for record in records),
            "logical_bytes": sum(int(record["canonical_bytes"]) for record in records),
            "query_rows": int(query.shape[0]),
            "dimension": int(query.shape[1]),
        },
        "configuration": {
            "device": args.device,
            "gpu_memory_gb": args.gpu_memory_gb,
            "workspace_gb": args.workspace_gb,
            "host_cache_gb": args.host_cache_gb,
            "io_batch_gb": args.io_batch_gb,
            "trials": args.trials,
        },
        "latency_ms": {
            mode: {
                "samples": values,
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
            }
            for mode, values in samples.items()
        },
        "speedup": serial_mean / overlap_mean,
        "maximum_score_delta": maximum_score_delta,
        "pipeline_status": pipeline_status,
    }
    document = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    print(document, end="")


if __name__ == "__main__":
    main()
