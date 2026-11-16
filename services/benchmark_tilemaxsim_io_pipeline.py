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
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from services.benchmark_tilemaxsim_ablation import (
    encode_frame,
    request_round_trip,
)


def load_records(path: Path, maximum_candidates: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(records) >= maximum_candidates:
                break
            record = json.loads(line)
            if int(record["tensor_rows"]) <= 0 or int(record["canonical_bytes"]) <= 0:
                raise ValueError("invalid tensor shape or size in manifest")
            records.append(record)
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
    io_batch_mib: int,
    frame: bytes,
    directory: Path,
) -> tuple[float, list[tuple[int, float]], dict[str, object]]:
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
        "--io-batch-mib",
        str(io_batch_mib),
        "--tensor-read-workers",
        "16",
        "--scheduler-quantum-candidates",
        "65536",
        "--scheduler-quantum-tokens",
        "1000000000",
        "--scheduler-quantum-fmas",
        "18446744073709551615",
        "--contract-root",
        f"{contract}={shard_root}",
        "--request-timeout-ms",
        "60000",
        "--socket-io-timeout-ms",
        "60000",
        "--once",
    ]
    log = (directory / f"{mode}-{trial}.log").open("w+")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(3000):
            if socket_path.exists() or process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is not None or not socket_path.exists():
            process.wait(timeout=5)
            log.flush(); log.seek(0); output = log.read()
            raise RuntimeError(f"{mode} daemon failed to start: {output}")
        latency_ms, scores = request_round_trip(socket_path, frame, timeout_s=70)
        process.wait(timeout=70)
        log.flush(); log.seek(0); output = log.read()
        if process.returncode != 0:
            raise RuntimeError(f"{mode} daemon failed: {output}")
        events = [
            json.loads(line) for line in output.splitlines() if line.startswith("{")
        ]
        stages = [event for event in events if event.get("event") == "tilemaxsim_io_pipeline"]
        return latency_ms, sorted(scores), {"stages": stages}
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        log.close()


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
    parser.add_argument("--io-batch-mib", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidates < 2 or args.trials < 1 or not 1 <= args.io_batch_mib <= 1024:
        parser.error("candidates must be >= 2, trials positive, and io-batch-mib between 1 and 1024")

    records = load_records(args.descriptor_manifest, args.candidates)
    query = np.load(args.query_embedding, allow_pickle=False)
    if query.dtype != np.dtype("float16"):
        query = query.astype("<f2")
    frame = encode_frame(records, args.contract, query, 9_001)
    # Fresh daemon caches per trial; shared filesystem page caches stay intact.
    samples: dict[str, list[float]] = {"disabled": [], "serial": [], "overlap": []}
    pipeline_status: dict[str, dict[str, object]] = {}
    reference_scores: list[tuple[int, float]] | None = None
    maximum_score_delta = 0.0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for trial in range(args.trials):
            # Alternate order so a systematic first-run effect is not assigned
            # to the same mode in every trial.
            modes = ("disabled", "serial", "overlap") if trial % 2 == 0 else ("overlap", "serial", "disabled")
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
                    io_batch_mib=args.io_batch_mib,
                    frame=frame,
                    directory=directory,
                )
                if any(not math.isfinite(score) for _, score in scores):
                    raise RuntimeError("daemon returned a non-finite score")
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

    if maximum_score_delta > 1e-4:
        raise RuntimeError(f"score equivalence failed: {maximum_score_delta}")
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
            "io_batch_mib": args.io_batch_mib,
            "filesystem_cache": "unmodified",
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
        "speedup_vs_serial": serial_mean / overlap_mean,
        "speedup_vs_legacy": statistics.fmean(samples["disabled"]) / overlap_mean,
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
