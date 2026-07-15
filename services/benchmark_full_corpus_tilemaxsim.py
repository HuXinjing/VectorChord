# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Benchmark exact full-corpus TileMaxSim through the native CUDA daemon."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from services.benchmark_tilemaxsim_ablation import encode_frame, request_round_trip

MAX_BATCH_CANDIDATES = 65_536
MAX_BATCH_TOKENS = 1_000_000
MAX_BATCH_TENSOR_BYTES = 1024**3


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": max(samples),
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError(f"empty JSONL file: {path}")
    return records


def descriptor_batches(
    descriptors: list[dict[str, object]], query_rows: int
) -> list[tuple[int, list[dict[str, object]]]]:
    batches: list[tuple[int, list[dict[str, object]]]] = []
    start = 0
    current: list[dict[str, object]] = []
    tokens = query_rows
    tensor_bytes = query_rows * int(descriptors[0]["tensor_dim"]) * 2
    for descriptor in descriptors:
        rows = int(descriptor["tensor_rows"])
        scalar_bytes = 2 if descriptor["tensor_dtype"] == "float16" else 4
        payload_bytes = rows * int(descriptor["tensor_dim"]) * scalar_bytes
        would_overflow = current and (
            len(current) == MAX_BATCH_CANDIDATES
            or tokens + rows > MAX_BATCH_TOKENS
            or tensor_bytes + payload_bytes > MAX_BATCH_TENSOR_BYTES
        )
        if would_overflow:
            batches.append((start, current))
            start += len(current)
            current = []
            tokens = query_rows
            tensor_bytes = query_rows * int(descriptor["tensor_dim"]) * scalar_bytes
        current.append(descriptor)
        tokens += rows
        tensor_bytes += payload_bytes
    if current:
        batches.append((start, current))
    return batches


def recall(expected: np.ndarray, actual: list[int], top_k: int) -> float:
    return len(set(expected[:top_k].tolist()).intersection(actual[:top_k])) / top_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--query-dataset", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    descriptors = load_jsonl(args.descriptor_manifest)
    queries = json.loads(args.query_dataset.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError("query dataset must be a nonempty JSON array")
    if args.query_limit < 0:
        parser.error("--query-limit must be nonnegative")
    queries = queries[: args.query_limit or None]
    gold = np.load(args.gold, allow_pickle=False)
    query_latencies: list[float] = []
    round_trip_latencies: list[float] = []
    batch_latencies: list[float] = []
    cases = []

    for query_index, query_record in enumerate(queries):
        query_id = query_record.get("query_id")
        if not isinstance(query_id, str):
            raise ValueError(f"query {query_index} has no query_id")
        query = np.load(args.query_embeddings / f"{query_id}.npy", allow_pickle=False)
        if query.dtype != np.dtype("float16"):
            query = query.astype("<f2")
        batches = descriptor_batches(descriptors, int(query.shape[0]))
        scores = np.empty(len(descriptors), dtype=np.float32)
        per_batch = []
        started = time.perf_counter()
        for batch_index, (offset, batch) in enumerate(batches):
            frame = encode_frame(
                batch,
                args.contract,
                query,
                (query_index + 1) * 100_000 + batch_index,
            )
            latency_ms, results = request_round_trip(args.socket, frame)
            per_batch.append(latency_ms)
            batch_latencies.append(latency_ms)
            for candidate_id, score in results:
                scores[offset + candidate_id] = score
        latency_ms = (time.perf_counter() - started) * 1000
        query_latencies.append(latency_ms)
        round_trip_ms = sum(per_batch)
        round_trip_latencies.append(round_trip_ms)
        ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        expected = gold[f"q{query_index}_idx"].astype(np.int64, copy=False)
        cases.append(
            {
                "query_index": query_index,
                "query_id": query_id,
                "query_rows": int(query.shape[0]),
                "batches": len(batches),
                "latency_ms": latency_ms,
                "cuda_round_trip_ms": round_trip_ms,
                "batch_latency_ms": summary(per_batch),
                "recall_at_10": recall(expected, ranking, 10),
                "recall_at_20": recall(expected, ranking, 20),
                "top_20": ranking[:20],
            }
        )

    report = {
        "benchmark": "full_corpus_native_cuda_tilemaxsim_v1",
        "corpus": {
            "descriptors": len(descriptors),
            "logical_tensor_bytes": sum(
                int(item["canonical_bytes"]) for item in descriptors
            ),
        },
        "queries": len(cases),
        "latency_ms": summary(query_latencies),
        "cuda_round_trip_ms": summary(round_trip_latencies),
        "batch_latency_ms": summary(batch_latencies),
        "quality": {
            "mean_recall_at_10": statistics.fmean(
                item["recall_at_10"] for item in cases
            ),
            "mean_recall_at_20": statistics.fmean(
                item["recall_at_20"] for item in cases
            ),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
