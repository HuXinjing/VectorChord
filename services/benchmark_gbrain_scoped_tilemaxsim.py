# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Benchmark real CUDA TileMaxSim over a GBrain-style lexical/structured scope."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np

from services.benchmark_full_corpus_tilemaxsim import descriptor_batches
from services.benchmark_tilemaxsim_ablation import encode_frame, request_round_trip


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
        return [json.loads(line) for line in stream if line.strip()]


def select_method(report: dict[str, object], method_name: str) -> dict[str, object]:
    methods = report.get("methods")
    if not isinstance(methods, list):
        raise ValueError("candidate report has no methods array")
    for method in methods:
        if isinstance(method, dict) and method.get("method") == method_name:
            return method
    raise ValueError(f"candidate method {method_name!r} was not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--page-manifest", required=True, type=Path)
    parser.add_argument("--query-dataset", required=True, type=Path)
    parser.add_argument("--corpus-json", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-method", default="es_bm25")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw_descriptors = load_jsonl(args.descriptor_manifest)
    page_metadata = {
        str(record["page_key"]): record for record in load_jsonl(args.page_manifest)
    }
    descriptors = [
        {**descriptor, **page_metadata.get(str(descriptor.get("page_key")), {})}
        for descriptor in raw_descriptors
    ]
    queries = json.loads(args.query_dataset.read_text(encoding="utf-8"))
    corpus_records = json.loads(args.corpus_json.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    method = select_method(candidate_report, args.candidate_method)
    per_query = method.get("per_query_results")
    if not isinstance(queries, list) or not queries:
        raise ValueError("query dataset must be a nonempty JSON array")
    if not isinstance(corpus_records, list):
        raise ValueError("corpus JSON must be an array")
    if not isinstance(per_query, dict):
        raise ValueError("candidate method has no per_query_results object")
    if args.query_limit < 0:
        parser.error("--query-limit must be nonnegative")
    queries = queries[: args.query_limit or None]

    descriptors_by_doc: dict[str, list[dict[str, object]]] = {}
    descriptors_by_doc_page: dict[tuple[str, int], list[dict[str, object]]] = {}
    for descriptor in descriptors:
        doc_name = descriptor.get("doc_name")
        if isinstance(doc_name, str):
            descriptors_by_doc.setdefault(doc_name, []).append(descriptor)
            descriptors_by_doc_page.setdefault(
                (doc_name, int(descriptor.get("page_no", 0))), []
            ).append(descriptor)
    chunks_by_id = {
        str(record["_id"]): record["_source"] for record in corpus_records
    }

    cases: list[dict[str, object]] = []
    scope_latencies: list[float] = []
    cuda_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    for query_index, query_record in enumerate(queries):
        query_id = query_record.get("query_id")
        gold_doc = query_record.get("gold_doc_name")
        if not isinstance(query_id, str) or not isinstance(gold_doc, str):
            raise ValueError(f"query {query_index} lacks query_id or gold_doc_name")
        raw_candidates = per_query.get(query_id, [])
        if not isinstance(raw_candidates, list):
            raise ValueError(f"candidate list for {query_id} is not an array")

        scope_started = time.perf_counter()
        candidate_docs: list[str] = []
        seen_docs: set[str] = set()
        for row in raw_candidates:
            doc_name = row.get("doc_name") if isinstance(row, dict) else None
            if isinstance(doc_name, str) and doc_name not in seen_docs:
                seen_docs.add(doc_name)
                candidate_docs.append(doc_name)
        scoped_descriptors: list[dict[str, object]] = []
        seen_pages: set[str] = set()
        for row in raw_candidates:
            if not isinstance(row, dict):
                continue
            chunk_id = str(row.get("chunk_id", ""))
            doc_name = row.get("doc_name")
            source = chunks_by_id.get(chunk_id)
            content = source.get("content_with_weight", "") if isinstance(source, dict) else ""
            page_numbers = {
                int(value) for value in re.findall(r"::(\d+)", str(content))
            }
            mapped = [
                descriptor
                for page_no in sorted(page_numbers)
                for descriptor in descriptors_by_doc_page.get((str(doc_name), page_no), [])
            ]
            # Preserve recall when an imported chunk has no page markers.
            if not mapped and isinstance(doc_name, str):
                mapped = descriptors_by_doc.get(doc_name, [])
            for descriptor in mapped:
                page_key = str(descriptor.get("page_key", ""))
                if page_key and page_key not in seen_pages:
                    seen_pages.add(page_key)
                    scoped_descriptors.append(descriptor)
        scope_ms = (time.perf_counter() - scope_started) * 1000
        scope_latencies.append(scope_ms)

        query = np.load(args.query_embeddings / f"{query_id}.npy", allow_pickle=False)
        if query.dtype != np.dtype("float16"):
            query = query.astype("<f2")
        scores: list[tuple[float, str, int]] = []
        cuda_ms = 0.0
        started = time.perf_counter()
        batches = (
            descriptor_batches(scoped_descriptors, int(query.shape[0]))
            if scoped_descriptors
            else []
        )
        for batch_index, (_, batch) in enumerate(batches):
            frame = encode_frame(
                batch,
                args.contract,
                query,
                (query_index + 1) * 100_000 + batch_index,
            )
            batch_ms, results = request_round_trip(args.socket, frame)
            cuda_ms += batch_ms
            for candidate_id, score in results:
                descriptor = batch[candidate_id]
                scores.append(
                    (
                        float(score),
                        str(descriptor.get("doc_name", "")),
                        int(descriptor.get("page_no", 0)),
                    )
                )
        end_to_end_ms = (time.perf_counter() - started) * 1000 + scope_ms
        cuda_latencies.append(cuda_ms)
        end_to_end_latencies.append(end_to_end_ms)
        scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        ranked_docs: list[str] = []
        seen_ranked: set[str] = set()
        for _, doc_name, _ in scores:
            if doc_name and doc_name not in seen_ranked:
                seen_ranked.add(doc_name)
                ranked_docs.append(doc_name)

        cases.append(
            {
                "query_index": query_index,
                "query_id": query_id,
                "gold_doc_name": gold_doc,
                "candidate_chunks": len(raw_candidates),
                "candidate_docs": len(candidate_docs),
                "candidate_tensors": len(scoped_descriptors),
                "candidate_scope_contains_gold_doc": gold_doc in seen_docs,
                "scope_materialization_ms": scope_ms,
                "cuda_round_trip_ms": cuda_ms,
                "end_to_end_ms": end_to_end_ms,
                "doc_hit_at_1": gold_doc in ranked_docs[:1],
                "doc_hit_at_5": gold_doc in ranked_docs[:5],
                "doc_hit_at_10": gold_doc in ranked_docs[:10],
                "top_10_docs": ranked_docs[:10],
            }
        )

    report = {
        "benchmark": "gbrain_scoped_native_cuda_tilemaxsim_v1",
        "scope": {
            "candidate_method": args.candidate_method,
            "policy": "all deduplicated candidate chunks; every source page referenced by each chunk",
            "full_corpus_tensor_scan": False,
        },
        "queries": len(cases),
        "candidate_chunks": summary([float(item["candidate_chunks"]) for item in cases]),
        "candidate_docs": summary([float(item["candidate_docs"]) for item in cases]),
        "candidate_tensors": summary([float(item["candidate_tensors"]) for item in cases]),
        "scope_materialization_ms": summary(scope_latencies),
        "cuda_round_trip_ms": summary(cuda_latencies),
        "end_to_end_ms": summary(end_to_end_latencies),
        "quality": {
            "candidate_doc_recall": statistics.fmean(
                float(item["candidate_scope_contains_gold_doc"]) for item in cases
            ),
            "doc_hit_at_1": statistics.fmean(float(item["doc_hit_at_1"]) for item in cases),
            "doc_hit_at_5": statistics.fmean(float(item["doc_hit_at_5"]) for item in cases),
            "doc_hit_at_10": statistics.fmean(float(item["doc_hit_at_10"]) for item in cases),
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
