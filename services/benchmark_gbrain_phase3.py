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
# Copyright (c) 2026 Hu Xinjing

"""Measure native Phase 3A ranking against a committed GBrain TileMaxSim gold."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from services.tilemaxsim_cuda_sidecar import positive_int
except ModuleNotFoundError:  # Allow direct `python services/...` invocation.
    from tilemaxsim_cuda_sidecar import positive_int

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")
PAGE_KEY = re.compile(r"^[0-9a-f]{40}$")
PROFILE_PREFIX = "vchordrq_maxsim_profile "


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def sql_halfvec_array(tensor: np.ndarray) -> str:
    if tensor.ndim != 2 or tensor.shape[1] <= 0 or tensor.shape[0] <= 0:
        raise ValueError("query tensor must have shape [rows, dimension]")
    if tensor.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("query tensor must use float16 or float32")
    if not np.isfinite(tensor).all():
        raise ValueError("query tensor contains non-finite values")
    vectors = []
    for row in tensor:
        value = "[" + ",".join(format(float(item), ".8g") for item in row) + "]"
        vectors.append("'" + value + "'::halfvec")
    return "ARRAY[" + ",".join(vectors) + "]"


def load_manifest(path: Path) -> list[str]:
    page_keys = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            page_key = record.get("page_key")
            if not isinstance(page_key, str) or not PAGE_KEY.fullmatch(page_key):
                raise ValueError(f"invalid page_key at manifest line {line_number}")
            page_keys.append(page_key)
    if not page_keys:
        raise ValueError("page manifest is empty")
    return page_keys


def load_queries(
    results_path: Path, embeddings: Path, limit: int
) -> list[tuple[str, Path]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    cases = results.get("queries", {}).get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("results JSON has no queries.cases")
    queries = []
    for case in cases[: limit or None]:
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not re.fullmatch(r"[0-9a-f]+", query_id):
            raise ValueError("invalid query ID in results JSON")
        path = embeddings / f"{query_id}.npy"
        if not path.is_file():
            raise ValueError(f"missing query embedding: {path}")
        queries.append((query_id, path))
    return queries


def recall(expected: list[str], actual: list[str], top_k: int) -> float:
    expected_set = set(expected[:top_k])
    return len(expected_set.intersection(actual[:top_k])) / top_k


def candidate_recall(expected: list[str], candidates: list[str], top_k: int) -> float:
    expected_set = set(expected[:top_k])
    return len(expected_set.intersection(candidates)) / top_k


def parse_profile(stderr: str) -> dict[str, int] | None:
    profiles = []
    for line in stderr.splitlines():
        marker = line.find(PROFILE_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(PROFILE_PREFIX) :].strip()
        profile = json.loads(payload)
        if not isinstance(profile, dict) or profile.get("schema_version") != 1:
            raise RuntimeError("unexpected MaxSim profile schema")
        if any(not isinstance(value, int) for value in profile.values()):
            raise RuntimeError("MaxSim profile values must be integers")
        profiles.append(profile)
    if len(profiles) > 1:
        raise RuntimeError("query emitted multiple MaxSim profiles")
    return profiles[0] if profiles else None


def execute_query(
    psql_command: list[str],
    table: str,
    index: str | None,
    query_sql: str,
    endpoint: str,
    probes: str,
    refine: int,
    candidate_limit: int,
    timeout_ms: int,
    profile: bool,
) -> tuple[list[str], float, dict[str, int] | None]:
    if index is None:
        search_sql = f"""
SELECT page_key
FROM {table}
ORDER BY embedding @# {query_sql}
LIMIT {candidate_limit};
"""
    else:
        search_sql = f"""
SELECT p.page_key
FROM vchordrq_maxsim_search(
       '{index}'::regclass,
       {query_sql},
       {candidate_limit},
       {candidate_limit}
     ) WITH ORDINALITY AS r(public_id, similarity, result_order)
JOIN {table} AS p ON p.id = r.public_id
ORDER BY r.result_order;
"""
    sql = f"""
SET statement_timeout = '{timeout_ms}ms';
SET enable_seqscan = off;
SET vchordrq.probes = '{probes}';
SET vchordrq.maxsim_refine = {refine};
SET vchordrq.maxsim_candidate_limit = {candidate_limit};
SET vchordrq.maxsim_gpu_endpoint = '{endpoint}';
SET vchordrq.maxsim_gpu_timeout_ms = {timeout_ms};
SET vchordrq.maxsim_backend = 'gpu';
SET vchordrq.maxsim_profile = {"on" if profile else "off"};
{search_sql}
"""
    started = time.perf_counter()
    result = subprocess.run(
        [*psql_command, "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"],
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(5.0, timeout_ms / 1000.0 + 5.0),
        check=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"psql failed: {diagnostic}")
    page_keys = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    invalid = [page_key for page_key in page_keys if not PAGE_KEY.fullmatch(page_key)]
    if invalid:
        raise RuntimeError(f"psql returned non-page-key output: {invalid[0]!r}")
    if len(page_keys) != len(set(page_keys)):
        raise RuntimeError("Phase 3 query returned duplicate page keys")
    query_profile = parse_profile(result.stderr)
    if profile and query_profile is None:
        raise RuntimeError("query did not emit a MaxSim profile")
    return page_keys, latency_ms, query_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--psql-command", default="psql")
    parser.add_argument("--table", default="kb_colqwen_pages")
    parser.add_argument(
        "--index",
        help="use the Phase 3B external-tensor API with this MaxSim index",
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--probes", default="8")
    parser.add_argument("--refine", type=int, default=512)
    parser.add_argument("--candidate-limit", type=positive_int, default=100)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--timeout-ms", type=positive_int, default=60000)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable vchordrq.maxsim_profile and include phase metrics",
    )
    args = parser.parse_args()

    if not IDENTIFIER.fullmatch(args.table):
        parser.error(
            "--table must be an unquoted SQL identifier, optionally schema-qualified"
        )
    if args.index is not None and not IDENTIFIER.fullmatch(args.index):
        parser.error(
            "--index must be an unquoted SQL identifier, optionally schema-qualified"
        )
    if not args.endpoint.startswith("/") or "'" in args.endpoint:
        parser.error("--endpoint must be an absolute Unix-socket path without quotes")
    if args.refine < 0:
        parser.error("--refine must be nonnegative")
    if args.query_limit < 0:
        parser.error("--query-limit must be nonnegative")
    if args.profile and args.index is None:
        parser.error("--profile requires --index")
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", args.probes):
        parser.error("--probes must be a comma-separated list of nonnegative integers")
    psql_command = shlex.split(args.psql_command)
    if not psql_command:
        parser.error("--psql-command must not be empty")

    page_keys = load_manifest(args.manifest)
    queries = load_queries(args.results_json, args.query_embeddings, args.query_limit)
    gold = np.load(args.gold, allow_pickle=False)
    per_query = []
    latencies = []
    for query_number, (query_id, query_path) in enumerate(queries):
        index_key = f"q{query_number}_idx"
        if index_key not in gold:
            raise ValueError(f"gold archive is missing {index_key}")
        gold_indices = gold[index_key].astype(np.int64, copy=False).tolist()
        if any(index < 0 or index >= len(page_keys) for index in gold_indices):
            raise ValueError(f"gold archive {index_key} contains an invalid page index")
        expected = [page_keys[index] for index in gold_indices]
        query = np.load(query_path, allow_pickle=False)
        actual, latency_ms, query_profile = execute_query(
            psql_command,
            args.table,
            args.index,
            sql_halfvec_array(query),
            args.endpoint,
            args.probes,
            args.refine,
            args.candidate_limit,
            args.timeout_ms,
            args.profile,
        )
        latencies.append(latency_ms)
        per_query.append(
            {
                "query_id": query_id,
                "query_rows": int(query.shape[0]),
                "returned_candidates": len(actual),
                "latency_ms": round(latency_ms, 3),
                "recall_at_10": recall(expected, actual, 10),
                "recall_at_20": recall(expected, actual, 20),
                "candidate_recall_at_10": candidate_recall(expected, actual, 10),
                "candidate_recall_at_20": candidate_recall(expected, actual, 20),
                "top_10": actual[:10],
                **({"profile": query_profile} if query_profile is not None else {}),
            }
        )

    output = {
        "benchmark": (
            "gbrain_vectorchord_phase3b_external_v1"
            if args.index is not None
            else "gbrain_vectorchord_phase3a_v1"
        ),
        "corpus_pages": len(page_keys),
        "query_count": len(per_query),
        "configuration": {
            "table": args.table,
            "index": args.index,
            "probes": args.probes,
            "maxsim_refine": args.refine,
            "candidate_limit": args.candidate_limit,
        },
        "metrics": {
            "recall_at_10": sum(item["recall_at_10"] for item in per_query)
            / len(per_query),
            "recall_at_20": sum(item["recall_at_20"] for item in per_query)
            / len(per_query),
            "candidate_recall_at_10": sum(
                item["candidate_recall_at_10"] for item in per_query
            )
            / len(per_query),
            "candidate_recall_at_20": sum(
                item["candidate_recall_at_20"] for item in per_query
            )
            / len(per_query),
        },
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "per_query": per_query,
    }
    profiles = [item["profile"] for item in per_query if "profile" in item]
    if profiles:
        profile_keys = sorted(set.intersection(*(set(profile) for profile in profiles)))
        output["profile_summary"] = {
            key: {
                "mean": sum(profile[key] for profile in profiles) / len(profiles),
                "p50": percentile([profile[key] for profile in profiles], 0.50),
                "p95": percentile([profile[key] for profile in profiles], 0.95),
                "max": max(profile[key] for profile in profiles),
            }
            for key in profile_keys
            if key != "schema_version"
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
