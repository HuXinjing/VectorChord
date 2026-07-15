# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Benchmark PostgreSQL/pgvector HNSW with original text-embedding vectors."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)?$")


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


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(format(float(value), ".8g") for value in vector) + "]"


def run_psql(command: list[str], sql: str) -> str:
    result = subprocess.run(
        [*command, "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"],
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"psql failed: {diagnostic}")
    return result.stdout


def prepare_table(
    command: list[str], table: str, vectors: np.ndarray, m: int, ef_construction: int
) -> float:
    dimension = int(vectors.shape[1])
    process = subprocess.Popen(
        [*command, "-X", "-q", "-v", "ON_ERROR_STOP=1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    started = time.perf_counter()
    process.stdin.write(
        f"DROP TABLE IF EXISTS {table};\n"
        f"CREATE TABLE {table} (id integer PRIMARY KEY, embedding vector({dimension}) NOT NULL);\n"
        f"COPY {table} (id, embedding) FROM STDIN;\n"
    )
    for index, vector in enumerate(vectors):
        process.stdin.write(f"{index}\t{vector_text(vector)}\n")
    process.stdin.write(
        "\\.\n"
        f"CREATE INDEX ON {table} USING hnsw (embedding vector_ip_ops) "
        f"WITH (m={m}, ef_construction={ef_construction});\n"
        f"ANALYZE {table};\n"
    )
    process.stdin.close()
    stdout = process.stdout.read() if process.stdout is not None else ""
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"psql prepare failed: {stderr.strip() or stdout.strip()}")
    return (time.perf_counter() - started) * 1000


def execute_queries(
    command: list[str], table: str, queries: np.ndarray, ef_search: int, top_k: int
) -> list[tuple[int, float, list[int]]]:
    dimension = int(queries.shape[1])
    query_values = ",\n".join(
        f"({index}, '{vector_text(vector)}'::vector({dimension}))"
        for index, vector in enumerate(queries)
    )
    sql = f"""
SET hnsw.ef_search = {ef_search};
CREATE TEMP TABLE single_vector_queries (
  query_index integer PRIMARY KEY,
  embedding vector({dimension}) NOT NULL
) ON COMMIT PRESERVE ROWS;
INSERT INTO single_vector_queries VALUES
{query_values};
CREATE TEMP TABLE single_vector_results (
  query_index integer PRIMARY KEY,
  latency_ms double precision NOT NULL,
  ids integer[] NOT NULL
) ON COMMIT PRESERVE ROWS;
DO $bench$
DECLARE
  q record;
  started timestamptz;
  result_ids integer[];
BEGIN
  -- Warm PostgreSQL buffers and the HNSW path before recording latency.
  FOR q IN SELECT * FROM single_vector_queries ORDER BY query_index LOOP
    EXECUTE 'SELECT array_agg(id ORDER BY distance, id) FROM (
               SELECT id, embedding <#> $1 AS distance
                 FROM {table}
                ORDER BY embedding <#> $1
                LIMIT {top_k}
             ) ranked'
      INTO result_ids USING q.embedding;
  END LOOP;
  FOR q IN SELECT * FROM single_vector_queries ORDER BY query_index LOOP
    started := clock_timestamp();
    EXECUTE 'SELECT array_agg(id ORDER BY distance, id) FROM (
               SELECT id, embedding <#> $1 AS distance
                 FROM {table}
                ORDER BY embedding <#> $1
                LIMIT {top_k}
             ) ranked'
      INTO result_ids USING q.embedding;
    INSERT INTO single_vector_results VALUES (
      q.query_index,
      extract(epoch FROM clock_timestamp() - started) * 1000.0,
      result_ids
    );
  END LOOP;
END
$bench$;
SELECT query_index, latency_ms, array_to_string(ids, ',')
  FROM single_vector_results
 ORDER BY query_index;
"""
    rows = []
    for line in run_psql(command, sql).splitlines():
        if not line.strip():
            continue
        query_index, latency_ms, ids = line.split("|")
        rows.append(
            (
                int(query_index),
                float(latency_ms),
                [int(value) for value in ids.split(",")],
            )
        )
    return rows


def recall(expected: np.ndarray, actual: list[int], top_k: int) -> float:
    return len(set(expected[:top_k].tolist()).intersection(actual[:top_k])) / top_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-vectors", type=Path)
    parser.add_argument("--query-vectors", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument(
        "--corpus-json",
        type=Path,
        help="RAG/GBrain corpus JSON whose _source.q_1024_vec is the stored text embedding",
    )
    parser.add_argument(
        "--query-dataset",
        type=Path,
        help="query JSON whose q_1024_vec is the original text query embedding",
    )
    parser.add_argument("--psql-command", default="psql")
    parser.add_argument("--table", default="public.tilemaxsim_single_vector_bench")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not IDENTIFIER.fullmatch(args.table):
        parser.error("--table must be an unquoted SQL identifier")
    if min(args.m, args.ef_construction, args.ef_search, args.top_k) <= 0:
        parser.error("HNSW parameters and top-k must be positive")
    command = shlex.split(args.psql_command)
    if not command:
        parser.error("--psql-command must not be empty")

    corpus_chunk_ids: list[str] | None = None
    corpus_doc_names: list[str] | None = None
    query_records: list[dict[str, object]] | None = None
    if args.corpus_json or args.query_dataset:
        if not args.corpus_json or not args.query_dataset:
            parser.error("--corpus-json and --query-dataset must be used together")
        corpus_records = json.loads(args.corpus_json.read_text(encoding="utf-8"))
        query_records = json.loads(args.query_dataset.read_text(encoding="utf-8"))
        if not isinstance(corpus_records, list) or not isinstance(query_records, list):
            raise ValueError("corpus and query JSON inputs must be arrays")
        page_vectors = np.asarray(
            [record["_source"]["q_1024_vec"] for record in corpus_records],
            dtype=np.float32,
        )
        query_vectors = np.asarray(
            [record["q_1024_vec"] for record in query_records],
            dtype=np.float32,
        )
        corpus_chunk_ids = [str(record["_id"]) for record in corpus_records]
        corpus_doc_names = [str(record["_source"]["docnm_kwd"]) for record in corpus_records]
    else:
        if not args.page_vectors or not args.query_vectors or not args.gold:
            parser.error(
                "use --corpus-json/--query-dataset or all of "
                "--page-vectors/--query-vectors/--gold"
            )
        page_vectors = np.load(args.page_vectors, mmap_mode="r", allow_pickle=False)
        query_vectors = np.load(args.query_vectors, mmap_mode="r", allow_pickle=False)
    if page_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("page and query vectors must be rank-2 arrays")
    if page_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("page and query dimensions differ")
    prepare_ms = (
        prepare_table(
            command, args.table, page_vectors, args.m, args.ef_construction
        )
        if args.prepare
        else None
    )
    query_rows = execute_queries(
        command, args.table, query_vectors, args.ef_search, args.top_k
    )
    cases = []
    if query_records is not None and corpus_chunk_ids is not None and corpus_doc_names is not None:
        for query_index, latency_ms, ids in query_rows:
            query_record = query_records[query_index]
            gold_chunk = str(query_record["gold_chunk_id"])
            gold_doc = str(query_record["gold_doc_name"])
            ranked_chunks = [corpus_chunk_ids[index] for index in ids]
            ranked_docs = [corpus_doc_names[index] for index in ids]
            cases.append(
                {
                    "query_index": query_index,
                    "query_id": query_record.get("query_id"),
                    "latency_ms": latency_ms,
                    "chunk_hit_at_1": gold_chunk in ranked_chunks[:1],
                    "chunk_hit_at_10": gold_chunk in ranked_chunks[:10],
                    "chunk_hit_at_20": gold_chunk in ranked_chunks[:20],
                    "doc_hit_at_1": gold_doc in ranked_docs[:1],
                    "doc_hit_at_10": gold_doc in ranked_docs[:10],
                    "doc_hit_at_20": gold_doc in ranked_docs[:20],
                    "top_20_chunk_ids": ranked_chunks[:20],
                    "top_20_doc_names": ranked_docs[:20],
                }
            )
    else:
        assert args.gold is not None
        gold = np.load(args.gold, allow_pickle=False)
        for query_index, latency_ms, ids in query_rows:
            expected = gold[f"q{query_index}_idx"].astype(np.int64, copy=False)
            cases.append(
                {
                    "query_index": query_index,
                    "latency_ms": latency_ms,
                    "recall_at_10": recall(expected, ids, 10),
                    "recall_at_20": recall(expected, ids, 20),
                    "top_20": ids[:20],
                }
            )
    latencies = [item["latency_ms"] for item in cases]
    report = {
        "benchmark": "postgres_pgvector_hnsw_text_embedding_v2",
        "corpus_vectors": int(page_vectors.shape[0]),
        "query_vectors": int(query_vectors.shape[0]),
        "dimension": int(page_vectors.shape[1]),
        "configuration": {
            "table": args.table,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "top_k": args.top_k,
        },
        "prepare_ms": prepare_ms,
        "latency_ms": summary(latencies),
        "quality": (
            {
                key: statistics.fmean(float(item[key]) for item in cases)
                for key in (
                    "chunk_hit_at_1",
                    "chunk_hit_at_10",
                    "chunk_hit_at_20",
                    "doc_hit_at_1",
                    "doc_hit_at_10",
                    "doc_hit_at_20",
                )
            }
            if query_records is not None
            else {
                "mean_recall_at_10": statistics.fmean(
                    item["recall_at_10"] for item in cases
                ),
                "mean_recall_at_20": statistics.fmean(
                    item["recall_at_20"] for item in cases
                ),
            }
        ),
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
