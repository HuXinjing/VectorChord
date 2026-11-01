"""Paired measurements against a disposable pgvector database and batch API."""
from __future__ import annotations

import argparse
import atexit
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import struct
import time

import numpy as np
import psycopg
from psycopg.adapt import Dumper
from psycopg.pq import Format
from psycopg.types import TypeInfo

from .index import Contract, SingleVectorIndex, normalized
from .server import packet, unpack


def connection(config, ef=200, exact=False):
    conn = psycopg.connect(host=config["pgHost"], port=config.get("pgPort", 5432),
                          dbname="single_vector_bench", user="postgres", password=config["pgPassword"],
                          autocommit=True, prepare_threshold=0)
    info = TypeInfo.fetch(conn, "vector")
    class VectorDumper(Dumper):
        oid = info.oid
        format = Format.BINARY
        def dump(self, value):
            return struct.pack(">hh", len(value), 0) + np.asarray(value, dtype=">f4").tobytes()
    conn.adapters.register_dumper(np.ndarray, VectorDumper)
    conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef),))
    conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
    conn.execute("SET statement_timeout = '30s'")
    if exact:
        conn.execute("SET enable_indexscan=off")
        conn.execute("SET enable_bitmapscan=off")
    return conn


def sql_query(q, sources, k=20):
    # Pure distance ORDER BY retains the actual current ANN index contract.
    return ("WITH candidates AS MATERIALIZED (SELECT id, embedding <=> %b::vector AS distance "
            "FROM single_vectors WHERE source=ANY(%s) ORDER BY distance LIMIT 100) "
            "SELECT id,1-distance AS score FROM candidates ORDER BY distance LIMIT %s", (q, sources, k))


def prepare(config, data, output):
    dimensions = int(data["vectors"].shape[1])
    if not 1 <= dimensions <= 2000:
        raise ValueError("unsupported_pgvector_hnsw_dimensions")
    with psycopg.connect(host=config["pgHost"], port=config.get("pgPort", 5432),
                         dbname="single_vector_bench", user="postgres", password=config["pgPassword"],
                         autocommit=True) as conn:
        assert conn.execute("SELECT current_database()").fetchone()[0] == "single_vector_bench"
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
        conn.execute(f"CREATE TABLE single_vectors(id bigint PRIMARY KEY,source integer NOT NULL,embedding vector({dimensions}) NOT NULL)")
        started = time.perf_counter()
        with conn.cursor().copy("COPY single_vectors(id,source,embedding) FROM STDIN") as copy:
            for i, source, vector in zip(data["ids"], data["sources"], data["vectors"]):
                copy.write_row((int(i), int(source), "[" + ",".join(format(float(v), ".9g") for v in vector) + "]"))
        loaded = time.perf_counter()
        conn.execute("SET maintenance_work_mem='512MB'")
        conn.execute("CREATE INDEX single_vector_hnsw ON single_vectors USING hnsw(embedding vector_cosine_ops)")
        conn.execute("ANALYZE single_vectors")
        info = {"load_seconds": loaded-started, "index_seconds": time.perf_counter()-loaded,
                "extensions": conn.execute("SELECT extname,extversion FROM pg_extension WHERE extname IN ('vector','vchord')").fetchall()}
    with connection(config) as conn:
        text, params = sql_query(data["holdout_queries"][0], sorted(set(map(int, data["sources"]))))
        plan = conn.execute("EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) " + text, params).fetchone()[0]
        info["plan"] = plan
        if "single_vector_hnsw" not in json.dumps(plan):
            raise RuntimeError("Baseline did not use the HNSW index")
    output.write_text(json.dumps(info, indent=2))


def api_search(config, meta, queries, sources, k=20):
    conn = http.client.HTTPConnection("127.0.0.1", config.get("apiPort", 29417), timeout=300)
    data = packet({"version": 1, "rows": len(queries), "k": k, "contract": meta["contract"],
                   "allowed_sources": sources}, np.ascontiguousarray(queries, dtype="<f4").tobytes())
    started = time.perf_counter()
    conn.request("POST", "/v1/single-vector/search", data,
                 {"Authorization": "Bearer " + config["apiToken"], "Content-Type": "application/octet-stream"})
    response = conn.getresponse(); body = response.read(); conn.close()
    if response.status != 200:
        raise RuntimeError("batch_api_status_" + str(response.status))
    header, payload = unpack(body)
    count = header["rows"] * header["k"]
    if len(payload) != count * 12:
        raise RuntimeError("invalid_response_length")
    ids = np.frombuffer(payload[:count*8], dtype="<i8").reshape(header["rows"], header["k"]).copy()
    scores = np.frombuffer(payload[count*8:], dtype="<f4").reshape(header["rows"], header["k"]).copy()
    return ids, scores, {"wall_ms": (time.perf_counter()-started)*1000,
                         "request_bytes": len(data), "response_bytes": len(body), "server": header["metrics"]}


def cpu_search(config, queries, sources, concurrency=1, ef=200, exact=False, budget=300, pool=None):
    started = time.perf_counter()
    def worker(offset):
        completed = 0; samples = []; ids = []; scores = []
        with (nullcontext(pool[0][offset]) if pool else connection(config, ef, exact)) as conn:
            for position in range(offset, len(queries), concurrency):
                if time.perf_counter()-started > budget: break
                tick = time.perf_counter()
                rows = conn.execute(*sql_query(queries[position], sources)).fetchall()
                samples.append((time.perf_counter()-tick)*1000)
                completed += 1
                if len(queries) <= 2048:
                    ids.append((position, [int(x[0]) for x in rows]))
                    scores.append((position, [float(x[1]) for x in rows]))
        return completed, samples, ids, scores
    if pool:
        results = list(pool[1].map(worker, range(concurrency)))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(worker, range(concurrency)))
    elapsed = time.perf_counter()-started
    samples = [s for r in results for s in r[1]]
    return {"requested": len(queries), "completed": sum(r[0] for r in results), "wall_ms": elapsed*1000,
            "throughput": sum(r[0] for r in results)/elapsed,
            "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
            "concurrency": concurrency, "ef_search": ef, "exact": exact,
            "ids": sorted(v for r in results for v in r[2]),
            "scores": sorted(v for r in results for v in r[3])}


def correctness(index, data, queries, sources, fp16=False):
    ids, scores, metrics = index.search(queries, 20, contract=index.contract, allowed_sources=sources)
    documents = normalized(data["vectors"])
    oracle = normalized(queries) @ documents.T
    selected = np.argsort(-oracle, axis=1)[:, :20]
    ref_ids = data["ids"][selected]
    by_id = {int(v): i for i, v in enumerate(data["ids"])}
    error = max(float(np.max(np.abs(scores[i] - oracle[i, [by_id[int(v)] for v in ids[i]]]))) for i in range(len(queries)))
    overlap = float(np.mean([len(set(a)&set(b))/20 for a, b in zip(ids, ref_ids)]))
    kth = np.take_along_axis(oracle, selected[:, -1:], axis=1)
    valid = all(np.all(oracle[i, [by_id[int(v)] for v in ids[i]]] >= kth[i, 0]-2e-5) for i in range(len(queries)))
    passed = error <= (5e-4 if fp16 else 2e-5) and (overlap >= .99 if fp16 else valid)
    return {"max_score_error": error, "strict_top20_overlap": overlap,
            "tie_equivalent_topk": bool(valid), "passed": bool(passed), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--counts", default="1,32,256,2048,100000")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.data, allow_pickle=False)); meta = json.loads(args.metadata.read_text())
    config = json.loads(args.config.read_text()); sources = sorted(set(map(int, data["sources"])))
    if args.prepare:
        prepare(config, data, args.output / "postgres-plan.json"); return
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != meta["sha256"]:
        raise RuntimeError("fixture_hash_mismatch")
    rng = np.random.default_rng(20261101)
    # Document embeddings are explicitly held out, not reported as natural-language queries.
    bases = data["holdout_queries"]
    counts = [int(x) for x in args.counts.split(",")]
    q = bases[rng.integers(0, len(bases), size=max(counts))].copy()
    q += rng.normal(0, .005, q.shape).astype(np.float32)
    (args.output / "query-provenance.json").write_text(json.dumps({
        "seed": 20261101, "queries": len(q), "numpy": np.__version__,
        "query_sha256": hashlib.sha256(q.tobytes()).hexdigest(), "data_sha256": meta["sha256"],
    }, indent=2))
    contract = Contract(**meta["contract"])
    index = SingleVectorIndex(data["vectors"], data["ids"], data["sources"], contract, profile=args.profile)
    checked = correctness(index, data, q[655:719] if len(q) >= 719 else q[:32], sources, args.profile=="fp16")
    (args.output / "correctness.json").write_text(json.dumps(checked, indent=2))
    if not checked["passed"]:
        raise RuntimeError("correctness_gate_failed")
    pool = None
    if not args.gpu_only:
        connections = [connection(config) for _ in range(4)]
        executor = ThreadPoolExecutor(max_workers=4)
        def close_pool():
            executor.shutdown(wait=True)
            for conn in connections: conn.close()
        atexit.register(close_pool)
        pool = (connections, executor)
    warm_shapes = {min(count, index.query_block) for count in counts}
    warm_shapes.update(count % index.query_block for count in counts if count % index.query_block)
    for _ in range(2):
        for shape in sorted(warm_shapes):
            index.search(q[:shape], 20, contract=contract, allowed_sources=sources)
        if not args.gpu_only: cpu_search(config, q[:32], sources, concurrency=4, pool=pool)
    order = random.Random(20261101); rows = []
    for count in counts:
        for block in range(args.blocks):
            arms = ["gpu_resident"] if args.gpu_only else ["cpu_ann", "gpu_resident"]
            order.shuffle(arms)
            for arm in arms:
                if arm == "cpu_ann":
                    result = cpu_search(config, q[:count], sources, concurrency=4 if count>=2048 else 1, pool=pool)
                    # Retain raw IDs for quality assessment only in bounded samples.
                    result.pop("ids", None); result.pop("scores", None)
                else:
                    _, _, result = index.search(q[:count], 20, contract=contract, allowed_sources=sources)
                    result.update(completed=count, throughput=count*1000/result["wall_ms"])
                row = {"count": count, "block": block, "arm": arm, "profile": args.profile, **result}
                rows.append(row)
                (args.output / "samples.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
        # Measure the binary API separately, without hiding serialization or result transfer.
        if args.profile == "fp32" and not args.gpu_only:
            _, _, api = api_search(config, meta, q[:count], sources)
            (args.output / ("api-"+str(count)+".json")).write_text(json.dumps(api, indent=2))
    if args.gpu_only: return
    quality = []
    for ef in [200, 1000]:
        result = cpu_search(config, q[:256], sources, ef=ef)
        expected, _, _ = index.search(q[:256], 20, contract=contract, allowed_sources=sources)
        quality.append({"ef": ef, "strict_top20_overlap": float(np.mean([
            len(set(row)&set(expected[position]))/20 for position,row in result["ids"]])),
                        "p50_ms": result["p50_ms"], "p95_ms": result["p95_ms"]})
    (args.output / "ann-quality.json").write_text(json.dumps(quality, indent=2))
    with connection(config, exact=True) as exact_conn, ThreadPoolExecutor(max_workers=1) as exact_executor:
        exact_pool = ([exact_conn], exact_executor)
        for _ in range(2): cpu_search(config, q[:2], sources, pool=exact_pool, exact=True)
        exact_samples = [cpu_search(config, q[:min(256, len(q))], sources, pool=exact_pool, exact=True) for _ in range(args.blocks)]
    for sample in exact_samples:
        sample.pop("ids", None); sample.pop("scores", None)
    (args.output / "cpu-exact-samples.json").write_text(json.dumps(exact_samples, indent=2))


if __name__ == "__main__": main()
