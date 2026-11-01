"""Internal registered-snapshot broker. No query can supply SQL or credentials.

Manifests are produced by NeoBrain's existing SQL planner. The broker reads
vectors directly from the colocated database, borrows already-admitted tenant
leases, and returns only candidate identities/scores. It never encodes text.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import signal
import struct
import threading
import time
import uuid

import numpy as np
import psycopg
import torch
from psycopg.adapt import Loader
from psycopg.pq import Format
from psycopg.types import TypeInfo

from .dispatch import Dispatcher, Policy, Snapshot, SnapshotCatalog
from .index import Contract, SingleVectorIndex
from .server import unpack


EPOCH_SQL = """SELECT COALESCE(jsonb_object_agg(source_id,units),'{}'::jsonb)::text FROM (
 SELECT source_id,sum(units)::text AS units FROM knowledge_vector_gpu_changes
 WHERE source_id=ANY(%s::text[]) GROUP BY source_id) witnesses"""


class Pool:
    def __init__(self, database, maximum=4):
        self.database = database
        self.slots = threading.BoundedSemaphore(maximum)
        self.idle = queue.LifoQueue()

    @contextmanager
    def connection(self):
        if not self.slots.acquire(timeout=2): raise RuntimeError("database_capacity")
        conn = None
        try:
            try: conn = self.idle.get_nowait()
            except queue.Empty: conn = psycopg.connect(self.database, autocommit=True, connect_timeout=3,
                application_name="neobrain-schema-writer-v160",
                options="-c statement_timeout=3000 -c lock_timeout=1000 -c idle_in_transaction_session_timeout=10000")
            yield conn
        finally:
            if conn is not None:
                if conn.closed or conn.broken: conn.close()
                else: self.idle.put(conn)
            self.slots.release()

    def close(self):
        while not self.idle.empty(): self.idle.get_nowait().close()


@dataclass(frozen=True)
class Manifest:
    identity: str
    definition: dict

    def contract(self, epoch):
        return Contract(self.definition["tenant"], self.definition["model"], epoch, self.identity)


class Proof:
    """One child reference per parent. Releasing a timed-out parent cannot free
    its weighted capacity until all real GPU child work has ended.
    """
    def __init__(self, pool, manifest, epoch, parents, group):
        self.pool, self.manifest, self.epoch = pool, manifest, epoch
        self.parents = sorted(set(parents))
        self.children = [str(uuid.uuid4()) for _ in self.parents]
        self.group = group
        self._lock = threading.RLock()
        self._valid_until = 0.
        with pool.connection() as conn, conn.transaction():
            if conn.execute(EPOCH_SQL, (manifest.definition["sources"],)).fetchone()[0] != epoch:
                raise RuntimeError("snapshot_changed")
            owned = conn.execute("""SELECT bool_and(COALESCE(knowledge_vector_gpu_borrow(child,parent,%s,%s),false))
              FROM (SELECT child,parent FROM unnest(%s::text[],%s::text[])AS tickets(child,parent)
                    ORDER BY parent) ordered""",
              (manifest.identity,group,self.children,self.parents)).fetchone()[0]
            if owned is not True: raise PermissionError("lease_not_owned")
        try: self.check(force=True)
        except Exception:
            self.release()
            raise

    def check(self, force=False):
        with self._lock:
            if not force and time.monotonic() < self._valid_until: return
            checked_at = time.monotonic()
            with self.pool.connection() as conn:
                row = conn.execute("""SELECT count(*),
                  EXTRACT(epoch FROM min(expires_at)-clock_timestamp()),
                  (SELECT COALESCE(jsonb_object_agg(source_id,units),'{}'::jsonb)::text FROM (
                    SELECT source_id,sum(units)::text AS units FROM knowledge_vector_gpu_changes
                    WHERE source_id=ANY(%s::text[]) GROUP BY source_id) witnesses)
                  FROM retrieval_capacity_leases WHERE token=ANY(%s::text[])
                  AND tenant_key=%s AND resource_group=%s AND NOT closing AND expires_at>now()""",
                  (self.manifest.definition["sources"],self.parents,self.manifest.definition["tenant"],self.group)).fetchone()
            if row[0] != len(self.parents): raise PermissionError("lease_not_owned")
            if row[2] != self.epoch: raise RuntimeError("snapshot_changed")
            # A short cohort receipt avoids repeating the same PostgreSQL proof
            # per row. It never outlives a parent lease. Final delivery always
            # forces a fresh check; invalidation cannot escape as a result.
            self._valid_until = checked_at+max(0,min(.05,float(row[1] or 0)))

    def release(self):
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT knowledge_vector_gpu_release(%s::text[],%s::text[])", (self.children,self.parents))


class BorrowedAdmission:
    def __init__(self): self.local = threading.local()

    def acquire(self, tenant, deadline):
        proof, parent = self.local.proof, self.local.parent
        if tenant != proof.manifest.definition["tenant"]: raise PermissionError("tenant_changed")
        class Borrowed:
            capacity_key = parent
            def check(self): proof.check()
            # The HTTP batch owner releases the reference after ALL row work
            # ends. This is not permission to delete the parent's real lease.
            def release(self): pass
        return Borrowed()


class Broker:
    def __init__(self, config):
        self.config = config
        identity = [str(torch.__version__), str(np.__version__), config.get("device","cuda:0"),
                    str(torch.get_num_threads()), str(config.get("resident_bytes",3*1024**3))]
        if str(config.get("device","cuda:0")).startswith("cuda"):
            properties = torch.cuda.get_device_properties(config.get("device","cuda:0"))
            identity += [str(getattr(properties,"uuid",properties.name)),str(properties.total_memory)]
        for name in ["registered.py","dispatch.py","index.py"]:
            identity.append(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
        self.runtime = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.pool = Pool(config["database"], config.get("database_connections", 4))
        with self.pool.connection() as conn:
            privileges = conn.execute("""SELECT rolsuper OR rolbypassrls OR
              has_table_privilege(current_user,'content_chunks','INSERT,UPDATE,DELETE')
              FROM pg_roles WHERE rolname=current_user""").fetchone()[0]
            if privileges and not config.get("allow_privileged_test_database",False):
                raise PermissionError("gpu_database_role_must_be_read_only")
        self.catalog = SnapshotCatalog(config.get("resident_bytes", 3*1024**3))
        self.admission = BorrowedAdmission()
        self.proofs = threading.local()
        self.loading = set()
        self.load_lock = threading.Lock()
        self.loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector-snapshot")
        self.stop = threading.Event()
        self.database_ready_at = 0.
        self.dispatch = Dispatcher(self.catalog, Policy(1, "disabled", 0, max_pending=256), self.admission,
                                   self._score, cpu_workers=2, observe=lambda event: print(json.dumps({"event":"vector_dispatch",**event}),flush=True))
        self.maintenance = threading.Thread(target=self._maintain, daemon=True)
        self.maintenance.start()

    def manifest(self, identity):
        with self.pool.connection() as conn:
            row = conn.execute("""SELECT definition,tenant_key,source_ids FROM knowledge_vector_gpu_manifests
              WHERE manifest_id=%s AND expires_at>now()""", (identity,)).fetchone()
        if row is None or hashlib.sha256(row[0].encode()).hexdigest() != identity:
            raise PermissionError("manifest_not_registered")
        definition = json.loads(row[0])
        if (set(definition) != {"version", "tenant", "sources", "model", "dimensions", "candidate_k", "select", "params", "search", "ranking", "settings"}
            or definition["version"] != 1 or definition["tenant"] != row[1] or definition["sources"] != row[2]
            or not isinstance(definition["select"], str) or not definition["select"].lstrip().upper().startswith("SELECT ")
            or not 1 <= definition["dimensions"] <= 65536 or not 1 <= definition["candidate_k"] <= 1024):
            raise PermissionError("manifest_invalid")
        return Manifest(identity, definition)

    def policy(self, manifest, epoch, force=False):
        if force:
            return Policy(1, manifest.identity+epoch, time.monotonic()+30, force_gpu=True)
        with self.pool.connection() as conn:
            row = conn.execute("""SELECT crossover,margin,expires_at FROM knowledge_vector_gpu_calibrations
              WHERE manifest_id=%s AND epoch=%s AND transport_fingerprint=%s AND runtime_fingerprint=%s AND expires_at>now()""",
              (manifest.identity, epoch, self.config["transport"], self.runtime)).fetchone()
        if row is None: return None
        return Policy(float(row[0]), manifest.identity+epoch,
                      time.monotonic()+max(0, row[2].timestamp()-time.time()), margin=max(2.,float(row[1])))

    def prepare(self, manifest, epoch):
        calibration = manifest.identity+epoch
        contract = manifest.contract(epoch)
        value = self.catalog.acquire(contract, (0,), calibration)
        if value:
            self.catalog.release(value)
            return True
        with self.load_lock:
            if manifest.identity not in self.loading and len(self.loading) < 8:
                self.loading.add(manifest.identity)
                self.loader.submit(self._load, manifest, epoch)
        return False

    def _load(self, manifest, epoch):
        try:
            self.catalog.invalidate(manifest.definition["tenant"], manifest.definition["model"], manifest.identity)
            with self.pool.connection() as conn, conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                conn.execute("SET LOCAL statement_timeout='8s'")
                conn.execute("SET LOCAL lock_timeout='100ms'")
                if conn.execute(EPOCH_SQL, (manifest.definition["sources"],)).fetchone()[0] != epoch:
                    raise RuntimeError("snapshot_changed")
                count = psycopg.RawCursor(conn).execute("SELECT count(*) FROM ("+manifest.definition["select"]+") v",
                                                     manifest.definition["params"]).fetchone()[0]
                size = count * (manifest.definition["dimensions"]*4+8)
                if (not count or size>self.config.get("max_snapshot_bytes",1024**3)
                    or size+self.catalog.resident_bytes+256*1024**2 > self.catalog.limit):
                    raise RuntimeError("snapshot_budget")
                info = TypeInfo.fetch(conn, "vector")
                class VectorLoader(Loader):
                    format = Format.BINARY
                    def load(self, data):
                        dimensions, reserved = struct.unpack_from(">hh", data)
                        if reserved or len(data) != 4+dimensions*4: raise ValueError("vector_wire_shape")
                        return np.frombuffer(data, dtype=">f4", offset=4).astype(np.float32)
                conn.adapters.register_loader(info.oid, VectorLoader)
                vectors = np.empty((count, manifest.definition["dimensions"]), dtype=np.float32)
                identities = np.empty(count, dtype=np.int64)
                position = 0
                with psycopg.RawServerCursor(conn, "snapshot_"+uuid.uuid4().hex) as cursor:
                    cursor.execute(manifest.definition["select"], manifest.definition["params"], binary=True)
                    for identity, vector in cursor:
                        identities[position], vectors[position] = identity, vector
                        position += 1
                if position != count: raise RuntimeError("snapshot_rows")
            contract = manifest.contract(epoch)
            index = SingleVectorIndex(vectors, identities, np.zeros(count, dtype=np.int64), contract,
                                      device=self.config.get("device", "cuda:0"), profile="fp32",
                                      memory_limit_bytes=self.catalog.limit)
            snapshot = Snapshot(contract, frozenset([0]), vectors.shape[1], size, index,
                                manifest.identity+epoch, time.monotonic()+300)
            def verify():
                with self.pool.connection() as conn:
                    if conn.execute(EPOCH_SQL, (manifest.definition["sources"],)).fetchone()[0] != epoch:
                        raise RuntimeError("snapshot_changed")
            self.catalog.publish(snapshot, verify)
            print(json.dumps({"event": "snapshot_ready", "rows": count, "dimensions": vectors.shape[1]}), flush=True)
        except Exception as error:
            print(json.dumps({"event": "snapshot_unavailable", "error_type": type(error).__name__,
                              "sqlstate": getattr(error,"sqlstate",None)}), flush=True)
        finally:
            with self.load_lock: self.loading.discard(manifest.identity)

    def _maintain(self):
        while not self.stop.is_set():
            try:
                self.catalog.prune()
                with self.pool.connection() as conn:
                    identities = conn.execute("""SELECT manifest_id FROM knowledge_vector_gpu_manifests
                      WHERE expires_at>now() ORDER BY expires_at DESC LIMIT 8""").fetchall()
                self.database_ready_at = time.monotonic()
                for (identity,) in identities:
                    manifest = self.manifest(identity)
                    with self.pool.connection() as conn:
                        epoch = conn.execute(EPOCH_SQL, (manifest.definition["sources"],)).fetchone()[0]
                    self.prepare(manifest, epoch)
                if self.config.get("compact_changes",False):
                    with self.pool.connection() as conn: conn.execute("SELECT knowledge_vector_gpu_compact()")
            except Exception: self.database_ready_at = 0.
            self.stop.wait(2)

    def _score(self, snapshot, values, k, sources):
        ids, scores, _ = snapshot.backend.search(values, k, contract=snapshot.contract, allowed_sources=sources)
        return [[{"id": int(identity), "score": float(score)} for identity, score in zip(row, scored)
                if identity >= 0] for row, scored in zip(ids, scores)]

    def search(self, metadata, payload, force=False):
        required = {"version", "manifestId", "epoch", "leases", "transport", "rows", "k"}
        if (set(metadata) != required or type(metadata["version"]) is not int or metadata["version"] != 1
            or type(metadata["rows"]) is not int or not 1 <= metadata["rows"] <= 64
            or type(metadata["k"]) is not int or not 1 <= metadata["k"] <= 1024
            or not isinstance(metadata["leases"], list) or len(metadata["leases"]) != metadata["rows"]
            or any(not isinstance(s, str) or len(s)>128 for s in metadata["leases"])
            or metadata["transport"] != self.config["transport"]):
            raise ValueError("invalid_vector_request")
        manifest = self.manifest(metadata["manifestId"])
        if metadata["k"] != manifest.definition["candidate_k"]: raise ValueError("candidate_contract_changed")
        dimensions = manifest.definition["dimensions"]
        if len(payload) != metadata["rows"]*dimensions*4: raise ValueError("vector_wire_shape")
        values = np.frombuffer(payload, dtype="<f4").reshape(metadata["rows"], dimensions)
        if not np.isfinite(values).all() or np.any(np.all(values == 0, axis=1)): raise ValueError("invalid_vectors")
        policy = self.policy(manifest, metadata["epoch"], force)
        result = {"version": 1, "manifestId": manifest.identity, "epoch": metadata["epoch"], "runtime": self.runtime,
                  "results": [{"backend": "cpu"} for _ in values]}
        if policy is None or not self.prepare(manifest, metadata["epoch"]): return result
        proof = Proof(self.pool, manifest, metadata["epoch"], metadata["leases"], self.config["resource_group"])
        handles = []
        try:
            if force:
                with self.pool.connection() as conn:
                    weight = conn.execute("SELECT sum(weight) FROM retrieval_capacity_leases WHERE token=ANY(%s::text[])", (proof.parents,)).fetchone()[0]
                if int(weight or 0) < len(values): raise PermissionError("calibration_capacity")
            with self.dispatch.cohort():
                for vector, parent in zip(values, metadata["leases"]):
                    self.admission.local.proof, self.admission.local.parent = proof, parent
                    handles.append(self.dispatch.submit(vector, contract=manifest.contract(metadata["epoch"]), sources=[0],
                        k=metadata["k"], deadline=time.monotonic()+2, authorize=proof.check, cpu=lambda: None, policy=policy))
            rows = [handle.result.result() for handle in handles]
            proof.check(force=True)
            result["results"] = [{"backend": row["backend"], **({"candidates": row["result"]} if row["backend"] == "gpu" else {})} for row in rows]
            return result
        finally:
            # Do not detach active work when a client disconnects or a row fails.
            for handle in handles:
                try: handle.result.result()
                except Exception: pass
            proof.release()
            self.admission.local.proof = self.admission.local.parent = None

    def close(self):
        self.stop.set()
        self.maintenance.join()
        self.loader.shutdown(wait=True)
        self.dispatch.close()
        self.pool.close()


def handler(broker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        def log_message(self, *_): pass
        def send(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.send(200 if self.path == "/health" else 404,
                      {"ready": not broker.stop.is_set() and time.monotonic()-broker.database_ready_at<15,
                       "runtime": broker.runtime})
        def do_POST(self):
            self.connection.settimeout(5)
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer "+broker.config["token"]):
                self.close_connection = True; self.send(403, {"error": "unauthorized"}); return
            if self.path != "/v1/registered-vector/search":
                self.close_connection = True; self.send(404, {}); return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if not 4 <= size <= 64*65536*4+32772: raise ValueError("request_size")
                meta, payload = unpack(self.rfile.read(size),max_header=32768)
                force = False
                supplied = self.headers.get("X-Vector-Operator", "")
                if supplied:
                    expected = broker.config.get("operator_token", "")
                    if not expected or not hmac.compare_digest(supplied, expected): raise PermissionError("operator_denied")
                    force = True
                self.send(200, broker.search(meta, payload, force))
            except PermissionError:
                self.close_connection = True; self.send(403, {"error": "scope_or_lease"})
            except (ValueError, TypeError, KeyError):
                self.close_connection = True; self.send(400, {"error": "request_contract"})
            except Exception:
                self.close_connection = True; self.send(503, {"error": "backend_not_ready"})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.config.stat().st_mode & 0o077:
        raise ValueError("broker_config_permissions")
    config = json.loads(args.config.read_text())
    if len(config.get("token", "")) < 32 or not config.get("transport") or not config.get("resource_group"):
        raise ValueError("broker_config_invalid")
    broker = Broker(config)
    server = ThreadingHTTPServer((config.get("host", "127.0.0.1"), config.get("port", 29421)), handler(broker))
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps({"event": "registered_vector_ready", "port": server.server_port}), flush=True)
    try: server.serve_forever()
    finally:
        server.server_close()
        broker.close()


if __name__ == "__main__": main()
