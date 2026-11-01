"""Bounded CPU/GPU dispatch; authority and cluster admission remain host-owned.

The host publishes revision-qualified snapshots and invalidates them on upstream
changes. Every response is checked against live authority. This module never
infers a tenant or an ACL from a request body, or calls an encoding model.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Protocol

import numpy as np

from .index import Contract, validated


class Lease(Protocol):
    def check(self) -> None: ...
    def release(self) -> None: ...


class ClusterAdmission(Protocol):
    """Must enforce one tenant budget across all application replicas."""
    def acquire(self, tenant: str, deadline: float) -> Lease: ...


@dataclass(frozen=True)
class Policy:
    crossover: float
    calibration: str
    expires_at: float
    margin: float = 1.2
    collect_seconds: float = .002
    max_batch: int = 256
    max_pending: int = 1024
    max_pending_per_tenant: int = 64

    def __post_init__(self):
        if (not math.isfinite(self.crossover) or self.crossover < 1 or not self.calibration
            or not math.isfinite(self.expires_at) or not math.isfinite(self.margin) or self.margin < 1
            or not 0 <= self.collect_seconds <= .01 or not 1 <= self.max_batch <= 100000
            or not 1 <= self.max_pending_per_tenant <= self.max_pending):
            raise ValueError("invalid_dispatch_policy")

    @property
    def threshold(self) -> int:
        # Strictly exceed the measured crossover times the requested margin.
        return math.floor(self.crossover * self.margin) + 1


@dataclass(frozen=True)
class Snapshot:
    contract: Contract
    sources: frozenset[int]
    dimensions: int
    resident_bytes: int
    backend: object
    calibration: str
    expires_at: float


@dataclass
class _Version:
    snapshot: Snapshot
    readers: int = 0
    active: bool = True


class SnapshotCatalog:
    """Atomic publication; retired resources stay leased until real work ends.

    A producer must build outside its publication transaction, verify the live
    revision, then publish. Invalidations must precede serving a new revision.
    The host's independent authority callback is still mandatory on delivery.
    """
    def __init__(self, memory_limit_bytes: int):
        if memory_limit_bytes < 1: raise ValueError("invalid_snapshot_budget")
        self.limit = memory_limit_bytes
        self._lock = threading.RLock()
        self._active: dict[tuple[str, str], _Version] = {}
        self._retired: list[_Version] = []

    def publish(self, snapshot: Snapshot, verify_revision: Callable[[], None]):
        if (snapshot.resident_bytes < 1 or snapshot.dimensions < 1 or not snapshot.sources
            or not all((snapshot.contract.tenant, snapshot.contract.model, snapshot.contract.generation))
            or not snapshot.calibration or snapshot.expires_at <= time.monotonic()):
            raise ValueError("invalid_snapshot")
        with self._lock:
            verify_revision()
            key = (snapshot.contract.tenant, snapshot.contract.model)
            previous = self._active.get(key)
            # Publication is immutable; the same generation cannot be redefined.
            if previous and previous.snapshot.contract == snapshot.contract:
                raise ValueError("snapshot_generation_already_published")
            live = sum(v.snapshot.resident_bytes for v in self._active.values())
            live += sum(v.snapshot.resident_bytes for v in self._retired)
            if previous and not previous.readers: live -= previous.snapshot.resident_bytes
            if live + snapshot.resident_bytes > self.limit:
                raise RuntimeError("snapshot_budget_exhausted")
            if previous: self._retire(previous)
            self._active[key] = _Version(snapshot)

    def _retire(self, version):
        version.active = False
        if version.readers: self._retired.append(version)

    def invalidate(self, tenant: str, model: str):
        with self._lock:
            previous = self._active.pop((tenant, model), None)
            if previous: self._retire(previous)

    def acquire(self, contract: Contract, sources: tuple[int, ...], calibration: str):
        with self._lock:
            value = self._active.get((contract.tenant, contract.model))
            if (value is None or value.snapshot.contract != contract
                or value.snapshot.calibration != calibration
                or value.snapshot.expires_at <= time.monotonic()
                or not set(sources).issubset(value.snapshot.sources)):
                return None
            value.readers += 1
            return value

    def current(self, version):
        with self._lock:
            return version.active and version.snapshot.expires_at > time.monotonic()

    def release(self, version):
        with self._lock:
            version.readers -= 1
            if version.readers < 0: raise RuntimeError("snapshot_lease_underflow")
            if not version.active and not version.readers:
                self._retired.remove(version)


@dataclass
class Handle:
    result: Future = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self):
        # Running read-only work may not be interruptible. Its quota and GPU
        # slot stay held until it actually ends; its result is never delivered.
        self.cancelled.set()


@dataclass
class _Request:
    vector: np.ndarray
    contract: Contract
    sources: tuple[int, ...]
    k: int
    deadline: float
    authorize: Callable[[], None]
    cpu: Callable[[], object]
    lease: Lease
    handle: Handle
    arrived: float
    started: bool = False

    def check(self):
        if self.handle.cancelled.is_set() or self.handle.result.cancelled(): raise RuntimeError("query_cancelled")
        if time.monotonic() >= self.deadline: raise TimeoutError("query_deadline")
        self.lease.check()
        self.authorize()


class Dispatcher:
    """Collect compatible arrivals; CPU is the default and failure fallback.

    gpu(snapshot, query_matrix, k, sources) returns one result per input row.
    It must bound its own transport/compute deadline. Admission must be the
    existing shared tenant authority; this layer does not grant extra quota.
    """
    def __init__(self, catalog: SnapshotCatalog, policy: Policy, admission: ClusterAdmission,
                 gpu: Callable, *, cpu_workers: int = 4, observe: Callable | None = None,
                 failure_cooldown_seconds: float = 1, gpu_batches: int = 2):
        if not 1 <= cpu_workers <= 256: raise ValueError("invalid_cpu_workers")
        if type(gpu_batches) is not int or not 1 <= gpu_batches <= 8:
            raise ValueError("invalid_gpu_queue_budget")
        if not math.isfinite(failure_cooldown_seconds) or not 0 <= failure_cooldown_seconds <= 60:
            raise ValueError("invalid_gpu_cooldown")
        self.catalog, self.policy, self.admission, self.gpu = catalog, policy, admission, gpu
        self._failure_cooldown = failure_cooldown_seconds
        self._gpu_retry_at = 0.
        self.observe = observe or (lambda event: None)
        self._condition = threading.Condition()
        self._groups: dict[tuple, deque] = {}
        self._tenants: dict[str, int] = {}
        self._pending = 0
        self._closed = False
        self._cpu = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="vector-cpu")
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector-gpu")
        # One execution thread; the default permits one running batch and one
        # waiting batch, both still charged to their original tenant leases.
        self._gpu_slot = threading.BoundedSemaphore(gpu_batches)
        self._scheduler = threading.Thread(target=self._run, name="vector-dispatch", daemon=True)
        self._scheduler.start()

    def submit(self, vector, *, contract: Contract, sources: list[int], k: int,
               deadline: float, authorize: Callable[[], None], cpu: Callable[[], object]) -> Handle:
        raw = np.asarray(vector)
        if raw.ndim != 1 or not 1 <= len(raw) <= 65536:
            raise ValueError("invalid_dispatch_dimensions")
        value = validated(raw[None, :]).copy()[0]
        if (type(k) is not int or not 1 <= k <= 1024 or not math.isfinite(deadline)
            or deadline <= time.monotonic() or not all((contract.tenant, contract.model, contract.generation))
            or any(type(s) is not int for s in sources)):
            raise ValueError("invalid_dispatch_request")
        allowed = tuple(sorted(set(sources)))
        authorize()
        lease = self.admission.acquire(contract.tenant, deadline)
        handle = Handle()
        request = _Request(value, contract, allowed, k, deadline, authorize, cpu, lease, handle, time.monotonic())
        key = (contract, allowed, k, len(value))
        try:
            with self._condition:
                if self._closed: raise RuntimeError("dispatcher_closed")
                if (self._pending >= self.policy.max_pending or
                    self._tenants.get(contract.tenant, 0) >= self.policy.max_pending_per_tenant):
                    raise RuntimeError("dispatcher_capacity")
                self._groups.setdefault(key, deque()).append(request)
                self._pending += 1
                self._tenants[contract.tenant] = self._tenants.get(contract.tenant, 0) + 1
                self._condition.notify()
        except BaseException:
            lease.release()
            raise
        return handle

    def _event(self, **event):
        # No vectors, prompts, credentials or user-defined labels in telemetry.
        try: self.observe(event)
        except Exception: pass

    def _finish(self, request, result=None, error=None):
        try:
            request.lease.release()
        except Exception as failure:
            error = failure
        finally:
            with self._condition:
                self._pending -= 1
                tenant = request.contract.tenant
                self._tenants[tenant] -= 1
                if not self._tenants[tenant]: del self._tenants[tenant]
            if not request.handle.result.cancelled():
                if error is None: request.handle.result.set_result(result)
                else: request.handle.result.set_exception(error)

    @staticmethod
    def _start(request):
        if not request.started:
            request.started = True
            if not request.handle.result.set_running_or_notify_cancel():
                raise RuntimeError("query_cancelled")

    def _cpu_one(self, request, reason):
        try:
            self._start(request)
            request.check()
            result = request.cpu()
            request.check()
        except BaseException as error:
            self._finish(request, error=error)
        else:
            self._finish(request, result={"backend": "cpu", "result": result, "reason": reason})
        self._event(backend="cpu", reason=reason, rows=1)

    def _gpu_batch(self, requests, version):
        valid = []
        try:
            for request in requests:
                try:
                    self._start(request)
                    request.check()
                except BaseException as error: self._finish(request, error=error)
                else: valid.append(request)
            if not valid: return
            if len(valid) < self.policy.threshold:
                for request in valid: self._cpu.submit(self._cpu_one, request, "batch_shrank")
                return
            if not self.catalog.current(version):
                raise RuntimeError("snapshot_changed")
            snapshot = version.snapshot
            if any(len(r.vector) != snapshot.dimensions for r in valid):
                raise ValueError("snapshot_dimension_mismatch")
            result = self.gpu(snapshot, np.stack([r.vector for r in valid]), valid[0].k, valid[0].sources)
            if len(result) != len(valid): raise RuntimeError("gpu_result_rows")
            if not self.catalog.current(version): raise RuntimeError("snapshot_changed")
        except Exception:
            # Search is read-only. This never repeats encoding/model charges.
            # Authority/deadline is rechecked by the existing CPU implementation.
            for request in valid: self._cpu.submit(self._cpu_one, request, "gpu_unavailable")
            with self._condition:
                self._gpu_retry_at = time.monotonic() + self._failure_cooldown
            self._event(backend="gpu", reason="fallback", rows=len(valid))
        else:
            for request, value in zip(valid, result):
                try: request.check()
                except BaseException as error: self._finish(request, error=error)
                else: self._finish(request, result={"backend": "gpu", "result": value})
            self._event(backend="gpu", reason="batch", rows=len(valid))
        finally:
            self.catalog.release(version)
            self._gpu_slot.release()

    def _run(self):
        while True:
            with self._condition:
                if self._closed and not self._groups: return
                if not self._groups:
                    self._condition.wait(); continue
                # Dict insertion order rotates serviced groups to the end.
                key, group = next(iter(self._groups.items()))
                delay = group[0].arrived + self.policy.collect_seconds - time.monotonic()
                if delay > 0:
                    self._condition.wait(delay); continue
                requests = [group.popleft() for _ in range(min(len(group), self.policy.max_batch))]
                del self._groups[key]
                if group: self._groups[key] = group
            first = requests[0]
            eligible = (len(requests) >= self.policy.threshold
                        and self.policy.expires_at > time.monotonic() >= self._gpu_retry_at)
            version = self.catalog.acquire(first.contract, first.sources, self.policy.calibration) if eligible else None
            if version and self._gpu_slot.acquire(blocking=False):
                self._gpu.submit(self._gpu_batch, requests, version)
            else:
                if version: self.catalog.release(version)
                for request in requests: self._cpu.submit(self._cpu_one, request, "below_threshold_or_not_ready")

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._scheduler.join()
        # GPU failures may still schedule a CPU fallback. Drain GPU first.
        self._gpu.shutdown(wait=True)
        self._cpu.shutdown(wait=True)
