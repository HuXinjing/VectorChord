import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from single_vector_gpu.dispatch import Dispatcher, Policy, Snapshot, SnapshotCatalog
from single_vector_gpu.index import Contract
from single_vector_gpu.calibration import estimate_crossover


class Admission:
    """Test-only shared authority; production supplies the cluster lease store."""
    def __init__(self, maximum=16):
        self.maximum, self.counts, self.lock = maximum, {}, threading.Lock()

    def acquire(self, tenant, deadline):
        with self.lock:
            if self.counts.get(tenant, 0) >= self.maximum: raise RuntimeError("tenant_capacity")
            self.counts[tenant] = self.counts.get(tenant, 0) + 1
        owner = self
        class Lease:
            released = False
            def check(self):
                if self.released: raise RuntimeError("lease_lost")
            def release(self):
                with owner.lock:
                    if self.released: raise RuntimeError("double_release")
                    self.released = True
                    owner.counts[tenant] -= 1
        return Lease()


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract("tenant-a", "model-v1", "generation-a")
        self.catalog = SnapshotCatalog(100)
        self.admission = Admission()
        self.calls = []
        self.routers = []
        self.publish()

    def tearDown(self):
        for router in self.routers: router.close()
        self.assertTrue(all(n == 0 for n in self.admission.counts.values()))

    def publish(self, contract=None, size=40):
        self.catalog.publish(Snapshot(contract or self.contract, frozenset([1, 2]), 3, size,
                                      object(), "calibrated-environment", time.monotonic()+30), lambda: None)

    def router(self, gpu=None, gpu_batches=2, **overrides):
        def score(snapshot, values, k, sources):
            self.calls.append((snapshot.contract, sources, len(values)))
            return [tuple(v) for v in values]
        options = dict(crossover=2.1, calibration="calibrated-environment", expires_at=time.monotonic()+30,
                       collect_seconds=.01, max_pending=32, max_pending_per_tenant=16)
        options.update(overrides)
        result = Dispatcher(self.catalog, Policy(**options), self.admission, gpu or score, gpu_batches=gpu_batches)
        self.routers.append(result)
        return result

    def query(self, router, **overrides):
        options = dict(contract=self.contract, sources=[1], k=2, deadline=time.monotonic()+5,
                       authorize=lambda: None, cpu=lambda: "cpu-result")
        options.update(overrides)
        return router.submit(np.array([1., 2., 3.]), **options)

    def test_threshold_is_strictly_above_measured_margin(self):
        self.assertEqual(Policy(4.91, "environment", time.monotonic()+1).threshold, 6)
        self.assertEqual(Policy(5, "environment", time.monotonic()+1).threshold, 7)

    def test_small_cohort_cpu_and_large_cohort_gpu(self):
        router = self.router()
        self.assertEqual(self.query(router).result.result(2)["backend"], "cpu")
        handles = [self.query(router) for _ in range(3)]
        self.assertEqual([h.result.result(2)["backend"] for h in handles], ["gpu"]*3)
        self.assertEqual(self.calls[-1][2], 3)

    def test_equivalent_rpc_policies_batch_despite_expiry_clock_conversion_jitter(self):
        router = self.router()
        policies = [Policy(2.1, "calibrated-environment", time.monotonic()+30+i*.000001,
                           collect_seconds=.01) for i in range(3)]
        handles = [self.query(router, policy=policy) for policy in policies]
        self.assertEqual([h.result.result(2)["backend"] for h in handles], ["gpu"]*3)
        self.assertEqual(self.calls[-1][2], 3)

    def test_equivalent_policy_group_preserves_each_requests_expiration(self):
        router = self.router()
        policies = [Policy(2.1, "calibrated-environment", time.monotonic()+30,
                           collect_seconds=.01) for _ in range(2)]
        policies.append(Policy(2.1, "calibrated-environment", time.monotonic()-1,
                               collect_seconds=.01))
        handles = [self.query(router, policy=policy) for policy in policies]
        self.assertEqual([h.result.result(2)["backend"] for h in handles], ["cpu"]*3)
        self.assertEqual(self.calls, [])

    def test_sources_models_and_tenants_are_never_unioned(self):
        router = self.router()
        handles = [self.query(router, sources=[source]) for source in [1, 2, 1, 2]]
        handles += [self.query(router, contract=Contract("other", "model-v1", "generation-a")) for _ in range(2)]
        self.assertTrue(all(h.result.result(2)["backend"] == "cpu" for h in handles))
        self.assertEqual(self.calls, [])

    def test_expired_policy_or_missing_snapshot_falls_back(self):
        router = self.router(expires_at=time.monotonic()-1)
        self.assertTrue(all(h.result.result(2)["backend"] == "cpu" for h in [self.query(router) for _ in range(3)]))
        self.catalog.invalidate(self.contract.tenant, self.contract.model)
        router = self.router()
        self.assertTrue(all(h.result.result(2)["backend"] == "cpu" for h in [self.query(router) for _ in range(3)]))

    def test_gpu_failure_falls_back_once_without_model_calls(self):
        calls = []
        def fail(*args): raise RuntimeError("device_unavailable")
        router = self.router(gpu=fail)
        handles = [self.query(router, cpu=lambda: calls.append(1)) for _ in range(3)]
        self.assertTrue(all(h.result.result(2)["backend"] == "cpu" for h in handles))
        self.assertEqual(len(calls), 3)

    def test_permission_revocation_blocks_both_paths(self):
        allowed = threading.Event(); allowed.set()
        def auth():
            if not allowed.is_set(): raise PermissionError("revoked")
        def gpu(*args):
            allowed.clear()
            return [1, 2, 3]
        router = self.router(gpu=gpu)
        handles = [self.query(router, authorize=auth) for _ in range(3)]
        for handle in handles:
            with self.assertRaises(PermissionError): handle.result.result(2)

    def test_invalidation_during_gpu_discards_old_results(self):
        def gpu(*args):
            self.catalog.invalidate(self.contract.tenant, self.contract.model)
            return ["obsolete"]*3
        router = self.router(gpu=gpu)
        handles = [self.query(router) for _ in range(3)]
        self.assertTrue(all(h.result.result(2)["result"] == "cpu-result" for h in handles))

    def test_cancelled_and_expired_requests_do_not_invoke_cpu(self):
        router = self.router()
        handle = self.query(router, cpu=lambda: self.fail("cancelled work ran"))
        handle.cancel()
        with self.assertRaisesRegex(RuntimeError, "cancelled"): handle.result.result(2)
        handle = self.query(router, deadline=time.monotonic()+.001, cpu=lambda: self.fail("expired work ran"))
        with self.assertRaises(TimeoutError): handle.result.result(2)

    def test_running_cancel_keeps_quota_until_actual_completion(self):
        entered, release = threading.Event(), threading.Event()
        def cpu():
            entered.set(); release.wait(2)
        router = self.router()
        handle = self.query(router, cpu=cpu)
        self.assertTrue(entered.wait(1)); handle.cancel()
        self.assertEqual(self.admission.counts[self.contract.tenant], 1)
        release.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"): handle.result.result(2)

    def test_shared_admission_enforces_limit_across_dispatchers(self):
        self.admission.maximum = 1
        first, second = self.router(), self.router()
        entered, release = threading.Event(), threading.Event()
        def cpu(): entered.set(); release.wait(2)
        handle = self.query(first, cpu=cpu)
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError, "tenant_capacity"): self.query(second)
        finally: release.set()
        self.assertEqual(handle.result.result(2)["backend"], "cpu")

    def test_retired_snapshot_budget_stays_reserved_while_leased(self):
        version = self.catalog.acquire(self.contract, (1,), "calibrated-environment")
        self.catalog.invalidate(self.contract.tenant, self.contract.model)
        self.assertFalse(self.catalog.current(version))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.publish(Contract("tenant-a", "model-v1", "generation-b"), size=80)
        self.catalog.release(version)
        self.publish(Contract("tenant-a", "model-v1", "generation-b"), size=80)

    def test_revision_check_and_generation_immutability(self):
        with self.assertRaisesRegex(ValueError, "already_published"): self.publish()
        def fail(): raise PermissionError("changed")
        with self.assertRaises(PermissionError):
            self.catalog.publish(Snapshot(Contract("tenant-a", "model-v1", "generation-b"),
                frozenset([1]), 3, 40, None, "calibrated-environment", time.monotonic()+30), fail)
        version = self.catalog.acquire(self.contract, (1,), "calibrated-environment")
        self.assertIsNotNone(version)
        self.catalog.release(version)

    def test_busy_gpu_does_not_create_an_unbounded_device_queue(self):
        entered, release = threading.Event(), threading.Event()
        def gpu(*args):
            entered.set(); release.wait(2)
            return [1, 2, 3]
        router = self.router(gpu=gpu, gpu_batches=1)
        handles = [self.query(router) for _ in range(3)]
        self.assertTrue(entered.wait(1))
        try:
            second = [self.query(router) for _ in range(3)]
            self.assertTrue(all(h.result.result(1)["backend"] == "cpu" for h in second))
        finally: release.set()
        self.assertTrue(all(h.result.result(1)["backend"] == "gpu" for h in handles))

    def test_cpu_response_is_rechecked_after_permission_change(self):
        permitted = [True]
        def auth():
            if not permitted[0]: raise PermissionError("revoked")
        def cpu(): permitted[0] = False
        handle = self.query(self.router(), authorize=auth, cpu=cpu)
        with self.assertRaises(PermissionError): handle.result.result(1)

    def test_calibration_requires_observed_stable_crossing_and_repeats(self):
        samples = [{"count": n, "arm": arm, "arrival_p95_ms": ms}
                   for n, cpu in [(1, 3), (4, 8), (8, 16), (16, 32)]
                   for arm, ms in [("cpu", cpu), ("gpu", 10)] for _ in range(5)]
        estimate = estimate_crossover(samples)
        self.assertEqual(estimate["crossover"], 5)
        self.assertEqual(estimate["threshold"], 7)
        with self.assertRaises(ValueError): estimate_crossover(samples[:3])
        with self.assertRaisesRegex(ValueError, "not_observed"):
            estimate_crossover([{**s, "arrival_p95_ms": 1} for s in samples])


if __name__ == "__main__": unittest.main()
