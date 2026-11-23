from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import registered


class RegisteredIpcTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.ref = b"Q" * 16
        self.candidate = {"id": 19, "score": .995}
        self.broker = registered.Broker.__new__(registered.Broker)
        self.broker.pool, self.broker.runtime = object(), "candidate"
        self.broker.config = {
            "transport": "local", "query_ipc_model": "dense", "resource_group": "search",
        }
        self.broker.manifest = lambda _: SimpleNamespace(
            identity="manifest",
            definition={"candidate_k": 1, "dimensions": 4, "model": "dense"},
            contract=lambda epoch: ("snapshot", epoch),
        )
        self.broker.policy = lambda *_: "calibrated"
        self.broker.prepare = lambda *_: True
        self.broker.admission = SimpleNamespace(local=SimpleNamespace())
        self.broker.ipc_inbox = SimpleNamespace(claim=self.claim)
        self.broker.dispatch = SimpleNamespace(cohort=nullcontext, submit=self.submit)
        self.metadata = {
            "version": 2, "manifestId": "manifest", "epoch": "epoch", "leases": ["lease"],
            "transport": "local", "rows": 1, "k": 1,
        }

    @contextmanager
    def claim(self, refs, leases, dimensions):
        self.assertEqual((refs, leases, dimensions), ([self.ref], ["lease"], 4))
        self.events.append("claim")
        try:
            yield ["device-query"]
        finally:
            self.events.append("release-reference")

    def submit(self, query, **scope):
        self.assertEqual(query, "device-query")
        self.assertEqual((scope["contract"], scope["sources"]), (("snapshot", "epoch"), [0]))
        scope["authorize"]()
        self.events.append("score")
        result = Future()
        result.set_result({"backend": "gpu", "result": [self.candidate]})
        return SimpleNamespace(result=result)

    def proof(self, pool, manifest, epoch, leases, group, operator_weight):
        self.assertEqual((epoch, leases, group), ("epoch", ["lease"], "search"))
        self.assertEqual(operator_weight, 0)
        self.events.append("borrow")
        return SimpleNamespace(
            check=lambda force=False: self.events.append("delivery" if force else "authorize"),
            release=lambda: self.events.append("release-lease"),
        )

    def test_reference_retains_the_authorized_scope_and_releases_after_delivery_check(self):
        with patch.object(registered, "Proof", side_effect=self.proof):
            result = self.broker.search(self.metadata, self.ref)
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["results"], [{"backend": "gpu", "candidates": [self.candidate]}])
        self.assertEqual(self.events, [
            "borrow", "claim", "authorize", "score", "delivery",
            "release-reference", "release-lease",
        ])

    def test_denied_borrow_never_claims_reference(self):
        with patch.object(registered, "Proof", side_effect=PermissionError()):
            with self.assertRaises(PermissionError):
                self.broker.search(self.metadata, self.ref)
        self.assertEqual(self.events, [])

    def test_no_calibration_returns_cpu_before_claim(self):
        self.broker.policy = lambda *_: None
        result = self.broker.search(self.metadata, self.ref)
        self.assertEqual(result["results"], [{"backend": "cpu"}])
        self.assertEqual(self.events, [])
