import unittest
import http.client
from http.server import ThreadingHTTPServer
import threading
import numpy as np

from single_vector_gpu.index import Contract, SingleVectorIndex, normalized
from single_vector_gpu.server import packet, unpack, handler


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract("tenant-a", "test-encoder-v1", "snapshot-7")
        self.rng = np.random.default_rng(427)
        self.docs = self.rng.normal(size=(73, 17)).astype(np.float32)
        self.ids = np.arange(73, dtype=np.int64) + 1000
        self.sources = np.arange(73) % 3
        self.index = SingleVectorIndex(self.docs, self.ids, self.sources, self.contract,
                                       device="cpu", query_block=7, document_block=11)

    def test_chunk_boundaries_match_cosine_oracle_and_identity(self):
        queries = self.rng.normal(size=(19, 17)).astype(np.float32)
        ids, scores, _ = self.index.search(queries, 9, contract=self.contract, allowed_sources=[0, 2])
        ref = normalized(queries) @ normalized(self.docs).T
        ref[:, self.sources == 1] = -np.inf
        selected = np.argsort(-ref, axis=1)[:, :9]
        np.testing.assert_array_equal(ids, self.ids[selected])
        np.testing.assert_allclose(scores, np.take_along_axis(ref, selected, axis=1), atol=1e-6)

    def test_negative_scores_empty_scope_and_partial_topk(self):
        docs = np.array([[1., 0.], [-1., 0.], [0., 1.]], dtype=np.float32)
        index = SingleVectorIndex(docs, np.array([7, 9, 11]), np.array([0, 1, 1]), self.contract,
                                  device="cpu", query_block=1, document_block=2)
        ids, scores, _ = index.search(np.array([[1., 0.]]), 3, contract=self.contract, allowed_sources=[1])
        self.assertEqual(ids.tolist(), [[11, 9, -1]])
        self.assertEqual(scores[0, :2].tolist(), [0., -1.])
        ids, _, _ = index.search(docs[:1], 2, contract=self.contract, allowed_sources=[])
        self.assertEqual(ids.tolist(), [[-1, -1]])

    def test_scope_and_version_are_checked_before_search(self):
        for contract in [Contract("tenant-b", "test-encoder-v1", "snapshot-7"),
                         Contract("tenant-a", "test-encoder-v2", "snapshot-7"),
                         Contract("tenant-a", "test-encoder-v1", "snapshot-8")]:
            with self.assertRaises(PermissionError):
                self.index.search(self.docs[:1], 3, contract=contract, allowed_sources=[0])
        with self.assertRaises(PermissionError):
            self.index.search(self.docs[:1], 3, contract=self.contract, allowed_sources=[999])

    def test_invalid_input_and_resident_budget(self):
        for query in [np.zeros((1, 17)), np.full((1, 17), np.nan), np.ones((1, 16))]:
            with self.assertRaises(ValueError):
                self.index.search(query, 2, contract=self.contract, allowed_sources=[0])
        with self.assertRaises(ValueError):
            SingleVectorIndex(self.docs, self.ids, self.sources, self.contract, device="cpu", memory_limit_bytes=1)
        with self.assertRaises(ValueError):
            SingleVectorIndex(self.docs, np.zeros(73), self.sources, self.contract, device="cpu")

    def test_query_normalization_preserves_extreme_finite_nonzero_vectors(self):
        queries = np.array([[1e-38, -1e-38], [1e38, -1e38]], dtype=np.float32)
        index = SingleVectorIndex(np.array([[1., -1.], [-1., 1.]], dtype=np.float32),
                                  np.array([5, 6]), np.array([1, 1]), self.contract, device="cpu")
        ids, scores, _ = index.search(queries, 2, contract=self.contract, allowed_sources=[1])
        np.testing.assert_array_equal(ids, [[5, 6], [5, 6]])
        np.testing.assert_allclose(scores, [[1., -1.], [1., -1.]], atol=1e-6)

    def test_binary_packet_has_no_pickle_dependency_and_rejects_truncation(self):
        meta, data = unpack(packet({"version": 1}, b"\x01\x02"))
        self.assertEqual(meta, {"version": 1}); self.assertEqual(bytes(data), b"\x01\x02")
        for value in [b"", b"\xff\xff\xff\xff", b"\x01\x00\x00\x00"]:
            with self.assertRaises(ValueError): unpack(value)

    def test_loopback_api_auth_scope_and_binary_result(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.index, "a" * 64))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            query = self.docs[:2]
            metadata = {"version": 1, "rows": 2, "k": 3,
                        "contract": self.contract.__dict__, "allowed_sources": [0, 1, 2]}
            for token, contract, expected in [("bad", self.contract.__dict__, 403),
                ("a"*64, Contract("tenant-b", "test-encoder-v1", "snapshot-7").__dict__, 403),
                ("a"*64, self.contract.__dict__, 200)]:
                metadata["contract"] = contract
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                connection.request("POST", "/v1/single-vector/search", packet(metadata, query.tobytes()),
                                   {"Authorization": "Bearer " + token})
                response = connection.getresponse(); body = response.read(); connection.close()
                self.assertEqual(response.status, expected)
                if expected == 200:
                    meta, payload = unpack(body)
                    self.assertEqual((meta["rows"], meta["k"], len(payload)), (2, 3, 72))
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=3)


if __name__ == "__main__": unittest.main()
