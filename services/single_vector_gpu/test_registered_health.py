"""Deployment readiness must expose a reviewed code version, not a live DB value."""
from http.server import ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.request import build_opener, ProxyHandler
from single_vector_gpu.registered import handler, WRITER_SCHEMA_VERSION


class HealthTest(unittest.TestCase):
    def test_readiness_keeps_explicit_writer_identity_when_database_is_unready(self):
        broker = SimpleNamespace(stop=threading.Event(), database_ready_at=time.monotonic(), runtime='test-runtime')
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(broker))
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            for ready in (True, False):
                if not ready:
                    broker.stop.set()
                with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{server.server_port}/health', timeout=2) as response:
                    payload = json.load(response)
                self.assertEqual(payload['ready'], ready)
                self.assertEqual(payload['writer_schema_version'], WRITER_SCHEMA_VERSION)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
