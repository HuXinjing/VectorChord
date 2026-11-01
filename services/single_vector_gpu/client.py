"""Persistent binary client; a failed request is never implicitly replayed."""
from __future__ import annotations

from dataclasses import asdict
import http.client
import threading
import time

import numpy as np

from .index import Contract
from .server import packet, unpack


class BatchClient:
    def __init__(self, host: str, port: int, token: str, contract: Contract, *, timeout: float = 300):
        self.contract = contract
        self._token = token
        self._connection = http.client.HTTPConnection(host, port, timeout=timeout)
        self._slot = threading.Lock()

    def close(self):
        self._connection.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def search(self, queries: np.ndarray, k: int, allowed_sources: list[int]):
        if not self._slot.acquire(blocking=False):
            raise RuntimeError("client_already_inflight")
        started = time.perf_counter()
        try:
            values = np.ascontiguousarray(queries, dtype="<f4")
            if values.ndim != 2 or not 1 <= len(values) <= 100000 or values.nbytes > 1024**3:
                raise ValueError("request_size_limit")
            header = packet({"version": 1, "rows": len(values), "k": k,
                             "contract": asdict(self.contract), "allowed_sources": allowed_sources}, b"")
            prepared = time.perf_counter()
            c = self._connection
            c.putrequest("POST", "/v1/single-vector/search")
            c.putheader("Authorization", "Bearer " + self._token)
            c.putheader("Content-Type", "application/octet-stream")
            c.putheader("Content-Length", str(len(header) + values.nbytes))
            c.endheaders(header)
            # Buffer-protocol send avoids another full query-matrix bytes copy.
            # Socket/kernel/network copies still exist; this is not sendfile.
            c.send(memoryview(values).cast("B"))
            sent = time.perf_counter()
            response = c.getresponse()
            length = int(response.getheader("Content-Length", "-1"))
            if response.status != 200 or length < 4 or length > 256 * 1024**2 + 8196:
                raise RuntimeError("batch_response_rejected")
            body = response.read()
            meta, payload = unpack(body)
            if meta.get("version") != 1 or meta.get("rows") != len(values) or not 1 <= meta.get("k", 0) <= k:
                raise RuntimeError("response_shape_mismatch")
            count = meta["rows"] * meta["k"]
            if len(payload) != count * 12:
                raise RuntimeError("response_length_mismatch")
            ids = np.frombuffer(payload[:count*8], dtype="<i8").reshape(meta["rows"], meta["k"])
            scores = np.frombuffer(payload[count*8:], dtype="<f4").reshape(meta["rows"], meta["k"])
            return ids, scores, {
                "full_wall_ms": (time.perf_counter()-started)*1000,
                "client_prepare_ms": (prepared-started)*1000,
                "send_ms": (sent-prepared)*1000,
                "request_bytes": len(header)+values.nbytes, "response_bytes": len(body),
                "server": meta["metrics"],
            }
        except Exception:
            self.close()
            raise
        finally:
            self._slot.release()
