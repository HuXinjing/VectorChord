"""Loopback-only experimental batch API; its token grants one frozen snapshot."""
from __future__ import annotations

import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import struct
import threading

import numpy as np

from .index import Contract, SingleVectorIndex


def packet(metadata: dict, payload: bytes) -> bytes:
    header = json.dumps(metadata, separators=(",", ":")).encode()
    return struct.pack("<I", len(header)) + header + payload


def unpack(data: bytes, max_header: int = 8192) -> tuple[dict, memoryview]:
    if len(data) < 4:
        raise ValueError("truncated_packet")
    size = struct.unpack_from("<I", data)[0]
    if size > max_header or len(data) < size + 4:
        raise ValueError("invalid_header")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError("duplicate_header_field")
            value[key] = item
        return value
    def invalid_constant(_): raise ValueError("invalid_json_number")
    header = json.loads(data[4:size+4].decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(header, dict):
        raise ValueError("invalid_header")
    return header, memoryview(data)[size+4:]


def handler(index: SingleVectorIndex, token: str, max_queries: int = 100_000):
    slot = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Headers and the binary result are separate writes. A persistent
        # request must not wait for TCP's small-packet acknowledgement timer.
        disable_nagle_algorithm = True

        def log_message(self, *_):
            pass

        def send_bytes(self, status, data, content_type="application/octet-stream"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_bytes(200 if self.path == "/health" else 404, b"{}", "application/json")

        def do_POST(self):
            self.connection.settimeout(30)
            if self.path != "/v1/single-vector/search":
                self.close_connection = True
                self.send_bytes(404, b"{}"); return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.close_connection = True
                self.send_bytes(403, b"{}"); return
            if not slot.acquire(blocking=False):
                self.close_connection = True
                self.send_bytes(503, b'{"error":"capacity"}', "application/json"); return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 4 or length > min(1024**3, max_queries * index.dimensions * 4 + 8196):
                    raise ValueError("request_size_limit")
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("truncated_packet")
                meta, payload = unpack(data)
                if set(meta) != {"version", "rows", "k", "contract", "allowed_sources"} or type(meta["version"]) is not int or meta["version"] != 1:
                    raise ValueError("invalid_request")
                rows = meta["rows"]
                if type(rows) is not int or not 1 <= rows <= max_queries:
                    raise ValueError("query_count_limit")
                if type(meta["k"]) is not int or not 1 <= meta["k"] <= 1024 or rows * min(meta["k"], len(index.sources)) * 12 > 256 * 1024**2:
                    raise ValueError("result_size_limit")
                if len(payload) != rows * index.dimensions * 4:
                    raise ValueError("invalid_payload_length")
                contract = Contract(**meta["contract"])
                queries = np.frombuffer(payload, dtype="<f4").reshape(rows, index.dimensions)
                ids, scores, metrics = index.search(queries, meta["k"], contract=contract,
                                                    allowed_sources=meta["allowed_sources"])
                result = packet({"version": 1, "rows": len(ids), "k": ids.shape[1], "metrics": metrics},
                                ids.astype("<i8").tobytes() + scores.astype("<f4").tobytes())
                self.send_bytes(200, result)
            except PermissionError:
                self.close_connection = True
                self.send_bytes(403, b'{"error":"scope"}', "application/json")
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                self.close_connection = True
                self.send_bytes(400, b'{"error":"invalid_request"}', "application/json")
            except (BrokenPipeError, ConnectionError, TimeoutError):
                self.close_connection = True
            except RuntimeError:
                self.close_connection = True
                self.send_bytes(503, b'{"error":"compute_unavailable"}', "application/json")
            finally:
                slot.release()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29417)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--query-block", type=int, default=512)
    args = parser.parse_args()
    meta = json.loads(args.metadata.read_text())
    frozen = args.data.read_bytes()
    if hashlib.sha256(frozen).hexdigest() != meta["sha256"] or meta["contract"]["generation"] != meta["sha256"]:
        raise ValueError("snapshot_hash_mismatch")
    with np.load(io.BytesIO(frozen), allow_pickle=False) as data:
        index = SingleVectorIndex(data["vectors"], data["ids"], data["sources"], Contract(**meta["contract"]),
                                  device=args.device, profile=args.profile, query_block=args.query_block)
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise ValueError("token_too_short")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(index, token))
    print(json.dumps({"event": "single_vector_ready", "port": args.port, "profile": args.profile,
                      "documents": len(index.sources), "dimensions": index.dimensions,
                      "load_ms": index.load_ms}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
