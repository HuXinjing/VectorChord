# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import (
    decode_response,
    external_request_frame,
    scheduled_external_request_frame,
)
from services.tilemaxsim_shard import ImmutableShardWriter


class RustDaemonTest(unittest.TestCase):
    @staticmethod
    def _release_binary() -> Path:
        return Path(__file__).parent / "tilemaxsimd" / "target" / "release" / "tilemaxsimd"

    def run_daemon(
        self,
        devices: list[int],
        documents: list[np.ndarray] | None = None,
        query: np.ndarray | None = None,
        gpu_memory_gb: str = "0.05",
        workspace_gb: str = "0.02",
        resident: bool = False,
        scheduled: bool = False,
        tcp: bool = False,
        io_pipeline: str = "overlap",
        io_batch_gb: str = "0.25",
    ) -> tuple[str, list[tuple[int, float]]]:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_root = root / "cache"
            shard_root.mkdir()
            if documents is None:
                documents = [
                    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2"),
                    np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype="<f2"),
                ]
            if query is None:
                query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            references = []
            writer = ImmutableShardWriter(
                shard_root, target_bytes=4096, alignment=256, fsync=False
            )
            try:
                for document in documents:
                    payload = document.tobytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    writer.add(
                        digest,
                        payload,
                        document.shape[0],
                        document.shape[1],
                        "float16",
                    )
                    references.append(f"sha256://{digest}")
                writer.finish()
            finally:
                writer.close()
            candidates = [
                (11 + index, reference, document.tolist())
                for index, (reference, document) in enumerate(
                    zip(references, documents, strict=True)
                )
            ]
            if scheduled:
                frame, _ = scheduled_external_request_frame(
                    901,
                    protocol.DTYPE_F16,
                    query.tolist(),
                    "model@1",
                    candidates,
                    "tenant-a",
                    17,
                    8_000,
                )
            else:
                frame, _ = external_request_frame(
                    901,
                    protocol.DTYPE_F16,
                    query.tolist(),
                    "model@1",
                    candidates,
                )
            socket_path = root / "tilemaxsimd.sock"
            status_socket_path = root / "tilemaxsimd-status.sock"
            tcp_port = None
            if tcp:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    tcp_port = reservation.getsockname()[1]
            command = [
                os.fspath(binary),
                "--socket",
                os.fspath(socket_path),
                "--status-socket",
                os.fspath(status_socket_path),
            ]
            if tcp_port is not None:
                command.extend(("--listen", f"127.0.0.1:{tcp_port}"))
            for device in devices:
                command.extend(("--gpu-memory-gb", f"{device}={gpu_memory_gb}"))
            command.extend(
                (
                    "--gpu-workspace-gb",
                    workspace_gb,
                    "--host-cache-gb",
                    "0.01",
                    "--io-pipeline",
                    io_pipeline,
                    "--io-batch-gb",
                    io_batch_gb,
                    "--contract-root",
                    f"model@1={shard_root}",
                    "--once",
                )
            )
            if resident:
                manifest = root / "resident.jsonl"
                with manifest.open("w", encoding="utf-8") as stream:
                    for index, (reference, document) in enumerate(
                        zip(references, documents, strict=True)
                    ):
                        digest = reference.removeprefix("sha256://")
                        stream.write(
                            json.dumps(
                                {
                                    "page_key": str(index),
                                    "tensor_ref": reference,
                                    "tensor_rows": document.shape[0],
                                    "tensor_dim": document.shape[1],
                                    "tensor_dtype": "float16",
                                    "tensor_checksum": f"sha256:{digest}",
                                    "canonical_bytes": document.nbytes,
                                }
                            )
                            + "\n"
                        )
                command.extend(
                    (
                        "--gpu-cache-mode",
                        "resident",
                        "--resident-manifest",
                        f"model@1={manifest}",
                    )
                )
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(1000):
                    if (
                        socket_path.exists() and status_socket_path.exists()
                    ) or process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                self.assertTrue(socket_path.exists())
                self.assertTrue(status_socket_path.exists())
                for path, expected in (
                    ("/livez", b'200 OK'),
                    ("/healthz", b'200 OK'),
                    ("/metrics", b"tilemaxsim_ready 1"),
                ):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as status:
                        status.connect(os.fspath(status_socket_path))
                        status.sendall(
                            f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
                        )
                        response_parts = []
                        while part := status.recv(4096):
                            response_parts.append(part)
                    self.assertIn(expected, b"".join(response_parts))
                    if path == "/metrics":
                        metrics_body = b"".join(response_parts)
                        self.assertIn(b"tilemaxsim_gpu_cache_bytes", metrics_body)
                        self.assertIn(b"tilemaxsim_host_cache_bytes", metrics_body)
                        self.assertIn(b"tilemaxsim_storage_read_bytes_total", metrics_body)
                        self.assertIn(b"tilemaxsim_io_pipeline_enabled", metrics_body)
                        self.assertNotIn(b"tenant-a", metrics_body)
                probe = subprocess.run(
                    [
                        os.fspath(binary.with_name("tilemaxsimctl")),
                        "--socket",
                        os.fspath(status_socket_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr)
                family = socket.AF_INET if tcp_port is not None else socket.AF_UNIX
                with socket.socket(family, socket.SOCK_STREAM) as connection:
                    if tcp_port is None:
                        connection.connect(os.fspath(socket_path))
                    else:
                        connection.connect(("127.0.0.1", tcp_port))
                    connection.sendall(frame)
                    header = protocol.receive_exact(connection, protocol.HEADER.size)
                    body_bytes = protocol.HEADER.unpack(header)[4]
                    response = header + protocol.receive_exact(connection, body_bytes)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
                request_id, status, results = decode_response(response)
                self.assertEqual(request_id, 901)
                self.assertEqual(status, 0)
                self.assertEqual(
                    [item[0] for item in results],
                    list(range(11, 11 + len(documents))),
                )
                query_f32 = query.astype(np.float32)
                for (_, actual), document in zip(results, documents, strict=True):
                    expected = float(
                        (query_f32 @ document.astype(np.float32).T).max(axis=1).sum()
                    )
                    self.assertAlmostEqual(actual, expected, delta=0.02)
                self.assertIn('"event":"tilemaxsim_rust_ready"', output)
                assert isinstance(results, list)
                return output, results
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    def test_gpu_assignment_is_required_before_startup(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            socket_path = root / "tilemaxsimd.sock"
            completed = subprocess.run(
                [
                    os.fspath(binary),
                    "--socket",
                    os.fspath(socket_path),
                    "--contract-root",
                    f"model@1={root}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("--gpu-memory-gb", completed.stderr)
            self.assertFalse(socket_path.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_unavailable_configured_gpu_fails_before_socket_ready(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ImmutableShardWriter(
                root, target_bytes=4096, alignment=256, fsync=False
            )
            payload = np.asarray([[1.0, 0.0]], dtype="<f2").tobytes()
            writer.add(
                hashlib.sha256(payload).hexdigest(),
                payload,
                1,
                2,
                "float16",
            )
            writer.finish()
            writer.close()
            socket_path = root / "tilemaxsimd.sock"
            completed = subprocess.run(
                [
                    os.fspath(binary),
                    "--socket",
                    os.fspath(socket_path),
                    "--gpu-memory-gb",
                    f"{torch.cuda.device_count() + 100}=0.05",
                    "--gpu-workspace-gb",
                    "0.02",
                    "--host-cache-gb",
                    "0.01",
                    "--contract-root",
                    f"model@1={root}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            output = completed.stdout + completed.stderr
            self.assertNotEqual(completed.returncode, 0, output)
            self.assertIn("cudaSetDevice", output)
            self.assertFalse(socket_path.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_external_v2_shard_round_trip_matches_protocol_oracle(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        self.run_daemon([device])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_io_overlap_matches_serial_scores_and_reports_pipeline_cycles(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        serial_output, serial_results = self.run_daemon(
            [device], io_pipeline="serial", io_batch_gb="0.000000004"
        )
        overlap_output, overlap_results = self.run_daemon(
            [device], io_pipeline="overlap", io_batch_gb="0.000000004"
        )
        self.assertEqual(overlap_results, serial_results)
        serial_events = [
            json.loads(line) for line in serial_output.splitlines() if line.startswith("{")
        ]
        overlap_events = [
            json.loads(line)
            for line in overlap_output.splitlines()
            if line.startswith("{")
        ]
        serial_cache = next(
            event["cache"]
            for event in serial_events
            if event.get("event") == "tilemaxsim_rust_request"
        )
        overlap_cache = next(
            event["cache"]
            for event in overlap_events
            if event.get("event") == "tilemaxsim_rust_request"
        )
        self.assertEqual(serial_cache["io_pipeline"]["mode"], "serial")
        self.assertEqual(overlap_cache["io_pipeline"]["mode"], "overlap")
        self.assertEqual(serial_cache["io_pipeline"]["overlap_cycles"], 0)
        self.assertGreaterEqual(overlap_cache["io_pipeline"]["overlap_cycles"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tcp_scoring_round_trip_matches_protocol_oracle(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        output, results = self.run_daemon([device], tcp=True, scheduled=True)
        self.assertIn('"listen":"127.0.0.1:', output)
        self.assertEqual([candidate for candidate, _ in results], [11, 12])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_exact_scores_are_repeatable_across_cuda_contexts(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        generator = np.random.default_rng(20260715)
        document = generator.normal(size=(79, 64)).astype("<f2")
        query = generator.normal(size=(31, 64)).astype("<f2")
        observed = [
            self.run_daemon([device], [document], query)[1][0][1]
            for _ in range(5)
        ]
        self.assertEqual(observed, [observed[0]] * len(observed))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_query_rows_can_cross_the_cuda_grid_y_limit(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        document = np.asarray([[1.0, 0.0]], dtype="<f2")
        query = np.zeros((65_536, 2), dtype="<f2")
        query[:, 0] = 1.0
        _, results = self.run_daemon([device], [document], query)
        self.assertAlmostEqual(results[0][1], 65_536.0, delta=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_published_object_is_queryable_without_daemon_restart(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_root = root / "objects-root"
            object_root.mkdir()
            socket_path = root / "tilemaxsimd.sock"
            process = subprocess.Popen(
                [
                    os.fspath(binary),
                    "--socket", os.fspath(socket_path),
                    "--gpu-memory-gb", f"{device}=0.05",
                    "--gpu-workspace-gb", "0.02",
                    "--host-cache-gb", "0.01",
                    "--contract-root", f"model@1={object_root}",
                    "--once",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(1000):
                    if socket_path.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                document = np.asarray(
                    [[1.0, 0.0], [0.0, 1.0]], dtype="<f2"
                )
                payload = document.tobytes()
                digest = hashlib.sha256(payload).hexdigest()
                published = subprocess.run(
                    [
                        os.fspath(binary.with_name("tilemaxsimctl")),
                        "publish-object",
                        "--root", os.fspath(object_root),
                        "--rows", "2",
                        "--dimension", "2",
                        "--dtype", "float16",
                        "--expected-sha256", digest,
                    ],
                    input=payload,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(published.returncode, 0, published.stderr.decode())
                descriptor = json.loads(published.stdout)
                self.assertEqual(descriptor["tensor_ref"], f"sha256://{digest}")

                frame, _ = external_request_frame(
                    902,
                    protocol.DTYPE_F16,
                    document.tolist(),
                    "model@1",
                    [(77, descriptor["tensor_ref"], document.tolist())],
                )
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(os.fspath(socket_path))
                    connection.sendall(frame)
                    header = protocol.receive_exact(connection, protocol.HEADER.size)
                    body_bytes = protocol.HEADER.unpack(header)[4]
                    response = header + protocol.receive_exact(connection, body_bytes)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
                request_id, status, results = decode_response(response)
                self.assertEqual((request_id, status), (902, 0))
                self.assertEqual(results[0][0], 77)
                self.assertAlmostEqual(results[0][1], 2.0, delta=0.02)
                events = [
                    json.loads(line) for line in output.splitlines()
                    if line.startswith("{")
                ]
                request = next(
                    event for event in events
                    if event.get("event") == "tilemaxsim_rust_request"
                )
                self.assertEqual(request["cache"]["batch_read_calls"], 1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_sigkill_restart_recovers_stale_socket_and_cache_root(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_root = root / "cache"
            shard_root.mkdir()
            document = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            payload = document.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer = ImmutableShardWriter(
                shard_root, target_bytes=4096, alignment=256, fsync=False
            )
            try:
                writer.add(digest, payload, 2, 2, "float16")
                writer.finish()
            finally:
                writer.close()
            socket_path = root / "tilemaxsimd.sock"
            command = [
                os.fspath(binary),
                "--socket", os.fspath(socket_path),
                "--gpu-memory-gb", f"{device}=0.05",
                "--gpu-workspace-gb", "0.02",
                "--host-cache-gb", "0.01",
                "--contract-root", f"model@1={shard_root}",
            ]

            crashed = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(1000):
                if socket_path.exists() or crashed.poll() is not None:
                    break
                time.sleep(0.01)
            self.assertIsNone(crashed.poll())
            crashed.kill()
            crashed.communicate(timeout=5)
            self.assertTrue(socket_path.exists())

            status_socket_path = root / "restarted-status.sock"
            restarted = subprocess.Popen(
                [
                    *command,
                    "--status-socket", os.fspath(status_socket_path),
                    "--once",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(1000):
                    if status_socket_path.exists() or restarted.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(restarted.poll())
                frame, _ = external_request_frame(
                    903,
                    protocol.DTYPE_F16,
                    document.tolist(),
                    "model@1",
                    [(88, f"sha256://{digest}", document.tolist())],
                )
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(os.fspath(socket_path))
                    connection.sendall(frame)
                    header = protocol.receive_exact(connection, protocol.HEADER.size)
                    body_bytes = protocol.HEADER.unpack(header)[4]
                    response = header + protocol.receive_exact(connection, body_bytes)
                output, _ = restarted.communicate(timeout=10)
                self.assertEqual(restarted.returncode, 0, output)
                request_id, status, results = decode_response(response)
                self.assertEqual((request_id, status), (903, 0))
                self.assertEqual(results[0][0], 88)
            finally:
                if restarted.poll() is None:
                    restarted.terminate()
                    restarted.wait(timeout=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_external_v3_scheduled_round_trip_hashes_tenant_and_preserves_priority(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        output, _ = self.run_daemon([device], scheduled=True)
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        request = next(
            event for event in events if event.get("event") == "tilemaxsim_rust_request"
        )
        self.assertNotIn("tenant", request)
        self.assertRegex(request["tenant_hash"], r"^[0-9a-f]{16}$")
        self.assertEqual(request["priority"], 17)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "two CUDA devices are unavailable",
    )
    def test_multi_gpu_scheduler_uploads_and_scores_on_each_device(self) -> None:
        output, _ = self.run_daemon([0, 1])
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        request_event = next(
            event for event in events if event.get("event") == "tilemaxsim_rust_request"
        )
        devices = request_event["cache"]["devices"]
        self.assertEqual(len(devices), 2)
        self.assertEqual([device["h2d_batches"] for device in devices], [1, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_one_request_larger_than_gpu_cache_is_scored_in_chunks(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        rows, dimension = 480, 320
        first = np.zeros((rows, dimension), dtype="<f2")
        second = np.zeros((rows, dimension), dtype="<f2")
        first[:, 0] = 1.0
        second[:, 1] = 1.0
        query = np.zeros((2, dimension), dtype="<f2")
        query[0, 0] = 1.0
        query[1, 1] = 1.0
        output, _ = self.run_daemon(
            [device],
            [first, second],
            query,
            gpu_memory_gb="0.0010",
            workspace_gb="0.0005",
        )
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        request_event = next(
            event for event in events if event.get("event") == "tilemaxsim_rust_request"
        )
        cache = request_event["cache"]["devices"][0]
        self.assertEqual(cache["h2d_batches"], 2)
        self.assertGreaterEqual(cache["gpu_admission_rejections"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_resident_manifest_is_pinned_before_socket_ready(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        output, _ = self.run_daemon([device], resident=True)
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        prewarm = next(
            event
            for event in events
            if event.get("event") == "tilemaxsim_rust_prewarm_complete"
        )
        self.assertEqual(prewarm["entries"], 2)
        self.assertEqual(prewarm["cache"]["devices"][0]["gpu_pinned_entries"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_concurrent_readers_do_not_block_fair_priority_scheduler(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_root = root / "cache"
            shard_root.mkdir()
            document = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            payload = document.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer = ImmutableShardWriter(
                shard_root, target_bytes=4096, alignment=256, fsync=False
            )
            try:
                writer.add(digest, payload, 2, 2, "float16")
                writer.finish()
            finally:
                writer.close()
            query = document.tolist()
            priorities = [-2, 9, 3, 7, 0, 5, 1, 8]
            frames = []
            for index, priority in enumerate(priorities):
                frame, _ = scheduled_external_request_frame(
                    1_000 + index,
                    protocol.DTYPE_F16,
                    query,
                    "model@1",
                    [(11, f"sha256://{digest}", document.tolist())],
                    f"tenant-{index % 2}",
                    priority,
                    8_000,
                )
                frames.append(frame)
            socket_path = root / "tilemaxsimd.sock"
            process = subprocess.Popen(
                [
                    os.fspath(binary),
                    "--socket", os.fspath(socket_path),
                    "--gpu-memory-gb", f"{device}=0.05",
                    "--gpu-workspace-gb", "0.02",
                    "--host-cache-gb", "0.01",
                    "--contract-root", f"model@1={shard_root}",
                    "--scheduler-batch-window-ms", "100",
                    "--socket-io-timeout-ms", "500",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            slow = None
            try:
                for _ in range(1000):
                    if socket_path.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                slow.connect(os.fspath(socket_path))
                slow.sendall(b"VCTM")

                def call(frame: bytes) -> int:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                        connection.settimeout(10)
                        connection.connect(os.fspath(socket_path))
                        connection.sendall(frame)
                        header = protocol.receive_exact(connection, protocol.HEADER.size)
                        body_bytes = protocol.HEADER.unpack(header)[4]
                        response = header + protocol.receive_exact(connection, body_bytes)
                    request_id, status, _ = decode_response(response)
                    self.assertEqual(status, 0)
                    return request_id

                with ThreadPoolExecutor(max_workers=len(frames)) as executor:
                    completed = list(executor.map(call, frames))
                self.assertEqual(set(completed), set(range(1_000, 1_000 + len(frames))))
                slow.close()
                slow = None
                process.terminate()
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
                events = [
                    json.loads(line)
                    for line in output.splitlines()
                    if line.startswith("{")
                ]
                processed = [
                    event["priority"]
                    for event in events
                    if event.get("event") == "tilemaxsim_rust_request"
                ]
                # All public priorities share the default fair-priority band.
                # Urgency breaks the first tie, then equal-cost tenants
                # alternate instead of one tenant draining its whole queue.
                self.assertEqual(processed, [9, 3, 8, 1, 7, 0, 5, -2])
            finally:
                if slow is not None:
                    slow.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_overload_rejects_one_tenant_without_crashing_daemon(self) -> None:
        binary = self._release_binary()
        if not binary.exists():
            self.skipTest("release tilemaxsimd binary has not been built")
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_root = root / "cache"
            shard_root.mkdir()
            document = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            payload = document.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer = ImmutableShardWriter(
                shard_root, target_bytes=4096, alignment=256, fsync=False
            )
            try:
                writer.add(digest, payload, 2, 2, "float16")
                writer.finish()
            finally:
                writer.close()
            frames = [
                scheduled_external_request_frame(
                    request_id,
                    protocol.DTYPE_F16,
                    document.tolist(),
                    "model@1",
                    [(11, f"sha256://{digest}", document.tolist())],
                    "noisy-tenant",
                    0,
                    8_000,
                )[0]
                for request_id in (2_001, 2_002)
            ]
            socket_path = root / "tilemaxsimd.sock"
            process = subprocess.Popen(
                [
                    os.fspath(binary),
                    "--socket", os.fspath(socket_path),
                    "--gpu-memory-gb", f"{device}=0.05",
                    "--gpu-workspace-gb", "0.02",
                    "--host-cache-gb", "0.01",
                    "--contract-root", f"model@1={shard_root}",
                    "--max-queued-requests", "1",
                    "--max-tenant-queued-requests", "1",
                    "--scheduler-batch-window-ms", "500",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            def receive(connection: socket.socket) -> tuple[int, int]:
                header = protocol.receive_exact(connection, protocol.HEADER.size)
                body_bytes = protocol.HEADER.unpack(header)[4]
                response = header + protocol.receive_exact(connection, body_bytes)
                request_id, status, _ = decode_response(response)
                return request_id, status

            first = None
            try:
                for _ in range(1000):
                    if socket_path.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                first.settimeout(10)
                first.connect(os.fspath(socket_path))
                first.sendall(frames[0])
                time.sleep(0.05)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as second:
                    second.settimeout(10)
                    second.connect(os.fspath(socket_path))
                    second.sendall(frames[1])
                    self.assertEqual(receive(second), (2_002, 2))
                self.assertEqual(receive(first), (2_001, 0))
                self.assertIsNone(process.poll())
            finally:
                if first is not None:
                    first.close()
                if process.poll() is None:
                    process.terminate()
                    output, _ = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()
