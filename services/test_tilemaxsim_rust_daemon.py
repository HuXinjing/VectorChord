# Copyright (c) 2026 HuXinjing

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import (
    decode_response,
    external_request_frame,
)
from services.tilemaxsim_shard import ImmutableShardWriter

SOAK_SECONDS = float(os.environ.get("VECTORCHORD_TILEMAXSIM_SOAK_SECONDS", "0"))


class RustDaemonTest(unittest.TestCase):
    binary = (
        Path(__file__).parent
        / "tilemaxsimd"
        / "target"
        / "release"
        / "vchord-tilemaxsimd"
    )
    control_binary = binary.with_name("vchord-tilemaxsimctl")

    def require_binary(self) -> Path:
        if not self.binary.exists():
            self.skipTest("release vchord-tilemaxsimd binary has not been built")
        return self.binary

    @staticmethod
    def receive_response(connection: socket.socket) -> bytes:
        header = protocol.receive_exact(connection, protocol.HEADER.size)
        body_bytes = protocol.HEADER.unpack(header)[4]
        return header + protocol.receive_exact(connection, body_bytes)

    def wait_until_ready(
        self, process: subprocess.Popen[str], socket_path: Path
    ) -> None:
        for _ in range(1000):
            if socket_path.exists() or process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertIsNone(process.poll())
        self.assertTrue(socket_path.exists())

    @staticmethod
    def write_fixture(root: Path) -> tuple[Path, bytes]:
        shard_root = root / "cache"
        shard_root.mkdir()
        documents = [
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2"),
            np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype="<f2"),
        ]
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
        frame, _ = external_request_frame(
            901,
            protocol.DTYPE_F16,
            query.tolist(),
            "model@1",
            [
                (11 + index, reference, document.tolist())
                for index, (reference, document) in enumerate(
                    zip(references, documents, strict=True)
                )
            ],
        )
        return shard_root, frame

    def start_runtime_daemon(
        self,
        root: Path,
        device: int,
        *extra: str,
    ) -> tuple[subprocess.Popen[str], Path, bytes]:
        binary = self.require_binary()
        shard_root, frame = self.write_fixture(root)
        socket_path = root / "tilemaxsimd.sock"
        command = [
            os.fspath(binary),
            "--socket",
            os.fspath(socket_path),
            "--gpu-memory-gb",
            f"{device}=0.05",
            "--gpu-workspace-gb",
            "0.02",
            "--host-cache-gb",
            "0.01",
            "--contract-root",
            f"model@1={shard_root}",
            *extra,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.wait_until_ready(process, socket_path)
        return process, socket_path, frame

    def request(self, socket_path: Path, frame: bytes) -> tuple[int, int, list]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(os.fspath(socket_path))
            connection.sendall(frame)
            return decode_response(self.receive_response(connection))

    @staticmethod
    def free_device() -> int:
        return max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )

    def run_daemon(
        self,
        devices: list[int],
        documents: list[np.ndarray] | None = None,
        query: np.ndarray | None = None,
        gpu_memory_gb: str = "0.05",
        workspace_gb: str = "0.02",
        resident: bool = False,
    ) -> tuple[str, list[tuple[int, float]]]:
        binary = self.require_binary()
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
            frame, _ = external_request_frame(
                901,
                protocol.DTYPE_F16,
                query.tolist(),
                "model@1",
                [
                    (11 + index, reference, document.tolist())
                    for index, (reference, document) in enumerate(
                        zip(references, documents, strict=True)
                    )
                ],
            )
            socket_path = root / "tilemaxsimd.sock"
            command = [
                os.fspath(binary),
                "--socket",
                os.fspath(socket_path),
            ]
            for device in devices:
                command.extend(("--gpu-memory-gb", f"{device}={gpu_memory_gb}"))
            command.extend(
                (
                    "--gpu-workspace-gb",
                    workspace_gb,
                    "--host-cache-gb",
                    "0.01",
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
                self.wait_until_ready(process, socket_path)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(os.fspath(socket_path))
                    connection.sendall(frame)
                    response = self.receive_response(connection)
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_external_v2_shard_round_trip_matches_protocol_oracle(self) -> None:
        self.run_daemon([self.free_device()])

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
        device = self.free_device()
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
        device = self.free_device()
        output, _ = self.run_daemon([device], resident=True)
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        prewarm = next(
            event
            for event in events
            if event.get("event") == "tilemaxsim_rust_prewarm_complete"
        )
        self.assertEqual(prewarm["entries"], 2)
        self.assertEqual(prewarm["cache"]["devices"][0]["gpu_pinned_entries"], 2)

    def test_cli_help_version_and_invalid_values(self) -> None:
        binary = self.require_binary()
        help_result = subprocess.run(
            [os.fspath(binary), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for expected in (
            "vchord-tilemaxsimd",
            "--gpu-memory-gb <GPU=GB>",
            "--request-timeout-ms <MILLISECONDS>",
            "--allow-peer-uid <UID>",
            "EXAMPLES:",
        ):
            self.assertIn(expected, help_result.stdout)
        version_result = subprocess.run(
            [os.fspath(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(version_result.returncode, 0, version_result.stderr)
        self.assertRegex(version_result.stdout, r"^vchord-tilemaxsimd \d+\.\d+\.\d+\n$")
        invalid = subprocess.run(
            [
                os.fspath(binary),
                "--socket",
                "/tmp/vchord-invalid.sock",
                "--gpu-memory-gb",
                "0=1",
                "--contract-root",
                "model@1=/tmp",
                "--gpu-block-kib",
                "3",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("power of two", invalid.stderr)
        self.assertTrue(self.control_binary.exists())
        control_help = subprocess.run(
            [os.fspath(self.control_binary), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(control_help.returncode, 0, control_help.stderr)
        self.assertIn("status", control_help.stdout)
        self.assertIn("STATUS EXIT CODES:", control_help.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_half_packet_does_not_block_another_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, socket_path, frame = self.start_runtime_daemon(
                Path(directory),
                self.free_device(),
                "--max-inflight",
                "2",
                "--request-timeout-ms",
                "1500",
            )
            slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                slow.connect(os.fspath(socket_path))
                slow.sendall(frame[:4])
                time.sleep(0.1)
                started = time.monotonic()
                _, status, _ = self.request(socket_path, frame)
                elapsed = time.monotonic() - started
                self.assertEqual(status, 0)
                self.assertLess(elapsed, 1.0)
            finally:
                slow.close()
                process.terminate()
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_concurrent_clients_receive_correct_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, socket_path, frame = self.start_runtime_daemon(
                Path(directory),
                self.free_device(),
                "--max-inflight",
                "8",
                "--request-timeout-ms",
                "3000",
            )
            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    responses = list(
                        executor.map(lambda _: self.request(socket_path, frame), range(8))
                    )
                self.assertEqual([response[1] for response in responses], [0] * 8)
                self.assertTrue(
                    all(
                        [candidate for candidate, _ in response[2]] == [11, 12]
                        for response in responses
                    )
                )
            finally:
                process.terminate()
                output, _ = process.communicate(timeout=5)
            events = [
                json.loads(line) for line in output.splitlines() if line.startswith("{")
            ]
            completed = [
                event
                for event in events
                if event.get("event") == "tilemaxsim_rust_request"
                and event.get("status") == "ok"
            ]
            self.assertEqual(len(completed), 8, output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_full_connection_queue_returns_resource_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, socket_path, frame = self.start_runtime_daemon(
                Path(directory),
                self.free_device(),
                "--max-inflight",
                "1",
                "--backlog",
                "1",
                "--request-timeout-ms",
                "1500",
            )
            first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            third = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                first.connect(os.fspath(socket_path))
                first.sendall(frame[:4])
                time.sleep(0.1)
                second.connect(os.fspath(socket_path))
                second.sendall(frame[:4])
                time.sleep(0.1)
                third.settimeout(2)
                third.connect(os.fspath(socket_path))
                _, status, _ = decode_response(self.receive_response(third))
                self.assertEqual(status, 2)
            finally:
                first.close()
                second.close()
                third.close()
                process.terminate()
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)
            self.assertIn('"status":"connection_backlog_full"', output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tensor_batch_limit_rejects_before_gpu_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, socket_path, frame = self.start_runtime_daemon(
                Path(directory),
                self.free_device(),
                "--max-batch-tokens",
                "1",
            )
            try:
                _, status, _ = self.request(socket_path, frame)
                self.assertEqual(status, 2)
            finally:
                process.terminate()
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)
            self.assertIn('"status":"batch_limit"', output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_sigterm_removes_socket_ready_and_pid_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready_path = root / "ready.json"
            pid_path = root / "tilemaxsimd.pid"
            process, socket_path, frame = self.start_runtime_daemon(
                root,
                self.free_device(),
                "--request-timeout-ms",
                "300",
                "--shutdown-grace-ms",
                "1000",
                "--ready-file",
                os.fspath(ready_path),
                "--pid-file",
                os.fspath(pid_path),
            )
            partial: socket.socket | None = None
            output = ""
            try:
                for _ in range(200):
                    if ready_path.exists() and pid_path.exists():
                        break
                    self.assertIsNone(process.poll())
                    time.sleep(0.01)
                self.assertTrue(ready_path.exists())
                self.assertEqual(int(pid_path.read_text().strip()), process.pid)
                ready = json.loads(ready_path.read_text())
                self.assertEqual(ready["pid"], process.pid)
                self.assertEqual(ready["schema_version"], 1)
                status = subprocess.run(
                    [
                        os.fspath(self.control_binary),
                        "status",
                        "--socket",
                        os.fspath(socket_path),
                        "--ready-file",
                        os.fspath(ready_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(status.returncode, 0, status.stderr)
                self.assertIn("accepting TileMaxSim connections", status.stdout)
                partial = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                partial.connect(os.fspath(socket_path))
                partial.sendall(frame[:4])
                started = time.monotonic()
                process.terminate()
                output, _ = process.communicate(timeout=3)
                elapsed = time.monotonic() - started
            finally:
                if partial is not None:
                    partial.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            self.assertEqual(process.returncode, 0, output)
            self.assertLess(elapsed, 2)
            self.assertFalse(socket_path.exists())
            self.assertFalse(ready_path.exists())
            self.assertFalse(pid_path.exists())
            self.assertIn('"event":"tilemaxsim_rust_stopped"', output)
            self.assertIn('"drained":true', output)
            stopped_status = subprocess.run(
                [
                    os.fspath(self.control_binary),
                    "status",
                    "--socket",
                    os.fspath(socket_path),
                    "--quiet",
                ],
                check=False,
            )
            self.assertEqual(stopped_status.returncode, 1)

    @unittest.skipUnless(
        torch.cuda.is_available() and SOAK_SECONDS > 0,
        "set VECTORCHORD_TILEMAXSIM_SOAK_SECONDS on a GPU release runner",
    )
    def test_configured_concurrency_and_disconnect_soak(self) -> None:
        def snapshot(pid: int) -> tuple[int, int, int]:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            gpu_mib = next(
                int(fields[1])
                for fields in (
                    [value.strip() for value in line.split(",")]
                    for line in output.splitlines()
                )
                if int(fields[0]) == pid
            )
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
            rss_kib = int(
                next(line for line in status if line.startswith("VmRSS:")).split()[1]
            )
            file_descriptors = len(list(Path(f"/proc/{pid}/fd").iterdir()))
            return gpu_mib, rss_kib, file_descriptors

        with tempfile.TemporaryDirectory() as directory:
            process, socket_path, frame = self.start_runtime_daemon(
                Path(directory),
                self.free_device(),
                "--max-inflight",
                "32",
                "--backlog",
                "256",
                "--max-queued-requests",
                "256",
                "--request-timeout-ms",
                "5000",
                "--shutdown-grace-ms",
                "5000",
            )
            recent_output: deque[str] = deque(maxlen=100)

            def drain_output() -> None:
                assert process.stdout is not None
                recent_output.extend(process.stdout)

            reader = threading.Thread(target=drain_output, daemon=True)
            reader.start()
            errors: list[str] = []
            try:
                for _ in range(100):
                    self.assertEqual(self.request(socket_path, frame)[1], 0)
                before = snapshot(process.pid)
                deadline = time.monotonic() + SOAK_SECONDS

                def valid_worker() -> int:
                    completed = 0
                    while time.monotonic() < deadline:
                        try:
                            status = self.request(socket_path, frame)[1]
                            if status == 0:
                                completed += 1
                            else:
                                errors.append(f"valid request returned status {status}")
                        except Exception as error:  # captured for the main test thread
                            errors.append(f"valid request failed: {error!r}")
                    return completed

                def disconnect_worker() -> int:
                    completed = 0
                    while time.monotonic() < deadline:
                        try:
                            with socket.socket(
                                socket.AF_UNIX, socket.SOCK_STREAM
                            ) as connection:
                                connection.connect(os.fspath(socket_path))
                                connection.sendall(frame[:4])
                            completed += 1
                            time.sleep(0.005)
                        except Exception as error:  # captured for the main test thread
                            errors.append(f"disconnect churn failed: {error!r}")
                    return completed

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(valid_worker) for _ in range(8)]
                    futures += [executor.submit(disconnect_worker) for _ in range(2)]
                    counts = [future.result() for future in futures]
                self.assertFalse(errors, errors[:10])
                self.assertGreater(sum(counts[:8]), 0)
                self.assertGreater(sum(counts[8:]), 0)
                status = subprocess.run(
                    [
                        os.fspath(self.control_binary),
                        "status",
                        "--socket",
                        os.fspath(socket_path),
                        "--quiet",
                    ],
                    check=False,
                )
                self.assertEqual(status.returncode, 0)
                after = snapshot(process.pid)
                self.assertLessEqual(after[0] - before[0], 2)
                self.assertLessEqual(after[1] - before[1], 64 * 1024)
                self.assertLessEqual(after[2] - before[2], 2)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                reader.join(timeout=2)
                if process.stdout is not None:
                    process.stdout.close()
            self.assertEqual(process.returncode, 0, "".join(recent_output))


if __name__ == "__main__":
    unittest.main()
