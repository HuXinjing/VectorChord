# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2025-2026 TensorChord Inc.

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import (
    decode_response,
    external_request_frame,
)
from services.tilemaxsim_shard import ImmutableShardWriter


class RustDaemonTest(unittest.TestCase):
    def run_daemon(
        self,
        devices: list[int],
        documents: list[np.ndarray] | None = None,
        query: np.ndarray | None = None,
        gpu_memory_gb: str = "0.05",
        workspace_gb: str = "0.02",
        resident: bool = False,
    ) -> tuple[str, list[tuple[int, float]]]:
        binary = (
            Path(__file__).parent
            / "tilemaxsimd"
            / "target"
            / "release"
            / "tilemaxsimd"
        )
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
                for _ in range(1000):
                    if socket_path.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNone(process.poll())
                self.assertTrue(socket_path.exists())
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(os.fspath(socket_path))
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_external_v2_shard_round_trip_matches_protocol_oracle(self) -> None:
        device = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.mem_get_info(index)[0],
        )
        self.run_daemon([device])

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
            gpu_memory_gb="0.0012",
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


if __name__ == "__main__":
    unittest.main()
