# This software is licensed under the repository's dual license model.
"""Opt-in acceptance for an isolated Vulkan container, without NumPy/PyTorch.

Start services/compose.tilemaxsimd-vulkan-wsl.yml with contract local-test@1:
TILEMAXSIM_VULKAN_CONTAINER=vectorchord-amd-tilemaxsimd-1 \
    python3 -m unittest services.test_tilemaxsim_vulkan_container -v
"""
import json
import os
import socket
import struct
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import (
    decode_response,
    external_request_frame,
)


@unittest.skipUnless(os.getenv("TILEMAXSIM_VULKAN_CONTAINER"), "requires assigned Vulkan container")
class VulkanContainerTest(unittest.TestCase):
    def test_published_tensors_and_concurrent_scoring(self):
        container = os.environ["TILEMAXSIM_VULKAN_CONTAINER"]
        port = int(os.getenv("TILEMAXSIM_VULKAN_PORT", "39192"))
        logs = subprocess.run(
            ["docker", "logs", container], capture_output=True, text=True, check=True
        ).stdout
        events = [json.loads(line) for line in logs.splitlines() if line.startswith("{")]
        ready = [event for event in events if event.get("event") == "tilemaxsim_rust_ready"][-1]
        backend = ready["cache"]["devices"][0]["backend"]
        self.assertEqual(backend["backend"], "vulkan")
        self.assertNotIn("llvmpipe", backend["name"].lower())
        print("Acceptance GPU:", backend["name"])

        dimension = 320
        query = [[((q * 7 + d * 3) % 17 - 8) / 16 for d in range(dimension)] for q in range(3)]
        documents = [
            [[((r * 11 + d * 5 + c) % 23 - 11) / 16 for d in range(dimension)] for r in range(3 + c)]
            for c in range(2)
        ]
        expected = [
            (c + 1, sum(max(sum(a * b for a, b in zip(q, row)) for row in doc) for q in query))
            for c, doc in enumerate(documents)
        ]
        for dtype, code, label in [(1, "f", "float32"), (2, "e", "float16")]:
            with self.subTest(dtype=label):
                candidates = []
                for c, doc in enumerate(documents):
                    flat = [value for row in doc for value in row]
                    publication = subprocess.run(
                        ["docker", "exec", "-i", container, "tilemaxsimctl", "publish-object",
                         "--root", "/var/lib/vectorchord", "--rows", str(len(doc)),
                         "--dimension", str(dimension), "--dtype", label],
                        input=struct.pack(f"<{len(flat)}{code}", *flat), capture_output=True, check=True,
                    )
                    descriptor = json.loads(publication.stdout)
                    candidates.append((c + 1, descriptor["tensor_ref"], doc))

                def score(request_id):
                    frame, _ = external_request_frame(request_id, dtype, query, "local-test@1", candidates)
                    with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
                        connection.sendall(frame)
                        header = protocol.receive_exact(connection, protocol.HEADER.size)
                        body = protocol.receive_exact(connection, protocol.HEADER.unpack(header)[4])
                    actual_id, status, result = decode_response(header + body)
                    self.assertEqual((actual_id, status), (request_id, 0), result)
                    self.assertEqual(len(result), len(expected))
                    for (actual_id, value), (expected_id, expected_value) in zip(result, expected):
                        self.assertEqual(actual_id, expected_id)
                        self.assertAlmostEqual(value, expected_value, places=4)

                # Cold upload, followed by cache reuse under concurrent callers.
                score(1000)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(score, range(1001, 1017)))


if __name__ == "__main__":
    unittest.main()
