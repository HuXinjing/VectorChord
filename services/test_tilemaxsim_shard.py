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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from devtools import tilemaxsim_reference_sidecar as protocol
from services.tilemaxsim_cuda_sidecar import ContentAddressedResolver
from services.tilemaxsim_shard import ImmutableShardWriter, load_index


class ImmutableShardTest(unittest.TestCase):
    def test_builder_defaults_to_shards_and_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            manifest = source / "pages.jsonl"
            records = []
            for index in range(3):
                tensor = np.full((2, 4), index + 1, dtype="<f2")
                path = source / f"page-{index}.npy"
                np.save(path, tensor, allow_pickle=False)
                records.append(
                    {
                        "page_key": f"page-{index}",
                        "embedding_file": path.name,
                        "n_tokens": 2,
                        "dim": 4,
                    }
                )
            with manifest.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            cache = root / "cache"
            descriptors = root / "descriptors.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "services.build_tilemaxsim_tensor_cache",
                    "--manifest",
                    os.fspath(manifest),
                    "--cache-root",
                    os.fspath(cache),
                    "--descriptor-manifest",
                    os.fspath(descriptors),
                    "--workers",
                    "2",
                    "--shard-size-gb",
                    "0.00001",
                    "--no-fsync",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["storage_format"], "shards")
            self.assertTrue((cache / "tilemaxsim-shards-v1.json").is_file())
            self.assertEqual(len(load_index(cache / "tilemaxsim-shards-v1.json").entries), 3)
            self.assertEqual(len(descriptors.read_text().splitlines()), 3)
            self.assertFalse(list(cache.glob("??/*.bin")))

    def test_round_trip_batches_adjacent_tensor_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [
                np.arange(32, dtype="<f2").reshape(8, 4),
                np.arange(32, 64, dtype="<f2").reshape(8, 4),
                np.arange(64, 96, dtype="<f2").reshape(8, 4),
            ]
            writer = ImmutableShardWriter(root, target_bytes=8192, alignment=256, fsync=False)
            descriptors = []
            try:
                for tensor in tensors:
                    payload = tensor.tobytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    writer.add(digest, payload, 8, 4, "float16")
                    descriptors.append(
                        protocol.ExternalTensorRequest(
                            "model@1",
                            f"sha256://{digest}",
                            8,
                            4,
                            protocol.DTYPE_F16,
                            f"sha256:{digest}",
                        )
                    )
                index_path = writer.finish()
            finally:
                writer.close()

            index = load_index(index_path)
            self.assertEqual(len(index.shards), 1)
            self.assertEqual(len(index.entries), 3)
            resolver = ContentAddressedResolver(
                {"model@1": root}, 1024, verify_full_shards=True
            )
            try:
                first = resolver.resolve_many(descriptors)
                second = resolver.resolve_many(descriptors)
                self.assertEqual([item.payload for item in first], [t.tobytes() for t in tensors])
                self.assertFalse(any(item.cache_hit for item in first))
                self.assertTrue(all(item.cache_hit for item in second))
                status = resolver.status()
                self.assertEqual(status["verified_shards"], 1)
                self.assertEqual(status["batch_read_calls"], 1)
            finally:
                resolver.close()

    def test_whole_shard_checksum_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = np.eye(4, dtype="<f2")
            payload = tensor.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer = ImmutableShardWriter(root, target_bytes=4096, alignment=256, fsync=False)
            try:
                writer.add(digest, payload, 4, 4, "float16")
                index = load_index(writer.finish())
            finally:
                writer.close()
            shard_path = root / next(iter(index.shards))
            with shard_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"\xff")
            resolver = ContentAddressedResolver(
                {"model@1": root}, 0, verify_full_shards=True
            )
            try:
                request = protocol.ExternalTensorRequest(
                    "model@1",
                    f"sha256://{digest}",
                    4,
                    4,
                    protocol.DTYPE_F16,
                    f"sha256:{digest}",
                )
                with self.assertRaisesRegex(protocol.SidecarError, "shard checksum"):
                    resolver.resolve(request)
            finally:
                resolver.close()

    def test_deduplicates_equal_payloads_across_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = np.ones((2, 4), dtype="<f2").tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            writer = ImmutableShardWriter(root, target_bytes=4096, alignment=256, fsync=False)
            try:
                writer.add(digest, payload, 2, 4, "float16")
                writer.add(digest, payload, 2, 4, "float16")
                index = load_index(writer.finish())
            finally:
                writer.close()
            self.assertEqual(len(index.entries), 1)
            self.assertEqual(sum(shard.size for shard in index.shards.values()), 256)


if __name__ == "__main__":
    unittest.main()
