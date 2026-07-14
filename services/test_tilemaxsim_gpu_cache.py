# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# We welcome any commercial collaboration or support. For inquiries
# regarding the licenses, please contact us at:
# vectorchord-inquiry@tensorchord.ai
#
# Copyright (c) 2025-2026 TensorChord Inc.

from __future__ import annotations

import time
import unittest

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from services.tilemaxsim_cuda_sidecar import ResidentTorchTileMaxsimEngine
from services.tilemaxsim_gpu_cache import (
    FreeExtentAllocator,
    GpuArenaSpec,
    GpuResourcePool,
    GpuTensorCache,
    parse_gpu_memory_gb,
    parse_memory_gb,
)


def available_device() -> str:
    index = max(
        range(torch.cuda.device_count()),
        key=lambda candidate: torch.cuda.mem_get_info(candidate)[0],
    )
    return f"cuda:{index}"


class GpuCacheUnitTest(unittest.TestCase):
    def test_public_memory_configuration_uses_gb(self) -> None:
        self.assertEqual(parse_memory_gb("20"), 20 * 1024**3)
        self.assertEqual(parse_memory_gb("0.5"), 512 * 1024**2)
        self.assertEqual(
            parse_gpu_memory_gb("2=12"),
            GpuArenaSpec("cuda:2", 12 * 1024**3),
        )
        self.assertEqual(
            parse_gpu_memory_gb("cuda:2=12.5"),
            GpuArenaSpec("cuda:2", int(12.5 * 1024**3)),
        )
        with self.assertRaisesRegex(ValueError, "GPU=GB"):
            parse_gpu_memory_gb("cuda:0")
        with self.assertRaisesRegex(ValueError, "byte suffixes"):
            parse_gpu_memory_gb("0=20GiB")

    def test_extent_allocator_coalesces_released_ranges(self) -> None:
        allocator = FreeExtentAllocator(4096, alignment=256)
        first = allocator.allocate(300)
        second = allocator.allocate(700)
        self.assertEqual(first, (0, 512))
        self.assertEqual(second, (512, 768))
        assert first is not None and second is not None
        allocator.release(*first)
        allocator.release(*second)
        self.assertEqual(allocator.extents, [(0, 4096)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_pool_rejects_budget_larger_than_currently_free_memory(self) -> None:
        device = available_device()
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
        with self.assertRaisesRegex(RuntimeError, "cannot acquire"):
            GpuResourcePool(
                [GpuArenaSpec(device, free_bytes + 1024 * 1024)],
                1024 * 1024,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_cache_evicts_only_released_entries(self) -> None:
        device = available_device()
        pool = GpuResourcePool(
            [GpuArenaSpec(device, 16 * 1024 * 1024)], 8 * 1024 * 1024
        )
        try:
            cache = GpuTensorCache(pool, allow_eviction=True)
            rows, dimension = 8192, 320
            payload = np.ones((rows, dimension), dtype="<f2").tobytes()
            first, first_hit = cache.acquire(
                ("model", "first"),
                rows,
                dimension,
                protocol.DTYPE_F16,
                lambda: payload,
            )
            self.assertFalse(first_hit)
            self.assertEqual(float(first.tensor()[0, 0].cpu()), 1.0)
            with self.assertRaisesRegex(protocol.SidecarError, "insufficient"):
                cache.acquire(
                    ("model", "second"),
                    rows,
                    dimension,
                    protocol.DTYPE_F16,
                    lambda: payload,
                )
            cache.release(first)
            second, second_hit = cache.acquire(
                ("model", "second"),
                rows,
                dimension,
                protocol.DTYPE_F16,
                lambda: payload,
            )
            self.assertFalse(second_hit)
            cache.release(second)
            self.assertEqual(cache.status()["evictions"], 1)
        finally:
            pool.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_resident_engine_scores_gpu_cache_handles(self) -> None:
        device = available_device()
        pool = GpuResourcePool(
            [GpuArenaSpec(device, 32 * 1024 * 1024)], 16 * 1024 * 1024
        )
        try:
            cache = GpuTensorCache(pool, allow_eviction=False)
            query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            document = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            handle, hit = cache.acquire(
                ("model", "document"),
                2,
                2,
                protocol.DTYPE_F16,
                document.tobytes,
                pin=True,
            )
            self.assertFalse(hit)
            engine = ResidentTorchTileMaxsimEngine(pool, 16 * 1024 * 1024, False, 1)
            results, _, _ = engine.score(
                query.tobytes(),
                2,
                2,
                protocol.DTYPE_F16,
                [(7, handle)],
                time.monotonic() + 5,
                lambda: False,
            )
            cache.release(handle)
            self.assertEqual(results, [(7, 2.0)])
            second, second_hit = cache.acquire(
                ("model", "document"),
                2,
                2,
                protocol.DTYPE_F16,
                lambda: self.fail("GPU hit must not call the payload loader"),
            )
            cache.release(second)
            self.assertTrue(second_hit)
        finally:
            pool.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_ragged_resident_kernel_matches_torch_for_320_dimensions(self) -> None:
        device = available_device()
        pool = GpuResourcePool(
            [GpuArenaSpec(device, 64 * 1024 * 1024)], 32 * 1024 * 1024
        )
        try:
            generator = np.random.default_rng(7)
            query = generator.standard_normal((44, 320)).astype("<f2")
            documents = [
                generator.standard_normal((17, 320)).astype("<f2"),
                generator.standard_normal((35, 320)).astype("<f2"),
            ]
            cache = GpuTensorCache(pool, allow_eviction=False)
            handles = []
            for index, document in enumerate(documents):
                handle, _ = cache.acquire(
                    ("model", index),
                    document.shape[0],
                    document.shape[1],
                    protocol.DTYPE_F16,
                    document.tobytes,
                )
                handles.append(handle)
            engine = ResidentTorchTileMaxsimEngine(pool, 32 * 1024 * 1024, False, 1)
            results, _, _ = engine.score(
                query.tobytes(),
                query.shape[0],
                query.shape[1],
                protocol.DTYPE_F16,
                [(index, handle) for index, handle in enumerate(handles)],
                time.monotonic() + 5,
                lambda: False,
            )
            query_device = torch.from_numpy(query).to(device)
            expected = []
            for index, document in enumerate(documents):
                document_device = torch.from_numpy(document).to(device)
                score = (
                    (query_device @ document_device.transpose(0, 1))
                    .amax(dim=1)
                    .sum(dtype=torch.float32)
                    .item()
                )
                expected.append((index, score))
            for handle in handles:
                cache.release(handle)
            for (_, actual), (_, reference) in zip(results, expected, strict=True):
                self.assertAlmostEqual(actual, reference, delta=0.1)
        finally:
            pool.close()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "two CUDA devices are unavailable",
    )
    def test_resident_engine_scores_shards_on_multiple_gpus(self) -> None:
        pool = GpuResourcePool(
            [
                GpuArenaSpec("cuda:0", 32 * 1024 * 1024),
                GpuArenaSpec("cuda:1", 32 * 1024 * 1024),
            ],
            16 * 1024 * 1024,
        )
        try:
            cache = GpuTensorCache(pool, allow_eviction=False)
            identity = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            half = np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype="<f2")
            first, _ = cache.acquire(
                ("model", "first"),
                2,
                2,
                protocol.DTYPE_F16,
                identity.tobytes,
            )
            second, _ = cache.acquire(
                ("model", "second"),
                2,
                2,
                protocol.DTYPE_F16,
                half.tobytes,
            )
            self.assertNotEqual(first.device, second.device)
            engine = ResidentTorchTileMaxsimEngine(pool, 16 * 1024 * 1024, False, 1)
            results, _, _ = engine.score(
                identity.tobytes(),
                2,
                2,
                protocol.DTYPE_F16,
                [(1, first), (2, second)],
                time.monotonic() + 5,
                lambda: False,
            )
            cache.release(first)
            cache.release(second)
            self.assertEqual(sorted(results), [(1, 2.0), (2, 1.0)])
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
