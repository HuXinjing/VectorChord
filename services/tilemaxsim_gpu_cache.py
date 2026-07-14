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

"""Process-owned GPU arenas and a bounded tensor cache for TileMaxSim."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import blake2b
from math import ceil
from typing import Callable

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol


_GPU_MEMORY_GB = re.compile(r"^(?:cuda:)?([0-9]+)=([0-9]+(?:\.[0-9]+)?)$")
_MEMORY_GB = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
GIB = 1024**3
DEFAULT_BLOCK_BYTES = 256 * 1024
DEFAULT_STAGING_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GpuArenaSpec:
    device: str
    total_bytes: int


def parse_gpu_memory_gb(value: str) -> GpuArenaSpec:
    """Parse the public ``GPU=GB`` configuration into an internal byte budget."""

    match = _GPU_MEMORY_GB.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "GPU memory must be GPU=GB, for example 1=20; byte suffixes are not accepted"
        )
    raw_index, raw_gb = match.groups()
    try:
        total_bytes = int(Decimal(raw_gb) * GIB)
    except InvalidOperation as error:
        raise ValueError("GPU memory GB value is invalid") from error
    if total_bytes <= 0:
        raise ValueError("GPU memory GB value must be positive")
    return GpuArenaSpec(f"cuda:{int(raw_index)}", total_bytes)


def parse_memory_gb(value: str) -> int:
    if _MEMORY_GB.fullmatch(value.strip()) is None:
        raise ValueError("memory size must be a positive number of GB")
    try:
        total_bytes = int(Decimal(value.strip()) * GIB)
    except InvalidOperation as error:
        raise ValueError("memory size must be a positive number of GB") from error
    if total_bytes <= 0:
        raise ValueError("memory size must be a positive number of GB")
    return total_bytes


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class FreeExtentAllocator:
    """Best-fit allocator for contiguous slices of a preallocated byte arena."""

    def __init__(self, capacity: int, alignment: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("arena capacity must be positive")
        self.capacity = capacity
        self.alignment = alignment
        self.extents: list[tuple[int, int]] = [(0, capacity)]

    @property
    def free_bytes(self) -> int:
        return sum(length for _, length in self.extents)

    @property
    def largest_free_extent(self) -> int:
        return max((length for _, length in self.extents), default=0)

    def allocation_bytes(self, payload_bytes: int) -> int:
        return _align_up(payload_bytes, self.alignment)

    def allocate(self, payload_bytes: int) -> tuple[int, int] | None:
        required = self.allocation_bytes(payload_bytes)
        choices = [
            (length, index, start)
            for index, (start, length) in enumerate(self.extents)
            if length >= required
        ]
        if not choices:
            return None
        length, index, start = min(choices)
        if length == required:
            self.extents.pop(index)
        else:
            self.extents[index] = (start + required, length - required)
        return start, required

    def release(self, start: int, length: int) -> None:
        if start < 0 or length <= 0 or start + length > self.capacity:
            raise ValueError("released extent is outside the arena")
        self.extents.append((start, length))
        self.extents.sort()
        merged: list[tuple[int, int]] = []
        for extent_start, extent_length in self.extents:
            if merged and merged[-1][0] + merged[-1][1] == extent_start:
                previous_start, previous_length = merged[-1]
                merged[-1] = (previous_start, previous_length + extent_length)
            else:
                merged.append((extent_start, extent_length))
        self.extents = merged


class FixedBlockAllocator:
    """Fixed-base-block buddy/slab allocator used by resident tensors.

    A tensor gets one contiguous power-of-two slab.  This preserves the dense
    memory layout required by the Tensor Core kernel while bounding internal
    fragmentation and guaranteeing that released buddies coalesce.
    """

    def __init__(self, capacity: int, block_bytes: int = DEFAULT_BLOCK_BYTES) -> None:
        if capacity <= 0 or block_bytes <= 0:
            raise ValueError("arena capacity and block size must be positive")
        if block_bytes % 256:
            raise ValueError("GPU block size must be 256-byte aligned")
        self.block_bytes = block_bytes
        self.block_count = capacity // block_bytes
        if self.block_count == 0:
            raise ValueError("arena must contain at least one GPU block")
        self.capacity = self.block_count * block_bytes
        self._free: dict[int, set[tuple[int, int, int]]] = {}
        self._allocated: dict[int, tuple[int, int, int]] = {}
        start = 0
        remaining = self.block_count
        while remaining:
            order = remaining.bit_length() - 1
            size = 1 << order
            self._free.setdefault(order, set()).add((start, start, order))
            start += size
            remaining -= size

    @property
    def free_blocks(self) -> int:
        return sum(len(items) * (1 << order) for order, items in self._free.items())

    @property
    def free_bytes(self) -> int:
        return self.free_blocks * self.block_bytes

    @property
    def largest_free_extent(self) -> int:
        return max(
            (1 << order) * self.block_bytes
            for order, items in self._free.items()
            if items
        ) if self.free_blocks else 0

    def blocks_for(self, payload_bytes: int) -> int:
        if payload_bytes <= 0:
            raise ValueError("payload size must be positive")
        raw = ceil(payload_bytes / self.block_bytes)
        return 1 << (raw - 1).bit_length()

    def allocation_bytes(self, payload_bytes: int) -> int:
        return self.blocks_for(payload_bytes) * self.block_bytes

    def allocate(self, payload_bytes: int) -> tuple[int, ...] | None:
        required = self.blocks_for(payload_bytes)
        order = required.bit_length() - 1
        available_order = next(
            (
                candidate
                for candidate in range(order, self.block_count.bit_length())
                if self._free.get(candidate)
            ),
            None,
        )
        if available_order is None:
            return None
        start, root_start, root_order = self._free[available_order].pop()
        while available_order > order:
            available_order -= 1
            buddy = start + (1 << available_order)
            self._free.setdefault(available_order, set()).add(
                (buddy, root_start, root_order)
            )
        blocks = tuple(range(start, start + required))
        self._allocated[start] = (order, root_start, root_order)
        return blocks

    def release(self, blocks: tuple[int, ...]) -> None:
        if (
            not blocks
            or len(set(blocks)) != len(blocks)
            or blocks != tuple(range(blocks[0], blocks[0] + len(blocks)))
        ):
            raise ValueError("released GPU blocks are invalid")
        allocation = self._allocated.pop(blocks[0], None)
        if allocation is None:
            raise ValueError("GPU block was released more than once")
        order, root_start, root_order = allocation
        if len(blocks) != 1 << order:
            raise ValueError("released GPU slab has the wrong size")
        start = blocks[0]
        while order < root_order:
            buddy = root_start + (((start - root_start) ^ (1 << order)))
            item = (buddy, root_start, root_order)
            free = self._free.setdefault(order, set())
            if item not in free:
                break
            free.remove(item)
            start = min(start, buddy)
            order += 1
        self._free.setdefault(order, set()).add((start, root_start, root_order))


class TinyLfuSketch:
    """A small aging count-min sketch for cache admission and GDSF frequency."""

    def __init__(self, width: int = 4096, depth: int = 4) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("TinyLFU dimensions must be positive")
        self.width = width
        self.depth = depth
        self.tables = [[0] * width for _ in range(depth)]
        self.samples = 0
        self.reset_at = width * 10

    def _indices(self, key: tuple[object, ...]) -> tuple[int, ...]:
        digest = blake2b(repr(key).encode("utf-8"), digest_size=16).digest()
        return tuple(
            int.from_bytes(digest[row * 4 : row * 4 + 4], "little") % self.width
            for row in range(self.depth)
        )

    def increment(self, key: tuple[object, ...]) -> int:
        indices = self._indices(key)
        for row, index in enumerate(indices):
            if self.tables[row][index] < 65535:
                self.tables[row][index] += 1
        self.samples += 1
        estimate = min(self.tables[row][index] for row, index in enumerate(indices))
        if self.samples >= self.reset_at:
            for table in self.tables:
                for index, value in enumerate(table):
                    table[index] = value // 2
            self.samples //= 2
        return max(1, estimate)

    def estimate(self, key: tuple[object, ...]) -> int:
        return min(
            self.tables[row][index]
            for row, index in enumerate(self._indices(key))
        )


class GpuArena:
    """A CUDA byte buffer acquired atomically during process startup."""

    def __init__(
        self,
        spec: GpuArenaSpec,
        workspace_bytes: int,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> None:
        self.device = torch.device(spec.device)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        if self.device.index is None or self.device.index >= torch.cuda.device_count():
            raise RuntimeError(f"configured CUDA device is unavailable: {spec.device}")
        if workspace_bytes <= 0 or workspace_bytes >= spec.total_bytes:
            raise RuntimeError(
                f"{spec.device} allocation must exceed its TileMaxSim workspace"
            )
        self.total_bytes = spec.total_bytes
        self.workspace_bytes = workspace_bytes
        raw_capacity = spec.total_bytes - workspace_bytes
        self.allocator = FixedBlockAllocator(raw_capacity, block_bytes)
        self.capacity = self.allocator.capacity
        self.reserved_workspace_bytes = spec.total_bytes - self.capacity
        if self.capacity <= 0:
            raise RuntimeError(f"{spec.device} has no aligned tensor-cache capacity")
        self.storage: torch.Tensor | None = None
        self.host_staging: torch.Tensor | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self.h2d_batches = 0
        self.h2d_copy_calls = 0
        self.h2d_bytes = 0

        with torch.cuda.device(self.device):
            free_bytes, device_bytes = torch.cuda.mem_get_info(self.device)
            self.device_bytes = device_bytes
            if free_bytes < spec.total_bytes:
                raise RuntimeError(
                    f"cannot acquire {spec.total_bytes} bytes on {spec.device}: "
                    f"only {free_bytes} bytes are free"
                )
            try:
                self.storage = torch.empty(
                    self.capacity, dtype=torch.uint8, device=self.device
                )
                # Reserve the remaining configured budget in PyTorch's CUDA
                # allocator. Releasing this temporary tensor leaves the block
                # in the process-owned caching allocator for TileMaxSim
                # workspaces instead of returning it to another process.
                workspace = torch.empty(
                    self.reserved_workspace_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                )
                torch.cuda.synchronize(self.device)
                del workspace
                staging_bytes = min(self.capacity, DEFAULT_STAGING_BYTES)
                staging_bytes = max(
                    self.allocator.block_bytes,
                    staging_bytes // self.allocator.block_bytes * self.allocator.block_bytes,
                )
                self.host_staging = torch.empty(
                    staging_bytes, dtype=torch.uint8, pin_memory=True
                )
                # Fault and pin every staging page during startup so the first
                # cache-miss batch does not pay a request-path NUMA/page cost.
                self.host_staging.zero_()
                self.copy_stream = torch.cuda.Stream(device=self.device)
            except Exception:
                self.storage = None
                self.host_staging = None
                self.copy_stream = None
                torch.cuda.empty_cache()
                raise

    def tensor(
        self,
        blocks: tuple[int, ...],
        payload_bytes: int,
        rows: int,
        dimension: int,
        dtype: int,
    ) -> torch.Tensor:
        if self.storage is None or self.host_staging is None or self.copy_stream is None:
            raise RuntimeError("GPU arena is closed")
        scalar_dtype = torch.float32 if dtype == protocol.DTYPE_F32 else torch.float16
        block_bytes = self.allocator.block_bytes
        if all(right == left + 1 for left, right in zip(blocks, blocks[1:])):
            raw = self.storage.narrow(0, blocks[0] * block_bytes, payload_bytes)
        else:
            raw = torch.cat(
                [
                    self.storage.narrow(0, block * block_bytes, block_bytes)
                    for block in blocks
                ]
            ).narrow(0, 0, payload_bytes)
        return raw.view(scalar_dtype).reshape(rows, dimension)

    def copy_many_from_host(
        self, items: list[tuple[tuple[int, ...], bytes]]
    ) -> None:
        if self.storage is None:
            raise RuntimeError("GPU arena is closed")
        if not items:
            return
        block_bytes = self.allocator.block_bytes
        flattened: list[tuple[int, bytes, int, int]] = []
        for blocks, payload in items:
            for ordinal, block in enumerate(blocks):
                start = ordinal * block_bytes
                length = min(block_bytes, max(0, len(payload) - start))
                flattened.append((block, payload, start, length))
        flattened.sort(key=lambda item: item[0])
        staging = self.host_staging
        staging_array = staging.numpy()
        staging_blocks = staging.numel() // block_bytes
        copy_calls = 0
        with torch.cuda.device(self.device):
            stream = self.copy_stream
            for chunk_start in range(0, len(flattened), staging_blocks):
                chunk = flattened[chunk_start : chunk_start + staging_blocks]
                for index, (_, payload, source_start, source_length) in enumerate(
                    chunk
                ):
                    if not source_length:
                        continue
                    start = index * block_bytes
                    staging_array[start : start + source_length] = np.frombuffer(
                        payload,
                        dtype=np.uint8,
                        count=source_length,
                        offset=source_start,
                    )
                with torch.cuda.stream(stream):
                    run_start = 0
                    for index in range(1, len(chunk) + 1):
                        if (
                            index < len(chunk)
                            and chunk[index][0] == chunk[index - 1][0] + 1
                        ):
                            continue
                        first_block = chunk[run_start][0]
                        count = index - run_start
                        length = count * block_bytes
                        self.storage.narrow(
                            0, first_block * block_bytes, length
                        ).copy_(
                            staging.narrow(0, run_start * block_bytes, length),
                            non_blocking=True,
                        )
                        copy_calls += 1
                        run_start = index
                stream.synchronize()
        self.h2d_batches += 1
        self.h2d_copy_calls += copy_calls
        self.h2d_bytes += sum(len(payload) for _, payload in items)

    def copy_from_host(self, blocks: tuple[int, ...], payload: bytes) -> None:
        self.copy_many_from_host([(blocks, payload)])

    def status(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "allocated_gb": round(self.total_bytes / GIB, 3),
            "tensor_capacity_gb": round(self.capacity / GIB, 3),
            "workspace_gb": round(self.workspace_bytes / GIB, 3),
            "allocated_bytes": self.total_bytes,
            "tensor_capacity_bytes": self.capacity,
            "workspace_bytes": self.workspace_bytes,
            "tensor_free_bytes": self.allocator.free_bytes,
            "largest_free_extent_bytes": self.allocator.largest_free_extent,
            "block_bytes": self.allocator.block_bytes,
            "block_count": self.allocator.block_count,
            "free_blocks": self.allocator.free_blocks,
            "h2d_batches": self.h2d_batches,
            "h2d_copy_calls": self.h2d_copy_calls,
            "h2d_bytes": self.h2d_bytes,
            "host_staging_bytes": self.host_staging.numel()
            if self.host_staging is not None
            else 0,
        }

    def close(self) -> None:
        if self.storage is None:
            return
        with torch.cuda.device(self.device):
            self.storage = None
            self.host_staging = None
            self.copy_stream = None
            torch.cuda.empty_cache()


class GpuResourcePool:
    """Own all configured CUDA allocations or fail without a partial pool."""

    def __init__(
        self,
        specs: list[GpuArenaSpec],
        workspace_bytes: int,
        block_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> None:
        if not specs:
            raise RuntimeError("at least one GPU allocation is required")
        devices = [spec.device for spec in specs]
        if len(devices) != len(set(devices)):
            raise RuntimeError("each CUDA device may be configured only once")
        self.arenas: list[GpuArena] = []
        try:
            for spec in specs:
                self.arenas.append(GpuArena(spec, workspace_bytes, block_bytes))
        except Exception:
            self.close()
            raise

    @property
    def primary_device(self) -> torch.device:
        return self.arenas[0].device

    def status(self) -> list[dict[str, object]]:
        return [arena.status() for arena in self.arenas]

    def close(self) -> None:
        for arena in self.arenas:
            arena.close()
        self.arenas.clear()


@dataclass
class _GpuCacheEntry:
    key: tuple[object, ...]
    arena: GpuArena
    blocks: tuple[int, ...]
    allocated_bytes: int
    payload_bytes: int
    rows: int
    dimension: int
    dtype: int
    references: int = 0
    pinned: bool = False
    priority: float = 0.0


@dataclass(frozen=True)
class GpuTensorHandle:
    entry: _GpuCacheEntry

    @property
    def arena(self) -> GpuArena:
        return self.entry.arena

    @property
    def device(self) -> torch.device:
        return self.entry.arena.device

    @property
    def rows(self) -> int:
        return self.entry.rows

    @property
    def dimension(self) -> int:
        return self.entry.dimension

    @property
    def dtype(self) -> int:
        return self.entry.dtype

    @property
    def payload_bytes(self) -> int:
        return self.entry.payload_bytes

    @property
    def offset_bytes(self) -> int:
        return self.entry.blocks[0] * self.entry.arena.allocator.block_bytes

    @property
    def block_ids(self) -> tuple[int, ...]:
        return self.entry.blocks

    @property
    def block_bytes(self) -> int:
        return self.entry.arena.allocator.block_bytes

    def tensor(self) -> torch.Tensor:
        return self.entry.arena.tensor(
            self.entry.blocks,
            self.entry.payload_bytes,
            self.entry.rows,
            self.entry.dimension,
            self.entry.dtype,
        )


@dataclass(frozen=True)
class GpuTensorLoad:
    key: tuple[object, ...]
    rows: int
    dimension: int
    dtype: int
    payload: bytes
    pin: bool = False


@dataclass(frozen=True)
class GpuAcquireBatch:
    handles: tuple[GpuTensorHandle | None, ...]
    bypassed: tuple[int, ...]
    deferred: tuple[int, ...]
    hits: int
    misses: int
    admitted: int


class GpuTensorCache:
    """Fixed-block GPU cache with TinyLFU admission and GDSF eviction."""

    def __init__(self, pool: GpuResourcePool, allow_eviction: bool) -> None:
        self.pool = pool
        self.allow_eviction = allow_eviction
        self.entries: OrderedDict[tuple[object, ...], _GpuCacheEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.loaded_bytes = 0
        self.admission_rejections = 0
        self.inflation = 0.0
        self.sketch = TinyLfuSketch()

    def _find_arena(self, payload_bytes: int) -> GpuArena | None:
        candidates = [
            arena
            for arena in self.pool.arenas
            if arena.allocator.largest_free_extent
            >= arena.allocator.allocation_bytes(payload_bytes)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda arena: arena.allocator.free_bytes)

    def _victim(self, arena: GpuArena | None = None) -> _GpuCacheEntry | None:
        candidates = [
            entry
            for entry in self.entries.values()
            if not entry.references
            and not entry.pinned
            and (arena is None or entry.arena is arena)
        ]
        return min(candidates, key=lambda entry: entry.priority, default=None)

    def _evict(self, entry: _GpuCacheEntry) -> None:
        current = self.entries.pop(entry.key, None)
        if current is not entry:
            raise RuntimeError("GPU eviction directory is inconsistent")
        entry.arena.allocator.release(entry.blocks)
        self.inflation = max(self.inflation, entry.priority)
        self.evictions += 1

    @staticmethod
    def _entry_cost(entry: _GpuCacheEntry) -> int:
        return len(entry.blocks)

    def _priority(self, key: tuple[object, ...], blocks: int) -> float:
        return self.inflation + self.sketch.estimate(key) / max(1, blocks)

    def _allocate(
        self,
        key: tuple[object, ...],
        payload_bytes: int,
        enforce_admission: bool,
    ) -> tuple[GpuArena, tuple[int, ...], int] | None:
        arena = self._find_arena(payload_bytes)
        capable = [
            candidate
            for candidate in self.pool.arenas
            if candidate.allocator.capacity
            >= candidate.allocator.allocation_bytes(payload_bytes)
        ]
        if not capable:
            raise protocol.SidecarError(
                protocol.STATUS_RESOURCE_LIMIT,
                "one tensor exceeds every configured GPU block arena",
            )
        if arena is None and self.allow_eviction:
            for candidate in sorted(
                capable, key=lambda item: item.allocator.free_bytes, reverse=True
            ):
                required_blocks = candidate.allocator.blocks_for(payload_bytes)
                required_bytes = candidate.allocator.allocation_bytes(payload_bytes)
                while candidate.allocator.largest_free_extent < required_bytes:
                    victim = self._victim(candidate)
                    if victim is None:
                        break
                    candidate_priority = self._priority(key, required_blocks)
                    if enforce_admission and candidate_priority <= victim.priority:
                        self.admission_rejections += 1
                        return None
                    self._evict(victim)
                if candidate.allocator.largest_free_extent >= required_bytes:
                    arena = candidate
                    break
        if arena is None:
            raise protocol.SidecarError(
                protocol.STATUS_RESOURCE_LIMIT,
                "configured GPU tensor arenas have insufficient free blocks",
            )
        blocks = arena.allocator.allocate(payload_bytes)
        assert blocks is not None
        return arena, blocks, len(blocks) * arena.allocator.block_bytes

    def acquire(
        self,
        key: tuple[object, ...],
        rows: int,
        dimension: int,
        dtype: int,
        loader: Callable[[], bytes],
        *,
        pin: bool = False,
    ) -> tuple[GpuTensorHandle, bool]:
        with self.lock:
            frequency = self.sketch.increment(key)
            cached = self.entries.get(key)
            if cached is not None:
                cached.references += 1
                cached.pinned = cached.pinned or pin
                cached.priority = self.inflation + frequency / self._entry_cost(cached)
                self.entries.move_to_end(key)
                self.hits += 1
                return GpuTensorHandle(cached), True

        payload = loader()
        expected = protocol.checked_tensor_bytes(rows, dimension, dtype)
        if len(payload) != expected:
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "resolved tensor byte length does not match its shape",
            )

        with self.lock:
            cached = self.entries.get(key)
            if cached is not None:
                frequency = self.sketch.increment(key)
                cached.references += 1
                cached.pinned = cached.pinned or pin
                cached.priority = self.inflation + frequency / self._entry_cost(cached)
                self.entries.move_to_end(key)
                self.hits += 1
                return GpuTensorHandle(cached), True
            allocated = self._allocate(key, len(payload), enforce_admission=False)
            assert allocated is not None
            arena, blocks, allocated_bytes = allocated
            try:
                arena.copy_from_host(blocks, payload)
            except Exception:
                arena.allocator.release(blocks)
                raise
            entry = _GpuCacheEntry(
                key,
                arena,
                blocks,
                allocated_bytes,
                len(payload),
                rows,
                dimension,
                dtype,
                references=1,
                pinned=pin,
                priority=self._priority(key, len(blocks)),
            )
            self.entries[key] = entry
            self.misses += 1
            self.loaded_bytes += len(payload)
            return GpuTensorHandle(entry), False

    def acquire_many(
        self,
        loads: list[GpuTensorLoad],
        *,
        enforce_admission: bool = True,
        record_access: bool = True,
        count_stats: bool = True,
    ) -> GpuAcquireBatch:
        """Acquire a request working set and upload all new slabs in batches.

        ``deferred`` items could not be allocated while earlier handles in the
        same request are referenced.  The caller scores/releases the returned
        handles and retries those items.  ``bypassed`` items lost TinyLFU
        admission and should be scored through the bounded streaming engine.
        """

        handles: list[GpuTensorHandle | None] = [None] * len(loads)
        bypassed: list[int] = []
        deferred: list[int] = []
        new_entries: list[tuple[int, _GpuCacheEntry, bytes]] = []
        hits = 0
        misses = 0
        admitted = 0
        with self.lock:
            try:
                for index, load in enumerate(loads):
                    expected = protocol.checked_tensor_bytes(
                        load.rows, load.dimension, load.dtype
                    )
                    if len(load.payload) != expected:
                        raise protocol.SidecarError(
                            protocol.STATUS_INVALID_REQUEST,
                            "resolved tensor byte length does not match its shape",
                        )
                    frequency = (
                        self.sketch.increment(load.key)
                        if record_access
                        else max(1, self.sketch.estimate(load.key))
                    )
                    cached = self.entries.get(load.key)
                    if cached is not None:
                        cached.references += 1
                        cached.pinned = cached.pinned or load.pin
                        cached.priority = (
                            self.inflation + frequency / self._entry_cost(cached)
                        )
                        self.entries.move_to_end(load.key)
                        handles[index] = GpuTensorHandle(cached)
                        hits += 1
                        continue
                    misses += 1
                    try:
                        allocation = self._allocate(
                            load.key,
                            len(load.payload),
                            enforce_admission and not load.pin,
                        )
                    except protocol.SidecarError as error:
                        if "one tensor exceeds" in str(error):
                            bypassed.append(index)
                            continue
                        deferred.append(index)
                        continue
                    if allocation is None:
                        bypassed.append(index)
                        continue
                    arena, blocks, allocated_bytes = allocation
                    entry = _GpuCacheEntry(
                        load.key,
                        arena,
                        blocks,
                        allocated_bytes,
                        len(load.payload),
                        load.rows,
                        load.dimension,
                        load.dtype,
                        references=1,
                        pinned=load.pin,
                        priority=self._priority(load.key, len(blocks)),
                    )
                    self.entries[load.key] = entry
                    handles[index] = GpuTensorHandle(entry)
                    new_entries.append((index, entry, load.payload))
                    admitted += 1

                by_arena: dict[int, tuple[GpuArena, list[tuple[tuple[int, ...], bytes]]]] = {}
                for _, entry, payload in new_entries:
                    bucket = by_arena.setdefault(id(entry.arena), (entry.arena, []))
                    bucket[1].append((entry.blocks, payload))
                for arena, items in by_arena.values():
                    arena.copy_many_from_host(items)
                if count_stats:
                    self.hits += hits
                    self.misses += misses
                self.loaded_bytes += sum(len(payload) for _, _, payload in new_entries)
            except Exception:
                new_ids = {id(entry) for _, entry, _ in new_entries}
                for handle in handles:
                    if handle is None:
                        continue
                    entry = handle.entry
                    if id(entry) in new_ids:
                        if self.entries.pop(entry.key, None) is entry:
                            entry.arena.allocator.release(entry.blocks)
                    else:
                        entry.references -= 1
                raise
        return GpuAcquireBatch(
            tuple(handles),
            tuple(bypassed),
            tuple(deferred),
            hits,
            misses,
            admitted,
        )

    def probe_many(
        self, keys: list[tuple[object, ...]]
    ) -> tuple[tuple[GpuTensorHandle | None, ...], tuple[int, ...]]:
        """Acquire GPU hits without resolving the corresponding host payloads."""

        handles: list[GpuTensorHandle | None] = [None] * len(keys)
        misses = []
        with self.lock:
            for index, key in enumerate(keys):
                frequency = self.sketch.increment(key)
                entry = self.entries.get(key)
                if entry is None:
                    misses.append(index)
                    self.misses += 1
                    continue
                entry.references += 1
                entry.priority = self.inflation + frequency / self._entry_cost(entry)
                self.entries.move_to_end(key)
                handles[index] = GpuTensorHandle(entry)
                self.hits += 1
        return tuple(handles), tuple(misses)

    def release(self, handle: GpuTensorHandle) -> None:
        with self.lock:
            entry = self.entries.get(handle.entry.key)
            if entry is not handle.entry or entry.references <= 0:
                raise RuntimeError("GPU tensor handle was released more than once")
            entry.references -= 1

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "entries": len(self.entries),
                "pinned_entries": sum(entry.pinned for entry in self.entries.values()),
                "active_references": sum(
                    entry.references for entry in self.entries.values()
                ),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "admission_rejections": self.admission_rejections,
                "policy": "tinylfu-gdsf",
                "gdsf_inflation": self.inflation,
                "loaded_bytes": self.loaded_bytes,
                "arenas": self.pool.status(),
            }
