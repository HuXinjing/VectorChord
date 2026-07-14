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
from typing import Callable

import torch

from devtools import tilemaxsim_reference_sidecar as protocol


_GPU_MEMORY_GB = re.compile(r"^(?:cuda:)?([0-9]+)=([0-9]+(?:\.[0-9]+)?)$")
_MEMORY_GB = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
GIB = 1024**3


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


class GpuArena:
    """A CUDA byte buffer acquired atomically during process startup."""

    def __init__(self, spec: GpuArenaSpec, workspace_bytes: int) -> None:
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
        self.capacity = (spec.total_bytes - workspace_bytes) // 256 * 256
        self.reserved_workspace_bytes = spec.total_bytes - self.capacity
        if self.capacity <= 0:
            raise RuntimeError(f"{spec.device} has no aligned tensor-cache capacity")
        self.storage: torch.Tensor | None = None
        self.allocator = FreeExtentAllocator(self.capacity)

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
            except Exception:
                self.storage = None
                torch.cuda.empty_cache()
                raise

    def tensor(
        self, offset: int, payload_bytes: int, rows: int, dimension: int, dtype: int
    ) -> torch.Tensor:
        if self.storage is None:
            raise RuntimeError("GPU arena is closed")
        scalar_dtype = torch.float32 if dtype == protocol.DTYPE_F32 else torch.float16
        return (
            self.storage.narrow(0, offset, payload_bytes)
            .view(scalar_dtype)
            .reshape(rows, dimension)
        )

    def copy_from_host(self, offset: int, payload: bytes) -> None:
        if self.storage is None:
            raise RuntimeError("GPU arena is closed")
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        self.storage.narrow(0, offset, len(payload)).copy_(source)

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
        }

    def close(self) -> None:
        if self.storage is None:
            return
        with torch.cuda.device(self.device):
            self.storage = None
            torch.cuda.empty_cache()


class GpuResourcePool:
    """Own all configured CUDA allocations or fail without a partial pool."""

    def __init__(self, specs: list[GpuArenaSpec], workspace_bytes: int) -> None:
        if not specs:
            raise RuntimeError("at least one GPU allocation is required")
        devices = [spec.device for spec in specs]
        if len(devices) != len(set(devices)):
            raise RuntimeError("each CUDA device may be configured only once")
        self.arenas: list[GpuArena] = []
        try:
            for spec in specs:
                self.arenas.append(GpuArena(spec, workspace_bytes))
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
    offset: int
    allocated_bytes: int
    payload_bytes: int
    rows: int
    dimension: int
    dtype: int
    references: int = 0
    pinned: bool = False


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
        return self.entry.offset

    def tensor(self) -> torch.Tensor:
        return self.entry.arena.tensor(
            self.entry.offset,
            self.entry.payload_bytes,
            self.entry.rows,
            self.entry.dimension,
            self.entry.dtype,
        )


class GpuTensorCache:
    """Thread-safe LRU directory over one or more process-owned GPU arenas."""

    def __init__(self, pool: GpuResourcePool, allow_eviction: bool) -> None:
        self.pool = pool
        self.allow_eviction = allow_eviction
        self.entries: OrderedDict[tuple[object, ...], _GpuCacheEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.loaded_bytes = 0

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

    def _evict_one(self) -> bool:
        for key, entry in self.entries.items():
            if entry.references or entry.pinned:
                continue
            self.entries.pop(key)
            entry.arena.allocator.release(entry.offset, entry.allocated_bytes)
            self.evictions += 1
            return True
        return False

    def _allocate(self, payload_bytes: int) -> tuple[GpuArena, int, int]:
        arena = self._find_arena(payload_bytes)
        while arena is None and self.allow_eviction and self._evict_one():
            arena = self._find_arena(payload_bytes)
        if arena is None:
            raise protocol.SidecarError(
                protocol.STATUS_RESOURCE_LIMIT,
                "configured GPU tensor arenas have insufficient contiguous capacity",
            )
        allocated = arena.allocator.allocate(payload_bytes)
        assert allocated is not None
        return arena, allocated[0], allocated[1]

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
            cached = self.entries.get(key)
            if cached is not None:
                cached.references += 1
                cached.pinned = cached.pinned or pin
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
                cached.references += 1
                cached.pinned = cached.pinned or pin
                self.entries.move_to_end(key)
                self.hits += 1
                return GpuTensorHandle(cached), True
            arena, offset, allocated_bytes = self._allocate(len(payload))
            try:
                arena.copy_from_host(offset, payload)
            except Exception:
                arena.allocator.release(offset, allocated_bytes)
                raise
            entry = _GpuCacheEntry(
                key,
                arena,
                offset,
                allocated_bytes,
                len(payload),
                rows,
                dimension,
                dtype,
                references=1,
                pinned=pin,
            )
            self.entries[key] = entry
            self.misses += 1
            self.loaded_bytes += len(payload)
            return GpuTensorHandle(entry), False

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
                "loaded_bytes": self.loaded_bytes,
                "arenas": self.pool.status(),
            }
