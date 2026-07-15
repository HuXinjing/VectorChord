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
# Copyright (c) 2026 Hu Xinjing

"""Immutable, content-addressed TileMaxSim tensor shard format.

The data files contain canonical tensor bytes and alignment padding only.  A
generation index maps tensor SHA-256 digests to byte ranges.  Data files are
published under their own SHA-256, so a writer never mutates a visible shard.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


FORMAT = "vectorchord.tilemaxsim.shards"
VERSION = 1
INDEX_NAME = "tilemaxsim-shards-v1.json"
SHARD_DIRECTORY = "shards"
DEFAULT_ALIGNMENT = 4096
DEFAULT_SHARD_BYTES = 2 * 1024**3


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class ShardEntry:
    digest: str
    shard: str
    offset: int
    length: int
    rows: int
    dimension: int
    dtype: str


@dataclass(frozen=True)
class ShardFile:
    name: str
    size: int
    checksum: str


@dataclass(frozen=True)
class ShardIndex:
    alignment: int
    shards: dict[str, ShardFile]
    entries: dict[str, ShardEntry]


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ImmutableShardWriter:
    """Deterministically publish bounded immutable shards and one atomic index."""

    def __init__(
        self,
        root: Path,
        target_bytes: int = DEFAULT_SHARD_BYTES,
        alignment: int = DEFAULT_ALIGNMENT,
        fsync: bool = True,
    ) -> None:
        if target_bytes <= 0:
            raise ValueError("target shard bytes must be positive")
        if target_bytes < alignment:
            raise ValueError("target shard bytes must be at least one alignment unit")
        align_up(0, alignment)
        self.root = root
        self.shard_root = root / SHARD_DIRECTORY
        self.target_bytes = target_bytes
        self.alignment = alignment
        self.fsync = fsync
        self.entries: dict[str, ShardEntry] = {}
        self.shards: list[ShardFile] = []
        self._stream: BinaryIO | None = None
        self._temporary: Path | None = None
        self._hasher: hashlib._Hash | None = None
        self._offset = 0
        self._pending: list[tuple[str, int, int, int, int, str]] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_root.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        descriptor, name = tempfile.mkstemp(
            prefix=".tilemaxsim-shard-", suffix=".tmp", dir=self.shard_root
        )
        self._stream = os.fdopen(descriptor, "wb", closefd=True)
        self._temporary = Path(name)
        self._hasher = hashlib.sha256()
        self._offset = 0
        self._pending = []

    def _write(self, payload: bytes | memoryview) -> None:
        assert self._stream is not None and self._hasher is not None
        self._stream.write(payload)
        self._hasher.update(payload)
        self._offset += len(payload)

    def add(
        self,
        digest: str,
        payload: bytes | memoryview,
        rows: int,
        dimension: int,
        dtype: str,
    ) -> None:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid tensor SHA-256 digest")
        if digest in self.entries or any(item[0] == digest for item in self._pending):
            return
        if not payload:
            raise ValueError("tensor payload must not be empty")
        if rows <= 0 or dimension <= 0 or dtype not in ("float16", "float32"):
            raise ValueError("invalid tensor metadata")
        padded = align_up(len(payload), self.alignment)
        if self._stream is not None and self._offset and self._offset + padded > self.target_bytes:
            self._finish_shard()
        if self._stream is None:
            self._open()
        offset = self._offset
        self._write(payload)
        padding = padded - len(payload)
        if padding:
            self._write(bytes(min(padding, self.alignment)))
            remaining = padding - min(padding, self.alignment)
            while remaining:
                chunk = min(remaining, self.alignment)
                self._write(bytes(chunk))
                remaining -= chunk
        self._pending.append((digest, offset, len(payload), rows, dimension, dtype))

    def _finish_shard(self) -> None:
        if self._stream is None:
            return
        assert self._temporary is not None and self._hasher is not None
        stream = self._stream
        stream.flush()
        if self.fsync:
            os.fsync(stream.fileno())
        stream.close()
        digest = self._hasher.hexdigest()
        name = f"sha256-{digest}.vts"
        destination = self.shard_root / name
        size = self._offset
        if destination.exists():
            if destination.stat().st_size != size or _digest_file(destination) != digest:
                raise ValueError(f"existing immutable shard is corrupt: {destination}")
            self._temporary.unlink()
        else:
            os.replace(self._temporary, destination)
            if self.fsync:
                descriptor = os.open(self.shard_root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        relative = f"{SHARD_DIRECTORY}/{name}"
        shard = ShardFile(relative, size, f"sha256:{digest}")
        self.shards.append(shard)
        for tensor_digest, offset, length, rows, dimension, dtype in self._pending:
            self.entries[tensor_digest] = ShardEntry(
                tensor_digest,
                relative,
                offset,
                length,
                rows,
                dimension,
                dtype,
            )
        self._stream = None
        self._temporary = None
        self._hasher = None
        self._offset = 0
        self._pending = []

    def finish(self) -> Path:
        self._finish_shard()
        if not self.entries:
            raise ValueError("cannot publish an empty shard generation")
        document = {
            "format": FORMAT,
            "version": VERSION,
            "alignment": self.alignment,
            "shards": [
                {"name": shard.name, "bytes": shard.size, "checksum": shard.checksum}
                for shard in self.shards
            ],
            "entries": [
                {
                    "digest": entry.digest,
                    "shard": entry.shard,
                    "offset": entry.offset,
                    "length": entry.length,
                    "rows": entry.rows,
                    "dimension": entry.dimension,
                    "dtype": entry.dtype,
                }
                for entry in self.entries.values()
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{INDEX_NAME}.", suffix=".tmp", dir=self.root, text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
                json.dump(document, stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                if self.fsync:
                    os.fsync(stream.fileno())
            destination = self.root / INDEX_NAME
            os.replace(temporary, destination)
            if self.fsync:
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return destination
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._temporary is not None:
            try:
                self._temporary.unlink()
            except FileNotFoundError:
                pass
        self._stream = None
        self._temporary = None


def parse_index(document: object) -> ShardIndex:
    if not isinstance(document, dict):
        raise ValueError("shard index must be a JSON object")
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError("unsupported TileMaxSim shard index")
    alignment = document.get("alignment")
    if not isinstance(alignment, int):
        raise ValueError("shard index has invalid alignment")
    align_up(0, alignment)
    raw_shards = document.get("shards")
    raw_entries = document.get("entries")
    if not isinstance(raw_shards, list) or not isinstance(raw_entries, list):
        raise ValueError("shard index has invalid arrays")
    shards: dict[str, ShardFile] = {}
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ValueError("shard index contains an invalid shard")
        name, size, checksum = raw.get("name"), raw.get("bytes"), raw.get("checksum")
        if (
            not isinstance(name, str)
            or not name.startswith(f"{SHARD_DIRECTORY}/sha256-")
            or "/" in name[len(SHARD_DIRECTORY) + 1 :]
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(checksum, str)
            or not checksum.startswith("sha256:")
            or name != f"{SHARD_DIRECTORY}/sha256-{checksum.removeprefix('sha256:')}.vts"
            or name in shards
        ):
            raise ValueError("shard index contains invalid shard metadata")
        shards[name] = ShardFile(name, size, checksum)
    entries: dict[str, ShardEntry] = {}
    intervals: dict[str, list[tuple[int, int]]] = {name: [] for name in shards}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("shard index contains an invalid tensor entry")
        digest = raw.get("digest")
        shard = raw.get("shard")
        offset = raw.get("offset")
        length = raw.get("length")
        rows = raw.get("rows")
        dimension = raw.get("dimension")
        dtype = raw.get("dtype")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not isinstance(shard, str)
            or shard not in shards
            or not isinstance(offset, int)
            or offset < 0
            or offset % alignment
            or not isinstance(length, int)
            or length <= 0
            or not isinstance(rows, int)
            or rows <= 0
            or not isinstance(dimension, int)
            or dimension <= 0
            or dtype not in ("float16", "float32")
            or digest in entries
        ):
            raise ValueError("shard index contains invalid tensor metadata")
        scalar_bytes = 2 if dtype == "float16" else 4
        if length != rows * dimension * scalar_bytes:
            raise ValueError("shard tensor length disagrees with its shape")
        end = offset + length
        if end > shards[shard].size:
            raise ValueError("shard tensor range is outside its data file")
        intervals[shard].append((offset, align_up(end, alignment)))
        entries[digest] = ShardEntry(
            digest, shard, offset, length, rows, dimension, dtype
        )
    if not entries:
        raise ValueError("shard index contains no tensor entries")
    for shard, ranges in intervals.items():
        previous_end = 0
        for start, end in sorted(ranges):
            if start < previous_end:
                raise ValueError(f"overlapping tensor ranges in {shard}")
            previous_end = end
    return ShardIndex(alignment, shards, entries)


def load_index(path: Path) -> ShardIndex:
    with path.open(encoding="utf-8") as stream:
        return parse_index(json.load(stream))


def iter_index_entries(index: ShardIndex) -> Iterable[ShardEntry]:
    return index.entries.values()
