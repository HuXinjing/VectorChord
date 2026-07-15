# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Benchmark CUDA sidecar that resolves verified tensors from an NPY corpus.

This adapter avoids duplicating a large benchmark corpus into the production
content-addressed cache layout. It accepts only descriptors from a precomputed
manifest and verifies the canonical NPY payload against the descriptor SHA-256
before returning it to the normal bounded CUDA service.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import signal
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from devtools import tilemaxsim_reference_sidecar as protocol
from services import tilemaxsim_cuda_sidecar as sidecar
from services.build_tilemaxsim_tensor_cache import canonical_tensor


@dataclass(frozen=True)
class TensorRecord:
    path: Path
    rows: int
    dimension: int
    dtype: int
    checksum: str


class NpyManifestResolver:
    """Resolve only allowlisted, descriptor-verified benchmark NPY tensors."""

    def __init__(
        self,
        model_contract_id: str,
        source_manifest: Path,
        descriptor_manifest: Path,
        cache_bytes: int,
    ) -> None:
        self.model_contract_id = model_contract_id
        self.records = self._load_records(source_manifest, descriptor_manifest)
        self.cache = sidecar.PayloadCache(cache_bytes)

    @staticmethod
    def _json_lines(path: Path) -> list[dict[str, object]]:
        records = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(record)
        if not records:
            raise ValueError(f"{path}: manifest is empty")
        return records

    @classmethod
    def _load_records(
        cls, source_manifest: Path, descriptor_manifest: Path
    ) -> dict[str, TensorRecord]:
        source_root = source_manifest.resolve(strict=True).parent
        sources: dict[str, tuple[Path, int, int]] = {}
        for record in cls._json_lines(source_manifest):
            page_key = record.get("page_key")
            relative = record.get("embedding_file")
            rows = record.get("n_tokens")
            dimension = record.get("dim")
            if (
                not isinstance(page_key, str)
                or not isinstance(relative, str)
                or not isinstance(rows, int)
                or rows <= 0
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("source manifest contains an invalid tensor record")
            path = (source_root / relative).resolve(strict=True)
            try:
                path.relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    f"embedding path escapes source root: {relative}"
                ) from error
            if page_key in sources:
                raise ValueError(f"duplicate source page_key: {page_key}")
            sources[page_key] = (path, rows, dimension)

        resolved: dict[str, TensorRecord] = {}
        seen_pages = set()
        for descriptor in cls._json_lines(descriptor_manifest):
            page_key = descriptor.get("page_key")
            tensor_ref = descriptor.get("tensor_ref")
            rows = descriptor.get("tensor_rows")
            dimension = descriptor.get("tensor_dim")
            dtype_name = descriptor.get("tensor_dtype")
            checksum = descriptor.get("tensor_checksum")
            source = sources.get(page_key) if isinstance(page_key, str) else None
            if source is None:
                raise ValueError(f"descriptor has unknown page_key: {page_key!r}")
            source_path, source_rows, source_dimension = source
            if rows != source_rows or dimension != source_dimension:
                raise ValueError(f"descriptor shape disagrees for page {page_key}")
            dtype = {
                "float16": protocol.DTYPE_F16,
                "float32": protocol.DTYPE_F32,
            }.get(dtype_name)
            if dtype is None:
                raise ValueError(f"descriptor has invalid dtype for page {page_key}")
            if not isinstance(tensor_ref, str) or not tensor_ref.startswith(
                "sha256://"
            ):
                raise ValueError(
                    f"descriptor has invalid tensor_ref for page {page_key}"
                )
            digest = tensor_ref.removeprefix("sha256://")
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or checksum != f"sha256:{digest}"
            ):
                raise ValueError(f"descriptor has invalid SHA-256 for page {page_key}")
            tensor_record = TensorRecord(source_path, rows, dimension, dtype, checksum)
            previous = resolved.get(tensor_ref)
            if previous is not None:
                if (
                    previous.rows != tensor_record.rows
                    or previous.dimension != tensor_record.dimension
                    or previous.dtype != tensor_record.dtype
                    or previous.checksum != tensor_record.checksum
                ):
                    raise ValueError(f"conflicting duplicate tensor_ref: {tensor_ref}")
            else:
                resolved[tensor_ref] = tensor_record
            seen_pages.add(page_key)
        if seen_pages != set(sources):
            raise ValueError("descriptor manifest does not cover the source manifest")
        return resolved

    def resolve(
        self, request: protocol.ExternalTensorRequest
    ) -> sidecar.ResolvedPayload:
        if request.model_contract_id != self.model_contract_id:
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "model contract is not allowlisted by this benchmark resolver",
            )
        record = self.records.get(request.tensor_ref)
        if record is None:
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "tensor reference is not present in the benchmark manifest",
            )
        if (
            request.rows != record.rows
            or request.dimension != record.dimension
            or request.dtype != record.dtype
            or not hmac.compare_digest(request.checksum, record.checksum)
        ):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "tensor descriptor disagrees with the benchmark manifest",
            )
        key = (
            request.tensor_ref,
            request.rows,
            request.dimension,
            request.dtype,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return sidecar.ResolvedPayload(cached, True)

        dtype_name = "float16" if record.dtype == protocol.DTYPE_F16 else "float32"
        tensor = canonical_tensor(record.path, record.rows, record.dimension)
        if tensor.dtype != np.dtype(dtype_name):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "NPY dtype disagrees with the benchmark descriptor",
            )
        payload = memoryview(tensor).cast("B").tobytes()
        actual_checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if not hmac.compare_digest(actual_checksum, record.checksum):
            raise protocol.SidecarError(
                protocol.STATUS_INVALID_REQUEST,
                "NPY payload checksum disagrees with the benchmark descriptor",
            )
        self.cache.put(key, payload)
        return sidecar.ResolvedPayload(payload, False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--socket-mode", type=sidecar.parse_mode, default=0o600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-contract", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument(
        "--max-request-bytes", type=sidecar.positive_int, default=64 * 1024 * 1024
    )
    parser.add_argument(
        "--max-batch-tokens", type=sidecar.positive_int, default=1_000_000
    )
    parser.add_argument(
        "--max-tensor-bytes", type=sidecar.positive_int, default=1024 * 1024 * 1024
    )
    parser.add_argument("--max-candidates", type=sidecar.positive_int, default=65_536)
    parser.add_argument(
        "--max-device-bytes",
        type=sidecar.positive_int,
        default=8 * 1024 * 1024 * 1024,
    )
    parser.add_argument(
        "--cache-bytes", type=sidecar.nonnegative_int, default=2 * 1024 * 1024 * 1024
    )
    parser.add_argument(
        "--request-timeout-ms", type=sidecar.positive_int, default=60_000
    )
    parser.add_argument("--max-inflight", type=sidecar.positive_int, default=8)
    parser.add_argument("--max-cuda-inflight", type=sidecar.positive_int, default=1)
    parser.add_argument("--backlog", type=sidecar.positive_int, default=64)
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()

    if not args.model_contract:
        parser.error("--model-contract must not be empty")
    limits = protocol.Limits(
        max_request_bytes=args.max_request_bytes,
        max_batch_tokens=args.max_batch_tokens,
        max_tensor_bytes=args.max_tensor_bytes,
        max_candidates=args.max_candidates,
    )
    resolver = NpyManifestResolver(
        args.model_contract,
        args.manifest,
        args.descriptor_manifest,
        args.cache_bytes,
    )
    metrics = sidecar.JsonMetrics()
    engine = sidecar.TorchTileMaxsimEngine(
        args.device,
        args.max_device_bytes,
        args.allow_tf32,
        args.max_cuda_inflight,
    )
    service = sidecar.TileMaxsimService(
        limits, resolver, engine, args.request_timeout_ms, metrics
    )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    sidecar.serve(
        args.socket,
        args.socket_mode,
        args.backlog,
        args.max_inflight,
        service,
        stop,
    )


if __name__ == "__main__":
    main()
