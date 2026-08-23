# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Build and evaluate orthogonal TileMaxSim compression ablations on one corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import triton

from services.tilemaxsim_quantization import (
    ProductQuantizer,
    ResidualProductQuantizer,
    pool_tokens,
    quantize_float8,
    quantize_int8,
    sample_training_rows,
    train_opq,
    train_product_quantizer,
    train_residual_product_quantizer,
)
from services.tilemaxsim_quantized_triton import (
    ragged_adc_tilemaxsim,
    ragged_scaled_tilemaxsim,
)
from services.tilemaxsim_triton import ragged_tilemaxsim_fp16


@dataclass(frozen=True)
class Variant:
    name: str
    pooling: int = 1
    encoding: str = "fp16"
    subspaces: int = 0
    centroids: int = 0
    residual_stages: int = 1
    opq_iterations: int = 0
    fused: bool = True
    storage_name: str = ""
    normalize_pooling: bool = False


VARIANTS = (
    Variant("exact-fp16"),
    Variant("document-mean-fp16", pooling=0, normalize_pooling=True),
    Variant("pool2-fp16", pooling=2),
    Variant("pool2-normalized-fp16", pooling=2, normalize_pooling=True),
    Variant("pool4-fp16", pooling=4),
    Variant("pool4-normalized-fp16", pooling=4, normalize_pooling=True),
    Variant("int8", encoding="int8"),
    Variant("fp8-e4m3", encoding="fp8"),
    Variant("int8-unfused", encoding="int8", fused=False, storage_name="int8"),
    Variant(
        "fp8-e4m3-unfused",
        encoding="fp8",
        fused=False,
        storage_name="fp8-e4m3",
    ),
    Variant("pq-m8-b8", encoding="pq", subspaces=8, centroids=256),
    Variant("pq-m16-b4", encoding="pq", subspaces=16, centroids=16),
    Variant("pq-m16-b8", encoding="pq", subspaces=16, centroids=256),
    Variant(
        "pq-m16-b8-unfused",
        encoding="pq",
        subspaces=16,
        centroids=256,
        fused=False,
        storage_name="pq-m16-b8",
    ),
    Variant("pq-m32-b8", encoding="pq", subspaces=32, centroids=256),
    Variant(
        "rpq2-m16-b8", encoding="pq", subspaces=16, centroids=256, residual_stages=2
    ),
    Variant(
        "rpq2-m16-b8-unfused",
        encoding="pq",
        subspaces=16,
        centroids=256,
        residual_stages=2,
        fused=False,
        storage_name="rpq2-m16-b8",
    ),
    Variant(
        "rpq3-m16-b8", encoding="pq", subspaces=16, centroids=256, residual_stages=3
    ),
    Variant("opq-m16-b8", encoding="pq", subspaces=16, centroids=256, opq_iterations=4),
    Variant("pool2-int8", pooling=2, encoding="int8"),
    Variant("pool2-fp8-e4m3", pooling=2, encoding="fp8"),
    Variant("pool2-pq-m16-b8", pooling=2, encoding="pq", subspaces=16, centroids=256),
    Variant(
        "pool2-rpq2-m16-b8",
        pooling=2,
        encoding="pq",
        subspaces=16,
        centroids=256,
        residual_stages=2,
    ),
    Variant(
        "pool2-opq-m16-b8",
        pooling=2,
        encoding="pq",
        subspaces=16,
        centroids=256,
        opq_iterations=4,
    ),
    Variant(
        "pool2-opq-rpq2-m16-b8",
        pooling=2,
        encoding="pq",
        subspaces=16,
        centroids=256,
        residual_stages=2,
        opq_iterations=4,
    ),
    Variant(
        "pool2-normalized-opq-rpq2-m16-b8",
        pooling=2,
        encoding="pq",
        subspaces=16,
        centroids=256,
        residual_stages=2,
        opq_iterations=4,
        normalize_pooling=True,
    ),
    Variant(
        "pool4-normalized-opq-rpq3-m16-b8",
        pooling=4,
        encoding="pq",
        subspaces=16,
        centroids=256,
        residual_stages=3,
        opq_iterations=4,
        normalize_pooling=True,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class ShardReader:
    def __init__(self, root: Path) -> None:
        index = json.loads((root / "tilemaxsim-shards-v1.json").read_text())
        self.root = root
        self.entries = {item["digest"]: item for item in index["entries"]}
        self.handles: dict[str, Any] = {}

    def load(self, digest: str) -> np.ndarray:
        item = self.entries[digest]
        relative = item["shard"]
        handle = self.handles.get(relative)
        if handle is None:
            handle = (self.root / relative).open("rb")
            self.handles[relative] = handle
        handle.seek(int(item["offset"]))
        payload = handle.read(int(item["length"]))
        if len(payload) != int(item["length"]):
            raise IOError(f"short shard read for {digest}")
        return np.frombuffer(payload, dtype="<f2").reshape(
            int(item["rows"]), int(item["dimension"])
        )

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


class Dataset:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        coverage = self.manifest.get("coverage", {})
        required = {
            "corpus_documents": 1124,
            "documents": 1124,
            "queries": 81,
        }
        if any(int(coverage.get(key, -1)) != value for key, value in required.items()):
            raise ValueError(f"dataset coverage gate failed: {coverage}")
        if coverage.get("relevant_documents") != coverage.get(
            "relevant_documents_covered"
        ):
            raise ValueError("not all qrel documents have visual tensors")
        self.corpus = [
            json.loads(line)
            for line in Path(self.manifest["corpus"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.doc_ids = [item["doc_id"] for item in self.corpus]
        self.pages = [self.manifest["documents"][doc_id] for doc_id in self.doc_ids]
        self.qrels = json.loads(
            Path(self.manifest["qrels"]).read_text(encoding="utf-8")
        )["queries"]
        self.query_path = Path(self.manifest["query_tensors"])
        self.shards = ShardReader(Path(self.manifest["shard_root"]))

    def close(self) -> None:
        self.shards.close()

    def load_page(self, page: dict[str, Any]) -> torch.Tensor:
        source = page["source"]
        if source["kind"] == "shard":
            array = self.shards.load(source["digest"])
        elif source["kind"] == "file":
            array = np.memmap(
                source["path"],
                mode="r",
                dtype="<f2",
                shape=(int(page["tensor_rows"]), int(page["tensor_dim"])),
            )
        else:
            raise ValueError(f"unknown tensor source: {source['kind']}")
        # Shard buffers and memmaps are read-only; copy before exposing to torch.
        return torch.from_numpy(np.array(array, copy=True))

    def iter_documents(
        self, pooling: int, normalize_pooling: bool = False
    ) -> Iterator[torch.Tensor]:
        for index in range(len(self.pages)):
            yield self.load_document(index, pooling, normalize_pooling)

    def load_document(
        self, index: int, pooling: int, normalize_pooling: bool = False
    ) -> torch.Tensor:
        if pooling == 0:
            tokens = torch.cat(
                [self.load_page(page) for page in self.pages[index]], dim=0
            )
            pooled = tokens.mean(dim=0, keepdim=True)
            return torch.nn.functional.normalize(pooled.float(), dim=1).to(tokens.dtype)
        return torch.cat(
            [
                pool_tokens(self.load_page(page), pooling, normalize=normalize_pooling)
                for page in self.pages[index]
            ],
            dim=0,
        )

    def row_counts(self, pooling: int) -> np.ndarray:
        if pooling == 0:
            return np.ones(len(self.pages), dtype=np.int32)
        return np.asarray(
            [
                sum(math.ceil(int(page["tensor_rows"]) / pooling) for page in pages)
                for pages in self.pages
            ],
            dtype=np.int32,
        )


def variant_by_name(name: str) -> Variant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"unknown variant {name!r}")


def storage_config(variant: Variant) -> dict[str, Any]:
    return {
        "pooling": variant.pooling,
        "encoding": variant.encoding,
        "subspaces": variant.subspaces,
        "centroids": variant.centroids,
        "residual_stages": variant.residual_stages,
        "opq_iterations": variant.opq_iterations,
        "normalize_pooling": variant.normalize_pooling,
    }


def train_quantizer(
    variant: Variant, training: torch.Tensor, iterations: int, seed: int
) -> ProductQuantizer | ResidualProductQuantizer:
    if variant.residual_stages > 1:
        return train_residual_product_quantizer(
            training,
            variant.subspaces,
            variant.centroids,
            variant.residual_stages,
            iterations=iterations,
            seed=seed,
            opq_outer_iterations=variant.opq_iterations,
        )
    if variant.opq_iterations:
        return train_opq(
            training,
            variant.subspaces,
            variant.centroids,
            outer_iterations=variant.opq_iterations,
            kmeans_iterations=iterations,
            seed=seed,
        )
    return train_product_quantizer(
        training,
        variant.subspaces,
        variant.centroids,
        iterations=iterations,
        seed=seed,
    )


def save_quantizer(
    path: Path, quantizer: ProductQuantizer | ResidualProductQuantizer
) -> None:
    stages = (
        (quantizer,) if isinstance(quantizer, ProductQuantizer) else quantizer.stages
    )
    arrays: dict[str, np.ndarray] = {
        "stage_count": np.asarray([len(stages)], dtype=np.int32)
    }
    for index, stage in enumerate(stages):
        arrays[f"codebooks_{index}"] = stage.codebooks.detach().cpu().numpy()
        arrays[f"has_rotation_{index}"] = np.asarray(
            [stage.rotation is not None], dtype=np.bool_
        )
        if stage.rotation is not None:
            arrays[f"rotation_{index}"] = stage.rotation.detach().cpu().numpy()
    np.savez(path, **arrays)


def load_quantizer(path: Path) -> ProductQuantizer | ResidualProductQuantizer:
    with np.load(path, allow_pickle=False) as archive:
        stage_count = int(archive["stage_count"][0])
        if stage_count <= 0:
            raise ValueError("quantizer has no stages")
        stages = []
        for index in range(stage_count):
            codebooks = torch.from_numpy(archive[f"codebooks_{index}"].copy())
            rotation = (
                torch.from_numpy(archive[f"rotation_{index}"].copy())
                if bool(archive[f"has_rotation_{index}"][0])
                else None
            )
            stages.append(ProductQuantizer(codebooks, rotation))
    if len(stages) == 1:
        return stages[0]
    return ResidualProductQuantizer(tuple(stages))


def build_variant(
    dataset: Dataset,
    variant: Variant,
    output_root: Path,
    training_rows: int,
    iterations: int,
    seed: int,
    device: torch.device,
) -> Path:
    output = output_root / (variant.storage_name or variant.name)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    source_manifest_checksum = file_sha256(dataset.manifest_path)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("complete")
            and metadata.get("storage") == storage_config(variant)
            and metadata.get("source_manifest_checksum") == source_manifest_checksum
        ):
            return output
    original_rows = dataset.row_counts(variant.pooling)
    order = np.argsort(original_rows, kind="stable")
    rows = original_rows[order]
    stored_doc_ids = [dataset.doc_ids[int(index)] for index in order]
    offsets = np.zeros(len(rows), dtype=np.int64)
    offsets[1:] = np.cumsum(rows[:-1], dtype=np.int64)
    total_rows = int(rows.sum())
    dimension = int(dataset.manifest["dimension"])
    np.save(output / "rows.npy", rows, allow_pickle=False)
    np.save(output / "offsets.npy", offsets, allow_pickle=False)
    (output / "doc-ids.json").write_text(
        json.dumps(stored_doc_ids, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    base_name = {
        (0, True): "document-mean-fp16",
        (1, False): "exact-fp16",
        (2, False): "pool2-fp16",
        (2, True): "pool2-normalized-fp16",
        (4, False): "pool4-fp16",
        (4, True): "pool4-normalized-fp16",
    }.get((variant.pooling, variant.normalize_pooling))
    base_root = output_root / base_name if base_name else None
    use_base = (
        variant.encoding != "fp16"
        and base_root is not None
        and (base_root / "metadata.json").is_file()
        and (base_root / "values.npy").is_file()
    )

    def source_documents() -> Iterator[torch.Tensor]:
        if use_base:
            assert base_root is not None
            base_metadata = json.loads(
                (base_root / "metadata.json").read_text(encoding="utf-8")
            )
            if (
                not base_metadata.get("complete")
                or base_metadata.get("source_manifest_checksum")
                != source_manifest_checksum
            ):
                raise ValueError(f"incomplete or stale FP16 base cache: {base_root}")
            base_rows = np.load(base_root / "rows.npy", mmap_mode="r")
            base_offsets = np.load(base_root / "offsets.npy", mmap_mode="r")
            if not np.array_equal(base_rows, rows):
                raise ValueError(
                    "FP16 base row layout disagrees with compression variant"
                )
            base_doc_ids = json.loads((base_root / "doc-ids.json").read_text())
            if base_doc_ids != stored_doc_ids:
                raise ValueError(
                    "FP16 base document order disagrees with compression variant"
                )
            base_values = np.load(base_root / "values.npy", mmap_mode="r")
            for index, count in enumerate(base_rows):
                start = int(base_offsets[index])
                end = start + int(count)
                yield torch.from_numpy(np.array(base_values[start:end], copy=True))
            return
        for original_index in order:
            yield dataset.load_document(
                int(original_index),
                variant.pooling,
                variant.normalize_pooling,
            )

    quantizer = None
    training_ms = 0.0
    if variant.encoding == "pq":
        started = time.perf_counter()
        training = sample_training_rows(source_documents(), training_rows, seed)
        quantizer = train_quantizer(variant, training.to(device), iterations, seed)
        training_ms = (time.perf_counter() - started) * 1000
        save_quantizer(output / "quantizer.npz", quantizer)

    started = time.perf_counter()
    if variant.encoding == "fp16":
        values = np.lib.format.open_memmap(
            output / "values.npy", mode="w+", dtype="<f2", shape=(total_rows, dimension)
        )
    elif variant.encoding == "int8":
        values = np.lib.format.open_memmap(
            output / "values.npy",
            mode="w+",
            dtype=np.int8,
            shape=(total_rows, dimension),
        )
        scales = np.lib.format.open_memmap(
            output / "scales.npy", mode="w+", dtype="<f2", shape=(total_rows,)
        )
    elif variant.encoding == "fp8":
        values = np.lib.format.open_memmap(
            output / "values.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total_rows, dimension),
        )
        scales = np.lib.format.open_memmap(
            output / "scales.npy", mode="w+", dtype="<f2", shape=(total_rows,)
        )
    elif variant.encoding == "pq":
        values = np.lib.format.open_memmap(
            output / "codes.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total_rows, variant.residual_stages, variant.subspaces),
        )
    else:
        raise ValueError(f"unsupported encoding: {variant.encoding}")

    position = 0
    for index, document in enumerate(source_documents()):
        end = position + document.shape[0]
        if end - position != rows[index]:
            raise RuntimeError("encoded row count disagrees with manifest")
        if variant.encoding == "fp16":
            values[position:end] = document.numpy()
        elif variant.encoding == "int8":
            encoded = quantize_int8(document)
            values[position:end] = encoded.values.numpy()
            scales[position:end] = encoded.scales.numpy()
        elif variant.encoding == "fp8":
            encoded = quantize_float8(document)
            # NumPy has no portable float8 dtype; persist the exact raw byte.
            values[position:end] = encoded.values.view(torch.uint8).numpy()
            scales[position:end] = encoded.scales.numpy()
        else:
            assert quantizer is not None
            # PQ's temporary distance arena is [rows, subspaces, centroids].
            # Stream large documents so cache construction remains bounded on
            # a GPU shared with serving workloads. Keep at most ~16 MiB of
            # distance workspace per encode call (before allocator overhead).
            distance_bytes_per_row = variant.subspaces * variant.centroids * 4
            encoding_rows = max(
                1, min(8192, (16 * 1024**2) // distance_bytes_per_row)
            )
            chunk_position = position
            for document_chunk in document.split(encoding_rows):
                encoded = quantizer.encode(document_chunk.to(device))
                stages = (
                    (encoded,) if isinstance(encoded, torch.Tensor) else encoded
                )
                chunk_end = chunk_position + document_chunk.shape[0]
                values[chunk_position:chunk_end] = torch.stack(
                    tuple(stage.cpu() for stage in stages), dim=1
                ).numpy()
                chunk_position = chunk_end
            if chunk_position != end:
                raise RuntimeError("chunked PQ encoding lost document rows")
        position = end
    values.flush()
    if variant.encoding in ("int8", "fp8"):
        scales.flush()
    encoding_ms = (time.perf_counter() - started) * 1000
    files = [path for path in output.iterdir() if path.is_file()]
    metadata = {
        "version": 1,
        "complete": True,
        "storage": storage_config(variant),
        "documents": len(rows),
        "rows": total_rows,
        "dimension": dimension,
        "training_rows": min(training_rows, total_rows) if quantizer else 0,
        "training_ms": training_ms,
        "encoding_ms": encoding_ms,
        "artifact_bytes": sum(path.stat().st_size for path in files),
        "source_manifest": str(dataset.manifest_path.resolve()),
        "source_manifest_checksum": source_manifest_checksum,
        "source_fp16_cache": str(base_root.resolve()) if use_base else None,
    }
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    return output


def batches(
    rows: np.ndarray, bytes_per_row: int, maximum_bytes: int
) -> Iterator[tuple[int, int]]:
    start = 0
    used = 0
    for index, count in enumerate(rows):
        size = int(count) * bytes_per_row
        if index > start and used + size > maximum_bytes:
            yield start, index
            start, used = index, 0
        # An individual document may be larger than the transfer budget. Keep
        # it as a singleton range; score_variant streams that document in token
        # chunks and merges the per-query-token maxima exactly.
        if size > maximum_bytes:
            yield index, index + 1
            start, used = index + 1, 0
            continue
        used += size
    if start < len(rows):
        yield start, len(rows)


def score_unfused_document_chunks(
    query: torch.Tensor,
    values: np.ndarray,
    scales: np.ndarray | None,
    quantizer: ProductQuantizer | ResidualProductQuantizer | None,
    variant: Variant,
    row_start: int,
    row_end: int,
    batch_bytes: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """Score one oversized document without changing exact MaxSim semantics."""
    dimension = query.shape[1]
    if variant.encoding in ("int8", "fp8"):
        encoded_bytes_per_row = dimension + 2
    else:
        encoded_bytes_per_row = variant.subspaces * variant.residual_stages
    # Peak includes compressed input, FP16 reconstruction, the FP32 operand
    # used to preserve dot accumulation semantics, and the query-by-chunk
    # similarity matrix. The factor of two leaves allocator/workspace headroom
    # on a GPU shared with other services.
    peak_bytes_per_row = (
        encoded_bytes_per_row
        + dimension * 2
        + dimension * 4
        + query.shape[0] * 4
    )
    chunk_rows = max(1, batch_bytes // (2 * peak_bytes_per_row))
    maxima = torch.full(
        (query.shape[0],), -torch.inf, dtype=torch.float32, device=device
    )
    transfer_ms = 0.0
    kernel_ms = 0.0
    for chunk_start in range(row_start, row_end, chunk_rows):
        chunk_end = min(row_end, chunk_start + chunk_rows)
        transfer_started = time.perf_counter()
        encoded = torch.from_numpy(
            np.array(values[chunk_start:chunk_end], copy=True)
        )
        if variant.encoding == "fp8":
            encoded = encoded.view(torch.float8_e4m3fn)
        encoded = encoded.to(device)
        chunk_scales = None
        if scales is not None:
            chunk_scales = torch.from_numpy(
                np.array(scales[chunk_start:chunk_end], copy=True)
            ).to(device)
        torch.cuda.synchronize(device)
        transfer_ms += (time.perf_counter() - transfer_started) * 1000

        kernel_started = time.perf_counter()
        if variant.encoding in ("int8", "fp8"):
            assert chunk_scales is not None
            reconstructed = encoded.to(torch.float16) * chunk_scales.to(
                torch.float16
            )[:, None]
        else:
            assert quantizer is not None
            stage_codes = tuple(
                encoded[:, index, :] for index in range(variant.residual_stages)
            )
            reconstructed = (
                quantizer.decode(stage_codes[0])
                if isinstance(quantizer, ProductQuantizer)
                else quantizer.decode(stage_codes)
            ).to(torch.float16)
        chunk_maxima = torch.matmul(
            query.to(torch.float32), reconstructed.to(torch.float32).T
        ).amax(dim=1)
        maxima = torch.maximum(maxima, chunk_maxima)
        torch.cuda.synchronize(device)
        kernel_ms += (time.perf_counter() - kernel_started) * 1000
    return maxima.sum(), transfer_ms, kernel_ms


def ragged_batch_metadata(
    rows: np.ndarray, start: int, end: int, row_width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    selected = rows[start:end]
    offsets = np.zeros(len(selected), dtype=np.int64)
    offsets[1:] = np.cumsum(selected[:-1], dtype=np.int64) * row_width
    return (
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(selected.copy()).to(device),
        int(selected.max()),
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def score_variant(
    dataset: Dataset,
    variant: Variant,
    cache: Path,
    device: torch.device,
    batch_bytes: int,
    warmups: int,
) -> dict[str, Any]:
    rows = np.load(cache / "rows.npy", mmap_mode="r")
    offsets = np.load(cache / "offsets.npy", mmap_mode="r")
    stored_doc_ids = json.loads((cache / "doc-ids.json").read_text(encoding="utf-8"))
    if len(stored_doc_ids) != len(rows) or set(stored_doc_ids) != set(dataset.doc_ids):
        raise ValueError("cache document IDs disagree with the frozen corpus")
    dimension = int(dataset.manifest["dimension"])
    quantizer = None
    if variant.encoding == "fp16":
        values = np.load(cache / "values.npy", mmap_mode="r")
        bytes_per_row = dimension * 2
    elif variant.encoding == "int8":
        values = np.load(cache / "values.npy", mmap_mode="r")
        scales = np.load(cache / "scales.npy", mmap_mode="r")
        bytes_per_row = dimension + 2
    elif variant.encoding == "fp8":
        values = np.load(cache / "values.npy", mmap_mode="r")
        scales = np.load(cache / "scales.npy", mmap_mode="r")
        bytes_per_row = dimension + 2
    else:
        values = np.load(cache / "codes.npy", mmap_mode="r")
        bytes_per_row = variant.subspaces * variant.residual_stages
        quantizer = load_quantizer(cache / "quantizer.npz")
    if not variant.fused and variant.encoding != "fp16":
        # The unfused oracle holds compressed input plus reconstruction. INT8
        # and FP8 reconstruct directly to FP16. PQ decode uses FP32 codebooks;
        # conservatively include decode, OPQ/RPQ intermediate, and final FP16
        # arenas so the planner does not rely on allocator luck.
        bytes_per_row += dimension * (10 if variant.encoding == "pq" else 2)
    batch_ranges = list(batches(rows, bytes_per_row, batch_bytes))
    planned_batch_bytes = [
        int(rows[start:end].sum()) * bytes_per_row for start, end in batch_ranges
    ]
    oversized_singletons = sum(
        end == start + 1 and planned > batch_bytes
        for (start, end), planned in zip(
            batch_ranges, planned_batch_bytes, strict=True
        )
    )
    relevant = {item["id"]: set(item["relevant"]) for item in dataset.qrels}
    latencies = []
    transfer_latencies = []
    kernel_latencies = []
    query_prep_latencies = []
    results = []

    with np.load(dataset.query_path, allow_pickle=False) as queries:
        for query_index, item in enumerate(dataset.qrels):
            started_query = time.perf_counter()
            query = torch.from_numpy(queries[item["id"]].astype("<f2", copy=False)).to(
                device
            )
            query_luts = None
            if variant.encoding == "pq" and variant.fused:
                assert quantizer is not None
                quantizer_stages = (
                    (quantizer,)
                    if isinstance(quantizer, ProductQuantizer)
                    else quantizer.stages
                )
                query_luts = [
                    stage.adc_lut(query).to(device) for stage in quantizer_stages
                ]
            torch.cuda.synchronize(device)
            query_prep_ms = (time.perf_counter() - started_query) * 1000
            scores = torch.empty(len(rows), dtype=torch.float32)
            transfer_ms = 0.0
            kernel_ms = 0.0
            for start, end in batch_ranges:
                row_start = int(offsets[start])
                row_end = int(offsets[end - 1] + rows[end - 1])
                oversized_unfused = (
                    not variant.fused
                    and end == start + 1
                    and int(rows[start]) * bytes_per_row > batch_bytes
                )
                if oversized_unfused:
                    score, chunk_transfer_ms, chunk_kernel_ms = (
                        score_unfused_document_chunks(
                            query,
                            values,
                            scales if variant.encoding in ("int8", "fp8") else None,
                            quantizer,
                            variant,
                            row_start,
                            row_end,
                            batch_bytes,
                            device,
                        )
                    )
                    scores[start] = score.cpu()
                    transfer_ms += chunk_transfer_ms
                    kernel_ms += chunk_kernel_ms
                    continue
                transfer_started = time.perf_counter()
                batch_encoded = torch.from_numpy(
                    np.array(values[row_start:row_end], copy=True)
                )
                if variant.encoding == "fp8":
                    batch_encoded = batch_encoded.view(torch.float8_e4m3fn)
                batch_encoded = batch_encoded.to(device)
                row_width = dimension
                if variant.encoding == "pq" and variant.fused:
                    row_width = variant.subspaces * variant.residual_stages
                document_offsets, document_rows, maximum_rows = ragged_batch_metadata(
                    rows,
                    start,
                    end,
                    row_width,
                    device,
                )
                batch_scales = None
                scale_offsets = None
                if variant.encoding in ("int8", "fp8"):
                    batch_scales = torch.from_numpy(
                        np.array(scales[row_start:row_end], copy=True)
                    ).to(device)
                    scale_offsets, _, _ = ragged_batch_metadata(
                        rows, start, end, 1, device
                    )
                torch.cuda.synchronize(device)
                transfer_ms += (time.perf_counter() - transfer_started) * 1000
                kernel_started = time.perf_counter()
                if variant.encoding == "fp16":
                    batch_scores = ragged_tilemaxsim_fp16(
                        query,
                        batch_encoded.flatten(),
                        document_offsets,
                        document_rows,
                        maximum_rows,
                    )
                elif variant.encoding in ("int8", "fp8") and variant.fused:
                    assert batch_scales is not None and scale_offsets is not None
                    batch_scores = ragged_scaled_tilemaxsim(
                        query,
                        batch_encoded.flatten(),
                        batch_scales,
                        document_offsets,
                        scale_offsets,
                        document_rows,
                        maximum_rows,
                    )
                elif variant.encoding == "pq" and variant.fused:
                    assert query_luts is not None
                    batch_scores = ragged_adc_tilemaxsim(
                        query_luts,
                        batch_encoded.flatten(),
                        document_offsets,
                        document_rows,
                        maximum_rows,
                    )
                else:
                    if variant.encoding in ("int8", "fp8"):
                        assert batch_scales is not None
                        reconstructed = batch_encoded.to(
                            torch.float16
                        ) * batch_scales.to(torch.float16)[:, None]
                    else:
                        assert quantizer is not None
                        stage_codes = tuple(
                            batch_encoded[:, index, :]
                            for index in range(variant.residual_stages)
                        )
                        reconstructed = (
                            quantizer.decode(stage_codes[0])
                            if isinstance(quantizer, ProductQuantizer)
                            else quantizer.decode(stage_codes)
                        )
                    batch_scores = ragged_tilemaxsim_fp16(
                        query,
                        reconstructed.to(torch.float16).flatten(),
                        document_offsets,
                        document_rows,
                        maximum_rows,
                    )
                torch.cuda.synchronize(device)
                kernel_ms += (time.perf_counter() - kernel_started) * 1000
                scores[start:end] = batch_scores.cpu()
            latency_ms = (time.perf_counter() - started_query) * 1000
            # Physical length bucketing must never become a ranking signal.
            # Resolve exact score ties by canonical document ID, independent of
            # cache layout and pooling variant.
            ranking_indices = np.lexsort(
                (np.asarray(stored_doc_ids), -scores.numpy())
            ).tolist()
            ranking = [stored_doc_ids[index] for index in ranking_indices]
            gold = relevant[item["id"]]
            ranking_positions = {
                doc_id: index + 1 for index, doc_id in enumerate(ranking)
            }
            relevant_ranks = {
                doc_id: ranking_positions.get(doc_id) for doc_id in sorted(gold)
            }
            rank = min(
                (value for value in relevant_ranks.values() if value is not None),
                default=None,
            )
            results.append(
                {
                    "query_id": item["id"],
                    "query": item["query"],
                    "relevant": sorted(gold),
                    "rank": rank,
                    "relevant_ranks": relevant_ranks,
                    "recall_at_1": sum(
                        value is not None and value <= 1
                        for value in relevant_ranks.values()
                    )
                    / len(gold),
                    "recall_at_5": sum(
                        value is not None and value <= 5
                        for value in relevant_ranks.values()
                    )
                    / len(gold),
                    "recall_at_10": sum(
                        value is not None and value <= 10
                        for value in relevant_ranks.values()
                    )
                    / len(gold),
                    "top_20": ranking[:20],
                    "latency_ms": latency_ms,
                    "query_prep_ms": query_prep_ms,
                    "transfer_ms": transfer_ms,
                    "kernel_ms": kernel_ms,
                }
            )
            if query_index >= warmups:
                latencies.append(latency_ms)
                transfer_latencies.append(transfer_ms)
                kernel_latencies.append(kernel_ms)
                query_prep_latencies.append(query_prep_ms)
    metadata = json.loads((cache / "metadata.json").read_text())
    properties = torch.cuda.get_device_properties(device)
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "version": 1,
        "variant": asdict(variant),
        "dataset": dataset.manifest["coverage"],
        "cache": metadata,
        "environment": {
            "device": str(device),
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "git_revision": revision,
            "git_dirty": dirty,
        },
        "gpu_batch_bytes": batch_bytes,
        "gpu_batches": len(batch_ranges),
        "maximum_planned_batch_bytes": max(planned_batch_bytes),
        "oversized_singleton_documents": oversized_singletons,
        "oversized_unfused_documents_streamed": (
            oversized_singletons if not variant.fused else 0
        ),
        "padding_row_ratio": sum(
            int(rows[start:end].max()) * (end - start) for start, end in batch_ranges
        )
        / int(rows.sum()),
        "latency_ms": metric_summary(latencies),
        "transfer_ms": metric_summary(transfer_latencies),
        "kernel_ms": metric_summary(kernel_latencies),
        "query_prep_ms": metric_summary(query_prep_latencies),
        "quality": {
            "recall_at_1": statistics.fmean(item["recall_at_1"] for item in results),
            "recall_at_5": statistics.fmean(item["recall_at_5"] for item in results),
            "recall_at_10": statistics.fmean(item["recall_at_10"] for item in results),
            "hit_at_1": statistics.fmean(
                float(item["rank"] is not None and item["rank"] <= 1)
                for item in results
            ),
            "hit_at_5": statistics.fmean(
                float(item["rank"] is not None and item["rank"] <= 5)
                for item in results
            ),
            "hit_at_10": statistics.fmean(
                float(item["rank"] is not None and item["rank"] <= 10)
                for item in results
            ),
            "mrr": statistics.fmean(
                0.0 if item["rank"] is None else 1.0 / item["rank"] for item in results
            ),
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--training-rows", type=int, default=20_000)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-batch-gb", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.list_variants:
        print("\n".join(item.name for item in VARIANTS))
        return
    missing_paths = [
        name
        for name in ("manifest", "cache_root", "report_root")
        if getattr(args, name) is None
    ]
    if missing_paths:
        parser.error(
            "the following arguments are required: "
            + ", ".join("--" + name.replace("_", "-") for name in missing_paths)
        )
    names = args.variant or [item.name for item in VARIANTS]
    variants = [variant_by_name(name) for name in names]
    if args.training_rows <= 0 or args.kmeans_iterations <= 0:
        parser.error("training rows and k-means iterations must be positive")
    if args.gpu_batch_gb <= 0 or args.warmups < 0:
        parser.error("GPU batch GB must be positive and warmups nonnegative")
    dataset = Dataset(args.manifest)
    try:
        args.cache_root.mkdir(parents=True, exist_ok=True)
        args.report_root.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            print(
                json.dumps({"event": "variant_build_started", "variant": variant.name}),
                flush=True,
            )
            cache = build_variant(
                dataset,
                variant,
                args.cache_root,
                args.training_rows,
                args.kmeans_iterations,
                args.seed,
                torch.device(args.device),
            )
            if args.build_only:
                continue
            print(
                json.dumps(
                    {"event": "variant_benchmark_started", "variant": variant.name}
                ),
                flush=True,
            )
            report = score_variant(
                dataset,
                variant,
                cache,
                torch.device(args.device),
                int(args.gpu_batch_gb * 1024**3),
                args.warmups,
            )
            destination = args.report_root / f"{variant.name}.json"
            destination.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "variant_complete",
                        "variant": variant.name,
                        **report["quality"],
                        "mean_latency_ms": report["latency_ms"]["mean"],
                    }
                ),
                flush=True,
            )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
