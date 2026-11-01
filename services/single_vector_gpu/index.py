"""Opt-in, snapshot-scoped single-vector cosine search. No SQL route is changed."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class Contract:
    tenant: str
    model: str
    generation: str


def validated(values: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or (dimensions and values.shape[1] != dimensions):
        raise ValueError("invalid_vector_shape")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_vector")
    if not np.all(np.any(values != 0, axis=1)):
        raise ValueError("zero_vector")
    return np.ascontiguousarray(values)


def normalized(values: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    values = validated(values, dimensions)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    return np.ascontiguousarray(values / norms[:, None], dtype=np.float32)


class SingleVectorIndex:
    """Immutable vectors and identities; the caller supplies an exact scope contract.

    One call has one authorized source set. A higher-level scheduler must group
    only compatible calls; it must not union their allowed source sets.
    Boundary ties may select any valid tied ID, matching distance-only SQL.
    """

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, sources: np.ndarray,
                 contract: Contract, *, device: str = "cuda:0", query_block: int = 512,
                 document_block: int = 8192, profile: str = "fp32",
                 memory_limit_bytes: int = 4 * 1024**3):
        if profile not in ("fp32", "fp16"):
            raise ValueError("unsupported_profile")
        if query_block < 1 or document_block < 1 or query_block * document_block * 4 > 128 * 1024**2:
            raise ValueError("workspace_limit")
        raw = normalized(vectors)
        ids = np.asarray(ids, dtype=np.int64)
        sources = np.asarray(sources, dtype=np.int64)
        if raw.shape[0] == 0 or ids.shape != (len(raw),) or sources.shape != ids.shape:
            raise ValueError("invalid_identity_shape")
        if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate_or_negative_identity")
        if not all((contract.tenant, contract.model, contract.generation)):
            raise ValueError("missing_contract")
        self.device = torch.device(device)
        self.contract, self.profile = contract, profile
        self.query_block, self.document_block = query_block, document_block
        self.sources, self.dimensions = sources.copy(), raw.shape[1]
        self.memory_limit_bytes = memory_limit_bytes
        dtype = torch.float32 if profile == "fp32" else torch.float16
        estimated = raw.size * torch.empty((), dtype=dtype).element_size() + len(raw) * 8
        workspace = query_block * document_block * 4 + query_block * self.dimensions * 4
        if estimated + workspace * 3 > memory_limit_bytes:
            raise ValueError("index_memory_limit")
        # Own-process precision policy; never changes driver or other services.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        self.vectors = torch.as_tensor(raw, device=self.device, dtype=dtype)
        self.ids = torch.as_tensor(ids, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_ms = (time.perf_counter() - started) * 1000

    @torch.inference_mode()
    def search(self, queries: np.ndarray, k: int, *, contract: Contract,
               allowed_sources: Iterable[int]) -> tuple[np.ndarray, np.ndarray, dict]:
        started = time.perf_counter()
        if contract != self.contract:
            raise PermissionError("snapshot_contract_mismatch")
        if type(k) is not int or not 1 <= k <= 1024:
            raise ValueError("invalid_topk")
        source_values = list(allowed_sources)
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in source_values):
            raise ValueError("invalid_source_identity")
        allowed = frozenset(int(x) for x in source_values)
        known = frozenset(int(x) for x in self.sources)
        if not allowed.issubset(known):
            raise PermissionError("source_not_in_snapshot")
        prepare = time.perf_counter()
        q = validated(queries, self.dimensions)
        k = min(k, len(self.sources))
        output_ids = np.full((len(q), k), -1, dtype=np.int64)
        output_scores = np.full((len(q), k), -np.inf, dtype=np.float32)
        mask = np.isin(self.sources, list(allowed))
        prepare_ms = (time.perf_counter() - prepare) * 1000
        h2d_ms = compute_ms = d2h_ms = 0.0
        if not mask.any() or not len(q):
            return output_ids, output_scores, {"wall_ms": (time.perf_counter()-started)*1000, "empty_scope": True}
        gpu_mask = torch.as_tensor(mask, device=self.device)
        for start in range(0, len(q), self.query_block):
            end = min(start + self.query_block, len(q))
            h2d_start = time.perf_counter()
            raw_batch = torch.as_tensor(q[start:end], dtype=torch.float32, device=self.device)
            h2d_ms += (time.perf_counter() - h2d_start) * 1000
            begin = finish = None
            if self.device.type == "cuda":
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(self.device))
            compute_start = time.perf_counter()
            # Normalize on device. Max scaling preserves finite nonzero inputs
            # without squaring tiny/large FP32 values into underflow/overflow.
            scaled = raw_batch / raw_batch.abs().amax(dim=1, keepdim=True)
            batch = (scaled / torch.linalg.vector_norm(scaled, dim=1, keepdim=True)).to(self.vectors.dtype)
            best = torch.full((len(batch), k), -torch.inf, device=self.device, dtype=torch.float32)
            best_rows = torch.full((len(batch), k), -1, device=self.device, dtype=torch.int64)
            for column in range(0, len(self.sources), self.document_block):
                stop = min(column + self.document_block, len(self.sources))
                docs = self.vectors[column:stop]
                if self.profile == "fp16" and self.device.type == "cuda":
                    # FP16 storage has a distinct contract; accumulators stay FP32.
                    scores = torch.mm(batch, docs.T, out_dtype=torch.float32)
                else:
                    scores = torch.mm(batch, docs.T).float()
                scores.masked_fill_(~gpu_mask[column:stop][None, :], -torch.inf)
                values, rows = torch.topk(scores, min(k, stop-column), dim=1)
                merged_values = torch.cat((best, values), dim=1)
                merged_rows = torch.cat((best_rows, rows + column), dim=1)
                best, chosen = torch.topk(merged_values, k, dim=1)
                best_rows = merged_rows.gather(1, chosen)
            result_ids = self.ids[best_rows.clamp_min(0)]
            result_ids.masked_fill_(~torch.isfinite(best), -1)
            if finish is not None:
                finish.record(torch.cuda.current_stream(self.device)); finish.synchronize()
                compute_ms += begin.elapsed_time(finish)
            else:
                compute_ms += (time.perf_counter()-compute_start)*1000
            d2h_start = time.perf_counter()
            output_ids[start:end] = result_ids.cpu().numpy()
            output_scores[start:end] = best.cpu().numpy()
            d2h_ms += (time.perf_counter()-d2h_start)*1000
        return output_ids, output_scores, {
            "wall_ms": (time.perf_counter()-started)*1000,
            "prepare_ms": prepare_ms, "h2d_submit_ms": h2d_ms,
            "compute_event_ms": compute_ms, "d2h_ms": d2h_ms,
            "query_count": len(q), "documents": len(self.sources), "profile": self.profile,
        }
