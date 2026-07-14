# Copyright (c) 2026 HuXinjing

"""Fused ragged FP16 TileMaxSim over a process-owned GPU tensor arena."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ragged_tilemaxsim_fp16_kernel(
    query,
    documents,
    document_offsets,
    document_rows,
    scores,
    query_rows,
    dimension: tl.constexpr,
    max_document_rows: tl.constexpr,
    block_query: tl.constexpr,
    block_document: tl.constexpr,
    block_dimension: tl.constexpr,
):
    document_index = tl.program_id(0).to(tl.int64)
    query_block_index = tl.program_id(1).to(tl.int64)
    query_indices = (query_block_index * block_query + tl.arange(0, block_query)).to(
        tl.int64
    )
    query_mask = query_indices < query_rows
    document_base = tl.load(document_offsets + document_index).to(tl.int64)
    valid_document_rows = tl.load(document_rows + document_index)
    running_max = tl.full([block_query], value=float("-inf"), dtype=tl.float32)

    for document_start in range(0, max_document_rows, block_document):
        document_indices = (document_start + tl.arange(0, block_document)).to(tl.int64)
        document_mask = document_indices < valid_document_rows
        similarities = tl.zeros([block_query, block_document], dtype=tl.float32)
        for dimension_start in range(0, dimension, block_dimension):
            dimension_indices = (dimension_start + tl.arange(0, block_dimension)).to(
                tl.int64
            )
            dimension_mask = dimension_indices < dimension
            query_pointers = (
                query + query_indices[:, None] * dimension + dimension_indices[None, :]
            )
            query_tile = tl.load(
                query_pointers,
                mask=query_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            document_pointers = (
                documents
                + document_base
                + document_indices[:, None] * dimension
                + dimension_indices[None, :]
            )
            document_tile = tl.load(
                document_pointers,
                mask=document_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            similarities += tl.dot(query_tile, tl.trans(document_tile))
        similarities = tl.where(document_mask[None, :], similarities, float("-inf"))
        running_max = tl.maximum(running_max, tl.max(similarities, axis=1))

    running_max = tl.where(query_mask, running_max, 0.0)
    tl.atomic_add(scores + document_index, tl.sum(running_max, axis=0))


def ragged_tilemaxsim_fp16(
    query: torch.Tensor,
    document_arena: torch.Tensor,
    document_offsets: torch.Tensor,
    document_rows: torch.Tensor,
    maximum_document_rows: int,
) -> torch.Tensor:
    if query.device.type != "cuda" or document_arena.device != query.device:
        raise ValueError("query and document arena must be on the same CUDA device")
    if query.dtype != torch.float16 or document_arena.dtype != torch.float16:
        raise ValueError("ragged TileMaxSim currently requires FP16 tensors")
    if query.ndim != 2 or document_arena.ndim != 1:
        raise ValueError("invalid query or document arena shape")
    if document_offsets.dtype != torch.int64 or document_rows.dtype != torch.int32:
        raise ValueError("invalid ragged TileMaxSim metadata dtype")
    if document_offsets.shape != document_rows.shape:
        raise ValueError("document offsets and rows must have the same shape")
    count = document_offsets.numel()
    if count == 0:
        return torch.empty(0, dtype=torch.float32, device=query.device)
    if maximum_document_rows <= 0:
        raise ValueError("maximum document rows must be positive")
    block_query = 32
    block_document = 32
    block_dimension = 128
    padded_document_rows = (
        (maximum_document_rows + block_document - 1) // block_document * block_document
    )
    query_blocks = (query.shape[0] + block_query - 1) // block_query
    scores = torch.zeros(count, dtype=torch.float32, device=query.device)
    with torch.cuda.device(query.device):
        _ragged_tilemaxsim_fp16_kernel[(count, query_blocks)](
            query,
            document_arena,
            document_offsets,
            document_rows,
            scores,
            query.shape[0],
            dimension=query.shape[1],
            max_document_rows=padded_document_rows,
            block_query=block_query,
            block_document=block_document,
            block_dimension=block_dimension,
        )
    return scores
