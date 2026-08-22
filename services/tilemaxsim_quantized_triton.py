# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Fused ragged quantized TileMaxSim kernels.

INT8 and FP8 payloads are dequantized in registers immediately before the
tensor-core dot; no FP16 document arena is materialized.  PQ-family payloads
use query-side inner-product lookup tables and accumulate ADC values directly
from codes.  OPQ uses the same ADC kernel after rotating the query, while
residual PQ supplies one LUT/code stage at a time and accumulates before max.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _ragged_scaled_tilemaxsim_kernel(
    query,
    documents,
    scales,
    document_offsets,
    scale_offsets,
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
    query_indices = query_block_index * block_query + tl.arange(0, block_query)
    query_mask = query_indices < query_rows
    document_base = tl.load(document_offsets + document_index).to(tl.int64)
    scale_base = tl.load(scale_offsets + document_index).to(tl.int64)
    valid_document_rows = tl.load(document_rows + document_index)
    running_max = tl.full([block_query], float("-inf"), tl.float32)

    for document_start in tl.range(0, max_document_rows, block_document):
        document_indices = document_start + tl.arange(0, block_document)
        document_mask = document_indices < valid_document_rows
        row_scales = tl.load(
            scales + scale_base + document_indices,
            mask=document_mask,
            other=0.0,
        ).to(tl.float32)
        similarities = tl.zeros([block_query, block_document], tl.float32)
        for dimension_start in range(0, dimension, block_dimension):
            dimensions = dimension_start + tl.arange(0, block_dimension)
            dimension_mask = dimensions < dimension
            query_tile = tl.load(
                query + query_indices[:, None] * dimension + dimensions[None, :],
                mask=query_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            quantized_tile = tl.load(
                documents
                + document_base
                + document_indices[:, None] * dimension
                + dimensions[None, :],
                mask=document_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            # Scaling after dot is valid because the scale is per document row.
            similarities += tl.dot(query_tile, tl.trans(quantized_tile))
        similarities *= row_scales[None, :]
        similarities = tl.where(document_mask[None, :], similarities, float("-inf"))
        running_max = tl.maximum(running_max, tl.max(similarities, axis=1))

    tl.atomic_add(
        scores + document_index,
        tl.sum(tl.where(query_mask, running_max, 0.0), axis=0),
    )


@triton.jit
def _ragged_adc_tilemaxsim_kernel(
    luts,
    codes,
    document_code_offsets,
    document_rows,
    scores,
    query_rows,
    stages: tl.constexpr,
    subspaces: tl.constexpr,
    centroids: tl.constexpr,
    max_document_rows: tl.constexpr,
    block_query: tl.constexpr,
    block_document: tl.constexpr,
):
    document_index = tl.program_id(0).to(tl.int64)
    query_block_index = tl.program_id(1).to(tl.int64)
    query_indices = query_block_index * block_query + tl.arange(0, block_query)
    query_mask = query_indices < query_rows
    document_base = tl.load(document_code_offsets + document_index).to(tl.int64)
    valid_document_rows = tl.load(document_rows + document_index)
    running_max = tl.full([block_query], float("-inf"), tl.float32)

    for document_start in tl.range(0, max_document_rows, block_document):
        document_indices = document_start + tl.arange(0, block_document)
        document_mask = document_indices < valid_document_rows
        similarities = tl.zeros([block_query, block_document], tl.float32)
        for stage in range(stages):
            for subspace in range(subspaces):
                code = tl.load(
                    codes
                    + document_base
                    + (document_indices * stages + stage) * subspaces
                    + subspace,
                    mask=document_mask,
                    other=0,
                ).to(tl.int64)
                value = tl.load(
                    luts
                    + ((query_indices[:, None] * stages + stage) * subspaces + subspace)
                    * centroids
                    + code[None, :],
                    mask=query_mask[:, None] & document_mask[None, :],
                    other=0.0,
                )
                similarities += value
        similarities = tl.where(document_mask[None, :], similarities, float("-inf"))
        running_max = tl.maximum(running_max, tl.max(similarities, axis=1))

    tl.atomic_add(
        scores + document_index,
        tl.sum(tl.where(query_mask, running_max, 0.0), axis=0),
    )


def _ragged_metadata(
    document_offsets: torch.Tensor,
    document_rows: torch.Tensor,
    device: torch.device,
) -> int:
    if document_offsets.device != device or document_rows.device != device:
        raise ValueError("ragged metadata must share the payload device")
    if document_offsets.dtype != torch.int64 or document_rows.dtype != torch.int32:
        raise ValueError("ragged offsets must be int64 and rows must be int32")
    if document_offsets.ndim != 1 or document_offsets.shape != document_rows.shape:
        raise ValueError("ragged metadata shapes disagree")
    return document_offsets.numel()


def ragged_scaled_tilemaxsim(
    query: torch.Tensor,
    document_arena: torch.Tensor,
    scale_arena: torch.Tensor,
    document_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    document_rows: torch.Tensor,
    maximum_document_rows: int,
) -> torch.Tensor:
    """Score INT8 or FP8 rows with per-row scales and no dequantized arena."""

    if query.device.type != "cuda" or document_arena.device != query.device:
        raise ValueError("query and document arena must share a CUDA device")
    if query.dtype != torch.float16:
        raise ValueError("fused scaled TileMaxSim requires an FP16 query")
    supported = {torch.int8}
    if hasattr(torch, "float8_e4m3fn"):
        supported.add(torch.float8_e4m3fn)
    if document_arena.dtype not in supported:
        raise ValueError("document arena must use INT8 or float8_e4m3fn")
    if document_arena.ndim != 1 or scale_arena.ndim != 1:
        raise ValueError("document and scale arenas must be flat")
    if scale_arena.dtype not in (torch.float16, torch.float32):
        raise ValueError("scale arena must use FP16 or FP32")
    count = _ragged_metadata(document_offsets, document_rows, query.device)
    if (
        scale_offsets.dtype != torch.int64
        or scale_offsets.shape != document_offsets.shape
    ):
        raise ValueError("scale offsets must match document offsets")
    if scale_offsets.device != query.device:
        raise ValueError("scale offsets must share the query device")
    if query.ndim != 2 or maximum_document_rows <= 0:
        raise ValueError("invalid query shape or maximum document rows")
    if count == 0:
        return torch.empty(0, dtype=torch.float32, device=query.device)
    block_query, block_document, block_dimension = 32, 32, 128
    padded_rows = triton.cdiv(maximum_document_rows, block_document) * block_document
    scores = torch.zeros(count, dtype=torch.float32, device=query.device)
    with torch.cuda.device(query.device):
        _ragged_scaled_tilemaxsim_kernel[
            (count, triton.cdiv(query.shape[0], block_query))
        ](
            query,
            document_arena,
            scale_arena,
            document_offsets,
            scale_offsets,
            document_rows,
            scores,
            query.shape[0],
            dimension=query.shape[1],
            max_document_rows=padded_rows,
            block_query=block_query,
            block_document=block_document,
            block_dimension=block_dimension,
        )
    return scores


def ragged_adc_tilemaxsim(
    luts: torch.Tensor | Sequence[torch.Tensor],
    code_arena: torch.Tensor,
    document_code_offsets: torch.Tensor,
    document_rows: torch.Tensor,
    maximum_document_rows: int,
) -> torch.Tensor:
    """Fused PQ/OPQ/RPQ ADC over a stage-major uint8 code arena.

    ``code_arena`` stores each row as ``[stage][subspace]``.  A single LUT is
    ordinary PQ or OPQ; multiple LUTs are residual-PQ stages.
    """

    if isinstance(luts, torch.Tensor):
        lut_sequence = (luts,)
    else:
        lut_sequence = tuple(luts)
    if not lut_sequence:
        raise ValueError("at least one ADC LUT is required")
    first = lut_sequence[0]
    if first.device.type != "cuda" or first.ndim != 3:
        raise ValueError("ADC LUTs must be [query, subspace, centroid] CUDA tensors")
    if any(
        item.shape != first.shape or item.device != first.device
        for item in lut_sequence
    ):
        raise ValueError("ADC LUT stages must have identical shape and device")
    if (
        code_arena.device != first.device
        or code_arena.dtype != torch.uint8
        or code_arena.ndim != 1
    ):
        raise ValueError("ADC codes must be a flat uint8 arena on the LUT device")
    count = _ragged_metadata(document_code_offsets, document_rows, first.device)
    if maximum_document_rows <= 0:
        raise ValueError("maximum document rows must be positive")
    if count == 0:
        return torch.empty(0, dtype=torch.float32, device=first.device)
    stages = len(lut_sequence)
    packed_luts = torch.stack(
        [item.float() for item in lut_sequence], dim=1
    ).contiguous()
    block_query, block_document = 16, 32
    padded_rows = triton.cdiv(maximum_document_rows, block_document) * block_document
    scores = torch.zeros(count, dtype=torch.float32, device=first.device)
    with torch.cuda.device(first.device):
        _ragged_adc_tilemaxsim_kernel[
            (count, triton.cdiv(first.shape[0], block_query))
        ](
            packed_luts,
            code_arena,
            document_code_offsets,
            document_rows,
            scores,
            first.shape[0],
            stages=stages,
            subspaces=first.shape[1],
            centroids=first.shape[2],
            max_document_rows=padded_rows,
            block_query=block_query,
            block_document=block_document,
        )
    return scores
