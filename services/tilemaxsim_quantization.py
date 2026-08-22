# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

"""Trainable compression formats and reference scorers for TileMaxSim.

The implementations in this module deliberately keep encoding independent of
the benchmark and of the GPU kernels.  This gives every fused implementation a
small, deterministic oracle and makes combinations such as pooling + OPQ +
residual PQ explicit rather than hiding them in an experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def exact_tilemaxsim(query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
    """Return the FP32 MaxSim sum used as the quality oracle."""

    query = _matrix(query, "query")
    document = _matrix(document, "document")
    if query.shape[1] != document.shape[1]:
        raise ValueError("query and document dimensions disagree")
    return (query.float() @ document.float().T).amax(dim=1).sum()


def pool_tokens(
    tokens: torch.Tensor,
    factor: int,
    *,
    preserve_prefix: int = 0,
    normalize: bool = False,
) -> torch.Tensor:
    """Pool consecutive token groups without using application semantics.

    Consecutive pooling is deterministic, has no benchmark-derived vocabulary,
    and retains an optional model-defined prefix (for example special tokens).
    The final short group is averaged over its real members, never zero padding.
    """

    tokens = _matrix(tokens, "tokens")
    if factor <= 0:
        raise ValueError("pooling factor must be positive")
    if not 0 <= preserve_prefix <= tokens.shape[0]:
        raise ValueError("preserve_prefix is outside the token range")
    if factor == 1 or preserve_prefix == tokens.shape[0]:
        return tokens.clone()
    prefix = tokens[:preserve_prefix]
    body = tokens[preserve_prefix:]
    pooled = [part.mean(dim=0) for part in body.split(factor)]
    result = torch.cat((prefix, torch.stack(pooled)), dim=0) if pooled else prefix
    if normalize:
        result = torch.nn.functional.normalize(result.float(), dim=1).to(tokens.dtype)
    return result


def sample_training_rows(
    documents: Iterable[torch.Tensor], maximum_rows: int, seed: int
) -> torch.Tensor:
    """Uniformly sample tokens from a stream using bounded random priorities.

    Assigning every row an independent continuous random key and retaining the
    largest ``maximum_rows`` keys is equivalent to reservoir sampling, while
    allowing each incoming tensor to be processed as a vectorized batch.
    """

    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reservoir: torch.Tensor | None = None
    priorities: torch.Tensor | None = None
    dimension: int | None = None
    for document in documents:
        document = _matrix(document, "document").detach().float().cpu()
        if dimension is None:
            dimension = document.shape[1]
            reservoir = torch.empty((0, dimension), dtype=torch.float32)
            priorities = torch.empty(0, dtype=torch.float64)
        elif document.shape[1] != dimension:
            raise ValueError("training documents have inconsistent dimensions")
        assert reservoir is not None and priorities is not None
        incoming_priorities = torch.rand(
            document.shape[0], dtype=torch.float64, generator=generator
        )
        candidates = torch.cat((reservoir, document), dim=0)
        candidate_priorities = torch.cat((priorities, incoming_priorities), dim=0)
        keep = min(maximum_rows, candidates.shape[0])
        selected = torch.topk(candidate_priorities, keep, sorted=False).indices
        reservoir = candidates[selected]
        priorities = candidate_priorities[selected]
    if reservoir is None or reservoir.shape[0] == 0:
        raise ValueError("training documents are empty")
    return reservoir.clone()


def _kmeans(
    values: torch.Tensor, clusters: int, iterations: int, generator: torch.Generator
) -> torch.Tensor:
    values = _matrix(values, "training values").float()
    if not 1 <= clusters <= values.shape[0]:
        raise ValueError("clusters must not exceed training rows")
    if iterations <= 0:
        raise ValueError("k-means iterations must be positive")
    first = int(
        torch.randint(values.shape[0], (), generator=generator, device=values.device)
    )
    centers = [values[first]]
    nearest = torch.sum((values - centers[0]) ** 2, dim=1)
    for _ in range(1, clusters):
        total = nearest.sum()
        if total <= 0:
            candidate = int(
                torch.randint(
                    values.shape[0], (), generator=generator, device=values.device
                )
            )
        else:
            candidate = int(torch.multinomial(nearest / total, 1, generator=generator))
        centers.append(values[candidate])
        distance = torch.sum((values - centers[-1]) ** 2, dim=1)
        nearest = torch.minimum(nearest, distance)
    centroids = torch.stack(centers)
    for _ in range(iterations):
        assignments = torch.cdist(values, centroids).argmin(dim=1)
        counts = torch.bincount(assignments, minlength=clusters)
        # CUDA index_add uses atomic floating-point accumulation and changes a
        # few low bits across runs.  The one-hot GEMM has a fixed reduction
        # schedule for a fixed environment, making seeded benchmark codebooks
        # reproducible without moving training back to the CPU.
        membership = torch.nn.functional.one_hot(assignments, num_classes=clusters).to(
            values.dtype
        )
        sums = membership.T @ values
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        if not nonempty.all():
            # Re-seed an empty cell with a high-error point.  This is a general
            # k-means invariant and does not depend on evaluation examples.
            error = torch.sum((values - centroids[assignments]) ** 2, dim=1)
            replacements = error.argsort(descending=True)
            for empty, replacement in zip(
                (~nonempty).nonzero().flatten(), replacements, strict=False
            ):
                centroids[empty] = values[replacement]
    return centroids


@dataclass(frozen=True)
class ProductQuantizer:
    """Product codebooks, optionally preceded by an orthogonal rotation."""

    codebooks: torch.Tensor  # [subspaces, centroids, subdimension]
    rotation: torch.Tensor | None = None  # row-vector convention: x @ rotation

    @property
    def subspaces(self) -> int:
        return self.codebooks.shape[0]

    @property
    def centroids(self) -> int:
        return self.codebooks.shape[1]

    @property
    def dimension(self) -> int:
        return self.codebooks.shape[0] * self.codebooks.shape[2]

    def rotate(self, values: torch.Tensor) -> torch.Tensor:
        values = _matrix(values, "values").float()
        if values.shape[1] != self.dimension:
            raise ValueError("values and quantizer dimensions disagree")
        if self.rotation is None:
            return values
        return values @ self.rotation.to(values.device)

    def encode(self, values: torch.Tensor, *, batch_rows: int = 8192) -> torch.Tensor:
        if batch_rows <= 0:
            raise ValueError("PQ encoding batch_rows must be positive")
        rotated = self.rotate(values)
        books = self.codebooks.to(rotated.device)
        dtype = torch.uint8 if self.centroids <= 256 else torch.int16
        book_norms = torch.sum(books * books, dim=2)
        encoded = []
        for batch in rotated.split(batch_rows):
            chunks = batch.reshape(-1, self.subspaces, self.codebooks.shape[2])
            distances = (
                torch.sum(chunks * chunks, dim=2)[:, :, None]
                + book_norms[None, :, :]
                - 2 * torch.einsum("nmd,mkd->nmk", chunks, books)
            )
            encoded.append(distances.argmin(dim=2).to(dtype))
        return torch.cat(encoded, dim=0)

    def decode(
        self, codes: torch.Tensor, *, undo_rotation: bool = True
    ) -> torch.Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.subspaces:
            raise ValueError("invalid PQ code shape")
        books = self.codebooks.to(codes.device)
        pieces = [
            books[index][codes[:, index].long()] for index in range(self.subspaces)
        ]
        decoded = torch.cat(pieces, dim=1)
        if undo_rotation and self.rotation is not None:
            decoded = decoded @ self.rotation.to(decoded.device).T
        return decoded

    def adc_lut(self, query: torch.Tensor) -> torch.Tensor:
        rotated = self.rotate(query)
        chunks = rotated.reshape(-1, self.subspaces, self.codebooks.shape[2])
        return torch.einsum("qmd,mkd->qmk", chunks, self.codebooks.to(rotated.device))


def train_product_quantizer(
    training: torch.Tensor,
    subspaces: int,
    centroids: int,
    *,
    iterations: int = 20,
    seed: int = 0,
    rotation: torch.Tensor | None = None,
) -> ProductQuantizer:
    training = _matrix(training, "training").float()
    if training.shape[1] % subspaces:
        raise ValueError("dimension must be divisible by subspaces")
    if rotation is not None:
        if rotation.shape != (training.shape[1], training.shape[1]):
            raise ValueError("rotation has the wrong shape")
        training = training @ rotation.float().to(training.device)
    generator = torch.Generator(device=training.device).manual_seed(seed)
    chunks = training.reshape(training.shape[0], subspaces, -1)
    books = [
        _kmeans(chunks[:, index], centroids, iterations, generator)
        for index in range(subspaces)
    ]
    return ProductQuantizer(torch.stack(books), rotation)


def train_opq(
    training: torch.Tensor,
    subspaces: int,
    centroids: int,
    *,
    outer_iterations: int = 4,
    kmeans_iterations: int = 12,
    seed: int = 0,
) -> ProductQuantizer:
    """Train OPQ by alternating PQ assignment and orthogonal Procrustes."""

    training = _matrix(training, "training").float()
    if outer_iterations <= 0:
        raise ValueError("OPQ iterations must be positive")
    rotation = torch.eye(training.shape[1], device=training.device)
    quantizer: ProductQuantizer | None = None
    for iteration in range(outer_iterations):
        quantizer = train_product_quantizer(
            training,
            subspaces,
            centroids,
            iterations=kmeans_iterations,
            seed=seed + iteration,
            rotation=rotation,
        )
        codes = quantizer.encode(training)
        reconstructed_rotated = quantizer.decode(codes, undo_rotation=False)
        # min_R ||X R - Y|| with R orthogonal: R = U V^T for X^T Y = U S V^T.
        left, _, right = torch.linalg.svd(training.T @ reconstructed_rotated)
        rotation = left @ right
    assert quantizer is not None
    return train_product_quantizer(
        training,
        subspaces,
        centroids,
        iterations=kmeans_iterations,
        seed=seed + outer_iterations,
        rotation=rotation,
    )


@dataclass(frozen=True)
class ResidualProductQuantizer:
    stages: tuple[ProductQuantizer, ...]

    @property
    def dimension(self) -> int:
        return self.stages[0].dimension

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        residual = _matrix(values, "values").float()
        codes = []
        for stage in self.stages:
            stage_codes = stage.encode(residual)
            codes.append(stage_codes)
            residual = residual - stage.decode(stage_codes)
        return tuple(codes)

    def decode(self, codes: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(codes) != len(self.stages):
            raise ValueError("residual PQ stage count disagrees")
        return sum(
            (stage.decode(code) for stage, code in zip(self.stages, codes, strict=True))
        )


def train_residual_product_quantizer(
    training: torch.Tensor,
    subspaces: int,
    centroids: int,
    stages: int,
    *,
    iterations: int = 20,
    seed: int = 0,
    opq_outer_iterations: int = 0,
) -> ResidualProductQuantizer:
    training = _matrix(training, "training").float()
    if stages <= 0:
        raise ValueError("residual PQ stages must be positive")
    residual = training
    trained = []
    for stage_index in range(stages):
        if stage_index == 0 and opq_outer_iterations:
            quantizer = train_opq(
                residual,
                subspaces,
                centroids,
                outer_iterations=opq_outer_iterations,
                kmeans_iterations=iterations,
                seed=seed,
            )
        else:
            quantizer = train_product_quantizer(
                residual,
                subspaces,
                centroids,
                iterations=iterations,
                seed=seed + stage_index,
            )
        codes = quantizer.encode(residual)
        residual = residual - quantizer.decode(codes)
        trained.append(quantizer)
    return ResidualProductQuantizer(tuple(trained))


def adc_tilemaxsim(
    query: torch.Tensor,
    codes: torch.Tensor | Sequence[torch.Tensor],
    quantizer: ProductQuantizer | ResidualProductQuantizer,
) -> torch.Tensor:
    """Exact asymmetric-distance computation over codes, without reconstruction."""

    if isinstance(quantizer, ProductQuantizer):
        code_stages = (codes,) if isinstance(codes, torch.Tensor) else tuple(codes)
        quantizers = (quantizer,)
    else:
        if isinstance(codes, torch.Tensor):
            raise ValueError("residual PQ requires one code tensor per stage")
        code_stages = tuple(codes)
        quantizers = quantizer.stages
    if len(code_stages) != len(quantizers):
        raise ValueError("code and quantizer stage counts disagree")
    similarities: torch.Tensor | None = None
    for stage_codes, stage in zip(code_stages, quantizers, strict=True):
        if stage_codes.ndim != 2 or stage_codes.shape[1] != stage.subspaces:
            raise ValueError("invalid PQ code shape")
        lut = stage.adc_lut(query)
        stage_similarity = torch.zeros(
            (query.shape[0], stage_codes.shape[0]),
            dtype=lut.dtype,
            device=lut.device,
        )
        stage_codes = stage_codes.to(lut.device)
        for subspace in range(stage.subspaces):
            stage_similarity += lut[:, subspace][:, stage_codes[:, subspace].long()]
        similarities = (
            stage_similarity
            if similarities is None
            else similarities + stage_similarity
        )
    assert similarities is not None
    return similarities.amax(dim=1).sum()


@dataclass(frozen=True)
class Int8Tokens:
    values: torch.Tensor
    scales: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.values.float() * self.scales.float()[:, None]


def quantize_int8(values: torch.Tensor) -> Int8Tokens:
    values = _matrix(values, "values").float()
    maximum = values.abs().amax(dim=1)
    scales = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = torch.clamp(torch.round(values / scales[:, None]), -127, 127).to(
        torch.int8
    )
    return Int8Tokens(quantized, scales.to(torch.float16))


def int8_tilemaxsim(query: torch.Tensor, document: Int8Tokens) -> torch.Tensor:
    query = _matrix(query, "query").float()
    if query.shape[1] != document.values.shape[1]:
        raise ValueError("query and INT8 document dimensions disagree")
    integer_dot = query @ document.values.to(query.device).float().T
    return (
        (integer_dot * document.scales.to(query.device).float()[None, :])
        .amax(dim=1)
        .sum()
    )


@dataclass(frozen=True)
class Float8Tokens:
    values: torch.Tensor
    scales: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.values.float() * self.scales.float()[:, None]


def quantize_float8(values: torch.Tensor) -> Float8Tokens:
    values = _matrix(values, "values").float()
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build has no float8_e4m3fn dtype")
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    maximum = values.abs().amax(dim=1)
    scales = torch.where(maximum > 0, maximum / fp8_max, torch.ones_like(maximum))
    quantized = (values / scales[:, None]).to(torch.float8_e4m3fn)
    return Float8Tokens(quantized, scales.to(torch.float16))


def float8_tilemaxsim(query: torch.Tensor, document: Float8Tokens) -> torch.Tensor:
    return exact_tilemaxsim(query, document.dequantize().to(query.device))
