# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# Copyright (c) 2026 Hu Xinjing

from __future__ import annotations

import unittest

import torch

from services.tilemaxsim_quantization import (
    adc_tilemaxsim,
    exact_tilemaxsim,
    float8_tilemaxsim,
    int8_tilemaxsim,
    pool_tokens,
    quantize_float8,
    quantize_int8,
    sample_training_rows,
    train_opq,
    train_product_quantizer,
    train_residual_product_quantizer,
)

try:
    from services.tilemaxsim_quantized_triton import (
        ragged_adc_tilemaxsim,
        ragged_scaled_tilemaxsim,
    )
except ImportError:
    ragged_adc_tilemaxsim = None
    ragged_scaled_tilemaxsim = None


class TileMaxSimQuantizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = torch.Generator().manual_seed(711)
        self.training = torch.randn(96, 8, generator=self.generator)
        self.query = torch.randn(5, 8, generator=self.generator)
        self.document = torch.randn(13, 8, generator=self.generator)

    def test_pooling_preserves_prefix_and_uses_real_tail_count(self) -> None:
        tokens = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        pooled = pool_tokens(tokens, 3, preserve_prefix=1)
        expected = torch.stack((tokens[0], tokens[1:4].mean(0), tokens[4]))
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(pool_tokens(tokens, 1), tokens)

    def test_reservoir_sampling_is_bounded_and_deterministic(self) -> None:
        documents = [self.training[:40], self.training[40:]]
        first = sample_training_rows(documents, 17, seed=42)
        second = sample_training_rows(documents, 17, seed=42)
        self.assertEqual(first.shape, (17, 8))
        torch.testing.assert_close(first, second)

    def test_pq_adc_equals_explicit_reconstruction_score(self) -> None:
        quantizer = train_product_quantizer(self.training, 4, 8, iterations=4, seed=9)
        codes = quantizer.encode(self.document)
        expected = exact_tilemaxsim(self.query, quantizer.decode(codes))
        actual = adc_tilemaxsim(self.query, codes, quantizer)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_residual_pq_adc_equals_explicit_reconstruction_score(self) -> None:
        quantizer = train_residual_product_quantizer(
            self.training, 4, 8, 2, iterations=4, seed=13
        )
        codes = quantizer.encode(self.document)
        expected = exact_tilemaxsim(self.query, quantizer.decode(codes))
        actual = adc_tilemaxsim(self.query, codes, quantizer)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_opq_rotation_is_orthogonal_and_adc_is_consistent(self) -> None:
        quantizer = train_opq(
            self.training,
            4,
            8,
            outer_iterations=2,
            kmeans_iterations=3,
            seed=17,
        )
        assert quantizer.rotation is not None
        identity = torch.eye(8)
        torch.testing.assert_close(
            quantizer.rotation.T @ quantizer.rotation,
            identity,
            atol=2e-5,
            rtol=2e-5,
        )
        codes = quantizer.encode(self.document)
        expected = exact_tilemaxsim(self.query, quantizer.decode(codes))
        actual = adc_tilemaxsim(self.query, codes, quantizer)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_int8_zero_rows_and_fused_reference(self) -> None:
        document = self.document.clone()
        document[0].zero_()
        quantized = quantize_int8(document)
        self.assertTrue(torch.isfinite(quantized.scales).all())
        expected = exact_tilemaxsim(self.query, quantized.dequantize())
        actual = int8_tilemaxsim(self.query, quantized)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch has no FP8")
    def test_fp8_reference_uses_per_token_scales(self) -> None:
        quantized = quantize_float8(self.document)
        expected = exact_tilemaxsim(self.query, quantized.dequantize())
        actual = float8_tilemaxsim(self.query, quantized)
        torch.testing.assert_close(actual, expected)

    def test_invalid_dimensions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            train_product_quantizer(self.training, 3, 8)
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            exact_tilemaxsim(self.query, self.document[:, :-1])
        with self.assertRaisesRegex(ValueError, "factor"):
            pool_tokens(self.document, 0)


@unittest.skipUnless(
    torch.cuda.is_available() and ragged_scaled_tilemaxsim is not None,
    "CUDA Triton is unavailable",
)
class QuantizedTritonTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(812)
        self.query = torch.randn(7, 8, generator=generator, dtype=torch.float16)
        self.documents = (
            torch.randn(11, 8, generator=generator),
            torch.randn(5, 8, generator=generator),
        )
        self.rows = torch.tensor([11, 5], dtype=torch.int32, device="cuda")

    def test_fused_int8_matches_reference_without_dequantized_arena(self) -> None:
        quantized = tuple(quantize_int8(item) for item in self.documents)
        values = torch.cat([item.values.flatten() for item in quantized]).cuda()
        scales = torch.cat([item.scales for item in quantized]).cuda()
        offsets = torch.tensor([0, 11 * 8], dtype=torch.int64, device="cuda")
        scale_offsets = torch.tensor([0, 11], dtype=torch.int64, device="cuda")
        actual = ragged_scaled_tilemaxsim(
            self.query.cuda(), values, scales, offsets, scale_offsets, self.rows, 11
        )
        expected = torch.stack(
            [int8_tilemaxsim(self.query, item) for item in quantized]
        ).cuda()
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-3)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch has no FP8")
    def test_fused_fp8_matches_reference_without_dequantized_arena(self) -> None:
        quantized = tuple(quantize_float8(item) for item in self.documents)
        values = torch.cat([item.values.flatten() for item in quantized]).cuda()
        scales = torch.cat([item.scales for item in quantized]).cuda()
        offsets = torch.tensor([0, 11 * 8], dtype=torch.int64, device="cuda")
        scale_offsets = torch.tensor([0, 11], dtype=torch.int64, device="cuda")
        actual = ragged_scaled_tilemaxsim(
            self.query.cuda(), values, scales, offsets, scale_offsets, self.rows, 11
        )
        expected = torch.stack(
            [float8_tilemaxsim(self.query, item) for item in quantized]
        ).cuda()
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-3)

    def test_fused_adc_matches_reference_for_residual_pq(self) -> None:
        training = torch.cat(self.documents)
        quantizer = train_residual_product_quantizer(
            training, 2, 8, 2, iterations=3, seed=19
        )
        encoded = [quantizer.encode(item) for item in self.documents]
        # Convert stage-major tuples to row-major [row, stage, subspace].
        packed = (
            torch.cat([torch.stack(item, dim=1) for item in encoded]).flatten().cuda()
        )
        offsets = torch.tensor([0, 11 * 2 * 2], dtype=torch.int64, device="cuda")
        luts = [stage.adc_lut(self.query).cuda() for stage in quantizer.stages]
        actual = ragged_adc_tilemaxsim(luts, packed, offsets, self.rows, 11)
        expected = torch.stack(
            [adc_tilemaxsim(self.query, codes, quantizer) for codes in encoded]
        ).cuda()
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    def test_gpu_pq_training_and_chunked_encoding(self) -> None:
        training = torch.cat(self.documents).cuda()
        quantizer = train_product_quantizer(training, 2, 8, iterations=2, seed=23)
        repeated = train_product_quantizer(training, 2, 8, iterations=2, seed=23)
        codes = quantizer.encode(training, batch_rows=3)
        self.assertEqual(codes.shape, (16, 2))
        self.assertEqual(codes.device.type, "cuda")
        self.assertTrue(torch.equal(quantizer.codebooks, repeated.codebooks))


if __name__ == "__main__":
    unittest.main()
