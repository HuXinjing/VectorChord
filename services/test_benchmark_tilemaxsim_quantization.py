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

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from services.benchmark_tilemaxsim_quantization import (
    Dataset,
    Variant,
    batches,
    build_variant,
    load_quantizer,
    save_quantizer,
    score_unfused_document_chunks,
    score_variant,
)
from services.tilemaxsim_quantization import train_residual_product_quantizer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class QuantizationBenchmarkIntegrationTest(unittest.TestCase):
    def test_batch_planner_streams_an_oversized_document_as_singleton(self) -> None:
        rows = np.asarray([2, 20, 3], dtype=np.int32)
        self.assertEqual(list(batches(rows, 4, 32)), [(0, 1), (1, 2), (2, 3)])

    def test_chunked_unfused_int8_matches_whole_document_maxsim(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.integers(-127, 128, size=(23, 8), dtype=np.int8)
        scales = rng.uniform(0.001, 0.02, size=23).astype("<f2")
        query = torch.from_numpy(rng.normal(size=(5, 8)).astype("<f2")).cuda()
        actual, _, _ = score_unfused_document_chunks(
            query,
            values,
            scales,
            None,
            Variant("int8-unfused", encoding="int8", fused=False),
            0,
            len(values),
            8 * (8 * 3 + 2),
            torch.device("cuda:0"),
        )
        reconstructed = torch.from_numpy(values).cuda().to(torch.float16) * (
            torch.from_numpy(scales).cuda().to(torch.float16)[:, None]
        )
        expected = (
            query.float().matmul(reconstructed.float().T).amax(dim=1).sum()
        )
        torch.testing.assert_close(actual, expected)

    def test_quantizer_cache_round_trip_has_no_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quantizer.npz"
            training = torch.randn(48, 8, generator=torch.Generator().manual_seed(4))
            expected = train_residual_product_quantizer(
                training,
                2,
                4,
                2,
                iterations=2,
                seed=5,
                opq_outer_iterations=1,
            )
            save_quantizer(path, expected)
            actual = load_quantizer(path)
            values = training[:7]
            torch.testing.assert_close(
                actual.decode(actual.encode(values)),
                expected.decode(expected.encode(values)),
            )

    def test_compression_build_reuses_matching_fp16_pooling_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_dataset(root)
            tensor_path = root / "one-row.f16"
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype="<f2").tofile(tensor_path)
            manifest = json.loads(manifest_path.read_text())
            page = {
                "page_key": "shared",
                "page_no": 1,
                "tensor_rows": 1,
                "tensor_dim": 4,
                "tensor_dtype": "float16",
                "source": {"kind": "file", "path": str(tensor_path)},
            }
            manifest["documents"] = {doc_id: [page] for doc_id in manifest["documents"]}
            manifest_path.write_text(json.dumps(manifest))
            dataset = Dataset(manifest_path)
            cache_root = root / "cache"
            try:
                exact = build_variant(
                    dataset,
                    Variant("exact-fp16"),
                    cache_root,
                    training_rows=8,
                    iterations=1,
                    seed=3,
                    device=torch.device("cuda:0"),
                )
                compressed = build_variant(
                    dataset,
                    Variant("int8", encoding="int8"),
                    cache_root,
                    training_rows=8,
                    iterations=1,
                    seed=3,
                    device=torch.device("cuda:0"),
                )
                document_mean = build_variant(
                    dataset,
                    Variant(
                        "document-mean-fp16",
                        pooling=0,
                        normalize_pooling=True,
                    ),
                    cache_root,
                    training_rows=8,
                    iterations=1,
                    seed=3,
                    device=torch.device("cuda:0"),
                )
            finally:
                dataset.close()
            metadata = json.loads((compressed / "metadata.json").read_text())
            self.assertEqual(metadata["source_fp16_cache"], str(exact.resolve()))
            self.assertEqual(np.load(compressed / "values.npy").shape, (1124, 4))
            self.assertTrue(np.all(np.load(document_mean / "rows.npy") == 1))

    def test_pq_build_streams_documents_without_changing_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_dataset(root)
            tensor_path = root / "many-rows.f16"
            source = np.random.default_rng(29).normal(size=(9000, 4)).astype("<f2")
            source.tofile(tensor_path)
            one_row_path = root / "one-row.f16"
            source[:1].tofile(one_row_path)
            manifest = json.loads(manifest_path.read_text())
            page = {
                "page_key": "shared",
                "page_no": 1,
                "tensor_rows": len(source),
                "tensor_dim": 4,
                "tensor_dtype": "float16",
                "source": {"kind": "file", "path": str(tensor_path)},
            }
            small_page = {
                **page,
                "page_key": "small",
                "tensor_rows": 1,
                "source": {"kind": "file", "path": str(one_row_path)},
            }
            document_ids = list(manifest["documents"])
            manifest["documents"] = {
                doc_id: [page if index == 0 else small_page]
                for index, doc_id in enumerate(document_ids)
            }
            manifest_path.write_text(json.dumps(manifest))
            dataset = Dataset(manifest_path)
            variant = Variant(
                "pq-streamed", encoding="pq", subspaces=2, centroids=4
            )
            try:
                cache = build_variant(
                    dataset,
                    variant,
                    root / "cache",
                    training_rows=16,
                    iterations=1,
                    seed=31,
                    device=torch.device("cuda:0"),
                )
                quantizer = load_quantizer(cache / "quantizer.npz")
                expected = quantizer.encode(torch.from_numpy(source).cuda()).cpu()
                rows = np.load(cache / "rows.npy")
                offsets = np.load(cache / "offsets.npy")
                large_index = int(rows.argmax())
                start = int(offsets[large_index])
                actual = np.load(cache / "codes.npy")[
                    start : start + len(source), 0
                ]
            finally:
                dataset.close()
            np.testing.assert_array_equal(actual, expected.numpy())

    def make_dataset(self, root: Path) -> Path:
        corpus = root / "corpus.jsonl"
        documents = [f"doc-{index:04d}" for index in range(1124)]
        corpus.write_text(
            "".join(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "canonical_path": f"{doc_id}.pdf",
                        "aliases": [],
                        "filenames": [f"{doc_id}.pdf"],
                    }
                )
                + "\n"
                for doc_id in documents
            )
        )
        qrels = {
            "queries": [
                {
                    "id": f"q-{index:03d}",
                    "query": f"synthetic query {index}",
                    "relevant": [documents[0]],
                }
                for index in range(81)
            ]
        }
        qrels_path = root / "qrels.json"
        qrels_path.write_text(json.dumps(qrels))
        query_path = root / "queries.npz"
        np.savez(
            query_path,
            **{
                item["id"]: np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype="<f2")
                for item in qrels["queries"]
            },
        )
        shard_root = root / "shards"
        shard_root.mkdir()
        (shard_root / "tilemaxsim-shards-v1.json").write_text(
            json.dumps({"entries": []})
        )
        manifest = {
            "version": 1,
            "dimension": 4,
            "corpus": str(corpus),
            "qrels": str(qrels_path),
            "query_tensors": str(query_path),
            "shard_root": str(shard_root),
            # Scoring consumes the already-built cache, so pages can be empty.
            "documents": {doc_id: [] for doc_id in documents},
            "coverage": {
                "corpus_documents": 1124,
                "documents": 1124,
                "queries": 81,
                "relevant_documents": 1,
                "relevant_documents_covered": 1,
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def write_cache(self, root: Path, variant: Variant) -> Path:
        cache = root / variant.name
        cache.mkdir()
        rows = np.ones(1124, dtype=np.int32)
        offsets = np.arange(1124, dtype=np.int64)
        values = np.full((1124, 4), -1.0, dtype="<f2")
        values[0] = [1.0, 0.0, 0.0, 0.0]
        np.save(cache / "rows.npy", rows)
        np.save(cache / "offsets.npy", offsets)
        np.save(cache / "values.npy", values)
        (cache / "doc-ids.json").write_text(
            json.dumps([f"doc-{index:04d}" for index in range(1124)])
        )
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "variant": {
                        "name": variant.name,
                        "pooling": 1,
                        "encoding": "fp16",
                        "subspaces": 0,
                        "centroids": 0,
                        "residual_stages": 1,
                        "opq_iterations": 0,
                    },
                    "artifact_bytes": values.nbytes,
                }
            )
        )
        return cache

    def write_int8_cache(self, root: Path) -> Path:
        cache = root / "int8"
        cache.mkdir()
        rows = np.ones(1124, dtype=np.int32)
        values = np.full((1124, 4), -127, dtype=np.int8)
        values[0] = [127, 0, 0, 0]
        np.save(cache / "rows.npy", rows)
        np.save(cache / "offsets.npy", np.arange(1124, dtype=np.int64))
        np.save(cache / "values.npy", values)
        np.save(cache / "scales.npy", np.full(1124, 1 / 127, dtype="<f2"))
        (cache / "doc-ids.json").write_text(
            json.dumps([f"doc-{index:04d}" for index in range(1124)])
        )
        (cache / "metadata.json").write_text(
            json.dumps({"complete": True, "artifact_bytes": values.nbytes})
        )
        return cache

    def test_full_1124_by_81_scoring_and_qrel_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = Dataset(self.make_dataset(root))
            variant = Variant("synthetic-exact")
            cache = self.write_cache(root, variant)
            try:
                report = score_variant(
                    dataset,
                    variant,
                    cache,
                    torch.device("cuda:0"),
                    16 * 1024**2,
                    warmups=1,
                )
            finally:
                dataset.close()
            self.assertEqual(len(report["cases"]), 81)
            self.assertEqual(report["quality"]["recall_at_1"], 1.0)
            self.assertEqual(report["quality"]["mrr"], 1.0)
            self.assertEqual(report["dataset"]["documents"], 1124)

    def test_fused_and_unfused_int8_have_identical_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = Dataset(self.make_dataset(root))
            cache = self.write_int8_cache(root)
            try:
                fused = score_variant(
                    dataset,
                    Variant("int8", encoding="int8"),
                    cache,
                    torch.device("cuda:0"),
                    16 * 1024**2,
                    warmups=1,
                )
                unfused = score_variant(
                    dataset,
                    Variant("int8-unfused", encoding="int8", fused=False),
                    cache,
                    torch.device("cuda:0"),
                    16 * 1024**2,
                    warmups=1,
                )
            finally:
                dataset.close()
            self.assertEqual(
                [item["top_20"] for item in fused["cases"]],
                [item["top_20"] for item in unfused["cases"]],
            )
