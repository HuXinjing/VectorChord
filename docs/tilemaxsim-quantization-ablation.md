# TileMaxSim compression and fusion ablation

This research suite measures compression and kernel execution changes in visual
TileMaxSim. It deliberately excludes FTS, graph traversal, Fact Store signals,
query expansion, reranking, and application-specific boosts.

## Frozen evaluation contract

- Corpus: 1,124 deduplicated PDFs.
- Queries: 81 strict questions.
- Ground truth: the strict qrels paired with those 81 questions.
- Representation: ColQwen page tensors, logically concatenated by document.
- Primary quality metric: macro document Recall@1/5/10. For questions with
  multiple relevant documents, partial retrieval receives partial credit.
- Secondary metrics: Hit@1/5/10, MRR, and top-k overlap with exact FP16
  TileMaxSim.
- Performance: query preparation, host-to-device transfer, fused kernel, and
  end-to-end retrieval latency are recorded separately after one warmup query.

Dataset preparation fails closed unless all 1,124 document IDs, all 81 query
tensors, and every qrel document have visual tensors. Page token counts are
stored per page and are not assumed to be a fixed value.

## Ablation dimensions

The default matrix includes:

- Exact FP16, consecutive mean pooling factors 2 and 4 with and without
  post-pooling L2 normalization, and normalized whole-document mean pooling.
- Per-token symmetric INT8 and scaled FP8 E4M3.
- PQ with 8, 16, and 32 subspaces and 4- or 8-bit codes.
- Two- and three-stage residual PQ.
- OPQ followed by PQ.
- Pooling combined independently with scalar quantization, PQ, residual PQ,
  and OPQ.
- The compatible full stacks: raw or normalized pooling + OPQ first stage +
  residual PQ + fused ADC.
- Fused/unfused execution pairs for INT8, FP8, PQ, and residual PQ. Unfused
  variants materialize a full FP16 document arena before exact TileMaxSim;
  fused variants dequantize in registers or score ADC lookup tables directly.

INT8, FP8, and PQ are alternative document encodings, so they are not stacked
on the same token payload. Pooling and OPQ/residual stages are orthogonal and
are combined where mathematically valid.

## Reproduction

Prepare a complete execution manifest, reusing any compatible historical page
tensors and writing only missing tensors to a new output directory:

```bash
python -m services.prepare_tilemaxsim_quantization_dataset \
  --corpus "$CORPUS_JSONL" \
  --qrels "$QRELS_JSON" \
  --page-text "$PAGE_TEXT_JSONL" \
  --descriptors "$DESCRIPTORS_JSONL" \
  --shard-root "$SHARD_ROOT" \
  --pdf-root "$PDF_ROOT" \
  --supplement "$COMPATIBLE_EXECUTION_MANIFEST" \
  --pooling-url "$COLQWEN_POOLING_URL" \
  --output "$DATASET_OUTPUT"
```

Build and run the full matrix:

```bash
python -m services.benchmark_tilemaxsim_quantization \
  --manifest "$DATASET_OUTPUT/execution-manifest.json" \
  --cache-root "$CACHE_ROOT" \
  --report-root "$REPORT_ROOT" \
  --device cuda:0 \
  --gpu-batch-gb 0.5
```

Individual configurations can be selected with repeated `--variant`; list the
registered matrix with `--list-variants`. Artifacts are resumable and use
non-pickle NPZ codebook files.

Generate JSON and Markdown comparisons after all runs:

```bash
python -m services.summarize_tilemaxsim_quantization \
  --report-root "$REPORT_ROOT" \
  --baseline exact-fp16 \
  --output-json "$REPORT_ROOT/summary.json" \
  --output-markdown "$REPORT_ROOT/summary.md"
```
