<div align="center">

# VectorChord TileMaxSim

**An open-source VectorChord fork focused on exact multi-vector retrieval in PostgreSQL.**

</div>

This fork extends VectorChord's `vchordrq` index with exact late-interaction
TileMaxSim retrieval. It is intended for applications that store one array of
token vectors per document and need PostgreSQL-native multi-vector search.

## What this fork adds

- Exact TileMaxSim reranking on CPU, plus an optional CUDA sidecar backend.
- Full caller-scoped tensor sets without an artificial candidate-count cap;
  device-memory pressure is handled by the GPU page cache and request batching.
- External tensor-source registration for deployments that keep full-precision
  token tensors outside the indexed PostgreSQL value.
- PostgreSQL-aware permission, MVCC, row-visibility, cancellation, and timeout
  handling for the external-tensor search path.
- Planner statistics and cost estimation for multi-vector queries.
- Deterministic correctness, registry, sidecar-protocol, and planner-cost tests.

## Performance ablation

We measured each cache-path optimization independently on the same development
machine. The corpus contained 34,054 tensor descriptors (34,027 unique tensors,
16.28 GB of logical FP16 tensor data). The request-level tests sampled 100
candidates containing 47.81 MB of tensor data. Absolute latency depends on the
storage and GPU, so the same-run comparisons are more useful than the raw
numbers.

| Optimization | Baseline | Optimized | Result |
| --- | ---: | ---: | ---: |
| Immutable shards and batched reads | 333.38 ms, sequential files | 56.52 ms | 5.90x faster |
| Shards versus batched legacy files | 87.56 ms | 56.52 ms | 1.55x faster |
| Batched host-to-device transfer | 37.06 ms, 100 transfers | 14.19 ms, one transfer | 2.61x faster |
| TinyLFU/GDSF admission | 69.98% LRU hit rate | 76.18% hit rate | +6.20 percentage points |
| Rust/CUDA cold request | 855.34 ms, Python/Triton | 93.86 ms | 9.11x faster |
| Rust/CUDA warm request p50 | 14.26 ms, Python/Triton | 2.09 ms | 6.82x faster |
| Rust/CUDA warm request p95 | 14.84 ms, Python/Triton | 2.20 ms | 6.75x faster |

A full resident-cache run assigned 20 GiB to one GPU: 18 GiB for tensors and
2 GiB for the TileMaxSim workspace. All 34,027 unique tensors were pinned before
the service became ready. Process-to-ready time was 23.07 seconds for the
Python/Triton sidecar and 14.86 seconds for the Rust/CUDA daemon, a 35.6%
reduction. This is a one-time prewarm cost; resident warm requests do not read
the tensors from disk.

The GPU cache now suballocates exact contiguous page runs from one CUDA arena,
using best-fit size buckets and address-ordered coalescing. On the full corpus,
the 32 KiB default reduced allocated tensor space from 17.840 GB with the former
256 KiB power-of-two buddy allocator to 16.725 GB. It recovered 1.115 GB of GPU
space and reduced internal rounding waste from 8.82% to 2.74%. The default can
be overridden with `--gpu-block-kib`.

In a deterministic 20,000-event churn trace, the former buddy allocator had 815
failed cache-allocation attempts, while the segregated page-run allocator had
637; neither recorded an external-fragmentation failure on this workload. Exact
byte extents had 585 failures, all caused by external fragmentation. Page-run
metadata processing added about 1.4 microseconds per event versus the buddy
baseline. These are cache-admission attempts, not failed search requests: the
runtime evicts unpinned entries or streams oversized working sets in chunks.

Rust/CUDA and Python/Triton produced identical top-10 results. The maximum
absolute score difference was 5.25e-6 and the mean difference was 3.45e-6.
The benchmark driver is
[`services/benchmark_tilemaxsim_ablation.py`](services/benchmark_tilemaxsim_ablation.py);
run it with `--help` for the corpus, cache-root, device, and output arguments.

### Full-source retrieval and cache-scheduler stress

We separately scored all 34,054 descriptors for each query to exercise a
working set larger than the cache. This is both a storage/cache stress test and
an upper bound for general semantic retrieval when no safe narrower hard scope
exists. GBrain still applies source, ACL, type, and other mandatory filters, but
lexical or graph recall must not hide semantic paraphrases. The traditional
embedding/HNSW path remains a separate `single_vector` mode and is not a
dependency of tensor retrieval.

With an 18 GiB tensor arena, all 34,027 unique tensors were resident and native
daemon round trips averaged 1.08 seconds for a deliberately exhaustive scan. A
3 GiB tensor arena plus 1 GiB host cache produced the same exact top-K, but two
sequential scans caused 68,106 misses, 61,499 evictions, 32.53 GB of host-to-
device transfer, and only two cache hits. Native round trips averaged 18.47
seconds. The slowdown is cyclic cache thrashing and repeated I/O, not TileMaxSim
compute scaling linearly with GPU-memory capacity. Candidate-scoped queries
whose hot working set fits the cache do not exhibit this full-scan behavior.

The diagnostic driver is
[`services/benchmark_full_corpus_tilemaxsim.py`](services/benchmark_full_corpus_tilemaxsim.py).
A GBrain-scoped comparison against its original text-embedding/pgvector HNSW
path is reported separately; tensor-derived pooled vectors are not a valid
single-vector baseline.

### Rejected lexical hard-gate ablation and the traditional embedding baseline

We then used 40 real queries and their original 1,024-dimensional
`text-embedding-v4` vectors. The single-vector baseline searched 1,200 stored
chunk embeddings with PostgreSQL pgvector HNSW (`m=16`, `ef_search=40`). Tensor
mode did not read those vectors: a real lexical result list supplied every
returned chunk without a second truncation, the chunks were mapped to their
source-page tensors, and the native CUDA daemon ran exact TileMaxSim only over
that scope. This ablation tests whether lexical recall is safe as a mandatory
gate; it is not the accepted general semantic query plan.

| Path | Mean latency | p95 | Quality |
| --- | ---: | ---: | ---: |
| Original text embedding + pgvector HNSW | 5.80 ms | 5.86 ms | document hit@1 1.000 |
| Candidate-scoped TileMaxSim, 18 GiB resident tensor cache, native round trips | 45.56 ms | 134.99 ms | document hit@1 0.475; hit@5 0.650 |
| Candidate-scoped TileMaxSim, resident Python driver end to end | 112.33 ms | 319.87 ms | same ranking |
| Candidate-scoped TileMaxSim, 3 GiB LRU tensor cache, native round trips | 573.43 ms | 1,812.70 ms | same ranking |
| Candidate-scoped TileMaxSim, small-cache Python driver end to end | 657.46 ms | 2,011.99 ms | same ranking |

The lexical scope contained 19.3 chunks and 13.2 documents on average. Because
the source chunks span multiple PDF pages, that mapped to 1,459 page tensors on
average (p95 4,000); no whole-corpus tensor scan occurred. Candidate document
recall was 0.650, and TileMaxSim retained the relevant document in its top five
for all covered queries, so scope generation—not exact reranking—set the recall
ceiling in this run. A 35% loss before TileMaxSim is unacceptable, so lexical
and non-relational graph results neither exclude candidates nor reorder general
tensor retrieval. Only explicit relationship intent may use and fuse GBrain's
complete graph scope as a hard range. The HNSW baseline is an intentionally strong upper-bound
workload whose queries are excerpts from their gold chunks; page-tensor and
text-chunk granularity differ, so its quality number is not a claim that the two
rankers are interchangeable.

The 3 GiB run completed every request with identical ranking, but incurred
24,432 GPU misses and 11.68 GB of host-to-device transfer across the query
sequence. Its 5.85x driver slowdown relative to resident mode is therefore a
cache-locality result that happens to resemble the 6x cache-capacity ratio; GPU
TileMaxSim arithmetic does not become six times slower when memory is smaller.

The reproducible drivers are
[`services/benchmark_gbrain_scoped_tilemaxsim.py`](services/benchmark_gbrain_scoped_tilemaxsim.py)
and
[`services/benchmark_postgres_single_vector.py`](services/benchmark_postgres_single_vector.py).

## Bounded multi-tenant GPU scheduling

TileMaxSim remains opt-in. Ordinary single-vector VectorChord use does not
start the CUDA daemon and does not require a GPU-memory setting. When an
operator enables TileMaxSim, `tilemaxsimd` reserves exactly the configured
CUDA devices and GiB allocations at startup and fails closed if any allocation
cannot be obtained.

The default `fair-priority` scheduler combines explicit request urgency with
weighted tenant fairness. Higher numeric priority is more urgent; requests in
the configured priority band are selected by normalized GPU service consumed,
not merely by request count. The default band spans the complete public priority
range, so urgency breaks ties and reorders work inside one tenant without letting
a high-priority noisy tenant starve an under-served tenant. Waiting requests age
up to the maximum priority. Operators that require global strict priority can
select `priority`; a narrower band is an explicit stronger-urgency trade-off.
This is a serving-design adaptation rather than wire compatibility with vLLM:
VectorChord intentionally uses higher-number-means-more-urgent throughout its
SQL, MCP, and IPC contracts.
`fair` and strict `priority` (priority then FCFS) are also available. This takes
the useful serving ideas from vLLM—bounded work budgets, continuous scheduling,
and resumable long work—while adding tenant isolation: a large request is split
at candidate/token quanta and re-enters the scheduler between CUDA launches.
It is cooperative preemption between kernels, not interruption of an executing
CUDA kernel.

Admission is bounded both globally and per tenant before work enters the
scheduler. Client disconnects and end-to-end deadlines are checked between
quanta. GPU and host cache ownership also have per-tenant caps; optional GPU
reservations and tenant scheduling weights can be configured for differentiated
service. Tenant identifiers are accepted only as scheduling domains, never as
authorization evidence, and request logs expose only a stable tenant hash.

Memory flags use GiB rather than byte counts. A representative launch is:

```shell
tilemaxsimd \
  --socket /run/vectorchord/tilemaxsim.sock \
  --status-socket /run/vectorchord/tilemaxsim-status.sock \
  --gpu-memory-gb 0=20 \
  --gpu-workspace-gb 2 \
  --host-cache-gb 8 \
  --max-inflight-request-gb 1 \
  --contract-root MODEL_CONTRACT_ID=/srv/vectorchord/tensors \
  --scheduler-policy fair-priority \
  --max-queued-requests 128 \
  --max-tenant-queued-requests 16 \
  --scheduler-quantum-fmas 4000000000 \
  --tenant-weight foreground=2 \
  --tenant-cache-reservation foreground=4
```

The optional status socket serves HTTP `GET /healthz` and Prometheus
`GET /metrics`. Metrics include readiness, scheduler depth, active CUDA work,
completed/error/timeout/disconnect outcomes, and global/per-tenant admission
rejections without exporting tenant identifiers.

`GET /livez` reports process liveness separately from readiness. The packaged
`tilemaxsimctl` probe can wait on the status socket without curl or a TCP port.
An opt-in hardened systemd unit and environment example are provided under
`deploy/systemd`; the CUDA container uses the same probe for its health check.

The in-flight request budget is also expressed in GiB. A reader must reserve
its complete declared frame after the fixed header is validated, and keeps that
permit through completion, timeout, or disconnect. This bounds aggregate query
and descriptor memory even when many clients submit maximum-size frames at once.
The FMA quantum additionally bounds one non-preemptible CUDA launch by actual
`query rows × document rows × dimension` work. A single candidate above that
operator-configured limit is rejected instead of monopolizing a shared GPU.

PostgreSQL sends protocol v3 scheduling metadata only when
`vchordrq.maxsim_tenant` is set; otherwise it retains protocol v2 compatibility.
GBrain derives that tenant value from authenticated runtime context and may set
`vchordrq.maxsim_priority` in the range -100 through 100. Priority changes
latency ordering only and never bypasses PostgreSQL row visibility, ACL, source,
or other mandatory filters.

The implementation is currently under active development. Its SQL interfaces
and deployment packaging may change before a stable release. This repository
contains only the public implementation and public-facing project information;
private planning and application documentation are intentionally excluded.

This work is based on
[supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord). The
original VectorChord README is retained below for upstream installation,
licensing, and project information.

---

<div align="center">

# VectorChord

**Ready for the Billion-Scale Era. Host 100M vectors on a single i4i.xlarge ($247/mo) and [scale seamlessly to 1B+](https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql).**

[Official Site][official-site-link] · [Blog][blog-link] · [Docs][docs-link] · [Feedback][github-issues-link] · [Contact Us][email-link]

[![][github-release-shield]][github-release-link]
[![][docker-release-shield]][docker-release-link]
[![][docker-pulls-shield]][docker-pulls-link]
[![][ghcr-release-shield]][ghcr-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][discord-shield]][discord-link]
[![][X-shield]][X-link]
[![][deepwiki-shield]][deepwiki-link]
[![][license-1-shield]][license-1-link]
[![][license-2-shield]][license-2-link]

</div>

VectorChord (vchord) is a PostgreSQL extension engineered for scalable, high-performance, and cost-effective vector search.

To efficiently store vectors while preserving search quality, VectorChord applies RaBitQ[^1] compression together with autonomous reranking. With VectorChord, you can store 400,000 vectors for just $1, enabling significant savings: 6x more vectors compared to Pinecone's optimized storage and 26x more than pgvector/pgvecto.rs for the same price.

[^1]: Gao, Jianyang, and Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 2.3 (2024): 1-27.

![][image-compare]

## Features

VectorChord introduces remarkable enhancements over pgvecto.rs and pgvector:

**💰 Affordable Vector Search**: Host 100M × 768-dimensional vectors → AWS i4i.xlarge ($247/month)[^2], host 1B × 96-dimensional vectors → i7ie.6xlarge ($2246/month)[^3], helping you keep infrastructure costs down while maintaining competitive search quality.

[^2]: Please check out our [blog post](https://blog.vectorchord.ai/vectorchord-store-400k-vectors-for-1-in-postgresql) for more details.
[^3]: Please check out our [blog post](https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql) for more details.

**⚡ Accelerated Index Build**: Index 100 million vectors in just 20 minutes. Powered by hierarchical K-means and highly optimized disk operations, VectorChord eliminates the bottleneck of vector indexing on a single machine with limited hardware resources.

[^4]: Please check out our [blog post](https://blog.vectorchord.ai/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql#heading-hierarchical-k-means) for more technique details and [document](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html#hierarchical-k-means) for usages.

**📈 Smoothly Scale Up**: Scale with confidence as your data grows. Through dimensionality reduction and sampling[^5], VectorChord effectively controls memory growth, enabling 1B-vector indexes to be built on machines with 128GB of memory in practice.

[^5]: Please check out our [blog post](https://blog.vectorchord.ai/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql#heading-dimensionality-reduction) for more technique details and [document](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html#reduce-sampling-factor) for usages.

**🔌 Seamless Integration**: Fully compatible with pgvector data types and syntax while providing optimal defaults out of the box - no complex parameter tuning needed. Just drop in VectorChord for enhanced experience.

**💾 Efficient Storage with Low-Bit Data type**: Drastically reduce storage costs with our [native 4-bit (RaBitQ4) and 8-bit (RaBitQ8) vector types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html). Achieve massive space savings without compromising search quality—RaBitQ8 maintains high precision with <1% recall loss.

## Quick Start

For new users, we recommend using the Docker image to get started quickly. If you do not prefer Docker, please read [installation guide](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html) for other installation methods.

```bash
docker run \
  --name vectorchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d ghcr.io/tensorchord/vchord-postgres:pg18-v1.1.1
```

> [!NOTE]
> In addition to the base image with the VectorChord extension, we provide an all-in-one image, `tensorchord/vchord-suite:pg17-latest`. This comprehensive image includes all official TensorChord extensions, including `VectorChord`, `VectorChord-bm25` and `pg_tokenizer.rs` . Developers should select an image tag that is compatible with their extension's version, as indicated in [the support matrix](https://github.com/tensorchord/VectorChord-images?tab=readme-ov-file#support-matrix).

Then you can connect to the database using the `psql` command line tool. The default username is `postgres`, and the default password is `mysecretpassword`.

```bash
psql -h localhost -p 5432 -U postgres
```

Now you can play with VectorChord!

VectorChord depends on pgvector, including the vector representation. Since you can use them directly, your application can be easily migrated without pain!

```sql
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```

Similar to pgvector, you can create a table with vector column and insert some rows to it.

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) SELECT ARRAY[random(), random(), random()]::real[] FROM generate_series(1, 1000);
```

With VectorChord, you can create `vchordrq` indexes.

```SQL
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops);
```

And then perform a vector search using `SELECT ... ORDER BY ... LIMIT ...`.

```SQL
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```

For more usage, please read:

- [Indexing](https://docs.vectorchord.ai/vectorchord/usage/indexing.html)
- [Multi-Vector Retrieval](https://docs.vectorchord.ai/vectorchord/usage/indexing-with-maxsim-operators.html)
- [Quantization Types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html)
- [Graph Index](https://docs.vectorchord.ai/vectorchord/usage/graph-index.html)
- [Quantization Types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html)
- [Similarity Filter](https://docs.vectorchord.ai/vectorchord/usage/range-query.html)
- [PostgreSQL Tuning](https://docs.vectorchord.ai/vectorchord/usage/performance-tuning.html)
- [Monitoring](https://docs.vectorchord.ai/vectorchord/usage/monitoring.html)
- [Fallback Parameters](https://docs.vectorchord.ai/vectorchord/usage/fallback-parameters.html)
- [Measure Recall](https://docs.vectorchord.ai/vectorchord/usage/measure-recall.html)
- [Prewarm](https://docs.vectorchord.ai/vectorchord/usage/prewarm.html)
- [Prefilter](https://docs.vectorchord.ai/vectorchord/usage/prefilter.html)
- [Prefetch](https://docs.vectorchord.ai/vectorchord/usage/prefetch.html)
- [Rerank in Table](https://docs.vectorchord.ai/vectorchord/usage/rerank-in-table.html)
- [Partitioning Tuning](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html)
- [External Build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)

## License

This software is licensed under a dual license model:

1. **GNU Affero General Public License v3 (AGPLv3)**: You may use, modify, and distribute this software under the terms of the AGPLv3.

2. **Elastic License v2 (ELv2)**: You may also use, modify, and distribute this software under the Elastic License v2, which has specific restrictions.

You may choose either license based on its terms. The original VectorChord code
retains its upstream copyright notices. TileMaxSim fork additions are Copyright
(c) 2026 Hu Xinjing; use this fork's
[issue tracker](https://github.com/HuXinjing/VectorChord/issues) for support.

[cost-estimation]: https://github.com/user-attachments/assets/168fe550-6465-4eee-a224-8c848c301e3d
[image-compare]: https://github.com/user-attachments/assets/2d985f1e-7093-4c3a-8bf3-9f0b92c0e7e7
[license-1-link]: https://github.com/HuXinjing/VectorChord#license
[license-1-shield]: https://img.shields.io/badge/License-AGPLv3-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=
[license-2-link]: https://github.com/HuXinjing/VectorChord#license
[license-2-shield]: https://img.shields.io/badge/License-ELv2-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=

[docker-release-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-release-shield]: https://img.shields.io/docker/v/tensorchord/vchord-postgres?color=369eff&label=docker&labelColor=black&logo=docker&logoColor=white&style=flat
[github-release-link]: https://github.com/HuXinjing/VectorChord/releases
[github-release-shield]: https://img.shields.io/github/v/release/HuXinjing/VectorChord?color=369eff&labelColor=black&logo=github&style=flat
[ghcr-release-link]: https://github.com/users/HuXinjing/packages/container/package/vectorchord-tilemaxsimd
<!-- GHCR badge is not supported by shields.io yet, use docker badge instead -->
[ghcr-release-shield]: https://img.shields.io/docker/v/tensorchord/vchord-postgres?color=369eff&label=GHCR&labelColor=black&logo=github&logoColor=white&style=flat
[docker-pulls-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/vchord-postgres?color=45cc11&labelColor=black&style=flat&sort=semver
[previous-docker-pulls-link]: https://hub.docker.com/r/tensorchord/pgvecto-rs
[previous-docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/pgvecto-rs?color=45cc11&labelColor=black&style=flat&sort=semver
[github-downloads-link]: https://github.com/HuXinjing/VectorChord/releases
[github-downloads-shield]: https://img.shields.io/github/downloads/HuXinjing/VectorChord/total?color=45cc11&labelColor=black&style=flat&sort=semver
[discord-link]: https://discord.gg/KqswhpVgdU
[discord-shield]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[X-link]: https://x.com/TensorChord
[X-shield]: https://img.shields.io/badge/follow-%40tensorchord-1DA1F2?logo=x&style=flat&logoColor=white&color=1da1f2
[deepwiki-link]: https://deepwiki.com/HuXinjing/VectorChord
[deepwiki-shield]: https://deepwiki.com/badge.svg
[blog-link]: https://blog.vectorchord.ai/
[official-site-link]: https://vectorchord.ai/
[github-issues-link]: https://github.com/HuXinjing/VectorChord/issues
[email-link]: https://github.com/HuXinjing/VectorChord/issues
[docs-link]: https://docs.vectorchord.ai/vectorchord/
