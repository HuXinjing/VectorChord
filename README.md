<div align="center">

# VectorChord TileMaxSim

**PostgreSQL-native vector and tensor retrieval with exact TileMaxSim and a persistent GPU cache.**

English | [简体中文](README.zh-CN.md)

</div>

VectorChord TileMaxSim is an open-source retrieval engine built on PostgreSQL
and derived from the upstream
[VectorChord](https://github.com/supervc-stack/VectorChord) codebase. It preserves
the traditional single-vector path while adding exact late-interaction retrieval
for documents represented as arrays of token vectors.

This repository is maintained as its own project. Its public README describes
this project's implementation, measured results, limitations, and roadmap; it
does not reproduce the upstream project's README or present upstream product
claims as our own.

## Project scope

VectorChord TileMaxSim provides infrastructure primitives:

- vector, tensor-descriptor, and retrieval-metadata storage in PostgreSQL;
- traditional single-vector retrieval and exact multi-vector TileMaxSim;
- PostgreSQL permission, MVCC, row-visibility, cancellation, and timeout
  semantics around retrieval;
- a persistent GPU tensor arena, host-memory cache, and immutable disk shards;
- bounded admission, fair/priority scheduling, cache quotas, health probes, and
  metrics for the optional GPU service.

It does not implement application identity, authentication, ACL policy, entity
registries, facts, events, relationship graphs, communities, or query-intent
routing. Those belong to an authenticated application or knowledge-governance
layer. VectorChord accepts only the already-authorized candidate scope and
opaque scheduling hints supplied by that caller.

## Core capabilities

- Exact TileMaxSim reranking on CPU and through the native Rust/CUDA
  `tilemaxsimd` backend.
- No artificial cap on a caller-authorized tensor scope. Oversized working sets
  are processed in bounded chunks.
- External tensor-source registration, so full token tensors can remain outside
  the indexed PostgreSQL value while stable public IDs and descriptors remain
  queryable.
- A three-level tensor path:
  - L0: persistent GPU page cache;
  - L1: bounded host-memory cache;
  - L2: immutable tensor shards or content-addressed objects on durable storage.
- Batched shard reads, batched host-to-device transfers, content-addressed
  deduplication, TinyLFU/GDSF admission, and a coalescing page-run allocator.
- Optional resident prewarming that completes before the daemon reports ready.
- Per-daemon fair, strict-priority, or fair-priority scheduling, request aging,
  tenant weights, deadlines, disconnect cancellation, and bounded CUDA work
  quanta.
- PostgreSQL planner statistics and cost estimation for multi-vector queries.
- Prometheus metrics and separate liveness/readiness probes.
- Correctness, registry, protocol, planner, real-GPU, and fault-path tests.

TileMaxSim is opt-in. Ordinary single-vector operation does not start the CUDA
daemon and does not require a GPU-memory setting. When enabled, `tilemaxsimd`
reserves the configured devices and GiB allocations during startup and exits if
any requested device or allocation cannot be obtained.

## Architecture

```text
Authenticated application / retrieval planner
                  │ authorized IDs + scheduling hints
                  ▼
            PostgreSQL + VectorChord
                  │ tensor descriptors
                  ▼
              tilemaxsimd
        ┌─────────┼──────────┐
        │         │          │
      L0 GPU   L1 host    L2 durable
      cache     cache      shards
        │
        └── exact CUDA TileMaxSim
```

PostgreSQL remains the source of row visibility and hard filtering. The daemon
does not infer authorization from tenant or priority fields. It receives an
explicit tensor set, resolves the corresponding immutable objects through the
cache hierarchy, and returns exact scores.

## GPU cache and scheduling

The default `fair-priority` scheduler combines explicit urgency with weighted
fairness. Large requests are split by candidate count, document tokens, and
actual `query rows × document rows × dimension` work. A request re-enters the
scheduler between CUDA launches, allowing another scheduling domain to run.
An executing CUDA kernel cannot be interrupted, so the FMA quantum bounds the
cooperative, non-preemptible interval.

GPU and host caches have per-scheduling-domain maximums. An optional
`--tenant-cache-reservation TENANT=GB` protects already-warmed GPU pages from
cross-domain eviction below the configured allocation. This is currently an
aggregate byte reservation, not a permanent partition for one particular
knowledge base: it does not prewarm data, does not provide an L1 host-cache
minimum, and does not distinguish multiple sources owned by one tenant. These
gaps are listed explicitly in the roadmap.

Memory flags use GiB rather than raw byte counts. A representative launch is:

```shell
tilemaxsimd \
  --socket /run/vectorchord/tilemaxsim.sock \
  --listen 0.0.0.0:9191 \
  --status-socket /run/vectorchord/tilemaxsim-status.sock \
  --gpu-memory-gb 0=20 \
  --gpu-workspace-gb 2 \
  --host-cache-gb 8 \
  --io-pipeline overlap \
  --io-batch-gb 0.05 \
  --max-inflight-request-gb 1 \
  --contract-root MODEL_CONTRACT_ID=/srv/vectorchord/tensors \
  --scheduler-policy fair-priority \
  --max-queued-requests 128 \
  --max-tenant-queued-requests 16 \
  --scheduler-quantum-fmas 4000000000 \
  --scheduler-quantum-io-gb 1 \
  --tenant-weight foreground=2 \
  --tenant-cache-reservation foreground=4
```

PostgreSQL can connect through a local Unix socket or a `tcp://HOST:PORT`
endpoint. `GET /livez`, `GET /healthz`, and `GET /metrics` expose process
liveness, readiness, and bounded operational metrics. See
[`docs/TILEMAXSIM_CUDA_SIDECAR.md`](docs/TILEMAXSIM_CUDA_SIDECAR.md) and
[`docs/TILEMAXSIM_IPC_V2.md`](docs/TILEMAXSIM_IPC_V2.md) for deployment and
protocol details.

## Measured performance

The development corpus contains 34,054 descriptors, 34,027 unique tensors, and
16.28 GB of logical FP16 tensor data. Absolute latency depends on storage, CPU,
and GPU hardware; same-machine comparisons are more meaningful than isolated
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
| Small-VRAM cold I/O pipeline | 757.96 ms, serial | 685.85 ms, overlap | 1.11x faster; identical scores |

The I/O-pipeline row used five cold trials over 1,000 real tensors
(478,016,000 logical bytes) with only 0.5 GiB assigned to the daemon: 0.4 GiB
for tensor pages and 0.1 GiB for CUDA workspace. The 0.05 GiB bounded resolver
stage overlaps SSD/L1 resolution of batch N+1, H2D upload of batch N, and exact
TileMaxSim for batch N-1. The maximum score delta between serial and overlap
was zero. Results are hardware- and batch-size-sensitive, so production must
measure the exposed pipeline metrics rather than assume this exact ratio.

A 20 GiB run assigned 18 GiB to tensors and 2 GiB to workspace. Prewarming all
34,027 unique tensors took 14.86 seconds in the Rust/CUDA daemon versus 23.07
seconds in the Python/Triton implementation. The default 32 KiB page-run
allocator reduced allocated space from 17.840 GB with the former 256 KiB buddy
allocator to 16.725 GB, reducing internal rounding waste from 8.82% to 2.74%.

### Full-range and small-cache stress

With all tensors resident, an intentionally exhaustive exact scan of all 34,054
descriptors averaged 1.08 seconds. A 3 GiB tensor arena plus 1 GiB host cache
returned the same exact top-K, but sequential full scans caused 68,106 misses,
61,499 evictions, and 32.53 GB of host-to-device transfer; native round trips
averaged 18.47 seconds.

The slowdown is cyclic cache thrashing and repeated I/O, not TileMaxSim
arithmetic becoming linearly slower as memory shrinks. Candidate-scoped queries
whose hot working set fits the cache do not exhibit this full-scan behavior.

### Candidate-scope ablation

Forty real queries compared the original 1,024-dimensional text embedding with
pgvector HNSW against exact TileMaxSim over a lexical candidate scope:

| Path | Mean latency | p95 | Quality |
| --- | ---: | ---: | ---: |
| Original text embedding + pgvector HNSW | 5.80 ms | 5.86 ms | document hit@1 1.000 |
| Candidate-scoped TileMaxSim, 18 GiB resident cache | 45.56 ms | 134.99 ms | hit@1 0.475; hit@5 0.650 |
| Candidate-scoped TileMaxSim, 3 GiB cache | 573.43 ms | 1,812.70 ms | identical TileMaxSim ranking |

The lexical scope covered the relevant document for only 65% of queries.
TileMaxSim kept the relevant document in its top five for every covered query,
so candidate generation—not exact scoring—set the recall ceiling. Lexical and
ordinary graph matches must therefore not be the only hard gate for general
semantic retrieval. Authorization filters remain hard; relational graph scopes
may be used for explicitly relational intent when the application can construct
a complete range.

Reproducible drivers:

- [`services/benchmark_tilemaxsim_ablation.py`](services/benchmark_tilemaxsim_ablation.py)
- [`services/benchmark_full_corpus_tilemaxsim.py`](services/benchmark_full_corpus_tilemaxsim.py)
- [`services/benchmark_gbrain_scoped_tilemaxsim.py`](services/benchmark_gbrain_scoped_tilemaxsim.py)
- [`services/benchmark_postgres_single_vector.py`](services/benchmark_postgres_single_vector.py)
- [`services/benchmark_tilemaxsim_io_pipeline.py`](services/benchmark_tilemaxsim_io_pipeline.py)

## Limitations

This project is suitable for controlled production pilots. Strict low-latency,
high-availability, or large multi-user deployments must address or explicitly
accept the following limitations before general availability.

### Retrieval scale

- Exact TileMaxSim is a scorer, not a tensor-native approximate-nearest-
  neighbour index. Full residency removes transfer time but not exact scoring
  arithmetic.
- Large low-latency corpora still need a measured high-recall candidate
  generator or a future tensor-native ANN before exact TileMaxSim.
- A cache smaller than the active working set remains correct but can thrash;
  production capacity must cover the hot set or preserve a safe high-recall
  range.
- The current Tutti-inspired pipeline still uses CPU shard reads and pinned
  host-to-device copies; it is not GPU io_uring or a GPUDirect Storage backend.
  SSD, PCIe, and H2D contention make `--io-batch-gb` hardware-sensitive. Use
  `--io-pipeline serial` as the ablation/fallback mode and inspect
  `tilemaxsim_io_pipeline_*` before enabling overlap for an SLO.

### Cache and multi-user isolation

- Fairness, priority, admission, and reservations are enforced independently by
  each daemon. Replicas do not share a global queue, entitlement ledger, or
  residency-aware router.
- A GPU reservation protects aggregate bytes for a scheduling tenant, not a
  named `source_id` or knowledge base. There is no dynamic cache-domain API,
  source-level prewarm contract, or host-cache minimum.
- Reservations are configured at daemon startup. On multi-GPU daemons the
  current value is applied to each device, rather than being expressed as one
  explicit cluster-wide total.
- Globally resident manifests are pinned under a service-owned domain. If
  pinned pages consume the entire arena, unrelated cold tensors cannot evict
  them and the request fails instead of waiting for pinned data to leave.
- Preemption occurs only between CUDA launches. A running kernel is not
  interruptible.
- Tenant and priority values are scheduling hints, not authorization evidence.

### Operations and security

- The TileMaxSim TCP protocol has no built-in TLS or client authentication. It
  must remain on a private network or behind a mutually authenticated proxy.
- A replacement GPU replica starts cold. Durable storage preserves correctness,
  but availability and warm-cache latency are separate SLOs.
- PostgreSQL WAL archiving, PITR, cross-site disaster recovery, and verified
  restore procedures remain database-platform responsibilities.
- Metrics exist, but ready-made SLO dashboards and alert rules are not yet
  included.
- Very large tensor registries and garbage collection require scale-specific
  operational testing.

### Release status

The SQL surface, external-tensor protocol, image tags, and deployment packaging
remain under active development. Real-GPU and fault-path tests do not replace
multi-hour, many-user soak and chaos testing on target hardware.

## Outlook: an AI infrastructure retrieval plane

VectorChord TileMaxSim is intended to become the retrieval plane in a broader,
composable AI infrastructure rather than absorb application or model-serving
responsibilities:

```text
Agent / application layer
          │
          ▼
AI gateway: identity, quota, priority, deadline, trace
          │
          ├── Knowledge layer ── PostgreSQL / VectorChord ── tilemaxsimd
          │
          └── Ray Serve ── vLLM and other model engines

Kubernetes / KubeRay: placement, GPU nodes, scaling, recovery
Object storage: model weights, tensor shards, ingestion artifacts
Prometheus + OpenTelemetry: end-to-end observability
```

In this design:

- [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/) coordinates
  distributed model replicas, autoscaling, and model-aware routing.
- [vLLM](https://docs.vllm.ai/) owns generation batching, model parallelism,
  KV-cache management, and prefix-cache reuse.
- VectorChord owns PostgreSQL retrieval semantics and vector/tensor metadata;
  `tilemaxsimd` owns exact TileMaxSim, persistent tensor residency, and bounded
  retrieval scheduling.
- The authenticated application owns identity, ACL, graph/fact/event
  governance, query intent, and the final RAG pipeline.

The components should not silently compete for the same GPU memory. Initial
deployments should place vLLM and TileMaxSim in separate GPU pools. Kubernetes
or another resource manager must account for the memory reserved by the
long-lived TileMaxSim daemon before scheduling model workers; explicit MIG or
other hardware partitioning can be evaluated later.

The next infrastructure milestones are:

1. Separate `scheduler_tenant` from an opaque `cache_domain`, typically derived
   by the authorized caller from tenant and source identity.
2. Add GPU and host `min_resident_gb`, `max_gb`, and prewarm contracts per cache
   domain, with unambiguous per-device and fleet-wide semantics.
3. Build a bounded global admission layer and a residency-aware router across
   TileMaxSim replicas, while leaving exact local quantum scheduling in each
   daemon.
4. Propagate one request envelope—request ID, tenant, source, priority,
   deadline, trace ID, and cache domain—through retrieval and model serving.
5. Validate cold-start recovery, rolling upgrades, PostgreSQL failover, GPU-pod
   loss, multi-tenant isolation, and long-running soak/chaos behavior.
6. Add a tensor-native ANN or another recall-measured first stage for corpora
   where exact full-range TileMaxSim cannot meet the latency SLO.

Ray is an orchestration layer, not durable storage. PostgreSQL and immutable
object storage remain authoritative. `tilemaxsimd` should remain a supervised,
stateful GPU service rather than a short-lived generic Ray task, so its reserved
arena and cache lifetime remain predictable.

## Security boundary

PostgreSQL sends scheduled protocol metadata only when
`vchordrq.maxsim_tenant` is set. Priority affects latency ordering only; it
cannot bypass row visibility, ACL, source, or other mandatory filters. Request
logs expose a stable hash rather than raw tenant identifiers.

This public repository contains implementation and public-facing project
information only. Private application and planning documents are intentionally
excluded.

## Build and test

Useful entry points include:

```shell
make build
cargo test --locked --workspace --exclude vchord --no-fail-fast
cargo test --manifest-path services/tilemaxsimd/Cargo.toml --locked
```

The CUDA container and its test stage are defined in
[`services/Dockerfile.tilemaxsimd`](services/Dockerfile.tilemaxsimd). PostgreSQL
tensor-source integration is documented in
[`docs/MAXSIM_TENSOR_SOURCES.md`](docs/MAXSIM_TENSOR_SOURCES.md).

## Acknowledgements

We thank the maintainers and contributors of the upstream
[supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord)
project. Its PostgreSQL extension, vector types, `vchordrq` index, and existing
MaxSim support provide the foundation on which this project builds.

The project's IO-aware GPU MaxSim direction and the `TileMaxSim` name are
informed by Ashutosh Sharma's paper,
[“TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization”](https://arxiv.org/abs/2606.26439),
arXiv:2606.26439 (2026), and its accompanying open-source
[ashutoshuiuc/tilemaxsim](https://github.com/ashutoshuiuc/tilemaxsim)
Triton implementation. We gratefully acknowledge that work and its published
analysis of fused, tiled MaxSim scoring.

The bounded asynchronous tensor pipeline is informed by
[“Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving”](https://arxiv.org/abs/2605.03375),
particularly its GPU-native object, asynchronous I/O, and slack-aware
scheduling principles. This implementation currently adopts the scheduling
and pipelining ideas only; it does not include Tutti's GPU io_uring storage
stack.

This repository is a separate PostgreSQL/Rust/CUDA systems integration. Its
IPC protocols, persistent three-level tensor cache, allocator, multi-user
scheduler, database bindings, and operational packaging are maintained here;
the acknowledgement does not imply endorsement or shared maintenance by the
upstream projects or paper author.

## Project provenance and license

This project is derived from
[supervc-stack/VectorChord](https://github.com/supervc-stack/VectorChord).
Original source files retain their upstream copyright and license notices.
TileMaxSim project additions are Copyright (c) 2026 Hu Xinjing.

The repository is offered under the license terms in [`LICENSE`](LICENSE),
including AGPLv3 and Elastic License v2 options where applicable. Use this
project's [issue tracker](https://github.com/HuXinjing/VectorChord/issues) for
support and project-specific reports.
