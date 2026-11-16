# VectorChord Native CUDA TileMaxSim Daemon

## Scope

`services/tilemaxsimd` is VectorChord's production CUDA execution service for
external TileMaxSim protocol v2/v3. PostgreSQL remains responsible for MVCC
visibility, hard filters, descriptor validation, and public-ID mapping. The
daemon owns only bounded tensor loading, host/GPU caches, scheduling, and exact
TileMaxSim execution.

The daemon does not fetch S3 or HTTP objects and does not make authorization,
graph, Fact, community, or application-routing decisions. An application such
as GBrain publishes immutable tensors into a configured node-local shard root.

The Python sidecar remains a reference and conformance implementation. New
production deployments should use the Rust/CUDA daemon.

## Fail-closed resource contract

TileMaxSim is opt-in. Do not start this service when no GPU memory has been
explicitly assigned. Every `--gpu-memory-gb GPU=GB` value is required, uses GiB
rather than bytes, and is reserved with one CUDA allocation during startup.
An invalid device, insufficient free memory, failed pinned-host allocation, or
failed CUDA stream creation terminates the process before either socket becomes
ready.

The configured allocation is split into a persistent tensor arena and
`--gpu-workspace-gb`. The workspace must be smaller than every GPU allocation.
Host cache and aggregate in-flight request budgets are also configured in GiB.

## Build the production image

The build and runtime CUDA images are explicit deployment inputs:

```shell
docker build \
  --build-arg CUDA_DEVEL_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  --build-arg CUDA_RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
  -f services/Dockerfile.tilemaxsimd \
  -t vectorchord-tilemaxsimd:local \
  .
```

Pin both images by digest in a production build. The container health check
expects the status socket at
`/run/vectorchord/tilemaxsim-status.sock`; mount the socket directory so the
PostgreSQL process can reach the protocol socket.

## Start the daemon

```shell
tilemaxsimd \
  --socket /run/vectorchord/tilemaxsim.sock \
  --status-socket /run/vectorchord/tilemaxsim-status.sock \
  --socket-mode 660 \
  --status-socket-mode 660 \
  --gpu-memory-gb 0=20 \
  --gpu-workspace-gb 2 \
  --host-cache-gb 8 \
  --max-inflight-request-gb 1 \
  --contract-root 'MODEL_CONTRACT_ID=/var/lib/vectorchord/tensors' \
  --scheduler-policy fair-priority \
  --max-connections 256 \
  --max-queued-requests 128 \
  --max-tenant-queued-requests 16 \
  --scheduler-max-microbatch-requests 8 \
  --scheduler-min-shared-candidates-milli 500
```

Multiple GPU assignments may be supplied. Device selection and the requested
allocation are never inferred from all currently free VRAM.

The example systemd unit and environment file are in `deploy/systemd`. The unit
is deliberately opt-in: enabling it is the operator's explicit TileMaxSim/GPU
configuration. It waits for readiness, reloads immutable shard indexes with
SIGHUP, restarts on failure, and gives SIGTERM shutdown time to stop accepts and
drain admitted work.

## Health and metrics

The status Unix socket serves:

- `GET /livez`: the status server is alive;
- `GET /healthz`: the protocol listener and scheduler are ready;
- `GET /metrics`: bounded Prometheus resource, scheduler, cache, transfer,
  storage, latency, timeout, and outcome metrics. GPU labels contain only the
  configured numeric device/slot; application scheduling-domain names are
  never exported.
- `GET /v1/config`: effective non-secret instance settings and the stable
  application request-control contract;
- `GET /v1/cache`: JSON L0/L1 cache and adaptive GPU state;
- `POST /v1/reload`: atomically reload the shard index. Quantizer activation is
  resolved atomically per request and needs no reload. The endpoint is accepted
  only over the Unix status socket; TCP status is read-only. See
  [`TILEMAXSIM_MANAGEMENT_API.md`](TILEMAXSIM_MANAGEMENT_API.md) for all runtime
  operations and schemas.

`GET /v1/cache` also reports the accelerator backend, device name and
architecture, CUDA driver/runtime/cuBLAS versions, SM count, warp size, memory
bus and clock, shared-memory limit, and calibrated dispatch state. This runtime
fingerprint is part of performance diagnosis; a model name alone is not a safe
kernel-selection contract.

CUDA builds produce an explicit fat binary. The default CUDA 12 build targets
native `sm_80`, `sm_89`, `sm_90`, and Hopper-specific `sm_90a` code plus
`compute_90` PTX. Override it
with a validated toolchain-specific list, for example a CUDA 13 Blackwell
builder:

```shell
TILEMAXSIM_CUDA_ARCHS=89,90,120,121,121-virtual cargo build \
  --release --manifest-path services/tilemaxsimd/Cargo.toml
```

The repository also provides `services/Dockerfile.tilemaxsimd-blackwell`, a
pinned multi-architecture CUDA 13.3 build that emits native SM120/SM121 code
and compute_121 PTX. It is separate from the CUDA 12 Ada/Hopper image so an
older production toolchain cannot silently drop the DGX Spark target.

An architecture appearing in the binary means only that it can be loaded.
Architecture-specific kernels and same-device numerical/performance regression
remain required before that device is advertised as tuned.

The daemon core is backend-neutral. CUDA remains the default build, while the
exact CPU reference build has no CUDA or cuBLAS link dependency:

```shell
cargo build --release --manifest-path services/tilemaxsimd/Cargo.toml
cargo build --release --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-cpu
```

Use `--device-memory-gb 0=GB` for new deployments. The historical
`--gpu-memory-gb` spelling remains a visible compatibility alias. The CPU
backend implements exact FP16/FP32 MaxSim and fails closed for unsupported
quantized profiles; it is both a no-accelerator deployment path and the common
numerical oracle for vendor backends.

Use the image's dependency-free probe from systemd, Docker, or Kubernetes:

```shell
tilemaxsimctl \
  --socket /run/vectorchord/tilemaxsim-status.sock \
  --wait-timeout-ms 30000
```

Readiness is removed before shutdown. Unexpected scheduler or status-thread
exit makes the whole daemon fail, so a supervisor cannot keep routing requests
to a process whose CUDA worker has disappeared.

The most useful production signals are:

- `tilemaxsim_pending_requests` and `tilemaxsim_scheduler_queue_depth` for
  saturation;
- `tilemaxsim_admission_rejections_total` and `tilemaxsim_timeouts_total` for
  overload or an undersized deadline;
- `tilemaxsim_gpu_cache_events_total`, `tilemaxsim_gpu_h2d_bytes_total`, and
  `tilemaxsim_storage_read_bytes_total` for cache churn and cold-load traffic;
- `tilemaxsim_gpu_cache_bytes` for free space, largest free extent, payload,
  allocator waste, and pinned capacity;
- `tilemaxsim_host_cache_bytes` and `tilemaxsim_host_cache_events_total` for
  the L1 cache;
- `tilemaxsim_request_duration_seconds`, `tilemaxsim_queue_duration_seconds`,
  and `tilemaxsim_gpu_duration_seconds` for rate-derived mean latency;
- `tilemaxsim_requests_admitted_total{priority_class=...}` and
  `tilemaxsim_scheduler_requeues_total` for priority/cooperative scheduling.

At minimum, alert when readiness is zero, any admission-rejection rate is
nonzero under expected load, timeout ratio exceeds the application SLO, queue
depth remains above 80% of its limit, or cache admission rejections rise while
`largest_free_extent / free` is small. A rising miss/eviction/H2D rate with
stable traffic means the assigned GPU cache is too small or the access set has
poor locality; it is not a PostgreSQL HNSW symptom.

## Immutable tensor storage

Each model contract maps to one absolute shard root. A protocol descriptor uses
a content-addressed reference and matching checksum:

```text
tensor_ref      = sha256://0123...cdef
tensor_checksum = sha256:0123...cdef
tensor_rows     = 747
tensor_dim      = 320
tensor_dtype    = float16
```

Online writers publish canonical tensor bytes through VectorChord's publisher,
so application code never owns or duplicates the disk layout:

```shell
install -d -m 0750 /var/lib/vectorchord/tensors
tilemaxsimctl publish-object \
  --root /var/lib/vectorchord/tensors \
  --rows 747 \
  --dimension 320 \
  --dtype float16 \
  --expected-sha256 0123...cdef < document.tensor.f16
```

The command fsyncs a temporary object, publishes it by immutable hard link, and
returns the JSON descriptor. Repeating the same publication is idempotent. A
daemon that already has the contract root open can read the new content address
without restart or SIGHUP. SIGHUP is needed only for a newly published bulk
shard index. It never changes the configured contract-to-root mapping. The
daemon validates contract, digest, shape, dtype, length, and checksum before GPU
admission.

## Batching, scheduling, and failure semantics

- Protocol v6 sends one logical external descriptor set. The daemon splits it
  into bounded scheduling/GPU quanta and returns only the deterministic global
  top-k. This removes serial per-batch TCP setup from the normal path.
- Protocols v2 through v5 remain accepted during rolling upgrades. Legacy
  client batches share one logical deadline; a new IPC round trip does not
  refresh it.
- Admission is bounded by connections, queued requests, per-scheduling-domain
  queue depth, and aggregate frame bytes.
- Long batches re-enter the selected scheduler between candidate/token quanta.
  This is cooperative preemption between CUDA kernels; a running kernel cannot
  be interrupted safely.
- `fair-priority` preserves weighted fairness within the configured priority
  band. Strict global priority is available with `--scheduler-policy priority`.
- Disconnects and deadlines are checked between quanta. CUDA, shard, checksum,
  workspace, or timeout failure fails the logical request; PostgreSQL does not
  expose partial results.

Scheduling-domain strings affect latency ordering and cache quotas only. They
are not authorization evidence.

## Verification

```shell
cargo test --manifest-path services/tilemaxsimd/Cargo.toml --locked
python3 -m unittest -v services.test_tilemaxsim_rust_daemon
```

The Python integration suite exercises real Unix sockets and, when CUDA is
available, multi-GPU, resident-cache, oversized-working-set, concurrent-reader,
health, and score-equivalence paths. Application-level corpus recall and
end-to-end latency remain a release acceptance gate rather than an inference
from synthetic service tests.

## Bounded read/compute overlap on the Hopper release

The optional pipeline reads and verifies the next immutable tensor batch while
this batch is uploaded and scored. It does not change canonical tensor shapes,
bytes, GPU layout, quantization contracts, or the Hopper fused kernel. The
accelerator worker retains exclusive ownership: upload and compute calls are
still serialized through the existing backend interface. This ports the storage
prefetch part of `feature/tutti-io-pipeline`, not its concurrent GPU aliases or a
GPUDirect Storage implementation.

```shell
--io-pipeline overlap --io-batch-mib 64 --tensor-read-workers 16
```

- `disabled` is the default and retains the original whole-quantum loader.
- `serial` uses the same bounded batches without overlapping, for ablation.
- `overlap` uses a capacity-one channel. At most one batch is consumed, one is
  queued and one is resolving. Each is bounded by the configured payload budget
  (1 through 1024 MiB), except one indivisible tensor may exceed that budget and
  remains subject to the existing descriptor and device capacity checks.
- Only the exact profile uses the new path. Quantized paths retain their loader
  and encoding checks. Fully resident requests retain their existing fast paths.
- Host-cache admissions remain ordered and retain per-owner limits. GPU quota,
  lease, transient admission, ready-state and error cleanup use the existing
  common implementation. Early consumer errors drop the receiver before joining
  the worker, so a blocked sender cannot deadlock shutdown. This does not change
  cooperative cancellation/deadline boundaries or interrupt running I/O/kernels.
- `tilemaxsim_io_pipeline` events report completed batches, resolve time,
  consumer time and elapsed time. Resolve and consumer times overlap and must not
  be added to estimate wall time. Consumer time includes existing admission,
  upload, compute and cleanup, not just a CUDA kernel.

### H200 experiment, 2026-11-16

A separate process used the already assigned scorer GPU, a 2 GiB total GPU
allocation (0.5 GiB workspace), a 1 MiB host cache, 16 readers and 64 MiB batches.
Production was neither restarted nor reconfigured. Inputs came from the frozen
resident descriptor snapshot and immutable corpus. A deterministic 32-row,
320-dimensional query tensor was taken from corpus tensor bytes; this is a
storage/compute experiment, not an Agent query or recall benchmark.

Each mode had a fresh daemon cache. Shared filesystem page caches were not
purged, so these are application-cold and over-capacity results, not physical
SSD cold-read measurements. Mode order was reversed in the second trial.
Medians of two requests per mode, measured through the local Unix scoring socket:

| Candidates | Original loader | Batched serial | Batched overlap |
| ---: | ---: | ---: | ---: |
| 512 | 0.230 s | 0.237 s | 0.189 s |
| 8,192 | 3.125 s | 3.541 s | 2.139 s |
| 45,438 | 16.488 s | 17.045 s | 12.150 s |

All candidate IDs and single-request scores matched exactly. In the full case,
process peak RSS was approximately 20.7 GiB with the original loader and at most
0.63 GiB with overlap. Fully resident requests showed no established speedup.
These two-sample measurements do not establish P95, concurrent tenant throughput,
physical SSD performance, end-to-end RAG latency or cache-recovery speedup.

A separate shape check submitted four resident requests, each with 32 query rows
and dimension 320. All three loader modes actually invoked the Hopper fused
kernel. Their maximum deviation from the single-request reference was the same
`3.6239624e-5`, below the existing GPU test tolerance `1e-4`.

Reproduce a scoped ablation with an explicitly supplied binary and resources:

```shell
python -m services.benchmark_tilemaxsim_io_pipeline \
  --descriptor-manifest "$DESCRIPTORS_JSONL" \
  --query-embedding "$QUERY_NPY" --shard-root "$SHARD_ROOT" \
  --rust-binary "$DAEMON_BINARY" --contract "$MODEL_CONTRACT" \
  --device 0 --gpu-memory-gb 2 --workspace-gb 0.5 \
  --host-cache-gb 0.001 --io-batch-mib 64 --candidates 8192 --trials 3
```

The benchmark starts isolated processes, compares original/serial/overlap modes,
checks candidate/score equivalence and does not evict shared filesystem caches.
Its oversized cooperative quantum is for isolated ablation only; keep production
fairness and deadline settings when evaluating a rollout.
