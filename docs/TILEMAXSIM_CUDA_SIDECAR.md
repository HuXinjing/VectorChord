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
  --max-tenant-queued-requests 16
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

- PostgreSQL splits one logical candidate set into protocol-bounded batches and
  merges a deterministic global top-k.
- All batches share the caller's one logical deadline; a new IPC round trip does
  not refresh it.
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
