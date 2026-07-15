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
- `GET /metrics`: bounded Prometheus scheduler and outcome counters.

Use the image's dependency-free probe from systemd, Docker, or Kubernetes:

```shell
tilemaxsimctl \
  --socket /run/vectorchord/tilemaxsim-status.sock \
  --wait-timeout-ms 30000
```

Readiness is removed before shutdown. Unexpected scheduler or status-thread
exit makes the whole daemon fail, so a supervisor cannot keep routing requests
to a process whose CUDA worker has disappeared.

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

The shard publisher must write complete immutable records and atomically
publish its index. SIGHUP reloads shard indexes but never changes the configured
contract-to-root mapping. The daemon validates contract, digest, shape, dtype,
length, and finite values before GPU admission.

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
