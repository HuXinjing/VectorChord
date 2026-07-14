# Native TileMaxSim service

`vchord-tilemaxsimd` is the optional Rust/CUDA execution service for exact
TileMaxSim reranking. It owns explicitly assigned GPU memory, a decoded host
cache, immutable tensor-shard readers, cache admission, and the scoring
scheduler. Base pgvector and VectorChord queries do not start this process or
reserve GPU memory. If `vchordrq.maxsim_gpu_endpoint` is not configured, there
is no TileMaxSim daemon requirement.

The service is Linux-only and requires an NVIDIA driver plus a CUDA toolkit at
build time. The supplied image builds against CUDA 12.6. The runtime driver
must satisfy NVIDIA's compatibility requirements for that CUDA version.

## Build and command reference

Build both installed commands from the repository root:

```bash
cargo build --release --locked \
  --manifest-path services/tilemaxsimd/Cargo.toml
install -m 0755 services/tilemaxsimd/target/release/vchord-tilemaxsimd \
  /usr/local/bin/vchord-tilemaxsimd
install -m 0755 services/tilemaxsimd/target/release/vchord-tilemaxsimctl \
  /usr/local/bin/vchord-tilemaxsimctl
```

The build host needs `nvcc`, a C++ compiler, and Rust. The CLI is the canonical
reference:

```bash
vchord-tilemaxsimd --help
vchord-tilemaxsimd --version
vchord-tilemaxsimctl --help
vchord-tilemaxsimctl status --help
```

All memory arguments exposed to operators use GB, interpreted as GiB
(`1024^3` bytes). Raw byte counts are intentionally not accepted for GPU or
host cache sizing.

## Minimal start

```bash
install -d -m 0750 /run/vectorchord
vchord-tilemaxsimd \
  --socket /run/vectorchord/tilemaxsim.sock \
  --gpu-memory-gb 0=20 \
  --gpu-workspace-gb 2 \
  --host-cache-gb 8 \
  --contract-root model@1=/var/lib/vectorchord/tensors/model-v1 \
  --ready-file /run/vectorchord/tilemaxsim.ready \
  --pid-file /run/vectorchord/tilemaxsimd.pid
```

Each `GPU=GB` value is a strict, process-owned allocation. Every configured
arena is allocated before the socket and readiness file are created. A missing
GPU, duplicate device, insufficient free memory, resident-manifest overflow,
or shard validation failure makes startup fail nonzero. There is no silent CPU
fallback and no partial ready state.

The GPU allocation contains both tensor pages and the per-device workspace.
For example, `--gpu-memory-gb 0=20 --gpu-workspace-gb 2` leaves 18 GiB for
cached tensors. `--gpu-block-kib 32` is the default arena page size. Use `lru`
for bounded admission and eviction, or use `resident` with one or more
`--resident-manifest MODEL=PATH` arguments to pin the full declared working set
before readiness.

## PostgreSQL connection

The relevant PostgreSQL settings are ordinary VectorChord GUCs:

```sql
ALTER SYSTEM SET vchordrq.maxsim_backend = 'gpu';
ALTER SYSTEM SET vchordrq.maxsim_gpu_endpoint =
  '/run/vectorchord/tilemaxsim.sock';
ALTER SYSTEM SET vchordrq.maxsim_gpu_timeout_ms = 2000;
ALTER SYSTEM SET vchordrq.maxsim_gpu_max_batch_tokens = 250000;
ALTER SYSTEM SET vchordrq.maxsim_gpu_max_batch_bytes = 268435456;
SELECT pg_reload_conf();
```

Keep PostgreSQL's timeout no greater than the daemon's
`--request-timeout-ms`, or use the same value for both. PostgreSQL also limits
encoded batch tokens and bytes before connecting. Candidate generation,
clustering, graph construction, tenant policy, and application routing remain
outside this daemon.

By default only clients with the daemon's effective Unix UID are accepted.
When PostgreSQL runs under another account, add its numeric UID explicitly:

```bash
--allow-peer-uid "$(id -u postgres)"
```

The socket directory and `--socket-mode` must also permit PostgreSQL to
connect. `SO_PEERCRED` is checked on every connection; an allowed GID refers to
the client's effective primary GID, not a supplementary group. Do not expose
the socket through a network proxy.

## Health, shutdown, and observability

```bash
vchord-tilemaxsimctl status \
  --socket /run/vectorchord/tilemaxsim.sock \
  --ready-file /run/vectorchord/tilemaxsim.ready
```

The status command sends a valid zero-candidate v2 request. It therefore checks
the listener, an I/O worker, the bounded request queue, and the GPU scheduler,
while reading no tensor and launching no scoring kernel. Exit status is `0`
when ready, `1` when unavailable, and `2` for invalid CLI usage.

SIGINT and SIGTERM first remove the ready file and close the listening socket,
then drain accepted work for `--shutdown-grace-ms`. PID, ready, and socket paths
are guarded by device/inode identity so shutdown does not delete a replacement
file. A second instance refuses to unlink an active socket. A core worker panic
or an expired drain deadline produces a nonzero process exit.

Standard output contains one JSON object per lifecycle or request event. Event
names are `tilemaxsim_rust_ready`, `tilemaxsim_rust_prewarm_complete`,
`tilemaxsim_rust_request`, and `tilemaxsim_rust_stopped`; each stable event has
`schema_version: 1`. Request events include queue, compute, total latency, peer
credentials, candidate count, client presence, and cache counters. Send stdout
and stderr to the service manager or a structured log collector.

CUDA kernels cannot be safely preempted in the middle of a launch. Deadlines
and disconnect cancellation are enforced before and between scoring chunks;
one already-running kernel completes before its request is released.

The GPU integration suite includes a configurable soak gate. Release runners
set `VECTORCHORD_TILEMAXSIM_SOAK_SECONDS=3600` (or longer) before running
`python -m unittest services.test_tilemaxsim_rust_daemon`. The soak mixes valid
concurrent requests with incomplete-client disconnects and fails on result,
health, GPU-memory, RSS, FD-recovery, or shutdown regressions. It is skipped
when the variable is absent so ordinary development runs remain short.

## Bounded load behavior

`--max-inflight` bounds clients being read or awaiting a response.
`--backlog` bounds both the kernel listen backlog and the accepted-connection
queue. `--max-queued-requests` bounds parsed work waiting for the single
request scheduler. Full queues return a resource-limit response instead of
growing memory without limit. `--max-request-mb` bounds each encoded frame;
`--max-batch-tokens` and `--max-batch-mb` independently bound the decoded
query-plus-candidate tensor working set before it enters the GPU queue.

Tune PostgreSQL candidate limits and batch limits before increasing daemon
queues. More queue depth absorbs bursts but does not add GPU throughput and
increases tail latency. Multiple configured GPUs execute the chunks of one
request concurrently; requests themselves enter the scheduler in arrival
order.

## systemd

Create a dedicated unprivileged `vectorchord` user, make tensor roots readable
but not writable by that user, and install the public templates:

```bash
install -D -m 0644 services/packaging/vchord-tilemaxsimd.service \
  /etc/systemd/system/vchord-tilemaxsimd.service
install -D -m 0640 services/packaging/tilemaxsimd.conf.example \
  /etc/vectorchord/tilemaxsimd.conf
systemctl daemon-reload
systemctl enable --now vchord-tilemaxsimd
systemctl status vchord-tilemaxsimd
```

Edit the example first. In particular, replace the PostgreSQL UID, GPU size,
contract ID, and tensor path. If PostgreSQL starts immediately after the unit,
keep the `Before=postgresql.service` ordering and make PostgreSQL require this
unit when GPU TileMaxSim is mandatory.

## Container and Kubernetes

The runtime image uses UID/GID `65532`, drops root, and handles SIGTERM. Build
it with:

```bash
docker build -f services/Dockerfile.tilemaxsimd -t vectorchord-tilemaxsimd .
```

Mount the Unix-socket directory read-write, mount immutable tensor shards
read-only, and pass one or more GPUs through the NVIDIA container runtime.
Visible devices are renumbered inside many containers, so a container given
one host GPU usually configures it as `0=GB`.

The Kubernetes example at
[`../packaging/tilemaxsimd-kubernetes.yaml`](../packaging/tilemaxsimd-kubernetes.yaml)
runs the daemon as a PostgreSQL sidecar with a shared `emptyDir`, non-root
security settings, an NVIDIA GPU limit, and startup/readiness/liveness probes.
Replace both example images, the PVC, sizes, contract ID, and PostgreSQL UID.

## Upgrade, restore, and cache recovery

For a planned daemon restart, stop or redirect GPU TileMaxSim queries, send
SIGTERM, wait for a clean exit, replace the binaries or image, and require a
successful status probe before restoring traffic. Do not run two daemon
versions against the same socket path.

Back up PostgreSQL data together with the immutable shard set and the descriptor
manifests referenced by its tensor-source registry. Preserve content hashes,
relative shard metadata, file modes, and model contract IDs. GPU contents, the
decoded host cache, PID files, ready files, and sockets are disposable and must
not be backed up. After restore, validate shard access and start the daemon; LRU
caches refill on demand and resident caches rebuild completely before ready.
