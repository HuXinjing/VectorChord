# VectorChord CUDA TileMaxSim Sidecar

## Scope

`services/tilemaxsim_cuda_sidecar.py` is the deployable CUDA executor for IPC
v1 and v2. PostgreSQL remains responsible for candidate generation, MVCC row
visibility, descriptor validation, and final public-ID mapping. The sidecar
performs bounded tensor resolution and exact TileMaxSim only.

The service does not fetch from S3, HTTP, or another network destination.
GBrain may populate a node-local immutable cache from its storage system. This
keeps storage credentials, application authorization, and routing outside the
VectorChord extension while retaining one VectorChord-owned retrieval call.

Production acceptance still requires an application-level corpus benchmark.
The service implementation alone is not that acceptance result.

## Runtime Requirements

- Linux with a Unix-domain socket;
- PyTorch built for the installed NVIDIA driver/CUDA runtime;
- one CUDA device by default (`--device cuda:0`);
- a service account that can read configured cache roots and create the socket;
- the PostgreSQL service account must be able to connect to the socket.

CPU mode exists for conformance tests only:

```text
python3 -m services.tilemaxsim_cuda_sidecar --device cpu ...
```

The production command must select a CUDA device. Startup performs a CUDA
matrix multiply and synchronization, so an unusable device fails before the
socket is advertised.

For container deployment, build the minimal service image on top of the
operations-approved PyTorch/CUDA runtime that matches the node driver:

```text
docker build \
  --build-arg BASE_IMAGE=<approved-pytorch-cuda-runtime-image> \
  -f services/Dockerfile.tilemaxsim \
  -t vectorchord-tilemaxsim:local \
  .
```

The Dockerfile intentionally has no default base image. CUDA/PyTorch/driver
compatibility is a deployment input and must not drift through an implicit
`latest` tag.

## Content-Addressed Tensor Cache

Each model contract is mapped explicitly to one absolute cache root. A v2
reference has exactly this form:

```text
sha256://0123456789abcdef...64-lowercase-hex-characters
```

For digest `abcdef...`, the canonical row-major scalar payload is stored at:

```text
<contract-root>/ab/abcdef....bin
```

The registered checksum must be `sha256:<same-digest>`. The service verifies:

- exact model-contract root mapping;
- reference/checksum agreement;
- regular file and exact descriptor byte length;
- SHA-256 content digest;
- finite f16/f32 values;
- descriptor shape and dtype.

Directory and file symlinks below the configured root are rejected. Cache
publishers should write a temporary file, flush it, and atomically rename it to
the digest path only after the complete payload is available. Existing digest
paths are immutable.

Example SQL descriptor values:

```text
tensor_ref      = sha256://0123...cdef
tensor_checksum = sha256:0123...cdef
tensor_rows     = 747
tensor_dim      = 320
tensor_dtype    = float16
```

## Start the Service

```text
python3 -m services.tilemaxsim_cuda_sidecar \
  --socket /run/vectorchord/tilemaxsim.sock \
  --socket-mode 660 \
  --device cuda:0 \
  --contract-root 'colqwen3.5@revision+preprocessing-hash=/var/cache/gbrain/colqwen35' \
  --request-timeout-ms 2000 \
  --max-request-bytes 1073741824 \
  --max-tensor-bytes 1073741824 \
  --max-batch-tokens 1000000 \
  --max-device-bytes 8589934592 \
  --cache-bytes 8589934592 \
  --max-inflight 8 \
  --max-cuda-inflight 1
```

Run the process under an operations-managed supervisor. The socket directory,
process user/group, cache roots, CUDA device visibility, and resource limits
belong in that supervisor configuration. `SIGINT` and `SIGTERM` stop new
accepts, drain accepted work, and remove only the socket inode created by the
process.

The sidecar and PostgreSQL limits must agree. IPC v1 embeds tensor payloads, so
`--max-request-bytes` must accommodate the inline batch. IPC v2 carries only
descriptors but independently accounts for all declared canonical tensor bytes.
PostgreSQL's `vchordrq.maxsim_gpu_timeout_ms` and the service request timeout
both cover the complete request; operations should leave enough margin for
normal socket scheduling without allowing abandoned work to run indefinitely.

## Batching, Backpressure, and Failure Semantics

- The listener accepts at most `--max-inflight` connections. Further clients
  remain in the bounded Unix-socket backlog and are governed by their
  PostgreSQL-side deadline.
- `--max-cuda-inflight` bounds simultaneous work on one device. Waiting for a
  slot consumes the same overall request deadline.
- A request may be split into device-memory-bounded candidate groups. Results
  are returned only after every group succeeds.
- Peer disconnect and deadline checks occur before every device group.
- CUDA OOM, missing/corrupt tensors, unsupported references, timeout, and
  non-finite scores fail the complete response. Partial candidate results are
  never returned.
- f16 input scalars are promoted to f32 before matrix multiplication. TF32 is
  disabled by default; `--allow-tf32` must be enabled only after corpus-level
  score/ranking tolerance is accepted.

The service emits one-line JSON events to standard output. Request events
contain request ID, protocol version, source kind, dimensions, candidate and
token counts, content-cache hits, resolution time, CUDA queue time, compute
time, total time, status, and the Unix peer PID/UID/GID where supported. The
peer PID plus request ID disambiguates PostgreSQL backend-local request
counters. Tensor references and public/application IDs are not logged.

## Verification and Load Probe

Protocol, resolver, deadline, device-chunking, live-socket, CPU-equivalence,
and optional CUDA-equivalence tests:

```text
python3 -m unittest -v \
  devtools/test_tilemaxsim_reference_sidecar.py \
  services/test_tilemaxsim_cuda_sidecar.py
```

Synthetic ColQwen-shaped load probe:

```text
python3 -m services.benchmark_tilemaxsim_cuda \
  --device cuda:0 \
  --dtype f16 \
  --dimension 320 \
  --query-rows 32 \
  --document-rows 747 \
  --candidates 256 \
  --warmup 3 \
  --iterations 20
```

The command emits reproducibility metadata, latency percentiles, peak CUDA
allocation/reservation, and a deterministic score checksum as JSON. It is a
runtime smoke/load probe, not a substitute for recall and end-to-end latency on
the committed GBrain corpus.
