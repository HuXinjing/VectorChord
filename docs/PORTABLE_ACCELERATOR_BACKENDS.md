# Portable accelerator backends

TileMaxSim has one wire protocol, scheduler, L0/L1/L2 cache, immutable tensor
format, quantization contract and management model. Hardware runtimes implement
the Rust `AcceleratorBackend` contract and are built as isolated executors or
images; vendor SDKs are never linked into one universal binary.

## Build boundaries

| Build | Feature | Vendor dependency | Current status |
| --- | --- | --- | --- |
| Shared backend SDK | `backend-core` | none | available |
| CPU reference | `backend-cpu` | none | exact FP16/FP32 available |
| NVIDIA | `backend-cuda` | CUDA Runtime, cuBLAS | available; RTX 4090 validated |
| Apple | `backend-metal` | Metal/Foundation | exact FP16/FP32 experimental; macOS GPU gate required |
| Ascend | downstream `backend-ascend` executor | CANN/Ascend C | implementation pending on Ascend CI |
| MetaX | downstream `backend-metax` executor | MXMACA/mcBLAS | implementation pending on MetaX CI |

The public build publishes separate images from
`services/Dockerfile.tilemaxsimd` and
`services/Dockerfile.tilemaxsimd-cpu`. The latter links no CUDA library and
uses runtime AVX2/FMA on x86-64 or NEON on AArch64. It is the deployable
fallback for hosts whose native accelerator backend is unavailable.

NVIDIA production stages use digest-pinned CUDA runtime images matching their
builder generation (CUDA 12.6 for Ada/Hopper and CUDA 13.3 for Blackwell).
`tilemaxsimd` dynamically links cuBLAS and cuBLASLt, so a plain Ubuntu final
stage is not runnable even though the CUDA driver is injected by the container
runtime. CI now builds the final stage, runs both CLI smoke checks, and rejects
any `ldd` dependency reported as missing; testing only the compiler stage is
not considered image validation.

The CUDA executor also fails during startup when compute capability cannot be
read, is older than SM80, or a Blackwell-class device is paired with a pre-13.0
runtime. The published fatbin has no pre-SM80 code, and the separately tuned
SM120/SM121 artifact is built with CUDA 13; delaying either mismatch until the
first online request would make the advertised capability contract false.

Vendor executors depend on the library without another runtime:

```toml
tilemaxsimd = { path = "../tilemaxsimd", default-features = false, features = ["backend-core"] }
```

They implement the versioned `backend_sdk::BackendProvider` and
`AcceleratorBackend`, then call `daemon::run_with_backend_provider` from their
own executable. The provider API major is checked before resources are
acquired. This is a source-level SDK, not a promise that Rust trait objects are
a stable binary ABI: every vendor executor is compiled against the matching
core crate and links only its own runtime.

Providers must publish truthful `DeviceInfo` and `BackendCapabilities`, and
pass `run_conformance_probe` on the target device.
Unsupported profiles fail before cache mutation or execution; substituting a
different precision is forbidden.

The shared conformance probe executes both FP32 and FP16 exact MaxSim against
known values. If a backend advertises fused multiquery, the same probe submits
two independent FP16 requests in one batch and validates their separate
results. A backend therefore cannot become eligible merely by setting a
capability flag without implementing the corresponding execution path.
Every daemon runs this probe on every configured device before constructing
its cache or opening a scoring listener. A mismatch aborts startup with the
backend slot and device identity; successful reports are emitted as structured
startup events for deployment evidence.

The Apple build is produced independently and never links CUDA:

```bash
cargo build --release --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-metal --bin tilemaxsimd
```

It reserves one `MTLStorageModeShared` arena in Apple unified memory and runs
native MSL FP16/FP32 exact MaxSim. Compatible resident requests share one
command-buffer submission and segmented result reduction. INT8, FP8 and PQ are
reported unsupported rather than silently falling back. The macOS ARM64 CI
job compiles the Objective-C++ bridge and runs the same device-level exact
conformance probe as CUDA. Until that external job passes, Metal remains
experimental rather than production-supported.

## NVIDIA architecture policy

The CUDA 12 artifact contains native `sm_80`, `sm_89`, `sm_90`, `sm_90a` and
forward-compatible `compute_90` PTX. The independent CUDA 13.3 Blackwell image
(`services/Dockerfile.tilemaxsimd-blackwell`) contains native `sm_120` and
`sm_121` plus `compute_121` PTX; SM121 is the GB10/DGX Spark target. Its opt-in
CI job inspects the resulting cubins instead of treating a build flag as proof.
Runtime dispatch uses the device architecture and a
production-shaped 320-dimensional microbenchmark rather than a model-name-only
rule.

- Ada/RTX 4090 uses aligned `half2` loads, FP32 accumulation, scoped
  persisting-L2 query windows, 16-byte `cp.async` document loads,
  32-query-row document-tile reuse, high-priority compute streams and batched
  cuBLAS Tensor Core GEMM. INT8 loads are vectorized and E4M3 decoding uses
  native FP8 conversion instructions on SM89; quantized continuous batches
  decode a document row once into shared memory for all query rows in a tile.
- Hopper/H200 selects the `sm_90a` image. Batched cuBLAS owns Hopper matrix-core
  instruction and data-movement selection, receives an architecture-sized
  32-MiB stable workspace, and starts with 64-query-row document-tile reuse.
  Startup calibration compares 8, 32 and 64 rows and may select a smaller tile
  when it is faster on the actual device. The custom MaxSim reduction remains
  architecture-neutral. Resident batches execute concurrently across devices
  instead of serializing an eight-GPU node. H200 is not marked validated until
  same-device conformance, Nsight and latency tests pass.
- Blackwell/GB10 uses the isolated CUDA 13 SM121 image and the same startup
  numerical calibration. It is loadable and artifact-checked, but it is not
  advertised as tuned until the exact/quantized conformance and latency gates
  run on a physical DGX Spark.
- Cache uploads use the device's least-urgent stream priority and foreground
  scoring uses its most-urgent priority. This does not interrupt an executing
  kernel; scheduler quantum boundaries remain the preemption points.

Repeated Tensor Core batches cache only the bounded host-side mapping from a
candidate's row-count sequence to uniform GEMM groups and reuse the A/B/C
pointer staging vectors. Actual arena offsets are rebuilt and uploaded on every
call, so L0 eviction or relocation cannot leave a stale device address in the
plan. A row-count change invalidates the grouping, and plans above 65,536
candidates are transient to prevent an unusually large request from retaining
unbounded host memory. This removes allocator/map work without changing the
cuBLAS precision mode or cache lifecycle.

The startup calibration uses 16 resident candidates with 32 document rows at
dimension 320. It first selects the fastest numerically equivalent document
reuse tile, then repeats the tile/matrix comparison and records the first
query-row crossover. A failed or divergent variant is rejected; a failed or
divergent matrix path is disabled.

On the available RTX 4090, the production-shaped 64-candidate, eight-request,
32-query-row, 320-dimensional benchmark changed from approximately 0.477 ms to
0.112 ms for the fused tile path (about 4.26x). The cuBLAS path measured about
0.112--0.114 ms, so calibration correctly kept the tile path for that shape.
This is a kernel microbenchmark, not an end-to-end retrieval latency claim.
Generated `sm_89` SASS was inspected and contains the asynchronous copy
and native E4M3 conversion instructions; compilation alone is not counted as
evidence that the fast path exists. OPQ rotations are materialized once per
query and rotation stage before LUT generation instead of being recomputed for
every centroid, reducing the rotation arithmetic by the centroid count while
preserving the persisted quantization contract.

Quantization is not advertised as an unconditional resident-kernel speedup.
On the same RTX 4090 with 512 resident candidates, 32 query rows and dimension
320, exact FP16 measured about 0.110 ms, INT8 0.118 ms and FP8 0.119 ms. For an
eight-request fused batch the corresponding measurements were about 0.571 ms,
0.605 ms and 0.604 ms. At this working-set size, native conversion still costs
roughly 6--8% more than the bytes it saves. INT8/FP8 remain useful for fitting
more tensors in L0 and avoiding L1/L2 misses; the caller must choose them under
an explicit accuracy/storage contract rather than expecting every resident
shape to run faster. Larger cold/cache-pressure experiments remain a separate
acceptance gate.

PQ, OPQ and residual-PQ also participate in continuous batching when requests
share the same immutable quantization contract and candidate set. The native
path builds rotation/LUT state for the combined query rows, performs one ADC
MaxSim scan, then applies a segmented reduction at the original request
boundaries. It never merges different contracts. On the available RTX 4090,
512 resident candidates, eight requests, 32 query rows and dimension 320 took
about 1.083 ms as eight individual PQ calls and 0.912 ms as one continuous
batch (1.19x). This run occurred on a shared GPU and is retained as directional
microbenchmark evidence; an exclusive-device distribution is still required
for a release performance claim.

Nsight attribution showed ADC-MaxSim consuming about 82.5% of PQ batch kernel
time on RTX 4090. A second exact ADC mapping therefore assigns one short
document task to each warp and removes block-wide barriers, while the original
eight-warp cooperative mapping remains selected for documents with more rows.
A back-to-back 512-candidate sweep placed the crossover between four and eight
document rows: the warp-task path reduced the two-row batch from about 0.275 ms
to 0.176 ms (1.56x) and the four-row batch from 0.291 ms to 0.251 ms (1.16x),
but regressed at eight rows. The value is not hard-coded across architectures:
startup repeats both kernels at two, four and eight rows with 512 candidates,
checks their scores, requires a median improvement greater than 5%, and exposes
the selected maximum through `pq_warp_task_max_document_rows`. The available
RTX 4090 selected four rows. Calibration failure safely selects the cooperative
kernel. The H200 hardware gate records its independently selected value before
the path can be described as Hopper-tuned.

## Vendor acceptance gates

Every native backend needs target-hardware evidence for:

1. exact FP32 and FP16 MaxSim against the CPU oracle;
2. declared quantized profiles and immutable-contract failure cases;
3. variable document/query rows, empty/oversized and alignment boundaries;
4. concurrent priority, deadline, tenant reservation and cache churn;
5. driver/runtime version reporting and unsupported-version startup failure;
6. cold L2→L1→L0 and resident latency, p50/p95/p99, sustained concurrency;
7. clean device-loss and out-of-memory handling without corrupting daemon state.

Passing a compile or exposing a device name is not sufficient to advertise a
backend as production-supported.

The manually dispatched `tilemaxsimd-hardware-validation.yml` workflow is the
H200 release gate. Its self-hosted runner must carry the `gpu-h200` label. The
runner script refuses a device whose name or compute capability is not H200 /
9.0, builds native SM90/SM90a code, inspects the cubins and instructions, runs
all ignored real-device numerical and latency tests serially, and uploads the
device fingerprint, raw output and checksum manifest. It additionally records
the runtime-selected cuBLAS kernel with Nsight Systems and fails unless Nsight
Compute observes activity on the SM tensor pipeline. This matters because the
cuBLAS implementation is selected from the installed library at runtime and is
not contained in this repository's cubin. A queued or skipped job is not
validation evidence.
