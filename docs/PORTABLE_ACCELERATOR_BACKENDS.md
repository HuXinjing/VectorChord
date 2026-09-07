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

Vendor executors depend on the library without another runtime:

```toml
tilemaxsimd = { path = "../tilemaxsimd", default-features = false, features = ["backend-core"] }
```

They must implement `AcceleratorBackend`, publish truthful `DeviceInfo` and
`BackendCapabilities`, and pass `run_conformance_probe` on the target device.
Unsupported profiles fail before cache mutation or execution; substituting a
different precision is forbidden.

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
