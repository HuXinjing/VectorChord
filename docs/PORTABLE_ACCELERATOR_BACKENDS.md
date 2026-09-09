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
| NVIDIA | `backend-cuda` | CUDA Runtime, cuBLAS | available; RTX 4090 validated; H200 functional gate passed, profiling gate pending |
| AMD / Vulkan | `backend-vulkan` | Vulkan loader and hardware ICD | experimental exact FP16/FP32; see [AMD/WSL guide](TILEMAXSIM_AMD_VULKAN.md) |
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
runtime. Before publication, operators must build the final stage, run both CLI
smoke checks, and reject any `ldd` dependency reported as missing; testing only
the compiler stage is not considered image validation.

The CUDA executor also fails during startup when compute capability cannot be
read, is older than SM80, or a Blackwell-class device is paired with a pre-13.0
runtime. The published fatbin has no pre-SM80 code, and the separately tuned
SM120/SM121 artifact is built with CUDA 13; delaying either mismatch until the
first online request would make the advertised capability contract false.
Non-critical device telemetry is queried through runtime attributes shared by
CUDA 12 and CUDA 13. In particular, CUDA 13 removed the legacy
`cudaDeviceProp::memoryClockRate` field; an unavailable clock is now reported
as unknown instead of preventing an otherwise compatible H200/Blackwell
executor from compiling or starting.

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
reported unsupported rather than silently falling back. A macOS ARM64 release
validation must compile the Objective-C++ bridge and run the same device-level
exact conformance probe as CUDA. Until that validation passes, Metal remains
experimental rather than production-supported.

## NVIDIA architecture policy

The CUDA 12 artifact contains native `sm_80`, `sm_89`, `sm_90`, `sm_90a` and
forward-compatible `compute_90` PTX. The independent CUDA 13.3 Blackwell image
(`services/Dockerfile.tilemaxsimd-blackwell`) contains native `sm_120` and
`sm_121` plus `compute_121` PTX; SM121 is the GB10/DGX Spark target. Runtime
dispatch uses the device architecture and a
production-shaped 320-dimensional microbenchmark rather than a model-name-only
rule. Release validation must inspect the resulting cubins instead of treating
a build flag as proof.

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
  instead of serializing an eight-GPU node. Physical H200 compilation,
  conformance and latency microbenchmarks have passed; the Nsight attribution
  and full cold-cache/concurrency gates remain required before H200 is marked
  production-validated.
- Blackwell/GB10 uses the isolated CUDA 13 SM121 image and the same startup
  numerical calibration. It is loadable and artifact-checked, but it is not
  advertised as tuned until the exact/quantized conformance and latency gates
  run on a physical DGX Spark.
- Cache uploads use the device's least-urgent stream priority and foreground
  scoring uses its most-urgent priority. This does not interrupt an executing
  kernel; scheduler quantum boundaries remain the preemption points.

Exact, quantized and PQ requests pack their query and descriptor control data
into the arena's persistent pinned-host staging region before one asynchronous
H2D submission. Because the extra host copy is not profitable for every PCIe,
NUMA and integrated topology, startup compares the packed and direct-pageable
paths on the actual device, checks numerical equivalence and retains packing
only after a median improvement of at least 2%. Oversized control payloads
fall back to bounded direct copies rather than increasing pinned memory. The
selected path is exposed as `pinned_control_staging` in device status.
Both variants are warmed before measurement and subsequent samples alternate
AB/BA order, preventing clock ramp or cache warmth from systematically
favouring the second implementation.
The same record exposes `control_staging_speedup_milli` (direct median divided
by packed median, in thousandths), so operators can audit a device-specific
choice instead of inferring it from the GPU model name.
Prometheus exports the selected tile width, PQ short-document crossover,
pinned-control decision and measured ratio, persisting-L2 reservation and
matrix-engine workspace under the bounded `tilemaxsim_gpu_tuning_value`
metric. No model name or tenant identifier is used as a label.

SM80+ exact tile kernels also contain a two-stage shared-memory pipeline that
can enqueue row N+1 with `cp.async` while warps score row N. It is not enabled
merely because the instruction exists: startup compares single and double
buffering at the production dimension, rejects score drift, requires at least
a 5% median win and checks that two rows fit in the device's per-block shared
memory. `double_buffered_tile` and `double_buffer_speedup_milli` expose the
decision. This keeps Ada on the simpler path when synchronization dominates,
while allowing Hopper to select overlap only when its physical H200 result
supports it.
With the 64-candidate startup shape, the shared RTX 4090 measured a 1.003
single/double ratio during calibration; a separate eight-request run measured
1.007. Both are below the 1.05 admission threshold, so Ada retained the
single-buffer kernel. This rejected experiment is intentionally kept behind
the calibrated dispatch rather than being presented as an optimization win.

On the shared RTX 4090, three back-to-back 512-candidate, 32-query-row,
320-dimensional A/B runs measured direct pageable control transfers at
0.1154--0.1204 ms and pinned packed transfers at 0.1045--0.1066 ms, a
1.10--1.13x reduction for this kernel call. The benchmark validates identical
scores and never asserts that packing must win, because topology and driver
behaviour differ; the H200 gate runs the same A/B benchmark and records its own
decision.

Repeated Tensor Core batches cache only the bounded host-side mapping from a
candidate's row-count sequence to uniform GEMM groups and reuse the A/B/C
pointer staging vectors. Actual arena offsets are rebuilt and uploaded on every
call, so L0 eviction or relocation cannot leave a stale device address in the
plan. A row-count change invalidates the grouping, and plans above 65,536
candidates are transient to prevent an unusually large request from retaining
unbounded host memory. This removes allocator/map work without changing the
cuBLAS precision mode or cache lifecycle.

The startup calibration first selects the fastest numerically equivalent
document-reuse tile, then measures matrix crossover points at 64, 512 and 4,096
resident candidates. Each candidate bucket covers one, eight and 32 distinct
document-row groups and 64--2,048 aggregate query rows at dimension 320. Buckets
that do not fit the configured tensor arena or workspace are skipped rather
than borrowing unreserved VRAM. Within each viable bucket the grouped-GEMM
candidate chunk is selected at that bucket's measured query-row crossover from
64/256/1,024/4,096 and the bucket size, bounded by the native workspace.
Selecting it at a fixed unrelated query shape is specifically avoided because
matrix scratch per candidate changes with batch rows. Runtime normalizes
candidate work by total document rows, selects the nearest calibrated grouping profile, scales for dimension and
uses the measured query-row crossover. A failed or divergent bucket is rejected;
a runtime matrix failure disables the matrix path for that device and retries
through the established tile path.

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

## Physical H200 validation (2026-09-07)

Commit `2fb9dbe` was built offline with CUDA 13.0 for native `sm_90`, `sm_90a`
and `compute_90` on one NVIDIA H200 (compute capability 9.0, driver 595.91.07).
The GPU was shared with an existing workload; the test used only otherwise free
memory and did not stop or reconfigure that workload. All 17 ignored CUDA
device tests passed, covering the backend conformance probe, exact FP16/FP32,
fused multiquery, INT8, FP8, PQ, OPQ, residual PQ, variable-query Tensor Core
equivalence, continuous batching and cache-safe quantizer reclamation.

Five independent startup-calibration runs made the same choices:

- document-tile query rows: 8;
- PQ warp-task maximum document rows: 8;
- pinned control staging: enabled, with measured median ratios of 1.053--1.070;
- double-buffered document tiles: disabled, with ratios of 1.028--1.042, below
  the 1.05 admission threshold.

The final complete run measured the 64-candidate, eight-request Tensor Core
path at 0.1334 ms versus 0.1460 ms for the selected single-buffer tile path
(1.094x). Earlier repeated isolated A/B probes measured 1.155--1.162x; the
spread is reported because the GPU was shared, so these numbers are evidence
for dispatch direction rather than a production latency SLO. The same complete
run measured PQ continuous batching at 0.6803 ms versus 0.9862 ms for eight
individual calls (1.450x). An eight-request INT8/FP8 batch measured 1.018x and
1.019x versus exact FP16 respectively, while single-request quantized scoring
was slower; this supports retaining quantization as an explicit capacity and
accuracy contract rather than enabling it as an unconditional latency mode.

The direct double-buffer benchmark regressed to 0.898x in the final run. The
paired AB/BA startup calibration therefore correctly retained single buffering
instead of applying a Hopper model-name rule. GPU memory before and after the
validation was identical (65,837 MiB used, 77,322 MiB free), and all transferred
source/toolchain files were removed from the remote host.

This run does **not** close the complete H200 release gate. At the time of the
run the host did not provide `nvdisasm`, Nsight Systems or Nsight Compute, and
no packages were installed. These tools were subsequently installed in an
isolated user-owned directory without changing the driver: `nvdisasm` decoded
an SM90a cubin, Nsight Systems captured ten SM90 Tensor Core GEMMs, and a
sudo-scoped Nsight Compute probe measured 41.33% Tensor Pipeline activity.
The full VectorChord workflow has not yet been rerun under those profilers, so
its CUDA 13 cubin instruction audit and kernel-specific hardware-counter proof
remain pending, as do end-to-end
L2-to-L1-to-L0 cold-cache distributions and sustained multi-tenant p50/p95/p99
tests on an exclusive device. Any future release gate must fail closed when
those tools or artifacts are absent.

### Resident concurrency sweep

A follow-up synthetic FP16 sweep on the same shared H200 used 32 query rows per
request, 32 rows per document, dimension 320 and resident L0 tensors. The table
reports single-buffer tile time divided by explicit Tensor Core time; values
above one favour Tensor Core. Each point through 4,096 candidates used 20 timed
iterations. The 34,054-candidate points used five timed iterations, and its
8/16/32/64-request points were repeated three times.

| Candidates | 2 requests | 8 requests | 16 requests | 32 requests | 64 requests |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.58x | 1.10x | 1.56x | 2.14x | 2.59x |
| 512 | 1.52x | 4.40x | 6.65x | 6.97x | 7.59x |
| 4,096 | 5.27x | 9.53x | 10.52x | 10.69x | 10.97x |
| 34,054 | 7.56x | 10.67x | 13.82x | 15.53x | 16.11x |

For 34,054 candidates, the median tile/Tensor times across the repeated large
batches were 41.5824/3.8990 ms at eight requests, 83.1257/5.9925 ms at 16,
165.1969/10.6387 ms at 32 and 329.3798/20.4426 ms at 64. The 64-request speedup
was tightly grouped at 15.989--16.130x despite the shared device.

This result exposed a dispatch-model gap: the former startup probe used 64
candidates and left `tensor_threshold_rows` disabled (`u32::MAX`) even though
explicit matrix dispatch won strongly once candidate count or concurrency grew.
The current implementation addresses that gap with the multi-bucket work and
grouping model described above. Dispatch compares measured crossover
dot-product work with the actual query rows, total document rows and dimension,
while row-count cardinality, maximum document rows and row-imbalance ratio
select the nearest calibrated shape profile. The daemon's
default candidate quantum is derived from the largest useful bucket common to
every active device; a non-zero operator setting overrides it. On RTX 4090 it retained tile for 64 candidates
and two requests, selected Tensor Core for 64 candidates and 16 requests, and
selected Tensor Core for 512 candidates at both two and eight requests; each
choice agreed with the directly measured faster path. On the physical H200 the
same automatic policy retained tile at 64 candidates/two requests (Tensor Core
was only 0.59x), then selected Tensor Core at 512/two (1.62x), 4,096/two
(5.92x), 34,054/eight (11.12x) and 34,054/64 (15.82x). The last two shapes
selected a 4,096-candidate grouped-GEMM chunk within their configured workspace.
A deliberately tested fixed-512-query-row chunk calibration had selected 1,024
candidates and regressed the 34,054/eight Tensor path to 12.75 ms; calibrating
the chunk at the bucket crossover restored it to 3.72 ms. This rejected design
is recorded to prevent reintroducing a global tile-size assumption. The sweep
was followed by the complete H200 CUDA suite, with all 17 real-device tests
passing. It excludes scheduler queueing, H2D cache misses, PostgreSQL candidate generation and network
latency, and its repeated synthetic document shape is more GEMM-friendly than a
variable-length production corpus.

Native Tensor failures are classified at the ABI boundary. Request and
workspace-capacity failures fall back without disabling unrelated shapes. A
bucket retries after 60 seconds and doubles its cooldown up to 30 minutes after
each failed half-open probe. Only three consecutive CUDA/cuBLAS/device failures
open a device-wide circuit; its cooldown starts at 30 seconds and doubles up to
10 minutes. Successful Tensor work closes and resets the relevant backoff.
Only rate-limited closed/open/half-open transitions are logged, while suppressed
log events and every failure remain visible through cumulative Prometheus
counters. Management JSON and Prometheus expose each class separately. The retained
`adaptive_tensor_threshold_rows` metric is a deprecated minimum-over-buckets
summary, not a global dispatch threshold.

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

The repository provides `services/tilemaxsimd/scripts/validate_nvidia_hardware.sh`
for an operator-run H200 release gate. The script refuses a device whose name or
compute capability is not H200 / 9.0, builds native SM90/SM90a code, inspects
the cubins and instructions, and runs real-device numerical and latency tests.
Release evidence should retain the device fingerprint, raw output and checksum
manifest. It must also record the runtime-selected cuBLAS kernel with Nsight
Systems and require Nsight Compute activity on the SM tensor pipeline. This
matters because the cuBLAS implementation is selected from the installed
library at runtime and is not contained in this repository's cubin. An omitted
or partial run is not validation evidence.
