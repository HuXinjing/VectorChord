# Hopper fused FP16 scoring

The CUDA service provides an opt-in Hopper WGMMA path for exact FP16 batches:

```text
TILEMAXSIM_HOPPER_FUSED=1
```

The default is disabled. Values other than `0` and `1` fail startup. Enabling it
requires an SM90 device and a build containing the `90a` image. Other backends,
precision profiles and single-request paths are unchanged. Build with the normal
CUDA architecture list to retain portable images; `90a` alone is appropriate only
for a specifically scoped Hopper deployment.

The kernel combines two 64-row query tiles with a 128-row document tile, prefetches
two stages, and computes local document maxima in registers. It writes partial
maxima instead of the full similarity matrix. Final merging preserves query order.
Document padding is excluded before max reduction, including negative scores.
No BF16/FP8 conversion, precision substitution or graph cache is introduced.

Query boundaries, document ranges/alignment, output sizes, grid bounds and arena
workspace are checked before submission. Unsupported shapes and capacity declines
use the existing dispatcher. After submission, all error paths drain the stream;
a device error is returned without scoring again through another implementation.
A submitted-operation error disables the new path until restart, and an open
existing Tensor device circuit also prevents new-path submission.
The existing worker keeps exclusive ownership of the arena through completion.
Tenant authorization, candidate admission, cache generations, quotas, deadlines,
leases and eviction checks still happen in the existing engine/scheduler.

The native descriptor cache stores addresses and shapes only. Its key includes
all candidate offsets/row counts and the dimension, and every call repeats arena
bounds checks. It does not cache permissions, residency, tensor values or scores.
It is bounded to 65,536 entries (roughly 9 MiB metadata per GPU). Replacement keys
are published only after successful descriptor creation. Device control metadata
is uploaded on every call, so workspace reuse by other algorithms is safe.

Status cache device entries expose `hopper_fused_enabled`, `hopper_fused_calls`
and `hopper_fused_declines`. Startup emits `tilemaxsim_hopper_fused_enabled`.
Observe actual call counts; an enabled flag alone does not prove that a workload
uses this path. Declines can occur during startup backend conformance checks.

Roll out with a separately scoped, bounded-memory candidate daemon or container.
Compare the same frozen query bytes, candidate identities and tensor checksums
against the existing service through the real SQL/codec path. Preserve arguments,
volumes, model contracts, tenant boundaries and non-target services. Record image
and binary hashes plus rollback configuration before changing the deployed image.
Removing the environment flag restores the previous dispatcher; retaining the
previous image provides binary rollback. Do not run old/new database containers
against the same data volume simultaneously.

The standalone experiments found a roughly 35% native-throughput gain for wider
tiles relative to the prior fused kernel, and a larger combined gain with host
buffer reuse/graph experiments. These are not service QPS or tenant concurrency
claims. The production path intentionally omits graph caching until its independent
benefit justifies graph lifetime and resource management complexity.

The WGMMA instruction wrapper is adapted from NVIDIA CUTLASS/CuTe; the retained
BSD-3-Clause license is in `native/CUTLASS_LICENSE.txt` under the service directory.
