# Durable GPU working-set recovery

`--cache-recovery-state /private/persistent/cache.json` enables recovery of the
logical LRU working set after process, CUDA-context, container or host failure.
This is not a fixed startup prewarm manifest and does not pin otherwise evictable
entries. The immutable tensor roots and the checkpoint directory must both
survive the failure.

## Contract

The scheduler commits an atomic checkpoint before acknowledging completed
scoring or administrative work that changes residency or pin state. Cache hits
with unchanged membership and pin state do not rewrite it. A checkpoint records
content-addressed descriptors, scoring/quantization contracts, query dtype,
actual owning tenant and pin flags. It never records device pointers, reference
counts, in-flight jobs, live connections or authentication credentials.

Restart validates the checkpoint checksum/schema and the original immutable
payloads, then uses the existing preparation/upload and quota paths. It restores
all recorded entries and pin flags before opening the serving listeners or
advertising readiness. Current GPU capacity, tenant reservations, tenant limits,
pin limits and active quantizer contracts still apply. No quota is increased or
permanent pin introduced to force a restore to fit.

The durability boundary is the last committed completed-work state. Mutations
from work still in flight or cancelled before that boundary can be lost.
Replacement-policy frequency sketches, access ordering, physical GPU addresses,
host-cache contents and performance counters are not restored. A different
multi-GPU layout may redistribute entries; it is not a byte-identical process
snapshot. The subsequent cache remains evictable under its normal policy.

## Files and failure handling

The directory must be canonical, owned by the daemon UID and not writable by
other users. Mount it on durable storage and provision it with mode 0700 before
starting the daemon. State and the exclusive lifetime lease use mode 0600.
Publication writes a bounded temporary file, fsyncs it, atomically renames it,
then fsyncs the parent directory. An orphaned temporary file is never authoritative.
The lease is released by the kernel even on SIGKILL. A second daemon cannot own
the same checkpoint concurrently.

A malformed checkpoint, invalid owner/contract, missing or corrupt tensor,
insufficient quota/capacity, or inability to persist a new checkpoint fails
closed. Startup failures do not advertise readiness; runtime persistence failures
reject the response and stop the scheduler. Preserve the state for diagnosis.
For an intentional corpus retirement or configuration reduction that invalidates
the checkpoint, an operator must explicitly archive the checkpoint while the
service is stopped and choose to establish a new working set. Do not silently
reset it or disable validation. Coordinate tensor garbage collection accordingly.

The format is versioned and checksummed, capped at one million entries and
128 MiB. Metadata tracking is also bounded. The checkpoint contains internal
contract/tenant identifiers; do not publish it or expose it through public APIs.
The scoring wire protocol is unchanged. Recovery is opt-in and LRU-only.

## Deployment and rollback

Keep the checkpoint outside an emptyDir. Use single-replica Recreate when the
old and new GPU allocations cannot coexist. Add a startup probe whose budget
covers the measured restore time; otherwise a liveness probe can repeatedly
kill an otherwise valid long restore. Readiness must remain unavailable until
restoration finishes. Preserve the previous image, arguments and storage.

A version without recovery cannot export the complete resident set. The first
upgrade therefore needs a controlled initial workload to establish the first
checkpoint. This migration boundary must not be described as recovery of an
unrecorded old state. After that checkpoint exists, validate with a real restart
and an identical request, comparing scores and additional storage-read/H2D
counters. Do not clear global OS/GPU caches or disturb unrelated services.

Rollback removes the new flag and restores the previous image/configuration;
retain checkpoint files for diagnosis. The older daemon does not consume them.
No database migration is required.

Events `tilemaxsim_cache_checkpoint_committed` and
`tilemaxsim_cache_recovery_complete` include entry count and elapsed milliseconds.
Checkpoint time is part of completed-request delivery, not GPU compute time.

## Verification

From the repository root:

```sh
cargo test --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-cpu --all-targets --locked
cargo build --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-cpu --locked
python3 services/tilemaxsimd/scripts/test_cache_recovery_process.py \
  --cpu-binary services/tilemaxsimd/target/debug/tilemaxsimd
```

The process test uses SIGKILL, two tenants, real content-addressed objects,
identical before/after scores, no post-restore reads/uploads for cache hits,
checkpoint corruption and runtime persistence failure. CPU tests are lifecycle
proof, not CUDA correctness or performance evidence. Build the formal CUDA test
image and separately validate the target GPU deployment before acceptance.

## Building from an existing local CUDA installation

`services/Dockerfile.tilemaxsimd.offline` is an alternative full source build.
It uses pinned Rust/Ubuntu base images, a read-only CUDA toolkit BuildKit context
and a read-only Cargo registry context. It does not inherit application binaries
or layers from a preceding TileMaxSim image. Both daemons are compiled anew;
the runtime copies only the matching CUDA shared libraries and those binaries.
The normal NVIDIA-image Dockerfile remains available.

```sh
docker build --target test -f services/Dockerfile.tilemaxsimd.offline \
  --build-context cuda-toolkit=/path/to/cuda-toolkit \
  --build-context cargo-registry=/path/to/cargo/registry \
  -t tilemaxsimd:verified-build .
```

Use the same contexts for the runtime build. Record `nvcc --version`, hash the
toolkit input manifest, and pass `SOURCE_REVISION`, `SOURCE_BRANCH` and
`CUDA_TOOLKIT_SHA256` as build arguments. Keep compiler and runtime libraries
from the same verified toolkit; do not select a compiler solely because its
layers happen to be cached. A compiler-version upgrade needs its own validation.
The GPU driver version reported by nvidia-smi is not the application's CUDA
runtime version. This alternative avoids downloading a duplicate multi-gigabyte
CUDA development image on an already provisioned build machine. Rust test tools
and small Ubuntu runtime dependencies may still need network access.

## Bounded parallel tensor loading

`--tensor-read-workers` configures immutable object/shard read and checksum
workers (default 8, configurable 1 through 256, capped by available CPU
parallelism). It is a loader concurrency budget, not a cgroup CPU limit.
File identity/length/checksum validation remains mandatory. Loose object reads
fill one shared allocation directly; packed reads avoid an intermediate payload
copy. Cache access/admission mutations for loose objects retain descriptor order,
including error-prefix behavior. Existing packed-shard admission ordering remains
unchanged. Speculative reads can avoid repeated I/O for duplicate immutable
objects; storage counters include verified reads actually performed.

Host-cache eviction uses global and per-owner priority indexes. Its density/age
policy, tenant/global byte limits and rejection rules remain unchanged. Equal
priorities now use a deterministic content-key tie-break, rather than HashMap
iteration order. Cache eviction does not invalidate payloads already borrowed by
a request. No forced pinning, quota increase or checksum bypass is involved.

A CPU-only comparison on 30.100 used the same frozen checkpoint with 45,438
entries and 21,720,627,200 payload bytes, a 16 GiB host cache and the existing 80%
per-owner limit. Verified reads took 40.60 / 18.33 / 12.97 / 9.23 seconds with
1 / 4 / 8 / 16 workers respectively. These are read/validation/cache timings on
a shared host, not GPU recovery or Agent end-to-end timings. The deployment uses
16 workers. Its container and cgroup ancestors had no CPU quota, no throttling,
and 192 available logical CPUs; no CPU hard limit was introduced.
