# Experimental AMD Vulkan executor

`backend-vulkan` adds exact MaxSim compute through Vulkan using wgpu. It is
independent of CUDA and ROCm. AMD on native Linux can use Mesa RADV; WSL2 needs
a Vulkan-to-D3D12 driver such as Mesa Dozen (`dzn`). Installing ROCm or merely
having `/dev/dxg` does not supply that Vulkan driver.

## Capability and limits

- FP32 and packed FP16 tensors, with FP32 dot products and reductions.
  Native shader FP16 and matrix instructions are not required.
- One 64-thread workgroup per candidate/query row; query-row maxima are summed
  on the host. Batch requests execute sequentially, not as fused multiquery.
- INT8, FP8, PQ and OPQ/RPQ are explicitly unsupported.
- One resident tensor buffer. Its size must fit the adapter's storage-buffer
  binding limit, which can be much smaller than total VRAM. Requests also must
  fit the configured workspace and device dispatch limits.
- Uploads use four-byte-aligned offsets, compatible with the daemon's cache
  allocator. Odd FP16 payloads are padded for Vulkan copies.
- Hardware adapters only in the executable. A software Vulkan driver is never
  silently substituted for an unavailable GPU.

This is an experimental correctness implementation, not a tuned replacement
for the CUDA executor. Performance and long-running concurrent workloads must
be measured on the deployment GPU before production use.

## Build and test

From the repository root:

```bash
cargo build --release --locked --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-vulkan
cargo test --locked --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-vulkan
cargo run --release --locked --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-vulkan --example vulkan_probe
```

The probe requires a real hardware Vulkan adapter and executes actual FP16 and
FP32 upload/scoring. Its JSON includes adapter identity and the numerical result.
The daemon performs the same conformance check before opening its listeners.

The more extensive device gate includes a scalar reference at dimensions 1,
37, 320 and 513, odd FP16 payloads, negative maxima, batches and invalid inputs:

```bash
cargo test --locked --manifest-path services/tilemaxsimd/Cargo.toml \
  --no-default-features --features backend-vulkan \
  vulkan_conformance_and_edge_cases -- --ignored --nocapture
```

For shader regression testing without hardware, explicitly set
`TILEMAXSIM_VULKAN_TEST_SOFTWARE=1` and select Lavapipe with `VK_DRIVER_FILES`.
That switch exists only in the unit test and does not enable software fallback
in the daemon. Software results are not hardware acceptance evidence.

## RX 6400 on WSL2

The target machine has an RX 6400 with approximately 4 GB VRAM and WSL2.
RX 6400 is absent from AMD's [ROCm 7.2.1 WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html).
This backend therefore uses Vulkan, not an unverified ROCm architecture override.

Before startup, `vulkaninfo --summary` must show the real RX 6400. Ubuntu's Mesa
package may contain RADV and Lavapipe but omit Dozen. RADV requires native DRM
GPU access and cannot use WSL's `/dev/dxg`; Lavapipe is CPU rendering. A working
Dozen ICD, Windows D3D12 driver, `/dev/dxg`, and the WSL libraries under
`/usr/lib/wsl` are required for this path. Keep locally built Dozen isolated and
select its JSON using `VK_DRIVER_FILES`; do not replace system Mesa libraries.

Dozen reports no Vulkan conformance certification, so wgpu hides it by default.
For experimental Dozen testing, explicitly set
`TILEMAXSIM_VULKAN_ALLOW_NONCONFORMANT=1`. This permits enumeration only: it
does not bypass the daemon's numerical probe or permit CPU fallback.

Use a small initial budget to fit adapter limits and leave VRAM for Windows:

```bash
mkdir -p /tmp/tilemaxsim-vulkan/run /tmp/tilemaxsim-vulkan/tensors
services/tilemaxsimd/target/release/tilemaxsimd \
  --socket /tmp/tilemaxsim-vulkan/run/scoring.sock \
  --status-socket /tmp/tilemaxsim-vulkan/run/status.sock \
  --listen 127.0.0.1:9191 --status-listen 127.0.0.1:9090 \
  --device-memory-gb 0=0.125 --gpu-workspace-gb 0.03125 \
  --host-cache-gb 0.125 --max-inflight-request-gb 0.0625 \
  --contract-root 'your-model@version=/tmp/tilemaxsim-vulkan/tensors'
```

Do not use the CUDA deployment's 8 GB arena and 2 GB workspace defaults on a
4 GB card. Increase budgets only within the actual adapter limits.

## Container and NeoBrain integration

```bash
docker build -f services/Dockerfile.tilemaxsimd-vulkan \
  -t vectorchord-tilemaxsimd:vulkan .
```

The image includes the distribution's Vulkan loader and Mesa drivers. It does
not install a Windows driver or bundle a custom Dozen build. Native Linux needs
`/dev/dri` access and matching render-group permissions; WSL2 needs `/dev/dxg`,
the WSL runtime/driver directories and a compatible Dozen ICD mounted into the
container. Verify adapter identity inside the final container, not only on its
host. The existing NVIDIA container device reservation does not apply.

NeoBrain still needs the matching VectorChord PostgreSQL extension, a pooling
model endpoint, a model contract and published tensor objects. This executor
scores tensors; it does not serve ColQwen or generate embeddings. PostgreSQL
continues to use `vchordrq.maxsim_backend = 'gpu'` and the sidecar TCP endpoint;
that existing setting selects external scoring, not a NVIDIA-specific runtime.
In particular,
`NEOBRAIN_TILEMAXSIM_POOLING_URL`, `NEOBRAIN_TILEMAXSIM_POOLING_MODEL` and
`NEOBRAIN_TILEMAXSIM_MODEL_CONTRACT` are NeoBrain service configuration. Adding
AMD MaxSim support does not make a CUDA-only pooling container AMD-compatible.

### WSL image with Dozen included

For x86-64 WSL2, the additional Dockerfile builds pinned Mesa 25.2.8 from its
checksum-verified source archive and copies only the Dozen driver into the
runtime. The Windows-provided D3D12 libraries remain mounted from WSL:

```bash
docker build -f services/Dockerfile.tilemaxsimd-vulkan \
  -t vectorchord-tilemaxsimd:portable-vulkan .
docker build -f services/Dockerfile.tilemaxsimd-vulkan-wsl \
  -t vectorchord-tilemaxsimd:portable-vulkan-wsl .
docker compose -p vectorchord-amd \
  -f services/compose.tilemaxsimd-vulkan-wsl.yml up -d
curl --fail http://127.0.0.1:39092/healthz
```

This standalone Compose service uses loopback ports 39192 (scoring) and 39092
(status), a persistent tensor volume, and a 96 MiB tensor arena plus 32 MiB
workspace. Set `TILEMAXSIM_MODEL_CONTRACT` to the pooling model's real contract
before connecting NeoBrain; `local-test@1` is only the diagnostic default.

The WSL image explicitly opts into nonconformant Dozen enumeration. On WSL,
the executor retains the D3D12 runtime's code mapping until process exit using
`RTLD_LOCAL | RTLD_NODELETE`. This prevents driver thread-exit callbacks from
calling unloaded code after the last Vulkan instance is destroyed. Device
buffers and queues are still released normally. Do not replace this with
`LD_PRELOAD`: interposing the D3D12 symbols caused recursive calls in testing.

## Recorded acceptance (2026-09-09)

Target: AMD Radeon RX 6400 (PCI vendor/device `1002:743f`, approximately 4 GB),
Windows display driver `32.0.21030.2001`, Ubuntu 24.04 on WSL2, Mesa Dozen 25.2.8.

- Rust unit/CLI tests: 59 passed; the opt-in device test was run separately.
- Clippy with warnings denied passed.
- Real RX 6400 device gate passed FP32 scalar-reference comparisons at dimensions
  1/37/320/513, packed FP16 at 37/320, odd payloads, negative scores, batches and
  invalid ranges. Test-thread teardown also passed after the WSL loader fix.
- The final WSL container identified the real RX 6400 and passed its startup
  conformance check and health check.
- PostgreSQL 16 with VectorChord 1.2.0 and pgvector 0.8.5 passed external-source
  registration and `vchordrq_tilemaxsim_rerank` through both `vector[]` and
  `halfvec[]`: the RX 6400 service returned candidate scores 1.5 and -0.5.
- Graceful container shutdown returned exit code 0; restart restored readiness.
- Container integration: tensor publication, TCP scoring, cold/cache-hit paths
  and 4 concurrent clients passed 34 FP16/FP32 requests at dimension 320.

Reproduce the container test against the standalone diagnostic deployment:

```bash
TILEMAXSIM_VULKAN_CONTAINER=vectorchord-amd-tilemaxsimd-1 \
  python3 -m unittest services.test_tilemaxsim_vulkan_container -v
```

These are functional acceptance results, not a throughput/latency benchmark,
a long-duration reliability certification, or evidence for other AMD models.
Native Linux RADV and other Windows driver versions still require their own
device gates. Dozen itself remains a nonconformant experimental Vulkan driver.
