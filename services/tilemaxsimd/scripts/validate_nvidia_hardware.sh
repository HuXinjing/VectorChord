#!/usr/bin/env bash
set -euo pipefail

device=${VCTM_TEST_GPU:-0}
expected_name=${VCTM_EXPECTED_GPU_NAME:-NVIDIA H200}
expected_cc=${VCTM_EXPECTED_COMPUTE_CAPABILITY:-9.0}
evidence_dir=${VCTM_EVIDENCE_DIR:-}
crate_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for command in nvidia-smi nvcc cargo cuobjdump sha256sum nsys ncu; do
  command -v "$command" >/dev/null || {
    echo "required command is unavailable: $command" >&2
    exit 1
  }
done

if [[ -z "$evidence_dir" ]]; then
  evidence_dir=$(mktemp -d "${TMPDIR:-/tmp}/tilemaxsim-hardware.XXXXXX")
  cleanup=true
else
  mkdir -p "$evidence_dir"
  cleanup=false
fi
trap 'if [[ "$cleanup" == true ]]; then rm -rf "$evidence_dir"; fi' EXIT

gpu_name=$(nvidia-smi --id="$device" --query-gpu=name --format=csv,noheader | head -1)
compute_capability=$(nvidia-smi --id="$device" --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "$gpu_name" != *"$expected_name"* ]]; then
  echo "hardware gate expected '$expected_name', found '$gpu_name'" >&2
  exit 1
fi
if [[ "$compute_capability" != "$expected_cc" ]]; then
  echo "hardware gate expected compute capability $expected_cc, found $compute_capability" >&2
  exit 1
fi

{
  printf 'tested_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git -C "$crate_root" rev-parse HEAD)"
  printf 'device_index=%s\n' "$device"
  nvidia-smi --id="$device" \
    --query-gpu=name,uuid,compute_cap,driver_version,memory.total,pci.bus_id \
    --format=csv,noheader
  nvcc --version | tail -1
} | tee "$evidence_dir/device.txt"

export TILEMAXSIM_CUDA_ARCHS=${TILEMAXSIM_CUDA_ARCHS:-90,90a,90-virtual}
cargo build --release --locked --manifest-path "$crate_root/Cargo.toml" \
  --no-default-features --features backend-cuda
"$crate_root/scripts/verify_cuda_artifact.sh" \
  | tee "$evidence_dir/artifact.txt"

RUST_TEST_THREADS=1 VCTM_TEST_GPU="$device" \
  cargo test --release --locked --manifest-path "$crate_root/Cargo.toml" \
    --no-default-features --features backend-cuda --lib -- \
    --ignored --nocapture 2>&1 | tee "$evidence_dir/tests.txt"
if ! grep -E 'cuda_tuning architecture=sm_90 .*double_buffered_tile=Some\((true|false)\).*double_buffer_speedup_milli=Some\([0-9]+\)' \
    "$evidence_dir/tests.txt" | tee "$evidence_dir/tuning.txt"; then
  echo "H200 test output did not contain a complete runtime tuning decision" >&2
  exit 1
fi

test_binary=$(cargo test --release --locked --manifest-path "$crate_root/Cargo.toml" \
  --no-default-features --features backend-cuda --lib --no-run 2>&1 \
  | sed -n 's/^  Executable unittests src\/lib.rs (\(.*\))$/\1/p' | tail -1)
if [[ -z "$test_binary" || ! -x "$test_binary" ]]; then
  echo "unable to locate the CUDA library test executable" >&2
  exit 1
fi

VCTM_TEST_GPU="$device" nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --force-overwrite=true \
  --output="$evidence_dir/tensor-core" "$test_binary" \
  benchmark_batched_tensor_core_for_320d_candidates --ignored --nocapture
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv "$evidence_dir/tensor-core.nsys-rep" \
  >"$evidence_dir/nsys-stats.csv"

VCTM_TEST_GPU="$device" ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:.*(gemm|Gemm|Mma).*' --launch-count 1 \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
  --csv --log-file "$evidence_dir/ncu-tensor.csv" "$test_binary" \
  benchmark_batched_tensor_core_for_320d_candidates --ignored --nocapture
if ! grep -q 'sm__pipe_tensor_cycles_active' "$evidence_dir/ncu-tensor.csv" ||
   grep -q 'No kernels were profiled' "$evidence_dir/ncu-tensor.csv"; then
  echo "Nsight Compute did not prove execution on the tensor pipeline" >&2
  exit 1
fi

find "$evidence_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum | tee "$evidence_dir/SHA256SUMS"
printf 'hardware evidence written to %s\n' "$evidence_dir"
