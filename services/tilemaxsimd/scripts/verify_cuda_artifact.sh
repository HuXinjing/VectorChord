#!/usr/bin/env bash
set -euo pipefail

archive=${1:-}
if [[ -z "$archive" ]]; then
  crate_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  archive=$(find "$crate_root/target/release/build" -path '*/out/libtilemaxsim_cuda.a' \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [[ -z "$archive" || ! -f "$archive" ]]; then
  echo "CUDA archive was not found" >&2
  exit 1
fi

listing=$(cuobjdump --list-elf "$archive")
for image in sm_80 sm_89 sm_90 sm_90a; do
  if ! grep -q "tilemaxsim_cuda.${image}.cubin" <<<"$listing"; then
    echo "CUDA archive is missing native ${image} code" >&2
    exit 1
  fi
done

ptx=$(cuobjdump --dump-ptx "$archive")
if ! grep -qE '\.target[[:space:]]+sm_90' <<<"$ptx"; then
  echo "CUDA archive is missing forward-compatible compute_90 PTX" >&2
  exit 1
fi

temporary=$(mktemp -d "${TMPDIR:-/tmp}/tilemaxsim-cuda.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
archive=$(realpath "$archive")
for image in sm_89 sm_90a; do
  (
    cd "$temporary"
    cuobjdump -xelf "tilemaxsim_cuda.${image}.cubin" "$archive" >/dev/null
    sass=$(cuobjdump -sass "tilemaxsim_cuda.${image}.cubin")
    if ! grep -q 'LDGSTS' <<<"$sass"; then
      echo "${image} fused tile kernel has no asynchronous global-to-shared copy" >&2
      exit 1
    fi
    if ! grep -qE 'F2FP[^[:space:]]*\.E4M3' <<<"$sass"; then
      echo "${image} quantized kernel has no native E4M3 conversion" >&2
      exit 1
    fi
  )
done

echo "verified CUDA images: sm_80 sm_89 sm_90 sm_90a compute_90; async tile copy and native E4M3: sm_89 sm_90a"
