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
expected_cubins=${VCTM_EXPECTED_CUBINS:-sm_80,sm_89,sm_90,sm_90a}
expected_ptx=${VCTM_EXPECTED_PTX:-sm_90}
async_images=${VCTM_VERIFY_ASYNC_IMAGES-sm_89,sm_90a}
fp8_images=${VCTM_VERIFY_FP8_IMAGES-sm_89,sm_90a}
pq_images=${VCTM_VERIFY_PQ_IMAGES-sm_89,sm_90a}
double_buffer_images=${VCTM_VERIFY_DOUBLE_BUFFER_IMAGES-sm_89,sm_90a}
IFS=',' read -r -a cubins <<<"$expected_cubins"
for image in "${cubins[@]}"; do
  if ! grep -q "tilemaxsim_cuda.${image}.cubin" <<<"$listing"; then
    echo "CUDA archive is missing native ${image} code" >&2
    exit 1
  fi
done

ptx=$(cuobjdump --dump-ptx "$archive")
if ! grep -qE "\.target[[:space:]]+${expected_ptx}([,+[:space:]]|$)" <<<"$ptx"; then
  echo "CUDA archive is missing forward-compatible ${expected_ptx} PTX" >&2
  exit 1
fi

temporary=$(mktemp -d "${TMPDIR:-/tmp}/tilemaxsim-cuda.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
archive=$(realpath "$archive")
IFS=',' read -r -a inspected <<<"$async_images,$fp8_images,$pq_images,$double_buffer_images"
for image in $(printf '%s\n' "${inspected[@]}" | awk 'NF && !seen[$0]++'); do
  (
    cd "$temporary"
    cuobjdump -xelf "tilemaxsim_cuda.${image}.cubin" "$archive" >/dev/null
    sass=$(cuobjdump -sass "tilemaxsim_cuda.${image}.cubin")
    if [[ ",$async_images," == *",$image,"* ]] && ! grep -q 'LDGSTS' <<<"$sass"; then
      echo "${image} fused tile kernel has no asynchronous global-to-shared copy" >&2
      exit 1
    fi
    if [[ ",$fp8_images," == *",$image,"* ]] && ! grep -qE 'F2FP[^[:space:]]*\.E4M3' <<<"$sass"; then
      echo "${image} quantized kernel has no native E4M3 conversion" >&2
      exit 1
    fi
    if [[ ",$pq_images," == *",$image,"* ]]; then
      resources=$(cuobjdump --dump-resource-usage "tilemaxsim_cuda.${image}.cubin")
      if ! grep -q 'pq_adc_maxsim_warp_task_kernel' <<<"$resources" ||
         ! grep -q 'pq_adc_maxsim_kernel' <<<"$resources"; then
        echo "${image} artifact is missing a PQ ADC dispatch variant" >&2
        exit 1
      fi
    fi
    if [[ ",$double_buffer_images," == *",$image,"* ]]; then
      resources=$(cuobjdump --dump-resource-usage "tilemaxsim_cuda.${image}.cubin")
      if ! grep -q 'ELh1ELb0E' <<<"$resources" ||
         ! grep -q 'ELh1ELb1E' <<<"$resources"; then
        echo "${image} artifact is missing an exact single/double-buffer dispatch variant" >&2
        exit 1
      fi
    fi
  )
done

echo "verified CUDA images: ${expected_cubins} and ${expected_ptx} PTX; async tile copy: ${async_images:-not asserted}; native E4M3: ${fp8_images:-not asserted}; PQ ADC variants: ${pq_images:-not asserted}; exact row-buffer variants: ${double_buffer_images:-not asserted}"
