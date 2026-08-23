// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// Copyright (c) 2026 Hu Xinjing

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

struct VctmGpu {
  int device;
  unsigned char *allocation;
  size_t total_bytes;
  size_t tensor_bytes;
  size_t workspace_bytes;
  unsigned char *host_staging;
  size_t host_staging_bytes;
  cudaStream_t upload_stream;
  cudaStream_t compute_stream;
};

struct VctmQuantizer {
  int device;
  float *payload;
  size_t payload_values;
  uint32_t dimension;
  uint16_t stages;
  uint16_t subspaces;
  uint16_t centroids;
  uint16_t rotation_mask;
};

static int fail(char *error, size_t capacity, const char *message);
static int cuda_fail(char *error, size_t capacity, const char *operation,
                     cudaError_t status);

static bool checked_mul(size_t left, size_t right, size_t *output) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) return false;
  *output = left * right;
  return true;
}

extern "C" int vctm_quantizer_create(
    int device, const unsigned char *payload, size_t payload_bytes,
    uint32_t dimension, uint16_t stages, uint16_t subspaces,
    uint16_t centroids, uint16_t rotation_mask, VctmQuantizer **output,
    char *error, size_t error_capacity) {
  if (output == nullptr || payload == nullptr || payload_bytes == 0 ||
      payload_bytes % sizeof(float) != 0 || dimension == 0 || stages == 0 ||
      stages > 16 || subspaces == 0 || centroids < 2 || centroids > 256 ||
      dimension % subspaces != 0 || (rotation_mask >> stages) != 0) {
    return fail(error, error_capacity, "invalid PQ quantizer");
  }
  size_t rotations = 0, books = 0, expected_values = 0;
  if (!checked_mul(static_cast<size_t>(__builtin_popcount(rotation_mask)), dimension, &rotations) ||
      !checked_mul(rotations, dimension, &rotations) ||
      !checked_mul(stages, subspaces, &books) ||
      !checked_mul(books, centroids, &books) ||
      !checked_mul(books, dimension / subspaces, &books) ||
      rotations > std::numeric_limits<size_t>::max() - books) {
    return fail(error, error_capacity, "PQ quantizer shape overflows address space");
  }
  expected_values = rotations + books;
  if (payload_bytes / sizeof(float) != expected_values)
    return fail(error, error_capacity, "PQ quantizer payload length disagrees with its shape");
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  auto *quantizer = new VctmQuantizer{device, nullptr, payload_bytes / sizeof(float),
      dimension, stages, subspaces, centroids, rotation_mask};
  status = cudaMalloc(reinterpret_cast<void **>(&quantizer->payload), payload_bytes);
  if (status == cudaSuccess)
    status = cudaMemcpy(quantizer->payload, payload, payload_bytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    if (quantizer->payload != nullptr) cudaFree(quantizer->payload);
    delete quantizer;
    return cuda_fail(error, error_capacity, "PQ quantizer upload", status);
  }
  *output = quantizer;
  return 0;
}

extern "C" void vctm_quantizer_destroy(VctmQuantizer *quantizer) {
  if (quantizer == nullptr) return;
  cudaSetDevice(quantizer->device);
  cudaFree(quantizer->payload);
  delete quantizer;
}

static int fail(char *error, size_t capacity, const char *message) {
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, capacity, "%s", message);
  }
  return 1;
}

static int cuda_fail(char *error, size_t capacity, const char *operation,
                     cudaError_t status) {
  if (error != nullptr && capacity != 0) {
    std::snprintf(error, capacity, "%s: %s", operation,
                  cudaGetErrorString(status));
  }
  return 1;
}

extern "C" int vctm_gpu_create(int device, size_t total_bytes,
                                size_t workspace_bytes, VctmGpu **output,
                                char *error, size_t error_capacity) {
  if (output == nullptr || total_bytes == 0 || workspace_bytes == 0 ||
      workspace_bytes >= total_bytes) {
    return fail(error, error_capacity, "invalid GPU arena configuration");
  }
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  }
  size_t free_bytes = 0;
  size_t device_bytes = 0;
  status = cudaMemGetInfo(&free_bytes, &device_bytes);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "cudaMemGetInfo", status);
  }
  if (free_bytes < total_bytes) {
    return fail(error, error_capacity,
                "configured GPU memory is not currently available");
  }
  auto *gpu = new VctmGpu{};
  gpu->device = device;
  gpu->total_bytes = total_bytes;
  gpu->tensor_bytes = ((total_bytes - workspace_bytes) / 256) * 256;
  gpu->workspace_bytes = total_bytes - gpu->tensor_bytes;
  gpu->host_staging_bytes =
      std::min(gpu->tensor_bytes, static_cast<size_t>(64) * 1024 * 1024);
  status = cudaMalloc(reinterpret_cast<void **>(&gpu->allocation), total_bytes);
  if (status != cudaSuccess) {
    delete gpu;
    return cuda_fail(error, error_capacity, "cudaMalloc", status);
  }
  if ((status = cudaStreamCreateWithFlags(&gpu->upload_stream,
                                           cudaStreamNonBlocking)) != cudaSuccess ||
      (status = cudaStreamCreateWithFlags(&gpu->compute_stream,
                                           cudaStreamNonBlocking)) != cudaSuccess) {
    if (gpu->upload_stream != nullptr) cudaStreamDestroy(gpu->upload_stream);
    cudaFree(gpu->allocation);
    delete gpu;
    return cuda_fail(error, error_capacity, "cudaStreamCreate", status);
  }
  status = cudaHostAlloc(reinterpret_cast<void **>(&gpu->host_staging),
                         gpu->host_staging_bytes, cudaHostAllocPortable);
  if (status != cudaSuccess) {
    cudaStreamDestroy(gpu->upload_stream);
    cudaStreamDestroy(gpu->compute_stream);
    cudaFree(gpu->allocation);
    delete gpu;
    return cuda_fail(error, error_capacity, "cudaHostAlloc", status);
  }
  std::memset(gpu->host_staging, 0, gpu->host_staging_bytes);
  *output = gpu;
  return 0;
}

extern "C" void vctm_gpu_destroy(VctmGpu *gpu) {
  if (gpu == nullptr) return;
  cudaSetDevice(gpu->device);
  cudaStreamDestroy(gpu->upload_stream);
  cudaStreamDestroy(gpu->compute_stream);
  cudaFreeHost(gpu->host_staging);
  cudaFree(gpu->allocation);
  delete gpu;
}

extern "C" size_t vctm_gpu_tensor_bytes(const VctmGpu *gpu) {
  return gpu == nullptr ? 0 : gpu->tensor_bytes;
}

extern "C" int vctm_gpu_upload_batch(
    VctmGpu *gpu, const uint64_t *offsets,
    const unsigned char *const *payloads, const size_t *lengths, size_t count,
    char *error, size_t error_capacity) {
  if (gpu == nullptr || offsets == nullptr || payloads == nullptr ||
      lengths == nullptr) {
    return fail(error, error_capacity, "invalid upload batch");
  }
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  }
  for (size_t i = 0; i < count; ++i) {
    if (offsets[i] > gpu->tensor_bytes ||
        lengths[i] > gpu->tensor_bytes - offsets[i]) {
      return fail(error, error_capacity, "upload is outside the tensor arena");
    }
  }
  size_t item = 0;
  size_t item_offset = 0;
  while (item < count) {
    size_t staging_offset = 0;
    while (item < count && staging_offset < gpu->host_staging_bytes) {
      const size_t remaining = lengths[item] - item_offset;
      const size_t chunk =
          std::min(remaining, gpu->host_staging_bytes - staging_offset);
      std::memcpy(gpu->host_staging + staging_offset,
                  payloads[item] + item_offset, chunk);
      status = cudaMemcpyAsync(gpu->allocation + offsets[item] + item_offset,
                               gpu->host_staging + staging_offset, chunk,
                               cudaMemcpyHostToDevice, gpu->upload_stream);
      if (status != cudaSuccess) {
        return cuda_fail(error, error_capacity, "cudaMemcpyAsync(H2D)", status);
      }
      staging_offset += chunk;
      item_offset += chunk;
      if (item_offset == lengths[item]) {
        item += 1;
        item_offset = 0;
      }
    }
    status = cudaStreamSynchronize(gpu->upload_stream);
    if (status != cudaSuccess) {
      return cuda_fail(error, error_capacity, "cudaStreamSynchronize(upload)",
                       status);
    }
  }
  return 0;
}

template <typename Scalar>
__device__ float scalar_to_float(Scalar value);

template <>
__device__ float scalar_to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ float scalar_to_float<float>(float value) {
  return value;
}

template <typename Scalar>
__global__ void tilemaxsim_kernel(const Scalar *query, uint32_t query_rows,
                                  uint32_t dimension,
                                  const unsigned char *documents,
                                  const uint64_t *document_offsets,
                                  const uint32_t *document_rows,
                                  size_t task_count, float *maxima) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps = blockDim.x >> 5;
  __shared__ float warp_best[8];
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const size_t candidate = task / query_rows;
    const uint32_t query_row = static_cast<uint32_t>(task % query_rows);
    const auto *document = reinterpret_cast<const Scalar *>(
        documents + document_offsets[candidate]);
    const Scalar *query_vector =
        query + static_cast<size_t>(query_row) * dimension;
    float best = -CUDART_INF_F;
    for (uint32_t row = warp; row < document_rows[candidate]; row += warps) {
      const Scalar *document_vector =
          document + static_cast<size_t>(row) * dimension;
      float dot = 0.0f;
      for (uint32_t index = lane; index < dimension; index += 32) {
        dot = fmaf(scalar_to_float(query_vector[index]),
                   scalar_to_float(document_vector[index]), dot);
      }
      for (int delta = 16; delta != 0; delta >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, delta);
      }
      if (lane == 0) best = fmaxf(best, dot);
    }
    if (lane == 0) warp_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      float maximum = -CUDART_INF_F;
      for (uint32_t index = 0; index < warps; ++index) {
        maximum = fmaxf(maximum, warp_best[index]);
      }
      maxima[task] = maximum;
    }
    __syncthreads();
  }
}

template <typename QueryScalar>
__global__ void tilemaxsim_int8_kernel(
    const QueryScalar *query, uint32_t query_rows, uint32_t dimension,
    const unsigned char *documents, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t task_count, float *maxima) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps = blockDim.x >> 5;
  __shared__ float warp_best[8];
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const size_t candidate = task / query_rows;
    const uint32_t query_row = static_cast<uint32_t>(task % query_rows);
    const auto *document = reinterpret_cast<const int8_t *>(
        documents + document_offsets[candidate]);
    const size_t code_bytes =
        static_cast<size_t>(document_rows[candidate]) * dimension;
    const size_t scale_offset =
        (code_bytes + alignof(float) - 1) &
        ~(static_cast<size_t>(alignof(float)) - 1);
    const auto *scales = reinterpret_cast<const float *>(
        document + scale_offset);
    const QueryScalar *query_vector =
        query + static_cast<size_t>(query_row) * dimension;
    float best = -CUDART_INF_F;
    for (uint32_t row = warp; row < document_rows[candidate]; row += warps) {
      const int8_t *document_vector =
          document + static_cast<size_t>(row) * dimension;
      float dot = 0.0f;
      for (uint32_t index = lane; index < dimension; index += 32) {
        dot = fmaf(scalar_to_float(query_vector[index]),
                   static_cast<float>(document_vector[index]), dot);
      }
      dot *= scales[row];
      for (int delta = 16; delta != 0; delta >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, delta);
      }
      if (lane == 0) best = fmaxf(best, dot);
    }
    if (lane == 0) warp_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      float maximum = -CUDART_INF_F;
      for (uint32_t index = 0; index < warps; ++index) {
        maximum = fmaxf(maximum, warp_best[index]);
      }
      maxima[task] = maximum;
    }
    __syncthreads();
  }
}

__device__ float e4m3fn_to_float(uint8_t bits) {
  const float sign = (bits & 0x80) == 0 ? 1.0f : -1.0f;
  const uint8_t exponent = (bits >> 3) & 0x0f;
  const uint8_t fraction = bits & 0x07;
  if (exponent == 0) return sign * static_cast<float>(fraction) * 0.001953125f;
  return sign * (1.0f + static_cast<float>(fraction) * 0.125f) *
         exp2f(static_cast<float>(exponent) - 7.0f);
}

template <typename QueryScalar>
__global__ void tilemaxsim_fp8_kernel(
    const QueryScalar *query, uint32_t query_rows, uint32_t dimension,
    const unsigned char *documents, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t task_count, float *maxima) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps = blockDim.x >> 5;
  __shared__ float warp_best[8];
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const size_t candidate = task / query_rows;
    const uint32_t query_row = static_cast<uint32_t>(task % query_rows);
    const uint8_t *document = documents + document_offsets[candidate];
    const size_t code_bytes = static_cast<size_t>(document_rows[candidate]) * dimension;
    const size_t scale_offset = (code_bytes + 3) & ~static_cast<size_t>(3);
    const float *scales = reinterpret_cast<const float *>(document + scale_offset);
    const QueryScalar *query_vector = query + static_cast<size_t>(query_row) * dimension;
    float best = -CUDART_INF_F;
    for (uint32_t row = warp; row < document_rows[candidate]; row += warps) {
      const uint8_t *document_vector = document + static_cast<size_t>(row) * dimension;
      float dot = 0.0f;
      for (uint32_t index = lane; index < dimension; index += 32) {
        dot = fmaf(scalar_to_float(query_vector[index]),
                   e4m3fn_to_float(document_vector[index]), dot);
      }
      dot *= scales[row];
      for (int delta = 16; delta != 0; delta >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, delta);
      }
      if (lane == 0) best = fmaxf(best, dot);
    }
    if (lane == 0) warp_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      float maximum = -CUDART_INF_F;
      for (uint32_t index = 0; index < warps; ++index) maximum = fmaxf(maximum, warp_best[index]);
      maxima[task] = maximum;
    }
    __syncthreads();
  }
}

__global__ void tilemaxsim_sum_kernel(const float *maxima, uint32_t query_rows,
                                      size_t count, float *scores) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= count) return;
  float score = 0.0f;
  for (uint32_t query_row = 0; query_row < query_rows; ++query_row) {
    score += maxima[candidate * query_rows + query_row];
  }
  scores[candidate] = score;
}

template <typename QueryScalar>
__global__ void pq_lut_kernel(
    const QueryScalar *query, uint32_t query_rows, const VctmQuantizer quantizer,
    float *luts, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  size_t cursor = index;
  const uint16_t centroid = cursor % quantizer.centroids; cursor /= quantizer.centroids;
  const uint16_t subspace = cursor % quantizer.subspaces; cursor /= quantizer.subspaces;
  const uint16_t stage = cursor % quantizer.stages; cursor /= quantizer.stages;
  const uint32_t query_row = static_cast<uint32_t>(cursor);
  const uint32_t subdimension = quantizer.dimension / quantizer.subspaces;
  const size_t rotations = __popc(static_cast<unsigned int>(quantizer.rotation_mask));
  const size_t codebook_base = rotations * quantizer.dimension * quantizer.dimension;
  size_t rotation_rank = 0;
  for (uint16_t item = 0; item < stage; ++item)
    rotation_rank += (quantizer.rotation_mask >> item) & 1;
  float dot = 0.0f;
  for (uint32_t local = 0; local < subdimension; ++local) {
    const uint32_t dimension = static_cast<uint32_t>(subspace) * subdimension + local;
    float query_value = scalar_to_float(query[static_cast<size_t>(query_row) * quantizer.dimension + dimension]);
    if ((quantizer.rotation_mask >> stage) & 1) {
      query_value = 0.0f;
      const float *rotation = quantizer.payload + rotation_rank * quantizer.dimension * quantizer.dimension;
      for (uint32_t source = 0; source < quantizer.dimension; ++source)
        query_value = fmaf(scalar_to_float(query[static_cast<size_t>(query_row) * quantizer.dimension + source]),
                           rotation[static_cast<size_t>(source) * quantizer.dimension + dimension], query_value);
    }
    const size_t center = (((static_cast<size_t>(stage) * quantizer.subspaces + subspace)
        * quantizer.centroids + centroid) * subdimension + local);
    dot = fmaf(query_value, quantizer.payload[codebook_base + center], dot);
  }
  luts[index] = dot;
}

__global__ void pq_adc_maxsim_kernel(
    const float *luts, uint32_t query_rows, const VctmQuantizer quantizer,
    const unsigned char *documents, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t task_count, float *maxima) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps = blockDim.x >> 5;
  __shared__ float warp_best[8];
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const size_t candidate = task / query_rows;
    const uint32_t query_row = static_cast<uint32_t>(task % query_rows);
    const uint8_t *codes = documents + document_offsets[candidate];
    float best = -CUDART_INF_F;
    for (uint32_t row = warp; row < document_rows[candidate]; row += warps) {
      float similarity = 0.0f;
      const size_t row_base = static_cast<size_t>(row) * quantizer.stages * quantizer.subspaces;
      const size_t lut_base = static_cast<size_t>(query_row) * quantizer.stages * quantizer.subspaces * quantizer.centroids;
      for (uint32_t flat = lane; flat < static_cast<uint32_t>(quantizer.stages) * quantizer.subspaces; flat += 32) {
        const uint8_t code = codes[row_base + flat];
        similarity += luts[lut_base + static_cast<size_t>(flat) * quantizer.centroids + code];
      }
      for (int delta = 16; delta != 0; delta >>= 1)
        similarity += __shfl_down_sync(0xffffffff, similarity, delta);
      if (lane == 0) best = fmaxf(best, similarity);
    }
    if (lane == 0) warp_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
      float maximum = -CUDART_INF_F;
      for (uint32_t item = 0; item < warps; ++item) maximum = fmaxf(maximum, warp_best[item]);
      maxima[task] = maximum;
    }
    __syncthreads();
  }
}

static size_t aligned(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

static bool reserve_aligned(size_t *cursor, size_t bytes, size_t *offset) {
  constexpr size_t alignment = 256;
  *offset = *cursor;
  if (bytes > std::numeric_limits<size_t>::max() - *cursor) return false;
  const size_t end = *cursor + bytes;
  if (end > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
  *cursor = aligned(end, alignment);
  return true;
}

extern "C" int vctm_gpu_score(
    VctmGpu *gpu, const unsigned char *query, size_t query_bytes,
    uint32_t query_rows, uint32_t dimension, uint8_t dtype,
    uint8_t scoring_profile,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  if (gpu == nullptr || query == nullptr || document_offsets == nullptr ||
      document_rows == nullptr || output == nullptr || query_rows == 0 ||
      dimension == 0 || count == 0) {
    return fail(error, error_capacity, "invalid TileMaxSim score request");
  }
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  }
  unsigned char *workspace = gpu->allocation + gpu->tensor_bytes;
  if (count > std::numeric_limits<size_t>::max() / query_rows) {
    return fail(error, error_capacity, "TileMaxSim maxima count overflow");
  }
  const size_t maxima_count = count * query_rows;
  if (count > std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
      count > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
      count > std::numeric_limits<size_t>::max() / sizeof(float) ||
      maxima_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return fail(error, error_capacity, "TileMaxSim workspace size overflow");
  }
  size_t cursor = 0;
  size_t query_offset = 0;
  size_t offsets_offset = 0;
  size_t rows_offset = 0;
  size_t maxima_offset = 0;
  size_t scores_offset = 0;
  if (!reserve_aligned(&cursor, query_bytes, &query_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint64_t), &offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint32_t), &rows_offset) ||
      !reserve_aligned(&cursor, maxima_count * sizeof(float), &maxima_offset) ||
      !reserve_aligned(&cursor, count * sizeof(float), &scores_offset)) {
    return fail(error, error_capacity, "TileMaxSim workspace size overflow");
  }
  if (cursor > gpu->workspace_bytes) {
    return fail(error, error_capacity,
                "TileMaxSim request exceeds the configured GPU workspace");
  }
  status = cudaMemcpyAsync(workspace + query_offset, query, query_bytes,
                           cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(workspace + offsets_offset, document_offsets,
                             count * sizeof(uint64_t), cudaMemcpyHostToDevice,
                             gpu->compute_stream);
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(workspace + rows_offset, document_rows,
                             count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             gpu->compute_stream);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "CUDA workspace initialization",
                     status);
  }
  // A CUDA grid's Y dimension is limited to 65,535 even on modern devices.
  // Flatten candidate/query-row work into X and let blocks stride when the
  // task count is larger. This covers every protocol-valid query shape without
  // constructing an excessive launch grid.
  constexpr size_t maximum_kernel_blocks = 65'535;
  const size_t kernel_blocks = std::min(maxima_count, maximum_kernel_blocks);
  dim3 grid(static_cast<unsigned int>(kernel_blocks));
  dim3 block(256);
  if (scoring_profile == 1 && dtype == 2) {
    tilemaxsim_kernel<half><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const half *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else if (scoring_profile == 1 && dtype == 1) {
    tilemaxsim_kernel<float><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else if (scoring_profile == 2 && dtype == 2) {
    tilemaxsim_int8_kernel<half><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const half *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else if (scoring_profile == 2 && dtype == 1) {
    tilemaxsim_int8_kernel<float><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else if (scoring_profile == 3 && dtype == 2) {
    tilemaxsim_fp8_kernel<half><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const half *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else if (scoring_profile == 3 && dtype == 1) {
    tilemaxsim_fp8_kernel<float><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  } else {
    return fail(error, error_capacity,
                "unsupported tensor dtype or scoring profile");
  }
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    tilemaxsim_sum_kernel<<<blocks, threads, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + maxima_offset), query_rows,
        count, reinterpret_cast<float *>(workspace + scores_offset));
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpyAsync(output, workspace + scores_offset,
                             count * sizeof(float), cudaMemcpyDeviceToHost,
                             gpu->compute_stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(gpu->compute_stream);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "TileMaxSim CUDA execution", status);
  }
  return 0;
}

extern "C" int vctm_gpu_score_pq(
    VctmGpu *gpu, const VctmQuantizer *quantizer,
    const unsigned char *query, size_t query_bytes, uint32_t query_rows,
    uint8_t dtype, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t count, float *output,
    char *error, size_t error_capacity) {
  if (gpu == nullptr || quantizer == nullptr || query == nullptr || query_rows == 0 ||
      count == 0 || document_offsets == nullptr || document_rows == nullptr || output == nullptr ||
      quantizer->device != gpu->device) return fail(error, error_capacity, "invalid PQ score request");
  const size_t scalar_bytes = dtype == 1 ? sizeof(float) : dtype == 2 ? sizeof(half) : 0;
  size_t expected_query_values = 0, expected_query_bytes = 0;
  if (scalar_bytes == 0 || !checked_mul(query_rows, quantizer->dimension, &expected_query_values) ||
      !checked_mul(expected_query_values, scalar_bytes, &expected_query_bytes) ||
      query_bytes != expected_query_bytes)
    return fail(error, error_capacity, "PQ query byte length disagrees with its shape");
  size_t codes_per_row = 0;
  if (!checked_mul(quantizer->stages, quantizer->subspaces, &codes_per_row))
    return fail(error, error_capacity, "PQ document shape overflows address space");
  for (size_t index = 0; index < count; ++index) {
    size_t code_bytes = 0;
    if (document_rows[index] == 0 || !checked_mul(document_rows[index], codes_per_row, &code_bytes) ||
        document_offsets[index] > gpu->tensor_bytes ||
        code_bytes > gpu->tensor_bytes - static_cast<size_t>(document_offsets[index]))
      return fail(error, error_capacity, "PQ document range exceeds the GPU tensor arena");
  }
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  size_t maxima_count = 0, lut_count = 0;
  if (!checked_mul(count, query_rows, &maxima_count) ||
      !checked_mul(query_rows, quantizer->stages, &lut_count) ||
      !checked_mul(lut_count, quantizer->subspaces, &lut_count) ||
      !checked_mul(lut_count, quantizer->centroids, &lut_count))
    return fail(error, error_capacity, "PQ request shape overflows address space");
  unsigned char *workspace = gpu->allocation + gpu->tensor_bytes;
  size_t cursor = 0, query_offset = 0, offsets_offset = 0, rows_offset = 0,
         lut_offset = 0, maxima_offset = 0, scores_offset = 0;
  if (!reserve_aligned(&cursor, query_bytes, &query_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint64_t), &offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint32_t), &rows_offset) ||
      !reserve_aligned(&cursor, lut_count * sizeof(float), &lut_offset) ||
      !reserve_aligned(&cursor, maxima_count * sizeof(float), &maxima_offset) ||
      !reserve_aligned(&cursor, count * sizeof(float), &scores_offset) ||
      cursor > gpu->workspace_bytes)
    return fail(error, error_capacity, "PQ request exceeds configured GPU workspace");
  status = cudaMemcpyAsync(workspace + query_offset, query, query_bytes, cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + offsets_offset, document_offsets, count * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + rows_offset, document_rows, count * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  constexpr unsigned int threads = 256;
  const unsigned int lut_blocks = static_cast<unsigned int>((lut_count + threads - 1) / threads);
  if (status == cudaSuccess && dtype == 2)
    pq_lut_kernel<half><<<lut_blocks, threads, 0, gpu->compute_stream>>>(reinterpret_cast<const half *>(workspace + query_offset), query_rows, *quantizer, reinterpret_cast<float *>(workspace + lut_offset), lut_count);
  else if (status == cudaSuccess && dtype == 1)
    pq_lut_kernel<float><<<lut_blocks, threads, 0, gpu->compute_stream>>>(reinterpret_cast<const float *>(workspace + query_offset), query_rows, *quantizer, reinterpret_cast<float *>(workspace + lut_offset), lut_count);
  else if (status == cudaSuccess) return fail(error, error_capacity, "unsupported PQ query dtype");
  if (status == cudaSuccess) status = cudaGetLastError();
  const size_t kernel_blocks = std::min(maxima_count, static_cast<size_t>(65'535));
  if (status == cudaSuccess) pq_adc_maxsim_kernel<<<static_cast<unsigned int>(kernel_blocks), threads, 0, gpu->compute_stream>>>(
      reinterpret_cast<const float *>(workspace + lut_offset), query_rows, *quantizer,
      gpu->allocation, reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
      reinterpret_cast<const uint32_t *>(workspace + rows_offset), maxima_count,
      reinterpret_cast<float *>(workspace + maxima_offset));
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) tilemaxsim_sum_kernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, gpu->compute_stream>>>(
      reinterpret_cast<const float *>(workspace + maxima_offset), query_rows, count,
      reinterpret_cast<float *>(workspace + scores_offset));
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaMemcpyAsync(output, workspace + scores_offset, count * sizeof(float), cudaMemcpyDeviceToHost, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(gpu->compute_stream);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "PQ ADC-MaxSim CUDA execution", status);
  return 0;
}
