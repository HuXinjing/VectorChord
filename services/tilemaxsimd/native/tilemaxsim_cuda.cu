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
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
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
  cublasHandle_t cublas;
  int compute_major;
  uint32_t tile_queries_per_warp;
  uint32_t pq_warp_task_max_document_rows;
  size_t matrix_engine_workspace_bytes;
  int compute_stream_priority;
  size_t persisting_l2_bytes;
  size_t access_policy_max_window_bytes;
  std::vector<uint32_t> tensor_cached_document_rows;
  std::map<uint32_t, std::vector<uint32_t>> tensor_cached_row_groups;
  std::vector<const void *> tensor_a_pointers;
  std::vector<const void *> tensor_b_pointers;
  std::vector<void *> tensor_c_pointers;
  std::vector<uint32_t> tensor_candidate_indexes;
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

// CUDA access-policy windows are stream state. Keep the policy scoped to one
// score operation so an error path cannot accidentally pin unrelated request
// metadata or result buffers in L2. Failure is deliberately non-fatal: MIG and
// some driver/device combinations expose the property but reject the limit.
class ScopedPersistingQuery {
 public:
  ScopedPersistingQuery(VctmGpu *gpu, void *base, size_t bytes)
      : gpu_(gpu), enabled_(false) {
    if (gpu == nullptr || gpu->persisting_l2_bytes == 0 ||
        gpu->access_policy_max_window_bytes == 0 || bytes == 0) return;
    cudaStreamAttrValue value{};
    value.accessPolicyWindow.base_ptr = base;
    value.accessPolicyWindow.num_bytes = std::min(
        bytes, std::min(gpu->persisting_l2_bytes,
                        gpu->access_policy_max_window_bytes));
    value.accessPolicyWindow.hitRatio = 1.0f;
    value.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    value.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    enabled_ = cudaStreamSetAttribute(
                   gpu->compute_stream, cudaStreamAttributeAccessPolicyWindow,
                   &value) == cudaSuccess;
    if (!enabled_) cudaGetLastError();
  }

  ~ScopedPersistingQuery() {
    if (!enabled_) return;
    cudaStreamAttrValue value{};
    value.accessPolicyWindow.num_bytes = 0;
    if (cudaStreamSetAttribute(gpu_->compute_stream,
                               cudaStreamAttributeAccessPolicyWindow,
                               &value) != cudaSuccess)
      cudaGetLastError();
  }

  ScopedPersistingQuery(const ScopedPersistingQuery &) = delete;
  ScopedPersistingQuery &operator=(const ScopedPersistingQuery &) = delete;

 private:
  VctmGpu *gpu_;
  bool enabled_;
};

static int fail(char *error, size_t capacity, const char *message);
static int cuda_fail(char *error, size_t capacity, const char *operation,
                     cudaError_t status);
static int cublas_fail(char *error, size_t capacity, const char *operation,
                       cublasStatus_t status) {
  if (error != nullptr && capacity != 0)
    std::snprintf(error, capacity, "%s: cuBLAS status %d", operation, static_cast<int>(status));
  return 1;
}
static bool reserve_aligned(size_t *cursor, size_t bytes, size_t *offset);

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
  gpu->tile_queries_per_warp = 1;
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
  int least_priority = 0, greatest_priority = 0;
  status = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  if (status != cudaSuccess) {
    cudaFree(gpu->allocation);
    delete gpu;
    return cuda_fail(error, error_capacity,
                     "cudaDeviceGetStreamPriorityRange", status);
  }
  gpu->compute_stream_priority = greatest_priority;
  if ((status = cudaStreamCreateWithPriority(&gpu->upload_stream,
                                              cudaStreamNonBlocking,
                                              least_priority)) != cudaSuccess ||
      (status = cudaStreamCreateWithPriority(&gpu->compute_stream,
                                              cudaStreamNonBlocking,
                                              greatest_priority)) != cudaSuccess) {
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
  cublasStatus_t blas_status = cublasCreate(&gpu->cublas);
  if (blas_status == CUBLAS_STATUS_SUCCESS)
    blas_status = cublasSetStream(gpu->cublas, gpu->compute_stream);
  if (blas_status == CUBLAS_STATUS_SUCCESS)
    blas_status = cublasSetMathMode(gpu->cublas, CUBLAS_TENSOR_OP_MATH);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    if (gpu->cublas != nullptr) cublasDestroy(gpu->cublas);
    cudaFreeHost(gpu->host_staging);
    cudaStreamDestroy(gpu->upload_stream);
    cudaStreamDestroy(gpu->compute_stream);
    cudaFree(gpu->allocation);
    delete gpu;
    return cublas_fail(error, error_capacity, "cublasCreate", blas_status);
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    gpu->compute_major = properties.major;
    gpu->tile_queries_per_warp =
        properties.major >= 9 ? 8 : properties.major >= 8 ? 4 : 1;
  }
  // Give cuBLAS stable scratch space instead of letting it allocate during a
  // latency-sensitive score call. Hopper gets a larger budget because its
  // grouped Tensor Core algorithms can profit from more staging workspace.
  const size_t desired_blas_workspace =
      gpu->compute_major >= 9 ? static_cast<size_t>(32) * 1024 * 1024
                              : static_cast<size_t>(4) * 1024 * 1024;
  gpu->matrix_engine_workspace_bytes =
      std::min(desired_blas_workspace, gpu->workspace_bytes / 4) &
      ~static_cast<size_t>(255);
  if (gpu->matrix_engine_workspace_bytes != 0) {
    void *blas_workspace = gpu->allocation + gpu->total_bytes -
                           gpu->matrix_engine_workspace_bytes;
    if (cublasSetWorkspace(gpu->cublas, blas_workspace,
                           gpu->matrix_engine_workspace_bytes) ==
        CUBLAS_STATUS_SUCCESS) {
      gpu->workspace_bytes -= gpu->matrix_engine_workspace_bytes;
    } else {
      gpu->matrix_engine_workspace_bytes = 0;
    }
  }
  if (gpu->compute_major != 0 &&
      properties.persistingL2CacheMaxSize > 0 &&
      properties.accessPolicyMaxWindowSize > 0) {
    gpu->persisting_l2_bytes = std::min(
        static_cast<size_t>(properties.persistingL2CacheMaxSize),
        static_cast<size_t>(4) * 1024 * 1024);
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           gpu->persisting_l2_bytes) == cudaSuccess) {
      gpu->access_policy_max_window_bytes =
          static_cast<size_t>(properties.accessPolicyMaxWindowSize);
    } else {
      gpu->persisting_l2_bytes = 0;
      gpu->access_policy_max_window_bytes = 0;
      cudaGetLastError();
    }
  }
  std::memset(gpu->host_staging, 0, gpu->host_staging_bytes);
  *output = gpu;
  return 0;
}

extern "C" void vctm_gpu_destroy(VctmGpu *gpu) {
  if (gpu == nullptr) return;
  cudaSetDevice(gpu->device);
  cublasDestroy(gpu->cublas);
  cudaStreamDestroy(gpu->upload_stream);
  cudaStreamDestroy(gpu->compute_stream);
  cudaFreeHost(gpu->host_staging);
  cudaFree(gpu->allocation);
  delete gpu;
}

extern "C" size_t vctm_gpu_tensor_bytes(const VctmGpu *gpu) {
  return gpu == nullptr ? 0 : gpu->tensor_bytes;
}

extern "C" int vctm_gpu_set_tile_queries_per_warp(VctmGpu *gpu,
                                                     uint32_t value) {
  if (gpu == nullptr || (value != 1 && value != 4 && value != 8)) return 1;
  gpu->tile_queries_per_warp = value;
  return 0;
}

extern "C" uint32_t vctm_gpu_tile_queries_per_warp(const VctmGpu *gpu) {
  return gpu == nullptr ? 0 : gpu->tile_queries_per_warp;
}

extern "C" int vctm_gpu_set_pq_warp_task_max_document_rows(
    VctmGpu *gpu, uint32_t rows) {
  if (gpu == nullptr) return 1;
  gpu->pq_warp_task_max_document_rows = rows;
  return 0;
}

extern "C" uint32_t vctm_gpu_pq_warp_task_max_document_rows(
    const VctmGpu *gpu) {
  return gpu == nullptr ? 0 : gpu->pq_warp_task_max_document_rows;
}

extern "C" int vctm_gpu_compute_capability(const VctmGpu *gpu, int *major, int *minor) {
  if (gpu == nullptr || major == nullptr || minor == nullptr) return 1;
  if (cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, gpu->device) != cudaSuccess) return 1;
  if (cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, gpu->device) != cudaSuccess) return 1;
  return 0;
}

extern "C" int vctm_gpu_device_name(const VctmGpu *gpu, char *name, size_t capacity) {
  if (gpu == nullptr || name == nullptr || capacity == 0) return 1;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, gpu->device) != cudaSuccess) return 1;
  std::snprintf(name, capacity, "%s", properties.name);
  return 0;
}

extern "C" int vctm_gpu_device_info(
    const VctmGpu *gpu, int *driver_version, int *runtime_version,
    int *cublas_version, uint64_t *total_memory_bytes,
    int *multiprocessors, int *warp_size, int *memory_bus_width_bits,
    int *memory_clock_khz, uint64_t *shared_memory_per_block_bytes,
    int *compute_stream_priority, uint64_t *persisting_l2_bytes,
    uint64_t *matrix_engine_workspace_bytes) {
  if (gpu == nullptr || driver_version == nullptr || runtime_version == nullptr ||
      cublas_version == nullptr || total_memory_bytes == nullptr ||
      multiprocessors == nullptr || warp_size == nullptr ||
      memory_bus_width_bits == nullptr || memory_clock_khz == nullptr ||
      shared_memory_per_block_bytes == nullptr ||
      compute_stream_priority == nullptr || persisting_l2_bytes == nullptr ||
      matrix_engine_workspace_bytes == nullptr) return 1;
  cudaDeviceProp properties{};
  if (cudaDriverGetVersion(driver_version) != cudaSuccess ||
      cudaRuntimeGetVersion(runtime_version) != cudaSuccess ||
      cublasGetVersion(gpu->cublas, cublas_version) != CUBLAS_STATUS_SUCCESS ||
      cudaGetDeviceProperties(&properties, gpu->device) != cudaSuccess) return 1;
  *total_memory_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
  *multiprocessors = properties.multiProcessorCount;
  *warp_size = properties.warpSize;
  *memory_bus_width_bits = properties.memoryBusWidth;
  *memory_clock_khz = properties.memoryClockRate;
  *shared_memory_per_block_bytes =
      static_cast<uint64_t>(properties.sharedMemPerBlock);
  *compute_stream_priority = gpu->compute_stream_priority;
  *persisting_l2_bytes = static_cast<uint64_t>(gpu->persisting_l2_bytes);
  *matrix_engine_workspace_bytes =
      static_cast<uint64_t>(gpu->matrix_engine_workspace_bytes);
  return 0;
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
__device__ Scalar scalar_from_float(float value);

template <>
__device__ half scalar_from_float<half>(float value) {
  return __float2half(value);
}

template <>
__device__ float scalar_from_float<float>(float value) {
  return value;
}

template <typename Scalar>
__device__ float lane_dot(const Scalar *left, const Scalar *right,
                          uint32_t dimension, uint32_t lane) {
  float dot = 0.0f;
  for (uint32_t index = lane; index < dimension; index += 32)
    dot = fmaf(scalar_to_float(left[index]), scalar_to_float(right[index]), dot);
  return dot;
}

__device__ float e4m3fn_to_float(uint8_t bits);

template <typename QueryScalar>
__device__ float lane_dot_int8(const QueryScalar *query, const int8_t *document,
                               uint32_t dimension, uint32_t lane) {
  float dot = 0.0f;
  if ((dimension & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(document) & 3U) == 0) {
    const auto *packed_document = reinterpret_cast<const char4 *>(document);
    const uint32_t vectors = dimension / 4;
    for (uint32_t index = lane; index < vectors; index += 32) {
      const char4 values = packed_document[index];
      const size_t base = static_cast<size_t>(index) * 4;
      dot = fmaf(scalar_to_float(query[base]), static_cast<float>(values.x), dot);
      dot = fmaf(scalar_to_float(query[base + 1]), static_cast<float>(values.y), dot);
      dot = fmaf(scalar_to_float(query[base + 2]), static_cast<float>(values.z), dot);
      dot = fmaf(scalar_to_float(query[base + 3]), static_cast<float>(values.w), dot);
    }
    return dot;
  }
  for (uint32_t index = lane; index < dimension; index += 32)
    dot = fmaf(scalar_to_float(query[index]),
               static_cast<float>(document[index]), dot);
  return dot;
}

template <typename QueryScalar>
__device__ float lane_dot_fp8(const QueryScalar *query, const uint8_t *document,
                              uint32_t dimension, uint32_t lane) {
  float dot = 0.0f;
  if ((dimension & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(document) & 3U) == 0) {
    const auto *packed_document = reinterpret_cast<const uint32_t *>(document);
    const uint32_t vectors = dimension / 4;
    for (uint32_t index = lane; index < vectors; index += 32) {
      __nv_fp8x4_e4m3 packed;
      packed.__x = packed_document[index];
      const float4 values = static_cast<float4>(packed);
      const size_t base = static_cast<size_t>(index) * 4;
      dot = fmaf(scalar_to_float(query[base]), values.x, dot);
      dot = fmaf(scalar_to_float(query[base + 1]), values.y, dot);
      dot = fmaf(scalar_to_float(query[base + 2]), values.z, dot);
      dot = fmaf(scalar_to_float(query[base + 3]), values.w, dot);
    }
    return dot;
  }
  for (uint32_t index = lane; index < dimension; index += 32)
    dot = fmaf(scalar_to_float(query[index]),
               e4m3fn_to_float(document[index]), dot);
  return dot;
}

template <>
__device__ float lane_dot<half>(const half *left, const half *right,
                                uint32_t dimension, uint32_t lane) {
  if ((dimension & 1U) != 0) {
    float scalar_dot = 0.0f;
    for (uint32_t index = lane; index < dimension; index += 32)
      scalar_dot = fmaf(__half2float(left[index]), __half2float(right[index]),
                        scalar_dot);
    return scalar_dot;
  }
  float dot = 0.0f;
  const uint32_t pairs = dimension / 2;
  const auto *left2 = reinterpret_cast<const half2 *>(left);
  const auto *right2 = reinterpret_cast<const half2 *>(right);
  for (uint32_t index = lane; index < pairs; index += 32) {
    const float2 a = __half22float2(left2[index]);
    const float2 b = __half22float2(right2[index]);
    dot = fmaf(a.x, b.x, dot);
    dot = fmaf(a.y, b.y, dot);
  }
  return dot;
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
      float dot = lane_dot(query_vector, document_vector, dimension, lane);
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
      float dot = lane_dot_int8(query_vector, document_vector, dimension, lane);
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

__device__ float native_e4m3fn_to_float(uint8_t bits) {
#if __CUDA_ARCH__ >= 890
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
#else
  return e4m3fn_to_float(bits);
#endif
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
      float dot = lane_dot_fp8(query_vector, document_vector, dimension, lane);
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

template <typename Scalar, uint32_t QueriesPerWarp, uint8_t ScoringProfile>
__global__ void tilemaxsim_multiquery_kernel(
    const Scalar *queries, uint32_t total_query_rows, uint32_t dimension,
    const unsigned char *documents, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t candidate_count, float *maxima) {
  extern __shared__ unsigned char shared_bytes[];
  auto *document_vector = reinterpret_cast<Scalar *>(shared_bytes);
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  constexpr uint32_t warps = 8;
  constexpr uint32_t query_tile_rows = warps * QueriesPerWarp;
  const uint32_t query_tiles =
      (total_query_rows + query_tile_rows - 1) / query_tile_rows;
  const size_t task_count = candidate_count * query_tiles;
  for (size_t task = blockIdx.x; task < task_count; task += gridDim.x) {
    const size_t candidate = task / query_tiles;
    const uint32_t query_base =
        static_cast<uint32_t>(task % query_tiles) * query_tile_rows;
    const unsigned char *encoded_document = documents + document_offsets[candidate];
    const auto *document = reinterpret_cast<const Scalar *>(encoded_document);
    const size_t code_bytes =
        static_cast<size_t>(document_rows[candidate]) * dimension;
    const size_t scale_offset = (code_bytes + 3) & ~static_cast<size_t>(3);
    const auto *scales = reinterpret_cast<const float *>(
        encoded_document + scale_offset);
    float best[QueriesPerWarp];
#pragma unroll
    for (uint32_t local = 0; local < QueriesPerWarp; ++local)
      best[local] = -CUDART_INF_F;
    for (uint32_t row = 0; row < document_rows[candidate]; ++row) {
      const Scalar *source = document + static_cast<size_t>(row) * dimension;
      if (ScoringProfile != 1) {
        const float scale = scales[row];
        for (uint32_t column = threadIdx.x; column < dimension;
             column += blockDim.x) {
          float value = 0.0f;
          const size_t index = static_cast<size_t>(row) * dimension + column;
          if (ScoringProfile == 2)
            value = static_cast<float>(
                reinterpret_cast<const int8_t *>(encoded_document)[index]);
          else
            value = native_e4m3fn_to_float(encoded_document[index]);
          document_vector[column] = scalar_from_float<Scalar>(value * scale);
        }
        __syncthreads();
      } else {
#if __CUDA_ARCH__ >= 800
      const size_t row_bytes = static_cast<size_t>(dimension) * sizeof(Scalar);
      if ((reinterpret_cast<uintptr_t>(source) & 15U) == 0 &&
          (row_bytes & 15U) == 0) {
        auto *destination_bytes =
            reinterpret_cast<unsigned char *>(document_vector);
        const auto *source_bytes =
            reinterpret_cast<const unsigned char *>(source);
        for (size_t byte = static_cast<size_t>(threadIdx.x) * 16;
             byte < row_bytes; byte += static_cast<size_t>(blockDim.x) * 16)
          __pipeline_memcpy_async(destination_bytes + byte,
                                  source_bytes + byte, 16);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
      } else {
        for (uint32_t column = threadIdx.x; column < dimension;
             column += blockDim.x)
          document_vector[column] = source[column];
        __syncthreads();
      }
#else
        for (uint32_t column = threadIdx.x; column < dimension; column += blockDim.x)
        document_vector[column] = source[column];
      __syncthreads();
#endif
      }
#pragma unroll
      for (uint32_t local = 0; local < QueriesPerWarp; ++local) {
        const uint32_t query_row = query_base + warp * QueriesPerWarp + local;
        if (query_row < total_query_rows) {
          const Scalar *query = queries + static_cast<size_t>(query_row) * dimension;
          float dot = lane_dot(query, document_vector, dimension, lane);
          for (int delta = 16; delta != 0; delta >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, delta);
          if (lane == 0) best[local] = fmaxf(best[local], dot);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (uint32_t local = 0; local < QueriesPerWarp; ++local) {
      const uint32_t query_row = query_base + warp * QueriesPerWarp + local;
      if (lane == 0 && query_row < total_query_rows)
        maxima[candidate * total_query_rows + query_row] = best[local];
    }
  }
}

template <typename Scalar>
static void launch_multiquery_tile(
    VctmGpu *gpu, const Scalar *queries, uint32_t total_query_rows,
    uint32_t dimension, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t count, float *maxima,
    size_t shared_bytes, uint8_t scoring_profile) {
  const uint32_t queries_per_warp = gpu->tile_queries_per_warp;
  const uint32_t query_tile_rows = 8 * queries_per_warp;
  const size_t tasks = count *
      ((static_cast<size_t>(total_query_rows) + query_tile_rows - 1) /
       query_tile_rows);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min(tasks, static_cast<size_t>(65'535)));
#define VCTM_LAUNCH_TILE(PROFILE, QPW)                                       \
  tilemaxsim_multiquery_kernel<Scalar, QPW, PROFILE>                        \
      <<<blocks, 256, shared_bytes, gpu->compute_stream>>>(                 \
          queries, total_query_rows, dimension, gpu->allocation,            \
          document_offsets, document_rows, count, maxima)
#define VCTM_DISPATCH_TILE(PROFILE)                                          \
  if (queries_per_warp == 8)                                                \
    VCTM_LAUNCH_TILE(PROFILE, 8);                                           \
  else if (queries_per_warp == 4)                                           \
    VCTM_LAUNCH_TILE(PROFILE, 4);                                           \
  else                                                                      \
    VCTM_LAUNCH_TILE(PROFILE, 1)
  if (scoring_profile == 1) {
    VCTM_DISPATCH_TILE(1);
  } else if (scoring_profile == 2) {
    VCTM_DISPATCH_TILE(2);
  } else {
    VCTM_DISPATCH_TILE(3);
  }
#undef VCTM_DISPATCH_TILE
#undef VCTM_LAUNCH_TILE
}

__global__ void tilemaxsim_segmented_sum_kernel(
    const float *maxima, uint32_t total_query_rows,
    const uint32_t *query_offsets, uint32_t request_count,
    size_t candidate_count, float *scores) {
  const size_t task_count = candidate_count * request_count;
  for (size_t task = blockIdx.x * blockDim.x + threadIdx.x; task < task_count;
       task += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t candidate = task / request_count;
    const uint32_t request = static_cast<uint32_t>(task % request_count);
    float score = 0.0f;
    for (uint32_t row = query_offsets[request]; row < query_offsets[request + 1]; ++row)
      score += maxima[candidate * total_query_rows + row];
    scores[static_cast<size_t>(request) * candidate_count + candidate] = score;
  }
}

extern "C" int vctm_gpu_score_batch(
    VctmGpu *gpu, const unsigned char *queries, size_t query_bytes,
    const uint32_t *query_offsets, uint32_t request_count,
    uint32_t total_query_rows, uint32_t dimension, uint8_t dtype,
    uint8_t scoring_profile,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  if (gpu == nullptr || queries == nullptr || query_offsets == nullptr || request_count < 2 ||
      total_query_rows == 0 || dimension == 0 || document_offsets == nullptr ||
      document_rows == nullptr || count == 0 || output == nullptr ||
      (scoring_profile != 1 && scoring_profile != 2 && scoring_profile != 3) ||
      query_offsets[0] != 0 || query_offsets[request_count] != total_query_rows)
    return fail(error, error_capacity, "invalid multi-query TileMaxSim request");
  const size_t scalar_bytes = dtype == 1 ? sizeof(float) : dtype == 2 ? sizeof(half) : 0;
  size_t expected_values = 0, expected_bytes = 0;
  if (scalar_bytes == 0 || !checked_mul(total_query_rows, dimension, &expected_values) ||
      !checked_mul(expected_values, scalar_bytes, &expected_bytes) || expected_bytes != query_bytes)
    return fail(error, error_capacity, "multi-query byte length disagrees with shape");
  for (size_t candidate = 0; candidate < count; ++candidate) {
    size_t values = 0, payload_bytes = 0;
    if (document_rows[candidate] == 0 ||
        !checked_mul(document_rows[candidate], dimension, &values))
      return fail(error, error_capacity, "invalid multi-query document shape");
    if (scoring_profile == 1) {
      if (!checked_mul(values, scalar_bytes, &payload_bytes))
        return fail(error, error_capacity, "multi-query document size overflow");
    } else {
      if (values > std::numeric_limits<size_t>::max() - 3)
        return fail(error, error_capacity, "quantized batch size overflow");
      const size_t aligned_codes = (values + 3) & ~static_cast<size_t>(3);
      size_t scale_bytes = 0;
      if (!checked_mul(document_rows[candidate], sizeof(float), &scale_bytes) ||
          aligned_codes > std::numeric_limits<size_t>::max() - scale_bytes)
        return fail(error, error_capacity, "quantized batch size overflow");
      payload_bytes = aligned_codes + scale_bytes;
    }
    if (document_offsets[candidate] > gpu->tensor_bytes ||
        payload_bytes > gpu->tensor_bytes - document_offsets[candidate])
      return fail(error, error_capacity, "multi-query document exceeds GPU arena");
  }
  for (uint32_t index = 0; index < request_count; ++index)
    if (query_offsets[index] >= query_offsets[index + 1])
      return fail(error, error_capacity, "multi-query offsets must be strictly increasing");
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  size_t maxima_count = 0, score_count = 0;
  if (!checked_mul(count, total_query_rows, &maxima_count) ||
      !checked_mul(count, request_count, &score_count))
    return fail(error, error_capacity, "multi-query workspace shape overflow");
  unsigned char *workspace = gpu->allocation + gpu->tensor_bytes;
  size_t cursor = 0, query_offset = 0, query_offsets_offset = 0,
         offsets_offset = 0, rows_offset = 0, maxima_offset = 0, scores_offset = 0;
  if (!reserve_aligned(&cursor, query_bytes, &query_offset) ||
      !reserve_aligned(&cursor, (request_count + 1) * sizeof(uint32_t), &query_offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint64_t), &offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint32_t), &rows_offset) ||
      !reserve_aligned(&cursor, maxima_count * sizeof(float), &maxima_offset) ||
      !reserve_aligned(&cursor, score_count * sizeof(float), &scores_offset) ||
      cursor > gpu->workspace_bytes)
    return fail(error, error_capacity, "multi-query request exceeds configured GPU workspace");
  status = cudaMemcpyAsync(workspace + query_offset, queries, query_bytes, cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + query_offsets_offset, query_offsets, (request_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + offsets_offset, document_offsets, count * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + rows_offset, document_rows, count * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  const size_t shared_bytes = static_cast<size_t>(dimension) * scalar_bytes;
  if (status == cudaSuccess && dtype == 2)
    launch_multiquery_tile(gpu,
        reinterpret_cast<const half *>(workspace + query_offset),
        total_query_rows, dimension,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset), count,
        reinterpret_cast<float *>(workspace + maxima_offset), shared_bytes,
        scoring_profile);
  else if (status == cudaSuccess && dtype == 1)
    launch_multiquery_tile(gpu,
        reinterpret_cast<const float *>(workspace + query_offset),
        total_query_rows, dimension,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset), count,
        reinterpret_cast<float *>(workspace + maxima_offset), shared_bytes,
        scoring_profile);
  if (status == cudaSuccess) status = cudaGetLastError();
  constexpr unsigned int threads = 256;
  if (status == cudaSuccess) tilemaxsim_segmented_sum_kernel<<<static_cast<unsigned int>(std::min((score_count + threads - 1) / threads, static_cast<size_t>(65'535))), threads, 0, gpu->compute_stream>>>(reinterpret_cast<const float *>(workspace + maxima_offset), total_query_rows, reinterpret_cast<const uint32_t *>(workspace + query_offsets_offset), request_count, count, reinterpret_cast<float *>(workspace + scores_offset));
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaMemcpyAsync(output, workspace + scores_offset, score_count * sizeof(float), cudaMemcpyDeviceToHost, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(gpu->compute_stream);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "multi-query TileMaxSim CUDA execution", status);
  return 0;
}

__global__ void tilemaxsim_batched_gemm_reduce_kernel(
    const float *similarities, uint32_t query_rows, uint32_t document_rows,
    const uint32_t *query_offsets, uint32_t request_count,
    const uint32_t *candidate_indexes, uint32_t group_count,
    size_t matrix_stride, size_t candidate_count, float *scores) {
  const size_t tasks = static_cast<size_t>(group_count) * request_count;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < tasks; task += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint32_t group = static_cast<uint32_t>(task / request_count);
    const uint32_t request = static_cast<uint32_t>(task % request_count);
    const float *matrix = similarities + static_cast<size_t>(group) * matrix_stride;
    float score = 0.0f;
    for (uint32_t query = query_offsets[request]; query < query_offsets[request + 1]; ++query) {
      float maximum = -CUDART_INF_F;
      for (uint32_t row = 0; row < document_rows; ++row)
        maximum = fmaxf(maximum, matrix[static_cast<size_t>(query) * document_rows + row]);
      score += maximum;
    }
    scores[static_cast<size_t>(request) * candidate_count + candidate_indexes[group]] = score;
  }
}

extern "C" int vctm_gpu_score_batch_tensor(
    VctmGpu *gpu, const unsigned char *queries, size_t query_bytes,
    const uint32_t *query_offsets, uint32_t request_count,
    uint32_t total_query_rows, uint32_t dimension,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  if (gpu == nullptr || queries == nullptr || query_offsets == nullptr || request_count < 2 ||
      total_query_rows == 0 || dimension == 0 || document_offsets == nullptr ||
      document_rows == nullptr || count == 0 || output == nullptr || query_offsets[0] != 0 ||
      query_offsets[request_count] != total_query_rows)
    return fail(error, error_capacity, "invalid tensor-core TileMaxSim request");
  if (total_query_rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      dimension > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    return fail(error, error_capacity, "tensor-core GEMM shape exceeds cuBLAS integer limits");
  for (uint32_t index = 0; index < request_count; ++index)
    if (query_offsets[index] >= query_offsets[index + 1])
      return fail(error, error_capacity, "tensor-core query offsets must be strictly increasing");
  size_t expected_values = 0, expected_bytes = 0;
  if (!checked_mul(total_query_rows, dimension, &expected_values) ||
      !checked_mul(expected_values, sizeof(half), &expected_bytes) || expected_bytes != query_bytes)
    return fail(error, error_capacity, "tensor-core query byte length disagrees with shape");
  constexpr size_t max_cached_tensor_candidates = 65'536;
  const bool cache_row_groups = count <= max_cached_tensor_candidates;
  const bool reuse_row_groups = cache_row_groups &&
      gpu->tensor_cached_document_rows.size() == count &&
      std::equal(gpu->tensor_cached_document_rows.begin(),
                 gpu->tensor_cached_document_rows.end(), document_rows);
  for (size_t candidate = 0; candidate < count; ++candidate) {
    if (document_rows[candidate] == 0 ||
        document_rows[candidate] > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        document_offsets[candidate] > gpu->tensor_bytes)
      return fail(error, error_capacity, "invalid tensor-core document descriptor");
    size_t values = 0, bytes = 0;
    if (!checked_mul(document_rows[candidate], dimension, &values) ||
        !checked_mul(values, sizeof(half), &bytes) || bytes > gpu->tensor_bytes - document_offsets[candidate])
      return fail(error, error_capacity, "tensor-core document exceeds GPU arena");
  }
  std::map<uint32_t, std::vector<uint32_t>> transient_row_groups;
  if (!reuse_row_groups) {
    auto &replacement = cache_row_groups ? gpu->tensor_cached_row_groups
                                         : transient_row_groups;
    replacement.clear();
    for (size_t candidate = 0; candidate < count; ++candidate)
      replacement[document_rows[candidate]].push_back(
          static_cast<uint32_t>(candidate));
    if (cache_row_groups)
      gpu->tensor_cached_document_rows.assign(document_rows,
                                               document_rows + count);
  }
  const auto &row_groups = cache_row_groups ? gpu->tensor_cached_row_groups
                                            : transient_row_groups;
  size_t score_values = 0;
  if (!checked_mul(request_count, count, &score_values))
    return fail(error, error_capacity, "tensor-core workspace shape overflow");
  unsigned char *workspace = gpu->allocation + gpu->tensor_bytes;
  size_t cursor = 0, query_offset = 0, query_offsets_offset = 0,
         scores_offset = 0;
  if (!reserve_aligned(&cursor, query_bytes, &query_offset) ||
      !reserve_aligned(&cursor, (request_count + 1) * sizeof(uint32_t), &query_offsets_offset) ||
      !reserve_aligned(&cursor, score_values * sizeof(float), &scores_offset) ||
      cursor > gpu->workspace_bytes)
    return fail(error, error_capacity, "tensor-core request exceeds configured GPU workspace");
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  status = cudaMemcpyAsync(workspace + query_offset, queries, query_bytes, cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + query_offsets_offset, query_offsets, (request_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "tensor-core workspace initialization", status);
  ScopedPersistingQuery query_policy(gpu, workspace + query_offset, query_bytes);
  const float alpha = 1.0f, beta = 0.0f;
  for (const auto &[rows, candidates] : row_groups) {
    size_t matrix_values = 0;
    size_t matrix_bytes = 0;
    if (!checked_mul(total_query_rows, rows, &matrix_values) ||
        !checked_mul(matrix_values, sizeof(float), &matrix_bytes))
      return fail(error, error_capacity, "batched GEMM matrix shape overflow");
    constexpr size_t metadata_bytes =
        3 * sizeof(void *) + sizeof(uint32_t) + 4 * 256;
    if (matrix_bytes > std::numeric_limits<size_t>::max() - metadata_bytes)
      return fail(error, error_capacity, "batched GEMM workspace shape overflow");
    const size_t per_candidate = matrix_bytes + metadata_bytes;
    const size_t available = gpu->workspace_bytes - cursor;
    const size_t chunk_capacity = std::max(
        static_cast<size_t>(1),
        std::min({candidates.size(), available / per_candidate,
                  max_cached_tensor_candidates}));
    if (available < per_candidate)
      return fail(error, error_capacity, "tensor-core batch exceeds configured GPU workspace");
    for (size_t begin = 0; begin < candidates.size(); begin += chunk_capacity) {
      const size_t batch = std::min(chunk_capacity, candidates.size() - begin);
      size_t scratch = cursor, matrix_offset = 0, a_offset = 0, b_offset = 0,
             c_offset = 0, candidates_offset = 0;
      if (!reserve_aligned(&scratch, batch * matrix_bytes, &matrix_offset) ||
          !reserve_aligned(&scratch, batch * sizeof(void *), &a_offset) ||
          !reserve_aligned(&scratch, batch * sizeof(void *), &b_offset) ||
          !reserve_aligned(&scratch, batch * sizeof(void *), &c_offset) ||
          !reserve_aligned(&scratch, batch * sizeof(uint32_t), &candidates_offset) ||
          scratch > gpu->workspace_bytes)
        return fail(error, error_capacity, "tensor-core batch workspace packing failed");
      gpu->tensor_a_pointers.resize(batch);
      gpu->tensor_b_pointers.resize(batch);
      gpu->tensor_c_pointers.resize(batch);
      gpu->tensor_candidate_indexes.resize(batch);
      for (size_t item = 0; item < batch; ++item) {
        const uint32_t candidate = candidates[begin + item];
        gpu->tensor_candidate_indexes[item] = candidate;
        gpu->tensor_a_pointers[item] =
            gpu->allocation + document_offsets[candidate];
        gpu->tensor_b_pointers[item] = workspace + query_offset;
        gpu->tensor_c_pointers[item] =
            workspace + matrix_offset + item * matrix_bytes;
      }
      status = cudaMemcpyAsync(workspace + a_offset, gpu->tensor_a_pointers.data(), batch * sizeof(void *), cudaMemcpyHostToDevice, gpu->compute_stream);
      if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + b_offset, gpu->tensor_b_pointers.data(), batch * sizeof(void *), cudaMemcpyHostToDevice, gpu->compute_stream);
      if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + c_offset, gpu->tensor_c_pointers.data(), batch * sizeof(void *), cudaMemcpyHostToDevice, gpu->compute_stream);
      if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + candidates_offset, gpu->tensor_candidate_indexes.data(), batch * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
      if (status != cudaSuccess) return cuda_fail(error, error_capacity, "batched GEMM metadata upload", status);
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(total_query_rows);
    const int k = static_cast<int>(dimension);
    cublasStatus_t blas = cublasGemmBatchedEx(
        gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
        reinterpret_cast<const void *const *>(workspace + a_offset), CUDA_R_16F, k,
        reinterpret_cast<const void *const *>(workspace + b_offset), CUDA_R_16F, k, &beta,
        reinterpret_cast<void *const *>(workspace + c_offset), CUDA_R_32F, m,
        static_cast<int>(batch), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas != CUBLAS_STATUS_SUCCESS)
      return cublas_fail(error, error_capacity, "cublasGemmBatchedEx", blas);
    const unsigned int threads = 128;
    const size_t reduction_tasks = batch * request_count;
    tilemaxsim_batched_gemm_reduce_kernel<<<static_cast<unsigned int>(std::min((reduction_tasks + threads - 1) / threads, static_cast<size_t>(65'535))), threads, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + matrix_offset), total_query_rows,
        rows, reinterpret_cast<const uint32_t *>(workspace + query_offsets_offset),
        request_count, reinterpret_cast<const uint32_t *>(workspace + candidates_offset),
        static_cast<uint32_t>(batch), matrix_values, count,
        reinterpret_cast<float *>(workspace + scores_offset));
    status = cudaGetLastError();
    if (status != cudaSuccess) return cuda_fail(error, error_capacity, "tensor-core reduction", status);
    }
  }
  status = cudaMemcpyAsync(output, workspace + scores_offset, score_values * sizeof(float), cudaMemcpyDeviceToHost, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(gpu->compute_stream);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "tensor-core TileMaxSim execution", status);
  return 0;
}

template <typename QueryScalar>
__global__ void rotate_queries_kernel(
    const QueryScalar *query, uint32_t query_rows,
    const VctmQuantizer quantizer, float *rotated, size_t count) {
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < count; task += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t cursor = task;
    const uint32_t destination = cursor % quantizer.dimension;
    cursor /= quantizer.dimension;
    const uint32_t query_row = cursor % query_rows;
    const size_t rotation_rank = cursor / query_rows;
    const float *rotation = quantizer.payload +
        rotation_rank * quantizer.dimension * quantizer.dimension;
    const QueryScalar *query_vector =
        query + static_cast<size_t>(query_row) * quantizer.dimension;
    float value = 0.0f;
    for (uint32_t source = 0; source < quantizer.dimension; ++source)
      value = fmaf(scalar_to_float(query_vector[source]),
                   rotation[static_cast<size_t>(source) * quantizer.dimension +
                            destination],
                   value);
    rotated[task] = value;
  }
}

template <typename QueryScalar>
__global__ void pq_lut_kernel(
    const QueryScalar *query, uint32_t query_rows, const VctmQuantizer quantizer,
    const float *rotated_queries, float *luts, size_t count) {
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
      query_value = rotated_queries[
          (rotation_rank * query_rows + query_row) * quantizer.dimension +
          dimension];
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

// Short and medium documents do not contain enough rows to amortize the
// block-wide barriers in the cooperative kernel above. Assign one independent
// candidate/query task to each warp so eight tasks share a block without
// shared memory or cross-warp synchronization. This preserves exact ADC and
// MaxSim semantics; only the mapping of work to warps changes.
__global__ void pq_adc_maxsim_warp_task_kernel(
    const float *luts, uint32_t query_rows, const VctmQuantizer quantizer,
    const unsigned char *documents, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t task_count, float *maxima) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp_in_block = threadIdx.x >> 5;
  const uint32_t warps_per_block = blockDim.x >> 5;
  size_t task = static_cast<size_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const size_t task_stride = static_cast<size_t>(gridDim.x) * warps_per_block;
  const uint32_t code_count =
      static_cast<uint32_t>(quantizer.stages) * quantizer.subspaces;
  for (; task < task_count; task += task_stride) {
    const size_t candidate = task / query_rows;
    const uint32_t query_row = static_cast<uint32_t>(task % query_rows);
    const uint8_t *codes = documents + document_offsets[candidate];
    const size_t lut_base = static_cast<size_t>(query_row) * code_count *
                            quantizer.centroids;
    float best = -CUDART_INF_F;
    for (uint32_t row = 0; row < document_rows[candidate]; ++row) {
      const size_t row_base = static_cast<size_t>(row) * code_count;
      float similarity = 0.0f;
      for (uint32_t flat = lane; flat < code_count; flat += 32) {
        const uint8_t code = codes[row_base + flat];
        similarity +=
            luts[lut_base + static_cast<size_t>(flat) * quantizer.centroids + code];
      }
      for (int delta = 16; delta != 0; delta >>= 1)
        similarity += __shfl_down_sync(0xffffffff, similarity, delta);
      if (lane == 0) best = fmaxf(best, similarity);
    }
    if (lane == 0) maxima[task] = best;
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

static int score_pq_batch_impl(
    VctmGpu *gpu, const VctmQuantizer *quantizer,
    const unsigned char *query, size_t query_bytes, uint32_t query_rows,
    const uint32_t *query_offsets, uint32_t request_count, uint8_t dtype,
    const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t count, float *output,
    char *error, size_t error_capacity) {
  if (gpu == nullptr || quantizer == nullptr || query == nullptr || query_rows == 0 ||
      query_offsets == nullptr || request_count == 0 || query_offsets[0] != 0 ||
      query_offsets[request_count] != query_rows ||
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
  uint32_t maximum_document_rows = 0;
  for (size_t index = 0; index < count; ++index) {
    size_t code_bytes = 0;
    if (document_rows[index] == 0 || !checked_mul(document_rows[index], codes_per_row, &code_bytes) ||
        document_offsets[index] > gpu->tensor_bytes ||
        code_bytes > gpu->tensor_bytes - static_cast<size_t>(document_offsets[index]))
      return fail(error, error_capacity, "PQ document range exceeds the GPU tensor arena");
    maximum_document_rows = std::max(maximum_document_rows, document_rows[index]);
  }
  cudaError_t status = cudaSetDevice(gpu->device);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "cudaSetDevice", status);
  for (uint32_t index = 0; index < request_count; ++index)
    if (query_offsets[index] >= query_offsets[index + 1])
      return fail(error, error_capacity, "PQ query offsets must be strictly increasing");
  size_t maxima_count = 0, score_count = 0, lut_count = 0, rotated_count = 0;
  const size_t rotations =
      __builtin_popcount(static_cast<unsigned int>(quantizer->rotation_mask));
  if (!checked_mul(count, query_rows, &maxima_count) ||
      !checked_mul(count, request_count, &score_count) ||
      !checked_mul(query_rows, quantizer->stages, &lut_count) ||
      !checked_mul(lut_count, quantizer->subspaces, &lut_count) ||
      !checked_mul(lut_count, quantizer->centroids, &lut_count) ||
      !checked_mul(rotations, query_rows, &rotated_count) ||
      !checked_mul(rotated_count, quantizer->dimension, &rotated_count))
    return fail(error, error_capacity, "PQ request shape overflows address space");
  size_t rotated_bytes = 0, lut_bytes = 0, maxima_bytes = 0;
  if (!checked_mul(rotated_count, sizeof(float), &rotated_bytes) ||
      !checked_mul(lut_count, sizeof(float), &lut_bytes) ||
      !checked_mul(maxima_count, sizeof(float), &maxima_bytes))
    return fail(error, error_capacity, "PQ workspace size overflows address space");
  unsigned char *workspace = gpu->allocation + gpu->tensor_bytes;
  size_t cursor = 0, query_offset = 0, query_offsets_offset = 0,
         offsets_offset = 0, rows_offset = 0,
         rotated_offset = 0, lut_offset = 0, maxima_offset = 0,
         scores_offset = 0;
  if (!reserve_aligned(&cursor, query_bytes, &query_offset) ||
      !reserve_aligned(&cursor, (request_count + 1) * sizeof(uint32_t), &query_offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint64_t), &offsets_offset) ||
      !reserve_aligned(&cursor, count * sizeof(uint32_t), &rows_offset) ||
      !reserve_aligned(&cursor, rotated_bytes, &rotated_offset) ||
      !reserve_aligned(&cursor, lut_bytes, &lut_offset) ||
      !reserve_aligned(&cursor, maxima_bytes, &maxima_offset) ||
      !reserve_aligned(&cursor, score_count * sizeof(float), &scores_offset) ||
      cursor > gpu->workspace_bytes)
    return fail(error, error_capacity, "PQ request exceeds configured GPU workspace");
  status = cudaMemcpyAsync(workspace + query_offset, query, query_bytes, cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + query_offsets_offset, query_offsets, (request_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + offsets_offset, document_offsets, count * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaMemcpyAsync(workspace + rows_offset, document_rows, count * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu->compute_stream);
  constexpr unsigned int threads = 256;
  if (status == cudaSuccess && rotated_count != 0) {
    const unsigned int rotation_blocks = static_cast<unsigned int>(std::min(
        (rotated_count + threads - 1) / threads,
        static_cast<size_t>(65'535)));
    if (dtype == 2)
      rotate_queries_kernel<half><<<rotation_blocks, threads, 0,
          gpu->compute_stream>>>(
          reinterpret_cast<const half *>(workspace + query_offset), query_rows,
          *quantizer, reinterpret_cast<float *>(workspace + rotated_offset),
          rotated_count);
    else if (dtype == 1)
      rotate_queries_kernel<float><<<rotation_blocks, threads, 0,
          gpu->compute_stream>>>(
          reinterpret_cast<const float *>(workspace + query_offset), query_rows,
          *quantizer, reinterpret_cast<float *>(workspace + rotated_offset),
          rotated_count);
    status = cudaGetLastError();
  }
  const unsigned int lut_blocks = static_cast<unsigned int>((lut_count + threads - 1) / threads);
  if (status == cudaSuccess && dtype == 2)
    pq_lut_kernel<half><<<lut_blocks, threads, 0, gpu->compute_stream>>>(reinterpret_cast<const half *>(workspace + query_offset), query_rows, *quantizer, reinterpret_cast<const float *>(workspace + rotated_offset), reinterpret_cast<float *>(workspace + lut_offset), lut_count);
  else if (status == cudaSuccess && dtype == 1)
    pq_lut_kernel<float><<<lut_blocks, threads, 0, gpu->compute_stream>>>(reinterpret_cast<const float *>(workspace + query_offset), query_rows, *quantizer, reinterpret_cast<const float *>(workspace + rotated_offset), reinterpret_cast<float *>(workspace + lut_offset), lut_count);
  else if (status == cudaSuccess) return fail(error, error_capacity, "unsupported PQ query dtype");
  if (status == cudaSuccess) status = cudaGetLastError();
  constexpr size_t warps_per_block = threads / 32;
  const bool use_warp_tasks = gpu->pq_warp_task_max_document_rows != 0 &&
                              maximum_document_rows <=
                                  gpu->pq_warp_task_max_document_rows;
  const size_t kernel_blocks = std::min(
      use_warp_tasks ? (maxima_count + warps_per_block - 1) / warps_per_block
                     : maxima_count,
      static_cast<size_t>(65'535));
  if (status == cudaSuccess && use_warp_tasks)
    pq_adc_maxsim_warp_task_kernel<<<static_cast<unsigned int>(kernel_blocks),
        threads, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + lut_offset), query_rows,
        *quantizer, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  else if (status == cudaSuccess)
    pq_adc_maxsim_kernel<<<static_cast<unsigned int>(kernel_blocks), threads, 0,
        gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + lut_offset), query_rows,
        *quantizer, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        maxima_count, reinterpret_cast<float *>(workspace + maxima_offset));
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) tilemaxsim_segmented_sum_kernel<<<static_cast<unsigned int>(std::min((score_count + threads - 1) / threads, static_cast<size_t>(65'535))), threads, 0, gpu->compute_stream>>>(
      reinterpret_cast<const float *>(workspace + maxima_offset), query_rows,
      reinterpret_cast<const uint32_t *>(workspace + query_offsets_offset),
      request_count, count, reinterpret_cast<float *>(workspace + scores_offset));
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaMemcpyAsync(output, workspace + scores_offset, score_count * sizeof(float), cudaMemcpyDeviceToHost, gpu->compute_stream);
  if (status == cudaSuccess) status = cudaStreamSynchronize(gpu->compute_stream);
  if (status != cudaSuccess) return cuda_fail(error, error_capacity, "PQ ADC-MaxSim CUDA execution", status);
  return 0;
}

extern "C" int vctm_gpu_score_pq(
    VctmGpu *gpu, const VctmQuantizer *quantizer,
    const unsigned char *query, size_t query_bytes, uint32_t query_rows,
    uint8_t dtype, const uint64_t *document_offsets,
    const uint32_t *document_rows, size_t count, float *output,
    char *error, size_t error_capacity) {
  const uint32_t query_offsets[2] = {0, query_rows};
  return score_pq_batch_impl(
      gpu, quantizer, query, query_bytes, query_rows, query_offsets, 1, dtype,
      document_offsets, document_rows, count, output, error, error_capacity);
}

extern "C" int vctm_gpu_score_pq_batch(
    VctmGpu *gpu, const VctmQuantizer *quantizer,
    const unsigned char *queries, size_t query_bytes,
    const uint32_t *query_offsets, uint32_t request_count,
    uint32_t total_query_rows, uint8_t dtype,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  return score_pq_batch_impl(
      gpu, quantizer, queries, query_bytes, total_query_rows, query_offsets,
      request_count, dtype, document_offsets, document_rows, count, output,
      error, error_capacity);
}
