// Copyright (c) 2026 HuXinjing

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
                                  const uint32_t *document_rows, float *scores) {
  const uint32_t candidate = blockIdx.x;
  const uint32_t query_row = blockIdx.y;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t warps = blockDim.x >> 5;
  const auto *document = reinterpret_cast<const Scalar *>(
      documents + document_offsets[candidate]);
  const Scalar *query_vector = query + static_cast<size_t>(query_row) * dimension;
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
  __shared__ float warp_best[8];
  if (lane == 0) warp_best[warp] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = -CUDART_INF_F;
    for (uint32_t index = 0; index < warps; ++index) {
      maximum = fmaxf(maximum, warp_best[index]);
    }
    atomicAdd(scores + candidate, maximum);
  }
}

static size_t aligned(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

extern "C" int vctm_gpu_score(
    VctmGpu *gpu, const unsigned char *query, size_t query_bytes,
    uint32_t query_rows, uint32_t dimension, uint8_t dtype,
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
  size_t cursor = 0;
  const size_t query_offset = cursor;
  cursor = aligned(cursor + query_bytes, 256);
  const size_t offsets_offset = cursor;
  cursor = aligned(cursor + count * sizeof(uint64_t), 256);
  const size_t rows_offset = cursor;
  cursor = aligned(cursor + count * sizeof(uint32_t), 256);
  const size_t scores_offset = cursor;
  cursor = aligned(cursor + count * sizeof(float), 256);
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
  if (status == cudaSuccess)
    status = cudaMemsetAsync(workspace + scores_offset, 0,
                             count * sizeof(float), gpu->compute_stream);
  if (status != cudaSuccess) {
    return cuda_fail(error, error_capacity, "CUDA workspace initialization",
                     status);
  }
  dim3 grid(static_cast<unsigned int>(count), query_rows);
  dim3 block(256);
  if (dtype == 2) {
    tilemaxsim_kernel<half><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const half *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        reinterpret_cast<float *>(workspace + scores_offset));
  } else if (dtype == 1) {
    tilemaxsim_kernel<float><<<grid, block, 0, gpu->compute_stream>>>(
        reinterpret_cast<const float *>(workspace + query_offset), query_rows,
        dimension, gpu->allocation,
        reinterpret_cast<const uint64_t *>(workspace + offsets_offset),
        reinterpret_cast<const uint32_t *>(workspace + rows_offset),
        reinterpret_cast<float *>(workspace + scores_offset));
  } else {
    return fail(error, error_capacity, "unsupported tensor dtype");
  }
  status = cudaGetLastError();
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
