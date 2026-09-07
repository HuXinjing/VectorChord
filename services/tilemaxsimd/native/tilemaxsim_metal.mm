// This software is licensed under the repository's dual license model.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

struct VctmMetal {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLBuffer> arena;
  id<MTLComputePipelineState> fp16_pipeline;
  id<MTLComputePipelineState> fp32_pipeline;
  size_t tensor_bytes;
  size_t workspace_bytes;
};

struct ScoreParameters {
  uint32_t query_rows;
  uint32_t dimension;
  uint32_t candidate_count;
};

static constexpr const char *kTileMaxSimSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ScoreParameters {
  uint query_rows;
  uint dimension;
  uint candidate_count;
};

#define DEFINE_MAXSIM(NAME, SCALAR)                                         \
kernel void NAME(                                                          \
    device const uchar *arena [[buffer(0)]],                               \
    device const ulong *document_offsets [[buffer(1)]],                    \
    device const uint *document_rows [[buffer(2)]],                        \
    device const SCALAR *query [[buffer(3)]],                              \
    device float *maxima [[buffer(4)]],                                    \
    constant ScoreParameters &parameters [[buffer(5)]],                    \
    uint task [[threadgroup_position_in_grid]],                            \
    uint tid [[thread_index_in_threadgroup]]) {                            \
  const uint candidate = task / parameters.query_rows;                     \
  const uint query_row = task % parameters.query_rows;                     \
  if (candidate >= parameters.candidate_count) return;                     \
  device const SCALAR *document =                                          \
      reinterpret_cast<device const SCALAR *>(                             \
          arena + document_offsets[candidate]);                            \
  threadgroup float partial[256];                                          \
  threadgroup float best;                                                  \
  if (tid == 0) best = -INFINITY;                                          \
  threadgroup_barrier(mem_flags::mem_threadgroup);                         \
  for (uint row = 0; row < document_rows[candidate]; ++row) {              \
    float dot = 0.0f;                                                      \
    for (uint column = tid; column < parameters.dimension; column += 256)  \
      dot = fma(float(query[query_row * parameters.dimension + column]),   \
                float(document[row * parameters.dimension + column]), dot);\
    partial[tid] = dot;                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    for (uint stride = 128; stride != 0; stride >>= 1) {                   \
      if (tid < stride) partial[tid] += partial[tid + stride];             \
      threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                      \
    if (tid == 0) best = max(best, partial[0]);                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
  }                                                                        \
  if (tid == 0) maxima[task] = best;                                       \
}

DEFINE_MAXSIM(tilemaxsim_f16, half)
DEFINE_MAXSIM(tilemaxsim_f32, float)
)METAL";

static int fail(char *error, size_t capacity, const char *message) {
  if (error != nullptr && capacity != 0)
    std::snprintf(error, capacity, "%s", message);
  return 1;
}

static int ns_fail(char *error, size_t capacity, const char *operation,
                   NSError *failure) {
  if (error != nullptr && capacity != 0) {
    const char *detail = failure == nil ? "unknown Metal error"
                                        : failure.localizedDescription.UTF8String;
    std::snprintf(error, capacity, "%s: %s", operation, detail);
  }
  return 1;
}

static bool checked_mul(size_t left, size_t right, size_t *output) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    return false;
  *output = left * right;
  return true;
}

static bool reserve_aligned(size_t *cursor, size_t bytes, size_t *offset) {
  constexpr size_t alignment = 256;
  *offset = *cursor;
  if (bytes > std::numeric_limits<size_t>::max() - *cursor) return false;
  const size_t end = *cursor + bytes;
  if (end > std::numeric_limits<size_t>::max() - alignment + 1) return false;
  *cursor = (end + alignment - 1) / alignment * alignment;
  return true;
}

extern "C" int vctm_metal_create(int ordinal, size_t total_bytes,
                                   size_t workspace_bytes,
                                   VctmMetal **output, char *error,
                                   size_t error_capacity) {
  @autoreleasepool {
    if (output == nullptr || ordinal < 0 || total_bytes == 0 ||
        workspace_bytes == 0 || workspace_bytes >= total_bytes)
      return fail(error, error_capacity, "invalid Metal arena configuration");
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (static_cast<NSUInteger>(ordinal) >= devices.count)
      return fail(error, error_capacity, "configured Metal device does not exist");
    id<MTLDevice> device = devices[static_cast<NSUInteger>(ordinal)];
    if (total_bytes > device.maxBufferLength)
      return fail(error, error_capacity, "Metal arena exceeds maxBufferLength");
    id<MTLBuffer> arena = [device newBufferWithLength:total_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (arena == nil || queue == nil)
      return fail(error, error_capacity, "Metal arena or command queue allocation failed");
    NSError *failure = nil;
    NSString *source = [NSString stringWithUTF8String:kTileMaxSimSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                 options:nil
                                                   error:&failure];
    if (library == nil)
      return ns_fail(error, error_capacity, "compile Metal TileMaxSim library", failure);
    id<MTLFunction> fp16 = [library newFunctionWithName:@"tilemaxsim_f16"];
    id<MTLFunction> fp32 = [library newFunctionWithName:@"tilemaxsim_f32"];
    id<MTLComputePipelineState> fp16_pipeline =
        [device newComputePipelineStateWithFunction:fp16 error:&failure];
    if (fp16_pipeline == nil)
      return ns_fail(error, error_capacity, "create Metal FP16 pipeline", failure);
    id<MTLComputePipelineState> fp32_pipeline =
        [device newComputePipelineStateWithFunction:fp32 error:&failure];
    if (fp32_pipeline == nil)
      return ns_fail(error, error_capacity, "create Metal FP32 pipeline", failure);
    auto *backend = new VctmMetal{};
    backend->device = device;
    backend->queue = queue;
    backend->arena = arena;
    backend->fp16_pipeline = fp16_pipeline;
    backend->fp32_pipeline = fp32_pipeline;
    backend->tensor_bytes = ((total_bytes - workspace_bytes) / 256) * 256;
    backend->workspace_bytes = total_bytes - backend->tensor_bytes;
    std::memset(arena.contents, 0, total_bytes);
    *output = backend;
    return 0;
  }
}

extern "C" void vctm_metal_destroy(VctmMetal *backend) {
  delete backend;
}

extern "C" size_t vctm_metal_tensor_bytes(const VctmMetal *backend) {
  return backend == nullptr ? 0 : backend->tensor_bytes;
}

extern "C" int vctm_metal_device_info(
    const VctmMetal *backend, char *name, size_t name_capacity,
    uint64_t *recommended_working_set_bytes, uint64_t *max_buffer_bytes) {
  @autoreleasepool {
    if (backend == nullptr || name == nullptr || name_capacity == 0 ||
        recommended_working_set_bytes == nullptr || max_buffer_bytes == nullptr)
      return 1;
    std::snprintf(name, name_capacity, "%s", backend->device.name.UTF8String);
    *recommended_working_set_bytes =
        static_cast<uint64_t>(backend->device.recommendedMaxWorkingSetSize);
    *max_buffer_bytes = static_cast<uint64_t>(backend->device.maxBufferLength);
    return 0;
  }
}

extern "C" int vctm_metal_upload_batch(
    VctmMetal *backend, const uint64_t *offsets,
    const unsigned char *const *payloads, const size_t *lengths, size_t count,
    char *error, size_t error_capacity) {
  if (backend == nullptr || offsets == nullptr || payloads == nullptr ||
      lengths == nullptr)
    return fail(error, error_capacity, "invalid Metal upload batch");
  auto *destination = static_cast<unsigned char *>(backend->arena.contents);
  for (size_t index = 0; index < count; ++index) {
    if (payloads[index] == nullptr || offsets[index] > backend->tensor_bytes ||
        lengths[index] > backend->tensor_bytes - offsets[index])
      return fail(error, error_capacity, "Metal upload exceeds tensor arena");
    std::memcpy(destination + offsets[index], payloads[index], lengths[index]);
  }
  return 0;
}

static int score_batch_impl(
    VctmMetal *backend, const unsigned char *query, size_t query_bytes,
    uint32_t query_rows, uint32_t dimension, uint8_t dtype,
    const uint32_t *query_offsets, uint32_t request_count,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  @autoreleasepool {
    if (backend == nullptr || query == nullptr || query_rows == 0 ||
        query_offsets == nullptr || request_count == 0 ||
        query_offsets[0] != 0 || query_offsets[request_count] != query_rows ||
        dimension == 0 || document_offsets == nullptr ||
        document_rows == nullptr || count == 0 ||
        count > std::numeric_limits<uint32_t>::max() || output == nullptr)
      return fail(error, error_capacity, "invalid Metal score request");
    for (uint32_t request = 0; request < request_count; ++request)
      if (query_offsets[request] >= query_offsets[request + 1])
        return fail(error, error_capacity,
                    "Metal query offsets must be strictly increasing");
    const size_t scalar_bytes = dtype == 1 ? sizeof(float) : dtype == 2 ? 2 : 0;
    size_t query_values = 0, expected_query_bytes = 0, maxima_count = 0,
           output_count = 0;
    if (scalar_bytes == 0 || !checked_mul(query_rows, dimension, &query_values) ||
        !checked_mul(query_values, scalar_bytes, &expected_query_bytes) ||
        expected_query_bytes != query_bytes ||
        !checked_mul(query_rows, count, &maxima_count) ||
        !checked_mul(request_count, count, &output_count))
      return fail(error, error_capacity, "Metal query shape is invalid");
    (void)output_count;
    for (size_t candidate = 0; candidate < count; ++candidate) {
      size_t values = 0, bytes = 0;
      if (document_rows[candidate] == 0 ||
          !checked_mul(document_rows[candidate], dimension, &values) ||
          !checked_mul(values, scalar_bytes, &bytes) ||
          document_offsets[candidate] > backend->tensor_bytes ||
          bytes > backend->tensor_bytes - document_offsets[candidate])
        return fail(error, error_capacity, "Metal document exceeds tensor arena");
    }
    size_t cursor = 0, query_offset = 0, offsets_offset = 0, rows_offset = 0,
           maxima_offset = 0, maxima_bytes = 0, offsets_bytes = 0,
           rows_bytes = 0;
    if (!checked_mul(maxima_count, sizeof(float), &maxima_bytes) ||
        !checked_mul(count, sizeof(uint64_t), &offsets_bytes) ||
        !checked_mul(count, sizeof(uint32_t), &rows_bytes) ||
        !reserve_aligned(&cursor, query_bytes, &query_offset) ||
        !reserve_aligned(&cursor, offsets_bytes, &offsets_offset) ||
        !reserve_aligned(&cursor, rows_bytes, &rows_offset) ||
        !reserve_aligned(&cursor, maxima_bytes, &maxima_offset) ||
        cursor > backend->workspace_bytes)
      return fail(error, error_capacity, "Metal score exceeds configured workspace");
    auto *workspace = static_cast<unsigned char *>(backend->arena.contents) +
                      backend->tensor_bytes;
    std::memcpy(workspace + query_offset, query, query_bytes);
    std::memcpy(workspace + offsets_offset, document_offsets, offsets_bytes);
    std::memcpy(workspace + rows_offset, document_rows, rows_bytes);
    id<MTLCommandBuffer> command = [backend->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline =
        dtype == 1 ? backend->fp32_pipeline : backend->fp16_pipeline;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:backend->arena offset:0 atIndex:0];
    [encoder setBuffer:backend->arena
                 offset:backend->tensor_bytes + offsets_offset atIndex:1];
    [encoder setBuffer:backend->arena
                 offset:backend->tensor_bytes + rows_offset atIndex:2];
    [encoder setBuffer:backend->arena
                 offset:backend->tensor_bytes + query_offset atIndex:3];
    [encoder setBuffer:backend->arena
                 offset:backend->tensor_bytes + maxima_offset atIndex:4];
    ScoreParameters parameters{query_rows, dimension,
                               static_cast<uint32_t>(count)};
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(maxima_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
      return ns_fail(error, error_capacity, "Metal TileMaxSim execution",
                     command.error);
    const auto *maxima = reinterpret_cast<const float *>(workspace + maxima_offset);
    for (uint32_t request = 0; request < request_count; ++request) {
      for (size_t candidate = 0; candidate < count; ++candidate) {
        float score = 0.0f;
        for (uint32_t row = query_offsets[request];
             row < query_offsets[request + 1]; ++row)
          score += maxima[candidate * query_rows + row];
        output[static_cast<size_t>(request) * count + candidate] = score;
      }
    }
    return 0;
  }
}

extern "C" int vctm_metal_score(
    VctmMetal *backend, const unsigned char *query, size_t query_bytes,
    uint32_t query_rows, uint32_t dimension, uint8_t dtype,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  const uint32_t query_offsets[2] = {0, query_rows};
  return score_batch_impl(
      backend, query, query_bytes, query_rows, dimension, dtype, query_offsets,
      1, document_offsets, document_rows, count, output, error,
      error_capacity);
}

extern "C" int vctm_metal_score_batch(
    VctmMetal *backend, const unsigned char *queries, size_t query_bytes,
    const uint32_t *query_offsets, uint32_t request_count,
    uint32_t total_query_rows, uint32_t dimension, uint8_t dtype,
    const uint64_t *document_offsets, const uint32_t *document_rows,
    size_t count, float *output, char *error, size_t error_capacity) {
  return score_batch_impl(
      backend, queries, query_bytes, total_query_rows, dimension, dtype,
      query_offsets, request_count, document_offsets, document_rows, count,
      output, error, error_capacity);
}
