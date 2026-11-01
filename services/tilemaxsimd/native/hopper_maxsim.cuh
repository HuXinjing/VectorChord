// Hopper FP16 MaxSim, using the validated register-max epilogue.
// The inline WGMMA instruction wrapper is adapted from NVIDIA CUTLASS/CuTe.
// Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See CUTLASS_LICENSE.txt for the retained BSD-3-Clause license.
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>
namespace vctm_hopper {
constexpr uint32_t Tile=64;
__device__ uint32_t fused_shared_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ uint64_t fused_matrix_descriptor(const uint16_t *pointer) {
  // K-major 16-bit elements, 128-byte swizzle; eight rows per stride atom.
  return ((static_cast<uint64_t>(fused_shared_address(pointer)) >> 4) & 0x3fffULL) |
         (64ULL << 32) | (1ULL << 62);
}

__device__ void fused_load_tile(void *destination, const CUtensorMap &map,
                          uint32_t k, uint32_t row, uint64_t *barrier) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      : : "r"(fused_shared_address(destination)), "l"(&map), "r"(k), "r"(row),
          "r"(fused_shared_address(barrier)) : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ void fused_wait_tile(uint64_t *barrier, uint32_t phase) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{ .reg .pred ready; WAIT_TILE: "
      "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra WAIT_TILE; }"
      : : "r"(fused_shared_address(barrier)), "r"(phase) : "memory");
#else
  asm volatile("trap;");
#endif
}

template <uint32_t Columns, size_t... I>
__device__ __forceinline__ void pipelined_multiply(uint64_t a,uint64_t b,float (&accumulator)[Columns/2],std::index_sequence<I...>) {
  static_assert(Columns==128);
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; "
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0; }"
    : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3]), "+f"(accumulator[4]), "+f"(accumulator[5]), "+f"(accumulator[6]), "+f"(accumulator[7]), "+f"(accumulator[8]), "+f"(accumulator[9]), "+f"(accumulator[10]), "+f"(accumulator[11]), "+f"(accumulator[12]), "+f"(accumulator[13]), "+f"(accumulator[14]), "+f"(accumulator[15]), "+f"(accumulator[16]), "+f"(accumulator[17]), "+f"(accumulator[18]), "+f"(accumulator[19]), "+f"(accumulator[20]), "+f"(accumulator[21]), "+f"(accumulator[22]), "+f"(accumulator[23]), "+f"(accumulator[24]), "+f"(accumulator[25]), "+f"(accumulator[26]), "+f"(accumulator[27]), "+f"(accumulator[28]), "+f"(accumulator[29]), "+f"(accumulator[30]), "+f"(accumulator[31]), "+f"(accumulator[32]), "+f"(accumulator[33]), "+f"(accumulator[34]), "+f"(accumulator[35]), "+f"(accumulator[36]), "+f"(accumulator[37]), "+f"(accumulator[38]), "+f"(accumulator[39]), "+f"(accumulator[40]), "+f"(accumulator[41]), "+f"(accumulator[42]), "+f"(accumulator[43]), "+f"(accumulator[44]), "+f"(accumulator[45]), "+f"(accumulator[46]), "+f"(accumulator[47]), "+f"(accumulator[48]), "+f"(accumulator[49]), "+f"(accumulator[50]), "+f"(accumulator[51]), "+f"(accumulator[52]), "+f"(accumulator[53]), "+f"(accumulator[54]), "+f"(accumulator[55]), "+f"(accumulator[56]), "+f"(accumulator[57]), "+f"(accumulator[58]), "+f"(accumulator[59]), "+f"(accumulator[60]), "+f"(accumulator[61]), "+f"(accumulator[62]), "+f"(accumulator[63])
    : "l"(a),"l"(b),"r"(1));
#endif
}
template <uint32_t Columns, uint32_t QueryTiles, uint32_t Stages>
__global__ void pipelined_fp16_kernel(
    const __grid_constant__ CUtensorMap query_map,
    uint32_t query_rows, uint32_t document_rows, uint32_t dimension,
    uint32_t tiles_per_document, float *output,
    const CUtensorMap *document_maps, uint32_t document_count) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(128) uint16_t buffers[];
  __shared__ __align__(8) uint64_t barriers[Stages];
  constexpr uint32_t QueryElements=Tile*Tile;
  constexpr uint32_t StageElements=QueryTiles*QueryElements+Columns*Tile;
  const uint32_t query_tiles=(query_rows+Tile*QueryTiles-1)/(Tile*QueryTiles);
  const uint64_t task=uint64_t(blockIdx.y)*gridDim.x+blockIdx.x;
  const uint32_t query_base=uint32_t(task%query_tiles)*Tile*QueryTiles;
  const uint64_t document_task=task/query_tiles;
  const uint32_t candidate=uint32_t(document_task/tiles_per_document);
  const uint32_t tile=uint32_t(document_task%tiles_per_document);
  const bool second=QueryTiles==2 && query_base+Tile<query_rows;
  if(threadIdx.x==0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" : : "l"(&document_maps[candidate]) : "memory");
    for(uint32_t i=0;i<Stages;i++)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(fused_shared_address(&barriers[i])) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
  }
  __syncthreads();
  const auto prefetch=[&](uint32_t block) {
    if(threadIdx.x==0) {
      const uint32_t stage=block%Stages;
      uint16_t *base=buffers+stage*StageElements;
      const uint32_t query_count=1+uint32_t(second);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : : "r"(fused_shared_address(&barriers[stage])),
            "r"(uint32_t((query_count*QueryElements+Columns*Tile)*sizeof(uint16_t))) : "memory");
      fused_load_tile(base,query_map,block*Tile,query_base,&barriers[stage]);
      if constexpr(QueryTiles==2)
        if(second)fused_load_tile(base+QueryElements,query_map,block*Tile,query_base+Tile,&barriers[stage]);
      fused_load_tile(base+QueryTiles*QueryElements,document_maps[candidate],block*Tile,tile*Columns,&barriers[stage]);
    }
  };
  const uint32_t blocks=dimension/Tile;
  for(uint32_t i=0;i<Stages && i<blocks;i++)prefetch(i);
  float accumulator[QueryTiles][Columns/2]={};
  for(uint32_t block=0;block<blocks;block++) {
    const uint32_t stage=block%Stages;
    uint16_t *base=buffers+stage*StageElements;
    fused_wait_tile(&barriers[stage],(block/Stages)&1);
    #pragma unroll
    for(uint32_t qt=0;qt<QueryTiles;qt++) {
      if(qt && !second)continue;
      #pragma unroll
      for(uint32_t i=0;i<Columns/2;i++)asm volatile("" : "+f"(accumulator[qt][i]) : : "memory");
    }
    asm volatile("wgmma.fence.sync.aligned;" : : : "memory");
    #pragma unroll
    for(uint32_t k=0;k<Tile;k+=16) {
      #pragma unroll
      for(uint32_t qt=0;qt<QueryTiles;qt++) {
        if(qt && !second)continue;
        pipelined_multiply<Columns>(fused_matrix_descriptor(base+qt*QueryElements+k),
          fused_matrix_descriptor(base+QueryTiles*QueryElements+k),accumulator[qt],std::make_index_sequence<Columns/2>{});
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" : : : "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" : : : "memory");
    #pragma unroll
    for(uint32_t qt=0;qt<QueryTiles;qt++) {
      if(qt && !second)continue;
      #pragma unroll
      for(uint32_t i=0;i<Columns/2;i++)asm volatile("" : "+f"(accumulator[qt][i]) : : "memory");
    }
    __syncthreads();
    if(block+Stages<blocks)prefetch(block+Stages);
  }
  const uint32_t lane=threadIdx.x%32;
  const uint32_t local_row=(threadIdx.x/32)*16+lane/4;
  #pragma unroll
  for(uint32_t qt=0;qt<QueryTiles;qt++) {
    if(qt && !second)continue;
    float maxima[2]={-CUDART_INF_F,-CUDART_INF_F};
    #pragma unroll
    for(uint32_t i=0;i<Columns/2;i++) {
      const uint32_t column=tile*Columns+(lane%4)*2+(i/4)*8+i%2;
      if(column<document_rows)maxima[(i%4)/2]=fmaxf(maxima[(i%4)/2],accumulator[qt][i]);
    }
    #pragma unroll
    for(uint32_t part=0;part<2;part++) {
      float maximum=maxima[part];
      maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,2,4));
      maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,1,4));
      const uint32_t row=query_base+qt*Tile+local_row+part*8;
      if(lane%4==0 && row<query_rows)
        output[(size_t(candidate)*query_rows+row)*tiles_per_document+tile]=maximum;
    }
  }
#else
  if(threadIdx.x==0)asm volatile("trap;");
#endif
}
__global__ void fused_merge_maxima(const float *partials, uint32_t query_rows,
    uint32_t tiles, const uint32_t *query_offsets, uint32_t requests,
    const uint32_t *candidate_indexes, uint32_t groups, size_t candidates,
    float *output) {
  const size_t tasks = static_cast<size_t>(groups) * requests;
  for (size_t task = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < tasks; task += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint32_t group = task / requests;
    const uint32_t request = task % requests;
    const uint32_t candidate = candidate_indexes[group];
    if (candidate >= candidates) continue;
    float score = 0.0f;
    for (uint32_t query = query_offsets[request]; query < query_offsets[request + 1]; ++query) {
      float maximum = -CUDART_INF_F;
      for (uint32_t tile = 0; tile < tiles; ++tile)
        maximum = fmaxf(maximum, partials[(static_cast<size_t>(group) * query_rows + query) * tiles + tile]);
      score += maximum;
    }
    output[static_cast<size_t>(request) * candidates + candidate] = score;
  }
}
}
