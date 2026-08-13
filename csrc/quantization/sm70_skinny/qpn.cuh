// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
// Adapted for 1Cat-vLLM's format-independent SM70 execution layer.

#pragma once

#include "common.cuh"

namespace vllm::sm70_skinny {

// Quadpair-N core for M=4..16. FormatPolicy supplies one already-prepacked
// K16/N32 B fragment; the core owns activation reuse, Volta mma.m8n8k4,
// four-warp K partitioning, and the sole cross-warp reduction barrier.
template <typename FormatPolicy, int MT>
__global__ void qpn_kernel(typename FormatPolicy::Params params,
                           const half* __restrict__ input,
                           half* __restrict__ output, int output_size,
                           int input_size, int rows) {
  constexpr int kWarps = 4;
  __shared__ float partials[kWarps][MT * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int local_row =
      (lane & 3) + ((lane & 16) ? 4 : 0);  // A row and B local column
  const int group_count = input_size / 16;
  const int groups_per_warp = group_count / kWarps;
  const int first_group = warp * groups_per_warp;
  const auto thread_state = FormatPolicy::make_thread_state(params);

  float accumulator[MT][8];
#pragma unroll
  for (int tile_m = 0; tile_m < MT; ++tile_m) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accumulator[tile_m][index] = 0.0f;
    }
  }

#pragma unroll 4
  for (int group = first_group; group < first_group + groups_per_warp;
       ++group) {
    half2 weights[8];
    FormatPolicy::load_fragment(params, thread_state, tile, group, lane,
                                weights);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int tile_m = 0; tile_m < MT; ++tile_m) {
      const int row = tile_m * 8 + local_row;
      uint4 input_01 = make_uint4(0, 0, 0, 0);
      uint4 input_23 = make_uint4(0, 0, 0, 0);
      if (row < rows) {
        const half* input_row = input + static_cast<size_t>(row) * input_size;
        input_01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input_23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }
      const unsigned* a0 = reinterpret_cast<const unsigned*>(&input_01);
      const unsigned* a1 = reinterpret_cast<const unsigned*>(&input_23);
      mma_m8n8k4(accumulator[tile_m], a0[0], a0[1], b[0], b[1]);
      mma_m8n8k4(accumulator[tile_m], a0[2], a0[3], b[2], b[3]);
      mma_m8n8k4(accumulator[tile_m], a1[0], a1[1], b[4], b[5]);
      mma_m8n8k4(accumulator[tile_m], a1[2], a1[3], b[6], b[7]);
    }
  }

#pragma unroll
  for (int tile_m = 0; tile_m < MT; ++tile_m) {
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int row = (index & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col =
          (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
      partials[warp][(tile_m * 8 + row) * 32 + quadpair * 8 + col] =
          accumulator[tile_m][index];
    }
  }
  __syncthreads();
  for (int element = threadIdx.x; element < MT * 256; element += blockDim.x) {
    const float value = partials[0][element] + partials[1][element] +
                        partials[2][element] + partials[3][element];
    const int row = element >> 5;
    const int col = element & 31;
    if (row < rows) {
      output[static_cast<size_t>(row) * output_size +
             static_cast<size_t>(tile) * 32 + col] = __float2half(value);
    }
  }
}

}  // namespace vllm::sm70_skinny
