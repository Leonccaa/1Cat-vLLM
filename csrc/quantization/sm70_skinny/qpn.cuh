// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
// Derived from https://github.com/dnv2003/v100-skinny (MIT License).
// Adapted for 1Cat-vLLM's format-independent SM70 execution layer.

#pragma once

#include "common.cuh"

namespace vllm::sm70_skinny {

// Quadpair-N core for M=4..16. FormatPolicy supplies one already-prepacked
// K16/N32 B fragment; the core owns activation reuse, Volta mma.m8n8k4,
// four-warp K partitioning, and the sole cross-warp reduction barrier.
//
// Launch geometry note. One block owns one N32 tile across the whole of K, so
// without KSplits the grid is exactly N/32 blocks of 128 threads regardless of
// K. For the per-rank N that TP4 decode actually produces (roughly 1k..4k)
// that is 32..128 blocks on an 80-SM V100 - at the low end more than half the
// SMs never receive work, and there are nowhere near enough concurrent loads
// in flight to reach the HBM roof. SIMT does not have this problem because its
// grid is N/8 blocks of 256 threads, i.e. 8x the threads for the same N.
//
// KSplits fixes that by also partitioning K across blocks: grid becomes
// (N/32, KSplits) and each block accumulates its own K slice. KSplits == 1
// keeps the original single-pass behaviour and writes FP16 directly; KSplits
// > 1 writes FP32 partials that qpn_reduce_kernel sums in a fixed split order,
// so the result stays deterministic.
template <typename FormatPolicy, int MT, typename OutT, bool Split>
__global__ void qpn_kernel(typename FormatPolicy::Params params,
                           const half* __restrict__ input,
                           OutT* __restrict__ output, int output_size,
                           int input_size, int rows, int k_splits) {
  constexpr int kWarps = 4;
  __shared__ float partials[kWarps][MT * 256];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int split = Split ? blockIdx.y : 0;
  const int quadpair = (lane >> 2) & 3;
  const int local_row =
      (lane & 3) + ((lane & 16) ? 4 : 0);  // A row and B local column
  const int group_count = input_size / 16;
  // The launcher guarantees group_count % (kWarps * k_splits) == 0, so every
  // warp of every split gets the same number of K16 groups and no group is
  // silently dropped. With Split=false the compiler sees k_splits==1 and drops
  // the extra integer division from the prologue.
  const int groups_per_split = Split ? group_count / k_splits : group_count;
  const int groups_per_warp = groups_per_split / kWarps;
  const int first_group =
      (Split ? split * groups_per_split : 0) + warp * groups_per_warp;
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
      const size_t offset = static_cast<size_t>(row) * output_size +
                            static_cast<size_t>(tile) * 32 + col;
      if constexpr (Split) {
        output[static_cast<size_t>(split) * rows * output_size + offset] =
            value;
      } else {
        output[offset] = __float2half(value);
      }
    }
  }
}

// Deterministic fixed-order reduction of the KSplits > 1 FP32 partials.
//
// Templated purely for linkage: this header is included by both skinny_awq.cu
// and skinny_nvfp4.cu, and a plain __global__ in a header would emit the same
// strong symbol in both translation units.
template <typename Accum>
__global__ void qpn_reduce_kernel(const Accum* __restrict__ partials,
                                  half* __restrict__ output, int rows,
                                  int output_size, int k_splits) {
  const size_t total = static_cast<size_t>(rows) * output_size;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += stride) {
    Accum sum = 0.0f;
    for (int split = 0; split < k_splits; ++split) {
      sum += partials[static_cast<size_t>(split) * total + index];
    }
    output[index] = __float2half(sum);
  }
}

// Two blocks per SM on the 80-SM V100 PCIe parts these kernels target.
constexpr int kQpnTargetBlocks = 160;

// Runs the split-K pass and, when needed, the deterministic reduction.
// `workspace` may be null when k_splits == 1.
template <typename FormatPolicy, int MT>
void launch_qpn_kernel(typename FormatPolicy::Params params, const half* input,
                       half* output, float* workspace, int output_size,
                       int input_size, int rows, int k_splits,
                       cudaStream_t stream) {
  const dim3 grid(output_size / 32, k_splits);
  const dim3 block(128);
  if (k_splits == 1) {
    qpn_kernel<FormatPolicy, MT, half, false><<<grid, block, 0, stream>>>(
        params, input, output, output_size, input_size, rows, 1);
    return;
  }
  qpn_kernel<FormatPolicy, MT, float, true><<<grid, block, 0, stream>>>(
      params, input, workspace, output_size, input_size, rows, k_splits);
  const size_t total = static_cast<size_t>(rows) * output_size;
  const int reduce_block = 256;
  size_t reduce_grid = (total + reduce_block - 1) / reduce_block;
  if (reduce_grid > 65535) {
    reduce_grid = 65535;
  }
  qpn_reduce_kernel<float>
      <<<static_cast<int>(reduce_grid), reduce_block, 0, stream>>>(
          workspace, output, rows, output_size, k_splits);
}

// Smallest power-of-two K split that brings the block count up to
// target_blocks, subject to two constraints: every warp of every split must
// receive a whole number of K16 groups (otherwise groups are silently
// dropped), and each warp must keep enough groups that the per-block shared
// reduction and output write stay amortized.
constexpr int qpn_choose_k_splits(int input_size, int output_size,
                                  int target_blocks) {
  constexpr int kWarps = 4;
  constexpr int kMinGroupsPerWarp = 4;
  constexpr int kMaxSplits = 16;
  const int group_count = input_size / 16;
  const int tiles = output_size / 32;
  int splits = 1;
  while (splits < kMaxSplits) {
    if (tiles * splits >= target_blocks) {
      break;
    }
    const int next = splits * 2;
    if (group_count % (kWarps * next) != 0) {
      break;
    }
    if (group_count / (kWarps * next) < kMinGroupsPerWarp) {
      break;
    }
    splits = next;
  }
  return splits;
}

// Keep this contract table in sync with the parametrized Python mirror test
// in tests/quantization/test_sm70_skinny_awq.py.  These include every dense
// Qwen3.6-27B TP4 shape plus split and minimum-work boundary cases.
static_assert(qpn_choose_k_splits(1536, 5120, 160) == 1);
static_assert(qpn_choose_k_splits(4352, 5120, 160) == 1);
static_assert(qpn_choose_k_splits(5120, 8704, 160) == 1);
static_assert(qpn_choose_k_splits(5120, 4096, 160) == 2);
static_assert(qpn_choose_k_splits(5120, 1792, 160) == 4);
static_assert(qpn_choose_k_splits(128, 32, 160) == 1);
static_assert(qpn_choose_k_splits(512, 32, 160) == 2);
static_assert(qpn_choose_k_splits(4096, 32, 160) == 16);

}  // namespace vllm::sm70_skinny
