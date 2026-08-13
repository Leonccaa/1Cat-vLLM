// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
// Adapted for 1Cat-vLLM's format-independent SM70 execution layer.

#pragma once

#include "common.cuh"

namespace vllm::sm70_skinny {

// FormatPolicy contract:
//   Params, ThreadState, Segment
//   make_thread_state(params)
//   stage_pairs(dst, base_pair, uint4)
//   load_segment(params, state, output_col, absolute_k, segment)
//   decode_word(segment, word_index, half2[4])
// A segment is deliberately fixed at 16 K values. Both NVFP4 group-16 and
// AWQ group-128 can stream through this core; only metadata cadence and the
// register decoder differ.
template <typename FormatPolicy, int M, int KC, int RowsPerWarp = 1,
          bool Argmax = false>
__global__ void simt_kernel(typename FormatPolicy::Params params,
                            const half* __restrict__ input,
                            half* __restrict__ output, int output_size,
                            int input_size, half* __restrict__ block_values,
                            int* __restrict__ block_indices) {
  static_assert(FormatPolicy::kSegmentK == 16,
                "SM70 Skinny SIMT core requires K16 decode segments.");
  extern __shared__ char shared_raw[];
  half2* staged_input = reinterpret_cast<half2*>(shared_raw);
  constexpr int kPairsPerChunk = KC / 2;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_col =
      (static_cast<int>(blockIdx.x) * 8 + warp) * RowsPerWarp;
  const auto thread_state = FormatPolicy::make_thread_state(params);

  float accumulator[RowsPerWarp][M];
#pragma unroll
  for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      accumulator[row][m] = 0.0f;
    }
  }

  int chunk_start = 0;
  for (; chunk_start + KC <= input_size; chunk_start += KC) {
    __syncthreads();
    for (int index = threadIdx.x; index < M * (KC / 8); index += blockDim.x) {
      const int m = index / (KC / 8);
      const int vector_index = index % (KC / 8);
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(m) * input_size + chunk_start +
          vector_index * 8);
      FormatPolicy::stage_pairs(staged_input + m * kPairsPerChunk,
                                vector_index * 4, value);
    }
    __syncthreads();

#pragma unroll
    for (int iteration = 0; iteration < KC / 512; ++iteration) {
      const int segment = lane + 32 * iteration;
      typename FormatPolicy::Segment packed[RowsPerWarp];
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
        FormatPolicy::load_segment(params, thread_state, output_col + row,
                                   chunk_start + segment * 16, packed[row]);
      }

      // Limit the FP16 accumulation window to one K16 segment, then flush to
      // FP32. This lets integer decode and HFMA2 issue on separate Volta pipes
      // without allowing long-K activation outliers to overflow half range.
      half2 half_accumulator[RowsPerWarp][M];
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          half_accumulator[row][m] = __float2half2_rn(0.0f);
        }
      }
#pragma unroll
      for (int word = 0; word < 2; ++word) {
        half2 weights[RowsPerWarp][4];
#pragma unroll
        for (int row = 0; row < RowsPerWarp; ++row) {
          FormatPolicy::decode_word(packed[row], word, weights[row]);
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const int staged_index =
              swizzle_pair_index(segment * 8 + word * 4 + pair);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const half2 activation =
                staged_input[m * kPairsPerChunk + staged_index];
#pragma unroll
            for (int row = 0; row < RowsPerWarp; ++row) {
              half_accumulator[row][m] = __hfma2(weights[row][pair], activation,
                                                 half_accumulator[row][m]);
            }
          }
        }
      }
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float2 value = __half22float2(half_accumulator[row][m]);
          accumulator[row][m] += value.x + value.y;
        }
      }
    }
  }

  const int tail = input_size - chunk_start;
  if (tail > 0) {
    __syncthreads();
    for (int index = threadIdx.x; index < M * (tail / 8); index += blockDim.x) {
      const int m = index / (tail / 8);
      const int vector_index = index % (tail / 8);
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(m) * input_size + chunk_start +
          vector_index * 8);
      FormatPolicy::stage_pairs(staged_input + m * kPairsPerChunk,
                                vector_index * 4, value);
    }
    __syncthreads();
    const int segment_count = tail / 16;
    for (int segment = lane; segment < segment_count; segment += 32) {
      typename FormatPolicy::Segment packed[RowsPerWarp];
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
        FormatPolicy::load_segment(params, thread_state, output_col + row,
                                   chunk_start + segment * 16, packed[row]);
      }
      half2 half_accumulator[RowsPerWarp][M];
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          half_accumulator[row][m] = __float2half2_rn(0.0f);
        }
      }
#pragma unroll
      for (int word = 0; word < 2; ++word) {
        half2 weights[RowsPerWarp][4];
#pragma unroll
        for (int row = 0; row < RowsPerWarp; ++row) {
          FormatPolicy::decode_word(packed[row], word, weights[row]);
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const int staged_index =
              swizzle_pair_index(segment * 8 + word * 4 + pair);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const half2 activation =
                staged_input[m * kPairsPerChunk + staged_index];
#pragma unroll
            for (int row = 0; row < RowsPerWarp; ++row) {
              half_accumulator[row][m] = __hfma2(weights[row][pair], activation,
                                                 half_accumulator[row][m]);
            }
          }
        }
      }
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float2 value = __half22float2(half_accumulator[row][m]);
          accumulator[row][m] += value.x + value.y;
        }
      }
    }
  }

  if constexpr (!Argmax) {
#pragma unroll
    for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float value = warp_sum(accumulator[row][m]);
        if (lane == 0) {
          output[static_cast<size_t>(m) * output_size + output_col + row] =
              __float2half(value);
        }
      }
    }
  } else {
    static_assert(M == 1, "fused argmax supports exact M=1");
    __shared__ half warp_value[8];
    __shared__ int warp_index[8];
    half best_value = __float2half(-3.0e38f);
    int best_index = -1;
#pragma unroll
    for (int row = 0; row < RowsPerWarp; ++row) {
      const float value = warp_sum(accumulator[row][0]);
      if (lane == 0) {
        const half rounded = __float2half(value);
        if (__hgt(rounded, best_value)) {
          best_value = rounded;
          best_index = output_col + row;
        }
      }
    }
    if (lane == 0) {
      warp_value[warp] = best_value;
      warp_index[warp] = best_index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      half block_best = warp_value[0];
      int block_index = warp_index[0];
#pragma unroll
      for (int other_warp = 1; other_warp < 8; ++other_warp) {
        if (__hgt(warp_value[other_warp], block_best)) {
          block_best = warp_value[other_warp];
          block_index = warp_index[other_warp];
        }
      }
      block_values[blockIdx.x] = block_best;
      block_indices[blockIdx.x] = block_index;
    }
  }
}

// Grouped-MoE companion to simt_kernel. Each block-row consumes one already
// permuted routed token and selects its expert bank on device. This preserves
// the coalesced N-major/K-streaming path without host-side expert dispatch or
// a second base-backend copy of the expert weights.
template <typename FormatPolicy, int KC>
__global__ void moe_simt_kernel(typename FormatPolicy::Params base_params,
                                const half* __restrict__ input,
                                const int* __restrict__ expert_ids,
                                half* __restrict__ output, int rows,
                                int num_experts, int output_size,
                                int input_size) {
  static_assert(FormatPolicy::kSegmentK == 16,
                "SM70 Skinny MoE SIMT core requires K16 decode segments.");
  extern __shared__ char shared_raw[];
  half2* staged_input = reinterpret_cast<half2*>(shared_raw);

  const int row = blockIdx.y;
  if (row >= rows) {
    return;
  }
  const int expert = expert_ids[row];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_col = static_cast<int>(blockIdx.x) * 8 + warp;
  auto params = base_params;
  FormatPolicy::select_expert(params, expert, output_size, input_size);
  const auto thread_state = FormatPolicy::make_thread_state(params);

  float accumulator = 0.0f;
  int chunk_start = 0;
  for (; chunk_start + KC <= input_size; chunk_start += KC) {
    __syncthreads();
    for (int vector_index = threadIdx.x; vector_index < KC / 8;
         vector_index += blockDim.x) {
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(row) * input_size + chunk_start +
          vector_index * 8);
      FormatPolicy::stage_pairs(staged_input, vector_index * 4, value);
    }
    __syncthreads();

#pragma unroll
    for (int iteration = 0; iteration < KC / 512; ++iteration) {
      const int segment = lane + 32 * iteration;
      typename FormatPolicy::Segment packed;
      FormatPolicy::load_segment(params, thread_state, output_col,
                                 chunk_start + segment * 16, packed);
      half2 half_accumulator = __float2half2_rn(0.0f);
#pragma unroll
      for (int word = 0; word < 2; ++word) {
        half2 weights[4];
        FormatPolicy::decode_word(packed, word, weights);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const int staged_index =
              swizzle_pair_index(segment * 8 + word * 4 + pair);
          half_accumulator = __hfma2(weights[pair], staged_input[staged_index],
                                     half_accumulator);
        }
      }
      const float2 value = __half22float2(half_accumulator);
      accumulator += value.x + value.y;
    }
  }

  const int tail = input_size - chunk_start;
  if (tail > 0) {
    __syncthreads();
    for (int vector_index = threadIdx.x; vector_index < tail / 8;
         vector_index += blockDim.x) {
      const uint4 value = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(row) * input_size + chunk_start +
          vector_index * 8);
      FormatPolicy::stage_pairs(staged_input, vector_index * 4, value);
    }
    __syncthreads();
    const int segment_count = tail / 16;
    for (int segment = lane; segment < segment_count; segment += 32) {
      typename FormatPolicy::Segment packed;
      FormatPolicy::load_segment(params, thread_state, output_col,
                                 chunk_start + segment * 16, packed);
      half2 half_accumulator = __float2half2_rn(0.0f);
#pragma unroll
      for (int word = 0; word < 2; ++word) {
        half2 weights[4];
        FormatPolicy::decode_word(packed, word, weights);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const int staged_index =
              swizzle_pair_index(segment * 8 + word * 4 + pair);
          half_accumulator = __hfma2(weights[pair], staged_input[staged_index],
                                     half_accumulator);
        }
      }
      const float2 value = __half22float2(half_accumulator);
      accumulator += value.x + value.y;
    }
  }

  const float value = warp_sum(accumulator);
  if (lane == 0) {
    output[static_cast<size_t>(row) * output_size + output_col] =
        __float2half(value);
  }
}

}  // namespace vllm::sm70_skinny
