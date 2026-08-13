// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "../common.cuh"

namespace vllm::sm70_skinny {

// Convert eight logical uint4 values to four half2 pairs. For a word whose
// nibbles are in logical K order, the result is (k0,k4), (k1,k5), ... and
// therefore matches stage_interleaved_pairs. QPN prepack instead permutes the
// input nibbles so the same primitive emits adjacent B-fragment pairs.
SM70_SKINNY_INLINE uint4 convert_awq_u4x8(unsigned source) {
  uint4 result;
  auto* halves = reinterpret_cast<unsigned*>(&result);
  constexpr unsigned kBottomMask = 0x000f000fu;
  constexpr unsigned kTopMask = 0x00f000f0u;
  constexpr unsigned kMagic = 0x64006400u;
  constexpr unsigned kImmediate = (0xf0 & 0xcc) | 0xaa;
  const unsigned top = source >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(halves[0])
               : "r"(source), "n"(kBottomMask), "n"(kMagic), "n"(kImmediate));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(halves[1])
               : "r"(source), "n"(kTopMask), "n"(kMagic), "n"(kImmediate));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(halves[2])
               : "r"(top), "n"(kBottomMask), "n"(kMagic), "n"(kImmediate));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(halves[3])
               : "r"(top), "n"(kTopMask), "n"(kMagic), "n"(kImmediate));

  constexpr unsigned kOneSixteenth = 0x2c002c00u;
  constexpr unsigned kNegative64 = 0xd400d400u;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(halves[0])
               : "r"(halves[0]), "r"(kMagic));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(halves[1])
               : "r"(halves[1]), "r"(kOneSixteenth), "r"(kNegative64));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(halves[2])
               : "r"(halves[2]), "r"(kMagic));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(halves[3])
               : "r"(halves[3]), "r"(kOneSixteenth), "r"(kNegative64));
  return result;
}

SM70_SKINNY_INLINE void dequant_awq_word(unsigned packed, half2 scale,
                                         half2 bias, half2 output[4]) {
  const uint4 converted = convert_awq_u4x8(packed);
  const half2* values = reinterpret_cast<const half2*>(&converted);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    output[pair] = __hfma2(values[pair], scale, bias);
  }
}

struct AwqG128SimtPolicy {
  static constexpr int kSegmentK = 16;
  static constexpr int kGroupSize = 128;

  struct Params {
    const uint8_t* codes;
    const half* scales;
    const half* biases;
    int input_size;
  };

  struct ThreadState {};

  struct Segment {
    uint2 codes;
    half2 scale;
    half2 bias;
  };

  SM70_SKINNY_INLINE static ThreadState make_thread_state(const Params&) {
    return {};
  }

  SM70_SKINNY_INLINE static void select_expert(Params& params, int expert,
                                               int output_size,
                                               int input_size) {
    params.codes +=
        static_cast<size_t>(expert) * output_size * (input_size / 2);
    const size_t metadata_stride =
        static_cast<size_t>(output_size) * (input_size / kGroupSize);
    params.scales += static_cast<size_t>(expert) * metadata_stride;
    params.biases += static_cast<size_t>(expert) * metadata_stride;
  }

  SM70_SKINNY_INLINE static void stage_pairs(half2* destination, int base_pair,
                                             const uint4& value) {
    stage_interleaved_pairs(destination, base_pair, value);
  }

  SM70_SKINNY_INLINE static void load_segment(const Params& params,
                                              const ThreadState&,
                                              int output_col, int absolute_k,
                                              Segment& segment) {
    const uint8_t* code_row = params.codes + static_cast<size_t>(output_col) *
                                                 (params.input_size / 2);
    const int group_count = params.input_size / kGroupSize;
    const int group = absolute_k / kGroupSize;
    const size_t metadata_index =
        static_cast<size_t>(output_col) * group_count + group;
    // Deliberately plain loads. Marking the weight stream __ldcs (evict-first)
    // looked like a 6-21% win in the standalone harness, but cost 1.3% of
    // end-to-end decode on Qwen3.6-27B AWQ TP4 (68.73 -> 67.84 tok/s, same
    // build otherwise). The harness rotates weight buffers past L2 and still
    // failed to predict this: in the real model these GEMMs interleave with
    // attention and norms, so the cache state they see is nothing like a
    // back-to-back GEMM loop. Do not reintroduce without an end-to-end A/B.
    segment.codes = *reinterpret_cast<const uint2*>(code_row + absolute_k / 2);
    segment.scale = __half2half2(params.scales[metadata_index]);
    segment.bias = __half2half2(params.biases[metadata_index]);
  }

  SM70_SKINNY_INLINE static void decode_word(const Segment& segment, int word,
                                             half2 output[4]) {
    const unsigned packed = word == 0 ? segment.codes.x : segment.codes.y;
    dequant_awq_word(packed, segment.scale, segment.bias, output);
  }
};

struct AwqG128QpnPolicy {
  static constexpr int kGroupSize = 128;

  struct Params {
    const uint8_t* codes;
    const half* scales;
    const half* biases;
    int k16_group_count;
  };

  struct ThreadState {};

  SM70_SKINNY_INLINE static ThreadState make_thread_state(const Params&) {
    return {};
  }

  SM70_SKINNY_INLINE static void load_fragment(const Params& params,
                                               const ThreadState&, int tile,
                                               int group, int lane,
                                               half2 output[8]) {
    const uint2* codes =
        reinterpret_cast<const uint2*>(params.codes) +
        static_cast<size_t>(tile) * params.k16_group_count * 32 + lane;
    const int metadata_group_count = params.k16_group_count / 8;
    const size_t metadata_index =
        (static_cast<size_t>(tile) * metadata_group_count + group / 8) * 32 +
        lane;
    const uint2 packed = __ldcs(codes + static_cast<size_t>(group) * 32);
    const half2 scale = __half2half2(__ldg(params.scales + metadata_index));
    const half2 bias = __half2half2(__ldg(params.biases + metadata_index));
    dequant_awq_word(packed.x, scale, bias, output);
    dequant_awq_word(packed.y, scale, bias, output + 4);
  }
};

}  // namespace vllm::sm70_skinny
