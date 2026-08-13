// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors

#pragma once

#include "../common.cuh"

namespace vllm::sm70_skinny {

SM70_SKINNY_INLINE half2 fp8e4m3_to_half2(unsigned char value) {
  const unsigned short bits = (((unsigned short)value & 0x80u) << 8) |
                              (((unsigned short)value & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}

SM70_SKINNY_INLINE void decode_nvfp4_tm(unsigned packed, half2 scale,
                                        half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned value0 =
      ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  unsigned value1 =
      ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  unsigned value2 =
      ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  unsigned value3 = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
  output[0] = __hmul2(*reinterpret_cast<half2*>(&value0), scale);
  output[1] = __hmul2(*reinterpret_cast<half2*>(&value1), scale);
  output[2] = __hmul2(*reinterpret_cast<half2*>(&value2), scale);
  output[3] = __hmul2(*reinterpret_cast<half2*>(&value3), scale);
}

SM70_SKINNY_INLINE half2 decode_nvfp4_lut(unsigned packed, int pair,
                                          half2 scale) {
  constexpr unsigned kLutLow = 0x3e3c3800u;
  constexpr unsigned kLutHigh = 0x46444240u;
  const unsigned magnitude = (packed & 0x77777777u) >> (8 * pair);
  const unsigned sign = (packed & 0x88888888u) >> (8 * pair);
  const unsigned selector =
      ((magnitude & 0x7u) << 4) | ((magnitude & 0x70u) << 8);
  unsigned halves = __byte_perm(kLutLow, kLutHigh, selector);
  halves |= ((sign & 0x8u) << 12) | ((sign & 0x80u) << 24);
  return __hmul2(*reinterpret_cast<half2*>(&halves), scale);
}

struct Nvfp4SimtPolicy {
  static constexpr int kSegmentK = 16;

  struct Params {
    const uint8_t* codes;
    const uint8_t* scales;
    int input_size;
    float global_scale;
  };

  struct ThreadState {
    half2 global_scale;
  };

  struct Segment {
    uint2 codes;
    half2 scale;
  };

  SM70_SKINNY_INLINE static ThreadState make_thread_state(
      const Params& params) {
#ifdef SKINNY_LUT_CVT
    return {__float2half2_rn(params.global_scale)};
#else
    return {__float2half2_rn(params.global_scale * 16384.0f)};
#endif
  }

  SM70_SKINNY_INLINE static void stage_pairs(half2* destination, int base_pair,
                                             const uint4& value) {
#ifdef SKINNY_LUT_CVT
    stage_adjacent_pairs(destination, base_pair, value);
#else
    stage_interleaved_pairs(destination, base_pair, value);
#endif
  }

  SM70_SKINNY_INLINE static void load_segment(const Params& params,
                                              const ThreadState& state,
                                              int output_col, int absolute_k,
                                              Segment& segment) {
    const uint8_t* code_row = params.codes + static_cast<size_t>(output_col) *
                                                 (params.input_size / 2);
    const uint8_t* scale_row =
        params.scales +
        static_cast<size_t>(output_col) * (params.input_size / kSegmentK);
    segment.codes = *reinterpret_cast<const uint2*>(code_row + absolute_k / 2);
    segment.scale = __hmul2(fp8e4m3_to_half2(scale_row[absolute_k / kSegmentK]),
                            state.global_scale);
  }

  SM70_SKINNY_INLINE static void decode_word(const Segment& segment, int word,
                                             half2 output[4]) {
    const unsigned packed = word == 0 ? segment.codes.x : segment.codes.y;
#ifdef SKINNY_LUT_CVT
  #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      output[pair] = decode_nvfp4_lut(packed, pair, segment.scale);
    }
#else
    decode_nvfp4_tm(packed, segment.scale, output);
#endif
  }
};

struct Nvfp4QpnPolicy {
  struct Params {
    const uint8_t* codes;
    const uint8_t* scales;
    int group_count;
    float global_scale;
  };

  struct ThreadState {
    half2 global_scale;
  };

  SM70_SKINNY_INLINE static ThreadState make_thread_state(
      const Params& params) {
    return {__float2half2_rn(params.global_scale * 16384.0f)};
  }

  SM70_SKINNY_INLINE static void load_fragment(const Params& params,
                                               const ThreadState& state,
                                               int tile, int group, int lane,
                                               half2 output[8]) {
    const uint2* codes = reinterpret_cast<const uint2*>(params.codes) +
                         static_cast<size_t>(tile) * params.group_count * 32 +
                         lane;
    const uint8_t* scales =
        params.scales + static_cast<size_t>(tile) * params.group_count * 32 +
        lane;
    const uint2 packed = __ldcs(codes + static_cast<size_t>(group) * 32);
    const half2 scale = __hmul2(
        fp8e4m3_to_half2(__ldg(scales + static_cast<size_t>(group) * 32)),
        state.global_scale);
    decode_nvfp4_tm(packed.x, scale, output);
    decode_nvfp4_tm(packed.y, scale, output + 4);
  }
};

}  // namespace vllm::sm70_skinny
