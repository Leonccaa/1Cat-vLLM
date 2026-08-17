// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors

#pragma once

#include "../common.cuh"

namespace vllm::sm70_skinny {

// E4M3 -> FP16 by field re-alignment plus a 2^8 exponent-bias correction
// (E4M3 bias 7, FP16 bias 15). Subnormal E4M3 inputs land on FP16 subnormals
// and come out exact.
//
// Known limitation: this is a pure bit shuffle with no special-case handling,
// so the E4M3 NaN encodings 0x7F/0xFF decode to the finite value +-480 instead
// of propagating NaN. A corrupt scale byte therefore yields wrong numbers
// rather than an obvious NaN. Detecting that belongs in the load-time
// self-check, not in this inner-loop helper.
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

  // decode_nvfp4_tm leaves the E2M1 magnitude in FP16 bits [11:9], i.e. the
  // true value scaled by 2^-14. The compensating 16384 is folded into the
  // thread-resident global scale so the inner loop stays two ALU ops per pair.
  //
  // Range contract: the compensated group scale must stay inside FP16, so
  // `fp8_scale * global_scale < 65504 / 16384 ~= 4.0`. Since the largest E2M1
  // code is 6, that bounds representable weights at |w| < 24, far above any
  // real checkpoint. skinny_nvfp4_gemm_* checks this at launch.
  SM70_SKINNY_INLINE static ThreadState make_thread_state(
      const Params& params) {
    return {__float2half2_rn(params.global_scale * 16384.0f)};
  }

  SM70_SKINNY_INLINE static void stage_pairs(half2* destination, int base_pair,
                                             const uint4& value) {
    stage_interleaved_pairs(destination, base_pair, value);
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
    // Plain loads: see AwqG128SimtPolicy::load_segment for the end-to-end
    // measurement that rejected __ldcs here.
    segment.codes = *reinterpret_cast<const uint2*>(code_row + absolute_k / 2);
    segment.scale = __hmul2(fp8e4m3_to_half2(scale_row[absolute_k / kSegmentK]),
                            state.global_scale);
  }

  SM70_SKINNY_INLINE static void decode_word(const Segment& segment, int word,
                                             half2 output[4]) {
    const unsigned packed = word == 0 ? segment.codes.x : segment.codes.y;
    decode_nvfp4_tm(packed, segment.scale, output);
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
