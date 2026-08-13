// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
// Adapted for 1Cat-vLLM's format-independent SM70 execution layer.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vllm::sm70_skinny {

#define SM70_SKINNY_INLINE __device__ __forceinline__

SM70_SKINNY_INLINE int swizzle_pair_index(int pair) {
  return (pair & ~7) | ((pair ^ (pair >> 5)) & 7);
}

// Stage eight contiguous FP16 activations in the pairing consumed by the
// register decoders: (k, k+4), (k+1, k+5), ... . Format policies whose
// decoder emits adjacent pairs can select stage_adjacent_pairs instead.
SM70_SKINNY_INLINE void stage_interleaved_pairs(half2* dst, int base_pair,
                                                const uint4& value) {
  const unsigned* words = reinterpret_cast<const unsigned*>(&value);
  const unsigned reordered[4] = {
      __byte_perm(words[0], words[2], 0x5410),
      __byte_perm(words[0], words[2], 0x7632),
      __byte_perm(words[1], words[3], 0x5410),
      __byte_perm(words[1], words[3], 0x7632),
  };
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    dst[swizzle_pair_index(base_pair + index)] =
        *reinterpret_cast<const half2*>(&reordered[index]);
  }
}

SM70_SKINNY_INLINE void stage_adjacent_pairs(half2* dst, int base_pair,
                                             const uint4& value) {
  const half2* pairs = reinterpret_cast<const half2*>(&value);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    dst[swizzle_pair_index(base_pair + index)] = pairs[index];
  }
}

SM70_SKINNY_INLINE float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

SM70_SKINNY_INLINE void mma_m8n8k4(float (&accumulator)[8], unsigned a0,
                                   unsigned a1, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
        "+f"(accumulator[3]), "+f"(accumulator[4]), "+f"(accumulator[5]),
        "+f"(accumulator[6]), "+f"(accumulator[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

}  // namespace vllm::sm70_skinny
