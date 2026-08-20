// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
// Block-128 FP8 recurrence and 1Cat integration: 2026 Leonccaa contributors.

// Volta small-M GEMM for compressed-tensors E4M3 weights with 128x128
// block scales. The weight bytes are prepacked into QPN fragment order at
// model load. One normalized FP32 accumulator bank carries exact block-scale
// semantics across K:
//
//   C_0 = P_0
//   C_j = C_(j-1) * (s_(j-1) / s_j) + P_j
//   Y   = C_last * 256 * s_last

// E4M3 is decoded as value / 256. The factor cancels from every ratio and is
// folded into the final scale. Split-K warps restore their absolute scale
// before the shared-memory reduction.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "core/registration.h"

namespace {

#define DEV_INLINE __device__ __forceinline__

#define MMA_8N8K4(C, A0, A1, B0, B1)                                \
  asm volatile(                                                       \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]),           \
        "+f"(C[4]), "+f"(C[5]), "+f"(C[6]), "+f"(C[7])            \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Decode eight E4M3 bytes into four half2 values. QPN prepack arranges the
// bytes so each output holds the adjacent-K pair consumed by m8n8k4. The
// generated fp16 values are exactly E4M3 / 256 for all finite weight codes.
DEV_INLINE void fp8x8_to_half2x4(const uint2 q, half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x3F803F80u;
  unsigned p[4];
  p[0] = __byte_perm(q.x, q.y, 0x0400);
  p[1] = __byte_perm(q.x, q.y, 0x0501);
  p[2] = __byte_perm(q.x, q.y, 0x0602);
  p[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned v = ((p[i] << 8) & kSign) |
                       ((p[i] << 7) & kExponentMantissa);
    out[i] = *reinterpret_cast<const half2*>(&v);
  }
}

template <int SplitK, int NAcc>
__global__ void skinny_fp8_qpn8_b128_kernel(
    const uint8_t* __restrict__ codes,
    const float* __restrict__ scales256,
    const float* __restrict__ ratios,
    const half* __restrict__ x,
    half* __restrict__ y,
    int n,
    int k,
    int m) {
  __shared__ float partials[SplitK > 1 ? SplitK : 1]
                           [SplitK > 1 ? 256 : 1];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups = k >> 4;
  const int groups_per_warp = groups / SplitK;
  const int group_start = warp * groups_per_warp;
  const int k_blocks = k >> 7;
  const int scale_row = tile >> 2;
  int current_k_block = group_start >> 3;
  const uint4* code_base = reinterpret_cast<const uint4*>(codes) +
                           static_cast<size_t>(tile) * groups * 32 + lane;

  float accum[NAcc][8];
#pragma unroll
  for (int a = 0; a < NAcc; ++a) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[a][i] = 0.0f;
    }
  }

#pragma unroll 4
  for (int group = group_start;
       group < group_start + groups_per_warp;
       ++group) {
    const int k_block = group >> 3;
    if (k_block != current_k_block) {
      const float ratio = __ldg(
          ratios + static_cast<size_t>(scale_row) * k_blocks + k_block);
#pragma unroll
      for (int a = 0; a < NAcc; ++a) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          accum[a][i] *= ratio;
        }
      }
      current_k_block = k_block;
    }

    const uint4 q = __ldcs(code_base + static_cast<size_t>(group) * 32);
    half2 b[8];
    fp8x8_to_half2x4(make_uint2(q.x, q.y), b);
    fp8x8_to_half2x4(make_uint2(q.z, q.w), b + 4);
    const unsigned* b_regs = reinterpret_cast<const unsigned*>(b);

    uint4 a_lo = make_uint4(0, 0, 0, 0);
    uint4 a_hi = make_uint4(0, 0, 0, 0);
    if (row < m) {
      const half* x_row = x + static_cast<size_t>(row) * k;
      a_lo = *reinterpret_cast<const uint4*>(x_row + group * 16);
      a_hi = *reinterpret_cast<const uint4*>(x_row + group * 16 + 8);
    }
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&a_lo);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&a_hi);
    MMA_8N8K4(accum[0], a0[0], a0[1], b_regs[0], b_regs[1]);
    MMA_8N8K4(accum[1 % NAcc], a0[2], a0[3], b_regs[2], b_regs[3]);
    MMA_8N8K4(accum[2 % NAcc], a1[0], a1[1], b_regs[4], b_regs[5]);
    MMA_8N8K4(accum[3 % NAcc], a1[2], a1[3], b_regs[6], b_regs[7]);
  }

#pragma unroll
  for (int a = 1; a < NAcc; ++a) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[0][i] += accum[a][i];
    }
  }

  const float final_scale = __ldg(
      scales256 + static_cast<size_t>(scale_row) * k_blocks +
      current_k_block);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int out_row =
        (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int out_col =
        (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partials[warp][out_row * 32 + quadpair * 8 + out_col] =
        accum[0][i] * final_scale;
  }
  __syncthreads();

  for (int element = threadIdx.x; element < 256; element += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int split = 0; split < SplitK; ++split) {
      value += partials[split][element];
    }
    const int out_row = element >> 5;
    const int out_col = element & 31;
    if (out_row < m) {
      y[static_cast<size_t>(out_row) * n +
        static_cast<size_t>(tile) * 32 + out_col] = __float2half(value);
    }
  }
}

template <int SplitK, int NAcc>
void launch_skinny_fp8_qpn8_b128(
    const torch::Tensor& x,
    const torch::Tensor& codes,
    const torch::Tensor& scales256,
    const torch::Tensor& ratios,
    torch::Tensor& output,
    int64_t n,
    int64_t k,
    int64_t m) {
  auto stream = at::cuda::getCurrentCUDAStream();
  skinny_fp8_qpn8_b128_kernel<SplitK, NAcc>
      <<<dim3(static_cast<unsigned>(n / 32)),
         dim3(static_cast<unsigned>(32 * SplitK)), 0, stream>>>(
          codes.data_ptr<uint8_t>(), scales256.data_ptr<float>(),
          ratios.data_ptr<float>(),
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(n), static_cast<int>(k), static_cast<int>(m));
}

torch::Tensor sm70_fp8_qpn8_b128_gemm(
    const torch::Tensor& x,
    const torch::Tensor& codes,
    const torch::Tensor& scales256,
    const torch::Tensor& ratios,
    int64_t n,
    int64_t split_k,
    int64_t nacc) {
  c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() && x.dim() == 2,
              "SM70 FP8 QPN8 requires contiguous FP16 [M,K] activations");
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == at::kByte &&
                  codes.is_contiguous(),
              "SM70 FP8 QPN8 requires contiguous uint8 packed weights");
  TORCH_CHECK(scales256.is_cuda() && scales256.scalar_type() == at::kFloat &&
                  scales256.is_contiguous(),
              "SM70 FP8 QPN8 requires contiguous FP32 scales");
  TORCH_CHECK(ratios.is_cuda() && ratios.scalar_type() == at::kFloat &&
                  ratios.is_contiguous(),
              "SM70 FP8 QPN8 requires contiguous FP32 ratios");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(m >= 1 && m <= 8, "SM70 FP8 QPN8 supports M=1..8, got ", m);
  TORCH_CHECK(n > 0 && n % 128 == 0, "SM70 FP8 QPN8 requires N % 128 == 0");
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "SM70 FP8 QPN8 requires K % 128 == 0");
  TORCH_CHECK((k / 16) % split_k == 0,
              "SM70 FP8 QPN8 requires K/16 divisible by split-K");
  TORCH_CHECK(codes.numel() == n * k, "SM70 FP8 QPN8 weight size mismatch");
  const int64_t metadata_size = (n / 128) * (k / 128);
  TORCH_CHECK(scales256.numel() == metadata_size &&
                  ratios.numel() == metadata_size,
              "SM70 FP8 QPN8 metadata size mismatch");

  auto output = torch::empty({m, n}, x.options());
  const int key = static_cast<int>(split_k * 10 + nacc);
  switch (key) {
    case 41:
      launch_skinny_fp8_qpn8_b128<4, 1>(x, codes, scales256, ratios,
                                         output, n, k, m);
      break;
    case 42:
      launch_skinny_fp8_qpn8_b128<4, 2>(x, codes, scales256, ratios,
                                         output, n, k, m);
      break;
    case 81:
      launch_skinny_fp8_qpn8_b128<8, 1>(x, codes, scales256, ratios,
                                         output, n, k, m);
      break;
    case 82:
      launch_skinny_fp8_qpn8_b128<8, 2>(x, codes, scales256, ratios,
                                         output, n, k, m);
      break;
    case 161:
      launch_skinny_fp8_qpn8_b128<16, 1>(x, codes, scales256, ratios,
                                          output, n, k, m);
      break;
    case 162:
      launch_skinny_fp8_qpn8_b128<16, 2>(x, codes, scales256, ratios,
                                          output, n, k, m);
      break;
    case 321:
      launch_skinny_fp8_qpn8_b128<32, 1>(x, codes, scales256, ratios,
                                          output, n, k, m);
      break;
    case 322:
      launch_skinny_fp8_qpn8_b128<32, 2>(x, codes, scales256, ratios,
                                          output, n, k, m);
      break;
    default:
      TORCH_CHECK(false,
                  "SM70 FP8 QPN8 split-K must be 4, 8, 16, or 32 and "
                  "NACC must be 1 or 2");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sm70_fp8_qpn8_b128_gemm", &sm70_fp8_qpn8_b128_gemm);
}
