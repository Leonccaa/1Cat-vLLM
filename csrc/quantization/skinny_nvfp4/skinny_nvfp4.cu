// SPDX-License-Identifier: MIT
// Copyright (c) 2026 v100-skinny contributors
//
// SM70 skinny NVFP4 dequant-GEMM: y[M,N] = x[M,K] @ W[K,N].
//
// This is the production small-M subset adapted from v100-skinny commit
// f8194f7c3c9269fa74ee70b5029d53c20098f4c8. 1Cat dispatches FP16 M<=3
// to SIMT and M=4..16 to QPN; the Python adapter explicitly converts BF16
// activations to FP16, while TurboMind remains the fallback for unsupported
// shapes and larger M.
//
// Packed format (0.5625 bytes/weight):
//   codes  uint8 [N][K/2]   two E2M1 codes per byte, low nibble = even k
//   scales uint8 [N][K/16]  FP8-E4M3 per 16-k group
//   gscale float            global scale, applied in the kernel

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../sm70_skinny/formats/nvfp4.cuh"
#include "../sm70_skinny/qpn.cuh"
#include "../sm70_skinny/simt.cuh"

using vllm::sm70_skinny::Nvfp4QpnPolicy;
using vllm::sm70_skinny::Nvfp4SimtPolicy;

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
static void check_inputs(const torch::Tensor& x, const torch::Tensor& codes,
                         const torch::Tensor& scales, int64_t& m, int64_t& n,
                         int64_t& k) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 &&
              scales.is_contiguous());
  m = x.size(0);
  k = x.size(1);
  n = codes.size(0);
  TORCH_CHECK(codes.size(1) * 2 == k, "codes/x K mismatch");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) * 16 == k);
}
torch::Tensor skinny_gemm_simt(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 1024;
  TORCH_CHECK(k % 128 == 0 && k >= 128, "K must be a multiple of 128");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  auto y = torch::empty({m, n}, x.options());
  // Short-K rows leave <2 weight loads in flight per thread; two rows
  // per warp restores latency hiding (shape diagnostic: out_proj K=1536
  // ran at 66% of flagship bandwidth with one row per warp).
  const bool two_rows = (k <= 2048) && (n % 16 == 0);
  const dim3 grid(two_rows ? n / 16 : n / 8), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)m * (KC / 2) * sizeof(half2);
  const Nvfp4SimtPolicy::Params params = {codes.data_ptr<uint8_t>(),
                                          scales.data_ptr<uint8_t>(), (int)k,
                                          (float)gscale};

#define LAUNCH_SIMT(MM)                                                      \
  if (two_rows)                                                              \
    vllm::sm70_skinny::simt_kernel<Nvfp4SimtPolicy, MM, KC, 2>               \
        <<<grid, block, smem, stream>>>(                                     \
            params, reinterpret_cast<const half*>(x.data_ptr<at::Half>()),   \
            reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k, \
            nullptr, nullptr);                                               \
  else                                                                       \
    vllm::sm70_skinny::simt_kernel<Nvfp4SimtPolicy, MM, KC, 1>               \
        <<<grid, block, smem, stream>>>(                                     \
            params, reinterpret_cast<const half*>(x.data_ptr<at::Half>()),   \
            reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k, \
            nullptr, nullptr)

  switch (m) {
    case 1:
      LAUNCH_SIMT(1);
      break;
    case 2:
      LAUNCH_SIMT(2);
      break;
    case 3:
      LAUNCH_SIMT(3);
      break;
    default:
      TORCH_CHECK(false, "simt kernel supports M in 1..3, got ", m);
  }
#undef LAUNCH_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
// The QPN dataflow now lives in sm70_skinny/qpn.cuh. NVFP4 contributes only
// its fragment loader and register decoder through Nvfp4QpnPolicy.
torch::Tensor skinny_gemm_qpn(torch::Tensor x, torch::Tensor qcodes,
                              torch::Tensor qscales, double gscale, int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(m >= 4 && m <= 16, "qpn supports M 4..16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64 (4-warp split of 16-k groups)");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1), "qpn codes size");
  TORCH_CHECK(qscales.numel() == n * (k >> 4), "qpn scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const Nvfp4QpnPolicy::Params params = {qcodes.data_ptr<uint8_t>(),
                                         qscales.data_ptr<uint8_t>(),
                                         (int)(k / 16), (float)gscale};
  if (m <= 8)
    vllm::sm70_skinny::qpn_kernel<Nvfp4QpnPolicy, 1>
        <<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
            params, reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k,
            (int)m);
  else
    vllm::sm70_skinny::qpn_kernel<Nvfp4QpnPolicy, 2>
        <<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
            params, reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k,
            (int)m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "skinny_nvfp4_gemm_simt(Tensor x, Tensor codes, Tensor scales, "
      "float gscale) -> Tensor");
  ops.impl("skinny_nvfp4_gemm_simt", torch::kCUDA, &skinny_gemm_simt);
  ops.def(
      "skinny_nvfp4_gemm_qpn(Tensor x, Tensor qcodes, Tensor qscales, "
      "float gscale, int n) -> Tensor");
  ops.impl("skinny_nvfp4_gemm_qpn", torch::kCUDA, &skinny_gemm_qpn);
}
