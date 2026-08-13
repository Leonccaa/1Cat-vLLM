// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <torch/library.h>

#include "formats/awq.cuh"
#include "qpn.cuh"
#include "simt.cuh"

using vllm::sm70_skinny::AwqG128QpnPolicy;
using vllm::sm70_skinny::AwqG128SimtPolicy;

namespace {

void check_simt_inputs(const torch::Tensor& input, const torch::Tensor& codes,
                       const torch::Tensor& scales, const torch::Tensor& biases,
                       int64_t group_size, int64_t& rows, int64_t& output_size,
                       int64_t& input_size) {
  TORCH_CHECK(
      input.is_cuda() && input.dtype() == torch::kHalf && input.is_contiguous(),
      "SM70 Skinny AWQ expects contiguous FP16 input.");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
                  codes.is_contiguous() && codes.dim() == 2,
              "SM70 Skinny AWQ expects uint8 [N,K/2] codes.");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf &&
                  scales.is_contiguous() && scales.dim() == 2,
              "SM70 Skinny AWQ expects FP16 [N,K/128] scales.");
  TORCH_CHECK(biases.is_cuda() && biases.dtype() == torch::kHalf &&
                  biases.is_contiguous() && biases.sizes() == scales.sizes(),
              "SM70 Skinny AWQ scale/bias shape mismatch.");
  TORCH_CHECK(group_size == 128,
              "SM70 Skinny AWQ currently supports group_size=128 only.");
  rows = input.size(0);
  input_size = input.size(1);
  output_size = codes.size(0);
  TORCH_CHECK(codes.size(1) * 2 == input_size,
              "SM70 Skinny AWQ codes/input K mismatch.");
  TORCH_CHECK(scales.size(0) == output_size &&
                  scales.size(1) * group_size == input_size,
              "SM70 Skinny AWQ metadata shape mismatch.");
}

torch::Tensor skinny_awq_gemm_simt(torch::Tensor input, torch::Tensor codes,
                                   torch::Tensor scales, torch::Tensor biases,
                                   int64_t group_size) {
  int64_t rows, output_size, input_size;
  check_simt_inputs(input, codes, scales, biases, group_size, rows, output_size,
                    input_size);
  TORCH_CHECK(rows >= 1 && rows <= 3,
              "SM70 Skinny AWQ SIMT supports M=1..3, got ", rows);
  TORCH_CHECK(input_size % 128 == 0 && output_size % 8 == 0,
              "SM70 Skinny AWQ SIMT alignment mismatch.");

  constexpr int kChunkK = 1024;
  auto output = torch::empty({rows, output_size}, input.options());
  const bool two_rows_per_warp = input_size <= 2048 && output_size % 16 == 0;
  const dim3 grid(two_rows_per_warp ? output_size / 16 : output_size / 8);
  const dim3 block(256);
  const int shared_bytes = rows * (kChunkK / 2) * sizeof(half2);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const AwqG128SimtPolicy::Params params = {
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(biases.data_ptr<at::Half>()),
      static_cast<int>(input_size),
  };

#define LAUNCH_AWQ_SIMT(MM)                                                    \
  if (two_rows_per_warp)                                                       \
    vllm::sm70_skinny::simt_kernel<AwqG128SimtPolicy, MM, kChunkK, 2>          \
        <<<grid, block, shared_bytes, stream>>>(                               \
            params, reinterpret_cast<const half*>(input.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),              \
            static_cast<int>(output_size), static_cast<int>(input_size),       \
            nullptr, nullptr);                                                 \
  else                                                                         \
    vllm::sm70_skinny::simt_kernel<AwqG128SimtPolicy, MM, kChunkK, 1>          \
        <<<grid, block, shared_bytes, stream>>>(                               \
            params, reinterpret_cast<const half*>(input.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),              \
            static_cast<int>(output_size), static_cast<int>(input_size),       \
            nullptr, nullptr)

  switch (rows) {
    case 1:
      LAUNCH_AWQ_SIMT(1);
      break;
    case 2:
      LAUNCH_AWQ_SIMT(2);
      break;
    case 3:
      LAUNCH_AWQ_SIMT(3);
      break;
    default:
      TORCH_CHECK(false, "unreachable AWQ SIMT M");
  }
#undef LAUNCH_AWQ_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor skinny_awq_gemm_qpn(torch::Tensor input, torch::Tensor codes,
                                  torch::Tensor scales, torch::Tensor biases,
                                  int64_t group_size, int64_t output_size) {
  TORCH_CHECK(
      input.is_cuda() && input.dtype() == torch::kHalf && input.is_contiguous(),
      "SM70 Skinny AWQ QPN expects contiguous FP16 input.");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
                  codes.is_contiguous(),
              "SM70 Skinny AWQ QPN expects contiguous uint8 codes.");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf &&
                  scales.is_contiguous() && biases.is_cuda() &&
                  biases.dtype() == torch::kHalf && biases.is_contiguous(),
              "SM70 Skinny AWQ QPN metadata dtype mismatch.");
  TORCH_CHECK(group_size == 128,
              "SM70 Skinny AWQ QPN currently supports group_size=128 only.");
  const int64_t rows = input.size(0);
  const int64_t input_size = input.size(1);
  TORCH_CHECK(rows >= 4 && rows <= 16,
              "SM70 Skinny AWQ QPN supports M=4..16, got ", rows);
  TORCH_CHECK(input_size % 128 == 0 && output_size % 32 == 0,
              "SM70 Skinny AWQ QPN alignment mismatch.");
  TORCH_CHECK(codes.numel() == output_size * input_size / 2,
              "SM70 Skinny AWQ QPN code size mismatch.");
  TORCH_CHECK(scales.numel() == output_size * input_size / group_size &&
                  biases.numel() == scales.numel(),
              "SM70 Skinny AWQ QPN metadata size mismatch.");

  auto output = torch::empty({rows, output_size}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const AwqG128QpnPolicy::Params params = {
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(biases.data_ptr<at::Half>()),
      static_cast<int>(input_size / 16),
  };
  if (rows <= 8) {
    vllm::sm70_skinny::qpn_kernel<AwqG128QpnPolicy, 1>
        <<<dim3(static_cast<int>(output_size / 32)), dim3(128), 0, stream>>>(
            params, reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            static_cast<int>(output_size), static_cast<int>(input_size),
            static_cast<int>(rows));
  } else {
    vllm::sm70_skinny::qpn_kernel<AwqG128QpnPolicy, 2>
        <<<dim3(static_cast<int>(output_size / 32)), dim3(128), 0, stream>>>(
            params, reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            static_cast<int>(output_size), static_cast<int>(input_size),
            static_cast<int>(rows));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void skinny_awq_moe_gemm_simt_out(torch::Tensor output, torch::Tensor input,
                                  torch::Tensor expert_ids, torch::Tensor codes,
                                  torch::Tensor scales, torch::Tensor biases,
                                  int64_t group_size) {
  TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kHalf &&
                  input.is_contiguous() && input.dim() == 2,
              "SM70 Skinny AWQ MoE expects contiguous FP16 [rows,K] input.");
  TORCH_CHECK(output.is_cuda() && output.dtype() == torch::kHalf &&
                  output.is_contiguous() && output.dim() == 2,
              "SM70 Skinny AWQ MoE expects contiguous FP16 [rows,N] output.");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.dtype() == torch::kInt32 &&
                  expert_ids.is_contiguous() && expert_ids.dim() == 1,
              "SM70 Skinny AWQ MoE expects contiguous int32 expert ids.");
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
                  codes.is_contiguous() && codes.dim() == 3,
              "SM70 Skinny AWQ MoE expects uint8 [E,N,K/2] codes.");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kHalf &&
                  scales.is_contiguous() && scales.dim() == 3 &&
                  biases.is_cuda() && biases.dtype() == torch::kHalf &&
                  biases.is_contiguous() && biases.sizes() == scales.sizes(),
              "SM70 Skinny AWQ MoE metadata mismatch.");
  TORCH_CHECK(group_size == 128,
              "SM70 Skinny AWQ MoE currently supports group_size=128 only.");

  const int64_t rows = input.size(0);
  const int64_t input_size = input.size(1);
  const int64_t num_experts = codes.size(0);
  const int64_t output_size = codes.size(1);
  TORCH_CHECK(rows == output.size(0) && rows == expert_ids.numel(),
              "SM70 Skinny AWQ MoE row count mismatch.");
  TORCH_CHECK(output.size(1) == output_size && codes.size(2) * 2 == input_size,
              "SM70 Skinny AWQ MoE weight shape mismatch.");
  TORCH_CHECK(scales.size(0) == num_experts && scales.size(1) == output_size &&
                  scales.size(2) == input_size / group_size,
              "SM70 Skinny AWQ MoE metadata shape mismatch.");
  TORCH_CHECK(input_size % 128 == 0 && output_size % 8 == 0,
              "SM70 Skinny AWQ MoE alignment mismatch.");
  TORCH_CHECK(rows <= 65535,
              "SM70 Skinny AWQ MoE supports at most 65535 routed rows.");
  if (rows == 0) {
    return;
  }

  constexpr int kChunkK = 1024;
  const AwqG128SimtPolicy::Params params = {
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(biases.data_ptr<at::Half>()),
      static_cast<int>(input_size),
  };
  vllm::sm70_skinny::moe_simt_kernel<AwqG128SimtPolicy, kChunkK>
      <<<dim3(static_cast<int>(output_size / 8), static_cast<int>(rows)),
         dim3(256), (kChunkK / 2) * sizeof(half2),
         at::cuda::getCurrentCUDAStream()>>>(
          params, reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          static_cast<int>(rows), static_cast<int>(num_experts),
          static_cast<int>(output_size), static_cast<int>(input_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "skinny_awq_gemm_simt(Tensor input, Tensor codes, Tensor scales, "
      "Tensor biases, int group_size) -> Tensor");
  ops.impl("skinny_awq_gemm_simt", torch::kCUDA, &skinny_awq_gemm_simt);
  ops.def(
      "skinny_awq_gemm_qpn(Tensor input, Tensor codes, Tensor scales, "
      "Tensor biases, int group_size, int output_size) -> Tensor");
  ops.impl("skinny_awq_gemm_qpn", torch::kCUDA, &skinny_awq_gemm_qpn);
  ops.def(
      "skinny_awq_moe_gemm_simt_out(Tensor(a!) output, Tensor input, "
      "Tensor expert_ids, Tensor codes, Tensor scales, Tensor biases, "
      "int group_size) -> ()");
  ops.impl("skinny_awq_moe_gemm_simt_out", torch::kCUDA,
           &skinny_awq_moe_gemm_simt_out);
}
