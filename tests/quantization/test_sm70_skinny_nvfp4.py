# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.nvfp4 import skinny
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a16_nvfp4 as ct_w4a16,
)


def _unpack_nibble(codes: torch.Tensor, row: int, column: int) -> int:
    packed = int(codes[row, column // 2])
    return (packed >> (4 * (column & 1))) & 0xF


def test_qpn_prepack_matches_fragment_order_reference():
    n, k = 32, 64
    codes = torch.arange(n * (k // 2), dtype=torch.int64)
    codes = codes.remainder(256).to(torch.uint8).view(n, k // 2)
    scales = torch.arange(n * (k // 16), dtype=torch.int64)
    scales = scales.remainder(256).to(torch.uint8).view(n, k // 16)

    actual_codes, actual_scales = skinny.qpn_prepack(codes, scales)

    assert actual_codes is not None
    assert actual_scales is not None
    expected_codes: list[int] = []
    expected_scales: list[int] = []
    korder = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
    for group in range(k // 16):
        for lane in range(32):
            column = ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0)
            selected = [
                _unpack_nibble(codes, column, group * 16 + offset) for offset in korder
            ]
            expected_codes.extend(
                selected[index] | (selected[index + 1] << 4)
                for index in range(0, 16, 2)
            )
            expected_scales.append(int(scales[column, group]))

    assert actual_codes.tolist() == expected_codes
    assert actual_scales.tolist() == expected_scales


def test_qpn_prepack_rejects_ineligible_shape_without_copy():
    codes = torch.zeros((31, 32), dtype=torch.uint8)
    scales = torch.zeros((31, 4), dtype=torch.uint8)

    qcodes, qscales = skinny.qpn_prepack(codes, scales)

    assert qcodes is None
    assert qscales is None


def _native_buffers(n: int, k: int):
    codes = torch.zeros((n, k // 2), dtype=torch.uint8)
    scales = torch.zeros((n, k // 16), dtype=torch.uint8)
    qcodes = torch.zeros(n * (k // 2), dtype=torch.uint8)
    qscales = torch.zeros(n * (k // 16), dtype=torch.uint8)
    return codes, scales, qcodes, qscales


def test_bf16_activation_is_explicitly_converted_and_restored(monkeypatch):
    n, k = 32, 128
    buffers = _native_buffers(n, k)
    seen_dtypes = []

    def fake_simt(input, codes, scales, global_scale):
        del codes, scales, global_scale
        seen_dtypes.append(input.dtype)
        return input.sum(dim=1, keepdim=True).expand(-1, n).contiguous()

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", fake_simt)
    x = torch.ones((1, k), dtype=torch.bfloat16)

    out = skinny._skinny_nvfp4_linear_impl(x, *buffers, 1.0, n, k)

    assert seen_dtypes == [torch.float16]
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, n)


def test_turbomind_hybrid_op_dispatches_on_runtime_rows(monkeypatch):
    n, k = 32, 128
    buffers = _native_buffers(n, k)
    calls: list[tuple[str, int]] = []

    def simt(input, codes, scales, global_scale):
        del codes, scales, global_scale
        calls.append(("simt", input.shape[0]))
        return input.new_ones((input.shape[0], n))

    def base(out, input, weight, scales, group_size, k_ld, q_ld):
        del weight, scales, group_size, k_ld, q_ld
        calls.append(("base", input.shape[0]))
        out.fill_(2)

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", simt)
    monkeypatch.setattr(_sm70_ops, "nvfp4_gemm_sm70_out", base)
    base_weight = torch.empty((1, 1), dtype=torch.int32)
    base_scales = torch.empty((1, 1), dtype=torch.float16)

    small = skinny._skinny_nvfp4_turbomind_linear_impl(
        torch.ones((1, k), dtype=torch.float16),
        *buffers,
        1.0,
        base_weight,
        base_scales,
        n,
        k,
        16,
        0,
        0,
    )
    large = skinny._skinny_nvfp4_turbomind_linear_impl(
        torch.ones((17, k), dtype=torch.float16),
        *buffers,
        1.0,
        base_weight,
        base_scales,
        n,
        k,
        16,
        0,
        0,
    )

    assert calls == [("simt", 1), ("base", 17)]
    assert torch.equal(small, torch.ones_like(small))
    assert torch.equal(large, torch.full_like(large, 2))


def test_large_m_delegates_to_selected_base_without_dtype_rewrite(monkeypatch):
    n, k, m = 32, 128, 17
    buffers = _native_buffers(n, k)
    calls = []

    class FakeBase:
        def apply_weights(self, layer, input, bias=None):
            del layer, bias
            calls.append(input.dtype)
            return input.sum(dim=1, keepdim=True).expand(-1, n).contiguous()

    monkeypatch.setattr(
        skinny.SkinnyNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    kernel = skinny.SkinnyNvFp4LinearKernel(NvFp4LinearLayerConfig(), FakeBase())
    layer = SimpleNamespace(
        input_size_per_partition=k,
        output_size_per_partition=n,
        skinny_codes=buffers[0],
        skinny_scales=buffers[1],
        skinny_qpn_codes=buffers[2],
        skinny_qpn_scales=buffers[3],
        skinny_global_scale=1.0,
        skinny_disabled_routes=set(),
    )
    x = torch.ones((m, k), dtype=torch.bfloat16)

    out = kernel.apply_weights(layer, x)

    assert calls == [torch.bfloat16]
    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)


def test_skinny_kernel_rejects_non_sm70():
    supported, reason = skinny.SkinnyNvFp4LinearKernel.is_supported(80)

    assert not supported
    assert reason == "requires exact CUDA capability 7.0"


def test_modelopt_nvfp4_min_capability_follows_base_and_overlay(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_QUANT_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_SM70_SKINNY", raising=False)
    monkeypatch.delenv("VLLM_SM70_NVFP4_TURBOMIND", raising=False)
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.setenv("VLLM_SM70_SKINNY", "off")
    # Disabling the overlay does not disable the default TurboMind base.
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "0")
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 75

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 70

    monkeypatch.delenv("VLLM_SM70_SKINNY", raising=False)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    assert modelopt.ModelOptNvFp4Config.get_min_capability() == 70


def test_modelopt_w4a16_selects_generic_skinny_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    monkeypatch.setattr(modelopt.sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(modelopt, "init_nvfp4_linear_kernel", lambda: sentinel)
    config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    method = modelopt.ModelOptNvFp4W4A16LinearMethod(config)

    assert method.kernel is sentinel


def test_modelopt_w4a4_selects_generic_skinny_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    monkeypatch.setattr(modelopt.sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(modelopt, "init_nvfp4_linear_kernel", lambda: sentinel)
    config = modelopt.ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    method = modelopt.ModelOptNvFp4LinearMethod(config)

    assert method.kernel is sentinel


def test_ct_w4a16_preserves_non_sm70_marlin_base(monkeypatch):
    calls: list[str] = []

    class FakeMarlin:
        def process_weights_after_loading(self, layer):
            calls.append("marlin")
            assert layer.weight.dtype == torch.uint8

    monkeypatch.setattr(ct_w4a16.sm70_tm, "is_exact_sm70_cuda_platform", lambda: False)
    monkeypatch.setattr(
        ct_w4a16,
        "init_nvfp4_linear_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected SM70 selector")),
    )
    monkeypatch.setattr(
        ct_w4a16,
        "MarlinNvFp4LinearKernel",
        lambda config: FakeMarlin(),
    )
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros((8, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((8, 8), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32), requires_grad=False
    )

    scheme = ct_w4a16.CompressedTensorsW4A16Fp4()
    scheme.process_weights_after_loading(layer)

    assert calls == ["marlin"]


def test_nvfp4_fp32_reference_matches_elementwise_dequant():
    """Check the ground truth itself against an explicit per-element decode.

    The reference deliberately avoids the kernel's 16384 exponent trick and its
    bit-shuffle FP8 path, so this test pins it to the plain E2M1 x E4M3
    definition rather than to the kernel it is meant to police.
    """
    torch.manual_seed(0)
    n, k = 16, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(40, 80, (n, k // 16), dtype=torch.uint8)
    x = (torch.rand(2, k) - 0.5).to(torch.float16)
    gscale = 0.037

    actual = skinny.nvfp4_fp32_reference(codes, scales, gscale, x)

    magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected = torch.zeros(2, n, dtype=torch.float32)
    for col in range(n):
        for kk in range(k):
            byte = int(codes[col, kk // 2])
            nib = byte & 0xF if kk % 2 == 0 else byte >> 4
            value = magnitude[nib & 0x7]
            if nib & 0x8:
                value = -value
            group_scale = float(scales[col, kk // 16].view(torch.float8_e4m3fn).float())
            weight = value * group_scale * gscale
            for row in range(2):
                expected[row, col] += float(x[row, kk]) * weight

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_nvfp4_fp32_reference_chunking_is_transparent():
    torch.manual_seed(2)
    n, k = 64, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(40, 80, (n, k // 16), dtype=torch.uint8)
    x = (torch.rand(1, k) - 0.5).to(torch.float16)

    whole = skinny.nvfp4_fp32_reference(codes, scales, 0.5, x, chunk=n)
    chunked = skinny.nvfp4_fp32_reference(codes, scales, 0.5, x, chunk=5)
    torch.testing.assert_close(whole, chunked)


def test_nvfp4_release_route_frees_disabled_layout():
    codes, scales, qcodes, qscales = _native_buffers(32, 128)
    layer = SimpleNamespace(
        skinny_codes=torch.nn.Parameter(codes, requires_grad=False),
        skinny_scales=torch.nn.Parameter(scales, requires_grad=False),
        skinny_qpn_codes=torch.nn.Parameter(qcodes, requires_grad=False),
        skinny_qpn_scales=torch.nn.Parameter(qscales, requires_grad=False),
        skinny_disabled_routes=set(),
    )

    skinny._release_route(layer, "simt")

    assert layer.skinny_codes.numel() == 0
    assert layer.skinny_scales.numel() == 0
    assert layer.skinny_qpn_codes.numel() != 0
    assert layer.skinny_disabled_routes == {"simt"}
