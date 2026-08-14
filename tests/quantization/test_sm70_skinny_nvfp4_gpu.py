# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.nvfp4 import (
    NvFp4LinearLayerConfig,
    SkinnyNvFp4LinearKernel,
    TurboMindNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4 import skinny as nvfp4_skinny
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a16_nvfp4 as ct_w4a16,
)


def _is_exact_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_sm70_skinny_adapter_routes_and_matches_turbomind(monkeypatch):
    supported, reason = SkinnyNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    class FakeLayer(torch.nn.Module):
        pass

    torch.manual_seed(20260812)
    monkeypatch.setenv("VLLM_SM70_SKINNY", "auto")
    device = torch.device("cuda:0")
    n, k = 5120, 1536
    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        (torch.rand(n, k // 16, device=device) * 400 + 8).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.0021, dtype=torch.float32, device=device),
        requires_grad=False,
    )

    base_kernel = TurboMindNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel = SkinnyNvFp4LinearKernel(NvFp4LinearLayerConfig(), base_kernel)
    kernel.process_weights_after_loading(layer)

    route_calls = {"simt": 0, "qpn": 0}
    original_simt = _sm70_ops.skinny_nvfp4_gemm_simt
    original_qpn = _sm70_ops.skinny_nvfp4_gemm_qpn

    def counted_simt(*args, **kwargs):
        route_calls["simt"] += 1
        return original_simt(*args, **kwargs)

    def counted_qpn(*args, **kwargs):
        route_calls["qpn"] += 1
        return original_qpn(*args, **kwargs)

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", counted_simt)
    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_qpn", counted_qpn)
    compiled_apply = torch.compile(
        lambda values: kernel.apply_weights(layer, values),
        dynamic=True,
        fullgraph=True,
    )

    cases = [
        (1, torch.float16, "simt"),
        (3, torch.float16, "turbomind"),
        (4, torch.float16, "qpn"),
        (8, torch.float16, "qpn"),
        (9, torch.float16, "qpn"),
        (16, torch.float16, "qpn"),
        (17, torch.float16, "turbomind"),
        (2048, torch.float16, "turbomind"),
        (1, torch.bfloat16, "simt"),
        (8, torch.bfloat16, "qpn"),
        (17, torch.bfloat16, "turbomind"),
    ]
    for m, dtype, expected_route in cases:
        before = route_calls.copy()
        x = (torch.randn(m, k, device=device) * 0.1).to(dtype)
        actual = compiled_apply(x)
        reference = base_kernel.apply_weights(layer, x)

        denominator = reference.float().abs().max().clamp(min=1e-6)
        relative_error = (
            (actual.float() - reference.float()).abs().max() / denominator
        ).item()
        assert torch.isfinite(actual).all()
        assert actual.dtype == dtype
        assert relative_error < 3e-2
        if expected_route == "simt":
            assert route_calls["simt"] == before["simt"] + 1
            assert route_calls["qpn"] == before["qpn"]
        elif expected_route == "qpn":
            assert route_calls["qpn"] == before["qpn"] + 1
            assert route_calls["simt"] == before["simt"]
        else:
            assert route_calls == before


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_sm70_skinny_adapter_decorates_real_marlin_base(monkeypatch):
    supported, reason = SkinnyNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    marlin_supported, reason = MarlinNvFp4LinearKernel.is_supported()
    if not marlin_supported:
        pytest.skip(reason)

    class FakeLayer(torch.nn.Module):
        pass

    torch.manual_seed(20260812)
    device = torch.device("cuda:0")
    n, k = 512, 512
    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.params_dtype = torch.float16
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        (torch.rand(n, k // 16, device=device) * 400 + 8).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.0021, dtype=torch.float32, device=device),
        requires_grad=False,
    )

    base_kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel = SkinnyNvFp4LinearKernel(NvFp4LinearLayerConfig(), base_kernel)
    kernel.process_weights_after_loading(layer)

    assert layer.skinny_disabled_routes == set()
    assert {route for _, route in layer.skinny_validated_routes} == {"simt", "qpn"}

    route_calls = {"simt": 0, "qpn": 0}
    original_simt = _sm70_ops.skinny_nvfp4_gemm_simt
    original_qpn = _sm70_ops.skinny_nvfp4_gemm_qpn

    def counted_simt(*args, **kwargs):
        route_calls["simt"] += 1
        return original_simt(*args, **kwargs)

    def counted_qpn(*args, **kwargs):
        route_calls["qpn"] += 1
        return original_qpn(*args, **kwargs)

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", counted_simt)
    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_qpn", counted_qpn)
    compiled_apply = torch.compile(
        lambda values: kernel.apply_weights(layer, values),
        dynamic=True,
        fullgraph=True,
    )

    for m, expected_route in ((1, "simt"), (8, "qpn"), (17, None)):
        before = route_calls.copy()
        x = (torch.randn(m, k, device=device) * 0.01).half()
        actual = compiled_apply(x)
        reference = base_kernel.apply_weights(layer, x)
        denominator = reference.float().abs().max().clamp(min=1e-6)
        relative_error = (
            (actual.float() - reference.float()).abs().max() / denominator
        ).item()
        assert torch.isfinite(actual).all()
        assert relative_error < 3e-2
        if expected_route == "simt":
            assert route_calls["simt"] == before["simt"] + 1
            assert route_calls["qpn"] == before["qpn"]
        elif expected_route == "qpn":
            assert route_calls["qpn"] == before["qpn"] + 1
            assert route_calls["simt"] == before["simt"]
        else:
            assert route_calls == before


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_ct_native_block_scale_loading_matches_fp32_reference(monkeypatch):
    """Exercise the Medium/TC compressed-tensors contract through a real kernel."""
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
    monkeypatch.setenv("VLLM_SM70_SKINNY", "off")
    torch.manual_seed(20260813)
    device = torch.device("cuda:0")
    n, k = 512, 512
    layer = torch.nn.Module()
    layer.logical_widths = [n]
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight_packed = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        (
            torch.randint(2, 9, (n, k // 16), dtype=torch.float32, device=device)
            * 0.015625
        ).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([6048.0], dtype=torch.float32, device=device),
        requires_grad=False,
    )
    raw_codes = layer.weight_packed.detach().clone()
    raw_scales = layer.weight_scale.detach().clone()
    x = (torch.randn(3, k, device=device) * 0.1).half()

    scheme = ct_w4a16.CompressedTensorsW4A16Fp4()
    scheme.process_weights_after_loading(layer)
    actual = scheme.apply_weights(layer, x)
    reference = nvfp4_skinny.nvfp4_fp32_reference(raw_codes, raw_scales, 1.0, x)

    rms_error = (actual.float() - reference.float()).square().mean().sqrt()
    rms_reference = reference.float().square().mean().sqrt().clamp(min=1e-6)
    assert layer._sm70_nvfp4_scale_convention == "native-block"
    assert float(layer.weight_global_scale) == 1.0
    assert torch.isfinite(actual).all()
    assert float(rms_error / rms_reference) < 3e-2
