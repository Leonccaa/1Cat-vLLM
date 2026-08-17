# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops
from vllm.model_executor.layers.quantization import sm70_skinny


def _is_exact_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_sm70_skinny_awq_routes_and_matches_turbomind(monkeypatch):
    required = (
        "skinny_awq_gemm_simt",
        "skinny_awq_gemm_qpn",
        "awq_sm70_prepare",
        "awq_gemm_sm70_out",
    )
    missing = [name for name in required if not hasattr(torch.ops._C, name)]
    if missing:
        pytest.skip("missing SM70 extension ops: " + ", ".join(missing))

    torch.manual_seed(20260812)
    monkeypatch.setenv("VLLM_SM70_SKINNY", "auto")
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")
    device = torch.device("cuda:0")
    n, k = 5120, 1536
    qweight = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device).view(
        torch.int32
    )
    qzeros = torch.randint(
        0, 256, (k // 128, n // 2), dtype=torch.uint8, device=device
    ).view(torch.int32)
    scales = (
        torch.rand((k // 128, n), dtype=torch.float16, device=device) * 0.05 + 0.001
    )
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    tm_weight, tm_scales, meta = _sm70_ops.awq_sm70_prepare(
        qweight, scales, qzeros, 128, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])

    def base_apply(input: torch.Tensor) -> torch.Tensor:
        kernel_input = (
            input if input.dtype == torch.float16 else input.to(torch.float16)
        )
        out = torch.empty(
            (kernel_input.shape[0], n),
            dtype=torch.float16,
            device=kernel_input.device,
        )
        _sm70_ops.awq_gemm_sm70_out(
            out, kernel_input, tm_weight, tm_scales, 128, k_ld, q_ld, False
        )
        return out if input.dtype == torch.float16 else out.to(input.dtype)

    route_calls = {"simt": 0, "qpn": 0}
    original_simt = _sm70_ops.skinny_awq_gemm_simt
    original_qpn = _sm70_ops.skinny_awq_gemm_qpn

    def counted_simt(*args, **kwargs):
        route_calls["simt"] += 1
        return original_simt(*args, **kwargs)

    def counted_qpn(*args, **kwargs):
        route_calls["qpn"] += 1
        return original_qpn(*args, **kwargs)

    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_simt", counted_simt)
    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_qpn", counted_qpn)

    cases = (
        (1, torch.float16, "simt"),
        (3, torch.float16, "turbomind"),
        (4, torch.float16, "qpn"),
        (8, torch.float16, "qpn"),
        (9, torch.float16, "qpn"),
        (16, torch.float16, "qpn"),
        (17, torch.float16, "turbomind"),
        (1, torch.bfloat16, "simt"),
        (8, torch.bfloat16, "qpn"),
        (17, torch.bfloat16, "turbomind"),
    )
    for m, dtype, expected_route in cases:
        before = route_calls.copy()
        values = torch.arange(m * k, dtype=torch.int32, device=device)
        x = ((values.remainder(31) - 15).to(dtype) * 1e-3).view(m, k)
        actual = sm70_skinny.try_apply_awq_state(state, x)
        reference = base_apply(x)
        if actual is None:
            actual = reference

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
def test_sm70_skinny_awq_decorates_real_marlin_base(monkeypatch):
    required = (
        "skinny_awq_gemm_simt",
        "skinny_awq_gemm_qpn",
        "gptq_marlin_repack",
        "marlin_gemm",
    )
    missing = [name for name in required if not hasattr(torch.ops._C, name)]
    if missing:
        pytest.skip("missing SM70 extension ops: " + ", ".join(missing))

    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMarlinLinearMethod,
    )

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    monkeypatch.setenv("VLLM_SM70_SKINNY", "on")
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")

    torch.manual_seed(20260812)
    device = torch.device("cuda:0")
    n, k = 512, 512
    config = AWQMarlinConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
    )
    method = AWQMarlinLinearMethod(config)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        k,
        [n],
        k,
        n,
        torch.float16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.to(device)
    layer.qweight.data.copy_(
        torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device).view(
            torch.int32
        )
    )
    layer.qzeros.data.copy_(
        torch.randint(
            0,
            256,
            (k // 128, n // 2),
            dtype=torch.uint8,
            device=device,
        ).view(torch.int32)
    )
    layer.scales.data.copy_(
        torch.rand((k // 128, n), dtype=torch.float16, device=device) * 0.05 + 0.001
    )

    method.process_weights_after_loading(layer)
    state = layer._awq_sm70_skinny
    assert "Marlin" in type(method.kernel).__name__
    assert state.disabled_routes == set()
    assert {route for _, route in state.validated_routes} == {"simt", "qpn"}

    for m, expected_route in ((1, "simt"), (8, "qpn"), (17, None)):
        x = (torch.randn(m, k, device=device) * 0.01).half()
        actual = method.apply(layer, x)
        reference = method.kernel.apply_weights(layer, x, None)
        denominator = reference.float().abs().max().clamp(min=1e-6)
        relative_error = (
            (actual.float() - reference.float()).abs().max() / denominator
        ).item()
        assert torch.isfinite(actual).all()
        assert relative_error < 3e-2
        assert sm70_skinny.select_awq_route(state, m) == expected_route


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_sm70_skinny_awq_grouped_moe_matches_per_expert_turbomind():
    if not hasattr(torch.ops._C, "skinny_awq_moe_gemm_simt_out"):
        pytest.skip("missing SM70 Skinny AWQ grouped MoE op")

    torch.manual_seed(20260812)
    device = torch.device("cuda:0")
    experts, rows, n, k = 4, 9, 512, 512
    qweight = torch.randint(
        0, 256, (experts, k, n // 2), dtype=torch.uint8, device=device
    ).view(torch.int32)
    qzeros = torch.randint(
        0,
        256,
        (experts, k // 128, n // 2),
        dtype=torch.uint8,
        device=device,
    ).view(torch.int32)
    scales = (
        torch.rand((experts, k // 128, n), dtype=torch.float16, device=device) * 0.05
        + 0.001
    )
    codes, logical_scales, biases = sm70_skinny.prepare_awq_moe_bank(
        qweight, scales, qzeros, 128
    )
    expert_ids = torch.tensor(
        [0, 3, 1, 1, 2, 0, 3, 2, 1], dtype=torch.int32, device=device
    )
    values = torch.arange(rows * k, dtype=torch.int32, device=device)
    x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(rows, k)
    actual = torch.empty((rows, n), dtype=torch.float16, device=device)
    _sm70_ops.skinny_awq_moe_gemm_simt_out(
        actual, x, expert_ids, codes, logical_scales, biases, 128
    )

    reference = torch.empty_like(actual)
    prepared = [
        _sm70_ops.awq_sm70_prepare(
            qweight[expert], scales[expert], qzeros[expert], 128, False
        )
        for expert in range(experts)
    ]
    for row, expert in enumerate(expert_ids.cpu().tolist()):
        weight, tm_scales, meta = prepared[expert]
        _sm70_ops.awq_gemm_sm70_out(
            reference[row : row + 1],
            x[row : row + 1],
            weight,
            tm_scales,
            128,
            int(meta[0]),
            int(meta[1]),
            False,
        )

    denominator = reference.float().abs().max().clamp(min=1e-6)
    relative_error = (
        (actual.float() - reference.float()).abs().max() / denominator
    ).item()
    assert torch.isfinite(actual).all()
    assert relative_error < 3e-2


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@pytest.mark.parametrize("m", [1, 300])
@torch.inference_mode()
def test_sm70_awq_fallback_is_finite_and_matches_native_fp32(monkeypatch, m):
    from vllm.model_executor.layers.quantization.awq import (
        AWQConfig,
        AWQLinearMethod,
    )

    torch.manual_seed(20260813 + m)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    device = torch.device("cuda:0")
    n, k, group_size = 512, 512, 128
    qweight = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device).view(
        torch.int32
    )
    qzeros = torch.randint(
        0, 256, (k // group_size, n // 2), dtype=torch.uint8, device=device
    ).view(torch.int32)
    scales = (
        torch.rand((k // group_size, n), dtype=torch.float16, device=device) * 0.03
        + 0.002
    )
    layer = torch.nn.Module()
    layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
    layer.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
    layer.scales = torch.nn.Parameter(scales, requires_grad=False)
    x = (torch.randn(m, k, device=device) * 0.02).half()

    method = AWQLinearMethod(AWQConfig(4, group_size, True))
    actual = method.apply(layer, x)
    reference = sm70_skinny.awq_native_fp32_reference(
        qweight, scales, qzeros, group_size, x
    )

    assert torch.isfinite(actual).all()
    rms_reference = reference.square().mean().sqrt().clamp(min=1e-8)
    rms_relative = (actual.float() - reference).square().mean().sqrt() / rms_reference
    assert float(rms_relative) < 1e-3
