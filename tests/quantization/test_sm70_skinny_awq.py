# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _sm70_ops, envs
from vllm.model_executor.layers.quantization import (
    awq as awq_module,
)
from vllm.model_executor.layers.quantization import (
    awq_triton,
    sm70_residency,
    sm70_skinny,
)


def _pack_awq_rows(logical: torch.Tensor) -> torch.Tensor:
    inverse = torch.argsort(torch.tensor(sm70_skinny._AWQ_REVERSE_PACK_ORDER))
    packed_order = logical.view(logical.shape[0], -1, 8).index_select(-1, inverse)
    byte_view = packed_order[..., 0::2] | (packed_order[..., 1::2] << 4)
    return byte_view.contiguous().view(logical.shape[0], -1).view(torch.int32)


def _overlay_device() -> torch.device:
    """Device the graph-safe overlay ops can actually dispatch on.

    ``direct_register_custom_op`` registers these ops for CUDA and Meta only,
    so feeding them CPU tensors raises NotImplementedError on any machine that
    has a GPU. Tests that go through ``torch.ops.vllm.*`` must therefore follow
    the available device rather than assume CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _native_awq(n: int, k: int, device: torch.device | None = None):
    logical_weight = (
        torch.arange(k * n, dtype=torch.int64).remainder(16).to(torch.uint8)
    )
    logical_weight = logical_weight.view(k, n)
    logical_zeros = torch.arange((k // 128) * n, dtype=torch.int64)
    logical_zeros = logical_zeros.remainder(16).to(torch.uint8).view(k // 128, n)
    scales = (
        torch.arange((k // 128) * n, dtype=torch.float16).view(k // 128, n) / 1024
        + 0.125
    )
    qweight = _pack_awq_rows(logical_weight)
    qzeros = _pack_awq_rows(logical_zeros)
    if device is not None:
        qweight = qweight.to(device)
        scales = scales.to(device)
        qzeros = qzeros.to(device)
    return qweight, scales, qzeros


def _unpack_code(codes: torch.Tensor, row: int, column: int) -> int:
    packed = int(codes[row, column // 2])
    return (packed >> (4 * (column & 1))) & 0xF


@pytest.mark.parametrize("m", [1, 300])
def test_sm70_classic_awq_fallback_uses_triton_dequant(monkeypatch, m):
    n, k = 32, 128
    qweight, scales, qzeros = _native_awq(n, k)
    layer = SimpleNamespace(qweight=qweight, scales=scales, qzeros=qzeros)
    method = awq_module.AWQLinearMethod(awq_module.AWQConfig(4, 128, True))
    x = torch.arange(m * k, dtype=torch.float16).view(m, k) / 1024
    dequantized = torch.arange(k * n, dtype=torch.float16).view(k, n) / 2048

    monkeypatch.setattr(awq_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        awq_module.current_platform, "has_device_capability", lambda capability: False
    )
    monkeypatch.setattr(
        awq_triton,
        "awq_dequantize_triton",
        lambda *args, **kwargs: dequantized,
    )
    monkeypatch.setattr(
        awq_module.ops,
        "awq_gemm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("classic AWQ GEMM must not run on SM70")
        ),
    )
    monkeypatch.setattr(
        awq_module.ops,
        "awq_dequantize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("classic AWQ dequant must not run on SM70")
        ),
    )

    actual = method.apply(layer, x)

    torch.testing.assert_close(actual, x @ dequantized)


def test_unpack_awq_dense_restores_logical_values_and_bias():
    n, k = 32, 128
    qweight, scales, qzeros = _native_awq(n, k)

    codes, logical_scales, biases = sm70_skinny.unpack_awq_dense(
        qweight, scales, qzeros, 128
    )

    expected_weight = torch.arange(k * n).remainder(16).view(k, n).t()
    expected_zeros = torch.arange(n).remainder(16).to(torch.float16).view(n, 1)
    assert all(
        _unpack_code(codes, row, column) == int(expected_weight[row, column])
        for row in range(n)
        for column in range(k)
    )
    assert torch.equal(logical_scales, scales.t())
    assert torch.equal(biases, -expected_zeros * logical_scales)


def test_qpn_prepack_awq_matches_fragment_reference():
    n, k = 32, 256
    qweight, scales, qzeros = _native_awq(n, k)
    codes, logical_scales, biases = sm70_skinny.unpack_awq_dense(
        qweight, scales, qzeros, 128
    )

    qcodes, qscales, qbiases = sm70_skinny.qpn_prepack_awq(
        codes, logical_scales, biases
    )

    assert qcodes is not None and qscales is not None and qbiases is not None
    expected_codes: list[int] = []
    for group in range(k // 16):
        for lane in range(32):
            column = ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0)
            selected = [
                _unpack_code(codes, column, group * 16 + offset)
                for offset in sm70_skinny._QPN_K_ORDER
            ]
            expected_codes.extend(
                selected[index] | (selected[index + 1] << 4)
                for index in range(0, 16, 2)
            )
    col_order = [
        ((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0)
        for lane in range(32)
    ]
    expected_scales = logical_scales[col_order].t().reshape(-1)
    expected_biases = biases[col_order].t().reshape(-1)
    assert qcodes.tolist() == expected_codes
    assert torch.equal(qscales, expected_scales)
    assert torch.equal(qbiases, expected_biases)


def test_prepare_awq_state_respects_resident_layout(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)

    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "qpn")
    qpn = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert not qpn.has_simt
    assert qpn.has_qpn

    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")
    both = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert both.has_simt
    assert both.has_qpn


@pytest.mark.parametrize(
    ("input_size", "output_size", "expected_splits"),
    [
        # Qwen3.6-27B TP4 dense shapes.
        (1536, 5120, 1),
        (4352, 5120, 1),
        (5120, 8704, 1),
        (5120, 4096, 2),
        # Narrow-N and minimum-groups-per-warp boundaries.
        (5120, 1792, 4),
        (128, 32, 1),
        (512, 32, 2),
        (4096, 32, 16),
    ],
)
def test_qpn_split_geometry_matches_cuda_contract(
    input_size: int, output_size: int, expected_splits: int
):
    # The identical table is compiled as static_asserts beside the C++ policy.
    assert sm70_skinny.qpn_k_splits(input_size, output_size) == expected_splits


def test_prepare_awq_moe_bank_is_one_n_major_layout():
    n, k, experts = 32, 128, 3
    native = [_native_awq(n, k) for _ in range(experts)]
    qweight = torch.stack([item[0] for item in native])
    scales = torch.stack([item[1] for item in native])
    qzeros = torch.stack([item[2] for item in native])

    codes, logical_scales, biases = sm70_skinny.prepare_awq_moe_bank(
        qweight, scales, qzeros, 128
    )

    assert codes.shape == (experts, n, k // 2)
    assert logical_scales.shape == (experts, n, k // 128)
    assert biases.shape == logical_scales.shape
    for expert in range(experts):
        expected = sm70_skinny.unpack_awq_dense(
            qweight[expert], scales[expert], qzeros[expert], 128
        )
        assert torch.equal(codes[expert], expected[0])
        assert torch.equal(logical_scales[expert], expected[1])
        assert torch.equal(biases[expert], expected[2])


def test_awq_overlay_dispatches_simt_qpn_then_delegates_base(monkeypatch):
    n, k = 32, 128
    # Goes through torch.ops.vllm.sm70_skinny_awq_linear, which is registered
    # for CUDA/Meta only.
    device = _overlay_device()
    qweight, scales, qzeros = _native_awq(n, k, device)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    calls: list[str] = []

    def simt(input, codes, scales, biases, group_size):
        del codes, scales, biases, group_size
        calls.append("simt")
        return input.new_zeros((input.shape[0], n))

    def qpn(input, codes, scales, biases, group_size, output_size):
        del codes, scales, biases, group_size
        calls.append("qpn")
        return input.new_zeros((input.shape[0], output_size))

    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_simt", simt)
    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_qpn", qpn)

    for m in (1, 8, 17):
        out = sm70_skinny.try_apply_awq_state(
            state, torch.ones((m, k), dtype=torch.float16, device=device)
        )
        if out is None:
            calls.append("selected_base")
    assert calls == ["simt", "qpn", "selected_base"]


def test_awq_route_ledger_keeps_distinct_shapes(monkeypatch):
    monkeypatch.setattr(sm70_skinny, "_awq_route_log_seen", set())

    def simt(input, codes, scales, biases, group_size):
        del scales, biases, group_size
        return input.new_zeros((input.shape[0], codes.shape[0]))

    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_simt", simt)
    for n in (32, 64):
        k = 128
        qweight, scales, qzeros = _native_awq(n, k)
        codes, logical_scales, biases = sm70_skinny.unpack_awq_dense(
            qweight, scales, qzeros, 128
        )
        empty = torch.empty(0)
        out = sm70_skinny._try_awq_skinny_linear(
            torch.ones((1, k), dtype=torch.float16),
            codes,
            logical_scales,
            biases,
            empty,
            empty,
            empty,
            n,
            k,
            128,
        )
        assert out is not None

    assert sm70_skinny._awq_route_log_seen == {
        ("simt", 1, 32, 128, torch.float16),
        ("simt", 1, 64, 128, torch.float16),
    }


def test_awq_bf16_is_explicitly_converted_and_restored(monkeypatch):
    n, k = 32, 128
    qweight, scales, qzeros = _native_awq(n, k)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "simt")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    seen: list[torch.dtype] = []

    def simt(input, codes, scales, biases, group_size):
        del codes, scales, biases, group_size
        seen.append(input.dtype)
        return input.new_zeros((input.shape[0], n))

    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_simt", simt)
    out = sm70_skinny._skinny_awq_linear_impl(
        torch.ones((1, k), dtype=torch.bfloat16),
        state.codes,
        state.scales,
        state.biases,
        state.qpn_codes,
        state.qpn_scales,
        state.qpn_biases,
        n,
        k,
        128,
    )
    assert seen == [torch.float16]
    assert out.dtype == torch.bfloat16


def test_awq_turbomind_hybrid_op_dispatches_on_runtime_rows(monkeypatch):
    n, k = 32, 128
    qweight, scales, qzeros = _native_awq(n, k)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "simt")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    calls: list[tuple[str, int]] = []

    def simt(input, codes, scales, biases, group_size):
        del codes, scales, biases, group_size
        calls.append(("simt", input.shape[0]))
        return input.new_ones((input.shape[0], n))

    def base(out, input, weight, scales, group_size, k_ld, q_ld):
        del weight, scales, group_size, k_ld, q_ld
        calls.append(("base", input.shape[0]))
        out.fill_(2)

    monkeypatch.setattr(_sm70_ops, "skinny_awq_gemm_simt", simt)
    monkeypatch.setattr(_sm70_ops, "awq_gemm_sm70_out", base)
    base_weight = torch.empty((1, 1), dtype=torch.int32)
    base_scales = torch.empty((1, 1), dtype=torch.float16)

    small = sm70_skinny._skinny_awq_turbomind_linear_impl(
        torch.ones((1, k), dtype=torch.float16),
        state.codes,
        state.scales,
        state.biases,
        state.qpn_codes,
        state.qpn_scales,
        state.qpn_biases,
        base_weight,
        base_scales,
        n,
        k,
        128,
        0,
        0,
    )
    large = sm70_skinny._skinny_awq_turbomind_linear_impl(
        torch.ones((17, k), dtype=torch.float16),
        state.codes,
        state.scales,
        state.biases,
        state.qpn_codes,
        state.qpn_scales,
        state.qpn_biases,
        base_weight,
        base_scales,
        n,
        k,
        128,
        0,
        0,
    )

    assert calls == [("simt", 1), ("base", 17)]
    assert torch.equal(small, torch.ones_like(small))
    assert torch.equal(large, torch.full_like(large, 2))


def test_skinny_awq_env_contract(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "simt")
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_MOE", "1")
    assert envs.use_sm70_skinny_awq()
    assert envs.get_sm70_skinny_awq_layout() == "simt"
    assert envs.use_sm70_skinny_awq_moe()

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    monkeypatch.setenv("VLLM_SM70_SKINNY", "off")
    assert envs.get_sm70_quant_base_backend() == "marlin"
    assert not envs.use_sm70_skinny_awq()


def test_selected_base_backend_respects_existing_selectors(monkeypatch):
    import vllm.config

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "auto")
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(kernel_config=SimpleNamespace(linear_backend="marlin")),
    )
    assert sm70_skinny.selected_base_backend() == "marlin"

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
    assert sm70_skinny.selected_base_backend() == "turbomind"

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    assert sm70_skinny.selected_base_backend() == "marlin"

    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "skinny")
    assert sm70_skinny.selected_base_backend() == "marlin"
    assert envs.get_sm70_skinny_mode() == "on"


def test_awq_self_check_disables_only_failing_route(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)

    monkeypatch.setattr(
        _sm70_ops,
        "skinny_awq_gemm_simt",
        lambda x, *_: torch.full((x.shape[0], 32), 100.0, dtype=x.dtype),
    )
    monkeypatch.setattr(
        _sm70_ops,
        "skinny_awq_gemm_qpn",
        lambda x, *args: torch.zeros((x.shape[0], 32), dtype=x.dtype),
    )
    sm70_skinny.validate_awq_state(
        state,
        lambda x: torch.zeros((x.shape[0], 32), dtype=x.dtype),
        "fake-base",
    )

    assert state.disabled_routes == {"simt"}
    assert state.codes.numel() == 0
    assert state.scales.numel() == 0
    assert state.biases.numel() == 0
    assert sm70_skinny.select_awq_route(state, 1) is None
    assert sm70_skinny.select_awq_route(state, 8) == "qpn"


def test_awq_force_on_fails_closed_on_self_check_error(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    monkeypatch.setenv("VLLM_SM70_SKINNY", "on")
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "simt")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    monkeypatch.setattr(
        _sm70_ops,
        "skinny_awq_gemm_simt",
        lambda x, *_: torch.full((x.shape[0], 32), 100.0, dtype=x.dtype),
    )

    with pytest.raises(RuntimeError, match="requires the AWQ simt self-check"):
        sm70_skinny.validate_awq_state(
            state,
            lambda x: torch.zeros((x.shape[0], 32), dtype=x.dtype),
            "fake-base",
        )

    assert state.codes.numel() == 0


def test_awq_fp32_reference_matches_elementwise_dequant():
    """The FP32 ground truth is only worth having if it is itself checked.

    Compare the vectorized reference against an explicit per-element walk of
    the packed bytes, so a nibble-order or group-cadence slip in the reference
    cannot silently bless a matching slip in the kernel.
    """
    torch.manual_seed(0)
    n, k, group = 24, 256, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = (torch.rand(n, k // group) * 0.02 + 0.002).to(torch.float16)
    biases = (-torch.rand(n, k // group) * 0.1).to(torch.float16)
    x = (torch.rand(2, k) - 0.5).to(torch.float16)

    actual = sm70_skinny.awq_fp32_reference(codes, scales, biases, x)

    expected = torch.zeros(2, n, dtype=torch.float32)
    for col in range(n):
        for kk in range(k):
            byte = int(codes[col, kk // 2])
            quant = byte & 0xF if kk % 2 == 0 else byte >> 4
            weight = quant * float(scales[col, kk // group]) + float(
                biases[col, kk // group]
            )
            for row in range(2):
                expected[row, col] += float(x[row, kk]) * weight

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_awq_fp32_reference_chunking_is_transparent():
    torch.manual_seed(1)
    n, k = 64, 256
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = (torch.rand(n, k // 128) * 0.02 + 0.002).to(torch.float16)
    biases = (-torch.rand(n, k // 128) * 0.1).to(torch.float16)
    x = (torch.rand(1, k) - 0.5).to(torch.float16)

    whole = sm70_skinny.awq_fp32_reference(codes, scales, biases, x, chunk=n)
    chunked = sm70_skinny.awq_fp32_reference(codes, scales, biases, x, chunk=7)
    torch.testing.assert_close(whole, chunked)


def test_residency_policy_drops_shape_below_roi_threshold(monkeypatch):
    """A shape whose overlay wins nothing must release its VRAM."""
    qweight, scales, qzeros = _native_awq(32, 128)
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert state.has_simt

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    monkeypatch.setattr(
        sm70_residency, "time_apply", lambda fn, iterations=12, device=None: 10.0
    )
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "1.0")

    kept = sm70_skinny.apply_residency_policy(state, lambda x: x)

    assert kept is False
    assert state.codes.numel() == 0
    assert state.scales.numel() == 0
    assert "simt" in state.disabled_routes
    assert sm70_skinny.select_awq_route(state, 1) is None


def test_residency_policy_keeps_shape_that_earns_its_memory(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    timings = iter([500.0, 10.0])  # base slow, skinny fast
    monkeypatch.setattr(
        sm70_residency,
        "time_apply",
        lambda fn, iterations=12, device=None: next(timings),
    )
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "0")

    assert sm70_skinny.apply_residency_policy(state, lambda x: x) is True
    assert state.codes.numel() != 0
    assert state.disabled_routes == set()


def test_force_on_bypasses_performance_residency_gate(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    monkeypatch.setenv("VLLM_SM70_SKINNY", "on")
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "999")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    monkeypatch.setattr(
        sm70_residency,
        "time_apply",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("force-on must not run the performance gate")
        ),
    )

    assert sm70_skinny.apply_residency_policy(state, lambda x: x) is True
    assert state.has_simt


def test_tp_consensus_propagates_a_peer_failure(monkeypatch):
    import vllm.distributed

    group = SimpleNamespace(world_size=4, device_group=object())
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: group)

    def peer_failed(tensor, op, group):
        del op, group
        assert tensor.tolist() == [1.0, 12.5]
        tensor[0] = 0.0

    monkeypatch.setattr(torch.distributed, "all_reduce", peer_failed)
    assert sm70_residency.agree_across_tp(12.5, device="cpu") is None


def test_tp_consensus_uses_local_value_before_tp_initialization(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_is_initialized", lambda: False
    )
    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: (_ for _ in ()).throw(AssertionError("must not access TP group")),
    )

    assert sm70_residency.agree_across_tp(12.5, device="cpu") == 12.5


def test_tp_consensus_uses_local_value_before_distributed_initialization(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not enter a collective before initialization")
        ),
    )

    assert sm70_residency.agree_across_tp(12.5, device="cpu") == 12.5


def test_tp_consensus_propagates_collective_failure(monkeypatch):
    import vllm.distributed

    group = SimpleNamespace(world_size=4, device_group=object())
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated NCCL failure")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated NCCL failure"):
        sm70_residency.agree_across_tp(12.5, device="cpu")


def test_residency_decision_is_cached_per_shape(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    calls = []

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "0")

    def _timer(fn, iterations=12, device=None):
        calls.append(1)
        return 500.0 if len(calls) % 2 else 10.0

    monkeypatch.setattr(sm70_residency, "time_apply", _timer)
    for _ in range(4):
        state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
        sm70_skinny.apply_residency_policy(state, lambda x: x)

    # Two timed measurements for the first layer, none for the repeats.
    assert len(calls) == 2


def test_residency_cache_key_includes_format(monkeypatch):
    """The same shape and route in AWQ and NVFP4 must be measured separately."""
    decisions: dict[sm70_residency.ResidencyKey, sm70_residency.ResidencyDecision] = {}
    calls = []
    timings = iter([500.0, 10.0, 10.0, 10.0])

    def timer(fn, iterations=12, device=None):
        del fn, iterations, device
        calls.append(1)
        return next(timings)

    monkeypatch.setattr(sm70_residency, "time_apply", timer)

    def benchmark(route: str) -> sm70_residency.RouteBenchmark:
        del route
        return sm70_residency.RouteBenchmark(
            lambda: torch.empty(0), lambda: torch.empty(0)
        )

    awq_released = []
    nvfp4_released = []
    common = dict(
        output_size=32,
        input_size=128,
        routes=["simt"],
        device=torch.device("cpu"),
        overlay_mib=1.0,
        min_roi=1.0,
        force_on=False,
        decisions=decisions,
        make_benchmark=benchmark,
    )
    sm70_residency.apply_route_policy(
        format_name="AWQ",
        release_route=awq_released.append,
        **common,
    )
    sm70_residency.apply_route_policy(
        format_name="NVFP4",
        release_route=nvfp4_released.append,
        **common,
    )

    assert len(calls) == 4
    assert set(decisions) == {
        ("AWQ", 32, 128, "simt"),
        ("NVFP4", 32, 128, "simt"),
    }
    assert awq_released == []
    assert nvfp4_released == ["simt"]


def test_residency_summary_is_ranked_and_format_scoped(monkeypatch):
    decisions = {
        ("AWQ", 999, 128, "simt"): sm70_residency.ResidencyDecision(
            roi=9.0, mib=1.0, saved_us=9.0, keep=True
        ),
        ("NVFP4", 64, 256, "simt"): sm70_residency.ResidencyDecision(
            roi=1.0, mib=2.0, saved_us=2.0, keep=True
        ),
        ("NVFP4", 32, 128, "qpn"): sm70_residency.ResidencyDecision(
            roi=2.0, mib=3.0, saved_us=6.0, keep=False
        ),
    }
    messages = []
    monkeypatch.setattr(
        sm70_residency.logger,
        "info_once",
        lambda message, *args: messages.append(message % args),
    )

    sm70_residency.log_residency_summary(
        decisions=decisions,
        format_name="NVFP4",
        min_roi=1.5,
    )

    assert len(messages) == 1
    summary = messages[0]
    assert "SM70 Skinny NVFP4 residency summary" in summary
    assert "N=   999" not in summary
    assert summary.index("qpn") < summary.index("simt")
    assert "kept 2.0 MiB/layer-set, dropped 3.0 MiB/layer-set" in summary


def test_residency_scores_simt_and_qpn_independently(monkeypatch):
    """SIMT and QPN serve different M and must be kept or dropped separately.

    Coupling them means an MTP-only win pays for a decode layout that earns
    nothing, or the reverse.
    """
    qweight, scales, qzeros = _native_awq(32, 128)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "both")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert state.has_simt and state.has_qpn

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "0.001")

    # simt: base 10 vs skinny 10 -> no gain.  qpn: base 500 vs skinny 10 -> big.
    seq = iter([10.0, 10.0, 500.0, 10.0])
    monkeypatch.setattr(
        sm70_residency,
        "time_apply",
        lambda fn, iterations=12, device=None: next(seq),
    )

    assert sm70_skinny.apply_residency_policy(state, lambda x: x) is True
    assert state.codes.numel() == 0, "SIMT should have been released"
    assert state.qpn_codes.numel() != 0, "QPN earned its memory and must stay"
    assert state.disabled_routes == {"simt"}
    assert sm70_skinny.select_awq_route(state, 1) is None
    assert sm70_skinny.select_awq_route(state, 8) == "qpn"


def test_residency_measures_qpn_when_simt_is_not_resident(monkeypatch):
    """With layout=qpn there is no SIMT buffer; the gate must still apply."""
    qweight, scales, qzeros = _native_awq(32, 128)
    monkeypatch.setenv("VLLM_SM70_SKINNY_AWQ_LAYOUT", "qpn")
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    assert not state.has_simt and state.has_qpn

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "1.0")
    calls = []

    def _timer(fn, iterations=12, device=None):
        calls.append(1)
        return 10.0  # no gain -> must drop

    monkeypatch.setattr(sm70_residency, "time_apply", _timer)

    kept = sm70_skinny.apply_residency_policy(state, lambda x: x)

    assert calls, "QPN-only layout was silently skipped instead of measured"
    assert kept is False
    assert state.qpn_codes.numel() == 0
    assert state.disabled_routes == {"qpn"}


def test_awq_native_fp32_reference_matches_skinny_layout_reference():
    """The native-tensor ground truth must agree with the Skinny-layout one.

    They start from different representations - int32 checkpoint packing vs the
    prepacked N-major bytes - so agreement exercises the prepack itself.

    They are not bit-identical by construction: the Skinny layout folds the
    zero point into an FP16 ``bias = -z*s`` computed at prepack time, whereas
    the native form evaluates ``(q - z) * s`` directly, so they differ by the
    FP16 rounding of that bias. Use a random fixture rather than the correlated
    arange one, which drives the output to near-zero and lets that 1-ulp term
    dominate a relative comparison.
    """
    torch.manual_seed(0)
    n, k = 32, 256
    logical_weight = torch.randint(0, 16, (k, n), dtype=torch.uint8)
    logical_zeros = torch.randint(0, 4, (k // 128, n), dtype=torch.uint8)
    scales = (torch.rand(k // 128, n) * 0.02 + 0.01).to(torch.float16)
    qweight = _pack_awq_rows(logical_weight)
    qzeros = _pack_awq_rows(logical_zeros)

    codes, logical_scales, biases = sm70_skinny.unpack_awq_dense(
        qweight, scales, qzeros, 128
    )
    x = (torch.rand(2, k) - 0.5).to(torch.float16)

    from_native = sm70_skinny.awq_native_fp32_reference(qweight, scales, qzeros, 128, x)
    from_skinny = sm70_skinny.awq_fp32_reference(codes, logical_scales, biases, x)

    scale = from_native.abs().max().clamp(min=1e-6)
    relative = (from_native - from_skinny).abs().max() / scale
    assert relative < 1e-3, f"references disagree by {relative:.3e}"
