# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm import _sm70_ops, envs
from vllm.model_executor.layers.quantization import sm70_skinny


def _pack_awq_rows(logical: torch.Tensor) -> torch.Tensor:
    inverse = torch.argsort(torch.tensor(sm70_skinny._AWQ_REVERSE_PACK_ORDER))
    packed_order = logical.view(logical.shape[0], -1, 8).index_select(-1, inverse)
    byte_view = packed_order[..., 0::2] | (packed_order[..., 1::2] << 4)
    return byte_view.contiguous().view(logical.shape[0], -1).view(torch.int32)


def _native_awq(n: int, k: int):
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
    return _pack_awq_rows(logical_weight), scales, _pack_awq_rows(logical_zeros)


def _unpack_code(codes: torch.Tensor, row: int, column: int) -> int:
    packed = int(codes[row, column // 2])
    return (packed >> (4 * (column & 1))) & 0xF


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
    qweight, scales, qzeros = _native_awq(n, k)
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
            state, torch.ones((m, k), dtype=torch.float16)
        )
        if out is None:
            calls.append("selected_base")
    assert calls == ["simt", "qpn", "selected_base"]


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
    assert sm70_skinny.select_awq_route(state, 1) is None
    assert sm70_skinny.select_awq_route(state, 8) == "qpn"


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
    # Base and Skinny take the same time -> zero saving -> negative ROI floor
    # is the only thing that could keep it.
    monkeypatch.setattr(sm70_skinny, "_time_apply", lambda fn, iterations=20: 10.0)
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "1.0")

    kept = sm70_skinny.apply_residency_policy(state, lambda x: x)

    assert kept is False
    assert state.codes.numel() == 0
    assert state.scales.numel() == 0
    assert state.disabled_routes == {"simt", "qpn"}
    assert sm70_skinny.select_awq_route(state, 1) is None


def test_residency_policy_keeps_shape_that_earns_its_memory(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    timings = iter([500.0, 10.0])  # base slow, skinny fast
    monkeypatch.setattr(
        sm70_skinny, "_time_apply", lambda fn, iterations=20: next(timings)
    )
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "0")

    assert sm70_skinny.apply_residency_policy(state, lambda x: x) is True
    assert state.codes.numel() != 0
    assert state.disabled_routes == set()


def test_residency_decision_is_cached_per_shape(monkeypatch):
    qweight, scales, qzeros = _native_awq(32, 128)
    calls = []

    monkeypatch.setattr(sm70_skinny, "_residency_decisions", {})
    monkeypatch.setenv("VLLM_SM70_SKINNY_MIN_ROI", "0")

    def _timer(fn, iterations=20):
        calls.append(1)
        return 500.0 if len(calls) % 2 else 10.0

    monkeypatch.setattr(sm70_skinny, "_time_apply", _timer)
    for _ in range(4):
        state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
        sm70_skinny.apply_residency_policy(state, lambda x: x)

    # Two timed measurements for the first layer, none for the repeats.
    assert len(calls) == 2
