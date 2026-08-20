# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import sm70_skinny_fp8 as skinny

pytestmark = pytest.mark.skip_global_cleanup


def test_sm70_skinny_mode_is_orthogonal_to_base(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_SKINNY", raising=False)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "marlin")
    assert envs.get_sm70_quant_backend() == "marlin"
    assert envs.get_sm70_skinny_mode() == "auto"
    assert not envs.use_sm70_skinny_fp8()

    monkeypatch.setenv("VLLM_SM70_SKINNY", "on")
    assert envs.get_sm70_quant_backend() == "marlin"
    assert envs.get_sm70_skinny_mode() == "on"
    assert envs.use_sm70_skinny_fp8()

    monkeypatch.setenv("VLLM_SM70_SKINNY", "off")
    assert not envs.use_sm70_skinny_fp8()


def test_sm70_skinny_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_SKINNY", "sometimes")
    with pytest.raises(ValueError, match="must be one of auto, on, off"):
        envs.get_sm70_skinny_mode()


def test_qpn8_prepack_fragment_order():
    n, k = 128, 128
    weight = torch.arange(n * k, dtype=torch.int32).view(n, k).to(torch.uint8)
    packed = skinny.qpn8_prepack(weight).view(n // 32, k // 16, 32, 16)

    k_order = skinny._QPN_K_ORDER
    for tile in (0, 3):
        for group in (0, 7):
            for lane in (0, 3, 16, 31):
                column = (
                    tile * 32
                    + ((lane >> 2) & 3) * 8
                    + (lane & 3)
                    + (4 if lane & 16 else 0)
                )
                for element in (0, 7, 15):
                    source_k = group * 16 + k_order[element]
                    assert (
                        packed[tile, group, lane, element] == weight[column, source_k]
                    )

    assert torch.equal(
        torch.sort(packed.flatten().to(torch.int32)).values,
        torch.sort(weight.flatten().to(torch.int32)).values,
    )


def test_block_metadata_encodes_recurrence_ratios():
    scales = torch.tensor([[1.0, 2.0, 0.5], [4.0, 1.0, 2.0]], dtype=torch.float32)
    scales256, ratios = skinny.make_block_metadata(scales)
    torch.testing.assert_close(scales256, scales * 256.0)
    torch.testing.assert_close(
        ratios,
        torch.tensor([[1.0, 0.5, 4.0], [1.0, 4.0, 0.5]]),
    )


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    [
        (4352, 5120, (16, 2)),
        (5120, 4352, (16, 2)),
        (2560, 5120, (32, 1)),
        (1536, 5120, (32, 1)),
        (5120, 1536, (16, 2)),
        (128, 128, (8, 2)),
    ],
)
def test_choose_launch_geometry(n, k, expected):
    assert skinny.choose_launch_geometry(n, k) == expected


def test_hybrid_op_routes_only_supported_small_m(monkeypatch):
    calls = []

    def qpn(*args, **kwargs):
        calls.append("qpn")
        x = args[0]
        return x.new_empty((x.shape[0], args[4]))

    def base(**kwargs):
        calls.append("base")
        x = kwargs["input"]
        return x.new_empty((x.shape[0], kwargs["size_n"]))

    monkeypatch.setattr(skinny.ops, "sm70_fp8_qpn8_b128_gemm", qpn)
    monkeypatch.setattr(skinny, "apply_fp8_marlin_linear", base)

    codes = torch.empty(1, dtype=torch.uint8)
    metadata = torch.empty(1)
    base_weight = torch.empty(1, dtype=torch.int32)
    base_scales = torch.empty(1, dtype=torch.float16)
    workspace = torch.empty(1, dtype=torch.int32)

    def call(x, bias=None):
        return skinny._sm70_skinny_fp8_marlin_linear_impl(
            x,
            codes,
            metadata,
            metadata,
            base_weight,
            base_scales,
            workspace,
            bias,
            128,
            128,
            8,
            2,
        )

    assert call(torch.empty((8, 128), dtype=torch.float16)).shape == (8, 128)
    assert calls.pop() == "qpn"

    assert call(torch.empty((9, 128), dtype=torch.float16)).shape == (9, 128)
    assert calls.pop() == "base"

    assert call(torch.empty((1, 128), dtype=torch.bfloat16)).shape == (1, 128)
    assert calls.pop() == "base"

    bias = torch.empty(128, dtype=torch.float16)
    assert call(torch.empty((1, 128), dtype=torch.float16), bias).shape == (1, 128)
    assert calls.pop() == "base"


def test_turbomind_hybrid_preserves_small_m_and_prefill_routes(monkeypatch):
    calls = []

    def qpn(*args, **kwargs):
        calls.append("qpn")
        x = args[0]
        return x.new_zeros((x.shape[0], args[4]))

    def base(out, *args, **kwargs):
        calls.append("base")
        out.fill_(2)

    def prefill(out, *args, **kwargs):
        calls.append("prefill")
        out.fill_(3)

    monkeypatch.setattr(skinny.ops, "sm70_fp8_qpn8_b128_gemm", qpn)
    monkeypatch.setattr(skinny.sm70_ops, "fp8_gemm_sm70_out", base)
    monkeypatch.setattr(skinny.sm70_ops, "fp8_gemm_sm70_prefill_dispatch_out", prefill)

    codes = torch.empty(1, dtype=torch.uint8)
    metadata = torch.empty(1)
    base_weight = torch.empty(1, dtype=torch.int32)
    base_scales = torch.empty(1, dtype=torch.float16)

    def call(x, bias=None, workspace_ptr=0):
        return skinny._sm70_skinny_fp8_turbomind_linear_impl(
            x,
            codes,
            metadata,
            metadata,
            base_weight,
            base_scales,
            bias,
            128,
            128,
            8,
            2,
            128,
            128,
            False,
            workspace_ptr,
            3920,
        )

    assert call(torch.empty((8, 128), dtype=torch.float16)).shape == (8, 128)
    assert calls.pop() == "qpn"

    assert torch.equal(
        call(torch.empty((9, 128), dtype=torch.float16)),
        torch.full((9, 128), 2.0, dtype=torch.float16),
    )
    assert calls.pop() == "base"

    assert torch.equal(
        call(torch.empty((4096, 128), dtype=torch.float16), workspace_ptr=1234),
        torch.full((4096, 128), 3.0, dtype=torch.float16),
    )
    assert calls.pop() == "prefill"

    bias = torch.ones(128, dtype=torch.float16)
    assert torch.equal(
        call(torch.empty((1, 128), dtype=torch.float16), bias),
        torch.full((1, 128), 3.0, dtype=torch.float16),
    )
    assert calls.pop() == "base"


def test_turbomind_hybrid_restores_gated_silu_output_order(monkeypatch):
    def base(out, *args, **kwargs):
        out.copy_(torch.arange(out.numel(), dtype=out.dtype).view_as(out))

    monkeypatch.setattr(skinny.sm70_ops, "fp8_gemm_sm70_out", base)
    x = torch.empty((9, 4), dtype=torch.float16)
    output = skinny._sm70_skinny_fp8_turbomind_linear_impl(
        x,
        torch.empty(1, dtype=torch.uint8),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.float16),
        None,
        8,
        4,
        4,
        1,
        4,
        8,
        True,
        0,
        3920,
    )
    expected = torch.arange(72, dtype=torch.float16).view(9, 4, 2)
    expected = expected.transpose(1, 2).reshape(9, 8)
    assert torch.equal(output, expected)
