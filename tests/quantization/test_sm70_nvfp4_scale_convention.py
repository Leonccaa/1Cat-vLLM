# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as ct_w4a4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a16_nvfp4 as ct_w4a16,
)


def _layer(
    block_scales: torch.Tensor,
    disk_global: torch.Tensor,
    logical_widths: list[int] | None = None,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight_scale = torch.nn.Parameter(block_scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(disk_global, requires_grad=False)
    if logical_widths is not None:
        layer.logical_widths = logical_widths
    return layer


def test_native_block_convention_keeps_block_scales_and_uses_identity_global():
    original = torch.tensor([[0.0037, 0.0100], [0.0781, 0.0312]], dtype=torch.float32)
    layer = _layer(original.clone(), torch.tensor([6048.0]))

    convention = sm70_tm.normalize_nvfp4_ct_global_scale_for_sm70(layer)

    assert convention == "native-block"
    assert layer._sm70_nvfp4_scale_convention == "native-block"
    assert float(layer.weight_global_scale) == 1.0
    torch.testing.assert_close(layer.weight_scale, original)


def test_folded_block_convention_reciprocates_disk_global():
    layer = _layer(
        torch.tensor([[22.0, 40.0], [8.0, 16.0]], dtype=torch.float32),
        torch.tensor([6048.0]),
    )

    convention = sm70_tm.normalize_nvfp4_ct_global_scale_for_sm70(layer)

    assert convention == "folded-block"
    assert float(layer.weight_global_scale) == pytest.approx(1.0 / 6048.0)


def test_fused_layer_with_mixed_scale_conventions_fails_closed():
    layer = _layer(
        torch.tensor(
            [[0.01, 0.02], [0.03, 0.04], [8.0, 16.0], [4.0, 2.0]],
            dtype=torch.float32,
        ),
        torch.tensor([6048.0, 6048.0]),
        logical_widths=[2, 2],
    )

    with pytest.raises(RuntimeError, match="mixes native-block and folded-block"):
        sm70_tm.normalize_nvfp4_ct_global_scale_for_sm70(layer)


@pytest.mark.parametrize(
    ("block_scales", "disk_global", "message"),
    [
        (torch.tensor([[float("nan")]]), torch.tensor([6048.0]), "NaN or Inf"),
        (torch.tensor([[1.0]]), torch.tensor([0.0]), "must be positive"),
    ],
)
def test_invalid_scale_metadata_fails_closed(block_scales, disk_global, message):
    layer = _layer(block_scales, disk_global)

    with pytest.raises(RuntimeError, match=message):
        sm70_tm.normalize_nvfp4_ct_global_scale_for_sm70(layer)


def _native_ct_layer(*, with_input_scale: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.logical_widths = [2, 2]
    layer.input_size_per_partition = 16
    layer.output_size_per_partition = 4
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros((4, 8), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor(
            [[0.00390625], [0.0078125], [0.015625], [0.03125]],
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([6048.0, 6048.0]), requires_grad=False
    )
    if with_input_scale:
        layer.input_global_scale = torch.nn.Parameter(
            torch.tensor([4.0, 4.0]), requires_grad=False
        )
    return layer


def test_w4a16_loading_path_normalizes_native_ct_scale_before_kernel(monkeypatch):
    calls: list[str] = []

    class FakeKernel:
        def process_weights_after_loading(self, layer):
            calls.append("kernel")
            assert layer._sm70_nvfp4_scale_convention == "native-block"
            assert float(layer.weight_global_scale) == 1.0

    monkeypatch.setattr(ct_w4a16.sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(ct_w4a16, "init_nvfp4_linear_kernel", FakeKernel)
    layer = _native_ct_layer(with_input_scale=False)

    ct_w4a16.CompressedTensorsW4A16Fp4().process_weights_after_loading(layer)

    assert calls == ["kernel"]
    assert hasattr(layer, "weight")
    assert not hasattr(layer, "weight_packed")


def test_w4a4_loading_path_normalizes_native_ct_scale_before_kernel(monkeypatch):
    calls: list[str] = []

    class FakeKernel:
        def process_weights_after_loading(self, layer):
            calls.append("kernel")
            assert layer._sm70_nvfp4_scale_convention == "native-block"
            assert float(layer.weight_global_scale) == 1.0
            assert float(layer.alpha) == pytest.approx(0.25)

    monkeypatch.setattr(ct_w4a4.sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(ct_w4a4.sm70_tm, "use_turbomind", lambda default: True)
    monkeypatch.setattr(ct_w4a4, "init_nvfp4_linear_kernel", FakeKernel)
    layer = _native_ct_layer(with_input_scale=True)

    ct_w4a4.CompressedTensorsW4A4Fp4().process_weights_after_loading(layer)

    assert calls == ["kernel"]
    assert hasattr(layer, "weight")
    assert not hasattr(layer, "weight_packed")
