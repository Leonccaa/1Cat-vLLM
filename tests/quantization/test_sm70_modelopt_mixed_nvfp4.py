# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow SM70 admission and shape gates for ModelOpt mixed NVFP4."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    _prepare_compact_slot_groups,
    _validate_weight_layout,
    validate_nvfp4_sm70_moe_contract,
)


def _mixed_config() -> ModelOptMixedPrecisionConfig:
    fp8 = ModelOptFp8Config("FP8", True, None, [])
    nvfp4 = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )
    w4a16 = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers={"model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"}},
        fp8_config=fp8,
        nvfp4_config=nvfp4,
        w4a16_nvfp4_config=w4a16,
    )


def _moe_contract(**overrides):
    values = {
        "num_experts": 256,
        "experts_per_token": 8,
        "hidden_dim": 2048,
        "intermediate_size_per_partition": 128,
        "tp_size": 4,
        "moe_parallel_config": SimpleNamespace(use_all2all_kernels=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mixed_min_capability_requires_exact_sm70_and_both_turbomind_routes():
    with (
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "use_turbomind", side_effect=[True, True]),
    ):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 70

    with (
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "use_turbomind", side_effect=[True, False]),
    ):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 89

    with patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=False):
        assert ModelOptMixedPrecisionConfig.get_min_capability() == 89


def test_nvfp4_grouped_prefill_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", raising=False)
    assert envs.VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL

    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL", "0")
    assert not envs.VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_experts", 128),
        ("experts_per_token", 6),
        ("hidden_dim", 4096),
        ("intermediate_size_per_partition", 96),
        ("tp_size", 2),
    ],
)
def test_nvfp4_moe_contract_rejects_non_qwen36_shapes(field, value):
    validate_nvfp4_sm70_moe_contract(_moe_contract())
    with pytest.raises(NotImplementedError):
        validate_nvfp4_sm70_moe_contract(_moe_contract(**{field: value}))


def test_nvfp4_moe_contract_rejects_shape_consistent_unvalidated_tp8():
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        validate_nvfp4_sm70_moe_contract(
            _moe_contract(tp_size=8, intermediate_size_per_partition=64)
        )


def _meta_layer() -> SimpleNamespace:
    experts, hidden, intermediate = 256, 2048, 128
    return SimpleNamespace(
        local_num_experts=experts,
        moe_config=_moe_contract(),
        w13_weight=torch.empty(
            experts, 2 * intermediate, hidden // 2, dtype=torch.uint8, device="meta"
        ),
        w13_weight_scale=torch.empty(
            experts,
            2 * intermediate,
            hidden // 16,
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        w13_weight_scale_2=torch.empty(experts, 2, device="meta"),
        w2_weight=torch.empty(
            experts, hidden, intermediate // 2, dtype=torch.uint8, device="meta"
        ),
        w2_weight_scale=torch.empty(
            experts,
            hidden,
            intermediate // 16,
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        w2_weight_scale_2=torch.empty(experts, device="meta"),
    )


def test_nvfp4_moe_weight_layout_is_explicit():
    layer = _meta_layer()
    _validate_weight_layout(layer)

    layer.w2_weight = torch.empty(256, 2048, 32, dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="w2_weight"):
        _validate_weight_layout(layer)


def test_nvfp4_sm70_moe_owns_routing_without_generic_modular_wrapper():
    method = ModelOptNvFp4SM70MoEMethod(
        quant_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
        ),
        moe_config=_moe_contract(),
    )

    assert method.maybe_make_prepare_finalize() is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an exact SM70 CUDA device",
)
def test_nvfp4_compact_groups_keep_duplicate_expert_slots_independent():
    sorted_expert_ids = torch.tensor(
        [3, 3, 3, 17, 17, 42, 88, 88], dtype=torch.int32, device="cuda"
    )
    compact_offsets = torch.empty(9, dtype=torch.int32, device="cuda")
    active_expert_ids = torch.empty(8, dtype=torch.int32, device="cuda")

    _prepare_compact_slot_groups(sorted_expert_ids, compact_offsets, active_expert_ids)

    assert torch.equal(compact_offsets.cpu(), torch.arange(9, dtype=torch.int32))
    assert torch.equal(active_expert_ids.cpu(), sorted_expert_ids.cpu())


def test_mixed_w4a16_moe_requires_turbomind_on_sm70():
    config = _mixed_config()

    class FakeRoutedExperts:
        moe_config = _moe_contract()

    with (
        patch.object(modelopt, "RoutedExperts", FakeRoutedExperts),
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "should_use_nvfp4_moe_turbomind", return_value=False),
        pytest.raises(NotImplementedError, match="TurboMind"),
    ):
        config.get_quant_method(FakeRoutedExperts(), "model.layers.0.mlp.experts")
