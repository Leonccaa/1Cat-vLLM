# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _effective_group_size,
    _repeat_awq_groups,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_qwen38_flash_next_tp4_expands_g128_to_g32() -> None:
    assert _effective_group_size(128, 2560, 640 // 4) == 32


def test_qwen38_flash_next_tp4_keeps_native_g32() -> None:
    assert _effective_group_size(32, 2560, 640 // 4) == 32


def test_aligned_partition_keeps_checkpoint_group_size() -> None:
    assert _effective_group_size(128, 2560, 1024 // 4) == 128


@pytest.mark.parametrize("name", ["w13_scales", "w2_qzeros"])
def test_group_parameters_repeat_without_changing_values(name: str) -> None:
    source = torch.arange(6).reshape(2, 3)

    actual = _repeat_awq_groups(source, name, repeat_factor=4)

    assert actual.shape == (8, 3)
    torch.testing.assert_close(actual, source.repeat_interleave(4, dim=0))


def test_fused_expert_group_parameters_repeat_on_group_axis() -> None:
    source = torch.arange(12).reshape(2, 2, 3)

    actual = _repeat_awq_groups(source, "w2_scales", repeat_factor=2)

    assert actual.shape == (2, 4, 3)
    torch.testing.assert_close(actual, source.repeat_interleave(2, dim=1))


def test_qweight_is_not_repeated() -> None:
    source = torch.arange(6).reshape(2, 3)

    assert _repeat_awq_groups(source, "w2_qweight", repeat_factor=4) is source
