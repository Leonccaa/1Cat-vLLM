# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM70 small-M FP16 skinny GEMM for Qwen3.8 dense projections.

``VLLM_SM70_FP16_SKINNY=1`` routes checkpoint-FP16 dense projections whose
row count is in ``[VLLM_SM70_FP16_SKINNY_MIN_M, VLLM_SM70_FP16_SKINNY_MAX_M]``
(default 17..48: MTP4 verifier widths above the M <= 16 lanes) to
``vllm.model_executor.kernels.linear.sm70_fp16_skinny``. Two hooks:

* projections that already run inside an opaque SM70 Qwen3.8 op (FP16 GEMV
  role projections, fused GDN input, fused HC, shared-expert up) call the
  kernel in place of their ``F.linear`` fallback; every M <= 16 lane keeps
  priority;
* remaining plain ``UnquantizedLinearMethod`` projections in the role list
  below get :class:`Sm70Fp16SkinnyLinearMethod`, whose decode-graph ``apply``
  is one opaque op that falls back to ``F.linear`` outside the row range.

The LM head is separately opt-in (``VLLM_SM70_FP16_SKINNY_LM_HEAD=1``). With
the umbrella unset nothing is replaced and every path is unchanged. TP
semantics are unchanged: the kernel is a per-rank GEMM and reductions stay in
the layers.
"""

from __future__ import annotations

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import sm70_fp16_skinny as skinny
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Checkpoint-FP16 dense projections of the Qwen3.8 target and MTP draft.
SKINNY_ROLE_SUFFIXES: tuple[str, ...] = (
    ".linear_attn.in_proj_qkvz",
    ".linear_attn.in_proj_ba",
    ".linear_attn.out_proj",
    ".self_attn.qkv_proj",
    ".self_attn.o_proj",
    ".self_attn.indexer.index_qk_proj",
    ".mlp.gate",
    ".mlp.shared_expert.gate_up_proj",
    ".mlp.shared_expert.down_proj",
    ".mlp.shared_expert_gate",
    ".input_mix_weight_down_block_inject",
    ".input_mix_weight_down",
    ".input_mix_weight_up",
    ".fc_embedding",
    ".fc_hidden",
    ".ple.key_proj",
    ".ple.value_proj",
)


class Sm70Fp16SkinnyLinearMethod(UnquantizedLinearMethod):
    """Unquantized linear whose decode-graph projection may use the kernel."""

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            use_sm70_decode_graph_semantics()
            and not envs.VLLM_BATCH_INVARIANT
            # A prepared TurboMind/online-QPN8 layer keeps its own route.
            and not getattr(layer, "_sm70_f16_prepared", False)
            and not getattr(layer, "_sm70_qwen4_exp_online_qpn8", False)
        ):
            return torch.ops.vllm.sm70_fp16_skinny_linear(x, layer.weight, bias)
        return super().apply(layer, x, bias)


def is_skinny_role(prefix: str, shape: tuple[int, int]) -> bool:
    return bool(
        prefix.endswith(SKINNY_ROLE_SUFFIXES) and min(shape) > 0 and shape[1] % 16 == 0
    )


def enable_sm70_fp16_skinny(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> int:
    """Mark admitted projections before checkpoint loading; returns a count."""
    del vllm_config
    if not envs.VLLM_SM70_FP16_SKINNY:
        return 0
    if (
        dtype != torch.float16
        or not current_platform.is_device_capability((7, 0))
        or envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
    ):
        logger.warning_once(
            "SM70 FP16 skinny GEMM opt-in rejected: dtype=%s sm70=%s online_qpn8=%s.",
            dtype,
            current_platform.is_device_capability((7, 0)),
            envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8,
        )
        return 0
    replaced = 0
    lm_heads = 0
    for child in module.modules():
        if isinstance(child, ParallelLMHead):
            if envs.VLLM_SM70_FP16_SKINNY_LM_HEAD:
                # Read by UnquantizedEmbeddingMethod.apply after the existing
                # SM70 LM-head fast paths.
                child._sm70_fp16_skinny_lm_head = True
                lm_heads += 1
            continue
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        if not is_skinny_role(str(getattr(child, "prefix", "")), shape):
            continue
        child.quant_method = Sm70Fp16SkinnyLinearMethod()
        replaced += 1
    lo, hi = skinny.row_range()
    logger.info_once(
        "SM70 FP16 skinny GEMM enabled for M %d..%d: %d plain projections "
        "(plus FP16 lane fallbacks), %d LM heads.",
        lo,
        hi,
        replaced,
        lm_heads,
    )
    return replaced


__all__ = [
    "SKINNY_ROLE_SUFFIXES",
    "Sm70Fp16SkinnyLinearMethod",
    "enable_sm70_fp16_skinny",
    "is_skinny_role",
]
