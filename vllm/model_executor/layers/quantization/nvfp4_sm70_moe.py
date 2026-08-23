# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 TurboMind NVFP4 MoE for Qwen3.6-35B-A3B.

The route keeps ModelOpt W4A16_NVFP4 expert weights packed. It combines the
checkpoint's FP8 block scales with its explicit ModelOpt global scales once at
load time, repacks both tensors for TurboMind, and never materializes an FP16
expert-weight copy.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.nn import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    is_exact_sm70_cuda,
    unpack_mxfp4_weight,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_QWEN36_HIDDEN_SIZE: Final = 2048
_QWEN36_INTERMEDIATE_SIZE: Final = 512
_QWEN36_NUM_EXPERTS: Final = 256
_QWEN36_TOP_K: Final = 8
_QWEN36_SUPPORTED_TP_SIZES: Final = (1, 2, 4)
_GRAPH_SAFE_MAX_TOKENS: Final = 18
_COMPACT_GROUPED_MAX_TOKENS: Final = 8


@triton.jit
def _prepare_compact_slot_groups_kernel(
    sorted_expert_ids_ptr,
    compact_offsets_ptr,
    active_expert_ids_ptr,
    TOTAL_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < TOTAL_SLOTS
    expert_ids = tl.load(
        sorted_expert_ids_ptr + offsets,
        mask=valid,
        other=-1,
    )
    tl.store(
        compact_offsets_ptr + offsets,
        offsets,
        mask=offsets <= TOTAL_SLOTS,
    )
    tl.store(
        active_expert_ids_ptr + offsets,
        expert_ids,
        mask=valid,
    )


def _prepare_compact_slot_groups(
    sorted_expert_ids: torch.Tensor,
    compact_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
) -> None:
    total_slots = sorted_expert_ids.numel()
    max_slots = _COMPACT_GROUPED_MAX_TOKENS * _QWEN36_TOP_K
    if not (0 < total_slots <= max_slots):
        raise ValueError(f"Unsupported SM70 NVFP4 active-expert slots: {total_slots}")
    block = triton.next_power_of_2(total_slots + 1)
    # TurboMind's compact grouped dispatch forces one row per group. Keep each
    # routed slot independent even when adjacent slots select the same expert;
    # coalescing duplicate expert IDs would make the forced one-row scheduler
    # silently skip or miscompute the additional rows.
    _prepare_compact_slot_groups_kernel[(1,)](
        sorted_expert_ids,
        compact_offsets,
        active_expert_ids,
        TOTAL_SLOTS=total_slots,
        BLOCK=block,
        num_warps=1,
    )


def validate_nvfp4_sm70_moe_contract(moe: FusedMoEConfig) -> None:
    """Reject every model or topology outside the Qwen3.6 SM70 contract."""
    if moe.num_experts != _QWEN36_NUM_EXPERTS:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports Qwen3.6-35B-A3B "
            f"with {_QWEN36_NUM_EXPERTS} experts, got {moe.num_experts}."
        )
    if moe.experts_per_token != _QWEN36_TOP_K:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports top-k="
            f"{_QWEN36_TOP_K}, got {moe.experts_per_token}."
        )
    if moe.hidden_dim != _QWEN36_HIDDEN_SIZE:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports hidden size "
            f"{_QWEN36_HIDDEN_SIZE}, got {moe.hidden_dim}."
        )
    if moe.tp_size not in _QWEN36_SUPPORTED_TP_SIZES:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports tensor parallel "
            f"sizes {_QWEN36_SUPPORTED_TP_SIZES}, got {moe.tp_size}."
        )
    local_intermediate = moe.intermediate_size_per_partition
    if local_intermediate <= 0 or local_intermediate % NVFP4_GROUP_SIZE:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE requires a positive local intermediate "
            f"size divisible by {NVFP4_GROUP_SIZE}, got {local_intermediate}."
        )
    if local_intermediate * max(moe.tp_size, 1) != _QWEN36_INTERMEDIATE_SIZE:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE currently supports Qwen3.6 expert "
            f"intermediate size {_QWEN36_INTERMEDIATE_SIZE}; got local="
            f"{local_intermediate}, tp_size={moe.tp_size}."
        )
    if moe.moe_parallel_config.use_all2all_kernels:
        raise NotImplementedError(
            "SM70 TurboMind NVFP4 MoE does not support DP+EP all-to-all."
        )


def _validate_weight_layout(layer: RoutedExperts) -> None:
    local_experts = int(layer.local_num_experts)
    hidden = int(layer.moe_config.hidden_dim)
    intermediate = int(layer.moe_config.intermediate_size_per_partition)
    expected = {
        "w13_weight": (local_experts, 2 * intermediate, hidden // 2),
        "w13_weight_scale": (
            local_experts,
            2 * intermediate,
            hidden // NVFP4_GROUP_SIZE,
        ),
        "w13_weight_scale_2": (local_experts, 2),
        "w2_weight": (local_experts, hidden, intermediate // 2),
        "w2_weight_scale": (
            local_experts,
            hidden,
            intermediate // NVFP4_GROUP_SIZE,
        ),
        "w2_weight_scale_2": (local_experts,),
    }
    tensors = {name: getattr(layer, name) for name in expected}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f"SM70 NVFP4 MoE layout mismatch for {name}: "
                f"expected {shape}, got {tuple(tensors[name].shape)}."
            )
    if layer.w13_weight.dtype != torch.uint8 or layer.w2_weight.dtype != torch.uint8:
        raise TypeError("SM70 NVFP4 MoE requires packed uint8 expert weights.")
    if (
        layer.w13_weight_scale.dtype != torch.float8_e4m3fn
        or layer.w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("SM70 NVFP4 MoE requires FP8 E4M3 block scales.")


class ModelOptNvFp4SM70MoEMethod(ModelOptNvFp4FusedMoE):
    """Qwen3.6 ModelOpt W4A16_NVFP4 experts on native TurboMind SM70."""

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        FusedMoEMethodBase.__init__(self, moe_config)
        if quant_config.quant_method != "W4A16_NVFP4":
            raise NotImplementedError(
                "SM70 TurboMind ModelOpt NVFP4 MoE currently requires "
                "W4A16_NVFP4 checkpoint weights."
            )
        self.quant_config = quant_config
        self.use_a16 = True
        self.use_global_sf = False
        validate_nvfp4_sm70_moe_contract(moe_config)

    @property
    def supports_eplb(self) -> bool:
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        # This method owns routing, TurboMind expert GEMMs, and unpermutation.
        # Do not wrap it in the generic ModelOpt modular-kernel path.
        del routing_tables
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        required_ops = (
            "nvfp4_sm70_prepare",
            "nvfp4_moe_dense_stage_sm70_out",
            "awq_moe_build_strided_ptrs",
        )
        missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if missing:
            raise RuntimeError(
                "Qwen3.6 NVFP4 MoE on SM70 requires the TurboMind extension "
                "with " + ", ".join(missing) + "."
            )
        if not hasattr(torch.ops._moe_C, "moe_permute_with_scratch"):
            raise RuntimeError(
                "Qwen3.6 NVFP4 MoE on SM70 requires graph-safe MoE permute ops."
            )
        if self.moe.has_bias:
            raise NotImplementedError("SM70 NVFP4 MoE does not support expert bias.")
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently supports SwiGLU/SILU only."
            )
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "SM70 NVFP4 MoE does not support router weights on input."
            )
        if layer.expert_map is not None:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently requires fully replicated experts."
            )
        if layer.local_num_experts != layer.global_num_experts:
            raise NotImplementedError(
                "SM70 NVFP4 MoE currently requires local and global experts to match."
            )

        validate_nvfp4_sm70_moe_contract(layer.moe_config)
        _validate_weight_layout(layer)
        num_experts = int(layer.local_num_experts)
        hidden = int(layer.moe_config.hidden_dim)
        intermediate = int(layer.moe_config.intermediate_size_per_partition)

        w13_tm_weights: list[torch.Tensor] = []
        w13_tm_scales: list[torch.Tensor] = []
        w13_meta: list[torch.Tensor] = []
        w2_tm_weights: list[torch.Tensor] = []
        w2_tm_scales: list[torch.Tensor] = []
        w2_meta: list[torch.Tensor] = []
        for expert_id in range(num_experts):
            w13_packed = unpack_mxfp4_weight(layer.w13_weight[expert_id].data)
            w13_scales = layer.w13_weight_scale[expert_id].float().clone()
            w13_global = layer.w13_weight_scale_2[expert_id].float()
            w13_scales[:intermediate].mul_(w13_global[0])
            w13_scales[intermediate:].mul_(w13_global[1])
            prepared_w13 = sm70_ops.nvfp4_sm70_prepare(
                w13_packed,
                w13_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
            )
            w13_tm_weights.append(prepared_w13[0])
            w13_tm_scales.append(prepared_w13[1])
            w13_meta.append(prepared_w13[2])

            w2_packed = unpack_mxfp4_weight(layer.w2_weight[expert_id].data)
            w2_scales = (
                layer.w2_weight_scale[expert_id].float()
                * layer.w2_weight_scale_2[expert_id].float()
            )
            prepared_w2 = sm70_ops.nvfp4_sm70_prepare(
                w2_packed,
                w2_scales.half().t().contiguous(),
                NVFP4_GROUP_SIZE,
            )
            w2_tm_weights.append(prepared_w2[0])
            w2_tm_scales.append(prepared_w2[1])
            w2_meta.append(prepared_w2[2])

        layer.w13_tm_weight = Parameter(
            torch.stack(w13_tm_weights), requires_grad=False
        )
        layer.w13_tm_scales = Parameter(torch.stack(w13_tm_scales), requires_grad=False)
        layer.w2_tm_weight = Parameter(torch.stack(w2_tm_weights), requires_grad=False)
        layer.w2_tm_scales = Parameter(torch.stack(w2_tm_scales), requires_grad=False)

        w13_k_ld = int(w13_meta[0][0].item())
        w13_q_ld = int(w13_meta[0][1].item())
        w2_k_ld = int(w2_meta[0][0].item())
        w2_q_ld = int(w2_meta[0][1].item())
        w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w13_tm_weight,
            layer.w13_tm_scales,
            w13_k_ld,
            w13_q_ld,
            num_experts,
        )
        w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
            layer.w2_tm_weight,
            layer.w2_tm_scales,
            w2_k_ld,
            w2_q_ld,
            num_experts,
        )
        layer.w13_strided_ptrs_w = Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = Parameter(w2_ptrs[1], requires_grad=False)

        layer.sm70_nvfp4_moe = True
        layer.sm70_nvfp4_num_experts = num_experts
        layer.sm70_nvfp4_hidden_size = hidden
        layer.sm70_nvfp4_intermediate_size = intermediate
        layer.sm70_nvfp4_w13_k_dim = hidden
        layer.sm70_nvfp4_w13_n_dim = 2 * intermediate
        layer.sm70_nvfp4_w2_k_dim = intermediate
        layer.sm70_nvfp4_w2_n_dim = hidden
        layer.sm70_nvfp4_group_size = NVFP4_GROUP_SIZE
        layer.sm70_nvfp4_graph_safe_max_tokens = _GRAPH_SAFE_MAX_TOKENS
        self._allocate_graph_safe_decode_buffers(layer)

        del layer.w13_weight
        del layer.w13_weight_scale
        del layer.w13_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_weight
        del layer.w2_weight_scale
        del layer.w2_weight_scale_2
        del layer.w2_input_scale
        logger.info_once(
            "SM70 ModelOpt NVFP4 TurboMind MoE path enabled for "
            "Qwen3.6-35B-A3B (local_experts=%d, graph_safe_decode=B1-B%d, "
            "compact_grouped_decode=B1-B%d).",
            num_experts,
            _GRAPH_SAFE_MAX_TOKENS,
            _COMPACT_GROUPED_MAX_TOKENS,
        )

    def _allocate_graph_safe_decode_buffers(self, layer: RoutedExperts) -> None:
        device = layer.w13_tm_weight.device
        max_slots = _GRAPH_SAFE_MAX_TOKENS * _QWEN36_TOP_K
        experts = int(layer.sm70_nvfp4_num_experts)
        hidden = int(layer.sm70_nvfp4_hidden_size)
        intermediate = int(layer.sm70_nvfp4_intermediate_size)

        layer._nvfp4_sm70_output = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_permuted_input = torch.empty(
            max_slots, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_gate_up = torch.empty(
            max_slots, 2 * intermediate, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_intermediate = torch.empty(
            max_slots, intermediate, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_sorted_output = torch.empty(
            max_slots, hidden, dtype=torch.float16, device=device
        )
        layer._nvfp4_sm70_expert_offsets = torch.empty(
            experts + 1, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_expert_offsets64 = torch.empty(
            experts + 1, dtype=torch.int64, device=device
        )
        layer._nvfp4_sm70_inv_permuted_idx = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS,
            _QWEN36_TOP_K,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_topk_ids = torch.empty(
            _GRAPH_SAFE_MAX_TOKENS,
            _QWEN36_TOP_K,
            dtype=torch.int32,
            device=device,
        )
        layer._nvfp4_sm70_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(_GRAPH_SAFE_MAX_TOKENS, _QWEN36_TOP_K)
        layer._nvfp4_sm70_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_permuted_experts_id = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_sorted_row_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_topk_ids_for_sort = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            max_slots, layer.global_num_experts
        )
        layer._nvfp4_sm70_sort_workspace = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
        layer._nvfp4_sm70_dense_expert_ids = torch.arange(
            experts, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_compact_offsets = torch.empty(
            max_slots + 1, dtype=torch.int32, device=device
        )
        layer._nvfp4_sm70_active_expert_ids = torch.empty(
            max_slots, dtype=torch.int32, device=device
        )

    @staticmethod
    def _persistent_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        slots = num_tokens * _QWEN36_TOP_K
        return {
            "output": layer._nvfp4_sm70_output[:num_tokens],
            "permuted_input": layer._nvfp4_sm70_permuted_input[:slots],
            "gate_up": layer._nvfp4_sm70_gate_up[:slots],
            "intermediate": layer._nvfp4_sm70_intermediate[:slots],
            "sorted_output": layer._nvfp4_sm70_sorted_output[:slots],
            "expert_offsets": layer._nvfp4_sm70_expert_offsets,
            "expert_offsets64": layer._nvfp4_sm70_expert_offsets64,
            "inv_permuted_idx": layer._nvfp4_sm70_inv_permuted_idx[:num_tokens],
            "topk_ids": layer._nvfp4_sm70_topk_ids[:num_tokens],
            "token_expert_indices": (
                layer._nvfp4_sm70_token_expert_indices[:num_tokens]
            ),
            "permuted_idx": layer._nvfp4_sm70_permuted_idx[:slots],
            "sort_workspace": layer._nvfp4_sm70_sort_workspace,
            "permuted_experts_id": layer._nvfp4_sm70_permuted_experts_id[:slots],
            "sorted_row_idx": layer._nvfp4_sm70_sorted_row_idx[:slots],
            "topk_ids_for_sort": layer._nvfp4_sm70_topk_ids_for_sort[:slots],
            "dense_expert_ids": layer._nvfp4_sm70_dense_expert_ids,
            "compact_offsets": layer._nvfp4_sm70_compact_offsets[: slots + 1],
            "active_expert_ids": layer._nvfp4_sm70_active_expert_ids[:slots],
        }

    @staticmethod
    def _eager_buffers(
        layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        device = layer.w13_tm_weight.device
        slots = num_tokens * _QWEN36_TOP_K
        experts = int(layer.sm70_nvfp4_num_experts)
        hidden = int(layer.sm70_nvfp4_hidden_size)
        intermediate = int(layer.sm70_nvfp4_intermediate_size)
        workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
            slots, layer.global_num_experts
        )
        return {
            "output": torch.empty(
                num_tokens, hidden, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                slots, hidden, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                slots, 2 * intermediate, dtype=torch.float16, device=device
            ),
            "intermediate": torch.empty(
                slots, intermediate, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                slots, hidden, dtype=torch.float16, device=device
            ),
            "expert_offsets": torch.empty(
                experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, _QWEN36_TOP_K, dtype=torch.int32, device=device
            ),
            "topk_ids": torch.empty(
                num_tokens, _QWEN36_TOP_K, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                slots, dtype=torch.int32, device=device
            ).view(num_tokens, _QWEN36_TOP_K),
            "permuted_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "sort_workspace": torch.empty(
                workspace_size, dtype=torch.int8, device=device
            ),
            "permuted_experts_id": torch.empty(slots, dtype=torch.int32, device=device),
            "sorted_row_idx": torch.empty(slots, dtype=torch.int32, device=device),
            "topk_ids_for_sort": torch.empty(slots, dtype=torch.int32, device=device),
            "dense_expert_ids": layer._nvfp4_sm70_dense_expert_ids,
            "compact_offsets": torch.empty(slots + 1, dtype=torch.int32, device=device),
            "active_expert_ids": torch.empty(slots, dtype=torch.int32, device=device),
        }

    def _get_buffers(
        self, layer: RoutedExperts, num_tokens: int
    ) -> dict[str, torch.Tensor]:
        if 0 < num_tokens <= _GRAPH_SAFE_MAX_TOKENS:
            return self._persistent_buffers(layer, num_tokens)
        return self._eager_buffers(layer, num_tokens)

    @staticmethod
    def _apply_swiglu(
        layer: RoutedExperts, out: torch.Tensor, gate_up: torch.Tensor
    ) -> None:
        if layer.swiglu_limit is None:
            torch.ops._C.silu_and_mul(out, gate_up)
        else:
            torch.ops._C.silu_and_mul_with_clamp(
                out, gate_up, float(layer.swiglu_limit)
            )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
            raise TypeError("SM70 NVFP4 MoE requires CUDA FP16 activations [M, H].")
        if not is_exact_sm70_cuda(x, enabled=True):
            raise RuntimeError("SM70 NVFP4 MoE dispatch is restricted to CUDA SM70.")
        if x.shape[1] != _QWEN36_HIDDEN_SIZE:
            raise ValueError(
                "SM70 NVFP4 MoE activation hidden size mismatch: expected "
                f"{_QWEN36_HIDDEN_SIZE}, got {x.shape[1]}."
            )
        if tuple(topk_ids.shape) != (x.shape[0], _QWEN36_TOP_K):
            raise ValueError("SM70 NVFP4 MoE requires top-k IDs with shape [M, 8].")
        if tuple(topk_weights.shape) != tuple(topk_ids.shape):
            raise ValueError("SM70 NVFP4 MoE top-k weights and IDs must share shape.")
        if topk_weights.dtype != torch.float32:
            raise TypeError("SM70 NVFP4 MoE requires float32 top-k weights.")

        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, _QWEN36_HIDDEN_SIZE))
        buffers = self._get_buffers(layer, num_tokens)
        output = buffers["output"]
        output.zero_()
        slots = num_tokens * _QWEN36_TOP_K
        topk_ids_i32 = buffers["topk_ids"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        buffers["permuted_idx"].fill_(slots)
        torch.ops._moe_C.moe_permute_with_scratch(
            x,
            topk_ids_i32,
            buffers["token_expert_indices"],
            layer.expert_map,
            layer.global_num_experts,
            layer.local_num_experts,
            _QWEN36_TOP_K,
            buffers["permuted_input"],
            buffers["expert_offsets64"],
            buffers["inv_permuted_idx"],
            buffers["permuted_idx"],
            buffers["sort_workspace"],
            buffers["permuted_experts_id"],
            buffers["sorted_row_idx"],
            buffers["topk_ids_for_sort"],
        )
        buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        if num_tokens <= _COMPACT_GROUPED_MAX_TOKENS:
            _prepare_compact_slot_groups(
                buffers["permuted_experts_id"],
                buffers["compact_offsets"],
                buffers["active_expert_ids"],
            )
            stage_offsets = buffers["compact_offsets"]
            stage_expert_ids = buffers["active_expert_ids"]
            stage_experts = slots
        else:
            stage_offsets = buffers["expert_offsets"]
            stage_expert_ids = buffers["dense_expert_ids"]
            stage_experts = int(layer.sm70_nvfp4_num_experts)

        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            buffers["gate_up"],
            buffers["permuted_input"],
            stage_offsets,
            stage_expert_ids,
            layer.w13_strided_ptrs_w,
            layer.w13_strided_ptrs_s,
            stage_experts,
            layer.sm70_nvfp4_w13_k_dim,
            layer.sm70_nvfp4_w13_n_dim,
            layer.sm70_nvfp4_group_size,
        )
        self._apply_swiglu(layer, buffers["intermediate"], buffers["gate_up"])
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            buffers["sorted_output"],
            buffers["intermediate"],
            stage_offsets,
            stage_expert_ids,
            layer.w2_strided_ptrs_w,
            layer.w2_strided_ptrs_s,
            stage_experts,
            layer.sm70_nvfp4_w2_k_dim,
            layer.sm70_nvfp4_w2_n_dim,
            layer.sm70_nvfp4_group_size,
        )
        torch.ops._moe_C.moe_unpermute(
            buffers["sorted_output"],
            topk_weights,
            buffers["inv_permuted_idx"],
            buffers["expert_offsets64"],
            _QWEN36_TOP_K,
            output,
        )
        return output

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, x, router_logits, input_ids
        raise NotImplementedError("SM70 NVFP4 MoE is not a monolithic route.")

    def get_fused_moe_quant_config(  # type: ignore[override]
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None
