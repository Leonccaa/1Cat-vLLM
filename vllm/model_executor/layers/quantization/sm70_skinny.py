# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

AWQ_GROUP_SIZE = 128
_SELF_CHECK_TOL = 3e-2
_AWQ_REVERSE_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_QPN_K_ORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)

_awq_route_log_seen: set[tuple[str, int, torch.dtype]] = set()


def selected_base_backend(default_auto: str = "turbomind") -> str:
    """Resolve the base backend without conflating it with the overlay.

    ``VLLM_SM70_QUANT_BACKEND`` remains the SM70-specific base selector. When
    it is ``auto`` (or the legacy ``skinny`` alias), an explicit generic
    ``--linear-backend`` remains authoritative; otherwise the caller's
    already-validated V100 default is used.
    """
    requested = envs.get_sm70_quant_base_backend()
    if requested != "auto":
        return requested

    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    linear_backend = (
        config.kernel_config.linear_backend if config is not None else "auto"
    )
    return default_auto if linear_backend == "auto" else linear_backend


def missing_awq_ops() -> list[str]:
    """Return Skinny AWQ ops required by the selected resident layout."""
    layout = envs.get_sm70_skinny_awq_layout()
    names: list[str] = []
    if layout in ("simt", "both"):
        names.append("skinny_awq_gemm_simt")
    if layout in ("qpn", "both"):
        names.append("skinny_awq_gemm_qpn")
    return [name for name in names if not hasattr(torch.ops._C, name)]


@dataclass
class SM70SkinnyAwqState:
    codes: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    qpn_codes: torch.Tensor
    qpn_scales: torch.Tensor
    qpn_biases: torch.Tensor
    group_size: int
    input_size: int
    output_size: int
    disabled_routes: set[str]
    validated_routes: set[tuple[str, str]]

    @property
    def has_simt(self) -> bool:
        return self.codes.numel() != 0

    @property
    def has_qpn(self) -> bool:
        return self.qpn_codes.numel() != 0

    @property
    def shape_key(self) -> tuple[int, int, bool, bool]:
        return (self.output_size, self.input_size, self.has_simt, self.has_qpn)


def _unpack_awq_rows(packed: torch.Tensor) -> torch.Tensor:
    """Return logical uint4 rows from AutoAWQ's interleaved int32 layout."""
    if packed.dtype != torch.int32 or packed.dim() != 2:
        raise TypeError(
            "SM70 Skinny AWQ prepack expects a two-dimensional int32 tensor."
        )
    byte_view = packed.contiguous().view(torch.uint8)
    nibbles = torch.stack((byte_view & 0xF, byte_view >> 4), dim=-1)
    nibbles = nibbles.view(packed.shape[0], -1, 8)
    order = torch.tensor(_AWQ_REVERSE_PACK_ORDER, device=packed.device)
    return nibbles.index_select(-1, order).reshape(packed.shape[0], -1)


def unpack_awq_dense(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert native AWQ tensors into logical N-major codes/metadata."""
    if group_size != AWQ_GROUP_SIZE:
        raise ValueError(
            f"SM70 Skinny AWQ supports group_size={AWQ_GROUP_SIZE}, got {group_size}."
        )
    if scales.dtype != torch.float16 or scales.dim() != 2:
        raise TypeError("SM70 Skinny AWQ prepack expects two-dimensional FP16 scales.")
    if qzeros.dtype != torch.int32 or qzeros.dim() != 2:
        raise TypeError("SM70 Skinny AWQ prepack expects two-dimensional int32 zeros.")
    if qweight.device != scales.device or qweight.device != qzeros.device:
        raise ValueError("SM70 Skinny AWQ tensors must be on the same device.")

    k = qweight.shape[0]
    n = qweight.shape[1] * 8
    groups = k // group_size
    if k % group_size or scales.shape != (groups, n):
        raise ValueError("SM70 Skinny AWQ scale shape does not match qweight.")
    if qzeros.shape != (groups, n // 8):
        raise ValueError("SM70 Skinny AWQ zero shape does not match qweight.")

    logical_weight = _unpack_awq_rows(qweight).t().contiguous()
    codes = (logical_weight[:, 0::2] | (logical_weight[:, 1::2] << 4)).contiguous()
    logical_scales = scales.t().contiguous()
    logical_zeros = _unpack_awq_rows(qzeros).t().to(torch.float16).contiguous()
    biases = (-logical_zeros * logical_scales).contiguous()
    return codes, logical_scales, biases


def qpn_prepack_awq(
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Prearrange AWQ codes and group-128 metadata as N32/K16 fragments."""
    if codes.dtype != torch.uint8 or codes.dim() != 2:
        raise TypeError("SM70 Skinny AWQ QPN prepack expects uint8 [N,K/2] codes.")
    if scales.dtype != torch.float16 or biases.dtype != torch.float16:
        raise TypeError("SM70 Skinny AWQ QPN prepack expects FP16 metadata.")
    if scales.shape != biases.shape or scales.dim() != 2:
        raise ValueError("SM70 Skinny AWQ QPN scale/bias shape mismatch.")
    if codes.device != scales.device or codes.device != biases.device:
        raise ValueError("SM70 Skinny AWQ QPN tensors must be on the same device.")

    n, packed_k = codes.shape
    k = packed_k * 2
    if scales.shape != (n, k // AWQ_GROUP_SIZE):
        raise ValueError("SM70 Skinny AWQ QPN metadata shape mismatch.")
    if n % 32 or k % AWQ_GROUP_SIZE:
        return None, None, None

    device = codes.device
    tiles = n // 32
    k16_groups = k // 16
    metadata_groups = k // AWQ_GROUP_SIZE
    lane = torch.arange(32, device=device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(_QPN_K_ORDER, device=device)
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).view(n, k)
    group = torch.arange(k16_groups, device=device)
    kidx = group.view(k16_groups, 1) * 16 + korder.view(1, 16)
    qcodes = torch.empty((tiles, k16_groups, 32, 8), dtype=torch.uint8, device=device)

    # Bound the broadcast indexing temporaries to roughly 300 MiB.
    tile_chunk = max(1, 36864 // k16_groups)
    for tile_start in range(0, tiles, tile_chunk):
        tile_end = min(tile_start + tile_chunk, tiles)
        tile_count = tile_end - tile_start
        ncol = torch.arange(tile_start, tile_end, device=device).view(
            tile_count, 1
        ) * 32 + col.view(1, 32)
        selected = nibbles[
            ncol.view(tile_count, 1, 32, 1).expand(tile_count, k16_groups, 32, 16),
            kidx.view(1, k16_groups, 1, 16).expand(tile_count, k16_groups, 32, 16),
        ]
        qcodes[tile_start:tile_end] = selected[..., 0::2] | (selected[..., 1::2] << 4)

    ncol = torch.arange(tiles, device=device).view(tiles, 1) * 32 + col.view(1, 32)
    metadata_index = torch.arange(metadata_groups, device=device)
    qscales = scales[
        ncol.view(tiles, 1, 32).expand(tiles, metadata_groups, 32),
        metadata_index.view(1, metadata_groups, 1).expand(tiles, metadata_groups, 32),
    ]
    qbiases = biases[
        ncol.view(tiles, 1, 32).expand(tiles, metadata_groups, 32),
        metadata_index.view(1, metadata_groups, 1).expand(tiles, metadata_groups, 32),
    ]
    return (
        qcodes.view(-1).contiguous(),
        qscales.contiguous().view(-1),
        qbiases.contiguous().view(-1),
    )


def prepare_awq_state(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> SM70SkinnyAwqState:
    codes, logical_scales, biases = unpack_awq_dense(
        qweight, scales, qzeros, group_size
    )
    layout = envs.get_sm70_skinny_awq_layout()
    empty_codes = codes.new_empty(0)
    empty_meta = logical_scales.new_empty(0)

    if layout in ("qpn", "both"):
        qpn_codes, qpn_scales, qpn_biases = qpn_prepack_awq(
            codes, logical_scales, biases
        )
    else:
        qpn_codes = qpn_scales = qpn_biases = None

    keep_simt = layout in ("simt", "both")
    return SM70SkinnyAwqState(
        codes=codes if keep_simt else empty_codes,
        scales=logical_scales if keep_simt else empty_meta,
        biases=biases if keep_simt else empty_meta,
        qpn_codes=qpn_codes if qpn_codes is not None else empty_codes,
        qpn_scales=qpn_scales if qpn_scales is not None else empty_meta,
        qpn_biases=qpn_biases if qpn_biases is not None else empty_meta,
        group_size=group_size,
        input_size=qweight.shape[0],
        output_size=qweight.shape[1] * 8,
        disabled_routes=set(),
        validated_routes=set(),
    )


def prepare_awq_moe_bank(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one replacement-layout N-major bank for grouped MoE execution."""
    if qweight.dim() != 3 or scales.dim() != 3 or qzeros.dim() != 3:
        raise ValueError("SM70 Skinny AWQ MoE expects expert-major 3D tensors.")
    if not (qweight.shape[0] == scales.shape[0] == qzeros.shape[0]):
        raise ValueError("SM70 Skinny AWQ MoE expert counts do not match.")

    expert_codes: list[torch.Tensor] = []
    expert_scales: list[torch.Tensor] = []
    expert_biases: list[torch.Tensor] = []
    for expert in range(qweight.shape[0]):
        codes, logical_scales, biases = unpack_awq_dense(
            qweight[expert],
            scales[expert].to(torch.float16).contiguous(),
            qzeros[expert],
            group_size,
        )
        expert_codes.append(codes)
        expert_scales.append(logical_scales)
        expert_biases.append(biases)
    return (
        torch.stack(expert_codes).contiguous(),
        torch.stack(expert_scales).contiguous(),
        torch.stack(expert_biases).contiguous(),
    )


def _skinny_awq_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    out = _try_awq_skinny_linear(
        x,
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        n,
        k,
        group_size,
    )
    if out is None:
        raise RuntimeError(
            f"SM70 Skinny AWQ has no route for M={x.shape[0]}, N={n}, K={k}; "
            "use a graph-safe overlay op that owns the selected base fallback."
        )
    return out


def _try_awq_skinny_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor | None:
    """Run a supported Skinny route, or return ``None`` at runtime.

    This helper is called from opaque hybrid custom ops as well as the
    Skinny-only test op.  Keeping the M decision inside an opaque op is
    essential: a Python branch in ``LinearMethod.apply`` can be specialized
    while torch.compile traces M=1 and then be reused by a large-M prefill.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"SM70 Skinny AWQ supports FP16 or BF16 activations, got {x.dtype}."
        )

    output_dtype = x.dtype
    kernel_x = x if x.dtype == torch.float16 else x.to(torch.float16)
    m = kernel_x.shape[0]
    use_qpn = (
        4 <= m <= 16
        and k % AWQ_GROUP_SIZE == 0
        and n % 32 == 0
        and qpn_codes.numel() == n * (k // 2)
        and qpn_scales.numel() == n * (k // AWQ_GROUP_SIZE)
        and qpn_biases.numel() == qpn_scales.numel()
    )
    use_simt = (
        1 <= m <= 3
        and k % AWQ_GROUP_SIZE == 0
        and n % 8 == 0
        and codes.numel() == n * (k // 2)
        and scales.numel() == n * (k // AWQ_GROUP_SIZE)
        and biases.numel() == scales.numel()
    )

    route = "qpn" if use_qpn else "simt" if use_simt else "unsupported"
    # Keep diagnostics out of Dynamo traces. The hybrid custom op may be
    # invoked while a new dtype specialization is being compiled, and Python
    # logger/set side effects would otherwise turn an opaque runtime-M dispatch
    # into a graph break.
    if not torch.compiler.is_compiling():
        route_key = (route, m, output_dtype)
        if route_key not in _awq_route_log_seen:
            _awq_route_log_seen.add(route_key)
            logger.info(
                "SM70 Skinny AWQ route: M=%d N=%d K=%d dtype=%s -> %s",
                m,
                n,
                k,
                output_dtype,
                route,
            )

    from vllm import _sm70_ops as sm70_ops

    if use_qpn:
        out = sm70_ops.skinny_awq_gemm_qpn(
            kernel_x, qpn_codes, qpn_scales, qpn_biases, group_size, n
        )
    elif use_simt:
        out = sm70_ops.skinny_awq_gemm_simt(kernel_x, codes, scales, biases, group_size)
    else:
        return None
    return out if output_dtype == torch.float16 else out.to(output_dtype)


def _skinny_awq_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        k,
        group_size,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_awq_linear",
    op_func=_skinny_awq_linear_impl,
    fake_impl=_skinny_awq_linear_fake,
)


def _skinny_awq_turbomind_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    out = _try_awq_skinny_linear(
        x,
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        n,
        k,
        group_size,
    )
    if out is not None:
        return out

    from vllm import _sm70_ops as sm70_ops

    output_dtype = x.dtype
    kernel_x = x if x.dtype == torch.float16 else x.to(torch.float16)
    out = torch.empty(
        (kernel_x.shape[0], n), dtype=torch.float16, device=kernel_x.device
    )
    sm70_ops.awq_gemm_sm70_out(
        out,
        kernel_x,
        base_weight,
        base_scales,
        group_size,
        k_ld,
        q_ld,
    )
    return out if output_dtype == torch.float16 else out.to(output_dtype)


def _skinny_awq_turbomind_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        base_weight,
        base_scales,
        k,
        group_size,
        k_ld,
        q_ld,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_awq_turbomind_linear",
    op_func=_skinny_awq_turbomind_linear_impl,
    fake_impl=_skinny_awq_turbomind_linear_fake,
)


def _skinny_awq_marlin_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_zeros: torch.Tensor,
    base_g_idx: torch.Tensor,
    base_perm: torch.Tensor,
    base_workspace: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    is_k_full: bool,
) -> torch.Tensor:
    out = _try_awq_skinny_linear(
        x,
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        n,
        k,
        group_size,
    )
    if out is not None:
        return out

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    return apply_gptq_marlin_linear(
        input=x,
        weight=base_weight,
        weight_scale=base_scales,
        weight_zp=base_zeros,
        g_idx=base_g_idx,
        g_idx_sort_indices=base_perm,
        workspace=base_workspace,
        wtype=scalar_types.uint4,
        output_size_per_partition=n,
        input_size_per_partition=k,
        is_k_full=is_k_full,
    )


def _skinny_awq_marlin_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    qpn_biases: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_zeros: torch.Tensor,
    base_g_idx: torch.Tensor,
    base_perm: torch.Tensor,
    base_workspace: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    is_k_full: bool,
) -> torch.Tensor:
    del (
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        base_weight,
        base_scales,
        base_zeros,
        base_g_idx,
        base_perm,
        base_workspace,
        k,
        group_size,
        is_k_full,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_awq_marlin_linear",
    op_func=_skinny_awq_marlin_linear_impl,
    fake_impl=_skinny_awq_marlin_linear_fake,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().abs().max().clamp(min=1e-6)
    error = (actual.float() - expected.float()).abs().max() / denominator
    return float(error.item())


def validate_awq_state(
    state: SM70SkinnyAwqState,
    reference_apply: Callable[[torch.Tensor], torch.Tensor],
    base_backend: str,
) -> None:
    """Validate resident routes against the selected base backend.

    A failure disables only that route on this layer state. The base backend
    remains intact and owns every unsupported or disabled call.
    """

    from vllm import _sm70_ops as sm70_ops

    cases: list[tuple[int, str]] = []
    if state.has_simt:
        cases.append((1, "simt"))
    if state.has_qpn:
        cases.append((8, "qpn"))
    for m, route in cases:
        validation_key = (base_backend, route)
        if validation_key in state.validated_routes or route in state.disabled_routes:
            continue
        try:
            values = torch.arange(
                m * state.input_size,
                device=state.codes.device,
                dtype=torch.int32,
            )
            x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(
                m, state.input_size
            )
            reference = reference_apply(x)
            if route == "simt":
                actual = sm70_ops.skinny_awq_gemm_simt(
                    x,
                    state.codes,
                    state.scales,
                    state.biases,
                    state.group_size,
                )
            else:
                actual = sm70_ops.skinny_awq_gemm_qpn(
                    x,
                    state.qpn_codes,
                    state.qpn_scales,
                    state.qpn_biases,
                    state.group_size,
                    state.output_size,
                )
            relative_error = _relative_error(actual, reference)
            if not math.isfinite(relative_error) or relative_error > _SELF_CHECK_TOL:
                raise RuntimeError(
                    f"{route} relative error {relative_error:.3e} exceeds "
                    f"{_SELF_CHECK_TOL:.3e}"
                )
        except Exception:
            state.disabled_routes.add(route)
            logger.exception(
                "SM70 Skinny AWQ %s self-check failed for N=%d K=%d "
                "against base=%s; disabling only this route.",
                route,
                state.output_size,
                state.input_size,
                base_backend,
            )
            continue
        state.validated_routes.add(validation_key)
        logger.info_once(
            "SM70 Skinny AWQ %s self-check passed for N=%d K=%d against base=%s.",
            route,
            state.output_size,
            state.input_size,
            base_backend,
        )


def select_awq_route(state: SM70SkinnyAwqState, rows: int) -> str | None:
    if 1 <= rows <= 3 and state.has_simt and "simt" not in state.disabled_routes:
        return "simt"
    if 4 <= rows <= 16 and state.has_qpn and "qpn" not in state.disabled_routes:
        return "qpn"
    return None


def try_apply_awq_state(
    state: SM70SkinnyAwqState,
    x: torch.Tensor,
) -> torch.Tensor | None:
    reshaped_x = x.reshape(-1, state.input_size).contiguous()
    if select_awq_route(state, reshaped_x.shape[0]) is None:
        return None
    return torch.ops.vllm.sm70_skinny_awq_linear(
        reshaped_x,
        state.codes,
        state.scales,
        state.biases,
        state.qpn_codes,
        state.qpn_scales,
        state.qpn_biases,
        state.output_size,
        state.input_size,
        state.group_size,
    )


def apply_awq_turbomind_overlay(
    state: SM70SkinnyAwqState,
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    """Graph-safe Skinny overlay with a TurboMind base fallback."""
    reshaped_x = x.reshape(-1, state.input_size).contiguous()
    codes = state.codes if "simt" not in state.disabled_routes else state.codes[:0]
    scales = state.scales if "simt" not in state.disabled_routes else state.scales[:0]
    biases = state.biases if "simt" not in state.disabled_routes else state.biases[:0]
    qpn_codes = (
        state.qpn_codes if "qpn" not in state.disabled_routes else state.qpn_codes[:0]
    )
    qpn_scales = (
        state.qpn_scales if "qpn" not in state.disabled_routes else state.qpn_scales[:0]
    )
    qpn_biases = (
        state.qpn_biases if "qpn" not in state.disabled_routes else state.qpn_biases[:0]
    )
    return torch.ops.vllm.sm70_skinny_awq_turbomind_linear(
        reshaped_x,
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        base_weight,
        base_scales,
        state.output_size,
        state.input_size,
        state.group_size,
        k_ld,
        q_ld,
    )


def apply_awq_marlin_overlay(
    state: SM70SkinnyAwqState,
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_zeros: torch.Tensor,
    base_g_idx: torch.Tensor,
    base_perm: torch.Tensor,
    base_workspace: torch.Tensor,
    is_k_full: bool,
) -> torch.Tensor:
    """Graph-safe Skinny overlay with a Marlin base fallback."""
    reshaped_x = x.reshape(-1, state.input_size).contiguous()
    codes = state.codes if "simt" not in state.disabled_routes else state.codes[:0]
    scales = state.scales if "simt" not in state.disabled_routes else state.scales[:0]
    biases = state.biases if "simt" not in state.disabled_routes else state.biases[:0]
    qpn_codes = (
        state.qpn_codes if "qpn" not in state.disabled_routes else state.qpn_codes[:0]
    )
    qpn_scales = (
        state.qpn_scales if "qpn" not in state.disabled_routes else state.qpn_scales[:0]
    )
    qpn_biases = (
        state.qpn_biases if "qpn" not in state.disabled_routes else state.qpn_biases[:0]
    )
    return torch.ops.vllm.sm70_skinny_awq_marlin_linear(
        reshaped_x,
        codes,
        scales,
        biases,
        qpn_codes,
        qpn_scales,
        qpn_biases,
        base_weight,
        base_scales,
        base_zeros,
        base_g_idx,
        base_perm,
        base_workspace,
        state.output_size,
        state.input_size,
        state.group_size,
        is_k_full,
    )
