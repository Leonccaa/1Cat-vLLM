# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_residency
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

AWQ_GROUP_SIZE = 128
_SELF_CHECK_TOL = 3e-2
_AWQ_REVERSE_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_QPN_K_ORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)
_QPN_TARGET_BLOCKS = 160

_awq_route_log_seen: set[tuple[str, int, int, int, torch.dtype]] = set()


def qpn_k_splits(
    input_size: int,
    output_size: int,
    target_blocks: int = _QPN_TARGET_BLOCKS,
) -> int:
    """Mirror the CUDA QPN split-K launch policy for load-time diagnostics."""
    warps = 4
    min_groups_per_warp = 4
    max_splits = 16
    group_count = input_size // 16
    tiles = output_size // 32
    splits = 1
    while splits < max_splits:
        if tiles * splits >= target_blocks:
            break
        next_splits = splits * 2
        if group_count % (warps * next_splits) != 0:
            break
        if group_count // (warps * next_splits) < min_groups_per_warp:
            break
        splits = next_splits
    return splits


def log_qpn_split_geometry(format_name: str, input_size: int, output_size: int) -> None:
    """Log once per unique shape whether QPN split-K will actually engage."""
    splits = qpn_k_splits(input_size, output_size)
    logger.info_once(
        "SM70 Skinny %s QPN split-K %s for N=%d K=%d: "
        "splits=%d, N32 tiles=%d, target_blocks=%d.",
        format_name,
        "active" if splits > 1 else "inactive",
        output_size,
        input_size,
        splits,
        output_size // 32,
        _QPN_TARGET_BLOCKS,
    )


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
    # Checkpoint-native (qweight, scales, qzeros), retained only while the
    # strict self-check needs them so the FP32 ground truth can start from the
    # on-disk format and therefore also cover the prepack. Released by
    # validate_awq_state; None when the strict check is off.
    native: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

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
        if qpn_codes is not None:
            log_qpn_split_geometry("AWQ", qweight.shape[0], qweight.shape[1] * 8)
    else:
        qpn_codes = qpn_scales = qpn_biases = None
        logger.info_once(
            "SM70 Skinny AWQ QPN disabled by layout=simt; split-K is not "
            "part of this load."
        )

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
        native=(
            (qweight, scales, qzeros) if envs.use_sm70_skinny_strict_check() else None
        ),
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
    simt_max_rows = envs.get_sm70_skinny_simt_max_rows()
    use_qpn = (
        4 <= m <= 16
        and k % AWQ_GROUP_SIZE == 0
        and n % 32 == 0
        and qpn_codes.numel() == n * (k // 2)
        and qpn_scales.numel() == n * (k // AWQ_GROUP_SIZE)
        and qpn_biases.numel() == qpn_scales.numel()
    )
    use_simt = (
        1 <= m <= simt_max_rows
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
        route_key = (route, m, n, k, output_dtype)
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


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Report more than one number.

    A single max-relative scalar hides exactly the failures worth catching: a
    handful of bad elements disappear into the max over a large output, and
    near-zero outputs look fine relative to the tensor-wide maximum.
    """
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().max().clamp(min=1e-6)
    quantiles = torch.quantile(
        difference.flatten().float(),
        torch.tensor([0.5, 0.99, 1.0], device=difference.device),
    )
    return {
        "max_rel": float((difference.max() / denominator).item()),
        "rms": float(difference.pow(2).mean().sqrt().item()),
        "p50": float(quantiles[0].item()),
        "p99": float(quantiles[1].item()),
        "max_abs": float(quantiles[2].item()),
    }


def awq_fp32_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    x: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Independent FP32 ground truth for the AWQ g128 Skinny layout.

    Deliberately shares no code with the kernels or with the base backends:
    comparing Skinny against TurboMind or Marlin cannot catch an error that
    both make, and both consume the same checkpoint through similar unpack
    logic. This walks the packed bytes directly in FP32.

    ``codes`` is uint8 [N, K/2] with the low nibble at even k, ``scales`` and
    ``biases`` are FP16 [N, K/128], and ``x`` is [M, K]. Chunked over N so a
    vocabulary-sized projection does not materialize an N*K FP32 weight.

    Scope: this starts from the Skinny layout, so it validates the kernel but
    not the native-AWQ to Skinny prepack that produced that layout. Use
    :func:`awq_native_fp32_reference` to cover the prepack as well.
    """
    n, packed_k = codes.shape
    k = packed_k * 2
    groups = scales.shape[1]
    out = x.new_empty((x.shape[0], n), dtype=torch.float32)
    x32 = x.float()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = codes[start:stop]
        nibbles = torch.stack((block & 0xF, block >> 4), dim=-1)
        quantized = nibbles.reshape(stop - start, k).float()
        weight = quantized * scales[start:stop].float().repeat_interleave(
            k // groups, dim=1
        ) + biases[start:stop].float().repeat_interleave(k // groups, dim=1)
        out[:, start:stop] = x32 @ weight.t()
    return out


def awq_native_fp32_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    x: torch.Tensor,
    chunk: int = 1024,
) -> torch.Tensor:
    """FP32 ground truth computed from the checkpoint-native AWQ tensors.

    Starts one step earlier than :func:`awq_fp32_reference`: from the int32
    tensors as they appear on disk, so it covers the native-to-Skinny prepack
    (nibble de-interleave, transpose to N-major, byte repack, ``bias = -z*s``)
    in addition to the kernel. It also works for any resident layout, since it
    never touches the Skinny buffers.

    It performs the shifts, transpose and group expansion independently; it
    necessarily shares the AWQ nibble-order convention, because that
    convention *is* the on-disk format definition rather than an implementation
    choice.

    ``qweight`` is int32 [K, N/8], ``qzeros`` int32 [G, N/8], ``scales``
    FP16 [G, N], ``x`` is [M, K]. Returns [M, N] in FP32. Chunked over K to
    bound the dequantized weight.
    """
    # Logical index j lives at nibble slot AWQ_SLOT[j] inside each int32.
    awq_slot = torch.tensor(
        [0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.long, device=qweight.device
    )
    shifts = (awq_slot * 4).to(torch.int32)

    def unpack(packed: torch.Tensor) -> torch.Tensor:
        # [rows, cols] int32 -> [rows, cols * 8] uint8, logical order.
        wide = (packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
        return wide.reshape(packed.shape[0], -1)

    k, n = qweight.shape[0], qweight.shape[1] * 8
    zeros = unpack(qzeros).float()  # [G, N]
    scales32 = scales.float()  # [G, N]

    out = x.new_zeros((x.shape[0], n), dtype=torch.float32)
    x32 = x.float()
    for start in range(0, k, chunk):
        stop = min(start + chunk, k)
        q = unpack(qweight[start:stop]).float()  # [chunk, N]
        rows = torch.arange(start, stop, device=qweight.device) // group_size
        weight = (q - zeros[rows]) * scales32[rows]  # [chunk, N]
        out += x32[:, start:stop] @ weight
    return out


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
        local_error: Exception | None = None
        local_ok = route not in state.disabled_routes
        if local_ok and validation_key not in state.validated_routes:
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
                if (
                    not math.isfinite(relative_error)
                    or relative_error > _SELF_CHECK_TOL
                ):
                    raise RuntimeError(
                        f"{route} relative error {relative_error:.3e} exceeds "
                        f"{_SELF_CHECK_TOL:.3e}"
                    )
                if envs.use_sm70_skinny_strict_check() and state.native is not None:
                    # Start from checkpoint-native tensors so the independent
                    # reference also covers the prepack.
                    truth = awq_native_fp32_reference(
                        state.native[0],
                        state.native[1],
                        state.native[2],
                        state.group_size,
                        x,
                    )
                    stats = _error_stats(actual, truth)
                    logger.info(
                        "SM70 Skinny AWQ %s FP32 ground truth N=%d K=%d: "
                        "max_rel=%.3e rms=%.3e p50=%.3e p99=%.3e max_abs=%.3e",
                        route,
                        state.output_size,
                        state.input_size,
                        stats["max_rel"],
                        stats["rms"],
                        stats["p50"],
                        stats["p99"],
                        stats["max_abs"],
                    )
                    if (
                        not math.isfinite(stats["max_rel"])
                        or stats["max_rel"] > _SELF_CHECK_TOL
                    ):
                        raise RuntimeError(
                            f"{route} FP32 ground-truth relative error "
                            f"{stats['max_rel']:.3e} exceeds {_SELF_CHECK_TOL:.3e}"
                        )
            except Exception as exc:
                local_ok = False
                local_error = exc
                logger.exception(
                    "SM70 Skinny AWQ %s local self-check failed for N=%d K=%d "
                    "against base=%s.",
                    route,
                    state.output_size,
                    state.input_size,
                    base_backend,
                )

        if not sm70_residency.all_ranks_succeeded(local_ok, device=state.codes.device):
            _release_awq_route(state, route)
            if local_error is None:
                logger.error(
                    "SM70 Skinny AWQ %s self-check failed on another TP rank "
                    "for N=%d K=%d; disabling this route on every rank.",
                    route,
                    state.output_size,
                    state.input_size,
                )
            if envs.get_sm70_skinny_mode() == "on":
                raise RuntimeError(
                    "VLLM_SM70_SKINNY=on requires the AWQ "
                    f"{route} self-check to pass on every TP rank."
                ) from local_error
            continue
        state.validated_routes.add(validation_key)
        logger.info_once(
            "SM70 Skinny AWQ %s self-check passed for N=%d K=%d against base=%s.",
            route,
            state.output_size,
            state.input_size,
            base_backend,
        )

    # The native tensors exist only for the strict check; the caller frees the
    # originals right after loading, so do not keep a second reference alive.
    state.native = None


_residency_decisions: dict[
    sm70_residency.ResidencyKey, sm70_residency.ResidencyDecision
] = {}


def _release_awq_route(state: SM70SkinnyAwqState, route: str) -> None:
    """Release a disabled AWQ layout immediately instead of retaining dead VRAM."""
    empty_codes = state.codes.new_empty(0)
    empty_meta = state.scales.new_empty(0)
    if route == "simt":
        state.codes = empty_codes
        state.scales = empty_meta
        state.biases = empty_meta
    elif route == "qpn":
        state.qpn_codes = empty_codes
        state.qpn_scales = empty_meta
        state.qpn_biases = empty_meta
    else:
        raise ValueError(f"Unknown SM70 Skinny AWQ route {route!r}.")
    state.disabled_routes.add(route)


def overlay_mib(output_size: int, input_size: int, layouts: int = 1) -> float:
    """Resident overlay bytes for one dense layer, in MiB.

    AWQ g128 costs 0.5 byte/weight of codes plus an FP16 scale and an FP16 bias
    per 128 weights, i.e. 0.53125 byte/weight, per resident layout.
    """
    return output_size * input_size * 0.53125 * layouts / (1024.0 * 1024.0)


def apply_residency_policy(
    state: SM70SkinnyAwqState,
    reference_apply: Callable[[torch.Tensor], torch.Tensor],
) -> bool:
    """Release the overlay layouts that do not earn their memory.

    The overlay holds a second full copy of every weight it covers, so on a
    32 GiB V100 it trades directly against KV cache and context length. That
    trade is not uniform across layers: where the base backend already
    saturates HBM there is no latency left to win, but the memory cost is the
    same as anywhere else.

    SIMT and QPN are scored separately, each at the M it actually serves (M=1
    and M=8) and each charged only its own copy of the weights. They are
    independent residency decisions: a shape can be worth keeping for decode
    and not worth keeping for MTP verification, or the reverse.

    Decisions are cached per (format, N, K, route), so the cost is a few timed
    GEMMs per distinct shape rather than per layer, and each measurement is
    reduced across the TP group so every rank reaches the same verdict.

    Returns True while any Skinny route is still resident.
    """
    for disabled_route in tuple(state.disabled_routes):
        _release_awq_route(state, disabled_route)

    routes = []
    if state.has_simt and "simt" not in state.disabled_routes:
        routes.append("simt")
    if state.has_qpn and "qpn" not in state.disabled_routes:
        routes.append("qpn")
    if not routes:
        return False

    from vllm import _sm70_ops as sm70_ops

    device = state.codes.device

    def make_benchmark(route: str) -> sm70_residency.RouteBenchmark:
        rows = 1 if route == "simt" else 8
        values = torch.arange(rows * state.input_size, device=device, dtype=torch.int32)
        x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(
            rows, state.input_size
        )

        def base_apply() -> torch.Tensor:
            return reference_apply(x)

        if route == "simt":

            def skinny_apply() -> torch.Tensor:
                return sm70_ops.skinny_awq_gemm_simt(
                    x,
                    state.codes,
                    state.scales,
                    state.biases,
                    state.group_size,
                )

        else:

            def skinny_apply() -> torch.Tensor:
                return sm70_ops.skinny_awq_gemm_qpn(
                    x,
                    state.qpn_codes,
                    state.qpn_scales,
                    state.qpn_biases,
                    state.group_size,
                    state.output_size,
                )

        return sm70_residency.RouteBenchmark(base_apply, skinny_apply)

    def release_route(route: str) -> None:
        _release_awq_route(state, route)

    retained = sm70_residency.apply_route_policy(
        format_name="AWQ",
        output_size=state.output_size,
        input_size=state.input_size,
        routes=routes,
        device=device,
        overlay_mib=overlay_mib(state.output_size, state.input_size),
        min_roi=envs.get_sm70_skinny_min_roi(),
        force_on=envs.get_sm70_skinny_mode() == "on",
        decisions=_residency_decisions,
        make_benchmark=make_benchmark,
        release_route=release_route,
    )
    return bool(retained)


def log_residency_summary() -> None:
    """One ranked table so a real threshold can be chosen from real numbers."""
    sm70_residency.log_residency_summary(
        decisions=_residency_decisions,
        format_name="AWQ",
        min_roi=envs.get_sm70_skinny_min_roi(),
    )


def select_awq_route(state: SM70SkinnyAwqState, rows: int) -> str | None:
    if (
        1 <= rows <= envs.get_sm70_skinny_simt_max_rows()
        and state.has_simt
        and "simt" not in state.disabled_routes
    ):
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
