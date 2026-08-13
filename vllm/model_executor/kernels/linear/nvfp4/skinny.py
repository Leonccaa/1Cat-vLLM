# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_SELF_CHECK_TOL = 3e-2
_route_log_seen: set[tuple[str, int, torch.dtype]] = set()


def qpn_prepack(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Prepack checkpoint-native NVFP4 bytes for the QPN fragment map.

    The result is a pure permutation of weight nibbles and scale bytes. It is
    built once at weight load and consumed by the M=4..16 QPN kernel.
    """
    if codes.dim() != 2 or scales.dim() != 2:
        raise ValueError("NVFP4 codes and scales must both be two-dimensional.")
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("QPN prepack expects uint8 views of NVFP4 codes and scales.")
    if codes.device != scales.device:
        raise ValueError("NVFP4 codes and scales must be on the same device.")

    n, k2 = codes.shape
    k = k2 * 2
    if scales.shape != (n, k // 16):
        raise ValueError(
            "NVFP4 scale shape mismatch: expected "
            f"{(n, k // 16)}, got {tuple(scales.shape)}."
        )
    if n % 32 or k % 64:
        return None, None

    device = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=device)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
        device=device,
    )
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    group = torch.arange(groups, device=device)
    kidx = group.view(groups, 1) * 16 + korder.view(1, 16)
    qcodes = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=device)
    qscales = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=device)

    # Broadcast indices are much larger than the payload. Bound temporary
    # storage to roughly 300 MiB even for a large vocabulary projection.
    tile_chunk = max(1, 36864 // groups)
    for tile_start in range(0, tiles, tile_chunk):
        tile_end = min(tile_start + tile_chunk, tiles)
        tile_count = tile_end - tile_start
        ncol = torch.arange(tile_start, tile_end, device=device).view(
            tile_count, 1
        ) * 32 + col.view(1, 32)
        selected = nibbles[
            ncol.view(tile_count, 1, 32, 1).expand(tile_count, groups, 32, 16),
            kidx.view(1, groups, 1, 16).expand(tile_count, groups, 32, 16),
        ]
        qcodes[tile_start:tile_end] = selected[..., 0::2] | (selected[..., 1::2] << 4)
        qscales[tile_start:tile_end] = scales[
            ncol.view(tile_count, 1, 32).expand(tile_count, groups, 32),
            group.view(1, groups, 1).expand(tile_count, groups, 32),
        ]

    return qcodes.view(-1).contiguous(), qscales.view(-1).contiguous()


def _skinny_nvfp4_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    out = _try_skinny_nvfp4_linear(
        x,
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        n,
        k,
    )
    if out is None:
        raise RuntimeError(
            f"SM70 Skinny NVFP4 has no route for M={x.shape[0]}, N={n}, K={k}; "
            "use a graph-safe overlay op that owns the selected base fallback."
        )
    return out


def _try_skinny_nvfp4_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor | None:
    """Run a supported route, or return ``None`` inside a hybrid op."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"SM70 skinny NVFP4 supports FP16 or BF16 activations, but got {x.dtype}."
        )

    # The small-M kernels execute in FP16 on Volta. For a BF16 model, make that
    # conversion explicit and restore the public output dtype afterwards.
    output_dtype = x.dtype
    kernel_x = x if x.dtype == torch.float16 else x.to(torch.float16)
    m = kernel_x.shape[0]

    use_qpn = (
        4 <= m <= 16
        and k % 64 == 0
        and n % 32 == 0
        and qpn_codes.numel() == n * (k // 2)
        and qpn_scales.numel() == n * (k // 16)
    )
    use_simt = (
        1 <= m <= 3
        and k % 128 == 0
        and n % 8 == 0
        and codes.numel() == n * (k // 2)
        and scales.numel() == n * (k // 16)
    )

    if use_qpn:
        route = "qpn"
    elif use_simt:
        route = "simt"
    else:
        route = "unsupported"
    # Keep diagnostics out of Dynamo traces. The hybrid custom op may be
    # invoked while a new dtype specialization is being compiled, and Python
    # logger/set side effects would otherwise turn an opaque runtime-M dispatch
    # into a graph break.
    if not torch.compiler.is_compiling():
        route_key = (route, m, output_dtype)
        if route_key not in _route_log_seen:
            _route_log_seen.add(route_key)
            logger.info(
                "SM70 skinny NVFP4 route: M=%d N=%d K=%d dtype=%s -> %s",
                m,
                n,
                k,
                output_dtype,
                route,
            )

    from vllm import _sm70_ops as sm70_ops

    if use_qpn:
        out = sm70_ops.skinny_nvfp4_gemm_qpn(
            kernel_x, qpn_codes, qpn_scales, global_scale, n
        )
    elif use_simt:
        out = sm70_ops.skinny_nvfp4_gemm_simt(kernel_x, codes, scales, global_scale)
    else:
        return None

    return out if output_dtype == torch.float16 else out.to(output_dtype)


def _skinny_nvfp4_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    n: int,
    k: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        k,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_nvfp4_linear",
    op_func=_skinny_nvfp4_linear_impl,
    fake_impl=_skinny_nvfp4_linear_fake,
)


def _skinny_nvfp4_turbomind_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    n: int,
    k: int,
    base_group_size: int,
    base_k_ld: int,
    base_q_ld: int,
) -> torch.Tensor:
    out = _try_skinny_nvfp4_linear(
        x,
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        n,
        k,
    )
    if out is not None:
        return out

    from vllm import _sm70_ops as sm70_ops

    output_dtype = x.dtype
    kernel_x = x if x.dtype == torch.float16 else x.to(torch.float16)
    out = torch.empty(
        (kernel_x.shape[0], n), dtype=torch.float16, device=kernel_x.device
    )
    sm70_ops.nvfp4_gemm_sm70_out(
        out,
        kernel_x,
        base_weight,
        base_scales,
        base_group_size,
        base_k_ld,
        base_q_ld,
    )
    return out if output_dtype == torch.float16 else out.to(output_dtype)


def _skinny_nvfp4_turbomind_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    n: int,
    k: int,
    base_group_size: int,
    base_k_ld: int,
    base_q_ld: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        base_weight,
        base_scales,
        k,
        base_group_size,
        base_k_ld,
        base_q_ld,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_nvfp4_turbomind_linear",
    op_func=_skinny_nvfp4_turbomind_linear_impl,
    fake_impl=_skinny_nvfp4_turbomind_linear_fake,
)


def _skinny_nvfp4_marlin_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_global_scale: torch.Tensor,
    base_workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    out = _try_skinny_nvfp4_linear(
        x,
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        n,
        k,
    )
    if out is not None:
        return out

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )

    return apply_fp4_marlin_linear(
        input=x,
        weight=base_weight,
        weight_scale=base_scales,
        weight_global_scale=base_global_scale,
        workspace=base_workspace,
        size_n=n,
        size_k=k,
    )


def _skinny_nvfp4_marlin_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    qpn_codes: torch.Tensor,
    qpn_scales: torch.Tensor,
    global_scale: float,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_global_scale: torch.Tensor,
    base_workspace: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    del (
        codes,
        scales,
        qpn_codes,
        qpn_scales,
        global_scale,
        base_weight,
        base_scales,
        base_global_scale,
        base_workspace,
        k,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_nvfp4_marlin_linear",
    op_func=_skinny_nvfp4_marlin_linear_impl,
    fake_impl=_skinny_nvfp4_marlin_linear_fake,
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().abs().max().clamp(min=1e-6)
    error = (actual.float() - expected.float()).abs().max() / denominator
    return float(error.item())


# E2M1: sign in bit 3, magnitude in bits 2:0.
_E2M1_MAGNITUDE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_fp32_reference(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    x: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Independent FP32 ground truth for the NVFP4 group-16 Skinny layout.

    Shares no code with the kernels or the base backends. In particular it does
    not reproduce the kernel's 16384 exponent trick or its bit-shuffle FP8
    decode, so a mistake in either shows up here rather than cancelling out.

    ``codes`` is uint8 [N, K/2] holding two E2M1 nibbles per byte with the low
    nibble at even k, ``scales`` is uint8 [N, K/16] of FP8-E4M3 bytes, and
    ``x`` is [M, K]. Chunked over N to bound the dequantized weight.
    """
    n, packed_k = codes.shape
    k = packed_k * 2
    groups = scales.shape[1]
    lut = torch.tensor(_E2M1_MAGNITUDE, dtype=torch.float32, device=codes.device)
    out = x.new_empty((x.shape[0], n), dtype=torch.float32)
    x32 = x.float()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = codes[start:stop]
        nibbles = torch.stack((block & 0xF, block >> 4), dim=-1).reshape(
            stop - start, k
        )
        magnitude = lut[(nibbles & 0x7).long()]
        signed = torch.where(nibbles & 0x8 != 0, -magnitude, magnitude)
        # torch decodes E4M3 natively; no need to mirror the kernel's shifts.
        group_scale = (
            scales[start:stop].view(torch.float8_e4m3fn).float() * global_scale
        )
        weight = signed * group_scale.repeat_interleave(k // groups, dim=1)
        out[:, start:stop] = x32 @ weight.t()
    return out


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Report more than one number; a lone max-relative scalar hides
    localized errors and near-zero outputs."""
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


@dataclass
class _ResidencyDecision:
    """Per-(N, K) verdict on whether the overlay earns its VRAM."""

    roi: float  # microseconds saved per MiB of overlay
    mib: float
    saved_us: float
    keep: bool


_residency_decisions: dict[tuple[int, int], _ResidencyDecision] = {}


class SkinnyNvFp4LinearKernel(NvFp4LinearKernel):
    """SM70 small-M NVFP4 decorator over a selected base kernel."""

    def __init__(
        self,
        config: NvFp4LinearLayerConfig,
        base_kernel: NvFp4LinearKernel,
    ) -> None:
        supported, reason = self.is_supported()
        if not supported:
            raise ValueError(f"SM70 Skinny NVFP4 is unavailable: {reason}")
        self.config = config
        self.base_kernel = base_kernel

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            if (
                not current_platform.is_cuda()
                or not current_platform.is_device_capability((7, 0))
            ):
                return False, "requires exact CUDA capability 7.0"
        elif compute_capability != 70:
            return False, "requires exact CUDA capability 7.0"

        required_ops = (
            "skinny_nvfp4_gemm_simt",
            "skinny_nvfp4_gemm_qpn",
        )
        missing = [name for name in required_ops if not hasattr(torch.ops._C, name)]
        if missing:
            return False, "missing SM70 extension ops: " + ", ".join(missing)
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def _validate_shape(self, layer: torch.nn.Module) -> None:
        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        cases = [(1, "simt")]
        if layer.skinny_qpn_codes.numel() > 0:
            cases.append((8, "qpn"))

        for m, route in cases:
            if route in layer.skinny_disabled_routes:
                continue
            validation_key = (type(self.base_kernel).__name__, route)
            if validation_key in layer.skinny_validated_routes:
                continue
            try:
                values = torch.arange(
                    m * k, device=layer.skinny_codes.device, dtype=torch.int32
                )
                x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(m, k)
                reference = self.base_kernel.apply_weights(layer, x, None)
                if route == "simt":
                    actual = torch.ops._C.skinny_nvfp4_gemm_simt(
                        x,
                        layer.skinny_codes,
                        layer.skinny_scales,
                        layer.skinny_global_scale,
                    )
                else:
                    actual = torch.ops._C.skinny_nvfp4_gemm_qpn(
                        x,
                        layer.skinny_qpn_codes,
                        layer.skinny_qpn_scales,
                        layer.skinny_global_scale,
                        n,
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
                if envs.use_sm70_skinny_strict_check():
                    # Comparing against the base backend is circular; both
                    # consume the same checkpoint through similar unpack logic.
                    stats = _error_stats(
                        actual,
                        nvfp4_fp32_reference(
                            layer.skinny_codes,
                            layer.skinny_scales,
                            layer.skinny_global_scale,
                            x,
                        ),
                    )
                    logger.info(
                        "SM70 Skinny NVFP4 %s FP32 ground truth N=%d K=%d: "
                        "max_rel=%.3e rms=%.3e p50=%.3e p99=%.3e max_abs=%.3e",
                        route,
                        n,
                        k,
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
                            f"{stats['max_rel']:.3e} exceeds "
                            f"{_SELF_CHECK_TOL:.3e}"
                        )
            except Exception:
                layer.skinny_disabled_routes.add(route)
                logger.exception(
                    "SM70 Skinny NVFP4 %s self-check failed for N=%d K=%d "
                    "against base=%s; disabling only this route.",
                    route,
                    n,
                    k,
                    type(self.base_kernel).__name__,
                )
                continue
            layer.skinny_validated_routes.add(validation_key)
            logger.info_once(
                "SM70 Skinny NVFP4 %s self-check passed for N=%d K=%d against base=%s.",
                route,
                n,
                k,
                type(self.base_kernel).__name__,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != torch.uint8 or layer.weight.dim() != 2:
            raise TypeError("Skinny NVFP4 expects native uint8 [N, K/2] weights.")

        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        if layer.weight.shape != (n, k // 2):
            raise ValueError(
                f"Skinny NVFP4 expected weight shape {(n, k // 2)}, "
                f"got {tuple(layer.weight.shape)}."
            )

        # Keep an overlay-owned native layout. The selected base is free to
        # replace or repack the checkpoint tensors in place.
        layer.register_parameter(
            "skinny_codes", Parameter(layer.weight.data.clone(), requires_grad=False)
        )
        layer.register_parameter(
            "skinny_scales",
            Parameter(
                layer.weight_scale.data.view(torch.uint8).clone().contiguous(),
                requires_grad=False,
            ),
        )
        layer.skinny_global_scale = float(
            layer.weight_global_scale.data.float().max().item()
        )

        qcodes, qscales = qpn_prepack(layer.skinny_codes.data, layer.skinny_scales.data)
        empty = layer.skinny_codes.data.new_empty(0)
        layer.register_parameter(
            "skinny_qpn_codes",
            Parameter(qcodes if qcodes is not None else empty, requires_grad=False),
        )
        layer.register_parameter(
            "skinny_qpn_scales",
            Parameter(qscales if qscales is not None else empty, requires_grad=False),
        )
        layer.skinny_disabled_routes = set()
        layer.skinny_validated_routes = set()
        layer.skinny_enabled = True

        self.base_kernel.process_weights_after_loading(layer)

        self._validate_shape(layer)
        if not self._apply_residency_policy(layer):
            return
        logger.info_once(
            "SM70 Skinny NVFP4 dense overlay enabled: SIMT M<=3, QPN M=4..16, base=%s.",
            type(self.base_kernel).__name__,
        )

    def _apply_residency_policy(self, layer: torch.nn.Module) -> bool:
        """Release the overlay layouts that do not earn their VRAM.

        NVFP4 keeps a second copy of every covered weight at 0.5625 byte each
        (plus another for QPN), which on a 32 GiB V100 comes straight out of
        KV cache. Where the base backend already saturates HBM there is no
        latency left to win, so the copy is pure cost.

        SIMT and QPN are scored separately at the M each actually serves and
        charged only their own copy, because a shape can be worth keeping for
        decode and not for MTP verification, or the reverse. Timing is L2-cold
        and reduced across the TP group so every rank agrees - see
        sm70_skinny._time_apply and _agree_across_tp for why both matter.
        """
        from vllm.model_executor.layers.quantization.sm70_skinny import (
            _agree_across_tp,
            _time_apply,
        )

        n = layer.output_size_per_partition
        k = layer.input_size_per_partition
        per_layout_mib = n * k * 0.5625 / (1024.0 * 1024.0)

        routes = []
        if (
            layer.skinny_codes.numel() > 0
            and "simt" not in layer.skinny_disabled_routes
        ):
            routes.append("simt")
        if (
            layer.skinny_qpn_codes.numel() > 0
            and "qpn" not in layer.skinny_disabled_routes
        ):
            routes.append("qpn")
        if not routes:
            layer.skinny_enabled = False
            return False

        device = layer.skinny_codes.device
        for route in routes:
            key = (n, k, route)
            decision = _residency_decisions.get(key)
            if decision is None:
                rows = 1 if route == "simt" else 8
                values = torch.arange(rows * k, device=device, dtype=torch.int32)
                x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(rows, k)
                if route == "simt":

                    def skinny(x=x):
                        return torch.ops._C.skinny_nvfp4_gemm_simt(
                            x,
                            layer.skinny_codes,
                            layer.skinny_scales,
                            layer.skinny_global_scale,
                        )

                else:

                    def skinny(x=x):
                        return torch.ops._C.skinny_nvfp4_gemm_qpn(
                            x,
                            layer.skinny_qpn_codes,
                            layer.skinny_qpn_scales,
                            layer.skinny_global_scale,
                            n,
                        )

                try:
                    base_us = _time_apply(
                        lambda x=x: self.base_kernel.apply_weights(layer, x),
                        device=device,
                    )
                    skinny_us = _time_apply(skinny, device=device)
                except Exception:
                    logger.exception(
                        "SM70 Skinny NVFP4 %s residency measurement failed for "
                        "N=%d K=%d; keeping the overlay.",
                        route,
                        n,
                        k,
                    )
                    continue
                saved = _agree_across_tp(base_us - skinny_us)
                roi = saved / per_layout_mib if per_layout_mib > 0 else 0.0
                decision = _ResidencyDecision(
                    roi=roi,
                    mib=per_layout_mib,
                    saved_us=saved,
                    keep=roi >= envs.get_sm70_skinny_min_roi(),
                )
                _residency_decisions[key] = decision
                logger.info(
                    "SM70 Skinny NVFP4 residency %s N=%d K=%d: base=%.1fus "
                    "skinny=%.1fus saved=%.1fus overlay=%.1fMiB "
                    "roi=%.3fus/MiB -> %s",
                    route,
                    n,
                    k,
                    base_us,
                    skinny_us,
                    saved,
                    per_layout_mib,
                    roi,
                    "keep" if decision.keep else "drop",
                )
            if decision.keep:
                continue
            empty = layer.skinny_codes.data.new_empty(0)
            if route == "simt":
                layer.skinny_codes = Parameter(empty, requires_grad=False)
                layer.skinny_scales = Parameter(empty, requires_grad=False)
            else:
                layer.skinny_qpn_codes = Parameter(empty, requires_grad=False)
                layer.skinny_qpn_scales = Parameter(empty, requires_grad=False)
            layer.skinny_disabled_routes.add(route)

        layer.skinny_enabled = (
            layer.skinny_codes.numel() > 0 or layer.skinny_qpn_codes.numel() > 0
        )
        return layer.skinny_enabled

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        if not getattr(layer, "skinny_enabled", True):
            # Residency policy released this shape's overlay; go straight to
            # the base backend rather than paying the hybrid op's dispatch.
            return self.base_kernel.apply_weights(layer, x, bias)
        reshaped_x = x.reshape(-1, k).contiguous()
        rows = reshaped_x.shape[0]
        simt_codes = (
            layer.skinny_codes
            if "simt" not in layer.skinny_disabled_routes
            else layer.skinny_codes[:0]
        )
        simt_scales = (
            layer.skinny_scales
            if "simt" not in layer.skinny_disabled_routes
            else layer.skinny_scales[:0]
        )
        qpn_codes = (
            layer.skinny_qpn_codes
            if "qpn" not in layer.skinny_disabled_routes
            else layer.skinny_qpn_codes[:0]
        )
        qpn_scales = (
            layer.skinny_qpn_scales
            if "qpn" not in layer.skinny_disabled_routes
            else layer.skinny_qpn_scales[:0]
        )

        # The hybrid op is deliberately opaque to torch.compile.  It sees the
        # real runtime M and owns both the Skinny route and the selected base
        # fallback, so a decode trace cannot pin a later prefill to Skinny.
        base_name = type(self.base_kernel).__name__
        if base_name == "TurboMindNvFp4LinearKernel":
            from vllm.model_executor.layers.quantization import sm70_turbomind

            base_state = getattr(layer, sm70_turbomind.STATE_ATTR)
            out = torch.ops.vllm.sm70_skinny_nvfp4_turbomind_linear(
                reshaped_x,
                simt_codes,
                simt_scales,
                qpn_codes,
                qpn_scales,
                layer.skinny_global_scale,
                base_state.weight,
                base_state.scales,
                n,
                k,
                base_state.group_size,
                base_state.k_ld,
                base_state.q_ld,
            )
            if bias is not None:
                out.add_(bias)
            return out.reshape(x.shape[:-1] + (n,))
        if base_name == "MarlinNvFp4LinearKernel":
            out = torch.ops.vllm.sm70_skinny_nvfp4_marlin_linear(
                reshaped_x,
                simt_codes,
                simt_scales,
                qpn_codes,
                qpn_scales,
                layer.skinny_global_scale,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                n,
                k,
            )
            if bias is not None:
                out.add_(bias)
            return out.reshape(x.shape[:-1] + (n,))

        # Keep the simple eager decorator behavior for third-party test/base
        # kernels. Production SM70 selection currently resolves to one of the
        # two graph-safe hybrid paths above.
        use_simt = (
            1 <= rows <= 3
            and "simt" not in layer.skinny_disabled_routes
            and k % 128 == 0
            and n % 8 == 0
            and layer.skinny_codes.numel() == n * (k // 2)
        )
        use_qpn = (
            4 <= rows <= 16
            and "qpn" not in layer.skinny_disabled_routes
            and k % 64 == 0
            and n % 32 == 0
            and layer.skinny_qpn_codes.numel() == n * (k // 2)
        )
        if not (use_simt or use_qpn):
            return self.base_kernel.apply_weights(layer, x, bias)
        out = torch.ops.vllm.sm70_skinny_nvfp4_linear(
            reshaped_x,
            simt_codes,
            simt_scales,
            qpn_codes,
            qpn_scales,
            layer.skinny_global_scale,
            n,
            k,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(x.shape[:-1] + (n,))
