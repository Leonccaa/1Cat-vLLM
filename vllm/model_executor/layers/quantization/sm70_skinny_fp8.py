# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental SM70 small-M overlay for block-128 E4M3 weights.

The selected base linear kernel remains authoritative. This module keeps a
second, QPN8-prepacked copy only when ``VLLM_SM70_SKINNY=on``. A graph-opaque
hybrid op sends supported FP16 M=1..8 calls to QPN8 and every other call to the
already-selected Marlin base. ``auto`` remains base-only until full-model and
one-copy gates pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import vllm._custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

STATE_ATTR = "_sm70_fp8_qpn8_state"
BLOCK_SIZE = (128, 128)
MAX_ROWS = 8
_SELF_CHECK_TOL = 3e-3
_QPN_K_ORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)
_logged_shapes: set[tuple[int, int, int, int]] = set()


@dataclass(frozen=True)
class SM70Fp8Qpn8State:
    codes: torch.Tensor
    scales256: torch.Tensor
    ratios: torch.Tensor
    output_size: int
    input_size: int
    split_k: int
    nacc: int

    @property
    def overlay_mib(self) -> float:
        total = self.codes.numel() * self.codes.element_size()
        total += self.scales256.numel() * self.scales256.element_size()
        total += self.ratios.numel() * self.ratios.element_size()
        return total / (1024.0 * 1024.0)


def _is_exact_sm70_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) == (7, 0)


def qpn8_prepack(weight_bytes: torch.Tensor) -> torch.Tensor:
    """Permute uint8 [N,K] E4M3 bytes into QPN8 fragment order."""
    if weight_bytes.dtype != torch.uint8 or weight_bytes.dim() != 2:
        raise TypeError("SM70 FP8 QPN8 prepack expects uint8 [N,K] weights.")
    if not weight_bytes.is_contiguous():
        weight_bytes = weight_bytes.contiguous()

    n, k = weight_bytes.shape
    if n % 128 or k % 128:
        raise ValueError("SM70 FP8 QPN8 requires N and K divisible by 128.")

    device = weight_bytes.device
    tiles = n // 32
    groups = k // 16
    lane = torch.arange(32, device=device)
    column = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) != 0).to(torch.int64) * 4
    k_order = torch.tensor(_QPN_K_ORDER, device=device)
    group = torch.arange(groups, device=device)
    k_index = group.view(groups, 1) * 16 + k_order.view(1, 16)
    packed = torch.empty((tiles, groups, 32, 16), dtype=torch.uint8, device=device)

    # Broadcasted indices are much larger than the result. Keep temporary
    # storage around 300 MiB for large projections such as a quantized lm_head.
    tile_chunk = max(1, 36864 // max(groups, 1))
    for tile_start in range(0, tiles, tile_chunk):
        tile_end = min(tile_start + tile_chunk, tiles)
        tile_count = tile_end - tile_start
        n_index = torch.arange(tile_start, tile_end, device=device).view(
            tile_count, 1
        ) * 32 + column.view(1, 32)
        packed[tile_start:tile_end] = weight_bytes[
            n_index.view(tile_count, 1, 32, 1).expand(tile_count, groups, 32, 16),
            k_index.view(1, groups, 1, 16).expand(tile_count, groups, 32, 16),
        ]
    return packed.view(-1).contiguous()


def make_block_metadata(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build final absolute scales and adjacent K-block scale ratios."""
    if scales.dim() != 2:
        raise ValueError("SM70 FP8 QPN8 scales must be [N/128,K/128].")
    scales32 = scales.float().contiguous()
    if not bool(torch.isfinite(scales32).all()) or not bool((scales32 > 0).all()):
        raise ValueError("SM70 FP8 QPN8 requires finite positive block scales.")
    ratios = torch.ones_like(scales32)
    ratios[:, 1:] = scales32[:, :-1] / scales32[:, 1:]
    return (scales32 * 256.0).contiguous(), ratios.contiguous()


def choose_launch_geometry(output_size: int, input_size: int) -> tuple[int, int]:
    """Choose a deterministic geometry from the real-shape micro-gate.

    Narrow N needs more split-K warps to fill Volta; wider projections reached
    their best bandwidth at split16. The divisibility walk keeps the policy
    format-generic instead of encoding checkpoint shape names.
    """
    groups = input_size // 16
    preferred = 32 if output_size < 4096 else 16
    for split_k in (32, 16, 8, 4):
        if split_k <= preferred and groups % split_k == 0:
            return split_k, 1 if split_k == 32 else 2
    raise ValueError(
        f"SM70 FP8 QPN8 has no split-K geometry for N={output_size}, K={input_size}."
    )


def prepare_state(
    weight: torch.Tensor,
    scales: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int] | None,
) -> SM70Fp8Qpn8State | None:
    """Build the opt-in overlay from checkpoint-native tensors."""
    if not envs.use_sm70_skinny_fp8():
        return None
    if not _is_exact_sm70_tensor(weight):
        return None
    if tuple(weight_block_size or ()) != BLOCK_SIZE:
        return None
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return None
    if not ops.sm70_marlin_available() or not hasattr(
        torch.ops._C, "sm70_fp8_qpn8_b128_gemm"
    ):
        logger.warning_once(
            "VLLM_SM70_SKINNY=on requested block-128 FP8 QPN8, but the "
            "SM70 extension op is unavailable; retaining the selected base."
        )
        return None

    n, k = weight.shape
    expected_scales = (n // 128, k // 128)
    if n % 128 or k % 128 or tuple(scales.shape) != expected_scales:
        logger.warning_once(
            "SM70 FP8 QPN8 skipped unsupported weight/scale geometry: "
            "weight=%s scale=%s.",
            tuple(weight.shape),
            tuple(scales.shape),
        )
        return None

    split_k, nacc = choose_launch_geometry(n, k)
    codes = qpn8_prepack(weight.contiguous().view(torch.uint8))
    scales256, ratios = make_block_metadata(scales)
    return SM70Fp8Qpn8State(
        codes=codes,
        scales256=scales256,
        ratios=ratios,
        output_size=n,
        input_size=k,
        split_k=split_k,
        nacc=nacc,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().norm().clamp(min=1e-6)
    return float(((actual.float() - expected.float()).norm() / denominator).item())


def validate_and_attach_state(
    layer: torch.nn.Module,
    base_kernel: object,
    state: SM70Fp8Qpn8State | None,
) -> bool:
    """Cross-check QPN8 against the selected base after base preparation."""
    if state is None:
        return False

    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )

    if not isinstance(base_kernel, MarlinFP8ScaledMMLinearKernel):
        logger.warning_once(
            "SM70 FP8 QPN8 currently has no graph-safe wrapper for base=%s; "
            "retaining the selected base.",
            type(base_kernel).__name__,
        )
        return False

    n, k = state.output_size, state.input_size
    try:
        values = torch.arange(8 * k, device=state.codes.device, dtype=torch.int32)
        x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(8, k)
        expected = base_kernel.apply_weights(layer, x, None)
        actual = ops.sm70_fp8_qpn8_b128_gemm(
            x,
            state.codes,
            state.scales256,
            state.ratios,
            n,
            state.split_k,
            state.nacc,
        )
        error = _relative_error(actual, expected)
        if not bool(torch.isfinite(actual).all()) or not math.isfinite(error):
            raise RuntimeError("non-finite QPN8 output or error")
        if error > _SELF_CHECK_TOL:
            raise RuntimeError(
                f"relative error {error:.3e} exceeds {_SELF_CHECK_TOL:.3e}"
            )
    except Exception:
        if envs.get_sm70_skinny_mode() == "on":
            logger.exception(
                "SM70 FP8 QPN8 self-check failed for N=%d K=%d against %s.",
                n,
                k,
                type(base_kernel).__name__,
            )
            raise
        logger.warning_once(
            "SM70 FP8 QPN8 self-check rejected N=%d K=%d; retaining %s.",
            n,
            k,
            type(base_kernel).__name__,
        )
        return False

    setattr(layer, STATE_ATTR, state)
    shape_key = (n, k, state.split_k, state.nacc)
    if shape_key not in _logged_shapes:
        _logged_shapes.add(shape_key)
        logger.info(
            "SM70 FP8 QPN8 enabled for N=%d K=%d: M=1..8, split-K=%d, "
            "NACC=%d, overlay=%.1f MiB, base=%s.",
            n,
            k,
            state.split_k,
            state.nacc,
            state.overlay_mib,
            type(base_kernel).__name__,
        )
    return True


def _sm70_skinny_fp8_marlin_linear_impl(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales256: torch.Tensor,
    ratios: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_workspace: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    split_k: int,
    nacc: int,
) -> torch.Tensor:
    if x.dtype == torch.float16 and bias is None and 1 <= x.shape[0] <= MAX_ROWS:
        return ops.sm70_fp8_qpn8_b128_gemm(
            x, codes, scales256, ratios, n, split_k, nacc
        )
    return apply_fp8_marlin_linear(
        input=x,
        weight=base_weight,
        weight_scale=base_scales,
        workspace=base_workspace,
        size_n=n,
        size_k=k,
        bias=bias,
    )


def _sm70_skinny_fp8_marlin_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales256: torch.Tensor,
    ratios: torch.Tensor,
    base_weight: torch.Tensor,
    base_scales: torch.Tensor,
    base_workspace: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    split_k: int,
    nacc: int,
) -> torch.Tensor:
    del (
        codes,
        scales256,
        ratios,
        base_weight,
        base_scales,
        base_workspace,
        bias,
        k,
        split_k,
        nacc,
    )
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="sm70_skinny_fp8_marlin_linear",
    op_func=_sm70_skinny_fp8_marlin_linear_impl,
    fake_impl=_sm70_skinny_fp8_marlin_linear_fake,
)


def apply_weights(
    layer: torch.nn.Module,
    base_kernel: object,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the graph-safe overlay, or call the selected base unchanged."""
    state = getattr(layer, STATE_ATTR, None)
    if not isinstance(state, SM70Fp8Qpn8State):
        return base_kernel.apply_weights(layer, x, bias)

    k = state.input_size
    output_shape = x.shape[:-1] + (state.output_size,)
    x_2d = x.reshape(-1, k).contiguous()
    out = torch.ops.vllm.sm70_skinny_fp8_marlin_linear(
        x_2d,
        state.codes,
        state.scales256,
        state.ratios,
        layer.weight,
        layer.weight_scale_inv,
        layer.workspace,
        bias,
        state.output_size,
        state.input_size,
        state.split_k,
        state.nacc,
    )
    return out.reshape(output_shape)
