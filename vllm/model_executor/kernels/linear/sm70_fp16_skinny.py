# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 small-M FP16 dense GEMM for verify-width decode (M = 1..48 rows).

Native op: ``csrc/sm70_turbomind/ops/fp16_skinny_gemm_sm70.cu``. It reads the
original row-major checkpoint weight once per call for all rows (m8n8k4 row
blocks share each weight fragment), splits K over warps and optionally over
CTAs, and reduces every partial sum in a fixed order (no atomics), so the
result is deterministic and independent of M for a fixed K partition.

Everything is opt-in through ``VLLM_SM70_FP16_SKINNY``; admitted row counts are
``[VLLM_SM70_FP16_SKINNY_MIN_M, VLLM_SM70_FP16_SKINNY_MAX_M]`` (default 17..48,
the MTP4 verifier widths above the existing M <= 16 lanes). Callers keep their
own fallback: :func:`skinny_linear` returns ``None`` when not admitted.

Launch plans ``(ng, warps_n, warps_k, splits)`` come from a small per-(N, K)
table; ``VLLM_SM70_FP16_SKINNY_CONFIG`` overrides entries for tuning, e.g.
``"4120x2560=1,1,2,1;336x10240@17-32=1,1,4,6;1x2560=off"``. The K partition is
``warps_k * splits`` contiguous ranges; ``ng`` (8-column groups per warp) and
``warps_n`` (column warps per CTA) only change the work assignment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MAX_ROWS = 48
_MAX_WARPS = 8
_MAX_SPLITS = 64
_OP = "fp16_skinny_gemm_sm70_out"
_OP2 = "fp16_skinny_gemm2_sm70_out"


class SkinnyConfig(NamedTuple):
    ng: int  # 8-column groups per warp (1, 2 or 4; 4 only for M <= 24)
    warps_n: int  # warps splitting the CTA's columns
    warps_k: int  # warps splitting K inside a CTA
    splits: int  # CTAs splitting K (> 1 adds a fixed-order reduce kernel)


# Untuned starting plans for the Qwen3.8 TP4 per-rank shapes (inferred from
# the V100 occupancy model, not measured): one config per (N, K) for every M,
# so the arithmetic is batch invariant across verifier widths. The CTA count
# is chosen to stay within one resident wave at 128 registers/thread (16
# warps per SM) where the shape allows.
_DEFAULT_PLANS: dict[tuple[int, int], SkinnyConfig] = {
    # GDN in_proj_qkvz + in_proj_ba fused (two weights, one launch).
    (4120, 2560): SkinnyConfig(1, 1, 2, 1),
    (4096, 2560): SkinnyConfig(1, 1, 2, 1),  # GDN in_proj_qkvz alone
    (24, 2560): SkinnyConfig(1, 1, 8, 1),  # GDN in_proj_ba alone
    (3584, 2560): SkinnyConfig(1, 1, 2, 1),  # QSA qkv_proj
    (2560, 1536): SkinnyConfig(1, 1, 4, 1),  # GDN out_proj, QSA o_proj
    (640, 2560): SkinnyConfig(1, 1, 8, 1),  # QSA indexer, MTP fc_*
    (512, 2560): SkinnyConfig(1, 1, 8, 1),  # MoE router
    (320, 2560): SkinnyConfig(1, 1, 8, 1),  # shared expert gate_up
    (2560, 160): SkinnyConfig(1, 1, 2, 1),  # shared expert down
    (1, 2560): SkinnyConfig(1, 1, 8, 1),  # shared_expert_gate
    (336, 10240): SkinnyConfig(1, 1, 4, 6),  # HC down + inject
    (320, 10240): SkinnyConfig(1, 1, 4, 6),  # final mixer down
    (10240, 320): SkinnyConfig(1, 2, 1, 1),  # HC / final mixer up
    (10240, 2560): SkinnyConfig(1, 2, 2, 1),  # PLE key
    (2560, 2560): SkinnyConfig(1, 1, 4, 1),  # PLE value
    (62080, 2560): SkinnyConfig(1, 2, 2, 1),  # LM head (vocab shard)
}


def _parse_config_value(text: str) -> SkinnyConfig | None:
    if text.strip().lower() == "off":
        return None
    fields = [int(v) for v in text.split(",")]
    if len(fields) != 4:
        raise ValueError("expected ng,warps_n,warps_k,splits")
    return SkinnyConfig(*fields)


@lru_cache(maxsize=8)
def _parse_overrides(
    spec: str,
) -> tuple[tuple[int, int, int, int, SkinnyConfig | None], ...]:
    """Parse ``NxK[@lo-hi]=ng,wn,wk,s|off`` entries separated by ';'."""
    entries = []
    for raw in spec.split(";"):
        item = raw.strip()
        if not item:
            continue
        try:
            key, value = item.split("=", 1)
            rows = (1, MAX_ROWS)
            if "@" in key:
                key, row_text = key.split("@", 1)
                lo, hi = row_text.split("-", 1)
                rows = (int(lo), int(hi))
            n_text, k_text = key.lower().split("x", 1)
            entries.append(
                (int(n_text), int(k_text), rows[0], rows[1], _parse_config_value(value))
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid VLLM_SM70_FP16_SKINNY_CONFIG entry {item!r}: {exc}"
            ) from exc
    return tuple(entries)


def _heuristic_config(n: int, k: int) -> SkinnyConfig:
    """Plan for shapes without a table entry (same for every M)."""
    steps = k // 16
    groups = (n + 7) // 8
    if groups >= 1024:
        warps_n, warps_k, splits = 2, 2, 1
    elif groups >= 256:
        warps_n, warps_k, splits = 1, 2, 1
    elif groups >= 64:
        warps_n, warps_k, splits = 1, 8, 1
    else:
        # Few column groups: split K across CTAs to cover the SMs.
        warps_n, warps_k = 1, 4
        splits = max(1, min(_MAX_SPLITS, 240 // max(groups, 1)))
    # Keep at least two 64-element steps per partition.
    while warps_k > 1 and warps_k * splits * 8 > steps:
        warps_k //= 2
    while splits > 1 and warps_k * splits * 8 > steps:
        splits -= 1
    return SkinnyConfig(1, warps_n, warps_k, max(splits, 1))


def config_is_valid(config: SkinnyConfig, n: int, k: int, m: int) -> bool:
    return bool(
        config.ng in (1, 2, 4)
        and (config.ng != 4 or m <= 24)
        and config.warps_n >= 1
        and config.warps_k >= 1
        and config.warps_n * config.warps_k <= _MAX_WARPS
        and 1 <= config.splits <= _MAX_SPLITS
        and config.splits * config.warps_k <= k // 16
        and n >= 1
        and 1 <= m <= MAX_ROWS
    )


def select_config(n: int, k: int, m: int) -> SkinnyConfig | None:
    """Launch plan for (N, K, M); ``None`` when the shape is switched off."""
    spec = envs.VLLM_SM70_FP16_SKINNY_CONFIG or ""
    if spec:
        for on, ok, lo, hi, config in _parse_overrides(spec):
            if on == n and ok == k and lo <= m <= hi:
                if config is not None and not config_is_valid(config, n, k, m):
                    raise ValueError(
                        f"VLLM_SM70_FP16_SKINNY_CONFIG plan {tuple(config)} is "
                        f"invalid for N={n} K={k} M={m}"
                    )
                return config
    config = _DEFAULT_PLANS.get((n, k)) or _heuristic_config(n, k)
    if not config_is_valid(config, n, k, m):
        config = _heuristic_config(n, k)
    return config


def row_range() -> tuple[int, int]:
    lo = max(1, int(envs.VLLM_SM70_FP16_SKINNY_MIN_M))
    hi = min(MAX_ROWS, int(envs.VLLM_SM70_FP16_SKINNY_MAX_M))
    return lo, hi


def rows_admitted(m: int) -> bool:
    if not envs.VLLM_SM70_FP16_SKINNY:
        return False
    lo, hi = row_range()
    return lo <= m <= hi


@lru_cache(maxsize=1)
def native_available() -> bool:
    return hasattr(torch.ops._C, _OP) and hasattr(torch.ops._C, _OP2)


def _is_sm70() -> bool:
    return current_platform.is_device_capability(70)


def _device_ok(t: torch.Tensor) -> bool:
    return t.is_cuda


def _launch(
    out: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor,
    config: SkinnyConfig,
) -> None:
    torch.ops._C.fp16_skinny_gemm_sm70_out(out, x2d, weight, bias, workspace, *config)


def _launch2(
    outs: tuple[torch.Tensor, ...],
    x2d: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    workspace: torch.Tensor,
    config: SkinnyConfig,
) -> None:
    torch.ops._C.fp16_skinny_gemm2_sm70_out(*outs, x2d, w0, w1, workspace, *config)


def _weight_ok(weight: torch.Tensor, x: torch.Tensor) -> bool:
    return bool(
        weight.ndim == 2
        and weight.dtype == torch.float16
        and _device_ok(weight)
        and weight.device == x.device
        and weight.shape[0] >= 1
        and weight.shape[1] == x.shape[-1]
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


def _input_2d(x: torch.Tensor) -> torch.Tensor:
    """Row-major view with unit column stride and 16-byte aligned rows."""
    x2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
    m = x2d.shape[0]
    aligned = (
        x2d.stride(-1) == 1
        and (m == 1 or x2d.stride(0) % 8 == 0)
        and x2d.data_ptr() % 16 == 0
    )
    return x2d if aligned else x2d.clone(memory_format=torch.contiguous_format)


def _common_ok(x: torch.Tensor) -> bool:
    return bool(
        x.ndim >= 2
        and x.dtype == torch.float16
        and _device_ok(x)
        and x.shape[-1] >= 16
        and x.shape[-1] % 16 == 0
        and not envs.VLLM_BATCH_INVARIANT
        # FP32 accumulation only: an explicit FP16-accumulation request keeps
        # the cuBLAS route that honors it.
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
        and _is_sm70()
        and native_available()
    )


def _workspace(x2d: torch.Tensor, config: SkinnyConfig, n: int) -> torch.Tensor:
    numel = config.splits * x2d.shape[0] * n if config.splits > 1 else 0
    return torch.empty((numel,), dtype=torch.float32, device=x2d.device)


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor | None:
    """``F.linear(x, weight, bias)`` on the skinny kernel, or ``None``."""
    if not envs.VLLM_SM70_FP16_SKINNY or x.ndim < 2:
        return None
    m = x.numel() // max(x.shape[-1], 1)
    if not rows_admitted(m) or not _common_ok(x) or not _weight_ok(weight, x):
        return None
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if bias is not None and not (
        bias.dtype == torch.float16
        and bias.device == x.device
        and bias.shape == (n,)
        and bias.is_contiguous()
    ):
        return None
    config = select_config(n, k, m)
    if config is None:
        return None
    x2d = _input_2d(x)
    out = x2d.new_empty((m, n))
    _launch(out, x2d, weight, bias, _workspace(x2d, config, n), config)
    logger.info_once(
        "SM70 FP16 skinny GEMM route: N=%d K=%d plan=%s.", n, k, tuple(config)
    )
    return out if x.ndim == 2 else out.reshape(*x.shape[:-1], n)


def skinny_linear_or_fallback(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    out = skinny_linear(x, weight, bias)
    if out is None:
        return torch.nn.functional.linear(x, weight, bias)
    return out


def skinny_dual_linear(
    x: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    split0: int,
    split1: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Two projections of one 2D input in one launch, each split in two.

    Returns contiguous ``(y0[:, :split0], y0[:, split0:], y1[:, :split1],
    y1[:, split1:])`` with ``yi = x @ wi.T``, or ``None`` when not admitted.
    """
    if not envs.VLLM_SM70_FP16_SKINNY or x.ndim != 2:
        return None
    m = x.shape[0]
    if not (
        rows_admitted(m)
        and _common_ok(x)
        and _weight_ok(w0, x)
        and _weight_ok(w1, x)
        and 0 < split0 < w0.shape[0]
        and 0 < split1 < w1.shape[0]
    ):
        return None
    n0, n1, k = int(w0.shape[0]), int(w1.shape[0]), int(x.shape[1])
    config = select_config(n0 + n1, k, m)
    if config is None:
        return None
    x2d = _input_2d(x)
    outs = tuple(
        x2d.new_empty((m, cols)) for cols in (split0, n0 - split0, split1, n1 - split1)
    )
    _launch2(outs, x2d, w0, w1, _workspace(x2d, config, n0 + n1), config)
    logger.info_once(
        "SM70 FP16 skinny dual GEMM route: N=%d+%d K=%d plan=%s.",
        n0,
        n1,
        k,
        tuple(config),
    )
    return outs  # type: ignore[return-value]


def _sm70_fp16_skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return skinny_linear_or_fallback(x, weight, bias)


def _sm70_fp16_skinny_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


# Opaque to torch.compile: the row-count decision happens per call, so a
# dynamically compiled graph never bakes in one M-dependent branch.
direct_register_custom_op(
    op_name="sm70_fp16_skinny_linear",
    op_func=_sm70_fp16_skinny_linear,
    fake_impl=_sm70_fp16_skinny_linear_fake,
)


__all__ = [
    "MAX_ROWS",
    "SkinnyConfig",
    "config_is_valid",
    "native_available",
    "row_range",
    "rows_admitted",
    "select_config",
    "skinny_dual_linear",
    "skinny_linear",
    "skinny_linear_or_fallback",
]
