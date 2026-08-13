# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared load-time residency policy for the SM70 Skinny overlay.

This module deliberately contains no quantization-format knowledge.  AWQ and
NVFP4 supply route-specific benchmark callables and release callbacks, while
this module owns CUDA timing, TP consensus, cache consistency, and the common
keep/drop decision loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ResidencyKey = tuple[int, int, str]
ApplyFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class ResidencyDecision:
    """Per-shape verdict on whether one overlay route earns its VRAM."""

    roi: float  # microseconds saved per MiB of overlay
    mib: float
    saved_us: float
    keep: bool


@dataclass(frozen=True)
class RouteBenchmark:
    """The two calls compared for one format-specific execution route."""

    base_apply: ApplyFn
    skinny_apply: ApplyFn


# Volta's L2 is 6 MiB; 24 MiB of dirty traffic evicts it comfortably.
_L2_FLUSH_BYTES = 24 << 20
_l2_flush_buffer: torch.Tensor | None = None


def _as_cuda_device(device: torch.device | int | None) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"SM70 residency timing requires CUDA, got {resolved}.")
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _flush_l2(device: torch.device) -> None:
    """Evict L2 so the next call measures a cold model-weight stream."""
    global _l2_flush_buffer
    if _l2_flush_buffer is None or _l2_flush_buffer.device != device:
        _l2_flush_buffer = torch.empty(
            _L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device
        )
    _l2_flush_buffer.zero_()


def time_apply(
    fn: ApplyFn,
    iterations: int = 12,
    device: torch.device | int | None = None,
) -> float:
    """Return the median L2-cold CUDA execution time in microseconds."""
    cuda_device = _as_cuda_device(device)
    with torch.cuda.device(cuda_device):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(cuda_device)
        samples: list[float] = []
        for _ in range(iterations):
            _flush_l2(cuda_device)
            torch.cuda.synchronize(cuda_device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def agree_across_tp(
    value: float | None,
    device: torch.device | int | str | None = None,
) -> float | None:
    """Return the least optimistic valid value across the TP group.

    An unavailable distributed/TP group is a legitimate early-load state, so
    it uses the local value.  Once a TP collective is entered, however, every
    exception must propagate.  Silently falling back after an NCCL failure can
    desynchronize ranks and offset every later collective.
    """
    valid = value is not None and math.isfinite(value)
    local_value = value if valid else 0.0
    local_result = local_value if valid else None

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_result

    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_is_initialized,
    )

    if not tensor_model_parallel_is_initialized():
        return local_result

    group = get_tp_group()
    if group.world_size <= 1:
        return local_result

    if device is None:
        tensor_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        tensor_device = torch.device("cuda", device)
    else:
        tensor_device = torch.device(device)
    tensor = torch.tensor(
        [1.0 if valid else 0.0, local_value],
        dtype=torch.float64,
        device=tensor_device,
    )
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=group.device_group
    )
    if tensor[0].item() < 1.0:
        return None
    return float(tensor[1].item())


def all_ranks_succeeded(
    local_ok: bool,
    device: torch.device | int | str | None = None,
) -> bool:
    """Return true only when every initialized TP rank succeeded."""
    return agree_across_tp(0.0 if local_ok else None, device=device) is not None


def apply_route_policy(
    *,
    format_name: str,
    output_size: int,
    input_size: int,
    routes: Sequence[str],
    device: torch.device,
    overlay_mib: float,
    min_roi: float,
    force_on: bool,
    decisions: MutableMapping[ResidencyKey, ResidencyDecision],
    make_benchmark: Callable[[str], RouteBenchmark],
    release_route: Callable[[str], None],
) -> set[str]:
    """Apply the shared residency gate and return the routes left resident."""
    retained = set(routes)
    if not retained:
        return retained
    if force_on:
        logger.info_once(
            "VLLM_SM70_SKINNY=on keeps every self-check-passing %s route; "
            "the performance residency gate is bypassed.",
            format_name,
        )
        return retained

    for route in routes:
        key = (output_size, input_size, route)
        decision = decisions.get(key)

        # A mixed cache hit/miss would otherwise send only some ranks into the
        # measurement collectives.  Make every rank remeasure in that case.
        if not all_ranks_succeeded(decision is not None, device=device):
            base_us: float | None = None
            skinny_us: float | None = None
            local_saved: float | None = None
            try:
                benchmark = make_benchmark(route)
                base_us = time_apply(benchmark.base_apply, device=device)
                skinny_us = time_apply(benchmark.skinny_apply, device=device)
                local_saved = base_us - skinny_us
            except Exception:
                logger.exception(
                    "SM70 Skinny %s %s local residency measurement failed "
                    "for N=%d K=%d.",
                    format_name,
                    route,
                    output_size,
                    input_size,
                )

            # A failed rank still participates so healthy peers cannot hang in
            # the collective.  Collective failures themselves are not caught.
            saved = agree_across_tp(local_saved, device=device)
            if saved is None:
                logger.warning_once(
                    "SM70 Skinny %s %s residency measurement failed on at "
                    "least one TP rank for N=%d K=%d; keeping the overlay on "
                    "every rank.",
                    format_name,
                    route,
                    output_size,
                    input_size,
                )
                continue

            assert base_us is not None and skinny_us is not None
            roi = saved / overlay_mib if overlay_mib > 0 else 0.0
            decision = ResidencyDecision(
                roi=roi,
                mib=overlay_mib,
                saved_us=saved,
                keep=roi >= min_roi,
            )
            decisions[key] = decision
            logger.info(
                "SM70 Skinny %s residency %s N=%d K=%d: base=%.1fus "
                "skinny=%.1fus saved=%.1fus overlay=%.1fMiB "
                "roi=%.3fus/MiB -> %s",
                format_name,
                route,
                output_size,
                input_size,
                base_us,
                skinny_us,
                saved,
                overlay_mib,
                roi,
                "keep" if decision.keep else "drop",
            )

        assert decision is not None
        if decision.keep:
            continue
        release_route(route)
        retained.remove(route)

    return retained
