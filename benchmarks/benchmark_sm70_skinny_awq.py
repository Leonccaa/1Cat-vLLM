# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the generic SM70 Skinny AWQ core with selected base kernels.

The defaults are Qwen3.6-27B TP4 rank-local dense shapes. The reported
effective bandwidth counts logical packed AWQ codes plus the scale and zero
bias bytes consumed by Skinny; it is not a hardware DRAM counter.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization import sm70_skinny

DEFAULT_SHAPES = (
    (8704, 5120),
    (5120, 4352),
    (4096, 5120),
    (5120, 1536),
)
DEFAULT_ROWS = (1, 3, 5, 8, 9, 16, 17)


def _parse_pair(value: str) -> tuple[int, int]:
    fields = value.lower().split("x")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("expected NxK")
    return int(fields[0]), int(fields[1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=_parse_pair, action="append")
    parser.add_argument("--rows", type=int, action="append")
    parser.add_argument(
        "--layout",
        choices=("simt", "qpn", "both"),
        default="both",
    )
    parser.add_argument(
        "--base",
        choices=("turbomind", "marlin", "both"),
        default="both",
        help="Base kernel(s) to measure in the same process.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _require_sm70(device: torch.device, bases: tuple[str, ...]) -> None:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 0):
        raise RuntimeError(f"Expected exact SM70, got {capability}.")
    names = ["skinny_awq_gemm_simt", "skinny_awq_gemm_qpn"]
    if "turbomind" in bases:
        names.extend(("awq_sm70_prepare", "awq_gemm_sm70_out"))
    if "marlin" in bases:
        names.extend(("gptq_marlin_repack", "marlin_gemm"))
    for name in names:
        if not hasattr(torch.ops._C, name):
            raise RuntimeError(f"Missing required op _C::{name}.")


def _random_awq(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n % 32 or k % sm70_skinny.AWQ_GROUP_SIZE:
        raise ValueError(f"Skinny benchmark shape must align to N32/K128: {n}x{k}")
    qweight_bytes = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8, device=device)
    qweight = qweight_bytes.contiguous().view(torch.int32)
    qzero_bytes = torch.randint(
        0,
        256,
        (k // sm70_skinny.AWQ_GROUP_SIZE, n // 2),
        dtype=torch.uint8,
        device=device,
    )
    qzeros = qzero_bytes.contiguous().view(torch.int32)
    scales = (
        torch.rand(
            (k // sm70_skinny.AWQ_GROUP_SIZE, n),
            dtype=torch.float16,
            device=device,
        )
        * 0.05
        + 0.001
    )
    return qweight, scales, qzeros


def _time_call(
    fn: Callable[[], torch.Tensor | None],
    warmup: int,
    iters: int,
    batch_repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / batch_repeats)
    ordered = sorted(samples)
    return {
        "mean_us": statistics.fmean(samples),
        "min_us": ordered[0],
        "p50_us": ordered[len(ordered) // 2],
        "p90_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
    }


def _prepare_turbomind_base(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    n: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    tm_weight, tm_scales, meta = sm70_ops.awq_sm70_prepare(
        qweight, scales, qzeros, 128, False
    )
    k_ld, q_ld = int(meta[0]), int(meta[1])

    def apply(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
        sm70_ops.awq_gemm_sm70_out(
            out,
            x,
            tm_weight,
            tm_scales,
            128,
            k_ld,
            q_ld,
            False,
        )
        return out

    return apply


def _prepare_marlin_base(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    n: int,
    k: int,
) -> tuple[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Prepare the real AWQMarlin weight-conversion and kernel path at TP1."""
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMarlinLinearMethod,
    )

    config = AWQMarlinConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
    )
    method = AWQMarlinLinearMethod(config)
    layer = torch.nn.Module()
    old_base = os.environ.get("VLLM_SM70_QUANT_BACKEND")
    old_skinny = os.environ.get("VLLM_SM70_SKINNY")
    os.environ["VLLM_SM70_QUANT_BACKEND"] = "marlin"
    os.environ["VLLM_SM70_SKINNY"] = "off"
    try:
        with (
            patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(
                parameter, "get_tensor_model_parallel_world_size", return_value=1
            ),
        ):
            method.create_weights(
                layer,
                k,
                [n],
                k,
                n,
                torch.float16,
                weight_loader=lambda *args, **kwargs: None,
            )
            layer.to(qweight.device)
            layer.qweight.data.copy_(qweight)
            layer.qzeros.data.copy_(qzeros)
            layer.scales.data.copy_(scales)
            method.process_weights_after_loading(layer)
    finally:
        if old_base is None:
            os.environ.pop("VLLM_SM70_QUANT_BACKEND", None)
        else:
            os.environ["VLLM_SM70_QUANT_BACKEND"] = old_base
        if old_skinny is None:
            os.environ.pop("VLLM_SM70_SKINNY", None)
        else:
            os.environ["VLLM_SM70_SKINNY"] = old_skinny

    kernel_name = type(method.kernel).__name__
    if "Marlin" not in kernel_name:
        raise RuntimeError(f"Expected a Marlin base, selected {kernel_name}.")

    def apply(x: torch.Tensor) -> torch.Tensor:
        return method.kernel.apply_weights(layer, x, None)

    return kernel_name, apply


@torch.inference_mode()
def _run_shape(
    n: int,
    k: int,
    rows: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iters: int,
    batch_repeats: int,
    bases: tuple[str, ...],
) -> list[dict[str, object]]:
    qweight, scales, qzeros = _random_awq(n, k, device)
    state = sm70_skinny.prepare_awq_state(qweight, scales, qzeros, 128)
    base_apply: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    if "turbomind" in bases:
        base_apply["turbomind"] = _prepare_turbomind_base(qweight, scales, qzeros, n)
    if "marlin" in bases:
        marlin_name, marlin_apply = _prepare_marlin_base(qweight, scales, qzeros, n, k)
        base_apply["marlin"] = marlin_apply
    else:
        marlin_name = ""
    results: list[dict[str, object]] = []
    logical_bytes = n * k // 2 + n * (k // 128) * 4

    for m in rows:
        values = torch.arange(m * k, dtype=torch.int32, device=device)
        x = ((values.remainder(31) - 15).to(torch.float16) * 1e-3).view(m, k)
        candidates: list[tuple[str, Callable[[], torch.Tensor]]] = []
        if m <= 3 and state.has_simt:
            candidates.append(
                (
                    "skinny_simt",
                    partial(
                        sm70_ops.skinny_awq_gemm_simt,
                        x,
                        state.codes,
                        state.scales,
                        state.biases,
                        128,
                    ),
                )
            )
        if 4 <= m <= 16 and state.has_qpn:
            candidates.append(
                (
                    "skinny_qpn",
                    partial(
                        sm70_ops.skinny_awq_gemm_qpn,
                        x,
                        state.qpn_codes,
                        state.qpn_scales,
                        state.qpn_biases,
                        128,
                        n,
                    ),
                )
            )

        references: dict[str, torch.Tensor] = {}
        base_timings: dict[str, dict[str, float]] = {}
        for base_name, apply_base in base_apply.items():
            run_base = partial(apply_base, x)
            references[base_name] = run_base()
            torch.cuda.synchronize(device)
            timing = _time_call(run_base, warmup, iters, batch_repeats)
            base_timings[base_name] = timing
            results.append(
                {
                    "backend": base_name,
                    "kernel": marlin_name if base_name == "marlin" else "TurboMind",
                    "m": m,
                    "n": n,
                    "k": k,
                    **timing,
                    "effective_weight_gbps": logical_bytes / timing["mean_us"] / 1e3,
                }
            )

        for backend, candidate in candidates:
            actual = candidate()
            torch.cuda.synchronize(device)
            timing = _time_call(candidate, warmup, iters, batch_repeats)
            row: dict[str, object] = {
                "backend": backend,
                "m": m,
                "n": n,
                "k": k,
                **timing,
                "effective_weight_gbps": logical_bytes / timing["mean_us"] / 1e3,
            }
            for base_name, reference in references.items():
                diff = (actual.float() - reference.float()).abs()
                denominator = reference.float().abs().max().clamp(min=1e-6)
                row[f"max_abs_diff_vs_{base_name}"] = float(diff.max().item())
                row[f"relative_max_error_vs_{base_name}"] = float(
                    (diff.max() / denominator).item()
                )
                row[f"speedup_vs_{base_name}"] = (
                    base_timings[base_name]["mean_us"] / timing["mean_us"]
                )
            results.append(row)
    return results


def main() -> int:
    args = _parse_args()
    if args.warmup < 0 or args.iters < 1 or args.batch_repeats < 1:
        raise ValueError("warmup must be non-negative; iters/repeats must be positive.")
    device = torch.device(args.device)
    bases = ("turbomind", "marlin") if args.base == "both" else (args.base,)
    _require_sm70(device, bases)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    os.environ["VLLM_SM70_SKINNY_AWQ_LAYOUT"] = args.layout
    shapes = tuple(args.shape or DEFAULT_SHAPES)
    rows = tuple(args.rows or DEFAULT_ROWS)

    results: list[dict[str, object]] = []
    for n, k in shapes:
        results.extend(
            _run_shape(
                n,
                k,
                rows,
                device,
                args.warmup,
                args.iters,
                args.batch_repeats,
                bases,
            )
        )
    payload = {
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "layout": args.layout,
        "bases": bases,
        "effective_bandwidth_definition": (
            "(packed codes + FP16 scales + FP16 zero biases) / CUDA-event time"
        ),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
