#!/usr/bin/env python3
"""Reproducible single-stream NVFP4-emulation benchmark for 4x V100.

The synthetic token prompts make prompt lengths exact and avoid tokenizer
variance.  This measures feasibility-path latency, not production serving.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Iterable


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _summary(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "p95": _percentile(samples, 0.95),
    }


def _prompt_tokens(length: int, salt: int) -> list[int]:
    # Deterministic, varied, in-vocabulary IDs for an exact-length synthetic
    # workload.  The salt avoids accidentally reusing an identical prefix.
    return [1000 + ((index * 7919 + salt * 104729) % 50000) for index in range(length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--kv-cache-mib", type=int, default=64)
    args = parser.parse_args()
    if args.kv_cache_mib <= 0:
        parser.error("--kv-cache-mib must be positive")

    os.environ.setdefault("VLLM_PLE_CPU_OFFLOAD", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        dtype="half",
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.98,
        kv_cache_memory_bytes=args.kv_cache_mib << 20,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        moe_backend="emulation",
        disable_log_stats=False,
    )
    startup_seconds = time.perf_counter() - started

    salt = 0

    def run_case(prompt_tokens: int, output_tokens: int) -> dict[str, float | int]:
        nonlocal salt
        salt += 1
        prompt = TokensPrompt(prompt_token_ids=_prompt_tokens(prompt_tokens, salt))
        params = SamplingParams(
            temperature=0,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        case_started = time.perf_counter()
        result = llm.generate([prompt], params, use_tqdm=False)[0]
        wall_seconds = time.perf_counter() - case_started
        generated = len(result.outputs[0].token_ids)
        metrics = result.metrics
        ttft_seconds = (
            float(metrics.first_token_latency)
            if metrics is not None and metrics.first_token_latency > 0
            else wall_seconds
        )
        decode_seconds = (
            float(metrics.last_token_ts - metrics.first_token_ts)
            if metrics is not None
            and metrics.last_token_ts > metrics.first_token_ts
            and generated > 1
            else 0.0
        )
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": generated,
            "wall_seconds": wall_seconds,
            "ttft_seconds": ttft_seconds,
            "prefill_tokens_per_second": prompt_tokens / ttft_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                (generated - 1) / decode_seconds if decode_seconds > 0 else 0.0
            ),
        }

    def report_sample(kind: str, sample: dict[str, float | int]) -> None:
        print(
            "NVFP4_BENCHMARK_SAMPLE_JSON="
            + json.dumps({"kind": kind, **sample}, sort_keys=True),
            flush=True,
        )

    warmup = run_case(16, 2)
    report_sample("initial_warmup", warmup)
    raw_cases: list[dict[str, float | int | str]] = []
    # Put the longest, highest-memory prefill last. If it cannot fit in the
    # narrow NVFP4-emulation headroom, the log still retains useful lower-risk
    # prefill and decode samples.
    for prompt_tokens, output_tokens, case_name in (
        (32, 1, "prefill_32"),
        (128, 1, "prefill_128"),
        (16, args.decode_tokens, f"decode_16x{args.decode_tokens}"),
        (240, 1, "prefill_240"),
    ):
        case_warmup = run_case(prompt_tokens, output_tokens)
        report_sample(f"{case_name}_warmup", case_warmup)
        for _ in range(args.repeats):
            sample = {
                "case": case_name,
                **run_case(prompt_tokens, output_tokens),
            }
            raw_cases.append(sample)
            report_sample("measured", sample)

    summaries = {}
    for case_name in sorted({str(case["case"]) for case in raw_cases}):
        matching = [case for case in raw_cases if case["case"] == case_name]
        summaries[case_name] = {
            "samples": len(matching),
            "wall_seconds": _summary(float(case["wall_seconds"]) for case in matching),
            "ttft_seconds": _summary(float(case["ttft_seconds"]) for case in matching),
            "prefill_tokens_per_second": _summary(
                float(case["prefill_tokens_per_second"]) for case in matching
            ),
            "decode_tokens_per_second": _summary(
                float(case["decode_tokens_per_second"])
                for case in matching
                if float(case["decode_tokens_per_second"]) > 0
            )
            if any(float(case["decode_tokens_per_second"]) > 0 for case in matching)
            else None,
        }

    print(
        "NVFP4_BENCHMARK_JSON="
        + json.dumps(
            {
                "configuration": {
                    "tensor_parallel_size": 4,
                    "dtype": "float16",
                    "max_model_len": 256,
                    "max_num_seqs": 1,
                    "kv_cache_memory_bytes_per_gpu": args.kv_cache_mib << 20,
                    "enforce_eager": True,
                    "prefix_caching": False,
                    "mtp": False,
                    "moe_backend": "emulation",
                    "prompt_kind": "deterministic synthetic token IDs",
                },
                "startup_seconds": startup_seconds,
                "warmup": warmup,
                "raw_cases": raw_cases,
                "summaries": summaries,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
