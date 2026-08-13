#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure exact-length OpenAI completion streaming latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import regex as re

_SPEC_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def _post_json(url: str, payload: dict[str, object], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _tokenize(base_url: str, model: str, prompt: str, timeout: float) -> list[int]:
    with _post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": prompt},
        timeout,
    ) as response:
        return json.loads(response.read())["tokens"]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[position]


def _build_prompt_ids(
    base_url: str, model: str, input_len: int, timeout: float
) -> list[int]:
    prefix = (
        "你是一名负责大模型推理性能的工程师。请阅读材料，最后给出严谨的"
        "瓶颈分析和下一步优化建议。\n\n材料：\n"
    )
    paragraph = (
        "在单请求 decode 中，需要分别观察矩阵乘、注意力、跨卡通信和调度"
        "等待。任何优化都必须保持相同采样参数和数值语义，并通过完整输出质量"
        "检查。性能结论应区分端到端 TPOT、CUDA Graph 墙钟、kernel service "
        "time 以及 profiler 本身的开销。\n"
    )
    suffix = (
        "\n问题：请基于以上材料总结三个最重要的瓶颈，解释判断依据，并提出"
        "不会牺牲输出质量的优化顺序。"
    )
    prefix_ids = _tokenize(base_url, model, prefix, timeout)
    context_ids = _tokenize(base_url, model, paragraph * 80, timeout)
    suffix_ids = _tokenize(base_url, model, suffix, timeout)
    context_len = input_len - len(prefix_ids) - len(suffix_ids)
    if context_len < 0 or len(context_ids) < context_len:
        raise ValueError("Unable to construct the requested input length")
    prompt_ids = prefix_ids + context_ids[:context_len] + suffix_ids
    if len(prompt_ids) != input_len:
        raise AssertionError(
            f"Expected {input_len} prompt tokens, got {len(prompt_ids)}"
        )
    return prompt_ids


def _read_spec_metrics(base_url: str, timeout: float) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as response:
            rendered = response.read().decode()
    except OSError:
        return {}

    values: dict[str, float] = {}
    for name in _SPEC_METRICS:
        pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$")
        values[name] = sum(
            float(match.group(1))
            for line in rendered.splitlines()
            if (match := pattern.match(line)) is not None
        )
    return values


def _run_once(
    base_url: str,
    payload: dict[str, object],
    input_len: int,
    timeout: float,
) -> dict[str, object]:
    started = time.perf_counter()
    token_times: list[float] = []
    text_parts: list[str] = []
    usage = None
    finish_reason = None
    with _post_json(f"{base_url}/v1/completions", payload, timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("text", "")
                if delta:
                    token_times.append(time.perf_counter())
                    text_parts.append(delta)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()

    intervals_ms = [
        (right - left) * 1000
        for left, right in zip(token_times, token_times[1:], strict=False)
    ]
    ttft_s = token_times[0] - started if token_times else None
    decode_s = finished - token_times[0] if token_times else None
    completion_tokens = (
        int(usage["completion_tokens"])
        if usage is not None and "completion_tokens" in usage
        else len(token_times)
    )
    decode_tokens = max(completion_tokens - 1, 0)
    wall_s = finished - started
    return {
        "usage": usage,
        "stream_chunks": len(token_times),
        "finish_reason": finish_reason,
        "generation_wall_ms": wall_s * 1000,
        "ttft_ms": ttft_s * 1000 if ttft_s is not None else None,
        "prefill_proxy_tok_s": input_len / ttft_s if ttft_s else None,
        "decode_wall_ms": decode_s * 1000 if decode_s is not None else None,
        "decode_tok_s": decode_tokens / decode_s if decode_s else None,
        "output_tok_s": completion_tokens / wall_s,
        "e2e_total_tok_s": (input_len + completion_tokens) / wall_s,
        "tpot_mean_ms": statistics.mean(intervals_ms) if intervals_ms else None,
        "tpot_p50_ms": _percentile(intervals_ms, 0.50) if intervals_ms else None,
        "tpot_p90_ms": _percentile(intervals_ms, 0.90) if intervals_ms else None,
        "tpot_p99_ms": _percentile(intervals_ms, 0.99) if intervals_ms else None,
        "output": "".join(text_parts),
    }


def _median_summary(runs: list[dict[str, object]]) -> dict[str, float]:
    fields = (
        "generation_wall_ms",
        "ttft_ms",
        "prefill_proxy_tok_s",
        "decode_wall_ms",
        "decode_tok_s",
        "output_tok_s",
        "e2e_total_tok_s",
        "tpot_mean_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p99_ms",
    )
    return {
        field: statistics.median(
            float(run[field]) for run in runs if run[field] is not None
        )
        for field in fields
        if any(run[field] is not None for run in runs)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup_runs < 0:
        parser.error("--repeats must be positive and --warmup-runs nonnegative")
    prompt_ids = _build_prompt_ids(
        args.base_url, args.model, args.input_len, args.timeout
    )
    payload = {
        "model": args.model,
        "prompt": prompt_ids,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "ignore_eos": args.ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    warmup_payload = dict(payload)
    warmup_payload["max_tokens"] = min(args.max_tokens, args.warmup_output_tokens)
    for _ in range(args.warmup_runs):
        _run_once(args.base_url, warmup_payload, args.input_len, args.timeout)

    metrics_before = _read_spec_metrics(args.base_url, args.timeout)
    runs = [
        _run_once(args.base_url, payload, args.input_len, args.timeout)
        for _ in range(args.repeats)
    ]
    metrics_after = _read_spec_metrics(args.base_url, args.timeout)
    metric_delta = {
        name: metrics_after[name] - metrics_before.get(name, 0.0)
        for name in _SPEC_METRICS
        if name in metrics_after
    }
    drafted = metric_delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = metric_delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    result = {
        "contract": {
            "input_len": args.input_len,
            "max_output_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "warmup_runs": args.warmup_runs,
            "warmup_output_tokens": args.warmup_output_tokens,
            "repeats": args.repeats,
        },
        "runs": runs,
        "median": _median_summary(runs),
        "speculative_metrics_delta": metric_delta,
        "speculative_acceptance_rate": accepted / drafted if drafted > 0 else None,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
