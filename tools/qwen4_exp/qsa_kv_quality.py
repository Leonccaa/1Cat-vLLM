#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze and run matched token-ID quality checks for QSA KV cache modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Input is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    path.chmod(0o600)


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    path = output.with_name(output.name + ".manifest.json")
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _chat_token_ids(
    tokenizer: Any, messages: list[dict[str, Any]], **kwargs
) -> list[int]:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    token_ids = list(token_ids)
    if not token_ids or not all(
        isinstance(token, int) and token >= 0 for token in token_ids
    ):
        raise ValueError("Chat template returned invalid token IDs")
    return token_ids


def prepare_long(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model = str(Path(args.model).resolve())
    sources = [Path(path).resolve() for path in args.case_jsonl]
    requested = list(dict.fromkeys(args.case_id))
    if not requested:
        raise ValueError("At least one --case-id is required")
    cases: dict[str, tuple[dict[str, Any], Path]] = {}
    for source in sources:
        for case in _read_jsonl(source):
            case_id = str(case.get("id", ""))
            if not case_id or case_id in cases:
                raise ValueError(f"Missing or duplicate long-quality id: {case_id!r}")
            cases[case_id] = (case, source)
    missing = [case_id for case_id in requested if case_id not in cases]
    if missing:
        raise ValueError("Unknown long-quality ids: " + ", ".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    rows = []
    for case_id in requested:
        case, source = cases[case_id]
        prompt = str(case["prompt"])
        if _sha256_text(prompt) != case.get("prompt_sha256"):
            raise ValueError(f"Prompt SHA-256 mismatch for {case_id}")
        token_ids = _chat_token_ids(
            tokenizer,
            [{"role": "user", "content": prompt}],
        )
        rows.append(
            {
                "schema_version": 1,
                "id": case_id,
                "category": "long_retrieval",
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": _token_ids_sha256(token_ids),
                "max_tokens": min(
                    int(case["max_completion_tokens"]), args.max_tokens_cap
                ),
                "gold_answers": [str(value) for value in case["gold_answers"]],
                "score_mode": str(case.get("score_mode", "contains_all")),
                "source": {
                    "jsonl": str(source),
                    "jsonl_sha256": _sha256(source),
                    "prompt_sha256": case["prompt_sha256"],
                    "benchmark": case.get("benchmark"),
                    "task": case.get("task"),
                    "depth_percent": case.get("depth_percent"),
                },
            }
        )
    output = Path(args.output_jsonl).resolve()
    _write_jsonl(output, rows)
    _write_manifest(
        output,
        {
            "schema_version": 1,
            "purpose": "frozen-qsa-kv-long-quality-token-ids",
            "model": model,
            "source_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in sources
            ],
            "record_count": len(rows),
            "active_tokens": sum(len(row["prompt_token_ids"]) for row in rows),
            "output_sha256": _sha256(output),
            "case_ids": requested,
            "enable_thinking": False,
            "calibration_membership": False,
        },
    )


def _tool_call_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]


def _select_tool_cases(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    remaining = sorted(candidates, key=lambda row: row["content_sha256"])
    selected = []
    seen_tools: set[str] = set()
    seen_sessions: set[str] = set()
    while remaining and len(selected) < limit:
        best = max(
            range(len(remaining)),
            key=lambda index: (
                len(set(remaining[index]["gold_answers"]) - seen_tools),
                remaining[index]["session"] not in seen_sessions,
                -int(remaining[index]["content_sha256"][:16], 16),
            ),
        )
        row = remaining.pop(best)
        selected.append(row)
        seen_tools.update(row["gold_answers"])
        seen_sessions.add(row["session"])
    return selected


def prepare_tools(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model = str(Path(args.model).resolve())
    source = Path(args.heldout_jsonl).resolve()
    corpus_manifest_path = Path(args.corpus_manifest).resolve()
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    expected = corpus_manifest.get("heldout", corpus_manifest)
    if not isinstance(expected, dict):
        raise ValueError("Held-out corpus manifest entry must be an object")
    source_rows = _read_jsonl(source)
    expected_sha256 = expected.get("sha256", expected.get("corpus_sha256"))
    if expected_sha256 is not None and _sha256(source) != expected_sha256:
        raise ValueError("Held-out tool source SHA-256 does not match its manifest")
    expected_count = expected.get("record_count")
    if expected_count is not None and len(source_rows) != expected_count:
        raise ValueError("Held-out tool source count does not match its manifest")

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    candidates = []
    for source_index, source_row in enumerate(source_rows, 1):
        messages = source_row.get("messages")
        tools = source_row.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            continue
        audit = source_row.get("audit", {})
        if not isinstance(audit, dict):
            audit = {}
        source_id = str(source_row.get("id", source_index))
        source_row_sha256 = _sha256_text(
            json.dumps(
                source_row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        source_session = str(
            source_row.get("session_id")
            or audit.get("source_session_sha256")
            or source_id
        )
        source_session_sha256 = _sha256_text(source_session)
        for call_index in _tool_call_indices(messages):
            if call_index == 0:
                continue
            calls = messages[call_index].get("tool_calls") or []
            gold_names = sorted(
                {
                    str(call.get("function", {}).get("name", ""))
                    for call in calls
                    if call.get("function", {}).get("name")
                }
            )
            if not gold_names:
                continue
            prompt_messages = messages[:call_index]
            token_ids = _chat_token_ids(tokenizer, prompt_messages, tools=tools)
            if len(token_ids) > args.max_prompt_tokens:
                continue
            candidates.append(
                {
                    "schema_version": 1,
                    "id": f"tool-select-{source_row_sha256[:20]}-m{call_index}",
                    "category": "heldout_tool_selection",
                    "prompt_token_ids": token_ids,
                    "prompt_token_ids_sha256": _token_ids_sha256(token_ids),
                    "max_tokens": args.max_tokens,
                    "gold_answers": gold_names,
                    "score_mode": "contains_any",
                    "session": source_session_sha256,
                    "content_sha256": _sha256_text(f"{source_row_sha256}:{call_index}"),
                    "source": {
                        "source_id_sha256": _sha256_text(source_id),
                        "source_row_sha256": source_row_sha256,
                        "source_session_sha256": source_session_sha256,
                        "prefix_message_count": call_index,
                    },
                }
            )
    selected = _select_tool_cases(candidates, args.limit)
    if len(selected) < args.limit:
        raise ValueError(
            f"Only {len(selected)} eligible held-out tool cases for limit {args.limit}"
        )
    rows = []
    for row in selected:
        row = dict(row)
        row.pop("session")
        row.pop("content_sha256")
        rows.append(row)
    output = Path(args.output_jsonl).resolve()
    _write_jsonl(output, rows)
    _write_manifest(
        output,
        {
            "schema_version": 1,
            "purpose": "frozen-qsa-kv-heldout-tool-quality-token-ids",
            "model": model,
            "source_jsonl": str(source),
            "source_sha256": _sha256(source),
            "corpus_manifest": str(corpus_manifest_path),
            "corpus_manifest_sha256": _sha256(corpus_manifest_path),
            "record_count": len(rows),
            "active_tokens": sum(len(row["prompt_token_ids"]) for row in rows),
            "output_sha256": _sha256(output),
            "calibration_membership": False,
            "heldout_quality_derivative": True,
        },
    )


def prepare_trace(args: argparse.Namespace) -> None:
    source = Path(args.prepared_jsonl).resolve()
    rows = _read_jsonl(source)
    if not 1 <= args.source_line <= len(rows):
        raise ValueError("--source-line is outside the prepared input")
    source_row = rows[args.source_line - 1]
    token_ids = source_row.get("prompt_token_ids")
    if not isinstance(token_ids, list) or len(token_ids) < args.tokens:
        raise ValueError("Prepared trace source has too few prompt tokens")
    token_ids = token_ids[: args.tokens]
    output_row = {
        "schema_version": 1,
        "id": f"qsa-page4-trace-{args.tokens}",
        "category": "qsa_page4_trace",
        "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": _token_ids_sha256(token_ids),
        "max_tokens": 1,
        "gold_answers": [],
        "score_mode": "none",
        "source": {
            "prepared_jsonl": str(source),
            "prepared_sha256": _sha256(source),
            "source_line": args.source_line,
        },
    }
    output = Path(args.output_jsonl).resolve()
    _write_jsonl(output, [output_row])
    _write_manifest(
        output,
        {
            "schema_version": 1,
            "purpose": "frozen-qsa-page4-trace-token-ids",
            "source_jsonl": str(source),
            "source_sha256": _sha256(source),
            "record_count": 1,
            "active_tokens": len(token_ids),
            "output_sha256": _sha256(output),
            "calibration_membership": False,
        },
    )


def _score(row: dict[str, Any], text: str) -> dict[str, Any] | None:
    mode = row.get("score_mode", "none")
    if mode == "none":
        return None
    expected = [str(value) for value in row.get("gold_answers", [])]
    normalized = text.casefold()
    matched = [value for value in expected if value.casefold() in normalized]
    if mode == "contains_all":
        correct = len(matched) == len(expected)
    elif mode == "contains_any":
        correct = bool(matched)
    else:
        raise ValueError(f"Unsupported score mode: {mode}")
    return {
        "mode": mode,
        "expected": expected,
        "matched": matched,
        "correct": correct,
        "score": len(matched) / len(expected) if expected else 0.0,
    }


def _post_completion(
    url: str,
    model: str,
    row: dict[str, Any],
    timeout: int,
    seed: int,
) -> dict[str, Any]:
    prompt_token_ids = row["prompt_token_ids"]
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": seed,
        "max_tokens": int(row["max_tokens"]),
        "stream": False,
        "skip_special_tokens": False,
        "add_special_tokens": False,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0]
    returned_prompt = choice.get("prompt_token_ids")
    output_token_ids = choice.get("token_ids")
    if returned_prompt != prompt_token_ids:
        raise ValueError("Server returned different prompt token IDs")
    if not isinstance(output_token_ids, list):
        raise TypeError("Server did not return output token IDs")
    text = str(choice.get("text") or "")
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "usage": body.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_token_ids_verified": True,
        "outputs": [
            {
                "token_ids": output_token_ids,
                "first_token_id": output_token_ids[0] if output_token_ids else None,
                "text": text,
                "text_sha256": _sha256_text(text),
                "token_ids_sha256": _token_ids_sha256(output_token_ids),
            }
        ],
        "evaluation": _score(row, text),
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600)


def run_api(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Quality result already exists: {output}")
    input_paths = [Path(path).resolve() for path in args.input_jsonl]
    rows = [row for path in input_paths for row in _read_jsonl(path)]
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Quality input IDs must be nonempty and unique")
    for row in rows:
        token_ids = row.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"Invalid prompt token IDs for {row['id']}")
        if _token_ids_sha256(token_ids) != row.get("prompt_token_ids_sha256"):
            raise ValueError(f"Prompt token SHA-256 mismatch for {row['id']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "label": args.label,
        "api_url": args.url,
        "model": args.model,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": args.seed,
            "repeats": args.repeat,
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ],
        "records": [],
        "complete": False,
    }
    _write_result(output, result)
    for row in rows:
        for repeat in range(1, args.repeat + 1):
            record = {
                "id": row["id"],
                "category": row.get("category"),
                "repeat": repeat,
                "prompt_tokens": len(row["prompt_token_ids"]),
                "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                "started_unix": time.time(),
            }
            try:
                record.update(
                    _post_completion(
                        args.url,
                        args.model,
                        row,
                        args.timeout,
                        args.seed,
                    )
                )
            except urllib.error.HTTPError as error:
                record.update(
                    {
                        "status": "http_error",
                        "http_status": error.code,
                        "detail": error.read().decode(errors="replace")[:4096],
                    }
                )
            except Exception as error:  # noqa: BLE001  # Preserve failed evidence.
                record.update({"status": "error", "detail": repr(error)})
            record["finished_unix"] = time.time()
            result["records"].append(record)
            _write_result(output, result)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "repeat": repeat,
                        "status": record["status"],
                        "correct": (record.get("evaluation") or {}).get("correct"),
                        "elapsed_seconds": record.get("elapsed_seconds"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result["complete"] = all(row.get("status") == "ok" for row in result["records"])
    result["completed_unix"] = time.time()
    _write_result(output, result)
    if not result["complete"]:
        raise SystemExit(1)


def _common_prefix(left: list[int], right: list[int]) -> int:
    prefix = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        prefix += 1
    return prefix


def _stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[list[int]]] = defaultdict(list)
    for record in records:
        if record.get("status") == "ok":
            by_id[str(record["id"])].append(record["outputs"][0]["token_ids"])
    repeated = {key: values for key, values in by_id.items() if len(values) > 1}
    stable = sum(
        len({tuple(ids) for ids in values}) == 1 for values in repeated.values()
    )
    return {
        "repeated_case_count": len(repeated),
        "exact_stable_case_count": stable,
        "exact_stable_ratio": stable / len(repeated) if repeated else None,
    }


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_records = {(row["id"], row["repeat"]): row for row in left["records"]}
    right_records = {(row["id"], row["repeat"]): row for row in right["records"]}
    common = sorted(set(left_records) & set(right_records))
    rows = []
    for key in common:
        left_row = left_records[key]
        right_row = right_records[key]
        input_equal = left_row.get("prompt_token_ids_sha256") == right_row.get(
            "prompt_token_ids_sha256"
        )
        left_ids = (left_row.get("outputs") or [{}])[0].get("token_ids") or []
        right_ids = (right_row.get("outputs") or [{}])[0].get("token_ids") or []
        left_eval = left_row.get("evaluation") or {}
        right_eval = right_row.get("evaluation") or {}
        left_elapsed = left_row.get("elapsed_seconds")
        right_elapsed = right_row.get("elapsed_seconds")
        rows.append(
            {
                "id": key[0],
                "repeat": key[1],
                "category": left_row.get("category"),
                "input_equal": input_equal,
                "first_token_equal": (left_ids[:1] == right_ids[:1]),
                "exact_output_tokens": left_ids == right_ids,
                "common_prefix_tokens": _common_prefix(left_ids, right_ids),
                "left_output_tokens": len(left_ids),
                "right_output_tokens": len(right_ids),
                "left_correct": left_eval.get("correct"),
                "right_correct": right_eval.get("correct"),
                "right_over_left_elapsed": (
                    right_elapsed / left_elapsed
                    if isinstance(left_elapsed, (int, float))
                    and isinstance(right_elapsed, (int, float))
                    and left_elapsed > 0
                    else None
                ),
            }
        )
    comparable = len(rows)
    ratios = [
        row["right_over_left_elapsed"]
        for row in rows
        if row["right_over_left_elapsed"] is not None
    ]
    return {
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "left_record_count": len(left_records),
        "right_record_count": len(right_records),
        "common_record_count": comparable,
        "input_equal_count": sum(row["input_equal"] for row in rows),
        "first_token_equal_ratio": (
            sum(row["first_token_equal"] for row in rows) / comparable
            if comparable
            else None
        ),
        "exact_output_ratio": (
            sum(row["exact_output_tokens"] for row in rows) / comparable
            if comparable
            else None
        ),
        "left_correct_count": sum(row["left_correct"] is True for row in rows),
        "right_correct_count": sum(row["right_correct"] is True for row in rows),
        "correctness_regressions": sum(
            row["left_correct"] is True and row["right_correct"] is not True
            for row in rows
        ),
        "mean_right_over_left_elapsed": (sum(ratios) / len(ratios) if ratios else None),
        "left_stability": _stability(left["records"]),
        "right_stability": _stability(right["records"]),
        "records": rows,
    }


def compare(args: argparse.Namespace) -> None:
    left_path = Path(args.left).resolve()
    right_path = Path(args.right).resolve()
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    if args.left_label:
        left["label"] = args.left_label
    if args.right_label:
        right["label"] = args.right_label
    report = compare_results(left, right)
    report["left_path"] = str(left_path)
    report["right_path"] = str(right_path)
    report["left_sha256"] = _sha256(left_path)
    report["right_sha256"] = _sha256(right_path)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"Comparison output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        output.chmod(0o600)
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    long_parser = subparsers.add_parser("prepare-long")
    long_parser.add_argument("--model", required=True)
    long_parser.add_argument("--case-jsonl", action="append", required=True)
    long_parser.add_argument("--case-id", action="append", required=True)
    long_parser.add_argument("--max-tokens-cap", type=int, default=128)
    long_parser.add_argument("--output-jsonl", required=True)
    long_parser.set_defaults(func=prepare_long)

    tool_parser = subparsers.add_parser("prepare-tools")
    tool_parser.add_argument("--model", required=True)
    tool_parser.add_argument("--heldout-jsonl", required=True)
    tool_parser.add_argument("--corpus-manifest", required=True)
    tool_parser.add_argument("--limit", type=int, default=6)
    tool_parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    tool_parser.add_argument("--max-tokens", type=int, default=128)
    tool_parser.add_argument("--output-jsonl", required=True)
    tool_parser.set_defaults(func=prepare_tools)

    trace_parser = subparsers.add_parser("prepare-trace")
    trace_parser.add_argument("--prepared-jsonl", required=True)
    trace_parser.add_argument("--source-line", type=int, default=1)
    trace_parser.add_argument("--tokens", type=int, default=4096)
    trace_parser.add_argument("--output-jsonl", required=True)
    trace_parser.set_defaults(func=prepare_trace)

    run_parser = subparsers.add_parser("run-api")
    run_parser.add_argument("--input-jsonl", action="append", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--url", required=True)
    run_parser.add_argument("--model", default="qwen")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--repeat", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--timeout", type=int, default=7200)
    run_parser.set_defaults(func=run_api)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--left-label")
    compare_parser.add_argument("--right-label")
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=compare)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
