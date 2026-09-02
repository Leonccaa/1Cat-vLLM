#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a deterministic, non-evaluation QSA tool-call calibration corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_records",
            "description": "Search a synthetic record collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_metric",
            "description": "Calculate an aggregate over supplied numeric values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["operation", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_source",
            "description": "Inspect a synthetic source file without modifying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "symbol": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
]


def build_rows(count_per_family: int) -> list[dict[str, object]]:
    rows = []
    for index in range(count_per_family):
        rows.append(
            {
                "id": f"tool-zh-selection-{index:04d}",
                "category": "tool_call_chinese_selection",
                "tools": TOOLS,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"请只读查询合成记录批次 {index:04d}，找出包含标签 "
                            f"alpha-{index % 11} 的前三项。"
                        ),
                    }
                ],
            }
        )
        values = [index + 0.25, index + 1.5, index + 2.75]
        rows.append(
            {
                "id": f"tool-en-cycle-{index:04d}",
                "category": "tool_call_english_cycle",
                "tools": TOOLS,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Compute the mean for synthetic batch {index:04d}.",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "calculate_metric",
                                    "arguments": {
                                        "operation": "mean",
                                        "values": values,
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {"mean": sum(values) / len(values), "batch": index}
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "The synthetic mean was computed successfully.",
                    },
                ],
            }
        )
        rows.append(
            {
                "id": f"tool-code-multiturn-{index:04d}",
                "category": "tool_call_code_multiturn",
                "tools": TOOLS,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Inspect /synthetic/module_{index:04d}.py for function "
                            "normalize, then explain its return contract."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "I will inspect the requested synthetic source.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "inspect_source",
                                    "arguments": {
                                        "path": f"/synthetic/module_{index:04d}.py",
                                        "symbol": "normalize",
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": (
                            "def normalize(values):\n"
                            "    total = sum(values) or 1\n"
                            "    return [value / total for value in values]\n"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "It returns a new list whose elements are divided by the "
                            "sum, using one when the sum is zero."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Would the input list be mutated?",
                    },
                ],
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    if args.count_per_family <= 0:
        raise ValueError("--count-per-family must be positive")
    output = Path(args.output_jsonl).resolve()
    manifest = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Tool calibration output already exists: {output}")
    rows = build_rows(args.count_per_family)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "synthetic-tool-call-kv-calibration-only",
                "record_count": len(rows),
                "count_per_family": args.count_per_family,
                "category_counts": {
                    category: sum(row["category"] == category for row in rows)
                    for category in sorted({str(row["category"]) for row in rows})
                },
                "output_jsonl": str(output),
                "output_sha256": digest,
                "quality_set_membership": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--count-per-family", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
