# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_tool():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "qwen4_exp"
        / "qsa_kv_quality.py"
    )
    spec = importlib.util.spec_from_file_location("qsa_kv_quality_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_trace_freezes_exact_prefix(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "prepared.jsonl"
    output = tmp_path / "trace.jsonl"
    source.write_text(
        json.dumps({"prompt_token_ids": list(range(16))}) + "\n",
        encoding="utf-8",
    )

    tool.prepare_trace(
        argparse.Namespace(
            prepared_jsonl=str(source),
            source_line=1,
            tokens=8,
            output_jsonl=str(output),
        )
    )

    row = json.loads(output.read_text())
    manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert row["prompt_token_ids"] == list(range(8))
    assert row["prompt_token_ids_sha256"] == tool._token_ids_sha256(list(range(8)))
    assert manifest["active_tokens"] == 8


def _result(label: str, token_ids: list[int], correct: bool) -> dict:
    return {
        "label": label,
        "records": [
            {
                "id": "case",
                "category": "test",
                "repeat": 1,
                "status": "ok",
                "prompt_token_ids_sha256": "prompt-hash",
                "elapsed_seconds": 2.0 if label == "left" else 1.0,
                "outputs": [{"token_ids": token_ids}],
                "evaluation": {"correct": correct},
            }
        ],
    }


def test_compare_results_reports_first_token_and_correctness_regression() -> None:
    tool = _load_tool()

    report = tool.compare_results(
        _result("left", [10, 11], True),
        _result("right", [10, 12], False),
    )

    assert report["input_equal_count"] == 1
    assert report["first_token_equal_ratio"] == 1.0
    assert report["exact_output_ratio"] == 0.0
    assert report["correctness_regressions"] == 1
    assert report["mean_right_over_left_elapsed"] == 0.5


def test_compare_can_override_historical_run_labels(tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(_result("fp16-p1024", [10], True)))
    right.write_text(json.dumps(_result("e4m3-p256", [10], True)))

    tool.compare(
        argparse.Namespace(
            left=str(left),
            right=str(right),
            left_label="fp16-kv",
            right_label="e4m3-calibrated-kv",
            output=None,
        )
    )

    report = json.loads(capsys.readouterr().out)
    assert report["left_label"] == "fp16-kv"
    assert report["right_label"] == "e4m3-calibrated-kv"


def test_score_supports_tool_any_and_retrieval_all() -> None:
    tool = _load_tool()

    assert tool._score(
        {"score_mode": "contains_any", "gold_answers": ["exec", "wait"]},
        "I will call wait now",
    )["correct"]
    assert not tool._score(
        {"score_mode": "contains_all", "gold_answers": ["one", "two"]},
        "one",
    )["correct"]


def test_tool_call_indices_include_later_agent_actions() -> None:
    tool = _load_tool()

    assert tool._tool_call_indices(
        [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "tool_calls": [{"id": "first"}]},
            {"role": "tool", "content": "running"},
            {"role": "assistant", "tool_calls": [{"id": "wait"}]},
        ]
    ) == [1, 3]


def test_prepare_tools_accepts_generic_heldout_corpus(
    tmp_path: Path, monkeypatch
) -> None:
    tool = _load_tool()
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "heldout.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "quality.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "private-case-id",
                "session_id": "private-session-id",
                "messages": [
                    {"role": "user", "content": "Find the current value."},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}},
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_sha256": tool._sha256(source),
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(tool, "_chat_token_ids", lambda *args, **kwargs: [1, 2, 3])

    tool.prepare_tools(
        argparse.Namespace(
            model=str(model),
            heldout_jsonl=str(source),
            corpus_manifest=str(manifest_path),
            limit=1,
            max_prompt_tokens=128,
            max_tokens=16,
            output_jsonl=str(output),
        )
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    output_manifest = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(encoding="utf-8")
    )
    serialized = output.read_text(encoding="utf-8")
    assert row["category"] == "heldout_tool_selection"
    assert row["gold_answers"] == ["lookup"]
    assert row["prompt_token_ids"] == [1, 2, 3]
    assert row["source"]["source_id_sha256"] == tool._sha256_text("private-case-id")
    assert "private-case-id" not in serialized
    assert "private-session-id" not in serialized
    assert output_manifest["purpose"] == (
        "frozen-qsa-kv-heldout-tool-quality-token-ids"
    )
