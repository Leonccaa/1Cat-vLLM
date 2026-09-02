# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "qwen4_exp"
        / "qsa_tool_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("qsa_tool_corpus_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_corpus_covers_selection_cycle_and_multiturn() -> None:
    tool = _load_tool()
    rows = tool.build_rows(3)

    assert len(rows) == 9
    assert {row["category"] for row in rows} == {
        "tool_call_chinese_selection",
        "tool_call_english_cycle",
        "tool_call_code_multiturn",
    }
    assert all(row["tools"] == tool.TOOLS for row in rows)
    assert any(
        any(message.get("role") == "tool" for message in row["messages"])
        for row in rows
    )
