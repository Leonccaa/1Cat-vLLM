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
        / "qsa_long_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("qsa_long_corpus_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_corpus_overlap_gate_catches_different_window_alignment() -> None:
    tool = _load_tool()
    quality = list(range(400))
    candidate = [999] * 7 + quality[31:180] + [998] * 20

    quality_shingles = tool.rolling_shingles(quality, width=32, stride=1)
    candidate_shingles = tool.rolling_shingles(candidate, width=32, stride=16)

    assert quality_shingles & candidate_shingles
