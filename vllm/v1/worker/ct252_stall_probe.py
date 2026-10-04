# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CT252 diagnostic (not for upstream): locate multi-second GPU segments.

When CT252_STALL_PROBE=1, heavy custom ops record a CUDA event on entry. The QSA
indexer's per-layer scalar sync is timed on the host; if it waits longer than
CT252_STALL_PROBE_S seconds, the GPU time between consecutive recent events is
logged, which names the op segment that held the GPU.
"""

import collections
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("CT252_STALL_PROBE") == "1"
THRESHOLD_S = float(os.environ.get("CT252_STALL_PROBE_S", "1.0"))
_EVENTS: collections.deque = collections.deque(maxlen=768)


STEP_MS = float(os.environ.get("CT252_STALL_PROBE_STEP_MS", "1000"))
STEP_MAX_TOKENS = int(os.environ.get("CT252_STALL_PROBE_STEP_TOKENS", "4000"))
_STEP: list = []


def _category(label: str) -> str:
    head = label.split(":")[0]
    return "kvc_pre" if head == "kvc_pre" else "kvc_post" if head == "kvc_post" else head


def _flush_step() -> None:
    """Summarize the previous step if it was a slow mixed (small) step."""
    items = list(_STEP)
    _STEP.clear()
    if len(items) < 2:
        return
    try:
        if not items[-1][1].query():
            return
        total = items[0][1].elapsed_time(items[-1][1])
    except RuntimeError:
        return
    tokens = [int(l.split(":")[1]) for l, _ in items if l.startswith("moe:")]
    num_tokens = max(tokens) if tokens else -1
    if total < STEP_MS or not 0 < num_tokens <= STEP_MAX_TOKENS:
        return
    by_pair = collections.Counter()
    singles = []
    layer = "?"
    for (l0, e0), (l1, e1) in zip(items, items[1:]):
        if l0.startswith("qsa:"):
            layer = l0[4:].split(".")[0]
        try:
            ms = e0.elapsed_time(e1)
        except RuntimeError:
            continue
        by_pair[f"{_category(l0)}>{_category(l1)}"] += ms
        singles.append((ms, f"L{layer} {l0}>{l1}"))
    singles.sort(reverse=True)
    logger.warning(
        "CT252 stall probe: slow mixed step %.0f ms (%d tokens, %d marks): %s",
        total,
        num_tokens,
        len(items),
        [(k, round(v, 1)) for k, v in by_pair.most_common(8)]
        + [("top", [(round(ms, 1), s) for ms, s in singles[:5]])],
    )


def mark(label: str) -> None:
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    if label == "step":
        _flush_step()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    _EVENTS.append((label, event, time.monotonic()))
    _STEP.append((label, event))


def timed_item(tensor: torch.Tensor, label: str):
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return tensor.item()
    mark(label)
    start = time.monotonic()
    value = tensor.item()
    waited = time.monotonic() - start
    if waited >= THRESHOLD_S:
        _dump(label, waited)
    return value


def _dump(label: str, waited: float) -> None:
    items = list(_EVENTS)
    segments = []
    for (l0, e0, h0), (l1, e1, _) in zip(items, items[1:]):
        try:
            ms = e0.elapsed_time(e1)
        except RuntimeError:
            continue
        segments.append((ms, l0, l1, h0))
    total = sum(s[0] for s in segments)
    segments.sort(reverse=True)
    logger.warning(
        "CT252 stall probe: %s waited %.2f s; %d recent segments span %.0f ms; "
        "slowest: %s",
        label,
        waited,
        len(segments),
        total,
        [(round(ms, 1), a, b) for ms, a, b, _ in segments[:10]],
    )
    _EVENTS.clear()


def timed_call(label: str, fn, *args, **kwargs):
    """Mark around a host call and log it if the host side alone was slow."""
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return fn(*args, **kwargs)
    mark(f"{label}:in")
    start = time.monotonic()
    result = fn(*args, **kwargs)
    took = time.monotonic() - start
    mark(f"{label}:out")
    if took >= THRESHOLD_S:
        logger.warning("CT252 stall probe: host call %s took %.2f s", label, took)
    return result
