# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CT252 diagnostic (not for upstream): locate multi-second GPU segments.

When CT252_STALL_PROBE=1, heavy custom ops record a CUDA event on entry. The QSA
indexer's per-layer scalar sync is timed on the host; if it waits longer than
CT252_STALL_PROBE_S seconds, the GPU time between consecutive recent events is
logged, which names the op segment that held the GPU.
"""

import collections
import gc
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("CT252_STALL_PROBE") == "1"
GC_FREEZE = os.environ.get("CT252_WORKER_GC_FREEZE") == "1"
GC_LOG_MS = float(os.environ.get("CT252_GC_LOG_MS", "50"))
# Steps counted by mark("step") include startup warmup runs; freeze after them.
FREEZE_AT_STEP = int(os.environ.get("CT252_GC_FREEZE_AT_STEP", "20"))
_gc_start: dict = {}
_gc_frozen = False


def _gc_callback(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_start["t"] = time.perf_counter()
        return
    started = _gc_start.pop("t", None)
    if started is None:
        return
    ms = (time.perf_counter() - started) * 1000.0
    if ms >= GC_LOG_MS:
        logger.warning(
            "CT252 stall probe: GC generation %s took %.0f ms "
            "(collected %s, frozen %d)",
            info.get("generation"),
            ms,
            info.get("collected"),
            gc.get_freeze_count(),
        )


if ENABLED:
    gc.callbacks.append(_gc_callback)


def _maybe_freeze_gc() -> None:
    """Freeze the worker's long-lived objects once, as EngineCore does."""
    global _gc_frozen
    if _gc_frozen or not GC_FREEZE:
        return
    _gc_frozen = True
    start = time.perf_counter()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    logger.warning(
        "CT252 stall probe: froze %d worker objects in %.0f ms",
        gc.get_freeze_count(),
        (time.perf_counter() - start) * 1000.0,
    )
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


_ALLOC_KEYS = (
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
)
_last_alloc: dict = {}
_step_count = 0


def _check_allocator() -> None:
    """Log whenever the caching allocator retried, freed or synced all streams."""
    global _last_alloc
    stats = torch.cuda.memory_stats()
    now = {k: stats.get(k, 0) for k in _ALLOC_KEYS}
    if _last_alloc and any(
        now[k] != _last_alloc[k]
        for k in (
            "num_alloc_retries",
            "num_ooms",
            "num_sync_all_streams",
            "num_device_free",
        )
    ):
        logger.warning(
            "CT252 stall probe: allocator event at step %d: %s; reserved %.0f MiB, "
            "allocated %.0f MiB, device alloc/free +%d/+%d",
            _step_count,
            {k: now[k] - _last_alloc[k] for k in _ALLOC_KEYS[:3]},
            stats.get("reserved_bytes.all.current", 0) / 2**20,
            stats.get("allocated_bytes.all.current", 0) / 2**20,
            now["num_device_alloc"] - _last_alloc["num_device_alloc"],
            now["num_device_free"] - _last_alloc["num_device_free"],
        )
    _last_alloc = now


def mark(label: str) -> None:
    global _step_count
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    if label == "step":
        _step_count += 1
        if _step_count == FREEZE_AT_STEP:
            _maybe_freeze_gc()
        _flush_step()
        _check_allocator()
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
