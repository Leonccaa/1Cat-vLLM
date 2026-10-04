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
import types

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


CENSUS = os.environ.get("CT252_GC_CENSUS") == "1"
CENSUS_TRIGGER = os.environ.get("CT252_GC_CENSUS_TRIGGER", "/tmp/ct252_gc_census")
_census_mtime = 0.0


def _owner(obj, ignore: set) -> str:
    """Name the attribute or key that holds a large container, two levels up."""
    names = []
    for ref in gc.get_referrers(obj):
        if id(ref) in ignore or isinstance(ref, types.FrameType):
            continue
        if isinstance(ref, dict):
            key = next((k for k, v in ref.items() if v is obj), None)
            holder = next(
                (
                    type(r).__qualname__
                    for r in gc.get_referrers(ref)
                    if getattr(r, "__dict__", None) is ref
                ),
                "dict",
            )
            names.append(f"{holder}.{key}")
        else:
            names.append(type(ref).__qualname__)
        if len(names) >= 3:
            break
    return ",".join(names)


def _census(reason: str) -> None:
    """Log what the worker's GC-tracked heap is made of."""
    start = time.perf_counter()
    objs = gc.get_objects()
    counts = collections.Counter(
        f"{type(o).__module__}.{type(o).__qualname__}" for o in objs
    )
    edges = 0
    large = []
    for o in objs:
        if isinstance(o, (list, tuple, dict, set, frozenset, collections.deque)):
            n = len(o)
            edges += n
            if n >= 50_000:
                large.append((n, o))
    large.sort(key=lambda item: -item[0])
    ignore = {id(objs), id(large), *(id(item) for item in large)}
    described = []
    for n, o in large[:5]:
        first = next(iter(o), None)
        described.append(
            (n, type(o).__qualname__, type(first).__qualname__, _owner(o, ignore))
        )
    logger.warning(
        "CT252 stall probe: GC census (%s) at step %d: %d tracked objects, "
        "%d container entries, frozen %d, counts %s, %.0f ms; top types %s; "
        "large containers %s",
        reason,
        _step_count,
        len(objs),
        edges,
        gc.get_freeze_count(),
        gc.get_count(),
        (time.perf_counter() - start) * 1000.0,
        counts.most_common(15),
        described,
    )
    del objs, large


def _maybe_census() -> None:
    global _census_mtime
    try:
        mtime = os.stat(CENSUS_TRIGGER).st_mtime
    except OSError:
        return
    if mtime > _census_mtime:
        _census_mtime = mtime
        _census("trigger")


THRESHOLD_S = float(os.environ.get("CT252_STALL_PROBE_S", "1.0"))
_EVENTS: collections.deque = collections.deque(maxlen=768)


STEP_MS = float(os.environ.get("CT252_STALL_PROBE_STEP_MS", "1000"))
STEP_MAX_TOKENS = int(os.environ.get("CT252_STALL_PROBE_STEP_TOKENS", "4000"))
_STEP: list = []


def _category(label: str) -> str:
    head = label.split(":")[0]
    return (
        "kvc_pre" if head == "kvc_pre" else "kvc_post" if head == "kvc_post" else head
    )


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
    tokens = [int(l.split(":")[1]) for l, _, _ in items if l.startswith("moe:")]
    num_tokens = max(tokens) if tokens else -1
    if total < STEP_MS or not 0 < num_tokens <= STEP_MAX_TOKENS:
        return
    by_pair = collections.Counter()
    singles = []
    layer = "?"
    for (l0, e0, h0), (l1, e1, h1) in zip(items, items[1:]):
        if l0.startswith("qsa:"):
            layer = l0[4:].split(".")[0]
        try:
            ms = e0.elapsed_time(e1)
        except RuntimeError:
            continue
        by_pair[f"{_category(l0)}>{_category(l1)}"] += ms
        singles.append((ms, f"L{layer} {l0}>{l1} host {(h1 - h0) * 1000.0:.0f}"))
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
            if CENSUS:
                _census("before freeze")
            _maybe_freeze_gc()
        if CENSUS:
            _maybe_census()
        _flush_step()
        _check_allocator()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    now = time.monotonic()
    _EVENTS.append((label, event, now))
    _STEP.append((label, event, now))


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
    for (l0, e0, h0), (l1, e1, h1) in zip(items, items[1:]):
        try:
            ms = e0.elapsed_time(e1)
        except RuntimeError:
            continue
        # GPU gap and the host time between the two enqueues: a GPU gap with
        # a matching host gap means the host was late; a GPU gap with no host
        # gap means the stream was held by something else.
        segments.append((ms, l0, l1, (h1 - h0) * 1000.0))
    total = sum(s[0] for s in segments)
    segments.sort(reverse=True)
    logger.warning(
        "CT252 stall probe: %s waited %.2f s; %d recent segments span %.0f ms; "
        "slowest (gpu_ms, host_ms): %s",
        label,
        waited,
        len(segments),
        total,
        [(round(ms, 1), round(host, 1), a, b) for ms, a, b, host in segments[:8]],
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
