# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Per-call device time of a callable, eager or replayed from a CUDA graph.

Each sample spans ``calls`` launches, so CUDA event resolution (about 0.5 us on some GPUs)
quantizes a per-call time by at most resolution / calls.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _events(count: int) -> list[torch.cuda.Event]:
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def eager(fn: Callable[[], object], *, samples: int = 30, calls: int = 100, warmup: int = 5):
    """``samples`` per-call times in microseconds, each the mean of ``calls`` back-to-back calls."""
    for _ in range(warmup):
        fn()
    starts, ends = _events(samples), _events(samples)
    for start, end in zip(starts, ends, strict=True):
        start.record()
        for _ in range(calls):
            fn()
        end.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 / calls for s, e in zip(starts, ends, strict=True)]


def graph(fn: Callable[[], object], *, samples: int = 30, calls: int = 100, warmup: int = 3):
    """Like :func:`eager`, but ``calls`` launches are captured once and replayed per sample.

    This removes host launch cost, as an engine running CUDA graphs sees it. ``fn`` must be
    warm (compiled, plans pinned) before the first call here; warm-up calls run eagerly.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(captured, stream=stream):
        for _ in range(calls):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    captured.replay()
    starts, ends = _events(samples), _events(samples)
    for start, end in zip(starts, ends, strict=True):
        start.record()
        captured.replay()
        end.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 / calls for s, e in zip(starts, ends, strict=True)]


TIMERS = {"graph": graph, "eager": eager}
