# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Capture preparation without CUDA execution."""

from contextlib import contextmanager
from types import SimpleNamespace

from benchmark.kernel_tuning import block_fp8 as bench


def test_median_warms_every_op_before_eager_calls_and_capture(monkeypatch):
    events = []
    capture_stream = SimpleNamespace(wait_stream=lambda other: None)
    active_stream = None

    @contextmanager
    def use_stream(stream):
        nonlocal active_stream
        active_stream = stream
        yield
        active_stream = None

    def record(event):
        assert active_stream is capture_stream
        events.append(event)

    @contextmanager
    def capture(graph, *, stream):
        assert stream is capture_stream
        events.append("capture")
        with use_stream(stream):
            yield

    cuda = SimpleNamespace(
        Stream=lambda: capture_stream,
        current_stream=lambda: None,
        stream=use_stream,
        synchronize=lambda: None,
        CUDAGraph=lambda: SimpleNamespace(replay=lambda: None),
        graph=capture,
        Event=lambda **kw: SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, elapsed_time=lambda end: 0.004
        ),
    )
    monkeypatch.setattr(bench.torch, "cuda", cuda)
    result = bench._median_us(
        [lambda: record("call a"), lambda: record("call b")],
        iterations=2,
        warmups=[lambda: record("warm a"), lambda: record("warm b")],
    )
    assert events == ["warm a", "warm b", "call a", "call b", "capture", "call a", "call b"]
    assert result == 2.0
