# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks for the host-time benchmark selection and timing protocol."""

import pytest

from benchmark.kernel_tuning import linear_host_overhead as bench


def test_parse_args():
    args = bench.parse_args([])
    assert (64, 4096) in args.shapes and args.m == [1, 8, 16, 64]
    args = bench.parse_args(["--shapes", "67:264,64:4096,67:264", "--m", "5,64,5"])
    assert args.shapes == [(67, 264), (64, 4096)] and args.m == [5, 64]
    for argv in (
        ["--shapes", "0:8"],
        ["--shapes", "8:7"],
        ["--shapes", "8"],
        ["--shapes", "8:8:8"],
        ["--m", "0"],
        ["--m", "65"],
    ):
        with pytest.raises(SystemExit):
            bench.parse_args(argv)


def test_timing_interleaves_and_synchronizes_outside_loops(monkeypatch):
    events = []
    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(bench.F, "linear", lambda *a: events.append("baseline"))
    clock = iter(range(0, 80_000, 1000))

    def now():
        events.append("time")
        return next(clock)

    monkeypatch.setattr(bench.time, "perf_counter_ns", now)
    result = bench.measure(lambda *a: events.append("linear"), None, None)
    assert events[:21] == ["linear"] * 10 + ["baseline"] * 10 + ["sync"]
    expected = events[:21]
    for r in range(20):
        for provider in ("linear", "baseline") if r % 2 == 0 else ("baseline", "linear"):
            expected += ["sync", "time"] + [provider] * 100 + ["time"]
    assert events == expected + ["sync"]
    assert result == pytest.approx(
        {
            "linear_median_us": 0.01,
            "linear_p90_us": 0.01,
            "f_linear_median_us": 0.01,
            "f_linear_p90_us": 0.01,
            "median_delta_us": 0.0,
        }
    )
