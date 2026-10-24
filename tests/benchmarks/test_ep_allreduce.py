# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Import-safe parsing and presentation of paired collective measurements."""

import pytest

from benchmark.kernel_tuning import ep_allreduce as bench


def test_parse_args():
    args = bench.parse_args(
        [
            "--out",
            "results.jsonl",
            "--tokens",
            "1,4,8,4",
            "--candidates",
            "cute,nccl_ar",
            "--probe-only",
        ]
    )
    assert args.tokens == [1, 4, 8]
    assert args.candidates == ["cute", "nccl_ar"]
    assert (args.hidden, args.rounds, args.calls, args.seed) == (2048, 15, 20, 0)
    assert args.probe_only


@pytest.mark.parametrize(
    "option,value",
    [("--tokens", "0,1"), ("--tokens", "x"), ("--calls", "0"), ("--candidates", "unknown")],
)
def test_invalid_args(option, value):
    with pytest.raises(SystemExit):
        bench.parse_args(["--out", "results.jsonl", option, value])


def test_table():
    table = bench.format_table(
        [
            {
                "hidden": 2048,
                "tokens": 4,
                "cute_us": 8,
                "market_ar_us": 20,
                "market_ar_norm_us": None,
                "speedup_vs_market_ar": 2.5,
                "speedup_vs_market_ar_norm": None,
            },
            {
                "hidden": 4096,
                "tokens": 8,
                "cute_us": 10,
                "market_ar_us": 20,
                "market_ar_norm_us": 25,
                "speedup_vs_market_ar": 2,
                "speedup_vs_market_ar_norm": 2.5,
            },
        ]
    )
    assert "| 2048 | 4 | 8.000 | 20.000 | — | 2.500 | — |" in table
    assert "| 4096 | 8 | 10.000 | 20.000 | 25.000 | 2.000 | 2.500 |" in table
    assert "Market AR + norm" in table


def test_norm_bar_candidates():
    assert set(bench.NORM_CANDIDATES) <= set(bench.CANDIDATES)


def test_warp_override_restores_schedule():
    from ctikernels.collective._host_exchange.cute import exchange

    original = exchange.schedule
    with bench._flag_warps(8):
        assert exchange.schedule(1, 2048, slots=188, link_rate=35)[1] == 8
        assert exchange.schedule(9000, 2048, slots=188, link_rate=35) == original(
            9000, 2048, slots=188, link_rate=35
        )
    assert exchange.schedule is original


def test_probe_reports_missing_api(monkeypatch):
    import sys
    from types import ModuleType

    package = ModuleType("flashinfer")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.comm", ModuleType("flashinfer.comm"))
    workspace, report = bench.probe_pcie_ipc(0, 2048, 8192)
    assert workspace is None
    assert report["status"] == "unavailable"
    assert "ImportError" in report["reason"]
