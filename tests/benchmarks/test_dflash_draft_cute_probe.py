# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Sweep coverage and stock baseline reuse without CUDA."""

import itertools
import json
from types import SimpleNamespace

import torch

from benchmark.kernel_tuning import dflash_draft_cute_probe as probe


def test_sweep_visits_every_default_shape(monkeypatch):
    visited = []
    monkeypatch.setattr("sys.argv", ["probe", "--sweep"])
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "7")
    monkeypatch.setattr(probe, "sweep_shape", lambda *args: visited.append(args))
    probe.main()
    expected = [
        (b, length, hq, hk, q, causal, window, 1, 3)
        for b, length, (hq, hk), (q, causal, window) in itertools.product(
            (1, 2, 4),
            (128, 1024, 4096),
            ((32, 8), (16, 4)),
            tuple(
                (q, causal, window)
                for q in (8, 16)
                for causal, window in ((False, None), (True, 4096), (False, 2048))
            ),
        )
    ]
    assert visited == expected


def test_sweep_reuses_one_stock_measurement(monkeypatch, capsys):
    for name in ("MIN_CHUNK", "TARGET_CTAS", "KEY_TILE", "KEY_GROUPS", "STAGES"):
        monkeypatch.setenv(f"CTI_DRAFT_{name}", "1")
    timings, oracles = [], []

    class Fixture:
        def __init__(self, *args, **kwargs):
            self.query = self.k = self.v = self.out = torch.zeros(1)
            self.op = SimpleNamespace(
                prepare=lambda: None,
                run=lambda **kwargs: None,
                _implementation=SimpleNamespace(splits=7),
            )

        def stage(self, *args):
            pass

        def oracle(self):
            oracles.append(1)
            return self.out

    def latency(call, warmup, **kwargs):
        call()
        timings.append((warmup, kwargs))
        return len(timings)

    monkeypatch.setattr(probe, "Fixture", Fixture)
    monkeypatch.setattr(probe, "fixed_plan", lambda f, *args: f.op)
    monkeypatch.setattr(probe, "graph_latency", latency)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )
    probe.sweep_shape(1, 128, 32, 8, 16, False, None, 1, 3)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 48
    assert len(timings) == 97
    assert len(oracles) == 1
    assert {row["flashinfer_fixed_plan_run_us"] for row in rows} == {1}
    assert {
        (r["min_chunk"], r["target_ctas"], r["key_tile"], r["key_groups"], r["stages"])
        for r in rows
    } == set(itertools.product((64, 128, 256), (170, 340), (32, 64), (1, 2), (2, 3)))
