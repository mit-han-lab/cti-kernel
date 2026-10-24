# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU smoke checks for the slot benchmark's bindings, pairing and CLI."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import gdn_k2_slot_ab as bench
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs


def test_cpu_bindings_advance_state_and_match_bitwise():
    weights, x, initial, history = case_inputs(1, 1, 1, 1, 3, "ordinary", 23, "cpu")
    drafts = x[-2:].transpose(0, 1).contiguous().view(1, 2, -1)
    buffers = [
        bench.RoundBuffers(weights, drafts, initial, history, hk=1, hv=1, depth=2, rounds=2)
        for _ in range(2)
    ]
    a, b = buffers
    for binding in (bench.BoundRound(a, 0), bench.BoundRound(b)):
        binding.prepare()
        binding.call()
        binding.buffers.check_replay()
    for left, right in zip(a.comparable().values(), b.comparable().values(), strict=True):
        assert torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    # The static graph for another invocation slot borrows that slot's views.
    other = bench.BoundRound(a, 1)
    assert other.pending.raw.data_ptr() == a.raw[1].data_ptr()
    other.prepare()
    other.call()
    a.check_replay()
    a.prepare()
    assert not a.state.metadata[:, 1].any()
    assert not a.raw.any()


def test_abba_self_pairs_and_ratio_reduction():
    calls = []
    a_times = iter([1, 100, 10, 10, 10, 10])
    b_times = iter([2, 50])

    def sample(name, times):
        calls.append(name)
        return next(times)

    result = bench.paired_samples(
        SimpleNamespace(sample=lambda: sample("A", a_times)),
        SimpleNamespace(sample=lambda: sample("B", b_times)),
        2,
    )
    assert calls == list("ABBAAAAA")
    assert result["samples"]["pairs_us"] == [(1, 2), (100, 50)]
    assert result["samples"]["self_pairs_us"] == [(10, 10), (10, 10)]
    assert result["median_ratio_a_over_b"] == 1.25
    assert result["epsilon"] == 0
    assert result["verdict"] == "no-regression"


def test_cli_matrix_writes_json_lines_and_table(monkeypatch, tmp_path, capsys):
    calls = []

    def measure(hk, hv, batch, q, **kwargs):
        calls.append((hk, hv, batch, q, kwargs))
        return {
            "device": "test device",
            "shape": [hk, hv],
            "b": batch,
            "q": q,
            "a_us": 2.0,
            "b_us": 1.0,
            "median_ratio_a_over_b": 2.0,
            "epsilon": 0.01,
            "verdict": "no-regression",
            "bitwise": True,
        }

    monkeypatch.setattr(bench, "measure", measure)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "rows.jsonl"
    bench.main(
        [
            "--shapes",
            "16,32",
            "8,16",
            "--batches",
            "1",
            "2",
            "4",
            "--qs",
            "8",
            "16",
            "--pairs",
            "20",
            "--out",
            str(output),
        ]
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len(calls) == 12
    assert all(call[-1] == {"pairs": 20, "rounds": 20, "depth": 2} for call in calls)
    assert {tuple(row["shape"]) for row in rows} == {(16, 32), (8, 16)}
    assert "A/B" in capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [("--qs", "1"), ("--pairs", "0"), ("--depth", "0")])
def test_invalid_counts(flag, value):
    with pytest.raises(SystemExit):
        bench.parse_args(["--out", "unused.jsonl", flag, value])
