# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DF2 graph gates, provider reporting and benchmark query selection."""

from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import dflash_draft_attention as benchmark


@pytest.mark.parametrize("eager_us,status", [(1, "PASS"), (3, "FAIL")])
@pytest.mark.parametrize("complete_us,graph_status", [(1, "PASS"), (10, "FAIL")])
@pytest.mark.parametrize("run_us,run_status", [(0.5, "PASS"), (5, "FAIL")])
def test_eager_and_graph_gates(
    monkeypatch, eager_us, status, complete_us, graph_status, run_us, run_status
):
    measurements = []
    times = {
        "candidate": eager_us,
        "reference": 2,
        "complete": complete_us,
        "fixed": 1,
        "run": run_us,
    }

    def measure(call, repetitions):
        measurements.append((call, repetitions))
        return times[call]

    monkeypatch.setattr(benchmark, "measure", measure)
    report = benchmark.timing_report(
        "candidate", "reference", "complete", "fixed", "run", calls=100
    )
    assert report["speed"] == report["speed_eager"] == report["eager_gate"]["speed"] == status
    assert report["speed_graph"] == graph_status
    assert report["speed_graph_run_only"] == run_status
    assert report["gate_reference"] == (
        "stock FlashInfer BatchPrefillWithPagedKVCacheWrapper fa2 fixed-plan CUDA graph "
        "(host plan outside timing; cannot follow changing metadata)"
    )
    assert "graph_gate_reason" not in report
    assert report["eager_gate"]["candidate_median_us"] == eager_us
    for index, (candidate, reference) in enumerate(
        (("candidate", "reference"), ("complete", "fixed"), ("run", "fixed"))
    ):
        assert measurements[index * 80 : (index + 1) * 80] == (
            [(reference, 100)] * 40
            + [
                (name, 100)
                for pair in range(20)
                for name in ((candidate, reference) if pair % 2 == 0 else (reference, candidate))
            ]
        )
    assert len(measurements) == 240
    assert report["complete_vs_fixed_plan"]["is_gate"] is True
    assert report["complete_vs_fixed_plan"]["speed"] == graph_status
    assert report["run_only_vs_fixed_plan"]["is_gate"] is False
    assert report["run_only_vs_fixed_plan"]["speed"] == run_status
    assert "informational" in report["modes"]["run_only_vs_fixed_plan"]
    assert report["complete_vs_fixed_plan"]["candidate_median_us"] == complete_us
    assert report["eager_gate"]["pairs"] == 20
    assert report["eager_gate"]["calls_per_interval"] == 100


@pytest.mark.parametrize(
    "config,contexts",
    [
        ("causal_sliding", (0, 1, 4095, 4096, 4097)),
        ("noncausal_sliding", (0, 1, 2047, 2048, 2049)),
        ("full", (0, 1, 128, 4095, 4096, 4097)),
    ],
)
@pytest.mark.parametrize("query_length", [None, *benchmark.QUERY_LENGTHS])
def test_visible_key_counts_match_exact_masks(config, contexts, query_length):
    qlen, causal, window = benchmark.CONFIGS[config]
    qlen = qlen if query_length is None else query_length
    for context in contexts:
        visible = [
            {
                key
                for key in range(context + qlen)
                if (not causal or key <= position)
                and (window is None or abs(position - key) < window)
            }
            for position in range(context, context + qlen)
        ]
        counts, union = benchmark.visible_key_counts(config, context, query_length=query_length)
        assert counts == [len(keys) for keys in visible]
        assert union == len(set().union(*visible))
        assert union == (
            context + qlen if window is None else min(context + qlen, window + qlen - 1)
        )


@pytest.mark.parametrize("ratio,status", [(1.1, "PASS"), (1.1001, "FAIL")])
def test_graph_gates_include_reference_noise(monkeypatch, ratio, status):
    def samples():
        for _ in range(3):
            yield from [1.1, 1] * 20
            for index in range(20):
                yield from (ratio, 1) if index % 2 == 0 else (1, ratio)

    values = samples()
    monkeypatch.setattr(benchmark, "measure", lambda *_: next(values))
    report = benchmark.timing_report(None, None, None, None, None, calls=100)
    assert report["speed_graph"] == report["speed_graph_run_only"] == status
    assert report["complete_vs_fixed_plan"]["epsilon"] == pytest.approx(0.1)


def test_provider_report_records_available_tuning():
    tuning = {
        "splits": 8,
        "key_tile": 32,
        "stages": 2,
        "key_groups": 1,
        "min_chunk": 64,
        "short_union": 416,
        "inline_merge": True,
    }
    op = SimpleNamespace(provider="cute-splitkv", _implementation=SimpleNamespace(**tuning))
    assert benchmark.provider_report(op) == {"provider": "cute-splitkv", "cute_tuning": tuning}
    op._implementation = SimpleNamespace(splits=4)
    report = benchmark.provider_report(op)
    assert report["cute_tuning"] == dict.fromkeys(tuning) | {"splits": 4}
    op.provider = "flashinfer-fa2-device-schedule"
    assert benchmark.provider_report(op) == {"provider": op.provider, "cute_tuning": None}


@pytest.mark.parametrize("provider", ["auto", "cute", "flashinfer"])
@pytest.mark.parametrize("queries", [None, list(benchmark.QUERY_LENGTHS)])
def test_cli_dispatches_masks_queries_and_provider(monkeypatch, provider, queries):
    argv = ["benchmark", "--batch-sizes", "1", "--contexts", "128", "--provider", provider]
    if queries is not None:
        argv += ["--queries", *map(str, queries)]
    monkeypatch.setattr(benchmark.sys, "argv", argv)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(benchmark, "package_version", lambda _: "test")
    calls, rows = [], []

    def run_case(config, b, context, hq, hk, **kwargs):
        calls.append((config, b, context, hq, hk, kwargs))
        return {"speed": "PASS"}

    monkeypatch.setattr(benchmark, "run_case", run_case)
    monkeypatch.setattr(benchmark, "emit", rows.append)
    benchmark.main()
    expected = (
        {(config, q) for q in queries for config in benchmark.CONFIGS}
        if queries is not None
        else {(config, values[0]) for config, values in benchmark.CONFIGS.items()}
    )
    assert {(call[0], call[-1]["query_length"]) for call in calls} == expected
    for _config, b, _context, hq, hk, kwargs in calls:
        assert kwargs["provider"] == (None if provider == "auto" else provider)
        assert kwargs["calls"] == 100 and kwargs["warmup"] == 20
        assert b == 1 and (hq, hk) in ((32, 8), (16, 4))
    for config, q in expected:
        boundary = benchmark.CONFIGS[config][2] or 4096
        assert {
            call[2] for call in calls if call[0] == config and call[-1]["query_length"] == q
        } == {128, boundary - 1, boundary, boundary + 1}
    assert rows[0]["provider_requested"] == provider
    assert {(row["mask"], row["query_length"]) for row in rows[1:]} == expected
    assert all(row["gate_reference"] == benchmark.GATE_REFERENCE for row in rows)


def test_cli_cpu_unavailable_reports_graph_na(monkeypatch):
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "--queries", "5"])
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    rows = []
    monkeypatch.setattr(benchmark, "emit", rows.append)
    benchmark.main()
    for row in rows[1:]:
        assert row["speed_graph"] == row["speed_graph_run_only"] == "N/A"
        assert row["gate_reference"] == benchmark.GATE_REFERENCE
        assert row["sol"] is None
        assert "graph_gate_reason" not in row
