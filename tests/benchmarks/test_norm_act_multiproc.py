# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import builtins
import random
import runpy
import statistics
from pathlib import Path

import pytest

from benchmark.kernel_tuning import norm_act_multiproc as bench


def processes(raws, *, m=256, regime="hot"):
    return [
        {"process": i, "rows": [{"M": m, "regime": regime, "raw": raw}], "returncode": 0}
        for i, raw in enumerate(raws, 1)
    ]


def samples(count=20, **overrides):
    return {name: [1.0] * count for name in ("ours", "aa", "vllm", "sglang")} | overrides


def test_import_needs_neither_torch_nor_numpy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "numpy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    runpy.run_path(str(Path(bench.__file__)))


def test_bootstrap_default_seed_resamples_and_percentiles():
    groups = [[0.5, 0.9], [1.0, 1.2, 1.7]]
    rng = random.Random(20261016)
    values = sum(groups, [])
    medians = sorted(statistics.median(rng.choices(values, k=5)) for _ in range(10000))
    expected = [
        medians[249] * 0.025 + medians[250] * 0.975,
        medians[9749] * 0.975 + medians[9750] * 0.025,
    ]
    assert bench.bootstrap_ci(groups) == pytest.approx(expected)
    assert bench.bootstrap_ci(groups) == bench.bootstrap_ci(groups)
    assert bench.percentile([0, 10], 0.25) == 2.5


def test_hierarchical_bootstrap_captures_between_process_variation():
    groups = [[1.0] * 20, [2.0] * 20, [3.0] * 20]
    assert bench.bootstrap_ci(groups) == [2.0, 2.0]
    assert bench.bootstrap_ci(groups, hierarchical=True) == [1.0, 3.0]


@pytest.mark.parametrize("groups", ([], [[]], [[1.0], []]))
def test_bootstrap_rejects_empty_processes(groups):
    with pytest.raises(ValueError, match="nonempty"):
        bench.bootstrap_ci(groups)


def test_pooling_preserves_rounds_and_weights_pairs_not_process_medians():
    stats = bench.pooled_statistics({1: [1.0] * 20, 2: [2.0] * 20, 3: [3.0] * 60}, resamples=200)
    assert stats["n_pairs"] == 100 and stats["n_procs"] == 3
    assert stats["median"] == 3.0
    assert [p["median"] for p in stats["process_medians"]] == [1.0, 2.0, 3.0]
    raw = samples(vllm=[0.8, 2.0] * 10, sglang=[2.0, 0.8] * 10, cute_test=[0.4] * 20)
    row = bench.summarize(processes([raw] * 3), resamples=200)[0]
    assert row["candidates"]["ours"]["median"] == 0.8
    assert row["candidates"]["ours"]["verdict"] == "FAIL"
    assert row["candidates"]["cute_test"]["median"] == 2.0
    assert row["candidates"]["cute_test"]["verdict"] == "PASS"
    assert set(row["candidates"]) == {"ours", "aa", "cute_test"}


def test_aa_direction_noise_floor_and_ci_half_width():
    assert bench.aa_tolerance({"median": 1.0, "ci95": [1.0, 1.0]}) == 0.005
    assert bench.aa_tolerance({"median": 0.98, "ci95": [0.95, 1.01]}) == pytest.approx(0.05)
    row = bench.summarize(processes([samples(aa=[1.1] * 20)] * 3), resamples=200)[0]
    assert row["aa"]["median"] == 1.1  # aa/ours, not the inverse.
    assert row["tol"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "median,low,pairs,procs,expected",
    [
        (1.0, 0.98, 60, 3, "PASS"),
        (0.999, 0.99, 60, 3, "FAIL"),
        (1.01, 0.979, 60, 3, "FAIL"),
        (1.01, 1.0, 59, 3, "INSUFFICIENT"),
        (1.01, 1.0, 60, 2, "INSUFFICIENT"),
    ],
)
def test_verdict_requires_both_gates_and_minimum_coverage(median, low, pairs, procs, expected):
    stats = {"median": median, "ci95": [low, 1.1], "n_pairs": pairs, "n_procs": procs}
    assert bench.verdict(stats, 0.02) == expected


def test_consumer_ab_is_median_of_paired_candidate_over_ours():
    raw = samples(ours=[1.0, 4.0] * 10, aa=[1.0, 4.0] * 10, cute_test=[2.0] * 20)
    row = bench.summarize(processes([raw] * 3), keep=["cute_test"], consumer=True, resamples=200)[0]
    stats = row["consumer_ab"]["cute_test"]
    assert stats["median"] == 1.25  # Not ratio of medians (2/2.5).
    assert stats["ci95"] == [0.5, 2.0]
    assert stats["numerator"] == "cute_test" and stats["denominator"] == "ours"
    assert row["consumer"] == "down_proj_mm_4096"


def test_rows_and_missing_candidates_do_not_inflate_process_counts():
    data = processes([samples(cute_test=[1.0] * 20), samples(), samples()])
    data[0]["rows"].append({"M": 16, "regime": "cold", "raw": samples()})
    rows = bench.summarize(data, resamples=100)
    assert [(r["M"], r["regime"]) for r in rows] == [(16, "cold"), (256, "hot")]
    assert rows[0]["candidates"]["ours"]["verdict"] == "INSUFFICIENT"
    assert rows[1]["candidates"]["ours"]["verdict"] == "PASS"
    candidate = rows[1]["candidates"]["cute_test"]
    assert candidate["n_pairs"] == 20 and candidate["n_procs"] == 1
    assert candidate["verdict"] == "INSUFFICIENT"


@pytest.mark.parametrize("bad", ([1.0], [0.0] * 20, [float("nan")] * 20))
def test_malformed_pairs_rejected(bad):
    with pytest.raises(ValueError):
        bench.summarize(processes([samples(aa=bad)]), resamples=10)


def test_collect_row_retains_only_raw_samples_and_rejects_duplicates():
    import json

    process = {"process": 1, "rows": []}
    raw = samples()
    line = "ROW " + json.dumps({"M": 256, "regime": "hot", "raw": raw, "providers": {}})
    bench.collect_row(line, process)
    assert process["rows"] == [{"M": 256, "regime": "hot", "raw": raw}]
    with pytest.raises(ValueError, match="duplicate"):
        bench.collect_row(line, process)


def test_driver_defaults_and_passthrough_command():
    defaults = bench.parse_args([])
    assert (defaults.procs, defaults.rounds, defaults.rows, defaults.regimes) == (
        4,
        20,
        [256],
        ["hot", "cold"],
    )
    args = bench.parse_args(
        [
            "--procs",
            "3",
            "--rounds",
            "22",
            "--rows",
            "16",
            "256",
            "--regimes",
            "cold",
            "--consumer",
            "--keep",
            "flashinfer,cute_test",
            "--market-root",
            "market dir",
            "--sglang-root",
            "sglang dir",
            "--",
            "--check",
        ]
    )
    command = bench.child_command(args)
    assert command[:3] == [bench.sys.executable, "-m", "benchmark.kernel_tuning.norm_act_paired"]
    assert "--raw" in command and "--consumer" in command and "--check" in command
    for flag, value in (
        ("--keep", "flashinfer,cute_test"),
        ("--market-root", "market dir"),
        ("--sglang-root", "sglang dir"),
        ("--rounds", "22"),
        ("--op", "silu_mul"),
    ):
        assert command[command.index(flag) + 1] == value


def test_paired_parser_raw_consumer_keep():
    from benchmark.kernel_tuning.norm_act_paired import parse_args

    roots = ["--market-root", ".", "--sglang-root", "."]
    defaults = parse_args(roots)
    assert not defaults.raw and not defaults.consumer and defaults.keep is None
    keep = "flashinfer,cute_t64_v1_e8_p1_te,cute_t256_v1_e2_p1_te_cstef"
    args = parse_args([*roots, "--raw", "--consumer", "--keep", keep])
    assert args.raw and args.consumer and args.keep == keep
    with pytest.raises(SystemExit, match="2"):
        parse_args([*roots, "--keep", "cute_unknown"])
    with pytest.raises(SystemExit, match="2"):
        parse_args([*roots, "--op", "add_norm", "--consumer"])


def test_consumer_wrapper_uses_activation_output_and_exposes_capture_op():
    import torch

    from benchmark.kernel_tuning.norm_act_paired import DownProjCall, cached_capture_plans
    from ctikernels.activation.inference import SiluAndMul

    x = torch.tensor([[1.0, 2.0]])
    out, y = torch.empty_like(x), torch.empty((1, 3))
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def activation(x, out):
        out.copy_(x * 2)
        return out

    call = DownProjCall(activation, weight, y)
    assert call(x, out) is y
    assert torch.equal(y, (x * 2) @ weight)
    op = SiluAndMul()
    with cached_capture_plans(DownProjCall(op, weight, y)):
        assert op._require_cached_plan
    assert not op._require_cached_plan


@pytest.mark.parametrize("consumer", (False, True))
@pytest.mark.parametrize("raw", (False, True))
def test_silu_row_keeps_checks_bare_and_wraps_all_capture_candidates(monkeypatch, consumer, raw):
    from types import SimpleNamespace

    import torch

    from benchmark.kernel_tuning import norm_act_paired as paired

    args = paired.parse_args(
        [
            "--market-root",
            ".",
            "--sglang-root",
            ".",
            "--width",
            "16",
            "--layers",
            "2",
            "--check",
        ]
    )
    args.consumer, args.raw = consumer, raw
    props = SimpleNamespace(
        name="cpu test", major=12, minor=1, multi_processor_count=1, L2_cache_size=128
    )
    monkeypatch.setattr(paired, "install_profiler_guard", lambda: None)
    monkeypatch.setattr(paired, "profiler_clean", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *args: None)
    for name in ("randn", "empty"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *a, fn=original, **k: fn(*a, **(k | {"device": "cpu"}))
        )

    class Activation:
        provider = "test"

    candidates = {name: Activation() for name in ("ours", "aa", "vllm", "sglang", "cute_test")}
    monkeypatch.setitem(paired.OPERATIONS, "silu_mul", lambda *a: (candidates, {}))
    checked = []

    def check(actual, inputs, out):
        assert actual is candidates
        checked.append(True)
        return {"passed": True}

    captured = []

    def capture(fn, inputs, outputs, layers):
        assert checked and layers == 2
        if consumer:
            assert isinstance(fn, paired.DownProjCall)
            assert fn.op in candidates.values()
            assert fn.w_down.shape == (16, 4096) and fn.y.shape == (2, 4096)
            assert fn.w_down.dtype == inputs[0].dtype == fn.y.dtype
        else:
            assert fn in candidates.values()
        captured.append(fn)
        return fn

    measured = {name: [1.123456] * 30 for name in candidates}
    monkeypatch.setattr(paired, "check_candidates", check)
    monkeypatch.setattr(paired, "capture_bank", capture)
    monkeypatch.setattr(paired, "measure", lambda *a: measured)
    row = paired.run_row(args, 2, "hot")
    assert len(captured) == len(candidates)
    assert ("consumer" in row) == consumer and ("raw" in row) == raw
    if consumer:
        assert row["consumer"] == "down_proj_mm_4096"
        assert all(fn.w_down is captured[0].w_down and fn.y is captured[0].y for fn in captured)
    if raw:
        assert row["raw"] == {name: [1.1235] * 30 for name in candidates}
