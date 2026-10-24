# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU regressions for measured admission and acceptance/step bookkeeping."""

import json
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import linear_smallm as bench
from benchmark.kernel_tuning import linear_smallm_probe as tuning
from benchmark.kernel_tuning import linear_smallm_table as table
from ctikernels.linear import cublas
from ctikernels.linear._dispatch import M_BUCKETS, Config


def test_market_ours_keeps_dispatch_fallback(monkeypatch):
    from ctikernels.linear import inference

    class Operation:
        provider = "cublas"
        kernels_per_call = 1

        def __init__(self, *, provider):
            assert provider == "auto"

        def __call__(self, x, weight):
            return "fallback output"

    monkeypatch.setattr(inference, "Linear", Operation)
    providers, _ = bench.discover_providers()
    assert providers["ours"](None, None) == "fallback output"
    assert providers["ours"].provider == "cublas"


def test_market_result_accuracy_and_exception_message(monkeypatch):
    import torch

    randn = torch.randn
    monkeypatch.setattr(bench.torch, "randn", lambda *a, **kw: randn(*a, dtype=kw["dtype"]))
    monkeypatch.setattr(
        bench, "Instance", lambda name, call, x, w: SimpleNamespace(output=call(x, w))
    )
    monkeypatch.setattr(bench, "paired", lambda *a: ([10.0] * 20, [10.0] * 20))
    monkeypatch.setattr(bench, "emit", lambda r: None)
    lengths = []
    accuracy = bench.accuracy

    def check(*args):
        lengths.append(args[-1])
        return accuracy(*args)

    def broken(x, w):
        raise RuntimeError("capture diagnostic " + "x" * 400)

    monkeypatch.setattr(bench, "accuracy", check)
    providers, _ = bench.discover_providers()
    providers["broken"] = broken
    args = SimpleNamespace(
        cold=True, pairs=20, reps_per_sample=1, providers=["cublas", "ours", "broken"], trace=False
    )
    rows = bench.run_shape(
        args,
        7,
        8,
        3,
        [("test", {"name": "test", "count_per_step": 1})],
        providers,
        {},
        None,
        1e9,
        "CPU",
    )
    assert lengths == [8]
    assert rows[1]["status"] == "ok" and rows[1]["actual_provider"] == "cublas"
    assert rows[1]["correct"] and rows[1]["logical_kernels_per_call"] == 1
    assert rows[2]["phase"] == "capture_or_check"
    assert rows[2]["reason"].startswith("RuntimeError: capture diagnostic ")
    assert len(rows[2]["reason"]) == 300


def test_market_accuracy_uses_full_precision_cublas(monkeypatch):
    import torch

    matmul = torch.backends.cuda.matmul
    flags = []

    def linear(x, w):
        # Encode the reduced-precision flag active at the call in the output.
        flags.append(matmul.allow_bf16_reduced_precision_reduction)
        return torch.full((x.shape[0], w.shape[0]), float(flags[-1]), dtype=torch.bfloat16)

    randn = torch.randn
    monkeypatch.setattr(bench.torch, "randn", lambda *a, **kw: randn(*a, dtype=kw["dtype"]))
    monkeypatch.setattr(bench, "F", SimpleNamespace(linear=linear))
    monkeypatch.setattr(
        bench, "Instance", lambda name, call, x, w: SimpleNamespace(output=call(x, w))
    )
    monkeypatch.setattr(bench, "paired", lambda *a: ([10.0] * 20, [10.0] * 20))
    monkeypatch.setattr(bench, "emit", lambda r: None)
    baselines = []

    def accuracy(output, baseline, *args):
        baselines.append(baseline)
        return {"correct": True}

    monkeypatch.setattr(bench, "accuracy", accuracy)
    args = SimpleNamespace(
        cold=True, pairs=20, reps_per_sample=1, providers=["cublas", "ours"], trace=False
    )
    providers = {"cublas": linear, "ours": linear}
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = bench.MARKET_BF16_REDUCED_PRECISION
    try:
        rows = bench.run_shape(
            args,
            7,
            8,
            3,
            [("test", {"name": "test", "count_per_step": 1})],
            providers,
            {},
            None,
            1e9,
            "CPU",
        )
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous
    assert bench.MARKET_BF16_REDUCED_PRECISION is True
    # Timed instances run under the market flag; the accuracy baseline does not.
    assert len(baselines) == 1 and bool((baselines[0] == 0).all())
    assert flags.count(False) == 1 and flags.count(True) >= 2
    assert all(row["cublas_bf16_reduced_precision"] is True for row in rows)


def probe(m=1, ratio=1.03, cfg=None, **updates):
    return dict(
        kind="smallm_probe",
        device_class="rtx5090",
        N=24576,
        K=4096,
        M=m,
        config=asdict(cfg or Config()),
        paired_speedup=ratio,
        ours_us=100,
        correct=True,
        status="ok",
        provider="cute",
        **updates,
    )


def test_winners_ties_holes_and_unmeasured_devices():
    faster = Config(128, 128, 2, 1, 8)
    rows = [
        probe(m=1),
        probe(m=2, ratio=1.0299),
        probe(m=3),
        probe(m=4),
        probe(m=4, ratio=1.04, cfg=faster),
    ]
    chosen = table.select_configs(rows)
    assert chosen["rtx5090"][24576, 4096] == {(1, 1): Config(), (3, 3): Config(), (4, 4): faster}
    assert chosen["pro6000"] == chosen["gb10"] == {}
    assert (34816, 5120) not in chosen["rtx5090"]
    assert table.select_configs([probe(m=1), probe(m=2)])["rtx5090"][24576, 4096] == {
        (1, 2): Config()
    }


def test_repeats_use_median_and_incorrect_or_warm_do_not_qualify():
    rows = [probe(ratio=1.2), probe(ratio=0.6)]
    assert table.select_configs(rows)["rtx5090"] == {}
    assert table.select_configs([probe(mode="warm")])["rtx5090"] == {}
    bad = probe()
    bad.update(status="incorrect", correct=False)
    assert table.select_configs([probe(), bad])["rtx5090"] == {}


def test_legacy_context_resets_per_log_and_gb10_can_be_added(tmp_path):
    legacy = probe()
    legacy.pop("device_class")
    paths = [tmp_path / "first.log", tmp_path / "second.log"]
    environment = {"kind": "probe_environment", "N": 24576, "K": 4096, "device_class": "gb10"}
    paths[0].write_text(
        "ignored banner\n" + "\n".join("RESULT " + json.dumps(r) for r in [environment, legacy])
    )
    paths[1].write_text("RESULT " + json.dumps(legacy))
    rows = table.read_logs(paths)
    assert rows[-1]["device_class"] is None
    chosen = table.select_configs(rows)
    assert chosen["gb10"][24576, 4096] == {(1, 1): Config()}
    target = tmp_path / "dispatch.py"
    target.write_text("before\n" + table.BEGIN + "\nstale\n" + table.END + "\nafter\n")
    table.rewrite(target, chosen)
    text = target.read_text()
    assert text.startswith("before\n") and text.endswith("\nafter\n")
    assert "stale" not in text
    namespace = {"Config": Config}
    exec("result = " + table.python_literal(chosen), namespace)
    assert namespace["result"] == chosen


def test_acceptance_protocol_and_boundary():
    args = bench.parse_args(["--step", "--model", "all"])
    assert args.accept and args.step and args.pairs == 20 and args.reps_per_sample == 1
    assert args.m == [1, 2, 3, 4, 6, 8]
    assert bench.parse_args([]).reps_per_sample == 5
    for argv in [["--accept", "--warm"], ["--accept", "--pairs", "8"], ["--step", "--m", "65"]]:
        with pytest.raises(SystemExit):
            bench.parse_args(argv)
    assert bench.acceptance_verdict(0.99, 0.01, True) == "PASS"
    assert bench.acceptance_verdict(0.98, 0.01, True) == "FAIL"
    assert bench.acceptance_verdict(1.1, 0.01, False) == "FAIL"


@pytest.mark.parametrize("pairs", [20, 21, 59, 60, 80])
def test_acceptance_allows_at_least_twenty_pairs(pairs):
    assert bench.parse_args(["--accept", "--pairs", str(pairs)]).pairs == pairs
    assert bench.parse_args(["--step", "--pairs", str(pairs)]).pairs == pairs


def test_acceptance_rejects_nineteen_pairs():
    with pytest.raises(SystemExit):
        bench.parse_args(["--accept", "--pairs", "19"])


def test_bootstrap_speedup_ci_deterministic_and_known_examples():
    state = random.getstate()
    samples = [1.0, 1.1, 1.2, 1.3, 2.0, 4.0]
    seed = "4096:4096:3:cold"
    result = bench.bootstrap_speedup_ci(samples, seed)
    assert bench.bootstrap_speedup_ci(samples, seed) == result
    assert random.getstate() == state
    assert result[0] < 1.25 < result[1]
    assert bench.bootstrap_speedup_ci([1.2] * 60, seed) == (1.2, 1.2)
    # For five draws, the median is 10 iff at least three draws pick the outlier
    # (probability 0.05792); thus the 2.5/97.5 percentiles are 1 and 10.
    assert bench.bootstrap_speedup_ci([1.0, 1.0, 1.0, 1.0, 10.0], seed) == (1.0, 10.0)
    assert bench.percentile([1, 2, 3, 4], 0.025) == pytest.approx(1.075)


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_bootstrap_rejects_invalid_samples(samples):
    with pytest.raises(ValueError, match="paired speedups"):
        bench.bootstrap_speedup_ci(samples, 0)


@pytest.mark.parametrize("pairs", [20, 59, 60])
def test_gb10_low_pair_warning_does_not_stop_acceptance(monkeypatch, capsys, pairs):
    monkeypatch.setattr(bench.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        bench.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=1, multi_processor_count=48),
    )
    monkeypatch.setattr(bench.torch.cuda, "empty_cache", lambda: None)
    measured = []

    def run_shape(args, *unused):
        measured.append(args.pairs)
        return [{"verdict": "PASS"}]

    monkeypatch.setattr(bench, "run_accept_shape", run_shape)
    args = bench.parse_args(["--accept", "--pairs", str(pairs)])
    assert bench.run_acceptance(args, {(8, 8): []}, None, "GB10") == 0
    assert measured == [pairs]
    captured = capsys.readouterr()
    assert ("cannot qualify CuTe rows" in captured.err) is (pairs < 60)
    assert '"kind":"accept_completion"' in captured.out


def test_acceptance_clears_and_restores_overrides(monkeypatch):
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "cute")
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "16,64,2,32")
    with bench.shipped_dispatch():
        assert bench.os.environ["CTI_LINEAR_PROVIDER"] == "auto"
        assert "CTI_LINEAR_CONFIG" not in bench.os.environ
    assert bench.os.environ["CTI_LINEAR_PROVIDER"] == "cute"
    assert bench.os.environ["CTI_LINEAR_CONFIG"] == "16,64,2,32"


def test_step_layouts_alias_counts_and_deltas():
    rows = []
    for model, shapes in bench.builtin_shapes().items():
        for m in (1, 2, 3, 4, 6, 8):
            for shape in shapes:
                rows.append(
                    dict(
                        shape,
                        kind="smallm_accept",
                        device="RTX 5090",
                        model=model,
                        M=m,
                        ours_us=m,
                        cublas_us=2 * m,
                        verdict="PASS",
                    )
                )
    steps = bench.step_estimates(rows)
    assert len(steps) == (len(bench.builtin_shapes()) * 2 - 1) * 6
    for row in steps:
        if row["model"] == "qwen3.5-9b":
            assert row["calls_per_step"] == (129 if row["layout"] == "fused" else 153)
        count, m = row["calls_per_step"], row["M"]
        assert row["ours_step_ms"] == pytest.approx(count * m / 1000)
        assert row["ours_delta_ms"] == pytest.approx(count * (m - 1) / 1000)
        assert row["cublas_delta_ms"] == pytest.approx(2 * count * (m - 1) / 1000)


def test_markdown_emits_all_measurement_types_and_public_device_labels():
    text = table.markdown(
        [
            probe(),
            {"kind": "smallm_accept", "device": "NVIDIA GB10", "verdict": "PASS"},
            {"kind": "smallm_step", "device": "NVIDIA RTX PRO 6000", "M": 4},
            {"device": "unexpected-label", "us_median": 10, "provider": "cublas"},
        ]
    )
    for label in (
        "RTX 5090",
        "RTX PRO 6000",
        "GB10",
        "Acceptance",
        "Decode-step",
        "Market",
        "PASS",
    ):
        assert label in text
    assert "unexpected-label" not in text


@pytest.mark.parametrize("copies", [None, (2, 5)])
def test_pair_order_and_flush_outside_timing(monkeypatch, copies):
    calls = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            calls.append("event")

        def synchronize(self):
            calls.append("sync")

        def elapsed_time(self, other):
            return 0.01

    monkeypatch.setattr(bench.torch.cuda, "Event", Event)

    def instance(name, side):
        result = SimpleNamespace(graph=SimpleNamespace(replay=lambda: calls.append(name)))
        if copies is not None:
            result.copies = copies[side]
        return result

    a, b = bench.paired(
        instance("left", 0), instance("right", 1), 2, 1, lambda: calls.append("flush")
    )
    assert a == [10 / (copies[0] if copies else 1)] * 2
    assert b == [10 / (copies[1] if copies else 1)] * 2
    assert calls == [
        v
        for name in ("left", "right", "right", "left")
        for v in ("flush", "event", name, "event", "sync")
    ]


@pytest.mark.parametrize("copies", [None, 2, 3])
def test_kernels_per_replay_normalizes_cuda_events(monkeypatch, copies):
    torch = bench.torch
    events = [SimpleNamespace(device_type=torch.autograd.DeviceType.CUDA) for _ in range(6)]
    events.append(SimpleNamespace(device_type=torch.autograd.DeviceType.CPU))
    calls = []

    def profile(*, activities):
        assert activities == [torch.profiler.ProfilerActivity.CUDA]
        return nullcontext(SimpleNamespace(events=lambda: events))

    monkeypatch.setattr(torch.profiler, "profile", profile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    instance = SimpleNamespace(graph=SimpleNamespace(replay=lambda: calls.append("replay")))
    if copies is not None:
        instance.copies = copies
    assert bench.kernels_per_replay(instance) == 6 // (copies or 1)
    assert calls == ["replay", "sync"]


@pytest.mark.parametrize(
    "n,k,l2_bytes,expected,meets_target",
    [
        (1024, 2048, 32 << 20, 256, True),  # 1 GiB floor
        (4096, 4096, 512 << 20, 48, True),  # 3x L2 dominates
        (2049, 2048, 32 << 20, 128, True),  # round up partial copies
        (32768, 32768, 32 << 20, 2, True),  # minimum two copies
        (1, 8, 32 << 20, 256, False),  # copy-count cap
        (5000, 10000, 3 << 30, 60, False),  # 6 GB cap
    ],
)
def test_acceptance_copies_bounds(n, k, l2_bytes, expected, meets_target):
    copies = bench.acceptance_copies(n, k, l2_bytes)
    assert copies == expected
    assert 2 <= copies <= 256
    total_bytes = copies * 2 * n * k
    assert total_bytes <= 6e9
    if meets_target:
        assert total_bytes >= 3 * l2_bytes
        assert total_bytes >= 1 << 30
    else:
        assert copies == 256 or (copies + 1) * 2 * n * k > 6e9


def test_default_margin_rejects_near_ties():
    rows = [probe(m=1, ratio=1.02), probe(m=2, ratio=1.04)]
    assert set(table.select_configs(rows)["rtx5090"][24576, 4096]) == {(2, 2)}


def launch(m, cfg=None):
    cfg = cfg or Config()
    return {"config": asdict(cfg), "mtiles": (m + 7) // 8, "cta_threads": 32 * cfg.warps}


def acceptance(m=1, ratio=1.031, cfg=None, **updates):
    return {
        "launch_config": launch(m, cfg),
        "kind": "smallm_accept",
        "device_class": "rtx5090",
        "N": 24576,
        "K": 4096,
        "M": m,
        "name": "projection",
        "paired_speedup": ratio,
        "correct": True,
        "verdict": "PASS",
        "provider": "cute",
        "cublas_bf16_reduced_precision": True,
        "_complete": True,
        "_finished": True,
        "_round": "current round",
        **updates,
    }


def with_engine(rows):
    """Give each cold CuTe acceptance row a matching engine-regime twin.

    Lists that already carry engine-regime rows are returned unchanged.
    """
    rows = list(rows)
    if any(row.get("l2") == "engine" for row in rows):
        return rows
    twinned = []
    for row in rows:
        twinned.append(row)
        if (
            row.get("kind") == "smallm_accept"
            and row.get("provider") == "cute"
            and row.get("l2", "cold") == "cold"
        ):
            twinned.append(dict(row, l2="engine"))
    return twinned


def gate(chosen, rows, **kwargs):
    """Gate cold evidence that also passed the (required) engine regime."""
    return table.gate_by_acceptance(chosen, with_engine(rows), **kwargs)


@pytest.mark.parametrize("path", ["bf16", "fp32", "gated", "merged"])
@pytest.mark.parametrize("regime", ["cold", "engine", "hot"])
@pytest.mark.parametrize("edge", [3, 4])
@pytest.mark.parametrize(
    "update,kept",
    [
        ({"pairs": 20}, False),
        ({"pairs": 59}, False),
        ({"ci_low": None}, False),
        ({"ci_low": 0.99}, False),
        ({"ci_low": 1.0}, False),
        ({"ci_low": float("nan")}, False),
        ({"ci_low": 1.01}, True),
        ({"epsilon": 0.3}, False),
        ({"paired_speedup": 1.03}, False),
    ],
)
def test_gb10_qualification_every_record(path, regime, edge, update, kept):
    dtype = "fp32" if path == "fp32" else "bf16"
    shape = (64, 4096) if path == "merged" else (24576, 4096)
    models = ("qwen3.5-9b", "deepseek-v4-flash") if path == "merged" else ("test",)
    if path == "merged":
        families = table.model_families()[0]
        models = tuple(next(m for m, f in families.items() if f == fam) for fam in ("qwen", "dsv4"))
    rows = [
        acceptance(
            m=m,
            ratio=1.2,
            N=shape[0],
            K=shape[1],
            model=model,
            l2=l2,
            device_class="gb10",
            out_dtype=dtype,
            pairs=60,
            ci_low=1.01,
            ci_high=1.3,
        )
        for model in models
        for l2 in ("cold", "engine", "hot")
        for m in (3, 4)
    ]
    if path == "gated":
        shape = (*shape, 8192)
        for row in rows:
            row.update(
                kind="gate_accept",
                provider="cute-gated",
                gate_stride=8192,
                market_bf16_reduced_precision=True,
            )
            row["launch_config"]["gate_stride"] = 8192
    chosen = {"gb10": {shape: {(3, 4): Config()}}}
    assert table.gate_by_acceptance(chosen, rows, gated=path == "gated", out_dtype=dtype)["gb10"]
    target = next(r for r in rows if r["l2"] == regime and r["M"] == edge)
    target.update(update)
    if update == {"ci_low": None}:
        target.pop("ci_low")  # Legacy records have no CI fields at all.
    result = table.gate_by_acceptance(chosen, rows, gated=path == "gated", out_dtype=dtype)
    assert bool(result["gb10"]) is kept
    # Each regime must also exist at both edges, including hot on GB10.
    rows.remove(target)
    assert not table.gate_by_acceptance(chosen, rows, gated=path == "gated", out_dtype=dtype)[
        "gb10"
    ]


@pytest.mark.parametrize("kind", ["rtx5090", "pro6000"])
def test_rtx_legacy_twenty_pairs_without_ci_still_qualify(kind):
    chosen = {kind: {(24576, 4096): {(3, 4): Config()}}}
    rows = [acceptance(m=m, device_class=kind, pairs=20) for m in (3, 4)]
    assert gate(chosen, rows)[kind] == chosen[kind]


def test_gb10_intervals_in_all_plain_reports():
    row = acceptance(
        ratio=1.071,
        device_class="gb10",
        device="GB10",
        model="qwen3-8b",
        name="mlp.gate_up",
        ours_us=10.0,
        cublas_us=10.71,
        sol_us=8.0,
        epsilon=0.01,
        ci_low=1.052,
        ci_high=1.09,
        pairs=60,
        count_per_step=36,
    )
    assert "1.0710 [1.0520, 1.0900]" in table.markdown([row])
    assert "1.0710 [1.0520, 1.0900]" in table.fp32_markdown([dict(row, out_dtype="fp32")])
    chosen = {"gb10": {(24576, 4096): {(1, 1): Config()}}}
    assert "1.071 [1.052, 1.090]" in table.bxq_markdown([row], chosen)
    fallback = table.bxq_markdown([row], {})
    assert "F.linear fallback (CuTe measured 1.071 [1.052, 1.090])" in fallback
    assert "1.000 [1.052, 1.090]" not in fallback


def test_cold_pass_without_finished_engine_evidence_falls_back():
    chosen = {"rtx5090": {(24576, 4096): {(5, 8): Config()}}}
    cold = [acceptance(m=5, ratio=1.2), acceptance(m=8, ratio=1.2)]
    engine = [dict(row, l2="engine") for row in cold]
    expected = chosen["rtx5090"]
    assert table.gate_by_acceptance(chosen, cold + engine)["rtx5090"] == expected
    # Engine records absent: cuBLAS.
    assert table.gate_by_acceptance(chosen, cold)["rtx5090"] == {}
    # Engine record at one bucket edge only: cuBLAS.
    assert table.gate_by_acceptance(chosen, cold + engine[:1])["rtx5090"] == {}
    # Unfinished engine run, even with a losing 0.90x timing: cuBLAS.
    unfinished = [dict(row, _finished=False, _complete=False) for row in engine]
    assert table.gate_by_acceptance(chosen, cold + unfinished)["rtx5090"] == {}
    slow = [dict(unfinished[0], paired_speedup=0.90, verdict="FAIL"), unfinished[1]]
    assert table.gate_by_acceptance(chosen, cold + slow)["rtx5090"] == {}
    # A finished engine run that does not pass: cuBLAS.
    losing = [dict(row, paired_speedup=0.90, verdict="FAIL") for row in engine]
    assert table.gate_by_acceptance(chosen, cold + losing)["rtx5090"] == {}


def test_shared_inventory_keys_are_declared_merges():
    # Two model families producing the same (dtype, N, K) key share one table row,
    # so every such overlap must be declared and gated per family.
    assert table.shared_keys() == table.MERGED_KEYS


def test_shared_key_keeps_cute_only_where_every_family_qualifies():
    cfg = Config(16, 128, 2, 32, 1, "reg", 4)
    chosen = {"rtx5090": {(64, 4096): {(3, 4): cfg}}}

    def site(model, ratio=1.2, verdict="PASS"):
        return with_engine(
            [
                acceptance(m=m, ratio=ratio, cfg=cfg, N=64, model=model, verdict=verdict)
                for m in (3, 4)
            ]
        )

    qwen, dsv4 = site("qwen3.5-9b"), site("dsv4-flash")
    assert gate(chosen, qwen + dsv4)["rtx5090"] == chosen["rtx5090"]
    # One family's evidence alone (either one) cannot decide the shared row.
    assert gate(chosen, dsv4)["rtx5090"] == {}
    assert gate(chosen, qwen)["rtx5090"] == {}
    # Evidence without a model tag counts for no family.
    assert gate(chosen, site(None) + dsv4)["rtx5090"] == {}
    # A win at one site and a loss at the other: cuBLAS.
    assert gate(chosen, site("qwen3.5-9b", 0.9, "FAIL") + dsv4)["rtx5090"] == {}
    # Unshared keys keep the per-key rule.
    unshared = {"rtx5090": {(24576, 4096): {(3, 4): Config()}}}
    rows = [acceptance(m=m, ratio=1.2, model="qwen3.5-9b") for m in (3, 4)]
    assert gate(unshared, rows)["rtx5090"] == unshared["rtx5090"]


def test_acceptance_gate_drops_slow_missing_and_boundary_rows():
    chosen = table.select_configs([probe(m=m, ratio=1.2) for m in range(1, 6)])
    rows = [acceptance(m=1, ratio=1.0299), acceptance(m=3), acceptance(m=4, ratio=1.1)]
    gated = gate(chosen, rows)
    assert gated == {"rtx5090": {(24576, 4096): {(3, 4): Config()}}, "pro6000": {}, "gb10": {}}
    assert chosen["rtx5090"][24576, 4096] == {(1, 5): Config()}
    # Exactly 1.03 is not strictly above the margin (shared ``qualifies`` gate).
    rows[1] = acceptance(m=3, ratio=1.03)
    assert gate(chosen, rows)["rtx5090"] == {}


def test_acceptance_gate_requires_both_bucket_edges():
    cfg = Config(splits=3)
    chosen = {"rtx5090": {(24576, 4096): {(57, 64): cfg}}}
    upper = acceptance(m=64, ratio=1.2, cfg=cfg)
    assert gate(chosen, [upper])["rtx5090"] == {}
    lower = acceptance(m=57, ratio=1.2, cfg=cfg)
    assert gate(chosen, [upper, lower])["rtx5090"] == {(24576, 4096): {(57, 64): cfg}}
    lower["paired_speedup"] = 1.02
    assert gate(chosen, [upper, lower])["rtx5090"] == {}


@pytest.mark.parametrize("reduced_precision", [False, None])
def test_acceptance_requires_market_bf16_reduced_precision(reduced_precision):
    chosen = table.select_configs([probe(ratio=1.5)])
    row = acceptance(ratio=1.5, cublas_bf16_reduced_precision=reduced_precision)
    if reduced_precision is None:
        del row["cublas_bf16_reduced_precision"]
    assert table.accepted_speedups([row]) == {}
    assert gate(chosen, [row])["rtx5090"] == {}
    passing = dict(row, cublas_bf16_reduced_precision=True)
    expected = {(24576, 4096): {(1, 1): Config()}}
    assert gate(chosen, [passing])["rtx5090"] == expected
    # Non-market aliases are skipped, even when their timing verdict fails.
    ignored = dict(row, name="other", paired_speedup=0.9, verdict="FAIL")
    assert gate(chosen, [passing, ignored])["rtx5090"] == expected
    assert gate(chosen, [passing, dict(ignored, correct=False)])["rtx5090"] == {}


@pytest.mark.parametrize(
    "updates",
    [{"_complete": False}, {"correct": False}, {"verdict": "FAIL"}, {"provider": "cublas"}],
)
def test_acceptance_gate_requires_complete_correct_pass_and_cute(updates):
    chosen = table.select_configs([probe()])
    assert gate(chosen, [acceptance(**updates)])["rtx5090"] == {}


def test_acceptance_gate_uses_minimum_over_projection_aliases():
    chosen = table.select_configs([probe(m=1), probe(m=2)])
    rows = [
        acceptance(m=1, ratio=1.1, name="first"),
        acceptance(m=1, ratio=1.02, name="second"),
        acceptance(m=2, ratio=1.1, name="first"),
        acceptance(m=2, ratio=1.031, name="second"),
    ]
    assert gate(chosen, rows)["rtx5090"][24576, 4096] == {(2, 2): Config()}


def test_acceptance_gate_uses_latest_current_run_per_device():
    chosen = table.select_configs(
        [probe(m=m) for m in (1, 2, 3)] + [{**probe(), "device_class": "gb10"}]
    )
    rows = [
        acceptance(m=1, _source=10, _round="previous round"),
        acceptance(m=2, _source=0),
        acceptance(m=3, _source=1),
        acceptance(m=4, _source=1),
        *[
            acceptance(device_class="gb10", _source=0, pairs=60, ci_low=1.01, l2=l2)
            for l2 in ("cold", "hot")
        ],
    ]
    gated = gate(chosen, rows)
    assert gated["rtx5090"][24576, 4096] == {(3, 4): Config()}
    assert gated["gb10"][24576, 4096] == {(1, 1): Config()}


def test_previous_acceptance_alone_never_qualifies_dispatch():
    chosen = table.select_configs([probe(m=1, ratio=1.2)])
    rows = [acceptance(m=1, ratio=1.2, _source=0, _round="previous round")]
    assert gate(chosen, rows)["rtx5090"] == {}
    supplied = [acceptance(m=1, ratio=1.2, _source=0, _round="supplied round")]
    assert gate(chosen, supplied)["rtx5090"] == {}


def test_cli_rewrite_requires_current_accept_logs(tmp_path, capsys):
    log = tmp_path / "previous.log"
    log.write_text("")
    with pytest.raises(SystemExit):
        table.main(["--previous-accept", str(log)])
    assert "--accept" in capsys.readouterr().err


@pytest.mark.parametrize("margin", ["nan", "inf", "-inf", "0.01", "0", "-1"])
def test_invalid_margin_rejected_by_selection_and_cli(margin, capsys):
    message = "--margin must be finite and >= 0.03"
    with pytest.raises(ValueError, match=message):
        table.select_configs([probe(ratio=1.2)], margin=float(margin))
    with pytest.raises(SystemExit) as exc:
        table.main([f"--margin={margin}"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("with_probes", [False, True])
def test_dispatch_cli_requires_acceptance_rows(tmp_path, capsys, rewrite, with_probes):
    source = tmp_path / "probe.log"
    source.write_text("RESULT " + json.dumps(probe(ratio=1.2)) if with_probes else "")
    target = tmp_path / "dispatch.py"
    target.write_text("unchanged\n")
    argv = [str(source)] + (["--rewrite", str(target)] if rewrite else [])
    with pytest.raises(SystemExit) as exc:
        table.main(argv)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "dispatch generation requires current acceptance RESULT rows" in captured.err
    assert not captured.out
    assert target.read_text() == "unchanged\n"


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("with_probes", [False, True])
def test_dispatch_cli_gates_probes_or_committed_table(
    tmp_path, monkeypatch, capsys, rewrite, with_probes
):
    from ctikernels.linear import _dispatch as dispatch

    cfg = Config(splits=2)
    probes = [probe(m=m, ratio=1.2, cfg=cfg) for m in (1, 2, 3)]
    committed = table.select_configs(probes)
    monkeypatch.setattr(dispatch, "TUNED", committed)
    rows = (probes if with_probes else []) + [
        acceptance(m=1, ratio=1.02, cfg=cfg),
        acceptance(m=3, cfg=cfg),
        acceptance(m=4, cfg=cfg),
        acceptance(m=1, ratio=1.02, cfg=cfg, l2="engine"),
        acceptance(m=3, cfg=cfg, l2="engine"),
        acceptance(m=4, cfg=cfg, l2="engine"),
        {"kind": "accept_completion", "complete": True, "failures": 0},
    ]
    source = tmp_path / "current.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
    target = tmp_path / "dispatch.py"
    target.write_text(table.BEGIN + "\nTUNED = {}\n" + table.END + "\n")
    argv = ["--accept", str(source)] + (["--rewrite", str(target)] if rewrite else [])
    assert table.main(argv) == 0
    captured = capsys.readouterr()
    namespace = {"Config": Config}
    exec(target.read_text() if rewrite else "TUNED = " + captured.out, namespace)
    assert namespace["TUNED"] == {
        "rtx5090": {(24576, 4096): {(3, 4): cfg}},
        "pro6000": {},
        "gb10": {},
    }
    assert dispatch.TUNED["rtx5090"][24576, 4096] == {(1, 3): cfg}


@pytest.mark.parametrize(
    "returncode,finished,reported_crash,expected",
    [
        (1, True, False, None),  # completed correctness failure, not a crash
        (0, True, False, None),
        (1, False, False, "child_exit"),
        (-11, True, False, "child_exit"),  # fault during teardown
        (0, False, False, "child_exit"),  # no completion record
        (1, False, True, None),  # already reported by the child
    ],
)
def test_child_failure_classification(returncode, finished, reported_crash, expected):
    assert tuning.child_failure(returncode, finished, reported_crash) == expected


def test_shape_subset_parent_correctness_exit(monkeypatch, capsys):
    # Replay the observed sequence at the parent seam: an earlier candidate
    # fails, the winner succeeds at M=8, shape_completion reports failure.
    children = []

    class Child:
        returncode = 1

        def __init__(self, cmd, **kwargs):
            children.append(self)
            _, self.args = json.loads(cmd[-1])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self):
            rows = [
                {**probe(), "status": "incorrect", "correct": False},
                {**probe(m=8), "kind": "probe_progress", "phase": "timing"},
                probe(m=8),
                {"kind": "shape_completion", "status": "incorrect", "failures": 1},
            ]
            return "\n".join("RESULT " + json.dumps(r) for r in rows), None

    monkeypatch.setattr(tuning.subprocess, "Popen", Child)
    monkeypatch.setattr(tuning, "START", tuning.time.monotonic())
    monkeypatch.setattr(sys, "argv", ["probe", "--shape", "24576", "4096", "--m", "1,8", "--tune"])
    assert tuning.main() == 1
    assert len(children) == 1
    assert "--child" in children[0].args
    assert "--pairs" not in children[0].args  # preserve device-aware default
    rows = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines()]
    crashes = [r for r in rows if r.get("status") == "crash"]
    assert crashes == []
    assert rows[-1]["failures"] == 1 and rows[-1]["expected_shapes"] == 1


@pytest.mark.parametrize("expired_before_start", [False, True])
def test_probe_budget_waits_for_running_child_and_starts_no_more_shapes(
    monkeypatch, capsys, expired_before_start
):
    now = 2 if expired_before_start else 0
    children = []

    class Child:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            children.append(self)
            self.awaited = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert self.awaited

        def communicate(self):
            nonlocal now
            now = 2  # The child finishes after the one-second budget.
            self.awaited = True
            return 'RESULT {"kind": "shape_completion", "status": "ok", "failures": 0}', None

    monkeypatch.setattr(tuning.subprocess, "Popen", Child)
    monkeypatch.setattr(tuning.time, "monotonic", lambda: now)
    monkeypatch.setattr(tuning, "START", 0)
    monkeypatch.setattr(tuning, "builtin_shapes", lambda: {"qwen3.5-9b": []})
    monkeypatch.setattr(sys, "argv", ["probe", "--budget-seconds", "1"])
    assert tuning.main() == 1  # The remaining shapes were not started.
    assert len(children) == int(not expired_before_start)
    rows = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines()]
    assert all(row.get("status") != "crash" for row in rows)
    assert rows[-1] == {
        "out_dtype": "bf16",
        "kind": "probe_completion",
        "complete": False,
        "failures": 0,
        "shapes": len(children),
        "expected_shapes": 4,
        "elapsed_seconds": 2,
    }


def test_device_aware_candidates_pairs_and_table_roundtrip():
    configs = tuning.candidates("auto", Config(), 24576, 4096, "gb10")
    assert len(configs) == 10 and sum(c.variant == "reg" for c in configs) == 6
    assert tuning.pair_count("gb10", 248320, 4096, None, "tune") == 3
    assert tuning.pair_count("gb10", 248320, 4096, None, "best") == 6
    assert tuning.pair_count("gb10", 24576, 4096, None, "tune") == 6
    assert tuning.pair_count("rtx5090", 248320, 4096, None, "tune") == 8
    assert tuning.pair_count("gb10", 248320, 4096, 12, "tune") == 12
    rows = [{**probe(m=m, ratio=1.1, cfg=configs[7]), "device_class": "gb10"} for m in (1, 3)]
    chosen = table.select_configs(rows)
    assert chosen["gb10"][24576, 4096] == {(1, 1): configs[7], (3, 3): configs[7]}
    assert chosen["rtx5090"] == {}


def test_probe_checks_actual_shared_memory_limit_before_cuda_work(monkeypatch, capsys):
    props = SimpleNamespace(
        major=12,
        minor=1,
        multi_processor_count=48,
        name="NVIDIA GB10",
        shared_memory_per_block_optin=48 * 1024,
    )
    monkeypatch.setattr(tuning.torch.cuda, "get_device_properties", lambda _: props)
    # Restore probe overrides at test exit.
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    monkeypatch.delenv("CTI_LINEAR_CONFIG", raising=False)
    assert tuning.run_shape(SimpleNamespace(shape=(4096, 12288), tune="auto")) == 1
    result = json.loads(capsys.readouterr().out.strip()[7:])
    assert result["phase"] == "setup" and result["error"] == "ValueError"


@pytest.mark.parametrize("kind,rate", [("rtx5090", 1681.1), ("pro6000", 1533.1), ("gb10", 244.6)])
def test_sol_builtin_ceiling_and_unknown_device(monkeypatch, kind, rate):
    def absent(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(bench.importlib, "import_module", absent)
    calls = []
    bw, meta = bench.sol_bandwidth(kind, lambda: calls.append(True) or 123)
    assert bw == rate * 1e9 and not calls
    assert meta["sol_bandwidth_policy"] == "max(read-only,copy)"
    bw, meta = bench.sol_bandwidth(None, lambda: calls.append(True) or 123)
    assert bw == 123 and calls == [True] and meta["sol_source"] == "in_script_copy"


def test_sol_uses_merged_loader_and_preserves_invalid_table_errors(monkeypatch):
    requested = []

    def load(kind):
        requested.append(kind)
        return {"memory": {"dram_read_gbs": 1700, "dram_copy_gbs": 1400}}

    module = SimpleNamespace(load_ceilings=load)
    monkeypatch.setattr(bench.importlib, "import_module", lambda _: module)
    bw, meta = bench.sol_bandwidth("pro6000", lambda: pytest.fail("unexpected measurement"))
    assert requested == ["rtx-pro-6000"] and bw == 1700e9
    assert meta["sol_source"] == "benchmark.sol.sol"

    def invalid(kind):
        raise ValueError("invalid ceiling")

    module.load_ceilings = invalid
    with pytest.raises(ValueError, match="invalid ceiling"):
        bench.sol_bandwidth("gb10", lambda: 0)


def test_register_config_probe_and_table_roundtrip():
    cfg = Config(32, 128, 2, 3, 1, "reg", 4)
    assert tuning.candidates("32,128,2,3,1,reg,4", Config()) == [cfg]
    chosen = table.select_configs([probe(cfg=cfg, ratio=1.1)])
    namespace = {"Config": Config}
    exec("result = " + table.python_literal(chosen), namespace)
    assert namespace["result"] == chosen
    legacy = probe(ratio=1.1)
    legacy["config"].pop("variant")
    legacy["config"].pop("prefetch")
    assert table.select_configs([legacy])["rtx5090"][24576, 4096][1, 1] == Config()


def test_report_read_ceiling_errors_and_current_acceptance_precedence():
    market = {
        "device": "NVIDIA GB10",
        "mode": "cold",
        "model": "qwen3.5-9b",
        "N": 64,
        "K": 4096,
        "M": 1,
        "provider": "cublas",
        "status": "ok",
        "us_median": 10,
        "SOL_us": 999,
        "util": 999,
    }
    previous = {
        "kind": "smallm_accept",
        "device": "RTX 5090",
        "model": "qwen3.5-9b",
        "name": "gdn.in_proj_ba",
        "N": 64,
        "K": 4096,
        "M": 1,
        "verdict": "FAIL",
        "provider": "cute",
        "_round": "previous round",
        "_source": 4,
        "_finished": True,
    }
    current = {**previous, "verdict": "PASS", "_round": "current round", "_source": 0}
    rows = [
        market,
        {**market, "provider": "ours", "status": "error", "reason": "private diagnostic"},
        previous,
        current,
    ]
    result = table.markdown(rows)
    sol = 2 * (64 * 4096 + 4096 + 64) / (244.6 * 1000)
    assert f"10.000 / {sol / 10:.3f}" in result
    assert f"{sol:.4f}" in result and "999" not in result
    assert "error; rerun pending" in result and "private diagnostic" not in result
    assert "current round; 1/1 rows PASS" in result
    assert "previous round;" not in result
    assert "Pending: no decode-step RESULT rows supplied." in result
    assert table.latest_rows(rows, lambda r: r.get("kind") == "smallm_accept")["rtx5090"] == [
        current
    ]


def test_report_cli_missing_inputs_and_legacy_probe_completion(tmp_path, capsys):
    source = tmp_path / "probe.log"
    row = probe(ratio=0.9)
    row.pop("device_class")
    source.write_text(
        "\n".join(
            "RESULT " + json.dumps(r)
            for r in [
                {"kind": "probe_environment", "device_class": "gb10", "N": 24576, "K": 4096},
                row,
                {"kind": "probe_completion", "complete": True, "failures": 0},
            ]
        )
    )
    assert (
        table.main(
            ["--markdown", "--probe", str(source), "--accept", str(tmp_path / "missing.log")]
        )
        == 0
    )
    result = capsys.readouterr().out
    assert "runs with successful completion markers: 1" in result
    assert "0.9000 | no" in result  # show best probe even when it loses
    assert str(source) not in result
    assert "Pending: no finished acceptance RESULT rows supplied." in result


@pytest.mark.parametrize("mode", [[], ["--accept"], ["--step"]])
def test_bxq_and_explicit_m_parsing(mode):
    assert bench.parse_args([*mode, "--m", "bxq"]).m == list(bench.BXQ_MS)
    accept = bench.parse_args([*mode, "--m", "accept"]).m
    assert accept == sorted(accept) and set(bench.BXQ_MS) <= set(accept)
    assert {m for bucket in M_BUCKETS for m in bucket} <= set(accept)
    assert set(accept) - set(bench.BXQ_MS) == {17, 25, 33, 40, 41, 49, 56, 57}
    assert bench.parse_args([*mode, "--m", "64,9,33,9"]).m == [64, 9, 33]
    for bad in ("0", "65", "", "1,x", "-1"):
        with pytest.raises(SystemExit):
            bench.parse_args([*mode, "--m", bad])


def test_new_model_geometries_and_all_selection():
    models = bench.builtin_shapes()
    expected = {
        "qwen3-8b": [
            (6144, 4096, 36),
            (4096, 4096, 36),
            (24576, 4096, 36),
            (4096, 12288, 36),
            (151936, 4096, 1),
        ],
        "qwen3.5-27b-tp2": [
            (7168, 5120, 16),
            (5120, 3072, 16),
            (8240, 5120, 48),
            (5120, 3072, 48),
            (17408, 5120, 64),
            (5120, 8704, 64),
            (124160, 5120, 1),
        ],
        "dflash-qwen3-8b": [
            (4096, 20480, 1),
            (2048, 4096, 5),
            (6144, 4096, 5),
            (4096, 4096, 5),
            (24576, 4096, 5),
            (4096, 12288, 5),
        ],
        "dflash-qwen3.5-9b": [
            (4096, 32768, 1),
            (2048, 4096, 6),
            (6144, 4096, 6),
            (4096, 4096, 6),
            (24576, 4096, 6),
            (4096, 12288, 6),
        ],
        "dflash-qwen3.5-27b": [
            (5120, 40960, 1),
            (2048, 5120, 6),
            (6144, 5120, 6),
            (5120, 4096, 6),
            (34816, 5120, 6),
            (5120, 17408, 6),
        ],
    }
    for model, geometry in expected.items():
        rows = models[model]
        assert [(r["N"], r["K"], r["count_per_step"]) for r in rows] == geometry
        assert [r["name"] for r in rows] == (
            ["fc", "context_kv", "attention.qkv", "attention.o", "mlp.gate_up", "mlp.down"]
            if model.startswith("dflash")
            else ["attention.qkv", "attention.o", "mlp.gate_up", "mlp.down", "lm_head"]
            if model == "qwen3-8b"
            else [
                "attention.qkv",
                "attention.o",
                "gdn.in_proj_qkvzba",
                "gdn.out_proj",
                "mlp.gate_up",
                "mlp.down",
                "lm_head",
            ]
        )
        if model.startswith("dflash"):
            assert [r["row_domain"] for r in rows] == ["context"] * 2 + ["block"] * 4
    assert models["qwen3.5-27b-tp2"][2]["layout"] == "fused"
    assert models["qwen3.5-27b"][2]["N"] == 16480
    shapes = bench.load_shapes(bench.parse_args(["--model", "all"]))
    assert {model for aliases in shapes.values() for model, _ in aliases} == set(models)


def test_round_estimates_layouts_context_and_missing_measurements():
    rows = [
        dict(
            shape,
            kind="smallm_accept",
            device="RTX 5090",
            model=model,
            M=m,
            ours_us=m,
            cublas_us=2 * m,
            verdict="PASS",
        )
        for model, shapes in bench.builtin_shapes().items()
        for shape in shapes
        for m in bench.BXQ_MS
    ]
    rounds = bench.round_estimates(rows)
    target = next(r for r in rounds if r["model"] == "qwen3-8b" and r["B"] == 4 and r["Q"] == 16)
    assert target["M"] == 64 and target["calls_per_round"] == 145
    assert target["ours_round_ms"] == pytest.approx(145 * 64 / 1000)
    draft = next(
        r for r in rounds if r["model"] == "dflash-qwen3-8b" and r["B"] == 3 and r["Q"] == 9
    )
    assert draft["context_rows"] == 3 and draft["M"] == 27
    assert draft["ours_round_ms"] == pytest.approx((6 * 3 + 20 * 27) / 1000)
    assert draft["cublas_round_ms"] == 2 * draft["ours_round_ms"]
    assert {r["layout"] for r in rounds if r["model"].endswith("tp2")} == {"fused"}
    q6 = {(r["B"], r["M"]) for r in rounds if r["model"] == "qwen3.5-9b" and r["Q"] == 6}
    assert q6 == {(1, 6), (2, 12), (3, 18), (4, 24)}
    assert all(r["M"] in bench.BXQ_MS for r in rounds)
    partial = [r for r in rows if r["M"] == 16]
    assert all(not r["model"].startswith("dflash") for r in bench.round_estimates(partial))
    assert all(r["ours_delta_ms"] is None for r in bench.step_estimates(partial))


def test_bxq_ranges_expand_only_buckets_with_current_acceptance(tmp_path):
    ms = [*bench.BXQ_MS, 33, 63]
    cfg = Config(32, 128, 2, 3, 1, "reg", 4)
    selected = table.select_configs([probe(m=m, cfg=cfg) for m in reversed(ms)])
    ranges = selected["rtx5090"][24576, 4096]
    assert set(ranges) == {
        (1, 6),
        (8, 10),
        (12, 12),
        (15, 16),
        (18, 18),
        (20, 20),
        (24, 24),
        (27, 27),
        (32, 33),
        (36, 36),
        (48, 48),
        (63, 64),
    }
    accepted = [acceptance(m=m, cfg=cfg) for m in ms if m != 9]
    accepted += [acceptance(m=9, cfg=cfg, _round="previous round"), acceptance(m=7, cfg=cfg)]
    accepted += [acceptance(m=m, cfg=cfg) for m in (9, 17, 25, 40, 41, 57)]
    gated = gate(selected, accepted)
    covered = {m for lo, hi in gated["rtx5090"][24576, 4096] for m in range(lo, hi + 1)}
    assert covered == set(range(1, 49)) | set(range(57, 65))
    # No current M in 49..56: that whole bucket remains absent.
    assert not covered.intersection(range(49, 57))
    target = tmp_path / "dispatch.py"
    target.write_text(table.BEGIN + "\nTUNED = {}\n" + table.END + "\n")
    table.rewrite(target, gated)
    namespace = {"Config": Config}
    exec(target.read_text(), namespace)
    assert namespace["TUNED"] == gated
    for m in (0, 65):
        with pytest.raises(ValueError, match="dimensions"):
            table.select_configs([probe(m=m)])


@pytest.mark.parametrize("kind,rate", [("rtx5090", 1681.1), ("pro6000", 1533.1), ("gb10", 244.6)])
@pytest.mark.parametrize("l2", ["cold", "hot"])
@pytest.mark.parametrize("pairs", [20, 60])
def test_acceptance_bxq_allocation_sol_and_comparator(monkeypatch, kind, rate, l2, pairs):
    assert bench.hot_copies(2048, 4096) == 32 and bench.hot_copies(24576, 4096) == 16
    import torch

    randn = torch.randn
    sizes = []

    def allocate(*a, **kw):
        sizes.append(a)
        return randn(*a, dtype=kw["dtype"])

    monkeypatch.setattr(torch, "randn", allocate)
    monkeypatch.setattr(
        torch, "Generator", lambda **kw: SimpleNamespace(manual_seed=lambda seed: None)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(L2_cache_size=32 << 20)
    )
    captures, eager, comparisons = [], [], []
    linear, accuracy = bench.F.linear, bench.accuracy

    def eager_linear(x, w):
        output = linear(x, w)
        eager.append((len(captures), output))
        return output

    def instance(name, fn, x, w, rotations=()):
        captured = SimpleNamespace(output=fn(x, w), copies=1 + len(rotations))
        captures.append((name, captured, w, rotations))
        return captured

    def paired(left, right, count, reps, flush):
        dispatched, calibration, baseline = [entry[1] for entry in captures[-3:]]
        assert left is baseline
        assert right is calibration or right is dispatched
        assert count == pairs and reps == 1 and flush is None
        comparisons.append(right)
        return [10.0] * count, [5.0] * count

    def check_accuracy(output, baseline, *args):
        # The comparator is the eager F.linear result, before any graph capture.
        assert baseline is eager[-4][1]
        assert eager[-4][0] == len(captures) - 3
        return accuracy(output, baseline, *args)

    monkeypatch.setattr(bench.F, "linear", eager_linear)
    monkeypatch.setattr(bench, "Instance", instance)
    monkeypatch.setattr(bench, "paired", paired)
    monkeypatch.setattr(bench, "accuracy", check_accuracy)
    monkeypatch.setattr(bench, "kernels_per_replay", lambda _: 1)
    monkeypatch.setattr(bench, "emit", lambda _: None)
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "invalid")
    args = bench.parse_args(["--accept", "--m", "9,64", "--l2", l2, "--pairs", str(pairs)])
    rows = bench.run_accept_shape(
        args,
        7,
        8,
        [("test", {"name": "fc", "count_per_step": 1})],
        object() if l2 == "hot" else None,
        "test",
        kind,
    )
    assert sizes == [(64, 8), (7, 8)]
    assert [entry[0] for entry in captures] == ["dispatched", "cublas", "cublas"] * 2
    assert comparisons == [captures[i][1] for i in (1, 0, 4, 3)]
    weight, rotations = captures[0][2:]
    copies = 256  # cold: >= 1 GiB capped at 256; hot: >= 512 MiB capped at 256
    assert len(rotations) == copies - 1
    # Cold rotates distinct copies; hot repeats the same weight tensor.
    assert len({w.data_ptr() for w in (weight, *rotations)}) == (copies if l2 == "cold" else 1)
    for w in rotations:
        assert torch.equal(w, weight)
    assert all(entry[2] is weight and entry[3] is rotations for entry in captures)
    assert [row["M"] for row in rows] == [9, 64]
    for row in rows:
        assert row["timing_method"] == ("rotating_weights" if l2 == "cold" else "repeated_weight")
        assert row["rotation_copies"] == copies
        assert row["l2"] == l2
        assert row["pairs"] == row["self_pairs"] == pairs
        assert row["ci_low"] == row["ci_high"] == row["paired_speedup"] == 2.0
        assert row["ci_method"] == "bootstrap_median_percentile95_2000"
        assert row["correct"] and row["verdict"] == "PASS"
        sol = 2 * (7 * 8 + row["M"] * 8 + row["M"] * 7) / (rate * 1000)
        assert row["sol_us"] == pytest.approx(sol)
        assert row["sol_fraction"] == pytest.approx(sol / 5)
        assert row["cublas_sol_fraction"] == pytest.approx(sol / 10)
        assert row["fastest_market"] == "F.linear"
        assert "SM10x" in row["fastest_market_provenance"]


@pytest.mark.parametrize("out_dtype", ["bf16", "fp32"])
def test_acceptance_records_market_bf16_precision_and_restores_flag(monkeypatch, out_dtype):
    torch = bench.torch
    matmul = torch.backends.cuda.matmul
    assert bench.MARKET_BF16_REDUCED_PRECISION is True
    for previous in (False, True):
        monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", previous)
        with bench.bf16_reduced_precision(not previous):
            assert matmul.allow_bf16_reduced_precision_reduction is not previous
        assert matmul.allow_bf16_reduced_precision_reduction is previous
        with pytest.raises(RuntimeError, match="restore precision"):
            with bench.bf16_reduced_precision(not previous):
                raise RuntimeError("restore precision")
        assert matmul.allow_bf16_reduced_precision_reduction is previous

    # Run acceptance with CPU tensors and simulated graph capture/timing.
    for name in ("randn", "Generator"):
        original = getattr(torch, name)

        def cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu)
    calls = []
    mm = torch.mm
    if out_dtype == "fp32":

        def market(a, b, *, out_dtype):
            assert a.dtype == b.dtype == torch.bfloat16 and out_dtype == torch.float32
            calls.append(matmul.allow_bf16_reduced_precision_reduction)
            return mm(a.float(), b.float())

        monkeypatch.setattr(torch, "mm", market)
        monkeypatch.setattr(cublas, "_out_dtype_mm", lambda x: True)
    captured_flags = []

    def instance(name, fn, x, weight, rotations):
        # The dispatched op's cuBLAS fallback is timed with the market flag too.
        captured_flags.append((name, matmul.allow_bf16_reduced_precision_reduction))
        return SimpleNamespace(output=fn(x, weight))

    monkeypatch.setattr(bench, "Instance", instance)
    monkeypatch.setattr(bench, "hot_copies", lambda *args: 1)
    monkeypatch.setattr(bench, "paired", lambda *args: ([10.0] * 20, [10.0] * 20))
    monkeypatch.setattr(bench, "kernels_per_replay", lambda _: 1)
    emitted = []
    monkeypatch.setattr(bench, "emit", emitted.append)
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "invalid")
    monkeypatch.setattr(matmul, "allow_bf16_reduced_precision_reduction", False)
    rows = bench.run_accept_shape(
        bench.parse_args(["--accept", "--m", "1", "--l2", "hot", "--out-dtype", out_dtype]),
        7,
        8,
        [("test", {"name": "fc", "count_per_step": 1})],
        None,
        "CPU",
        "rtx5090",
    )
    assert len(rows) == 1 and emitted == rows
    assert rows[0]["out_dtype"] == out_dtype
    if out_dtype == "fp32":
        assert calls == [False, True, True, True, True]
        assert rows[0]["fastest_market"] == "torch.mm(out_dtype=torch.float32)"
    assert rows[0]["cublas_bf16_reduced_precision"] is True
    assert captured_flags == [("dispatched", True), ("cublas", True), ("cublas", True)]
    assert matmul.allow_bf16_reduced_precision_reduction is False


def test_acceptance_engine_alternates_cold_and_hot_weights(monkeypatch):
    torch = bench.torch
    for name in ("randn", "Generator"):
        original = getattr(torch, name)

        def cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu)

    captured = []

    def instance(name, fn, x, weight, rotations):
        captured.append((name, weight, rotations))
        return SimpleNamespace(output=fn(x, weight))

    flush = object()

    def paired(left, right, pairs, reps, actual_flush):
        assert actual_flush is flush
        return [10.0] * pairs, [10.0] * pairs

    monkeypatch.setattr(bench, "Instance", instance)
    monkeypatch.setattr(bench, "acceptance_copies", lambda *args: 3)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(L2_cache_size=1)
    )
    monkeypatch.setattr(bench, "paired", paired)
    monkeypatch.setattr(bench, "kernels_per_replay", lambda _: 1)
    emitted = []
    monkeypatch.setattr(bench, "emit", emitted.append)
    rows = bench.run_accept_shape(
        bench.parse_args(["--accept", "--m", "1", "--l2", "engine"]),
        7,
        8,
        [("test", {"name": "fc", "count_per_step": 1})],
        flush,
        "CPU",
        "rtx5090",
    )
    assert len(rows) == 1 and emitted == rows
    assert [name for name, _, _ in captured] == ["dispatched", "cublas", "cublas"]
    for _, weight, rotations in captured:
        assert weight.device.type == "cpu"
        assert len(rotations) == 4
        assert all(rotation is weight for rotation in rotations[1::2])
        clones = rotations[::2]
        assert len({id(clone) for clone in clones}) == len(clones)
        assert all(clone is not weight and torch.equal(clone, weight) for clone in clones)
        assert rows[0]["rotation_copies"] == len(rotations) + 1
    assert rows[0]["timing_method"] == "alternating_cold_hot_weights"


def test_acceptance_selects_accuracy_baseline_by_provider(monkeypatch):
    from ctikernels.linear import inference

    torch = bench.torch
    matmul = torch.backends.cuda.matmul
    for name in ("randn", "Generator"):
        original = getattr(torch, name)

        def cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu)

    linear_outputs = []

    def linear(x, weight):
        output = x.new_full(
            (x.shape[0], weight.shape[0]), matmul.allow_bf16_reduced_precision_reduction
        )
        linear_outputs.append(output)
        return output

    class Operation:
        launch_config = None
        kernels_per_call = 1

        def __init__(self, *, provider):
            assert provider == "auto"
            self.provider = selected_provider

        def __call__(self, x, weight):
            return linear(x, weight)

    baselines = []

    def accuracy(output, baseline, ref, condition, k):
        baselines.append(baseline)
        return {"correct": True}

    monkeypatch.setattr(inference, "Linear", Operation)
    monkeypatch.setattr(bench.F, "linear", linear)
    monkeypatch.setattr(bench, "accuracy", accuracy)
    monkeypatch.setattr(
        bench, "Instance", lambda name, fn, x, w, rotations: SimpleNamespace(output=fn(x, w))
    )
    monkeypatch.setattr(bench, "hot_copies", lambda *args: 1)
    monkeypatch.setattr(bench, "paired", lambda *args: ([10.0] * 20, [10.0] * 20))
    monkeypatch.setattr(bench, "kernels_per_replay", lambda _: 1)
    monkeypatch.setattr(bench, "emit", lambda row: None)

    for selected_provider, expected_flag in (("cute", False), ("cublas", True)):
        linear_outputs.clear()
        baselines.clear()
        rows = bench.run_accept_shape(
            bench.parse_args(["--accept", "--m", "1", "--l2", "hot"]),
            7,
            8,
            [("test", {"name": "fc", "count_per_step": 1})],
            None,
            "CPU",
            "rtx5090",
        )
        assert rows[0]["provider"] == selected_provider
        assert len(baselines) == 1
        assert baselines[0] is linear_outputs[int(expected_flag)]
        assert baselines[0].tolist() == [[float(expected_flag)] * 7]


def test_bxq_sweep_schema_correctness_and_cli(tmp_path):
    import torch

    from benchmark.kernel_tuning import linear_smallm_bxq_sweep as sweep

    args = sweep.parse_args(["--shapes", "67:264,67:264,129:520"])
    assert args.shapes == [(67, 264), (129, 520)] and args.m == list(bench.BXQ_MS)
    assert args.pairs == 5 and args.budget == 1600
    for argv in (
        ["--shapes", "2:3"],
        ["--m", "65"],
        ["--pairs", "0"],
        ["--budget", "nan"],
        ["--shapes", "1:8:16"],
        ["--configs", "64,128"],
        ["--configs", "64,128,2,1,4,tma,4"],
    ):
        with pytest.raises(SystemExit):
            sweep.parse_args(argv)
    assert sweep.parse_args(["--configs", "64,128,2,2;64,128,2,2;128,64,3,4,4"]).configs == [
        Config(64, 128, 2, 2),
        Config(128, 64, 3, 4, 4),
    ]
    assert sweep.parse_args([]).configs is None
    assert len(sweep.parse_args(["--model", "all"]).shapes) == len(
        bench.load_shapes(bench.parse_args(["--model", "all"]))
    )
    for n, k, l2 in [(1, 8, 100), (67, 264, 32 * 1024**2), (151936, 4096, 96 * 1024**2)]:
        assert sweep.rotation_count(n, k, l2) * 2 * n * k >= 3 * l2
    ref = torch.zeros(1, dtype=torch.float64)
    baseline = torch.tensor([0.5])
    metrics = sweep.error_metrics(torch.tensor([1.0]), baseline, ref)
    assert metrics["correct"]
    bad = sweep.error_metrics(torch.tensor([1.1]), baseline, ref)
    assert not bad["correct"]
    assert not sweep.error_metrics(torch.tensor([float("nan")]), baseline, ref)["correct"]
    cfg = Config(32, 128, 2, 3, 1, "reg", 4)
    row = sweep.probe_row(67, 264, 64, "gb10", cfg, metrics, [10, 20, 10], [12, 21, 12])
    assert row["paired_speedup"] == 1.2 and row["ours_us"] == 10
    source = tmp_path / "sweep.log"
    source.write_text("RESULT " + json.dumps(row))
    chosen = table.select_configs(table.read_logs([source]))
    assert chosen["gb10"][67, 264] == {(64, 64): cfg}
    rejected = sweep.probe_row(67, 264, 64, "gb10", cfg, bad)
    assert table.select_configs([row, rejected])["gb10"] == {}


def test_bxq_sweep_cooperative_budget_and_policy_restore(monkeypatch, capsys):
    from benchmark.kernel_tuning import linear_smallm_bxq_sweep as sweep

    ticks = iter([0, 2, 2])
    monkeypatch.setattr(sweep.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sweep.torch.cuda, "get_device_properties", lambda _: pytest.fail("expired"))
    previous = sweep.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    assert sweep.main(["--shapes", "67:264", "--budget", "1"]) == 1
    assert sweep.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == previous
    rows = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines()]
    assert [r["kind"] for r in rows] == ["budget", "probe_completion"]
    assert not rows[-1]["complete"]


@pytest.mark.parametrize("expire_during_timing", [False, True])
def test_bxq_sweep_cpu_simulation(monkeypatch, capsys, expire_during_timing):
    """Exercise the run loop with CPU operands, including bucket skip and deadline."""
    import torch

    from benchmark.kernel_tuning import linear_smallm_bxq_sweep as sweep

    now = 0
    monkeypatch.setattr(sweep.time, "monotonic", lambda: now)
    props = SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=170,
        L2_cache_size=100,
        shared_memory_per_block_optin=99 * 1024,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    for name in ("randn", "empty", "Generator"):
        original = getattr(torch, name)

        def cpu(*a, _original=original, **kw):
            kw.pop("device", None)
            return _original(*a, **kw)

        monkeypatch.setattr(torch, name, cpu)
    good = Config(32, 128, 2, 1, 1, "reg", 4)
    good_split = Config(32, 128, 2, 3, 1, "reg", 4)
    bad = Config(32, 128, 2, 2, 1, "reg", 4)
    overflow = Config(64, 256, 2)
    monkeypatch.setattr(sweep, "bxq_configs", lambda *a: [overflow, good, good_split, bad])
    launched = []

    class Operation:
        provider = "cute"

        def __init__(self, config, provider):
            assert config != overflow
            self.config = config
            launched.append(config)

        def __call__(self, x, w):
            return sweep.F.linear(x, w) + (100 if self.config == bad else 0)

    monkeypatch.setattr(sweep, "Linear", Operation)
    captures = []
    graphs = []

    def graph(fn, weights, x):
        outputs = [fn(x, w) for w in weights]
        captures.append((fn, len(weights), x.shape[0]))
        captured = SimpleNamespace(replay=lambda: None, fn=fn)
        graphs.append(captured)
        return captured, outputs

    monkeypatch.setattr(sweep, "graph_of", graph)
    timed = []

    def timing(graph):
        nonlocal now
        assert graph is graphs[-1 if graph.fn is sweep.F.linear else -2]
        timed.append(graph.fn)
        if expire_during_timing:
            now = 2
        return 12 if graph.fn is sweep.F.linear else 10

    monkeypatch.setattr(sweep, "time_graph", timing)
    assert sweep.main(["--shapes", "7:8", "--m", "64", "--pairs", "2", "--budget", "1"]) == int(
        expire_during_timing
    )
    rows = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines()]
    expected_configs = [good] if expire_during_timing else [good, good_split]
    assert len(captures) == 2 * len(expected_configs)
    for i, cfg in enumerate(expected_configs):
        candidate, copies, m = captures[2 * i]
        assert isinstance(candidate, Operation) and candidate.config == cfg
        assert (copies, m) == (3, 64)
        assert captures[2 * i + 1] == (sweep.F.linear, 3, 64)
    if expire_during_timing:
        assert launched == [good]
        assert rows[-2]["kind"] == "budget" and rows[-2]["phase"] == "timing"
        assert len(timed) == 2  # finish the in-flight pair, then stop cooperatively
    else:
        assert launched == [good, good_split, bad]
        assert len(timed) == 8
        for i in (0, 4):
            assert timed[i] is timed[i + 3] is sweep.F.linear
            assert timed[i + 1] is timed[i + 2] is captures[i // 2][0]
        probes = [r for r in rows if r["kind"] == "smallm_probe"]
        assert [r["correct"] for r in probes] == [True, True, False]
        for row in probes[:2]:
            assert row["ours_us"] == pytest.approx(10 / 3)
            assert row["cublas_us"] == 4
        assert table.select_configs(probes)["rtx5090"][7, 8] == {(64, 64): good}
    assert rows[-1]["complete"] is not expire_during_timing


def test_merge_candidates_keeps_base_and_prefers_new_winners():
    old, new = Config(), Config(splits=2)
    base = {"rtx5090": {(24576, 4096): {(1, 3): old}}, "pro6000": {}, "gb10": {}}
    selected = table.select_configs([probe(m=m, ratio=1.2, cfg=new) for m in (3, 4)])
    merged = table.merge_candidates(base, selected)
    assert merged["rtx5090"][24576, 4096] == {(1, 2): old, (3, 4): new}


def test_candidate_cli_selects_without_acceptance(tmp_path, monkeypatch, capsys):
    from ctikernels.linear import _dispatch as dispatch

    monkeypatch.setattr(dispatch, "TUNED", {"rtx5090": {}, "pro6000": {}, "gb10": {}})
    source = tmp_path / "probe.log"
    source.write_text("RESULT " + json.dumps(probe(m=9, ratio=1.2)))
    assert table.main(["--candidate", "--probe", str(source)]) == 0
    assert "(9, 9): Config(" in capsys.readouterr().out
    empty = tmp_path / "empty.log"
    empty.write_text("")
    with pytest.raises(SystemExit) as exc:
        table.main(["--candidate", "--probe", str(empty)])
    assert exc.value.code == 2


def test_acceptance_gate_rejects_gain_inside_self_pair_noise():
    chosen = table.select_configs([probe(m=m, ratio=1.2) for m in (1, 2)])
    rows = [acceptance(m=1, ratio=1.1, epsilon=0.12), acceptance(m=2, ratio=1.1, epsilon=0.02)]
    gated = gate(chosen, rows)
    assert gated["rtx5090"] == {(24576, 4096): {(2, 2): Config()}}


def test_acceptance_for_one_configuration_never_qualifies_another(tmp_path, capsys):
    a, b = Config(), Config(splits=2)
    evidence = [acceptance(m=m, ratio=1.2, cfg=a) for m in (1, 2)]
    for cfg, expected in ((a, {(1, 2): a}), (b, None)):
        chosen = table.select_configs([probe(m=m, ratio=1.2, cfg=cfg) for m in (1, 2)])
        refused = []
        gated = gate(chosen, evidence, refused=refused)
        assert gated["rtx5090"].get((24576, 4096)) == expected
        assert refused == ([] if cfg == a else [("rtx5090", 24576, 4096, m, b) for m in (1, 2)])
    # Same N/K/M evidence through the CLI: probes select b, acceptance measured a.
    rows = [probe(m=m, ratio=1.3, cfg=b) for m in (1, 2)] + evidence
    rows.append({"kind": "accept_completion", "complete": True, "failures": 0})
    source = tmp_path / "current.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
    target = tmp_path / "dispatch.py"
    target.write_text(table.BEGIN + "\nTUNED = {}\n" + table.END + "\n")
    assert table.main(["--accept", str(source), "--rewrite", str(target)]) == 0
    assert "refused rtx5090 N=24576 K=4096 M=1" in capsys.readouterr().err
    namespace = {"Config": Config}
    exec(target.read_text(), namespace)
    assert namespace["TUNED"] == {"rtx5090": {}, "pro6000": {}, "gb10": {}}


@pytest.mark.parametrize(
    "launch_config",
    [
        None,
        {"config": asdict(Config()), "mtiles": 2, "cta_threads": 128},  # wrong M bucket
        {"config": asdict(Config()), "mtiles": 1, "cta_threads": 64},  # wrong CTA shape
        {"config": {}, "mtiles": 1, "cta_threads": 128},  # Config defaults never fill gaps
        {"config": {"rows": 64, "splits": 1}, "mtiles": 1, "cta_threads": 128},
        {"config": dict(asdict(Config()), extra=1), "mtiles": 1, "cta_threads": 128},
    ],
)
def test_acceptance_without_exact_launch_record_qualifies_nothing(launch_config):
    chosen = table.select_configs([probe(m=1, ratio=1.2)])
    rows = [acceptance(m=1, ratio=1.2, launch_config=launch_config)]
    assert gate(chosen, rows)["rtx5090"] == {}


def test_bxq_markdown_lists_every_row_with_fallbacks_and_rounds():
    a, b = Config(), Config(splits=2)
    common = {"cublas_us": 10.0, "sol_us": 8.0, "epsilon": 0.01, "model": "m", "count_per_step": 2}
    rows = [
        acceptance(m=1, ratio=1.25, cfg=a, ours_us=8.0, name="p", **common),
        # Measured b, but the table holds a: reported as F.linear at F.linear time.
        acceptance(m=4, ratio=1.25, cfg=b, ours_us=8.0, name="p", **common),
        acceptance(m=16, ratio=1.0, ours_us=10.0, name="p", provider="cublas", **common),
    ]
    rows[2]["launch_config"] = None
    tuned = {"rtx5090": {(24576, 4096): {(1, 4): a}}, "pro6000": {}, "gb10": {}}
    inventory = {"m": [{"name": "p", "N": 24576, "K": 4096, "count_per_step": 2}]}
    text = table.bxq_markdown(rows, tuned, inventory)
    assert "3 rows, 0 FAIL, 1 on CuTe" in text
    assert (
        "| 24576 | 4096 | 1 | m:p | CuTe | 8.00 | 10.00 | 1.250 | 0.0100 | 100.0 | PASS |" in text
    )
    assert "| 24576 | 4096 | 4 | m:p | F.linear | 10.00 | 10.00 | 1.000 |" in text
    assert "F.linear fallback (CuTe measured 1.250)" in text
    assert "| 24576 | 4096 | 16 | m:p | F.linear | 10.00 | 10.00 | 1.000 |" in text
    # B=1,Q=4 (M=4) and B=4,Q=4 (M=16) are the only complete rounds.
    assert "| m | 1 | 4 | 4 | 0.020 | 0.020 | 1.000 |" in text
    assert "| m | 4 | 4 | 16 | 0.020 | 0.020 | 1.000 |" in text
    assert "Pending: no current acceptance rows." in text


def _accept_log(tmp_path, *runs):
    rows = [row for run in runs for row in run]
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
    rows = table.read_logs([source])
    return {r["M"]: r["_complete"] for r in rows if r["kind"] == "smallm_accept"}


def _run(ms, run_id=None, *, finished=True, rows=None):
    stamp = {} if run_id is None else {"run_id": run_id}
    begin = [{"kind": "accept_begin", **stamp}] if run_id else []
    body = []
    for m in ms:
        row = acceptance(m=m, **stamp)
        del row["_complete"], row["_finished"]
        body.append(row)
    end = {"kind": "accept_completion", "complete": True, "failures": 0, **stamp}
    end["rows"] = len(ms) if rows is None else rows
    return begin + body + ([end] if finished else [])


@pytest.mark.parametrize("ids", [(None, None), ("a", "b")])
def test_completion_is_scoped_per_concatenated_run(tmp_path, ids):
    first, second = ids
    # A completed partition followed by a truncated one: only the first qualifies.
    done = _accept_log(tmp_path, _run((1, 2), first), _run((3, 4), second, finished=False))
    assert done == {1: True, 2: True, 3: False, 4: False}
    # A truncated partition followed by a completed one never borrows its marker.
    done = _accept_log(tmp_path, _run((1, 2), first, finished=False), _run((3, 4), second))
    assert done == (
        {1: False, 2: False, 3: False, 4: False}
        if first is None
        else {1: False, 2: False, 3: True, 4: True}
    )


def test_run_completion_requires_its_own_begin_and_row_count(tmp_path):
    begin, *body, end = _run((1, 2), "a")
    assert _accept_log(tmp_path, [*body, end]) == {1: False, 2: False}
    assert _accept_log(tmp_path, _run((1, 2), "a", rows=3)) == {1: False, 2: False}
    assert _accept_log(tmp_path, [begin, *body, dict(end, run_id="b")]) == {1: False, 2: False}


def test_bxq_markdown_omits_unfinished_runs_and_reports_finished_failures(tmp_path):
    tuned = {"rtx5090": {(24576, 4096): {(1, 2): Config()}}, "pro6000": {}, "gb10": {}}
    extra = {"cublas_us": 10.0, "ours_us": 8.0, "sol_us": 8.0, "epsilon": 0.01, "model": "m"}

    def report(*runs):
        logged = [
            dict(r, **extra) if r["kind"] == "smallm_accept" else r for run in runs for r in run
        ]
        source = tmp_path / "accept.log"
        source.write_text("\n".join("RESULT " + json.dumps(r) for r in logged))
        rows = table.read_logs([source], round_label="current round")
        return table.bxq_markdown(rows, tuned, {})

    for run_id in (None, "a"):
        text = report(_run((1,), run_id, finished=False))
        assert "## RTX 5090\n\nPending: no current acceptance rows." in text
        assert "PASS" not in text
        # A finished run with failures is reported, though it qualifies nothing.
        failed = _run((1,), run_id)
        failed[-1]["failures"] = 1
        text = report(failed)
        assert "1 rows, 0 FAIL, 1 on CuTe" in text
        assert (
            table.accepted_speedups(
                table.read_logs([tmp_path / "accept.log"], round_label="current round")
            )
            == {}
        )


def test_markdown_omits_unfinished_runs(tmp_path):
    extra = {"cublas_us": 10.0, "ours_us": 8.0, "sol_us": 8.0, "epsilon": 0.01}
    extra.update(model="qwen3.5-9b", name="mlp.gate_up")

    def report(*runs):
        logged = [
            dict(r, **extra) if r["kind"] == "smallm_accept" else r for run in runs for r in run
        ]
        source = tmp_path / "accept.log"
        source.write_text("\n".join("RESULT " + json.dumps(r) for r in logged))
        return table.markdown(table.read_logs([source], round_label="current round"))

    for run_id in (None, "a"):
        text = report(_run((1,), run_id, finished=False))
        assert "### RTX 5090\n\nPending: no finished acceptance RESULT rows supplied." in text
        assert "rows PASS" not in text and "| PASS |" not in text
    # A finished run after an unfinished one is reported alone.
    text = report(_run((2,), "b", finished=False), _run((1,), "c"))
    assert "current round; 1/1 rows PASS" in text


def test_model_totals_require_the_full_projection_inventory():
    shapes = bench.builtin_shapes()
    gate_up = next(s for s in shapes["qwen3.5-9b"] if s["name"] == "mlp.gate_up")
    rows = [
        dict(
            gate_up,
            kind="smallm_accept",
            device="RTX 5090",
            model="qwen3.5-9b",
            M=m,
            ours_us=1.0,
            cublas_us=1.0,
            verdict="PASS",
        )
        for m in bench.BXQ_MS
    ]
    assert bench.round_estimates(rows) == []
    assert bench.step_estimates(rows) == []
    # Unknown models have no inventory to check against: no totals.
    assert bench.round_estimates([dict(r, model="unknown") for r in rows]) == []
    # An explicit inventory holding only that projection does produce totals.
    inventory = {"qwen3.5-9b": [gate_up]}
    rounds = bench.round_estimates([dict(r, device="rtx5090") for r in rows], inventory)
    assert rounds and all(r["calls_per_round"] == 32 for r in rounds)


def _retry(m, run_id, **updates):
    run = _run((m,), run_id)
    run[-2].update(updates)
    if updates.get("verdict") == "FAIL":
        run[-1]["failures"] = 1
    return run


@pytest.mark.parametrize("ids", [(None, None), ("a", "b")])
@pytest.mark.parametrize("failure", [{"correct": False, "verdict": "FAIL"}, {"verdict": "FAIL"}])
def test_retries_resolve_per_key_in_run_order(tmp_path, ids, failure):
    chosen = table.select_configs([probe(m=1, ratio=1.2)])
    source = tmp_path / "accept.log"

    def gated(*runs):
        rows = [row for run in runs for row in run]
        source.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
        current = table.read_logs([source], round_label="current round")
        return gate(chosen, current)["rtx5090"]

    first, second = ids
    # A completed PASS followed by a completed failing retry: not qualified.
    assert gated(_retry(1, first), _retry(1, second, **failure)) == {}
    # A failing run followed by a completed PASS retry: qualified.
    assert gated(_retry(1, first, **failure), _retry(1, second)) == {
        (24576, 4096): {(1, 1): Config()}
    }


def test_correctness_failure_in_unfinished_run_invalidates_earlier_pass(tmp_path):
    chosen = table.select_configs([probe(m=1, ratio=1.2)])
    failing = _run((1,), "b", finished=False)
    failing[-1].update(correct=False, verdict="FAIL", launch_config=None)
    rows = [*_run((1,), "a"), *failing]
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
    current = table.read_logs([source], round_label="current round")
    assert gate(chosen, current)["rtx5090"] == {}


def test_qualification_report_and_round_totals_share_retry_resolution(tmp_path):
    common = {"cublas_us": 10.0, "sol_us": 8.0, "epsilon": 0.01, "model": "m"}
    # A correctness FAIL at 0.80x, then a complete PASS retry at 1.25x (M=4: B=1, Q=4).
    failing = _retry(4, "a", correct=False, verdict="FAIL", paired_speedup=0.8, ours_us=12.5)
    passing = _retry(4, "b", paired_speedup=1.25, ours_us=8.0)
    logged = [
        {**common, **r, "name": "p"} if r["kind"] == "smallm_accept" else r
        for r in (*failing, *passing)
    ]
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in logged))
    rows = table.read_logs([source], round_label="current round")
    tuned = gate(
        table.select_configs([probe(m=4, ratio=1.2)]),
        [*rows, acceptance(m=3, ratio=1.25)],
    )
    assert tuned["rtx5090"] == {(24576, 4096): {(3, 4): Config()}}
    inventory = {"m": [{"name": "p", "N": 24576, "K": 4096, "count_per_step": 1000}]}
    text = table.bxq_markdown(rows, tuned, inventory)
    assert "1 rows, 0 FAIL, 1 on CuTe" in text
    assert "| 24576 | 4096 | 4 | m:p | CuTe | 8.00 | 10.00 | 1.250 |" in text
    assert "| m | 1 | 4 | 4 | 8.000 | 10.000 | 1.250 |" in text
    assert "12.50" not in text and "0.800" not in text


def test_rows_and_round_totals_use_the_same_cold_l2_regime(tmp_path):
    common = {"cublas_us": 10.0, "sol_us": 8.0, "epsilon": 0.01, "model": "m"}
    cold = _retry(4, "a", paired_speedup=1.25, ours_us=8.0)
    # A later finished hot-L2 re-pair of the same row: faster, but not the engine's regime.
    hot = _retry(4, "b", paired_speedup=1.527, ours_us=2.375, cublas_us=3.625)
    for r in hot:
        r["l2"] = "hot"
    logged = [
        {**common, **r, "name": "p"} if r["kind"] == "smallm_accept" else r for r in (*cold, *hot)
    ]
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in logged))
    rows = table.read_logs([source], round_label="current round")
    tuned = gate(
        table.select_configs([probe(m=4, ratio=1.2)]),
        [*rows, acceptance(m=3, ratio=1.25)],
    )
    assert tuned["rtx5090"] == {(24576, 4096): {(3, 4): Config()}}
    inventory = {"m": [{"name": "p", "N": 24576, "K": 4096, "count_per_step": 1000}]}
    text = table.bxq_markdown(rows, tuned, inventory)
    assert "1 rows, 0 FAIL, 1 on CuTe" in text
    assert "| 24576 | 4096 | 4 | m:p | CuTe | 8.00 | 10.00 | 1.250 |" in text
    assert "| m | 1 | 4 | 4 | 8.000 | 10.000 | 1.250 |" in text
    assert "2.375" not in text and "1.527" not in text
    rounds = bench.round_estimates([dict(r, device="rtx5090") for r in rows], inventory)
    assert {(r["ours_round_ms"], r["cublas_round_ms"]) for r in rounds} == {(8.0, 10.0)}
    report = table.markdown(rows)
    assert "current round; 1/1 rows PASS" in report
    assert "2.375" not in report and "1.527" not in report


def _gated_log(tmp_path, *runs, m=1):
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for run in runs for r in run))
    rows = table.read_logs([source], round_label="current round")
    chosen = table.select_configs([probe(m=m, ratio=1.2)])
    return gate(chosen, rows)["rtx5090"]


def _alias(run, name, **updates):
    for row in run:
        if row["kind"] == "smallm_accept":
            row.update(name=name, **updates)
    return run


@pytest.mark.parametrize("launch_config", ["same", None])
def test_correctness_failure_of_one_alias_invalidates_the_others(tmp_path, launch_config):
    failure = {"correct": False, "verdict": "FAIL"}
    if launch_config is None:
        failure["launch_config"] = None
    other = launch(1, Config(splits=2))
    runs = (
        _alias(_retry(1, "a"), "x"),
        _alias(_retry(1, "b", **failure), "y"),
        # y passes again, but with another configuration: x's stale PASS stays invalid.
        _alias(_retry(1, "c", launch_config=other), "y"),
    )
    assert _gated_log(tmp_path, *runs) == {}
    # Both aliases must now measure the same config for the whole bucket.
    assert _gated_log(tmp_path, *runs, _alias(_retry(1, "d"), "x")) == {}
    assert _gated_log(
        tmp_path, *runs, _alias(_retry(1, "d"), "x"), _alias(_retry(1, "e"), "y")
    ) == {(24576, 4096): {(1, 1): Config()}}


def test_unfinished_alias_never_vetoes_finished_evidence(tmp_path):
    unfinished = _alias(_run((1,), "b", finished=False), "y")
    assert _gated_log(tmp_path, _alias(_retry(1, "a"), "x"), unfinished) == {
        (24576, 4096): {(1, 1): Config()}
    }


@pytest.mark.parametrize(
    "updates",
    [
        None,
        {"ratio": 1.029},
        {"cfg": Config(splits=3)},
        {"provider": "cublas"},
        {"correct": False},
        {"launch_config": None},
    ],
)
def test_whole_bucket_acceptance_requires_every_measured_m(updates):
    chosen = {"rtx5090": {(24576, 4096): {(5, 16): Config()}}}
    rows = [
        acceptance(m=5, ratio=1.2),
        acceptance(m=7, **(updates or {})),
        acceptance(m=8, ratio=1.2),
    ]
    refused = []
    gated = gate(chosen, rows, refused=refused)
    assert gated["rtx5090"] == ({(24576, 4096): {(5, 8): Config()}} if updates is None else {})
    assert bool(refused) == (updates == {"cfg": Config(splits=3)})
    assert gate(chosen, [])["rtx5090"] == {}


@pytest.mark.parametrize(
    "hot_ratio,kept", [(1.02, False), (1.03, False), (1.031, True), (1.2, True)]
)
@pytest.mark.parametrize("separate_logs", [False, True])
def test_hot_and_cold_acceptance_both_must_pass(hot_ratio, kept, separate_logs):
    chosen = {"rtx5090": {(24576, 4096): {(5, 8): Config()}}}
    cold = acceptance(m=5, ratio=1.2, _source=0)
    hot = acceptance(m=5, ratio=hot_ratio, l2="hot", _source=int(separate_logs))
    rows = [cold, hot]
    assert len(table.resolved_acceptance(rows)["rtx5090"]) == 2
    edge = acceptance(m=8, ratio=1.2, _source=0)
    rows.append(edge)
    assert bool(gate(chosen, rows)["rtx5090"]) is kept
    assert gate(chosen, [hot])["rtx5090"] == {}
    assert gate(chosen, [hot, edge])["rtx5090"] == {}
    # Explicit cold retries replace legacy rows without an l2 field.
    retry = acceptance(m=5, ratio=1.01, l2="cold", _source=int(separate_logs))
    assert gate(chosen, [*rows, retry])["rtx5090"] == {}


def test_engine_acceptance_only_vetoes_cold_qualified_buckets():
    chosen = {"rtx5090": {(24576, 4096): {(5, 8): Config()}}}
    cold = [acceptance(m=m, ratio=1.2, l2="cold") for m in (5, 8)]
    expected = chosen["rtx5090"]
    assert gate(chosen, cold)["rtx5090"] == expected

    engine = acceptance(m=5, ratio=1.2, l2="engine", _complete=False, _finished=True)
    failed = dict(engine, verdict="FAIL")
    assert gate(chosen, [*cold, failed])["rtx5090"] == {}
    # A PASS in a finished engine run still permits its key if another row failed;
    # both bucket edges need one.
    assert gate(chosen, [*cold, engine])["rtx5090"] == {}
    assert gate(chosen, [*cold, engine, dict(engine, M=8)])["rtx5090"] == expected
    engine_only = [acceptance(m=m, ratio=1.2, l2="engine") for m in (5, 8)]
    assert gate(chosen, engine_only)["rtx5090"] == {}


def test_l2_correctness_retry_restores_only_its_regime():
    cold = acceptance(ratio=1.2)
    bad = acceptance(l2="hot", correct=False, verdict="FAIL")
    good = acceptance(l2="hot", ratio=1.1)
    assert table.accepted_speedups([cold, bad]) == {}
    assert list(table.accepted_speedups([cold, bad, good]).values()) == [1.1]


def test_bucketize_majority_tie_and_uncovered_buckets():
    a, b = Config(), Config(splits=3)
    raw = {"rtx5090": {(24576, 4096): {(5, 7): a, (8, 8): b, (9, 9): a, (16, 16): b, (57, 57): a}}}
    assert table.bucketize(raw) == {
        "rtx5090": {(24576, 4096): {(5, 8): a, (9, 16): b, (57, 64): a}},
        "pro6000": {},
        "gb10": {},
    }
    assert table.bucketize(table.bucketize(raw)) == table.bucketize(raw)


def test_candidate_cli_bucketizes_merged_table(tmp_path, monkeypatch, capsys):
    from ctikernels.linear import _dispatch as dispatch

    base = {"rtx5090": {(24576, 4096): {(5, 7): Config()}}}
    monkeypatch.setattr(dispatch, "TUNED", base)
    source = tmp_path / "probe.log"
    source.write_text("RESULT " + json.dumps(probe(m=8, cfg=Config(splits=3))))
    assert table.main(["--candidate", "--bucketize", "--probe", str(source)]) == 0
    namespace = {"Config": Config}
    exec("TUNED = " + capsys.readouterr().out, namespace)
    assert namespace["TUNED"]["rtx5090"] == {(24576, 4096): {(5, 8): Config()}}
    with pytest.raises(SystemExit):
        table.main(["--bucketize"])


def test_acceptance_l2_parsing():
    assert bench.parse_args(["--accept"]).l2 == "cold"
    args = bench.parse_args(["--accept", "--l2", "hot"])
    assert args.l2 == "hot" and args.pairs == 20 and args.reps_per_sample == 1
    with pytest.raises(SystemExit):
        bench.parse_args(["--l2", "warm"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--accept", "--l2", "hot", "--pairs", "8"])


def test_acceptance_l2_engine_parsing():
    args = bench.parse_args(["--accept", "--l2", "engine"])
    assert args.l2 == "engine" and args.accept
    assert args.pairs == 20 and args.reps_per_sample == 1


def test_finished_hot_run_with_failures_only_vetoes_its_failing_rows(tmp_path):
    chosen = table.select_configs([probe(m=1, ratio=1.2), probe(m=2, ratio=1.2)])
    cold = _run((1, 2), "c")
    hot = _run((1, 2), "h")
    for row in hot:
        row["l2"] = "hot"
    hot[2].update(verdict="FAIL", paired_speedup=0.8)  # M=2 hot loses
    hot[-1]["failures"] = 1
    source = tmp_path / "accept.log"
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in (*cold, *hot)))
    rows = table.read_logs([source], round_label="current round")
    gated = gate(chosen, rows)["rtx5090"]
    # M=1 keeps its complete cold PASS; M=2 is vetoed by its hot FAIL.
    assert gated == {(24576, 4096): {(1, 1): Config()}}
    # Hot evidence alone never qualifies.
    source.write_text("\n".join("RESULT " + json.dumps(r) for r in hot))
    rows = table.read_logs([source], round_label="current round")
    assert gate(chosen, rows)["rtx5090"] == {}


def test_documented_recipe_needs_engine_logs_at_every_bucket_edge(tmp_path, monkeypatch):
    from ctikernels.linear import _dispatch

    cfg = Config(splits=2)
    tuned = {"rtx5090": {(24576, 4096): {(17, 24): cfg}}, "pro6000": {}, "gb10": {}}
    monkeypatch.setattr(_dispatch, "TUNED", tuned)  # --rewrite re-gates the committed table
    accept_ms = bench.parse_ms("accept")

    def log(name, l2, ms):
        rows = [acceptance(m=m, ratio=1.2, cfg=cfg, l2=l2, run_id=l2) for m in ms if 17 <= m <= 24]
        for row in rows:
            del row["_complete"], row["_finished"], row["_round"]
        end = {"kind": "accept_completion", "complete": True, "failures": 0, "run_id": l2}
        lines = [{"kind": "accept_begin", "run_id": l2}, *rows, dict(end, rows=len(rows))]
        path = tmp_path / name
        path.write_text("\n".join("RESULT " + json.dumps(r) for r in lines))
        return str(path)

    def rewrite(*logs):
        target = tmp_path / "dispatch.py"
        target.write_text(table.BEGIN + "\nTUNED = {}\n" + table.END + "\n")
        assert table.main(["--accept", *logs, "--rewrite", str(target)]) == 0
        namespace = {"Config": Config}
        exec(target.read_text(), namespace)
        return namespace["TUNED"]["rtx5090"]

    cold, engine = log("cold.log", "cold", accept_ms), log("engine.log", "engine", accept_ms)
    assert rewrite(cold) == {}
    # --m bxq lacks the lower edge M=17.
    bxq = [log(f"{l2}_bxq.log", l2, bench.BXQ_MS) for l2 in ("cold", "engine")]
    assert rewrite(*bxq) == {}
    assert rewrite(cold, engine) == {(24576, 4096): {(17, 24): cfg}}


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_output_dtype_evidence_isolation(dtype):
    other = "fp32" if dtype == "bf16" else "bf16"
    rows = [probe(out_dtype=dtype, ratio=1.2), dict(probe(out_dtype=other), correct=False)]
    chosen = table.select_configs(rows, out_dtype=dtype)
    assert chosen["rtx5090"][24576, 4096] == {(1, 1): Config()}
    assert table.select_configs(rows, out_dtype=other)["rtx5090"] == {}
    assert gate(chosen, [acceptance(out_dtype=other)], out_dtype=dtype)["rtx5090"] == {}
    evidence = [acceptance(out_dtype=dtype), acceptance(out_dtype=other, correct=False)]
    assert gate(chosen, evidence, out_dtype=dtype)["rtx5090"] == chosen["rtx5090"]
    assert table.select_configs([probe()], out_dtype="fp32")["rtx5090"] == {}
    assert gate(chosen, [acceptance()], out_dtype="fp32")["rtx5090"] == {}
    assert gate(chosen, [acceptance()], out_dtype="bf16")["rtx5090"] == chosen["rtx5090"]


@pytest.mark.parametrize("only_measured", [False, True])
def test_mixed_output_dtype_rewrite(monkeypatch, tmp_path, only_measured):
    from ctikernels.linear import _dispatch

    bf16, fp32 = Config(), Config(splits=2)
    target = tmp_path / "dispatch.py"
    target.write_text(
        table.BEGIN + '\nTUNED = {"rtx5090": {}}\n' + table.END + "\n"
        '# BEGIN GENERATED SMALLM F32 TABLE\nTUNED_F32 = {"rtx5090": {}}\n'
        "# END GENERATED SMALLM F32 TABLE\n"
    )
    rows = []
    for dtype, cfg in (("bf16", bf16), ("fp32", fp32)):
        for regime in ("cold", "engine"):
            rows += [acceptance(out_dtype=dtype, cfg=cfg, l2=regime)]
            rows += [{"kind": "accept_completion", "out_dtype": dtype, "complete": True}]
    log = tmp_path / "mixed.log"
    log.write_text("\n".join("RESULT " + json.dumps(r) for r in rows))
    for name, cfg in (("TUNED", bf16), ("TUNED_F32", fp32)):
        monkeypatch.setattr(_dispatch, name, {"rtx5090": {(24576, 4096): {(1, 1): cfg}}})
    args = ["--accept", str(log), "--rewrite", str(target)]
    assert table.main(args + (["--only-measured"] if only_measured else [])) == 0
    scope = {"Config": Config}
    exec(target.read_text(), scope)
    assert scope["TUNED"]["rtx5090"][24576, 4096] == {(1, 1): bf16}
    assert scope["TUNED_F32"]["rtx5090"][24576, 4096] == {(1, 1): fp32}


def test_fp32_tool_selection_and_legacy_records(tmp_path, capsys):
    from benchmark.kernel_tuning import linear_smallm_bxq_sweep as sweep

    shapes = "256:4096,512:4096,1024:4096,2048:4096"
    args = sweep.parse_args(["--out-dtype", "fp32", "--shapes", shapes])
    assert args.out_dtype == "fp32" and len(args.shapes) == 4
    inventory = tmp_path / "shapes.json"
    inventory.write_text(
        json.dumps(
            {
                "models": {
                    "model": [
                        {"N": n, "K": k, "name": "projection", "count_per_step": 1}
                        for n, k in args.shapes
                    ]
                }
            }
        )
    )
    args = bench.parse_args(["--accept", "--out-dtype", "fp32", "--shapes", str(inventory)])
    assert len(bench.load_shapes(args)) == 4
    bench.emit({"kind": "smallm_probe"})
    assert json.loads(capsys.readouterr().out[7:])["out_dtype"] == "bf16"


@pytest.mark.parametrize("run_id", [None, "shared"])
def test_completion_markers_cannot_cross_output_dtypes(run_id):
    rows = [
        {"kind": "accept_begin", "out_dtype": "fp32", "run_id": run_id},
        acceptance(out_dtype="fp32", run_id=run_id),
        {"kind": "accept_completion", "complete": True, "run_id": run_id},
    ]
    table.mark_completion(rows)
    assert not rows[1]["_complete"]
    rows.append(
        {"kind": "accept_completion", "out_dtype": "fp32", "complete": True, "run_id": run_id}
    )
    table.mark_completion(rows)
    assert rows[1]["_complete"]


def test_fp32_markdown_uses_market_mm():
    report = table.fp32_markdown([acceptance(out_dtype="fp32", ours_us=2.0)])
    assert "Linear FP32 output" in report
    assert "torch.mm(x, weight.t(), out_dtype=torch.float32)" in report
    assert "| fp32 | CuTe |" in report
    assert "F.linear" not in report


def test_fp32_sweep_forces_dtype_and_times_market_flags(monkeypatch, capsys):
    import torch

    from benchmark.kernel_tuning import linear_smallm_bxq_sweep as sweep

    for name in ("randn", "empty", "Generator"):
        original = getattr(torch, name)

        def cpu(*args, _original=original, **kwargs):
            kwargs.pop("device", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12,
            minor=0,
            multi_processor_count=170,
            L2_cache_size=64,
            shared_memory_per_block_optin=99 * 1024,
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    mm = torch.mm
    flags = []

    def market(x, wt, *, out_dtype):
        assert x.dtype == wt.dtype == torch.bfloat16 and out_dtype == torch.float32
        flags.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        return mm(x.float(), wt.float())

    monkeypatch.setattr(torch, "mm", market)

    class Operation:
        provider = "cute"

        def __init__(self, *, config, provider):
            assert provider == "cute" and config.splits == 1

        def __call__(self, x, w, *, out_dtype):
            assert out_dtype == torch.float32
            return mm(x.float(), w.float().t())

    monkeypatch.setattr(sweep, "Linear", Operation)
    monkeypatch.setattr(
        sweep,
        "graph_of",
        lambda fn, ws, x: (SimpleNamespace(replay=lambda: None), [fn(x, w) for w in ws]),
    )
    monkeypatch.setattr(sweep, "time_graph", lambda _: 10.0)
    assert (
        sweep.main(
            [
                "--out-dtype",
                "fp32",
                "--shapes",
                "7:8",
                "--m",
                "3",
                "--pairs",
                "1",
                "--configs",
                "16,64,2,1",
            ]
        )
        == 0
    )
    rows = [json.loads(line[7:]) for line in capsys.readouterr().out.splitlines()]
    assert all(row["out_dtype"] == "fp32" for row in rows)
    probe_row = next(row for row in rows if row["kind"] == "smallm_probe")
    assert probe_row["correct"] and probe_row["max_normalized_limit"] > 0
    assert probe_row["fastest_market"] == "torch.mm(out_dtype=torch.float32)"
    assert flags[0] is False and all(flags[1:])
