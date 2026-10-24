# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contracts for paired decode qualification; no CUDA initialization."""

import math
import random
import sys
from pathlib import Path

import pytest

from benchmark.kernel_tuning import gdn_decode_compare as compare

from ...support.child import run_child


@pytest.mark.parametrize("count,index", [(20, 18), (60, 56)])
def test_epsilon_uses_nearest_rank(count, index):
    pairs = [[1024 + i, 1024] for i in range(count)]
    random.Random(7).shuffle(pairs)
    assert compare.noise_epsilon(pairs) == index / 1024
    # Include ratios below one: absolute deviation, not signed deviation.
    assert compare.noise_epsilon([[1024 - i, 1024] for i in range(count)]) == index / 1024


@pytest.mark.parametrize(
    "ratio,epsilon,expected",
    [
        (1.125, 0.125, "tie"),
        (0.875, 0.125, "tie"),
        (math.nextafter(1.125, math.inf), 0.125, "win"),
        (math.nextafter(0.875, 0), 0.125, "loss"),
        (1.0, 0.0, "tie"),
        (1.01, 0.0, "win"),
        (0.99, 0.0, "loss"),
    ],
)
def test_verdict_thresholds(ratio, epsilon, expected):
    assert compare.verdict(ratio, epsilon) == expected


def test_bootstrap_is_deterministic_and_does_not_modify_global_rng():
    values = [0.7 + i * 0.04 for i in range(20)]
    state = random.getstate()
    first = compare.bootstrap_median_ci(values)
    assert first == compare.bootstrap_median_ci(values)
    assert random.getstate() == state
    assert first[0] <= 1.08 <= first[1]
    assert compare.bootstrap_median_ci([2.0] * 20) == [2.0, 2.0]


def test_pairs_use_comparator_over_ours_and_median_of_ratios():
    pairs = [[2.0, 1.0], [100.0, 100.0], [12.0, 4.0]]
    result = compare.pair_stats(pairs, [[1.0, 1.0]] * 3)
    assert result["ratios"] == [2, 1, 3]
    assert result["median_ratio"] == 2
    assert result["comparator_us"] == 12
    assert result["ours_us"] == 4
    assert result["verdict"] == "win"
    assert result["pairs_us"] == pairs


def test_vs_faster_picks_smallest_paired_ratio():
    self_pairs = [[1.0, 1.0]] * 20
    comparisons = {
        "sglang": compare.pair_stats([[2.0, 1.0]] * 20, self_pairs),
        "vllm": compare.pair_stats([[0.9, 1.0]] * 20, self_pairs),
    }
    fastest = compare.vs_faster(comparisons)
    assert fastest["comparator"] == "vllm"
    assert fastest["median_ratio"] == 0.9
    assert fastest["verdict"] == "loss"
    assert compare.vs_faster({}) is None


def test_vs_faster_excludes_product_but_preserves_its_comparison():
    self_pairs = [[1.0, 1.0]] * 20
    comparisons = {
        name: compare.pair_stats([[ratio, 1.0]] * 20, self_pairs)
        for name, ratio in (("product", 0.5), ("sglang", 1.2), ("vllm", 0.9))
    }
    assert compare.vs_faster(comparisons)["comparator"] == "vllm"
    assert comparisons["product"]["median_ratio"] == 0.5
    assert compare.vs_faster({"product": comparisons["product"]}) is None


@pytest.mark.parametrize("bad", [[], [[0, 1]], [[-1, 2]], [[math.nan, 1]], [[1, math.inf]], [[1]]])
def test_reject_invalid_timing_pairs(bad):
    with pytest.raises(ValueError):
        compare.noise_epsilon(bad)


def test_reject_incomplete_pairs():
    with pytest.raises(ValueError, match="incomplete"):
        compare.pair_stats([[1, 1]], [[1, 1]] * 2)


def test_argument_defaults_and_external_roots():
    args = compare.parse_args(
        ["--sglang-root", "checkouts/sglang/python", "--vllm-root", "checkouts/vllm-0.31.0"]
    )
    assert args.batches == [1, 2, 3, 4, 6, 8]
    assert args.shapes == [(16, 32), (16, 48)]
    assert args.providers == ["ours", "sglang", "vllm"]
    assert args.ours == "product"
    assert args.window == 8
    assert args.pairs == 20
    assert args.seconds == 1500
    assert args.regime == "cold"
    assert args.sglang_root == Path("checkouts/sglang/python")
    assert args.l2_bytes is None
    assert args.ceilings is None


def test_unified_product_comparator_needs_no_external_roots():
    args = compare.parse_args(["--ours", "unified", "--providers", "product", "product"])
    assert args.ours == "unified"
    assert args.providers == ["ours", "product"]
    assert args.sglang_root is None
    assert args.vllm_root is None
    assert "product" in compare.PROVIDERS


@pytest.mark.parametrize("window", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("mode", ["auto", "delayed"])
def test_window_argument(window, mode):
    args = compare.parse_args(
        ["--providers", "ours", "--ours", "unified", "--ours-mode", mode, "--window", str(window)]
    )
    assert args.window == window
    assert args.ours_mode == mode
    assert compare.parse_args(["--providers", "ours"]).window == 8


def test_unified_all_comparators():
    args = compare.parse_args(
        [
            "--ours",
            "unified",
            "--providers",
            "ours",
            "product",
            "sglang",
            "vllm",
            "--sglang-root",
            "checkouts/sglang/python",
            "--vllm-root",
            "checkouts/vllm-0.31.0",
        ]
    )
    assert args.providers == ["ours", "product", "sglang", "vllm"]


def test_argument_subset_always_includes_ours_and_deduplicates():
    args = compare.parse_args(
        [
            "--providers",
            "vllm",
            "vllm",
            "--vllm-root",
            "checkouts/vllm-0.31.0",
            "--shapes",
            "8,16",
            "16,48",
            "--batches",
            "2",
            "6",
            "--pairs",
            "60",
            "--seconds",
            "3.5",
            "--regime",
            "warm",
            "--l2-bytes",
            "1024",
            "--ceilings",
            "ceilings/device.json",
            "--jsonl",
            "results/comparison.jsonl",
        ]
    )
    assert args.providers == ["ours", "vllm"]
    assert args.shapes == [(8, 16), (16, 48)]
    assert args.batches == [2, 6]
    assert args.pairs == 60
    assert args.seconds == 3.5
    assert args.regime == "warm"
    assert args.l2_bytes == 1024
    assert args.jsonl == Path("results/comparison.jsonl")
    assert compare.parse_args(["--providers", "ours"]).sglang_root is None


@pytest.mark.parametrize(
    "args",
    [
        ["--pairs", "0"],
        *[["--window", value] for value in ("0", "1", "-2", "3", "24", "2.5", "nan")],
        ["--batches", "-1"],
        ["--shapes", "0,32"],
        ["--shapes", "16,33"],
        ["--shapes", "16"],
        ["--shapes", "a,b"],
        ["--seconds", "nan"],
        ["--seconds", "0"],
        ["--l2-bytes", "0"],
        ["--providers", "sglang"],
        ["--providers", "vllm"],
        ["--ours", "invalid"],
        ["--providers", "product"],
        ["--ours", "product", "--providers", "product"],
        ["--ours", "unified", "--providers", "product", "sglang"],
        ["--ours", "unified", "--providers", "product", "vllm"],
    ],
)
def test_bad_arguments(args):
    with pytest.raises(SystemExit):
        compare.parse_args(["--providers", "ours", *args])


@pytest.mark.parametrize("window", [2, 8, 32, 64])
def test_rotation_covers_three_l2_and_at_least_256_groups(window):
    copies, calls = compare.rotation(3, 48, 96 * 1024**2, "cold", window)
    assert copies * 4 * 3 * 48 * 128**2 >= 3 * 96 * 1024**2
    assert (copies - 1) * 4 * 3 * 48 * 128**2 < 3 * 96 * 1024**2
    assert calls == window * copies * math.ceil(256 / copies)
    assert compare.rotation(3, 48, 96 * 1024**2, "warm", window) == (1, window * 256)


@pytest.mark.parametrize(
    "kind,route,model,provider,window",
    [
        ("product", None, "per-step-writeback", "product", 1),
        ("unified", "product", "per-step-writeback", "product", 1),
        ("unified", "immediate", "per-step-writeback", "product", 1),
        ("unified", "delayed", "delayed", "cute", compare.STEPS),
    ],
)
@pytest.mark.parametrize("replay_window", [8, 32])
@pytest.mark.parametrize("replay_missing", [False, True])
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48), (8, 16), (8, 24)])
def test_sol_model_follows_effective_route(
    monkeypatch, kind, route, model, provider, window, hk, hv, replay_missing, replay_window
):
    from types import SimpleNamespace

    import torch

    from benchmark.kernel_tuning import gdn_replay_common, gdn_replay_inputs, gdn_replay_sol
    from ctikernels.gdn.replay import cute

    # Exercise row reporting with CPU stand-ins for allocation, correctness and timing.
    runner = SimpleNamespace(
        op=SimpleNamespace(provider="triton", decode_route=lambda b: route),
        launch_config=lambda: {},
        reset=lambda *args: None,
    )

    def factory(got_hk, got_hv, *args, **kwargs):
        assert (got_hk, got_hv) == (hk, hv)
        if kwargs:
            assert kwargs["window"] == replay_window
            assert kwargs["mode"] == (route if route in ("immediate", "delayed") else "auto")
        return runner

    monkeypatch.setattr(gdn_replay_common, "Product", factory)
    monkeypatch.setattr(compare, "Unified", factory)
    monkeypatch.setattr(cute, "effective_config", lambda *args: cute.Config())
    monkeypatch.setattr(
        gdn_replay_inputs,
        "case_inputs",
        lambda *args: (SimpleNamespace(args=lambda: ()), (), torch.zeros(1), torch.zeros(1)),
    )
    monkeypatch.setattr(compare, "check_correctness", lambda *args: {"ours": {"correct": True}})
    monkeypatch.setattr(compare, "check_unified_correctness", lambda *args: {"correct": True})
    monkeypatch.setattr(compare, "_memory_check", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    arange = torch.arange
    monkeypatch.setattr(torch, "arange", lambda *args, **kw: arange(*args, dtype=kw["dtype"]))
    monkeypatch.setattr(
        compare,
        "DecodeGraph",
        lambda *args: SimpleNamespace(
            profile={"kernels_per_call_including_memset": 2.0}, trial=lambda: 5.0
        ),
    )
    reports = []

    def report(*args):
        reports.append(args)
        return {"SOL": 0.5}

    monkeypatch.setattr(gdn_replay_sol, "report", report)
    flags = ["--ours", kind, "--providers", "ours", "--pairs", "1", "--ceilings", "ceilings.json"]
    if route in ("immediate", "delayed"):
        flags += ["--ours-mode", route]
    if replay_missing:
        from benchmark.kernel_tuning import gdn_decode_external

        def unavailable(root, sm, got_hk, got_hv, *args):
            assert (got_hk, got_hv) == (hk, hv)
            raise gdn_decode_external.UnavailableComparator(
                "SGLang checkout has no standalone ReplaySSM leaf"
            )

        monkeypatch.setattr(gdn_decode_external, "SGLangReplayDecode", unavailable)
        flags += ["--providers", "sglang_replay", "--sglang-root", "unused"]
    flags += ["--window", str(replay_window)]
    args = compare.parse_args(flags)
    row = {
        "copies": 1,
        "calls": 1,
        "profile": {},
        "self_pairs_us": [],
        "raw_pairs_us": {},
        "comparisons": {},
        "median_us": {},
    }
    compare.benchmark_row(args, hk, hv, 1, compare.Budget(30), row)
    assert row["status"] == "ok"
    if replay_missing:
        assert row["skipped"]["sglang_replay"] == {
            "status": "skipped",
            "reason": "SGLang checkout has no standalone ReplaySSM leaf",
            "shape": [hk, hv],
        }
        assert "sglang_replay" not in row["comparisons"]
    assert row["sol_model"] == model
    assert row["sol"] == {"SOL": 0.5}
    window = replay_window if route == "delayed" else window
    assert reports == [(provider, 1, hk, hv, window, 5.0, 2, args.ceilings, args.regime)]


def test_import_and_argument_parsing_do_not_import_gpu_or_serving_packages():
    code = """
import builtins
import sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'triton', 'vllm', 'sglang', 'ctikernels'}:
        raise AssertionError('unexpected dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import benchmark.kernel_tuning.gdn_decode_compare as m
import benchmark.kernel_tuning.gdn_decode_external
m.parse_args(['--providers', 'ours', '--pairs', '60'])
m.parse_args(['--ours', 'unified', '--providers', 'ours', 'product'])
assert not any(name.split('.')[0] in {'vllm', 'sglang', 'torch'} for name in sys.modules)
"""
    run_child(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        timeout=600,
        check=True,
    )


@pytest.mark.parametrize("batch", [1, 3])
def test_projection_preparation_is_independent_and_preserves_gemm_strides(batch):
    import torch

    from benchmark.kernel_tuning.gdn_decode_external import ExternalDecode

    runner = object.__new__(ExternalDecode)
    runner.hk, runner.hv, runner.dc = 1, 2, 512
    width = runner.dc + runner.hv * 128 + 2 * runner.hv
    x = torch.arange(batch * width).reshape(batch, width)
    saved = x.clone()
    qkvz, ba = runner.prepare(x)
    assert qkvz.is_contiguous() and ba.is_contiguous()
    assert torch.equal(torch.cat((qkvz, ba), dim=-1), x)
    b, a = ba.chunk(2, dim=-1)
    assert b.is_contiguous() == (batch == 1)
    assert a.is_contiguous() == (batch == 1)
    z = qkvz[:, runner.dc :].reshape(batch, runner.hv, 128)
    flattened = z.reshape(-1, 128)
    assert (flattened.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()) == (
        batch == 1
    )
    qkvz.zero_()  # vLLM's conv overwrites qkv; it must never alter the shared inputs.
    ba.zero_()
    assert torch.equal(x, saved)


def test_emit_preserves_raw_timings_and_flushes(tmp_path, capsys):
    row = {
        "B": 2,
        "shape": [16, 32],
        "regime": "cold",
        "status": "budget_truncated",
        "median_us": {},
        "comparisons": {},
        "self_pairs_us": [[1.2, 1.3]],
        "sol": None,
    }
    path = tmp_path / "result.jsonl"
    with path.open("w") as stream:
        compare.emit(row, stream)
        assert '"self_pairs_us": [[1.2, 1.3]]' in path.read_text()
    assert "budget_truncated" in capsys.readouterr().out


@pytest.mark.parametrize("fault", [None, "output", "state", "nan", "inactive", "history"])
def test_fresh_step_checks_mapping_finiteness_and_unselected_slots(fault):
    from types import SimpleNamespace

    import torch

    initial = torch.ones(5, 2, 2, 2)
    history = torch.arange(5 * 3 * 4).reshape(5, 3, 4).float()
    xs = torch.arange(3 * 4).reshape(1, 3, 4).float()

    def make_runner(external):
        state = initial.clone()
        cache = history.transpose(1, 2).contiguous() if external else history.clone()
        canonical = cache.transpose(1, 2) if external else cache

        def step(x, slots):
            indices = slots.long()
            state[indices] += 1
            canonical[indices] = torch.cat((canonical[indices, 1:], x[:, None, :]), dim=1)
            out = torch.ones(x.shape[0], 4)
            if external:
                if fault == "output":
                    out *= 1.03
                elif fault == "state":
                    state[indices] *= 1.03
                elif fault == "nan":
                    out[0, 0] = math.nan
                elif fault == "inactive":
                    state[0] += 1
                elif fault == "history":
                    canonical[0] += 1
            return out

        return SimpleNamespace(
            state=state,
            history=cache,
            weights=(),
            prepare=lambda x: x.clone(),
            step=step,
            op=lambda x, slots, state, history: step(x, slots),
            canonical_history=lambda: canonical,
        )

    runners = {"ours": make_runner(False), "sglang": make_runner(True)}
    checks = compare.check_correctness(runners, xs, initial, history, 3, compare.Budget(30))
    assert checks["ours"]["correct"]
    assert checks["sglang"]["correct"] == (fault is None)
    for name, runner in runners.items():
        assert torch.equal(runner.state, initial)
        restored = runner.history if name == "ours" else runner.canonical_history()
        assert torch.equal(restored, history)


@pytest.mark.parametrize("fault", [None, "late_output", "flush", "inactive", "history", "nan"])
@pytest.mark.parametrize("window", [2, 8, 32])
def test_unified_checks_wraps_partial_flush_and_sentinels(fault, window):
    from types import SimpleNamespace

    import torch

    initial = torch.ones(5, 1, 2, 2)
    history = torch.zeros(5, 3, 4)
    xs = torch.arange(window * 3 * 4).reshape(window, 3, 4).float()
    state, cache = initial.clone(), history.clone()
    checkpoint, conv_history = initial.clone(), history.clone()
    calls = []

    def reference_step(x, slots, state, history):
        indices = slots.long()
        state[indices] += 1
        history[indices] = torch.cat((history[indices, 1:], x[:, None]), dim=1)
        return x.repeat(1, 32)

    def unified_step(x, slots, positions, out):
        assert positions.tolist() == [len(calls)] * 3
        calls.append(x.clone())
        out.copy_(x.repeat(1, 32))
        if fault == "late_output" and len(calls) == 2 * window + 1:
            out.mul_(1.1)
        if fault == "nan" and len(calls) == 2 * window + 3:
            out[0, 0] = math.nan
        return out

    def flush(slots):
        assert len(calls) == 2 * window + 3
        assert slots.tolist() == [3, 2, 1]
        checkpoint[slots.long()] = state[slots.long()]
        conv_history[slots.long()] = cache[slots.long()]
        if fault == "flush":
            checkpoint[slots.long()] *= 1.1
        elif fault == "inactive":
            checkpoint[-1] += 1
        elif fault == "history":
            conv_history[slots.long()] += 1

    runner = SimpleNamespace(
        state=SimpleNamespace(checkpoint=checkpoint, conv_history=conv_history),
        step=unified_step,
        flush=flush,
    )
    reference = SimpleNamespace(op=reference_step, weights=(), state=state, history=cache)
    result = compare.check_unified_correctness(
        runner, reference, xs, initial, history, 3, compare.Budget(30), window
    )
    assert result["steps"] == 2 * window + 3
    assert len(result["outputs"]) == 2 * window + 3
    assert result["correct"] == (fault is None)
