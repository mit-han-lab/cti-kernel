# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""NumPy-free CPU checks of gated benchmark orchestration and market protocol."""

import sys
import types
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmark.kernel_tuning import linear_gate as gate
from ctikernels.linear._dispatch import Config


@pytest.mark.parametrize("mode,pairs", [("--accept", 20), ("--probe", 7)])
def test_parse(mode, pairs):
    args = gate.parse_args([mode, "--m", "1,2,3,4"])
    assert args.m == [1, 2, 3, 4] and args.pairs == pairs
    assert args.max_candidates == 8 and args.budget_seconds == 1500
    assert gate.parse_args([mode, "--m", "bxq"]).m == gate.parse_ms("bxq")
    assert gate.parse_args([mode]).m == gate.parse_ms("accept")


@pytest.mark.parametrize("pairs", [20, 21, 60])
def test_accept_pair_count(pairs):
    assert gate.parse_args(["--accept", "--pairs", str(pairs)]).pairs == pairs


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--probe", "--accept"],
        ["--accept", "--pairs", "7"],
        ["--accept", "--pairs", "19"],
        ["--accept", "--m", "65"],
        ["--probe", "--max-candidates", "0"],
        ["--probe", "--budget-seconds", "nan"],
    ],
)
def test_parse_rejects(argv):
    with pytest.raises(SystemExit):
        gate.parse_args(argv)


@pytest.mark.parametrize("fails", [True, False])
def test_sglang_loader_restores_modules(tmp_path, monkeypatch, fails):
    parent, sentinel = types.ModuleType("sglang"), types.ModuleType("sglang.srt.utils")
    monkeypatch.setitem(sys.modules, "sglang", parent)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sentinel)
    before = {k: v for k, v in sys.modules.items() if k == "sglang" or k.startswith("sglang.")}
    path = tmp_path / "elementwise.py"
    path.write_text(
        """from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import is_hip
assert is_arch_support_pdl() is False
assert is_hip() is False
def fused_sigmoid_mul(x, gate, inplace=False):
    return x, gate, inplace
"""
        + ("raise RuntimeError('test failure')\n" if fails else "")
    )
    if fails:
        with pytest.raises(RuntimeError):
            gate.load_sglang(path)
    else:
        assert gate.load_sglang(path)(1, 2) == (1, 2, False)
    after = {k: v for k, v in sys.modules.items() if k == "sglang" or k.startswith("sglang.")}
    assert before == after


def test_missing_market_accept_error_probe_skip(capsys):
    assert gate.main(["--accept"]) == 1
    assert gate.main(["--probe"]) == 0
    output = capsys.readouterr().out
    assert '"status":"error"' in output and '"status":"skip"' in output


def test_candidate_budget_and_diversity():
    from ctikernels.linear.cute import mtiles_for

    for m in (1, 8, 24, 64):
        configs = gate.candidates(4096, 4096, m, "rtx5090")
        assert len(configs) <= 8
        assert {c.variant for c in configs} == {"smem", "reg"}
        assert {1, 2, 4} <= {c.splits for c in configs}
        assert all(gate.gated_shared_bytes(c, mtiles_for(m)) <= 99 * 1024 for c in configs)


def test_legacy_shared_memory_api():
    class Legacy:
        variant, stages, tile_k = "smem", 2, 128

        def shared_bytes_for(self, mtiles):
            return 2 * (64 + 8 * mtiles) * 136 * 2

    assert gate.gated_shared_bytes(Legacy(), 3) == 2 * (64 + 2 * 8 * 3) * 136 * 2


@pytest.mark.parametrize("mode,pair_count", [("--accept", 20), ("--accept", 60), ("--probe", 7)])
@pytest.mark.parametrize("faster", ["vllm", "sglang"])
def test_fake_measurement_schema_and_protocol(monkeypatch, mode, pair_count, faster):
    from ctikernels.linear import inference

    args = gate.parse_args([mode, "--m", "2", "--pairs", str(pair_count)])
    args.run_id = "cpu"
    shape = dict(gate.builtin_shapes()["9b"], N=4, K=256, gate_stride=512)
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    g = torch.zeros(2, 512, dtype=torch.bfloat16)[:, :256]
    w = torch.ones(4, 256, dtype=torch.bfloat16)
    captures, pairs, flags = [], [], []

    class Op:
        provider, kernels_per_call = "cute-gated", 1
        launch_config = {
            "config": asdict(Config()),
            "mtiles": 1,
            "cta_threads": 128,
            "gate_stride": 512,
        }

        def __init__(self, *, provider, config):
            assert provider == ("auto" if args.accept else "cute")

        def __call__(self, x, w, *, gate):
            return F.linear((x.float() * torch.sigmoid(gate.float())).to(x.dtype), w)

    def instance(name, call, xx, ww, rotations):
        captures.append(name)
        flags.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        assert xx is x and rotations == [w]
        return SimpleNamespace(name=name, output=call(xx, ww), copies=2)

    def paired(a, b, count, reps, flush):
        pairs.append((a.name, b.name, count))
        times = {
            "ours": 8.0,
            "dynamic": 10.0,
            "static": 12.0,
            "sglang": 11.0 if faster == "vllm" else 9.0,
        }
        return [times[a.name]] * count, [times[b.name]] * count

    def sglang(xx, gg, *, inplace):
        assert gg.shape == (2, 1, 256) and gg.stride(0) == 512 and inplace is False
        return xx * torch.sigmoid(gg.view_as(xx))

    monkeypatch.setattr(inference, "Linear", Op)
    monkeypatch.setattr(gate, "Instance", instance)
    monkeypatch.setattr(gate, "paired", paired)
    monkeypatch.setattr(gate, "kernels_per_replay", lambda _: 1)
    compiled = {mode: lambda xx, gg: xx * torch.sigmoid(gg) for mode in ("dynamic", "static")}
    row = gate.measure_row(
        args,
        shape,
        2,
        x,
        g,
        w,
        [w],
        compiled,
        sglang,
        None,
        "rtx5090",
        "RTX 5090",
        None if args.accept else Config(),
    )
    required = {
        "device",
        "device_class",
        "model",
        "name",
        "N",
        "K",
        "M",
        "gate_stride",
        "l2",
        "timing_method",
        "rotation_copies",
        "pairs",
        "self_pairs",
        "reps_per_sample",
        "provider",
        "launch_config",
        "run_id",
        "kernels_per_call",
        "logical_kernels_per_call",
        "ours_us",
        "vllm_us",
        "vllm_mode",
        "vllm_dynamic_us",
        "vllm_static_us",
        "sglang_us",
        "faster_comparator",
        "market_us",
        "speedup_vs_vllm",
        "speedup_vs_sglang",
        "paired_speedup",
        "epsilon",
        "epsilon_method",
        "verdict",
        "sol_us",
        "sol_fraction",
        "market_sol_fraction",
        "fastest_market",
        "market_bf16_reduced_precision",
        "correct",
        "vllm_correct",
        "sglang_correct",
    }
    assert required <= row.keys()
    assert row["correct"] and row["vllm_correct"] and row["sglang_correct"]
    assert row["faster_comparator"] == faster and row["vllm_mode"] == "dynamic"
    assert row["ours_us"] == 8 and row["epsilon"] == 0 and row["verdict"] == "PASS"
    assert row["sol_us"] == 2 * (4 * 256 + 2 * 2 * 256 + 2 * 4) / (1681.1 * 1e3)
    assert captures[0] == "ours" and all(flags)
    assert row["pairs"] == pair_count
    if args.accept:
        expected_ratio = (10.0 if faster == "vllm" else 9.0) / 8.0
        assert row["paired_speedup"] == row["ci_low"] == row["ci_high"] == expected_ratio
        assert row["ci_method"] == "bootstrap_median_percentile95_2000"
        assert row["self_pairs"] == row["selection_pairs"] == pair_count
        assert sum(count for a, b, count in pairs if b == "ours") == 2 * pair_count
        assert sum(count for a, b, count in pairs if a == b) == pair_count
    else:
        assert [(a, count) for a, b, count in pairs if b == "ours"] == [
            ("dynamic" if faster == "vllm" else "sglang", 7)
        ]


@pytest.mark.parametrize("l2", ["cold", "engine", "hot"])
def test_weight_rotation_regimes(monkeypatch, l2):
    weight = torch.ones(1)
    monkeypatch.setattr(gate, "acceptance_copies", lambda *a: 3)
    monkeypatch.setattr(gate, "hot_copies", lambda *a: 4)
    rotations = gate.weight_rotations(weight, {"N": 1, "K": 1}, l2, 1024)
    if l2 == "cold":
        assert len(rotations) == 2 and all(w is not weight for w in rotations)
    elif l2 == "engine":
        assert len(rotations) == 4
        assert rotations[1] is weight and rotations[3] is weight
        assert rotations[0] is not weight and rotations[2] is not weight
    else:
        assert len(rotations) == 3 and all(w is weight for w in rotations)


@pytest.mark.parametrize("budget_expired", [True, False])
@pytest.mark.parametrize("kind,pairs", [("rtx5090", 20), ("gb10", 20), ("gb10", 60)])
def test_fake_run_markers_and_budget(monkeypatch, capsys, budget_expired, kind, pairs):
    from benchmark.kernel_tuning import linear_smallm_table as table
    from ctikernels.linear import _dispatch

    args = gate.parse_args(["--accept", "--model", "9b", "--m", "1,2", "--pairs", str(pairs)])
    captured = []
    elapsed = [0.0]
    monkeypatch.setattr(gate.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(gate, "emit", captured.append)
    monkeypatch.setattr(_dispatch, "device_class", lambda *a: kind)
    monkeypatch.setattr(
        gate.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170, L2_cache_size=1024),
    )
    monkeypatch.setattr(gate.torch.cuda, "get_device_name", lambda _: "RTX 5090")
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gate, "L2Flush", lambda _: None)
    monkeypatch.setattr(gate, "make_inputs", lambda *a: (torch.ones(2, 1),) * 3)
    monkeypatch.setattr(gate, "weight_rotations", lambda *a: [])
    monkeypatch.setattr(gate, "compile_vllm", lambda *a: {})
    monkeypatch.setattr(gate, "fused", lambda *a: True)

    def measure(args, shape, m, *a):
        elapsed[0] = args.budget_seconds + 1 if budget_expired else 0
        return dict(
            shape,
            kind="gate_accept",
            M=m,
            run_id=args.run_id,
            correct=True,
            verdict="PASS",
            status="ok",
        )

    monkeypatch.setattr(gate, "measure_row", measure)
    assert gate.run(args, None) == int(budget_expired)
    stderr = capsys.readouterr().err
    if kind == "gb10" and pairs < 60:
        assert "--pairs >= 60" in stderr and "cannot qualify" in stderr
    else:
        assert not stderr
    assert captured[0]["kind"] == "gate_accept_begin"
    assert captured[-1]["kind"] == "gate_accept_completion"
    assert captured[-1]["rows"] == (1 if budget_expired else 2)
    assert captured[-1]["failures"] == 0
    assert captured[-1]["complete"] is not budget_expired
    assert bool(captured[-1]["skipped"]) is budget_expired
    assert len({r["run_id"] for r in captured}) == 1
    table.mark_completion(captured)
    assert captured[1]["_complete"] is not budget_expired


def test_loader_failure_has_bounded_diagnostics(monkeypatch, capsys):
    def broken(_):
        raise RuntimeError("private diagnostic must not be copied")

    monkeypatch.setattr(gate, "load_sglang", broken)
    assert gate.main(["--accept"]) == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output and "private diagnostic" not in output


def test_accept_records_unfused_rows_as_fallback_without_timing(monkeypatch):
    args = gate.parse_args(["--accept", "--model", "9b", "--m", "1,2"])
    captured = []
    monkeypatch.setattr(gate, "emit", captured.append)
    monkeypatch.setattr(
        gate.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170, L2_cache_size=1024),
    )
    monkeypatch.setattr(gate.torch.cuda, "get_device_name", lambda _: "RTX 5090")
    monkeypatch.setattr(gate.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gate, "L2Flush", lambda _: None)
    monkeypatch.setattr(gate, "make_inputs", lambda *a: (torch.ones(2, 1),) * 3)
    monkeypatch.setattr(gate, "weight_rotations", lambda *a: [])
    monkeypatch.setattr(gate, "compile_vllm", lambda *a: {})
    monkeypatch.setattr(gate, "fused", lambda shape, m: m == 1)
    monkeypatch.setattr(
        gate,
        "measure_row",
        lambda args, shape, m, *a: dict(
            shape, kind="gate_accept", M=m, run_id=args.run_id, verdict="PASS", status="ok"
        ),
    )
    assert gate.run(args, None) == 0
    rows = [r for r in captured if r["kind"] == "gate_accept"]
    assert [(r["M"], r["verdict"]) for r in rows] == [(1, "PASS"), (2, "FALLBACK")]
    assert rows[1]["provider"] == "unfused" and rows[1]["launch_config"] is None
    assert captured[-1]["failures"] == 0 and captured[-1]["complete"] is True
