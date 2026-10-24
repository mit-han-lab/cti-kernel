# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU tests of the offline tuner: search, keep rule, sensitivity and merge."""

from __future__ import annotations

import json

import pytest

from benchmark.kernel_tuning.tune import __main__ as cli
from benchmark.kernel_tuning.tune import harness, shapes
from ctikernels._runtime.tables import DeviceClass, Dim, TunedOp, encode

PRO = DeviceClass(12, 0, 188)


def _op():
    return TunedOp(
        "test.tune",
        version=1,
        space={"block": (16, 32, 64), "warps": (4, 8)},
        dims=(Dim("m", (1, 4, 16, 64)), Dim("k")),
        default=lambda shape, device: {"block": 16, "warps": 4},
    )


def _cost(shape, config):
    """Fake time: m <= 4 likes block 32, larger m likes block 64; warps 8 is 2% faster."""
    best = 32 if shape["m"] <= 4 else 64
    return (10 + abs(config["block"] - best) / 8) * (0.98 if config["warps"] == 8 else 1.0)


def _case(shape, broken=()):
    state = {}

    def run(config):
        def launch():
            if config["block"] in broken:
                raise RuntimeError("does not compile")
            state["out"] = [1.0]
            return _cost(shape, config)

        return launch

    return harness.Case(shape, run, lambda: state["out"], lambda a, b: a == b)


def _timer(launch):
    return [launch()] * 3  # the fake launch returns its own cost


def _search(op, shape, broken=()):
    return harness.search(op, PRO, _case(shape, broken), _timer, list, lambda a, b: a == b)


def test_search_skips_configs_the_op_declares_invalid():
    op = TunedOp(
        "test.valid",
        version=1,
        space={"block": (16, 32, 64), "warps": (4, 8)},
        dims=(Dim("m", (1, 4, 16, 64)), Dim("k")),
        default=lambda shape, device: {"block": 16, "warps": 4},
        valid=lambda config, shape: config["block"] <= 32,
    )
    bucket = _search(op, {"m": 64, "k": 128})
    assert bucket.best["block"] == 32
    assert {us for config, us in bucket.tried if config["block"] == 64} == {None}


def test_search_ignores_gains_below_the_move_threshold(monkeypatch):
    monkeypatch.setattr(harness, "MOVE", 0.05)  # warps 8 gains only 2%
    assert _search(_op(), {"m": 2, "k": 128}).best == {"block": 32, "warps": 4}


def test_search_finds_best_and_skips_broken_configs():
    op = _op()
    bucket = _search(op, {"m": 2, "k": 128}, broken=(64,))
    assert bucket.best == {"block": 32, "warps": 8}
    assert any(us is None for _, us in bucket.tried)
    assert bucket.key == (4, 128)
    assert list(bucket.errors) == ["RuntimeError: does not compile"] and not bucket.all_rejected


def test_search_flags_buckets_whose_candidates_all_fail():
    op = _op()
    case = _case({"m": 2, "k": 128})
    case.close = lambda got, ref: False  # e.g. an adapter comparing mismatched tensors
    bucket = harness.search(op, PRO, case, _timer, list, lambda a, b: a == b)
    assert bucket.all_rejected and bucket.best == op.default(PRO, m=2, k=128)


def _speedup(bucket, a, b):
    return _cost(bucket.shape, a) / _cost(bucket.shape, b)


def test_keep_rule_inherits_unless_three_percent_faster():
    op = _op()
    buckets = [_search(op, {"m": m, "k": 128}) for m in (1, 4, 16, 64)]
    rows, provenance = harness.keep_rows(op, PRO, buckets, _speedup)
    # m=1 beats the default; m=4 equals m=1's best; m=16 switches block; m=64 inherits.
    assert rows == {(1, 128): {"block": 32, "warps": 8}, (16, 128): {"block": 64, "warps": 8}}
    assert [row["kept"] for row in provenance] == [True, False, True, False]
    assert all(row["speedup_vs_default"] >= 1 for row in provenance)


def test_sensitivity_probes_adjacent_neighbours_only():
    op = _op()
    buckets = [_search(op, {"m": m, "k": k}) for m in (1, 4, 16, 64) for k in (128, 256)]
    probed = []

    def probe(bucket, configs):
        probed.append((bucket.shape["m"], len(configs)))
        return [_cost(bucket.shape, c) for c in configs]

    report = harness.summarize([harness.sensitivity(op, buckets, probe)])
    assert set(report) == {"m"}  # k never changes the best config
    # Per k, m=4 <-> m=16 is the only adjacent pair whose best differs, probed both ways.
    assert sorted(probed) == [(4, 2), (4, 2), (16, 2), (16, 2)]
    assert report["m"]["pairs"] == 4 and report["m"]["over_keep"] == 4


def test_merge_unions_jobs_and_rejects_conflicts(tmp_path):
    op = _op()
    device = {"device_class": "sm120-170", "device": "test"}
    for job, rows in (
        ("a", {(1, 128): {"block": 32, "warps": 4}}),
        ("b", {(1, 256): {"block": 64, "warps": 4}}),
    ):
        (tmp_path / job).mkdir()
        (tmp_path / job / "test.tune@sm120-170.json").write_text(encode(op, rows))
        report = {"m": {"pairs": 1, "speedups": [1.1]}, "k": {"pairs": 0, "speedups": []}}
        (tmp_path / job / "test.tune@sm120-170.provenance.json").write_text(
            harness.provenance(op, device, [], report)
        )
    cli.merge([tmp_path / "a", tmp_path / "b"], tmp_path)
    merged = json.loads((tmp_path / "test.tune@sm120-170.json").read_text())
    assert [row[0] for row in merged["rows"]] == [[1, 128], [1, 256]]
    (tmp_path / "b" / "test.tune@sm120-170.json").write_text(
        encode(op, {(1, 128): {"block": 64, "warps": 4}})
    )
    with pytest.raises(SystemExit):
        cli.merge([tmp_path / "a", tmp_path / "b"], tmp_path)


def test_shards_partition_whole_groups():
    op = _op()
    every = [{"m": m, "k": k} for m in (1, 4, 16) for k in (128, 256, 512)]
    parts = [harness.shard(op, every, i, 2) for i in range(2)]
    assert sorted(map(str, parts[0] + parts[1])) == sorted(map(str, every))
    assert {s["k"] for s in parts[0]}.isdisjoint({s["k"] for s in parts[1]})


def test_imported_rows_follow_the_keep_rule():
    op = TunedOp(
        "test.import",
        version=1,
        space={"chunks": (1, 2, 4, 8)},
        dims=(Dim("m", (1024, 2048, 4096)), Dim("n")),
        default=lambda shape, device: {"chunks": 1},
    )
    speedups = {1024: {1: 1.0, 2: 1.01, 4: 1.0, 8: 0.9}, 2048: {1: 1.0, 2: 1.0, 4: 1.08, 8: 1.06},
                4096: {1: 1.0, 2: 1.0, 4: 1.09, 8: 1.11}}  # fmt: skip
    measured = [
        {"shape": {"m": m, "n": 4096}, "config": {"chunks": c}, "speedup": v}
        for m, row in speedups.items()
        for c, v in row.items()
    ]
    rows, provenance = harness.imported(op, PRO, measured)
    # 1024: +1% stays default; 2048: chunks 4 (+8%); 4096: chunks 8 only +1.8% over 4 -> inherits.
    assert rows == {(2048, 4096): {"chunks": 4}}
    assert [r["kept"] for r in provenance] == [False, True, False]


def test_annotate_adds_measured_sol_fraction():
    from benchmark.sol.sol import Call, Kernel, estimate

    call = Call([Kernel("copy", dram_bytes={"activations": 256 << 20}, n_ctas=1024)])
    sol_us = estimate(call, "rtx-pro-6000").t_ref_us
    case = harness.Case({"m": 1}, lambda config: lambda: config["us"], list, None, sol=lambda: call)
    cases = type("Cases", (), {"get": lambda self, shape: case})()
    rows = [{"shape": {"m": 1}, "config": {"us": sol_us / 0.95}},
            {"shape": {"m": 1}, "config": {"us": sol_us / 0.80}}]  # fmt: skip
    cli._annotate(object(), "sm120-188", cases, rows, lambda launch: [launch()] * 3)
    assert [r["sol_fraction"] for r in rows] == [0.95, 0.8]
    assert [r["under_bar"] for r in rows] == [False, True]
    gb300 = [{"shape": {"m": 1}, "config": {"us": 5.0}}]
    cli._annotate(object(), "sm103-148", cases, gb300, lambda launch: [launch()])
    assert gb300[0]["us"] == 5.0 and "sol_fraction" not in gb300[0]  # no measured ceiling


@pytest.mark.parametrize("model", sorted(shapes.MODELS))
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_projection_shapes_split_by_tp(model, tp):
    full = {p.name: p for p in shapes.projections(model, 1)}
    for p in shapes.projections(model, tp):
        assert p.k * sum(p.sizes) * tp >= full[p.name].k * sum(full[p.name].sizes)
