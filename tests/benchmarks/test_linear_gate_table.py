# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU regressions for gated run completion, identity and dispatch qualification."""

import json
from dataclasses import asdict

import pytest

from benchmark.kernel_tuning import linear_smallm_table as table
from ctikernels.linear._dispatch import Config, qualifies
from ctikernels.linear.cute import mtiles_for

SHAPE = (4096, 4096, 10240)


def row(m=3, cfg=None, **updates):
    cfg = cfg or Config()
    return {
        "kind": "gate_accept",
        "device_class": "rtx5090",
        "model": "qwen3.5-9b",
        "name": "o_proj",
        "N": SHAPE[0],
        "K": SHAPE[1],
        "gate_stride": SHAPE[2],
        "M": m,
        "l2": "cold",
        "provider": "cute-gated",
        "launch_config": {
            "config": asdict(cfg),
            "mtiles": mtiles_for(m),
            "cta_threads": cfg.warps * 32,
            "gate_stride": SHAPE[2],
        },
        "market_bf16_reduced_precision": True,
        "paired_speedup": 1.2,
        "epsilon": 0.01,
        "verdict": "PASS",
        "correct": True,
        "ours_us": 10.0,
        "vllm_us": 12.0,
        "vllm_mode": "dynamic",
        "sglang_us": 13.0,
        "faster_comparator": "vllm",
        "market_us": 12.0,
        "sol_us": 8.0,
        "kernels_per_call": 1,
    } | updates


def run(rows, run_id, complete=True, ended=True):
    prefix = rows[0]["kind"]
    result = [
        {"kind": prefix + "_begin", "run_id": run_id},
        *(dict(r, run_id=run_id) for r in rows),
    ]
    if ended:
        result.append(
            {
                "kind": prefix + "_completion",
                "run_id": run_id,
                "rows": len(rows),
                "complete": complete,
                "failures": sum(r.get("verdict") == "FAIL" for r in rows),
            }
        )
    return result


def logged(tmp_path, *runs):
    path = tmp_path / "accept.log"
    path.write_text("\n".join("RESULT " + json.dumps(r) for records in runs for r in records))
    return table.read_logs([path], round_label="current round")


def evidence(tmp_path, **updates):
    return logged(
        tmp_path,
        run([row(m) for m in (3, 4)], "cold"),
        run([row(m, l2="engine", **updates) for m in (3, 4)], "engine"),
    )


def chosen(cfg=None):
    return {"rtx5090": {SHAPE: {(3, 4): cfg or Config()}}}


def qualify(rows, cfg=None):
    return table.gate_by_acceptance(chosen(cfg), rows, gated=True)["rtx5090"]


def test_both_edges_and_regimes_keep_bucket(tmp_path):
    assert qualify(evidence(tmp_path)) == chosen()["rtx5090"]


@pytest.mark.parametrize(
    "kind,pairs,ci_low,kept",
    [
        ("gb10", 60, 1.1, True),
        ("gb10", 20, 1.1, False),
        ("gb10", 60, 1.0, False),
        ("gb10", 60, None, False),
        ("rtx5090", 20, None, True),
        ("rtx5090", 20, 0.99, True),
        ("pro6000", 20, None, True),
    ],
)
def test_gated_pair_count_and_ci_qualification(tmp_path, kind, pairs, ci_low, kept):
    rows = logged(
        tmp_path,
        *[
            run(
                [
                    row(m, device_class=kind, l2=l2, pairs=pairs, ci_low=ci_low, ci_high=1.3)
                    for m in (3, 4)
                ],
                l2,
            )
            for l2 in ("cold", "engine", "hot")
        ],
    )
    candidate = {kind: {SHAPE: {(3, 4): Config()}}}
    result = table.gate_by_acceptance(candidate, rows, gated=True)[kind]
    assert result == (candidate[kind] if kept else {})


@pytest.mark.parametrize(
    "speedup,epsilon,kept",
    [(1.0625, 0.0625, False), (1.0626, 0.0625, True), (1.03, 0.001, False), (1.031, 0.001, True)],
)
def test_gain_must_be_strictly_above_margin_and_noise(tmp_path, speedup, epsilon, kept):
    # Same predicate as warm-up autotune: strictly above 1 + max(0.03, epsilon).
    rows = evidence(tmp_path, paired_speedup=speedup, epsilon=epsilon)
    assert (qualify(rows) == chosen()["rtx5090"]) is kept
    assert qualifies(speedup, epsilon) is kept


@pytest.mark.parametrize("engine", ["missing", "unfinished", "incomplete", "one_edge"])
def test_engine_required(tmp_path, engine):
    cold = run([row(m) for m in (3, 4)], "cold")
    rows = [row(m, l2="engine") for m in ((3,) if engine == "one_edge" else (3, 4))]
    extra = (
        []
        if engine == "missing"
        else run(rows, "engine", complete=engine != "incomplete", ended=engine != "unfinished")
    )
    assert qualify(logged(tmp_path, cold, extra)) == {}


@pytest.mark.parametrize(
    "update",
    [
        {"cfg": Config(splits=2)},
        {"gate_stride": 14336},
        {"launch_config": None},
        {"market_bf16_reduced_precision": False},
        {"provider": "unfused+cute"},
        {"paired_speedup": 1.029},
        {"epsilon": 0.25},
        {"correct": False, "verdict": "FAIL"},
    ],
)
def test_evidence_must_match_configuration_stride_and_market(tmp_path, update):
    assert qualify(evidence(tmp_path, **update)) == {}


def test_config_a_cannot_qualify_b(tmp_path):
    assert qualify(evidence(tmp_path), Config(splits=2)) == {}


@pytest.mark.parametrize("fail_first", [True, False])
@pytest.mark.parametrize("correctness_failure", [True, False])
def test_retry_order(tmp_path, fail_first, correctness_failure):
    good = [row(m) for m in (3, 4)]
    bad = [row(m, correct=not correctness_failure, verdict="FAIL") for m in (3, 4)]
    first, last = (bad, good) if fail_first else (good, bad)
    rows = logged(
        tmp_path,
        run(first, "first"),
        run(last, "last"),
        run([row(m, l2="engine") for m in (3, 4)], "engine"),
    )
    assert bool(qualify(rows)) is fail_first


def test_hot_only_vetoes_and_unfinished_correctness_failure_invalidates(tmp_path):
    base = evidence(tmp_path)
    assert (
        qualify(base + logged(tmp_path, run([row(3, l2="hot", paired_speedup=1.01)], "hot"))) == {}
    )
    failure = run([row(3, correct=False, verdict="FAIL")], "bad", ended=False)
    assert qualify(base + logged(tmp_path, failure)) == {}


def test_completion_count_and_prefix_are_scoped(tmp_path):
    records = run([row(3), row(4)], "cold")
    records[-1]["rows"] = 3
    assert not any(r.get("_finished") for r in logged(tmp_path, records))
    records[-1].update(rows=2, kind="accept_completion")
    assert not any(r.get("_finished") for r in logged(tmp_path, records))


def test_partitioned_files_preserve_edges_and_retry_order(tmp_path):
    records = []
    for i, (m, l2) in enumerate(((3, "cold"), (4, "cold"), (3, "engine"), (4, "engine"))):
        part = tmp_path / f"part_{i}.log"
        part.write_text("\n".join("RESULT " + json.dumps(r) for r in run([row(m, l2=l2)], str(i))))
        records.append(part)
    assert qualify(table.read_logs(records, round_label="current round")) == chosen()["rtx5090"]


def test_rewrite_literal_and_cli(tmp_path, monkeypatch, capsys):
    from ctikernels.linear import _dispatch

    monkeypatch.setattr(_dispatch, "TUNED_GATED", chosen(), raising=False)
    evidence(tmp_path)
    target = tmp_path / "dispatch.py"
    plain = table.BEGIN + "\nTUNED = {}\n" + table.END + "\n"
    target.write_text(plain + table.GATED_BEGIN + "\nTUNED_GATED = {}\n" + table.GATED_END + "\n")
    assert (
        table.main(["--gated", "--accept", str(tmp_path / "accept.log"), "--rewrite", str(target)])
        == 0
    )
    assert target.read_text().startswith(plain)
    namespace = {"Config": Config}
    exec(target.read_text(), namespace)
    assert namespace["TUNED_GATED"]["rtx5090"] == chosen()["rtx5090"]
    assert table.main(["--gated", "--markdown", "--accept", str(tmp_path / "accept.log")]) == 0
    assert "o_proj per-step totals" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        table.main(["--gated", "--accept", str(tmp_path / "accept.log"), "--margin", "0.02"])


def test_probe_candidates_require_same_config_at_observed_edges(tmp_path):
    a, b = Config(), Config(splits=2)
    probes = [
        row(m, cfg=cfg, kind="gate_probe", paired_speedup=ratio)
        for m, cfg, ratio in ((3, a, 1.4), (4, a, 1.02), (3, b, 1.15), (4, b, 1.2))
    ]
    rows = logged(tmp_path, run(probes, "probe"))
    assert table.select_gated_configs(rows)["rtx5090"] == chosen(b)["rtx5090"]
    notes = []
    one = logged(tmp_path, run([probes[0]], "one"))
    assert table.select_gated_configs(one, notes=notes)["rtx5090"] == chosen(a)["rtx5090"]
    assert "only edge M=3" in notes[0]
    assert not table.select_gated_configs(logged(tmp_path, run(probes, "cut", ended=False)))[
        "rtx5090"
    ]


def test_probe_cli(tmp_path, capsys):
    logged(tmp_path, run([row(3, kind="gate_probe"), row(4, kind="gate_probe")], "probe"))
    assert table.main(["--gated", "--probe", str(tmp_path / "accept.log"), "--candidates"]) == 0
    namespace = {"Config": Config}
    exec("result = " + capsys.readouterr().out, namespace)
    assert namespace["result"]["rtx5090"] == chosen()["rtx5090"]


def test_markdown_omits_unfinished_and_resolves_retries(tmp_path):
    rows = logged(
        tmp_path,
        run([row(3, ours_us=999.0)], "cut", ended=False),
        run([row(4, ours_us=77.0)], "old"),
        run([row(4)], "new"),
    )
    text = table.gated_markdown(rows)
    assert "999" not in text and "77.0000" not in text
    assert "10.0000" in text and "0.0800" in text  # 8 o_proj calls per step
    assert "12.0000 (dynamic)" in text


def test_gb10_markdown_shows_intervals_per_regime(tmp_path):
    rows = logged(
        tmp_path,
        *[
            run([row(3, device_class="gb10", l2=l2, ci_low=low, ci_high=1.3)], l2)
            for l2, low in (("cold", 1.01), ("engine", 1.02), ("hot", 1.03))
        ],
    )
    text = table.gated_markdown(rows)
    for low in (1.01, 1.02, 1.03):
        assert f"1.2000 [{low:.4f}, 1.3000]" in text


def test_probe_candidates_must_win_in_every_l2_regime():
    from benchmark.kernel_tuning import linear_smallm_table as table
    from ctikernels.linear._dispatch import Config

    def probe(cfg, l2, ratio):
        return {
            "kind": "gate_probe",
            "device_class": "rtx5090",
            "N": 4096,
            "K": 4096,
            "gate_stride": 10240,
            "M": 1,
            "l2": l2,
            "provider": "cute-gated",
            "correct": True,
            "paired_speedup": ratio,
            "launch_config": {
                "config": dict(vars(cfg)),
                "mtiles": 1,
                "cta_threads": 32 * cfg.warps,
                "gate_stride": 10240,
            },
            "_finished": True,
        }

    fast_cold, balanced = Config(64, 128, 3, 1), Config(64, 128, 3, 4)
    rows = [
        probe(fast_cold, "cold", 1.20),
        probe(fast_cold, "hot", 0.90),
        probe(balanced, "cold", 1.10),
        probe(balanced, "hot", 1.05),
    ]
    selected = table.select_gated_configs(rows)
    assert selected["rtx5090"][(4096, 4096, 10240)] == {(1, 1): balanced}


def test_markdown_lists_fallback_rows_and_every_regime(tmp_path):
    fallback = {
        "kind": "gate_accept",
        "device_class": "rtx5090",
        "model": "qwen3.5-9b",
        "name": "o_proj",
        "N": SHAPE[0],
        "K": SHAPE[1],
        "gate_stride": SHAPE[2],
        "M": 64,
        "l2": "cold",
        "provider": "unfused",
        "launch_config": None,
        "status": "fallback",
        "verdict": "FALLBACK",
    }
    rows = logged(
        tmp_path,
        run([row(4), fallback], "cold"),
        run([row(4, l2="engine", paired_speedup=1.31)], "engine"),
        run([row(4, l2="hot", paired_speedup=1.42)], "hot"),
    )
    text = table.gated_markdown(rows)
    assert "| 64 | market path |" in text
    assert "1.3100" in text and "1.4200" in text and "PASS/PASS/PASS" in text


def test_markdown_never_joins_regimes_of_another_launch(tmp_path):
    other = Config(64, 128, 2, 4)
    rows = logged(
        tmp_path,
        run([row(4)], "cold"),
        run([row(4, cfg=other, l2="engine", paired_speedup=1.31)], "engine"),
    )
    text = table.gated_markdown(rows)
    assert "1.3100" not in text and "PASS/other launch" in text


def test_markdown_never_joins_regimes_of_another_gate_stride(tmp_path):
    rows = logged(
        tmp_path,
        run([row(4)], "cold"),
        run([row(4, l2="engine", gate_stride=12288, paired_speedup=1.31)], "engine"),
    )
    text = table.gated_markdown(rows)
    assert "1.3100" not in text and "PASS/other launch" in text


def test_probe_compile_error_in_one_regime_blocks_the_candidate():
    cfg = Config(64, 128, 3, 4)
    ok = {
        "kind": "gate_probe",
        "device_class": "rtx5090",
        "N": SHAPE[0],
        "K": SHAPE[1],
        "gate_stride": SHAPE[2],
        "M": 1,
        "l2": "cold",
        "provider": "cute-gated",
        "status": "ok",
        "correct": True,
        "paired_speedup": 1.2,
        "launch_config": {
            "config": asdict(cfg),
            "mtiles": 1,
            "cta_threads": 32 * cfg.warps,
            "gate_stride": SHAPE[2],
        },
        "_finished": True,
    }
    failed = {
        k: v for k, v in ok.items() if k not in ("launch_config", "correct", "paired_speedup")
    } | {"l2": "hot", "status": "compile_error", "config": asdict(cfg)}
    assert table.select_gated_configs([ok])["rtx5090"] == {SHAPE: {(1, 1): cfg}}
    assert table.select_gated_configs([ok, failed])["rtx5090"] == {}
