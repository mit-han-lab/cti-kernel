# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import json

from benchmark.kernel_tuning import linear_dsv4_report as report

SHAPES = list(report.CALLS)


def run(run_id, correct=True, speedup=1.5, m=2, shapes=SHAPES):
    """One acceptance run per output dtype, as `linear_smallm.py --accept` writes them."""
    lines = []
    for dtype in ("bf16", "fp32"):
        group = [(n, k) for n, k, out_dtype in shapes if out_dtype == dtype]
        if not group:
            continue
        rid = f"{run_id}-{dtype}"
        lines.append({"kind": "accept_begin", "run_id": rid, "out_dtype": dtype})
        for n, k in group:
            lines.append(
                {
                    "kind": "smallm_accept",
                    "run_id": rid,
                    "device": "NVIDIA GB10",
                    "device_class": "gb10",
                    "out_dtype": dtype,
                    "provider": "cute",
                    "N": n,
                    "K": k,
                    "M": m,
                    "l2": "cold",
                    "ours_us": 10.0 / speedup,
                    "cublas_us": 10.0,
                    "paired_speedup": speedup,
                    "epsilon": 0.01,
                    "correct": correct,
                    "verdict": "PASS" if correct else "FAIL",
                }
            )
        lines.append(
            {
                "kind": "accept_completion",
                "run_id": rid,
                "out_dtype": dtype,
                "rows": len(group),
                "failures": 0 if correct else len(group),
                "complete": True,
            }
        )
    return ["RESULT " + json.dumps(line) for line in lines]


def render(tmp_path, *runs):
    path = tmp_path / "accept.log"
    path.write_text("\n".join(line for lines in runs for line in lines) + "\n")
    latest = report.records([path])
    return report.markdown(latest), report.step_totals(latest)


def test_correctness_fail_rows_are_shown_and_excluded_from_totals(tmp_path):
    table, totals = render(tmp_path, run("a", correct=False, speedup=2.0))
    assert table.count("| FAIL | FAIL |") == len(SHAPES)
    assert "2.000" not in table and "2.000" not in totals
    assert "| gb10 | 2 | FAIL: " in totals and "excluded" in totals


def test_fail_after_pass_invalidates_and_later_pass_restores(tmp_path, monkeypatch):
    # Keep the mixed-provider total independent of future tuning-table updates.
    monkeypatch.setattr(
        report._dispatch,
        "TUNED",
        {"gb10": {(64, 4096): {(1, 4): report._dispatch.Config()}}},
    )
    monkeypatch.setattr(report._dispatch, "TUNED_F32", {})
    _, totals = render(tmp_path, run("a", speedup=1.25), run("b", correct=False))
    assert "FAIL: " in totals and "1.250" not in totals
    table, totals = render(tmp_path, run("a", correct=False), run("b", speedup=1.25))
    assert "FAIL" not in table and "FAIL" not in totals
    # Totals use the shipped table: CuTe time where dispatched, cuBLAS elsewhere.
    assert "| gb10 | 2 | 0.798 | 0.840 | 1.053 |" in totals


def test_partial_fail_names_only_failing_projection(tmp_path):
    bad = [(512, 4096, "fp32")]
    good = [shape for shape in SHAPES if shape not in bad]
    _, totals = render(tmp_path, run("a", shapes=good), run("b", correct=False, shapes=bad))
    assert "FAIL: indexer.compressor.wkv_gate |" in totals


def test_unfinished_run_rows_are_pending(tmp_path):
    lines = [line for line in run("a", speedup=1.25) if "accept_completion" not in line]
    table, totals = render(tmp_path, lines)
    assert "1.250" not in table and totals.count("\n") == 1


def test_gb10_intervals_are_shown(tmp_path):
    lines = []
    for line in run("a", speedup=1.071):
        record = json.loads(line.removeprefix("RESULT "))
        if record["kind"] == "smallm_accept":
            record.update(ci_low=1.052, ci_high=1.09)
        lines.append("RESULT " + json.dumps(record))
    text, _ = render(tmp_path, lines)
    assert text.count("1.071 [1.052, 1.090] (0.010)") == len(SHAPES)


def test_qwen3_moe_model_names_rows_and_counts_48_routers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        report._dispatch, "TUNED", {"gb10": {(128, 2048): {(2, 2): report._dispatch.Config()}}}
    )
    path = tmp_path / "accept.log"
    shapes = list(report.QWEN3_MOE_CALLS)
    path.write_text("\n".join(run("a", speedup=2.0, shapes=shapes + SHAPES)) + "\n")
    assert report.main([str(path), "--model", "qwen3-30b-a3b"]) == 0
    table, totals = capsys.readouterr().out.split("\n\n")
    assert "router.gate" in table and "lm_head" in table and "indexer" not in table
    # 48 routers at 5 us (CuTe) + lm_head at 10 us (cuBLAS) vs 49 cuBLAS calls at 10 us.
    assert "| gb10 | 2 | 0.250 | 0.490 | 1.960 |" in totals
