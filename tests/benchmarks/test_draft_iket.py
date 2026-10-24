# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks of native IKET decoding, CTA aggregation and runner failures."""

import json

import pytest

from benchmark.kernel_tuning import draft_iket as bench
from benchmark.kernel_tuning import draft_iket_node as node


def decoded_trace():
    def event(name, start, end, locations):
        return {"rangeNameIdx": name, "startTs": start, "endTs": end, "warpLocIdxs": locations}

    return {
        "stringTable": ["attn.walk", "attn.merge", "attn.merge_wait", "merge.all"],
        "locationTable": [
            {"ctaId": [0, 0, 0], "warpId": 0},
            {"ctaId": [0, 0, 0], "warpId": 1},
            {"ctaId": [1, 0, 0], "warpId": 0},
        ],
        "launches": [
            {
                "kernelName": "Attention.kernel",
                "ranges": [
                    event(0, 10, 30, [0, 1]),
                    event(0, 20, 40, [1]),  # Warp overlap: 30 ns, not 60.
                    event(0, 120, 190, [2]),
                    event(1, 40, 80, [0, 1]),
                    event(2, 40, 50, [0, 1]),
                    event(1, 200, 290, [2]),
                    event(2, 200, 220, [2]),
                ],
                "warpLifetimes": [
                    {"startTs": 0, "endTs": 100, "locIdx": 0},
                    {"startTs": 2, "endTs": 99, "locIdx": 1},
                    {"startTs": 100, "endTs": 300, "locIdx": 2},
                ],
            },
            {
                "kernelName": "Merge.kernel_merge",
                "ranges": [event(3, 400, 450, [0, 1])],
                "warpLifetimes": [{"startTs": 390, "endTs": 490, "locIdx": 0}],
            },
        ],
    }


def summarize(tmp_path, data):
    path = tmp_path / "synthetic.trace.json"
    path.write_text(json.dumps(data))
    return bench.summarize(path)


def test_summary_unions_warps_per_cta_and_keeps_kernels_separate(tmp_path, capsys):
    rows = summarize(tmp_path, decoded_trace())
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == rows
    by_name = {row["range"]: row for row in rows}
    walk = by_name["attn.walk"]
    assert walk["kernel"] == "Attention.kernel"
    assert walk["count"] == walk["sampled_ctas"] == 2
    assert walk["total_duration_ns"] == 100
    assert walk["median_duration_per_cta_ns"] == walk["mean_duration_per_cta_ns"] == 50
    assert walk["total_cta_span_ns"] == 300
    assert walk["fraction_kernel_span"] == pytest.approx(1 / 3)
    assert walk["span_basis"] == ["warp_lifetimes"]
    merge, wait = by_name["attn.merge"], by_name["attn.merge_wait"]
    assert merge["total_duration_ns"] == 130
    assert wait["total_duration_ns"] == 30
    assert wait["nested_in"] == "attn.merge"
    assert wait["fraction_kernel_span"] == pytest.approx(0.1)
    assert merge["total_cta_span_ns"] == 300  # Nested wait never increases span.
    legacy = by_name["merge.all"]
    assert legacy["kernel"] == "Merge.kernel_merge"
    assert legacy["count"] == 1
    assert legacy["mean_duration_per_cta_ns"] == 50
    assert legacy["fraction_kernel_span"] == 0.5


def test_graph_launches_stay_separate_and_fallback_span_uses_endpoints(tmp_path):
    data = decoded_trace()
    data["launches"] = data["launches"][:1]
    data["graphLaunches"] = {"graph:0": data["launches"]}
    data["launches"][0].pop("warpLifetimes")
    rows = summarize(tmp_path, data)
    walk = next(row for row in rows if row["range"] == "attn.walk")
    assert walk["count"] == walk["sampled_ctas"] == 4
    assert walk["total_duration_ns"] == 200
    assert walk["mean_duration_per_cta_ns"] == 50
    assert walk["total_cta_span_ns"] == 2 * (70 + 170)
    assert walk["span_basis"] == ["range_endpoints"]


def test_large_integer_timestamps_preserve_nanosecond_differences(tmp_path):
    data = decoded_trace()
    for launch in data["launches"]:
        for event in launch["ranges"] + launch["warpLifetimes"]:
            event["startTs"] += 2**60
            event["endTs"] += 2**60
    rows = summarize(tmp_path, data)
    assert next(row for row in rows if row["range"] == "attn.walk")["total_duration_ns"] == 100


@pytest.mark.parametrize("data", [{}, {"launches": []}])
def test_missing_decoded_ranges_fail(tmp_path, data):
    with pytest.raises(ValueError, match="decoded IKET|no DRAFT ranges"):
        summarize(tmp_path, data)


def test_missing_locations_fail_instead_of_counting_warps_as_ctas(tmp_path):
    data = decoded_trace()
    data["launches"][0]["ranges"][0].pop("warpLocIdxs")
    with pytest.raises(ValueError, match="no warp locations"):
        summarize(tmp_path, data)


def test_workload_args():
    args = bench.parse_args(
        [
            "--heads",
            "32/8",
            "--query",
            "8",
            "--batch",
            "2",
            "--length",
            "4096",
            "--causal",
            "--window",
            "2048",
            "--merge",
            "legacy",
        ]
    )
    assert (args.q_heads, args.kv_heads, args.hd, args.capacity) == (32, 8, 128, 32768)
    assert (args.query, args.batch, args.length, args.window, args.causal) == (
        8,
        2,
        4096,
        2048,
        True,
    )
    assert bench.parse_args(["--window", "None"]).window is None


@pytest.mark.parametrize("args", [["--heads", "7/2"], ["--window", "0"], ["--length", "1"]])
def test_invalid_workload_args(args):
    with pytest.raises(SystemExit):
        bench.parse_args(args)


@pytest.mark.parametrize(
    ("profile_rc", "summary_rc", "expected"), [(0, 0, 0), (7, 0, 7), (0, 8, 8)]
)
def test_node_uses_module_launcher_and_propagates_failures(
    tmp_path, monkeypatch, capsys, profile_rc, summary_rc, expected
):
    output = tmp_path / "trace"
    output.mkdir()
    (output / "synthetic.trace.json").write_text("{}")
    monkeypatch.setattr(node.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(node.sys, "argv", ["draft_iket_node.py", "--merge", "tree"])
    calls = []

    def call(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CTI_DRAFT_IKET"] == "1"
        return profile_rc if len(calls) == 1 else summary_rc

    monkeypatch.setattr(node.subprocess, "call", call)
    assert node.main() == expected
    assert "from iket.cli.main import entrypoint" in calls[0][2]
    assert calls[0][-2:] == ["--merge", "tree"]
    assert calls[0][calls[0].index("--enabled-cluster") + 1] == "0,0,0"
    assert "--summarize" in calls[1]
    assert f"DRAFT_IKET_RC {profile_rc}" in capsys.readouterr().out
