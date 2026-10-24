# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU accounting and saved default-kernel timing regressions (no GPU imports)."""

import json
from pathlib import Path

import pytest

from benchmark.kernel_tuning import gdn_prefill_sol as model

ROOT = Path(__file__).resolve().parents[3]
# The shared SOL helper and ceilings are vendored under benchmark/sol (see PROVENANCE.md).
HELPER_ROOT = ROOT


@pytest.fixture(autouse=True)
def helper_root(monkeypatch):
    assert (HELPER_ROOT / "benchmark/sol/sol.py").is_file()
    monkeypatch.setattr(model, "HELPER_ROOT", HELPER_ROOT)


def test_gdn_prefill_sol_accounting():
    model.self_test()


# Saved p3 default profiles: ours, tune=""; durations in microseconds.
# Only row contracts and timing numbers are retained from the instrumented logs.
PROFILE_TIMES = [
    (
        "rtx5090",
        "16x48:4096+4096",
        {"prepare_key": 195.02, "prepare_value": 395.09, "recurrence": 737.95, "norm": 256.84},
    ),
    (
        "rtx5090",
        "16x48:8192",
        {"prepare_key": 196.6, "prepare_value": 394.08, "recurrence": 760.8, "norm": 249.19},
    ),
    (
        "rtx-pro-6000",
        "16x48:4096+4096",
        {"prepare_key": 185.63, "prepare_value": 444.98, "recurrence": 864.13, "norm": 278.15},
    ),
    ("rtx-pro-6000", "16x32:4096", {"prepare_value": 196.32, "recurrence": 349.37, "norm": 72.74}),
    (
        "rtx-pro-6000",
        "16x48:2048",
        {"prepare_key": 62.58, "prepare_value": 115.19, "recurrence": 211.01, "norm": 40.67},
    ),
    (
        "rtx-pro-6000",
        "16x48:8192",
        {"prepare_key": 185.66, "prepare_value": 444.71, "recurrence": 846.4, "norm": 276.94},
    ),
    ("rtx-pro-6000", "16x32:8192", {"prepare_value": 373.8, "recurrence": 713.05, "norm": 180.63}),
    ("gb10", "16x32:1024", {"prepare_value": 326.55, "recurrence": 354.82, "norm": 142.17}),
    (
        "gb10",
        "16x48:512",
        {"prepare_key": 81.74, "prepare_value": 155.19, "recurrence": 185.03, "norm": 89.82},
    ),
    ("gb10", "16x32:4096", {"prepare_value": 1225.8, "recurrence": 1206.55, "norm": 647.18}),
]


@pytest.mark.parametrize("device,label,measured", PROFILE_TIMES)
def test_default_kernel_bounds(device, label, measured):
    row = next(
        r
        for r in model.rows()
        if f"{r['hk']}x{r['hv']}:" + "+".join(map(str, r["lengths"])) == label
    )
    result = model.analyze(device, row)
    kernels = result["scenarios"]["l2_aware"]["implementation"]["per_kernel"]
    for kernel in kernels:
        if kernel["name"] == "flags_memset":
            continue
        assert kernel["t_ref_us"] <= measured[kernel["name"]], kernel["name"]


def test_large_call_keeps_rereads_and_local_precision_planes_in_l2():
    call, detail = model.build_call("rtx-pro-6000", 16, 48, [8192], [False])
    assert not detail["fits_l2"]
    key = detail["ledger"]["prepare_key"]
    rec = detail["ledger"]["recurrence"]
    assert key["intermediate_writes"]["kl"] > 0
    assert key["dram_intermediate_writes"]["kl"] == 0
    assert key["dram_intermediate_reads"]["q"] == 0
    # Q_hi/K_hi fit between key prepare and recurrence despite the large call.
    for buffer in ("q", "k"):
        assert rec["intermediate_reads"][buffer] > detail["allocations_bytes"][buffer]
        assert rec["dram_intermediate_reads"][buffer] == 0
    # Every external repeat remains in L2, independently of interface spilling.
    for kernel in call.kernels:
        for category, size in detail["ledger"][kernel.name]["repeated"].items():
            assert kernel.l2_resident_bytes.get(category, 0) == size


def test_serial_interface_spills_only_capacity_excess_in_both_directions():
    _, detail = model.build_call("rtx5090", 16, 48, [8192], [False])
    edge = next(
        e
        for e in detail["producer_consumer_interfaces"]
        if e["producer"] == "recurrence" and e["consumer"] == "norm"
    )
    assert not edge["streaming"] and not edge["fits_l2"]
    assert sum(edge["dram_bytes"].values()) == (
        sum(edge["unique_bytes"].values()) - detail["l2_capacity_bytes"]
    )
    for buffer, size in edge["dram_bytes"].items():
        assert detail["ledger"]["recurrence"]["dram_intermediate_writes"][buffer] == size
        assert detail["ledger"]["norm"]["dram_intermediate_reads"][buffer] == size


def test_pdl_window_streams_and_serial_stages_still_sum():
    _, detail = model.build_call("gb10", 16, 32, [4096], [False])
    edge = next(
        e
        for e in detail["producer_consumer_interfaces"]
        if e["producer"] == "prepare_value" and e["consumer"] == "recurrence"
    )
    assert sum(edge["unique_bytes"].values()) > detail["l2_capacity_bytes"]
    assert edge["streaming"] and edge["fits_l2"]
    assert sum(edge["dram_bytes"].values()) == 0
    assert edge["live_chunks"] == detail["tune"]["prefetch_depth"]
    for row in model.rows():
        result = model.analyze("rtx-pro-6000", row)
        if not result["tune"]["overlap"]:
            kernels = result["scenarios"]["l2_aware"]["implementation"]["per_kernel"]
            assert result["sol_us"] == sum(k["t_ref_us"] for k in kernels)


def test_dram_bandwidth_direction_follows_actual_transfers():
    call, _ = model.build_call("rtx5090", 16, 48, [16], [False])
    assert {k.name: k.dram_mode for k in call.kernels} == {
        "prepare_key": "read",
        "prepare_value": "read",
        "recurrence": "write",
        "norm": "copy",
    }


def test_profile_reader_checks_every_default_row(tmp_path):
    filenames = {"rtx5090": "rtx5090", "rtx-pro-6000": "rtx-pro-6000", "gb10": "gb10"}
    for device, filename in filenames.items():
        lines = ["non-JSON chatter", '{"incomplete":']
        for d, label, measured in PROFILE_TIMES:
            if d != device:
                continue
            record = {
                "provider": "ours",
                "tune": "",
                "row": label,
                "top": [(f"_saved_{k}_kernel", v) for k, v in measured.items()],
            }
            lines.extend(
                json.dumps(r)
                for r in (
                    record,
                    record | {"provider": "comparator", "row": "ignored"},
                    record | {"tune": "nondefault", "row": "ignored"},
                )
            )
        (tmp_path / f"{filename}.log").write_text("\n".join(lines))
    comparisons = model.profile_comparisons(tmp_path)
    assert len(comparisons) == sum(len(measured) for _, _, measured in PROFILE_TIMES)
    assert all(r["ratio"] <= 1 for r in comparisons)
    with (tmp_path / "rtx5090.log").open("a") as stream:
        stream.write(
            "\n" + json.dumps({"provider": "ours", "tune": "", "row": "16x48:8192", "top": []})
        )
    with pytest.raises(ValueError, match="exactly one entry"):
        model.profile_comparisons(tmp_path)
