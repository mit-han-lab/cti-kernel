# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
import json

import pytest

from benchmark.kernel_tuning import gdn_k2_sol


def test_round_bytes_follow_q_and_accepted():
    verify, commit = gdn_k2_sol.round_bytes(1, 16, 48, 64, 32)
    width = (2 * 16 + 48) * 128 + 2 * 48
    proj = width + 48 * 128
    assert verify["activations"] == 2 * 64 * proj
    assert verify["intermediates"] == 2 * 64 * width + 64
    assert commit["intermediates"] == 2 * 32 * width + 64
    assert commit["kv_state"] == 2 * verify["kv_state"]


@pytest.mark.parametrize("device", ["rtx5090", "rtx-pro-6000", "gb10"])
def test_row_sol_regimes(device):
    s = gdn_k2_sol.row_sol(device, 1, 16, 32, 33, 16, 50.0)
    assert s["kernels_per_round"] == 2
    assert 0 < s["warm"]["t_ref_us"] <= s["cold"]["t_ref_us"]
    assert s["cold"]["SOL"] == pytest.approx(s["cold"]["t_ref_us"] / 50.0)
    # A footprint beyond L2 falls back to the DRAM model in the warm regime too.
    big = gdn_k2_sol.row_sol(device, 64, 16, 48, 64, 32, 50.0)
    assert not big["warm"]["l2_resident"]
    assert big["warm"]["t_ref_us"] == big["cold"]["t_ref_us"]


def test_main_reads_receipts(tmp_path, capsys):
    row = {
        "device": "NVIDIA GeForce RTX 5090",
        "shape": {"hk": 16, "hv": 32},
        "q": 64,
        "b": 2,
        "accepted_per_row": 32,
        "median_us": {"ours": 60.0},
    }
    path = tmp_path / "v.txt"
    path.write_text("noise\n" + json.dumps(row) + "\n")
    gdn_k2_sol.main([str(path)])
    out = capsys.readouterr().out.splitlines()
    assert json.loads(out[0])["source"] == "benchmark/sol/PROVENANCE.md"
    assert out[-1].startswith("| NVIDIA GeForce RTX 5090 | (16,32) | 64 | 2 | 60.0 |")
