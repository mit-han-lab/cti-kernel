# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Fast CPU-only checks for the frozen v2 matrix and fused-kernel accounting."""

import math

import pytest

from benchmark.kernel_tuning import gdn_prefill_sol as v1
from benchmark.kernel_tuning import gdn_prefill_v2_matrix as matrix
from benchmark.kernel_tuning import gdn_prefill_v2_sol as model


def test_frozen_matrix():
    matrix.self_test()
    assert matrix.CASES == tuple((tuple(ls), tuple(flags)) for ls, flags in v1.SMALL + v1.LONG)
    assert {(r["hk"], r["hv"]) for r in matrix.rows()} == {(16, 32), (16, 48), (8, 16), (8, 24)}
    assert {shape for shape, g in matrix.GEOMETRIES.items() if g["optional"]} == {(4, 8), (4, 12)}
    first = matrix.rows()
    first[0]["lengths"][0] = 999
    assert matrix.rows()[0]["lengths"] == list(v1.SMALL[0][0])


def test_hand_computed_bytes():
    # 129 tokens, 2 sequences, 3 chunks, 1 initial slot. QK=2048, V=2048,
    # conv=4096 channels. At split=2, each QK column has four consumers.
    row = {"hk": 8, "hv": 16, "case": -1, "lengths": [64, 65], "initial": [True, False]}
    result = model.analyze("rtx5090", row, vsplit=2)
    assert result["ledger"]["dram_reads"] == {
        "projected_qk": 528384,
        "projected_v": 528384,
        "projected_z": 528384,
        "projected_ba": 8256,
        "initial_state": 1048576,
        "initial_history": 24576,
        "conv_weights": 32768,
        "gate_weights": 128,
        "norm_weight": 512,
        "metadata": 22,
    }
    assert result["ledger"]["dram_writes"] == {
        "final_state": 2097152,
        "final_history": 49152,
        "y": 528384,
    }
    assert result["dram_bytes"] == 5374678
    expected_repeats = {
        "projected_qk": 1585152,
        "projected_ba": 8256,
        "projected_v": 0,
        "projected_z": 0,
        "qk_halo": 49152,
        "v_halo": 12288,
        "initial_history": 36864,
        "conv_weights": 131072,
        "gate_weights": 384,
        "norm_weight": 15872,
        "metadata": 810,
    }
    assert result["ledger"]["l2_repeated"] == expected_repeats
    assert result["l2_bytes"] == sum(expected_repeats.values())
    # The fused boundary has the same unique external bytes as v1, without scratch.
    assert result["dram_bytes"] == v1.compulsory_formula(8, 16, [64, 65], [True, False])


@pytest.mark.parametrize("device", matrix.DEVICES)
@pytest.mark.parametrize("row", matrix.rows(), ids=lambda r: f"{r['hk']}x{r['hv']}-{r['case']}")
def test_all_rows_and_vsplit_tradeoff(device, row):
    result = model.analyze_row(device, row)
    variants = result["variants"]
    assert result["best"]["sol_us"] == min(r["sol_us"] for r in variants)
    assert result["best_vsplit"] == result["best"]["vsplit"]
    assert len({r["dram_bytes"] for r in variants}) == 1
    for r in variants:
        for key in (
            "t_dram",
            "t_l2",
            "t_compute",
            "t_chain",
            "t_chain_overlapped",
            "t_graph",
            "sol_us",
        ):
            assert math.isfinite(r[key]) and r[key] > 0
        assert r["sol_us"] == max(
            r[key] for key in ("t_dram", "t_l2", "t_compute", "t_chain", "t_graph")
        )
        chain = r["ledger"]["chain"]
        assert sum(sum(group.values()) for group in chain.values()) == r["chunk_cta_flops"]
        assert r["t_chain_overlapped"] / r["t_chain"] == pytest.approx(
            sum(chain["on_chain"].values()) / r["chunk_cta_flops"]
        )
        assert r["grid"] == len(row["lengths"]) * row["hv"] * r["vsplit"]
        assert 0 < r["wave_efficiency"] <= 1
        assert r["dram_bytes"] == v1.compulsory_formula(
            row["hk"], row["hv"], row["lengths"], row["initial"]
        )
    for left, right in zip(variants, variants[1:], strict=False):
        assert left["t_chain"] > right["t_chain"]
        assert left["l2_bytes"] < right["l2_bytes"]
        assert left["flops"] < right["flops"]  # Shared chunk math repeats per slice.


@pytest.mark.parametrize("hv", (8, 12, 16, 24, 32, 48))
def test_precision_planes_and_padded_chunk_work(hv):
    single = model.chunk_flops(hv, planes=dict.fromkeys(model.PLANES, 1))
    assert single == {
        "gram": 1048576,
        "qk": 1048576,
        "inverse": 524288,  # 2 FLOPs/MAC * 16 block products * 4 terms * 16^3.
        "w": 1048576,
        "u": 1048576,
        "wh": 2097152,
        "qh": 2097152,
        "ar": 1048576,
        "state_update": 2097152,
    }
    default = model.chunk_flops(hv)
    split = model.chunk_flops(hv, 4)
    precision = model.PLANES | {"qk": 3 if hv in (12, 24, 48) else 1}
    for product, count in single.items():
        assert default[product] == count * precision[product]
        assert split[product] == default[product] / (
            1 if product in ("gram", "qk", "inverse", "w") else 4
        )
    assert sum(default.values()) == (15204352 if hv in (12, 24, 48) else 13107200)
    assert model.chunk_flops(hv, planes={"inverse": 3})["inverse"] == single["inverse"] * 3
    assert model.chunk_flops(hv, planes={"qk": 2})["qk"] == 2097152
    row = matrix.rows()[0]
    short = model.analyze("gb10", row)
    padded = model.analyze("gb10", row | {"lengths": [64]})
    assert short["flops"] == padded["flops"]
    assert short["dram_bytes"] < padded["dram_bytes"]


def test_chain_and_wave_units():
    row = {"hk": 16, "hv": 48, "case": -1, "lengths": [64, 65], "initial": [True, False]}
    result = model.analyze("gb10", row)
    table = model.ceilings("gb10")
    peak = table["tensor_core"][model.MMA]["tflops_gpu"] * 1e6
    capacity = table["sm_count"]
    assert result["waves"] == math.ceil(96 / capacity)
    assert result["wave_efficiency"] == 96 / (result["waves"] * capacity)
    assert result["t_compute"] == pytest.approx(result["flops"] / peak / result["wave_efficiency"])
    assert result["t_chain"] == pytest.approx(2 * result["chunk_cta_flops"] / (peak / capacity))
    chain = result["ledger"]["chain"]
    assert chain["on_chain"] == {
        "wh": 2097152,
        "qh": 2097152,
        "ar": 1048576,
        "state_update": 2097152,
    }
    assert chain["prologue"] == {
        "gram": 1048576,
        "qk": 3145728,
        "inverse": 524288,
        "w": 1048576,
        "u": 2097152,
    }
    assert result["chunk_cta_flops"] == 15204352
    assert result["t_chain_overlapped"] == pytest.approx(2 * 7340032 / (peak / capacity))
    overridden = model.analyze("gb10", row, planes={"u": 3, "wh": 2})
    assert overridden["t_chain"] / result["t_chain"] == pytest.approx(35 / 29)
    assert overridden["t_chain_overlapped"] / result["t_chain_overlapped"] == pytest.approx(18 / 14)
    assert result["t_dram"] == pytest.approx(
        result["dram_bytes"] / (table["memory"]["dram_copy_gbs"] * 1000)
    )
    assert result["t_l2"] == pytest.approx(
        result["l2_bytes"] / (table["memory"]["l2_read_gbs"] * 1000)
    )
    higher_residency = model.analyze("gb10", row, ctas_per_sm=2)
    assert higher_residency["t_chain"] == result["t_chain"]
    assert higher_residency["waves"] == math.ceil(96 / (2 * capacity))


def test_short_history_and_invalid_inputs():
    ledger = model.byte_ledger(4, 8, [1], [False], 4)
    assert ledger["dram_reads"]["initial_history"] == 0
    assert ledger["dram_reads"]["initial_state"] == 0
    assert ledger["l2_repeated"]["qk_halo"] == 0
    assert ledger["dram_writes"]["final_history"] == 6 * (2 * 4 * 128 + 8 * 128)
    for lengths, initial in (([0], [False]), ([1], []), ([1], [1])):
        with pytest.raises(ValueError):
            model.byte_ledger(8, 16, lengths, initial)
    with pytest.raises(ValueError, match="vsplit"):
        model.chunk_flops(16, 3)
    with pytest.raises(ValueError, match="hv"):
        model.chunk_flops(17)
    with pytest.raises(ValueError, match="planes"):
        model.chunk_flops(16, planes={"gram": 0})
    with pytest.raises(ValueError, match="planes"):
        model.chunk_flops(16, planes={"unknown": 2})
