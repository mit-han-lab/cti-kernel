# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired CI arithmetic without importing FlashInfer or initializing CUDA."""

import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[2] / "benchmark/dflash_sampling/seeded_bench.py"
    spec = importlib.util.spec_from_file_location("seeded_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_summary(bench):
    result = bench.paired_summary([2.0] * 20, [(1.0, 1.0)] * 20)
    assert result == {"gm": 2.0, "lo": 2.0, "hi": 2.0, "epsilon": 0.0, "claim": True}
    logs = [-0.1, 0.1] * 10
    result = bench.paired_summary([math.exp(x) for x in logs], [(1.0, 1.1)] * 20)
    margin = bench.T_CRITICAL * math.sqrt(0.2 / 19) / math.sqrt(20)
    assert result["gm"] == pytest.approx(1)
    assert result["lo"] == pytest.approx(math.exp(-margin))
    assert result["hi"] == pytest.approx(math.exp(margin))
    assert result["epsilon"] == pytest.approx(0.1)
    assert not result["claim"]
    assert not bench.paired_summary([1.05] * 20, [(1, 1.1)] * 20)["claim"]
    with pytest.raises(ValueError, match="twenty"):
        bench.paired_summary([1] * 19, [(1, 1)] * 20)
    with pytest.raises(ValueError, match="positive"):
        bench.paired_summary([0] * 20, [(1, 1)] * 20)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("min_p", ["softmax", "min_p_sampling_from_probs"]),
        (
            "all",
            ["softmax", "top_k_renorm_probs", "top_p_renorm_probs", "min_p_sampling_from_probs"],
        ),
    ],
)
def test_min_p_flashinfer_order(bench, mode, expected):
    from types import SimpleNamespace

    calls = []
    current = object()
    logits, temperatures, k, p, min_p = (object() for _ in range(5))

    def step(name, parameter):
        def call(source, actual):
            nonlocal current
            assert source is current and actual is parameter
            calls.append(name)
            current = object()
            return current

        return call

    fi = SimpleNamespace(
        **{
            "softmax": step("softmax", temperatures),
            "top_k_renorm_probs": step("top_k_renorm_probs", k),
            "top_p_renorm_probs": step("top_p_renorm_probs", p),
            "min_p_sampling_from_probs": step("min_p_sampling_from_probs", min_p),
        }
    )
    for call in bench.flashinfer_calls(fi, logits, temperatures, k, p, mode, min_p):
        current = logits
        calls.clear()
        assert call() is current
        assert calls == expected


def test_sol_reference_uses_shared_helper_with_residency(bench):
    from benchmark.sol.sol import load_ceilings

    table = load_ceilings("gb10")
    row = bench.sol_reference("gb10", 4, 248320, 5)
    assert row["sol_residency"] == "l2"
    assert row["ideal_bytes"] == 4 * 248320 * 10
    l2 = table["memory"]["l2_read_gbs"] * 1e3
    floor = table["fixed_costs"]["graph_node_pdl_us"]
    assert row["sol_us"] == pytest.approx(4 * 248320 * 6 / l2 + 4 * 248320 * 4 / l2 + 3 * floor)
    # A working set larger than L2 is charged to DRAM.
    big = bench.sol_reference("gb10", 64, 248320, 5)
    assert big["sol_residency"] == "dram"
    assert big["sol_us"] > 64 * 248320 * 10 / l2
