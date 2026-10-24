# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only planning, cycle, fit and fake-tensor compilation checks."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load this standalone benchmark without importing the target package or torch.
SOURCE = Path(__file__).resolve().parents[2] / "benchmark/kernel_tuning/dflash_dependent_rt.py"
SPEC = importlib.util.spec_from_file_location("dflash_dependent_rt_test_target", SOURCE)
rt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rt
SPEC.loader.exec_module(rt)


def test_defaults():
    args = rt.parse_args([])
    assert args.modes == rt.MODES
    assert args.ks == rt.KS
    assert (args.ctas, args.repeat, args.launches) == (1, 25, 64)
    assert args.device is None
    assert args.timer == "wall"


def test_subsets_and_sorting():
    args = rt.parse_args(
        [
            "--modes",
            "fence,l2",
            "--ks",
            "8,0,2",
            "--ctas",
            "8",
            "--repeat",
            "3",
            "--launches",
            "16",
            "--device",
            "0",
            "--timer",
            "events",
        ]
    )
    assert args.modes == ("fence", "l2")
    assert args.ks == (0, 2, 8)
    assert (args.ctas, args.repeat, args.launches, args.device) == (8, 3, 16, 0)
    assert args.timer == "events"


@pytest.mark.parametrize(
    "argv",
    [
        ["--modes", ""],
        ["--modes", "unknown"],
        ["--modes", "l2,l2"],
        ["--ks", ""],
        ["--ks", "0"],
        ["--ks", "1,2"],
        ["--ks", "0,3"],
        ["--ks", "0,1,1"],
        ["--ks", "0,no"],
        ["--ctas", "0"],
        ["--ctas", "-1"],
        ["--repeat", "0"],
        ["--repeat", "no"],
        ["--launches", "0"],
        ["--device", "-1"],
        ["--seed", "-1"],
        ["--timer", "unknown"],
    ],
)
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit) as error:
        rt.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("mode", rt.MODES)
@pytest.mark.parametrize("ctas", [1, 17, 128])
def test_plans(mode, ctas):
    args = rt.parse_args(["--ctas", str(ctas)])
    l2 = 96 << 20
    plan = rt.make_plan(args, mode, 128, l2)
    assert plan.buffer_bytes == plan.lines_per_cta * ctas * rt.LINE_BYTES
    if mode == "dram":
        assert plan.buffer_bytes >= 2 * l2
        assert plan.lines_per_cta > plan.launches * max(plan.ks)
        assert plan.eviction_bytes == 4 * l2
    else:
        assert 0 < plan.buffer_bytes <= min(4 << 20, l2 // 4)
        assert plan.eviction_bytes == 0


def test_dram_small_l2_still_has_no_repeated_lines_in_one_replay():
    args = rt.parse_args(["--ctas", "64"])
    plan = rt.make_plan(args, "dram", 64, 1 << 20)
    assert plan.lines_per_cta == 64 * 32 + 1
    assert plan.buffer_bytes >= 2 << 20


@pytest.mark.parametrize(
    "mode,sms,l2",
    [("bad", 128, 1 << 20), ("l2", 0, 1 << 20), ("dram", 128, 0), ("dram", 128, 1 << 34)],
)
def test_invalid_plans(mode, sms, l2):
    with pytest.raises(ValueError):
        rt.make_plan(rt.parse_args([]), mode, sms, l2)


def test_ctas_bounded_by_sms():
    with pytest.raises(ValueError, match="SM count"):
        rt.make_plan(rt.parse_args(["--ctas", "9"]), "l2", 8, 1 << 20)


def test_fit_recovers_known_slope_and_intercept():
    fit = rt.fit_latency({k: 700 + 125 * k for k in rt.KS})
    assert fit["slope_ns_per_rt"] == pytest.approx(125)
    assert fit["intercept_ns"] == pytest.approx(700)
    assert fit["r_squared"] == pytest.approx(1)
    assert fit["fit_ks"] == list(rt.KS)
    assert max(abs(value) for value in fit["residual_ns"].values()) < 1e-9


def test_fit_does_not_hide_negative_or_flat_slopes():
    assert rt.fit_latency({0: 100, 1: 90})["slope_ns_per_rt"] == -10
    flat = rt.fit_latency({0: 100, 1: 100})
    assert flat["slope_ns_per_rt"] == 0
    assert flat["r_squared"] is None


@pytest.mark.parametrize("medians", [{0: 1}, {0: 1, 1: float("nan")}, {0: 0, 1: 2}])
def test_invalid_fits(medians):
    with pytest.raises(ValueError):
        rt.fit_latency(medians)


def test_affinity_parser_respects_allowed_cpus():
    cpuinfo = """processor : 0
CPU part : 0xd85

processor : 1
CPU part : 0xd05

processor : 2
CPU part : 0xD85

processor : 3
CPU part : malformed
"""
    assert rt.x925_cores(cpuinfo, {1, 2, 3}) == [2]
    assert rt.x925_cores("processor : 0\nmodel name : generic CPU", {0}) == []


def test_random_cycle_covers_all_nodes_and_is_reproducible():
    order = rt.cycle_order(97, 19)
    assert sorted(order) == list(range(97))
    assert order == rt.cycle_order(97, 19)
    assert order != rt.cycle_order(97, 20)
    successors = dict(zip(order, order[1:] + order[:1], strict=True))
    current = order[0]
    visited = set()
    for _ in order:
        assert current not in visited
        visited.add(current)
        current = successors[current]
    assert current == order[0]


def test_allocated_chains_on_cpu():
    pytest.importorskip("torch")
    args = rt.parse_args(["--ctas", "3"])
    plan = rt.make_plan(args, "l2", 3, 3 * 128 * 4 * 17)
    buf, out, scratch = rt.allocate_buffers(plan, "cpu", 7)
    assert not scratch.any()
    for cta in range(plan.ctas):
        start = int(out[cta * rt.LINE_WORDS])
        current = start
        seen = set()
        for _ in range(plan.lines_per_cta):
            assert current not in seen
            assert current % rt.LINE_WORDS == 0
            assert current // (rt.LINE_WORDS * plan.lines_per_cta) == cta
            seen.add(current)
            current = int(buf[current])
        assert current == start


@pytest.mark.parametrize("arch", rt.ARCHES)
@pytest.mark.parametrize("mode", rt.MODES)
@pytest.mark.parametrize("k", rt.KS)
def test_offline_compile(monkeypatch, arch, mode, k):
    # No CUDA context, tensors, device query or launch; failures are not skipped
    # when the DSL is installed. File caching is unnecessary for this smoke test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass")
    compiled = rt.compile_kernel(mode, k, 1, arch)
    assert callable(compiled)


@pytest.mark.parametrize("arch", rt.ARCHES)
def test_offline_compile_multi_cta(monkeypatch, arch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass")
    assert callable(rt.compile_kernel("fence", 32, 17, arch))
