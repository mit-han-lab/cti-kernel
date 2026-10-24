# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks for the port's acceptance inputs and output-dtype ULP gate."""

import json
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.gdn_prefill_port import _ulp_cases as ulp


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype,bits", [(torch.bfloat16, 8), (torch.float32, 24)])
def test_spacing_at_reference_including_binade_edges(dtype, bits):
    minimum = 2.0 ** (-125 - bits)
    values = torch.tensor(
        [
            0,
            -0.0,
            minimum,
            -minimum,
            minimum / 2,
            2.0**-126,
            0.5,
            1 - 2.0**-30,
            1,
            -1,
            1.5,
            2,
            -4,
            2.0**127,
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [minimum] * 6
        + [
            2.0**-bits,
            2.0**-bits,
            2.0 ** (1 - bits),
            2.0 ** (1 - bits),
            2.0 ** (1 - bits),
            2.0 ** (2 - bits),
            2.0 ** (3 - bits),
            2.0 ** (128 - bits),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(ulp.ulp_spacing(values, dtype), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,bits", [(torch.bfloat16, 8), (torch.float32, 24)])
def test_raw_ulp_counts_are_strict_and_use_unrounded_fp64_reference(dtype, bits):
    spacing = 2.0 ** (1 - bits)
    exact = torch.ones(6, dtype=torch.float64)
    got = (exact + torch.tensor([0, 1, 4, 5, 6, 0]) * spacing).to(dtype)
    metrics = ulp.ulp_metrics(got, exact)
    assert metrics == {"max_ulp": 6.0, "gt4": 2, "finite": True, "elements": 6}
    # Rounding the oracle first would erase this quarter-ULP error.
    fractional = ulp.ulp_metrics(torch.tensor([1], dtype=dtype), exact[:1] + spacing / 4)
    assert fractional["max_ulp"] == 0.25
    # Just below a binade boundary uses the reference's smaller spacing.
    boundary = ulp.ulp_metrics(torch.tensor([1], dtype=dtype), exact[:1] - spacing / 4)
    assert boundary["max_ulp"] == 0.5
    tiny = torch.tensor([0, 0], dtype=torch.float64)
    got_tiny = torch.tensor([4, 5], dtype=torch.float64).mul(2.0 ** (-125 - bits)).to(dtype)
    assert ulp.ulp_metrics(got_tiny, tiny)["gt4"] == 1


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_outputs_or_references_fail(invalid):
    one = torch.tensor([1], dtype=torch.float32)
    good = ulp.ulp_metrics(one, one.double())
    for actual, exact in ((one * invalid, one.double()), (one, one.double() * invalid)):
        bad = ulp.ulp_metrics(actual, exact)
        assert bad["max_ulp"] == math.inf
        assert bad["gt4"] == 1
        assert not bad["finite"]
        assert not ulp.ulp_pass(bad, good)
        assert not ulp.ulp_pass(bad, bad)
        assert not ulp.ulp_pass(good, bad)


def test_paired_rule_requires_both_maximum_and_tail_and_handles_empty_views():
    fla = {"max_ulp": 10, "gt4": 4, "finite": True}
    assert ulp.ulp_pass(fla, fla)
    assert not ulp.ulp_pass({**fla, "max_ulp": 11, "gt4": 0}, fla)
    assert not ulp.ulp_pass({**fla, "max_ulp": 9, "gt4": 5}, fla)
    empty = ulp.ulp_metrics(torch.empty(0, dtype=torch.float32), torch.empty(0).double())
    assert empty == {"max_ulp": 0.0, "gt4": 0, "finite": True, "elements": 0}
    assert ulp.ulp_pass(empty, empty)
    with pytest.raises(ValueError, match="shapes"):
        ulp.ulp_metrics(torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError, match="BF16 or FP32"):
        ulp.ulp_metrics(torch.empty(0, dtype=torch.float16), torch.empty(0))


def test_owner_layouts_and_exhaustive_regimes():
    cases = ulp.acceptance_cases()
    assert {(c.hk, c.hv, sum(c.lengths), len(c.lengths)) for c in cases} == {
        (8, 24, 11, 1),
        (8, 24, 31, 4),
        (8, 24, 1280, 5),
        (8, 16, 274, 5),
        (8, 16, 10654, 84),
        (16, 32, 657, 27),
        (16, 32, 3598, 29),
    }
    assert len(cases) == 21
    for case in cases:
        assert min(case.lengths) > 0
        for regime in ("normal", "heavy", "robust"):
            assert replace(case, kind=regime) in cases
    geoms = ((8, 16), (8, 24), (16, 32), (16, 48))
    exhaustive = ulp.exhaustive_cases(geoms, ((16,), (64,), (1, 2, 61)))
    assert len(exhaustive) == len(set(exhaustive))
    assert set(cases) <= set(exhaustive)
    for hk, hv in geoms:
        assert {c.kind for c in exhaustive if (c.hk, c.hv) == (hk, hv)} == set(ulp.REGIMES)
        assert any(
            c.hk == hk
            and c.hv == hv
            and c.kind == "normal"
            and sum(c.lengths) <= 16
            and len(c.lengths) >= 7
            for c in exhaustive
        )
    assert all(c.hk == 8 and c.hv == 24 for c in ulp.exhaustive_cases(((8, 24),), ((64,),)))


@pytest.mark.parametrize("case", [c for c in ulp.acceptance_cases() if c.kind == "normal"])
def test_acceptance_input_shapes(case):
    args = ulp.make_inputs(case, ulp.SEEDS[0])
    x, conv, A, dt, norm, cu, slots, initial, state, history = args
    n, t, hv = len(case.lengths), sum(case.lengths), case.hv
    dc = (2 * case.hk + hv) * 128
    assert x.shape == (t, dc + hv * 128 + 2 * hv)
    assert conv.shape == (dc, 4)
    assert A.shape == dt.shape == (hv,)
    assert norm.shape == (128,)
    assert torch.diff(cu).tolist() == list(case.lengths)
    assert cu.dtype == slots.dtype == torch.int32
    assert initial.shape == slots.shape == (n,)
    assert initial.dtype == torch.bool
    assert state.shape == (n + 8, hv, 128, 128)
    assert history.shape == (n + 8, 3, dc)
    assert state.dtype == A.dtype == dt.dtype == norm.dtype == torch.float32
    assert x.dtype == conv.dtype == history.dtype == torch.bfloat16
    assert all(t.is_contiguous() and t.device.type == "cpu" for t in args)
    owned = slots[slots >= 0]
    assert owned.unique().numel() == owned.numel()
    assert bool(((slots >= -1) & (slots < n + 8)).all())


@pytest.mark.parametrize("kind", ulp.REGIMES)
@pytest.mark.parametrize("hk,hv", [(8, 16), (8, 24), (16, 32), (16, 48)])
def test_regimes_are_deterministic_and_preserve_metadata_and_poison(kind, hk, hv):
    case = ulp.ULPCase(hk, hv, kind, (2, 1, 1, 1, 1, 1, 1))
    args = ulp.make_inputs(case, ulp.SEEDS[0])
    for actual, repeated in zip(args, ulp.make_inputs(case, ulp.SEEDS[0]), strict=True):
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0, equal_nan=True)
    x, _, _, _, norm, _, slots, initial, state, history = args
    b_start = (2 * hk + 2 * hv) * 128
    assert x[:, b_start : b_start + hv].abs().max() <= 9
    assert torch.count_nonzero(x[:, b_start + hv :]) < x.shape[0] * hv  # constant heads
    fresh = slots[(slots >= 0) & ~initial].long()
    assert bool(torch.isnan(state[fresh]).all())
    assert bool(torch.isnan(history[fresh]).all())
    if kind == "heavy":
        assert bool(initial.all()) and bool((slots >= 0).all())
        assert bool((x[0, : b_start - hv * 128] == 300).any())
    if kind in ("adversarial", "robust"):
        torch.testing.assert_close(norm[:3], torch.tensor([0.0, 1e-3, 4.0]))
        untouched = sorted(set(range(state.shape[0])) - set(slots.tolist()))
        assert bool(torch.isnan(state[untouched]).all())


def test_json_rows_and_summary_gate_every_seed(monkeypatch, capsys):
    from benchmark.kernel_tuning.gdn_prefill_port import cp2_precision as probe
    from ctikernels.gdn.chunk_ws.prefill import PortOptions

    case = ulp.acceptance_cases()[0]
    good = {"max_ulp": 2.0, "gt4": 0, "finite": True, "elements": 1, "pass": True}

    def measure(ops, case, seed, device):
        result = {name: {q: dict(good) for q in ("y", "state")} for name in ("fla", "port", "v1")}
        if seed == ulp.SEEDS[-1]:
            result["port"]["state"].update(max_ulp=math.inf, gt4=1, finite=False, **{"pass": False})
        return {"seed": seed, **result}

    monkeypatch.setattr(probe, "measure_ulp_seed", measure)
    ops = {"port": SimpleNamespace(port_options=PortOptions()), "v1": None}
    assert not probe.run_ulp_cases([case], lambda *args: ops, "cpu")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("ULP_ROW ") and lines[1].startswith("ULP_SUMMARY ")
    row, summary = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert len(row["seeds"]) == 5
    assert row["seeds"][-1]["port"]["state"]["max_ulp"] == "inf"
    assert row["gate"] == {"port": {"y": True, "state": False}, "v1": {"y": True, "state": True}}
    assert not row["pass"] and not summary["pass"]
    assert summary["failed"] == 1
    assert summary["failures"][0]["seed"] == ulp.SEEDS[-1]
    assert summary["failures"][0]["quantities"] == ["state"]


def test_measure_seed_compares_all_y_and_only_owned_final_states(monkeypatch):
    from benchmark.kernel_tuning.gdn_prefill_port import cp2_precision as probe

    from ..gpu import _gdn_reference as reference

    def exact_rounded(hk, *args):
        x, conv, A, dt, norm, cu, slots, initial, state, history = args
        y, final, conv_final = reference.prefill(
            x, reference.Weights(conv, A, dt, norm), hk, cu, slots, initial, state, history
        )
        state.copy_(final)
        history.copy_(conv_final)
        return y.bfloat16()

    monkeypatch.setattr(
        probe, "_fla_prefill", lambda geom, *args: exact_rounded(geom.k_heads, *args)
    )
    case = ulp.ULPCase(8, 16, "normal", (1, 1))
    ops = dict.fromkeys(("port", "v1"), lambda *args: exact_rounded(case.hk, *args))
    result = probe.measure_ulp_seed(ops, case, ulp.SEEDS[0], "cpu")
    for provider in ops:
        assert result[provider]["y"]["elements"] == 2 * 16 * 128
        for quantity in ("y", "state"):
            assert result[provider][quantity]["max_ulp"] <= 0.5
            assert result[provider][quantity]["gt4"] == 0
            assert result[provider][quantity]["pass"]


@pytest.mark.parametrize("mode", ["--ulp", "--all"])
def test_cli_routes_ulp_failures_and_preserves_port_options(monkeypatch, capsys, mode):
    from benchmark.kernel_tuning.gdn_prefill_port import cp2_precision as probe
    from ctikernels.gdn.chunk_ws.prefill import PortOptions

    monkeypatch.setattr(
        sys,
        "argv",
        ["cp2_precision.py", mode, "--geoms", "8x24", "--port-options", "prefetch=4"],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for key in ("CTI_GDN_PREFILL_CUTE", "CTI_GDN_CUTE_TUNE", "CTI_GDN_PREFILL_PORT"):
        monkeypatch.setenv(key, "")  # Restore main's environment mutations after the test.
    monkeypatch.setattr(probe, "LENGTHS", [[16]])
    monkeypatch.setattr(probe, "KINDS", ("ordinary",))
    monkeypatch.setattr(
        probe,
        "GatedDeltaNetPrefill",
        lambda **kwargs: SimpleNamespace(provider="cute", selected_variant=lambda *args: "cute"),
    )
    monkeypatch.setattr(
        probe,
        "PortPrefill",
        lambda num_k_heads, num_v_heads, device, **kwargs: SimpleNamespace(
            port_options=PortOptions(**kwargs)
        ),
    )
    selected, frozen_seeds = [], []

    def ulp_run(cases, get_ops, device):
        selected.extend(cases)
        first = cases[0]
        options = get_ops(first.hk, first.hv, first.lengths)["port"].port_options
        assert options.prefetch == 4
        return False

    def frozen_run(ops, case, seed):
        frozen_seeds.append(seed)
        metric = {"rms_abs": 1.0, "max_abs": 1.0, "finite": True, "rms_pass": True}
        result = {q: {"ours": metric, "fla": metric} for q in ("y", "state")}
        result.update(
            history_pass=True,
            untouched_pass={"state": True, "history": True},
            excluded_slot_minus_one_rows=0,
        )
        return dict.fromkeys(("port", "v1"), result)

    monkeypatch.setattr(probe, "run_ulp_cases", ulp_run)
    monkeypatch.setattr(probe, "measure_seed", frozen_run)
    assert probe.main() == 1
    assert selected and all((c.hk, c.hv) == (8, 24) for c in selected)
    if mode == "--all":
        assert frozen_seeds == list(range(5))
        assert {c.kind for c in selected} == set(ulp.REGIMES)
        assert "PREC_SUMMARY" in capsys.readouterr().out
    else:
        assert not frozen_seeds
        assert selected == [c for c in ulp.acceptance_cases() if (c.hk, c.hv) == (8, 24)]
