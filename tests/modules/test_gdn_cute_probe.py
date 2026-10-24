# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks of the precision probe's reference reuse and gate aggregation."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ..gpu import _gdn_reference as reference


def _probe():
    spec = importlib.util.spec_from_file_location(
        "cute_probe", Path(__file__).resolve().parents[2] / "scripts" / "cute_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed_index", [0, 1])
def test_reference_prefix_reuse_is_bitwise_exact(seed_index):
    probe = _probe()
    generator = torch.Generator().manual_seed(19)
    hk, hv, dc = 1, 2, 512
    weights = reference.Weights(
        torch.randn(dc, 4, generator=generator).bfloat16(),
        torch.randn(hv, generator=generator),
        torch.full((hv,), -8.0),
        torch.randn(128, generator=generator),
    )
    x = torch.randn(9, dc + hv * 128 + 2 * hv, generator=generator).bfloat16()
    base = (
        torch.randn(8, hv, 128, 128, generator=generator),
        torch.randn(8, 3, dc, generator=generator).bfloat16(),
    )
    y, states = probe.precision_prefix_reference(weights, x, hk, base, seed_index, [1, 4, 9])
    for length in [1, 4, 9]:
        exact = reference.prefill(
            x[:length],
            weights,
            hk,
            torch.tensor([0, length]),
            torch.tensor([7]),
            torch.tensor([bool(seed_index % 2)]),
            *base,
        )
        assert torch.equal(y[:length], exact[0])
        assert torch.equal(states[length], exact[1][[7]])


def test_precision_sweep_shares_baselines_and_averages_seed_maxima(monkeypatch):
    probe = _probe()
    assert ";".join(probe.PRECISION_TUNES) == ";aq_qk=0;aq_out=1;aq_qk=0,aq_out=0"
    timing = [probe.read_tune(value) for value in probe.OUTPUT_TUNES]
    assert {(c["aq_qk"], c["aq_out"]) for c in timing} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    configs = [probe.read_tune(value) for value in probe.PRECISION_TUNES]
    assert {(c["aq_qk"], c["aq_out"]) for c in configs} == {(1, 0), (0, 0), (1, 1)}
    assert all(c["o_form"] == "legacy" for c in configs)
    calls = {"reference": 0, "fla": 0, "cute": 0}
    constructed = []
    records = []

    def inputs(hk, hv, lengths, kind, seed_index):
        # Ragged input exercises the non-prefix reference fallback.
        x = torch.full((sum(lengths), 2), seed_index, dtype=torch.float64)
        weights = reference.Weights(torch.empty(1), torch.empty(1), torch.empty(1), torch.empty(1))
        base = (torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2))
        return weights, x, (), base, [p.clone() for p in base], [0]

    def exact(x, *args):
        calls["reference"] += 1
        return torch.zeros_like(x), torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2)

    def fla(geometry, x, *args):
        calls["fla"] += 1
        args[-2][0] = 2 * (x[0, 0] + 1)
        return torch.full_like(x, 2 * (x[0, 0] + 1))

    class Op:
        def selected_variant(self, *args):
            return "cute"

        def __call__(self, x, *args):
            calls["cute"] += 1
            # Each config must receive a fresh input pool.
            assert torch.count_nonzero(args[-2]) == 0
            args[-2][0] = x[0, 0] + 1
            return torch.full_like(x, x[0, 0] + 1)

    monkeypatch.setattr(probe, "_precision_inputs", inputs)
    monkeypatch.setattr(probe.reference, "prefill", exact)
    monkeypatch.setattr(probe, "_fla_prefill", fla)

    def construct(*args):
        constructed.append(probe.read_tune())
        return Op()

    monkeypatch.setattr(probe, "construct", construct)
    monkeypatch.setattr(probe, "emit", records.append)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "")
    probe.precision_sweep([(1, 2)], [[1, 2, 61]], ["ordinary", "slow"], 5, configs)
    assert calls == {"reference": 10, "fla": 10, "cute": 10 * len(configs)}
    results = [r for r in records if r["mode"] == "precision"]
    assert len(results) == 2 * len(configs)
    assert constructed == configs
    for result in results:
        assert result["cute"] == {"mean_y_max": 3, "mean_state_max": 3}
        assert result["fla-core"] == {"mean_y_max": 6, "mean_state_max": 6}
        assert result["ratio"] == {"y": 0.5, "state": 0.5}
        assert result["gate_pass"] and result["state_target_pass"]


def test_per_config_shape_groups_and_small_speed_defaults():
    probe = _probe()
    plan = probe.sweep_plan(["", "bv=32"], "16x48|16x32", "1024;2048|2048")
    assert plan == [("", [(16, 48)], [[1024], [2048]]), ("bv=32", [(16, 32)], [[2048]])]
    assert probe.sweep_plan([""], "16x48", None)[0][2] == [[1024], [2048]]
    assert len(probe.sweep_plan(["", "bv=32"], "16x48", "2048")) == 2


@pytest.mark.parametrize(
    "geoms,lengths",
    [("16x48|16x32|16x48", "2048"), ("16x48", "2048|1024|64"), ("2x4", "64"), ("16x48", "0")],
)
def test_invalid_shape_groups_fail_on_cpu(geoms, lengths):
    with pytest.raises(ValueError):
        _probe().sweep_plan(["", "bv=32"], geoms, lengths)
