# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Runner-only fused-v2 parity and five-seed FP64 gates; full sweeps are exhaustive."""

from __future__ import annotations

import statistics

import pytest
import torch

from ctikernels.gdn import _triton_prefill

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import ABSOLUTE_FLOOR, SLOTS, _max_error, _rms_error
from .test_gdn_cute_prefill import (
    _REFERENCE_CACHE,
    _check_disabled_capture_entries,
    _check_intermediate_captures,
    _inputs,
    _op,
    _poison,
    _pools,
    _precision_inputs,
    _precision_reference,
    _prefill_case,
    _reason,
    _same,
)

_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
GEOMETRIES = [(16, 32), (16, 48), (8, 16), (8, 24)]
LENGTHS = [[16], [64], [200], [1024], [2048], [32, 96, 200, 184], [6000, 1500, 500, 192]]


@pytest.fixture(autouse=True)
def _policy(monkeypatch):
    # v1/v2 schedules only; the sm120 port table would claim some rows (it has its own suite).
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    # Every bucket CuTe: the provider table's Triton rows are qualified by their own suites.
    monkeypatch.setattr(_triton_prefill.PROVIDER, "table", lambda device_class: {})
    yield
    # Large exhaustive rows must not retain whole output/reference sets on CPU.
    _REFERENCE_CACHE.clear()


def _pair(hk, hv, lengths, vsplit, monkeypatch):
    # The baseline is frozen v1 by explicit selection: an empty tune string would
    # enable the measured row tunes, which route some GB10 rows to v2 itself.
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "v2=0")
    v1 = _op(hk, hv, lengths)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"v2=1,vsplit={vsplit}")
    v2 = _op(hk, hv, lengths)
    tokens, count = sum(lengths), len(lengths)
    for op, expected in ((v1, 0), (v2, 1)):
        launcher = op._kernel._cute
        assert not launcher.row_tunes
        assert launcher._row_tune(tokens, count)[0].get("v2", 0) == expected
    return v1, v2


def _within_v1_error(name, v2, v1, exact):
    """Difference RMS within v1's FP64 error scale; max uses triangle bound."""
    assert torch.isfinite(v2).all(), name
    delta, scale = _rms_error(v2, v1), _rms_error(v1, exact)
    assert delta <= scale + ABSOLUTE_FLOOR, (name, "RMS parity", delta, scale)
    delta, scale = _max_error(v2, v1), _max_error(v1, exact)
    assert delta <= 2 * scale + ABSOLUTE_FLOOR, (name, "max parity", delta, scale)


@pytest.mark.parametrize(
    "hk,hv,vsplit,length_index,initial",
    ci_cases(
        GEOMETRIES,
        [1, 2, 4],
        range(len(LENGTHS)),
        [False, True],
        core=[
            (16, 32, 1, 0, False),
            (16, 48, 2, 1, True),
            (8, 16, 4, 2, True),
            (8, 24, 1, 5, False),
        ],
    ),
)
def test_parity(hk, hv, vsplit, length_index, initial, monkeypatch):
    lengths = LENGTHS[length_index]
    v1, v2 = _pair(hk, hv, lengths, vsplit, monkeypatch)
    slots = [7] if len(lengths) == 1 else [9, 2, -1, 5]
    meta = _prefill_case(lengths, slots, [initial] * len(slots))
    weights, x = _inputs(hk, hv, lengths, "ordinary", 4200)
    base = _pools(hk, hv, seed=4202)
    read = {s for s in slots if s >= 0 and initial}
    poisoned = [_poison(p, read, "nan", 4203) for p in base]
    p1, p2 = ([p.clone() for p in poisoned] for _ in range(2))
    written = sorted(s for s in slots if s >= 0)
    exact = _precision_reference(
        hk, hv, lengths, f"v2-parity-{initial}", 0, x, weights, meta, base, written
    )
    with torch.inference_mode():
        y1 = v1(x, *weights.args(), *meta, *p1)
        y2 = v2(x, *weights.args(), *meta, *p2)
    torch.cuda.synchronize()
    _within_v1_error("y", y2, y1, exact[0])
    _within_v1_error("state", p2[0][written], p1[0][written], exact[1])
    assert _same(p2[1][written], p1[1][written])
    assert _same(p2[1][written], exact[2])
    untouched = [s for s in range(SLOTS) if s not in written]
    for pool, old in zip(p2, poisoned, strict=True):
        assert _same(pool[untouched], old[untouched])


@pytest.mark.parametrize(
    "hk,hv,vsplit,length_index,kind",
    ci_cases(
        GEOMETRIES,
        [1, 2, 4],
        range(len(LENGTHS)),
        ["ordinary", "heavy"],
        core=[
            (16, 32, 1, 2, "ordinary"),
            (16, 48, 2, 2, "ordinary"),
            (8, 24, 4, 0, "heavy"),
        ],
    ),
)
def test_precision_against_fp64(hk, hv, vsplit, length_index, kind, monkeypatch):
    lengths = LENGTHS[length_index]
    v1, v2 = _pair(hk, hv, lengths, vsplit, monkeypatch)
    # Per seed: (v2 RMS, v1 RMS, v2 max-abs, v1 max-abs) against FP64.
    errors = {"y": [], "state": []}
    for seed in range(5):
        weights, x, meta, base, pools, written = _precision_inputs(hk, hv, lengths, kind, seed)
        exact = _precision_reference(hk, hv, lengths, kind, seed, x, weights, meta, base, written)
        p1, p2 = ([p.clone() for p in pools] for _ in range(2))
        with torch.inference_mode():
            y1 = v1(x, *weights.args(), *meta, *p1)
            y2 = v2(x, *weights.args(), *meta, *p2)
        torch.cuda.synchronize()
        assert torch.isfinite(y2).all()
        for name, ours, baseline, reference in (
            ("y", y2, y1, exact[0]),
            ("state", p2[0][written], p1[0][written], exact[1]),
        ):
            errors[name].append(
                (
                    _rms_error(ours, reference),
                    _rms_error(baseline, reference),
                    _max_error(ours, reference),
                    _max_error(baseline, reference),
                )
            )
        assert _same(p2[1][written], exact[2])
    _seed_spread_gate(errors)


def _seed_spread_gate(errors):
    """v2 no worse than v1 CuTe against FP64, by the seed-spread rule of the fla-core
    capture exports (commit 1a4929d): for RMS and max-abs separately, with r_s the
    per-seed ratio v2/v1, mean(v2) <= mean(v1) * (1 + 2 * stdev(r_s) / sqrt(5)). The
    margin is two standard errors of the measured spread and is zero where v2 is bitwise
    equal to v1. A per-seed strict bound rejects pure FP32 reassociation noise: on RTX
    PRO 6000 (8x16, [1024] and [2048]) v2's y RMS was above v1's by 1e-10 relative."""
    for name, samples in errors.items():
        for metric in (0, 2):
            ours = [e[metric] for e in samples]
            baseline = [e[metric + 1] for e in samples]
            ratios = [o / max(b, ABSOLUTE_FLOOR) for o, b in zip(ours, baseline, strict=True)]
            margin = 2 * statistics.stdev(ratios) / len(ratios) ** 0.5
            assert sum(ours) <= sum(baseline) * (1 + margin) + ABSOLUTE_FLOOR, (
                name,
                metric,
                samples,
            )


@pytest.mark.parametrize(
    "case,initial,hk,hv,vsplit",
    ci_cases(
        ["serial-varlen", "long-varlen", "long-single"],
        [False, True],
        GEOMETRIES,
        [1, 2, 4],
        core=[
            ("long-single", True, 16, 48, 1),
            ("serial-varlen", False, 16, 32, 2),
            ("long-varlen", True, 8, 24, 4),
        ],
    ),
)
def test_v2_captures_match_independent_truncated_runs(hk, hv, initial, case, vsplit, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"v2=1,vsplit={vsplit}")
    _check_intermediate_captures(hk, hv, initial, case)


@pytest.mark.parametrize(
    "hk,hv,vsplit", ci_cases(GEOMETRIES, [1, 2, 4], core=[(16, 32, 1), (8, 24, 4)])
)
def test_v2_disabled_capture_entries_write_nothing(hk, hv, vsplit, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"v2=1,vsplit={vsplit}")
    _check_disabled_capture_entries(hk, hv, [200, 63, 1, 0])


@pytest.mark.parametrize(
    "hk,hv,vsplit,length_index,initial",
    ci_cases(
        GEOMETRIES,
        [1, 2, 4],
        range(len(LENGTHS)),
        [False, True],
        core=[
            (8, 24, 2, 3, True),
            (8, 16, 1, 6, False),
            (16, 48, 4, 5, True),
            (16, 32, 2, 0, False),
        ],
    ),
)
def test_fused_norm_bitwise_matches_separate_norm(
    hk, hv, vsplit, length_index, initial, monkeypatch
):
    """The epilogue norm (cluster DSMEM sums at vsplit > 1) keeps the norm kernel's arithmetic."""
    lengths = LENGTHS[length_index]
    ops = []
    for snorm in (1, 0):
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"v2=1,vsplit={vsplit},v2_snorm={snorm}")
        ops.append(_op(hk, hv, lengths))
    slots = [7] if len(lengths) == 1 else [9, 2, -1, 5]
    meta = _prefill_case(lengths, slots, [initial] * len(slots))
    weights, x = _inputs(hk, hv, lengths, "ordinary", 4300)
    base = _pools(hk, hv, seed=4302)
    pools = [[p.clone() for p in base] for _ in ops]
    with torch.inference_mode():
        ys = [op(x, *weights.args(), *meta, *pool) for op, pool in zip(ops, pools, strict=True)]
    torch.cuda.synchronize()
    assert _same(ys[1], ys[0])
    for fused, separate in zip(pools[1], pools[0], strict=True):
        assert _same(fused, separate)
