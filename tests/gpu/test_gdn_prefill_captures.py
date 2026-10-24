# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GDN prefill captures under the SGLang rule.

A call captures sequence n at p = (L // 64) * 64 into ``capture_rows[n]`` when the row is
>= 0 and p >= 64: the FP32 state after p tokens and the conv history at p. Checked on every
provider the device binds (port, CuTe v1 and v2, Triton), each against itself, bitwise:

- N1: captures change no y, slot state or slot history;
- N2: a capture equals what a [0, p) call leaves in its slot (N3: the slot itself at p == L);
- N4: the captured history is rows p-3..p-1 of ``projected``;
- N5: a [p, L) call resumed from the capture reproduces y[p:L] and the final state.

Rows that capture nothing (row -1, L < 64, ``capture_rows=None``) keep their NaN poison.
The port runs one long sequence with context parallelism (CP). A cold [0, p) call matches it
bitwise only under the same cp_plan, so other CP captures are held to the state tolerance
(not guaranteed); CP resume (N5, unverified) is a report test.
"""

from __future__ import annotations

import functools

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _cute_prefill, _triton_prefill, inference
from ctikernels.gdn.chunk_ws._cp import cp_plan

from .test_gdn import (
    STATE_TOLERANCE,
    _assert_close,
    _capture,
    _pools,
    _prefill_case,
    _projected,
    _same,
    _weights,
    _widths,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")

GRID = 64  # GatedDeltaNetPrefill.CAPTURE_GRID (test_capture_grid)
GEOMETRIES = [(16, 32), (16, 48)]
PROVIDERS = ["port", "v1", "v2", "triton"]
# Read at construction. An explicit CuTe tune disables the measured row tunes, which route
# GB10 rows to v2 (and sm120 one-sequence rows to o_split), so one variant serves every T.
ENVIRONMENT = {
    "port": {"CTI_GDN_PREFILL_CUTE": None, "CTI_GDN_PREFILL_PORT": "1", "CTI_GDN_CUTE_TUNE": None},
    "v1": {"CTI_GDN_PREFILL_CUTE": None, "CTI_GDN_PREFILL_PORT": "0", "CTI_GDN_CUTE_TUNE": "v2=0"},
    "v2": {"CTI_GDN_PREFILL_CUTE": None, "CTI_GDN_PREFILL_PORT": "0", "CTI_GDN_CUTE_TUNE": "v2=1"},
    "triton": {},
}
VARIANT = {"port": "port", "v1": "cute", "v2": "cute", "triton": "triton"}
LENGTHS = [1, 63, 64, 65, 128, 200, 1024, 1025, 1100, 4096]
# One-sequence port CP on 170 and 188 SMs: 3000's [0, 2944) prefix has the same plan
# (mid-segment capture, N2 bitwise); 6144 ends on the grid (N3).
CP_LENGTHS = [3000, 6144]
# 300 captures at its last segment's start (local p = 0) with a non-CP prefix; 1025's
# prefix plan differs at Hv 32.
REPORT_LENGTHS = [300, 1025, 3000]
POOL = 20  # slots 1-8, capture rows 9-19
# lengths, slots, has_initial_state, capture_rows: empty, stateless (slot -1), continued and
# fresh sequences; rows that skip (-1), see L < 64 or L = 0, and capture at p == L.
PACKED = {
    "mixed": (
        [513, 0, 64, 1000, 63],
        [3, 4, -1, 5, 6],
        [True, False, False, False, True],
        [9, 10, 11, 12, 13],
    ),
    "skips": (
        [1000, 65, 2, 513, 64, 0],
        [-1, 6, 7, 8, 2, 1],
        [False, True, True, True, False, True],
        [14, -1, 15, 16, -1, 17],
    ),
}
# One graph (T = 512, N = 4) replayed with new lengths, slots, flags and rows.
GRAPH_CASES = [
    ([200, 100, 64, 148], [1, 2, -1, 3], [True, False, False, True], [9, 10, 11, -1]),
    ([64, 1, 383, 64], [4, 5, 6, -1], [False, True, True, False], [-1, 12, 13, 14]),
    ([512, 0, 0, 0], [7, 8, 1, -1], [True, False, True, False], [15, 16, 17, 18]),
    ([128, 128, 128, 128], [1, 2, 3, 4], [False] * 4, [-1, -1, -1, -1]),
]


@functools.cache
def _unbound(provider: str) -> str | None:
    """Why ``provider`` is not bound on this device, or None."""
    device = torch.device("cuda")
    if provider == "triton":
        if torch.cuda.get_device_capability(device) < (8, 9):
            return "the Triton prefill needs compute capability 8.9 or newer"
        return None
    kind = _triton_prefill.device_type(device)
    if kind is None or not _cute_prefill.available():
        return "CuTe prefill serves RTX 5090, RTX PRO 6000 and GB10 with nvidia-cutlass-dsl 4.8.0"
    if provider == "port" and kind not in (_triton_prefill.RTX_5090, _triton_prefill.RTX_PRO_6000):
        return "the port binds on sm120 RTX 5090 and RTX PRO 6000 only"
    return None


@pytest.fixture(scope="module")
def ops():
    """One operation per (provider, geometry), shared so each compiles once."""
    cache = {}

    def get(provider, hk, hv):
        reason = _unbound(provider)
        if reason is not None:
            pytest.skip(reason)
        if (provider, hk, hv) not in cache:
            with pytest.MonkeyPatch.context() as patch:
                for name, value in ENVIRONMENT[provider].items():
                    if value is None:
                        patch.delenv(name, raising=False)
                    else:
                        patch.setenv(name, value)
                if provider == "triton":
                    from ctikernels.gdn.triton.prefill import Prefill

                    patch.setattr(
                        inference,
                        "_prefill_kernel",
                        lambda geometry, device, **options: Prefill(device, geometry),
                    )
                cache[provider, hk, hv] = GatedDeltaNetPrefill(
                    num_k_heads=hk, num_v_heads=hv, device="cuda"
                )
        return cache[provider, hk, hv]

    yield get
    _inputs.cache_clear()
    _base.cache_clear()


@functools.cache
def _inputs(hk, hv, tokens):
    """Weights with slow decay (the state accumulates over every chunk) and ``projected``."""
    seed = 7000 + tokens
    return _weights(hk, hv, seed, dt_bias=(-8.0, -6.0)), _projected(tokens, hk, hv, seed + 1)


@functools.cache
def _base(hk, hv):
    return _pools(hk, hv, seed=7001, slots=POOL)


def _poisoned(hk, hv, slots, initial, rows):
    """Random pools, NaN in fresh slots (never read) and in every capture row."""
    pools = [pool.clone() for pool in _base(hk, hv)]
    for row in {*rows, *(s for s, i in zip(slots, initial, strict=True) if not i)} - {-1}:
        for pool in pools:
            pool[row] = float("nan")
    return pools


def _plan(hv, tokens):
    """The port's one-sequence (segments, segment length)."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return cp_plan(tokens, hv, sms)


def _rows(rows):
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _check(provider, op, hk, hv, lengths, slots, initial, rows, *, report=None):
    """N1-N5 for one call. Under port CP, N2 is bitwise when the [0, p) call has the same
    plan (else within STATE_TOLERANCE) and N5 is skipped; ``report`` "n2" prints that
    tolerance's measured differences and "n5" runs N5 there too."""
    count, tokens = len(lengths), sum(lengths)
    assert op.selected_variant(tokens, count) == VARIANT[provider]
    weights, x = _inputs(hk, hv, tokens)
    begins = [sum(lengths[:n]) for n in range(count)]
    captures = {
        n: (rows[n], lengths[n] // GRID * GRID)
        for n in range(count)
        if rows[n] >= 0 and lengths[n] >= GRID
    }
    base = _poisoned(hk, hv, slots, initial, rows)
    plain, captured = [[pool.clone() for pool in base] for _ in range(2)]
    call = (x, *weights.args(), *_prefill_case(lengths, slots, initial))
    with torch.inference_mode():
        y = op(*call, *plain, capture_rows=None)
        y_captured = op(*call, *captured, capture_rows=_rows(rows))
    # N1; nothing outside the slots and the capturing rows is written.
    assert _same(y_captured, y)
    slotted, written = {s for s in slots if s >= 0}, {row for row, _ in captures.values()}
    for old, new, start in zip(plain, captured, base, strict=True):
        for row in range(POOL):
            if row not in slotted:
                assert _same(old[row], start[row]), ("capture_rows=None wrote", row)
            if row not in written:
                assert _same(new[row], old[row]), ("N1 or a stray capture write", row)
    if not captures:
        return
    cp = provider == "port" and count == 1 and _plan(hv, tokens)[0] > 1
    # N2: the same call truncated to [0, p), each capturing sequence's slot at its capture row.
    prefix = [captures[n][1] if n in captures else lengths[n] for n in range(count)]
    prefix_slots = [captures[n][0] if n in captures else slots[n] for n in range(count)]
    prefix_initial = [initial[n] and slots[n] >= 0 for n in range(count)]
    cold = [pool.clone() for pool in base]
    for n, (row, _) in captures.items():
        if prefix_initial[n]:
            for pool in cold:
                pool[row] = pool[slots[n]]
    prefix_x = torch.cat([x[b : b + length] for b, length in zip(begins, prefix, strict=True)])
    with torch.inference_mode():
        op(prefix_x, *weights.args(), *_prefill_case(prefix, prefix_slots, prefix_initial), *cold)
    conv_dim = _widths(hk, hv)[0]
    for n, (row, p) in captures.items():
        state, history = captured[0][row], captured[1][row]
        context = (provider, lengths[n], p, n)
        if not cp or _plan(hv, p) == _plan(hv, lengths[n]):
            assert _same(state, cold[0][row]), ("N2", *context)
        else:
            if report == "n2":
                diff = (state.double() - cold[0][row].double()).abs().max().item()
                peak = cold[0][row].double().abs().max().item()
                print(f"N2 CP vs cold {context}: max abs {diff:.3e}, max rel {diff / peak:.3e}")
            _assert_close(state, cold[0][row], **STATE_TOLERANCE)
        assert _same(history, cold[1][row]), ("N2 history", *context)
        assert _same(history, x[begins[n] + p - 3 : begins[n] + p, :conv_dim]), ("N4", *context)
        if p == lengths[n] and slots[n] >= 0:
            assert _same(state, captured[0][slots[n]]), ("N3", *context)
            assert _same(history, captured[1][slots[n]]), ("N3 history", *context)
    if cp and report != "n5":
        return
    # N5: resume each capture in its row and run the rest of its sequence.
    tails = [n for n, (_, p) in captures.items() if p < lengths[n]]
    if not tails:
        return
    resumed = [pool.clone() for pool in captured]
    spans = [(begins[n] + captures[n][1], begins[n] + lengths[n]) for n in tails]
    meta = _prefill_case(
        [end - start for start, end in spans], [captures[n][0] for n in tails], [True] * len(tails)
    )
    tail_x = torch.cat([x[start:end] for start, end in spans])
    with torch.inference_mode():
        y_tail = op(tail_x, *weights.args(), *meta, *resumed)
    offset = 0
    for n, (start, end) in zip(tails, spans, strict=True):
        row, context = captures[n][0], (provider, lengths[n], n)
        assert _same(y_tail[offset : offset + end - start], y[start:end]), ("N5 y", *context)
        if slots[n] >= 0:
            assert _same(resumed[0][row], captured[0][slots[n]]), ("N5 state", *context)
            assert _same(resumed[1][row], captured[1][slots[n]]), ("N5 history", *context)
        offset += end - start


def test_capture_grid():
    assert GatedDeltaNetPrefill.CAPTURE_GRID == GRID


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("initial", [False, True])
def test_one_sequence(ops, provider, hk, hv, length, initial):
    _check(provider, ops(provider, hk, hv), hk, hv, [length], [3], [initial], [9])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("case", list(PACKED))
def test_packed(ops, provider, hk, hv, case):
    _check(provider, ops(provider, hk, hv), hk, hv, *PACKED[case])


@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("length", CP_LENGTHS)
@pytest.mark.parametrize("initial", [False, True])
def test_port_context_parallel(ops, hk, hv, length, initial):
    """N1-N4 under CP, N2 bitwise (same plan as the prefix) or N3 (p == L)."""
    op = ops("port", hk, hv)
    if _plan(hv, length)[0] < 2:
        pytest.skip(f"cp_plan keeps T={length} on one segment on this device")
    _check("port", op, hk, hv, [length], [3], [initial], [9])


@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("length", REPORT_LENGTHS)
@pytest.mark.parametrize("report", ["n2", "n5"])
def test_port_context_parallel_report(ops, hk, hv, length, report):
    """Read, not gated: under CP, "n2" holds capture vs cold [0, p) to STATE_TOLERANCE when
    the plans differ (bitwise otherwise) and prints the measured max abs/rel difference;
    "n5" demands bitwise resume (N5 under CP is unverified)."""
    op = ops("port", hk, hv)
    if _plan(hv, length)[0] < 2:
        pytest.skip(f"cp_plan keeps T={length} on one segment on this device")
    _check("port", op, hk, hv, [length], [3], [True], [9], report=report)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
def test_graph_replay_follows_new_rows_and_lengths(ops, provider, hk, hv):
    """Captured once after warm_up alone (it covers captures), replayed with new rows,
    lengths, slots and flags: y and both pools bitwise equal to eager calls."""
    op = ops(provider, hk, hv)
    weights, x = _inputs(hk, hv, 512)
    lengths, slots, initial, rows = GRAPH_CASES[0]
    statics = [*_prefill_case(lengths, slots, initial), _rows(rows)]
    pools = [pool.clone() for pool in _base(hk, hv)]
    op.warm_up(512, max_sequences=4)
    with torch.inference_mode():
        graph, y = _capture(
            lambda: op(x, *weights.args(), *statics[:3], *pools, capture_rows=statics[3])
        )
    for lengths, slots, initial, rows in GRAPH_CASES:
        values = [*_prefill_case(lengths, slots, initial), _rows(rows)]
        base = _poisoned(hk, hv, slots, initial, rows)
        eager = [pool.clone() for pool in base]
        with torch.inference_mode():
            expected = op(x, *weights.args(), *values[:3], *eager, capture_rows=values[3])
        for static, value in zip([*statics, *pools], [*values, *base], strict=True):
            static.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected), lengths
        assert all(_same(a, b) for a, b in zip(pools, eager, strict=True)), lengths
        for row, length in zip(rows, lengths, strict=True):
            if row >= 0 and length >= GRID:
                assert not eager[0][row].isnan().any(), (lengths, row)
