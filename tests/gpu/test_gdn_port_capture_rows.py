# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Port prefill ``capture_rows``: the state and conv history at p = L // 64 * 64."""

import pytest
import torch

from ctikernels.gdn.chunk_ws._cp import cp_plan
from ctikernels.gdn.chunk_ws.prefill import PortPrefill

from .test_gdn_chunk_ws import _reason, _same_bits
from .test_gdn_port_prefill import make_case

_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
LENGTHS = (1, 63, 64, 65, 127, 128, 1000, 1024, 1025, 4096)
# Index 1 (65 tokens) is slot -1 in make_case.
PACKED = (1000, 65, 63, 128, 1, 1025, 4096, 64, 127, 1024)
GEOMETRIES = ((16, 32), (16, 48), (8, 16), (8, 24))


@pytest.fixture(scope="module")
def operators():
    cache = {}

    def get(hk, hv, cp=False, segments=None):
        key = hk, hv, cp, segments
        if key not in cache:
            cache[key] = PortPrefill(hk, hv, "cuda", cp=cp, cp_segments=segments)
        return cache[key]

    return get


def case(hk, hv, lengths, seed=13):
    """make_case plus one NaN capture row per sequence after its pools."""
    x, weights, meta, (state, conv) = make_case(hk, hv, lengths, seed)
    n = len(lengths)
    state = torch.cat([state, torch.full_like(state[:n], float("nan"))])
    conv = torch.cat([conv, torch.full_like(conv[:n], float("nan"))])
    rows = torch.arange(n + 2, 2 * n + 2, dtype=torch.int32, device=x.device)
    return x, weights, meta, (state, conv), rows


def clone(pools):
    return tuple(t.clone() for t in pools)


def call(op, x, weights, meta, pools, rows=None):
    return op(x, *weights.args(), *meta, *pools, capture_rows=rows)


def pack(x, cu, ranges):
    """Rows [cu[i] + a, cu[i] + b) of each (i, a, b), and their cu_seqlens."""
    parts = [x[cu[i] + a : cu[i] + b] for i, a, b in ranges]
    lengths = [b - a for _, a, b in ranges]
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return torch.cat(parts), offsets.to(x.device)


def check(op, hk, hv, lengths, *, bitwise_prefix=True, prefix_op=None):
    """(a)-(e) for one call; returns the CP prefix state error when not bitwise."""
    x, weights, meta, base, rows = case(hk, hv, lengths)
    cu, slots, initial = meta
    n = len(lengths)
    cu_host = cu.tolist()
    slots_host, rows_host = slots.tolist(), rows.tolist()
    plain = clone(base)
    y_plain = call(op, x, weights, meta, plain)
    pools = clone(base)
    y = call(op, x, weights, meta, pools, rows)
    # (a) y, slots and conv histories are those of the call without captures.
    _same_bits(y, y_plain)
    captured = [i for i, length in enumerate(lengths) if length >= 64]
    keep = [r for r in range(base[0].shape[0]) if r not in {rows_host[i] for i in captured}]
    for got, expected, original in zip(pools, plain, base, strict=True):
        _same_bits(got[keep], expected[keep])
        # (e) None writes no row; L < 64 writes nothing.
        _same_bits(expected[n + 2 :], original[n + 2 :])
    if not captured:
        return None
    # (b) The capture is the slot of the [0, p) prefix call (slot -1: a fresh row).
    prefix_slots = [s if s >= 0 else rows_host[i] for i, s in enumerate(slots_host)]
    prefix_initial = [bool(v) and s >= 0 for v, s in zip(initial.tolist(), slots_host, strict=True)]
    xp, cup = pack(x, cu_host, [(i, 0, length // 64 * 64) for i, length in enumerate(lengths)])
    prefix_meta = (
        cup,
        torch.tensor(prefix_slots, dtype=torch.int32, device=x.device),
        torch.tensor(prefix_initial, device=x.device),
    )
    prefix = clone(base)
    call(prefix_op or op, xp, weights, prefix_meta, prefix)
    worst = 0.0
    for i in captured:
        got, expected = pools[0][rows_host[i]], prefix[0][prefix_slots[i]]
        assert torch.isfinite(got).all()
        if bitwise_prefix:
            _same_bits(got, expected)
        else:
            scale = expected.abs().max().clamp_min(1e-30)
            worst = max(worst, ((got - expected).abs().max() / scale).item())
        _same_bits(pools[1][rows_host[i]], prefix[1][prefix_slots[i]])
        # (c) p == L: the capture is the slot.
        if lengths[i] % 64 == 0 and slots_host[i] >= 0:
            for k in range(2):
                _same_bits(pools[k][rows_host[i]], pools[k][slots_host[i]])
    # (d) Resume [p, L) from the restored capture: y and the final slot are the full call's.
    tails = [i for i in captured if lengths[i] % 64]
    if tails:
        xr, cur = pack(x, cu_host, [(i, lengths[i] // 64 * 64, lengths[i]) for i in tails])
        resume_meta = (
            cur,
            torch.tensor([rows_host[i] for i in tails], dtype=torch.int32, device=x.device),
            torch.ones(len(tails), dtype=torch.bool, device=x.device),
        )
        resumed = clone(pools)
        y_resume = call(op, xr, weights, resume_meta, resumed)
        offset = 0
        for i in tails:
            p, length = lengths[i] // 64 * 64, lengths[i]
            _same_bits(y_resume[offset : offset + length - p], y[cu_host[i] + p : cu_host[i + 1]])
            offset += length - p
            if slots_host[i] >= 0:
                for k in range(2):
                    _same_bits(resumed[k][rows_host[i]], pools[k][slots_host[i]])
    return worst


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("lengths", [(length,) for length in LENGTHS] + [PACKED])
def test_capture_rows(operators, hk, hv, lengths):
    check(operators(hk, hv), hk, hv, lengths)


@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("length", [200, 256, 1000, 1024, 1025, 4096])
@pytest.mark.parametrize("segments", [None, 4])
def test_capture_rows_cp(operators, hk, hv, length, segments):
    op = operators(hk, hv, True, segments)
    count, seg_len = cp_plan(length, hv, op.num_sms, segments)
    if count < 2:
        pytest.skip("not a CP plan")
    local = length - (count - 1) * seg_len
    site = "B" if local < 64 else "C" if local % 64 == 0 else "A"
    worst = check(op, hk, hv, (length,), bitwise_prefix=False, prefix_op=operators(hk, hv))
    print(f"cp L={length} J={count} seg={seg_len} site={site} prefix max rel diff={worst:.3e}")
    # Not bitwise to the non-CP prefix: the start of the last segment comes from the fixup.
    assert worst < 5e-2


@pytest.mark.parametrize("hv", [32, 48])
def test_capture_rows_all_negative(operators, hv):
    x, weights, meta, base, rows = case(16, hv, PACKED)
    plain, pools = clone(base), clone(base)
    _same_bits(
        call(operators(16, hv), x, weights, meta, pools, torch.full_like(rows, -1)),
        call(operators(16, hv), x, weights, meta, plain),
    )
    for got, expected in zip(pools, plain, strict=True):
        _same_bits(got, expected)


def test_capture_rows_validation(operators):
    op = operators(16, 32)
    x, weights, meta, base, rows = case(16, 32, (65, 128))
    for bad, error in (
        (rows.long(), TypeError),
        (rows[:1], ValueError),
        (rows.cpu(), ValueError),
        (torch.stack([rows, rows], 1)[:, 0], ValueError),
    ):
        with pytest.raises(error):
            call(op, x, weights, meta, clone(base), bad)


@pytest.mark.parametrize(
    "lengths,cp,segments", [(PACKED, False, None), ((1000,), True, 4), ((1024,), True, 4)]
)
def test_capture_rows_graph_replay(operators, lengths, cp, segments):
    """(h) One capture, replayed with new rows (including -1) and new projected rows."""
    op = operators(16, 32, cp, segments)
    x, weights, meta, base, rows = case(16, 32, lengths)
    graph_rows = rows.clone()
    pools = clone(base)
    out = torch.empty(x.shape[0], 32 * 128, dtype=x.dtype, device=x.device)
    op(x, *weights.args(), *meta, *pools, out=out, capture_rows=graph_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, *weights.args(), *meta, *pools, out=out, capture_rows=graph_rows)
    n = len(lengths)
    for new_rows in (
        rows.flip(0),
        torch.where(torch.arange(n, device=x.device) % 2 == 0, rows, -1),
    ):
        x.copy_(torch.randn_like(x))
        graph_rows.copy_(new_rows)
        eager_pools = clone(base)
        eager = call(op, x, weights, meta, eager_pools, new_rows.clone())
        for dst, src in zip(pools, base, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        _same_bits(out, eager)
        for got, expected in zip(pools, eager_pools, strict=True):
            _same_bits(got, expected)
