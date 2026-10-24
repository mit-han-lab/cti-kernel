# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks of the diagnostic against the independent token recurrence."""

import pytest
import torch

from ._gdn_reference import EPS, Weights, _Sequence, _silu
from ._gdn_v2_reference import chunk0_intermediates


@pytest.mark.parametrize("length,slot,initial", [(1, 1, True), (17, 0, False), (65, -1, True)])
@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_chunk0_matches_token_reference(length, slot, initial, vsplit):
    # Nonzero begin, a grouped value head, poisoned unread pools, and unequal
    # channel/tap values catch layout/history/head errors independently of WY.
    hk, hv, head, begin = 2, 6, 4, 3
    dc = (2 * hk + hv) * 128
    rng = torch.Generator().manual_seed(73)
    x = torch.randn(begin + length, dc + hv * 128 + 2 * hv, generator=rng).bfloat16()
    weights = Weights(
        (torch.randn(dc, 4, generator=rng) * 0.3).bfloat16(),
        torch.full((hv,), -2.0),
        torch.linspace(-1.5, -0.5, hv),
        torch.linspace(0.9, 1.1, 128),
    )
    state = torch.randn(2, hv, 128, 128, generator=rng) * 0.2
    history = torch.randn(2, 3, dc, generator=rng).bfloat16()
    if not (slot >= 0 and initial):
        state.fill_(float("nan"))
        history.fill_(float("nan"))
    meta = (
        torch.tensor([0, begin, begin + length]),
        torch.tensor([-1, slot]),
        torch.tensor([False, initial]),
    )
    refs = [
        chunk0_intermediates(
            x,
            *weights.args()[:3],
            *meta,
            state,
            history,
            k_heads=hk,
            sequence=1,
            value_head=head,
            vsplit=vsplit,
            part=p,
        )
        for p in range(vsplit)
    ]
    sequence = _Sequence(
        weights,
        hk,
        state[slot] if slot >= 0 and initial else torch.zeros(hv, 128, 128),
        history[slot] if slot >= 0 and initial else torch.zeros(3, dc, dtype=torch.bfloat16),
    )
    valid = min(64, length)
    exact_y = torch.stack(
        [sequence.step(row).view(hv, 128)[head] for row in x[begin : begin + valid]]
    )
    o = torch.cat([r["o"] for r in refs], dim=1)[:valid]
    z = x[begin : begin + valid, dc + head * 128 : dc + (head + 1) * 128].double()
    y = (
        o
        * torch.rsqrt(o.square().mean(-1, keepdim=True) + EPS)
        * weights.norm_weight.double()
        * _silu(z)
    )
    torch.testing.assert_close(y, exact_y, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.cat([r["H1"] for r in refs]), sequence.pool_state()[head], atol=1e-12, rtol=1e-12
    )
    for ref in refs:
        assert all(
            value.dtype == torch.float64 and value.isfinite().all() for value in ref.values()
        )
        for name in ("Q_h", "K_h", "L", "A_h", "W", "B", "U", "r", "o", "beta"):
            assert torch.count_nonzero(ref[name][valid:]) == 0
        assert torch.all(ref["g"][valid:] == ref["g"][valid - 1])
        torch.testing.assert_close(
            (torch.eye(64) + ref["L"]) @ ref["S"],
            torch.eye(64, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_v_channels_taps_and_fresh_halo(vsplit):
    # Zero K makes S=I and U=B. Only one V head has data/taps; poisoned history
    # must not enter the first three rows. Nonconstant channels catch strip offsets.
    hk, hv, head, length = 2, 6, 4, 5
    dc = (2 * hk + hv) * 128
    x = torch.zeros(length, dc + hv * 128 + 2 * hv, dtype=torch.bfloat16)
    channel = 2 * hk * 128 + head * 128
    values = (torch.arange(length)[:, None] * 128 + torch.arange(128)[None]) / 128
    x[:, channel : channel + 128] = values
    taps = torch.tensor([1.0, -0.5, 0.25, 2.0], dtype=torch.bfloat16)
    cw = torch.zeros(dc, 4, dtype=torch.bfloat16)
    cw[channel : channel + 128] = taps
    expected = torch.zeros(64, 128, dtype=torch.float64)
    values = x[:, channel : channel + 128].double()
    for row in range(length):
        conv = sum(values[row - lag] * taps[3 - lag].double() for lag in range(min(row, 3) + 1))
        expected[row] = 0.5 * torch.nn.functional.silu(conv)
    for part in range(vsplit):
        ref = chunk0_intermediates(
            x,
            cw,
            torch.zeros(hv),
            torch.zeros(hv),
            torch.tensor([0, length]),
            torch.tensor([0]),
            torch.tensor([False]),
            torch.full((1, hv, 128, 128), float("nan")),
            torch.full((1, 3, dc), float("nan")),
            k_heads=hk,
            sequence=0,
            value_head=head,
            vsplit=vsplit,
            part=part,
        )
        target = expected[:, part * (128 // vsplit) : (part + 1) * (128 // vsplit)]
        torch.testing.assert_close(ref["B"], target, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ref["U"], target, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("merge,split", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_v1_scratch_decoder(merge, split, vsplit):
    from scripts.debug_gdn_v2 import v1_chunk0

    # Ragged sequence 2 starts at chunk 3, not at its sequence index. Leave an
    # unused chunk, and poison low slabs to expose hi/lo or row-stride confusion.
    hk, hv, chunks, head = 2, 6, 6, 4
    heads = hv if merge else hk
    tune = {
        "form": "wy",
        "o_form": "legacy",
        "merge_prepare": merge,
        "w_split": split,
        "aq_out": split,
    }
    q = torch.full((1 + split, chunks, heads, 64, 128), float("nan"))
    k = torch.full((chunks, heads, 64, 128), float("nan"))
    w = torch.full((chunks, hv, 1 + split, 64, 128), float("nan"))
    u = torch.full((chunks, hv, 64, 128), float("nan"))
    aq = torch.full((chunks, hv, 1 + split, 64, 64), float("nan"))
    g = torch.full((chunks, hv, 64), float("nan"))
    tile = torch.arange(8192, dtype=torch.float32).view(64, 128)
    kh = head if merge else head // (hv // hk)
    q[0, 3, kh], k[3, kh] = tile, tile + 1
    w[3, head, 0], u[3, head] = tile + 2, tile + 3
    aq[3, head, 0], g[3, head] = tile[:, :64] + 4, tile[0, :64] + 5
    scratch = {
        name: t.flatten() for name, t in {"q": q, "k": k, "w": w, "u": u, "aq": aq, "g": g}.items()
    }
    for part in range(vsplit):
        got = v1_chunk0(scratch, tune, [65, 1, 70], hk, hv, 2, head, vsplit, part)
        for name, target in (
            ("Q_h", tile),
            ("K_h", tile + 1),
            ("W", tile + 2),
            ("A_h", tile[:, :64] + 4),
            ("g", tile[0, :64] + 5),
        ):
            torch.testing.assert_close(got[name], target)
        start, end = part * (128 // vsplit), (part + 1) * (128 // vsplit)
        torch.testing.assert_close(got["U"], tile[:, start:end] + 3)


def test_summary_excludes_padding_and_keeps_nan_coordinates(capsys):
    from scripts.debug_gdn_v2 import summarize

    actual = torch.zeros(64, 128)
    reference = torch.ones_like(actual)
    actual[15, 9] = float("nan")
    actual[16:] = float("nan")
    summarize("Q_h", actual, reference, token_rows=16)
    output = capsys.readouterr().out
    assert "max_abs_diff=nan" in output and "ref_max_abs=1" in output
    assert "nonfinite=1 " in output and "first_nan(row,col)=(15, 9)" in output
    assert "pad nonfinite count=6144" in output
    actual = torch.zeros(1, 64)
    actual[0, 16:] = float("nan")
    summarize("g", actual, torch.ones_like(actual), token_cols=16)
    output = capsys.readouterr().out
    assert "nonfinite=0 " in output and "first_nan(row,col)=None" in output
    assert "pad nonfinite count=48" in output
    actual[0, 7] = float("nan")
    summarize("beta", actual, torch.ones_like(actual), token_cols=16, row_offset=1)
    assert "first_nan(row,col)=(1, 7)" in capsys.readouterr().out
