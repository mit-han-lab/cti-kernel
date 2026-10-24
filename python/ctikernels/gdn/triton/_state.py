# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Algorithm reference: the chunked gated delta rule's state pass and output (the
# flash-linear-attention papers; fla-core 0.5.2's chunk_delta_h and chunk_o kernels); no kernel
# code is copied.

"""GDN prefill, state part: one program per (sequence, value head, value block) walks the
sequence's 64-token chunks with the state S [K, BV] in FP32 registers, read in place from
``state[slot]`` (zeros unless continued):

    h  = S rounded to BF16, stored per chunk for the output kernel (as fla-core's h)
    v' = u - w h                         (written over u, BF16)
    S  = 2^gamma_last S + k^T (2^(gamma_last - gamma) v')   (v' as BF16 high + low parts)

At the end it writes S to ``state[slot]`` for slots >= 0 that are non-empty or fresh. A
capturing sequence (``capture_rows[n] >= 0``, read only when ``capturing``) with
p = floor(L / 64) * 64 >= 64 also gets the state after chunk p / 64 - 1's update (the state
carried at p) in ``state[row]``. The programs of a sequence also write its conv history (the
last three rows of [history ++ its rows of projected]) to the slot, and rows p-3..p-1 to the
capture row, each a slice of the channels; the features kernel read the old history before this launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import HISTORY, Geometry, K, V
from ._intra import CHUNK, chunks_before

_BV = 32
_BN = 128  # sequences per scan step
_WARPS = 4
_STAGES = 2  # RTX PRO 6000: 20-23% faster than one stage now that the output is split out


@triton.jit
def _history(projected, p_stride, conv_state, slot, continued, end, length, row, cols, active,
             HISTORY: tl.constexpr, WIDTH: tl.constexpr):  # fmt: skip
    """The last three rows of [old history ++ rows ending at ``end`` (``length`` of them)]
    for ``cols``, their address in ``conv_state[row]`` and mask; nothing is read unless
    ``active``."""
    r = tl.arange(0, 4)  # oldest first; row 3 is padding
    m = (r < HISTORY)[:, None] & (cols < WIDTH)[None, :] & active
    back = (HISTORY - r)[:, None]
    from_x = m & (length >= back)
    x = tl.load(projected + (end - back) * p_stride + cols[None, :], mask=from_x, other=0.0)
    old = conv_state + slot * (HISTORY * WIDTH) + (HISTORY + length - back) * WIDTH + cols[None, :]
    kept = tl.load(old, mask=m & (length < back) & continued, other=0.0)
    out = conv_state + row * (HISTORY * WIDTH) + r[:, None] * WIDTH + cols[None, :]
    return out, tl.where(from_x, x, kept), m


@triton.jit
def _update(kT, vs):
    """k^T v' with v' as BF16 high and low parts (fla-core rounds it to BF16 once)."""
    high = vs.to(kT.dtype)
    return tl.dot(kT, high) + tl.dot(kT, (vs - high.to(tl.float32)).to(kT.dtype))


# One binary with and without captures: ``capturing`` is a runtime flag.
@triton.jit(do_not_specialize=["capturing"])
def _state_kernel(
    k, w, u, gamma, hbuf, cu_seqlens, slot_indices, has_initial, state,
    capture_rows, capturing, projected, p_stride, conv_state,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    CHUNK: tl.constexpr, HISTORY: tl.constexpr, WIDTH: tl.constexpr,
    BW: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    n, hv, vb = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    hk = hv // (HV // HK)
    bos = tl.load(cu_seqlens + n).to(tl.int64)
    eos = tl.load(cu_seqlens + n + 1).to(tl.int64)
    length = eos - bos
    slot = tl.load(slot_indices + n).to(tl.int64)
    continued = (slot >= 0) & (tl.load(has_initial + n) != 0)
    i = tl.arange(0, CHUNK)
    c = tl.arange(0, D)
    vc = vb * BV + tl.arange(0, BV)
    # S [K, BV]: the slot's state is stored value-major ([Hv, V, K]); read it transposed.
    head = (hv * D + vc)[None, :] * D + c[:, None]
    S = tl.load(state + slot * (HV * D * D) + head, mask=continued, other=0.0)
    # Conv history: this program's slice of the channels, read before any write.
    program = hv * (D // BV) + vb
    cols = program * BW + tl.arange(0, BW)
    writes = (slot >= 0) & ((length > 0) | (continued == 0))
    final_conv = _history(projected, p_stride, conv_state, slot, continued, eos, length, slot,
                          cols, writes, HISTORY, WIDTH)  # fmt: skip
    # The capture point p = floor(L / 64) * 64, the end of the chunk ``stop``.
    row = tl.load(capture_rows + n, mask=capturing != 0, other=-1).to(tl.int64)
    p = length // CHUNK * CHUNK
    on = (row >= 0) & (p >= CHUNK)
    stop = p // CHUNK - 1
    capture_conv = _history(projected, p_stride, conv_state, slot, continued, bos + p, p, row,
                            cols, on, HISTORY, WIDTH)  # fmt: skip
    # The sequence's first entry in the chunk table: h is stored per entry.
    first = chunks_before(cu_seqlens, n, CHUNK, BN)
    for chunk in range(0, tl.cdiv(length, CHUNK)):
        t = bos + chunk * CHUNK + i
        m = t < eos
        kT = tl.load(k + (t * HK + hk)[None, :] * D + c[:, None], mask=m[None, :], other=0.0)
        ww = tl.load(w + (t * HV + hv)[:, None] * D + c[None, :], mask=m[:, None], other=0.0)
        tv = (t * HV + hv)[:, None] * D + vc[None, :]
        uu = tl.load(u + tv, mask=m[:, None], other=0.0)
        gam = tl.load(gamma + t * HV + hv, mask=m, other=0.0)
        Sb = S.to(kT.dtype)
        tl.store(hbuf + ((first + chunk) * HV + hv) * D * D + c[:, None] * D + vc[None, :], Sb)
        vn = uu.to(tl.float32) - tl.dot(ww, Sb)
        tl.store(u + tv, vn.to(u.dtype.element_ty), mask=m[:, None])
        last = tl.minimum(length - chunk * CHUNK, CHUNK) - 1
        g = tl.sum(tl.where(i == last, gam, 0.0), 0)
        vs = tl.where(m[:, None], vn * tl.exp2(g - gam)[:, None], 0.0)
        S = S * tl.exp2(g) + _update(kT, vs)
        if on & (chunk == stop):
            tl.store(state + row * (HV * D * D) + head, S)
    if on:
        tl.store(capture_conv[0], capture_conv[1], mask=capture_conv[2])
    if writes:
        tl.store(state + slot * (HV * D * D) + head, S)
        tl.store(final_conv[0], final_conv[1], mask=final_conv[2])


def state(
    geometry: Geometry,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gamma: torch.Tensor,
    capacity: int,
    projected: torch.Tensor,
    cu_seqlens: torch.Tensor,
    slot_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    state: torch.Tensor,
    conv_state: torch.Tensor,
    capture_rows: torch.Tensor | None,
    launch,
) -> torch.Tensor:
    """h [capacity, Hv, K, V] BF16 per chunk table entry; v' written over u; updates the pools
    in place (see the module)."""
    hv, count = geometry.v_heads, cu_seqlens.shape[0] - 1
    hbuf = k.new_empty(capacity, hv, K, V)
    if count:
        programs = hv * (V // _BV)
        launch(
            _state_kernel,
            (count, hv, V // _BV),
            k, w, u, gamma, hbuf, cu_seqlens, slot_indices, has_initial_state, state,
            slot_indices if capture_rows is None else capture_rows,
            int(capture_rows is not None),
            projected, projected.stride(0), conv_state,
            HK=geometry.k_heads, HV=hv, D=K, BV=_BV, CHUNK=CHUNK, HISTORY=HISTORY,
            WIDTH=geometry.conv_dim,
            BW=triton.next_power_of_2(triton.cdiv(geometry.conv_dim, programs)), BN=_BN,
            num_warps=_WARPS, num_stages=_STAGES,
        )  # fmt: skip
    return hbuf


__all__ = ["state"]
