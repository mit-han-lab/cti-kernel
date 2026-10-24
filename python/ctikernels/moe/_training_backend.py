# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen MoE autograd, slot layout and common routing/token launches.

Numerics follow unsloth_zoo 2026.10.7's Qwen3.5-MoE training path (a numerical reference only;
no Unsloth code is used). The router and shared gate are HF's torch ops (bitwise), the experts
round at BF16 gate/up, SiLU, product and down output, and the combine sums the BF16 weighted
products in its fixed order.
Every reduction has a fixed order: results are bitwise repeatable run to run.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
import triton

from .triton import _kernels as K
from .triton._slot_layout import build_layout as _stable_layout
from .triton._slot_layout import layout_allocations as _stable_allocations


def _launch(blk):
    # Launch options of the block table; "stages" is optional (Triton's default otherwise).
    if "stages" in blk:
        return {"num_warps": blk["warps"], "num_stages": blk["stages"]}
    return {"num_warps": blk["warps"]}


class GemmBackend(Protocol):
    """Expert GEMMs over a chunk's expert-sorted ``SlotLayout``.

    ``x``/``dy`` are contiguous BF16 [c,H]. Slot p = t*kp1+j; ``wslot`` is
    flattened FP32 [c*kp1] holding the BF16 routing weights and the BF16 shared-gate sigmoid. Layout rows are padded to ``block_m``: sorted_slot
    [npad] is int32 (-1 for padding), block_expert [npad/BM] is int32 (E for
    shared, > E past the end), and inv maps slots to padded rows (-1 if absent).
    E is the local expert count. No shared expert means nis=ni and s13/s2=None.

    ``a``/``gsave``/``usave`` are BF16 [npad,NMAX], NMAX=max(ni,nis); saved
    buffers are used only with save/saved. ``y`` (the unweighted down output) and ``dxp``
    are BF16 [npad,H]. ``gg`` is BF16 [npad,2*NMAX], dg in [0,n), du in [NMAX,NMAX+n), where
    n is ni or nis for the block. ``dwpart`` is zero-initialized FP32 [npad,NT];
    bwd_slot writes one partial per live row per I-tile, leaving unused tiles
    zero. NT=dw_tiles(..., saved=saved), NMAX % NT == 0. Saved and
    recomputed backward may partition I differently. Padding is never read by combine.
    Gate/up recomputation must use the same arithmetic as the saved path.
    ``device_layout`` selects CuTe's slot layout; otherwise Triton's (stable
    order within each expert). Atomic layout placement can reorder rows within each expert;
    independent GEMM row results and the inverse slot mapping are unchanged.
    ``device_token`` selects CuTe's token backward for top_k <= 8; larger
    selections retain Triton. CTI_MOE_CUTE_TOKEN=0 disables this path for A/B.
    """

    device_layout: bool
    device_token: bool

    def block_m(self, device, tokens: int, ni: int, nis: int) -> int: ...

    def dw_tiles(self, device, tokens, ni, nis, *, saved=False) -> int: ...

    def gate_up(self, x, w13, s13, lay, a, gsave, usave, *, ni, nis, kp1, save) -> None: ...

    def down(self, a, w2, s2, lay, y, *, ni, nis) -> None: ...

    def bwd_slot(
        self,
        x,
        dy,
        w13,
        s13,
        w2,
        s2,
        lay,
        wslot,
        gsave,
        usave,
        gg,
        dwpart,
        *,
        ni,
        nis,
        kp1,
        saved,
    ) -> None: ...

    def dx_pair(self, gg, w13, s13, lay, dxp, *, ni, nis) -> None: ...

    def bwd_token(self, dxp, inv, dwpart, wslot, sel, r, sg, dx, valid) -> None: ...


def _route(x, r, sg, top_k, routing, valid, renormalize):
    """HF's Qwen3_5MoeTopKRouter and shared gate as Unsloth runs them: BF16 logits, FP32 softmax
    over all E, top-k (or the replayed ``routing``), FP32 renormalisation, BF16 weights; the
    shared gate sigmoid(x . w_sg) in BF16. Returns int32 sel, the forward slot weights (BF16
    values), the router's FP32 weights for the backward, bool live, uint8 record and, without
    renormalisation, the unselected experts' FP32 probabilities."""
    t = x.shape[0]
    probs = torch.softmax(F.linear(x, r), dim=-1, dtype=torch.float32)
    if routing is None:
        top, ids = torch.topk(probs, top_k, dim=-1)
    else:
        ids = routing.long()
        top = probs.gather(1, ids)
    w = top / top.sum(dim=-1, keepdim=True) if renormalize else top
    live = torch.arange(t, device=x.device)
    live = live < valid.reshape(1).clamp(0, t) if valid is not None else live >= 0
    w = torch.where(live[:, None], w, 0.0)
    sel = torch.where(live[:, None], ids, 0).to(torch.int32)
    record = sel.to(torch.uint8)
    slot_w = [w.to(torch.bfloat16).float()]
    grad_w = [w]
    if sg is not None:
        gate = torch.where(live[:, None], torch.sigmoid(F.linear(x, sg)), 0.0).float()
        slot_w.append(gate)
        grad_w.append(gate)
    rest = None
    if not renormalize:
        rest = torch.where(live[:, None], probs.scatter(1, ids, 0.0), 0.0)
    return (
        sel.contiguous(),
        torch.cat(slot_w, dim=1).contiguous(),
        torch.cat(grad_w, dim=1).contiguous(),
        live,
        record,
        rest,
    )


def _chunk_valid(valid, c0, c):
    if valid is None:
        return None
    return (valid.clamp(0, c0 + c) - c0).clamp(0, c).to(torch.int32).reshape(1)


def _unselected_router_term(inv, dwpart, ws, rest, rf, kp1, top_k):
    # Full-softmax router derivative on the experts the token did not select:
    # sum_{i not selected} -p_i S R_i with S = sum_j w_j dw_j over this rank's slots.
    # S is a rank partial (remote dw_j = 0), so summing ranks recovers the full term.
    c = rest.shape[0]
    pos = inv.view(c, kp1)[:, :top_k].long()
    dw = dwpart.sum(dim=1)
    dw = torch.where(pos >= 0, dw[pos.clamp_min(0)], 0.0)
    s = (ws.view(c, kp1)[:, :top_k] * dw).sum(dim=1, keepdim=True)
    return -s * (rest @ rf)


def _add_extra(dx, extra):
    # The token kernel already rounded dx to BF16; masked rows stay +0.0.
    if extra is not None:
        dx.copy_((dx.float() + extra).to(torch.bfloat16))


def _held(nbytes):
    # The CUDA caching allocator hands out multiples of 512 bytes.
    return -(-nbytes // 512) * 512


def training_bytes(tokens, h, ni, nis, e, top_k, shared, backend, device, *, chunk_tokens,
                   save_activations, renormalize):  # fmt: skip
    """``moe_block_training_bytes`` on a resolved backend: ``(saved, backward)``."""
    if tokens == 0:
        return 0, 0
    nmax, kp1 = max(ni, nis), top_k + shared
    saved = _held(2 * tokens * h) + _held(4 * tokens * top_k) + 2 * _held(4 * tokens * kp1)
    backward = _held(2 * tokens * h) + _held(tokens * top_k)  # dx; autograd's zero record grad
    if not renormalize:
        saved += _held(4 * tokens * e)  # unselected probabilities
        backward += _held(4 * e * h)  # FP32 router
    layout_allocations = _stable_allocations
    if backend.device_layout:
        from .cute._routing import layout_allocations
    chunk_peak = 0
    for c0 in range(0, tokens, chunk_tokens):
        c = min(chunk_tokens, tokens - c0)
        bm, p = backend.block_m(device, c, ni, nis), c * kp1
        npad = (p + bm - 1) // bm * bm + (e + 1) * bm  # every expert's last block partial
        saved += sum(_held(n) for n in layout_allocations(p, e, bm))
        if save_activations:
            saved += 2 * _held(2 * npad * nmax)
        nt = backend.dw_tiles(device, c, ni, nis, saved=save_activations)
        chunk = _held(4 * npad * nmax) + _held(4 * npad * nt) + _held(2 * npad * h)
        if not renormalize:
            # the unselected-router term and its FP32 add onto dx (upper bound)
            chunk += 2 * _held(4 * npad) + 4 * _held(8 * c * top_k) + 14 * c * h + 4 * 512
        chunk_peak = max(chunk_peak, chunk)
    return saved, backward + chunk_peak


class _MoEBlock(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        r,
        sg,
        w13,
        w2,
        s13,
        s2,
        top_k,
        expert_map,
        routing,
        valid,
        save_activations,
        chunk_tokens,
        backend: GemmBackend | None = None,
        renormalize=True,
    ):
        from .triton._provider import _blocks
        from .triton._provider import backend as triton_backend

        t, h = x.shape
        if t == 0:
            record = torch.empty((0, top_k), dtype=torch.uint8, device=x.device)
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(record)
            return torch.empty_like(x), record
        e = w13.shape[0]
        ni = w2.shape[2]
        nis = s2.shape[1] if s2 is not None else ni
        nmax = max(ni, nis)
        kp1 = top_k + (sg is not None)
        blk = _blocks(x.device, ni, nis)
        if backend is None:
            backend = triton_backend(x.device, ni, nis, h)
        bnh = min(blk["bn"], h)
        if h % bnh:
            bnh = min(64, h)
        build_layout = _stable_layout
        layout_args = {}
        if backend.device_layout:
            from .cute._routing import build_layout

            layout_args = {"capability": backend.capability}
        sel, wslot, wgrad, live, record, rest = _route(x, r, sg, top_k, routing, valid, renormalize)
        out = torch.empty_like(x)
        chunks = []
        for c0 in range(0, t, chunk_tokens):
            c1 = min(t, c0 + chunk_tokens)
            c = c1 - c0
            bm = backend.block_m(x.device, c, ni, nis)
            lay = build_layout(
                sel[c0:c1],
                live[c0:c1],
                e,
                bm,
                expert_map,
                has_shared=sg is not None,
                **layout_args,
            )
            npad = lay.sorted_slot.numel()
            a = torch.empty(npad, nmax, dtype=torch.bfloat16, device=x.device)
            gs = (
                torch.empty(npad, nmax, dtype=torch.bfloat16, device=x.device)
                if save_activations
                else a
            )
            us = (
                torch.empty(npad, nmax, dtype=torch.bfloat16, device=x.device)
                if save_activations
                else a
            )
            vc = _chunk_valid(valid, c0, c)
            y = torch.empty(npad, h, dtype=torch.bfloat16, device=x.device)
            # Keep each GEMM pair adjacent: even valid-token clamping launches.
            # Layout has completed before gate_up, so down may read it before
            # its PDL wait. Routing/layout must not signal early dependents.
            backend.gate_up(
                x[c0:c1],
                w13,
                s13,
                lay,
                a,
                gs,
                us,
                ni=ni,
                nis=nis,
                kp1=kp1,
                save=save_activations,
            )
            backend.down(a, w2, s2, lay, y, ni=ni, nis=nis)
            K._combine_kernel[(c, h // bnh)](
                y,
                lay.inv,
                wslot[c0:c1],
                out[c0:c1],
                vc if vc is not None else lay.inv,
                c,
                h,
                top_k,
                sg is not None,
                vc is not None,
                bnh,
                **_launch(blk),
            )
            chunks.append(
                (
                    c0,
                    c1,
                    lay,
                    gs if save_activations else None,
                    us if save_activations else None,
                    vc,
                )
            )
        ctx.save_for_backward(x, r, sg, w13, w2, s13, s2, sel, wslot, wgrad, rest)
        ctx.chunks = chunks
        ctx.cfg = (top_k, blk, bnh, backend)
        ctx.mark_non_differentiable(record)
        return out, record

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy, _drecord):
        x = ctx.saved_tensors[0]
        if x.shape[0] == 0:
            return (torch.empty_like(x),) + (None,) * (len(ctx.needs_input_grad) - 1)
        x, r, sg, w13, w2, s13, s2, sel, wslot, wgrad, rest = ctx.saved_tensors
        top_k, blk, bnh, backend = ctx.cfg
        t, h = x.shape
        ni = w2.shape[2]
        nis = s2.shape[1] if s2 is not None else ni
        nmax = max(ni, nis)
        kp1 = top_k + (sg is not None)
        dy = dy.to(torch.bfloat16).contiguous()
        dx = torch.empty_like(x)
        kpad = triton.next_power_of_2(top_k)
        rf = r.float() if rest is not None else None
        gg = dwpart = dxp = extra = None
        for c0, c1, lay, gs, us, vc in ctx.chunks:
            gg = dwpart = dxp = extra = None  # the previous chunk's, before this one allocates
            c = c1 - c0
            npad = lay.sorted_slot.numel()
            saved = gs is not None
            nt = backend.dw_tiles(x.device, c, ni, nis, saved=saved)
            gg = torch.empty(npad, 2 * nmax, dtype=torch.bfloat16, device=x.device)
            dwpart = torch.zeros(npad, nt, dtype=torch.float32, device=x.device)
            ws = wslot[c0:c1].reshape(-1)
            wg = wgrad[c0:c1].reshape(-1)
            dxp = torch.empty(npad, h, dtype=torch.bfloat16, device=x.device)
            # Zeroing and allocation precede the pair; dx_pair's pre-wait reads
            # use the layout retained from forward, never bwd_slot's gg output.
            backend.bwd_slot(
                x[c0:c1],
                dy[c0:c1],
                w13,
                s13,
                w2,
                s2,
                lay,
                ws,
                gs,
                us,
                gg,
                dwpart,
                ni=ni,
                nis=nis,
                kp1=kp1,
                saved=saved,
            )
            backend.dx_pair(gg, w13, s13, lay, dxp, ni=ni, nis=nis)
            extra = None
            if rest is not None:
                extra = _unselected_router_term(lay.inv, dwpart, wg, rest[c0:c1], rf, kp1, top_k)
            if backend.device_token and top_k <= 8:
                backend.bwd_token(
                    dxp,
                    lay.inv,
                    dwpart,
                    wg,
                    sel[c0:c1].contiguous(),
                    r,
                    sg,
                    dx[c0:c1],
                    vc,
                )
                _add_extra(dx[c0:c1], extra)
                continue
            K._bwd_token_kernel[(c, h // bnh)](
                dxp,
                lay.inv,
                dwpart,
                wg,
                sel[c0:c1].contiguous(),
                r,
                sg if sg is not None else r,
                dx[c0:c1],
                vc if vc is not None else lay.inv,
                c,
                h,
                top_k,
                kp1,
                nt,
                kpad,
                vc is not None,
                bnh,
                HAS_SHARED=sg is not None,
                **_launch(blk),
            )
            _add_extra(dx[c0:c1], extra)
        return (dx,) + (None,) * (len(ctx.needs_input_grad) - 1)
