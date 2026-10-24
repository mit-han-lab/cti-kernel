# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Portable reference and a deliberately simple, per-pair high precision oracle."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def apply_(out, x, a, b, ids, scale, top_k, routed_weights=None, z_out=None, *, down=False):
    """Grouped FP32 shrink/expand, single BF16 rounding after addition to the cache. Down
    projections read per-pair rows and may scale each pair's delta by ``routed_weights``."""
    ids = (
        ids.reshape(-1)
        if ids is not None
        else torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
    )
    slices = a.shape[1]
    n = b.shape[-2]
    for e in range(a.shape[0]):
        pairs = torch.where(ids == e)[0]
        rows = pairs if down else pairs // top_k
        for s in range(slices):
            z = x[rows].float() @ a[e, s].float().T
            delta = z @ b[e, s].float().T
            multiplier = scale
            if routed_weights is not None:
                multiplier = scale * routed_weights.reshape(-1)[pairs, None].float()
            old = out[pairs, s * n : (s + 1) * n].float()
            update = multiplier * delta
            out[pairs, s * n : (s + 1) * n] = torch.where(update == 0, old, old + update).to(
                out.dtype
            )
            if z_out is not None:
                z_out[pairs, s] = z.to(z_out.dtype)
    return out


@torch.no_grad()
def oracle(out, x, a, b, ids, *, scaling, top_k, routed_weights=None, dtype=torch.float64):
    """Return (rounded output, unrounded delta, shrink) using independent per-pair loops.

    Factors are canonical [E,S,r,K]/[E,S,N,r]. Invalid pairs have zero delta/shrink.
    This slow helper is for tests; it synchronizes on CUDA routing ids.
    """
    p, slices, rank = out.shape[0], a.shape[1], a.shape[2]
    delta = torch.zeros(out.shape, dtype=dtype, device=out.device)
    saved = torch.zeros((p, slices, rank), dtype=dtype, device=out.device)
    scale = torch.as_tensor(scaling, dtype=dtype, device=out.device)
    n = b.shape[-2]
    for pair in range(p):
        e = 0 if ids is None else int(ids.reshape(-1)[pair])
        if not 0 <= e < a.shape[0]:
            continue
        row = pair // top_k if routed_weights is None else pair
        w = 1 if routed_weights is None else routed_weights.reshape(-1)[pair].to(dtype)
        for s in range(slices):
            z = torch.mv(a[e, s].to(dtype), x[row].to(dtype))
            saved[pair, s] = z
            delta[pair, s * n : (s + 1) * n] = scale * w * torch.mv(b[e, s].to(dtype), z)
    return (out.to(dtype) + delta).to(out.dtype), delta, saved


def backward_shrink(x, factor, ids, *, row_divisor=1):
    """[rows,K] @ [E,r,K]^T -> [pairs,r], without assuming contiguous rows."""
    result = torch.zeros((ids.numel(), factor.shape[1]), device=x.device, dtype=torch.float32)
    for e in range(factor.shape[0]):
        pairs = torch.where(ids.flatten() == e)[0]
        result[pairs] = x[pairs // row_divisor].float() @ factor[e].float().T
    return result


def backward_expand(out, z, factor, ids, scale):
    result = out.float().clone()
    for e in range(factor.shape[0]):
        pairs = torch.where(ids.flatten() == e)[0]
        result[pairs] += scale * (z[pairs].float() @ factor[e].float().T)
    return result


def backward_base(x, weight, ids, *, row_divisor=1):
    return backward_shrink(x, weight, ids, row_divisor=row_divisor)


def backward_act(c1, dc2u, rw, ids, experts, *, need_dc1=True, need_dw=True, need_c2=True):
    valid = ((ids >= 0) & (ids < experts)).reshape(-1, 1)
    g, u = c1.float().chunk(2, dim=-1)
    sigmoid = torch.sigmoid(g)
    silu = g * sigmoid
    c2 = silu * u
    weighted = dc2u.float() * rw.reshape(-1, 1)
    dc1 = None
    if need_dc1:
        dg = weighted * u * (sigmoid + silu * (1 - sigmoid))
        du = weighted * silu
        dc1 = torch.where(valid, torch.cat((dg, du), dim=-1), 0)
    dw = torch.where(valid[:, 0], (dc2u.float() * c2).sum(-1), 0) if need_dw else None
    c2 = torch.where(valid, c2, 0) if need_c2 else None
    return dc1, dw, c2


def backward_grouped_outer(
    left,
    right,
    ids,
    experts,
    scale,
    *,
    left_divisor=1,
    right_divisor=1,
    pair_scale=None,
    out=None,
    **alignment,
):
    """Grouped L^T R; gather rows only in this oracle, never in the CUDA path."""
    result = torch.zeros(
        (experts, left.shape[1], right.shape[1]), device=left.device, dtype=torch.float32
    )
    for e in range(experts):
        pairs = torch.where(ids.flatten() == e)[0]
        lhs = left[pairs // left_divisor].float()
        if pair_scale is not None:
            lhs = lhs * pair_scale.flatten()[pairs, None]
        result[e] = scale * (lhs.T @ right[pairs // right_divisor].float())
    if out is not None:
        out.copy_(result)
        return out
    return result


class TorchSteps:
    shrink = staticmethod(
        lambda x, a, ids, alignment, divisor, **kw: backward_shrink(x, a, ids, row_divisor=divisor)
    )
    expand = staticmethod(
        lambda out, z, b, ids, alignment, scale: backward_expand(out, z, b, ids, scale)
    )
    act = staticmethod(backward_act)
    outer = staticmethod(backward_grouped_outer)
    outer_buffer = staticmethod(lambda factor, ids: torch.empty_like(factor, dtype=torch.float32))
    base = staticmethod(backward_base)


def autograd_apply_(out, x, a, b, ids, scale, top_k, routed_weights, z_out, activated, *, down):
    """Differentiable FP32 grouped shrink/expand with the kernels' contract.

    One BF16 rounding of the cache sum (or of the SwiGLU when ``activated`` is
    given); invalid pairs and zero contributions leave destinations untouched.
    Writes in place, so autograd tracks the caller's destination.
    """
    p = out.shape[0]
    e, slices, rank, k = a.shape
    n = b.shape[-2]
    flat = (
        torch.zeros(p, dtype=torch.long, device=x.device) if ids is None else ids.reshape(-1).long()
    )
    valid = (flat >= 0) & (flat < e)
    # Empty-slice sums are exact zeros that keep the update graph-connected, so a
    # batch without valid local routes (all remote under EP) backprops zeros.
    anchor = sum(
        t.reshape(-1)[:0].float().sum()
        for t in (x, a, b, routed_weights)
        if t is not None and t.requires_grad
    )
    delta = torch.zeros((p, slices * n), dtype=torch.float32, device=x.device) + anchor
    for expert in flat[valid].unique().tolist():
        pairs = torch.where(flat == expert)[0]
        rows = pairs if down else pairs // top_k
        z = (x[rows].float() @ a[expert].float().reshape(slices * rank, k).T).view(-1, slices, rank)
        update = torch.einsum("msr,snr->msn", z, b[expert].float()).reshape(-1, slices * n)
        if routed_weights is not None:
            update = update * routed_weights.reshape(-1)[pairs, None].float()
        delta = delta.index_add(0, pairs, update)
        if z_out is not None:
            z_out[pairs] = z.detach().to(z_out.dtype)
    delta = delta * scale.float()
    old = out.float()
    # Zero contributions leave destinations bit-identical (x - +0.0 keeps -0.0),
    # yet both branches carry d/d(delta) = 1, so B = 0 at init still trains.
    summed = torch.where(delta == 0, old - (delta.detach() - delta), old + delta)
    if activated is None:
        out.copy_(torch.where(valid[:, None], summed.to(out.dtype), out))
        return out
    gate, up = summed.chunk(2, dim=-1)
    act = (F.silu(gate) * up).to(activated.dtype)
    activated.copy_(torch.where(valid[:, None], act, activated))
    return activated


@torch.no_grad()
def w13_act(cache2, cache1, x, a, b, ids, scale, top_k, z_out):
    for expert in range(a.shape[0]):
        pairs = (
            torch.arange(x.shape[0], device=x.device)
            if ids is None
            else torch.where(ids.reshape(-1) == expert)[0]
        )
        values = []
        for s in range(2):
            z = x[pairs // top_k].float() @ a[expert, s].float().T
            delta = z @ b[expert, s].float().T
            n = b.shape[-2]
            base = cache1[pairs, s * n : (s + 1) * n].float()
            update = scale * delta
            values.append(torch.where(update == 0, base, base + update))
            if z_out is not None:
                z_out[pairs, s] = z.to(z_out.dtype)
        cache2[pairs] = (F.silu(values[0]) * values[1]).to(cache2.dtype)
    return cache2


def fused_experts_autograd(x, w13, w2, topk_weights, ids, factors, scaling):
    """FP32 differentiable routed MLP; factors are the six routed leaves or None."""
    scale = torch.as_tensor(scaling, dtype=torch.float32, device=x.device).reshape(1).detach()
    t, h = x.shape
    e, twice_i, _ = w13.shape
    top_k = ids.shape[1]
    flat = ids.reshape(-1)
    valid = (flat >= 0) & (flat < e)
    c1 = torch.zeros((t * top_k, twice_i), dtype=x.dtype, device=x.device)
    for expert in flat[valid].unique().tolist():
        pairs = torch.where(flat == expert)[0]
        c1 = c1.index_copy(
            0, pairs, (x[pairs // top_k].float() @ w13[expert].float().T).to(x.dtype)
        )
    if factors is not None:
        a13, b13 = torch.stack(factors[:2], 1), torch.stack(factors[2:4], 1)
        autograd_apply_(c1, x, a13, b13, ids, scale, top_k, None, None, None, down=False)
    gate, up = c1.float().chunk(2, dim=-1)
    c2 = (F.silu(gate) * up).to(x.dtype)
    rw = topk_weights.reshape(-1)
    c3 = torch.zeros((t * top_k, h), dtype=x.dtype, device=x.device)
    for expert in flat[valid].unique().tolist():
        pairs = torch.where(flat == expert)[0]
        base = (c2[pairs].float() @ w2[expert].float().T) * rw[pairs, None].float()
        c3 = c3.index_copy(0, pairs, base.to(x.dtype))
    if factors is not None:
        a2, b2 = factors[4].unsqueeze(1), factors[5].unsqueeze(1)
        autograd_apply_(c3, c2, a2, b2, ids, scale, top_k, topk_weights, None, None, down=True)
    # Exact-zero empty-slice sums keep the output graph-connected when no route
    # is local (all remote under EP, base-only too), so backward yields zeros.
    anchor = sum(
        (v.reshape(-1)[:0].float().sum() for v in (x, topk_weights) if v.requires_grad),
        torch.zeros((), device=x.device),
    )
    out = c3.view(t, top_k, h).float().sum(dim=1)
    return (out - (anchor.detach() - anchor)).to(x.dtype)
