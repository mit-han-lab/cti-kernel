# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own frozen MoE-block dX p2c oracle and precision gate (2026-10-11).
"""References and the precision gate for the frozen MoE block tests."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# The oracle and the Unsloth-order level U take the selection from a record (ours or a replayed one)
# and compute the weights from the current logits over it, so they evaluate the op's function.
# The oracle's selected logits are RNE of the FP64 logits at every dtype, resolved per solution
# where two values are admissible (the per-solution logit resolution). Gradients come from torch autograd, never from
# the closed form the kernels use; the closed form serves only kappa. Work is grouped by expert so
# the real shapes fit in memory.


def round_bf16_from_fp64(z: torch.Tensor) -> torch.Tensor:
    """RNE to BF16 of FP64 values without double rounding (FP64 -> FP32 round-to-odd -> BF16 RNE)."""

    f = z.to(torch.float32)
    over = f.to(torch.float64).abs() > z.abs()
    f = torch.where(over, torch.nextafter(f, torch.zeros_like(f)), f)
    inexact = f.to(torch.float64) != z
    bits = f.view(torch.int32)
    bits = torch.where(inexact, bits | 1, bits)
    return bits.view(torch.float32).to(torch.bfloat16)


def round_bf16(z: torch.Tensor) -> torch.Tensor:
    return round_bf16_from_fp64(z) if z.dtype == torch.float64 else z.to(torch.bfloat16)


def live_rows(t: int, valid, device) -> torch.Tensor:
    return torch.arange(t, device=device) < (int(valid) if valid is not None else t)


def _groups(sel: torch.Tensor):
    # (expert, token rows, slot) groups of the selection, experts ascending.
    k = sel.shape[1]
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat).tolist() if flat.numel() else []
    start = 0
    for e, n in enumerate(counts):
        if n:
            pairs = order[start : start + n]
            yield e, pairs // k, pairs % k, pairs
            start += n


def selected_logits(x, r, sel):
    """Per selected logit [T, k] FP64: RNE of the FP64 logit, and the other admissible value
    (the allowed logit interval; equal to the first where the interval has one value)."""

    xd, rd = x.double(), r.double()
    z = (xd @ rd.T).gather(1, sel)
    delta = (xd.abs() @ rd.abs().T).gather(1, sel) * 2.0**-16
    zh = round_bf16_from_fp64(z).double()
    lo = round_bf16_from_fp64(z - delta).double()
    hi = round_bf16_from_fp64(z + delta).double()
    return zh, torch.where(zh == lo, hi, lo)


def oracle(
    x,
    r,
    sg,
    w13,
    w2,
    s13,
    s2,
    sel,
    dy,
    valid=None,
    dtype=torch.float64,
    kappa=True,
    zsel=None,
    renormalize=True,
    unselected_grad=True,
):
    """Output, dX and (optionally) their kappa at ``dtype``: FP64 for sampled rows, FP32 (TF32 off)
    for all rows. ``zsel`` [T, k] overrides the selected logits' values (default: RNE of the FP64
    logits). Masked rows are zero. ``renormalize=False`` (norm_topk_prob off) weights the selected
    experts by the softmax over all E BF16-rounded logits; it has no kappa. ``unselected_grad=False``
    is the ablation that drops the router derivative through the unselected experts' logits."""

    t = x.shape[0]
    live = live_rows(t, valid, x.device)
    xr = torch.where(live[:, None], x.to(dtype), 0.0).detach().requires_grad_()
    if zsel is None:
        zsel = selected_logits(xr.detach(), r, sel)[0]
    # the BF16 logit rounding as a straight-through step (dz = dz_hat), as the contract specifies
    zg = (xr @ r.to(dtype).T).gather(1, sel)
    zs = zg + (zsel.to(dtype) - zg).detach()
    w = torch.softmax(zs - zs.max(dim=1, keepdim=True).values.detach(), dim=1)
    if not renormalize:
        if kappa:
            raise ValueError("kappa covers only the renormalized router")
        # Autograd through ALL experts' router rows; the same straight-through BF16 rounding.
        zf = xr @ r.to(dtype).T
        zf = zf + (round_bf16(zf.detach()).to(dtype) - zf).detach()
        if not unselected_grad:
            keep = torch.zeros_like(zf).scatter(1, sel, 1.0)
            zf = zf.detach() + keep * (zf - zf.detach())
        w = torch.softmax(zf - zf.max(dim=1, keepdim=True).values.detach(), dim=1).gather(1, sel)
    p = (w[:, :, None] * routed(xr, sel, w13, w2, dtype)).sum(dim=1)
    sh = s = game = None
    out = p
    if sg is not None:
        nis = s2.shape[1]
        gu_s = xr @ s13.to(dtype).T
        sh = (F.silu(gu_s[:, :nis]) * gu_s[:, nis:]) @ s2.to(dtype).T
        s = xr @ sg.to(dtype).T
        game = torch.sigmoid(s)
        out = p + game * sh
    out = torch.where(live[:, None], out, 0.0)
    dyr = torch.where(live[:, None], dy.to(dtype), 0.0)
    (dx,) = torch.autograd.grad(out, xr, dyr)
    dx = torch.where(live[:, None], dx, 0.0)
    out = out.detach()
    if not kappa:
        return out, dx, None, None
    with torch.no_grad():
        kp, kdx = _kappa(
            xr.detach(),
            r,
            sg,
            w13,
            w2,
            s13,
            s2,
            sel,
            dyr,
            w.detach(),
            p.detach(),
            sh.detach() if sh is not None else None,
            s.detach() if s is not None else None,
            game.detach() if game is not None else None,
            out,
            dx,
            dtype,
        )
    kp = torch.where(live[:, None], kp, 0.0)
    kdx = torch.where(live[:, None], kdx, 0.0)
    return out, dx, kp, kdx


def _kappa(xr, r, sg, w13, w2, s13, s2, sel, dy, w, p, sh, s, game, out, dx, dtype):
    # kappa_P: the frozen precision contract. kappa_dX: the running-error derivation, first-order running error of the
    # level's backward (BF16 at g, u, act, dA, dg, du, dx_pair, s and dX; FP32 elsewhere). The
    # logits count as exact: their ambiguity is resolved instead (the per-solution logit resolution).
    t, h = xr.shape
    k = sel.shape[1]

    def parts(xrows, wa13, wa2, dyrows, wgt, kwgt):
        ni = wa2.shape[1]
        wg, wu = wa13[:ni].to(dtype), wa13[ni:].to(dtype)
        g, u = xrows @ wg.T, xrows @ wu.T
        sig = torch.sigmoid(g)
        silu = g * sig
        d1 = sig * (1 + g * (1 - sig))
        d2 = sig * (1 - sig) * (2 + g * (1 - 2 * sig))
        a = silu * u
        ka = (d1 * u * g).abs() + 2 * a.abs()
        acond_h = ka @ wa2.to(dtype).abs().T  # A_e[h] of kappa_P
        da = dyrows @ wa2.to(dtype)
        q = (da * a).sum(dim=1)
        kq = (da.abs() * (a.abs() + ka)).sum(dim=1)
        delta = wgt[:, None] * da
        kdelta = (wgt[:, None] + kwgt[:, None]) * da.abs()
        dg, du = delta * u * d1, delta * silu
        kdg = (
            (u * d1).abs() * kdelta
            + (delta * d1).abs() * u.abs()
            + (delta * u * d2).abs() * g.abs()
            + dg.abs()
        )
        kdu = silu.abs() * kdelta + (delta * d1).abs() * g.abs() + du.abs()
        kdxp = kdg @ wg.abs() + kdu @ wu.abs() + (dg @ wg + du @ wu).abs()
        return acond_h, kdxp, q, kq

    kp = torch.zeros(t, h, dtype=dtype, device=xr.device)
    kdx = torch.zeros(t, h, dtype=dtype, device=xr.device)
    kq = torch.zeros(t, k, dtype=dtype, device=xr.device)
    for e, tok, slot, _ in _groups(sel):
        wgt = w[tok, slot]
        acond_h, kdxp, _, kqe = parts(xr[tok], w13[e], w2[e], dy[tok], wgt, torch.zeros_like(wgt))
        kp = kp.index_add(0, tok, wgt[:, None] * acond_h)
        kdx = kdx.index_add(0, tok, kdxp)
        kq[tok, slot] = kqe
    kp = kp + (w[:, :, None] * routed(xr, sel, w13, w2, dtype)).abs().sum(dim=1)
    if sg is None:
        kp = kp + out.abs()
        wkq = w * kq
        kdz = w * ((1 - w) * kq + (wkq.sum(dim=1, keepdim=True) - wkq))
        kdx = kdx + torch.einsum("tk,tkh->th", kdz, r.to(dtype).abs()[sel]) + dx.abs()
        return kp, kdx
    g1, s1 = game[:, 0], s[:, 0]
    acond_s, kdxp_s, q_s, kq_s = parts(xr, s13, s2, dy, g1, g1 * (1 - g1) * s1.abs())
    kp = (
        kp
        + p.abs()
        + game * acond_s
        + 3 * game * sh.abs()
        + game * (1 - game) * s.abs() * sh.abs()
        + out.abs()
    )
    wkq = w * kq
    kdz = w * ((1 - w) * kq + (wkq.sum(dim=1, keepdim=True) - wkq))
    kds = g1 * (1 - g1) * kq_s + q_s.abs() * g1 * (1 - g1) * (1 - 2 * g1).abs() * s1.abs()
    kdx = (
        kdx
        + kdxp_s
        + torch.einsum("tk,tkh->th", kdz, r.to(dtype).abs()[sel])
        + kds[:, None] * sg.to(dtype).abs()
        + dx.abs()
    )
    return kp, kdx


def oracle_state(x, r, sg, w13, w2, s13, s2, sel, dy, dtype=torch.float64):
    """The unresolved oracle at ``dtype`` plus what resolution needs (valid rows only)."""

    args = (x, r, sg, w13, w2, s13, s2, sel, dy)
    zh, alt = selected_logits(x, r, sel)
    out, dx, kp, kdx = oracle(*args, dtype=dtype, zsel=zh)
    return {
        "args": args,
        "dtype": dtype,
        "zh": zh,
        "alt": alt,
        "out": out,
        "dx": dx,
        "kp": kp,
        "kdx": kdx,
    }


def resolve(state, cand_out=None, cand_dx=None):
    """the per-solution logit resolution's resolution: per token, among the selected logits with two admissible values,
    the combination whose weights best explain the solution (smallest row L2 error), chosen per view;
    at most 2^4 combinations, the four ambiguous logits of largest weight effect. Returns
    (out, dx, kappa_out, kappa_dx) of the resolved oracle."""

    out, dx, kp, kdx = (state[n].clone() for n in ("out", "dx", "kp", "kdx"))
    zh, alt = state["zh"], state["alt"]
    amb = alt != zh
    rows = amb.any(dim=1).nonzero().flatten()
    if rows.numel() == 0:
        return out, dx, kp, kdx
    x, r, sg, w13, w2, s13, s2, sel, dy = state["args"]
    zr, ar = zh[rows], alt[rows]
    w = torch.softmax(zr - zr.max(dim=1, keepdim=True).values, dim=1)
    eff = torch.where(amb[rows], w * (ar - zr).abs(), -1.0)
    m = min(4, sel.shape[1])
    top = eff.topk(m, dim=1).indices
    top_ok = eff.gather(1, top) >= 0
    best = {"out": torch.full((rows.numel(),), math.inf, dtype=torch.float64, device=x.device)}
    best["dx"] = best["out"].clone()
    for c in range(2**m):
        bits = ((c >> torch.arange(m, device=x.device)) & 1).bool()[None, :] & top_ok
        flip = torch.zeros_like(amb[rows]).scatter(1, top, bits)
        oc, dc, kpc, kdc = oracle(
            x[rows],
            r,
            sg,
            w13,
            w2,
            s13,
            s2,
            sel[rows],
            dy[rows],
            dtype=state["dtype"],
            zsel=torch.where(flip, ar, zr),
        )
        for view, cand, val, kap, dst, kdst in (
            ("out", cand_out, oc, kpc, out, kp),
            ("dx", cand_dx, dc, kdc, dx, kdx),
        ):
            if cand is None:
                continue
            err = (cand[rows].double() - val.double()).pow(2).sum(dim=1)
            better = err < best[view]
            best[view] = torch.where(better, err, best[view])
            dst[rows] = torch.where(better[:, None], val, dst[rows])
            kdst[rows] = torch.where(better[:, None], kap, kdst[rows])
    return out, dx, kp, kdx


def routed(xr, sel, w13, w2, dtype):
    """Per-slot routed expert outputs [T, k, H] at ``dtype``, differentiable in ``xr``."""

    t, h = xr.shape
    ni = w2.shape[2]
    d = torch.zeros(t * sel.shape[1], h, dtype=dtype, device=xr.device)
    for e, tok, _, pairs in _groups(sel):
        gu = xr[tok] @ w13[e].to(dtype).T
        d = d.index_put((pairs,), (F.silu(gu[:, :ni]) * gu[:, ni:]) @ w2[e].to(dtype).T)
    return d.reshape(t, sel.shape[1], h)


def unsloth_combine(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Unsloth's weighted top-k sum of [T, k, H] BF16 expert outputs with [T, k] BF16 weights:
    each product rounded to BF16, then four interleaved FP32 partial sums (slot j into j % 4)
    added as ((a0 + a1) + a2) + a3 and rounded once."""

    parts = [torch.zeros(y.shape[0], y.shape[2], dtype=torch.float32, device=y.device)] * 4
    for j in range(y.shape[1]):
        p = (w[:, j, None].float() * y[:, j].float()).to(torch.bfloat16).float()
        parts[j % 4] = parts[j % 4] + p
    return (((parts[0] + parts[1]) + parts[2]) + parts[3]).to(torch.bfloat16)


def unsloth_routing(x, r, sg, top_k, renormalize=True):
    """HF's Qwen3_5MoeTopKRouter as Unsloth runs it: (ids [T, k] int64, BF16 weights [T, k]) and
    the BF16 shared-gate sigmoid [T, 1] (None without a shared expert)."""

    probs = torch.softmax(F.linear(x, r), dim=-1, dtype=torch.float32)
    top, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        top = top / top.sum(dim=-1, keepdim=True)
    gate = torch.sigmoid(F.linear(x, sg)) if sg is not None else None
    return ids, top.to(torch.bfloat16), gate


def unsloth(x, r, sg, w13, w2, s13, s2, sel, dy, valid=None, renormalize=True, mapping=None):
    """U: the Qwen3.5-MoE block as Unsloth 2026.10.7 trains it (unsloth_zoo's patched sparse block
    and native grouped-GEMM experts) in BF16 torch, differentiated by autograd.

    Router as HF's: BF16 logits, FP32 softmax over all E, the record's top-k, FP32 renormalisation,
    BF16 weights. Experts: BF16 gate/up, BF16 SiLU, BF16 product, BF16 down output. Combine:
    ``unsloth_combine``. Shared expert: separate gate/up linears, sigmoid of the BF16 gate logit,
    BF16 product and a BF16 add after the routed sum. ``mapping`` (global id -> local row or -1)
    gives one EP rank's partial: remote slots add exactly nothing.
    """

    t, h = x.shape
    k = sel.shape[1]
    live = live_rows(t, valid, x.device)
    xb = torch.where(live[:, None], x, 0.0).detach().requires_grad_()
    probs = torch.softmax(F.linear(xb, r), dim=1, dtype=torch.float32).gather(1, sel)
    if renormalize:
        probs = probs / probs.sum(dim=1, keepdim=True)
    w = probs.to(torch.bfloat16)
    y = torch.zeros(t * k, h, dtype=torch.bfloat16, device=x.device)
    for expert, tok, _, pairs in _groups(sel):
        e = expert if mapping is None else int(mapping[expert])
        if e < 0:
            continue
        gate, up = F.linear(xb[tok], w13[e]).chunk(2, dim=-1)
        y = y.index_put((pairs,), F.linear(F.silu(gate) * up, w2[e]))
    out = unsloth_combine(y.view(t, k, h), w)
    if sg is not None:
        nis = s2.shape[1]
        sh = F.linear(F.silu(F.linear(xb, s13[:nis])) * F.linear(xb, s13[nis:]), s2)
        out = out + torch.sigmoid(F.linear(xb, sg)) * sh
    out = torch.where(live[:, None], out, 0.0)
    dyb = torch.where(live[:, None], dy, 0.0).to(torch.bfloat16)
    (dx,) = torch.autograd.grad(out, xb, dyb)
    return out.detach(), torch.where(live[:, None], dx, 0.0)


def admissible_sets(x, r, sel, valid=None) -> torch.Tensor:
    """The allowed routing set check per valid row: each selected expert, at the top of its admissible
    interval, may rank before each dropped expert at the bottom of its interval. [T] bool, masked
    rows True."""

    t = x.shape[0]
    e = r.shape[0]
    xd, rd = x.double(), r.double()
    z = xd @ rd.T
    delta = (xd.abs() @ rd.abs().T) * 2.0**-16
    lo = round_bf16_from_fp64(z - delta).double()
    hi = round_bf16_from_fp64(z + delta).double()
    zh = round_bf16_from_fp64(z).double()
    m = zh.max(dim=1, keepdim=True).values
    eta = 2.0**-22 * (1 + zh.abs().max(dim=1, keepdim=True).values)
    chosen = torch.zeros(t, e, dtype=torch.bool, device=x.device)
    chosen.scatter_(1, sel, True)
    ok = chosen.sum(dim=1) == sel.shape[1]  # distinct ids
    ids = torch.arange(e, device=x.device)
    for j in range(sel.shape[1]):
        s = sel[:, j]
        v1 = hi.gather(1, s[:, None])
        v2 = lo
        before = (
            (v1 > v2)
            | ((v1 == v2) & (s[:, None] < ids[None, :]))
            | ((v1 != v2) & ((v1 - v2).abs() <= eta))
            | ((v1 < m - 80) & (v2 < m - 80))
        )
        ok &= (before | chosen).all(dim=1)
    return ok | ~live_rows(t, valid, x.device)


ABSOLUTE_FLOOR = 2.0**-126  # errors up to this count as exact (the frozen precision contract)
_KAPPA_FLOOR = 2.0**-3  # scale floor: a fraction of the row RMS
_BIAS_FLOOR = 2.0**-6  # bias ignores oracle values below this fraction of the row RMS


def _bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    # BF16 spacing at |v| (8 significand bits), the smallest subnormal's at zero.
    _, ex = torch.frexp(v.abs().double())
    return torch.where(v == 0, 2.0**-133, torch.exp2((ex - 8).double())).clamp_min(2.0**-133)


def metrics(
    cand: torch.Tensor, ref: torch.Tensor, kappa: torch.Tensor, row_se: bool = False
) -> dict:
    """Errors of ``cand`` against its resolved oracle ``ref`` (valid rows): conditioned ULPs at
    max(|ref|, row RMS / 8, kappa), absolute rms, row-relative errors, and (FP64 oracle only) the
    exact-rounding mask and the relative biases beyond correct rounding (away from zero, and towards
    +inf). ``row_se``: the bias SEs clustered by row instead of per element."""

    c = cand.double()
    o = ref.double()
    if o.numel() == 0:
        return {"empty": True, "finite": True}
    err = c - o
    forgiven = err.abs() <= ABSOLUTE_FLOOR
    err = torch.where(forgiven, 0.0, err)
    rms_row = o.pow(2).mean(dim=1, keepdim=True).sqrt()
    scale = torch.maximum(torch.maximum(o.abs(), _KAPPA_FLOOR * rms_row), kappa.double())
    ulp = torch.nan_to_num(err.abs() / _bf16_ulp(scale), nan=math.inf)
    rown, rowe = o.norm(dim=1), err.norm(dim=1)
    rel = torch.where(rown > 0, rowe / rown.clamp_min(1e-300), rowe)
    m = {
        "finite": bool(torch.isfinite(c).all()),
        "max_ulp": ulp.max().item(),
        "rms_error": err.pow(2).mean().sqrt().item(),
        "row_rel_mean": rel.mean().item(),
        "row_rel_max": rel.max().item(),
        "_abs": err.abs(),
        "_scale": scale,
        "_exact": None,
    }
    if ref.dtype == torch.float64:
        rnd = round_bf16_from_fp64(o).double()
        m["_exact"] = (c == rnd) | forgiven
        excess = torch.where(forgiven, 0.0, c - rnd)
        keep = o.abs() >= _BIAS_FLOOR * rms_row
        safe = torch.where(o != 0, o, 1.0)
        # "bias": shift away from zero (excess / o); "dir_bias": shift towards +inf (excess / |o|),
        # which cancels in "bias" between positive and negative oracle values
        for key, rel in (("bias", excess / safe), ("dir_bias", excess / safe.abs())):
            m[key], m[f"{key}_se"] = _mean_se(rel, keep, row_se)
        m["exact_rate"] = m["_exact"].float().mean().item()
    return m


def _mean_se(rel: torch.Tensor, keep: torch.Tensor, row_se: bool) -> tuple[float, float]:
    terms = rel[keep]
    n = terms.numel()
    mean = terms.mean().item() if n else 0.0
    if n < 2:
        return mean, 0.0
    if row_se:
        # dX errors are correlated within a row (one W1 product per pair), so the elements of a row
        # are not independent samples: cluster-robust SE with rows as clusters
        dev = torch.where(keep, rel - mean, 0.0)
        return mean, (dev.sum(dim=1).pow(2).sum().sqrt() / n).item()
    return mean, (terms.std() / math.sqrt(n)).item()


def paired_gate(cand: dict, refs: dict, kind: str = "full", level_name: str = "U") -> list[str]:
    """the frozen precision contract's gate against the finite references, all of them levels (empty list = pass;
    a non-finite level fails):
    max ULP <= max(4, the references' max); rms error and row-relative mean <= 1.05x the best;
    max ULP <= max(best, 1); the pointwise gate (no element over every reference's error there
    by more than 1 conditioned ULP); the row cap (4x the level's row-relative max). Full views add the
    McNemar exactness test against the most exact reference (3 sigma) and the two bias checks, away
    from zero and towards +inf (each within max(2^-14, 6 SE, 1.5x the best reference's), or no
    larger than the level's)."""

    if cand.get("empty"):
        return []
    if not cand["finite"]:
        return ["non-finite"]
    refs = {n: r for n, r in refs.items() if r["finite"]}  # a non-finite reference is not valid
    if level_name not in refs:
        return [f"reference {level_name} is non-finite"]
    slack = ABSOLUTE_FLOOR
    fails = []
    floor = max([4.0] + [r["max_ulp"] for r in refs.values()])
    if cand["max_ulp"] > floor:
        fails.append(f"max_ulp {cand['max_ulp']:.3g} over the hard floor {floor:.3g}")
    best_max = min(r["max_ulp"] for r in refs.values())
    if cand["max_ulp"] > max(best_max, 1.0):
        fails.append(f"max_ulp {cand['max_ulp']:.3g} > max(best {best_max:.3g}, 1)")
    for key in ("rms_error", "row_rel_mean"):
        best = min(r[key] for r in refs.values())
        if cand[key] > 1.05 * best + slack:
            fails.append(f"{key} {cand[key]:.3g} > 1.05 x {best:.3g}")
    envelope = torch.stack([r["_abs"] for r in refs.values()]).amax(dim=0)
    units = (cand["_abs"] - envelope).clamp_min(0) / _bf16_ulp(cand["_scale"])
    if float(units.max()) > 1.0:
        fails.append(
            f"pointwise: {int((units > 1.0).sum())} elements over the references by > 1 ULP "
            f"(max {float(units.max()):.3g})"
        )
    cap = 4.0 * refs[level_name]["row_rel_max"]
    if cand["row_rel_max"] > cap + slack:
        fails.append(
            f"row_rel_max {cand['row_rel_max']:.3g} > 4 x {level_name} {refs[level_name]['row_rel_max']:.3g}"
        )
    if kind == "full":
        name = max(refs, key=lambda n: refs[n]["exact_rate"])
        worse = int((~cand["_exact"] & refs[name]["_exact"]).sum())
        better = int((cand["_exact"] & ~refs[name]["_exact"]).sum())
        if worse > better + 3.0 * math.sqrt(worse + better):
            fails.append(f"exactness below {name} ({worse} worse, {better} better)")
        for key in ("bias", "dir_bias"):
            best_bias = min((r[key] for r in refs.values()), key=abs)
            allowed = max(2.0**-14, 6.0 * cand[f"{key}_se"], 1.5 * abs(best_bias))
            if abs(cand[key]) > allowed and abs(cand[key]) > abs(refs[level_name][key]):
                fails.append(
                    f"{key} {cand[key]:.3g} > {allowed:.3g} and > {level_name}'s {refs[level_name][key]:.3g}"
                )
    return fails


def public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def make_weights(
    h=64,
    e=16,
    ni=32,
    nis=None,
    seed=0,
    device="cpu",
    logit_std=2.0,
    sg_scale=1.0,
    shared_scale=1.0,
    has_shared=True,
):
    nis = ni if nis is None else nis
    g = torch.Generator().manual_seed(seed)

    def rn(shape, scale):
        return (torch.randn(shape, generator=g) * scale).to(device=device, dtype=torch.bfloat16)

    return {
        "r": rn((e, h), logit_std / math.sqrt(h)),
        "sg": rn((1, h), sg_scale / math.sqrt(h)) if has_shared else None,
        "w13": rn((e, 2 * ni, h), 1 / math.sqrt(h)),
        "w2": rn((e, h, ni), 1 / math.sqrt(ni)),
        "s13": rn((2 * nis, h), shared_scale / math.sqrt(h)) if has_shared else None,
        "s2": rn((h, nis), 1 / math.sqrt(nis)) if has_shared else None,
    }


def make_x(t, h=64, seed=1, device="cpu"):
    return torch.randn(t, h, generator=torch.Generator().manual_seed(seed)).to(
        device=device, dtype=torch.bfloat16
    )


def shard_weights(wt, ep, rank):
    """Contiguous expert placement; shared gate and up rows are sliced separately."""
    e = wt["w13"].shape[0]
    count = e // ep
    start = rank * count
    shard = dict(wt)
    shard["w13"] = wt["w13"][start : start + count].contiguous()
    shard["w2"] = wt["w2"][start : start + count].contiguous()
    if wt["s2"] is not None:
        ni = wt["s2"].shape[1]
        width = ni // ep
        lo = rank * width
        shard["s13"] = torch.cat((wt["s13"][lo : lo + width], wt["s13"][ni + lo : ni + lo + width]))
        shard["s2"] = wt["s2"][:, lo : lo + width].contiguous()
    mapping = torch.full((e,), -1, dtype=torch.int32, device=wt["r"].device)
    mapping[start : start + count] = torch.arange(count, dtype=torch.int32, device=mapping.device)
    return shard, mapping


def op_grad(x, wt, dy, top_k=4, **kwargs):
    from ctikernels.moe import moe_block

    xr = x.detach().clone().requires_grad_()
    out, rec = moe_block(xr, *wt.values(), top_k=top_k, **kwargs)
    dx = torch.autograd.grad(out, xr, dy)[0]
    return out.detach(), rec, dx


def unsloth_ranks(x, wt, sel, dy, ep=1):
    """U for ``ep`` contiguous EP ranks: each rank's partial in Unsloth's order, the partials
    summed in FP32 and rounded once (as the caller's all-reduce of BF16 partials)."""

    if ep == 1:
        return unsloth(x, *wt.values(), sel, dy)
    shards = (shard_weights(wt, ep, rank) for rank in range(ep))
    parts = [unsloth(x, *s.values(), sel, dy, mapping=m) for s, m in shards]
    return tuple(sum(p[i].float() for p in parts).bfloat16() for i in range(2))


def check_gate(x, wt, dy, out, rec, dx, dtype=torch.float64, row_se=False, ep=1):
    """Unchanged paired gate, resolving admissible rounded logits per candidate."""
    state = oracle_state(x, *wt.values(), rec.long(), dy, dtype=dtype)
    lv = unsloth_ranks(x, wt, rec.long(), dy, ep)
    rc, rl = resolve(state, out, dx), resolve(state, *lv)
    for i, candidate in enumerate((out, dx)):
        mc = metrics(candidate, rc[i], rc[i + 2], row_se=row_se)
        ml = metrics(lv[i], rl[i], rl[i + 2], row_se=row_se)
        fails = paired_gate(mc, {"U": ml}, kind="full" if dtype == torch.float64 else "coarse")
        assert not fails, ("out" if i == 0 else "dx", fails)


def check_planted_bias(x, wt, dy, top_k=4):
    """Ported planted-bias regression, with both element and row-clustered SE."""
    _, rec, dx = op_grad(x, wt, dy, top_k=top_k)
    st = oracle_state(x, *wt.values(), rec.long(), dy)
    lv = unsloth(x, *wt.values(), rec.long(), dy)
    rl = resolve(st, None, lv[1])
    frac = 0.5 if x.shape[1] >= 1024 else 1.0
    pick = (torch.rand(dx.shape, generator=torch.Generator().manual_seed(61)) < frac).to(x.device)
    pick &= dx != 0
    d = dx.double()
    outward = torch.where(pick, d + _bf16_ulp(d) * d.sign(), d).bfloat16()
    positive = torch.where(pick, d + _bf16_ulp(d), d).bfloat16()
    for candidate, check in ((dx, None), (outward, "bias"), (positive, "dir_bias")):
        rc = resolve(st, None, candidate)
        for row_se in (False, True):
            ml = metrics(lv[1], rl[1], rl[3], row_se=row_se)
            mc = metrics(candidate, rc[1], rc[3], row_se=row_se)
            fails = paired_gate(mc, {"U": ml})
            if check is None:
                assert not fails, fails
            else:
                assert any(f.startswith(check + " ") for f in fails), (check, fails)


# MoE inference as vLLM 0.31.0 computes it on our devices (FlashInfer CUTLASS fused MoE;
# DeepGEMM / FlashInfer for block-FP8): unweighted BF16 FC2 rows per route, combined by the
# finalize kernel (cutlass_fused_moe_kernels.cuh finalizeMoeRoutingKernel: an FP32 FMA chain
# in route order from zero, one rounding), then the gated shared expert in BF16 torch ops
# (qwen2_moe.py:116-121) added in BF16 (moe_runner.py:786).


def fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """FP32 fused multiply-add a * b + c with one rounding (as FFMA): the FP64 product is
    exact, the FP64 sum's rounding error is recovered by two-sum, and a tie of the sum
    between two FP32 values is broken by that error's sign."""
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bp = s - cd
    err = (p - bp) + (cd - (s - bp))
    lo = s.float()
    other = torch.where(s > lo.double(), torch.nextafter(lo, lo + 1), torch.nextafter(lo, lo - 1))
    tie = (s != lo.double()) & (s == (lo.double() + other.double()) / 2) & (err != 0)
    toward = (err > 0) == (other > lo)
    return torch.where(tie & toward, other, lo)


def combine_reference(y: torch.Tensor, w: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The finalize: y [T, k, H] unweighted BF16 rows, w [T, k] FP32 route weights, valid
    [T, k]; acc = fma32(w_j, y_j, acc) for j = 0..k-1 (invalid routes skipped), BF16 once."""
    acc = torch.zeros(y.shape[0], y.shape[2], dtype=torch.float32, device=y.device)
    for j in range(y.shape[1]):
        step = fma32(w[:, j : j + 1].expand_as(acc), y[:, j].float(), acc)
        acc = torch.where(valid[:, j : j + 1], step, acc)
    return acc.bfloat16()


def shared_join_reference(
    routed: torch.Tensor, shared: torch.Tensor, gate_logit: torch.Tensor
) -> torch.Tensor:
    """vLLM's gated shared expert join on BF16 tensors: sigmoid(gate) * shared, then + routed,
    each a BF16 torch op (FP32 math, one rounding)."""
    return F.sigmoid(gate_logit) * shared + routed


def vllm_topk_softmax(logits, top_k, renormalize=True, selected=None):
    """vLLM 0.31's softmax top-k routing (topk_softmax_kernels.cu, topkGating, whose warp
    order a power-of-two E up to 512 follows) on ``logits`` [T, E] (their dtype sets the
    16-byte load width): (weights FP32
    [T, k], experts int32 [T, k] in pick order, every expert's probability [T, E]), in its
    arithmetic order (see ``ctikernels.triton.topk_softmax``). ``selected`` replays given
    experts (their probabilities, renormalised in slot order)."""
    z = logits.float()
    t, e = z.shape
    ex = torch.exp(z - z.max(dim=1, keepdim=True).values)
    if e <= 512 and not e & (e - 1):
        elts = min(16, logits.element_size() * e) // logits.element_size()
        vpt = elts * max(1, e // (elts * 32))
        tpr, ldg = e // vpt, vpt // elts
        lanes = torch.zeros(t, tpr, device=z.device)
        g = ex.view(t, ldg, tpr, elts)
        for ld in range(ldg):
            for i in range(elts):
                lanes = lanes + g[:, ld, :, i]
        while lanes.shape[1] > 1:  # the xor butterfly as lane 0 ends it
            m = lanes.shape[1] // 2
            lanes = lanes[:, :m] + lanes[:, m:]
        row_sum = lanes[:, 0]
    else:  # ctikernels.moe.route's order: pairwise halving over the padded power-of-two tile,
        # or (streamed: E past 4096 or k past 32) over 1024-expert chunks added in order
        width = 1 << (e - 1).bit_length()
        stream = width > 4096 or top_k > 32
        chunk = 1024 if stream else width
        padded = torch.zeros(t, -(-e // chunk) * chunk, device=z.device)
        padded[:, :e] = ex
        row_sum = torch.zeros(t, device=z.device)
        for c in padded.split(chunk, dim=1):
            part = c
            while part.shape[1] > 1:
                part = part[:, : part.shape[1] // 2] + part[:, part.shape[1] // 2 :]
            row_sum = row_sum + part[:, 0]
    probs = ex * (1.0 / row_sum)[:, None]
    probs = torch.where(probs.isfinite(), probs, 0.0)  # vLLM's clamp of nonfinite values
    if selected is not None:
        ids = selected.long()
        values = probs.gather(1, ids)
    else:
        p, every = probs.clone(), torch.arange(e, device=z.device).expand(t, e)
        picks, vals = [], []
        for _ in range(top_k):
            best = p.max(dim=1).values
            pick = torch.where(p == best[:, None], every, e).min(dim=1).values
            picks.append(pick)
            vals.append(best)
            p = torch.where(every == pick[:, None], -1.0, p)
        ids, values = torch.stack(picks, 1), torch.stack(vals, 1)
    if renormalize:
        total = torch.zeros(t, device=z.device)
        for k in range(top_k):
            total = total + values[:, k]
        values = values * (1.0 / torch.where(total > 0, total, 1.0))[:, None]
    return values, ids.int(), probs


def _gemm(a, w):
    """BF16 a . w^T with FP32 accumulation, rounded once."""
    return (a.float() @ w.float().T).bfloat16()


def vllm_rows(x, w13, w2, topk_ids, expert_map=None):
    """Per-route unweighted FC2 rows [T, k, H] BF16 and their validity, as vLLM's BF16 MoE
    on sm_12x (FlashInfer CUTLASS, the SM80 fused gated GEMM: fused_moe_kernel_routine.cuh
    steps 3-4.1): FC1 kept in FP32, SiLU(gate) * up in FP32 rounded once to BF16, BF16 FC2
    (FP32 accumulation). GEMMs here accumulate in FP32 and round once (torch's BF16 cuBLAS may
    reduce split-K partials in BF16); a kernel's accumulation order may differ in the last bit."""
    t, k = topk_ids.shape
    e = w13.shape[0]
    ids = topk_ids.long()
    if expert_map is not None:
        ids = torch.where(ids >= 0, expert_map.long()[ids.clamp_min(0)], -1)
    valid = (ids >= 0) & (ids < e)
    y = torch.zeros(t, k, w2.shape[1], dtype=torch.bfloat16, device=x.device)
    i = w2.shape[2]
    for expert in torch.unique(ids[valid]).tolist():
        tok, slot = torch.nonzero(ids == expert, as_tuple=True)
        gu = x[tok].float() @ w13[expert].float().T
        act = (F.silu(gu[:, :i]) * gu[:, i:]).bfloat16()
        y[tok, slot] = _gemm(act, w2[expert])
    return y, valid


def vllm_infer_reference(x, w13, w2, topk_weights, topk_ids, *, shared=None, expert_map=None):
    """vLLM 0.31's MoE block output [T, H] BF16. ``shared`` = (w_gate_up [2SI, H], w_down
    [H, SI], w_gate [1, H]) or None; ``topk_weights`` FP32 as routed."""
    y, valid = vllm_rows(x, w13, w2, topk_ids, expert_map)
    routed = combine_reference(y, topk_weights.float(), valid)
    if shared is None:
        return routed
    w_gate_up, w_down, w_gate = shared
    si = w_down.shape[1]
    # vLLM's dense MLP: BF16 gate/up; SiluAndMul native under Inductor (custom ops off by
    # default), so SiLU * up in FP32 rounded once
    gu = _gemm(x, w_gate_up).float()
    s_y = _gemm((F.silu(gu[:, :si]) * gu[:, si:]).bfloat16(), w_down)
    return shared_join_reference(routed, s_y, _gemm(x, w_gate))
