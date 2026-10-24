# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Slow, static-shape PyTorch reference; float64 is an oracle option only."""

import torch

from .._common import Quantized, RouterOutput, decode_e2m1, decode_mxfp8, encode_ue8m0


def quantize(x):
    # frexp on amax itself avoids division underflow and logarithmic boundary error.
    groups = x.float().reshape(x.shape[0], x.shape[1] // 32, 32)
    amax = groups.abs().amax(dim=-1)
    fraction, exponent = torch.frexp(amax)
    scales = encode_ue8m0(exponent - 9 + (fraction > 0.875).int())
    scales = torch.where(amax == 0, 127, scales).to(torch.uint8)
    normalized = torch.ldexp(groups, (127 - scales.int()).unsqueeze(-1))
    normalized = torch.where((amax == 0).unsqueeze(-1), 0.0, normalized)
    values = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
    return Quantized(values, scales)


def scores_from_logits(z):
    # sqrt(exp(z)) can be representable even when exp(z) underflows. Below
    # -20 the relative correction to exp(z/2) is < exp(-20)/4 (< one FP32 ulp).
    ordinary = (z.clamp_min(0) + torch.log1p(torch.exp(-z.abs()))).sqrt()
    return torch.where(z < -20, torch.exp(z * 0.5), ordinary)


def weights_from_logits(z):
    softplus = z.clamp_min(0) + torch.log1p(torch.exp(-z.abs()))
    log_score = 0.5 * torch.where(z < -20, z, softplus.log())
    relative = torch.exp(log_score - log_score.amax(dim=-1, keepdim=True))
    return 1.5 * relative / relative.sum(dim=-1, keepdim=True)


def router(
    x, gate, bias, expert_map, token_ids=None, tid2eid=None, valid_mask=None, dtype=torch.float32
):
    z = x.to(dtype) @ gate.to(dtype).T
    scores = scores_from_logits(z)
    valid = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    if valid_mask is not None:
        valid = valid & valid_mask
    if tid2eid is None:
        # Stable sort defines ascending-ID ties without perturbing the scores.
        global_ids = torch.argsort(scores + bias, descending=True, stable=True)[:, :6]
    else:
        valid = valid & (token_ids >= 0) & (token_ids < tid2eid.shape[0])
        global_ids = tid2eid[token_ids.clamp(0, tid2eid.shape[0] - 1)].long()
        valid = valid & ((global_ids >= 0) & (global_ids < gate.shape[0])).all(dim=-1)
    safe = global_ids.clamp(0, gate.shape[0] - 1)
    weights = weights_from_logits(z.gather(1, safe))
    local = expert_map[safe]
    keep = valid[:, None] & (local >= 0)
    return RouterOutput(
        torch.where(keep, local, -1).int(),
        torch.where(keep, weights, 0.0).float(),
        torch.where(valid[:, None], global_ids, -1).int(),
    )


def routed(
    x,
    ids,
    weights,
    native,
    *,
    dtype=torch.float32,
    naive=False,
    out=None,
    trace=None,
    unquantized=False,
    return_accumulator=False,
):
    """No dynamic indices/shapes or host readback. Decode one expert at a time.

    trace, when supplied by a test, receives stage operands/results before rounding.
    Never retain this trace for production-sized layers.
    """
    a = x.to(dtype) if naive or unquantized else decode_mxfp8(quantize(x), dtype)
    slots = [torch.zeros_like(x, dtype=dtype) for _ in range(6)]
    for expert in range(native.w13.shape[0]):
        w13 = decode_e2m1(native.w13[expert], native.w13_scales[expert], dtype)
        w2 = decode_e2m1(native.w2[expert], native.w2_scales[expert], dtype)
        if naive:
            w13 = w13.bfloat16().to(dtype)
            w2 = w2.bfloat16().to(dtype)
        fc1 = a @ w13.T
        gate, up = fc1.chunk(2, dim=-1)
        gate, up = gate.clamp_max(10), up.clamp(-10, 10)
        activation = gate * torch.sigmoid(gate) * up
        for slot in range(6):
            keep = ids[:, slot] == expert
            route = torch.where(keep, weights[:, slot], 0.0).to(dtype)
            h = activation * route[:, None]
            hb = h.bfloat16()
            qh = None if naive or unquantized else quantize(hb)
            b = h if unquantized else hb.to(dtype) if naive else decode_mxfp8(qh, dtype)
            fc2 = b @ w2.T
            slots[slot] = slots[slot] + torch.where(keep[:, None], fc2, 0.0)
            if trace is not None:
                trace.append(
                    {
                        "expert": expert,
                        "slot": slot,
                        "a": a,
                        "w13": w13,
                        "fc1": fc1,
                        "h": h,
                        "hb": hb,
                        "qh": qh,
                        "b": b,
                        "w2": w2,
                        "fc2": fc2,
                    }
                )
    combined = torch.zeros_like(x, dtype=dtype)
    for partial in slots:
        combined = combined + partial
    has_local = ((ids >= 0) & (ids < native.w13.shape[0])).any(dim=-1, keepdim=True)
    combined = torch.where(has_local, combined, 0.0)
    if trace is not None:
        trace.append({"combined": combined, "slots": slots})
    if return_accumulator:
        return combined
    if out is None:
        out = torch.empty_like(x)
    out.copy_(combined)
    return out
