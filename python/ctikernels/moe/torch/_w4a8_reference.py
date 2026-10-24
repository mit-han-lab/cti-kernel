# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FP32 dequantization reference for W4A8 (MXFP4 x MXFP8) routed experts; any device."""

import torch
import torch.nn.functional as F

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_mxfp4(packed, scales):
    """[..., N, K/2] E2M1 pairs (low nibble first) and [..., N, K/32] UE8M0 -> FP32 [..., N, K]."""
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    table = torch.tensor(_E2M1, device=packed.device)
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return values * _pow2(scales).repeat_interleave(32, -1)


def _pow2(scales):
    return torch.exp2(scales.view(torch.uint8).float() - 127)


def quantize_mxfp8(x):
    """BF16 [..., K] -> (E4M3 [..., K], UE8M0 uint8 [..., K/32]): per 32 values, the smallest
    power of two mapping the absmax to <= 448 (2^0 for all-zero groups), RNE values."""
    x = x.to(torch.bfloat16)
    groups = x.unflatten(-1, (-1, 32))
    peak = (groups.view(torch.int16).int() & 0x7FFF).amax(-1)
    sf = ((peak >> 7) - 8 + ((peak & 127) > 96).int()).clamp(0, 254)
    sf = torch.where(peak == 0, 127, sf)
    scaled = groups.float() * torch.exp2(127 - sf.float()).unsqueeze(-1)
    q = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    return q, sf.to(torch.uint8)


def dequant_mxfp8(q, scales):
    return q.float() * _pow2(scales).repeat_interleave(32, -1)


@torch.no_grad()
def mxfp4_experts(
    x,
    w13,
    s13,
    w2,
    s2,
    topk_weights,
    topk_ids,
    *,
    expert_map=None,
    swiglu_limit=None,
    exact=False,
    lora=None,
    scaling=1.0,
):
    """Same quantization points as the kernels, FP32 GEMMs, slot-ordered sum, one BF16 rounding.
    ``exact``: activations stay unquantized (FP32 truth over the dequantized weights).

    ``lora`` = (gate_a, up_a, gate_b, up_b, down_a, down_b), each [E_local, ...]: FP32 deltas
    of the BF16 input join gate/up before the clamp, and of the (BF16) weighted activation
    join each route's down projection before the sum."""
    t, h = x.shape
    inter = w13.shape[1] // 2
    qdq = (lambda a: a.float()) if exact else (lambda a: dequant_mxfp8(*quantize_mxfp8(a)))
    xq = qdq(x)
    ids = topk_ids.long()
    if expert_map is not None:
        valid = ids >= 0
        ids = torch.where(valid, expert_map.long()[ids.clamp(min=0)], -1)
    routes = torch.zeros(t, ids.shape[1], h, dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        if e < 0:
            continue
        rows, slots = (ids == e).nonzero(as_tuple=True)
        gu = xq[rows] @ dequant_mxfp4(w13[e], s13[e]).T
        u, g = gu[:, :inter], gu[:, inter:]  # w13 rows: up, then gate
        if lora is not None:
            ga, ua, gb, ub = (f[e].float() for f in lora[:4])
            xr = x[rows].float()
            g = g + scaling * ((xr @ ga.T) @ gb.T)
            u = u + scaling * ((xr @ ua.T) @ ub.T)
        if swiglu_limit is not None:
            g = g.clamp(max=swiglu_limit)
            u = u.clamp(-swiglu_limit, swiglu_limit)
        act = F.silu(g) * u * topk_weights[rows, slots, None]
        act = act if exact else act.to(torch.bfloat16)
        routes[rows, slots] = qdq(act) @ dequant_mxfp4(w2[e], s2[e]).T
        if lora is not None:
            da, db = (f[e].float() for f in lora[4:])
            routes[rows, slots] += scaling * ((act.float() @ da.T) @ db.T)
    out = routes[:, 0]
    for slot in range(1, ids.shape[1]):
        out = out + routes[:, slot]
    return out if exact else out.to(torch.bfloat16)
