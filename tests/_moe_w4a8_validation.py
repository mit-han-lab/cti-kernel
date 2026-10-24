# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Test/benchmark diagnostics only: float64 oracles and host-side assertions.

Never call from an inference launch or graph capture. These helpers intentionally
synchronize when converting metrics to Python values.
"""

import torch

from ctikernels.moe._common import Quantized, decode_mxfp8, decode_ue8m0
from ctikernels.moe.torch._w4a8_tiled_reference import routed

U32 = 2.0**-24
TINY = 2.0**-126


def gemm_bound(a, b):
    return (2 * a.shape[-1] * U32) * (a.double().abs() @ b.double().abs().T) + TINY


def bf16_ulp(x):
    exponent = torch.floor(torch.log2(x.double().abs().clamp_min(2.0**-133))) - 7
    return torch.exp2(exponent).clamp_min(2.0**-133)


def metrics(actual, oracle):
    difference = actual.double() - oracle.double()
    if not torch.isfinite(difference).all():
        raise ValueError("non-finite precision result")
    norm = torch.linalg.vector_norm(oracle.double())
    return {
        "rel_l2": float(torch.linalg.vector_norm(difference) / norm.clamp_min(1e-300)),
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
    }


def check_stages(x, ids, weights, native, *, observed=None, actual_out=None):
    """Assert fixed stage gates; optional Triton trace replaces measured stages.

    Diagnostic only: gathers sorted rows and synchronizes. Never used in launches.
    Pass a bounded token subset; native should contain only a few local experts.
    """
    t32, t64 = [], []
    y32 = routed(x, ids, weights, native, trace=t32)
    y64 = routed(x, ids, weights, native, dtype=torch.float64, trace=t64)
    if observed is not None:
        assert actual_out is not None
        y32 = actual_out[: len(x)]
        observed_q = observed["qh"]
        if observed["order"] is not None:
            order = observed["order"].long()
            inverse = torch.zeros(len(x) * 6, dtype=torch.int64, device=x.device)
            keep = (order >= 0) & (order < len(x) * 6)
            inverse[order[keep]] = torch.arange(len(order), device=x.device)[keep]
            observed_q = Quantized(
                observed_q.values.view(torch.uint8)[inverse].view(torch.float8_e4m3fn),
                observed_q.scales[inverse],
            )
        for a in t32[:-1]:
            slot, expert = a["slot"], a["expert"]
            pair = torch.arange(len(x), device=x.device) * 6 + slot
            keep = (ids[:, slot] == expert)[:, None]
            fc1 = torch.cat((observed["gate"][pair], observed["up"][pair]), -1)
            a["fc1"] = torch.where(keep, fc1, a["fc1"])
            a["qh"] = Quantized(
                torch.where(
                    keep,
                    observed_q.values.view(torch.uint8)[pair],
                    a["qh"].values.view(torch.uint8),
                ).view(torch.float8_e4m3fn),
                torch.where(keep, observed_q.scales[pair], a["qh"].scales),
            )
            a["b"] = decode_mxfp8(a["qh"])
            a["fc2"] = torch.where(keep, observed["partials"][pair], a["fc2"])
    output_bound = torch.zeros_like(x, dtype=torch.float64)
    for a, b in zip(t32[:-1], t64[:-1], strict=True):
        bound1 = gemm_bound(a["a"], a["w13"])
        exact1 = a["a"].double() @ a["w13"].double().T
        assert torch.all((a["fc1"].double() - exact1).abs() <= bound1)
        exact2 = a["b"].double() @ a["w2"].double().T
        bound2 = gemm_bound(a["b"], a["w2"])
        assert torch.all((a["fc2"].double() - exact2).abs() <= bound2)
        bg, bu = bound1.chunk(2, -1)
        g = b["fc1"].chunk(2, -1)[0].clamp_max(10)
        silu = g * torch.sigmoid(g)
        route = torch.where(ids[:, a["slot"]] == a["expert"], weights[:, a["slot"]], 0).double()[
            :, None
        ]
        bh = route.abs() * (11 * bg + silu.abs() * bu + 1.1 * bg * bu) + 8 * U32 * b[
            "h"
        ].abs().clamp_min(TINY)
        low = (b["h"] - bh).bfloat16().double()
        high = (b["h"] + bh).bfloat16().double()
        q32, q64 = a["qh"], b["qh"]
        v32, v64 = decode_mxfp8(q32, torch.float64), decode_mxfp8(q64, torch.float64)
        changed = v32 != v64
        scale64 = decode_ue8m0(q64.scales, torch.float64).repeat_interleave(32, -1)
        # Adjacent finite E4M3 codes, with sign symmetry, define RNE cells.
        code = (q64.values.view(torch.uint8) & 127).long()
        grid = (
            torch.arange(127, device=x.device, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
        )
        center = grid[code]
        below, above = grid[(code - 1).clamp_min(0)], grid[(code + 1).clamp_max(126)]
        negative = torch.signbit(b["hb"])
        lower_mid = torch.where(negative, -(center + above) / 2, (center + below) / 2) * scale64
        upper_mid = torch.where(negative, -(center + below) / 2, (center + above) / 2) * scale64
        midpoint = ((low <= lower_mid) & (high >= lower_mid)) | (
            (low <= upper_mid) & (high >= upper_mid)
        )
        amin = torch.where((low <= 0) & (high >= 0), 0.0, torch.minimum(low.abs(), high.abs()))
        amax = torch.maximum(low.abs(), high.abs())
        group_lo = amin.reshape(x.shape[0], -1, 32).amax(-1)
        group_hi = amax.reshape(x.shape[0], -1, 32).amax(-1)
        boundary = decode_ue8m0(q64.scales, torch.float64) * 448
        crosses = ((group_lo <= boundary) & (group_hi >= boundary)) | (
            (group_lo <= boundary / 2) & (group_hi >= boundary / 2)
        )
        assert torch.all(~changed | midpoint | crosses.repeat_interleave(32, -1))
        scale32 = decode_ue8m0(q32.scales, torch.float64).repeat_interleave(32, -1)
        coarse = torch.maximum(scale32, scale64)
        magnitude = torch.maximum(v32.abs(), v64.abs()) / coarse
        step = (
            torch.exp2(torch.floor(torch.log2(magnitude.clamp_min(2.0**-9))) - 3).clamp_min(2.0**-9)
            * coarse
        )
        assert torch.all((v32 - v64).abs() <= step)
        delta = (v32 - v64).abs() @ b["w2"].abs().T
        output_bound += torch.where(
            (ids[:, a["slot"]] == a["expert"])[:, None], bound2 + delta, 0.0
        )
    combined = t64[-1]["combined"]
    output_bound += (6 * U32 / (1 - 6 * U32)) * sum(s.abs() for s in t64[-1]["slots"])
    assert torch.all((y32.double() - y64.double()).abs() <= bf16_ulp(y64) + output_bound)
    return metrics(y32, y64), combined, output_bound
