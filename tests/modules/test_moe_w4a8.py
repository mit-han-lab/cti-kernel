# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""W4A8 MoE reference formats and argument checks (CPU)."""

import pytest
import torch

from ctikernels.moe import MXFP4Experts, pack_mxfp4_experts
from ctikernels.moe.torch._w4a8_reference import dequant_mxfp4, dequant_mxfp8, quantize_mxfp8


def _e4m3_values():
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return codes[codes.isfinite()].unique()


def test_mxfp8_is_nearest_e4m3_under_smallest_pow2_scale():
    torch.manual_seed(0)
    x = torch.randn(64, 32) * torch.exp2(torch.randint(-140, 120, (64, 1)).float())
    x[0] = 0
    x[1, :] = 448 * 2.0**-3  # exactly at the E4M3 max: scale 2^-3, no bump
    x[2, 0] = 449 * 2.0**-3  # just above: one more exponent
    x = x.to(torch.bfloat16)
    q, s = quantize_mxfp8(x)
    amax = x.float().abs().amax(-1)
    expect = torch.where(
        amax > 0, torch.ceil(torch.log2(amax / 448)).clamp(-127, 127) + 127, 127
    ).to(torch.uint8)
    assert torch.equal(s.flatten(), expect)
    scaled = x.float() * torch.exp2(127 - s.float()).repeat_interleave(32, -1)
    grid = _e4m3_values()
    nearest = (scaled.unsqueeze(-1) - grid).abs().min(-1).values
    assert torch.equal((q.float() - scaled).abs(), nearest)
    assert dequant_mxfp8(q, s).isfinite().all()


def test_mxfp4_nibble_order_and_scale():
    packed = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8).repeat(1, 8)
    scales = torch.tensor([[128]], dtype=torch.uint8)
    w = dequant_mxfp4(packed, scales)
    assert w.shape == (1, 32)
    assert w[0, :4].tolist() == [1.0, 2.0, -1.0, -12.0]


def test_pack_and_constructor_checks():
    e, i, h = 2, 128, 256
    gate = torch.zeros(e, i, h // 2, dtype=torch.uint8)
    gs = torch.full((e, i, h // 32), 127, dtype=torch.uint8)
    down = torch.zeros(e, h, i // 2, dtype=torch.uint8)
    ds = torch.full((e, h, i // 32), 127, dtype=torch.uint8)
    up, us = gate + 1, gs + 1
    w13, s13, w2, s2 = pack_mxfp4_experts(gate, gs, up, us, down, ds)
    assert w13.shape == (e, 2 * i, h // 2) and s13.shape == (e, 2 * i, h // 32)
    assert torch.equal(w13[:, :i], up) and torch.equal(s13[:, i:], gs)  # canonical: up first
    with pytest.raises(ValueError, match="gate_scale"):
        pack_mxfp4_experts(gate, gs[:, :, :1], gate, gs, down, ds)
    with pytest.raises(ValueError, match="CUDA"):
        MXFP4Experts(w13, s13, w2, s2, scales_layout="canonical")
    with pytest.raises(ValueError, match="multiples of 128"):
        MXFP4Experts(w13[:, :, :32], s13[:, :, :2], w2, s2, scales_layout="canonical")


def test_interleave_scales_in_place():
    from ctikernels.moe._common import deinterleave_scales, interleave_scales_

    s = torch.randint(0, 256, (2, 256, 8), dtype=torch.uint8)
    t = interleave_scales_(s.clone())
    n, g = 200, 5  # byte ((n//128)*(K/128) + g//4)*512 + (n%32)*16 + (n%128)//32*4 + g%4
    assert t[1].flatten()[(1 * 2 + 1) * 512 + 8 * 16 + 2 * 4 + 1] == s[1, n, g]
    assert torch.equal(deinterleave_scales(t), s)
    rows = torch.randperm(256)
    assert torch.equal(deinterleave_scales(interleave_scales_(s.clone(), rows)), s[:, rows])
    with pytest.raises(ValueError, match="N % 128"):
        interleave_scales_(s[:, :64].clone())


def test_reference_lora_terms():
    from ctikernels.moe.torch._w4a8_reference import mxfp4_experts

    torch.manual_seed(7)
    h, i, e, r = 64, 32, 3, 8
    w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)
    s13 = torch.randint(119, 123, (e, 2 * i, h // 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)
    s2 = torch.randint(119, 123, (e, h, i // 32), dtype=torch.uint8)
    x = torch.randn(4, h).bfloat16()
    ids = torch.tensor([[0, 1], [2, -1], [1, 1], [0, 2]], dtype=torch.int32)
    weights = torch.rand(4, 2)
    shapes = ((e, r, h),) * 2 + ((e, i, r),) * 2 + ((e, r, i), (e, h, r))
    lora = [torch.randn(s).bfloat16() * 0.1 for s in shapes]
    args = (x, w13, s13, w2, s2, weights, ids)
    base = mxfp4_experts(*args, exact=True, swiglu_limit=3.0)
    zero_b = [*lora[:2], torch.zeros_like(lora[2]), torch.zeros_like(lora[3]), lora[4], lora[5] * 0]
    assert torch.equal(mxfp4_experts(*args, exact=True, swiglu_limit=3.0, lora=zero_b), base)
    out = mxfp4_experts(*args, exact=True, swiglu_limit=3.0, lora=lora, scaling=0.5)
    # Independent composition for token 1 (one route, expert 2).
    ga, ua, gb, ub, da, db = (f[2].float() for f in lora)
    from ctikernels.moe.torch._w4a8_reference import dequant_mxfp4

    xr = x[1].float()
    gu = xr @ dequant_mxfp4(w13[2], s13[2]).T  # w13 rows: up, then gate
    g = (gu[i:] + 0.5 * (ga @ xr) @ gb.T).clamp(max=3.0)
    u = (gu[:i] + 0.5 * (ua @ xr) @ ub.T).clamp(-3.0, 3.0)
    act = torch.nn.functional.silu(g) * u * weights[1, 0]
    y = act @ dequant_mxfp4(w2[2], s2[2]).T + 0.5 * (da @ act) @ db.T
    torch.testing.assert_close(out[1], y, rtol=1e-5, atol=1e-5)
