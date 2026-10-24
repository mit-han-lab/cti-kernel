# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""LoRA training with FP32 master factors: each op casts them to BF16 at every use. Masters
holding BF16 values give the BF16 leaves' outputs and input gradients bit for bit, and factor
gradients that round to theirs; masters with values BF16 cannot hold give the run on their
explicit BF16 casts (no kernel reads an FP32 factor as BF16)."""

import pytest
import torch

from ctikernels.linear.training import DenseProduct, lora_projection_training
from ctikernels.mlp.training import gated_mlp_lora
from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora

from .test_moe_lora_triton import block_quantize, rand, routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run(fn, x, factors, dy):
    leaves = [x.detach().clone().requires_grad_()]
    leaves += [f.detach().clone().requires_grad_() for f in factors]
    out = fn(leaves[0], *leaves[1:])
    return out, torch.autograd.grad(out, leaves, dy)


def _check(fn, x, factors, dy):
    out16, grads16 = _run(fn, x, factors, dy)
    out32, grads32 = _run(fn, x, [f.float() for f in factors], dy)
    assert torch.equal(out32, out16) and torch.equal(grads32[0], grads16[0])
    for g32, g16 in zip(grads32[1:], grads16[1:], strict=True):
        assert g32.dtype == torch.float32 and torch.equal(g32.bfloat16(), g16)
    # values off the BF16 grid: the per-use cast, not a reinterpretation of FP32 memory
    masters = [f.float() * (1 + 2**-11) for f in factors]
    assert not any(torch.equal(m.bfloat16().float(), m) for m in masters)
    out_m, grads_m = _run(fn, x, masters, dy)
    out_c, grads_c = _run(fn, x, [m.bfloat16() for m in masters], dy)
    assert torch.equal(out_m, out_c) and torch.equal(grads_m[0], grads_c[0])
    for gm, gc in zip(grads_m[1:], grads_c[1:], strict=True):
        assert gm.dtype == torch.float32 and torch.equal(gm.bfloat16(), gc)


@pytest.mark.parametrize("t", [1, 64, 300])
def test_lora_projection(t):
    torch.manual_seed(t)
    h, out, r = 512, 768, 16
    x, dy, w = rand(t, h, scale=1.0), rand(t, out, scale=1.0), rand(out, h)
    factors = (rand(r, h), rand(512, r), rand(r, h), rand(256, r))
    fn = lambda x, *f: lora_projection_training(x, DenseProduct(w), f, scaling=2.0)
    _check(fn, x, factors, dy)


@pytest.mark.parametrize("t", [1, 64, 300])
def test_gated_mlp_lora(t):
    torch.manual_seed(t)
    h, i, r = 512, 384, 16
    x, dy = rand(t, h, scale=1.0), rand(t, h, scale=1.0)
    gate_up, down = DenseProduct(rand(2 * i, h)), DenseProduct(rand(h, i))
    factors = (rand(r, h), rand(i, r), rand(r, h), rand(i, r), rand(r, i), rand(h, r))
    fn = lambda x, *f: gated_mlp_lora(x, gate_up, down, f[:4], f[4:], scaling=2.0)
    _check(fn, x, factors, dy)


@pytest.mark.parametrize("t", [1, 64, 300])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("base", ["bf16", "fp8"])
def test_moe_fused_experts_lora(t, masked, base):
    """Routed experts (shared-A), with the gated shared expert on the BF16 base."""
    torch.manual_seed(t + masked)
    e, h, i, si, r = 64, 512, 256, 256, 16
    x, dy, w13, w2 = rand(t, h, scale=1.0), rand(t, h, scale=1.0), rand(e, 2 * i, h), rand(e, h, i)
    ids, rw = routing(t, 8, e, masked)
    factors = [rand(e, r, h), rand(2, e, i, r), rand(e, r, i), rand(e, h, r)]
    kw = {"scaling": 2.0}
    if base == "fp8":
        (w13, s13, _), (w2, s2, _) = block_quantize(w13), block_quantize(w2)
        kw.update(w13_scale=s13, w2_scale=s2)
    else:
        factors += [rand(2, r, h), rand(2, si, r), rand(r, si), rand(h, r)]
        kw["shared_expert"] = (rand(2 * si, h), rand(h, si), rand(1, h))
    fn = lambda x, *f: fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights(*f), **kw)
    _check(fn, x, factors, dy)


def test_moe_inference_casts_fp32_masters_once():
    torch.manual_seed(0)
    e, h, i, r = 64, 512, 256, 16
    x, w13, w2 = rand(4, h, scale=1.0), rand(e, 2 * i, h), rand(e, h, i)
    ids, rw = routing(4, 8, e, False)
    factors = [rand(e, r, h) * 1.001, rand(2, e, i, r), rand(e, r, i), rand(e, h, r)]
    masters = MoELoRAWeights(*(f.float() * (1 + 2**-11) for f in factors))
    with torch.no_grad():
        out = fused_experts_lora(x, w13, w2, rw, ids, masters, scaling=2.0)
        want = fused_experts_lora(x, w13, w2, rw, ids, masters.to(torch.bfloat16), scaling=2.0)
    assert torch.equal(out, want)
