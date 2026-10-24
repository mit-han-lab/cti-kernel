# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""LoRA on every projection of a Qwen3.5-35B-A3B-FP8-style GDN layer and attention block.

GDN: in_proj_qkv, in_proj_z and out_proj on block-FP8 bases (LoRA fused into the Triton
GEMM epilogue), in_proj_b and in_proj_a on BF16 bases (the LoRA projection plan), one
decode step in between. Attention: q (carrying the output gate), k, v and o on block-FP8
bases. Everything is compared with an FP64 reference that sees the same FP8 activation
quantization and BF16 roundings of the stored intermediates.
"""

from __future__ import annotations

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode
from ctikernels.linear.block_fp8 import BlockFP8Linear, quantize_per_token_group_fp8
from ctikernels.linear.lora import (
    lora_projection,
    lora_shrink,
    try_prepare_lora_inference_plan,
)

from . import _gdn_reference as reference
from .test_block_fp8 import _random_weight, _weight64
from .test_gdn import _pools, _weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the Triton LoRA providers run on cc >= 10.0",
)
F64 = torch.float64
HIDDEN, RANK, SCALING = 1024, 16, 0.5
K_HEADS, V_HEADS = 8, 16  # GDN geometry; K = V = 128


def _factors(k, sizes, seed):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        t.to(torch.bfloat16).cuda()
        for n in sizes
        for t in (
            torch.randn(RANK, k, generator=generator) * 0.03,
            torch.randn(n, RANK, generator=generator) * 0.03,
        )
    )


def _delta64(x, factors):
    return torch.cat(
        [
            SCALING * (x.to(F64) @ a.to(F64).T) @ b.to(F64).T
            for a, b in zip(factors[::2], factors[1::2], strict=True)
        ],
        dim=1,
    )


class _FP8Projection:
    """Block-FP8 base + LoRA (Triton epilogue), with its FP64 reference."""

    def __init__(self, k, sizes, seed):
        self.sizes = sizes
        weight, scale = _random_weight(sum(sizes), k)
        self.op = BlockFP8Linear(weight, scale, provider="triton-fp8")
        self.weight64 = _weight64(weight, scale)
        self.factors = _factors(k, sizes, seed)

    def __call__(self, x):
        lora = lora_shrink(x, self.factors, scaling=SCALING, output_sizes=self.sizes)
        return self.op(x, lora=lora)

    def reference(self, x):
        q, s = quantize_per_token_group_fp8(x)  # the base sees the FP8-quantized activation
        x64 = q.to(F64) * s.to(F64).repeat_interleave(128, 1)[:, : x.shape[1]]
        return x64 @ self.weight64.T + _delta64(x, self.factors)


class _BF16Projection:
    def __init__(self, k, n, seed):
        generator = torch.Generator().manual_seed(seed)
        self.weight = (torch.randn(n, k, generator=generator) * 0.03).to(torch.bfloat16).cuda()
        self.factors = _factors(k, (n,), seed + 1)
        self.plan = try_prepare_lora_inference_plan(
            self.factors, input_size=k, output_sizes=(n,), scaling=SCALING
        )
        assert self.plan is not None

    def __call__(self, x):
        out = lora_projection(
            x, self.weight, None, self.factors, scaling=SCALING, inference_plan=self.plan
        )
        assert self.plan.provider.endswith("-lora-v1")
        return out

    def reference(self, x):
        return x.to(F64) @ self.weight.to(F64).T + _delta64(x, self.factors)


def _relative(got, expected):
    return float((got.to(F64) - expected).norm() / expected.norm())


@pytest.mark.parametrize("batch", [1, 2, 4])
@torch.inference_mode()
def test_gdn_layer_with_lora_on_every_projection(batch):
    q_width, v_width = K_HEADS * 128, V_HEADS * 128
    qkv = _FP8Projection(HIDDEN, (q_width, q_width, v_width), seed=1)
    z = _FP8Projection(HIDDEN, (v_width,), seed=2)
    b = _BF16Projection(HIDDEN, V_HEADS, seed=3)
    a = _BF16Projection(HIDDEN, V_HEADS, seed=4)
    out_proj = _FP8Projection(v_width, (HIDDEN,), seed=5)
    weights = _weights(K_HEADS, V_HEADS, seed=6)
    state, conv_state = _pools(K_HEADS, V_HEADS, seed=7)
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    x = (torch.randn(batch, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)

    projected = torch.cat((qkv(x), z(x), b(x), a(x)), dim=1)
    projected64 = torch.cat(
        (qkv.reference(x), z.reference(x), b.reference(x), a.reference(x)), dim=1
    )
    assert _relative(projected, projected64) < 1e-2
    y_ref, _, _ = reference.decode(
        projected64.to(torch.bfloat16), weights, K_HEADS, slots, state, conv_state
    )
    op = GatedDeltaNetDecode(num_k_heads=K_HEADS, num_v_heads=V_HEADS, device="cuda")
    y = op(projected, *weights.args(), slots, state.clone(), conv_state.clone())
    assert _relative(y, y_ref) < 2e-2
    out = out_proj(y)
    assert _relative(out, out_proj.reference(y)) < 1e-2  # the stage itself
    assert _relative(out, out_proj.reference(y_ref.to(torch.bfloat16))) < 3e-2  # end to end


@pytest.mark.parametrize("batch", [1, 4, 33])
@torch.inference_mode()
def test_attention_projections_with_lora(batch):
    heads, kv_heads, dim = 8, 2, 128
    q = _FP8Projection(HIDDEN, (heads * dim, heads * dim), seed=11)  # q | output gate
    k = _FP8Projection(HIDDEN, (kv_heads * dim,), seed=12)
    v = _FP8Projection(HIDDEN, (kv_heads * dim,), seed=13)
    o = _FP8Projection(heads * dim, (HIDDEN,), seed=14)
    x = (torch.randn(batch, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    for projection in (q, k, v):
        assert _relative(projection(x), projection.reference(x)) < 1e-2
    gate = q(x)[:, heads * dim :]
    attention = (torch.randn(batch, heads * dim, device="cuda")).to(torch.bfloat16)
    gated = (attention.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    assert _relative(o(gated), o.reference(gated)) < 1e-2
