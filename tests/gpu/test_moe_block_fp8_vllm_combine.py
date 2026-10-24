# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Block-FP8 MoE inference combines as vLLM's finalize, bitwise, on every provider path.

A path's per-route rows y_k are its own unweighted BF16 FC2 rows: with one-hot route weights
(1 on slot k, 0 elsewhere) the finalize returns exactly y_k. The call with the real weights
must then equal the shared reference's combine (``_moe_reference.combine_reference``: FP32 FMA
in route order from +0, one rounding) of those rows, bit for bit. The rows themselves match a
torch block-FP8 reference to GEMM accumulation order. Qwen3.5-35B-A3B-FP8 shapes."""

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE, _dsl_available

from ..modules import _moe_reference as ref
from .test_moe_block_fp8_cute import decoded, decoded_weight, random_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="block-FP8 MoE needs compute capability 8.9+",
)
E, H, INTER, TOP_K = 256, 2048, 512, 8
DECODE, PREFILL = (1, 4, 8), (64, 512, 2048)
_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((12, 0), (12, 1))
CUTE = pytest.mark.skipif(not (_SM12 and _dsl_available()), reason="cute-fp8 runs on sm120/121")


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    return random_weights(E, H, INTER)


def _routes(t, e=E, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed + t)
    ids = torch.stack([torch.randperm(e, generator=g, device="cuda")[:TOP_K] for _ in range(t)])
    ids = ids.int()
    if t > 1:
        ids[-1, -2:] = -1  # padding routes
    w = torch.rand(t, TOP_K, generator=g, device="cuda")
    return (w / w.sum(-1, keepdim=True)).contiguous(), ids.contiguous()


def _x(t, seed=2):
    g = torch.Generator(device="cuda").manual_seed(seed + t)
    return torch.randn(t, H, generator=g, device="cuda").bfloat16()


def _rows(call, w, ids):
    """[T, k, H] rows of ``call`` via one-hot route weights."""
    rows = []
    for k in range(ids.shape[1]):
        onehot = torch.zeros_like(w)
        onehot[:, k] = 1.0
        rows.append(call(onehot))
    return torch.stack(rows, 1)


def _reference_rows(x, params, ids, mapping=None, bf16_gate_up=False, samples=48):
    """Torch block-FP8 rows on a sample of the live routes: 1x128 E4M3 activations,
    block-scaled E4M3 weights, FP32 GEMMs (Triton stores gate/up in BF16), SiLU * up rounded
    to BF16 and requantized."""
    w13, s13, w2, s2 = params
    q, s = quantize_per_token_group_fp8(x)
    xa = decoded(q, s)
    local = ids.long()
    if mapping is not None:
        local = torch.where(local >= 0, mapping.long()[local.clamp_min(0)], -1)
    live = (local >= 0).nonzero()
    live = live[torch.randperm(len(live), device=x.device)[:samples]]
    rows = []
    for t, k in live.tolist():
        e = local[t, k].item()
        gu = xa[t] @ decoded_weight(w13[e], s13[e]).T
        if bf16_gate_up:
            gu = gu.bfloat16().float()
        act = (F.silu(gu[:INTER]) * gu[INTER:]).bfloat16()
        qa, sa = quantize_per_token_group_fp8(act[None])
        rows.append(decoded(qa, sa)[0] @ decoded_weight(w2[e], s2[e]).T)
    return live, torch.stack(rows)


def _check(call, x, w, ids, params, mapping=None, triton=False):
    out = call(w)
    rows = _rows(call, w, ids)
    valid = ids >= 0
    if mapping is not None:
        valid &= mapping[ids.clamp_min(0).long()] >= 0
    assert torch.equal(out, ref.combine_reference(rows, w, valid))  # the finalize, bitwise
    live, want = _reference_rows(x, params, ids, mapping, bf16_gate_up=triton)
    got = rows[live[:, 0], live[:, 1]].float()
    err = (got - want).norm() / want.norm()
    assert err < 1e-2, err  # rows: GEMM accumulation order only
    assert not rows[~valid].any()  # remote and padding routes add nothing


def _op(weights, provider, **kw):
    return BlockFP8MoE(*weights, top_k=TOP_K, provider=provider, **kw)


@pytest.mark.parametrize("t", DECODE + PREFILL)
def test_triton(weights, t):
    # T <= 8 at top-8 runs the route_blocks layout (one block per route); larger T aligns.
    op = _op(weights, "triton-fp8")
    x, (w, ids) = _x(t), _routes(t)
    _check(lambda ww: op(x, ww, ids), x, w, ids, weights, triton=True)


@pytest.mark.parametrize("t", (4, 512))
def test_triton_expert_parallel(weights, t):
    local = E // 2
    mapping = torch.full((2 * E,), -1, dtype=torch.int32, device="cuda")
    mapping[local : local + E] = torch.arange(E, dtype=torch.int32, device="cuda")
    op = _op(weights, "triton-fp8", expert_map=mapping, num_global_experts=2 * E)
    x, (w, ids) = _x(t), _routes(t, e=2 * E)
    _check(lambda ww: op(x, ww, ids), x, w, ids, weights, mapping, triton=True)


@pytest.mark.parametrize("t", (4, 512))
def test_triton_lora_down_hook(weights, t):
    # The hook path combines in masked_reduce: a zero delta is bitwise the plain combine, and a
    # nonzero delta enters each route as BF16(y + delta) before the same FMA chain.
    op = _op(weights, "triton-fp8")
    x, (w, ids) = _x(t), _routes(t)
    g = torch.Generator(device="cuda").manual_seed(3)
    delta = (torch.randn(ids.numel(), H, generator=g, device="cuda") * 1e-2).bfloat16()
    zero = torch.zeros_like(delta)
    assert torch.equal(op.forward_staged(x, w, ids, down_hook=lambda a: zero), op(x, w, ids))
    rows = _rows(lambda ww: op(x, ww, ids), w, ids)
    lora = (rows.float() + delta.view_as(rows).float()).bfloat16()
    out = op.forward_staged(x, w, ids, down_hook=lambda a: delta)
    assert torch.equal(out, ref.combine_reference(lora, w, ids >= 0))


@CUTE
@pytest.mark.parametrize("t", DECODE)
@pytest.mark.parametrize("preset", ["decode-n32", "decode-n64"])
def test_cute_decode(weights, t, preset):
    op = _op(weights, "cute-fp8").set_config(preset)
    x, (w, ids) = _x(t), _routes(t)
    _check(lambda ww: op(x, ww, ids), x, w, ids, weights)


@CUTE
@pytest.mark.parametrize("t", DECODE + PREFILL)
def test_cute_prefill(weights, t):
    op = _op(weights, "cute-fp8").set_config(
        "swap16" if t <= 4 else "swap32" if t <= 128 else "group64"
    )
    x, (w, ids) = _x(t), _routes(t)
    _check(lambda ww: op(x, ww, ids), x, w, ids, weights)


@pytest.mark.parametrize(
    "provider,preset", [("triton-fp8", None), pytest.param("cute-fp8", "decode-n32", marks=CUTE)]
)
def test_capture_replays_the_eager_combine(weights, provider, preset):
    op = _op(weights, provider)
    if preset is not None:
        op.set_config(preset)
    t = 4
    x, (w, ids) = _x(t), _routes(t)
    op.prepare([t])
    want = op(x, w, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, w, ids)
    w.copy_(torch.flip(w, (1,)))
    graph.replay()
    assert torch.equal(out, op(x, w, ids))
    assert not torch.equal(out, want)  # the captured combine read the new weights
