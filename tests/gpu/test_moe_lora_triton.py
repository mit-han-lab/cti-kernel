# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton MoE LoRA provider (forced on every device): FP64 oracles, graphs, shared workspace."""

import pytest
import torch
import torch.nn.functional as F
import triton

from ctikernels import GraphWorkspace
from ctikernels.moe import MoEBlock, fused_experts, route, to_local_topk_ids
from ctikernels.moe.lora import (
    MoELoRAWeights,
    fused_experts_lora,
    fused_experts_lora_training_bytes,
    moe_lora_w2_,
    moe_lora_w13_,
    moe_lora_w13_act_,
    shared_expert_lora,
)
from ctikernels.moe_block_fp8 import BlockFP8MoE
from ctikernels.moe_impl import moe_align_block_size

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="CUDA with compute capability >= 8.9 is required",
)


def rand(*shape, scale=0.05):
    return (torch.randn(shape, device="cuda") * scale).bfloat16()


def routing(t, top_k, experts, masked):
    ids = torch.randint(0, experts, (t, top_k), device="cuda", dtype=torch.int32)
    if masked and t:
        ids[ids >= experts // 2] = -1  # EP: routes held by another rank
        ids[0, 0] = experts + 3  # out of range: also nothing
    return ids, torch.rand((t, top_k), device="cuda")


def alignment(ids, experts, block_m):
    safe = torch.where((ids >= 0) & (ids < experts), ids, experts)
    sorted_ids, block_experts, padded = moe_align_block_size(safe, block_m, experts)
    return {
        "sorted_token_ids": sorted_ids,
        "expert_ids": block_experts,
        "num_tokens_post_padded": padded,
        "block_m": block_m,
    }


def projection_oracle(before, x, a, b, ids, rw, top_k, scale, act):
    """FP64 per-expert shrink/expand; returns (expected cache or activation, delta, z)."""
    p, slices, n = before.shape[0], len(a), b[0].shape[-2]
    delta = torch.zeros(before.shape, dtype=torch.float64, device="cuda")
    z = torch.zeros((p, slices, a[0].shape[-2]), dtype=torch.float64, device="cuda")
    flat = ids.flatten()
    for e in flat.unique().tolist():
        if not 0 <= e < a[0].shape[0]:
            continue
        pairs = torch.where(flat == e)[0]
        rows = pairs // top_k if rw is None else pairs
        w = 1.0 if rw is None else rw.flatten()[pairs, None].double()
        for s in range(slices):
            z[pairs, s] = x[rows].double() @ a[s][e].double().T
            delta[pairs, s * n : (s + 1) * n] = scale * w * (z[pairs, s] @ b[s][e].double().T)
    full = before.double() + delta
    if act:
        return F.silu(full[:, :n]) * full[:, n:], delta, z
    return full, delta, z


def rel(actual, expected):
    return float((actual.double() - expected).norm() / expected.norm().clamp_min(1e-30))


@pytest.mark.parametrize("t", [0, 1, 4, 40, 300])
@pytest.mark.parametrize("rank", [1, 8, 12, 16, 64, 128, 136, 256])
@pytest.mark.parametrize("op", ["w13", "w13_act", "w2"])
@pytest.mark.parametrize("aligned", [False, True])
def test_projection_matches_fp64(t, rank, op, aligned):
    torch.manual_seed(rank * 1000 + t)
    experts, h, i, top_k, scale = 64, 256, 192, 8, 0.7
    down = op == "w2"
    k, n, slices = (i, h, 1) if down else (h, i, 2)
    a = [rand(experts, rank, k) for _ in range(slices)]
    b = [rand(experts, n, rank) for _ in range(slices)]
    ids, rw = routing(t, top_k, experts, masked=True)
    x = rand(t * top_k if down else t, k, scale=0.5)
    before = rand(t * top_k, slices * n, scale=0.01)
    z = torch.full((t * top_k, slices, rank), -9.0, device="cuda")
    kw = {"scaling": scale, "top_k": top_k, "provider": "triton", "z_out": z}
    if aligned and t:
        kw.update(alignment(ids, experts, 16))
    factors = (a[0], b[0]) if down else (a[0], a[1], b[0], b[1])
    actual = before.clone()
    if op == "w13_act":
        result = moe_lora_w13_act_(
            torch.empty(t * top_k, n, dtype=torch.bfloat16, device="cuda"),
            actual,
            x,
            factors,
            ids,
            **kw,
        )
    elif down:
        result = moe_lora_w2_(actual, x, factors, ids, rw, **kw)
    else:
        result = moe_lora_w13_(actual, x, factors, ids, **kw)
    expected, delta, saved = projection_oracle(
        before, x, a, b, ids, rw if down else None, top_k, scale, op == "w13_act"
    )
    valid = ((ids >= 0) & (ids < experts)).flatten()
    if t == 0:
        return
    if op == "w13_act":
        assert torch.equal(actual, before)  # the gate/up cache is read only
        assert rel(result[valid], expected[valid]) <= 1e-2
    else:
        assert rel(actual.double() - before.double(), delta) <= 1e-2
        assert torch.equal(actual[~valid], before[~valid])
    torch.testing.assert_close(z[valid].double(), saved[valid], atol=1e-4, rtol=1e-3)
    assert torch.all(z[~valid] == -9)


@pytest.mark.parametrize("t", [1, 4, 40, 300])
@pytest.mark.parametrize("rank", [16, 136])
@pytest.mark.parametrize("aligned", [False, True])
def test_w2_fp32_cache_accumulates_without_bf16_rounding(t, rank, aligned):
    """An FP32 cache3 (auto resolves to Triton on every device) gets the weighted delta in
    FP32: O(1) cache values would lose ~2^-9 relative to a BF16 round trip."""
    torch.manual_seed(t + rank)
    experts, h, i, top_k = 64, 256, 128, 8
    ids, rw = routing(t, top_k, experts, True)
    x = rand(t * top_k, i, scale=1.0)
    a, b = (rand(experts, rank, i),), (rand(experts, h, rank),)
    before = torch.randn((t * top_k, h), device="cuda")
    out = before.clone()
    kw = alignment(ids, experts, 16) if aligned else {}
    moe_lora_w2_(out, x, (*a, *b), ids, rw, scaling=0.5, top_k=top_k, **kw)
    _, delta, _ = projection_oracle(before, x, a, b, ids, rw, top_k, 0.5, False)
    assert out.dtype == torch.float32
    assert rel(out - before, delta) <= 1e-2
    untouched = delta.abs().sum(1) == 0
    assert torch.equal(out[untouched], before[untouched])
    with pytest.raises(ValueError, match="BF16 cache3"):
        moe_lora_w2_(out, x, (*a, *b), ids, rw, scaling=0.5, top_k=top_k, provider="cute")


def moe_case(t, experts, h, i, rank, masked, shared=False):
    x, w13, w2 = rand(t, h, scale=1.0), rand(experts, 2 * i, h), rand(experts, h, i)
    ids, rw = routing(t, 8, experts, masked)
    routed = (rand(experts, rank, h), rand(experts, rank, h), rand(experts, i, rank),
              rand(experts, i, rank), rand(experts, rank, i), rand(experts, h, rank))  # fmt: skip
    extra = (rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank)) if shared else ()
    return x, w13, w2, rw, ids, MoELoRAWeights.separate(*routed, *extra)


def shared_a_case(t, experts, h, i, rank, masked, shared=False):
    """moe_case with the shared-A (PEFT gate_up_proj) layout, the only trainable one."""
    x, w13, w2, rw, ids, sep = moe_case(t, experts, h, i, rank, masked, shared)
    b = torch.stack((sep.gate_b, sep.up_b))
    return x, w13, w2, rw, ids, MoELoRAWeights(sep.gate_a, b, sep.down_a, sep.down_b,
                                               *(sep.shared or ()))  # fmt: skip


def unsloth_routed(x, w13, w2, rw, ids, lora, scale):
    """Unsloth's grouped-GEMM expert order in torch (BF16 throughout): PEFT LoRA adds, BF16
    SiLU and SiLU * up, unweighted BF16 down rows, then moe_block's combine with BF16 weights."""
    from ..modules._moe_reference import unsloth_combine

    t, top_k = ids.shape
    i = w2.shape[-1]
    a, b, a2, b2 = lora.leaves
    y = torch.zeros(t * top_k, x.shape[1], dtype=x.dtype, device=x.device)
    flat = ids.flatten()
    for e in flat.unique().tolist():
        if not 0 <= e < w13.shape[0]:
            continue
        pairs = torch.where(flat == e)[0]
        inp = x[pairs // top_k]
        z = F.linear(inp, a[e])
        gate = F.linear(inp, w13[e, :i]) + F.linear(z, b[0, e]) * scale
        up = F.linear(inp, w13[e, i:]) + F.linear(z, b[1, e]) * scale
        c2 = F.silu(gate) * up
        y[pairs] = F.linear(c2, w2[e]) + F.linear(F.linear(c2, a2[e]), b2[e]) * scale
    return unsloth_combine(y.view(t, top_k, -1), rw.bfloat16())


@pytest.mark.parametrize("t", [1, 4, 16, 64, 300])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("rank", [16, 64])
def test_bf16_base_is_fused_experts_bitwise(t, ep, rank):
    """A BF16 base is fused_experts' own pipeline: no LoRA and zero factors give its bits.
    EP2 mirrors an engine's two calls: unadapted, fused_experts on local ids with the other
    rank's routes parked on local expert 0 at weight 0; adapted, the local ids (-1 remote),
    local_only=False and no global expert count."""
    torch.manual_seed(t * 3 + ep + rank)
    e, h, i = 128, 512, 256
    x, w13, w2 = rand(t, h, scale=1.0), rand(e, 2 * i, h), rand(e, h, i)
    ids, rw = routing(t, 8, e, masked=False)
    el = e // ep
    w13l, w2l = w13[:el].contiguous(), w2[:el].contiguous()
    if ep == 1:
        ref = fused_experts(x, w13l, w2l, rw, ids)
        local = ids
    else:
        emap = torch.full((e,), -1, dtype=torch.int32, device="cuda")
        emap[:el] = torch.arange(el, dtype=torch.int32, device="cuda")
        held = emap[ids.long()] >= 0
        parked = emap[ids.long()].where(held, 0).int()
        ref = fused_experts(x, w13l, w2l, rw.where(held, 0.0), parked)
        local = to_local_topk_ids(ids, emap)
    zero = MoELoRAWeights.separate(
        *(torch.zeros(s, dtype=torch.bfloat16, device="cuda") for s in (
            (el, rank, h), (el, rank, h), (el, i, rank), (el, i, rank), (el, rank, i), (el, h, rank)))
    )  # fmt: skip
    kw = {"scaling": 0.5, "provider": "triton", "local_only": ep == 1}
    for lora in (None, zero):
        out = fused_experts_lora(x, w13l, w2l, rw, local, lora, **kw)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), lora is None


@pytest.mark.parametrize("t", [1, 4, 64, 300, 2048])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_bf16_inference_combines_as_vllm(monkeypatch, t, masked, shared):
    """Triton BF16-base inference with an adapter: the reduce takes unweighted BF16 route rows
    (the down delta in each, one rounding) and gives vLLM's finalize of them (FP32 FMA chain in
    route order, one rounding) and its BF16 gated shared join, bit for bit."""
    from ctikernels.moe.triton import _lora_ops

    from ..modules._moe_reference import combine_reference, shared_join_reference

    torch.manual_seed(t + 7 * masked + 3 * shared)
    e, h, i, si = 64, 512, 256, 256
    x, w13, w2, rw, ids, lora = moe_case(t, e, h, i, 16, masked, shared=shared)
    seen = []
    reduce = _lora_ops.moe_sum_reduce_triton

    def capture(rows, out, *args, **kwargs):
        seen.append((rows.clone(), kwargs))
        return reduce(rows, out, *args, **kwargs)

    monkeypatch.setattr(_lora_ops, "moe_sum_reduce_triton", capture)
    kw = {"scaling": 0.5, "provider": "triton"}
    if shared:
        kw["shared_expert"] = (rand(2 * si, h), rand(h, si), rand(1, h))
    out = fused_experts_lora(x, w13, w2, rw, ids, lora, **kw)
    ((rows, kwargs),) = seen
    assert rows.dtype == torch.bfloat16 and torch.equal(kwargs["topk_weights"], rw)
    valid = (ids >= 0) & (ids < e)
    want = combine_reference(rows, rw, valid)
    if shared:
        want = shared_join_reference(want, kwargs["shared"], kwargs["shared_gate"])
    assert torch.equal(out, want)


def moe_oracle(x, w13, w2, rw, ids, lora, scale):
    """FP64 routed MLP with per-expert LoRA (base and LoRA unrounded)."""
    t, top_k = ids.shape
    out = torch.zeros(x.shape, dtype=torch.float64, device="cuda")
    ga, ua, gb, ub, da, db = (f.double() for f in lora.routed)
    for tok in range(t):
        xt = x[tok].double()
        for j in range(top_k):
            e = int(ids[tok, j])
            if not 0 <= e < w13.shape[0]:
                continue
            gu = w13[e].double() @ xt
            i = gu.shape[0] // 2
            g = gu[:i] + scale * gb[e] @ (ga[e] @ xt)
            u = gu[i:] + scale * ub[e] @ (ua[e] @ xt)
            c2 = F.silu(g) * u
            y = w2[e].double() @ c2 + scale * db[e] @ (da[e] @ c2)
            out[tok] += float(rw[tok, j]) * y
    return out


def shared_oracle(x, w_gate_up, w_down, w_gate, factors, scale):
    """FP64 gated shared expert sigmoid(x . w_gate) * shared(x), LoRA (as13, bs13, as2, bs2)."""
    xd, i = x.double(), w_down.shape[1]
    gu = xd @ w_gate_up.double().T
    if factors is not None:
        as13, bs13, as2, bs2 = (f.double() for f in factors)
        gu = gu + scale * torch.cat((xd @ as13[0].T @ bs13[0].T, xd @ as13[1].T @ bs13[1].T), 1)
    act = F.silu(gu[:, :i]) * gu[:, i:]
    y = act @ w_down.double().T
    if factors is not None:
        y = y + scale * act @ as2.T @ bs2.T
    return torch.sigmoid(xd @ w_gate.double().T) * y


def qwen35_block(experts=256, h=2048, i=512, si=512, scale=0.02):
    return (rand(experts, h, scale=scale), rand(experts, 2 * i, h, scale=scale),
            rand(experts, h, i, scale=scale), rand(2 * si, h, scale=scale),
            rand(h, si, scale=scale), rand(1, h, scale=scale))  # fmt: skip


@pytest.mark.parametrize("t", [1, 4, 64])
def test_block_matches_composed_block_and_zero_adapter_is_no_lora(t):
    """With shared_expert, a zero adapter's MoE block is as accurate as MoEBlock's composed
    block (fused_experts' BF16 recipe and the shared expert) and bitwise equal to no LoRA."""
    torch.manual_seed(t)
    router, w13, w2, sgu, sd, sg = qwen35_block()
    x = rand(t, 2048, scale=1.0)
    rw, ids = route(x, router, 8, renormalize=True)
    block = MoEBlock("composed")(x.clone(), router, w13, w2, sgu, sd, sg)
    zero = lambda *s: torch.zeros(s, device="cuda", dtype=torch.bfloat16)
    lora = MoELoRAWeights.separate(rand(256, 16, 2048), rand(256, 16, 2048), zero(256, 512, 16),
                          zero(256, 512, 16), rand(256, 16, 512), zero(256, 2048, 16),
                          rand(2, 16, 2048), zero(2, 512, 16), rand(16, 512), zero(2048, 16))  # fmt: skip
    kw = {"scaling": 2.0, "shared_expert": (sgu, sd, sg)}
    ours = fused_experts_lora(x, w13, w2, rw, ids.int(), lora, **kw)
    assert torch.equal(ours, fused_experts_lora(x, w13, w2, rw, ids.int(), None, **kw))
    reference = moe_oracle(x, w13, w2, rw, ids, lora, 2.0) + shared_oracle(
        x, sgu, sd, sg, None, 2.0
    )
    error = lambda y: float((y.double() - reference).abs().mean())
    assert error(ours) <= 1.05 * error(block)


@pytest.mark.parametrize("t", [1, 4, 40])
def test_block_ep2_tp2_partials_sum_to_the_block(t):
    """EP2 experts and TP2 shared I-windows (with their shared-LoRA slices): each rank's
    partial is MoEBlock's partial convention, remote (-1) routes add nothing and the weights
    are not renormalized; the two partials sum to the FP64 block."""
    torch.manual_seed(100 + t)
    e, h, i, si, r = 64, 512, 256, 256, 16
    router, w13, w2, sgu, sd, sg = qwen35_block(e, h, i, si, scale=0.05)
    x = rand(t, h, scale=1.0)
    routed_weights, ids = route(x, router, 8, renormalize=True)
    rw = routed_weights * 0.9  # any weights: none are renormalized after dropping remote routes
    routed = (rand(e, r, h), rand(e, r, h), rand(e, i, r), rand(e, i, r), rand(e, r, i),
              rand(e, h, r))  # fmt: skip
    shared = (rand(2, r, h), rand(2, si, r), rand(r, si), rand(h, r))
    full = moe_oracle(x, w13, w2, rw, ids, MoELoRAWeights.separate(*routed), 0.5)
    full = full + shared_oracle(x, sgu, sd, sg, shared, 0.5)
    total = torch.zeros_like(full)
    for rank in range(2):
        experts, window = slice(32 * rank, 32 * (rank + 1)), slice(128 * rank, 128 * (rank + 1))
        expert_map = torch.full((e,), -1, dtype=torch.int32, device="cuda")
        expert_map[experts] = torch.arange(32, dtype=torch.int32, device="cuda")
        local = to_local_topk_ids(ids.int(), expert_map)
        gu = torch.cat((sgu[:si][window], sgu[si:][window])).contiguous()
        down = sd[:, window].contiguous()
        lora = MoELoRAWeights.separate(
            *(f[experts].contiguous() for f in routed),
            shared[0], shared[1][:, window].contiguous(), shared[2][:, window].contiguous(),
            shared[3],
        )  # fmt: skip
        part = fused_experts_lora(
            x, w13[experts].contiguous(), w2[experts].contiguous(), rw, local, lora,
            scaling=0.5, shared_expert=(gu, down, sg),
        )  # fmt: skip
        mine = moe_oracle(x, w13[experts], w2[experts], rw, local, lora, 0.5)
        mine = mine + shared_oracle(x, gu, down, sg, lora.shared, 0.5)
        assert rel(part, mine) <= 1e-2
        block = MoEBlock("composed")(x.clone(), router, w13[experts].contiguous(),
                                     w2[experts].contiguous(), gu, down, sg,
                                     expert_map=expert_map)  # fmt: skip
        base = fused_experts_lora(x, w13[experts].contiguous(), w2[experts].contiguous(),
                                  routed_weights, local, None, scaling=0.5,
                                  shared_expert=(gu, down, sg))  # fmt: skip
        assert rel(base, block.double()) <= 1e-2
        total += part.double()
    assert rel(total.bfloat16(), full) <= 1e-2


@pytest.mark.parametrize("t", [0, 1, 2, 3, 4, 24])
@pytest.mark.parametrize("model", ["qwen3.5-35b-a3b", "qwen3-30b-a3b"])
@pytest.mark.parametrize("rank", [16, 128])
@pytest.mark.parametrize("masked", [False, True])
def test_composition_matches_fp64(t, model, rank, masked):
    torch.manual_seed(t + rank)
    experts, i = (256, 512) if model == "qwen3.5-35b-a3b" else (128, 768)
    x, w13, w2, rw, ids, lora = moe_case(t, experts, 2048, i, rank, masked)
    kw = {"scaling": 0.5, "local_only": not masked}
    out = fused_experts_lora(x, w13, w2, rw, ids, lora, provider="triton", **kw)
    if t == 0:
        assert out.shape == x.shape
        return
    expected = moe_oracle(x, w13, w2, rw, ids, lora, 0.5)
    reference = fused_experts_lora(x, w13, w2, rw, ids, lora, provider="torch", **kw)
    # Per row: no worse than the FP32 torch composition (both round c1 and c2 to BF16).
    for row in range(t):
        bound = max(2 * rel(reference[row], expected[row]), 4e-3)
        assert rel(out[row], expected[row]) <= bound


@pytest.mark.parametrize("t", [1, 4, 64])
@pytest.mark.parametrize("rank", [8, 40])
def test_shared_expert(t, rank):
    torch.manual_seed(7)
    h, i = 512, 256
    x = rand(t, h, scale=1.0)
    w_gate_up, w_down = rand(2 * i, h), rand(h, i)
    shared = (rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank))
    out = shared_expert_lora(x, w_gate_up, w_down, shared, 0.5, provider="triton")
    ref = shared_expert_lora(x, w_gate_up, w_down, shared, 0.5, provider="torch")
    assert rel(out, ref.double()) <= 1e-2


@pytest.mark.parametrize("t", [1, 8, 64, 300])
@pytest.mark.parametrize("tensor_scale", [False, True])
def test_shared_expert_inference_is_vllm_punica(t, tensor_scale):
    """Triton shared-expert inference is vLLM's dense MLP with punica LoRA: BF16 GEMMs, each
    delta out = BF16(out + BF16(BF16(x . A^T) . BF16(B * s)^T)), SiLU * up rounded once.
    Against that transcription with FP32-accumulated GEMMs: GEMM order only."""
    torch.manual_seed(t + tensor_scale)
    h, i, rank, s = 512, 256, 16, 0.5
    x = rand(t, h, scale=1.0)
    w13, w2 = rand(2 * i, h), rand(h, i)
    a13, b13, a2, b2 = rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank)
    scale = torch.tensor([s], device="cuda") if tensor_scale else s
    out = shared_expert_lora(x, w13, w2, (a13, b13, a2, b2), scale, provider="triton")
    gemm = lambda a, w: (a.float() @ w.float().T).bfloat16()

    def lora(base, inp, a, b):
        delta = gemm(gemm(inp, a), (b.float() * s).bfloat16())
        return (base.float() + delta.float()).bfloat16()

    gate = lora(gemm(x, w13[:i]), x, a13[0], b13[0])
    up = lora(gemm(x, w13[i:]), x, a13[1], b13[1])
    act = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    want = lora(gemm(act, w2), act, a2, b2)
    assert rel(out, want.double()) <= 1e-3
    assert (out == want).float().mean() >= 0.95


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()  # warm-up on the capture stream
        with torch.cuda.graph(graph, stream=stream):
            out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, out


@pytest.mark.parametrize("t,rank", [(1, 16), (4, 128), (24, 16), (24, 256)])
def test_graph_replay_is_eager_and_shares_one_workspace(t, rank):
    """Graphs of several layers in one GraphWorkspace replay eager bits; the scratch is
    one region sized at the first capture, not one per graph or layer."""
    torch.manual_seed(3)
    layers = [moe_case(t, 128, 2048, 768, rank, masked=True) for _ in range(3)]
    run = lambda: [fused_experts_lora(*c, scaling=0.5, provider="triton") for c in layers]
    eager = run()
    with GraphWorkspace(max_rows=32) as ws:
        graphs = [capture(run) for _ in range(2)]
        size = ws.nbytes
        graphs.append(capture(run))
        assert ws.nbytes == size
    for graph, outs in graphs:
        graph.replay()
        torch.cuda.synchronize()
        for got, ref in zip(outs, eager, strict=True):
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("shared", [False, True])
def test_capture_ladder_largest_first_in_one_workspace(rank, shared):
    """As the engine captures: one GraphWorkspace, each size warmed then captured, largest
    first down to T=1 (aligned, pair and fused decode paths); every slot a smaller size's
    path takes was reserved before the first capture, and replays are eager bits."""
    torch.manual_seed(5)
    e, h, i, sizes = 128, 512, 256, (96, 24, 8, 4, 2, 1)
    x, w13, w2, rw, ids, lora = moe_case(sizes[0], e, h, i, rank, masked=True, shared=shared)
    kw = {"scaling": 0.5, "provider": "triton"}
    if shared:
        kw["shared_expert"] = (rand(2 * i, h), rand(h, i), rand(1, h))
    run = lambda t: fused_experts_lora(x[:t], w13, w2, rw[:t], ids[:t], lora, **kw)
    eager = {t: run(t) for t in sizes}
    with GraphWorkspace(max_rows=sizes[0]):
        graphs = {t: capture(lambda t=t: run(t)) for t in sizes}
    for t, (graph, out) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager[t].view(torch.int16)), t


@pytest.mark.parametrize("rank", [16, 256])
def test_low_level_ops_under_capture_match_eager(rank):
    """The engine's call shape: base alignment passed in; the split shrink's FP32 partials come
    from the workspace reserved at the eager call."""
    torch.manual_seed(5)
    experts, h, i, t = 128, 2048, 768, 24
    x, _, _, rw, ids, lora = moe_case(t, experts, h, i, rank, masked=True)
    align = alignment(ids, experts, 16)
    c1, c2 = rand(t * 8, 2 * i), rand(t * 8, i)
    c3 = rand(t * 8, h)

    def run():
        o1, o3 = c1.clone(), c3.clone()
        kw = {"scaling": 0.5, "top_k": 8, "provider": "triton", **align}
        moe_lora_w13_(o1, x, lora.routed[:4], ids, **kw)
        moe_lora_w2_(o3, c2, lora.routed[4:], ids, rw, **kw)
        return o1, o3

    eager = run()
    with GraphWorkspace(max_rows=t):
        run()
        graph, outs = capture(run)
    graph.replay()
    torch.cuda.synchronize()
    for got, ref in zip(outs, eager, strict=True):
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("rank", [16, 64])
def test_insertion_ops_capture_ladder_largest_first(rank):
    """The insertion ops and the standalone shared expert, captured as the engine does (one
    workspace, largest size first down to T=1: aligned then pair paths) replay eager bits."""
    torch.manual_seed(6)
    e, h, i, sizes = 128, 512, 256, (96, 24, 8, 1)
    x, _, _, rw, ids, lora = moe_case(sizes[0], e, h, i, rank, masked=True)
    w_gate_up, w_down = rand(2 * i, h), rand(h, i)
    shared = (rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank))
    c1, c3 = rand(sizes[0] * 8, 2 * i), rand(sizes[0] * 8, h)
    aligns = {t: alignment(ids[:t], e, 16) for t in sizes}

    def run(t):
        # masked routes leave cache2 rows untouched: start them defined
        c2, o3 = torch.zeros(t * 8, i, dtype=c1.dtype, device="cuda"), c3[: t * 8].clone()
        kw = {"scaling": 0.5, "top_k": 8, "provider": "triton", **aligns[t]}
        moe_lora_w13_act_(c2, c1[: t * 8], x[:t], lora.routed[:4], ids[:t], **kw)
        moe_lora_w2_(o3, c2, lora.routed[4:], ids[:t], rw[:t], **kw)
        y = shared_expert_lora(x[:t], w_gate_up, w_down, shared, 0.5, provider="triton")
        return c2, o3, y

    eager = {t: run(t) for t in sizes}
    with GraphWorkspace(max_rows=sizes[0]):
        graphs = {t: capture(lambda t=t: run(t)) for t in sizes}
    for t, (graph, outs) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        for name, got, ref in zip(("c2", "c3", "shared"), outs, eager[t], strict=True):
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (t, name)


def ep_slice(weights, rank, ranks):
    """Rank ``rank``'s contiguous expert window of every [E, ...] tensor."""
    e = weights[0].shape[0] // ranks
    return [w[rank * e : (rank + 1) * e].contiguous() for w in weights]


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("t", [1, 4, 40, 300])
def test_expert_parallel_partials_sum_to_the_full_layer(ranks, t):
    """Each rank runs its local experts on -1-masked local ids; the partials sum to the full
    layer, remote routes cost nothing (poisoned expert 0 never leaks), and the old
    expert-0/weight-0 convention gives the same partial."""
    torch.manual_seed(ranks * 100 + t)
    experts, h, i, rank = 128, 512, 256, 16
    x, w13, w2, rw, ids, lora = moe_case(t, experts, h, i, rank, masked=False)
    ids[ids % (experts // ranks) == 0] += 1  # no valid route uses a rank's local expert 0
    expected = moe_oracle(x, w13, w2, rw, ids, lora, 0.5)
    total = torch.zeros_like(expected)
    for r in range(ranks):
        local = experts // ranks
        expert_map = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[r * local : (r + 1) * local] = torch.arange(
            local, device="cuda", dtype=torch.int32
        )
        local_ids = to_local_topk_ids(ids, expert_map)
        lw13, lw2, *factors = ep_slice([w13, w2, *lora.routed], r, ranks)
        lw13[0], lw2[0] = float("nan"), float("nan")  # remote routes must never touch expert 0
        part = fused_experts_lora(
            x,
            lw13,
            lw2,
            rw,
            local_ids,
            MoELoRAWeights.separate(*factors),
            scaling=0.5,
            provider="triton",
        )
        assert torch.isfinite(part).all()
        total += part.double()
        if ranks > 1:
            lw13[0], lw2[0] = 0.0, 0.0
            parked = torch.where(local_ids >= 0, local_ids, 0)
            old = fused_experts_lora(
                x, lw13, lw2, torch.where(local_ids >= 0, rw, 0.0), parked,
                MoELoRAWeights.separate(*factors), scaling=0.5, provider="triton",
            )  # fmt: skip
            torch.testing.assert_close(old, part, atol=1e-6, rtol=1e-3)
    for row in range(t):
        assert rel(total[row], expected[row]) <= 1e-2


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("t", [1, 64])
def test_shared_expert_tensor_parallel_partials_sum_to_full(tp, t):
    """Column-parallel gate/up and row-parallel down shards of the shared expert (factor
    windows on Is) give partial outputs whose single sum is the full shared MLP."""
    torch.manual_seed(tp + t)
    h, i, rank = 512, 256, 16
    x = rand(t, h, scale=1.0)
    w_gate_up, w_down = rand(2 * i, h), rand(h, i)
    a13, b13, a2, b2 = rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank)
    full = shared_expert_lora(x, w_gate_up, w_down, (a13, b13, a2, b2), 0.5, provider="torch")
    s = i // tp
    total = torch.zeros(t, h, dtype=torch.float64, device="cuda")
    for r in range(tp):
        cols = slice(r * s, (r + 1) * s)
        gate_up = torch.cat((w_gate_up[:i][cols], w_gate_up[i:][cols])).contiguous()
        shard = (a13, b13[:, cols].contiguous(), a2[:, cols].contiguous(), b2)
        part = shared_expert_lora(
            x, gate_up, w_down[:, cols].contiguous(), shard, 0.5, provider="triton"
        )
        total += part.double()
    assert rel(total, full.double()) <= 1e-2


def block_quantize(w):
    """E4M3 weights [E, N, K] with FP32 128x128 block scales, and their FP64 dequantization."""
    e, n, k = w.shape
    blocks = w.double().reshape(e, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-10) / 448
    q = (blocks / scale[:, :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = q.double() * scale[:, :, None, :, None]
    return q.reshape(e, n, k), scale.float(), dequant.reshape(e, n, k)


def token_qdq(x):
    """1x128 per-token quantize-dequantize with the kernels' FP32 recipe (amax floor 1e-10,
    q = E4M3_RN(clamp(x * fp32(448 / amax))), scale = amax * fp32(1 / 448)), as FP64."""
    groups = x.float().reshape(x.shape[0], -1, 128)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    q = (groups * (448 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q.double() * (amax * (1 / 448)).double()).reshape(x.shape)


def fp8_emulation(x, w13d, w2d, rw, ids, factors, scale, i):
    """FP64 routed MLP over block-dequantized weights with the kernels' activation recipe and
    rounding points (BF16 FC1 cache after the LoRA add, BF16 product, 1x128 quantization)."""
    out = torch.zeros(x.shape, dtype=torch.float64, device="cuda")
    xq = token_qdq(x)
    for tok in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[tok, j])
            if not 0 <= e < w13d.shape[0]:
                continue
            gu = (w13d[e] @ xq[tok]).bfloat16().double()
            xt = x[tok].double()
            g, u, lora2 = gu[:i], gu[i:], 0.0
            if factors is not None:
                ga, ua, gb, ub, da, db = (f[e].double() for f in factors)
                g, u = g + scale * gb @ (ga @ xt), u + scale * ub @ (ua @ xt)
            act = (F.silu(g.bfloat16().float()) * u.bfloat16().float()).bfloat16().double()
            if factors is not None:
                lora2 = scale * db @ (da @ act)
            out[tok] += float(rw[tok, j]) * (w2d[e] @ token_qdq(act[None])[0] + lora2)
    return out


@pytest.mark.parametrize("provider", ["triton", "cute", "torch"])
@pytest.mark.parametrize("t", [1, 4, 8, 40])
@pytest.mark.parametrize("graph", [False, True])
def test_bf16_lora_joins_the_shared_expert_as_vllm(provider, t, graph):
    """BF16-base MoE-LoRA inference with the caller's shared expert output and gate: bitwise
    the routed output joined as the engine joins it, BF16 ops sigmoid(gate) * shared + routed,
    on every provider (Triton in its reduce, the others by the one-launch join), eager and
    replayed; without them the bits are the routed output's."""
    from ctikernels.moe.cute._lora_provider import available

    if provider == "cute" and (
        not available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    ):
        pytest.skip("MoE LoRA CuTe targets sm_120/sm_121")
    if provider == "torch" and graph:
        pytest.skip("the torch provider is not graph-capturable")
    torch.manual_seed(t * 5 + graph)
    x, w13, w2, rw, ids, lora = moe_case(t, 64, 512, 256, 16, False)
    shared, gate = rand(t, 512, scale=1.0), rand(t, 1, scale=2.0)
    kw = {"scaling": 0.5, "provider": provider}
    routed = fused_experts_lora(x, w13, w2, rw, ids, lora, **kw)
    run = lambda: fused_experts_lora(x, w13, w2, rw, ids, lora, **kw, shared=shared,
                                     shared_gate=gate)  # fmt: skip
    if graph:
        replay, out = capture(run)
        out.zero_()
        replay.replay()
        torch.cuda.synchronize()
    else:
        out = run()
    want = torch.sigmoid(gate) * shared + routed  # BF16 ops: each rounded
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("t", [1, 4, 8, 40])  # P = 8t: the fused decode combine up to 64
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_block_fp8_lora_joins_the_shared_expert_as_vllm(t, masked, graph):
    """Block-FP8 MoE-LoRA inference with the caller's shared expert output and gate: bitwise
    the routed output joined as the engine joins it, BF16 ops sigmoid(gate) * shared + routed
    (vLLM's order), at decode (the down expand's fused combine) and above, eager and replayed."""
    torch.manual_seed(t * 3 + masked)
    experts, h, i, scale = 64, 512, 256, 0.5
    x, w13, w2, rw, ids, lora = moe_case(t, experts, h, i, 16, masked)
    q13, s13, _ = block_quantize(w13)
    q2, s2, _ = block_quantize(w2)
    shared, gate = rand(t, h, scale=1.0), rand(t, 1, scale=2.0)
    kw = {"scaling": scale, "provider": "triton", "w13_scale": s13, "w2_scale": s2}
    routed = fused_experts_lora(x, q13, q2, rw, ids, lora, **kw)
    run = lambda: fused_experts_lora(
        x, q13, q2, rw, ids, lora, **kw, shared=shared, shared_gate=gate
    )
    if graph:
        replay, out = capture(run)
        out.zero_()
        replay.replay()
        torch.cuda.synchronize()
    else:
        out = run()
    want = torch.sigmoid(gate) * shared + routed  # BF16 ops: each rounded
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("t", [0, 1, 4, 40])
@pytest.mark.parametrize("rank", [16, 128])
@pytest.mark.parametrize("masked", [False, True])
def test_block_fp8_base(t, rank, masked):
    """The base alone equals BlockFP8MoE bit for bit; with LoRA, the error against the FP64
    emulation stays within twice the base's own (E4M3 rounding flips dominate both)."""
    torch.manual_seed(t * 7 + rank)
    experts, h, i, scale = 64, 512, 256, 0.5
    x, w13, w2, rw, ids, lora = moe_case(t, experts, h, i, rank, masked)
    q13, s13, w13d = block_quantize(w13)
    q2, s2, w2d = block_quantize(w2)
    kw = {"scaling": scale, "provider": "triton", "w13_scale": s13, "w2_scale": s2}
    out = fused_experts_lora(x, q13, q2, rw, ids, lora, **kw)
    base = fused_experts_lora(x, q13, q2, rw, ids, None, **kw)
    if t == 0:
        assert out.shape == base.shape == x.shape
        return
    if not masked:
        reference = BlockFP8MoE(q13, s13, q2, s2, top_k=8)(x, rw, ids)
        assert torch.equal(base.view(torch.int16), reference.view(torch.int16))
    base_err = rel(base, fp8_emulation(x, w13d, w2d, rw, ids, None, scale, i))
    err = rel(out, fp8_emulation(x, w13d, w2d, rw, ids, lora.routed, scale, i))
    assert err <= max(2 * base_err, 1e-2)


@pytest.mark.parametrize("t", [1, 4, 64, 300])
@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("masked", [False, True])
def test_training_gradients_match_fp32_autograd_and_are_bitwise_deterministic(t, rank, masked):
    """dX, d(topk_weights) and the six factor gradients through the frozen BF16 base."""
    torch.manual_seed(t + rank)
    experts, h, i = 64, 512, 256
    x, w13, w2, rw, ids, lora = shared_a_case(t, experts, h, i, rank, masked)
    dy = rand(t, h, scale=1.0)

    def grads(provider):
        leaves = [v.detach().clone().requires_grad_() for v in (x, rw, *lora.leaves)]
        out = fused_experts_lora(
            leaves[0], w13, w2, leaves[1], ids, MoELoRAWeights(*leaves[2:]), scaling=0.5,
            provider=provider,
        )  # fmt: skip
        return torch.autograd.grad(out, leaves, dy)

    first, second, reference = grads("triton"), grads("triton"), grads("torch")
    names = ("dx", "dw", "gate_up_a", "gate_up_b", "down_a", "down_b")
    for name, a, b, ref in zip(names, first, second, reference, strict=True):
        assert torch.equal(
            a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
            b.view(torch.int16) if b.dtype == torch.bfloat16 else b,
        ), name
        err = float((a.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))
        assert err <= 2e-2, (name, err)


@pytest.mark.parametrize("t", [1, 4, 64, 300])
@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("shared", [False, True])
def test_block_training_is_unsloth_order_and_grads_track_fp32(t, rank, shared):
    """Trainable factors: the forward is Unsloth's order (its FP64-oracle error is no worse
    than a torch emulation of Unsloth's routed experts, which differs only in GEMM accumulation
    order; the gated shared expert is HF's MLP with PEFT LoRA, torch ops added in BF16); dx,
    d(topk_weights) and every factor gradient are bitwise run to run and track the FP32 torch
    autograd as closely as the emulation's own gradients do."""
    from ctikernels.moe.triton._lora_train import shared_forward

    torch.manual_seed(7 * t + rank)
    e, h, i, si = 64, 512, 256, 256
    x, w13, w2, rw, ids, lora = shared_a_case(t, e, h, i, rank, True, shared=True)
    shared_w = (rand(2 * si, h), rand(h, si), rand(1, h)) if shared else None
    factors = lora.leaves + (lora.shared if shared else ())
    dy = rand(t, h, scale=1.0)
    scale = torch.tensor(0.5, device="cuda")

    def emulate(leaves):
        weights = MoELoRAWeights(*leaves[2:])
        out = unsloth_routed(leaves[0], w13, w2, leaves[1], ids, weights, 0.5)
        if shared:
            out = out + shared_forward(leaves[0], *shared_w, weights.shared, scale)
        return out

    oracle = moe_oracle(x, w13, w2, rw, ids, lora, 0.5)
    if shared:
        oracle = oracle + shared_oracle(x, *shared_w, lora.shared, 0.5)
    emu_leaves = [v.detach().clone().requires_grad_() for v in (x, rw, *factors)]
    emulated = emulate(emu_leaves)
    emulated_grads = torch.autograd.grad(emulated, emu_leaves, dy)

    def run(provider):
        leaves = [v.detach().clone().requires_grad_() for v in (x, rw, *factors)]
        out = fused_experts_lora(
            leaves[0], w13, w2, leaves[1], ids, MoELoRAWeights(*leaves[2:]), scaling=0.5,
            provider=provider, shared_expert=shared_w,
        )  # fmt: skip
        return out, torch.autograd.grad(out, leaves, dy)

    (out, first), (again, second), (_, reference) = run("triton"), run("triton"), run("torch")
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
    assert rel(out, oracle) <= 1.05 * rel(emulated, oracle) + 1e-4
    for j, (a, b, emu, ref) in enumerate(
        zip(first, second, emulated_grads, reference, strict=True)
    ):
        assert torch.equal(a, b), j
        err, emu_err = rel(a, ref.double()), rel(emu, ref.double())
        assert err <= max(2e-2, 1.05 * emu_err + 1e-3), (j, err, emu_err)


@pytest.mark.parametrize("t", [1, 64])
def test_shared_expert_lora_trains_unsloth_order(t):
    """shared_expert_lora training: HF's MLP with PEFT LoRA Linears in BF16 torch ops (bitwise
    the torch expression), gradients tracking the FP32 torch autograd."""
    from ctikernels.moe.triton._lora_train import shared_mlp

    torch.manual_seed(t)
    h, si, r = 512, 256, 16
    x, w13, w2 = rand(t, h, scale=1.0), rand(2 * si, h), rand(h, si)
    lora = (rand(2, r, h), rand(2, si, r), rand(r, si), rand(h, r))
    dy = rand(t, h, scale=1.0)

    def run(provider):
        leaves = [v.detach().clone().requires_grad_() for v in (x, *lora)]
        out = shared_expert_lora(leaves[0], w13, w2, tuple(leaves[1:]), 0.5, provider=provider)
        return out, torch.autograd.grad(out, leaves, dy)

    (out, first), (_, reference) = run("triton"), run("torch")
    expected = shared_mlp(x, w13, w2, lora, torch.tensor(0.5, device="cuda"))
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    for j, (a, ref) in enumerate(zip(first, reference, strict=True)):
        err = float((a.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30))
        assert err <= 2e-2, (j, err)


def _held(tensor):
    return -(-tensor.numel() * tensor.element_size() // 512) * 512


def _fp8_base(w):
    """E4M3 weights with FP32 128x128 block scales (no dequantized copy)."""
    e, n, k = w.shape
    blocks = w.float().view(e, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-10) / 448
    q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(e, n, k), scale


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("t", [1, 300, 4096])
def test_training_bytes_match_measured_memory(t, rank, ep, fp8):
    """fused_experts_lora_training_bytes at 35B shapes (BF16 and block-FP8 bases) against the
    memory the forward keeps and the backward's peak: equal up to the caching allocator's
    unsplit blocks; the forward's peak, its output included, fits in their sum."""
    if fp8 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("block-FP8 needs compute capability 8.9")
    torch.manual_seed(t + rank)
    e, h, i = 256, 2048, 512
    x, w13, w2, dy = rand(t, h, scale=1.0), rand(e, 2 * i, h), rand(e, h, i), rand(t, h, scale=1.0)
    base = {}
    if fp8:
        (w13, base["w13_scale"]), (w2, base["w2_scale"]) = _fp8_base(w13), _fp8_base(w2)
    ids, rw = routing(t, 8, e, ep == 2)
    rw = rw / rw.sum(-1, keepdim=True)
    leaves = [v.requires_grad_() for v in (x, rw, rand(e, rank, h), rand(2, e, i, rank),
                                          rand(e, rank, i), rand(e, h, rank))]  # fmt: skip
    step = lambda: fused_experts_lora(
        leaves[0], w13, w2, leaves[1], ids, MoELoRAWeights(*leaves[2:]), scaling=2.0, **base
    )
    torch.autograd.grad(step(), leaves, dy)  # compile and warm every cache first
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = step()
    torch.cuda.synchronize()
    inputs = _held(x) + _held(rw) + _held(ids)
    forward = torch.cuda.max_memory_allocated() - start + inputs
    saved = torch.cuda.memory_allocated() - start - _held(out) + inputs
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    grads = torch.autograd.grad(out, leaves, dy)
    backward = torch.cuda.max_memory_allocated() - before
    del grads
    bound = fused_experts_lora_training_bytes(
        t, hidden=h, intermediate=i, experts=e, top_k=8, rank=rank, device=x.device,
        block_fp8=fp8,
    )  # fmt: skip
    for got, want in zip((saved, backward), bound, strict=True):
        assert want - (64 << 10) <= got <= want + (8 << 20), (got, want)
    assert forward <= sum(bound) + (8 << 20), (forward, bound)  # the forward's transients fit


def shared_case(t, rank, h=512, si=256):
    x, w13, w2 = rand(t, h, scale=1.0), rand(2 * si, h), rand(h, si)
    lora = (rand(2, rank, h), rand(2, si, rank), rand(rank, si), rand(h, rank))
    return x, w13, w2, lora, rand(t, h, scale=1.0)


@pytest.mark.parametrize("t", [1, 64, 300])
@pytest.mark.parametrize("rank", [16, 64])
@pytest.mark.parametrize("tensor_scale", [False, True])
def test_shared_mlp_backward_is_the_inductor_graph(t, rank, tensor_scale):
    """SharedMLP, forward and backward bit for bit, is the torch transcription of the Inductor
    kernels of Unsloth 2026.10.7's compiled shared MLP (rounding points pinned)."""
    from ctikernels.moe.triton._lora_train import shared_mlp

    torch.manual_seed(t + rank)
    x, w13, w2, (a13, b13, a2, b2), g = shared_case(t, rank)
    si, s = w2.shape[1], 0.5
    leaves = [v.detach().clone().requires_grad_() for v in (x, a13, b13, a2, b2)]
    scale = torch.tensor(s, device="cuda") if tensor_scale else s
    out = shared_mlp(leaves[0], w13, w2, tuple(leaves[1:]), scale)
    grads = torch.autograd.grad(out, leaves, g)

    def lin(inp, w, a, b):  # the forward as committed: FP32 B product, one rounding
        delta = F.linear(F.linear(inp, a).float(), b.float())
        return (F.linear(inp, w).float() + delta * s).bfloat16()

    gate, up = lin(x, w13[:si], a13[0], b13[0]), lin(x, w13[si:], a13[1], b13[1])
    h = (F.silu(gate.float()) * up.float()).bfloat16()  # one rounding
    assert torch.equal(out, lin(h, w2, a2, b2))
    scaled = lambda v: (v.float() * s).bfloat16()
    dz_d = scaled(g @ b2)
    dh = (dz_d @ a2).float() + (g @ w2).float()  # never rounded
    sg = torch.sigmoid(gate.float())
    d_up = (dh * F.silu(gate.float())).bfloat16()
    d_gate = (dh * up.float() * sg * (gate.float() * (1 - sg) + 1)).bfloat16()
    dz_g, dz_u = scaled(d_gate @ b13[0]), scaled(d_up @ b13[1])
    dx = torch.addmm(torch.addmm(dz_u @ a13[1], d_up, w13[si:]), dz_g, a13[0])
    db13 = (scaled(d_gate.T @ F.linear(x, a13[0])), scaled(d_up.T @ F.linear(x, a13[1])))
    want = (
        torch.addmm(dx, d_gate, w13[:si]),
        torch.stack((dz_g.T @ x, dz_u.T @ x)),
        torch.stack(db13),
        dz_d.T @ h,
        scaled(g.T @ F.linear(h, a2)),
    )
    for j, (a, b) in enumerate(zip(grads, want, strict=True)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), j


@pytest.mark.parametrize("t", [64, 300])
def test_shared_mlp_grads_match_compiled_unsloth_accuracy(t):
    """Against FP64, SharedMLP's gradients are as accurate as torch.compile of Unsloth's shared
    MLP (HF's MLP with unsloth_zoo's addmm lora_forward), which is how Unsloth trains it."""
    from ctikernels.moe.triton._lora_train import shared_mlp

    torch.manual_seed(t)
    x, w13, w2, lora, g = shared_case(t, 16)
    si, s = w2.shape[1], 0.5

    def unsloth(x, a13, b13, a2, b2):
        lin = lambda inp, w, a, b: torch.addmm(F.linear(inp, w), F.linear(inp, a), b.T, alpha=s)
        h = F.silu(lin(x, w13[:si], a13[0], b13[0])) * lin(x, w13[si:], a13[1], b13[1])
        return lin(h, w2, a2, b2)

    def fp64(x, a13, b13, a2, b2):
        lin = lambda inp, w, a, b: inp @ w.double().T + s * (inp @ a.T) @ b.T
        h = F.silu(lin(x, w13[:si], a13[0], b13[0])) * lin(x, w13[si:], a13[1], b13[1])
        return lin(h, w2, a2, b2)

    def grads(fn, dtype=None):
        leaves = [v.detach().to(dtype or v.dtype).requires_grad_() for v in (x, *lora)]
        return torch.autograd.grad(fn(*leaves), leaves, g.to(dtype or g.dtype))

    torch._dynamo.reset()
    truth = grads(fp64, torch.float64)
    compiled = grads(torch.compile(unsloth, dynamic=False))
    ours = grads(lambda x, *f: shared_mlp(x, w13, w2, f, torch.tensor(s, device="cuda")))
    for j, (a, c, ref) in enumerate(zip(ours, compiled, truth, strict=True)):
        assert rel(a, ref) <= 1.02 * rel(c, ref) + 1e-5, (j, rel(a, ref), rel(c, ref))


@pytest.mark.parametrize("t", [1, 4, 24, 64, 300])
@pytest.mark.parametrize("rank", [16, 64, 136])
@pytest.mark.parametrize("shared", [False, True])
def test_shared_a_inference_is_separate_with_equal_a_bitwise(t, rank, shared):
    """Shared-A gate_up (one shrink) gives the bits of the separate layout run with A_gate ==
    A_up and B = [B_gate; B_up], on every inference path (fused decode, overlap, aligned,
    general split shrink), with and without the gated shared expert."""
    torch.manual_seed(t + rank)
    e, h, i, si = 64, 512, 256, 256
    x, w13, w2, rw, ids, lora = shared_a_case(t, e, h, i, rank, True, shared=True)
    a, b, a2, b2 = lora.leaves
    twin = MoELoRAWeights.separate(a, a.clone(), b[0].clone(), b[1].clone(), a2, b2, *lora.shared)
    shared_w = (rand(2 * si, h), rand(h, si), rand(1, h)) if shared else None
    kw = {"scaling": 0.5, "provider": "triton", "shared_expert": shared_w}
    got = fused_experts_lora(x, w13, w2, rw, ids, lora, **kw)
    want = fused_experts_lora(x, w13, w2, rw, ids, twin, **kw)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("pairs", [64, 512])  # the pair kernel and the aligned block path
def test_inference_silu_is_torch_silu_bitwise(pairs):
    """The fused SwiGLU of the LoRA gate/up expand is F.silu(g) * u in FP32, rounded once: with
    a zero B the cache keeps its bits, so with u = 1 the activation is torch's SiLU of every
    finite BF16 gate value (subnormal outputs included)."""
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = codes[torch.isfinite(codes)].cuda()
    e, h, rank, top_k = 8, 256, 16, 2
    i = triton.cdiv(finite.numel(), pairs)
    i = triton.cdiv(i, 16) * 16
    t = pairs // top_k
    ids, _ = routing(t, top_k, e, masked=False)
    x = rand(t, h, scale=1.0)
    a, b = rand(e, rank, h), torch.zeros(2, e, i, rank, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(pairs * i, dtype=torch.bfloat16, device="cuda")
    gate[: finite.numel()] = finite
    c1 = torch.cat((gate.view(pairs, i), torch.ones_like(gate).view(pairs, i)), 1)
    act = torch.empty(pairs, i, dtype=torch.bfloat16, device="cuda")
    moe_lora_w13_act_(act, c1, x, (a, b), ids, scaling=0.5, top_k=top_k, provider="triton",
                      **alignment(ids, e, 16))  # fmt: skip
    want = F.silu(gate.view(pairs, i).float()).bfloat16()
    assert torch.equal(act.view(torch.int16), want.view(torch.int16))


def test_training_act_is_torch_bitwise():
    """The training forward's activation is torch's F.silu(gate) * up on BF16 tensors, bit for
    bit, on every finite BF16 gate value (subnormal SiLU outputs included)."""
    from ctikernels.moe.triton._lora_train import _act_kernel

    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = codes[torch.isfinite(codes)].cuda()
    i = 512
    p = triton.cdiv(finite.numel(), i)
    gate = torch.zeros(p * i, dtype=torch.bfloat16, device="cuda")
    gate[: finite.numel()] = finite
    c1 = torch.cat((gate.view(p, i), rand(p, i, scale=2.0)), 1)
    c2 = torch.empty(p, i, dtype=torch.bfloat16, device="cuda")
    _act_kernel[(p, triton.cdiv(i, 1024))](c1, c2, i, BN=1024, num_warps=4)
    want = F.silu(c1[:, :i]) * c1[:, i:]
    assert torch.equal(c2.view(torch.int16), want.view(torch.int16))


def test_training_act_backward_silu_is_torch_bitwise():
    """The training act backward recomputes c2 = BF16(BF16(silu(g)) * u) with torch's F.silu: with
    u = 1 every finite BF16 gate value gives F.silu's BF16 bits (subnormals included)."""
    from ctikernels.moe.triton._lora_backward import TritonSteps

    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite = codes[torch.isfinite(codes)].cuda()
    i = 256
    p = triton.cdiv(finite.numel(), i)
    gate = torch.zeros(p * i, dtype=torch.bfloat16, device="cuda")
    gate[: finite.numel()] = finite
    c1 = torch.cat((gate.view(p, i), torch.ones_like(gate).view(p, i)), 1)
    ids = torch.zeros(p, 1, dtype=torch.int32, device="cuda")
    dc2 = torch.zeros(p, i, dtype=torch.float32, device="cuda")
    _, _, c2 = TritonSteps.act(c1, dc2, None, ids, 1, need_dc1=False, need_dw=False,
                               need_c2=True, round_silu=True)  # fmt: skip
    want = F.silu(gate.view(p, i).float()).bfloat16().float()
    assert torch.equal(c2, want)


@pytest.mark.parametrize("t", [1, 4, 64])
@pytest.mark.parametrize("masked", [False, True])
def test_block_fp8_training_is_deterministic_and_tracks_the_dequantized_base(t, masked):
    """FP8-base training (STE through the activation quantizers): bitwise run to run, and
    within the activation-quantization noise of training on the BF16-dequantized weights."""
    torch.manual_seed(t)
    experts, h, i, rank = 64, 512, 256, 16
    x, w13, w2, rw, ids, lora = shared_a_case(t, experts, h, i, rank, masked)
    q13, s13, w13d = block_quantize(w13)
    q2, s2, w2d = block_quantize(w2)
    dy = rand(t, h, scale=1.0)

    def grads(fp8):
        leaves = [v.detach().clone().requires_grad_() for v in (x, rw, *lora.leaves)]
        base = (q13, q2) if fp8 else (w13d.bfloat16(), w2d.bfloat16())
        kw = {"w13_scale": s13, "w2_scale": s2} if fp8 else {}
        out = fused_experts_lora(
            leaves[0], *base, leaves[1], ids, MoELoRAWeights(*leaves[2:]), scaling=0.5,
            provider="triton", **kw,
        )  # fmt: skip
        return torch.autograd.grad(out, leaves, dy)

    first, second, reference = grads(True), grads(True), grads(False)
    for a, b, ref in zip(first, second, reference, strict=True):
        assert torch.equal(a, b)
        assert rel(a, ref.double()) <= 6e-2
