# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Run the fast subset with -m 'not exhaustive'; full qualification uses -m exhaustive."""

import gc

import pytest
import torch
import torch.nn.functional as F

from ctikernels.moe import to_local_topk_ids
from ctikernels.moe._lora_weights import FactorSlices
from ctikernels.moe.cute._lora_provider import available
from ctikernels.moe.lora import (
    MoELoRAWeights,
    fused_experts_lora,
    moe_lora_w2_,
    moe_lora_w13_,
    moe_lora_w13_act_,
    shared_expert_lora,
    shared_lora_w2_,
    shared_lora_w13_,
)
from ctikernels.moe.torch._lora_reference import oracle

from .._moe_lora_factors import leaves13, packed, separate

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not available(),
    reason="CUDA and nvidia-cutlass-dsl 4.8.0 are required",
)


@pytest.fixture(autouse=True)
def device_and_precision():
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("MoE LoRA CuTe targets sm_120/sm_121")
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def rand(*shape, scale=0.05):
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).bfloat16()


def kernel_args(a, b):
    """A compiled kernel's factor arguments for packed [E,S,R,C] factors: slice zero of a
    slice-major copy and the element gap to slice one (fresh copies: call per launch)."""
    args = []
    for factor in (a, b):
        major = factor.movedim(1, 0).contiguous()
        args += [major[0], major.stride(0) if major.shape[0] > 1 else 0]
    return args


def workspace_case(t, rank, global_e, h, i, ep_rank, with_lora, *, id_dtype=torch.int32):
    e = global_e if ep_rank is None else global_e // 2
    x, w13, w2 = rand(t, h), rand(e, 2 * i, h), rand(e, h, i)
    ids = torch.randint(global_e, (t, 8), dtype=id_dtype, device="cuda")
    if ep_rank is not None:
        expert_map = torch.full((global_e,), -1, dtype=torch.int32, device="cuda")
        expert_map[ep_rank * e : (ep_rank + 1) * e] = torch.arange(e, device="cuda")
        ids = to_local_topk_ids(ids, expert_map).to(id_dtype)
        # Always exercise remote routes, even in decode; include wide int64 ids.
        ids[:, 0] = -1
        ids[:, -1] = (1 << 32) if id_dtype == torch.int64 else e
    rw = torch.rand(t, 8, device="cuda")
    lora = (
        packed(rand(e, 2, rank, h), rand(e, 2, i, rank), rand(e, rank, i), rand(e, h, rank))
        if with_lora
        else None
    )
    return (x, w13, w2, rw, ids, lora), {
        "scaling": 0.7,
        "provider": "cute",
        "local_only": ep_rank is None,
        "num_global_experts": global_e,
    }


@pytest.mark.parametrize("t", [1, 64, 1024])
@pytest.mark.parametrize("rank", [16, 128])
def test_separate_gate_up_leaves(t, rank):
    """Gate/up leaves in separate allocations (any gate-to-up offset) give the bits of
    slice-major storage; the separate layout is inference-only."""
    torch.manual_seed(13)
    (x, w13, w2, rw, ids, lora), kwargs = workspace_case(t, rank, 128, 2048, 768, 0, True)
    up_a, up_b = lora.up_a.clone(), lora.up_b.clone()  # allocated before gate
    reordered = MoELoRAWeights.separate(
        lora.gate_a.clone(), up_a, lora.gate_b.clone(), up_b, *lora.routed[4:]
    )
    expected = fused_experts_lora(x, w13, w2, rw, ids, lora, **kwargs)
    for weights in (separate(lora), reordered):
        actual = fused_experts_lora(x, w13, w2, rw, ids, weights, **kwargs)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    params = [f.detach().clone().requires_grad_() for f in reordered.routed]
    with pytest.raises(ValueError, match="inference-only"):
        fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights.separate(*params), **kwargs)


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("t", [1, 19])  # t=1: naive alignment (t * top_k * 4 <= e)
def test_routed_training_after_inference_mode_serving(monkeypatch, provider, t):
    """Serving under inference_mode creates the cached scaling tensor (and, at small T, the
    naive alignment); a later training call saves them for backward."""
    from ctikernels.moe import _lora_common
    from ctikernels.moe import lora as lora_module

    if provider == "cute" and not available():
        pytest.skip("CuTe MoE LoRA provider unavailable")
    monkeypatch.setattr(_lora_common, "_SCALES", {})
    monkeypatch.setattr(lora_module, "_NAIVE", {})
    torch.manual_seed(31)
    (x, w13, w2, rw, ids, lora), kwargs = workspace_case(t, 16, 128, 512, 128, None, True)
    kwargs["provider"] = provider
    lora = MoELoRAWeights(lora.gate_a, torch.stack((lora.gate_b, lora.up_b)), *lora.routed[4:])
    with torch.inference_mode():
        fused_experts_lora(x, w13, w2, rw, ids, lora, **kwargs)
    params = [f.detach().clone().requires_grad_() for f in lora.leaves]
    out = fused_experts_lora(x, w13, w2, rw, ids, MoELoRAWeights(*params), **kwargs)
    grads = torch.autograd.grad(out.float().square().sum(), params)
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("provider", ["triton", "cute"])
def test_shared_training_after_inference_mode_serving(monkeypatch, provider):
    from ctikernels.moe import _lora_common

    if provider == "cute" and not available():
        pytest.skip("CuTe MoE LoRA provider unavailable")
    monkeypatch.setattr(_lora_common, "_SCALES", {})
    torch.manual_seed(37)
    h, rank = 128, 16
    shared = (rand(2, rank, h), rand(2, 32, rank), rand(rank, 32), rand(h, rank))
    x, sw13, sw2 = rand(3, h, scale=0.3), rand(64, h), rand(h, 32)
    with torch.inference_mode():
        shared_expert_lora(x, sw13, sw2, shared, 0.7, provider=provider)
    params = tuple(t.detach().clone().requires_grad_() for t in shared)
    out = shared_expert_lora(x, sw13, sw2, params, 0.7, provider=provider)
    grads = torch.autograd.grad(out.float().square().sum(), params)
    assert all(torch.isfinite(g).all() for g in grads)


def check_workspace_peak_and_bits(monkeypatch, args, kwargs):
    from ctikernels.moe import _lora_workspace as _workspace

    monkeypatch.setattr(_workspace, "ENABLED", True)
    warm = fused_experts_lora(*args, **kwargs)
    del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fused_experts_lora(*args, **kwargs)
    torch.cuda.synchronize()
    # The output's cached block may be larger than the output: count what stays live.
    peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    assert torch.cuda.memory_allocated() - before >= out.numel() * out.element_size()
    # Alignment is call-owned (rc7's moe_align_block_size takes no output buffers): base
    # and LoRA alignments, each int32 ids padded per local expert at BM <= 128.
    pairs, experts = args[4].numel(), args[1].shape[0]
    alignment = 2 * 4 * (2 * (pairs + (experts + 1) * 127) + experts + 3)
    assert peak <= 64 * 1024 + alignment, (
        f"steady-state scratch peak: {peak} bytes; shape={args[0].shape}"
    )
    monkeypatch.setattr(_workspace, "ENABLED", False)
    expected = fused_experts_lora(*args, **kwargs)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("t", [1, 19, 447, 2113])
@pytest.mark.parametrize("rank", [16, 128])
@pytest.mark.parametrize("ep_rank", [None, 0, 1])
@pytest.mark.parametrize("with_lora", [False, True])
def test_resident_workspace_small_expert_count(monkeypatch, t, rank, ep_rank, with_lora):
    args, kwargs = workspace_case(t, rank, 16, 2048, 128, ep_rank, with_lora)
    check_workspace_peak_and_bits(monkeypatch, args, kwargs)


@pytest.mark.parametrize("ep_rank", [0, 1])
@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("t,rank", [(1, 16), (19, 128), (2113, 128)])
def test_resident_workspace_skipped_routes_int64(monkeypatch, ep_rank, with_lora, t, rank):
    args, kwargs = workspace_case(t, rank, 128, 256, 128, ep_rank, with_lora, id_dtype=torch.int64)
    check_workspace_peak_and_bits(monkeypatch, args, kwargs)


@pytest.mark.exhaustive
@pytest.mark.parametrize("t,rank,experts,i", [(19, 16, 256, 512), (2113, 128, 128, 768)])
@pytest.mark.parametrize("ep_rank", [None, 0, 1])
@pytest.mark.parametrize("with_lora", [False, True])
def test_resident_workspace_model_shapes(monkeypatch, t, rank, experts, i, ep_rank, with_lora):
    from ctikernels.moe import _lora_workspace as _workspace

    _workspace.release()
    gc.collect()
    torch.cuda.empty_cache()
    local_e = experts if ep_rank is None else experts // 2
    # Include the temporary FP32 random-weight construction and both outputs.
    required = local_e * 3 * i * 2048 * 8 + t * 8 * (2048 + 3 * i) * 4
    free, _ = torch.cuda.mem_get_info()
    if free < required:
        pytest.skip(f"workspace model row needs approximately {required / 2**30:.1f} GiB free")
    args, kwargs = workspace_case(t, rank, experts, 2048, i, ep_rank, with_lora)
    check_workspace_peak_and_bits(monkeypatch, args, kwargs)


@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("ep_rank", [None, 0, 1])
def test_resident_workspace_forced_split(monkeypatch, rank, ep_rank):
    from ctikernels.moe import lora as _ops

    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", 4)
    # Each z buffer exceeds the peak allowance, catching allocations off the arena.
    args, kwargs = workspace_case(382, rank, 16, 256, 128, ep_rank, True)
    check_workspace_peak_and_bits(monkeypatch, args, kwargs)


@pytest.mark.exhaustive
@pytest.mark.parametrize("experts,h,i", [(256, 2048, 512), (16, 256, 128)])
@pytest.mark.parametrize("rank", [32, 128])
@pytest.mark.parametrize("t", [19, 64, 382])
@pytest.mark.parametrize("ep_rank", [None, 0, 1])
@pytest.mark.parametrize("concentrated", [False, True])
def test_routed_split_chunks_bitwise(monkeypatch, experts, h, i, rank, t, ep_rank, concentrated):
    from ctikernels.moe import lora as _ops

    torch.manual_seed(83)
    args, kwargs = workspace_case(t, rank, experts, h, i, ep_rank, True)
    if concentrated:
        # Repeated top-8 slots use just two global experts, one per EP shard.
        ids = (torch.arange(8, device="cuda", dtype=torch.int32) % 2) * (experts // 2)
        ids = ids.expand(t, 8).contiguous()
        if ep_rank is not None:
            local_e = experts // 2
            ids = ids - ep_rank * local_e
            ids = torch.where((ids >= 0) & (ids < local_e), ids, -1)
        args = (*args[:4], ids, args[5])
    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", 0)
    expected = fused_experts_lora(*args, **kwargs)
    # The fused block already converts shrink accumulators to BF16. Saving and
    # restoring that z leaves the expand arithmetic and per-element order intact.
    for chunks in (1, 2, 4, 8):
        monkeypatch.setattr(_ops, "_SPLIT_ROUTED", chunks)
        actual = fused_experts_lora(*args, **kwargs)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), chunks
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_experts_lora(*args, **kwargs)
        graph.replay()
        assert torch.equal(captured.view(torch.int16), expected.view(torch.int16)), chunks


def test_resident_workspace_graph_survives_growth(monkeypatch):
    from ctikernels.moe import _lora_workspace as _workspace

    monkeypatch.setattr(_workspace, "ENABLED", True)
    args, kwargs = workspace_case(64, 16, 16, 128, 64, 1, True)
    large_args, _ = workspace_case(2048, 16, 16, 128, 64, 1, True)
    large_args = (large_args[0], args[1], args[2], large_args[3], large_args[4], args[5])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        warm = fused_experts_lora(*args, **kwargs)
        del warm
    stream.synchronize()
    with torch.cuda.stream(stream):
        key = _workspace._key(args[0].device)
        original = _workspace._ARENAS[key]
    with torch.cuda.graph(graph, stream=stream):
        captured = fused_experts_lora(*args, **kwargs)
    with torch.cuda.stream(stream):
        large = fused_experts_lora(*large_args, **kwargs)
        del large
        assert _workspace._ARENAS[key].buffer.numel() > original.buffer.numel()
        assert any(arena is original for arena in _workspace._RETIRED)
        graph.replay()
        expected = fused_experts_lora(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))


def grouped_oracle(before, x, a, b, ids, rw, top_k, scale):
    """FP32 oracle, grouped only to keep the large qualification sweep affordable."""
    delta = torch.zeros_like(before, dtype=torch.float32)
    z = torch.zeros((before.shape[0], a.shape[1], a.shape[2]), device="cuda")
    host_ids = ids.flatten().cpu()
    for e in host_ids.unique().tolist():
        if not 0 <= e < a.shape[0]:
            continue
        pairs = torch.where(host_ids == e)[0].to("cuda")
        rows = pairs // top_k if rw is None else pairs
        for s in range(a.shape[1]):
            shrink = x[rows].float() @ a[e, s].float().T
            expand = shrink @ b[e, s].float().T
            weight = 1 if rw is None else rw.flatten()[pairs, None]
            n = b.shape[-2]
            delta[pairs, s * n : (s + 1) * n] = scale * weight * expand
            z[pairs, s] = shrink
    return (before.float() + delta).bfloat16(), delta, z


def assert_delta(actual, before, expected, delta, *, rounded_z=False):
    # Small nonzero initial caches make the relative delta metric meaningful.
    # Final-value tolerance additionally allows one BF16 ULP at each magnitude.
    observed = actual.float() - before.float()
    denom = delta.norm().clamp_min(1e-12)
    assert float((observed - delta).norm() / denom) <= 1e-2
    if not rounded_z:
        magnitude = expected.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
        ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
        assert torch.all((actual.float() - expected.float()).abs() <= ulp + 1e-6)
    # Rounding z can perturb a near-cancelled expand by many output ULPs. For
    # MMA use the delta L2 criterion above, plus exact preservation of zero
    # contributions, rather than an unsound elementwise one-ULP bound.
    # Only rows without any contribution are bit-exact; a valid row can hit an
    # exactly cancelled FP32 oracle element that BF16 z leaves at a tiny nonzero.
    zero = (delta == 0).all(dim=-1)
    assert torch.equal(actual.view(torch.int16)[zero], before.view(torch.int16)[zero])


def run_case(
    t,
    rank,
    experts,
    down,
    routing="mixed",
    *,
    adapter=False,
    block_m=32,
    zdtype=None,
    hidden=2048,
    intermediate=512,
):
    torch.manual_seed(90)
    h, i, top_k = hidden, intermediate, 8
    k, n, slices = (i, h, 1) if down else (h, i, 2)
    a, b = rand(experts, slices, rank, k), rand(experts, slices, n, rank)
    ids = torch.randint(0, experts, (t, top_k), device="cuda", dtype=torch.int32)
    ids[ids >= experts // 2] = -1
    if routing == "skewed":
        ids.fill_(1)
    elif routing == "masked":
        ids.fill_(-1)
    # Include both negative and too-large local ids.
    if routing == "mixed":
        ids[0, 0] = experts + 3
    rw = torch.rand((t, top_k), device="cuda", dtype=torch.float32)
    inp = rand(t * top_k if down else t, k, scale=0.3)
    before = rand(t * top_k, slices * n, scale=0.0001)
    expected, delta, saved = grouped_oracle(
        before, inp, a, b, ids, rw if down else None, top_k, 0.7
    )
    factors = (a[:, 0], b[:, 0]) if down else leaves13(a, b)
    if adapter:
        factors = tuple(torch.stack((torch.zeros_like(f), f)) for f in factors)
    z = None if zdtype is None else torch.full(saved.shape, -9, dtype=zdtype, device="cuda")
    kwargs = {
        "scaling": 0.7,
        "top_k": top_k,
        "provider": "cute",
        "adapter_id": int(adapter),
        "z_out": z,
    }
    if t * top_k > 128:
        from ctikernels.moe_impl import moe_align_block_size

        safe = torch.where((ids >= 0) & (ids < experts), ids, 0)
        sorted_ids, block_experts, padded = moe_align_block_size(safe, block_m, experts)
        kwargs.update(
            sorted_token_ids=sorted_ids,
            expert_ids=block_experts,
            num_tokens_post_padded=padded,
            block_m=block_m,
        )
    actual = before.clone()
    op = moe_lora_w2_ if down else moe_lora_w13_
    extra = (rw,) if down else ()
    op(actual, inp, factors, ids, *extra, **kwargs)
    assert_delta(actual, before, expected, delta, rounded_z=t * top_k > 128)
    valid = ((ids >= 0) & (ids < experts)).flatten()
    assert torch.equal(actual[~valid], before[~valid])
    if z is not None:
        torch.testing.assert_close(
            z[valid].float(),
            saved[valid],
            # FP32 z: MMA and SIMT accumulation orders differ from the oracle's.
            atol=0.005 if zdtype == torch.bfloat16 else 2e-5,
            rtol=0.004 if zdtype == torch.bfloat16 else 1e-4,
        )
        assert torch.all(z[~valid] == -9)
    # Every compiled specialization also gets an all-masked and zero-B no-op check.
    ids.fill_(-1)
    untouched = before.clone()
    op(untouched, inp, factors, ids, *extra, **kwargs)
    assert torch.equal(untouched, before)
    ids.fill_(0)
    for factor in factors[2:] if len(factors) == 4 else factors[1:]:
        factor.zero_()
    if t * top_k > 128:
        # Rebuild after changing routing so the zero-adapter check actually
        # visits every row (the previous alignment belongs to the old ids).
        sorted_ids, block_experts, padded = moe_align_block_size(
            ids.to(torch.int32), block_m, experts
        )
        kwargs.update(
            sorted_token_ids=sorted_ids, expert_ids=block_experts, num_tokens_post_padded=padded
        )
    untouched = before.clone()
    untouched[0, 0] = -0.0
    zero_expected = untouched.clone()
    op(untouched, inp, factors, ids, *extra, **kwargs)
    assert torch.equal(untouched.view(torch.int16), zero_expected.view(torch.int16))


@pytest.mark.parametrize("t,rank,experts", [(1, 8, 8), (8, 96, 8), (64, 32, 8), (3, 16, 256)])
@pytest.mark.parametrize("down", [False, True])
def test_fast_projections(t, rank, experts, down):
    run_case(
        t,
        rank,
        experts,
        down,
        routing="skewed" if t == 64 else "mixed",
        adapter=t == 3,
        zdtype=torch.float32,
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [1, 2, 3, 4, 8, 16, 64, 256, 1024])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("experts", [8, 256])
@pytest.mark.parametrize("down", [False, True])
def test_full_shapes(t, rank, experts, down):
    run_case(t, rank, experts, down, block_m=(16, 32, 64)[rank % 3], zdtype=torch.bfloat16)


@pytest.mark.parametrize("t", [1, 2, 8, 16, 64, 256, 1024, 4096])
@pytest.mark.parametrize("rank", [16, 32, 64])
@pytest.mark.parametrize("down", [False, True])
def test_qwen3_30b_routed_shapes(t, rank, down):
    run_case(t, rank, 128, down, hidden=2048, intermediate=768, zdtype=torch.bfloat16)


@pytest.mark.exhaustive
@pytest.mark.parametrize("routing", ["skewed", "masked"])
@pytest.mark.parametrize("t", [2, 256])
@pytest.mark.parametrize("down", [False, True])
def test_routing_extremes(routing, t, down):
    run_case(t, 128, 256, down, routing)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("t", [2, 64])
def test_live_scale_graph(down, t):
    torch.manual_seed(4)
    top_k, e, rank, k, n = 8, 8, 8, 128, 64
    slices = 1 if down else 2
    a, b = rand(e, slices, rank, k), rand(e, slices, n, rank)
    factors = (a[:, 0], b[:, 0]) if down else leaves13(a, b)
    ids = torch.randint(0, e, (t, top_k), device="cuda", dtype=torch.int64)
    ids[0, 0] = 1 << 32
    rw = torch.rand((t, top_k), device="cuda")
    x = rand(t * top_k if down else t, k, scale=0.3)
    before = rand(t * top_k, slices * n, scale=0.0001)
    out = before.clone()
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    op = moe_lora_w2_ if down else moe_lora_w13_
    extra = (rw,) if down else ()
    # Compile outside capture on a side stream; capture and replay read the same address.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(out, x, factors, ids, *extra, scaling=scale, top_k=top_k, provider="cute")
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out.copy_(before)
        op(out, x, factors, ids, *extra, scaling=scale, top_k=top_k, provider="cute")
    for value in (0.0, 0.5, 1.25, -0.75):
        scale.fill_(value)
        graph.replay()
        expected, delta, _ = grouped_oracle(
            before, x, a, b, ids, rw if down else None, top_k, value
        )
        assert_delta(out, before, expected, delta, rounded_z=t * top_k > 128)


def test_shared_and_routed_compositions():
    torch.manual_seed(25)
    e, h, i, rank, t, top_k = 8, 128, 64, 16, 3, 2
    w = packed(
        rand(e, 2, rank, h),
        rand(e, 2, i, rank),
        rand(e, rank, i),
        rand(e, h, rank),
        as13=rand(2, rank, h),
        bs13=rand(2, 32, rank),
        as2=rand(rank, 32),
        bs2=rand(h, rank),
    )
    x = rand(t, h, scale=0.3)
    base13, base2 = rand(e, 2 * i, h), rand(e, h, i)
    ids = torch.tensor([[0, 1], [-1, 3], [e, -1]], dtype=torch.int32, device="cuda")
    rw = torch.rand(t, top_k, device="cuda")
    # Independent whole-layer FP32 oracle, including each BF16 cache boundary.
    c1 = torch.zeros(t * top_k, 2 * i, device="cuda", dtype=torch.bfloat16)
    c3 = torch.zeros(t * top_k, h, device="cuda", dtype=torch.bfloat16)
    for p, expert in enumerate(ids.flatten().tolist()):
        if 0 <= expert < e:
            c1[p] = (x[p // top_k].float() @ base13[expert].float().T).bfloat16()
    c1, _, _ = oracle(c1, x, w.a13, w.b13, ids, scaling=0.7, top_k=top_k, dtype=torch.float32)
    g, u = c1.float().chunk(2, -1)
    c2 = (F.silu(g) * u).bfloat16()
    for p, expert in enumerate(ids.flatten().tolist()):
        if 0 <= expert < e:
            c3[p] = ((c2[p].float() @ base2[expert].float().T) * rw.flatten()[p]).bfloat16()
    c3, _, _ = oracle(
        c3,
        c2,
        w.a2[:, None],
        w.b2[:, None],
        ids,
        scaling=0.7,
        top_k=top_k,
        routed_weights=rw,
        dtype=torch.float32,
    )
    expected = c3.reshape(t, top_k, h).float().sum(1).bfloat16()
    actual = fused_experts_lora(x, base13, base2, rw, ids, w, scaling=0.7, provider="cute")
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.02)
    assert torch.count_nonzero(actual[-1]) == 0
    sw13, sw2 = rand(64, h), rand(h, 32)
    sc1 = (x.float() @ sw13.float().T).bfloat16()
    expected1, _, _ = oracle(
        sc1, x, w.as13[None], w.bs13[None], None, scaling=0.7, top_k=1, dtype=torch.float32
    )
    shared_lora_w13_(sc1, x, w, scaling=0.7, provider="cute")
    torch.testing.assert_close(sc1, expected1, atol=0.002, rtol=0.008)
    sg, su = expected1.float().chunk(2, -1)
    sc2 = (F.silu(sg) * su).bfloat16()
    base = (sc2.float() @ sw2.float().T).bfloat16()
    sexpected, _, _ = oracle(
        base,
        sc2,
        w.as2[None, None],
        w.bs2[None, None],
        None,
        scaling=0.7,
        top_k=1,
        dtype=torch.float32,
    )
    shared_lora_w2_(base, sc2, w.shared, scaling=0.7, provider="cute")
    torch.testing.assert_close(base, sexpected, atol=2e-4, rtol=0.02)
    torch.testing.assert_close(
        shared_expert_lora(x, sw13, sw2, w.shared, 0.7, provider="cute"),
        sexpected,
        atol=2e-4,
        rtol=0.02,
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [1, 3, 64])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("intermediate", [256, 512])
def test_shared_full_shapes(t, rank, intermediate):
    torch.manual_seed(19)
    for down in (False, True):
        k, n, slices = (intermediate, 2048, 1) if down else (2048, intermediate, 2)
        a, b = rand(1, slices, rank, k), rand(1, slices, n, rank)
        x = rand(t, k, scale=0.3)
        before = rand(t, slices * n, scale=0.0001)
        ids = torch.zeros((t, 1), dtype=torch.int32, device="cuda")
        expected, delta, _ = grouped_oracle(before, x, a, b, ids, None, 1, 0.7)
        out = before.clone()
        if down:
            shared_lora_w2_(out, x, (a[0, 0], b[0, 0]), scaling=0.7, provider="cute")
        else:
            shared_lora_w13_(out, x, (a[0], b[0]), scaling=0.7, provider="cute")
        # Shared rows above 16 tokens, and rows the measured table sends to the
        # dense/library kernels, consume BF16 z; decode CuTe pairs keep FP32 z.
        from ctikernels.moe import lora as _ops

        mode, _ = _ops._shared_choice(x, t, rank, k, n, slices)
        assert_delta(out, before, expected, delta, rounded_z=t > 16 or mode != "cute")


@pytest.mark.parametrize("act", [False, True])
def test_alignment_padding_and_negative_block_expert(act):
    # A synthetic aligned schedule tests skipped blocks, pair padding and unused
    # capacity independently of the base aligner's choice of padding sentinel.
    t, top_k, bm, rank, k, n = 65, 2, 16, 8, 32, 24
    ids = torch.full((t, top_k), -1, dtype=torch.int64, device="cuda")
    ids[0, 0] = 0
    ids[0, 1] = 1 << 32  # Must not narrow to expert zero.
    a, b = rand(2, 2, rank, k), rand(2, 2, n, rank)
    x, before = rand(t, k, scale=0.3), rand(t * top_k, 2 * n, scale=0.0001)
    sorted_ids = torch.full((bm * 3,), t * top_k, dtype=torch.int32, device="cuda")
    sorted_ids[0], sorted_ids[1], sorted_ids[bm] = 0, 1, 2
    experts = torch.tensor([0, -1, 0], dtype=torch.int32, device="cuda")
    padded = torch.tensor([bm * 2], dtype=torch.int32, device="cuda")
    out = (
        torch.full((t * top_k, n), -19, dtype=before.dtype, device="cuda")
        if act
        else before.clone()
    )
    op = moe_lora_w13_act_ if act else moe_lora_w13_
    extra = (before,) if act else ()
    op(
        out,
        *extra,
        x,
        leaves13(a, b),
        ids,
        scaling=0.7,
        top_k=top_k,
        provider="cute",
        sorted_token_ids=sorted_ids,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        block_m=bm,
    )
    expected, delta, _ = grouped_oracle(before, x, a, b, ids, None, top_k, 0.7)
    if act:
        gate, up = (before.float() + delta).chunk(2, -1)
        expected = (F.silu(gate) * up).bfloat16()
        torch.testing.assert_close(out[:1], expected[:1], atol=2e-5, rtol=0.02)
        assert torch.all(out[1:] == -19)
    else:
        assert_delta(out, before, expected, delta, rounded_z=True)
        assert torch.equal(out[1:], before[1:])


@pytest.mark.parametrize(
    "rank,block_m,down,zdtype",
    [
        (8, 16, False, torch.float32),
        (16, 32, True, torch.bfloat16),
        (64, 64, False, None),
        (96, 16, True, torch.float32),
    ],
)
def test_block_fast(rank, block_m, down, zdtype):
    run_case(64, rank, 8, down, block_m=block_m, zdtype=zdtype)


@pytest.mark.parametrize("t", [64, 512])
@pytest.mark.parametrize("rank", [8, 16, 32, 128])
@pytest.mark.parametrize("bm", [16, 32])
@pytest.mark.parametrize("down", [False, True], ids=["w13_act", "w2"])
@pytest.mark.parametrize("shrink_stages", [4, 6])
def test_block_k128_pipeline(t, rank, bm, down, shrink_stages):
    from ctikernels.moe.cute._lora_provider import block_pipeline, compile_kernel
    from ctikernels.moe_impl import moe_align_block_size

    torch.manual_seed(53)
    e, top_k = 4, 8
    k, n, slices = (512, 2048, 1) if down else (2048, 512, 2)
    a, b = rand(e, slices, rank, k), rand(e, slices, n, rank)
    x = rand(t * top_k if down else t, k, scale=0.3)
    ids = torch.randint(e, (t, top_k), dtype=torch.int64, device="cuda")
    ids[::3, 0], ids[0, 1], ids[-1] = -1, 1 << 32, -1
    valid = ((ids >= 0) & (ids < e)).flatten()
    safe = torch.where(valid.view_as(ids), ids, 0).int()
    alignment = moe_align_block_size(safe, bm, e)
    rw = torch.rand(t, top_k, device="cuda") if down else None
    before = rand(t * top_k, slices * n, scale=0.0001)
    scale = torch.tensor([0.7], device="cuda")
    bn, stages = block_pipeline(bm, rank)

    def run(block_k, depth):
        out = before.clone()
        activated = None if down else torch.full_like(out[:, :n], -19)
        kernel = compile_kernel(
            k,
            n,
            rank,
            slices,
            top_k,
            e,
            torch.cuda.get_device_capability(),
            down=down,
            act=not down,
            block_m=bm,
            block_n=bn,
            stages=stages,
            block_k=block_k,
            shrink_stages=depth,
            id_dtype=torch.int64,
        )
        kernel(
            x,
            *kernel_args(a, b),
            out,
            ids,
            rw,
            scale,
            None,
            *alignment,
            *((activated,) if not down else ()),
        )
        if not down:
            assert torch.equal(out.view(torch.int16), before.view(torch.int16))
        return out if down else activated

    default = run(64, stages)
    actual = run(128, shrink_stages)
    expected, delta, _ = grouped_oracle(before, x, a, b, ids, rw, top_k, 0.7)
    if down:
        for result in (default, actual):
            assert_delta(result, before, expected, delta, rounded_z=True)
        assert_delta(actual, before, default, default.float() - before.float(), rounded_z=True)
    else:
        gate, up = (before.float() + delta).chunk(2, -1)
        expected = F.silu(gate) * up
        for reference in (expected, default.float()):
            error = (actual[valid].float() - reference[valid]).norm()
            assert error / reference[valid].norm().clamp_min(1e-12) < 0.01
    # Only untouched rows require bitwise equality; BK changes the K reduction.
    assert torch.equal(actual[~valid].view(torch.int16), default[~valid].view(torch.int16))


@pytest.mark.parametrize("down", [False, True], ids=["w13_act", "w2"])
def test_block_pipeline_provider_overrides(down, monkeypatch):
    from ctikernels.moe.cute import _lora_provider as provider

    torch.manual_seed(59)
    t, k, n, rank = 64, 72, 136, 16
    slices = 1 if down else 2
    a, b = rand(1, slices, rank, k), rand(1, slices, n, rank)
    x, before = rand(t, k, scale=0.3), rand(t, slices * n, scale=0.0001)
    scale = torch.tensor([0.7], device="cuda")
    compiled = []
    original_compile = provider.compile_kernel

    def compile_spy(*args, **kwargs):
        compiled.append((kwargs["block_k"], kwargs["shrink_stages"]))
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(provider, "compile_kernel", compile_spy)
    monkeypatch.setattr(provider, "_COMPILED", {})
    overrides = {}
    monkeypatch.setattr(provider, "_BLOCK_OVERRIDES", overrides)

    def run():
        out = before.clone()
        activated = None if down else torch.full_like(out[:, :n], -19)
        return provider.apply_(
            out,
            x,
            *(FactorSlices(f.movedim(1, 0).contiguous().unbind()) for f in (a, b)),
            None,
            scale,
            1,
            down=down,
            block_m=16,
            activated=activated,
        )

    default = run()
    overrides.update(block_k=128, shrink_stages=6)
    actual = run()
    assert (actual.float() - default.float()).norm() / default.float().norm() < 0.01
    overrides.clear()
    assert torch.equal(run().view(torch.int16), default.view(torch.int16))
    # Small-SM parts default to 128-wide K tiles when they fit.
    small = torch.cuda.get_device_properties(0).multi_processor_count < 64
    assert compiled == [(128 if small else 64, 4), (128, 6)]


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [64, 256, 1024, 4096])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("block_m", [16, 32, 64])
@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize(
    "experts,routing,zdtype",
    [(128, "mixed", torch.float32), (256, "skewed", torch.bfloat16)],
)
@pytest.mark.parametrize("block_n,stages", [(64, 3), (64, 4), (128, 3), (128, 4)])
def test_block_prefill_sweep(
    t, rank, block_m, down, experts, routing, zdtype, block_n, stages, monkeypatch
):
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    monkeypatch.setattr(_cute_provider, "block_pipeline", lambda bm, r: (block_n, stages))
    run_case(t, rank, experts, down, routing, block_m=block_m, zdtype=zdtype)


def run_shared_block(
    t, rank, intermediate, down, block_m, zdtype, *, direct=True, block_n=64, stages=3
):
    from ctikernels.moe.cute._lora_provider import compile_kernel

    torch.manual_seed(27)
    k, n, slices = (intermediate, 2048, 1) if down else (2048, intermediate, 2)
    a, b = rand(1, slices, rank, k), rand(1, slices, n, rank)
    x = rand(t, k, scale=0.3)
    before = rand(t, slices * n, scale=0.0001)
    before[0, 0] = -0.0
    ids = torch.zeros((t, 1), device="cuda", dtype=torch.int32)
    expected, delta, saved = grouped_oracle(before, x, a, b, ids, None, 1, 0.7)
    z = torch.full(saved.shape, -9, dtype=zdtype, device="cuda")
    out = before.clone()
    scale = torch.tensor([0.7], device="cuda")
    if direct:
        # Direct launch covers BM16/32/64 and shared T64 independently of the
        # public dispatch threshold, with no alignment or routing allocation.
        kernel = compile_kernel(
            k,
            n,
            rank,
            slices,
            1,
            1,
            torch.cuda.get_device_capability(),
            down=down,
            shared=True,
            block_m=block_m,
            chunks=3,
            block_n=block_n,
            stages=stages,
            z_dtype=zdtype,
        )

        def launch():
            kernel(x, *kernel_args(a, b), out, None, None, scale, z, None, None, None)
    else:
        op = shared_lora_w2_ if down else shared_lora_w13_
        factors = (a[0, 0], b[0, 0]) if down else (a[0], b[0])

        def launch():
            op(out, x, factors, scaling=scale, provider="cute", z_out=z)

    launch()
    assert_delta(out, before, expected, delta, rounded_z=direct or t > 128)
    torch.testing.assert_close(
        z.float(),
        saved,
        atol=0.005 if zdtype == torch.bfloat16 else 2e-5,
        rtol=0.004 if zdtype == torch.bfloat16 else 1e-4,
    )
    for zero_factor in (a, b):
        original = zero_factor.clone()
        zero_factor.zero_()
        out.copy_(before)
        launch()
        if direct:
            assert torch.equal(out.view(torch.int16), before.view(torch.int16))
        else:
            # The shared library path may round -0.0 + 0 to +0.0.
            assert torch.equal(out, before)
        zero_factor.copy_(original)


@pytest.mark.parametrize("down,intermediate", [(False, 256), (True, 512)])
def test_shared_block_dispatch(down, intermediate):
    run_shared_block(256, 128, intermediate, down, 64, torch.float32, direct=False)


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [64, 256, 1024, 4096])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("intermediate", [256, 512])
@pytest.mark.parametrize("block_m", [16, 32, 64])
@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("block_n,stages", [(64, 3), (64, 4), (128, 3), (128, 4)])
def test_shared_block_sweep(t, rank, intermediate, block_m, down, block_n, stages):
    dtype = torch.float32 if down else torch.bfloat16
    run_shared_block(t, rank, intermediate, down, block_m, dtype, block_n=block_n, stages=stages)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("block_n,stages", [(64, 3), (64, 4), (128, 3), (128, 4)])
@pytest.mark.parametrize("block_k,shrink_stages", [(64, None), (128, 4), (128, 6)])
def test_block_vector_tails_and_partial_dense_rows(down, block_n, stages, block_k, shrink_stages):
    from ctikernels.moe.cute._lora_provider import compile_kernel

    # K and rank have eight-element tails; N spans a partial BN64 tile. The
    # final dense row block is partial, and chunks own disjoint output tiles.
    t, k, n, rank, slices = 131, 72, 136, 24, 1 if down else 2
    a, b = rand(1, slices, rank, k), rand(1, slices, n, rank)
    x, before = rand(t, k, scale=0.3), rand(t, slices * n, scale=0.0001)
    ids = torch.zeros((t, 1), device="cuda", dtype=torch.int32)
    expected, delta, _ = grouped_oracle(before, x, a, b, ids, None, 1, 0.7)
    out = before.clone()
    kernel = compile_kernel(
        k,
        n,
        rank,
        slices,
        1,
        1,
        torch.cuda.get_device_capability(),
        down=down,
        shared=True,
        block_m=16,
        block_n=block_n,
        stages=stages,
        block_k=block_k,
        shrink_stages=shrink_stages,
        chunks=2,
    )
    kernel(
        x,
        *kernel_args(a, b),
        out,
        None,
        None,
        torch.tensor([0.7], device="cuda"),
        None,
        None,
        None,
        None,
    )
    assert_delta(out, before, expected, delta, rounded_z=True)


@pytest.mark.parametrize("block_n,stages", [(64, 3), (128, 4)])
@pytest.mark.parametrize("down", [False, True])
def test_block_pipeline_zero_elements(block_n, stages, down):
    from ctikernels.moe.cute._lora_block import BlockKernel
    from ctikernels.moe.cute._lora_provider import compile_kernel

    # Enough output tiles to wrap both rings, with a partial final row block.
    # Alternating zero-B columns exercise each half of the packed epilogue.
    t, k, n, rank, bm = 65, 128, 520, 16, 32
    slices = 1 if down else 2
    a, b = rand(1, slices, rank, k), rand(1, slices, n, rank)
    b[:, :, ::2].zero_()
    x = rand(t, k, scale=0.3)
    before = rand(t, slices * n, scale=0.0001)
    ids = torch.zeros((t, 1), dtype=torch.int32, device="cuda")
    expected, delta, saved = grouped_oracle(before, x, a, b, ids, None, 1, 0.7)
    # Signed zero, infinities, and noncanonical NaN payloads must survive.
    patterns = torch.tensor([-32768, 32640, -128, 32705], dtype=torch.int16, device="cuda")
    bits = before.view(torch.int16)
    bits[:, ::2] = patterns[torch.arange(slices * n // 2, device="cuda") % 4]
    out = before.clone()
    z = torch.full_like(saved, -9)
    config = BlockKernel(
        k, n, rank, slices, 1, 1, down=down, shared=True, block_m=bm, block_n=block_n, stages=stages
    )
    assert config.total_r == slices * rank
    assert config.smem_bytes <= 99 * 1024
    kernel = compile_kernel(
        k,
        n,
        rank,
        slices,
        1,
        1,
        torch.cuda.get_device_capability(),
        down=down,
        shared=True,
        block_m=bm,
        block_n=block_n,
        stages=stages,
        z_dtype=torch.float32,
    )
    kernel(
        x,
        *kernel_args(a, b),
        out,
        None,
        None,
        torch.tensor([0.7], device="cuda"),
        z,
        None,
        None,
        None,
    )
    assert torch.equal(out.view(torch.int16)[:, ::2], bits[:, ::2])
    # The untouched exceptional values do not enter the error metric.
    assert_delta(out[:, 1::2], before[:, 1::2], expected[:, 1::2], delta[:, 1::2], rounded_z=True)
    torch.testing.assert_close(z, saved, atol=2e-5, rtol=1e-4)


def run_act_case(t, rank, bm, routing="mixed", *, shared=False, zdtype=torch.float32):
    torch.manual_seed(43)
    e, k, n, top_k = (1 if shared else 8), 256, 136, (1 if shared else 8)
    a, b = rand(e, 2, rank, k), rand(e, 2, n, rank)
    x = rand(t, k, scale=0.3)
    c1 = rand(t * top_k, 2 * n, scale=0.02)
    c1[:, 0] = -0.0
    original = c1.clone()
    ids = torch.randint(e, (t, top_k), device="cuda", dtype=torch.int64)
    if not shared:
        if routing == "skewed":
            ids.fill_(1)
        elif routing == "masked":
            ids.fill_(-1)
        else:
            ids[:, ::3] = -1
            ids[0, 1] = 1 << 32
    valid = ((ids >= 0) & (ids < e)).flatten()
    _, delta, saved = grouped_oracle(c1, x, a, b, ids, None, top_k, 0.7)
    gate, up = (c1.float() + delta).chunk(2, -1)
    expected = F.silu(gate) * up
    c2 = torch.full((t * top_k, n), -19, dtype=torch.bfloat16, device="cuda")
    z = None if zdtype is None else torch.full_like(saved, -29, dtype=zdtype)
    if shared:
        # Explicit BM16/32/64 plus partial rows/chunks, including T <= 128.
        from ctikernels.moe.cute._lora_provider import block_pipeline, compile_kernel

        bn, stages = block_pipeline(bm, rank)
        kernel = compile_kernel(
            k,
            n,
            rank,
            2,
            1,
            1,
            torch.cuda.get_device_capability(),
            shared=True,
            block_m=bm,
            block_n=bn,
            stages=stages,
            chunks=3,
            act=True,
            z_dtype=zdtype,
        )
        scale = torch.tensor([0.7], device="cuda")

        def launch():
            kernel(x, *kernel_args(a, b), c1, None, None, scale, z, None, None, None, c2)
    else:
        from ctikernels.moe_impl import moe_align_block_size

        # Remote pairs ride in expert 0's blocks; the kernel must skip them by id.
        safe = torch.where((ids >= 0) & (ids < e), ids, 0).to(torch.int32)
        sorted_ids, experts, padded = moe_align_block_size(safe, bm, e)

        def launch():
            moe_lora_w13_act_(
                c2,
                c1,
                x,
                leaves13(a, b),
                ids,
                scaling=0.7,
                top_k=top_k,
                sorted_token_ids=sorted_ids,
                expert_ids=experts,
                num_tokens_post_padded=padded,
                block_m=bm,
                provider="cute",
                z_out=z,
            )

    launch()
    # Shrink is BF16 inside the block. Use an L2 bound against the full FP32
    # oracle; near-cancelled elements do not admit an elementwise ULP bound.
    error = (c2[valid].float() - expected[valid]).norm()
    assert error / expected[valid].norm().clamp_min(1e-12) < 0.01
    assert torch.all(c2[~valid] == -19)
    assert torch.equal(c1.view(torch.int16), original.view(torch.int16))
    if z is not None:
        torch.testing.assert_close(
            z[valid].float(),
            saved[valid],
            atol=0.005 if zdtype == torch.bfloat16 else 2e-5,
            rtol=0.004 if zdtype == torch.bfloat16 else 1e-4,
        )
        assert torch.all(z[~valid] == -29)
    # Zero B must retain activation of the base, with no gate/up BF16 round.
    b.zero_()
    c2.fill_(-19)
    launch()
    gate, up = c1.float().chunk(2, -1)
    zero_expected = (F.silu(gate) * up).bfloat16()
    torch.testing.assert_close(c2[valid], zero_expected[valid], atol=0, rtol=0)
    assert torch.equal(c2[valid].view(torch.int16), zero_expected[valid].view(torch.int16))
    assert torch.all(c2[~valid] == -19)
    assert torch.equal(c1.view(torch.int16), original.view(torch.int16))


@pytest.mark.parametrize(
    "t,rank,bm,routing,shared",
    [
        (64, 8, 16, "mixed", False),
        (256, 96, 32, "skewed", False),
        (64, 32, 16, "masked", False),
        (131, 16, 32, "mixed", True),
        (256, 128, 64, "mixed", True),
    ],
)
def test_act_fast(t, rank, bm, routing, shared):
    run_act_case(t, rank, bm, routing, shared=shared)


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [64, 256, 1024, 4096])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("bm", [16, 32])
@pytest.mark.parametrize("routing,shared", [("mixed", False), ("skewed", False), ("mixed", True)])
@pytest.mark.parametrize("zdtype", [None, torch.bfloat16, torch.float32])
def test_act_sweep(t, rank, bm, routing, shared, zdtype):
    run_act_case(t, rank, bm, routing, shared=shared, zdtype=zdtype)


def _compile_silu_probe(capability):
    """Test the actual epilogue helper before BF16 hides FP32 errors; fake compile."""
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.runtime import make_fake_stream

    from ctikernels.moe.cute._lora_block import silu_fp32

    @cute.kernel
    def evaluate(source: cute.Tensor, target: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 128 + tid
        if index < source.shape[0]:
            target[index] = silu_fp32(source[index])

    @cute.jit
    def launch(source, target, stream):
        evaluate(source, target).launch(
            grid=(cute.ceil_div(source.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream
        )

    count = cute.sym_int()
    tensor = cute.runtime.make_fake_tensor(Float32, (count,), (1,), assumed_align=16)
    return cute.compile(
        launch,
        tensor,
        tensor,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )


def test_act_silu_fp32_accuracy():
    torch.manual_seed(65)
    bf16 = torch.arange(-32768, 32768, device="cuda").to(torch.int16).view(torch.bfloat16).float()
    source = torch.cat(
        (
            bf16,
            torch.linspace(-100, 100, 1_000_001, device="cuda"),
            torch.randn(1_000_000, device="cuda") * 20,
        )
    ).contiguous()
    expected = F.silu(source)
    actual = torch.empty_like(source)
    _compile_silu_probe(torch.cuda.get_device_capability())(source, actual)
    finite = torch.isfinite(expected)
    ulp = (torch.nextafter(expected, torch.full_like(expected, float("inf"))) - expected).abs()
    assert torch.all((actual[finite] - expected[finite]).abs() <= 2 * ulp[finite])
    torch.testing.assert_close(actual[~finite], expected[~finite], equal_nan=True)


@pytest.mark.parametrize("t", [64, 512, 2048])
@pytest.mark.parametrize("shared", [False, True])
def test_act_composition_fp32_oracle(t, shared):
    torch.manual_seed(87)
    e, k, n, rank, top_k = 4, 64, 32, 16, 2
    x = rand(t, k, scale=0.3)
    w = packed(
        rand(e, 2, rank, k),
        rand(e, 2, n, rank),
        rand(e, rank, n),
        rand(e, k, rank),
        as13=rand(2, rank, k),
        bs13=rand(2, n, rank),
        as2=rand(rank, n),
        bs2=rand(k, rank),
    )
    if shared:
        e, top_k = 1, 1
        a13, b13, a2, b2 = w.as13[None], w.bs13[None], w.as2[None, None], w.bs2[None, None]
    else:
        a13, b13, a2, b2 = w.a13, w.b13, w.a2[:, None], w.b2[:, None]
    w13, w2 = rand(e, 2 * n, k), rand(e, k, n)
    ids = torch.randint(e, (t, top_k), dtype=torch.int32, device="cuda")
    rw = torch.ones(t, top_k, device="cuda") if shared else torch.rand(t, top_k, device="cuda")
    if not shared:
        ids[::3, 0] = -1
        ids[-1] = e
    c1 = torch.zeros(t * top_k, 2 * n, dtype=torch.bfloat16, device="cuda")
    c3 = torch.zeros(t * top_k, k, dtype=torch.bfloat16, device="cuda")
    for expert in range(e):
        pairs = torch.where(ids.flatten() == expert)[0]
        c1[pairs] = (x[pairs // top_k].float() @ w13[expert].float().T).bfloat16()
    rounded, delta, _ = grouped_oracle(c1, x, a13, b13, ids, None, top_k, 0.7)
    # T64 uses the existing pair path (P128 routed/P64 shared); larger
    # batches use the fused epilogue and omit the intermediate cache rounding.
    gate, up = (c1.float() + delta if t * top_k > 128 else rounded.float()).chunk(2, -1)
    c2 = (F.silu(gate) * up).bfloat16()
    for expert in range(e):
        pairs = torch.where(ids.flatten() == expert)[0]
        base = c2[pairs].float() @ w2[expert].float().T
        c3[pairs] = (base * rw.flatten()[pairs, None]).bfloat16()
    c3, _, _ = grouped_oracle(c3, c2, a2, b2, ids, None if shared else rw, 1, 0.7)
    expected = c3.reshape(t, top_k, k).float().sum(1).bfloat16()
    if shared:
        actual = shared_expert_lora(x, w13[0], w2[0], w.shared, 0.7, provider="cute")
    else:
        actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7, provider="cute")
        assert torch.count_nonzero(actual[-1]) == 0
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.015


@pytest.mark.parametrize("experts", [256, 128])
@pytest.mark.parametrize("t", [16, 64, 512])
def test_fused_experts_ep_remote_pairs_real_expert_counts(experts, t):
    """EP: remote routes (-1 and out-of-range ids) contribute nothing, base included."""
    torch.manual_seed(5)
    h, i, top_k, rank = 2048, 512, 8, 16
    w = MoELoRAWeights.separate(
        rand(experts, rank, h),
        rand(experts, rank, h),
        rand(experts, i, rank),
        rand(experts, i, rank),
        rand(experts, rank, i),
        rand(experts, h, rank),
    )
    w13, w2 = rand(experts, 2 * i, h, scale=0.02), rand(experts, h, i, scale=0.02)
    x = rand(t, h, scale=0.3)
    ids = torch.randint(0, 2 * experts, (t, top_k), dtype=torch.int32, device="cuda")
    ids[ids >= experts] = -1
    ids[0, 0] = experts + 5
    rw = torch.rand(t, top_k, device="cuda")
    valid = (ids >= 0) & (ids < experts)
    expected = torch.zeros(t, h, device="cuda")
    for p in range(t * top_k):
        tok, e = p // top_k, int(ids.flatten()[p])
        if not 0 <= e < experts:
            continue
        xs = x[tok].float()
        gate = (
            xs @ w13[e, :i].float().T + 0.7 * (xs @ w.gate_a[e].float().T) @ w.gate_b[e].float().T
        )
        up = xs @ w13[e, i:].float().T + 0.7 * (xs @ w.up_a[e].float().T) @ w.up_b[e].float().T
        act = F.silu(gate) * up
        down = act @ w2[e].float().T + 0.7 * (act @ w.down_a[e].float().T) @ w.down_b[e].float().T
        expected[tok] += rw.flatten()[p] * down
    actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7, provider="cute")
    assert (actual.float() - expected).norm() / expected.norm() < 0.02
    # Tokens whose every route is remote produce exact zeros.
    ids[1] = -1
    actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7, provider="cute")
    assert torch.count_nonzero(actual[1]) == 0
    assert bool(valid.any())


@pytest.mark.parametrize("t", [64, 512, 2048])
@pytest.mark.parametrize("rank", [8, 16, 64, 128])
@pytest.mark.parametrize("experts,masked", [(8, True), (64, True), (64, False)])
def test_overlap_eager_and_graph_bitwise(t, rank, experts, masked):
    from ctikernels.moe.cute._lora_provider import uses_block_path

    torch.manual_seed(43)
    h, i = 128, 128
    # T64 also exercises the block path on small-SM devices at low ranks.
    top_k = 32 if t == 64 else 8
    assert uses_block_path(
        t * top_k, rank, torch.cuda.get_device_properties(0).multi_processor_count
    )
    weights = packed(
        rand(experts, 2, rank, h),
        rand(experts, 2, i, rank),
        rand(experts, rank, i),
        rand(experts, h, rank),
    )
    x, w13, w2 = rand(t, h, scale=0.3), rand(experts, 2 * i, h), rand(experts, h, i)
    ids = torch.randint(experts, (t, top_k), device="cuda", dtype=torch.int32)
    if masked:
        ids[::3, 0], ids[::5, 1], ids[-1] = -1, experts + 7, -1
    rw = torch.rand(t, top_k, device="cuda")
    scale = torch.tensor([0.7], device="cuda")

    def run(overlap):
        return fused_experts_lora(
            x,
            w13,
            w2,
            rw,
            ids,
            weights,
            scaling=scale,
            provider="cute",
            overlap=overlap,
            local_only=not masked,
        )

    expected = run(False)
    for overlap in (None, True):
        assert torch.equal(run(overlap).view(torch.int16), expected.view(torch.int16))
    if masked:
        assert torch.count_nonzero(expected[-1]) == 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(True)
    for factor in (0.3, -0.8):
        x.copy_(rand(t, h, scale=0.4))
        scale.fill_(factor)
        graph.replay()
        actual = captured.clone()
        eager = run(False)
        assert torch.equal(actual.view(torch.int16), eager.view(torch.int16))
        assert torch.equal(run(True).view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("rank", [8, 16, 64, 128])
@pytest.mark.parametrize("kind", ["act", "up", "down", "shared_up", "shared_down"])
@pytest.mark.parametrize("bm", [16, 32, 64])
def test_split_block_stages_bitwise(rank, kind, bm):
    from ctikernels.moe.cute._lora_provider import block_pipeline, compile_kernel
    from ctikernels.moe_impl import moe_align_block_size

    torch.manual_seed(47)
    shared, down, act = kind.startswith("shared"), kind.endswith("down"), kind == "act"
    t, top_k, e = 67, 1 if shared else 2, 1 if shared else 4
    k, n, slices = 72, 136, 1 if down else 2
    p = t * top_k
    a, b = rand(e, slices, rank, k), rand(e, slices, n, rank)
    x = rand(p if down else t, k, scale=0.3)
    ids, rw = None, None
    alignment = (None, None, None)
    valid = torch.ones(p, dtype=torch.bool, device="cuda")
    if not shared:
        ids = torch.randint(e, (t, top_k), dtype=torch.int64, device="cuda")
        ids[0, 0], ids[1, 1], ids[-1] = -1, 1 << 32, -1
        valid = ((ids >= 0) & (ids < e)).flatten()
        safe = torch.where(valid.view_as(ids), ids, 0).int()
        alignment = moe_align_block_size(safe, bm, e)
        rw = torch.rand(t, top_k, device="cuda") if down else None
    before = rand(p, slices * n, scale=0.002)
    full, split = before.clone(), before.clone()
    act_full = torch.full((p, n), -19, dtype=torch.bfloat16, device="cuda") if act else None
    act_split = act_full.clone() if act else None
    z_full = torch.full((p, slices, rank), -29, dtype=torch.bfloat16, device="cuda")
    z_split = z_full.clone()
    scale = torch.tensor([0.7], device="cuda")
    bn, stages = block_pipeline(bm, rank)

    def kernel(stage):
        return compile_kernel(
            k,
            n,
            rank,
            slices,
            top_k,
            e,
            torch.cuda.get_device_capability(),
            down=down,
            shared=shared,
            act=act,
            stage=stage,
            block_m=bm,
            block_n=bn,
            stages=stages,
            chunks=3,
            z_dtype=torch.bfloat16,
            id_dtype=torch.int64,
        )

    full_kernel, shrink_kernel, expand_kernel = (kernel(s) for s in ("full", "shrink", "expand"))
    full_kernel(
        x,
        *kernel_args(a, b),
        full,
        ids,
        rw,
        scale,
        z_full,
        *alignment,
        *((act_full,) if act else ()),
    )
    shrink_kernel(
        x,
        *kernel_args(a, b),
        split,
        ids,
        rw,
        scale,
        z_split,
        *alignment,
        *((act_split,) if act else ()),
    )
    assert torch.equal(split.view(torch.int16), before.view(torch.int16))
    if act:
        assert torch.all(act_split == -19)
    assert torch.equal(z_split.view(torch.int16), z_full.view(torch.int16))
    assert torch.all(z_split[~valid] == -29)
    # Expand must consume saved z only: changing X/A cannot affect its result.
    x.fill_(float("nan"))
    a.fill_(float("nan"))
    expand_kernel(
        x,
        *kernel_args(a, b),
        split,
        ids,
        rw,
        scale,
        z_split,
        *alignment,
        *((act_split,) if act else ()),
    )
    assert torch.equal(split.view(torch.int16), full.view(torch.int16))
    if act:
        assert torch.equal(act_split.view(torch.int16), act_full.view(torch.int16))
        assert torch.all(act_split[~valid] == -19)
    assert torch.equal(z_split.view(torch.int16), z_full.view(torch.int16))


def dense_oracle(before, x, a, b, scaling):
    """Independent FP64 dense math, including unrounded saved shrink values."""
    z = torch.stack([x.double() @ factor.double().T for factor in a], dim=1)
    delta = torch.cat([z[:, s] @ b[s].double().T for s in range(a.shape[0])], dim=-1) * scaling
    return (before.double() + delta).bfloat16(), delta.float(), z.float()


def run_dense_case(monkeypatch, t, rank, down, config, zdtype, *, k=None, n=None):
    from ctikernels.moe.cute import _lora_dense as _cute_dense
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    torch.manual_seed(101)
    k = k or (512 if down else 2048)
    n = n or (2048 if down else 512)
    slices = 1 if down else 2
    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", "dense")
    monkeypatch.setattr(_cute_dense, "DENSE_OVERRIDE", {} if config is None else {"config": config})
    a, b = rand(slices, rank, k), rand(slices, n, rank)
    x, before = rand(t, k, scale=0.3), rand(t, slices * n, scale=0.0001)
    expected, delta, saved = dense_oracle(before, x, a, b, 0.7)
    z = None if zdtype is None else torch.full_like(saved, -29, dtype=zdtype)
    out = before.clone()
    fn = shared_lora_w2_ if down else shared_lora_w13_
    factors = (a[0], b[0]) if down else (a, b)
    fn(out, x, factors, scaling=0.7, provider="cute", z_out=z)
    assert_delta(out, before, expected, delta, rounded_z=True)
    if z is not None:
        torch.testing.assert_close(
            z.float(),
            saved,
            atol=0.005 if zdtype == torch.bfloat16 else 2e-5,
            rtol=0.004 if zdtype == torch.bfloat16 else 1e-4,
        )
    # Atomic split reductions may change summation order on each launch.
    out2 = before.clone()
    fn(out2, x, factors, scaling=0.7, provider="cute", z_out=z)
    assert_delta(out2, before, expected, delta, rounded_z=True)
    if config is not None and config[1] == 1:
        assert torch.equal(out.view(torch.int16), out2.view(torch.int16))


@pytest.mark.parametrize(
    "t,rank,down,config,zdtype",
    [
        (1, 8, True, (16, 1, 16, 64), None),
        (15, 16, False, (32, 1, 32, 128, 3), torch.float32),
        (17, 32, True, (64, 16, 64, 128, 4), torch.bfloat16),
        (64, 64, False, (32, 4, 16, 128, 4), torch.float32),
        (512, 64, False, (32, 32, 32, 128, 3), torch.float32),
        (512, 128, True, (32, 8, 32, 64, 4), torch.bfloat16),
        (512, 128, False, (16, 32, 32, 128, 2), torch.float32),
        (100, 96, False, (32, 8, 32, 64), torch.bfloat16),
        (128, 128, False, (16, 2, 64, 128), None),
    ],
)
def test_dense_fast(monkeypatch, t, rank, down, config, zdtype):
    run_dense_case(monkeypatch, t, rank, down, config, zdtype)


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [1, 15, 17, 64, 100, 128, 513, 1024])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("zdtype", [None, torch.bfloat16, torch.float32])
def test_dense_sweep(monkeypatch, t, rank, down, zdtype):
    run_dense_case(monkeypatch, t, rank, down, None, zdtype)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("config", [(16, 1, 16, 64, 3), (32, 8, 32, 128, 4), (64, 16, 64, 128)])
def test_dense_vector_tails_and_empty_splits(monkeypatch, down, config):
    # Rank24 down exercises three n8 tiles with all four warps splitting K.
    # K72 has two BK tiles; split counts 8/16 include fully empty splits.
    run_dense_case(monkeypatch, 17, 24, down, config, torch.float32, k=72, n=520)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("zero", ["a", "b", "b_columns"])
@pytest.mark.parametrize("config", [(32, 1, 32, 128, 3), (32, 8, 32, 128, 4)])
def test_dense_zero_adapter_bits(monkeypatch, down, zero, config):
    from ctikernels.moe.cute import _lora_dense as _cute_dense
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", "dense")
    monkeypatch.setattr(_cute_dense, "DENSE_OVERRIDE", {"config": config})
    t, k, n, rank, slices = 17, 72, 520, 8, 1 if down else 2
    a, b = rand(slices, rank, k), rand(slices, n, rank)
    if zero == "a":
        a.zero_()
    elif zero == "b":
        b.zero_()
    else:
        b[:, ::2].zero_()
    x = rand(t, k)
    before = rand(t, slices * n)
    preserved = slice(None, None, 2) if zero == "b_columns" else slice(None)
    bits = before.view(torch.int16)
    # Include negative zero and payloads in the preserved half words.
    patterns = torch.tensor([-32768, 0, 32640, -128, 32705], dtype=torch.int16, device="cuda")
    width = bits[:, preserved].shape[1]
    bits[:, preserved] = patterns[torch.arange(width, device="cuda") % len(patterns)]
    out = before.clone()
    fn = shared_lora_w2_ if down else shared_lora_w13_
    fn(out, x, (a[0], b[0]) if down else (a, b), scaling=0.7, provider="cute")
    assert torch.equal(out.view(torch.int16)[:, preserved], bits[:, preserved])


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("zdtype", [torch.bfloat16, torch.float32])
def test_dense_live_scale_graph(monkeypatch, down, zdtype):
    from ctikernels.moe.cute import _lora_dense as _cute_dense
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", "dense")
    monkeypatch.setattr(_cute_dense, "DENSE_OVERRIDE", {"config": (32, 4, 32, 128, 4)})
    t, k, n, rank, slices = 17, 72, 520, 16, 1 if down else 2
    a, b = rand(slices, rank, k), rand(slices, n, rank)
    x, before = rand(t, k, scale=0.3), rand(t, slices * n, scale=0.0001)
    out = torch.empty_like(before)
    z = torch.empty((t, slices, rank), device="cuda", dtype=zdtype)
    scale = torch.tensor([0.7], device="cuda")
    fn = shared_lora_w2_ if down else shared_lora_w13_

    def run():
        # This predecessor produces the cache that expand prefetches before its wait.
        out.copy_(before)
        fn(out, x, (a[0], b[0]) if down else (a, b), scaling=scale, provider="cute", z_out=z)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for multiplier in (0.3, -0.8, 0.0):
        scale.fill_(multiplier)
        x.copy_(rand(t, k, scale=0.3))
        before.copy_(rand(t, slices * n, scale=0.0001))
        before[:, 0] = -0.0
        graph.replay()
        expected, delta, saved = dense_oracle(before, x, a, b, multiplier)
        assert_delta(out, before, expected, delta, rounded_z=True)
        torch.testing.assert_close(
            z.float(),
            saved,
            atol=0.005 if zdtype == torch.bfloat16 else 2e-5,
            rtol=0.004 if zdtype == torch.bfloat16 else 1e-4,
        )


def _routed_layer_reference(x, w13, w2, ids, rw, w, scale, chunk=16384):
    """FP32 routed MLP with LoRA, token-chunked so long bands fit beside the layer."""
    t, top_k, i = x.shape[0], ids.shape[1], w2.shape[2]
    out = torch.empty(t, x.shape[1], device="cuda")
    for start in range(0, t, chunk):
        xs, flat = x[start : start + chunk].float(), ids[start : start + chunk].flatten()
        acc = torch.zeros(xs.shape, device="cuda")
        weights = rw[start : start + chunk].flatten()
        for e in flat.unique().tolist():
            if not 0 <= e < w13.shape[0]:
                continue
            pairs = torch.where(flat == e)[0]
            xe = xs[pairs // top_k]
            c1 = xe @ w13[e].float().T
            for s in range(2):
                a_s, b_s = (w.gate_a, w.up_a)[s][e], (w.gate_b, w.up_b)[s][e]
                lora = (xe @ a_s.float().T) @ b_s.float().T
                c1[:, s * i : (s + 1) * i] += scale * lora
            act = F.silu(c1[:, :i]) * c1[:, i:]
            down = (
                act @ w2[e].float().T
                + scale * (act @ w.down_a[e].float().T) @ w.down_b[e].float().T
            )
            acc.index_add_(0, pairs // top_k, weights[pairs, None] * down)
        out[start : start + chunk] = acc
    return out


def _shared_layer_reference(x, w13, w2, lora, scale, chunk=16384):
    as13, bs13, as2, bs2 = (f.float() for f in lora)
    i = w2.shape[1]
    out = torch.empty(x.shape, device="cuda")
    for start in range(0, x.shape[0], chunk):
        xs = x[start : start + chunk].float()
        c1 = xs @ w13.float().T
        for s in range(2):
            c1[:, s * i : (s + 1) * i] += scale * (xs @ as13[s].T) @ bs13[s].T
        act = F.silu(c1[:, :i]) * c1[:, i:]
        out[start : start + chunk] = act @ w2.float().T + scale * (act @ as2.T) @ bs2.T
    return out


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [16384, 65536, 262144])
@pytest.mark.parametrize("rank", [16, 128])
@pytest.mark.parametrize("kind", ["routed", "shared"])
def test_long_context_bands(t, rank, kind):
    """One row per long band up to the 262144-token model maximum, in one call."""
    run_long_context_band(t, rank, kind)


def run_long_context_band(t, rank, kind, *, hidden=2048, intermediate=512, experts=256):
    h, i, e, top_k = hidden, intermediate, experts, 8
    # Include the larger gate/up and activation caches for I768, plus scratch
    # and FP32 reference temporaries. Check free memory before allocating.
    per_token = 66_000 + top_k * 6 * (i - 512) if kind == "routed" else 26_000
    need = t * per_token + (4 << 30)
    # Earlier tests leave blocks in the caching allocator; release them so the
    # free-memory check sees the device, not this process's cache.
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"T={t} needs ~{need >> 30} GiB free, device has {free >> 30} GiB")
    torch.manual_seed(t + rank)
    x = rand(t, h, scale=0.3)
    if kind == "shared":
        w13, w2 = rand(2 * i, h, scale=0.02), rand(h, i, scale=0.02)
        lora = (rand(2, rank, h), rand(2, i, rank), rand(rank, i), rand(h, rank))
        actual = shared_expert_lora(x, w13, w2, lora, 0.7)
        expected = _shared_layer_reference(x, w13, w2, lora, 0.7)
    else:
        w = MoELoRAWeights.separate(
            rand(e, rank, h),
            rand(e, rank, h),
            rand(e, i, rank),
            rand(e, i, rank),
            rand(e, rank, i),
            rand(e, h, rank),
        )
        w13, w2 = rand(e, 2 * i, h, scale=0.02), rand(e, h, i, scale=0.02)
        ids = torch.randint(0, e, (t, top_k), dtype=torch.int32, device="cuda")
        # EP: about one route in 16 is remote, plus an out-of-range id.
        ids[torch.rand(ids.shape, device="cuda") < 1 / 16] = -1
        ids[0, 0] = e + 1
        rw = torch.rand(t, top_k, device="cuda")
        actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7)
        expected = _routed_layer_reference(x, w13, w2, ids, rw, w, 0.7)
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm() < 0.02
    # The last rows are covered too (no 32-bit offset wrap at the band's end).
    tail = slice(t - 64, t)
    assert (actual[tail].float() - expected[tail]).norm() / expected[tail].norm() < 0.03


@pytest.mark.parametrize("t", [16384, 65536, 262144])
@pytest.mark.parametrize("rank", [16, 64])
def test_qwen3_30b_long_context_bands(t, rank):
    run_long_context_band(t, rank, "routed", intermediate=768, experts=128)


@pytest.mark.parametrize(
    "t,rank,ep",
    [(t, rank, 1) for t in (1, 8, 64, 512, 4096) for rank in (16, 32, 64)]
    + [(8, 32, 2), (512, 32, 2)],
)
def test_qwen3_30b_layer(t, rank, ep):
    torch.manual_seed(87)
    h, i, e, top_k = 2048, 768, 128 // ep, 8
    w = packed(rand(e, 2, rank, h), rand(e, 2, i, rank), rand(e, rank, i), rand(e, h, rank))
    assert w.shared is None
    w13, w2 = rand(e, 2 * i, h, scale=0.02), rand(e, h, i, scale=0.02)
    x = rand(t, h, scale=0.3)
    ids = torch.randint(0, 128, (t, top_k), device="cuda", dtype=torch.int32)
    if ep == 2:
        ids[ids >= e] = -1
        ids[0, 0] = e + 1
        ids[1] = -1
        ids[-1, 0] = e - 1
    rw = torch.rand(t, top_k, device="cuda")
    expected = _routed_layer_reference(x, w13, w2, ids, rw, w, 0.7)
    actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7, provider="cute")
    assert torch.isfinite(actual).all()
    tolerance = 0.02 if ep == 2 else 0.015
    assert (actual.float() - expected).norm() / expected.norm() < tolerance
    if ep == 2:
        assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("kind", ["shared13", "shared2", "routed13", "routed2", "layer"])
@pytest.mark.parametrize("provider", ["auto", "cute"])
def test_training_forward_trainable_factors(kind, provider):
    """Trainable A (T1024, r128): low-level calls and compositions keep gradients."""
    torch.manual_seed(31)
    t, rank, h, i, e, top_k = 1024, 128, 2048, 512, 256, 8
    shared, down = kind.startswith("shared"), kind.endswith("2")
    if kind == "layer":
        w = MoELoRAWeights(
            rand(e, rank, h), rand(2, e, i, rank), rand(e, rank, i), rand(e, h, rank)
        )
        a = w.gate_up_a.clone().requires_grad_(True)
        frozen_lora = w
        w = MoELoRAWeights(a, *w.leaves[1:])
        w13, w2 = rand(e, 2 * i, h, scale=0.02), rand(e, h, i, scale=0.02)
        x = rand(t, h, scale=0.3)
        ids = torch.randint(0, e, (t, top_k), dtype=torch.int32, device="cuda")
        ids[::5, 0] = -1
        rw = torch.rand(t, top_k, device="cuda")
        out = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.7, provider=provider)
        with torch.no_grad():
            frozen = fused_experts_lora(x, w13, w2, rw, ids, frozen_lora, scaling=0.7)
        assert out.requires_grad
        assert (out.float() - frozen.float()).norm() / frozen.float().norm() < 0.015
        out.float().square().sum().backward()
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
        return
    k, n, slices = (i, h, 1) if down else (h, i, 2)
    experts = 1 if shared else e
    a = rand(experts, slices, rank, k)
    b = rand(experts, slices, n, rank)
    p = t if shared else t * top_k
    ids = None if shared else torch.randint(0, e, (t, top_k), dtype=torch.int32, device="cuda")
    rw = torch.rand(t, top_k, device="cuda") if not shared and down else None
    x = rand(p if down else t, k, scale=0.3)
    before = rand(p, slices * n, scale=0.0001)
    leaf = (a[0, 0] if shared and down else a[0] if shared else a[:, 0] if down else a).clone()
    leaf.requires_grad_(True)
    factors = (leaf, (b[0, 0] if down else b[0]) if shared else (b[:, 0] if down else b))
    if not shared and not down:
        factors = leaves13(*factors)
    actual = before.clone()
    # On CUDA the insertion ops are inference-only (no silent torch path); training runs
    # through fused_experts_lora. provider="torch" keeps the differentiable reference.
    if shared:
        op = shared_lora_w2_ if down else shared_lora_w13_
        call = lambda prov: op(actual, x, factors, scaling=0.7, provider=prov)
    elif down:
        call = lambda prov: moe_lora_w2_(
            actual, x, factors, ids, rw, scaling=0.7, top_k=top_k, provider=prov
        )
    else:
        call = lambda prov: moe_lora_w13_(
            actual, x, factors, ids, scaling=0.7, top_k=top_k, provider=prov
        )
    with pytest.raises(ValueError, match="inference-only on CUDA"):
        call(provider)
    call("torch")
    assert actual.requires_grad
    # FP32 autograd reference on the same leaf.
    ref_a = leaf.reshape(a.shape)
    flat = torch.zeros(p, dtype=torch.long, device="cuda") if shared else ids.flatten().long()
    rows = (
        torch.arange(p, device="cuda")
        if down
        else torch.arange(p, device="cuda") // (1 if shared else top_k)
    )
    expected = torch.empty(p, slices * n, device="cuda")
    for expert in flat.unique().tolist():
        pairs = torch.where(flat == expert)[0]
        for s in range(slices):
            z = x[rows[pairs]].float() @ ref_a[expert, s].float().T
            d = 0.7 * (z @ b[expert, s].float().T)
            if rw is not None:
                d = d * rw.flatten()[pairs, None]
            expected[pairs, s * n : (s + 1) * n] = d
    expected = before.float() + expected
    grad = torch.randn(actual.shape, device="cuda")
    (actual.float() * grad).sum().backward()
    ours = leaf.grad.clone()
    leaf.grad = None
    (expected * grad).sum().backward()
    assert (actual.float() - expected.detach()).norm() / (expected - before.float()).norm() < 1e-2
    assert (ours - leaf.grad).norm() / leaf.grad.norm() < 1e-2


@pytest.mark.parametrize("init", ["zero_b", "zero_a", "both", "all_remote"])
@pytest.mark.parametrize("kind", ["shared13", "shared2", "routed13", "routed2"])
def test_training_gradients_peft_init(kind, init):
    """Trainable A and B (T1024, r128) vs FP32 autograd, including B = 0 (PEFT init)."""
    run_training_gradients_peft_init(kind, init)


@pytest.mark.parametrize("kind", ["routed13", "routed2"])
def test_qwen3_30b_training_gradients_peft_init(kind):
    run_training_gradients_peft_init(kind, "zero_b", t=8, rank=16, intermediate=768, experts=128)


def run_training_gradients_peft_init(
    kind, init, *, t=1024, rank=128, hidden=2048, intermediate=512, experts=256
):
    torch.manual_seed(37)
    h, i, e, top_k = hidden, intermediate, experts, 8
    shared, down = kind.startswith("shared"), kind.endswith("2")
    if shared and init == "all_remote":
        pytest.skip("shared rows have no routes")
    k, n, slices = (i, h, 1) if down else (h, i, 2)
    experts = 1 if shared else e
    a = rand(experts, slices, rank, k)
    b = rand(experts, slices, n, rank)
    if init == "zero_b":
        b.zero_()
    elif init == "zero_a":
        a.zero_()
    p = t if shared else t * top_k
    ids = None if shared else torch.randint(0, e, (t, top_k), dtype=torch.int32, device="cuda")
    if init == "all_remote":
        ids.fill_(-1)
        ids[::3, 1] = e
    rw = torch.rand(t, top_k, device="cuda") if not shared and down else None
    x = rand(p if down else t, k, scale=0.3)
    before = rand(p, slices * n, scale=0.0001)

    def view(v):
        return (v[0, 0] if down else v[0]) if shared else (v[:, 0] if down else v)

    leaves = [view(a).clone().requires_grad_(True), view(b).clone().requires_grad_(True)]
    actual = before.clone()
    if shared:
        op = shared_lora_w2_ if down else shared_lora_w13_
        op(actual, x, tuple(leaves), scaling=0.7, provider="torch")
    elif down:
        moe_lora_w2_(actual, x, tuple(leaves), ids, rw, scaling=0.7, top_k=top_k, provider="torch")
    else:
        moe_lora_w13_(actual, x, leaves13(*leaves), ids, scaling=0.7, top_k=top_k, provider="torch")
    assert actual.requires_grad
    grad = torch.randn(actual.shape, device="cuda")
    (actual.float() * grad).sum().backward()
    ours = [leaf.grad.clone() for leaf in leaves]
    if init == "all_remote":
        assert torch.equal(actual, before)
        for g in ours:
            assert torch.equal(g, torch.zeros_like(g))
        return
    for leaf in leaves:
        leaf.grad = None
    ref_a, ref_b = leaves[0].reshape(a.shape), leaves[1].reshape(b.shape)
    flat = torch.zeros(p, dtype=torch.long, device="cuda") if shared else ids.flatten().long()
    rows = torch.arange(p, device="cuda") // (1 if down or shared else top_k)
    pieces, index = [], []
    for expert in flat.unique().tolist():
        pairs = torch.where(flat == expert)[0]
        for s in range(slices):
            z = x[rows[pairs]].float() @ ref_a[expert, s].float().T
            d = 0.7 * (z @ ref_b[expert, s].float().T)
            if rw is not None:
                d = d * rw.flatten()[pairs, None]
            pieces.append(F.pad(d, (s * n, (slices - 1 - s) * n)))
            index.append(pairs)
    expected = before.float().index_add(0, torch.cat(index), torch.cat(pieces))
    (expected * grad).sum().backward()
    for mine, leaf, zero in zip(ours, leaves, ("zero_b", "zero_a"), strict=True):
        if init == zero:
            # The other factor is zero, so this gradient is exactly zero.
            assert torch.equal(mine, torch.zeros_like(mine))
        else:
            assert mine.abs().sum() > 0
            assert (mine - leaf.grad).norm() / leaf.grad.norm() < 1e-2


def test_training_base_only_all_remote_backprop_zero():
    """Base-only training forward (lora=None) with every route remote under EP."""
    torch.manual_seed(41)
    t, h, i, e, top_k = 1024, 2048, 512, 256, 8
    ids = torch.full((t, top_k), -1, dtype=torch.int32, device="cuda")
    ids[::3, 1] = e
    x = rand(t, h, scale=0.3).requires_grad_(True)
    rw = torch.rand(t, top_k, device="cuda").requires_grad_(True)
    w13, w2 = rand(e, 2 * i, h, scale=0.02), rand(e, h, i, scale=0.02)
    out = fused_experts_lora(x, w13, w2, rw, ids, None, scaling=0.5)
    assert out.requires_grad
    assert torch.equal(out, torch.zeros_like(out))
    (out.float() * torch.randn(out.shape, device="cuda")).sum().backward()
    for leaf in (x, rw):
        assert torch.equal(leaf.grad, torch.zeros_like(leaf.grad))


def test_resident_workspace_default_capture_stream_borrows_arena(monkeypatch):
    """torch.cuda.graph captures on a private stream: the capture must reuse the
    arena the warm-up grew instead of allocating its intermediates."""
    from ctikernels.moe import _lora_workspace as _workspace

    monkeypatch.setattr(_workspace, "ENABLED", True)
    args, kwargs = workspace_case(19, 32, 16, 256, 128, 1, True)
    warm = fused_experts_lora(*args, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_experts_lora(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
    assert peak <= 64 * 1024, f"capture scratch peak: {peak} bytes"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), warm.view(torch.int16))


def heavy_case(seed, t, global_e, h, i, rank, ep_rank):
    """Heavy-tailed rows, a massive input channel and outlier B columns (EP2 half)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    host = torch.Generator().manual_seed(seed)
    e = global_e // 2
    x = torch.randn(t, h, generator=g, device="cuda")
    x *= torch.exp(1.5 * torch.randn(t, 1, generator=g, device="cuda"))
    x[: max(1, t // 8), int(torch.randint(h, (1,), generator=host))] *= 64.0
    logits = torch.randn(t, global_e, generator=g, device="cuda")
    weights, ids = logits.softmax(-1).topk(8, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    w13 = torch.randn(e, 2 * i, h, generator=g, device="cuda") * h**-0.5
    w2 = torch.randn(e, h, i, generator=g, device="cuda") * i**-0.5
    a13 = (torch.rand(e, 2, rank, h, generator=g, device="cuda") * 2 - 1) * h**-0.5
    b13 = torch.randn(e, 2, i, rank, generator=g, device="cuda") * (0.5 * rank**-0.5)
    a2 = (torch.rand(e, rank, i, generator=g, device="cuda") * 2 - 1) * i**-0.5
    b2 = torch.randn(e, h, rank, generator=g, device="cuda") * (0.5 * rank**-0.5)
    b13[:, :, int(torch.randint(i, (1,), generator=host))] *= 32.0
    b2[:, int(torch.randint(h, (1,), generator=host))] *= 32.0
    expert_map = torch.full((global_e,), -1, dtype=torch.int32, device="cuda")
    expert_map[ep_rank * e : (ep_rank + 1) * e] = torch.arange(e, dtype=torch.int32, device="cuda")
    local = to_local_topk_ids(ids.to(torch.int32), expert_map)
    bf = torch.bfloat16
    return (x.to(bf), w13.to(bf), w2.to(bf), weights, local), tuple(
        v.to(bf) for v in (a13, b13, a2, b2)
    )


def chain(inputs, factors, scaling, *, market):
    """FP64 routed output; ``market`` rounds as vLLM 0.31's Triton MoE LoRA path does
    (BF16 caches, B scaled at load, BF16 deltas added to BF16 caches)."""
    x, w13, w2, weights, local = inputs
    a13, b13, a2, b2 = factors
    f64 = torch.float64
    r = (lambda v: v.to(torch.bfloat16).to(f64)) if market else (lambda v: v)
    i = w2.shape[2]
    out = torch.zeros(x.shape[0], local.shape[1], x.shape[1], dtype=f64, device="cuda")
    for expert in torch.unique(local[local >= 0]).tolist():
        token, choice = (local == expert).nonzero(as_tuple=True)
        part = x[token].to(f64)
        b_up = b13[expert].to(f64) * scaling
        b_down = b2[expert].to(f64) * scaling
        if market:
            b_up, b_down = r(b_up), r(b_down)
        c1 = r(part @ w13[expert].to(f64).t())
        z = [r(part @ a13[expert, s].to(f64).t()) for s in range(2)]
        gate = r(c1[:, :i] + r(z[0] @ b_up[0].t()))
        up = r(c1[:, i:] + r(z[1] @ b_up[1].t()))
        act = r(F.silu(gate) * up)
        weight = weights[token, choice].to(f64)[:, None]
        base = r(weight * (act @ w2[expert].to(f64).t()))
        delta = r(weight * (r(act @ a2[expert].to(f64).t()) @ b_down.t()))
        out[token, choice] = r(base + delta)
    total = out.sum(1)
    return r(total) if market else total


def test_heavy_inputs_at_least_as_accurate_as_market_rounding_chain():
    """Precision finding (EP2 rank 128, heavy seed): on heavy-tailed inputs the
    RMS error is set by a few outputs near 1e5, one BF16 rounding each. Pooled over seeds,
    ours must stay at or below vLLM's rounding chain (emulated here; the emulation matched
    the vLLM 0.31.0 wheel within 0.6% RMS on 24 seeds of the reported shape)."""
    ours_sq = market_sq = 0.0
    for seed in range(6):
        inputs, factors = heavy_case(seed, 256, 32, 1024, 256, 128, 1)
        x, w13, w2, weights, local = inputs
        exact = chain(inputs, factors, 0.73, market=False)
        market = chain(inputs, factors, 0.73, market=True)
        lora = packed(*factors)
        ours = fused_experts_lora(
            x, w13, w2, weights, local, lora, scaling=0.73, num_global_experts=32
        ).to(torch.float64)
        ours_sq += float((ours - exact).square().sum())
        market_sq += float((market - exact).square().sum())
    assert ours_sq <= market_sq, (ours_sq / market_sq) ** 0.5
