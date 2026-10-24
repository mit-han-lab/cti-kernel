# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Ordered reduction and staged LoRA numerical/graph qualification on SM12x."""

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE, _dsl_available

from . import test_moe_block_fp8_e2e_accuracy as e2e

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or not _dsl_available(),
    reason="requires SM120/121 and CuTe DSL 4.8.0",
)


def bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def ordered_reference(down, ids, weights, mapping=None, delta=None):
    """vLLM's combine: the unweighted row plus the LoRA delta, rounded once per route, then
    FlashInfer's FMA chain over live routes."""
    from ..modules._moe_reference import combine_reference

    live = ids >= 0
    if mapping is not None:
        live &= mapping[ids.clamp_min(0).long()] >= 0
    rows = down.view(*ids.shape, -1)
    if delta is not None:
        rows = (rows.float() + delta.view_as(rows).float()).bfloat16()
    return combine_reference(rows, weights, live)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("h,top_k", [(128, 1), (2048, 8), (2176, 8)])
@pytest.mark.parametrize("has_map", [False, True])
@torch.inference_mode()
def test_ordered_reduce(m, h, top_k, has_map):
    from ctikernels._moe_block_fp8_reduce import masked_reduce

    torch.manual_seed(412)
    ids = torch.randint(4, (m, top_k), dtype=torch.int32, device="cuda")
    ids.flatten()[::5] = -1
    mapping = torch.tensor([1, -1, 0, -1], device="cuda", dtype=torch.int32) if has_map else None
    live = ids >= 0
    if mapping is not None:
        live &= mapping[ids.clamp_min(0).long()] >= 0
    down = torch.randn(m * top_k, h, device="cuda").bfloat16()
    # Cancellation with a wide exponent range distinguishes slot order from
    # a parallel tree and BF16 accumulation from FP32 accumulation.
    down *= torch.exp2(torch.randint(-12, 12, (m * top_k, 1), device="cuda").float())
    down[~live.flatten()] = float("nan")
    delta = torch.randn_like(down)
    delta[~live.flatten()] = float("nan")
    weights = torch.rand(m, top_k, device="cuda")
    out = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
    for extra in (None, delta):
        masked_reduce(down, ids, out, weights, expert_map=mapping, delta=extra)
        expected = ordered_reference(down, ids, weights, mapping, extra)
        bitwise(out, expected)
        masked_reduce(down, ids, out, weights, expert_map=mapping, delta=extra)
        bitwise(out, expected)


@pytest.mark.parametrize("i", [512, 768])
@torch.inference_mode()
def test_activation_export_is_quantized_value(i):
    from ctikernels._moe_block_fp8_cute_act import silu_mul_quant_fp8
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8

    gu = torch.randn(32, 2 * i, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(32, i, device="cuda", dtype=torch.float8_e4m3fn)
    s = torch.empty(32, i // 128, device="cuda")
    act = torch.empty(32, i, device="cuda", dtype=torch.bfloat16)
    silu_mul_quant_fp8(gu, q, s)
    old_q, old_s = q.clone(), s.clone()
    silu_mul_quant_fp8(gu, q, s, act_out=act)
    assert torch.equal(q.view(torch.uint8), old_q.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), old_s.view(torch.int32))
    expected_q, expected_s = quantize_per_token_group_fp8(act)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), expected_s.view(torch.int32))


@pytest.fixture(scope="module", autouse=True)
def exact_fp32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(
    scope="module", params=[(256, 2048, 512), (128, 2048, 768)], ids=["qwen3.5", "qwen3"]
)
def params(request):
    e, h, i = request.param
    torch.manual_seed(671)
    tensors = []
    for n, k in ((2 * i, h), (h, i)):
        w = torch.empty(e, n, k, device="cuda", dtype=torch.float8_e4m3fn)
        for expert in range(e):
            w[expert].copy_((torch.randn(n, k, device="cuda") * 16).to(w.dtype))
        s = (torch.exp2(torch.rand(e, n // 128, k // 128, device="cuda") - 10)).bfloat16()
        tensors.extend((w, s))
    return tuple(tensors)


def low_rank(value, local, a, b):
    shrink = torch.bmm(a[local], value.float().unsqueeze(-1))
    return torch.bmm(b[local], shrink).squeeze(-1).bfloat16()


def oracle(x, weights, ids, params, mapping, factors=None):
    """Unquantized routed MoE and LoRA in FP64 on the exact input operands."""
    w13, s13, w2, s2 = params
    local = ids.long() if mapping is None else mapping[ids.clamp_min(0).long()].long()
    local = torch.where(ids >= 0, local, -1).flatten()
    routes = torch.zeros(ids.numel(), x.shape[1], device=x.device, dtype=torch.float64)
    for e in local.unique().tolist():
        if e < 0:
            continue
        r = (local == e).nonzero().flatten()
        source = x[r // ids.shape[1]].double()
        gu = source @ e2e.dequantize(w13[e], s13[e]).T
        if factors is not None:
            ga, gb, da, db = factors
            gu += (source @ ga[e].double().T) @ gb[e].double().T
        g, u = gu.chunk(2, -1)
        act = torch.nn.functional.silu(g) * u
        value = act @ e2e.dequantize(w2[e], s2[e]).T
        if factors is not None:
            value += (act @ da[e].double().T) @ db[e].double().T
        routes[r] = value * weights.flatten()[r, None].double()
    total = torch.zeros_like(x, dtype=torch.float64)
    for slot in routes.view(*ids.shape, -1).unbind(1):
        total += slot
    return total


def lora_hooks(x, ids, mapping, factors):
    ga, gb, da, db = factors
    local = ids if mapping is None else mapping[ids.clamp_min(0).long()]
    local = local.flatten().clamp_min(0).long()

    def gate_hook(gu):
        # Deliberately also write inactive rows: the operator must discard them.
        gu.add_(low_rank(x.repeat_interleave(ids.shape[1], 0), local, ga, gb))

    def down_hook(act):
        return low_rank(act, local, da, db)

    return gate_hook, down_hook


def original_lora_chain(x, weights, ids, params, mapping, factors):
    """Insert torch deltas into the unchanged #28 helper's BF16 caches.

    ids already masks remote routes to -1. Scoped replacements affect only the
    e2e helper, preserving its alignment, quantizers, default config and sum.
    """
    gate_hook, down_hook = lora_hooks(x, ids, mapping, factors)
    split, reduce = e2e.silu_and_mul, e2e.moe_sum_reduce_triton
    activation = None

    def activate(gu, act):
        nonlocal activation
        gate_hook(gu)
        split(gu, act)
        activation = act

    def sum_routes(down, out, **combine):
        # vLLM adds the unweighted delta to the unweighted BF16 row; the combine weights both.
        delta = down_hook(activation).view_as(down)
        down.copy_(torch.where(ids[..., None] >= 0, down.float() + delta.float(), 0))
        reduce(down, out, **combine)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(e2e, "silu_and_mul", activate)
        patch.setattr(e2e, "moe_sum_reduce_triton", sum_routes)
        return e2e.original_chain(x, weights, ids, params)


def triton_lora_noise(x, weights, ids, params, mapping, factors, base_errors, reference):
    samples = [base_errors]
    g = torch.Generator(device=x.device).manual_seed(1234)
    for _ in range(e2e.PERMUTATIONS):
        perm = torch.randperm(ids.shape[1], device=x.device, generator=g)
        out = original_lora_chain(
            x, weights[:, perm].contiguous(), ids[:, perm].contiguous(), params, mapping, factors
        )
        samples.append(e2e.errors(out, reference))
    t = torch.tensor(samples, dtype=torch.float64)
    mean = t.mean(0)
    return torch.where(mean > 0, t.std(0) / mean.clamp_min(1e-300), 0.0).tolist()


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("fused_act", [False, True])
@torch.inference_mode()
def test_staged_lora(params, m, ep, fused_act, monkeypatch):
    monkeypatch.setenv("CTI_MOE_FP8_CUTE", "1" if fused_act else "0")
    e, _, h = params[0].shape
    i, top_k = params[2].shape[-1], 8
    torch.manual_seed(723 + m)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(e, (m, top_k), device="cuda", dtype=torch.int32)
    ids[:, 0], ids[:, 1] = 0, e - 1  # Both EP ranks have live work even at M=1.
    ids[:, 2] = -1
    weights = torch.rand(m, top_k, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    for rank in range(ep):
        start, end = rank * e // ep, (rank + 1) * e // ep
        mapping = None
        if ep == 2:
            mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
            mapping[start:end] = torch.arange(end - start, device="cuda", dtype=torch.int32)
        local_params = tuple(t[start:end] for t in params)
        op = BlockFP8MoE(*local_params, top_k=top_k, provider="triton-fp8", expert_map=mapping)
        plain = op(x, weights, ids)
        bitwise(op.forward_staged(x, weights, ids), plain)
        zero = torch.zeros(m * top_k, h, device="cuda", dtype=torch.bfloat16)
        actual, act = op.forward_staged(
            x,
            weights,
            ids,
            gate_up_hook=lambda gu: gu.add_(0),
            down_hook=lambda act, zero=zero: zero,
            return_act=True,
        )
        bitwise(actual, plain)
        assert act.shape == (m * top_k, i)
        bitwise(op.forward_staged(x, weights, ids, down_hook=lambda act: None), plain)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("skewed", [False, True])
@pytest.mark.parametrize("fused_act", [False, True])
@torch.inference_mode()
def test_staged_lora_no_worse_than_original_triton(params, m, ep, skewed, fused_act, monkeypatch):
    monkeypatch.setenv("CTI_MOE_FP8_CUTE", "1" if fused_act else "0")
    e, _, h = params[0].shape
    i, top_k = params[2].shape[-1], 8
    # The accepted e2e helpers use module constants for their Qwen3.5 shape.
    for name, value in (("E", e), ("H", h), ("INTERMEDIATE", i), ("TOP_K", top_k)):
        monkeypatch.setattr(e2e, name, value)
    rows, spreads = [], []
    for seed in e2e.SEEDS:
        x, weights, ids = e2e.case(m, skewed, seed)
        if skewed and seed == e2e.SEEDS[0]:
            # Guarantee an EP2 rank with no local routes at every batch size.
            ids = torch.where(ids >= 0, ids % 4, -1)
        g = torch.Generator(device="cuda").manual_seed(723 + seed)
        factors = (
            torch.randn(e, 4, h, device="cuda", generator=g) / h**0.5,
            torch.randn(e, 2 * i, 4, device="cuda", generator=g) * 0.3,
            torch.randn(e, 4, i, device="cuda", generator=g) / i**0.5,
            torch.randn(e, h, 4, device="cuda", generator=g) * 0.3,
        )
        for rank in range(ep):
            start, end = rank * e // ep, (rank + 1) * e // ep
            mapping = None
            if ep > 1:
                mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
                mapping[start:end] = torch.arange(end - start, device="cuda", dtype=torch.int32)
            local_params = tuple(t[start:end] for t in params)
            local_factors = tuple(t[start:end] for t in factors)
            op = BlockFP8MoE(
                *local_params,
                top_k=top_k,
                provider="triton-fp8",
                expert_map=mapping,
                num_global_experts=e,
            )
            gate_hook, down_hook = lora_hooks(x, ids, mapping, local_factors)
            ours = op.forward_staged(x, weights, ids, gate_up_hook=gate_hook, down_hook=down_hook)
            plain = op(x, weights, ids)
            on_rank = (ids >= start) & (ids < end)
            rank_ids = torch.where(on_rank, ids, -1)
            base = original_lora_chain(x, weights, rank_ids, params, mapping, local_factors)
            reference = oracle(x, weights, ids, local_params, mapping, local_factors)
            a, b = e2e.errors(ours, reference), e2e.errors(base, reference)
            omitted = e2e.errors(plain, reference)
            live = on_rank.any().item()
            if not live:
                assert a == b == omitted == (0.0, 0.0)
                bitwise(ours, torch.zeros_like(ours))
                bitwise(base, torch.zeros_like(base))
            rows.append((seed, rank, a, b, omitted, live))
            spreads.append(
                triton_lora_noise(
                    x, weights, rank_ids, params, mapping, local_factors, b, reference
                )
            )
    # Same measured gate as e2e: three sigma, pooled over seeds and EP ranks.
    s = torch.tensor(spreads, dtype=torch.float64)
    tau = (3 * s.pow(2).mean(0).sqrt()).tolist()
    for seed, rank, a, b, omitted, live in rows:
        print(
            f"LoRA ep={ep} m={m} i={i} skewed={skewed} fused_act={fused_act} "
            f"seed={seed} rank={rank} ours max={a[0]:.4e} rms={a[1]:.4e} "
            f"triton max={b[0]:.4e} rms={b[1]:.4e} "
            f"tau max={tau[0]:.3e} rms={tau[1]:.3e}"
        )
        for metric, actual, baseline, missing, noise in zip(
            ("max", "rms"), a, b, omitted, tau, strict=True
        ):
            bound = baseline * (1 + noise)
            assert actual <= bound, (metric, seed, rank, a, b, tau)
            if live:
                assert missing > bound, ("omitted delta", metric, seed, rank, omitted, b, tau)
    ratios = torch.tensor([(*a, *b) for _, _, a, b, _, _ in rows], dtype=torch.float64)
    ratios = torch.where(ratios[:, 2:] > 0, ratios[:, :2] / ratios[:, 2:].clamp_min(1e-300), 1.0)
    mean_max, mean_rms = ratios.mean(0).tolist()
    assert mean_max <= 1.0 and mean_rms <= 1.0, (mean_max, mean_rms)


@torch.inference_mode()
def test_staged_graph_and_current_stream(params):
    e, _, h = params[0].shape
    i, m, top_k = params[2].shape[-1], 4, 8
    mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    mapping[: e // 2] = torch.arange(e // 2, device="cuda", dtype=torch.int32)
    op = BlockFP8MoE(*(t[: e // 2] for t in params), provider="triton-fp8", expert_map=mapping)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(m * top_k, device="cuda", dtype=torch.int32).view(m, top_k) % e
        weights = torch.full((m, top_k), 1 / top_k, device="cuda")
        gate_delta = torch.randn(m * top_k, 2 * i, device="cuda", dtype=torch.bfloat16) * 0.1
        shrink = torch.randn(i, 4, device="cuda", dtype=torch.bfloat16) * 0.05
        expand = torch.randn(4, h, device="cuda", dtype=torch.bfloat16) * 0.05

        def gate_hook(gu):
            gu.add_(gate_delta)

        def down_hook(act):
            return (act @ shrink) @ expand

        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(RuntimeError, match="staged=True"):
                op.forward_staged(x, weights, ids, gate_up_hook=gate_hook, down_hook=down_hook)
        op.prepare([m], staged=True)
        # Warm the application's two matmuls as well as the provider chain.
        expected, expected_act = op.forward_staged(
            x, weights, ids, gate_up_hook=gate_hook, down_hook=down_hook, return_act=True
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, captured_act = op.forward_staged(
                x, weights, ids, gate_up_hook=gate_hook, down_hook=down_hook, return_act=True
            )
        graph.replay()
        replay, replay_act = captured.clone(), captured_act.clone()
        # Only remote and padding routes; even nonzero hook outputs contribute zero.
        ids.fill_(e - 1)
        ids[:, ::2] = -1
        graph.replay()
        masked = captured.clone()
        ids.fill_(1)
        x.mul_(0.5)
        graph.replay()
        changed = captured.clone()
        eager = op.forward_staged(x, weights, ids, gate_up_hook=gate_hook, down_hook=down_hook)
    torch.cuda.current_stream().wait_stream(stream)
    bitwise(replay, expected)
    bitwise(replay_act, expected_act)
    assert torch.count_nonzero(masked).item() == 0
    bitwise(changed, eager)
