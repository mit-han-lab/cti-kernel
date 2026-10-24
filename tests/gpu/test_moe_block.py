# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""MoE numerical/EP/graph checks. Reuse weights to keep the warm-cache suite short."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from ctikernels import GraphWorkspace  # noqa: E402
from ctikernels.moe import Experts, MoEBlock, fused_experts  # noqa: E402
from ctikernels.moe._block import admits_shape, hardware  # noqa: E402
from ctikernels.moe._route import ROUTER_SPLIT  # noqa: E402
from ctikernels.moe._routing_check import classify_topk  # noqa: E402

from ..modules import _moe_reference as ref  # noqa: E402


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(23)
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    shapes = (
        (256, 2048),
        (256, 1024, 2048),
        (256, 2048, 512),
        (1024, 2048),
        (2048, 512),
        (1, 2048),
    )
    return tuple(torch.randn(shape, **opts).mul_(0.02) for shape in shapes)


def _admit(provider, rows, local=256, si=512):
    if provider == "triton_block" and not admits_shape(
        hardware(torch.device("cuda", torch.cuda.current_device())),
        rows=rows,
        hidden=2048,
        experts=256,
        inter=512,
        local_experts=local,
        shared_inter=si,
        dtype=torch.bfloat16,
    ):
        pytest.skip("whole-block provider does not admit this device/shape")


def _reference(x, weights, ids, mapping=None, valid=None):
    """FP32 oracle: an independent torch expert loop on the BF16 inputs.

    It is conditioned on the provider's selected IDs, which ``_check_routing``
    first verifies against an independent top-k (only genuine boundary ties may
    differ). The routing weights follow the model: BF16 logits, FP32 softmax at
    the selected IDs, renormalized; everything after that is FP32.
    """
    router, up, down, shared_up, shared_down, gate_w = weights
    count = x.shape[0] if valid is None else max(0, min(valid, x.shape[0]))
    ids = ids[:count].long()
    xf = x[:count].float()
    logits = F.linear(xf, router.float()).bfloat16().float()
    probs = logits.softmax(-1).gather(1, ids)
    probs /= probs.sum(-1, keepdim=True)
    result = torch.zeros_like(x, dtype=torch.float32)
    route_cpu = ids.cpu()
    mapping_cpu = torch.arange(256) if mapping is None else mapping.cpu()
    for expert in route_cpu.unique().tolist():
        local = int(mapping_cpu[expert])
        if local < 0:
            continue
        rows, slots = torch.where(route_cpu == expert)
        rows, slots = rows.to(x.device), slots.to(x.device)
        gate, value = F.linear(xf[rows], up[local].float()).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * value, down[local].float())
        result.index_add_(0, rows, y * probs[rows, slots, None])
    gate, value = F.linear(xf, shared_up.float()).chunk(2, dim=-1)
    shared = F.linear(F.silu(gate) * value, shared_down.float())
    result[:count] += F.linear(xf, gate_w.float()).sigmoid() * shared
    return result


def _hf_bf16(x, weights, ids, mapping=None, valid=None):
    """HF's BF16 eager Qwen3.5-MoE block, with its rounding points (transformers 5.20,
    models/qwen3_5_moe/modeling_qwen3_5_moe.py; qwen3_next is the same):

    - ``Qwen3_5MoeTopKRouter``: BF16 ``F.linear`` logits, ``softmax(dtype=float)``,
      FP32 top-k renormalization, then ``.to(router_logits.dtype)`` (BF16 weights);
    - ``Qwen3_5MoeExperts`` (eager): per hit expert in ascending global index, BF16
      gate_up linear, ``act_fn(gate) * up`` in BF16, BF16 down linear, ``*`` the BF16
      routing weight, and ``index_add_`` into a BF16 zero tensor;
    - ``Qwen3_5MoeMLP`` shared expert in BF16 (gate/up/down, act and product), and
      ``F.sigmoid(shared_expert_gate(x)) * shared`` in BF16, then the BF16 sum.

    It is conditioned on the provider's verified IDs, like ``_reference``. For an EP
    rank the same formula runs over the local experts and the shared I-slice. The dense
    GEMMs (router, shared expert, gate) see the same [T, H] batch as the provider: cuBLAS's
    kernel, and with PyTorch's default BF16 reduced-precision split-K reduction its
    rounding, depend on T, so HF on this batch has that rounding too.
    """
    router, up, down, shared_up, shared_down, gate_w = weights
    count = x.shape[0] if valid is None else max(0, min(valid, x.shape[0]))
    ids = ids[:count].long()
    xb = x[:count]
    logits = F.linear(x, router)[:count]
    top = F.softmax(logits, dtype=torch.float, dim=-1).gather(1, ids)
    top /= top.sum(dim=-1, keepdim=True)
    top = top.to(logits.dtype)
    routed = torch.zeros_like(xb)
    route_cpu = ids.cpu()
    mapping_cpu = torch.arange(256) if mapping is None else mapping.cpu()
    for expert in route_cpu.unique().tolist():
        local = int(mapping_cpu[expert])
        if local < 0:
            continue
        rows, slots = torch.where(route_cpu == expert)
        rows, slots = rows.to(x.device), slots.to(x.device)
        gate, value = F.linear(xb[rows], up[local]).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * value, down[local])
        routed.index_add_(0, rows, y * top[rows, slots, None])
    gate, value = F.linear(x, shared_up).chunk(2, dim=-1)
    shared = (F.sigmoid(F.linear(x, gate_w)) * F.linear(F.silu(gate) * value, shared_down))[:count]
    result = torch.zeros_like(x)
    result[:count] = routed + shared
    return result


ROUTING = {"ties": 0, "mismatches": 0}


def _check_routing(x, router, ids, valid=None):
    """Independent top-k check; only genuine BF16-logit boundary ties may differ."""
    ties, mismatches = classify_topk(x, router, ids, valid)
    ROUTING["ties"] += ties
    ROUTING["mismatches"] += mismatches
    assert mismatches == 0, (ties, mismatches)


def _vllm_bf16(x, weights, ids, mapping=None, valid=None):
    """vLLM 0.31's block (``ref.vllm_infer_reference``) on the provider's verified IDs, with
    FP32 renormalized softmax weights over the BF16 logits."""
    router, up, down, shared_up, shared_down, gate_w = weights
    count = x.shape[0] if valid is None else max(0, min(valid, x.shape[0]))
    ids = ids[:count].long()
    top = F.linear(x, router)[:count].float().softmax(-1).gather(1, ids)
    top /= top.sum(-1, keepdim=True)
    shared = (shared_up, shared_down, gate_w)
    return ref.vllm_infer_reference(
        x[:count], up, down, top, ids, shared=shared, expert_map=mapping
    )


def _block_errors(actual, x, weights, ids, mapping=None, valid=None):
    """Routing first, then |ours - FP32| and the market floor (the larger of |HF BF16 - FP32|
    and |vLLM - FP32|) per valid element."""
    _check_routing(x, weights[0], ids, valid)
    assert torch.isfinite(actual).all()
    count = x.shape[0] if valid is None else max(0, min(valid, x.shape[0]))
    oracle = _reference(x, weights, ids, mapping, valid)[:count]
    hf = (_hf_bf16(x, weights, ids, mapping, valid)[:count].float() - oracle).abs()
    vllm = (_vllm_bf16(x, weights, ids, mapping, valid).float() - oracle).abs()
    return (actual[:count].float() - oracle).abs().flatten(), torch.maximum(hf, vllm).flatten()


def _assert_floor(errors):
    """Ours vs FP32 against the market floor over the given samples: no worse in RMS; the max,
    one element where stacks a rounding apart land either side, within 25% of the floor's."""
    error = torch.cat([e for e, _ in errors])
    floor_error = torch.cat([f for _, f in errors])
    if error.numel() == 0:
        return
    assert error.max() <= 1.25 * floor_error.max(), ("max", error.max(), floor_error.max())
    rms, floor_rms = error.square().mean().sqrt(), floor_error.square().mean().sqrt()
    assert rms <= floor_rms, ("rms", rms, floor_rms)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 16, 64, 128, 256, 512])
@pytest.mark.parametrize("provider", ["auto", "composed", "triton_block"])
def test_whole_block_reference(weights, rows, provider):
    _admit(provider, rows)
    # The floor gate pools 4 seeds: a single small sample sits at the BF16 output
    # rounding for both ours and HF, where one draw decides the max either way.
    errors = []
    for seed in range(4):
        torch.manual_seed(1000 + seed)
        x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
        op = MoEBlock(provider)
        actual = op(x, *weights)
        errors.append(_block_errors(actual, x, weights, op.routing_ids))
    _assert_floor(errors)
    assert op.provider in ("triton_block", "composed")
    assert op.kernels_per_call in (None, 1, 2)
    expected_provider = MoEBlock().decision(rows, x.device) if provider == "auto" else provider
    assert op.provider == expected_provider


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 16, 64, 256])
def test_triton_block_routes_as_vllm(weights, rows, ties):
    """The whole-block router is vLLM's topk_softmax on its own BF16 logits (the router
    partials summed in split order, rounded once), bit for bit; with ``ties`` every odd
    expert duplicates its even neighbour, so each pick ties and the lower id must win."""
    _admit("triton_block", rows)
    router = weights[0].clone()
    if ties:
        router[1::2] = router[0::2]
    torch.manual_seed(5 + rows)
    x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
    op = MoEBlock("triton_block")
    op(x, router, *weights[1:])
    (ws,) = op._workspaces.values()
    n = ROUTER_SPLIT * rows * (256 + 16)
    if "buffer" in ws:
        start = ws["layout"]["off"]["plog"]
        plog = ws["buffer"][start : start + n].view(torch.float32)
    else:
        plog = ws["plog"].view(torch.float32)[:n]
    parts = plog.view(ROUTER_SPLIT, rows, 256 + 16)[:, :, :256]
    logits = parts[0].clone()
    for part in parts[1:]:
        logits += part
    logits = logits.bfloat16()
    want_w, want_ids, _ = ref.vllm_topk_softmax(logits, 8)
    assert torch.equal(op.routing_ids.int(), want_ids)
    assert torch.equal(op.routing_weights, want_w)
    want = ref._gemm(x, router).double()
    assert float((logits.double() - want).norm() / want.norm()) <= 4e-3  # the router GEMM


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8, 16, 64, 128, 256])
def test_triton_block_is_vllm_finalize_and_shared_join(weights, rows):
    """The whole-block kernel's output is vLLM's from its own unweighted BF16 slot rows and
    FP32 routing weights: the finalize's FMA chain in slot order, one rounding, then the
    BF16 gated shared row added in BF16 (bitwise); its rows and gate track the torch rows."""
    _admit("triton_block", rows)
    torch.manual_seed(77 + rows)
    x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
    op = MoEBlock("triton_block")
    out = op(x, *weights)
    (ws,) = op._workspaces.values()

    def view(name, n, dtype):  # one offset buffer (sm_120) or named parts (GB10)
        if "buffer" in ws:
            start = ws["layout"]["off"][name]
            return ws["buffer"][start : start + n].view(dtype)
        return ws[name].view(dtype)[:n]

    y = view("y", (rows * 8 + rows) * 2048, torch.float32).view(rows * 9, 2048)
    slots = y[: rows * 8].view(rows, 8, 2048).bfloat16()
    shared = y[rows * 8 :].bfloat16()
    valid = view("ploc", rows * 8, torch.int32).view(rows, 8) >= 0
    w = view("pw", rows * 8, torch.float32).view(rows, 8)
    assert torch.equal(slots.float(), y[: rows * 8].view(rows, 8, 2048))  # BF16 values
    want = ref.combine_reference(slots, w, valid) + shared  # BF16 add
    assert torch.equal(out, want)
    # the rows and the gate against torch (GEMM accumulation order may differ in the last bit)
    rows_ref, valid_ref = ref.vllm_rows(x, weights[1], weights[2], op.routing_ids)
    assert torch.equal(valid, valid_ref)
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
    assert rel(slots[valid], rows_ref[valid]) <= 2e-3
    if "buffer" in ws:  # GB10's last CTA resets gamma to its sentinel for the next call
        gam = view("gam", rows, torch.float32)
        assert torch.equal(gam.bfloat16().float(), gam)
        assert rel(gam, F.sigmoid(F.linear(x, weights[5]))[:, 0]) <= 4e-3
    full = ref.vllm_infer_reference(x, weights[1], weights[2], w, op.routing_ids,
                                    shared=(weights[3], weights[4], weights[5]))  # fmt: skip
    # the shared expert's FC1 rounding differs (vLLM's dense MLP rounds it): cancelling
    # outputs keep a summand's ulp, so the norm, not elementwise
    assert rel(out, full) <= 2e-3


@pytest.mark.parametrize("provider", ["composed", "triton_block"])
@pytest.mark.parametrize("ep", [2, 4])
@pytest.mark.parametrize("random_partition", [False, True])
def test_ep_shared_i_slices(weights, ep, random_partition, provider):
    _admit(provider, 8, 256 // ep, 512 // ep)
    x = torch.randn((8, 2048), device="cuda", dtype=torch.bfloat16)
    valid = torch.full((1,), 5, device="cuda", dtype=torch.int32)
    whole_op = MoEBlock(provider)
    whole = whole_op(x, *weights, valid_tokens=valid)
    _check_routing(x, weights[0], whole_op.routing_ids, 5)
    order = (
        torch.randperm(256, device="cuda") if random_partition else torch.arange(256, device="cuda")
    )
    total = torch.zeros_like(whole, dtype=torch.float32)
    scale = whole.float().abs().max()
    router, up, down, shared_up, shared_down, gate = weights
    for rank in range(ep):
        local = order.chunk(ep)[rank]
        mapping = torch.full((256,), -1, dtype=torch.int32, device="cuda")
        mapping[local] = torch.arange(local.numel(), dtype=torch.int32, device="cuda")
        lo, hi = rank * (512 // ep), (rank + 1) * (512 // ep)
        sliced = (
            router,
            up[local].contiguous(),
            down[local].contiguous(),
            torch.cat((shared_up[lo:hi], shared_up[512 + lo : 512 + hi])),
            shared_down[:, lo:hi].contiguous(),
            gate,
        )
        partial_op = MoEBlock(provider)
        partial = partial_op(x, *sliced, expert_map=mapping, valid_tokens=valid)
        _check_routing(x, router, partial_op.routing_ids, 5)
        assert torch.count_nonzero(partial[5:].view(torch.int16)) == 0
        total += partial.float()
        scale = torch.maximum(scale, partial.float().abs().max())
    # Every BF16 partial and the whole block round once: at most half an ulp each.
    bound = (ep + 1) * 2.0**-8 * scale
    assert (total - whole.float()).abs().max() <= bound


@pytest.mark.parametrize("provider", ["composed", "triton_block"])
def test_shared_without_routed_hit_and_padding(weights, provider):
    _admit(provider, 4, 64, 128)
    router, up, down, shared_up, shared_down, gate = weights
    # One valid row whose shared slice is its whole output sits at the BF16 floor, like
    # HF's, so the floor gate pools 8 seeds (max and RMS over all of them).
    errors = []
    for seed in range(8):
        torch.manual_seed(3000 + seed)
        x = torch.randn((4, 2048), device="cuda", dtype=torch.bfloat16)
        whole_op = MoEBlock(provider)
        whole_op(x, *weights)
        selected = set(whole_op.routing_ids[0].cpu().tolist())
        local = torch.tensor([e for e in range(256) if e not in selected][:64], device="cuda")
        sliced = (
            router,
            up[local].contiguous(),
            down[local].contiguous(),
            torch.cat((shared_up[:128], shared_up[512:640])),
            shared_down[:, :128].contiguous(),
            gate,
        )
        mapping = torch.full((256,), -1, device="cuda", dtype=torch.int32)
        mapping[local] = torch.arange(64, device="cuda", dtype=torch.int32)
        valid = torch.full((1,), 1, device="cuda", dtype=torch.int32)
        op = MoEBlock(provider)
        actual = op(x, *sliced, expert_map=mapping, valid_tokens=valid)
        errors.append(_block_errors(actual, x, sliced, op.routing_ids, mapping, valid=1))
    _assert_floor(errors)
    assert actual[0].abs().max() > 0
    assert torch.count_nonzero(actual[1:].view(torch.int16)) == 0
    if provider == "triton_block":
        op.routing_ids.fill_(165)
        op(x, *sliced, expert_map=mapping, valid_tokens=valid)
        assert (op.routing_ids[1:] == 165).all()
    else:
        assert (op.routing_ids[1:] == -1).all()
    valid.zero_()
    assert (
        torch.count_nonzero(
            op(x, *sliced, expert_map=mapping, valid_tokens=valid).view(torch.int16)
        )
        == 0
    )


@pytest.mark.parametrize(
    "rows,provider",
    [
        (1, "triton_block"),
        (4, "triton_block"),
        (8, "triton_block"),
        (64, "triton_block"),
        (512, "composed"),
    ],
)
def test_block_graph_new_routing(weights, rows, provider):
    _admit(provider, rows)
    x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
    valid = torch.full((1,), rows, device="cuda", dtype=torch.int32)
    op = MoEBlock(provider)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, *weights, valid_tokens=valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = op(x, *weights, valid_tokens=valid)
    captured_ids = op.routing_ids
    old_ids = captured_ids.clone()
    errors = []
    for valid_count in (rows, max(0, rows - 1), rows):
        x.normal_()
        valid.fill_(valid_count)
        graph.replay()
        saved = captured.clone()
        replay_ids = captured_ids.clone()
        eager = op(x, *weights, valid_tokens=valid)
        if provider == "triton_block":
            assert torch.equal(saved, eager)
        else:
            torch.testing.assert_close(saved, eager, atol=1e-3, rtol=0.02)
        assert torch.count_nonzero(saved[valid_count:].view(torch.int16)) == 0
        if valid_count:
            assert not torch.equal(old_ids[:valid_count], replay_ids[:valid_count])
        # Validate changed routing independently as well as against eager execution.
        errors.append(_block_errors(saved, x, weights, replay_ids, valid=valid_count))
    _assert_floor(errors)


@pytest.mark.parametrize(
    "config", [None, {"up": {"BLOCK_SIZE_M": 32}, "down": {"BLOCK_SIZE_M": 16}}]
)
def test_experts_cache_growth_alias_and_map(config):
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    w1 = torch.randn((4, 128, 128), **opts).mul_(0.02)
    w2 = torch.randn((4, 128, 64), **opts).mul_(0.02)
    mapping = torch.tensor([-1, 2, -1, 0, 1, -1, 3, -1], device="cuda", dtype=torch.int32)
    op = Experts(config=config)
    previous_capacity = (0, 0, 0)
    for rows in (1, 8, 3, 32, 2):
        x = torch.randn((rows, 128), **opts)
        ids = torch.randint(-1, 8, (rows, 2), device="cuda", dtype=torch.int32)
        weights = torch.rand((rows, 2), device="cuda")
        weights /= weights.sum(-1, keepdim=True)
        expected = fused_experts(x, w1, w2, weights, ids, expert_map=mapping, config=config)
        actual = op(x, w1, w2, weights, ids, expert_map=mapping, out=x)
        assert actual is x
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        capacity = tuple(b.numel() for b in next(iter(op._buffers.values())))
        assert all(a >= b for a, b in zip(capacity, previous_capacity, strict=True))
        previous_capacity = capacity
        assert op.provider == "triton" and set(op.launch_config) == {"up", "down"}


@pytest.mark.parametrize("provider", ["composed", "triton_block"])
def test_warmed_block_has_no_host_synchronization(weights, provider):
    _admit(provider, 8)
    x = torch.randn((8, 2048), device="cuda", dtype=torch.bfloat16)
    op = MoEBlock(provider)
    op(x, *weights)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        op(x, *weights)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.parametrize("provider", ["composed", "triton_block"])
def test_routing_ties_versus_mismatches(weights, provider):
    """Many independent rows: count genuine boundary ties and require no true mismatch."""
    _admit(provider, 8)
    ties = 0
    for seed in range(32):
        torch.manual_seed(5000 + seed)
        x = torch.randn((8, 2048), device="cuda", dtype=torch.bfloat16)
        op = MoEBlock(provider)
        op(x, *weights)
        tie, mismatch = classify_topk(x, weights[0], op.routing_ids)
        assert mismatch == 0, (seed, tie, mismatch)
        ties += tie
    print(f"ROUTING {provider} rows=256 ties={ties} mismatches=0")


def test_routing_classifier_flags_a_wrong_expert(weights):
    x = torch.randn((4, 2048), device="cuda", dtype=torch.bfloat16)
    logits = F.linear(x.double(), weights[0].double())
    ids = logits.topk(8, dim=-1).indices.int()
    assert classify_topk(x, weights[0], ids) == (0, 0)
    ids[1, -1] = int(logits[1].argmin())  # a clearly wrong expert, far from the boundary
    assert classify_topk(x, weights[0], ids) == (0, 1)


def test_routed_outputs_combine_as_vllm():
    """FC2 stores unweighted BF16 rows; the block combines them as vLLM (FlashInfer's FMA chain
    in route order, one rounding)."""
    pytest.importorskip("triton")
    from ..modules._moe_reference import combine_reference

    dtype = torch.bfloat16
    e, inter, h, t = 64, 256, 512, 33
    torch.manual_seed(11)
    opts = {"device": "cuda", "dtype": dtype}
    w1 = torch.randn(e, 2 * inter, h, **opts).mul_(0.02)
    w2 = torch.randn(e, h, inter, **opts).mul_(0.02)
    x = torch.randn(t, h, **opts)
    ids = torch.stack([torch.randperm(e, device="cuda")[:8] for _ in range(t)]).int()
    probs = torch.rand(t, 8, device="cuda")
    probs /= probs.sum(-1, keepdim=True)
    op = Experts()
    out = op(x, w1, w2, probs, ids)
    buffers = next(iter(op._buffers.values()))
    assert buffers[1].dtype == dtype
    rows = buffers[1][: t * 8 * h].view(t, 8, h)
    expected = combine_reference(rows, probs, ids >= 0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def _sentinels():
    """Fill the current stream's free allocator blocks (including the storage a superseded
    buffer released) with NaN, largest pieces first, until each size needs a new segment."""
    sentinels = []
    for shift in range(26, 11, -1):  # 64 MiB down to 4 KiB
        while True:
            reserved = torch.cuda.memory_reserved()
            sentinels.append(
                torch.full((1 << shift - 1,), float("nan"), dtype=torch.bfloat16, device="cuda")
            )
            if torch.cuda.memory_reserved() > reserved:
                break
    return sentinels


def _replay_after_growth(fill, run, rows, larger):
    """Warm and capture at ``rows``, run eagerly at ``larger`` on the capture stream (the
    cache key), take the freed storage with NaN sentinels, then replay with new inputs: the
    graph leaves the sentinels intact and computes the captured rows as an eager call does.
    ``fill(rows, seed)`` writes new inputs into the static ones that ``run(rows)`` uses."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    fill(rows, 1)
    fill(larger, 7)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        run(rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(rows)
    with torch.cuda.stream(stream):
        run(larger)
        # Freed blocks are reused only by allocations on the stream that freed them.
        sentinels = _sentinels()
    torch.cuda.synchronize()
    for seed in (2, 3):
        fill(rows, seed)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            graph.replay()
            replayed = captured.clone()
            expected = run(rows)
        torch.cuda.synchronize()
        assert torch.equal(replayed, expected)
    for sentinel in sentinels:
        assert torch.isnan(sentinel).all(), "a captured graph wrote freed memory"


def _expert_inputs(w1, mapping, valid):
    static = {}

    def fill(rows, seed):
        torch.manual_seed(seed)
        x = torch.randn(rows, w1.shape[2], device="cuda", dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(w1.shape[0], device="cuda")[:8] for _ in range(rows)])
        probs = torch.rand(rows, 8, device="cuda")
        probs /= probs.sum(-1, keepdim=True)
        count = torch.full((1,), rows, dtype=torch.int32, device="cuda")
        args = (x, probs, ids.int(), count)
        if rows in static:
            for dst, src in zip(static[rows], args, strict=True):
                dst.copy_(src)
        else:
            static[rows] = args

    def run(op, w2):
        def call(rows):
            x, probs, ids, count = static[rows]
            return op(
                x, w1, w2, probs, ids, expert_map=mapping, valid_tokens=count if valid else None
            )

        return call

    return fill, run


def _expert_weights(seed):
    e, inter, h = 128, 768, 2048
    torch.manual_seed(seed)
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    w1 = torch.randn(e, 2 * inter, h, **opts).mul_(0.02)
    return w1, torch.randn(e, h, inter, **opts).mul_(0.02)


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_experts_graph_keeps_storage_after_eager_growth(valid, mapped):
    """M1: a larger eager call replaces the cached intermediates; a graph captured earlier
    keeps the storage it uses (Qwen3-30B-A3B widths)."""
    pytest.importorskip("triton")
    w1, w2 = _expert_weights(4)
    mapping = None
    if mapped:
        mapping = torch.arange(w1.shape[0], dtype=torch.int32, device="cuda")
        mapping[1::2] = -1
    op = Experts()
    fill, run = _expert_inputs(w1, mapping, valid)
    _replay_after_growth(fill, run(op, w2), 16, 300)
    assert op._graph_storage


def test_block_composed_graph_keeps_storage_after_eager_growth(weights):
    """The engine's scenario at Qwen3.5-35B-A3B widths: capture the composed block above the
    whole-block rows, run an eager batch of 1024 rows on the stream, then replay."""
    pytest.importorskip("triton")
    op = MoEBlock("composed")
    static = {}

    def fill(rows, seed):
        torch.manual_seed(seed)
        x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
        static.setdefault(rows, x).copy_(x)

    _replay_after_growth(fill, lambda rows: op(static[rows], *weights), 72, 1024)


def test_prewarmed_experts_hold_no_extra_storage():
    """Warmed at the most rows first (as the engine runner does), graphs captured at fewer
    rows reuse the warm cache: pinning holds no storage beyond it."""
    pytest.importorskip("triton")
    w1, w2 = _expert_weights(5)
    op = Experts()
    fill, run = _expert_inputs(w1, None, False)
    call = run(op, w2)
    for rows in (1, 2, 4, 64, 256, 512):
        fill(rows, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    with torch.cuda.stream(stream):
        call(512)
        for rows in (1, 2, 4, 64, 512):
            call(rows)
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1], stream=stream):
                call(rows)
        call(256)
    torch.cuda.synchronize()
    cached = {b.data_ptr() for bufs in op._buffers.values() for b in bufs}
    pinned = {ptr for ptr, _ in op._graph_storage}
    assert pinned and pinned <= cached


BUCKETS = (64, 48, 40, 32, 24, 16, 8, 4, 2, 1)


@pytest.mark.parametrize("scoped", [False, True])
def test_block_graphs_share_one_workspace(weights, scoped, record_property):
    """Engine order at Qwen3.5-35B-A3B shapes over two layers: per bucket, largest first, an
    eager call then a capture on another stream. All graphs reuse one eagerly sized
    workspace (each op's own, or the scope's slot shared by both), and every replay equals an
    eager call bitwise."""
    _admit("triton_block", BUCKETS[0])
    ops = [MoEBlock("triton_block") for _ in range(2)]
    xs = {rows: torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16) for rows in BUCKETS}
    scope = GraphWorkspace(max_rows=BUCKETS[0]) if scoped else nullcontext()
    stream, pool = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs = {}, {}
    with scope:
        for op in ops:
            op(xs[BUCKETS[0]], *weights)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        held = [ws for op in ops for ws in op._workspaces.values()]
        for rows in BUCKETS:
            for op in ops:
                op(xs[rows], *weights)
            graphs[rows] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[rows], pool=pool, stream=stream):
                outputs[rows] = [op(xs[rows], *weights) for op in ops]
        torch.cuda.synchronize()
        grown = torch.cuda.memory_allocated() - before
        record_property("graph_bytes", grown)
        assert [ws for op in ops for ws in op._workspaces.values()] == held
        assert len(held) == (0 if scoped else len(ops)) and not any(op._retired for op in ops)
        # Workspace bytes stay in the eager sizing; graphs hold only their outputs.
        assert grown <= len(ops) * sum(BUCKETS) * 2048 * 2 + (1 << 20)
        for rows in BUCKETS:
            xs[rows].normal_()
            graphs[rows].replay()
            for op, output in zip(ops, outputs[rows], strict=True):
                assert torch.equal(output, op(xs[rows], *weights))
    if scoped:
        scope.close()


@pytest.mark.parametrize("scoped", [False, True])
def test_block_capture_beyond_eager_rows_raises(weights, scoped):
    _admit("triton_block", 8)
    op = MoEBlock("triton_block")
    scope = GraphWorkspace(max_rows=4) if scoped else nullcontext()
    with scope:
        op(torch.randn((4, 2048), device="cuda", dtype=torch.bfloat16), *weights)
        x = torch.randn((8, 2048), device="cuda", dtype=torch.bfloat16)
        message = "8 rows exceed max_rows=4" if scoped else "eager call of at least 8 rows"
        with pytest.raises(RuntimeError, match=message):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                op(x, *weights)


def test_block_triton_graph_keeps_workspace_after_eager_growth(weights):
    _admit("triton_block", 64)
    op = MoEBlock("triton_block")
    static = {}

    def fill(rows, seed):
        torch.manual_seed(seed)
        x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
        static.setdefault(rows, x).copy_(x)

    _replay_after_growth(fill, lambda rows: op(static[rows], *weights), 8, 64)
    assert len(op._retired) == 1
