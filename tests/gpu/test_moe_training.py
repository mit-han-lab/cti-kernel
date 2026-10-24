# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CUDA qualification of Qwen3.5 and Qwen3 MoE dX; CTI_MOE_PROVIDER selects GEMMs."""

from __future__ import annotations

import pytest
import torch

from ctikernels.moe import moe_block, moe_block_training_bytes, training

from ..modules import _moe_reference as ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TOKENS = (1, 2, 3, 7, 8, 16, 64, 127, 128, 1024, 4097)


@pytest.fixture(scope="module")
def weights():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return ref.make_weights(h=2048, e=256, ni=512, seed=60, device="cuda")


@pytest.fixture(scope="module")
def qwen3_weights():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return ref.make_weights(h=2048, e=128, ni=768, has_shared=False, seed=60, device="cuda")


def inputs(t, seed=1):
    return ref.make_x(t, h=2048, seed=seed, device="cuda")


def _exact_backend(x, wt):
    # Triton is bitwise reproducible across modes. CuTe may tile saved and recomputed
    # backward differently.
    ni = wt["w2"].shape[2]
    nis = wt["s2"].shape[1] if wt["s2"] is not None else ni
    from ctikernels.moe.cute._provider import CuteBackend

    backend = training._provider_for(x.device, ni, nis, x.shape[1])
    return not isinstance(backend, CuteBackend)


def _same_results(actual, expected, x, wt):
    exact = _exact_backend(x, wt)
    for got, want in zip(actual, expected, strict=True):
        if exact or not got.is_floating_point():
            assert torch.equal(got, want)
        else:
            assert torch.isfinite(got).all() and torch.isfinite(want).all()
            error = (got.float() - want.float()).norm()
            assert error <= 1e-3 * want.float().norm().clamp_min(1e-30)


def _fp32_check(x, weights, dy, out, rec, dx):
    # Bounded temporary memory, but every row is checked against autograd.
    for start in range(0, x.shape[0], 256):
        sl = slice(start, start + 256)
        oo, dd, _, _ = ref.oracle(
            x[sl], *weights.values(), rec[sl].long(), dy[sl], dtype=torch.float32, kappa=False
        )
        for got, want in ((out[sl], oo), (dx[sl], dd)):
            assert (got.float() - want).norm() / want.norm().clamp_min(1e-30) < 0.02


def _sum_ranks(x, wt, dy, ep, **kwargs):
    if ep == 1:
        return ref.op_grad(x, wt, dy, top_k=8, **kwargs)
    outs, dxs, record = [], [], None
    for rank in range(ep):
        shard, mapping = ref.shard_weights(wt, ep, rank)
        out, rec, dx = ref.op_grad(x, shard, dy, top_k=8, expert_map=mapping, **kwargs)
        if record is not None:
            assert torch.equal(record, rec)
        record = rec
        outs.append(out.float())
        dxs.append(dx.float())
    return sum(outs).bfloat16(), record, sum(dxs).bfloat16()


def _correctness(weights, t, ep, replay=False):
    x, dy = inputs(t, 100 + t), inputs(t, 200 + t)
    kw = {}
    if replay:
        g = torch.Generator().manual_seed(t)
        e = weights["r"].shape[0]
        kw["routing"] = torch.stack([torch.randperm(e, generator=g)[:8] for _ in range(t)]).to(
            device=x.device, dtype=torch.uint8
        )
    out, rec, dx = _sum_ranks(x, weights, dy, ep, **kw)
    assert torch.isfinite(out).all() and torch.isfinite(dx).all()
    if replay:
        assert torch.equal(rec, kw["routing"])
    else:
        assert ref.admissible_sets(x, weights["r"], rec.long()).all()
    # FP32 autograd on every row, bounded temporary memory; FP64 paired gate on sampled rows.
    _fp32_check(x, weights, dy, out, rec, dx)
    idx = torch.linspace(0, t - 1, min(t, 16), device=x.device).long()
    # The row-clustered SE needs several rows: with one or two it is (close to) zero, and the bias
    # statistics would compare a single row's mean error. Smaller T keeps the FP32 check above.
    if idx.numel() >= 8:
        ref.check_gate(x[idx], weights, dy[idx], out[idx], rec[idx], dx[idx], row_se=True, ep=ep)
    if ep > 1:
        full = ref.op_grad(x[idx], weights, dy[idx], top_k=8, routing=rec[idx].contiguous())
        for got, want in ((out[idx], full[0]), (dx[idx], full[2])):
            assert (got.float() - want.float()).norm() / want.float().norm() < 0.02


def _routing_check(x, wt, dy, out, rec, dx, norm):
    """FP32 autograd on every row and FP64 autograd on sampled rows, for either router mode.

    With norm_topk_prob off (and several rows), the FP64 ablation without the unselected
    experts' router derivative must be much farther from dX than the full derivative."""
    for start in range(0, x.shape[0], 256):
        sl = slice(start, start + 256)
        oo, dd, _, _ = ref.oracle(
            x[sl],
            *wt.values(),
            rec[sl].long(),
            dy[sl],
            dtype=torch.float32,
            kappa=False,
            renormalize=norm,
        )
        for got, want in ((out[sl], oo), (dx[sl], dd)):
            assert (got.float() - want).norm() / want.norm().clamp_min(1e-30) < 0.02
    idx = torch.linspace(0, x.shape[0] - 1, min(x.shape[0], 16), device=x.device).long()
    args = (x[idx], *wt.values(), rec[idx].long(), dy[idx])
    oo, dd, _, _ = ref.oracle(*args, kappa=False, renormalize=norm)
    errors = []
    for got, want in ((out[idx], oo), (dx[idx], dd)):
        errors.append((got.double() - want).norm() / want.norm().clamp_min(1e-300))
        assert errors[-1] < 0.01
    # One row's unselected-router term can sit within a few BF16 ulps; T >= 7 discriminates.
    if not norm and idx.numel() >= 7:
        _, ablated, _, _ = ref.oracle(*args, kappa=False, renormalize=False, unselected_grad=False)
        assert (dx[idx].double() - ablated).norm() / dd.norm() > 5 * errors[1]
    wrong, _, _, _ = ref.oracle(*args, kappa=False, renormalize=not norm)
    assert (out[idx].double() - wrong).norm() / oo.norm() > 0.05


@pytest.mark.parametrize("norm", (True, False))
@pytest.mark.parametrize("geometry", ("qwen35", "qwen3"))
@pytest.mark.parametrize("ep", (1, 2, 4))
@pytest.mark.parametrize("t", (1, 7, 64, 4097))
def test_norm_topk_prob(weights, qwen3_weights, norm, geometry, ep, t):
    wt = weights if geometry == "qwen35" else qwen3_weights
    x, dy = inputs(t, 500 + t), inputs(t, 600 + t)
    kw = {"norm_topk_prob": norm}
    if t > 1024:
        kw.update(chunk_tokens=1024, save_activations=True)
    got = _sum_ranks(x, wt, dy, ep, **kw)
    assert torch.isfinite(got[0]).all() and torch.isfinite(got[2]).all()
    assert ref.admissible_sets(x, wt["r"], got[1].long()).all()
    _routing_check(x, wt, dy, *got, norm)
    if t == 64:
        # Masked tail rows stay +0.0 even with NaN inputs; live rows are unchanged.
        v = torch.tensor(37, dtype=torch.int32, device="cuda")
        xn, dn = x.clone(), dy.clone()
        xn[37:], dn[37:] = float("nan"), float("nan")
        masked = _sum_ranks(xn, wt, dn, ep, valid_tokens=v, routing=got[1], **kw)
        for a, b in zip(masked, got, strict=True):
            assert not a[37:].count_nonzero()
            if a.is_floating_point():
                assert not torch.signbit(a[37:]).any()
                error = (a[:37].float() - b[:37].float()).norm()
                assert error <= 1e-2 * b[:37].float().norm()


@pytest.mark.parametrize("t", TOKENS)
def test_real_shape(weights, t):
    _correctness(weights, t, ep=1)


@pytest.mark.parametrize("ep", (1, 2))
@pytest.mark.parametrize("t", (1, 64, 1024))
def test_qwen3_no_shared(qwen3_weights, ep, t):
    _correctness(qwen3_weights, t, ep=ep)


@pytest.mark.parametrize("ep", (2, 4))
@pytest.mark.parametrize("t", (1, 7, 64, 128))
def test_expert_parallel(weights, ep, t):
    _correctness(weights, t, ep=ep)


@pytest.mark.parametrize("save", (False, True))
def test_chunked_equals_single_chunk(weights, save):
    # Row math is independent of the chunking while every chunk uses the same tile bucket
    # (two 2048-token chunks and one 4096-token chunk).
    t = 4096
    x, dy = inputs(t, 5), inputs(t, 6)
    whole = ref.op_grad(x, weights, dy, top_k=8, save_activations=save)
    split = ref.op_grad(
        x, weights, dy, top_k=8, routing=whole[1], save_activations=save, chunk_tokens=2048
    )
    _same_results(split, whole, x, weights)


def _held(tensor):
    return -(-tensor.numel() * tensor.element_size() // 512) * 512


@pytest.mark.parametrize("save", (False, True))
@pytest.mark.parametrize("norm", (True, False))
@pytest.mark.parametrize("t", (1, 300, 4096, 9000))
def test_training_bytes_bound_measured_memory(weights, t, norm, save):
    """moe_block_training_bytes at 35B shapes: the saved state and the backward's peak match
    the bound (renormalized routing; an upper bound otherwise), up to the caching allocator's
    unsplit blocks (at most 1 MiB for each of the backward's four large buffers); the forward's
    peak, its output included, fits in their sum."""
    wt = weights
    x, dy = inputs(t, 7).requires_grad_(), inputs(t, 8)
    run = lambda: moe_block(
        x, wt["r"], wt["sg"], wt["w13"], wt["w2"], wt["s13"], wt["s2"], top_k=8,
        save_activations=save, norm_topk_prob=norm,
    )  # fmt: skip
    torch.autograd.backward(run()[0], dy)  # compile and warm every cache first
    x.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out, record = run()
    torch.cuda.synchronize()
    forward = torch.cuda.max_memory_allocated() - start + _held(x)
    saved = torch.cuda.memory_allocated() - start - _held(out) - _held(record) + _held(x)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.autograd.backward(out, dy)
    backward = torch.cuda.max_memory_allocated() - before
    bound = moe_block_training_bytes(
        t, hidden=2048, intermediate=512, experts=256, top_k=8, shared_intermediate=512,
        device=x.device, save_activations=save, norm_topk_prob=norm,
    )  # fmt: skip
    for got, most in zip((saved, backward), bound, strict=True):
        assert got <= most + (4 << 20), (got, most)
        if norm:
            assert most - got <= 64 << 10, (got, most)
    assert forward <= sum(bound) + (4 << 20), (forward, bound)  # the forward's transients fit


@pytest.mark.parametrize("ep", (1, 2, 4))
def test_replay(weights, ep):
    _correctness(weights, 16, ep=ep, replay=True)


@pytest.mark.exhaustive
@pytest.mark.parametrize("ep", (1, 2, 4))
@pytest.mark.parametrize(
    "t",
    (1, 2, 3, 7, 8, 15, 16, 31, 64, 127, 128, 255, 256, 511, 1024, 1025, 2048, 4096, 4097, 8192),
)
@pytest.mark.parametrize("replay", (False, True))
def test_long_sweep(weights, ep, t, replay):
    _correctness(weights, t, ep=ep, replay=replay)


@pytest.mark.parametrize("ep", (1, 2, 4))
@pytest.mark.parametrize("has_shared", (True, False))
def test_saved_valid_and_identity(weights, qwen3_weights, ep, has_shared):
    weights = weights if has_shared else qwen3_weights
    wt, mapping = ref.shard_weights(weights, ep, 0)
    x, dy = inputs(16, 81), inputs(16, 82)
    for v in (0, 1, 7, 16):
        xn, dn = x.clone(), dy.clone()
        xn[v:], dn[v:] = float("nan"), float("nan")
        valid = torch.tensor(v, dtype=torch.int32, device=x.device)
        kw = {"top_k": 8, "expert_map": mapping, "valid_tokens": valid}
        a = ref.op_grad(xn, wt, dn, **kw)
        b = ref.op_grad(xn, wt, dn, save_activations=True, **kw)
        _same_results(a, b, xn, wt)
        for result in (a, b):
            assert all(not u[v:].count_nonzero() for u in result)
            assert all(not torch.signbit(u[v:]).any() for u in (result[0], result[2]))
        assert torch.isfinite(a[0]).all() and torch.isfinite(a[2]).all()
        if ep == 1:
            kw["expert_map"] = None
            c = ref.op_grad(xn, wt, dn, **kw)
            _same_results(a, c, xn, wt)


def test_planted_bias(weights):
    ref.check_planted_bias(inputs(64, 160), weights, inputs(64, 260), top_k=8)


@pytest.mark.parametrize(
    "h,ni,nis",
    (
        (16, 16, 16),
        (32, 32, 16),
        (64, 16, 16),
        (64, 32, 32),
        (64, 64, 32),
        (128, 64, 128),
        (256, 128, 64),
    ),
)
def test_dims_smoke(h, ni, nis):
    wt = ref.make_weights(h=h, e=8, ni=ni, nis=nis, seed=h * 1000 + ni, device="cuda")
    x, dy = (ref.make_x(24, h=h, seed=s, device="cuda") for s in (1, 2))
    a = ref.op_grad(x, wt, dy, top_k=2)
    b = ref.op_grad(x, wt, dy, top_k=2, save_activations=True)
    _same_results(a, b, x, wt)
    _, dx, _, _ = ref.oracle(x, *wt.values(), a[1].long(), dy, dtype=torch.float32, kappa=False)
    assert (a[2].float() - dx).norm() / dx.norm() < 0.02


@pytest.mark.parametrize("provider", ("auto", "cute", "triton"))
@pytest.mark.parametrize("geometry", ("qwen35", "qwen3"))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("saved", (False, True))
@pytest.mark.parametrize("mapped", (False, True))
@pytest.mark.parametrize("graph_capture", (False, True), ids=("eager", "graph"))
def test_empty_tokens(monkeypatch, provider, geometry, masked, saved, mapped, graph_capture):
    from ctikernels.moe.cute._provider import CuteBackend, available
    from ctikernels.moe.triton._provider import _TritonBackend

    cap = torch.cuda.get_device_capability()
    if provider == "cute" and cap not in ((12, 0), (12, 1)):
        pytest.skip("sm_120/121 required for CuTe")
    use_cute = provider == "cute" or (provider == "auto" and cap in ((12, 0), (12, 1)))
    if use_cute and not available():
        pytest.skip("CuTe DSL 4.8.0 required")
    if provider == "auto":
        monkeypatch.delenv("CTI_MOE_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("CTI_MOE_PROVIDER", provider)
    for name in ("CONFIG", "LAYOUT", "FUSED", "TOKEN", "ROUTER", "COMBINE", "PDL"):
        monkeypatch.delenv(f"CTI_MOE_CUTE_{name}", raising=False)
    shared = geometry == "qwen35"
    ni, nis = (64, 128) if shared else (128, 128)
    wt = ref.make_weights(h=64, e=8, ni=ni, nis=nis, has_shared=shared, device="cuda")
    mapping = None
    if mapped:
        wt, mapping = ref.shard_weights(wt, 2, 0)
    x = ref.make_x(0, h=64, device="cuda").requires_grad_()
    dy = torch.empty_like(x)
    valid = torch.tensor(0, dtype=torch.int32, device="cuda") if masked else None
    routing = torch.empty(0, 2, dtype=torch.uint8, device="cuda")
    backend = training._provider_for(x.device, ni, wt["s2"].shape[1] if shared else ni, 64)
    assert isinstance(backend, CuteBackend if use_cute else _TritonBackend)

    def step(norm, replay):
        out, record = moe_block(
            x,
            *wt.values(),
            top_k=2,
            expert_map=mapping,
            routing=replay,
            valid_tokens=valid,
            save_activations=saved,
            norm_topk_prob=norm,
        )
        return out, record, torch.autograd.grad(out, x, dy)[0]

    def check(result):
        out, record, dx = result
        for value in (out, dx):
            assert value.shape == (0, 64) and value.dtype == torch.bfloat16
            assert value.device == x.device and value.is_contiguous()
        assert record.shape == (0, 2) and record.dtype == torch.uint8
        assert record.device == x.device and record.is_contiguous()
        assert not record.requires_grad

    for norm in (False, True):
        for replay in (None, routing):
            if not graph_capture:
                check(step(norm, replay))
                continue
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step(norm, replay)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = step(norm, replay)
            for _ in range(2):
                graph.replay()
                check(captured)
    torch.cuda.synchronize()


@pytest.mark.parametrize("norm", (True, False))
@pytest.mark.parametrize("ep", (1, 2, 4))
@pytest.mark.parametrize("replay", (False, True))
def test_cuda_graph(weights, ep, replay, norm):
    wt, mapping = ref.shard_weights(weights, ep, 0)
    x, dy = inputs(16, 41).requires_grad_(), inputs(16, 42)
    valid = torch.tensor(16, dtype=torch.int32, device="cuda")
    routing = torch.arange(8, device="cuda", dtype=torch.uint8).expand(16, -1).contiguous()

    def step():
        out, rec = moe_block(
            x,
            *wt.values(),
            top_k=8,
            expert_map=mapping,
            routing=routing if replay else None,
            valid_tokens=valid,
            norm_topk_prob=norm,
        )
        return out, rec, torch.autograd.grad(out, x, dy)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed, v in ((43, 16), (44, 7), (45, 1), (46, 0)):
        with torch.no_grad():
            x.copy_(inputs(16, seed))
            dy.copy_(inputs(16, seed + 100))
            x[v:], dy[v:] = float("nan"), float("nan")
            valid.fill_(v)
            routing.copy_((torch.arange(8, device="cuda") + seed).to(torch.uint8).expand(16, -1))
        graph.replay()
        eager = step()
        _same_results(captured, eager, x, wt)
        for result in (captured, eager):
            assert all(not a[v:].count_nonzero() for a in result)
            assert all(not torch.signbit(a[v:]).any() for a in (result[0], result[2]))


@pytest.mark.parametrize("norm", (True, False))
@pytest.mark.parametrize("geometry", ("qwen35", "qwen3"))
@pytest.mark.parametrize("t", (1, 64, 4097))
def test_router_bitwise_vs_unsloth(weights, qwen3_weights, norm, geometry, t):
    """Selections, records, BF16 route weights and the BF16 shared gate equal Unsloth's bits."""
    from ctikernels.moe._training_backend import _route

    wt = weights if geometry == "qwen35" else qwen3_weights
    x = inputs(t, 700 + t)
    k = 8
    ids, top, gate = ref.unsloth_routing(x, wt["r"], wt["sg"], k, renormalize=norm)
    sel, slot_w, _, live, record, _ = _route(x, wt["r"], wt["sg"], k, None, None, norm)
    assert live.all() and torch.equal(sel.long(), ids) and torch.equal(record, ids.to(torch.uint8))
    assert torch.equal(slot_w[:, :k], top.float())
    if gate is not None:
        assert torch.equal(slot_w[:, k:], gate.float())
    _, rec = moe_block(x, *wt.values(), top_k=k, norm_topk_prob=norm)
    assert torch.equal(rec, ids.to(torch.uint8))


@pytest.mark.parametrize("provider", ("triton", "cute"))
@pytest.mark.parametrize("geometry", ("qwen35", "qwen3"))
@pytest.mark.parametrize("saved", (False, True))
@pytest.mark.parametrize("t", (1, 64, 4097))
def test_run_to_run_bitwise(weights, qwen3_weights, monkeypatch, provider, geometry, saved, t):
    """No atomics: repeated runs give identical output, record and dX bits."""
    monkeypatch.setenv("CTI_MOE_PROVIDER", provider)
    wt = weights if geometry == "qwen35" else qwen3_weights
    x, dy = inputs(t, 300 + t), inputs(t, 400 + t)
    kw = {"save_activations": saved, "chunk_tokens": 1024}
    first = ref.op_grad(x, wt, dy, **kw)
    for _ in range(2):
        for got, want in zip(ref.op_grad(x, wt, dy, **kw), first, strict=True):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@pytest.mark.parametrize("saved", (False, True))
@pytest.mark.parametrize("has_shared", (False, True))
@pytest.mark.parametrize("t,chunk", ((17, 16), (1025, 1024)))
def test_cute_pdl_eager_and_graph(monkeypatch, saved, has_shared, t, chunk):
    """PDL on/off must agree, including changing masks and inputs on graph replay."""
    from ctikernels.moe.cute._provider import CuteBackend, available

    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)) or not available():
        pytest.skip("sm_120/121 and nvidia-cutlass-dsl 4.8.0 required")
    monkeypatch.setenv("CTI_MOE_PROVIDER", "cute")
    monkeypatch.setenv("CTI_MOE_CUTE_LAYOUT", "1")
    monkeypatch.delenv("CTI_MOE_CUTE_CONFIG", raising=False)
    wt = ref.make_weights(h=256, e=8, ni=128, nis=128, has_shared=has_shared, device="cuda")
    wt, mapping = ref.shard_weights(wt, 2, 0)
    x = ref.make_x(t, h=256, seed=501, device="cuda").requires_grad_()
    dy = ref.make_x(t, h=256, seed=502, device="cuda")
    valid = torch.tensor(t, dtype=torch.int32, device="cuda")
    routing = torch.arange(2, dtype=torch.uint8, device="cuda").expand(t, -1).contiguous()

    def step():
        out, rec = moe_block(
            x,
            *wt.values(),
            top_k=2,
            expert_map=mapping,
            routing=routing,
            valid_tokens=valid,
            save_activations=saved,
            chunk_tokens=chunk,
        )
        return out, rec, torch.autograd.grad(out, x, dy)[0]

    def same(actual, expected):
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all() and torch.isfinite(want).all()
            # Compare storage bits too, including the sign of zero.
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))

    graphs, captures, eager = [], [], []
    for enabled in ("0", "1"):
        monkeypatch.setenv("CTI_MOE_CUTE_PDL", enabled)
        backend = training._provider_for(x.device, 128, 64 if has_shared else 128, 256)
        assert isinstance(backend, CuteBackend) and backend.pdl == (enabled == "1")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = step()
        torch.cuda.current_stream().wait_stream(stream)
        eager.append(result)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captures.append(step())
        graphs.append(graph)
    same(eager[1], eager[0])

    # Includes empty chunks, remote-only routes, short K and unused layout CTAs.
    for seed, v in ((502, t), (503, chunk - 1), (504, 1), (505, 0)):
        with torch.no_grad():
            x.copy_(ref.make_x(t, h=256, seed=seed, device="cuda"))
            dy.copy_(ref.make_x(t, h=256, seed=seed + 100, device="cuda"))
            x[v:], dy[v:] = float("nan"), float("nan")
            valid.fill_(v)
            routing.copy_(((torch.arange(2, device="cuda") + seed) % 8).expand(t, -1))
        eager = []
        for i, enabled in enumerate(("0", "1")):
            monkeypatch.setenv("CTI_MOE_CUTE_PDL", enabled)
            graphs[i].replay()
            eager.append(step())
            same(captures[i], eager[i])
            assert all(not value[v:].count_nonzero() for value in captures[i])
        same(eager[1], eager[0])
        same(captures[1], captures[0])
