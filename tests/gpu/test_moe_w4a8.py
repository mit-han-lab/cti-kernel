# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""W4A8 (MXFP4 x MXFP8) routed experts: accuracy vs the FP32 dequant reference, expert
parallelism, determinism, graph capture and zero rows."""

import pytest
import torch

from ctikernels import GraphWorkspace
from ctikernels.moe import MXFP4Experts
from ctikernels.moe._common import interleave_scales_
from ctikernels.moe.torch._w4a8_reference import mxfp4_experts

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0),
        reason="cc >= 8.0",
    ),
]

SMALL = (512, 256, 16)  # H, I, experts
DSV4 = (4096, 2048, 256)  # DeepSeek V4 Flash routed experts
TOP_K = 6


def _weights(h, i, e, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)

    def scales(*shape, center):
        return torch.randint(center - 2, center + 2, shape, generator=g, device="cuda").to(
            torch.uint8
        )

    # Scales keep gate/up and outputs O(1) for unit-variance inputs.
    return (
        nibbles(e, 2 * i, h // 2),
        scales(e, 2 * i, h // 32, center=121),
        nibbles(e, h, i // 2),
        scales(e, h, i // 32, center=122),
    )


def _clones(w):
    # The op takes over (rewrites in place) the tensors it is given.
    return [t.clone() for t in w]


def _routes(t, global_e, seed=1, pad=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = [torch.randperm(global_e, generator=g, device="cuda")[:TOP_K] for _ in range(t)]
    ids = torch.stack(ids).to(torch.int32) if t else torch.empty(0, TOP_K, dtype=torch.int32)
    ids = ids.cuda()
    if pad and t > 1:
        ids[-1] = -1
        ids[0, 1] = -1
    w = torch.rand(t, TOP_K, generator=g, device="cuda")
    return (w / w.sum(-1, keepdim=True)).contiguous(), ids.contiguous()


def _x(t, h, seed=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(t, h, generator=g, device="cuda").to(torch.bfloat16)


def _close(out, ref):
    out, ref = out.float(), ref.float()
    # Kernel and reference differ only in FP32 summation order, which can flip an MXFP8
    # rounding of the intermediate; such flips stay rare and small.
    rel = (out - ref).norm() / ref.norm().clamp_min(1e-30)
    assert rel < 2e-3, rel
    assert torch.isclose(out, ref, rtol=2e-2, atol=2e-2 * ref.abs().max()).float().mean() > 0.999


def _truth_error(out, x, w, weights, ids, **kw):
    truth = mxfp4_experts(x, *w, weights, ids, exact=True, **kw)
    return ((out.float() - truth).norm() / truth.norm()).item()


_SM12 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="the cute provider runs on sm_120/sm_121",
)

# MXFP8 activations (two quantization points) against the FP32 truth, any provider.
TRUTH_BOUND = 0.06
PROVIDERS = [
    "triton",
    pytest.param(
        "flashinfer",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
            reason="FlashInfer W4A8 runs on the sm100 and sm120 families",
        ),
    ),
]


@pytest.mark.parametrize("t", [1, 2, 3, 4, 17, 64, 257])
@pytest.mark.parametrize("limit", [None, 1.0])
def test_matches_reference(t, limit):
    h, i, e = SMALL
    w = _weights(h, i, e)
    x = _x(t, h)
    weights, ids = _routes(t, e)
    op = MXFP4Experts(*_clones(w), swiglu_limit=limit, provider="triton", scales_layout="canonical")
    out = op(x, weights, ids)
    _close(out, mxfp4_experts(x, *w, weights, ids, swiglu_limit=limit))
    assert op.provider == "triton"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("t", [1, 4, 33])
def test_expert_parallel_partial_sum(provider, t):
    h, i, e = SMALL
    global_e = 4 * e
    w = _weights(h, i, e)
    g = torch.Generator().manual_seed(3)
    expert_map = torch.full((global_e,), -1, dtype=torch.int32)
    expert_map[torch.randperm(global_e, generator=g)[:e]] = torch.arange(e, dtype=torch.int32)
    expert_map = expert_map.cuda()
    x = _x(t, h)
    weights, ids = _routes(t, global_e, pad=True)
    op = MXFP4Experts(*_clones(w), swiglu_limit=10.0, provider=provider, scales_layout="canonical")
    out = op(x, weights, ids, expert_map=expert_map)
    assert op.provider == provider
    kw = {"expert_map": expert_map, "swiglu_limit": 10.0}
    remote = (expert_map[ids.clamp(min=0).long()] < 0) | (ids < 0)
    assert not out[remote.all(-1)].any()  # tokens without a local route
    if provider == "triton":
        _close(out, mxfp4_experts(x, *w, weights, ids, **kw))
    if not remote.all():
        assert _truth_error(out, x, w, weights, ids, **kw) < TRUTH_BOUND


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("t", [1, 4, 512])
def test_dsv4_geometry(provider, t):
    h, i, e = DSV4
    w = _weights(h, i, e)
    x = _x(t, h)
    weights, ids = _routes(t, e)
    op = MXFP4Experts(*_clones(w), swiglu_limit=10.0, provider=provider, scales_layout="canonical")
    assert op.workspace_bytes == 0  # one resident copy: the given tensors
    out = op(x, weights, ids)
    if provider == "triton":
        _close(out, mxfp4_experts(x, *w, weights, ids, swiglu_limit=10.0))
    assert _truth_error(out, x, w, weights, ids, swiglu_limit=10.0) < TRUTH_BOUND
    assert torch.equal(out, op(x, weights, ids))  # deterministic


@pytest.mark.parametrize("provider", PROVIDERS)
def test_one_canonical_copy(provider):
    w = _weights(*SMALL)
    given = _clones(w)
    op = MXFP4Experts(*given, swiglu_limit=10.0, provider=provider, scales_layout="canonical")
    op(_x(4, SMALL[0]), *_routes(4, SMALL[2]))
    if op.layout == "up_first":  # weights read as-is; scales interleaved in place
        assert torch.equal(given[0], w[0]) and torch.equal(given[2], w[2])
        assert torch.equal(given[1], interleave_scales_(w[1].clone()))
    assert all(torch.equal(a, b) for a, b in zip(op.canonical_weights(), w, strict=True))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_scales_stated_interleaved(provider):
    # Scales already in the interleave (e.g. held by another op) are read as they are.
    w = _weights(*SMALL)
    x, (weights, ids) = _x(4, SMALL[0]), _routes(4, SMALL[2])
    kw = {"swiglu_limit": 10.0, "provider": provider}
    canonical = MXFP4Experts(*_clones(w), scales_layout="canonical", **kw)
    given = [
        w[0].clone(),
        interleave_scales_(w[1].clone()),
        w[2].clone(),
        interleave_scales_(w[3].clone()),
    ]
    stated = MXFP4Experts(*given, scales_layout="interleaved", **kw)
    assert torch.equal(stated(x, weights, ids), canonical(x, weights, ids))
    with pytest.raises(ValueError, match="scales_layout"):
        MXFP4Experts(*_clones(w), scales_layout="swizzled", **kw)


@pytest.mark.parametrize("provider", [*PROVIDERS, pytest.param("cute", marks=_SM12)])
def test_no_resident_copy_at_dsv4(provider):
    # Construction at the DSV4 Flash EP2 shape holds nothing beyond the given tensors
    # (expert weights are 1.59 GiB per layer here): only per-expert vectors.
    w = _weights(DSV4[0], DSV4[1], DSV4[2] // 2)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    op = MXFP4Experts(*w, swiglu_limit=10.0, provider=provider, scales_layout="canonical")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 1 << 20
    assert op.workspace_bytes == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="FlashInfer runs on the sm100 and sm120 families",
)
def test_interleave_matches_flashinfer():
    from flashinfer import block_scale_interleave

    s = _weights(*SMALL)[1]
    assert torch.equal(interleave_scales_(s.clone()), block_scale_interleave(s).reshape_as(s))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_graph_replay_and_zero_rows(provider):
    h, i, e = SMALL
    op = MXFP4Experts(
        *_weights(h, i, e), swiglu_limit=10.0, provider=provider, scales_layout="canonical"
    )
    assert op(_x(0, h), *_routes(0, e)).shape == (0, h)
    t = 4
    x, out = _x(t, h), torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
    weights, ids = _routes(t, e)
    with GraphWorkspace(max_rows=64):
        op(x, weights, ids, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, weights, ids, out=out)
        for seed in (5, 6):
            new_w, new_ids = _routes(t, e, seed=seed)
            weights.copy_(new_w)
            ids.copy_(new_ids)
            x.copy_(_x(t, h, seed=seed))
            graph.replay()
            assert torch.equal(out, op(x, weights, ids))


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 12),
    reason="FlashInfer runs here",
)
def test_auto_outside_flashinfer_families_raises():
    with pytest.raises(NotImplementedError, match="sm100 or sm120"):
        MXFP4Experts(*_weights(*SMALL), scales_layout="canonical")


@_SM12
@pytest.mark.parametrize("t", [1, 4, 4100])
def test_cute_provider_ep2(t):
    pytest.importorskip("cutlass")
    h, i, e = DSV4
    local = e // 2
    w = _weights(h, i, local)
    expert_map = torch.full((e,), -1, dtype=torch.int32, device="cuda")
    expert_map[local:] = torch.arange(local, dtype=torch.int32, device="cuda")
    x = _x(t, h)
    weights, ids = _routes(t, e, pad=True)
    op = MXFP4Experts(*_clones(w), swiglu_limit=10.0, provider="cute", scales_layout="canonical")
    assert op.provider == "cute" and op.layout == "up_first" and op.workspace_bytes == 0
    out = op(x, weights, ids, expert_map=expert_map)
    remote = (expert_map[ids.clamp(min=0).long()] < 0) | (ids < 0)
    assert not out[remote.all(-1)].any()
    kw = {"expert_map": expert_map, "swiglu_limit": 10.0}
    assert _truth_error(out, x, w, weights, ids, **kw) < TRUTH_BOUND
    assert torch.equal(out, op(x, weights, ids, expert_map=expert_map))  # deterministic


@_SM12
def test_cute_provider_graph_replay():
    pytest.importorskip("cutlass")
    h, i, e = DSV4
    op = MXFP4Experts(
        *_weights(h, i, e // 2), swiglu_limit=10.0, provider="cute", scales_layout="canonical"
    )
    t = 4
    x, out = _x(t, h), torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
    weights, ids = _routes(t, e // 2)
    op(x, weights, ids, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weights, ids, out=out)
    for seed in (5, 6):
        new_w, new_ids = _routes(t, e // 2, seed=seed)
        weights.copy_(new_w)
        ids.copy_(new_ids)
        graph.replay()
        assert torch.equal(out, op(x, weights, ids))
