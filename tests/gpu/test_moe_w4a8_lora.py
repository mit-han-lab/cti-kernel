# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DSV4 W4A8 routed experts with per-expert LoRA fused in the CuTe kernels (sm_120/121)."""

import pytest
import torch

from ctikernels import GraphWorkspace
from ctikernels.moe import MXFP4Experts
from ctikernels.moe.lora import MoELoRAWeights
from ctikernels.moe.torch._w4a8_reference import mxfp4_experts

from .test_moe_w4a8 import TRUTH_BOUND, _routes, _weights, _x

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="the fused W4A8 LoRA kernels run on sm_120/sm_121",
)

H, INTER, LOCAL, GLOBAL, LIMIT = 4096, 2048, 128, 256, 10.0


def _lora(rank, seed=4, experts=LOCAL, inter=INTER):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def a(*shape):  # Kaiming-uniform A, as PEFT initializes it
        return ((torch.rand(shape, generator=g, device="cuda") * 2 - 1) * shape[-1] ** -0.5).to(
            torch.bfloat16
        )

    def b(*shape):  # B trained away from zero
        return (torch.randn(shape, generator=g, device="cuda") * 0.5 * rank**-0.5).to(
            torch.bfloat16
        )

    return MoELoRAWeights.separate(
        a(experts, rank, H),
        a(experts, rank, H),
        b(experts, inter, rank),
        b(experts, inter, rank),
        a(experts, rank, inter),
        b(experts, H, rank),
    )


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("t,rank", [(1, 16), (5000, 16)])  # 5000: one call above 4096
def test_ep2_lora_matches_reference_and_truth(t, rank):
    w = _weights(H, INTER, LOCAL)
    x = _x(t, H)
    weights, ids = _routes(t, GLOBAL)
    ids[0, 0] = 3  # at least one local route on rank 0
    expert_map = torch.full((GLOBAL,), -1, dtype=torch.int32, device="cuda")
    expert_map[:LOCAL] = torch.arange(LOCAL, dtype=torch.int32, device="cuda")
    lora, scaling = _lora(rank), 2.0
    op = MXFP4Experts(*[t.clone() for t in w], swiglu_limit=LIMIT, scales_layout="canonical")
    out = op(x, weights, ids, expert_map=expert_map, lora=lora, scaling=scaling)
    kw = {"expert_map": expert_map, "swiglu_limit": LIMIT, "lora": lora.routed, "scaling": scaling}
    ref = mxfp4_experts(x, *w, weights, ids, **kw)
    truth = mxfp4_experts(x, *w, weights, ids, exact=True, **kw)
    base = mxfp4_experts(x, *w, weights, ids, expert_map=expert_map, swiglu_limit=LIMIT)
    assert _rel(out, ref) < 5e-3
    assert _rel(out, truth) < TRUTH_BOUND
    # The LoRA contribution itself, not hidden under the base error.
    assert _rel(out.float() - base.float(), ref.float() - base.float()) < 0.03
    again = op(x, weights, ids, expert_map=expert_map, lora=lora, scaling=scaling)
    assert torch.equal(out, again)  # deterministic


def test_lora_after_base_call_shares_weights():
    # One op serves both paths over one resident copy: the given tensors.
    w = _weights(H, INTER, LOCAL)
    given = [t.clone() for t in w]
    x = _x(4, H)
    weights, ids = _routes(4, LOCAL)
    lora, kw = _lora(16), {"swiglu_limit": LIMIT}
    op = MXFP4Experts(*given, **kw, scales_layout="canonical")
    base = op(x, weights, ids)
    out = op(x, weights, ids, lora=lora, scaling=2.0)
    assert op.layout == "up_first"
    assert all(torch.equal(a, b) for a, b in zip(op.canonical_weights(), w, strict=True))
    ref = mxfp4_experts(x, *w, weights, ids, lora=lora.routed, scaling=2.0, **kw)
    assert _rel(out, ref) < 5e-3
    assert torch.equal(base, op(x, weights, ids))


def test_graph_replay_shares_one_slot():
    # Two layers' ops in one GraphWorkspace: one arena, replays match eager calls.
    t, rank = 4, 16
    w = _weights(H, INTER, LOCAL)
    ops = [
        MXFP4Experts(*[t.clone() for t in w], swiglu_limit=LIMIT, scales_layout="canonical")
        for _ in range(2)
    ]
    lora, scaling = _lora(rank), torch.tensor([2.0], device="cuda")
    x, outs = _x(t, H), [torch.empty(t, H, dtype=torch.bfloat16, device="cuda") for _ in ops]
    weights, ids = _routes(t, LOCAL)
    with GraphWorkspace(max_rows=8) as group:
        for op, out in zip(ops, outs, strict=True):
            op(x, weights, ids, out=out, lora=lora, scaling=scaling)
        reserved = group.nbytes
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for op, out in zip(ops, outs, strict=True):
                op(x, weights, ids, out=out, lora=lora, scaling=scaling)
        assert group.nbytes == reserved
        for seed in (5, 6):
            new_w, new_ids = _routes(t, LOCAL, seed=seed)
            weights.copy_(new_w)
            ids.copy_(new_ids)
            x.copy_(_x(t, H, seed=seed))
            scaling.fill_(seed / 4)
            graph.replay()
            for op, out in zip(ops, outs, strict=True):
                assert torch.equal(out, op(x, weights, ids, lora=lora, scaling=scaling))


def test_reset_scratch_after_lost_tickets():
    # A memory-saver pause can lose the zeroed tickets; reset_scratch restores them.
    w = _weights(H, INTER, LOCAL)
    x = _x(4, H)
    weights, ids = _routes(4, LOCAL)
    op = MXFP4Experts(*w, swiglu_limit=LIMIT, scales_layout="canonical")
    lora = _lora(16)
    expected = op(x, weights, ids, lora=lora, scaling=2.0)
    for tickets in op._w4a8_lora._tickets.values():
        tickets.fill_(5)
    op.reset_scratch()
    assert torch.equal(op(x, weights, ids, lora=lora, scaling=2.0), expected)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="eager only"), torch.cuda.graph(graph):
        op.reset_scratch()


def test_zero_rows_and_unsupported_geometry():
    w = _weights(H, INTER, LOCAL)
    op = MXFP4Experts(*w, swiglu_limit=LIMIT, scales_layout="canonical")
    empty = torch.empty(0, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.empty(0, 6, dtype=torch.int32, device="cuda")
    out = op(empty, ids.float(), ids, lora=_lora(16))
    assert out.shape == (0, H)
    small = MXFP4Experts(*_weights(512, 256, 16), swiglu_limit=LIMIT, scales_layout="canonical")
    with pytest.raises(NotImplementedError, match="DSV4 Flash geometry"):
        x = _x(1, 512)
        weights, ids = _routes(1, 16)
        leaves = (torch.zeros(16, 8, 512), torch.zeros(16, 8, 512), torch.zeros(16, 256, 8))
        leaves = leaves + (
            torch.zeros(16, 256, 8),
            torch.zeros(16, 8, 256),
            torch.zeros(16, 512, 8),
        )
        small(x, weights, ids, lora=MoELoRAWeights.separate(*(t.cuda().bfloat16() for t in leaves)))


@pytest.mark.parametrize("t", [1, 300, 5000])
def test_cute_base_maps_global_ids(t):
    # The no-LoRA CuTe entry maps global ids in PREP: equal to the pre-mapped local ids.
    from ctikernels.moe import to_local_topk_ids
    from ctikernels.moe._common import ExpertWeights
    from ctikernels.moe.cute._w4a8 import Config, W4A8Experts

    w = _weights(H, INTER, LOCAL)
    x = _x(t, H)
    weights, ids = _routes(t, GLOBAL)
    expert_map = torch.full((GLOBAL,), -1, dtype=torch.int32, device="cuda")
    expert_map[LOCAL:] = torch.arange(LOCAL, dtype=torch.int32, device="cuda")
    op = W4A8Experts(ExpertWeights(*w), config=Config(up_first=True, combine="rows"))
    mapped = op(x, ids, weights, expert_map=expert_map)
    local = op(x, to_local_topk_ids(ids, expert_map), weights)
    assert torch.equal(mapped, local)
    ref = mxfp4_experts(x, *w, weights, ids, expert_map=expert_map, swiglu_limit=LIMIT)
    assert _rel(mapped, ref) < 5e-3


@pytest.mark.parametrize("local", [64, 256])  # EP4, EP1
@pytest.mark.parametrize("t", [1, 300])
def test_any_local_expert_count(local, t):
    w = _weights(H, INTER, local)
    x = _x(t, H)
    weights, ids = _routes(t, GLOBAL)
    ids[0, 0] = 1  # a local route on this rank
    expert_map = torch.full((GLOBAL,), -1, dtype=torch.int32, device="cuda")
    expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
    kw = {"expert_map": expert_map, "swiglu_limit": LIMIT}
    op = MXFP4Experts(
        *[t.clone() for t in w], swiglu_limit=LIMIT, provider="cute", scales_layout="canonical"
    )
    base = op(x, weights, ids, expert_map=expert_map)
    assert _rel(base, mxfp4_experts(x, *w, weights, ids, **kw)) < 5e-3
    lora = _lora(16, experts=local)
    out = op(x, weights, ids, expert_map=expert_map, lora=lora, scaling=2.0)
    ref = mxfp4_experts(x, *w, weights, ids, lora=lora.routed, scaling=2.0, **kw)
    assert _rel(out, ref) < 5e-3
    assert torch.equal(out, op(x, weights, ids, expert_map=expert_map, lora=lora, scaling=2.0))


@pytest.mark.parametrize("t", [1, 4, 300])
def test_expert_tp2_halves_sum_to_the_layer(t):
    # Expert TP2: each rank holds every expert with half of I (canonical [up | gate] rows,
    # w2 K columns); B13 rows and A_down columns split, A13 and B_down whole. The two partial
    # outputs, without and with LoRA, sum to the full layer.
    w = _weights(H, INTER, GLOBAL)
    x = _x(t, H)
    weights, ids = _routes(t, GLOBAL)
    lora, half = _lora(16, experts=GLOBAL), INTER // 2
    base, out = (torch.zeros(t, H, device="cuda") for _ in range(2))
    for r in range(2):
        rows = torch.cat([torch.arange(half) + r * half, torch.arange(half) + INTER + r * half])
        w13, s13 = (v[:, rows.cuda()].contiguous() for v in w[:2])
        w2 = w[2][:, :, r * half // 2 : (r + 1) * half // 2].contiguous()
        s2 = w[3][:, :, r * half // 32 : (r + 1) * half // 32].contiguous()
        cols = slice(r * half, (r + 1) * half)
        part = MoELoRAWeights.separate(
            lora.gate_a, lora.up_a, lora.gate_b[:, cols].contiguous(),
            lora.up_b[:, cols].contiguous(), lora.down_a[:, :, cols].contiguous(), lora.down_b,
        )  # fmt: skip
        op = MXFP4Experts(
            w13, s13, w2, s2, swiglu_limit=LIMIT, provider="cute", scales_layout="canonical"
        )
        base += op(x, weights, ids).float()
        out += op(x, weights, ids, lora=part, scaling=2.0).float()
    assert _rel(base, mxfp4_experts(x, *w, weights, ids, swiglu_limit=LIMIT)) < 5e-3
    ref = mxfp4_experts(x, *w, weights, ids, swiglu_limit=LIMIT, lora=lora.routed, scaling=2.0)
    assert _rel(out, ref) < 5e-3
