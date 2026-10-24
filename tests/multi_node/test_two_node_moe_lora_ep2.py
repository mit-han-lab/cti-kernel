# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""MoE + LoRA EP2 across two nodes: each rank runs its half of the experts and its shared-expert
I-window (``fused_experts_lora(..., shared_expert=)``); the two partials sum to the block. Run
the same module on both ranks; only the Store coordinates (partials travel through it)."""

import io
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ctikernels.moe import route, to_local_topk_ids
from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CTI_TWO_NODE_RANK") and os.environ.get("CTI_TWO_NODE_STORE")),
    reason="set CTI_TWO_NODE_RANK and CTI_TWO_NODE_STORE on both hosts",
)

E, H, INTER, SI, R, K = 256, 2048, 512, 512, 16, 8  # Qwen3.5-35B-A3B MoE block, LoRA r16


@pytest.fixture(scope="module")
def pair():
    rank = int(os.environ["CTI_TWO_NODE_RANK"])
    address, port = os.environ["CTI_TWO_NODE_STORE"].rsplit(":", 1)
    torch.cuda.set_device(0)
    store = dist.TCPStore(address, int(port), 2, rank == 0, timedelta(seconds=120))
    return rank, store


def layer():
    """The same weights on both ranks (seeded on the device)."""
    g = torch.Generator(device="cuda").manual_seed(1234)
    r = lambda *s: (torch.randn(*s, generator=g, device="cuda") * 0.02).bfloat16()
    base = {"router": r(E, H), "w13": r(E, 2 * INTER, H), "w2": r(E, H, INTER)}
    base.update(sgu=r(2 * SI, H), sd=r(H, SI), sg=r(1, H))
    routed = [r(E, R, H), r(E, R, H), r(E, INTER, R), r(E, INTER, R), r(E, R, INTER), r(E, H, R)]
    shared = [r(2, R, H), r(2, SI, R), r(R, SI), r(H, R)]
    return base, routed, shared


def shard(base, routed, shared, rank):
    """This rank's experts (half) and shared-expert I-window, with the matching LoRA slices."""
    experts, window = (
        slice(rank * E // 2, (rank + 1) * E // 2),
        slice(rank * SI // 2, (rank + 1) * SI // 2),
    )
    sgu = torch.cat((base["sgu"][:SI][window], base["sgu"][SI:][window])).contiguous()
    weights = (base["w13"][experts].contiguous(), base["w2"][experts].contiguous())
    lora = MoELoRAWeights.separate(*(f[experts].contiguous() for f in routed), shared[0],
                          shared[1][:, window].contiguous(), shared[2][:, window].contiguous(),
                          shared[3])  # fmt: skip
    expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
    expert_map[experts] = torch.arange(E // 2, dtype=torch.int32, device="cuda")
    return weights, lora, (sgu, base["sd"][:, window].contiguous(), base["sg"]), expert_map


def reference(x, rw, ids, base, routed, shared, scale):
    """FP32 MoE block with per-expert and shared LoRA."""
    xf, out = x.float(), torch.zeros(x.shape, device="cuda")
    ga, ua, gb, ub, da, db = (f.float() for f in routed)
    for k in range(K):
        for e in ids[:, k].unique().tolist():
            rows = (ids[:, k] == e).nonzero().flatten()
            xe = xf[rows]
            gu = xe @ base["w13"][e].float().T
            act = F.silu(gu[:, :INTER] + scale * xe @ ga[e].T @ gb[e].T) * (
                gu[:, INTER:] + scale * xe @ ua[e].T @ ub[e].T
            )
            y = act @ base["w2"][e].float().T + scale * act @ da[e].T @ db[e].T
            out[rows] += rw[rows, k, None] * y
    as13, bs13, as2, bs2 = (f.float() for f in shared)
    gu = xf @ base["sgu"].float().T
    act = F.silu(gu[:, :SI] + scale * xf @ as13[0].T @ bs13[0].T) * (
        gu[:, SI:] + scale * xf @ as13[1].T @ bs13[1].T
    )
    y = act @ base["sd"].float().T + scale * act @ as2.T @ bs2.T
    return out + torch.sigmoid(xf @ base["sg"].float().T) * y


def exchange(store, key, rank, tensor):
    """The peer's tensor (bytes through the Store)."""
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    store.set(f"{key}/{rank}", buffer.getvalue())
    return torch.load(io.BytesIO(store.get(f"{key}/{1 - rank}"))).cuda()


@pytest.mark.parametrize("t", [1, 4, 64])
def test_ep2_partials_sum_to_the_block(pair, t):
    """EP2 + TP2 shared expert across the two nodes: the summed partials match the FP32 block
    and the single-GPU layer; remote routes add nothing; both ranks agree bit for bit."""
    rank, store = pair
    base, routed, shared = layer()
    (w13, w2), lora, shared_expert, expert_map = shard(base, routed, shared, rank)
    x = (
        torch.randn(t, H, generator=torch.Generator(device="cuda").manual_seed(t), device="cuda")
    ).bfloat16()
    rw, ids = route(x, base["router"], K, renormalize=True)
    local = to_local_topk_ids(ids.int(), expert_map)
    partial = fused_experts_lora(
        x, w13, w2, rw, local, lora, scaling=0.5, shared_expert=shared_expert
    )
    peer = exchange(store, f"moe-lora-ep2/{t}/partial", rank, partial)
    first, second = (partial, peer) if rank == 0 else (peer, partial)
    block = (first.float() + second.float()).bfloat16()  # the exit all-reduce: rank 0 + rank 1
    expected = reference(x, rw, ids.long(), base, routed, shared, 0.5)
    single = fused_experts_lora(
        x, base["w13"], base["w2"], rw, ids.int(), MoELoRAWeights.separate(*routed, *shared),
        scaling=0.5, shared_expert=(base["sgu"], base["sd"], base["sg"]),
    )  # fmt: skip
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
    assert rel(block, expected) <= 1e-2
    assert rel(block, single) <= 1e-2
    other = exchange(store, f"moe-lora-ep2/{t}/block", rank, block)
    assert torch.equal(block.view(torch.int16), other.view(torch.int16))
