# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DSV4 Flash routed MoE (W4A8, + per-expert LoRA) as real EP2 across two hosts.

Run the same selection on both hosts (one GPU each) with CTI_TWO_NODE_RANK and
CTI_TWO_NODE_STORE set (see README). Every rank builds the same global weights and keeps its
128 experts; the exit all-reduce (NCCL) sums the two partials. Each rank checks the result
against the single-node reference over all 256 experts, and both ranks must hold identical bytes.
"""

import hashlib
import json
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CTI_TWO_NODE_RANK") and os.environ.get("CTI_TWO_NODE_STORE")),
    reason="set CTI_TWO_NODE_RANK and CTI_TWO_NODE_STORE on both hosts",
)
EXPERTS, LOCAL, H, INTER, TOP_K = 256, 128, 4096, 2048, 6


@pytest.fixture(scope="module")
def group():
    rank = int(os.environ["CTI_TWO_NODE_RANK"])
    address, port = os.environ["CTI_TWO_NODE_STORE"].rsplit(":", 1)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("the CuTe W4A8 experts run on sm_120/sm_121")
    store = dist.TCPStore(address, int(port), 2, rank == 0, timedelta(seconds=120))
    dist.init_process_group("nccl", store=store, rank=rank, world_size=2)
    yield rank, store
    dist.destroy_process_group()


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def ints(lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=torch.int32)

    return (
        ints(0, 256, EXPERTS, 2 * INTER, H // 2).to(torch.uint8),
        ints(119, 123, EXPERTS, 2 * INTER, H // 32).to(torch.uint8),
        ints(0, 256, EXPERTS, H, INTER // 2).to(torch.uint8),
        ints(120, 124, EXPERTS, H, INTER // 32).to(torch.uint8),
    )


def _lora(rank, seed=4):
    from ctikernels.moe.lora import MoELoRAWeights

    g = torch.Generator(device="cuda").manual_seed(seed)

    def a(*shape):
        return ((torch.rand(shape, generator=g, device="cuda") * 2 - 1) * shape[-1] ** -0.5).to(
            torch.bfloat16
        )

    def b(*shape):
        return (torch.randn(shape, generator=g, device="cuda") * 0.5 * rank**-0.5).to(
            torch.bfloat16
        )

    return MoELoRAWeights.separate(
        a(EXPERTS, rank, H), a(EXPERTS, rank, H), b(EXPERTS, INTER, rank), b(EXPERTS, INTER, rank),
        a(EXPERTS, rank, INTER), b(EXPERTS, H, rank),
    )  # fmt: skip


def _routes(t, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g, device="cuda")[:TOP_K] for _ in range(t)]
    )
    w = torch.rand(t, TOP_K, generator=g, device="cuda")
    return (w / w.sum(-1, keepdim=True)).contiguous(), ids.to(torch.int32).contiguous()


@pytest.mark.parametrize("t,lora_rank", [(1, 16), (300, 16), (300, 0)])
def test_ep2_matches_single_node_and_ranks_agree(group, t, lora_rank):
    from ctikernels.moe import MXFP4Experts
    from ctikernels.moe.lora import MoELoRAWeights
    from ctikernels.moe.torch._w4a8_reference import mxfp4_experts

    rank, store = group
    lo, hi = rank * LOCAL, (rank + 1) * LOCAL
    full = _weights()
    expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
    expert_map[lo:hi] = torch.arange(LOCAL, dtype=torch.int32, device="cuda")
    # The op takes over (rewrites in place) its tensors: give it copies of this rank's slice.
    op = MXFP4Experts(
        *(w[lo:hi].clone() for w in full),
        swiglu_limit=10.0,
        provider="cute",
        scales_layout="canonical",
    )
    weights, ids = _routes(t)
    x = torch.randn(t, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    x = x.to(torch.bfloat16)
    kw = {"swiglu_limit": 10.0}
    if lora_rank:
        lora_full = _lora(lora_rank)
        lora = MoELoRAWeights.separate(*(f[lo:hi].contiguous() for f in lora_full.routed))
        out = op(x, weights, ids, expert_map=expert_map, lora=lora, scaling=2.0)
        kw.update(lora=lora_full.routed, scaling=2.0)
    else:
        out = op(x, weights, ids, expert_map=expert_map)
    dist.all_reduce(out)
    ref = mxfp4_experts(x, *full, weights, ids, **kw)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    digest = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    key = f"dsv4-ep2/{t}/{lora_rank}"
    store.set(f"{key}/{rank}", json.dumps({"rel": rel, "hash": digest}))
    peer = json.loads(store.get(f"{key}/{1 - rank}"))
    assert rel < 5e-3 and peer["rel"] < 5e-3, (rel, peer["rel"])
    assert digest == peer["hash"], "the ranks' all-reduced outputs differ"
