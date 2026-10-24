# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""GDN decode under TP2 across two nodes: column-parallel in_proj -> decode on the rank's head
shard -> row-parallel out_proj -> all-reduce. Run the same module on both ranks (see README)."""

import hashlib
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from ctikernels.collective._two_node.proxy import barrier
from ctikernels.collective.two_node import TwoNodeAllReduce, support
from ctikernels.gdn import GatedDeltaNetDecode

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CTI_TWO_NODE_RANK") and os.environ.get("CTI_TWO_NODE_STORE")),
    reason="set CTI_TWO_NODE_RANK and CTI_TWO_NODE_STORE on both hosts",
)
HK, HV, HIDDEN = 16, 32, 4096  # Qwen3.5-9B; each rank holds (8, 16)
CONV = (2 * HK + HV) * 128


def exchange(ctx, key, value):
    ctx.store.set(f"gdn/{key}/{ctx.rank}", json.dumps(value))
    return json.loads(ctx.store.get(f"gdn/{key}/{1 - ctx.rank}"))


def agree(ctx, key, ok, tensor=None):
    """Both ranks pass, and with `tensor`, hold the same bits."""
    local = {"ok": bool(ok)}
    if tensor is not None:
        payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        local["hash"] = hashlib.sha256(payload).hexdigest()
    peer = exchange(ctx, key, local)
    assert local["ok"] and peer["ok"], key
    assert local.get("hash") == peer.get("hash"), f"ranks differ: {key}"


def shard_columns(rank):
    """Projected columns (q | k | v | z | b | a) of this rank's heads."""
    hk, hv = HK // 2, HV // 2
    q = torch.arange(rank * hk * 128, (rank + 1) * hk * 128)
    k = q + HK * 128
    v = torch.arange(rank * hv * 128, (rank + 1) * hv * 128) + 2 * HK * 128
    z = v - 2 * HK * 128 + CONV
    b = torch.arange(rank * hv, (rank + 1) * hv) + CONV + HV * 128
    return torch.cat([q, k, v, z, b, b + HV])


class Model:
    """Seeded full-geometry weights and pools, the same on both ranks."""

    def __init__(self, batch, seed):
        gen = torch.Generator().manual_seed(seed)
        width = CONV + HV * 128 + 2 * HV
        self.slots = 2 * batch + 1
        self.x = [torch.randn(batch, HIDDEN, generator=gen).bfloat16().cuda() for _ in range(4)]
        self.w_in = (0.02 * torch.randn(width, HIDDEN, generator=gen)).bfloat16().cuda()
        self.w_out = (0.02 * torch.randn(HIDDEN, HV * 128, generator=gen)).bfloat16().cuda()
        self.conv_weight = (0.5 * torch.randn(CONV, 4, generator=gen)).bfloat16().cuda()
        self.A_log = (torch.rand(HV, generator=gen) * 2.8 - 0.7).cuda()
        self.dt_bias = (4 * torch.rand(HV, generator=gen) - 2).cuda()
        self.norm_weight = (1 + 0.1 * torch.randn(128, generator=gen)).cuda()
        self.state = (0.2 * torch.randn(self.slots, HV, 128, 128, generator=gen)).cuda()
        self.conv_state = torch.randn(self.slots, 3, CONV, generator=gen).bfloat16().cuda()
        order = torch.randperm(self.slots - 1, generator=gen)[:batch].to(torch.int32)
        order[batch // 2] = -1 if batch > 2 else order[batch // 2]  # one padding row
        self.slot_indices = order.cuda()

    def shard(self, rank):
        """This rank's (8, 16) slice: weights, pools and the out_proj columns."""
        hv = HV // 2
        cols = shard_columns(rank).cuda()
        conv = cols[: (2 * HK // 2 + hv) * 128]
        heads = slice(rank * hv, (rank + 1) * hv)
        return SimpleNamespace(
            cols=cols,
            w_in=self.w_in[cols].contiguous(),
            w_out=self.w_out[:, heads.start * 128 : heads.stop * 128].contiguous(),
            weights=(
                self.conv_weight[conv].contiguous(),
                self.A_log[heads].contiguous(),
                self.dt_bias[heads].contiguous(),
                self.norm_weight,
            ),
            state=self.state[:, heads].contiguous(),
            conv_state=self.conv_state[:, :, conv].contiguous(),
            heads=heads,
            conv=conv,
        )

    def weights(self):
        return self.conv_weight, self.A_log, self.dt_bias, self.norm_weight


@pytest.fixture(scope="module")
def pair():
    rank = int(os.environ["CTI_TWO_NODE_RANK"])
    address, port = os.environ["CTI_TWO_NODE_STORE"].rsplit(":", 1)
    torch.cuda.set_device(0)
    store = dist.TCPStore(address, int(port), 2, rank == 0, timedelta(seconds=120))
    ctx = SimpleNamespace(rank=rank, store=store)
    reason = support()
    peer = exchange(ctx, "support", reason)
    if reason or peer:
        pytest.skip(reason or peer)
    ctx.op = TwoNodeAllReduce(rank=rank, store=store, prefix="gdn/provider")
    yield ctx
    torch.cuda.synchronize()
    barrier(store, "gdn/finished")
    ctx.op.close()
    # Rank 0 hosts the store: it may exit only after rank 1 has left the barrier.
    if rank == 1:
        store.set("gdn/released", "1")
    else:
        store.get("gdn/released")


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_shard_decode_matches_the_full_geometry(pair, batch):
    """Batch-invariant decode on the (8, 16) shard is bitwise the full (16, 32) decode's slice:
    output, state and conv history, over four steps."""
    model = Model(batch, seed=batch)
    mine = model.shard(pair.rank)
    full = GatedDeltaNetDecode(num_k_heads=HK, num_v_heads=HV, device="cuda", batch_invariant=True)
    local = GatedDeltaNetDecode(
        num_k_heads=HK // 2, num_v_heads=HV // 2, device="cuda", batch_invariant=True
    )
    for step, x in enumerate(model.x):
        projected = x @ model.w_in.t()
        y_full = full(
            projected, *model.weights(), model.slot_indices, model.state, model.conv_state
        )
        y = local(
            projected[:, mine.cols], *mine.weights, model.slot_indices, mine.state, mine.conv_state
        )
        heads = slice(mine.heads.start * 128, mine.heads.stop * 128)
        agree(pair, f"shard/{batch}/{step}/y", torch.equal(y, y_full[:, heads]))
    ok = torch.equal(mine.state, model.state[:, mine.heads]) and torch.equal(
        mine.conv_state, model.conv_state[:, :, mine.conv]
    )
    agree(pair, f"shard/{batch}/pools", ok)


def tp2_chain(pair, local, model, mine, x, out):
    projected = x @ mine.w_in.t()
    y = local(projected, *mine.weights, model.slot_indices, mine.state, mine.conv_state)
    return pair.op(y @ mine.w_out.t(), out)


@pytest.mark.parametrize("batch", [1, 4])
def test_tp2_chain_matches_the_single_gpu_chain(pair, batch):
    """The all-reduced TP2 chain is identical on both ranks and within BF16 GEMM rounding of the
    single-GPU chain (FP32 out_proj)."""
    model, reference = Model(batch, seed=100 + batch), Model(batch, seed=100 + batch)
    mine = model.shard(pair.rank)
    local = GatedDeltaNetDecode(num_k_heads=HK // 2, num_v_heads=HV // 2, device="cuda")
    full = GatedDeltaNetDecode(num_k_heads=HK, num_v_heads=HV, device="cuda")
    for step, x in enumerate(model.x):
        out = tp2_chain(pair, local, model, mine, x, None)
        y = full(
            x @ reference.w_in.t(),
            *reference.weights(),
            reference.slot_indices,
            reference.state,
            reference.conv_state,
        )
        expected = y.float() @ reference.w_out.float().t()
        error = (out.float() - expected).abs().max().item()
        scale = expected.abs().max().item()
        agree(pair, f"chain/{batch}/{step}", error <= 2e-2 * scale, out)


def test_tp2_chain_graph_replay_equals_eager(pair):
    """The chain captured in one CUDA graph (decode, GEMMs and the all-reduce) replays bitwise
    as the eager chain, on both ranks."""
    batch = 2
    graphed, eager = Model(batch, seed=7), Model(batch, seed=7)
    g, e = graphed.shard(pair.rank), eager.shard(pair.rank)
    local = GatedDeltaNetDecode(num_k_heads=HK // 2, num_v_heads=HV // 2, device="cuda")
    pair.op.prepare([batch * HIDDEN * 2])
    x = torch.empty_like(graphed.x[0])
    out = torch.empty(batch, HIDDEN, dtype=torch.bfloat16, device="cuda")
    warm = Model(batch, seed=8)
    w = warm.shard(pair.rank)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tp2_chain(pair, local, warm, w, warm.x[0], out)  # warm up the op and provider
    stream.synchronize()
    barrier(pair.store, "gdn/graph/capture-ready")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        tp2_chain(pair, local, graphed, g, x, out)
    for step in range(4):
        with torch.cuda.stream(stream):
            x.copy_(graphed.x[step])
            graph.replay()
        stream.synchronize()
        expected = tp2_chain(pair, local, eager, e, eager.x[step], None)
        agree(pair, f"graph/{step}", torch.equal(out, expected), out)
    agree(
        pair,
        "graph/pools",
        torch.equal(g.state, e.state) and torch.equal(g.conv_state, e.conv_state),
    )
