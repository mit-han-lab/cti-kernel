# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired CUDA-event timings of frozen MoE forward and input backward.

Run: python -m benchmark.kernel_tuning.moe_dx_paired --ep 2 --graph
SOL is an optimistic floor: BF16 expert weight traffic (one read in forward,
 two in forward+backward), external activation I/O, and expert MMA FLOPs.
Routing, layout, intermediate traffic and recomputation FLOPs are excluded.
The shared shard counts as one expert hit by every token. Times include host
launch gaps and the fallback's single counts.tolist() synchronization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark._support import distribution
from ctikernels.moe import moe_block

CEILINGS = {"5090": (1681.0, 247.4), "PRO 6000": (1533.0, 466.1), "GB10": (244.6, 124.0)}
DEFAULT_TOKENS = (1, 2, 4, 8, 16, 64, 256, 1024, 4096, 8192)


def sol_floor(
    device_name,
    *,
    tokens,
    hidden,
    intermediate,
    shared_intermediate,
    local_slots,
    distinct_local,
    backward=True,
):
    """Return microseconds; GB/s and TFLOPS are decimal, weights are BF16.

    Forward+backward: 12*slots*H*I FLOPs + 12*T*H*Is, and twice the
    BF16 bytes of 3*H*(distinct*I+Is) weights, plus x/out/dy/dx I/O.
    Forward: half those FLOPs/bytes. Recompute's extra 4*slots*H*I
    (+ shared) is deliberately absent from the reported ideal floor.
    """
    ceiling = next((v for name, v in CEILINGS.items() if name in device_name), None)
    if ceiling is None:
        return None
    bandwidth, mma = ceiling
    passes = 2 if backward else 1
    weight_bytes = 2 * 3 * hidden * (distinct_local * intermediate + shared_intermediate)
    activation_bytes = passes * 2 * tokens * hidden * 2
    byte_count = passes * weight_bytes + activation_bytes
    flops = passes * 6 * hidden * (local_slots * intermediate + tokens * shared_intermediate)
    return max(byte_count / (bandwidth * 1e9), flops / (mma * 1e12)) * 1e6


def torch_fallback(x, r, sg, w13, w2, s13, s2, *, top_k, expert_map, routing=None):
    """Frozen BF16 expert recipe with FP32 weights and one host counts readback."""
    logits = F.linear(x, r).float()
    if routing is None:
        sel = torch.sort(logits, descending=True, stable=True, dim=1).indices[:, :top_k]
    else:
        sel = routing.long()
    probabilities = torch.softmax(logits, dim=1).gather(1, sel)
    weights = probabilities / probabilities.sum(dim=1, keepdim=True)
    e, ni, nis = w13.shape[0], w2.shape[2], s2.shape[1]
    local = expert_map[sel].long()
    flat = torch.where(local >= 0, local, e).flatten()
    order = flat.argsort(stable=True)
    # Fixed-size counting avoids bincount's device-maximum readback on CUDA.
    counts_device = torch.zeros(e + 1, dtype=torch.int64, device=x.device)
    counts_device.scatter_add_(0, flat, torch.ones_like(flat))
    counts = counts_device.tolist()  # The only host sync for grouping.
    out = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts[:e]):
        if count:
            slots = order[start : start + count]
            idx, j = slots // top_k, slots % top_k
            gu = F.linear(x[idx], w13[expert])
            y = F.linear(F.silu(gu[:, :ni]) * gu[:, ni:], w2[expert])
            y = (y.float() * weights[idx, j, None]).to(x.dtype)
            out = out.index_add(0, idx, y)
        start += count
    gu = F.linear(x, s13)
    shared = F.linear(F.silu(gu[:, :nis]) * gu[:, nis:], s2)
    gamma = torch.sigmoid(F.linear(x.float(), sg.float()))
    return (out.float() + gamma * shared.float()).to(x.dtype)


def make_inputs(t, ep, seed):
    """Rank zero's contiguous placement; gate/up shared rows are already sliced."""
    h, e, ni, nis = 2048, 256, 512, 512 // ep
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rn(shape, scale):
        return (torch.randn(shape, generator=gen, device="cuda") * scale).bfloat16()

    x = rn((t, h), 1).requires_grad_()
    dy = rn((t, h), 1)
    weights = (
        rn((e, h), 2 / h**0.5),
        rn((1, h), 1 / h**0.5),
        rn((e // ep, 2 * ni, h), 1 / h**0.5),
        rn((e // ep, h, ni), 1 / ni**0.5),
        rn((2 * nis, h), 1 / h**0.5),
        rn((h, nis), 1 / 512**0.5),
    )
    mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    mapping[: e // ep] = torch.arange(e // ep, device="cuda", dtype=torch.int32)
    return x, dy, weights, mapping


def _timed(fn):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000, result


def benchmark_row(t, ep, *, repeats, warmup, graph_mode, save_activations, seed):
    x, dy, weights, mapping = make_inputs(t, ep, seed)
    with torch.no_grad():
        _, record = moe_block(x, *weights, top_k=8, expert_map=mapping)
    kw = {"top_k": 8, "expert_map": mapping, "routing": record}
    local = mapping[record.long()]
    slots_local = (local >= 0).sum().item()
    distinct = local[local >= 0].unique().numel()
    name = torch.cuda.get_device_name(x.device)
    rows = []
    for backward in (False, True):

        def op(backward=backward, x=x):
            with torch.set_grad_enabled(backward):
                out, _ = moe_block(x, *weights, **kw, save_activations=save_activations)
                return (out, torch.autograd.grad(out, x, dy)[0]) if backward else (out, None)

        def fallback(backward=backward):
            with torch.set_grad_enabled(backward):
                out = torch_fallback(x, *weights, **kw)
                return (out, torch.autograd.grad(out, x, dy)[0]) if backward else (out, None)

        for _ in range(warmup):
            op()
            fallback()
        eager = op()
        base = fallback()
        dx_diff = (eager[1].float() - base[1].float()).abs().max().item() if backward else None
        graph = None
        if graph_mode:
            # A fresh leaf: the eager calls above left AccumulateGrad nodes on the default stream.
            xg = x.detach().clone().requires_grad_()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    op(x=xg)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op(x=xg)
            graph.replay()
            assert torch.equal(captured[0], eager[0])
            if backward:
                assert torch.equal(captured[1], eager[1])
        a, b, g = [], [], []
        # ABAB throughout (with the op's graph replay as a third column); the fallback is eager.
        for _ in range(repeats):
            a.append(_timed(op)[0])
            b.append(_timed(fallback)[0])
            if graph is not None:
                g.append(_timed(graph.replay)[0])
        da, db = distribution(a), distribution(b)
        dg = distribution(g) if g else None
        floor = sol_floor(
            name,
            tokens=t,
            hidden=2048,
            intermediate=512,
            shared_intermediate=512 // ep,
            local_slots=slots_local,
            distinct_local=distinct,
            backward=backward,
        )
        row = {
            "device": name,
            "T": t,
            "ep": ep,
            "mode": "fwd+bwd" if backward else "fwd",
            "graph": graph_mode,
            "fallback_graph": False,
            "save_activations": save_activations,
            "op_us": da["p50"],
            "op_iqr_us": da["iqr"],
            "fallback_us": db["p50"],
            "fallback_iqr_us": db["iqr"],
            "speedup": db["p50"] / da["p50"],
            "sol_us": floor,
            "sol_percent": 100 * floor / da["p50"] if floor is not None else None,
            "op_graph_us": dg["p50"] if dg else None,
            "op_graph_iqr_us": dg["iqr"] if dg else None,
            "graph_sol_percent": 100 * floor / dg["p50"] if dg and floor is not None else None,
            "dx_max_diff": dx_diff,
            "slots_local": slots_local,
            "distinct_local": distinct,
            "pairs": repeats,
        }
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tokens = [int(t) for t in args.tokens.split(",")]
    if min(tokens) < 1 or args.repeats < 1 or args.warmup < 1:
        parser.error("tokens, repeats and warmup must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for timing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rows = []
    for t in tokens:
        rows.extend(
            benchmark_row(
                t,
                args.ep,
                repeats=args.repeats,
                warmup=args.warmup,
                graph_mode=args.graph,
                save_activations=args.save_activations,
                seed=args.seed + t,
            )
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
