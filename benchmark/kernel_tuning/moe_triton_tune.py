# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline FC1/FC2 tile sweep using random top-8 routes and CUDA graph replay.

Run with the package installed, or PYTHONPATH=python. The default shortlist
covers every requested tile axis in at most 32 candidates per projection.
Compilation is reused across M. --full-grid explores the full Cartesian grid.
Each timed graph cycles --slots independent routings over weight copies totalling
at least 4x L2, so small M is timed with DRAM-resident weights as in a model.
--global-experts/--rank time one EP rank: routes over all experts, -1 map entries.
JSON uses device -> "E,I" -> up/down -> M -> config; convert the shape key to a
Python tuple when adding the result to ctikernels.moe._configs.TUNED.
"""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path


def expert_mapping(experts, global_experts, rank, device="cpu"):
    """Contiguous EP ownership; local slots are always 0..experts-1."""
    import torch

    if experts < 1 or global_experts < experts or global_experts % experts:
        raise ValueError("global experts must be a positive multiple of local experts")
    if not 0 <= rank < global_experts // experts:
        raise ValueError("rank must be in the EP group")
    mapping = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
    mapping[rank * experts : (rank + 1) * experts] = torch.arange(
        experts, dtype=torch.int32, device=device
    )
    return mapping


def candidates(shared_limit, full=False):
    keys = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    )
    grid = itertools.product(
        (16, 32, 64, 128), (32, 64, 128), (64, 128), (1, 16), (4, 8), (2, 3, 4)
    )
    kept = []
    for bm, bn, bk, gm, warps, stages in grid:
        # Operand staging plus a conservative allowance for layouts/accumulators.
        if 2 * (bm + bn) * bk * stages + 8192 > shared_limit:
            continue
        if not full:
            # Up to 6 configurations per BM: rotate BN, BK, GM, warps and stages
            # instead of recompiling every combination of the same tile.
            n = (32, 64, 128).index(bn)
            if (gm == 16) != (n != 0) or warps != (8 if bm == 128 or bn == 128 else 4):
                continue
            if stages != (2 + (n + (bk == 128)) % 3):
                continue
        kept.append(dict(zip(keys, (bm, bn, bk, gm, warps, stages), strict=True)))
    return kept


def graph_time(calls, torch):
    for call in calls:
        call()  # compile and warm up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (5 * len(calls)))
    return statistics.median(samples)


def weight_layers(l2_bytes, layer_bytes):
    """Independent weight copies cycled per call, so that weights stream from DRAM as in a model."""
    return max(1, -(-4 * l2_bytes // layer_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--inter", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--global-experts", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--slots", type=int, default=8, help="independent random routings per timed graph"
    )
    parser.add_argument("--layers", type=int, help="weight copies (default: 4x L2 in total)")
    parser.add_argument("--m", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument(
        "--minutes",
        type=float,
        default=24,
        help="soft time budget; finish the current candidate before stopping",
    )
    args = parser.parse_args()
    rows = sorted({int(m) for m in args.m.split(",")})
    args.global_experts = args.global_experts if args.global_experts is not None else args.experts
    if args.inter <= 0 or args.hidden <= 0 or not rows or rows[0] < 1:
        parser.error("inter, hidden and M must be positive")
    if not 1 <= args.top_k <= args.global_experts:
        parser.error("top-k must be in [1, global-experts]")
    try:
        mapping = expert_mapping(args.experts, args.global_experts, args.rank)
    except ValueError as error:
        parser.error(str(error))

    import torch
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    from ctikernels.moe._block import hardware
    from ctikernels.moe._configs import lookup
    from ctikernels.moe_impl import fused_moe_kernel_triton, moe_align_block_size

    torch.manual_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    configs = candidates(props.shared_memory_per_block_optin, args.full_grid)
    opts = {"device": device, "dtype": torch.bfloat16}
    l2 = getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0)) or 128 << 20
    layers = args.layers or weight_layers(l2, args.experts * 3 * args.hidden * args.inter * 2)
    # Each timed call reads its own layer and routing: small M must not run with L2-hot weights.
    w1s = [
        torch.randn((args.experts, 2 * args.inter, args.hidden), **opts).mul_(0.02)
        for _ in range(layers)
    ]
    w2s = [
        torch.randn((args.experts, args.hidden, args.inter), **opts).mul_(0.02)
        for _ in range(layers)
    ]
    mapping = mapping.to(device)
    result = {"up": {}, "down": {}}
    failed = set()
    deadline = time.monotonic() + args.minutes * 60
    # M-major order yields both projections for each completed row bucket.
    for m in rows:
        torch.manual_seed(args.seed + 7919 * m)
        x = torch.randn((m, args.hidden), **opts)
        activated = torch.randn((m * args.top_k, args.inter), **opts)
        routes = []
        for _ in range(args.slots):
            logits = torch.randn((m, args.global_experts), device=device, dtype=torch.float32)
            weights, ids = logits.softmax(-1).topk(args.top_k, dim=-1)
            routes.append((ids.to(torch.int32), weights / weights.sum(-1, keepdim=True)))
        fc1 = torch.empty((m, args.top_k, 2 * args.inter), **opts)
        fc2 = torch.empty((m, args.top_k, args.hidden), **opts)
        aligned = {
            bm: [
                moe_align_block_size(ids, bm, args.experts, expert_map=mapping) for ids, _ in routes
            ]
            for bm in {c["BLOCK_SIZE_M"] for c in configs}
        }
        for phase in ("up", "down"):
            best, best_ms = None, float("inf")
            # The installed table's entry competes too, and its time is reported.
            current = lookup(hardware(device), args.experts, args.inter, m, phase)
            if current and current["BLOCK_SIZE_M"] not in aligned:
                bm = current["BLOCK_SIZE_M"]
                aligned[bm] = [
                    moe_align_block_size(ids, bm, args.experts, expert_map=mapping)
                    for ids, _ in routes
                ]
            current_ms = None
            for config in ([current] if current else []) + configs:
                signature = (phase, tuple(config.items()))
                if signature in failed or time.monotonic() >= deadline:
                    continue

                def call(
                    slot,
                    config=config,
                    phase=phase,
                    routes=routes,
                    aligned=aligned,
                    up=(x, fc1),
                    down=(activated, fc2),
                ):
                    ids, weights = routes[slot]
                    layer = slot % layers
                    a, c = up if phase == "up" else down
                    b = w1s[layer] if phase == "up" else w2s[layer]
                    fused_moe_kernel_triton(
                        a,
                        b,
                        c,
                        weights,
                        ids,
                        *aligned[config["BLOCK_SIZE_M"]][slot],
                        phase == "down",
                        args.top_k if phase == "up" else 1,
                        config,
                        torch.bfloat16,
                    )

                calls = [lambda slot=slot, call=call: call(slot) for slot in range(args.slots)]
                try:
                    elapsed = graph_time(calls, torch)
                except (CompilationError, OutOfResources, RuntimeError) as error:
                    # Compile/resource failures exclude this candidate; device execution
                    # failures must stop the sweep instead of publishing suspect timings.
                    if isinstance(error, RuntimeError) and not any(
                        word in str(error).lower() for word in ("compile", "ptxas", "resource")
                    ):
                        raise
                    failed.add(signature)
                    continue
                if config is current and current_ms is None:
                    current_ms = elapsed
                if elapsed < best_ms:
                    best, best_ms = config, elapsed
            if best is not None:
                result[phase][m] = best
                print(
                    f"M={m} {phase}: best {best_ms * 1000:.1f} us, installed "
                    f"{'n/a' if current_ms is None else f'{current_ms * 1000:.1f} us'} "
                    f"{json.dumps(best, sort_keys=True)}",
                    flush=True,
                )
        if time.monotonic() >= deadline:
            break
    table = {hardware(device) or "unknown": {f"{args.experts},{args.inter}": result}}
    output = json.dumps(table, indent=2) + "\n"
    # The table is printed first: it survives an unwritable output directory.
    print(output, end="", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    if any(len(result[p]) != len(rows) for p in result):
        raise SystemExit(
            "Time budget exhausted or no valid tile: table contains completed buckets only"
        )


if __name__ == "__main__":
    main()
