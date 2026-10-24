# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Interleaved CUDA-graph grouped MoE sweep; one JSON record per M/preset.

python -m benchmark.kernel_tuning.block_fp8_moe_cute --batches 1,2,3,4,64,300 --check
Ratios are Triton/CuTe (larger is faster). Eight independent routings per graph.
"""

import argparse
import json
import statistics

import torch

from ctikernels._moe_block_fp8_cute_table import PRESETS, valid_presets
from ctikernels.moe_block_fp8 import BlockFP8MoE

from .block_fp8 import EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, _random_weight


def cases_for(m):
    cases = []
    for _ in range(8):
        weights, ids = torch.randn(m, EXPERTS, device="cuda").softmax(-1).topk(TOP_K, dim=-1)
        cases.append(
            (
                torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
                (weights / weights.sum(-1, keepdim=True)).contiguous(),
                ids.to(torch.int32),
            )
        )
    return cases


def capture(op, cases):
    for case in cases:
        op(*case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            op(*case)
    return graph


def paired_times(graphs, iterations, replays):
    for _ in range(3):
        for graph in graphs:
            graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = [[], []]
    for round_id in range(iterations):
        for index in (0, 1) if round_id % 2 == 0 else (1, 0):
            start.record()
            for _ in range(replays):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / (8 * replays))
    return tuple(statistics.median(values) for values in samples)


def breakdown(op, cases, baseline, iterations, replays):
    """Isolated graph costs, eight routings; inputs/workspaces stay live per graph.

    Quantize and align include their full existing launch chains. GEMM/reduce
    graphs use precomputed inputs, so their warm-cache sum need not equal forward.
    """
    from ctikernels._moe_block_fp8_cute import _functions, masked_reduce
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.moe_impl import moe_align_block_size

    config = op.config_for(cases[0][0].shape[0])
    fc1, fc2 = _functions(op, config)
    launches = {name: [] for name in ("quantize", "align", "fc1", "fc2", "reduce")}
    for x, weights, ids in cases:
        q, scales = quantize_per_token_group_fp8(x)
        sorted_ids, experts, padded = moe_align_block_size(ids, config.bm, op.e)
        h = torch.empty((sorted_ids.numel(), op.i), device=x.device, dtype=x.dtype)
        down = torch.empty((ids.numel(), op.h), device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        first = (
            q,
            scales,
            op.w13.view(-1, op.h),
            op.s13,
            sorted_ids,
            experts,
            padded,
            weights.view(-1),
            h,
        )
        second = (
            h,
            op.s2.view(-1, op.i // 128),
            op.w2.view(-1, op.i),
            op.s2,
            sorted_ids,
            experts,
            padded,
            weights.view(-1),
            down,
        )
        fc1(*first)
        fc2(*second)
        launches["quantize"].append(lambda x=x: quantize_per_token_group_fp8(x))
        launches["align"].append(lambda ids=ids: moe_align_block_size(ids, config.bm, op.e))
        launches["fc1"].append(lambda args=first: fc1(*args))
        launches["fc2"].append(lambda args=second: fc2(*args))
        launches["reduce"].append(
            lambda d=down, ids=ids, out=out, w=weights: masked_reduce(d, ids, out, w)
        )
    result = {}
    for name, functions in launches.items():
        graph = capture(lambda function: function(), [(f,) for f in functions])
        _, result[name] = paired_times((baseline, graph), iterations, replays)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,2,3,4,64,300,1024,4096")
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="isolated per-stage graph costs; select one grouped preset",
    )
    parser.add_argument("--check-rtol", type=float, default=0.03)
    args = parser.parse_args()
    if min(args.iterations, args.replays) < 1:
        parser.error("iterations and replays must be positive")
    names = args.presets.split(",")
    if any(name not in PRESETS for name in names):
        parser.error("unknown preset")
    if args.breakdown and (len(names) != 1 or PRESETS[names[0]].path != "grouped"):
        parser.error("--breakdown requires --presets with one grouped preset")
    torch.manual_seed(19)
    w13, s13 = _random_weight(EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2, s2 = _random_weight(EXPERTS, HIDDEN, INTERMEDIATE)
    baseline = BlockFP8MoE(w13, s13, w2, s2, provider="triton-fp8")
    candidate = BlockFP8MoE(w13, s13, w2, s2, provider="cute-fp8")
    for m in map(int, args.batches.split(",")):
        valid = valid_presets(m)
        if m == 0:
            parser.error("timing requires M > 0")
        cases = cases_for(m)
        baseline.prepare([m])
        triton_graph = capture(baseline, cases)
        expected = [baseline(*case).float() for case in cases] if args.check else []
        for name in names:
            if name not in valid:
                continue
            candidate.set_config(name).prepare([m])
            errors = []
            for case, reference in zip(cases, expected, strict=False):
                result = candidate(*case).float()
                relative = ((result - reference).norm() / reference.norm().clamp_min(1e-30)).item()
                if not relative <= args.check_rtol:
                    raise AssertionError((m, name, "relative L2", relative))
                errors.append(relative)
            cute_graph = capture(candidate, cases)
            triton_us, cute_us = paired_times(
                (triton_graph, cute_graph), args.iterations, args.replays
            )
            costs = (
                breakdown(candidate, cases, triton_graph, args.iterations, args.replays)
                if args.breakdown
                else None
            )
            print(
                json.dumps(
                    {
                        "M": m,
                        "preset": name,
                        "us": cute_us,
                        "triton_us": triton_us,
                        "ratio_triton_over_cute": triton_us / cute_us,
                        "relative_l2": max(errors) if errors else None,
                        "device": torch.cuda.get_device_name(),
                        **({"breakdown_us": costs} if costs is not None else {}),
                    }
                ),
                flush=True,
            )
            del cute_graph
        del triton_graph


if __name__ == "__main__":
    main()
