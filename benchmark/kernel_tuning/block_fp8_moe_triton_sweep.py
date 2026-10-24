# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Graph-replay FC1/FC2 sweep for Qwen block-FP8 experts, with global EP routing.

Run with ``python -m benchmark.kernel_tuning.block_fp8_moe_triton_sweep``.
Redirect stdout to JSONL; ``--emit-table results.jsonl`` emits Python source.
The default grid and time budget target a single-device run under 20 minutes;
compilation of an individual candidate cannot be interrupted by the budget.
"""

import argparse
import functools
import itertools
import json
import pprint
import statistics
import time
from pathlib import Path

import torch

from benchmark.kernel_tuning.block_fp8_moe_sol import SHAPES
from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE
from ctikernels.moe_impl import block_fp8_moe_config, fused_moe_kernel_triton, moe_align_block_size

BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def candidates(m, e, exhaustive=False, shape="qwen3.5", *, capability):
    _, h, i, top_k = SHAPES[shape]
    baseline = block_fp8_moe_config(m, e, 2 * i, h, top_k, capability=capability)
    bms = [bm for bm in (16, 32, 64, 128) if bm == 16 or bm <= m * top_k]
    groups = (1,) if m <= 32 else (1, 8, 16)
    grid = [
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": group,
            "num_warps": warps,
            "num_stages": stages,
        }
        for bm, bn, group, warps, stages in itertools.product(
            bms, (32, 64, 128, 256), groups, (4, 8), (2, 3, 4, 5)
        )
    ]
    if not exhaustive:
        # Cover every row tile and all stage/warp choices without compiling the
        # full Cartesian grid at every M. Extreme sizes use a smaller grid.
        count = 12 if m <= 4 or m >= 1024 else 24
        grid = [grid[index * (len(grid) - 1) // (count - 1)] for index in range(count)]
    return [baseline] + [c for c in grid if c != baseline]


def capture(calls):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    return graph


def measure(calls, samples, replays):
    graph = capture(calls)
    for _ in range(3):
        graph.replay()
    # Flush outside each timed replay: eight small-M routings alone can still
    # fit in a large L2, especially for EP2 FC2. Exclude eviction from latency.
    eviction = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(replays)
    ]
    times = []
    for _ in range(samples):
        for start, end in events:
            eviction.zero_()
            start.record()
            graph.replay()
            end.record()
        events[-1][1].synchronize()
        times.append(
            sum(start.elapsed_time(end) for start, end in events) * 1000 / (len(calls) * replays)
        )
    return statistics.median(times)


def make_weights(e, shape="qwen3.5"):
    _, h, i, _ = SHAPES[shape]
    result = []
    for n, k in ((2 * i, h), (h, i)):
        w = torch.empty(e, n, k, device="cuda", dtype=torch.float8_e4m3fn)
        for expert in range(e):
            w[expert].copy_((torch.randn(n, k, device="cuda") * 16).to(w.dtype))
        scale = torch.full((e, n // 128, k // 128), 1 / 512, device="cuda", dtype=torch.bfloat16)
        result.extend((w, scale))
    return tuple(result)


def make_cases(m, e, device="cuda", shape="qwen3.5"):
    global_e, h, _, top_k = SHAPES[shape]
    if e not in (global_e, global_e // 2):
        raise ValueError("E_local must match EP1 or EP2 of the selected shape")
    cases = []
    # Disjoint local windows across the eight routings at small M rotate the
    # working set. Large M naturally touches most experts. Exactly 1/ep of
    # routes are local to rank zero (and no ids are remapped on the host).
    orders = [torch.randperm(e, device=device) for _ in range(global_e // e)]
    for routing in range(8):
        ids = torch.randint(global_e, (m, top_k), device=device, dtype=torch.int32)
        for rank in range(global_e // e):
            slots = ids[:, rank :: global_e // e]
            offsets = (
                torch.arange(slots.numel(), device=device) + routing * min(slots.numel(), e // 8)
            ) % e
            slots.copy_((rank * e + orders[rank][offsets]).view_as(slots).to(torch.int32))
        x = torch.randn(m, h, device=device, dtype=torch.bfloat16)
        weights = torch.rand(m, top_k, device=device)
        weights /= weights.sum(-1, keepdim=True)
        cases.append((x, weights, ids))
    return cases


def projection_calls(op, cases, config, second, inputs):
    calls = []
    for (x, weights, ids), (q, qs, qh, qhs) in zip(cases, inputs, strict=True):
        sorted_ids, experts, padded = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], op.num_global_experts
        )
        target = torch.empty(
            x.shape[0], op.top_k, op.h if second else 2 * op.i, device="cuda", dtype=torch.bfloat16
        )
        calls.append(
            functools.partial(
                fused_moe_kernel_triton,
                qh if second else q,
                op.w2 if second else op.w13,
                target,
                weights,
                ids,
                sorted_ids,
                experts,
                padded,
                second,
                1 if second else op.top_k,
                config,
                torch.bfloat16,
                a_scale=qhs if second else qs,
                b_scale=op.s2 if second else op.s13,
                block_fp8_provider="triton-fp8",
                expert_map=op.expert_map,
                num_global_experts=op.num_global_experts,
            )
        )
    return calls


def prepare_inputs(op, cases, config):
    inputs = []
    for x, weights, ids in cases:
        q, qs = quantize_per_token_group_fp8(x)
        sorted_ids, experts, padded = moe_align_block_size(
            ids, config["BLOCK_SIZE_M"], op.num_global_experts
        )
        gu = torch.empty(x.shape[0], op.top_k, 2 * op.i, device="cuda", dtype=torch.bfloat16)
        fused_moe_kernel_triton(
            q,
            op.w13,
            gu,
            weights,
            ids,
            sorted_ids,
            experts,
            padded,
            False,
            op.top_k,
            config,
            torch.bfloat16,
            a_scale=qs,
            b_scale=op.s13,
            expert_map=op.expert_map,
            block_fp8_provider="triton-fp8",
        )
        act = torch.empty(x.shape[0] * op.top_k, op.i, device="cuda", dtype=torch.bfloat16)
        silu_and_mul(gu.view(-1, 2 * op.i), act)
        qh, qhs = quantize_per_token_group_fp8(act)
        inputs.append((q, qs, qh, qhs))
    return inputs


def table_from_records(records, shape=None):
    pairs = {}
    for row in records:
        if row.get("gemm") in ("fc1", "fc2"):
            # Records predating --shape describe Qwen3.5 only.
            _, h, i, top_k = SHAPES[row.get("shape", "qwen3.5")]
            dimensions = (row.get("H", h), row.get("I", i), row.get("top_k", top_k))
            if shape is not None and dimensions != SHAPES[shape][1:]:
                continue
            key = (tuple(row["device_class"]), (row["E_local"], *dimensions), row["M"])
            pairs.setdefault(key, {})[row["gemm"]] = row["best"]["config"]
    table = {}
    for (device, dimensions, m), pair in sorted(pairs.items()):
        if set(pair) != {"fc1", "fc2"}:
            raise ValueError("each M/E_local needs both FC1 and FC2 records")
        if pair["fc1"]["BLOCK_SIZE_M"] != pair["fc2"]["BLOCK_SIZE_M"]:
            raise ValueError("FC1 and FC2 must share BLOCK_SIZE_M")
        table.setdefault(device, {}).setdefault(dimensions, []).append(
            (m, pair["fc1"], pair["fc2"])
        )
    return table


def emit_table(jsonl_paths, shape=None):
    """Merge JSONL sweep results into the compact MEASURED table format."""
    records = []
    for path in jsonl_paths:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line)
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M", "num_warps", "num_stages")
    measured = {
        device: {
            dimensions: tuple(
                (m, tuple(fc1[k] for k in keys), tuple(fc2[k] for k in keys))
                for m, fc1, fc2 in rows
            )
            for dimensions, rows in shapes.items()
        }
        for device, shapes in table_from_records(records, shape).items()
    }
    return "MEASURED = " + pprint.pformat(measured, sort_dicts=True, width=100) + "\n"


class TunedMoE(BlockFP8MoE):
    def use_configs(self, configs):
        self.configs = configs
        self._ready.clear()
        return self

    def config_for(self, num_tokens):
        return getattr(self, "configs", None) or super().config_for(num_tokens)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--shape", choices=SHAPES, default="qwen3.5")
    parser.add_argument("--experts", help="local expert counts; defaults to EP1,EP2")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--budget-seconds", type=float, default=1080)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--full", action="store_true", help="also compare whole-forward graphs")
    parser.add_argument("--emit-table", nargs="+")
    args = parser.parse_args()
    if args.emit_table:
        print(emit_table(args.emit_table, args.shape), end="")
        return
    global_e, h, i, top_k = SHAPES[args.shape]
    batches, sizes = (
        list(map(int, args.batches.split(","))),
        list(map(int, (args.experts or f"{global_e},{global_e // 2}").split(","))),
    )
    if min(batches + [args.samples, args.replays, args.budget_seconds]) <= 0:
        parser.error("batch sizes, timing counts and budget must be positive")
    if any(e not in (global_e, global_e // 2) for e in sizes):
        parser.error(f"E_local must be {global_e} or {global_e // 2}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        parser.error("requires CUDA compute capability >= 8.9")
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources, PTXASError

    torch.manual_seed(191)
    device = (
        *torch.cuda.get_device_capability(),
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    deadline = time.monotonic() + args.budget_seconds
    records = []
    shape_info = {"shape": args.shape, "E_global": global_e, "H": h, "I": i, "top_k": top_k}
    remaining = len(batches) * len(sizes)
    for e in sizes:
        params = make_weights(e, args.shape)
        mapping = torch.full((global_e,), -1, dtype=torch.int32, device="cuda")
        mapping[:e] = torch.arange(e, device="cuda", dtype=torch.int32)
        op = TunedMoE(*params, expert_map=mapping, provider="triton-fp8")
        for m in batches:
            pair_deadline = time.monotonic() + max(0, deadline - time.monotonic()) / remaining
            remaining -= 1
            cases = make_cases(m, e, shape=args.shape)
            baseline = block_fp8_moe_config(m, e, 2 * i, h, top_k, capability=op.device_class[:2])
            inputs = prepare_inputs(op, cases, baseline)
            results = [[], []]
            failures = 0
            tried = 0
            grid = candidates(m, e, args.exhaustive, args.shape, capability=op.device_class[:2])
            for config in grid:
                if tried and time.monotonic() >= pair_deadline:
                    break
                tried += 1
                # Both projections are timed at each BM; choose BM jointly below.
                for second in (False, True):
                    try:
                        calls = projection_calls(op, cases, config, second, inputs)
                        us = measure(calls, args.samples, args.replays)
                    except (CompilationError, OutOfResources, PTXASError):
                        failures += 1
                        continue
                    results[int(second)].append({"us": us, "config": config})
                del calls
            shared = {r["config"]["BLOCK_SIZE_M"] for r in results[0]} & {
                r["config"]["BLOCK_SIZE_M"] for r in results[1]
            }
            if not shared:
                raise RuntimeError("no shared BLOCK_SIZE_M compiled for both projections")
            best_by_bm = {
                bm: [
                    sorted(
                        (r for r in result if r["config"]["BLOCK_SIZE_M"] == bm),
                        key=lambda r: r["us"],
                    )
                    for result in results
                ]
                for bm in shared
            }
            bm = min(shared, key=lambda b: sum(rows[0]["us"] for rows in best_by_bm[b]))
            chosen = tuple(rows[0]["config"] for rows in best_by_bm[bm])
            for gemm, rows in zip(("fc1", "fc2"), best_by_bm[bm], strict=True):
                record = {
                    "device_class": device,
                    "M": m,
                    "E_local": e,
                    **shape_info,
                    "gemm": gemm,
                    "best": rows[0],
                    "top3": rows[:3],
                    "candidates": tried,
                    "failures": failures,
                    "budget_limited": tried < len(grid),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
            if args.full:
                timings = []
                for configs in (chosen, (baseline, baseline)):
                    op.use_configs(configs).prepare([m])
                    timings.append(
                        measure(
                            [functools.partial(op, *case) for case in cases],
                            args.samples,
                            args.replays,
                        )
                    )
                print(
                    json.dumps(
                        {
                            "device_class": device,
                            "M": m,
                            "E_local": e,
                            **shape_info,
                            "full_us": timings[0],
                            "placeholder_us": timings[1],
                        }
                    ),
                    flush=True,
                )
            del inputs, cases
        del op, params
    table = table_from_records(records)
    print(
        json.dumps(
            {
                "TABLE": [
                    {
                        "device_class": device,
                        "shapes": [{"shape": key, "rows": rows} for key, rows in experts.items()],
                    }
                    for device, experts in table.items()
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
