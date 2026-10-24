# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""SP7: prepare_round + fixed_query_addresses vs the fused prepare_block, paired.

Run as python -m benchmark.dflash_round.split_bench. Each step is captured
UNROLL times back to back in one CUDA graph (GPU time per execution including the
in-graph node gap); 20 interleaved pairs per point in one process.
"""

import argparse
import json
import math
import statistics
from itertools import product

import torch

from benchmark.sol.sol import Call, Kernel, estimate, load_ceilings
from ctikernels.speculative import warm_up
from tests._dflash_round_reference import clone, fixture, prepare

from .bench import capture, measure, node_counts, paired_ratio

DEVICE_TABLES = {"RTX 5090": "rtx5090", "RTX PRO 6000": "rtx-pro-6000", "GB10": "gb10"}


def sol_reference(gpu, b, q, f, use_pdl):
    """Shared-helper SOL (benchmark/sol) for fused and split preparation.

    Useful traffic is the KV-index copy, 4·B·(F+Q) bytes read and written. The
    graph replays the same table rows, so when it fits in L2 every byte is served
    at the L2 ceiling; prepare_round moves only O(B·Q) bytes (graph-node floor).
    """
    device = next((v for k, v in DEVICE_TABLES.items() if k in gpu), None)
    if device is None:
        return None
    table = load_ceilings(device)
    size = 4 * b * (f + q)
    resident = 2 * size <= table["l2_bytes"]

    def kernel(name, nbytes):
        dram = {"kv_state": nbytes} if nbytes else {}
        return Kernel(
            name,
            dram_bytes=dram,
            l2_resident_bytes=dict(dram) if resident else {},
            cache_state="warm" if resident else "cold",
            dram_mode="copy",
            pdl=use_pdl,
        )

    mode = "pdl_chain" if use_pdl else "serial"
    fused = estimate(Call([kernel("prepare_block", size)], mode=mode), table, bw_model="large")
    split = estimate(
        Call([kernel("prepare_round", 0), kernel("fixed_query_addresses", size)], mode=mode),
        table,
        bw_model="large",
    )
    return {
        "fused": fused.t_ref_us,
        "split": split.t_ref_us,
        "residency": "l2" if resident else "dram",
        "limiting": split.limiting_resource,
    }


def point(b, q, f, args):
    m = max(4096, f + q + 1)
    t = fixture(b, q, m, s=0)
    t["frontier_table"].fill_(f)
    t["done_table"].zero_()
    t["use_pdl"] = args.use_pdl
    fused, split = clone(t, args.device), clone(t, args.device)
    warm_up(q, m, 0, device=args.device, use_pdl=args.use_pdl)
    prepare(fused)
    prepare(split, True)
    for name, value in fused.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, split[name]), name
    graphs = {
        "fused": capture(lambda: prepare(fused)),
        "split": capture(lambda: prepare(split, True)),
    }
    nodes = {
        k: node_counts(fn)
        for k, fn in (("fused", lambda: prepare(fused)), ("split", lambda: prepare(split, True)))
    }
    pairs = []
    for i in range(args.pairs):
        pair = {}
        for name in ("fused", "split") if i % 2 == 0 else ("split", "fused"):
            pair[name] = measure(graphs[name], args.replays)
        pairs.append(pair)
    # paired_ratio reports comparator/ours; here "ours" is split, the comparator fused.
    ratio = paired_ratio([{"cute": p["split"], "fused": p["fused"]} for p in pairs], "fused")
    deltas = [p["split"] - p["fused"] for p in pairs]
    mean = statistics.mean(deltas)
    half = 2.093 * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    gpu = torch.cuda.get_device_name()
    sol = sol_reference(gpu, b, q, f, args.use_pdl)
    print(
        json.dumps(
            {
                "type": "split_pair",
                "gpu": gpu,
                "B": b,
                "Q": q,
                "F": f,
                "M": m,
                "use_pdl": args.use_pdl,
                "fused_us": statistics.median(p["fused"] for p in pairs),
                "split_us": statistics.median(p["split"] for p in pairs),
                "delta_us": mean,
                "delta_hi95_us": mean + half,
                "within_20us": mean + half <= 20.0,
                "fused_over_split": ratio,
                "launches": {"fused": 1, "split": 2},
                "nodes": nodes,
                "sol_us": sol,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--use-pdl", action="store_true")
    parser.add_argument("--q", nargs="+", type=int, default=[16])
    parser.add_argument("--f", nargs="+", type=int, default=[100, 4096, 32768])
    args = parser.parse_args()
    with torch.cuda.device(args.device):
        for b, q, f in product(range(1, 5), args.q, args.f):
            point(b, q, f, args)


if __name__ == "__main__":
    main()
