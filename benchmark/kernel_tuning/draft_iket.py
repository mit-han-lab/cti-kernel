# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""One eager CuTe DRAFT invocation, or a CPU-only decoded IKET summary.

Usage: python benchmark/kernel_tuning/draft_iket_node.py --heads 32/8 --hd 128
    --query 16 --length 4096 --causal --window 4096 --merge distributed

IKET API and decoded JSON schema:
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def source_root():
    for root in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (root / "python/ctikernels/attention/draft.py").is_file():
            return root
    raise FileNotFoundError("run from the shipped repository root")


def union_duration(intervals):
    """Elapsed time covered by at least one warp; concurrent warps count once."""
    total = 0
    end = None
    for first, last in sorted(intervals):
        total += last - first if end is None else max(0, last - max(first, end))
        end = last if end is None else max(end, last)
    return total


def summarize(path):
    """Print and return per-kernel, per-phase CTA statistics in nanoseconds.

    Native decoded startTs/endTs are device timer nanoseconds. Union warp ranges
    within each CTA/launch before aggregating: count is CTA/launch samples, not
    warp events. Repeated launches of the same kernel remain distinct samples.
    Fractions divide total phase time by the sum of sampled CTA lifetime spans
    (or outermost recorded range endpoints when lifetimes are unavailable).
    merge_wait is inclusive in merge; nested phases never enlarge the denominator.
    A single-cluster capture describes that sampled CTA, not whole-grid latency.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "launches" not in data:
        raise ValueError("expected decoded IKET JSON with launches, not a raw manifest")
    strings = data.get("stringTable", [])
    locations = data.get("locationTable", [])
    phases = defaultdict(list)
    spans = defaultdict(list)
    bases = defaultdict(set)
    launches = list(data["launches"])
    for graph in data.get("graphLaunches", {}).values():
        launches.extend(graph)

    def lookup(table, index):
        return table[str(index)] if isinstance(table, dict) else table[index]

    def cta(index):
        return tuple(lookup(locations, index)["ctaId"])

    def interval(event):
        first, last = event["startTs"], event["endTs"]
        if last < first:
            raise ValueError("decoded range ends before it starts")
        return first, last

    for launch in launches:
        kernel = launch["kernelName"]
        ranges = defaultdict(lambda: defaultdict(list))
        lifetimes = defaultdict(list)
        for event in launch.get("ranges", []):
            name = (
                lookup(strings, event["rangeNameIdx"])
                if "rangeNameIdx" in event
                else event.get("rangeName", "")
            )
            if not name.startswith(("attn.", "merge.")):
                continue
            indices = event.get("warpLocIdxs", [])
            if not indices:
                raise ValueError("decoded DRAFT range has no warp locations")
            # A decoder range may already combine multiple participating warps.
            for block in {cta(index) for index in indices}:
                ranges[block][name].append(interval(event))
        for event in launch.get("warpLifetimes", []):
            lifetimes[cta(event["locIdx"])].append(interval(event))
        for block, names in ranges.items():
            lifetime = lifetimes[block]
            bounds = lifetime or [pair for pairs in names.values() for pair in pairs]
            span = max(last for _, last in bounds) - min(first for first, _ in bounds)
            if span <= 0:
                raise ValueError("decoded CTA has no positive kernel span")
            spans[kernel].append(span)
            bases[kernel].add("warp_lifetimes" if lifetime else "range_endpoints")
            for name, pairs in names.items():
                phases[kernel, name].append(union_duration(pairs))
    if not phases:
        raise ValueError("no DRAFT ranges found; enable CTI_DRAFT_IKET=1 and decode the trace")
    rows = []
    for (kernel, name), durations in sorted(phases.items()):
        row = {
            "kernel": kernel,
            "range": name,
            "count": len(durations),
            "total_duration_ns": sum(durations),
            "median_duration_per_cta_ns": statistics.median(durations),
            "mean_duration_per_cta_ns": statistics.mean(durations),
            "fraction_kernel_span": sum(durations) / sum(spans[kernel]),
            "sampled_ctas": len(spans[kernel]),
            "total_cta_span_ns": sum(spans[kernel]),
            "span_basis": sorted(bases[kernel]),
            "nested_in": "attn.merge" if name == "attn.merge_wait" else None,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def window_arg(value):
    return None if value.lower() == "none" else int(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="32/8", help="Q/KV heads, e.g. 32/8")
    parser.add_argument("--hd", type=int, default=128, choices=(64, 128, 256))
    parser.add_argument("--query", type=int, default=16)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--window", type=window_arg, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=512, help="total KV length including queries")
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument(
        "--merge",
        choices=("legacy", "distributed", "tree"),
        default=os.environ.get("CTI_DRAFT_MERGE", "distributed"),
    )
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args(argv)
    if args.summarize:
        return args
    try:
        args.q_heads, args.kv_heads = map(int, args.heads.split("/"))
    except ValueError:
        parser.error("--heads must be Q/KV, e.g. 32/8")
    if args.q_heads < 1 or args.kv_heads < 1 or args.q_heads % args.kv_heads:
        parser.error("--heads requires positive, dividing Q/KV heads")
    if not 2 <= args.query <= 32 or not args.query <= args.length <= args.capacity:
        parser.error("require 2 <= query <= 32 and query <= length <= capacity")
    if args.batch < 1 or (args.window is not None and args.window < 1):
        parser.error("--batch and --window must be positive")
    return args


def workload(args):
    root = source_root()
    sys.path[:0] = [str(root), str(root / "python")]
    os.environ["CTI_DRAFT_IKET"] = "1"
    os.environ["CTI_DRAFT_MERGE"] = args.merge
    import torch

    from ctikernels.attention.draft import DraftAttention

    device = torch.device("cuda")
    geometry = {
        "query_length": args.query,
        "num_qo_heads": args.q_heads,
        "num_kv_heads": args.kv_heads,
        "head_dim": args.hd,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": args.causal,
        "window": args.window,
        "provider": "cute",
    }
    sizes = DraftAttention.workspace_size(
        batch_size=args.batch, max_kv_tokens=args.capacity, device=device, **geometry
    )
    workspace = [torch.empty(size, dtype=torch.uint8, device=device) for size in sizes]
    tokens = args.batch * args.length
    indices = torch.zeros(args.batch * args.capacity, dtype=torch.int32, device=device)
    indices[:tokens] = torch.arange(tokens, dtype=torch.int32, device=device)
    op = DraftAttention(
        *workspace,
        request_indices=torch.arange(args.batch, dtype=torch.int32, device=device),
        positions=torch.arange(
            args.length - args.query, args.length, dtype=torch.int64, device=device
        ).repeat(args.batch),
        token_indptr=torch.arange(args.batch + 1, dtype=torch.int32, device=device) * args.length,
        token_indices=indices,
        query_indptr=torch.empty(args.batch + 1, dtype=torch.int32, device=device),
        page_indptr=torch.empty(args.batch + 1, dtype=torch.int32, device=device),
        page_indices=torch.empty(args.batch * args.capacity, dtype=torch.int32, device=device),
        last_page_lengths=torch.empty(args.batch, dtype=torch.int32, device=device),
        **geometry,
    )
    torch.manual_seed(0)
    q = torch.randn(
        args.batch * args.query, args.q_heads, args.hd, dtype=torch.bfloat16, device=device
    )
    kv = tuple(
        torch.randn(tokens, 1, args.kv_heads, args.hd, dtype=torch.bfloat16, device=device)
        for _ in range(2)
    )
    out = torch.empty_like(q)
    # run-iket executes discovery and collection separately; no warmup or graphs.
    op.prepare_and_run(q=q, paged_kv_cache=kv, out=out)
    torch.cuda.synchronize()
    print(
        f"DRAFT_IKET B={args.batch} Q={args.query} L={args.length} C={args.capacity} "
        f"heads={args.heads} hd={args.hd} causal={args.causal} window={args.window} "
        f"requested_merge={args.merge} actual_merge={op._implementation.merge_mode} "
        f"splits={op._implementation.splits} gpu={torch.cuda.get_device_name()!r}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        workload(args)


if __name__ == "__main__":
    main()
