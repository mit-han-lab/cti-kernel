# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Graph CUDA-event microbenchmark. Run as python -m benchmark.dflash_round.bench."""

import argparse
import importlib
import json
import math
import statistics
from itertools import product

import torch

from ctikernels.speculative import warm_up
from tests._dflash_round_reference import clone, commit, fixture, fr_commit, fr_prepare, prepare

# Executions of the timed step per graph replay. One small graph per replay
# measures the host's graph-launch rate, not the GPU; back-to-back copies in one
# graph measure GPU time per execution including the in-graph node gap.
UNROLL = 50
# Per-repetition samples and per-pair timings; off by default because some
# runners cap a job's output stream.
VERBOSE = False


def capture(fn, copies=None):
    copies = UNROLL if copies is None else copies
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(copies):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def measure(graph, replays):
    """Microseconds per step execution, over ``replays`` executions in total."""
    launches = max(1, replays // UNROLL)
    graph.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(launches):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (launches * UNROLL)


def emit(point, framework, step, samples, *, note="", kind="timing"):
    value = statistics.median(samples) if samples else None
    print(
        json.dumps(
            dict(
                type=kind,
                **point,
                framework=framework,
                step=step,
                median_us=value,
                **({"samples_us": samples} if VERBOSE else {}),
                note=note,
            )
        ),
        flush=True,
    )
    return value


# Two-sided 95% Student t quantiles by degrees of freedom (the bench uses 20 pairs).
T95 = {9: 2.262, 19: 2.093, 29: 2.045}


def paired_ratio(pairs, comparator):
    """Comparator/ours time ratio: geometric mean over pairs and its 95% bounds."""
    logs = [math.log(p[comparator] / p["cute"]) for p in pairs]
    mean = statistics.mean(logs)
    half = T95.get(len(logs) - 1, 2.0) * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "pairs_n": len(logs),
        "ratio": math.exp(mean),
        "ratio_lo95": math.exp(mean - half),
        "ratio_hi95": math.exp(mean + half),
    }


def node_counts(fn):
    """Graph node types of one captured execution of ``fn``: {type name: count}."""
    try:
        from cuda.bindings import runtime as cudart

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        raw = cudart.cudaGraph_t(init_value=graph.raw_cuda_graph())
        err, _, count = cudart.cudaGraphGetNodes(raw, numNodes=0)
        err, nodes, count = cudart.cudaGraphGetNodes(raw, numNodes=count)
        kinds = {}
        for node in nodes[:count]:
            err, kind = cudart.cudaGraphNodeGetType(node)
            name = getattr(kind, "name", str(kind)).removeprefix("cudaGraphNodeType")
            kinds[name] = kinds.get(name, 0) + 1
        return kinds
    except Exception as error:  # noqa: BLE001 -- diagnostic only; never fails timing
        return {"error": f"{type(error).__name__}: {error}"[:200]}


def graph_steps(steps):
    graphs = {}
    for name, fn, _note in steps:
        if fn is not None:
            graphs[name] = capture(fn)
    # None steps describe work outside this GPU bookkeeping scope; report their notes.
    total = None
    launches = [fn for _, fn, _ in steps if fn is not None]
    if launches:

        def run():
            for fn in launches:
                fn()

        total = capture(run)
    return graphs, total


def benchmark_point(b, q, f, m, args, comparators):
    point = {
        "B": b,
        "Q": q,
        "F": f,
        "M": m,
        "use_pdl": args.use_pdl,
        "replays": args.replays,
        "unroll": UNROLL,
    }
    t = fixture(b, q, m, s=3)
    t["frontier_table"].fill_(f)
    t["done_table"].zero_()
    t["output_ends"].fill_(m)
    t["stop_token_ids"].fill_(-999)
    t["use_pdl"] = args.use_pdl
    # Match the external comparator fixture's acceptance distribution.
    counts = [(q - 1) // 2] if b == 1 else [(q - 1) * row // (b - 1) for row in range(b)]
    t["proposals"].copy_(t["tokens"][:, :-1])
    for row, a in enumerate(counts):
        if a < q - 1:
            t["proposals"][row, a] += 1000
    ours = clone(t, args.device)
    baseline = clone(t, args.device)
    # C always consumes saved P state. P ignores the row tables C updates only
    # in the combined total: the baseline does the same. Restore fronts before
    # timing totals with a separate captured reset node excluded from timing.
    warm_up(q, m, 3, device=args.device, use_pdl=args.use_pdl)
    prepare(ours)
    fr_prepare(baseline)
    steps = {
        "cute": [("prepare", lambda: prepare(ours), ""), ("commit", lambda: commit(ours), "")],
        "torch": [
            ("prepare", lambda: fr_prepare(baseline), "FR + D call-site operations"),
            ("commit", lambda: fr_commit(baseline), "FR + D call-site operations"),
        ],
    }
    for name, factory in comparators.items():
        try:
            steps[name] = factory(b, q, f, m, args.device)
        except (FileNotFoundError, ModuleNotFoundError):
            steps[name] = [
                (
                    "unavailable",
                    None,
                    "Comparator sources/dependencies unavailable; supply --comparator-root.",
                )
            ]
    compiled = {name: graph_steps(seq) for name, seq in steps.items()}
    # Keep F fixed across all measurements. Combined P+C may otherwise converge
    # to done rows and benchmark dummy work. Resets run BEFORE the start event.
    reset_ours = capture(
        lambda: (ours["frontier_table"].fill_(f), ours["done_table"].zero_()), copies=1
    )
    reset_torch = capture(
        lambda: (baseline["frontier_table"].fill_(f), baseline["done_table"].zero_()), copies=1
    )
    resets = {"cute": reset_ours, "torch": reset_torch}
    # Combined totals use immutable saved state: make P+C repeatable by giving C
    # separate destination frontier/done tables. These addresses have identical
    # layouts; P's F and done inputs are invariant throughout all timed replays.
    ours_commit = dict(
        ours, frontier_table=ours["frontier_table"].clone(), done_table=ours["done_table"].clone()
    )
    torch_commit = dict(
        baseline,
        frontier_table=baseline["frontier_table"].clone(),
        done_table=baseline["done_table"].clone(),
    )
    compiled["cute"] = (compiled["cute"][0], capture(lambda: (prepare(ours), commit(ours_commit))))
    compiled["torch"] = (
        compiled["torch"][0],
        capture(lambda: (fr_prepare(baseline), fr_commit(torch_commit))),
    )
    totals_fn = {
        "cute": lambda: (prepare(ours), commit(ours_commit)),
        "torch": lambda: (fr_prepare(baseline), fr_commit(torch_commit)),
    }
    for name, seq in steps.items():
        if name not in totals_fn and any(fn is not None for _, fn, _ in seq):
            totals_fn[name] = lambda seq=seq: [fn() for _, fn, _ in seq if fn is not None]
    for name, fn in totals_fn.items():
        print(json.dumps(dict(type="graph_nodes", **point, framework=name, nodes=node_counts(fn))))
    for reset in resets.values():
        reset.replay()
    samples = {
        (name, step): [] for name, seq in steps.items() for step, fn, _ in seq if fn is not None
    }
    totals = {name: [] for name, (_, total) in compiled.items() if total is not None}
    # Interleave framework steps so clock/thermal drift does not select a winner.
    for repetition in range(args.repetitions):
        names = list(steps)
        if repetition % 2:
            names.reverse()
        for name in names:
            graphs, total = compiled[name]
            if name in resets:
                resets[name].replay()
            for step, graph in graphs.items():
                samples[name, step].append(measure(graph, args.replays))
            if name in resets:
                resets[name].replay()
            if total is not None:
                totals[name].append(measure(total, args.replays))
    medians = {}
    for name, seq in steps.items():
        for step, _, note in seq:
            emit(point, name, step, samples.get((name, step), []), note=note)
        medians[name] = emit(
            point, name, "total", totals.get(name, []), note="P+C graph" if name in resets else ""
        )
    candidates = [name for name in comparators if medians.get(name) is not None]
    if candidates:
        fastest = min(candidates, key=lambda name: medians[name])
        pairs = []
        for i in range(20):
            pair = {}
            for name in ("cute", fastest) if i % 2 == 0 else (fastest, "cute"):
                if name in resets:
                    resets[name].replay()
                pair[name] = measure(compiled[name][1], args.replays)
            pairs.append(pair)
        print(
            json.dumps(
                dict(
                    type="paired_total",
                    **point,
                    comparator=fastest,
                    **({"pairs": pairs} if VERBOSE else {}),
                    **paired_ratio(pairs, fastest),
                    median_delta_us=statistics.median(p["cute"] - p[fastest] for p in pairs),
                )
            ),
            flush=True,
        )
    return medians["cute"], statistics.median(samples["cute", "prepare"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--use-pdl", action="store_true")
    parser.add_argument("--comparator-root")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print samples and pairs")
    parser.add_argument("--b", nargs="+", type=int, help="override the B sweep")
    parser.add_argument("--q", nargs="+", type=int, help="override the Q sweep")
    parser.add_argument("--unroll", type=int, default=UNROLL, help="step executions per graph")
    args = parser.parse_args()
    globals()["UNROLL"] = max(1, args.unroll)
    globals()["VERBOSE"] = args.verbose
    if args.replays < 1 or args.repetitions < 1:
        parser.error("positive timing counts required")
    try:
        module = importlib.import_module("benchmark.dflash_round.comparators")
    except ModuleNotFoundError as error:
        if error.name != "benchmark.dflash_round.comparators":
            raise
        comparators = {}
    else:
        if args.comparator_root and hasattr(module, "configure"):
            module.configure(args.comparator_root)
        comparators = module.COMPARATORS
        if hasattr(module, "configure") and args.comparator_root is None:
            print(
                json.dumps(
                    {
                        "type": "comparator_status",
                        "note": "Supply --comparator-root to enable configured comparators.",
                    }
                ),
                flush=True,
            )
            comparators = {}
    with torch.cuda.device(args.device):
        points = (
            [(1, 8, 100)]
            if args.smoke
            else product(args.b or range(1, 5), args.q or (4, 5, 8, 9, 12, 16), (100, 4096, 32768))
        )
        for b, q, f in points:
            benchmark_point(b, q, f, max(4096, f + q + 1), args, comparators)
        for q in (8,) if args.smoke else args.q or (4, 5, 8, 9, 12, 16):
            times = []
            for m in (4096, 262144):
                _, p = benchmark_point(4, q, 100, m, args, comparators)
                times.append(p)
            print(
                json.dumps(
                    {
                        "type": "same_F_capacity_check",
                        "B": 4,
                        "Q": q,
                        "F": 100,
                        "delta_us": times[1] - times[0],
                        "within_1us": abs(times[1] - times[0]) <= 1,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
