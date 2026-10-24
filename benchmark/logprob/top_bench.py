# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DN3 warm-L2 CUDA-graph comparison; trusted upstream source loaded at runtime.

Torch comparator semantics follow vLLM 0.31.0 V1 and SGLang 72730d6;
no code copied. V2 executes the supplied upstream definitions through the
existing AST loader, with only its result container replaced by SimpleNamespace.
"""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from ctikernels.sampling import TopLogprobs

PROVIDERS = ("ours", "vllm_v2", "vllm_v1_torch", "sglang_torch")


def cases():
    for vocab, dtype in (
        (151936, torch.bfloat16),
        (248320, torch.bfloat16),
        (151936, torch.float32),
    ):
        for n in (1, 2, 3, 4, 16, 32, 64):
            for k in (1, 5, 20):
                yield vocab, dtype, n, k, "raw"
        for n in (1, 4, 16):
            yield vocab, dtype, n, 5, "processed"


def summarize(samples):
    medians = {
        name: statistics.median(times) for name, times in samples.items() if name != "gather"
    }
    market = [time for name, time in medians.items() if name != "ours"]
    result = {"median_us": medians}
    if "ours" in medians and market:
        result["ours_over_fastest_market"] = medians["ours"] / min(market)
    if "gather" in samples:
        result["gather_us"] = statistics.median(samples["gather"])
    return result


def torch_market(logits, k, selected, *, ranks):
    """Small semantic baseline; upstream tie order is deliberately unspecified."""
    x = logits.float()
    lp = x.log_softmax(-1)
    top_lp, ids = torch.topk(lp, k, dim=-1)
    selected_lp = lp.gather(1, selected[:, None])[:, 0]
    result = {"ids": ids, "top": top_lp, "selected": selected_lp}
    if ranks:
        result["ranks"] = (x >= x.gather(1, selected[:, None])).sum(-1)
    return result


def load_vllm(root):
    from benchmark.dflash_round.comparators import load_definitions

    return load_definitions(
        Path(root) / "vllm/v1/worker/gpu/sample/logprob.py",
        (
            "_MAX_TOPK_BLOCK",
            "_topk_log_softmax_kernel",
            "_ranks_kernel",
            "compute_token_logprobs",
            "compute_topk_scores",
        ),
        module_name="_dn3_vllm_v2_logprob",
        globals={"LogprobsTensors": SimpleNamespace},
    )


def parity(ours, market):
    """Report ties as set differences; compare scores independent of tied IDs."""
    ids_match = torch.equal(
        ours["ids"].long().sort(-1).values, market["ids"].long().sort(-1).values
    )
    selected_diff = (ours["selected"] - market["selected"]).abs().max().item()
    top_diff = (ours["top"].sort(-1).values - market["top"].sort(-1).values).abs().max().item()
    return {
        "ids_as_sets": ids_match,
        "selected_max_abs_diff": selected_diff,
        "top_max_abs_diff": top_diff,
        "logprobs_close": max(selected_diff, top_diff) <= 2e-4,
        "ranks_equal": torch.equal(ours["ranks"].long(), market["ranks"].long()),
    }


# --cold: a buffer of twice the L2 size, rewritten before every timed call.
L2_FLUSH = None


def _cold(call):
    def flushed():
        L2_FLUSH.zero_()
        return call()

    return flushed


def capture(call, executions=20):
    # Warm on a side stream, including upstream JIT compilation, before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(executions):
            result = call()
    return graph, result


def measure_interleaved(callables, replays):
    """Warm L2 by default; with L2_FLUSH every call starts cold and the median time of
    the flush alone (its own graph, interleaved likewise) is subtracted."""
    if L2_FLUSH is not None:
        callables = {name: _cold(call) for name, call in callables.items()}
        callables["flush"] = L2_FLUSH.zero_
    graphs, outputs = {}, {}
    for name, call in callables.items():
        graphs[name], outputs[name] = capture(call)
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(replays)
        ]
        for name in graphs
    }
    # Each provider and the gather appear once per replay round, never a
    # complete timing sweep before the others. L2 is warm; there is no flush.
    for replay in range(replays):
        for name, graph in graphs.items():
            start, end = events[name][replay]
            start.record()
            graph.replay()
            end.record()
    torch.cuda.synchronize()
    samples = {
        name: [start.elapsed_time(end) * 1000 / 20 for start, end in pairs]
        for name, pairs in events.items()
    }
    if L2_FLUSH is not None:
        flush = statistics.median(samples.pop("flush"))
        samples = {name: [t - flush for t in times] for name, times in samples.items()}
    check = (
        parity(outputs["ours"], outputs["vllm_v2"])
        if {"ours", "vllm_v2"} <= outputs.keys()
        else None
    )
    return summarize(samples), check


def bench_row(vllm, vocab, dtype, n, k, mode, replays, row_chunk, providers=PROVIDERS):
    device = torch.device("cuda", torch.cuda.current_device())
    # Both implementations see identical rows. Ours includes indexed gathering;
    # market inputs have already been gathered, as in their full-batch APIs.
    source = torch.randn((n + 2, vocab + 8), dtype=dtype, device=device)[:, :vocab]
    indices = torch.arange(n - 1, -1, -1, dtype=torch.int32, device=device)
    selected = torch.arange(n, dtype=torch.int32, device=device) % vocab
    probabilities = source.float().softmax(-1) if mode == "processed" else None
    gather_source = source if probabilities is None else probabilities
    gather_indices = indices.long()
    market_rows = gather_source.index_select(0, gather_indices)
    if probabilities is not None:
        market_rows = market_rows.log()
    market_selected = selected.long()
    op = TopLogprobs(device, vocab, k, row_chunk=row_chunk)
    outputs = {
        "selected": torch.empty(n, device=device),
        "ranks": torch.empty(n, dtype=torch.int32, device=device),
        "ids": torch.empty((n, k), dtype=torch.int32, device=device),
        "top": torch.empty((n, k), device=device),
    }
    args = {
        "row_indices": indices,
        "num_rows": torch.tensor([n], dtype=torch.int32, device=device),
        "selected_ids": selected,
        "top_k": torch.full((n,), k, dtype=torch.int32, device=device),
        "mode": mode,
        "probabilities": probabilities,
        "out_selected_logprobs": outputs["selected"],
        "out_ranks": outputs["ranks"],
        "out_top_ids": outputs["ids"],
        "out_top_logprobs": outputs["top"],
    }
    if mode == "processed":
        args.update(
            temperatures=torch.ones(n, device=device),
            filter_top_k=torch.full((n,), vocab, dtype=torch.int32, device=device),
            top_p=torch.ones(n, device=device),
        )

    def ours():
        op.run(source, **args)
        return outputs

    def gather():
        # Match the provider's static chunks; time only index_select.
        return [gather_source.index_select(0, rows) for rows in gather_indices.split(row_chunk)]

    def v2():
        result = vllm.compute_topk_scores(market_rows, k, market_selected)
        return {
            "ids": result.logprob_token_ids[:, 1:],
            "top": result.logprobs[:, 1:],
            "selected": result.logprobs[:, 0],
            "ranks": result.selected_token_ranks,
        }

    available = {
        "ours": ours,
        "vllm_v2": v2,
        "vllm_v1_torch": lambda: torch_market(market_rows, k, market_selected, ranks=True),
        "sglang_torch": lambda: torch_market(market_rows, k, market_selected, ranks=False),
    }
    callables = {name: available[name] for name in providers}
    callables["gather"] = gather
    measured, check = measure_interleaved(callables, replays)
    return dict(
        device=torch.cuda.get_device_name(device),
        V=vocab,
        dtype=str(dtype).removeprefix("torch."),
        n=n,
        k=k,
        mode=mode,
        provider=op.provider,
        row_chunk=row_chunk,
        executions_per_graph=20,
        replays=replays,
        l2="warm" if L2_FLUSH is None else "cold (flush time subtracted)",
        market_inputs="gathered logits" if mode == "raw" else "precomputed FP32 log(p)",
        parity_vllm_v2=check,
        **measured,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--vllm-path",
        type=Path,
        help="Trusted vLLM 0.31.0 bundle directory (required only for vllm_v2)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
        help="Providers to time (default: all); gather is always timed",
    )
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Run only the first N rows (smoke check)")
    parser.add_argument("--cold", action="store_true", help="Evict L2 before every timed call")
    args = parser.parse_args(argv)
    if args.cold:
        global L2_FLUSH
        l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        L2_FLUSH = torch.empty(2 * l2 // 4, dtype=torch.int32, device="cuda")
    if args.replays < 30 or args.row_chunk < 1 or (args.limit is not None and args.limit < 1):
        parser.error("require replays >= 30, row-chunk >= 1 and limit >= 1")
    if "vllm_v2" in args.providers and args.vllm_path is None:
        parser.error("--vllm-path is required when timing vllm_v2")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; this harness does no CPU timing")
    vllm = load_vllm(args.vllm_path) if "vllm_v2" in args.providers else None
    torch.manual_seed(1701)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as output:
        for index, case in enumerate(cases()):
            if args.limit is not None and index >= args.limit:
                break
            row = bench_row(vllm, *case, args.replays, args.row_chunk, args.providers)
            line = json.dumps(row, sort_keys=True, allow_nan=False)
            times = " ".join(f"{name}={us:.1f}" for name, us in row["median_us"].items())
            print(
                f"{row['mode']} V={row['V']} {row['dtype']} n={row['n']} k={row['k']}: "
                f"{times} gather={row['gather_us']:.1f} us",
                flush=True,
            )
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
