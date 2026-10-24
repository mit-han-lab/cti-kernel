# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Graph-replay selected logprobs benchmark, with adjacent paired measurements.

Run from the repository root: python -m benchmark.logprob.selected_bench --out results/selected
Comparator expressions are independently written from the task's public contracts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.sol.sol import Call, Kernel, estimate, load_ceilings
from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.logprob.cute import Provider, choose_splits
from ctikernels.logprob.linear import _TorchProvider


def interval(ratios, seed=517):
    """Paired bootstrap 95% interval of the median per-round speed ratio."""
    rng = random.Random(seed)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return [samples[49], samples[1949]]


def capture(function):
    for _ in range(3):
        function()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    graph.replay()
    return graph, output


def elapsed(graph, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def measure(functions, rounds, repeats):
    graphs = {name: capture(function) for name, function in functions.items()}
    competitors = [name for name in functions if name != "ours"]
    pairs = {name: [] for name in [*competitors, "aa"]}
    names = list(pairs)
    for round_index in range(rounds):
        rotated = names[round_index % len(names) :] + names[: round_index % len(names)]
        for name in rotated:
            other = "ours" if name == "aa" else name
            order = [("a", "ours"), ("b", other)]
            if round_index % 2:
                order.reverse()
            times = {label: elapsed(graphs[key][0], repeats) for label, key in order}
            pairs[name].append([times["a"], times["b"]])
    medians = {"ours": statistics.median(a for name in competitors for a, _ in pairs[name])}
    summaries = {}
    for name, values in pairs.items():
        ratios = [b / a for a, b in values]
        summaries[name] = {
            "ratio": statistics.median(ratios),
            "ci95": interval(ratios),
            "pairs_us": values,
        }
        if name != "aa":
            medians[name] = statistics.median(b for _, b in values)
    aa_errors = sorted(abs(b / a - 1) for a, b in pairs["aa"])
    epsilon = aa_errors[min(len(aa_errors) - 1, int(0.95 * len(aa_errors)))]
    market = [name for name in ("sglang", "vllm", "torch") if name in competitors]
    fastest = min(market, key=medians.get)
    return {
        "median_us": medians,
        "paired": summaries,
        "fastest_comparator": fastest,
        "speedup": summaries[fastest]["ratio"],
        "ci95": summaries[fastest]["ci95"],
        "aa_epsilon": epsilon,
    }


def post_sol(rows, vocab, sm_count, ceilings, measured):
    size = rows * vocab * 2
    resident = size <= ceilings["l2_bytes"]
    call = Call(
        [
            Kernel(
                "selected_logprobs",
                dram_bytes={"intermediates": size},
                l2_resident_bytes={"intermediates": size} if resident else {},
                cache_state="warm" if resident else "cold",
                n_ctas=rows * choose_splits(rows, vocab, sm_count),
            )
        ]
    )
    return estimate(call, ceilings).to_dict(measured)


def run_row(hidden_size, vocab, rows, temperature, rounds, repeats, ceilings):
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02
    weight = torch.randn(vocab, hidden_size, device="cuda", dtype=torch.bfloat16)
    targets = torch.arange(rows, device="cuda", dtype=torch.int64) % vocab
    op = ChunkedLinearSelectedTokenLogprobs()
    old_op = ChunkedLinearSelectedTokenLogprobs()
    old_op._providers[hidden.device] = _TorchProvider()

    def sglang():
        logits = F.linear(hidden, weight).float() / temperature
        return logits.log_softmax(dim=-1).gather(1, targets[:, None])[:, 0]

    def vllm():
        # Division occurs in the head dtype; dtype= promotes only log_softmax.
        logits = F.linear(hidden, weight) / temperature
        return logits.log_softmax(dim=-1, dtype=torch.float32).gather(1, targets[:, None])[:, 0]

    full = measure(
        {
            "ours": lambda: op.forward(hidden, weight, targets, temperature=temperature),
            "sglang": sglang,
            "vllm": vllm,
            "old": lambda: old_op.forward(hidden, weight, targets, temperature=temperature),
        },
        rounds,
        repeats,
    )
    if op.provider != "cute":
        raise RuntimeError("benchmark requires the CuTe provider")
    logits = F.linear(hidden, weight)
    output = torch.empty(rows, device="cuda", dtype=torch.float32)
    provider = Provider(hidden.device)

    def fused():
        provider(logits, targets, output, temperature)
        return output

    def post_torch():
        return (logits.float() / temperature).log_softmax(-1).gather(1, targets[:, None])[:, 0]

    post = measure({"ours": fused, "torch": post_torch}, rounds, repeats)
    post["sol"] = post_sol(rows, vocab, provider.sm_count, ceilings, post["median_us"]["ours"])
    return {"H": hidden_size, "V": vocab, "n": rows, "full": full, "post_gemm": post}


def markdown(report):
    lines = [
        "# Selected-token logprobs",
        "",
        f"BF16, T={report['temperature']:g}; {report['rounds']} rotated adjacent-pair rounds, "
        f"{report['repeats']} graph replays per event sample.",
        "",
        "Speedup is the median paired ratio against the faster market comparator; "
        "intervals are paired bootstrap 95% CIs. Epsilon is the 95th percentile "
        "absolute A/A ratio deviation. Times are microseconds.",
        "",
    ]
    for section in ("full", "post_gemm"):
        lines += [f"## {section}", ""]
        columns = ["ours", "sglang", "vllm", "old"] if section == "full" else ["ours", "torch"]
        lines += [
            "| H | V | n | "
            + " | ".join(columns)
            + " | faster | speedup [95% CI] | epsilon | SOL us |",
            "|---|---|---|" + "---|" * (len(columns) + 4),
        ]
        for row in report["rows"]:
            data = row[section]
            times = " | ".join(f"{data['median_us'][name]:.3f}" for name in columns)
            low, high = data["ci95"]
            sol = f"{data['sol']['t_ref_us']:.3f}" if "sol" in data else "—"
            lines.append(
                f"| {row['H']} | {row['V']} | {row['n']} | {times} | "
                f"{data['fastest_comparator']} | {data['speedup']:.3f} [{low:.3f}, {high:.3f}] | "
                f"{data['aa_epsilon']:.4f} | {sol} |"
            )
        lines.append("")
    lines += [
        "SOL covers post-GEMM logits reads (n × V × 2 bytes), with L2 residency "
        "when the logits fit the measured device L2. The helper also applies its "
        "graph-node floor. It excludes scalar arithmetic and reduction synchronization.",
        "",
        "vLLM-style division rounds in BF16 before FP32 log_softmax; the other "
        "arithmetic paths divide in FP32. These are timing comparators, not identical numerical contracts.",
        "",
    ]
    return "\n".join(lines)


def _compact(value):
    """Stdout copy without per-pair samples (runner streams are size-capped)."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in ("pairs_us", "assumptions")}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output stem (.json and .md)")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ceilings", help="shared SOL device name or calibrated JSON path")
    args = parser.parse_args()
    if args.rounds < 20 or args.repeats < 1:
        parser.error("require at least 20 rounds and one replay")
    name = torch.cuda.get_device_name()
    source = args.ceilings
    if source is None:
        source = next(
            (
                key
                for label, key in (
                    ("5090", "rtx5090"),
                    ("PRO 6000", "rtx-pro-6000"),
                    ("GB10", "gb10"),
                )
                if label in name
            ),
            None,
        )
    if source is None:
        parser.error("provide --ceilings with calibrated SOL JSON for this device")
    ceilings = load_ceilings(
        json.loads(Path(source).read_text()) if Path(source).is_file() else source
    )
    torch.manual_seed(517)
    report = {
        "device": name,
        "dtype": "bfloat16",
        "temperature": args.temperature,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "rows": [],
    }
    with torch.inference_mode():
        for hidden_size in (2048, 4096):
            for vocab in (151936, 248320):
                for rows in (1, 2, 4, 8, 16, 32, 64):
                    report["rows"].append(
                        run_row(
                            hidden_size,
                            vocab,
                            rows,
                            args.temperature,
                            args.rounds,
                            args.repeats,
                            ceilings,
                        )
                    )
                    print(
                        json.dumps(_compact(report["rows"][-1]), separators=(",", ":")), flush=True
                    )
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
                    args.out.with_suffix(".md").write_text(markdown(report))


if __name__ == "__main__":
    main()
