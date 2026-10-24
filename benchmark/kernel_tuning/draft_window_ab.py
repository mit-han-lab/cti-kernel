# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Single-process paired A/B of DraftAttention arms sharing Q/K/V buffers.

Arms are name=path[@ENV=V+ENV=V]; the first arm is the candidate. Each path
contains python/ctikernels, copied into ./_arms/<name>/ctikernels_<name> with
absolute package imports rewritten. The _arms scratch directory is created
under the current working directory, is not ignored, and is left for manual
cleanup. Loading an arm replaces its existing scratch copy. Environment
overrides apply only during workspace allocation and operator construction.

For each mask, batch and context, rotate at least five distinct BF16 K/V
buffers covering >= 2x L2, with physical lengths rounded to 256 tokens.
Page size is 1; capacity defaults to 32768. All arms share the same seeded
buffers and use an independent dense FP32 reference for the last layer.
Each round rebuilds one CUDA graph per arm in shuffled order. Each graph
contains one call per layer, after a side-stream warmup, and gets three
warmup replays. Each pair times arms in shuffled order, taking the median of
five CUDA-event replays per arm, divided by the layer count. There are
pairs // rounds pairs per round; any remainder is discarded.

JSONL on stdout includes host, per-pair (type: pair), and aggregate (type: ab2) records. Ratios are
arm_us / cand_us, so values above one favor the candidate. Bootstrap 2000
resamples of the per-pair ratios for a percentile 95% CI of their median;
not_slower is ci_high >= 1. With arms named base and base2 (two copies of one
baseline tree), aa_base2_vs_base reports the A/A noise floor the same way.
Medians use the upper middle value for even sample counts. Buffer, order and bootstrap seeds are respectively 0, 7, 1.
GB10 timing is pinned to available Cortex-X925 cores. Torch is imported
only inside functions, so importing this module requires neither Torch nor CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

MASKS = {
    "full": (16, False, None, 5),
    "c4096": (16, True, 4096, 5),
    "n2048": (8, False, 2048, 5),
    "dsv4": (8, True, 2048, 5),
    "q3moe": (16, False, None, 5),
    "q3moe_causal": (16, True, None, 5),
    # Causal full VERIFY with --heads (e.g. Qwen3.5 16,4,256: 8 attention layers).
    "cfull": (16, True, None, 8),
}


BOOTSTRAP_SEED = 1
BOOTSTRAP_RESAMPLES = 2000


def emit(value):
    print(json.dumps(value), flush=True)


def pin_cores():
    """Restrict Linux affinity to available Cortex-X925 cores (CPU part 0xd85)."""
    cores = []
    try:
        for stanza in re.split(r"\n\s*\n", Path("/proc/cpuinfo").read_text().strip()):
            fields = dict(line.split(":", 1) for line in stanza.splitlines() if ":" in line)
            fields = {key.strip().lower(): value.strip() for key, value in fields.items()}
            if int(fields.get("cpu part", "0"), 16) == 0xD85:
                cores.append(int(fields["processor"]))
        cores = sorted(set(cores) & os.sched_getaffinity(0))
        if cores:
            os.sched_setaffinity(0, cores)
    except Exception:  # noqa: BLE001 - affinity is best effort on every host.
        return cores
    return cores


def parse_arm(spec):
    """Split an arm spec, preserving equals signs inside override values."""
    name, rest = spec.split("=", 1)
    path, _, overrides = rest.partition("@")
    env = dict(item.split("=", 1) for item in overrides.split("+") if item)
    return name, path, env


def load_arm(name, path):
    """Copy and import an isolated DraftAttention package beneath the current cwd."""
    destination = Path("_arms") / name
    package = "ctikernels_" + name
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(
        Path(path) / "python" / "ctikernels",
        destination / package,
        ignore=shutil.ignore_patterns("__pycache__", "csrc"),
    )
    for source in (destination / package).rglob("*.py"):
        original = source.read_text()
        rewritten = re.sub(r"\bctikernels(?=[.\s])", package, original)
        if rewritten != original:
            source.write_text(rewritten)
    sys.path.insert(0, str(destination.resolve()))
    return importlib.import_module(package + ".attention.draft").DraftAttention


def median(values):
    """Return the upper middle sample, matching the original benchmark."""
    return sorted(values)[len(values) // 2]


def bootstrap(ratios, it=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Return the median ratio and its seeded percentile 95% bootstrap CI."""
    rng = random.Random(seed)
    medians = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(it))
    return median(ratios), medians[int(0.025 * it)], medians[int(0.975 * it) - 1]


def pair_record(mask, batch, length, round_index, pair_index, elapsed):
    candidate = next(iter(elapsed))
    return {
        "type": "pair",
        "mask": mask,
        "B": batch,
        "L": length,
        "round": round_index,
        "pair": pair_index,
        "candidate": candidate,
        "us": elapsed,
        "ratio_vs_candidate": {name: us / elapsed[candidate] for name, us in elapsed.items()},
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def mask_heads(mask, heads):
    return (32, 4, 128) if mask in ("q3moe", "q3moe_causal") else heads


@dataclass
class Buffers:
    q: object
    keys: list
    values: list
    indices: object
    positions: object
    token_indptr: object
    padded_length: int
    num_layers: int


def setup_buffers(batch, ctx, query_length, heads, layers, cap, device):
    import torch

    hq, hk, head_dim = heads
    kv_length = ctx + query_length
    padded_length = math.ceil(kv_length / 256) * 256
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    num_layers = max(
        layers, math.ceil(2 * l2_bytes / (2 * batch * padded_length * hk * head_dim * 2))
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    keys = [
        torch.randn(
            batch * padded_length,
            1,
            hk,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(num_layers)
    ]
    values = [
        torch.randn(
            batch * padded_length,
            1,
            hk,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(num_layers)
    ]
    q = torch.randn(
        batch * query_length,
        hq,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    indices = torch.zeros(batch * cap, dtype=torch.int32)
    indices[: batch * kv_length] = (
        torch.arange(batch, dtype=torch.int32)[:, None] * padded_length
        + torch.arange(kv_length, dtype=torch.int32)
    ).flatten()
    indices = indices.to(device)
    positions = torch.arange(ctx, kv_length, dtype=torch.int64, device=device).repeat(batch)
    token_indptr = (torch.arange(batch + 1, dtype=torch.int32) * kv_length).to(device)
    return Buffers(q, keys, values, indices, positions, token_indptr, padded_length, num_layers)


def reference(buffers, batch, ctx, query_length, heads, causal, window, device):
    """Dense FP32 GQA reference of the last layer, preserving the original math."""
    import torch

    hq, hk, head_dim = heads
    kv_length = ctx + query_length
    keys = buffers.keys[-1].view(batch, buffers.padded_length, hk, head_dim)[:, :kv_length].float()
    values = (
        buffers.values[-1].view(batch, buffers.padded_length, hk, head_dim)[:, :kv_length].float()
    )
    scores = torch.einsum(
        "bqhgd,bkhd->bhgqk",
        buffers.q.view(batch, query_length, hk, hq // hk, head_dim).float(),
        keys,
    ) / math.sqrt(head_dim)
    positions = torch.arange(ctx, kv_length, device=device)[:, None]
    key_positions = torch.arange(kv_length, device=device)[None]
    mask = torch.ones((query_length, kv_length), dtype=torch.bool, device=device)
    if window is not None:
        mask &= (positions - key_positions).abs() < window
    if causal:
        mask &= key_positions <= positions
    return torch.einsum(
        "bhgqk,bkhd->bqhgd", scores.masked_fill(~mask, -torch.inf).softmax(-1), values
    ).reshape(batch * query_length, hq, head_dim)


def build_graph(draft_attention, env, buffers, geometry, batch, cap, device, expected):
    """Construct, check and capture an arm; return its graph and live resources."""
    import torch

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        sizes = draft_attention.workspace_size(
            batch_size=batch, max_kv_tokens=cap, device=device, **geometry
        )
        workspace = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
        output = torch.empty_like(buffers.q)
        op = draft_attention(
            *workspace,
            request_indices=torch.arange(batch, dtype=torch.int32, device=device),
            positions=buffers.positions,
            token_indptr=buffers.token_indptr,
            token_indices=buffers.indices,
            query_indptr=torch.empty(batch + 1, dtype=torch.int32, device=device),
            page_indptr=torch.empty(batch + 1, dtype=torch.int32, device=device),
            page_indices=torch.empty(batch * cap, dtype=torch.int32, device=device),
            last_page_lengths=torch.empty(batch, dtype=torch.int32, device=device),
            **geometry,
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    op.prepare()
    implementation = op._implementation
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(buffers.num_layers):
            op.run(
                q=buffers.q, paged_kv_cache=(buffers.keys[index], buffers.values[index]), out=output
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    info = {
        "provider": op.provider,
        "err": (output.float() - expected).abs().max().item(),
        "tune": {
            key: getattr(implementation, key)
            for key in (
                "splits",
                "key_tile",
                "short_union",
                "merge_mode",
                "chunk_cap",
                "pieces",
                "inline_merge",
            )
            if hasattr(implementation, key)
        },
    }
    info.update({key: getattr(op, key) for key in ("pdl", "merge") if hasattr(op, key)})
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(buffers.num_layers):
            op.run(
                q=buffers.q, paged_kv_cache=(buffers.keys[index], buffers.values[index]), out=output
            )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, info, (op, workspace, output)


def time_pairs(graphs, names, times, pairs, num_layers, rng):
    """Append paired samples, randomizing arm order for every pair."""
    import torch

    for _ in range(pairs):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            samples = []
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0 / num_layers)
            times[name].append(median(samples))


def benchmark_case(arms, batch, ctx, mask, args, heads, device, rng):
    import torch

    query_length, causal, window, layers = MASKS[mask]
    buffers = setup_buffers(batch, ctx, query_length, heads, layers, args.cap, device)
    expected = reference(buffers, batch, ctx, query_length, heads, causal, window, device)
    hq, hk, head_dim = heads
    geometry = {
        "query_length": query_length,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": head_dim,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
    }
    names = list(arms)
    times = {name: [] for name in names}
    info = {}
    for round_index in range(args.rounds):
        graphs, keep = {}, []
        order = names[:]
        rng.shuffle(order)
        for name in order:
            draft_attention, env = arms[name]
            graph, info[name], resources = build_graph(
                draft_attention, env, buffers, geometry, batch, args.cap, device, expected
            )
            graphs[name] = graph
            keep.append(resources)
        first_pair = len(times[names[0]])
        time_pairs(graphs, names, times, args.pairs // args.rounds, buffers.num_layers, rng)
        for pair_index in range(first_pair, len(times[names[0]])):
            emit(
                pair_record(
                    mask,
                    batch,
                    ctx + query_length,
                    round_index,
                    pair_index,
                    {name: times[name][pair_index] for name in names},
                )
            )
        del graphs, keep, graph, resources
    result = {
        "n": buffers.num_layers,
        "info": info,
        "pairs": len(times[names[0]]),
        "rounds": args.rounds,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    candidate = times[names[0]]
    result["us"] = {name: median(times[name]) for name in names}
    for name in names[1:]:
        middle, low, high = bootstrap(
            [elapsed / cand for elapsed, cand in zip(times[name], candidate, strict=False)]
        )
        result["vs_" + name] = {
            "speed_median": middle,
            "ci_low": low,
            "ci_high": high,
            "not_slower": high >= 1.0,
        }
    if "base" in times and "base2" in times:
        # A/A noise floor: two copies of the same baseline tree.
        middle, low, high = bootstrap(
            [first / second for first, second in zip(times["base"], times["base2"], strict=False)]
        )
        result["aa_base2_vs_base"] = {
            "speed_median": middle,
            "ci_low": low,
            "ci_high": high,
            "not_slower": high >= 1.0,
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name=path[@ENV=V+...]; first arm is the candidate",
    )
    parser.add_argument(
        "--masks",
        default="full,c4096,n2048",
        help="comma-separated masks; q3moe/q3moe_causal use Q16, 32/4 heads, hd128",
    )
    parser.add_argument("--batches", default="1,2,3,4")
    parser.add_argument("--ctxs", default="128,512,4096")
    parser.add_argument("--lens", default=None, help="kv lengths L (ctx = L - Q)")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--heads", default="32,8,128")
    parser.add_argument("--label", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cores = pin_cores()
    import torch

    arms = {}
    for spec in args.arm:
        name, path, env = parse_arm(spec)
        arms[name] = (load_arm(name, path), env)
    heads = tuple(map(int, args.heads.split(",")))
    emit(
        {
            "type": "host",
            "gpu": torch.cuda.get_device_name(0),
            "cap": torch.cuda.get_device_capability(0),
            "x925": cores,
            "label": args.label,
            "arms": args.arm,
        }
    )
    rng = random.Random(7)
    device = torch.device("cuda")
    for mask in args.masks.split(","):
        query_length, causal, window, _ = MASKS[mask]
        case_heads = mask_heads(mask, heads)
        for batch in map(int, args.batches.split(",")):
            for value in map(int, (args.lens or args.ctxs).split(",")):
                ctx = value - query_length if args.lens else value
                record = {
                    "type": "ab2",
                    "mask": mask,
                    "B": batch,
                    "ctx": ctx,
                    "Q": query_length,
                    "causal": causal,
                    "W": window,
                    "cap": args.cap,
                    "heads": list(case_heads),
                }
                try:
                    record.update(
                        benchmark_case(arms, batch, ctx, mask, args, case_heads, device, rng)
                    )
                    torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001 - retain a receipt for each failed case.
                    record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                emit(record)


if __name__ == "__main__":
    main()
