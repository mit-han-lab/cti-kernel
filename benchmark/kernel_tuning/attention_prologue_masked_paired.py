# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Valid-row cost of the prologue's masked KV store (PM1): masked and unmasked compile-time
variants of the same CuTe kernel source, every cache index valid.

    python -m benchmark.kernel_tuning.attention_prologue_masked_paired

Candidates are ``base`` (unmasked), ``base2`` (a separate unmasked Launcher for the A/A control),
and ``masked`` (masked=True). Optionally, ``--baseline path/to/_prologue_cute.py`` loads a legacy
unmasked kernel for base/base2, adapting its constructor to omit the trailing masked flag.

Each baseline has a separate Launcher and compilation cache. Each round re-allocates every
buffer (fresh placement), captures one CUDA graph per candidate of ``--calls`` chained calls over 8
layers' qkv and caches (PDL chains as in a model), and times the candidates' replays (median of 15)
in rotated order. Per row: the geometric mean of base/masked over the rounds with a 95% t-interval of
the log ratios, the A/A ratio, and eps = A/A half-width + |A/A - 1|; a row passes iff the lower
bound of base/masked >= 1 - eps. Run it in several fresh processes and pool the ``ratios`` of the JSON
rows: per-process code placement moves sub-µs rows by a few percent. GB10 event timestamps are
coarse: keep ``--calls`` at 256 so a graph spans hundreds of microseconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from functools import partial
from types import SimpleNamespace

import torch

from ctikernels.attention import _prologue

GEOMETRIES = {"hd256": (16, 4, (256, 64, True)), "hd128": (32, 8, (128, 128, False))}
LAYERS, SLOTS, POSITIONS = 8, 4096, 32768


def _module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_module(path, name):
    module = _module(path, name)

    def kernel(*key):
        assert key[-1] is False, "a legacy baseline must be unmasked"
        return module.Kernel(*key[:-1])

    return SimpleNamespace(Kernel=kernel)


def _launchers(device, kernels, variant, baseline=None):
    modules = {
        name: _legacy_module(baseline, f"prologue_{name}") if baseline else kernels
        for name in ("base", "base2")
    }
    launchers = {
        name: _prologue.Launcher(device, module, variant) for name, module in modules.items()
    }
    launchers["masked"] = partial(_prologue.Launcher(device, kernels, variant), masked=True)
    return launchers


def _inputs(device, geometry, tokens, cache):
    q_heads, kv_heads, (head_dim, rotary_dim, _) = GEOMETRIES[geometry]
    dtype = torch.bfloat16 if cache == "bf16" else torch.float8_e4m3fn
    width = (q_heads + 2 * kv_heads) * head_dim
    layers = []
    for _ in range(LAYERS):
        k_cache = torch.zeros(SLOTS, kv_heads, head_dim, device=device).to(dtype)
        layers.append((torch.randn(tokens, width, device=device, dtype=torch.bfloat16), k_cache,
                       torch.zeros_like(k_cache)))  # fmt: skip
    weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16) * 0.1
    table = torch.randn(POSITIONS, rotary_dim, device=device)
    positions = torch.randint(0, POSITIONS, (tokens,), device=device)
    indices = torch.randperm(SLOTS, device=device)[:tokens].to(torch.int32)
    scale = torch.full((1,), 0.5, device=device)
    return q_heads, kv_heads, layers, weight, table, positions, indices, scale


def _graph(launch, inputs, cache, calls):
    q_heads, kv_heads, layers, weight, table, positions, indices, scale = inputs
    extra = {} if cache == "bf16" else {"scales": (scale, scale), "cache": cache}

    def run():
        for call in range(calls):
            qkv, k_cache, v_cache = layers[call % LAYERS]
            launch(qkv, weight, weight, table, positions, k_cache, v_cache, indices, 1e-6,
                   q_heads, kv_heads, **extra)  # fmt: skip

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time(graph, calls, reps=15):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(times)


def _interval(ratios):
    logs = [math.log(r) for r in ratios]
    mean, half = statistics.mean(logs), 1.96 * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", help="optional legacy unmasked _prologue_cute.py")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument("--tokens", default="1,2,3,4,64,128,256,512")
    parser.add_argument("--geometries", default="hd256,hd128")
    parser.add_argument("--caches", default="bf16,e4m3")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    from ctikernels.attention import _prologue_cute  # needs nvidia-cutlass-dsl

    device = torch.device("cuda", torch.cuda.current_device())
    rows = []
    for geometry in args.geometries.split(","):
        variant = GEOMETRIES[geometry][2]
        launchers = _launchers(device, _prologue_cute, variant, args.baseline)
        names = list(launchers)
        for cache in args.caches.split(","):
            for tokens in map(int, args.tokens.split(",")):
                samples = {name: [] for name in names}
                for r in range(args.rounds):
                    inputs = _inputs(device, geometry, tokens, cache)
                    graphs = {n: _graph(launchers[n], inputs, cache, args.calls) for n in names}
                    for name in names[r % 3 :] + names[: r % 3]:
                        samples[name].append(_time(graphs[name], args.calls))
                    del graphs, inputs
                ratios = [b / n for b, n in zip(samples["base"], samples["masked"], strict=True)]
                aa = [b / b2 for b, b2 in zip(samples["base"], samples["base2"], strict=True)]
                ratio, low, high = _interval(ratios)
                aa_ratio, aa_low, aa_high = _interval(aa)
                eps = (aa_high - aa_low) / 2 + abs(aa_ratio - 1)
                row = {
                    "device": torch.cuda.get_device_name(device), "geometry": geometry,
                    "cache": cache, "tokens": tokens,
                    "us": {name: statistics.median(v) for name, v in samples.items()},
                    "base_over_new": ratio, "ci": [low, high], "aa": aa_ratio, "eps": eps,
                    "verdict": "pass" if low >= 1 - eps else "FAIL",
                    "ratios": ratios, "aa_ratios": aa,
                }  # fmt: skip
                rows.append(row)
                print(f"{geometry} {cache} T={tokens}: base {row['us']['base']:.3f} us, masked "
                      f"{row['us']['masked']:.3f} us, base/masked {ratio:.4f} [{low:.4f}, {high:.4f}] "
                      f"eps {eps:.4f} {row['verdict']}", flush=True)  # fmt: skip
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle)
    return rows


if __name__ == "__main__":
    main()
