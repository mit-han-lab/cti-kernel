#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GP-CP1: paired cold-L2 single-replay timings of chunk_ws (ours), optionally against FlashInfer.

Needs CuTe DSL 4.8.0; add --with-fi (flashinfer-python 0.7.0.post1) for FlashInfer. This file is a GPU
benchmark, not part of offline verification. --ncu emits CP1_ours / CP1_fi NVTX
ranges around one eager launch each after warmup, and never invokes a profiler.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
import sys
from pathlib import Path

import torch

from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120

# The independent reference lives in tests/ of the source tree.
reference_root = Path(__file__).resolve().parents[3]
if not (reference_root / "tests/modules/_gdn_chunk_ws_reference.py").is_file():
    reference_root = Path.cwd()  # flat script copy: cwd is the source root.
sys.path.insert(0, str(reference_root))
from tests.modules._gdn_chunk_ws_reference import (  # noqa: E402
    fi_state,
    flashinfer_kernel,
    inputs,
    reference,
    rms_rel,
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def clocks(device):
    if not hasattr(torch.cuda, "clock_rate"):
        return None
    try:
        return torch.cuda.clock_rate(device)
    except (ImportError, RuntimeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--lengths", default="4096,4096")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument(
        "--with-fi", action="store_true", help="also compare FlashInfer 0.7.0.post1"
    )
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) <= 0 or args.pairs <= 0:
        parser.error("positive lengths and pairs required")
    fi = None
    if args.with_fi:
        try:
            version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "0.7.0.post1":
            parser.error("--with-fi requires flashinfer-python 0.7.0.post1; use WITH_FI070=1")
        fi = flashinfer_kernel()
        if fi is None:
            parser.error("FlashInfer delta_rule_prefill_dsl is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    q, k, v, g, beta, base, cu = inputs(args.hk, args.hv, lengths, seed=args.seed, device=device)
    n = len(lengths)
    slots_host, initial_host = list(range(1, n + 1)), [True] * n
    slots = torch.tensor(slots_host, dtype=torch.int32, device=device)
    initial = torch.tensor(initial_host, dtype=torch.bool, device=device)
    pools, outputs, providers = {}, {}, {}
    scale = 128**-0.5

    def make_launch(op, out, pool):
        def launch():
            op(q, k, v, g, beta, out, pool, slots, initial, cu, scale)

        return launch

    pools["ours"], outputs["ours"] = base.clone(), torch.empty_like(v)
    op = ChunkPrefillSm120(device, args.hk, args.hv)
    providers["ours"] = make_launch(op, outputs["ours"], pools["ours"])
    if fi is not None:
        init_fi, indices = fi_state(base, slots_host, initial_host)
        pools["fi"], outputs["fi"] = init_fi.clone(), torch.empty_like(v)
        alpha = g.exp()

        def upstream():
            fi(
                outputs["fi"],
                pools["fi"],
                q,
                k,
                v,
                init_fi,
                alpha,
                beta,
                cu,
                scale,
                state_indices=indices,
            )

        providers["fi"] = upstream

    def restore(provider):
        pools[provider].copy_(init_fi if provider == "fi" else base)

    for _ in range(3):
        for name, launch in providers.items():
            restore(name)
            launch()
    torch.cuda.synchronize()
    heads = sorted({0, min(1, args.hv - 1), args.hv - 1})
    expected_o, expected_s = reference(
        q,
        k,
        v,
        g,
        beta,
        base,
        lengths,
        slots_host,
        initial_host,
        scale,
        heads=heads,
    )
    errors = {}
    for name in providers:
        diff = outputs[name].float() - outputs["ours"].float()
        errors[name] = {
            "ours_o_max_abs": diff.abs().max().item(),
            "ours_o_rms_abs": diff.square().mean().sqrt().item(),
            "fp64_o_rms_rel": rms_rel(outputs[name][:, heads], expected_o),
            "fp64_state_rms_rel": rms_rel(pools[name][slots_host][:, heads], expected_s),
        }
        assert all(math.isfinite(value) for value in errors[name].values()), errors[name]
    print("correctness " + json.dumps(errors), flush=True)
    if fi is not None:
        for kind in ("o", "state"):
            ours_error = errors["ours"][f"fp64_{kind}_rms_rel"]
            fi_error = errors["fi"][f"fp64_{kind}_rms_rel"]
            if not ours_error <= 1.02 * fi_error + 1e-6:
                raise AssertionError(f"CP1 {kind} accuracy gate failed: {ours_error} vs {fi_error}")
    chunks = sum((length + 63) // 64 for length in lengths)
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "clock_rate_mhz": clocks(device),
        "hk": args.hk,
        "hv": args.hv,
        "lengths": lengths,
        "chunks": chunks,
        "seed": args.seed,
        "pairs": args.pairs,
        "reference_heads": heads,
        "mode": "ncu" if args.ncu else "timing",
        "errors": errors,
    }
    if args.ncu:
        for name, launch in providers.items():
            restore(name)
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range(f"CP1_{name}"):
                launch()
            torch.cuda.synchronize()
        print("CP1_RESULT " + json.dumps(result), flush=True)
        return

    graphs = {}
    for name, launch in providers.items():
        restore(name)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()  # Exactly one kernel node per provider.
        graphs[name] = graph
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def sample(name):
        restore(name)
        flush.zero_()  # Both reset and L2 flush are outside the event interval.
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        return 1000.0 * start.elapsed_time(end)

    # A/A calibration estimates relative timing noise for each provider separately.
    calibration = {}
    for name in providers:
        pairs = [(sample(name), sample(name)) for _ in range(20)]
        ratios = [b / a for a, b in pairs]
        calibration[name] = {
            "pairs": 20,
            "median_ratio": statistics.median(ratios),
            "eps": percentile([abs(ratio - 1) for ratio in ratios], 0.95),
            "samples_us": pairs,
        }

    def timing(values):
        return {
            "median_us": statistics.median(values),
            "p95_us": percentile(values, 0.95),
            "us_per_chunk": statistics.median(values) / chunks,
            "samples_us": values,
        }

    comparisons = {}
    # Each provider gets its own ABAB stream, always paired with ours immediately.
    # Include ours/ours as a visible timing row even for an ours-only invocation.
    for name in providers:
        pairs = [(sample("ours"), sample(name)) for _ in range(args.pairs)]
        ours_samples, other_samples = zip(*pairs, strict=True)
        paired_ratios = [b / a for a, b in pairs]
        ratio = statistics.median(paired_ratios)
        eps = max(calibration["ours"]["eps"], calibration[name]["eps"])
        row = {
            "ours": timing(ours_samples),
            "other": timing(other_samples),
            "ratio_other_ours": ratio,
            "ratio_p95": percentile(paired_ratios, 0.95),
            "ratio_of_medians": statistics.median(other_samples) / statistics.median(ours_samples),
            "eps": eps,
            "ratio_noise_band": [ratio * (1 - eps), ratio * (1 + eps)],
        }
        comparisons[name] = row
        print(
            f"{name}/ours {ratio:.5f}; {name} {row['other']['median_us']:.3f} us; "
            f"ours {row['ours']['median_us']:.3f} us; A/A relative eps {eps:.5f}",
            flush=True,
        )
    result.update(
        {
            "comparisons": comparisons,
            "aa_calibration": calibration,
            "clock_rate_mhz_after": clocks(device),
        }
    )
    if fi is not None:
        result["ratio_ours_fi"] = 1.0 / comparisons["fi"]["ratio_other_ours"]
    print("CP1_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
