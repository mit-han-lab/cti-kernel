#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CP2 whole path versus CP1/FI prepared-input kernels; cold-L2 ABAB graph replays."""

import argparse
import importlib.metadata
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120
from ctikernels.gdn.chunk_ws.prefill import PortPrefill

root = Path(__file__).resolve().parents[3]
if not (root / "tests/gpu/_gdn_reference.py").is_file():
    root = Path.cwd()
sys.path.insert(0, str(root))
from benchmark.kernel_tuning.gdn_prefill_port.cp1_bench import percentile  # noqa: E402
from benchmark.kernel_tuning.gdn_prefill_port.cp1_precision import prepare  # noqa: E402
from tests.gpu.test_gdn_port_prefill import make_case  # noqa: E402
from tests.modules._gdn_chunk_ws_reference import fi_state, flashinfer_kernel  # noqa: E402


def build(
    hk,
    hv,
    lengths,
    producer_registers=72,
    iket=False,
    with_fi=False,
    prefetch=2,
    sweep=False,
    q_lookahead=False,
    remap_y=False,
    compare_default=False,
):
    x, weights, meta, base = make_case(hk, hv, lengths)
    # Timing uses owned sequences throughout, with initial state reads.
    cu, slots, initial = meta
    slots.copy_(torch.arange(1, len(lengths) + 1, device=x.device, dtype=torch.int32))
    initial.fill_(True)
    history = base[1].clone()
    q, k, v, g, beta = prepare(x, weights, hk, meta, history)
    op = PortPrefill(
        hk,
        hv,
        x.device,
        producer_registers=producer_registers,
        iket=iket,
        prefetch=prefetch,
        cp=False,
        q_lookahead=q_lookahead,
        remap_y=remap_y,
    )
    cp1 = ChunkPrefillSm120(x.device, hk, hv)
    fused_pools = tuple(t.clone() for t in base)
    cp1_pool = base[0].clone()
    y = torch.empty(x.shape[0], hv * 128, dtype=torch.bfloat16, device=x.device)
    o = torch.empty_like(v)
    providers = {
        "fused": lambda: op(x, *weights.args(), *meta, *fused_pools, out=y),
        "cp1": lambda: cp1(q, k, v, g, beta, o, cp1_pool, slots, initial, cu, 128**-0.5),
    }
    pools = {"fused": fused_pools, "cp1": (cp1_pool,)}
    originals = {"fused": base, "cp1": (base[0],)}
    if with_fi:
        if importlib.metadata.version("flashinfer-python") != "0.7.0.post1":
            raise RuntimeError("--with-fi requires flashinfer-python 0.7.0.post1")
        fi = flashinfer_kernel()
        if fi is None:
            raise RuntimeError("FlashInfer delta_rule_prefill_dsl unavailable")
        fi_initial, indices = fi_state(base[0], slots.tolist(), initial.tolist())
        fi_pool, fi_out, alpha = fi_initial.clone(), torch.empty_like(v), g.exp()
        providers["fi"] = lambda: fi(
            fi_out, fi_pool, q, k, v, fi_initial, alpha, beta, cu, 128**-0.5, state_indices=indices
        )
        pools["fi"], originals["fi"] = (fi_pool,), (fi_initial,)

    if sweep:
        # Reuse the exact same inputs and CP1 graph for every setting. Each fused
        # closure owns its operation, output and mutable pools through capture.
        del providers["fused"], pools["fused"], originals["fused"]
        for pf in (2, 4):
            for budget in (56, 72):
                name = f"fused_pf{pf}_p{budget}"
                setting = PortPrefill(
                    hk,
                    hv,
                    x.device,
                    producer_registers=budget,
                    prefetch=pf,
                    iket=iket,
                    cp=False,
                    q_lookahead=q_lookahead,
                    remap_y=remap_y,
                )
                buffers = tuple(t.clone() for t in base)
                output = torch.empty_like(y)
                providers[name] = lambda setting=setting, buffers=buffers, output=output: setting(
                    x, *weights.args(), *meta, *buffers, out=output
                )
                pools[name], originals[name] = buffers, base

    if compare_default:
        # Same inputs, budget and capture protocol; only the two
        # producer experiments differ. Pair directly in this invocation.
        default_op = PortPrefill(
            hk,
            hv,
            x.device,
            producer_registers=producer_registers,
            prefetch=prefetch,
            cp=False,
        )
        default_pools = tuple(t.clone() for t in base)
        default_y = torch.empty_like(y)
        providers["port_default"] = lambda: default_op(
            x, *weights.args(), *meta, *default_pools, out=default_y
        )
        pools["port_default"], originals["port_default"] = default_pools, base

    def restore(name):
        for dst, src in zip(pools[name], originals[name], strict=True):
            dst.copy_(src)

    return providers, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--lengths", default="1024")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--producer-registers", type=int, default=72)
    parser.add_argument("--remap-y", action="store_true")
    parser.add_argument("--q-lookahead", action="store_true")
    parser.add_argument("--prefetch", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument(
        "--sweep", action="store_true", help="pair budgets 56/72 and prefetch 2/4 against CP1"
    )
    parser.add_argument(
        "--compare-default",
        action="store_true",
        help="also pair directly against the default producer schedule/layout",
    )
    parser.add_argument("--with-fi", action="store_true")
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(",")]
    if not lengths or min(lengths) < 1 or args.pairs < 1:
        parser.error("positive lengths and pairs required")
    providers, restore = build(
        args.hk,
        args.hv,
        lengths,
        args.producer_registers,
        with_fi=args.with_fi,
        prefetch=args.prefetch,
        q_lookahead=args.q_lookahead,
        remap_y=args.remap_y,
        sweep=args.sweep,
        compare_default=args.compare_default,
    )
    graphs = {}
    for name, launch in providers.items():
        for _ in range(3):
            restore(name)
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graphs[name] = graph
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))

    def sample(name):
        restore(name)
        flush.zero_()
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000

    calibration = {}
    for name in providers:
        pairs = [(sample(name), sample(name)) for _ in range(20)]
        calibration[name] = {
            "eps": percentile([abs(b / a - 1) for a, b in pairs], 0.95),
            "samples_us": pairs,
        }
    chunks = sum((length + 63) // 64 for length in lengths)

    def stats(values):
        return {
            "median_us": statistics.median(values),
            "p95_us": percentile(values, 0.95),
            "us_per_sequence_chunk": statistics.median(values) / chunks,
            "samples_us": values,
        }

    comparisons = {}
    fused_names = [name for name in providers if name.startswith("fused")]
    comparators = [name for name in providers if not name.startswith("fused")]
    for fused_name, name in itertools.product(fused_names, comparators):
        pairs = [(sample(fused_name), sample(name)) for _ in range(args.pairs)]
        aa, bb = zip(*pairs, strict=True)
        ratio = statistics.median([a / b for a, b in pairs])
        eps = max(calibration[fused_name]["eps"], calibration[name]["eps"])
        row = {
            "fused": stats(aa),
            "comparator": stats(bb),
            "ratio_fused_comparator": ratio,
            "eps": eps,
            "ratio_noise_band": [ratio * (1 - eps), ratio * (1 + eps)],
        }
        comparisons[f"{fused_name}/{name}"] = row
        print(
            f"{fused_name}/{name}: {ratio:.5f}, {statistics.median(aa):.3f} / "
            f"{statistics.median(bb):.3f} us, eps={eps:.5f}",
            flush=True,
        )
    print(
        "CP2_RESULT "
        + json.dumps(
            {
                "hk": args.hk,
                "hv": args.hv,
                "lengths": lengths,
                "producer_registers": args.producer_registers if not args.sweep else [56, 72],
                "q_lookahead": args.q_lookahead,
                "remap_y": args.remap_y,
                "prefetch": args.prefetch if not args.sweep else [2, 4],
                "sweep": args.sweep,
                "compare_default": args.compare_default,
                "sequence_chunks": chunks,
                "comparisons": comparisons,
                "self_pairs": calibration,
                "gpu": torch.cuda.get_device_name(),
                "correctness": "run test_gdn_port_prefill.py separately",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
