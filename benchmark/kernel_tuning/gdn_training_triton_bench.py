# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GDN training step time, our Triton provider versus fla-core (and CuTe where it resolves),
at Qwen3.5 GDN geometries and T in {1k, 4k, 16k}.

One step is the provider's forward plus its backward of ``projected``. The providers run on
the same GPU and inputs, interleaved A-B-A: per row, ``--pairs`` rounds, each timing every
provider with CUDA events over ``--reps`` steps after warm-up. Prints one ``ROW`` JSON line per
(geometry, workload) with each provider's median ms, the speedups (baseline / triton) and each
provider's fraction of the step's speed of light: max(essential bytes / measured bandwidth,
chunk matmul FLOPs / measured BF16 tensor peak). Essential bytes: projected read by the forward
and the backward, y written, dy read, d projected written. FLOPs per 64-token chunk and value
head (C = 64, D = 128): 10 C^2 D + 6 C D^2 forward, 24 C^2 D + 12 C D^2 backward.

    python benchmark/kernel_tuning/gdn_training_triton_bench.py [--pairs 9] [--reps 5]
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from ctikernels.gdn import _cute, _fla
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.triton.training import Training

# Qwen3.5-9B / 35B-A3B (16, 32), 27B (16, 48), each at TP1 and TP2.
GEOMETRIES = [(16, 32), (16, 48), (8, 16), (8, 24)]
WORKLOADS = {"single-1k": [1024], "single-4k": [4096], "single-16k": [16384],
             "uniform-8x1k": [1024] * 8}  # fmt: skip


def _inputs(geometry: Geometry, lengths: list[int], device):
    g = torch.Generator(device=device).manual_seed(0)
    t = sum(lengths)
    rand = lambda *s: torch.randn(*s, device=device, generator=g)
    weights = (
        (0.5 * rand(geometry.conv_dim, 4)).to(torch.bfloat16),
        torch.rand(geometry.v_heads, device=device, generator=g) * 2.8 - 0.7,
        torch.rand(geometry.v_heads, device=device, generator=g) * 4 - 2,
        1 + 0.1 * rand(128),
    )
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    projected = rand(t, geometry.proj_width).to(torch.bfloat16)
    dy = rand(t, geometry.value_width).to(torch.bfloat16)
    return projected, weights, cu.to(device), dy


def _time(kernel, geometry, args, reps: int) -> float:
    projected, weights, cu, dy = args

    def step():
        _, saved = kernel.train_forward(geometry, projected, *weights, cu)
        kernel.train_backward(geometry, saved, dy)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def _bandwidth(device) -> float:
    """Measured device-to-device copy bandwidth, bytes/s (read + write)."""
    x = torch.empty(1 << 28, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        y.copy_(x)
    end.record()
    end.synchronize()
    return 2 * x.numel() * 4 * 10 / (start.elapsed_time(end) / 1e3)


def _tensor_peak(device) -> float:
    """Measured BF16 matmul throughput, FLOP/s."""
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        a @ a
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        a @ a
    end.record()
    end.synchronize()
    return 2 * 8192**3 * 10 / (start.elapsed_time(end) / 1e3)


def sol_ms(geometry: Geometry, lengths: list[int], bandwidth: float, peak: float) -> float:
    t, c, d = sum(lengths), 64, 128
    chunks = sum((n + c - 1) // c for n in lengths)
    io = 2 * (2 * t * geometry.proj_width + 2 * t * geometry.value_width)  # BF16
    flops = chunks * geometry.v_heads * (34 * c * c * d + 18 * c * d * d)
    return 1e3 * max(io / bandwidth, flops / peak)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--geometries", default=";".join(f"{h},{v}" for h, v in GEOMETRIES))
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    providers = {"triton": Training(torch.cuda.get_device_capability(device))}
    try:
        providers["fla-core"] = _fla.load("GatedDeltaNetTraining", device)
    except RuntimeError as missing:  # not installed, or refused on this device
        print("no fla-core:", missing, flush=True)
    cute = _cute.training(device)
    if cute is not None:
        providers["cute"] = cute
    bandwidth, peak = _bandwidth(device), _tensor_peak(device)
    print(f"SOL basis: {bandwidth / 1e12:.2f} TB/s, {peak / 1e12:.0f} TFLOP/s BF16", flush=True)
    for hk, hv in (map(int, g.split(",")) for g in args.geometries.split(";")):
        geometry = Geometry(hk, hv)
        for name in args.workloads.split(","):
            inputs = _inputs(geometry, WORKLOADS[name], device)
            for kernel in providers.values():
                _time(kernel, geometry, inputs, 2)  # compile and warm up
            times = {p: [] for p in providers}
            order = list(providers)
            for i in range(args.pairs):
                for p in order if i % 2 == 0 else order[::-1]:  # A-B-A
                    times[p].append(_time(providers[p], geometry, inputs, args.reps))
            ms = {p: statistics.median(v) for p, v in times.items()}
            speedup = {p: ms[p] / ms["triton"] for p in ms if p != "triton"}
            sol = sol_ms(geometry, WORKLOADS[name], bandwidth, peak)
            row = {"device": torch.cuda.get_device_name(device), "geometry": [hk, hv],
                   "workload": name, "ms": ms, "speedup_vs": speedup, "sol_ms": sol,
                   "sol_fraction": {p: sol / ms[p] for p in ms}}  # fmt: skip
            print("ROW", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
