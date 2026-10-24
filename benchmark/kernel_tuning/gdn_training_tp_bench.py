# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GatedDeltaNetTraining step time, CuTe versus fla-core, at the TP-local geometries.

One training step is the forward under grad mode plus the backward of ``projected``. Both
providers run interleaved on the same GPU and inputs: per row, ``--pairs`` pairs in
alternating order, each sample timed with CUDA events over ``--reps`` steps. Rows follow the
gdn-train request grid (single, uniform 1024-token and ragged lognormal batches). Prints one
``ROW`` JSON line per (geometry, workload) with the median ratio fla-core / CuTe.

    python benchmark/kernel_tuning/gdn_training_tp_bench.py [--pairs 15] [--reps 3]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from ctikernels.gdn import GatedDeltaNetTraining, _cute, training

K = V = 128
GEOMETRIES = [(8, 16), (8, 24), (4, 8), (4, 12)]


def _lengths(name: str) -> list[int]:
    if name.startswith("single"):
        return [int(name.split("-")[1])]
    if name.startswith("uniform"):
        return [1024] * (int(name.split("-")[1]) // 1024)
    total = int(name.split("-")[1])  # ragged lognormal, fixed seed
    g = torch.Generator().manual_seed(7)
    raw = torch.empty(64).log_normal_(math.log(total / 8), 0.8, generator=g).tolist()
    out, used = [], 0
    for x in raw:
        n = max(1, min(int(x), total - used))
        out.append(n)
        used += n
        if used == total:
            break
    if used < total:
        out.append(total - used)
    return out


WORKLOADS = ["single-1024", "single-4097", "single-16384", "uniform-8192", "ragged-8192"]


def _inputs(hk: int, hv: int, lengths: list[int], device):
    conv_dim = 2 * hk * K + hv * V
    p = conv_dim + hv * V + 2 * hv
    t = sum(lengths)
    g = torch.Generator(device=device).manual_seed(0)
    projected = torch.randn(t, p, device=device, generator=g).to(torch.bfloat16)
    weights = (
        (0.5 * torch.randn(conv_dim, 4, device=device, generator=g)).to(torch.bfloat16),
        torch.empty(hv, device=device).uniform_(0.0, 2.7, generator=g),
        torch.randn(hv, device=device, generator=g),
        torch.ones(V, device=device),
    )
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    dy = torch.randn(t, hv * V, device=device, generator=g).to(torch.bfloat16)
    return projected, weights, cu.to(device), max(lengths), dy


def _op(hk: int, hv: int, provider: str, device) -> GatedDeltaNetTraining:
    saved = training._cute_geometry, _cute.training, training._triton
    try:
        if provider == "cute":
            training._cute_geometry = lambda geometry, device: True
        else:
            _cute.training = training._triton = lambda device: None
        op = GatedDeltaNetTraining(num_k_heads=hk, num_v_heads=hv, device=device)
    finally:
        training._cute_geometry, _cute.training, training._triton = saved
    assert op.provider == provider, (op.provider, provider)
    return op


def _step(op, projected, weights, cu, longest, dy):
    leaf = projected.detach().requires_grad_(True)
    y = op(leaf, *weights, cu, max_sequence_length=longest)
    (grad,) = torch.autograd.grad(y, leaf, dy)
    return grad


def _time(fn, reps: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--geometries", default=",".join(f"{a}x{b}" for a, b in GEOMETRIES))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    args = ap.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    print("GPU " + json.dumps({"name": torch.cuda.get_device_name(device)}), flush=True)
    for geometry in args.geometries.split(","):
        hk, hv = map(int, geometry.split("x"))
        ops = {name: _op(hk, hv, name, device) for name in ("cute", "fla-core")}
        chosen = GatedDeltaNetTraining(num_k_heads=hk, num_v_heads=hv, device=device).provider
        for workload in args.workloads.split(","):
            lengths = _lengths(workload)
            case = _inputs(hk, hv, lengths, device)
            for op in ops.values():  # compile and warm
                for _ in range(2):
                    _step(op, *case)
            torch.cuda.synchronize()
            times = {name: [] for name in ops}
            for pair in range(args.pairs):
                order = list(ops) if pair % 2 == 0 else list(ops)[::-1]
                for name in order:
                    times[name].append(_time(lambda op=ops[name], c=case: _step(op, *c), args.reps))
            ratios = [f / c for f, c in zip(times["fla-core"], times["cute"], strict=True)]
            row = {
                "geometry": [hk, hv],
                "workload": workload,
                "T": sum(lengths),
                "N": len(lengths),
                "chosen": chosen,
                "cute_ms": statistics.median(times["cute"]),
                "fla_ms": statistics.median(times["fla-core"]),
                "fla_over_cute": statistics.median(ratios),
            }
            print("ROW " + json.dumps(row), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
