# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Eager Prologue host time for 36 target and five draft layers sharing one operation.

Run with ``python -m benchmark.kernel_tuning.attention_prologue_host``. Timing includes
only the op call and its returned Q view; cache squeeze views and synchronization are outside timing.
The checksum covers all Q outputs and final caches from a fixed six-step sequence,
independent of --steps. This measures host submission, not GPU execution latency.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path

TARGET_LAYERS = 36
DRAFT_LAYERS = 5
STEP_TOKENS = (101, 64, 87, 70, 128, 66)
DRAFT_TOKENS = 64
# Qwen3-8B and Qwen3.5-9B full attention, matching attention_prologue_engine.VARIANTS.
GEOMETRIES = {"qwen3": (32, 8, 128, 128, False), "qwen35": (16, 4, 256, 64, True)}
PAGES = 2048
MAX_POSITION = 4096


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", choices=("bf16", "fp8"), default="bf16")
    result.add_argument("--masked", action="store_true")
    result.add_argument("--geometry", choices=GEOMETRIES, default="qwen3")
    result.add_argument(
        "--steps", type=int, default=100, help="measured steps after 20 warmup steps"
    )
    result.add_argument("--json", type=Path, metavar="PATH")
    return result


def build(args, device):
    from ctikernels.attention.inference import Prologue

    q_heads, kv_heads, head_dim, rotary_dim, zero_centered = GEOMETRIES[args.geometry]
    dtype = torch.bfloat16 if args.cache == "bf16" else torch.float8_e4m3fn
    op = Prologue(
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        eps=1e-6,
        device=device,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        zero_centered=zero_centered,
        cache_dtype=dtype,
        masked=args.masked,
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    layers = []
    for _ in range(TARGET_LAYERS + DRAFT_LAYERS):
        weights = [
            torch.nn.Parameter(
                (int(not zero_centered) + 0.1 * torch.randn(head_dim, generator=generator)).to(
                    device=device, dtype=torch.bfloat16
                ),
                requires_grad=False,
            )
            for _ in range(2)
        ]
        pools = [
            torch.zeros(PAGES, 1, kv_heads, head_dim, dtype=dtype, device=device) for _ in range(2)
        ]
        source = torch.randn(
            max(STEP_TOKENS), (q_heads + 2 * kv_heads) * head_dim, generator=generator
        ).to(device=device, dtype=torch.bfloat16)
        layers.append((*weights, *pools, source))
    inverse = (1e7 if zero_centered else 1e6) ** -(
        torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim
    )
    angles = torch.arange(MAX_POSITION, dtype=torch.float64)[:, None] * inverse
    table = torch.cat((angles.cos(), angles.sin()), -1).to(device=device, dtype=torch.float32)
    return op, layers, table


def step_metadata(device, tokens, step, masked):
    generator = torch.Generator(device="cpu").manual_seed(1000 + step)
    positions = torch.randint(MAX_POSITION, (tokens,), generator=generator)
    indices = torch.randperm(PAGES, generator=generator)[:tokens].to(torch.int32)
    if masked:
        indices[::7] = -1
    return positions.to(device), indices.to(device)


def run_step(op, layers, table, device, step, *, outputs=None):
    tokens = STEP_TOKENS[step % len(STEP_TOKENS)]
    target = (tokens, *step_metadata(device, tokens, step, op.masked))
    draft = (DRAFT_TOKENS, *step_metadata(device, DRAFT_TOKENS, 10000 + step, op.masked))
    times = []
    for index, (qw, kw, kc, vc, source) in enumerate(layers):
        count, positions, indices = target if index < TARGET_LAYERS else draft
        qkv = source[:count].clone()
        k_cache, v_cache = kc.squeeze(1), vc.squeeze(1)
        start = time.perf_counter_ns()
        q = op(qkv, qw, kw, table, positions, k_cache, v_cache, indices)
        times.append((time.perf_counter_ns() - start) / 1000)
        if outputs is not None:
            outputs.append(q)
    torch.cuda.synchronize(device)
    return times


def hash_tensor(digest, tensor):
    """Hash logical bytes, including FP8, without a NumPy dependency."""
    cpu = tensor.detach().contiguous().cpu()
    digest.update(ctypes.string_at(cpu.data_ptr(), cpu.numel() * cpu.element_size()))


def checksum(op, layers, table, device):
    digest = hashlib.sha256()
    for step in range(len(STEP_TOKENS)):
        outputs = []
        run_step(op, layers, table, device, step, outputs=outputs)
        for output in outputs:
            hash_tensor(digest, output)
    for _, _, k_cache, v_cache, _ in layers:
        hash_tensor(digest, k_cache)
        hash_tensor(digest, v_cache)
    return digest.hexdigest()


def summary(samples):
    values = sorted(samples)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p10": values[len(values) // 10],
        "p90": values[9 * len(values) // 10],
    }


def main(argv=None):
    global torch
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.steps < 1:
        arguments.error("--steps must be positive")
    torch = importlib.import_module("torch")
    device = torch.device("cuda", 0)
    with torch.no_grad():
        op, layers, table = build(args, device)
        op.warm_up(max(STEP_TOKENS))
        digest = checksum(op, layers, table, device)
        for step in range(20):
            run_step(op, layers, table, device, step)
        calls, steps = [], []
        for step in range(args.steps):
            times = run_step(op, layers, table, device, step)
            calls.extend(times)
            steps.append(sum(times[:TARGET_LAYERS]) / 1000)
    report = {
        "geometry": args.geometry,
        "cache": args.cache,
        "masked": args.masked,
        "provider": op.provider,
        "steps": args.steps,
        "checksum": digest,
        "device": torch.cuda.get_device_name(device),
        "us_per_call": summary(calls),
        "ms_per_step_36": summary(steps),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
