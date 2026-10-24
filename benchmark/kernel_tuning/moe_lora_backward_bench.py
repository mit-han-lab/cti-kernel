# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired routed-expert MoE LoRA training forward/backward timings (shared-A gate_up, the
Triton training path) against PEFT-style BF16 torch loops and FP32 autograd.

Run with --quick for smoke. Each JSON row reports all gradient errors, medians,
and the median paired torch/ours ratio. Routed EP rows are local partial outputs.
CUDA graphs are used for ours and the pre-indexed BF16 expert loops when possible;
the FP32 reference retains its routing synchronization and uses CUDA events.
"""

import argparse
import gc
import json
import random
import statistics

import torch
import torch.nn.functional as F

from ctikernels.moe._lora_weights import routed_slices
from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora
from ctikernels.moe.torch._lora_reference import fused_experts_autograd

MODELS = {"qwen3-30b-a3b": (128, 2048, 768), "qwen35-35b-a3b": (256, 2048, 512)}
FACTOR_NAMES = ("gate_up_a", "gate_up_b", "down_a", "down_b")
GRAD_NAMES = ("x", "routing_weights", *FACTOR_NAMES)


def positive_list(text):
    try:
        values = tuple(dict.fromkeys(int(v) for v in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen35-35b-a3b")
    parser.add_argument("--tokens", type=positive_list)
    parser.add_argument("--ranks", type=positive_list)
    parser.add_argument("--ep", choices=(1, 2), type=int, default=1)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--graphs", choices=("auto", "off"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.tokens = args.tokens or ((1, 8) if args.quick else (1, 8, 64, 512, 2048))
    args.ranks = args.ranks or ((16,) if args.quick else (16, 32, 64))
    if args.samples < 1:
        parser.error("--samples must be positive")
    return args


def bf16_experts(x, w13, w2, rw, factors, pairs_by_expert, top_k, scale):
    """PEFT-style expert loops: BF16 GEMMs, FP32 activation and routing arithmetic; gate and up
    share A (one shrink), B [2, E, I, r]."""
    a13, b13, a2, b2 = factors
    out = torch.zeros_like(x, dtype=torch.float32)
    anchor = sum(v.reshape(-1)[:0].float().sum() for v in (x, rw, *factors))
    for e, pairs in pairs_by_expert:
        tokens = pairs // top_k
        inp = x[tokens]
        base = inp @ w13[e].T
        z13 = inp @ a13[e].T
        delta = torch.cat([z13 @ b13[s][e].T for s in range(2)], dim=-1)
        c1 = (base.float() + scale * delta.float()).to(x.dtype)
        gate, up = c1.float().chunk(2, -1)
        c2 = (F.silu(gate) * up).to(x.dtype)
        weight = rw.flatten()[pairs, None]
        base2 = ((c2 @ w2[e].T).float() * weight).to(x.dtype)
        delta2 = (c2 @ a2[e].T) @ b2[e].T
        value = (base2.float() + scale * weight * delta2.float()).to(x.dtype)
        out = out.index_add(0, tokens, value.float())
    return (out + anchor).to(x.dtype)


def relative_errors(actual, reference, names):
    result = {}
    for name, got, ref in zip(names, actual, reference, strict=True):
        denominator = ref.float().norm()
        numerator = (got.float() - ref.float()).norm()
        error = (
            float(numerator / denominator)
            if denominator > 0
            else (0.0 if numerator == 0 else float("inf"))
        )
        result[name] = error
    return result


def make_runner(fn, graphs, before=None):
    # Autograd's allocator and lazy compilation both need eager warmup. ``before``
    # (untimed, outside any graph) restores inputs an implementation overwrites.
    before = before or (lambda: None)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            before()
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    if graphs:
        graph = torch.cuda.CUDAGraph()
        try:
            before()
            with torch.cuda.graph(graph):
                result = fn()
        except RuntimeError:
            # Keep diagnostics to a mode label; event timings still include all
            # launches and host gaps for implementations that cannot capture.
            return fn, "events"

        def replay():
            graph.replay()
            return result

        return replay, "graph"
    return fn, "events"


def paired_times(runners, samples, before=None):
    before = before or {}
    measurements = {name: [] for name in runners}
    rng = random.Random(17)
    events = {
        name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for name in runners
    }
    for _ in range(samples):
        order = list(runners)
        rng.shuffle(order)
        for name in order:
            start, stop = events[name]
            if name in before:
                before[name]()
            start.record()
            runners[name]()
            stop.record()
            stop.synchronize()
            measurements[name].append(start.elapsed_time(stop))
    return measurements


def run_row(model, tokens, rank, ep, samples, graphs):
    torch.manual_seed(71)
    global_e, h, i = MODELS[model]
    e, k = global_e // ep, 8
    rand = lambda *shape, scale=0.02: (torch.randn(shape, device="cuda") * scale).bfloat16()
    x = rand(tokens, h, scale=0.3)
    w13, w2 = rand(e, 2 * i, h, scale=h**-0.5), rand(e, h, i, scale=i**-0.5)
    factors = [
        rand(*shape) for shape in ((e, rank, h), (2, e, i, rank), (e, rank, i), (e, h, rank))
    ]
    ids = torch.randint(global_e, (tokens, k), device="cuda", dtype=torch.int32)
    ids[ids >= e] = -1
    rw = torch.rand((tokens, k), device="cuda")
    dy = rand(tokens, h, scale=0.3)
    scale = torch.tensor([0.7], device="cuda")
    host = ids.flatten().cpu()
    pairs = [
        (ex, torch.where(host == ex)[0].to("cuda")) for ex in host.unique().tolist() if 0 <= ex < e
    ]
    names = GRAD_NAMES
    functions = {}
    for mode in ("ours", "torch", "fp32"):
        dtype = torch.float32 if mode == "fp32" else torch.bfloat16
        inp = x.to(dtype).detach().clone().requires_grad_()
        weights = rw.detach().clone().requires_grad_()
        fs = [f.to(dtype).detach().clone().requires_grad_() for f in factors]
        base13, base2 = w13.to(dtype), w2.to(dtype)
        leaves = (inp, weights, *fs)

        def run(
            mode=mode, inp=inp, weights=weights, fs=fs, base13=base13, base2=base2, leaves=leaves
        ):
            if mode == "ours":
                lora = MoELoRAWeights(*fs)
                y = fused_experts_lora(inp, base13, base2, weights, ids, lora, scaling=scale)
            elif mode == "torch":
                y = bf16_experts(inp, base13, base2, weights, fs, pairs, k, scale)
            else:
                y = fused_experts_autograd(
                    inp, base13, base2, weights, ids, routed_slices(*fs), scale
                )
            return torch.autograd.grad(y, leaves, dy.to(y.dtype))

        functions[mode] = run
    expected = functions["fp32"]()
    errors = {
        mode: relative_errors(functions[mode](), expected, names) for mode in ("ours", "torch")
    }
    del expected
    runners, modes = {}, {}
    for name, fn in functions.items():
        runners[name], modes[name] = make_runner(fn, graphs and name != "fp32")
    measurements = paired_times(runners, samples)
    times = {name: statistics.median(values) for name, values in measurements.items()}
    ratios = [t / o for t, o in zip(measurements["torch"], measurements["ours"], strict=True)]
    max_error = max(errors["ours"].values())
    # JSON never emits nonstandard NaN/Infinity: failed numerical rows use null.
    finite_errors = {
        mode: {name: value if value < float("inf") else None for name, value in values.items()}
        for mode, values in errors.items()
    }
    return {
        "model": model,
        "kind": "routed",
        "T": tokens,
        "rank": rank,
        "ep": ep,
        "E_local": e,
        "H": h,
        "I": i,
        "top_k": k,
        "samples": samples,
        "time_ms": times,
        "timing": modes,
        "torch_over_ours": statistics.median(ratios),
        "grad_relative_error": finite_errors,
        "max_grad_relative_error": max_error if max_error < float("inf") else None,
        "correct": all(value <= 0.025 for value in errors["ours"].values()),
        "error_gate": 0.025,
    }


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for tokens in args.tokens:
            for rank in args.ranks:
                row = {"model": args.model, "kind": "routed", "T": tokens, "rank": rank}
                row["ep"] = args.ep
                if not args.dry_run:
                    gc.collect()
                    torch.cuda.empty_cache()
                    row = run_row(
                        args.model, tokens, rank, args.ep, args.samples, args.graphs == "auto"
                    )
                print(json.dumps(row, allow_nan=False), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    main()
