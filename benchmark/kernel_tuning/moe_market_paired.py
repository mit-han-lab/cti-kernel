# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired frozen MoE forward + input backward against market training paths.

Run with python -m benchmark.kernel_tuning.moe_market_paired --model qwen35
--unsloth-path /path/to/unsloth/kernels/moe. No third-party kernels are bundled.
Every replay captures a complete training step, including router/gate gradients.
SOL excludes routing, layout, intermediate traffic and recomputation, as in
moe_dx_paired. Ratios greater than one favor ours; failed sanity checks cannot pass.
A row passes only with valid paired results from BOTH market baselines; otherwise
it fails with "baseline_missing" naming each missing baseline and its reason.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import math
import os
import subprocess
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.moe_dx_paired import make_inputs as make_qwen35_inputs
from benchmark.kernel_tuning.moe_dx_paired import sol_floor as base_sol_floor
from ctikernels.moe import moe_block

DEFAULT_TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
MARKETS = ("torch_gmm", "unsloth")
MODELS = {"qwen35": (2048, 256, 512, 512), "qwen3": (2048, 128, 768, 0)}


def make_inputs(t, ep, seed, model):
    if model == "qwen35":
        return make_qwen35_inputs(t, ep, seed)
    h, e, ni, _ = MODELS[model]
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rn(shape, scale):
        return (torch.randn(shape, generator=gen, device="cuda") * scale).bfloat16()

    x, dy = rn((t, h), 1).requires_grad_(), rn((t, h), 1)
    weights = (
        rn((e, h), 2 / h**0.5),
        None,
        rn((e // ep, 2 * ni, h), 1 / h**0.5),
        rn((e // ep, h, ni), 1 / ni**0.5),
        None,
        None,
    )
    mapping = torch.full((e,), -1, dtype=torch.int32, device=x.device)
    mapping[: e // ep] = torch.arange(e // ep, dtype=torch.int32, device=x.device)
    return x, dy, weights, mapping


def sol_floor(device_name, *, model, ep, tokens, local_slots, distinct_local):
    h, _, ni, nis = MODELS[model]
    return base_sol_floor(
        device_name,
        tokens=tokens,
        hidden=h,
        intermediate=ni,
        shared_intermediate=nis // ep,
        local_slots=local_slots,
        distinct_local=distinct_local,
        backward=True,
    )


def silu_mul(gu):
    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


def shared(x, weights, out):
    _, sg, _, _, s13, s2 = weights
    if sg is not None:
        sh = F.linear(silu_mul(F.linear(x, s13)), s2)
        out = out + torch.sigmoid(F.linear(x.float(), sg.float())) * sh.float()
    return out.to(x.dtype)


def groups(record, mapping, e):
    """Fixed-size device grouping; remote slots follow all local slots."""
    local = mapping[record.long()].flatten().long()
    ids = torch.where(local >= 0, local, e)
    order = ids.argsort(stable=True)
    counts = torch.zeros(e + 1, dtype=torch.int32, device=record.device)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    return ids, order, counts


def torch_experts(x, w13, w2, record, mapping, probs, grouped_mm):
    e, slots = w13.shape[0], record.numel()
    ids, order, counts = groups(record, mapping, e)
    # Pad each local M to 16, with a fixed upper bound on storage. Offsets and
    # placements stay on-device; the trailing remote region has no GEMM group.
    starts = F.pad(counts.cumsum(0), (1, 0))[:-1]
    offs = ((counts[:e] + 15) // 16 * 16).cumsum(0, dtype=torch.int32)
    padded_starts = F.pad(offs, (1, 0)).long()
    pos = torch.arange(slots, device=x.device) + (padded_starts - starts)[ids[order]]
    live = ids[order] < e
    capacity = (slots + 15 * e + 31) // 16 * 16
    xs = x.new_zeros((capacity, x.shape[1]))
    xs.index_copy_(0, pos, torch.where(live[:, None], x[order // record.shape[1]], 0))
    gu = grouped_mm(xs, w13.transpose(1, 2), offs=offs)
    # Unwritten tails may contain NaNs: mask before nonlinearities, not by *0.
    gu = torch.where((torch.arange(capacity, device=x.device) < offs[-1])[:, None], gu, 0)
    down = grouped_mm(silu_mul(gu), w2.transpose(1, 2), offs=offs)
    values = torch.where(live[:, None], down[pos], 0).float()
    values = values * probs.flatten()[order, None]
    # FP32 atomics avoid BF16 running-sum error; no weight gradients are built.
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    return out.index_add_(0, order // record.shape[1], values)


def unsloth_experts(x, w13, w2, record, mapping, probs, grouped_gemm):
    e, k = w13.shape[0], record.shape[1]
    ids, order, counts = groups(record, mapping, e)
    # The external backward checks dX_only, not W.requires_grad. Explicitly
    # suppress dW for the strongest frozen-weight training form.
    kw = {
        "m_sizes": counts[:e],
        "gather_indices": order,
        "topk": k,
        "autotune": True,
        "dX_only": True,
    }
    gu = grouped_gemm(x, w13, permute_x=True, is_first_gemm=True, **kw)
    gu = torch.where((ids[order] < e)[:, None], gu, 0)
    down = grouped_gemm(silu_mul(gu), w2, permute_y=True, is_first_gemm=False, **kw)
    down = torch.where((ids < e)[:, None], down, 0)
    return (down.view(x.shape[0], k, -1).float() * probs[..., None]).sum(dim=1)


def market_forward(x, weights, record, mapping, expert_fn, gemm):
    r, _, w13, w2, _, _ = weights
    # Selected-set softmax equals full softmax followed by top-k renormalization.
    # Keep it FP32 and in autograd, including remote logits in its denominator.
    probs = F.linear(x, r).float().gather(1, record.long()).softmax(dim=1)
    return shared(x, weights, expert_fn(x, w13, w2, record, mapping, probs, gemm))


def reference(x, dy, weights, record, mapping):
    """FP32 oracle with straight-through BF16 logits; chunked to bound memory."""
    r, sg, w13, w2, s13, s2 = weights
    outputs, gradients = [], []
    for lo in range(0, x.shape[0], 256):
        xr = x[lo : lo + 256].float().detach().requires_grad_()
        sel = record[lo : lo + 256].long()
        logits = F.linear(xr, r.float())
        logits = logits + (logits.bfloat16().float() - logits).detach()
        probs = logits.gather(1, sel).softmax(dim=1)
        local = mapping[sel]
        out = xr * 0
        for expert in range(w13.shape[0]):
            tok, slot = (local == expert).nonzero(as_tuple=True)
            if tok.numel():
                value = F.linear(
                    silu_mul(F.linear(xr[tok], w13[expert].float())), w2[expert].float()
                )
                out = out.index_add(0, tok, value * probs[tok, slot, None])
        if sg is not None:
            sh = F.linear(silu_mul(F.linear(xr, s13.float())), s2.float())
            out = out + torch.sigmoid(F.linear(xr, sg.float())) * sh
        dx = torch.autograd.grad(out, xr, dy[lo : lo + 256].float())[0]
        outputs.append(out.detach())
        gradients.append(dx)
    return torch.cat(outputs), torch.cat(gradients)


def sanity(got, want):
    errors = [
        ((g.float() - w).norm() / w.norm().clamp_min(1e-30)).item()
        for g, w in zip(got, want, strict=True)
    ]
    return {
        "out_error": errors[0] if math.isfinite(errors[0]) else None,
        "dx_error": errors[1] if math.isfinite(errors[1]) else None,
        "correct": all(math.isfinite(v) and v < 0.02 for v in errors),
    }


def load_unsloth(path):
    if path is None or not (path / "grouped_gemm" / "interface.py").is_file():
        raise ImportError("--unsloth-path must contain grouped_gemm/interface.py")
    sys.path.insert(0, str(path.resolve()))
    stub = types.ModuleType("unsloth")
    stub.DEVICE_TYPE = "cuda"
    sys.modules["unsloth"] = stub
    if importlib.util.find_spec("pandas") is None:
        # tuning.py uses pandas only inside unused KernelResult reporting methods.
        stub = types.ModuleType("pandas")
        stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
        sys.modules["pandas"] = stub
    # Leave TMA detection and OutOfResources config pruning to the external code.
    return importlib.import_module("grouped_gemm.interface").grouped_gemm


def exception_text(exc):
    return f"{type(exc).__name__}: {exc}"


def step(forward, x, dy):
    out = forward(x)
    return out.detach(), torch.autograd.grad(out, x, dy)[0]


def capture(forward, x, dy):
    xg = x.detach().clone().requires_grad_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(forward, xg, dy)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = step(forward, xg, dy)

    def replay(_xg=xg):
        # xg was allocated outside the graph pool: keep its storage alive.
        graph.replay()
        return result

    return replay


def sample(fn, k):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(k):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / k


def stats(values):
    v = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(v, torch.tensor([0.25, 0.5, 0.75], dtype=v.dtype)).tolist()
    return {"median_us": q[1], "iqr_us": q[2] - q[0], "samples_us": values}


def ratio_stats(a, b):
    ratios = torch.tensor(b, dtype=torch.float64) / torch.tensor(a, dtype=torch.float64)
    gen = torch.Generator().manual_seed(1729)
    draws = torch.randint(len(a), (2000, len(a)), generator=gen)
    medians = torch.quantile(ratios[draws], 0.5, dim=1)
    return {
        "median_ratio": torch.quantile(ratios, 0.5).item(),
        "ci95": torch.quantile(medians, torch.tensor([0.025, 0.975], dtype=medians.dtype)).tolist(),
    }


def prepare(forward, x, dy, want, graph_enabled, report):
    eager = lambda: step(forward, x, dy)
    report.update(sanity(eager(), want))
    if not report["correct"]:
        return {}
    # Complete all autotuning, including backward, before calibration or capture.
    for _ in range(3):
        eager()
    modes = {"eager": eager}
    if graph_enabled:
        try:
            replay = capture(forward, x, dy)
            report["graph_sanity"] = sanity(replay(), want)
            if report["graph_sanity"]["correct"]:
                modes["graph"] = replay
            else:
                report["graph_error"] = "captured training step failed correctness"
        except Exception as exc:  # noqa: BLE001 - record unsupported capture and retain eager
            report["graph_error"] = exception_text(exc)
    return modes


def measure(runners, anchor, pairs, min_sample_us, row):
    """One shared k per row; alternate AB/BA separately for every comparison."""
    failed = set()

    def timed(key, k):
        if key in failed:
            return None
        try:
            us = sample(runners[key], k)
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError("nonpositive or nonfinite event duration")
            return us
        except Exception as exc:  # noqa: BLE001 - retain other timings on runtime failure
            row[key[0]][f"{key[1]}_timing_error"] = exception_text(exc)
            failed.add(key)
            return None

    # Calibrate with batches to include launch throughput, not just isolated calls.
    pilot = [us for key in runners if (us := timed(key, 3)) is not None]
    if not pilot:
        return None
    fastest = min(pilot)
    k = max(1, math.ceil(1.3 * min_sample_us / fastest))
    for _ in range(3):
        durations = [us * k for key in runners if (us := timed(key, k)) is not None]
        if not durations:
            return None
        shortest = min(durations)
        if shortest >= 1.1 * min_sample_us:
            break
        k = math.ceil(k * 1.3 * min_sample_us / max(shortest, 0.001))
    row["calls_per_sample"] = k
    samples = {key: [] for key in runners}
    paired = {key: [] for key in runners if key != anchor}
    for p in range(pairs):
        if anchor in failed:
            anchor = None
        for key in runners:
            if key in failed or (key == anchor and len(runners) - len(failed) > 1):
                continue
            if anchor is None or key == anchor:
                if (us := timed(key, k)) is not None:
                    samples[key].append(us)
                continue
            order = (anchor, key) if p % 2 == 0 else (key, anchor)
            times = {name: timed(name, k) for name in order}
            if any(us is None for us in times.values()):
                continue
            samples[anchor].append(times[anchor])
            samples[key].append(times[key])
            paired[key].append(times[anchor])
    observed = [us * k for values in samples.values() for us in values]
    row["min_observed_sample_us"] = min(observed) if observed else None
    row["sample_duration_ok"] = bool(observed) and min(observed) >= min_sample_us
    for (path, mode), values in samples.items():
        if not values or (path, mode) in failed:
            continue
        report = row[path].setdefault("timings", {})
        report[mode] = stats(values)
        if (
            anchor is not None
            and anchor not in failed
            and (path, mode) != anchor
            and len(paired[path, mode]) == pairs
        ):
            report[mode].update(ratio_stats(paired[path, mode], values))
            report[mode]["paired_ours_us"] = paired[path, mode]
    return anchor if anchor not in failed else None


def baseline_failure(path, report, requested):
    if path not in requested:
        return "not requested"
    if "error" in report:
        return report["error"]
    if report.get("correct") is False:
        return "failed correctness"
    errors = [f"{key}: {value}" for key, value in report.items() if key.endswith("_timing_error")]
    return "; ".join(errors) or "no valid timing"


def qualify(row, anchor, requested):
    """Set row["passed"]: ours beats the faster market with CI_low > 1, and BOTH market
    baselines produced valid paired timings; otherwise list them in "baseline_missing"."""
    markets, missing = [], []
    for path in MARKETS:
        report = row.get(path, {})
        timings = report.get("timings", {})
        if timings:
            best = min(timings, key=lambda mode: timings[mode]["median_us"])
            report["best_mode"] = best
            markets.append((timings[best]["median_us"], path, timings[best]))
            if "ci95" not in timings[best]:
                missing.append(f"{path}: {best} timing has no complete paired comparison")
        else:
            missing.append(f"{path}: {baseline_failure(path, report, requested)}")
    row["passed"] = False
    if missing:
        # Both market baselines are required; never pass against the survivor alone.
        row["baseline_missing"] = missing
    elif markets:
        _, name, best = min(markets)
        row["market_best"] = name
        row["passed"] = (
            anchor is not None and best.get("ci95", [0])[0] > 1 and row["sample_duration_ok"]
        )


def benchmark_row(t, args, device_name):
    row = {
        "model": args.model,
        "T": t,
        "ep": args.ep,
        "pairs": args.pairs,
        "min_sample_us": args.min_sample_us,
        "graph_enabled": not args.no_graph,
        "seed": args.seed + t,
        "CTI_MOE_PROVIDER": os.environ.get("CTI_MOE_PROVIDER"),
        "save_activations": args.save_activations,
        "chunk_tokens": args.chunk_tokens,
        "passed": False,
    }
    x, dy, weights, mapping = make_inputs(t, args.ep, args.seed + t, args.model)
    try:
        with torch.no_grad():
            _, record = moe_block(x, *weights, top_k=8, expert_map=mapping)
        row["routing_source"] = "ours"
        ours_available = True
    except Exception as exc:  # noqa: BLE001 - market paths must survive unsupported providers
        row["ours"] = "unsupported" if args.model == "qwen3" else "error"
        row["ours_error"] = exception_text(exc)
        ours_available = False
        with torch.no_grad():
            record = (
                F.linear(x, weights[0])
                .float()
                .argsort(dim=1, descending=True, stable=True)[:, :8]
                .to(torch.uint8)
                .contiguous()
            )
        row["routing_source"] = "torch_stable_fallback"
    local = mapping[record.long()]
    row["local_slots"] = int((local >= 0).sum())
    row["distinct_local"] = local[local >= 0].unique().numel()
    row["sol_us"] = sol_floor(
        device_name,
        model=args.model,
        ep=args.ep,
        tokens=t,
        local_slots=row["local_slots"],
        distinct_local=row["distinct_local"],
    )
    want = reference(x, dy, weights, record, mapping)
    runners = {}
    for path in args.paths:
        if path == "ours" and not ours_available:
            continue
        report = row[path] = {}
        try:
            if path == "ours":
                forward = lambda xx: moe_block(
                    xx,
                    *weights,
                    top_k=8,
                    expert_map=mapping,
                    routing=record,
                    save_activations=args.save_activations,
                    chunk_tokens=args.chunk_tokens,
                )[0]
            else:
                gemm = (
                    load_unsloth(args.unsloth_path)
                    if path == "unsloth"
                    else getattr(F, "grouped_mm", None)
                )
                if gemm is None:
                    gemm = torch._grouped_mm
                expert_fn = unsloth_experts if path == "unsloth" else torch_experts
                forward = lambda xx, gemm=gemm, expert_fn=expert_fn: market_forward(
                    xx, weights, record, mapping, expert_fn, gemm
                )
            modes = prepare(forward, x, dy, want, not args.no_graph, report)
            runners.update({(path, mode): fn for mode, fn in modes.items()})
        except Exception as exc:  # noqa: BLE001 - one failed market path must not abort the row
            report["error"] = exception_text(exc)
    del want
    anchor = ("ours", "eager" if args.no_graph else "graph")
    if anchor not in runners:
        anchor = None
    if runners:
        anchor = measure(runners, anchor, args.pairs, args.min_sample_us, row)
    qualify(row, anchor, args.paths)
    if anchor is not None and row["sol_us"] is not None:
        row["sol_percent"] = 100 * row["sol_us"] / row["ours"]["timings"][anchor[1]]["median_us"]
    return row


def unsloth_commit(path):
    if path is not None:
        for parent in (path.resolve(), *path.resolve().parents):
            if (parent / ".git").exists():
                result = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return result.stdout.strip() if result.returncode == 0 else None
    return None


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen35")
    parser.add_argument("--ep", choices=(1, 2, 4), type=int, default=1)
    parser.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    parser.add_argument("--pairs", type=int, help="default 30; GB10 default/minimum 60")
    parser.add_argument("--min-sample-us", type=float, default=2500)
    parser.add_argument("--paths", default="ours,torch_gmm,unsloth")
    parser.add_argument("--unsloth-path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-graph", action="store_true")
    # Autograd market paths keep their gate/up activations; this is our equivalent.
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    args = parser.parse_args()
    try:
        tokens = [int(t) for t in args.tokens.split(",")]
    except ValueError:
        parser.error("--tokens must be comma-separated integers")
    args.paths = list(dict.fromkeys(args.paths.split(",")))
    if (
        min(tokens) < 1
        or (args.pairs is not None and args.pairs < 1)
        or not math.isfinite(args.min_sample_us)
        or args.min_sample_us <= 0
    ):
        parser.error("tokens, pairs and min-sample-us must be positive and finite")
    if not set(args.paths) <= {"ours", "torch_gmm", "unsloth"}:
        parser.error("--paths accepts ours,torch_gmm,unsloth")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for timing")
    import triton

    device_name = torch.cuda.get_device_name()
    args.pairs = max(args.pairs or 30, 60) if "GB10" in device_name else args.pairs or 30
    env = {
        "type": "environment",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": device_name,
        "CTI_MOE_PROVIDER": os.environ.get("CTI_MOE_PROVIDER"),
        "unsloth_commit": unsloth_commit(args.unsloth_path),
    }
    print(json.dumps(env, allow_nan=False), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for t in tokens:
        torch.cuda.reset_peak_memory_stats()
        try:
            with redirect_stdout(sys.stderr):
                row = benchmark_row(t, args, device_name)
        except Exception as exc:  # noqa: BLE001 - preserve partial runs and try the next geometry
            row = {
                "model": args.model,
                "T": t,
                "ep": args.ep,
                "passed": False,
                "error": exception_text(exc),
            }
        row["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        row["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        if args.out:
            with args.out.open("a") as output:
                output.write(line + "\n")
                output.flush()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
