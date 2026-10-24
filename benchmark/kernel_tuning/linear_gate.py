# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Fused o_proj gate + Linear market acceptance and bounded forced-config probes.

Run from the repository root with --accept or --probe. SGLang's comparator is
loaded only from --sglang-file; no SGLang package import or vendored kernel is used.
The cooperative budget stops new shape/M work, never interrupts running work.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import math
import statistics
import sys
import time
import types
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.kernel_tuning.linear_smallm import (
    READ_GBPS,
    Instance,
    L2Flush,
    acceptance_copies,
    acceptance_verdict,
    accuracy,
    bf16_reduced_precision,
    bootstrap_speedup_ci,
    emit,
    hot_copies,
    kernels_per_replay,
    paired,
    parse_ms,
    percentile95,
    reference_and_condition,
    shipped_dispatch,
)

FASTEST_MARKET = "max(vLLM Inductor gate + F.linear, SGLang fused_sigmoid_mul + F.linear)"


def builtin_shapes():
    return {
        key: {
            "model": model,
            "name": "o_proj",
            "N": n,
            "K": k,
            "gate_stride": stride,
            "count_per_step": count,
        }
        for key, model, n, k, stride, count in (
            ("9b", "qwen3.5-9b", 4096, 4096, 10240, 8),
            ("27b", "qwen3.8-27b", 5120, 6144, 14336, 16),
            ("27b-tp2", "qwen3.8-27b-tp2", 5120, 3072, 7168, 16),
        )
    }


def load_sglang(path):
    """Execute only the supplied elementwise file, restoring all SGLang modules."""
    if path is None or not path.is_file():
        raise FileNotFoundError("SGLang comparator file is required")
    names = (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.jit",
        "sglang.kernels.jit.utils",
        "sglang.srt",
        "sglang.srt.utils",
    )
    previous = {k: v for k, v in sys.modules.items() if k == "sglang" or k.startswith("sglang.")}
    try:
        for name in list(previous):
            del sys.modules[name]
        for name in names:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, module)
        sys.modules["sglang.kernels.jit.utils"].is_arch_support_pdl = lambda *a, **kw: False
        sys.modules["sglang.srt.utils"].is_hip = lambda *a, **kw: False
        name = "sglang._gate_comparator"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.fused_sigmoid_mul
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(previous)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--accept", action="store_true")
    mode.add_argument("--probe", action="store_true")
    ap.add_argument("--model", choices=(*builtin_shapes(), "all"), default="all")
    ap.add_argument("--m", default="accept")
    ap.add_argument("--sglang-file", type=Path)
    ap.add_argument("--l2", choices=("cold", "engine", "hot"), default="cold")
    ap.add_argument("--pairs", type=int)
    ap.add_argument("--reps-per-sample", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument(
        "--configs",
        help="probe configs 'rows,tile_k,stages,splits[,warps,variant,prefetch];...' "
        "after the plain table row (default: heuristic candidates)",
    )
    ap.add_argument("--budget-seconds", type=float, default=1500)
    args = ap.parse_args(argv)
    try:
        args.m = parse_ms(args.m)
    except ValueError as exc:
        ap.error(str(exc))
    args.pairs = (20 if args.accept else 7) if args.pairs is None else args.pairs
    if min(args.pairs, args.reps_per_sample, args.max_candidates) < 1:
        ap.error("pairs, repetitions and candidates must be positive")
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        ap.error("budget must be finite and positive")
    if args.accept and args.pairs < 20:
        ap.error("acceptance requires at least 20 pairs")
    return args


def gated_shared_bytes(cfg, mtiles):
    if "gated" in inspect.signature(cfg.shared_bytes_for).parameters:
        return cfg.shared_bytes_for(mtiles, gated=True)
    stages = 1 if cfg.variant == "reg" else cfg.stages
    return cfg.shared_bytes_for(mtiles) + stages * 8 * mtiles * (cfg.tile_k + 8) * 2


def candidates(n, k, m, kind, limit=8, explicit=None):
    from ctikernels.linear import _dispatch as dispatch
    from ctikernels.linear.cute import mtiles_for

    default = next(
        (
            cfg
            for (lo, hi), cfg in dispatch.TUNED.get(kind, {}).get((n, k), {}).items()
            if lo <= m <= hi
        ),
        None,
    )
    if explicit:
        options = ([default] if default is not None else []) + [
            dispatch.parse_config(setting) for setting in explicit.split(";")
        ]
        return [
            cfg
            for cfg in dict.fromkeys(options)
            if gated_shared_bytes(cfg, mtiles_for(m)) <= 99 * 1024
        ][:limit]
    options = ([default] if default is not None else []) + (
        dispatch.bxq_configs(n, k, kind)
        if hasattr(dispatch, "bxq_configs")
        else dispatch.tune_configs(default or dispatch.Config(), n, k, kind)
    )
    options = [
        cfg for cfg in dict.fromkeys(options) if gated_shared_bytes(cfg, mtiles_for(m)) <= 99 * 1024
    ]
    selected = [default] if default in options else []
    # Cover register/shared variants and 1/2/4 splits before filling remaining slots.
    for splits in (1, 2, 4):
        for variant in ("smem", "reg"):
            cfg = next((c for c in options if c.variant == variant and c.splits == splits), None)
            if cfg is not None and cfg not in selected:
                selected.append(cfg)
    return list(dict.fromkeys(selected + options))[:limit]


def make_inputs(shape, max_m):
    n, k, stride = shape["N"], shape["K"], shape["gate_stride"]
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn(max_m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    gate = torch.randn(max_m, stride, device="cuda", dtype=torch.bfloat16, generator=gen) * 3
    gate[:, :4] = gate.new_tensor([100, -100, 100, -100])
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * k**-0.5
    return x, gate[:, :k], weight


def weight_rotations(weight, shape, l2, l2_bytes):
    n, k = shape["N"], shape["K"]
    if l2 == "hot":
        return [weight] * (hot_copies(n, k) - 1)
    copies = acceptance_copies(n, k, l2_bytes)
    if l2 == "engine":
        return [w for _ in range(copies - 1) for w in (weight.clone(), weight)]
    return [weight.clone() for _ in range(copies - 1)]


def compile_vllm(x, gate):
    """Prime symbolic M with a non-singleton input, keeping static inputs unmarked."""
    dynamic = torch.compile(lambda x, g: x * torch.sigmoid(g), fullgraph=True)
    static = torch.compile(lambda x, g: x * torch.sigmoid(g), fullgraph=True, dynamic=False)
    xx = x.new_zeros((2, x.shape[1]))
    gg = torch.empty_strided(
        (2, gate.shape[1]), gate.stride(), device=gate.device, dtype=gate.dtype
    )
    gg.zero_()
    torch._dynamo.mark_dynamic(xx, 0)
    torch._dynamo.mark_dynamic(gg, 0)
    dynamic(xx, gg)
    static(x, gate)
    return {"dynamic": dynamic, "static": static}


def capture_market(x, gate, weight, rotations, compiled, sglang):
    calls = {
        mode: (lambda xx, w, fn=fn: F.linear(fn(xx, gate), w)) for mode, fn in compiled.items()
    }
    calls["sglang"] = lambda xx, w: F.linear(
        sglang(xx, gate.view(xx.shape[0], -1, 256), inplace=False), w
    )
    # Capture both possible calibration graphs now: no captures after selection/timing.
    return {
        name: (
            Instance(name, call, x, weight, rotations),
            Instance(name, call, x, weight, rotations),
        )
        for name, call in calls.items()
    }


def medians(left, right, count, reps, flush):
    a, b = paired(left, right, count, reps, flush)
    return (
        statistics.median(a),
        statistics.median(b),
        statistics.median(aa / bb for aa, bb in zip(a, b, strict=True)),
    )


class CandidateCompileError(RuntimeError):
    """A forced candidate could not initialize/capture; only its type is reported."""


def measure_row(
    args, shape, m, x, gate, weight, rotations, compiled, sglang, flush, kind, device, cfg=None
):
    from ctikernels.linear.inference import Linear

    op = Linear(provider="auto" if args.accept else "cute", config=cfg)
    activation = (x.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    ref, condition = reference_and_condition(activation, weight)
    with bf16_reduced_precision(False):
        strict = F.linear(activation, weight)
    with bf16_reduced_precision(True):
        market = F.linear(activation, weight)
        # Every capture clears cuBLAS workspaces; capture CuTe before vendor graphs.
        try:
            ours = Instance("ours", lambda xx, w: op(xx, w, gate=gate), x, weight, rotations)
        except Exception as exc:
            if args.accept:
                raise
            raise CandidateCompileError(type(exc).__name__) from None
        graphs = capture_market(x, gate, weight, rotations, compiled, sglang)
    selection_pairs = args.pairs if args.accept else 5
    dynamic_us, static_us, _ = medians(
        graphs["dynamic"][0], graphs["static"][0], selection_pairs, args.reps_per_sample, flush
    )
    mode = "dynamic" if dynamic_us <= static_us else "static"
    vllm, sgl = graphs[mode][0], graphs["sglang"][0]
    vllm_us, sglang_us, _ = medians(vllm, sgl, selection_pairs, args.reps_per_sample, flush)
    faster = "vllm" if vllm_us <= sglang_us else "sglang"
    if ours.output.shape != strict.shape or ours.output.dtype != x.dtype:
        raise ValueError("Wrong dispatched output contract")
    baseline = strict if op.provider == "cute-gated" else market
    metrics = accuracy(ours.output, baseline, ref, condition, shape["K"])
    for name, instance in (("vllm", vllm), ("sglang", sgl)):
        metrics.update(
            {
                name + "_" + key: value
                for key, value in accuracy(
                    instance.output, baseline, ref, condition, shape["K"]
                ).items()
            }
        )
    # Select the market using independent head-to-head medians, then calibrate
    # its noise before the direct pairs. Report those same medians so selection
    # cannot change when the direct pairs' sampling noise differs.
    epsilon = 0.0
    if args.accept:
        left, right = graphs[mode if faster == "vllm" else "sglang"]
        a, b = paired(left, right, args.pairs, args.reps_per_sample, flush)
        epsilon = percentile95([abs(aa / bb - 1) for aa, bb in zip(a, b, strict=True)])
    timings, speedups = {}, {}
    for name, instance in (("vllm", vllm), ("sglang", sgl)):
        if args.accept or name == faster:
            a, b = paired(instance, ours, args.pairs, args.reps_per_sample, flush)
            speedups[name] = [aa / bb for aa, bb in zip(a, b, strict=True)]
            timings[name] = (
                statistics.median(a),
                statistics.median(b),
                statistics.median(speedups[name]),
            )
    _, ours_us, ratio = timings[faster]
    interval = {}
    if args.accept:
        ci_low, ci_high = bootstrap_speedup_ci(
            speedups[faster],
            seed=f"{shape['N']}:{shape['K']}:{shape['gate_stride']}:{m}:{args.l2}:{faster}",
        )
        interval = {
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_method": "bootstrap_median_percentile95_2000",
        }
    market_us = vllm_us if faster == "vllm" else sglang_us
    count = kernels_per_replay(ours)
    if count < 1:
        raise RuntimeError("CUDA profiler did not report graph kernels")
    n, k = shape["N"], shape["K"]
    sol = 2 * (n * k + 2 * m * k + m * n) / (READ_GBPS[kind] * 1e3)
    return dict(
        shape,
        kind="gate_accept" if args.accept else "gate_probe",
        device=device,
        device_class=kind,
        M=m,
        l2=args.l2,
        timing_method={
            "cold": "rotating_weights",
            "engine": "alternating_cold_hot_weights",
            "hot": "repeated_weight",
        }[args.l2],
        rotation_copies=len(rotations) + 1,
        pairs=args.pairs,
        self_pairs=args.pairs if args.accept else 0,
        selection_pairs=selection_pairs,
        reps_per_sample=args.reps_per_sample,
        provider=op.provider,
        launch_config=op.launch_config,
        run_id=args.run_id,
        kernels_per_call=count,
        logical_kernels_per_call=op.kernels_per_call,
        ours_us=ours_us,
        vllm_us=vllm_us,
        vllm_mode=mode,
        vllm_dynamic_us=dynamic_us,
        vllm_static_us=static_us,
        sglang_us=sglang_us,
        faster_comparator=faster,
        market_us=market_us,
        speedup_vs_vllm=timings["vllm"][2] if "vllm" in timings else None,
        speedup_vs_sglang=timings["sglang"][2] if "sglang" in timings else None,
        paired_speedup=ratio,
        ratio=ratio,
        epsilon=epsilon,
        epsilon_method="linear_p95_abs_self_ratio_minus_one" if args.accept else "none",
        verdict=acceptance_verdict(ratio, epsilon, metrics["correct"]),
        status="ok" if metrics["correct"] else "incorrect",
        sol_us=sol,
        sol_fraction=sol / ours_us,
        market_sol_fraction=sol / market_us,
        fastest_market=FASTEST_MARKET,
        market_bf16_reduced_precision=True,
        **interval,
        **metrics,
    )


def fused(shape, m):
    from ctikernels.linear.inference import Linear

    return Linear(provider="auto").gate_fused(shape["N"], shape["K"], shape["gate_stride"], m)


def run(args, sglang):
    from ctikernels.linear._dispatch import device_class

    started = time.monotonic()
    props = torch.cuda.get_device_properties(0)
    kind = device_class((props.major, props.minor), props.multi_processor_count)
    if kind not in READ_GBPS:
        raise ValueError("a measured device class is required")
    if args.accept and kind == "gb10" and args.pairs < 60:
        print(
            "Warning: GB10 acceptance needs --pairs >= 60; "
            f"this {args.pairs}-pair run cannot qualify gated CuTe rows.",
            file=sys.stderr,
        )
    device = torch.cuda.get_device_name(0)
    flush = None if args.l2 == "hot" else L2Flush(props.L2_cache_size)
    args.run_id = uuid.uuid4().hex
    prefix = "gate_accept" if args.accept else "gate_probe"
    emit({"kind": prefix + "_begin", "run_id": args.run_id, "device": device})
    rows, failures, skipped = 0, 0, []
    for key, shape in builtin_shapes().items():
        if args.model not in (key, "all"):
            continue
        inputs = None
        for m in args.m:
            if time.monotonic() - started >= args.budget_seconds:
                skipped.append({"model": shape["model"], "M": m})
                continue
            if inputs is None:
                inputs = make_inputs(shape, max(args.m))
            x, gate, weight = inputs
            rotations = weight_rotations(weight, shape, args.l2, props.L2_cache_size)
            compiled = compile_vllm(x[:m], gate[:m])
            configs = (
                [None]
                if args.accept
                else candidates(shape["N"], shape["K"], m, kind, args.max_candidates, args.configs)
            )
            if not configs:
                skipped.append({"model": shape["model"], "M": m, "reason": "no_fitting_candidates"})
            if args.accept and not fused(shape, m):
                # No fused table row: the engine keeps its own gate kernel + Linear,
                # i.e. the market path itself. Record it without timing.
                row = dict(
                    shape,
                    kind="gate_accept",
                    device=device,
                    device_class=kind,
                    M=m,
                    l2=args.l2,
                    run_id=args.run_id,
                    provider="unfused",
                    launch_config=None,
                    status="fallback",
                    verdict="FALLBACK",
                )
                emit(row)
                rows += 1
                print(f"# {shape['model']} M={m}: FALLBACK (no fused row)", flush=True)
                configs = []
            for cfg in configs:
                try:
                    row = measure_row(
                        args,
                        shape,
                        m,
                        x[:m],
                        gate[:m],
                        weight,
                        rotations,
                        compiled,
                        sglang,
                        flush,
                        kind,
                        device,
                        cfg,
                    )
                except CandidateCompileError as exc:
                    row = dict(
                        shape,
                        kind="gate_probe",
                        M=m,
                        l2=args.l2,
                        run_id=args.run_id,
                        device_class=kind,
                        status="compile_error",
                        config=asdict(cfg),
                        error=str(exc),
                    )
                emit(row)
                rows += 1
                failures += int(row.get("verdict") == "FAIL")
                print(
                    f"# {shape['model']} M={m}: {row.get('verdict', row['status'])} "
                    f"ratio={row.get('paired_speedup')} provider={row.get('provider')}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
            del rotations, compiled
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
    emit(
        {
            "kind": prefix + "_completion",
            "run_id": args.run_id,
            "rows": rows,
            "failures": failures,
            "complete": not skipped,
            "skipped": skipped,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return int(bool(failures or skipped))


def main(argv=None):
    args = parse_args(argv)
    try:
        sglang = load_sglang(args.sglang_file)
    except FileNotFoundError:
        prefix = "gate_accept" if args.accept else "gate_probe"
        emit(
            {
                "kind": prefix + "_completion",
                "complete": False,
                "rows": 0,
                "failures": int(args.accept),
                "status": "error" if args.accept else "skip",
                "reason": "missing_sglang_comparator",
            }
        )
        return int(args.accept)
    except Exception as exc:  # noqa: BLE001 -- loader errors must not expose local paths
        emit({"kind": "gate_error", "error": type(exc).__name__})
        return 1
    try:
        with (
            torch.inference_mode(),
            shipped_dispatch(),
            torch._dynamo.config.patch(cache_size_limit=512),
        ):
            return run(args, sglang)
    except Exception as exc:  # noqa: BLE001 -- no paths or environment values in diagnostics
        emit({"kind": "gate_error", "error": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
