# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""BF16 M<=64 linear market sweep; no CUDA activity at import time.

Weights are contiguous [N,K]; inputs [M,K]; outputs BF16 [M,N]. In market mode,
each provider captures one call into a separate CUDA graph/pool. A sample is the
median of --reps-per-sample single replays. In each pair, providers interleave
at replay granularity, alternating which goes first with pair + repetition.
Cold (default): read >=2x L2 bytes BEFORE recording each start event. Warm:
no flush. Twenty baseline self-pairs calibrate epsilon = linear p95 of
abs(A/B - 1). Ratios are medians of paired cublas_us/provider_us (higher wins).
When ours exists, also pair it directly against every comparator.

SOL_us = 2*(N*K + M*K + M*N) / BW_bytes_per_second * 1e6;
BW is max(measured read-only, copy) from the SOL ceilings. The faster ceiling
is an optimistic streaming envelope, counting weight, activation and output
bytes once. Unknown devices use best of ten 1 GiB copies (read plus write).
This is a weight-streaming reference, not measured traffic; utilization is unclipped.
Warm-cache utilization can exceed 1. p95 uses linear interpolation.

Plug-ins: register_provider('ours', callable(x, weight) -> output), or pass
--provider-hook package.module:function. The function receives the registry
and may add callables. ctikernels.linear.inference.Linear is discovered lazily;
"ours" measures shipped auto dispatch, including cuBLAS fallbacks. actual_provider
and logical_kernels_per_call distinguish those from CuTe timings.
QuACK uses gemm_interface.gemm(x, weight.T, tuned=False): heuristic selection,
no benchmark-time autotuning. Optional providers emit explicit skip/error rows.
Shape tables are TP=1 unless named TP2, with alternative GDN packing layouts;
counts for alternatives must NOT be added to get a model total.

--accept measures the shipped dispatcher (including cuBLAS fallback), with --pairs
interleaved pairs and baseline self-pairs (default 20; GB10 requires >=60).
Default M=1,2,3,4,6,8; --m bxq adds B*Q; --m accept adds every
bucket edge, which table qualification requires.
With --l2 cold (default), acceptance graphs rotate weight copies targeting >=3x L2 and >=1 GiB per graph,
bounded by 256 copies and 6 GB (at least two copies). This avoids L2 residency
and amortizes event-timer resolution of ~2 us on some nodes. Replay time and
profiled kernel counts are divided by copies to report per-call measurements.
--l2 hot calls one weight 16..256 times per graph (>= 512 MiB read) with no
flush, retaining the pairing/epsilon protocol. --l2 engine models consecutive
layer calls: after the flush, the graph alternates the cold copies with one
recurring weight (cold, hot, cold, hot, ...: hot whenever two weights fit in L2).
--step reports decode GEMM sums/deltas and B*Q round estimates. Acceptance SOL
uses the round-3 measured DRAM read ceilings: 1681.1/1533.1/244.6 GB/s on 5090/PRO 6000/GB10.
These modes default to one replay per sample; market mode defaults to five.
Acceptance profiles outside timing and returns nonzero for any correctness/parity FAIL.
Timed cuBLAS graphs use torch's default allow_bf16_reduced_precision_reduction
(True), as vLLM, SGLang and the engine call F.linear; the flag changes cuBLAS's
kernel choice (e.g. 24576x4096 at M>=43). CuTe rows are checked against a
False baseline, so the correctness gate is never loosened by a lower-precision
comparator; cuBLAS fallback rows against the True result (their own path).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

PROVIDERS: dict[str, Callable] = {}
BXQ_MS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 32, 36, 48, 64)
READ_GBPS = {"rtx5090": 1681.1, "pro6000": 1533.1, "gb10": 244.6}
MARKET_PROVENANCE = (
    "sm120/sm121: vLLM 0.22.1 linear.py:220-228 -> layers/utils.py:92-98,321-327; "
    "SGLang quantization/unquant.py:205-255 (CuTe only on SM10x); both F.linear"
)
FP32_MARKET_PROVENANCE = "BF16 operands: torch.mm(x, weight.t(), out_dtype=torch.float32)"
# torch's default, which vLLM, SGLang and the engine keep for F.linear.
MARKET_BF16_REDUCED_PRECISION = True


def accept_ms():
    """B x Q rows plus every dispatch bucket edge, which table qualification needs."""
    from ctikernels.linear._dispatch import M_BUCKETS

    return sorted({*BXQ_MS, *(m for bucket in M_BUCKETS for m in bucket)})


def parse_ms(value):
    """Decode and B x Q rows ("bxq"), those plus every bucket edge ("accept"), or a list in 1..64."""
    if value in ("bxq", "accept"):
        ms = list(BXQ_MS) if value == "bxq" else accept_ms()
    else:
        ms = list(dict.fromkeys(map(int, value.split(","))))
    if not ms or min(ms) < 1 or max(ms) > 64:
        raise ValueError("M must be in 1..64, bxq or accept")
    return ms


def sol_bandwidth(kind, measure):
    """Use the shared SOL loader when merged, with a small transitional table.

    Only the sustained DRAM term is used, not the full multi-resource/floor
    estimator: this benchmark explicitly reports compulsory-byte streaming SOL.
    """
    # measured sol-ceilings (read-only, copy), decimal GB/s
    copy_gbps = {"rtx5090": 1508, "pro6000": 1452, "gb10": 219}
    if kind not in READ_GBPS:
        return measure(), {"sol_source": "in_script_copy", "sol_bandwidth_policy": "copy"}
    read, copy = READ_GBPS[kind], copy_gbps[kind]
    source = "sol-ceilings built-in"
    try:
        sol = importlib.import_module("benchmark.sol.sol")
    except ModuleNotFoundError as exc:
        if exc.name not in ("benchmark.sol", "benchmark.sol.sol"):
            raise
    else:
        try:
            table = sol.load_ceilings("rtx-pro-6000" if kind == "pro6000" else kind)
        except FileNotFoundError:
            pass
        else:
            read, copy = (table["memory"][key] for key in ("dram_read_gbs", "dram_copy_gbs"))
            source = "benchmark.sol.sol"
    return max(read, copy) * 1e9, {
        "sol_source": source,
        "sol_read_GBps": read,
        "sol_copy_GBps": copy,
        "sol_bandwidth_policy": "max(read-only,copy)",
    }


def register_provider(name: str, call: Callable) -> None:
    """Register a BF16 F.linear-compatible callable, before main()."""
    if not callable(call):
        raise TypeError("Provider must be callable")
    PROVIDERS[name] = call


def builtin_shapes():
    """Calls per forward, TP1 unless named TP2.

    qwen3.5-27b is the TP1 Qwen3.8-27B geometry; TP2 is per rank.
    DFlash block rows use M=B*Q; fc/context_kv use context rows. The draft
    shares its target's lm_head, which is not counted again here.
    """
    # Official Qwen model config.json text_config dimensions, TP=1.
    # H, layers, attention heads, KV heads, GDN value heads, dense/shared FFN.
    configs = {
        "qwen3.5-9b": (4096, 32, 16, 4, 32, 12288),
        "qwen3.5-27b": (5120, 64, 24, 4, 48, 17408),
        "qwen3.5-35b-a3b-dense": (2048, 40, 16, 2, 32, 512),
    }
    models = {}
    for model, (h, layers, heads, kv, hv, ffn) in configs.items():
        gdn, attn = layers * 3 // 4, layers // 4
        qkvz = 2 * 16 * 128 + 2 * hv * 128
        rows = []

        def add(name, n, k, count, rows=rows, **metadata):
            rows.append(dict(name=name, N=n, K=k, count_per_step=count, **metadata))

        add("gdn.in_proj_qkvz", qkvz, h, gdn, layout="split")
        add("gdn.in_proj_ba", 2 * hv, h, gdn, layout="split")
        add("gdn.in_proj_qkvzba", qkvz + 2 * hv, h, gdn, layout="fused")
        add("gdn.out_proj", h, hv * 128, gdn)
        # Gated Q contributes 2*heads*head_dim; K and V each kv*head_dim.
        add("attention.qkv", (2 * heads + 2 * kv) * 256, h, attn)
        add("attention.o", h, heads * 256, attn)
        prefix = "shared_expert" if "a3b" in model else "mlp"
        add(prefix + ".gate_up", 2 * ffn, h, layers)
        add(prefix + ".down", h, ffn, layers)
        if "a3b" in model:
            add("router.gate", 256, h, layers)
            add("shared_expert.gate", 1, h, layers)
        add("lm_head", 248320, h, 1)
        models[model] = rows
    models["qwen3-8b"] = [
        {"name": name, "N": n, "K": k, "count_per_step": count}
        for name, n, k, count in (
            ("attention.qkv", 6144, 4096, 36),
            ("attention.o", 4096, 4096, 36),
            ("mlp.gate_up", 24576, 4096, 36),
            ("mlp.down", 4096, 12288, 36),
            ("lm_head", 151936, 4096, 1),
        )
    ]
    # Qwen3-30B-A3B (Qwen3-MoE, K-Q6): the replicated router and lm_head. Its
    # attention projections and routed experts are outside this inventory.
    models["qwen3-30b-a3b"] = [
        {"name": "router.gate", "N": 128, "K": 2048, "count_per_step": 48},
        {"name": "lm_head", "N": 151936, "K": 2048, "count_per_step": 1},
    ]
    models["qwen3.5-27b-tp2"] = [
        dict(name=name, N=n, K=k, count_per_step=count, **metadata)
        for name, n, k, count, metadata in (
            ("attention.qkv", 7168, 5120, 16, {}),
            ("attention.o", 5120, 3072, 16, {}),
            ("gdn.in_proj_qkvzba", 8240, 5120, 48, {"layout": "fused"}),
            ("gdn.out_proj", 5120, 3072, 48, {}),
            ("mlp.gate_up", 17408, 5120, 64, {}),
            ("mlp.down", 5120, 8704, 64, {}),
            ("lm_head", 124160, 5120, 1, {}),
        )
    ]
    for model, h, fc_k, layers, attn_k, ffn in (
        ("dflash-qwen3-8b", 4096, 20480, 5, 4096, 12288),
        ("dflash-qwen3.5-9b", 4096, 32768, 6, 4096, 12288),
        ("dflash-qwen3.5-27b", 5120, 40960, 6, 4096, 17408),
    ):
        models[model] = [
            {"name": name, "N": n, "K": k, "count_per_step": count, "row_domain": domain}
            for name, n, k, count, domain in (
                ("fc", h, fc_k, 1, "context"),
                ("context_kv", 2048, h, layers, "context"),
                ("attention.qkv", 6144, h, layers, "block"),
                ("attention.o", h, attn_k, layers, "block"),
                ("mlp.gate_up", 2 * ffn, h, layers, "block"),
                ("mlp.down", h, ffn, layers, "block"),
            )
        ]
    return models


def percentile95(values):
    return percentile(values, 0.95)


def percentile(values, fraction):
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_speedup_ci(speedups, seed):
    """95% percentile CI of median paired cuBLAS/ours ratios (2000 resamples);
    the shared implementation that warm-up autotune also uses."""
    from ctikernels.linear._dispatch import bootstrap_median_ci

    return bootstrap_median_ci(speedups, seed)


def emit(record):
    record = {"out_dtype": "bf16", **record}
    print("RESULT " + json.dumps(record, allow_nan=False, separators=(",", ":")), flush=True)


@contextmanager
def blas_library(name):
    previous = torch.backends.cuda.preferred_blas_library()
    try:
        torch.backends.cuda.preferred_blas_library(name)
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def discover_providers(hook=None):
    providers = {
        "cublas": lambda x, w: F.linear(x, w),
        "cublaslt": lambda x, w: F.linear(x, w),
    }
    unavailable = {}
    try:
        from quack.gemm_interface import gemm

        providers["quack"] = lambda x, w: gemm(x, w.t(), tuned=False)
    except Exception as exc:  # noqa: BLE001 -- optional dependency
        unavailable["quack"] = exception_reason(exc)
    try:
        linear = importlib.import_module("ctikernels.linear.inference")
        providers["ours"] = linear.Linear(provider="auto")
    except Exception as exc:  # noqa: BLE001 -- optional phase-2 provider
        unavailable["ours"] = exception_reason(exc)
    providers.update(PROVIDERS)
    if hook:
        module, function = hook.split(":", 1)
        getattr(importlib.import_module(module), function)(providers)
    if any(not callable(call) for call in providers.values()):
        raise TypeError("Provider registry values must be callable")
    return providers, unavailable


def exception_reason(exc):
    """Bound provider diagnostics in machine-readable RESULT rows."""
    return f"{type(exc).__name__}: {exc}"[:300]


class Instance:
    """One graph calling the op once per weight copy; timings are per call.

    rotations are extra weight copies (acceptance rotates >= 3x L2 so no weight
    stays cached, and long replays stay far above event-timer resolution).
    """

    def __init__(self, name, call, x, weight, rotations=()):
        library = "cublaslt" if name == "cublaslt" else "cublas"
        weights = (weight, *rotations)
        self.copies = len(weights)
        with blas_library(library):
            # Compile / initialize outside capture on a side stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call(x, weight)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                outputs = [call(x, w) for w in weights]
            self.output = outputs[0]
            for _ in range(3):
                self.graph.replay()
            torch.cuda.synchronize()


class L2Flush:
    """Read eviction buffer; only a scalar/reduction scratch is dirtied per flush."""

    def __init__(self, l2_bytes):
        if l2_bytes <= 0:
            raise ValueError("L2 cache size unavailable")
        self.buffer = torch.zeros(max(32 << 20, 2 * l2_bytes), device="cuda", dtype=torch.uint8)
        self.output = torch.empty((), device="cuda", dtype=torch.uint8)
        # Evict setup's dirty lines and warm the reduction outside measurements.
        self()
        self()
        torch.cuda.synchronize()

    @property
    def nbytes(self):
        return self.buffer.numel()

    def __call__(self):
        torch.amax(self.buffer, dim=0, out=self.output)


def reference_and_condition(x, w):
    """FP64 result and sum(abs(x*w)); chunk W to bound vocabulary-head memory."""
    xd = x.double()
    values, conditions = [], []
    for part in w.split(1024):
        wd = part.double()
        values.append(xd @ wd.t())
        conditions.append(xd.abs() @ wd.abs().t())
    return torch.cat(values, dim=1), torch.cat(conditions, dim=1)


def accuracy(y, baseline, ref, condition, k, *, out_dtype="bf16"):
    """Compare BF16 output to cuBLAS using three independent error gates.

    Allow an extra half BF16 ULP beyond max(cuBLAS, half ULP), with a
    condition-scale floor for cancellation. Also bound condition-normalized
    error and allow 5% RMS slack. No additive RMS tolerance: an exact baseline
    must stay exact.
    """
    if k <= 0:
        raise ValueError("reduction length K must be positive")
    err, base = (y.double() - ref).abs(), (baseline.double() - ref).abs()
    denom = condition.clamp_min(torch.finfo(torch.float64).tiny)
    absolute, base_absolute = err.max().item(), base.max().item()
    normalized, base_normalized = (err / denom).max().item(), (base / denom).max().item()
    rms, base_rms = err.square().mean().sqrt().item(), base.square().mean().sqrt().item()
    fp32_slack = 2 * torch.finfo(torch.float32).eps
    # Random-walk accumulation model: independent rounding perturbations grow
    # as sqrt(K). Anchor the condition-scale budget at K=1024 and never shrink
    # it below two eps32. This is a statistical allowance, not a worst-case
    # forward-error bound; normalized error and RMS remain independent gates.
    accumulation_floor = fp32_slack * max(1.0, math.sqrt(k / 1024))
    if out_dtype == "fp32":
        # No output rounding: gate condition-normalized error at the FP32
        # accumulation floor, max(2x cuBLAS, 4 eps32). Split-K chains meet it;
        # one 4096-long chain (split 1) reaches ~5.5 eps32 at M >= 33 and fails.
        # A missing chunk or fragment error stays orders of magnitude above.
        normalized_rms = (err / denom).square().mean().sqrt().item()
        base_normalized_rms = (base / denom).square().mean().sqrt().item()
        eps32 = torch.finfo(torch.float32).eps
        normalized_limit = max(2 * base_normalized, 4 * eps32)
        rms_limit = max(2 * base_normalized_rms, eps32)
        return {
            "correct": bool(torch.isfinite(y).all())
            and normalized <= normalized_limit
            and normalized_rms <= rms_limit,
            "max_abs": absolute,
            "cublas_max_abs": base_absolute,
            "max_normalized": normalized,
            "cublas_max_normalized": base_normalized,
            "max_normalized_limit": normalized_limit,
            "rms_normalized": normalized_rms,
            "cublas_rms_normalized": base_normalized_rms,
            "rms_normalized_limit": rms_limit,
            "rms": rms,
            "cublas_rms": base_rms,
        }
    ulp = torch.exp2(
        torch.floor(torch.log2(ref.abs().clamp_min(torch.finfo(torch.float64).tiny))) - 7
    )
    ulp = torch.maximum(ulp, denom * accumulation_floor)
    ulp_error, base_ulp_error = (err / ulp).max().item(), (base / ulp).max().item()
    ulp_limit = max(base_ulp_error, 0.5) + 0.5
    # A quarter of the upper bound on a BF16 ULP is 2^-9 * |ref|.
    # Normalize that allowance by sum|x*w| and retain two FP32 epsilons for
    # cancellation. Since |ref| <= sum|x*w|, the additive allowance is at most
    # 2^-9 + two epsilons, far below an O(1) missing-chunk/fragment error.
    normalized_slack = 2**-9 * (ref.abs() / denom).max().item() + fp32_slack
    normalized_limit = base_normalized * 1.5 + normalized_slack
    rms_limit = base_rms * 1.05
    ok = bool(torch.isfinite(y).all())
    ok = ok and ulp_error <= ulp_limit and rms <= rms_limit
    ok = ok and normalized <= normalized_limit
    return {
        "correct": ok,
        "ulp_condition_floor": accumulation_floor,
        "max_abs": absolute,
        "cublas_max_abs": base_absolute,
        "max_ulp_error": ulp_error,
        "cublas_max_ulp_error": base_ulp_error,
        "max_ulp_error_limit": ulp_limit,
        "max_normalized": normalized,
        "cublas_max_normalized": base_normalized,
        "max_normalized_limit": normalized_limit,
        "rms": rms,
        "cublas_rms": base_rms,
        "rms_limit": rms_limit,
    }


def timed(run, flush, events):
    if flush is not None:
        flush()
    start, end = events
    start.record()
    run()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000
    if not math.isfinite(us) or us <= 0:
        raise ValueError("Invalid event duration")
    return us


def paired(left, right, count, reps, flush):
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    samples = ([], [])
    runs = (left.graph.replay, right.graph.replay)
    copies = (getattr(left, "copies", 1), getattr(right, "copies", 1))
    for pair in range(count):
        rounds = ([], [])
        for rep in range(reps):
            for side in (0, 1) if (pair + rep) % 2 == 0 else (1, 0):
                rounds[side].append(timed(runs[side], flush, events) / copies[side])
        for side in (0, 1):
            samples[side].append(statistics.median(rounds[side]))
    return samples


def measure_bandwidth(flush):
    byte_count = 1 << 30
    src = torch.ones(byte_count, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    seconds = min(timed(lambda: dst.copy_(src), flush, events) for _ in range(10)) * 1e-6
    return 2 * byte_count / seconds


def kernel_names(instance):
    # Profiling is opt-in and outside all timing. No trace file is needed.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        instance.graph.replay()
        torch.cuda.synchronize()
    return sorted(
        {
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def run_shape(args, n, k, m, aliases, providers, unavailable, flush, bw, device):
    base = {
        "device": device,
        "N": n,
        "K": k,
        "M": m,
        "mode": "cold" if args.cold else "warm",
        "pairs": args.pairs,
        "reps_per_sample": args.reps_per_sample,
        "epsilon_method": "linear_p95_abs_self_ratio_minus_one",
        "ratio_definition": "cublas_us/provider_us",
        "SOL_us": 2 * (n * k + m * k + m * n) / bw * 1e6,
        "BW_GBps": bw / 1e9,
        "cublas_bf16_reduced_precision": MARKET_BF16_REDUCED_PRECISION,
    }
    records, instances = {}, {}
    # Shared immutable weights keep memory bounded even for the vocabulary head.
    x = torch.randn(max(getattr(args, "m", [m])), k, dtype=torch.bfloat16, device="cuda")[:m]
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    reference, condition = reference_and_condition(x, weight)
    # Accuracy is gated against full-precision cuBLAS; timing uses the market flag.
    with bf16_reduced_precision(False):
        full_precision = F.linear(x, weight)
    for name in args.providers:
        records[name] = dict(
            base,
            provider=name,
            status="ok",
            us_median=None,
            p95_us=None,
            ratio_vs_cublas=None,
            epsilon=None,
            util=None,
        )
        if name not in providers:
            records[name].update(status="skipped", reason=unavailable.get(name, "not_registered"))
            continue
        try:
            # Match acceptance even if a tuning override was left in the shell.
            with shipped_dispatch():
                instances[name] = Instance(name, providers[name], x, weight)
            if name == "ours":
                records[name].update(
                    actual_provider=getattr(providers[name], "provider", "custom"),
                    logical_kernels_per_call=getattr(providers[name], "kernels_per_call", None),
                )
            if name != "cublas":
                output, ref = instances[name].output, full_precision
                if output.dtype != torch.bfloat16 or output.shape != ref.shape:
                    raise ValueError("Wrong output contract")
                metrics = accuracy(output, ref, reference, condition, k)
                records[name].update(metrics)
                if not metrics["correct"]:
                    raise RuntimeError("output failed K-aware ULP, normalized-error or RMS gate")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one unsupported provider must not end sweep
            records[name].update(
                status="error", reason=exception_reason(exc), phase="capture_or_check"
            )
            instances.pop(name, None)
            if name == "cublas":
                raise RuntimeError("Baseline capture failed") from None
    calibration = Instance("cublas", providers["cublas"], x, weight)
    a, b = paired(instances["cublas"], calibration, 20, args.reps_per_sample, flush)
    epsilon = percentile95([abs(left / right - 1) for left, right in zip(a, b, strict=True)])
    samples = defaultdict(list)
    samples["cublas"].extend(a + b)
    records["cublas"]["ratio_vs_cublas"] = 1.0
    for name, instance in instances.items():
        if name == "cublas":
            continue
        a, b = paired(instances["cublas"], instance, args.pairs, args.reps_per_sample, flush)
        samples["cublas"].extend(a)
        samples[name].extend(b)
        records[name]["ratio_vs_cublas"] = statistics.median(
            left / right for left, right in zip(a, b, strict=True)
        )
        if name == "ours":
            records["cublas"]["ratio_vs_ours"] = statistics.median(
                right / left for left, right in zip(a, b, strict=True)
            )
    if "ours" in instances:
        for name, instance in instances.items():
            if name in ("ours", "cublas"):
                continue
            a, b = paired(instances["ours"], instance, args.pairs, args.reps_per_sample, flush)
            samples["ours"].extend(a)
            samples[name].extend(b)
            records[name]["ratio_vs_ours"] = statistics.median(
                left / right for left, right in zip(a, b, strict=True)
            )
        records["ours"]["ratio_vs_ours"] = 1.0
    for name, instance in instances.items():
        median = statistics.median(samples[name])
        records[name].update(
            us_median=median,
            p95_us=percentile95(samples[name]),
            epsilon=epsilon,
            util=base["SOL_us"] / median,
        )
        if args.trace:
            try:
                records[name]["kernel_names"] = kernel_names(instance)
            except Exception as exc:  # noqa: BLE001 -- profiler availability is optional
                records[name]["trace_error"] = exception_reason(exc)
    result = []
    for model, shape in aliases:
        for record in records.values():
            row = dict(
                record, model=model, name=shape["name"], count_per_step=shape["count_per_step"]
            )
            result.append(row)
            emit(row)
    return result


@contextmanager
def shipped_dispatch():
    """Acceptance always measures the shipped auto table, independent of shell overrides."""
    keys = ("CTI_LINEAR_PROVIDER", "CTI_LINEAR_CONFIG", "CTI_LINEAR_AUTOTUNE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]] = "auto"
    os.environ.pop(keys[1], None)
    os.environ[keys[2]] = "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def bf16_reduced_precision(value):
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = value
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous


def acceptance_verdict(ratio, epsilon, correct):
    return "PASS" if correct and ratio >= 1 - epsilon else "FAIL"


def kernels_per_replay(instance):
    """Actual CUDA activity count, including vendor split/reduction launches."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        instance.graph.replay()
        torch.cuda.synchronize()
    total = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events())
    return total // getattr(instance, "copies", 1)


def acceptance_copies(n, k, l2_bytes):
    """Weight copies per acceptance graph: >= 3x L2 and >= 1 GiB, <= 6 GB and 256."""
    nbytes = 2 * n * k
    wanted = math.ceil(max(3 * l2_bytes, 1 << 30) / nbytes)
    return max(2, min(wanted, 256, int(6e9 // nbytes)))


def hot_copies(n, k):
    """Back-to-back calls on one weight per hot-L2 graph: >= 512 MiB read, 16..256."""
    return max(16, min(256, math.ceil((512 << 20) / (2 * n * k))))


def run_accept_shape(args, n, k, aliases, flush, device, kind):
    from ctikernels.linear.inference import Linear

    out_dtype = getattr(args, "out_dtype", "bf16")
    fp32 = out_dtype == "fp32"
    dtype = torch.float32 if fp32 else torch.bfloat16

    def market_call(x, w):
        return torch.mm(x, w.t(), out_dtype=torch.float32)

    if not fp32:
        market_call = F.linear

    # One allocation/reference per N,K, shared by M values and logical aliases.
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    x = torch.randn(max(args.m), k, dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    ref, condition = reference_and_condition(x, weight)
    if args.l2 == "cold":
        copies = acceptance_copies(n, k, torch.cuda.get_device_properties(0).L2_cache_size)
        rotations = [weight.clone() for _ in range(copies - 1)]
    elif args.l2 == "engine":
        # Engine-like consecutive layer calls: cold weight copies alternate with
        # one recurring weight, which is L2-hot again whenever it fits beside the
        # previous call's weight. One graph per side, paired as in cold mode.
        cold = acceptance_copies(n, k, torch.cuda.get_device_properties(0).L2_cache_size)
        rotations = [w for _ in range(cold - 1) for w in (weight.clone(), weight)]
        copies = len(rotations) + 1
    else:
        # Hot L2: the graph calls the SAME weight back to back (no flush), enough
        # times that replays stay far above event-timer resolution.
        copies = hot_copies(n, k)
        rotations = [weight] * (copies - 1)
        flush = None
    rows = []
    for m in args.m:
        xx = x[:m]
        op = Linear(provider="auto")
        # CuTe is gated against full-precision cuBLAS; a cuBLAS fallback is the
        # market path itself and is compared with the market-flag result.
        with bf16_reduced_precision(False):
            expected = market_call(xx, weight)
        with bf16_reduced_precision(MARKET_BF16_REDUCED_PRECISION):
            market = market_call(xx, weight)
        # Capture CuTe first and cuBLAS last: every capture clears cuBLAS
        # workspaces, so no later capture may free a timed cuBLAS graph's workspace.
        # cuBLAS chooses its kernel at capture: time the market's setting, also
        # for the dispatched op's cuBLAS fallback.
        with bf16_reduced_precision(MARKET_BF16_REDUCED_PRECISION):
            dispatched = Instance(
                "dispatched", partial(op, out_dtype=dtype) if fp32 else op, xx, weight, rotations
            )
            calibration = Instance("cublas", market_call, xx, weight, rotations)
            baseline = Instance("cublas", market_call, xx, weight, rotations)
        if dispatched.output.shape != expected.shape or dispatched.output.dtype != dtype:
            raise ValueError("Wrong dispatched output contract")
        baseline_output = expected if op.provider == "cute" else market
        metrics = accuracy(
            dispatched.output,
            baseline_output,
            ref[:m],
            condition[:m],
            k,
            **({"out_dtype": "fp32"} if fp32 else {}),
        )
        a, b = paired(baseline, calibration, args.pairs, args.reps_per_sample, flush)
        epsilon = percentile95([abs(aa / bb - 1) for aa, bb in zip(a, b, strict=True)])
        a, b = paired(baseline, dispatched, args.pairs, args.reps_per_sample, flush)
        speedups = [aa / bb for aa, bb in zip(a, b, strict=True)]
        ratio = statistics.median(speedups)
        ci_low, ci_high = bootstrap_speedup_ci(speedups, seed=f"{n}:{k}:{m}:{args.l2}")
        # Outside timing; the op property counts logical cuBLAS calls, so profile
        # both graphs to report actual vendor kernels as well as CuTe launches.
        count = kernels_per_replay(dispatched)
        baseline_count = kernels_per_replay(baseline)
        if min(count, baseline_count) < 1:
            raise RuntimeError("CUDA profiler did not report graph kernels")
        ours_us, cublas_us = statistics.median(b), statistics.median(a)
        sol_us = (2 * (n * k + m * k) + (4 if fp32 else 2) * m * n) / (READ_GBPS[kind] * 1e3)
        record = dict(
            kind="smallm_accept",
            out_dtype=out_dtype,
            device=device,
            device_class=kind,
            fastest_market="torch.mm(out_dtype=torch.float32)" if fp32 else "F.linear",
            fastest_market_provenance=FP32_MARKET_PROVENANCE if fp32 else MARKET_PROVENANCE,
            cublas_bf16_reduced_precision=MARKET_BF16_REDUCED_PRECISION,
            N=n,
            K=k,
            M=m,
            mode=args.l2,
            l2=args.l2,
            timing_method={
                "cold": "rotating_weights",
                "engine": "alternating_cold_hot_weights",
                "hot": "repeated_weight",
            }[args.l2],
            rotation_copies=copies,
            pairs=args.pairs,
            self_pairs=args.pairs,
            reps_per_sample=args.reps_per_sample,
            provider=op.provider,
            run_id=getattr(args, "run_id", None),
            # Exact measured CuTe configuration; dispatch qualification matches it.
            launch_config=op.launch_config,
            kernels_per_call=count,
            logical_kernels_per_call=op.kernels_per_call,
            cublas_kernels_per_call=baseline_count,
            ours_us=ours_us,
            cublas_us=cublas_us,
            sol_us=sol_us,
            sol_fraction=sol_us / ours_us,
            cublas_sol_fraction=sol_us / cublas_us,
            sol_read_GBps=READ_GBPS[kind],
            sol_source="sol-ceilings built-in",
            sol_bandwidth_policy="read-only",
            paired_speedup=ratio,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_method="bootstrap_median_percentile95_2000",
            epsilon=epsilon,
            epsilon_method="linear_p95_abs_self_ratio_minus_one",
            verdict=acceptance_verdict(ratio, epsilon, metrics["correct"]),
            **metrics,
        )
        for model, shape in aliases:
            row = dict(
                record,
                model=model,
                name=shape["name"],
                layout=shape.get("layout", "both"),
                count_per_step=shape["count_per_step"],
                row_domain=shape.get("row_domain", "block"),
            )
            rows.append(row)
            emit(row)
            print(
                f"# out_dtype={out_dtype} {device} {model} {shape['name']} M={m}: {row['verdict']} "
                f"provider={op.provider} kernels={count} cuBLAS/dispatched={ratio:.6f} "
                f"epsilon={epsilon:.6f}",
                flush=True,
            )
        del dispatched, calibration, baseline, op
    return rows


def model_projections(rows, inventory):
    """Measured rows per (device, model), with the model's full projection inventory.

    Models outside ``inventory`` get no totals: coverage cannot be checked.
    Only cold-L2 (rotating-weight) rows count: the engine streams every weight
    once per step or round, far beyond L2, so hot-L2 timings never enter a sum.
    """
    inventory = builtin_shapes() if inventory is None else inventory
    grouped = defaultdict(list)
    for row in rows:
        if (
            row.get("kind") == "smallm_accept"
            and row.get("l2", "cold") == "cold"
            and row.get("model") in inventory
        ):
            grouped[row["device"], row["model"]].append(row)
    for (device, model), entries in sorted(grouped.items()):
        measured = {(r["name"], r["N"], r["K"], r["M"]): r for r in entries}
        shapes = inventory[model]
        for layout in model_layouts(shapes):
            required = [s for s in shapes if s.get("layout", "both") in (layout, "both")]
            yield device, model, layout, required, measured


def covered(required, measured, rows_for):
    """Each required projection's measured row, or None if any one is missing."""
    parts = []
    for shape in required:
        row = measured.get((shape["name"], shape["N"], shape["K"], rows_for(shape)))
        if row is None:
            return None
        parts.append((shape, row))
    return parts or None


def step_estimates(rows, inventory=None):
    """Sum medians times call counts, keeping alternative GDN layouts separate.

    A total is emitted only when every projection of the model's inventory
    (builtin shapes by default) was measured at that M.
    """
    totals = {}
    for device, model, layout, required, measured in model_projections(rows, inventory):
        for m in sorted({key[3] for key in measured}):
            parts = covered(required, measured, lambda _shape, m=m: m)
            if parts is None:
                continue
            totals[device, model, layout, m] = {
                "kind": "smallm_step",
                "device": device,
                "model": model,
                "layout": layout,
                "M": m,
                "calls_per_step": sum(s["count_per_step"] for s, _ in parts),
                "ours_step_ms": sum(r["ours_us"] * s["count_per_step"] for s, r in parts) / 1000,
                "cublas_step_ms": sum(r["cublas_us"] * s["count_per_step"] for s, r in parts)
                / 1000,
                "acceptance_passed": all(r["verdict"] == "PASS" for _, r in parts),
            }
    for (device, model, layout, _m), row in totals.items():
        base = totals.get((device, model, layout, 1))
        row["ours_delta_ms"] = row["ours_step_ms"] - base["ours_step_ms"] if base else None
        row["cublas_delta_ms"] = row["cublas_step_ms"] - base["cublas_step_ms"] if base else None
    return list(totals.values())


def model_layouts(shapes):
    """Do not invent a split total for a model whose inventory has only fused GDN shapes."""
    layouts = {r.get("layout", "both") for r in shapes} - {"both"}
    return sorted(layouts) if layouts else ("fused", "split")


def round_estimates(rows, inventory=None):
    """Isolated GEMM sums for B=1..4, Q=4/5/6/8/9/12/16.

    DFlash context projections assume B context rows per round; block projections
    use B*Q. Emit only when every projection of the model's inventory (builtin
    shapes by default) was measured at its row count.
    """
    result = []
    for device, model, layout, required, measured in model_projections(rows, inventory):
        for b in range(1, 5):
            for q in (4, 5, 6, 8, 9, 12, 16):

                def rows_for(shape, b=b, q=q):
                    return b if shape.get("row_domain") == "context" else b * q

                parts = covered(required, measured, rows_for)
                if parts is None:
                    continue
                result.append(
                    {
                        "kind": "smallm_round",
                        "device": device,
                        "model": model,
                        "layout": layout,
                        "B": b,
                        "Q": q,
                        "M": b * q,
                        "context_rows": b
                        if any(s.get("row_domain") == "context" for s, _ in parts)
                        else None,
                        "calls_per_round": sum(s["count_per_step"] for s, _ in parts),
                        "ours_round_ms": sum(r["ours_us"] * s["count_per_step"] for s, r in parts)
                        / 1000,
                        "cublas_round_ms": sum(
                            r["cublas_us"] * s["count_per_step"] for s, r in parts
                        )
                        / 1000,
                        "acceptance_passed": all(r["verdict"] == "PASS" for _, r in parts),
                    }
                )
    return result


def run_acceptance(args, shapes, flush, device):
    from ctikernels.linear._dispatch import device_class

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    kind = device_class((props.major, props.minor), props.multi_processor_count)
    if kind not in READ_GBPS:
        raise ValueError("acceptance requires a measured device class")
    if kind == "gb10" and args.pairs < 60:
        print(
            "Warning: GB10 acceptance needs --pairs >= 60; "
            f"this {args.pairs}-pair run cannot qualify CuTe rows.",
            file=sys.stderr,
        )
    rows = []
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    started = time.monotonic()
    # Completion is scoped per run: concatenated partition logs stay separable.
    args.run_id = uuid.uuid4().hex
    emit(
        {
            "kind": "accept_begin",
            "device": device,
            "run_id": args.run_id,
            "out_dtype": getattr(args, "out_dtype", "bf16"),
        }
    )
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        with torch.inference_mode(), shipped_dispatch():
            for (n, k), aliases in shapes.items():
                rows.extend(run_accept_shape(args, n, k, aliases, flush, device, kind))
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    if args.step:
        models = load_models(args)
        for row in step_estimates(rows, models) + round_estimates(rows, models):
            emit(dict(row, run_id=args.run_id, out_dtype=getattr(args, "out_dtype", "bf16")))
    failures = sum(row["verdict"] != "PASS" for row in rows)
    emit(
        {
            "kind": "accept_completion",
            "out_dtype": getattr(args, "out_dtype", "bf16"),
            "device": device,
            "run_id": args.run_id,
            "rows": len(rows),
            "failures": failures,
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return int(failures != 0)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapes", type=Path, help="shape inventory JSON (arbitrary positive N/K)")
    ap.add_argument("--out-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument(
        "--model",
        choices=[*builtin_shapes(), "all"],
        default=None,
        help="default: 9B for builtins, all models for a JSON file",
    )
    ap.add_argument("--m", default="1,2,3,4,6,8")
    ap.add_argument("--providers", default="cublas,cublaslt,quack,ours")
    ap.add_argument("--provider-hook", help="module:function that populates the callable registry")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument(
        "--reps-per-sample",
        type=int,
        default=None,
        help="default: 1 for acceptance/step, 5 for market",
    )
    ap.add_argument(
        "--accept", action="store_true", help="shipped-dispatch parity protocol (>=20 pairs)"
    )
    ap.add_argument(
        "--step", action="store_true", help="run acceptance and sum decode/round GEMM estimates"
    )
    ap.add_argument(
        "--l2", choices=("cold", "hot", "engine"), default="cold", help="acceptance cache regime"
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--cold", dest="cold", action="store_true", default=True)
    mode.add_argument("--warm", dest="cold", action="store_false")
    ap.add_argument("--trace", action="store_true", help="untimed CUDA kernel names per graph")
    args = ap.parse_args(argv)
    args.accept = args.accept or args.step
    if args.out_dtype == "fp32" and not args.accept:
        ap.error("--out-dtype fp32 requires --accept")
    if args.reps_per_sample is None:
        args.reps_per_sample = 1 if args.accept else 5
    try:
        args.m = parse_ms(args.m)
        if min(args.pairs, args.reps_per_sample) < 1:
            raise ValueError
        if args.accept and (args.pairs < 20 or not args.cold):
            ap.error("accept/step requires cold mode and at least 20 pairs")
        args.providers = list(dict.fromkeys(["cublas", *args.providers.split(",")]))
        if "" in args.providers:
            raise ValueError
        if args.provider_hook and args.provider_hook.count(":") != 1:
            raise ValueError
    except ValueError:
        ap.error("Invalid sweep selection")
    return args


def load_models(args):
    """Full projection inventory per model (shape file or builtins), before --model."""
    return json.loads(args.shapes.read_text())["models"] if args.shapes else builtin_shapes()


def load_shapes(args):
    models = load_models(args)
    selected = args.model or ("all" if args.shapes else "qwen3.5-9b")
    if selected != "all":
        models = {selected: models[selected]}
    unique = defaultdict(list)
    for model, rows in models.items():
        for row in rows:
            for field in ("N", "K", "count_per_step"):
                if type(row[field]) is not int or row[field] <= 0:
                    raise ValueError("Shape dimensions and counts must be positive integers")
            unique[row["N"], row["K"]].append((model, row))
    if not unique:
        raise ValueError("No shapes selected")
    return unique


def main(argv=None):
    args = parse_args(argv)
    shapes = load_shapes(args)
    if not torch.cuda.is_available():
        emit({"kind": "error", "reason": "CUDA_required"})
        return 1
    device = torch.cuda.get_device_name()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    # L2 attribute is provided by current PyTorch. Fail explicitly on older builds
    # rather than claim an unverified fixed-size allocation is >=2x L2.
    l2_bytes = properties.L2_cache_size
    if l2_bytes <= 0:
        raise RuntimeError("L2 cache size unavailable")
    cold = args.l2 in ("cold", "engine") if args.accept else args.cold
    flush = L2Flush(l2_bytes) if cold else None
    if args.accept:
        return run_acceptance(args, shapes, flush, device)
    providers, unavailable = discover_providers(args.provider_hook)
    rows = []
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            MARKET_BF16_REDUCED_PRECISION
        )
        with torch.inference_mode():
            from ctikernels.linear._dispatch import device_class

            kind = device_class(
                (properties.major, properties.minor), properties.multi_processor_count
            )
            bw, sol_metadata = sol_bandwidth(kind, lambda: measure_bandwidth(flush))
            emit(
                {
                    "kind": "bandwidth",
                    "device": device,
                    "BW_GBps": bw / 1e9,
                    **sol_metadata,
                    "L2_bytes": l2_bytes,
                    "flush_bytes": flush.nbytes if flush is not None else 0,
                }
            )
            for (n, k), aliases in shapes.items():
                for m in args.m:
                    rows.extend(
                        run_shape(args, n, k, m, aliases, providers, unavailable, flush, bw, device)
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    print(
        "# model | shape | M | provider | median us | p95 us | cublas/provider | epsilon | SOL us | util | status"
    )
    for row in rows:
        numbers = [
            "-" if row[key] is None else f"{row[key]:.4f}"
            for key in ("us_median", "p95_us", "ratio_vs_cublas", "epsilon", "SOL_us", "util")
        ]
        print(
            " | ".join(
                [
                    "# " + row["model"],
                    row["name"],
                    str(row["M"]),
                    row["provider"],
                    *numbers,
                    row["status"],
                ]
            )
        )
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 -- bounded diagnostics, no traceback
        emit({"kind": "error", "reason": exception_reason(exc)})
        raise SystemExit(1) from None
