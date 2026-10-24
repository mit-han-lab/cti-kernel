# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired engine-regime measurements. Add operations through OPERATIONS.

No GPU initialization at import. JSON receipts are prefixed ROW; build diagnostics
are sent to stderr. SGLang source stays outside this repository.

C4 example: python -m benchmark.kernel_tuning.norm_act_paired --op add_norm
--market-root ../market --rows 16 --regimes engine --check --profile
--vllm-mode shipped. Norm workers install FI 0.7.1 privately in a temp dir;
--fi071 DIR reuses an overlay. --fi07 DIR separately qualifies our provider.
--candidates ours,static,cute,flashinfer,fi_default,triton limits local providers;
ours and A/A are always retained. --sweep adds all eligible CuTe configurations.
Norm market rows also retain explicit-config cute and pair it with cute_remote;
xproc reports the median per-round remote/local time ratio.
FP32 add_norm rows also pair a size-matched rw_copy diagnostic by default;
--no-rw-copy disables it. Its times and sol_witness never enter market gates.
SiLU also requires --sglang-root DIR for the existing SGLang JIT source tree.
--vllm-mode auto tries source-proven compilation then uses shipped kernels;
--allow-vllm-mismatch is diagnostic only and cannot yield a qualifying PASS.
SiLU --raw adds per-round samples; --keep selects exact extra candidate names.
--consumer captures each activation followed by a shared 4096-wide down-projection
GEMM in the same graph, for an engine consumer A/B in the hot/cold regimes.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard, profiler_clean
from benchmark.kernel_tuning.norm_act_norm import (
    norm_candidates,
    run_norm_row,
)
from benchmark.kernel_tuning.norm_act_reference import (
    error_metrics,
    market_formulas,
    no_worse,
    reference,
)
from ctikernels.activation._dispatch import tune_configs
from ctikernels.activation.inference import SiluAndMul

SGL_REVISION = "72730d6360a5894b167500d5d1ad82c34c6bb116"


def median_ci(values):
    """Exact distribution-free >=95% median CI by Binomial(n, 1/2) order ranks.

    Select largest k with P(Bin(n,1/2) <= k-1) <= .025. No asymptotic
    normal approximation, numpy/scipy dependency or bootstrap randomness.
    """
    values = sorted(values)
    n = len(values)
    if not n:
        raise ValueError("empty sample")
    k, tail = 0, 0
    for rank in range(1, n // 2 + 1):
        tail += math.comb(n, rank - 1)
        if tail / (2**n) > 0.025:
            break
        k = rank
    return statistics.median(values), ([values[k - 1], values[n - k]] if k else [0.0, float("inf")])


def paired_summary(samples, markets=("vllm", "sglang")):
    """Within-round market ratios; FP32 SOL rows are traffic-bound at t_dram_us >= 4."""
    markets = tuple(name for name in markets if name in samples)
    if not markets:
        raise ValueError("no market samples")
    ratios = {
        name: [market / ours for market, ours in zip(times, samples["ours"], strict=True)]
        for name, times in samples.items()
        if name not in ("ours", "rw_copy")
    }
    aa, aa_ci = median_ci(ratios["aa"])
    epsilon = max(0.005, abs(1 - aa) + (aa_ci[1] - aa_ci[0]) / 2)
    comparison = {
        name: dict(zip(("median", "ci95"), median_ci(values), strict=True))
        for name, values in ratios.items()
    }
    # Min within each paired round: the faster market cannot be chosen after looking
    # at a favorable aggregate. This is at least as strict as either market alone.
    faster = [min(values) for values in zip(*(ratios[name] for name in markets), strict=True)]
    ratio, ci = median_ci(faster)
    receipt = {
        "markets": list(markets),
        "us": {name: statistics.median(times) for name, times in samples.items()},
        "ratios": comparison,
        "faster_ratio": ratio,
        "faster_ci95": ci,
        "epsilon": epsilon,
        "verdict": "PASS" if ci[0] >= 1 - epsilon else "FAIL",
    }
    for name in ("sgl_fp32", "sgl_fp32c"):
        if name in comparison:
            receipt[name + "_ratio"] = comparison[name]["median"]
            receipt[name + "_ci95"] = comparison[name]["ci95"]
    if {"sgl_fp32", "sgl_fp32c"} <= samples.keys():
        ratio, ci = median_ci(
            [
                min(eager, compiled)
                for eager, compiled in zip(ratios["sgl_fp32"], ratios["sgl_fp32c"], strict=True)
            ]
        )
        receipt.update(fp32_market_ratio=ratio, fp32_market_ci95=ci, fp32_market_ci95_lower=ci[0])
    if "cute_remote" in samples:
        control_ratio, control_ci = median_ci(
            [
                remote / local
                for remote, local in zip(samples["cute_remote"], samples["cute"], strict=True)
            ]
        )
        receipt["cross_process_control"] = {
            "local": "cute",
            "remote": "cute_remote",
            "ratio_median": control_ratio,
            "ci95": control_ci,
        }
    return receipt


@lru_cache(maxsize=1)
def measure_bandwidth():
    """Cold FP32 copy/add/read-only sum on max(512 MiB, 4*L2) buffers.

    Mixed read/write SOL uses max(copy, add); sum is a separate read-only witness.
    Measured once per process with CUDA events, without graph or profiler timing.
    """
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if not l2:
        raise RuntimeError("torch device properties must expose L2 cache size")
    size = max(512 * 1024**2, 4 * l2)
    elements = (size + 3) // 4
    a = torch.ones(elements, device="cuda", dtype=torch.float32)
    b, c = torch.ones_like(a), torch.empty_like(a)
    size = a.numel() * a.element_size()
    measurements = {}
    for name, call, moved in (
        ("copy", lambda: c.copy_(a), 2 * size),
        ("add", lambda: torch.add(a, b, out=c), 3 * size),
        ("sum", lambda: torch.sum(a), size),
    ):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(20)
        ]
        for start, end in events:
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        median_us = statistics.median(start.elapsed_time(end) * 1000 for start, end in events)
        measurements[name] = {
            "bytes": moved,
            "median_us": median_us,
            "bw_gbs": moved / (median_us * 1000),
        }
    return {
        "buffer_bytes": size,
        "l2_bytes": l2,
        "reps": 20,
        "warmup": 5,
        "measurements": measurements,
        "bw_gbs": max(measurements[name]["bw_gbs"] for name in ("copy", "add")),
        "bw_read_gbs": measurements["sum"]["bw_gbs"],
    }


def sol_summary(m, width, times_us, bw_gbs, provider):
    """FP32 read/write SOL from max(copy, add); traffic-bound at t_dram_us >= 4."""
    moved = 12 * m * width + 2 * width
    t_dram_us = moved / (bw_gbs * 1000)
    # Below 4 us the launch latency dominates; apply the bandwidth bar above it.
    traffic_bound = t_dram_us >= 4.0
    sol = {
        name: t_dram_us / us
        for name, us in times_us.items()
        if name in ("ours", "static", "triton") or name.startswith("cute")
    }
    # cute_remote uses same-dtype market buffers, not the FP32 residual contract.
    sol.pop("cute_remote", None)
    receipt = {
        "dram_bytes": moved,
        "t_dram_us": t_dram_us,
        "traffic_bound": traffic_bound,
        "sol": sol,
    }
    if "rw_copy" in times_us:
        receipt["sol_witness"] = {name: times_us["rw_copy"] / times_us[name] for name in sol}
    if traffic_bound:
        bar = 0.95 if provider and (provider == "cute" or provider.startswith("cute_")) else 0.90
        receipt.update(sol_bar=bar, sol_pass=sol.get("ours", 0) >= bar)
    return receipt


def pin_x925():
    if platform.machine() != "aarch64":
        return
    cpus = []
    for entry in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict(line.split(":", 1) for line in entry.splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get("CPU part", "").lower() == "0xd85":
            cpus.append(int(fields["processor"]))
    allowed = set(cpus) & os.sched_getaffinity(0)
    if allowed:
        os.sched_setaffinity(0, allowed)


def _sglang_paths(root):
    for base in (root, root / "python/sglang/kernels/jit", root / "kernels/jit"):
        if (base / "csrc/elementwise/activation.cuh").is_file():
            return base
    raise FileNotFoundError(
        "--sglang-root must contain csrc/elementwise/activation.cuh and include/sgl_kernel"
    )


def load_sglang(root, dtype):
    """Minimal tvm-ffi loader for the unmodified market ActivationKernel path.

    Compile defaults studied in SGLang kernels/jit/utils/{arch,compile/*}.py
    at SGL_REVISION. We deliberately use the supplied source, not an installed
    SGLang package whose revision could differ. No vendor kernel source is embedded.
    """
    from tvm_ffi.cpp import load_inline

    base = _sglang_paths(root.resolve())
    header = base / "csrc/elementwise/activation.cuh"
    major, minor = torch.cuda.get_device_capability()
    scalar = {torch.bfloat16: "__nv_bfloat16", torch.float16: "__half"}[dtype]
    pdl = major >= 9
    # SGLang's get_default_target_flags plus get_activation_cuda_cflags (CUDA).
    flags = [
        f"-DSGL_CUDA_ARCH={major * 100 + minor * 10}",
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
    ]
    # Match SGLang's arch suffix using the actual nvcc version, not torch's toolkit.
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_root) / "bin/nvcc") if cuda_root else shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("SGLang market build requires nvcc")
    version_text = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version_text)
    if match is None:
        raise RuntimeError("cannot determine SGLang nvcc target")
    version = tuple(map(int, match.groups()))
    suffix = "f" if (major, minor) == (12, 0) and version >= (12, 9) else "a" if major >= 9 else ""
    target = f"{major}.{minor}{suffix}"
    source = (
        "#include <tvm/ffi/function.h>\n"
        f'#include "{header.as_posix()}"\n'
        "namespace sglang {\n"
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation, (ActivationKernel<{scalar}, {str(pdl).lower()}>::run_activation));\n"
        "}\n"
    )
    # Include the dependency tree in the identity; changed headers cannot reuse stale builds.
    digest = hashlib.sha256()
    for path in [header, *sorted((base / "include").rglob("*"))]:
        if path.is_file():
            digest.update(path.read_bytes())
    prior_target = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = target
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module = load_inline(
                name=f"cti_market_activation_{digest.hexdigest()[:12]}_{major}{minor}_{str(dtype).split('.')[-1]}",
                cuda_sources=source,
                extra_include_paths=[str(base / "include")],
                extra_cflags=["-std=c++20", "-O3"],
                extra_cuda_cflags=flags,
            )
    finally:
        if prior_target is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = prior_target

    def call(x, out):
        module.run_activation(x, out, "silu")
        return out

    return call, {
        "loader": "minimal_tvm_ffi",
        "loader_reason": "source-only market snapshot, no importable full SGLang package",
        "revision": SGL_REVISION,
        "source_sha256": digest.hexdigest(),
        "flags": flags,
        "target": target,
        "nvcc": list(version),
        "pdl": pdl,
    }


def vllm_native(x):
    # vLLM 0.31.0 model_executor/layers/activation.py: SiluAndMul.forward_native.
    # This expression is eligible only after proof against the shipped JIT body.
    width = x.shape[-1] // 2
    return F.silu(x[..., :width]) * x[..., width:]


def silu_candidates(args, dtype, x, out):
    ours = SiluAndMul(provider="auto")
    aa = SiluAndMul(provider="auto")
    from benchmark.kernel_tuning.norm_act_probe import provider_spec, run_probe
    from benchmark.kernel_tuning.norm_act_vllm import VllmSite, optional_sites

    site, omission = optional_sites(lambda: VllmSite(args, "silu", (x,)))

    def vllm(x, out):
        if args.consumer:
            result = site(x, output=out)
            # Compiled forward_native owns its output; shipped kernels use out directly.
            if result is not out:
                out.copy_(result)
            return out
        return site(x)

    sglang, sgl_meta = load_sglang(args.sglang_root, dtype)
    candidates = {
        "ours": ours,
        "aa": aa,
        "flashinfer": SiluAndMul(provider="flashinfer"),
        **({"vllm": vllm} if site else {}),
        "sglang": sglang,
    }
    if args.sweep:
        for config in tune_configs():
            name = config.name
            if re.search(args.sweep_filter or "", name):
                candidates[name] = SiluAndMul(provider="cute", config=config)
    selected = (
        set(args.candidates.split(","))
        if args.candidates
        else {"ours", "flashinfer", "fi_default", "cute"}
    )
    if args.keep is not None:
        selected = (set(args.candidates.split(",")) if args.candidates else set()) | set(
            args.keep.split(",")
        )
        for config in tune_configs():
            if config.name in selected:
                candidates[config.name] = SiluAndMul(provider="cute", config=config)
    if "fi_default" in selected:
        candidates["fi_default"] = SiluAndMul(provider="flashinfer")
    if "cute" in selected:
        candidates["cute"] = SiluAndMul(provider="cute")
    candidates = {
        name: fn
        for name, fn in candidates.items()
        if name in selected | {"ours", "aa", "vllm", "sglang"}
        or (args.sweep and name.startswith("cute_"))
    }
    for fn in candidates.values():
        fn(x, out)
    metadata = {"vllm": omission or site.metadata, "sglang": sgl_meta}
    if args.profile:
        kernels = run_probe(
            {
                "kind": "sglang_silu",
                "op": "silu_mul",
                "m": x.shape[0],
                "width": args.width,
                "dtype": args.dtype,
                "sglang_root": args.sglang_root,
            }
        )["kernels"]
        if len(kernels) != 1 or "act_and_mul_kernel" not in kernels[0]["name"]:
            raise RuntimeError(f"unexpected SGLang dispatch: {kernels}")
        metadata["sglang"]["kernels"] = kernels
        for name, fn in candidates.items():
            if name not in ("vllm", "sglang"):
                metadata.setdefault(name, {}).update(
                    run_probe(
                        provider_spec(
                            fn,
                            op=args.op,
                            m=x.shape[0],
                            width=args.width,
                            dtype=args.dtype,
                        )
                    )
                )
    return candidates, metadata


OPERATIONS = {
    "silu_mul": silu_candidates,
    "add_norm": norm_candidates,
    "gemma_add_norm": norm_candidates,
    "add_norm_sum": norm_candidates,
    "gemma_add_norm_sum": norm_candidates,
}


def check_candidates(candidates, inputs, out):
    """Check every distinct benchmark input before capture; report worst-buffer metrics."""
    worst = {}
    passed = True
    for x in inputs:
        exact = reference(x)
        results = {name: error_metrics(fn(x, out), exact) for name, fn in candidates.items()}
        formulas = {name: error_metrics(value, exact) for name, value in market_formulas(x).items()}
        for name, metrics in (results | formulas).items():
            worst[name] = {
                key: max(value, worst.get(name, {}).get(key, 0.0)) for key, value in metrics.items()
            }
        market = [results[name] for name in ("vllm", "sglang") if name in results]
        market.extend(formulas.values())
        for name, metrics in results.items():
            # Market comparisons are strict for every CuTe candidate, not a tolerance
            # that could hide a numerical regression behind timing success.
            if name in ("ours", "aa") or name.startswith("cute_"):
                passed &= no_worse(metrics, market)
            # All candidates must at least agree within one output ULP for random inputs.
            passed &= metrics["max_ulp"] <= 1.0
    return {"passed": bool(passed), "metrics": worst}


def warm_capture_call(fn, inputs, outputs, *, restore=None):
    """Warm every view privately, or on lock-step IPC pools with explicit restore."""
    from benchmark.kernel_tuning.norm_act_pools import clone_pool_views

    if restore is None:
        storage = {}
        # Preserve shared input/output allocations as well as each view's offset.
        inputs = clone_pool_views(inputs, storage)
        outputs = clone_pool_views(outputs, storage)
    else:
        restore()
    try:
        for i, (x, out) in enumerate(zip(inputs, outputs, strict=True)):
            if hasattr(fn, "call_index"):
                fn.call_index(x, out, i)
            else:
                fn(x, out)
    finally:
        if restore is not None:
            restore()


@contextlib.contextmanager
def cached_capture_plans(fn):
    """Assert local resolution hits before any lazy provider setup can run.

    SumNormCall wraps NormCall, which owns AddRMSNorm; activation handles can
    also be passed directly. Market wrappers have their own eager warmup above.
    Restore flags even if capture raises so subsequent eager warmup can retry.
    """
    from ctikernels.activation.inference import SiluAndMul
    from ctikernels.norm.inference import AddRMSNorm

    while not isinstance(fn, (AddRMSNorm, SiluAndMul)):
        fn = getattr(fn, "norm", getattr(fn, "op", None))
        if fn is None:
            yield
            return
    previous = fn._require_cached_plan
    fn._require_cached_plan = True
    try:
        yield
    finally:
        fn._require_cached_plan = previous


def capture_bank(
    fn, inputs, outputs, layers, *, shared_pool=None, capture_stream=None, warm_restore=None
):
    graphs, retained = [], []
    stream = capture_stream if capture_stream is not None else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Compilation, module loading and function attributes must be initialized
        # on this stream, on every pool view, before the first graph begins.
        warm_capture_call(fn, inputs, outputs, restore=warm_restore)
        stream.synchronize()
        # Torch no longer collects garbage when a capture begins. A collection during
        # capture can destroy an earlier row's graphs or unload its compiled modules,
        # which invalidates the capture (seen on GB10 from the second row on).
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
    try:
        _capture_graphs(fn, inputs, outputs, layers, stream, shared_pool, graphs, retained)
    finally:
        if gc_was_enabled:
            gc.enable()
    torch.cuda.current_stream().wait_stream(stream)
    # Every output stays live during capture, preserving distinct output stores.
    # These outputs have no consumers between candidate replays. Graphs own the
    # pool lifetime, so release Python output references after the complete bank
    # to reuse allocations across strictly sequential local market banks.
    return graphs, retained if shared_pool is None else []


def _capture_graphs(fn, inputs, outputs, layers, stream, shared_pool, graphs, retained):
    with torch.cuda.stream(stream):
        for start in range(0, len(inputs), layers):
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with cached_capture_plans(fn), torch.cuda.graph(graph, stream=stream, pool=shared_pool):
                begin.record()
                for i in range(start, start + layers):
                    if hasattr(fn, "call_index"):
                        retained.append(fn.call_index(inputs[i], outputs[i], i))
                    else:
                        retained.append(fn(inputs[i], outputs[i]))
                end.record()
            # Captured event nodes exclude Python enqueue gaps from microsecond timings.
            graphs.append((graph, begin, end))


def measure(banks, rounds, layers, before=None):
    profiler_clean()
    names = list(banks)
    samples = {name: [] for name in names}
    # Warm replay, including the entire cold pool; compile/capture are not timed.
    for name, bank in banks.items():
        if hasattr(bank, "sample"):
            torch.cuda.synchronize()
            bank.sample()
            continue
        if before is not None:
            before(name)
        for graph, _begin, _end in bank[0]:
            graph.replay()
    torch.cuda.synchronize()
    for round_index in range(rounds):
        # A seeded shuffle per round: a cyclic rotation keeps every candidate behind the
        # same neighbour, which biased identical kernels (FlashInfer as itself and as the
        # SGLang column) by up to 5% in hot in-place rows.
        order = list(names)
        random.Random(20261015 + round_index).shuffle(order)
        for name in order:
            profiler_clean()
            if hasattr(banks[name], "sample"):
                torch.cuda.synchronize()
                samples[name].append(banks[name].sample())
                continue
            if before is not None:
                before(name)
            graphs = banks[name][0]
            for graph, _begin, _end in graphs:
                graph.replay()
            graphs[-1][2].synchronize()
            samples[name].append(
                sum(begin.elapsed_time(end) for _graph, begin, end in graphs)
                * 1000
                / (layers * len(graphs))
            )
    return samples


class SumNormCall:
    """D2 alternates plain norm and aux capture by graph layer index."""

    def __init__(self, norm, fused=False):
        self.norm, self.fused = norm, fused

    @property
    def provider(self):
        return getattr(self.norm, "provider", None)

    @property
    def dispatch(self):
        return getattr(self.norm, "dispatch", self.provider)

    def call_index(self, tensors, unused, index):
        if index % 2 == 0:
            return self.norm(tensors[:3])
        return self(tensors)

    def __call__(self, tensors, unused=None):
        x, residual, weight, summed = tensors
        if self.fused:
            return self.norm.op(
                x,
                residual,
                weight,
                1e-6,
                zero_centered=self.norm.zero_centered,
                sum_out=summed,
            )
        torch.add(x, residual, out=summed)
        return self.norm((x, residual, weight))


class DownProjCall:
    """Activation and its consumer share a graph; op remains visible to capture guards."""

    def __init__(self, op, w_down, y):
        self.op, self.w_down, self.y = op, w_down, y

    def __call__(self, x, out):
        self.op(x, out)
        return torch.mm(out, self.w_down, out=self.y)


def run_norm_sum_row(args, m, regime):
    return run_norm_row(args, m, regime)


def run_row(args, m, regime):
    install_profiler_guard()
    if args.op.endswith("_sum"):
        return run_norm_sum_row(args, m, regime)
    if args.op != "silu_mul":
        return run_norm_row(args, m, regime)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if not l2:
        raise RuntimeError("torch device properties must expose L2 cache size")
    input_bytes = m * 2 * args.width * 2
    # Input pool alone exceeds 2x L2, so each candidate has the same cold working set
    # even when its compiler owns graph output allocations. Every graph still has L calls.
    count = (
        args.layers
        if regime == "hot"
        else max(args.layers, math.ceil((2 * l2 + 1) / input_bytes / args.layers) * args.layers)
    )
    torch.manual_seed(20261015 + m)
    first = torch.randn((m, 2 * args.width), device="cuda", dtype=dtype)
    out = torch.empty((m, args.width), device="cuda", dtype=dtype)
    candidates, metadata = OPERATIONS[args.op](args, dtype, first, out)
    inputs = (
        [first] * count
        if regime == "hot"
        else [first] + [torch.randn_like(first) for _ in range(count - 1)]
    )
    outputs = (
        [out] * count
        if regime == "hot"
        else [out] + [torch.empty_like(out) for _ in range(count - 1)]
    )
    # Mark each pool tensor just as the forward_native input is marked by vLLM.
    for tensor in inputs:
        torch._dynamo.mark_dynamic(tensor, 0)
    check = (
        check_candidates(candidates, [first] if regime == "hot" else inputs, out)
        if args.check
        else None
    )
    receipt = {
        "profiler_clean": profiler_clean(),
        "op": args.op,
        "device": props.name,
        "capability": [props.major, props.minor],
        "sms": props.multi_processor_count,
        "M": m,
        "width": args.width,
        "dtype": args.dtype,
        "regime": regime,
        "L": args.layers,
        "rounds": args.rounds,
        "l2_bytes": l2,
        "pool_input_bytes": input_bytes * (1 if regime == "hot" else count),
        "graphs_per_candidate": count // args.layers,
        "dispatch": candidates["ours"].provider,
        "providers": metadata,
        "check": check,
    }
    if metadata.get("vllm", {}).get("omitted"):
        receipt["omitted"] = {"vllm": metadata["vllm"]["reason"]}
    if args.consumer:
        receipt["consumer"] = "down_proj_mm_4096"
    if check and not check["passed"]:
        receipt["verdict"] = "NUMERICS_FAIL"
        return receipt
    if args.consumer:
        generator = torch.Generator(device=first.device).manual_seed(20261016)
        w_down = (
            torch.randn((args.width, 4096), device=first.device, dtype=dtype, generator=generator)
            * 0.02
        )
        y = torch.empty((m, 4096), device=first.device, dtype=dtype)
        candidates = {name: DownProjCall(fn, w_down, y) for name, fn in candidates.items()}
    banks = {
        name: capture_bank(fn, inputs, outputs, args.layers) for name, fn in candidates.items()
    }
    samples = measure(banks, args.rounds, args.layers)
    receipt.update(paired_summary(samples))
    if args.raw:
        receipt["raw"] = {name: [round(us, 4) for us in times] for name, times in samples.items()}
    if args.sweep:
        receipt["configs"] = {c.name: asdict(c) for c in tune_configs()}
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=tuple(OPERATIONS), default="silu_mul")
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--market-root", type=Path, help="shipped market sources and kernels")
    parser.add_argument("--vllm-mode", choices=("shipped", "compiled", "auto"), default="shipped")
    parser.add_argument("--allow-vllm-mismatch", action="store_true", help="diagnostics only")
    parser.add_argument("--fi071", type=Path, help="reuse private FlashInfer 0.7.1 worker overlay")
    parser.add_argument("--candidates", help="comma-separated local candidates; ours/A-A retained")
    parser.add_argument("--rows", type=int, nargs="+")
    parser.add_argument("--width", type=int)
    parser.add_argument("--residual-dtype", choices=("same", "fp32"), default="same")
    parser.add_argument(
        "--no-rw-copy", action="store_true", help="disable the paired FP32 add_norm copy witness"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--regimes", nargs="+", choices=("hot", "engine", "cold"))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "30")))
    parser.add_argument("--layers", type=int)
    parser.add_argument("--fi07", type=Path, help="FlashInfer 0.7.0.post1 install for norm child")
    parser.add_argument("--fi07-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--configs", help="explicit norm CuTe configs: t,r,p,s,trig,split separated by ; or /"
    )
    parser.add_argument("--sweep-filter", help="regex on CuTe config names kept by --sweep")
    parser.add_argument("--raw", action="store_true", help="SiLU per-round samples in ROW receipts")
    parser.add_argument("--consumer", action="store_true", help="SiLU followed by down-proj GEMM")
    parser.add_argument("--keep", help="comma-separated exact extra SiLU candidate names")
    parser.add_argument(
        "--static-weight",
        type=int,
        choices=(0, 1),
        default=None,
        help="norm weight prefetch opt-in; default 1 for same residual, 0 for fp32",
    )
    args = parser.parse_args(argv)
    if args.configs:
        from ctikernels.norm._dispatch import parse_config

        if args.op == "silu_mul":
            parser.error("--configs is supported only for norm")
        try:
            for setting in re.split(r"[;/]", args.configs):
                parse_config(setting)
        except ValueError as error:
            parser.error(str(error))
    if args.static_weight is None:
        args.static_weight = int(args.residual_dtype == "same")
    if args.residual_dtype == "fp32" and (args.op == "silu_mul" or args.fi07 or args.fi07_child):
        parser.error("--residual-dtype fp32 requires norm without --fi07")
    if args.op != "silu_mul" and (args.raw or args.consumer or args.keep is not None):
        parser.error("--raw, --consumer and --keep are supported only for silu_mul")
    if args.keep is not None:
        known = {"ours", "aa", "vllm", "sglang", "flashinfer", "fi_default", "cute"}
        known.update(config.name for config in tune_configs())
        unknown = set(args.keep.split(",")) - known
        if unknown:
            parser.error("unknown --keep candidates: " + ", ".join(sorted(unknown)))
    if args.market_root is None and not args.fi07_child:
        parser.error("--market-root is required for market comparisons")
    if args.op.endswith("_sum") and (args.fi07 or args.fi07_child):
        parser.error("sum-output pairs do not support --fi07")
    if args.regimes is None and args.op.endswith("_sum"):
        args.regimes = ["engine"]
    if args.regimes is None:
        args.regimes = ["hot", "cold"] if args.op == "silu_mul" else ["hot", "engine", "cold"]
    if args.op == "silu_mul" and "engine" in args.regimes:
        parser.error("engine is supported for add_norm and gemma_add_norm")
    if args.op == "silu_mul" and args.sglang_root is None:
        parser.error("silu_mul requires --sglang-root")
    if args.op == "silu_mul" and (args.fi07 or args.fi07_child):
        parser.error("--fi07 is supported for add_norm and gemma_add_norm")
    if args.fi07 is not None and not (args.fi07 / "flashinfer").is_dir():
        parser.error("--fi07 must contain a FlashInfer installation")
    if args.rows is None:
        args.rows = {
            "silu_mul": [1, 4, 16, 64, 256, 1024],
            "add_norm": [1, 2, 3, 4, 16, 64],
            "add_norm_sum": [1, 4, 16, 64, 256, 1024],
            "gemma_add_norm_sum": [1, 4, 16, 64, 256, 1024],
            "gemma_add_norm": [256, 512, 1024, 2048, 4096, 8192],
        }[args.op]
    if args.width is None:
        args.width = 12288 if args.op == "silu_mul" else 4096
    if args.layers is None:
        args.layers = 32 if args.op == "silu_mul" else 36
    if (
        min(args.rows) <= 0
        or args.width <= 0
        or args.width % (16 if args.op == "silu_mul" else 8)
        or args.rounds < 6
        or args.layers <= 0
    ):
        parser.error("positive M/L, aligned width (SiLU:16, norm:8), at least 6 rounds required")
    return args


def main(argv=None):
    install_profiler_guard()
    args = parse_args(argv)
    pin_x925()
    rows = []
    with torch.inference_mode():
        bandwidth = measure_bandwidth() if args.residual_dtype == "fp32" else None
        if bandwidth is not None:
            print(
                "HEADER "
                + json.dumps(
                    {
                        "bandwidth": bandwidth,
                        "bw_gbs": bandwidth["bw_gbs"],
                        "bw_read_gbs": bandwidth["bw_read_gbs"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        for m in args.rows:
            for regime in args.regimes:
                from benchmark.kernel_tuning.norm_act_vllm import MarketError

                try:
                    row = run_row(args, m, regime)
                except MarketError as error:
                    row = {
                        "op": args.op,
                        "M": m,
                        "regime": regime,
                        "dispatch": None,
                        "verdict": "MARKET_ERROR",
                        "error": str(error),
                        "providers": error.receipt,
                    }
                if bandwidth is not None:
                    row.update(
                        bandwidth=bandwidth,
                        bw_gbs=bandwidth["bw_gbs"],
                        bw_read_gbs=bandwidth["bw_read_gbs"],
                    )
                    if "us" in row:
                        row.update(
                            sol_summary(
                                m, args.width, row["us"], bandwidth["bw_gbs"], row["dispatch"]
                            )
                        )
                if args.allow_vllm_mismatch and row["verdict"] == "PASS":
                    row["verdict"] = "DIAGNOSTIC"
                if args.allow_vllm_mismatch:
                    row["candidate_verdicts"] = {
                        name: "DIAGNOSTIC" if verdict == "PASS" else verdict
                        for name, verdict in row.get("candidate_verdicts", {}).items()
                    }
                # CUDA graph pools from prior rows must not accumulate in the cache.
                torch.cuda.empty_cache()
                row["profiler_clean"] = profiler_clean()
                rows.append(row)
                print("ROW " + json.dumps(row, separators=(",", ":")), flush=True)
    columns = ("ours", "vllm", "vllm_a", "vllm_b", "sglang", "flashinfer") + (
        ("flashinfer07", "ours_fi07") if args.fi07 else ()
    )
    if args.residual_dtype == "fp32":
        columns += ("sgl_fp32", "sgl_fp32c")
        if args.op == "add_norm" and not args.no_rw_copy:
            columns += ("rw_copy",)
    print(
        "M regime "
        + " ".join(name + "_us" for name in columns)
        + " faster_ratio CI95 eps provider verdict xproc"
        + (
            " sgl_fp32_ratio CI95 sgl_fp32c_ratio CI95 fp32_market_ratio CI95 t_dram_us traffic_bound sol sol_bar sol_pass"
            if args.residual_dtype == "fp32"
            else ""
        )
        + (" unfused_ratio CI95" if args.op.endswith("_sum") else "")
    )
    for row in rows:
        times = row.get("us", {})
        print(
            row["M"],
            row["regime"],
            *(f"{times.get(k, float('nan')):.3f}" for k in columns),
            f"{row.get('faster_ratio', float('nan')):.3f}",
            row.get("faster_ci95"),
            f"{row.get('epsilon', float('nan')):.4f}",
            row["dispatch"],
            row["verdict"],
            f"{row.get('cross_process_control', {}).get('ratio_median', float('nan')):.3f}",
            *(
                [
                    row.get("sgl_fp32_ratio"),
                    row.get("sgl_fp32_ci95"),
                    row.get("sgl_fp32c_ratio"),
                    row.get("sgl_fp32c_ci95"),
                    row.get("fp32_market_ratio"),
                    row.get("fp32_market_ci95"),
                    row.get("t_dram_us"),
                    row.get("traffic_bound"),
                    row.get("sol"),
                    row.get("sol_bar"),
                    row.get("sol_pass"),
                ]
                if args.residual_dtype == "fp32"
                else []
            ),
            *(
                [row.get("unfused_ratio"), row.get("unfused_ci95")]
                if args.op.endswith("_sum")
                else []
            ),
        )
    return int(any(row["verdict"] in ("NUMERICS_FAIL", "MARKET_ERROR") for row in rows))


if __name__ == "__main__":
    sys.exit(main())
