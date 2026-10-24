#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Whole-block paired parity against SGLang 72730d6 and vLLM 0.31.0.

Copies the original workload and timing protocol: paired ABAB (rotating provider
order), CUDA graphs, 256 MiB L2 flush outside events, eps from 20 self pairs, and
unchanged pass/fail/inconclusive verdicts. Product means v1, CuTe on sm120.
--ours port lazily loads PortPrefill; --port-options passes typed constructor options.
--extra-ours adds a separate port/product v1 pair after market graphs are released,
with its own 20 self pairs. The original market interleaving is unchanged.
Port output and written pools are compared to v1 before timing; differences and
untouched-row preservation are recorded without numerical assertions.
Case indices are GLOBAL: small=0..7, long=8..16; --cases intersects --set.
One ours provider per invocation; inherited environment is left untouched (and
never dumped), except during --with-fi's additional product construction.
--with-fi verifies row dispatch as cute/flashinfer and defaults to 30 pairs.

Timing: separate graphs/pools, 3 graph warmups, 256 MiB zero_ L2 flush OUTSIDE
CUDA events around exactly one replay. State evolves between timed replays as
in whole_prefill. Restore initial pools before each paired experiment, not
inside a timed call. Calibration always has 20 pairs, independent of --pairs.
Statistics use full precision; only emitted sample arrays round to 0.01 us.
p95 is sorted linear interpolation at index .95*(n-1), zero-based.

Roof model (decimal GB/s, TFLOP/s; t_roof in MICROSECONDS):
  T=sum(lens), N=len(lens), I=count(initial), K=V=128, W=4,
  Q=2*Hk*K+Hv*V.
  F=7*T*Hv*K*V + 2*W*T*Q (useful recurrence + convolution FLOPs).
  D=2*T*Q                       # projected BF16 mixed qkv read
    + 2*T*(2*Hv)                # projected BF16 b,a read (not FP32)
    + 2*T*Hv*V                  # projected BF16 z read
    + 2*T*Hv*V                  # BF16 output write
    + 2*(I+N)*(W-1)*Q           # BF16 conv history read(initial)/write(all)
    + 4*(I+N)*Hv*K*V            # FP32 recurrent read(initial)/write(all)
    + 2*W*Q + 4*(2*Hv+V)       # conv, A_log, dt_bias, norm weights
    + 4*(N+1) + 4*N + N         # cu_seqlens, slot_indices, initial flags.
Only used slots count, not the two extra pool slots. All frozen lengths >= W-1.
D is compulsory boundary traffic, not measured DRAM traffic: intermediates,
repeated loads and cache effects are excluded. Small norm/gate FLOPs omitted.
BW=max(best copy BW, best reduction BW), each best of 10, bytes=read+write.
Copy: 1 GiB BF16 source + 1 GiB destination; reduction: same 1 GiB source to
FP32 scalar. MMA: BF16 8192^3 torch.matmul, FP32 accumulation with reduced-
precision reduction disabled, BF16 output. Roof utilization is a useful-work
lower-bound estimate and is not clipped at 1.

No CUDA activity at import time. No exception messages or argv/env dumps.
"""

import argparse
import gc
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import memory_guard
import whole_prefill as harness

# Reuse its torch module, without initializing CUDA at import.
torch = harness.torch
CALIBRATION_PAIRS = 20
GEOMETRIES = ("16x32", "16x48", "8x16", "8x24")


class CuteProviderRequiredError(RuntimeError):
    """Refuse to measure a fallback in a CuTe-only experiment."""


class FlashInferRequiredError(RuntimeError):
    """Refuse to measure a fallback in a FlashInfer experiment."""


def require_cute(label):
    if label != "cute":
        raise CuteProviderRequiredError("Expected product provider cute")


def product_factory(hk, hv, device, t, n, *, flashinfer=False):
    """Construct a product op, restore the environment, then verify row dispatch."""
    from ctikernels.gdn import GatedDeltaNetPrefill

    overrides = {"CTI_GDN_PREFILL_FLASHINFER": "1", "CTI_GDN_PREFILL_FORCE": "flashinfer"}
    previous = {key: os.environ.get(key) for key in overrides} if flashinfer else {}
    try:
        if flashinfer:
            os.environ.update(overrides)
        op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=device)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    expected = "flashinfer" if flashinfer else "cute"
    error = FlashInferRequiredError if flashinfer else CuteProviderRequiredError
    selected = (
        op.selected_provider(t, n)
        if hasattr(op, "selected_provider")
        else getattr(op, "provider", None)
    )
    if selected != expected:
        raise error("Unexpected product provider for row")

    def call(*a):
        y = op(*a)
        used = getattr(op._kernel, "last_provider", expected)
        if used != expected and not torch.cuda.is_current_stream_capturing():
            raise error("Unexpected runtime product provider")
        return y

    return (lambda a: a), call, expected


def geometry_metadata(hk, hv):
    # market/sglang72_triton.py:_features tests the post-conv QKV width,
    # NOT the full projection width (which also contains z, b and a).
    width = 2 * hk * harness.K + hv * harness.V
    return {"sgl_qkv_width": width, "sgl_qkv_split": "fused" if width <= 8192 else "split_view"}


QuietParser = harness.QuietParser


def percentile95(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def model(hk, hv, lengths, initial):
    t, n, i = sum(lengths), len(lengths), sum(initial)
    k, v, w = harness.K, harness.V, 4
    q = 2 * hk * k + hv * v
    f = 7 * t * hv * k * v + 2 * w * t * q
    d = (
        2 * t * q
        + 4 * t * hv
        + 4 * t * hv * v
        + 2 * (i + n) * (w - 1) * q
        + 4 * (i + n) * hv * k * v
        + 2 * w * q
        + 4 * (2 * hv + v)
        + 4 * (n + 1)
        + 4 * n
        + n
    )
    return f, d


def emit(record, output):
    line = json.dumps(record, separators=(",", ":"), allow_nan=False)
    # Persist first, so stdout truncation cannot lose completed records.
    if output is not None:
        output.write(line + "\n")
        output.flush()
    print(line, flush=True)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


class Instance:
    def __init__(
        self, provider, hk, hv, lengths, initial, seed, device, *, factory=None, port_options=None
    ):
        prepare, self.call, self.label = (
            harness._provider(provider, hk, hv, device, port_options=port_options)
            if factory is None
            else factory
        )
        self.base = harness._case(hk, hv, lengths, initial, seed, device)
        self.args = prepare(self.base)
        self.provider = provider
        # All providers expose state and native conv last.
        self.pools = self.args[-2:]
        self.original = tuple(pool.clone() for pool in self.pools)
        self.graph = None
        self.graph_output = None
        # The product routes CUDA-graph capture away from FlashInfer (capture goes to
        # CuTe), so the FI side is timed eagerly: host launch time is inside its events,
        # a conservative bound against the graph-timed market paths.
        self.eager = provider == "fi"

    def run(self):
        if self.graph is not None:
            return self.graph.replay()
        return self.call(*self.args)

    @property
    def contract_pools(self):
        # vLLM native slot 0 is null; all contract slots are shifted by one.
        offset = 1 if self.provider == "vllm31" else 0
        return tuple(pool[offset:] for pool in self.pools)

    @property
    def backend(self):
        return self.args[6].backend if self.provider == "vllm31" else self.provider

    @property
    def flashinfer_version(self):
        return self.args[6].flashinfer_version if self.provider == "vllm31" else None

    def reset(self):
        for pool, original in zip(self.pools, self.original, strict=False):
            pool.copy_(original)

    def capture(self):
        # Same preparation and whole call capture boundary as harness._time.
        self.call(*self.args)
        self.call(*self.args)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.call(*self.args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if self.eager:
            for _ in range(3):
                self.graph_output = self.call(*self.args)
            torch.cuda.synchronize()
            return
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_output = self.call(*self.args)
        torch.cuda.synchronize()
        for _ in range(3):
            self.graph.replay()
        torch.cuda.synchronize()


def max_abs(left, right):
    # Chunk comparisons to avoid three large temporary FP32 output buffers.
    maximum = 0.0
    left, right = left.reshape(-1), right.reshape(-1)
    for start in range(0, left.numel(), 1 << 20):
        diff = (
            (left[start : start + (1 << 20)].float() - right[start : start + (1 << 20)].float())
            .abs()
            .max()
            .item()
        )
        if not math.isfinite(diff):
            return None
        maximum = max(maximum, diff)
    return maximum


def eager_check(ours, sgl, n):
    ours.reset()
    sgl.reset()
    a = ours.call(*ours.args)
    b = sgl.call(*sgl.args)
    torch.cuda.synchronize()
    result = {
        "max_abs_output": max_abs(a, b),
        "max_abs_state": max_abs(
            ours.contract_pools[0][1 : n + 1], sgl.contract_pools[0][1 : n + 1]
        ),
        "max_abs_conv_state": max_abs(
            ours.contract_pools[1][1 : n + 1], sgl.contract_pools[1][1 : n + 1]
        ),
    }
    ours.reset()
    sgl.reset()
    return result


def timed(run, flush, events):
    flush.zero_()
    start, end = events
    start.record()
    run()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000.0
    if not math.isfinite(us) or us <= 0:
        raise ValueError("Invalid CUDA event interval")
    return us


REPS_PER_SAMPLE = 1  # set by --reps-per-sample
MIN_SAMPLE_US = 0.0  # set by --min-sample-us: sample = mean of k flushed replays, k * t >= this


def sample(run, flush, events):
    # One timing sample = median of REPS_PER_SAMPLE back-to-back flushed replays (odd count).
    values = sorted(timed(run, flush, events) for _ in range(REPS_PER_SAMPLE))
    return values[len(values) // 2]


def interleaved(instances, count, flush, *, reps_per_sample=None):
    """Rotate provider order each pair/round; median of rounds per provider."""
    for instance in instances:
        instance.reset()
    torch.cuda.synchronize()
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    samples = [[] for _ in instances]
    runs = [instance.run for instance in instances]
    reps = REPS_PER_SAMPLE if reps_per_sample is None else reps_per_sample
    mean = False
    if MIN_SAMPLE_US > 0:
        # Same k for every provider of this row, from the slowest one's warm time.
        probe = max(statistics.median(timed(run, flush, events) for _ in range(5)) for run in runs)
        reps = max(reps, math.ceil(MIN_SAMPLE_US / max(probe, 1e-3)))
        mean = True
        print(f"# sample: mean of {reps} flushed replays (probe {probe:.1f} us)", flush=True)
    for i in range(count):
        rounds = [[] for _ in instances]
        for r in range(reps):
            order = [(i + r + j) % len(instances) for j in range(len(instances))]
            for index in order:
                rounds[index].append(timed(runs[index], flush, events))
        for index, values in enumerate(rounds):
            samples[index].append(statistics.fmean(values) if mean else statistics.median(values))
    return samples


def paired(left, right, count, flush):
    return interleaved((left, right), count, flush)


def parity_verdict(speedup, eps):
    # eps > 0.02 is INCONCLUSIVE, whatever the speedup.
    return "inconclusive" if eps > 0.02 else "pass" if speedup >= 1 - eps else "fail"


def timing_statistics(ours, sgl, sgl_a, sgl_b, vllm=None, vllm_a=None, vllm_b=None, *, fi=None):
    """Compute unrounded paired ratios; legacy speedup/eps retain SGLang meaning."""
    speed_sgl = statistics.median(b / a for a, b in zip(ours, sgl, strict=False))
    eps_sgl = percentile95([abs(a / b - 1) for a, b in zip(sgl_a, sgl_b, strict=False)])
    result = {
        "ours_median_us": statistics.median(ours),
        "sgl_median_us": statistics.median(sgl),
        "speedup": speed_sgl,
        "eps": eps_sgl,
        "speedup_sglang": speed_sgl,
        "eps_sglang": eps_sgl,
        "vllm_median_us": None,
        "speedup_vllm": None,
        "eps_vllm": None,
        "faster": "sglang72",
        "speedup_best": speed_sgl,
        "eps_best": eps_sgl,
    }
    if vllm is not None:
        speed_vllm = statistics.median(b / a for a, b in zip(ours, vllm, strict=False))
        eps_vllm = percentile95([abs(a / b - 1) for a, b in zip(vllm_a, vllm_b, strict=False)])
        faster = "vllm31" if statistics.median(vllm) < statistics.median(sgl) else "sglang72"
        result.update(
            vllm_median_us=statistics.median(vllm),
            speedup_vllm=speed_vllm,
            eps_vllm=eps_vllm,
            faster=faster,
            speedup_best=statistics.median(
                min(b, c) / a for a, b, c in zip(ours, sgl, vllm, strict=False)
            ),
            eps_best=eps_vllm if faster == "vllm31" else eps_sgl,
        )
    result["verdict"] = parity_verdict(result["speedup_best"], result["eps_best"])
    if fi is not None:
        best = sgl if vllm is None else [min(b, c) for b, c in zip(sgl, vllm, strict=False)]
        result.update(
            fi_median_us=statistics.median(fi),
            speedup_fi_best=statistics.median(b / a for a, b in zip(fi, best, strict=False)),
            speedup_fi_sglang=statistics.median(b / a for a, b in zip(fi, sgl, strict=False)),
            speedup_fi_vllm=None
            if vllm is None
            else statistics.median(b / a for a, b in zip(fi, vllm, strict=False)),
            speedup_fi_vs_ours=statistics.median(b / a for a, b in zip(fi, ours, strict=False)),
        )
        result["verdict_fi"] = parity_verdict(result["speedup_fi_best"], result["eps_best"])
    return result


def trace_counts(instance, label):
    """Count GPU activities, not CPU launch APIs (one graph launch != one kernel).

    Chrome activity categories also include graph-node activities that some
    PyTorch versions omit from the correlated FunctionEvent list.
    """
    try:
        instance.reset()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            instance.run()
            torch.cuda.synchronize()
        trace_dir = Path(__file__).resolve().parent / ".whole-parity-traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parity-trace-", dir=trace_dir) as directory:
            path = Path(directory) / "trace.json"
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text()).get("traceEvents", [])
        kernels = sum(e.get("cat", "").lower() == "kernel" for e in events)
        memops = sum(e.get("cat", "").lower() in {"gpu_memcpy", "gpu_memset"} for e in events)
        if kernels == 0:
            raise RuntimeError("No CUDA kernel activities collected")
        return kernels, memops
    except Exception as exc:  # noqa: BLE001 - sanitize diagnostics
        print(f"# trace {label} unavailable ({type(exc).__name__}); continuing", flush=True)
        return None, None


def best_seconds(call, flush):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    return min(timed(call, flush, events) for _ in range(10)) * 1e-6


def measure_roof(device, flush):
    print("# roof: 1 GiB copy + BF16 sum, then 8192^3 BF16 matmul; best of 10", flush=True)
    if not memory_guard.check("parity-roof", estimated_gib=4):
        return {"bw_GBps": None, "c_tflops": None, "error": "memory_guard"}
    src = dst = scalar = a = b = c = None
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        byte_count = 1 << 30
        src = torch.ones(byte_count // 2, dtype=torch.bfloat16, device=device)
        dst = torch.empty_like(src)
        scalar = torch.empty((), dtype=torch.float32, device=device)
        copy_bw = (2 * byte_count) / best_seconds(lambda: dst.copy_(src), flush) / 1e9
        sum_bw = (
            (byte_count + 4)
            / best_seconds(lambda: torch.sum(src, dim=0, dtype=torch.float32, out=scalar), flush)
            / 1e9
        )
        src = dst = scalar = None
        cleanup()
        size = 8192
        a = torch.full((size, size), 0.01, dtype=torch.bfloat16, device=device)
        b = torch.full_like(a, 0.01)
        c = torch.empty_like(a)
        # cuBLAS BF16 matmul uses FP32 accumulation; disallow BF16 partial reductions.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        seconds = best_seconds(lambda: torch.matmul(a, b, out=c), flush)
        return {
            "bw_GBps": max(copy_bw, sum_bw),
            "c_tflops": 2 * size**3 / seconds / 1e12,
            "copy_GBps": copy_bw,
            "sum_GBps": sum_bw,
            "bw_method": "max(copy,read_reduce);best_of_10;read+write_bytes",
            "matmul_accum": "fp32",
            "matmul_output": "bf16",
        }
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        src = dst = scalar = a = b = c = None
        cleanup()


def row_estimate_gib(hk, hv, lengths, vllm=True, *, instance_count=None):
    # Normally four instances: ours, both comparators, one calibration partner.
    # --with-fi adds a fifth instance, accounted for via instance_count.
    # Calibration partners are destroyed sequentially. Count base/native/original
    # pools, conv history, chunk workspaces and the extra vLLM scratch slots.
    # memory_guard adds 8 GiB GPU margin and retains >=32 GiB host RAM on GB10.
    t, n = sum(lengths), len(lengths)
    q = 2 * hk * harness.K + hv * harness.V
    inputs = 2 * t * (q + hv * harness.V + 2 * hv)
    slots = 2 * n + 3  # native: null + contract n+2 + n scratch
    states = 3 * slots * (4 * hv * harness.K * harness.V + 2 * 3 * q)
    workspace = 48 * t * hv * harness.K
    count = (4 if vllm else 3) if instance_count is None else instance_count
    return max(12, count * (inputs + states + workspace) / 2**30 + 2)


def run_row(args, geom, ci, lengths, initial, device, flush, roof):
    hk, hv = map(int, geom.split("x"))
    with_fi = getattr(args, "with_fi", False)
    record = {
        "gpu": torch.cuda.get_device_name(),
        "geom": geom,
        "case": ci,
        "lens": lengths,
        "provider_ours": None,
        "provider_ours_requested": args.ours,
        "T": sum(lengths),
        "N": len(lengths),
        "ours_us": [],
        "sgl_us": [],
        "sglA_us": [],
        "sglB_us": [],
        "speedup": None,
        "eps": None,
        "verdict": "inconclusive",
        "eps_method": "linear_p95",
        "provider_vllm": "vllm31" if args.vllm else None,
        "vllm_us": [],
        "vllmA_us": [],
        "vllmB_us": [],
        "speedup_sglang": None,
        "eps_sglang": None,
        "speedup_vllm": None,
        "eps_vllm": None,
        "speedup_best": None,
        "eps_best": None,
        "faster": None,
    }
    record.update(
        v1_median_us=None,
        speedup_v1=None,
        eps_v1=None,
        verdict_v1="inconclusive",
        v1_us=[],
        v1A_us=[],
        v1B_us=[],
    )
    if with_fi:
        record.update(
            fi_us=[],
            fi_median_us=None,
            speedup_fi_best=None,
            speedup_fi_sglang=None,
            speedup_fi_vllm=None,
            speedup_fi_vs_ours=None,
            verdict_fi="inconclusive",
            kernels_fi=None,
        )
    record.update(geometry_metadata(hk, hv))
    if args.trace:
        record.update(
            kernels_ours=None,
            memops_ours=None,
            kernels_sgl=None,
            memops_sgl=None,
            kernels_vllm=None,
            memops_vllm=None,
        )
        if with_fi:
            record["memops_fi"] = None
    if args.roof:
        f, d = model(hk, hv, lengths, initial)
        record.update(F=f, D=d, t_roof=None, util=None, bound=None)
    start = time.perf_counter()
    instances = []
    ours = sgl_a = vllm = fi = v1 = partner = instance = None
    phase = "admission"
    try:
        if not memory_guard.check(
            f"parity-{geom}-{ci}",
            row_estimate_gib(
                hk,
                hv,
                lengths,
                args.vllm,
                instance_count=3 + int(args.vllm) + int(with_fi),
            ),
        ):
            record["error"] = "memory_guard"
            return record
        phase = "build"
        providers = (args.ours, "sglang72") + (("vllm31",) if args.vllm else ())
        if with_fi:
            if args.ours != "product":
                raise CuteProviderRequiredError("FlashInfer comparison requires product ours")
            providers += ("fi",)
        for provider in providers:
            phase = "build_" + provider
            kwargs = {"port_options": args.port_options}
            if provider == "fi" or (
                provider == "product"
                and (with_fi or torch.cuda.get_device_capability(device) == (12, 0))
            ):
                kwargs["factory"] = product_factory(
                    hk, hv, device, sum(lengths), len(lengths), flashinfer=provider == "fi"
                )
            instances.append(
                Instance(provider, hk, hv, lengths, initial, 1234 + ci, device, **kwargs)
            )
            kwargs.clear()
            if len(instances) == 1:
                record["provider_ours"] = instances[0].label
                if hk == 8 and args.ours == "product":
                    require_cute(instances[0].label)
        ours, sgl_a = instances[:2]
        vllm = instances[2] if args.vllm else None
        fi = instances[-1] if with_fi else None
        if vllm is not None:
            record["vllm_backend"] = vllm.backend
            record["vllm_flashinfer_version"] = vllm.flashinfer_version
            if hk == 8 and vllm.backend != "flashinfer":
                raise FlashInferRequiredError("TP2 requires FlashInfer")
        phase = "eager_check"
        if args.ours == "port":
            if v1 is None:
                # TP2 (Hk=8): this branch's v1 is not CuTe there (the CuTe TP2 v1 ships in
                # rc-0.1.13). The eager check then uses the default v1; provider_v1 names it.
                factory = (
                    product_factory(hk, hv, device, sum(lengths), len(lengths))
                    if torch.cuda.get_device_capability(device) == (12, 0) and hk != 8
                    else None
                )
                v1 = Instance(
                    "product", hk, hv, lengths, initial, 1234 + ci, device, factory=factory
                )
                factory = None
            record["provider_v1"] = v1.label
            ours.reset()
            v1.reset()
            record["port_vs_product"] = harness.check_port_product(
                ours.call, ours.args, v1.call, v1.args
            )
            v1 = None
            cleanup()
        record.update(eager_check(ours, sgl_a, len(lengths)))
        if vllm is not None:
            record.update(
                {
                    key + "_vllm": value
                    for key, value in eager_check(ours, vllm, len(lengths)).items()
                }
            )
        if fi is not None:
            for other, suffix in ((ours, "_fi"), (sgl_a, "_sglang_fi"), (vllm, "_vllm_fi")):
                if other is not None:
                    record.update(
                        {
                            key + suffix: value
                            for key, value in eager_check(fi, other, len(lengths)).items()
                        }
                    )
            other = None
        print(
            f"# check {geom} case={ci} "
            + json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key.startswith("max_abs_") or key == "port_vs_product"
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        for label, instance in zip(providers, instances, strict=False):
            phase = "capture_" + label
            if label != "fi":
                print(f"# {geom} case={ci}: capture + 3 warmups {label}", flush=True)
            instance.capture()
        instance = None
        phase = "paired_timing"
        print(
            f"# {geom} case={ci}: {args.pairs} interleaved pairs; "
            "20 self-calibration pairs per comparator",
            flush=True,
        )
        samples = interleaved(instances, args.pairs, flush)
        ours_us, sgl_us = samples[:2]
        vllm_us = samples[2] if args.vllm else None
        fi_us = samples[-1] if with_fi else None
        calibration = {}
        # Only one extra graph/input pool exists at a time, including during capture.
        for provider, primary in (("sglang72", sgl_a), ("vllm31", vllm)):
            if primary is None:
                continue
            phase = "calibration_" + provider
            factory = (
                product_factory(hk, hv, device, sum(lengths), len(lengths))
                if provider == "product" and torch.cuda.get_device_capability(device) == (12, 0)
                else None
            )
            partner = Instance(
                provider, hk, hv, lengths, initial, 1234 + ci, device, factory=factory
            )
            factory = None
            partner.capture()
            calibration[provider] = paired(primary, partner, CALIBRATION_PAIRS, flush)
            partner = None
            cleanup()
        a_us, b_us = calibration["sglang72"]
        va_us, vb_us = calibration.get("vllm31", (None, None))
        record.update(
            timing_statistics(ours_us, sgl_us, a_us, b_us, vllm_us, va_us, vb_us, fi=fi_us)
        )
        if args.ours == "product":
            record["v1_median_us"] = record["ours_median_us"]
        if with_fi:
            record["fi_us"] = [round(x, 2) for x in fi_us]
        for key, values in (
            ("ours_us", ours_us),
            ("sgl_us", sgl_us),
            ("sglA_us", a_us),
            ("sglB_us", b_us),
            ("vllm_us", vllm_us),
            ("vllmA_us", va_us),
            ("vllmB_us", vb_us),
        ):
            record[key] = [] if values is None else [round(x, 2) for x in values]
        if args.roof and roof.get("bw_GBps") and roof.get("c_tflops"):
            bw_us = record["D"] / (roof["bw_GBps"] * 1e3)
            compute_us = record["F"] / (roof["c_tflops"] * 1e6)
            roof_us = max(bw_us, compute_us)
            record.update(
                t_roof=roof_us,
                util=roof_us / statistics.median(ours_us),
                bound="bw" if bw_us >= compute_us else "compute",
            )
        if args.trace:
            phase = "trace"
            print(f"# {geom} case={ci}: untimed CUDA traces", flush=True)
            record["kernels_ours"], record["memops_ours"] = trace_counts(ours, "ours")
            record["kernels_sgl"], record["memops_sgl"] = trace_counts(sgl_a, "sglang72")
            if vllm is not None:
                record["kernels_vllm"], record["memops_vllm"] = trace_counts(vllm, "vllm31")
            if fi is not None:
                record["kernels_fi"], record["memops_fi"] = trace_counts(fi, "fi")
        if args.extra_ours:
            # Preserve the market experiment and its peak instance count on 32 GiB GPUs.
            instances.clear()
            sgl_a = vllm = fi = primary = None
            cleanup()
            phase = "paired_v1"
            factory = (
                product_factory(hk, hv, device, sum(lengths), len(lengths))
                if torch.cuda.get_device_capability(device) == (12, 0)
                else None
            )
            v1 = Instance("product", hk, hv, lengths, initial, 1234 + ci, device, factory=factory)
            factory = None
            v1.capture()
            print(f"# {geom} case={ci}: {args.pairs} port/v1 pairs; 20 v1 self pairs", flush=True)
            ours_v1_us, v1_us = paired(ours, v1, args.pairs, flush)
            phase = "calibration_product"
            factory = (
                product_factory(hk, hv, device, sum(lengths), len(lengths))
                if torch.cuda.get_device_capability(device) == (12, 0)
                else None
            )
            partner = Instance(
                "product", hk, hv, lengths, initial, 1234 + ci, device, factory=factory
            )
            factory = None
            partner.capture()
            v1a_us, v1b_us = paired(v1, partner, CALIBRATION_PAIRS, flush)
            speedup_v1 = statistics.median(b / a for a, b in zip(ours_v1_us, v1_us, strict=False))
            eps_v1 = percentile95([abs(a / b - 1) for a, b in zip(v1a_us, v1b_us, strict=False)])
            record.update(
                ours_v1_median_us=statistics.median(ours_v1_us),
                ours_v1_us=[round(x, 2) for x in ours_v1_us],
                v1_median_us=statistics.median(v1_us),
                speedup_v1=speedup_v1,
                eps_v1=eps_v1,
                verdict_v1=parity_verdict(speedup_v1, eps_v1),
                v1_us=[round(x, 2) for x in v1_us],
                v1A_us=[round(x, 2) for x in v1a_us],
                v1B_us=[round(x, 2) for x in v1b_us],
            )
    except Exception as exc:  # noqa: BLE001 - sanitize diagnostics
        # Never echo exception text: compilation errors can include env or argv.
        record.update(error=type(exc).__name__, error_phase=phase, verdict="inconclusive")
        if with_fi:
            record["verdict_fi"] = "inconclusive"
        if isinstance(
            exc,
            (
                torch.cuda.OutOfMemoryError,
                CuteProviderRequiredError,
                FlashInferRequiredError,
                harness.PortPrefillUnavailableError,
            ),
        ):
            record["stop_after_row"] = True
        if isinstance(exc, harness.PortPrefillUnavailableError):
            record["error_hint"] = harness.PORT_UNAVAILABLE
    finally:
        instances.clear()
        ours = sgl_a = vllm = fi = v1 = partner = instance = primary = None
        other = kwargs = factory = None
        cleanup()
        record["row_seconds"] = round(time.perf_counter() - start, 3)
    return record


def summary(rows):
    columns = (
        "ours_median_us",
        "v1_median_us",
        "vllm_median_us",
        "sgl_median_us",
        "speedup_v1",
        "speedup_vllm",
        "speedup_sglang",
        "speedup_best",
    )
    with_fi = any("fi_us" in row for row in rows)
    if with_fi:
        columns += (
            "fi_median_us",
            "speedup_fi_best",
            "speedup_fi_sglang",
            "speedup_fi_vllm",
            "speedup_fi_vs_ours",
        )
    print("# summary: ratios = comparator/ours; >1 means ours faster", flush=True)
    print(
        "# gpu | geom | row | T | N | ours us | v1 us | vllm us | sglang us | "
        "v1/ours | vllm/ours | sglang/ours | best/ours"
        + (" | fi us | fi best | fi sglang | fi vllm | fi ours" if with_fi else "")
        + " | verdict | v1 verdict"
        + (" | verdict_fi" if with_fi else ""),
        flush=True,
    )
    for row in rows:
        values = ["-" if row.get(k) is None else f"{row[k]:.4f}" for k in columns]
        print(
            f"# {row['gpu']} | {row['geom']} | {row['case']} | {row['T']} | {row['N']} | "
            + " | ".join(values)
            + f" | {row['verdict']} | {row['verdict_v1']}"
            + (f" | {row.get('verdict_fi')}" if with_fi else ""),
            flush=True,
        )
    for geom in list(dict.fromkeys(row["geom"] for row in rows)) + ["overall"]:
        group = [row for row in rows if geom == "overall" or row["geom"] == geom]
        counts = {
            v: sum(row["verdict"] == v for row in group) for v in ("pass", "fail", "inconclusive")
        }
        # Retain the legacy overall geomean/min line for SGLang-only log readers.
        for key in ("speedup", "speedup_sglang", "speedup_vllm", "speedup_best"):
            speeds = [row[key] for row in group if row.get(key) is not None]
            stats = (
                f"geomean={math.exp(statistics.mean(map(math.log, speeds))):.6f} "
                f"min={min(speeds):.6f}"
                if speeds
                else "geomean=- min=-"
            )
            label = "" if key == "speedup" else key + " "
            print(
                f"# {geom}: {label}{stats} measured={len(speeds)}/{len(group)} {counts}", flush=True
            )


def parse_args(argv=None):
    ap = QuietParser(
        prog="whole_parity.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", choices=("all", "small", "long"), default="all")
    ap.add_argument("--geoms", default="16x32,16x48", help="subset of 16x32,16x48,8x16,8x24")
    ap.add_argument(
        "--cases", default="", help="comma-separated GLOBAL indices 0..16; intersects --set"
    )
    ap.add_argument("--pairs", type=int, default=None, help="default: 20; 30 with --with-fi")
    ap.add_argument(
        "--with-fi", action="store_true", help="include verified product FlashInfer provider"
    )
    ap.add_argument("--reps-per-sample", type=int, default=1)
    ap.add_argument(
        "--min-sample-us",
        type=float,
        default=0.0,
        help="each sample = mean of k flushed replays with k*t >= this (GB10 noise protocol)",
    )
    ap.add_argument(
        "--ours",
        choices=("product", "port"),
        default="product",
        help="ours side: product v1 (CuTe on sm120) or the new port",
    )
    ap.add_argument(
        "--extra-ours", action="store_true", help="with --ours port, also pair against product v1"
    )
    ap.add_argument(
        "--port-options",
        type=harness.parse_port_options,
        default={},
        metavar="k=v,k=v",
        help="PortPrefill constructor options: ints, true/false, or strings",
    )
    ap.add_argument(
        "--vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include upstream-dispatched vllm31 (default on; --no-vllm disables)",
    )
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--roof", action="store_true")
    ap.add_argument("--out", type=Path, help="append JSONL to a file inside this worktree")
    args = ap.parse_args(argv)
    if args.pairs is None:
        args.pairs = 30 if args.with_fi else 20
    try:
        args.geoms = list(dict.fromkeys(args.geoms.split(",")))
        if not args.geoms or any(g not in GEOMETRIES for g in args.geoms):
            raise ValueError
        args.cases = set(map(int, args.cases.split(","))) if args.cases else set(range(17))
        if (
            not args.cases <= set(range(17))
            or args.pairs < 1
            or args.reps_per_sample < 1
            or args.reps_per_sample % 2 == 0
            or not math.isfinite(args.min_sample_us)
            or args.min_sample_us < 0
            or (args.with_fi and args.ours != "product")
            or (args.extra_ours and args.ours != "port")
        ):
            raise ValueError
        selected = (
            set(range(8))
            if args.set == "small"
            else set(range(8, 17))
            if args.set == "long"
            else set(range(17))
        )
        args.cases &= selected
        if not args.cases:
            raise ValueError
    except ValueError:
        ap.error("Invalid workload selection")
    return args


def _gpu_apps(label):
    """Host quietness record: other compute processes on the GPU (pid, MiB only)."""
    import subprocess

    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        out = ["unavailable"]
    mine = str(os.getpid())
    others = []
    for line in out:
        pid, sep, memory = line.partition(",")
        amount = memory.strip().split()
        if sep and pid.strip().isdigit() and pid.strip() != mine and amount and amount[0].isdigit():
            others.append({"pid": int(pid), "MiB": int(amount[0])})
    print(
        "# gpu_apps " + label + ": " + json.dumps(others if out != ["unavailable"] else None),
        flush=True,
    )


def main():
    global REPS_PER_SAMPLE
    args = parse_args()
    REPS_PER_SAMPLE = args.reps_per_sample
    _gpu_apps("start")
    global MIN_SAMPLE_US
    MIN_SAMPLE_US = args.min_sample_us
    output = None
    try:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            output = args.out.open("a", encoding="utf-8")
        if not torch.cuda.is_available():
            print("# CUDA GPU required; no benchmark executed", flush=True)
            return 1
        if not memory_guard.check("parity-start", estimated_gib=1):
            print("# insufficient memory for flush buffer; stopped", flush=True)
            return 1
        device = torch.device("cuda")
        flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
        roof = {}
        if args.roof:
            try:
                with torch.no_grad():
                    roof = measure_roof(device, flush)
            except Exception as exc:  # noqa: BLE001 - sanitize diagnostics
                roof = {"bw_GBps": None, "c_tflops": None, "error": type(exc).__name__}
                emit({"roof": roof, "gpu": torch.cuda.get_device_name()}, output)
                # Do not continue allocating after a roof failure/OOM.
                return 1
            emit({"roof": roof, "gpu": torch.cuda.get_device_name()}, output)
        print(
            f"# reps_per_sample={REPS_PER_SAMPLE} (sample = median of that many flushed replays)",
            flush=True,
        )
        print(
            "# protocol: graph; p95 linear interpolation; calibration=20 pairs; "
            "global cases=0..16; arrays rounded to 0.01 us; t_roof in us",
            flush=True,
        )
        rows = []
        for geom in args.geoms:
            for ci, (lengths, initial) in enumerate(harness.SMALL + harness.LONG):
                if ci not in args.cases:
                    continue
                print(f"# start {geom} case={ci} lens={lengths}", flush=True)
                with torch.no_grad():
                    record = run_row(args, geom, ci, lengths, initial, device, flush, roof)
                emit(record, output)
                rows.append(record)
                if record.get("stop_after_row"):
                    summary(rows)
                    return 1
        summary(rows)
        _gpu_apps("end")
        return int(any("error" in row for row in rows))
    except Exception as exc:  # noqa: BLE001 - sanitize diagnostics
        print(f"# benchmark stopped ({type(exc).__name__})", flush=True)
        return 1
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
