# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
# Per-candidate external failures must be reported safely and not abort other rows.

"""Paired hd128 attention-prologue benchmark against SGLang and a vLLM surrogate.

Run as a module or script. Each graph contains 50 calls on a rotating pool of
50 identical pre-filled QKV buffers, one per call; all slots reset before each
replay, outside CUDA-event timing. Read-only inputs and KV caches stay shared.
Pools over 4 GiB require free CUDA memory of at least twice the pool size;
otherwise the row is skipped. Twenty independent CuTe self-pairs calibrate
p95 noise per row. Comparisons alternate order, use
an exact binomial order-statistic 95% median CI, and retry ambiguous results
with at least 40 fresh pairs. The primary bar compares against real SGLang
dispatch. The market bar requires SGLang and the vLLM norm/RoPE SURROGATE,
selecting the one with lower median latency. Engine is reported separately.

--check-imports, --build-sglang and --check-vllm are preflight-only modes and
never launch CUDA kernels. Compiler/dependency diagnostics are discarded so
reports contain neither machine paths nor potentially credential-bearing logs.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

# Permit direct execution from an uninstalled checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

CALLS = 50
ROWS = (1, 2, 3, 4, 16, 32, 48, 64)
CEILINGS = {"RTX 5090": 1510.4, "RTX PRO 6000": 1452.5, "GB10": 219.0}
EPS = 1e-6


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-python", type=Path, help="72730d6 market tree's python directory")
    p.add_argument(
        "--vllm-cache",
        type=Path,
        default=Path("/tmp/attention-prologue-vllm-cache"),
        help="wheel/extension cache directory",
    )
    p.add_argument("--rows", default=",".join(map(str, ROWS)), help="comma-separated token counts")
    p.add_argument("--pairs", type=int, default=20, help="initial pairs (minimum 6)")
    p.add_argument("--json", type=Path, help="write complete report and paired samples")
    p.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument(
        "--sweep", action="store_true", help="also pair each launcher configuration against default"
    )
    p.add_argument(
        "--sweep-configs", help="semicolon-separated q_rows,k_rows,v_rows,warps,stream tuples"
    )
    p.add_argument(
        "--check-imports", action="store_true", help="only import the real SGLang kernel modules"
    )
    p.add_argument(
        "--build-sglang",
        action="store_true",
        help="only compile/load the three JIT modules; no launch",
    )
    p.add_argument(
        "--check-vllm", action="store_true", help="only download/extract/symbol-check/load vLLM"
    )
    p.add_argument(
        "--vllm-distinct-outputs",
        action="store_true",
        help="cache-pressure diagnostic: retain each vLLM output; never produces a verdict",
    )
    return p


def protocol_metadata(args):
    return {
        "vllm_distinct_outputs": args.vllm_distinct_outputs,
        "diagnostic": args.vllm_distinct_outputs,
    }


def diagnostic_verdicts(value):
    """Suppress comparison verdicts throughout a diagnostic report, including retries."""
    if isinstance(value, dict):
        if "verdict" in value:
            value["verdict"] = "diagnostic"
        for child in value.values():
            diagnostic_verdicts(child)
    elif isinstance(value, list):
        for child in value:
            diagnostic_verdicts(child)


def percentile95(values):
    ordered = sorted(values)
    offset = 0.95 * (len(ordered) - 1)
    lo = int(offset)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (offset - lo)


def median_ci(values):
    """Exact two-sided binomial interval; coverage is at least 95%, including ties."""
    ordered = sorted(values)
    n = len(ordered)
    valid = [
        k for k in range(1, n // 2 + 1) if 2 * sum(math.comb(n, i) for i in range(k)) <= 0.05 * 2**n
    ]
    if not valid:
        raise ValueError("At least six pairs are required for a finite 95% median CI")
    k = max(valid)
    return [ordered[k - 1], ordered[n - k]]


def verdict(ci, epsilon):
    if ci[0] > 1 + epsilon:
        return "faster"
    if ci[1] < 1 - epsilon:
        return "below"
    return "noise"


class PoolMemoryError(RuntimeError):
    """A full pool cannot fit with the required free-memory margin."""


def check_pool_memory(tokens, device):
    size = CALLS * tokens * 6144 * 2
    if size > 4 * 1024**3 and torch.cuda.mem_get_info(device)[0] < 2 * size:
        raise PoolMemoryError("Insufficient free memory for the full QKV pool")


class Buffers:
    def __init__(self, tokens, device, seed):
        check_pool_memory(tokens, device)
        generator = torch.Generator().manual_seed(seed + tokens)
        self.original = torch.randn(tokens, 48 * 128, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        self.pool = self.original.unsqueeze(0).expand(CALLS, -1, -1).clone()
        self.qkv = self.pool.unbind(0)
        self.qw = (1 + 0.2 * torch.randn(128, generator=generator)).to(
            device=device, dtype=torch.bfloat16
        )
        self.kw = (1 + 0.2 * torch.randn(128, generator=generator)).to(
            device=device, dtype=torch.bfloat16
        )
        frequencies = 1e6 ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
        angles = torch.arange(4096, dtype=torch.float64)[:, None] * frequencies
        self.table = torch.cat((angles.cos(), angles.sin()), dim=1).to(
            device=device, dtype=torch.float32
        )
        # vLLM warms its cached table in query.dtype before the compiled region.
        self.vllm_table = self.table.to(torch.bfloat16)
        self.positions = torch.randint(4096, (tokens,), generator=generator).to(device)
        # All candidates use the same random permutation; zero is SGLang padding.
        slots = max(256, -(-(tokens + 1) // 16) * 16)
        self.indices = (torch.randperm(slots - 1, generator=generator)[:tokens] + 1).to(
            device=device, dtype=torch.int32
        )
        self.indices64 = self.indices.long()  # vLLM expects int64; outside timing.
        self.kc = torch.zeros(slots, 8, 128, device=device, dtype=torch.bfloat16)
        self.vc = torch.zeros_like(self.kc)
        self.scale = torch.ones((), device=device, dtype=torch.float32)
        self.q, self.k, self.v = zip(
            *(qkv.view(tokens, 48, 128).split((32, 8, 8), dim=1) for qkv in self.qkv),
            strict=True,
        )
        self.args = tuple(
            (qkv, self.qw, self.kw, self.table, self.positions, self.kc, self.vc, self.indices)
            for qkv in self.qkv
        )

    def reset(self):
        self.pool.copy_(self.original)
        self.kc.zero_()
        self.vc.zero_()

    def reference(self):
        q, k, v = self.original.cpu().double().view(-1, 48, 128).split((32, 8, 8), 1)
        c, s = self.table.cpu().double()[self.positions.cpu()].unsqueeze(1).chunk(2, -1)
        outputs, conditioning = [], []
        for heads, weights in ((q, self.qw), (k, self.kw)):
            normalized = heads / (heads.square().mean(-1, keepdim=True) + EPS).sqrt()
            normalized = normalized * weights.cpu().double()
            a, b = normalized.chunk(2, -1)
            outputs.append(torch.cat((a * c - b * s, b * c + a * s), -1))
            conditioning.append(
                torch.cat(((a * c).abs() + (b * s).abs(), (b * c).abs() + (a * s).abs()), -1)
            )
        return (*outputs, v), (*conditioning, v.abs())


def bf16_ulp(magnitude):
    import torch

    # Spacing at the reference binade, with BF16 subnormals handled explicitly.
    exponent = torch.floor(torch.log2(magnitude.abs().clamp_min(2.0**-126))) - 7
    return torch.pow(2.0, exponent)


def check(operation, b):
    expected, conditioning = b.reference()
    b.reset()
    q = operation(0)
    torch.cuda.synchronize()
    actual = (
        q.detach().cpu().double(),
        b.kc[b.indices64].cpu().double(),
        b.vc[b.indices64].cpu().double(),
    )
    errors = {}
    okay = True
    for label, got, ref, scale in zip(("q", "k", "v"), actual, expected, conditioning, strict=True):
        difference = (got.reshape_as(ref) - ref).abs()
        finite = bool(torch.isfinite(difference).all())
        raw = difference / bf16_ulp(ref)
        # Cancellation can make output-relative ULP arbitrarily large. Four
        # BF16 ULPs of |a*cos|+|b*sin| allow native intermediate BF16 rounding.
        bounded = difference / bf16_ulp(scale)
        passed = finite and (
            bool((difference == 0).all()) if label == "v" else bool((bounded <= 4).all())
        )
        errors[label] = {
            "max_abs": difference.max().item() if finite else None,
            "max_ulp": raw.max().item() if finite else None,
            "max_conditioned_ulp": bounded.max().item() if finite else None,
            "passed": passed,
        }
        okay &= passed
    untouched = torch.ones(b.kc.shape[0], dtype=torch.bool, device=b.kc.device)
    untouched[b.indices64] = False
    unchanged = bool((b.kc[untouched] == 0).all() and (b.vc[untouched] == 0).all())
    errors["untouched_cache_rows"] = unchanged
    b.reset()
    return okay and unchanged, errors


class Graph:
    def __init__(self, operation, buffers):
        self.buffers = buffers
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                buffers.reset()
                operation(0)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        buffers.reset()
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        self.outputs = [] if getattr(buffers, "retain_outputs", False) else None
        with torch.cuda.graph(self.graph):
            for slot in range(CALLS):
                self.output = operation(slot)
                if self.outputs is not None:
                    self.outputs.append(self.output)
        self.graph.instantiate()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            self.sample()

    def sample(self):
        self.buffers.reset()
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / CALLS

    def kernels(self):
        try:
            return self.graph_nodes()
        except Exception:
            return self.profile_kernels()

    def graph_nodes(self):
        from cuda.bindings import driver

        def checked(result):
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA graph inspection failed")
            return result[1:]

        names = Counter()
        node_types = Counter()

        def visit(handle):
            _, count = checked(driver.cuGraphGetNodes(handle, 0))
            nodes, _ = checked(driver.cuGraphGetNodes(handle, count))
            for node in nodes:
                (kind,) = checked(driver.cuGraphNodeGetType(node))
                node_types[kind.name] += 1
                if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                    (params,) = checked(driver.cuGraphKernelNodeGetParams(node))
                    (name,) = checked(driver.cuFuncGetName(params.func))
                    names[name.decode("utf-8", errors="replace")] += 1
                elif kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
                    (child,) = checked(driver.cuGraphChildGraphNodeGetGraph(node))
                    visit(child)

        visit(driver.CUgraph(self.graph.raw_cuda_graph()))
        if not names or sum(names.values()) % CALLS:
            raise RuntimeError("Unexpected CUDA graph kernel count")
        return {
            "kernels_per_call": sum(names.values()) / CALLS,
            "kernel_names": {name: count / CALLS for name, count in sorted(names.items())},
            "graph_nodes_per_call": {kind: count / CALLS for kind, count in node_types.items()},
            "kernel_count_method": "cuGraphGetNodes + cuGraphNodeGetType + cuFuncGetName",
        }

    def profile_kernels(self):
        # Profile the captured replay itself, rather than estimating from APIs.
        from torch.profiler import ProfilerActivity, profile

        self.buffers.reset()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.graph.replay()
            torch.cuda.synchronize()
        names = Counter(
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.lower().startswith(("memcpy", "memset", "[memory"))
        )
        if not names or sum(names.values()) % CALLS:
            raise RuntimeError("Profiler did not yield complete per-call kernel counts")
        return {
            "kernels_per_call": sum(names.values()) / CALLS,
            "kernel_names": {name: count / CALLS for name, count in sorted(names.items())},
            "kernel_count_method": "CUDA profiler: one 50-call graph replay",
        }


def paired(left, right, count):
    samples = []
    for index in range(count):
        if index % 2:
            b = right.sample()
            a = left.sample()
        else:
            a = left.sample()
            b = right.sample()
        samples.append([a, b])
    return samples


def summarize(samples, epsilon):
    ratios = [b / a for a, b in samples]
    ci = median_ci(ratios)
    return {
        "cute_us": statistics.median(a for a, _ in samples),
        "comparator_us": statistics.median(b for _, b in samples),
        "speedup": statistics.median(ratios),
        "ci95": ci,
        "verdict": verdict(ci, epsilon),
        "pairs": len(samples),
        "samples_us": samples,
    }


def compare(left, right, count, epsilon):
    initial = summarize(paired(left, right, count), epsilon)
    if initial["verdict"] != "noise":
        return initial
    retried = summarize(paired(left, right, max(40, 2 * count)), epsilon)
    retried["initial"] = initial
    return retried


def engine_operation(b):
    from ctikernels.norm.inference import rms_norm_
    from ctikernels.rotary.inference import apply_rotary_
    from ctikernels.store import store_cache

    def run(slot):
        q, k, v = b.q[slot], b.k[slot], b.v[slot]
        rms_norm_(q, b.qw, EPS, zero_centered=False)
        rms_norm_(k, b.kw, EPS, zero_centered=False)
        apply_rotary_(q.flatten(1), k.flatten(1), b.table, b.positions, head_dim=128)
        store_cache(b.kc, b.vc, b.indices, k.flatten(1), v.flatten(1))
        return q

    return run


def sglang_operation(b, modules):
    norm, rope, cache = modules

    def run(slot):
        q, k, v = b.q[slot], b.k[slot], b.v[slot]
        norm.fused_inplace_qknorm(q, k, b.qw, b.kw, EPS, head_dim=128)
        rope.apply_rope_with_cos_sin_cache_inplace(
            q, k, b.table, b.positions, is_neox=True, fused_args=None
        )
        cache.store_cache(
            k.flatten(1),
            v.flatten(1),
            b.kc.flatten(1),
            b.vc.flatten(1),
            b.indices,
            row_bytes=2048,
            v_row_bytes=2048,
            size_limit=b.kc.shape[0],
        )
        return q

    return run


def vllm_operation(b, native, cache_op):
    if cache_op is None:
        from ctikernels.store import store_cache

    def run(slot):
        q, k = native(b.qkv[slot], b.qw, b.kw, b.vllm_table, b.positions)
        if cache_op is None:
            store_cache(b.kc, b.vc, b.indices, k.flatten(1), b.v[slot].flatten(1))
        else:
            cache_op(
                k,
                b.v[slot],
                b.kc.view(-1, 16, 8, 128),
                b.vc.view(-1, 16, 8, 128),
                b.indices64,
                "auto",
                b.scale,
                b.scale,
            )
        return q

    return run


def prepare(name, factory, tokens, args, metadata=None):
    record = dict(metadata or {})
    record["status"] = "NA (unavailable)"
    phase = "construct"
    graph = None
    try:
        with quiet_external():
            b = Buffers(tokens, args.device, args.seed)
            b.retain_outputs = args.vllm_distinct_outputs and name.startswith("vllm")
            op = factory(b)
            phase = "sanity"
            okay, errors = check(op, b)
            record["errors"] = errors
            if okay:
                phase = "capture"
                graph = Graph(op, b)
                record["status"] = "ok"
                try:
                    record.update(graph.kernels())
                except Exception as error:
                    record.update(
                        kernels_per_call=None,
                        kernel_names={},
                        kernel_count_error=type(error).__name__,
                    )
            else:
                record["status"] = "NA (mismatch)"
    except PoolMemoryError:
        record.update(status="NA (pool memory)", error_type="PoolMemoryError", failure_phase=phase)
    except Exception as error:
        record.update(error_type=type(error).__name__, failure_phase=phase)
    if "errors" in record:
        parts = []
        for label in ("q", "k", "v"):
            e = record["errors"][label]
            if e["max_abs"] is None:
                parts.append(f"{label}: nonfinite")
            else:
                parts.append(
                    f"{label}: abs={e['max_abs']:.5g} ulp={e['max_ulp']:.3g} conditioned={e['max_conditioned_ulp']:.3g}"
                )
        print(f"T={tokens} {name} {record['status']}; " + "; ".join(parts), flush=True)
    else:
        print(
            f"T={tokens} {name} {record['status']} ({phase}: {record.get('error_type', 'unknown')})",
            flush=True,
        )
    return graph, record


def sweep_configs(text):
    if text:
        configs = [tuple(map(int, part.split(","))) for part in text.split(";")]
    else:
        configs = [
            (q, k, v, w, s)
            for (q, k, v), w, s in itertools.product(
                ((1, 1, 1), (2, 1, 1), (2, 2, 2), (4, 2, 2), (4, 4, 4)), (2, 4, 8), (0, 1)
            )
        ]
    for config in configs:
        if (
            len(config) != 5
            or any(v not in (1, 2, 4) for v in config[:3])
            or config[3] not in (2, 4, 8)
            or config[4] not in (0, 1)
        ):
            raise ValueError("Invalid sweep configuration")
    return [(*c[:4], bool(c[4])) for c in configs]


def run_sweep(op, baseline, tokens, args, epsilon):
    launcher = op._kernel
    if "config" not in inspect.signature(launcher.__call__).parameters:
        return {"status": "NA (Launcher config override unavailable)"}
    results = []
    for config in sweep_configs(args.sweep_configs):

        def factory(b, config=config):
            def run(slot):
                launcher(*b.args[slot], EPS, 32, 8, config=config)
                return b.q[slot]

            return run

        graph, record = prepare(f"sweep {config}", factory, tokens, args, {"config": config})
        if graph is not None:
            record["paired"] = compare(baseline, graph, args.pairs, epsilon)
            record["us"] = record["paired"]["comparator_us"]
            print(f"T={tokens} sweep {config}: {record['us']:.3f} us/call", flush=True)
        results.append(record)
        del graph
    valid = [r for r in results if r["status"] == "ok"]
    best = min(valid, key=lambda r: r["us"]) if valid else None
    return {"status": "ok", "results": results, "best_config": best["config"] if best else None}


def render(report):
    print(f"\nDevice: {report['device']}; 50 calls/replay; times in us/call.")
    print("ULP gate: <=4 ULP of the RoPE contribution magnitude; V must be exact.")
    print(
        "\n| T | workload | cute | engine | sglang | vllm* | kernels c/e/s/v | vs SGLang [95% CI] | verdict | market | speedup [95% CI] | verdict | SOL us / % |"
    )
    print("|---:|:---|---:|---:|---:|---:|:---|:---|:---|:---|:---|:---|:---|")
    for row in report["rows"]:
        candidates = row["candidates"]
        timings, kernels = [], []
        for name in ("cute", "engine", "sglang", "vllm"):
            r = candidates.get(name, {})
            timings.append(
                f"{r['us']:.3f}" if "us" in r else row.get("status", r.get("status", "NA"))
            )
            k = r.get("kernels_per_call")
            kernels.append(f"{k:g}" if k is not None else "NA")
        bar = row.get("bar", {})
        ratio = (
            f"{bar['speedup']:.3f} [{bar['ci95'][0]:.3f}, {bar['ci95'][1]:.3f}]"
            if "speedup" in bar
            else "NA"
        )
        sglang = row.get("bar_sglang", {})
        sglang_ratio = (
            f"{sglang['speedup']:.3f} [{sglang['ci95'][0]:.3f}, {sglang['ci95'][1]:.3f}]"
            if "speedup" in sglang
            else "NA"
        )
        comparator = bar.get("comparator", "NA")
        if comparator == "vllm":
            comparator = "vllm*"
        sol = row.get("sol_us")
        sol_text = (
            f"{sol:.4f} / {row['sol_percent']:.2f}%"
            if sol is not None and "sol_percent" in row
            else "NA"
        )
        print(
            f"| {row['tokens']} | {row['workload']} | "
            + " | ".join(timings)
            + f" | {'/'.join(kernels)} | {sglang_ratio} | {sglang.get('verdict', 'NA')} | {comparator} | {ratio} | {bar.get('verdict', 'NA')} | {sol_text} |"
        )
    print(
        "* vllm: SURROGATE norm/RoPE under generic Inductor, not vLLM's compiled graph; "
        "wheel KV write when available. Primary verdict: vs SGLang."
    )
    if report.get("vllm", {}).get("vllm_kv_write") == "stand-in":
        print("vLLM KV write: stand-in (ctikernels.store_cache); bar is provisional.")


def write_report(report, destination):
    if report.get("protocol", {}).get("diagnostic"):
        diagnostic_verdicts(report)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"JSON: {destination.name}")


def main(argv=None):
    args = parser().parse_args(argv)
    global torch, quiet_external
    import torch

    from benchmark.kernel_tuning._prologue_market import (
        build_sglang,
        compile_vllm,
        load_vllm_cache,
        quiet_external,
        sglang_modules,
    )

    preflight = args.check_imports or args.build_sglang or args.check_vllm
    if (not preflight or args.check_imports or args.build_sglang) and not args.sglang_python:
        parser().error("--sglang-python is required for SGLang")
    report = {
        "schema_version": 1,
        "operation": "Qwen3 hd128 prologue",
        "calls_per_graph": CALLS,
        "torch": torch.__version__,
        "seed": args.seed,
        "parameters": {
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "rotary_dim": 128,
            "eps": EPS,
            "zero_centered": False,
            "theta": 1e6,
            "positions_range": [0, 4096],
            "cache_slots": "max(256, T + 1 rounded up to 16)",
        },
        "engine": {
            "entrypoints": ["rms_norm_ (Q)", "rms_norm_ (K)", "apply_rotary_", "store_cache"],
            "included_in_market_bar": False,
        },
        "rows": [],
        "device": "unavailable",
    }
    modules = None
    if args.sglang_python:
        try:
            with quiet_external():
                modules, report["sglang"] = sglang_modules(args.sglang_python)
            print("SGLang: imported real norm, RoPE and KV-store modules", flush=True)
            if report["sglang"]["revision_verified"]:
                print(
                    "SGLang revision: " + report["sglang"]["revision"] + " (verified)", flush=True
                )
            else:
                print(
                    "WARNING: SGLang revision unverified; mismatched files: "
                    + ", ".join(report["sglang"]["revision_mismatches"]),
                    flush=True,
                )
        except Exception as error:
            report["sglang"] = {"status": "NA (import)", "error_type": type(error).__name__}
            if isinstance(error, ModuleNotFoundError):
                report["sglang"]["missing_module"] = error.name
            print(f"SGLang import: {type(error).__name__}", flush=True)
            if isinstance(error, ModuleNotFoundError):
                print(f"Missing dependency: {error.name}", flush=True)
    if args.build_sglang and modules:
        print("SGLang: compiling JIT modules without launching kernels", flush=True)
        try:
            with quiet_external():
                report["sglang"]["build"] = build_sglang(modules)
        except Exception as error:
            report["sglang"]["build"] = {"status": "unverified", "error_type": type(error).__name__}
        print("SGLang build: " + json.dumps(report["sglang"]["build"]), flush=True)
    if preflight:
        if args.check_vllm:
            print("vLLM: checking wheel, cache-op symbol and extension load", flush=True)
            with quiet_external():
                _, report["vllm"] = load_vllm_cache(args.vllm_cache)
            print("vLLM: " + json.dumps(report["vllm"]), flush=True)
        report["mode"] = "preflight; no GPU kernels launched"
        write_report(report, args.json)
        failed = bool((args.check_imports or args.build_sglang) and modules is None)
        failed |= bool(
            args.build_sglang and report.get("sglang", {}).get("build", {}).get("status") != "built"
        )
        failed |= bool(
            args.check_vllm and report.get("vllm", {}).get("vllm_kv_write") == "stand-in"
        )
        return int(failed)
    tokens = [int(t) for t in args.rows.split(",")]
    if args.pairs < 6 or not tokens or any(t < 1 for t in tokens):
        parser().error("use --pairs >=6 and positive --rows")
    if not torch.cuda.is_available():
        print("No visible CUDA device; timing and GPU correctness checks were not run.")
        report["status"] = "NA (no CUDA device)"
        write_report(report, args.json)
        return 1
    args.device = torch.device("cuda", args.device)
    torch.cuda.set_device(args.device)
    report["device"] = torch.cuda.get_device_name(args.device)
    report["protocol"] = {
        **protocol_metadata(args),
        "initial_pairs": args.pairs,
        "retry_pairs": max(40, 2 * args.pairs),
        "self_pairs": 20,
        "reset": "before replay, outside timing",
        "inputs": "rotating pool of CALLS identical pre-filled QKV buffers, one per call, restored outside timing",
        "pool_memory": "pools over 4 GiB require free CUDA memory >=2x pool bytes; otherwise skip row",
        "ci": "exact binomial order statistics, coverage >=95%",
        "bar_sglang": "primary: CuTe vs real SGLang dispatch",
        "bar": "lower median of SGLang and selected vLLM surrogate variant",
        "sanity": "FP64 reference; <=4 contribution-magnitude BF16 ULP; V exact",
        "slots": "unique random nonzero slots; slot zero reserved by SGLang",
    }
    ceiling = next((v for k, v in CEILINGS.items() if k in report["device"]), None)
    report["dram_ceiling_gbps"] = ceiling
    with quiet_external():
        from ctikernels.attention.inference import Prologue

        op = Prologue(
            num_q_heads=32,
            num_kv_heads=8,
            eps=EPS,
            device=args.device,
            head_dim=128,
            rotary_dim=128,
            zero_centered=False,
        )
    assert op.provider == "cute", "Benchmark requires Prologue provider=cute"
    print("vLLM SURROGATE: preparing extension and dynamic/static Inductor functions", flush=True)
    with quiet_external():
        cache_op, report["vllm"] = load_vllm_cache(args.vllm_cache)
        compiled = compile_vllm()
    report["vllm"].update(
        native="SURROGATE: Inductor default mode, fullgraph QK norm + NeoX RoPE",
        table_dtype="BF16 (native warmed cache)",
        kv_graph_boundary="opaque custom op",
    )
    # Match vLLM's marked dynamic token dimension, with a non-unit priming size.
    dynamic_error = None
    try:
        with quiet_external(), torch.inference_mode():
            prime = Buffers(16, args.device, args.seed)
            torch._dynamo.mark_dynamic(prime.qkv[0], 0)
            torch._dynamo.mark_dynamic(prime.positions, 0)
            compiled["vllm_dynamic"](
                prime.qkv[0], prime.qw, prime.kw, prime.vllm_table, prime.positions
            )
            torch.cuda.synchronize()
        del prime
    except Exception as error:
        dynamic_error = type(error).__name__
    with torch.inference_mode():
        for t in tokens:
            row = {
                "tokens": t,
                "workload": f"decode B={t}"
                if t <= 4
                else f"DFlash B={t // 16}"
                if t <= 64
                else "prefill",
                "candidates": {},
                "comparisons": {},
                "bar_sglang": {"verdict": "NA (SGLang comparison unavailable)"},
                "bar": {"verdict": "NA (both market comparators required)", "vllm_surrogate": True},
            }
            records = row["candidates"]
            cute_factory = lambda b: lambda slot: op(*b.args[slot])
            cute, records["cute"] = prepare("cute", cute_factory, t, args, {"provider": "cute"})
            if cute is None:
                row["status"] = records["cute"]["status"]
                report["rows"].append(row)
                write_report(report, args.json)
                continue
            other, self_record = prepare("cute self", cute_factory, t, args)
            if other is None:
                if self_record["status"] == "NA (pool memory)":
                    row["status"] = self_record["status"]
                    report["rows"].append(row)
                    write_report(report, args.json)
                    del cute
                    continue
                raise RuntimeError("Second CuTe graph unavailable for noise calibration")
            with quiet_external():
                calibration = paired(cute, other, 20)
            epsilon = percentile95([abs(a / b - 1) for a, b in calibration])
            row.update(epsilon=epsilon, self_pairs_us=calibration)
            del other
            graphs = {}
            factories = {"engine": engine_operation}
            if modules:
                factories["sglang"] = lambda b: sglang_operation(b, modules)
            else:
                records["sglang"] = {"status": "NA (import)"}
            for name, function in compiled.items():
                if name == "vllm_dynamic" and dynamic_error:
                    records[name] = {"status": "NA (compile)", "error_type": dynamic_error}
                else:
                    factories[name] = lambda b, f=function: vllm_operation(b, f, cache_op)
            for name, factory in factories.items():
                label = name + " SURROGATE" if name.startswith("vllm") else name
                metadata = {"vllm_surrogate": True} if name.startswith("vllm") else None
                graph, record = prepare(label, factory, t, args, metadata)
                records[name] = record
                if record["status"] == "NA (pool memory)":
                    row["status"] = record["status"]
                    break
                if graph:
                    graphs[name] = graph
            if row.get("status") == "NA (pool memory)":
                report["rows"].append(row)
                write_report(report, args.json)
                del graphs, cute, graph
                continue
            # Static and dynamic variants are both validated and paired; report
            # whichever complete surrogate+KV-write graph has lower median latency.
            for name, graph in graphs.items():
                label = name + " SURROGATE" if name.startswith("vllm") else name
                print(f"T={t}: pairing cute vs {label}", flush=True)
                with quiet_external():
                    comparison = compare(cute, graph, args.pairs, epsilon)
                row["comparisons"][name] = comparison
                records[name]["us"] = comparison["comparator_us"]
            if "sglang" in row["comparisons"]:
                result = row["comparisons"]["sglang"]
                if not report["sglang"]["revision_verified"]:
                    result["verdict"] = "unverified-revision"
                row["bar_sglang"] = {
                    key: result[key] for key in ("speedup", "ci95", "verdict", "pairs")
                }
                row["bar_sglang"]["comparator"] = "sglang"
            variants = [n for n in ("vllm_static", "vllm_dynamic") if n in row["comparisons"]]
            if variants:
                selected = min(variants, key=lambda n: records[n]["us"])
                records["vllm"] = {
                    **records[selected],
                    "selected_variant": selected,
                    "vllm_kv_write": report["vllm"]["vllm_kv_write"],
                }
                row["comparisons"]["vllm"] = row["comparisons"][selected]
            else:
                mismatched = any(
                    records[n]["status"] == "NA (mismatch)" for n in ("vllm_static", "vllm_dynamic")
                )
                records["vllm"] = {
                    "status": "NA (mismatch)" if mismatched else "NA (both variants unavailable)"
                }
            market = [n for n in ("sglang", "vllm") if n in row["comparisons"]]
            records["cute"]["us"] = statistics.median(a for a, _ in calibration)
            if len(market) == 2:
                winner = min(market, key=lambda n: records[n]["us"])
                result = row["comparisons"][winner]
                row["bar"] = {key: result[key] for key in ("speedup", "ci95", "verdict", "pairs")}
                row["bar"].update(
                    comparator=winner,
                    provisional=report["vllm"]["vllm_kv_write"] == "stand-in",
                    vllm_surrogate=True,
                )
                if not report["sglang"]["revision_verified"]:
                    row["bar"]["verdict"] = "unverified-revision"
                records["cute"]["us"] = result["cute_us"]
            # Logical minimum traffic: QKV read + Q/K/V output, FP32 table
            # rows, int64 positions, int32 slots, and the two BF16 weight vectors.
            row["bytes_moved"] = t * (2 * 48 * 128 * 2 + 128 * 4 + 8 + 4) + 2 * 128 * 2
            row["sol_us"] = row["bytes_moved"] / (ceiling * 1000) if ceiling else None
            if ceiling:
                row["sol_percent"] = 100 * row["sol_us"] / records["cute"]["us"]
            if args.sweep:
                row["sweep"] = run_sweep(op, cute, t, args, epsilon)
            report["rows"].append(row)
            write_report(report, args.json)
            del graphs, cute, graph
    render(report)
    write_report(report, args.json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Exception text can contain absolute paths, hostnames, or environment
        # diagnostics. Safe type + phase records above are the public errors.
        print(f"Benchmark failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
