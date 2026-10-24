# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""C4/C5 benchmark support. No CUDA initialization or vendor import at import time."""

import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard, profiler_clean
from benchmark.kernel_tuning.norm_act_probe import provider_spec, run_probe
from benchmark.kernel_tuning.norm_act_reference import (
    fp32_residual_formulas,
    no_worse,
    norm_market_formulas,
    norm_pair_metrics,
    norm_reference,
    sglang_fp32_residual_native,
)
from ctikernels.norm._dispatch import Config, device_class, entry_for, parse_config, tune_configs
from ctikernels.norm.inference import AddRMSNorm


class NormCall:
    def __init__(self, zero_centered, provider="auto", config=None, *, static_weight=True):
        self.op = AddRMSNorm(provider=provider, config=config, static_weight=static_weight)
        self.zero_centered = zero_centered

    @property
    def provider(self):
        return self.op.provider

    @property
    def dispatch(self):
        config = self.op.dispatched_config
        return config.name if config is not None else self.provider

    def __call__(self, tensors, unused=None):
        return self.op(*tensors, 1e-6, zero_centered=self.zero_centered)


class SglangFP32Call:
    """Native eager/compiled comparators, allocating both outputs as SGLang does."""

    def __init__(self, zero_centered, *, compiled=False):
        self.zero_centered = zero_centered
        self.compiled = compiled
        self.native = (
            torch.compile(sglang_fp32_residual_native, fullgraph=True, dynamic=False)
            if compiled
            else sglang_fp32_residual_native
        )

    def __call__(self, tensors, unused=None):
        return self.native(*tensors[:3], zero_centered=self.zero_centered)


def rw_copy(tensors, unused=None):
    source, destination = tensors
    return destination.copy_(source)


def config_name(config):
    return config.name


def flashinfer_metadata():
    import flashinfer.norm as norm

    base = Path(norm.__file__).parent
    digest = hashlib.sha256()
    # Include scheduling and reduction helpers, not just the kernel class.
    for path in sorted(base.rglob("*.py")):
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "version": importlib.metadata.version("flashinfer-python"),
        "source_sha256": digest.hexdigest(),
        "cuda_norm_opt_in": getattr(norm, "_USE_CUDA_NORM", None),
        "pdl": "FlashInfer default (enabled on sm>=90)",
    }


def cross_process_config(args, m, dtype, props):
    """Pin the row's auto-table CuTe layout, or the default for other routes."""
    entry = entry_for(
        device_class((props.major, props.minor), props.multi_processor_count),
        args.width,
        str(dtype).removeprefix("torch."),
        m,
        args.op.startswith("gemma"),
        bool(args.static_weight),
        residual_dtype="float32" if getattr(args, "residual_dtype", "same") == "fp32" else None,
    )
    return entry.config if entry is not None and entry.provider == "cute" else Config()


def candidate_configs(args):
    configs = []
    if args.sweep:
        configs.extend(
            c
            for c in tune_configs(args.width)
            if (c.static_weight or not args.static_weight)
            and re.search(getattr(args, "sweep_filter", None) or "", c.name)
        )
    configs.extend(
        parse_config(s) for s in re.split(r"[;/]", getattr(args, "configs", None) or "") if s
    )
    return list({c.name: c for c in configs}.values())


def norm_candidates(args, dtype, tensors, unused=None, *, control_config=None):
    """Local providers only; market dispatch is owned by the row runner."""
    import inspect

    zero = args.op.startswith("gemma")
    static_weight = bool(args.static_weight)
    factories = {
        "ours": lambda: NormCall(zero, static_weight=static_weight),
        "aa": lambda: NormCall(zero, static_weight=static_weight),
        "cute": lambda: NormCall(zero, "cute", control_config, static_weight=static_weight),
        "flashinfer": lambda: NormCall(zero, "flashinfer"),
        "fi_default": lambda: NormCall(zero, "flashinfer"),
        "static": lambda: NormCall(zero, static_weight=True),
        "triton": lambda: NormCall(zero, "triton"),
    }
    requested = getattr(args, "candidates", None)
    selected = set(requested.split(",")) if requested else set(factories)
    unknown = selected - factories.keys()
    if unknown:
        raise ValueError(f"unknown local candidates: {sorted(unknown)}")
    # ours/A-A define the paired denominator and noise floor in every run.
    selected |= {"ours", "aa"}
    if control_config is not None:
        selected.add("cute")
    skipped = {}
    source = inspect.getsource(AddRMSNorm)
    if not zero and "triton provider requires zero_centered=True" in source:
        selected.discard("triton")
        skipped["triton"] = "installed library only supports zero-centered Triton"
    fp32_residual = getattr(args, "residual_dtype", "same") == "fp32"
    if fp32_residual:
        for name in ("flashinfer", "fi_default"):
            selected.discard(name)
            skipped[name] = (
                "FlashInfer requires a same-dtype residual; BF16 market columns are separate"
            )
    candidates = {name: factory() for name, factory in factories.items() if name in selected}
    if fp32_residual:
        candidates["sgl_fp32"] = SglangFP32Call(zero)
        candidates["sgl_fp32c"] = SglangFP32Call(zero, compiled=True)
    candidates.update(
        {
            c.name: NormCall(zero, "cute", c, static_weight=c.static_weight)
            for c in candidate_configs(args)
        }
    )
    from benchmark.kernel_tuning.norm_act_pools import warm_norm_call

    for fn in candidates.values():
        warm_norm_call(fn, tensors[:3])
    metadata = {"skipped": skipped}
    if fp32_residual:
        for name, fn in candidates.items():
            metadata[name] = {
                "residual_dtype": "float32",
                "kernel": (
                    "SGLang 72730d6 RMSNorm.forward_native (torch eager FP32 add, clone, pow/mean, rsqrt, multiply, cast)"
                    if name == "sgl_fp32"
                    else "SGLang 72730d6 RMSNorm.forward_native (torch.compile, fullgraph=True, dynamic=False, default Inductor options)"
                    if name == "sgl_fp32c"
                    else "ctikernels.norm.cute.Kernel.kernel"
                    if fn.provider == "cute"
                    else "ctikernels.norm._zero_centered_rms_norm._residual_norm_kernel"
                ),
            }
    if "flashinfer" in candidates or "fi_default" in candidates:
        metadata["flashinfer"] = flashinfer_metadata()
    return candidates, metadata


def profile_providers(args, m, candidates, metadata):
    for name, fn in candidates.items():
        metadata.setdefault(name, {}).update(
            run_probe(
                provider_spec(
                    fn,
                    op=args.op,
                    m=m,
                    width=args.width,
                    dtype=args.dtype,
                    residual_dtype=getattr(args, "residual_dtype", "same"),
                )
            )
        )


def check_norm_candidates(
    candidates, inputs, zero_centered, extra_markets=None, *, record_inputs=False, shared_pools=None
):
    """The C3 per-metric best-market gate, for both outputs and each identical input.

    Real FlashInfer/Inductor kernels and torch formulas are markets. FI 0.7's
    child supplies per-input metrics, never an aggregate from unrelated rows.
    Bitwise equality with FlashInfer is informational only.
    """
    worst = {name: {"normalized": {}, "residual": {}} for name in candidates}
    passed = dict.fromkeys(candidates, True)
    bitwise = dict.fromkeys(candidates, True) if "flashinfer" in candidates else {}
    residual_bits = bitwise.copy()
    per_input = []
    formula_names = []
    if extra_markets is not None and len(extra_markets) != len(inputs):
        raise ValueError("FlashInfer 0.7 check input count differs from parent")
    for index, tensors in enumerate(inputs):
        if shared_pools is not None:
            shared_pools.restore()
        exact, _ = norm_reference(*tensors[:3], zero_centered=zero_centered)
        residual_exact = (tensors[0].float() + tensors[1].float()).double()
        mixed = tensors[1].dtype == torch.float32 and tensors[0].dtype != torch.float32
        formulas = (fp32_residual_formulas if mixed else norm_market_formulas)(
            *tensors[:3], zero_centered=zero_centered
        )
        formula_names = list(formulas)
        outputs = {}
        for name, fn in candidates.items():
            if shared_pools is None:
                outputs[name] = fn(tuple(t.clone() for t in tensors))
            else:
                shared_pools.restore()
                # Candidates operate on IPC views; snapshot results before the next
                # candidate restores and overwrites the same physical buffers.
                outputs[name] = tuple(t.clone() for t in fn(tensors))
        if shared_pools is not None:
            shared_pools.restore()
        metrics = {
            name: norm_pair_metrics(pair, exact, residual_exact) for name, pair in outputs.items()
        }
        markets = [
            metrics[name]
            for name in ("vllm", "sglang", "flashinfer", "flashinfer07")
            if name in metrics
        ]
        markets += [norm_pair_metrics(pair, exact, residual_exact) for pair in formulas.values()]
        if extra_markets is not None:
            markets.append(extra_markets[index])
        if record_inputs:
            per_input.append(metrics)
        for name, buffers in metrics.items():
            for buffer, result in buffers.items():
                worst[name][buffer] = {
                    key: max(value, worst[name][buffer].get(key, 0.0))
                    for key, value in result.items()
                }
                passed[name] &= all(math.isfinite(value) for value in result.values())
                passed[name] &= no_worse(result, [market[buffer] for market in markets])
            if "flashinfer" in outputs:
                for target, output_index in ((bitwise, 0), (residual_bits, 1)):
                    target[name] &= torch.equal(
                        outputs[name][output_index].contiguous().view(torch.uint8),
                        outputs["flashinfer"][output_index].contiguous().view(torch.uint8),
                    )
    result = {
        "passed": all(passed.values()),
        "candidate_passed": passed,
        "metrics": {name: pair["normalized"] for name, pair in worst.items()},
        "residual_metrics": {name: pair["residual"] for name, pair in worst.items()},
        "bitwise_flashinfer": bitwise,
        "residual_bitwise_flashinfer": residual_bits,
        "markets": [
            name for name in ("vllm", "sglang", "flashinfer", "flashinfer07") if name in candidates
        ]
        + formula_names
        + (["flashinfer07"] if extra_markets is not None else []),
    }
    if record_inputs:
        result["per_input"] = per_input
    return result


def check_sum_norm_candidates(
    candidates, inputs, zero_centered, extra_markets=None, *, record_inputs=False, shared_pools=None
):
    """Check both D2 parities and guard columns around each strided aux slice."""
    sum_checks = dict.fromkeys(candidates, True)

    def checked(name, fn):
        def call(triple):
            h, r, w = triple[:3]
            m, width = h.shape
            if shared_pools is None:
                buffer = torch.full((m, 3 * width), 17.0, device=h.device, dtype=r.dtype)
                aux = buffer[:, width : 2 * width]
            else:
                aux = triple[3]
                buffer = aux.as_strided((m, 3 * width), aux.stride(), aux.storage_offset() - width)
                buffer.fill_(17)
            expected = h + r
            even = [t.clone() for t in triple] if shared_pools is None else (h, r, w)
            fn.call_index((*even, aux), None, 0)
            valid = bool((buffer == 17).all())
            if shared_pools is not None:
                shared_pools.restore()
            pair = fn.call_index((h, r, w, aux), None, 1)
            valid &= (
                torch.equal(aux, expected)
                and bool((buffer[:, :width] == 17).all())
                and bool((buffer[:, 2 * width :] == 17).all())
            )
            sum_checks[name] &= valid
            return pair

        return call

    result = check_norm_candidates(
        {name: checked(name, fn) for name, fn in candidates.items()},
        inputs,
        zero_centered,
        extra_markets,
        record_inputs=record_inputs,
        shared_pools=shared_pools,
    )
    result["sum_out"] = sum_checks
    for name, valid in sum_checks.items():
        result["candidate_passed"][name] &= valid
    result["passed"] &= all(sum_checks.values())
    return result


def fi07_child(args, m, regime):
    """A fresh interpreter gives the requested install import/metadata precedence.

    FlashInfer 0.7, dispatched ours and A/A are paired within the child.
    """
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "runpy.run_module('benchmark.kernel_tuning.norm_act_paired',run_name='__main__')",
        str(args.fi07.resolve()),
        "--fi07-child",
        "--op",
        args.op,
        "--rows",
        str(m),
        "--regimes",
        regime,
        "--width",
        str(args.width),
        "--dtype",
        args.dtype,
        "--layers",
        str(args.layers),
        "--rounds",
        str(args.rounds),
        "--static-weight",
        str(getattr(args, "static_weight", 1)),
    ]
    if args.check:
        command.append("--check")
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        rows = [
            json.loads(line[4:]) for line in result.stdout.splitlines() if line.startswith("ROW ")
        ]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("invalid FlashInfer 0.7 receipt object")
        if len(rows) == 1 and result.returncode == 0:
            row = rows[0]
            if not math.isfinite(row["us"]["flashinfer07"]) or row["us"]["flashinfer07"] <= 0:
                raise ValueError("invalid FlashInfer 0.7 timing")
            if not math.isfinite(row["us"]["ours"]) or row["us"]["ours"] <= 0:
                raise ValueError("invalid FlashInfer 0.7 ours timing")
            ratio = row["ratios"]["flashinfer07"]
            lo, hi = ratio["ci95"]
            if not math.isfinite(ratio["median"]) or not 0 <= lo <= ratio["median"] <= hi:
                raise ValueError("invalid FlashInfer 0.7 paired ratio")
            if not math.isfinite(row["epsilon"]) or row["epsilon"] < 0.005:
                raise ValueError("invalid FlashInfer 0.7 A/A epsilon")
            lo, hi = row["ci95_us"]
            if not 0 <= lo <= row["us"]["flashinfer07"] <= hi:
                raise ValueError("invalid FlashInfer 0.7 timing interval")
            if row["providers"]["flashinfer07"]["version"] != "0.7.0.post1":
                raise ValueError("unexpected FlashInfer child version")
            return row
        # Native import/build stderr may contain environment values. Never echo it.
        detail = "child failed; native stderr omitted"
        if len(rows) == 1 and rows[0].get("verdict") == "NUMERICS_FAIL":
            detail = "child numerical gate failed: " + json.dumps(
                {
                    key: (rows[0].get("check") or {}).get(key)
                    for key in ("candidate_passed", "metrics", "residual_metrics")
                }
            )
        raise RuntimeError(f"exit={result.returncode}, receipts={len(rows)}: {detail}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        reason = f"{type(error).__name__}: {error}"
        print(
            f"FlashInfer 0.7 child failed (M={m}, {regime}): {reason}", file=sys.stderr, flush=True
        )
        return {"status": "ERROR", "error": reason}


def run_fi07_row(args, m, regime):
    install_profiler_guard()
    from benchmark.kernel_tuning.norm_act_paired import (
        capture_bank,
        measure,
        median_ci,
        paired_summary,
    )

    zero = args.op == "gemma_add_norm"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if not l2:
        raise RuntimeError("torch device properties must expose L2 cache size")
    # Run the independent install before allocating this row's pools. Large C5
    # pools plus retained native outputs must not coexist with a second CUDA
    # context's pools. Release cached allocations from earlier rows as well.
    child_receipt = None
    if args.fi07:
        torch.cuda.empty_cache()
        child_receipt = fi07_child(args, m, regime)
    from benchmark.kernel_tuning.norm_act_pools import build_norm_pools, pool_spec

    pools = build_norm_pools(pool_spec(args, m, regime, l2))
    x_pool, r_pool, w_pool = pools.x, pools.residual, pools.weight
    saved_x, saved_r = pools.saved_x, pools.saved_r
    tensors, inputs, count = pools.tensors, pools.inputs, pools.count
    unique = len(tensors)
    if args.fi07_child:
        metadata = {"flashinfer07": flashinfer_metadata()}
        if metadata["flashinfer07"]["version"] != "0.7.0.post1":
            raise RuntimeError("--fi07 must resolve flashinfer-python==0.7.0.post1")
        candidates = {
            "ours": NormCall(zero, static_weight=bool(getattr(args, "static_weight", 1))),
            "aa": NormCall(zero, static_weight=bool(getattr(args, "static_weight", 1))),
            "flashinfer07": NormCall(zero, "flashinfer"),
        }
        for fn in candidates.values():
            fn(tuple(t.clone() for t in tensors[0]))
    else:
        candidates, metadata = norm_candidates(args, dtype, tensors[0])
    if args.fi07_child and getattr(args, "profile", False):
        profile_providers(args, m, candidates, metadata)
    extra_markets = None
    if args.check and child_receipt is not None and child_receipt.get("status") != "ERROR":
        try:
            extra_markets = [row["flashinfer07"] for row in child_receipt["check"]["per_input"]]
            if len(extra_markets) != len(tensors):
                raise ValueError("check input count differs from parent")
        except (KeyError, TypeError, ValueError) as error:
            reason = f"FlashInfer 0.7 check receipt invalid: {error}"
            print(reason, file=sys.stderr, flush=True)
            child_receipt = {"status": "ERROR", "error": reason}
    check = (
        check_norm_candidates(
            {"flashinfer07": candidates["flashinfer07"]} if args.fi07_child else candidates,
            tensors,
            zero,
            extra_markets,
            record_inputs=args.fi07_child,
        )
        if args.check
        else None
    )

    def restore(unused=None):
        r_pool.copy_(saved_r)
        # Restore shared x last, modelling the preceding GEMM's L2-hot output.
        x_pool.copy_(saved_x)

    banks = {}
    for name, fn in candidates.items():
        restore()
        banks[name] = capture_bank(fn, inputs, [None] * count, args.layers)
    samples = measure(banks, args.rounds, args.layers, before=restore)
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
        "pool_input_bytes": sum(t.numel() * t.element_size() for t in (x_pool, r_pool, w_pool)),
        "distinct_weights": unique,
        "graphs_per_candidate": count // args.layers,
        "dispatch": getattr(candidates["ours"], "dispatch", candidates["ours"].provider)
        if "ours" in candidates
        else "flashinfer07",
        "static_weight": bool(getattr(args, "static_weight", 1)),
        "providers": metadata,
        "check": check,
        "restoration": "before each candidate/round, outside timed graphs",
    }
    if args.fi07_child:
        receipt.update(paired_summary(samples, markets=("flashinfer07",)))
        receipt.update(
            ci95_us=median_ci(samples["flashinfer07"])[1],
            verdict="PASS" if check is None or check["passed"] else "NUMERICS_FAIL",
        )
    else:
        receipt.update(paired_summary(samples))
        if check and not check["candidate_passed"]["ours"]:
            receipt["verdict"] = "NUMERICS_FAIL"
        if args.sweep or getattr(args, "configs", None):
            receipt["configs"] = {config_name(c): asdict(c) for c in candidate_configs(args)}
            receipt["candidate_verdicts"] = {
                name: (
                    "NUMERICS_FAIL"
                    if check and not check["candidate_passed"][name]
                    else "PASS"
                    if median_ci(
                        [
                            min(v, s) / candidate
                            for v, s, candidate in zip(
                                samples["vllm"], samples["sglang"], times, strict=True
                            )
                        ]
                    )[1][0]
                    >= 1 - receipt["epsilon"]
                    else "FAIL"
                )
                for name, times in samples.items()
                if name.startswith("cute") or name == "triton"
            }
        if child_receipt is not None:
            apply_fi07_market(receipt, child_receipt, samples, zero)

    return receipt


def apply_fi07_market(receipt, child, samples, zero_centered):
    """Use within-process paired ratios to select the fastest market for the row."""
    from benchmark.kernel_tuning.norm_act_paired import median_ci

    receipt["flashinfer07"] = child
    receipt["flashinfer07_role"] = "market"
    if child.get("status") == "ERROR":
        receipt["flashinfer07_error"] = child["error"]
        if receipt["verdict"] != "NUMERICS_FAIL":
            receipt["verdict"] = "MARKET_ERROR"
        if zero_centered:
            for name, verdict in receipt.get("candidate_verdicts", {}).items():
                if verdict != "NUMERICS_FAIL":
                    receipt["candidate_verdicts"][name] = "MARKET_ERROR"
        return
    receipt["us"]["flashinfer07"] = child["us"]["flashinfer07"]
    receipt["us"]["ours_fi07"] = child["us"]["ours"]
    receipt["ratios"]["flashinfer07"] = child["ratios"]["flashinfer07"]
    receipt["epsilon_fi07"] = child["epsilon"]
    fastest = min(
        (receipt["ratios"][name] for name in ("vllm", "sglang", "flashinfer07")),
        key=lambda ratio: ratio["median"],
    )
    receipt["faster_ratio"] = fastest["median"]
    receipt["faster_ci95"] = fastest["ci95"]
    # Preserve the existing independent C5 sweep gates for candidates that were
    # not timed in the child. They cannot supply a paired FI 0.7 row verdict.
    comparisons = {}
    for name, times in samples.items():
        if name != "triton" and not name.startswith("cute"):
            continue
        median, ci = median_ci(times)
        lower = child["ci95_us"][0] / ci[1]
        upper = child["ci95_us"][1] / ci[0] if ci[0] > 0 else float("inf")
        comparisons[name] = {
            "ratio": child["us"]["flashinfer07"] / median,
            # Bounds from two independent marginal 95% intervals (>=90% joint).
            "ratio_bounds": [lower, upper],
            "verdict": "PASS" if lower >= 1 - receipt["epsilon"] else "FAIL",
        }
        if zero_centered and comparisons[name]["verdict"] == "FAIL":
            verdicts = receipt.get("candidate_verdicts", {})
            if verdicts.get(name) == "PASS":
                verdicts[name] = "FAIL"
    receipt["flashinfer07_comparison"] = {
        "measurement": "independent process; marginal 95% CI ratio bounds",
        "candidates": comparisons,
    }
    receipt["epsilon"] = max(receipt["epsilon"], child["epsilon"])
    if receipt["verdict"] != "NUMERICS_FAIL":
        receipt["verdict"] = (
            "PASS" if receipt["faster_ci95"][0] >= 1 - receipt["epsilon"] else "FAIL"
        )


class VllmNormCall:
    """Select A/B by graph call index, so every even-L graph has equal site counts."""

    def __init__(self, sites, pools, site=None, buffers=None):
        self.sites, self.pools, self.site = sites, pools, site
        self.buffers = buffers

    def call_index(self, tensors, unused, index):
        x, residual, weight = tensors[:3]
        site = self.site or ("a" if index % 2 == 0 else ("d2" if "d2" in self.sites else "b"))
        kwargs = {}
        if self.buffers is not None and self.sites[site].compiled is None:
            buffer_index = 0 if self.pools.spec["regime"] == "hot" else index
            kwargs = {
                "output": self.buffers[0][buffer_index],
                "residual_output": self.buffers[1][buffer_index],
            }
        if site == "d2":
            kwargs["aux_output"] = tensors[3]
        if site in ("b", "d2"):
            return self.sites[site](
                x, self.pools.previous_hidden(index), residual, weight, **kwargs
            )
        return self.sites[site](x, residual, weight, **kwargs)


def merge_market_metrics(*metrics):
    """Per-metric best comparator on one input, preserving the no_worse gate."""
    return {
        buffer: {key: min(item[buffer][key] for item in metrics) for key in metrics[0][buffer]}
        for buffer in metrics[0]
    }


def run_norm_row(args, m, regime):
    install_profiler_guard()
    if getattr(args, "fi07_child", False):
        return run_fi07_row(args, m, regime)
    from benchmark.kernel_tuning.norm_act_market import RemoteCandidate, start_market
    from benchmark.kernel_tuning.norm_act_paired import (
        SumNormCall,
        capture_bank,
        measure,
        median_ci,
        paired_summary,
    )
    from benchmark.kernel_tuning.norm_act_pools import (
        build_norm_pools,
        build_rw_copy_pools,
        fp32_residual_pools,
        pool_spec,
        warm_norm_call,
    )
    from benchmark.kernel_tuning.norm_act_reference import error_metrics
    from benchmark.kernel_tuning.norm_act_vllm import MarketError, VllmSite, optional_sites

    zero, summed = args.op.startswith("gemma"), args.op.endswith("_sum")
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if not l2:
        raise RuntimeError("torch device properties must expose L2 cache size")
    if args.layers % 2:
        raise MarketError("norm layer mean requires even --layers for equal A/B counts")
    child = fi07_child(args, m, regime) if getattr(args, "fi07", None) else None
    spec = pool_spec(args, m, regime, l2)
    pools = build_norm_pools(spec)
    fp32_residual = getattr(args, "residual_dtype", "same") == "fp32"
    local_pools = fp32_residual_pools(pools) if fp32_residual else pools
    first = local_pools.tensors[0][:3]
    control_config = cross_process_config(args, m, first[0].dtype, props)
    candidates, metadata = norm_candidates(
        args, first[0].dtype, first, control_config=control_config
    )
    if summed:
        base = candidates
        candidates = {
            name: SumNormCall(fn, fused=name in ("ours", "aa") or name.startswith("cute"))
            for name, fn in base.items()
        }
        candidates["unfused"] = SumNormCall(base["ours"])
    if getattr(args, "profile", False):
        profile_providers(args, m, candidates, metadata)
    x, residual, weight = (t.clone() for t in pools.tensors[0][:3])
    sites, omission = optional_sites(
        lambda: {
            site: VllmSite(
                args,
                site,
                (x, pools.previous_hidden(0), residual, weight)
                if site in ("b", "d2")
                else (x, residual, weight),
                zero,
            )
            for site in (("a", "d2") if summed else ("a", "b"))
        }
    )
    metadata["vllm"] = omission or {site: fn.metadata for site, fn in sites.items()}
    # Candidates execute serially, so their native output allocations can alias.
    # This bounds the combined parent/worker memory at large cold M without an
    # extra GPU copy or changing any input or JIT instruction.
    output_buffers = (torch.empty_like(pools.residual), torch.empty_like(pools.residual))
    market_calls = {"vllm": VllmNormCall(sites, pools, buffers=output_buffers)} if sites else {}
    if sites and not summed:
        market_calls.update(
            {
                "vllm_" + site: VllmNormCall(sites, pools, site, output_buffers)
                for site in ("a", "b")
            }
        )
    worker, setup = start_market(args, pools, cute_config=control_config)
    try:
        if setup["count"] != pools.count or setup["distinct_weights"] != len(pools.tensors):
            raise MarketError("worker pool geometry differs from parent")
        metadata["sglang"] = setup["metadata"]
        metadata["fi071_direct"] = {"version": "0.7.1", "role": "cross-process wrapper control"}
        metadata["cute_remote"] = {
            "config": asdict(control_config),
            "static_weight": bool(args.static_weight),
            "role": "same-dtype market-pool CuTe diagnostic"
            if fp32_residual
            else "same-kernel cross-process control",
        }
        remote_names = ("sglang", "fi071_direct", "cute_remote")
        check = None
        if args.check:
            # Check every distinct input, but keep only worst metrics in public ROWs.
            remote = []
            for start in range(0, len(pools.tensors), 64):
                result = worker.request(
                    "check", start=start, stop=min(start + 64, len(pools.tensors))
                )
                if summed and not all(
                    result.get("sum_out", {}).get(name, False) for name in remote_names
                ):
                    raise MarketError("worker D2 aux capture or guard columns failed")
                remote.extend(result["per_input"])
            if len(remote) != len(pools.tensors) or any(
                not math.isfinite(value)
                for row in remote
                for pair in row.values()
                for metrics in pair.values()
                for value in metrics.values()
            ):
                raise MarketError("invalid worker numerical metrics")
            extra = []
            vllm_checks = {}
            for i, triple in enumerate(pools.tensors):
                h, r, w = triple[:3]
                exact, _ = norm_reference(h, r, w, zero_centered=zero)
                residual_exact = (h.float() + r.float()).double()
                site_metrics = {}
                markets = [remote[i]["sglang"], remote[i]["fi071_direct"]]
                if sites:
                    out = sites["a"](h, r, w)
                    comparable = {
                        "normalized": error_metrics(out, exact),
                        "residual": error_metrics(residual_exact.to(r.dtype), residual_exact),
                    }
                    site_metrics["a"] = {"normalized": comparable["normalized"]}
                    h1 = pools.previous_hidden(i)
                    s1_fp32 = h1.float() + r.float()
                    s1 = s1_fp32.to(h.dtype)
                    b_exact, _ = norm_reference(h, s1, w, zero_centered=zero)
                    b_site = "d2" if summed else "b"
                    if summed:
                        aux_buffer = torch.full(
                            (m, 3 * args.width), 17.0, device=h.device, dtype=h.dtype
                        )
                        aux = aux_buffer[:, args.width : 2 * args.width]
                        pair = sites[b_site](h, h1, r, w, aux_output=aux)
                        if not (
                            torch.equal(pair[2], aux)
                            and torch.equal(aux, pair[1])
                            and bool((aux_buffer[:, : args.width] == 17).all())
                            and bool((aux_buffer[:, 2 * args.width :] == 17).all())
                        ):
                            raise MarketError("vLLM D2 aux capture or guard columns failed")
                    else:
                        pair = sites[b_site](h, h1, r, w)
                    site_metrics[b_site] = norm_pair_metrics(
                        pair, b_exact, (h.float() + s1.float()).double()
                    )
                    # Shipped B keeps the rematerialized intermediate in FP32;
                    # report both references, without rewriting its arithmetic.
                    shipped_exact, _ = norm_reference(h, s1_fp32, w, zero_centered=zero)
                    site_metrics[b_site + "_shipped_fp32"] = norm_pair_metrics(
                        pair, shipped_exact, (h.float() + s1_fp32).double()
                    )
                    if summed:
                        site_metrics[b_site]["aux"] = error_metrics(
                            pair[2], (h.float() + s1.float()).double()
                        )
                        site_metrics[b_site + "_shipped_fp32"]["aux"] = error_metrics(
                            pair[2], (h.float() + s1_fp32).double()
                        )
                    for site, buffers in site_metrics.items():
                        target = vllm_checks.setdefault(site, {})
                        for buffer, metrics in buffers.items():
                            target[buffer] = {
                                key: max(value, target.get(buffer, {}).get(key, 0))
                                for key, value in metrics.items()
                            }
                    markets.append(comparable)
                if any(
                    not math.isfinite(value)
                    for metric in markets
                    for buffer in metric.values()
                    for value in buffer.values()
                ):
                    raise MarketError("nonfinite market error metrics")
                if child is not None and child.get("status") != "ERROR":
                    markets.append(child["check"]["per_input"][i]["flashinfer07"])
                extra.append(merge_market_metrics(*markets))
            checker = check_sum_norm_candidates if summed else check_norm_candidates
            check = checker(
                candidates,
                [triple[:3] for triple in local_pools.tensors],
                zero,
                None if fp32_residual else extra,
            )
            check["markets"] = (
                ["torch_fp32", "sgl_fp32"]
                if fp32_residual
                else [
                    *(["vllm_a"] if sites else []),
                    "sglang",
                    "fi071_direct",
                    "vllm_formula",
                    "flashinfer_formula",
                ]
            )
            check["vllm_sites"] = vllm_checks
            if summed:
                check["aux_capture"] = {
                    **({"vllm_d2": True} if sites else {}),
                    "sglang": True,
                    "fi071_direct": True,
                }
            if any(
                not math.isfinite(value)
                for site in vllm_checks.values()
                for buffer in site.values()
                for value in buffer.values()
            ):
                raise MarketError("nonfinite vLLM site error metrics")
            check["remote_metrics"] = {
                name: {
                    buffer: {
                        key: max(row[name][buffer][key] for row in remote)
                        for key in remote[0][name][buffer]
                    }
                    for buffer in remote[0][name]
                }
                for name in remote_names
            }
        # Warm every specialization on private storage; capture is never timed.
        for fn in candidates.values():
            warm_norm_call(fn, local_pools.tensors[0])
        worker.request("warm")
        for name in remote_names:
            metadata[name].update(worker.request("profile", candidate=name))
            if not metadata[name]["kernels"]:
                raise MarketError("worker profiler did not report kernels")

        witness = (
            build_rw_copy_pools(local_pools)
            if fp32_residual and args.op == "add_norm" and not getattr(args, "no_rw_copy", False)
            else None
        )

        def restore(name=None):
            if name == "rw_copy":
                witness.restore()
                return
            pools.restore()
            if fp32_residual:
                local_pools.restore()

        banks = {}
        market_pool = torch.cuda.graph_pool_handle()
        market_stream = torch.cuda.Stream()
        for name, fn in (candidates | market_calls).items():
            restore()
            banks[name] = capture_bank(
                fn,
                pools.inputs if name in market_calls else local_pools.inputs,
                [None] * pools.count,
                args.layers,
                **(
                    {"shared_pool": market_pool, "capture_stream": market_stream}
                    if name in market_calls
                    else {}
                ),
            )
        if witness is not None:
            restore("rw_copy")
            banks["rw_copy"] = capture_bank(
                rw_copy, witness.inputs, [None] * pools.count, args.layers
            )
            metadata["rw_copy"] = {
                "role": "size-matched read/write diagnostic; excluded from market gates",
                "kernel": "torch.Tensor.copy_",
                "dtype": "float32",
                "read_bytes": witness.inputs[0][0].numel() * 4,
                "write_bytes": witness.inputs[0][1].numel() * 4,
                "buffer_pairs": len(local_pools.tensors),
                "graphs": pools.count // args.layers,
            }
        banks.update({name: RemoteCandidate(worker, name) for name in remote_names})
        samples = measure(banks, args.rounds, args.layers, before=restore)
        receipt = {
            "profiler_clean": profiler_clean(),
            "op": args.op,
            "device": props.name,
            "M": m,
            "width": args.width,
            "dtype": args.dtype,
            "regime": regime,
            "L": args.layers,
            "rounds": args.rounds,
            "l2_bytes": l2,
            "pool_input_bytes": local_pools.input_bytes,
            "market_pool_input_bytes": pools.input_bytes,
            "residual_dtype": "fp32" if fp32_residual else "same",
            "h1_bytes": pools.h1.numel() * pools.h1.element_size(),
            "market_output_storage": "graph-owned allocations shared across sequential local market banks; no output consumer between banks",
            "vllm_output_bytes": sum(t.numel() * t.element_size() for t in output_buffers)
            if sites
            else 0,
            "vllm_output_reuse": "shared between serialized shipped candidates",
            "distinct_weights": len(pools.tensors),
            "graphs_per_candidate": pools.count // args.layers,
            "static_weight": bool(args.static_weight),
            "dispatch": getattr(candidates["ours"], "dispatch", candidates["ours"].provider),
            "shared_buffers": setup["shared_buffers"],
            "providers": metadata,
            "check": check,
            "restoration": "shared CUDA IPC pools and restore sources; restore residual, h1, x before each local/remote sample outside timing; synchronous process handoff",
            "sequence": "D2: even plain / odd aux; vllm A/B-aux, ours fused sum_out, unfused add, sglang residual copy"
            if summed
            else "vllm alternates A/B; site-only columns diagnostic",
        }
        receipt.update(paired_summary(samples))
        if omission:
            receipt["omitted"] = {"vllm": omission["reason"]}
        if fp32_residual:
            receipt.pop("cross_process_control", None)
            receipt["market_contract"] = (
                "same-dtype markets remain separate; fp32_market_ratio is min(sgl_fp32, sgl_fp32c)/ours within each paired round"
            )
            receipt["restoration"] = (
                "separate FP32 local and same-dtype market residual pools; all restored outside timing"
            )
        wrapper_ratio, wrapper_ci = median_ci(
            [s / d for s, d in zip(samples["sglang"], samples["fi071_direct"], strict=True)]
        )
        receipt["wrapper_control"] = {"sglang_over_fi071_direct": wrapper_ratio, "ci95": wrapper_ci}
        if check and not check["candidate_passed"]["ours"]:
            receipt["verdict"] = "NUMERICS_FAIL"
        receipt["candidate_verdicts"] = {
            name: "NUMERICS_FAIL"
            if check and not check["candidate_passed"][name]
            else "PASS"
            if median_ci(
                [
                    min(values[:-1]) / values[-1]
                    for values in zip(
                        *(samples[market] for market in receipt["markets"]),
                        samples[name],
                        strict=True,
                    )
                ]
            )[1][0]
            >= 1 - receipt["epsilon"]
            else "FAIL"
            for name in candidates
            if name != "aa"
        }
        if args.sweep or getattr(args, "configs", None):
            receipt["configs"] = {
                config_name(c): asdict(c)
                for c in candidate_configs(args)
                if config_name(c) in candidates
            }
        if summed:
            receipt["unfused_ratio"] = receipt["ratios"]["unfused"]["median"]
            receipt["unfused_ci95"] = receipt["ratios"]["unfused"]["ci95"]
        if child is not None:
            # FI 0.7 qualifies OUR provider; it never replaces either real market.
            receipt["flashinfer07"] = child
            receipt["flashinfer07_role"] = "provider qualification"
            if child.get("status") == "ERROR":
                receipt["verdict"] = "MARKET_ERROR"
            else:
                if child.get("check"):
                    child["check"].pop("per_input", None)
                receipt["us"].update(
                    flashinfer07=child["us"]["flashinfer07"], ours_fi07=child["us"]["ours"]
                )
                receipt["ratios"]["flashinfer07"] = child["ratios"]["flashinfer07"]
                receipt["epsilon_fi07"] = child["epsilon"]
        return receipt
    finally:
        try:
            worker.request("free")
        finally:
            worker.close()
