# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen, paired, cold-weight parity against SGLang or vLLM.

Example: --rows linear/9b/all --pairs 20. Add --candidates for owned-plan sweeps.
Each slash-separated field accepts comma-separated values, or 'all'. Progress
is stderr; JSON records are stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import torch

from ..sol import sol
from .nvfp4 import _dispatch_metadata, _forced, _weight

# (H, I); "27b-lm" is the Qwen3.8-27B TP1 lm_head (N=248320, K=5120), linear rows only.
SHAPES = {
    "9b": (4096, 12288),
    "27b": (5120, 17408),
    "35b-shared": (2048, 512),
    "27b-lm": (248320, 5120),
}
OPS = ("linear", "gated")
BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
WORKLOAD = tuple(product(OPS, ("9b", "27b", "35b-shared"), BATCHES)) + tuple(
    ("linear", "27b-lm", m) for m in BATCHES
)


def _percentile(ordered, fraction):
    position = fraction * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_ci(ratios):
    """95% percentile interval of paired median speedup; 2000 resamples, seed 0."""
    if not ratios or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("ratios must be nonempty, positive and finite")
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def x925_cpus(cpuinfo):
    """Logical CPU IDs for Cortex-X925 (0xd85), excluding Cortex-A725 (0xd87)."""
    cpus = set()
    fields = {}
    for line in [*cpuinfo.splitlines(), ""]:
        if not line.strip():
            if fields.get("CPU part", "").lower() == "0xd85" and "processor" in fields:
                cpus.add(int(fields["processor"]))
            fields = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return sorted(cpus)


def _pin_x925():
    cpus = x925_cpus(Path("/proc/cpuinfo").read_text())
    if not cpus:
        raise RuntimeError("X925 pinning required, but /proc/cpuinfo has no Cortex-X925 CPUs")
    os.sched_setaffinity(0, cpus)
    return cpus


def _device_name():
    # PyTorch's name query initializes CUDA. Keep discovery out of the timing
    # process so its affinity can be set before CUDA creates any worker threads.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, torch; print(json.dumps(torch.cuda.get_device_name(0)))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _timing_setup(args):
    device_name = _device_name()
    gb10 = "GB10" in device_name.upper()
    args.pairs = max(60, args.pairs) if gb10 else args.pairs
    args.pin_x925 = args.pin_x925 or gb10
    cpus = _pin_x925() if args.pin_x925 else []
    return device_name, cpus


def parity_statistics(
    ours, reference, noise_a, noise_b, *, comparator="sglang", device_name="", identical_kernel=None
):
    """Paired speedup, bootstrap CI, relative noise and device-specific verdict.

    ``identical_kernel`` says whether our GEMM is the comparator's GEMM (same kernel names);
    only then does the identical-kernel tie rule apply. ``acceptance`` combines the rules.
    """
    if not ours or len(ours) != len(reference) or not noise_a or len(noise_a) != len(noise_b):
        raise ValueError("timings must be nonempty matched pairs")
    if any(
        not math.isfinite(t) or t <= 0 for ts in (ours, reference, noise_a, noise_b) for t in ts
    ):
        raise ValueError("timings must be positive and finite")
    ratios = [b / a for a, b in zip(ours, reference, strict=True)]
    speedup = statistics.median(ratios)
    ci_low, ci_high = bootstrap_ci(ratios)
    noise = sorted(abs(a / b - 1) for a, b in zip(noise_a, noise_b, strict=True))
    eps = _percentile(noise, 0.95)
    if "GB10" in device_name.upper():
        verdict = "pass" if ci_low > 1.0 else "fail"
    else:
        verdict = "inconclusive" if eps > 0.02 else "pass" if speedup >= 1 - eps else "fail"
    # Identical-kernel tie rule, only for rows whose GEMM is the same
    # vendor kernel as the comparator's: >= 60 pairs, median >= 1.00, CI_low >= 1 - eps.
    if identical_kernel is None:
        tie_verdict = "kernel-identity-unknown"
    elif not identical_kernel:
        tie_verdict = "not-identical-kernel"
    elif len(ratios) < 60:
        tie_verdict = "insufficient-pairs"
    else:
        tie_verdict = "pass" if speedup >= 1.0 and ci_low >= 1 - eps else "fail"
    # A different kernel that wins outright: >= 60 pairs and CI_low > 1.
    win = len(ratios) >= 60 and ci_low > 1.0
    if win:
        acceptance = "pass"
    elif identical_kernel:
        acceptance = "pass" if tie_verdict == "pass" else "fail"
    else:
        acceptance = verdict
    return {
        "speedup": speedup,
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "eps": eps,
        "verdict": verdict,
        "tie_verdict": tie_verdict,
        "identical_kernel": identical_kernel,
        "win_ci_low_above_1": win,
        "acceptance": acceptance,
        "ours_us": statistics.median(ours),
        f"{comparator}_us": statistics.median(reference),
        "noise_a_us": statistics.median(noise_a),
        "noise_b_us": statistics.median(noise_b),
        "pairs": len(ours),
    }


def select_rows(spec):
    fields = spec.split("/")
    if len(fields) != 3:
        raise ValueError("--rows expects op-list/shape-list/M-list (or all per field)")
    selected = []
    for field, allowed in zip(fields, (OPS, tuple(SHAPES), BATCHES), strict=True):
        values = set(allowed) if field == "all" else set(field.split(","))
        if allowed is BATCHES:
            values = {int(v) for v in values}
        if not values <= set(allowed):
            raise ValueError(f"unknown --rows values: {values - set(allowed)}")
        selected.append(values)
    return [
        row
        for row in WORKLOAD
        if all(v in choices for v, choices in zip(row, selected, strict=True))
    ]


def _capture(calls):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def kernel_names(op, x):
    """Kernel names of a warmed, single-call captured graph replay, in launch order."""
    graph = _capture([lambda: op(x)])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        graph.replay()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.lower().startswith(("memcpy", "memset"))
    ]
    if not names:
        raise RuntimeError("CUDA profiler returned no kernel events; cannot certify launch count")
    return names


def kernels_per_call(op, x):
    """Count CUDA kernel events on a warmed, single-call captured graph replay."""
    return len(kernel_names(op, x))


def identical_gemm(ours, theirs):
    """True when both calls launch the same GEMM kernels (names containing "gemm")."""
    gemms = [sorted(name for name in names if "gemm" in name.lower()) for names in (ours, theirs)]
    return bool(gemms[0]) and gemms[0] == gemms[1]


def _elapsed_ms(graph, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def graph_cycles(elapsed_ms):
    """Repeat a complete cold rotation to reach about 1 ms, never inflate slow graphs."""
    return max(1, math.ceil(1 / max(elapsed_ms, 1e-6)))


def _timed_graph(calls):
    graph = _capture(calls)
    cycles = graph_cycles(_elapsed_ms(graph, 1))
    if cycles > 1:
        del graph
        graph = _capture(calls * cycles)
    return graph, len(calls) * cycles


def _sample(graph, calls, replays_per_sample):
    return statistics.median(
        _elapsed_ms(graph, 1) * 1000 / calls for _ in range(replays_per_sample)
    )


def _check_budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("measurement budget exhausted; sweep is incomplete")


def _pairs(graphs, pairs, replays_per_sample, deadline=None):
    times = ([], [])
    for pair in range(pairs):
        _check_budget(deadline)
        for index in (0, 1) if pair % 2 == 0 else (1, 0):
            times[index].append(_sample(*graphs[index], replays_per_sample))
    return times


def _round_robin(graphs, samples, replays_per_sample, deadline=None):
    times = [[] for _ in graphs]
    for sample in range(samples):
        _check_budget(deadline)
        for offset in range(len(graphs)):
            index = (sample + offset) % len(graphs)
            times[index].append(_sample(*graphs[index], replays_per_sample))
    return times


def _measure(
    graphs,
    pairs,
    replays_per_sample,
    deadline=None,
    *,
    comparator="sglang",
    device_name="",
    identical_kernel=None,
):
    ours, sglang = _pairs(graphs[:2], pairs, replays_per_sample, deadline)
    a, b = _pairs(graphs[1:], pairs, replays_per_sample, deadline)
    stats = parity_statistics(
        ours, sglang, a, b, comparator=comparator, device_name=device_name,
        identical_kernel=identical_kernel,
    )  # fmt: skip
    return {
        **stats,
        "samples_us": {
            "ours": ours,
            comparator: sglang,
            "noise_a": a,
            "noise_b": b,
        },
    }


def _measure_candidates(graphs, pairs, replays_per_sample, deadline=None):
    times = _round_robin(graphs, pairs, replays_per_sample, deadline)
    noise = parity_statistics(times[-2], times[-1], times[-2], times[-1])
    initial_eps = noise["eps"]
    if initial_eps > 0.02:
        pairs = max(60, pairs)
        print(
            f"remeasure candidates: eps={initial_eps:.4f}, {pairs} pairs",
            file=sys.stderr,
            flush=True,
        )
        times = _round_robin(graphs, pairs, replays_per_sample, deadline)
        noise = parity_statistics(times[-2], times[-1], times[-2], times[-1])
    return times, {
        "eps": noise["eps"],
        "initial_eps": initial_eps,
        "remeasured": initial_eps > 0.02,
    }


def _sol_record(kind, m, h, i, config, device_name, ours_us):
    """Useful-work SOL for one owned CuTe kernel, using shared empirical ceilings."""
    from ctikernels.linear._nvfp4_dispatch import Cute

    if not isinstance(config, Cute):
        return {"sol": None, "sol_note": "vendor-dispatched row: no SOL gate"}
    device = next(
        (
            key
            for label, key in (
                ("RTX 5090", "rtx5090"),
                ("RTX PRO 6000", "rtx-pro-6000"),
                ("GB10", "gb10"),
            )
            if label in device_name.upper()
        ),
        None,
    )
    if device is None:
        return {"sol": None, "sol_note": "no shared SOL ceilings for device"}
    gated = kind == "gated"
    n, k = (2 * i, h) if gated else (h, i)
    output_scales = math.ceil(m / 128) * math.ceil(i / 64) * 512 if gated else 0
    kernel = sol.Kernel(
        name=f"nvfp4_{kind}",
        dram_bytes={
            "weights": n * k // 2,
            # Block scales, FP32 globals (input/weights/output), and output block scales.
            "scales": n * k // 16 + (16 if gated else 8) + output_scales,
            "activations": 2 * m * k,
            "outputs": m * i // 2 if gated else 2 * m * n,
        },
        flops={"mxf4nvf4_m16n8k64": 2 * m * n * k},
        cache_state="cold",
        pdl=config.pdl,
        # Read-dominated useful traffic; no assumed activation/output L2 residency.
        dram_mode="read",
    )
    source = {"sol_ceilings": f"{device}.json"}
    try:
        reference = sol.estimate(sol.Call([kernel]), device, estimate_type="useful")
    except (OSError, ValueError) as error:
        # Do not expose filesystem paths from an unavailable ceilings file.
        reason = str(error) if isinstance(error, ValueError) else "shared ceilings file unavailable"
        return {**source, "sol": None, "sol_note": reason}
    return {
        **source,
        "sol": reference.to_dict(),
        "sol_us": reference.t_ref_us,
        "sol_fraction": reference.utilization(ours_us).ratio,
        "sol_binding_resource": reference.limiting_resource,
    }


def _gated_weights(gate, up):
    """Preserve weight bytes; exercise independent globals only in our operation."""
    ours_up = up._replace(global_scale=gate.global_scale * 1.25)
    market_up = up._replace(global_scale=gate.global_scale)
    return ours_up, market_up


def _gated_metadata(kind, comparator):
    return (
        {
            "ours_gate_up_globals": "independent-unequal",
            f"{comparator}_gate_up_globals": "collapsed-to-max (same kernels as unequal)",
        }
        if kind == "gated"
        else {}
    )


def _comparator_classes(comparator):
    if comparator == "sglang":
        from .nvfp4_sglang import SGLangGated, SGLangLinear

        return SGLangLinear, SGLangGated
    from functools import partial

    from .nvfp4_vllm import VLLMGated, VLLMLinear

    return VLLMLinear, partial(VLLMGated, eager=comparator == "vllm-eager")


def _operations(kind, name, rotation_bytes, comparator="sglang", provider="dispatch"):
    from ctikernels.linear._nvfp4_dispatch import UNFUSED, FlashInfer
    from ctikernels.linear.nvfp4 import NVFP4GateUpSiLU, NVFP4Linear

    linear, gated = _comparator_classes(comparator)
    options = {"provider": "flashinfer"} if provider == "flashinfer" else {}

    h, i = SHAPES[name]
    packed_bytes = h * i // 2 if kind == "linear" else h * i
    copies = max(2, math.ceil(rotation_bytes / packed_bytes))
    scale = torch.full((1,), 4 / 2688, device="cuda")
    ours, sglang = [], []
    for _ in range(copies):
        if kind == "linear":
            weight = _weight(h, i)
            ours.append(NVFP4Linear(weight, scale, **options))
            sglang.append(linear((weight,), scale))
        else:
            gate, up = _weight(i, h), _weight(i, h)
            up_ours, up_market = _gated_weights(gate, up)
            ours.append(
                NVFP4GateUpSiLU(gate, up_ours, input_scale=scale, output_scale=scale, **options)
            )
            sglang.append(gated(gate, up_market, scale, scale, shared=name == "35b-shared"))
        if provider == "flashinfer":
            _forced(ours[-1], UNFUSED if kind == "gated" else None, FlashInfer())
    return ours, sglang, copies * packed_bytes


def _row(
    kind,
    name,
    m,
    ours,
    sglang,
    cold_bytes,
    device,
    pairs,
    replays_per_sample,
    deadline=None,
    *,
    comparator="sglang",
):
    h, i = SHAPES[name]
    x = torch.randn(m, i if kind == "linear" else h, device="cuda", dtype=torch.bfloat16)
    sglang[0].probe(x)
    metadata = _dispatch_metadata(ours[0], m, gated=kind == "gated")
    names = {"ours": kernel_names(ours[0], x), comparator: kernel_names(sglang[0], x)}
    counts = {side: len(launched) for side, launched in names.items()}
    identical = identical_gemm(names["ours"], names[comparator])
    ours_calls = [lambda op=op: op(x) for op in ours]
    sglang_calls = [lambda op=op: op(x) for op in sglang]
    graphs = [_timed_graph(ours_calls), _timed_graph(sglang_calls), _timed_graph(sglang_calls)]
    measurement = {
        "comparator": comparator,
        "device_name": device["device"],
        "identical_kernel": identical,
    }
    result = _measure(graphs, pairs, replays_per_sample, deadline, **measurement)
    initial_eps = result["eps"]
    if initial_eps > 0.02:
        pairs = max(60, pairs)
        print(
            f"remeasure {kind}/{name}/{m}: eps={initial_eps:.4f}, {pairs} pairs",
            file=sys.stderr,
            flush=True,
        )
        result = _measure(graphs, pairs, replays_per_sample, deadline, **measurement)
    plan = ours[0].plan_for(m)
    config = plan[1] if kind == "gated" else plan
    return {
        "op": kind,
        "shape": name,
        "M": m,
        "H": h,
        "I": i,
        "N": h if kind == "linear" else 2 * i,
        "K": i if kind == "linear" else h,
        **metadata,
        **sglang[0].metadata(m),
        **result,
        "implementation_language": metadata["dispatched_provider"],
        "kernels_per_call": counts,
        "kernel_names": names,
        "copies": len(ours),
        "cold_packed_weight_bytes": cold_bytes,
        "replays_per_sample": replays_per_sample,
        "calls_per_graph": [calls for _, calls in graphs],
        "initial_eps": initial_eps,
        "remeasured": initial_eps > 0.02,
        **_gated_metadata(kind, comparator),
        **_sol_record(kind, m, h, i, config, device["device"], result["ours_us"]),
    }


def candidate_plans(kind, name, m, current, device, version):
    """Current plan, table QuACK tiles, qualified FI, and at most eight Triton plans."""
    from ctikernels.linear import _nvfp4_dispatch as dispatch
    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear._nvfp4_quack import Config

    gated = kind == "gated"
    path, config = current if gated else (None, current)
    provider = (
        "flashinfer"
        if isinstance(config, dispatch.FlashInfer)
        else "triton"
        if isinstance(config, dispatch.Small)
        else "cute"
        if isinstance(config, dispatch.Cute)
        else "quack"
    )
    plans = [(provider, path, config, True)]

    def add(provider, path, config):
        if not any(p[:3] == (provider, path, config) for p in plans):
            plans.append((provider, path, config, False))

    table = dispatch.GATED if gated else {}  # Linear has no table
    configs = dict.fromkeys(
        entry[-1]
        for shapes in table.values()
        for entries in shapes.values()
        for entry in entries
        if isinstance(entry[-1], Config)
    )
    for config in configs:
        for path in (dispatch.FUSED, dispatch.UNFUSED) if gated else (None,):
            add("quack", path, config)
    if fi.qualified(version, device[1]):
        add("flashinfer", dispatch.UNFUSED if gated else None, dispatch.FlashInfer())

    h, i = SHAPES[name]
    n, k = (2 * i, h) if gated else (h, i)
    shape = (h, i)
    small = [
        entry[-1]
        for entry in table.get(device, {}).get(shape, ())
        if isinstance(entry[-1], dispatch.Small)
    ]
    small.extend(
        entry[-1]
        for other, shapes in table.items()
        if other != device
        for entry in shapes.get(shape, ())
        if isinstance(entry[-1], dispatch.Small)
    )
    small.extend(
        dispatch.Small(*fields)
        for fields in (
            (1, 128, 256, 4, 2, True),
            (2, 32, 128, 4, 2, False),
            (4, 64, 128, 4, 2, False),
            (8, 128, 256, 4, 2, True),
            (16, 128, 128, 4, 2, False),
            (8, 32, 256, 4, 2, True),
            (8, 128, 128, 4, 2, True),
            (16, 64, 128, 4, 2, False),
        )
    )
    for config in small:
        if sum(p[0] == "triton" for p in plans) >= 8:
            break
        if dispatch.small_fits(config, m, n, k, gated=gated):
            add("triton", dispatch.TRITON if gated else None, config)
    return plans


def _cute_plans(device, gated, m, name):
    if not 1 <= m <= 32 or os.environ.get("CTI_PARITY_NO_CUTE") == "1":
        return None, ()
    try:
        from ctikernels.linear import _nvfp4_cute_small as cs

        if not cs.available(device):
            return None, ()
        from ctikernels.linear._nvfp4_dispatch import Cute

        configs = cs.sweep(device, gated=gated, shape=f"{name}-{'gated' if gated else 'linear'}")
        return cs, tuple(
            Cute(
                **{
                    key: value
                    for key, value in dataclasses.asdict(config).items()
                    if key != "probe"
                }
            )
            for config in configs
        )
    except Exception as error:  # noqa: BLE001 - experimental API is optional and in flux
        print(f"skip CuTe presets: {type(error).__name__}", file=sys.stderr, flush=True)
        return None, ()


def _cute_calls(cs, ops, x, config, gated):
    config = config.config()
    calls = []
    for op in ops:
        weight = cs.prepare_gated(*op._weights) if gated else cs.prepare(op._weights[0])
        workspace = cs.Workspace(weight, x.shape[0], config, gated=gated)
        if gated:
            call = lambda w=weight, ws=workspace, op=op: cs.gated(
                x, w, op.input_scale, op.output_scale, config, workspace=ws
            )
        else:
            call = lambda w=weight, ws=workspace, op=op: cs.linear(
                x, w, op.input_scale, config, workspace=ws
            )
        calls.append(call)
    return calls


def _candidate_record(provider, path, config, current):
    fields = (
        config._asdict()
        if provider == "cute"
        else dataclasses.asdict(config)
        if dataclasses.is_dataclass(config)
        else list(config)
    )
    return {
        "candidate_id": f"{provider}/{path}/{config!r}",
        "provider": provider,
        "dispatched_provider": provider,
        "implementation_language": provider,
        "path": path,
        "config": fields,
        "plan_repr": repr(config),
        "current": current,
    }


def _candidate_row(
    kind,
    name,
    m,
    ours,
    sglang,
    cold_bytes,
    device,
    pairs,
    replays_per_sample,
    deadline,
    *,
    comparator="sglang",
):
    from ctikernels.linear.nvfp4 import NVFP4GateUpSiLU, NVFP4Linear

    h, i = SHAPES[name]
    gated = kind == "gated"
    x = torch.randn(m, h if gated else i, device="cuda", dtype=torch.bfloat16)
    sglang[0].probe(x)
    base = {
        **device,
        **sglang[0].metadata(m),
        **_gated_metadata(kind, comparator),
        "row_id": f"{kind}/{name}/{m}",
        "op": kind,
        "shape": name,
        "M": m,
        "H": h,
        "I": i,
        "N": 2 * i if gated else h,
        "K": h if gated else i,
        "copies": len(ours),
        "cold_packed_weight_bytes": cold_bytes,
        "replays_per_sample": replays_per_sample,
    }
    plans = candidate_plans(
        kind,
        name,
        m,
        ours[0].plan_for(m),
        (device["device"], tuple(device["capability"])),
        device["flashinfer_version"],
    )
    cs, presets = _cute_plans(x.device, gated, m, name)
    plans.extend(
        ("cute", "cute" if gated else None, config, False)
        for config in presets
        if not any(p[0] == "cute" and p[2] == config for p in plans)
    )
    if cs is None and any(p[0] == "cute" for p in plans):
        from ctikernels.linear import _nvfp4_cute_small as cs
    # Captured graphs borrow prepared weights and CuTe workspaces from these calls.
    graphs, records, retained, configs = [], [], [], []
    flash_ops = None
    try:
        for provider, path, config, current in plans:
            _check_budget(deadline)
            record = {**base, "provider": provider, "current": current}
            try:
                record.update(_candidate_record(provider, path, config, current))
                print(
                    f"preparing {base['row_id']} {record['candidate_id']}",
                    file=sys.stderr,
                    flush=True,
                )
                if provider == "cute":
                    calls = _cute_calls(cs, ours, x, config, gated)
                else:
                    if provider == "flashinfer" and flash_ops is None:
                        flash_ops = [
                            NVFP4GateUpSiLU(
                                *op._weights,
                                input_scale=op.input_scale,
                                output_scale=op.output_scale,
                                provider="flashinfer",
                            )
                            if gated
                            else NVFP4Linear(op._weights[0], op.input_scale, provider="flashinfer")
                            for op in ours
                        ]
                    selected = flash_ops if provider == "flashinfer" else ours
                    for op in selected:
                        _forced(op, path, config)
                    calls = [lambda op=op: op(x) for op in selected]
                count = kernels_per_call(lambda _, call=calls[0]: call(), x)
                graph = _timed_graph(calls)
            except Exception as error:
                if current:
                    raise
                print(
                    "CAND "
                    + json.dumps(
                        {
                            **record,
                            "status": "skipped",
                            "verdict": "inconclusive",
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                continue
            configs.append(config)
            retained.append(calls)
            graphs.append(graph)
            records.append({**record, "kernels_per_call": count, "calls_per_graph": graph[1]})
        _check_budget(deadline)
        sglang_calls = [lambda op=op: op(x) for op in sglang]
        graphs.extend((_timed_graph(sglang_calls), _timed_graph(sglang_calls)))
        times, noise = _measure_candidates(graphs, pairs, replays_per_sample, deadline)
        reference = statistics.median(times[-2])
        for record, samples, config in zip(records, times[:-2], configs, strict=True):
            print(
                "CAND "
                + json.dumps(
                    {
                        **record,
                        **noise,
                        **parity_statistics(
                            samples,
                            times[-2],
                            times[-2],
                            times[-1],
                            comparator=comparator,
                            device_name=device["device"],
                        ),
                        **_sol_record(
                            kind, m, h, i, config, device["device"], statistics.median(samples)
                        ),
                        "status": "ok",
                        "median_us": statistics.median(samples),
                        "samples_us": samples,
                        "samples": len(samples),
                        f"{comparator}_us": reference,
                        f"{comparator}_samples_us": times[-2],
                        f"{comparator}_control_samples_us": times[-1],
                    }
                ),
                flush=True,
            )
    finally:
        for op in ours:
            op.__dict__.pop("plan_for", None)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="all/all/all")
    parser.add_argument("--pairs", type=int, default=20, help="initial pairs (GB10 minimum: 60)")
    parser.add_argument(
        "--pin-x925", action="store_true", help="pin to Cortex-X925 CPUs (automatic on GB10)"
    )
    parser.add_argument("--ms", help="comma-separated M filter, intersected with --rows")
    parser.add_argument("--replays-per-sample", type=int, default=5)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--provider", choices=("dispatch", "flashinfer"), default="dispatch")
    parser.add_argument("--budget-seconds", type=float, default=780)
    parser.add_argument("--sglang-python")
    parser.add_argument("--comparator", choices=("sglang", "vllm", "vllm-eager"), default="sglang")
    return parser


@torch.inference_mode()
def main():
    parser = _parser()
    args = parser.parse_args()
    if args.provider == "flashinfer" and args.candidates:
        parser.error("--candidates cannot be combined with --provider flashinfer")
    if args.pairs < 2:
        parser.error("--pairs must be at least 2")
    if (
        args.replays_per_sample < 1
        or not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
    ):
        parser.error("--replays-per-sample and --budget-seconds must be positive")
    try:
        rows = select_rows(args.rows)
        if args.ms is not None:
            selected = {m for _, _, m in select_rows(f"all/all/{args.ms}")}
            rows = [row for row in rows if row[2] in selected]
        if not rows:
            raise ValueError("row filters selected no rows")
    except ValueError as error:
        parser.error(str(error))
    fi_version = "0.7.1" if args.comparator == "sglang" else "0.7.0.post1"
    if importlib.metadata.version("flashinfer-python") != fi_version:
        raise RuntimeError(f"{args.comparator} parity requires FlashInfer {fi_version}")

    started = time.monotonic()
    deadline = started + args.budget_seconds
    device_name, cpus = _timing_setup(args)
    if args.comparator == "sglang":
        from .nvfp4_sglang import load

        load(args.sglang_python)
    else:
        from .nvfp4_vllm import load

        load()
    props = torch.cuda.get_device_properties(0)
    device = {
        "device": device_name,
        "x925_cpus": cpus,
        "speedup_ci_low": None,
        "speedup_ci_high": None,
        "capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "flashinfer_version": fi_version,
        "comparator": args.comparator,
        "provider_mode": args.provider,
    }
    if args.comparator != "sglang":
        device["vllm_version"] = importlib.metadata.version("vllm")
    l2 = getattr(props, "L2_cache_size", 0)
    if not l2:
        raise RuntimeError(
            "PyTorch did not report L2_cache_size; cannot establish cold-cache rotation"
        )
    rotation = 2 * l2 + 1
    device.update(L2_bytes=l2, rotation_target_bytes=rotation)
    for kind, name in product(OPS, SHAPES):
        batches = [m for op, shape, m in rows if (op, shape) == (kind, name)]
        if not batches:
            continue
        _check_budget(deadline)
        print(f"preparing {kind}/{name} cold weights", file=sys.stderr, flush=True)
        ours, sglang, cold_bytes = _operations(kind, name, rotation, args.comparator, args.provider)
        for m in batches:
            _check_budget(deadline)
            print(
                f"measuring {kind}/{name}/{m} ({time.monotonic() - started:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            if args.candidates:
                _candidate_row(
                    kind,
                    name,
                    m,
                    ours,
                    sglang,
                    cold_bytes,
                    device,
                    args.pairs,
                    args.replays_per_sample,
                    deadline,
                    comparator=args.comparator,
                )
            else:
                record = _row(
                    kind,
                    name,
                    m,
                    ours,
                    sglang,
                    cold_bytes,
                    device,
                    args.pairs,
                    args.replays_per_sample,
                    deadline,
                    comparator=args.comparator,
                )
                print(
                    "PARITY "
                    + json.dumps({**device, **record, "elapsed_s": time.monotonic() - started}),
                    flush=True,
                )
            gc.collect()
            torch.cuda.empty_cache()
        del ours, sglang
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as error:
        print(f"INCOMPLETE: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
