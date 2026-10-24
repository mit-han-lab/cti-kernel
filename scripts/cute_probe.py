#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Whole product CuTe/Triton/FLA checks and flushed graph timings; run from the source root.

All provider graphs reset identical input pools before their one call. Reset and L2 flush
are outside the timed replay. Timing alternates provider order and retains all 20 samples.
FP64 reference construction is deliberately outside warmup/capture/timing.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

# Run from the source tree root; expose its tests as a package.
sys.path.insert(0, str(Path.cwd()))
from ctikernels.gdn import GatedDeltaNetPrefill
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn._cute_prefill import compile_info, launch_info, read_tune
from tests.gpu._gdn_prefill_cases import (
    ABSOLUTE_FLOOR,
    _fla_prefill,
    _inputs,
    _max_error,
    _pools,
    _prefill_case,
    _rms_error,
    reference,
)
from tests.gpu.test_gdn_cute_prefill import LENGTHS, _precision_inputs

# Explicit --tune still selects precisely the caller's configs in either mode.
OUTPUT_TUNES = ("", "aq_precise=1", "aq_qk=0,aq_out=1", "aq_qk=0,aq_out=0")
PRECISION_TUNES = ("", "aq_qk=0", "aq_out=1", "aq_qk=0,aq_out=0")


def emit(record):
    print(json.dumps(record), flush=True)


def construct(hk, hv, mode):
    old = os.environ.get("CTI_GDN_PREFILL_CUTE")
    os.environ["CTI_GDN_PREFILL_CUTE"] = mode
    try:
        return GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    finally:
        if old is None:
            os.environ.pop("CTI_GDN_PREFILL_CUTE", None)
        else:
            os.environ["CTI_GDN_PREFILL_CUTE"] = old


def case(hk, hv, lengths, kind, reps, tune, only_cute, compile_directory):
    geometry = Geometry(hk, hv)
    weights, x = _inputs(hk, hv, lengths, kind, 1234)
    count = len(lengths)
    slots = list(reversed(range(count)))
    flags = [s % 2 == 0 for s in range(count)]
    meta = _prefill_case(lengths, slots, flags)
    base = _pools(hk, hv, seed=1236, slots=count + 2)
    timing_only = tune["stop_after"] != "all"
    exact = None if timing_only else reference.prefill(x, weights, hk, *meta, *base)
    cute = construct(hk, hv, "1")
    assert cute.selected_variant(sum(lengths), count) == "cute", (
        "CuTe unavailable; refusing mislabeled timing"
    )
    cute._kernel._cute.options += f" --keep-cubin --dump-dir={compile_directory}"
    prefix = (x, *weights.args(), *meta)
    providers = {"cute": cute}
    if not only_cute and not timing_only:
        triton = construct(hk, hv, "0")
        assert triton.selected_variant(sum(lengths), count) == "triton"
        providers.update(triton=triton, **{"fla-core": lambda *args: _fla_prefill(geometry, *args)})
    graphs, pools, samples = {}, {}, {name: [] for name in providers}
    identity = {
        "geometry": [hk, hv],
        "lengths": lengths,
        "kind": kind,
        "tune": tune,
        "timing_only": timing_only,
    }
    for name, op in providers.items():
        work = [p.clone() for p in base]
        pools[name] = work
        y = op(*prefix, *work)
        torch.cuda.synchronize()
        if name == "cute":
            launcher = cute._kernel._cute
            key = launcher.compile_key(Geometry(hk, hv), x, count)
            effective_tune = launcher._effective_tune(hv, key[3], key)
            identity.update(
                effective_tune=effective_tune,
                cute_launches=len(launch_info(hk, hv, sum(lengths), count, effective_tune)),
            )
            emit(
                {
                    **identity,
                    "provider": name,
                    "compile_info": compile_info(
                        compile_directory
                        if effective_tune["overlap"]
                        else Path(compile_directory) / "all",
                        hk,
                        hv,
                        sum(lengths),
                        count,
                        effective_tune,
                    ),
                    "admission_resources": launcher.resources.get(key),
                }
            )
        if not timing_only:
            emit(
                {
                    **identity,
                    "provider": name,
                    "check": {
                        "y_max": _max_error(y, exact[0]),
                        "y_rms": _rms_error(y, exact[0]),
                        "state_max": _max_error(work[0][slots], exact[1][slots]),
                        "state_rms": _rms_error(work[0][slots], exact[1][slots]),
                        "history_equal": torch.equal(work[1], exact[2]),
                    },
                }
            )
        # Warm on a non-default stream, then capture there. Compiles stay outside capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for dst, src in zip(work, base, strict=True):
                    dst.copy_(src)
                op(*prefix, *work)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_y = op(*prefix, *work)
        graphs[name] = (graph, captured_y)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    names = list(providers)
    for repetition in range(reps):
        for name in names if repetition % 2 == 0 else list(reversed(names)):
            for dst, src in zip(pools[name], base, strict=True):
                dst.copy_(src)
            flush.zero_()
            events[0].record()
            graphs[name][0].replay()
            events[1].record()
            events[1].synchronize()
            samples[name].append(events[0].elapsed_time(events[1]) * 1000.0)
    for name in names:
        emit(
            {
                **identity,
                "provider": name,
                "median_us": statistics.median(samples[name]),
                "samples_us": samples[name],
                "repetitions": reps,
                "flush_bytes": flush.numel(),
            }
        )

    if os.environ.get("CTI_GDN_PREFILL_CUTE_PROFILE") == "1":
        # Diagnostic only; timings above remain unprofiled. Requires working CUPTI.
        for dst, src in zip(pools["cute"], base, strict=True):
            dst.copy_(src)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            graphs["cute"][0].replay()
            torch.cuda.synchronize()
        kernels = [
            {"name": event.name, "duration_us": event.device_time_total}
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        emit(
            {
                **identity,
                "provider": "cute",
                "profiled_kernel_events": kernels,
                "profiled_event_count": len(kernels),
            }
        )


def precision_prefix_reference(weights, x, hk, base, seed_index, lengths):
    """Exact reference steps once for shared prefixes; snapshot each requested state.

    The single-sequence test always uses slot 7 and alternates its initial flag.
    Uses the reference's own operations/order, including convolution history; no
    alternate/chunked FP64 formula. Callers verify the generated rows are prefixes.
    """
    state, history = base[0][7], base[1][7]
    if not seed_index % 2:
        state, history = torch.zeros_like(state), torch.zeros_like(history)
    sequence = reference._Sequence(weights, hk, state, history)
    y = torch.empty(x.shape[0], weights.A_log.numel() * 128, dtype=torch.float64, device=x.device)
    checkpoints = set(lengths)
    states = {}
    for token in range(x.shape[0]):
        y[token] = sequence.step(x[token])
        if token + 1 in checkpoints:
            states[token + 1] = sequence.pool_state().clone()[None]
    return y, states


def precision_sweep(geometries, lengths_cases, kinds, seeds, configs):
    """Share reference/FLA work across configs and exact reference prefixes across lengths.

    No graphs, profiling, warmups, cache flushes, cubin dumps or Triton comparison.
    Operators persist across kinds/seeds/lengths; each shape/config compiles once.
    Each gate is reported separately, so an aggregate cannot hide a failing case.
    """
    started = time.monotonic()
    summary = [[] for _ in configs]
    single_lengths = [lengths[0] for lengths in lengths_cases if len(lengths) == 1]
    for hk, hv in geometries:
        operators = []
        for tune in configs:
            os.environ["CTI_GDN_CUTE_TUNE"] = ",".join(f"{k}={v}" for k, v in tune.items())
            op = construct(hk, hv, "1")
            assert op.selected_variant(64, 1) == "cute", "CuTe unavailable"
            operators.append(op)
        for kind in kinds:
            errors = [[[] for _ in configs] for _ in lengths_cases]
            for seed_index in range(seeds):
                if single_lengths:
                    weights, prefix_x, _, base, _, _ = _precision_inputs(
                        hk, hv, [max(single_lengths)], kind, seed_index
                    )
                    prefix_y, prefix_states = precision_prefix_reference(
                        weights, prefix_x, hk, base, seed_index, single_lengths
                    )
                for case_index, lengths in enumerate(lengths_cases):
                    weights, x, meta, base, poisoned, written = _precision_inputs(
                        hk, hv, lengths, kind, seed_index
                    )
                    if len(lengths) == 1 and torch.equal(x, prefix_x[: lengths[0]]):
                        ey, es = prefix_y[: lengths[0]], prefix_states[lengths[0]]
                    else:
                        # Ragged cases, or a generator whose output depends on its shape.
                        exact = reference.prefill(x, weights, hk, *meta, *base)
                        ey, es = exact[0], exact[1][written]
                        del exact
                    fla_pools = [p.clone() for p in base]
                    fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
                    fs = fla_pools[0][written]
                    fla = (_max_error(fy, ey), _max_error(fs, es))
                    fla_rms = (_rms_error(fy, ey), _rms_error(fs, es))
                    for index, op in enumerate(operators):
                        pools = [p.clone() for p in poisoned]
                        y = op(x, *weights.args(), *meta, *pools)
                        state = pools[0][written]
                        ours = (_max_error(y, ey), _max_error(state, es))
                        rms = (_rms_error(y, ey), _rms_error(state, es))
                        valid = bool(torch.isfinite(y).all() and torch.isfinite(state).all())
                        rms_ok = valid and all(
                            a <= b + ABSOLUTE_FLOOR for a, b in zip(rms, fla_rms, strict=True)
                        )
                        errors[case_index][index].append((*ours, *fla, rms_ok))
                    del ey, es, fy, fs, fla_pools, pools, y, state, base, poisoned, x, weights
                if single_lengths:
                    del prefix_y, prefix_states, prefix_x
            for case_index, lengths in enumerate(lengths_cases):
                for index, tune in enumerate(configs):
                    rows = errors[case_index][index]
                    cy, cs, fy, fs = [sum(row[i] for row in rows) / len(rows) for i in range(4)]
                    ratio = lambda a, b: a / b if b else (0.0 if a == 0 else None)
                    passed = cy <= fy + ABSOLUTE_FLOOR and cs <= fs + ABSOLUTE_FLOOR
                    passed = passed and all(row[4] for row in rows)
                    record = {
                        "mode": "precision",
                        "config": index,
                        "tune": tune,
                        "geometry": [hk, hv],
                        "lengths": lengths,
                        "kind": kind,
                        "seeds": seeds,
                        "cute": {"mean_y_max": cy, "mean_state_max": cs},
                        "fla-core": {"mean_y_max": fy, "mean_state_max": fs},
                        "ratio": {"y": ratio(cy, fy), "state": ratio(cs, fs)},
                        "gate_pass": passed,
                        "state_target_pass": cs <= 0.8 * fs,
                    }
                    emit(record)
                    summary[index].append(record)
    for index, records in enumerate(summary):
        emit(
            {
                "mode": "precision_summary",
                "config": index,
                "tune": configs[index],
                "cases": len(records),
                "seeds": seeds,
                "gate_pass": all(r["gate_pass"] for r in records),
                "state_target_pass": all(r["state_target_pass"] for r in records),
                "worst_state_ratio": max(
                    (r["ratio"]["state"] for r in records if r["ratio"]["state"] is not None),
                    default=None,
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
        )


def sweep_plan(configs, geoms, lengths, precision=False):
    """CPU-only parser; a single group broadcasts, | groups match tune indices.

    Example: --tune 'bv=32;bv=64' --geoms '16x32|16x48'
             --lengths '2048|1024;2048'. This avoids a large cross product.
    """

    def groups(value):
        values = value.split("|")
        if len(values) == 1:
            return values * len(configs)
        if len(values) != len(configs):
            raise ValueError("| groups must match the number of tune configurations")
        return values

    default_lengths = "64;1000;1024;2048;4096" if precision else "1024;2048"
    plans = []
    for raw, gs, ls in zip(configs, groups(geoms), groups(lengths or default_lengths), strict=True):
        geometry = [tuple(map(int, g.split("x"))) for g in gs.split(",")]
        cases = [list(map(int, case.split(","))) for case in ls.split(";")]
        if any(g not in ((16, 32), (16, 48)) for g in geometry):
            raise ValueError("--geoms supports 16x32 and 16x48")
        if any(not case or min(case) < 1 for case in cases):
            raise ValueError("lengths must be positive")
        if precision and (
            read_tune(raw)["stop_after"] != "all"
            or any(case not in LENGTHS + [[1, 2, 61]] for case in cases)
        ):
            raise ValueError("precision requires stop_after=all and product gate lengths")
        plans.append((raw, geometry, cases))
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geoms",
        default="16x32,16x48",
        help="comma-separated geometries; use | for one group per tune",
    )
    parser.add_argument(
        "--lengths",
        default=None,
        help="semicolon-separated cases, comma-separated lengths; | selects one group per tune",
    )
    parser.add_argument(
        "--kind",
        "--kinds",
        dest="kind",
        default=None,
        help="comma-separated ordinary,slow,heavy; precision defaults ordinary,slow",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--prec-only",
        action="store_true",
        help="exact test inputs/metrics; defaults to the five single-sequence LENGTHS",
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--only-cute", action="store_true", help="skip Triton/FLA runs; retain CuTe vs FP64 checks"
    )
    parser.add_argument(
        "--profile-kernels", action="store_true", help="optional CUPTI kernel timings"
    )
    parser.add_argument(
        "--tune",
        default=None,
        help="CuTe k=v pairs separated by ';'; defaults to four output forms (six with prec-only)",
    )
    args = parser.parse_args()
    try:
        configs = (
            args.tune.split(";")
            if args.tune is not None
            else list(PRECISION_TUNES)
            if args.prec_only
            else list(OUTPUT_TUNES)
        )
        for config in configs:
            read_tune(config)  # Validate before any GPU work; resolve defaults per geometry.
        if len(configs) > 6:
            parser.error("split the sweep into calls with at most six configurations")
        plans = sweep_plan(configs, args.geoms, args.lengths, args.prec_only)
    except ValueError as error:
        parser.error(str(error))
    if args.profile_kernels:
        os.environ["CTI_GDN_PREFILL_CUTE_PROFILE"] = "1"
    if args.reps < 1 or args.seeds < 1:
        parser.error("--reps and --seeds must be positive")
    kinds = (args.kind or ("ordinary,slow" if args.prec_only else "ordinary")).split(",")
    if any(kind not in ("ordinary", "slow", "heavy") for kind in kinds):
        parser.error("--kind must contain ordinary,slow,heavy")
    if args.prec_only and (args.only_cute or args.profile_kernels):
        parser.error("--prec-only requires FLA and excludes profiling")
    old_tune = os.environ.get("CTI_GDN_CUTE_TUNE")
    try:
        capability = torch.cuda.get_device_capability()
        # Identical shape groups retain reference reuse in precision mode.
        if args.prec_only:
            groups = {}
            for raw, geoms, lengths_cases in plans:
                group = (tuple(geoms), tuple(tuple(x) for x in lengths_cases))
                groups.setdefault(group, []).append(raw)
            for (geoms, lengths_cases), raw_configs in groups.items():
                for hk, hv in geoms:
                    resolved = [read_tune(raw, capability, (hk, hv)) for raw in raw_configs]
                    with torch.inference_mode():
                        precision_sweep([(hk, hv)], lengths_cases, kinds, args.seeds, resolved)
            return
        for raw, geoms, lengths_cases in plans:
            for hk, hv in geoms:
                tune = read_tune(raw, capability, (hk, hv))
                os.environ["CTI_GDN_CUTE_TUNE"] = ",".join(
                    f"{key}={value}" for key, value in sorted(tune.items())
                )
                for lengths in lengths_cases:
                    for kind in kinds:
                        run_case(hk, hv, lengths, kind, args, tune)
    finally:
        if old_tune is None:
            os.environ.pop("CTI_GDN_CUTE_TUNE", None)
        else:
            os.environ["CTI_GDN_CUTE_TUNE"] = old_tune


def run_case(hk, hv, lengths, kind, args, tune):
    try:
        with tempfile.TemporaryDirectory(prefix="gdn-cute-probe-") as directory:
            with torch.inference_mode():
                case(
                    hk,
                    hv,
                    lengths,
                    kind,
                    args.reps,
                    tune,
                    args.only_cute,
                    directory,
                )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
