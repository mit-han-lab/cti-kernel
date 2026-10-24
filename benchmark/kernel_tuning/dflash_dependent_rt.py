# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Dependent global round trips, measured as CUDA-graph ns per kernel launch.

Run with ``python -m benchmark.kernel_tuning.dflash_dependent_rt``. Each JSONL
record describes one device/mode, the raw per-k medians and an unconstrained
least-squares fit. CUDA/CuTe imports are lazy so arguments and plans work on CPU.

Each CTA has 32 threads but only lane zero chases pointers. Its final index is
also the next launch's starting cursor. k=0 still reads/writes that cursor: it
measures the common launch/prologue/epilogue floor, not an empty CUDA graph.
The atomic mode substitutes returning GPU-scope add(0) for each load; the fence
mode issues a private store, GPU acq_rel fence, then the dependent load.

Timing uses 64 launches per graph and 25 synchronized wall-clock replay samples
by default, following dflash_replay_floor's timing boundaries. Optional CUDA
events match dflash_draft_layers; these are different timing quantities.
"""

import argparse
import contextlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

MODES = ("l2", "dram", "atomic", "fence")
KS = (0, 1, 2, 4, 8, 16, 32)
ARCHES = ("sm_120a", "sm_121a")
LINE_BYTES = 128
LINE_WORDS = LINE_BYTES // 4


def csv_modes(value):
    values = tuple(item.strip() for item in value.split(","))
    if not values or any(item not in MODES for item in values):
        raise argparse.ArgumentTypeError("expected modes from l2,dram,atomic,fence")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("modes must be unique")
    return values


def csv_ks(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from None
    if len(set(values)) != len(values) or any(k not in KS for k in values):
        raise argparse.ArgumentTypeError("ks must be unique members of 0,1,2,4,8,16,32")
    if 0 not in values or len(values) < 2:
        raise argparse.ArgumentTypeError("ks must include the k=0 floor and a positive k")
    return tuple(sorted(values))


def positive(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=csv_modes, default=MODES)
    parser.add_argument("--ks", type=csv_ks, default=KS)
    parser.add_argument("--ctas", type=positive, default=1, help="1..device SM count")
    parser.add_argument("--repeat", type=positive, default=25)
    parser.add_argument("--launches", type=positive, default=64)
    parser.add_argument(
        "--timer",
        choices=("wall", "events"),
        default="wall",
        help="wall: replay-floor protocol; events: device graph elapsed time",
    )
    parser.add_argument("--device", type=int, help="CUDA ordinal; default: all visible devices")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.device is not None and args.device < 0:
        parser.error("device must be nonnegative")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args


@dataclass(frozen=True)
class Plan:
    mode: str
    ctas: int
    buffer_bytes: int
    lines_per_cta: int
    eviction_bytes: int
    launches: int
    repeat: int
    ks: tuple


def make_plan(args, mode, sms, l2_bytes):
    """Partition cache lines into disjoint cycles; never reuse DRAM nodes per replay."""
    if mode not in MODES:
        raise ValueError("unknown mode")
    if not 1 <= args.ctas <= sms:
        raise ValueError("ctas must be between 1 and the device SM count")
    if l2_bytes < LINE_BYTES * args.ctas * 4:
        raise ValueError("a positive, sufficiently large device L2 size is required")
    if mode == "dram":
        # Size alone cannot make a short, repeatedly replayed chase cold. Each
        # replay starts after eviction and consumes distinct cache lines, even
        # across launch boundaries. The +1 also covers the final cursor target.
        lines = max(
            math.ceil(2 * l2_bytes / (LINE_BYTES * args.ctas)),
            args.launches * max(args.ks) + 1,
        )
        eviction = 4 * l2_bytes
    else:
        # Leave capacity for cursors, scratch and unrelated runtime traffic.
        budget = min(4 << 20, l2_bytes // 4)
        lines = budget // (LINE_BYTES * args.ctas)
        eviction = 0
    size = lines * args.ctas * LINE_BYTES
    if size // 4 >= 2**31:
        raise ValueError("buffer exceeds the signed int32 index range")
    return Plan(mode, args.ctas, size, lines, eviction, args.launches, args.repeat, args.ks)


def cycle_order(lines, seed):
    """A random single cycle visits every line, unlike a random successor map."""
    order = list(range(lines))
    random.Random(seed).shuffle(order)
    return order


def fit_latency(medians):
    """Fit t(k) = intercept + k*slope, including zero; report noisy fits honestly."""
    xs = sorted(medians)
    if len(xs) < 2 or any(not math.isfinite(medians[k]) or medians[k] <= 0 for k in xs):
        raise ValueError("fit needs at least two finite, positive latency measurements")
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(medians.values())
    slope = sum((k - mean_x) * (medians[k] - mean_y) for k in xs) / sum(
        (k - mean_x) ** 2 for k in xs
    )
    intercept = mean_y - slope * mean_x
    residuals = {str(k): medians[k] - intercept - slope * k for k in xs}
    sse = sum(r * r for r in residuals.values())
    sst = sum((medians[k] - mean_y) ** 2 for k in xs)
    return {
        "slope_ns_per_rt": slope,
        "intercept_ns": intercept,
        "r_squared": 1 - sse / sst if sst else None,
        "residual_ns": residuals,
        "fit_ks": xs,
    }


def x925_cores(cpuinfo, allowed):
    cores = set()
    for stanza in re.split(r"\n\s*\n", cpuinfo.strip()):
        fields = dict(line.split(":", 1) for line in stanza.splitlines() if ":" in line)
        fields = {key.strip().lower(): value.strip() for key, value in fields.items()}
        try:
            if int(fields.get("cpu part", "0"), 16) == 0xD85:
                cores.add(int(fields["processor"]))
        except (ValueError, KeyError):
            continue
    return sorted(cores & set(allowed))


def pin_x925():
    if not hasattr(os, "sched_getaffinity"):
        return {"status": "unavailable", "cores": []}
    try:
        allowed = os.sched_getaffinity(0)
        cores = x925_cores(Path("/proc/cpuinfo").read_text(), allowed)
        if cores:
            os.sched_setaffinity(0, cores)
        return {"status": "pinned" if cores else "unchanged", "cores": cores or sorted(allowed)}
    except OSError:
        return {"status": "failed", "cores": []}


@lru_cache(maxsize=1)
def kernel_type():
    """Keep CPU planning independent of the optional CUDA DSL installation."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    class Chase:
        def __init__(self, mode, k, ctas):
            self.mode, self.k, self.ctas = mode, k, ctas

        @cute.jit
        def __call__(
            self, buf: cute.Tensor, out: cute.Tensor, scratch: cute.Tensor, stream: cuda.CUstream
        ):
            self.kernel(buf, out, scratch).launch(
                grid=(self.ctas, 1, 1), block=(32, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(self, buf: cute.Tensor, out: cute.Tensor, scratch: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            cta, _, _ = cute.arch.block_idx()
            if tid == 0:
                slot = cta * LINE_WORDS
                # Explicit memory operations keep the identical cursor read
                # and final store even for k=0 (no dead load/store elimination).
                cursor = out.iterator.toint() + cutlass.Int64(slot) * 4
                idx = cute.arch.inline_ptx(
                    "ld.global.cg.u32 {$w0}, [{$r0}];",
                    write_only_types=[cutlass.Int32],
                    read_only_args=[cursor],
                )
                for _ in cutlass.range_constexpr(self.k):
                    address = buf.iterator.toint() + cutlass.Int64(idx) * 4
                    if cutlass.const_expr(self.mode == "atomic"):
                        # A returned RMW result forms the NEXT address. add(0)
                        # preserves the cycle; this must not become a reduction.
                        idx = cute.arch.inline_ptx(
                            "atom.relaxed.gpu.global.add.u32 {$w0}, [{$r0}], 0;",
                            write_only_types=[cutlass.Int32],
                            read_only_args=[address],
                        )
                    elif cutlass.const_expr(self.mode == "fence"):
                        # One asm block preserves store -> fence -> load order.
                        # Private scratch avoids a forwarding load from the store.
                        dest = scratch.iterator.toint() + cutlass.Int64(slot) * 4
                        idx = cute.arch.inline_ptx(
                            "st.global.cg.u32 [{$r1}], {$r2}; "
                            "fence.acq_rel.gpu; "
                            "ld.global.cg.u32 {$w0}, [{$r0}];",
                            write_only_types=[cutlass.Int32],
                            read_only_args=[address, dest, idx],
                        )
                    else:
                        idx = cute.arch.inline_ptx(
                            "ld.global.cg.u32 {$w0}, [{$r0}];",
                            write_only_types=[cutlass.Int32],
                            read_only_args=[address],
                        )
                cute.arch.inline_ptx(
                    "st.global.cg.u32 [{$r0}], {$r1};", read_only_args=[cursor, idx]
                )

    return Chase


def compile_kernel(mode, k, ctas, arch):
    """Compile real device code using fake tensors/stream, with no CUDA allocation."""
    if mode not in MODES or k not in KS or ctas < 1 or arch not in ARCHES:
        raise ValueError("invalid kernel specialization")
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    fake = cute.runtime.make_fake_tensor
    tensors = [fake(cutlass.Int32, (cute.sym_int(),), (1,), assumed_align=16) for _ in range(3)]
    return cute.compile(
        kernel_type()(mode, k, ctas),
        *tensors,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


def allocate_buffers(plan, device, seed):
    import torch

    host = torch.zeros(plan.buffer_bytes // 4, dtype=torch.int32)
    cursors = torch.zeros(plan.ctas * LINE_WORDS, dtype=torch.int32)
    for cta in range(plan.ctas):
        order = torch.tensor(cycle_order(plan.lines_per_cta, seed + cta), dtype=torch.int64)
        indices = (order + cta * plan.lines_per_cta) * LINE_WORDS
        host[indices] = indices.roll(-1).to(torch.int32)
        cursors[cta * LINE_WORDS] = indices[0]
    return host.to(device), cursors.to(device), torch.zeros_like(cursors, device=device)


def device_l2_bytes(properties, ordinal):
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, name, 0)
        if value:
            return int(value)
    # No guessed hardware constants: the cold-buffer plan needs the real size.
    from cuda.bindings import runtime

    status, value = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize, ordinal
    )
    if int(status) or value <= 0:
        raise RuntimeError("cannot query device L2 size")
    return int(value)


def measure_mode(args, plan, ordinal, arch):
    import torch

    device = torch.device("cuda", ordinal)
    buf, out, scratch = allocate_buffers(plan, device, args.seed)
    kernels = {k: compile_kernel(plan.mode, k, plan.ctas, arch) for k in plan.ks}
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graphs = {}
    # Initialize TVM-FFI argument adapters and CUDA modules outside capture.
    with torch.cuda.stream(stream):
        for kernel in kernels.values():
            kernel(buf, out, scratch)
    stream.synchronize()
    for k, kernel in kernels.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(plan.launches):
                kernel(buf, out, scratch)
        graphs[k] = graph
    stream.synchronize()
    torch.cuda.current_stream(device).wait_stream(stream)
    sink = torch.empty((), dtype=torch.int64, device=device)
    eviction = (
        # Incompressible data avoids a zero-filled eviction buffer occupying
        # less effective cache capacity on hardware with memory compression.
        torch.randint(
            0,
            2**31 - 1,
            (plan.eviction_bytes // 4,),
            dtype=torch.int32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        if plan.eviction_bytes
        else None
    )

    def condition_cache():
        # Consume every sector of eviction memory, or warm every chase node.
        source = eviction if eviction is not None else buf[::LINE_WORDS]
        sink.copy_(source.sum(dtype=torch.int64).reshape(sink.shape))
        # Keep the fixed cursor read hot even when chase data is cold.
        sink.copy_(out.sum(dtype=torch.int64).reshape(sink.shape))

    for graph in graphs.values():
        for _ in range(3):
            condition_cache()
            graph.replay()
    torch.cuda.synchronize(device)
    samples = {k: [] for k in plan.ks}
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for repetition in range(plan.repeat):
        # Rotate/interleave k to spread clock and thermal drift across the fit.
        shift = repetition % len(plan.ks)
        for k in plan.ks[shift:] + plan.ks[:shift]:
            condition_cache()
            if args.timer == "wall":
                # Same boundaries as dflash_replay_floor.measure: conditioning
                # is complete before timing; the final synchronization counts.
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                graphs[k].replay()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter_ns() - start
            else:
                begin.record()
                graphs[k].replay()
                end.record()
                end.synchronize()
                elapsed = begin.elapsed_time(end) * 1e6
            samples[k].append(elapsed / plan.launches)
    medians = {k: statistics.median(values) for k, values in samples.items()}
    # Capture records but does not execute kernels. Count the one eager launch,
    # three warmup replays, and measured replays of each k specialization.
    steps = (1 + (3 + plan.repeat) * plan.launches) * sum(plan.ks)
    final = out[::LINE_WORDS].cpu().tolist()
    for cta, index in enumerate(final):
        order = cycle_order(plan.lines_per_cta, args.seed + cta)
        expected = (cta * plan.lines_per_cta + order[steps % plan.lines_per_cta]) * LINE_WORDS
        if index != expected:
            raise RuntimeError("pointer chase final index differs from the expected permutation")
    return {
        **asdict(plan),
        "median_ns": {str(k): medians[k] for k in plan.ks},
        "samples_ns": {str(k): values for k, values in samples.items()},
        "launch_floor_ns": medians[0],
        **fit_latency(medians),
        "final_indices": final,
        "correct": True,
        "cache_policy": "ld.global.cg" if plan.mode != "atomic" else "atomic L2",
        "cache_condition": "evict before each replay" if eviction is not None else "warm nodes",
        "dram_misses_verified": False,
        "timing": args.timer,
        "timing_unit": "ns per launch; one graph replay per sample",
        "active_lanes_per_cta": 1,
        "threads_per_cta": 32,
        "atomic_addend": 0 if plan.mode == "atomic" else None,
        "seed": args.seed,
    }


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
        if not torch.cuda.is_available():
            emit({"type": "error", "reason": "CUDA is required for timing"})
            return 1
        devices = [args.device] if args.device is not None else range(torch.cuda.device_count())
        for ordinal in devices:
            with torch.cuda.device(ordinal):
                properties = torch.cuda.get_device_properties(ordinal)
                arch = f"sm_{properties.major}{properties.minor}a"
                if arch not in ARCHES:
                    raise ValueError("supported devices are sm_120 and sm_121")
                l2_bytes = device_l2_bytes(properties, ordinal)
                plans = [
                    make_plan(args, mode, properties.multi_processor_count, l2_bytes)
                    for mode in args.modes
                ]
                affinity = pin_x925()
                for plan in plans:
                    with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
                        record = measure_mode(args, plan, ordinal, arch)
                    emit(
                        {
                            "type": "dependent_rt",
                            "device": properties.name,
                            "device_ordinal": ordinal,
                            "sms": properties.multi_processor_count,
                            "l2_bytes": l2_bytes,
                            "arch": arch,
                            "affinity": affinity,
                            **record,
                        }
                    )
        return 0
    except Exception as error:  # noqa: BLE001 - emit a receipt without environment details.
        # Exception strings can contain environment details; emit only the type.
        emit({"type": "error", "reason": "benchmark failed", "error_type": type(error).__name__})
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
