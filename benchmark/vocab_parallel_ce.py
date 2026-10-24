# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# torch_materialized is an independently written materialized reference to
# Megatron-style vocabulary-parallel CE; no third-party source copied.
"""CUDA-event comparison of vocabulary shards, TP1-local and real fused CE bars.

Run as ``python -m benchmark.vocab_parallel_ce --mode single`` on a GPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import queue as std_queue
import statistics
import sys
import tempfile
import time
import traceback
import types
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.vocab_parallel import SHARD_CHUNK_SIZE, merge_vocab_shard_stats

DEFAULT_SHAPES = [(2048, 4096, 151936), (8192, 4096, 151936), (4096, 5120, 248320)]

PATHS = (
    "tp1_local",
    "shard",
    "shard_saved",
    "shard_fp32",
    "torch_materialized",
    "megatron_core",
    "te_parallel_ce",
)
FUSED_PATHS = {"megatron_core", "te_parallel_ce"}


def _load_te(directory):
    """Load TE's Python/Triton CE without executing C-extension package initializers."""
    if directory is None:
        raise ImportError("--te-path must point to an unpacked Transformer Engine wheel")
    root = Path(directory).resolve() / "transformer_engine"
    if not (root / "pytorch" / "cross_entropy.py").is_file():
        raise ImportError("--te-path does not contain transformer_engine/pytorch/cross_entropy.py")
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.triton",
        "transformer_engine.common",
        "transformer_engine.common.triton",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")[1:]))]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, package)
    jit = types.ModuleType("transformer_engine.pytorch.jit")

    def no_torch_dynamo():
        return torch._dynamo.disable

    jit.no_torch_dynamo = no_torch_dynamo
    sys.modules[jit.__name__] = jit
    sys.modules["transformer_engine.pytorch"].jit = jit
    return importlib.import_module(
        "transformer_engine.pytorch.cross_entropy"
    ).parallel_cross_entropy


def _load_bars(args, comm):
    loaded, errors = {}, {}
    if args.mcore_path is not None:
        sys.path.insert(0, str(Path(args.mcore_path).resolve()))
    # Import Megatron before installing TE's deliberately minimal namespace.
    for path in ("megatron_core", "te_parallel_ce"):
        if path not in args.paths:
            continue
        try:
            if path == "megatron_core":
                loaded[path] = importlib.import_module(
                    "megatron.core.fusions.fused_cross_entropy"
                ).fused_vocab_parallel_cross_entropy
            else:
                loaded[path] = _load_te(args.te_path)
        except Exception as error:  # noqa: BLE001 -- optional dependency diagnostics belong in rows.
            errors[path] = f"{type(error).__name__}: {error}"
    if comm is not None:
        # Both ranks must skip a missing dependency before entering its collectives.
        peer_errors = [None] * dist.get_world_size()
        dist.all_gather_object(peer_errors, errors)
        for peer in peer_errors:
            errors.update(peer)
    return loaded, errors


def _mm(a, b):
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a, b)


def _measure(run, reps, use_graph, comm=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = None
    retained = None
    captured = False
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                retained = run()
            captured = True
        except RuntimeError:
            # Report eager fallback in the JSON row; never pass an eager timing
            # off as graph replay. Peers must agree on the timing mode.
            captured = False
        if comm is not None:
            flag = torch.tensor(int(captured), dtype=torch.int64)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            captured = bool(flag)
    invoke = graph.replay if captured else run
    timings = []
    with torch.cuda.stream(stream):
        for _ in range(reps):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            invoke()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end))
    if graph is not None:
        del retained, graph
    return statistics.median(timings), captured


def _measure_breakdown(run, reps):
    timings = {name: [] for name in ("forward", "merge", "finalize", "backward")}
    for iteration in range(reps + 3):
        events = [torch.cuda.Event(enable_timing=True)]
        names = []
        events[0].record()

        def mark(name, names=names, events=events):
            names.append(name)
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()

        retained = run(mark=mark)
        events[-1].synchronize()
        if iteration >= 3:
            for i, name in enumerate(names):
                timings[name].append(events[i].elapsed_time(events[i + 1]))
        del retained
    return {name: statistics.median(values) for name, values in timings.items()}


def _run_case(args, op, shape, kind, head_grad, rank, comm, tp_group, bars, bar_errors):
    n, h, v = shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    start, end = (0, (v + 1) // 2) if rank == 0 else ((v + 1) // 2, v)
    local_v = end - start
    torch.manual_seed(83)
    x = torch.randn(n, h, device="cuda", dtype=dtype, requires_grad=True)
    torch.manual_seed(83 + rank)
    w = torch.randn(local_v, h, device="cuda", dtype=dtype, requires_grad=head_grad)
    y = torch.arange(n, device="cuda", dtype=torch.int64) % v
    local_y = y.remainder(local_v)
    gathered = torch.empty(2, n, 3, device="cuda", dtype=torch.float32)
    variables = (x, w) if head_grad else (x,)
    upstream = torch.ones(() if kind == "ce" else (n,), device="cuda")

    def merge(stats):
        if comm is None:
            gathered[0].copy_(stats)
            gathered[1].copy_(stats)
            gathered[1, :, 2].zero_()
        else:
            comm.all_gather(gathered.view(-1), stats.view(-1))
        return merge_vocab_shard_stats(gathered)

    def reduce_dx(dx):
        dx = dx.float()
        if comm is not None:
            comm.all_reduce(dx, "sum")
        return dx

    def tp1(mark=lambda _: None):
        output = op.forward(x, w, local_y)
        mark("forward")
        mark("merge")
        mark("finalize")
        grads = torch.autograd.grad(output, variables, upstream)
        result = (output, reduce_dx(grads[0]), *grads[1:])
        mark("backward")
        return result

    def shard(mark=lambda _: None, *, save_probabilities=False, fp32_logits=False):
        part = op.shard_partials(
            x,
            w,
            y,
            vocab_start=start,
            vocab_size=v,
            save_probabilities=save_probabilities,
            fp32_logits=fp32_logits,
            chunk_size=args.shard_chunk_size,
            vocab_block_size=args.vocab_block_size,
        )
        mark("forward")
        merged = merge(part.stats)
        mark("merge")
        output = op.shard_finalize(part, merged)
        mark("finalize")
        grads = torch.autograd.grad(output, variables, upstream)
        result = (output, reduce_dx(grads[0]), *grads[1:])
        mark("backward")
        return result

    @torch.no_grad()
    def materialized(mark=lambda _: None):
        logits = _mm(x, w.t())
        maximum = logits.max(1).values
        exponentials = (logits - maximum[:, None]).exp_()
        targets = y - start
        owned = (targets >= 0) & (targets < local_v) & (y != -100)
        safe = targets.clamp(0, local_v - 1)
        selected = torch.where(owned, logits.gather(1, safe[:, None])[:, 0], 0.0)
        stats = torch.stack((maximum, exponentials.sum(1), selected), 1)
        mark("forward")
        merged = merge(stats)
        mark("merge")
        losses = torch.where(y != -100, merged[:, 0] - merged[:, 1], 0.0)
        output = losses.sum() if kind == "ce" else -losses
        mark("finalize")
        probabilities = (logits - merged[:, 0, None]).exp_()
        probabilities.scatter_add_(1, safe[:, None], -owned.float()[:, None])
        probabilities.masked_fill_((y == -100)[:, None], 0.0)
        if kind == "logprob":
            probabilities.neg_()
        dz = probabilities.to(dtype)
        dx = reduce_dx(_mm(dz, w).to(dtype))
        dw = _mm(dz.t(), x).to(dtype) if head_grad else None
        mark("backward")
        return output, dx, dw

    def fused(path, mark=lambda _: None):
        logits = F.linear(x, w)
        targets = local_y if comm is None else y
        if path == "megatron_core":
            # [sequence, batch, V_local]; global ids in pair mode, local ids in single.
            losses = bars[path](logits[:, None, :], targets[:, None], tp_group).reshape(n)
        else:
            # [batch, sequence, V_local]; this is TE's real Triton autograd operator.
            losses = bars[path](
                logits[None, :, :],
                targets[None, :],
                label_smoothing=0.0,
                reduce_loss=False,
                dist_process_group=tp_group if comm is not None else None,
                ignore_idx=-100,
            ).reshape(n)
        mark("forward")  # Includes the fused CE's internal MAX/SUM collectives.
        mark("merge")
        output = losses.sum() if kind == "ce" else -losses
        mark("finalize")
        grads = torch.autograd.grad(output, variables, upstream)
        result = (output, reduce_dx(grads[0]), *grads[1:])
        mark("backward")
        return result

    paths = [
        ("tp1_local", tp1),
        ("shard", lambda **kw: shard(save_probabilities=False, **kw)),
        ("shard_saved", lambda **kw: shard(save_probabilities=True, **kw)),
        ("shard_fp32", lambda **kw: shard(fp32_logits=True, **kw)),
        ("torch_materialized", materialized),
        ("megatron_core", lambda **kw: fused("megatron_core", **kw)),
        ("te_parallel_ce", lambda **kw: fused("te_parallel_ce", **kw)),
    ]
    rows = []
    for path, run in paths:
        if path not in args.paths:
            continue
        row = {
            "mode": args.mode,
            "rank": rank,
            "shape": list(shape),
            "tp": 2,
            "dtype": args.dtype,
            "operation": kind,
            "temperature": 1.0,
            "head_grad": head_grad,
            "path": path,
            "graph": not args.eager,
            "chunk_size": args.chunk_size,
            "shard_chunk_size": SHARD_CHUNK_SIZE
            if args.shard_chunk_size is None
            else args.shard_chunk_size,
            "vocab_block_size": args.vocab_block_size,
            "saved_buffer_bytes_per_row": local_v
            * (4 if path == "shard_fp32" and head_grad else x.element_size()),
            "saved_buffer_bytes": n
            * local_v
            * (4 if path == "shard_fp32" and head_grad else x.element_size()),
        }
        notes = []
        if path == "megatron_core":
            notes.append("Megatron-Core fused CE; targets contain no ignore_index (-100)")
        elif path == "te_parallel_ce":
            notes.append("TE loaded without its C extension; parallel_cross_entropy uses Triton")
        if comm is None:
            notes.append("single mode: no collective traffic; fused bars use local target ids")
        # Include graph pools and reference-provider FP32 temporaries conservatively.
        estimate = 4 * (x.numel() + w.numel()) * 4 + n * local_v * 32
        free, _ = torch.cuda.mem_get_info()
        fits = estimate < min(40 * 1024**3, free * 0.85)
        if comm is not None:
            fit_flag = torch.tensor(int(fits), dtype=torch.int64)
            dist.all_reduce(fit_flag, op=dist.ReduceOp.MIN)
            fits = bool(fit_flag)
        if path in bar_errors:
            row.update(median_ms=None, note=bar_errors[path])
        elif path in FUSED_PATHS and comm is not None and v % 2:
            row.update(
                median_ms=None, note="fused CE requires equal vocabulary shards in pair mode"
            )
        elif path in {"torch_materialized", "shard_saved", *FUSED_PATHS} and not fits:
            row.update(median_ms=None, note="full-shard estimate exceeds memory budget")
        elif path == "tp1_local" and kind == "ce" and op._cross_entropy_kernel is None:
            row.update(median_ms=None, note="liger-kernel unavailable")
        else:
            if comm is not None:
                dist.barrier()
            row["median_ms"], row["graph"] = _measure(run, args.reps, not args.eager, comm)
            if not args.eager and not row["graph"]:
                row["note"] = "capture unavailable; timed eagerly"
            if args.breakdown:
                row["eager_breakdown_ms"] = _measure_breakdown(run, args.reps)
        if "note" in row:
            notes.append(row["note"])
        if notes:
            row["note"] = "; ".join(notes)
        rows.append(row)
    times = {row["path"]: row["median_ms"] for row in rows}
    available_bars = [
        times[path]
        for path in ("tp1_local", "megatron_core", "te_parallel_ce")
        if times.get(path) is not None
    ]
    best_bar = min(available_bars) if available_bars else None
    for row in rows:
        own = row["median_ms"] if row["path"].startswith("shard") else times.get("shard")
        for label, path in (
            ("tp1_local", "tp1_local"),
            ("torch_materialized", "torch_materialized"),
            ("megatron_core", "megatron_core"),
            ("te", "te_parallel_ce"),
        ):
            denominator = times.get(path)
            row[f"ours_over_{label}"] = (
                own / denominator if own is not None and denominator else None
            )
        row["ours_over_best_bar"] = own / best_bar if own is not None and best_bar else None
        if comm is None:
            print(json.dumps(row), flush=True)
    return rows


def _run_shapes(args, rank=0, comm=None, tp_group=None):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    bars, bar_errors = _load_bars(args, comm)
    records = []
    for kind in args.operations:
        op = (
            FusedLinearCrossEntropyLoss(chunk_size=args.chunk_size, provider=args.provider)
            if kind == "ce"
            else ChunkedLinearSelectedTokenLogprobs(
                chunk_size=args.chunk_size, provider=args.provider
            )
        )
        for shape in args.shapes:
            for head_grad in (False, True) if kind == "ce" else (False,):
                records.extend(
                    _run_case(
                        args, op, shape, kind, head_grad, rank, comm, tp_group, bars, bar_errors
                    )
                )
    return records


def _pair_worker(rank, rendezvous, args, queue):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(minutes=10)
        )
        tp_group = dist.new_group(backend="nccl") if FUSED_PATHS.intersection(args.paths) else None
        from ctikernels import init_pynccl

        comm = init_pynccl(tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD)
        queue.put((rank, _run_shapes(args, rank, comm, tp_group), None))
    except Exception:  # noqa: BLE001 -- propagate worker diagnostics without ending peer processes.
        queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _shapes(value):
    try:
        shapes = [
            tuple(map(int, shape.lower().replace("x", ",").split(",")))
            for shape in value.split(";")
        ]
        if any(len(shape) != 3 or min(shape) < 1 or shape[2] < 2 for shape in shapes):
            raise ValueError
        return shapes
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use N,H,V;N,H,V with positive dimensions and V >= 2"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "pair"), default="single")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument("--mcore-path", type=Path, help="Megatron-Core 0.19.2 import root")
    parser.add_argument(
        "--te-path", type=Path, help="unpacked Transformer Engine 2.20.2 wheel root"
    )
    parser.add_argument("--shapes", type=_shapes, default=DEFAULT_SHAPES)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--shard-chunk-size", type=int, default=None)
    parser.add_argument("--vocab-block-size", type=int, default=None)
    parser.add_argument(
        "--operations", nargs="+", choices=("ce", "logprob"), default=["ce", "logprob"]
    )
    parser.add_argument(
        "--breakdown", action="store_true", help="also time eager forward/merge/finalize/backward"
    )
    parser.add_argument("--provider", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument(
        "--eager", action="store_true", help="time eager calls instead of graph replay"
    )
    args = parser.parse_args()
    if any(
        value is not None and value <= 0
        for value in (args.chunk_size, args.shard_chunk_size, args.vocab_block_size)
    ):
        parser.error("chunk and vocabulary block sizes must be positive")
    if args.reps < 10:
        parser.error("--reps must be at least 10")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    if args.mode == "single":
        if FUSED_PATHS.intersection(args.paths):
            torch.cuda.set_device(0)
            with tempfile.TemporaryDirectory() as directory:
                try:
                    dist.init_process_group(
                        "nccl",
                        rank=0,
                        world_size=1,
                        init_method=Path(directory, "rendezvous").as_uri(),
                        timeout=timedelta(minutes=10),
                    )
                    _run_shapes(args, tp_group=dist.group.WORLD)
                finally:
                    if dist.is_initialized():
                        dist.destroy_process_group()
        else:
            _run_shapes(args)
        return
    if torch.cuda.device_count() < 2:
        parser.error("--mode pair requires two GPUs")
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = Path(directory, "rendezvous").as_uri()
        processes = [
            context.Process(target=_pair_worker, args=(rank, rendezvous, args, queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results = []
        deadline = time.monotonic() + 1800
        while len(results) < 2:
            try:
                result = queue.get(timeout=1)
                if result[2] is not None:
                    raise RuntimeError(result[2])
                results.append(result)
            except std_queue.Empty:
                if time.monotonic() > deadline or any(
                    p.exitcode not in (None, 0) for p in processes
                ):
                    raise RuntimeError("benchmark worker failed to report") from None
        for process in processes:
            process.join(timeout=5)
            if process.exitcode != 0:
                raise RuntimeError("benchmark worker did not exit successfully")
        for _, rows, _ in sorted(results):
            for row in rows:
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
