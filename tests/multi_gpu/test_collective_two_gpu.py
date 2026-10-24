# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired qualification in one spawned process pair for the complete sweep."""

import hashlib
import queue as std_queue
import socket
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ctikernels.collective.inference import (
    AllReduceAddRMSNorm,
    exact_reference,
    market_composition,
    measured_floor,
    reference,
)

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 32, 64, 128, 189, 256, 512, 1024, 2048, 4096, 8192)
# Weighted-norm (gain w: Qwen3, the DFlash drafts) rows: decode, the flag/producer-consumer
# boundary and prefill chunks, plus a sliced call past max_tokens.
WEIGHTED_COUNTS = (1, 2, 4, 8, 16, 78, 189, 797, 3199, 8192, 9000)


def _inputs(rank, tokens, hidden, seed, comm, zero_centered=True):
    """Inputs plus the FP32 rank-order reference, the FP64 oracle and the market
    composition (NCCL all-reduce, BF16 residual add, RMSNorm) on the same inputs."""
    partial = torch.randn(
        tokens, hidden, generator=torch.Generator().manual_seed(seed + rank)
    ).bfloat16()
    parts = [torch.empty_like(partial) for _ in range(2)]
    dist.all_gather(parts, partial)
    generator = torch.Generator().manual_seed(seed + 100)
    residual = torch.randn(tokens, hidden, generator=generator).bfloat16()
    weight = torch.randn(hidden, generator=generator).bfloat16()
    norm = {"zero_centered": zero_centered}
    exact = exact_reference(torch.stack(parts), residual, weight, 1e-6, **norm)
    expected = (*reference(torch.stack(parts), residual, weight, 1e-6, **norm), exact)
    partial, residual, weight = partial.cuda(), residual.cuda(), weight.cuda()
    summed = partial.clone()
    comm.all_reduce(summed, "sum")
    market = market_composition(summed, residual, weight, 1e-6, **norm)
    return partial, residual, weight, (*expected, tuple(t.cpu() for t in market))


def _like(tensor):
    return {"device": tensor.device, "dtype": tensor.dtype}


def _ordered_bits(tensor):
    bits = tensor.contiguous().view(torch.int16).int()
    return torch.where(bits < 0, -32768 - bits, bits)


def _check(actual, expected, *, label, strict=True):
    x, r = (t.cpu() for t in actual)
    xo, ro, exact, market = expected
    digest = hashlib.sha256(
        x.view(torch.uint8).numpy().tobytes() + r.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    hashes = [None, None]
    dist.all_gather_object(hashes, digest)
    ulp = (_ordered_bits(x) - _ordered_bits(xo)).abs()
    unequal = int((x.view(torch.int16) != xo.view(torch.int16)).sum())
    stats = {
        "case": label,
        "max_ulp": int(ulp.max()),
        "nonidentical": unequal,
        "elements": x.numel(),
    }
    errors = []
    if hashes[0] != hashes[1]:
        errors.append(f"{label}: ranks differ")
    if strict:
        if not torch.equal(r.view(torch.int16), ro.view(torch.int16)):
            errors.append(f"{label}: residual is not bitwise equal")
        # Measured floor: error to the FP64 oracle at most the market composition's.
        failures, floor = measured_floor((x, r), market, exact)
        stats.update(floor)
        errors.extend(f"{label}: {failure}" for failure in failures)
    else:
        # NCCL rounds partial0 + partial1 before adding residual: cancellation
        # makes a relative-only tolerance unsuitable near zero.
        for value, oracle in zip((x, r), (xo, ro), strict=True):
            if not torch.allclose(value.float(), oracle.float(), rtol=0.02, atol=0.04):
                errors.append(f"{label}: NCCL rounding tolerance exceeded")
    return errors, stats


def _worker(rank, port, queue):
    errors, stats = [], []
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(minutes=20),
        )

        from ctikernels import init_pynccl

        # A 256-byte window (NCCL rejects an empty one): every tested tensor is larger, so
        # each all-reduce runs in place on the caller's tensor without staging copies.
        comm = init_pynccl(
            tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=256
        )

        def make(hidden, max_tokens=8192, **kwargs):
            return AllReduceAddRMSNorm(
                rank=rank,
                world_size=2,
                cpu_group=dist.group.WORLD,
                device=f"cuda:{rank}",
                hidden=hidden,
                max_tokens=max_tokens,
                eps=1e-6,
                **kwargs,
            )

        def check(actual, expected, label, strict=True):
            failures, measurement = _check(actual, expected, label=label, strict=strict)
            errors.extend(failures)
            stats.append(measurement)

        for hidden in (2048, 4096, 5120):
            op = make(hidden, provider="cute")
            for tokens in COUNTS:
                x, r, w, expected = _inputs(rank, tokens, hidden, hidden + tokens * 17, comm)
                check(op(x, r, w), expected, f"H={hidden}, T={tokens}")
            dist.barrier()
            op.close()
            op.close()
            try:
                op(x, r, w)
            except RuntimeError:
                pass
            else:
                errors.append("call after close did not raise")

        for hidden in (2048, 4096, 5120):
            op = make(hidden, provider="cute", zero_centered=False)
            for tokens in WEIGHTED_COUNTS:
                seed = 7 * hidden + tokens
                x, r, w, expected = _inputs(rank, tokens, hidden, seed, comm, False)
                check(op(x, r, w), expected, f"weighted H={hidden}, T={tokens}")
            # Graph replay of a weighted exit on a non-default stream.
            x, r, w, expected = _inputs(rank, 2, hidden, 5 * hidden, comm, False)
            xi, ri = x.clone(), r.clone()
            op.prepare([2])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                op(x, r, w)
            for replay in range(3):
                x.copy_(xi)
                r.copy_(ri)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                check((x, r), expected, f"weighted graph H={hidden}, replay={replay}")
            dist.barrier()
            op.close()

        op = make(2048, max_tokens=64, provider="cute")
        for layout in ("sliced", "row_stride"):
            x, r, w, expected = _inputs(rank, 200, 2048, 97, comm)
            if layout == "row_stride":
                view = torch.empty(200, 2056, device=x.device, dtype=x.dtype)[:, :2048]
                view.copy_(x)
                x = view
            check(op(x, r, w), expected, layout)
        # SM path only: layouts that would need staging copies are rejected.
        x, r, w, _ = _inputs(rank, 8, 2048, 98, comm)
        rejected = {
            "misaligned residual": (x, torch.empty(r.numel() + 1, **_like(r))[1:].view_as(r), w),
            "inner stride": (torch.empty(8, 4096, **_like(x))[:, ::2], r, w),
            "odd row stride": (torch.empty(8, 2052, **_like(x))[:, :2048], r, w),
            "misaligned weight": (x, r, torch.empty(2049, **_like(w))[1:]),
        }
        for label, tensors in rejected.items():
            try:
                op(*tensors)
            except ValueError:
                pass
            else:
                errors.append(f"{label} was not rejected")
        dist.barrier()
        op.close()

        op = make(2048, provider="cute")
        counts = [1, 4, 8, 64]
        op.prepare(counts)
        buffers = [_inputs(rank, t, 2048, 181 + t, comm) for t in counts]
        originals = [(x.clone(), r.clone()) for x, r, _, _ in buffers]
        eager = []
        for x, r, w, expected in buffers:
            op(x, r, w)
            eager.append((x.clone(), r.clone()))
            check((x, r), expected, f"graph eager T={x.shape[0]}")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for x, r, w, _ in buffers:
                op(x, r, w)
        for replay in range(3):
            for (x, r, _, _), (xi, ri) in zip(buffers, originals, strict=True):
                x.copy_(xi)
                r.copy_(ri)
            graph.replay()
            for (x, r, _, expected), (xe, re) in zip(buffers, eager, strict=True):
                check((x, r), expected, f"replay={replay}, T={x.shape[0]}")
                if not torch.equal(x, xe) or not torch.equal(r, re):
                    errors.append("graph/eager bits differ")
            x, r, w, expected = _inputs(rank, 4, 2048, 333 + replay, comm)
            check(op(x, r, w), expected, f"interleaved eager {replay}")
        dist.barrier()
        op.close()

        # A graph also covers sliced exchanges with a padded row stride.
        op = make(2048, max_tokens=64, provider="cute")
        op.prepare([200])
        x, r, w, expected = _inputs(rank, 200, 2048, 813, comm)
        rv = torch.empty(200, 2056, **_like(r))[:, :2048]
        rv.copy_(r)
        xi, ri = x.clone(), r.clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, rv, w)
        for replay in range(2):
            x.copy_(xi)
            rv.copy_(ri)
            graph.replay()
            check((x, rv), expected, f"sliced strided graph {replay}")
        dist.barrier()
        op.close()

        with make(2048, provider="nccl", communicator=comm) as op:
            x, r, w, expected = _inputs(rank, 64, 2048, 113, comm)
            check(op(x, r, w), expected, "nccl", strict=False)
        dist.barrier()
    except Exception:  # noqa: BLE001 -- send worker failures to the parent without terminating peers.
        errors.append(traceback.format_exc())
    finally:
        queue.put((rank, errors, stats))
        if dist.is_initialized():
            dist.destroy_process_group()


def _collect(queue, processes):
    """Gather one result per worker; fail instead of blocking forever if a worker dies first."""
    results = []
    while len(results) < len(processes):
        try:
            results.append(queue.get(timeout=5))
        except std_queue.Empty:
            dead = [
                (rank, p.exitcode)
                for rank, p in enumerate(processes)
                if p.exitcode not in (None, 0)
            ]
            if dead:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                    process.join()
                pytest.fail(f"worker exited before reporting (rank, exitcode): {dead}")
    return results


def test_two_gpu_qualification():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(rank, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = _collect(queue, processes)
    for process in processes:
        process.join()
        assert process.exitcode == 0
    for rank, errors, stats in sorted(results):
        for measurement in stats:
            print({"rank": rank, **measurement})
        assert not errors, "\n".join(errors)
