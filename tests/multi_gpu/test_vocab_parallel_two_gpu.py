# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Real FP32 collectives and training graph replay, with fail-fast worker reports."""

import queue as std_queue
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from .._vocab_parallel_gates import (
    _check_results,
    _check_stats,
    _finish,
)

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")


def _reference(x, w, b, y, kind, train_head, tp1, chunk_size=29, output_only=False):
    from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
    from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs

    # Avoid CE's precomputed gradients for output-only diagnostic runs.
    x = x.detach().clone().requires_grad_(not output_only)
    w = w.detach().clone().requires_grad_(train_head and not output_only)
    b = b.detach().clone().requires_grad_(train_head and not output_only)
    if tp1 and kind == "ce":
        output = FusedLinearCrossEntropyLoss(chunk_size=chunk_size).forward(x, w, y, bias=b)
    elif tp1:
        output = ChunkedLinearSelectedTokenLogprobs(chunk_size=chunk_size).forward(
            x, w, y.clamp_min(0), bias=b, temperature=0.7
        )
        output = torch.where(y != -100, output, 0.0)
    else:
        loss = F.cross_entropy(
            F.linear(x, w, b) / (0.7 if kind == "logprob" else 1.0), y, reduction="none"
        )
        output = loss.sum() if kind == "ce" else -loss
    if output_only:
        return output.detach()
    gradients = torch.autograd.grad(output.sum(), (x, w, b) if train_head else (x,))
    return (output, *gradients)


def _worker(rank, rendezvous, queue, fp32_logits, save_probabilities, kind, train_head):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=90)
        )
        from ctikernels import init_pynccl
        from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
        from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
        from ctikernels.vocab_parallel import (
            merge_vocab_shard_stats,
            vocab_shard_merged_from_sums,
            vocab_shard_sum_terms,
        )

        comm = init_pynccl(tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD)
        measurements, all_checks = [], []
        for n, h, v in [(1, 256, 1000), (37, 1024, 32003), (512, 256, 1000)]:
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                torch.manual_seed(121)
                x_data = (torch.randn(n, h, device="cuda") / 4).to(dtype)
                full_w = (torch.randn(v, h, device="cuda") / 4).to(dtype)
                full_b = (torch.randn(v, device="cuda") / 4).to(dtype)
                split = (v + 1) // 2
                start, end = (0, split) if rank == 0 else (split, v)
                y = (torch.arange(n, device="cuda", dtype=torch.int64) * 79) % v
                if n > 1:
                    y[:4] = torch.tensor([-100, split - 1, split, v - 1], device="cuda")
                x = x_data.clone().requires_grad_()
                w = full_w[start:end].clone().requires_grad_(train_head)
                b = full_b[start:end].clone().requires_grad_(train_head)
                op = (
                    FusedLinearCrossEntropyLoss(chunk_size=29, shard_chunk_size=29, provider="auto")
                    if kind == "ce"
                    else ChunkedLinearSelectedTokenLogprobs(chunk_size=29, shard_chunk_size=29)
                )
                kwargs = {"temperature": 0.7} if kind == "logprob" else {}
                kwargs.update(fp32_logits=fp32_logits, save_probabilities=save_probabilities)

                def run(
                    route,
                    op=op,
                    x=x,
                    w=w,
                    y=y,
                    start=start,
                    b=b,
                    kwargs=kwargs,
                    n=n,
                    v=v,
                    train_head=train_head,
                    stats_out=None,
                ):
                    part = op.shard_partials(
                        x, w, y, vocab_start=start, vocab_size=v, bias=b, **kwargs
                    )
                    if stats_out is not None:
                        stats_out.append(part)
                    if route == "gather":
                        gathered = torch.empty((2, n, 3), dtype=torch.float32, device="cuda")
                        comm.all_gather(gathered.view(-1), part.stats.view(-1))
                        merged = merge_vocab_shard_stats(gathered)
                    else:
                        maximum = part.stats[:, 0].clone()
                        comm.all_reduce(maximum, "max")
                        sums = vocab_shard_sum_terms(part.stats, maximum)
                        comm.all_reduce(sums, "sum")
                        merged = vocab_shard_merged_from_sums(maximum, sums)
                    output = op.shard_finalize(part, merged)
                    gradients = torch.autograd.grad(output.sum(), (x, w, b) if train_head else (x,))
                    dx = gradients[0].float()
                    comm.all_reduce(dx, "sum")
                    return (output, dx, *gradients[1:])

                oracle = _reference(
                    x_data.double(),
                    full_w.double(),
                    full_b.double(),
                    y,
                    kind,
                    train_head,
                    False,
                )
                baseline = _reference(x_data, full_w, full_b, y, kind, train_head, True)
                if fp32_logits:
                    # Only the output uses FP32 TP1; keep head-dtype gradient baselines.
                    output = _reference(
                        x_data.float(),
                        full_w.float(),
                        full_b.float(),
                        y,
                        kind,
                        train_head,
                        True,
                        output_only=True,
                    )
                    baseline = (output, *baseline[1:])
                routes = []
                for route in ("gather", "reduce"):
                    case = (
                        n,
                        h,
                        v,
                        str(dtype),
                        kind,
                        train_head,
                        route,
                        fp32_logits,
                        save_probabilities,
                    )
                    print(f"rank {rank}: {case}", flush=True)
                    checks = []
                    stats_out = []
                    got = run(route, stats_out=stats_out)
                    if kind == "logprob":
                        assert torch.count_nonzero(got[0][y == -100]) == 0
                    assert torch.count_nonzero(got[1][y == -100]) == 0
                    _check_stats(
                        stats_out[0],
                        x,
                        w,
                        b,
                        y,
                        start,
                        0.7 if kind == "logprob" else 1.0,
                        fp32_logits,
                        checks=checks,
                        tp1=(full_w, full_b, 29, kind),
                    )
                    local_baseline = (*baseline[:2], *(g[start:end] for g in baseline[2:]))
                    local_oracle = (*oracle[:2], *(g[start:end] for g in oracle[2:]))
                    _check_results(
                        got,
                        local_baseline,
                        local_oracle,
                        dtype,
                        checks=checks,
                        fp32_logits=fp32_logits,
                    )
                    prefix = f"rank {rank} {case} "
                    all_checks.extend(
                        dict(check, label=prefix + check["label"]) for check in checks
                    )
                    measurements.append(case)
                    routes.append(got[0])
                torch.testing.assert_close(routes[0], routes[1], rtol=1e-5, atol=1e-5)
                if n == 37 and dtype == torch.bfloat16 and kind == "ce" and not train_head:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        # A fresh leaf first used on the capture stream: the routes
                        # above bound x's AccumulateGrad node to the default stream.
                        xc = x.detach().clone().requires_grad_(True)
                        for _ in range(3):
                            run("gather", x=xc)
                    stream.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = run("gather", x=xc)
                    for replay in range(2):
                        with torch.no_grad():
                            xc.copy_(x_data * (replay + 2))
                            y.copy_((torch.arange(n, device="cuda") * 53 + replay) % v)
                            y[0] = -100
                        graph.replay()
                        eager = run("gather", x=xc)
                        torch.cuda.synchronize()
                        for got, expected in zip(captured, eager, strict=True):
                            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
                    del graph
        torch.cuda.synchronize()
        queue.put((rank, None, measurements, all_checks))
    except Exception:  # noqa: BLE001 -- report failures without terminating peers.
        queue.put((rank, traceback.format_exc(), [], []))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize("kind,train_head", [("ce", False), ("ce", True), ("logprob", False)])
def test_two_gpu_vocab_parallel(fp32_logits, save_probabilities, kind, train_head):
    if kind == "ce":
        from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss

        if FusedLinearCrossEntropyLoss(provider="auto")._cross_entropy_kernel is None:
            pytest.skip("TP1 CE requires liger-kernel")
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = Path(directory, "rendezvous").as_uri()
        processes = [
            context.Process(
                target=_worker,
                args=(rank, rendezvous, queue, fp32_logits, save_probabilities, kind, train_head),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 115
        results = []
        while len(results) < 2:
            try:
                result = queue.get(timeout=1)
                results.append(result)
                assert result[1] is None, result[1]
            except std_queue.Empty:
                assert time.monotonic() < deadline, "two-GPU qualification exceeded 115 seconds"
                assert all(p.exitcode in (None, 0) for p in processes), (
                    "worker exited before reporting"
                )
        for process in processes:
            process.join(timeout=2)
            assert process.exitcode == 0
        checks = []
        for rank, _, measurements, rank_checks in sorted(results):
            print(f"rank {rank}: {measurements}")
            checks.extend(rank_checks)
        _finish(checks)


def _range_worker(rank, rendezvous, queue):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=90)
        )
        from ctikernels import init_pynccl
        from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
        from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
        from ctikernels.vocab_parallel import vocab_shard_merged_from_sums, vocab_shard_sum_terms

        from ..gpu.test_vocab_parallel import _range_check_results, _range_reference

        comm = init_pynccl(tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD)
        cases = [
            ("ce", reduction, train)
            for reduction in ("sum", "mean", "none")
            for train in (False, True)
        ]
        cases.extend([("logprob", "scalar", False), ("logprob", "rows", False)])
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for kind, mode, train_head in cases:
                torch.manual_seed(151)
                n, h, v = 65, 256, 1003
                x_data = (torch.randn(n, h, device="cuda") / 4).to(dtype)
                full_w = (torch.randn(v, h, device="cuda") / 4).to(dtype)
                full_b = (torch.randn(v, device="cuda") / 4).to(dtype)
                start, end = (0, 502) if rank == 0 else (502, v)
                x = torch.empty(n, h + 17, device="cuda", dtype=dtype)[:, :h]
                x.copy_(x_data).requires_grad_()
                w = full_w[start:end].clone().requires_grad_(train_head)
                b = full_b[start:end].clone().requires_grad_(train_head)
                y = torch.arange(n * 2, device="cuda", dtype=torch.int64)[::2]
                y.mul_(79).remainder_(v)
                y[:4] = torch.tensor([-7, 501, 502, 1002], device="cuda")
                t = torch.linspace(0.5, 1.5, n * 2, device="cuda")[::2] if mode == "rows" else 0.7
                reduction = mode if kind == "ce" else "sum"
                g = (
                    torch.randn(n * 2, device="cuda")[::2]
                    if kind == "logprob" or reduction == "none"
                    else torch.tensor(0.37, device="cuda")
                )
                op = (
                    FusedLinearCrossEntropyLoss(
                        provider="auto", reduction=reduction, ignore_index=-7
                    )
                    if kind == "ce"
                    else ChunkedLinearSelectedTokenLogprobs(ignore_index=-7)
                )
                kwargs = {"temperature": t} if kind == "logprob" else {}
                part = op.shard_partials(
                    x,
                    w,
                    y,
                    bias=b,
                    vocab_start=start,
                    vocab_size=v,
                    save_probabilities=mode in ("none", "rows"),
                    **kwargs,
                )
                maximum = part.stats[:, 0].clone()
                comm.all_reduce(maximum, "max")
                sums = vocab_shard_sum_terms(part.stats, maximum)
                comm.all_reduce(sums, "sum")
                merged = vocab_shard_merged_from_sums(maximum, sums)
                output = op.shard_finalize(part, merged)
                gradients = torch.autograd.grad(output, (x, w, b) if train_head else (x,), g)
                dx = gradients[0].float()
                comm.all_reduce(dx, "sum")
                options = {
                    "kind": kind,
                    "reduction": reduction,
                    "train_head": train_head,
                    "temperature": t if kind == "logprob" else 1.0,
                }
                baseline = _range_reference(x_data, full_w, full_b, y, g, tp1=True, **options)
                oracle = _range_reference(
                    x_data.double(), full_w.double(), full_b.double(), y, g, **options
                )
                baseline = (*baseline[:2], *(value[start:end] for value in baseline[2:]))
                oracle = (*oracle[:2], *(value[start:end] for value in oracle[2:]))
                print(
                    f"rank {rank}: range {dtype} {kind} {mode} train_head={train_head}", flush=True
                )
                _range_check_results((output, dx, *gradients[1:]), baseline, oracle, dtype)
        torch.cuda.synchronize()
        queue.put((rank, None))
    except BaseException:  # noqa: BLE001 -- report failures without terminating peers.
        queue.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_gpu_reduction_and_temperature_range():
    from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss

    if FusedLinearCrossEntropyLoss(provider="auto")._cross_entropy_kernel is None:
        pytest.skip("TP1 CE requires liger-kernel")
    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = Path(directory, "rendezvous").as_uri()
        processes = [
            context.Process(target=_range_worker, args=(rank, rendezvous, queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results = []
        deadline = time.monotonic() + 115
        while len(results) < 2:
            try:
                results.append(queue.get(timeout=1))
            except std_queue.Empty:
                assert time.monotonic() < deadline, (
                    "two-GPU range qualification exceeded 115 seconds"
                )
                assert all(p.exitcode in (None, 0) for p in processes), (
                    "worker exited before reporting"
                )
        for process in processes:
            process.join(timeout=2)
            assert process.exitcode == 0
        for rank, error in results:
            assert error is None, f"rank {rank}: {error}"
