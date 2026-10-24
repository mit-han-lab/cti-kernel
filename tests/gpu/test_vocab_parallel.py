# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch
import torch.nn.functional as F

from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.vocab_parallel import merge_vocab_shard_stats

from .._vocab_parallel_gates import (
    _check_results,
    _check_stats,
    _finish,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def full_fp32_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _operation(kind, provider, chunk_size=7):
    if provider == "triton":
        pytest.importorskip("triton")
    if kind == "logprob":
        return ChunkedLinearSelectedTokenLogprobs(
            chunk_size=chunk_size, shard_chunk_size=7, provider=provider
        )
    return FusedLinearCrossEntropyLoss(chunk_size=chunk_size, shard_chunk_size=7, provider=provider)


def _inputs(x, w, b, head_grad):
    return (
        x.detach().clone().requires_grad_(),
        w.detach().clone().requires_grad_(head_grad),
        b.detach().clone().requires_grad_(head_grad),
    )


def _dense(x, w, b, y, kind, upstream, head_grad, tp1=False, chunk_size=7, output_only=False):
    x, w, b = _inputs(x, w, b, head_grad)
    if output_only:
        # TP1 CE precomputes gradients in forward; diagnostics only need its loss.
        for value in (x, w, b):
            value.requires_grad_(False)
    if tp1:
        op = _operation(kind, "torch", chunk_size)
        if kind == "ce":
            output = op.forward(x, w, y, bias=b)
        else:
            output = op.forward(x, w, y.clamp_min(0), bias=b, temperature=0.7)
            output = torch.where(y != -100, output, 0.0)
    else:
        losses = F.cross_entropy(
            F.linear(x, w, b) / (0.7 if kind == "logprob" else 1.0), y, reduction="none"
        )
        output = losses.sum() if kind == "ce" else -losses
    if output_only:
        return output.detach()
    grads = torch.autograd.grad(output, (x, w, b) if head_grad else (x,), upstream.to(output.dtype))
    return (output.detach(), *grads)


def _simulate(x, w, b, y, kind, upstream, head_grad, tp, provider, *, checks, **options):
    op = _operation(kind, provider)
    parts, inputs = [], []
    offset = 0
    for shard in w.tensor_split(tp):
        count = shard.shape[0]
        xi, wi, bi = _inputs(x, shard, b[offset : offset + count], head_grad)
        inputs.append((xi, wi, bi) if head_grad else (xi,))
        kwargs = {"temperature": 0.7} if kind == "logprob" else {}
        parts.append(
            op.shard_partials(
                xi, wi, y, vocab_start=offset, vocab_size=w.shape[0], bias=bi, **kwargs, **options
            )
        )
        mode = "saved" if options.get("save_probabilities", False) else "recompute"
        assert parts[-1].provider == f"{provider}:{mode}"
        _check_stats(
            parts[-1],
            xi,
            wi,
            bi,
            y,
            offset,
            kwargs.get("temperature", 1.0),
            options.get("fp32_logits", False),
            checks=checks,
            tp1=(w, b, 7, kind),
        )
        offset += count
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    outputs = [op.shard_finalize(p, merged) for p in parts]
    gradients = [
        torch.autograd.grad(output, ins, upstream)
        for output, ins in zip(outputs, inputs, strict=True)
    ]
    if kind == "logprob":
        assert all(torch.count_nonzero(output[y == -100]) == 0 for output in outputs)
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
    result = [outputs[0], sum(g[0].float() for g in gradients)]
    if head_grad:
        result.extend(torch.cat([g[i] for g in gradients]) for i in (1, 2))
    return tuple(result)


@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("provider", ["torch", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind,head_grad", [("ce", True), ("ce", False), ("logprob", False)])
def test_shards_vs_tp1_and_fp64(
    tp, provider, dtype, kind, head_grad, fp32_logits, save_probabilities
):
    if kind == "ce" and _operation(kind, provider)._cross_entropy_kernel is None:
        pytest.skip("TP1 CE requires liger-kernel")
    print(
        f"tp={tp} provider={provider} kind={kind} head_grad={head_grad} "
        f"fp32_logits={fp32_logits} save_probabilities={save_probabilities}"
    )
    torch.manual_seed(46)
    x = (torch.randn(37, 256, device="cuda") / 4).to(dtype)
    w = (torch.randn(1003, 256, device="cuda") / 4).to(dtype)
    b = (torch.randn(1003, device="cuda") / 4).to(dtype)
    y = torch.arange(37, device="cuda", dtype=torch.int64) * 23 % 1003
    offsets = [0]
    for shard in w.tensor_split(tp):
        offsets.append(offsets[-1] + shard.shape[0])
    boundaries = [-100, 0, 1002]
    for boundary in offsets[1:-1]:
        boundaries.extend((boundary - 1, boundary, boundary + 1))
    y[: len(boundaries)] = torch.tensor(boundaries, device="cuda")
    upstream = (
        torch.tensor(0.375, device="cuda") if kind == "ce" else torch.randn(37, device="cuda")
    )
    checks = []
    actual = _simulate(
        x,
        w,
        b,
        y,
        kind,
        upstream,
        head_grad,
        tp,
        provider,
        checks=checks,
        fp32_logits=fp32_logits,
        save_probabilities=save_probabilities,
        vocab_block_size=128,
    )
    oracle = _dense(x.double(), w.double(), b.double(), y, kind, upstream, head_grad)
    baseline = _dense(x, w, b, y, kind, upstream, head_grad, tp1=True)
    if fp32_logits:
        # Only the output uses FP32 TP1; gradient gates retain the head-dtype baseline.
        output = _dense(
            x.float(),
            w.float(),
            b.float(),
            y,
            kind,
            upstream,
            head_grad,
            tp1=True,
            output_only=True,
        )
        baseline = (output, *baseline[1:])
    _check_results(
        actual,
        baseline,
        oracle,
        dtype,
        checks=checks,
        fp32_logits=fp32_logits,
    )
    assert torch.count_nonzero(actual[1][0]) == 0
    _finish(checks)


@pytest.mark.parametrize("provider", ["torch", "triton"])
@pytest.mark.parametrize("kind,head_grad", [("ce", True), ("ce", False), ("logprob", False)])
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_training_graph_replay(provider, kind, head_grad, save_probabilities, fp32_logits, dtype):
    torch.manual_seed(17)
    op = _operation(kind, provider)
    x = torch.randn(11, 32, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(101, 32, device="cuda", dtype=dtype, requires_grad=head_grad)
    b = torch.randn(101, device="cuda", dtype=dtype, requires_grad=head_grad)
    y = torch.arange(11, device="cuda", dtype=torch.int64)
    upstream = torch.tensor(0.5, device="cuda") if kind == "ce" else torch.randn(11, device="cuda")

    def run():
        parts = []
        for start, end in [(0, 51), (51, 101)]:
            kwargs = {"temperature": 0.7} if kind == "logprob" else {}
            parts.append(
                op.shard_partials(
                    x,
                    w[start:end],
                    y,
                    vocab_start=start,
                    vocab_size=w.shape[0],
                    bias=b[start:end],
                    save_probabilities=save_probabilities,
                    fp32_logits=fp32_logits,
                    **kwargs,
                )
            )
        merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
        outputs = [op.shard_finalize(p, merged) for p in parts]
        # Both shard paths share x here, so autograd adds dX partials for us.
        total = outputs[0] + outputs[1]
        gradients = torch.autograd.grad(total, (x, w, b) if head_grad else (x,), upstream)
        return (outputs[0], *gradients)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    for replay in range(2):
        with torch.no_grad():
            x.copy_(torch.randn_like(x) / (replay + 2))
            y.copy_((torch.arange(11, device="cuda") * 13 + replay) % 101)
            y[0] = -100
        graph.replay()
        eager = run()
        torch.cuda.synchronize()
        for got, expected in zip(captured, eager, strict=True):
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("provider", ["torch", "triton"])
@pytest.mark.parametrize("head_grad", [False, True])
def test_online_reduction_large_uneven_vocab(provider, head_grad):
    if _operation("ce", provider)._cross_entropy_kernel is None:
        pytest.skip("TP1 CE requires liger-kernel")
    torch.manual_seed(49)
    x = torch.randn(7, 32, device="cuda") / 4
    w = torch.randn(32003, 32, device="cuda") / 4
    b = torch.randn(32003, device="cuda") / 4
    y = torch.tensor([-100, 0, 16000, 16001, 16002, 16003, 32002], device="cuda")
    g = torch.tensor(0.37, device="cuda")
    checks = []
    actual = _simulate(x, w, b, y, "ce", g, head_grad, 2, provider, checks=checks)
    oracle = _dense(x.double(), w.double(), b.double(), y, "ce", g, head_grad)
    baseline = _dense(x, w, b, y, "ce", g, head_grad, tp1=True)
    _check_results(
        actual,
        baseline,
        oracle,
        x.dtype,
        checks=checks,
    )
    _finish(checks)


def _range_reference(
    x,
    w,
    b,
    y,
    upstream,
    *,
    kind,
    train_head,
    reduction="sum",
    temperature=1.0,
    tp1=False,
    fp32_logits=False,
):
    x, w, b = _inputs(x, w, b, train_head)
    inputs = (x, w, b) if train_head else (x,)
    if tp1 and kind == "ce":
        op = FusedLinearCrossEntropyLoss(
            chunk_size=1024,
            provider="auto",
            ignore_index=-7,
            reduction=reduction,
        )
        if op._cross_entropy_kernel is None:
            pytest.skip("TP1 CE requires liger-kernel")
        output = op.forward(x, w, y, bias=b)
    elif tp1 and kind == "logprob":
        op = ChunkedLinearSelectedTokenLogprobs(chunk_size=1024, ignore_index=-7)
        output = op.forward(x, w, y, bias=b, temperature=temperature)
    else:
        chunks = [F.linear(rows, w, b) for rows in x.split(1024)]
        logits = torch.cat(chunks)
        t = temperature[:, None] if isinstance(temperature, torch.Tensor) else temperature
        losses = F.cross_entropy(logits / t, y, ignore_index=-7, reduction="none")
        output = -losses if kind == "logprob" else (losses if reduction == "none" else losses.sum())
        if kind == "ce" and reduction == "mean":
            output = output / (y != -7).sum()
    gradients = torch.autograd.grad(output, inputs, upstream.to(output.dtype))
    if tp1 and fp32_logits and x.dtype != torch.float32:
        # Keep the head-dtype gradient baseline while checking FP32 outputs.
        float_result = _range_reference(
            x.float(),
            w.float(),
            b.float(),
            y,
            upstream,
            kind=kind,
            train_head=train_head,
            reduction=reduction,
            temperature=temperature,
            tp1=True,
        )
        output = float_result[0]
    return output.detach(), *gradients


def _range_check_results(actual, baseline, oracle, dtype, *, fp32_logits=False, tp=None):
    from .._vocab_parallel_gates import _assert_gate, _head_dtype_mode

    checks = []
    for index, (got, old, ref) in enumerate(zip(actual, baseline, oracle, strict=True)):
        if got.numel() == 0 or (index == 0 and bool(torch.isnan(old).all())):
            torch.testing.assert_close(
                got, old, rtol=0, atol=0, equal_nan=True, check_dtype=index == 0
            )
            continue
        _assert_gate(
            got,
            old,
            checks=checks,
            label=("output", "dX", "dW", "db")[index],
            dtype=dtype,
            oracle=ref if index or _head_dtype_mode(dtype, fp32_logits) else None,
            output=index == 0,
            # TP4 range grid: each of the tp shards adds at most one head-dtype
            # rounding (dX partials, shard logits), so the bound grows with tp.
            # Every other caller (TP2) keeps the 1.1x rule.
            factor=1.1 if tp is None else tp,
        )
    _finish(checks)


def _range_simulate(
    x,
    w,
    b,
    y,
    upstream,
    *,
    kind,
    train_head,
    reduction="sum",
    temperature=1.0,
    saved=False,
    fp32_logits=False,
):
    options = {"provider": "triton", "shard_chunk_size": 1024, "ignore_index": -7}
    op = (
        FusedLinearCrossEntropyLoss(reduction=reduction, **options)
        if kind == "ce"
        else ChunkedLinearSelectedTokenLogprobs(**options)
    )
    parts, inputs = [], []
    split = (w.shape[0] + 3) // 4
    for start in range(0, w.shape[0], split):
        end = min(start + split, w.shape[0])
        xi = torch.empty((x.shape[0], x.shape[1] + 17), device=x.device, dtype=x.dtype)[
            :, : x.shape[1]
        ]
        xi.copy_(x).requires_grad_()
        wi = w[start:end].detach().clone().requires_grad_(train_head)
        bi = b[start:end].detach().clone().requires_grad_(train_head)
        inputs.append((xi, wi, bi) if train_head else (xi,))
        kwargs = {"temperature": temperature} if kind == "logprob" else {}
        parts.append(
            op.shard_partials(
                xi,
                wi,
                y,
                bias=bi,
                vocab_start=start,
                vocab_size=w.shape[0],
                save_probabilities=saved,
                fp32_logits=fp32_logits,
                **kwargs,
            )
        )
    assert len(parts) == 4
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    outputs = [op.shard_finalize(p, merged) for p in parts]
    grads = [
        torch.autograd.grad(out, ins, upstream) for out, ins in zip(outputs, inputs, strict=True)
    ]
    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], rtol=0, atol=0, equal_nan=True)
    output = outputs[0]
    if kind == "ce" and reduction != "none":
        # The scalar must be exactly the FP32 sum/mean of the per-row losses;
        # the per-row vector is what the oracle rule judges.
        valid = y != -7
        output = torch.where(valid, merged[:, 0] - merged[:, 1], 0.0)
        expected = output.sum() if reduction == "sum" else output.sum() / valid.sum()
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0, equal_nan=True)
    result = [output, sum(g[0].float() for g in grads)]
    if train_head:
        result.extend(torch.cat([g[i] for g in grads]) for i in (1, 2))
    return tuple(result)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("ignore", ["none", "mixed", "all"])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("n,v", [(0, 7), (1, 7), (65, 257), (1027, 257), (1, 151937)])
def test_tp4_ce_range(dtype, reduction, ignore, train_head, n, v):
    torch.manual_seed(147)
    x = (torch.randn(n, 256, device="cuda") / 4).to(dtype)
    w = (torch.randn(v, 256, device="cuda") / 4).to(dtype)
    b = (torch.randn(v, device="cuda") / 4).to(dtype)
    y = torch.arange(n * 2, device="cuda", dtype=torch.int64)[::2]
    y.mul_(79).remainder_(v)
    if n:
        y[-1] = v - 1
    if ignore != "none":
        y[:: 3 if ignore == "mixed" else 1] = -7
    upstream = (
        torch.randn(n * 2, device="cuda")[::2]
        if reduction == "none"
        else torch.tensor(0.37, device="cuda")
    )
    options = {"kind": "ce", "train_head": train_head, "reduction": reduction}
    actual = _range_simulate(x, w, b, y, upstream, saved=reduction != "sum", **options)
    baseline = _range_reference(x, w, b, y, upstream, tp1=True, **options)
    oracle = _range_reference(x.double(), w.double(), b.double(), y, upstream, **options)
    if reduction != "none":
        # The 1.1x oracle rule judges the per-row losses;
        # _range_simulate requires the scalar to equal their FP32 sum/mean.
        rows = {**options, "reduction": "none"}
        ones = torch.ones(n, device="cuda")
        tp1_rows = _range_reference(x, w, b, y, ones, tp1=True, **rows)[0]
        oracle_rows = _range_reference(x.double(), w.double(), b.double(), y, ones, **rows)[0]
        baseline = (tp1_rows, *baseline[1:])
        oracle = (oracle_rows, *oracle[1:])
    _range_check_results(actual, baseline, oracle, dtype, tp=4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("temperature_mode", [1.0, 0.7, "rows"])
@pytest.mark.parametrize("ignore", ["none", "mixed", "all"])
@pytest.mark.parametrize("n,v", [(0, 7), (1, 7), (65, 257), (1027, 257), (1, 151937)])
def test_tp4_logprob_range(dtype, temperature_mode, ignore, n, v):
    torch.manual_seed(148)
    x = (torch.randn(n, 256, device="cuda") / 4).to(dtype)
    w = (torch.randn(v, 256, device="cuda") / 4).to(dtype)
    b = (torch.randn(v, device="cuda") / 4).to(dtype)
    y = torch.arange(n * 2, device="cuda", dtype=torch.int64)[::2]
    y.mul_(79).remainder_(v)
    if n:
        y[-1] = v - 1
    if ignore != "none":
        y[:: 3 if ignore == "mixed" else 1] = -7
    t = (
        torch.linspace(0.5, 1.5, n * 2, device="cuda")[::2]
        if temperature_mode == "rows"
        else temperature_mode
    )
    upstream = torch.randn(n * 2, device="cuda")[::2]
    options = {"kind": "logprob", "train_head": False, "temperature": t}
    actual = _range_simulate(x, w, b, y, upstream, saved=temperature_mode == "rows", **options)
    baseline = _range_reference(x, w, b, y, upstream, tp1=True, **options)
    oracle = _range_reference(x.double(), w.double(), b.double(), y, upstream, **options)
    _range_check_results(actual, baseline, oracle, dtype, tp=4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "kind,train_head,reduction,per_row_t",
    [
        ("ce", True, "sum", False),
        ("ce", True, "none", False),
        ("ce", False, "mean", False),
        ("logprob", False, "sum", False),
        ("logprob", False, "sum", True),
    ],
)
@pytest.mark.parametrize("saved", [False, True])
def test_new_runtime_values_capture_without_compile(
    dtype, kind, train_head, reduction, per_row_t, saved
):
    from ctikernels.vocab_parallel import _shard_finalize, _shard_partials, _triton_kernels

    kernels = _triton_kernels()
    device = torch.cuda.current_device()

    def cache_sizes():
        try:
            return tuple(len(kernel.device_caches[device][0]) for kernel in kernels)
        except (AttributeError, KeyError, TypeError, IndexError):
            pytest.skip("Triton per-device JIT cache inspection is unavailable")

    cache_sizes()

    def inputs(n, v, h, stride, start, temperature):
        x = torch.randn(n, h + stride, device="cuda", dtype=dtype)[:, :h].requires_grad_()
        w = torch.randn(v, h + stride, device="cuda", dtype=dtype)[:, :h].requires_grad_(train_head)
        y = torch.empty(n * stride, device="cuda", dtype=torch.int64)[::stride]
        y.copy_(torch.arange(n, device="cuda") % v + start)
        t = (
            torch.full((n * stride,), temperature, device="cuda")[::stride]
            if per_row_t
            else temperature
        )
        g = (
            torch.randn(n * stride, device="cuda")[::stride]
            if kind == "logprob" or reduction == "none"
            else torch.tensor(0.37, device="cuda")
        )
        return x, w, y, t, g, start

    def run(values):
        x, w, y, t, g, start = values
        part = _shard_partials(
            x,
            w,
            y,
            vocab_start=start,
            vocab_size=start + w.shape[0],
            temperature=t,
            kind=kind,
            reduction=reduction,
            provider="triton",
            save_probabilities=saved,
        )
        merged = merge_vocab_shard_stats(part.stats[None])
        output = _shard_finalize(part, merged, kind=kind)
        return output, *torch.autograd.grad(output, (x, w) if train_head else (x,), g)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = inputs(16, 16, 32, 1, 0, 1.0)
        run(warm)
        changed = inputs(17, 19, 33, 3, 17, 0.7)
    stream.synchronize()
    before = cache_sizes()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(changed)
    assert cache_sizes() == before
    graph.replay()
    eager = run(changed)
    torch.cuda.synchronize()
    assert cache_sizes() == before
    for got, expected in zip(captured, eager, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("train_head", [False, True])
def test_empty_ce_launches_no_gemm_or_triton(monkeypatch, reduction, saved, train_head):
    import ctikernels.vocab_parallel as vp

    def forbidden(*args, **kwargs):
        raise AssertionError("empty CE launched a GEMM or Triton kernel")

    op = FusedLinearCrossEntropyLoss(provider="triton", reduction=reduction)
    op._cross_entropy_kernel = object()
    x = torch.empty(0, 256, device="cuda", requires_grad=True)
    w = torch.randn(7, 256, device="cuda", requires_grad=train_head)
    b = torch.randn(7, device="cuda", requires_grad=train_head)
    y = torch.empty(0, device="cuda", dtype=torch.int64)
    monkeypatch.setattr(vp, "_triton_kernels", forbidden)
    monkeypatch.setattr(vp, "_logits", forbidden)
    monkeypatch.setattr(vp, "_mm", forbidden)
    monkeypatch.setattr(torch, "mm", forbidden)
    monkeypatch.setattr(torch, "addmm", forbidden)
    part = op.shard_partials(x, w, y, bias=b, vocab_start=0, vocab_size=7, save_probabilities=saved)
    shard = op.shard_finalize(part, merge_vocab_shard_stats(part.stats[None]))
    tp1 = op.forward(x, w, y, bias=b)
    torch.testing.assert_close(shard, tp1, rtol=0, atol=0, equal_nan=True)
    for output in (shard, tp1):
        grads = torch.autograd.grad(output.sum(), (x, w, b) if train_head else (x,))
        assert all(torch.count_nonzero(g) == 0 for g in grads)
