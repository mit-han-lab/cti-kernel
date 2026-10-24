# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Ignore-row qualification against same-projection and locally merged oracles."""

import pytest
import torch
import torch.nn.functional as F

from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.vocab_parallel import merge_vocab_shard_stats

from .test_selected_logprobs import _relative_l2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@pytest.fixture(autouse=True)
def full_fp32_matmul(monkeypatch):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(params=("cute", "torch"))
def tp1_provider(request, monkeypatch):
    if request.param == "torch":
        import ctikernels.logprob.cute as cute

        monkeypatch.setattr(cute, "unavailable_reason", lambda: "forced torch ignore test")
    return request.param


def _inputs(rows, vocab, dtype, ignore, ignore_index=-100):
    torch.manual_seed(1903)
    hidden = torch.randn(rows, 16, device="cuda", dtype=dtype) * 0.1
    weight = torch.randn(vocab, 16, device="cuda", dtype=dtype)
    bias = torch.randn(vocab, device="cuda", dtype=dtype) * 0.1
    targets = torch.randint(vocab, (rows,), device="cuda")
    targets.masked_fill_(targets == ignore_index, vocab // 2)
    if ignore == "all":
        targets.fill_(ignore_index)
    elif ignore == "mixed":
        targets[::3] = ignore_index
        if rows > 1:
            targets[1] = 0
        if rows > 2:
            targets[2] = vocab - 1
    elif rows:
        targets[0] = 0
        targets[-1] = vocab - 1
    return hidden, weight, bias, targets


def _same_projection_fp64(hidden, weight, targets, bias, temperature, ignore_index=-100):
    result = torch.empty(hidden.shape[0], device=hidden.device)
    for start in range(0, hidden.shape[0], 64):
        end = start + 64
        logits = F.linear(hidden[start:end], weight, bias)
        scaled = (logits.float() / temperature).double()
        peak = scaled.amax(-1)
        total = (scaled - peak[:, None]).exp().sum(-1)
        y = targets[start:end]
        selected = scaled.gather(1, y.clamp_min(0)[:, None])[:, 0]
        result[start:end] = torch.where(y != ignore_index, selected - peak - total.log(), 0.0)
    return result


@pytest.mark.parametrize("rows", (0, 1, 2, 63, 64, 65, 4096))
@pytest.mark.parametrize("vocab", (151936, 248320, 151937))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ignore", ("none", "mixed", "all"))
@pytest.mark.parametrize("temperature", (1.0, 0.7))
def test_ignore_forward_grid(rows, vocab, dtype, ignore, temperature, tp1_provider):
    hidden, weight, bias, targets = _inputs(rows, vocab, dtype, ignore)
    op = ChunkedLinearSelectedTokenLogprobs()
    actual = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
    assert op.provider == tp1_provider
    expected = _same_projection_fp64(hidden, weight, targets, bias, temperature)
    valid = targets != -100
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-6, atol=1e-5)
    assert torch.count_nonzero(actual[~valid]) == 0
    assert actual.shape == (rows,) and actual.dtype == torch.float32


@pytest.mark.parametrize("rows", (1, 2, 65))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ignore", ("mixed", "all"))
@pytest.mark.parametrize("temperature", (1.0, 0.7))
def test_ignore_hidden_gradient(rows, dtype, ignore, temperature, tp1_provider):
    hidden, weight, bias, targets = _inputs(rows, 151937, dtype, ignore)
    hidden.requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    valid = targets != -100
    reference_chunks = []
    for start in range(0, rows, 64):
        logits = F.linear(reference_hidden[start : start + 64], weight, bias).float() / temperature
        selected = logits.log_softmax(-1).gather(
            1, targets[start : start + 64].clamp_min(0)[:, None]
        )[:, 0]
        reference_chunks.append(torch.where(valid[start : start + 64], selected, 0.0))
    reference = torch.cat(reference_chunks)
    op = ChunkedLinearSelectedTokenLogprobs()
    actual = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
    assert op.provider == tp1_provider
    upstream = torch.randn(rows, device="cuda")
    actual.backward(upstream)
    reference.backward(upstream)
    assert torch.count_nonzero(hidden.grad[~valid]) == 0
    if dtype == torch.float32:
        torch.testing.assert_close(
            hidden.grad[valid], reference_hidden.grad[valid], rtol=1e-5, atol=1e-6
        )
    else:
        assert _relative_l2(hidden.grad[valid], reference_hidden.grad[valid]) <= 4e-3


@pytest.mark.parametrize("ignore_index", (-1, 5))
def test_custom_ignore_index(ignore_index, tp1_provider, monkeypatch):
    hidden, weight, bias, targets = _inputs(4, 97, torch.float32, "mixed", ignore_index)
    hidden.requires_grad_()
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=ignore_index)
    actual = op.forward(hidden, weight, targets, bias=bias)
    assert op.provider == tp1_provider
    expected = _same_projection_fp64(hidden, weight, targets, bias, 1.0, ignore_index)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-5)
    actual.sum().backward()
    assert torch.count_nonzero(actual[targets == ignore_index]) == 0
    assert torch.count_nonzero(hidden.grad[targets == ignore_index]) == 0
    targets[1] = -100
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="row 1"):
        op.forward(hidden, weight, targets, bias=bias)


@pytest.mark.parametrize("bad", (97, -2))
def test_invalid_tp1_checks(bad, tp1_provider, monkeypatch):
    hidden, weight, bias, targets = _inputs(3, 97, torch.float32, "none")
    targets[1] = bad
    op = ChunkedLinearSelectedTokenLogprobs()
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="row 1"):
        op.forward(hidden, weight, targets, bias=bias)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    op.forward(hidden, weight, targets, bias=bias)
    assert op.provider == tp1_provider
    torch.cuda.synchronize()


@pytest.mark.parametrize("provider", ("triton", "torch"))
@pytest.mark.parametrize("bad", (98, -2, -100))
def test_invalid_shard_checks(provider, bad, monkeypatch):
    hidden, weight, bias, targets = _inputs(3, 98, torch.float32, "none")
    targets[1] = bad
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=-1, provider=provider)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="row 1"):
        op.shard_partials(hidden, weight[:49], targets, vocab_start=0, vocab_size=98)
    with pytest.raises(ValueError, match="vocab_size"):
        op.shard_partials(hidden, weight[49:], targets, vocab_start=49, vocab_size=97)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    part = op.shard_partials(hidden, weight[:49], targets, vocab_start=0, vocab_size=98)
    op.shard_finalize(part, merge_vocab_shard_stats(part.stats[None]))
    torch.cuda.synchronize()


@pytest.mark.parametrize("mode", ("tp1", "triton", "torch"))
def test_ignore_capture_with_engine_checks(mode, monkeypatch):
    hidden, weight, bias, targets = _inputs(65, 151936, torch.float32, "mixed")
    hidden.requires_grad_()
    op = ChunkedLinearSelectedTokenLogprobs(provider="auto" if mode == "tp1" else mode)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")

    def forward():
        if mode == "tp1":
            return op.forward(hidden, weight, targets, bias=bias, temperature=0.7)
        parts = [
            op.shard_partials(
                hidden,
                weight[start : start + 75968],
                targets,
                bias=bias[start : start + 75968],
                vocab_start=start,
                vocab_size=151936,
                temperature=0.7,
            )
            for start in (0, 75968)
        ]
        merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
        return sum(op.shard_finalize(p, merged) for p in parts) / 2

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            eager = forward()
            eager.sum().backward()
            hidden.grad = None
    torch.cuda.current_stream().wait_stream(stream)
    if mode == "tp1":
        assert op.provider == "cute"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
        captured.sum().backward()
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        assert torch.count_nonzero(hidden.grad[targets == -100]) == 0


@pytest.mark.parametrize("provider", ("triton", "torch"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ignore", ("none", "mixed", "all"))
@pytest.mark.parametrize("ignore_index", (-100, -1, 5))
@pytest.mark.parametrize("fp32_logits", (False, True))
@pytest.mark.parametrize("save", (False, True))
def test_ignore_shards(provider, dtype, ignore, ignore_index, fp32_logits, save):
    hidden, weight, bias, targets = _inputs(65, 151936, dtype, ignore, ignore_index)
    op = ChunkedLinearSelectedTokenLogprobs(
        ignore_index=ignore_index, provider=provider, shard_chunk_size=64
    )
    inputs = [hidden.clone().requires_grad_() for _ in range(2)]
    parts = [
        op.shard_partials(
            x,
            weight[start : start + 75968],
            targets,
            vocab_start=start,
            vocab_size=151936,
            bias=bias[start : start + 75968],
            temperature=0.7,
            fp32_logits=fp32_logits,
            save_probabilities=save,
        )
        for x, start in zip(inputs, (0, 75968), strict=True)
    ]
    assert all(part.provider == f"{provider}:{'saved' if save else 'recompute'}" for part in parts)
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    outputs = [op.shard_finalize(part, merged) for part in parts]
    baseline = ChunkedLinearSelectedTokenLogprobs(ignore_index=ignore_index)
    x, w, b = (
        (hidden.float(), weight.float(), bias.float()) if fp32_logits else (hidden, weight, bias)
    )
    expected = baseline.forward(x, w, targets, bias=b, temperature=0.7)
    valid = targets != ignore_index
    upstream = torch.randn(65, device="cuda")
    gradients = [
        torch.autograd.grad(out, x, upstream)[0] for out, x in zip(outputs, inputs, strict=True)
    ]
    for out, grad in zip(outputs, gradients, strict=True):
        tolerance = 1e-5 if dtype == torch.float32 or fp32_logits else 1e-2
        torch.testing.assert_close(out, expected, rtol=0, atol=tolerance)
        assert torch.count_nonzero(out[~valid]) == 0
        assert torch.count_nonzero(grad[~valid]) == 0
    assert torch.count_nonzero(sum(g.float() for g in gradients)[~valid]) == 0


def test_cute_ignore_value_reuses_compilation():
    from ctikernels.logprob.cute import Provider

    logits = torch.randn(2, 151937, device="cuda")
    output = torch.empty(2, device="cuda")
    provider = Provider(logits.device)
    for ignore_index in (-100, -1, 5):
        targets = torch.tensor([ignore_index, 0], device="cuda")
        provider(logits, targets, output, 0.7, ignore_index)
        assert output[0] == 0
    assert len(provider._compiled) == 1
