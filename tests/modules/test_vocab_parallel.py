# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch
import torch.nn.functional as F

from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss
from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.logprob.linear import _TorchProvider
from ctikernels.vocab_parallel import (
    SHARD_CHUNK_SIZE,
    _shard_finalize,
    _shard_partials,
    merge_vocab_shard_stats,
    vocab_shard_merged_from_sums,
    vocab_shard_sum_terms,
)

from .._vocab_parallel_gates import (
    _assert_gate,
    _check_results,
    _check_stats,
    _finish,
)


def test_fp32_gradient_gate_uses_tensor_max_not_oracle_ratio_or_unit_floor():
    checks = []
    tp1 = torch.tensor([0.001])
    got = tp1 + 2e-9
    # Oracle ratio is infinite, but the relative FP32 gate passes.
    _assert_gate(got, tp1, checks=checks, label="dX", dtype=torch.float32, oracle=tp1.double())
    _finish(checks)
    _assert_gate(
        tp1 + 2e-8, tp1, checks=checks, label="dW", dtype=torch.float32, oracle=tp1.double()
    )
    with pytest.raises(pytest.fail.Exception, match="FAIL.*dW"):
        _finish(checks)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_gradient_gate_has_no_additive_slack(dtype):
    checks = []
    tp1 = torch.zeros(1, dtype=dtype)
    _assert_gate(
        torch.tensor([1e-7], dtype=dtype),
        tp1,
        checks=checks,
        label="dX",
        dtype=dtype,
        oracle=tp1.double(),
    )
    with pytest.raises(pytest.fail.Exception, match="FAIL.*dX"):
        _finish(checks)


def test_merge_routes_and_rank_order():
    stats = torch.tensor(
        [
            [[2.0, 3.0, 7.0], [0.0, 1.0, 0.0]],
            [[4.0, 2.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 4.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    maximum = stats[:, :, 0].max(0).values
    terms = [vocab_shard_sum_terms(shard, maximum) for shard in stats]
    sums = terms[0] + terms[1] + terms[2]
    expected = torch.stack((maximum + sums[:, 0].log(), sums[:, 1]), 1)
    out = torch.empty(2, 2)
    actual = merge_vocab_shard_stats(stats, out=out)
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(vocab_shard_merged_from_sums(maximum, sums), actual, rtol=0, atol=0)
    assert actual[0, 1] == 7
    assert actual[1, 1] == 0
    for _ in range(3):
        assert torch.equal(merge_vocab_shard_stats(stats), actual)
    assert vocab_shard_sum_terms(stats[0], maximum, out=out) is out
    assert vocab_shard_merged_from_sums(maximum, sums, out=sums) is sums
    torch.testing.assert_close(sums, expected)


@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize("tp", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "kind,train_hidden,train_weight,train_bias",
    [
        ("ce", True, True, True),
        ("ce", True, False, False),
        ("ce", False, True, False),
        ("ce", False, False, True),
        ("ce", True, False, True),
        ("logprob", True, False, False),
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_shards_match_dense_double(
    tp, kind, train_hidden, train_weight, train_bias, use_bias, fp32_logits, save_probabilities
):
    if train_bias and not use_bias and not train_hidden and not train_weight:
        pytest.skip("no trainable input")
    torch.manual_seed(13)
    x = torch.randn(7, 5) / 3
    w = torch.randn(11, 5) / 3
    b = torch.randn(11) / 3 if use_bias else None
    y = torch.tensor([-100, 0, 2, 3, 5, 8, 10])
    temperature = 0.7 if kind == "logprob" else 1.0
    xr = x.double().requires_grad_(train_hidden)
    wr = w.double().requires_grad_(train_weight)
    br = b.double().requires_grad_(train_bias) if b is not None else None
    z = F.linear(xr, wr, br) / temperature
    losses = F.cross_entropy(z, y, reduction="none")
    ref = losses.sum() if kind == "ce" else -losses
    upstream = torch.tensor(0.37) if kind == "ce" else torch.randn(7)
    ref.backward(upstream.double())
    checks = []
    parts = []
    offset = 0
    for ws in w.tensor_split(tp):
        size = ws.shape[0]
        part = _shard_partials(
            x.clone().requires_grad_(train_hidden),
            ws.clone().requires_grad_(train_weight),
            y,
            bias=b[offset : offset + size].clone().requires_grad_(train_bias)
            if b is not None
            else None,
            vocab_start=offset,
            vocab_size=w.shape[0],
            temperature=temperature,
            chunk_size=3,
            fp32_logits=fp32_logits,
            save_probabilities=save_probabilities,
            provider="torch",
            kind=kind,
            _allow_cpu=True,
        )
        _check_stats(
            part,
            x,
            ws,
            b[offset : offset + size] if b is not None else None,
            y,
            offset,
            temperature,
            fp32_logits,
            checks=checks,
        )
        assert part.stats.is_contiguous() and part.stats.dtype == torch.float32
        assert (part._average is not None) == (
            train_hidden
            and not train_weight
            and not (use_bias and train_bias)
            and not save_probabilities
        )
        parts.append(part)
        offset += size
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    for part in parts:
        result = _shard_finalize(part, merged, kind=kind)
        torch.testing.assert_close(result.double(), ref, rtol=1e-6, atol=1e-6)
        result.backward(upstream)
        if train_hidden:
            assert torch.count_nonzero(part._hidden.grad[0]) == 0
    if train_hidden:
        torch.testing.assert_close(
            sum(p._hidden.grad for p in parts).double(), xr.grad, rtol=2e-5, atol=2e-7
        )
    if train_weight:
        torch.testing.assert_close(
            torch.cat([p._weight.grad for p in parts]).double(), wr.grad, rtol=2e-5, atol=2e-7
        )
    if use_bias and train_bias:
        torch.testing.assert_close(
            torch.cat([p._bias.grad for p in parts]).double(), br.grad, rtol=2e-5, atol=2e-7
        )

    _finish(checks)


def _part(**overrides):
    kwargs = {
        "hidden": torch.randn(3, 2),
        "weight": torch.randn(4, 2),
        "target_ids": torch.tensor([0, -100, 3]),
        "vocab_start": 0,
        "vocab_size": 4,
        "_allow_cpu": True,
        "provider": "torch",
    }
    kwargs.update(overrides)
    return _shard_partials(**kwargs)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"vocab_start": True}, TypeError),
        ({"vocab_start": -1}, ValueError),
        ({"vocab_start": 1.5}, TypeError),
        ({"vocab_size": None}, TypeError),
        ({"vocab_size": True}, TypeError),
        ({"vocab_size": 4.0}, TypeError),
        ({"vocab_size": 3}, ValueError),
        ({"vocab_start": 1, "vocab_size": 4}, ValueError),
        ({"weight": torch.empty(0, 2)}, ValueError),
        ({"weight": torch.empty(4, 3)}, ValueError),
        ({"target_ids": torch.zeros(3, dtype=torch.int32)}, TypeError),
        ({"target_ids": torch.zeros(3, 1, dtype=torch.int64)}, ValueError),
        ({"weight": torch.empty(4, 2, dtype=torch.float64)}, TypeError),
        ({"bias": torch.zeros(5)}, ValueError),
        ({"temperature": 0}, ValueError),
        ({"temperature": float("nan")}, ValueError),
        ({"temperature": True}, TypeError),
        ({"provider": "bad"}, ValueError),
        ({"chunk_size": 0}, ValueError),
        ({"fp32_logits": 1}, TypeError),
        ({"save_probabilities": 1}, TypeError),
        ({"vocab_block_size": 0}, ValueError),
        ({"vocab_block_size": True}, ValueError),
        ({"_allow_cpu": False}, ValueError),
        ({"kind": "logprob", "weight": torch.randn(4, 2, requires_grad=True)}, ValueError),
        ({"kind": "logprob", "bias": torch.randn(4, requires_grad=True)}, ValueError),
        ({"target_ids": torch.empty(3, dtype=torch.int64, device="meta")}, ValueError),
    ],
)
def test_validation(override, error):
    with pytest.raises(error):
        _part(**override)


@pytest.mark.parametrize("available", [False, True])
def test_provider_validation_does_not_load_triton(monkeypatch, available):
    import ctikernels.vocab_parallel as vp

    def forbidden():
        raise AssertionError("CPU provider validation must not load Triton kernels")

    monkeypatch.setattr(vp, "_triton_available", lambda: available)
    monkeypatch.setattr(vp, "_triton_kernels", forbidden)
    assert _part(provider="auto")._provider == "torch"
    assert _part(provider="torch")._provider == "torch"
    if available:
        with pytest.raises(ValueError, match="Triton requires CUDA inputs"):
            _part(provider="triton")
    else:
        with pytest.raises(RuntimeError, match="requires Triton"):
            _part(provider="triton")


def test_finalize_validation_and_single_use():
    part = _part()
    for merged, error in [
        (torch.empty(3, 3), ValueError),
        (torch.empty(3, 2, dtype=torch.float64), TypeError),
        (torch.empty(3, 2, device="meta"), ValueError),
    ]:
        with pytest.raises(error):
            _shard_finalize(part, merged, kind="ce")
    with pytest.raises(TypeError):
        _shard_finalize(part, torch.empty(3, 2), kind="logprob")
    merged = merge_vocab_shard_stats(part.stats[None])
    _shard_finalize(part, merged, kind="ce")
    with pytest.raises(RuntimeError, match="already"):
        _shard_finalize(part, merged, kind="ce")


@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize("kind", ["ce", "logprob"])
def test_all_ignored_and_frozen_backward_does_not_recompute(kind, monkeypatch, save_probabilities):
    import ctikernels.vocab_parallel as vp

    x = torch.randn(3, 2, requires_grad=True)
    part = _part(
        hidden=x,
        target_ids=torch.full((3,), -100, dtype=torch.int64),
        kind=kind,
        save_probabilities=save_probabilities,
    )
    merged = merge_vocab_shard_stats(part.stats[None])

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen backward must not recompute logits")

    monkeypatch.setattr(vp, "_logits", forbidden)
    result = _shard_finalize(part, merged, kind=kind)
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("save_probabilities", [False, True])
def test_empty_logprob(save_probabilities):
    x = torch.empty(0, 2, requires_grad=True)
    part = _part(
        hidden=x,
        target_ids=torch.empty(0, dtype=torch.int64),
        kind="logprob",
        save_probabilities=save_probabilities,
    )
    result = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="logprob")
    result.sum().backward()
    assert result.shape == (0,) and x.grad.shape == x.shape


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
def test_public_shard_requires_vocab_size(op_type):
    op = op_type(provider="torch")
    with pytest.raises(TypeError, match="required keyword-only argument: 'vocab_size'"):
        op.shard_partials(
            torch.randn(3, 2), torch.randn(4, 2), torch.tensor([0, -100, 3]), vocab_start=0
        )


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
@pytest.mark.parametrize("bad", [-1, 8, 999])
def test_public_shard_checks_global_target_bounds(monkeypatch, op_type, bad):
    import ctikernels.cross_entropy as ce_module
    import ctikernels.logprob.linear as lp_module

    def cpu_entry(*args, **kwargs):
        return _shard_partials(*args, **kwargs, _allow_cpu=True)

    monkeypatch.setattr(ce_module, "_shard_partials", cpu_entry)
    monkeypatch.setattr(lp_module, "_shard_partials", cpu_entry)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    op = op_type(provider="torch")
    hidden, weight = torch.randn(4, 2), torch.randn(4, 2)
    # Ignore is valid, as are IDs owned by another shard and the last global ID.
    targets = torch.tensor([-100, 0, 7, bad])
    with pytest.raises(ValueError, match="row 3"):
        op.shard_partials(hidden, weight, targets, vocab_start=4, vocab_size=8)
    targets[-1] = 4
    op.shard_partials(hidden, weight, targets, vocab_start=4, vocab_size=8)
    targets[-1] = bad
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    op.shard_partials(hidden, weight, targets, vocab_start=4, vocab_size=8)


@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("shard_options", [{}, {"shard_chunk_size": 2}])
def test_public_shard_methods(monkeypatch, fp32_logits, shard_options):
    import ctikernels.cross_entropy as ce_module
    import ctikernels.logprob.linear as lp_module

    def cpu_entry(*args, **kwargs):
        return _shard_partials(*args, **kwargs, _allow_cpu=True)

    monkeypatch.setattr(ce_module, "_shard_partials", cpu_entry)
    monkeypatch.setattr(lp_module, "_shard_partials", cpu_entry)
    for op in [
        FusedLinearCrossEntropyLoss(
            chunk_size=7, provider="torch", fp32_logits=fp32_logits, **shard_options
        ),
        ChunkedLinearSelectedTokenLogprobs(
            chunk_size=7, provider="torch", fp32_logits=fp32_logits, **shard_options
        ),
    ]:
        part = op.shard_partials(
            torch.randn(3, 2),
            torch.randn(4, 2),
            torch.tensor([0, -100, 3]),
            vocab_start=0,
            vocab_size=4,
        )
        assert part._fp32_logits == fp32_logits
        assert part._chunk_size == shard_options.get("shard_chunk_size", SHARD_CHUNK_SIZE)
        override = op.shard_partials(
            torch.randn(3, 2),
            torch.randn(4, 2),
            torch.tensor([0, -100, 3]),
            vocab_start=0,
            vocab_size=4,
            fp32_logits=not fp32_logits,
            save_probabilities=True,
            chunk_size=1,
            vocab_block_size=3,
        )
        assert override._fp32_logits != fp32_logits
        assert override._saved_projection is not None and override._average is None
        assert override._chunk_size == 1 and override._vocab_block_size == 3
        result = op.shard_finalize(part, merge_vocab_shard_stats(part.stats[None]))
        assert result.dtype == torch.float32


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
def test_shard_chunk_size_default_and_property(op_type):
    assert SHARD_CHUNK_SIZE == 1024
    assert _part()._chunk_size == SHARD_CHUNK_SIZE
    op = op_type(provider="torch")
    assert op.shard_chunk_size == SHARD_CHUNK_SIZE
    assert op.chunk_size == (256 if op_type is FusedLinearCrossEntropyLoss else 64)
    custom = op_type(chunk_size=7, shard_chunk_size=2, provider="torch")
    assert custom.chunk_size == 7 and custom.shard_chunk_size == 2
    with pytest.raises(AttributeError):
        custom.shard_chunk_size = 3


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_shard_chunk_size_validation(op_type, value):
    error = (
        TypeError
        if op_type is FusedLinearCrossEntropyLoss and type(value) is not int
        else ValueError
    )
    with pytest.raises(error, match="shard_chunk_size"):
        op_type(shard_chunk_size=value, provider="torch")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32_logits", [False, True])
def test_low_precision_projection_dtype(dtype, fp32_logits):
    x = torch.tensor([[1.125, 1.0625]], dtype=dtype)
    w = torch.tensor([[1.125, 1.0625], [-0.75, 0.125]], dtype=dtype)
    part = _shard_partials(
        x,
        w,
        torch.tensor([0]),
        vocab_start=0,
        vocab_size=w.shape[0],
        kind="logprob",
        provider="torch",
        _allow_cpu=True,
        fp32_logits=fp32_logits,
    )
    merged = merge_vocab_shard_stats(part.stats[None])
    result = _shard_finalize(part, merged, kind="logprob")
    logits = x.double() @ w.double().t()
    if not fp32_logits:
        logits = logits.to(dtype).double()
    ref = logits.log_softmax(-1)[:, 0]
    torch.testing.assert_close(result.double(), ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "function,args",
    [
        (merge_vocab_shard_stats, (torch.empty(0, 3, 3),)),
        (merge_vocab_shard_stats, (torch.empty(2, 3, 3, dtype=torch.float64),)),
        (vocab_shard_sum_terms, (torch.empty(3, 2), torch.empty(3))),
        (vocab_shard_sum_terms, (torch.empty(3, 3), torch.empty(4))),
        (vocab_shard_merged_from_sums, (torch.empty(3), torch.empty(3, 3))),
    ],
)
def test_merge_metadata(function, args):
    with pytest.raises((ValueError, TypeError)):
        function(*args)


def test_stats_are_detached_and_merge_out_validates():
    part = _part(hidden=torch.randn(3, 2, requires_grad=True))
    assert not part.stats.requires_grad
    assert part._average is not None and not part._average.requires_grad
    gathered = part.stats[None]
    for out, error in [
        (torch.empty(3, 3), ValueError),
        (torch.empty(3, 2, dtype=torch.float64), TypeError),
        (torch.empty(3, 2, requires_grad=True), ValueError),
    ]:
        with pytest.raises(error):
            merge_vocab_shard_stats(gathered, out=out)


@pytest.mark.parametrize("save_probabilities", [False, True])
def test_noncontiguous_inputs_and_merged(save_probabilities):
    torch.manual_seed(91)
    x = torch.randn(2, 3).t().requires_grad_()
    w = torch.randn(2, 4).t().requires_grad_()
    b = torch.randn(8)[::2].requires_grad_()
    y = torch.tensor([0, 9, -100, 9, 3, 9])[::2]
    part = _part(
        hidden=x,
        weight=w,
        bias=b,
        target_ids=y,
        chunk_size=2,
        save_probabilities=save_probabilities,
    )
    merged = merge_vocab_shard_stats(part.stats[None])
    strided = torch.empty(3, 4)
    strided[:, ::2] = merged
    loss = _shard_finalize(part, strided[:, ::2], kind="ce")
    actual = torch.autograd.grad(loss, (x, w, b))
    ref = F.cross_entropy(F.linear(x, w, b), y, reduction="sum")
    expected = torch.autograd.grad(ref, (x, w, b))
    torch.testing.assert_close(loss, ref)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("vocab_block_size", [None, 128, 193])
@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("save_probabilities", [False, True])
def test_vocab_blocks_tail_bias_temperature(vocab_block_size, fp32_logits, save_probabilities):
    torch.manual_seed(61)
    x = (torch.randn(137, 13) / 4).requires_grad_()
    w = (torch.randn(517, 13) / 4).requires_grad_()
    b = (torch.randn(517) / 4).requires_grad_()
    y = torch.arange(137) * 19 % 517
    y[:6] = torch.tensor([-100, 127, 128, 385, 386, 516])
    original = [t.detach().clone() for t in (x, w, b, y)]
    part = _shard_partials(
        x,
        w,
        y,
        bias=b,
        vocab_start=0,
        vocab_size=w.shape[0],
        temperature=0.7,
        chunk_size=37,
        provider="torch",
        _allow_cpu=True,
        vocab_block_size=vocab_block_size,
        fp32_logits=fp32_logits,
        save_probabilities=save_probabilities,
    )
    actual = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="ce")
    got = torch.autograd.grad(actual * 0.37, (x, w, b))
    reference = [t.detach().double().requires_grad_() for t in (x, w, b)]
    expected = F.cross_entropy(F.linear(*reference) / 0.7, y, reduction="sum")
    want = torch.autograd.grad(expected * 0.37, reference)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-5)
    for grad, ref in zip(got, want, strict=True):
        assert grad.dtype == x.dtype
        torch.testing.assert_close(grad.double(), ref, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(got[0][0]) == 0
    for value, before in zip((x, w, b, y), original, strict=True):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("train_weight,train_bias", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("vocab_block_size", [None, 3])
def test_saved_trainable_head_saves_logits_without_recompute(
    monkeypatch, dtype, fp32_logits, train_weight, train_bias, vocab_block_size
):
    import ctikernels.vocab_parallel as vp

    torch.manual_seed(73)
    x = torch.randn(7, 5).to(dtype).requires_grad_()
    w = torch.randn(11, 5).to(dtype).requires_grad_(train_weight)
    b = torch.randn(11).to(dtype).requires_grad_(train_bias)
    y = torch.tensor([2, -100, 4, 5, 9, 12, 13])
    options = {
        "hidden": x,
        "weight": w,
        "bias": b,
        "target_ids": y,
        "vocab_start": 3,
        "vocab_size": 14,
        "temperature": 0.7,
        "chunk_size": 3,
        "fp32_logits": fp32_logits,
        "vocab_block_size": vocab_block_size,
    }
    inputs = [t for t in (x, w, b) if t.requires_grad]
    recompute = _part(**options)
    reference = _shard_finalize(
        recompute, merge_vocab_shard_stats(recompute.stats[None]), kind="ce"
    )
    expected = torch.autograd.grad(reference * 0.37, inputs)
    part = _part(**options, save_probabilities=True)
    logits = torch.cat(
        [
            F.linear(rows.float(), w.float(), b.float()) if fp32_logits else F.linear(rows, w, b)
            for rows in x.split(3)
        ]
    )
    assert part._saved_projection.dtype == (torch.float32 if fp32_logits else dtype)
    torch.testing.assert_close(part._saved_projection, logits, rtol=0, atol=0)
    assert part._average is None
    output = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="ce")

    def forbidden(*args, **kwargs):
        raise AssertionError("saved backward must not recompute logits")

    monkeypatch.setattr(vp, "_logits", forbidden)
    grads = torch.autograd.grad(output * 0.37, inputs, retain_graph=True)
    for got, want in zip(grads, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
    assert torch.count_nonzero(grads[0][1]) == 0
    with pytest.raises(RuntimeError, match="only one backward"):
        torch.autograd.grad(output, inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32_logits", [False, True])
def test_saved_frozen_head_still_saves_unnormalized_probabilities(dtype, fp32_logits):
    torch.manual_seed(74)
    x = torch.randn(3, 2).to(dtype).requires_grad_()
    w = torch.randn(4, 2).to(dtype)
    b = torch.randn(4).to(dtype)
    part = _part(
        hidden=x,
        weight=w,
        bias=b,
        kind="logprob",
        temperature=0.7,
        fp32_logits=fp32_logits,
        save_probabilities=True,
    )
    logits = F.linear(x.float(), w.float(), b.float()) if fp32_logits else F.linear(x, w, b)
    logits = logits.float() / 0.7
    expected = (logits - logits.max(1).values[:, None]).exp().to(dtype)
    expected[1] = 0
    torch.testing.assert_close(part._saved_projection, expected, rtol=0, atol=0)
    assert part._average is None


@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize(
    "kind,head_grad,dtype",
    [
        ("ce", True, torch.bfloat16),
        ("ce", False, torch.bfloat16),
        ("logprob", False, torch.bfloat16),
        ("logprob", False, torch.float16),
    ],
)
def test_low_precision_gradients_against_oracle(
    fp32_logits, save_probabilities, kind, head_grad, dtype, monkeypatch
):
    if kind == "ce":
        pytest.skip("TP1 CE uses a CUDA-only Liger kernel; CPU cannot produce this baseline")
    print(
        f"kind={kind} head_grad={head_grad} fp32_logits={fp32_logits} "
        f"save_probabilities={save_probabilities}"
    )
    torch.manual_seed(44)
    x = (torch.randn(37, 32) / 4).to(dtype)
    w = (torch.randn(517, 32) / 4).to(dtype)
    b = (torch.randn(517) / 4).to(dtype)
    y = torch.arange(37) * 19 % 517
    y[:5] = torch.tensor([-100, 127, 128, 258, 516])
    temperature = 0.7 if kind == "logprob" else 1.0
    upstream = torch.tensor(0.37) if kind == "ce" else torch.randn(37)

    def reference(reference_dtype, chunk_size=7, output_only=False):
        inputs = [
            value.to(reference_dtype)
            .detach()
            .requires_grad_((i == 0 or head_grad) and not output_only)
            for i, value in enumerate((x, w, b))
        ]
        if reference_dtype != torch.float64:
            op = ChunkedLinearSelectedTokenLogprobs(chunk_size=chunk_size)
            # Exercise TP1's actual math with CPU tensors; only CUDA validation is bypassed.
            monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
            # CPU tensors take the torch provider even in a CUDA build with CUDA hidden.
            monkeypatch.setattr(op, "_get_provider", lambda device: _TorchProvider())
            output = op.forward(
                inputs[0], inputs[1], y.clamp_min(0), bias=inputs[2], temperature=temperature
            )
            output = torch.where(y != -100, output, 0.0)
        else:
            logits = F.linear(*inputs)
            losses = F.cross_entropy(logits / temperature, y, reduction="none")
            output = losses.sum() if kind == "ce" else -losses
        if output_only:
            return output.detach()
        grads = torch.autograd.grad(
            output, inputs if head_grad else inputs[:1], upstream.to(output.dtype)
        )
        return (output.detach(), *grads)

    checks = []
    parts = []
    start = 0
    for ws in w.tensor_split(2):
        end = start + ws.shape[0]
        parts.append(
            _shard_partials(
                x.clone().requires_grad_(),
                ws.clone().requires_grad_(head_grad),
                y,
                bias=b[start:end].clone().requires_grad_(head_grad),
                vocab_start=start,
                vocab_size=w.shape[0],
                temperature=temperature,
                chunk_size=7,
                vocab_block_size=128,
                kind=kind,
                fp32_logits=fp32_logits,
                save_probabilities=save_probabilities,
                provider="torch",
                _allow_cpu=True,
            )
        )
        _check_stats(
            parts[-1],
            x,
            ws,
            b[start:end],
            y,
            start,
            temperature,
            fp32_logits,
            checks=checks,
            tp1=(w, b, 7, kind),
        )
        start = end
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    results, grads = [], []
    for part in parts:
        results.append(_shard_finalize(part, merged, kind=kind))
        variables = (part._hidden, part._weight, part._bias) if head_grad else (part._hidden,)
        grads.append(torch.autograd.grad(results[-1], variables, upstream))
        assert all(grad.dtype == dtype for grad in grads[-1])
    if kind == "logprob":
        assert all(torch.count_nonzero(result[y == -100]) == 0 for result in results)
    actual = [results[0].detach(), sum(g[0].float() for g in grads)]
    if head_grad:
        actual.extend(torch.cat([g[i] for g in grads]) for i in (1, 2))
    baseline, oracle = reference(dtype), reference(torch.float64)
    if fp32_logits:
        # Only the output uses FP32 TP1; gradient gates retain the head-dtype baseline.
        baseline = (reference(torch.float32, output_only=True), *baseline[1:])
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32_logits", [False, True])
def test_output_gate_policy_and_hard_failure_table(dtype, fp32_logits):
    tp1 = torch.tensor([1.0])
    oracle = torch.tensor([1.01], dtype=torch.float64)
    got = torch.tensor([1.001])
    checks = []
    _check_results((got,), (tp1,), (oracle,), dtype, checks=checks, fp32_logits=fp32_logits)
    # Oracle rule only for 16-bit logits; FP32 logits keep 1e-5 against TP1.
    head_dtype_mode = not fp32_logits and dtype != torch.float32
    assert checks[0]["passed"] == head_dtype_mode
    if head_dtype_mode:
        _finish(checks)
    else:
        with pytest.raises(pytest.fail.Exception, match="FAIL.*output"):
            _finish(checks)
    # An exact TP1 oracle match leaves zero allowance in the oracle rule.
    _check_results((got,), (tp1,), (tp1.double(),), dtype, checks=checks)
    with pytest.raises(pytest.fail.Exception, match="Strict vocab-parallel gates") as error:
        _finish(checks)
    assert str(error.value).count("output:") == 2
    assert checks[-1]["limit"] == (1e-5 if dtype == torch.float32 else 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_weight,train_bias", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("fp32_logits", [False, True])
@pytest.mark.parametrize("budget_offset", [-1, 0, 1, None, "zero"])
@pytest.mark.parametrize("override", [None, False, True])
def test_saved_budget_resolution(
    dtype, train_weight, train_bias, fp32_logits, budget_offset, override
):
    train_head = train_weight or train_bias
    element_size = 4 if train_head and fp32_logits else torch.empty((), dtype=dtype).element_size()
    size = 3 * 4 * element_size
    budget = (
        None if budget_offset is None else 0 if budget_offset == "zero" else size + budget_offset
    )
    part = _part(
        hidden=torch.randn(3, 2).to(dtype).requires_grad_(),
        weight=torch.randn(4, 2).to(dtype).requires_grad_(train_weight),
        bias=torch.randn(4).to(dtype).requires_grad_(train_bias),
        fp32_logits=fp32_logits,
        shard_saved_bytes_budget=budget,
        save_probabilities=override,
    )
    saved = override if override is not None else budget is not None and budget >= size
    assert (part._saved_projection is not None) == saved
    assert part.provider == ("torch:saved" if saved else "torch:recompute")
    assert part._provider == "torch"
    if saved:
        assert part._saved_projection.numel() * part._saved_projection.element_size() == size
    with pytest.raises(AttributeError):
        part.provider = "torch:saved"


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
@pytest.mark.parametrize("budget", [None, 0, 47, 48, 49])
def test_constructor_saved_budget_forwarding(monkeypatch, op_type, budget):
    import ctikernels.cross_entropy as ce
    import ctikernels.logprob.linear as lp

    def cpu_entry(*args, **kwargs):
        return _shard_partials(*args, **kwargs, _allow_cpu=True)

    monkeypatch.setattr(ce, "_shard_partials", cpu_entry)
    monkeypatch.setattr(lp, "_shard_partials", cpu_entry)
    op = op_type(provider="torch", shard_saved_bytes_budget=budget)
    part = op.shard_partials(
        torch.randn(3, 2),
        torch.randn(4, 2),
        torch.tensor([0, -100, 3]),
        vocab_start=0,
        vocab_size=4,
    )
    mode = "saved" if budget is not None and budget >= 48 else "recompute"
    assert part.provider == f"torch:{mode}"


@pytest.mark.parametrize(
    "op_type", [FusedLinearCrossEntropyLoss, ChunkedLinearSelectedTokenLogprobs]
)
@pytest.mark.parametrize(
    "value,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError), ("4", TypeError)]
)
def test_saved_budget_validation(op_type, value, error):
    with pytest.raises(error, match="shard_saved_bytes_budget"):
        op_type(provider="torch", shard_saved_bytes_budget=value)
    with pytest.raises(error, match="shard_saved_bytes_budget"):
        _part(shard_saved_bytes_budget=value)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("block", [None, 3])
def test_trainable_fp32_scales_accumulated_gradients(saved, block):
    torch.manual_seed(314)
    x = torch.randn(17, 9).requires_grad_()
    w = torch.randn(23, 9).requires_grad_()
    b = torch.randn(23).requires_grad_()
    y = torch.arange(17) % 23
    y[0] = -100

    def gradients(upstream):
        part = _part(
            hidden=x,
            weight=w,
            bias=b,
            target_ids=y,
            vocab_size=w.shape[0],
            save_probabilities=saved,
            vocab_block_size=block,
        )
        loss = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="ce")
        return torch.autograd.grad(loss, (x, w, b), torch.tensor(upstream))

    base, scaled = gradients(1.0), gradients(0.375)
    for got, unscaled in zip(scaled, base, strict=True):
        torch.testing.assert_close(
            got, unscaled * 0.375, rtol=torch.finfo(torch.float32).eps, atol=0
        )


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("fp32_logits", [False, True])
def test_trainable_bf16_bias_scales_fp32_sum_before_cast(saved, fp32_logits):
    torch.manual_seed(315)
    x = torch.randn(17, 9).bfloat16().requires_grad_()
    w = torch.randn(23, 9).bfloat16().requires_grad_()
    b = torch.randn(23).bfloat16().requires_grad_()
    y = torch.arange(17) % 23
    y[0] = -100
    part = _part(
        hidden=x,
        weight=w,
        bias=b,
        target_ids=y,
        vocab_size=w.shape[0],
        save_probabilities=saved,
        fp32_logits=fp32_logits,
    )
    merged = merge_vocab_shard_stats(part.stats[None])
    logits = F.linear(x.float(), w.float(), b.float()) if fp32_logits else F.linear(x, w, b)
    dz = (logits.float() - merged[:, :1]).exp()
    dz[torch.arange(17), y.clamp_min(0)] -= 1
    dz[0] = 0
    expected = (dz.bfloat16().sum(0, dtype=torch.float32) * 0.375).bfloat16()
    prescaled = (dz * 0.375).bfloat16().sum(0, dtype=torch.float32).bfloat16()
    assert not torch.equal(expected, prescaled), "fixture must distinguish upstream placement"
    loss = _shard_finalize(part, merged, kind="ce")
    got = torch.autograd.grad(loss, b, torch.tensor(0.375))[0]
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("budget", [None, 0, 1])
def test_empty_logprob_budget_policy(budget):
    part = _part(
        hidden=torch.empty(0, 2, requires_grad=True),
        target_ids=torch.empty(0, dtype=torch.int64),
        kind="logprob",
        shard_saved_bytes_budget=budget,
    )
    assert part.provider == ("torch:saved" if budget == 1 else "torch:recompute")
    result = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="logprob")
    result.sum().backward()
    assert part._hidden.grad.shape == (0, 2)


def test_auto_without_triton_is_a_visible_reroute(monkeypatch, caplog):
    from ctikernels import _reroute, vocab_parallel

    monkeypatch.setattr(vocab_parallel, "_triton_available", lambda: False)
    monkeypatch.setattr(_reroute, "_WARNED", set())
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    with caplog.at_level("WARNING", logger=_reroute.__name__):
        first = vocab_parallel._resolve_provider("auto", True)
        second = vocab_parallel._resolve_provider("auto", True)
    assert first == second == "torch"
    assert first.reason == "Triton is not installed"
    assert len(caplog.records) == 1
    assert vocab_parallel._resolve_provider("auto", False).reason is None
    assert vocab_parallel._resolve_provider("torch", True).reason is None
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(_reroute.RerouteError, match="Triton is not installed"):
        vocab_parallel._resolve_provider("auto", True)


def test_part_provider_carries_reroute_reason(monkeypatch):
    from ctikernels import vocab_parallel
    from ctikernels._reroute import Provider

    torch.manual_seed(3)
    part = _shard_partials(
        torch.randn(3, 4),
        torch.randn(5, 4),
        torch.tensor([0, 1, 2]),
        vocab_start=0,
        vocab_size=5,
        provider="torch",
        _allow_cpu=True,
    )
    assert part.provider == "torch:recompute" and part.provider.reason is None
    monkeypatch.setattr(part, "_provider", Provider("torch", "Triton is not installed"))
    assert part.provider == "torch:recompute"
    assert part.provider.reason == "Triton is not installed"
    assert vocab_parallel.Provider is Provider


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("n,ignored", [(0, False), (1, False), (7, False), (7, True)])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("saved", [False, True])
def test_ce_reductions_strided_rows(reduction, n, ignored, train_head, saved):
    torch.manual_seed(131)
    x = (torch.randn(n, 10) / 4)[:, :5].requires_grad_()
    w = (torch.randn(7, 5) / 4).requires_grad_(train_head)
    b = (torch.randn(7) / 4).requires_grad_(train_head)
    y = torch.arange(n * 3)[::3] % 7
    # Keep an actual row stride through the target view.
    target_storage = torch.empty(n * 3, dtype=torch.int64)
    target_storage[::3] = y
    y = target_storage[::3]
    y[::3] = -7
    if ignored:
        y.fill_(-7)
    parts = []
    for start, end in [(0, 2), (2, 4), (4, 6), (6, 7)]:
        parts.append(
            _shard_partials(
                x,
                w[start:end],
                y,
                bias=b[start:end],
                vocab_start=start,
                vocab_size=7,
                reduction=reduction,
                ignore_index=-7,
                save_probabilities=saved,
                provider="torch",
                _allow_cpu=True,
                chunk_size=3,
            )
        )
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    outputs = [_shard_finalize(p, merged, kind="ce") for p in parts]
    reference = F.cross_entropy(F.linear(x, w, b), y, ignore_index=-7, reduction=reduction)
    for output in outputs:
        torch.testing.assert_close(output, reference, equal_nan=True)
    upstream = torch.randn(n * 2)[::2] if reduction == "none" else torch.tensor(0.37)
    inputs = (x, w, b) if train_head else (x,)
    got = torch.autograd.grad(sum(outputs), inputs, upstream)
    expected = torch.autograd.grad(reference, inputs, upstream)
    for actual, want in zip(got, expected, strict=True):
        torch.testing.assert_close(actual, want, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("tensor_temperature", [False, True])
def test_temperature_arithmetic_and_strides(saved, tensor_temperature):
    from ctikernels.vocab_parallel import _scaled_logits, _torch_stats

    torch.manual_seed(19)
    x = torch.randn(7, 10)[:, :5].requires_grad_()
    w = torch.randn(11, 5) / 4
    t = torch.linspace(0.3, 1.7, 14)[::2] if tensor_temperature else 0.7
    y = torch.arange(14, dtype=torch.int64)[::2] % 11
    z = x.detach() @ w.t()
    scaled = z.float() / (t[:, None] if tensor_temperature else t)
    torch.testing.assert_close(_scaled_logits(z, t), scaled, rtol=0, atol=0)
    stats = torch.empty(7, 3)
    probabilities = torch.empty_like(z)
    _torch_stats(z, y, 0, stats, probabilities, t, -100)
    maximum = scaled.max(1).values
    torch.testing.assert_close(stats[:, 0], maximum, rtol=0, atol=0)
    torch.testing.assert_close(stats[:, 2], scaled.gather(1, y[:, None])[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(probabilities, (scaled - maximum[:, None]).exp(), rtol=0, atol=0)
    part = _shard_partials(
        x,
        w,
        y,
        vocab_start=0,
        vocab_size=11,
        temperature=t,
        kind="logprob",
        chunk_size=3,
        provider="torch",
        _allow_cpu=True,
        save_probabilities=saved,
    )
    output = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="logprob")
    expected = (F.linear(x, w).float() / (t[:, None] if tensor_temperature else t)).log_softmax(1)
    expected = expected.gather(1, y[:, None])[:, 0]
    upstream = torch.randn(14)[::2]
    (got_dx,) = torch.autograd.grad(output, x, upstream)
    (ref_dx,) = torch.autograd.grad(expected, x, upstream)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(got_dx, ref_dx, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "temperature,error",
    [
        (torch.ones(2), ValueError),
        (torch.ones(3, 1), ValueError),
        (torch.ones(3, dtype=torch.float64), TypeError),
        (torch.ones(3, device="meta"), ValueError),
    ],
)
def test_tensor_temperature_metadata(temperature, error):
    with pytest.raises(error, match="temperature"):
        _part(kind="logprob", temperature=temperature)


def test_temperature_values_only_checked_under_engine_checks(monkeypatch):
    for invalid in [0.0, -1.0, float("inf"), float("nan")]:
        temperature = torch.tensor([1.0, invalid, 0.7])
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
        with pytest.raises(ValueError, match="temperature"):
            _part(kind="logprob", temperature=temperature)
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
        _part(kind="logprob", temperature=temperature)
    with pytest.raises(TypeError, match="CE temperature must be a scalar"):
        _part(temperature=torch.ones(3))


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("saved", [False, True])
def test_empty_ce_has_no_projection_or_kernel(monkeypatch, reduction, saved):
    import ctikernels.vocab_parallel as vp

    def forbidden(*args, **kwargs):
        raise AssertionError("empty CE must not launch a projection or Triton kernel")

    monkeypatch.setattr(vp, "_logits", forbidden)
    monkeypatch.setattr(vp, "_mm", forbidden)
    monkeypatch.setattr(vp, "_triton_kernels", forbidden)
    x, w, b = (
        torch.empty(0, 3, requires_grad=True),
        torch.randn(1, 3, requires_grad=True),
        torch.randn(1, requires_grad=True),
    )
    part = _shard_partials(
        x,
        w,
        torch.empty(0, dtype=torch.int64),
        bias=b,
        vocab_start=0,
        vocab_size=1,
        reduction=reduction,
        save_probabilities=saved,
        provider="torch",
        _allow_cpu=True,
    )
    assert part.stats.shape == (0, 3)
    merged = merge_vocab_shard_stats(part.stats[None])
    maximum = torch.empty(0)
    sums = vocab_shard_sum_terms(part.stats, maximum)
    assert vocab_shard_merged_from_sums(maximum, sums).shape == (0, 2)
    output = _shard_finalize(part, merged, kind="ce")
    output.sum().backward()
    assert x.grad.shape == (0, 3)
    assert torch.count_nonzero(w.grad) == torch.count_nonzero(b.grad) == 0
    if reduction == "mean":
        assert output.isnan()
    elif reduction == "sum":
        assert output == 0
    else:
        assert output.shape == (0,)


def test_public_ce_options_and_logprob_tensor_temperature(monkeypatch):
    import ctikernels.cross_entropy as ce
    import ctikernels.logprob.linear as lp

    def cpu_entry(*args, **kwargs):
        return _shard_partials(*args, **kwargs, _allow_cpu=True)

    monkeypatch.setattr(ce, "_shard_partials", cpu_entry)
    monkeypatch.setattr(lp, "_shard_partials", cpu_entry)
    x = torch.randn(3, 2, requires_grad=True)
    w = torch.randn(4, 2)
    y = torch.tensor([0, -7, 3])
    op = FusedLinearCrossEntropyLoss(provider="torch", ignore_index=-7, reduction="none")
    part = op.shard_partials(x, w, y, vocab_start=0, vocab_size=4)
    loss = op.shard_finalize(part, merge_vocab_shard_stats(part.stats[None]))
    torch.testing.assert_close(
        loss, F.cross_entropy(x @ w.t(), y, ignore_index=-7, reduction="none")
    )
    temperature = torch.tensor([0.7, 1.0, 1.3])
    with pytest.raises(TypeError, match="CE temperature must be a scalar"):
        op.shard_partials(x, w, y, vocab_start=0, vocab_size=4, temperature=temperature)
    op = ChunkedLinearSelectedTokenLogprobs(provider="torch", ignore_index=-7)
    part = op.shard_partials(x, w, y, vocab_start=0, vocab_size=4, temperature=temperature)
    assert part._temperature is temperature
    output = op.shard_finalize(part, merge_vocab_shard_stats(part.stats[None]))
    expected = -F.cross_entropy(
        (x @ w.t()) / temperature[:, None], y, ignore_index=-7, reduction="none"
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_ce_accepts_all_head_dtypes(dtype, reduction):
    x = torch.zeros(3, 2, dtype=dtype, requires_grad=True)
    w = torch.zeros(4, 2, dtype=dtype, requires_grad=True)
    b = torch.zeros(4, dtype=dtype, requires_grad=True)
    y = torch.tensor([0, -100, 3])
    part = _part(hidden=x, weight=w, bias=b, target_ids=y, reduction=reduction)
    output = _shard_finalize(part, merge_vocab_shard_stats(part.stats[None]), kind="ce")
    reference = F.cross_entropy(F.linear(x, w, b).float(), y, reduction=reduction)
    torch.testing.assert_close(output, reference)
    got = torch.autograd.grad(output.sum(), (x, w, b))
    expected = torch.autograd.grad(reference.sum(), (x, w, b))
    for actual, want in zip(got, expected, strict=True):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
