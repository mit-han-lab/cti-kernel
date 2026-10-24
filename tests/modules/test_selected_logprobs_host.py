# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host contract tests; never initialize CUDA."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.logprob import ChunkedLinearSelectedTokenLogprobs
from ctikernels.logprob.cute import _provider
from ctikernels.logprob.linear import _ChunkedLinearSelectedLogprobsFunction, _TorchProvider


@pytest.mark.parametrize("rows", (0, 1, 2, 3, 4, 8, 64, 65, 4096))
@pytest.mark.parametrize("vocab", (1, 97, 4099, 151936, 248320, 2**31 - 1))
def test_split_geometry(rows, vocab):
    splits = _provider.choose_splits(rows, vocab, 132)
    if rows == 0:
        assert splits == 0
        return
    assert splits >= 1
    for vector in (4, 8):
        extent = (vocab + vector - 1 + splits - 1) // splits
        quantum = 128 * vector
        tile = (extent + quantum - 1) // quantum * quantum
        assert tile <= 16384
        assert tile * splits >= vocab + vector - 1
    if rows <= 8 and vocab >= 151936:
        assert rows * splits >= 132
    if vocab == 97:
        assert splits == 1


@pytest.mark.parametrize("temperature", (0.7, 1.0, 1e-3, 1e4))
def test_python_temperature_is_rounded_to_float32_before_division(temperature):
    values = torch.tensor([1.0, 3.1415927, 12345.67, -0.1234567])
    result = values / temperature
    scalar = torch.tensor(temperature, dtype=torch.float32)
    # FP64 division of FP32 operands, rounded once, is the div.rn oracle here.
    expected = (values.double() / scalar.double()).float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, values / scalar, rtol=0, atol=0)
    if temperature == 1e-3:
        assert not torch.equal(result, values * (1.0 / temperature))


@pytest.mark.parametrize(
    "temperature", (0.7, 1.0, 3.0, 1e-3, 1e4, 1e-45, 1e-50, 3.5e38, 1e39, 1e-300)
)
def test_inverse_temperature_is_the_fp32_cast_of_the_double_reciprocal(temperature):
    # CUDA scales by this reciprocal (BinaryDivTrueKernel); CPU torch divides.
    expected = torch.tensor(1.0 / temperature, dtype=torch.float64).float().item()
    assert _provider.inverse_temperature(temperature) == expected
    assert _provider.inverse_temperature(1e-3) == 1000.0


def test_inverse_temperature_stays_on_cpu_under_another_default_device():
    # A non-CPU default device must not move the cast (meta tensors cannot .item()).
    expected = torch.tensor(1 / 0.37, device="cpu").float().item()
    _provider.inverse_temperature.cache_clear()
    torch.set_default_device("meta")
    try:
        assert _provider.inverse_temperature(0.37) == expected
    finally:
        torch.set_default_device(None)


@pytest.mark.parametrize(
    "importable,major,expected", ((True, 8, "cute"), (True, 7, "torch"), (False, 9, "torch"))
)
def test_provider_choice_is_cached_per_device(monkeypatch, importable, major, expected):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    import ctikernels.logprob.cute as cute_provider

    monkeypatch.setattr(
        cute_provider, "unavailable_reason", lambda: None if importable else "CuTe unavailable"
    )
    constructor = Mock(side_effect=lambda device: SimpleNamespace(name="cute", device=device))
    monkeypatch.setattr(cute_provider, "Provider", constructor)
    capability = Mock(return_value=(major, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    op = ChunkedLinearSelectedTokenLogprobs()
    first = op._get_provider(torch.device("cuda:0"))
    assert op._get_provider(torch.device("cuda:0")) is first
    assert op.provider == expected
    assert (op.provider.reason is None) == (expected == "cute")
    second = op._get_provider(torch.device("cuda:1"))
    assert second is not first
    assert capability.call_count == (2 if importable else 0)
    assert constructor.call_count == (2 if expected == "cute" else 0)


def test_torch_fallback_chunking_and_backward_on_cpu():
    hidden = torch.randn(7, 5, requires_grad=True)
    weight = torch.randn(11, 5)
    targets = torch.arange(7, dtype=torch.int64)
    provider = Mock(wraps=_TorchProvider())
    actual = _ChunkedLinearSelectedLogprobsFunction.apply(
        hidden, weight, targets, None, 0.7, 3, provider, -100
    )
    reference = (hidden @ weight.T / 0.7).log_softmax(-1).gather(1, targets[:, None])[:, 0]
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn(7)
    got = torch.autograd.grad(actual, hidden, upstream)[0]
    expected = torch.autograd.grad(reference, hidden, upstream)[0]
    torch.testing.assert_close(got, expected)
    assert [call.args[0].shape[0] for call in provider.call_args_list] == [3, 3, 1]


def test_empty_batch_never_calls_provider():
    hidden = torch.empty(0, 4, requires_grad=True)
    provider = Mock(side_effect=AssertionError("empty batch launched"))
    out = _ChunkedLinearSelectedLogprobsFunction.apply(
        hidden,
        torch.empty(1, 4),
        torch.empty(0, dtype=torch.int64),
        None,
        1.0,
        64,
        provider,
        -100,
    )
    out.sum().backward()
    assert out.shape == (0,) and out.dtype == torch.float32
    assert hidden.grad.shape == hidden.shape
    provider.assert_not_called()


def test_special_value_contract():
    logits = torch.tensor(
        [
            [0.0, -float("inf"), 1.0],
            [-float("inf")] * 3,
            [0.0, float("inf"), 1.0],
            [0.0, float("nan"), 1.0],
        ]
    )
    target = torch.tensor([1, 0, 0, 0])
    output = torch.empty(4)
    _TorchProvider()(logits, target, output, 1.0)
    assert output[0].isneginf() and output[2].isneginf()
    assert output[1].isnan() and output[3].isnan()
    # torch.log_softmax differs from the existing gather-minus-logsumexp at +inf.
    assert logits.log_softmax(-1)[2].isnan().all()


def test_workspace_and_compilation_are_reused_and_capture_never_grows(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132, major=8, minor=0),
    )
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    compiled = Mock()
    compiler = Mock(return_value=compiled)
    monkeypatch.setattr(_provider, "compile_kernel", compiler)
    empty, zeros = torch.empty, torch.zeros
    allocate = Mock(side_effect=lambda *args, **kwargs: empty(*args, **{**kwargs, "device": "cpu"}))
    initialize = Mock(
        side_effect=lambda *args, **kwargs: zeros(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch, "zeros", initialize)
    provider = _provider.Provider(torch.device("cuda:0"))
    logits, targets, output = empty(1, 97), zeros(1, dtype=torch.int64), empty(1)
    provider(logits, targets, output, 0.7)
    original_workspace = provider._workspace
    capturing = True
    provider(logits, targets, output, 1e-3)
    assert provider._workspace is original_workspace
    assert compiler.call_count == allocate.call_count == initialize.call_count == 1
    # The kernel receives PyTorch CUDA's FP32 reciprocal, not T.
    assert compiled.call_args.args[-2] == _provider.inverse_temperature(1e-3)
    assert compiled.call_args.args[-1] == -100
    provider(logits, targets, output, 1e-3, 5)
    assert compiled.call_args.args[-1] == 5
    assert compiler.call_count == 1
    larger = empty(65, 97)
    with pytest.raises(RuntimeError, match="warm selected logprobs"):
        provider(larger, zeros(65, dtype=torch.int64), empty(65), 1.0)
    with pytest.raises(RuntimeError, match="warm selected logprobs"):
        provider(empty(1, 98), targets, output, 1.0)
    assert compiler.call_count == allocate.call_count == initialize.call_count == 1
    capturing = False
    provider(larger, zeros(65, dtype=torch.int64), empty(65), 1.0)
    assert provider._retired_workspaces[0] is original_workspace
    assert allocate.call_count == initialize.call_count == 2
    # Dynamic row count shares this single-split specialization.
    assert compiler.call_count == 1
    temperature = torch.linspace(0.3, 1.7, 130)[::2]
    provider(larger, zeros(65, dtype=torch.int64), empty(65), temperature)
    assert compiler.call_count == 2
    assert compiler.call_args.args[3] is True
    assert compiled.call_args.args[-3] is temperature
    capturing = True
    provider(larger, zeros(65, dtype=torch.int64), empty(65), temperature * 2)
    assert compiler.call_count == 2


@pytest.mark.parametrize("trainable", ("weight", "bias"))
def test_trainable_head_is_rejected_before_device_dispatch(trainable):
    hidden = torch.randn(1, 4)
    weight = torch.randn(8, 4, requires_grad=trainable == "weight")
    bias = torch.randn(8, requires_grad=trainable == "bias")
    with pytest.raises(ValueError, match="frozen vocabulary head"):
        ChunkedLinearSelectedTokenLogprobs().forward(
            hidden,
            weight,
            torch.zeros(1, dtype=torch.int64),
            bias=bias,
        )


def test_benchmark_pairs_alternate_order_and_report_paired_ratios(monkeypatch):
    from benchmark.logprob import selected_bench

    calls = []
    times = {"ours": 1.0, "sglang": 2.0, "vllm": 3.0, "old": 4.0}
    monkeypatch.setattr(selected_bench, "capture", lambda function: (function(), None))

    def elapsed(name, repeats):
        calls.append(name)
        return times[name]

    monkeypatch.setattr(selected_bench, "elapsed", elapsed)
    result = selected_bench.measure({name: lambda name=name: name for name in times}, 20, 1)
    assert result["speedup"] == 2.0
    assert result["ci95"] == [2.0, 2.0]
    assert result["aa_epsilon"] == 0.0
    assert result["fastest_comparator"] == "sglang"
    assert result["median_us"] == times
    for round_index in range(20):
        pairs = calls[round_index * 8 : (round_index + 1) * 8]
        for offset in range(0, 8, 2):
            assert pairs[offset + round_index % 2] == "ours"
    assert calls[:2] == ["ours", "sglang"]
    assert calls[8:10] == ["vllm", "ours"]


@pytest.mark.parametrize("value", (True, False, 1.5, "-100", None, torch.tensor(-100)))
def test_ignore_index_requires_plain_int(value):
    with pytest.raises(TypeError, match="ignore_index"):
        ChunkedLinearSelectedTokenLogprobs(ignore_index=value)


@pytest.mark.parametrize("value", (-100, -1, 5))
def test_ignore_index_is_read_only(value):
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=value)
    assert op.ignore_index == value
    assert ChunkedLinearSelectedTokenLogprobs().ignore_index == -100
    with pytest.raises(AttributeError):
        op.ignore_index = 3


@pytest.mark.parametrize("cause", ("version", "capability"))
def test_reroute_reason_checks_and_one_log(monkeypatch, caplog, cause):
    import ctikernels._reroute as reroutes
    import ctikernels.logprob.cute as cute

    monkeypatch.setattr(reroutes, "_WARNED", set())
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    _provider.unavailable_reason.cache_clear()
    if cause == "version":
        monkeypatch.setattr(_provider.importlib.metadata, "version", lambda name: "4.7.0")
        reason = "CuTe requires nvidia-cutlass-dsl 4.8.0; found 4.7.0"
        assert not _provider.available()
    else:
        monkeypatch.setattr(cute, "unavailable_reason", lambda: None)
        reason = "CuTe requires capability >= 8.0; got 7.5"
    try:
        op = ChunkedLinearSelectedTokenLogprobs()
        for _ in range(2):
            with pytest.raises(reroutes.RerouteError, match=reason):
                op._get_provider(torch.device("cuda:0"))
            assert op._providers == {}
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
        with caplog.at_level("WARNING", logger="ctikernels._reroute"):
            for instance in (op, op, ChunkedLinearSelectedTokenLogprobs()):
                instance._get_provider(torch.device("cuda:0"))
                assert instance.provider == "torch"
                assert instance.provider.reason == reason
        assert sum("selected_logprobs rerouted" in record.message for record in caplog.records) == 1
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
        with pytest.raises(reroutes.RerouteError):
            op._get_provider(torch.device("cuda:0"))
    finally:
        _provider.unavailable_reason.cache_clear()


def test_missing_cutlass_has_visible_reason(monkeypatch):
    def missing(name):
        raise _provider.importlib.metadata.PackageNotFoundError(name)

    _provider.unavailable_reason.cache_clear()
    monkeypatch.setattr(_provider.importlib.metadata, "version", missing)
    try:
        assert "PackageNotFoundError" in _provider.unavailable_reason()
        assert not _provider.available()
    finally:
        _provider.unavailable_reason.cache_clear()


@pytest.mark.parametrize("ignore_index", (-100, -1, 5))
@pytest.mark.parametrize("all_ignored", (False, True))
def test_torch_ignore_forward_and_backward(ignore_index, all_ignored):
    torch.manual_seed(53)
    hidden = torch.randn(7, 5, requires_grad=True)
    weight, bias = torch.randn(11, 5), torch.randn(11)
    targets = torch.tensor([ignore_index, 0, 10, ignore_index, 2, 3, ignore_index])
    if all_ignored:
        targets.fill_(ignore_index)
    valid = targets != ignore_index
    actual = _ChunkedLinearSelectedLogprobsFunction.apply(
        hidden, weight, targets, bias, 0.7, 3, _TorchProvider(), ignore_index
    )
    reference = (hidden @ weight.T + bias).div(0.7).log_softmax(-1)
    reference = torch.where(valid, reference.gather(1, targets.clamp_min(0)[:, None])[:, 0], 0.0)
    upstream = torch.randn(7)
    got = torch.autograd.grad(actual, hidden, upstream)[0]
    expected = torch.autograd.grad(reference, hidden, upstream)[0]
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(got, expected)
    assert torch.count_nonzero(actual[~valid]) == 0
    assert torch.count_nonzero(got[~valid]) == 0


@pytest.mark.parametrize("ignore_index", (-100, -1, 5))
@pytest.mark.parametrize("save", (False, True))
def test_shard_ignore_is_carried_through_finalize_and_backward(monkeypatch, ignore_index, save):
    import ctikernels.logprob.linear as linear
    from ctikernels.vocab_parallel import _shard_partials, merge_vocab_shard_stats

    monkeypatch.setattr(
        linear, "_shard_partials", lambda *a, **kw: _shard_partials(*a, **kw, _allow_cpu=True)
    )
    torch.manual_seed(57)
    hidden = torch.randn(5, 4)
    weight = torch.randn(12, 4)
    targets = torch.tensor([ignore_index, 0, 11, 3, ignore_index])
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=ignore_index, provider="torch")
    inputs = [hidden.clone().requires_grad_() for _ in range(2)]
    parts = [
        op.shard_partials(
            x,
            weight[start : start + 6],
            targets,
            vocab_start=start,
            vocab_size=12,
            save_probabilities=save,
        )
        for x, start in zip(inputs, (0, 6), strict=True)
    ]
    assert all(p._ignore_index == ignore_index for p in parts)
    # Finalize uses the partial's ignore value, not the receiving op's configuration.
    finalizer = ChunkedLinearSelectedTokenLogprobs(ignore_index=99)
    merged = merge_vocab_shard_stats(torch.stack([p.stats for p in parts]))
    outputs = [finalizer.shard_finalize(p, merged) for p in parts]
    grads = [torch.autograd.grad(out.sum(), x)[0] for out, x in zip(outputs, inputs, strict=True)]
    reference_hidden = hidden.clone().requires_grad_()
    valid = targets != ignore_index
    expected = (
        (reference_hidden @ weight.T).log_softmax(-1).gather(1, targets.clamp_min(0)[:, None])[:, 0]
    )
    expected = torch.where(valid, expected, 0.0)
    expected_grad = torch.autograd.grad(expected.sum(), reference_hidden)[0]
    for out, grad, part in zip(outputs, grads, parts, strict=True):
        torch.testing.assert_close(out, expected)
        assert torch.count_nonzero(out[~valid]) == torch.count_nonzero(grad[~valid]) == 0
        torch.testing.assert_close(
            part.stats[~valid], torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1)
        )
    torch.testing.assert_close(sum(grads), expected_grad)


@pytest.mark.parametrize("bad", (-100, -2, 11))
def test_tp1_checks_before_provider_dispatch(monkeypatch, bad):
    op = ChunkedLinearSelectedTokenLogprobs(ignore_index=-1)
    monkeypatch.setattr(op, "_validate_inputs", lambda *a, **kw: None)
    provider = Mock(return_value=_TorchProvider())
    monkeypatch.setattr(op, "_get_provider", provider)
    hidden, weight = torch.randn(4, 3), torch.randn(11, 3)
    targets = torch.tensor([-1, 0, bad, -2])
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="row 2"):
        op.forward(hidden, weight, targets)
    provider.assert_not_called()
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    op.forward(hidden, weight, targets)
    provider.assert_called_once()


@pytest.mark.parametrize("vocab_size,bad", ((12, -2), (12, 12)))
def test_shard_target_checks(monkeypatch, vocab_size, bad):
    from ctikernels.vocab_parallel import _shard_partials

    args = (torch.randn(3, 4), torch.randn(6, 4), torch.tensor([-1, 0, bad]))
    kwargs = {
        "vocab_start": 6,
        "vocab_size": vocab_size,
        "ignore_index": -1,
        "kind": "logprob",
        "provider": "torch",
        "_allow_cpu": True,
    }
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="row 2"):
        _shard_partials(*args, **kwargs)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    _shard_partials(*args, **kwargs)


def test_shard_vocab_size_and_global_positive_ids(monkeypatch):
    from ctikernels.vocab_parallel import _shard_partials

    args = (torch.randn(2, 4), torch.randn(6, 4), torch.tensor([-1, 999]))
    kwargs = {"vocab_start": 6, "ignore_index": -1, "kind": "logprob", "_allow_cpu": True}
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    _shard_partials(*args, **kwargs, vocab_size=1000)
    with pytest.raises(ValueError, match="row 1"):
        _shard_partials(*args, **kwargs, vocab_size=999)
    with pytest.raises(ValueError, match="vocab_size"):
        _shard_partials(*args, **kwargs, vocab_size=11)
    with pytest.raises(TypeError, match="vocab_size"):
        _shard_partials(*args, **kwargs, vocab_size=True)


def test_target_checks_do_not_read_ids_during_capture(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.vocab_parallel import _validate_target_ids

    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    targets = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"), detach=Mock())
    _validate_target_ids(targets, -100, 11)
    targets.detach.assert_not_called()


@pytest.mark.parametrize("value", (float("inf"), -float("inf"), float("nan")))
def test_ignored_rows_stay_zero_with_nonfinite_head(value):
    hidden = torch.ones(2, 3, requires_grad=True)
    weight = torch.full((7, 3), value)
    targets = torch.full((2,), -100, dtype=torch.int64)
    actual = _ChunkedLinearSelectedLogprobsFunction.apply(
        hidden, weight, targets, None, 1.0, 64, _TorchProvider(), -100
    )
    actual.sum().backward()
    assert torch.count_nonzero(actual) == 0
    assert torch.count_nonzero(hidden.grad) == 0


@pytest.mark.parametrize("rows", (0, 1, 2, 7))
@pytest.mark.parametrize("stride", (1, 3))
def test_row_temperature_chunking_and_backward(monkeypatch, rows, stride):
    torch.manual_seed(811)
    hidden = torch.randn(rows, 5, requires_grad=True)
    weight = torch.randn(11, 5)
    bias = torch.randn(11)
    targets = torch.arange(rows) % 11
    targets[1::3] = -7
    temperature = torch.linspace(0.3, 1.7, rows * stride)[::stride]
    op = ChunkedLinearSelectedTokenLogprobs(chunk_size=3, ignore_index=-7)
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    provider = Mock(wraps=_TorchProvider())
    monkeypatch.setattr(op, "_get_provider", lambda device: provider)
    actual = op.forward(hidden, weight, targets, bias=bias, temperature=temperature)
    logits = (hidden @ weight.T + bias) / temperature[:, None]
    selected = logits.log_softmax(-1).gather(1, targets.clamp_min(0)[:, None])[:, 0]
    reference = torch.where(targets != -7, selected, 0.0)
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn(rows * 2)[::2]
    got = torch.autograd.grad(actual, hidden, upstream)[0]
    expected = torch.autograd.grad(reference, hidden, upstream)[0]
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    for index, call in enumerate(provider.call_args_list):
        torch.testing.assert_close(call.args[3], temperature[index * 3 : index * 3 + 3])
        assert call.args[3].stride() == temperature.stride()


@pytest.mark.parametrize(
    "temperature,error",
    [
        (torch.ones(2), ValueError),
        (torch.ones(3, 1), ValueError),
        (torch.ones(3, dtype=torch.float64), TypeError),
        (torch.ones(3, dtype=torch.float16), TypeError),
        (torch.ones(3, device="meta"), ValueError),
        (True, TypeError),
        (0.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_tp1_temperature_validation(monkeypatch, temperature, error):
    op = ChunkedLinearSelectedTokenLogprobs()
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    provider = Mock(side_effect=AssertionError("invalid temperature reached dispatch"))
    monkeypatch.setattr(op, "_get_provider", provider)
    with pytest.raises(error, match="temperature"):
        op.forward(
            torch.zeros(3, 2),
            torch.zeros(4, 2),
            torch.zeros(3, dtype=torch.int64),
            temperature=temperature,
        )
    provider.assert_not_called()


@pytest.mark.parametrize("invalid", (0.0, -1.0, float("inf"), float("nan")))
def test_tp1_temperature_values_checked_only_in_eager_checks(monkeypatch, invalid):
    op = ChunkedLinearSelectedTokenLogprobs()
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    provider = Mock(wraps=_TorchProvider())
    monkeypatch.setattr(op, "_get_provider", lambda device: provider)
    args = (torch.zeros(3, 2), torch.zeros(4, 2), torch.zeros(3, dtype=torch.int64))
    temperature = torch.tensor([0.7, invalid, 1.3])
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(ValueError, match="temperature"):
        op.forward(*args, temperature=temperature)
    provider.assert_not_called()
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    op.forward(*args, temperature=temperature)
    provider.assert_called_once()


def test_cute_single_row_duplicates_temperature(monkeypatch):
    op = ChunkedLinearSelectedTokenLogprobs(chunk_size=2)
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    provider = Mock(wraps=_TorchProvider())
    provider.name = "cute"
    monkeypatch.setattr(op, "_get_provider", lambda device: provider)
    temperature = torch.tensor([0.7, 1.3, 0.9])
    op.forward(torch.randn(3, 2), torch.randn(4, 2), torch.arange(3), temperature=temperature)
    assert provider.call_count == 2
    assert provider.call_args.args[0].shape == (2, 4)
    torch.testing.assert_close(provider.call_args.args[3], temperature[-1:].expand(2))
