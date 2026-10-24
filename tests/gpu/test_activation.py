# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from ctikernels import _flashinfer_version
from ctikernels._norm_act_reroute import RerouteError
from ctikernels.activation import inference as inference_activation
from ctikernels.activation import training as training_activation


def _reference_activation(name: str, x: torch.Tensor) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    if name == "silu":
        output = F.silu(gate) * up
    elif name == "gelu":
        output = F.gelu(gate) * up
    else:
        raise AssertionError(f"unexpected activation {name}")
    return output.to(x.dtype)


def _assert_numerically_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if expected.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        return

    difference = actual.detach().float() - expected.detach().float()
    reference = expected.detach().float()
    error_rms = float(difference.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )
    allowed_rms = 1e-4 + 4e-3 * reference_rms
    assert error_rms <= allowed_rms, (
        f"RMS error {error_rms} exceeds {allowed_rms}; "
        f"reference_rms={reference_rms}, max_absolute={float(difference.abs().max())}"
    )
    assert relative_l2 <= 4e-3, (
        f"relative L2 error {relative_l2} exceeds 0.004; "
        f"max_absolute={float(difference.abs().max())}"
    )


@pytest.mark.parametrize("shape", ((17, 256), (2, 5, 6144)))
def test_training_silu_fp32_fallback_matches_native_eager_exactly(
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(20260729)
    x_data = torch.randn(shape, device="cuda", dtype=torch.float32)
    upstream = torch.randn(
        shape[:-1] + (shape[-1] // 2,),
        device="cuda",
        dtype=torch.float32,
    )

    reference_x = x_data.clone().requires_grad_()
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    reference_output = F.silu(reference_gate) * reference_up
    (reference_output * upstream).sum().backward()

    x = x_data.clone().requires_grad_()
    output = training_activation.silu_and_mul(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert x.grad is not None and reference_x.grad is not None
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)


def test_training_silu_accepts_noncontiguous_input() -> None:
    torch.manual_seed(20260729)
    source_data = torch.randn((2, 64, 5), device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn((2, 5, 32), device="cuda", dtype=torch.bfloat16)

    reference_source = source_data.clone().requires_grad_()
    reference_x = reference_source.transpose(1, 2)
    reference_x.retain_grad()
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    reference_output = F.silu(reference_gate) * reference_up
    (reference_output * upstream).sum().backward()

    source = source_data.clone().requires_grad_()
    x = source.transpose(1, 2)
    x.retain_grad()
    output = training_activation.silu_and_mul(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert not x.is_contiguous()
    assert x.grad is not None and reference_x.grad is not None
    assert source.grad is not None and reference_source.grad is not None
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)


def test_training_silu_falls_back_for_noncontiguous_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260805)
    x_data = torch.randn((7, 64), device="cuda", dtype=torch.bfloat16)
    upstream_storage = torch.randn((7, 64), device="cuda", dtype=torch.bfloat16)
    upstream = upstream_storage[:, ::2]

    reference_x = x_data.clone().requires_grad_()
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    reference_output = F.silu(reference_gate) * reference_up
    (reference_grad,) = torch.autograd.grad(reference_output, reference_x, upstream)

    def reject_fused_backward(*_args: object) -> torch.Tensor:
        raise AssertionError("non-contiguous upstream must use the PyTorch fallback")

    monkeypatch.setattr(
        training_activation,
        "_fused_silu_and_mul_backward",
        reject_fused_backward,
    )
    x = x_data.clone().requires_grad_()
    output = training_activation.silu_and_mul(x)
    (grad_x,) = torch.autograd.grad(output, x, upstream)

    assert not upstream.is_contiguous()
    torch.testing.assert_close(grad_x, reference_grad, rtol=0, atol=0)


def test_training_silu_empty_input_uses_fallback() -> None:
    x = torch.empty((0, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = training_activation.silu_and_mul(x)
    output.sum().backward()

    assert output.shape == (0, 128)
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_training_silu_saves_only_merged_input() -> None:
    torch.manual_seed(20260730)
    storage = torch.randn((23 * 6144 + 1,), device="cuda", dtype=torch.bfloat16)
    original_storage = storage.clone()
    x = storage[1:].view(23, 6144).detach().requires_grad_()
    saved_tensors: list[torch.Tensor] = []

    assert x.is_contiguous()
    assert x.storage_offset() == 1

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(
        pack,
        lambda tensor: tensor,
    ):
        output = training_activation.silu_and_mul(x)

    assert len(saved_tensors) == 1
    (saved_input,) = saved_tensors
    assert saved_input.shape == x.shape
    assert saved_input.stride() == x.stride()
    assert saved_input.storage_offset() == x.storage_offset()
    assert saved_input.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    output.sum().backward()
    assert x.grad is not None
    assert x.grad.is_contiguous()
    assert x.grad.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)


def test_training_silu_rejects_higher_order_gradients() -> None:
    torch.manual_seed(20260731)
    x = torch.randn((7, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = training_activation.silu_and_mul(x)
    (grad_x,) = torch.autograd.grad(output.sum(), x, create_graph=True)

    assert not grad_x.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(grad_x.sum(), x)


def test_training_silu_passes_first_order_gradcheck() -> None:
    torch.manual_seed(20260801)
    x = torch.randn((2, 6), dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(training_activation.silu_and_mul, (x,))


def test_training_silu_fp16_matches_expanded_reference() -> None:
    torch.manual_seed(20260804)
    x_data = torch.randn((17, 256), device="cuda", dtype=torch.float16)
    upstream = torch.randn((17, 128), device="cuda", dtype=torch.float16)

    reference_x = x_data.clone().requires_grad_()
    reference_output = _reference_activation("silu", reference_x)
    (reference_output * upstream).sum().backward()

    x = x_data.clone().requires_grad_()
    output = training_activation.silu_and_mul(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    _assert_numerically_close(output, reference_output)
    assert x.grad is not None and reference_x.grad is not None
    _assert_numerically_close(x.grad, reference_x.grad)


def test_training_silu_cuda_graph_replays_forward_and_backward() -> None:
    torch.manual_seed(20260802)
    x = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn((17, 3072), device="cuda", dtype=torch.bfloat16)

    (training_activation.silu_and_mul(x) * upstream).sum().backward()
    x.grad = None
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = training_activation.silu_and_mul(x)
        (output * upstream).sum().backward()

    replay_x = torch.randn_like(x)
    replay_upstream = torch.randn_like(upstream)
    reference_x = replay_x.detach().clone().requires_grad_()
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    reference_output = F.silu(reference_gate) * reference_up
    (reference_output * replay_upstream).sum().backward()

    with torch.no_grad():
        x.copy_(replay_x)
        upstream.copy_(replay_upstream)
        assert x.grad is not None
        x.grad.zero_()
    graph.replay()
    torch.cuda.synchronize()

    _assert_numerically_close(output, reference_output)
    assert x.grad is not None and reference_x.grad is not None
    _assert_numerically_close(x.grad, reference_x.grad)


def test_training_silu_supports_fullgraph_compile() -> None:
    torch.manual_seed(20260803)
    compiled = torch.compile(training_activation.silu_and_mul, fullgraph=True)
    x = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn((17, 3072), device="cuda", dtype=torch.bfloat16)

    output = compiled(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    reference_x = x.detach().clone().requires_grad_()
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    reference_output = F.silu(reference_gate) * reference_up
    (reference_output * upstream).sum().backward()

    _assert_numerically_close(output, reference_output)
    assert x.grad is not None and reference_x.grad is not None
    _assert_numerically_close(x.grad, reference_x.grad)


@pytest.mark.parametrize("name", ("silu", "gelu"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shape", ((17, 256), (2, 5, 6144), (3, 19_456)))
def test_training_activation_matches_expanded_reference(
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(20260721)
    operation: Callable[[torch.Tensor], torch.Tensor] = getattr(
        training_activation, name + "_and_mul"
    )
    x_data = torch.randn(shape, device="cuda", dtype=dtype)
    upstream = torch.randn(shape[:-1] + (shape[-1] // 2,), device="cuda", dtype=dtype)

    reference_x = x_data.clone().requires_grad_()
    reference_output = _reference_activation(name, reference_x)
    (reference_output * upstream).sum().backward()

    x = x_data.clone().requires_grad_()
    output = operation(x)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    _assert_numerically_close(output, reference_output)
    assert x.grad is not None and reference_x.grad is not None
    _assert_numerically_close(x.grad, reference_x.grad)
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("silu", "gelu"))
def test_training_activation_preserves_branched_input(name: str) -> None:
    torch.manual_seed(20260722)
    operation: Callable[[torch.Tensor], torch.Tensor] = getattr(
        training_activation, name + "_and_mul"
    )
    x_data = torch.randn((23, 6144), device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn((23, 3072), device="cuda", dtype=torch.bfloat16)

    reference_x = x_data.clone().requires_grad_()
    reference_output = _reference_activation(name, reference_x)
    ((reference_output * upstream).sum() + 0.125 * reference_x.float().square().sum()).backward()

    x = x_data.clone().requires_grad_()
    output = operation(x)
    ((output * upstream).sum() + 0.125 * x.float().square().sum()).backward()
    torch.cuda.synchronize()

    assert x.grad is not None and reference_x.grad is not None
    _assert_numerically_close(x.grad, reference_x.grad)
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("silu", "gelu"))
def test_training_activation_accepts_frozen_input(name: str) -> None:
    torch.manual_seed(20260724)
    operation: Callable[[torch.Tensor], torch.Tensor] = getattr(
        training_activation, name + "_and_mul"
    )
    x = torch.randn((29, 6144), device="cuda", dtype=torch.bfloat16)
    original_x = x.clone()
    expected = _reference_activation(name, x)

    output = operation(x)
    torch.cuda.synchronize()

    assert not output.requires_grad
    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    _assert_numerically_close(output, expected)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("silu", "gelu"))
@pytest.mark.parametrize("use_out", (False, True))
def test_inference_activation_output_contract(name: str, use_out: bool) -> None:
    torch.manual_seed(20260725)
    operation: Callable[..., torch.Tensor] = getattr(inference_activation, name + "_and_mul")
    x = torch.randn((19, 256), device="cuda", dtype=torch.bfloat16)
    original_x = x.clone()
    expected = _reference_activation(name, x)
    out = torch.empty_like(expected) if use_out else None

    with torch.inference_mode():
        result = operation(x, out=out)
    torch.cuda.synchronize()

    assert result.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    if out is not None:
        assert result.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    _assert_numerically_close(result, expected)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("silu", "gelu"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_inference_activation_cuda_graph(name: str, dtype: torch.dtype) -> None:
    torch.manual_seed(20260728)
    operation: Callable[..., torch.Tensor] = getattr(inference_activation, name + "_and_mul")
    x = torch.randn((17, 256), device="cuda", dtype=dtype)
    out = torch.empty((17, 128), device="cuda", dtype=dtype)

    with torch.inference_mode():
        operation(x, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        result = operation(x, out=out)

    replay_x = torch.randn_like(x)
    expected = _reference_activation(name, replay_x)
    x.copy_(replay_x)
    graph.replay()
    torch.cuda.synchronize()

    assert result.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    _assert_numerically_close(out, expected)


@pytest.mark.parametrize("name", ("silu", "gelu"))
def test_inference_activation_rejects_active_autograd(name: str) -> None:
    torch.manual_seed(20260726)
    operation: Callable[..., torch.Tensor] = getattr(inference_activation, name + "_and_mul")
    x = torch.randn((13, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    original_x = x.detach().clone()

    with pytest.raises(RuntimeError, match="does not support autograd"):
        operation(x)

    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("silu", "gelu"))
def test_inference_activation_rejects_gradient_output_storage(name: str) -> None:
    torch.manual_seed(20260727)
    operation: Callable[..., torch.Tensor] = getattr(inference_activation, name + "_and_mul")
    x = torch.randn((13, 256), device="cuda", dtype=torch.bfloat16)
    out = torch.randn((13, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    original_out = out.detach().clone()

    with torch.inference_mode(), pytest.raises(RuntimeError, match="cannot write"):
        operation(x, out=out)

    torch.testing.assert_close(out, original_out, rtol=0, atol=0)


# C3 qualification uses an unrounded FP64 oracle, not provider-to-provider tolerance.
from benchmark.kernel_tuning.norm_act_reference import (  # noqa: E402
    FP32_ACTIVATION_KINDS,
    FP32_ACTIVATION_ROWS,
    FP32_ACTIVATION_WIDTHS,
    RELU2_WIDTHS,
    error_metrics,
    fp32_activation_formulas,
    market_formulas,
    no_worse,
    reference,
)
from benchmark.kernel_tuning.norm_act_reference import (  # noqa: E402
    support_range_formula as _support_range_formula,
)
from ctikernels.activation._dispatch import TABLE, Config, config_for  # noqa: E402

from ._exhaustive import ci_cases  # noqa: E402

_INDUCTOR = []


def _markets(x):
    """The market formulas plus vLLM's real kernel: Inductor-compiled forward_native."""
    if not _INDUCTOR:
        _INDUCTOR.append(
            torch.compile(lambda t: F.silu(t[..., : t.shape[-1] // 2]) * t[..., t.shape[-1] // 2 :])
        )
    outputs = dict(market_formulas(x))
    if x.numel():
        outputs["vllm_inductor"] = _INDUCTOR[0](x)
    return outputs


@pytest.mark.parametrize(
    "m,width,dtype",
    ci_cases(
        (0, 1, 3, 4, 16, 17, 64, 256, 1024),
        (8, 128, 12288),
        (torch.bfloat16, torch.float16),
        core={(m, 12288, torch.bfloat16) for m in (0, 1, 4, 16, 64, 256, 1024)}
        | {(3, 8, torch.float16), (17, 128, torch.float16)},
    ),
)
def test_cute_silu_fp64_and_market_formulas(m, width, dtype):
    torch.manual_seed(20261015 + m)
    x = torch.randn((m, 2 * width), device="cuda", dtype=dtype)
    before = x.clone()
    out = torch.empty((m, width), device=x.device, dtype=x.dtype)
    op = inference_activation.SiluAndMul(provider="cute")
    result = op(x, out)
    assert result is out
    assert op.provider == ("empty" if m == 0 else "cute")
    exact = reference(x)
    ours = error_metrics(result, exact)
    market = [error_metrics(y, exact) for y in _markets(x).values()]
    assert no_worse(ours, market), (ours, market)
    if m and _flashinfer_version.allowed():
        flash = inference_activation.SiluAndMul(provider="flashinfer")(x)
        # FI's approximate exp/reciprocal may cross a rounding boundary. The
        # ordinary-input bound is one output ULP, not unconditional bitwise equality.
        spacing = (
            torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(torch.finfo(dtype).tiny))))
            * torch.finfo(dtype).eps
        )
        assert ((result.double() - flash.double()).abs() / spacing).max().item() <= 1
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_graph_and_dynamic_m(pdl, dtype):
    from ctikernels.activation.cute import _provider

    op = inference_activation.SiluAndMul(provider="cute", config=Config(64, 2, pdl))
    x = torch.randn((3, 256), device="cuda", dtype=dtype)
    op(x)
    keys = set(_provider._COMPILED)
    x = torch.randn((2, 17, 256), device="cuda", dtype=dtype)
    out = torch.empty((2, 17, 128), device=x.device, dtype=dtype)
    op(x, out)
    assert set(_provider._COMPILED) == keys  # M/rank do not specialize the compiler.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(x, out)
    x.normal_()
    graph.replay()
    assert result is out
    exact = reference(x)
    metrics = error_metrics(out, exact)
    assert no_worse(metrics, [error_metrics(y, exact) for y in _markets(x).values()])


def test_silu_dispatch_selects_table_and_general(monkeypatch):
    monkeypatch.setenv(inference_activation.VARIABLE, "auto")
    monkeypatch.delenv("CTI_ACT_PROVIDER", raising=False)
    x = torch.randn((4, 24576), device="cuda", dtype=torch.bfloat16)
    kind, capability = inference_activation._hardware(x.device)
    operation = inference_activation.SiluAndMul()
    operation(x)
    assert operation.provider == "cute"
    table = config_for(kind, 12288, "bfloat16", 4) if capability[0] == 12 else None
    assert operation.dispatched_config == (table or inference_activation.GeneralConfig(8))
    operation(torch.randn((3, 256), device=x.device, dtype=x.dtype))
    assert operation.provider == "cute"
    assert operation.dispatched_config.name == "cute_general_n8"
    if _flashinfer_version.allowed():
        monkeypatch.setenv(inference_activation.VARIABLE, "flashinfer")
        operation(x)
        assert operation.provider == "flashinfer"
    monkeypatch.setenv(inference_activation.VARIABLE, "auto")
    for value in (
        x.float(),
        x[:, ::2],
        torch.randn(24577, device="cuda", dtype=x.dtype)[1:].view(1, -1),
        torch.randn(3, 18, device="cuda", dtype=x.dtype),
    ):
        actual = operation(value)
        assert operation.provider == "cute"
        _assert_numerically_close(actual, _reference_activation("silu", value))


def test_cute_silu_out_rejects_input_alias_and_bad_metadata():
    x = torch.randn((1, 256), device="cuda", dtype=torch.float16)
    operation = inference_activation.SiluAndMul(provider="cute")
    with pytest.raises(ValueError, match="overlap input"):
        operation(x, x[:, :128])
    with pytest.raises(ValueError, match="shape, dtype"):
        operation(x, torch.empty(1, 128, device=x.device, dtype=torch.bfloat16))
    out = torch.empty(129, device=x.device, dtype=x.dtype)[1:].view(1, 128)
    assert operation(x, out) is out
    assert operation.dispatched_config.name == "cute_general_n1"
    with pytest.raises(RuntimeError, match="aligned contiguous"):
        inference_activation.SiluAndMul(provider="cute", config=Config())(x, out)


@pytest.mark.exhaustive
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_all_finite_gate_encodings(dtype):
    # Every finite gate encoding, including BF16 negative tails and subnormals.
    gates = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    gates = gates[torch.isfinite(gates)]
    gates = F.pad(gates, (0, (-gates.numel()) % 8)).cuda()
    op = inference_activation.SiluAndMul(provider="cute")
    for multiplier in (0.0, 0.25, 1.0, -3.0, 128.0):
        x = torch.cat((gates, torch.full_like(gates, multiplier))).view(1, -1)
        actual = op(x)
        exact = reference(x)
        ours = error_metrics(actual, exact)
        market = [error_metrics(y, exact) for y in _markets(x).values()]
        assert no_worse(ours, market), (dtype, multiplier, ours, market)


def _distinct_shipped_activation_configs():
    configs = {
        (width, config)
        for table in TABLE.values()
        for (width, _dtype), bands in table.items()
        for config in bands.values()
    }
    return [
        pytest.param(width, config, id=f"w{width}-{config.name}")
        for width, config in sorted(configs, key=lambda item: (item[0], item[1].name))
    ]


@pytest.mark.exhaustive
@pytest.mark.parametrize("width,config", _distinct_shipped_activation_configs())
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_shipped_activation_all_finite_encodings(width, config, dtype):
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    encodings = encodings[torch.isfinite(encodings)]
    padding = (-encodings.numel()) % width
    gates = F.pad(encodings, (0, padding)).view(-1, width).cuda()
    generator = torch.Generator().manual_seed(20261016)
    shuffled = encodings[torch.randperm(encodings.numel(), generator=generator)]
    shuffled = F.pad(shuffled, (0, padding)).view_as(gates).cuda()
    op = inference_activation.SiluAndMul(provider="cute", config=config)
    wide_config = replace(config, elems=8)
    wide = inference_activation.SiluAndMul(provider="cute", config=wide_config)
    for multiplier in (0.0, 0.25, 1.0, -3.0, 128.0, None):
        # The paired pass covers every finite up encoding as well as every gate.
        gate = gates if multiplier is not None else shuffled
        up = torch.full_like(gates, multiplier) if multiplier is not None else gates
        x = torch.cat((gate, up), dim=-1)
        actual = op(x)
        assert op.provider == "cute"
        exact = reference(x)
        ours = error_metrics(actual, exact)
        markets = [error_metrics(y, exact) for y in _markets(x).values()]
        assert no_worse(ours, markets), (width, config, dtype, multiplier, ours, markets)
        # Every access width computes each element alike.
        if config.elems != 8:
            expected = wide(x)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
                width,
                config,
                dtype,
                multiplier,
            )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_zero_is_bitwise_flashinfer(dtype):
    if not _flashinfer_version.allowed():
        pytest.skip("FlashInfer comparison requires the qualified pin")
    x = torch.zeros((4, 24576), device="cuda", dtype=dtype)
    ours = inference_activation.SiluAndMul(provider="cute")(x)
    flash = inference_activation.SiluAndMul(provider="flashinfer")(x)
    assert torch.equal(ours.view(torch.int16), flash.view(torch.int16))


def test_cute_silu_disjoint_views_share_allocation():
    storage = torch.randn(384, device="cuda", dtype=torch.bfloat16)
    x, out = storage[:256].view(1, 256), storage[256:].view(1, 128)
    before = x.clone()
    result = inference_activation.SiluAndMul(provider="cute")(x, out)
    assert result is out
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    exact = reference(x)
    assert no_worse(
        error_metrics(out, exact), [error_metrics(y, exact) for y in _markets(x).values()]
    )


def _shipped_activation_cases():
    from ctikernels.activation._dispatch import TABLE

    seen = set()
    cases = []
    for kind, table in TABLE.items():
        for (width, dtype), bands in table.items():
            for (lo, _hi), config in bands.items():
                key = (width, config)
                core = dtype == "bfloat16" and key not in seen
                if core:
                    seen.add(key)
                cases.append(
                    pytest.param(
                        kind,
                        width,
                        dtype,
                        max(1, lo),
                        config,
                        marks=() if core else pytest.mark.exhaustive,
                    )
                )
    return cases


@pytest.mark.parametrize("kind,width,dtype,m,config", _shipped_activation_cases())
def test_shipped_activation_configurations(kind, width, dtype, m, config):
    torch.manual_seed(20261015 + m)
    x = torch.randn((m, 2 * width), device="cuda", dtype=getattr(torch, dtype))
    forced = inference_activation.SiluAndMul(provider="cute", config=config)
    result = forced(x)
    exact = reference(x)
    assert no_worse(
        error_metrics(result, exact), [error_metrics(y, exact) for y in _markets(x).values()]
    )
    if inference_activation._hardware(x.device)[0] == kind:
        auto = inference_activation.SiluAndMul(provider="auto")
        actual = auto(x)
        assert auto.provider == "cute"
        assert torch.equal(actual.view(torch.int16), result.view(torch.int16))


@pytest.mark.parametrize("elems", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_silu_negative_tail_is_bitwise_sglang_core(elems, dtype):
    # Mix negative-tail and ordinary lanes within vectors: SGLang's g / (1 + expf(-g)) * up,
    # whose expf overflows to a signed zero below about -88.
    gates = torch.tensor(
        [-81, 1, -90, -1, -100, 0, -120, 3, -80, -79, -82, -88, -92, -110, -140, -200],
        device="cuda",
        dtype=dtype,
    )
    up = torch.full_like(gates, 128)
    x = torch.cat((gates, up)).view(1, -1)
    op = inference_activation.SiluAndMul(provider="cute", config=Config(64, 1, True, elems))
    result = op(x)
    expected = market_formulas(x)["sglang_formula"]
    assert torch.equal(result.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", (1, 3, 256))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_e16_is_bitwise_e8_core(m, dtype):
    torch.manual_seed(20261016 + m)
    width = 12288
    # Exercise 16-byte aligned buffers that are deliberately not 32-byte aligned.
    storage = torch.randn(m * 2 * width + 8, device="cuda", dtype=dtype)
    x = storage[8:].view(m, 2 * width)
    out_storage = torch.empty(m * width + 8, device=x.device, dtype=dtype)
    out = out_storage[8:].view(m, width)
    assert x.data_ptr() % 32 == out.data_ptr() % 32 == 16
    config = Config(128, 1, False, 16)
    wide = inference_activation.SiluAndMul(provider="cute", config=config)
    narrow = inference_activation.SiluAndMul(provider="cute", config=replace(config, elems=8))
    for mixed_tail in (False, True):
        if mixed_tail:
            # Negative tails in only the second half of each e16 vector.
            x[:, 8:width:32] = -90
        expected = narrow(x)
        actual = wide(x, out)
        assert actual is out and wide.provider == "cute"
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("width", (8, 10, 24, 12296))
def test_cute_silu_e16_rejects_partial_vector_core(width):
    x = torch.randn(3, 2 * width, device="cuda", dtype=torch.bfloat16)
    op = inference_activation.SiluAndMul(provider="cute", config=Config(elems=16))
    with pytest.raises(ValueError, match="elems=16 requires width divisible by 16"):
        op(x)


@pytest.mark.parametrize("shape", [(2, 1, 24576), (2, 256, 24576), (2, 600, 24576)])
@pytest.mark.parametrize("provider", ["cute", "flashinfer", "auto"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, pytest.param(torch.float16, marks=pytest.mark.exhaustive)]
)
def test_silu_higher_rank_is_bitwise_flattened(shape, provider, dtype):
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("forced FlashInfer requires the qualified pin")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    width = shape[-1] // 2
    op = inference_activation.SiluAndMul(provider=provider)
    expected = op(x.view(-1, 2 * width))
    flat_provider = op.provider
    out = torch.empty((*shape[:-1], width), device=x.device, dtype=dtype)
    result = op(x, out)
    assert result is out and op.provider == flat_provider
    assert torch.equal(result.view(torch.int16), expected.view_as(result).view(torch.int16))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_early_trigger_chain_core(dtype):
    """The second PDL call consumes the first output, including in graph replay."""
    from dataclasses import replace

    config = Config(128, 2, True, 4)
    end = inference_activation.SiluAndMul(provider="cute", config=config)
    early = inference_activation.SiluAndMul(
        provider="cute", config=replace(config, trigger="early")
    )
    x = torch.randn(3, 1024, device="cuda", dtype=dtype)
    middle = torch.empty(3, 512, device="cuda", dtype=dtype)
    out = torch.empty(3, 256, device="cuda", dtype=dtype)
    end_middle = end(x)
    expected = end(end_middle)
    early(x, middle)
    _assert_numerically_close(middle, _reference_activation("silu", x))
    assert torch.equal(middle.view(torch.uint8), end_middle.view(torch.uint8))
    early(middle, out)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        early(x, middle)
        early(middle, out)
    for _ in range(2):
        middle.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "cache,elems,m,dtype",
    ci_cases(
        ("none", "st_ef", "ld_el", "both"),
        (2, 4, 8, 16),
        (1, 3, 256),
        (torch.bfloat16, torch.float16),
        # Every cache policy and access width once in core, mixing M and dtype;
        # the full 96-case product is exhaustive.
        core={
            ("st_ef", 2, 1, torch.bfloat16),
            ("ld_el", 4, 3, torch.float16),
            ("both", 8, 256, torch.bfloat16),
            ("both", 16, 3, torch.float16),
            ("st_ef", 16, 256, torch.float16),
            ("ld_el", 8, 1, torch.bfloat16),
            ("none", 8, 256, torch.bfloat16),
        },
    ),
)
def test_cute_silu_cache_is_bitwise_none_core(cache, elems, m, dtype):
    torch.manual_seed(20261016 + m)
    width = 12288
    # Include e16's two v4 accesses at the provider's minimum 16-byte alignment.
    storage = torch.randn(m * 2 * width + 8, device="cuda", dtype=dtype)
    x = storage[8:].view(m, 2 * width)
    out_storage = torch.empty(m * width + 8, device="cuda", dtype=dtype)
    out = out_storage[8:].view(m, width)
    assert x.data_ptr() % 32 == out.data_ptr() % 32 == 16
    config = Config(128, 1, True, elems, "early", cache)
    hinted = inference_activation.SiluAndMul(provider="cute", config=config)
    baseline = inference_activation.SiluAndMul(
        provider="cute", config=replace(config, cache="none")
    )
    for negative_tail in (False, True):
        if negative_tail:
            # Mixed ordinary and negative-tail lanes, some only in an e16's second half.
            for lane, gate in enumerate((-81, -90, -100, -120, -140, -200)):
                x[:, 8 + lane : width : 32] = gate
                x[:, width + 8 + lane :: 32] = 128
        expected = baseline(x)
        out.fill_(float("nan"))
        actual = hinted(x, out)
        assert actual is out and hinted.provider == baseline.provider == "cute"
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("elems", (2, 4, 8, 16))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_cute_silu_cache_both_graph_chain_core(elems, dtype):
    width = 12288
    config = Config(128, 2, True, elems, "early", "both")
    op = inference_activation.SiluAndMul(provider="cute", config=config)
    baseline = inference_activation.SiluAndMul(
        provider="cute", config=replace(config, cache="none")
    )
    x = torch.randn(3, 4 * width, device="cuda", dtype=dtype)
    x[:, 8 : 2 * width : 32] = -90
    middle = torch.empty(3, 2 * width, device="cuda", dtype=dtype)
    out = torch.empty(3, width, device="cuda", dtype=dtype)
    op(x, middle)
    op(middle, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, middle)
        op(middle, out)
    for _ in range(2):
        # Change input and poison intermediates to expose stale/early PDL reads.
        x[:, 2 * width :].mul_(0.5)
        expected = op(op(x))
        assert torch.equal(expected.view(torch.int16), baseline(baseline(x)).view(torch.int16))
        middle.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


# Measured with python -m benchmark.kernel_tuning.norm_act_tau --fp32-activation on the
# RTX 5090, RTX PRO 6000 and GB10 (torch 2.13.0+cu130; identical maxima on all three).
# NA1 method: seeds 0..31 (0..7 above 262144 outputs), plus 20261016+M+width; ALL
# support-range shapes; same-input symmetric pair excess per metric across three
# faithful FP32 comparators, maximized per kind and per size band, rounded UP to a
# power of two. Rows of fewer than 64 outputs set a far larger spread (a 4-element RMS
# is dominated by one value), so they get their own band rather than loosening the
# large one. Measured maxima (max_ulp/rms/rms_ulp), >= 64 outputs: silu 0.588/0.880/
# 0.196, gelu 0.044/0.882/0.042, gelu_tanh 0.410/0.971/0.099; < 64 outputs: silu
# 1.518/11.58/2.292, gelu 3.007/37.50/2.832, gelu_tanh 3.569/36.70/2.176; relu2 0.
# See benchmark/kernel_tuning/results/norm_act.md. Never reuse the 16-bit TAU here.
TAU_FP32 = {
    "silu": {"max_ulp": 1.0, "rms": 1.0, "rms_ulp": 0.25},
    "gelu": {"max_ulp": 0.0625, "rms": 1.0, "rms_ulp": 0.0625},
    "gelu_tanh": {"max_ulp": 0.5, "rms": 1.0, "rms_ulp": 0.125},
    "relu2": {"max_ulp": 0.0, "rms": 0.0, "rms_ulp": 0.0},
}
TAU_FP32_SMALL = {
    "silu": {"max_ulp": 2.0, "rms": 16.0, "rms_ulp": 4.0},
    "gelu": {"max_ulp": 4.0, "rms": 64.0, "rms_ulp": 4.0},
    "gelu_tanh": {"max_ulp": 4.0, "rms": 64.0, "rms_ulp": 4.0},
    "relu2": {"max_ulp": 0.0, "rms": 0.0, "rms_ulp": 0.0},
}
SMALL_OUTPUTS = 64


def tau_fp32(kind, outputs):
    return (TAU_FP32_SMALL if outputs < SMALL_OUTPUTS else TAU_FP32)[kind]


@pytest.mark.parametrize("kind", FP32_ACTIVATION_KINDS)
@pytest.mark.parametrize("metric", ("max_ulp", "rms", "rms_ulp"))
def test_fp32_activation_gate_uses_best_per_metric(kind, metric):
    tau = TAU_FP32[kind]
    # Different comparators win different metrics. A worse comparator must not
    # hide even a one-step violation of the best comparator's noise allowance.
    markets = [
        dict.fromkeys(tau, 0.5) | {metric: 1000.0},
        dict.fromkeys(tau, 1000.0) | {metric: 0.5},
    ]
    ours = {key: 0.5 * (1 + value) for key, value in tau.items()}
    assert no_worse(ours, markets, tau=tau)
    ours[metric] = math.nextafter(ours[metric], math.inf)
    assert not no_worse(ours, markets, tau=tau)
    assert not no_worse(ours, list(reversed(markets)), tau=tau)


def _check_support_range_accuracy(kind, dtype, m, width):
    torch.manual_seed(20261016 + m + width)
    factor = 1 if kind == "relu2" else 2
    x = torch.randn(m, factor * width, device="cuda", dtype=dtype)
    original = x.clone()
    operation = getattr(inference_activation, kind if kind == "relu2" else kind + "_and_mul")
    result = operation(x)
    exact = _support_range_formula(kind, x, exact=True)
    ours = error_metrics(result, exact)
    if dtype == torch.float32:
        markets = {
            name: error_metrics(value, exact)
            for name, value in fp32_activation_formulas(kind, x).items()
        }
        tau = tau_fp32(kind, m * width)
        assert no_worse(ours, list(markets.values()), tau=tau), (
            kind,
            dtype,
            m,
            width,
            ours,
            markets,
            tau,
        )
    else:
        comparator = _support_range_formula(kind, x, exact=False)
        market = error_metrics(comparator, exact)
        assert no_worse(ours, [market]), (kind, dtype, m, width, ours, market)
    assert result.shape == (m, width) and result.dtype == dtype
    assert torch.equal(x, original)
    out = torch.empty_like(result)
    assert operation(x, out) is out
    assert torch.equal(out.view(torch.uint8), result.view(torch.uint8))


@pytest.mark.parametrize(
    "kind,m,width",
    ci_cases(
        FP32_ACTIVATION_KINDS[:-1],
        FP32_ACTIVATION_ROWS,
        FP32_ACTIVATION_WIDTHS,
        core={
            (kind, m, width)
            for kind in ("silu", "gelu", "gelu_tanh")
            for m in (0, 1, 3, 64, 257)
            for width in (4, 8, 12, 13, 6144, 12288)
            if m in (0, 3) or width == 6144
        },
    ),
)
def test_fp32_activation_support_range(kind, m, width):
    # SGLang JIT uses D % 4 (pre-Blackwell) / D % 8 (Blackwell) for FP32.
    # D=4/12 are tails relative to our old D%8 admission; D=13 is an extension.
    _check_support_range_accuracy(kind, torch.float32, m, width)


@pytest.mark.parametrize(
    "dtype,m,width",
    ci_cases(
        [torch.float16, torch.bfloat16, torch.float32],
        FP32_ACTIVATION_ROWS,
        RELU2_WIDTHS,
        core={
            (dtype, m, width)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
            for m in (0, 1, 3, 64, 257)
            for width in (1, 13, 6144, 12288)
            if m in (0, 64) or width == 6144
        },
    ),
)
def test_relu2_support_range(dtype, m, width):
    # SGLang checks total elements, not row width: M=64,D=13 is valid on both
    # 16-byte and 32-byte vector architectures, in every supported dtype.
    _check_support_range_accuracy("relu2", dtype, m, width)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [0, 3])
def test_tanh_gelu_forced_flashinfer_bitwise(monkeypatch, dtype, m):
    if not _flashinfer_version.allowed():
        pytest.skip("FlashInfer comparison requires the qualified pin")
    import flashinfer

    x = torch.randn(m, 2 * 6144, device="cuda", dtype=dtype)
    calls = []
    real = inference_activation.flashinfer.gelu_tanh_and_mul

    def recorded(x, out=None):
        calls.append(1)
        return real(x, out)

    monkeypatch.setattr(inference_activation.flashinfer, "gelu_tanh_and_mul", recorded)
    op = inference_activation.SiluAndMul(activation="gelu_tanh", provider="flashinfer")
    out = op(x)
    assert op.provider == ("flashinfer" if m else "empty")
    assert calls == ([1] if m else [])
    if m:
        assert torch.equal(out.view(torch.int16), flashinfer.gelu_tanh_and_mul(x).view(torch.int16))
    else:
        assert out.shape == (0, 6144)


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "relu2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_activation_empty_gpu_never_launches(monkeypatch, name, dtype):
    def unexpected(*args, **kwargs):
        pytest.fail("empty activation launched or inspected provider")

    monkeypatch.setattr(inference_activation, "_cute_activation", unexpected)
    monkeypatch.setattr(inference_activation, "_hardware", unexpected)
    monkeypatch.setattr(inference_activation._flashinfer_version, "allowed", unexpected)
    if name != "relu2":
        monkeypatch.setattr(inference_activation.flashinfer, name, unexpected)
    x = torch.empty(2, 0, 26, device="cuda", dtype=dtype)
    out = torch.empty(2, 0, 26 if name == "relu2" else 13, device="cuda", dtype=dtype)
    operation = getattr(inference_activation, name)
    assert operation(x).shape == out.shape
    assert operation(x, out) is out


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh", "relu2"])
@pytest.mark.parametrize("width", [8, 13])
@pytest.mark.parametrize("layout", ["offset", "strided"])
def test_new_cute_activation_offset_and_strided_buffers(kind, width, layout):
    factor = 1 if kind == "relu2" else 2
    cols = factor * width
    x = (
        torch.randn(3 * cols + 1, device="cuda")[1:].view(3, cols)
        if layout == "offset"
        else torch.randn(3, cols * 2, device="cuda")[:, ::2]
    )
    storage = torch.full((3 * width * 2 + 1,), float("nan"), device="cuda")
    out = (
        storage[1 : 3 * width + 1].view(3, width)
        if layout == "offset"
        else storage[1:].view(3, width * 2)[:, ::2]
    )
    operation = getattr(inference_activation, kind if kind == "relu2" else kind + "_and_mul")
    expected = operation(x.contiguous())
    assert operation(x, out) is out
    assert torch.equal(out.contiguous().view(torch.int32), expected.view(torch.int32))
    assert torch.isnan(storage[0])
    if layout == "strided":
        assert torch.isnan(storage[2::2]).all()


def test_fp32_gelu_graph_replay_bitwise():
    x = torch.randn(3, 26, device="cuda")
    out = torch.empty(3, 13, device="cuda")
    inference_activation.gelu_and_mul(x, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        inference_activation.gelu_and_mul(x, out)
    for _ in range(2):
        x.uniform_(-4, 4)
        eager = inference_activation.gelu_and_mul(x)
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out.view(torch.int32), eager.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_relu2_comparison_and_rounding_semantics(dtype):
    x = torch.tensor(
        [float("nan"), -float("inf"), -0.0, 0.0, -1.0, 1.0078125, 2.0, float("inf")],
        device="cuda",
        dtype=dtype,
    )
    expected = _support_range_formula("relu2", x, exact=False)
    result = inference_activation.relu2(x)
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_relu2_inplace_matches_out_of_place(dtype):
    x = torch.randn(3, 13, device="cuda", dtype=dtype)
    expected = inference_activation.relu2(x)
    assert inference_activation.relu2(x, x) is x
    assert torch.equal(x.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["auto", "flashinfer", "triton"])
@pytest.mark.parametrize("has_out", [False, True])
def test_silu_warm_up_graph_routes_core(provider, has_out):
    if provider == "flashinfer" and not _flashinfer_version.allowed():
        pytest.skip("forced FlashInfer requires the qualified pin")
    op = inference_activation.SiluAndMul(provider=provider)
    device = torch.device("cuda", torch.cuda.current_device())
    routes = op.warm_up(128, width=12288, dtype=torch.bfloat16, device=device)
    op._require_cached_plan = True
    for m in (1, 4, 16, 64, 65, 128):
        route = next(r for r in routes if r["m"][0] <= m <= r["m"][1])
        x = torch.randn((m, 24576), device=device, dtype=torch.bfloat16)
        out = torch.empty((m, 12288), device=device, dtype=x.dtype) if has_out else None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op(x, out)
        assert op.provider == route["provider"]
        assert (None if op.dispatched_config is None else op.dispatched_config.name) == route[
            "config"
        ]
        graph.replay()
        expected = op(x.clone(), torch.empty_like(actual) if has_out else None)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_auto_cute_independent_of_flashinfer(monkeypatch, kind, dtype):
    monkeypatch.setattr(_flashinfer_version, "allowed", lambda: False)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    op = inference_activation.SiluAndMul(activation=kind, provider="auto")
    x = torch.randn(5, 55, device="cuda", dtype=dtype)[:, 1:49]
    out = torch.empty(5, 29, device="cuda", dtype=dtype)[:, 1:25]
    assert op(x, out) is out
    assert op.provider == "cute" and op.provider.reason is None
    _check_na_accuracy(kind, x, out)


# NA-ACT compares per-element rounding against FP64, including the negative tail.
# References: SGLang 72730d6 act_and_mul and vLLM forward_native (Inductor).
def _na_oracle(kind, x):
    g, u = x.double().chunk(2, dim=-1)
    if kind == "silu":
        return (g / (1 + torch.exp(-g))) * u
    # Tail-accurate FP64 forms: Phi = erfc(-g/sqrt(2))/2 (torch's ndtr cancels like
    # 1 + erf on CUDA), (1 + tanh(z))/2 as 1/(1 + exp(-2z)).
    if kind == "gelu":
        return g * (0.5 * torch.special.erfc(-g / math.sqrt(2))) * u
    z = math.sqrt(2 / math.pi) * (g + 0.044715 * g**3)
    return g / (1 + torch.exp(-2 * z)) * u


def _na_composition(kind, x):
    g, u = x.float().chunk(2, dim=-1)
    gate = (
        F.silu(g)
        if kind == "silu"
        else F.gelu(g, approximate="tanh" if kind == "gelu_tanh" else "none")
    )
    return (gate * u).to(x.dtype)


def _binomial_upper(n, c):
    """Exact binomial upper tail, summed via PMF recurrence (no normal approximation)."""
    assert c > n / 2
    term = math.exp(
        math.lgamma(n + 1) - math.lgamma(c + 1) - math.lgamma(n - c + 1) - n * math.log(2)
    )
    total = term
    for k in range(c, n):
        term *= (n - k) / (k + 1)
        total += term
        if term <= total * 1e-15:
            break
    return total


def _check_na_accuracy(kind, x, actual, reference_output=None):
    # Chunk the oracle and masks to bound scratch memory, even on the exhaustive grid.
    width = actual.shape[-1]
    rows = max(1, (1 << 18) // width)
    xrows = x.reshape(-1, 2 * width)
    yrows = actual.reshape(-1, width)
    refrows = None if reference_output is None else reference_output.reshape(-1, width)
    b = c = 0
    for start in range(0, xrows.shape[0], rows):
        xx, ours = xrows[start : start + rows], yrows[start : start + rows]
        exact = _na_oracle(kind, xx)
        rounded = exact.to(actual.dtype)
        ref = _na_composition(kind, xx) if refrows is None else refrows[start : start + rows]
        # Spacing above |rounded| includes subnormals and zero without log2(0).
        magnitude = rounded.abs()
        ulp = (
            torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))).double()
            - magnitude.double()
        )
        ours_error = (ours.double() - exact).abs()
        ref_error = (ref.double() - exact).abs()
        ours_exact, ref_exact = ours == rounded, ref == rounded
        b += int((ours_exact & ~ref_exact).sum())
        c += int((ref_exact & ~ours_exact).sum())
        if actual.dtype == torch.float32:
            # FP32 math cannot round FP32 outputs once: argument rounding is amplified by
            # the activation's condition number (several ulp for erfc/exp near |g| ~ 2.5),
            # so bound the worst element by the composition's worst (or 8 ulp).
            ours_max, ref_max = float((ours_error / ulp).max()), float((ref_error / ulp).max())
            assert ours_max <= max(8.0, ref_max), (kind, actual.dtype, start, ours_max, ref_max)
        else:
            assert torch.all((ours_error <= ulp) | (ref_error > ulp)), (
                kind,
                actual.dtype,
                start,
                float((ours_error / ulp).max()),
            )
    if c > b:
        p = _binomial_upper(b + c, c)
        assert p >= 0.01, (kind, actual.dtype, "McNemar", b, c, p)


def _na_input(m, width, dtype, *, pad=0, offset=0):
    x = torch.randn(m, 2 * width + pad + offset, device="cuda", dtype=dtype)
    x.mul_(3)
    x = x[:, offset : offset + 2 * width]
    for lane, value in enumerate((-100, -80.5, 30, 0, -0.0)):
        if lane < width:
            x[:, lane] = value
    return x


_NA_WIDTHS = (8, 16, 24, 1000, 3072, 4352, 6144, 8704, 12288, 17408, 25600)
_NA_ROWS = (0, 1, 2, 3, 4, 5, 16, 64, 65, 512, 4096, 16384)
_NA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_NA_KINDS = ("silu", "gelu", "gelu_tanh")


@pytest.mark.parametrize(
    "width,m,dtype,kind,strided",
    ci_cases(
        _NA_WIDTHS,
        _NA_ROWS,
        _NA_DTYPES,
        _NA_KINDS,
        (False, True),
        core={
            (w, m, dtype, kind, strided)
            for w in (8, 24, 1000, 12288, 17408)
            for m in (0, 1, 3, 65, 512)
            for dtype in _NA_DTYPES
            for kind in _NA_KINDS
            for strided in (False, True)
        },
    ),
)
def test_na_activation_grid(width, m, dtype, kind, strided):
    torch.manual_seed(20261018 + width + m)
    pad = 8 if strided else 0
    required = m * (3 * width + 2 * pad) * torch.empty((), dtype=dtype).element_size()
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        pytest.skip("insufficient free memory for inputs, output and chunked oracle")
    x = _na_input(m, width, dtype, pad=pad)
    storage = torch.full((m, width + pad), 7, device="cuda", dtype=dtype)
    out = storage[:, :width]
    op = inference_activation.SiluAndMul(activation=kind, provider="auto")
    assert op(x, out) is out
    assert op.provider == ("cute" if m else "empty")
    if m:
        device_kind, capability = inference_activation._hardware(x.device)
        tuned = (
            config_for(device_kind, width, str(dtype).removeprefix("torch."), m)
            if kind == "silu" and x.is_contiguous() and out.is_contiguous() and capability[0] == 12
            else None
        )
        assert op.dispatched_config == (
            tuned
            or inference_activation.GeneralConfig(inference_activation._vector_lanes(x, out, width))
        )
        _check_na_accuracy(kind, x, out)
    if pad:
        assert torch.all(storage[:, width:] == 7)


@pytest.mark.parametrize("width", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("kind", _NA_KINDS)
@pytest.mark.parametrize("dtype", _NA_DTYPES)
def test_na_odd_widths(width, kind, dtype):
    x = _na_input(3, width, dtype, pad=3, offset=1)
    out = torch.empty(3, width + 2, device="cuda", dtype=dtype)[:, 1 : width + 1]
    op = inference_activation.SiluAndMul(activation=kind)
    assert op(x, out) is out
    assert op.provider == "cute" and op.dispatched_config.name == "cute_general_n1"
    _check_na_accuracy(kind, x, out)


@pytest.mark.parametrize("kind", _NA_KINDS)
@pytest.mark.parametrize("dtype", _NA_DTYPES)
@pytest.mark.parametrize("collapsible", [False, True])
def test_na_and_layouts(monkeypatch, kind, dtype, collapsible):
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    width = 24
    x = _na_input(6, width, dtype, pad=8).view(2, 3, 2 * width)
    out = torch.empty(2, 3, width + 8, device="cuda", dtype=dtype)[:, :, :width]
    if not collapsible:
        x, out = x.transpose(0, 1), out.transpose(0, 1)
    op = inference_activation.SiluAndMul(activation=kind)
    assert op(x, out) is out
    assert op.provider == "cute"
    assert bool(op.provider.reason) == (not collapsible)
    _check_na_accuracy(kind, x, out)
    if not collapsible:
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
        with pytest.raises(RerouteError, match="packed"):
            op(x, out)


@pytest.mark.parametrize("kind,width", [("silu", 17408), ("gelu_tanh", 12288)])
@pytest.mark.parametrize("m", [1, 4, 64])
def test_na_real_inductor(kind, width, m):
    def forward_native(x):
        d = x.shape[-1] // 2
        gate = F.silu(x[..., :d]) if kind == "silu" else F.gelu(x[..., :d], approximate="tanh")
        return gate * x[..., d:]

    x = _na_input(m, width, torch.bfloat16)
    actual = inference_activation.SiluAndMul(activation=kind)(x)
    _check_na_accuracy(kind, x, actual)
    compiled = torch.compile(forward_native, fullgraph=True)
    _check_na_accuracy(kind, x, actual, compiled(x))


@pytest.mark.parametrize("kind", _NA_KINDS)
@pytest.mark.parametrize("dtype", _NA_DTYPES)
@pytest.mark.parametrize("strided", [False, True])
def test_na_warm_capture_general(monkeypatch, kind, dtype, strided):
    from ctikernels.activation.cute import _provider

    width, m = 17408, 3
    op = inference_activation.SiluAndMul(activation=kind)
    op.warm_up(m, width=width, dtype=dtype, device="cuda")
    x = _na_input(m, width, dtype, pad=8 if strided else 0)
    out = torch.empty(m, width + (8 if strided else 0), device="cuda", dtype=dtype)[:, :width]

    def unexpected(*a, **kw):
        pytest.fail("warmed capture must not compile")

    monkeypatch.setattr(_provider, "compile_general", unexpected)
    monkeypatch.setattr(_provider, "compile_kernel", unexpected)
    # The first call on these strided views is captured, using the warmed lane ABI.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(x, out)
    for _ in range(2):
        x.normal_()
        expected = op(x)
        out.fill_(float("nan"))
        graph.replay()
        assert actual is out
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("m", [1, 4, 65])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_na_tuned_general_bitwise(m, dtype):
    kind, capability = inference_activation._hardware(torch.device("cuda"))
    if capability[0] != 12 or kind is None:
        pytest.skip("requires a tuned table device")
    x = _na_input(m, 12288, dtype)
    op = inference_activation.SiluAndMul()
    tuned = op(x)
    assert isinstance(op.dispatched_config, Config)
    general = inference_activation._cute_activation(x, None, "silu", tuned.shape)
    assert torch.equal(tuned.view(torch.uint8), general.view(torch.uint8))


@pytest.mark.exhaustive
@pytest.mark.parametrize("kind", _NA_KINDS)
def test_na_int64_offsets(kind):
    width, m = 16384, 65537
    required = m * 3 * width * 2
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        pytest.skip("insufficient free memory for int64 indexing case")
    x = torch.zeros(m, 2 * width, device="cuda", dtype=torch.bfloat16)
    x[0].normal_()
    x[-1].normal_()
    assert x.numel() > 2**31
    out = inference_activation.SiluAndMul(activation=kind)(x)
    for row in (0, m - 1):
        _check_na_accuracy(kind, x[row : row + 1], out[row : row + 1])


@pytest.mark.parametrize("dtype", _NA_DTYPES)
def test_na_relu2_native_strided_inplace(dtype):
    x = torch.randn(2, 3, 29, device="cuda", dtype=dtype)[:, :, :13]
    expected = _support_range_formula("relu2", x, exact=False)
    assert inference_activation.relu2(x, x) is x
    assert torch.equal(x.contiguous().view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("kind", _NA_KINDS)
@pytest.mark.parametrize("dtype", _NA_DTYPES)
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_na_partial_vector_alignment(kind, dtype, lanes):
    width = 24
    x = _na_input(3, width, dtype, pad=lanes, offset=lanes)
    storage = torch.empty(3, width + 2 * lanes, device="cuda", dtype=dtype)
    out = storage[:, lanes : width + lanes]
    op = inference_activation.SiluAndMul(activation=kind)
    assert op(x, out) is out
    assert op.dispatched_config.name == f"cute_general_n{lanes}"
    _check_na_accuracy(kind, x, out)
