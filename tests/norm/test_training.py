# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU norm training tests for autograd, saved tensors, recomputation, and training behavior."""

from __future__ import annotations

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from ctikernels.norm import _zero_centered_rms_norm as zero_centered_kernels
from ctikernels.norm import inference as inference_norm
from ctikernels.norm import training as training_norm

from ..support.accuracy import assert_accurate
from . import oracle

pytestmark = pytest.mark.gpu


def _assert_norm_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shape", ((17, 128), (2, 5, 2560)))
@pytest.mark.parametrize("train_weight", (False, True))
def test_rms_norm_training_matches_expanded_reference(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    train_weight: bool,
) -> None:
    torch.manual_seed(20260721)
    eps = 1e-6
    x_data = torch.randn(shape, device="cuda", dtype=dtype)
    weight_data = torch.randn(shape[-1], device="cuda", dtype=dtype)
    upstream = torch.randn_like(x_data)
    original_upstream = upstream.clone()

    baseline_x = x_data.clone().requires_grad_()
    baseline_weight = weight_data.clone().requires_grad_(train_weight)
    baseline_output = oracle.rms_norm(baseline_x, baseline_weight, eps, upcast=False)
    (baseline_output * upstream).sum().backward()

    reference_x = x_data.double().requires_grad_()
    reference_weight = weight_data.double().requires_grad_(train_weight)
    reference_output = oracle.rms_norm(reference_x, reference_weight, eps)
    (reference_output * upstream).sum().backward()

    x = x_data.clone().requires_grad_()
    weight = weight_data.clone().requires_grad_(train_weight)
    output = training_norm.rms_norm(x, weight, eps)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")
    if train_weight:
        assert_accurate(
            weight.grad, reference_weight.grad, baseline_weight.grad, family="norm", name="dweight"
        )
    else:
        assert weight.grad is None and reference_weight.grad is None
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_data, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shape", ((17, 128), (2, 5, 2560)))
@pytest.mark.parametrize("train_weight", (False, True))
def test_add_rms_norm_training_matches_expanded_reference(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    train_weight: bool,
) -> None:
    torch.manual_seed(20260722)
    eps = 1e-6
    x_data = torch.randn(shape, device="cuda", dtype=dtype)
    residual_data = torch.randn_like(x_data)
    weight_data = torch.randn(shape[-1], device="cuda", dtype=dtype)
    output_upstream = torch.randn_like(x_data)
    residual_upstream = torch.randn_like(x_data)
    original_output_upstream = output_upstream.clone()
    original_residual_upstream = residual_upstream.clone()

    baseline_x = x_data.clone().requires_grad_()
    baseline_residual = residual_data.clone().requires_grad_()
    baseline_weight = weight_data.clone().requires_grad_(train_weight)
    baseline_output, baseline_sum = oracle.add_rms_norm(
        baseline_x, baseline_residual, baseline_weight, eps, normalize="stored_sum", upcast=False
    )
    (
        (baseline_output * output_upstream).sum() + (baseline_sum * residual_upstream).sum()
    ).backward()

    reference_x = x_data.double().requires_grad_()
    reference_residual = residual_data.double().requires_grad_()
    reference_weight = weight_data.double().requires_grad_(train_weight)
    reference_sum = reference_x + reference_residual
    # Normalize the stored sum, with FP64 derivatives of the stored value.
    stored = oracle.add_rms_norm(x_data, residual_data, weight_data, eps, upcast=False)[1]
    reference_sum = reference_sum + (stored.double() - reference_sum).detach()
    reference_output = oracle.rms_norm(reference_sum, reference_weight, eps)
    (
        (reference_output * output_upstream).sum() + (reference_sum * residual_upstream).sum()
    ).backward()

    x = x_data.clone().requires_grad_()
    residual = residual_data.clone().requires_grad_()
    weight = weight_data.clone().requires_grad_(train_weight)
    output, residual_sum = training_norm.add_rms_norm(x, residual, weight, eps)
    ((output * output_upstream).sum() + (residual_sum * residual_upstream).sum()).backward()
    torch.cuda.synchronize()

    input_storage_pointers = {
        x.untyped_storage().data_ptr(),
        residual.untyped_storage().data_ptr(),
    }
    assert output.untyped_storage().data_ptr() not in input_storage_pointers
    assert residual_sum.untyped_storage().data_ptr() not in input_storage_pointers
    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    torch.testing.assert_close(residual_sum, baseline_sum, rtol=0, atol=0)
    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")
    assert_accurate(
        residual.grad,
        reference_residual.grad,
        baseline_residual.grad,
        family="norm",
        name="dresidual",
    )
    if train_weight:
        assert_accurate(
            weight.grad, reference_weight.grad, baseline_weight.grad, family="norm", name="dweight"
        )
    else:
        assert weight.grad is None and reference_weight.grad is None
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_data, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_data, rtol=0, atol=0)
    torch.testing.assert_close(output_upstream, original_output_upstream, rtol=0, atol=0)
    torch.testing.assert_close(residual_upstream, original_residual_upstream, rtol=0, atol=0)


def test_rms_norm_training_preserves_branched_upstream_gradient() -> None:
    """Guard against backward implementations that overwrite ``grad_output``."""

    torch.manual_seed(20260723)
    eps = 1e-6
    shape = (23, 128)
    x_data = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    first_weight_data = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    second_weight_data = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)
    original_upstream = upstream.clone()

    baseline_x = x_data.clone().requires_grad_()
    baseline_first = oracle.rms_norm(baseline_x, first_weight_data, eps, upcast=False)
    baseline_second = oracle.rms_norm(baseline_first, second_weight_data, eps, upcast=False)
    ((baseline_x + baseline_second) * upstream).sum().backward()

    reference_x = x_data.double().requires_grad_()
    reference_first = oracle.rms_norm(reference_x, first_weight_data, eps)
    reference_second = oracle.rms_norm(reference_first, second_weight_data, eps)
    ((reference_x + reference_second) * upstream).sum().backward()

    x = x_data.clone().requires_grad_()
    first = training_norm.rms_norm(x, first_weight_data, eps)
    second = training_norm.rms_norm(first, second_weight_data, eps)
    ((x + second) * upstream).sum().backward()
    torch.cuda.synchronize()

    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


def test_frozen_rms_norm_training_returns_fresh_input_gradient() -> None:
    torch.manual_seed(20260731)
    eps = 1e-6
    x = torch.randn((23, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    original_upstream = upstream.clone()

    output = training_norm.rms_norm(x, weight, eps)
    (grad_x,) = torch.autograd.grad(output, x, upstream)
    torch.cuda.synchronize()

    assert grad_x.untyped_storage().data_ptr() != upstream.untyped_storage().data_ptr()
    assert weight.grad is None
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


def test_rms_norm_training_matches_reference_for_trainable_qkv_view() -> None:
    torch.manual_seed(20260728)
    eps = 1e-6
    num_tokens = 13
    num_heads = 4
    head_dim = 128
    qkv_data = torch.randn(
        (num_tokens, 3 * num_heads * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_data = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(
        (num_tokens, num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )

    baseline_qkv = qkv_data.clone().requires_grad_()
    baseline_q = baseline_qkv.split(num_heads * head_dim, dim=-1)[0].view(
        num_tokens, num_heads, head_dim
    )
    baseline_weight = weight_data.clone().requires_grad_()
    baseline_output = oracle.rms_norm(baseline_q, baseline_weight, eps, upcast=False)
    (baseline_output * upstream).sum().backward()

    reference_qkv = qkv_data.double().requires_grad_()
    reference_q = reference_qkv.split(num_heads * head_dim, dim=-1)[0].view(
        num_tokens, num_heads, head_dim
    )
    reference_weight = weight_data.double().requires_grad_()
    reference_output = oracle.rms_norm(reference_q, reference_weight, eps)
    (reference_output * upstream).sum().backward()

    qkv = qkv_data.clone().requires_grad_()
    q = qkv.split(num_heads * head_dim, dim=-1)[0].view(num_tokens, num_heads, head_dim)
    weight = weight_data.clone().requires_grad_()
    output = training_norm.rms_norm(q, weight, eps)
    (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert not q.is_contiguous()
    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    assert_accurate(qkv.grad, reference_qkv.grad, baseline_qkv.grad, family="norm", name="dx")
    assert_accurate(
        weight.grad, reference_weight.grad, baseline_weight.grad, family="norm", name="dweight"
    )


@pytest.mark.parametrize("prefix_shape", ((13,), (2, 5)))
@pytest.mark.parametrize("component", (0, 1))
def test_frozen_rms_norm_training_preserves_strided_qkv_view(
    prefix_shape: tuple[int, ...],
    component: int,
) -> None:
    torch.manual_seed(20260801 + component)
    eps = 1e-6
    num_heads = 4
    head_dim = 128
    merged_width = 3 * num_heads * head_dim
    qkv_data = torch.randn(
        prefix_shape + (merged_width,),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(
        prefix_shape + (num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )

    baseline_qkv = qkv_data.clone().requires_grad_()
    baseline_component = baseline_qkv.split(num_heads * head_dim, dim=-1)[component].view(
        prefix_shape + (num_heads, head_dim)
    )
    baseline_output = oracle.rms_norm(baseline_component, weight, eps, upcast=False)
    (baseline_output * upstream).sum().backward()

    reference_qkv = qkv_data.double().requires_grad_()
    reference_component = reference_qkv.split(num_heads * head_dim, dim=-1)[component].view(
        prefix_shape + (num_heads, head_dim)
    )
    reference_output = oracle.rms_norm(reference_component, weight, eps)
    (reference_output * upstream).sum().backward()

    qkv = qkv_data.clone().requires_grad_()
    qkv_component = qkv.split(num_heads * head_dim, dim=-1)[component].view(
        prefix_shape + (num_heads, head_dim)
    )
    input_signature = (
        qkv_component.untyped_storage().data_ptr(),
        qkv_component.storage_offset(),
        qkv_component.stride(),
    )
    saved_signatures: list[tuple[int, int, tuple[int, ...]]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_signatures.append(
            (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor.stride(),
            )
        )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = training_norm.rms_norm(qkv_component, weight, eps)
        (output * upstream).sum().backward()
    torch.cuda.synchronize()

    assert not qkv_component.is_contiguous()
    assert qkv_component.stride(-1) == 1
    assert output.is_contiguous()
    assert output.untyped_storage().data_ptr() != qkv_component.untyped_storage().data_ptr()
    assert input_signature in saved_signatures
    assert weight.grad is None
    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    assert_accurate(qkv.grad, reference_qkv.grad, baseline_qkv.grad, family="norm", name="dx")
    torch.testing.assert_close(qkv, qkv_data, rtol=0, atol=0)


def test_frozen_rms_norm_training_accumulates_across_two_backward_calls() -> None:
    torch.manual_seed(20260803)
    eps = 1e-6
    x_data = torch.randn((3, 7, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    first_upstream = torch.randn_like(x_data)
    second_upstream = torch.randn_like(x_data)

    baseline_x = x_data.clone().requires_grad_()
    baseline_output = oracle.rms_norm(baseline_x, weight, eps, upcast=False)
    baseline_output.backward(first_upstream, retain_graph=True)
    baseline_output.backward(second_upstream)

    reference_x = x_data.double().requires_grad_()
    reference_output = oracle.rms_norm(reference_x, weight, eps)
    reference_output.backward(first_upstream.double(), retain_graph=True)
    reference_output.backward(second_upstream.double())

    x = x_data.clone().requires_grad_()
    output = training_norm.rms_norm(x, weight, eps)
    output.backward(first_upstream, retain_graph=True)
    output.backward(second_upstream)
    torch.cuda.synchronize()

    assert weight.grad is None
    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")


def test_frozen_rms_norm_training_uses_current_cuda_stream() -> None:
    torch.manual_seed(20260804)
    eps = 1e-6
    x_data = torch.randn((19, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)
    baseline_x = x_data.clone().requires_grad_()
    baseline_output = oracle.rms_norm(baseline_x, weight, eps, upcast=False)
    (baseline_output * upstream).sum().backward()

    reference_x = x_data.double().requires_grad_()
    reference_output = oracle.rms_norm(reference_x, weight, eps)
    (reference_output * upstream).sum().backward()

    current_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current_stream)
    with torch.cuda.stream(stream):
        assert torch.cuda.current_stream() == stream
        x = x_data.clone().requires_grad_()
        output = training_norm.rms_norm(x, weight, eps)
        (output * upstream).sum().backward()
    stream.synchronize()

    assert weight.grad is None
    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")


def test_frozen_rms_norm_training_rejects_prefix_that_requires_copy() -> None:
    x = torch.randn((2, 3, 4, 128), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="two strides without a copy"):
        training_norm.rms_norm(x, weight, 1e-6)


@pytest.mark.parametrize("hidden_size", (4097, 16_385))
def test_frozen_rms_norm_matches_reference_across_launch_boundaries(hidden_size: int) -> None:
    """Exercise masked tails and the 16-/32-warp launches with few tokens."""

    torch.manual_seed(20260909)
    x_data = torch.randn((3, hidden_size), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)
    original_upstream = upstream.clone()

    baseline_x = x_data.clone().requires_grad_()
    baseline_output = oracle.rms_norm(baseline_x, weight, 1e-6, upcast=False)
    baseline_output.backward(upstream)

    reference_x = x_data.double().requires_grad_()
    reference_output = oracle.rms_norm(reference_x, weight, 1e-6)
    reference_output.backward(upstream.double())

    x = x_data.clone().requires_grad_()
    output = training_norm.rms_norm(x, weight, 1e-6)
    output.backward(upstream)

    assert_accurate(output, reference_output, baseline_output, family="norm", name="output")
    assert_accurate(x.grad, reference_x.grad, baseline_x.grad, family="norm", name="dx")
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


@pytest.mark.parametrize(
    "operation",
    (inference_norm.rms_norm, inference_norm.rms_norm_),
)
def test_inference_rms_norm_rejects_active_weight_gradient(operation) -> None:
    torch.manual_seed(20260729)
    eps = 1e-6
    x = torch.randn((11, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    original_x = x.clone()

    with pytest.raises(ValueError, match="frozen weight"):
        operation(x, weight, eps)

    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


def test_inference_add_rms_norm_rejects_active_weight_gradient() -> None:
    torch.manual_seed(20260730)
    eps = 1e-6
    x = torch.randn((11, 128), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    original_x = x.clone()
    original_residual = residual.clone()

    with pytest.raises(ValueError, match="frozen weight"):
        inference_norm.add_rms_norm_(x, residual, weight, eps)

    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("hidden_size", (256, 4096, 4097, 5120))
def test_zero_centered_training_rms_norm_matches_fp64_autograd(
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261006)
    eps = 1e-6
    x_data = torch.randn((3, 7, hidden_size), device="cuda", dtype=dtype)
    # One row's mean square sits far below eps, so the backward must add eps too.
    x_data[0, 0].mul_(1e-4)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    upstream = torch.randn_like(x_data)
    original_upstream = upstream.clone()

    baseline_x = x_data.clone().requires_grad_()
    oracle.rms_norm(
        baseline_x, weight, eps, zero_centered=True, out_dtype=dtype, upcast=False
    ).backward(upstream)

    reference_x = x_data.double().requires_grad_()
    oracle.rms_norm(reference_x, weight, eps, zero_centered=True).backward(upstream.double())

    x = x_data.clone().requires_grad_()
    output = training_norm.rms_norm(x, weight, eps, zero_centered=True)
    output.backward(upstream)
    with torch.inference_mode():
        inference_output = inference_norm.rms_norm(x_data, weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, inference_output, rtol=0, atol=0)
    oracle.check_rms_norm(output, x_data, weight, eps, zero_centered=True)
    assert_accurate(
        x.grad, reference_x.grad, baseline_x.grad.to(x.grad.dtype), family="norm", name="dx"
    )
    assert x.grad.untyped_storage().data_ptr() != upstream.untyped_storage().data_ptr()
    assert weight.grad is None
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original_upstream, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("hidden_size", (4096, 5120))
def test_zero_centered_training_add_rms_norm_matches_fp64_autograd(
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    torch.manual_seed(20261007)
    eps = 1e-6
    x_data = torch.randn((2, 9, hidden_size), device="cuda", dtype=dtype)
    residual_data = torch.randn_like(x_data)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_upstream = torch.randn_like(x_data)
    sum_upstream = torch.randn_like(x_data)
    summed_data = x_data + residual_data

    # HF normalizes the sum rounded to the input dtype; both inputs receive its gradient.
    baseline_x = x_data.clone().requires_grad_()
    baseline_residual = residual_data.clone().requires_grad_()
    baseline_output, baseline_sum = oracle.add_rms_norm(
        baseline_x,
        baseline_residual,
        weight,
        eps,
        zero_centered=True,
        normalize="stored_sum",
        upcast=False,
    )
    torch.autograd.backward((baseline_output, baseline_sum), (output_upstream, sum_upstream))

    reference_sum = summed_data.double().requires_grad_()
    torch.autograd.backward(
        (oracle.rms_norm(reference_sum, weight, eps, zero_centered=True), reference_sum),
        (output_upstream.double(), sum_upstream.double()),
    )

    x = x_data.clone().requires_grad_()
    residual = residual_data.clone().requires_grad_()
    output, summed = training_norm.add_rms_norm(x, residual, weight, eps, zero_centered=True)
    torch.autograd.backward((output, summed), (output_upstream, sum_upstream))
    fused_x, fused_residual = x_data.clone(), residual_data.clone()
    with torch.inference_mode():
        inference_norm.add_rms_norm_(
            fused_x, fused_residual, weight, eps, zero_centered=True, normalize="stored_sum"
        )
    torch.cuda.synchronize()

    input_storage_pointers = {
        x.untyped_storage().data_ptr(),
        residual.untyped_storage().data_ptr(),
    }
    assert output.untyped_storage().data_ptr() not in input_storage_pointers
    assert summed.untyped_storage().data_ptr() not in input_storage_pointers
    torch.testing.assert_close(summed, summed_data, rtol=0, atol=0)
    torch.testing.assert_close(summed, fused_residual, rtol=0, atol=0)
    torch.testing.assert_close(output, fused_x, rtol=0, atol=0)
    assert_accurate(x.grad, reference_sum.grad, baseline_x.grad, family="norm", name="dx")
    assert_accurate(
        residual.grad, reference_sum.grad, baseline_residual.grad, family="norm", name="dresidual"
    )
    assert weight.grad is None
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_data, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (128, 2048, 4096, 4097, 5120))
def test_zero_centered_training_add_rms_norm_keeps_an_fp32_residual_like_inference(
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    """Training and inference run the same compiled residual-stream kernel, so their outputs
    are equal bitwise; the FP32 residual's gradient stays FP32 and x's is rounded once."""

    torch.manual_seed(20261018)
    eps = 1e-6
    x_data = torch.randn((4, 128, hidden_size), device="cuda", dtype=dtype)
    residual_data = torch.randn((4, 128, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_upstream = torch.randn_like(x_data)
    sum_upstream = torch.randn_like(residual_data)
    summed_data = residual_data + x_data.float()

    # Both inputs receive the gradient of the stored FP32 sum.
    baseline_x = x_data.clone().requires_grad_()
    baseline_residual = residual_data.clone().requires_grad_()
    baseline_output, baseline_sum = oracle.add_rms_norm(
        baseline_x,
        baseline_residual,
        weight,
        eps,
        zero_centered=True,
        normalize="stored_sum",
        upcast=False,
    )
    torch.autograd.backward((baseline_output, baseline_sum), (output_upstream, sum_upstream))

    reference_sum = summed_data.double().requires_grad_()
    torch.autograd.backward(
        (oracle.rms_norm(reference_sum, weight, eps, zero_centered=True), reference_sum),
        (output_upstream.double(), sum_upstream.double()),
    )

    x = x_data.clone().requires_grad_()
    residual = residual_data.clone().requires_grad_()
    output, summed = training_norm.add_rms_norm(x, residual, weight, eps, zero_centered=True)
    torch.autograd.backward((output, summed), (output_upstream, sum_upstream))
    fused_x, fused_residual = x_data.clone(), residual_data.clone()
    with torch.inference_mode():
        inference_norm.AddRMSNorm(provider="triton")(
            fused_x, fused_residual, weight, eps, zero_centered=True
        )
    torch.cuda.synchronize()

    assert output.dtype == dtype and summed.dtype == torch.float32
    assert x.grad.dtype == dtype and residual.grad.dtype == torch.float32
    torch.testing.assert_close(summed, summed_data, rtol=0, atol=0)
    torch.testing.assert_close(summed, fused_residual, rtol=0, atol=0)
    torch.testing.assert_close(output, fused_x, rtol=0, atol=0)
    assert_accurate(
        residual.grad, reference_sum.grad, baseline_residual.grad, family="norm", name="dresidual"
    )
    assert_accurate(x.grad, reference_sum.grad, baseline_x.grad, family="norm", name="dx")
    torch.testing.assert_close(x, x_data, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_data, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_zero_centered_training_rms_norm_backpropagates_into_merged_projection(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20261008)
    eps = 1e-6
    num_tokens, num_q_heads, num_kv_heads, head_dim = 11, 8, 2, 256
    q_size, kv_size = num_q_heads * head_dim, num_kv_heads * head_dim
    widths = (q_size, kv_size, kv_size, q_size)  # q | k | v | output gate
    projection_data = torch.randn((num_tokens, sum(widths)), device="cuda", dtype=dtype)
    q_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
    k_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
    q_upstream = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k_upstream = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)

    def heads(projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, _, _ = projection.split(widths, dim=-1)
        return (
            q.view(num_tokens, num_q_heads, head_dim),
            k.view(num_tokens, num_kv_heads, head_dim),
        )

    baseline_projection = projection_data.clone().requires_grad_()
    baseline_q, baseline_k = heads(baseline_projection)
    torch.autograd.backward(
        (
            oracle.rms_norm(
                baseline_q, q_weight, eps, zero_centered=True, out_dtype=dtype, upcast=False
            ),
            oracle.rms_norm(
                baseline_k, k_weight, eps, zero_centered=True, out_dtype=dtype, upcast=False
            ),
        ),
        (q_upstream, k_upstream),
    )

    reference_projection = projection_data.double().requires_grad_()
    reference_q, reference_k = heads(reference_projection)
    torch.autograd.backward(
        (
            oracle.rms_norm(reference_q, q_weight, eps, zero_centered=True),
            oracle.rms_norm(reference_k, k_weight, eps, zero_centered=True),
        ),
        (q_upstream.double(), k_upstream.double()),
    )

    projection = projection_data.clone().requires_grad_()
    q, k = heads(projection)
    q_output = training_norm.rms_norm(q, q_weight, eps, zero_centered=True)
    k_output = training_norm.rms_norm(k, k_weight, eps, zero_centered=True)
    torch.autograd.backward((q_output, k_output), (q_upstream, k_upstream))
    with torch.inference_mode():
        data_q, data_k = heads(projection_data)
        inference_q = inference_norm.rms_norm(data_q, q_weight, eps, zero_centered=True)
        inference_k = inference_norm.rms_norm(data_k, k_weight, eps, zero_centered=True)
    torch.cuda.synchronize()

    assert not q.is_contiguous() and not k.is_contiguous()
    torch.testing.assert_close(q_output, inference_q, rtol=0, atol=0)
    torch.testing.assert_close(k_output, inference_k, rtol=0, atol=0)
    assert_accurate(
        projection.grad,
        reference_projection.grad,
        baseline_projection.grad.to(projection.grad.dtype),
        family="norm",
        name="dx",
    )
    assert projection.grad[:, q_size + kv_size :].count_nonzero().item() == 0
    torch.testing.assert_close(projection, projection_data, rtol=0, atol=0)


def test_zero_centered_training_rms_norm_copies_only_an_unviewable_upstream() -> None:
    torch.manual_seed(20261009)
    eps = 1e-6
    x_data = torch.randn((5, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)

    baseline_x = x_data.clone().requires_grad_()
    oracle.rms_norm(
        baseline_x, weight, eps, zero_centered=True, out_dtype=x_data.dtype, upcast=False
    ).sum().backward()

    reference_x = x_data.double().requires_grad_()
    oracle.rms_norm(reference_x, weight, eps, zero_centered=True).sum().backward()

    x = x_data.clone().requires_grad_()
    # sum() hands backward an expanded, stride-0 gradient.
    training_norm.rms_norm(x, weight, eps, zero_centered=True).sum().backward()
    torch.cuda.synchronize()

    assert_accurate(
        x.grad, reference_x.grad, baseline_x.grad.to(x.grad.dtype), family="norm", name="dx"
    )


@pytest.mark.parametrize("use_reentrant", (False, True))
def test_zero_centered_training_norms_recompute_bitwise_under_checkpointing(
    use_reentrant: bool,
) -> None:
    torch.manual_seed(20261010)
    eps = 1e-6
    x_data = torch.randn((4, 6, 4096), device="cuda", dtype=torch.bfloat16)
    residual_data = torch.randn_like(x_data)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)

    def block(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        normalized, summed = training_norm.add_rms_norm(
            x, residual, weight, eps, zero_centered=True
        )
        return summed + training_norm.rms_norm(normalized, weight, eps, zero_centered=True)

    eager_x = x_data.clone().requires_grad_()
    eager_residual = residual_data.clone().requires_grad_()
    eager_output = block(eager_x, eager_residual)
    eager_output.backward(upstream)

    checkpoint_x = x_data.clone().requires_grad_()
    checkpoint_residual = residual_data.clone().requires_grad_()
    checkpoint_output = checkpoint(
        block,
        checkpoint_x,
        checkpoint_residual,
        use_reentrant=use_reentrant,
        preserve_rng_state=False,
    )
    checkpoint_output.backward(upstream)
    torch.cuda.synchronize()

    torch.testing.assert_close(checkpoint_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_x.grad, eager_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_residual.grad, eager_residual.grad, rtol=0, atol=0)


@pytest.mark.parametrize("use_reentrant", (False, True))
def test_zero_centered_fp32_residual_recomputes_bitwise_under_checkpointing(
    use_reentrant: bool,
) -> None:
    torch.manual_seed(20261019)
    eps = 1e-6
    x_data = torch.randn((4, 6, 4096), device="cuda", dtype=torch.bfloat16)
    residual_data = torch.randn((4, 6, 4096), device="cuda", dtype=torch.float32)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(residual_data)

    def block(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        normalized, summed = training_norm.add_rms_norm(
            x, residual, weight, eps, zero_centered=True
        )
        # The next block's input norm reads the FP32 residual into x's dtype.
        next_input = training_norm.rms_norm(
            summed, weight, eps, zero_centered=True, out_dtype=x.dtype
        )
        return summed + normalized * next_input

    eager_x = x_data.clone().requires_grad_()
    eager_residual = residual_data.clone().requires_grad_()
    eager_output = block(eager_x, eager_residual)
    eager_output.backward(upstream)

    checkpoint_x = x_data.clone().requires_grad_()
    checkpoint_residual = residual_data.clone().requires_grad_()
    checkpoint_output = checkpoint(
        block,
        checkpoint_x,
        checkpoint_residual,
        use_reentrant=use_reentrant,
        preserve_rng_state=False,
    )
    checkpoint_output.backward(upstream)
    torch.cuda.synchronize()

    assert eager_residual.grad.dtype == torch.float32
    torch.testing.assert_close(checkpoint_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_x.grad, eager_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_residual.grad, eager_residual.grad, rtol=0, atol=0)


def test_zero_centered_training_rms_norm_uses_current_stream_and_accepts_empty_rows() -> None:
    torch.manual_seed(20261011)
    eps = 1e-6
    x_data = torch.randn((19, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)
    baseline_x = x_data.clone().requires_grad_()
    oracle.rms_norm(
        baseline_x, weight, eps, zero_centered=True, out_dtype=x_data.dtype, upcast=False
    ).backward(upstream)

    reference_x = x_data.double().requires_grad_()
    oracle.rms_norm(reference_x, weight, eps, zero_centered=True).backward(upstream.double())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = x_data.clone().requires_grad_()
        training_norm.rms_norm(x, weight, eps, zero_centered=True).backward(upstream)
        empty = torch.empty((0, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        empty_output = training_norm.rms_norm(empty, weight, eps, zero_centered=True)
        empty_output.sum().backward()
    stream.synchronize()

    assert_accurate(
        x.grad, reference_x.grad, baseline_x.grad.to(x.grad.dtype), family="norm", name="dx"
    )
    assert empty_output.shape == empty.shape
    assert empty.grad is not None and empty.grad.shape == empty.shape


@pytest.mark.parametrize("grad_enabled", (True, False))
def test_zero_centered_training_norms_reject_a_trainable_weight(grad_enabled: bool) -> None:
    x = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn_like(x)
    weight = torch.zeros(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.set_grad_enabled(grad_enabled):
        with pytest.raises(ValueError, match="frozen weight"):
            training_norm.rms_norm(x, weight, 1e-6, zero_centered=True)
        with pytest.raises(ValueError, match="frozen weight"):
            training_norm.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)


@pytest.mark.parametrize("fp32_residual", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (128, 2048, 4096, 4097, 5120))
def test_residual_norms_compile_one_forward_and_one_backward(
    fp32_residual: bool,
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    """Every entry point, mode and token count of one hidden size and dtype set runs one
    forward and one backward binary.

    Triton keys a compiled kernel on its arguments' dtypes, pointer alignment, integer
    values and constexprs (Triton 3.7.1 `runtime/jit.py:720,860-885`). The residual-stream
    kernels take runtime strides and eps but no row count. Their mode flags are integers
    they do not specialize on; aligned operands in this test ensure the single row of a
    ``[1, hidden]`` view with row stride 1 compiles nothing new either.
    """

    torch.manual_seed(20261021)
    eps = 1e-6
    residual_dtype = torch.float32 if fp32_residual else dtype
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    kernels = (
        zero_centered_kernels._residual_norm_kernel,
        zero_centered_kernels._residual_norm_backward_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    def operand(dtype: torch.dtype, rows: int, transposed: bool) -> torch.Tensor:
        if transposed:
            return torch.randn((hidden_size, 1), device="cuda", dtype=dtype).t()
        return torch.randn((rows, hidden_size), device="cuda", dtype=dtype)

    for rows, transposed in ((1, False), (1, True), (7, False), (512, False), (8192, False)):
        x = operand(dtype, rows, transposed)
        residual = operand(residual_dtype, rows, transposed)
        with torch.inference_mode():
            for normalize in ("fp32_sum", "stored_sum"):
                for store_residual in (True, False):
                    inference_norm.AddRMSNorm(provider="triton")(
                        x,
                        residual,
                        weight,
                        eps,
                        zero_centered=True,
                        store_residual=store_residual,
                        normalize=normalize,
                    )
            inference_norm.residual_rms_norm(
                residual, weight, eps, zero_centered=True, out_dtype=dtype
            )
        # Backward with and without the gradient of the sum or of the passed residual.
        for used in (2, 1):
            for save in ("sum", "inputs"):
                x = operand(dtype, rows, transposed).requires_grad_()
                residual = operand(residual_dtype, rows, transposed).requires_grad_()
                outputs = training_norm.add_rms_norm(
                    x, residual, weight, eps, zero_centered=True, save=save
                )[:used]
                torch.autograd.backward(outputs, [torch.randn_like(output) for output in outputs])
            residual = operand(residual_dtype, rows, transposed).requires_grad_()
            normalized, passed = training_norm.residual_rms_norm(
                residual, weight, eps, zero_centered=True, out_dtype=dtype
            )
            outputs = (normalized, passed)[:used]
            torch.autograd.backward(outputs, [torch.randn_like(output) for output in outputs])
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    assert [len(kernel.device_caches[device][0]) for kernel in kernels] == [1, 1]


@pytest.mark.parametrize("fp32_residual", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (128, 2048, 4096, 4097, 5120))
def test_residual_norms_agree_between_training_and_inference(
    fp32_residual: bool,
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    """Reading the stored sum, in place or into new storage, fused or on that sum alone, every
    entry point gives the same bits, and backward gives the same gradients from either kept
    state."""

    torch.manual_seed(20261023)
    eps = 1e-6
    residual_dtype = torch.float32 if fp32_residual else dtype
    x_data = torch.randn((4, 33, hidden_size), device="cuda", dtype=dtype)
    residual_data = torch.randn((4, 33, hidden_size), device="cuda", dtype=residual_dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    upstream = torch.randn_like(x_data)
    sum_upstream = torch.randn_like(residual_data)

    x, residual = x_data.clone(), residual_data.clone()
    unstored_x, unstored_residual = x_data.clone(), residual_data.clone()
    with torch.inference_mode():
        inference_norm.add_rms_norm_(
            x, residual, weight, eps, zero_centered=True, normalize="stored_sum"
        )
        inference_norm.add_rms_norm_(
            unstored_x,
            unstored_residual,
            weight,
            eps,
            zero_centered=True,
            store_residual=False,
            normalize="stored_sum",
        )
        residual_output = inference_norm.residual_rms_norm(
            residual, weight, eps, zero_centered=True, out_dtype=dtype
        )
    trained = {}
    for save in ("sum", "inputs"):
        x_leaf = x_data.clone().requires_grad_()
        residual_leaf = residual_data.clone().requires_grad_()
        normalized, summed = training_norm.add_rms_norm(
            x_leaf, residual_leaf, weight, eps, zero_centered=True, save=save
        )
        torch.autograd.backward((normalized, summed), (upstream, sum_upstream))
        trained[save] = (normalized, summed, x_leaf.grad, residual_leaf.grad)
    trained_residual_output, _ = training_norm.residual_rms_norm(
        residual, weight, eps, zero_centered=True, out_dtype=dtype
    )
    torch.cuda.synchronize()

    for output in (unstored_x, residual_output, trained_residual_output):
        torch.testing.assert_close(output, x, rtol=0, atol=0)
    for normalized, summed, _, _ in trained.values():
        torch.testing.assert_close(normalized, x, rtol=0, atol=0)
        torch.testing.assert_close(summed, residual, rtol=0, atol=0)
    torch.testing.assert_close(unstored_residual, residual_data, rtol=0, atol=0)
    for kept_sum, kept_inputs in zip(trained["sum"][2:], trained["inputs"][2:], strict=True):
        torch.testing.assert_close(kept_sum, kept_inputs, rtol=0, atol=0)


@pytest.mark.parametrize("fp32_residual", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("hidden_size", (256, 4096, 4097, 5120))
def test_residual_norm_gradients_match_the_grouped_backward_and_fp64(
    fp32_residual: bool,
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    """Backward forms one FP32 gradient of the stored value and rounds it once per input.

    The oracle is the grouped backward of the stored sum as an FP32 input plus the sum's own
    gradient, added in FP32 and rounded to each input's dtype; FP64 autograd bounds it.
    """

    torch.manual_seed(20261024)
    eps = 1e-6
    residual_dtype = torch.float32 if fp32_residual else dtype
    x_data = torch.randn((2, 9, hidden_size), device="cuda", dtype=dtype)
    residual_data = torch.randn((2, 9, hidden_size), device="cuda", dtype=residual_dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x_data)
    sum_upstream = torch.randn_like(residual_data)
    summed_data = (x_data.float() + residual_data.float()).to(residual_dtype)

    stored = summed_data.to(torch.float32, copy=True).requires_grad_()
    grouped = training_norm.rms_norm(stored, weight, eps, zero_centered=True, out_dtype=dtype)
    (norm_gradient,) = torch.autograd.grad(grouped, stored, upstream)
    grouped_gradient = norm_gradient + sum_upstream.float()
    baseline_x = x_data.clone().requires_grad_()
    baseline_residual = residual_data.clone().requires_grad_()
    baseline_output, baseline_sum = oracle.add_rms_norm(
        baseline_x,
        baseline_residual,
        weight,
        eps,
        zero_centered=True,
        normalize="stored_sum",
        upcast=False,
    )
    torch.autograd.backward((baseline_output, baseline_sum), (upstream, sum_upstream))

    reference = summed_data.double().requires_grad_()
    torch.autograd.backward(
        (oracle.rms_norm(reference, weight, eps, zero_centered=True), reference),
        (upstream.double(), sum_upstream.double()),
    )

    x = x_data.clone().requires_grad_()
    residual = residual_data.clone().requires_grad_()
    normalized, summed = training_norm.add_rms_norm(x, residual, weight, eps, zero_centered=True)
    torch.autograd.backward((normalized, summed), (upstream, sum_upstream))
    residual_leaf = summed_data.clone().requires_grad_()
    residual_normalized, passed = training_norm.residual_rms_norm(
        residual_leaf, weight, eps, zero_centered=True, out_dtype=dtype
    )
    torch.autograd.backward((residual_normalized, passed), (upstream, sum_upstream))
    torch.cuda.synchronize()

    assert x.grad.dtype == dtype and residual.grad.dtype == residual_dtype
    torch.testing.assert_close(x.grad, grouped_gradient.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(residual.grad, grouped_gradient.to(residual_dtype), rtol=0, atol=0)
    torch.testing.assert_close(
        residual_leaf.grad, grouped_gradient.to(residual_dtype), rtol=0, atol=0
    )
    assert_accurate(x.grad, reference.grad, baseline_x.grad, family="norm", name="dx")
    assert_accurate(
        residual.grad, reference.grad, baseline_residual.grad, family="norm", name="dresidual"
    )


@pytest.mark.parametrize("use_reentrant", (False, True))
@pytest.mark.parametrize("residual_dtype", (torch.bfloat16, torch.float32))
def test_residual_norm_segments_recompute_bitwise_under_checkpointing(
    residual_dtype: torch.dtype,
    use_reentrant: bool,
) -> None:
    """Layers in pair form, chained in checkpointed segments, match one unsegmented run.

    Each layer runs its input norm, a mixer, its post-attention norm (keeping its inputs)
    and an MLP. A segment ends in a torch add of its last pair and the next one starts with
    the residual's own norm: that add is the fused add and that norm the fused norm's binary,
    so segments of one or two layers, recomputed in backward, give the output and every
    gradient of a single segment bit for bit.
    """

    torch.manual_seed(20261025)
    eps = 1e-6
    num_layers, num_tokens, hidden_size = 4, 24, 4096
    dtype = torch.bfloat16
    norm_weights = torch.randn((2 * num_layers + 1, hidden_size), device="cuda", dtype=dtype)
    scale_data = torch.randn((2 * num_layers, hidden_size), device="cuda", dtype=dtype)
    embedding_data = torch.randn((num_tokens, hidden_size), device="cuda", dtype=dtype)
    upstream = torch.randn_like(embedding_data)

    def segment(
        first: int, last: int, residual: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        x = None
        for layer in range(first, last):
            input_weight, post_weight = norm_weights[2 * layer], norm_weights[2 * layer + 1]
            if x is None:
                x, residual = training_norm.residual_rms_norm(
                    residual, input_weight, eps, zero_centered=True, out_dtype=dtype
                )
            else:
                x, residual = training_norm.add_rms_norm(
                    x, residual, input_weight, eps, zero_centered=True
                )
            x, residual = training_norm.add_rms_norm(
                x * scales[2 * layer], residual, post_weight, eps, zero_centered=True, save="inputs"
            )
            x = x * scales[2 * layer + 1]
        return residual + x

    results = {}
    for layers_per_segment in (num_layers, 1, 2):
        embedding = embedding_data.clone().requires_grad_()
        scales = scale_data.clone().requires_grad_()
        residual = embedding.to(residual_dtype)
        for first in range(0, num_layers, layers_per_segment):
            last = first + layers_per_segment
            if layers_per_segment == num_layers:
                residual = segment(first, last, residual, scales)
            else:
                residual = checkpoint(
                    segment,
                    first,
                    last,
                    residual,
                    scales,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=False,
                )
        output, _ = training_norm.residual_rms_norm(
            residual, norm_weights[-1], eps, zero_centered=True, out_dtype=dtype
        )
        output.backward(upstream)
        results[layers_per_segment] = (output, embedding.grad, scales.grad)
    torch.cuda.synchronize()

    unsegmented = results.pop(num_layers)
    assert all(tensor is not None for tensor in unsegmented)
    for segmented in results.values():
        for actual, expected in zip(segmented, unsegmented, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", (2048, 4096, 4104))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_standard_fp32_residual_triton_training_bits_and_backward(hidden, dtype):
    torch.manual_seed(20261016)
    data = torch.randn(3, hidden, device="cuda", dtype=dtype)
    residual_data = torch.randn(3, hidden, device="cuda", dtype=torch.float32)
    weight = torch.randn(
        hidden, device="cuda", dtype=torch.bfloat16 if dtype == torch.bfloat16 else dtype
    )
    upstream, residual_upstream = torch.randn_like(data), torch.randn_like(residual_data)
    reference_sum = (data.float() + residual_data).double().requires_grad_()
    reference_norm = oracle.rms_norm(reference_sum, weight)
    baseline_x = data.clone().requires_grad_()
    baseline_residual = residual_data.clone().requires_grad_()
    baseline_output, baseline_sum = oracle.add_rms_norm(
        baseline_x, baseline_residual, weight, normalize="stored_sum", upcast=False
    )
    torch.autograd.backward((baseline_output, baseline_sum), (upstream, residual_upstream))
    baseline_leaf = (data.float() + residual_data).detach().requires_grad_()
    baseline_norm = oracle.rms_norm(baseline_leaf, weight, out_dtype=dtype, upcast=False)
    torch.autograd.backward((baseline_norm, baseline_leaf), (upstream, residual_upstream))
    torch.autograd.backward(
        (reference_norm, reference_sum), (upstream.double(), residual_upstream.double())
    )
    trained = []
    for save in ("sum", "inputs"):
        x, residual = data.clone().requires_grad_(), residual_data.clone().requires_grad_()
        output, summed = training_norm.add_rms_norm(
            x, residual, weight, 1e-6, zero_centered=False, save=save
        )
        torch.autograd.backward((output, summed), (upstream, residual_upstream))
        assert_accurate(x.grad, reference_sum.grad, baseline_x.grad, family="norm", name="dx")
        assert_accurate(
            residual.grad,
            reference_sum.grad,
            baseline_residual.grad,
            family="norm",
            name="dresidual",
        )
        trained.append((output, summed, x.grad, residual.grad))
    for a, b in zip(*trained, strict=True):
        _assert_norm_bits(a, b)
    with torch.no_grad():
        for normalize in ("fp32_sum", "stored_sum"):
            x, residual = data.clone(), residual_data.clone()
            inference_norm.AddRMSNorm(provider="triton")(
                x, residual, weight, 1e-6, zero_centered=False, normalize=normalize
            )
            _assert_norm_bits(x, trained[0][0])
            _assert_norm_bits(residual, trained[0][1])
        inferred = inference_norm.residual_rms_norm(
            residual, weight, 1e-6, zero_centered=False, out_dtype=dtype
        )
    leaf = residual.detach().clone().requires_grad_()
    normalized, passed = training_norm.residual_rms_norm(
        leaf, weight, 1e-6, zero_centered=False, out_dtype=dtype
    )
    _assert_norm_bits(normalized, inferred)
    torch.autograd.backward((normalized, passed), (upstream, residual_upstream))
    assert_accurate(
        leaf.grad, reference_sum.grad, baseline_leaf.grad, family="norm", name="dresidual"
    )
