# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _assert_gradient_close(actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    assert actual is not None and expected is not None
    if expected.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return

    difference = actual.float() - expected.float()
    reference = expected.float()
    error_rms = float(difference.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    max_absolute = float(difference.abs().max())
    allowed_rms = 2e-3 + 1.5e-2 * reference_rms
    assert error_rms <= allowed_rms, (
        f"gradient RMS error {error_rms} exceeds {allowed_rms}; "
        f"reference_rms={reference_rms}, max_absolute={max_absolute}"
    )
    # Avoid division only for an exactly zero reference. Small nonzero gradients
    # must retain the original relative gate: replacing them with zero is a bug.
    if reference_rms > 0:
        relative_l2 = float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
        )
        assert relative_l2 <= 2e-2, (
            f"gradient relative L2 error {relative_l2} exceeds 0.02; max_absolute={max_absolute}"
        )


def test_gradient_check_rejects_dropped_small_gradients() -> None:
    expected = torch.full((4, 8), 1e-3, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="relative L2"):
        _assert_gradient_close(torch.zeros_like(expected), expected)


def test_chunked_fused_linear_cross_entropy_matches_hidden_gradient() -> None:
    torch.manual_seed(20260719)
    hidden_data = torch.randn((19, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((257, 64), device="cuda", dtype=torch.bfloat16)
    target_ids = torch.randint(0, 257, (19,), device="cuda", dtype=torch.int64)

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, weight).float(),
        target_ids,
        reduction="sum",
    )
    reference_loss.backward()

    hidden_states = hidden_data.detach().clone().requires_grad_()
    loss = FusedLinearCrossEntropyLoss(chunk_size=7).forward(
        hidden_states,
        weight,
        target_ids,
    )
    loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(loss, reference_loss, rtol=0.003, atol=0.003)
    _assert_gradient_close(hidden_states.grad, reference_hidden.grad)


@pytest.mark.parametrize(
    ("train_hidden", "train_weight", "train_bias"),
    (
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ),
)
def test_chunked_fused_linear_cross_entropy_honors_independent_gradients(
    train_hidden: bool,
    train_weight: bool,
    train_bias: bool,
) -> None:
    torch.manual_seed(20260719)
    hidden_data = torch.randn((19, 64), device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn((257, 64), device="cuda", dtype=torch.bfloat16)
    bias_data = torch.randn((257,), device="cuda", dtype=torch.bfloat16)
    target_ids = torch.randint(0, 257, (19,), device="cuda", dtype=torch.int64)

    reference_hidden = hidden_data.detach().clone().requires_grad_(train_hidden)
    reference_weight = weight_data.detach().clone().requires_grad_(train_weight)
    reference_bias = bias_data.detach().clone().requires_grad_(train_bias)
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, reference_weight, reference_bias).float(),
        target_ids,
        reduction="sum",
    )
    reference_loss.backward()

    hidden_states = hidden_data.detach().clone().requires_grad_(train_hidden)
    weight = weight_data.detach().clone().requires_grad_(train_weight)
    bias = bias_data.detach().clone().requires_grad_(train_bias)
    loss = FusedLinearCrossEntropyLoss(chunk_size=7).forward(
        hidden_states,
        weight,
        target_ids,
        bias=bias,
    )
    loss.backward()
    torch.cuda.synchronize()

    for is_trainable, actual, expected in (
        (train_hidden, hidden_states.grad, reference_hidden.grad),
        (train_weight, weight.grad, reference_weight.grad),
        (train_bias, bias.grad, reference_bias.grad),
    ):
        if is_trainable:
            _assert_gradient_close(actual, expected)
        else:
            assert actual is None and expected is None


@pytest.mark.parametrize("use_non_default_stream", (False, True))
def test_chunked_fused_linear_cross_entropy_scales_without_mutating_inputs(
    use_non_default_stream: bool,
) -> None:
    torch.manual_seed(20260719)
    hidden_data = torch.randn((19, 64), device="cuda", dtype=torch.bfloat16)
    weight_data = torch.randn((257, 64), device="cuda", dtype=torch.bfloat16)
    bias_data = torch.randn((257,), device="cuda", dtype=torch.bfloat16)
    target_ids = torch.randint(0, 257, (19,), device="cuda", dtype=torch.int64)
    original_target_ids = target_ids.clone()
    scale = torch.tensor(0.375, device="cuda", dtype=torch.float32)

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_weight = weight_data.detach().clone().requires_grad_()
    reference_bias = bias_data.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, reference_weight, reference_bias).float(),
        target_ids,
        reduction="sum",
    )
    (reference_loss * scale).backward()

    stream = torch.cuda.Stream() if use_non_default_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hidden_states = hidden_data.detach().clone().requires_grad_()
        weight = weight_data.detach().clone().requires_grad_()
        bias = bias_data.detach().clone().requires_grad_()
        loss = FusedLinearCrossEntropyLoss(chunk_size=7).forward(
            hidden_states,
            weight,
            target_ids,
            bias=bias,
        )
        (loss * scale).backward()
    stream.synchronize()

    _assert_gradient_close(hidden_states.grad, reference_hidden.grad)
    _assert_gradient_close(weight.grad, reference_weight.grad)
    _assert_gradient_close(bias.grad, reference_bias.grad)
    torch.testing.assert_close(hidden_states, hidden_data, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_data, rtol=0, atol=0)
    torch.testing.assert_close(bias, bias_data, rtol=0, atol=0)
    torch.testing.assert_close(target_ids, original_target_ids, rtol=0, atol=0)


@pytest.mark.parametrize("use_non_default_stream", (False, True))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_fused_linear_cross_entropy_matches_pytorch_on_cuda(
    use_non_default_stream: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260719)
    operation = FusedLinearCrossEntropyLoss()
    hidden_data = torch.randn((19, 64), device="cuda", dtype=dtype)
    weight_data = torch.randn((257, 64), device="cuda", dtype=dtype)
    bias_data = torch.randn((257,), device="cuda", dtype=dtype)
    target_ids = torch.randint(0, 257, (19,), device="cuda", dtype=torch.int64)

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_weight = weight_data.detach().clone().requires_grad_()
    reference_bias = bias_data.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, reference_weight, reference_bias).float(),
        target_ids,
        reduction="sum",
    )
    reference_loss.backward()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream() if use_non_default_stream else torch.cuda.default_stream()
    with torch.cuda.stream(stream):
        hidden_states = hidden_data.detach().clone().requires_grad_()
        weight = weight_data.detach().clone().requires_grad_()
        bias = bias_data.detach().clone().requires_grad_()
        loss = operation.forward(hidden_states, weight, target_ids, bias=bias)
        loss.backward()
    stream.synchronize()

    assert loss.dtype == torch.float32
    assert loss.shape == ()
    torch.testing.assert_close(loss, reference_loss, rtol=0.003, atol=0.003)
    _assert_gradient_close(hidden_states.grad, reference_hidden.grad)
    _assert_gradient_close(weight.grad, reference_weight.grad)
    _assert_gradient_close(bias.grad, reference_bias.grad)
    torch.testing.assert_close(hidden_states, hidden_data, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_data, rtol=0, atol=0)
    torch.testing.assert_close(bias, bias_data, rtol=0, atol=0)

    with torch.cuda.stream(stream):
        head_only_weight = weight_data.detach().clone().requires_grad_()
        head_only_loss = operation.forward(hidden_data, head_only_weight, target_ids)
        head_only_loss.backward()
    stream.synchronize()
    reference_head_only_weight = weight_data.detach().clone().requires_grad_()
    F.cross_entropy(
        F.linear(hidden_data, reference_head_only_weight).float(),
        target_ids,
        reduction="sum",
    ).backward()

    _assert_gradient_close(head_only_weight.grad, reference_head_only_weight.grad)


def test_default_chunk_policy_supports_qwen3_dimensions() -> None:
    torch.manual_seed(20260719)
    num_tokens = 257
    hidden_size = 1024
    vocab_size = 151_936
    hidden_data = torch.randn(
        (num_tokens, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (vocab_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    target_ids = torch.randint(
        0,
        vocab_size,
        (num_tokens,),
        device="cuda",
        dtype=torch.int64,
    )

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, weight).float(),
        target_ids,
        reduction="sum",
    )
    reference_loss.backward()

    hidden_states = hidden_data.detach().clone().requires_grad_()
    loss = FusedLinearCrossEntropyLoss().forward(hidden_states, weight, target_ids)
    loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(loss, reference_loss, rtol=0.003, atol=0.003)
    _assert_gradient_close(hidden_states.grad, reference_hidden.grad)


@pytest.mark.parametrize("scale", (0.0, 1e-7), ids=("zero", "near-zero"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_cross_entropy_handles_zero_and_near_zero_upstream(
    scale: float,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260909)
    hidden_data = torch.randn((7, 32), device="cuda", dtype=dtype)
    weight = torch.randn((33, 32), device="cuda", dtype=dtype)
    target_ids = torch.randint(0, 33, (7,), device="cuda", dtype=torch.int64)
    reference_hidden = hidden_data.clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, weight).float(), target_ids, reduction="sum"
    )
    (reference_loss * scale).backward()

    hidden_states = hidden_data.clone().requires_grad_()
    loss = FusedLinearCrossEntropyLoss(chunk_size=3).forward(hidden_states, weight, target_ids)
    (loss * scale).backward()

    _assert_gradient_close(hidden_states.grad, reference_hidden.grad)
    assert hidden_states.grad is not None and reference_hidden.grad is not None
    if scale == 0.0:
        torch.testing.assert_close(
            hidden_states.grad, torch.zeros_like(hidden_states), rtol=0, atol=0
        )
    else:
        # A tiny upstream must still propagate, rather than pass by rounding
        # the entire gradient to zero under an ordinary absolute tolerance.
        difference = hidden_states.grad.float() - reference_hidden.grad.float()
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference_hidden.grad.float()
        )
        assert float(relative_l2) <= (1e-5 if dtype == torch.float32 else 2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ignore_index_rows_contribute_nothing(dtype: torch.dtype) -> None:
    torch.manual_seed(20261016)
    hidden_data = torch.randn((23, 64), device="cuda", dtype=dtype)
    weight_data = torch.randn((257, 64), device="cuda", dtype=dtype)
    target_ids = torch.randint(0, 257, (23,), device="cuda", dtype=torch.int64)
    target_ids[torch.tensor([0, 3, 4, 5, 11, 22], device="cuda")] = -100
    # Leave garbage in the caching allocator so an uninitialized buffer shows up.
    torch.full((4096,), float("nan"), device="cuda").sum()

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    reference_weight = weight_data.detach().clone().requires_grad_()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, reference_weight).float(),
        target_ids,
        reduction="sum",
    )
    reference_loss.backward()

    hidden_states = hidden_data.detach().clone().requires_grad_()
    weight = weight_data.detach().clone().requires_grad_()
    loss = FusedLinearCrossEntropyLoss(chunk_size=5).forward(hidden_states, weight, target_ids)
    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(loss)
    tolerance = 1e-4 if dtype == torch.float32 else 0.003
    torch.testing.assert_close(loss, reference_loss, rtol=tolerance, atol=tolerance)
    assert torch.all(hidden_states.grad[target_ids == -100] == 0)
    expected_weight_grad = reference_weight.grad.to(weight.grad.dtype)
    if dtype == torch.float32:
        # Chunked and unchunked FP32 matmuls round differently in the last bits.
        torch.testing.assert_close(hidden_states.grad, reference_hidden.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=1e-4, atol=1e-5)
    else:
        _assert_gradient_close(hidden_states.grad, reference_hidden.grad)
        _assert_gradient_close(weight.grad, expected_weight_grad)

    all_ignored = torch.full_like(target_ids, -100)
    zero = FusedLinearCrossEntropyLoss(chunk_size=5).forward(hidden_data, weight_data, all_ignored)
    assert float(zero) == 0.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_default_constructor_matches_explicit_sum(dtype):
    torch.manual_seed(152)
    x_data = (torch.randn(65, 256, device="cuda") / 4).to(dtype)
    w_data = (torch.randn(257, 256, device="cuda") / 4).to(dtype)
    b_data = (torch.randn(257, device="cuda") / 4).to(dtype)
    y = torch.arange(65, device="cuda", dtype=torch.int64) * 3
    y[::5] = -100
    results = []
    for op in (
        FusedLinearCrossEntropyLoss(),
        FusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum"),
    ):
        x = x_data.clone().requires_grad_()
        w = w_data.clone().requires_grad_()
        b = b_data.clone().requires_grad_()
        loss = op.forward(x, w, y, bias=b)
        results.append((loss, *torch.autograd.grad(loss, (x, w, b))))
    for got, expected in zip(*results, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
