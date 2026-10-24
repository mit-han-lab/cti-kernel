# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from ctikernels.cross_entropy import FusedLinearCrossEntropyLoss


class _FakeCrossEntropyKernel:
    def __init__(self) -> None:
        self.launches: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def __getitem__(self, grid: tuple[int, ...]) -> object:
        def launch(**kwargs: object) -> None:
            self.launches.append((grid, kwargs))
            logits = kwargs["X_ptr"]
            target_ids = kwargs["Y_ptr"]
            loss = kwargs["loss_ptr"]
            assert isinstance(logits, torch.Tensor)
            assert isinstance(target_ids, torch.Tensor)
            assert isinstance(loss, torch.Tensor)
            # Like the provider, ignored rows store no loss and get zero gradients.
            kept = target_ids != kwargs["ignore_index"]
            loss[kept] = F.cross_entropy(logits[kept].float(), target_ids[kept], reduction="none")
            if kwargs["HAS_GRADIENTS"]:
                gradients = torch.softmax(logits.float(), dim=-1)
                rows = torch.arange(logits.shape[0])[kept]
                gradients[rows, target_ids[kept]] -= 1
                gradients[~kept] = 0
                logits.copy_(gradients)

        return launch


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    kernel: object,
) -> ModuleType:
    package = ModuleType("liger_kernel")
    package.__path__ = []
    ops = ModuleType("liger_kernel.ops")
    ops.__path__ = []
    cross_entropy = ModuleType("liger_kernel.ops.cross_entropy")
    cross_entropy.liger_cross_entropy_kernel = kernel
    monkeypatch.setitem(sys.modules, "liger_kernel", package)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops", ops)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.cross_entropy", cross_entropy)
    return cross_entropy


def test_constructor_fails_actionably_without_liger_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "liger_kernel", None)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops", None)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.cross_entropy", None)

    with pytest.raises(RuntimeError, match=r"liger-kernel==0\.8\.0"):
        FusedLinearCrossEntropyLoss()


@pytest.mark.parametrize(
    ("chunk_size", "error_type"),
    ((True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)),
)
def test_constructor_rejects_invalid_chunk_size(
    chunk_size: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="chunk_size"):
        FusedLinearCrossEntropyLoss(chunk_size=chunk_size)


def test_forward_uses_bound_provider_with_fixed_loss_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeCrossEntropyKernel()
    provider_module = _install_fake_provider(monkeypatch, kernel)
    operation = FusedLinearCrossEntropyLoss(chunk_size=2)
    assert operation.chunk_size == 2
    monkeypatch.setattr(operation, "_validate_inputs", lambda *args, **kwargs: None)
    provider_module.liger_cross_entropy_kernel = object()

    hidden_states = torch.randn(5, 3)
    weight = torch.randn(7, 3)
    target_ids = torch.tensor([1, 4, 6, 2, 0], dtype=torch.int64)
    bias = torch.randn(7)
    loss = operation.forward(hidden_states, weight, target_ids, bias=bias)

    expected_loss = F.cross_entropy(
        F.linear(hidden_states, weight, bias),
        target_ids,
        reduction="sum",
    )
    torch.testing.assert_close(loss, expected_loss)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert [grid for grid, _ in kernel.launches] == [(2,), (2,), (1,)]
    for chunk_index, (grid, kwargs) in enumerate(kernel.launches):
        start = chunk_index * operation.chunk_size
        end = start + grid[0]
        torch.testing.assert_close(
            kwargs["X_ptr"], F.linear(hidden_states[start:end], weight, bias)
        )
        torch.testing.assert_close(kwargs["Y_ptr"], target_ids[start:end])
        assert kwargs["loss_ptr"].shape == (grid[0],)
        assert kwargs["loss_ptr"].dtype == torch.float32
        for name, expected in {
            "X_stride": weight.shape[0],
            "Y_stride": 1,
            "loss_stride": 1,
            "token_accuracy_stride": 0,
            "predicted_tokens_stride": 0,
            "n_cols": weight.shape[0],
            "n_non_ignore": hidden_states.shape[0],
            "sum_non_ignore_weight": hidden_states.shape[0],
            "weight_sum": 0.0,
            "ignore_index": -100,
            "lse_square_scale": 0.0,
            "label_smoothing": 0.0,
            "reduction": "sum",
            "BLOCK_SIZE": 8,
            "num_warps": 32,
        }.items():
            assert kwargs[name] == expected, name
        for name in (
            "weight_ptr",
            "z_loss_ptr",
            "token_accuracy_ptr",
            "predicted_tokens_ptr",
            "softcap",
        ):
            assert kwargs[name] is None, name
        for name in (
            "RETURN_Z_LOSS",
            "RETURN_TOKEN_ACCURACY",
            "RETURN_PREDICTED_TOKENS",
            "HAS_WEIGHT",
            "HAS_SOFTCAPPING",
            "HAS_GRADIENTS",
        ):
            assert kwargs[name] is False, name


def test_forward_consumes_provider_in_place_logit_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeCrossEntropyKernel()
    _install_fake_provider(monkeypatch, kernel)
    operation = FusedLinearCrossEntropyLoss(chunk_size=2)
    monkeypatch.setattr(operation, "_validate_inputs", lambda *args, **kwargs: None)
    hidden_states = torch.randn(5, 3, requires_grad=True)
    reference_hidden = hidden_states.detach().clone().requires_grad_()
    weight = torch.randn(7, 3)
    target_ids = torch.tensor([1, 4, 6, 2, 0], dtype=torch.int64)

    loss = operation.forward(hidden_states, weight, target_ids)
    loss.backward()
    reference_loss = F.cross_entropy(
        F.linear(reference_hidden, weight), target_ids, reduction="sum"
    )
    reference_loss.backward()

    assert all(kwargs["HAS_GRADIENTS"] is True for _, kwargs in kernel.launches)
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(hidden_states.grad, reference_hidden.grad)


@pytest.mark.parametrize(
    ("num_tokens", "expected_grids"),
    (
        (255, [(255,)]),
        (256, [(256,)]),
        (257, [(256,), (1,)]),
    ),
)
def test_default_chunk_policy_is_fixed_at_256_tokens(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_grids: list[tuple[int, ...]],
) -> None:
    kernel = _FakeCrossEntropyKernel()
    _install_fake_provider(monkeypatch, kernel)
    operation = FusedLinearCrossEntropyLoss()
    monkeypatch.setattr(operation, "_validate_inputs", lambda *args, **kwargs: None)

    hidden_states = torch.randn(num_tokens, 3)
    weight = torch.randn(7, 3)
    target_ids = torch.arange(num_tokens, dtype=torch.int64).remainder(7)
    operation.forward(hidden_states, weight, target_ids)

    assert [grid for grid, _ in kernel.launches] == expected_grids


@pytest.mark.parametrize(
    ("hidden_states", "weight", "target_ids", "bias", "error_type", "message"),
    [
        (
            torch.empty(3, 2, 4),
            torch.empty(7, 4),
            torch.empty(3, dtype=torch.int64),
            None,
            ValueError,
            "hidden_states must have shape",
        ),
        (
            torch.empty(3, 4),
            torch.empty(7, 5),
            torch.empty(3, dtype=torch.int64),
            None,
            ValueError,
            "same hidden size",
        ),
        (
            torch.empty(3, 4),
            torch.empty(7, 4),
            torch.empty(2, dtype=torch.int64),
            None,
            ValueError,
            "same token count",
        ),
        (
            torch.empty(3, 4),
            torch.empty(7, 4),
            torch.empty(3, dtype=torch.int32),
            None,
            TypeError,
            "target_ids must use int64",
        ),
        (
            torch.empty(3, 4, dtype=torch.float64),
            torch.empty(7, 4, dtype=torch.float64),
            torch.empty(3, dtype=torch.int64),
            None,
            TypeError,
            "must use bfloat16, float16 or float32",
        ),
        (
            torch.empty(3, 4),
            torch.empty(7, 4),
            torch.empty(3, dtype=torch.int64),
            torch.empty(8),
            ValueError,
            "bias must have shape",
        ),
    ],
)
def test_forward_rejects_invalid_metadata_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None,
    error_type: type[Exception],
    message: str,
) -> None:
    _install_fake_provider(monkeypatch, _FakeCrossEntropyKernel())
    operation = FusedLinearCrossEntropyLoss()

    with pytest.raises(error_type, match=message):
        operation.forward(hidden_states, weight, target_ids, bias=bias)


def test_forward_rejects_host_inputs_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch, _FakeCrossEntropyKernel())
    operation = FusedLinearCrossEntropyLoss()

    with pytest.raises(ValueError, match="must reside on CUDA"):
        operation.forward(
            torch.empty(3, 4),
            torch.empty(7, 4),
            torch.empty(3, dtype=torch.int64),
        )


def test_forward_ignores_minus_100_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = _FakeCrossEntropyKernel()
    _install_fake_provider(monkeypatch, kernel)
    operation = FusedLinearCrossEntropyLoss(chunk_size=3)
    monkeypatch.setattr(operation, "_validate_inputs", lambda *args, **kwargs: None)
    # Poison fresh allocations: an uninitialized loss buffer would leak NaN.
    real_empty = torch.empty

    def poisoned_empty(*args: object, **kwargs: object) -> torch.Tensor:
        return real_empty(*args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch, "empty", poisoned_empty)

    torch.manual_seed(0)
    hidden_data = torch.randn(7, 3)
    weight = torch.randn(5, 3)
    target_ids = torch.tensor([1, -100, 4, -100, -100, 0, 2], dtype=torch.int64)
    hidden_states = hidden_data.clone().requires_grad_()
    loss = operation.forward(hidden_states, weight, target_ids)
    loss.backward()

    reference_hidden = hidden_data.clone().requires_grad_()
    expected = F.cross_entropy(F.linear(reference_hidden, weight), target_ids, reduction="sum")
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(hidden_states.grad, reference_hidden.grad)
    assert torch.all(hidden_states.grad[target_ids == -100] == 0)


@pytest.mark.parametrize("ignore_index", [True, 1.0, None, "-7"])
def test_ignore_index_must_be_python_int(ignore_index):
    with pytest.raises(TypeError, match="ignore_index"):
        FusedLinearCrossEntropyLoss(provider="torch", ignore_index=ignore_index)


@pytest.mark.parametrize("reduction", [None, True, "SUM", "batchmean", []])
def test_invalid_reduction(reduction):
    with pytest.raises(ValueError, match="reduction"):
        FusedLinearCrossEntropyLoss(provider="torch", reduction=reduction)


def test_ce_options_are_read_only():
    op = FusedLinearCrossEntropyLoss(provider="torch", ignore_index=-7, reduction="mean")
    assert op.ignore_index == -7 and op.reduction == "mean"
    with pytest.raises(AttributeError):
        op.ignore_index = -100
    with pytest.raises(AttributeError):
        op.reduction = "sum"


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,all_ignored", [(0, False), (1, False), (7, False), (7, True)])
def test_reductions_and_empty_match_torch(monkeypatch, reduction, dtype, n, all_ignored):
    kernel = _FakeCrossEntropyKernel()
    _install_fake_provider(monkeypatch, kernel)
    op = FusedLinearCrossEntropyLoss(chunk_size=3, reduction=reduction, ignore_index=-7)
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    torch.manual_seed(81)
    x = (torch.randn(n, 10) / 4).to(dtype)[:, :5].requires_grad_()
    w = (torch.randn(11, 5) / 4).to(dtype).requires_grad_()
    b = (torch.randn(11) / 4).to(dtype).requires_grad_()
    targets = torch.arange(n * 2, dtype=torch.int64)
    y = targets[::2]
    y.remainder_(11)
    if n > 1:
        y[1::3] = -7
    if all_ignored:
        y.fill_(-7)
    loss = op.forward(x, w, y, bias=b)
    # The provider must receive strided targets directly.
    assert all(call["Y_stride"] == y.stride(0) for _, call in kernel.launches)
    if reduction == "none":
        assert all(not call["HAS_GRADIENTS"] for _, call in kernel.launches)
    reference = F.cross_entropy(F.linear(x, w, b).float(), y, ignore_index=-7, reduction=reduction)
    torch.testing.assert_close(loss, reference, equal_nan=True)
    upstream = torch.randn(n * 2)[::2] if reduction == "none" else torch.tensor(0.5)
    got = torch.autograd.grad(loss, (x, w, b), upstream)
    expected = torch.autograd.grad(reference, (x, w, b), upstream)
    for actual, want in zip(got, expected, strict=True):
        tolerance = 1e-5 if dtype == torch.float32 else 0.02
        torch.testing.assert_close(actual, want, rtol=tolerance, atol=tolerance * 0.1)
    if n == 0 or all_ignored:
        # PyTorch mean is NaN while every input gradient is zero.
        assert all(torch.count_nonzero(g) == 0 for g in (*got, *expected))
        if reduction == "mean":
            assert loss.isnan() and reference.isnan()
    if n == 0:
        assert kernel.launches == []


def test_none_applies_upstream_before_rounding(monkeypatch):
    kernel = _FakeCrossEntropyKernel()
    _install_fake_provider(monkeypatch, kernel)
    op = FusedLinearCrossEntropyLoss(reduction="none", chunk_size=2)
    monkeypatch.setattr(op, "_validate_inputs", lambda *args, **kwargs: None)
    torch.manual_seed(31)
    x = torch.randn(3, 5, dtype=torch.bfloat16).requires_grad_()
    w = torch.randn(7, 5, dtype=torch.bfloat16)
    y = torch.tensor([1, 2, 3])
    upstream = torch.tensor([0.317, -0.713, 1.113])
    loss = op.forward(x, w, y)
    (dx,) = torch.autograd.grad(loss, x, upstream)
    expected = torch.empty_like(x)
    for begin in range(0, 3, 2):
        end = min(begin + 2, 3)
        dz = (x[begin:end] @ w.t()).float().softmax(1)
        dz.scatter_add_(1, y[begin:end, None], -torch.ones(end - begin, 1))
        dz *= upstream[begin:end, None]
        expected[begin:end] = dz.to(w.dtype) @ w
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
