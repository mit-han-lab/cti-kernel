# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch

from ctikernels.linear.lora import lora_projection, try_prepare_lora_inference_plan


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _InferencePlanSpy:
    def __init__(self, *, can_project: bool, result: torch.Tensor) -> None:
        self._can_project = can_project
        self.result = result
        self.can_project_calls = 0
        self.project_calls = 0

    def can_project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        factors: tuple[torch.Tensor, ...],
        *,
        scaling: float,
    ) -> bool:
        del x, weight, bias, factors, scaling
        self.can_project_calls += 1
        return self._can_project

    def project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        del x, weight, bias
        self.project_calls += 1
        return self.result


def test_cpu_materialization_does_not_prepare_cuda_inference_metadata() -> None:
    input_size = 1024
    output_sizes = (2048, 1024, 1024)
    rank = 64
    factors = tuple(
        tensor
        for output_size in output_sizes
        for tensor in (
            torch.empty(rank, input_size, dtype=torch.bfloat16),
            torch.empty(output_size, rank, dtype=torch.bfloat16),
        )
    )

    assert (
        try_prepare_lora_inference_plan(
            factors,
            input_size=input_size,
            output_sizes=output_sizes,
            scaling=2.0,
        )
        is None
    )


def test_training_projection_ignores_inference_plan() -> None:
    x = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(3, 4)
    factors = (torch.randn(2, 4, requires_grad=True), torch.randn(3, 2, requires_grad=True))
    plan = _InferencePlanSpy(can_project=True, result=torch.empty(2, 3))

    output = lora_projection(
        x,
        weight,
        None,
        factors,
        scaling=0.5,
        inference_plan=plan,
    )
    output.sum().backward()

    assert plan.can_project_calls == 0
    assert plan.project_calls == 0
    assert x.grad is not None
    assert all(factor.grad is not None for factor in factors)


@torch.inference_mode()
def test_inference_projection_uses_plan_only_when_it_can_project() -> None:
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    factors = (torch.randn(2, 4), torch.randn(3, 2))
    expected_fallback = lora_projection(x, weight, None, factors, scaling=0.5)

    rejected = _InferencePlanSpy(can_project=False, result=torch.empty_like(expected_fallback))
    actual_fallback = lora_projection(
        x,
        weight,
        None,
        factors,
        scaling=0.5,
        inference_plan=rejected,
    )
    _assert_exact(actual_fallback, expected_fallback)
    assert rejected.can_project_calls == 1
    assert rejected.project_calls == 0

    expected_planned = torch.randn_like(expected_fallback)
    accepted = _InferencePlanSpy(can_project=True, result=expected_planned)
    actual_planned = lora_projection(
        x,
        weight,
        None,
        factors,
        scaling=0.5,
        inference_plan=accepted,
    )
    assert actual_planned is expected_planned
    assert accepted.can_project_calls == 1
    assert accepted.project_calls == 1
