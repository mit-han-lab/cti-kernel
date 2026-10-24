# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear.lora import (
    LoRAInferencePlan,
    lora_projection,
    try_prepare_lora_inference_plan,
)

_FP32_ATOL = 2e-6
_FP32_RTOL = 2e-6
_FP32_COLLECTION_RELATIVE_L2 = 2e-5
_BF16_ATOL = 1e-4
_BF16_RTOL = 4e-3
_BF16_COLLECTION_RELATIVE_L2 = 4e-3
_MEASURED_LORA_TOPOLOGIES = (
    (1024, (2048, 1024, 1024)),
    (2048, (1024,)),
    (1024, (3072, 3072)),
    (3072, (1024,)),
    (2560, (4096, 1024, 1024)),
    (1536, (2304, 768)),
)


def _tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    requires_grad: bool,
) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(requires_grad)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_weights: tuple[torch.Tensor, ...],
    scaling: float,
) -> torch.Tensor:
    deltas = []
    for index in range(0, len(lora_weights), 2):
        lora_a = lora_weights[index]
        lora_b = lora_weights[index + 1]
        deltas.append(F.linear(F.linear(x, lora_a), lora_b) * scaling)
    return F.linear(x, weight, bias) + torch.cat(deltas, dim=-1)


def _oracle_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().float().requires_grad_(tensor.requires_grad)


def _small_batch_lora_case(
    *,
    batch_size: int,
    input_size: int,
    output_sizes: tuple[int, ...],
    seed: int,
    rank: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    LoRAInferencePlan,
]:
    torch.manual_seed(seed)
    x = torch.randn(batch_size, input_size, device="cuda", dtype=dtype) * 0.25
    weight = (
        torch.randn(
            sum(output_sizes),
            input_size,
            device="cuda",
            dtype=dtype,
        )
        * 0.02
    )
    factors = tuple(
        tensor
        for output_size in output_sizes
        for tensor in (
            torch.randn(rank, input_size, device="cuda", dtype=dtype) * 0.02,
            torch.randn(output_size, rank, device="cuda", dtype=dtype) * 0.02,
        )
    )
    plan = try_prepare_lora_inference_plan(
        factors,
        input_size=input_size,
        output_sizes=output_sizes,
        scaling=2.0,
        deterministic=deterministic,
    )
    assert plan is not None
    assert plan.can_project(x, weight, None, factors, scaling=2.0)
    return x, weight, factors, plan


def test_graph_workspace_largest_first_ladder_reserves_base_split_k(monkeypatch) -> None:
    """The engine's ladder through lora_projection in one GraphWorkspace, no caller
    workspace: the largest bucket's base GEMM runs cuBLAS (no split), smaller buckets split
    after the delta's capture froze the workspace. Replays equal eager bitwise."""
    from ctikernels import GraphWorkspace
    from ctikernels.linear import inference
    from ctikernels.linear._dispatch import Config

    split = Config(32, 128, 2, 3, 1, "reg", 4)
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE", "0")
    monkeypatch.setattr(
        inference, "tuned_config", lambda kind, n, k, edge, **_: split if edge <= 16 else None
    )
    rows = (32, 16, 4)
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=max(rows), input_size=1024, output_sizes=(512, 256), seed=7
    )
    graphs, outputs, eager = [], [], []
    with torch.no_grad(), GraphWorkspace(max(rows)) as group:
        for m in rows:
            call = lambda m=m: lora_projection(
                x[:m], weight, None, factors, scaling=2.0, inference_plan=plan
            )
            eager.append(call().clone())
            assert (plan.base_provider == "cublas") == (m > 16)
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1]):
                outputs.append(call())
    for graph, out, expected in zip(graphs, outputs, eager, strict=True):
        graph.replay()
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    del graphs
    group.close()


def _assert_collection_matches_oracle(
    named_pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        atol = _FP32_ATOL
        rtol = _FP32_RTOL
        collection_bound = _FP32_COLLECTION_RELATIVE_L2
    else:
        atol = _BF16_ATOL
        rtol = _BF16_RTOL
        collection_bound = _BF16_COLLECTION_RELATIVE_L2

    collection_error_sum = 0.0
    collection_reference_sum = 0.0
    for name, actual, expected in named_pairs:
        difference = actual.detach().double() - expected.detach().double()
        reference = expected.detach().double()
        error_sum = float(difference.square().sum())
        reference_sum = float(reference.square().sum())
        error_rms = math.sqrt(error_sum / difference.numel())
        reference_rms = math.sqrt(reference_sum / reference.numel())
        allowed_rms = atol + rtol * reference_rms
        max_absolute = float(difference.abs().max())
        assert error_rms <= allowed_rms, (
            f"{name} RMS error {error_rms} exceeds {allowed_rms}; "
            f"reference_rms={reference_rms}, max_absolute={max_absolute}"
        )
        collection_error_sum += error_sum
        collection_reference_sum += reference_sum

    assert collection_reference_sum > 0
    collection_relative_l2 = math.sqrt(collection_error_sum / collection_reference_sum)
    assert collection_relative_l2 <= collection_bound, (
        f"collection relative L2 error {collection_relative_l2} exceeds {collection_bound}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_sizes", [(11,), (7, 5, 9)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_lora_projection_matches_independent_fp32_oracle(
    dtype: torch.dtype,
    output_sizes: tuple[int, ...],
    has_bias: bool,
) -> None:
    torch.manual_seed(20260721)
    input_size = 13
    rank = 3
    scaling = 0.37
    source = _tensor((2, input_size, 5), dtype=dtype, requires_grad=True)
    x = source.transpose(1, 2)
    x.retain_grad()
    weight = _tensor((sum(output_sizes), input_size), dtype=dtype, requires_grad=True)
    bias = _tensor((sum(output_sizes),), dtype=dtype, requires_grad=True) if has_bias else None
    lora_weights = tuple(
        tensor
        for output_size in output_sizes
        for tensor in (
            _tensor((rank, input_size), dtype=dtype, requires_grad=True),
            _tensor((output_size, rank), dtype=dtype, requires_grad=True),
        )
    )
    inputs = (x, weight, *(() if bias is None else (bias,)), *lora_weights)
    originals = tuple(tensor.detach().clone() for tensor in inputs)

    reference_x = _oracle_copy(x)
    reference_weight = _oracle_copy(weight)
    reference_bias = None if bias is None else _oracle_copy(bias)
    reference_lora_weights = tuple(_oracle_copy(tensor) for tensor in lora_weights)
    grad_output = torch.randn(
        (*x.shape[:-1], weight.shape[0]),
        device="cuda",
        dtype=torch.float32,
    )

    output = lora_projection(x, weight, bias, lora_weights, scaling=scaling)
    reference = _reference(
        reference_x,
        reference_weight,
        reference_bias,
        reference_lora_weights,
        scaling,
    )
    (output.float() * grad_output).sum().backward()
    (reference * grad_output).sum().backward()

    named_pairs = [("output", output, reference)]
    gradient_pairs = [
        ("x.grad", x.grad, reference_x.grad),
        ("weight.grad", weight.grad, reference_weight.grad),
    ]
    if bias is not None:
        assert reference_bias is not None
        gradient_pairs.append(("bias.grad", bias.grad, reference_bias.grad))
    gradient_pairs.extend(
        (f"lora_factors[{index}].grad", actual.grad, expected.grad)
        for index, (actual, expected) in enumerate(
            zip(lora_weights, reference_lora_weights, strict=True)
        )
    )
    for name, actual, expected in gradient_pairs:
        assert actual is not None
        assert expected is not None
        named_pairs.append((name, actual, expected))
    _assert_collection_matches_oracle(named_pairs, dtype)

    assert all(
        torch.equal(tensor, before) for tensor, before in zip(inputs, originals, strict=True)
    )
    assert output.untyped_storage().data_ptr() not in {
        tensor.untyped_storage().data_ptr() for tensor in inputs
    }


def test_lora_projection_respects_individual_gradient_selection() -> None:
    torch.manual_seed(17)
    x = _tensor((7, 9), dtype=torch.float32, requires_grad=True)
    weight = _tensor((12, 9), dtype=torch.float32, requires_grad=False)
    bias = _tensor((12,), dtype=torch.float32, requires_grad=True)
    lora_weights = (
        _tensor((4, 9), dtype=torch.float32, requires_grad=True),
        _tensor((5, 4), dtype=torch.float32, requires_grad=False),
        _tensor((4, 9), dtype=torch.float32, requires_grad=False),
        _tensor((7, 4), dtype=torch.float32, requires_grad=True),
    )

    output = lora_projection(x, weight, bias, lora_weights, scaling=0.73)
    output.square().sum().backward(retain_graph=True)

    assert x.grad is not None
    assert weight.grad is None
    assert bias.grad is not None
    assert lora_weights[0].grad is not None
    assert lora_weights[1].grad is None
    assert lora_weights[2].grad is None
    assert lora_weights[3].grad is not None

    trainable = (x, bias, lora_weights[0], lora_weights[3])
    first_gradients = tuple(tensor.grad.clone() for tensor in trainable)
    for tensor in trainable:
        tensor.grad = None
    output.square().sum().backward()
    for tensor, first_gradient in zip(trainable, first_gradients, strict=True):
        torch.testing.assert_close(tensor.grad, first_gradient, rtol=0, atol=0)


def test_lora_projection_rejects_higher_order_gradients() -> None:
    torch.manual_seed(19)
    x = _tensor((3, 5), dtype=torch.float32, requires_grad=True)
    weight = _tensor((7, 5), dtype=torch.float32, requires_grad=False)
    lora_weights = (
        _tensor((2, 5), dtype=torch.float32, requires_grad=True),
        _tensor((7, 2), dtype=torch.float32, requires_grad=True),
    )

    output = lora_projection(x, weight, None, lora_weights, scaling=0.5)
    (grad_x,) = torch.autograd.grad(output.sum(), x, create_graph=True)
    assert not grad_x.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(grad_x.sum(), x)


def test_zero_initialized_lora_is_exact_and_does_not_save_full_delta() -> None:
    torch.manual_seed(23)
    rows, input_size, rank = 17, 13, 4
    output_sizes = (7, 5)
    x = _tensor((rows, input_size), dtype=torch.bfloat16, requires_grad=True)
    weight = _tensor((sum(output_sizes), input_size), dtype=torch.bfloat16, requires_grad=False)
    lora_weights = tuple(
        tensor
        for output_size in output_sizes
        for tensor in (
            _tensor((rank, input_size), dtype=torch.bfloat16, requires_grad=True),
            torch.zeros(
                (output_size, rank),
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
        )
    )
    saved_shapes = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = lora_projection(x, weight, None, lora_weights, scaling=0.5)

    torch.testing.assert_close(output, F.linear(x, weight), rtol=0, atol=0)
    assert (rows, sum(output_sizes)) not in saved_shapes
    assert saved_shapes.count((rows, rank)) == len(output_sizes)


def test_lora_projection_rejects_incomplete_output_slice_coverage() -> None:
    x = _tensor((3, 5), dtype=torch.float32, requires_grad=True)
    weight = _tensor((7, 5), dtype=torch.float32, requires_grad=False)
    factors = (
        _tensor((2, 5), dtype=torch.float32, requires_grad=True),
        _tensor((6, 2), dtype=torch.float32, requires_grad=True),
    )

    with pytest.raises(ValueError, match="cover the base projection exactly"):
        lora_projection(x, weight, None, factors, scaling=0.5)


@torch.inference_mode()
def test_lora_projection_cuda_graph_replay_observes_weight_updates() -> None:
    torch.manual_seed(29)
    x = torch.randn((8, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((12, 16), device="cuda", dtype=torch.bfloat16)
    lora_a = torch.randn((4, 16), device="cuda", dtype=torch.bfloat16)
    lora_b = torch.zeros((12, 4), device="cuda", dtype=torch.bfloat16)
    lora_weights = (lora_a, lora_b)

    for _ in range(3):
        lora_projection(x, weight, None, lora_weights, scaling=0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = lora_projection(x, weight, None, lora_weights, scaling=0.5)

    graph.replay()
    first = captured.clone()
    torch.testing.assert_close(
        first,
        lora_projection(x, weight, None, lora_weights, scaling=0.5),
        rtol=0,
        atol=0,
    )

    lora_b.normal_()
    graph.replay()
    second = captured.clone()
    torch.testing.assert_close(
        second,
        lora_projection(x, weight, None, lora_weights, scaling=0.5),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(first, second)


@pytest.mark.parametrize(("input_size", "output_sizes"), _MEASURED_LORA_TOPOLOGIES)
@torch.inference_mode()
def test_small_batch_lora_structural_topologies_match_oracle(
    input_size: int,
    output_sizes: tuple[int, ...],
) -> None:
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=4,
        input_size=input_size,
        output_sizes=output_sizes,
        seed=20260722 + input_size + sum(output_sizes),
    )
    actual = lora_projection(
        x,
        weight,
        None,
        factors,
        scaling=2.0,
        inference_plan=plan,
    )
    expected = _reference(
        x.float(),
        weight.float(),
        None,
        tuple(factor.float() for factor in factors),
        2.0,
    )
    _assert_collection_matches_oracle([("small_batch", actual, expected)], x.dtype)
    bias = torch.zeros(sum(output_sizes), device="cuda", dtype=torch.bfloat16)
    assert plan.can_project(x, weight, bias, factors, scaling=2.0)


@pytest.mark.parametrize(
    ("batch_size", "rank"),
    [(1, 16), (2, 64), (4, 128), (8, 16), (32, 32), (64, 128)],
)
@torch.inference_mode()
def test_small_batch_lora_decode_graph_matches_oracle_and_observes_live_updates(
    batch_size: int,
    rank: int,
) -> None:
    input_size, output_sizes = _MEASURED_LORA_TOPOLOGIES[0]
    scaling = 2.0
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=batch_size,
        input_size=input_size,
        output_sizes=output_sizes,
        seed=20260722 + batch_size + input_size + sum(output_sizes),
        rank=rank,
    )
    original_x = x.clone()
    original_weight = weight.clone()
    original_factors = tuple(factor.clone() for factor in factors)

    for _ in range(3):
        lora_projection(
            x,
            weight,
            None,
            factors,
            scaling=scaling,
            inference_plan=plan,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = lora_projection(
            x,
            weight,
            None,
            factors,
            scaling=scaling,
            inference_plan=plan,
        )

    graph.replay()
    before = captured.clone()
    before_reference = _reference(
        x.float(),
        weight.float(),
        None,
        tuple(factor.float() for factor in factors),
        scaling,
    )
    _assert_collection_matches_oracle([("captured.before", before, before_reference)], x.dtype)
    assert torch.equal(x, original_x)
    assert torch.equal(weight, original_weight)
    assert all(
        torch.equal(factor, original)
        for factor, original in zip(factors, original_factors, strict=True)
    )
    input_pointers = {x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()}
    input_pointers.update(factor.untyped_storage().data_ptr() for factor in factors)
    assert captured.untyped_storage().data_ptr() not in input_pointers

    pointers = tuple(factor.data_ptr() for factor in factors)
    for factor in factors:
        factor.add_(0.01)
    assert tuple(factor.data_ptr() for factor in factors) == pointers
    graph.replay()
    after = captured.clone()
    after_reference = _reference(
        x.float(),
        weight.float(),
        None,
        tuple(factor.float() for factor in factors),
        scaling,
    )
    _assert_collection_matches_oracle([("captured.after", after, after_reference)], x.dtype)
    assert not torch.equal(before, after)


@torch.inference_mode()
def test_small_batch_lora_uses_the_current_cuda_stream() -> None:
    input_size, output_sizes = _MEASURED_LORA_TOPOLOGIES[0]
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=4,
        input_size=input_size,
        output_sizes=output_sizes,
        seed=20260731,
    )
    current_stream = torch.cuda.current_stream()
    operation_stream = torch.cuda.Stream()
    operation_stream.wait_stream(current_stream)

    with torch.cuda.stream(operation_stream):
        actual = lora_projection(
            x,
            weight,
            None,
            factors,
            scaling=2.0,
            inference_plan=plan,
        )
    current_stream.wait_stream(operation_stream)

    expected = _reference(
        x.float(),
        weight.float(),
        None,
        tuple(factor.float() for factor in factors),
        2.0,
    )
    _assert_collection_matches_oracle([("current_stream", actual, expected)], x.dtype)


@pytest.mark.parametrize(
    ("output_sizes", "rank", "accepted_token_counts", "rejected_token_counts"),
    [
        ((2304, 768), 128, (1, 64, 65, 200, 1024), (0,)),
        ((2304,), 128, (1, 2, 3, 4, 8, 64, 65, 1024), (0,)),
    ],
)
@torch.inference_mode()
def test_small_batch_lora_selection_is_structural(
    output_sizes: tuple[int, ...],
    rank: int,
    accepted_token_counts: tuple[int, ...],
    rejected_token_counts: tuple[int, ...],
) -> None:
    _require_segmented_lora()
    input_size = 1536
    _, weight, factors, plan = _small_batch_lora_case(
        batch_size=accepted_token_counts[0],
        input_size=input_size,
        output_sizes=output_sizes,
        seed=20260806 + rank + len(output_sizes),
        rank=rank,
    )
    for num_tokens in accepted_token_counts:
        x = torch.empty(num_tokens, input_size, device="cuda", dtype=torch.bfloat16)
        assert plan.can_project(x, weight, None, factors, scaling=2.0)
    for num_tokens in rejected_token_counts:
        x = torch.empty(num_tokens, input_size, device="cuda", dtype=torch.bfloat16)
        assert not plan.can_project(x, weight, None, factors, scaling=2.0)


@pytest.mark.parametrize(("output_sizes", "rank"), [((2304, 768), 513), ((2304,), 513)])
@torch.inference_mode()
def test_lora_plan_rejects_unsupported_rank_ranges(
    output_sizes: tuple[int, ...],
    rank: int,
) -> None:
    input_size = 1536
    factors = tuple(
        tensor
        for output_size in output_sizes
        for tensor in (
            torch.empty(rank, input_size, device="cuda", dtype=torch.bfloat16),
            torch.empty(output_size, rank, device="cuda", dtype=torch.bfloat16),
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
@torch.inference_mode()
def test_single_slice_rank128_eager_and_graph(dtype, has_bias, batch_size):
    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference

    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=batch_size,
        input_size=1024,
        output_sizes=(2048,),
        rank=128,
        dtype=dtype,
        seed=715 + batch_size,
    )
    bias = torch.randn(2048, device="cuda", dtype=dtype) * 0.01 if has_bias else None
    assert plan.can_project(x, weight, bias, factors, scaling=2.0)
    call = lambda: lora_projection(x, weight, bias, factors, scaling=2.0, inference_plan=plan)
    eager = call()
    base = plan.base(x, weight, bias)
    base64 = F.linear(x.double(), weight.double(), None if bias is None else bias.double())
    delta64 = delta_reference(x, factors)
    assert delta_correctness(eager, base, delta64, base64)["ok"]
    if has_bias or dtype == torch.float16:
        assert plan.base_provider == "torch"
    else:
        assert plan.base_provider in ("cute", "cublas")
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert delta_correctness(captured, base, delta64, base64)["ok"]


@pytest.mark.parametrize("provider", ["cute", "cublas"])
@torch.inference_mode()
def test_small_batch_linear_base_oracle_and_graph(provider):
    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference
    from ctikernels.linear.inference import Linear

    if provider == "cute" and torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("the CuTe small-M base Linear serves sm_120/sm_121")
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=3,
        input_size=1024,
        output_sizes=(2048,),
        rank=128,
        seed=713,
    )
    # Exercise both implementations independently of this GPU's measured table.
    plan._linear = Linear(provider=provider)
    call = lambda: lora_projection(x, weight, None, factors, scaling=2.0, inference_plan=plan)
    eager = call()  # Compiles the M=3..4 bucket before capture.
    base = plan.base(x, weight)
    assert plan.base_provider == provider
    assert plan._linear.decision(2048, 1024, 3)["provider"] == provider
    base64 = F.linear(x.double(), weight.double())
    delta64 = delta_reference(x, factors)
    assert delta_correctness(eager, base, delta64, base64)["ok"]
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert delta_correctness(captured, base, delta64, base64)["ok"]
    assert plan.base_provider == provider
    # Zero delta isolates the exact base result, including graph replay.
    for b in factors[1::2]:
        b.zero_()
    graph.replay()
    torch.testing.assert_close(captured, base, atol=0, rtol=0)


def _require_segmented_lora():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("segmented LoRA requires compute capability 8.9 or newer")


# Cover every requested axis in twelve small cases instead of a Cartesian sweep.
# Uneven, unaligned slice widths exercise tails; each dtype gets bias and no bias.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rank,tokens,output_sizes,has_bias",
    [
        (8, 1, (33,), False),
        (16, 3, (31, 17), True),
        (64, 64, (33, 17, 9), False),
        (128, 65, (16, 7, 9, 13, 3, 5), True),
        (256, 200, (31, 17), False),
        (512, 1024, (33, 17, 9), True),
    ],
)
@torch.inference_mode()
def test_segmented_lora_projection_ranges_and_live_graph_updates(
    dtype, rank, tokens, output_sizes, has_bias
):
    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference

    _require_segmented_lora()
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=tokens,
        input_size=64,
        output_sizes=output_sizes,
        rank=rank,
        dtype=dtype,
        seed=901 + rank,
    )
    bias = torch.randn(sum(output_sizes), device="cuda", dtype=dtype) * 0.01 if has_bias else None
    assert plan.can_project(x, weight, bias, factors, scaling=2.0)
    call = lambda: lora_projection(x, weight, bias, factors, scaling=2.0, inference_plan=plan)
    base = plan.base(x, weight, bias)
    base64 = F.linear(x.double(), weight.double(), None if bias is None else bias.double())

    def check(output):
        assert delta_correctness(output, base, delta_reference(x, factors), base64)["ok"]
        assert plan.provider == "triton-lora-v1"

    eager = call()
    check(eager)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # Warm this M and establish metadata readiness on the capture stream.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    graph.replay()
    before = captured.clone()
    check(before)
    torch.testing.assert_close(before, eager, atol=1e-4, rtol=4e-3)
    pointers = tuple(f.data_ptr() for f in factors)
    for factor in factors:
        factor.add_(0.01)
    assert tuple(f.data_ptr() for f in factors) == pointers
    graph.replay()
    after = captured.clone()
    check(after)
    check(call())
    assert not torch.equal(before, after)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_lora_projection_frozen_triton_table_row(monkeypatch, dtype):
    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference
    from ctikernels.linear import _lora_policy, _lora_segmented

    hardware = (torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    if hardware not in _lora_policy._TRITON_HARDWARE:
        pytest.skip("frozen Triton provider only admits the three original devices")

    def unexpected_segmented(*args, **kwargs):
        pytest.fail("the segmented delta must not run on a frozen-Triton table row")

    monkeypatch.setattr(_lora_segmented, "lora_delta_factors_", unexpected_segmented)
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=64,
        input_size=64,
        output_sizes=(33, 17, 9),
        rank=128,
        dtype=dtype,
        seed=1003,
        deterministic=False,
    )
    row = (plan._policy.device_class, 3, 128, 64, 64)
    monkeypatch.setitem(_lora_policy.PROVIDER_TABLE, row, "triton")
    bias = torch.zeros(59, device="cuda", dtype=dtype)
    call = lambda: lora_projection(x, weight, bias, factors, scaling=2.0, inference_plan=plan)
    eager = call()
    assert plan.provider == "triton-small-batch-v1"
    base = plan.base(x, weight, bias)
    base64 = F.linear(x.double(), weight.double(), bias.double())
    assert delta_correctness(eager, base, delta_reference(x, factors), base64)["ok"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for factor in factors:
        factor.add_(0.01)
    graph.replay()
    assert delta_correctness(captured, base, delta_reference(x, factors), base64)["ok"]
    assert plan._select_provider(65) == "segmented"  # past the table's token edges


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_replaced_factor_storage_rebinds_eagerly_and_raises_in_capture(dtype):
    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference

    _require_segmented_lora()
    x, weight, factors, _ = _small_batch_lora_case(
        batch_size=8, input_size=64, output_sizes=(33, 17), rank=16, dtype=dtype, seed=1201
    )
    params = tuple(torch.nn.Parameter(f.clone()) for f in factors)
    plan = try_prepare_lora_inference_plan(
        params, input_size=64, output_sizes=(33, 17), scaling=2.0
    )
    base = plan.base(x, weight)
    base64 = F.linear(x.double(), weight.double())

    def call():
        with torch.no_grad():
            return lora_projection(x, weight, None, params, scaling=2.0, inference_plan=plan)

    def check(output):
        assert plan.provider == "triton-lora-v1"
        ref = delta_reference(x, tuple(p.detach() for p in params))
        assert delta_correctness(output, base, ref, base64)["ok"]

    check(call())
    # In-place updates keep the binding, including inside a captured graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    with torch.no_grad():
        for p in params:
            p.mul_(-1.5)
    graph.replay()
    check(captured)
    # Replacing storage rebinds on the next eager call: the result uses new values.
    old = params[0].data_ptr()
    params[0].data = torch.randn_like(params[0]) * 0.05
    params[3].data = torch.randn_like(params[3]) * 0.05
    assert params[0].data_ptr() != old
    check(call())
    # A capture after an unobserved replacement raises instead of reading stale storage.
    params[1].data = torch.randn_like(params[1]) * 0.05
    with pytest.raises(RuntimeError, match="storage was replaced"):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            call()
    check(call())
    params[1].data = torch.zeros(params[1].shape[0] + 1, params[1].shape[1], dtype=dtype)
    with pytest.raises(ValueError, match="prepare a new inference plan"):
        call()


@torch.inference_mode()
def test_deterministic_default_is_bitwise_repeatable_on_measured_triton_rows(monkeypatch):
    from ctikernels.linear import _lora_policy

    _require_segmented_lora()
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=64, input_size=6144, output_sizes=(512,), rank=64, seed=1301
    )
    # Pretend this row was measured faster with Triton on this device class.
    device_class = plan._policy.device_class
    monkeypatch.setitem(_lora_policy.PROVIDER_TABLE, (device_class, 1, 64, 64, 6144), "triton")
    assert plan.deterministic
    call = lambda p: lora_projection(x, weight, None, factors, scaling=2.0, inference_plan=p)
    first = call(plan)
    assert plan.provider == "triton-lora-v1"
    for _ in range(5):
        assert torch.equal(call(plan), first)
    hardware = (torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    if hardware in _lora_policy._TRITON_HARDWARE:
        fast = try_prepare_lora_inference_plan(
            factors, input_size=6144, output_sizes=(512,), scaling=2.0, deterministic=False
        )
        assert not fast.deterministic
        call(fast)
        assert fast.provider == "triton-small-batch-v1"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_warm_up_returns_routes_and_capture_never_compiles(dtype):
    from ctikernels.linear import _lora_segmented

    _require_segmented_lora()
    max_m = 72
    x, weight, factors, plan = _small_batch_lora_case(
        batch_size=max_m, input_size=256, output_sizes=(96, 32), rank=64, dtype=dtype, seed=1401
    )
    routes = plan.warm_up(max_m, weight)
    delta = [r for r in routes if r["op"] == "delta"]
    assert delta[0]["m"][0] == 1 and delta[-1]["m"][1] == max_m
    for left, right in zip(delta, delta[1:], strict=False):
        assert left["m"][1] + 1 == right["m"][0]
    assert all(r["provider"] == "triton-lora-v1" and r["config"] is not None for r in delta)
    assert all(r["op"] in ("delta", "base") for r in routes)
    compiled = len(_lora_segmented._COMPILED)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for m in (1, 3, 17, 64, max_m):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = lora_projection(
                x[:m], weight, None, factors, scaling=2.0, inference_plan=plan
            )
        graph.replay()
        torch.cuda.synchronize()
        eager = lora_projection(x[:m], weight, None, factors, scaling=2.0, inference_plan=plan)
        assert torch.equal(captured, eager)
    assert len(_lora_segmented._COMPILED) == compiled
    with pytest.raises(ValueError):
        plan.warm_up(0)
