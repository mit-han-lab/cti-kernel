# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.utils.checkpoint import checkpoint

# Training rotary runs on flash-attn, an optional dependency.
pytest.importorskip("flash_attn.layers.rotary")

from ctikernels.rotary.training import apply_rotary

HEAD_DIM = 128
NUM_QUERY_HEADS = 8
NUM_KEY_HEADS = 2
MAX_POSITION = 512
RELATIVE_L2_LIMIT = 3e-5
# Partial rotary as Qwen3.5 configures it: 64 of 256 channels, base 1e7.
PARTIAL_HEAD_DIM = 256
PARTIAL_ROTARY_DIM = 64
PARTIAL_BASE = 10_000_000


def _rotary_cache(
    rotary_dim: int = HEAD_DIM,
    base: float = 10_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    frequencies = torch.outer(
        torch.arange(MAX_POSITION, dtype=torch.float32, device="cuda"),
        inv_freq,
    )
    return frequencies.cos(), frequencies.sin()


def _oracle(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    selected_cos = cos.index_select(0, positions).unsqueeze(-2)
    selected_sin = sin.index_select(0, positions).unsqueeze(-2)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        (
            first * selected_cos - second * selected_sin,
            second * selected_cos + first * selected_sin,
        ),
        dim=-1,
    ).to(x.dtype)


def _fp64_partial_oracle(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """FP64 partial NeoX rotary of transformers 5.16.1 `apply_rotary_pos_emb`.

    `models/qwen3_5/modeling_qwen3_5.py:557-592` rotates the leading ``rotary_dim``
    channels as ``x * cos + rotate_half(x) * sin`` and concatenates the rest. The
    FP32 caches are widened, not recomputed, so only the operation's error remains.
    """

    rotary_dim = 2 * cos.shape[1]
    selected_cos = cos.double().index_select(0, positions).repeat(1, 2).unsqueeze(-2)
    selected_sin = sin.double().index_select(0, positions).repeat(1, 2).unsqueeze(-2)
    rotated, passed = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return torch.cat((rotated * selected_cos + rotated_half * selected_sin, passed), dim=-1)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    reference = torch.linalg.vector_norm(expected.float())
    assert reference > 0
    return float((difference / reference).detach())


def _branched_loss(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    query_float = query.float()
    key_float = key.float()
    return (
        0.25 * query_float.square().sum()
        + query_float.sin().sum()
        + 0.5 * key_float.square().sum()
        + key_float.cos().sum()
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert _relative_l2(actual, expected) <= RELATIVE_L2_LIMIT


def _padded_case(
    head_dim: int = HEAD_DIM,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    int | None,
]:
    sequence_length = 257
    positions = torch.arange(11, 11 + sequence_length, device="cuda")
    query = torch.randn(
        2,
        sequence_length,
        NUM_QUERY_HEADS,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(
        2,
        sequence_length,
        NUM_KEY_HEADS,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return query, key, positions, None, None


def _packed_case(
    head_dim: int = HEAD_DIM,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    lengths = (71, 193, 43)
    starts = (5, 17, 3)
    positions = torch.cat(
        [
            torch.arange(start, start + length, device="cuda")
            for start, length in zip(starts, lengths, strict=True)
        ]
    )
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    query = torch.randn(
        sum(lengths),
        NUM_QUERY_HEADS,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(
        sum(lengths),
        NUM_KEY_HEADS,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return query, key, positions, cu_seqlens, max(lengths)


@pytest.mark.parametrize("case", (_padded_case, _packed_case), ids=("padded", "packed"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_training_rotary_matches_independent_oracle(
    case: Callable[
        [],
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            int | None,
        ],
    ],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260722)
    query, key, positions, cu_seqlens, max_sequence_length = case()
    query, key = query.to(dtype), key.to(dtype)
    cos, sin = _rotary_cache()
    query_snapshot = query.clone()
    key_snapshot = key.clone()

    actual_query = query.detach().clone().requires_grad_()
    actual_key = key.detach().clone().requires_grad_()
    rotated_query, rotated_key = apply_rotary(
        actual_query,
        actual_key,
        cos,
        sin,
        positions,
        cu_seqlens=cu_seqlens,
        max_sequence_length=max_sequence_length,
    )
    _branched_loss(rotated_query, rotated_key).backward()

    expected_query = query.detach().clone().requires_grad_()
    expected_key = key.detach().clone().requires_grad_()
    oracle_query = _oracle(expected_query, cos, sin, positions)
    oracle_key = _oracle(expected_key, cos, sin, positions)
    _branched_loss(oracle_query, oracle_key).backward()

    assert not torch._C._overlaps(rotated_query, actual_query)
    assert not torch._C._overlaps(rotated_key, actual_key)
    torch.testing.assert_close(actual_query, query_snapshot, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, key_snapshot, rtol=0, atol=0)
    _assert_close(rotated_query, oracle_query)
    _assert_close(rotated_key, oracle_key)
    assert actual_query.grad is not None and expected_query.grad is not None
    assert actual_key.grad is not None and expected_key.grad is not None
    _assert_close(actual_query.grad, expected_query.grad)
    _assert_close(actual_key.grad, expected_key.grad)


def test_training_rotary_uses_current_stream_and_accumulates_backward() -> None:
    torch.manual_seed(20260723)
    query, key, positions, cu_seqlens, max_sequence_length = _packed_case()
    cos, sin = _rotary_cache()
    query.requires_grad_()
    key.requires_grad_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        first_query, first_key = apply_rotary(
            query,
            key,
            cos,
            sin,
            positions,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        _branched_loss(first_query, first_key).backward()
        first_query_gradient = query.grad.clone()
        first_key_gradient = key.grad.clone()

        second_query, second_key = apply_rotary(
            query,
            key,
            cos,
            sin,
            positions,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
        )
        _branched_loss(second_query, second_key).backward()
        completion = torch.cuda.Event()
        completion.record(stream)

    torch.cuda.current_stream().wait_event(completion)
    torch.testing.assert_close(query.grad, 2 * first_query_gradient, rtol=0, atol=0)
    torch.testing.assert_close(key.grad, 2 * first_key_gradient, rtol=0, atol=0)


def test_training_rotary_composes_with_reentrant_checkpointing() -> None:
    torch.manual_seed(20260724)
    query, key, positions, _, _ = _padded_case()
    cos, sin = _rotary_cache()

    eager_query = query.detach().clone().requires_grad_()
    eager_key = key.detach().clone().requires_grad_()
    eager_outputs = apply_rotary(eager_query, eager_key, cos, sin, positions)
    _branched_loss(*eager_outputs).backward()

    checkpoint_query = query.detach().clone().requires_grad_()
    checkpoint_key = key.detach().clone().requires_grad_()

    def rotate(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary(q, k, cos, sin, positions)

    checkpoint_outputs = checkpoint(
        rotate,
        checkpoint_query,
        checkpoint_key,
        use_reentrant=True,
        preserve_rng_state=False,
    )
    _branched_loss(*checkpoint_outputs).backward()

    torch.testing.assert_close(checkpoint_outputs, eager_outputs, rtol=0, atol=0)
    assert checkpoint_query.grad is not None and eager_query.grad is not None
    assert checkpoint_key.grad is not None and eager_key.grad is not None
    torch.testing.assert_close(checkpoint_query.grad, eager_query.grad, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_key.grad, eager_key.grad, rtol=0, atol=0)


@pytest.mark.parametrize("case", (_padded_case, _packed_case), ids=("padded", "packed"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_partial_training_rotary_matches_fp64_oracle(
    case: Callable[
        [int],
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            int | None,
        ],
    ],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20261003)
    query, key, positions, cu_seqlens, max_sequence_length = case(PARTIAL_HEAD_DIM)
    query, key = query.to(dtype), key.to(dtype)
    cos, sin = _rotary_cache(PARTIAL_ROTARY_DIM, PARTIAL_BASE)
    query_upstream = torch.randn_like(query)
    key_upstream = torch.randn_like(key)

    actual_query = query.clone().requires_grad_()
    actual_key = key.clone().requires_grad_()
    rotated_query, rotated_key = apply_rotary(
        actual_query,
        actual_key,
        cos,
        sin,
        positions,
        cu_seqlens=cu_seqlens,
        max_sequence_length=max_sequence_length,
    )
    torch.autograd.backward((rotated_query, rotated_key), (query_upstream, key_upstream))

    expected_query = query.double().requires_grad_()
    expected_key = key.double().requires_grad_()
    oracle_query = _fp64_partial_oracle(expected_query, cos, sin, positions)
    oracle_key = _fp64_partial_oracle(expected_key, cos, sin, positions)
    torch.autograd.backward(
        (oracle_query, oracle_key),
        (query_upstream.double(), key_upstream.double()),
    )

    rotary = PARTIAL_ROTARY_DIM
    for x, rotated, oracle, gradient, upstream, oracle_gradient in (
        (
            query,
            rotated_query,
            oracle_query,
            actual_query.grad,
            query_upstream,
            expected_query.grad,
        ),
        (key, rotated_key, oracle_key, actual_key.grad, key_upstream, expected_key.grad),
    ):
        assert rotated.dtype == dtype and gradient is not None and gradient.dtype == dtype
        torch.testing.assert_close(rotated[..., rotary:], x[..., rotary:], rtol=0, atol=0)
        torch.testing.assert_close(gradient[..., rotary:], upstream[..., rotary:], rtol=0, atol=0)
        _assert_close(rotated[..., :rotary], oracle[..., :rotary].to(dtype))
        _assert_close(gradient[..., :rotary], oracle_gradient[..., :rotary].to(dtype))
    torch.testing.assert_close(actual_query, query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
