# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import pytest
import torch

from ctikernels.rotary.vision import apply_rotary


def _reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Eager vision RoPE (transformers 5.16.1, `models/qwen3_5/modeling_qwen3_5.py:882-893`)."""

    first, second = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (x.float() * cos[:, None] + rotated * sin[:, None]).to(x.dtype)


def _angles(num_tokens: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.randn(num_tokens, head_dim // 2, dtype=torch.float32, device="cuda")
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos(), angles.sin()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_vision_rotary_matches_eager_reference_on_merged_qkv_views(dtype: torch.dtype) -> None:
    torch.manual_seed(13)
    num_tokens, num_heads, head_dim = 257, 16, 72
    qkv = torch.randn(num_tokens, 3, num_heads, head_dim, dtype=dtype, device="cuda")
    query, key, _ = qkv.unbind(1)
    cos, sin = _angles(num_tokens, head_dim)
    original = qkv.clone()

    with torch.inference_mode():
        actual_query, actual_key = apply_rotary(query, key, cos, sin)
    torch.cuda.synchronize()

    assert not query.is_contiguous() and not key.is_contiguous()
    assert actual_query.is_contiguous() and actual_key.is_contiguous()
    torch.testing.assert_close(actual_query, _reference(query, cos, sin), rtol=0, atol=0)
    torch.testing.assert_close(actual_key, _reference(key, cos, sin), rtol=0, atol=0)
    torch.testing.assert_close(qkv, original, rtol=0, atol=0)


def test_vision_rotary_uses_current_cuda_stream() -> None:
    torch.manual_seed(14)
    num_tokens, num_heads, head_dim = 64, 16, 72
    qkv = torch.randn(num_tokens, 3, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cos, sin = _angles(num_tokens, head_dim)
    expected_query = _reference(qkv[:, 0], cos, sin)
    expected_key = _reference(qkv[:, 1], cos, sin)
    staged = torch.zeros_like(qkv)
    query, key, _ = staged.unbind(1)
    with torch.inference_mode():
        apply_rotary(query, key, cos, sin)  # compile before the stream is held below
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        # Hold the stream so that a launch on any other stream reads the inputs unwritten.
        torch.cuda._sleep(50_000_000)
        staged.copy_(qkv)
        actual_query, actual_key = apply_rotary(query, key, cos, sin)
        completion = torch.cuda.Event()
        completion.record(stream)
    torch.cuda.current_stream().wait_event(completion)

    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)


def test_vision_rotary_replays_in_cuda_graph() -> None:
    torch.manual_seed(15)
    num_tokens, num_heads, head_dim = 64, 16, 72
    qkv = torch.zeros(num_tokens, 3, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    query, key, _ = qkv.unbind(1)
    cos = torch.zeros(num_tokens, head_dim, device="cuda")
    sin = torch.zeros_like(cos)

    with torch.inference_mode():
        apply_rotary(query, key, cos, sin)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        actual_query, actual_key = apply_rotary(query, key, cos, sin)

    for _ in range(3):
        fresh_cos, fresh_sin = _angles(num_tokens, head_dim)
        qkv.copy_(torch.randn_like(qkv))
        cos.copy_(fresh_cos)
        sin.copy_(fresh_sin)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_query, _reference(query, cos, sin), rtol=0, atol=0)
        torch.testing.assert_close(actual_key, _reference(key, cos, sin), rtol=0, atol=0)


def test_vision_rotary_accepts_empty_tokens() -> None:
    query = torch.empty((0, 2, 8), device="cuda", dtype=torch.bfloat16)
    cos = torch.empty((0, 8), device="cuda")

    with torch.inference_mode():
        actual_query, actual_key = apply_rotary(query, query, cos, cos)

    assert actual_query.shape == query.shape and actual_key.shape == query.shape


def test_vision_rotary_rejects_invalid_inputs() -> None:
    query = torch.randn((4, 2, 8), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((4, 8), device="cuda")
    sin = torch.zeros((4, 8), device="cuda")

    with pytest.raises(ValueError, match="CUDA"):
        apply_rotary(query.cpu(), query.cpu(), cos.cpu(), sin.cpu())
    with pytest.raises(ValueError, match="even"):
        apply_rotary(query[..., :7], query[..., :7], cos[:, :7], sin[:, :7])
    with pytest.raises(ValueError, match=r"\[tokens, head_dim\]"):
        apply_rotary(query, query, cos[:3], sin[:3])
    with pytest.raises(TypeError, match="FP32"):
        apply_rotary(query, query, cos.bfloat16(), sin.bfloat16())
    strided = torch.randn((4, 2, 16), device="cuda", dtype=torch.bfloat16)[..., ::2]
    with pytest.raises(ValueError, match="last dimension"):
        apply_rotary(strided, strided, cos, sin)
    with pytest.raises(RuntimeError, match="autograd"):
        apply_rotary(query.float().requires_grad_(), query.float(), cos, sin)
