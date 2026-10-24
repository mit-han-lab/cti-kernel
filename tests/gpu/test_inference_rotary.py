# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import pytest
import torch

from ctikernels.rotary.inference import apply_rotary_, apply_rotary_key_

_HEAD_DIM = 128
_MAX_POSITION = 512


def _rotary_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32, device="cuda") / _HEAD_DIM)
    )
    frequencies = torch.outer(
        torch.arange(_MAX_POSITION, dtype=torch.float32, device="cuda"),
        inv_freq,
    )
    cos = frequencies.cos()
    sin = frequencies.sin()
    return cos, sin, torch.cat((cos, sin), dim=-1)


def _oracle(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    selected_cos = cos.index_select(0, positions).unsqueeze(-2)
    selected_sin = sin.index_select(0, positions).unsqueeze(-2)
    first, second = value.float().chunk(2, dim=-1)
    return torch.cat(
        (
            first * selected_cos - second * selected_sin,
            second * selected_cos + first * selected_sin,
        ),
        dim=-1,
    ).to(value.dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_inference_rotary_matches_oracle_in_place_on_current_stream(dtype: torch.dtype) -> None:
    torch.manual_seed(20260730)
    positions = torch.arange(7, 264, device="cuda")
    query = torch.randn(
        positions.numel(),
        8,
        _HEAD_DIM,
        dtype=dtype,
        device="cuda",
    )
    key = torch.randn(
        positions.numel(),
        2,
        _HEAD_DIM,
        dtype=dtype,
        device="cuda",
    )
    cos, sin, cache = _rotary_cache()
    expected_query = _oracle(query, cos, sin, positions)
    expected_key = _oracle(key, cos, sin, positions)
    query_pointer = query.data_ptr()
    key_pointer = key.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.inference_mode():
        actual_query, actual_key = apply_rotary_(
            query,
            key,
            cache,
            positions,
            head_dim=_HEAD_DIM,
        )
        completion = torch.cuda.Event()
        completion.record(stream)

    torch.cuda.current_stream().wait_event(completion)
    assert actual_query is query and actual_query.data_ptr() == query_pointer
    assert actual_key is key and actual_key.data_ptr() == key_pointer
    torch.testing.assert_close(actual_query, expected_query, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_key, expected_key, rtol=2e-2, atol=2e-2)


def test_inference_rotary_cuda_graph_replays_changed_inputs() -> None:
    torch.manual_seed(20260909)
    positions = torch.arange(17, device="cuda")
    query = torch.randn((17, 8, _HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((17, 2, _HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cos, sin, cache = _rotary_cache()

    with torch.inference_mode():
        apply_rotary_(query, key, cache, positions, head_dim=_HEAD_DIM)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        actual_query, actual_key = apply_rotary_(query, key, cache, positions, head_dim=_HEAD_DIM)

    for offset in (7, 211):
        replay_query = torch.randn_like(query)
        replay_key = torch.randn_like(key)
        replay_positions = torch.arange(offset, offset + 17, device="cuda")
        expected_query = _oracle(replay_query, cos, sin, replay_positions)
        expected_key = _oracle(replay_key, cos, sin, replay_positions)
        query.copy_(replay_query)
        key.copy_(replay_key)
        positions.copy_(replay_positions)
        graph.replay()
        torch.cuda.synchronize()

        assert actual_query is query and actual_key is key
        torch.testing.assert_close(actual_query, expected_query, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_key, expected_key, rtol=2e-2, atol=2e-2)


def _cpu_key_cache(head_dim: int) -> torch.Tensor:
    inverse = 10_000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    frequencies = torch.outer(torch.arange(_MAX_POSITION, dtype=torch.float32), inverse)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


def _cpu_key_oracle(
    key: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    assert key.device.type == cache.device.type == positions.device.type == "cpu"
    head_dim = cache.shape[1]
    value = key.reshape(key.shape[0], -1, head_dim)
    cos, sin = cache.chunk(2, dim=-1)
    return _oracle(value, cos, sin, positions.to(torch.int64)).reshape_as(key)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("tokens", [1, 17, 65])
@pytest.mark.parametrize("layout", ["merged_qkv", "strided_heads"])
def test_key_only_rotary_matches_cpu_and_preserves_other_storage(
    dtype: torch.dtype, head_dim: int, tokens: int, layout: str
) -> None:
    generator = torch.Generator().manual_seed(20260909 + tokens + head_dim)
    heads = 3
    shape = (
        (tokens + 2, (heads + 4) * head_dim)
        if layout == "merged_qkv"
        else (tokens + 2, heads + 2, 2 * head_dim)
    )
    original = torch.randn(shape, generator=generator, dtype=dtype)

    def key_view(storage):
        if layout == "merged_qkv":
            return storage[1:-1, head_dim : (heads + 1) * head_dim]
        return storage[1:-1, 1:-1, ::2]

    cache_cpu = _cpu_key_cache(head_dim)
    positions_cpu = (torch.arange(tokens) * 7 + 3) % _MAX_POSITION
    expected = _cpu_key_oracle(key_view(original), cache_cpu, positions_cpu)
    storage = original.to("cuda")
    key = key_view(storage)
    cache = cache_cpu.to("cuda")
    position_storage = torch.full((2 * tokens,), -1, device="cuda", dtype=torch.int64)
    positions = position_storage[::2]
    positions.copy_(positions_cpu)
    original_positions = position_storage.clone()
    pointer = key.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        actual = apply_rotary_key_(key, cache, positions, head_dim=head_dim)
        completion = torch.cuda.Event()
        completion.record()
    torch.cuda.current_stream().wait_event(completion)

    assert actual is key and actual.data_ptr() == pointer
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
    written = torch.zeros(shape, dtype=torch.bool)
    key_view(written).fill_(True)
    torch.testing.assert_close(storage.cpu()[~written], original[~written], rtol=0, atol=0)
    torch.testing.assert_close(cache.cpu(), cache_cpu, rtol=0, atol=0)
    torch.testing.assert_close(position_storage, original_positions, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_key_only_rotary_graph_replays_changed_key_and_positions(dtype: torch.dtype) -> None:
    head_dim, tokens, heads = 128, 17, 2
    cache_cpu = _cpu_key_cache(head_dim)
    cache = cache_cpu.to("cuda")
    storage = torch.zeros((tokens, (heads + 2) * head_dim), dtype=dtype, device="cuda")
    key = storage[:, head_dim : (heads + 1) * head_dim]
    positions = torch.arange(tokens, device="cuda")
    with torch.inference_mode():
        apply_rotary_key_(key, cache, positions, head_dim=head_dim)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        actual = apply_rotary_key_(key, cache, positions, head_dim=head_dim)

    generator = torch.Generator().manual_seed(20260909)
    for offset in (7, 211, 3):
        original = torch.randn(storage.shape, generator=generator, dtype=dtype)
        replay_positions = torch.arange(offset, offset + tokens)
        expected = _cpu_key_oracle(
            original[:, head_dim : (heads + 1) * head_dim], cache_cpu, replay_positions
        )
        storage.copy_(original)
        positions.copy_(replay_positions)
        graph.replay()
        torch.cuda.synchronize()

        assert actual is key
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            storage[:, :head_dim].cpu(), original[:, :head_dim], rtol=0, atol=0
        )
        torch.testing.assert_close(
            storage[:, -head_dim:].cpu(), original[:, -head_dim:], rtol=0, atol=0
        )
        torch.testing.assert_close(cache.cpu(), cache_cpu, rtol=0, atol=0)
        torch.testing.assert_close(positions.cpu(), replay_positions, rtol=0, atol=0)


def test_rotary_operations_take_int64_positions_only() -> None:
    query = torch.randn((3, 2, _HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((3, 1, _HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    snapshot = query.clone(), key.clone()
    _, _, cache = _rotary_cache()
    positions = torch.arange(3, device="cuda", dtype=torch.int32)
    with pytest.raises(TypeError, match="int64"):
        apply_rotary_(query, key, cache, positions, head_dim=_HEAD_DIM)
    with pytest.raises(TypeError, match="int64"):
        apply_rotary_key_(key, cache, positions, head_dim=_HEAD_DIM)
    assert torch.equal(query, snapshot[0]) and torch.equal(key, snapshot[1])
