# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch


class _RotaryCall:
    def __init__(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kwargs: dict[str, object],
    ) -> None:
        self.x = x
        self.cos = cos
        self.sin = sin
        self.kwargs = kwargs


def _load_training_rotary(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[_RotaryCall],
) -> ModuleType:
    def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(_RotaryCall(x, cos, sin, kwargs))
        return x.clone()

    package = ModuleType("flash_attn")
    layers = ModuleType("flash_attn.layers")
    rotary = ModuleType("flash_attn.layers.rotary")
    rotary.apply_rotary_emb = apply_rotary_emb
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.layers", layers)
    monkeypatch.setitem(sys.modules, "flash_attn.layers.rotary", rotary)
    monkeypatch.delitem(sys.modules, "ctikernels.rotary.training", raising=False)
    return importlib.import_module("ctikernels.rotary.training")


def _hf_partial_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Partial NeoX rotary as transformers 5.16.1 writes it (`apply_rotary_pos_emb`)."""

    rotary_dim = 2 * cos.shape[1]
    selected_cos = cos.index_select(0, positions).repeat(1, 2).unsqueeze(-2)
    selected_sin = sin.index_select(0, positions).repeat(1, 2).unsqueeze(-2)
    rotated, passed = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return torch.cat((rotated * selected_cos + rotated_half * selected_sin, passed), dim=-1)


def test_padded_rotary_passes_shared_position_offset_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    query = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 1, 8, dtype=torch.bfloat16)
    query_snapshot = query.clone()
    key_snapshot = key.clone()
    cos = torch.randn(32, 4)
    sin = torch.randn(32, 4)
    positions = torch.arange(7, 12)

    rotated_query, rotated_key = training_rotary.apply_rotary(
        query,
        key,
        cos,
        sin,
        positions,
    )

    assert len(calls) == 2
    assert calls[0].x is query
    assert calls[1].x is key
    for call in calls:
        assert call.kwargs["interleaved"] is False
        assert call.kwargs["inplace"] is False
        assert call.kwargs["cu_seqlens"] is None
        assert call.kwargs["max_seqlen"] is None
        torch.testing.assert_close(call.kwargs["seqlen_offsets"], torch.tensor([7, 7]))
    assert rotated_query.data_ptr() != query.data_ptr()
    assert rotated_key.data_ptr() != key.data_ptr()
    torch.testing.assert_close(query, query_snapshot)
    torch.testing.assert_close(key, key_snapshot)


def test_packed_rotary_passes_each_sequence_offset_and_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    query = torch.randn(7, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(7, 1, 8, dtype=torch.bfloat16)
    cos = torch.randn(32, 4)
    sin = torch.randn(32, 4)
    positions = torch.tensor([2, 3, 4, 9, 10, 11, 12])
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    training_rotary.apply_rotary(
        query,
        key,
        cos,
        sin,
        positions,
        cu_seqlens=cu_seqlens,
        max_sequence_length=4,
    )

    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["cu_seqlens"] is cu_seqlens
        assert call.kwargs["max_seqlen"] == 4
        torch.testing.assert_close(call.kwargs["seqlen_offsets"], torch.tensor([2, 9]))


def test_fp32_rotary_preserves_deterministic_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    query = torch.randn(2, 5, 3, 8, requires_grad=True)
    key = torch.randn(2, 5, 1, 8, requires_grad=True)
    cos = torch.randn(32, 4)
    sin = torch.randn(32, 4)
    positions = torch.arange(7, 12)

    actual_query, actual_key = training_rotary.apply_rotary(
        query,
        key,
        cos,
        sin,
        positions,
    )

    selected_cos = cos.index_select(0, positions).view(1, 5, 1, 4)
    selected_sin = sin.index_select(0, positions).view(1, 5, 1, 4)

    def reference(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat(
            (
                first * selected_cos - second * selected_sin,
                second * selected_cos + first * selected_sin,
            ),
            dim=-1,
        )

    assert not calls
    torch.testing.assert_close(actual_query, reference(query), rtol=0, atol=0)
    torch.testing.assert_close(actual_key, reference(key), rtol=0, atol=0)
    assert actual_query.data_ptr() != query.data_ptr()
    assert actual_key.data_ptr() != key.data_ptr()


def test_partial_rotary_hands_the_narrow_caches_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    query = torch.randn(7, 3, 16, dtype=torch.bfloat16)
    key = torch.randn(7, 1, 16, dtype=torch.bfloat16)
    cos = torch.randn(32, 2)
    sin = torch.randn(32, 2)

    training_rotary.apply_rotary(
        query,
        key,
        cos,
        sin,
        torch.tensor([2, 3, 4, 9, 10, 11, 12]),
        cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_sequence_length=4,
    )

    assert len(calls) == 2
    assert calls[0].x is query
    assert calls[1].x is key
    for call in calls:
        assert call.cos is cos
        assert call.sin is sin


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_fp32_partial_rotary_rotates_leading_channels_and_passes_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    head_dim, rotary_dim = 16, 4
    cos = torch.randn(32, rotary_dim // 2)
    sin = torch.randn(32, rotary_dim // 2)
    if layout == "padded":
        positions = torch.arange(7, 12)
        query_data = torch.randn(2, 5, 3, head_dim)
        key_data = torch.randn(2, 5, 1, head_dim)
        packing = {}
    else:
        positions = torch.tensor([2, 3, 4, 9, 10, 11, 12])
        query_data = torch.randn(7, 3, head_dim)
        key_data = torch.randn(7, 1, head_dim)
        packing = {
            "cu_seqlens": torch.tensor([0, 3, 7], dtype=torch.int32),
            "max_sequence_length": 4,
        }
    query = query_data.clone().requires_grad_()
    key = key_data.clone().requires_grad_()
    query_upstream = torch.randn_like(query_data)
    key_upstream = torch.randn_like(key_data)

    actual_query, actual_key = training_rotary.apply_rotary(
        query, key, cos, sin, positions, **packing
    )
    torch.autograd.backward((actual_query, actual_key), (query_upstream, key_upstream))

    reference_query = query_data.double().requires_grad_()
    reference_key = key_data.double().requires_grad_()
    torch.autograd.backward(
        (
            _hf_partial_rotary(reference_query, cos.double(), sin.double(), positions),
            _hf_partial_rotary(reference_key, cos.double(), sin.double(), positions),
        ),
        (query_upstream.double(), key_upstream.double()),
    )

    assert not calls
    for x, rotated, gradient, upstream, reference_gradient in (
        (query_data, actual_query, query.grad, query_upstream, reference_query.grad),
        (key_data, actual_key, key.grad, key_upstream, reference_key.grad),
    ):
        expected = _hf_partial_rotary(x, cos, sin, positions)
        torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            gradient[..., rotary_dim:], upstream[..., rotary_dim:], rtol=0, atol=0
        )
        torch.testing.assert_close(gradient, reference_gradient.float(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("cos_shape", "sin_shape", "key_head_dim", "match"),
    (
        ((32,), (32,), 8, "share shape"),
        ((32, 4), (32, 3), 8, "share shape"),
        ((32, 4), (32, 4), 6, "share head_dim"),
        ((32, 5), (32, 5), 8, "rotary_dim 10 must be"),
        ((32, 0), (32, 0), 8, "rotary_dim 0 must be"),
    ),
)
def test_training_rotary_rejects_incompatible_caches(
    monkeypatch: pytest.MonkeyPatch,
    cos_shape: tuple[int, ...],
    sin_shape: tuple[int, ...],
    key_head_dim: int,
    match: str,
) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    with pytest.raises(ValueError, match=match):
        training_rotary.apply_rotary(
            torch.randn(2, 5, 3, 8, dtype=torch.bfloat16),
            torch.randn(2, 5, 1, key_head_dim, dtype=torch.bfloat16),
            torch.randn(cos_shape),
            torch.randn(sin_shape),
            torch.arange(5),
        )
    assert not calls


@pytest.mark.parametrize(
    (
        "query_shape",
        "key_shape",
        "num_positions",
        "cu_seqlens",
        "max_sequence_length",
        "match",
    ),
    (
        ((5, 3, 8), (5, 1, 8), 5, None, None, "padded query"),
        ((2, 5, 3, 8), (2, 4, 1, 8), 5, None, None, "batch sequence geometry"),
        ((5, 3, 8), (5, 1, 8), 4, (0, 5), 5, "token count"),
        ((5, 3, 8), (5, 1, 8), 5, (0, 5), None, "max_sequence_length"),
    ),
)
def test_training_rotary_rejects_incompatible_layout_metadata(
    monkeypatch: pytest.MonkeyPatch,
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    num_positions: int,
    cu_seqlens: tuple[int, ...] | None,
    max_sequence_length: int | None,
    match: str,
) -> None:
    training_rotary = _load_training_rotary(monkeypatch, [])
    boundaries = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        training_rotary.apply_rotary(
            torch.randn(query_shape),
            torch.randn(key_shape),
            torch.randn(32, 4),
            torch.randn(32, 4),
            torch.arange(num_positions),
            cu_seqlens=boundaries,
            max_sequence_length=max_sequence_length,
        )


def test_training_rotary_takes_int64_positions_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[_RotaryCall] = []
    training_rotary = _load_training_rotary(monkeypatch, calls)
    query = torch.randn(1, 5, 3, 8, dtype=torch.bfloat16)
    cos = sin = torch.randn(32, 4)
    positions = torch.arange(5, dtype=torch.int32)
    with pytest.raises(TypeError, match="int64"):
        training_rotary.apply_rotary(query, query.clone(), cos, sin, positions)
    assert calls == []
