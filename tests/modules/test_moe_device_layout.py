# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU semantics of the unordered CuTe layout contract versus the stable path."""

import pytest
import torch

# The MoE training modules import Triton; the CPU CI env has none (the GPU suites run these).
pytest.importorskip("triton")

from ._moe_layout_reference import (
    assert_layout,
    build_slot_layout,
    layout_reference,
    old_route,
)
from ._moe_layout_reference import slot_expert as _slot_expert


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("mapped", (False, True))
def test_empty_tokens_without_compile(monkeypatch, shared, mapped):
    from ctikernels.moe.cute import _routing as device

    def unexpected(*args, **kwargs):
        pytest.fail("empty routing/layout reached compilation or launch")

    monkeypatch.setattr(device, "_compiled", unexpected)
    sel, live = torch.empty(0, 2, dtype=torch.int32), torch.empty(0, dtype=torch.bool)
    mapping = torch.arange(8, dtype=torch.int32) if mapped else None
    actual = device.build_layout(sel, live, 8, 16, mapping, has_shared=shared, capability=(12, 0))
    expected = build_slot_layout(torch.empty(0, dtype=torch.int64), 8, 16)
    assert actual.block_m == expected.block_m
    for name in ("sorted_slot", "block_expert", "inv"):
        got, want = getattr(actual, name), getattr(expected, name)
        assert got.dtype == want.dtype and got.device == want.device
        assert torch.equal(got, want)


@pytest.mark.parametrize("t", (1, 7, 64, 1000, 8192))
@pytest.mark.parametrize(
    "shared,replay,mapped",
    ((True, False, False), (True, True, True), (False, False, True), (False, True, False)),
)
def test_contract(t, shared, replay, mapped):
    rng = torch.Generator().manual_seed(t)
    e, k, local = 17, 8, 5 if mapped else 17
    # Discrete logits force many ties, including at the selection boundary.
    logits = torch.randint(-3, 4, (t, e), generator=rng).bfloat16()
    s = torch.randn(t, generator=rng)
    routing = torch.randint(e, (t, k), dtype=torch.uint8, generator=rng) if replay else None
    mapping = torch.arange(e, dtype=torch.int32) % local if mapped else None
    if mapped:
        mapping[::2] = -1
    valid = torch.tensor(t // 2, dtype=torch.int32)
    got = old_route(logits, s, k, routing, valid, has_shared=shared)
    se = _slot_expert(got[0].long(), got[2], local, mapping, shared)
    for bm in (16, 64):
        expected = build_slot_layout(se, local, bm)
        actual = layout_reference(got[0], got[2], local, bm, mapping, has_shared=shared)
        assert_layout(actual, expected)


@pytest.mark.parametrize("valid", (-(2**31), 0, 1, 7, 2**31 - 1, None))
@pytest.mark.parametrize("shared", (False, True))
def test_empty_local_and_valid_bounds(valid, shared):
    logits, s = torch.zeros(7, 256, dtype=torch.bfloat16), torch.zeros(7)
    v = None if valid is None else torch.tensor(valid, dtype=torch.int32)
    want = old_route(logits, s, 8, valid=v, has_shared=shared)
    mapping = torch.full((256,), -1, dtype=torch.int32)
    actual = layout_reference(want[0], want[2], 0, 32, mapping, has_shared=shared)
    expected = build_slot_layout(_slot_expert(want[0], want[2], 0, mapping, shared), 0, 32)
    assert_layout(actual, expected)


def test_backend_layout_flag(monkeypatch):
    from ctikernels.moe.cute import _provider as cute
    from ctikernels.moe.triton._provider import _TritonBackend

    monkeypatch.setattr(cute, "supports", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.delenv("CTI_MOE_CUTE_LAYOUT", raising=False)
    enabled = cute.CuteBackend("cuda", 64, 64, 64)
    assert enabled.device_layout
    monkeypatch.setenv("CTI_MOE_CUTE_LAYOUT", "0")
    assert not cute.CuteBackend("cuda", 64, 64, 64).device_layout
    assert enabled.device_layout
    monkeypatch.setenv("CTI_MOE_CUTE_LAYOUT", "1")
    assert not _TritonBackend.device_layout


@pytest.mark.parametrize("shared", (False, True))
def test_shape_only_cache_and_capture(monkeypatch, shared):
    from ctikernels.moe.cute import _routing as device

    monkeypatch.setattr(device, "_COMPILED", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiled, launched = [], []

    def fake(kind):
        def compile_fn(*key):
            compiled.append((kind, key))
            return lambda *args: launched.append((kind, args))

        return compile_fn

    monkeypatch.setattr(device, "compile_layout", fake("layout"))
    cap = (12, 0)
    for t in (7, 64, 8192):
        sel, live = torch.empty(t, 3, dtype=torch.int32), torch.empty(t, dtype=torch.bool)
        lay = device.build_layout(sel, live, 17, 16, has_shared=shared, capability=cap)
        p = t * (3 + shared)
        assert lay.inv.shape == (p,)
        assert lay.sorted_slot.shape == ((p + 15) // 16 * 16 + 18 * 16,)
        assert lay.block_expert.shape == (lay.sorted_slot.numel() // 16,)
        assert launched[-1][1][5] is lay.sorted_slot
    # Token counts are symbolic: only the small/large layout boundary adds a key.
    assert [kind for kind, _ in compiled] == ["layout", "layout"]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    device.build_layout(sel, live, 17, 16, has_shared=shared, capability=cap)
    assert len(compiled) == 2
    with pytest.raises(RuntimeError, match="Warm up"):
        device.build_layout(sel[:, :2], live, 17, 16, has_shared=shared, capability=cap)
