# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only checks of resident scratch lifetimes and caller-owned alignment."""

import weakref

import pytest
import torch

from ctikernels.moe import _lora_workspace as _workspace
from ctikernels.moe.lora import moe_lora_w13_act_


@pytest.fixture(autouse=True)
def isolated_arenas(monkeypatch):
    monkeypatch.setattr(_workspace, "_ARENAS", {})
    monkeypatch.setattr(_workspace, "_RETIRED", [])


def test_carved_views_alignment_disjointness_and_reuse():
    specs = [((3, 7), torch.bfloat16), ((17,), torch.int32), ((11,), torch.bool)]
    views = _workspace.carve("cpu", specs)
    for j, (view, (shape, dtype)) in enumerate(zip(views, specs, strict=True)):
        assert view.shape == shape and view.dtype == dtype and view.is_contiguous()
        assert view.data_ptr() % 256 == 0
        assert view.untyped_storage().data_ptr() == views[0].untyped_storage().data_ptr()
        if j:
            previous = views[j - 1]
            assert (
                view.data_ptr() >= previous.data_ptr() + previous.numel() * previous.element_size()
            )
        view.fill_(j + 1)
    assert torch.all(views[0] == 1)
    again = _workspace.carve(torch.device("cpu"), specs)
    assert [v.data_ptr() for v in again] == [v.data_ptr() for v in views]


def test_growth_rounding_and_old_buffer_release():
    chunk = 2 * 1024 * 1024
    _workspace.carve("cpu", [((1,), torch.uint8)])
    key = (torch.device("cpu"), None)
    old = weakref.ref(_workspace._ARENAS[key].buffer)
    assert old().numel() == chunk
    _workspace.carve("cpu", [((chunk + 1,), torch.uint8)])
    assert old() is None
    assert _workspace._ARENAS[key].buffer.numel() == 2 * chunk
    _workspace.carve("cpu", [((6 * chunk,), torch.uint8)])
    assert _workspace._ARENAS[key].buffer.numel() == 6 * chunk
    # 1.25 * 12 MiB = 15 MiB, rounded to 16 MiB, even for a one-byte overflow.
    _workspace.carve("cpu", [((6 * chunk + 1,), torch.uint8)])
    assert _workspace._ARENAS[key].buffer.numel() == 8 * chunk
    _workspace.release()
    assert not _workspace._ARENAS and not _workspace._RETIRED


def test_capture_retention_growth_fallback_and_release(monkeypatch):
    key = (torch.device("cpu"), None)
    original = _workspace.carve("cpu", [((4,), torch.int32)])[0]
    original.fill_(23)
    arena = _workspace._ARENAS[key]
    monkeypatch.setattr(_workspace, "_capturing", lambda device: True)
    captured = _workspace.carve("cpu", [((4,), torch.int32)])[0]
    assert captured.data_ptr() == original.data_ptr() and arena.captured
    capacity = arena.buffer.numel()
    fallback = _workspace.carve("cpu", [((capacity + 1,), torch.uint8)])[0]
    assert fallback.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert _workspace._ARENAS[key] is arena
    assert not _workspace._RETIRED
    _workspace.release()
    assert _workspace._ARENAS[key] is arena
    monkeypatch.setattr(_workspace, "_capturing", lambda device: False)
    _workspace.carve("cpu", [((capacity + 1,), torch.uint8)])[0].zero_()
    assert _workspace._RETIRED == [arena]
    assert torch.all(captured == 23)
    _workspace.release()
    assert not _workspace._ARENAS and _workspace._RETIRED == [arena]


def test_capture_without_warmup_does_not_install_arena(monkeypatch):
    monkeypatch.setattr(_workspace, "_capturing", lambda device: True)
    views = _workspace.carve("cpu", [((3,), torch.float32), ((7,), torch.int32)])
    assert [v.numel() for v in views] == [3, 7]
    assert not _workspace._ARENAS


def test_independent_stream_keys(monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(_workspace, "_key", lambda device: (device, 1))
    first = _workspace.carve(device, [((8,), torch.int32)])[0]
    monkeypatch.setattr(_workspace, "_key", lambda device: (device, 2))
    second = _workspace.carve(device, [((8,), torch.int32)])[0]
    assert first.data_ptr() != second.data_ptr()
    monkeypatch.setattr(_workspace, "_key", lambda device: (device, 1))
    assert _workspace.carve(device, [((8,), torch.int32)])[0].data_ptr() == first.data_ptr()


def test_activation_accepts_disjoint_arena_views_and_rejects_overlap():
    c1, c2 = _workspace.carve("cpu", [((2, 32), torch.bfloat16), ((2, 16), torch.bfloat16)])
    c1.fill_(1)
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    factors = tuple(
        torch.zeros(shape).bfloat16() for shape in ((1, 8, 16),) * 2 + ((1, 16, 8),) * 2
    )
    options = {
        "scaling": 1.0,
        "top_k": 2,
        "provider": "torch",
        "sorted_token_ids": torch.zeros(16, dtype=torch.int32),
        "expert_ids": torch.zeros(1, dtype=torch.int32),
        "num_tokens_post_padded": torch.tensor([16], dtype=torch.int32),
        "block_m": 16,
    }
    ids = torch.zeros(1, 2, dtype=torch.int32)
    moe_lora_w13_act_(c2, c1, x, factors, ids, **options)
    assert torch.all(c2 == torch.nn.functional.silu(torch.tensor(1.0)).bfloat16())
    with pytest.raises(ValueError, match="must not alias"):
        moe_lora_w13_act_(c1.view(-1)[8:40].view(2, 16), c1, x, factors, ids, **options)
