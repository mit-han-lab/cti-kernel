# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only token dispatch, specialization and graph-cache contracts."""

from types import SimpleNamespace

import pytest
import torch

# The MoE training modules import Triton; the CPU CI env has none (the GPU suites run these).
pytest.importorskip("triton")

from ctikernels.moe._training_backend import _MoEBlock
from ctikernels.moe.cute import _token
from ctikernels.moe.cute._provider import CuteBackend
from ctikernels.moe.triton._provider import _TritonBackend


@pytest.mark.parametrize("setting,enabled", ((None, True), ("0", False), ("1", True)))
def test_backend_flag(monkeypatch, setting, enabled):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr("ctikernels.moe.cute._provider.supports", lambda *_: True)
    if setting is None:
        monkeypatch.delenv("CTI_MOE_CUTE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CTI_MOE_CUTE_TOKEN", setting)
    be = CuteBackend("cuda", 64, 64, 64)
    assert be.device_token is enabled
    monkeypatch.setenv("CTI_MOE_CUTE_TOKEN", "0" if enabled else "1")
    assert be.device_token is enabled  # Retained by the forward's autograd context.
    assert not _TritonBackend.device_token


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("wide", (False, True))
def test_cache_and_capture(monkeypatch, shared, wide):
    compiled, launched = [], []

    def compile_fake(*key):
        compiled.append(key)
        return lambda *args: launched.append(args)

    monkeypatch.setattr(_token, "_COMPILED", {})
    monkeypatch.setattr(_token, "compile_token", compile_fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    be = object.__new__(CuteBackend)
    be.capability = (12, 0)
    k, h, rows = 2, 2048, 32
    r = torch.empty(256, h, dtype=torch.bfloat16)
    sg = torch.empty(1, h, dtype=torch.bfloat16) if shared else None
    for t in (1, 7, 1):  # Both counts reuse the same split specialization.
        dx = torch.empty(t, h, dtype=torch.bfloat16)
        dxp = torch.empty(rows, h, dtype=torch.bfloat16)
        inv = torch.empty(t * (k + shared), dtype=torch.int32)
        dwpart = torch.empty(rows, 8)
        ws = torch.empty(t * (k + shared))
        sel = torch.empty(t, k, dtype=torch.int64 if wide else torch.int32)
        be.bwd_token(dxp, inv, dwpart, ws, sel, r, sg, dx, None)
        assert launched[-1][0] is dxp and launched[-1][7] is dx
        assert launched[-1][6] is (sg if shared else r)
    assert compiled == [(h, 256, k, 8, shared, False, wide, 8, (12, 0))]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    be.bwd_token(dxp, inv, dwpart, ws, sel, r, sg, dx, None)
    valid = torch.tensor(0, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Warm up"):
        be.bwd_token(dxp, inv, dwpart, ws, sel, r, sg, dx, valid)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    be.bwd_token(dxp, inv, dwpart, ws, sel, r, sg, dx, valid)
    assert len(compiled) == 2 and compiled[-1][5]
    assert launched[-1][-1].data_ptr() == valid.data_ptr()
    be.capability = (12, 1)
    be.bwd_token(dxp, inv, dwpart, ws, sel, r, sg, dx, valid)
    assert len(compiled) == 3 and compiled[-1][-1] == (12, 1)


@pytest.mark.parametrize("t", (1, 7, 64, 1000, 4097, 8192, 16384))
@pytest.mark.parametrize("h", (64, 192, 256, 2048, 4096))
def test_grid_covers_each_vector_once(t, h):
    splits = _token.token_splits(t, h)
    columns = [
        (tile * splits + split) * 256 + lane * 8
        for split in range(splits)
        for tile in range((h + 256 * splits - 1) // (256 * splits))
        for lane in range(32)
        if (tile * splits + split) * 256 + lane * 8 < h
    ]
    assert sorted(columns) == list(range(0, h, 8))


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("enabled,k", ((True, 2), (False, 2), (True, 9)))
def test_block_backward_dispatch(monkeypatch, shared, enabled, k):
    from ctikernels.moe import _training_backend as _block

    t, h, rows = 7, 64, 128
    x = torch.empty(t, h, dtype=torch.bfloat16)
    r = torch.empty(16, h, dtype=torch.bfloat16)
    sg = torch.empty(1, h, dtype=torch.bfloat16) if shared else None
    w13, w2 = torch.empty(16, 128, h), torch.empty(16, h, 64)
    s13, s2 = (torch.empty(128, h), torch.empty(h, 64)) if shared else (None, None)
    sel, ws = torch.empty(t, k, dtype=torch.int32), torch.empty(t, k + shared)
    calls = []

    class Backend:
        device_token = enabled

        def dw_tiles(self, *args, **kwargs):
            return 2

        def bwd_slot(self, *args, **kwargs):
            pass

        def dx_pair(self, gg, w13, s13, lay, dxp, **kwargs):
            assert dxp.dtype == torch.bfloat16 and dxp.shape[0] == rows
            dxp.fill_(3)

        def bwd_token(self, dxp, inv, dwpart, weights, selected, router, gate, dx, valid):
            calls.append(("cute", dx.shape[0]))
            assert router is r and gate is sg and dwpart.shape == (rows, 2)
            assert weights.shape == (dx.shape[0] * (k + shared),)
            assert selected.is_contiguous() and valid.dtype == torch.int32
            assert dxp.shape[0] == rows
            dx.fill_(5)

    class TritonLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(("triton", grid[0]))
                assert kwargs["HAS_SHARED"] == shared
                args[7].fill_(5)

            return launch

    monkeypatch.setattr(_block.K, "_bwd_token_kernel", TritonLaunch())
    chunks = []
    for start, end in ((0, 3), (3, 6), (6, 7)):
        lay = SimpleNamespace(
            sorted_slot=torch.empty(rows, dtype=torch.int32),
            inv=torch.empty((end - start) * (k + shared), dtype=torch.int32),
        )
        chunks.append((start, end, lay, None, None, torch.tensor(1, dtype=torch.int32)))
    ctx = SimpleNamespace(
        saved_tensors=(x, r, sg, w13, w2, s13, s2, sel, ws, ws, None),
        cfg=(k, {"warps": 4}, 64, Backend()),
        chunks=chunks,
        needs_input_grad=(True,) * 16,
    )
    result = _MoEBlock.backward(ctx, torch.empty_like(x), None)
    assert calls == [("cute" if enabled and k <= 8 else "triton", c) for c in (3, 3, 1)]
    assert torch.equal(result[0], torch.full_like(x, 5))
    assert result[1:] == (None,) * 15
