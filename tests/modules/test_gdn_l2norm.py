# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks of L2 norm's validation and direct-JIT dispatch, without importing FLA."""

import sys
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _l2norm


@pytest.mark.parametrize("shape", [(128,), (1, 15, 16, 128), (17, 128), (8193, 128)])
def test_l2norm_bypasses_the_autotuner_and_preserves_shapes(monkeypatch, shape):
    calls = []
    compiled = object()

    class Jit:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append((grid, kwargs))
                return compiled

            return launch

    # No __getitem__ on the autotuners: only their .fn can be launched.
    monkeypatch.setitem(
        sys.modules,
        "fla.modules.l2norm",
        SimpleNamespace(
            l2norm_fwd_kernel=SimpleNamespace(fn=Jit()),
            l2norm_bwd_kernel=SimpleNamespace(fn=Jit()),
        ),
    )
    x = torch.empty(shape, dtype=torch.bfloat16)
    y, rstd = _l2norm.l2norm_forward(x, eps=1e-5)
    dx = _l2norm.l2norm_backward(y, rstd, torch.empty_like(y), eps=1e-5)
    assert y.shape == dx.shape == x.shape
    assert y.dtype == dx.dtype == x.dtype
    assert rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32
    assert len(calls) == 2
    for grid, args in calls:
        assert grid == ((x.numel() // 128 + 15) // 16,)
        assert args["T"] == x.numel() // 128 and args["eps"] == 1e-5
        assert {k: args[k] for k in ("D", "BD", "NB", "BT", "num_warps")} == {
            "D": 128,
            "BD": 128,
            "NB": 1,
            "BT": 16,
            "num_warps": 4,
        }
    assert _l2norm._launch_forward(x, y, rstd) is compiled
    assert _l2norm._launch_backward(y, rstd, dx, dx) is compiled


def test_empty_l2norm_neither_imports_fla_nor_launches(monkeypatch):
    monkeypatch.setitem(sys.modules, "fla.modules.l2norm", None)
    x = torch.empty(0, 16, 128, dtype=torch.bfloat16)
    y, rstd = _l2norm.l2norm_forward(x)
    dx = _l2norm.l2norm_backward(y, rstd, x)
    assert y.shape == dx.shape == x.shape
    assert y.dtype == dx.dtype == x.dtype
    assert rstd.shape == (0, 16) and rstd.dtype == torch.float32


@pytest.mark.parametrize(
    "x",
    [torch.empty(()), torch.empty(2, 64), torch.empty(2, 256)[:, ::2], torch.empty(4, 128)[::2]],
)
def test_l2norm_rejects_unsupported_width_or_strides(x):
    with pytest.raises(ValueError):
        _l2norm.l2norm_forward(x)
    with pytest.raises(ValueError):
        _l2norm.l2norm_backward(x, torch.empty(x.shape[:-1]), x)


def test_l2norm_backward_rejects_mismatched_or_strided_inputs():
    y, rstd = torch.empty(2, 128), torch.empty(2)
    for dy in (torch.empty(3, 128), torch.empty(2, 256)[:, ::2]):
        with pytest.raises(ValueError):
            _l2norm.l2norm_backward(y, rstd, dy)
    for invalid in (torch.empty(3), torch.empty(2, dtype=torch.bfloat16), torch.empty(4)[::2]):
        with pytest.raises(ValueError, match="rstd"):
            _l2norm.l2norm_backward(y, invalid, y)
