# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The public GDN ops' capture guard asks the op's bound device, not the current one."""

from contextlib import contextmanager

import pytest
import torch

from ctikernels.gdn import inference


class Kernel:
    def __init__(self):
        self.calls = 0

    def __call__(self, geometry, projected, *args, **kwargs):
        self.calls += 1
        return projected


@pytest.fixture
def two_devices(monkeypatch):
    """cuda:0 is current; ``capturing[i]`` says whether device i's current stream captures."""
    current = 0
    capturing = {0: False, 1: False}

    @contextmanager
    def device_context(device):
        nonlocal current
        previous, current = current, torch.device(device).index
        try:
            yield
        finally:
            current = previous

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[current])
    monkeypatch.setattr(inference, "cuda_device", lambda name, device: torch.device(device))
    # CPU tensors stand in for validated CUDA inputs; the guard itself stays real.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    return capturing


@pytest.mark.parametrize("bound_capturing", [True, False])
def test_prefill_guard_uses_bound_device(monkeypatch, two_devices, bound_capturing):
    two_devices.update({0: not bound_capturing, 1: bound_capturing})
    kernel = Kernel()
    monkeypatch.setattr(inference, "_prefill_kernel", lambda *a, **k: kernel)
    monkeypatch.setattr(inference, "check_prefill", lambda *a: (64, 3))
    op = inference.GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:1")
    x = torch.empty(64, 1)
    args = [torch.empty(1)] * 9
    args[5] = torch.empty(3)
    if bound_capturing:
        with pytest.raises(RuntimeError, match="GatedDeltaNetPrefill: run shape .* once eagerly"):
            op(x, *args)
        assert not kernel.calls
    else:
        # cuda:0 captures, but this op is bound to the idle cuda:1: an unwarmed eager call runs.
        assert op(x, *args) is x
        assert kernel.calls == 1
    assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("bound_capturing", [True, False])
def test_decode_guard_uses_bound_device(monkeypatch, two_devices, bound_capturing):
    two_devices.update({0: not bound_capturing, 1: bound_capturing})
    kernel = Kernel()
    monkeypatch.setattr(inference, "_decode_kernel", lambda *a, **k: kernel)
    monkeypatch.setattr(inference, "check_decode", lambda *a: 3)
    op = inference.GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:1")
    x = torch.empty(3, 1)
    args = [torch.empty(1)] * 7
    if bound_capturing:
        with pytest.raises(RuntimeError, match="GatedDeltaNetDecode: run shape .* once eagerly"):
            op(x, *args)
        assert not kernel.calls
    else:
        assert op(x, *args) is x
        assert kernel.calls == 1
    assert torch.cuda.current_device() == 0
