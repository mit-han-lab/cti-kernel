# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Resolution caches retain launch code, never caller buffers or pointer admission."""

import contextlib

import pytest
import torch

from ctikernels.activation import inference as act
from ctikernels.norm import inference as norm


@pytest.mark.parametrize("operation", ("norm", "activation"))
def test_cute_resolution_reuses_compiled_handle_with_fresh_buffers(monkeypatch, operation):
    module = norm if operation == "norm" else act
    if operation == "norm":
        from ctikernels.norm import cute
        from ctikernels.norm._dispatch import Config

        op = norm.AddRMSNorm(provider="cute", config=Config())
    else:
        from ctikernels.activation import cute
        from ctikernels.activation._dispatch import Config

        op = act.SiluAndMul(provider="cute", config=Config())
    monkeypatch.setattr(module, "_hardware", lambda _: ("gb10", (12, 1)))
    monkeypatch.setattr(module, "_admitted", lambda *args: True)
    monkeypatch.setattr(module._flashinfer_version, "allowed", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cute, "available", lambda: True)
    resolutions, launches = [], []

    def compile_handle(*args, **kwargs):
        resolutions.append((args, kwargs))

        def launch(*tensors):
            launches.append(tensors[0].data_ptr())
            target = tensors[0] if operation == "norm" else tensors[1]
            target.fill_(7)

        return launch

    monkeypatch.setattr(cute, "get_compiled", compile_handle)
    inputs = [torch.ones(3, 16, dtype=torch.bfloat16) for _ in range(2)]
    for x in inputs:
        if operation == "norm":
            op(x, torch.ones_like(x), torch.ones(16, dtype=x.dtype), 1e-6)
            result = x
        else:
            result = op(x)
        assert torch.all(result == 7)
    assert len(resolutions) == 1
    assert launches == [x.data_ptr() for x in inputs]

    # Changing shape must resolve again even though the lower compilation cache
    # deliberately supports dynamic M. Epsilon remains a dynamic launch argument.
    x = torch.ones(4, 16, dtype=torch.bfloat16)
    if operation == "norm":
        op(x, torch.ones_like(x), torch.ones(16, dtype=x.dtype), 2e-5)
    else:
        op(x)
    assert len(resolutions) == 2


@pytest.mark.parametrize("operation", ("norm", "activation"))
def test_capture_miss_does_not_cache_temporary_fallback(monkeypatch, operation):
    op = norm.AddRMSNorm() if operation == "norm" else act.SiluAndMul()
    capturing = [False]
    resolutions = []
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: torch.device("cuda:0")))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])

    def resolve(*args):
        resolutions.append(capturing[0])
        return ("flashinfer" if capturing[0] else "cute"), lambda *args: None

    monkeypatch.setattr(op, "_resolve", resolve)

    def call(rows):
        x = torch.ones(rows, 16)
        if operation == "norm":
            op(x, torch.ones_like(x), torch.ones(16), 1e-6)
        else:
            op(x)

    call(3)
    capturing[0] = True
    call(3)
    assert resolutions == [False]  # Warm signature is reused inside capture.
    call(4)
    assert op.provider == "flashinfer"
    capturing[0] = False
    call(4)
    assert op.provider == "cute"  # Eager warmup retries after the capture miss.
    capturing[0] = True
    call(4)
    assert resolutions == [False, True, False]


@pytest.mark.parametrize("operation", ("norm", "activation"))
def test_device_index_has_a_separate_resolution(monkeypatch, operation):
    op = norm.AddRMSNorm() if operation == "norm" else act.SiluAndMul()
    device = [0]
    resolutions = []
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: torch.device("cuda", device[0])))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        op, "_resolve", lambda *args: (resolutions.append(device[0]) or "cute", lambda *args: None)
    )
    x = torch.ones(3, 16)
    for index in (0, 0, 1, 1, 0):
        device[0] = index
        if operation == "norm":
            op(x, torch.ones_like(x), torch.ones(16), 1e-6)
        else:
            op(x)
    assert resolutions == [0, 1]
