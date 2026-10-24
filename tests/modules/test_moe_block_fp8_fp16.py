# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FP16 metadata, provider routing and dtype-specific capture readiness, without CUDA."""

from contextlib import nullcontext

import pytest
import torch

from ctikernels import _moe_block_fp8_cute_table as policy
from ctikernels.moe_block_fp8 import BlockFP8MoE, _check_inputs


class _NoDevice(nullcontext):
    """Stands in for torch.cuda.device around a CPU op; a class, so torch's own
    isinstance(device, torch.cuda.device) checks keep working on CUDA hosts."""

    def __init__(self, device):
        super().__init__()


@pytest.fixture
def op(monkeypatch):
    op = object.__new__(BlockFP8MoE)
    op.e, op.h, op.i, op.top_k = 2, 128, 128, 2
    op.w13 = torch.empty(2, 256, 128, dtype=torch.float8_e4m3fn)
    op.w2 = torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn)
    op.s13, op.s2 = torch.ones(2, 2, 1), torch.ones(2, 1, 1)
    op.device, op.device_class = torch.device("cpu"), (12, 0, 170)
    op.provider, op._requested_provider = "triton-fp8", None
    op._available, op._enabled, op._preset = True, True, None
    op.expert_map, op.num_global_experts = None, 2
    op._ready, op._route_blocks = set(), {}
    monkeypatch.setattr(torch.cuda, "device", _NoDevice)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(op, "_triton", lambda x, w, ids, config, **shared: x.clone())
    return op


def inputs(m=1, dtype=torch.float16):
    return torch.zeros(m, 128, dtype=dtype), torch.ones(m, 2), torch.zeros(m, 2, dtype=torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [0, 1])
def test_hidden_dtype(op, dtype, m):
    args = inputs(m, dtype)
    _check_inputs(*args, device=op.device, hidden=128, top_k=2)
    assert op(*args).dtype == dtype


@pytest.mark.parametrize("device", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
def test_fp16_bypasses_auto_cute_policy(op, monkeypatch, device):
    op.device_class = device
    preset = policy.heuristic(1)
    monkeypatch.setitem(policy.TABLE, device, [(4096, "cute-fp8", preset)])
    op.provider = op.provider_for(1)
    assert op.provider == "cute-fp8"
    expected = op._triton_config_for(1)
    calls = []

    def forward(x, w, ids, config, **shared):
        assert config == expected
        calls.append(x.dtype)
        return x.clone()

    monkeypatch.setattr(op, "_triton", forward)
    op.prepare([1], dtype=torch.float16)
    assert op(*inputs()).dtype == torch.float16
    assert calls == [torch.float16] * 3  # prepare warms plain and shared; then the call


@pytest.mark.parametrize("m", [0, 1])
def test_explicit_cute_fp16_and_fp32_rejected(op, m):
    with pytest.raises(TypeError, match="FP32 hidden dtype gap.*FP32 compute type"):
        op(*inputs(m, torch.float32))
    op.provider = op._requested_provider = "cute-fp8"
    with pytest.raises(TypeError, match="cute-fp8.*BF16 only.*FP16"):
        op(*inputs(m))
    with pytest.raises(TypeError, match="cute-fp8.*BF16 only"):
        op.prepare([m], dtype=torch.float16)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize(
    "kwargs",
    [{"gate_up_hook": lambda gu: None}, {"down_hook": lambda act: None}, {"return_act": True}],
)
def test_staging_and_dx_bf16_only(op, m, kwargs):
    args = inputs(m)
    with pytest.raises(TypeError, match=r"BF16 only; FP16 inference is supported by op\(x, ...\)"):
        op.forward_staged(*args, **kwargs)
    with pytest.raises(TypeError, match="BF16 only; FP16 inference"):
        op.dx(*args, torch.zeros_like(args[0], dtype=torch.bfloat16))
    assert op.forward_staged(*args).dtype == torch.float16
    with pytest.raises(TypeError, match="BF16 only; FP16 inference"):
        op.prepare([m], dtype=torch.float16, staged=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, "float16"])
def test_prepare_rejects_unsupported_dtype(op, dtype):
    with pytest.raises(TypeError, match="dtype must be"):
        op.prepare([1], dtype=dtype)
    assert not op._ready


@pytest.mark.parametrize("container", [list, tuple])
def test_prepare_dtype_readiness(op, monkeypatch, container):
    op.prepare([1])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert op(*inputs(dtype=torch.bfloat16)).dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match=r"prepare\(\[M\], dtype=x.dtype\)"):
        op(*inputs())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op.prepare(iter([1, 2]), dtype=container([torch.bfloat16, torch.float16]))
    assert len(op._ready) == 4
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for m in (1, 2):
        for dtype in (torch.bfloat16, torch.float16):
            assert op(*inputs(m, dtype)).dtype == dtype


def test_cute_activation_stays_bf16_only():
    pytest.importorskip("cutlass.cute")
    from ctikernels._moe_block_fp8_cute_act import silu_mul_quant_fp8

    q = torch.empty(2, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="gate_up"):
        silu_mul_quant_fp8(torch.zeros(2, 256, dtype=torch.float16), q, torch.empty(2, 1))
