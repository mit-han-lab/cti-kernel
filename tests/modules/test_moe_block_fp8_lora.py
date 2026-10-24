# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Staged metadata, orchestration and GPU-free compilation checks."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE, _apply_gate_up_hook, _check_down_delta


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
    op.device, op.device_class, op.provider = torch.device("cpu"), (12, 0, 1), "triton-fp8"
    op._available, op._enabled = True, True
    op.expert_map, op.num_global_experts = None, 2
    op._ready, op._route_blocks = set(), {}
    monkeypatch.setattr(torch.cuda, "device", _NoDevice)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return op


def inputs(m=1):
    return (
        torch.zeros(m, 128, dtype=torch.bfloat16),
        torch.ones(m, 2),
        torch.zeros(m, 2, dtype=torch.int32),
    )


@pytest.mark.parametrize("name", ["gate_up_hook", "down_hook"])
def test_hooks_must_be_callable(op, name):
    with pytest.raises(TypeError, match=name):
        op.forward_staged(*inputs(), **{name: 1})


def test_staged_validation(op, monkeypatch):
    with pytest.raises(TypeError, match="return_act"):
        op.forward_staged(*inputs(), return_act=1)
    with pytest.raises(TypeError, match="staged"):
        op.prepare([1], staged=1)
    x, w, ids = inputs()
    with pytest.raises(TypeError, match="topk_ids"):
        op.forward_staged(x, w, ids.long(), return_act=True)
    with pytest.raises(ValueError, match="shape"):
        op.forward_staged(x, w[:, :1], ids, return_act=True)
    monkeypatch.setattr(op, "provider_for", lambda m: "cute-fp8")
    with pytest.raises(NotImplementedError, match="triton-fp8"):
        op.forward_staged(x, w, ids, return_act=True)
    monkeypatch.setattr(op, "provider_for", lambda m: "triton-fp8")
    op._available = False
    with pytest.raises(RuntimeError, match="sm120/121"):
        op.forward_staged(x, w, ids, down_hook=lambda a: None)


def test_gate_hook_in_place_contract():
    gu = torch.zeros(2, 256, dtype=torch.bfloat16)
    _apply_gate_up_hook(lambda gu: gu.add_(1), gu)
    assert (gu == 1).all()
    with pytest.raises(ValueError, match="same gate_up"):
        _apply_gate_up_hook(lambda gu: gu.clone(), gu)
    with pytest.raises(ValueError, match="reallocate"):
        _apply_gate_up_hook(lambda gu: gu.set_(torch.zeros_like(gu)), gu)
    with pytest.raises(ValueError, match="shape"):
        _apply_gate_up_hook(lambda gu: gu.resize_(1, 512), gu)


@pytest.mark.parametrize("kind", ["dtype", "shape", "stride", "grad", "alignment", "device"])
def test_delta_metadata(kind):
    x = inputs()[0]
    delta = torch.zeros(2, 128, dtype=torch.bfloat16)
    error, match = ValueError, "shape"
    if kind == "dtype":
        delta, error, match = delta.float(), TypeError, "down delta"
    elif kind == "shape":
        delta = delta[:1]
    elif kind == "stride":
        delta, match = torch.zeros(128, 2, dtype=torch.bfloat16).T, "contiguous"
    elif kind == "grad":
        delta.requires_grad_()
        match = "inference"
    elif kind == "alignment":
        delta = torch.zeros(257, dtype=torch.bfloat16)[1:].view(2, 128)
        match = "aligned"
    else:
        delta, match = torch.empty_like(delta, device="meta"), "device"
    with pytest.raises(error, match=match):
        _check_down_delta(delta, x, 2)
    _check_down_delta(None, x, 2)
    _check_down_delta(torch.zeros(2, 128, dtype=torch.bfloat16), x, 2)


@pytest.mark.parametrize("fused_act", [False, True])
@pytest.mark.parametrize("has_delta", [False, True])
def test_staged_chain(op, monkeypatch, fused_act, has_delta):
    from ctikernels import moe_impl
    from ctikernels.activation import inference
    from ctikernels.linear import block_fp8

    calls = []
    config = op.config_for(1)
    for c in config:
        c["route_blocks"] = True
    config[1]["act"] = "cute-silu-quant" if fused_act else "split"
    monkeypatch.setattr(op, "config_for", lambda m: config)

    def quantize(x):
        return x, torch.ones(x.shape[0], x.shape[1] // 128)

    def gemm(q, w, out, *args, **kwargs):
        calls.append("fc1" if w is op.w13 else "fc2")
        if w is op.w2:
            assert (q == 3).all()
        out.fill_(1)

    def gate(gu):
        calls.append("gate")
        assert gu.shape == (2, 256) and (gu == 1).all()
        gu.add_(2)

    def activation(gu, act):
        calls.append("act")
        assert (gu == 3).all()
        act.copy_(gu[:, :128])

    def fused(gu, q, scales, act_out=None):
        activation(gu, act_out)
        q.copy_(act_out)

    def down(act):
        calls.append("down")
        assert (act == 3).all()
        return torch.full((2, 128), 4, dtype=torch.bfloat16) if has_delta else None

    def reduce(down, ids, out, weights, *, expert_map, delta):
        calls.append("reduce")
        assert (delta is None) == (not has_delta) and weights.dtype == torch.float32
        assert expert_map is op.expert_map
        out.fill_(5)

    monkeypatch.setattr(moe_impl, "fused_moe_kernel_triton", gemm)
    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quantize)
    monkeypatch.setattr(inference, "silu_and_mul", activation)
    monkeypatch.setitem(
        sys.modules, "ctikernels._moe_block_fp8_cute_act", SimpleNamespace(silu_mul_quant_fp8=fused)
    )
    monkeypatch.setitem(
        sys.modules, "ctikernels._moe_block_fp8_reduce", SimpleNamespace(masked_reduce=reduce)
    )
    out, act = op.forward_staged(*inputs(), gate_up_hook=gate, down_hook=down, return_act=True)
    assert (out == 5).all() and (act == 3).all()
    assert calls == ["fc1", "gate", "act", "fc2", "down", "reduce"]


def test_prepare_staged_readiness(op, monkeypatch):
    calls = []

    def forward(x, w, ids, config, **kw):
        calls.append(kw)
        return (x, x) if kw.get("return_act") else x

    monkeypatch.setattr(op, "_triton", forward)
    op.prepare([1])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert op.forward_staged(*inputs()).shape == (1, 128)
    with pytest.raises(RuntimeError, match="staged=True"):
        op.forward_staged(*inputs(), return_act=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op.prepare([1], staged=True)
    assert len(op._ready) == 4
    assert any(c.get("down_hook") is not None for c in calls)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for kw in ({"gate_up_hook": lambda a: None}, {"return_act": True}, {"down_hook": lambda a: a}):
        op.forward_staged(*inputs(), **kw)
    op._enabled = False  # Configuration changes invalidate readiness.
    with pytest.raises(RuntimeError, match="staged=True"):
        op.forward_staged(*inputs(), return_act=True)


def test_empty_staged(op):
    def fail(a):
        raise AssertionError("empty input must not call hooks")

    out, act = op.forward_staged(*inputs(0), gate_up_hook=fail, down_hook=fail, return_act=True)
    assert out.shape == (0, 128) and act.shape == (0, 128)


def test_reduce_validation():
    pytest.importorskip("cutlass.cute")
    from ctikernels._moe_block_fp8_reduce import masked_reduce

    x, w, ids = inputs()
    down = torch.zeros(2, 128, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="weights"):
        masked_reduce(down, ids, x, w.bfloat16(), delta=down)
    with pytest.raises(ValueError, match="nonempty vector"):
        masked_reduce(down, ids, x, w, expert_map=torch.empty(0, dtype=torch.int32))
    with pytest.raises(ValueError, match="shape"):
        masked_reduce(down[:1], ids, x, w)
    with pytest.raises(ValueError, match="CUDA device"):
        masked_reduce(down, ids, x, w)


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize(
    "has_map,has_delta", [(False, False), (False, True), (True, False), (True, True)]
)
def test_offline_reduction(arch, has_map, has_delta, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", arch)
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass.cute")
    from ctikernels._moe_block_fp8_reduce import compile_kernel

    fn = compile_kernel(
        2048,
        8,
        arch=arch,
        has_map=has_map,
        has_delta=has_delta,
        options=f"--keep-ptx --dump-dir={tmp_path}",
    )
    ptx = fn.__ptx__
    if "\n" not in ptx:
        ptx = Path(ptx).read_text()
    assert f".target {arch}" in ptx
    assert "ld.global.v4.b32" in ptx and "st.global.v4.b32" in ptx
    assert "cvt.rn.bf16.f32" in ptx or "cvt.rn.bf16x2.f32" in ptx
    assert "atom." not in ptx and "fma.rn.f32" in ptx  # the ordered FMA chain, no atomics
    assert ("add.bf16" in ptx) == has_delta  # y' = BF16(y + delta), one rounding


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize("i", [512, 768])
def test_offline_activation_export(arch, i, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", arch)
    pytest.importorskip("cutlass.cute")
    from ctikernels._moe_block_fp8_cute_act import compile_kernel

    compile_kernel(i, arch=arch, return_act=True)
