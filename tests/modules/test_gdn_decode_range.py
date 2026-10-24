# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""#25's extended dtypes stay local to decode and warm each pointer dtype; grad-marked
storage/inputs/outputs are admitted on replay with grad mode disabled; weights stay frozen."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode, _triton_decode, inference


class RecordingDecode:
    provider = "triton"
    variant = _triton_decode.TritonDecode.variant

    def __init__(self):
        self.calls = []

    def __call__(self, geometry, projected, *args, out=None):
        self.calls.append((projected, *args))
        return (
            out
            if out is not None
            else projected.new_empty(projected.shape[0], geometry.value_width)
        )


@pytest.fixture
def decode(monkeypatch):
    kernel = RecordingDecode()
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(inference, "_decode_kernel", lambda *a, **k: kernel)
    op = GatedDeltaNetDecode(num_k_heads=1, num_v_heads=2, device="cuda:0")
    g = op.geometry
    args = {
        "projected": torch.zeros(2, g.proj_width, dtype=torch.bfloat16),
        "conv_weight": torch.zeros(g.conv_dim, 4, dtype=torch.bfloat16),
        "A_log": torch.zeros(g.v_heads),
        "dt_bias": torch.zeros(g.v_heads),
        "norm_weight": torch.ones(128),
        "slot_indices": torch.tensor([0, -1], dtype=torch.int32),
        "state": torch.zeros(2, g.v_heads, 128, 128),
        "conv_state": torch.zeros(2, 3, g.conv_dim, dtype=torch.bfloat16),
    }
    return op, args, kernel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_decode_range_accepts_dtypes(decode, dtype, index_dtype):
    op, args, kernel = decode
    for name in ("A_log", "dt_bias", "norm_weight"):
        args[name] = args[name].to(dtype)
    args["slot_indices"] = args["slot_indices"].to(index_dtype)
    op(**args)
    assert len(kernel.calls) == 1
    assert all(a is b for a, b in zip(kernel.calls[0], args.values(), strict=True))


@pytest.mark.parametrize(
    "target", ["conv_weight", "A_log", "dt_bias", "norm_weight", "slot_indices"]
)
def test_decode_range_rejects_other_dtypes(decode, target):
    op, args, kernel = decode
    args[target] = args[target].double()
    with pytest.raises(TypeError, match=target):
        op(**args)
    assert not kernel.calls


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize(
    "target",
    ["projected", "conv_weight", "A_log", "dt_bias", "norm_weight", "state", "conv_state", "out"],
)
def test_decode_range_grad_rule(decode, context, target):
    op, args, kernel = decode
    if target == "out":
        args[target] = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    args[target] = torch.nn.Parameter(args[target])
    error = (
        ValueError if target in ("conv_weight", "A_log", "dt_bias", "norm_weight") else RuntimeError
    )
    with torch.enable_grad(), pytest.raises(error, match="grad"):
        op(**args)
    with torch.enable_grad(), pytest.raises(error, match="grad"):
        op(**args, _replay=True)
    assert not kernel.calls
    # Weights stay frozen on every route and in every grad mode. Replay alone
    # admits grad-marked pools under disabled grad, along with projected/out.
    if target not in ("projected", "out"):
        with context(), pytest.raises(error, match="grad"):
            op(**args)
        assert not kernel.calls
    if target in ("conv_weight", "A_log", "dt_bias", "norm_weight"):
        with context(), pytest.raises(ValueError, match="grad"):
            op(**args, _replay=True)
        assert not kernel.calls
    else:
        with context():
            op(**args, _replay=True)
        assert len(kernel.calls) == 1


@pytest.mark.parametrize("target", ["A_log", "dt_bias", "norm_weight", "slot_indices"])
def test_decode_range_capture_requires_dtype_warmup(decode, monkeypatch, target):
    op, args, kernel = decode
    op(**args)
    args[target] = args[target].to(torch.int64 if target == "slot_indices" else torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="once eagerly"):
        op(**args)
    assert len(kernel.calls) == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op(**args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op(**args)
    assert len(kernel.calls) == 3


def test_decode_range_warms_scratch_for_each_slot_dtype(decode, monkeypatch):
    _, args, _ = decode
    launches = []

    class ZeroKernel:
        def __getitem__(self, grid):
            def launch(scratch, slots, words, **kw):
                launches.append((slots.dtype, kw["PER_SLOT"]))

            return launch

    monkeypatch.setitem(
        sys.modules,
        "ctikernels.gdn._triton_decode_kernels",
        SimpleNamespace(zero_scratch_kernel=ZeroKernel()),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    pools = _triton_decode._Pools()
    for dtype in (torch.int32, torch.int64):
        pools.shared(args["state"], args["slot_indices"].to(dtype), 16, "flags")
    assert launches == [
        (torch.int32, False),
        (torch.int32, True),
        (torch.int64, False),
        (torch.int64, True),
    ]


def test_decode_range_plan_cache_tracks_dtypes(decode, monkeypatch):
    op, args, _ = decode
    launches, plans = [], []

    class DecodeKernel:
        def __getitem__(self, grid):
            return lambda *a, **kw: launches.append(a)

    monkeypatch.setitem(
        sys.modules,
        "ctikernels.gdn._triton_decode_kernels",
        SimpleNamespace(gdn_decode_kernel=DecodeKernel()),
    )
    monkeypatch.setattr(_triton_decode, "_POOLS", SimpleNamespace(shared=lambda *a: None))
    kernel = object.__new__(_triton_decode.TritonDecode)
    kernel._plans = {}

    def plan(*args):
        plans.append(args)
        return _triton_decode._Plan((1,), 1, (("shared", 16, "flags"),), {})

    monkeypatch.setattr(kernel, "_plan", plan)
    for target, dtype in (
        ("A_log", torch.bfloat16),
        ("dt_bias", torch.float16),
        ("norm_weight", torch.bfloat16),
        ("slot_indices", torch.int64),
    ):
        kernel(op.geometry, **args)
        args[target] = args[target].to(dtype)
        kernel(op.geometry, **args)
    assert len(plans) == len(kernel._plans) == 5
    assert len(launches) == 8


def test_reset_scratch_reaches_the_kernel_eagerly_and_is_refused_under_capture(decode, monkeypatch):
    op, args, kernel = decode
    pools = []
    kernel.reset_scratch = pools.append
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    op.reset_scratch(args["state"])
    op.reset_scratch()
    assert pools[0] is args["state"] and pools[1] is None
    capturing[0] = True
    with pytest.raises(RuntimeError, match="cannot be captured"):
        op.reset_scratch(args["state"])
    assert len(pools) == 2
    del kernel.reset_scratch
    op.reset_scratch(args["state"])  # a provider without scratch (fla-core): nothing to do
