# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Warm-up entry points (KW1-KW3): variant enumeration and idempotence, without CUDA."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels import radix
from ctikernels.attention import _prologue
from ctikernels.gdn import _cute_prefill
from ctikernels.gdn import inference as gdn_inference
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.inference import GatedDeltaNetPrefill, _CaptureGuard

# (capability, Hv, SMs): every tuned prefill geometry on its device type.
GDN_CASES = [
    ((12, 0), 16, 170),
    ((12, 0), 24, 188),
    ((12, 0), 32, 170),
    ((12, 0), 48, 188),
    ((12, 1), 16, 48),
    ((12, 1), 24, 48),
    ((12, 1), 32, 48),
    ((12, 1), 48, 48),
]


def _launcher(monkeypatch, cc, hv, sms):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cc)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )
    return _cute_prefill.Launcher(torch.device("cpu"), object(), (8 if hv in (16, 24) else 16, hv))


@pytest.mark.parametrize("cc,hv,sms", GDN_CASES)
def test_gdn_variant_keys_match_every_reachable_call(monkeypatch, cc, hv, sms):
    launcher = _launcher(monkeypatch, cc, hv, sms)
    geometry = Geometry(8 if hv in (16, 24) else 16, hv)
    stride, max_tokens, max_sequences = geometry.proj_width + 64, 4200, 6
    expected = set()
    for tokens in range(1, max_tokens + 1):
        for count in range(1, min(tokens, max_sequences) + 1):
            for captures in (0, 1):
                try:
                    _cute_prefill.validate_captures(launcher._row_tune(tokens, count)[0], captures)
                except ValueError:
                    continue
                expected.add(launcher.variant_key(geometry, tokens, count, stride, captures))
    keys = launcher.variant_keys(geometry, max_tokens, max_sequences, stride)
    assert set(keys) == expected
    for key, (tokens, count) in keys.items():
        assert launcher.variant_key(geometry, tokens, count, stride, key[6]) == key


def test_gdn_warm_up_compiles_each_key_once(monkeypatch):
    launcher = _launcher(monkeypatch, (12, 0), 32, 170)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiled = []
    monkeypatch.setattr(launcher, "_compile", lambda key: compiled.append(key) or key)
    keys = launcher.variant_keys(Geometry(16, 32), 2560, 3, Geometry(16, 32).proj_width)
    # Every variant comes with its capturing twin.
    assert all((*key[:-1], 1 - key[-1]) in keys for key in keys)
    assert launcher.warm_up(keys) == len(keys) == len(compiled) > 1
    assert launcher.warm_up(keys) == 0 and len(compiled) == len(keys)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="capture"):
        launcher.warm_up(keys)  # every key cached: still refused inside a capture
    launcher.compiled.clear()
    with pytest.raises(RuntimeError, match="capture"):
        launcher.warm_up(keys)
    assert len(compiled) == len(keys)


def test_gdn_eager_compile_builds_both_capture_executables(monkeypatch):
    """The capture guard records one entry per variant, so an eager call compiles both."""
    launcher = _launcher(monkeypatch, (12, 0), 32, 170)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(launcher, "_compile", lambda key: key)
    key = launcher.variant_key(Geometry(16, 32), 512, 1, Geometry(16, 32).proj_width)
    assert launcher._compile_eagerly(key) == key and key[6] == 0
    assert set(launcher.compiled) == {key, (*key[:-1], 1)}


def _prefill_op(monkeypatch, launcher, provider="cute"):
    op = object.__new__(GatedDeltaNetPrefill)
    op._geometry = Geometry(16, 32)
    op._device = torch.device("cpu")
    op._guard = _CaptureGuard("GatedDeltaNetPrefill", op._device)
    op._kernel = SimpleNamespace(
        _cute=launcher, provider=provider, selected_variant=lambda tokens, sequences: provider
    )
    return op


def test_gdn_prefill_warm_up_marks_capture_guard(monkeypatch):
    launcher = _launcher(monkeypatch, (12, 0), 32, 170)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(launcher, "_compile", lambda key: key)
    primed = []
    monkeypatch.setattr(gdn_inference, "prime", lambda kernel, *a: primed.append(kernel))
    op = _prefill_op(monkeypatch, launcher)
    count = op.warm_up(1000, max_sequences=2)
    assert count == len(launcher.compiled) > 0
    assert {key[6] for key in launcher.compiled} == {0, 1}
    # Each executable (with and without captures) loaded once.
    assert len(primed) == count and set(primed) == {(launcher, k) for k in launcher.compiled}
    assert op.warm_up(1000, max_sequences=2) == 0
    stride = Geometry(16, 32).proj_width
    # One guard entry per variant, without any capture count.
    assert op._guard._eager == {
        ((), ("cute", stride, launcher.tune_key, key)) for key in launcher.compiled if not key[6]
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"max_tokens": 8, "max_sequences": 0},
        {"max_tokens": 8, "row_stride": 64},
    ],
)
def test_gdn_prefill_warm_up_rejects_bad_bounds(monkeypatch, kwargs):
    op = _prefill_op(monkeypatch, None, provider="triton")
    with pytest.raises(ValueError):
        op.warm_up(**kwargs)


def test_gdn_prefill_warm_up_skips_other_providers(monkeypatch):
    assert _prefill_op(monkeypatch, None, provider="fla-core").warm_up(64) == 0


L2_5090 = 96 * 1024 * 1024


@pytest.mark.parametrize("variant", sorted(_prologue.DEVICES))
@pytest.mark.parametrize("heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("cache", [None, "e4m3", "e5m2"])
def test_prologue_warm_up_covers_every_token_count(monkeypatch, variant, heads, cache):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=L2_5090)
    )
    launcher = _prologue.Launcher(torch.device("cpu"), SimpleNamespace(), variant)
    compiled = []

    def compile_key(key):
        compiled.append(key)
        launcher._compiled[key] = key

    monkeypatch.setattr(launcher, "_compile", compile_key)
    count = launcher.warm_up(4096, *heads, cache=cache)
    assert count == len(compiled) == len(set(compiled)) > 1
    for tokens in range(1, 4097):
        config = _prologue.config(tokens, *heads, L2_5090, variant[0])
        assert (*heads, *config, *variant, cache, False, None) in launcher._compiled
    assert launcher.warm_up(4096, *heads, cache=cache) == 0
    # The masked kernel is a separate compile key per configuration.
    assert launcher.warm_up(4096, *heads, cache=cache, masked=True) == count
    assert all(key[:-2] + (True, None) in launcher._compiled for key in compiled[:count])


def test_prologue_op_warm_up_refuses_capture_after_warm_up(monkeypatch):
    op = object.__new__(_prologue.Prologue)
    op.device, op.num_q_heads, op.num_kv_heads = torch.device("cpu"), 32, 8
    op._cache, op.masked = None, False
    op._kernel = SimpleNamespace(warm_up=lambda *args, **kwargs: 0)
    op._axis_kernel = None
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert op.warm_up(64) == 0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="capture"):
        op.warm_up(64)


def test_prologue_op_warm_up_bounds_and_compose():
    op = object.__new__(_prologue.Prologue)
    op._kernel = None
    assert op.warm_up(16) == 0
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            op.warm_up(bad)


def test_radix_warm_up_loads_once(monkeypatch):
    calls = []
    module = SimpleNamespace(fast_compare_key=lambda x, y: 3)
    monkeypatch.setattr(radix, "load_aot", lambda *a, **k: calls.append(a) or module)
    radix._load_radix_module.cache_clear()
    try:
        radix.warm_up()
        radix.warm_up()
        assert radix.fast_compare_key(None, None) == 3
        assert calls == [("radix",)]
    finally:
        radix._load_radix_module.cache_clear()
