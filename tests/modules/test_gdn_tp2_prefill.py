# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only TP2 dispatch, fallback and geometry scheduling contracts."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _cute_prefill, _fla, _triton_prefill, inference
from ctikernels.gdn._contract import Geometry

from .test_gdn import ACCEPTED_DEVICES, _on_device, _prefill_arguments

TP2 = [(8, 16), (8, 24)]


@pytest.fixture
def no_port(monkeypatch):
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")


@pytest.mark.parametrize("kind", ACCEPTED_DEVICES)
@pytest.mark.parametrize("hk,hv", TP2)
@pytest.mark.parametrize("policy", ["enabled", "disabled", "unavailable"])
def test_tp2_binds_cute_or_triton(kind, hk, hv, policy, monkeypatch, no_port):
    _on_device(monkeypatch, *ACCEPTED_DEVICES[kind])
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "0" if policy == "disabled" else "1")
    sentinel = SimpleNamespace(provider="fla-core")
    calls = []

    def cute_load(device, heads):
        assert heads == (hk, hv)
        assert policy != "disabled"
        return None if policy == "unavailable" else object()

    def fla_load(*args):
        calls.append(args)
        return SimpleNamespace(prefill=sentinel)

    monkeypatch.setattr(_cute_prefill, "load", cute_load)
    monkeypatch.setattr(_fla, "load", fla_load)
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cpu")
    if policy == "enabled":
        assert op.provider == "cute" and op.supports_state_capture
        assert calls == []
        for tokens, count in ((1, 1), (17, 17), (65, 2), (2048, 3)):
            assert op.selected_variant(tokens, count) == "cute"
    else:
        assert op.provider == "triton" and op.supports_state_capture
        assert calls == []


@pytest.mark.parametrize("hk,hv", TP2)
@pytest.mark.parametrize("capture", [False, True])
def test_tp2_bound_cute_preserves_arguments_out_and_guard(hk, hv, capture, monkeypatch, no_port):
    _on_device(monkeypatch, *ACCEPTED_DEVICES[_triton_prefill.RTX_5090])
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    calls = []
    args = tuple(
        torch.empty(t.shape, dtype=t.dtype)
        for t in _prefill_arguments(hk, hv, tokens=4, sequences=1, slots=3).values()
    )
    rows = torch.ones(1, dtype=torch.int32) if capture else None
    kwargs = {"capture_rows": rows} if capture else {}
    output = torch.empty(4, hv * 128, dtype=torch.bfloat16)

    def prefill(geometry, *actual, **keywords):
        assert geometry == Geometry(hk, hv)
        assert all(a is b for a, b in zip(actual, args, strict=True))
        assert keywords.keys() == {"out", "capture_rows"}
        assert keywords["out"] is output and keywords["capture_rows"] is rows
        calls.append(1)
        return torch.full_like(output, 3)

    monkeypatch.setattr(_cute_prefill, "load", lambda *a: prefill)
    monkeypatch.setattr(_fla, "load", lambda *a: pytest.fail("in-call fla fallback"))
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cpu")
    assert op.selected_variant(4, 1) == "cute"
    assert op(*args, out=output, **kwargs) is output
    assert torch.all(output == 3) and calls == [1]
    # A mock callable without capture_variant retains the shape guard.
    assert all(shape == (4, 1) and variant[0] == "cute" for shape, variant in op._guard._eager)
    for weight in args[1:5]:
        weight.requires_grad_(True)
        with pytest.raises(ValueError, match="must not require grad"):
            op(*args, out=output, **kwargs)
        weight.requires_grad_(False)
    assert calls == [1]


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("hv,parent", [(16, 32), (24, 48)])
def test_tp2_provisional_tunes_and_overrides(cc, hv, parent, monkeypatch):
    assert (
        _cute_prefill.GEOMETRY_TUNES[cc, (8 if hv in (16, 24) else 16, hv)]
        == _cute_prefill.GEOMETRY_TUNES[cc, (8 if parent in (16, 24) else 16, parent)]
    )
    assert _cute_prefill.read_tune(
        "", cc, (8 if hv in (16, 24) else 16, hv)
    ) == _cute_prefill.read_tune("", cc, (8 if parent in (16, 24) else 16, parent))
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "bv=16,overlap=1,prep_mode=0")
    tune = _cute_prefill.read_tune(capability=cc, geometry=(8 if hv in (16, 24) else 16, hv))
    assert (tune["bv"], tune["overlap"], tune["prep_mode"]) == (16, 1, 0)
    for setting in ("aq_out=1", "o_form=pre"):
        assert _cute_prefill.read_tune(setting, cc, (8 if hv in (16, 24) else 16, hv))["bv"] == 32
        assert (
            _cute_prefill.read_tune(setting + ",bv=64", cc, (8 if hv in (16, 24) else 16, hv))["bv"]
            == 64
        )


@pytest.mark.parametrize("hv", [16, 24])
@pytest.mark.parametrize("sms", [48, 170, 188])
@pytest.mark.parametrize("captures", [0, 1])
def test_tp2_wave_class_and_grouped_prepare(hv, sms, captures):
    geometry = Geometry(8, hv)
    launcher = object.__new__(_cute_prefill.Launcher)
    launcher.tune = _cute_prefill.read_tune(
        "overlap=1,bv=32", (12, 0), (8 if hv in (16, 24) else 16, hv)
    )
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    launcher.sms = sms
    x = torch.empty(2048, geometry.proj_width, dtype=torch.bfloat16)
    variants = {}
    for count in (1, 2, 3, 5, 17):
        key = launcher.compile_key(geometry, x, count, captures)
        wave = count * hv * 4 <= sms
        assert key[3] == wave
        assert key == variants.setdefault(wave, key)
        tune = launcher._effective_tune(hv, wave, key)
        assert tune["prep_mode"] == 1
        assert tune["prefetch_depth"] == (4 if wave else 1)
        _cute_prefill.validate_captures(tune, captures)
        info = {r["kernel"]: r for r in _cute_prefill.launch_info(8, hv, 2048, count, tune, wave)}
        assert info["prepare_value"]["grid"] == [
            8 * _cute_prefill.chunk_capacity(2048, count),
            1,
            1,
        ]
        assert info["recurrence"]["grid"] == [4 * hv * count, 1, 1]
        assert info["recurrence"]["dynamic_smem_bytes"] + 2048 <= (96 if wave else 48) * 1024
