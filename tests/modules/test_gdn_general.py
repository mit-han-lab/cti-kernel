# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GEN-GDN admission, frozen dispatch, empty calls and launch limits, CPU only."""

import hashlib
import json
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _cute_prefill as cute
from ctikernels.gdn import _fla, _triton_decode, _triton_prefill, inference
from ctikernels.gdn._contract import Geometry, check_cu_seqlens

from ._gdn_legacy import PREFILL_DIGESTS, legacy_config
from .test_gdn import ACCEPTED_DEVICES, _on_device, _prefill_arguments

GRID = [(hk, r * hk) for r in (1, 2, 3, 4, 6, 8) for hk in range(1, 33)]
OLD = [(8, 16), (8, 24), (16, 32), (16, 48)]


def test_decode_configuration_is_frozen():
    """The measured cards' configurations, the wide rules now chosen by SM count alone; at 48
    SMs whole heads became 64/8 flags."""
    for (hk, hv), batch, sms in product(OLD, range(1, 257), (48, 170, 188)):
        wide = sms >= _triton_decode.WIDE_SMS
        expected = legacy_config(batch, hv, hk, sms, wide)
        if sms <= _triton_decode.SMALL_SMS and expected == (128, 8, False):
            expected = (64, 8, False)
        assert _triton_decode._config(batch, hv, hk, sms) == expected


def test_large_ratios_never_select_finishers():
    for (hk, hv), batch, sms in product(GRID, range(1, 257), (48, 108, 132, 170, 188)):
        if hv // hk > 4:
            assert not _triton_decode._config(batch, hv, hk, sms)[2]


def _launcher(monkeypatch, geometry, cc=(12, 0), sms=170):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cc)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms),
    )
    return cute.Launcher(torch.device("cpu"), object(), geometry)


@pytest.mark.parametrize("cc,sms,hk,hv,expected", PREFILL_DIGESTS)
def test_existing_prefill_compile_keys_and_selected_tunes_are_frozen(
    monkeypatch, cc, sms, hk, hv, expected
):
    launcher = _launcher(monkeypatch, (hk, hv), cc, sms)
    geometry = Geometry(hk, hv)
    records = []
    for tokens in (1, 64, 255, 256, 511, 512, 1024, 1025, 2047, 2048, 4095, 4096, 4097, 8192):
        for count in range(1, 257):
            key = launcher.variant_key(geometry, tokens, count, geometry.proj_width)
            records.append((key, launcher._effective_tune(hv, key[3], key)))
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == expected


@pytest.mark.parametrize("kind", ACCEPTED_DEVICES)
def test_every_geometry_admitted_without_fla(monkeypatch, kind):
    _on_device(monkeypatch, *ACCEPTED_DEVICES[kind])
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    seen = []
    monkeypatch.setattr(cute, "load", lambda device, geometry: seen.append(geometry) or object())
    monkeypatch.setattr(_fla, "load", lambda *a: pytest.fail("external provider requested"))
    for hk, hv in GRID:
        geometry = Geometry(hk, hv)
        assert _triton_decode.load(geometry, torch.device("cuda:0")).provider == "triton"
        op = _triton_prefill.load(geometry, torch.device("cuda:0"))
        assert op.provider == "cute"
        for t, n in ((0, 0), (0, 3), (1, 7), (65, 3)):
            assert op.selected_variant(t, n) == "cute"
    assert seen == GRID


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("heads", [(1, 1), (2, 4), (4, 16), (16, 16), (16, 32), (16, 48)])
def test_provider_resolution(monkeypatch, heads, available):
    _on_device(monkeypatch, *ACCEPTED_DEVICES[_triton_prefill.RTX_5090])
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    monkeypatch.setattr(cute, "load", lambda *a: object() if available else None)
    monkeypatch.setattr(_fla, "load", lambda *a: pytest.fail("fla-core bound"))
    op = inference.GatedDeltaNetPrefill(num_k_heads=heads[0], num_v_heads=heads[1], device="cuda:0")
    assert op.provider == ("cute" if available else "triton")


@pytest.mark.parametrize("kind", ACCEPTED_DEVICES)
def test_batch_invariant_pins_triton(monkeypatch, kind):
    # One provider for every T and N: our Triton prefill, never CuTe, the port or a table row.
    _on_device(monkeypatch, *ACCEPTED_DEVICES[kind])
    monkeypatch.setattr(cute, "load", lambda *a: pytest.fail("CuTe bound"))
    monkeypatch.setattr(_fla, "load", lambda *a: pytest.fail("fla-core bound"))
    op = inference.GatedDeltaNetPrefill(
        num_k_heads=16, num_v_heads=32, device="cuda:0", batch_invariant=True
    )
    assert op.provider == "triton" and not hasattr(op._kernel, "_cute")
    for t, n in ((0, 1), (16, 1), (1024, 1), (1024, 2), (8192, 4), (65536, 256)):
        assert op.selected_variant(t, n) == "triton"


@pytest.mark.parametrize("cc", [(8, 0), (8, 6)])
def test_batch_invariant_refused_below_cc8_9(monkeypatch, cc):
    _on_device(monkeypatch, "NVIDIA A100", cc)
    monkeypatch.setattr(_fla, "load", lambda *a: pytest.fail("fla-core bound"))
    with pytest.raises(ValueError, match="batch_invariant needs the Triton prefill"):
        inference.GatedDeltaNetPrefill(
            num_k_heads=16, num_v_heads=32, device="cuda:0", batch_invariant=True
        )


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_untuned_geometries_use_default_v1(monkeypatch, cc):
    for heads in GRID:
        if heads in OLD:
            continue
        launcher = _launcher(monkeypatch, heads, cc)
        assert launcher.tune == cute.DEFAULT_TUNE
        assert launcher.tune["aq_qk"] == 1
        assert launcher.row_tunes == ()
        for t, n in ((1, 1), (512, 2), (8192, 1)):
            tune, _ = launcher._row_tune(t, n)
            assert tune == cute.DEFAULT_TUNE and not tune.get("v2", 0)
        # Grouped prepare reuses shared storage per R iteration, including R=8.
        assert cute.resolve_tune(launcher.tune, heads[1])["prep_mode"] == int(heads[1] <= 32)
        assert cute.read_tune("aq_out=1", cc, heads)["bv"] == (32 if cc == (12, 1) else 64)
        assert cute.read_tune("aq_out=1,bv=64", cc, heads)["bv"] == 64
        for vsplit in (1, 2, 4):
            with pytest.raises(ValueError, match="precision-qualified only"):
                cute.read_tune(f"v2=1,vsplit={vsplit}", cc, heads)
            assert cute.v2_smem_layout(*heads, vsplit)["bytes"] <= 98816


def test_chunk_capacity_bounds_empty_and_nonempty_partitions():
    for n in range(6):
        for lengths in product((0, 1, 2, 63, 64, 65, 129), repeat=n):
            assert sum((length + 63) // 64 for length in lengths) <= cute.chunk_capacity(
                sum(lengths), n
            )
    assert cute.chunk_capacity(0, 100) == 0
    assert cute.chunk_capacity(1, 100) == 1
    for t, n in ((0, 0), (0, 3), (1, 4)):
        assert (
            check_cu_seqlens(
                torch.device("cpu"), torch.zeros(n + 1, dtype=torch.int32), t, allow_empty=True
            )
            == n
        )
    with pytest.raises(ValueError):
        check_cu_seqlens(torch.device("cpu"), torch.zeros(5, dtype=torch.int32), 1)


@pytest.mark.parametrize("tokens,count", [(64 * 70000, 1), (70000, 70000)])
def test_v1_folded_grid_limits(tokens, count):
    tune = cute.read_tune("overlap=0")
    cute.validate_launch(1, 8, tokens, count, tune)
    info = {r["kernel"]: r["grid"] for r in cute.launch_info(1, 8, tokens, count, tune)}
    chunks = cute.chunk_capacity(tokens, count)
    assert chunks == 70000
    assert info["prepare_key"] == [chunks, 1, 1]
    assert info["prepare_value"] == [8 * chunks, 1, 1]
    assert info["recurrence"] == [2 * 8 * count, 1, 1]
    with pytest.raises(ValueError, match="grid.x"):
        cute.validate_launch(1, 8, 1, 2**31 // 16, tune)


@pytest.mark.parametrize("heads", [(1, 8), (4, 12)])
@pytest.mark.parametrize("source", ["env", "table"])
def test_unqualified_v2_rejected_at_construction(monkeypatch, heads, source):
    _launcher(monkeypatch, heads)
    if source == "env":
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "v2=1")
    else:
        monkeypatch.setitem(cute.ROW_TUNES, ((12, 0), heads), ((1, None, None, {"v2": 1}),))
    with pytest.raises(ValueError, match=r"precision-qualified only.*1.9e-3.*1.2e-3"):
        cute.Launcher(torch.device("cpu"), object(), heads)


@pytest.mark.parametrize("sequences", [0, 3])
def test_zero_tokens_validate_then_dispatch_only_when_sequences_exist(monkeypatch, sequences):
    _on_device(monkeypatch, *ACCEPTED_DEVICES[_triton_prefill.RTX_5090])
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    calls = []

    def empty(geometry, *args, out=None, capture_rows=None):
        calls.append(args)
        return out if out is not None else args[0].new_empty((0, geometry.value_width))

    monkeypatch.setattr(cute, "load", lambda *a: empty)
    op = inference.GatedDeltaNetPrefill(num_k_heads=1, num_v_heads=8, device="cpu")
    args = {
        k: torch.empty(v.shape, dtype=v.dtype)
        for k, v in _prefill_arguments(1, 8, tokens=0, sequences=sequences, slots=3).items()
    }
    before = [args[k].clone() for k in ("state", "conv_state")]
    out = torch.empty(0, 8 * 128, dtype=torch.bfloat16)
    assert op(**args, out=out) is out
    assert op(**args).shape == out.shape
    for k, old in zip(("state", "conv_state"), before, strict=True):
        torch.testing.assert_close(args[k], old, rtol=0, atol=0, equal_nan=True)
    assert len(calls) == (2 if sequences else 0)
    for key in ("conv_weight", "A_log", "dt_bias", "norm_weight"):
        args[key].requires_grad_(True)
        with pytest.raises(ValueError, match="must not require grad"):
            op(**args)
        args[key].requires_grad_(False)


@pytest.mark.parametrize("heads", [(1, 0), (0, 1), (2, 3)])
def test_v2_layout_rejects_invalid_geometry(heads):
    with pytest.raises(ValueError):
        cute.v2_smem_layout(*heads, 1)


def test_warmup_includes_more_sequences_than_tokens(monkeypatch):
    launcher = _launcher(monkeypatch, (1, 8), sms=48)
    geometry = Geometry(1, 8)
    keys = launcher.variant_keys(geometry, 2, 10, geometry.proj_width)
    expected = {
        launcher.variant_key(geometry, t, n, geometry.proj_width, c)
        for t, n, c in product(range(1, 3), range(1, 11), range(2))
    }
    assert set(keys) == expected
    assert any(n > t for t, n in keys.values())


def test_v2_grid_limits_are_explicit():
    tune = cute.read_tune("v2=1,vsplit=4", (12, 0), (8, 16))
    count = (2**31 - 1) // 64
    cute.validate_launch(8, 16, 1, count, tune)
    with pytest.raises(ValueError, match="grid.x"):
        cute.validate_launch(8, 16, 1, count + 1, tune)
    with pytest.raises(ValueError, match="Hv <= 65535"):
        cute.validate_launch(1, 65536, 1, 1, tune)


@pytest.mark.parametrize("tokens", [0, 2])
@pytest.mark.parametrize("overlap", [0, 1])
def test_actual_host_launch_grids_and_scratch_for_t_less_than_n(monkeypatch, tokens, overlap):
    # Execute the real launch method after stripping DSL decorators/annotations;
    # record launches instead of compiling anything or initializing CUDA.
    import ast
    import copy
    from pathlib import Path

    path = Path(cute.__file__).with_name("_cute_prefill_kernels.py")
    tree = ast.parse(path.read_text())
    cls = copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Prefill")
    )
    cls.body = [m for m in cls.body if m.name in ("__init__", "__call__")]
    for method in cls.body:
        method.decorator_list = []
        for arg in method.args.args:
            arg.annotation = None
    namespace = {"const_expr": bool, "Int32": int, "cutlass": SimpleNamespace(min=min)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    geometry = Geometry(1, 8)
    launcher = _launcher(monkeypatch, (1, 8))
    launcher.tune = cute.read_tune(f"overlap={overlap},overlap_min_tokens=0,o_split=1,bv=32")
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    args = {
        k: torch.empty(v.shape, dtype=v.dtype)
        for k, v in _prefill_arguments(1, 8, tokens=tokens, sequences=6, slots=3).items()
    }
    key = launcher.compile_key(geometry, args["projected"], 6)
    recorded = []
    launcher.compiled[key] = lambda *a: recorded.append(a)
    launcher(geometry, *args.values())
    tensors = recorded[0]
    # q, k, w, u, aq, g use capacity 2, not N=6 nor the old negative-tail formula.
    assert [t.numel() for t in tensors[10:16]] == [
        tokens * 8192 * 2,
        tokens * 8192,
        tokens * 8 * 8192,
        tokens * 8 * 8192,
        tokens * 8 * 4096,
        tokens * 8 * 64,
    ]
    launches = []
    kernel = namespace["Prefill"](1, 8, geometry.proj_width, cute.resolve_tune(launcher.tune, 8))
    for name in ("prepare_key", "prepare_value", "recurrence", "output", "norm"):

        def record(*a, name=name):
            return SimpleNamespace(launch=lambda **kwargs: launches.append((name, kwargs["grid"])))

        setattr(kernel, name, record)
    mock_tensors = [SimpleNamespace(iterator=t, shape=t.shape) for t in tensors]
    kernel(*mock_tensors, object())
    assert launches == [
        (r["kernel"], tuple(r["grid"])) for r in cute.launch_info(1, 8, tokens, 6, launcher.tune)
    ]


@pytest.mark.parametrize("operation,provider", [("Decode", "fla-core"), ("Prefill", "fla-core")])
@pytest.mark.parametrize("checks", [False, True])
def test_construction_reroutes_have_reasons(monkeypatch, operation, provider, checks, caplog):
    from ctikernels import _reroute

    monkeypatch.setenv("CTI_ENGINE_CHECKS", str(int(checks)))
    monkeypatch.setattr(_reroute, "_WARNED", set())
    monkeypatch.setattr(_triton_decode, "load", lambda *a, **k: None)
    monkeypatch.setattr(_triton_prefill, "load", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    monkeypatch.setattr(
        _fla, "load", lambda *a: SimpleNamespace(decode=lambda *a: None, prefill=lambda *a: None)
    )
    cls = getattr(inference, "GatedDeltaNet" + operation)
    if checks:
        with pytest.raises(_reroute.RerouteError, match="CTI_ENGINE_CHECKS"):
            cls(num_k_heads=16, num_v_heads=32, device="cuda:0")
    else:
        for _ in range(2):
            op = cls(num_k_heads=16, num_v_heads=32, device="cuda:0")
            assert op.provider == provider
            assert isinstance(op.provider, _reroute.Provider) and op.provider.reason
        assert len(caplog.records) == 1


@pytest.mark.parametrize("heads", [(8, 16), (16, 48)])
@pytest.mark.parametrize("tune", ["v2=0", "v2=1"])
def test_empty_calls_share_nonempty_compile_key(monkeypatch, heads, tune):
    launcher = _launcher(monkeypatch, heads)
    launcher.tune = cute.read_tune(tune, (12, 0), heads)
    launcher.tune_key = tuple(sorted(launcher.tune.items()))
    geometry = Geometry(*heads)
    assert launcher.variant_key(geometry, 0, 3, geometry.proj_width) == launcher.variant_key(
        geometry, 1, 3, geometry.proj_width
    )
