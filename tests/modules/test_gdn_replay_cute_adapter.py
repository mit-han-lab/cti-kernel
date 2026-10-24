# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Exercise launcher binding/capture refusal without importing the GPU DSL."""

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import Config, _phase_launch
from ctikernels.gdn.replay.cute._phase_launch import (
    phase_launch,
    phase_tile,
    run,
    vector_publication,
)

from ._replay_case import case, projected

PHASES = ("verify", "commit", "reject", "flush", "bridge_step", "export_snapshot")


@pytest.mark.parametrize("sms", [48, 170, 188])
@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("batch", [0, 1, 2, 3, 4])
def test_phase_tile(batch, hv, sms):
    tile = phase_tile(batch, hv, sms)
    assert tile in (32, 64, 128)
    if tile < 128:
        assert batch * hv * (128 // tile) <= sms
    if tile > 32:
        assert batch * hv * (256 // tile) > sms


@pytest.mark.parametrize("sms,expected", [(127, 128), (128, 64), (255, 64), (256, 32)])
def test_phase_tile_wave_boundary(sms, expected):
    assert phase_tile(2, 32, sms) == expected


@pytest.mark.parametrize(
    "batch,hv,sms,q,split,expected",
    [
        # One wave of small tiles: 8 warps unless Q needs a second staging chunk.
        (1, 32, 170, 8, True, (32, 8, 1)),
        (1, 32, 170, 9, True, (32, 16, 1)),
        (2, 32, 170, 16, True, (64, 16, 1)),
        # Full-head CTAs within one wave: 16 warps share the prologue.
        (3, 32, 170, 4, True, (128, 16, 1)),
        (3, 48, 170, 4, True, (128, 16, 1)),
        (4, 32, 188, 16, True, (128, 16, 1)),
        # More head CTAs than SMs: 64-row tiles, two CTAs per SM.
        (4, 48, 170, 4, True, (64, 8, 2)),
        (4, 48, 188, 16, True, (64, 8, 2)),
        # GB10 (DRAM-bound) keeps full-head 16-warp CTAs.
        (4, 48, 48, 4, False, (128, 16, 1)),
        (1, 32, 48, 4, False, (128, 16, 1)),
    ],
)
def test_phase_launch_policy(batch, hv, sms, q, split, expected):
    assert phase_launch(batch, hv, sms, q, Config(split_oversubscribed=split)) == expected


def test_phase_launch_config_selects_per_device():
    from ctikernels.gdn.replay.cute._config import select

    assert select("NVIDIA GB10").split_oversubscribed is False
    for name in ("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell Server Edition"):
        assert select(name).split_oversubscribed is True
    with pytest.raises(ValueError):
        Config(min_blocks=4)
    with pytest.raises(ValueError):
        Config(phase_warps=32)


@pytest.mark.parametrize("sms,automatic", [(48, 32), (16, 64), (8, 128)])
def test_phase_tile_launcher_cache_separates_overrides(sms, automatic, monkeypatch):
    from types import SimpleNamespace

    op, state, _, weights, slots, _, status = case(batch=2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sms, name="test device"),
    )
    compiled, launched = [], []

    def compile_stub(*args, **kwargs):
        tile = args[-5]
        compiled.append(tile)
        return lambda *buffers: launched.append(tile)

    monkeypatch.setattr(_phase_launch, "compile_phases", compile_stub)
    for tile in (128, 64, 32, None, 128, 64, 32):
        monkeypatch.setattr(_phase_launch, "PHASE_TILE", tile)
        run(op, "flush", state=state, status=status, weights=weights, slot_indices=slots)
    assert compiled == [128, 64, 32]
    assert launched == [128, 64, 32, automatic, 128, 64, 32]
    assert {key[9] for key in op._cute_launchers} == {32, 64, 128}


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_cute_phase_dispatch_and_workspace(checks):
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=Config(),
        checks=checks,
    )
    for phase in ("step", *PHASES):
        assert op.provider[phase] == "cute"
    for phase in ("reset", "initialize", "handoff", "import_snapshot", "release"):
        assert op.provider[phase] == "triton"
    for phase in PHASES:
        assert not set(op.workspace_spec(phase, batch_size=2)) - {
            "pending.raw",
            "pending.metadata",
            "out",
            *(
                ("arrivals",)
                if checks == "trusted"
                and phase in ("commit", "flush", "bridge_step", "export_snapshot")
                else ()
            ),
        }


@pytest.mark.parametrize("mode", PHASES)
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("tile", [32, 64, 128])
def test_cached_phase_launcher_borrows_typed_buffers(mode, capture, tile, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", tile)
    op, state, pending, weights, slots, positions, status = case(batch=2)
    q = op.verify_length
    kw = {"slot_indices": slots}
    use_pending = mode in ("verify", "commit", "reject") or (mode == "export_snapshot" and capture)
    if use_pending:
        kw["pending"] = pending
    stride = 0
    if mode in ("verify", "bridge_step"):
        x = projected(op, 2, length=q if mode == "verify" else None)
        # Include a padded batch stride, with dense tokens inside each row.
        backing = torch.zeros(2, x[0].numel() + 16, dtype=torch.bfloat16)
        x = backing[:, : x[0].numel()].view(x.shape)
        kw.update(projected=x, positions=positions, out=torch.empty_like(x))
        stride = x.stride(0)
    if mode == "commit" or mode == "export_snapshot":
        kw["accepted_lengths"] = torch.zeros(2, dtype=torch.int32) if use_pending else None
    if mode == "export_snapshot":
        kw["pairs"] = torch.stack((slots, slots + state.metadata.shape[0]), 1)
        kw["offsets"] = torch.zeros(2, dtype=torch.int64) if use_pending else None
        kw["snapshot_valid"] = torch.zeros(2, dtype=torch.bool)
    if mode == "commit":
        # Public commit passes all three optional keys even when absent.
        kw["snapshot_rows"] = slots if capture else None
        kw["snapshot_offsets"] = positions if capture else None
        kw["snapshot_valid"] = torch.zeros(2, dtype=torch.bool) if capture else None
    key = (
        mode,
        2,
        stride,
        state.metadata.shape[0],
        state.snapshot_metadata.shape[0],
        2 if use_pending else 0,
        mode == "commit" and capture,
        op.cute_config,
        vector_publication(op, kw.get("pending")),
        tile,
        op.checks,
        _phase_launch.PDL,
        _phase_launch.FUSED_PUBLISH,
        False,
    )
    calls = []
    types = (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.int32,
        torch.int64,
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.bfloat16,
        torch.uint8,
        torch.int64,
        torch.int32,
        torch.int32,
    )
    from ctikernels.gdn.replay._layout import layout

    key += (tuple(enumerate(types)), layout(op.geometry, state, kw))
    op._cute_launchers[key] = lambda *args: calls.append(args)
    run(op, mode, state=state, status=status, weights=None if mode == "reject" else weights, **kw)
    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 24 and all(isinstance(t, torch.Tensor) and t.ndim == 1 for t in args)
    assert args[7].data_ptr() == state.checkpoint.data_ptr()
    assert args[14].dtype == torch.int64
    assert args[15].dtype == torch.int32
    assert args[18].dtype == torch.bool
    assert args[20].data_ptr() == status.record.data_ptr()
    if stride:
        assert args[0].data_ptr() == kw["projected"].data_ptr()
        assert args[0].numel() == stride + kw["projected"][0].numel()


def test_first_phase_compilation_is_refused_during_capture(monkeypatch):
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", 128)
    op, state, _, weights, slots, _, status = case(batch=2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm CuTe specialization before capture"):
        run(op, "flush", state=state, status=status, weights=weights, slot_indices=slots)


@pytest.mark.parametrize("qualified", [False, True])
def test_automatic_selection_keeps_portable_step(qualified, monkeypatch):
    from ctikernels.gdn.replay import _api, cute

    monkeypatch.setattr(_api, "_cute_qualified", lambda device: qualified)
    monkeypatch.setattr(cute, "select", lambda name, **kwargs: Config())
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 5090"
    )
    op = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cuda:0")
    assert op.provider["step"] == "triton"
    for phase in PHASES:
        assert op.provider[phase] == ("cute" if qualified else "triton")
    # The portable step borrows no CuTe scratch, in sizing or in validation.
    assert not {"arrivals", "controls", "sections"} & set(op.workspace_spec("step", batch_size=2))
    assert op.capabilities["step_scratch_fixed_bytes"] == 0


@pytest.mark.parametrize(
    "provider,expected", [("cute", "trusted"), ("triton", "trusted"), ("torch_reference", "strict")]
)
@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_checks_capabilities(provider, expected, checks):
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cpu" if provider == "torch_reference" else "cuda:0",
        provider=provider,
        cute_config=Config() if provider == "cute" else None,
        checks=checks,
    )
    assert op.checks == op.capabilities["checks"] == ("strict" if checks == "strict" else expected)


def test_checks_default_and_invalid():
    op = GatedDeltaNetReplay(
        num_k_heads=1, num_v_heads=4, device="cuda:0", provider="cute", cute_config=Config()
    )
    assert op.checks == "trusted"
    with pytest.raises(ValueError, match="checks"):
        GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cpu", checks="unknown")


@pytest.mark.parametrize("mode", PHASES)
@pytest.mark.parametrize("status_value", [None, object()])
def test_trusted_status_is_not_inspected(mode, status_value, monkeypatch):
    from types import SimpleNamespace

    op, state, pending, weights, slots, positions, _ = case(batch=2)
    # Exercise real CPU tensor validation with a stub CuTe backend; no CUDA imports.
    op.checks = "trusted"
    op.provider = dict(op.provider, **{mode: "cute"})
    calls = []
    op._backends["cute"] = SimpleNamespace(run=lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(op, "_check_status", lambda _: pytest.fail("trusted checked status"))
    kw = {"state": state, "status": status_value}
    scratch = op.workspace_spec(mode, batch_size=2)
    if "arrivals" in scratch:
        kw["workspace"] = {"arrivals": torch.zeros(2, dtype=torch.int32)}
    if mode == "export_snapshot":
        kw.update(
            pairs=torch.stack((slots, slots + state.metadata.shape[0]), 1),
            snapshot_valid=torch.zeros(2, dtype=torch.bool),
        )
    else:
        kw["slot_indices"] = slots
    if mode in ("verify", "commit", "reject"):
        kw["pending"] = pending
    if mode == "commit":
        kw["accepted_lengths"] = torch.zeros(2, dtype=torch.int32)
    if mode in ("verify", "bridge_step"):
        q = op.verify_length
        x = projected(op, 2, length=q if mode == "verify" else None)
        kw.update(
            positions=positions[:, None] + torch.arange(q) if mode == "verify" else positions,
            out=torch.empty(
                (2, q, op.geometry.value_width)
                if mode == "verify"
                else (2, op.geometry.value_width),
                dtype=torch.bfloat16,
            ),
        )
        getattr(op, mode)(x, *weights, **kw)
    elif mode == "reject":
        op.reject(**kw)
    else:
        getattr(op, mode)(*weights, **kw)
    assert len(calls) == 1
    if "arrivals" in scratch:
        assert calls[0]["arrivals"] is kw["workspace"]["arrivals"]
    assert not {"controls", "sections"} & scratch.keys()


def test_checks_and_pdl_launcher_cache_and_status_placeholders(monkeypatch):
    from types import SimpleNamespace

    op, state, _, weights, slots, _, status = case(batch=2)
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", 32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=170, name="test device"),
    )
    compiled, calls = [], []

    def compile_stub(*args, **kwargs):
        compiled.append(args[-4:-2])
        return lambda *buffers: calls.append(buffers)

    monkeypatch.setattr(_phase_launch, "compile_phases", compile_stub)
    for _ in range(2):
        for checks in ("trusted", "strict"):
            op.checks = checks
            for pdl in (True, False):
                monkeypatch.setattr(_phase_launch, "PDL", pdl)
                run(
                    op,
                    "flush",
                    state=state,
                    status=None if checks == "trusted" else status,
                    weights=weights,
                    slot_indices=slots,
                )
    assert compiled == [("trusted", True), ("trusted", False), ("strict", True), ("strict", False)]
    for buffers in calls[:2]:
        assert len(buffers) == 24
        assert [b.dtype for b in buffers[18:]] == [
            torch.bool,
            torch.bfloat16,
            torch.uint8,
            torch.int64,
            torch.int32,
            torch.int32,
        ]
        assert all(b.ndim == 1 for b in buffers)
        assert buffers[20].data_ptr() == state.ring.data_ptr()
    assert calls[2][20].data_ptr() == status.record.data_ptr()


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_portable_still_requires_status(checks):
    op, state, _, weights, slots, _, _ = case(batch=1)
    # Requesting trusted on CPU records the effective strict mode.
    op = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cpu", window=4, checks=checks)
    with pytest.raises(TypeError, match="status must be a bound ReplayStatus"):
        op.flush(*weights, slot_indices=slots, state=state)


def test_trusted_warm_signature_ignores_status(monkeypatch):
    from types import SimpleNamespace

    op, state, _, weights, slots, _, status = case(batch=1)
    op.checks = "trusted"
    op.provider = dict(op.provider, flush="cute")
    op._backends["cute"] = SimpleNamespace(run=lambda *args, **kw: None)
    op._warm.clear()
    for supplied in (None, status, object()):
        op.flush(
            *weights,
            slot_indices=slots,
            state=state,
            status=supplied,
            workspace={"arrivals": torch.zeros(1, dtype=torch.int32)},
        )
    assert len(op._warm) == 1
    op.checks = "strict"
    op.flush(*weights, slot_indices=slots, state=state, status=status)
    assert len(op._warm) == 2


@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_fused_k2_arrivals_specs(batch, checks):
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=48,
        device="cuda:0",
        provider="cute",
        cute_config=Config(),
        checks=checks,
    )
    for mode in PHASES:
        specs = op.workspace_spec(mode, batch_size=batch)
        needed = checks == "trusted" and mode in (
            "commit",
            "flush",
            "bridge_step",
            "export_snapshot",
        )
        assert ("arrivals" in specs) == needed
        if needed:
            spec = specs["arrivals"]
            assert spec.shape == (max(1, batch),)
            assert spec.dtype == torch.int32 and spec.initial_value == 0
            assert spec.payload_bytes == 4 * max(1, batch)
            assert spec.offset % 256 == 0
    if batch == 0:
        # Graph controls retain their existing positive-batch contract.
        with pytest.raises(ValueError, match="positive integers"):
            op.graph_data_spec(batch_size=batch)
        return
    graph = op.graph_data_spec(batch_size=batch)
    assert ("phase_arrivals" in graph) == (checks == "trusted")
    if checks == "trusted":
        spec = graph["phase_arrivals"]
        assert spec.shape == (2, max(1, batch))
        assert spec.initial_value == 0 and spec.dtype == torch.int32
        assert spec.payload_bytes == 2 * 4 * max(1, batch)
        # Decode scratch retains its independent contract.
        assert graph["arrivals"].shape == (max(1, batch),)


@pytest.mark.parametrize("bad", ["missing", "dtype", "shape", "strided", "alias"])
def test_fused_k2_arrivals_validation(bad):
    op, state, _, weights, slots, _, _ = case(batch=3)
    op.checks = "trusted"
    op.provider = dict(op.provider, flush="cute")
    arrivals = torch.zeros(3, dtype=torch.int32)
    if bad == "missing":
        workspace = None
    else:
        if bad == "dtype":
            arrivals = arrivals.to(torch.int64)
        elif bad == "shape":
            arrivals = arrivals[:1]
        elif bad == "strided":
            arrivals = torch.zeros(6, dtype=torch.int32)[::2]
        elif bad == "alias":
            arrivals = slots
        workspace = {"arrivals": arrivals}
    with pytest.raises((ValueError, TypeError), match="arrivals|overlapping"):
        op.flush(*weights, slot_indices=slots, state=state, workspace=workspace)


def test_fused_k2_cache_keys_and_borrowed_arrivals(monkeypatch):
    from types import SimpleNamespace

    op, state, _, weights, slots, _, status = case(batch=3)
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", 64)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=170, name="test device"),
    )
    compiled, calls = [], []

    def compile_stub(*args, **kwargs):
        compiled.append(args[-4:])
        return lambda *buffers: calls.append(buffers)

    monkeypatch.setattr(_phase_launch, "compile_phases", compile_stub)
    arrivals = torch.zeros(3, dtype=torch.int32)
    for _ in range(2):
        for checks in ("trusted", "strict"):
            op.checks = checks
            for fused in (True, False):
                monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", fused)
                run(
                    op,
                    "flush",
                    state=state,
                    status=status,
                    weights=weights,
                    slot_indices=slots,
                    arrivals=arrivals,
                )
    assert compiled == [
        (c, _phase_launch.PDL, f, False) for c in ("trusted", "strict") for f in (True, False)
    ]
    assert len(calls) == 8
    assert all(args[-1].data_ptr() == arrivals.data_ptr() for args in calls)
    assert {key[12] for key in op._cute_launchers} == {True, False}


def test_fused_k2_compile_cache_and_arrivals_descriptor(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    cutlass = ModuleType("cutlass")
    cute = ModuleType("cutlass.cute")
    runtime = ModuleType("cutlass.runtime")
    phases = ModuleType("ctikernels.gdn.replay.cute._phases")
    for name in ("BFloat16", "Boolean", "Float32", "Int32", "Int64", "Uint8"):
        setattr(cutlass, name, name)
    cutlass.cute = cute
    cute.sym_int64 = lambda: "dynamic"
    cute.runtime = SimpleNamespace(
        make_fake_tensor=lambda dtype, shape, stride, *, assumed_align: (dtype, assumed_align)
    )
    runtime.make_fake_stream = lambda **kw: "stream"
    phases.Phases = lambda *args: args
    compiled = []

    def compile_stub(phase, *args, **kw):
        compiled.append((phase, args))
        return object()

    cute.compile = compile_stub
    for module in (cutlass, cute, runtime, phases):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(_phase_launch, "version", lambda name: "4.8.0")
    # The product launch plan imports Triton; this test covers only the cache key
    # and the arrivals descriptor, so a fixed plan keeps it on the CPU CI env.
    monkeypatch.setattr(
        sys.modules["ctikernels.gdn.replay.cute"],
        "arithmetic_plan",
        lambda *args, **kw: (64, 8, False, False, False, False, False),
    )
    _phase_launch.compile_phases.cache_clear()
    try:
        results = {}
        for _ in range(2):
            for checks in ("strict", "trusted"):
                for fused in (False, True):
                    result = _phase_launch.compile_phases(
                        16,
                        32,
                        8,
                        9,
                        3,
                        0,
                        8,
                        4,
                        3,
                        True,
                        "commit",
                        Config(),
                        "120",
                        170,
                        tile=64,
                        checks=checks,
                        fused=fused,
                    )
                    assert result is results.setdefault((checks, fused), result)
        assert len(compiled) == 4
        assert len({id(result) for result in results.values()}) == 4
        assert [phase[-5:-1] for phase, _ in compiled] == [
            (c == "trusted", True, f, False) for c in ("strict", "trusted") for f in (False, True)
        ]
        for _ in range(2):
            delayed = _phase_launch.compile_phases(
                16,
                32,
                8,
                9,
                3,
                0,
                8,
                4,
                3,
                True,
                "commit",
                Config(),
                "120",
                170,
                tile=64,
                checks="trusted",
                fused=True,
                delayed=True,
            )
        assert len(compiled) == 5
        assert compiled[-1][0][-5:-1] == (True, True, True, True)
        assert all(delayed is not result for result in results.values())
        assert all(args[-2] == ("Int32", 4) for _, args in compiled)
    finally:
        _phase_launch.compile_phases.cache_clear()


@pytest.mark.parametrize("mode", [None, "eager", "delayed", "invalid"])
@pytest.mark.parametrize("provider", ["cute", "triton", "torch_reference"])
@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_commit_modes_and_flush_capture(mode, provider, checks, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unmeasured device")
    args = {
        "num_k_heads": 1,
        "num_v_heads": 4,
        "device": "cpu" if provider == "torch_reference" else "cuda:0",
        "provider": provider,
        "checks": checks,
        "commit_mode": mode,
        "cute_config": Config() if provider == "cute" else None,
    }
    supports = provider == "cute" and checks == "trusted"
    if mode == "invalid" or (mode == "delayed" and not supports):
        with pytest.raises(ValueError, match="commit"):
            GatedDeltaNetReplay(**args)
        return
    op = GatedDeltaNetReplay(**args)
    assert op.capabilities["commit_modes"] == (("eager", "delayed") if supports else ("eager",))
    assert ("flush" in op.capabilities["capture"]) == supports
    for batch in (0, 1, 3, 4, 12):
        assert op.commit_mode_for(batch) == (mode or "eager")
    monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", False)
    assert op.commit_mode_for(3) == "eager"
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            op.commit_mode_for(invalid)


def test_commit_policy_lookup_and_pinned_batch(monkeypatch):
    from ctikernels.gdn.replay.cute import _config

    for name in (*_config.COMMIT_POLICY, "unknown"):
        # Q=12 at window 8 and every B >= 5 or window-16 key are not delayed anywhere.
        assert _config.commit_mode(name, 3, 12, 8) == "eager"
        assert _config.commit_mode(name, 5, 4, 8) == "eager"
        assert _config.commit_mode(name, 1, 4, 16) == "eager"
    for name in _config.COMMIT_POLICY:
        # Measured: small Q whose accepted rows fit the window wins on every device.
        assert _config.commit_mode(name, 4, 4, 4) == "delayed"
        assert _config.commit_mode(name, 2, 5, 8) == "delayed"
    monkeypatch.setattr(
        _config,
        "COMMIT_POLICY",
        {
            "test": {(3, 8, 16): "delayed", (5, 8, 16): "delayed"},
            "callable": lambda b, q, w: "delayed" if b == 1 and q < w else "eager",
        },
    )
    assert _config.commit_mode("test suffix", 3, 8, 16) == "delayed"
    assert _config.commit_mode("test", 100, 8, 16) == "delayed"
    assert _config.commit_mode("test", 3, 9, 16) == "eager"
    assert _config.commit_mode("callable", 1, 4, 8) == "delayed"
    assert _config.commit_mode("callable", 2, 4, 8) == "eager"
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test suffix")
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        window=16,
        verify_length=8,
        cute_config=Config(),
        commit_mode="auto",
    )
    assert op.commit_mode_for(3) == "delayed"
    _config.COMMIT_POLICY["test"].clear()
    assert op.commit_mode_for(3) == "delayed"
    assert op.commit_mode_for(4) == "eager"


def test_commit_policy_geometry_rows_replace_the_shared_table(monkeypatch):
    from ctikernels.gdn.replay.cute import _config

    monkeypatch.setattr(_config, "COMMIT_POLICY", {"test": {(4, 4, 4): "delayed"}})
    monkeypatch.setattr(
        _config, "COMMIT_POLICY_GEOMETRY", {("test", (8, 16)): {(3, 5, 8): "delayed"}}
    )
    # Other geometries (and calls without one) keep the shared table.
    assert _config.commit_mode("test", 4, 4, 4) == "delayed"
    assert _config.commit_mode("test", 4, 4, 4, geometry=(16, 32)) == "delayed"
    assert _config.commit_mode("test", 3, 5, 8, geometry=(16, 32)) == "eager"
    # A geometry row replaces the shared table for that geometry only.
    assert _config.commit_mode("test suffix", 4, 4, 4, geometry=(8, 16)) == "eager"
    assert _config.commit_mode("test", 3, 5, 8, geometry=(8, 16)) == "delayed"
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test suffix")
    ops = {
        geometry: GatedDeltaNetReplay(
            num_k_heads=geometry[0],
            num_v_heads=geometry[1],
            device="cuda:0",
            provider="cute",
            window=window,
            verify_length=q,
            cute_config=Config(),
            commit_mode="auto",
        )
        for geometry, window, q in (((8, 16), 8, 5), ((16, 32), 4, 4))
    }
    assert ops[(8, 16)].commit_mode_for(3) == "delayed"
    assert ops[(8, 16)].commit_mode_for(4) == "eager"
    assert ops[(16, 32)].commit_mode_for(4) == "delayed"


def test_delayed_launcher_cache_and_eager_overrides(monkeypatch):
    op, state, pending, weights, slots, _, status = case(batch=3)
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", 64)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            multi_processor_count=170,
            name="test device",
        ),
    )
    compiled = []
    monkeypatch.setattr(
        _phase_launch,
        "compile_phases",
        lambda *args, **kwargs: compiled.append(args[-1]) or (lambda *buffers: None),
    )
    monkeypatch.setattr(op, "commit_mode_for", lambda batch: "delayed")
    for checks, fused, expected in (
        ("trusted", True, True),
        ("trusted", False, False),
        ("strict", True, False),
    ):
        op.checks = checks
        monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", fused)
        for _ in range(2):
            run(
                op,
                "commit",
                state=state,
                status=status,
                weights=weights,
                slot_indices=slots,
                pending=pending,
                accepted_lengths=torch.zeros(3, dtype=torch.int32),
            )
        assert compiled[-1] is expected
    assert compiled == [True, False, False]


def test_commit_mode_is_part_of_warm_signature(monkeypatch):
    from types import SimpleNamespace

    op, state, pending, weights, slots, _, _ = case(batch=2)
    op.checks = "trusted"
    op.provider = dict(op.provider, commit="cute")
    op._backends["cute"] = SimpleNamespace(run=lambda *args, **kwargs: None)
    for mode in ("eager", "delayed", "eager"):
        monkeypatch.setattr(op, "commit_mode_for", lambda batch, selected=mode: selected)
        op.commit(
            *weights,
            slot_indices=slots,
            state=state,
            pending=pending,
            accepted_lengths=torch.zeros(2, dtype=torch.int32),
            workspace={"arrivals": torch.zeros(2, dtype=torch.int32)},
        )
    signatures = [s for s in op._warm if s[0] == "commit"]
    assert len(signatures) == 2
    assert {s[2] for s in signatures} == {"eager", "delayed"}


@pytest.mark.parametrize(
    "name,qualified",
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", True),
        ("NVIDIA GB10", True),
        ("NVIDIA H100 80GB HBM3", False),
    ],
)
def test_cute_default_devices(name, qualified, monkeypatch):
    import importlib.metadata

    from ctikernels.gdn.replay import _api

    monkeypatch.setattr(importlib.metadata, "version", lambda package: "4.8.0")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    assert _api._cute_qualified("cuda:0") is qualified


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("checks", ["strict", "trusted"])
@pytest.mark.parametrize(
    "name", [None, "NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000", "NVIDIA GB10", "unknown"]
)
def test_product_strict_route_and_effective_config(monkeypatch, batch, checks, name):
    from ctikernels.gdn.replay.cute import effective_config

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda:0",
        provider="cute",
        checks=checks,
        cute_config=Config(provider="product", immediate=True) if name is None else None,
    )
    config = effective_config(op, batch)
    if name in ("NVIDIA GB10", "unknown"):
        assert op.decode_route(batch) == "delayed"
    else:
        assert op.decode_route(batch) == ("immediate" if checks == "strict" else "product")
        assert config.immediate
        assert set(op.workspace_spec("step", batch_size=batch)) == (
            {"out", "arrivals"} if checks == "strict" else {"out"}
        )
    assert config.provider == ("product" if op.decode_route(batch) == "product" else "cute")
    assert config.trusted == (checks == "trusted")


def test_cute_strict_phase_dispatch_and_zero_extra_workspace():
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=Config(),
        checks="strict",
    )
    for phase in ("step", *PHASES):
        assert op.provider[phase] == "cute"
    for phase in ("reset", "initialize", "handoff", "import_snapshot", "release"):
        assert op.provider[phase] == "triton"
    for phase in PHASES:
        assert not set(op.workspace_spec(phase, batch_size=2)) - {
            "pending.raw",
            "pending.metadata",
            "out",
        }


@pytest.mark.parametrize("tile", [32, 64, 128])
@pytest.mark.parametrize("speculative", [False, True])
def test_step_config_and_shared_budget(tile, speculative):
    config = Config(value_tile=tile, speculative_ring=speculative)
    for window in (4, 8, 16):
        assert config.shared_bytes(window) <= 99 * 1024
        assert config.stages_ring(window) == speculative
    assert not config.stages_ring(128)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"value_tile": 16}, "value_tile"),
        ({"value_tile": 256}, "value_tile"),
        ({"warps": 2}, "warps"),
        ({"lanes": 2}, "lanes"),
        ({"value_tile": 32, "lanes": 8}, "4 lanes"),
        ({"value_tile": 64, "snapshot_controls": True}, "snapshot_controls"),
        ({"speculative_ring": 1}, "speculative_ring"),
        ({"immediate": 1}, "immediate"),
        ({"provider": "unknown"}, "provider"),
        ({"provider": "product"}, "immediate"),
    ],
)
def test_invalid_step_config(options, match):
    with pytest.raises(ValueError, match=match):
        Config(**options)


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell"])
def test_batch_aware_step_dispatch(name):
    from ctikernels.gdn.replay.cute import select

    assert [select(name, b).value_tile for b in range(9)] == [
        128,
        32,
        32,
        64,
        64,
        128,
        128,
        128,
        128,
    ]
    assert all(select("NVIDIA GB10", b).value_tile == 128 for b in range(9))
    assert select("unknown", 1) == Config()
    assert select(name).immediate
    assert all(select(name, b).provider == "product" for b in range(9))
    assert select(name, 1, mode="immediate").provider == "cute"


@pytest.mark.parametrize(
    "name,routes",
    [
        ("NVIDIA GeForce RTX 5090", ["product"] * 10),
        ("NVIDIA RTX PRO 6000 Blackwell", ["product"] * 10),
        ("NVIDIA GB10", ["delayed"] * 10),
        ("unknown", ["delayed"] * 10),
    ],
)
def test_measured_decode_routes(monkeypatch, name, routes):
    from ctikernels.gdn.replay.cute import decode_route

    assert [decode_route(name, b) for b in range(10)] == routes
    assert {decode_route(name, b, mode="immediate") for b in range(10)} == {"immediate"}
    assert {decode_route(name, b, mode="delayed") for b in range(10)} == {"delayed"}
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(num_k_heads=16, num_v_heads=32, device="cuda:0", provider="cute")
    assert [op.decode_route(b) for b in range(10)] == routes
    ring = op.state_spec(num_slots=2, num_snapshot_rows=0)["ring"].payload_bytes
    assert (ring == 0) == ("delayed" not in routes)
    with pytest.raises(ValueError, match="batch_size"):
        op.decode_route(-1)


def test_step_cache_includes_effective_config_and_capture_reuses_it(monkeypatch):
    from types import SimpleNamespace

    from ctikernels.gdn.replay import cute

    op = SimpleNamespace(
        geometry=SimpleNamespace(k_heads=16, v_heads=32),
        window=8,
        device="cuda:0",
        mode="immediate",
        cute_config=Config(),
        _cute_device_name="NVIDIA GeForce RTX 5090",
        _cute_launchers={},
    )
    state = SimpleNamespace(
        metadata=torch.empty(8, 8, dtype=torch.int64),
        ring=torch.empty(8, 0, 1, dtype=torch.bfloat16),
    )
    calls = []

    def compile_step(*args, **kw):
        calls.append((args, kw))
        return object()

    monkeypatch.setattr(cute, "compile_step", compile_step)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="NVIDIA GeForce RTX 5090", multi_processor_count=170),
    )
    x = torch.empty(1, 128)
    first = cute.prepare(op, x, state, True, True)
    assert calls[-1][0][5] == Config(value_tile=32, immediate=True, speculative_ring=False)
    assert calls[-1][0][7:10] == (8, True, True)
    assert cute.prepare(op, torch.empty(4, 128), state, True, True) is not first
    assert calls[-1][0][5].value_tile == 64
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert cute.prepare(op, x, state, True, True) is first
    assert len(calls) == 2
    op._cute_device_name = None
    op.mode = "auto"
    op.cute_config = Config(value_tile=32, speculative_ring=False)
    with pytest.raises(RuntimeError, match="warm CuTe specialization"):
        cute.prepare(op, x, state, True, True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    changed = cute.prepare(op, x, state, True, True)
    assert changed is not first
    assert calls[-1][0][5] == op.cute_config
    assert len(op._cute_launchers) == 3


@pytest.mark.parametrize(
    "name,immediate",
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("NVIDIA RTX PRO 6000 Blackwell", True),
        ("unknown", False),
    ],
)
@pytest.mark.parametrize("mode", ["auto", "immediate", "delayed"])
def test_policy_sizing_and_modes(monkeypatch, name, immediate, mode):
    from ctikernels.gdn.replay.cute import effective_config

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=16, num_v_heads=32, device="cuda:0", provider="cute", mode=mode
    )
    expected = immediate if mode == "auto" else mode == "immediate"
    assert all(effective_config(op, b).immediate == expected for b in range(10))
    spec = op.state_spec(num_slots=3, num_snapshot_rows=2)
    assert (spec["ring"].payload_bytes == 0) == expected
    assert spec["conv_history"].shape[1] == 3
    assert spec["checkpoint"].dtype == torch.float32


def test_mixed_policy_retains_ring_and_product_routing(monkeypatch):
    from ctikernels.gdn.replay.cute._config import POLICY, all_immediate, select

    name = "NVIDIA GeForce RTX 5090"
    monkeypatch.setitem(POLICY, (name, None, 4), {"immediate": False})
    assert not all_immediate(name)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(num_k_heads=16, num_v_heads=32, device="cuda:0", provider="cute")
    assert op.state_spec(num_slots=1, num_snapshot_rows=0)["ring"].payload_bytes > 0
    monkeypatch.setitem(POLICY, (name, None, 4), {"provider": "product", "immediate": True})
    assert select(name, 3).provider == "product"
    assert select(name, 3, config=Config()).provider == "cute"
    assert not select(name, 3, config=Config()).immediate


@pytest.mark.parametrize("tile,bytes_", [(32, 6528), (64, 7360), (128, 9024)])
def test_immediate_scratch_is_window_independent(tile, bytes_):
    config = Config(immediate=True, value_tile=tile)
    assert all(config.shared_bytes(w) == bytes_ for w in (4, 8, 16))
    assert not config.stages_ring(4)


def test_explicit_mode_config_and_strict_tier():
    args = {"num_k_heads": 16, "num_v_heads": 32, "device": "cuda:0", "provider": "cute"}
    config = Config(value_tile=64, immediate=True)
    op = GatedDeltaNetReplay(**args, cute_config=config, precision="strict")
    assert op.cute_config == config
    assert op.mode == "immediate"
    with pytest.raises(ValueError, match="strict"):
        GatedDeltaNetReplay(**args, cute_config=config, precision="strict", mode="delayed")
    with pytest.raises(ValueError, match="strict"):
        GatedDeltaNetReplay(**args, cute_config=Config(), precision="strict")
    op = GatedDeltaNetReplay(**args, cute_config=config, mode="delayed")
    assert op.cute_config.value_tile == 64 and not op.cute_config.immediate


def test_strict_auto_policy_forces_immediate_on_delayed_device(monkeypatch):
    from ctikernels.gdn.replay.cute import effective_config

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GB10")
    op = GatedDeltaNetReplay(
        num_k_heads=16, num_v_heads=32, device="cuda:0", provider="cute", precision="strict"
    )
    assert all(effective_config(op, b).immediate for b in range(10))
    assert op.state_spec(num_slots=1, num_snapshot_rows=0)["ring"].payload_bytes == 0
    assert Config(32).value_tile == 32  # Existing positional override remains valid.


def test_ring_empty_adapter_warms_and_reuses_without_host_reads(monkeypatch):
    from ctikernels.gdn.replay import cute

    op, state, _, _, slots, _, _ = case(batch=2)
    op.cute_config = Config()
    flag = torch.empty(1, dtype=torch.bool)
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        cute, "compile_ring_empty", lambda *shape, **kw: lambda *args: calls.append((shape, args))
    )
    op.ring_empty(state, slots, flag)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op.ring_empty(state, slots, flag)
    assert len(calls) == 2
    assert calls[0][0] == (2, state.metadata.shape[0], "120", torch.int32)
    assert calls[0][1][0].data_ptr() == state.metadata.data_ptr()
    assert calls[0][1][2] is flag
    with pytest.raises(RuntimeError, match="warm CuTe"):
        op.ring_empty(state, slots[:1], flag)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"trusted": 1}, "trusted"),
        ({"flush_always": 1}, "flush_always"),
        ({"flush_always": True, "immediate": True}, "delayed CuTe"),
    ],
)
def test_trusted_config_validation(options, match):
    with pytest.raises(ValueError, match=match):
        Config(**options)


@pytest.mark.parametrize("checks", ["trusted", "strict"])
@pytest.mark.parametrize("batch", [0, 1, 3, 65])
def test_trusted_checks_override_and_scratch(checks, batch):
    from ctikernels.gdn.replay.cute import effective_config

    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=48,
        device="cuda:0",
        provider="cute",
        cute_config=Config(trusted=checks != "trusted"),
        checks=checks,
    )
    trusted = checks == "trusted"
    assert op.capabilities["checks"] == checks
    assert op.cute_config.trusted == trusted
    assert effective_config(op, batch).trusted == trusted
    spec = op.workspace_spec("step", batch_size=batch)["arrivals"]
    assert spec.shape == (max(1, batch) if trusted else 1,)
    assert spec.payload_bytes == 4 * spec.shape[0]
    assert spec.initial_value == 0 and spec.offset % 256 == 0
    assert op.capabilities["step_scratch_bytes_per_row"] == (4 if trusted else 0)
    assert op.capabilities["step_scratch_fixed_bytes"] == (0 if trusted else 4)
    if batch:
        assert op.graph_data_spec(batch_size=batch)["arrivals"].shape == spec.shape


def test_trusted_checks_portable_and_precision_independent():
    for checks in ("trusted", "strict"):
        op = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cpu", checks=checks)
        assert op.capabilities["checks"] == "strict"
        op = GatedDeltaNetReplay(
            num_k_heads=1,
            num_v_heads=4,
            device="cuda:0",
            provider="cute",
            cute_config=Config(immediate=True),
            precision="strict",
            checks=checks,
        )
        assert op.cute_config.immediate
        assert op.cute_config.trusted == (checks == "trusted")
    with pytest.raises(ValueError, match="checks"):
        GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cpu", checks="other")
    with pytest.raises(ValueError, match="snapshot_controls"):
        GatedDeltaNetReplay(
            num_k_heads=1,
            num_v_heads=4,
            device="cuda:0",
            provider="cute",
            cute_config=Config(snapshot_controls=True, trusted=False),
        )


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_switch_effective_mixed_policy(monkeypatch, checks):
    from ctikernels.gdn.replay.cute import effective_config
    from ctikernels.gdn.replay.cute._config import POLICY

    name = "test mixed policy"
    monkeypatch.setitem(POLICY, (name, None, 1), {"immediate": False})
    monkeypatch.setitem(POLICY, (name, None, 2), {"immediate": True})
    monkeypatch.setitem(POLICY, (name, None, 4), {"immediate": True, "provider": "product"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda:0",
        provider="cute",
        checks=checks,
    )
    assert not op._ringless
    assert not effective_config(op, 1).flush_always
    for batch in (2, 3, 4):
        config = effective_config(op, batch)
        assert op.decode_route(batch) == "immediate"
        assert config.provider == "cute" and not config.immediate and config.flush_always
        assert config.trusted == (checks == "trusted")


@pytest.mark.parametrize("status_present", [False, True])
def test_trusted_step_accepts_no_status_or_ignores_status(monkeypatch, status_present):
    from ctikernels.gdn.replay import cute

    op, state, _, weights, slots, pos, status = case(batch=2)
    op.cute_config = Config()
    op.checks = "trusted"
    op._backends["cute"] = cute
    op._step_config = op.cute_config
    op.provider = {"step": "cute"}
    calls = []
    monkeypatch.setattr(cute, "prepare", lambda *args: lambda *xs: calls.append(xs))
    monkeypatch.setattr(op, "_check_status", lambda _: pytest.fail("trusted checked status"))
    out = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    arrivals = torch.zeros(2, dtype=torch.int32)
    # The CPU fixture isolates host validation and ABI binding; no CUDA launch.
    op.step(
        projected(op, 2),
        *weights,
        slot_indices=slots,
        positions=pos,
        state=state,
        out=out,
        status=object() if status_present else None,
        workspace={"arrivals": arrivals},
    )
    assert len(calls) == 1 and calls[0][17] is arrivals
    with pytest.raises((ValueError, RuntimeError), match="arrivals"):
        op.step(
            projected(op, 2),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            workspace={"arrivals": arrivals[:1]},
        )


@pytest.mark.parametrize("geometry", [(8, 16), (8, 24), (16, 32), (16, 48)])
def test_geometry_policy_precedence(monkeypatch, geometry):
    from ctikernels.gdn.replay.cute import effective_config
    from ctikernels.gdn.replay.cute._config import POLICY, all_immediate, decode_route, select

    name = "NVIDIA GeForce RTX 5090"
    monkeypatch.setitem(POLICY, (name, geometry, 4), {"immediate": False, "value_tile": 64})
    assert select(name, 3, geometry=geometry) == Config(value_tile=64)
    assert select(name, 3).provider == "product"
    assert select("unknown", 3, geometry=geometry) == Config()
    assert not all_immediate(name, geometry=geometry)
    assert decode_route(name, 3, geometry=geometry) == "delayed"
    assert decode_route(name, 1, geometry=geometry) == "immediate"
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=geometry[0], num_v_heads=geometry[1], device="cuda:0", provider="cute"
    )
    assert op.decode_route(3) == "delayed"
    assert effective_config(op, 3).value_tile == 64
    assert effective_config(op, 1).flush_always
    assert op.bytes_per_slot()["ring"] > 0


@pytest.mark.parametrize("hk,hv", [(8, 16), (8, 24), (16, 32), (16, 48)])
@pytest.mark.parametrize("window", [4, 8, 16])
@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 5090", "NVIDIA GB10"])
def test_payload_bytes_match_regions(monkeypatch, hk, hv, window, name):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=hk, num_v_heads=hv, window=window, device="cuda:0", provider="cute"
    )
    live, snapshot = op.bytes_per_slot(), op.bytes_per_snapshot_row()
    assert (live["ring"] == 0) == (op.decode_route(1) == "product")
    for sizes in (live, snapshot):
        assert all(type(v) is int for v in sizes.values())
        assert sizes["total"] == sum(v for k, v in sizes.items() if k != "total")
        with pytest.raises(TypeError):
            sizes["total"] = 0
    for r in range(1, 4):
        for s in range(1, 4):
            regions = op.state_spec(num_slots=r, num_snapshot_rows=s)
            for key in ("checkpoint", "conv_history"):
                assert regions[key].payload_bytes == (r + s) * live[key]
                assert snapshot[key] == live[key]
            for key in ("ring", "metadata"):
                assert regions[key].payload_bytes == r * live[key]
            assert regions["snapshot_metadata"].payload_bytes == s * snapshot["snapshot_metadata"]
            assert sum(x.payload_bytes for x in regions.values()) == (
                r * live["total"] + s * snapshot["total"] + regions["pool_epoch"].payload_bytes
            )


def test_step_has_no_invocation_controls():
    import inspect

    parameters = inspect.signature(GatedDeltaNetReplay.step).parameters
    assert "complete" not in parameters
    assert "advance_positions" not in parameters


def test_snapshot_controls_constructor_compatibility():
    config = Config(snapshot_controls=True)
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda:0",
        provider="cute",
        checks="strict",
        cute_config=config,
    )
    assert op.cute_config.snapshot_controls and not op.cute_config.trusted


def test_product_step_borrows_pools_and_out_and_constructs_once(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ctikernels.gdn import inference
    from ctikernels.gdn.replay import cute

    op, state, _, weights, slots, pos, _ = case(batch=2)
    op.cute_config = Config(provider="product", immediate=True)
    op._step_config = op.cute_config
    op.checks = "trusted"
    op._ringless = True
    from dataclasses import replace

    state = replace(state, ring=state.ring[:, :0].contiguous())
    calls, constructions, branches = [], [], []
    current = SimpleNamespace(
        wait_stream=lambda stream: branches.append("join"),
        record_event=lambda: branches.append("event") or "event",
    )
    side = SimpleNamespace(wait_event=lambda event: branches.append("fork:" + event))

    def stream(device):
        branches.append("create")
        return side

    def advance(owner, pool, selected):
        assert owner is op and pool is state and selected is slots
        branches.append("advance")

    def construct(**kw):
        constructions.append(kw)

        def decode(*args, **kw):
            branches.append("decode")
            calls.append((args, kw))

        return decode

    monkeypatch.setattr(inference, "GatedDeltaNetDecode", construct)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(cute, "advance_positions", advance)
    out = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    for _ in range(2):
        op.step(
            projected(op, 2),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            workspace=object(),
            status=object(),
        )
    assert len(constructions) == 1 and len(calls) == 2
    assert calls[0][0][-2] is state.checkpoint
    assert calls[0][0][-1] is state.conv_history
    assert calls[0][1]["out"] is out
    # The branch forks from an event recorded before #25, but is enqueued only
    # after #25 accepted its arguments.
    once = ["event", "decode", "fork:event", "advance", "join"]
    assert branches == ["create", *once, *once]
    assert op._side_stream is side


def test_flush_requires_status_even_when_trusted():
    op, state, _, weights, slots, _, status = case(batch=2)
    op.cute_config, op.checks = Config(), "trusted"
    with pytest.raises(TypeError, match="bound ReplayStatus"):
        op.flush(*weights, slot_indices=slots, state=state, status=None)
    op.flush(*weights, slot_indices=slots, state=state, status=status)


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_flush_capture_capability_and_ringless_noop(monkeypatch, checks):
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=Config(),
        checks=checks,
    )
    assert ("flush" in op.capabilities["capture"]) == (checks == "trusted")
    assert bool(op.workspace_spec("flush", batch_size=2)) == (checks == "trusted")
    op._ringless = True
    assert bool(op.workspace_spec("flush", batch_size=2)) == (checks == "trusted")
    monkeypatch.setattr(op, "_run", lambda *a, **kw: pytest.fail("ringless flush launched"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op.flush(None, None, None, None, slot_indices=None, state=None, status=None)


def test_product_first_construction_refused_under_capture(monkeypatch):
    from ctikernels.gdn import inference

    op, state, _, weights, slots, pos, _ = case(batch=2)
    op.cute_config = Config(provider="product", immediate=True)
    op._step_config = op.cute_config
    op.checks = "trusted"
    op._ringless = True
    from dataclasses import replace

    state = replace(state, ring=state.ring[:, :0].contiguous())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(inference, "GatedDeltaNetDecode", lambda **kw: pytest.fail("constructed"))
    with pytest.raises(RuntimeError, match="warm.*before capture"):
        op.step(
            projected(op, 2),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_export_rejects_invalid_arithmetic_batch(batch):
    op, state, _, weights, slots, _, status = case()
    with pytest.raises(ValueError, match="arithmetic_batch"):
        op.export_snapshot(
            *weights,
            pairs=torch.stack((slots, slots + state.metadata.shape[0]), 1),
            state=state,
            status=status,
            snapshot_valid=torch.empty(2, dtype=torch.bool),
            arithmetic_batch=batch,
        )


def test_export_arithmetic_batch_cache_keeps_pair_count(monkeypatch):
    from types import SimpleNamespace

    from ctikernels.gdn.replay import cute
    from ctikernels.gdn.replay.cute import _phase_launch

    op, state, _, weights, slots, _, status = case()
    op.cute_config = Config()
    op._backends["cute"] = cute
    op._step_config = op.cute_config
    op.provider = dict(op.provider, export_snapshot="cute")
    compilations, calls = [], []

    def compile_phases(*args, **kwargs):
        compilations.append((*args, kwargs.get("arithmetic_batch")))
        return lambda *xs: calls.append(xs)

    monkeypatch.setattr(_phase_launch, "compile_phases", compile_phases)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test device",
            multi_processor_count=170,
        ),
    )
    for batch in (None, 2, 4, 2):
        op.export_snapshot(
            *weights,
            pairs=torch.stack((slots[:1], slots[:1] + state.metadata.shape[0]), 1),
            state=state,
            status=status,
            snapshot_valid=torch.empty(1, dtype=torch.bool),
            arithmetic_batch=batch,
        )
    assert len(calls) == 4 and len(compilations) == 3
    assert [args[4] for args in compilations] == [1, 1, 1]
    assert [args[-1] for args in compilations] == [None, 2, 4]


@pytest.mark.parametrize("batch", [None, 3])
def test_compile_phases_selects_recipe_independently_of_grid(monkeypatch, batch):
    import sys
    from types import ModuleType, SimpleNamespace

    from ctikernels.gdn.replay import cute
    from ctikernels.gdn.replay.cute import _phase_launch

    dsl = ModuleType("cutlass")
    for name in ("BFloat16", "Boolean", "Float32", "Int32", "Int64", "Uint8"):
        setattr(dsl, name, name)
    fake_cute = ModuleType("cutlass.cute")
    fake_cute.runtime = SimpleNamespace(make_fake_tensor=lambda *a, **kw: None)
    fake_cute.sym_int64 = lambda: 1
    fake_cute.compile = lambda phase, *a, **kw: phase
    dsl.cute = fake_cute
    monkeypatch.setitem(sys.modules, "cutlass", dsl)
    monkeypatch.setitem(sys.modules, "cutlass.cute", fake_cute)
    monkeypatch.setitem(
        sys.modules,
        "cutlass.runtime",
        SimpleNamespace(
            make_fake_stream=lambda **kw: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.gdn.replay.cute._phases",
        SimpleNamespace(
            Phases=lambda *args: args,
        ),
    )
    monkeypatch.setattr(_phase_launch, "version", lambda _: "4.8.0")
    plans = []
    monkeypatch.setattr(cute, "arithmetic_plan", lambda *args: plans.append(args) or "recipe")
    phase = _phase_launch.compile_phases.__wrapped__(
        16,
        48,
        4,
        8,
        1,
        0,
        4,
        4,
        0,
        False,
        "export_snapshot",
        Config(),
        "120",
        170,
        False,
        arithmetic_batch=batch,
    )
    assert phase[3] == 1 and phase[7] == "recipe"
    assert plans[0][2] == (1 if batch is None else batch)


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_reset_clears_borrowed_arrivals_before_provider(monkeypatch, checks):
    op, state, pending, _, _, _, status = case()
    op.cute_config = Config(trusted=checks == "trusted")
    op._step_config = op.cute_config
    op.checks = checks
    step = {
        name: torch.empty(spec.shape, dtype=spec.dtype)
        for name, spec in op.workspace_spec("step", batch_size=2).items()
    }
    graph = {
        name: torch.empty(spec.shape, dtype=spec.dtype)
        for name, spec in op.graph_data_spec(batch_size=3).items()
    }
    extra = step["arrivals"].clone()
    for tensor in (step["arrivals"], graph["arrivals"], extra):
        tensor.fill_(123)
    pending.metadata.fill_(123)
    backend = op._backends[op.provider["reset"]]
    original = backend.run

    def run(op, mode, **kw):
        assert not any(t.any() for t in (step["arrivals"], graph["arrivals"], extra))
        return original(op, mode, **kw)

    monkeypatch.setattr(backend, "run", run)
    op.reset(
        state=state,
        weight_revision=7,
        status=status,
        pending_blocks=iter([pending]),
        workspaces=(step, graph, extra),
    )
    assert not pending.metadata.any()
    assert (state.metadata[:, 6] == 7).all()


@pytest.mark.parametrize(
    "tensor",
    [
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.int32),
        torch.ones(0, dtype=torch.int32),
        torch.ones(3, dtype=torch.int32),
    ],
)
def test_reset_validates_workspace_before_zeroing(tensor):
    op, state, _, _, _, _, status = case()
    op.cute_config = Config()
    workspace = {"arrivals": tensor, "out": torch.empty(2, op.geometry.value_width)}
    with pytest.raises((ValueError, TypeError)):
        op.reset(state=state, weight_revision=7, status=status, workspaces=(workspace,))
    assert tensor.eq(1).all()


@pytest.mark.parametrize("batch", [1, 3])
def test_portable_export_ignores_arithmetic_batch(batch):
    from ._replay_case import clone, decode_steps, same

    op, state, _, weights, slots, pos, status = case()
    decode_steps(op, state, weights, slots, pos, status, 3)
    expected = clone(state)
    pairs = torch.stack((slots, slots + state.metadata.shape[0]), 1)
    valid = torch.empty(2, dtype=torch.bool)
    for target, arithmetic_batch in ((expected, None), (state, batch)):
        op.export_snapshot(
            *weights,
            pairs=pairs,
            state=target,
            status=status,
            snapshot_valid=valid,
            arithmetic_batch=arithmetic_batch,
        )
        assert valid.all()
    same(state, expected)


@pytest.mark.parametrize("step_mode", ["product", "immediate", "delayed"])
@pytest.mark.parametrize("commit_mode", ["eager", "auto", "delayed"])
def test_shared_ring_sizing_and_commit_to_step_route(step_mode, commit_mode, monkeypatch):
    from ctikernels.gdn.replay.cute import effective_config

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5090")
    config = Config(
        immediate=step_mode != "delayed", provider="product" if step_mode == "product" else "cute"
    )
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=config,
        commit_mode=commit_mode,
        window=8,
        verify_length=4,
    )
    needs_ring = step_mode == "delayed" or commit_mode != "eager"
    expected_bytes = (8 + 3) * (op.geometry.conv_dim + 8) * 2 if needs_ring else 0
    spec = op.state_spec(num_slots=3, num_snapshot_rows=2)
    assert spec["ring"].payload_bytes == expected_bytes * 3
    assert op.bytes_per_slot()["ring"] == expected_bytes
    for batch in (0, 1, 2, 3, 4, 8, 16):
        effective = effective_config(op, batch)
        if needs_ring and step_mode != "delayed":
            assert op.decode_route(batch) == "immediate"
            assert effective.provider == "cute" and effective.flush_always
            assert not effective.immediate
        elif step_mode == "product":
            assert op.decode_route(batch) == "product"
            assert effective.provider == "product"


def test_pending_commit_step_never_dispatches_product(monkeypatch):
    from ctikernels.gdn.replay.cute import effective_config

    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=Config(provider="product", immediate=True),
        commit_mode="delayed",
        window=4,
        verify_length=8,
    )
    _, state, _, weights, slots, positions, _ = case(batch=2)
    state.metadata[slots.long(), 3] = 2
    state.metadata[slots.long(), 0] -= 2
    calls = []
    monkeypatch.setattr(op, "_run", lambda mode, **kw: calls.append((mode, kw)))
    op.step(
        projected(op=case()[0], batch=2),
        *weights,
        slot_indices=slots,
        positions=positions,
        state=state,
        out=torch.empty(2, 512, dtype=torch.bfloat16),
    )
    assert len(calls) == 1 and calls[0][0] == "step"
    assert calls[0][1]["state"] is state
    assert op._product_decode is None
    assert effective_config(op, 2).flush_always


@pytest.mark.parametrize("checks,q", [("trusted", 16), ("strict", 4)])
def test_auto_commit_without_delayed_buckets_needs_no_ring(checks, q, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5090")
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda:0",
        provider="cute",
        cute_config=Config(provider="product", immediate=True),
        checks=checks,
        commit_mode="auto",
        window=8,
        verify_length=q,
    )
    assert op.bytes_per_slot()["ring"] == 0
    assert op.state_spec(num_slots=3, num_snapshot_rows=1)["ring"].payload_bytes == 0


def test_reset_initializes_both_step_and_k2_counters():
    op, state, _, _, _, _, status = case(batch=2)
    op.cute_config = op._step_config = Config()
    step = {"arrivals": torch.full((2,), 123, dtype=torch.int32)}
    phases = {"phase_arrivals": torch.full((2, 3), 123, dtype=torch.int32)}
    op.reset(state=state, weight_revision=5, status=status, workspaces=(step, phases))
    assert not step["arrivals"].any()
    assert not phases["phase_arrivals"].any()


def test_borrowed_full_ring_forces_product_policy_through_cute(monkeypatch):
    from types import SimpleNamespace

    from ctikernels.gdn.replay import cute

    op, state, _, weights, slots, positions, _ = case(batch=2)
    op.cute_config = op._step_config = Config(provider="product", immediate=True)
    op.checks, op._ringless = "trusted", True
    op.provider = dict(op.provider, step="cute")
    state.metadata[:, 3] = 2
    state.metadata[:, 0] -= 2
    compiled, launched = [], []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            multi_processor_count=170,
            name="test device",
        ),
    )

    def compile_step(*args, **kwargs):
        compiled.append(args[5])
        return lambda *buffers: launched.append(buffers)

    monkeypatch.setattr(cute, "compile_step", compile_step)
    output = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="arrivals"):
        op.step(
            projected(op, 2),
            *weights,
            slot_indices=slots,
            positions=positions,
            state=state,
            out=output,
        )
    op.step(
        projected(op, 2),
        *weights,
        slot_indices=slots,
        positions=positions,
        state=state,
        out=output,
        workspace={"arrivals": torch.zeros(2, dtype=torch.int32)},
    )
    assert op._product_decode is None
    assert len(launched) == len(compiled) == 1
    assert compiled[0].provider == "cute" and compiled[0].flush_always
    assert not compiled[0].immediate
