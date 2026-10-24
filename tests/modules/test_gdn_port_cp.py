# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Host-only CP planning, bounded workspace, and graph-capture dispatch contracts."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from ctikernels.gdn.chunk_ws._cp import cp_capacity, cp_cost, cp_plan, noncp_cost
from ctikernels.gdn.chunk_ws.prefill import PortOptions, compile_key

from .test_gdn_chunk_ws import _port_cpu


@pytest.mark.parametrize("hv,sms", [(32, 170), (48, 188), (32, 1), (256, 16), (1, 188)])
def test_planner_properties(hv, sms):
    cap = cp_capacity(hv, sms)
    assert 2 <= cap <= 16
    for t in [1, 63, 64, 65, 128, 200, 256, 1000, 1024, 4100, 16384, 1000000]:
        j, length = cp_plan(t, hv, sms)
        assert j == (t + length - 1) // length
        assert j <= cap
        chunks = (t + 63) // 64
        if j >= 2:
            assert length % 64 == 0
            assert cp_cost(j, length, hv, sms) < noncp_cost(chunks)
        else:
            assert length == t
        # Exhaustively compare the bounded candidate set, including ties.
        estimates = []
        for candidate in range(2, min(chunks, cap) + 1):
            ell = 64 * ((chunks + candidate - 1) // candidate)
            actual = (t + ell - 1) // ell
            estimates.append(cp_cost(actual, ell, hv, sms))
        best = cp_cost(j, length, hv, sms) if j > 1 else noncp_cost(chunks)
        assert best == min([noncp_cost(chunks), *estimates])


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5, 16, 1000])
def test_override(target):
    for t in (1, 64, 65, 128, 200, 1000, 4100, 16384):
        j, length = cp_plan(t, 48, 170, target)
        assert j == (t + length - 1) // length
        assert j <= min(target, cp_capacity(48, 170))
        if target == 1 or t <= 64:
            assert (j, length) == (1, t)
        else:
            assert j >= 2 and length % 64 == 0
    # A forced plan must bypass the cost gate.
    assert cp_plan(128, 48, 170, 2) == (2, 64)
    assert cp_plan(128, 48, 170) == (1, 128)
    assert cp_plan(200, 48, 170, 3) == (2, 128)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3"])
def test_invalid_options_and_plans(value):
    with pytest.raises(ValueError, match="cp_segments"):
        PortOptions(cp_segments=value)
    with pytest.raises(ValueError):
        cp_plan(1000, 32, 170, value)
    for args in ((value, 32, 170), (1000, value, 170), (1000, 32, value)):
        with pytest.raises(ValueError):
            cp_plan(*args)


def test_options_and_variant_keys():
    for budget in (32, 40, 48, 56, 64, 72, 80):
        for variant in ("full", "mn", "seg"):
            producer, math = PortOptions(producer_registers=budget).registers(variant)
            assert producer % 8 == math % 8 == 0
            assert 128 * producer + 256 * math <= 168 * 384
    assert PortOptions().registers("mn") == (56, 224)
    assert PortOptions().cp is True and PortOptions().cp_segments is None
    assert PortOptions(cp=False, cp_segments=3).cp is False
    assert PortOptions().pdl is True
    with pytest.raises(TypeError, match="pdl must"):
        PortOptions(pdl=1)
    assert compile_key(16, 32, 12352, PortOptions(pdl=True), "seg") != compile_key(
        16, 32, 12352, PortOptions(pdl=False), "seg"
    )
    with pytest.raises(TypeError, match="cp must"):
        PortOptions(cp=1)
    assert len({compile_key(16, 32, 12352, PortOptions(), v) for v in ("full", "mn", "seg")}) == 3


@pytest.mark.parametrize("pdl", [False, True])
def test_cp_capture_warmup_and_call_local_dispatch(monkeypatch, pdl):
    import sys
    from types import SimpleNamespace

    from ctikernels.gdn.chunk_ws import prefill

    # CPU dispatch tests do not import the optional CuTe compiler.
    _cp_fixup = SimpleNamespace(compile_fixup=None)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn.chunk_ws._cp_fixup", _cp_fixup)

    op, args = _port_cpu()
    op.port_options = replace(op.port_options, cp_segments=3, pdl=pdl)
    args[0] = torch.empty(1000, op.geometry.proj_width, dtype=torch.bfloat16)
    args[5] = torch.tensor([0, 1000], dtype=torch.int32)
    out = torch.empty(1000, 32 * 128, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eagerly before CUDA graph capture"):
        op(*args, out=out)
    events, builds, allocations = [], [], []

    def compile_mock(*args, **kw):
        assert args[-2].pdl is pdl
        variant = args[-1]
        builds.append(variant)

        def launch(*a):
            events.append(variant)
            allocations.append((a[11], a[12]))

        return launch, lambda *a: events.append("commit")

    monkeypatch.setattr(prefill, "compile_prefill", compile_mock)

    def compile_fixup_mock(hv, options, use_pdl):
        assert use_pdl is pdl
        return lambda *a: events.append("fixup")

    monkeypatch.setattr(_cp_fixup, "compile_fixup", compile_fixup_mock)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op(*args, out=out)
    assert events == ["mn", "fixup", "seg", "commit"]
    assert builds == ["mn", "seg"]
    maps, workspace = allocations[0]
    assert workspace.shape == (3, 3, 32, 128, 128)
    assert allocations[1][1] is workspace

    def fail(*a, **kw):
        raise AssertionError("warmed call compiled or read device data")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    monkeypatch.setattr(prefill, "compile_prefill", fail)
    events.clear()
    assert op(*args, out=out) is out
    assert events == ["mn", "fixup", "seg", "commit"]
    assert allocations[-1][0] is not maps
    assert allocations[-1][1] is not workspace
    # The same variant/stride/options must accept new T and actual J=2.
    args[0], args[5][1] = args[0][:128], 128
    events.clear()
    assert op(*args, out=out[:128]).shape[0] == 128
    assert events == ["mn", "seg", "commit"]
    assert allocations[-1][1].shape == (3, 2, 32, 128, 128)
    assert len(op.compiled) == 2
    # A new row stride requires eager compilation even if W already exists.
    args[0] = args[0].as_strided((128, op.geometry.proj_width), (op.geometry.proj_width + 8, 1))
    with pytest.raises(RuntimeError, match="eagerly before CUDA graph capture"):
        op(*args, out=out[:128])


@pytest.mark.parametrize("enabled,segments,n", [(False, None, 1), (True, 1, 1), (True, 4, 2)])
def test_non_cp_dispatch_uses_only_descriptor_workspace(monkeypatch, enabled, segments, n):
    from ctikernels.gdn.chunk_ws import prefill

    op, args = _port_cpu()
    op.port_options = PortOptions(cp=enabled, cp_segments=segments)
    args[0] = torch.empty(1024, op.geometry.proj_width, dtype=torch.bfloat16)
    args[5] = torch.tensor([0, 1024] if n == 1 else [0, 512, 1024], dtype=torch.int32)
    args[6] = torch.zeros(n, dtype=torch.int32)
    args[7] = torch.zeros(n, dtype=torch.bool)
    out = torch.empty(1024, 32 * 128, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    builds = []

    def compile_mock(*args, **kw):
        builds.append(args[-1])
        return lambda *a: None, lambda *a: None

    monkeypatch.setattr(prefill, "compile_prefill", compile_mock)
    op(*args, out=out)
    assert builds == ["full"]
    assert not hasattr(op, "cp_workspace")


@pytest.mark.parametrize("segments", [1, 2, 4])
def test_bench_profiler_graph_node_fallback(monkeypatch, tmp_path, segments):
    import json
    from types import SimpleNamespace

    from benchmark.kernel_tuning.gdn_prefill_port.cp3_bench import profile_replay

    labels = ["full", "commit"] if segments == 1 else ["A", "C", "commit"]
    if segments > 2:
        labels.insert(1, "fixup")
    launches = []

    class Profiler:
        def events(self):
            return []  # Runner builds can omit uncorrelated CUDA graph nodes.

        def export_chrome_trace(self, path):
            from pathlib import Path

            Path(path).write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"cat": "kernel", "ts": i * 10, "dur": i + 1, "name": label}
                            for i, label in reversed(list(enumerate(labels)))
                        ]
                    }
                )
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: nullcontext(Profiler()))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    graph = SimpleNamespace(replay=lambda: launches.append("replay"))
    result = profile_replay(graph, segments)
    assert list(result) == labels
    assert [value["us"] for value in result.values()] == list(range(1, len(labels) + 1))
    assert launches == ["replay"]
    assert not list(tmp_path.iterdir())


def test_host_api_plan_supports_and_warm_up(monkeypatch):
    import sys
    from types import SimpleNamespace

    from ctikernels.gdn.chunk_ws import prefill

    # CPU dispatch tests do not import the optional CuTe compiler.
    _cp_fixup = SimpleNamespace(compile_fixup=None)
    monkeypatch.setitem(sys.modules, "ctikernels.gdn.chunk_ws._cp_fixup", _cp_fixup)

    op, args = _port_cpu()
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    builds = []
    monkeypatch.setattr(
        prefill, "compile_prefill", lambda *a, **kw: builds.append(a[-1]) or (None, None)
    )
    monkeypatch.setattr(_cp_fixup, "compile_fixup", lambda *a: object())
    assert op.plan(0, 1) == (1, 0)
    assert op.plan(262144, 2) == (1, 262144)
    assert op.plan(262144, 1) == cp_plan(262144, op.hv, op.num_sms)
    assert op.variants(0, 4) == ("full",)
    assert op.variants(262144, 1) == ("mn", "seg")
    assert op.supports(args[0])
    width = op.geometry.proj_width
    assert not op.supports(torch.empty(2, width + 1, dtype=torch.bfloat16)[:, :width])
    assert not op.supports(torch.empty(2 * width + 1, dtype=torch.bfloat16)[1:].view(2, width))
    assert not op.supports(args[0].t())
    assert op.warm_up(width, cp=False) == 1
    assert op.warm_up(width, cp=True) == 3
    assert op.warm_up(width, cp=True) == 0
    assert builds == ["full", "mn", "seg"]
    assert op.warm_up(width + 8, cp=True) == 3
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA graph capture"):
        op.warm_up(width, cp=True)


@pytest.mark.parametrize("rows,lengths", [(0, [0]), (0, [0, 0, 0]), (1, [0, 1, 0])])
def test_empty_dispatch_without_host_reads(monkeypatch, rows, lengths):
    from ctikernels.gdn.chunk_ws import prefill

    op, args = _port_cpu()
    args[0] = args[0][:rows]
    args[5] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    args[6] = torch.full((len(lengths),), -1, dtype=torch.int32)
    args[7] = torch.zeros(len(lengths), dtype=torch.bool)
    events = []
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        prefill,
        "compile_prefill",
        lambda *a, **kw: (lambda *a: events.append("chunk"), lambda *a: events.append("commit")),
    )

    def fail(*a, **kw):
        raise AssertionError("host read")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    assert op(*args).shape == (rows, op.hv * 128)
    assert events == (["chunk", "commit"] if rows else ["commit"])


def test_misaligned_out_copies_to_destination(monkeypatch):
    from ctikernels.gdn.chunk_ws import prefill

    op, args = _port_cpu()
    rows = args[0].shape[0]
    out = torch.empty(rows * op.hv * 128 + 1, dtype=torch.bfloat16)[1:].view(rows, -1)
    assert not op.supports(args[0], out)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def compile_mock(*a, **kw):
        def launch(*a):
            assert a[10].data_ptr() % 16 == 0
            a[10].fill_(7)

        return launch, lambda *a: None

    monkeypatch.setattr(prefill, "compile_prefill", compile_mock)
    assert op(*args, out=out) is out
    assert (out == 7).all()


@pytest.mark.parametrize("lengths", [(1,), (65,), (0, 3, 0, 67)])
@pytest.mark.parametrize("decay", [0, -12, 4])
def test_chunked_fp64_oracle_matches_token_reference(lengths, decay):
    from ..gpu import _gdn_reference as reference
    from ..gpu.test_gdn_port_prefill import make_case

    x, weights, meta, base = make_case(1, 2, lengths, device="cpu")
    weights = replace(weights, dt_bias=weights.dt_bias + decay)
    # Avoid tiny batched matrices paying the host's large BLAS thread-pool cost.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        y, state, history, boundaries = reference.prefill_chunked(
            x, weights, 1, *meta, *base, value_heads=(0, 1), boundaries=(64,)
        )
        exact_y, exact_state, exact_history = reference.prefill(x, weights, 1, *meta, *base)
        torch.testing.assert_close(y.flatten(1), exact_y, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(state, exact_state, atol=2e-12, rtol=2e-12)
        assert torch.equal(history, exact_history)
        for n, length in enumerate(lengths):
            if length >= 64:
                short_meta = (
                    torch.tensor([0, 64], dtype=torch.int32),
                    meta[1][n : n + 1],
                    meta[2][n : n + 1],
                )
                start = sum(lengths[:n])
                _, expected, _ = reference.prefill(
                    x[start : start + 64], weights, 1, *short_meta, *base
                )
                slot = meta[1][n].item()
                if slot >= 0:
                    torch.testing.assert_close(
                        boundaries[n, 64], expected[slot], atol=2e-12, rtol=2e-12
                    )
    finally:
        torch.set_num_threads(threads)
