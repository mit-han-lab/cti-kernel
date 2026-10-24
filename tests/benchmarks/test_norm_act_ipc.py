# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import base64
import io
import pickle
import weakref
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import norm_act_market as market
from benchmark.kernel_tuning.norm_act_pools import (
    POOL_BASES,
    build_norm_pools,
    norm_pool_views,
    pool_spec,
)


def spec_for(regime="hot", summed=True):
    args = SimpleNamespace(
        op="add_norm" + ("_sum" if summed else ""),
        width=16,
        dtype="bf16",
        layers=4,
        static_weight=True,
    )
    return pool_spec(args, 3, regime, 1024)


@pytest.mark.parametrize("summed", [False, True])
def test_handle_payload_roundtrip_exports_all_bases_without_cuda(monkeypatch, summed):
    spec = spec_for(summed=summed)
    bases = {name: SimpleNamespace(device=torch.device("cuda:0"), name=name) for name in POOL_BASES}
    if not summed:
        bases["aux"] = None
    reduced, rebuilt = [], []

    def rebuild(name, handle, shape, stride, offset, dtype):
        rebuilt.append((name, handle, shape, stride, offset, dtype))
        return bases[name]

    def reduce(tensor):
        reduced.append(tensor.name)
        return rebuild, (tensor.name, b"\x00\xffipc", (3, 16), (48, 1), 16, torch.bfloat16)

    # Resolve the rebuild by module/name exactly as the real CUDA reducer does.
    monkeypatch.setattr(market.reductions, "reduce_tensor", reduce)
    original_import = market.importlib.import_module
    monkeypatch.setattr(
        market.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(rebuild=rebuild)
            if name == rebuild.__module__
            else original_import(name)
        ),
    )
    devices = []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(market, "norm_pool_views", lambda spec, **kwargs: kwargs)
    handles = market.encode_pool_handles(SimpleNamespace(**bases))
    handles = market.decode_message(market.encode_message({"handles": handles}))["handles"]
    assert set(handles) == set(POOL_BASES)
    module, function, args = pickle.loads(base64.b64decode(handles["saved_r"]))
    assert (module, function) == (rebuild.__module__, "rebuild")
    assert args == ("saved_r", b"\x00\xffipc", (3, 16), (48, 1), 16, torch.bfloat16)
    decoded = market.decode_pool_handles(spec, handles)
    assert all(decoded[name] is base for name, base in bases.items())
    assert reduced == [name for name in POOL_BASES if bases[name] is not None]
    assert [args[0] for args in rebuilt] == reduced
    assert devices == [torch.device("cuda:0")]


def test_ipc_failures_are_explicit_and_redacted(monkeypatch):
    pools = build_norm_pools(spec_for(), device="cpu")
    with pytest.raises(market.CudaIPCError, match="CUDA IPC pool sharing failed"):
        market.encode_pool_handles(pools)
    fake = SimpleNamespace(
        **dict.fromkeys(POOL_BASES, SimpleNamespace(device=torch.device("cuda")))
    )

    def unsupported(tensor):
        raise RuntimeError("native diagnostic must not escape")

    monkeypatch.setattr(market.reductions, "reduce_tensor", unsupported)
    with pytest.raises(market.CudaIPCError, match="private worker pools are not supported") as exc:
        market.encode_pool_handles(fake)
    assert "native diagnostic" not in str(exc.value)
    for handles in ({}, dict.fromkeys(POOL_BASES, "invalid base64"), dict.fromkeys(POOL_BASES)):
        with pytest.raises(market.CudaIPCError):
            market.decode_pool_handles(spec_for(), handles)


def test_export_failure_does_not_start_a_private_worker(monkeypatch):
    def unexpected_spawn(*args):
        pytest.fail("worker spawned after IPC export failed")

    monkeypatch.setattr(market, "MarketWorker", unexpected_spawn)
    pools = build_norm_pools(spec_for(), device="cpu")
    with pytest.raises(market.CudaIPCError):
        market.start_market(SimpleNamespace(), pools, cute_config=None)


@pytest.mark.parametrize("regime", ["hot", "engine", "cold"])
@pytest.mark.parametrize("summed", [False, True])
def test_shared_view_builder_preserves_aliases_layout_and_restore_sources(regime, summed):
    parent = build_norm_pools(spec_for(regime, summed), device="cpu")
    worker = norm_pool_views(parent.spec, **{name: getattr(parent, name) for name in POOL_BASES})
    assert worker.count == parent.count
    for local, remote in zip(parent.inputs, worker.inputs, strict=True):
        for a, b in zip(local, remote, strict=True):
            assert (a.shape, a.stride(), a.storage_offset(), a.data_ptr()) == (
                b.shape,
                b.stride(),
                b.storage_offset(),
                b.data_ptr(),
            )
    for name in ("x", "residual", "h1"):
        getattr(worker, name).zero_()
        assert not getattr(parent, name).any()
    parent.restore()
    for name, saved in (("x", "saved_x"), ("residual", "saved_r"), ("h1", "saved_h1")):
        assert torch.equal(getattr(worker, name), getattr(parent, saved))
        getattr(parent, name).fill_(5)
    worker.restore()
    assert torch.equal(parent.x, parent.saved_x)
    assert torch.equal(parent.residual, parent.saved_r)
    assert torch.equal(parent.h1, parent.saved_h1)


def test_worker_drops_graphs_retained_tensors_and_cycles_before_free_ack(monkeypatch):
    worker = market.Worker.__new__(market.Worker)

    class Reference:
        pass

    pools = Reference()
    graph = Reference()
    graph.pools = pools
    graph.cycle = graph
    pool_ref, graph_ref = weakref.ref(pools), weakref.ref(graph)
    worker.pools = pools
    worker.banks = {"cute_remote": ([graph], [pools])}
    worker.candidates = {"cute_remote": pools}
    del graph, pools
    events = []

    def sync():
        events.append(("sync", pool_ref() is None, graph_ref() is None))

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    assert worker.dispatch("free", {}) == {}
    events.append("ack")
    assert events == [("sync", False, False), ("sync", True, True), "empty_cache", "ack"]
    assert worker.pools is None and not worker.banks and not worker.candidates


@pytest.mark.parametrize("ok", [True, False])
def test_parent_retains_allocations_until_successful_free_ack(monkeypatch, ok):
    worker = market.MarketWorker.__new__(market.MarketWorker)
    pools = build_norm_pools(spec_for(), device="cpu")
    worker.shared_pools = pools
    worker.sequence = 0
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))

    class Reply:
        def readline(self):
            assert worker.shared_pools is pools
            request = market.decode_message(worker.process.stdin.getvalue())
            assert request["command"] == "free"
            assert events == ["sync"]
            events.append("ack")
            return market.encode_message(
                {"version": market.PROTOCOL_VERSION, "id": 1, "ok": ok, "result": {}}
            )

    worker.process = SimpleNamespace(stdin=io.StringIO(), stdout=Reply())
    if ok:
        worker.request("free")
        assert worker.shared_pools is None
    else:
        with pytest.raises(market.MarketError):
            worker.request("free")
        assert worker.shared_pools is pools


def test_remote_ipc_error_preserves_clear_safe_message(monkeypatch):
    worker = market.MarketWorker.__new__(market.MarketWorker)
    worker.sequence = 0
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    worker.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            market.encode_message(
                {
                    "version": market.PROTOCOL_VERSION,
                    "id": 1,
                    "ok": False,
                    "error_type": "CudaIPCError",
                }
            )
        ),
    )
    with pytest.raises(market.CudaIPCError, match="CUDA IPC pool sharing failed"):
        worker.request("setup")


def test_worker_restores_before_each_replay_outside_timing(monkeypatch):
    events = []
    worker = market.Worker.__new__(market.Worker)
    worker.layers = 2
    worker.pools = SimpleNamespace(restore=lambda: events.append("restore"))
    graph = SimpleNamespace(replay=lambda: events.append("replay"))

    def elapsed(end):
        events.append("elapsed")
        return 0.004

    worker.banks = {"cute_remote": ([(graph, SimpleNamespace(elapsed_time=elapsed), object())], [])}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    for _ in range(2):
        assert worker.sample("cute_remote") == 2
    assert events == ["restore", "replay", "sync", "elapsed"] * 2


def test_capture_warmup_uses_shared_views_and_restores_them():
    from benchmark.kernel_tuning.norm_act_paired import warm_capture_call

    pools = build_norm_pools(spec_for("engine"), device="cpu")
    seen = []

    class Candidate:
        def call_index(self, tensors, output, index):
            assert tensors is pools.inputs[index]
            seen.append(index)
            tensors[0].zero_()
            tensors[1].zero_()

    warm_capture_call(Candidate(), pools.inputs, [None] * pools.count, restore=pools.restore)
    assert seen == list(range(pools.count))
    assert torch.equal(pools.x, pools.saved_x)
    assert torch.equal(pools.residual, pools.saved_r)


def test_close_frees_shared_pools_before_exit_and_waits_cooperatively(monkeypatch):
    worker = market.MarketWorker.__new__(market.MarketWorker)
    worker.closed = False
    worker.sequence = 0
    worker.shared_pools = build_norm_pools(spec_for(), device="cpu")
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))

    class Pipe:
        def write(self, line):
            request = market.decode_message(line)
            if request["command"] == "free":
                assert worker.shared_pools is not None
            else:
                assert worker.shared_pools is None
            events.append(request["command"])

        def flush(self):
            pass

        def readline(self):
            return market.encode_message(
                {
                    "version": market.PROTOCOL_VERSION,
                    "id": worker.sequence,
                    "ok": True,
                    "result": {},
                }
            )

        def close(self):
            events.append("close_pipe")

    worker.process = SimpleNamespace(
        stdin=Pipe(), stdout=Pipe(), poll=lambda: None, wait=lambda: events.append("wait")
    )
    worker.close()
    worker.close()
    assert events == ["sync", "free", "sync", "exit", "close_pipe", "close_pipe", "wait"]
    assert worker.shared_pools is None
