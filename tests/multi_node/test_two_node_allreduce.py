# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Run the same module on both ranks; only the Store coordinates the tests."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from ctikernels.collective._two_node import proxy as proxy_module
from ctikernels.collective._two_node.proxy import barrier, module
from ctikernels.collective.two_node import TwoNodeAllReduce, support

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CTI_TWO_NODE_RANK") and os.environ.get("CTI_TWO_NODE_STORE")),
    reason="set CTI_TWO_NODE_RANK and CTI_TWO_NODE_STORE on both hosts",
)
SIZES = (16, 2048, 8192, 16384, 24576, 32768, 65536, 131072, 262144)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def raw(tensor):
    return bytes(tensor.detach().contiguous().view(torch.uint8).flatten().cpu().tolist())


def inputs(rank, step, size, dtype):
    gen = torch.Generator().manual_seed(900001 * rank + 1009 * step + size)
    count = size // torch.empty((), dtype=dtype).element_size()
    return torch.randn(count, generator=gen).to(dtype)


def reference(step, size, dtype):
    a, b = [inputs(r, step, size, dtype) for r in range(2)]
    return (a.float() + b.float()).to(dtype)


def exchange(ctx, key, value):
    ctx.store.set(f"{key}/{ctx.rank}", json.dumps(value))
    return json.loads(ctx.store.get(f"{key}/{1 - ctx.rank}"))


def verify(ctx, key, result, expected):
    payload = raw(result)
    local = {"ok": payload == raw(expected), "hash": hashlib.sha256(payload).hexdigest()}
    peer = exchange(ctx, key, local)
    assert local["ok"] and peer["ok"], f"rank-order mismatch: {key}"
    assert local["hash"] == peer["hash"], f"peer mismatch: {key}"


def check_pair(ctx, key):
    ok = True
    try:
        ctx.op.check()
    except RuntimeError:
        ok = False
    assert exchange(ctx, key, ok) and ok, "proxy or device reported an error"


@pytest.fixture(scope="module")
def pair():
    rank = int(os.environ["CTI_TWO_NODE_RANK"])
    address, port = os.environ["CTI_TWO_NODE_STORE"].rsplit(":", 1)
    torch.cuda.set_device(0)
    store = dist.TCPStore(address, int(port), 2, rank == 0, timedelta(seconds=120))
    ctx = SimpleNamespace(rank=rank, store=store, nccl=None, nccl_reason="")
    reason = support()
    peer = exchange(ctx, "support", reason)
    if reason or peer:
        pytest.skip(reason or peer)
    ctx.op = TwoNodeAllReduce(rank=rank, store=store, prefix="qualification/provider")
    # Preflight both libraries before entering a collective NCCL constructor.
    try:
        from ctikernels.pynccl import check_nccl_runtime_version, init_pynccl

        check_nccl_runtime_version()
    except Exception as error:  # noqa: BLE001 - optional comparison
        ctx.nccl_reason = f"NCCL preflight: {type(error).__name__}"
    peer_reason = exchange(ctx, "nccl/preflight", ctx.nccl_reason)
    group_created = False
    if not (ctx.nccl_reason or peer_reason):
        try:
            dist.init_process_group(
                "gloo",
                store=dist.PrefixStore("qualification/gloo", store),
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=120),
            )
            group_created = True
            ctx.nccl = init_pynccl(
                tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=0
            )
        except Exception as error:  # noqa: BLE001 - optional comparison
            ctx.nccl_reason = f"NCCL init: {type(error).__name__}: {str(error)[:300]}"
    else:
        ctx.nccl_reason = ctx.nccl_reason or peer_reason
    peer_reason = exchange(ctx, "nccl/initialized", ctx.nccl_reason)
    ctx.nccl_reason = ctx.nccl_reason or peer_reason
    yield ctx
    torch.cuda.synchronize()
    barrier(store, "qualification/finished")
    ctx.op.close()
    if group_created:
        dist.destroy_process_group()
    # Rank 0 hosts the store: it may exit only after rank 1 has left the barrier.
    if rank == 1:
        store.set("qualification/released", "1")
    else:
        store.get("qualification/released")


def test_eager_bitwise(pair):
    for dtype in DTYPES:
        for size in SIZES:
            for step in range(20):
                x = inputs(pair.rank, step, size, dtype).cuda()
                result = pair.op(x)
                verify(pair, f"eager/{dtype}/{size}/{step}", result, reference(step, size, dtype))
    check_pair(pair, "eager/check")


def test_nccl_and_measured_floor(pair):
    if pair.nccl_reason:
        pytest.skip(pair.nccl_reason)
    # The existing pynccl wrapper accepts BF16/FP16 only.
    for dtype in DTYPES[:2]:
        for size in SIZES:
            x = inputs(pair.rank, 97, size, dtype).cuda()
            ours, market = pair.op(x), x.clone()
            pair.nccl.all_reduce(market, "sum")
            verify(pair, f"nccl/{dtype}/{size}", ours, market)
            a, b = [inputs(r, 97, size, dtype).double() for r in range(2)]
            oracle = a + b
            err = [(value.cpu().double() - oracle).abs() for value in (ours, market)]
            metrics = [{"max": e.max().item(), "rms": e.square().mean().sqrt().item()} for e in err]
            ok = all(metrics[0][k] <= metrics[1][k] for k in ("max", "rms"))
            peer = exchange(pair, f"floor/{dtype}/{size}", ok)
            print(f"measured_floor {dtype} {size}: {metrics}")
            assert ok and peer


def test_graph_replay_nondefault_stream(pair):
    sizes = [((i % 4) + 1) * 8192 for i in range(43)]
    pair.op.prepare(sizes)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        xs = [torch.zeros(size // 2, device="cuda", dtype=torch.bfloat16) for size in sizes]
        ys = [torch.empty_like(x) for x in xs]
        # Warm up the provider outside capture, on the stream used for capture.
        pair.op(xs[0], ys[0])
    stream.synchronize()
    barrier(pair.store, "graph/capture-ready")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for x, out in zip(xs, ys, strict=True):
            pair.op(x, out)
    for replay in range(5):
        with torch.cuda.stream(stream):
            for i, (x, size) in enumerate(zip(xs, sizes, strict=True)):
                x.copy_(inputs(pair.rank, replay * 43 + i, size, torch.bfloat16))
            graph.replay()
        stream.synchronize()
        for i, (out, size) in enumerate(zip(ys, sizes, strict=True)):
            verify(
                pair, f"graph/{replay}/{i}", out, reference(replay * 43 + i, size, torch.bfloat16)
            )
        # Explicit host synchronization serializes the default and capture streams.
        eager = inputs(pair.rank, replay, 8192, torch.bfloat16).cuda()
        verify(
            pair, f"graph/eager/{replay}", pair.op(eager), reference(replay, 8192, torch.bfloat16)
        )
    check_pair(pair, "graph/check")


def test_in_place_and_fallback(pair):
    for dtype in DTYPES:
        x = inputs(pair.rank, 33, 8192, dtype).cuda()
        same = pair.op(x, x) is x
        assert exchange(pair, f"inplace/alias/{dtype}", same) and same
        verify(pair, f"inplace/{dtype}", x, reference(33, 8192, dtype))
    invocation = 0

    def fallback(x, out):
        nonlocal invocation
        key = f"fallback/{invocation}"
        invocation += 1
        # CPU store reference only for this test's eager fallback cases.
        peer = exchange(pair, key, x.float().cpu().tolist())
        other = torch.tensor(peer, dtype=x.dtype)
        local = x.cpu()
        a, b = (local, other) if pair.rank == 0 else (other, local)
        value = (a.float() + b.float()).to(x.dtype)
        target = torch.empty_like(x) if out is None else out
        target.copy_(value)
        return target

    pair.op.fallback = fallback
    try:
        for size in (2 * pair.op.max_bytes, 18):
            x = inputs(pair.rank, 49, size, torch.bfloat16).cuda()
            verify(pair, f"fallback/result/{size}", pair.op(x), reference(49, size, torch.bfloat16))
    finally:
        pair.op.fallback = None
    for size in (2 * pair.op.max_bytes, 18):
        rejected = False
        try:
            pair.op(inputs(pair.rank, 50, size, torch.bfloat16).cuda())
        except ValueError:
            rejected = True
        assert exchange(pair, f"fallback/rejected/{size}", rejected) and rejected


def test_missing_libibverbs_falls_back_to_nccl(pair, monkeypatch):
    if pair.nccl_reason:
        pytest.skip(pair.nccl_reason)
    if pair.rank == 1:
        # This rank reports what a host without libibverbs reports; rank 0
        # opens the real transport and must release it after the agreement.
        monkeypatch.setattr(
            proxy_module, "available", lambda: "libibverbs.so.1: cannot open shared object file"
        )

    def nccl_fallback(x, out):
        target = x if out is None else out.copy_(x)
        pair.nccl.all_reduce(target, "sum")
        return target

    op = TwoNodeAllReduce(
        rank=pair.rank, store=pair.store, prefix="qualification/no-verbs", fallback=nccl_fallback
    )
    try:
        print(f"no-verbs provider: active={op.active} reason={op.reason!r}")
        ok = not op.active and op.reason.startswith("rank 1: ") and "libibverbs" in op.reason
        assert exchange(pair, "no-verbs/agreed", ok) and ok
        for size in (8192, 32768):
            x = inputs(pair.rank, 61, size, torch.bfloat16).cuda()
            verify(pair, f"no-verbs/{size}", op(x), reference(61, size, torch.bfloat16))
    finally:
        op.close()


def test_worker_affinity_failure_falls_back_to_nccl(pair):
    if pair.nccl_reason:
        pytest.skip(pair.nccl_reason)

    def nccl_fallback(x, out):
        target = x if out is None else out.copy_(x)
        pair.nccl.all_reduce(target, "sum")
        return target

    # Rank 1's proxy worker is pinned to a CPU outside its cpuset; the real
    # extension must report the worker's failure before the started agreement.
    outside = next(c for c in range(1023, -1, -1) if c not in os.sched_getaffinity(0))
    op = TwoNodeAllReduce(
        rank=pair.rank,
        store=pair.store,
        prefix="qualification/bad-cpu",
        proxy_cpu=outside if pair.rank == 1 else None,
        fallback=nccl_fallback,
    )
    try:
        print(f"bad-cpu provider: active={op.active} reason={op.reason!r}")
        ok = not op.active and op.reason.startswith("rank 1: ") and "affinity" in op.reason
        assert exchange(pair, "bad-cpu/agreed", ok) and ok
        for size in (8192, 32768):
            x = inputs(pair.rank, 67, size, torch.bfloat16).cuda()
            verify(pair, f"bad-cpu/{size}", op(x), reference(67, size, torch.bfloat16))
    finally:
        op.close()


def test_adaptive_proxy(pair):
    pair.op.disarm()
    time.sleep(0.05)
    stats = pair.op.stats()
    ok = stats["state"] != "HOT" and stats["futex_sleeps"] > 0
    assert exchange(pair, "adaptive/cold", ok) and ok
    x = inputs(pair.rank, 51, 8192, torch.bfloat16).cuda()
    verify(pair, "adaptive/wake", pair.op(x), reference(51, 8192, torch.bfloat16))
    pair.op.arm()
    verify(pair, "adaptive/armed", pair.op(x), reference(51, 8192, torch.bfloat16))
    pair.op.disarm()
    time.sleep(0.05)
    start = pair.op.stats()
    time.sleep(1)
    end = pair.op.stats()
    fraction = (end["cpu_ns"] - start["cpu_ns"]) / (end["wall_ns"] - start["wall_ns"])
    print(f"idle proxy CPU fraction: {fraction:.4%}")
    peer = exchange(pair, "adaptive/cpu", fraction)
    assert max(fraction, peer) < 0.05


def test_check_and_verbs_symbols(pair):
    check_pair(pair, "final/check")
    has_nm = shutil.which("nm") is not None
    peer_has_nm = exchange(pair, "symbols/nm", has_nm)
    if not (has_nm and peer_has_nm):
        pytest.skip("nm unavailable")
    module()  # Ensure the lazy extension is loaded.
    # Locate the mapped extension without assuming a JIT cache root.
    from pathlib import Path

    libraries = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "two_node_proxy" in line and ".so" in line
    }
    violations = [] if libraries else ["extension mapping not found"]
    for library in libraries:
        symbols = subprocess.check_output(["nm", "-D", library], text=True)
        violations.extend(re.findall(r"\bU\s+ibv_\w+", symbols))
        if shutil.which("readelf"):
            dynamic = subprocess.check_output(["readelf", "-d", library], text=True)
            if re.search(r"NEEDED.*libibverbs", dynamic):
                violations.append("libibverbs NEEDED")
    ok = not violations
    assert exchange(pair, "symbols/result", ok) and ok


GATHER_SIZES = (16, 4096, 129280, 258560, 387840, 517120)


def byte_inputs(rank, step, size, dtype):
    gen = torch.Generator().manual_seed(900001 * rank + 1009 * step + size)
    return torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen).view(dtype)


@pytest.fixture(scope="module")
def gather_pair(pair):
    op = TwoNodeAllReduce(
        rank=pair.rank, store=pair.store, prefix="qualification/gather", max_bytes=524288
    )
    ctx = SimpleNamespace(rank=pair.rank, store=pair.store, op=op)
    yield ctx
    torch.cuda.synchronize()
    barrier(pair.store, "qualification/gather-finished")
    op.close()


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16])
@pytest.mark.parametrize("in_place", [False, True])
def test_all_gather_bytes(gather_pair, dtype, in_place):
    ctx = gather_pair
    ctx.op.prepare_all_gather(GATHER_SIZES)
    for size in GATHER_SIZES:
        expected = torch.stack([byte_inputs(r, 73, size, dtype) for r in range(2)])
        out = torch.empty(expected.shape, dtype=dtype, device="cuda")
        x = out[ctx.rank] if in_place else torch.empty_like(out[ctx.rank])
        x.view(torch.uint8).copy_(expected[ctx.rank].view(torch.uint8))
        assert ctx.op.all_gather(x, out) is out
        verify(ctx, f"gather/{dtype}/{in_place}/{size}", out, expected)
    check_pair(ctx, f"gather/{dtype}/{in_place}/check")


def test_graph_interleaved_all_reduce_all_gather(gather_pair):
    ctx = gather_pair
    ctx.op.prepare([8192])
    ctx.op.prepare_all_gather([129280])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        xs = [torch.zeros(4096, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        ys = [torch.empty_like(x) for x in xs]
        gs = [torch.zeros(64640, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        outs = [torch.empty((2, *x.shape), dtype=x.dtype, device=x.device) for x in gs]
        ctx.op(xs[0], ys[0])
        ctx.op.all_gather(gs[0], outs[0])
    stream.synchronize()
    barrier(ctx.store, "mixed-graph/capture-ready")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for x, y, g, out in zip(xs, ys, gs, outs, strict=True):
            ctx.op(x, y)
            ctx.op.all_gather(g, out)
    for replay in range(5):
        expected = [
            torch.stack([byte_inputs(r, 2 * replay + i, 129280, torch.bfloat16) for r in range(2)])
            for i in range(2)
        ]
        with torch.cuda.stream(stream):
            for i, (x, g) in enumerate(zip(xs, gs, strict=True)):
                x.copy_(inputs(ctx.rank, 2 * replay + i, 8192, torch.bfloat16))
                g.view(torch.uint8).copy_(expected[i][ctx.rank].view(torch.uint8))
            graph.replay()
        stream.synchronize()
        for i, (y, out) in enumerate(zip(ys, outs, strict=True)):
            verify(
                ctx,
                f"mixed-graph/{replay}/{i}/reduce",
                y,
                reference(2 * replay + i, 8192, torch.bfloat16),
            )
            verify(ctx, f"mixed-graph/{replay}/{i}/gather", out, expected[i])
    check_pair(ctx, "mixed-graph/check")


def local_layout(values, kind, out_shape):
    """(x, out) on the GPU holding `values` with a rank-local memory layout."""
    n, dtype = values.numel(), values.dtype
    x = torch.empty(n, dtype=dtype, device="cuda")
    out = torch.empty(out_shape, dtype=dtype, device="cuda")
    if kind == "misaligned":
        x = torch.empty(n + 1, dtype=dtype, device="cuda")[1:]
    elif kind == "strided":
        x = torch.empty(2 * n, dtype=dtype, device="cuda")[::2]
    elif kind == "out-misaligned":
        count = torch.Size(out_shape).numel()
        out = torch.empty(count + 1, dtype=dtype, device="cuda")[1:].view(out_shape)
    elif kind == "out-strided":
        out = torch.empty((*out_shape, 2), dtype=dtype, device="cuda")[..., 0]
    x.copy_(values.view(-1))
    return x.view(values.shape), out


ASYMMETRIC = ("misaligned", "strided", "out-misaligned", "out-strided")


@pytest.mark.parametrize("kind", ASYMMETRIC)
def test_asymmetric_layout_same_transport(gather_pair, kind):
    """Rank 0 passes a locally odd layout, rank 1 an aligned one; no fallback exists,
    so a rank-divergent transport choice would raise or hang instead of passing."""
    ctx = gather_pair
    assert ctx.op.fallback is None and ctx.op.gather_fallback is None
    mine = kind if ctx.rank == 0 else "aligned"
    ctx.op.prepare([8192])
    ctx.op.prepare_all_gather([16, 129280])
    for size in (16, 129280):
        expected = torch.stack([byte_inputs(r, 81, size, torch.bfloat16) for r in range(2)])
        x, out = local_layout(expected[ctx.rank].cuda(), mine, tuple(expected.shape))
        assert ctx.op.all_gather(x, out) is out
        verify(ctx, f"asym/{kind}/gather/{size}", out, expected)
    values = inputs(ctx.rank, 82, 8192, torch.bfloat16).cuda()
    x, out = local_layout(values, mine, tuple(values.shape))
    assert ctx.op(x, out) is out
    verify(ctx, f"asym/{kind}/reduce", out, reference(82, 8192, torch.bfloat16))
    check_pair(ctx, f"asym/{kind}/check")


def test_asymmetric_layout_graph_replay(gather_pair):
    """The staging copies are captured: rank 0 misaligned/strided, rank 1 aligned."""
    ctx = gather_pair
    ctx.op.prepare([8192])
    ctx.op.prepare_all_gather([16])
    gather_kind, reduce_kind = ("misaligned", "out-strided") if ctx.rank == 0 else ("aligned",) * 2
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g, gout = local_layout(torch.zeros(8, dtype=torch.bfloat16).cuda(), gather_kind, (2, 8))
        x, out = local_layout(torch.zeros(4096, dtype=torch.bfloat16).cuda(), reduce_kind, (4096,))
        ctx.op.all_gather(g, gout)
        ctx.op(x, out)
    stream.synchronize()
    barrier(ctx.store, "asym-graph/capture-ready")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ctx.op.all_gather(g, gout)
        ctx.op(x, out)
    for replay in range(4):
        expected = torch.stack([byte_inputs(r, 90 + replay, 16, torch.bfloat16) for r in range(2)])
        with torch.cuda.stream(stream):
            g.copy_(expected[ctx.rank].cuda())
            x.copy_(inputs(ctx.rank, 90 + replay, 8192, torch.bfloat16))
            graph.replay()
        stream.synchronize()
        verify(ctx, f"asym-graph/{replay}/gather", gout, expected)
        verify(
            ctx, f"asym-graph/{replay}/reduce", out, reference(90 + replay, 8192, torch.bfloat16)
        )
    check_pair(ctx, "asym-graph/check")
