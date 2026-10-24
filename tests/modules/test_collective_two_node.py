# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU coverage of discovery, dispatch and store-only bootstrap."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from ipaddress import IPv6Address
from types import SimpleNamespace

import pytest
import torch

from ctikernels.collective import two_node
from ctikernels.collective._two_node import discovery, proxy
from ctikernels.collective.two_node import (
    DISPATCH_MAX_BYTES,
    GATHER_EAGER_MAX_BYTES,
    GATHER_EAGER_MIN_BYTES,
    GATHER_GRAPH_MAX_BYTES,
    TwoNodeAllReduce,
    _device_error,
    _dispatch_reason,
    _staging,
    grid_policy,
    prefer_two_node,
    prefer_two_node_all_gather,
    support,
)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def gid(root, device, index, *, state="4: ACTIVE", kind="RoCE v2", value=None):
    if value is None:
        value = str(IPv6Address((0xFFFF << 32) | 1))
    base = f"{device}/ports/1"
    write(root, f"{base}/state", state)
    write(root, f"{base}/gid_attrs/types/{index}", kind)
    write(root, f"{base}/gids/{index}", value)


def test_roce_discovery(tmp_path):
    gid(tmp_path, "rdma_b", 10)
    gid(tmp_path, "rdma_b", 2)
    gid(tmp_path, "rdma_a", 3)
    gid(tmp_path, "down", 1, state="2: INIT")
    gid(tmp_path, "v1", 1, kind="IB/RoCE v1")
    gid(tmp_path, "ipv6", 1, value=str(IPv6Address(1 << 64)))
    gid(tmp_path, "empty", 1, value=str(IPv6Address(0)))
    gid(tmp_path, "bad", 1, value="invalid")
    write(tmp_path, "missing/ports/1/state", "4: ACTIVE")
    assert discovery.roce_devices(tmp_path) == [("rdma_a", 3), ("rdma_b", 2), ("rdma_b", 10)]
    assert discovery.roce_devices(tmp_path / "absent") == []


@pytest.mark.parametrize("field", ["cpu_capacity", "cpufreq/cpuinfo_max_freq"])
def test_big_cores(tmp_path, field):
    for cpu, value in enumerate([512, 1024, 512, 1024, 2048]):
        write(tmp_path, f"cpu{cpu}/{field}", str(value))
    write(tmp_path, "cpu4/online", "0")
    assert discovery.big_cores(tmp_path) == [1, 3]
    assert discovery.proxy_core(tmp_path) == 3
    assert discovery.proxy_core(tmp_path / "absent") == -1


def test_frequency_precedes_capacity(tmp_path):
    write(tmp_path, "cpu0/cpu_capacity", "1024")
    write(tmp_path, "cpu1/cpu_capacity", "512")
    write(tmp_path, "cpu1/cpufreq/cpuinfo_max_freq", "9999")
    assert discovery.big_cores(tmp_path) == [1]


def test_capacity_tier_tolerance(tmp_path):
    # GB10: X925 cores read 997-1024, A725 cores 718-731.
    for cpu, value in enumerate([718, 731, 997, 1017, 1024]):
        write(tmp_path, f"cpu{cpu}/cpu_capacity", str(value))
    assert discovery.big_cores(tmp_path) == [2, 3, 4]


def tensor(**kwargs):
    values = {
        "is_cuda": True,
        "device": torch.device("cuda:0"),
        "dtype": torch.bfloat16,
        "shape": (8,),
        "numel": lambda: 8,
        "element_size": lambda: 2,
        "is_contiguous": lambda: True,
        "data_ptr": lambda: 4096,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def reason(x, out=None, gather=False):
    return _dispatch_reason(x, out, max_bytes=256 * 1024, gather=gather)


def stage(x, out=None, rank=0, gather=False):
    nbytes = x.numel() * x.element_size()
    return _staging(x, out, nbytes=nbytes, rank=rank, gather=gather)


def test_valid_and_in_place():
    x = tensor()
    assert reason(x) is None
    assert reason(x, x) is None
    assert reason(x, tensor(data_ptr=lambda: 8192)) is None
    assert stage(x, x) == (False, False)
    assert stage(x, tensor(data_ptr=lambda: 8192)) == (False, False)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"dtype": torch.int32}, "dtype"),
        ({"numel": lambda: 7}, "multiple of 16"),
        ({"numel": lambda: 0}, "positive"),
        ({"numel": lambda: 262144}, "max_bytes"),
    ],
)
def test_symmetric_unsupported_input(changes, match):
    assert match in reason(tensor(**changes))


@pytest.mark.parametrize(
    "changes,match", [({"shape": (4, 2)}, "shape"), ({"dtype": torch.float16}, "dtype")]
)
def test_symmetric_unsupported_output(changes, match):
    assert match in reason(tensor(), tensor(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"is_cuda": False},
        {"device": torch.device("cuda:1")},
        {"is_contiguous": lambda: False},
        {"data_ptr": lambda: 4098},
    ],
)
@pytest.mark.parametrize("gather", [False, True])
def test_local_layout_never_selects_transport(changes, gather):
    """Alignment, contiguity and device are local facts: never a dispatch reason."""
    assert reason(tensor(**changes), gather=gather) is None


def test_device_error_is_local_value_error():
    device = torch.device("cuda:0")
    assert _device_error(tensor(), None, device) is None
    assert "x must" in _device_error(tensor(device=torch.device("cuda:1")), None, device)
    out = tensor(device=torch.device("cpu"))
    assert "out must" in _device_error(tensor(), out, device)


@pytest.mark.parametrize(
    "x_changes,out_changes,expected",
    [
        ({"data_ptr": lambda: 4098}, {}, (True, False)),
        ({"is_contiguous": lambda: False}, {}, (True, False)),
        ({}, {"data_ptr": lambda: 8194}, (False, True)),
        ({}, {"is_contiguous": lambda: False}, (False, True)),
    ],
)
def test_staging_plan(x_changes, out_changes, expected):
    out = tensor(data_ptr=lambda: 8192)
    out.__dict__.update(out_changes)
    assert stage(tensor(**x_changes), out) == expected


def test_partial_overlap_stages_input():
    x = tensor(shape=(32,), numel=lambda: 32)
    out = tensor(shape=(32,), numel=lambda: 32, data_ptr=lambda: 4112)
    assert stage(x, out) == (True, False)


def test_provider_fallback_and_value_error():
    op = object.__new__(TwoNodeAllReduce)
    op._closed, op.active, op.device, op.max_bytes = False, True, torch.device("cuda:0"), 16
    x, out = tensor(numel=lambda: 16), tensor()
    calls = []
    op.fallback = lambda a, b: calls.append((a, b)) or b
    assert op(x, out) is out
    assert calls == [(x, out)]
    op.fallback = None
    with pytest.raises(ValueError, match="max_bytes"):
        op(x)
    with pytest.raises(ValueError, match="multiple of 16"):
        op(tensor(numel=lambda: 7))


def test_bootstrap():
    store = torch.distributed.HashStore()

    class FakeProxy:
        def __init__(self, rank):
            self.rank = rank
            self.peer = None

        def local_endpoint(self):
            return bytes([self.rank]) * 40

        def connect(self, endpoint):
            self.peer = endpoint

    proxies = [FakeProxy(0), FakeProxy(1)]
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(proxy.bootstrap, p, store, "fresh", r, 262144)
            for r, p in enumerate(proxies)
        ]
        for task in tasks:
            task.result(timeout=5)
    assert proxies[0].peer == proxies[1].local_endpoint()
    assert proxies[1].peer == proxies[0].local_endpoint()


def test_bootstrap_rejects_mismatched_layout():
    store = torch.distributed.HashStore()
    fake = SimpleNamespace(
        local_endpoint=lambda: bytes(40), connect=lambda _: pytest.fail("connect")
    )
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(proxy.bootstrap, fake, store, "mismatch", r, 16 * (r + 1)) for r in range(2)
        ]
        for task in tasks:
            with pytest.raises(ValueError, match="same max_bytes"):
                task.result(timeout=5)


def test_barrier_timeout():
    with pytest.raises(TimeoutError):
        proxy.barrier(torch.distributed.HashStore(), "alone", timeout_s=0.01)


@pytest.mark.parametrize(
    "nbytes,expected",
    [
        (0, False),
        (15, False),
        (16, True),
        (8192, True),
        (DISPATCH_MAX_BYTES, True),
        (DISPATCH_MAX_BYTES + 16, False),
        (512 * 1024, False),
    ],
)
def test_dispatch(nbytes, expected):
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert prefer_two_node(nbytes, dtype) is expected
    assert not prefer_two_node(nbytes, torch.int32)


def test_grid_policy():
    assert grid_policy(16, 20) == (1, 512)
    assert grid_policy(65536, 20) == (1, 512)
    assert grid_policy(65552, 20) == (2, 512)
    assert grid_policy(262144, 2) == (2, 512)


def test_support_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert isinstance(support(), str)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert isinstance(support("invalid-device"), str)


def test_close_unregisters_before_free(monkeypatch):
    from contextlib import nullcontext

    events = []
    op = object.__new__(TwoNodeAllReduce)
    op._closed, op._registered, op.host, op.device = False, True, 4096, "cuda:0"
    op._proxy = SimpleNamespace(shutdown=lambda: events.append("shutdown"))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(cudaHostUnregister=lambda _: events.append("unregister") or 0),
    )
    op.close()
    op.close()
    assert events == ["sync", "unregister", "shutdown"]


@pytest.mark.parametrize("proxy_error,status", [("", 0), ("CQ error", 0), ("", 1)])
def test_check_synchronizes_before_reading_errors(monkeypatch, proxy_error, status):
    events = []
    op = object.__new__(TwoNodeAllReduce)
    op.device, op.active = "cuda:0", True
    op.stats = lambda: events.append("stats") or {"error": proxy_error}
    op.words = [None, None, None, SimpleNamespace(item=lambda: events.append("status") or status)]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    if proxy_error or status:
        with pytest.raises(RuntimeError):
            op.check()
    else:
        op.check()
    assert events[:2] == ["sync", "stats"]
    assert ("status" in events) == (not proxy_error)


def test_verbs_include_flags(tmp_path):
    from ctikernels.collective._two_node.proxy import verbs_include_flags

    write(tmp_path, "b/infiniband/verbs.h", "")
    assert verbs_include_flags([str(tmp_path / "a"), str(tmp_path / "b")]) == [
        "-idirafter",
        str(tmp_path / "b"),
    ]
    assert verbs_include_flags([str(tmp_path / "a")]) == []


def test_agree():
    store = torch.distributed.HashStore()
    with ThreadPoolExecutor(2) as pool:
        both = [pool.submit(proxy.agree, store, "ok", r, "", 5) for r in range(2)]
        one = [pool.submit(proxy.agree, store, "one", r, "no verbs" * r, 5) for r in range(2)]
        assert [t.result(timeout=5) for t in both] == ["", ""]
        assert [t.result(timeout=5) for t in one] == ["rank 1: no verbs"] * 2
    with pytest.raises(TimeoutError, match="peer rank 1 did not answer"):
        proxy.agree(store, "alone", 0, "", timeout_s=0.05)


def fake_transport(monkeypatch, *, open_errors=(), connect_errors=()):
    """Fake local open/connect per rank; records teardown of opened resources."""
    events = []

    def fake_open(self, device, ibdev, gid_index):
        self._proxy = SimpleNamespace(shutdown=lambda: events.append(("shutdown", self.rank)))
        if self.rank in open_errors:
            raise open_errors[self.rank]

    def fake_connect(self, store, prefix, proxy_cpu, backoff, timeout_s):
        if self.rank in connect_errors:
            raise connect_errors[self.rank]

    monkeypatch.setattr(TwoNodeAllReduce, "_open", fake_open)
    monkeypatch.setattr(TwoNodeAllReduce, "_connect", fake_connect)
    return events


def construct_pair(fallback, ranks=(0, 1), timeout_s=5, gather_fallback=None):
    store = torch.distributed.HashStore()
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(
                TwoNodeAllReduce,
                rank=r,
                store=store,
                prefix="agree",
                fallback=fallback,
                gather_fallback=gather_fallback,
                timeout_s=timeout_s,
            )
            for r in ranks
        ]
        results = []
        for task in tasks:
            try:
                results.append(task.result(timeout=10))
            except Exception as error:  # noqa: BLE001 - asserted by the caller
                results.append(error)
        return results


def test_both_ranks_ok_use_b1(monkeypatch):
    events = fake_transport(monkeypatch)
    ops = construct_pair(fallback=lambda x, out: x)
    assert [op.active for op in ops] == [True, True]
    assert events == []


@pytest.mark.parametrize(
    "case",
    [
        {"open_errors": {1: RuntimeError("libibverbs.so.1: cannot open shared object file")}},
        {"open_errors": {0: OSError("cudaHostRegister failed")}},
        {"connect_errors": {1: RuntimeError("QP transition to RTR failed")}},
        {"connect_errors": {0: TimeoutError("peer endpoint missing")}},
    ],
    ids=["missing-libibverbs", "failed-init", "peer-connect-failure", "local-connect-failure"],
)
def test_any_rank_failure_puts_both_on_fallback(monkeypatch, case):
    events = fake_transport(monkeypatch, **case)
    failing = next(iter({**case.get("open_errors", {}), **case.get("connect_errors", {})}))
    calls = []

    def fallback(x, out):
        calls.append(x)
        return x

    ops = construct_pair(fallback)
    assert [op.active for op in ops] == [False, False]
    for op in ops:
        assert op.reason.startswith(f"rank {failing}: ")
        assert op.stats() == {"error": "", "fallback": op.reason}
        op.prepare([8192])
        op.arm()
        op.disarm()
        op.check()
        # Every call, including sizes B1 would take, goes to the fallback.
        assert op("x") == "x"
    assert calls == ["x", "x"]
    assert sorted(events) == [("shutdown", 0), ("shutdown", 1)]
    for op in ops:
        op.close()
    assert len(events) == 2


def test_failure_without_fallback_raises_on_both_ranks(monkeypatch):
    fake_transport(monkeypatch, open_errors={1: RuntimeError("libibverbs.so.1 missing")})
    errors = construct_pair(fallback=None)
    for error in errors:
        assert isinstance(error, RuntimeError)
        assert "rank 1: RuntimeError: libibverbs.so.1 missing" in str(error)


def test_silent_peer_times_out_with_clear_error(monkeypatch):
    events = fake_transport(monkeypatch)
    (error,) = construct_pair(fallback=lambda x, out: x, ranks=(0,), timeout_s=0.05)
    assert isinstance(error, TimeoutError)
    assert "peer rank 1 did not answer agree/opened" in str(error)
    assert events == [("shutdown", 0)]


def test_worker_startup_failure_puts_both_on_fallback(monkeypatch):
    """Real _connect: bootstrap runs, then rank 1's proxy start() reports its
    worker's startup failure (as the extension does after its ready-or-error
    handshake), before the started agreement."""
    events = []

    class StartFailProxy:
        def __init__(self, rank):
            self.rank = rank

        def local_endpoint(self):
            return bytes(40)

        def connect(self, endpoint):
            events.append(("connect", self.rank))

        def start(self, cpu, *backoff):
            events.append(("start", self.rank, cpu))
            if self.rank == 1:
                raise RuntimeError("proxy worker startup failed: proxy CPU affinity failed")

        def shutdown(self):
            events.append(("shutdown", self.rank))

    def fake_open(self, device, ibdev, gid_index):
        self._proxy = StartFailProxy(self.rank)

    monkeypatch.setattr(TwoNodeAllReduce, "_open", fake_open)
    store = torch.distributed.HashStore()
    with ThreadPoolExecutor(2) as pool:
        tasks = [
            pool.submit(
                TwoNodeAllReduce,
                rank=r,
                store=store,
                prefix="worker",
                proxy_cpu=1023,
                fallback=lambda x, out: x,
                timeout_s=5,
            )
            for r in range(2)
        ]
        ops = [t.result(timeout=10) for t in tasks]
    assert [op.active for op in ops] == [False, False]
    for op in ops:
        assert (
            op.reason
            == "rank 1: RuntimeError: proxy worker startup failed: proxy CPU affinity failed"
        )
        assert op("x") == "x"
    assert sorted(e for e in events if e[0] == "start") == [("start", 0, 1023), ("start", 1, 1023)]
    assert sorted(e for e in events if e[0] == "shutdown") == [("shutdown", 0), ("shutdown", 1)]


def test_proxy_start_waits_for_worker_ready():
    """The extension's start() must not return before the worker reports."""
    from pathlib import Path

    import ctikernels

    source = (Path(ctikernels.__file__).parent / "csrc/src/two_node_proxy.cpp").read_text()
    start = source[source.index("void start(") : source.index("void arm(")]
    assert "wait_for(" in start and "proxy worker startup failed" in start
    run = source[source.index("void run(") :]
    assert run.index("pthread_setaffinity_np") < run.index('ready.set_value("")')


def test_probe_is_not_shipped():
    from pathlib import Path

    import ctikernels

    # Provenance comments may name it; no include, import or build source may.
    root = Path(ctikernels.__file__).parent
    for path in root.rglob("*"):
        if path.suffix in {".py", ".cpp", ".cu", ".cuh", ".h", ".txt", ".toml"}:
            for line in path.read_text(errors="ignore").splitlines():
                if "two_node_allreduce_probe" in line:
                    assert line.lstrip().startswith(("//", "#")), (path, line)
                    assert "#include" not in line, (path, line)


def gather_reason(x, out=None):
    return _dispatch_reason(x, out, max_bytes=524288, gather=True)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32, torch.bfloat16, torch.complex64])
@pytest.mark.parametrize("rank", [0, 1])
def test_gather_valid_and_in_place(dtype, rank):
    width = torch.empty((), dtype=dtype).element_size()
    x = tensor(dtype=dtype, shape=(2, 8), numel=lambda: 16, element_size=lambda: width)
    out = tensor(dtype=dtype, shape=(2, 2, 8), data_ptr=lambda: 8192)
    assert gather_reason(x) is None
    assert gather_reason(x, out) is None
    assert stage(x, out, rank, gather=True) == (False, False)
    x.data_ptr = lambda: 8192 + rank * 16 * width
    assert stage(x, out, rank, gather=True) == (False, False)
    x.data_ptr = lambda: 8192 + (1 - rank) * 16 * width
    assert stage(x, out, rank, gather=True) == (True, False)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"numel": lambda: 0}, "positive"),
        ({"numel": lambda: 7}, "multiple of 16"),
        ({"numel": lambda: 262152}, "max_bytes"),
    ],
)
def test_gather_symmetric_unsupported_input(changes, match):
    assert match in gather_reason(tensor(**changes))


@pytest.mark.parametrize(
    "changes,match", [({"shape": (16,)}, "shape"), ({"dtype": torch.float16}, "dtype")]
)
def test_gather_symmetric_unsupported_output(changes, match):
    values = {"shape": (2, 8), "data_ptr": lambda: 8192}
    values.update(changes)
    assert match in gather_reason(tensor(), tensor(**values))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("offset", [-16, 16, 48])
def test_gather_partial_overlap(rank, offset):
    x = tensor(shape=(16,), numel=lambda: 16, data_ptr=lambda: 8192 + offset)
    out = tensor(shape=(2, 16), data_ptr=lambda: 8192)
    assert stage(x, out, rank, gather=True) == (True, False)


@pytest.mark.parametrize("nbytes", [16, 4096, 129280, 258560, 387840, 517120, 524288])
def test_gather_payload_sizes(nbytes):
    assert gather_reason(tensor(numel=lambda: nbytes // 2)) is None


@pytest.mark.parametrize(
    "nbytes,capturing,expected",
    [
        (-16, False, False),
        (0, True, False),
        (15, True, False),
        (16, False, False),
        (16, True, True),
        (GATHER_EAGER_MIN_BYTES - 16, False, False),
        (GATHER_EAGER_MIN_BYTES, False, True),
        (129280, False, True),
        (GATHER_EAGER_MAX_BYTES, False, True),
        (GATHER_EAGER_MAX_BYTES + 16, False, False),
        (517120, False, False),
        (517120, True, True),
        (GATHER_GRAPH_MAX_BYTES, True, True),
        (GATHER_GRAPH_MAX_BYTES + 16, True, False),
    ],
)
def test_gather_dispatch(nbytes, capturing, expected):
    assert prefer_two_node_all_gather(nbytes, capturing) is expected


def test_gather_dispatch_reads_capture_state(monkeypatch):
    monkeypatch.setattr(two_node.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(two_node.torch.cuda, "is_current_stream_capturing", lambda: True)
    assert prefer_two_node_all_gather(517120)
    monkeypatch.setattr(two_node.torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not prefer_two_node_all_gather(517120)


def gather_provider():
    op = object.__new__(TwoNodeAllReduce)
    op._closed, op.active, op.rank = False, True, 0
    op.device, op.max_bytes, op.sms = torch.device("cuda:0"), 524288, 20
    op.gather_fallback, op.fallback = None, None
    op.compiled = {}
    return op


def test_gather_unsupported_fallback():
    op = gather_provider()
    x, out = tensor(numel=lambda: 7), tensor()
    calls = []
    op.fallback = lambda *args: pytest.fail("all-reduce fallback used for gather")
    op.gather_fallback = lambda a, b: calls.append((a, b)) or b
    assert op.all_gather(x, out) is out
    assert calls == [(x, out)]
    op.gather_fallback = None
    with pytest.raises(ValueError, match="multiple of 16"):
        op.all_gather(x)
    with pytest.raises(ValueError, match="shape"):
        op.all_gather(tensor(), out)


@pytest.mark.parametrize("gather_only", [False, True])
def test_inactive_op_specific_fallback(monkeypatch, gather_only):
    fake_transport(monkeypatch, open_errors={1: RuntimeError("unavailable")})
    calls = []
    fallback = lambda x, out: calls.append((x, out)) or "result"
    ops = construct_pair(
        fallback=None if gather_only else fallback,
        gather_fallback=fallback if gather_only else None,
    )
    for op in ops:
        assert not op.active
        op.prepare_all_gather([16])
        available = op.all_gather if gather_only else op
        unavailable = op if gather_only else op.all_gather
        assert available("x", "out") == "result"
        with pytest.raises(RuntimeError, match="rank 1: RuntimeError: unavailable"):
            unavailable("x")
    assert calls == [("x", "out"), ("x", "out")]


@pytest.mark.parametrize("nbytes", [-16, 0, 17, 524304])
def test_prepare_gather_invalid_size(nbytes):
    with pytest.raises(ValueError, match="positive multiples of 16 <= max_bytes"):
        gather_provider().prepare_all_gather([nbytes])


def test_prepare_gather_grid_and_closed():
    op = gather_provider()
    calls = []
    op._kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    op.prepare_all_gather([16, 65536, 65552, 517120])
    assert calls == [((None, ctas, 512), {"gather": True}) for ctas in (1, 1, 2, 8)]
    op._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        op.prepare_all_gather([16])
    with pytest.raises(RuntimeError, match="closed"):
        op.all_gather(tensor())


@pytest.fixture
def fake_cute_kernels(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    import ctikernels

    cute = SimpleNamespace(jit=lambda fn: fn, kernel=lambda fn: fn, Tensor=object)
    cute.runtime = SimpleNamespace(
        make_fake_tensor=lambda *a, **kw: "words",
        make_fake_stream=lambda **kw: "stream",
    )
    cute.compile = lambda *args, **kwargs: (args, kwargs)
    cutlass = SimpleNamespace(
        cute=cute,
        Float32=object(),
        Float16=object(),
        BFloat16=object(),
        Int32=int,
        Int64=int,
        const_expr=bool,
    )
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    monkeypatch.setitem(sys.modules, "cutlass._mlir.dialects", SimpleNamespace(llvm=None))
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.collective._host_exchange.cute.kernels",
        SimpleNamespace(_ptr=None, load_raw=None, put=None, store_raw=None),
    )
    name = "ctikernels.collective._two_node.cute.kernels"
    path = Path(ctikernels.__file__).parent / "collective/_two_node/cute/kernels.py"
    spec = importlib.util.spec_from_file_location(name, path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    monkeypatch.setitem(sys.modules, name, kernels)
    return kernels


@pytest.mark.parametrize("dtype", [None, torch.int8, torch.bfloat16, torch.float32])
def test_compile_step_gather(fake_cute_kernels, dtype):
    args, kwargs = fake_cute_kernels.compile_step(dtype, 2, 512, gather=True, options="extra")
    step = args[0]
    assert (step.dtype, step.ctas, step.threads, step.gather) == (None, 2, 512, True)
    assert kwargs == {"options": "--enable-tvm-ffi extra"}
    assert not fake_cute_kernels.compile_step(torch.bfloat16, 2, 512)[0][0].gather


def test_gather_cache_and_capture(monkeypatch, fake_cute_kernels):
    op = gather_provider()
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    reduce = op._kernel(torch.bfloat16, 2, 512)
    gather = op._kernel(torch.bfloat16, 2, 512, gather=True)
    assert gather is not reduce
    assert op._kernel(torch.int8, 2, 512, gather=True) is gather
    op.prepare_all_gather([129280])
    assert len(op.compiled) == 2
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert op._kernel(None, 2, 512, gather=True) is gather
    with pytest.raises(RuntimeError, match=r"prepare_all_gather\(\)"):
        op.prepare_all_gather([16])


@pytest.mark.parametrize("allocate", [False, True])
def test_gather_launch_shares_sequence(monkeypatch, allocate):
    op = gather_provider()
    op.words, op.base, op.layout = object(), 16384, (524288,)
    x, out = tensor(), tensor(shape=(2, 8), data_ptr=lambda: 8192)
    launches = []
    op._kernel = lambda *args, **kwargs: lambda *launch: launches.append(launch)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())

    def empty(shape, *, dtype, device):
        assert (shape, dtype, device) == ((2, 8), x.dtype, x.device)
        return out

    monkeypatch.setattr(torch, "empty", empty)
    assert op.all_gather(x, None if allocate else out) is out
    assert launches == [(4096, 8192, op.words, op.base, 524288, 1, 0)]


def simulated_pair(monkeypatch, payloads, *, gather):
    """Two CPU providers whose 'kernel' reads/writes real pointers like the GPU step.

    Each launch records the transport and requires 16 B-aligned operands; the
    peer contribution comes from payloads[1 - rank].
    """
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    used, ops = [], []
    for rank in (0, 1):
        op = gather_provider()
        op.rank, op.device, op.max_bytes, op.sms = rank, torch.device("cpu"), 4096, 1
        op.words, op.base, op.layout = None, 0, (4096,)
        op._stage = torch.empty(3 * op.max_bytes, dtype=torch.uint8)

        def launch(src, dst, words, base, slot, count, r, *, op=op):
            assert src % 16 == 0 and dst % 16 == 0, "kernel operands must be 16 B aligned"
            nbytes = 16 * count
            local = (ctypes.c_uint8 * nbytes).from_address(src)
            local = torch.frombuffer(bytearray(local), dtype=torch.uint8)
            peer = payloads[1 - r].contiguous().view(torch.uint8).flatten()
            if gather:
                halves = (local, peer) if r == 0 else (peer, local)
                result = torch.cat(halves)
            else:
                dtype = payloads[r].dtype
                a, b = (local, peer) if r == 0 else (peer, local)
                result = (a.view(dtype).float() + b.view(dtype).float()).to(dtype)
                result = result.view(torch.uint8)
            ctypes.memmove(dst, result.data_ptr(), result.numel())
            used.append((r, "b1"))

        op._kernel = lambda *args, launch=launch, **kwargs: launch
        fallback = lambda x, out, rank=rank: used.append((rank, "fallback")) or out
        op.fallback = op.gather_fallback = fallback
        ops.append(op)
    return ops, used


def layout(values, kind, out_shape=None):
    """Return (x, out) carrying `values` with a rank-local memory layout."""
    n, dtype = values.numel(), values.dtype
    if kind == "aligned":
        return values.clone(), torch.empty(out_shape or values.shape, dtype=dtype)
    if kind == "misaligned":
        x = torch.empty(n + 1, dtype=dtype)[1:]
        x.copy_(values)
        return x, torch.empty(out_shape or values.shape, dtype=dtype)
    if kind == "strided":
        x = torch.empty(2 * n, dtype=dtype)[::2]
        x.copy_(values)
        return x, torch.empty(out_shape or values.shape, dtype=dtype)
    shape = out_shape or values.shape
    count = int(torch.Size(shape).numel())
    if kind == "out-misaligned":
        return values.clone(), torch.empty(count + 1, dtype=dtype)[1:].view(shape)
    if kind == "out-strided":
        return values.clone(), torch.empty((*shape, 2), dtype=dtype)[..., 0]
    raise AssertionError(kind)


LAYOUTS = ["misaligned", "strided", "out-misaligned", "out-strided"]


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("odd_rank", [0, 1])
def test_gather_asymmetric_layout_same_transport(monkeypatch, kind, odd_rank):
    """Review regression: a 2 B-aligned 16 B BF16 slice on one rank, aligned on its peer."""
    payloads = [torch.arange(8, dtype=torch.bfloat16) + 100 * r for r in (0, 1)]
    ops, used = simulated_pair(monkeypatch, payloads, gather=True)
    expected = torch.stack(payloads)
    for rank, op in enumerate(ops):
        x, out = layout(payloads[rank], kind if rank == odd_rank else "aligned", (2, 8))
        assert op.all_gather(x, out) is out
        assert torch.equal(out, expected)
    assert used == [(0, "b1"), (1, "b1")]


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_reduce_asymmetric_layout_same_transport(monkeypatch, kind, dtype):
    payloads = [(torch.arange(16) * (r + 1)).to(dtype) for r in (0, 1)]
    ops, used = simulated_pair(monkeypatch, payloads, gather=False)
    expected = (payloads[0].float() + payloads[1].float()).to(dtype)
    for rank, op in enumerate(ops):
        x, out = layout(payloads[rank], kind if rank == 0 else "aligned")
        assert op(x, out) is out
        assert torch.equal(out, expected)
    assert used == [(0, "b1"), (1, "b1")]


@pytest.mark.parametrize("rank", [0, 1])
def test_gather_misaligned_in_place_and_partial_overlap(monkeypatch, rank):
    payloads = [torch.arange(8, dtype=torch.bfloat16) + 100 * r for r in (0, 1)]
    ops, used = simulated_pair(monkeypatch, payloads, gather=True)
    expected = torch.stack(payloads)
    op = ops[rank]
    # x is exactly out[rank] in a 2 B-aligned output: staged, still in place.
    out = torch.empty(17, dtype=torch.bfloat16)[1:].view(2, 8)
    x = out[rank]
    x.copy_(payloads[rank])
    assert op.all_gather(x, out) is out
    assert torch.equal(out, expected)
    # x overlaps out other than at out[rank]: the input is read from a copy.
    out = torch.empty(2, 8, dtype=torch.bfloat16)
    x = out.flatten()[(1 - rank) * 8 : (1 - rank) * 8 + 8]
    x.copy_(payloads[rank])
    assert op.all_gather(x, out) is out
    assert torch.equal(out, expected)
    assert used == [(rank, "b1"), (rank, "b1")]


def test_all_reduce_partial_overlap_reads_copy(monkeypatch):
    payloads = [torch.arange(8, dtype=torch.bfloat16) * (r + 1) for r in (0, 1)]
    ops, used = simulated_pair(monkeypatch, payloads, gather=False)
    buffer = torch.zeros(16, dtype=torch.bfloat16)
    x, out = buffer[:8], buffer[8:]
    x.copy_(payloads[0])
    assert ops[0](x, out) is out
    assert torch.equal(out, (payloads[0].float() + payloads[1].float()).to(torch.bfloat16))
    assert used == [(0, "b1")]


def test_symmetric_facts_fall_back_on_both_ranks(monkeypatch):
    """Odd sizes are identical on both ranks, so both ranks use the fallback."""
    payloads = [torch.zeros(3, dtype=torch.bfloat16) for _ in range(2)]
    ops, used = simulated_pair(monkeypatch, payloads, gather=True)
    for rank, op in enumerate(ops):
        aligned = rank == 0
        x = payloads[rank].clone() if aligned else torch.empty(4, dtype=torch.bfloat16)[1:]
        op.all_gather(x, torch.empty(2, 3, dtype=torch.bfloat16))
        op(x, torch.empty(3, dtype=torch.bfloat16))
    assert used == [(0, "fallback")] * 2 + [(1, "fallback")] * 2


def test_off_device_tensor_raises(monkeypatch):
    ops, used = simulated_pair(monkeypatch, [torch.zeros(8, dtype=torch.bfloat16)] * 2, gather=True)
    ops[0].device = torch.device("cuda:0")
    with pytest.raises(ValueError, match="provider device"):
        ops[0].all_gather(torch.zeros(8, dtype=torch.bfloat16))
    assert used == []
