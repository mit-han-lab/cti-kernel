# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU contract, lazy import, capture guard and attribution checks for GP-CP1."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120, _validate

from ..support.child import run_child

ROOT = Path(__file__).resolve().parents[2]


def _args():
    return {
        "q": torch.empty(2, 16, 128, dtype=torch.bfloat16),
        "k": torch.empty(2, 16, 128, dtype=torch.bfloat16),
        "v": torch.empty(2, 32, 128, dtype=torch.bfloat16),
        "g": torch.empty(2, 32),
        "beta": torch.empty(2, 32),
        "o": torch.empty(2, 32, 128, dtype=torch.bfloat16),
        "pool": torch.empty(3, 32, 128, 128),
        "slots": torch.empty(1, dtype=torch.int32),
        "initial": torch.empty(1, dtype=torch.bool),
        "cu_seqlens": torch.empty(2, dtype=torch.int32),
        "scale": 128**-0.5,
    }


def validate(args):
    _validate(torch.device("cpu"), 16, 32, **args)


@pytest.mark.parametrize("installed", [True, False])
def test_chunk_ws_import_laziness(installed):
    code = """
import importlib.abc
import importlib.metadata
import sys
import torch
class NoDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('cutlass', 'nvidia_cutlass_dsl'):
            raise AssertionError('eager DSL import')
sys.meta_path.insert(0, NoDSL())
def no_cuda(*args, **kwargs):
    raise AssertionError('CUDA initialization during import')
torch.cuda._lazy_init = no_cuda
old = importlib.metadata.version
def version(name):
    if name == 'nvidia-cutlass-dsl':
        if INSTALLED:
            return '4.8.0'
        raise importlib.metadata.PackageNotFoundError(name)
    return old(name)
importlib.metadata.version = version
from ctikernels.gdn import chunk_ws
from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120
from ctikernels.gdn.chunk_ws.prefill import PortPrefill
assert chunk_ws.available() is INSTALLED
assert 'cutlass' not in sys.modules
if not INSTALLED:
    assert chunk_ws.load('cuda', 16, 32) is None
""".replace("INSTALLED", repr(installed))
    result = run_child(
        [sys.executable, "-c", code],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", list(_args())[:-1])
def test_chunk_ws_rejects_dtype(name):
    args = _args()
    args[name] = args[name].to(torch.float64)
    with pytest.raises(ValueError, match=f"{name} must have dtype"):
        validate(args)


@pytest.mark.parametrize("name", list(_args())[:-1])
def test_chunk_ws_rejects_rank(name):
    args = _args()
    args[name] = args[name].unsqueeze(0)
    with pytest.raises(ValueError, match=f"{name} must have rank"):
        validate(args)


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta", "o", "pool", "initial", "cu_seqlens"])
def test_chunk_ws_rejects_shape(name):
    args = _args()
    shape = list(args[name].shape)
    shape[-1] += 1
    args[name] = torch.empty(shape, dtype=args[name].dtype)
    with pytest.raises(ValueError, match=f"{name} must have shape"):
        validate(args)


def test_chunk_ws_validation_never_reads_values(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("device value read")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    validate(_args())


def test_chunk_ws_rejects_noncontiguous_and_wrong_device():
    args = _args()
    args["pool"] = args["pool"].transpose(-1, -2)
    with pytest.raises(ValueError, match="pool must be contiguous"):
        validate(args)
    args = _args()
    with pytest.raises(ValueError, match="q must be on cuda:0"):
        _validate(torch.device("cuda:0"), 16, 32, **args)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), torch.tensor(0.1), True])
def test_chunk_ws_rejects_scale(scale):
    args = _args()
    args["scale"] = scale
    with pytest.raises(ValueError, match="finite host number"):
        validate(args)


@pytest.mark.parametrize("hk,hv", [(0, 32), (16, 24), (32, 16)])
def test_chunk_ws_rejects_geometry_before_cuda(hk, hv):
    with pytest.raises(ValueError, match="GVA requires"):
        ChunkPrefillSm120("cuda", hk, hv)


def test_chunk_ws_rejects_cpu_device():
    with pytest.raises(ValueError, match="CUDA device"):
        ChunkPrefillSm120("cpu", 16, 32)


def test_chunk_ws_refuses_first_compile_during_capture(monkeypatch):
    op = object.__new__(ChunkPrefillSm120)
    op.device, op.hk, op.hv, op.compiled = torch.device("cpu"), 16, 32, None
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eagerly before CUDA graph capture"):
        op(**_args())


def test_chunk_ws_headers_and_notices():
    names = {
        "_gates": ("alpha",),
        "_inverse": ("collective_inverse_hmma",),
        "_o_store": ("collective_store_tma",),
        "_mma": ("helpers",),
        "_kernel": ("delta_rule_sm120", "schedule"),
        "_launch": ("delta_rule_sm120",),
    }
    notice = (
        (ROOT / "THIRD_PARTY_NOTICES").read_text().split("\nFlashInfer (GDN chunk prefill)\n")[1]
    )
    notice = notice.split("\nMini-SGLang\n")[0]
    assert "Source: flashinfer-python 0.7.0.post1 wheel" in notice
    for name, originals in names.items():
        local = f"python/ctikernels/gdn/chunk_ws/{name}.py"
        header = (ROOT / local).read_text().split("\n\n")[0]
        assert header.startswith("# SPDX-License-Identifier: Apache-2.0\n")
        assert "SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team" in header
        assert "SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors" in header
        assert "Adapted from flashinfer 0.7.0.post1" in header
        assert local in notice
        for original in originals:
            upstream = f"flashinfer/gdn_kernels/delta_rule_dsl/{original}.py"
            assert upstream in header and upstream in notice


def _port_cpu():
    from ctikernels.gdn._contract import Geometry
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, PortPrefill

    op = object.__new__(PortPrefill)
    op.device, op.hk, op.hv = torch.device("cpu"), 16, 32
    op.geometry, op.port_options = Geometry(16, 32), PortOptions()
    op.compiled, op.options, op.nsmid = {}, "offline-test", 170
    op.num_sms, op.cp_capacity = 170, 11
    op.cp_fixup = None
    g = op.geometry
    args = [
        torch.empty(2, g.proj_width, dtype=torch.bfloat16),
        torch.empty(g.conv_dim, 4, dtype=torch.bfloat16),
        torch.empty(32),
        torch.empty(32),
        torch.empty(128),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([False]),
        torch.empty(1, 32, 128, 128),
        torch.empty(1, 3, g.conv_dim, dtype=torch.bfloat16),
    ]
    return op, args


@pytest.mark.parametrize("index", range(10))
def test_port_reuses_contract_dtype_checks(index):
    op, args = _port_cpu()
    args[index] = args[index].double()
    with pytest.raises(TypeError):
        op(*args)


def test_port_compile_key_capture_and_symbolic_sizes(monkeypatch):
    from ctikernels.gdn.chunk_ws import prefill

    op, args = _port_cpu()
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eagerly before CUDA graph capture"):
        op(*args)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def compile_mock(*values, **kw):
        calls.append(values)
        return (lambda *a: None), (lambda *a: None)

    monkeypatch.setattr(prefill, "compile_prefill", compile_mock)
    out = op(*args)
    assert out.shape == (2, 32 * 128) and out.dtype == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    # Change T, N and S without triggering another compile or any device value read.
    args[0] = torch.empty(5, args[0].shape[1], dtype=torch.bfloat16)
    args[5] = torch.tensor([0, 2, 5], dtype=torch.int32)
    args[6] = torch.tensor([1, -1], dtype=torch.int32)
    args[7] = torch.tensor([True, False])
    args[8] = torch.empty(3, 32, 128, 128)
    args[9] = torch.empty(3, 3, op.geometry.conv_dim, dtype=torch.bfloat16)

    def fail(*a, **kw):
        raise AssertionError("device value read")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    op(*args)
    assert len(calls) == 1
    args[0] = torch.empty(5, op.geometry.proj_width + 8, dtype=torch.bfloat16)[:, :-8]
    with pytest.raises(RuntimeError, match="row stride"):
        op(*args)


def test_port_options_and_frozen_weights():
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, PortPrefill, compile_key

    for p in (32, 40, 48, 56, 64, 72, 80):
        options = PortOptions(producer_registers=p)
        assert 128 * p + 256 * options.math_registers <= 168 * 384
        assert options.math_registers % 8 == 0
        assert compile_key(16, 32, 12352, options) != compile_key(16, 32, 12360, options)
    with pytest.raises(ValueError, match="producer_registers"):
        PortOptions(producer_registers=41)
    with pytest.raises(TypeError, match="unexpected keyword argument 'fused'"):
        PortPrefill(16, 32, "cuda", fused=False)
    op, args = _port_cpu()
    args[2].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        op(*args)


def test_port_iket_summary_decodes_string_table(tmp_path, capsys):
    import importlib.util
    import json

    path = ROOT / "benchmark/kernel_tuning/gdn_prefill_port/cp2_iket.py"
    spec = importlib.util.spec_from_file_location("cp2_iket_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "stringTable": [
                    "producer.wait_empty",
                    "producer.conv_norm",
                    "math.SK_NewV",
                    "math.wait_v",
                ],
                "launches": [
                    {
                        "ranges": [
                            {"rangeNameIdx": 0, "startTs": 0, "endTs": 10},
                            {"rangeNameIdx": 1, "startTs": 10, "endTs": 40},
                            {"rangeNameIdx": 2, "startTs": 0, "endTs": 100},
                            {"rangeNameIdx": 3, "startTs": 20, "endTs": 40},
                        ]
                    }
                ],
            }
        )
    )
    module.summarize(trace, 1)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    fractions = {row["role"]: row["wait_fraction"] for row in rows if "role" in row}
    assert fractions == {"producer": 0.25, "math": 0.2}


@pytest.mark.parametrize("prefetch", [2, 4, 8])
def test_port_prefetch_specializes_cache(prefetch):
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, compile_key

    options = PortOptions(prefetch=prefetch, producer_registers=72)
    assert options.math_registers == 216
    keys = {compile_key(16, 48, 16480, PortOptions(prefetch=pf)) for pf in (2, 4, 8)}
    assert len(keys) == 3
    assert options.prefetch == prefetch


@pytest.mark.parametrize("prefetch", [0, 1, 3, 16, True, 2.0])
def test_port_rejects_invalid_prefetch(prefetch):
    from ctikernels.gdn.chunk_ws.prefill import PortOptions

    with pytest.raises(ValueError, match="prefetch"):
        PortOptions(prefetch=prefetch)


def test_port_iket_summary_new_range_multiplicities(tmp_path, capsys):
    import json

    from benchmark.kernel_tuning.gdn_prefill_port.cp2_iket import summarize

    names = ["producer.conv", "producer.raw_wait", "producer.z_load", "producer.epilogue_norm"]
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps({"traceEvents": [{"name": name, "ph": "X", "dur": 10} for name in names]})
    )
    summarize(path, 2)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    for row in rows:
        if "range" in row:
            assert row["mean_duration_per_chunk"] == 320
        elif row["role"] == "producer":
            assert row["wait_fraction"] == 0.25


@pytest.mark.parametrize("tma", [False, True])
def test_port_iket_balanced_range_multiplicities(tmp_path, capsys, tma):
    import json

    from benchmark.kernel_tuning.gdn_prefill_port.cp2_iket import summarize

    names = [
        "producer.k",
        "producer.q",
        "producer.v",
        "producer.y",
        "producer.gates",
        "producer.wait_o",
    ]
    if tma:
        names += [
            "producer.tma_issue",
            "producer.wait_raw_k",
            "producer.wait_raw_q",
            "producer.wait_raw_v",
        ]
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps({"traceEvents": [{"name": name, "ph": "X", "dur": 10} for name in names]})
    )
    summarize(path, 2)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    durations = {row["range"]: row["mean_duration_per_chunk"] for row in rows if "range" in row}
    expected = [10, 10, 10, 20, 10, 10, 30, 10, 10, 10] if tma else [20, 20, 20, 20, 10, 10]
    assert durations == dict(zip(names, expected, strict=True))


@pytest.mark.parametrize("misalignment", ["base", "stride"])
def test_port_rejects_projected_tma_misalignment_before_cuda(monkeypatch, misalignment):
    op, args = _port_cpu()
    rows, width = args[0].shape
    if misalignment == "base":
        args[0] = torch.empty(rows * width + 1, dtype=torch.bfloat16)[1:].view(rows, width)
        message = "projected TMA base must be 16-byte aligned"
    else:
        args[0] = torch.empty(rows, width + 1, dtype=torch.bfloat16)[:, :width]
        message = "projected TMA row stride must be a multiple of 8"

    def fail(*args, **kwargs):
        raise AssertionError("CUDA reached before projected validation")

    monkeypatch.setattr(torch.cuda, "device", fail)
    with pytest.raises(ValueError, match=message):
        op(*args)


def test_port_watchdog_specializes_cache():
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, compile_key

    normal, debug = PortOptions(), PortOptions(watchdog=True)
    assert normal.watchdog is False
    assert compile_key(16, 48, 16480, normal) != compile_key(16, 48, 16480, debug)
    with pytest.raises(TypeError, match="watchdog must be a bool"):
        PortOptions(watchdog=1)


@pytest.mark.parametrize(
    "value,expected", [("64", [64]), ("[16,100]", [16, 100]), ("16,100", [16, 100])]
)
def test_cp2_debug_lengths(value, expected):
    from benchmark.kernel_tuning.gdn_prefill_port.cp2_debug import parse_lengths

    assert parse_lengths(value) == expected


@pytest.mark.parametrize("value", ["[]", "0", "-1", "[true]", "[1.5]", "bad"])
def test_cp2_debug_invalid_lengths(value):
    import argparse

    from benchmark.kernel_tuning.gdn_prefill_port.cp2_debug import parse_lengths

    with pytest.raises(argparse.ArgumentTypeError):
        parse_lengths(value)


def test_cp2_debug_catches_asynchronous_trap(monkeypatch, capsys):
    from benchmark.kernel_tuning.gdn_prefill_port import cp2_debug

    # CPU stand-ins exercise the actual run_case -> synchronize -> error path.
    x = torch.empty(1, 2)
    meta = (torch.tensor([0, 1]), torch.tensor([1]), torch.tensor([True]))
    pools = (torch.zeros(3, 1), torch.zeros(3, 1))

    class Weights:
        def args(self):
            return ()

    monkeypatch.setattr(cp2_debug, "make_case", lambda *args: (x, Weights(), meta, pools))
    monkeypatch.setattr(cp2_debug, "PortPrefill", lambda *args, **kwargs: lambda *args: x)

    def synchronize():
        print("GDN_WATCHDOG tag=31 blk=0 warp=1 lane=0 stage=0 parity=0 bx=0")
        raise RuntimeError("CUDA error: an illegal instruction was encountered")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert cp2_debug.main(["--lengths", "64", "--lengths", "128"]) == 1
    captured = capsys.readouterr()
    assert "GDN_WATCHDOG tag=31" in captured.out
    assert "GDN_DEBUG_ERROR RuntimeError" in captured.err
    assert "lengths=[128]" not in captured.out


def test_precision_probe_options_and_tp2_geometries():
    from benchmark.kernel_tuning.gdn_prefill_port.cp2_precision import (
        GEOMETRIES,
        parse_cases,
        parse_geoms,
        parse_port_options,
    )

    assert parse_port_options("") == {}
    assert parse_port_options("cp=false,prefetch=4") == {"cp": False, "prefetch": 4}
    with pytest.raises(ValueError, match="unknown port option"):
        parse_port_options("precision=base")
    assert parse_geoms(["16x32,16x48", "8x16", "8x24,8x24"]) == list(GEOMETRIES)
    assert parse_cases(["8,24,heavy:1024"]) == [(8, 24, "heavy", (1024,))]
    with pytest.raises(ValueError):
        parse_geoms(["8x32"])


def test_precision_probe_contract_uses_mean_rms_and_worst_max():
    from benchmark.kernel_tuning.gdn_prefill_port.cp2_precision import aggregate

    seeds = [
        {
            "state": {
                "ours": {"rms_abs": rms, "max_abs": largest, "finite": True, "rms_pass": rms <= 1},
                "fla": {"rms_abs": 1.0, "max_abs": 4.0, "finite": True},
            }
        }
        for rms, largest in zip([1.2, 1, 1, 1, 1], [5, 3, 3, 3, 3], strict=True)
    ]
    result = aggregate(seeds, "state", "ours")
    contract = result["user_contract"]
    assert contract["rms_ratio_vs_fla"] == pytest.approx(1.04)
    assert contract["max_ratio_vs_fla"] == 1.25
    assert contract["pass"]
    assert result["mean_largest_pass"]
    assert not result["pass"]  # A single seed fails the frozen RMS rule.
    seeds[0]["state"]["ours"]["max_abs"] = 5.01
    assert not aggregate(seeds, "state", "ours")["user_contract"]["max_pass"]
    seeds[0]["state"]["ours"]["finite"] = False
    assert not aggregate(seeds, "state", "ours")["user_contract"]["pass"]


def test_port_compile_prefill_needs_no_cuda_device(monkeypatch):
    """Offline builds (scripts/compile_gdn_chunk_ws.py) pass fake tensors and an explicit
    --gpu-arch; the shared history cache must not query or initialize CUDA."""
    import types

    from ctikernels.gdn.chunk_ws import prefill

    def no_cuda(*a, **kw):
        raise AssertionError("compile_prefill queried CUDA")

    for name in ("current_device", "get_device_capability", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    compiled = []

    class Storage:
        def size_in_bytes(self):
            return 0

    class Kernel:
        def __init__(self, *a):
            self.shared_storage = Storage()

    def compile_stub(fn, *a, options):
        compiled.append((fn, options))
        return types.SimpleNamespace()

    cute = types.ModuleType("cutlass.cute")
    cute.compile = compile_stub
    cute.sym_int = lambda: 1
    cute.runtime = types.SimpleNamespace(make_fake_tensor=lambda *a, **kw: object())
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = cute
    for dtype in ("BFloat16", "Float32", "Int8", "Int32", "Uint8"):
        setattr(cutlass, dtype, type(dtype, (int,), {}))
    runtime = types.ModuleType("cutlass.runtime")
    runtime.make_fake_stream = lambda **kw: object()
    fused = types.ModuleType("ctikernels.gdn.chunk_ws._fused")
    fused.FusedChunkKernel = Kernel
    fused.launch_history = "launch_history"
    for name, module in (
        ("cutlass", cutlass),
        ("cutlass.cute", cute),
        ("cutlass.runtime", runtime),
        ("ctikernels.gdn.chunk_ws._fused", fused),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(prefill, "_COMMITS", {})
    flags = "--enable-tvm-ffi --gpu-arch sm_120a"
    width = prefill._contract.Geometry(16, 32).proj_width
    first = prefill.compile_prefill(16, 32, width, flags, variant="mn")
    second = prefill.compile_prefill(16, 32, width, flags, variant="seg")
    assert first[1] is second[1]
    assert [fn for fn, _ in compiled].count("launch_history") == 1
    assert list(prefill._COMMITS) == [(None, 16, 32, width, flags)]
