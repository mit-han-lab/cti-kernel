# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import ctikernels
import ctikernels.pynccl as pynccl
from ctikernels.utils import KERNEL_PATH

from ..support.child import run_child


@pytest.fixture(autouse=True)
def clear_nccl_module_cache():
    pynccl._load_nccl_module.cache_clear()
    yield
    pynccl._load_nccl_module.cache_clear()


@pytest.fixture
def load_aot(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    monkeypatch.setattr(pynccl, "load_aot", mock)
    return mock


def _package_spec(location: Path) -> importlib.machinery.ModuleSpec:
    spec = importlib.machinery.ModuleSpec("nvidia.nccl", loader=None, is_package=True)
    spec.submodule_search_locations = [str(location)]
    return spec


def test_nccl_runtime_wheel_links_its_soname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_aot: Mock
) -> None:
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libnccl.so.2").touch()
    monkeypatch.setattr(importlib.util, "find_spec", Mock(return_value=_package_spec(tmp_path)))

    assert pynccl._load_nccl_module() is load_aot.return_value

    load_aot.assert_called_once_with(
        "pynccl",
        cuda_files=["pynccl.cu"],
        extra_ldflags=[f"-L{library_dir}", f"-Wl,-rpath,{library_dir}", "-l:libnccl.so.2"],
    )


@pytest.mark.parametrize("wheel", ["no_nvidia", "no_nccl", "absent", "not_package", "no_soname"])
def test_nccl_without_runtime_wheel_uses_default_linker_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_aot: Mock, wheel: str
) -> None:
    find_spec = {
        "no_nvidia": Mock(side_effect=ModuleNotFoundError(name="nvidia")),
        "no_nccl": Mock(side_effect=ModuleNotFoundError(name="nvidia.nccl")),
        "absent": Mock(return_value=None),
        "not_package": Mock(return_value=importlib.machinery.ModuleSpec("nvidia.nccl", None)),
        "no_soname": Mock(return_value=_package_spec(tmp_path)),
    }[wheel]
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert pynccl._load_nccl_module() is load_aot.return_value

    load_aot.assert_called_once_with(
        "pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-l:libnccl.so.2"]
    )


def test_nccl_does_not_hide_unrelated_import_failure(
    monkeypatch: pytest.MonkeyPatch, load_aot: Mock
) -> None:
    error = ModuleNotFoundError(name="unrelated_dependency")
    monkeypatch.setattr(importlib.util, "find_spec", Mock(side_effect=error))

    with pytest.raises(ModuleNotFoundError) as raised:
        pynccl._load_nccl_module()

    assert raised.value is error
    load_aot.assert_not_called()


def test_nccl_module_discovery_and_build_are_cached(
    monkeypatch: pytest.MonkeyPatch, load_aot: Mock
) -> None:
    find_spec = Mock(return_value=None)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    assert pynccl._load_nccl_module() is pynccl._load_nccl_module()

    find_spec.assert_called_once_with("nvidia.nccl")
    load_aot.assert_called_once()


def test_declared_dtypes_are_exactly_the_native_dtype_map() -> None:
    source = (KERNEL_PATH / "src" / "pynccl.cu").read_text(encoding="utf-8")
    body = re.search(r"kNCCLDtypeMap\s*=.*?\{(.*?)\};", source, re.DOTALL)
    assert body is not None
    entries = re.findall(r"\{\{DLDataTypeCode::(\w+),\s*(\d+),\s*(\d+)\},\s*(nccl\w+)\}", body[1])
    native = {
        ("kDLInt", "8", "1", "ncclInt8"): torch.int8,
        ("kDLUInt", "8", "1", "ncclUint8"): torch.uint8,
        ("kDLFloat", "64", "1", "ncclFloat64"): torch.float64,
        ("kDLFloat8_e4m3fn", "8", "1", "ncclFloat8e4m3"): torch.float8_e4m3fn,
        ("kDLFloat8_e5m2", "8", "1", "ncclFloat8e5m2"): torch.float8_e5m2,
        ("kDLFloat", "16", "1", "ncclFloat16"): torch.float16,
        ("kDLBfloat", "16", "1", "ncclBfloat16"): torch.bfloat16,
        ("kDLFloat", "32", "1", "ncclFloat32"): torch.float32,
        ("kDLInt", "32", "1", "ncclInt32"): torch.int32,
        ("kDLInt", "64", "1", "ncclInt64"): torch.int64,
    }
    assert set(entries) == native.keys()
    assert pynccl.PYNCCL_DTYPES == tuple(native[entry] for entry in entries)
    assert ctikernels.PYNCCL_DTYPES is pynccl.PYNCCL_DTYPES


def test_declared_reduce_ops_are_exactly_the_native_op_map() -> None:
    source = (KERNEL_PATH / "src" / "pynccl.cu").read_text(encoding="utf-8")
    body = re.search(r"kNCCLReduceOPMap\s*=.*?\{(.*?)\};", source, re.DOTALL)
    assert body is not None
    entries = re.findall(r'\{"(\w+)",\s*::nccl\w+\}', body[1])
    assert set(entries) == {"sum", "prod", "max", "min", "avg"}
    assert pynccl.PYNCCL_REDUCE_OPS == tuple(entries)
    assert ctikernels.PYNCCL_REDUCE_OPS is pynccl.PYNCCL_REDUCE_OPS


def test_pynccl_module_preserves_lazy_torch_import() -> None:
    # Isolate pynccl: the package's unrelated MoE exports already import torch.
    run_child(
        [
            sys.executable,
            "-c",
            "import sys, types; "
            "package = types.ModuleType('ctikernels'); "
            "package.__path__ = [sys.argv[1]]; "
            "sys.modules['ctikernels'] = package; "
            "import ctikernels.pynccl as p; "
            "assert 'torch' not in sys.modules; "
            "assert p.PYNCCL_REDUCE_OPS == ('sum', 'prod', 'max', 'min', 'avg'); "
            "from ctikernels.pynccl import PYNCCL_DTYPES; "
            "import torch; "
            "assert PYNCCL_DTYPES == "
            "(torch.int8, torch.uint8, torch.int32, torch.int64, torch.float16, "
            "torch.float32, torch.float64, torch.bfloat16, "
            "torch.float8_e4m3fn, torch.float8_e5m2); "
            "assert PYNCCL_DTYPES is p.PYNCCL_DTYPES",
            str(KERNEL_PATH.parent),
        ],
        timeout=600,
        check=True,
    )


@pytest.mark.parametrize(
    ("code", "version"),
    [(2804, (2, 8, 4)), (22701, (2, 27, 1)), (22907, (2, 29, 7)), (23007, (2, 30, 7))],
)
def test_nccl_version_code_decoding(code: int, version: tuple[int, int, int]) -> None:
    assert pynccl._decode_nccl_version(code) == version


@pytest.mark.parametrize(
    ("code", "admitted"),
    [(2804, False), (22608, False), (22700, False), (22701, True), (23007, True), (30000, False)],
)
def test_nccl_runtime_version_admission(load_aot: Mock, code: int, admitted: bool) -> None:
    load_aot.return_value.nccl_runtime_version.return_value = code
    if admitted:
        assert pynccl.check_nccl_runtime_version() == pynccl._decode_nccl_version(code)
    else:
        with pytest.raises(RuntimeError, match="is not supported"):
            pynccl.check_nccl_runtime_version()


def test_init_pynccl_rejects_old_runtime_before_creating_an_id(
    monkeypatch: pytest.MonkeyPatch, load_aot: Mock
) -> None:
    load_aot.return_value.nccl_runtime_version.return_value = 22608
    wrapper = Mock()
    monkeypatch.setattr(pynccl, "_get_pynccl_wrapper_cls", wrapper)

    with pytest.raises(RuntimeError, match="NCCL runtime 2.26.8"):
        pynccl.init_pynccl(tp_rank=0, tp_size=2, tp_cpu_group=Mock())

    load_aot.return_value.create_nccl_uid.assert_not_called()
    wrapper.assert_not_called()


def _records():
    return [
        {
            "host": "same-host-digest",
            "pci_bus_id": f"0000:0{rank}:00.0",
            "capability": 120 - rank * 30,
            "device": rank,
            "visible": {"0000:00:00.0": 0, "0000:01:00.0": 1},
            "p2p": {"0000:00:00.0": rank == 1, "0000:01:00.0": rank == 0},
            "max_size_bytes": 4096,
            "force_window": False,
        }
        for rank in range(2)
    ]


@pytest.mark.parametrize(
    "case, expected",
    [
        ("p2p", ("symmetric", "CUDA P2P probe passed")),
        ("zero", ("in_place", "max_size_bytes=0")),
        ("single", ("in_place", "single rank")),
        ("hosts", ("in_place", "ranks on different hosts")),
        ("invisible", ("in_place", "peer cuda device not visible")),
        ("no_p2p", ("in_place", "no CUDA P2P between cuda:0 and cuda:1")),
        ("asymmetric", ("in_place", "no CUDA P2P between cuda:1 and cuda:0")),
        ("forced", ("symmetric", "forced")),
        ("forced_zero", ("in_place", "max_size_bytes=0")),
    ],
)
def test_window_probe(case, expected):
    records = _records()
    if case in ("zero", "forced_zero"):
        for record in records:
            record["max_size_bytes"] = 0
    if case == "single":
        records = records[:1]
    if case == "hosts":
        records[1]["host"] = "different-host-digest"
    if case == "invisible":
        records[1]["visible"].pop(records[0]["pci_bus_id"])
    if case in ("no_p2p", "forced", "forced_zero"):
        records[0]["p2p"][records[1]["pci_bus_id"]] = False
    if case == "asymmetric":
        records[1]["p2p"][records[0]["pci_bus_id"]] = False
    if case in ("forced", "forced_zero"):
        for record in records:
            record["force_window"] = True
    assert pynccl._window_decision(records) == expected


@pytest.mark.parametrize("field,value", [("max_size_bytes", 0), ("force_window", True)])
def test_window_probe_requires_identical_settings(field, value):
    records = _records()
    records[1][field] = value
    with pytest.raises(RuntimeError, match="must agree"):
        pynccl._window_decision(records)


def test_window_probe_invalid_size():
    records = _records()
    for record in records:
        record["max_size_bytes"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        pynccl._window_decision(records)
    with pytest.raises(ValueError, match="at least one rank"):
        pynccl._window_decision([])


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_init_passes_communicator_minimum_and_probe_decision(monkeypatch, load_aot, force, rank):
    records = _records()
    records[1]["capability"] = 80
    records[0]["p2p"][records[1]["pci_bus_id"]] = False
    for record in records:
        record["force_window"] = force
    load_aot.return_value.nccl_runtime_version.return_value = 22701
    load_aot.return_value.create_nccl_uid.return_value = "test-uid"
    local_probe = Mock(return_value=records[rank])
    monkeypatch.setattr(pynccl, "_local_probe_record", local_probe)
    wrapper = Mock()
    monkeypatch.setattr(pynccl, "_get_pynccl_wrapper_cls", Mock(return_value=wrapper))
    group = Mock()

    def gather(output, value, *, group):
        output[:] = records

    def broadcast(output, *, src, group):
        assert src == 7  # CPU group rank 0 need not be global rank 0.
        output[:] = ["test-uid"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(torch.distributed, "get_global_rank", Mock(return_value=7))
    assert (
        pynccl.init_pynccl(
            tp_rank=rank, tp_size=2, tp_cpu_group=group, max_size_bytes=4096, _force_window=force
        )
        is wrapper.return_value
    )
    local_probe.assert_called_once_with(load_aot.return_value, 4096, force)
    wrapper.assert_called_once_with(
        rank, 2, 4096, "test-uid", 80, *pynccl._window_decision(records), False
    )


def test_stub_reduction_signatures_and_literal():
    import ast
    from typing import get_args

    assert get_args(pynccl.ReduceOp) == pynccl.PYNCCL_REDUCE_OPS
    tree = ast.parse(Path(pynccl.__file__).read_text())
    stub = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PyNCCLCommunicator"
    )
    methods = {n.name: n for n in stub.body if isinstance(n, ast.FunctionDef)}
    for name in ("all_reduce", "reduce_scatter"):
        op = next(a for a in methods[name].args.args if a.arg == "op")
        assert ast.unparse(op.annotation) == "ReduceOp"
    assert {
        "broadcast",
        "all_gather",
        "window_mode",
        "window_reason",
        "window_bytes",
        "get_buffer",
    } <= methods.keys()


@pytest.mark.parametrize("dtype,code", [(torch.float8_e4m3fn, 10), (torch.float8_e5m2, 12)])
def test_torch_fp8_ffi_dlpack_codes_on_cpu(dtype, code):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    tensor = tvm_ffi.from_dlpack(torch.empty(3, dtype=dtype))
    assert (tensor.dtype.type_code, tensor.dtype.bits, tensor.dtype.lanes) == (code, 8, 1)


def test_local_probe_uses_pci_identity_with_local_device_ordinals(monkeypatch):
    import hashlib
    import socket

    module = Mock()
    module.device_pci_bus_id.side_effect = lambda index: ("0000:AB:00.0", "0000:CD:00.0")[index]
    monkeypatch.setattr(socket, "gethostname", Mock(return_value="test-host-identity"))
    monkeypatch.setattr(Path, "read_text", Mock(return_value="test-boot-identity\n"))
    monkeypatch.setattr(torch.cuda, "current_device", Mock(return_value=1))
    monkeypatch.setattr(torch.cuda, "device_count", Mock(return_value=2))
    monkeypatch.setattr(torch.cuda, "get_device_capability", Mock(return_value=(12, 0)))
    peer_access = Mock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", peer_access)
    record = pynccl._local_probe_record(module, 4096, False)
    assert record["host"] == hashlib.sha256(b"test-host-identity:test-boot-identity").hexdigest()
    assert record["pci_bus_id"] == "0000:cd:00.0"
    assert record["capability"] == 120
    assert record["visible"] == {"0000:ab:00.0": 0, "0000:cd:00.0": 1}
    assert record["p2p"] == {"0000:ab:00.0": True, "0000:cd:00.0": False}
    peer_access.assert_called_once_with(1, 0)


def test_window_probe_handles_different_visible_device_order():
    records = _records()
    records[1]["device"] = 0
    records[1]["visible"] = {records[1]["pci_bus_id"]: 0, records[0]["pci_bus_id"]: 1}
    assert pynccl._window_decision(records) == ("symmetric", "CUDA P2P probe passed")


def test_report_benchmark_help_does_not_need_cuda():
    root = Path(__file__).resolve().parents[2]
    result = run_child(
        [sys.executable, "-m", "benchmark.collective.pynccl", "--help"],
        cwd=root,
        timeout=600,
        check=True,
    )
    assert "--rounds" in result.stdout and "--calls" in result.stdout and "--out" in result.stdout


def test_window_strings_are_plain_python_strings():
    source = (Path(pynccl.__file__).parent / "csrc/src/pynccl.cu").read_text()
    assert '.def("_window_mode"' in source and '.def("_window_reason"' in source
    tree = ast.parse(Path(pynccl.__file__).read_text())
    impl = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PyNCCLImpl")
    names = {n.name for n in impl.body if isinstance(n, ast.FunctionDef)}
    assert {"window_mode", "window_reason"} <= names
