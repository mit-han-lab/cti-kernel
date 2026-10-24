# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kernel/pynccl.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from .utils import load_aot

ReduceOp = Literal["sum", "prod", "max", "min", "avg"]


if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    PYNCCL_DTYPES: tuple[torch.dtype, ...]

    class PyNCCLCommunicator:
        """Collectives over contiguous CUDA tensors of the ``PYNCCL_DTYPES``."""

        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: ReduceOp) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def broadcast(self, tensor: torch.Tensor, root: int) -> None: ...
        @abstractmethod
        def reduce_scatter(
            self, output: torch.Tensor, input: torch.Tensor, op: ReduceOp
        ) -> None: ...
        @abstractmethod
        def window_mode(self) -> Literal["symmetric", "in_place"]: ...
        @abstractmethod
        def window_reason(self) -> str: ...
        @abstractmethod
        def window_bytes(self) -> int: ...
        @abstractmethod
        def get_buffer(self) -> int: ...
        @abstractmethod
        def device(self) -> int: ...

else:
    PyNCCLCommunicator = Any


PYNCCL_REDUCE_OPS = ("sum", "prod", "max", "min", "avg")


def __getattr__(name: str):
    if name == "PYNCCL_DTYPES":
        import torch

        # Match kNCCLDtypeMap without importing torch until callers need the tuple.
        dtypes = (
            torch.int8,
            torch.uint8,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        globals()[name] = dtypes
        return dtypes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def _load_nccl_module() -> Module:
    try:
        spec = importlib.util.find_spec("nvidia.nccl")
    except ModuleNotFoundError as error:
        if error.name not in {"nvidia", "nvidia.nccl"}:
            raise
        spec = None
    ldflags = []
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            library_dir = Path(location) / "lib"
            if (library_dir / "libnccl.so.2").is_file():
                ldflags.extend([f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"])
    # Runtime wheels provide the SONAME, not the development libnccl.so symlink.
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=[*ldflags, "-l:libnccl.so.2"])


# The bundled header is NCCL 2.27.1 (the first release with every entry point the
# extension calls); NCCL keeps its ABI within major version 2.
MIN_NCCL_VERSION = (2, 27, 1)
_NEXT_NCCL_MAJOR = 3


def _decode_nccl_version(code: int) -> tuple[int, int, int]:
    # NCCL_VERSION(X, Y, Z): X*1000 + Y*100 + Z up to 2.8, X*10000 + Y*100 + Z after.
    if code < 10000:
        return code // 1000, code % 1000 // 100, code % 100
    return code // 10000, code % 10000 // 100, code % 100


def nccl_runtime_version() -> tuple[int, int, int]:
    """The (major, minor, patch) of the libnccl.so.2 the extension loaded."""
    return _decode_nccl_version(int(_load_nccl_module().nccl_runtime_version()))


def check_nccl_runtime_version() -> tuple[int, int, int]:
    """Raise RuntimeError unless MIN_NCCL_VERSION <= runtime < 3.0.0."""
    version = nccl_runtime_version()
    if not MIN_NCCL_VERSION <= version < (_NEXT_NCCL_MAJOR, 0, 0):
        found = ".".join(map(str, version))
        need = ".".join(map(str, MIN_NCCL_VERSION))
        raise RuntimeError(
            f"NCCL runtime {found} is not supported: need >= {need} and < {_NEXT_NCCL_MAJOR}"
        )
    return version


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("ctiengine.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

        # Plain str: an FFI string holds a process-local handle, so one pickled into
        # another process (a benchmark report, say) crashes when it is freed there.
        def window_mode(self) -> str:
            return str(self._window_mode())

        def window_reason(self) -> str:
            return str(self._window_reason())

    return PyNCCLImpl


class _ProbeRecord(TypedDict):
    host: str
    pci_bus_id: str
    capability: int
    device: int
    visible: dict[str, int]
    p2p: dict[str, bool]
    max_size_bytes: int
    force_window: bool


def _window_decision(records: list[_ProbeRecord]) -> tuple[str, str]:
    """Pure, communicator-wide necessary-condition CUDA P2P probe decision."""
    if not records:
        raise ValueError("window probe needs at least one rank")
    sizes = {r["max_size_bytes"] for r in records}
    forces = {r["force_window"] for r in records}
    if len(sizes) != 1 or len(forces) != 1:
        raise RuntimeError("max_size_bytes and _force_window must agree on every rank")
    size = records[0]["max_size_bytes"]
    if size < 0:
        raise ValueError("max_size_bytes must be nonnegative")
    if size == 0:
        return "in_place", "max_size_bytes=0"
    if records[0]["force_window"]:
        return "symmetric", "forced"
    if len(records) == 1:
        return "in_place", "single rank"
    if len({r["host"] for r in records}) != 1:
        return "in_place", "ranks on different hosts"
    for rank, record in enumerate(records):
        for peer, remote in enumerate(records):
            if peer == rank:
                continue
            bus = remote["pci_bus_id"]
            if bus not in record["visible"]:
                return "in_place", "peer cuda device not visible"
            if not record["p2p"].get(bus, False):
                return "in_place", (
                    f"no CUDA P2P between cuda:{record['device']} and cuda:{record['visible'][bus]}"
                )
    return "symmetric", "CUDA P2P probe passed"


def _local_probe_record(module, max_size_bytes: int, force_window: bool) -> _ProbeRecord:
    import hashlib
    import socket

    import torch

    device = torch.cuda.current_device()
    # Identity is exchanged only in memory; neither the hostname nor boot ID is logged.
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    host = hashlib.sha256(f"{socket.gethostname()}:{boot_id}".encode()).hexdigest()
    visible = {
        str(module.device_pci_bus_id(index)).lower(): index
        for index in range(torch.cuda.device_count())
    }
    major, minor = torch.cuda.get_device_capability(device)
    return {
        "host": host,
        "pci_bus_id": str(module.device_pci_bus_id(device)).lower(),
        "capability": major * 10 + minor,
        "device": device,
        "visible": visible,
        "p2p": {
            bus: index != device and torch.cuda.can_device_access_peer(device, index)
            for bus, index in visible.items()
        },
        "max_size_bytes": max_size_bytes,
        "force_window": force_window,
    }


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
    _force_window: bool = False,
    _test_null_window: bool = False,
) -> PyNCCLCommunicator:
    """Create one communicator for all ``PYNCCL_DTYPES`` on the current CUDA device.

    All four collectives run on the caller's current stream and are graph-capturable.
    Tensors must be contiguous, on this device, and aligned to their element size.
    Validation precedes stream resolution and enqueue; empty calls do no NCCL work.

    ``all_reduce(input, op)`` reduces in place. ``reduce_scatter(output, input, op)``
    reduces equal chunks: input.numel() == tp_size * output.numel(). Both accept
    sum/prod/max/min/avg (``PYNCCL_REDUCE_OPS``). Integer avg is rejected; sum and
    divide explicitly instead. FP8 reductions require every rank's capability >= 9.0.
    ``all_gather(output, input)`` requires output.numel() == tp_size * input.numel().
    Both two-tensor calls require matching dtypes and no overlap, except when the
    smaller tensor is exactly this rank's chunk of the larger tensor.
    ``broadcast(tensor, root)`` copies in place from a group rank in [0, tp_size).
    Gathers and broadcasts copy bits, including FP8 on devices below sm90. Native
    uint8 broadcast is preferred to the supported padded 16-bit all-gather emulation.

    ``max_size_bytes`` is the requested staging window for small all-reduces; 0
    allocates nothing. Initialization exchanges host identity, PCI bus IDs, device
    visibility, CUDA peer access and capability over ``tp_cpu_group``. A window is
    allocated only for multiple ranks on one host with every peer visible and P2P
    accessible in every process. This is a necessary-condition CUDA P2P probe:
    NCCL 2.27-2.30 exposes no public symmetric-kernel support query. A passing probe
    does not guarantee NCCL will choose symmetric kernels. The window registration
    is the final probe: NCCL returns a null window when symmetric memory is
    unsupported on the communicator, and then the buffer is freed and the mode
    falls back to ``"in_place"`` with that reason. Otherwise all-reduce calls
    ncclAllReduce directly in place with no staging copies.

    ``window_mode()``, ``window_reason()``, and ``window_bytes()`` report the decision
    and allocated size; ``get_buffer()`` returns 0/null without a window.
    Benchmark-only ``_force_window=True`` bypasses the probe for a positive size,
    reports reason "forced", and allows timing staging on pairs without CUDA P2P.
    Window size and force setting must agree on every rank. Test-only
    ``_test_null_window=True`` treats the registration result as null.
    """
    import torch

    module = _load_nccl_module()
    # Reject the library on every rank before any CPU group exchange.
    check_nccl_runtime_version()
    cls = _get_pynccl_wrapper_cls()
    record = _local_probe_record(module, max_size_bytes, _force_window)
    records = [None] * tp_size
    torch.distributed.all_gather_object(records, record, group=tp_cpu_group)
    mode, reason = _window_decision(records)
    min_capability = min(r["capability"] for r in records)

    id_list = [module.create_nccl_uid() if tp_rank == 0 else None]
    torch.distributed.broadcast_object_list(
        id_list,
        src=torch.distributed.get_global_rank(tp_cpu_group, 0),
        group=tp_cpu_group,
    )
    nccl_id = id_list[0]
    assert nccl_id is not None, f"Failed to get NCCL unique ID on {tp_rank = }"
    return cls(
        tp_rank,
        tp_size,
        max_size_bytes,
        nccl_id,
        min_capability,
        mode,
        reason,
        _test_null_window,
    )
