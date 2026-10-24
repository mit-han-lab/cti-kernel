# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kernel/utils.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    return "ctiengine__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@functools.cache
def _cuda_identity() -> tuple[str, str]:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = str(pathlib.Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
    version_file = pathlib.Path(cuda_home) / "version.json"
    toolkit = (
        version_file.read_text()
        if version_file.is_file()
        else _command_output([str(pathlib.Path(cuda_home) / "bin/nvcc"), "--version"])
    )
    target = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if target is None:
        output = _command_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
        target = output.splitlines()[0] if output else "unknown"
    return toolkit, target


@functools.cache
def _compiler_identity(command: str) -> tuple[str, str, str]:
    # The full command (tvm-ffi compiles with its arguments, e.g. ABI defines), the
    # resolved executable and the first line of its --version output, so changed
    # arguments, a compiler upgrade or a different compiler on PATH change the key.
    argv = shlex.split(command) or [command]
    resolved = shutil.which(argv[0])
    path = os.path.realpath(resolved) if resolved else "unresolved:" + argv[0]
    output = _command_output([*argv, "--version"])
    return command, path, output.splitlines()[0] if output else "unknown"


@functools.cache
def _toolchain_identity(has_cuda: bool) -> tuple:
    try:
        import tvm_ffi

        ffi_version = tvm_ffi.__version__
    except ImportError:
        ffi_version = "none"
    identity = (
        ffi_version,
        sys.implementation.cache_tag,
        platform.machine(),
        _compiler_identity(os.environ.get("CXX", "c++")),
        _compiler_identity(os.environ.get("CC", "cc")),
    )
    if not has_cuda:
        return identity
    # nvcc is identified by the toolkit (version.json); its default host compiler is g++.
    return identity + (_compiler_identity("g++"),) + _cuda_identity()


def _module_key(
    kind: str,
    name: str,
    sources: dict[str, list[pathlib.Path]],
    wrappers: dict[str, list[Tuple[str, str]]],
    cflags: list[str],
    cuda_cflags: list[str],
    ldflags: list[str],
    include_paths: list[str],
    has_cuda: bool,
) -> str:
    kernel_path = KERNEL_PATH.resolve()
    roots = [pathlib.Path(path).resolve() for path in include_paths]

    def file_id(path: pathlib.Path, root_index: int) -> tuple[str, str]:
        if path.is_relative_to(kernel_path):
            return "<csrc>", path.relative_to(kernel_path).as_posix()
        return f"<inc{root_index}>", path.relative_to(roots[root_index]).as_posix()

    def include_file(
        path: pathlib.Path, quoted: bool, include: str
    ) -> tuple[pathlib.Path, tuple[str, str]] | None:
        candidates = [(path.parent / include, None)] if quoted else []
        candidates += [(root / include, i) for i, root in enumerate(roots)]
        for unresolved, index in candidates:
            candidate = unresolved.resolve()
            if not candidate.is_file():
                continue
            if index is not None and candidate.is_relative_to(roots[index]):
                return candidate, file_id(candidate, index)
            for i, root in enumerate(roots):
                if candidate.is_relative_to(root):
                    return candidate, file_id(candidate, i)
            if candidate.is_relative_to(kernel_path):  # e.g. a header beside a jit source
                return candidate, file_id(candidate, 0)
            return None  # Resolved outside the allowed include roots.
        return None

    source_ids = {
        language: [file_id(path.resolve(), 0) for path in paths]
        for language, paths in sources.items()
    }
    metadata = (
        "ctikernels-jit-v1",
        kind,
        name,
        source_ids,
        wrappers,
        [
            [flag.replace(str(KERNEL_PATH), "<csrc>") for flag in flags]
            for flags in (cflags, cuda_cflags, ldflags)
        ],
        [
            "<csrc>/" + root.relative_to(kernel_path).as_posix()
            if root.is_relative_to(kernel_path)
            else f"<inc{i}>"
            for i, root in enumerate(roots)
        ],
        _toolchain_identity(has_cuda),
    )
    digest = hashlib.sha256()

    def add(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    add(json.dumps(metadata, sort_keys=True).encode())
    pending = [
        (path.resolve(), file_id(path.resolve(), 0)) for paths in sources.values() for path in paths
    ]
    files = {}
    while pending:
        path, identity = pending.pop()
        if identity in files:
            continue
        contents = path.read_bytes()
        files[identity] = contents
        for match in re.finditer(rb'^\s*#\s*include\s*[<"]([^>"\r\n]+)[>"]', contents, re.M):
            included = include_file(path, b'"' in match[0], os.fsdecode(match[1]))
            if included is not None:
                pending.append(included)
    for identity, contents in sorted(files.items()):
        add(json.dumps(identity).encode())
        add(contents)
    return digest.hexdigest()


_CUDA_ARCH_LOCK = threading.Lock()


def _build_module(
    kind: str, name: str, build_dir: pathlib.Path, cuda_arch: str | None = None, **kwargs
) -> Module:
    from tvm_ffi.cpp import load, load_inline

    builder = load_inline if kind == "jit" else load
    # tvm-ffi always adds one -gencode, from this variable or the detected GPU, and takes
    # no argument for it; an arch-specific target (e.g. "10.3a") must replace it rather
    # than join it. Builds hold the lock so none sees another's temporary value.
    with _CUDA_ARCH_LOCK:
        if cuda_arch is None:
            return builder(name, build_directory=str(build_dir), **kwargs)
        previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = cuda_arch
        try:
            return builder(name, build_directory=str(build_dir), **kwargs)
        finally:
            if previous is None:
                del os.environ["TVM_FFI_CUDA_ARCH_LIST"]
            else:
                os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous


def _load_built(so_path: pathlib.Path) -> Module:
    import tvm_ffi

    return tvm_ffi.load_module(str(so_path))


def _load_cached(kind: str, name: str, key: str, build_directory: str | None, **kwargs) -> Module:
    directory = (
        pathlib.Path(build_directory)
        if build_directory is not None
        else pathlib.Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
        / f"{name}_{key[:16]}"
    )
    so_path = directory / f"{name}.so"
    marker = directory / "ctikernels.key"
    rebuild = os.environ.get("CTIKERNELS_JIT_REBUILD") == "1"

    def ready() -> bool:
        if rebuild or not so_path.is_file():
            return False
        try:
            return marker.read_bytes() == key.encode()
        except FileNotFoundError:
            return False

    if ready():
        return _load_built(so_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ctikernels.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            return _load_built(so_path)
        # Never let a stale binary receive the new key: ninja decides by mtimes,
        # so drop the old outputs and its state before building.
        marker.unlink(missing_ok=True)
        for stale in (so_path, directory / ".ninja_log", directory / ".ninja_deps"):
            stale.unlink(missing_ok=True)
        for stale in (*directory.glob("*.o"), *directory.glob("*.o.d")):
            stale.unlink(missing_ok=True)
        module = _build_module(kind, name, directory, **kwargs)
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as temporary:
            temporary_path = pathlib.Path(temporary.name)
            try:
                temporary.write(key)
                temporary.close()
                os.replace(temporary_path, marker)
            finally:
                temporary_path.unlink(missing_ok=True)
        return module


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    name = _make_name(*args)
    key = _module_key(
        "aot",
        name,
        {
            "cpp": [pathlib.Path(f) for f in cpp_files],
            "cuda": [pathlib.Path(f) for f in cuda_files],
        },
        {},
        DEFAULT_CFLAGS + extra_cflags,
        DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        DEFAULT_LDFLAGS + extra_ldflags,
        DEFAULT_INCLUDE + extra_include_paths,
        bool(cuda_files),
    )
    return _load_cached(
        "aot",
        name,
        key,
        build_directory,
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
    cuda_arch: str | None = None,
) -> Module:
    """Load a content-keyed module without ninja on cache hits.

    Keys ignore install paths and mtimes. Set CTIKERNELS_JIT_REBUILD=1 to
    force the build path, including in an explicit build_directory.
    ``cuda_arch`` (e.g. ``"10.3a"``) replaces tvm-ffi's own target; name it in
    ``args`` too, since it is not part of the key.
    """

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    name = _make_name(*args)
    key = _module_key(
        "jit",
        name,
        {"cpp": cpp_paths, "cuda": cuda_paths},
        {"cpp": cpp_wrappers, "cuda": cuda_wrappers},
        DEFAULT_CFLAGS + extra_cflags,
        DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        DEFAULT_LDFLAGS + extra_ldflags,
        DEFAULT_INCLUDE + extra_include_paths,
        bool(cuda_sources),
    )
    return _load_cached(
        "jit",
        name,
        key,
        build_directory,
        cuda_arch=cuda_arch,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
    )
