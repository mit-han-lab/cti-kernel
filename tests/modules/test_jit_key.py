# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import importlib.util
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python/ctikernels"


def import_utils(path, name):
    spec = importlib.util.spec_from_file_location(name, path / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def copies(tmp_path, monkeypatch):
    modules = []
    for label in ("a", "b"):
        path = tmp_path / label / "ctikernels"
        path.mkdir(parents=True)
        shutil.copy2(PACKAGE / "utils.py", path)
        shutil.copytree(PACKAGE / "csrc", path / "csrc")
        if label == "b":
            future = time.time() + 10 * 365 * 24 * 3600
            for file in path.rglob("*"):
                if file.is_file():
                    os.utime(file, (future, future))
        module = import_utils(path, f"jit_utils_{label}")
        monkeypatch.setattr(module, "_toolchain_identity", lambda has_cuda: ("test-toolchain",))
        modules.append(module)
    monkeypatch.delenv("CTIKERNELS_JIT_REBUILD", raising=False)
    return modules


def module_call(module, case):
    if case == "radix":
        return "aot", ("radix",), {"cpp_files": ["radix.cpp"]}
    if case == "request_local_sampling":
        return (
            "jit",
            (case, "0.7.0.post1"),
            {
                "cuda_files": [f"{case}.cu"],
                "cuda_wrappers": [("launch", "launch_request_sampling")],
            },
        )
    args = (
        module.make_cpp_args(2048, 1, *module.KernelConfig(128, 1, False))
        if case == "index"
        else module.make_cpp_args(2048, *module.KernelConfig(256, 1, True))
    )
    kernel = "IndexKernel" if case == "index" else "StoreKernel"
    return (
        "jit",
        (case, *args),
        {"cuda_files": [f"{case}.cu"], "cuda_wrappers": [("launch", f"{kernel}<{args}>::run")]},
    )


def key(module, case="index", **overrides):
    kind, args, kwargs = module_call(module, case)
    kwargs.update(overrides)
    return module._module_key(
        kind,
        module._make_name(*args),
        {
            language: [
                module.KERNEL_PATH / ("jit" if kind == "jit" else "src") / filename
                for filename in kwargs.get(f"{language}_files", [])
            ]
            for language in ("cpp", "cuda")
        },
        {language: kwargs.get(f"{language}_wrappers", []) for language in ("cpp", "cuda")}
        if kind == "jit"
        else {},
        module.DEFAULT_CFLAGS + kwargs.get("extra_cflags", []),
        module.DEFAULT_CUDA_CFLAGS + kwargs.get("extra_cuda_cflags", []),
        module.DEFAULT_LDFLAGS + kwargs.get("extra_ldflags", []),
        module.DEFAULT_INCLUDE + kwargs.get("extra_include_paths", []),
        bool(kwargs.get("cuda_files") or kwargs.get("cuda_wrappers")),
    )


@pytest.mark.parametrize("case", ["request_local_sampling", "index", "store", "radix"])
def test_key_ignores_install_path_and_future_mtimes(copies, case, tmp_path):
    keys = []
    for i, module in enumerate(copies):
        include = tmp_path / f"extra{i}"
        include.mkdir()
        (include / "unused.h").write_text("// unused\n")
        keys.append(
            key(
                module,
                case,
                extra_include_paths=[str(include)],
                extra_cflags=[f'-DROOT="{module.KERNEL_PATH}"'],
            )
        )
    assert keys[0] == keys[1]
    assert len(keys[0]) == 64
    int(keys[0], 16)


@pytest.mark.parametrize("case", ["request_local_sampling", "index", "store", "radix"])
@pytest.mark.parametrize("explicit_directory", [False, True])
def test_load_reuses_module_across_copies(copies, tmp_path, monkeypatch, case, explicit_directory):
    cache = tmp_path / "cache"
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(cache))
    built, loaded, calls, loads = object(), object(), [], []

    def build(kind, name, directory, **kwargs):
        calls.append((kind, name, directory, kwargs))
        (directory / f"{name}.so").write_bytes(b"dummy module")
        return built

    def load(path):
        loads.append(path)
        return loaded

    for module in copies:
        monkeypatch.setattr(module, "_build_module", build)
        monkeypatch.setattr(module, "_load_built", load)
    a, b = copies
    kind, args, kwargs = module_call(a, case)
    directory = (
        tmp_path / "explicit"
        if explicit_directory
        else cache / f"{a._make_name(*args)}_{key(a, case)[:16]}"
    )
    if explicit_directory:
        kwargs["build_directory"] = str(directory)
    assert getattr(a, f"load_{kind}")(*args, **kwargs) is built
    assert getattr(b, f"load_{kind}")(*args, **kwargs) is loaded
    assert getattr(a, f"load_{kind}")(*args, **kwargs) is loaded
    assert len(calls) == 1
    assert calls[0][:3] == (kind, a._make_name(*args), directory)
    assert loads == [directory / f"{a._make_name(*args)}.so"] * 2
    assert (directory / "ctikernels.key").read_text() == key(a, case)
    assert (directory / "ctikernels.lock").exists()
    assert not (directory / "lock").exists()
    build_kwargs = calls[0][3]
    assert build_kwargs["extra_cflags"] == a.DEFAULT_CFLAGS
    assert build_kwargs["extra_cuda_cflags"] == a.DEFAULT_CUDA_CFLAGS
    assert build_kwargs["extra_ldflags"] == a.DEFAULT_LDFLAGS
    assert build_kwargs["extra_include_paths"] == a.DEFAULT_INCLUDE
    if kind == "jit":
        assert build_kwargs["cuda_sources"] == [
            f'#include "{(a.KERNEL_PATH / "jit" / f"{case}.cu").resolve()}"',
            a._make_wrapper(kwargs["cuda_wrappers"][0]),
        ]
    else:
        assert build_kwargs["cpp_files"] == [str((a.KERNEL_PATH / "src/radix.cpp").resolve())]

    (directory / "ctikernels.key").write_bytes(b"\xffcorrupt")
    assert getattr(b, f"load_{kind}")(*args, **kwargs) is built
    assert len(calls) == 2
    assert (directory / "ctikernels.key").read_text() == key(b, case)
    monkeypatch.setenv("CTIKERNELS_JIT_REBUILD", "1")
    assert getattr(a, f"load_{kind}")(*args, **kwargs) is built
    assert len(calls) == 3
    monkeypatch.delenv("CTIKERNELS_JIT_REBUILD")
    (directory / f"{a._make_name(*args)}.so").unlink()
    assert getattr(a, f"load_{kind}")(*args, **kwargs) is built
    assert len(calls) == 4


def test_header_source_flags_and_wrappers_invalidate_key(copies):
    a, b = copies
    original = key(a)
    header = b.KERNEL_PATH / "include/ctiengine/tensor.h"
    with header.open("a") as file:
        file.write("\n// changed header\n")
    assert key(b) != original
    for flags in ("extra_cflags", "extra_cuda_cflags", "extra_ldflags"):
        assert key(a, **{flags: ["-DCHANGED"]}) != original
    assert key(a, cuda_wrappers=[("launch", "AnotherKernel::run")]) != original
    with (a.KERNEL_PATH / "jit/index.cu").open("a") as file:
        file.write("\n// changed source\n")
    assert key(a) != original


def test_transitive_includes_resolution_and_scope(copies, tmp_path):
    module = copies[0]
    include = tmp_path / "extra"
    nested = include / "nested"
    nested.mkdir(parents=True)
    source = module.KERNEL_PATH / "jit/synthetic.cu"
    source.write_text("#if 0\n#include <nested/outer.h>\n#endif\n")
    outer = nested / "outer.h"
    outer.write_text('#include "inner.h"\n#include <fallback.h>\n#include "../../outside.h"\n')
    inner = nested / "inner.h"
    inner.write_text('#include "outer.h"\n// cycle\n')
    fallback = include / "fallback.h"
    fallback.write_text("// reached via include root\n")
    shadow = include / "inner.h"
    shadow.write_text("// shadowed by relative header\n")
    outside = tmp_path / "outside.h"
    outside.write_text("// outside allowed roots\n")

    def current():
        return key(module, cuda_files=[source.name], extra_include_paths=[str(include)])

    original = current()
    inner.write_text(inner.read_text() + "// changed transitive header\n")
    assert current() != original
    original = current()
    fallback.write_text("// changed fallback header\n")
    assert current() != original
    original = current()
    shadow.write_text("// ignored edit\n")
    outside.write_text("// ignored edit\n")
    assert current() == original


def test_recheck_under_lock_and_failed_build(copies, tmp_path, monkeypatch):
    module = copies[0]
    directory = tmp_path / "build"
    sentinel = object()

    def another_builder_finished(fd, operation):
        assert operation == module.fcntl.LOCK_EX
        (directory / "test.so").write_bytes(b"dummy")
        (directory / "ctikernels.key").write_text("new-key")

    monkeypatch.setattr(module.fcntl, "flock", another_builder_finished)
    monkeypatch.setattr(module, "_load_built", lambda path: sentinel)
    assert module._load_cached("jit", "test", "new-key", str(directory)) is sentinel

    def fail(*args, **kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(module, "_build_module", fail)
    monkeypatch.setenv("CTIKERNELS_JIT_REBUILD", "1")
    with pytest.raises(RuntimeError, match="build failed"):
        module._load_cached("jit", "test", "new-key", str(directory))
    assert not (directory / "ctikernels.key").exists()


def test_toolchain_identity_is_lazy_and_cached(tmp_path, monkeypatch):
    module = import_utils(PACKAGE, "jit_utils_toolchain")
    monkeypatch.setitem(sys.modules, "tvm_ffi", None)
    monkeypatch.setenv("CXX", "test-cxx")
    monkeypatch.setenv("CC", "test-cc")
    calls = []

    def output(command):
        calls.append(command)
        return "nvcc test" if command[-1] == "--version" else "12.0\n12.1"

    monkeypatch.setattr(module, "_command_output", output)
    monkeypatch.setattr(module, "_compiler_identity", lambda command: (command, "v"))
    assert module._toolchain_identity(False) == (
        "none",
        sys.implementation.cache_tag,
        module.platform.machine(),
        ("test-cxx", "v"),
        ("test-cc", "v"),
    )
    assert calls == []
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(__version__="test-ffi"))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("TVM_FFI_CUDA_ARCH_LIST", raising=False)
    (tmp_path / "version.json").write_text('{"cuda": "test"}')
    identity = module._toolchain_identity(True)
    assert identity[0] == "test-ffi"
    assert identity[-3] == ("g++", "v")
    assert identity[-2:] == ('{"cuda": "test"}', "12.0")
    assert module._toolchain_identity(True) == identity
    assert len(calls) == 1
    (tmp_path / "version.json").unlink()
    module._cuda_identity.cache_clear()
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "12.0;12.1")
    assert module._cuda_identity() == ("nvcc test", "12.0;12.1")
    assert calls[-1] == [str(tmp_path / "bin/nvcc"), "--version"]
    module._cuda_identity.cache_clear()
    monkeypatch.delenv("TVM_FFI_CUDA_ARCH_LIST")
    monkeypatch.setattr(module, "_command_output", lambda command: "")
    assert module._cuda_identity()[1] == "unknown"


def test_cache_root_is_read_at_call_time(copies, tmp_path, monkeypatch):
    module = copies[0]
    directories = []

    def build(kind, name, directory, **kwargs):
        directories.append(directory)
        (directory / f"{name}.so").write_bytes(b"dummy")
        return object()

    monkeypatch.setattr(module, "_build_module", build)
    for root in ("cache-a", "cache-b"):
        monkeypatch.setenv("TVM_FFI_CACHE_DIR", str(tmp_path / root))
        module.load_aot("radix", cpp_files=["radix.cpp"])
    assert [directory.parent for directory in directories] == [
        tmp_path / "cache-a",
        tmp_path / "cache-b",
    ]
    assert directories[0].name == directories[1].name


def test_builder_and_loader_forward_to_tvm_ffi(copies, tmp_path, monkeypatch):
    module = copies[0]
    calls = []
    sentinel = object()

    def record(kind):
        def call(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return sentinel

        return call

    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(load_module=record("load")))
    monkeypatch.setitem(
        sys.modules,
        "tvm_ffi.cpp",
        SimpleNamespace(load_inline=record("jit"), load=record("aot")),
    )
    for kind in ("jit", "aot"):
        assert module._build_module(kind, "name", tmp_path, extra_cflags=["-O3"]) is sentinel
        assert calls[-1] == (
            kind,
            ("name",),
            {"build_directory": str(tmp_path), "extra_cflags": ["-O3"]},
        )
    assert module._load_built(tmp_path / "name.so") is sentinel
    assert calls[-1] == ("load", (str(tmp_path / "name.so"),), {})


def test_concurrent_cuda_arch_builds_each_see_their_own_target(copies, tmp_path, monkeypatch):
    module = copies[0]
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "8.9")
    seen = {}

    def build(name, **kwargs):
        first = os.environ["TVM_FFI_CUDA_ARCH_LIST"]
        time.sleep(0.05)  # a racing build would overwrite the variable meanwhile
        seen[name] = (first, os.environ["TVM_FFI_CUDA_ARCH_LIST"])

    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tvm_ffi.cpp", SimpleNamespace(load_inline=build, load=build))
    arches = {"a": "10.3a", "b": "12.0a", "c": None}
    threads = [
        threading.Thread(
            target=module._build_module, args=("jit", name, tmp_path), kwargs={"cuda_arch": arch}
        )
        for name, arch in arches.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {name: (arch or "8.9",) * 2 for name, arch in arches.items()}
    assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "8.9"


def test_cuda_wrappers_alone_require_cuda_identity(copies, tmp_path, monkeypatch):
    module = copies[0]
    targets = []

    def toolchain(has_cuda):
        targets.append(has_cuda)
        return ("test-toolchain",)

    monkeypatch.setattr(module, "_toolchain_identity", toolchain)
    monkeypatch.setattr(module, "_build_module", lambda *args, **kwargs: object())
    module.load_jit(
        "wrapper_only",
        cuda_wrappers=[("launch", "launch_test")],
        build_directory=str(tmp_path / "build"),
    )
    assert targets == [True]


def test_compiler_version_and_path_change_key(tmp_path, monkeypatch):
    module = import_utils(PACKAGE, "jit_utils_compiler")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "c++").write_text("")
    monkeypatch.setattr(module.shutil, "which", lambda name: str(bindir / name))
    versions = {"c++": "g++ (GCC) 13.2.0\nCopyright"}
    monkeypatch.setattr(module, "_command_output", lambda argv: versions[argv[0]])
    first = module._compiler_identity("c++")
    assert first == ("c++", str((bindir / "c++").resolve()), "g++ (GCC) 13.2.0")
    module._compiler_identity.cache_clear()
    versions["c++"] = "g++ (GCC) 14.1.0"
    assert module._compiler_identity("c++") != first

    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(__version__="test-ffi"))
    keys = []
    for version in ("13.2.0", "14.1.0"):
        module._compiler_identity.cache_clear()
        module._toolchain_identity.cache_clear()
        monkeypatch.setattr(module, "_command_output", lambda argv, v=version: f"{argv[0]} {v}")
        keys.append(key(module, "radix"))
    assert keys[0] != keys[1]

    # Same executable and version, different command arguments (ABI define).
    monkeypatch.setattr(module, "_command_output", lambda argv: f"{argv[0]} 13.2.0")
    abi_keys = []
    for abi in ("0", "1"):
        monkeypatch.setenv("CXX", f"c++ -D_GLIBCXX_USE_CXX11_ABI={abi}")
        module._compiler_identity.cache_clear()
        module._toolchain_identity.cache_clear()
        abi_keys.append(key(module, "radix"))
    assert abi_keys[0] != abi_keys[1]


def test_changed_source_with_older_mtime_rebuilds_explicit_directory(copies, tmp_path, monkeypatch):
    module = copies[0]
    directory = tmp_path / "build"
    source = module.KERNEL_PATH / "src" / "radix.cpp"
    builds = []

    def build(kind, name, build_dir, **kwargs):
        # Like ninja: keep an existing output that is newer than its sources.
        so = build_dir / f"{name}.so"
        obj = build_dir / "cpp_0.o"
        if not so.exists() or not obj.exists():
            obj.write_bytes(source.read_bytes())
            so.write_bytes(source.read_bytes())
        builds.append(so.read_bytes())
        return so.read_bytes()

    monkeypatch.setattr(module, "_build_module", build)
    monkeypatch.setattr(module, "_load_built", lambda path: path.read_bytes())
    old = module.load_aot("radix", cpp_files=["radix.cpp"], build_directory=str(directory))
    (directory / ".ninja_log").write_text("stale")
    old_marker = (directory / "ctikernels.key").read_text()
    past = time.time() - 365 * 24 * 3600
    source.write_bytes(source.read_bytes() + b"\n// changed\n")
    os.utime(source, (past, past))
    new = module.load_aot("radix", cpp_files=["radix.cpp"], build_directory=str(directory))
    assert new != old and new == source.read_bytes()
    assert (directory / f"{module._make_name('radix')}.so").read_bytes() == new
    assert (directory / "ctikernels.key").read_text() != old_marker
    assert not (directory / ".ninja_log").exists()
    assert len(builds) == 2
    assert module.load_aot("radix", cpp_files=["radix.cpp"], build_directory=str(directory)) == new
    assert len(builds) == 2
