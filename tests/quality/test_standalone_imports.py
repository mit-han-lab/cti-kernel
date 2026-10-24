# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "python"


def _imports(source: str) -> set[str]:
    """Include imports inside functions and TYPE_CHECKING, not just eager imports."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and node.args:
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            target = node.args[0]
            if function_name in {"__import__", "import_module"} and isinstance(
                target, ast.Constant
            ):
                if isinstance(target.value, str):
                    imported.add(target.value)
    return imported


@pytest.mark.parametrize(
    "source",
    [
        "import ctiengine.engine",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from ctiengine.models import ModelConfig",
        "def setup():\n    from ctiengine.env import ENV",
        'importlib.import_module("ctiengine.engine")',
        '__import__("ctiengine")',
    ],
)
def test_import_gate_sees_runtime_and_type_only_dependencies(source: str) -> None:
    assert any(name == "ctiengine" or name.startswith("ctiengine.") for name in _imports(source))


def test_standalone_sources_and_tests_do_not_import_engine_or_repository_test_helpers() -> None:
    sources = sorted((SOURCE_ROOT / "ctikernels").rglob("*.py"))
    tests = sorted((PACKAGE_ROOT / "tests").rglob("*.py"))
    assert sources and tests
    violations = {}
    for path in (*sources, *tests):
        forbidden = sorted(
            name
            for name in _imports(path.read_text(encoding="utf-8"))
            if name.split(".")[0] in {"ctiengine", "tests", "torch_memory_saver"}
        )
        if forbidden:
            violations[path.relative_to(PACKAGE_ROOT).as_posix()] = forbidden
    assert not violations, violations


def test_cpu_importable_wrappers_do_not_load_engine_or_cuda_providers(tmp_path: Path) -> None:
    code = """
import importlib
import importlib.abc
import sys
from pathlib import Path

class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ctiengine" or fullname.startswith("ctiengine."):
            raise AssertionError("kernel import attempted engine dependency: " + fullname)

sys.meta_path.insert(0, NoEngine())
for name in (
    "ctikernels", "ctikernels.__main__", "ctikernels.activation.inference",
    "ctikernels.attention.inference", "ctikernels.attention.varlen",
    "ctikernels.norm.inference", "ctikernels.rotary.vision", "ctikernels.gdn",
    "ctikernels.collective", "ctikernels.collective.inference",
    "ctikernels.collective._host_exchange.cute.exchange",
    "ctikernels.collective.two_node", "ctikernels.collective._two_node.proxy",
    "ctikernels.linear.lora", "ctikernels.linear.block_fp8",
    "ctikernels.linear.fp8_static", "ctikernels.linear._fp8_static_dispatch",
    "ctikernels.linear.cute._fp8_static_quant", "ctikernels.linear.torch._fp8_static",
    "ctikernels.linear.flashinfer._fp8_static",
    "ctikernels.linear.flashinfer._block_fp8", "ctikernels.linear.triton._block_fp8_provider",
    "ctikernels.linear.block_fp8_gated", "ctikernels.linear._block_fp8_gated_dispatch",
    "ctikernels.linear.flashinfer._block_fp8_gated", "ctikernels.linear.cute._block_fp8_act_quant",
    "ctikernels.linear._block_fp8_dispatch", "ctikernels.linear.cuda._block_fp8_sm12x",
    "ctikernels.cross_entropy", "ctikernels.logprob",
    "ctikernels.linear.nvfp4",
    "ctikernels.moe_impl", "ctikernels.moe", "ctikernels.moe.inference",
    "ctikernels.moe_block_fp8", "ctikernels._moe_block_fp8_cute_table",
    "ctikernels._moe_block_fp8_triton_table",
    "ctikernels.moe.training", "ctikernels.moe.triton_block", "ctikernels.moe.triton_block_gb10",
    "ctikernels.moe._configs", "ctikernels.moe._dispatch",
    "ctikernels.sampling", "ctikernels.utils",
    "ctikernels.moe.lora",
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
for prefix in (
    "ctiengine", "flashinfer", "sgl_kernel", "flash_attn", "liger_kernel", "torch_memory_saver",
    "nvidia", "tvm_ffi", "fla", "cutlass", "triton", "quack",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CTI_ENGINE_PYNCCL_MAX_BUFFER_SIZE"] = "not-a-size"
    subprocess.run(
        (sys.executable, "-c", code, str(SOURCE_ROOT)),
        cwd=tmp_path,
        env=environment,
        check=True,
    )


@pytest.mark.parametrize("leaf", ["inference", "varlen"])
def test_attention_leaf_does_not_import_other_leaf_or_native_providers(
    tmp_path: Path, leaf: str
) -> None:
    code = """
import importlib
import sys

leaf = sys.argv[1]
importlib.import_module("ctikernels.attention." + leaf)
for prefix in ("ctiengine", "flashinfer", "sgl_kernel", "flash_attn"):
    assert not any(
        name == prefix or name.startswith(prefix + ".") for name in sys.modules
    ), prefix
other = "varlen" if leaf == "inference" else "inference"
assert "ctikernels.attention." + other not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run((sys.executable, "-c", code, leaf), cwd=tmp_path, env=environment, check=True)


def test_every_package_module_imports_without_numpy(tmp_path: Path) -> None:
    code = r"""
import sys
sys.modules["numpy"] = None

import importlib
import importlib.abc
import pkgutil

class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ImportError("blocked NumPy dependency: " + fullname, name=fullname)

sys.meta_path.insert(0, NoNumpy())
optional = {
    "triton", "flashinfer", "cutlass", "sgl_kernel", "liger_kernel", "fla", "quack",
    "tvm_ffi", "cuda", "nvidia", "flash_attn",
}
imported = set()
skipped = set()

def numpy_importer(error):
    # The innermost non-importlib frame is the module that asked for NumPy.
    modules, tb = [], error.__traceback__
    while tb is not None:
        modules.append(tb.tb_frame.f_globals.get("__name__") or "")
        tb = tb.tb_next
    modules = [m for m in modules if not m.startswith(("importlib", "_frozen_importlib"))]
    return modules[-1] if modules else ""

def check_failure(name, error):
    # Inspect both links, including suppressed contexts: wrapping an optional
    # provider error must never hide a dependency on NumPy.
    pending, chain, seen = [error], [], set()
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        chain.append(item)
        pending.extend((item.__cause__, item.__context__))
    numpy_errors = [item for item in chain if isinstance(item, ImportError) and
                    (item.name or "").split(".")[0] == "numpy"]
    if numpy_errors:
        # An optional provider that needs NumPy itself (the CuTe DSL does) is
        # absent here; NumPy requested by any other module is a dependency.
        if all(numpy_importer(item).split(".")[0] in optional for item in numpy_errors):
            skipped.add(name)
            return
        raise AssertionError("NumPy dependency in " + name) from error
    if any(isinstance(item, ModuleNotFoundError) and
           (item.name or "").split(".")[0] in optional for item in chain):
        skipped.add(name)
        return
    raise AssertionError("Unexpected import failure in " + name) from error

def onerror(name):
    check_failure(name, sys.exc_info()[1])

import ctikernels
for info in pkgutil.walk_packages(ctikernels.__path__, ctikernels.__name__ + ".", onerror=onerror):
    try:
        importlib.import_module(info.name)
    except Exception as error:
        check_failure(info.name, error)
    else:
        imported.add(info.name)
assert "ctikernels.sampling._dflash.reference.ops" in imported
assert "ctikernels.sampling.inference" in imported
assert sys.modules["numpy"] is None
print(f"Imported {len(imported)} modules; skipped {len(skipped)} absent optional providers")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run((sys.executable, "-c", code), cwd=tmp_path, env=environment, check=True)
