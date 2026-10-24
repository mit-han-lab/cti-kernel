# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""A read-only AST model of ``python/ctikernels`` shared by the architecture guards.

Everything here is static: no package module is imported. The model is deliberately small
(modules, internal import edges, third-party imports, calls); each guard file defines its own
counts on top of it, and ``ledger.toml`` holds the exact values the tree must match.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "python" / "ctikernels"
HERE = Path(__file__).resolve().parent

# Op packages (current and planned). Anything else at the root is loose.
OPS = frozenset(
    {
        "activation", "attention", "collective", "dsv4_indexer", "embedding", "gdn", "head",
        "kvcache", "linear", "logprob", "mhc", "mlp", "moe", "norm", "rotary", "sampling",
        "speculative",
    }
)  # fmt: skip
# Source folders (one third-party library or language per folder).
SOURCES = frozenset(
    {
        "cublas", "cuda", "cute", "fla", "flash_attn", "flashinfer", "liger", "quack",
        "sgl_kernel", "torch", "triton",
    }
)  # fmt: skip
# Third-party library -> the source folder that may import it.
LIBRARY_SOURCE = {
    "cutlass": "cute",
    "flashinfer": "flashinfer",
    "triton": "triton",
    "fla": "fla",
    "quack": "quack",
    "flash_attn": "flash_attn",
    "liger_kernel": "liger",
    "sgl_kernel": "sgl_kernel",
}
OP_LEVEL_FILES = frozenset({"__init__.py", "inference.py", "training.py", "_common.py"})
ROOT_FILES = frozenset({"__init__.py", "__main__.py"})


def private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


@dataclass(frozen=True)
class Edge:
    """One internal import statement from ``source`` to module ``target``."""

    source: str
    target: str
    line: int
    local: bool  # inside a function or lambda body
    names: tuple[str, ...]  # imported names that are not submodules


@dataclass
class Module:
    name: str
    path: Path
    is_package: bool
    tree: ast.Module = field(repr=False)

    @property
    def rel(self) -> str:
        return self.path.relative_to(PACKAGE).as_posix()

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.name.split(".")[1:])

    @property
    def unit(self) -> str:
        """``<op>`` for op packages, the module name for root modules, ``""`` for the root."""
        return self.parts[0] if self.parts else ""

    def nodes(self, *types: type) -> list[ast.AST]:
        return [node for node in ast.walk(self.tree) if isinstance(node, types)]

    def calls(self) -> list[tuple[ast.Call, str]]:
        return [(node, ast.unparse(node.func)) for node in self.nodes(ast.Call)]


def _module_name(path: Path) -> tuple[str, bool]:
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _local_lines(tree: ast.Module) -> set[int]:
    """Ids of import nodes that sit inside a function or lambda."""
    local: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    local.add(id(inner))
    return local


class Package:
    def __init__(self, root: Path = PACKAGE) -> None:
        self.root = root
        self.modules: dict[str, Module] = {}
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            name, is_package = _module_name(path)
            self.modules[name] = Module(
                name, path, is_package, ast.parse(path.read_text(), str(path))
            )
        self.edges: list[Edge] = []
        self.third_party: list[tuple[Module, str, int, bool]] = []  # (module, library, line, local)
        for module in self.modules.values():
            self._scan_imports(module)

    def _resolve(self, module: Module, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = module.name.split(".")
        if not module.is_package:
            base.pop()
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def _scan_imports(self, module: Module) -> None:
        local = _local_lines(module.tree)
        for node in module.nodes(ast.Import, ast.ImportFrom):
            is_local = id(node) in local
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._third(module, alias.name, node.lineno, is_local)
                    if alias.name in self.modules:
                        self.edges.append(Edge(module.name, alias.name, node.lineno, is_local, ()))
                continue
            target = self._resolve(module, node)
            if not node.level:
                self._third(module, target, node.lineno, is_local)
            if target not in self.modules:
                continue
            names = []
            for alias in node.names:
                sub = f"{target}.{alias.name}"
                if sub in self.modules:
                    self.edges.append(Edge(module.name, sub, node.lineno, is_local, ()))
                else:
                    names.append(alias.name)
            if names:
                self.edges.append(Edge(module.name, target, node.lineno, is_local, tuple(names)))

    def _third(self, module: Module, target: str, line: int, is_local: bool) -> None:
        library = target.split(".")[0]
        if library in LIBRARY_SOURCE:
            self.third_party.append((module, library, line, is_local))

    def unit(self, name: str) -> str:
        return self.modules[name].unit


@functools.cache
def package() -> Package:
    return Package()


def load_toml(name: str) -> dict:
    with open(HERE / name, "rb") as file:
        return tomllib.load(file)


def signed_exemptions(guard: str) -> tuple[str, ...]:
    """Path globs (relative to the package) exempt from ``guard``; only a signed list counts."""
    data = load_toml("exemptions.toml")
    if not data.get("signed_by"):
        return ()
    return tuple(
        glob
        for entry in data.get("exemption", ())
        if guard in entry["guards"]
        for glob in entry["paths"]
    )


def exempt(rel: str, guard: str) -> bool:
    """Whether the package-relative path ``rel`` is exempt from ``guard`` by a signed entry."""
    return any(fnmatch(rel, glob) for glob in signed_exemptions(guard))


def check_ledger(guard: str, measured: dict[str, int]) -> None:
    """Every count must equal its ledger entry: a rise is a violation, a drop needs a ledger edit."""
    ledger = load_toml("ledger.toml")[guard]
    problems = []
    for key, value in sorted(measured.items()):
        expected = ledger.get(key)
        if expected is None:
            problems.append(f"{guard}.{key} = {value} has no ledger entry")
        elif value > expected:
            problems.append(f"{guard}.{key} rose {expected} -> {value}: a new violation")
        elif value < expected:
            problems.append(
                f"{guard}.{key} fell {expected} -> {value}: lower its entry in ledger.toml"
            )
    stale = sorted(set(ledger) - set(measured))
    problems += [f"{guard}.{key} in ledger.toml is no longer measured" for key in stale]
    assert not problems, "\n".join(problems)
