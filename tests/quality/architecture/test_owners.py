# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Single owners: process-wide concerns live in ``_runtime`` only.

Outside ``_runtime`` the counts below may only fall: CuTe compiles, capture queries, device
queries, environment access, DSL pins, module-level locks/caches/containers, side streams,
hidden pools and ``assert`` statements.
"""

from __future__ import annotations

import ast
import re

from ._package import Module, check_ledger, exempt, package

DEVICE_QUERIES = {"get_device_capability", "get_device_properties", "get_device_name"}
CONTAINERS = {
    "dict", "list", "set", "defaultdict", "OrderedDict",
    "WeakKeyDictionary", "WeakValueDictionary", "WeakSet",
}  # fmt: skip
# The process-wide pools, each removed in its batch.
HIDDEN_POOLS = {
    "linear/inference.py": "_SCRATCH",
    "moe_impl.py": "_IDENTITY_MAPS",
    "_alpha_gemm.py": "_device_zero",
    "gdn/_triton_decode.py": "_POOLS",
}


def _last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _module_level(module: Module):
    for node in module.tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            yield node.value


def _environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and ast.unparse(node) in {"os.environ", "environ"}


def _defines(module: Module, name: str) -> bool:
    for node in module.tree.body:
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return True
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return True
    return False


def measure() -> dict[str, int]:
    pkg = package()
    counts = dict.fromkeys(
        (
            "cute_compile_calls", "cute_compile_files", "capture_checks", "device_queries",
            "env_reads", "env_writes", "dsl_pin_files", "module_locks", "module_empty_containers",
            "cache_decorators", "side_streams", "hidden_pools", "asserts",
        ),
        0,
    )  # fmt: skip
    for module in pkg.modules.values():
        if module.parts[:1] == ("_runtime",) or exempt(module.rel, "owners"):
            continue
        compiles = sum(name.endswith("cute.compile") for _, name in module.calls())
        counts["cute_compile_calls"] += compiles
        counts["cute_compile_files"] += bool(compiles)
        lines = {"capture": set(), "device": set(), "read": set(), "write": set()}
        for node, name in module.calls():
            last = _last(name)
            if last == "is_current_stream_capturing":
                lines["capture"].add(node.lineno)
            elif last in DEVICE_QUERIES:
                lines["device"].add((last, node.lineno))
            elif name in {"os.getenv", "getenv"} or (
                _environ(node.func.value) if isinstance(node.func, ast.Attribute) else False
            ):
                kind = (
                    "read"
                    if last in {"get", "copy", "items", "keys"} or last == "getenv"
                    else "write"
                )
                lines[kind].add(node.lineno)
            elif name in {"torch.cuda.Stream", "torch.cuda.streams.Stream"}:
                counts["side_streams"] += 1
        for node in module.nodes(ast.Attribute):
            if node.attr == "multi_processor_count":
                lines["device"].add(("multi_processor_count", node.lineno))
        for node in module.nodes(ast.Subscript):
            if _environ(node.value):
                kind = "read" if isinstance(node.ctx, ast.Load) else "write"
                lines[kind].add(node.lineno)
        for node in module.nodes(ast.Compare):
            if any(_environ(c) for c in node.comparators):
                lines["read"].add(node.lineno)
        counts["capture_checks"] += len(lines["capture"])
        counts["device_queries"] += len(lines["device"])
        counts["env_reads"] += len(lines["read"])
        counts["env_writes"] += len(lines["write"])
        counts["dsl_pin_files"] += any(
            isinstance(n, ast.Constant) and n.value == "4.8.0" for n in module.nodes(ast.Constant)
        )
        for value in _module_level(module):
            if isinstance(value, ast.Call):
                last = _last(ast.unparse(value.func))
                counts["module_locks"] += last in {"Lock", "RLock"}
                counts["module_empty_containers"] += last in CONTAINERS and (
                    not value.args or last == "defaultdict"
                )
            elif isinstance(value, (ast.Dict, ast.List, ast.Set)):
                size = len(value.keys) if isinstance(value, ast.Dict) else len(value.elts)
                counts["module_empty_containers"] += size == 0
        for node in module.nodes(ast.FunctionDef, ast.AsyncFunctionDef):
            counts["cache_decorators"] += sum(
                bool(re.search(r"(^|\.)(lru_cache|cache)\b", ast.unparse(d)))
                for d in node.decorator_list
            )
        counts["hidden_pools"] += module.rel in HIDDEN_POOLS and _defines(
            module, HIDDEN_POOLS[module.rel]
        )
        counts["asserts"] += len(module.nodes(ast.Assert))
    return counts


def test_owner_counts_match_ledger() -> None:
    check_ledger("owners", measure())
