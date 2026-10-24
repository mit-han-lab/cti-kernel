# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Selection rules: no module-level tables, no device-name keys.

A table constant is a module-level UPPER_CASE name bound to a nested literal (optionally wrapped
in one call such as ``frozenset(...)``) holding at least 8 numbers. The per-Op rules (one
Selection per public Op, Segment shape, segment within support, eager route == graph route)
start with the first ``Segment`` in B2 and join this file then.
"""

from __future__ import annotations

import ast
import re

from ._package import check_ledger, exempt, package

UPPER = re.compile(r"_?[A-Z][A-Z0-9_]*$")
DEVICE_NAME = re.compile(r"(?i)\b(rtx|gb10|blackwell|h100|h200|b200|dgx)|5090|pro ?6000")
LITERALS = (ast.Dict, ast.List, ast.Tuple, ast.Set)


def _table(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets = [node.target]
    else:
        return False
    if not (isinstance(targets[0], ast.Name) and UPPER.match(targets[0].id)):
        return False
    value = node.value
    while isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
        value = value.args[0]
    if not isinstance(value, LITERALS):
        return False
    inner = [n for n in ast.walk(value) if n is not value]
    numbers = sum(isinstance(n, ast.Constant) and type(n.value) in (int, float) for n in inner)
    return numbers >= 8 and any(isinstance(n, LITERALS) for n in inner)


def _name_keys(module) -> set[int]:
    """Lines where a device-name string is used as a key: dict key, comparison, set/tuple member."""
    keys: list[ast.AST] = []
    for node in module.nodes(ast.Dict):
        keys += [k for k in node.keys if k is not None]
    for node in module.nodes(ast.Compare):
        keys += [node.left, *node.comparators]
    for node, name in module.calls():
        if name.endswith((".startswith", ".endswith")):
            keys += node.args
    lines = set()
    for key in keys:
        for leaf in ast.walk(key):
            if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                if DEVICE_NAME.search(leaf.value):
                    lines.add(leaf.lineno)
    return lines


def measure() -> dict[str, int]:
    tables = names = 0
    for module in package().modules.values():
        if exempt(module.rel, "selection"):
            continue
        tables += sum(map(_table, module.tree.body))
        names += len(_name_keys(module))
    return {"table_constants": tables, "device_name_key_lines": names}


def test_selection_counts_match_ledger() -> None:
    check_ledger("selection", measure())
