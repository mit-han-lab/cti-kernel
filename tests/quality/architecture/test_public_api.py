# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Public API: every public module's ``__all__`` equals public_api.toml.

public_api.toml is the B0 snapshot of today's exports, not the target list: the
target (about 50 op exports, D13) is fixed before B1, and each batch edits both sides together.
"""

from __future__ import annotations

import ast

from ._package import SOURCES, Module, check_ledger, exempt, load_toml, package, private


def _public(module: Module) -> bool:
    """No private part and not inside a source folder (L1 is private whatever its name)."""
    parts = module.parts
    return (
        not any(map(private, parts))
        and not any(part in SOURCES for part in parts)
        and parts[-1:] != ("__main__",)
    )


def _all(module: Module) -> list[str] | None:
    """The sorted ``__all__``; ``[]`` for a module with no statements, ``None`` when missing."""
    body = module.tree.body
    if all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) for n in body):
        return []
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                return sorted(ast.literal_eval(node.value))
    return None


def _modules() -> dict[str, list[str] | None]:
    return {
        m.name: _all(m)
        for m in package().modules.values()
        if _public(m) and not exempt(m.rel, "public_api")
    }


def measure() -> dict[str, int]:
    exports = _modules()
    names = {name for values in exports.values() if values for name in values}
    return {
        "public_modules": len(exports),
        "public_modules_without_all": sum(values is None for values in exports.values()),
        "exported_names": len(names),
    }


def test_all_matches_public_api_toml() -> None:
    expected = {name: sorted(names) for name, names in load_toml("public_api.toml").items()}
    actual = {name: names for name, names in _modules().items() if names}
    assert actual == expected


def test_public_api_counts_match_ledger() -> None:
    check_ledger("public_api", measure())
