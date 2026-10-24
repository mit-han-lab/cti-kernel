# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Package layout: op folders, an empty ``__init__``, libraries at home."""

from __future__ import annotations

import ast
from pathlib import Path

from ._package import (
    LIBRARY_SOURCE,
    OP_LEVEL_FILES,
    OPS,
    ROOT_FILES,
    SOURCES,
    check_ledger,
    exempt,
    package,
)


def _empty(tree: ast.Module) -> bool:
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # a docstring is allowed
    return not body


def _entries(folder: Path) -> list[Path]:
    """Children, ignoring caches and folders left holding only caches (stale checkouts)."""
    return [
        p
        for p in folder.iterdir()
        if p.name != "__pycache__"
        and (p.is_file() or any(c.name != "__pycache__" for c in p.iterdir()))
    ]


def measure() -> dict[str, int]:
    pkg = package()

    def counted(path: Path) -> bool:  # a folder is exempt when a glob covers its contents
        rel = path.relative_to(pkg.root).as_posix()
        return not (exempt(rel, "layout") or (path.is_dir() and exempt(rel + "/", "layout")))

    entries = [p for p in _entries(pkg.root) if counted(p)]
    ops = [p for p in entries if p.is_dir() and p.name in OPS]
    op_entries = [p for op in ops for p in _entries(op) if counted(p)]
    libraries = {
        (module.rel, library)
        for module, library, _, _ in pkg.third_party
        if not exempt(module.rel, "layout")
    }
    return {
        "root_loose_modules": sum(p.suffix == ".py" and p.name not in ROOT_FILES for p in entries),
        "root_extra_folders": sum(
            p.is_dir() and p.name not in OPS and p.name != "_runtime" for p in entries
        ),
        "op_level_extra_files": sum(
            p.is_file() and p.suffix == ".py" and p.name not in OP_LEVEL_FILES for p in op_entries
        ),
        "op_level_extra_folders": sum(p.is_dir() and p.name not in SOURCES for p in op_entries),
        "nonempty_op_inits": sum(
            not _empty(pkg.modules[f"ctikernels.{op.name}"].tree) for op in ops
        ),
        "libraries_outside_source": sum(
            LIBRARY_SOURCE[library] not in rel.split("/")[:-1] for rel, library in libraries
        ),
    }


def test_layout_counts_match_ledger() -> None:
    check_ledger("layout", measure())


def test_signed_layout_exemptions_take_effect(monkeypatch) -> None:
    """Only a signed exemption list lowers the counts; built here, independent of the real one."""
    from . import _package

    real_load = _package.load_toml
    entry = {"guards": ["layout"], "paths": ["moe/*", "moe_impl.py"]}

    def use(exemptions: dict) -> dict[str, int]:
        monkeypatch.setattr(
            _package,
            "load_toml",
            lambda name: exemptions if name == "exemptions.toml" else real_load(name),
        )
        return measure()

    none = use({})
    assert use({"signed_by": "", "exemption": [entry]}) == none  # unsigned: not applied
    signed = use({"signed_by": "someone", "exemption": [entry]})
    assert signed["root_loose_modules"] == none["root_loose_modules"] - 1
    assert signed["op_level_extra_files"] < none["op_level_extra_files"]
