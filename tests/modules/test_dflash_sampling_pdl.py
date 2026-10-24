# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Enforce PDL ordering without importing CuTe or initializing CUDA."""

import ast
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).parents[2] / "python/ctikernels/sampling/_dflash/cute"


@pytest.mark.parametrize("filename", ["kernels.py", "argmax.py"])
def test_sampling_pdl_entry_and_launch(filename):
    tree = ast.parse((KERNEL_DIR / filename).read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "_pdl"
        and any(alias.name == "PDL" and alias.asname is None for alias in node.names)
        for node in tree.body
    ), "Both modules must use the shared PDL switch"
    # No local assignment may silently override the shared switch.
    assert not any(
        isinstance(node, ast.Name) and node.id == "PDL" and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )
    expected = ast.parse(
        "if cutlass.const_expr(PDL):\n"
        "    cute.arch.griddepcontrol_wait()\n"
        "    cute.arch.griddepcontrol_launch_dependents()\n"
    ).body[0]
    kernels, launches = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        ):
            kernels.append(node)
            assert ast.dump(node.body[0]) == ast.dump(expected), (
                f"{filename}:{node.lineno} {node.name} must wait and trigger before all other statements"
            )
            calls = [
                ast.unparse(call.func)
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and ast.unparse(call.func).startswith("cute.arch.griddepcontrol_")
            ]
            assert sorted(calls) == [
                "cute.arch.griddepcontrol_launch_dependents",
                "cute.arch.griddepcontrol_wait",
            ], "No additional, data-dependent PDL calls"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "launch"
        ):
            launches.append(node)
            assert any(
                keyword.arg == "use_pdl" and ast.unparse(keyword.value) == "PDL"
                for keyword in node.keywords
            ), f"{filename}:{node.lineno} launch must use the shared PDL switch"
    assert kernels and launches, "Source inspection must find kernels and launches"
