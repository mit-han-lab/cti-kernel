# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Pairwise coverage and GPU sweep budgets, without importing torch or CUDA."""

import ast
import importlib.util
from itertools import combinations, product
from pathlib import Path
from random import getstate

import pytest

# Load the GPU-suite helper by path: standalone tests do not import repository test packages.
_SPEC = importlib.util.spec_from_file_location(
    "_gpu_exhaustive", Path(__file__).resolve().parents[1] / "gpu" / "_exhaustive.py"
)
_EXHAUSTIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXHAUSTIVE)
pairwise_cases = _EXHAUSTIVE.pairwise_cases


def assert_design(axes, core, cases):
    values = [case.values for case in cases]
    assert len(values) == len(set(values))
    assert {case.values for case in cases if not case.marks} == set(core)
    assert all([mark.name for mark in case.marks] == ["exhaustive"] for case in cases if case.marks)
    # Rebuild grouped axes independently of the generator's index representation.
    rows = []
    for values_row in values:
        row, offset = [], 0
        for axis in axes:
            grouped = isinstance(axis[0], tuple)
            width = len(axis[0]) if grouped else 1
            value = tuple(values_row[offset : offset + width]) if grouped else values_row[offset]
            assert value in axis
            row.append(value)
            offset += width
        assert offset == len(values_row)
        rows.append(row)
    for i, axis in enumerate(axes):
        assert {row[i] for row in rows} == set(axis)
    for i, j in combinations(range(len(axes)), 2):
        assert {(row[i], row[j]) for row in rows} == set(product(axes[i], axes[j]))
    return rows


@pytest.mark.parametrize("compile_axes", [(), (0,), (0, 2), (0, 1, 2)])
def test_pairwise_coverage_core_and_stability(compile_axes):
    axes = [(1, 2, 3), ((8, 2), (4, 4)), ("fp16", "bf16", "fp8")]
    core = [(1, 8, 2, "fp16"), (3, 4, 4, "bf16")]
    state = getstate()
    cases = pairwise_cases(*axes, core=core, seed=1701, compile_axes=compile_axes)
    assert_design(axes, core, cases)
    assert cases == pairwise_cases(
        *axes, core=list(reversed(core)), seed=1701, compile_axes=compile_axes
    )
    assert getstate() == state


def test_single_axis_and_seed():
    assert_design([(0, 1, 32768)], [(1,)], pairwise_cases((0, 1, 32768), core=[(1,)], seed=73))
    axes = [range(4)] * 3
    core = [(0, 0, 0)]
    assert pairwise_cases(*axes, core=core, seed=1) != pairwise_cases(*axes, core=core, seed=2)


@pytest.mark.parametrize("core", [[], [(3, "fp16")], [(1, "fp16"), (2, "fp8")]])
def test_invalid_core(core):
    with pytest.raises(ValueError, match="CI core"):
        pairwise_cases((1, 2), ("fp16", "bf16"), core=core, seed=73)


@pytest.mark.parametrize("axes", [(), ((),), ((1,), ())])
def test_empty_axes(axes):
    with pytest.raises(ValueError, match="axes must be nonempty"):
        pairwise_cases(*axes, core=[(1,)], seed=73)


@pytest.mark.parametrize("compile_axes", [(0, 0), (-1,), (1,)])
def test_invalid_compile_axes(compile_axes):
    with pytest.raises(ValueError, match="Compile axes"):
        pairwise_cases((1, 2), core=[(1,)], seed=73, compile_axes=compile_axes)


class _LiteralDtypes(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "torch":
            assert node.attr in {"bfloat16", "float16"}
            return ast.Constant(node.attr)
        return self.generic_visit(node)


def sweep_spec(name):
    # Read only the literal generator call: the CPU contract needs neither torch
    # nor imports/fixtures from the GPU test modules. Dtypes are opaque axis values.
    path = Path(__file__).parents[1] / "gpu" / f"test_{name}_cute.py"
    tree = ast.parse(path.read_text())
    (call,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pairwise_cases"
    )
    call = _LiteralDtypes().visit(call)
    return [ast.literal_eval(arg) for arg in call.args], {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
    }


@pytest.mark.parametrize(
    "name,expected_sizes,expected_core,other_cases,other_compile_keys",
    [
        (
            "decode_fp8",
            [5, 7, 3, 3, 12, 2, 3],
            {
                (1, 32, 8, 128, 16, 1, "bfloat16", 1.0, 1.0),
                (2, 24, 4, 64, 1, 257, "float16", 0.5, 0.5),
                (3, 64, 4, 256, 64, 1000, "bfloat16", 0.5, 0.5),
                (4, 4, 4, 64, 16, 0, "float16", 1.0, 1.0),
                (7, 12, 2, 128, 16, 17, "bfloat16", 0.5, 0.0),
                (1, 8, 2, 128, 1, 32768, "float16", 1.0, 1.0),
            },
            37,
            34,  # graph False/True share three kernels; all other cases counted separately
        ),
        (
            "prefill",
            [4, 5, 4, 5, 2, 2, 2],
            {
                (3, 64, "2048", 32, 8, 128, 16, "bfloat16", False),
                (1, 256, "2048", 32, 8, 128, 16, "float16", False),
                (3, 64, "2048", 16, 4, 256, 16, "bfloat16", False),
                (2, 16, "self", 16, 4, 128, 1, "float16", True),
                (4, 100, "mixed", 8, 2, 256, 16, "float16", True),
                (2, 1, "8192", 4, 4, 64, 1, "bfloat16", True),
                (1, 1, "self", 4, 4, 64, 16, "float16", False),
            },
            42,
            42,  # includes separate negative/zero/positive scale-sign specializations
        ),
    ],
)
def test_gpu_sweep_coverage_and_budgets(
    name, expected_sizes, expected_core, other_cases, other_compile_keys
):
    axes, options = sweep_spec(name)
    assert [len(axis) for axis in axes] == expected_sizes
    assert set(options["core"]) == expected_core
    cases = pairwise_cases(*axes, **options)
    rows = assert_design(axes, expected_core, cases)
    compile_keys = {tuple(row[i] for i in options["compile_axes"]) for row in rows}
    # Count all cases, even unmarked core cases, and conservatively reserve keys
    # for every other test. Config is fixed (or geometry-derived) within each grid.
    assert len(cases) + other_cases <= 400
    assert len(compile_keys) + other_compile_keys <= 120
