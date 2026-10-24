# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Device sweeps with explicit CI cores and bounded pairwise qualification."""

from itertools import combinations_with_replacement, product
from random import Random

import pytest


def _flatten(row):
    return tuple(
        value for entry in row for value in (entry if isinstance(entry, tuple) else (entry,))
    )


def _pairwise_rows(candidates, sizes, core, seed):
    """Greedy set cover over integer axis indices; ties use a local seeded order."""
    offsets = []
    offset = 0
    for i, j in combinations_with_replacement(range(len(sizes)), 2):
        # Diagonal pairs also cover individual values (including one-axis grids).
        offsets.append((i, j, offset))
        offset += sizes[i] * sizes[j]

    def mask(row):
        return sum(1 << (offset + row[i] * sizes[j] + row[j]) for i, j, offset in offsets)

    candidates = list(candidates)
    # Never iterate core sets: output must not depend on PYTHONHASHSEED.
    selected = [row for row in candidates if row in core]
    Random(seed).shuffle(candidates)
    masks = [mask(row) for row in candidates]
    uncovered = 0
    for bits in masks:
        uncovered |= bits
    for row in selected:
        uncovered &= ~mask(row)
    while uncovered:
        best = max(range(len(candidates)), key=lambda index: (masks[index] & uncovered).bit_count())
        selected.append(candidates[best])
        uncovered &= ~masks[best]
    return selected


def pairwise_cases(*axes, core, seed, compile_axes=()):
    """Cover every value and every pair, keeping exactly the explicit CI core.

    Tuple entries are grouped arguments, just as in ``ci_cases``. Axis order and
    a local seed make selection stable, including when ``core`` is an unordered
    set. Only non-core rows receive the exhaustive mark.

    ``compile_axes`` contains axis indices, before tuple flattening. When given,
    first pair-cover those axes (including all core projections), then restrict
    the full design to those compile keys. Crossing that design with all runtime
    values still admits every pair, while reusing kernels across runtime cases.
    The Cartesian product is only a CPU candidate pool, never a pytest sweep.
    """
    axes = [tuple(axis) for axis in axes]
    if not axes or any(not axis for axis in axes):
        raise ValueError("Pairwise axes must be nonempty")
    sizes = [len(axis) for axis in axes]
    candidates = list(product(*(range(size) for size in sizes)))

    def flatten(row):
        return _flatten(axes[i][value] for i, value in enumerate(row))

    core = set(core)
    core_rows = {row for row in candidates if flatten(row) in core}
    if not core or {flatten(row) for row in core_rows} != core:
        raise ValueError("CI core must be a nonempty subset of the full parameter sweep")
    compile_axes = tuple(compile_axes)
    if len(set(compile_axes)) != len(compile_axes) or any(
        i not in range(len(axes)) for i in compile_axes
    ):
        raise ValueError("Compile axes must be distinct valid axis indices")
    if compile_axes:
        compile_sizes = [sizes[i] for i in compile_axes]
        compile_core = {tuple(row[i] for i in compile_axes) for row in core_rows}
        compile_rows = set(
            _pairwise_rows(
                product(*(range(size) for size in compile_sizes)), compile_sizes, compile_core, seed
            )
        )
        candidates = [
            row for row in candidates if tuple(row[i] for i in compile_axes) in compile_rows
        ]
    return [
        pytest.param(*flatten(row), marks=() if row in core_rows else pytest.mark.exhaustive)
        for row in _pairwise_rows(candidates, sizes, core_rows, seed)
    ]


def ci_cases(*axes, core):
    """Expand axes (tuple entries are grouped arguments), marking all but core.

    Core rows use the flattened argument order in the parametrization. Checking
    membership prevents a typo or changed axis from silently dropping CI coverage.
    """
    cases = [
        tuple(value for entry in row for value in (entry if isinstance(entry, tuple) else (entry,)))
        for row in product(*axes)
    ]
    core = set(core)
    if not core or not core <= set(cases):
        raise ValueError("CI core must be a nonempty subset of the full parameter sweep")
    return [
        pytest.param(*case, marks=() if case in core else pytest.mark.exhaustive) for case in cases
    ]
