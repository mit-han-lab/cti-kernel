# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shared construction/kernel arithmetic; no CUDA imports or allocations."""


def counter_levels(splits, fan_in):
    """Number of counters at each level, including the root (none for one split)."""
    levels = []
    while splits > 1:
        splits = (splits + fan_in - 1) // fan_in
        levels.append(splits)
    return tuple(levels)


def merge_columns(rows, head_dim):
    # Bound inline staging to 32 KiB per partial; wider row tiles take two passes.
    return head_dim if rows * head_dim <= 8192 else head_dim // 2


def merge_geometry(rows, key_tile, key_groups, stages, inline, head_dim=128, distributed=False):
    v_stages = stages - 1 if key_tile == 64 and stages == 3 else stages
    arena = max(
        key_tile * head_dim * (stages + v_stages) * 2,
        rows * head_dim * 4 + rows * 8 if key_groups == 2 else 0,
        rows * head_dim * 2,
    )
    if distributed:
        # One of two merging CTAs may own half the rows, including LSE vectors.
        arena = max(arena, ((rows + 1) // 2) * (head_dim + 4) * 4)
        arena = (arena + 127) // 128 * 128
    fan_in = 4 if rows * head_dim * 4 * 4 <= arena else 2
    if inline:
        # Full fan-in payload, packed LSE vectors and a CTA-uniform winner word.
        # Small rings grow only enough for G=2; all levels reuse this same arena.
        arena = max(arena, fan_in * rows * (merge_columns(rows, head_dim) + 1) * 4 + 16)
        arena = (arena + 127) // 128 * 128
    return fan_in, arena
