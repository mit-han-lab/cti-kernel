# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""One plain-text table per op: a title, a header and aligned rows."""

from __future__ import annotations

from collections.abc import Sequence


def render(title: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in header]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]

    def line(row):
        return "| " + " | ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) + " |"

    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([f"## {title}", "", line(cells[0]), rule, *map(line, cells[1:]), ""])


def ratio_cell(ratio: float, ci: tuple[float, float]) -> str:
    """``0.93 [0.91, 0.95]``: B/A with its interval (below 1: the comparator is faster)."""
    return f"{ratio:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
