# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DSV4Pages.prepare zeroes exactly row 0: its 576 data bytes and its 8 scale bytes."""

from __future__ import annotations

import pytest
import torch

from ctikernels.attention.inference import DSV4Pages


@pytest.mark.parametrize("paged", [False, True], ids=["flat", "pages"])
@pytest.mark.parametrize("rows", [64, 2])
def test_prepare_zeroes_row0_only(rows, paged):
    pages = DSV4Pages(rows)
    cache = torch.full((3 * pages.page_bytes,), 0xFF, dtype=torch.uint8)
    pages.prepare(cache.view(3, -1) if paged else cache)  # [pages, page_bytes] views too
    zero = torch.zeros_like(cache, dtype=torch.bool)
    zero[:576] = True
    zero[rows * 576 : rows * 576 + 8] = True
    assert torch.equal(cache == 0, zero)


def test_prepare_checks_the_pool():
    with pytest.raises(ValueError):
        DSV4Pages(64).prepare(torch.zeros(100, dtype=torch.uint8))
    with pytest.raises(TypeError):
        DSV4Pages(2).prepare(torch.zeros(DSV4Pages(2).page_bytes, dtype=torch.int8))
