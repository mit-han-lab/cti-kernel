# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DSV4 extend buffer sizes follow ``max_seq_len`` (the longest request) on both paths."""

from __future__ import annotations

import pytest

from ctikernels.attention.inference import DSV4SparseAttention

_sizes = DSV4SparseAttention._extend_sizes


def test_sm12x_c128_table_follows_the_longest_request():
    rows, c128, total = _sizes(False, 512, 131072, 8)
    assert c128 == 1024  # 131072 // 128, 64-padded; independent of max_requests
    assert rows == {"window": 512 * 128, "c128": 512 * 1024, "lengths": 3 * 512, "bases": 18}
    assert total == 4 * sum(rows.values())
    assert _sizes(False, 512, 131072, 1)[1] == c128
    assert DSV4SparseAttention._extend_splits(c128) == (128 + 1024) // 64


@pytest.mark.parametrize("requests", [1, 8])
def test_sm10x_staged_rows_cover_every_request(requests):
    seq = 131072
    rows, c128, _ = _sizes(True, 512, seq, requests)
    assert c128 == 1024
    assert rows["c4_rows"] == requests * seq // 4 + requests
    assert rows["c128_rows"] == requests * seq // 128 + requests
    assert rows["window_rows"] == 512 + 127 * requests
