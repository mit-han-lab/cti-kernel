# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""kvcache.inference.RowCopy on CPU (the torch reference)."""

import pytest
import torch

from ctikernels.kvcache.inference import RowCopy

from ._row_copy_case import KINDS, bits, expected, pool

# -1 skips; valid targets distinct and disjoint from valid sources.
PAIRS = ([0, 3, -1, 5, 8, 2], [6, 1, 4, -1, 7, -1])


@pytest.mark.parametrize("kinds", [KINDS, KINDS[:3]], ids=["bytes", "words"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_copies_valid_rows_bit_for_bit_and_skips_the_rest(kinds, dtype):
    tensors = pool("cpu", kinds=kinds)
    sources, targets = (torch.tensor(v, dtype=dtype) for v in PAIRS)
    want = expected(tensors, sources, targets)
    op = RowCopy(tensors)
    assert op.provider == "torch"
    op(sources, targets)
    assert all(torch.equal(bits(a), bits(b)) for a, b in zip(tensors, want, strict=True))


@pytest.mark.parametrize(
    ("tensors", "match"),
    [
        ((), "at least one tensor"),
        ((torch.zeros(0, 3),), "at least one row"),
        ((torch.zeros(4, 6)[:, ::2],), "contiguous"),
        ((torch.zeros(4, 2, requires_grad=True),), "grad"),
    ],
)
def test_rejects_unusable_tensors(tensors, match):
    with pytest.raises(ValueError, match=match):
        RowCopy(tensors)


@pytest.mark.parametrize(
    ("sources", "targets", "match"),
    [
        (torch.zeros(2, dtype=torch.int64), torch.zeros(3, dtype=torch.int64), "one length"),
        (torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64), "int32 or both"),
        (torch.zeros(2), torch.zeros(2), "int32 or both"),
        (torch.zeros(1, 2, dtype=torch.int64), torch.zeros(1, 2, dtype=torch.int64), "1-D"),
        (torch.zeros(4, dtype=torch.int64)[::2], torch.zeros(2, dtype=torch.int64), "contiguous"),
    ],
)
def test_rejects_bad_indices(sources, targets, match):
    with pytest.raises(ValueError, match=match):
        RowCopy(pool("cpu"))(sources, targets)


def test_rejects_overlapping_rows():
    with pytest.raises(ValueError, match="overlap"):
        RowCopy((torch.zeros(12).as_strided((4, 3), (2, 1)),))


@pytest.mark.parametrize(
    ("sources", "targets", "match"),
    [
        ([0, 20], [1, 2], "-1 or in"),
        ([0, -2], [1, 2], "-1 or in"),
        ([0, 3], [1, 1], "distinct"),
        ([0, 3], [3, 1], "disjoint"),
    ],
)
def test_cpu_provider_enforces_the_index_contract(sources, targets, match):
    op = RowCopy(pool("cpu"))
    with pytest.raises(ValueError, match=match):
        op(torch.tensor(sources), torch.tensor(targets))


def test_row_strides_and_row_counts_per_tensor():
    """Rows strided past their size, and an index valid for one tensor but beyond another's
    rows (skipped there only)."""
    wide = torch.randn(9, 16)[:, :10]
    short = torch.randn(5, 3, dtype=torch.float64)
    sources, targets = torch.tensor([0, 7, 2]), torch.tensor([3, 1, 8])
    want = expected([wide, short], sources, targets)
    RowCopy((wide, short))(sources, targets)
    assert all(torch.equal(bits(a), bits(b)) for a, b in zip((wide, short), want, strict=True))
