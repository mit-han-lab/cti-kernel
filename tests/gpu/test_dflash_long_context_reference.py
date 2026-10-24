# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Audit the bounded-memory FP32 oracle against independent dense FP64 math."""

import pytest
import torch

from .._dflash_reference import chunked_draft_attention_reference, draft_attention_reference


@pytest.mark.parametrize("chunk_size", [1, 7, 16384])
@pytest.mark.parametrize(
    "causal,window", [(True, 11), (True, None), (False, 9), (False, None), (False, 1)]
)
def test_draft_chunked_reference_matches_dense(causal, window, chunk_size):
    generator = torch.Generator().manual_seed(1739)
    q = torch.randn(12, 4, 8, generator=generator, dtype=torch.bfloat16)
    k, v = (torch.randn(9, 4, 2, 8, generator=generator, dtype=q.dtype) for _ in range(2))
    # Repeated/non-monotonic physical pages and partial final pages, as on GPU.
    pages = torch.tensor([3, 1, 8, 3, 6, 2, 1, 0, 8, 5, 1, 4, 7, 3])
    indices = (pages[:, None] * 4 + torch.arange(4)).flatten()
    lengths = [53, 13]
    metadata = {
        "slots": torch.tensor([0, 1, -1]),
        "positions": torch.cat(
            [torch.arange(n - 4, n) + 1000 for n in lengths] + [torch.zeros(4, dtype=torch.int64)]
        ),
        "token_indptr": torch.tensor([0, 53, 66, 66]),
        "token_indices": torch.cat([indices[:53], indices.flip(0)[:13]]),
        "query_length": 4,
        "causal": causal,
        "window": window,
    }
    q[-4:].fill_(torch.nan)  # Inactive queries must never enter the math.
    expected = draft_attention_reference(q, k, v, **metadata, precision=torch.float64)
    actual = chunked_draft_attention_reference(q, k, v, **metadata, chunk_size=chunk_size)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(actual[-4:]) == 0


def test_draft_chunked_reference_large_logits():
    # Online maximum rescaling must not overflow when later chunks dominate.
    q = torch.full((2, 1, 8), 100.0)
    k = torch.arange(12, dtype=torch.float32).view(3, 4, 1, 1).expand(3, 4, 1, 8)
    v = k.flip(0)
    metadata = {
        "slots": torch.tensor([0]),
        "positions": torch.tensor([10, 11]),
        "token_indptr": torch.tensor([0, 12]),
        "token_indices": torch.arange(12),
        "query_length": 2,
        "causal": True,
        "window": None,
    }
    expected = draft_attention_reference(q, k, v, **metadata)
    actual = chunked_draft_attention_reference(q, k, v, **metadata, chunk_size=3)
    torch.testing.assert_close(actual, expected)
