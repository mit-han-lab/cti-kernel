# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Capacity rejection uses only small buffers; included in ordinary GPU CI."""

import pytest
import torch

from ctikernels.attention._draft import cute
from ctikernels.attention.draft import DraftAttention, _provider

from ._dflash import Fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("causal,window", [(False, None), (False, 2048), (True, 4096)])
def test_flashinfer_capacity_rejected_before_large_allocation(monkeypatch, causal, window):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    geometry = {
        "query_length": 8,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "page_size": 16,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
    }
    capacity = {"batch_size": 1, "max_kv_tokens": 2**31 - 1, "device": torch.device("cuda")}
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        DraftAttention.workspace_size(**geometry, **capacity, provider="flashinfer")

    f = Fixture(1, 8, 16, 16, 4, causal, window, c=16, provider="flashinfer", head_dim=256)
    # A stride-zero view advertises C without allocating its multi-GiB storage.
    # Admission must reject it before checking contiguity or other metadata sizes.
    tokens = f.token_idx[:1].expand(capacity["max_kv_tokens"])
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        DraftAttention(
            *f.workspaces,
            request_indices=f.slots,
            positions=f.positions,
            token_indptr=f.token_ptr,
            token_indices=tokens,
            query_indptr=f.query_ptr,
            page_indptr=f.page_ptr,
            page_indices=f.page_idx,
            last_page_lengths=f.last,
            **geometry,
            provider="flashinfer",
        )
    try:
        cute.check_support(**geometry, **capacity)
    except NotImplementedError:
        with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
            DraftAttention.workspace_size(**geometry, **capacity)
    else:
        assert _provider(None, geometry | capacity) is cute
        assert DraftAttention.workspace_size(
            **geometry, **capacity
        ) == DraftAttention.workspace_size(**geometry, **capacity, provider="cute")
