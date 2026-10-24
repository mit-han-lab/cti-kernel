# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Explicit provider coverage for the DRAFT contract tests."""

import pytest
import torch


@pytest.fixture(params=["cute", "flashinfer", "triton"])
def draft_provider(request):
    if request.param == "cute":
        from ctikernels.attention._draft.cute import check_support

        try:
            check_support(
                torch.device("cuda"), query_length=16, num_qo_heads=32, num_kv_heads=8, head_dim=128
            )
        except NotImplementedError as error:
            pytest.skip(str(error))
    elif request.param == "triton":
        if torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("the Triton DRAFT provider needs compute capability 8.9+")
    else:
        pytest.importorskip("flashinfer")
    return request.param
