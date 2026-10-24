# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Hopper-only native workspace qualification; excluded before SM120 collection."""

import pytest
import torch

from .test_paged_attention_workspace import _workspace_bound


@pytest.mark.parametrize("page", [1, 16])
def test_workspace_bound_sm90(page):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3/SM90 qualification requires Hopper")
    _workspace_bound("fa3", page)
