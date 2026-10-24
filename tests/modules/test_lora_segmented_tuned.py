# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The Triton segmented LoRA's TunedOp: host-rule defaults lie in the declared space."""

import pytest

pytest.importorskip("triton")

from ctikernels._runtime.tables import DeviceClass
from ctikernels.linear.triton import _lora_segmented as impl


@pytest.mark.parametrize(
    "device", [DeviceClass(10, 3, 152), DeviceClass(12, 0, 170), DeviceClass(12, 1, 48)]
)
@pytest.mark.parametrize("m", [1, 4, 33, 2048])
@pytest.mark.parametrize("rank", [16, 64, 128, 512])
@pytest.mark.parametrize("sizes", [(4096,), (8192, 1024, 1024)])
def test_defaults_lie_in_the_space(device, m, rank, sizes):
    shape = {"m": m, "adapters": 2, "k": 4096, "sizes": sizes, "rank": rank}
    config = impl.TUNED.lookup(device, **shape, low="rounded", dtype="bfloat16")
    assert config["rank_tile"] <= rank
    splits, per_split = impl.split_k(4096, config["splits"], config["shrink_block_k"])
    assert splits * per_split >= 4096 and per_split % config["shrink_block_k"] == 0
