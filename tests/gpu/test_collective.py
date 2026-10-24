# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Single-device coverage; compilation never starts a peer-waiting exchange."""

import pytest
import torch

from ctikernels.collective._host_exchange.cute.exchange import compile_kernels
from ctikernels.collective.inference import AllReduceAddRMSNorm, _support_reason, support
from ctikernels.norm.inference import add_rms_norm_


def test_local_matches_norm():
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(17, 2048, device=device, dtype=torch.bfloat16)
    r, w = torch.randn_like(x), torch.randn(2048, device=device, dtype=x.dtype)
    expected = add_rms_norm_(x.clone(), r.clone(), w, 1e-6, zero_centered=True)
    with AllReduceAddRMSNorm(
        rank=0, world_size=1, cpu_group=None, device=device, hidden=2048, max_tokens=8, eps=1e-6
    ) as op:
        assert op.provider == "local"
        actual = op(x, r, w)
    for a, b in zip(actual, expected, strict=False):
        assert torch.equal(a, b)


def test_local_support_reason():
    device = torch.device("cuda", torch.cuda.current_device())
    expected = _support_reason(
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        2,
        2048,
        torch.bfloat16,
        True,
        True,
    )
    assert (
        support(device, world_size=2, hidden=2048, dtype=torch.bfloat16, zero_centered=True)
        == expected
    )


@pytest.mark.parametrize("hidden", [2048, 4096])
def test_compile_flag_and_producer_consumer(hidden):
    slots = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    count = slots * 44 + 1
    cache = compile_kernels(hidden, [1, count], slots, 2, 0, rows_cap=count)
    assert {key[0] for key in cache} == {"flag", "pc"}
