# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""moe.cute._lora_provider.shrink_: the pair kernel's shrink-only launch (W4A8 LoRA small T).

Contract: z [P, slices, r] FP32 (or BF16) in pair order; pairs whose id is not a local
expert keep their contents."""

import pytest
import torch

from ctikernels.moe._lora_weights import FactorSlices

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="the CuTe pair kernel runs on sm_120/sm_121",
)


@pytest.mark.parametrize("tokens,rank,slices", [(1, 16, 2), (4, 64, 2), (3, 8, 1)])
def test_pair_order_fp32_and_remote_untouched(tokens, rank, slices):
    from ctikernels.moe.cute import _lora_provider

    gen = torch.Generator().manual_seed(tokens * rank)
    top_k, experts, k = 6, 16, 1024
    ids = torch.randint(-1, experts + 2, (tokens, top_k), generator=gen, dtype=torch.int32)
    x = torch.randn(tokens, k, generator=gen).bfloat16().cuda()
    a = [(torch.randn(experts, rank, k, generator=gen) * k**-0.5).bfloat16().cuda()]
    a = a if slices == 1 else [a[0], (a[0] * -0.5).contiguous()]
    z = torch.full((tokens * top_k, slices, rank), 7.0, device="cuda")
    _lora_provider.shrink_(z, x, FactorSlices(a), ids.cuda(), top_k)
    flat = ids.flatten().cuda()
    local = (flat >= 0) & (flat < experts)
    assert (z[~local] == 7.0).all()  # -1 and out-of-range ids: untouched
    rows = x.float()[torch.arange(tokens * top_k, device="cuda")[local] // top_k]
    for s in range(slices):
        ref = torch.einsum("prk,pk->pr", a[s].float()[flat[local].long()], rows)
        torch.testing.assert_close(z[local, s], ref, rtol=1e-4, atol=1e-4)
