# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""moe.lora.moe_lora_sorted_shrink_: the LoRA shrink over expert-sorted rows (Triton)."""

import pytest
import torch

from ctikernels.moe.lora import moe_lora_sorted_shrink_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sorted_rows(tokens, top_k, experts, gen):
    ids = torch.randint(-1, experts, (tokens, top_k), generator=gen)
    pairs = torch.arange(tokens * top_k)
    local = ids.flatten() >= 0
    order = torch.argsort(ids.flatten()[local], stable=True)
    pair = pairs[local][order]
    counts = torch.bincount(ids.flatten()[local], minlength=experts)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    # Capacity beyond the live rows, as PREP's workspace has.
    pad = torch.full((5,), 7, dtype=torch.long)
    return ids, torch.cat((pair // top_k, pad)), torch.cat((pair, pad)), offsets


@pytest.mark.parametrize("tokens,rank,slices", [(1, 16, 2), (64, 8, 2), (300, 128, 2), (77, 16, 1)])
def test_matches_fp32_reference_and_leaves_remote_pairs(tokens, rank, slices):
    gen = torch.Generator().manual_seed(tokens + rank)
    top_k, experts, k = 6, 32, 512
    ids, token, pair, offsets = _sorted_rows(tokens, top_k, experts, gen)
    x = torch.randn(tokens, k, generator=gen).bfloat16().cuda()
    a = [(torch.randn(experts, rank, k, generator=gen) * k**-0.5).bfloat16().cuda()]
    a = a * slices if slices == 1 else [a[0], (a[0] * 0.5).contiguous()]
    z = torch.full((tokens * top_k, slices, rank), float("nan"), device="cuda")
    moe_lora_sorted_shrink_(z, x, a, token.int().cuda(), pair.int().cuda(), offsets.int().cuda())
    flat = ids.flatten().cuda()
    live = flat >= 0
    assert z[~live].isnan().all()  # pairs without a local row: untouched
    rows = x.float()[torch.arange(tokens * top_k, device="cuda")[live] // top_k]
    for s in range(slices):
        ref = torch.einsum("prk,pk->pr", a[s].float()[flat[live]], rows)
        torch.testing.assert_close(z[live, s], ref, rtol=1e-4, atol=1e-4)
    again = torch.full_like(z, float("nan"))
    moe_lora_sorted_shrink_(
        again, x, a, token.int().cuda(), pair.int().cuda(), offsets.int().cuda()
    )
    assert torch.equal(z.nan_to_num(), again.nan_to_num())  # deterministic


def test_row_order_writes_entry_per_sorted_row():
    gen = torch.Generator().manual_seed(9)
    tokens, top_k, experts, k, rank = 50, 6, 32, 256, 16
    ids, token, pair, offsets = _sorted_rows(tokens, top_k, experts, gen)
    x = torch.randn(tokens, k, generator=gen).bfloat16().cuda()
    a = [(torch.randn(experts, rank, k, generator=gen) * k**-0.5).bfloat16().cuda() for _ in "gu"]
    by_pair = torch.zeros(tokens * top_k, 2, rank, device="cuda")
    by_row = torch.full_like(by_pair, float("nan"))
    t, p, o = token.int().cuda(), pair.int().cuda(), offsets.int().cuda()
    moe_lora_sorted_shrink_(by_pair, x, a, t, p, o)
    moe_lora_sorted_shrink_(by_row, x, a, t, None, o)
    live = int(offsets[-1])
    assert torch.equal(by_row[:live], by_pair[pair[:live].cuda().long()])
    assert by_row[live:].isnan().all()
