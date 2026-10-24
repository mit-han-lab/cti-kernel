# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Arithmetic and CPU fallback checks for the paired MoE benchmark."""

import pytest
import torch

from benchmark.kernel_tuning.moe_dx_paired import sol_floor, torch_fallback

from ..modules import _moe_reference as ref


@pytest.mark.parametrize(
    "name,bw,mma", (("RTX 5090", 1681, 247.4), ("RTX PRO 6000", 1533, 466.1), ("GB10", 244.6, 124))
)
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("tokens", (1, 8192))
def test_sol_floor(name, bw, mma, backward, tokens):
    # Rank-zero EP4 shard: 3 distinct experts hit by 2 local slots per token.
    factor = 2 if backward else 1
    weight_bytes = 2 * 3 * 2048 * (3 * 512 + 128)
    io_bytes = factor * 4 * tokens * 2048
    flops = factor * 6 * 2048 * (2 * tokens * 512 + tokens * 128)
    want = max((factor * weight_bytes + io_bytes) / (bw * 1e9), flops / (mma * 1e12)) * 1e6
    got = sol_floor(
        name,
        tokens=tokens,
        hidden=2048,
        intermediate=512,
        shared_intermediate=128,
        local_slots=2 * tokens,
        distinct_local=3,
        backward=backward,
    )
    assert got == pytest.approx(want)
    assert (
        sol_floor(
            "unknown",
            tokens=tokens,
            hidden=2048,
            intermediate=512,
            shared_intermediate=128,
            local_slots=0,
            distinct_local=0,
        )
        is None
    )


def test_fallback_ep_partials():
    wt = ref.make_weights(h=32, e=8, ni=32, nis=64)
    x, dy = ref.make_x(16, h=32).requires_grad_(), ref.make_x(16, h=32, seed=2)
    sel = torch.sort((x @ wt["r"].T).float(), dim=1, descending=True, stable=True).indices[:, :2]
    outs, dxs = [], []
    for rank in range(4):
        shard, mapping = ref.shard_weights(wt, 4, rank)
        out = torch_fallback(x, *shard.values(), top_k=2, expert_map=mapping, routing=sel.byte())
        outs.append(out.float())
        dxs.append(torch.autograd.grad(out, x, dy)[0].float())
    oo, dd, _, _ = ref.oracle(x.detach(), *wt.values(), sel, dy, dtype=torch.float32, kappa=False)
    assert (sum(outs) - oo).norm() / oo.norm() < 0.02
    assert (sum(dxs) - dd).norm() / dd.norm() < 0.02
