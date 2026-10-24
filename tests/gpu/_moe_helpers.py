# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import torch

from ctikernels.moe import moe_align_block_size_local


def _check_align(ids, mapping, local, block=16):
    sorted_ids, experts, total = moe_align_block_size_local(ids, block, mapping, local)
    ids, mapping = ids.cpu().flatten(), mapping.cpu()
    sorted_ids, experts, total = sorted_ids.cpu(), experts.cpu(), int(total.item())
    n = ids.numel()
    assert len(sorted_ids) == n + local * (block - 1)
    assert len(experts) == (len(sorted_ids) + block - 1) // block
    mapped = torch.full_like(ids, -1)
    valid = ids >= 0
    mapped[valid] = mapping[ids[valid].long()]
    counts = torch.bincount(mapped[mapped >= 0].long(), minlength=local)
    expected_total = int((((counts + block - 1) // block) * block).sum())
    assert total == expected_total
    assert (sorted_ids[total:] == n).all()
    assert (experts[total // block :] == -1).all()
    assert ((experts[: total // block] >= 0) & (experts[: total // block] < local)).all()
    if total:
        block_experts = experts[: total // block].repeat_interleave(block)
        assert torch.equal(experts[: total // block], experts[: total // block].sort().values)
        real = sorted_ids[:total]
        is_route = real < n
        assert (real[~is_route] == n).all()
        actual_keys = block_experts[is_route].long() * (n + 1) + real[is_route].long()
        routes = torch.where(mapped >= 0)[0]
        expected_keys = mapped[routes].long() * (n + 1) + routes
        assert torch.equal(actual_keys.sort().values, expected_keys.sort().values)
        expected_blocks = torch.arange(local, dtype=torch.int32).repeat_interleave(
            (counts + block - 1) // block
        )
        assert torch.equal(experts[: total // block], expected_blocks)
