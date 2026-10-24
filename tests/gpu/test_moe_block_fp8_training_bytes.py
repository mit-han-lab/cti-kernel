# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""``BlockFP8MoE.training_bytes`` against the allocator, at Qwen3.5-35B-A3B-FP8 widths.

Frozen experts train as the engine runs them: ``BlockFP8MoE``'s forward under an autograd
function that saves the input, the FP32 routing weights and int32 ids, and ``dx`` back. The
measured saved bytes and backward peak must sit in ``[function - 64 KiB, function + 8 MiB]``,
and the forward's transient peak within ``saved + backward``."""

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE

from .test_moe_block_fp8_cute import random_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="block-FP8 MoE needs compute capability 8.9+",
)
E, H, INTER, TOP_K = 256, 2048, 512, 8
KIB, MIB = 1024, 1024 * 1024


class _Frozen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ids, op):
        ctx.save_for_backward(x, weights, ids)
        ctx.op = op
        return op(x.detach(), weights.detach(), ids)

    @staticmethod
    def backward(ctx, dy):
        x, weights, ids = ctx.saved_tensors
        dx, dw = ctx.op.dx(x.detach(), weights.detach(), ids, dy.contiguous())
        return dx, dw, None, None


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    return random_weights(E, H, INTER)


def _step(op, t, groups):
    g = torch.Generator(device="cuda").manual_seed(t)
    x = torch.randn(t, H, generator=g, device="cuda").bfloat16().requires_grad_()
    logits = torch.randn(t, groups, generator=g, device="cuda")
    w, ids = logits.softmax(-1).topk(TOP_K, -1)
    return x, w.detach().requires_grad_(), ids.int()


@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("t", [1, 300, 4096])
def test_bytes_bound_the_allocator(weights, t, ep):
    local = E // ep
    kw = {}
    if ep > 1:
        mapping = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        mapping[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
        kw = {"expert_map": mapping, "num_global_experts": E}
    op = BlockFP8MoE(*(p[:local] for p in weights), top_k=TOP_K, **kw)
    saved, backward = op.training_bytes(t)
    for _ in range(2):  # compile and cache before measuring
        x, w, ids = _step(op, t, E)
        _Frozen.apply(x, w, ids, op).backward(torch.randn(t, H, device="cuda").bfloat16())
    del x, w, ids
    torch.cuda.synchronize()

    base = torch.cuda.memory_allocated()
    x, w, ids = _step(op, t, E)
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = _Frozen.apply(x, w, ids, op)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - start
    held = torch.cuda.memory_allocated() - base - out.untyped_storage().nbytes()
    dy = torch.randn_like(out)
    del ids  # the graph's saved copy is all that remains
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out.backward(dy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - start
    print(f"T={t} EP{ep}: saved {held} vs {saved}, backward {peak} vs {backward}, "
          f"forward transient {transient}")  # fmt: skip
    assert saved - 64 * KIB <= held <= saved + 8 * MIB
    assert backward - 64 * KIB <= peak <= backward + 8 * MIB
    assert transient <= saved + backward
