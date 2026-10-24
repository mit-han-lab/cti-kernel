# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-testable specification of the unordered device slot layout."""

import torch

from ctikernels.moe.triton._layout import SlotLayout


def layout_reference(sel, live, e, bm, expert_map=None, *, has_shared=True):
    """Reverse arrival order models a legal atomic scatter, deliberately unstable."""
    bucket = sel.long()
    if expert_map is not None:
        local = expert_map[bucket]
        bucket = torch.where(local >= 0, local, e + 1)
    if has_shared:
        bucket = torch.cat((bucket, torch.full_like(bucket[:, :1], e)), 1)
    bucket = torch.where(live[:, None], bucket, e + 1).flatten()
    p = bucket.numel()
    npad = (p + bm - 1) // bm * bm + (e + 1) * bm
    slots = torch.full((npad,), -1, dtype=torch.int32, device=sel.device)
    inv = torch.full((p,), -1, dtype=torch.int32, device=sel.device)
    experts = torch.full((npad // bm,), e + 1, dtype=torch.int32, device=sel.device)
    offset = 0
    for b in range(e + 1):
        arrivals = (bucket == b).nonzero().flatten().flip(0)
        n = arrivals.numel()
        padded = (n + bm - 1) // bm * bm
        slots[offset : offset + n] = arrivals.to(torch.int32)
        inv[arrivals] = torch.arange(offset, offset + n, dtype=torch.int32, device=sel.device)
        experts[offset // bm : (offset + padded) // bm] = b
        offset += padded
    return SlotLayout(slots, experts, inv, bm)


def assert_layout(actual, expected):
    """Check exact bucket multisets, padding, coverage and inverse, on CPU or GPU."""
    assert actual.block_m == expected.block_m
    for name in ("sorted_slot", "block_expert", "inv"):
        got, want = getattr(actual, name), getattr(expected, name)
        assert got.dtype == torch.int32 and got.shape == want.shape
    assert torch.equal(actual.block_expert, expected.block_expert)
    assert torch.equal(actual.inv < 0, expected.inv < 0)
    slots = actual.sorted_slot
    present = slots >= 0
    rows = torch.arange(slots.numel(), device=slots.device)[present]
    assert torch.equal(actual.inv[slots[present].long()].long(), rows)
    assert torch.equal(
        slots[present].sort().values, expected.sorted_slot[expected.sorted_slot >= 0].sort().values
    )
    assert torch.all(slots[~present] == -1)
    # Sorting each whole expert's padded region permits reordering across blocks.
    for e in actual.block_expert.unique().tolist():
        blocks = actual.block_expert == e
        a = slots.view(-1, actual.block_m)[blocks].flatten().sort().values
        b = expected.sorted_slot.view(-1, expected.block_m)[blocks].flatten().sort().values
        assert torch.equal(a, b)


def old_route(logits, s, k, routing=None, valid=None, *, has_shared=True):
    t = logits.shape[0]
    live = torch.arange(t, device=logits.device) < (t if valid is None else valid.clamp(0, t))
    sel = (
        torch.sort(logits.float(), descending=True, stable=True).indices[:, :k]
        if routing is None
        else routing.long()
    )
    sel = torch.where(live[:, None], sel, 0)
    z = logits.float().gather(1, sel)
    ws = torch.softmax(z - z.amax(1, keepdim=True), dim=1)
    if has_shared:
        ws = torch.cat((ws, s.sigmoid()[:, None]), 1)
    return sel, torch.where(live[:, None], ws, 0), live, sel.to(torch.uint8)


def slot_expert(sel, live, e, expert_map=None, has_shared=True):
    """Slot buckets: local expert, e for the shared slot, e + 1 for remote or dead."""
    if expert_map is not None:
        local = expert_map[sel].long()
        sel = torch.where(local >= 0, local, e + 1)
    se = sel
    if has_shared:
        se = torch.cat([sel, torch.full_like(sel[:, :1], e)], dim=1)
    return torch.where(live[:, None], se, e + 1).reshape(-1)


def build_slot_layout(slot_expert: torch.Tensor, num_experts: int, block_m: int) -> SlotLayout:
    """Layout for ``slot_expert`` [P] int64 (0..E-1 routed, E shared, E+1 remote/masked)."""

    p = slot_expert.numel()
    dev = slot_expert.device
    nbucket = num_experts + 2
    npad = (p + block_m - 1) // block_m * block_m + (num_experts + 1) * block_m
    counts = torch.zeros(nbucket, dtype=torch.int64, device=dev)
    counts.scatter_add_(0, slot_expert, torch.ones_like(slot_expert))
    off = torch.cumsum(counts, 0) - counts
    padded = (counts[: num_experts + 1] + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_off = pad_end - padded
    order = torch.argsort(slot_expert, stable=True)
    exp_sorted = slot_expert[order]
    rank = torch.arange(p, device=dev) - off[exp_sorted]
    live = exp_sorted <= num_experts
    pos = torch.where(live, pad_off[exp_sorted.clamp(max=num_experts)] + rank, npad)
    sorted_slot = torch.full((npad + 1,), -1, dtype=torch.int32, device=dev)
    sorted_slot.scatter_(0, pos, order.to(torch.int32))
    inv = torch.zeros(p, dtype=torch.int32, device=dev)
    inv.scatter_(0, order, torch.where(live, pos, -1).to(torch.int32))
    starts = torch.arange(0, npad, block_m, device=dev)
    block_expert = torch.searchsorted(pad_end, starts, right=True).to(torch.int32)
    return SlotLayout(sorted_slot[:npad], block_expert, inv, block_m)
