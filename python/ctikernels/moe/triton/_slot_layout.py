# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The deterministic MoE slot layout and alignment in Triton, for every CUDA device.

Layouts keep route order within each expert (no atomics): bitwise reproducible. Shape-only
grids and no host reads: graph-capturable, with scratch from the caller's (graph) pool.
"""

import torch
import triton

from ._layout import SlotLayout
from ._layout_kernels import (
    _layout_count_kernel,
    _layout_place_kernel,
    _layout_scan_kernel,
)

_CHUNK = 128  # slots per counting / placing program
_SUB = 32  # slots ranked together (a running row carries the rest)
_EB = 8  # buckets per scan program
_WARPS = (1, 1)  # count, scan (one warp each measured fastest)


class _Launcher:
    """Launches the layout kernels: compiled kernels by (kernel, constexprs, warps, pointer
    alignment, int width), so later calls go straight to the compiled launcher without the
    JIT's per-call argument binding; and whether each device takes programmatic dependent
    launch (sm_90+), queried once per device."""

    def __init__(self):
        self.compiled = {}
        self.pdl = {}

    def device_pdl(self, device):
        index = device.index
        if index not in self.pdl:
            self.pdl[index] = torch.cuda.get_device_capability(device)[0] >= 9
        return self.pdl[index]

    def __call__(self, kernel, grid, tensors, ints, consts, num_warps):
        if not hasattr(kernel, "constexprs"):  # the interpreter
            kernel[(grid,)](*tensors, *ints, *consts, False, num_warps=num_warps)
            return
        pdl = self.device_pdl(tensors[0].device)
        consts = consts + (pdl,)
        # Only the caller's tensors (the first three: ids, live, map) vary in dtype and
        # alignment; the rest come 16-byte aligned from _carve.
        dev = tensors[0].device.index
        key = (kernel, consts, num_warps, dev, max(ints) >= 2**31)
        for t in tensors[:3]:
            key += (t.dtype, t.data_ptr() & 15)
        hit = self.compiled.get(key)
        if hit is None:
            compiled = kernel[(grid,)](
                *tensors, *ints, *consts, num_warps=num_warps, launch_pdl=pdl
            )
            self.compiled[key] = (compiled.run, compiled.function, compiled.packed_metadata)
            return
        run, function, packed = hit
        stream = torch._C._cuda_getCurrentRawStream(dev)
        run(grid, 1, 1, stream, function, packed, None, None, None, *tensors, *ints, *consts)


_launch = _Launcher()


def _place_config(nb, nch):
    """(fused, blocks per block program, warps) of the place launch, measured on GB10
    (sm_121): up to a few chunks one fused launch (every program recounting) beats count +
    scan + place."""
    if nb <= 256:
        return (True, 64, 4) if nch <= 8 else (False, 8, 1)
    if nch <= 2:
        return True, 16, 2
    return False, 32, 1


def _pow2(n):
    return triton.next_power_of_2(max(1, n))


def _layout(
    a,
    live,
    mapping,
    slots,
    experts,
    inv,
    total,
    scratch,
    *,
    p,
    nblk,
    sent,
    e,
    g,
    npl,
    bm,
    align,
    k=1,
    shared=False,
    mapped=False,
    invalid=False,
):
    # count -> per-bucket column scan -> placement + block buckets + sentinel fill.
    nb = _pow2(npl + 1)
    nch = triton.cdiv(p, _CHUNK)
    cnt, meta = scratch[: nch * nb], scratch[nch * nb :]
    flags = (align, k, shared, mapped, invalid)
    fused, fb, warps = _place_config(nb, nch)
    if not fused:
        _launch(
            _layout_count_kernel, nch, (a, live, mapping, cnt, meta), (p, e, g, npl, nch),
            flags + (nb, _CHUNK), _WARPS[0],
        )  # fmt: skip
        eb = min(_EB, nb)
        _launch(
            _layout_scan_kernel, nb // eb, (cnt, meta), (nch, npl), (bm, nb, eb, 256), _WARPS[1]
        )
    _launch(
        _layout_place_kernel,
        nch + triton.cdiv(nblk, fb),
        (a, live, mapping, cnt, meta, slots, experts, inv, total),
        (p, slots.numel(), nblk, nch, sent, e, g, npl),
        flags + (bm, _pow2(bm), nb, _CHUNK, _SUB, fb, nb.bit_length() - 1, fused),
        warps,
    )


def _carve(dev, *sizes):
    """One int32 allocation cut into 16-byte aligned views of the given sizes."""
    starts, at = [], 0
    for n in sizes:
        starts.append(at)
        at += -(-n // 4) * 4
    buf = torch.empty(max(at, 1), dtype=torch.int32, device=dev)
    return [buf[s : s + n] for s, n in zip(starts, sizes, strict=True)]


def _scratch_size(p, npl):
    # per-chunk counts, then totals, starts, ends and the scan's arrival counter
    return (triton.cdiv(max(p, 1), _CHUNK) + 3) * _pow2(npl + 1) + 4


def layout_allocations(p, e, bm):
    """Bytes of each allocation ``build_layout`` keeps for ``p`` slots (one carved buffer)."""
    npad = (p + bm - 1) // bm * bm + (e + 1) * bm
    sizes = (npad, npad // bm, p, _scratch_size(p, e + 1))
    return (4 * max(sum(-(-n // 4) * 4 for n in sizes), 1),)


def build_layout(sel, live, e, bm, expert_map=None, *, has_shared=True):
    """The ``SlotLayout`` of ``sel`` [T, k] int32 global ids over ``e`` local experts, slots
    in slot order within each expert. ``expert_map`` maps global ids to local or -1."""
    t, k = sel.shape
    p = t * (k + has_shared)
    npad = (p + bm - 1) // bm * bm + (e + 1) * bm
    slots, experts, inv, scratch = _carve(sel.device, npad, npad // bm, p, _scratch_size(p, e + 1))
    lay = SlotLayout(slots, experts, inv, bm)
    if t == 0:
        lay.sorted_slot.fill_(-1)
        lay.block_expert.fill_(e + 1)
        return lay
    mapped = expert_map is not None
    sel = sel.contiguous()
    _layout(
        sel,
        live,
        expert_map if mapped else sel,
        lay.sorted_slot,
        lay.block_expert,
        lay.inv,
        sel,
        scratch,
        p=p,
        nblk=npad // bm,
        sent=-1,
        e=e,
        g=0,
        npl=e + 1,
        bm=bm,
        align=False,
        k=k,
        shared=has_shared,
        mapped=mapped,
    )
    return lay


def align_allocations(n, block_size, num_experts, *, capacity_experts=None,
                      invalid_bucket=False):  # fmt: skip
    """Bytes of each allocation ``align_block_size`` keeps for ``n`` routes (one carve)."""
    buckets = max(num_experts + invalid_bucket, capacity_experts or 0)
    capacity = n + buckets * (block_size - 1)
    sizes = (
        capacity,
        triton.cdiv(capacity, block_size),
        1,
        _scratch_size(n, num_experts + invalid_bucket),
    )
    return (4 * max(sum(-(-k // 4) * 4 for k in sizes), 1),)


def align_block_size(
    topk_ids,
    block_size,
    num_experts,
    expert_map=None,
    *,
    capacity_experts=None,
    invalid_bucket=False,
):
    """``moe_align_block_size``'s layout over ``topk_ids`` [T, k] (int32/int64 global ids,
    -1 padding): routes sorted by local expert (route order within one), each expert padded
    to whole blocks of ``block_size`` (any positive size); sentinel ``topk_ids.numel()``.
    Returns ``(sorted_ids, expert_ids, total)``. Routes to no local expert get no slot, or
    with ``invalid_bucket`` fill trailing blocks of expert -1 (zero rows in the GEMM)."""
    n = topk_ids.numel()
    dev = topk_ids.device
    buckets = max(num_experts + invalid_bucket, capacity_experts or 0)
    capacity = n + buckets * (block_size - 1)
    if capacity >= 2**31:
        raise ValueError("aligned route capacity must fit int32")
    nblk = triton.cdiv(capacity, block_size)
    sorted_ids, experts, total, scratch = _carve(
        dev, capacity, nblk, 1, _scratch_size(n, num_experts + invalid_bucket)
    )
    ids = topk_ids.contiguous().view(-1)
    mapped = expert_map is not None
    if n == 0:
        sorted_ids.fill_(0)
        experts.fill_(-1)
        total.zero_()
        return sorted_ids, experts, total
    _layout(
        ids,
        ids,
        expert_map if mapped else ids,
        sorted_ids,
        experts,
        sorted_ids,
        total,
        scratch,
        p=n,
        nblk=nblk,
        sent=n,
        e=num_experts,
        g=expert_map.numel() if mapped else num_experts,
        npl=num_experts + invalid_bucket,
        bm=block_size,
        align=True,
        mapped=mapped,
        invalid=invalid_bucket,
    )
    return sorted_ids, experts, total
