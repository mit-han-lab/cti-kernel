# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Atomic slot placement, with shape-only launch grids.

Placement within each expert is nondeterministic. Each GEMM row is independent
and inv recovers slot order, so placement does not change per-row results.
Warm up each specialization before graph capture; compilation never reads a GPU.
"""

import threading

import torch

from ..triton._layout import SlotLayout

_LOCK = threading.RLock()
_COMPILED = {}
SMALL_SLOTS = 4096


def compile_layout(e, k, shared, mapped, bm, small, capability):
    import cutlass.cute as cute
    from cutlass import Boolean, Int32
    from cutlass.runtime import make_fake_stream

    from ._routing_kernels import Layout

    t, rows, blocks, slots, global_e = (cute.sym_int() for _ in range(5))
    fake = cute.runtime.make_fake_tensor
    args = (
        fake(Int32, (t, k), (k, 1), assumed_align=4),
        fake(Boolean, (t,), (1,), assumed_align=1),
        fake(Int32, (global_e,), (1,), assumed_align=4),
        fake(Int32, (e + 2,), (1,), assumed_align=4),
        fake(Int32, (e + 2,), (1,), assumed_align=4),
        fake(Int32, (rows,), (1,), assumed_align=4),
        fake(Int32, (blocks,), (1,), assumed_align=4),
        fake(Int32, (slots,), (1,), assumed_align=4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(
        Layout(e, k, shared, mapped, bm, small),
        *args,
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )


def _compiled(kind, key):
    fn = _COMPILED.get((kind, key))
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up this MoE CuTe layout shape before CUDA graph capture")
        with _LOCK:
            fn = _COMPILED.get((kind, key))
            if fn is None:
                fn = compile_layout(*key)
                _COMPILED[kind, key] = fn
    return fn


def layout_allocations(p, e, bm):
    """Bytes of each allocation ``build_layout`` keeps for ``p`` slots."""
    npad = (p + bm - 1) // bm * bm + (e + 1) * bm
    return (4 * npad, 4 * (npad // bm), 4 * p)


def build_layout(sel, live, e, bm, expert_map=None, *, has_shared=True, capability):
    """Build the existing SlotLayout contract without materializing slot experts."""
    t, k = sel.shape
    p = t * (k + has_shared)
    npad = (p + bm - 1) // bm * bm + (e + 1) * bm
    small = p <= SMALL_SLOTS
    empty = lambda n: torch.empty(n, dtype=torch.int32, device=sel.device)
    lay = SlotLayout(empty(npad), empty(npad // bm), empty(p), bm)
    if t == 0:
        lay.sorted_slot.fill_(-1)
        lay.block_expert.fill_(e + 1)
        return lay
    counts, cursor = empty(e + 2), empty(e + 2)
    fn = _compiled("layout", (e, k, has_shared, expert_map is not None, bm, small, capability))
    fn(
        sel,
        live,
        sel.reshape(-1) if expert_map is None else expert_map,
        counts,
        cursor,
        lay.sorted_slot,
        lay.block_expert,
        lay.inv,
    )
    return lay
