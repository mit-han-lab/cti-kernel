# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton expert GEMM backend and static launch configuration for frozen MoE dX."""

from __future__ import annotations

import torch

from .._training_backend import _launch
from .._training_backend import _MoEBlock as _MoEBlock
from . import _kernels as K


def _blocks(device: torch.device, ni: int, nis: int) -> dict[str, int]:
    # Static per-arch table (no autotune at call time). A sweep of 48 block shapes (BM, BN in
    # {64, 128}, BK in {32, 64}, 4 or 8 warps, 2-4 stages) at H 2048, I 512, EP 1-4 found this
    # portable set within 2% of the best on the RTX 5090 and RTX PRO 6000; on GB10 (sm121) 128-wide
    # N tiles with two stages were 8-15% faster at 1024-4096 tokens and neutral elsewhere.
    blk = {"bm": 64, "bn": 64, "bk": 64, "warps": 4}
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    if capability == (12, 1):
        blk.update(bn=128, stages=2)
    if min(ni, nis) < 64 and capability is not None and capability[0] == 9:
        # Triton 3.7.1 on sm90 miscompiles the backward's recompute tile at 4 warps when H is one
        # 64-wide K step and the I tile is under 64 (dX off by O(1); test_dims_smoke). 8 warps are
        # correct on every supported H; real I (>= 64) keeps 4 warps, which time faster there.
        blk["warps"] = 8
    return blk


class _TritonBackend:
    device_layout = False
    device_token = False

    def __init__(self, blk):
        self.blk = blk

    def block_m(self, device, tokens: int, ni: int, nis: int) -> int:
        return self.blk["bm"]

    def dw_tiles(self, device, tokens, ni, nis, *, saved=False) -> int:
        return max(ni, nis) // self._tiles(ni, nis, ni)[1]

    def _tiles(self, ni, nis, h):
        blk = self.blk
        bm, bn, bk = blk["bm"], min(blk["bn"], ni, nis), min(blk["bk"], h)
        bnh, bki = min(blk["bn"], h), min(blk["bk"], ni, nis)
        # A 128-wide tile only where it divides the size (H, I and Is are multiples of 64).
        if ni % bn or nis % bn:
            bn = min(64, ni, nis)
        if h % bnh:
            bnh = min(64, h)
        return bm, bn, bk, bnh, bki

    def gate_up(self, x, w13, s13, lay, a, gsave, usave, *, ni, nis, kp1, save) -> None:
        h, e, nmax = x.shape[1], w13.shape[0], max(ni, nis)
        bm, bn, bk, _, _ = self._tiles(ni, nis, h)
        K._fwd_gate_up_kernel[(lay.num_blocks, nmax // bn)](
            x,
            w13,
            s13 if s13 is not None else w13,
            lay.sorted_slot,
            lay.block_expert,
            a,
            gsave,
            usave,
            h,
            ni,
            nis,
            nmax,
            e,
            kp1,
            save,
            bm,
            bn,
            bk,
            **_launch(self.blk),
        )

    def down(self, a, w2, s2, lay, y, *, ni, nis) -> None:
        h, e, nmax = y.shape[1], w2.shape[0], max(ni, nis)
        bm, _, _, bnh, bki = self._tiles(ni, nis, h)
        K._fwd_down_kernel[(lay.num_blocks, h // bnh)](
            a,
            w2,
            s2 if s2 is not None else w2,
            lay.sorted_slot,
            lay.block_expert,
            y,
            h,
            ni,
            nis,
            nmax,
            e,
            bm,
            bnh,
            bki,
            **_launch(self.blk),
        )

    def bwd_slot(
        self,
        x,
        dy,
        w13,
        s13,
        w2,
        s2,
        lay,
        wslot,
        gsave,
        usave,
        gg,
        dwpart,
        *,
        ni,
        nis,
        kp1,
        saved,
    ) -> None:
        h, e, nmax = x.shape[1], w13.shape[0], max(ni, nis)
        bm, bn, bk, _, _ = self._tiles(ni, nis, h)
        nt = dwpart.shape[1]
        K._bwd_slot_kernel[(lay.num_blocks, nt)](
            x,
            dy,
            w13,
            s13 if s13 is not None else w13,
            w2,
            s2 if s2 is not None else w2,
            lay.sorted_slot,
            lay.block_expert,
            wslot,
            gsave if saved else gg,
            usave if saved else gg,
            gg,
            dwpart,
            h,
            ni,
            nis,
            nmax,
            e,
            kp1,
            nt,
            saved,
            bm,
            bn,
            bk,
            **_launch(self.blk),
        )

    def dx_pair(self, gg, w13, s13, lay, dxp, *, ni, nis) -> None:
        h, e, nmax = dxp.shape[1], w13.shape[0], max(ni, nis)
        bm, _, _, bnh, bki = self._tiles(ni, nis, h)
        K._bwd_dx_pair_kernel[(lay.num_blocks, h // bnh)](
            gg,
            w13,
            s13 if s13 is not None else w13,
            lay.sorted_slot,
            lay.block_expert,
            dxp,
            h,
            ni,
            nis,
            nmax,
            e,
            bm,
            bnh,
            bki,
            **_launch(self.blk),
        )


def backend(device, ni, nis, h):
    return _TritonBackend(_blocks(device, ni, nis))
