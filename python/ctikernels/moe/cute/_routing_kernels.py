# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe DSL slot layout kernel; no inter-CTA barriers or host reads."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, const_expr


class Layout:
    def __init__(self, e, k, shared, mapped, bm, small):
        self.e, self.k, self.kp1 = e, k, k + shared
        self.shared, self.mapped, self.bm, self.small = shared, mapped, bm, small

    @cute.jit
    def bucket(self, sel, live, mapping, p):
        t, j = p // self.kp1, p % self.kp1
        bucket = Int32(self.e + 1)
        if live[t]:
            if j < self.k:
                bucket = sel[t, j]
                if const_expr(self.mapped):
                    local = mapping[bucket]
                    bucket = local if local >= 0 else self.e + 1
            else:
                bucket = Int32(self.e)
        return bucket

    @cute.jit
    def __call__(
        self, sel, live, mapping, counts, cursor, slots, experts, inv, stream: cuda.CUstream
    ):
        if const_expr(self.small):
            self.scan(sel, live, mapping, counts, cursor, slots, experts, inv).launch(
                grid=(1, 1, 1), block=(256, 1, 1), stream=stream
            )
        else:
            self.zero(counts).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
            self.place(sel, live, mapping, counts, cursor, slots, inv, False).launch(
                grid=(cute.ceil_div(inv.shape[0], 256), 1, 1), block=(256, 1, 1), stream=stream
            )
            self.scan(sel, live, mapping, counts, cursor, slots, experts, inv).launch(
                grid=(1, 1, 1), block=(256, 1, 1), stream=stream
            )
            self.place(sel, live, mapping, counts, cursor, slots, inv, True).launch(
                grid=(cute.ceil_div(inv.shape[0], 256), 1, 1), block=(256, 1, 1), stream=stream
            )

    @cute.kernel
    def zero(self, counts):
        tid, _, _ = cute.arch.thread_idx()
        for b in range(tid, self.e + 2, 256):
            counts[b] = 0

    @cute.kernel
    def place(self, sel, live, mapping, counts, cursor, slots, inv, scatter: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        p = bid * 256 + tid
        if p < inv.shape[0]:
            b = self.bucket(sel, live, mapping, p)
            if const_expr(scatter):
                pos = Int32(-1)
                if b <= self.e:
                    pos = cute.arch.atomic_add(cursor.iterator + b, Int32(1))
                    slots[pos] = p
                inv[p] = pos
            else:
                cute.arch.atomic_add(counts.iterator + b, Int32(1))

    @cute.kernel
    def scan(self, sel, live, mapping, counts, cursor, slots, experts, inv):
        tid, _, _ = cute.arch.thread_idx()
        alloc = utils.SmemAllocator()
        ends = alloc.allocate_tensor(Int32, cute.make_layout(self.e + 2), byte_alignment=16)
        curs = alloc.allocate_tensor(Int32, cute.make_layout(self.e + 2), byte_alignment=16)
        for b in range(tid, self.e + 2, 256):
            if const_expr(self.small):
                ends[b] = 0
            else:
                ends[b] = counts[b]
        cute.arch.sync_threads()
        if const_expr(self.small):
            for p in range(tid, inv.shape[0], 256):
                b = self.bucket(sel, live, mapping, p)
                cute.arch.atomic_add(ends.iterator + b, Int32(1))
            cute.arch.sync_threads()
        if tid == 0:
            offset = Int32(0)
            for b in range(self.e + 1):
                n = ends[b]
                curs[b] = offset
                if const_expr(not self.small):
                    cursor[b] = offset
                offset += ((n + self.bm - 1) // self.bm) * self.bm
                ends[b] = offset
        cute.arch.sync_threads()
        for row in range(tid, slots.shape[0], 256):
            slots[row] = -1
        for block in range(tid, experts.shape[0], 256):
            # upper_bound reproduces searchsorted(right=True), including empty
            # buckets and the E+1 sentinel for ALL blocks past the padded end.
            lo, hi = Int32(0), Int32(self.e + 1)
            while lo < hi:
                mid = (lo + hi) // 2
                if ends[mid] <= block * self.bm:
                    lo = mid + 1
                else:
                    hi = mid
            experts[block] = lo
        if const_expr(self.small):
            cute.arch.sync_threads()
            for p in range(tid, inv.shape[0], 256):
                b = self.bucket(sel, live, mapping, p)
                pos = Int32(-1)
                if b <= self.e:
                    pos = cute.arch.atomic_add(curs.iterator + b, Int32(1))
                    slots[pos] = p
                inv[p] = pos
