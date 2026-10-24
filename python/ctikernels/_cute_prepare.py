# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe DRAFT query/page CSR preparation."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass.utils import SmemAllocator

# CTAs per request that publish its page indices (grid-stride over the pages).
PAGE_TILES = 32


class Prepare:
    def __init__(self, query, page, write_pages=True):
        self.query, self.page = query, page
        self.write_pages = write_pages

    @cute.jit
    def __call__(
        self,
        slots: cute.Tensor,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        qp: cute.Tensor,
        pp: cute.Tensor,
        pi: cute.Tensor,
        last: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tiles = Int32(1)
        if const_expr(self.write_pages):
            # At most PAGE_TILES CTAs per request stride over its pages: a grid sized by
            # the page capacity launched ~101k mostly idle CTAs at B=99, C=131079.
            tiles = Int32(min((Int64(pi.shape[0]) // slots.shape[0] + 127) // 128, PAGE_TILES))
        self.kernel(slots, ptr, idx, qp, pp, pi, last).launch(
            # The rounded page capacity can exceed int32 before division by 128.
            grid=(tiles, slots.shape[0], 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        slots: cute.Tensor,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        qp: cute.Tensor,
        pp: cute.Tensor,
        pi: cute.Tensor,
        last: cute.Tensor,
    ):
        cute.arch.griddepcontrol_wait()
        tile, b, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        partial = SmemAllocator().allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        start, length, count = Int32(0), Int32(0), Int32(0)
        if slots[b] != -1:
            start = ptr[b]
            length = ptr[b + 1] - start
            count = Int32((Int64(length) + self.page - 1) // self.page)
        # Tiles past this request's pages leave before the prefix.
        busy = tile == 0
        if const_expr(self.write_pages):
            busy = busy | (tile * 128 < count)
        if busy:
            # Prefix over the earlier requests, rows strided over the CTA's 128 threads.
            # Page counts and their sum fit: pages <= tokens <= B*C < 2**31.
            queries, pages = Int32(0), Int32(0)
            for row in range(tid, b, 128):
                if slots[row] != -1:
                    queries += self.query
                    pages += Int32((Int64(ptr[row + 1] - ptr[row]) + self.page - 1) // self.page)
            queries = cute.arch.warp_redux_sync(queries, "add")
            pages = cute.arch.warp_redux_sync(pages, "add")
            if tid % 32 == 0:
                partial[tid // 32] = queries
                partial[4 + tid // 32] = pages
            cute.arch.barrier()
            queries = partial[0] + partial[1] + partial[2] + partial[3]
            pages = partial[4] + partial[5] + partial[6] + partial[7]
            if slots[b] != -1:
                if tid == 0 and tile == 0:
                    last[b] = (length - 1) % self.page + 1
                    qp[b + 1] = queries + self.query
            else:
                if tid == 0 and tile == 0:
                    last[b] = 0
                    qp[b + 1] = queries
            if tid == 0 and tile == 0:
                pp[b + 1] = pages + count
                if b == 0:
                    pp[0], qp[0] = 0, 0
            if const_expr(self.write_pages):
                stride = Int32(cute.arch.grid_dim()[0] * 128)
                # Admitted page starts and compact destinations are below B*C.
                for page in range(tile * 128 + tid, count, stride):
                    pi[pages + page] = idx[start + page * self.page] // self.page
        cute.arch.griddepcontrol_launch_dependents()
