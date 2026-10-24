# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Expand engine tables into a private, compact token CSR."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.runtime import make_fake_stream


class TableCSR:
    def __init__(self, query, tiles, use_pdl, row_metadata=False):
        self.query, self.tiles, self.use_pdl = query, tiles, use_pdl
        self.row_metadata = row_metadata

    @cute.jit
    def key(self, requests, b):
        key = requests[b]
        if cutlass.const_expr(self.row_metadata):
            key = b
        return key

    @cute.jit
    def length(self, requests, frontier, done, b, maximum):
        key = self.key(requests, b)
        f = frontier[key]
        length = Int32(self.query)
        if (done[key] == 0) & (f >= 0) & (f < maximum):
            length = Int32(f) + self.query
        return length

    @cute.jit
    def __call__(
        self,
        requests: cute.Tensor,
        frontier: cute.Tensor,
        done: cute.Tensor,
        table: cute.Tensor,
        ptr: cute.Tensor,
        indices: cute.Tensor,
        stride: Int64,
        guard: Int32,
        maximum: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(requests, frontier, done, table, ptr, indices, stride, guard, maximum).launch(
            grid=(self.tiles, requests.shape[0], 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        requests: cute.Tensor,
        frontier: cute.Tensor,
        done: cute.Tensor,
        table: cute.Tensor,
        ptr: cute.Tensor,
        indices: cute.Tensor,
        stride: Int64,
        guard: Int32,
        maximum: Int32,
    ):
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        tile, b, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        start = Int64(0)
        for row in range(b):
            start += Int64(self.length(requests, frontier, done, row, maximum))
        r = requests[b]
        key = self.key(requests, b)
        f = frontier[key]
        length, prefix = Int32(self.query), Int32(0)
        if (done[key] == 0) & (f >= 0) & (f < maximum):
            length = Int32(f) + self.query
            prefix = min(length, maximum)
        if tile == 0 and tid == 0:
            ptr[b + 1] = Int32(start + length)
            if b == 0:
                ptr[0] = 0
        for j in range(Int64(tile * 128 + tid), Int64(length), self.tiles * 128):
            value = Int32(0)
            if j < prefix:
                value = table[Int64(r) * stride + j]
            else:
                offset = Int32(j)
                if prefix != 0:
                    offset = min(Int32(j) - maximum, self.query - 1)
                value = guard + b * self.query + offset
            indices[start + j] = value
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@lru_cache(maxsize=None)
def compile_table_csr(query, capacity, arch, row_metadata=False):
    fake = cute.runtime.make_fake_tensor

    def tensor(kind, alignment=4):
        return fake(kind, (cute.sym_int(),), (1,), assumed_align=alignment)

    return cute.compile(
        TableCSR(
            query,
            min(128, (capacity + 127) // 128),
            int(arch[3:].rstrip("a")) >= 90,
            row_metadata,
        ),
        tensor(Int32),
        tensor(Int64, 8),
        tensor(cutlass.Uint8, 1),
        tensor(Int32),
        tensor(Int32),
        tensor(Int32),
        Int64(0),
        Int32(0),
        Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )
