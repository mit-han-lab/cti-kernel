# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Test-only delayed PDL producers and persistent row ownership checker."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T
from cutlass.runtime import make_fake_stream


@cute.jit
def timer():
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class DelayedCopy:
    @cute.jit
    def __call__(self, dst: tuple, src: tuple, stream: cuda.CUstream):
        self.kernel(dst, src).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, dst: tuple, src: tuple):
        cute.arch.griddepcontrol_launch_dependents()
        begin = timer()
        while timer() - begin < Uint64(90000):
            pass
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        for j in cutlass.range_constexpr(len(dst)):
            count = cute.size(dst[j].shape)
            k = Int64(tid)
            while k < count:
                if cutlass.const_expr(len(dst[j].shape) == 2):
                    row, col = k // dst[j].shape[1], k % dst[j].shape[1]
                    dst[j][row, col] = src[j][row, col]
                else:
                    dst[j][k] = src[j][k]
                k += 128


class RowChecker:
    """One writer per cell and error bit; repeated launches retain the error."""

    def __init__(self, mutate=False, mutation_row=1):
        self.mutate = mutate
        self.mutation_row = mutation_row

    @cute.jit
    def __call__(
        self,
        table: cute.Tensor,
        cache: cute.Tensor,
        last: cute.Tensor,
        errors: cute.Tensor,
        owned: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(table, cache, last, errors, owned).launch(
            grid=[1, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        table: cute.Tensor,
        cache: cute.Tensor,
        last: cute.Tensor,
        errors: cute.Tensor,
        owned: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.mutate):
            # Fault injection for a round implementation touching an unowned row.
            if tid == 0:
                table[self.mutation_row, 0] = table[self.mutation_row, 0] + Int32(73)
        else:
            # Check ALL non-request rows, including state and table padding.
            for row in range(table.shape[0]):
                if owned[row] == 0:
                    col = Int64(tid)
                    while col < table.shape[1]:
                        k = row * table.shape[1] + col
                        value = last[k]
                        if table[row, col] != value:
                            errors[k] = 1
                        table[row, col] = value + 1
                        last[k] = value + 1
                        col += 128
            offset = table.shape[0] * table.shape[1]
            for row in range(cache.shape[0]):
                if owned[row] == 0:
                    col = Int64(tid)
                    while col < cache.shape[1]:
                        k = offset + row * cache.shape[1] + col
                        value = last[k]
                        if cache[row, col] != value:
                            errors[k] = 1
                        cache[row, col] = value + 1
                        last[k] = value + 1
                        col += 128


def compile_copy(dst, src):
    return cute.compile(
        DelayedCopy(),
        tuple(from_dlpack(x) for x in dst),
        tuple(from_dlpack(x) for x in src),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compile_checker(table, cache, last, errors, owned, *, mutate=False, mutation_row=1):
    return cute.compile(
        RowChecker(mutate, mutation_row),
        from_dlpack(table),
        from_dlpack(cache),
        from_dlpack(last),
        from_dlpack(errors),
        from_dlpack(owned),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
