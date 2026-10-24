# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline copy lowering regression: correct partition sizes alone miss V aliasing."""

import re

import pytest

# The compiler is optional in the CPU-only installation.
pytest.importorskip("cutlass")
import cutlass
import cutlass.cute as cute
from cutlass.runtime import make_fake_stream
from cutlass.utils import SmemAllocator

from ctikernels.attention._draft.cute._kernel import Attention


class OperandCopy:
    def __init__(self, attention, operand):
        self.attention, self.operand = attention, operand

    @cute.jit
    def __call__(self, out: cute.Tensor, stage: cutlass.Int32, stream):
        a = self.attention
        mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(a.dtype, cutlass.Float32, (16, 8, 16)),
            (a.rows // 16, 1, 1),
            permutation_mnk=(a.rows, 16, 16),
        )
        self.kernel(out, stage, mma).launch(grid=(1, 1, 1), block=(a.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, out: cute.Tensor, stage: cutlass.Int32, mma: cute.TiledMma):
        a = self.attention
        tid, _, _ = cute.arch.thread_idx()
        group = tid // a.mma_threads
        the = mma.get_slice(tid % a.mma_threads)
        arena = SmemAllocator().allocate_tensor(
            a.dtype, cute.make_layout(a.smem_elements), byte_alignment=128
        )
        load = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=self.operand == "V", num_matrices=4),
            a.dtype,
        )
        if cutlass.const_expr(self.operand == "V"):
            sv = cute.make_tensor(arena.iterator, a.v_layout(transpose=True))
            source = sv[None, None, stage, group]
            fragment = the.make_fragment_B(the.partition_B(source))
            copy = cute.make_tiled_copy_B(load, mma).get_slice(tid % a.mma_threads)
            assert cute.size(fragment.shape[1]) == a.dim // 8
            assert cute.size(fragment.shape[2]) == a.nk // 16
        else:
            rows = cutlass.const_expr(a.rows if self.operand == "Q" else a.nk)
            source = cute.make_tensor(
                arena.iterator,
                cute.make_composed_layout(
                    cute.make_swizzle(3, 3, a.swizzle_shift),
                    0,
                    cute.make_layout((rows, a.dim), stride=(a.dim, 1)),
                ),
            )
            if cutlass.const_expr(self.operand == "Q"):
                fragment = the.make_fragment_A(the.partition_A(source))
                copy = cute.make_tiled_copy_A(load, mma).get_slice(tid % a.mma_threads)
            else:
                fragment = the.make_fragment_B(the.partition_B(source))
                copy = cute.make_tiled_copy_B(load, mma).get_slice(tid % a.mma_threads)
            assert cute.size(fragment.shape[2]) == a.dim // 16
        assert cute.size(the.partition_shape_C((a.rows, a.dim))) == a.dim // 2
        cute.copy(copy, copy.partition_S(source), copy.retile(fragment))
        # Make every register observable so repeated source addresses get CSE'd,
        # just as in the real PV product. Nothing is launched or read on a GPU.
        for block in cutlass.range_constexpr(cute.size(fragment.shape) // 8):
            for value in cutlass.range_constexpr(8):
                i = block * 8 + value
                out[tid * cute.size(fragment.shape) + i] = fragment[i]


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("rows", [16, 32, 48, 64])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("operand", ["Q", "K", "V"])
def test_operand_copy_covers_all_blocks(arch, dim, rows, groups, operand, tmp_path):
    attention = Attention(
        16,
        4,
        rows,
        4,
        True,
        False,
        1,
        cutlass.BFloat16,
        key_tile=32,
        key_groups=groups,
        stages=2,
        head_dim=dim,
    )
    compiled = cute.compile(
        OperandCopy(attention, operand),
        cute.runtime.make_fake_tensor(cutlass.BFloat16, (32768,), (1,), assumed_align=16),
        cutlass.Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch {arch} --keep-ptx --dump-dir {tmp_path}",
    )
    # Each x4 load supplies eight BF16 values per thread. With the old hd256
    # V swizzle only one load survived, broadcasting columns modulo eight.
    expected = dim // 16 * (1 if operand == "Q" else attention.nk // 16)
    instruction = r"ldmatrix\.sync\.aligned\.m8n8\.x4\."
    if operand == "V":
        instruction += r"trans\."
    assert len(re.findall(instruction + r"shared\.b16", compiled.__ptx__)) == expected
