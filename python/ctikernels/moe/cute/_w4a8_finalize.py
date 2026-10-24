# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Slot-ordered FP32 combine of BF16 routed rows (each rounded once), with one final BF16 RNE."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel_type():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int64

    class FinalizeKernel:
        def __init__(self, pdl, aligned):
            self.pdl = pdl
            self.aligned = aligned

        @cute.jit
        def __call__(
            self, y: cute.Tensor, slot_row: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream
        ):
            self.finalize(y, slot_row, out).launch(
                grid=[cute.size(out, mode=[0]), cute.size(out, mode=[1]) // 256, 1],
                block=[32, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def finalize(self, y: cute.Tensor, slot_row: cute.Tensor, out: cute.Tensor):
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            tid, _, _ = cute.arch.thread_idx()
            token, chunk, _ = cute.arch.block_idx()
            col = chunk * 256 + tid * 8
            width = cute.size(out, mode=[1])
            values = cute.make_rmem_tensor((8,), BFloat16)
            total = cute.make_rmem_tensor((8,), Float32)
            result = cute.make_rmem_tensor((8,), BFloat16)
            load = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), BFloat16, num_bits_per_copy=128)
            store = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), BFloat16, num_bits_per_copy=128)
            total.fill(0)
            for slot in cutlass.range_constexpr(6):
                row = slot_row[token, slot]
                if row >= 0:
                    src = cute.make_tensor(
                        cute.make_ptr(
                            BFloat16,
                            y.iterator.toint() + Int64(row * width + col) * 2,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(8),
                    )
                    cute.copy(load, src, values)
                    for j in cutlass.range_constexpr(8):
                        total[j] = total[j] + values[j].to(Float32)
            for j in cutlass.range_constexpr(8):
                result[j] = BFloat16(total[j])
            if cutlass.const_expr(self.aligned):
                dst = cute.make_tensor(
                    cute.make_ptr(
                        BFloat16,
                        out.iterator.toint() + Int64(token * width + col) * 2,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(8),
                )
                cute.copy(store, result, dst)
            else:
                for j in cutlass.range_constexpr(8):
                    out[token, col + j] = result[j]
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_launch_dependents()

    return FinalizeKernel


def arguments(ws, out):
    return ws.y, ws.slot_row, out


def compile_finalize(runtime, ws, out, *, pdl=True, aligned=None):
    # Warm both output alignments outside capture; caller-owned views can change.
    aligned = out.data_ptr() % 16 == 0 if aligned is None else aligned
    kernel = _kernel_type()(pdl, aligned)
    return runtime.compile(("w4a8_finalize", pdl, aligned), kernel, arguments(ws, out))
