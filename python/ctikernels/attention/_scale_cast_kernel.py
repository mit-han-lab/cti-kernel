# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Original CuTe elementwise epilogue; no third-party implementation copied.

Design references: attention/norm's register-vector copy pattern and NVIDIA's
public PTX ISA griddepcontrol and cvt instructions:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
FlashInfer v0.7.0.post1 decode.py / prefill.py supply the unscaled FP16 output.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int64


@cute.jit
def _vector(pointer, offset):
    return cute.make_tensor(
        cute.make_ptr(
            pointer.value_type,
            pointer.toint() + Int64(offset) * 2,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(8),
    )


class ScaleCast:
    def __init__(self, supports_pdl):
        self.supports_pdl = supports_pdl

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        pdl: Boolean,
        stream: cuda.CUstream,
    ):
        # Runtime host dispatch keeps one compilation per dtype pair/architecture,
        # including both launch modes and arbitrary host-known element counts.
        if pdl:
            self.kernel(source, out, scale).launch(
                grid=((Int64(source.shape[0]) + 1023) // 1024, 1, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            self.kernel(source, out, scale).launch(
                grid=((Int64(source.shape[0]) + 1023) // 1024, 1, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=False,
            )

    @cute.kernel
    def kernel(self, source: cute.Tensor, out: cute.Tensor, scale: Float32):
        if cutlass.const_expr(self.supports_pdl):
            # Also safe for a serialized launch: the predecessor already finished.
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        offset = (Int64(block) * 128 + thread) * 8
        if offset < Int64(source.shape[0]):
            values = cute.make_rmem_tensor(8, source.element_type)
            converted = cute.make_rmem_tensor(8, out.element_type)
            load = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), source.element_type, num_bits_per_copy=128
            )
            cute.copy(load, _vector(source.iterator, offset), values)
            converted.store((values.load().to(Float32) * scale).to(out.element_type))
            # Preserve the existing contiguous-output contract even for a view
            # beginning at an odd element. Normal allocations use 16-byte stores.
            if out.iterator.toint() % 16 == 0:
                store = cute.make_copy_atom(
                    cute.nvgpu.CopyR2GOp(), out.element_type, num_bits_per_copy=128
                )
                cute.copy(store, converted, _vector(out.iterator, offset))
            else:
                for i in cutlass.range_constexpr(8):
                    out[offset + i] = converted[i]
