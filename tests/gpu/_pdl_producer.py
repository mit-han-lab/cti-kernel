# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original one-CTA delayed writer for PDL dependency tests; public PTX only.

Import this helper inside GPU tests. A finite delay avoids any dependence on
consumer progress: CUDA permits serialization even when PDL is enabled.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.runtime import make_fake_stream


@cute.jit
def _globaltimer():
    return Int64(
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


class _Producer:
    def __init__(self, count):
        self.count = count

    @cute.jit
    def __call__(
        self,
        destination: cute.Tensor,
        source: cute.Tensor,
        stamps: cute.Tensor,
        delay_ns: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(destination, source, stamps, delay_ns).launch(
            grid=[1, 1, 1],
            block=[32, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, destination: cute.Tensor, source: cute.Tensor, stamps: cute.Tensor, delay_ns: Int64
    ):
        # Release the dependent grid FIRST, before even reading the timer.
        cute.arch.griddepcontrol_launch_dependents()
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            start = _globaltimer()
            now = _globaltimer()
            while now - start < delay_ns:
                now = _globaltimer()
            stamps[0] = start
            stamps[1] = now
        cute.arch.sync_threads()
        for chunk in cutlass.range_constexpr((self.count + 31) // 32):
            index = chunk * 32 + tid
            if index < self.count:
                destination[index] = source[index]


def compile_producer(count, dtype, capability):
    scalar = {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16}[dtype]
    fake = cute.runtime.make_fake_tensor
    operand = fake(scalar, (count,), (1,), assumed_align=16)
    stamps = fake(Int64, (2,), (1,), assumed_align=16)
    return cute.compile(
        _Producer(count),
        operand,
        operand,
        stamps,
        Int64(200_000),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )
