# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original single-launch split-row logsumexp, using public CuTe and PTX APIs."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.memory import SmemAllocator


@cute.jit
def multiply(a, b):
    # mul.rn is never contracted into an FMA with the later max subtraction.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def divide(a, b):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def maximum(a, b):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "max.NaN.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def arrive(ptr):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint().ir_value()],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def shifted_exp(value, peak):
    result = Float32(0)
    if peak == Float32(float("inf")):
        if value == peak:
            result = Float32(1)
    elif peak != Float32(float("-inf")):
        result = cute.math.exp(value - peak)
    return result


@cute.jit
def block_reduce(value, scratch, is_max: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    for stage in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, offset=16 >> stage)
        if cutlass.const_expr(is_max):
            value = maximum(value, other)
        else:
            value = value + other
    if lane == 0:
        scratch[warp] = value
    cute.arch.sync_threads()
    value = Float32(0)
    if cutlass.const_expr(is_max):
        value = Float32(float("-inf"))
    if lane < 4:
        value = scratch[lane]
    for stage in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, offset=16 >> stage)
        if cutlass.const_expr(is_max):
            value = maximum(value, other)
        else:
            value = value + other
    # All warps must finish reading before the next reduction reuses scratch.
    cute.arch.sync_threads()
    return value


class Kernel:
    def __init__(self, vocab, splits, element_bits, per_row_temperature=False):
        self.per_row_temperature = per_row_temperature
        self.vocab, self.splits = vocab, splits
        self.vector = 128 // element_bits
        quantum = 128 * self.vector
        # Align the physical row base down, then mask its prefix and suffix.
        # This retains vector loads even when V (and hence row stride) is odd.
        extent = (vocab + self.vector - 1 + splits - 1) // splits
        self.tile = (extent + quantum - 1) // quantum * quantum
        self.chunks = self.tile // quantum

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        targets: cute.Tensor,
        out: cute.Tensor,
        partials: cute.Tensor,
        counters: cute.Tensor,
        temperatures: cute.Tensor,
        inverse_temperature: Float32,
        ignore_index: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            logits.iterator,
            Int64(logits.stride[0]),
            targets.iterator,
            Int64(targets.stride[0]),
            out.iterator,
            partials.iterator,
            counters.iterator,
            temperatures.iterator,
            Int64(temperatures.stride[0]),
            inverse_temperature,
            ignore_index,
        ).launch(
            grid=[logits.shape[0] * self.splits, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        stride: Int64,
        targets: cute.Pointer,
        target_stride: Int64,
        out: cute.Pointer,
        partials: cute.Pointer,
        counters: cute.Pointer,
        temperatures: cute.Pointer,
        temperature_stride: Int64,
        inverse_temperature: Float32,
        ignore_index: Int64,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row, split = block // self.splits, block % self.splits
        if cutlass.const_expr(self.per_row_temperature):
            temperature = Float32(temperatures[Int64(row) * temperature_stride])
        allocator = SmemAllocator()
        scratch = allocator.allocate_tensor(
            Float32,
            cute.make_layout(4),
            byte_alignment=16,
        )
        election = allocator.allocate_tensor(Int32, cute.make_layout(1))
        values = cute.make_rmem_tensor((self.vector, self.chunks), Float32)
        load = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), logits.value_type, num_bits_per_copy=128)
        base = Int64(row) * stride
        # Include the actual pointer alignment, allowing sliced logits too.
        misalignment = (logits.toint() // (logits.value_type.width // 8) + base) % self.vector
        peak = Float32(float("-inf"))
        for chunk in cutlass.range_constexpr(self.chunks):
            col = Int64(split) * self.tile + (chunk * 128 + tid) * self.vector - misalignment
            pack = cute.make_rmem_tensor(self.vector, logits.value_type)
            if col >= 0 and col + self.vector <= self.vocab:
                ptr = cute.make_ptr(
                    logits.value_type,
                    logits.toint() + (base + col) * (logits.value_type.width // 8),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                cute.copy(load, cute.make_tensor(ptr, cute.make_layout(self.vector)), pack)
            else:
                for j in cutlass.range_constexpr(self.vector):
                    pack[j] = Float32(float("-inf")).to(logits.value_type)
                    if col + j >= 0 and col + j < self.vocab:
                        pack[j] = logits[base + col + j]
            for j in cutlass.range_constexpr(self.vector):
                if cutlass.const_expr(self.per_row_temperature):
                    value = divide(Float32(pack[j]), temperature)
                else:
                    # PyTorch CUDA scales by a host FP32 reciprocal (BinaryDivTrueKernel).
                    value = multiply(Float32(pack[j]), inverse_temperature)
                # Invalid lanes must remain neutral even if T rounds to 0/inf.
                if col + j < 0 or col + j >= self.vocab:
                    value = Float32(float("-inf"))
                values[j, chunk] = value
                peak = maximum(peak, value)
        peak = block_reduce(peak, scratch, True)
        total = Float32(0)
        for chunk in cutlass.range_constexpr(self.chunks):
            for j in cutlass.range_constexpr(self.vector):
                total = total + shifted_exp(values[j, chunk], peak)
        total = block_reduce(total, scratch, False)
        last = Int32(1)
        if cutlass.const_expr(self.splits > 1):
            if tid == 0:
                # The publishing thread also performs the release RMW. The
                # acquire/release chain makes every earlier partial visible to
                # the last CTA; its barrier passes that visibility to all lanes.
                partials[(Int64(row) * self.splits + split) * 2] = peak
                partials[(Int64(row) * self.splits + split) * 2 + 1] = total
                election[0] = arrive(counters + row)
            cute.arch.sync_threads()
            last = Int32(election[0] == self.splits - 1)
        if last == 1:
            if cutlass.const_expr(self.splits > 1):
                peak = Float32(float("-inf"))
                for index in range(tid, self.splits, 128):
                    peak = maximum(peak, partials[(Int64(row) * self.splits + index) * 2])
                peak = block_reduce(peak, scratch, True)
                total = Float32(0)
                for index in range(tid, self.splits, 128):
                    offset = (Int64(row) * self.splits + index) * 2
                    subtotal = partials[offset + 1]
                    if subtotal != 0:
                        total = total + subtotal * shifted_exp(partials[offset], peak)
                total = block_reduce(total, scratch, False)
            if tid == 0:
                target = targets[Int64(row) * target_stride]
                if target == ignore_index:
                    out[row] = Float32(0)
                else:
                    # Invalid ids have undefined results but never form an OOB load.
                    target = max(Int64(0), min(target, Int64(self.vocab - 1)))
                    if cutlass.const_expr(self.per_row_temperature):
                        selected = divide(Float32(logits[base + target]), temperature)
                    else:
                        selected = multiply(Float32(logits[base + target]), inverse_temperature)
                    out[row] = (selected - peak) - cute.math.log(total)
                if cutlass.const_expr(self.splits > 1):
                    counters[row] = Int32(0)
