# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Register-resident fused residual RMSNorm; original CuTe implementation.

Design references: FlashInfer 0.6.11/0.7 norm/kernels/fused_add_rmsnorm.py
(FP32 sum, one output rounding), vLLM 0.31 ir/ops/layernorm.py (market semantics).
No vendor implementation is embedded. Only documented CuTe and public PTX.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.memory import SmemAllocator


@cute.jit
def inverse_root(value):
    # Avoid rsqrt.approx and its overshoot at output rounding boundaries (see
    # attention/_prologue_cute.py). IEEE sqrt + division still needs the FP64
    # qualification gate: neither it nor an FP32 reduction guarantees exact output.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value()],
            "{ .reg .f32 root; sqrt.rn.f32 root, $1; div.rn.f32 $0, 0f3f800000, root; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def vector_at(base, offset, count: cutlass.Constexpr):
    ptr = cute.make_ptr(
        base.value_type,
        base.toint() + Int64(offset) * (base.value_type.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return cute.make_tensor(ptr, cute.make_layout(count))


@cute.jit
def load_residual(base, offset, destination):
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), base.value_type, num_bits_per_copy=128)
    if cutlass.const_expr(base.value_type == Float32):
        # An eight-element FP32 chunk is two aligned 128-bit vectors.
        for half in cutlass.range_constexpr(2):
            vector = cute.make_rmem_tensor(4, Float32)
            cute.copy(atom, vector_at(base, offset + half * 4, 4), vector)
            for j in cutlass.range_constexpr(4):
                destination[half * 4 + j] = vector[j]
    else:
        cute.copy(atom, vector_at(base, offset, 8), destination)


@cute.jit
def store_residual(base, offset, source):
    atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), base.value_type, num_bits_per_copy=128)
    if cutlass.const_expr(base.value_type == Float32):
        for half in cutlass.range_constexpr(2):
            vector = cute.make_rmem_tensor(4, Float32)
            for j in cutlass.range_constexpr(4):
                vector[j] = source[half * 4 + j]
            cute.copy(atom, vector, vector_at(base, offset + half * 4, 4))
    else:
        cute.copy(atom, source, vector_at(base, offset, 8))


@cute.jit
def push_cluster(address, rank, value):
    # Public PTX ISA mapa + st.shared::cluster; push each partial to all peers
    # so no remote access can outlive the final cluster barrier. Original design.
    llvm.inline_asm(
        None,
        [Int32(address).ir_value(), Int32(rank).ir_value(), value.ir_value()],
        "{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; st.shared::cluster.f32 [a], $2; }",
        "r,r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class Kernel:
    def __init__(self, hidden, zero_centered, config, store_residual=True, has_sum=False):
        self.hidden, self.zero_centered = hidden, zero_centered
        self.store_residual, self.has_sum = store_residual, has_sum
        self.threads, self.rows, self.pdl = config.threads, config.rows, config.pdl
        if not config.admits(hidden):
            raise ValueError("norm configuration exceeds 16 chunks per thread")
        self.split = config.split
        self.per = (hidden // 8 + self.split - 1) // self.split
        self.trigger = config.trigger
        self.static_weight = config.static_weight
        self.row_threads = config.threads // config.rows
        self.warps = self.row_threads // 32
        self.chunks = (hidden + self.row_threads * 8 - 1) // (self.row_threads * 8)
        if self.split > 1:
            self.chunks = (self.per + self.row_threads - 1) // self.row_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        weight: cute.Tensor,
        sum_out: cute.Tensor,
        sum_stride: Int64,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        kernel = self.kernel(
            x.iterator,
            residual.iterator,
            weight.iterator,
            sum_out.iterator,
            sum_stride,
            Int64(x.shape[0]),
            eps,
        )
        if cutlass.const_expr(self.split > 1):
            kernel.launch(
                grid=[x.shape[0] * self.split, 1, 1],
                cluster=[self.split, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )
        else:
            kernel.launch(
                grid=[(x.shape[0] + self.rows - 1) // self.rows, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        residual: cute.Pointer,
        weight: cute.Pointer,
        sum_out: cute.Pointer,
        sum_stride: Int64,
        m: Int64,
        eps: Float32,
    ):
        if cutlass.const_expr(self.split > 1):
            cute.arch.cluster_arrive_relaxed()
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        group, thread = tid // self.row_threads, tid % self.row_threads
        row = Int64(block) * self.rows + group
        if cutlass.const_expr(self.split > 1):
            row = Int64(block // self.split)
            rank = block % self.split
        lane, warp = thread % 32, thread // 32
        allocator = SmemAllocator()
        partials = allocator.allocate_tensor(
            Float32,
            cute.make_layout(self.rows * self.warps),
            byte_alignment=16,
        )
        if cutlass.const_expr(self.split > 1):
            slots = allocator.allocate_tensor(
                Float32, cute.make_layout(self.split), byte_alignment=16
            )
        load = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), x.value_type, num_bits_per_copy=128)
        store = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), x.value_type, num_bits_per_copy=128)
        load_w = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            weight.value_type,
            num_bits_per_copy=128,
        )
        weights = cute.make_rmem_tensor((8, self.chunks), weight.value_type)
        values = cute.make_rmem_tensor((8, self.chunks), x.value_type)
        residuals = cute.make_rmem_tensor((8, self.chunks), residual.value_type)
        # Only the public static-weight promise allows loads before the PDL wait.
        # Otherwise a preceding PDL producer may still be writing this buffer.
        if cutlass.const_expr(self.pdl and not self.static_weight):
            cute.arch.griddepcontrol_wait()
        for chunk in cutlass.range_constexpr(self.chunks):
            col = (chunk * self.row_threads + thread) * 8
            limit = self.hidden
            if cutlass.const_expr(self.split > 1):
                col = (rank * self.per + chunk * self.row_threads + thread) * 8
                limit = cutlass.min((rank + 1) * self.per * 8, self.hidden)
            if row < m and col < limit:
                cute.copy(load_w, vector_at(weight, col, 8), weights[None, chunk])
        if cutlass.const_expr(self.pdl and self.static_weight):
            cute.arch.griddepcontrol_wait()
        # Stage the entire row before ANY global store. Interleaving residual
        # stores with subsequent loads serializes cold DRAM trips under aliasing.
        for chunk in cutlass.range_constexpr(self.chunks):
            col = (chunk * self.row_threads + thread) * 8
            limit = self.hidden
            if cutlass.const_expr(self.split > 1):
                col = (rank * self.per + chunk * self.row_threads + thread) * 8
                limit = cutlass.min((rank + 1) * self.per * 8, self.hidden)
            if row < m and col < limit:
                cute.copy(load, vector_at(x, row * self.hidden + col, 8), values[None, chunk])
                load_residual(residual, row * self.hidden + col, residuals[None, chunk])
        if cutlass.const_expr(self.pdl and self.trigger == "early"):
            # Staging loads have been issued after our wait. Launching a dependent
            # only schedules it: its griddepcontrol.wait still waits for this grid
            # to complete and flush memory before consuming any of our outputs.
            cute.arch.griddepcontrol_launch_dependents()
        sums = cute.make_rmem_tensor((8, self.chunks), Float32)
        sums.fill(0.0)
        total = Float32(0)
        for chunk in cutlass.range_constexpr(self.chunks):
            col = (chunk * self.row_threads + thread) * 8
            limit = self.hidden
            if cutlass.const_expr(self.split > 1):
                col = (rank * self.per + chunk * self.row_threads + thread) * 8
                limit = cutlass.min((rank + 1) * self.per * 8, self.hidden)
            if row < m and col < limit:
                vr = cute.make_rmem_tensor(8, residual.value_type)
                for j in cutlass.range_constexpr(8):
                    value = Float32(values[j, chunk]) + Float32(residuals[j, chunk])
                    sums[j, chunk] = value
                    vr[j] = value.to(residual.value_type)
                    total = total + value * value
                if cutlass.const_expr(self.store_residual):
                    store_residual(residual, row * self.hidden + col, vr)
                if cutlass.const_expr(self.has_sum):
                    store_residual(sum_out, row * sum_stride + col, vr)
        for stage in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=16 >> stage)
        if lane == 0:
            partials[group * self.warps + warp] = total
        # Every thread participates, including inactive groups in an odd final CTA.
        cute.arch.sync_threads()
        total = Float32(0)
        # Zero-fill lanes outside the 1..32 warp partials. The same second-stage
        # shuffle works for one-warp rows and for a full 1024-thread row.
        if lane < self.warps:
            total = partials[group * self.warps + lane]
        for stage in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=16 >> stage)
        if cutlass.const_expr(self.split > 1):
            # All CTAs must have started before any DSMEM access.
            cute.arch.cluster_wait()
            if tid == 0:
                address = slots.iterator.toint() + rank * 4
                for peer in cutlass.range_constexpr(self.split):
                    push_cluster(address, peer, total)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            # Only local shared loads remain. Peers may now exit independently.
            # Rank order gives every CTA exactly the same reduction bits.
            total = Float32(0)
            for peer in cutlass.range_constexpr(self.split):
                total = total + slots[peer]
        inv = inverse_root(total * (1.0 / self.hidden) + eps)
        for chunk in cutlass.range_constexpr(self.chunks):
            col = (chunk * self.row_threads + thread) * 8
            limit = self.hidden
            if cutlass.const_expr(self.split > 1):
                col = (rank * self.per + chunk * self.row_threads + thread) * 8
                limit = cutlass.min((rank + 1) * self.per * 8, self.hidden)
            if row < m and col < limit:
                result = cute.make_rmem_tensor(8, x.value_type)
                for j in cutlass.range_constexpr(8):
                    scale = Float32(weights[j, chunk])
                    if cutlass.const_expr(self.zero_centered):
                        scale = scale + 1.0
                    result[j] = ((sums[j, chunk] * inv) * scale).to(x.value_type)
                cute.copy(store, result, vector_at(x, row * self.hidden + col, 8))
        if cutlass.const_expr(self.pdl and self.trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()
