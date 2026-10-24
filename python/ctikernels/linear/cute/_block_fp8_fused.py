# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""One-kernel matched-column block-FP8 FC1, FMA and MMA engines for M <= 16.

Reference design: a matched-column/last-CTA protocol and package CuTe quantizer.
Public PTX only: https://docs.nvidia.com/cuda/parallel-thread-execution/

FMA FP32 order: 16 sequential FMAs, 8-lane butterfly, scale each 128-term
dot, accumulate K strides in each subgroup, then sum subgroups in ascending
order. This differs from tensor-core GEMM. BF16 rounding and everything after
it use the SAME device arithmetic as the package activation quantizer.

FMA per-token running sums live in shared memory, owned by one subgroup leader;
weights are loaded once per K step for all tokens. Register use does not grow
with M. No spinning, inter-CTA scheduling assumptions, or host synchronization.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float16, Float32, Int32, Int64, Uint16, Uint32
from cutlass.runtime import make_fake_stream
from cutlass.utils import SmemAllocator

from ._block_fp8_quant_math import fp32_op, pack_pair, quant_scale, quant_value, silu_product


@cute.jit
def unpack_pair(bits):
    return cute.arch.inline_ptx(
        "{ .reg .b32 pair; .reg .b16 lo, hi;\n"
        "cvt.rn.f16x2.e4m3x2 pair, {$r0};\n"
        "mov.b32 {lo, hi}, pair;\n"
        "cvt.f32.f16 {$w0}, lo; cvt.f32.f16 {$w1}, hi; }",
        write_only_types=[Float32, Float32],
        read_only_args=[bits],
    )


class Kernel:
    def __init__(self, i, k, m_max, config, limit, ue8m0, debug):
        self.i, self.k, self.m_max = i, k, m_max
        self.r, self.warps = config.r, config.warps
        self.threads = config.warps * 32
        self.subgroups = config.warps * 4
        self.limit, self.ue8m0, self.debug = limit, ue8m0, debug
        self.smem = config.shared_bytes(m_max)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        sw: cute.Tensor,
        q: cute.Tensor,
        scale: cute.Tensor,
        hbuf: cute.Tensor,
        counters: cute.Tensor,
        qx,
        sx,
        gate_up,
        stream: cuda.CUstream,
    ):
        self.run(x, weight, sw, q, scale, hbuf, counters, qx, sx, gate_up).launch(
            grid=(self.i // self.r, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def load_weights(self, weight, words, col, kpos):
        for r in cutlass.range_constexpr(2 * self.r):
            for v in cutlass.range_constexpr(4):
                words[v, r] = Uint32(0)
            if kpos < self.k:
                n = col + r % self.r + (r // self.r) * self.i
                loaded = cute.arch.inline_ptx(
                    "ld.global.cg.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                    write_only_types=[Uint32, Uint32, Uint32, Uint32],
                    read_only_args=[Int64((weight.iterator + n * self.k + kpos).toint())],
                )
                for v in cutlass.range_constexpr(4):
                    words[v, r] = loaded[v]

    @cute.kernel
    def run(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        sw: cute.Tensor,
        q: cute.Tensor,
        scale: cute.Tensor,
        hbuf: cute.Tensor,
        counters: cute.Tensor,
        qx,
        sx,
        gate_up,
    ):
        bid32, _, _ = cute.arch.block_idx()
        tid32, _, _ = cute.arch.thread_idx()
        bid, tid = Int64(bid32), Int64(tid32)
        col = bid * self.r
        subgroup = tid // 8
        lane = tid % 8
        group = col // 128
        alloc = SmemAllocator()
        partial = alloc.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.m_max, 2 * self.r, self.subgroups),
                # Adjacent epilogue threads reduce adjacent columns. Keeping
                # columns contiguous avoids a subgroup-strided bank conflict.
                stride=(2 * self.r, 1, self.m_max * 2 * self.r),
            ),
        )
        last = alloc.allocate_tensor(Int32, cute.make_layout(1))
        # Frozen weights may be read before the predecessor finishes producing x.
        words = cute.make_rmem_tensor((4, 2 * self.r), Uint32)
        self.load_weights(weight, words, col, tid * 16)
        for offset in cutlass.range_constexpr(
            cute.ceil_div(self.m_max * 2 * self.r * self.subgroups, self.threads)
        ):
            idx = tid + offset * self.threads
            if idx < self.m_max * 2 * self.r * self.subgroups:
                partial[
                    idx // (2 * self.r * self.subgroups),
                    idx // self.subgroups % (2 * self.r),
                    idx % self.subgroups,
                ] = Float32(0)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

        for step in cutlass.range(cute.ceil_div(self.k, self.threads * 16), unroll=1):
            kpos = Int64(step) * self.threads * 16 + tid * 16
            kb = kpos // 128
            if step > 0:
                self.load_weights(weight, words, col, kpos)
            if step == cute.ceil_div(self.k, self.threads * 16) - 1:
                # All mainloop weight loads issued. Dependents must still wait
                # for this grid's completion before consuming q/scale or scratch.
                cute.arch.griddepcontrol_launch_dependents()
            for m in cutlass.range(x.shape[0], unroll=1):
                row = Int64(m)
                values = cute.make_rmem_tensor(16, Float32)
                values.fill(0)
                amax = Float32(0)
                for v in cutlass.range_constexpr(16):
                    if kpos < self.k:
                        values[v] = Float32(x[row, kpos + v])
                    amax = fp32_op(
                        "max.f32" if self.ue8m0 else "max.NaN.f32",
                        amax,
                        abs(values[v]),
                    )
                for shift in cutlass.range_constexpr(3):
                    amax = fp32_op(
                        "max.f32" if self.ue8m0 else "max.NaN.f32",
                        amax,
                        cute.arch.shuffle_sync_bfly(amax, offset=4 >> shift),
                    )
                stored, multiplier = quant_scale(amax, self.ue8m0)
                for pair in cutlass.range_constexpr(8):
                    bits = pack_pair(
                        quant_value(values[2 * pair], multiplier, self.ue8m0),
                        quant_value(values[2 * pair + 1], multiplier, self.ue8m0),
                    )
                    lo, hi = unpack_pair(bits)
                    values[2 * pair], values[2 * pair + 1] = lo, hi
                    if cutlass.const_expr(self.debug):
                        if bid == 0 and kpos < self.k:
                            target = cute.make_tensor(
                                cute.recast_ptr(qx.iterator + row * self.k + kpos, dtype=Uint16),
                                cute.make_layout(8),
                            )
                            target[pair] = bits
                if cutlass.const_expr(self.debug):
                    if bid == 0 and lane == 0 and kpos < self.k:
                        sx[row, kb] = stored
                for r in cutlass.range_constexpr(2 * self.r):
                    dot = Float32(0)
                    for pair in cutlass.range_constexpr(8):
                        bits = (words[pair // 2, r] >> ((pair % 2) * 16)).to(Uint16)
                        lo, hi = unpack_pair(bits)
                        dot = cute.arch.inline_ptx(
                            "{ .reg .f32 t; fma.rn.f32 t, {$r0}, {$r1}, {$r2};\n"
                            "fma.rn.f32 {$w0}, {$r3}, {$r4}, t; }",
                            write_only_types=[Float32],
                            read_only_args=[values[2 * pair], lo, dot, values[2 * pair + 1], hi],
                        )
                    for shift in cutlass.range_constexpr(3):
                        dot = fp32_op(
                            "add.rn.f32",
                            dot,
                            cute.arch.shuffle_sync_bfly(dot, offset=4 >> shift),
                        )
                    if lane == 0 and kpos < self.k:
                        n = col + r % self.r + (r // self.r) * self.i
                        scaled = fp32_op("mul.rn.f32", dot, stored)
                        scaled = fp32_op("mul.rn.f32", scaled, sw[n // 128, kb])
                        partial[row, r, subgroup] = fp32_op(
                            "add.rn.f32",
                            partial[row, r, subgroup],
                            scaled,
                        )
        cute.arch.sync_threads()
        for index in cutlass.range(tid, x.shape[0] * self.r, self.threads, unroll=1):
            row, r = Int64(index) // self.r, Int64(index) % self.r
            g, u = Float32(0), Float32(0)
            for slot in cutlass.range_constexpr(self.subgroups):
                g = fp32_op("add.rn.f32", g, partial[row, r, slot])
                u = fp32_op("add.rn.f32", u, partial[row, r + self.r, slot])
            g, u = Float32(g.to(BFloat16)), Float32(u.to(BFloat16))
            if cutlass.const_expr(self.debug):
                gate_up[row, col + r] = g.to(BFloat16)
                gate_up[row, self.i + col + r] = u.to(BFloat16)
            hbuf[row, col + r] = silu_product(g, u, self.limit).to(BFloat16)
        self.finish(x, q, scale, hbuf, counters, last, tid, group)

    @cute.jit
    def finish(self, x, q, scale, hbuf, counters, last, tid, group):
        # Every writer publishes, then the CTA hands off to its one atomic issuer.
        cute.arch.inline_ptx("fence.acq_rel.gpu;")
        cute.arch.sync_threads()
        if tid == 0:
            old = cute.arch.inline_ptx(
                "atom.release.gpu.global.add.u32 {$w0}, [{$r0}], 1;",
                write_only_types=[Int32],
                read_only_args=[Int64((counters.iterator + group).toint())],
            )
            last[0] = Int32(old == 128 // self.r - 1)
            if old == 128 // self.r - 1:
                # Atomic read + acquire fence pairs with all release arrivals.
                cute.arch.inline_ptx("fence.acq_rel.gpu;")
        cute.arch.sync_threads()
        if last[0] != 0:
            # 16 lanes/token, eight BF16 elements/lane: exactly the existing
            # SiLU quantizer's max tree, scale formula, conversion and stores.
            for wave in cutlass.range_constexpr(cute.ceil_div(self.m_max, self.threads // 16)):
                row = tid // 16 + wave * (self.threads // 16)
                start = group * 128 + (tid % 16) * 8
                values = cute.make_rmem_tensor(8, Float32)
                values.fill(0)
                if row < x.shape[0]:
                    for v in cutlass.range_constexpr(8):
                        values[v] = cute.arch.inline_ptx(
                            "{ .reg .b16 h; ld.global.cg.b16 h, [{$r0}];\ncvt.f32.bf16 {$w0}, h; }",
                            write_only_types=[Float32],
                            read_only_args=[
                                Int64((hbuf.iterator + row * self.i + start + v).toint())
                            ],
                        )
                amax = Float32(0)
                for v in cutlass.range_constexpr(8):
                    amax = fp32_op(
                        "max.f32" if self.ue8m0 else "max.NaN.f32",
                        amax,
                        abs(values[v]),
                    )
                for shift in cutlass.range_constexpr(4):
                    amax = fp32_op(
                        "max.f32" if self.ue8m0 else "max.NaN.f32",
                        amax,
                        cute.arch.shuffle_sync_bfly(amax, offset=8 >> shift),
                    )
                stored, multiplier = quant_scale(amax, self.ue8m0)
                if row < x.shape[0]:
                    target = cute.make_tensor(
                        cute.recast_ptr(q.iterator + row * self.i + start, dtype=Uint16),
                        cute.make_layout(4),
                    )
                    for pair in cutlass.range_constexpr(4):
                        target[pair] = pack_pair(
                            quant_value(values[2 * pair], multiplier, self.ue8m0),
                            quant_value(values[2 * pair + 1], multiplier, self.ue8m0),
                        )
                    if tid % 16 == 0:
                        scale[row, group] = stored
            # No next invocation can reuse hbuf/counters until this grid completes.
            cute.arch.sync_threads()
            if tid == 0:
                cute.arch.inline_ptx(
                    "st.release.gpu.global.b32 [{$r0}], {$r1};",
                    read_only_args=[Int64((counters.iterator + group).toint()), Int32(0)],
                )


class MMAKernel(Kernel):
    """16 MMA rows = eight gate rows followed by their eight matching up rows.

    Reference design: public PTX ISA mma.m16n8k32 E4M3 fragment layout.

    Lane = 4*g+t. For MMA s=0..3 and byte b=0..3 in K half h=0..1:
        logical K = 32*s + 16*h + 4*t + b
        physical K = 32*t + 8*s + 4*h + b
    Both operands use this bijection within each 128-K scale block. A registers
    are [gate(8*s), up(8*s), gate(8*s+4), up(8*s+4)] from each lane's contiguous
    32-byte slice. B registers read the same physical bytes for token g.
    C0/C1 hold gate(g) for tokens 2*t/2*t+1; C2/C3 hold the matching up(g).

    Only one weight block is resident per lane (16 uint32 registers). Activation
    staging covers warps_k blocks, independent of H; all row warps reuse it.
    K warps accumulate strided blocks, then reduce in ascending warp-K order.
    """

    def __init__(self, i, k, m_max, config, limit, ue8m0, debug):
        self.i, self.k, self.m_max = i, k, m_max
        self.warps_n, self.warps_k = config.warps_n, config.warps_k
        self.r = 8 * self.warps_n
        self.threads = 32 * self.warps_n * self.warps_k
        self.tiles_m = (m_max + 7) // 8
        self.tokens = self.tiles_m * 8
        self.limit, self.ue8m0, self.debug = limit, ue8m0, debug
        self.smem = config.shared_bytes(m_max)

    @cute.jit
    def load_mma_weights(self, weight, words, col, g, t, kb):
        for up in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                for v in cutlass.range_constexpr(4):
                    words[half * 4 + v, up] = Uint32(0)
                if kb < self.k // 128:
                    offset = (col + g + up * self.i) * self.k + kb * 128 + t * 32 + half * 16
                    loaded = cute.arch.inline_ptx(
                        "ld.global.cg.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                        write_only_types=[Uint32, Uint32, Uint32, Uint32],
                        read_only_args=[Int64((weight.iterator + offset).toint())],
                    )
                    for v in cutlass.range_constexpr(4):
                        words[half * 4 + v, up] = loaded[v]

    @cute.jit
    def quantize_stage(self, x, xwords, scales, qx, sx, bid, tid, step):
        # Sixteen lanes per (token, K block), eight contiguous input values/lane.
        # No duplication between row warps; every real x element is quantized once.
        for wave in cutlass.range_constexpr(
            cute.ceil_div(self.tokens * self.warps_k * 16, self.threads)
        ):
            index = tid + wave * self.threads
            row = index // (16 * self.warps_k)
            slot = index // 16 % self.warps_k
            kb = step * self.warps_k + slot
            start = kb * 128 + (tid % 16) * 8
            values = cute.make_rmem_tensor(8, Float32)
            values.fill(0)
            if row < x.shape[0] and kb < self.k // 128:
                address = Int64((x.iterator + row * x.stride[0] + start).toint())
                if address % 16 == 0:
                    loaded = cute.arch.inline_ptx(
                        "ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                        write_only_types=[Uint32, Uint32, Uint32, Uint32],
                        read_only_args=[address],
                    )
                    for v in cutlass.range_constexpr(4):
                        if cutlass.const_expr(x.element_type == BFloat16):
                            lo, hi = cute.arch.inline_ptx(
                                "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0};\n"
                                "cvt.f32.bf16 {$w0}, lo; cvt.f32.bf16 {$w1}, hi; }",
                                write_only_types=[Float32, Float32],
                                read_only_args=[loaded[v]],
                            )
                        else:
                            lo, hi = cute.arch.inline_ptx(
                                "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0};\n"
                                "cvt.f32.f16 {$w0}, lo; cvt.f32.f16 {$w1}, hi; }",
                                write_only_types=[Float32, Float32],
                                read_only_args=[loaded[v]],
                            )
                        values[2 * v], values[2 * v + 1] = lo, hi
                else:
                    for v in cutlass.range_constexpr(8):
                        values[v] = Float32(x[row, start + v])
            amax = Float32(0)
            for v in cutlass.range_constexpr(8):
                amax = fp32_op("max.f32" if self.ue8m0 else "max.NaN.f32", amax, abs(values[v]))
            for shift in cutlass.range_constexpr(4):
                amax = fp32_op(
                    "max.f32" if self.ue8m0 else "max.NaN.f32",
                    amax,
                    cute.arch.shuffle_sync_bfly(amax, offset=8 >> shift),
                )
            stored, multiplier = quant_scale(amax, self.ue8m0)
            packed = cute.make_rmem_tensor(4, Uint16)
            for pair in cutlass.range_constexpr(4):
                packed[pair] = pack_pair(
                    quant_value(values[2 * pair], multiplier, self.ue8m0),
                    quant_value(values[2 * pair + 1], multiplier, self.ue8m0),
                )
            if row < self.tokens:
                for word in cutlass.range_constexpr(2):
                    # XOR the word address by token%8 for conflict-free B loads.
                    # This shared-memory swizzle does not change physical K.
                    xwords[((tid % 16) * 2 + word) ^ (row % 8), row, slot] = Uint32(
                        packed[2 * word]
                    ) | (Uint32(packed[2 * word + 1]) << 16)
                if tid % 16 == 0:
                    scales[row, slot] = stored
            if cutlass.const_expr(self.debug):
                if bid == 0 and row < x.shape[0] and kb < self.k // 128:
                    target = cute.make_tensor(
                        cute.recast_ptr(qx.iterator + row * self.k + start, dtype=Uint16),
                        cute.make_layout(4),
                    )
                    for pair in cutlass.range_constexpr(4):
                        target[pair] = packed[pair]
                    if tid % 16 == 0:
                        sx[row, kb] = stored

    @cute.kernel
    def run(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        sw: cute.Tensor,
        q: cute.Tensor,
        scale: cute.Tensor,
        hbuf: cute.Tensor,
        counters: cute.Tensor,
        qx,
        sx,
        gate_up,
    ):
        bid32, _, _ = cute.arch.block_idx()
        tid32, _, _ = cute.arch.thread_idx()
        bid, tid = Int64(bid32), Int64(tid32)
        lane, warp = tid % 32, tid // 32
        wn, wk = warp % self.warps_n, warp // self.warps_n
        g, t = lane // 4, lane % 4
        col = bid * self.r + wn * 8
        group = col // 128
        alloc = SmemAllocator()
        xwords = alloc.allocate_tensor(Uint32, cute.make_layout((32, self.tokens, self.warps_k)))
        scales = alloc.allocate_tensor(Float32, cute.make_layout((self.tokens, self.warps_k)))
        partial = alloc.allocate_tensor(
            Float32,
            cute.make_layout((32, 4, self.tiles_m, self.warps_n, self.warps_k)),
        )
        last = alloc.allocate_tensor(Int32, cute.make_layout(1))
        words = cute.make_rmem_tensor((8, 2), Uint32)
        acc = cute.make_rmem_tensor((4, self.tiles_m), Float32)
        acc.fill(0)
        # Frozen weights are legal before the producer releases x. Each warp
        # preloads its first assigned 128-K block, including both gate/up rows.
        self.load_mma_weights(weight, words, col, g, t, wk)
        cute.arch.griddepcontrol_wait()
        for step in cutlass.range(cute.ceil_div(self.k // 128, self.warps_k), unroll=1):
            kb = Int64(step) * self.warps_k + wk
            if step > 0:
                self.load_mma_weights(weight, words, col, g, t, kb)
            if step == cute.ceil_div(self.k // 128, self.warps_k) - 1:
                cute.arch.griddepcontrol_launch_dependents()
            self.quantize_stage(x, xwords, scales, qx, sx, bid, tid, Int64(step))
            cute.arch.sync_threads()
            # This condition is warp-uniform, including inactive short-K warps.
            if kb < self.k // 128:
                for tile in cutlass.range_constexpr(self.tiles_m):
                    frag = cute.make_rmem_tensor(4, Float32)
                    frag.fill(0)
                    for s in cutlass.range_constexpr(4):
                        result = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
                            "{{$w0}, {$w1}, {$w2}, {$w3}},\n"
                            "{{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}},\n"
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[
                                words[2 * s, 0],
                                words[2 * s, 1],
                                words[2 * s + 1, 0],
                                words[2 * s + 1, 1],
                                xwords[(8 * t + 2 * s) ^ g, tile * 8 + g, wk],
                                xwords[(8 * t + 2 * s + 1) ^ g, tile * 8 + g, wk],
                                frag[0],
                                frag[1],
                                frag[2],
                                frag[3],
                            ],
                        )
                        for c in cutlass.range_constexpr(4):
                            frag[c] = result[c]
                    # Fresh 128-term dot, then two rounded muls and one add.
                    # Gate/up can have different scale rows even in one fragment.
                    for c in cutlass.range_constexpr(4):
                        row = tile * 8 + 2 * t + c % 2
                        n = col + g + (c // 2) * self.i
                        value = fp32_op("mul.rn.f32", frag[c], scales[row, wk])
                        value = fp32_op("mul.rn.f32", value, sw[n // 128, kb])
                        acc[c, tile] = fp32_op("add.rn.f32", acc[c, tile], value)
            # All consumers finish before the next stage overwrites xwords/scales.
            cute.arch.sync_threads()
        for tile in cutlass.range_constexpr(self.tiles_m):
            for c in cutlass.range_constexpr(4):
                partial[lane, c, tile, wn, wk] = acc[c, tile]
        cute.arch.sync_threads()
        if wk == 0:
            for tile in cutlass.range_constexpr(self.tiles_m):
                for c in cutlass.range_constexpr(4):
                    value = Float32(0)
                    for slot in cutlass.range_constexpr(self.warps_k):
                        value = fp32_op("add.rn.f32", value, partial[lane, c, tile, wn, slot])
                    acc[c, tile] = value
                for token in cutlass.range_constexpr(2):
                    row = tile * 8 + 2 * t + token
                    if row < x.shape[0]:
                        gate = Float32(acc[token, tile].to(BFloat16))
                        up = Float32(acc[token + 2, tile].to(BFloat16))
                        if cutlass.const_expr(self.debug):
                            gate_up[row, col + g] = gate.to(BFloat16)
                            gate_up[row, self.i + col + g] = up.to(BFloat16)
                        hbuf[row, col + g] = silu_product(gate, up, self.limit).to(BFloat16)
        self.finish(x, q, scale, hbuf, counters, last, tid, group)


def compile_kernel(i, k, dtype, bucket, config, capability, limit, ue8m0, layout, debug):
    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int64()
    x = fake(
        BFloat16 if dtype == torch.bfloat16 else Float16,
        (m, k),
        (cute.sym_int64(), 1),
        assumed_align=2,
    )
    weight = fake(Float8E4M3FN, (2 * i, k), (k, 1), assumed_align=16)
    sw = fake(Float32, (2 * i // 128, k // 128), (k // 128, 1), assumed_align=4)
    q = fake(Float8E4M3FN, (m, i), (i, 1), assumed_align=16)
    scale = fake(
        Float32,
        (m, i // 128),
        (i // 128, 1) if layout == "k" else (1, cute.sym_int64(divisibility=4)),
        assumed_align=4,
    )
    hbuf = fake(BFloat16, (16, i), (i, 1), assumed_align=16)
    counters = fake(Int32, (i // 128,), (1,), assumed_align=4)
    qx = fake(Float8E4M3FN, (m, k), (k, 1), assumed_align=16) if debug else None
    sx = fake(Float32, (m, k // 128), (k // 128, 1), assumed_align=4) if debug else None
    gate_up = fake(BFloat16, (m, 2 * i), (2 * i, 1), assumed_align=16) if debug else None
    major, minor = capability
    return cute.compile(
        (MMAKernel if config.engine == "mma" else Kernel)(
            i, k, bucket, config, limit, ue8m0, debug
        ),
        x,
        weight,
        sw,
        q,
        scale,
        hbuf,
        counters,
        qx,
        sx,
        gate_up,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a",
    )
