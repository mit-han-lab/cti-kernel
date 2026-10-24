# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Eight-CTA exit decode with L2 prefetch and deterministic DSMEM exchange."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

from .kernels import Kernel, _fma, _fp, _load8, _ptr, _store8


def _load_remote(pointer):
    # CuTe's generic arch.load currently excludes the DSMEM address space.
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Int32(pointer.toint()).ir_value()],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class ClusterKernel(Kernel):
    @cute.jit
    def __call__(
        self,
        x: Int64,
        r: Int64,
        p: Int64,
        c: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        w: Int64,
        ro: Int64,
        po: Int64,
        co: Int64,
        out: Int64,
        parts: Int64,
        counter: cute.Tensor,
        tokens: Int32,
        words: cute.Tensor,
        host: Int64,
        stride: Int64,
        rank: Int32,
        ysum: Int64,
        stream,
    ):
        # counter is only the common ABI's current-stream carrier; no access.
        self.cluster_kernel(x, r, p, c, fn, scale, base, w, ro, po, co, out, tokens).launch(
            grid=(8, 1, 1),
            block=(256, 1, 1),
            cluster=(8, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def warp_partial(self, value, partials, t, m: cutlass.Constexpr, warp, lane):
        for mask in (4, 8, 16):
            value = _fp(value, cute.arch.shuffle_sync_bfly(value, mask), "add")
        if const_expr(m < 25):
            for mask in (1, 2):
                value = _fp(value, cute.arch.shuffle_sync_bfly(value, mask), "add")
            if lane == 0:
                index = (t * 8 + warp) * 35 + m
                partials[index] = _fp(partials[index], value, "add")
        else:
            j = lane % 4
            if (lane < 4) & (j <= m - 25):
                index = 25 + j * 4 - j * (j - 1) // 2 + m - 25 - j
                index += (t * 8 + warp) * 35
                partials[index] = _fp(partials[index], value, "add")

    @cute.kernel
    def cluster_kernel(
        self,
        x: Int64,
        r: Int64,
        p: Int64,
        c: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        w: Int64,
        ro: Int64,
        po: Int64,
        co: Int64,
        out: Int64,
        tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        j, group = lane % 4, lane // 4
        size = self.size
        h0 = bid * size
        smem = cutlass.utils.SmemAllocator()
        cache = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 4 * size), 16)
        warp_parts = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 8 * 35), 16)
        # Keep publication separate from totals: peers can still be reading it.
        partials = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 35), 16)
        totals = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 35), 16)
        norm = smem.allocate_tensor(Float32, cute.make_layout(size), 16)
        small = smem.allocate_tensor(Float32, cute.make_layout(27), 16)
        gains = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 5), 16)
        self.trace_push("mhc.prefetch")
        # One bounded, 16-byte-aligned bulk request per stream/projection row.
        # This is a hint, not an async copy requiring a completion wait.
        if (lane == 0) & (h0 < self.h):
            # Bulk prefetch is a uniform instruction. Elect one lane per warp
            # to avoid the compiler serializing 32 distinct lane addresses.
            for row in range(warp, 96, 8):
                address = fn + Int64((row * self.h + h0) * 4)
                count = cutlass.min(size, self.h - h0) * 4
                cute.arch.inline_ptx(
                    "cp.async.bulk.prefetch.L2.global [{$r0}], {$r1};",
                    read_only_args=[address, count],
                )
        self.stage_small(scale, base, small, tid)
        for v in range(tid, size // 8, 256):
            nv = cute.make_rmem_tensor((8,), Float32)
            nv.fill(0.0)
            if h0 + v * 8 < self.h:
                nv.store(_load8(w + Int64((h0 + v * 8) * 2)))
            for k in cutlass.range_constexpr(8):
                norm[v * 8 + k] = nv[k]
        for v in range(tid, self.cap * 8 * 35, 256):
            warp_parts[v] = Float32(0.0)
        cute.arch.sync_threads()
        self.trace_pop()
        self.trace_push("mhc.grid_wait")
        cute.arch.griddepcontrol_wait()
        self.trace_pop()
        if const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # S=576 (e.g. H=4104) needs a second, masked 512-column iteration.
        for piece in cutlass.range_constexpr((size + 511) // 512):
            col = piece * 512 + warp * 64 + group * 8
            h = h0 + col
            rv = cute.make_rmem_tensor((8, self.cap), Float32)
            rv.fill(0.0)
            self.trace_push("mhc.loads_residual")
            for t in cutlass.range_constexpr(self.cap):
                if (t < tokens) & (col < size) & (h < self.h):
                    loaded = self.residual(x, r, p, c, ro, t, h, j)
                    for k in cutlass.range_constexpr(8):
                        rv[k, t] = loaded[k]
                        cache[(t * 4 + j) * size + col + k] = loaded[k]
            self.trace_pop()
            self.trace_push("mhc.projection")
            # Four rows at a time bounds live accumulators to 4*T plus 8*T R'.
            # Each fn vector is loaded once, outside the unrolled token loop.
            for row in range(6):
                sums = cute.make_rmem_tensor((self.cap, 4), Float32)
                sums.fill(0.0)
                for m in cutlass.range_constexpr(4):
                    for half in cutlass.range_constexpr(2):
                        fv = cute.make_rmem_tensor((4,), Float32)
                        fv.fill(0.0)
                        if (col < size) & (h < self.h):
                            address = fn + Int64(
                                (((row * 4 + m) * 4 + j) * self.h + h + half * 4) * 4
                            )
                            raw = cute.arch.load(
                                _ptr(Int32, address, 16), ir.VectorType.get([4], Int32.mlir_type)
                            )
                            fv.store(cute.TensorSSA(raw, (4,), Int32).bitcast(Float32))
                        for t in cutlass.range_constexpr(self.cap):
                            for k in cutlass.range_constexpr(4):
                                sums[t, m] = _fma(fv[k], rv[half * 4 + k, t], sums[t, m])
                for t in cutlass.range_constexpr(self.cap):
                    for m in cutlass.range_constexpr(4):
                        value = sums[t, m]
                        for mask in (4, 8, 16, 1, 2):
                            value = _fp(value, cute.arch.shuffle_sync_bfly(value, mask), "add")
                        if lane == 0:
                            index = (t * 8 + warp) * 35 + row * 4 + m
                            warp_parts[index] = _fp(warp_parts[index], value, "add")
            # All lanes enter every shuffle, including fully padded warps.
            for t in cutlass.range_constexpr(self.cap):
                gram = cute.make_rmem_tensor((5,), Float32)
                gram.fill(0.0)
                for k in cutlass.range_constexpr(8):
                    gram[0] = _fma(rv[k, t], rv[k, t], gram[0])
                    for stream_index in cutlass.range_constexpr(4):
                        other = cute.arch.shuffle_sync(rv[k, t], group * 4 + stream_index)
                        gram[1 + stream_index] = _fma(rv[k, t], other, gram[1 + stream_index])
                for m in cutlass.range_constexpr(5):
                    self.warp_partial(gram[m], warp_parts, t, 24 + m, warp, lane)
            self.trace_pop()
        self.trace_push("mhc.cta_reduce")
        cute.arch.sync_threads()
        for v in range(tid, tokens * 35, 256):
            value = Float32(0.0)
            for owner in cutlass.range_constexpr(8):
                value = _fp(value, warp_parts[((v // 35) * 8 + owner) * 35 + v % 35], "add")
            partials[v] = value
        self.trace_pop()
        self.trace_push("mhc.cluster_exchange")
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for v in range(tid, tokens * 35, 256):
            value = Float32(0.0)
            for peer in cutlass.range_constexpr(8):
                remote = cute.arch.map_dsmem_ptr(partials.iterator + v, peer)
                value = _fp(value, _load_remote(remote), "add")
            totals[v] = value
        # Release/acquire also protects DSMEM lifetime before any CTA can exit.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        self.trace_pop()
        self.trace_push("mhc.coefficients")
        if const_expr(self.chain == "lane16"):
            for first in range(warp * 2, tokens, 16):
                t = first + lane // 16
                tc = cutlass.min(t, tokens - 1)
                self.coefficients_lane16(
                    totals, tc * 35, small, gains, tc * 5, po, co, tc, bid == 0, t < tokens
                )
        else:
            if tid < tokens:
                self.coefficients(totals, tid * 35, small, gains, tid * 5, po, co, tid, bid == 0)
        cute.arch.sync_threads()
        self.trace_pop()
        self.trace_push("mhc.output")
        for t in range(tokens):
            for piece in cutlass.range_constexpr((size + 511) // 512):
                col = piece * 512 + warp * 64 + group * 8
                values = cute.make_rmem_tensor((8,), Float32)
                for k in cutlass.range_constexpr(8):
                    own = Float32(0.0)
                    if (col < size) & (h0 + col < self.h):
                        own = cache[(t * 4 + j) * size + col + k]
                    y = Float32(0.0)
                    for stream_index in cutlass.range_constexpr(4):
                        other = cute.arch.shuffle_sync(own, group * 4 + stream_index)
                        y = _fma(gains[t * 5 + stream_index], other, y)
                    nv = Float32(0.0)
                    if col < size:
                        nv = norm[col + k]
                    values[k] = _fp(_fp(y, gains[t * 5 + 4], "mul"), nv, "mul")
                if (j == 0) & (col < size) & (h0 + col < self.h):
                    _store8(out + Int64((t * self.h + h0 + col) * 2), values.load())
        self.trace_pop()
        if const_expr(not self.early):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()
