# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original tiled MoE LoRA prefill: BF16 MMA inputs, FP32 accumulators.

Each CTA owns an aligned row block and an N chunk of every slice. Shrink
fetches each gathered BK64/BK128 activation tile once, with factor panels pipelined
across rank and K. Expand pipelines factor panels together with scattered cache
rows. Cache arithmetic stays in shared memory until coalesced vector stores.
The two phases alias their workspaces; large shrink accumulators live in FP32
shared memory to bound register lifetimes.
PTX semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.utils import SmemAllocator


@cute.jit
def copy16(dst, src, offset, valid):
    address = src.toint() + (offset if valid else Int64(0)) * 2
    size = Int32(16) if valid else Int32(0)
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[Int32(dst.toint()), address, size],
    )


@cute.jit
def load_a(ptr):
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[Int32(ptr.toint())],
    )


@cute.jit
def load_b(ptr):
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[Int32(ptr.toint())],
    )


@cute.jit
def mma(a, b, c):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]],
    )


@cute.jit
def add_pair(ptr, first, second, multiplier):
    # Preserve each original half word when its delta is zero, including -0 and
    # NaN payloads. Separate mul/add prevent contraction across the cache add.
    cute.arch.inline_ptx(
        "{ .reg .b32 old, lo, hi, packed, bits; .reg .f32 d0, d1, f0, f1; "
        ".reg .pred p0, p1, any; "
        "mul.rn.f32 d0, {$r1}, {$r3}; mul.rn.f32 d1, {$r2}, {$r3}; "
        "setp.neu.f32 p0, d0, 0f00000000; setp.neu.f32 p1, d1, 0f00000000; "
        "or.pred any, p0, p1; "
        "@any ld.shared.b32 old, [{$r0}]; "
        "@any shl.b32 bits, old, 16; @any mov.b32 f0, bits; "
        "@any and.b32 bits, old, 0xffff0000; @any mov.b32 f1, bits; "
        "@any add.rn.f32 f0, f0, d0; @any add.rn.f32 f1, f1, d1; "
        "@any cvt.rn.bf16x2.f32 packed, f1, f0; "
        "@any selp.b32 lo, packed, old, p0; @any and.b32 lo, lo, 0x0000ffff; "
        "@any selp.b32 hi, packed, old, p1; @any and.b32 hi, hi, 0xffff0000; "
        "@any or.b32 packed, lo, hi; @any st.shared.b32 [{$r0}], packed; }",
        read_only_args=[Int32(ptr.toint()), first, second, multiplier],
    )


@cute.jit
def silu_fp32(g):
    # Accurate exp lowering (no fastmath), including overflow/subnormal handling.
    # RN division avoids the error amplification of an approximate reciprocal.
    exponential = cute.math.exp(-g, fastmath=False)
    denominator = cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, 0f3F800000;",
        write_only_types=[Float32],
        read_only_args=[exponential],
    )
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[g, denominator],
    )


@cute.jit
def cache_add(base, delta, scale):
    # Keep scale multiplication separate from addition; zero updates preserve
    # the base signed zero, including the zero-adapter activation contract.
    return cute.arch.inline_ptx(
        "{ .reg .f32 d, sum; .reg .pred zero; mul.rn.f32 d, {$r1}, {$r2}; "
        "setp.eq.f32 zero, d, 0f00000000; add.rn.f32 sum, {$r0}, d; "
        "selp.f32 {$w0}, {$r0}, sum, zero; }",
        write_only_types=[Float32],
        read_only_args=[base, delta, scale],
    )


@cute.jit
def store16(dst, src):
    cute.arch.inline_ptx(
        "{ .reg .b32 v0, v1, v2, v3; "
        "ld.shared.v4.b32 {v0, v1, v2, v3}, [{$r1}]; "
        "st.global.v4.b32 [{$r0}], {v0, v1, v2, v3}; }",
        read_only_args=[dst.toint(), Int32(src.toint())],
    )


class BlockKernel:
    def __init__(
        self,
        k,
        n,
        rank,
        slices,
        top_k,
        experts,
        *,
        down=False,
        shared=False,
        block_m=32,
        chunks=1,
        save_z=False,
        block_n=64,
        stages=3,
        block_k=64,
        shrink_stages=None,
        act=False,
        stage="full",
        row_divisor=None,
        row_stride=None,
        weighted=True,
    ):
        # Down: scale each pair's delta by its routing weight (unweighted: the delta alone).
        self.weighted = weighted
        if stage not in ("full", "shrink", "expand"):
            raise ValueError("stage must be full, shrink, or expand")
        if stage != "full" and not save_z:
            raise ValueError("split stages require a BF16 z buffer")
        self.stage = stage
        if block_m not in (16, 32, 64):
            raise ValueError("block_m must be 16, 32, or 64")
        if block_n not in (64, 128) or block_k not in (64, 128):
            raise ValueError("block_n and block_k must be 64 or 128")
        shrink_stages = stages if shrink_stages is None else shrink_stages
        if stages not in (3, 4, 5, 6) or shrink_stages not in (3, 4, 5, 6):
            raise ValueError("stages and shrink_stages must be between 3 and 6")
        # Deeper pipelines apply only to shrink; retain expand's existing ring.
        stages = min(stages, 4)
        if slices not in (1, 2) or chunks < 1:
            raise ValueError("slices must be 1 or 2 and chunks must be positive")
        if act and (down or slices != 2):
            raise ValueError("act requires the gate/up block kernel")
        self.act = act
        self.k, self.n, self.r = k, n, rank
        self.slices, self.top_k, self.experts = slices, top_k, experts
        self.down, self.shared, self.save_z = down, shared, save_z
        self.row_divisor = (1 if down or shared else top_k) if row_divisor is None else row_divisor
        self.row_stride = k if row_stride is None else row_stride
        self.bm, self.bn, self.chunks, self.stages = block_m, block_n, chunks, stages
        self.bk, self.shrink_stages = block_k, shrink_stages
        self.total_r = slices * rank
        # Deep BK128 rings cannot hold a full 64-row factor panel. Reuse the
        # large-rank path's 32-row panels and shared accumulators in that case.
        self.shared_acc = self.total_r > 64 or (
            block_k == 128 and shrink_stages >= 5 and self.total_r > 32
        )
        # Large ranks stream 32/64-row factor panels. Their FP32 accumulators stay
        # in registers (unrolled panel groups) while each warp holds at most 64;
        # otherwise they live in shared memory between panels.
        self.reg_acc = self.shared_acc and block_m // 16 * ((self.total_r + 31) // 32) <= 16
        wide = self.reg_acc and block_k == 64 and self.total_r % 64 == 0
        self.apanel = (64 if wide else 32) if self.shared_acc else (self.total_r + 7) // 8 * 8
        self.agroups = (self.total_r + self.apanel - 1) // self.apanel
        self.split_k = max(1, 32 // self.apanel)
        self.rt = (self.apanel + 31) // 32
        self.arows = self.rt * (32 // self.split_k)
        self.aswizzle = block_k == 128 or (self.total_r > 224 and shrink_stages >= 4)
        # BK128 rows span two bank cycles; XOR eight row segments so each
        # eight-row ldmatrix group addresses distinct banks without padding.
        self.swizzle_rows = block_k // 16
        self.apitch = block_k if self.aswizzle else block_k + 8
        self.xpitch = self.apitch
        self.xstages = (shrink_stages - 2) // self.agroups + 2
        self.xsize = block_m * self.xpitch
        self.asize = self.arows * self.apitch
        self.acc_r = (self.total_r + 31) // 32 * 32
        self.acc_size = block_m * self.acc_r if self.shared_acc and not self.reg_acc else 0
        self.partial_size = self.split_k * block_m * self.total_r if self.split_k > 1 else 0
        self.rpad = (rank + 15) // 16 * 16
        # XOR 8-element segments at large ranks instead of padding each row;
        # this leaves room for static shared storage and the driver reservation.
        self.zswizzle = rank >= 64 and self.rpad % 32 == 0
        self.zpitch = self.rpad if self.zswizzle else self.rpad + 8
        self.bpanel = 32 if self.rpad % 32 == 0 and not (block_n == 128 and stages == 4) else 16
        self.bpitch = self.bpanel + 8
        self.bgroups = self.rpad // self.bpanel
        self.nparts = block_n // 64
        self.expand_acc = block_m * 64 if self.bgroups > 1 else 0
        self.cstages = (stages - 2) // (self.bgroups * self.nparts) + 2
        self.bsize = 64 * self.bpitch
        self.csize = block_m * block_n
        self.zsize = slices * block_m * self.zpitch
        self.nt = ((n + block_n - 1) // block_n + chunks - 1) // chunks
        shrink_bytes = 2 * (self.xstages * self.xsize + shrink_stages * self.asize)
        shrink_bytes += 4 * (self.acc_size + self.partial_size)
        expand_bytes = 2 * (self.zsize + stages * self.bsize + self.cstages * self.csize)
        self.gate_size = self.csize if act else 0
        expand_bytes += 4 * (self.expand_acc + self.gate_size)
        # Z conversion follows a full drain, so it may overwrite the shrink ring.
        # Shared FP32 accumulators sit beyond Z to make conversion race-free.
        convert_bytes = 2 * self.zsize + 4 * (self.acc_size + self.partial_size)
        self.workspace_bytes = max(shrink_bytes, expand_bytes, convert_bytes)
        self.smem_bytes = self.workspace_bytes + block_m * 4
        if self.smem_bytes + 1024 > 99 * 1024:
            raise ValueError("block configuration exceeds the 99 KiB shared-memory limit")

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        a: cute.Tensor,
        a_gap: Int64,
        b: cute.Tensor,
        b_gap: Int64,
        out: cute.Tensor,
        ids,
        weights,
        scale: cute.Tensor,
        z,
        sorted_ids,
        expert_ids,
        padded,
        stream: cuda.CUstream,
        activated=None,
    ):
        yp = activated.iterator if const_expr(self.act and self.stage != "shrink") else out.iterator
        ip = ids.iterator if const_expr(ids is not None) else x.iterator
        wp = weights.iterator if const_expr(weights is not None) else x.iterator
        zp = z.iterator if const_expr(z is not None) else x.iterator
        si = sorted_ids.iterator if const_expr(not self.shared) else x.iterator
        ei = expert_ids.iterator if const_expr(not self.shared) else x.iterator
        pp = padded.iterator if const_expr(not self.shared) else x.iterator
        capacity = sorted_ids.shape[0] if const_expr(not self.shared) else out.shape[0]
        self.main(
            x.iterator,
            a.iterator,
            a_gap,
            b.iterator,
            b_gap,
            out.iterator,
            yp,
            ip,
            wp,
            scale.iterator,
            zp,
            si,
            ei,
            pp,
            out.shape[0],
            capacity,
        ).launch(
            grid=(cute.ceil_div(capacity, self.bm), 1, self.chunks),
            block=(128, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
            preferred_smem_carveout=100,
        )

    @cute.jit
    def stage_shrink(self, sx, sa, x, a, a_gap, pairs, e, unit, tid):
        tile, group = unit // self.agroups, unit % self.agroups
        if group == 0:
            for j in cutlass.range((self.bm * (self.bk // 8) + 127) // 128, unroll=1):
                v = tid + j * 128
                m, col = v // (self.bk // 8), (v % (self.bk // 8)) * 8
                if m < self.bm:
                    pair = pairs[m]
                    row = pair // self.row_divisor
                    scol = col ^ ((m % self.swizzle_rows) * 8) if const_expr(self.aswizzle) else col
                    copy16(
                        sx + (tile % self.xstages) * self.xsize + m * self.xpitch + scol,
                        x,
                        Int64(row) * self.row_stride + tile * self.bk + col,
                        (pair >= 0) & (tile * self.bk + col < self.k),
                    )
        for j in cutlass.range((self.arows * (self.bk // 8) + 127) // 128, unroll=1):
            v = tid + j * 128
            r, col = v // (self.bk // 8), (v % (self.bk // 8)) * 8
            if r < self.arows:
                rr = group * self.apanel + r
                scol = col ^ ((r % self.swizzle_rows) * 8) if const_expr(self.aswizzle) else col
                copy16(
                    sa + (unit % self.shrink_stages) * self.asize + r * self.apitch + scol,
                    a,
                    # Row rr of the slice-major [slices*r, K] panel: slice rr // r.
                    Int64(rr // self.r) * a_gap
                    + (Int64(e) * self.r + rr % self.r) * self.k
                    + tile * self.bk
                    + col,
                    (rr < self.total_r) & (tile * self.bk + col < self.k),
                )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def stage_expand(self, sb, sc, b, b_gap, out, pairs, e, unit, chunk, tid):
        segment, group = unit // self.bgroups, unit % self.bgroups
        output, half = segment // self.nparts, segment % self.nparts
        s, tile = output % self.slices, output // self.slices
        for j in cutlass.range((64 * (self.bpanel // 8) + 127) // 128, unroll=1):
            v = tid + j * 128
            row, col = v // (self.bpanel // 8), (v % (self.bpanel // 8)) * 8
            if row < 64:
                n = (chunk * self.nt + tile) * self.bn + half * 64 + row
                r = group * self.bpanel + col
                copy16(
                    sb + (unit % self.stages) * self.bsize + row * self.bpitch + col,
                    b,
                    Int64(s) * b_gap + (Int64(e) * self.n + n) * self.r + r,
                    (tile < self.nt) & (n < self.n) & (r < self.r),
                )
        if (group == 0) & (half == 0):
            for j in cutlass.range((self.bm * (self.bn // 8) + 127) // 128, unroll=1):
                v = tid + j * 128
                row, col = v // (self.bn // 8), (v % (self.bn // 8)) * 8
                if row < self.bm:
                    pair = pairs[row]
                    n = (chunk * self.nt + tile) * self.bn + col
                    copy16(
                        sc + (output % self.cstages) * self.csize + row * self.bn + col,
                        out,
                        Int64(pair) * self.slices * self.n + s * self.n + n,
                        (pair >= 0) & (tile < self.nt) & (n < self.n),
                    )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def save(self, sz, z, pairs, m, rr, chunk, value):
        s, r = rr // self.r, rr % self.r
        col = r ^ ((m % 4) * 8) if const_expr(self.zswizzle) else r
        sz[(s * self.bm + m) * self.zpitch + col] = value.to(BFloat16)
        if const_expr(self.save_z):
            pair = pairs[m]
            if (pair >= 0) & (chunk == 0):
                z[Int64(pair) * self.total_r + rr] = value.to(z.dtype)

    @cute.jit
    def load_x(self, ax, m, lane, step):
        col = lane // 16 * 8 + step
        if const_expr(self.aswizzle):
            col = col ^ ((lane % self.swizzle_rows) * 8)
        return load_a(ax + (m * 16 + lane % 16) * self.xpitch + col)

    @cute.jit
    def restore_z(self, buf, sp, z, tid):
        # Include padding and masked rows: ldmatrix consumes the entire tile.
        for j in cutlass.range((self.zsize + 127) // 128, unroll=1):
            index = tid + j * 128
            if index < self.zsize:
                sm, col = index // self.zpitch, index % self.zpitch
                s, m = sm // self.bm, sm % self.bm
                r = col ^ ((m % 4) * 8) if const_expr(self.zswizzle) else col
                pair = sp[m]
                value = BFloat16(0)
                if (pair >= 0) & (r < self.r):
                    value = z[Int64(pair) * self.total_r + s * self.r + r]
                buf[index] = value
        cute.arch.sync_threads()

    @cute.jit
    def load_factor(self, aa, rn, lane, step):
        col = (lane // 8 % 2) * 8 + step
        if const_expr(self.aswizzle):
            col = col ^ ((lane % self.swizzle_rows) * 8)
        return load_b(aa + (rn + lane % 8) * self.apitch + col)

    @cute.jit
    def shrink_panel(self, ax, aa, scratch, group, lane, warp):
        for m in cutlass.range(self.bm // 16, unroll=1):
            acc = cute.make_rmem_tensor((4,), Float32)
            for v in cutlass.range_constexpr(4):
                mr = m * 16 + lane // 4 + (v // 2) * 8
                rr = group * self.apanel + warp * 8 + lane % 4 * 2 + v % 2
                acc[v] = scratch[mr * self.acc_r + rr]
            for kk in cutlass.range(self.bk // 16, unroll=1):
                af = self.load_x(ax, m, lane, kk * 16)
                bf = self.load_factor(aa, warp * 8, lane, kk * 16)
                result = mma(af, bf, (acc[0], acc[1], acc[2], acc[3]))
                for v in cutlass.range_constexpr(4):
                    acc[v] = result[v]
            for v in cutlass.range_constexpr(4):
                mr = m * 16 + lane // 4 + (v // 2) * 8
                rr = group * self.apanel + warp * 8 + lane % 4 * 2 + v % 2
                scratch[mr * self.acc_r + rr] = acc[v]

    @cute.jit
    def shrink(self, buf, sp, x, a, a_gap, z, e, chunk, tid):
        lane, warp = tid % 32, tid // 32
        sx = buf
        sa = sx + self.xstages * self.xsize
        # Keep shared accumulators at the end of the aliased workspace.
        scratch = cute.recast_ptr(
            buf + self.workspace_bytes // 2 - 2 * (self.acc_size + self.partial_size), dtype=Float32
        )
        if const_expr(self.shared_acc and not self.reg_acc):
            for j in cutlass.range((self.acc_size + 127) // 128, unroll=1):
                index = tid + j * 128
                if index < self.acc_size:
                    scratch[index] = Float32(0)
        acc = cute.make_rmem_tensor((self.bm // 16, self.agroups * self.rt, 4), Float32)
        acc.fill(0)
        cute.arch.sync_threads()
        for ahead in cutlass.range(self.shrink_stages - 1, unroll=1):
            self.stage_shrink(sx, sa, x, a, a_gap, sp, e, Int32(ahead), tid)
        if const_expr(self.reg_acc):
            for tile in range((self.k + self.bk - 1) // self.bk):
                for group in cutlass.range_constexpr(self.agroups):
                    unit = tile * self.agroups + group
                    self.stage_shrink(
                        sx, sa, x, a, a_gap, sp, e, unit + self.shrink_stages - 1, tid
                    )
                    cute.arch.cp_async_wait_group(self.shrink_stages - 1)
                    cute.arch.sync_threads()
                    ax = sx + (tile % self.xstages) * self.xsize
                    aa = sa + (unit % self.shrink_stages) * self.asize
                    for kk in cutlass.range(self.bk // 16, unroll=1):
                        for m in cutlass.range_constexpr(self.bm // 16):
                            af = self.load_x(ax, m, lane, kk * 16)
                            for r in cutlass.range_constexpr(self.rt):
                                c = group * self.rt + r
                                bf = self.load_factor(aa, warp * 8 + r * 32, lane, kk * 16)
                                result = mma(
                                    af, bf, (acc[m, c, 0], acc[m, c, 1], acc[m, c, 2], acc[m, c, 3])
                                )
                                for v in cutlass.range_constexpr(4):
                                    acc[m, c, v] = result[v]
                    cute.arch.sync_threads()
        for unit in range(
            0 if self.reg_acc else ((self.k + self.bk - 1) // self.bk) * self.agroups
        ):
            self.stage_shrink(sx, sa, x, a, a_gap, sp, e, unit + self.shrink_stages - 1, tid)
            cute.arch.cp_async_wait_group(self.shrink_stages - 1)
            cute.arch.sync_threads()
            tile, group = unit // self.agroups, unit % self.agroups
            ax = sx + (tile % self.xstages) * self.xsize
            aa = sa + (unit % self.shrink_stages) * self.asize
            if const_expr(self.shared_acc):
                self.shrink_panel(ax, aa, scratch, group, lane, warp)
            else:
                for kk in cutlass.range(self.bk // 16 // self.split_k, unroll=1):
                    step = (warp // (4 // self.split_k) * (self.bk // 16 // self.split_k) + kk) * 16
                    for m in cutlass.range_constexpr(self.bm // 16):
                        af = self.load_x(ax, m, lane, step)
                        for r in cutlass.range_constexpr(self.rt):
                            rn = (warp % (4 // self.split_k)) * 8 + r * 32
                            bf = self.load_factor(aa, rn, lane, step)
                            result = mma(
                                af, bf, (acc[m, r, 0], acc[m, r, 1], acc[m, r, 2], acc[m, r, 3])
                            )
                            for v in cutlass.range_constexpr(4):
                                acc[m, r, v] = result[v]
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # All asynchronous writes must drain before the workspace becomes Z.
        for j in cutlass.range((self.zsize + 127) // 128, unroll=1):
            index = tid + j * 128
            if index < self.zsize:
                buf[index] = BFloat16(0)
        cute.arch.sync_threads()
        if const_expr(self.reg_acc):
            for m in cutlass.range_constexpr(self.bm // 16):
                for c in cutlass.range_constexpr(self.agroups * self.rt):
                    for v in cutlass.range_constexpr(4):
                        mr = m * 16 + lane // 4 + (v // 2) * 8
                        rr = (
                            c // self.rt * self.apanel
                            + c % self.rt * 32
                            + warp * 8
                            + lane % 4 * 2
                            + v % 2
                        )
                        if rr < self.total_r:
                            self.save(buf, z, sp, mr, rr, chunk, acc[m, c, v])
        elif const_expr(self.shared_acc):
            for j in cutlass.range((self.acc_size + 127) // 128, unroll=1):
                index = tid + j * 128
                if index < self.acc_size:
                    rr = index % self.acc_r
                    if rr < self.total_r:
                        self.save(buf, z, sp, index // self.acc_r, rr, chunk, scratch[index])
        else:
            for m in cutlass.range_constexpr(self.bm // 16):
                for r in cutlass.range_constexpr(self.rt):
                    for v in cutlass.range_constexpr(4):
                        mr = m * 16 + lane // 4 + (v // 2) * 8
                        rr = warp % (4 // self.split_k) * 8 + r * 32 + lane % 4 * 2 + v % 2
                        if rr < self.total_r:
                            if const_expr(self.split_k > 1):
                                w = warp // (4 // self.split_k)
                                scratch[(w * self.bm + mr) * self.total_r + rr] = acc[m, r, v]
                            else:
                                self.save(buf, z, sp, mr, rr, chunk, acc[m, r, v])
            if const_expr(self.split_k > 1):
                cute.arch.sync_threads()
                for j in cutlass.range((self.bm * self.total_r + 127) // 128, unroll=1):
                    index = tid + j * 128
                    if index < self.bm * self.total_r:
                        value = Float32(0)
                        for w in cutlass.range_constexpr(self.split_k):
                            value += scratch[w * self.bm * self.total_r + index]
                        self.save(
                            buf, z, sp, index // self.total_r, index % self.total_r, chunk, value
                        )
        cute.arch.sync_threads()

    @cute.jit
    def expand(self, buf, sp, b, b_gap, out, activated, weights, scale, e, chunk, tid):
        lane, warp = tid % 32, tid // 32
        sb = buf + self.zsize
        sc = sb + self.stages * self.bsize
        accum = cute.recast_ptr(sc + self.cstages * self.csize, dtype=Float32)
        gate = accum + self.expand_acc
        for ahead in cutlass.range(self.stages - 1, unroll=1):
            self.stage_expand(sb, sc, b, b_gap, out, sp, e, Int32(ahead), chunk, tid)
        for output in range(self.nt * self.slices):
            s, tile = output % self.slices, output // self.slices
            cache = sc + (output % self.cstages) * self.csize
            # BN128 stages one contiguous cache tile, then consumes two BN64
            # factor segments. Both halves finish before the vector-store barrier.
            for half in cutlass.range(self.nparts, unroll=1):
                for group in range(self.bgroups):
                    unit = (output * self.nparts + half) * self.bgroups + group
                    self.stage_expand(
                        sb, sc, b, b_gap, out, sp, e, unit + self.stages - 1, chunk, tid
                    )
                    cute.arch.cp_async_wait_group(self.stages - 1)
                    cute.arch.sync_threads()
                    bb = sb + (unit % self.stages) * self.bsize
                    for m in cutlass.range(self.bm // 16, unroll=1):
                        expanded = cute.make_rmem_tensor((2, 4), Float32)
                        expanded.fill(0)
                        if const_expr(self.bgroups > 1):
                            if group > 0:
                                for n in cutlass.range_constexpr(2):
                                    for v in cutlass.range_constexpr(4):
                                        row = m * 16 + lane // 4 + (v // 2) * 8
                                        col = warp * 16 + n * 8 + lane % 4 * 2 + v % 2
                                        expanded[n, v] = accum[row * 64 + col]
                        for kk in cutlass.range(self.bpanel // 16, unroll=1):
                            col = lane // 16 * 8 + group * self.bpanel + kk * 16
                            if const_expr(self.zswizzle):
                                col = col ^ ((lane % 4) * 8)
                            af = load_a(
                                buf + (s * self.bm + m * 16 + lane % 16) * self.zpitch + col
                            )
                            for n in cutlass.range_constexpr(2):
                                bf = load_b(
                                    bb
                                    + (warp * 16 + n * 8 + lane % 8) * self.bpitch
                                    + (lane // 8 % 2) * 8
                                    + kk * 16
                                )
                                result = mma(
                                    af,
                                    bf,
                                    (
                                        expanded[n, 0],
                                        expanded[n, 1],
                                        expanded[n, 2],
                                        expanded[n, 3],
                                    ),
                                )
                                for v in cutlass.range_constexpr(4):
                                    expanded[n, v] = result[v]
                        if group == self.bgroups - 1:
                            for rowhalf in cutlass.range_constexpr(2):
                                row = m * 16 + lane // 4 + rowhalf * 8
                                pair = sp[row]
                                multiplier = scale[0]
                                if const_expr(self.down and not self.shared and self.weighted):
                                    if pair >= 0:
                                        multiplier *= weights[pair]
                                for n in cutlass.range_constexpr(2):
                                    col = half * 64 + warp * 16 + n * 8 + lane % 4 * 2
                                    if const_expr(self.act):
                                        for v in cutlass.range_constexpr(2):
                                            index = row * self.bn + col + v
                                            value = cache_add(
                                                cache[index].to(Float32),
                                                expanded[n, rowhalf * 2 + v],
                                                multiplier,
                                            )
                                            if s == 0:
                                                gate[index] = value
                                            else:
                                                cache[index] = (silu_fp32(gate[index]) * value).to(
                                                    BFloat16
                                                )
                                    else:
                                        add_pair(
                                            cache + row * self.bn + col,
                                            expanded[n, rowhalf * 2],
                                            expanded[n, rowhalf * 2 + 1],
                                            multiplier,
                                        )
                        else:
                            if const_expr(self.bgroups > 1):
                                for n in cutlass.range_constexpr(2):
                                    for v in cutlass.range_constexpr(4):
                                        row = m * 16 + lane // 4 + (v // 2) * 8
                                        col = warp * 16 + n * 8 + lane % 4 * 2 + v % 2
                                        accum[row * 64 + col] = expanded[n, v]
                    cute.arch.sync_threads()
            cute.arch.sync_threads()
            for j in cutlass.range((self.bm * (self.bn // 8) + 127) // 128, unroll=1):
                v = tid + j * 128
                row, col = v // (self.bn // 8), (v % (self.bn // 8)) * 8
                if row < self.bm:
                    pair = sp[row]
                    n = (chunk * self.nt + tile) * self.bn + col
                    if (pair >= 0) & (n < self.n):
                        if const_expr(self.act):
                            if s == 1:
                                store16(
                                    activated + Int64(pair) * self.n + n,
                                    cache + row * self.bn + col,
                                )
                        else:
                            store16(
                                out + Int64(pair) * self.slices * self.n + s * self.n + n,
                                cache + row * self.bn + col,
                            )
            # No thread can recycle this cache slot until every vector is stored.
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    @cute.kernel
    def main(
        self,
        x: cute.Pointer,
        a: cute.Pointer,
        a_gap: Int64,
        b: cute.Pointer,
        b_gap: Int64,
        out: cute.Pointer,
        activated: cute.Pointer,
        ids: cute.Pointer,
        weights: cute.Pointer,
        scale: cute.Pointer,
        z: cute.Pointer,
        sorted_ids: cute.Pointer,
        expert_ids: cute.Pointer,
        padded: cute.Pointer,
        pairs: Int32,
        capacity: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, chunk = cute.arch.block_idx()
        allocator = SmemAllocator()
        buf = allocator.allocate_tensor(
            BFloat16, cute.make_layout(self.workspace_bytes // 2), byte_alignment=16
        )
        sp = allocator.allocate_tensor(Int32, cute.make_layout(self.bm), byte_alignment=16)
        cute.arch.griddepcontrol_wait()
        limit = pairs
        if const_expr(not self.shared):
            limit = padded[0]
        if block * self.bm < limit:
            e = Int32(0)
            if const_expr(not self.shared):
                e = expert_ids[block]
            if (e >= 0) & (e < self.experts):
                if tid < self.bm:
                    index = block * self.bm + tid
                    pair = Int32(-1)
                    if (index < capacity) & (index < limit):
                        if const_expr(self.shared):
                            pair = index
                        else:
                            candidate = sorted_ids[index]
                            if (candidate >= 0) & (candidate < pairs):
                                if ids[candidate].to(Int64) == e.to(Int64):
                                    pair = candidate
                    sp[tid] = pair
                cute.arch.sync_threads()
                if const_expr(self.stage == "expand"):
                    self.restore_z(buf.iterator, sp.iterator, z, tid)
                else:
                    self.shrink(buf.iterator, sp.iterator, x, a, a_gap, z, e, chunk, tid)
                if const_expr(self.stage != "shrink"):
                    self.expand(
                        buf.iterator,
                        sp.iterator,
                        b,
                        b_gap,
                        out,
                        activated,
                        weights,
                        scale,
                        e,
                        chunk,
                        tid,
                    )
        cute.arch.griddepcontrol_launch_dependents()
