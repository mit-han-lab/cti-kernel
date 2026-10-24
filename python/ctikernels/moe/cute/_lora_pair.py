# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original SIMT shrink/expand kernels with 16-byte factor loads and FP32 z.

PTX instruction semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/
The aligned block implementation is the correctness fallback for a future MMA path.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.runtime import make_fake_stream
from cutlass.utils import SmemAllocator

from ._lora_block import BlockKernel


@cute.jit
def load16(ptr, offset, valid):
    return cute.arch.inline_ptx(
        "{ .reg .pred p; setp.ne.b32 p, {$r1}, 0; "
        "mov.b32 {$w0}, 0; mov.b32 {$w1}, 0; mov.b32 {$w2}, 0; mov.b32 {$w3}, 0; "
        "@p ld.global.nc.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}]; }",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[ptr.toint() + Int64(offset) * 2, Int32(valid)],
    )


@cute.jit
def unpack(bits):
    return cute.arch.inline_ptx(
        "{ .reg .b32 lo, hi; shl.b32 lo, {$r0}, 16; "
        "and.b32 hi, {$r0}, 0xffff0000; mov.b32 {$w0}, lo; mov.b32 {$w1}, hi; }",
        write_only_types=[Float32, Float32],
        read_only_args=[bits],
    )


@cute.jit
def fma(a, b, c):
    return cute.arch.inline_ptx(
        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Float32],
        read_only_args=[a, b, c],
    )


@cute.jit
def reduce_sum(value, width: cutlass.Constexpr):
    for step in cutlass.range_constexpr(width.bit_length() - 1):
        other = cute.arch.inline_ptx(
            "shfl.sync.bfly.b32 {$w0}, {$r0}, {$r1}, 31, -1;",
            write_only_types=[Float32],
            read_only_args=[value, Int32(1 << step)],
        )
        value = value + other
    return value


@cute.jit
def copy16(dst, src, offset, valid):
    # src-size 0 zero-fills; masked copies keep a valid base address.
    address = src.toint() + (Int64(offset) if valid else Int64(0)) * 2
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[Int32(dst.toint()), address, Int32(16) if valid else Int32(0)],
    )


@cute.jit
def shared16(ptr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[Int32(ptr.toint())],
    )


@cute.jit
def load_bf16(ptr, offset, valid):
    # Predicated 2-byte load returned as FP32 (BF16 bits in the high half).
    return cute.arch.inline_ptx(
        "{ .reg .pred p; .reg .b16 h; .reg .b32 w; setp.ne.b32 p, {$r1}, 0; "
        "mov.b16 h, 0; @p ld.global.b16 h, [{$r0}]; "
        "cvt.u32.u16 w, h; shl.b32 w, w, 16; mov.b32 {$w0}, w; }",
        write_only_types=[Float32],
        read_only_args=[ptr.toint() + Int64(offset) * 2, Int32(valid)],
    )


@cute.jit
def cluster_barrier():
    cute.arch.inline_ptx("barrier.cluster.arrive.release.aligned;", read_only_args=[])
    cute.arch.inline_ptx("barrier.cluster.wait.acquire.aligned;", read_only_args=[])


@cute.jit
def remote_load(ptr, peer):
    return cute.arch.inline_ptx(
        "{ .reg .b32 addr; mapa.shared::cluster.u32 addr, {$r0}, {$r1}; "
        "ld.shared::cluster.f32 {$w0}, [addr]; }",
        write_only_types=[Float32],
        read_only_args=[Int32(ptr.toint()), Int32(peer)],
    )


class Kernel:
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
        cluster=1,
        block_m=0,
        chunks=1,
        save_z=False,
        shrink_only=False,
        weighted=True,
    ):
        self.k, self.n, self.r = k, n, rank
        self.slices, self.top_k, self.experts = slices, top_k, experts
        self.down, self.shared = down, shared
        # Down: scale each pair's delta by its routing weight (unweighted: the delta alone).
        self.weighted = weighted
        self.cluster, self.bm, self.chunks, self.save_z = cluster, block_m, chunks, save_z
        if shrink_only and (block_m or not save_z):
            raise ValueError("a shrink-only launch is the pair kernel's, with z")
        # Shrink only: write z (pair order) and no cache, for callers that fuse the expand.
        self.shrink_only = shrink_only
        self.width = 1 << (rank // 8 - 1).bit_length()
        self.kvecs = (k // 8 + cluster - 1) // cluster
        self.kt = (self.kvecs + 31) // 32
        # Pair kernel: K vectors per cluster rank, lanes per rank row (power of two),
        # vectors per lane, rank rows per 128-thread pass, passes and load batches.
        self.pkvecs = (k // 8 + cluster - 1) // cluster
        self.pl = min(32, 1 << (self.pkvecs - 1).bit_length())
        self.pvl = (self.pkvecs + self.pl - 1) // self.pl
        self.prpp = 128 // self.pl
        self.ppasses = (rank + self.prpp - 1) // self.prpp
        self.ppb = max(1, min(self.ppasses, 16 // self.pvl))
        # Expand rows per cluster rank; stage the B slab in shared memory when it fits.
        self.prows = (n + cluster - 1) // cluster
        self.pitch = rank + 8
        self.pslab = self.prows * self.pitch * 2 <= 72 * 1024 and not shrink_only
        self.ptiles = (self.prows + 128 // self.width - 1) // (128 // self.width)

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
    ):
        ip = ids.iterator if const_expr(ids is not None) else x.iterator
        wp = weights.iterator if const_expr(weights is not None) else x.iterator
        zp = z.iterator if const_expr(z is not None) else x.iterator
        if const_expr(self.bm == 0):
            self.pair(
                x.iterator,
                a.iterator,
                a_gap,
                b.iterator,
                b_gap,
                x.iterator if const_expr(self.shrink_only) else out.iterator,
                ip,
                wp,
                scale.iterator,
                zp,
            ).launch(
                grid=(
                    (z.shape[0] if const_expr(self.shrink_only) else out.shape[0]) * self.cluster,
                    self.slices,
                    1,
                ),
                block=(128, 1, 1),
                cluster=(self.cluster, 1, 1) if const_expr(self.cluster > 1) else None,
                stream=stream,
                use_pdl=True,
            )
        else:
            self.block(
                x.iterator,
                a.iterator,
                a_gap,
                b.iterator,
                b_gap,
                out.iterator,
                ip,
                wp,
                scale.iterator,
                zp,
                sorted_ids.iterator,
                expert_ids.iterator,
                padded.iterator,
                out.shape[0],
                sorted_ids.shape[0],
            ).launch(
                grid=(cute.ceil_div(sorted_ids.shape[0], self.bm), self.slices, self.chunks),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def shrink(self, x, a, a_gap, sx, row, e, s, valid, split, lane, warp):
        # Cache each x vector once per warp. Prefetch every independent A vector
        # for this rank row before starting its dependent FMA chain.
        xv = cute.make_rmem_tensor((self.kt, 4), Int32)
        av = cute.make_rmem_tensor((self.kt, 4), Int32)
        for t in cutlass.range_constexpr(self.kt):
            v = lane + t * 32
            col = (split * self.kvecs + v) * 8
            values = load16(x, Int64(row) * self.k + col, valid & (v < self.kvecs) & (col < self.k))
            for j in cutlass.range_constexpr(4):
                xv[t, j] = values[j]
        for rr in range(self.r // 4):
            r = rr * 4 + warp
            for t in cutlass.range_constexpr(self.kt):
                v = lane + t * 32
                col = (split * self.kvecs + v) * 8
                values = load16(
                    a,
                    Int64(s) * a_gap + (Int64(e) * self.r + r) * self.k + col,
                    valid & (v < self.kvecs) & (col < self.k),
                )
                for j in cutlass.range_constexpr(4):
                    av[t, j] = values[j]
            acc = Float32(0)
            for t in cutlass.range_constexpr(self.kt):
                for j in cutlass.range_constexpr(4):
                    aa, xx = unpack(av[t, j]), unpack(xv[t, j])
                    acc = fma(aa[0], xx[0], acc)
                    acc = fma(aa[1], xx[1], acc)
            acc = reduce_sum(acc, 32)
            if lane == 0:
                sx[r] = acc

    @cute.jit
    def expand(
        self,
        b,
        b_gap,
        out,
        sx,
        weights,
        scale,
        pair,
        e,
        s,
        valid,
        part,
        parts: cutlass.Constexpr,
        tid,
    ):
        width = self.width
        group, lane = tid // width, tid % width
        groups = 128 // width
        rows = (self.n + parts - 1) // parts
        tiles = (rows + groups - 1) // groups
        prefetch = min(8, tiles)
        bv = cute.make_rmem_tensor((prefetch, 4), Int32)
        zv = cute.make_rmem_tensor(8, Float32)
        zv.fill(0)
        if lane * 8 < self.r:
            for j in cutlass.range_constexpr(8):
                zv[j] = sx[lane * 8 + j]
        multiplier = scale[0]
        if const_expr(self.down and not self.shared and self.weighted):
            if valid:
                multiplier = multiplier * weights[pair]
        for batch in range((tiles + prefetch - 1) // prefetch):
            # Issue a batch of independent B vectors before any dependent FMAs.
            for d in cutlass.range_constexpr(prefetch):
                tile = batch * prefetch + d
                n = part * rows + tile * groups + group
                valid_n = valid & (n < self.n) & (tile * groups + group < rows)
                values = load16(
                    b,
                    Int64(s) * b_gap + (Int64(e) * self.n + n) * self.r + lane * 8,
                    valid_n & (lane * 8 < self.r),
                )
                for j in cutlass.range_constexpr(4):
                    bv[d, j] = values[j]
            for d in cutlass.range_constexpr(prefetch):
                tile = batch * prefetch + d
                n = part * rows + tile * groups + group
                valid_n = valid & (n < self.n) & (tile * groups + group < rows)
                acc = Float32(0)
                for j in cutlass.range_constexpr(4):
                    bb = unpack(bv[d, j])
                    acc = fma(bb[0], zv[2 * j], acc)
                    acc = fma(bb[1], zv[2 * j + 1], acc)
                acc = reduce_sum(acc, width)
                if valid_n & (lane == 0):
                    offset = Int64(pair) * (self.slices * self.n) + s * self.n + n
                    # Explicit mul/add forbid contraction across the cache addition.
                    delta = cute.arch.inline_ptx(
                        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32],
                        read_only_args=[multiplier, acc],
                    )
                    # A zero adapter also preserves the sign bit of a zero cache.
                    if delta != Float32(0):
                        value = cute.arch.inline_ptx(
                            "add.rn.f32 {$w0}, {$r0}, {$r1};",
                            write_only_types=[Float32],
                            read_only_args=[out[offset].to(Float32), delta],
                        )
                        out[offset] = value.to(out.dtype)

    @cute.jit
    def pair_shrink(self, x, a, a_gap, sx, row, e, s, valid, split, tid):
        # One load trip: every A vector of this rank's K range is issued (in batches of
        # up to 16 per thread) before the dependent FMA chains; pl lanes share a rank row.
        li, rp = tid % self.pl, tid // self.pl
        kv = self.k // 8
        xv = cute.make_rmem_tensor((self.pvl, 4), Int32)
        for j in cutlass.range_constexpr(self.pvl):
            v = li + j * self.pl
            cv = split * self.pkvecs + v
            values = load16(x, Int64(row) * self.k + cv * 8, valid & (v < self.pkvecs) & (cv < kv))
            for w in cutlass.range_constexpr(4):
                xv[j, w] = values[w]
        av = cute.make_rmem_tensor((self.ppb, self.pvl, 4), Int32)
        base = Int64(s) * a_gap + e * (self.r * self.k)
        for bt in cutlass.range_constexpr((self.ppasses + self.ppb - 1) // self.ppb):
            for p in cutlass.range_constexpr(self.ppb):
                rr = (bt * self.ppb + p) * self.prpp + rp
                for j in cutlass.range_constexpr(self.pvl):
                    v = li + j * self.pl
                    cv = split * self.pkvecs + v
                    values = load16(
                        a,
                        base + rr * self.k + cv * 8,
                        valid & (rr < self.r) & (v < self.pkvecs) & (cv < kv),
                    )
                    for w in cutlass.range_constexpr(4):
                        av[p, j, w] = values[w]
            for p in cutlass.range_constexpr(self.ppb):
                rr = (bt * self.ppb + p) * self.prpp + rp
                acc = Float32(0)
                for j in cutlass.range_constexpr(self.pvl):
                    for w in cutlass.range_constexpr(4):
                        aa, xx = unpack(av[p, j, w]), unpack(xv[j, w])
                        acc = fma(aa[0], xx[0], acc)
                        acc = fma(aa[1], xx[1], acc)
                acc = reduce_sum(acc, self.pl)
                if (li == 0) & (rr < self.r):
                    sx[rr] = acc

    @cute.jit
    def pair_stage_b(self, b, b_gap, sb, e, s, valid, split, tid):
        # B rows of this rank's output range are static once the expert is known:
        # stream them into shared memory while the shrink runs.
        rv = self.r // 8
        vecs = self.prows * rv
        for i in cutlass.range_constexpr((vecs + 127) // 128):
            idx = tid + i * 128
            if idx < vecs:
                rl, cv = idx // rv, idx % rv
                n = split * self.prows + rl
                copy16(
                    sb + rl * self.pitch + cv * 8,
                    b,
                    Int64(s) * b_gap + (e * self.n + n) * self.r + cv * 8,
                    valid & (n < self.n),
                )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def pair_expand_slab(self, out, sz, sb, weights, scale, pair, s, valid, split, tid):
        width = self.width
        group, lane = tid // width, tid % width
        groups = 128 // width
        zv = cute.make_rmem_tensor(8, Float32)
        zv.fill(0)
        if lane * 8 < self.r:
            for j in cutlass.range_constexpr(8):
                zv[j] = sz[lane * 8 + j]
        multiplier = scale[0]
        if const_expr(self.down and not self.shared and self.weighted):
            if valid:
                multiplier = multiplier * weights[pair]
        row_base = Int64(pair) * (self.slices * self.n) + s * self.n + split * self.prows
        # Prefetch the cache values this lane will update (independent of the shrink).
        ov = cute.make_rmem_tensor(self.ptiles, Float32)
        for t in cutlass.range_constexpr(self.ptiles):
            rl = t * groups + group
            ok = valid & (lane == 0) & (rl < self.prows) & (split * self.prows + rl < self.n)
            ov[t] = load_bf16(out, row_base + rl, ok)
        for t in cutlass.range_constexpr(self.ptiles):
            rl = t * groups + group
            ok = valid & (rl < self.prows) & (split * self.prows + rl < self.n)
            acc = Float32(0)
            if lane * 8 < self.r:
                if rl < self.prows:
                    bits = shared16(sb + rl * self.pitch + lane * 8)
                    for j in cutlass.range_constexpr(4):
                        bb = unpack(bits[j])
                        acc = fma(bb[0], zv[2 * j], acc)
                        acc = fma(bb[1], zv[2 * j + 1], acc)
            acc = reduce_sum(acc, width)
            if ok & (lane == 0):
                # Explicit mul/add forbid contraction across the cache addition.
                delta = cute.arch.inline_ptx(
                    "mul.rn.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Float32],
                    read_only_args=[multiplier, acc],
                )
                # A zero adapter also preserves the sign bit of a zero cache.
                if delta != Float32(0):
                    value = cute.arch.inline_ptx(
                        "add.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32],
                        read_only_args=[ov[t], delta],
                    )
                    out[row_base + rl] = value.to(out.dtype)

    @cute.kernel
    def pair(
        self,
        x: cute.Pointer,
        a: cute.Pointer,
        a_gap: Int64,
        b: cute.Pointer,
        b_gap: Int64,
        out: cute.Pointer,
        ids: cute.Pointer,
        weights: cute.Pointer,
        scale: cute.Pointer,
        z: cute.Pointer,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, s, _ = cute.arch.block_idx()
        pair, split = bx // self.cluster, bx % self.cluster
        allocator = SmemAllocator()
        smem = allocator.allocate_tensor(Float32, cute.make_layout(2 * self.r), byte_alignment=16)
        sb = smem.iterator
        if const_expr(self.pslab):
            sb = allocator.allocate_tensor(
                BFloat16, cute.make_layout(self.prows * self.pitch), byte_alignment=16
            ).iterator
        cute.arch.griddepcontrol_wait()
        e = Int64(0)
        if const_expr(not self.shared):
            e = ids[pair].to(Int64)
        valid = (e >= 0) & (e < self.experts)
        row = pair if const_expr(self.down) else pair // self.top_k
        if const_expr(self.pslab):
            self.pair_stage_b(b, b_gap, sb, e, s, valid, split, tid)
        self.pair_shrink(x, a, a_gap, smem.iterator, row, e, s, valid, split, tid)
        if const_expr(self.cluster > 1):
            cluster_barrier()
        else:
            cute.arch.sync_threads()
        if tid < self.r:
            total = Float32(0)
            for peer in cutlass.range_constexpr(self.cluster):
                if const_expr(self.cluster > 1):
                    total = total + remote_load(smem.iterator + tid, peer)
                else:
                    total = smem[tid]
            smem[self.r + tid] = total
            if const_expr(self.save_z):
                if valid & (split == 0):
                    z[(Int64(pair) * self.slices + s) * self.r + tid] = total.to(z.dtype)
        if const_expr(self.pslab):
            cute.arch.cp_async_wait_group(0)
        # All peers finish DSMEM reads before any CTA can exit or reuse shared memory.
        if const_expr(self.cluster > 1):
            cluster_barrier()
        else:
            cute.arch.sync_threads()
        if const_expr(self.shrink_only):
            pass
        elif const_expr(self.pslab):
            self.pair_expand_slab(
                out, smem.iterator + self.r, sb, weights, scale, pair, s, valid, split, tid
            )
        else:
            self.expand(
                b,
                b_gap,
                out,
                smem.iterator + self.r,
                weights,
                scale,
                pair,
                e,
                s,
                valid,
                split,
                self.cluster,
                tid,
            )
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def block(
        self,
        x: cute.Pointer,
        a: cute.Pointer,
        a_gap: Int64,
        b: cute.Pointer,
        b_gap: Int64,
        out: cute.Pointer,
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
        block, s, chunk = cute.arch.block_idx()
        smem = SmemAllocator().allocate_tensor(Float32, cute.make_layout(self.r), byte_alignment=16)
        cute.arch.griddepcontrol_wait()
        if block * self.bm < padded[0]:
            e = expert_ids[block]
            if (e >= 0) & (e < self.experts):
                # Correctness-first aligned block fallback: reuse one CTA across BM
                # sorted rows, retaining each shrink once for its complete N chunk.
                for m in range(self.bm):
                    index = block * self.bm + m
                    pair = Int32(-1)
                    if index < capacity:
                        pair = sorted_ids[index]
                    valid = (pair >= 0) & (pair < pairs)
                    if valid:
                        local = ids[pair].to(Int64)
                        valid = (local == e) & (local >= 0) & (local < self.experts)
                    row = pair if const_expr(self.down) else pair // self.top_k
                    self.shrink(
                        x,
                        a,
                        a_gap,
                        smem.iterator,
                        row,
                        e,
                        s,
                        valid,
                        Int32(0),
                        tid % 32,
                        tid // 32,
                    )
                    cute.arch.sync_threads()
                    if const_expr(self.save_z):
                        if valid & (chunk == 0) & (tid < self.r):
                            z[(Int64(pair) * self.slices + s) * self.r + tid] = smem[tid].to(
                                z.dtype
                            )
                    self.expand(
                        b,
                        b_gap,
                        out,
                        smem.iterator,
                        weights,
                        scale,
                        pair,
                        e,
                        s,
                        valid,
                        chunk,
                        self.chunks,
                        tid,
                    )
                    cute.arch.sync_threads()
        cute.arch.griddepcontrol_launch_dependents()


def compile_kernel(
    k,
    n,
    rank,
    slices,
    top_k,
    experts,
    capability,
    *,
    down=False,
    shared=False,
    cluster=1,
    block_m=0,
    chunks=1,
    z_dtype=None,
    id_dtype=torch.int32,
    block_n=64,
    stages=3,
    block_k=64,
    shrink_stages=None,
    act=False,
    stage="full",
    weighted=True,
):
    """Compile without a GPU. All routing sizes are symbolic, shape dimensions static."""
    fake = cute.runtime.make_fake_tensor
    p, t, capacity, blocks = (cute.sym_int() for _ in range(4))
    x = fake(BFloat16, (p if down else t, k), (k, 1), assumed_align=16)
    # Slice zero of each factor; later slices sit a runtime element gap away.
    a = fake(BFloat16, (experts, rank, k), (rank * k, k, 1), assumed_align=16)
    b = fake(BFloat16, (experts, n, rank), (n * rank, rank, 1), assumed_align=16)
    shrink_only = stage == "shrink" and not block_m
    out = (
        None if shrink_only else fake(BFloat16, (p, slices * n), (slices * n, 1), assumed_align=16)
    )
    activated = fake(BFloat16, (p, n), (n, 1), assumed_align=16) if act else None
    ids = (
        None
        if shared
        else fake(Int32 if id_dtype == torch.int32 else Int64, (t, top_k), (top_k, 1))
    )
    weights = (
        fake(Float32, (t, top_k), (top_k, 1))
        if down and weighted and not shared and not shrink_only
        else None
    )
    scale = fake(Float32, (1,), (1,))
    z = (
        None
        if z_dtype is None
        else fake(
            Float32 if z_dtype == torch.float32 else BFloat16,
            (p, slices, rank),
            (slices * rank, rank, 1),
            assumed_align=16,
        )
    )
    sorted_ids = fake(Int32, (capacity,), (1,)) if block_m and not shared else None
    expert_ids = fake(Int32, (blocks,), (1,)) if block_m and not shared else None
    padded = fake(Int32, (1,), (1,)) if block_m and not shared else None
    kernel_type = BlockKernel if block_m else Kernel
    kernel = kernel_type(
        k,
        n,
        rank,
        slices,
        top_k,
        experts,
        down=down,
        shared=shared,
        **(
            {
                "block_n": block_n,
                "stages": stages,
                "block_k": block_k,
                "shrink_stages": shrink_stages,
                "act": act,
                "stage": stage,
            }
            if block_m
            else {"cluster": cluster, "shrink_only": shrink_only}
        ),
        block_m=block_m,
        chunks=chunks,
        save_z=z_dtype is not None,
        weighted=weighted,
    )
    return cute.compile(
        kernel,
        x,
        a,
        Int64(0),
        b,
        Int64(0),
        out,
        ids,
        weights,
        scale,
        z,
        sorted_ids,
        expert_ids,
        padded,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        *((activated,) if act else ()),
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )
