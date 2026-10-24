# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Warp-owned query rows, paged GQA, and optional last-arrival KV reduction.

Design references (no third-party implementation copied): FlashAttention-2's
query partitioning and online softmax, https://arxiv.org/abs/2307.08691;
FlashInfer's packed GQA row order, https://docs.flashinfer.ai/api/attention.html.
Public instruction contracts: https://docs.nvidia.com/cuda/parallel-thread-execution/.
The repository's _draft/cute provides the CuTe MMA/ldmatrix layout pattern;
_decode_fp8 provides the cumulative last-arrival publication pattern.

D=64/128 retain Q in registers and use STAGES K slots plus one V tile, issuing
V_j after QK_j. D=256 retains Q in shared memory and uses STAGES-1 K slots
plus an independent V tile, issuing V_j before QK_j. The V and future-K groups
are separate: wait_group(1) permits K_{j+STAGES-1} to overlap softmax/PV.
A CTA barrier protects every shared slot reuse. Warps never split a query
row's keys within a CTA and never reduce with other warps.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.utils import SmemAllocator


@cute.jit
def pointer(base, offset, align: cutlass.Constexpr = 16):
    return cute.make_ptr(
        base.dtype,
        base.toint() + Int64(offset) * (base.dtype.width // 8),
        base.memspace,
        assumed_align=align,
    )


@cute.jit
def copy16(dst, src, offset, valid):
    cute.arch.inline_ptx(
        "{ .reg .pred zero; setp.eq.u32 zero, {$r2}, 0; "
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, zero; }",
        read_only_args=[dst, pointer(src, offset).toint(), Int32(valid)],
    )


@cute.jit
def partial(work, offset):
    # Bypass noncoherent L1 after another CTA publishes through its counter.
    return cute.arch.inline_ptx(
        "ld.global.cg.f32 {$w0}, [{$r0}];",
        write_only_types=[Float32],
        read_only_args=[pointer(work, offset, 4).toint()],
    )


@cute.jit
def partial4(work, offset):
    # Each lane owns an aligned group of eight dimensions in the merge.
    return cute.arch.inline_ptx(
        "ld.global.cg.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[pointer(work, offset).toint()],
    )


class Prefill:
    def __init__(
        self,
        batch,
        qmax,
        heads,
        group,
        dim,
        page,
        dtype,
        block_m,
        block_n,
        stages,
        splits,
        min_chunk,
        kv_dense=True,
    ):
        self.batch, self.qmax, self.heads, self.group = batch, qmax, heads, group
        self.dim, self.page, self.dtype = dim, page, dtype
        self.m, self.n, self.stages = block_m, block_n, stages
        self.splits, self.minimum = splits, min_chunk
        self.kv_dense = kv_dense
        self.threads = block_m // 16 * 32
        self.tiles = (qmax * group + block_m - 1) // block_m
        self.ring = stages * block_n * dim
        self.arena = max(block_m * dim, self.ring + block_n * dim)
        self.q_elements = block_m * dim if dim == 256 else 0
        self.k_stages = stages - 1 if dim == 256 else stages
        if dim == 256:
            self.arena = self.q_elements + self.ring
        self.lse_base = batch * heads * self.tiles * splits * block_m * dim
        self.shift = 4 if dim == 256 else 3
        # A key tile starting between pages may touch one additional page.
        self.page_slots = (
            block_n // page
            if block_n % page == 0
            else min(block_n, (block_n + page - 2) // page + 1)
        )

    @cute.jit
    def layout(
        self,
        rows: cutlass.Constexpr,
        stages: cutlass.Constexpr = 1,
        trans: cutlass.Constexpr = False,
    ):
        shape, stride = (rows, self.dim, stages), (self.dim, 1, rows * self.dim)
        if const_expr(trans):
            shape, stride = (self.dim, rows, stages), (1, self.dim, rows * self.dim)
        return cute.make_composed_layout(
            cute.make_swizzle(3, 3, self.shift), 0, cute.make_layout(shape, stride=stride)
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        qp: cute.Tensor,
        kp: cute.Tensor,
        indices: cute.Tensor,
        last: cute.Tensor,
        out: cute.Tensor,
        work: cute.Tensor,
        counters: cute.Tensor,
        scale: Float32,
        v_scale: Float32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.m // 16, 1, 1),
            permutation_mnk=(self.m, 16, 16),
        )
        self.kernel(
            q.iterator,
            k,
            v,
            qp,
            kp,
            indices,
            last,
            out.iterator,
            work.iterator,
            counters.iterator,
            q.stride[0],
            q.stride[1],
            scale,
            v_scale,
            mma,
        ).launch(
            grid=(self.tiles * self.splits, self.heads, self.batch),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def stage_pages(self, ids, indices, start, first, end, slot, tid):
        # Four-byte ca copies are valid for arbitrary CSR alignment/page size.
        # Zero-fill unused entries without issuing a global index load.
        if tid < self.page_slots:
            page = first // self.page + tid
            size = Int32(0)
            if (first < end) & (page * self.page < end):
                size = Int32(4)
            cute.arch.inline_ptx(
                "cp.async.ca.shared.global [{$r0}], [{$r1}], 4, {$r2};",
                read_only_args=[
                    (ids.iterator + ids.layout((tid, slot))).toint().to(Int32),
                    pointer(indices.iterator, Int64(start) + page, 4).toint(),
                    size,
                ],
            )

    @cute.jit
    def gather(
        self,
        destination,
        cache,
        ids,
        first,
        end,
        head,
        slot,
        tid,
        value: cutlass.Constexpr = False,
        storage_slot=None,
    ):
        # Both the segment and stage strides sit above the swizzle's bits.
        # Keep a single pre-swizzled per-thread address; advance by constants.
        row0, column = tid // (self.dim // 8), tid % (self.dim // 8) * 8
        shared_slot = slot
        if const_expr(storage_slot is not None):
            shared_slot = storage_slot
        if const_expr(value):
            shared_slot = Int32(0)
        copy_rows = self.threads // (self.dim // 8)
        if const_expr(self.dim == 256 and self.n % self.page == 0 and self.page % copy_rows == 0):
            # One page lookup/address per group of copies. Keeping the outer
            # loop rolled bounds address registers alongside the large O tile.
            for page_part in cutlass.range(self.n // self.page, unroll=1):
                page = ids[page_part, slot]
                if const_expr(self.kv_dense):
                    offset = (
                        Int64(page) * cache.stride[0]
                        + (row0 * self.heads + head) * self.dim
                        + column
                    )
                    step = copy_rows * self.heads * self.dim
                else:
                    offset = (
                        Int64(page) * cache.stride[0]
                        + Int64(row0) * cache.stride[1]
                        + Int64(head) * cache.stride[2]
                        + column
                    )
                    step = copy_rows * cache.stride[1]
                for within in cutlass.range_constexpr(self.page // copy_rows):
                    row = page_part * self.page + within * copy_rows
                    copy16(
                        destination + (shared_slot * self.n * self.dim + row * self.dim) * 2,
                        cache.iterator,
                        offset + within * step,
                        first + row + row0 < end,
                    )
        else:
            for part in cutlass.range(self.n * (self.dim // 8) // self.threads, unroll=1):
                row = row0 + part * copy_rows
                key = first + row
                if const_expr(self.n % self.page == 0):
                    # Dividing pages need no runtime KV-position division.
                    page = ids[row // self.page, slot]
                    within = row % self.page
                else:
                    page = ids[key // self.page - first // self.page, slot]
                    within = key % self.page
                if const_expr(self.kv_dense):
                    offset = (
                        Int64(page) * cache.stride[0]
                        + (within * self.heads + head) * self.dim
                        + column
                    )
                else:
                    offset = (
                        Int64(page) * cache.stride[0]
                        + Int64(within) * cache.stride[1]
                        + Int64(head) * cache.stride[2]
                        + column
                    )
                copy16(
                    destination + (shared_slot * self.n * self.dim + part * self.threads * 8) * 2,
                    cache.iterator,
                    offset,
                    key < end,
                )

    @cute.jit
    def softmax(
        self,
        score,
        acc,
        maxima,
        sums,
        first,
        end,
        length,
        queries,
        tile,
        tid,
        scale,
        diagonal: cutlass.Constexpr,
    ):
        lane, warp = tid % 32, tid // 32
        for rr in cutlass.range_constexpr(2):
            maximum = Float32(-float("inf"))
            for col in cutlass.range_constexpr(self.n // 8):
                for pair in cutlass.range_constexpr(2):
                    value = score[rr * 2 + pair, 0, col]
                    if const_expr(diagonal):
                        packed = tile * self.m + warp * 16 + lane // 4 + rr * 8
                        position = length - queries + packed // self.group
                        key = first + col * 8 + lane % 4 * 2 + pair
                        # Padded query rows have zero Q and are never stored;
                        # they need no score mask, even in a partial query CTA.
                        if (key >= end) | (key > position):
                            value = Float32(-float("inf"))
                    score[rr * 2 + pair, 0, col] = value
                    maximum = cute.arch.fmax(maximum, value)
            maximum = cute.arch.fmax(maximum, cute.arch.shuffle_sync_bfly(maximum, 1))
            maximum = cute.arch.fmax(maximum, cute.arch.shuffle_sync_bfly(maximum, 2))
            maximum = cute.arch.fmax(maximum, maxima[rr])
            safe = maximum
            if const_expr(diagonal):
                if maximum == -float("inf"):
                    safe = Float32(0)
            magnitude = abs(scale)
            factor = Float32(1)
            if scale != 0:
                factor = cute.math.exp2((maxima[rr] - safe) * magnitude, approx=True, ftz=True)
            bias = -safe * magnitude
            total = Float32(0)
            for col in cutlass.range_constexpr(self.n // 8):
                for pair in cutlass.range_constexpr(2):
                    value = score[rr * 2 + pair, 0, col]
                    exponent = cute.arch.inline_ptx(
                        "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                        write_only_types=[Float32],
                        read_only_args=[value, magnitude, bias],
                    )
                    if const_expr(diagonal):
                        if value == -float("inf"):
                            exponent = Float32(-float("inf"))
                    value = cute.math.exp2(exponent, approx=True, ftz=True)
                    score[rr * 2 + pair, 0, col] = value
                    total += value
            total += cute.arch.shuffle_sync_bfly(total, 1)
            total += cute.arch.shuffle_sync_bfly(total, 2)
            sums[rr] = sums[rr] * factor + total
            maxima[rr] = maximum
            # The predicate is shared by the four lanes owning a row. Avoid
            # rescaling 64/128 output registers when its maximum is unchanged.
            if factor != 1.0:
                for col in cutlass.range_constexpr(self.dim // 8):
                    for pair in cutlass.range_constexpr(2):
                        acc[rr * 2 + pair, 0, col] *= factor

    @cute.jit
    def store_vectors(self, staged, out, qstart, queries, tile, head, tid):
        values = cute.make_rmem_tensor((8,), self.dtype)
        for part in cutlass.range(self.m * self.dim // (8 * self.threads), unroll=1):
            element = (tid + part * self.threads) * 8
            packed = tile * self.m + element // self.dim
            if packed < queries * self.group:
                source = cute.make_tensor(pointer(staged.iterator, element), cute.make_layout(8))
                cute.copy(
                    cute.make_copy_atom(cute.nvgpu.CopyS2ROp(), self.dtype, num_bits_per_copy=128),
                    source,
                    values,
                )
                offset = (
                    (Int64(qstart) + packed // self.group) * self.heads * self.group
                    + head * self.group
                    + packed % self.group
                ) * self.dim + element % self.dim
                target = cute.make_tensor(pointer(out, offset), cute.make_layout(8))
                cute.copy(
                    cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=128),
                    values,
                    target,
                )

    @cute.jit
    def merge_weights(self, work, owner, count, tid):
        # The winner owns every partial now. Reuse the LSE workspace for
        # normalized weights, once per row rather than once per output vector.
        if tid < self.m:
            base = self.lse_base + Int64(owner) * self.splits * self.m + tid
            maximum = Float32(-float("inf"))
            for split in range(count):
                maximum = cute.arch.fmax(maximum, partial(work, base + split * self.m))
            safe = Float32(0)
            if maximum != -float("inf"):
                safe = maximum
            total = Float32(0)
            for split in range(count):
                total += cute.math.exp2(
                    partial(work, base + split * self.m) - safe, approx=True, ftz=True
                )
            inverse = Float32(0)
            if total > 0:
                inverse = 1.0 / total
            for split in range(count):
                weight = cute.math.exp2(
                    partial(work, base + split * self.m) - safe, approx=True, ftz=True
                )
                work[base + split * self.m] = weight * inverse
        cute.arch.barrier()

    @cute.jit
    def finish(
        self,
        work,
        counters,
        winner,
        staged,
        out,
        owner,
        count,
        qstart,
        queries,
        tile,
        head,
        tid,
        v_scale,
    ):
        cute.arch.inline_ptx("membar.gl;")
        cute.arch.barrier()
        counter = pointer(counters, owner, 4)
        if tid == 0:
            arrived = cute.arch.inline_ptx(
                "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], 1;",
                write_only_types=[Int32],
                read_only_args=[counter.toint()],
            )
            winner[0] = Int32(arrived == count - 1)
        cute.arch.barrier()
        if winner[0] != 0:
            if const_expr(self.dim == 256):
                self.merge_weights(work, owner, count, tid)
            # All active splits publish even when a row has no visible keys.
            # Stable LSE weighting handles all-empty rows without inf-inf/NaN.
            for part in cutlass.range(self.m * self.dim // (8 * self.threads), unroll=1):
                element = (tid + part * self.threads) * 8
                row, column = element // self.dim, element % self.dim
                base = Int64(owner) * self.splits * self.m + row
                safe = Float32(0)
                if const_expr(self.dim != 256):
                    maximum = Float32(-float("inf"))
                    for split in range(count):
                        maximum = cute.arch.fmax(
                            maximum, partial(work, self.lse_base + base + split * self.m)
                        )
                    if maximum != -float("inf"):
                        safe = maximum
                result = cute.make_rmem_tensor((8,), Float32)
                result.fill(0)
                total = Float32(0)
                for split in range(count):
                    weight = partial(work, self.lse_base + base + split * self.m)
                    if const_expr(self.dim != 256):
                        weight = cute.math.exp2(weight - safe, approx=True, ftz=True)
                        total += weight
                    for vector in cutlass.range_constexpr(2):
                        values = partial4(
                            work, (base + split * self.m) * self.dim + column + vector * 4
                        )
                        for d in cutlass.range_constexpr(4):
                            result[vector * 4 + d] += weight * values[d]
                if const_expr(self.dim == 256):
                    packed = tile * self.m + row
                    if packed < queries * self.group:
                        values = cute.make_rmem_tensor((8,), self.dtype)
                        for d in cutlass.range_constexpr(8):
                            values[d] = (result[d] * v_scale).to(self.dtype)
                        offset = (
                            (Int64(qstart) + packed // self.group) * self.heads * self.group
                            + head * self.group
                            + packed % self.group
                        ) * self.dim + column
                        target = cute.make_tensor(pointer(out, offset), cute.make_layout(8))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=128
                            ),
                            values,
                            target,
                        )
                else:
                    inverse = Float32(0)
                    if total > 0:
                        inverse = 1.0 / total
                    for d in cutlass.range_constexpr(8):
                        staged[row, column + d] = (result[d] * inverse * v_scale).to(self.dtype)
            if const_expr(self.dim != 256):
                cute.arch.barrier()
                self.store_vectors(staged, out, qstart, queries, tile, head, tid)
            cute.arch.barrier()
            if tid == 0:
                cute.arch.inline_ptx(
                    "st.relaxed.gpu.global.u32 [{$r0}], 0;", read_only_args=[counter.toint()]
                )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        k: cute.Tensor,
        v: cute.Tensor,
        qp: cute.Tensor,
        kp: cute.Tensor,
        indices: cute.Tensor,
        last: cute.Tensor,
        out: cute.Pointer,
        work: cute.Pointer,
        counters: cute.Pointer,
        qstride: Int64,
        qheadstride: Int64,
        scale: Float32,
        v_scale: Float32,
        mma: cute.TiledMma,
    ):
        cute.arch.griddepcontrol_wait()
        # Dependents still wait for completion, including the last-arrival merge.
        cute.arch.griddepcontrol_launch_dependents()
        x, head, b = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        tile, split = x // self.splits, x % self.splits
        qstart, queries = qp[b], qp[b + 1] - qp[b]
        start, pages = kp[b], kp[b + 1] - kp[b]
        length = Int64(0)
        if pages > 0:
            length = Int64(pages - 1) * self.page + last[b]
        if const_expr(self.splits == 1):
            # The direct path needs neither split-count division nor counters.
            chunk, count = length, Int32(1)
        else:
            chunk = (
                (max((length + self.splits - 1) // self.splits, Int64(self.minimum)) + self.n - 1)
                // self.n
            ) * self.n
            count = max(Int32(1), Int32((length + chunk - 1) // chunk))
        if (tile * self.m < queries * self.group) & (split < count):
            smem = SmemAllocator()
            arena = smem.allocate_tensor(
                self.dtype, cute.make_layout(self.arena), byte_alignment=128
            )
            ids = smem.allocate_tensor(
                Int32,
                cute.make_layout((self.n, self.stages), stride=(1, self.n)),
                byte_alignment=16,
            )
            winner = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
            sq = cute.make_tensor(arena.iterator, self.layout(self.m))[None, None, 0]
            sk = cute.make_tensor(
                arena.iterator + self.q_elements, self.layout(self.n, self.k_stages)
            )
            v_offset = self.q_elements + self.k_stages * self.n * self.dim
            v_stages = 1
            sv = cute.make_tensor(arena.iterator + v_offset, self.layout(self.n, v_stages))
            vt = cute.make_tensor(sv.iterator, self.layout(self.n, v_stages, trans=True))
            copy_offset = sk.layout((tid // (self.dim // 8), tid % (self.dim // 8) * 8, 0))
            k_copy = (sk.iterator + copy_offset).toint().to(Int32)
            v_copy = (sv.iterator + copy_offset).toint().to(Int32)
            staged = cute.make_tensor(
                arena.iterator, cute.make_layout((self.m, self.dim), stride=(self.dim, 1))
            )
            for part in cutlass.range(self.m * self.dim // (8 * self.threads), unroll=1):
                segment = tid + part * self.threads
                row, column = segment // (self.dim // 8), segment % (self.dim // 8) * 8
                packed = tile * self.m + row
                offset = (
                    (Int64(qstart) + packed // self.group) * qstride
                    + (head * self.group + packed % self.group) * qheadstride
                    + column
                )
                copy16(
                    (sq.iterator + sq.layout((row, column))).toint().to(Int32),
                    q,
                    offset,
                    packed < queries * self.group,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            # Fold a negative scale into Q once per CTA, before any QK work.
            # Negation is exact in either input dtype; positive/zero scales
            # retain the original Q and positive-path softmax arithmetic.
            # Each thread owns the same shared vectors it just copied, so
            # its wait_group suffices before negation. The existing barrier
            # below publishes both copies and negations to every Q reader.
            cute.arch.inline_ptx("// PREFILL_Q_SIGN_BEGIN")
            if scale < 0:
                for part in cutlass.range(self.m * self.dim // (8 * self.threads), unroll=1):
                    segment = tid + part * self.threads
                    row, column = segment // (self.dim // 8), segment % (self.dim // 8) * 8
                    for element in cutlass.range_constexpr(8):
                        sq[row, column + element] = -sq[row, column + element]
            cute.arch.inline_ptx("// PREFILL_Q_SIGN_END")
            cute.arch.barrier()
            the = mma.get_slice(tid)
            aq = the.make_fragment_A(the.partition_A(sq))
            bk = the.make_fragment_B(the.partition_B(sk[None, None, 0]))
            bv = the.make_fragment_B(the.partition_B(vt[None, None, 0]))
            load = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
            )
            load_t = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
            )
            cq = cute.make_tiled_copy_A(load, mma).get_slice(tid)
            ck = cute.make_tiled_copy_B(load, mma).get_slice(tid)
            cv = cute.make_tiled_copy_B(load_t, mma).get_slice(tid)
            q_source = cq.partition_S(sq)
            k_source = ck.partition_S(sk[None, None, 0])
            v_source = cv.partition_S(vt[None, None, 0])
            if const_expr(self.dim != 256):
                cute.copy(cq, q_source, cq.retile(aq))
            cute.arch.barrier()
            acc = cute.make_rmem_tensor(the.partition_shape_C((self.m, self.dim)), Float32)
            acc.fill(0)
            maxima, sums = (
                cute.make_rmem_tensor((2,), Float32),
                cute.make_rmem_tensor((2,), Float32),
            )
            maxima.fill(-float("inf"))
            sums.fill(0)
            begin = Int64(split) * chunk
            # No key above the last live query position can contribute to this CTA.
            last_row = min((tile + 1) * self.m, queries * self.group) - 1
            end = max(
                begin,
                min(length, (split + 1) * chunk, length - queries + last_row // self.group + 1),
            )
            if begin < end:
                for stage in cutlass.range_constexpr(self.stages):
                    self.stage_pages(ids, indices, start, begin + stage * self.n, end, stage, tid)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                for stage in cutlass.range_constexpr(self.stages - 1):
                    self.gather(k_copy, k, ids, begin + stage * self.n, end, head, stage, tid)
                    cute.arch.cp_async_commit_group()
                slot = Int32(0)
                k_slot = Int32(0)
                for first in range(begin, end, self.n):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.barrier()
                    # Add a stage stride to the already swizzled thread view,
                    # rather than recomposing the swizzle with a dynamic slot.
                    k_stage = cute.make_tensor(
                        pointer(
                            k_source.iterator,
                            (k_slot if const_expr(self.dim == 256) else slot) * self.n * self.dim,
                        ),
                        k_source.layout,
                    )
                    v_stage = cute.make_tensor(v_source.iterator, v_source.layout)
                    score = cute.make_rmem_tensor(the.partition_shape_C((self.m, self.n)), Float32)
                    score.fill(0)
                    if const_expr(self.dim == 256):
                        # V has its own tile: its copies overlap QK, without
                        # overwriting K or keeping a full Q fragment live.
                        self.gather(v_copy, v, ids, first, end, head, slot, tid, value=True)
                        # Q remains in shared memory. Bound QK unrolling to
                        # four chunks: full unrolling hoists enough fragments
                        # to spill with N=64 in the external assembler.
                        for inner in cutlass.range(self.dim // 16, unroll=4):
                            cute.copy(
                                cq,
                                q_source[None, None, inner],
                                cq.retile(aq)[None, None, 0],
                            )
                            cute.copy(
                                ck,
                                k_stage[None, None, inner],
                                ck.retile(bk)[None, None, 0],
                            )
                            cute.gemm(mma, score, aq[None, None, 0], bk[None, None, 0], score)
                    else:
                        for inner in cutlass.range_constexpr(self.dim // 16):
                            cute.copy(
                                ck,
                                k_stage[None, None, inner],
                                ck.retile(bk)[None, None, 0],
                            )
                            cute.gemm(mma, score, aq[None, None, inner], bk[None, None, 0], score)
                    # Current page IDs survive until V_j is issued. Retire QK
                    # before reusing its K slot on D=256 or its ID slot on all D.
                    if const_expr(self.dim != 256):
                        self.gather(v_copy, v, ids, first, end, head, slot, tid, value=True)
                    # Every warp has consumed current IDs. Refill that slot a
                    # whole iteration before its next K gather. IDs share V's
                    # completion group, so wait_group(1) also publishes them.
                    cute.arch.barrier()
                    self.stage_pages(
                        ids, indices, start, first + self.stages * self.n, end, slot, tid
                    )
                    cute.arch.cp_async_commit_group()
                    future = first + (self.stages - 1) * self.n
                    next_slot = (slot + self.stages - 1) % self.stages
                    if const_expr(self.dim == 256):
                        # The barrier above also retires every warp's QK read
                        # before this consumed K slot is refilled.
                        self.gather(
                            k_copy, k, ids, future, end, head, next_slot, tid, storage_slot=k_slot
                        )
                    else:
                        self.gather(k_copy, k, ids, future, end, head, next_slot, tid)
                    cute.arch.cp_async_commit_group()
                    if (first + self.n <= end) & (
                        first + self.n - 1 <= length - queries + tile * self.m // self.group
                    ):
                        self.softmax(
                            score,
                            acc,
                            maxima,
                            sums,
                            first,
                            end,
                            length,
                            queries,
                            tile,
                            tid,
                            scale * 1.4426950408889634,
                            diagonal=False,
                        )
                    else:
                        self.softmax(
                            score,
                            acc,
                            maxima,
                            sums,
                            first,
                            end,
                            length,
                            queries,
                            tile,
                            tid,
                            scale * 1.4426950408889634,
                            diagonal=True,
                        )
                    prob = cute.make_rmem_tensor((8, 1, self.n // 16), self.dtype)
                    for inner in cutlass.range_constexpr(self.n // 16):
                        for value in cutlass.range_constexpr(8):
                            prob[value, 0, inner] = score[value % 4, 0, inner * 2 + value // 4].to(
                                self.dtype
                            )
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.barrier()
                    for inner in cutlass.range_constexpr(self.n // 16):
                        if const_expr(self.dim == 256):
                            # Issue V loads in 16-column slices while keeping
                            # every accumulator slice statically indexed.
                            outputs = cute.logical_divide(acc, (None, None, 2))
                            values = cute.logical_divide(bv, (None, 2, None))
                            for column in cutlass.range_constexpr(self.dim // 16):
                                cute.copy(
                                    cv,
                                    v_stage[None, column, inner],
                                    cv.retile(bv)[None, 0, 0],
                                )
                                result = outputs[None, None, (None, column)]
                                cute.gemm(
                                    mma,
                                    result,
                                    prob[None, None, inner],
                                    values[None, (None, 0), 0],
                                    result,
                                )
                        else:
                            cute.copy(
                                cv,
                                v_stage[None, None, inner],
                                cv.retile(bv)[None, None, 0],
                            )
                            cute.gemm(mma, acc, prob[None, None, inner], bv[None, None, 0], acc)
                    cute.arch.barrier()
                    slot = (slot + 1) % self.stages
                    if const_expr(self.dim == 256):
                        k_slot = (k_slot + 1) % self.k_stages
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
            owner = (Int64(b) * self.heads + head) * self.tiles + tile
            base = (owner * self.splits + split) * self.m
            for rr in cutlass.range_constexpr(2):
                row = tid // 32 * 16 + tid % 32 // 4 + rr * 8
                inverse, lse = Float32(0), Float32(-float("inf"))
                if sums[rr] > 0:
                    inverse = 1.0 / sums[rr]
                    lse = maxima[rr] * abs(scale * 1.4426950408889634) + cute.math.log2(sums[rr])
                if const_expr(self.splits > 1):
                    if count > 1 and tid % 4 == 0:
                        work[self.lse_base + base + row] = lse
                for col in cutlass.range_constexpr(self.dim // 8):
                    for pair in cutlass.range_constexpr(2):
                        column = col * 8 + tid % 4 * 2 + pair
                        value = acc[rr * 2 + pair, 0, col] * inverse
                        if count == 1:
                            staged[row, column] = (value * v_scale).to(self.dtype)
                        else:
                            if const_expr(self.splits > 1):
                                work[(base + row) * self.dim + column] = value
            if count == 1:
                cute.arch.barrier()
                self.store_vectors(staged, out, qstart, queries, tile, head, tid)
            else:
                if const_expr(self.splits > 1):
                    self.finish(
                        work,
                        counters,
                        winner,
                        staged,
                        out,
                        owner,
                        count,
                        qstart,
                        queries,
                        tile,
                        head,
                        tid,
                        v_scale,
                    )
