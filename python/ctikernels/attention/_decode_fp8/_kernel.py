# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent FP8 fragment-swizzled decode, using public PTX and CuTe 4.8.

Designs studied: FlashInfer FP8 prefill fragment swizzles
(https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/attention),
FlashAttention-2 register reuse (https://arxiv.org/abs/2307.08691), and this
repository's attention/_draft/cute ring and last-arrival reduction. No external
implementation is copied. Instruction contracts: NVIDIA's public PTX ISA,
https://docs.nvidia.com/cuda/parallel-thread-execution/ (cvt, prmt, ldmatrix, mma).

Each warp owns a contiguous FP8 K/V ring, reused for its O/LSE reduction.
Distributed merge uses a cooperative grid and GPU acquire/release publication.
Only public PTX instructions are used; see the offline compilation receipt.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int32, Int64, Uint8, Uint16, Uint32, Uint64, const_expr
from cutlass.utils import SmemAllocator

from . import _warp_bytes


@cute.jit
def byte_pointer(base, offset):
    return cute.make_ptr(Uint8, base.toint() + Int64(offset), base.memspace, assumed_align=16)


@cute.jit
def copy16(dst, src, valid):
    size = Int32(0)
    if valid:
        size = Int32(16)
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[dst.toint().to(Int32), src.toint(), size],
    )


@cute.jit
def copy4(dst, src, valid):
    size = Int32(0)
    if valid:
        size = Int32(4)
    # Public PTX: 4-byte cp.async requires .ca (.cg only supports 16).
    cute.arch.inline_ptx(
        "cp.async.ca.shared.global [{$r0}], [{$r1}], 4, {$r2};",
        read_only_args=[dst.toint().to(Int32), src.toint(), size],
    )


@cute.jit
def fp8_pair(value):
    return cute.arch.inline_ptx(
        "cvt.rn.f16x2.e4m3x2 {$w0}, {$r0};",
        write_only_types=[Uint32],
        read_only_args=[value.to(Uint16)],
    )


@cute.jit
def fp8_bf16_pair(value):
    # Embed e4m3 magnitude at bf16 bits 10:4, sign at 15: value * 2^-120.
    # BF16 subnormals preserve e4m3 subnormals; exact rescaling preserves -0.
    return cute.arch.inline_ptx(
        "{ .reg .b32 bits, factor, zero; "
        "prmt.b32 bits, {$r0}, 0, 0x9180; "
        "shl.b32 bits, bits, 4; and.b32 bits, bits, 0x87f087f0; "
        "mov.b32 factor, 0x7b807b80; mov.b32 zero, 0x80008000; "
        "fma.rn.bf16x2 {$w0}, bits, factor, zero; }",
        write_only_types=[Uint32],
        read_only_args=[value],
    )


@cute.jit
def bf16_fp16_exact(value, previous=1):
    # BF16's largest FP16-finite value is 0x477f (65280), and the
    # smallest nonzero FP16 value is 0x3380 (2^-24). Between these bounds,
    # exponent 103 needs seven zero mantissa bits; exponent 110+ needs none.
    # Public PTX clamps shifts >=32, including unsigned underflow below 103.
    # Thread the AND through PTX so the compiler cannot reassociate it into
    # a wide tree of simultaneously live predicates while packing Q.
    return cute.arch.inline_ptx(
        "{ .reg .b32 mag, exponent, mask; .reg .pred exact; "
        "and.b32 mag, {$r0}, 0x7fff; "
        "shr.u32 exponent, mag, 7; sub.u32 exponent, exponent, 103; "
        "shr.u32 mask, 127, exponent; and.b32 mask, mask, mag; "
        "setp.eq.u32 exact, mask, 0; "
        "setp.ge.and.u32 exact, mag, 0x3380, exact; "
        "setp.le.and.u32 exact, mag, 0x477f, exact; "
        "setp.eq.or.u32 exact, mag, 0, exact; "
        "setp.ne.and.u32 exact, {$r1}, 0, exact; "
        "selp.u32 {$w0}, 1, 0, exact; }",
        write_only_types=[Int32],
        read_only_args=[value.bitcast(Uint16).to(Uint32), Int32(previous)],
    )


@cute.jit
def half_pair(lo, hi):
    # Keep persistent Q as packed b32, rather than two separately live b16
    # values that the compiler might only pack at each MMA in the key loop.
    return cute.arch.inline_ptx(
        "mov.b32 {$w0}, {{$r0}, {$r1}};",
        write_only_types=[Uint32],
        read_only_args=[lo.bitcast(Uint16), hi.bitcast(Uint16)],
    )


@cute.jit
def v_pair(word, odd: cutlass.Constexpr):
    # Pack bytes (0,2) or (1,3): adjacent keys at one even/odd dimension.
    selected = cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, 0, {$r1};",
        write_only_types=[Uint32],
        read_only_args=[word, Int32(0x4431 if odd else 0x4420)],
    )
    return fp8_pair(selected)


@cute.jit
def load_partial(work, index):
    # Bypass L1 across independently scheduled CTAs and successive graph replays.
    return cute.arch.inline_ptx(
        "ld.volatile.global.f32 {$w0}, [{$r0}];",
        write_only_types=[Float32],
        read_only_args=[(work + index).toint()],
    )


class Decode:
    def __init__(
        self,
        batch,
        heads,
        group,
        dim,
        page,
        dtype,
        splits,
        min_chunk,
        warps,
        keys_per_warp,
        stages,
        merge="distributed",
        kv_dense=True,
    ):
        self.batch, self.heads, self.group, self.dim = batch, heads, group, dim
        self.page, self.dtype, self.splits, self.minimum = page, dtype, splits, min_chunk
        self.warps, self.nk, self.stages = warps, keys_per_warp, stages
        self.threads, self.kt, self.pitch = warps * 32, warps * keys_per_warp, max(128, dim)
        self.distributed = merge == "distributed"
        self.kv_dense = kv_dense
        self.k_bytes = stages * self.nk * self.pitch
        self.warp_bytes = _warp_bytes(dim, keys_per_warp, stages)
        self.arena_bytes = max(
            warps * self.warp_bytes, splits * group * 4 if self.distributed else 0
        )
        # Two bounded page-id banks. Leave room for an unaligned first page;
        # every piece ends on a CTA key-tile boundary, except the chunk tail.
        self.page_cap = max(128, (self.kt + page - 2) // page + 1)
        self.piece_keys = max(1, (self.page_cap - 1) * page // self.kt) * self.kt
        self.metadata_bytes = 2 * self.page_cap * 4
        # One uniform branch guards all element comparisons in a score row.
        # Keep the branch in public PTX: ordinary DSL ifs can be if-converted,
        # speculating every key comparison even on a full stage.
        mask = ["{", ".reg .pred tail, invalid;"]
        for item in range(self.nk // 4):
            mask.append(f"mov.b32 {{$w{item}}}, {{$r{item + 3}}};")
        mask += ["setp.gt.u64 tail, {$r0}, {$r1};", "@!tail bra.uni MASK_DONE;"]
        for item in range(self.nk // 4):
            offset = item // 2 * 8 + item % 2
            mask += [
                "{ .reg .u64 key;",
                f"add.u64 key, {{$r2}}, {offset};",
                "setp.ge.u64 invalid, key, {$r1};",
                f"@invalid mov.b32 {{$w{item}}}, 0xff800000;",
                "}",
            ]
        self.mask_ptx = "\n".join(mask + ["MASK_DONE:", "}"])
        self.lse_base = batch * heads * splits * group * dim

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        last: cute.Tensor,
        out: cute.Tensor,
        work: cute.Tensor,
        counters: cute.Tensor,
        sm_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, (16, 8, 16)))
        qk_mma = mma
        if const_expr(self.dtype == cutlass.BFloat16):
            qk_mma = cute.make_tiled_mma(
                cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16))
            )
        self.kernel(
            q,
            k.iterator if const_expr(self.kv_dense) else k,
            v.iterator if const_expr(self.kv_dense) else v,
            ptr,
            idx,
            last,
            out.iterator,
            work.iterator,
            counters.iterator,
            sm_scale,
            k_scale,
            v_scale,
            mma,
            qk_mma,
        ).launch(
            grid=(self.splits, self.heads, self.batch),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
            cooperative=self.distributed and self.splits > 1,
        )

    @cute.jit
    def offset(self, row, column):
        # Byte columns 4..6 xor row bits 0..2; a 16-byte sector stays intact.
        return row * self.pitch + (column ^ ((row % 8) * 16))

    @cute.jit
    def q_offset(self, byte):
        # Q temporarily occupies the final K/V slots of warp zero. Priming
        # only fills stages-1 slots; a CTA barrier retires Q before slot reuse.
        size = self.nk * self.pitch
        return (self.stages - 1) * size + byte % size + byte // size * self.k_bytes

    @cute.jit
    def stage_pages(self, page_ids, idx, start, first, end, tid, bank):
        base = first // self.page
        pages = (end + self.page - 1) // self.page - base
        for part in cutlass.range_constexpr((self.page_cap + self.threads - 1) // self.threads):
            item = tid + part * self.threads
            if item < self.page_cap:
                copy4(
                    page_ids.iterator + bank * self.page_cap + item,
                    idx.iterator + start + base + item,
                    item < pages,
                )
        # The caller commits this together with the first K/V group. The next
        # bank is ready after draining that piece and the boundary CTA barrier.

    @cute.jit
    def gather_rows(
        self,
        arena,
        k,
        v,
        page_ids,
        page_base,
        bank,
        first,
        end,
        head,
        tid,
        slot,
        tail: cutlass.Constexpr,
    ):
        warp, lane = tid // 32, tid % 32
        for part in cutlass.range_constexpr(self.nk * (self.dim // 16) // 32):
            segment = lane + part * 32
            local_row, column = segment // (self.dim // 16), segment % (self.dim // 16) * 16
            row = warp * self.nk + local_row
            key = (first + row).to(Uint64)
            valid = True
            if const_expr(tail):
                valid = key < end
            page = Int32(0)
            if valid:
                # Shared broadcast supplies all columns; no global load or
                # shuffle lies on the per-stage address dependency chain.
                page = page_ids[bank * self.page_cap + key // self.page - page_base]
            if const_expr(self.kv_dense):
                source = (
                    (Int64(page) * self.page + key % self.page) * self.heads + head
                ) * self.dim + column
                k_source, v_source = byte_pointer(k, source), byte_pointer(v, source)
            else:
                k_source = byte_pointer(
                    k.iterator,
                    Int64(page) * k.stride[0]
                    + Int64(key % self.page) * k.stride[1]
                    + Int64(head) * k.stride[2]
                    + column,
                )
                v_source = byte_pointer(
                    v.iterator,
                    Int64(page) * v.stride[0]
                    + Int64(key % self.page) * v.stride[1]
                    + Int64(head) * v.stride[2]
                    + column,
                )
            target = (
                warp * self.warp_bytes
                + slot * self.nk * self.pitch
                + self.offset(local_row, column)
            )
            copy16(arena.iterator + target, k_source, valid)
            copy16(arena.iterator + self.k_bytes + target, v_source, valid)

    @cute.jit
    def gather(self, arena, k, v, page_ids, page_base, bank, first, end, head, tid, slot):
        if first + self.kt <= end:
            cute.arch.inline_ptx("// FP8_GATHER_FULL_BEGIN")
            self.gather_rows(
                arena, k, v, page_ids, page_base, bank, first, end, head, tid, slot, False
            )
            cute.arch.inline_ptx("// FP8_GATHER_FULL_END")
        elif first < end:
            cute.arch.inline_ptx("// FP8_GATHER_TAIL_BEGIN")
            self.gather_rows(
                arena, k, v, page_ids, page_base, bank, first, end, head, tid, slot, True
            )
            cute.arch.inline_ptx("// FP8_GATHER_TAIL_END")
        # Empty lookahead groups keep wait_group's age invariant at the tail.
        cute.arch.cp_async_commit_group()

    @cute.jit
    def prime(self, arena, k, v, page_ids, bank, first, end, head, tid):
        for stage in cutlass.range_constexpr(self.stages - 1):
            self.gather(
                arena,
                k,
                v,
                page_ids,
                first // self.page,
                bank,
                first + stage * self.kt,
                end,
                head,
                tid,
                stage,
            )

    @cute.jit
    def matrix(self, arena, slot, warp, lane, row, column, is_v: cutlass.Constexpr):
        # x1 uses the row addresses in lanes 0..7. Tile columns and row blocks
        # are constexpr; only slot/warp/lane arithmetic remains in the key loop.
        base = warp * self.warp_bytes + slot * self.nk * self.pitch
        if const_expr(is_v):
            base += self.k_bytes
        address = (arena.iterator + base + self.offset(row + lane % 8, column)).toint().to(Int32)
        if const_expr(is_v):
            word = cute.arch.inline_ptx(
                "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {{$w0}}, [{$r0}];",
                write_only_types=[Uint32],
                read_only_args=[address],
            )
        else:
            word = cute.arch.inline_ptx(
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {{$w0}}, [{$r0}];",
                write_only_types=[Uint32],
                read_only_args=[address],
            )
        return word

    @cute.jit
    def qk(self, arena, qwords, qk_mma, score, slot, warp, lane, bf16: cutlass.Constexpr):
        qfrag = cute.recast_tensor(qwords, cutlass.BFloat16 if bf16 else Float16)
        bk = cute.make_rmem_tensor((4, 1), qfrag.element_type)
        for inner in cutlass.range_constexpr(self.dim // 16):
            for n in cutlass.range_constexpr(self.nk // 8):
                word = self.matrix(arena, slot, warp, lane, n * 8, inner * 16, False)
                words = cute.recast_tensor(bk, Uint32)
                if const_expr(qfrag.element_type == cutlass.BFloat16):
                    words[0], words[1] = fp8_bf16_pair(word), fp8_bf16_pair(word >> 16)
                else:
                    words[0], words[1] = fp8_pair(word), fp8_pair(word >> 16)
                # B's k slots are [dim0,1,4,5,...,dim2,3,6,7,...].
                # Q was packed with this identical permutation once.
                cute.gemm(
                    qk_mma,
                    cute.local_tile(score, (4, 1, 1), (0, 0, n)),
                    qfrag[None, None, inner],
                    bk,
                    cute.local_tile(score, (4, 1, 1), (0, 0, n)),
                )
                if const_expr(self.group <= 8):
                    # MMA still produces 16 rows. Break the recurrence for
                    # padding rows so their accumulators need not stay live.
                    score[2, 0, n], score[3, 0, n] = Float32(0), Float32(0)

    @cute.jit
    def walk(
        self,
        arena,
        qwords,
        mma,
        qk_mma,
        k,
        v,
        page_ids,
        bank,
        begin,
        end,
        head,
        tid,
        acc,
        maxima,
        sums,
        scale,
        bf16: cutlass.Constexpr,
    ):
        warp, lane = tid // 32, tid % 32
        multiple = end - begin > self.kt
        if multiple:
            cute.arch.cp_async_wait_group(self.stages - 2)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        slot = Int32(0)
        # Each warp stops before its first empty block. Draining below still
        # retires its primed/lookahead copies, even when it runs no MMAs.
        warp_end = end - min(Uint64(warp * self.nk), end - begin)
        for first in range(begin, warp_end, self.kt):
            cute.arch.inline_ptx("// FP8_STAGE_BEGIN")
            if multiple:
                self.gather(
                    arena,
                    k,
                    v,
                    page_ids,
                    begin // self.page,
                    bank,
                    first + (self.stages - 1) * self.kt,
                    end,
                    head,
                    tid,
                    (slot + self.stages - 1) % self.stages,
                )
            score = cute.make_rmem_tensor((4, 1, self.nk // 8), Float32)
            score.fill(0)
            self.qk(arena, qwords, qk_mma, score, slot, warp, lane, bf16)
            for rr in cutlass.range_constexpr((self.group + 7) // 8):
                arguments = [first + self.kt, end, first + warp * self.nk + lane % 4 * 2]
                for n in cutlass.range_constexpr(self.nk // 8):
                    for pair in cutlass.range_constexpr(2):
                        arguments.append(score[rr * 2 + pair, 0, n] * scale)
                cute.arch.inline_ptx("// FP8_MASK_BEGIN")
                masked = cute.arch.inline_ptx(
                    self.mask_ptx,
                    write_only_types=[Float32] * (self.nk // 4),
                    read_only_args=arguments,
                )
                cute.arch.inline_ptx("// FP8_MASK_END")
                maximum = Float32(-float("inf"))
                for n in cutlass.range_constexpr(self.nk // 8):
                    for pair in cutlass.range_constexpr(2):
                        value = masked[n * 2 + pair]
                        score[rr * 2 + pair, 0, n] = value
                        maximum = max(maximum, value)
                maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1))
                maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 2))
                maximum = max(maximum, maxima[rr])
                safe = Float32(0)
                if maximum != -float("inf"):
                    safe = maximum
                factor = cute.math.exp2(maxima[rr] - safe)
                total = Float32(0)
                for n in cutlass.range_constexpr(self.nk // 8):
                    for pair in cutlass.range_constexpr(2):
                        probability = cute.math.exp2(score[rr * 2 + pair, 0, n] - safe)
                        score[rr * 2 + pair, 0, n] = probability
                        total += probability
                total += cute.arch.shuffle_sync_bfly(total, 1)
                total += cute.arch.shuffle_sync_bfly(total, 2)
                sums[rr], maxima[rr] = sums[rr] * factor + total, maximum
                for n in cutlass.range_constexpr(self.dim // 8):
                    for pair in cutlass.range_constexpr(2):
                        acc[rr * 2 + pair, 0, n] *= factor
            probability = cute.make_rmem_tensor((8, 1, self.nk // 16), Float16)
            for inner in cutlass.range_constexpr(self.nk // 16):
                for value in cutlass.range_constexpr(8):
                    if const_expr(value % 4 // 2 < (self.group + 7) // 8):
                        probability[value, 0, inner] = score[
                            value % 4, 0, inner * 2 + value // 4
                        ].to(Float16)
                    else:
                        probability[value, 0, inner] = Float16(0)
            bv = cute.make_rmem_tensor((4, 1), Float16)
            for inner in cutlass.range_constexpr(self.nk // 16):
                for n in cutlass.range_constexpr(self.dim // 16):
                    packed = cute.make_rmem_tensor((2,), Uint32)
                    for half in cutlass.range_constexpr(2):
                        packed[half] = self.matrix(
                            arena, slot, warp, lane, inner * 16 + half * 8, n * 16, True
                        )
                    for odd in cutlass.range_constexpr(2):
                        words = cute.recast_tensor(bv, Uint32)
                        words[0], words[1] = v_pair(packed[0], odd), v_pair(packed[1], odd)
                        cute.gemm(
                            mma,
                            cute.local_tile(acc, (4, 1, 1), (0, 0, n * 2 + odd)),
                            probability[None, None, inner],
                            bv,
                            cute.local_tile(acc, (4, 1, 1), (0, 0, n * 2 + odd)),
                        )
                        if const_expr(self.group <= 8):
                            acc[2, 0, n * 2 + odd] = Float32(0)
                            acc[3, 0, n * 2 + odd] = Float32(0)
            if multiple:
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_warp()
                slot = (slot + 1) % self.stages
            cute.arch.inline_ptx("// FP8_STAGE_END")
        # Also retire the next page bank in the one-stage path.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()

    @cute.jit
    def walk_pieces(
        self,
        arena,
        qwords,
        mma,
        qk_mma,
        k,
        v,
        page_ids,
        idx,
        start,
        begin,
        end,
        head,
        tid,
        acc,
        maxima,
        sums,
        scale,
        bf16: cutlass.Constexpr,
    ):
        bank = Int32(0)
        for piece in range(begin, end, self.piece_keys):
            piece_end = min(piece + self.piece_keys, end)
            self.walk(
                arena,
                qwords,
                mma,
                qk_mma,
                k,
                v,
                page_ids,
                bank,
                piece,
                piece_end,
                head,
                tid,
                acc,
                maxima,
                sums,
                scale,
                bf16,
            )
            if piece_end < end:
                # All page readers and async writers retire before a
                # bank is reused. No CTA barrier occurs per key stage.
                cute.arch.barrier()
                following = min(piece_end + self.piece_keys, end)
                if following < end:
                    self.stage_pages(
                        page_ids,
                        idx,
                        start,
                        following,
                        min(following + self.piece_keys, end),
                        tid,
                        bank,
                    )
                bank = 1 - bank
                self.prime(arena, k, v, page_ids, bank, piece_end, following, head, tid)

    @cute.jit
    def reduce_warps(
        self, arena, acc, maxima, sums, out, work, b, head, split, count, tid, v_scale
    ):
        warp, lane = tid // 32, tid % 32
        partial = cute.make_tensor(
            cute.recast_ptr(arena.iterator, dtype=Float32), cute.make_layout(self.arena_bytes // 4)
        )
        stat_base = 16 * self.dim
        warp_stride = self.warp_bytes // 4
        for rr in cutlass.range_constexpr((self.group + 7) // 8):
            row = lane // 4 + rr * 8
            inverse, lse = Float32(0), Float32(-float("inf"))
            if sums[rr] > 0:
                inverse = 1.0 / sums[rr]
                lse = maxima[rr] + cute.math.log2(sums[rr])
            if lane % 4 == 0:
                partial[warp * warp_stride + stat_base + row] = lse
            for n in cutlass.range_constexpr(self.dim // 8):
                for pair in cutlass.range_constexpr(2):
                    dim = n // 2 * 16 + lane % 4 * 4 + pair * 2 + n % 2
                    partial[warp * warp_stride + row * self.dim + dim] = (
                        acc[rr * 2 + pair, 0, n] * inverse
                    )
        cute.arch.barrier()
        if warp == 0:
            for rr in cutlass.range_constexpr((self.group + 7) // 8):
                row = lane // 4 + rr * 8
                maximum = Float32(-float("inf"))
                for w in cutlass.range_constexpr(self.warps):
                    maximum = max(maximum, partial[w * warp_stride + stat_base + row])
                safe = Float32(0)
                if maximum != -float("inf"):
                    safe = maximum
                weights = cute.make_rmem_tensor((self.warps,), Float32)
                total = Float32(0)
                for w in cutlass.range_constexpr(self.warps):
                    weights[w] = cute.math.exp2(partial[w * warp_stride + stat_base + row] - safe)
                    total += weights[w]
                inverse, lse = Float32(0), Float32(-float("inf"))
                if total > 0:
                    inverse = 1.0 / total
                    lse = maximum + cute.math.log2(total)
                base = ((Int64(b) * self.heads + head) * self.splits + split) * self.group + row
                if row < self.group:
                    if count > 1 and lane % 4 == 0:
                        work[self.lse_base + base] = lse
                    for n in cutlass.range_constexpr(self.dim // 8):
                        for pair in cutlass.range_constexpr(2):
                            dim = n // 2 * 16 + lane % 4 * 4 + pair * 2 + n % 2
                            value = Float32(0)
                            for w in cutlass.range_constexpr(self.warps):
                                value += (
                                    weights[w] * partial[w * warp_stride + row * self.dim + dim]
                                )
                            value *= inverse
                            if count == 1:
                                out[
                                    ((Int64(b) * self.heads + head) * self.group + row) * self.dim
                                    + dim
                                ] = (value * v_scale).to(self.dtype)
                            else:
                                work[base * self.dim + dim] = value

    @cute.jit
    def finish(self, work, counters, winner, out, b, head, count, tid, v_scale):
        # All producer threads fence their own stores, then thread zero publishes
        # the CTA. acq_rel GPU atomics form a cumulative chain of prior arrivals.
        cute.arch.inline_ptx("membar.gl;")
        cute.arch.barrier()
        counter = counters + b * self.heads + head
        if tid == 0:
            old = cute.arch.inline_ptx(
                "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], 1;",
                write_only_types=[Int32],
                read_only_args=[counter.toint()],
            )
            winner[0] = Int32(old == count - 1)
        cute.arch.barrier()
        if winner[0] != 0:
            for element in range(tid, self.group * self.dim, self.threads):
                row, dim = element // self.dim, element % self.dim
                base = (Int64(b) * self.heads + head) * self.splits * self.group + row
                maximum = Float32(-float("inf"))
                for s in range(count):
                    maximum = max(
                        maximum, load_partial(work, self.lse_base + base + s * self.group)
                    )
                total, result = Float32(0), Float32(0)
                for s in range(count):
                    weight = cute.math.exp2(
                        load_partial(work, self.lse_base + base + s * self.group) - maximum
                    )
                    total += weight
                    result += weight * load_partial(work, (base + s * self.group) * self.dim + dim)
                out[(Int64(b) * self.heads + head) * self.group * self.dim + element] = (
                    result / total * v_scale
                ).to(self.dtype)
            cute.arch.barrier()
            if tid == 0:
                cute.arch.inline_ptx(
                    "st.relaxed.gpu.global.u32 [{$r0}], 0;", read_only_args=[counter.toint()]
                )

    @cute.jit
    def finish_distributed(self, work, counters, arena, out, b, head, split, count, tid, v_scale):
        # CTA barrier gathers the stores, then thread zero's cumulative release
        # publishes them. Co-residency is checked against the compiled binary.
        cute.arch.barrier()
        counter = counters + (b * self.heads + head) * 2
        if tid == 0:
            generation = cute.arch.inline_ptx(
                "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];",
                write_only_types=[Uint32],
                read_only_args=[(counter + 1).toint()],
            )
            arrived = cute.arch.inline_ptx(
                "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], 1;",
                write_only_types=[Int32],
                read_only_args=[counter.toint()],
            )
            if arrived == count - 1:
                cute.arch.inline_ptx(
                    "st.relaxed.gpu.global.u32 [{$r0}], 0;",
                    read_only_args=[counter.toint()],
                )
                cute.arch.inline_ptx(
                    "st.release.gpu.global.u32 [{$r0}], {$r1};",
                    read_only_args=[(counter + 1).toint(), generation + Uint32(1)],
                )
            else:
                observed = generation
                polls = Int32(0)
                while observed == generation:
                    if polls < 64:
                        polls += 1
                    else:
                        cute.arch.inline_ptx("nanosleep.u32 32;")
                    observed = cute.arch.inline_ptx(
                        "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];",
                        write_only_types=[Uint32],
                        read_only_args=[(counter + 1).toint()],
                    )
        cute.arch.barrier()
        self.merge_tile(work, arena, out, b, head, split, count, tid, v_scale)

    @cute.jit
    def merge_tile(self, work, arena, out, b, head, split, count, tid, v_scale):
        # Partition vectors, not just rows: G=4 still uses dozens of merging
        # CTAs. A slice may cross a row boundary; each touched row has its own
        # normalized shared weights. Empty slices do not access the workspace.
        vectors = self.group * self.dim // 4
        first_vector, end_vector = vectors * split // count, vectors * (split + 1) // count
        if first_vector < end_vector:
            first_row = first_vector // (self.dim // 4)
            end_row = (end_vector + self.dim // 4 - 1) // (self.dim // 4)
            rows = end_row - first_row
            weights = cute.make_tensor(
                cute.recast_ptr(arena.iterator, dtype=Float32),
                cute.make_layout((self.splits, self.group), stride=(1, self.splits)),
            )
            base = (Int64(b) * self.heads + head) * self.splits * self.group
            for item in range(tid, count * rows, self.threads):
                row, source = item // count, item % count
                weights[source, row] = cute.arch.inline_ptx(
                    "ld.global.cg.f32 {$w0}, [{$r0}];",
                    write_only_types=[Float32],
                    read_only_args=[
                        (
                            work + self.lse_base + base + source * self.group + first_row + row
                        ).toint()
                    ],
                )
            cute.arch.barrier()
            lane, warp = tid % 32, tid // 32
            for row in range(warp, rows, self.warps):
                maximum = Float32(-float("inf"))
                for source in range(lane, count, 32):
                    maximum = max(maximum, weights[source, row])
                for shift in cutlass.range_constexpr(5):
                    maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift))
                safe = Float32(0)
                if maximum != -float("inf"):
                    safe = maximum
                total = Float32(0)
                for source in range(lane, count, 32):
                    weight = cute.math.exp2(weights[source, row] - safe)
                    weights[source, row] = weight
                    total += weight
                for shift in cutlass.range_constexpr(5):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                inverse = Float32(0)
                if total > 0:
                    inverse = 1.0 / total
                for source in range(lane, count, 32):
                    weights[source, row] *= inverse
            cute.arch.barrier()
            for vector in range(first_vector + tid, end_vector, self.threads):
                row, dim = vector // (self.dim // 4), vector % (self.dim // 4) * 4
                result = cute.make_rmem_tensor((4,), Float32)
                result.fill(0)
                for first in range(0, count, 4):
                    # Four independent 128-bit loads before any consumer FMA.
                    # .cg avoids stale L1 data across CTAs and graph replays.
                    values = cute.make_rmem_tensor((4, 4), Float32)
                    factors = cute.make_rmem_tensor((4,), Float32)
                    values.fill(0)
                    factors.fill(0)
                    for part in cutlass.range_constexpr(4):
                        source = first + part
                        if source < count:
                            factors[part] = weights[source, row - first_row]
                            address = (
                                work + (base + source * self.group + row) * self.dim + dim
                            ).toint()
                            values[0, part], values[1, part], values[2, part], values[3, part] = (
                                cute.arch.inline_ptx(
                                    "ld.global.cg.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                                    write_only_types=[Float32, Float32, Float32, Float32],
                                    read_only_args=[address],
                                )
                            )
                    for part in cutlass.range_constexpr(4):
                        for d in cutlass.range_constexpr(4):
                            result[d] += factors[part] * values[d, part]
                converted = cute.make_rmem_tensor((4,), self.dtype)
                for d in cutlass.range_constexpr(4):
                    converted[d] = (result[d] * v_scale).to(self.dtype)
                target = cute.make_tensor(
                    cute.make_ptr(
                        self.dtype,
                        (
                            out
                            + (Int64(b) * self.heads + head) * self.group * self.dim
                            + vector * 4
                        ).toint(),
                        out.memspace,
                        assumed_align=8,
                    ),
                    cute.make_layout(4),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=64
                    ),
                    converted,
                    target,
                )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k,
        v,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        last: cute.Tensor,
        out: cute.Pointer,
        work: cute.Pointer,
        counters: cute.Pointer,
        sm_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        mma: cute.TiledMma,
        qk_mma: cute.TiledMma,
    ):
        cute.arch.griddepcontrol_wait()
        # PDL trigger lets a dependent grid start; its wait still orders reads
        # after completion of this grid, including its last-arrival epilogues.
        cute.arch.griddepcontrol_launch_dependents()
        split, head, b = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        allocator = SmemAllocator()
        arena = allocator.allocate_tensor(
            Uint8, cute.make_layout(self.arena_bytes), byte_alignment=128
        )
        metadata = allocator.allocate_tensor(
            Int32, cute.make_layout(self.metadata_bytes // 4), byte_alignment=4
        )
        page_ids = metadata
        # CSR temporarily uses bank one. The first bank's completion barrier
        # retires every CSR reader before prefetching into bank one.
        csr = cute.make_tensor(metadata.iterator + self.page_cap, cute.make_layout(3))
        # CSR determines every first K/V address with the current ABI. Stage
        # it asynchronously once; no CSR or index global load enters walk().
        if tid == 0:
            copy4(csr.iterator, ptr.iterator + b, True)
            copy4(csr.iterator + 1, ptr.iterator + b + 1, True)
            copy4(csr.iterator + 2, last.iterator + b, True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        start = csr[0]
        pages = csr[1] - start
        length = Uint64(0)
        if pages > 0:
            length = Uint64(pages - 1) * self.page + Uint64(csr[2])
        count = Int32(min(max((length + self.minimum - 1) // self.minimum, 1), self.splits))
        # Match _split_geometry's CPU contract: whole 16-key blocks without
        # rounding up to a CTA stage, then remove any empty trailing splits.
        chunk = (max((length + Uint64(count) - 1) // Uint64(count), 1) + 15) // 16 * 16
        count = Int32((length + chunk - 1) // chunk)
        # The zero-length branch never divides by a zero chunk or touches KV.
        if length == 0:
            if split == 0:
                for element in range(tid, self.group * self.dim, self.threads):
                    out[(Int64(b) * self.heads + head) * self.group * self.dim + element] = (
                        self.dtype(0)
                    )
        else:
            if split < count:
                begin = Uint64(split) * chunk
                end = min(begin + chunk, length)
                winner = cute.make_tensor(
                    cute.recast_ptr(arena.iterator, dtype=Int32), cute.make_layout(1)
                )
                piece_end = min(begin + self.piece_keys, end)
                self.stage_pages(page_ids, idx, start, begin, piece_end, tid, 0)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if piece_end < end:
                    self.stage_pages(
                        page_ids,
                        idx,
                        start,
                        piece_end,
                        min(piece_end + self.piece_keys, end),
                        tid,
                        1,
                    )
                # Q, next-piece indices and the first K/V stage share a group.
                # Issue all priming K/V copies before waiting on Q's transfer.
                for vector in range(tid, self.group * self.dim // 8, self.threads):
                    source = (
                        q.iterator
                        + Int64(b) * q.stride[0]
                        + head * self.group * self.dim
                        + vector * 8
                    )
                    copy16(arena.iterator + self.q_offset(vector * 16), source, True)
                cute.arch.inline_ptx("// FP8_PRIME_BEGIN")
                self.prime(arena, k, v, page_ids, 0, begin, piece_end, head, tid)
                cute.arch.inline_ptx("// FP8_PRIME_END")
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.barrier()
                exact = Int32(1)
                qwords = cute.make_rmem_tensor((4, 1, self.dim // 16), Uint32)
                for inner in cutlass.range_constexpr(self.dim // 16):
                    for rr in cutlass.range_constexpr(2):
                        row = lane // 4 + rr * 8
                        dim = inner * 16 + lane % 4 * 4
                        packed = cute.make_rmem_tensor((2,), Uint32)
                        packed.fill(0)
                        if row < self.group:
                            address = (
                                (arena.iterator + self.q_offset((row * self.dim + dim) * 2))
                                .toint()
                                .to(Int32)
                            )
                            packed[0], packed[1] = cute.arch.inline_ptx(
                                "ld.shared.v2.u32 {{$w0}, {$w1}}, [{$r0}];",
                                write_only_types=[Uint32, Uint32],
                                read_only_args=[address],
                            )
                        values = cute.recast_tensor(packed, self.dtype)
                        for pair in cutlass.range_constexpr(2):
                            if const_expr(self.dtype == cutlass.BFloat16):
                                qwords[rr + pair * 2, 0, inner] = packed[pair]
                            else:
                                qwords[rr + pair * 2, 0, inner] = half_pair(
                                    values[pair * 2].to(Float16), values[pair * 2 + 1].to(Float16)
                                )
                    if const_expr(self.dtype == cutlass.BFloat16):
                        # Every warp packs the same Q tile. Divide the checks
                        # across warps while packing: each K-dimension stripe
                        # is checked once per CTA, with no extra shared reads.
                        if tid // 32 == inner % self.warps:
                            for word in cutlass.range_constexpr(4):
                                if const_expr(word % 2 < (self.group + 7) // 8):
                                    bits = qwords[word, 0, inner]
                                    lo = bits.to(Uint16).bitcast(cutlass.BFloat16)
                                    hi = (bits >> 16).to(Uint16).bitcast(cutlass.BFloat16)
                                    exact = bf16_fp16_exact(lo, exact)
                                    exact = bf16_fp16_exact(hi, exact)
                acc = cute.make_rmem_tensor((4, 1, self.dim // 8), Float32)
                maxima, sums = (
                    cute.make_rmem_tensor((2,), Float32),
                    cute.make_rmem_tensor((2,), Float32),
                )
                acc.fill(0)
                maxima.fill(-float("inf"))
                sums.fill(0)
                # Every warp consumes the same Q tile before warp zero can
                # overwrite its final ring slots with the first lookahead.
                if const_expr(self.dtype == cutlass.BFloat16):
                    # Reuse Q's retirement barrier: one CTA-wide vote, before
                    # the walk and before the ring reuses Q.
                    exact = cute.arch.inline_ptx(
                        "{ .reg .pred p, all_exact; "
                        "setp.ne.u32 p, {$r0}, 0; "
                        "bar.red.and.pred all_exact, 0, p; "
                        "selp.u32 {$w0}, 1, 0, all_exact; }",
                        write_only_types=[Int32],
                        read_only_args=[exact],
                    )
                    if exact != 0:
                        # Retain one packed Q register set. All consumers now
                        # interpret it as FP16; the fallback keeps BF16 bits.
                        for inner in cutlass.range_constexpr(self.dim // 16):
                            for word in cutlass.range_constexpr(4):
                                bits = qwords[word, 0, inner]
                                lo = bits.to(Uint16).bitcast(cutlass.BFloat16)
                                hi = (bits >> 16).to(Uint16).bitcast(cutlass.BFloat16)
                                qwords[word, 0, inner] = half_pair(lo.to(Float16), hi.to(Float16))
                else:
                    cute.arch.barrier()
                # RTX 5090: per-stage branching cost 3-28% vs the FP16-only kernel;
                # whole-walk specialisation costs 0-6% (per-CTA exactness vote and Q repack).
                if const_expr(self.dtype == cutlass.BFloat16):
                    if exact != 0:
                        self.walk_pieces(
                            arena,
                            qwords,
                            mma,
                            mma,
                            k,
                            v,
                            page_ids,
                            idx,
                            start,
                            begin,
                            end,
                            head,
                            tid,
                            acc,
                            maxima,
                            sums,
                            sm_scale * k_scale * 1.4426950408889634,
                            False,
                        )
                    else:
                        self.walk_pieces(
                            arena,
                            qwords,
                            mma,
                            qk_mma,
                            k,
                            v,
                            page_ids,
                            idx,
                            start,
                            begin,
                            end,
                            head,
                            tid,
                            acc,
                            maxima,
                            sums,
                            sm_scale * k_scale * 1.4426950408889634,
                            True,
                        )
                else:
                    self.walk_pieces(
                        arena,
                        qwords,
                        mma,
                        qk_mma,
                        k,
                        v,
                        page_ids,
                        idx,
                        start,
                        begin,
                        end,
                        head,
                        tid,
                        acc,
                        maxima,
                        sums,
                        sm_scale * k_scale * 1.4426950408889634,
                        False,
                    )
                self.reduce_warps(
                    arena, acc, maxima, sums, out, work, b, head, split, count, tid, v_scale
                )
                if const_expr(self.splits > 1):
                    if count > 1:
                        if const_expr(self.distributed):
                            self.finish_distributed(
                                work, counters, arena, out, b, head, split, count, tid, v_scale
                            )
                        else:
                            self.finish(work, counters, winner, out, b, head, count, tid, v_scale)
