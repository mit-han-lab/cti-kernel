# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: SGLang 72730d6 qwen3x_nvfp4_gemm_sm120.py (Apache-2.0).
# Recipe follows FlashInfer 0.7.0.post1 (csrc/nv_internal/tensorrt_llm/kernels/
# quantization_utils.cuh, cvt_warp_fp16_to_fp4), design reference only (Apache-2.0).
# Design reference: CUTLASS 4.8.0 blockscaled layouts and QuACK 0.6.5 quantization.
# No third-party implementation is copied or adapted.

"""Independent quantizer A/SFA ring alongside a B/SFB-only TMA ring.

Quantizer warps batch global loads before quantizing each A tile. Consumers wait
on both rings and issue only shared operand loads and MMA in the mainloop.
The last warp independently streams B/SFB, including before the PDL input wait.
A/B use 16-byte sectors across rows: each ldmatrix row group hits distinct banks.
SFA is K64-major; prepared SFB holds eight rows of four scales per 32-byte sector.
Each operand takes one tensor copy per stage, with transaction-counted full barriers
and consumer-counted empty barriers. Tensor-map bounds zero-fill the N/K tails.

Global splits publish compact partials before an acquire-release arrival; the last
CTA reduces in ascending split order. Clusters reduce through peer shared memory.
Split-one CTAs directly run the epilogue. One workspace requires serialized
calls/replays. Debug stores expose quantized A and FP32 epilogue inputs.

PTX reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
Sections: mma.m16n8k64 fragments, block scaling, ldmatrix, mbarrier, cache policy.
"""

from __future__ import annotations

import json
import os
import threading

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32, cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_ptr
from cutlass.utils import SmemAllocator

from .._nvfp4_cute_recipe import maximum as quant_max
from .._nvfp4_cute_recipe import mul as quant_mul
from .._nvfp4_cute_recipe import pack as quant_pack
from .._nvfp4_cute_recipe import scale as quant_scale
from . import available


@cute.jit
def reciprocal(value):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, 0f3f800000, {$r0};", write_only_types=[Float32], read_only_args=[value]
    )


@cute.jit
def divide(a, b):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};", write_only_types=[Float32], read_only_args=[a, b]
    )


@cute.jit
def shuffle_xor(value, mask, distance: cutlass.Constexpr):
    return cute.arch.inline_ptx(
        "shfl.sync.bfly.b32 {$w0}, {$r0}, {$r1}, 31, {$r2};",
        write_only_types=[Uint32],
        read_only_args=[value, Int32(distance), mask],
    )


@cute.jit
def quant_lanes(values, inverse, lanes: cutlass.Constexpr):
    # Each block spans adjacent lanes. Its explicit mask allows inactive row
    # groups to skip quantization without leaving missing shuffle participants.
    tid, _, _ = cute.arch.thread_idx()
    mask = Uint32((1 << lanes) - 1) << ((tid % 32 // lanes) * lanes)
    maximum = cute.math.absf(values[0])
    for i in cutlass.range_constexpr(1, 16 // lanes):
        maximum = quant_max(maximum, cute.math.absf(values[i]))
    for bit in cutlass.range_constexpr(2 if lanes == 4 else 4):
        maximum = quant_max(
            maximum, shuffle_xor(maximum.bitcast(Uint32), mask, 1 << bit).bitcast(Float32)
        )
    sf, rescale = quant_scale(maximum, inverse)
    word = Uint32(0)
    for i in cutlass.range_constexpr(16 // lanes):
        value = quant_mul(values[i], rescale)
        code = quant_pack(value, value).to(Uint32) & Uint32(15)
        word = word | (code << (4 * i))
    for bit in cutlass.range_constexpr(1 if lanes == 4 else 3):
        word = word | (shuffle_xor(word, mask, 1 << bit) << ((64 // lanes) * (1 << bit)))
    return word, sf.bitcast(Uint8)


@cute.jit
def scale_offset(row, block, atoms_k: cutlass.Constexpr):
    return (
        Int64(row // 128) * atoms_k * 512
        + Int64(block // 4) * 512
        + Int64(row % 32) * 16
        + Int64(row // 32 % 4) * 4
        + block % 4
    )


@cute.jit
def load_x8(ptr):
    return cute.arch.inline_ptx(
        "ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Uint32, Uint32, Uint32, Uint32],
        read_only_args=[ptr.toint()],
    )


@cute.jit
def quant_block(values, inverse):
    maximum = cute.math.absf(values[0])
    for i in cutlass.range_constexpr(1, 16):
        maximum = quant_max(maximum, cute.math.absf(values[i]))
    sf, rescale = quant_scale(maximum, inverse)
    scale = sf.bitcast(Uint8)
    words = cute.make_rmem_tensor(2, Uint32)
    words.fill(0)
    for pair in cutlass.range_constexpr(8):
        code = quant_pack(
            quant_mul(values[pair * 2], rescale), quant_mul(values[pair * 2 + 1], rescale)
        )
        words[pair // 4] = words[pair // 4] | (code.to(Uint32) << (8 * (pair % 4)))
    return words[0], words[1], scale


@cute.jit
def load_a(ptr):
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Uint32, Uint32, Uint32, Uint32],
        read_only_args=[Int32(ptr.toint())],
    )


@cute.jit
def load_b(ptr):
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Uint32, Uint32],
        read_only_args=[Int32(ptr.toint())],
    )


@cute.jit
def mma(a0, a1, a2, a3, b0, b1, sa, sb, c0, c1, c2, c3):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "f32.e2m1.e2m1.f32.ue4m3 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, "
        "{{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
        "{{$r8}, {$r9}, {$r10}, {$r11}}, {$r6}, {0, 0}, {$r7}, {0, 0};",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[a0, a1, a2, a3, b0, b1, sa, sb, c0, c1, c2, c3],
    )


@cute.jit
def wait_token(ptr, phase):
    # Blocking parity wait that yields an opaque zero. Lanes may leave the spin loop in
    # different iterations; bar.warp.sync reconverges the warp before the following
    # ldmatrix/mma.sync.aligned instructions. Operand loads are inline PTX
    # the compiler may treat as pure; adding the token to their addresses makes
    # every ldmatrix/ld.shared of a stage data-dependent on that stage's wait.
    return cute.arch.inline_ptx(
        "{ .reg .pred done; spin: mbarrier.try_wait.parity.shared::cta.b64 done, [{$r0}], {$r1}; "
        "@!done bra spin; bar.warp.sync 0xffffffff; mov.u32 {$w0}, 0; }",
        write_only_types=[Int32],
        read_only_args=[Int32(ptr.toint()), Int32(phase)],
    )


@cute.jit
def grid_token():
    # griddepcontrol.wait with an opaque zero result; x and scale reads add it.
    return cute.arch.inline_ptx(
        "{ griddepcontrol.wait; mov.u32 {$w0}, 0; }", write_only_types=[Int32]
    )


@cute.jit
def arrive_after(ptr, acc, groups: cutlass.Constexpr):
    # Consumer release ordered after every read of the stage. Accumulators that are
    # only asm inputs vanish from the PTX text, and ptxas then issued the arrive before
    # the last ldmatrix/ld.shared results were consumed, so a refill could overwrite the
    # stage while those reads were pending. The predicate reads the final result of
    # each N group (x == x is false only for NaN, so it cannot be folded); both paths
    # arrive exactly once, after the MMAs and hence after every operand load.
    deps = [acc[nn, 3] for nn in range(groups)]
    test = " ".join(
        "setp.eq.and.f32 ok, {$r%d}, {$r%d}, ok;" % (i + 1, i + 1) for i in range(groups)
    )
    cute.arch.inline_ptx(
        "{ .reg .pred ok; .reg .b64 state; setp.eq.u32 ok, 0, 0; "
        + test
        + " @ok mbarrier.arrive.shared::cta.b64 state, [{$r0}];"
        " @!ok mbarrier.arrive.release.cta.shared::cta.b64 state, [{$r0}]; }",
        read_only_args=[Int32(ptr.toint())] + deps,
    )


@cute.jit
def store_partial(ptr, v0, v1, v2, v3):
    cute.arch.inline_ptx(
        "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[ptr.toint(), v0, v1, v2, v3],
    )


@cute.jit
def load_partial(ptr):
    return cute.arch.inline_ptx(
        "ld.global.cg.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[ptr.toint()],
    )


@cute.jit
def load_peer(ptr, rank):
    return cute.arch.inline_ptx(
        "{ .reg .b32 remote; mapa.shared::cluster.u32 remote, {$r0}, {$r1}; "
        "ld.shared::cluster.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [remote]; }",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[Uint32(ptr.toint()), Int32(rank)],
    )


@cute.jit
def cluster_sync():
    cute.arch.inline_ptx("barrier.cluster.arrive.release; barrier.cluster.wait.acquire;")


@cute.jit
def debug_wait(
    ptr,
    errors,
    abort,
    kind,
    stage,
    phase,
    tries: cutlass.Constexpr = 1 << 26,
    cluster: cutlass.Constexpr = False,
    consumer_barrier: cutlass.Constexpr = False,
    pdl: cutlass.Constexpr = False,
):
    # Public PTX only. All lanes finish the bounded wait before a whole-warp exit;
    # a single exited lane must never leave its peers in ldmatrix or mma.sync.
    ready = cute.arch.inline_ptx(
        "{ .reg .pred done, more; .reg .u32 remaining; "
        "mov.u32 remaining, {$r2}; mov.u32 {$w0}, 0; "
        "poll: mbarrier.try_wait.parity.shared::cta.b64 done, [{$r0}], {$r1}, 1; "
        "@done bra complete; sub.u32 remaining, remaining, 1; "
        "setp.ne.u32 more, remaining, 0; @more bra poll; bra finish; "
        "complete: mov.u32 {$w0}, 1; finish: }",
        write_only_types=[Int32],
        read_only_args=[Int32(ptr.toint()), Int32(phase), Int32(tries)],
    )
    failed = cute.arch.inline_ptx(
        "{ .reg .pred bad, any_bad; setp.eq.u32 bad, {$r0}, 0; "
        "vote.sync.any.pred any_bad, bad, 0xffffffff; "
        "selp.u32 {$w0}, 1, 0, any_bad; }",
        write_only_types=[Int32],
        read_only_args=[ready],
    )
    if failed != 0:
        if cutlass.const_expr(pdl):
            cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        if tid % 32 == 0:
            cute.arch.atomic_exch(abort, Int32(1), sem="relaxed", scope="cta")
            owner = cute.arch.inline_ptx(
                "atom.global.cas.b32 {$w0}, [{$r0}], 0, 1;",
                write_only_types=[Int32],
                read_only_args=[errors.toint()],
            )
            if owner == 0:
                record = cute.make_tensor(errors, cute.make_layout(7))
                bn, split, bm = cute.arch.block_idx()
                record[1], record[2], record[3] = kind, stage, phase
                record[4], record[5], record[6] = bn, split, bm
        if cutlass.const_expr(cluster or consumer_barrier):
            # A local exit could strand peers at a counted barrier or free DSMEM.
            cute.arch.inline_ptx("trap;")
        else:
            cute.arch.inline_ptx("exit;")


class Kernel:
    def __init__(self, rows, n, k, stride, config, gated, debug, mode, debug_timeout=False):
        self.rows, self.n, self.k, self.stride = rows, n, k, stride
        self.nt, self.kt, self.stages = config.n_tile, config.k_tile, config.stages
        self.splits, self.warps = config.split_k, config.warps + config.qwarps
        self.consumers = config.warps - 1
        self.qwarps, self.a_stages, self.pdl = config.qwarps, config.a_stages, config.pdl
        # Each quantizer warp owns whole tiles (step % qwarps), so up to qwarps tiles
        # are in flight; its 32 lanes cover one tile's live K16 blocks.
        self.items = (min(rows, 16) * (self.kt // 16) + 31) // 32
        self.ctas_per_sm = config.ctas_per_sm
        self.probe = config.probe
        self.cluster = config.reduce_mode == "cluster"
        self.gated, self.debug, self.mode = gated, debug, mode
        self.debug_timeout = debug_timeout
        self.tiles = (
            ((n + 127) // 128 * 128 + self.nt - 1) // self.nt
            if gated
            else (n + self.nt - 1) // self.nt
        )
        self.iterations = (k + self.kt - 1) // self.kt
        self.per_split = (self.iterations + self.splits - 1) // self.splits
        smem = (
            max(self.a_stages * 9 * self.kt, 64 * self.nt)
            + self.stages * self.nt * self.kt * 9 // 16
            + 16 * (self.stages + self.a_stages)
            + 16
        )
        if smem > config.smem_limit:
            raise ValueError("configuration exceeds the shared-memory budget for the A ring")

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        ws: cute.Pointer,
        g0: cute.Pointer,
        g1: cute.Pointer,
        ig: cute.Pointer,
        og: cute.Pointer,
        out: cute.Pointer,
        os: cute.Pointer,
        partial: cute.Pointer,
        counter: cute.Pointer,
        debug_data: cute.Pointer,
        debug_scale: cute.Pointer,
        projection: cute.Pointer,
        postact: cute.Pointer,
        errors: cute.Pointer,
        stamps: cute.Pointer,
        probe_counts: cute.Pointer,
        stream: cuda.CUstream,
    ):
        # The K32 sector is the innermost dimension; N precedes the next sector in
        # shared memory. This is an unswizzled, bank-conflict-free ldmatrix layout.
        b_shape = (16, self.nt, self.kt // 32)
        sb_shape = (32, self.kt // 64, self.nt // 8)
        wb = cute.make_tensor(
            w, cute.make_layout((16, self.n, self.k // 32), stride=(1, self.k // 2, 16))
        )
        wsb = cute.make_tensor(ws, cute.make_layout((32, (self.k + 63) // 64, (self.n + 7) // 8)))
        b_atom, b_map = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), wb, cute.make_layout(b_shape), b_shape
        )
        sb_atom, sb_map = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), wsb, cute.make_layout(sb_shape), sb_shape
        )
        self.run(
            x,
            b_atom,
            b_map,
            sb_atom,
            sb_map,
            g0,
            g1,
            ig,
            og,
            out,
            os,
            partial,
            counter,
            debug_data,
            debug_scale,
            projection,
            postact,
            errors,
            stamps,
            probe_counts,
        ).launch(
            grid=(self.tiles, self.splits, (self.rows + 15) // 16),
            block=(self.warps * 32, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.ctas_per_sm,
            cluster=(1, self.splits, 1) if self.cluster else None,
            use_pdl=self.pdl,
        )

    @cute.jit
    def prefetch(self, b_atom, sb_atom, bg, bs, sg, ss, full, step, begin, policy):
        stage = step % self.stages
        tid, _, _ = cute.arch.thread_idx()
        if tid % 32 == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                full.iterator + stage, self.nt * self.kt * 9 // 16
            )
        cute.arch.sync_warp()
        # cute.copy elects its issuer internally: every producer lane must enter.
        cute.copy(
            b_atom,
            bg[None, begin + step],
            bs[None, stage],
            tma_bar_ptr=full.iterator + stage,
            cache_policy=policy,
        )
        cute.copy(
            sb_atom,
            sg[None, begin + step],
            ss[None, stage],
            tma_bar_ptr=full.iterator + stage,
            cache_policy=policy,
        )

    @cute.jit
    def wait(self, ptr, errors, abort, kind, stage, phase):
        # Producer-side waits: the following TMA copies / shared stores are ordered
        # side-effecting operations, so the intrinsic wait suffices.
        if cutlass.const_expr(self.debug_timeout):
            debug_wait(
                ptr,
                errors,
                abort,
                kind,
                stage,
                phase,
                cluster=self.cluster,
                consumer_barrier=True,
                pdl=self.pdl,
            )
        else:
            cute.arch.mbarrier_wait(ptr, phase)

    @cute.jit
    def wait_ready(self, ptr, errors, abort, kind, stage, phase):
        # Consumer-side waits return a token that the operand addresses must add.
        if cutlass.const_expr(self.debug_timeout):
            debug_wait(
                ptr,
                errors,
                abort,
                kind,
                stage,
                phase,
                cluster=self.cluster,
                consumer_barrier=True,
                pdl=self.pdl,
            )
        return wait_token(ptr, phase)

    @cute.jit
    def quant_a(self, x, a, sa, inverse, debug_data, debug_scale, qtid, bn, m0, tile, slot, token):
        live_rows = cutlass.min(16, self.rows - m0)
        # At most 64 packed load registers per batch. One lane owns a whole K16
        # block; all unconditional loads precede any conversion or quantization.
        for batch in cutlass.range_constexpr((self.items + 7) // 8):
            batch_items = min(8, self.items - batch * 8)
            loaded = cute.make_rmem_tensor((batch_items, 8), Uint32)
            valid = cute.make_rmem_tensor(batch_items, cutlass.Boolean)
            for turn in cutlass.range_constexpr(batch_items):
                item = (batch * 8 + turn) * 32 + qtid
                rr, kb = item // (self.kt // 16), item % (self.kt // 16)
                k0 = Int64(tile) * self.kt + kb * 16
                valid[turn] = (rr < live_rows) & (k0 < self.k)
                offset = (
                    Int64(m0 + cutlass.min(rr, live_rows - 1)) * self.stride
                    + cutlass.min(k0, Int64(self.k - 16))
                    + Int64(token)
                )
                x0, x1, x2, x3 = load_x8(x.iterator + offset)
                x4, x5, x6, x7 = load_x8(x.iterator + offset + Int64(8))
                loaded[turn, 0], loaded[turn, 1] = x0, x1
                loaded[turn, 2], loaded[turn, 3] = x2, x3
                loaded[turn, 4], loaded[turn, 5] = x4, x5
                loaded[turn, 6], loaded[turn, 7] = x6, x7
            for turn in cutlass.range_constexpr(batch_items):
                item = (batch * 8 + turn) * 32 + qtid
                rr, kb = item // (self.kt // 16), item % (self.kt // 16)
                k0 = Int64(tile) * self.kt + kb * 16
                vals = cute.make_rmem_tensor(16, Float32)
                for j in cutlass.range_constexpr(16):
                    bits = (loaded[turn, j // 2] >> (16 * (j % 2))).to(Uint16)
                    value = bits.bitcast(x.element_type).to(Float32)
                    vals[j] = Float32(cutlass.select_(valid[turn], value, Float32(0.0)))
                word0, word1, scale = quant_block(vals, inverse)
                if rr < live_rows:
                    dest = slot * (2 * self.kt) + (kb // 2) * 64 + rr * 4 + (kb % 2) * 2
                    cute.arch.inline_ptx(
                        "st.shared.v2.b32 [{$r0}], {{$r1}, {$r2}};",
                        read_only_args=[Int32((a.iterator + dest).toint()), word0, word1],
                    )
                    sa[slot * self.kt + (kb // 4) * 64 + rr * 4 + kb % 4] = scale
                    if cutlass.const_expr(self.debug):
                        if bn == 0 and k0 < self.k:
                            ptr = cute.make_tensor(
                                cute.recast_ptr(debug_data, dtype=Uint32),
                                cute.make_layout(self.rows * self.k // 8),
                            )
                            base = Int64(m0 + rr) * (self.k // 8) + k0 // 8
                            ptr[base], ptr[base + Int64(1)] = word0, word1
                            debug_scale[scale_offset(m0 + rr, k0 // 16, (self.k + 63) // 64)] = (
                                scale
                            )

    @cute.kernel
    def run(
        self,
        x: cute.Pointer,
        b_atom: cute.CopyAtom,
        b_map: cute.Tensor,
        sb_atom: cute.CopyAtom,
        sb_map: cute.Tensor,
        g0: cute.Pointer,
        g1: cute.Pointer,
        ig: cute.Pointer,
        og: cute.Pointer,
        out: cute.Pointer,
        os: cute.Pointer,
        partial: cute.Pointer,
        counter: cute.Pointer,
        debug_data: cute.Pointer,
        debug_scale: cute.Pointer,
        projection: cute.Pointer,
        postact: cute.Pointer,
        errors: cute.Pointer,
        stamps: cute.Pointer,
        probe_counts: cute.Pointer,
    ):
        x = cute.make_tensor(x, cute.make_layout(self.rows * self.stride))
        g0 = cute.make_tensor(g0, cute.make_layout(self.n if self.mode == 2 else 1))
        g1 = cute.make_tensor(g1, cute.make_layout(1))
        ig = cute.make_tensor(ig, cute.make_layout(1))
        og = cute.make_tensor(og, cute.make_layout(1))
        out = cute.make_tensor(
            out, cute.make_layout(self.rows * (self.n // 4 if self.gated else self.n))
        )
        os = cute.make_tensor(os, cute.make_layout(((self.n // 2 + 63) // 64) * 512))
        partial = cute.make_tensor(
            partial, cute.make_layout(self.tiles * self.splits * self.rows * self.nt)
        )
        projection = cute.make_tensor(
            projection, cute.make_layout(self.rows * self.n if self.debug else 1)
        )
        postact = cute.make_tensor(
            postact, cute.make_layout(self.rows * self.n // 2 if self.debug and self.gated else 1)
        )
        debug_scale = cute.make_tensor(
            debug_scale, cute.make_layout(((self.k + 63) // 64) * 512 if self.debug else 1)
        )
        tid, _, _ = cute.arch.thread_idx()
        bn, split, bm = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        n0, m0 = bn * self.nt, bm * 16
        alloc = SmemAllocator()
        a = alloc.allocate_tensor(
            Uint32, max(self.a_stages * 9 * self.kt // 4, 16 * self.nt), byte_alignment=128
        )
        sa = cute.make_tensor(
            cute.recast_ptr(a.iterator + self.a_stages * 2 * self.kt, dtype=Uint8),
            cute.make_layout(self.a_stages * self.kt),
        )
        b = alloc.allocate_tensor(
            Uint8, cute.make_layout((16, self.nt, self.kt // 32, self.stages)), byte_alignment=128
        )
        sb = alloc.allocate_tensor(
            Uint8,
            cute.make_layout((32, self.kt // 64, self.nt // 8, self.stages)),
            byte_alignment=128,
        )
        full = alloc.allocate_tensor(Int64, self.stages, byte_alignment=8)
        empty = alloc.allocate_tensor(Int64, self.stages, byte_alignment=8)
        a_full = alloc.allocate_tensor(Int64, self.a_stages, byte_alignment=8)
        a_empty = alloc.allocate_tensor(Int64, self.a_stages, byte_alignment=8)
        result = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=Float32), cute.make_layout(16 * self.nt)
        )
        last = alloc.allocate_tensor(Int32, 1, byte_alignment=4)
        gb = cute.local_tile(b_map, (16, self.nt, self.kt // 32), (0, bn, None))
        gs = cute.local_tile(sb_map, (32, self.kt // 64, self.nt // 8), (0, None, bn))
        bs, bg = cpasync.tma_partition(
            b_atom, 0, cute.make_layout(1), cute.group_modes(b, 0, 3), cute.group_modes(gb, 0, 3)
        )
        ss, sg = cpasync.tma_partition(
            sb_atom, 0, cute.make_layout(1), cute.group_modes(sb, 0, 3), cute.group_modes(gs, 0, 3)
        )
        sap = cute.make_tensor(
            cute.recast_ptr(sa.iterator, dtype=Uint32),
            cute.make_layout(self.a_stages * self.kt // 4),
        )
        sbp = cute.make_tensor(
            cute.recast_ptr(sb.iterator, dtype=Uint32),
            cute.make_layout(self.stages * self.nt * self.kt // 64),
        )
        acc = cute.make_rmem_tensor((self.nt // (8 * self.consumers), 4), Float32)
        acc.fill(0.0)
        begin = split * self.per_split
        count = cutlass.max(cutlass.min(self.per_split, self.iterations - begin), 0)
        live_rows = cutlass.min(16, self.rows - m0)
        if tid == (self.warps - 1) * 32:
            last[0] = Int32(0)
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full.iterator + stage, 1)
                cute.arch.mbarrier_init(empty.iterator + stage, self.consumers * 32)
            for slot in cutlass.range_constexpr(self.a_stages):
                cute.arch.mbarrier_init(a_full.iterator + slot, 32)
                cute.arch.mbarrier_init(a_empty.iterator + slot, self.consumers * 32)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.probe != "load_only"):
            # Padded rows remain zero in every slot; live rows are overwritten.
            for i in cutlass.range(tid, self.a_stages * 2 * self.kt, self.warps * 32):
                if cutlass.const_expr(self.probe == "no_prologue"):
                    a[i] = Uint32(0)
                elif i // 4 % 16 >= live_rows:
                    a[i] = Uint32(0)
            for i in cutlass.range(tid, self.a_stages * self.kt // 4, self.warps * 32):
                if cutlass.const_expr(self.probe == "no_prologue"):
                    sap[i] = Uint32(0)
                elif i % 16 >= live_rows:
                    sap[i] = Uint32(0)
        cute.arch.sync_threads()
        policy = cute.arch.inline_ptx(
            "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;",
            write_only_types=[Int64],
        )
        token = Int32(0)
        if cutlass.const_expr(self.pdl):
            if warp != self.warps - 1:
                token = grid_token()
        if warp == self.warps - 1:
            for step in cutlass.range(count):
                stage = step % self.stages
                if step >= self.stages:
                    self.wait(
                        empty.iterator + stage,
                        errors,
                        last.iterator,
                        2,
                        stage,
                        (step // self.stages - 1) % 2,
                    )
                self.prefetch(b_atom, sb_atom, bg, bs, sg, ss, full, step, begin, policy)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
        elif warp >= self.consumers:
            if cutlass.const_expr(self.probe != "load_only"):
                inv = Float32(0.0)
                if cutlass.const_expr(self.probe != "no_prologue"):
                    inv = reciprocal(ig[token])
                # Warp qw fills steps qw, qw + qwarps, ...; a_stages >= qwarps keeps every
                # a_empty parity wait within one phase of the consumers.
                qw = warp - self.consumers
                for turn in cutlass.range((count - qw + self.qwarps - 1) // self.qwarps):
                    step = qw + turn * self.qwarps
                    slot = step % self.a_stages
                    if step >= self.a_stages:
                        self.wait(
                            a_empty.iterator + slot,
                            errors,
                            last.iterator,
                            4,
                            slot,
                            (step // self.a_stages - 1) % 2,
                        )
                    if cutlass.const_expr(self.probe != "no_prologue"):
                        self.quant_a(
                            x,
                            a,
                            sa,
                            inv,
                            debug_data,
                            debug_scale,
                            lane,
                            bn,
                            m0,
                            begin + step,
                            slot,
                            token,
                        )
                    # Every lane releases its own shared stores; the consumer acquire
                    # then covers the whole slot without relying on warp-level ordering.
                    cute.arch.mbarrier_arrive(a_full.iterator + slot)
        else:
            for step in cutlass.range(count):
                stage, slot = step % self.stages, step % self.a_stages
                ready = self.wait_ready(
                    full.iterator + stage, errors, last.iterator, 1, stage, step // self.stages % 2
                )
                if cutlass.const_expr(self.probe != "load_only"):
                    ready = ready + self.wait_ready(
                        a_full.iterator + slot,
                        errors,
                        last.iterator,
                        3,
                        slot,
                        step // self.a_stages % 2,
                    )
                    bstage = b.iterator + (stage * (self.nt * self.kt // 2) + ready)
                    sbbase = stage * (self.nt * self.kt // 64) + ready
                    astage = a.iterator + (
                        slot * (2 * self.kt) + (lane // 16) * 64 + (lane % 16) * 4 + ready
                    )
                    sabase = slot * (self.kt // 4) + lane // 4 + (lane % 2) * 8 + ready
                    for kk in cutlass.range_constexpr(self.kt // 64):
                        a0, a1, a2, a3 = load_a(astage + kk * 128)
                        saf = sap[sabase + kk * 16]
                        for nn in cutlass.range_constexpr(self.nt // (8 * self.consumers)):
                            group = nn * self.consumers + warp
                            nr = group * 8 + lane % 8
                            bbase = (kk * 2 + lane // 8 % 2) * (16 * self.nt) + nr * 16
                            b0, b1 = load_b(bstage + bbase)
                            sbf = sbp[sbbase + group * (self.kt // 8) + kk * 8 + lane // 4]
                            if cutlass.const_expr(self.probe == "no_mma"):
                                # Keep every fragment live without issuing MMA. The bounded
                                # checksum is diagnostic work, not a numerical result.
                                bits = a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ saf ^ sbf
                                acc[nn, 0] = acc[nn, 0] + (bits & Uint32(255)).to(Float32)
                            else:
                                c0, c1, c2, c3 = mma(
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    b0,
                                    b1,
                                    saf,
                                    sbf,
                                    acc[nn, 0],
                                    acc[nn, 1],
                                    acc[nn, 2],
                                    acc[nn, 3],
                                )
                                acc[nn, 0], acc[nn, 1] = c0, c1
                                acc[nn, 2], acc[nn, 3] = c2, c3
                groups = self.nt // (8 * self.consumers)
                if step + self.stages < count:
                    arrive_after(empty.iterator + stage, acc, groups)
                if cutlass.const_expr(self.probe != "load_only"):
                    if step + self.a_stages < count:
                        arrive_after(a_empty.iterator + slot, acc, groups)
        # Every role joins before the combined A/SFA allocation becomes the result.
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.debug_timeout):
            if last[0] != 0:
                cute.arch.inline_ptx("exit;")
        live_rows = cutlass.min(16, self.rows - m0)
        if cutlass.const_expr(self.probe == "load_only"):
            if tid == 0:
                counts = cute.make_tensor(
                    probe_counts,
                    cute.make_layout(((self.rows + 15) // 16) * self.tiles * self.splits),
                )
                counts[(Int64(bm) * self.tiles + bn) * self.splits + split] = cutlass.max(
                    cutlass.min(self.per_split, self.iterations - split * self.per_split), 0
                )
        else:
            if warp < self.consumers:
                for nn in cutlass.range_constexpr(self.nt // (8 * self.consumers)):
                    for j in cutlass.range_constexpr(4):
                        rr = lane // 4 + (j // 2) * 8
                        cc = (nn * self.consumers + warp) * 8 + (lane % 4) * 2 + j % 2
                        if rr < live_rows:
                            result[rr * self.nt + cc] = acc[nn, j]
            cute.arch.sync_threads()

            tile_id = Int64(bm) * self.tiles + bn
            if cutlass.const_expr(self.splits > 1 and not self.cluster):
                base = ((Int64(bn) * self.splits + split) * self.rows + m0) * self.nt
                for v in cutlass.range(tid, live_rows * self.nt // 4, self.warps * 32):
                    i = v * 4
                    store_partial(
                        partial.iterator + base + i,
                        result[i],
                        result[i + 1],
                        result[i + 2],
                        result[i + 3],
                    )
                if cutlass.const_expr(self.probe != "no_reduce"):
                    # CTA join carries every writer's stores into the leader's release.
                    # The acquire-release RMW chain also acquires all earlier splits.
                    cute.arch.sync_threads()
                    if tid == 0:
                        previous = cute.arch.atomic_add(
                            counter + tile_id, Int32(1), sem="acq_rel", scope="gpu"
                        )
                        last[0] = Int32(previous == self.splits - 1)
                    cute.arch.sync_threads()
                    if last[0] != 0:
                        for v in cutlass.range(tid, live_rows * self.nt // 4, self.warps * 32):
                            i = v * 4
                            sums = cute.make_rmem_tensor(4, Float32)
                            sums.fill(0.0)
                            for rank in cutlass.range_constexpr(self.splits):
                                base = ((Int64(bn) * self.splits + rank) * self.rows + m0) * self.nt
                                v0, v1, v2, v3 = load_partial(partial.iterator + base + i)
                                sums[0], sums[1] = sums[0] + v0, sums[1] + v1
                                sums[2], sums[3] = sums[2] + v2, sums[3] + v3
                            for j in cutlass.range_constexpr(4):
                                result[i + j] = sums[j]
                        cute.arch.sync_threads()
                        self.epilogue(
                            result, g0, g1, ig, og, out, os, projection, postact, tid, n0, m0, bm
                        )
                        cute.arch.sync_threads()
                        if tid == 0:
                            cute.arch.atomic_exch(
                                counter + tile_id, Int32(0), sem="release", scope="gpu"
                            )
            elif cutlass.const_expr(self.cluster and self.probe != "no_reduce"):
                cluster_sync()
                if split == 0:
                    for v in cutlass.range(tid, live_rows * self.nt // 4, self.warps * 32):
                        i = v * 4
                        sums = cute.make_rmem_tensor(4, Float32)
                        sums.fill(0.0)
                        for rank in cutlass.range_constexpr(self.splits):
                            v0, v1, v2, v3 = load_peer(result.iterator + i, rank)
                            sums[0], sums[1] = sums[0] + v0, sums[1] + v1
                            sums[2], sums[3] = sums[2] + v2, sums[3] + v3
                        for j in cutlass.range_constexpr(4):
                            result[i + j] = sums[j]
                    cute.arch.sync_threads()
                    self.epilogue(
                        result, g0, g1, ig, og, out, os, projection, postact, tid, n0, m0, bm
                    )
                # Peers must retain their shared allocation until rank zero is done.
                cluster_sync()
            elif cutlass.const_expr(self.probe != "no_reduce"):
                self.epilogue(result, g0, g1, ig, og, out, os, projection, postact, tid, n0, m0, bm)

    @cute.jit
    def epilogue(self, result, g0, g1, ig, og, out, os, projection, postact, tid, n0, m0, bm):
        for i in cutlass.range(tid, cutlass.min(16, self.rows - m0) * self.nt, self.warps * 32):
            rr, cc = i // self.nt, n0 + i % self.nt
            factor = Float32(0.0)
            if cc < self.n:
                if cutlass.const_expr(self.mode == 2):
                    factor = g0[Int64(cc)]
                elif cutlass.const_expr(self.mode == 1):
                    factor = g0[0] if cc % 2 == 0 else g1[0]
                else:
                    factor = g0[0]
            value = result[i] * factor * ig[0]
            if cutlass.const_expr(self.debug):
                if m0 + rr < self.rows and cc < self.n:
                    projection[Int64(m0 + rr) * self.n + cc] = value
            if cutlass.const_expr(self.gated):
                result[i] = value
            else:
                if m0 + rr < self.rows and cc < self.n:
                    out[Int64(m0 + rr) * self.n + cc] = value.to(cutlass.BFloat16)
        if cutlass.const_expr(self.gated):
            cute.arch.sync_threads()
            inv = reciprocal(og[0])
            live_rows = cutlass.min(16, self.rows - m0)
            for turn in cutlass.range(
                (live_rows * (self.nt // 2) + self.warps * 32 - 1) // (self.warps * 32)
            ):
                item = turn * (self.warps * 32) + tid
                rr, cc = item // (self.nt // 2), item % (self.nt // 2)
                col = n0 // 2 + cc
                vals = cute.make_rmem_tensor(1, Float32)
                vals.fill(0.0)
                if rr < live_rows and col < self.n // 2:
                    gate, up = result[rr * self.nt + 2 * cc], result[rr * self.nt + 2 * cc + 1]
                    # SGLang's activation: gate / (1 + expf(-gate)) * up.
                    denominator = Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                    vals[0] = divide(gate, denominator) * up
                    if cutlass.const_expr(self.debug):
                        postact[Int64(m0 + rr) * (self.n // 2) + col] = vals[0]
                if rr < live_rows:
                    word, sf = quant_lanes(vals, inv, 16)
                    if col < self.n // 2 and tid % 8 == 0:
                        ptr = cute.make_tensor(
                            cute.recast_ptr(out.iterator, dtype=Uint32),
                            cute.make_layout(self.rows * self.n // 16),
                        )
                        ptr[Int64(m0 + rr) * (self.n // 16) + col // 8] = word
                    if col < ((self.n // 2 + 63) // 64) * 64 and tid % 16 == 0:
                        os[scale_offset(m0 + rr, col // 16, (self.n // 2 + 63) // 64)] = sf
            # Tile ownership includes all padded output columns; only the first M tile
            # clears padded rows, so M=17..32 has no overlapping stores.
            if bm == 0:
                for i in cutlass.range(tid, 128 * self.nt // 32, self.warps * 32):
                    rr, cb = i // (self.nt // 32), i % (self.nt // 32)
                    col = n0 // 2 + cb * 16
                    if rr >= self.rows and col < ((self.n // 2 + 63) // 64) * 64:
                        os[scale_offset(rr, col // 16, (self.n // 2 + 63) // 64)] = Uint8(0)


_cache = {}
_lock = threading.Lock()


def launch(x, weight, input_scale, output_scale, config, workspace):
    if not available(x.device):
        raise RuntimeError("cute NVFP4 requires DSL 4.8.0 and sm120/sm121")
    m, n, k, _, gated, debug, mode, _ = workspace.signature
    key = (x.device, x.dtype, x.stride(0), workspace.signature, workspace.debug_timeout)
    dtype = cutlass.BFloat16 if x.dtype == torch.bfloat16 else cutlass.Float16
    tensors = (
        x,
        weight.data,
        weight.scale,
        weight.global0,
        weight.global1,
        input_scale,
        output_scale,
        workspace.out,
        workspace.out_scale,
        workspace.partial,
        workspace.counters,
        workspace.debug_data,
        workspace.debug_scale,
        workspace.projection,
        workspace.postact,
        workspace.errors,
        workspace.stamps,
        workspace.probe_counts,
    )
    types = (
        dtype,
        Uint8,
        Uint8,
        Float32,
        Float32,
        Float32,
        Float32,
        Uint8 if gated else cutlass.BFloat16,
        Uint8,
        Float32,
        Int32,
        Uint8,
        Uint8,
        Float32,
        Float32,
        Int32,
        Int64,
        Int32,
    )
    pointers = tuple(
        make_ptr(
            t,
            a.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16 if i in (0, 1, 2) else min(a.element_size(), 4),
        )
        for i, (a, t) in enumerate(zip(tensors, types, strict=False))
    )
    with torch.cuda.device(x.device):
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        compiled = _cache.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prewarm this cute shape/config/dtype/stride outside capture")
            with _lock:
                compiled = _cache.get(key)
                if compiled is None:
                    kernel = Kernel(
                        m, n, k, x.stride(0), config, gated, debug, mode, workspace.debug_timeout
                    )
                    arch = (
                        "sm_121a"
                        if torch.cuda.get_device_capability(x.device) == (12, 1)
                        else "sm_120a"
                    )
                    compiled = cute.compile(kernel, *pointers, stream, options=f"--gpu-arch {arch}")
                    _cache[key] = compiled
                    if os.environ.get("CTI_CUTE_SMALL_REPORT_COMPILE") == "1":
                        print(
                            "CUTE_COMPILED "
                            + json.dumps(
                                {
                                    "M": m,
                                    "N": n,
                                    "K": k,
                                    "gated": gated,
                                    "debug": debug,
                                    "debug_timeout": workspace.debug_timeout,
                                    "config": vars(config),
                                }
                            ),
                            flush=True,
                        )
        compiled(*pointers, stream)
        if workspace.debug_timeout and not torch.cuda.is_current_stream_capturing():
            workspace.check_debug()
