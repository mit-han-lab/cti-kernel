# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: FlashInfer 0.7 GDN CTA ownership; SGLang 72730d6 / FLA chunked WY;
# independently implemented, no upstream kernel code copied.

"""Serialized fused WY core with the gated RMSNorm in its epilogue, then a history commit.

By default (v2_snorm=0) each chunk's y is normalized and written by the fused kernel;
at vsplit > 1 the V slices of one (sequence, head) run as one cluster and read each
other's per-token sums of squares over DSMEM, in the separate norm kernel's order, so
y is bitwise the same as with v2_snorm=1 (FP32 o and partial sums in DRAM, then the v1
norm kernel). A small second kernel commits the staged conv history after every fused
CTA (each a halo reader) has retired.

Eight warps own 16/vsplit physical V rows each. m16n8k16 pads each warp's
row tile to 16: vsplit=2 discards the upper eight rows, vsplit=4 also masks
lane groups 4..7. Persistent H stores only the two live C components at
vsplit>1 (64/32/32 FP32 values per thread for vsplit=1/2/4). All eight warps
participate in the prologue and chain. This deliberately trades padded MMA
work for simple warp-local C-to-A reuse without shared H/r/D operands.

C and A for a single m16n8k16 warp have the same flattened element order
when pairs of C's eight-column tiles are joined into A's sixteen-column
tiles. _register_gemm exploits this public MMA layout; B uses ldmatrix.
The update's transposed B operand is read with scalar sk loads indexed by
partition_B(identity). v2_ldt=1 selects ldmatrix.trans instead, for
investigation only: on GPU it loads wrong B fragments (state off by its
own magnitude, RTX PRO 6000, 2026-10-15).
FP32 output staging reuses U; norm partials, slice sums and inverse norms reuse Q after
the chain. See v2_smem_layout for
the arena offsets/lifetimes; production allocates no global chunk intermediates.

v2_debug=1 dumps local chunk zero to Launcher.last_debug, FP32
[N*Hv*vsplit, 14, 64, 128]. CTA index is (sequence*Hv+head)*vsplit+part.
Slots: Q_h, K_h, g/beta (rows 0/1), L, A_h, S, W, U, r, scaled o,
H1[0:64], H1[64:128], B_h, B_l. B/U/r/o use slice-local V columns; H1 uses slice-local
V rows and K columns. Unused cells stay zero. All instrumentation is constexpr
elided at v2_debug=0; the unused v1 op scratch argument carries the debug buffer.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T
from cutlass.utils import SmemAllocator

from ._cute_prefill import v2_smem_layout
from ._cute_prefill_kernels import (
    Prefill,
    _block,
    _exp,
    _gemm,
    _layout,
    _load_vector,
    _mma,
    _publish,
    _silu,
    _small_mma,
    _transpose,
)


# Public PTX ISA: cp.async src-size zero fill and rcp.approx.ftz.f32.
# https://docs.nvidia.com/cuda/parallel-thread-execution/
def _copy_raw16(dst, projected, history, use_history, valid):
    """One vector copy, including invalid rows; address selection uses selp."""
    llvm.inline_asm(
        None,
        [
            Int32(dst.toint()).ir_value(),
            projected.toint().ir_value(),
            history.toint().ir_value(),
            Int32(use_history).ir_value(),
            Int32(valid).ir_value(),
        ],
        "{ .reg .pred h, v; .reg .b64 src; .reg .b32 size; "
        "setp.ne.s32 h, $3, 0; setp.ne.s32 v, $4, 0; "
        "selp.b64 src, $2, $1, h; selp.b32 size, 16, 0, v; "
        "cp.async.cg.shared.global [$0], [src], 16, size; }",
        "r,l,l,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# Public PTX ISA: mapa and ld.shared::cluster (distributed shared memory).
def _load_cluster(address, rank):
    """FP32 load from the same shared offset in cluster CTA `rank`."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(address).ir_value(), Int32(rank).ir_value()],
            "{ .reg .b32 a; mapa.shared::cluster.u32 a, $1, $2; ld.shared::cluster.f32 $0, [a]; }",
            "=f,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _silu_fast(x, valid):
    # Reciprocal input is in [1,2]; keep the stable negative-x numerator.
    t = _exp(-abs(x))
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value(), t.ir_value(), Int32(valid).ir_value()],
            "{ .reg .f32 d, s, n; .reg .pred neg, live; "
            "add.f32 d, 1.0, $2; rcp.approx.ftz.f32 s, d; "
            "setp.lt.f32 neg, $1, 0.0; selp.f32 n, $2, 1.0, neg; "
            "mul.f32 s, s, n; mul.f32 s, $1, s; "
            "setp.ne.s32 live, $3, 0; selp.f32 $0, s, 0.0, live; }",
            "=f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _diagonal_mma(mma, a, b, acc):
    """FP32 operands as TF32 hi/lo; omit lo*lo at about FP32 error scale."""
    tid, _, _ = cute.arch.thread_idx()
    thread = mma.get_slice(tid % 32)
    pa, pb = thread.partition_A(a), thread.partition_B(b)
    ah, al = (thread.make_fragment_A(pa) for _ in range(2))
    bh, bl = (thread.make_fragment_B(pb) for _ in range(2))
    for e in cutlass.range_constexpr(cute.size(pa)):
        value = Float32(pa[e])
        ah[e] = Int32(cute.arch.cvt_f32_tf32(value))
        al[e] = Int32(cute.arch.cvt_f32_tf32(value - ah[e].bitcast(Float32)))
    for e in cutlass.range_constexpr(cute.size(pb)):
        value = Float32(pb[e])
        bh[e] = Int32(cute.arch.cvt_f32_tf32(value))
        bl[e] = Int32(cute.arch.cvt_f32_tf32(value - bh[e].bitcast(Float32)))
    for kk in cutlass.range_constexpr(2):
        cute.gemm(mma, acc, al[None, None, kk], bh[None, None, kk], acc)
        cute.gemm(mma, acc, ah[None, None, kk], bl[None, None, kk], acc)
        cute.gemm(mma, acc, ah[None, None, kk], bh[None, None, kk], acc)


@cute.jit
def _inverse_blocked_mma(lower, inverse, workspace):
    """Four concurrent 16x16 solves, recursively joining 2/4/8/16 blocks.

    For [[A,0],[B,D]], the new inverse block is -D^-1 B A^-1.
    Seed the 2x2 unit-lower inverses, then use two split MMAs per level.
    Unlike explicit N powers this avoids large cancelling intermediates for
    correlated keys. Diagonal products use TF32 hi/lo for FP32 accuracy.
    Keep v1's four-term BF16 products and ordering for off-diagonal blocks.
    Temporaries use the existing workspace; L and its debug slot survive.
    """
    tid, _, _ = cute.arch.thread_idx()
    lane, wid = tid % 32, tid // 32
    for e in range(tid, 4096, 256):
        inverse[e // 64, e % 64] = Float32(0.0)
    cute.arch.sync_threads()
    mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
        cute.make_layout((1, 1, 1)),
        permutation_mnk=(16, 16, 16),
    )
    coords = mma.get_slice(lane).partition_C(cute.make_identity_tensor((16, 16)))
    acc = cute.make_rmem_tensor(coords.shape, Float32)
    tmp = cute.make_tensor(
        cute.recast_ptr(workspace.iterator, dtype=Float32) + wid * 256,
        cute.make_layout((16, 16), stride=(16, 1)),
    )
    if wid < 4:
        dmma = cute.make_tiled_mma(
            warp.MmaTF32Op((16, 8, 8)),
            cute.make_layout((1, 1, 1)),
            permutation_mnk=(16, 16, 8),
        )
        diag = _block(inverse, wid, wid)
        for e in cutlass.range_constexpr(cute.size(coords)):
            r, c = coords[e]
            value = Float32(r == c)
            if r == c + 1 and r // 2 == c // 2:
                value = -lower[wid * 16 + r, wid * 16 + c]
            diag[r, c] = value
        cute.arch.sync_warp()
        for level in cutlass.range_constexpr(1, 4):
            # S is block diagonal with blocks of width span at entry.
            # B contains only each pair's bottom-left original L block.
            span = 1 << level
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                value = Float32(0.0)
                if (
                    r // (2 * span) == c // (2 * span)
                    and r % (2 * span) >= span
                    and c % (2 * span) < span
                ):
                    value = lower[wid * 16 + r, wid * 16 + c]
                tmp[r, c] = value
            cute.arch.sync_warp()
            acc.fill(0.0)
            _diagonal_mma(dmma, diag, _transpose(tmp), acc)
            cute.arch.sync_warp()
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                tmp[r, c] = -acc[e]
                acc[e] = diag[r, c]
            cute.arch.sync_warp()
            _diagonal_mma(dmma, tmp, _block(inverse, wid, wid, True), acc)
            cute.arch.sync_warp()
            _publish(acc, coords, diag)
            cute.arch.sync_warp()
    cute.arch.sync_threads()
    # The v1 off-diagonal blocked MMA path, with the same product/order.
    for distance in cutlass.range_constexpr(1, 4):
        if wid < 4 - distance:
            i, j = wid + distance, wid
            acc.fill(0.0)
            for offset in cutlass.range_constexpr(1, distance + 1):
                _small_mma(
                    mma,
                    _block(inverse, i, j + offset),
                    _block(lower, j + offset, j, True),
                    acc,
                    True,
                )
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                tmp[r, c] = -acc[e]
            cute.arch.sync_warp()
            acc.fill(0.0)
            _small_mma(mma, tmp, _block(inverse, j, j, True), acc, True)
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                inverse[i * 16 + r, j * 16 + c] = acc[e]
        cute.arch.sync_threads()


@cute.jit
def _tile(
    arena,
    offset: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    swizzle: cutlass.Constexpr = True,
):
    ptr = cute.recast_ptr(arena.iterator + offset, dtype=dtype)
    layout = cute.make_layout((m, n), stride=(n, 1))
    if const_expr(swizzle):
        layout = _layout(m, n)
    return cute.make_tensor(ptr, layout)


@cute.jit
def _debug_tile(dbg, block, slot: cutlass.Constexpr, tile, coords, column: cutlass.Constexpr = 0):
    """Use the consumer's C partition, including swizzled shared tile addressing."""
    for e in cutlass.range_constexpr(cute.size(coords)):
        i, j = coords[e]
        dbg[((Int64(block) * 14 + slot) * 64 + i) * 128 + column + j] = Float32(tile[i, j])
    # Debug reads must retire before any following arena alias is published.
    cute.arch.sync_threads()


@cute.jit
def _register_gemm(
    mma,
    left,
    right,
    acc,
    columns: cutlass.Constexpr,
    compact: cutlass.Constexpr = False,
    transpose: cutlass.Constexpr = False,
    scalar_key=None,
):
    """Register C[16,K] -> BF16 A, shared B[N,K]; FP32 accumulation.

    For compact H, each C N=8 tile retains c0/c1 only; c2/c3 are zero.
    Full residual fragments retain all four components (padding stays zero).
    scalar_key supplies the original swizzled K[T,K] tile for the update's
    scalar B-load ablation, bypassing both the transposed view and ldmatrix.
    """
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    thread = mma.get_slice(lane)
    # Identity partitions give the same shape as a shared A without allocating it.
    pa = thread.partition_A(cute.make_identity_tensor((16, columns)))
    ra = thread.make_fragment_A(pa.shape)
    # For this single-warp chain, partition_A(identity)[e] equals
    # partition_C(identity)[e]: (lane//4 + 8*(e//2%2),
    # 2*(lane%4) + e%2 + 8*(e//4)). Compact H drops the row+8 pair.
    # CPU identity-partition tests verify this for every lane and K=64/128.
    for e in cutlass.range_constexpr(cute.size(ra)):
        value = Float32(0.0)
        if const_expr(compact):
            if const_expr(e % 4 < 2):
                value = left[(e // 4) * 2 + e % 4]
        else:
            value = left[e]
        ra[e] = BFloat16(value)
    if const_expr(scalar_key is not None):
        cb = thread.partition_B(cute.make_identity_tensor((128, 64)))
        rb = thread.make_fragment_B(cb.shape)
    else:
        rb = thread.make_fragment_B(thread.partition_B(right))
        copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=2), BFloat16
            ),
            mma,
        )
        copier = copy.get_slice(lane)
        source, dest = copier.partition_S(right), copier.retile(rb)
    for kk in cutlass.range_constexpr(columns // 16):
        if const_expr(scalar_key is not None):
            bk, ck = rb[None, None, kk], cb[None, None, kk]
            for e in cutlass.range_constexpr(cute.size(bk)):
                ki, ti = ck[e]
                bk[e] = scalar_key[ti, ki]
        else:
            cute.copy(copy, source[None, None, kk], dest[None, None, kk])
        cute.gemm(mma, acc, ra[None, None, kk], rb[None, None, kk], acc)


class FusedPrefill(Prefill):
    def __init__(self, hk, hv, stride, tune, captures=0):
        self.memory = v2_smem_layout(hk, hv, tune["vsplit"])
        self.aligned_rows = stride % 8 == 0
        self.vsplit = tune["vsplit"]
        self.vs = 128 // self.vsplit
        self.rows = 16 // self.vsplit
        self.h_components = 4 if self.vsplit == 1 else 2
        self.debug = bool(tune.get("v2_debug", 0))
        self.kt = not tune.get("v2_ldt", 0)  # ldmatrix.trans loads wrong B fragments (GPU-verified)
        # Only inherited norm uses overlap=True: it commits staged history. The
        # fused launch itself is ordinary and has no PDL, flags or device waits.
        inherited = tune | {"overlap": 1, "bv": 32, "prepare_warps": 8}
        super().__init__(hk, hv, stride, inherited, captures=captures)
        # Fused CTAs export history before this capture-free norm commits slots.
        self.commit = Prefill(hk, hv, stride, inherited)
        self.aq_qk = hv // hk == 3
        # Gated RMSNorm in the fused epilogue (v2_snorm=0, default): the V slices of
        # one (sequence, head) form a cluster and pull each other's per-token sums
        # over DSMEM, so y is written directly and no FP32 o round-trips DRAM.
        self.fnorm = not tune.get("v2_snorm", 0)
        self.cluster = self.vsplit if self.fnorm else 1

    @cute.jit
    def conv_rows_v2(
        self,
        x,
        cw,
        history,
        raw,
        dst,
        start,
        begin,
        end,
        channel,
        slot,
        flag,
        normalize: cutlass.Constexpr = False,
        scale=None,
        extra_shared=None,
        columns: cutlass.Constexpr = 128,
    ):
        # Channel offsets are multiples of 32 (V slices) or 128 (Q/K/heads).
        # The compile key already contains stride. Misaligned base views are
        # legal too, so guard them once uniformly, outside the vector loops.
        if const_expr(self.aligned_rows):
            if x.toint() % 16 == 0 and history.toint() % 16 == 0:
                self.conv_aligned_v2(
                    x,
                    cw,
                    history,
                    raw,
                    dst,
                    start,
                    begin,
                    end,
                    channel,
                    slot,
                    flag,
                    normalize,
                    scale,
                    extra_shared,
                    columns,
                )
            else:
                self.conv_rows(
                    x,
                    cw,
                    history,
                    raw,
                    dst,
                    start,
                    begin,
                    end,
                    channel,
                    slot,
                    flag,
                    normalize,
                    scale=scale,
                    extra_shared=extra_shared,
                    columns=columns,
                )
        else:
            self.conv_rows(
                x,
                cw,
                history,
                raw,
                dst,
                start,
                begin,
                end,
                channel,
                slot,
                flag,
                normalize,
                scale=scale,
                extra_shared=extra_shared,
                columns=columns,
            )

    @cute.jit
    def conv_aligned_v2(
        self,
        x,
        cw,
        history,
        raw,
        dst,
        start,
        begin,
        end,
        channel,
        slot,
        flag,
        normalize: cutlass.Constexpr,
        scale,
        extra_shared,
        columns: cutlass.Constexpr,
    ):
        """Four consecutive rows x eight channels per active thread.

        256/128/64 threads cover widths 128/64/32; full warps stay active.
        Seven raw vectors per thread replace sixteen: three halo rows plus
        one new row for each output. The four-slot register ring is indexed
        only at compile time. Each thread loads each of its raw rows once.
        """
        assert columns in (32, 64, 128) and self.dc % 8 == 0
        assert self.stride % 8 == 0 and self.vs % 8 == 0
        tid, _, _ = cute.arch.thread_idx()
        for v in range(tid, 67 * (columns // 8), 256):
            row, col = v // (columns // 8), (v % (columns // 8)) * 8
            token = start - 3 + row
            use_history = (token < begin) & (slot >= 0) & (flag != 0)
            valid = ((token >= begin) & (token < end)) | use_history
            _copy_raw16(
                raw.iterator + raw.layout((row, col)),
                x + token * self.stride + channel + col,
                history + (slot * 3 + token - begin + 3) * self.dc + channel + col,
                use_history,
                valid,
            )
        cute.arch.cp_async_commit_group()
        packed_weights = cute.make_rmem_tensor((32,), BFloat16)
        weights = cute.make_rmem_tensor((32,), Float32)
        _load_vector(cw, (channel + (tid % (columns // 8)) * 8) * 4, packed_weights)
        for j in cutlass.range_constexpr(32):
            weights[j] = Float32(packed_weights[j])
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        if tid < columns * 2:
            first, col = (tid // (columns // 8)) * 4, (tid % (columns // 8)) * 8
            ring = cute.make_rmem_tensor((8, 4), Float32)
            samples = cute.make_rmem_tensor((8,), BFloat16)
            for tap in cutlass.range_constexpr(3):
                source = cute.make_tensor(
                    (raw.iterator + raw.layout((first + tap, col))).align(16), cute.make_layout(8)
                )
                cute.copy(atom, source, samples)
                for j in cutlass.range_constexpr(8):
                    ring[j, tap] = Float32(samples[j])
            for step in cutlass.range_constexpr(4):
                row = first + step
                source = cute.make_tensor(
                    (raw.iterator + raw.layout((row + 3, col))).align(16), cute.make_layout(8)
                )
                cute.copy(atom, source, samples)
                for j in cutlass.range_constexpr(8):
                    ring[j, (step + 3) % 4] = Float32(samples[j])
                values = cute.make_rmem_tensor((8,), Float32)
                packed = cute.make_rmem_tensor((8,), BFloat16)
                values.fill(0.0)
                for tap in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(8):
                        values[j] += ring[j, (step + tap) % 4] * weights[j * 4 + tap]
                square = Float32(0.0)
                for j in cutlass.range_constexpr(8):
                    values[j] = _silu_fast(values[j], start + row < end)
                    square += values[j] * values[j]
                factor = Float32(1.0)
                if const_expr(normalize):
                    for shift in cutlass.range_constexpr(4):
                        square += cute.arch.shuffle_sync_bfly(square, offset=1 << shift)
                    factor = cute.rsqrt(square + 1.0e-6)
                if const_expr(scale is not None):
                    factor *= scale[row]
                for j in cutlass.range_constexpr(8):
                    value = values[j] * factor
                    packed[j] = BFloat16(value)
                    if const_expr(extra_shared is not None):
                        extra_shared[row, col + j] = BFloat16(value - Float32(packed[j]))
                target = cute.make_tensor(
                    (dst.iterator + dst.layout((row, col))).align(16), cute.make_layout(8)
                )
                cute.copy(atom, packed, target)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cw: cute.Tensor,
        alog: cute.Tensor,
        bias: cute.Tensor,
        nw: cute.Tensor,
        cu: cute.Tensor,
        slots: cute.Tensor,
        initial: cute.Tensor,
        state: cute.Tensor,
        history: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        w: cute.Tensor,
        u: cute.Tensor,
        aq: cute.Tensor,
        g: cute.Tensor,
        hout: cute.Tensor,
        out: cute.Tensor,
        partial: cute.Tensor,
        y: cute.Tensor,
        kk: cute.Tensor,
        qk: cute.Tensor,
        kl: cute.Tensor,
        ready: cute.Tensor,
        qp: cute.Tensor,
        op: cute.Tensor,
        capture_offsets: cute.Tensor,
        capture_rows: cute.Tensor,
        hbuf: cute.Tensor,
        vbuf: cute.Tensor,
        stream: cuda.CUstream,
    ):
        count = Int32(slots.shape[0])
        self.fused_v2(
            x.iterator,
            cw.iterator,
            alog.iterator,
            bias.iterator,
            cu.iterator,
            slots.iterator,
            initial.iterator,
            state.iterator,
            history.iterator,
            hout.iterator,
            out.iterator,
            partial.iterator,
            nw.iterator,
            y.iterator,
            capture_offsets.iterator if const_expr(self.captures) else None,
            capture_rows.iterator if const_expr(self.captures) else None,
            op.iterator if const_expr(self.debug) else None,
        ).launch(
            grid=(count * self.hv * self.vsplit, 1, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=False,
            preferred_smem_carveout=100,
            **({"cluster": (self.cluster, 1, 1)} if self.cluster > 1 else {}),
        )
        if const_expr(self.fnorm):
            # History commit only: every fused CTA (each a halo reader) has retired.
            self.commit_v2(
                cu.iterator, slots.iterator, initial.iterator, history.iterator, hout.iterator
            ).launch(
                grid=(count, self.hv, 1),
                block=(128, 1, 1),
                stream=stream,
                use_pdl=False,
            )
        else:
            if x.shape[0] > 0:
                self.commit.norm(
                    x.iterator,
                    nw.iterator,
                    out.iterator,
                    partial.iterator,
                    y.iterator,
                    x.shape[0],
                    count,
                    cu.iterator,
                    slots.iterator,
                    history.iterator,
                    hout.iterator,
                    initial.iterator,
                    None,
                    None,
                ).launch(
                    grid=((x.shape[0] + 7) // 8, self.hv, 1),
                    block=(128, 1, 1),
                    stream=stream,
                    use_pdl=False,
                    preferred_smem_carveout=100,
                )
            else:
                # History commit only: every fused CTA (each a halo reader) has retired.
                self.commit_v2(
                    cu.iterator, slots.iterator, initial.iterator, history.iterator, hout.iterator
                ).launch(
                    grid=(count, self.hv, 1),
                    block=(128, 1, 1),
                    stream=stream,
                    use_pdl=False,
                )

    @cute.jit
    def norm_epilogue(self, x, nw, y, su, norms, arena, start, end, head, part, vbase):
        """Gated RMSNorm of one chunk; bitwise the separate v1 norm kernel's arithmetic.

        Per token: four 32-column sums (each over its warps in order) are added in
        column order from zero, then y = ((o * rsqrt(mean + eps)) * w) * silu(z).
        With vsplit > 1, every CTA of the cluster reads all four sums through DSMEM
        between two cluster barriers; the second one is waited for at the next
        chunk (or at exit), before this CTA's Q region is overwritten.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane, wid = tid % 32, tid // 32
        groups = self.vs // 32
        # Q is dead here: [0, 2048) holds the warp sums, then 64 x groups slice
        # sums, then 64 inverse norms.
        sums = cute.make_tensor(
            cute.recast_ptr(arena.iterator + 2048, dtype=Float32),
            cute.make_layout((groups, 64), stride=(64, 1)),
        )
        inverse = cute.make_tensor(
            cute.recast_ptr(arena.iterator + 3072, dtype=Float32), cute.make_layout(64)
        )
        # Issue every gate and weight load first: a load placed after a y store
        # cannot be hoisted above it (the pointers may alias), which would make each
        # one a serialized DRAM round trip. Rows are clamped to valid tokens.
        gate = cute.make_rmem_tensor((8 * groups,), Float32)
        weight = cute.make_rmem_tensor((groups,), Float32)
        last = end - 1
        for group in cutlass.range_constexpr(groups):
            weight[group] = nw[vbase + group * 32 + lane]
        for r in cutlass.range_constexpr(8):
            source = cutlass.min(start + wid + 8 * r, last)
            for group in cutlass.range_constexpr(groups):
                column = vbase + group * 32 + lane
                gate[r * groups + group] = Float32(
                    x[source * self.stride + self.dc + head * 128 + column]
                )
        if tid < 64 * groups:
            row, group = tid // groups, tid % groups
            square = Float32(0.0)
            for w in cutlass.range_constexpr(2 * self.vsplit):
                square += norms[group * (2 * self.vsplit) + w, row]
            sums[group, row] = square
        if const_expr(self.cluster > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        if tid < 64:
            total = Float32(0.0)
            for g in cutlass.range_constexpr(4):
                if const_expr(self.cluster > 1):
                    address = Int32((sums.iterator + (g % groups) * 64 + tid).toint())
                    total += _load_cluster(address, g // groups)
                else:
                    total += sums[g, tid]
            inverse[tid] = cute.rsqrt(total * (1.0 / 128.0) + 1.0e-6)
        if const_expr(self.cluster > 1):
            cute.arch.cluster_arrive()
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(8):
            row = wid + 8 * r
            token = start + row
            inv = inverse[row]
            for group in cutlass.range_constexpr(groups):
                value = su[row, group * 32 + lane] * inv * weight[group]
                value = value * _silu(gate[r * groups + group])
                if token < end:
                    y[(token * self.hv + head) * 128 + vbase + group * 32 + lane] = BFloat16(value)

    @cute.kernel
    def commit_v2(
        self, cu: cute.Pointer, slots: cute.Pointer, initial: cute.Pointer, history, hout
    ):
        seq, head, _ = cute.arch.block_idx()
        slot = Int64(slots[seq])
        if slot >= 0 and cu[seq + 1] == cu[seq] and initial[seq] == 0:
            self.zero_history(history, slot, head, 128)
        if slot >= 0 and cu[seq + 1] > cu[seq]:
            self.history_commit(history, hout, seq, slot, (2 * self.hk + head) * 128, 128)
            if head % (self.hv // self.hk) == 0:
                kh = head // (self.hv // self.hk)
                for which in cutlass.range_constexpr(2):
                    self.history_commit(history, hout, seq, slot, (which * self.hk + kh) * 128, 128)

    @cute.kernel
    def fused_v2(
        self,
        x: cute.Pointer,
        cw: cute.Pointer,
        alog: cute.Pointer,
        bias: cute.Pointer,
        cu: cute.Pointer,
        slots: cute.Pointer,
        initial: cute.Pointer,
        state: cute.Pointer,
        history: cute.Pointer,
        hout: cute.Pointer,
        out: cute.Pointer,
        partial: cute.Pointer,
        nw: cute.Pointer,
        y: cute.Pointer,
        offsets,
        rows,
        dbg=None,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, wid = tid % 32, tid // 32
        part = block % self.vsplit
        head = (block // self.vsplit) % self.hv
        seq = block // (self.vsplit * self.hv)
        kh = head // (self.hv // self.hk)
        begin, end = Int64(cu[seq]), Int64(cu[seq + 1])
        slot, flag = Int64(slots[seq]), initial[seq]
        vbase = part * self.vs
        alloc = SmemAllocator()
        arena = alloc.allocate_tensor(
            cutlass.Int8, cute.make_layout(self.memory["bytes"]), byte_alignment=128
        )
        sq = _tile(arena, 0, BFloat16, 64, 128)
        sk = _tile(arena, 16384, BFloat16, 64, 128)
        sa = _tile(arena, 32768, BFloat16, 64, 64)
        lower = _tile(arena, 40960, Float32, 64, 64, False)
        inverse = _tile(arena, 57344, Float32, 64, 64, False)
        workspace = _tile(arena, 73728, BFloat16, 64, 64)
        si = _tile(arena, self.memory["s_hi"], BFloat16, 64, 64)
        sw = _tile(arena, 40960, BFloat16, 64, 128)
        # Standalone strip layouts, matching v1's rh/rl (no local_tile of a
        # composed 64x128 swizzle for convolution or transposed ldmatrix).
        bcols = min(64, self.vs)
        strips = (self.vs + 63) // 64
        bh = _tile(arena, 57344, BFloat16, 64, bcols)
        bl = _tile(arena, 57344 + 128 * bcols, BFloat16, 64, bcols)
        su = _tile(arena, 57344, Float32, 64, self.vs, False)
        sg = cute.make_tensor(
            cute.recast_ptr(arena.iterator + self.memory["arena"], dtype=Float32),
            cute.make_layout(64),
        )
        sb = cute.make_tensor(sg.iterator + 64, cute.make_layout(64))
        mma = _mma(8, 32, 32)
        thread = mma.get_slice(tid)
        cg = thread.partition_C(cute.make_identity_tensor((64, 64)))
        cmw = thread.partition_C(cute.make_identity_tensor((64, 128)))
        cmu = thread.partition_C(cute.make_identity_tensor((64, self.vs)))
        cbs = thread.partition_C(cute.make_identity_tensor((64, bcols)))
        # One warp MMA with no N permutation: paired C tiles map directly to A.
        chain = _mma(1, 16, 8)
        ct = chain.get_slice(lane)
        ch = ct.partition_C(cute.make_identity_tensor((16, 128)))
        cr = ct.partition_C(cute.make_identity_tensor((16, 64)))
        # ch has shape ((2,2),1,16). Flat 4*n+c is
        # (vi,ki)=(lane//4+8*(c//2), 8*n+2*(lane%4)+c%2).
        # Thus c=0,1 are exactly the compact lower-eight-row components;
        # vsplit=4 additionally masks vi>=4. The same ch coordinates drive
        # initial state, updated C, H1 debug slots and final state below.
        h = cute.make_rmem_tensor((self.h_components * 16,), Float32)
        h.fill(0.0)
        for n in cutlass.range_constexpr(16):
            for c in cutlass.range_constexpr(self.h_components):
                vi, ki = ch[n * 4 + c]
                if vi < self.rows and slot >= 0 and flag != 0:
                    h[n * self.h_components + c] = state[
                        ((slot * self.hv + head) * 128 + vbase + wid * self.rows + vi) * 128 + ki
                    ]
        # Exactly one writer per history channel. Norm commits only after EVERY
        # fused CTA has retired, so sibling and V-slice halo reads cannot race.
        if const_expr(self.captures):
            if part == 0:
                self.capture_history(
                    x, history, offsets, rows, initial, seq, slot, begin, end, head, 256
                )
        if end > begin and part == 0 and slot >= 0:
            self.history_stage(
                x, history, hout, begin, end, slot, flag, seq, (2 * self.hk + head) * 128
            )
            if head % (self.hv // self.hk) == 0:
                self.history_stage(x, history, hout, begin, end, slot, flag, seq, kh * 128)
                self.history_stage(
                    x, history, hout, begin, end, slot, flag, seq, (self.hk + kh) * 128
                )
        for local in range(Int32((end - begin + 63) // 64)):
            start = begin + Int64(local) * 64
            if const_expr(self.cluster > 1):
                # Peers have read this CTA's previous norm sums (Q region) before
                # the convolution below overwrites Q.
                if local > 0:
                    cute.arch.cluster_wait()
            self.gates(x, alog, bias, sg, sb, start, end, head)
            raw = _tile(arena, 65536 if self.aq_qk else 32768, BFloat16, 68, 128, False)
            if const_expr(self.aq_qk):
                ql = _tile(arena, 32768, BFloat16, 64, 128)
                klow = _tile(arena, 49152, BFloat16, 64, 128)
                self.conv_rows_v2(
                    x,
                    cw,
                    history,
                    raw,
                    sq,
                    start,
                    begin,
                    end,
                    kh * 128,
                    slot,
                    flag,
                    True,
                    extra_shared=ql,
                )
                self.conv_rows_v2(
                    x,
                    cw,
                    history,
                    raw,
                    sk,
                    start,
                    begin,
                    end,
                    (self.hk + kh) * 128,
                    slot,
                    flag,
                    True,
                    extra_shared=klow,
                )
            else:
                self.conv_rows_v2(
                    x, cw, history, raw, sq, start, begin, end, kh * 128, slot, flag, True
                )
                self.conv_rows_v2(
                    x,
                    cw,
                    history,
                    raw,
                    sk,
                    start,
                    begin,
                    end,
                    (self.hk + kh) * 128,
                    slot,
                    flag,
                    True,
                )
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 0, sq, cmw)
                    _debug_tile(dbg, block, 1, sk, cmw)
                    if tid < 64:
                        dbg[(Int64(block) * 14 + 2) * 8192 + tid] = sg[tid]
                        dbg[(Int64(block) * 14 + 2) * 8192 + 128 + tid] = sb[tid]
                    cute.arch.sync_threads()
            gram = cute.make_rmem_tensor(cg.shape, Float32)
            qk = cute.make_rmem_tensor(cg.shape, Float32)
            gram.fill(0.0)
            qk.fill(0.0)
            _gemm(mma, sk, sk, gram)
            _gemm(mma, sq, sk, qk)
            if const_expr(self.aq_qk):
                _gemm(mma, ql, sk, qk)
                _gemm(mma, sq, klow, qk)
            # G/J remain FP32 registers. Barrier retires low Q/K/raw before L/A
            # overwrite them; arena reserves their full FP32 tiles for budgeting.
            cute.arch.sync_threads()
            for e in cutlass.range_constexpr(cute.size(cg)):
                i, j = cg[e]
                lv, av = Float32(0.0), Float32(0.0)
                if i > j:
                    lv = gram[e] * sb[i] * _exp(sg[i] - sg[j])
                if i >= j:
                    av = qk[e] * _exp(sg[i] - sg[j])
                lower[i, j], sa[i, j] = lv, BFloat16(av)
            cute.arch.sync_threads()
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 3, lower, cg)
                    _debug_tile(dbg, block, 4, sa, cg)
            _inverse_blocked_mma(lower, inverse, workspace)
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 5, inverse, cg)
            # Stage S in registers before publishing S_h (including alias safety).
            saved = cute.make_rmem_tensor(cg.shape, BFloat16)
            for e in cutlass.range_constexpr(cute.size(cg)):
                i, j = cg[e]
                saved[e] = BFloat16(inverse[i, j])
            cute.arch.sync_threads()
            _publish(saved, cg, si)
            cute.arch.sync_threads()
            # V raw fits inside retired L/E/W. Each strip uses its own 64x64
            # (64x32 for vsplit=4) MMA accumulator and the same standalone B pair.
            # Keep all results in registers until the B/U alias is retired:
            # publishing strip 0 early would be clobbered by strip 1's B stores.
            vr = _tile(arena, 40960, BFloat16, 68, bcols, False)
            up = cute.make_rmem_tensor((*cbs.shape, strips), Float32)
            up.fill(0.0)
            for strip in cutlass.range_constexpr(strips):
                self.conv_rows_v2(
                    x,
                    cw,
                    history,
                    vr,
                    bh,
                    start,
                    begin,
                    end,
                    (2 * self.hk + head) * 128 + vbase + strip * 64,
                    slot,
                    flag,
                    scale=sb,
                    extra_shared=bl,
                    columns=bcols,
                )
                if const_expr(self.debug):
                    if local == 0:
                        _debug_tile(dbg, block, 12, bh, cbs, strip * 64)
                        _debug_tile(dbg, block, 13, bl, cbs, strip * 64)
                _gemm(mma, si, _transpose(bh), up[None, None, None, strip], tb=True)
                _gemm(mma, si, _transpose(bl), up[None, None, None, strip], tb=True)
                # All ldmatrix reads must retire before reusing B or V raw.
                # This is required even when the diagnostic stores are absent.
                cute.arch.sync_threads()
            for strip in cutlass.range_constexpr(strips):
                # U is ordinary row-major FP32; strip stores retain its Vs stride.
                us = cute.local_tile(su, (64, bcols), (0, strip))
                _publish(up[None, None, None, strip], cbs, us)
            # Beta has retired after U; publish E row factors with U's existing
            # barrier, so hoisting adds no CTA synchronization.
            if tid < 64:
                sb[tid] *= _exp(sg[tid])
            cute.arch.sync_threads()
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 7, su, cmu)
            # E -> W in place: all E reads finish before any W publication.
            for e in range(tid, 8192, 256):
                i, j = e // 128, e % 128
                sw[i, j] = BFloat16(Float32(sk[i, j]) * sb[i])
            cute.arch.sync_threads()
            wp = cute.make_rmem_tensor(cmw.shape, Float32)
            wp.fill(0.0)
            _gemm(mma, si, _transpose(sw), wp, tb=True)
            cute.arch.sync_threads()
            _publish(wp, cmw, sw)
            cute.arch.sync_threads()
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 6, sw, cmw)
            # Transposed chain; no shared H, r or D. Every warp owns its V rows.
            r = cute.make_rmem_tensor(cr.shape, Float32)
            o = cute.make_rmem_tensor(cr.shape, Float32)
            r.fill(0.0)
            _register_gemm(chain, h, sw, r, 128, self.vsplit > 1)
            for e in cutlass.range_constexpr(cute.size(cr)):
                vi, ti = cr[e]
                value = Float32(0.0)
                if vi < self.rows:
                    value = su[ti, wid * self.rows + vi] - r[e]
                r[e] = value
            if const_expr(self.debug):
                if local == 0:
                    for e in cutlass.range_constexpr(cute.size(cr)):
                        vi, ti = cr[e]
                        if vi < self.rows:
                            dbg[
                                ((Int64(block) * 14 + 8) * 64 + ti) * 128 + wid * self.rows + vi
                            ] = r[e]
                    cute.arch.sync_threads()
            o.fill(0.0)
            _register_gemm(chain, h, sq, o, 128, self.vsplit > 1)
            for e in cutlass.range_constexpr(cute.size(cr)):
                vi, ti = cr[e]
                o[e] *= _exp(sg[ti])
            _register_gemm(chain, r, sa, o, 64)
            updated = cute.make_rmem_tensor(ch.shape, Float32)
            # FP32 residual survives output rounding; D is rounded AFTER decay.
            for e in cutlass.range_constexpr(cute.size(cr)):
                vi, ti = cr[e]
                r[e] *= _exp(sg[63] - sg[ti])
            updated.fill(0.0)
            for n in cutlass.range_constexpr(16):
                for c in cutlass.range_constexpr(self.h_components):
                    updated[n * 4 + c] = h[n * self.h_components + c] * _exp(sg[63])
            _register_gemm(
                chain,
                r,
                _transpose(sk),
                updated,
                64,
                transpose=True,
                scalar_key=sk if const_expr(self.kt) else None,
            )
            for n in cutlass.range_constexpr(16):
                for c in cutlass.range_constexpr(self.h_components):
                    h[n * self.h_components + c] = updated[n * 4 + c]
            if const_expr(self.captures):
                for capture in cutlass.range_constexpr(self.captures):
                    p = offsets[Int64(seq) * self.captures + capture]
                    target = Int64(rows[Int64(seq) * self.captures + capture])
                    if target >= 0 and p > 0 and p <= end - begin and p == (Int64(local) + 1) * 64:
                        for n in cutlass.range_constexpr(16):
                            for c in cutlass.range_constexpr(self.h_components):
                                vi, ki = ch[n * 4 + c]
                                if vi < self.rows:
                                    state[
                                        (
                                            (target * self.hv + head) * 128
                                            + vbase
                                            + wid * self.rows
                                            + vi
                                        )
                                        * 128
                                        + ki
                                    ] = h[n * self.h_components + c]
            if const_expr(self.debug):
                if local == 0:
                    for n in cutlass.range_constexpr(16):
                        for c in cutlass.range_constexpr(self.h_components):
                            vi, ki = ch[n * 4 + c]
                            if vi < self.rows:
                                vrow = wid * self.rows + vi
                                dbg[
                                    ((Int64(block) * 14 + 10 + vrow // 64) * 64 + vrow % 64) * 128
                                    + ki
                                ] = h[n * self.h_components + c]
                    cute.arch.sync_threads()
            # Every warp has consumed U before its storage becomes output.
            cute.arch.sync_threads()
            # Q is dead; its first 2 KiB hold one sum per warp and token.
            norms = _tile(arena, self.memory["norm_partials"], Float32, 8, 64, False)
            for n in cutlass.range_constexpr(8):
                for c in cutlass.range_constexpr(2):
                    value = o[n * 4 + c] * 0.08838834764831845
                    square = value * value
                    if const_expr(self.rows == 16):
                        value_hi = o[n * 4 + c + 2] * 0.08838834764831845
                        square += value_hi * value_hi
                    # Lanes differing by 4 own the same token, different V.
                    # For Vs=32 only lane groups 0..3 have physical V rows.
                    for shift in cutlass.range_constexpr(2 if self.rows == 4 else 3):
                        square += cute.arch.shuffle_sync_bfly(square, offset=4 << shift)
                    if lane < 4:
                        norms[wid, n * 8 + 2 * lane + c] = square
            for e in cutlass.range_constexpr(cute.size(cr)):
                vi, ti = cr[e]
                if vi < self.rows:
                    su[ti, wid * self.rows + vi] = o[e] * 0.08838834764831845
            cute.arch.sync_threads()
            if const_expr(self.debug):
                if local == 0:
                    _debug_tile(dbg, block, 9, su, cmu)
            if const_expr(self.fnorm):
                self.norm_epilogue(x, nw, y, su, norms, arena, start, end, head, part, vbase)
            else:
                for row in range(wid, 64, 8):
                    token = start + row
                    for group in cutlass.range_constexpr(self.vs // 32):
                        value = su[row, group * 32 + lane]
                        if token < end:
                            out[(token * self.hv + head) * 128 + vbase + group * 32 + lane] = value
                if tid < 64 * (self.vs // 32):
                    row, group = tid // (self.vs // 32), tid % (self.vs // 32)
                    square = Float32(0.0)
                    for w in cutlass.range_constexpr(2 * self.vsplit):
                        square += norms[group * (2 * self.vsplit) + w, row]
                    token = start + row
                    if token < end:
                        partial[(token * self.hv + head) * 4 + vbase // 32 + group] = square
            cute.arch.sync_threads()
        if const_expr(self.cluster > 1):
            # No CTA leaves while a peer may still read its shared memory.
            if end > begin:
                cute.arch.cluster_wait()
        if slot >= 0 and (end > begin or flag == 0):
            for n in cutlass.range_constexpr(16):
                for c in cutlass.range_constexpr(self.h_components):
                    vi, ki = ch[n * 4 + c]
                    if vi < self.rows:
                        state[
                            ((slot * self.hv + head) * 128 + vbase + wid * self.rows + vi) * 128
                            + ki
                        ] = h[n * self.h_components + c]
