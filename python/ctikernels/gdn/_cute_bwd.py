# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e (`fla/ops/common/chunk_o.py`):
# GradQK keeps chunk_bwd_kernel_dqkwg's key-split grid, value-tile accumulation, per-split gate
# gradient planes and last-row correction; see THIRD_PARTY_NOTICES.

"""CuTe DSL kernels of the GDN chunk backward on sm_120 / sm_121 (loaded by ``_cute``), in the
order they run; each forms its outputs with fla-core 0.5.2's chunk_gated_delta_rule_bwd roundings
(a few FP32 sums kept unrounded):

- ``WuB`` (per output, chunk and value head): W or U from A, as the forward's IntraP forms them.
- ``LocalDV`` (per chunk and value head): dv_local = bf16(scale (K Qᵀ ∘ 2^(g_i - g_j))
  ∘ causalᵀ) dO.
- ``StatesR`` (per V-slice, value head and sequence): the forward recurrence again, writing the
  state at each chunk's start h_c and v_new = U - W bf16(h_c).
- ``GradStatesD`` (per V-slice, value head and sequence, chunks in reverse): dh_c, the gradient of
  h_c, and dv2 = bf16(K bf16(dh_c)) 2^(g_last - g) + dv_local, with
  dh_(c-1) = 2^(g_last) dh_c + scale Qᵀ (2^g dO) - Wᵀ dv2.
- ``GradQK`` (per key half, chunk and value head): dq, dk before the GVA sum, dw and the decay gradient's
  dq / dk terms.
- ``GradWY`` (per chunk and key head): the WY backward, the GVA sums, dv, dbeta and dg.

Shapes and inputs as ``_cute_fwd``; h and dh are [chunks Hv K, V] BF16 indexed by the chunk table
(chunk_offsets[n] + c). FLA counterparts: recompute_w_u_fwd_kernel, chunk_bwd_kernel_dv_local,
chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
chunk_bwd_kernel_dqkwg and prepare_wy_repr_bwd_kernel.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.memory import SmemAllocator  # 4.8 (4.6: cutlass.utils)

from ._cute_fwd import BT, stage_bf16, store_c, store_rows, store_wu_one


def _atom(width: int):
    kind = warpgroup.SmemLayoutAtomKind
    return warpgroup.make_smem_layout_atom(kind.K_SW128 if width >= 64 else kind.K_SW64, BFloat16)


def _tile(rows: int, cols: int):
    return cute.tile_to_shape(_atom(cols), (rows, cols), order=(0, 1))


@cute.jit
def _load(
    tma: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    bar: cute.Pointer,
    row: Int32,
    col: Int32,
    rows: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
):
    """TMA tile [rows, cols] of ``src`` at element (row, col) into ``dst``, landing on ``bar``
    (its byte count set by the caller); one warp."""
    g = cute.local_tile(cute.domain_offset((row, col), src), (rows, cols), (0, 0))
    s_part, g_part = cpasync.tma_partition(
        tma, 0, cute.make_layout(1), cute.group_modes(dst, 0, 2), cute.group_modes(g, 0, 2)
    )
    cute.copy(tma, g_part, s_part, tma_bar_ptr=bar)


@cute.jit
def _store(
    tma: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    row: Int32,
    col: Int32,
    rows: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
):
    """TMA store of the smem tile ``src`` [rows, cols] to ``dst`` at element (row, col); one
    warp, then committed."""
    g = cute.local_tile(cute.domain_offset((row, col), dst), (rows, cols), (0, 0))
    s_part, g_part = cpasync.tma_partition(
        tma, 0, cute.make_layout(1), cute.group_modes(src, 0, 2), cute.group_modes(g, 0, 2)
    )
    cute.copy(tma, s_part, g_part)
    cute.arch.cp_async_bulk_commit_group()


def _expect(bar, nbytes: int):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)


class WuB:
    """W or U from A as fla-core's recompute_w_u forms them, per (output, chunk, value head); 4
    warps, 25 KB (three CTAs per SM)."""

    def __init__(self, hk: int, hv: int, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv == 128 and hv % hk == 0
        self.hk, self.hv, self.dk, self.dv = hk, hv, dk, dv
        self.group = hv // hk
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mA: cute.Tensor,
        mG: cute.Tensor,
        mB: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mK [T, hk*dk]; mV [T, hv*dv]; mA [T, hv*BT] BF16; mG, mB [T, hv] FP32;
        # mW [T, hv*dk], mU [T, hv*dv] BF16 out
        kv = _tile(BT, self.dk)
        tt = _tile(BT, BT)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, kv, (BT, self.dk))
        tma_v, tV = cpasync.make_tiled_tma_atom(op, mV, kv, (BT, self.dv))
        tma_a, tA = cpasync.make_tiled_tma_atom(op, mA, tt, (BT, BT))
        out = cpasync.CopyBulkTensorTileS2GOp()
        tma_ws, tWs = cpasync.make_tiled_tma_atom(out, mW, kv, (BT, self.dk))
        tma_us, tUs = cpasync.make_tiled_tma_atom(out, mU, kv, (BT, self.dv))
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(BT, 8, 16)
        )

        @cute.struct
        class Storage:
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv)], 1024]
            sT: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(tt)], 1024]
            g: cute.struct.MemRange[Float32, BT]
            b: cute.struct.MemRange[Float32, BT]
            bar: cute.struct.MemRange[cutlass.Int64, 2]

        self.storage = Storage
        grid = [2, n_chunks, self.hv]  # W and U of a chunk side by side (A read twice from L2)
        self.kernel(
            tma_k, tK, tma_v, tV, tma_a, tA, tma_ws, tWs, tma_us, tUs, mG, mB, mCu, mTab, mW, mU,
            tiled_mma, kv, tt,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_v: cute.CopyAtom,
        tV: cute.Tensor,
        tma_a: cute.CopyAtom,
        tA: cute.Tensor,
        tma_ws: cute.CopyAtom,
        tWs: cute.Tensor,
        tma_us: cute.CopyAtom,
        tUs: cute.Tensor,
        mG: cute.Tensor,
        mB: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        tiled_mma: cute.TiledMma,
        kv: cute.ComposedLayout,
        tt: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        which, ci, h = cute.arch.block_idx()  # which: 0 W, 1 U
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n = mTab[ci, 0]
        c = mTab[ci, 1]
        bos = mCu[n]
        eos = mCu[n + 1]
        row0 = bos + c * BT
        valid = eos - row0
        if valid > BT:
            valid = Int32(BT)
        hk = h // self.group

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: K or V, 1: A
        sg = st.g.get_tensor(cute.make_layout(BT))
        sb = st.b.get_tensor(cute.make_layout(BT))
        sX = st.sX.get_tensor(kv.outer, swizzle=kv.inner)
        sT = st.sT.get_tensor(tt.outer, swizzle=tt.inner)

        if warp_i == 0:
            with cute.arch.elect_one():
                for q in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(bar + q, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_i == 0:
            _expect(bar, BT * self.dk * 2)
            if which == 0:
                _load(tma_k, tK, sX, bar, row0, hk * self.dk, BT, self.dk)
            else:
                _load(tma_v, tV, sX, bar, row0, h * self.dv, BT, self.dv)
            _expect(bar + 1, BT * BT * 2)
            _load(tma_a, tA, sT, bar + 1, row0, h * BT, BT, BT)
        if tid < BT:
            gv = Float32(0.0)
            bv = Float32(0.0)
            if tid < valid:
                gv = mG[row0 + tid, h]
                bv = mB[row0 + tid, h]
            sg[tid] = cute.math.exp2(gv, fastmath=False)  # 2^g
            sb[tid] = bv
        cute.arch.mbarrier_wait(bar, 0)
        cute.arch.mbarrier_wait(bar + 1, 0)
        cute.arch.sync_threads()  # beta and 2^g visible
        s2r_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tiled_mma
        )
        if which == 0:
            store_wu_one(
                sT, sX, sb, sg, tiled_mma, s2r_a, mW, row0, valid, h * self.dk, tid, True, 128,
                tma_ws, tWs,
            )  # fmt: skip
        else:
            store_wu_one(
                sT, sX, sb, sg, tiled_mma, s2r_a, mU, row0, valid, h * self.dv, tid, False, 128,
                tma_us, tUs,
            )  # fmt: skip


class LocalDV:
    """dv_local = bf16(scale (K Qᵀ ∘ 2^(g_i - g_j)) ∘ causalᵀ) dO per (chunk, value head), as
    fla-core's chunk_bwd_dv_local; 4 warps, 48 KB (two CTAs per SM: A' reuses Q's tile)."""

    def __init__(self, hk: int, hv: int, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv == 128 and hv % hk == 0
        self.hk, self.hv, self.dk, self.dv = hk, hv, dk, dv
        self.group = hv // hk
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mDO: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDL: cute.Tensor,
        scale: Float32,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mQ, mK [T, hk*dk]; mDO [T, hv*dv] BF16; mG [T, hv] FP32; mDL [T, hv*dv] BF16 out
        kv = _tile(BT, self.dk)
        tt = _tile(BT, BT)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, kv, (BT, self.dk))
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, kv, (BT, self.dk))
        tma_o, tO = cpasync.make_tiled_tma_atom(op, mDO, kv, (BT, self.dv))
        tma_l, tL = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mDL, kv, (BT, self.dv)
        )
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(BT, 8, 16)
        )

        @cute.struct
        class Storage:
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv)], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv)], 1024]
            g: cute.struct.MemRange[Float32, BT]
            bar: cute.struct.MemRange[cutlass.Int64, 2]

        self.storage = Storage
        grid = [n_chunks, self.hv, 1]
        self.kernel(
            tma_q, tQ, tma_k, tK, tma_o, tO, tma_l, tL, mG, mCu, mTab, mDL, scale, tiled_mma, kv,
            tt,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        tma_q: cute.CopyAtom,
        tQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_o: cute.CopyAtom,
        tO: cute.Tensor,
        tma_l: cute.CopyAtom,
        tL: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDL: cute.Tensor,
        scale: Float32,
        tiled_mma: cute.TiledMma,
        kv: cute.ComposedLayout,
        tt: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        ci, h, _ = cute.arch.block_idx()
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n = mTab[ci, 0]
        c = mTab[ci, 1]
        bos = mCu[n]
        eos = mCu[n + 1]
        row0 = bos + c * BT
        valid = eos - row0
        if valid > BT:
            valid = Int32(BT)
        hk = h // self.group

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: K + Q, 1: dO
        sg = st.g.get_tensor(cute.make_layout(BT))
        sK = st.sK.get_tensor(kv.outer, swizzle=kv.inner)  # K, then dv_local's stage
        sQ = st.sQ.get_tensor(kv.outer, swizzle=kv.inner)
        sO = st.sO.get_tensor(kv.outer, swizzle=kv.inner)
        sP = cute.make_tensor(cute.recast_ptr(st.sQ.data_ptr(), tt.inner), tt.outer)  # A' [j][i]

        if warp_i == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_i == 0:
            tile = BT * self.dk * 2
            _expect(bar, 2 * tile)
            _load(tma_k, tK, sK, bar, row0, hk * self.dk, BT, self.dk)
            _load(tma_q, tQ, sQ, bar, row0, hk * self.dk, BT, self.dk)
            _expect(bar + 1, tile)
            _load(tma_o, tO, sO, bar + 1, row0, h * self.dv, BT, self.dv)
        if tid < BT:
            gv = Float32(0.0)
            if tid < valid:
                gv = mG[row0 + tid, h]
            sg[tid] = gv
        cute.arch.mbarrier_wait(bar, 0)
        cute.arch.sync_threads()

        # ---- A' = bf16(scale K Qᵀ ∘ 2^(g_i - g_j)) over j <= i < valid  (M = j, N = i, K = key)
        slc = tiled_mma.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tiled_mma
        )
        ld_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BFloat16), tiled_mma
        )
        ld_bt = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tiled_mma
        )
        acc_p = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BT, BT)))
        _gemm_ab(tiled_mma, ld_a, ld_b, acc_p, sK, sQ, tid, self.dk, True)
        cute.arch.sync_threads()  # K and Q consumed
        cP = slc.partition_C(cute.make_identity_tensor((BT, BT)))
        for e in cutlass.range_constexpr(cute.size(acc_p)):
            j, i = cP[e]
            x = Float32(0.0)
            if j <= i:
                if i < valid:
                    x = acc_p[e] * scale * cute.math.exp2(sg[i] - sg[j], fastmath=True)
            sP[j, i] = x.to(BFloat16)
        cute.arch.mbarrier_wait(bar + 1, 0)
        cute.arch.sync_threads()  # A' visible; dO landed

        # ---- dv_local = A' dO  (M = j, N = v, K = i), out through K's dead tile
        sO_t = cute.make_tensor(sO.iterator, cute.select(sO.layout, mode=[1, 0]))
        acc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BT, self.dv)))
        _gemm_ab(tiled_mma, ld_a, ld_bt, acc, sP, sO_t, tid, BT, True)
        cC = slc.partition_C(cute.make_identity_tensor((BT, self.dv)))
        store_c(
            acc, cC, tiled_mma, sK, tma_l, tL, mDL, row0, valid, h * self.dv, tid, self.dv,
            self.threads,
        )  # fmt: skip


class StatesR:
    """The forward recurrence again per (V-slice, value head, sequence): h_c out (TMA) and v_new
    (from registers). 8 warps, or 4 with bv = 32 (48 KB: two CTAs per SM)."""

    def __init__(
        self, hk: int, hv: int, bv: int = 64, warps: int = 8, dk: int = 128, dv: int = 128
    ):
        assert dk == 128 and dv % bv == 0 and bv in (32, 64) and hv % hk == 0
        assert warps == 8 or (warps == 4 and bv == 32)
        self.hk, self.hv, self.bv, self.dk, self.dv = hk, hv, bv, dk, dv
        self.group = hv // hk
        self.warps = warps
        self.threads = 32 * warps

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mOrd: cute.Tensor,
        mH: cute.Tensor,
        mVN: cute.Tensor,
        n_seqs: Int32,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mW [T, hv*dk], mU [T, hv*dv], mK [T, hk*dk] BF16; mG [T, hv] FP32; mCu [N+1], mTab,
        # mOrd [N] (the sequences, longest first) i32; mH [chunks*hv*dk, dv], mVN [T, hv*dv] BF16 out
        lk = _tile(BT, self.dk)
        lu = _tile(BT, self.bv)
        ls = _tile(self.dk, self.bv)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_w, tW = cpasync.make_tiled_tma_atom(op, mW, lk, (BT, self.dk))
        tma_u, tU = cpasync.make_tiled_tma_atom(op, mU, lu, (BT, self.bv))
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, lk, (BT, self.dk))
        tma_h, tH = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mH, ls, (self.dk, self.bv)
        )
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        if const_expr(self.warps == 8):
            tm_tok = cute.make_tiled_mma(
                mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=(BT, 16, 16)
            )
        else:
            tm_tok = cute.make_tiled_mma(
                mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(BT, 8, 16)
            )
        tm_key = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.warps, 1, 1)),
            permutation_mnk=(16 * self.warps, 8, 16),
        )

        @cute.struct
        class Storage:
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sU: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sS: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(ls)], 1024]
            sD: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            g: cute.struct.MemRange[Float32, 2 * BT]
            bar: cute.struct.MemRange[cutlass.Int64, 2]

        self.storage = Storage
        grid = [self.dv // self.bv, self.hv, n_seqs]
        self.kernel(
            tma_w, tW, tma_u, tU, tma_k, tK, tma_h, tH, mG, mCu, mTab, mOrd, mVN, n_chunks, tm_tok,
            tm_key, lk, lu, ls,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        tma_w: cute.CopyAtom,
        tW: cute.Tensor,
        tma_u: cute.CopyAtom,
        tU: cute.Tensor,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_h: cute.CopyAtom,
        tH: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mOrd: cute.Tensor,
        mVN: cute.Tensor,
        n_chunks: Int32,
        tm_tok: cute.TiledMma,
        tm_key: cute.TiledMma,
        lk: cute.ComposedLayout,
        lu: cute.ComposedLayout,
        ls: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        vs, h, rank = cute.arch.block_idx()
        n = mOrd[rank]  # longest sequences launch first
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        hk = h // self.group
        v0 = vs * self.bv
        bos = mCu[n]
        eos = mCu[n + 1]
        nt = (eos - bos + BT - 1) // BT
        # this sequence's first entry in the chunk table (entries are in sequence order)
        lo = Int32(0)
        hi = n_chunks
        for _ in cutlass.range_constexpr(24):
            if lo < hi:
                mid = (lo + hi) // 2
                if mTab[mid, 0] < n:
                    lo = mid + 1
                else:
                    hi = mid
        cho = lo

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: W + U, 1: K
        sgall = st.g.get_tensor(cute.make_layout((BT, 2), stride=(1, BT)))
        sW = st.sW.get_tensor(lk.outer, swizzle=lk.inner)
        sK = st.sK.get_tensor(lk.outer, swizzle=lk.inner)
        sU = st.sU.get_tensor(lu.outer, swizzle=lu.inner)
        sS = st.sS.get_tensor(ls.outer, swizzle=ls.inner)
        sD = st.sD.get_tensor(lu.outer, swizzle=lu.inner)
        sSt = cute.make_tensor(sS.iterator, cute.select(sS.layout, mode=[1, 0]))
        sKt = cute.make_tensor(sK.iterator, cute.select(sK.layout, mode=[1, 0]))
        sDt = cute.make_tensor(sD.iterator, cute.select(sD.layout, mode=[1, 0]))

        if warp_i == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        slc_tok = tm_tok.get_slice(tid)
        slc_key = tm_key.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tm_tok
        )
        ld_bt = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm_tok
        )
        ld_at_k = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), tm_key
        )
        ld_bt_k = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm_key
        )
        st_tok = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tm_tok
        )
        st_key = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tm_key
        )
        la_ = ld_a.get_slice(tid)
        lbt = ld_bt.get_slice(tid)
        latk = ld_at_k.get_slice(tid)
        lbtk = ld_bt_k.get_slice(tid)
        sto = st_tok.get_slice(tid)
        stk = st_key.get_slice(tid)
        cT = slc_tok.partition_C(cute.make_identity_tensor((BT, self.bv)))

        acc_s = tm_key.make_fragment_C(tm_key.partition_shape_C((self.dk, self.bv)))
        acc_s.fill(0.0)
        fs = cute.make_rmem_tensor_like(acc_s, BFloat16)
        fs.fill(0.0)
        cute.copy(st_key, stk.retile(fs), stk.partition_D(sS))
        cute.arch.fence_proxy("async.shared", space="cta")

        wu_bytes = (BT * self.dk + BT * self.bv) * 2
        if warp_i == 0:
            if nt > 0:
                _expect(bar, wu_bytes)
                _load(tma_w, tW, sW, bar, bos, h * self.dk, BT, self.dk)
                _load(tma_u, tU, sU, bar, bos, h * self.dv + v0, BT, self.bv)
                _expect(bar + 1, BT * self.dk * 2)
                _load(tma_k, tK, sK, bar + 1, bos, hk * self.dk, BT, self.dk)
        if tid < BT:
            gv = Float32(0.0)
            if tid < eos - bos:
                gv = mG[bos + tid, h]
            sgall[tid, 0] = gv

        for c in cutlass.range(nt, unroll=1):
            row0 = bos + c * BT
            valid = eos - row0
            if valid > BT:
                valid = Int32(BT)
            ph = c % 2
            cute.arch.sync_threads()  # g of this chunk and bf16(S) visible; last chunk done
            sg = sgall[None, ph]
            g_last = sg[valid - 1]
            if tid < BT:
                gv = Float32(0.0)
                if c + 1 < nt:
                    if tid < eos - row0 - BT:
                        gv = mG[row0 + BT + tid, h]
                sgall[tid, 1 - ph] = gv
            # h_c out; its reads of sS finish before sS is rewritten below
            if warp_i == 0:
                _store(tma_h, sS, tH, ((cho + c) * self.hv + h) * self.dk, v0, self.dk, self.bv)
            cute.arch.mbarrier_wait(bar, ph)

            # WS = W bf16(S)  (M = token, N = v, K = key)
            rW = tm_tok.make_fragment_A(slc_tok.partition_A(sW))
            rS = tm_tok.make_fragment_B(slc_tok.partition_B(sSt))
            pW = la_.partition_S(sW)
            pS = lbt.partition_S(sSt)
            rWv = la_.retile(rW)
            rSv = lbt.retile(rS)
            acc_w = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, self.bv)))
            acc_w.fill(0.0)
            for kb in cutlass.range_constexpr(self.dk // 16):
                cute.copy(ld_bt, pS[None, None, kb], rSv[None, None, kb])
                cute.copy(ld_a, pW[None, None, kb], rWv[None, None, kb])
                cute.gemm(tm_tok, acc_w, rW[None, None, kb], rS[None, None, kb], acc_w)
            # v_new = U - WS (rows past the end: 0), out; V'' = 2^(g_last - g) v_new -> sD
            fv = cute.make_rmem_tensor_like(acc_w, Float32)
            fd = cute.make_rmem_tensor_like(acc_w, BFloat16)
            for e in cutlass.range_constexpr(cute.size(acc_w)):
                i, j = cT[e]
                vp = Float32(0.0)
                if i < valid:
                    vp = sU[i, j].to(Float32) - acc_w[e]
                fv[e] = vp
                fd[e] = (vp * cute.math.exp2(g_last - sg[i], fastmath=True)).to(BFloat16)
            cute.copy(st_tok, sto.retile(fd), sto.partition_D(sD))
            _store_pairs(fv, cT, mVN, row0, valid, h * self.dv + v0)
            cute.arch.sync_threads()  # W, U consumed; sD visible
            if warp_i == 0:
                if c + 1 < nt:
                    _expect(bar, wu_bytes)
                    _load(tma_w, tW, sW, bar, row0 + BT, h * self.dk, BT, self.dk)
                    _load(tma_u, tU, sU, bar, row0 + BT, h * self.dv + v0, BT, self.bv)

            # S = 2^(g_last) S + Kᵀ V''  (M = key, N = v, K = token)
            cute.arch.mbarrier_wait(bar + 1, ph)
            decay = cute.math.exp2(g_last, fastmath=True)
            for e in cutlass.range_constexpr(cute.size(acc_s)):
                acc_s[e] = acc_s[e] * decay
            rKt = tm_key.make_fragment_A(slc_key.partition_A(sKt))
            rD = tm_key.make_fragment_B(slc_key.partition_B(sDt))
            pKt = latk.partition_S(sKt)
            pD = lbtk.partition_S(sDt)
            rKtv = latk.retile(rKt)
            rDv = lbtk.retile(rD)
            for kb in cutlass.range_constexpr(BT // 16):
                cute.copy(ld_at_k, pKt[None, None, kb], rKtv[None, None, kb])
                cute.copy(ld_bt_k, pD[None, None, kb], rDv[None, None, kb])
                cute.gemm(tm_key, acc_s, rKt[None, None, kb], rD[None, None, kb], acc_s)
            if warp_i == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()  # K and sD consumed; h_c's store has read sS
            if warp_i == 0:
                if c + 1 < nt:
                    _expect(bar + 1, BT * self.dk * 2)
                    _load(tma_k, tK, sK, bar + 1, row0 + BT, hk * self.dk, BT, self.dk)
            for e in cutlass.range_constexpr(cute.size(acc_s)):
                fs[e] = acc_s[e].to(BFloat16)
            cute.copy(st_key, stk.retile(fs), stk.partition_D(sS))
            cute.arch.fence_proxy("async.shared", space="cta")
        if warp_i == 0:
            cute.arch.cp_async_bulk_wait_group(0)


class GradStatesD:
    def __init__(self, hk: int, hv: int, bv: int = 64, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv % bv == 0 and bv in (32, 64) and hv % hk == 0
        self.hk, self.hv, self.bv, self.dk, self.dv = hk, hv, bv, dk, dv
        self.group = hv // hk
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mW: cute.Tensor,
        mDO: cute.Tensor,
        mDL: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mOrd: cute.Tensor,
        mDH: cute.Tensor,
        mDV2: cute.Tensor,
        scale: Float32,
        n_seqs: Int32,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mQ, mK [T, hk*dk]; mW [T, hv*dk]; mDO, mDL [T, hv*dv] BF16; mG [T, hv] FP32;
        # mCu [N+1], mTab, mOrd [N] (longest first) i32; mDH [chunks*hv*dk, dv], mDV2 [T, hv*dv]
        # BF16 out
        lk = _tile(BT, self.dk)
        lu = _tile(BT, self.bv)
        ls = _tile(self.dk, self.bv)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, lk, (BT, self.dk))
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, lk, (BT, self.dk))
        tma_w, tW = cpasync.make_tiled_tma_atom(op, mW, lk, (BT, self.dk))
        tma_o, tO = cpasync.make_tiled_tma_atom(op, mDO, lu, (BT, self.bv))
        tma_l, tL = cpasync.make_tiled_tma_atom(op, mDL, lu, (BT, self.bv))
        tma_h, tH = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mDH, ls, (self.dk, self.bv)
        )
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tm_tok = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=(BT, 16, 16)
        )
        tm_key = cute.make_tiled_mma(
            mma_op, cute.make_layout((8, 1, 1)), permutation_mnk=(self.dk, 8, 16)
        )

        @cute.struct
        class Storage:
            sQ: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sO: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sLo: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sL: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sH: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(ls)], 1024]
            g: cute.struct.MemRange[Float32, 2 * BT]
            bar: cute.struct.MemRange[cutlass.Int64, 5]

        self.storage = Storage
        grid = [self.dv // self.bv, self.hv, n_seqs]
        self.kernel(
            tma_q, tQ, tma_k, tK, tma_w, tW, tma_o, tO, tma_l, tL, tma_h, tH, mG, mCu, mTab, mOrd,
            n_chunks, mDV2, scale, tm_tok, tm_key, lk, lu, ls,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.jit
    def _issue(
        self,
        tma: cute.CopyAtom,
        src: cute.Tensor,
        dst: cute.Tensor,
        bar: cute.Pointer,
        row: Int32,
        col: Int32,
        cols: cutlass.Constexpr[int],
    ):
        _expect(bar, BT * cols * 2)
        _load(tma, src, dst, bar, row, col, BT, cols)

    @cute.kernel
    def kernel(
        self,
        tma_q: cute.CopyAtom,
        tQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_w: cute.CopyAtom,
        tW: cute.Tensor,
        tma_o: cute.CopyAtom,
        tO: cute.Tensor,
        tma_l: cute.CopyAtom,
        tL: cute.Tensor,
        tma_h: cute.CopyAtom,
        tH: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mOrd: cute.Tensor,
        n_chunks: Int32,
        mDV2: cute.Tensor,
        scale: Float32,
        tm_tok: cute.TiledMma,
        tm_key: cute.TiledMma,
        lk: cute.ComposedLayout,
        lu: cute.ComposedLayout,
        ls: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        vs, h, rank = cute.arch.block_idx()
        n = mOrd[rank]  # longest sequences launch first
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        hk = h // self.group
        v0 = vs * self.bv
        bos = mCu[n]
        eos = mCu[n + 1]
        nt = (eos - bos + BT - 1) // BT
        # this sequence's first entry in the chunk table (entries are in sequence order)
        lo = Int32(0)
        hi = n_chunks
        for _ in cutlass.range_constexpr(24):
            if lo < hi:
                mid = (lo + hi) // 2
                if mTab[mid, 0] < n:
                    lo = mid + 1
                else:
                    hi = mid
        cho = lo

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: Q, 1: K, 2: W, 3: dO, 4: dv_local
        sgall = st.g.get_tensor(cute.make_layout((BT, 2), stride=(1, BT)))
        sQ = st.sQ.get_tensor(lk.outer, swizzle=lk.inner)
        sK = st.sK.get_tensor(lk.outer, swizzle=lk.inner)
        sW = st.sW.get_tensor(lk.outer, swizzle=lk.inner)
        sO = st.sO.get_tensor(lu.outer, swizzle=lu.inner)  # dO, then hi(2^g scale dO) in place
        sLo = st.sLo.get_tensor(lu.outer, swizzle=lu.inner)  # lo(2^g scale dO)
        sL = st.sL.get_tensor(lu.outer, swizzle=lu.inner)  # dv_local, then dv2's stage
        sV = st.sV.get_tensor(lu.outer, swizzle=lu.inner)  # -bf16(dv2)
        sH = st.sH.get_tensor(ls.outer, swizzle=ls.inner)  # bf16(dh) [key][v]
        stage = cute.make_tensor(
            cute.recast_ptr(sL.iterator, dtype=BFloat16), cute.make_layout(BT * self.bv)
        )
        sHt = cute.make_tensor(sH.iterator, cute.select(sH.layout, mode=[1, 0]))
        sQt = cute.make_tensor(sQ.iterator, cute.select(sQ.layout, mode=[1, 0]))
        sWt = cute.make_tensor(sW.iterator, cute.select(sW.layout, mode=[1, 0]))
        sO_t = cute.make_tensor(sO.iterator, cute.select(sO.layout, mode=[1, 0]))
        sLot = cute.make_tensor(sLo.iterator, cute.select(sLo.layout, mode=[1, 0]))
        sVt = cute.make_tensor(sV.iterator, cute.select(sV.layout, mode=[1, 0]))

        if warp_i == 0:
            with cute.arch.elect_one():
                for q in cutlass.range_constexpr(5):
                    cute.arch.mbarrier_init(bar + q, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        slc_tok = tm_tok.get_slice(tid)
        slc_key = tm_key.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tm_tok
        )
        ld_bt = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm_tok
        )
        ld_at_k = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), tm_key
        )
        ld_bt_k = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm_key
        )
        st_tok = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tm_tok
        )
        st_key = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tm_key
        )
        la_ = ld_a.get_slice(tid)
        lbt = ld_bt.get_slice(tid)
        latk = ld_at_k.get_slice(tid)
        lbtk = ld_bt_k.get_slice(tid)
        sto = st_tok.get_slice(tid)
        stk = st_key.get_slice(tid)
        cT = slc_tok.partition_C(cute.make_identity_tensor((BT, self.bv)))

        acc_h = tm_key.make_fragment_C(tm_key.partition_shape_C((self.dk, self.bv)))
        acc_h.fill(0.0)
        fh = cute.make_rmem_tensor_like(acc_h, BFloat16)
        fh.fill(0.0)
        cute.copy(st_key, stk.retile(fh), stk.partition_D(sH))
        cute.arch.fence_proxy("async.shared", space="cta")

        if warp_i == 0:
            if nt > 0:
                last = bos + (nt - 1) * BT
                self._issue(tma_q, tQ, sQ, bar, last, hk * self.dk, self.dk)
                self._issue(tma_k, tK, sK, bar + 1, last, hk * self.dk, self.dk)
                self._issue(tma_w, tW, sW, bar + 2, last, h * self.dk, self.dk)
                self._issue(tma_o, tO, sO, bar + 3, last, h * self.dv + v0, self.bv)
                self._issue(tma_l, tL, sL, bar + 4, last, h * self.dv + v0, self.bv)
        if tid < BT:
            gv = Float32(0.0)
            if nt > 0:
                if tid < eos - bos - (nt - 1) * BT:
                    gv = mG[bos + (nt - 1) * BT + tid, h]
            sgall[tid, 0] = gv

        for it in cutlass.range(nt, unroll=1):
            c = nt - 1 - it
            row0 = bos + c * BT
            valid = eos - row0
            if valid > BT:
                valid = Int32(BT)
            ph = it % 2
            prev = row0 - BT
            cute.arch.sync_threads()  # g of this chunk and bf16(dh) visible; last chunk done
            sg = sgall[None, ph]
            g_last = sg[valid - 1]
            if tid < BT:
                gv = Float32(0.0)
                if c > 0:
                    gv = mG[prev + tid, h]
                sgall[tid, 1 - ph] = gv
            # dh_c out; its reads of sH finish before sH is rewritten below
            if warp_i == 0:
                _store(tma_h, sH, tH, ((cho + c) * self.hv + h) * self.dk, v0, self.dk, self.bv)
            cute.arch.mbarrier_wait(bar + 1, ph)
            cute.arch.mbarrier_wait(bar + 3, ph)
            cute.arch.mbarrier_wait(bar + 4, ph)

            # K bf16(dh)  (M = token, N = v, K = key)
            rK = tm_tok.make_fragment_A(slc_tok.partition_A(sK))
            rH = tm_tok.make_fragment_B(slc_tok.partition_B(sHt))
            pK = la_.partition_S(sK)
            pH = lbt.partition_S(sHt)
            rKv = la_.retile(rK)
            rHv = lbt.retile(rH)
            acc_v = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, self.bv)))
            acc_v.fill(0.0)
            for kb in cutlass.range_constexpr(self.dk // 16):
                cute.copy(ld_a, pK[None, None, kb], rKv[None, None, kb])
                cute.copy(ld_bt, pH[None, None, kb], rHv[None, None, kb])
                cute.gemm(tm_tok, acc_v, rK[None, None, kb], rH[None, None, kb], acc_v)
            # dv2 = K bf16(dh) 2^(g_last - g) + dv_local (rows past the end: 0); -bf16(dv2) -> sV.
            # hi / lo of scale 2^g dO (rows past the end: 0): hi in place of dO, lo -> sLo.
            fv = cute.make_rmem_tensor_like(acc_v, Float32)
            fn = cute.make_rmem_tensor_like(acc_v, BFloat16)
            for e in cutlass.range_constexpr(cute.size(acc_v)):
                i, j = cT[e]
                x = Float32(0.0)
                y = Float32(0.0)
                if i < valid:
                    x = acc_v[e] * cute.math.exp2(g_last - sg[i], fastmath=True)
                    x = x + sL[i, j].to(Float32)
                    y = sO[i, j].to(Float32) * (scale * cute.math.exp2(sg[i], fastmath=True))
                xb = x.to(BFloat16)
                fv[e] = xb.to(Float32)
                fn[e] = (-xb.to(Float32)).to(BFloat16)
                yh = y.to(BFloat16)
                sO[i, j] = yh
                sLo[i, j] = (y - yh.to(Float32)).to(BFloat16)
            cute.copy(st_tok, sto.retile(fn), sto.partition_D(sV))
            cute.arch.sync_threads()  # K and dv_local consumed; sV, hi and lo visible
            if warp_i == 0:
                if c > 0:
                    self._issue(tma_k, tK, sK, bar + 1, prev, hk * self.dk, self.dk)
            stage_bf16(fv, cT, stage, self.bv)
            cute.arch.sync_threads()
            store_rows(stage, mDV2, row0, valid, h * self.dv + v0, tid, self.bv, self.threads)

            # dh = 2^(g_last) dh + Qᵀ (hi + lo) - Wᵀ dv2  (M = key, N = v, K = token)
            cute.arch.mbarrier_wait(bar, ph)
            cute.arch.mbarrier_wait(bar + 2, ph)
            decay = cute.math.exp2(g_last, fastmath=True)
            for e in cutlass.range_constexpr(cute.size(acc_h)):
                acc_h[e] = acc_h[e] * decay
            for part in cutlass.range_constexpr(3):
                sA = sWt if const_expr(part == 2) else sQt
                sB = sO_t if const_expr(part == 0) else (sLot if const_expr(part == 1) else sVt)
                rA = tm_key.make_fragment_A(slc_key.partition_A(sA))
                rB = tm_key.make_fragment_B(slc_key.partition_B(sB))
                pA = latk.partition_S(sA)
                pB = lbtk.partition_S(sB)
                rAv = latk.retile(rA)
                rBv = lbtk.retile(rB)
                for kb in cutlass.range_constexpr(BT // 16):
                    cute.copy(ld_at_k, pA[None, None, kb], rAv[None, None, kb])
                    cute.copy(ld_bt_k, pB[None, None, kb], rBv[None, None, kb])
                    cute.gemm(tm_key, acc_h, rA[None, None, kb], rB[None, None, kb], acc_h)
            if warp_i == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()  # Q, W, dO, lo, sV and the stage consumed; dh_c stored
            if warp_i == 0:
                if c > 0:
                    self._issue(tma_q, tQ, sQ, bar, prev, hk * self.dk, self.dk)
                    self._issue(tma_w, tW, sW, bar + 2, prev, h * self.dk, self.dk)
                    self._issue(tma_o, tO, sO, bar + 3, prev, h * self.dv + v0, self.bv)
                    self._issue(tma_l, tL, sL, bar + 4, prev, h * self.dv + v0, self.bv)
            for e in cutlass.range_constexpr(cute.size(acc_h)):
                fh[e] = acc_h[e].to(BFloat16)
            cute.copy(st_key, stk.retile(fh), stk.partition_D(sH))
            cute.arch.fence_proxy("async.shared", space="cta")
        if warp_i == 0:
            cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def _row_sums(
    parts: cute.Tensor, i0: Int32, j0: Int32, red: cute.Tensor, tid: Int32, out: cute.Tensor
):
    """Deterministic per-row sums of a (4, 2, 1)-warp accumulator's per-thread partials
    ``parts`` (slot 0: the thread's row i0, slot 1: i0 + 8; (i0, j0) its first element): out[i]
    written by thread i < BT (``red`` holds BT x 8 FP32; the caller syncs before reading out)."""
    contributor = (j0 % 8) // 2 + 4 * ((j0 % 16) // 8)
    for s in cutlass.range_constexpr(2):
        red[(i0 + 8 * s) * 8 + contributor] = parts[s]
    cute.arch.sync_threads()
    if tid < BT:
        out[tid] = _tree([red[tid * 8 + q] for q in range(8)])


@cute.jit
def _row_sums4(
    parts: cute.Tensor, i0: Int32, j0: Int32, red: cute.Tensor, tid: Int32, out: cute.Tensor
):
    """Deterministic per-row sums of a (4, 1, 1)-warp accumulator's per-thread partials
    ``parts`` (slot 0: the thread's row i0, slot 1: i0 + 8; (i0, j0) its first element): out[i]
    written by thread i < BT (``red`` holds BT x 4 FP32; the caller syncs before reading out)."""
    contributor = (j0 % 8) // 2
    for s in cutlass.range_constexpr(2):
        red[(i0 + 8 * s) * 4 + contributor] = parts[s]
    cute.arch.sync_threads()
    if tid < BT:
        out[tid] = _tree([red[tid * 4 + q] for q in range(4)])


def _tree(values: list):
    """Pairwise sum of ``values`` (a fixed tree, built at trace time)."""
    while len(values) > 1:
        pairs = [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)]
        values = pairs + values[len(pairs) * 2 :]
    return values[0]


class GradQK:
    """dq, dk (per value head, before the GVA sum), dw and the decay gradient's dq / dk terms
    (fla-core's chunk_bwd_dqkwg), per (key half, chunk, value head) as fla-core splits the keys;
    4 warps, 41 KB (two CTAs per SM). V streams in two 64-wide blocks; each key half's dg terms go
    to its own rows of mDG ([2 T, Hv]; GradWY adds the two)."""

    def __init__(self, hk: int, hv: int, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv == 128 and hv % hk == 0
        self.hk, self.hv, self.dk, self.dv = hk, hv, dk, dv
        self.group = hv // hk
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mDO: cute.Tensor,
        mVN: cute.Tensor,
        mDV2: cute.Tensor,
        mH: cute.Tensor,
        mDH: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDQ: cute.Tensor,
        mDK: cute.Tensor,
        mDW: cute.Tensor,
        mDG: cute.Tensor,
        scale: Float32,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mQ, mK [T, hk*dk]; mDO, mVN, mDV2 [T, hv*dv] BF16; mH, mDH [chunks*hv*dk, dv] BF16;
        # mG [T, hv] FP32; mDQ, mDK, mDW [T, hv*dk] BF16 out; mDG [2T, hv] FP32 out
        lt = _tile(BT, BT)  # every tile is [64, 64]: V blocks, key halves of Q / K, ds
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_o, tO = cpasync.make_tiled_tma_atom(op, mDO, lt, (BT, BT))
        tma_n, tN = cpasync.make_tiled_tma_atom(op, mVN, lt, (BT, BT))
        tma_2, t2 = cpasync.make_tiled_tma_atom(op, mDV2, lt, (BT, BT))
        tma_h, tH = cpasync.make_tiled_tma_atom(op, mH, lt, (BT, BT))
        tma_d, tD = cpasync.make_tiled_tma_atom(op, mDH, lt, (BT, BT))
        tma_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, lt, (BT, BT))
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, lt, (BT, BT))
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tm = cute.make_tiled_mma(mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(BT, 8, 16))

        @cute.struct
        class Storage:
            sO: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lt)], 1024]
            sN: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lt)], 1024]
            s2: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lt)], 1024]
            sH: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lt)], 1024]
            sD: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lt)], 1024]
            red: cute.struct.MemRange[Float32, BT * 4]
            last: cute.struct.MemRange[Float32, 16]
            g: cute.struct.MemRange[Float32, BT]
            dq: cute.struct.MemRange[Float32, BT]
            dg: cute.struct.MemRange[Float32, BT]
            bar: cute.struct.MemRange[cutlass.Int64, 2]

        self.storage = Storage
        grid = [2, n_chunks, self.hv]  # key halves side by side: dO, v_new, dv2 come twice from L2
        self.kernel(
            tma_o, tO, tma_n, tN, tma_2, t2, tma_h, tH, tma_d, tD, tma_q, tQ, tma_k, tK, mG, mCu,
            mTab, mDQ, mDK, mDW, mDG, scale, tm, lt,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        tma_o: cute.CopyAtom,
        tO: cute.Tensor,
        tma_n: cute.CopyAtom,
        tN: cute.Tensor,
        tma_2: cute.CopyAtom,
        t2: cute.Tensor,
        tma_h: cute.CopyAtom,
        tH: cute.Tensor,
        tma_d: cute.CopyAtom,
        tD: cute.Tensor,
        tma_q: cute.CopyAtom,
        tQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDQ: cute.Tensor,
        mDK: cute.Tensor,
        mDW: cute.Tensor,
        mDG: cute.Tensor,
        scale: Float32,
        tm: cute.TiledMma,
        lt: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        kh, ci, h = cute.arch.block_idx()
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n = mTab[ci, 0]
        c = mTab[ci, 1]
        bos = mCu[n]
        eos = mCu[n + 1]
        row0 = bos + c * BT
        valid = eos - row0
        if valid > BT:
            valid = Int32(BT)
        hk = h // self.group
        hrow = (ci * self.hv + h) * self.dk + kh * BT  # this key half's rows of h_c / dh_c
        kcol = kh * BT  # this key half's columns within a head

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: a V block of dO, v_new, dv2, h, dh; 1: Q, K
        sg = st.g.get_tensor(cute.make_layout(BT))
        sdq = st.dq.get_tensor(cute.make_layout(BT))
        sdg = st.dg.get_tensor(cute.make_layout(BT))
        red = st.red.get_tensor(cute.make_layout(BT * 4))
        last = st.last.get_tensor(cute.make_layout(16))
        sO = st.sO.get_tensor(lt.outer, swizzle=lt.inner)  # dO block, then bf16(ds) [i][j]
        sN = st.sN.get_tensor(lt.outer, swizzle=lt.inner)  # v_new block, then dq's stage
        s2 = st.s2.get_tensor(lt.outer, swizzle=lt.inner)  # dv2 block, then dw's and dk's stage
        sH = st.sH.get_tensor(lt.outer, swizzle=lt.inner)  # h block [key][v], then Q [i][key]
        sD = st.sD.get_tensor(lt.outer, swizzle=lt.inner)  # dh block [key][v], then K [j][key]

        if warp_i == 0:
            with cute.arch.elect_one():
                for q in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(bar + q, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tile = BT * BT * 2
        if tid < BT:
            gv = Float32(0.0)
            if tid < valid:
                gv = mG[row0 + tid, h]
            sg[tid] = gv

        slc = tm.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tm
        )
        ld_at = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), tm
        )
        ld_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BFloat16), tm
        )
        ld_bt = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm
        )
        cH = slc.partition_C(cute.make_identity_tensor((BT, BT)))

        # ---- per V block (M = token, N = key or token j, K = v): dq_v = dO h_cᵀ, dw = -dv2 h_cᵀ,
        # dk_v = v_new dh_cᵀ, ds = dO v_newᵀ, and sum(h_c dh_c), this key half
        acc_q = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
        acc_w = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
        acc_k = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
        acc_s = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
        hdh = []
        for vb in cutlass.range_constexpr(self.dv // BT):
            if warp_i == 0:
                _expect(bar, 5 * tile)
                vcol = h * self.dv + vb * BT
                _load(tma_o, tO, sO, bar, row0, vcol, BT, BT)
                _load(tma_n, tN, sN, bar, row0, vcol, BT, BT)
                _load(tma_2, t2, s2, bar, row0, vcol, BT, BT)
                _load(tma_h, tH, sH, bar, hrow, vb * BT, BT, BT)
                _load(tma_d, tD, sD, bar, hrow, vb * BT, BT, BT)
            cute.arch.mbarrier_wait(bar, vb % 2)
            _gemm_ab(tm, ld_a, ld_b, acc_q, sO, sH, tid, BT, vb == 0)
            _gemm_ab(tm, ld_a, ld_b, acc_w, s2, sH, tid, BT, vb == 0)
            _gemm_ab(tm, ld_a, ld_b, acc_k, sN, sD, tid, BT, vb == 0)
            _gemm_ab(tm, ld_a, ld_b, acc_s, sO, sN, tid, BT, vb == 0)
            products = []
            for e in cutlass.range_constexpr(BT * BT // 128):
                idx = tid + e * 128
                r = idx // BT
                col = idx % BT
                products.append(sH[r, col].to(Float32) * sD[r, col].to(Float32))
            hdh.append(_tree(products))
            cute.arch.sync_threads()  # this V block's tiles consumed
        if warp_i == 0:
            _expect(bar + 1, 2 * tile)
            _load(tma_q, tQ, sH, bar + 1, row0, hk * self.dk + kcol, BT, BT)
            _load(tma_k, tK, sD, bar + 1, row0, hk * self.dk + kcol, BT, BT)
        g_last = Float32(0.0)
        if valid > 0:  # the chunk table's padding entries (valid <= 0) store nothing
            g_last = sg[valid - 1]

        # ---- dw out (staged in the dead dv2 block); ds = bf16(scale dO v_newᵀ ∘ 2^(g_i - g_j))
        # over j <= i < valid into the dead dO block
        wstage = cute.make_tensor(
            cute.recast_ptr(s2.iterator, dtype=BFloat16), cute.make_layout(BT * BT)
        )
        for e in cutlass.range_constexpr(cute.size(acc_w)):
            acc_w[e] = -acc_w[e]
        stage_bf16(acc_w, cH, wstage, BT)
        sS = sO
        for e in cutlass.range_constexpr(cute.size(acc_s)):
            i, j = cH[e]
            x = Float32(0.0)
            if j <= i:
                if i < valid:
                    x = acc_s[e] * cute.math.exp2(sg[i] - sg[j], fastmath=True) * scale
            sS[i, j] = x.to(BFloat16)
        cute.arch.sync_threads()  # dw's stage and ds visible
        store_rows(wstage, mDW, row0, valid, h * self.dk + kcol, tid, BT, 128)
        cute.arch.mbarrier_wait(bar + 1, 0)

        # ---- dq = scale 2^g dq_v + ds K, dk = 2^(g_last - g) dk_v + dsᵀ Q (rows past the end: 0);
        # dg = sum(dq q) - sum(dk k), plus 2^(g_last) sum(h dh) + sum(2^(g_last - g) dk_v k) on the
        # last row, over this key half
        sQ = sH
        sK = sD
        sKt = cute.make_tensor(sK.iterator, cute.select(sK.layout, mode=[1, 0]))  # [key][j]
        sQt = cute.make_tensor(sQ.iterator, cute.select(sQ.layout, mode=[1, 0]))  # [key][i]
        sSt = cute.make_tensor(sS.iterator, cute.select(sS.layout, mode=[1, 0]))  # [j][i]
        tails = []
        for e in cutlass.range_constexpr(cute.size(acc_q)):
            i, col = cH[e]
            xq = Float32(0.0)
            xk = Float32(0.0)
            if i < valid:
                xq = acc_q[e] * cute.math.exp2(sg[i], fastmath=True) * scale
                xk = acc_k[e] * cute.math.exp2(g_last - sg[i], fastmath=True)
            tails.append(xk * sK[i, col].to(Float32))
            acc_q[e] = xq
            acc_k[e] = xk
        _gemm_ab(tm, ld_a, ld_bt, acc_q, sS, sKt, tid, BT, False)
        _gemm_ab(tm, ld_at, ld_bt, acc_k, sSt, sQt, tid, BT, False)
        pq = [[], []]
        pk = [[], []]
        for e in cutlass.range_constexpr(cute.size(acc_q)):
            i, col = cH[e]
            s = (e // 2) % 2
            pq[s].append(acc_q[e] * sQ[i, col].to(Float32))
            pk[s].append(acc_k[e] * sK[i, col].to(Float32))
        cute.arch.sync_threads()
        _block_sums4(_tree(hdh), _tree(tails), last, tid)
        cute.arch.sync_threads()
        tail = last[8] * cute.math.exp2(g_last, fastmath=True) + last[9]
        i0, j0 = cH[0]
        parts = cute.make_rmem_tensor((2,), Float32)
        for s in cutlass.range_constexpr(2):
            parts[s] = _tree(pq[s])
        _row_sums4(parts, i0, j0, red, tid, sdq)
        cute.arch.sync_threads()
        for s in cutlass.range_constexpr(2):
            parts[s] = _tree(pk[s])
        _row_sums4(parts, i0, j0, red, tid, sdg)
        cute.arch.sync_threads()
        if tid < valid:
            x = sdq[tid] - sdg[tid]
            if tid == valid - 1:
                x = x + tail
            mDG[kh * mG.shape[0] + row0 + tid, h] = x

        # ---- dq, dk out (staged in the dead v_new and dv2 blocks)
        qstage = cute.make_tensor(
            cute.recast_ptr(sN.iterator, dtype=BFloat16), cute.make_layout(BT * BT)
        )
        stage_bf16(acc_q, cH, qstage, BT)
        stage_bf16(acc_k, cH, wstage, BT)
        cute.arch.sync_threads()
        store_rows(qstage, mDQ, row0, valid, h * self.dk + kcol, tid, BT, 128)
        store_rows(wstage, mDK, row0, valid, h * self.dk + kcol, tid, BT, 128)


@cute.jit
def _block_sums4(x: Float32, y: Float32, buf: cute.Tensor, tid: Int32):
    """buf[8], buf[9] = the sums of x and y over a 4-warp CTA in fixed trees (buf[0:8] are
    scratch); synced inside, readable after the caller's next sync."""
    for level in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> level)
        y = y + cute.arch.shuffle_sync_bfly(y, offset=16 >> level)
    if tid % 32 == 0:
        buf[tid // 32] = x
        buf[4 + tid // 32] = y
    cute.arch.sync_threads()
    if tid == 0:
        buf[8] = (buf[0] + buf[1]) + (buf[2] + buf[3])
        buf[9] = (buf[4] + buf[5]) + (buf[6] + buf[7])


@cute.jit
def _gemm_ab(
    tm: cute.TiledMma,
    ld_a: cute.TiledCopy,
    ld_b: cute.TiledCopy,
    acc: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    tid: Int32,
    depth: cutlass.Constexpr[int],
    clear: cutlass.Constexpr[bool],
):
    """acc (+)= sA sBᵀ over ``depth`` (sA [M, depth] through ld_a, sB [N, depth] through ld_b)."""
    slc = tm.get_slice(tid)
    la = ld_a.get_slice(tid)
    lb = ld_b.get_slice(tid)
    rA = tm.make_fragment_A(slc.partition_A(sA))
    rB = tm.make_fragment_B(slc.partition_B(sB))
    pA = la.partition_S(sA)
    pB = lb.partition_S(sB)
    rAv = la.retile(rA)
    rBv = lb.retile(rB)
    if const_expr(clear):
        acc.fill(0.0)
    for kb in cutlass.range_constexpr(depth // 16):
        cute.copy(ld_a, pA[None, None, kb], rAv[None, None, kb])
        cute.copy(ld_b, pB[None, None, kb], rBv[None, None, kb])
        cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)


@cute.jit
def _store_bf16(acc: cute.Tensor, st_c: cute.TiledCopy, stc: cute.TiledCopy, dst: cute.Tensor):
    """bf16(acc) into dst through the accumulator's st.matrix copy."""
    f = cute.make_rmem_tensor_like(acc, BFloat16)
    for e in cutlass.range_constexpr(cute.size(acc)):
        f[e] = acc[e].to(BFloat16)
    cute.copy(st_c, stc.retile(f), stc.partition_D(dst))


@cute.jit
def _store_pairs(
    values: cute.Tensor,
    coords: cute.Tensor,
    dst: cute.Tensor,
    row0: Int32,
    valid: Int32,
    col0: Int32,
):
    """dst[row0 + i, col0 + j] = bf16(values) (in pairs) for the rows i below ``valid``."""
    pair = cute.make_rmem_tensor((2,), BFloat16)
    for e in cutlass.range_constexpr(0, cute.size(values), 2):
        i, j = coords[e]
        if i < valid:
            pair[0] = values[e].to(BFloat16)
            pair[1] = values[e + 1].to(BFloat16)
            cute.autovec_copy(
                pair,
                cute.make_tensor(
                    dst.iterator + Int64(row0 + i) * dst.layout.stride[0] + col0 + j,
                    cute.make_layout(2),
                ),
            )


class GradWY:
    """The WY backward (fla-core's prepare_wy_repr_bwd) per (chunk, key head) over its Hv / Hk
    value heads, with the GVA sums: dq = sum dq_r, dk = sum (dk_r + dk2_r) (GradQK's dq_r, dk_r),
    and per value head dv, dbeta and dg = dg_qk + dg2. 8 warps; the next value head's tiles load
    as soon as their buffers free."""

    def __init__(self, hk: int, hv: int, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv == 128 and hv % hk == 0
        self.hk, self.hv, self.dk, self.dv = hk, hv, dk, dv
        self.group = hv // hk
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mA: cute.Tensor,
        mDW: cute.Tensor,
        mDU: cute.Tensor,
        mDQh: cute.Tensor,
        mDKh: cute.Tensor,
        mDGqk: cute.Tensor,
        mB: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDQ: cute.Tensor,
        mDK: cute.Tensor,
        mDV: cute.Tensor,
        mDB: cute.Tensor,
        mDG: cute.Tensor,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mK [T, hk*dk]; mV, mDU [T, hv*dv]; mA [T, hv*BT]; mDW, mDQh, mDKh [T, hv*dk] BF16;
        # mDGqk [2T, hv] (key halves), mB, mG [T, hv] FP32; mDQ, mDK [T, hk*dk], mDV [T, hv*dv]
        # BF16 out; mDB, mDG [T, hv] FP32 out
        lk = _tile(BT, 128)
        tt = _tile(BT, BT)
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, lk, (BT, self.dk))
        tma_v, tV = cpasync.make_tiled_tma_atom(op, mV, lk, (BT, self.dv))
        tma_w, tW = cpasync.make_tiled_tma_atom(op, mDW, lk, (BT, self.dk))
        tma_u, tU = cpasync.make_tiled_tma_atom(op, mDU, lk, (BT, self.dv))
        tma_x, tX = cpasync.make_tiled_tma_atom(op, mDKh, lk, (BT, self.dk))
        tma_a, tA = cpasync.make_tiled_tma_atom(op, mA, tt, (BT, BT))
        tma_ok, tOK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mDK, lk, (BT, self.dk)
        )
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tm = cute.make_tiled_mma(mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=(BT, 16, 16))

        @cute.struct
        class Storage:
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sU: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sX: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(tt)], 1024]
            red: cute.struct.MemRange[Float32, BT * 8]
            colred: cute.struct.MemRange[Float32, 4 * BT]
            db: cute.struct.MemRange[Float32, BT]
            dg: cute.struct.MemRange[Float32, BT]
            g: cute.struct.MemRange[Float32, BT]
            b: cute.struct.MemRange[Float32, BT]
            c: cute.struct.MemRange[Float32, BT]
            pow_g: cute.struct.MemRange[Float32, BT]
            bar: cute.struct.MemRange[cutlass.Int64, 5]

        self.storage = Storage
        grid = [n_chunks, self.hk, 1]
        self.kernel(
            tma_k, tK, tma_v, tV, tma_w, tW, tma_u, tU, tma_x, tX, tma_a, tA, tma_ok, tOK, mDQh,
            mDGqk, mB, mG, mCu, mTab, mDQ, mDK, mDV, mDB, mDG, tm, lk, tt,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)  # fmt: skip

    @cute.jit
    def _issue(
        self,
        tma: cute.CopyAtom,
        src: cute.Tensor,
        dst: cute.Tensor,
        bar: cute.Pointer,
        row: Int32,
        col: Int32,
        cols: cutlass.Constexpr[int],
    ):
        _expect(bar, BT * cols * 2)
        _load(tma, src, dst, bar, row, col, BT, cols)

    @cute.kernel
    def kernel(
        self,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_v: cute.CopyAtom,
        tV: cute.Tensor,
        tma_w: cute.CopyAtom,
        tW: cute.Tensor,
        tma_u: cute.CopyAtom,
        tU: cute.Tensor,
        tma_x: cute.CopyAtom,
        tX: cute.Tensor,
        tma_a: cute.CopyAtom,
        tA: cute.Tensor,
        tma_ok: cute.CopyAtom,
        tOK: cute.Tensor,
        mDQh: cute.Tensor,
        mDGqk: cute.Tensor,
        mB: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mDQ: cute.Tensor,
        mDK: cute.Tensor,
        mDV: cute.Tensor,
        mDB: cute.Tensor,
        mDG: cute.Tensor,
        tm: cute.TiledMma,
        lk: cute.ComposedLayout,
        tt: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        ci, hk, _ = cute.arch.block_idx()
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        n = mTab[ci, 0]
        c = mTab[ci, 1]
        bos = mCu[n]
        eos = mCu[n + 1]
        row0 = bos + c * BT
        valid = eos - row0
        if valid > BT:
            valid = Int32(BT)
        h0 = hk * self.group

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: K, 1: V, 2: dw, 3: du + A, 4: dk_r
        sg = st.g.get_tensor(cute.make_layout(BT))
        sb = st.b.get_tensor(cute.make_layout(BT))
        sc = st.c.get_tensor(cute.make_layout(BT))  # beta 2^g
        sgam = st.pow_g.get_tensor(cute.make_layout(BT))  # 2^g
        red = st.red.get_tensor(cute.make_layout(BT * 8))
        colred = st.colred.get_tensor(cute.make_layout(4 * BT))
        sdb = st.db.get_tensor(cute.make_layout(BT))
        sdg = st.dg.get_tensor(cute.make_layout(BT))
        sK = st.sK.get_tensor(lk.outer, swizzle=lk.inner)
        sV = st.sV.get_tensor(lk.outer, swizzle=lk.inner)
        sW = st.sW.get_tensor(lk.outer, swizzle=lk.inner)  # dw, then bf16(dA) / bf16(dA2)
        sU = st.sU.get_tensor(lk.outer, swizzle=lk.inner)  # du (= dv2), then bf16(M1)
        sX = st.sX.get_tensor(lk.outer, swizzle=lk.inner)  # GradQK's dk of this value head
        sA = st.sA.get_tensor(tt.outer, swizzle=tt.inner)
        sP = cute.make_tensor(cute.recast_ptr(st.sW.data_ptr(), tt.inner), tt.outer)
        sM = cute.make_tensor(cute.recast_ptr(st.sU.data_ptr(), tt.inner), tt.outer)
        sKt = cute.make_tensor(sK.iterator, cute.select(sK.layout, mode=[1, 0]))
        sWt = cute.make_tensor(sW.iterator, cute.select(sW.layout, mode=[1, 0]))
        sUt = cute.make_tensor(sU.iterator, cute.select(sU.layout, mode=[1, 0]))
        sAt = cute.make_tensor(sA.iterator, cute.select(sA.layout, mode=[1, 0]))
        sPt = cute.make_tensor(sP.iterator, cute.select(sP.layout, mode=[1, 0]))
        sMt = cute.make_tensor(sM.iterator, cute.select(sM.layout, mode=[1, 0]))

        if warp_i == 0:
            with cute.arch.elect_one():
                for q in cutlass.range_constexpr(5):
                    cute.arch.mbarrier_init(bar + q, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_i == 0:
            self._issue(tma_k, tK, sK, bar, row0, hk * self.dk, self.dk)
            self._issue(tma_v, tV, sV, bar + 1, row0, h0 * self.dv, self.dv)
            self._issue(tma_w, tW, sW, bar + 2, row0, h0 * self.dk, self.dk)
            _expect(bar + 3, (BT * self.dv + BT * BT) * 2)
            _load(tma_u, tU, sU, bar + 3, row0, h0 * self.dv, BT, self.dv)
            _load(tma_a, tA, sA, bar + 3, row0, h0 * BT, BT, BT)
            self._issue(tma_x, tX, sX, bar + 4, row0, h0 * self.dk, self.dk)

        slc = tm.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tm
        )
        ld_at = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), tm
        )
        ld_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BFloat16), tm
        )
        ld_bt = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tm
        )
        st_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tm
        )
        stc = st_c.get_slice(tid)
        cA = slc.partition_C(cute.make_identity_tensor((BT, BT)))
        cF = slc.partition_C(cute.make_identity_tensor((BT, 128)))
        cBn = slc.partition_B(cute.make_identity_tensor((BT, 128)))  # (token j, key / v)
        cBk = slc.partition_B(cute.make_identity_tensor((128, BT)))  # (key, token i)
        i0, j0 = cA[0]
        warp_m = i0 // 16

        cute.arch.mbarrier_wait(bar, 0)
        acc_kk = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
        _gemm_ab(tm, ld_a, ld_b, acc_kk, sK, sK, tid, 128, True)  # K Kᵀ (FP32)
        acc_k = tm.make_fragment_C(tm.partition_shape_C((BT, 128)))
        acc_k.fill(0.0)
        acc_q = cute.make_rmem_tensor(
            (32,), Float32
        )  # dq: row tid // 4, 32 columns from 32 (tid % 4)
        acc_q.fill(0.0)
        pb = cute.make_rmem_tensor((2,), Float32)
        pg = cute.make_rmem_tensor((2,), Float32)

        for r in cutlass.range(self.group, unroll=1):
            h = h0 + r
            ph = r % 2
            more = r + 1 < self.group
            cute.arch.sync_threads()  # the previous value head is done with g, beta and sX
            if tid < BT:
                gv = Float32(0.0)
                bv = Float32(0.0)
                if tid < valid:
                    gv = mG[row0 + tid, h]
                    bv = mB[row0 + tid, h]
                pow_g = cute.math.exp2(gv, fastmath=False)
                sg[tid] = gv
                sb[tid] = bv
                sgam[tid] = pow_g
                sc[tid] = bv * pow_g
            cute.arch.mbarrier_wait(bar + 1, ph)
            cute.arch.mbarrier_wait(bar + 2, ph)
            cute.arch.mbarrier_wait(bar + 3, ph)
            # A's rows past the end are the next sequence's: zero (fla-core masks them on load)
            if valid < BT:
                for e in cutlass.range_constexpr(BT * BT // 256):
                    idx = tid + e * 256
                    if idx // BT >= valid:
                        sA[idx // BT, idx % BT] = BFloat16(0.0)
            cute.arch.sync_threads()

            # dv = (Aᵀ du) beta, dbeta += sum(Aᵀ du v)  (M = i, N = v, K = m)
            acc = tm.make_fragment_C(tm.partition_shape_C((BT, 128)))
            _gemm_ab(tm, ld_at, ld_bt, acc, sAt, sUt, tid, BT, True)
            terms = [[], []]
            for e in cutlass.range_constexpr(cute.size(acc)):
                i, j = cF[e]
                terms[(e // 2) % 2].append(acc[e] * sV[i, j].to(Float32))
                acc[e] = acc[e] * sb[i]
            for s in cutlass.range_constexpr(2):
                pb[s] = _tree(terms[s])
            _store_pairs(acc, cF, mDV, row0, valid, h * self.dv)

            # dk += (Aᵀ dw) beta 2^g, dbeta += sum(Aᵀ dw k 2^g), dg += sum(Aᵀ dw k beta 2^g)
            _gemm_ab(tm, ld_at, ld_bt, acc, sAt, sWt, tid, BT, True)
            terms = [[], []]
            gterms = [[], []]
            for e in cutlass.range_constexpr(cute.size(acc)):
                i, j = cF[e]
                gx = sgam[i]
                kv = sK[i, j].to(Float32)
                terms[(e // 2) % 2].append(acc[e] * kv * gx)
                gterms[(e // 2) % 2].append(acc[e] * (kv * sc[i]))
                acc_k[e] = acc_k[e] + acc[e] * (gx * sb[i])
            for s in cutlass.range_constexpr(2):
                pb[s] = pb[s] + _tree(terms[s])
                pg[s] = _tree(gterms[s])

            # dA = dw bf16(k beta 2^g)ᵀ + du bf16(v beta)ᵀ, strictly lower over valid rows (the
            # B rows scaled on their fragments). dA, M1 and dA2 are rounded once to BF16, as fla-core.
            acc_a = tm.make_fragment_C(tm.partition_shape_C((BT, BT)))
            _gemm_scaled(tm, ld_a, ld_b, acc_a, sW, sK, cBn, sc, 0, tid, 128, True)
            _gemm_scaled(tm, ld_a, ld_b, acc_a, sU, sV, cBn, sb, 0, tid, 128, False)
            cute.arch.sync_threads()  # v, dw and du consumed
            if warp_i == 0:
                if more:
                    self._issue(tma_v, tV, sV, bar + 1, row0, (h + 1) * self.dv, self.dv)
            for e in cutlass.range_constexpr(cute.size(acc_a)):
                i, j = cA[e]
                x = Float32(0.0)
                if i > j:
                    if i < valid:
                        x = acc_a[e]
                acc_a[e] = x
            _store_bf16(acc_a, st_c, stc, sP)
            cute.arch.sync_threads()
            # M1 = dA Aᵀ, then dA2 = -(Aᵀ M1 ∘ 2^(g_i - g_j)), strictly lower (FP32 in acc_a)
            _gemm_ab(tm, ld_a, ld_b, acc_a, sP, sA, tid, BT, True)
            _store_bf16(acc_a, st_c, stc, sM)
            cute.arch.sync_threads()
            _gemm_ab(tm, ld_at, ld_bt, acc_a, sAt, sMt, tid, BT, True)
            for e in cutlass.range_constexpr(cute.size(acc_a)):
                i, j = cA[e]
                x = Float32(0.0)
                if i > j:
                    if i < valid:
                        x = -acc_a[e] * cute.math.exp2(sg[i] - sg[j], fastmath=True)
                acc_a[e] = x
            _store_bf16(acc_a, st_c, stc, sP)
            cute.arch.sync_threads()  # dA2 visible; A and M1 consumed
            if warp_i == 0:
                if more:
                    _expect(bar + 3, (BT * self.dv + BT * BT) * 2)
                    _load(tma_u, tU, sU, bar + 3, row0, (h + 1) * self.dv, BT, self.dv)
                    _load(tma_a, tA, sA, bar + 3, row0, (h + 1) * BT, BT, BT)

            # dk += (dA2 K) beta + dA2ᵀ bf16(k beta), dbeta += sum(dA2 K k)  (N = key)
            _gemm_ab(tm, ld_a, ld_bt, acc, sP, sKt, tid, BT, True)
            terms = [[], []]
            for e in cutlass.range_constexpr(cute.size(acc)):
                i, j = cF[e]
                terms[(e // 2) % 2].append(acc[e] * sK[i, j].to(Float32))
                acc_k[e] = acc_k[e] + acc[e] * sb[i]
            for s in cutlass.range_constexpr(2):
                pb[s] = pb[s] + _tree(terms[s])
            _gemm_scaled(tm, ld_at, ld_bt, acc_k, sPt, sKt, cBk, sb, 1, tid, BT, False)

            # dg += rowsum(dA2 ∘ K Kᵀ beta_i) - colsum(same): rows in a fixed tree; columns
            # down the lanes sharing a column, then over the four row bands
            terms = [[], []]
            cols = []
            for e in cutlass.range_constexpr(cute.size(acc_a)):
                i, j = cA[e]
                x = acc_a[e] * (acc_kk[e] * sb[i])
                terms[(e // 2) % 2].append(x)
                if const_expr((e // 2) % 2 == 0):
                    cols.append(x)
                else:
                    cols[len(cols) - 2 + e % 2] = cols[len(cols) - 2 + e % 2] + x
            for s in cutlass.range_constexpr(2):
                pg[s] = pg[s] + _tree(terms[s])
            for q in cutlass.range_constexpr(len(cols)):
                x = cols[q]
                for level in cutlass.range_constexpr(3):
                    x = x + cute.arch.shuffle_sync_bfly(x, offset=4 << level)
                cols[q] = x
            if lane < 4:
                for e in cutlass.range_constexpr(0, cute.size(acc_a), 4):
                    _, j = cA[e]
                    colred[warp_m * BT + j] = cols[e // 4 * 2]
                    colred[warp_m * BT + j + 1] = cols[e // 4 * 2 + 1]
            cute.arch.sync_threads()  # dA2 consumed; column partials visible
            if warp_i == 0:
                if more:
                    self._issue(tma_w, tW, sW, bar + 2, row0, (h + 1) * self.dk, self.dk)
            _row_sums(pb, i0, j0, red, tid, sdb)
            cute.arch.sync_threads()
            _row_sums(pg, i0, j0, red, tid, sdg)
            cute.arch.sync_threads()
            if tid < valid:
                col = _tree([colred[q * BT + tid] for q in range(4)])
                mDB[row0 + tid, h] = sdb[tid]
                qk = mDGqk[row0 + tid, h] + mDGqk[mG.shape[0] + row0 + tid, h]
                mDG[row0 + tid, h] = sdg[tid] - col + qk

            # the GVA sums: dk += GradQK's dk (its tile), dq += GradQK's dq (16 B rows)
            cute.arch.mbarrier_wait(bar + 4, ph)
            for e in cutlass.range_constexpr(cute.size(acc_k)):
                i, j = cF[e]
                acc_k[e] = acc_k[e] + sX[i, j].to(Float32)
            qrow = tid // 4
            if qrow < valid:
                part = cute.make_rmem_tensor((8,), BFloat16)
                for q in cutlass.range_constexpr(4):
                    col0 = (tid % 4) * 32 + q * 8
                    cute.autovec_copy(
                        cute.make_tensor(
                            mDQh.iterator
                            + Int64(row0 + qrow) * mDQh.layout.stride[0]
                            + h * self.dk
                            + col0,
                            cute.make_layout(8),
                        ),
                        part,
                    )
                    for t in cutlass.range_constexpr(8):
                        acc_q[q * 8 + t] = acc_q[q * 8 + t] + part[t].to(Float32)
            cute.arch.sync_threads()  # sX consumed
            if warp_i == 0:
                if more:
                    self._issue(tma_x, tX, sX, bar + 4, row0, (h + 1) * self.dk, self.dk)

        # ---- dq (16 B rows) and dk (staged in the dead v tile) out
        qrow = tid // 4
        if qrow < valid:
            part = cute.make_rmem_tensor((8,), BFloat16)
            for q in cutlass.range_constexpr(4):
                for t in cutlass.range_constexpr(8):
                    part[t] = acc_q[q * 8 + t].to(BFloat16)
                col0 = (tid % 4) * 32 + q * 8
                cute.autovec_copy(
                    part,
                    cute.make_tensor(
                        mDQ.iterator
                        + Int64(row0 + qrow) * mDQ.layout.stride[0]
                        + hk * self.dk
                        + col0,
                        cute.make_layout(8),
                    ),
                )
        store_c(acc_k, cF, tm, sV, tma_ok, tOK, mDK, row0, valid, hk * self.dk, tid, 128, 256)


@cute.jit
def _gemm_scaled(
    tm: cute.TiledMma,
    ld_a: cute.TiledCopy,
    ld_b: cute.TiledCopy,
    acc: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    cB: cute.Tensor,
    factor: cute.Tensor,
    by: cutlass.Constexpr[int],
    tid: Int32,
    depth: cutlass.Constexpr[int],
    clear: cutlass.Constexpr[bool],
):
    """acc (+)= sA bf16(diag-scaled sB)ᵀ: each B element times factor[its coordinate ``by``]
    (0: N, 1: K, from ``cB``, the partitioned identity of sB) before the MMA."""
    slc = tm.get_slice(tid)
    la = ld_a.get_slice(tid)
    lb = ld_b.get_slice(tid)
    rA = tm.make_fragment_A(slc.partition_A(sA))
    rB = tm.make_fragment_B(slc.partition_B(sB))
    pA = la.partition_S(sA)
    pB = lb.partition_S(sB)
    rAv = la.retile(rA)
    rBv = lb.retile(rB)
    if const_expr(clear):
        acc.fill(0.0)
    for kb in cutlass.range_constexpr(depth // 16):
        cute.copy(ld_a, pA[None, None, kb], rAv[None, None, kb])
        cute.copy(ld_b, pB[None, None, kb], rBv[None, None, kb])
        fb = rB[None, None, kb]
        cb = cB[None, None, kb]
        # an MMA atom's 4 B values per thread share their N coordinate and sit at 4 K
        # coordinates that repeat over the N tiles: read each factor once
        if const_expr(by == 1):
            fs = [factor[cb[v][1]] for v in range(4)]
            for e in cutlass.range_constexpr(cute.size(fb)):
                fb[e] = (fb[e].to(Float32) * fs[e % 4]).to(BFloat16)
        else:
            for t in cutlass.range_constexpr(cute.size(fb) // 4):
                f = factor[cb[4 * t][0]]
                for v in cutlass.range_constexpr(4):
                    fb[4 * t + v] = (fb[4 * t + v].to(Float32) * f).to(BFloat16)
        cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)
