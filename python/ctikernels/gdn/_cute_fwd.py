# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Implements the chunked gated delta rule (as in FLA's `chunk_gated_delta_rule`).

"""CuTe DSL kernels of the GDN chunk forward on sm_120 / sm_121 (loaded by ``_cute``).

``IntraP`` (one CTA per chunk and value head): L = strict_lower(diag(beta) (K Kᵀ ∘ 2^(g_i - g_j))),
T = (I + L)^-1 in FP32 (16x16 diagonal substitution, then 16 -> 32 -> 64 doubling), A = bf16(T),
W = A bf16(diag(beta 2^g) K) and U = A bf16(diag(beta) V): fla-core's recompute_w_u numerics,
which its backward reruns from A. ``RecurS`` (one CTA per V-slice, value head and sequence), per
chunk in order: V' = U - W bf16(S), O = scale (2^g Q bf16(S) + bf16(Q Kᵀ ∘ 2^(g_i - g_j) ∘ causal)
bf16(V')), S = 2^(g_last) S + Kᵀ bf16(2^(g_last - g) V'); S stays FP32 in registers and no states
or v_new are written.
Inputs as fla-core 0.5.2's chunked forward takes them: q, k [T, Hk, 128] normalised, v [T, Hv, 128],
g the chunk-local cumulative gate in the log2 domain and beta [T, Hv] FP32, value head h reading
key head h // (Hv / Hk). Rows past a sequence's end are masked; chunk-table entries past the last
chunk do nothing. FLA counterparts: chunk_gated_delta_rule_fwd_kkt_solve_kernel,
recompute_w_u_fwd_kernel, chunk_gated_delta_rule_fwd_kernel_h_blockdim64 and chunk_fwd_kernel_o.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.memory import SmemAllocator  # 4.8 (4.6: cutlass.utils)

BT = 64  # chunk
DIAG = 16  # diagonal block of the solve
LDL = BT + 1  # FP32 row pitch of the L/T buffer (bank spread)


def _swizzle(width: int) -> int:
    """The XOR span of stage_bf16 / store_rows for rows of ``width`` BF16 (16 B chunks)."""
    return min(8, width // 8)


@cute.jit
def stage_bf16(
    values: cute.Tensor,
    coords: cute.Tensor,
    stage: cute.Tensor,
    width: cutlass.Constexpr[int],
    col0: cutlass.Constexpr[int] = 0,
):
    """``values`` (one per coordinate of ``coords``, (row, column - col0)) as BF16 into
    ``stage``, rows of ``width`` with their 16 B chunks XOR-swizzled by row (spreads banks for
    store_rows)."""
    for e in cutlass.range_constexpr(cute.size(values)):
        m, n = coords[e]
        n = n + col0
        ch = (n // 8) ^ (m % _swizzle(width))
        stage[m * width + ch * 8 + n % 8] = values[e].to(BFloat16)


@cute.jit
def store_rows(
    stage: cute.Tensor,
    out: cute.Tensor,
    row0: Int32,
    valid: Int32,
    col0: Int32,
    tid: Int32,
    width: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    """Rows below ``valid`` of a stage_bf16 buffer to ``out`` [rows, ...] at (row0 + r, col0),
    16 B per copy."""
    chunks = width // 8
    for e in cutlass.range_constexpr(BT * chunks // threads):
        idx = tid + e * threads
        r = idx // chunks
        ch = idx % chunks
        if r < valid:
            src = cute.make_tensor(
                stage.iterator + r * width + (ch ^ (r % _swizzle(width))) * 8, cute.make_layout(8)
            )
            dst = cute.make_tensor(
                out.iterator + Int64(row0 + r) * out.layout.stride[0] + col0 + ch * 8,
                cute.make_layout(8),
            )
            cute.autovec_copy(src, dst)


@cute.jit
def store_wu(
    sT: cute.Tensor,
    sK: cute.Tensor,
    sV: cute.Tensor,
    sb: cute.Tensor,
    sgam: cute.Tensor,
    bar_v: cute.Pointer,
    tiled_mma: cute.TiledMma,
    s2r_a: cute.TiledCopy,
    mW: cute.Tensor,
    mU: cute.Tensor,
    row0: Int32,
    valid: Int32,
    h: Int32,
    tid: Int32,
    tma_w: cute.CopyAtom,
    tW: cute.Tensor,
    tma_u: cute.CopyAtom,
    tU: cute.Tensor,
):
    """W = sT bf16((K beta) 2^g) and U = sT bf16(V beta), as fla-core's recompute_w_u forms them
    from A = sT (BF16, [BT, BT]; written by the caller before the call), stored for rows below
    ``valid`` (whole chunks by TMA: tma_w / tma_u, S2G atoms of mW / mU). ``sgam`` holds 2^g per
    row. 128 threads; V lands on ``bar_v``; the results leave through the dead K / V tiles."""
    cute.arch.sync_threads()  # sT, beta and 2^g visible
    store_wu_one(
        sT, sK, sb, sgam, tiled_mma, s2r_a, mW, row0, valid, h * 128, tid, True, 128, tma_w, tW
    )
    cute.arch.mbarrier_wait(bar_v, 0)
    store_wu_one(
        sT, sV, sb, sgam, tiled_mma, s2r_a, mU, row0, valid, h * 128, tid, False, 128, tma_u, tU
    )


@cute.jit
def store_wu_one(
    sT: cute.Tensor,
    sX: cute.Tensor,
    sb: cute.Tensor,
    sgam: cute.Tensor,
    tiled_mma: cute.TiledMma,
    s2r_a: cute.TiledCopy,
    out: cute.Tensor,
    row0: Int32,
    valid: Int32,
    col0: Int32,
    tid: Int32,
    gamma: cutlass.Constexpr[bool],
    dk: cutlass.Constexpr[int] = 128,
    tma_out: cute.CopyAtom = None,
    t_out: cute.Tensor = None,
):
    """out = sT bf16(X beta (2^g if ``gamma``)) for X = sX [BT, dk] (K for W, V for U, or dk of
    their columns), the rows of X scaled on their MMA fragments; staged in sX, then stored for
    rows below ``valid`` (a whole chunk by one TMA store when ``tma_out``, out's S2G atom, is
    given). 128 threads; sT, sX, beta and 2^g visible on entry; sX is dead after."""
    slc_mma = tiled_mma.get_slice(tid)
    ta = s2r_a.get_slice(tid)
    s2r_bt = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BFloat16), tiled_mma
    )
    tbt = s2r_bt.get_slice(tid)
    tXrA = tiled_mma.make_fragment_A(slc_mma.partition_A(sT))
    xAs = ta.partition_S(sT)
    xAr = ta.retile(tXrA)
    cW = slc_mma.partition_C(cute.make_identity_tensor((BT, dk)))
    cB = slc_mma.partition_B(cute.make_identity_tensor((dk, BT)))  # (key / v, token)
    sXt = cute.make_tensor(sX.iterator, cute.select(sX.layout, mode=[1, 0]))
    tXrB = tiled_mma.make_fragment_B(slc_mma.partition_B(sXt))
    xBs = tbt.partition_S(sXt)
    xBr = tbt.retile(tXrB)
    accw = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BT, dk)))
    accw.fill(0.0)
    for kb in cutlass.range_constexpr(BT // 16):
        cute.copy(s2r_a, xAs[None, None, kb], xAr[None, None, kb])
        cute.copy(s2r_bt, xBs[None, None, kb], xBr[None, None, kb])
        fb = tXrB[None, None, kb]
        cb = cB[None, None, kb]
        # an MMA atom's 4 B values per thread sit at 4 tokens, the same for every N tile:
        # read their factors once
        fbeta = [sb[cb[v][1]] for v in range(4)]
        fgam = [sgam[cb[v][1]] for v in range(4)] if const_expr(gamma) else None
        for e in cutlass.range_constexpr(cute.size(fb)):
            x = fb[e].to(Float32) * fbeta[e % 4]
            if const_expr(gamma):
                x = x * fgam[e % 4]
            fb[e] = x.to(BFloat16)
        cute.gemm(tiled_mma, accw, tXrA[None, None, kb], tXrB[None, None, kb], accw)
    cute.arch.sync_threads()  # every warp done with the tiles and with sX
    if const_expr(tma_out is None):
        _store_staged(accw, cW, sX, out, row0, valid, col0, tid, dk, 128)
    else:
        store_c(accw, cW, tiled_mma, sX, tma_out, t_out, out, row0, valid, col0, tid, dk, 128)
    cute.arch.sync_threads()


@cute.jit
def store_c(
    acc: cute.Tensor,
    coords: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tile: cute.Tensor,
    tma_out: cute.CopyAtom,
    t_out: cute.Tensor,
    out: cute.Tensor,
    row0: Int32,
    valid: Int32,
    col0: Int32,
    tid: Int32,
    width: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    """acc (tiled_mma's C fragment of [BT, width], ``coords`` its identity) as BF16 into out's
    rows row0 + i below ``valid`` at column col0: a whole chunk through st.matrix into ``tile``
    (out's TMA smem layout; free on entry) and one TMA store, a partial one through a stage over
    tile's bytes and row stores. All threads; synced inside."""
    if valid == BT:
        st_c = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), BFloat16), tiled_mma
        )
        stc = st_c.get_slice(tid)
        fc = cute.make_rmem_tensor_like(acc, BFloat16)
        for e in cutlass.range_constexpr(cute.size(acc)):
            fc[e] = acc[e].to(BFloat16)
        cute.copy(st_c, stc.retile(fc), stc.partition_D(tile))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
            g = cute.local_tile(cute.domain_offset((row0, col0), t_out), (BT, width), (0, 0))
            s_part, g_part = cpasync.tma_partition(
                tma_out,
                0,
                cute.make_layout(1),
                cute.group_modes(tile, 0, 2),
                cute.group_modes(g, 0, 2),
            )
            cute.copy(tma_out, s_part, g_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
    else:
        _store_staged(acc, coords, tile, out, row0, valid, col0, tid, width, threads)


@cute.jit
def _store_staged(
    acc: cute.Tensor,
    coords: cute.Tensor,
    tile: cute.Tensor,
    out: cute.Tensor,
    row0: Int32,
    valid: Int32,
    col0: Int32,
    tid: Int32,
    width: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    """acc's rows below ``valid`` to out through a stage_bf16 stage over tile's bytes."""
    stage = cute.make_tensor(
        cute.recast_ptr(tile.iterator, dtype=BFloat16), cute.make_layout(BT * width)
    )
    stage_bf16(acc, coords, stage, width)
    cute.arch.sync_threads()
    store_rows(stage, out, row0, valid, col0, tid, width, threads)


class IntraP:
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
        mG: cute.Tensor,
        mB: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mA: cute.Tensor,
        n_chunks: Int32,
        stream: cuda.CUstream,
    ):
        # mK [T, hk*dk], mV [T, hv*dv], mG/mB [T, hv] FP32, mCu [N+1] i32, mTab [n_chunks, 2] i32
        # mW [T, hv*dk], mU [T, hv*dv], mA [T, hv*BT] BF16
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, BFloat16)
        kv_layout = cute.tile_to_shape(atom, (BT, self.dk), order=(0, 1))
        t_layout = cute.tile_to_shape(atom, (BT, BT), order=(0, 1))
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, kv_layout, (BT, self.dk))
        tma_v, tV = cpasync.make_tiled_tma_atom(op, mV, kv_layout, (BT, self.dv))
        out = cpasync.CopyBulkTensorTileS2GOp()
        tma_w, tW = cpasync.make_tiled_tma_atom(out, mW, kv_layout, (BT, self.dk))
        tma_u, tU = cpasync.make_tiled_tma_atom(out, mU, kv_layout, (BT, self.dv))
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 1, 1)), permutation_mnk=(BT, 8, 16)
        )

        @cute.struct
        class Storage:
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(kv_layout)], 1024]
            lt: cute.struct.Align[cute.struct.MemRange[Float32, BT * LDL], 1024]
            g: cute.struct.MemRange[Float32, BT]
            b: cute.struct.MemRange[Float32, BT]
            bar: cute.struct.MemRange[cutlass.Int64, 2]

        self.storage = Storage
        self.kernel(
            tma_k, tK, tma_v, tV, tma_w, tW, tma_u, tU, mG, mB, mCu, mTab, mW, mU, mA, tiled_mma,
            kv_layout, t_layout,
        ).launch(grid=[n_chunks, self.hv, 1], block=[self.threads, 1, 1], stream=stream)  # fmt: skip

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
        mG: cute.Tensor,
        mB: cute.Tensor,
        mCu: cute.Tensor,
        mTab: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        kv_layout: cute.ComposedLayout,
        t_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        ci, h, _ = cute.arch.block_idx()
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
        hk = h // self.group

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()
        sg = st.g.get_tensor(cute.make_layout(BT))
        sb = st.b.get_tensor(cute.make_layout(BT))
        sK = st.sK.get_tensor(kv_layout.outer, swizzle=kv_layout.inner)
        sV = st.sV.get_tensor(kv_layout.outer, swizzle=kv_layout.inner)
        # lt[r, c]: T on and below the diagonal; above it L transposed (lt[j, i] = L[i, j]), whose
        # dead blocks later hold the doubling's products P transposed.
        lt = st.lt.get_tensor(cute.make_layout((BT, BT), stride=(LDL, 1)))

        if warp_i == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_i == 0:
            gK = cute.local_tile(cute.domain_offset((row0, 0), tK), (BT, self.dk), (0, hk))
            gV = cute.local_tile(cute.domain_offset((row0, 0), tV), (BT, self.dv), (0, h))
            tKs, tKg = cpasync.tma_partition(
                tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            tVs, tVg = cpasync.tma_partition(
                tma_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar, BT * self.dk * 2)
                cute.arch.mbarrier_arrive_and_expect_tx(bar + 1, BT * self.dv * 2)
            cute.copy(tma_k, tKg, tKs, tma_bar_ptr=bar)
            cute.copy(tma_v, tVg, tVs, tma_bar_ptr=bar + 1)
        if tid < BT:
            gv = Float32(0.0)
            bv = Float32(0.0)
            if tid < valid:
                gv = mG[row0 + tid, h]
                bv = mB[row0 + tid, h]
            sg[tid] = gv
            sb[tid] = bv
        cute.arch.mbarrier_wait(bar, 0)
        cute.arch.sync_threads()

        # ---- L = strict_lower(beta_i k_i.k_j 2^(g_i - g_j)) -> lt[j, i]
        slc_mma = tiled_mma.get_slice(tid)
        s2r_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tiled_mma
        )
        s2r_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BFloat16), tiled_mma
        )
        ta = s2r_a.get_slice(tid)
        tb = s2r_b.get_slice(tid)
        tCrA = tiled_mma.make_fragment_A(slc_mma.partition_A(sK))
        tCrB = tiled_mma.make_fragment_B(slc_mma.partition_B(sK))
        tAs = ta.partition_S(sK)
        tAr = ta.retile(tCrA)
        tBs = tb.partition_S(sK)
        tBr = tb.retile(tCrB)
        acc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BT, BT)))
        acc.fill(0.0)
        for kb in cutlass.range_constexpr(self.dk // 16):
            cute.copy(s2r_a, tAs[None, None, kb], tAr[None, None, kb])
            cute.copy(s2r_b, tBs[None, None, kb], tBr[None, None, kb])
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        cC = slc_mma.partition_C(cute.make_identity_tensor((BT, BT)))
        for e in cutlass.range_constexpr(cute.size(acc)):
            i, j = cC[e]
            if i > j:
                # Rows past the end are zero, not beta (0) times 2^(0 - g_j), which overflows.
                lv = Float32(0.0)
                if i < valid:
                    lv = acc[e] * sb[i] * cute.math.exp2(sg[i] - sg[j], fastmath=True)
                lt[j, i] = lv
        cute.arch.sync_threads()

        # ---- diagonal blocks: T_bb = (I + L_bb)^-1 by forward substitution (warp b, lane = column)
        b0 = warp_i * DIAG
        col = lane % DIAG
        tcol = cute.make_rmem_tensor((DIAG,), Float32)
        for r in cutlass.range_constexpr(DIAG):
            v = Float32(0.0)
            if col == r:
                v = Float32(1.0)
            for q in cutlass.range_constexpr(r):
                v = v - lt[b0 + q, b0 + r] * tcol[q]
            tcol[r] = v
        # The loads above compile to 16 B vectors that also fetch the diagonal written below.
        cute.arch.sync_warp()
        if lane < DIAG:
            for r in cutlass.range_constexpr(DIAG):
                if r >= col:
                    lt[b0 + r, b0 + col] = tcol[r]
        cute.arch.sync_threads()

        # ---- level 1 (16 -> 32), blocks (1,0) and (3,2) at offset o = 0 / 32, 4 entries per thread
        #   P = L_ij T_jj:  P[r, c] = sum_{q >= c} lt[o + q, o + 16 + r] lt[o + q, o + c]
        #   T_ij = -T_ii P: T[o + 16 + r, o + c] = -sum_{q <= r} lt[o + 16 + r, o + 16 + q] P[q, c]
        o = (tid // 64) * 32
        r1 = (tid % 64) // 4
        c1 = (tid % 4) * 4
        p1 = cute.make_rmem_tensor((4,), Float32)
        p1.fill(0.0)
        for q in cutlass.range_constexpr(DIAG):
            a = lt[o + q, o + DIAG + r1]
            for cc in cutlass.range_constexpr(4):
                if q >= c1 + cc:
                    p1[cc] = p1[cc] + a * lt[o + q, o + c1 + cc]
        cute.arch.sync_threads()
        for cc in cutlass.range_constexpr(4):
            lt[o + c1 + cc, o + DIAG + r1] = p1[cc]  # P[r, c] stored at lt[o + c, o + 16 + r]
        cute.arch.sync_threads()
        p1.fill(0.0)
        for q in cutlass.range_constexpr(DIAG):
            a = Float32(0.0)
            if q <= r1:
                a = lt[o + DIAG + r1, o + DIAG + q]
            for cc in cutlass.range_constexpr(4):
                p1[cc] = p1[cc] + a * lt[o + c1 + cc, o + DIAG + q]
        for cc in cutlass.range_constexpr(4):
            lt[o + DIAG + r1, o + c1 + cc] = -p1[cc]
        cute.arch.sync_threads()

        # ---- level 2 (32 -> 64): P2 = L_(B,A) T_A, T_(B,A) = -T_B P2, 8 entries per thread
        r2 = tid // 4
        c2 = (tid % 4) * 8
        p2 = cute.make_rmem_tensor((8,), Float32)
        p2.fill(0.0)
        for q in cutlass.range_constexpr(2 * DIAG):
            a = lt[q, 2 * DIAG + r2]
            for cc in cutlass.range_constexpr(8):
                if q >= c2 + cc:
                    p2[cc] = p2[cc] + a * lt[q, c2 + cc]
        cute.arch.sync_threads()
        for cc in cutlass.range_constexpr(8):
            lt[c2 + cc, 2 * DIAG + r2] = p2[cc]
        cute.arch.sync_threads()
        p2.fill(0.0)
        for q in cutlass.range_constexpr(2 * DIAG):
            a = Float32(0.0)
            if q <= r2:
                a = lt[2 * DIAG + r2, 2 * DIAG + q]
            for cc in cutlass.range_constexpr(8):
                p2[cc] = p2[cc] + a * lt[c2 + cc, 2 * DIAG + q]
        for cc in cutlass.range_constexpr(8):
            lt[2 * DIAG + r2, c2 + cc] = -p2[cc]
        cute.arch.sync_threads()

        # ---- A = bf16(T) out (pairs); T kept in registers (32 per thread) for the W / U factors
        treg = cute.make_rmem_tensor((32,), Float32)
        for e in cutlass.range_constexpr(16):
            pidx = tid + e * 128
            i = pidx // 32
            j = (pidx % 32) * 2
            t0 = Float32(0.0)
            t1 = Float32(0.0)
            if j <= i:
                t0 = lt[i, j]
            if j + 1 <= i:
                t1 = lt[i, j + 1]
            treg[2 * e] = t0
            treg[2 * e + 1] = t1
            if i < valid:
                pa = cute.make_rmem_tensor((2,), BFloat16)
                pa[0] = t0.to(BFloat16)
                pa[1] = t1.to(BFloat16)
                dst = cute.make_tensor(
                    mA.iterator + Int64(row0 + i) * mA.layout.stride[0] + h * BT + j,
                    cute.make_layout(2),
                )
                cute.autovec_copy(pa, dst)
        cute.arch.sync_threads()

        # ---- W = bf16(T) bf16(diag(beta 2^g) K) and U = bf16(T) bf16(diag(beta) V), as fla-core's
        # recompute_w_u forms them from A (so its backward's recompute matches): bf16(T) as a BF16
        # tile in lt's bytes, the rows of K / V scaled in place (thread tid owns column tid), the
        # results staged (16 B chunks, XOR-swizzled by row) into the dead K / V tile, then stored.
        lptr = cute.recast_ptr(st.lt.data_ptr(), dtype=BFloat16)
        sT = cute.make_tensor(cute.recast_ptr(lptr, t_layout.inner), t_layout.outer)
        for e in cutlass.range_constexpr(16):
            pidx = tid + e * 128
            i = pidx // 32
            j = (pidx % 32) * 2
            for k2 in cutlass.range_constexpr(2):
                sT[i, j + k2] = treg[2 * e + k2].to(BFloat16)
        if tid < BT:
            sg[tid] = cute.math.exp2(sg[tid], fastmath=False)  # g is done with; 2^g from here
        store_wu(
            sT, sK, sV, sb, sg, bar + 1, tiled_mma, s2r_a, mW, mU, row0, valid, h, tid, tma_w, tW,
            tma_u, tU,
        )  # fmt: skip


class RecurS:
    def __init__(self, hk: int, hv: int, bv: int = 64, dk: int = 128, dv: int = 128):
        assert dk == 128 and dv % bv == 0 and bv in (32, 64) and hv % hk == 0
        self.hk, self.hv, self.bv, self.dk, self.dv = hk, hv, bv, dk, dv
        self.group = hv // hk
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        n_seqs: Int32,
        stream: cuda.CUstream,
    ):
        # mW [T, hv*dk], mU [T, hv*dv], mQ/mK [T, hk*dk], mG [T, hv] FP32, mCu [N+1] i32,
        # mO [T, hv*dv] BF16
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, BFloat16)
        atom_v = (
            atom
            if const_expr(self.bv == 64)
            else warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW64, BFloat16)
        )
        lk = cute.tile_to_shape(atom, (BT, self.dk), order=(0, 1))  # W, Q, K
        lu = cute.tile_to_shape(atom_v, (BT, self.bv), order=(0, 1))  # U, V', V'', O stage
        ls = cute.tile_to_shape(atom_v, (self.dk, self.bv), order=(0, 1))  # bf16(S) [key][v]
        la = cute.tile_to_shape(atom, (BT, BT), order=(0, 1))  # A_qk
        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_w, tW = cpasync.make_tiled_tma_atom(op, mW, lk, (BT, self.dk))
        tma_q, tQ = cpasync.make_tiled_tma_atom(op, mQ, lk, (BT, self.dk))
        tma_k, tK = cpasync.make_tiled_tma_atom(op, mK, lk, (BT, self.dk))
        tma_u, tU = cpasync.make_tiled_tma_atom(op, mU, lu, (BT, self.bv))
        mma_op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tm_tok = cute.make_tiled_mma(
            mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=(BT, 16, 16)
        )  # M = tokens
        tm_key = cute.make_tiled_mma(
            mma_op, cute.make_layout((8, 1, 1)), permutation_mnk=(self.dk, 8, 16)
        )  # M = keys

        @cute.struct
        class Storage:
            sW: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lk)], 1024]
            sU: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sS: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(ls)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sD: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(lu)], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(la)], 1024]
            g: cute.struct.MemRange[Float32, 2 * BT]
            bar: cute.struct.MemRange[cutlass.Int64, 3]

        self.storage = Storage
        self.kernel(
            tma_w,
            tW,
            tma_q,
            tQ,
            tma_k,
            tK,
            tma_u,
            tU,
            mG,
            mCu,
            mO,
            scale,
            tm_tok,
            tm_key,
            lk,
            lu,
            ls,
            la,
        ).launch(
            grid=[self.dv // self.bv, self.hv, n_seqs], block=[self.threads, 1, 1], stream=stream
        )

    @cute.jit
    def _issue_wu(
        self,
        tma_w: cute.CopyAtom,
        tW: cute.Tensor,
        tma_u: cute.CopyAtom,
        tU: cute.Tensor,
        sW: cute.Tensor,
        sU: cute.Tensor,
        bar: cute.Pointer,
        row0: Int32,
        h: Int32,
        v0: Int32,
    ):
        gWc = cute.local_tile(cute.domain_offset((row0, 0), tW), (BT, self.dk), (0, h))
        gUc = cute.local_tile(
            cute.domain_offset((row0, h * self.dv + v0), tU), (BT, self.bv), (0, 0)
        )
        a1, b1 = cpasync.tma_partition(
            tma_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gWc, 0, 2)
        )
        a4, b4 = cpasync.tma_partition(
            tma_u, 0, cute.make_layout(1), cute.group_modes(sU, 0, 2), cute.group_modes(gUc, 0, 2)
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, (BT * self.dk + BT * self.bv) * 2)
        cute.copy(tma_w, b1, a1, tma_bar_ptr=bar)
        cute.copy(tma_u, b4, a4, tma_bar_ptr=bar)

    @cute.jit
    def _issue_q(
        self,
        tma_q: cute.CopyAtom,
        tQ: cute.Tensor,
        sQ: cute.Tensor,
        bar: cute.Pointer,
        row0: Int32,
        hk: Int32,
    ):
        gQc = cute.local_tile(cute.domain_offset((row0, 0), tQ), (BT, self.dk), (0, hk))
        a2, b2 = cpasync.tma_partition(
            tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQc, 0, 2)
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, BT * self.dk * 2)
        cute.copy(tma_q, b2, a2, tma_bar_ptr=bar)

    @cute.jit
    def _issue_k(
        self,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        sK: cute.Tensor,
        bar: cute.Pointer,
        row0: Int32,
        hk: Int32,
    ):
        gKc = cute.local_tile(cute.domain_offset((row0, 0), tK), (BT, self.dk), (0, hk))
        a3, b3 = cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gKc, 0, 2)
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, BT * self.dk * 2)
        cute.copy(tma_k, b3, a3, tma_bar_ptr=bar)

    @cute.kernel
    def kernel(
        self,
        tma_w: cute.CopyAtom,
        tW: cute.Tensor,
        tma_q: cute.CopyAtom,
        tQ: cute.Tensor,
        tma_k: cute.CopyAtom,
        tK: cute.Tensor,
        tma_u: cute.CopyAtom,
        tU: cute.Tensor,
        mG: cute.Tensor,
        mCu: cute.Tensor,
        mO: cute.Tensor,
        scale: Float32,
        tm_tok: cute.TiledMma,
        tm_key: cute.TiledMma,
        lk: cute.ComposedLayout,
        lu: cute.ComposedLayout,
        ls: cute.ComposedLayout,
        la: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        vs, h, n = cute.arch.block_idx()
        warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        hk = h // self.group
        v0 = vs * self.bv
        bos = mCu[n]
        eos = mCu[n + 1]
        nt = (eos - bos + BT - 1) // BT

        smem = SmemAllocator()
        st = smem.allocate(self.storage)
        bar = st.bar.data_ptr()  # 0: W + U, 1: Q, 2: K
        sgall = st.g.get_tensor(cute.make_layout((BT, 2), stride=(1, BT)))
        sW = st.sW.get_tensor(lk.outer, swizzle=lk.inner)
        sQ = st.sQ.get_tensor(lk.outer, swizzle=lk.inner)
        sK = st.sK.get_tensor(lk.outer, swizzle=lk.inner)
        sU = st.sU.get_tensor(lu.outer, swizzle=lu.inner)
        sS = st.sS.get_tensor(ls.outer, swizzle=ls.inner)
        sV = st.sV.get_tensor(lu.outer, swizzle=lu.inner)
        sD = st.sD.get_tensor(lu.outer, swizzle=lu.inner)
        sA = st.sA.get_tensor(la.outer, swizzle=la.inner)
        sSt = cute.make_tensor(sS.iterator, cute.select(sS.layout, mode=[1, 0]))  # [v][key]
        sKt = cute.make_tensor(sK.iterator, cute.select(sK.layout, mode=[1, 0]))  # [key][token]
        sVt = cute.make_tensor(sV.iterator, cute.select(sV.layout, mode=[1, 0]))  # [v][token]
        sDt = cute.make_tensor(sD.iterator, cute.select(sD.layout, mode=[1, 0]))

        if warp_i == 0:
            with cute.arch.elect_one():
                for q in cutlass.range_constexpr(3):
                    cute.arch.mbarrier_init(bar + q, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        slc_tok = tm_tok.get_slice(tid)
        slc_key = tm_key.get_slice(tid)
        ld_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), tm_tok
        )
        ld_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BFloat16), tm_tok
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
        lb_ = ld_b.get_slice(tid)
        lbt = ld_bt.get_slice(tid)
        latk = ld_at_k.get_slice(tid)
        lbtk = ld_bt_k.get_slice(tid)
        sto = st_tok.get_slice(tid)
        stk = st_key.get_slice(tid)

        cT = slc_tok.partition_C(cute.make_identity_tensor((BT, self.bv)))  # [token, v]
        cA = slc_tok.partition_C(cute.make_identity_tensor((BT, BT)))  # [token, token]
        acc_s = tm_key.make_fragment_C(tm_key.partition_shape_C((self.dk, self.bv)))
        acc_s.fill(0.0)
        fs = cute.make_rmem_tensor_like(acc_s, BFloat16)
        fs.fill(0.0)
        cute.copy(st_key, stk.retile(fs), stk.partition_D(sS))

        if warp_i == 0:
            if nt > 0:
                self._issue_wu(tma_w, tW, tma_u, tU, sW, sU, bar, bos, h, v0)
                self._issue_q(tma_q, tQ, sQ, bar + 1, bos, hk)
                self._issue_k(tma_k, tK, sK, bar + 2, bos, hk)
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
            cute.arch.sync_threads()  # g of this chunk visible; previous chunk fully done
            sg = sgall[None, ph]
            g_last = sg[valid - 1]
            # prefetch the next chunk's gates into the other half
            if tid < BT:
                gv = Float32(0.0)
                if c + 1 < nt:
                    if tid < eos - row0 - BT:
                        gv = mG[row0 + BT + tid, h]
                sgall[tid, 1 - ph] = gv
            cute.arch.mbarrier_wait(bar, ph)
            cute.arch.mbarrier_wait(bar + 1, ph)

            # QS = Q bf16(S), WS = W bf16(S)  (M = token, N = v, K = key)
            rQ = tm_tok.make_fragment_A(slc_tok.partition_A(sQ))
            rW = tm_tok.make_fragment_A(slc_tok.partition_A(sW))
            rS = tm_tok.make_fragment_B(slc_tok.partition_B(sSt))
            pQ = la_.partition_S(sQ)
            pW = la_.partition_S(sW)
            pS = lbt.partition_S(sSt)
            rQv = la_.retile(rQ)
            rWv = la_.retile(rW)
            rSv = lbt.retile(rS)
            acc_q = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, self.bv)))
            acc_w = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, self.bv)))
            acc_q.fill(0.0)
            acc_w.fill(0.0)
            for kb in cutlass.range_constexpr(self.dk // 16):
                cute.copy(ld_bt, pS[None, None, kb], rSv[None, None, kb])
                cute.copy(ld_a, pQ[None, None, kb], rQv[None, None, kb])
                cute.copy(ld_a, pW[None, None, kb], rWv[None, None, kb])
                cute.gemm(tm_tok, acc_q, rQ[None, None, kb], rS[None, None, kb], acc_q)
                cute.gemm(tm_tok, acc_w, rW[None, None, kb], rS[None, None, kb], acc_w)
            # V' = U - WS (rows past the end: 0), V'' = 2^(g_last - g) V'
            fv = cute.make_rmem_tensor_like(acc_w, BFloat16)
            fd = cute.make_rmem_tensor_like(acc_w, BFloat16)
            for e in cutlass.range_constexpr(cute.size(acc_w)):
                i, j = cT[e]
                vp = Float32(0.0)
                if i < valid:
                    vp = sU[i, j].to(Float32) - acc_w[e]
                fv[e] = vp.to(BFloat16)
                fd[e] = (vp * cute.math.exp2(g_last - sg[i], fastmath=True)).to(BFloat16)
            cute.copy(st_tok, sto.retile(fv), sto.partition_D(sV))
            cute.copy(st_tok, sto.retile(fd), sto.partition_D(sD))
            cute.arch.sync_threads()  # W, U consumed; V', V'' visible
            if warp_i == 0:
                if c + 1 < nt:
                    self._issue_wu(tma_w, tW, tma_u, tU, sW, sU, bar, row0 + BT, h, v0)

            # A_qk = bf16(Q Kᵀ ∘ 2^(g_i - g_j) ∘ causal)  (M = token, N = token, K = key)
            cute.arch.mbarrier_wait(bar + 2, ph)
            rK = tm_tok.make_fragment_B(slc_tok.partition_B(sK))
            pK = lb_.partition_S(sK)
            rKv = lb_.retile(rK)
            acc_a = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, BT)))
            acc_a.fill(0.0)
            for kb in cutlass.range_constexpr(self.dk // 16):
                cute.copy(ld_a, pQ[None, None, kb], rQv[None, None, kb])
                cute.copy(ld_b, pK[None, None, kb], rKv[None, None, kb])
                cute.gemm(tm_tok, acc_a, rQ[None, None, kb], rK[None, None, kb], acc_a)
            fa = cute.make_rmem_tensor_like(acc_a, BFloat16)
            for e in cutlass.range_constexpr(cute.size(acc_a)):
                i, j = cA[e]
                av = Float32(0.0)
                if i >= j:
                    if i < valid:
                        av = acc_a[e] * cute.math.exp2(sg[i] - sg[j], fastmath=True)
                fa[e] = av.to(BFloat16)
            cute.copy(st_tok, sto.retile(fa), sto.partition_D(sA))
            cute.arch.sync_threads()  # A_qk visible; Q consumed
            if warp_i == 0:
                if c + 1 < nt:
                    self._issue_q(tma_q, tQ, sQ, bar + 1, row0 + BT, hk)

            # O = scale (2^g QS + A_qk V')  (M = token, N = v, K = token)
            rA = tm_tok.make_fragment_A(slc_tok.partition_A(sA))
            rV = tm_tok.make_fragment_B(slc_tok.partition_B(sVt))
            pA = la_.partition_S(sA)
            pV = lbt.partition_S(sVt)
            rAv = la_.retile(rA)
            rVv = lbt.retile(rV)
            acc_o = tm_tok.make_fragment_C(tm_tok.partition_shape_C((BT, self.bv)))
            acc_o.fill(0.0)
            for kb in cutlass.range_constexpr(BT // 16):
                cute.copy(ld_a, pA[None, None, kb], rAv[None, None, kb])
                cute.copy(ld_bt, pV[None, None, kb], rVv[None, None, kb])
                cute.gemm(tm_tok, acc_o, rA[None, None, kb], rV[None, None, kb], acc_o)

            # S = 2^(g_last) S + Kᵀ V''  (M = key, N = v, K = token)
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
            cute.arch.sync_threads()  # K, V', V'', A_qk, bf16(S) all consumed
            if warp_i == 0:
                if c + 1 < nt:
                    self._issue_k(tma_k, tK, sK, bar + 2, row0 + BT, hk)
            for e in cutlass.range_constexpr(cute.size(acc_s)):
                fs[e] = acc_s[e].to(BFloat16)
            cute.copy(st_key, stk.retile(fs), stk.partition_D(sS))
            # O staged into the V' tile (16 B chunks XOR-swizzled by row), then stored
            stage = cute.make_tensor(
                cute.recast_ptr(sV.iterator, dtype=BFloat16), cute.make_layout(BT * self.bv)
            )
            chunks = self.bv // 8
            for e in cutlass.range_constexpr(cute.size(acc_o)):
                i, j = cT[e]
                o = (acc_q[e] * cute.math.exp2(sg[i], fastmath=True) + acc_o[e]) * scale
                stage[i * self.bv + ((j // 8) ^ (i % chunks)) * 8 + j % 8] = o.to(BFloat16)
            cute.arch.sync_threads()
            for e in cutlass.range_constexpr(BT * chunks // 256):
                idx = tid + e * 256
                r = idx // chunks
                ch = idx % chunks
                if r < valid:
                    src = cute.make_tensor(
                        stage.iterator + r * self.bv + (ch ^ (r % chunks)) * 8, cute.make_layout(8)
                    )
                    dst = cute.make_tensor(
                        mO.iterator
                        + Int64(row0 + r) * mO.layout.stride[0]
                        + h * self.dv
                        + v0
                        + ch * 8,
                        cute.make_layout(8),
                    )
                    cute.autovec_copy(src, dst)
