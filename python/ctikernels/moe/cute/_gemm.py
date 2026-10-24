# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Grouped BF16 GEMMs with gathered rows and an asynchronous shared-memory ring.

Design reference: NVIDIA CUTLASS CuTe DSL ampere/tensorop_gemm.py (tiled MMA,
ldmatrix partitions and multistage buffering). This is an independent implementation.
https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere
The manual cp.async addressing follows the public NVIDIA PTX ISA.

B is always logically (N,K). MN-major B has its OWN swizzled (N,K) layout;
it is never a transposed view of a K-major swizzle. Both copies and ldmatrix
partition that exact layout. Gate/up and backward recompute call the same
mainloop with the same config, including MMA order and BF16 rounding.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint64, const_expr


@cute.jit
def _copy16(dst, src, offset, valid):
    address = src.toint() + (Int64(offset) if valid else Int64(0)) * 2
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[Int32(dst.toint()), address, Int32(16) if valid else Int32(0)],
    )


@cute.jit
def _shared_pair(dst, lo, hi):
    cute.arch.inline_ptx(
        "{ .reg .b32 v; cvt.rn.bf16x2.f32 v, {$r2}, {$r1}; st.shared.b32 [{$r0}], v; }",
        read_only_args=[Int32(dst.toint()), Float32(lo), Float32(hi)],
    )


@cute.jit
def _store16(dst, src):
    cute.arch.inline_ptx(
        "{ .reg .b32 v<4>; ld.shared.v4.b32 {v0,v1,v2,v3}, [{$r1}]; "
        "st.global.v4.b32 [{$r0}], {v0,v1,v2,v3}; }",
        read_only_args=[dst.toint(), Int32(src.toint())],
    )


@cute.jit
def _sigmoid(x):
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


@cute.jit
def _silu(x):
    # torch's F.silu: x / (1 + expf(-x)) with the accurate expf and an IEEE divide.
    return x / (1.0 + cute.math.exp(-x, fastmath=False))


class GroupedGemm:
    def __init__(
        self,
        kind,
        config,
        h,
        ni,
        nis,
        e,
        kp1,
        saved,
        has_shared,
        pdl=False,
    ):
        self.kind, self.cfg = kind, config
        self.h, self.ni, self.nis = h, ni, nis
        self.nmax, self.e, self.kp1 = max(ni, nis), e, kp1
        self.saved, self.has_shared = saved, has_shared
        self.pdl = pdl

    def layout(self, rows, mn=False, tma=False):
        c = self.cfg
        # A 64-element contiguous atom uses XOR of the eight 16-byte
        # segments with the other dimension's low three bits. Each cp.async
        # 128-byte transaction and each ldmatrix 8-row address group therefore
        # visits all 32 banks once. BN32 uses four segments (Swizzle<2,3,3>),
        # not a padded BN64 atom: cp.async and ldmatrix.trans must agree on
        # that physical N-major layout. BK128 tiles two K-major 64 atoms.
        width = min(64, c.bn if mn else c.bk)
        atom = (
            cute.make_layout((width, 8), stride=(1, width))
            if mn
            else cute.make_layout((8, width), stride=(width, 1))
        )
        # TMA descriptors and swizzled pointers use BYTE swizzles. The legacy
        # composed tensor layout instead swizzles BF16 ELEMENT offsets.
        sw = cute.make_composed_layout(
            cute.make_swizzle(3 if width == 64 else 2, 4 if tma else 3, 3), 0, atom
        )
        return cute.tile_to_shape(
            sw, (rows, c.bk, c.stages), (1, 0, 2) if tma and mn else (0, 1, 2)
        )

    @cute.jit
    def weight_descriptor(self, weight, layout, mn: cutlass.Constexpr, shared: cutlass.Constexpr):
        # Normalize physical [E,row,col] / [row,col] to logical [N,K,E].
        r, k = (1, 0) if const_expr(mn) else (0, 1)
        offset = 0 if const_expr(shared) else 1
        shape = (weight.shape[r + offset], weight.shape[k + offset])
        stride = (weight.stride[r + offset], weight.stride[k + offset])
        logical = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (*shape, 1 if const_expr(shared) else weight.shape[0]),
                stride=(*stride, 0 if const_expr(shared) else weight.stride[0]),
            ),
        )
        atom, tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            logical,
            cute.slice_(layout, (None, None, 0)),
            (self.cfg.bn, self.cfg.bk),
        )
        # A tile wider than the swizzle is split into full swizzle-width boxes.
        # Catch accidental element/byte swizzle mismatches, which otherwise
        # silently lower to tiny transactions with unaligned destinations.
        contiguous = self.cfg.bn if const_expr(mn) else self.cfg.bk
        other = self.cfg.bk if const_expr(mn) else self.cfg.bn
        assert cute.size(atom.layout_src_tv) == min(64, contiguous) * other
        return atom, tensor

    @cute.jit
    def weight_descriptors(self, weight, shared_weight, layout, mn: cutlass.Constexpr):
        routed = self.weight_descriptor(weight, layout, mn, False)
        shared = None
        if const_expr(self.has_shared):
            shared = self.weight_descriptor(shared_weight, layout, mn, True)
        return routed, shared

    @cute.jit
    def __call__(
        self, a, b, sb, slots, experts, ws, out, gs, us, dw, x, w13, s13, stream: cuda.CUstream
    ):
        c = self.cfg
        mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            cute.make_layout((c.wm, c.wn, 1)),
            permutation_mnk=(16 * c.wm, min(c.bn, c.wn * 16), 16),
        )
        la = self.layout(c.bm)
        lb = self.layout(c.bn, self.kind in ("dx_pair", "bwd_slot"), c.tma)
        lg = self.layout(c.bn, tma=c.tma)
        weights, gates = None, None
        if const_expr(c.tma):
            weights = self.weight_descriptors(b, sb, lb, self.kind in ("dx_pair", "bwd_slot"))
            if const_expr(self.kind == "bwd_slot" and not self.saved):
                gates = self.weight_descriptors(w13, s13, lg, False)
        cols = self.nmax if self.kind in ("gate_up", "bwd_slot") else self.h
        grid = (experts.shape[0], cols // c.bn, 1)
        if const_expr(c.raster == "n"):
            grid = (cols // c.bn, experts.shape[0], 1)
        self.kernel(
            a.iterator,
            b.iterator,
            sb.iterator,
            slots.iterator,
            experts.iterator,
            ws.iterator,
            out.iterator,
            gs.iterator,
            us.iterator,
            dw.iterator,
            x.iterator,
            w13.iterator,
            s13.iterator,
            mma,
            la,
            lb,
            lg,
            weights,
            gates,
        ).launch(
            grid=grid,
            block=(c.warps * 32, 1, 1),
            smem=c.shared_bytes(self.kind, self.saved),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def stage_a(self, sa, a, slots, row0, n, kt, total, tid, mode: cutlass.Constexpr):
        c = self.cfg
        stage = kt % c.stages
        for j in cutlass.range_constexpr(cute.ceil_div(c.bm * c.bk // 8, c.warps * 32)):
            v = tid + j * c.warps * 32
            r, k = v // (c.bk // 8), v % (c.bk // 8) * 8
            if r < c.bm:
                slot = slots[r]
                kk = kt * c.bk + k
                if const_expr(mode in ("gate", "q")):
                    off = Int64((slot if slot >= 0 else Int32(0)) // self.kp1) * self.h + kk
                elif const_expr(mode == "down"):
                    off = Int64(row0 + r) * self.nmax + kk
                else:
                    col = kk if kk < n else kk - n + self.nmax
                    off = Int64(row0 + r) * (2 * self.nmax) + col
                _copy16(sa.iterator + sa.layout((r, k, stage)), a, off, (slot >= 0) & (kt < total))

    @cute.jit
    def copy_weight_tile(self, descriptor, dst, barrier, expert, nt, kt):
        atom, tensor = descriptor
        tile = cute.local_tile(tensor[None, None, expert], (self.cfg.bn, self.cfg.bk), (nt, kt))
        smem, gmem = cute.nvgpu.cpasync.tma_partition(
            atom, 0, cute.make_layout(1), cute.group_modes(dst, 0, 2), cute.group_modes(tile, 0, 2)
        )
        # The DSL elects the issuing lane internally; call from a full warp.
        cute.copy(atom, gmem, smem, tma_bar_ptr=barrier)

    @cute.jit
    def stage_b_tma(
        self, sbb, su, weights, barriers, e, n0, n, kt, total, tid, mode: cutlass.Constexpr
    ):
        c = self.cfg
        stage = kt % c.stages
        # Do not launch unused tail transactions: all issued writes must have
        # been waited before the workspace becomes epilogue scratch.
        if (tid < 32) & (kt < total):
            barrier = barriers + stage
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    barrier, c.bn * c.bk * 2 * (2 if const_expr(mode == "gate") else 1)
                )
            if const_expr(self.has_shared):
                if e == weights[0][1].shape[2]:
                    self.copy_weight_tile(
                        weights[1], sbb[None, None, stage], barrier, 0, n0 // c.bn, kt
                    )
                    if const_expr(mode == "gate"):
                        self.copy_weight_tile(
                            weights[1], su[None, None, stage], barrier, 0, (n0 + n) // c.bn, kt
                        )
                else:
                    self.copy_weight_tile(
                        weights[0], sbb[None, None, stage], barrier, e, n0 // c.bn, kt
                    )
                    if const_expr(mode == "gate"):
                        self.copy_weight_tile(
                            weights[0], su[None, None, stage], barrier, e, (n0 + n) // c.bn, kt
                        )
            else:
                self.copy_weight_tile(
                    weights[0], sbb[None, None, stage], barrier, e, n0 // c.bn, kt
                )
                if const_expr(mode == "gate"):
                    self.copy_weight_tile(
                        weights[0], su[None, None, stage], barrier, e, (n0 + n) // c.bn, kt
                    )

    @cute.jit
    def stage_b(
        self, sbb, su, b, n0, n, kt, total, tid, mode: cutlass.Constexpr, weights, barriers, e
    ):
        c = self.cfg
        if const_expr(c.tma):
            self.stage_b_tma(sbb, su, weights, barriers, e, n0, n, kt, total, tid, mode)
        else:
            self.stage_b_async(sbb, su, b, n0, n, kt, total, tid, mode)

    @cute.jit
    def stage_b_async(self, sbb, su, b, n0, n, kt, total, tid, mode: cutlass.Constexpr):
        c = self.cfg
        stage = kt % c.stages
        for j in cutlass.range_constexpr(cute.ceil_div(c.bn * c.bk // 8, c.warps * 32)):
            v = tid + j * c.warps * 32
            if const_expr(mode in ("q", "dx")):
                nn, k = v % (c.bn // 8) * 8, v // (c.bn // 8)
                stride = n if const_expr(mode == "q") else self.h
                off = Int64(kt * c.bk + k) * stride + n0 + nn
            else:
                nn, k = v // (c.bk // 8), v % (c.bk // 8) * 8
                stride = self.h if const_expr(mode == "gate") else n
                off = Int64(n0 + nn) * stride + kt * c.bk + k
            if v < c.bn * c.bk // 8:
                _copy16(sbb.iterator + sbb.layout((nn, k, stage)), b, off, kt < total)
                if const_expr(mode == "gate"):
                    _copy16(
                        su.iterator + su.layout((nn, k, stage)),
                        b,
                        off + Int64(n) * self.h,
                        kt < total,
                    )

    @cute.jit
    def stage(
        self,
        sa,
        sbb,
        su,
        a,
        b,
        slots,
        row0,
        n0,
        n,
        kt,
        total,
        tid,
        mode: cutlass.Constexpr,
        weights,
        barriers,
        e,
    ):
        if const_expr(self.cfg.tma):
            self.stage_b(sbb, su, b, n0, n, kt, total, tid, mode, weights, barriers, e)
            self.stage_a(sa, a, slots, row0, n, kt, total, tid, mode)
        else:
            self.stage_a(sa, a, slots, row0, n, kt, total, tid, mode)
            self.stage_b(sbb, su, b, n0, n, kt, total, tid, mode, weights, barriers, e)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def multiply(
        self,
        storage,
        a,
        b,
        slots,
        row0,
        n0,
        n,
        tid,
        mma,
        la,
        lb,
        mode: cutlass.Constexpr,
        weights,
        barriers,
        e,
    ):
        c = self.cfg
        sa = cute.make_tensor(storage, la)
        if const_expr(c.tma):
            sbb = cute.make_tensor(
                cute.recast_ptr(storage + cute.cosize(la), swizzle_=lb.inner), lb.outer
            )
            su = cute.make_tensor(
                cute.recast_ptr(storage + cute.cosize(la) + cute.cosize(lb), swizzle_=lb.inner),
                lb.outer,
            )
        else:
            sbb = cute.make_tensor(storage + cute.cosize(la), lb)
            su = cute.make_tensor(storage + cute.cosize(la) + cute.cosize(lb), lb)
        mma_slice = mma.get_slice(tid)
        ra = mma.make_fragment_A(mma_slice.partition_A(sa)[None, None, None, 0])
        rb = mma.make_fragment_B(mma_slice.partition_B(sbb)[None, None, None, 0])
        rc = cute.make_rmem_tensor(
            mma_slice.partition_C(cute.make_identity_tensor((c.bm, c.bn))).shape, Float32
        )
        rc.fill(0)
        ru = cute.make_rmem_tensor(rc.shape, Float32)
        ru.fill(0)
        lca = cute.make_tiled_copy_A(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma
        )
        lcb = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(mode in ("q", "dx"), 4), BFloat16
            ),
            mma,
        )
        ca, cb = lca.get_slice(tid), lcb.get_slice(tid)
        va, vb = ca.partition_S(sa), cb.partition_S(sbb)
        da, db = ca.retile(ra), cb.retile(rb)
        if const_expr(mode == "gate"):
            rbu = mma.make_fragment_B(mma_slice.partition_B(su)[None, None, None, 0])
            vu, du = cb.partition_S(su), cb.retile(rbu)
        total = self.h // c.bk if const_expr(mode in ("gate", "q")) else n // c.bk
        if const_expr(mode == "dx"):
            total = 2 * n // c.bk
        if const_expr(self.pdl and mode in ("down", "dx")):
            # Only weights and the older layout may be read before the wait.
            # Without TMA, B joins A0 in group 0; groups 1..S-2 contain only A.
            # Thus wait_group(S-2) retires B for EVERY prefetched stage plus A0,
            # and subsequent waits retire each matching A before it is consumed.
            # TMA has its own per-stage transaction barrier, independent of A.
            for p in cutlass.range_constexpr(c.stages - 1):
                self.stage_b(sbb, su, b, n0, n, Int32(p), total, tid, mode, weights, barriers, e)
            cute.arch.griddepcontrol_wait()
            for p in cutlass.range_constexpr(c.stages - 1):
                self.stage_a(sa, a, slots, row0, n, Int32(p), total, tid, mode)
                cute.arch.cp_async_commit_group()
        else:
            for p in cutlass.range_constexpr(c.stages - 1):
                self.stage(
                    sa,
                    sbb,
                    su,
                    a,
                    b,
                    slots,
                    row0,
                    n0,
                    n,
                    Int32(p),
                    total,
                    tid,
                    mode,
                    weights,
                    barriers,
                    e,
                )
        for kt in range(total):
            stage = kt % c.stages
            cute.arch.cp_async_wait_group(c.stages - 2)
            if const_expr(c.tma):
                phase = kt // c.stages
                if const_expr(mode == "q" and not self.saved):
                    # Recompute has already used each barrier this many times.
                    # Keep its completed phase instead of reinitializing it.
                    phase += (self.h // c.bk + c.stages - 1 - stage) // c.stages
                cute.arch.mbarrier_wait(barriers + stage, phase & 1)
            cute.arch.sync_threads()
            # Retire old readers, then issue the next group before ldmatrix.
            # The destination is the previous iteration's stage, never this one.
            self.stage(
                sa,
                sbb,
                su,
                a,
                b,
                slots,
                row0,
                n0,
                n,
                kt + c.stages - 1,
                total,
                tid,
                mode,
                weights,
                barriers,
                e,
            )
            # The barrier also retires readers before the ring slot is reused.
            cute.copy(lca, va[None, None, 0, stage], da[None, None, 0])
            cute.copy(lcb, vb[None, None, 0, stage], db[None, None, 0])
            if const_expr(mode == "gate"):
                cute.copy(lcb, vu[None, None, 0, stage], du[None, None, 0])
            for kk in cutlass.range_constexpr(c.bk // 16):
                # Adjacent K fragments occupy different registers, allowing the
                # following ldmatrix instructions to overlap the current MMA.
                if const_expr(kk + 1 < c.bk // 16):
                    cute.copy(lca, va[None, None, kk + 1, stage], da[None, None, kk + 1])
                    cute.copy(lcb, vb[None, None, kk + 1, stage], db[None, None, kk + 1])
                    if const_expr(mode == "gate"):
                        cute.copy(lcb, vu[None, None, kk + 1, stage], du[None, None, kk + 1])
                cute.gemm(mma, rc, ra[None, None, kk], rb[None, None, kk], rc)
                if const_expr(mode == "gate"):
                    cute.gemm(mma, ru, ra[None, None, kk], rbu[None, None, kk], ru)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        return rc, ru

    @cute.jit
    def epilogue(self, storage, coord, values, out, slots, row0, n0, stride, tid):
        c = self.cfg
        width = min(c.bn, 64)
        layout = cute.tile_to_shape(
            cute.make_composed_layout(
                cute.make_swizzle(3 if width == 64 else 2, 3, 3),
                0,
                cute.make_layout((8, width), stride=(width, 1)),
            ),
            (c.bm, c.bn),
            (0, 1),
        )
        # The drained ring is scratch now. Packed C pairs scatter without bank
        # conflicts; the swizzle leaves each aligned 16-byte segment contiguous.
        for i in cutlass.range_constexpr(cute.size(coord) // 2):
            r, col = coord[2 * i]
            _shared_pair(storage + layout((r, col)), values[2 * i], values[2 * i + 1])
        cute.arch.sync_threads()
        for j in cutlass.range_constexpr(cute.ceil_div(c.bm * c.bn // 8, c.warps * 32)):
            v = tid + j * c.warps * 32
            r, col = v // (c.bn // 8), v % (c.bn // 8) * 8
            if r < c.bm:
                if slots[r] >= 0:
                    off = Int64(row0 + r) * stride + n0 + col
                    _store16(out + off, storage + layout((r, col)))
        # All readers must retire before another output or dw reduction reuses it.
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        a: cute.Pointer,
        b: cute.Pointer,
        sb: cute.Pointer,
        slots: cute.Pointer,
        experts: cute.Pointer,
        ws: cute.Pointer,
        out: cute.Pointer,
        gs: cute.Pointer,
        us: cute.Pointer,
        dw: cute.Pointer,
        x: cute.Pointer,
        w13: cute.Pointer,
        s13: cute.Pointer,
        mma: cute.TiledMma,
        la: cute.ComposedLayout,
        lb: cute.ComposedLayout,
        lg: cute.ComposedLayout,
        weights,
        gates,
    ):
        if const_expr(self.pdl and self.kind in ("gate_up", "bwd_slot")):
            cute.arch.griddepcontrol_wait()
        c = self.cfg
        tid, _, _ = cute.arch.thread_idx()
        pid_m, pid_n, _ = cute.arch.block_idx()
        if const_expr(c.raster == "n"):
            pid_m, pid_n = pid_n, pid_m
        e = experts[pid_m]
        expert_count = weights[0][1].shape[2] if const_expr(c.tma) else self.e
        n = self.nis if e == expert_count else self.ni
        n0, row0 = pid_n * c.bn, pid_m * c.bm
        live = e <= expert_count if const_expr(self.has_shared) else e < expert_count
        if const_expr(self.kind in ("gate_up", "bwd_slot")):
            live = live & (n0 < n)
        if live:
            alloc = utils.SmemAllocator()
            slotbuf = alloc.allocate_tensor(Int32, cute.make_layout(c.bm), byte_alignment=16)
            storage = alloc.allocate_tensor(
                BFloat16,
                cute.make_layout(c.workspace_bytes(self.kind, self.saved) // 2),
                byte_alignment=1024 if const_expr(c.tma) else 128,
            )
            barriers = None
            if const_expr(c.tma):
                barriers = alloc.allocate_tensor(
                    Uint64, cute.make_layout(c.stages), byte_alignment=8
                ).iterator
                if tid < 32:
                    with cute.arch.elect_one():
                        for s in cutlass.range_constexpr(c.stages):
                            cute.arch.mbarrier_init(barriers + s, 1)
                cute.arch.mbarrier_init_fence()
            for j in cutlass.range_constexpr(cute.ceil_div(c.bm, c.warps * 32)):
                r = tid + j * c.warps * 32
                if r < c.bm:
                    slotbuf[r] = slots[row0 + r]
            cute.arch.sync_threads()
            stride = (
                2 * self.ni * self.h
                if const_expr(self.kind in ("gate_up", "dx_pair"))
                else self.h * self.ni
            )
            bp = (b + Int64(e) * stride).align(16)
            if const_expr(self.has_shared):
                if e == expert_count:
                    bp = sb
            coord = mma.get_slice(tid).partition_C(cute.make_identity_tensor((c.bm, c.bn)))
            if const_expr(self.kind == "gate_up"):
                acc, up = self.multiply(
                    storage.iterator,
                    a,
                    bp,
                    slotbuf,
                    row0,
                    n0,
                    n,
                    tid,
                    mma,
                    la,
                    lb,
                    "gate",
                    weights,
                    barriers,
                    e,
                )
            elif const_expr(self.kind == "down"):
                acc, _ = self.multiply(
                    storage.iterator,
                    a,
                    bp,
                    slotbuf,
                    row0,
                    n0,
                    n,
                    tid,
                    mma,
                    la,
                    lb,
                    "down",
                    weights,
                    barriers,
                    e,
                )
            elif const_expr(self.kind == "dx_pair"):
                acc, _ = self.multiply(
                    storage.iterator,
                    a,
                    bp,
                    slotbuf,
                    row0,
                    n0,
                    n,
                    tid,
                    mma,
                    la,
                    lb,
                    "dx",
                    weights,
                    barriers,
                    e,
                )
            else:
                if const_expr(not self.saved):
                    gp = (w13 + Int64(e) * (2 * self.ni * self.h)).align(16)
                    if const_expr(self.has_shared):
                        if e == expert_count:
                            gp = s13
                    gate, up = self.multiply(
                        storage.iterator,
                        x,
                        gp,
                        slotbuf,
                        row0,
                        n0,
                        n,
                        tid,
                        mma,
                        la,
                        lg,
                        "gate",
                        gates,
                        barriers,
                        e,
                    )
                    # Round before q's mainloop, matching forward's exact rounding.
                    gate.store(gate.load().to(BFloat16).to(Float32))
                    up.store(up.load().to(BFloat16).to(Float32))
                else:
                    graw = cute.make_rmem_tensor(coord.shape, BFloat16)
                    uraw = cute.make_rmem_tensor(coord.shape, BFloat16)
                    graw.fill(0)
                    uraw.fill(0)
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32
                    )
                    for i in cutlass.range_constexpr(cute.size(coord) // 2):
                        r, col = coord[2 * i]
                        if slotbuf[r] >= 0:
                            off = cute.assume(Int64(row0 + r) * self.nmax + n0 + col, divby=2)
                            gpair = cute.make_rmem_tensor((2,), BFloat16)
                            upair = cute.make_rmem_tensor((2,), BFloat16)
                            cute.copy(
                                atom, cute.make_tensor(gs + off, cute.make_layout((2,))), gpair
                            )
                            cute.copy(
                                atom, cute.make_tensor(us + off, cute.make_layout((2,))), upair
                            )
                            graw[2 * i], graw[2 * i + 1] = gpair[0], gpair[1]
                            uraw[2 * i], uraw[2 * i + 1] = upair[0], upair[1]
                acc, _ = self.multiply(
                    storage.iterator,
                    a,
                    bp,
                    slotbuf,
                    row0,
                    n0,
                    n,
                    tid,
                    mma,
                    la,
                    lb,
                    "q",
                    weights,
                    barriers,
                    e,
                )
                if const_expr(self.saved):
                    # Defer conversion so saved loads can overlap q's mainloop.
                    gate = cute.make_rmem_tensor(coord.shape, Float32)
                    up = cute.make_rmem_tensor(coord.shape, Float32)
                    gate.store(graw.load().to(Float32))
                    up.store(uraw.load().to(Float32))
            if const_expr(self.pdl):
                # The wait has completed, including both mainloops on recompute.
                # Dependents can fetch weights while this CTA writes its outputs.
                cute.arch.griddepcontrol_launch_dependents()
            result = cute.make_rmem_tensor(coord.shape, Float32)
            if const_expr(self.kind == "bwd_slot"):
                result_u = cute.make_rmem_tensor(coord.shape, Float32)
                products = cute.make_rmem_tensor(coord.shape, Float32)
                partial = cute.make_tensor(
                    cute.recast_ptr(storage.iterator, dtype=Float32),
                    cute.make_layout((c.bm, c.wn), stride=(1, c.bm)),
                )
            for i in cutlass.range_constexpr(cute.size(coord)):
                r, col = coord[i]
                slot = slotbuf[r]
                if const_expr(self.kind == "gate_up"):
                    g, u = Float32(BFloat16(acc[i])), Float32(BFloat16(up[i]))
                    result[i] = Float32(BFloat16(_silu(g))) * u
                elif const_expr(self.kind in ("down", "dx_pair")):
                    result[i] = acc[i]
                else:
                    weight = Float32(0)
                    if slot >= 0:
                        weight = ws[slot]
                    # Autograd through BF16 SiLU and BF16 SiLU*up (Unsloth's rounding).
                    g, u = gate[i], up[i]
                    s = _sigmoid(g)
                    silu_b = Float32(BFloat16(_silu(g)))
                    ab = Float32(BFloat16(silu_b * u))
                    products[i] = acc[i] * ab
                    da = acc[i] * weight
                    result[i] = da * u * s * (1.0 + g * (1.0 - s))
                    result_u[i] = da * silu_b
            stride = self.nmax if const_expr(self.kind == "gate_up") else self.h
            if const_expr(self.kind == "bwd_slot"):
                stride = 2 * self.nmax
            self.epilogue(storage.iterator, coord, result, out, slotbuf, row0, n0, stride, tid)
            if const_expr(self.kind == "gate_up" and self.saved):
                self.epilogue(storage.iterator, coord, acc, gs, slotbuf, row0, n0, stride, tid)
                self.epilogue(storage.iterator, coord, up, us, slotbuf, row0, n0, stride, tid)
            if const_expr(self.kind == "bwd_slot"):
                self.epilogue(
                    storage.iterator,
                    coord,
                    result_u,
                    out + self.nmax,
                    slotbuf,
                    row0,
                    n0,
                    stride,
                    tid,
                )
            if const_expr(self.kind == "bwd_slot"):
                # Each MMA quad owns two columns of each of its rows. Reduce
                # its N fragments, then the four lanes. The MMA warp layout is
                # (WM, WN, 1), with M fastest; merge N-warps in fixed order.
                for m in cutlass.range_constexpr(cute.size(coord, mode=[1])):
                    for half in cutlass.range_constexpr(2):
                        v = Float32(0)
                        for nc in cutlass.range_constexpr(cute.size(coord, mode=[2])):
                            v += products[(0, half), m, nc] + products[(1, half), m, nc]
                        v += cute.arch.shuffle_sync_bfly(v, 1)
                        v += cute.arch.shuffle_sync_bfly(v, 2)
                        r = coord[(0, half), m, 0][0]
                        if tid % 4 == 0:
                            partial[r, tid // (32 * c.wm)] = v
                cute.arch.sync_threads()
                if tid < c.bm:
                    v = Float32(0)
                    for nw in cutlass.range_constexpr(c.wn):
                        v += partial[tid, nw]
                    if slotbuf[tid] >= 0:
                        dw[Int64(row0 + tid) * (self.nmax // c.bn) + pid_n] = v
        else:
            # Padding CTAs must also wait before signaling, even without a mainloop.
            if const_expr(self.pdl):
                if const_expr(self.kind in ("down", "dx_pair")):
                    cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
