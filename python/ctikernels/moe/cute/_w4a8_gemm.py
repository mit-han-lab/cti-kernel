# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# kernel_moe_mxfp8_mxfp4_fc1_act_q1.py, kernel_moe_mxfp8_mxfp4_fc2_finalize.py).
# Local modifications: L2 row prefetch, producer/consumer rings, equal K32 scale cadence,
# native gate/up addressing, masked epilogues, CTI compilation.

"""Warp-specialized persistent mixed-MMA engine. Imported only during GPU preparation."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int8, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.memory import SmemAllocator

from ._w4a8_debug import device_debug, validate_debug
from ._w4a8_epilogue import device_epilogues
from ._w4a8_layout import dsl_layouts, sf_cosize
from ._w4a8_schedule import device_schedule
from ._w4a8_watchdog import device_watchdog

timer, fold_fragment, store_sink = device_debug()
stamp, wait_stage, cta_barrier = device_watchdog()
publish = device_schedule()
stage_fc1, quantize, stage_fc2, scatter, lora_seed, lora_mma_seed, lora_partials = (
    device_epilogues()
)


@cute.jit
def partition_tma(atom, source, shared, rows: cutlass.Constexpr, width: cutlass.Constexpr):
    tiled = cute.local_tile(source, (rows, width), (None, None, None))
    return cpasync.tma_partition(
        atom,
        Int32(0),
        cute.make_layout(1),
        cute.group_modes(shared, 0, 2),
        cute.group_modes(tiled, 0, 2),
    )


@cute.jit
def copy_weight(
    atom,
    source,
    shared,
    barrier,
    stage,
    expert,
    ntile,
    tile_k,
    ntiles: cutlass.Constexpr,
    total_k: cutlass.Constexpr,
    bn: cutlass.Constexpr,
    weights: cutlass.Constexpr,
):
    # Every producer lane participates. A BN128 operand is two adjacent
    # 64-row boxes; local_tile retains the pointer's absolute swizzle phase
    # and the parent ring's stage stride.
    for box in cutlass.range_constexpr(bn // 64):
        part = cute.local_tile(shared, (64, 128), (box, 0, None))
        dest, src = partition_tma(atom, source, part, 64, 128)
        wn, wk, we = ntile * (bn // 64) + box, tile_k, expert
        cute.copy(atom, src[None, wn, wk, we], dest[None, stage], tma_bar_ptr=barrier)


@cute.jit
def prefetch_rows(
    weight_ptr,
    expert,
    tile,
    tile_k,
    lane,
    rows: cutlass.Constexpr,
    total_k: cutlass.Constexpr,
    bn: cutlass.Constexpr,
    span: cutlass.Constexpr,
):
    # Canonical [E, N, K/2] weights: each lane warms one contiguous span-K-tile run (64 B per
    # K tile) of its rows in L2, so DRAM serves whole runs instead of 64-B row pieces.
    for r in cutlass.range_constexpr(bn // 32):
        n = tile * bn + r * 32 + lane
        address = (Int64(expert) * rows + n) * (total_k * 64) + Int64(tile_k) * 64
        cute.arch.inline_ptx(
            f"cp.async.bulk.prefetch.L2.global [{{$r0}}], {span * 64};",
            read_only_args=[weight_ptr.toint() + address],
        )


@cute.jit
def load_fragment(copy, shared, fragment, tid, stage):
    thread = copy.get_slice(tid)
    source = thread.partition_S(shared)[None, None, None, stage]
    dest = thread.retile(fragment)
    for k in cutlass.range_constexpr(cute.size(dest, mode=[2])):
        cute.copy(copy, source[None, None, k], dest[None, None, k])


@cute.jit
def load_scale(copy, shared, fragment, tid, stage):
    thread = copy.get_slice(tid)
    cute.copy(copy, thread.partition_S(shared[None, None, stage]), thread.retile(fragment))


@cute.jit
def load_q_scale(copy, source, fragment, tid, row, valid_rows, tile_k, rows: cutlass.Constexpr):
    # The same scale-copy lane map as shared scales, with one packed K128 word
    # per row. Replicated lanes read identical words; MMA still selects K32 bytes.
    thread = copy.get_slice(tid)
    coords = thread.partition_S(cute.make_identity_tensor((rows, 1)))
    dest = thread.retile(fragment)
    dest.fill(Int32(0x7F7F7F7F))
    for j in cutlass.range_constexpr(cute.size(dest)):
        rr, _ = coords[j]
        # Unlike TMA, scalar loads need an explicit tail predicate, including
        # the last expert whose padded tile may extend beyond scale capacity.
        if rr < valid_rows:
            dest[j] = source[row + rr, tile_k, 0]


@cute.jit
def position_fp4(fragment):
    # ldmatrix.unpack places a nibble in each byte; mixed MMA expects bits 5:2.
    words = cute.recast_tensor(fragment, Uint32)
    words.store(words.load() * Uint32(4))


class GroupedGemm:
    def __init__(
        self,
        tile,
        grid,
        smem_limit,
        *,
        pdl=True,
        debug=False,
        debug_mode=None,
        debug_stamps=False,
        combine="atomic",
        lora_rank=0,
        up_first=False,
        lora_rows=False,
    ):
        validate_debug(debug_mode, debug_stamps)
        # LoRA factors' rank (0: base only); the epilogues add the rank-r deltas.
        self.lora_rank = lora_rank
        # FC1 weight rows: up then gate (canonical MXFP4Experts) or gate then up.
        self.up_first = up_first
        # FC1 z13 in sorted-row order (else pair order); non-swapped tiles then seed the
        # gate/up deltas on tensor cores.
        self.lora_rows = lora_rows
        self.tile = tile.check_smem(smem_limit)
        self.grid = grid
        self.smem_limit = smem_limit
        self.pdl = pdl
        self.combine = combine
        self.debug = debug
        self.debug_mode = debug_mode
        self.debug_stamps = debug_stamps
        t = self.tile

        # Struct construction and resource validation are plain host Python.
        @cute.struct
        class Storage:
            full: cute.struct.MemRange[Int64, t.stages]
            empty: cute.struct.MemRange[Int64, t.stages]
            work_full: cute.struct.MemRange[Int64, 2]
            work_empty: cute.struct.MemRange[Int64, 2]
            work: cute.struct.MemRange[Int32, 12]
            data: cute.struct.Align[cute.struct.MemRange[Uint8, t.data_bytes], 128]
            if not t.register_q_scales:
                sq: cute.struct.Align[cute.struct.MemRange[Int32, sf_cosize(t.rows, t.stages)], 128]
            sw: cute.struct.Align[cute.struct.MemRange[Int32, t.bn * t.stages * t.weights], 128]
            if not t.alias_epilogue:
                epilogue: cute.struct.Align[cute.struct.MemRange[BFloat16, t.rows * t.bn], 128]

            if debug_stamps:
                waits: cute.struct.MemRange[Int64, 8]

        self.storage = Storage
        actual = Storage.size_in_bytes()
        # The eight wait counters round up to the struct's 128-byte alignment.
        if actual != t.smem_bytes + (128 if debug_stamps else 0):
            raise ValueError(f"W4A8 shared struct {actual} differs from model {t.smem_bytes}")
        if actual > self.smem_limit:
            raise ValueError("W4A8 shared struct exceeds device opt-in limit")

    @cute.jit
    def __call__(
        self,
        q,
        qs,
        weight,
        ws,
        offsets,
        scale_offsets,
        row_meta,
        out,
        out_scales,
        trace0,
        trace1,
        trace_h,
        stamps,
        lora_z,
        lora_b0,
        lora_b1,
        lora_scale,
        lora_rows,
        lora_a2,
        lora_parts,
        lora_z2,
        lora_count,
        debug_sink,
        watchdog_ptr,
        stream,
    ):
        t = self.tile
        # Construct architecture-sensitive operations inside cute.compile,
        # after Runtime has installed the explicit sm_120a/sm_121a target.
        layouts = dsl_layouts()(t)
        mma = layouts.mma()
        k = cute.size(q, mode=[1])
        r = cute.size(q, mode=[0])
        p = cute.size(qs, mode=[1])
        e = cute.size(weight, mode=[0])
        n = cute.size(weight, mode=[1])
        gq = cute.make_tensor(
            cute.recast_ptr(q.iterator, dtype=Float8E4M3FN),
            cute.make_layout((r, k, 1), stride=(k, 1, r * k)),
        )
        gw = cute.make_tensor(
            cute.recast_ptr(weight.iterator, dtype=Float4E2M1FN),
            cute.make_layout((n, k, e), stride=(k, 1, n * k)),
        )
        gqs = cute.make_tensor(
            qs.iterator, cute.make_layout((p, k // 128, 1), stride=(1, p, p * (k // 128)))
        )
        # Weight scales in FlashInfer's 128x4 interleave: per expert [N/128, K/128, 32, 4]
        # words, so one BN128 tile's K128 words are a single 512-B run.
        words = k // 128
        gws = cute.make_tensor(
            ws.iterator,
            cute.make_layout(
                ((32, 4, n // 128), words, e), stride=((4, 1, 128 * words), 128, n * words)
            ),
        )
        aq, tq = layouts.tma(gq, layouts.q, t.rows)
        aw, tw = layouts.tma(gw, layouts.data_layout(64, Int8), 64, weight=True)
        asq, tsq = layouts.tma(gqs, layouts.sq, t.rows, scale=True)
        asw, tsw = layouts.tma(gws, layouts.sw, t.bn, scale=True)

        self.kernel(
            mma,
            aq,
            aw,
            asq,
            asw,
            tq,
            tw,
            tsq,
            tsw,
            gqs,
            offsets,
            scale_offsets,
            row_meta,
            out,
            out_scales,
            trace0,
            trace1,
            trace_h,
            stamps,
            lora_z,
            lora_b0,
            lora_b1,
            lora_scale,
            lora_rows,
            lora_a2,
            lora_parts,
            lora_z2,
            lora_count,
            weight.iterator,
            debug_sink,
            watchdog_ptr,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(288, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mma,
        aq,
        aw,
        asq,
        asw,
        tq,
        tw,
        tsq,
        tsw,
        gqs,
        offsets,
        scale_offsets,
        row_meta,
        out,
        out_scales,
        trace0,
        trace1,
        trace_h,
        stamps,
        lora_z,
        lora_b0,
        lora_b1,
        lora_scale,
        lora_rows,
        lora_a2,
        lora_parts,
        lora_z2,
        lora_count,
        weight_ptr,
        debug_sink,
        watchdog_ptr,
    ):
        t = self.tile
        # Layouts carry SSA values: rebuild in this isolated region from the
        # static tile instead of capturing the host trace's layouts via self.
        layouts = dsl_layouts()(t)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        # PREP produces both scheduling metadata and Q0; FC1 produces Q1.
        # Waiting only in the load warp (as upstream does) is insufficient here.
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        allocator = SmemAllocator()
        storage = allocator.allocate(self.storage)
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        work_full = storage.work_full.data_ptr()
        work_empty = storage.work_empty.data_ptr()
        work = storage.work.get_tensor(cute.make_layout((2, 6), stride=(6, 1)))
        data = storage.data.data_ptr()
        # TMA swizzles absolute shared addresses. The arena is only 128-byte
        # aligned, so carry the swizzle on each pointer, not on relative offsets.
        sq_data = cute.make_tensor(
            cute.recast_ptr(data, swizzle_=layouts.q.inner, dtype=Float8E4M3FN), layouts.q.outer
        )
        sw_data = cute.make_tensor(
            cute.recast_ptr(data + t.rows * 128 * t.stages, swizzle_=layouts.w.inner, dtype=Int8),
            layouts.w.outer,
        )
        if cutlass.const_expr(t.register_q_scales):
            # Shape-only view for scale-fragment construction below. The actual
            # loads address gqs by routed scale row and K tile, never this view.
            sq_scale = cute.make_tensor(gqs.iterator, layouts.sq)
        else:
            sq_scale = storage.sq.get_tensor(layouts.sq)
        sw_scale = storage.sw.get_tensor(layouts.sw)
        if cutlass.const_expr(t.stage == 1):
            su_data = cute.make_tensor(
                cute.recast_ptr(
                    data + (t.rows + t.bn) * 128 * t.stages, swizzle_=layouts.w.inner, dtype=Int8
                ),
                layouts.w.outer,
            )
            su_scale = cute.make_tensor(storage.sw.data_ptr() + t.bn * t.stages, layouts.sw)
        if cutlass.const_expr(t.alias_epilogue):
            # The 128-row epilogue aliases the operand ring only after all
            # math warps consume it. This plain BF16 view needs no TMA swizzle.
            shared_h = cute.make_tensor(
                cute.recast_ptr(data, dtype=BFloat16),
                cute.make_layout((t.rows, t.bn), stride=(t.bn, 1)),
            )
        else:
            shared_h = storage.epilogue.get_tensor(
                cute.make_layout((t.rows, t.bn), stride=(t.bn, 1))
            )
        if tid // 32 == 0:
            with cute.arch.elect_one():
                for init_stage in cutlass.range_constexpr(self.tile.stages):
                    cute.arch.mbarrier_init(full + init_stage, 1)
                    cute.arch.mbarrier_init(empty + init_stage, 8)
                for record in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(work_full + record, 1)
                    cute.arch.mbarrier_init(work_empty + record, 1)
        cute.arch.mbarrier_init_fence()
        cta_barrier(
            watchdog_ptr, t.watchdog, cta, tid, 1, t.stage, -1, 0, 0, -1, -1, t.transaction_bytes
        )

        total_k = cute.size(tq, mode=[1]) // 128
        ntiles = cute.size(out, mode=[1]) // t.bn
        sequence = Int32(0)
        iteration = Int32(0)
        linear = Int32(cta)
        if cutlass.const_expr(self.debug_stamps):
            waited = Int64(0)
            first_start = Int64(0)
            last_end = Int64(0)
            waits = storage.waits.get_tensor(cute.make_layout(8))
        # Fragment/copy construction calls jitted Layouts helpers, which cannot be
        # carried into the dynamic warp-role branch: build them for every warp first.
        # Keep logical Q/W names through TMA; exchange physical A/B only for
        # fragments. FC1's second weight tensor always means native up at I.
        sa, sb, ssa, ssb = sq_data, sw_data, sq_scale, sw_scale
        if cutlass.const_expr(t.swapped):
            sa, sb, ssa, ssb = sw_data, sq_data, sw_scale, sq_scale
        thread = mma.get_slice(tid % 256)
        ra = mma.make_fragment_A(thread.partition_A(sa)[None, None, None, 0])
        rb = mma.make_fragment_B(thread.partition_B(sb)[None, None, None, 0])
        ca, cb = layouts.data_copy(mma, "a"), layouts.data_copy(mma, "b")
        csa, csb = layouts.scale_copy(mma, "a"), layouts.scale_copy(mma, "b")
        # Register scale fragments take their shape from the linear scale layout; an
        # interleaved weight-scale tile is only read through its own layout by the copy.
        linear_w = cute.make_tensor(sw_scale.iterator, layouts.scale_layout(t.bn))
        fsa, fsb = sq_scale, linear_w
        if cutlass.const_expr(t.swapped):
            fsa, fsb = linear_w, sq_scale
        rsa = layouts.scale_fragment(fsa[None, None, 0], thread, tid % 256, "a")
        rsb = layouts.scale_fragment(fsb[None, None, 0], thread, tid % 256, "b")
        va, vb = layouts.scale_bytes(rsa), layouts.scale_bytes(rsb)
        if cutlass.const_expr(t.stage == 1):
            if cutlass.const_expr(t.swapped):
                ru = mma.make_fragment_A(thread.partition_A(su_data)[None, None, None, 0])
                rsu = layouts.scale_fragment(linear_w[None, None, 0], thread, tid % 256, "a")
            else:
                ru = mma.make_fragment_B(thread.partition_B(su_data)[None, None, None, 0])
                rsu = layouts.scale_fragment(linear_w[None, None, 0], thread, tid % 256, "b")
            vu = layouts.scale_bytes(rsu)
        shape = mma.partition_shape_C(t.physical[:2])
        acc = cute.make_rmem_tensor(shape, Float32)
        if cutlass.const_expr(t.stage == 1):
            up = cute.make_rmem_tensor(shape, Float32)
        coords = thread.partition_C(cute.make_identity_tensor(t.physical[:2]))
        tensor_seed = self.lora_rank > 0 and self.lora_rows and not t.swapped
        if cutlass.const_expr(tensor_seed):
            lora_mma = layouts.lora_mma()
            lora_thread = lora_mma.get_slice(tid % 256)
            m16, n16 = t.physical[0], t.physical[1]
            a_coords = lora_thread.partition_A(cute.make_identity_tensor((m16, 16)))
            b_coords = lora_thread.partition_B(cute.make_identity_tensor((n16, 16)))
            a_hi = cute.make_rmem_tensor(lora_mma.partition_shape_A((m16, 16)), BFloat16)
            a_mid = cute.make_rmem_tensor(lora_mma.partition_shape_A((m16, 16)), BFloat16)
            a_lo = cute.make_rmem_tensor(lora_mma.partition_shape_A((m16, 16)), BFloat16)
            frag_b = cute.make_rmem_tensor(lora_mma.partition_shape_B((n16, 16)), BFloat16)
        if tid // 32 == 8:
            active = Int32(1)
            while active != 0:
                record = iteration & 1
                if iteration >= 2:
                    wait_stage(
                        work_empty + record,
                        ((iteration // 2) - 1) & 1,
                        watchdog_ptr,
                        t.watchdog,
                        cta,
                        tid,
                        t.stage,
                        record,
                        iteration,
                        -1,
                        -1,
                        0,
                        -1,
                        site=12,
                    )
                publish(
                    offsets,
                    scale_offsets,
                    work,
                    record,
                    linear,
                    tid % 32,
                    t.rows,
                    ntiles,
                    t.order,
                    cute.size(offsets, mode=[0]) - 1,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(work_full + record)
                active = work[record, 5]
                if active != 0:
                    if cutlass.const_expr(self.debug_stamps):
                        if iteration == 0:
                            first_start = timer()
                    expert, row, scale_row = work[record, 0], work[record, 1], work[record, 2]
                    ntile = work[record, 3]
                    gate_tile, up_tile = ntile, ntile + ntiles
                    if cutlass.const_expr(self.up_first):
                        gate_tile, up_tile = ntile + ntiles, ntile
                    qsrc = cute.domain_offset((row, 0, 0), tq)
                    dq, gq = partition_tma(aq, qsrc, sq_data, t.rows, 128)
                    if cutlass.const_expr(not t.register_q_scales):
                        ssrc = cute.domain_offset((scale_row, 0, 0), tsq)
                        dsq, gqsf = partition_tma(asq, ssrc, sq_scale, t.rows, 1)
                    dsw, gws = partition_tma(asw, tsw, sw_scale, t.bn, 1)
                    if cutlass.const_expr(t.stage == 1):
                        dsu, gus = partition_tma(asw, tsw, su_scale, t.bn, 1)
                    for tile_k in cutlass.range(total_k):
                        s = sequence % t.stages
                        if sequence >= t.stages:
                            if cutlass.const_expr(self.debug_stamps):
                                wait_start = timer()
                            wait_stage(
                                empty + s,
                                ((sequence // t.stages) - 1) & 1,
                                watchdog_ptr,
                                t.watchdog,
                                cta,
                                tid,
                                t.stage,
                                s,
                                iteration,
                                expert,
                                ntile,
                                t.transaction_bytes,
                                tile_k,
                                site=10,
                            )
                            if cutlass.const_expr(self.debug_stamps):
                                waited += timer() - wait_start
                        issue_load = Int32(1)
                        if cutlass.const_expr(self.debug_mode == "mma_only"):
                            issue_load = Int32(tile_k == 0)
                        if issue_load != 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    full + s, t.transaction_bytes
                                )
                            # CuTe elects a lane internally; all 32 producer lanes issue each copy.
                            cute.copy(aq, gq[None, 0, tile_k, 0], dq[None, s], tma_bar_ptr=full + s)
                            if cutlass.const_expr(not t.register_q_scales):
                                cute.copy(
                                    asq,
                                    gqsf[None, 0, tile_k, 0],
                                    dsq[None, s],
                                    tma_bar_ptr=full + s,
                                )
                            copy_weight(
                                aw,
                                tw,
                                sw_data,
                                full + s,
                                s,
                                expert,
                                gate_tile,
                                tile_k,
                                ntiles,
                                total_k,
                                t.bn,
                                t.weights,
                            )
                            cute.copy(
                                asw,
                                gws[None, gate_tile, tile_k, expert],
                                dsw[None, s],
                                tma_bar_ptr=full + s,
                            )
                            if cutlass.const_expr(t.stage == 1):
                                copy_weight(
                                    aw,
                                    tw,
                                    su_data,
                                    full + s,
                                    s,
                                    expert,
                                    up_tile,
                                    tile_k,
                                    ntiles,
                                    total_k,
                                    t.bn,
                                    t.weights,
                                )
                                cute.copy(
                                    asw,
                                    gus[None, up_tile, tile_k, expert],
                                    dsu[None, s],
                                    tma_bar_ptr=full + s,
                                )
                            # The large-T LoRA variant (tensor-core seeds, sorted-row z)
                            # runs best without (GB10 T=2048: 9.97 vs 10.10 ms at span 3).
                            span = t.prefetch
                            if cutlass.const_expr(self.lora_rank > 0 and self.lora_rows):
                                span = 0
                            if cutlass.const_expr(span > 0):
                                if (tile_k % span == 0) & (tile_k + 2 * span <= total_k):
                                    rows = ntiles * t.weights * t.bn
                                    prefetch_rows(
                                        weight_ptr,
                                        expert,
                                        gate_tile,
                                        tile_k + span,
                                        tid % 32,
                                        rows,
                                        total_k,
                                        t.bn,
                                        span,
                                    )
                                    if cutlass.const_expr(t.stage == 1):
                                        prefetch_rows(
                                            weight_ptr,
                                            expert,
                                            up_tile,
                                            tile_k + span,
                                            tid % 32,
                                            rows,
                                            total_k,
                                            t.bn,
                                            span,
                                        )
                        else:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(full + s, 0)
                        sequence += 1
                    if cutlass.const_expr(t.alias_epilogue and self.debug_mode != "load_only"):
                        # The final empty arrival is deferred until epilogue
                        # readers drain. Do not refill ANY operand stage before
                        # it arrives, even when K does not fill the entire ring.
                        last = sequence - 1
                        if cutlass.const_expr(self.debug_stamps):
                            wait_start = timer()
                        wait_stage(
                            empty + last % t.stages,
                            (last // t.stages) & 1,
                            watchdog_ptr,
                            t.watchdog,
                            cta,
                            tid,
                            t.stage,
                            last % t.stages,
                            iteration,
                            expert,
                            ntile,
                            t.transaction_bytes,
                            total_k - 1,
                            site=10,
                        )
                        if cutlass.const_expr(self.debug_stamps):
                            waited += timer() - wait_start
                iteration += 1
                linear += self.grid
            if cutlass.const_expr(self.debug_stamps):
                if tid == 256:
                    stamps[cta, 0] = first_start
                    stamps[cta, 2] = waited
        else:
            record = Int32(0)
            wait_stage(
                work_full + record,
                0,
                watchdog_ptr,
                t.watchdog,
                cta,
                tid,
                t.stage,
                record,
                iteration,
                -1,
                -1,
                0,
                -1,
                site=11,
            )
            while work[record, 5] != 0:
                expert, row, scale_row = work[record, 0], work[record, 1], work[record, 2]
                ntile, end = work[record, 3], work[record, 4]
                if cutlass.const_expr(self.debug_mode in ("no_epilogue", "fragments_only")):
                    checksum = Uint32(0)
                if cutlass.const_expr(self.lora_rank > 0):
                    channel = ntile * t.bn
                    if cutlass.const_expr(tensor_seed and t.stage == 2):
                        lora_mma_seed(
                            lora_mma,
                            acc,
                            a_coords,
                            b_coords,
                            a_hi,
                            a_mid,
                            a_lo,
                            frag_b,
                            row,
                            end,
                            channel,
                            (lora_z, lora_b0, lora_scale, lora_rows),
                            0,
                            expert,
                            self.lora_rank,
                        )
                    elif cutlass.const_expr(tensor_seed):
                        for slot in cutlass.range_constexpr(2):
                            lora_mma_seed(
                                lora_mma,
                                acc if slot == 0 else up,
                                a_coords,
                                b_coords,
                                a_hi,
                                a_mid,
                                a_lo,
                                frag_b,
                                row,
                                end,
                                channel,
                                (lora_z, lora_b0 if slot == 0 else lora_b1, lora_scale, lora_rows),
                                slot,
                                expert,
                                self.lora_rank,
                            )
                    elif cutlass.const_expr(t.stage == 1):
                        seed = (lora_z, lora_b0, lora_scale, lora_rows)
                        lora_seed(
                            acc,
                            coords,
                            row,
                            end,
                            channel,
                            t.swapped,
                            seed,
                            0,
                            expert,
                            self.lora_rank,
                            self.lora_rows,
                        )
                        seed = (lora_z, lora_b1, lora_scale, lora_rows)
                        lora_seed(
                            up,
                            coords,
                            row,
                            end,
                            channel,
                            t.swapped,
                            seed,
                            1,
                            expert,
                            self.lora_rank,
                            self.lora_rows,
                        )
                    else:
                        seed = (lora_z, lora_b0, lora_scale, lora_rows)
                        lora_seed(
                            acc,
                            coords,
                            row,
                            end,
                            channel,
                            False,
                            seed,
                            0,
                            expert,
                            self.lora_rank,
                            self.lora_rows,
                        )
                else:
                    acc.fill(0)
                    if cutlass.const_expr(t.stage == 1):
                        up.fill(0)
                phase = Int32(0)
                for tile_k in cutlass.range(total_k):
                    s = sequence % t.stages
                    phase = (sequence // t.stages) & 1
                    if cutlass.const_expr(self.debug_stamps):
                        wait_start = timer()
                    wait_stage(
                        full + s,
                        phase,
                        watchdog_ptr,
                        t.watchdog,
                        cta,
                        tid,
                        t.stage,
                        s,
                        iteration,
                        expert,
                        ntile,
                        t.transaction_bytes,
                        tile_k,
                    )
                    if cutlass.const_expr(self.debug_stamps):
                        waited += timer() - wait_start
                    if cutlass.const_expr(self.debug_mode != "load_only"):
                        load_fragment(ca, sa, ra, tid, s)
                        load_fragment(cb, sb, rb, tid, s)
                        if cutlass.const_expr(t.register_q_scales):
                            load_q_scale(csa, gqs, rsa, tid, scale_row, end - row, tile_k, t.rows)
                        else:
                            load_scale(csa, ssa, rsa, tid, s)
                        load_scale(csb, ssb, rsb, tid, s)
                        if cutlass.const_expr(t.swapped):
                            position_fp4(ra)
                        else:
                            position_fp4(rb)
                        if cutlass.const_expr(t.stage == 1):
                            if cutlass.const_expr(t.swapped):
                                load_fragment(ca, su_data, ru, tid, s)
                                load_scale(csa, su_scale, rsu, tid, s)
                            else:
                                load_fragment(cb, su_data, ru, tid, s)
                                load_scale(csb, su_scale, rsu, tid, s)
                            position_fp4(ru)
                        if cutlass.const_expr(self.debug_mode == "fragments_only"):
                            checksum = fold_fragment(ra, checksum)
                            checksum = fold_fragment(rb, checksum)
                            checksum = fold_fragment(rsa, checksum)
                            checksum = fold_fragment(rsb, checksum)
                            if cutlass.const_expr(t.stage == 1):
                                checksum = fold_fragment(ru, checksum)
                                checksum = fold_fragment(rsu, checksum)
                    if cutlass.const_expr(self.debug_mode not in ("load_only", "fragments_only")):
                        for kk in cutlass.range_constexpr(4):
                            cute.gemm(
                                mma,
                                acc,
                                [ra[None, None, kk], va[None, None, kk, 0]],
                                [rb[None, None, kk], vb[None, None, kk, 0]],
                                acc,
                            )
                            if cutlass.const_expr(t.stage == 1):
                                if cutlass.const_expr(t.swapped):
                                    cute.gemm(
                                        mma,
                                        up,
                                        [ru[None, None, kk], vu[None, None, kk, 0]],
                                        [rb[None, None, kk], vb[None, None, kk, 0]],
                                        up,
                                    )
                                else:
                                    cute.gemm(
                                        mma,
                                        up,
                                        [ra[None, None, kk], va[None, None, kk, 0]],
                                        [ru[None, None, kk], vu[None, None, kk, 0]],
                                        up,
                                    )
                    # Release only after every lane in this warp has finished
                    # its operand reads; eight arrivals retire the stage.
                    if cutlass.const_expr(t.alias_epilogue and self.debug_mode != "load_only"):
                        if tile_k + 1 < total_k:
                            cute.arch.sync_warp()
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(empty + s)
                    else:
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(empty + s)
                    sequence += 1
                if cutlass.const_expr(t.alias_epilogue and self.debug_mode != "load_only"):
                    # A fast warp must not overwrite another warp's operands.
                    cta_barrier(
                        watchdog_ptr,
                        t.watchdog,
                        cta,
                        tid,
                        5,
                        t.stage,
                        -1,
                        phase,
                        iteration,
                        expert,
                        ntile,
                        t.transaction_bytes,
                        math_only=True,
                    )
                if cutlass.const_expr(self.debug_mode == "no_epilogue"):
                    checksum = fold_fragment(acc, checksum)
                    if cutlass.const_expr(t.stage == 1):
                        checksum = fold_fragment(up, checksum)
                if cutlass.const_expr(self.debug_mode in ("no_epilogue", "fragments_only")):
                    # One coalesced word per math lane/work item; output is invalid.
                    store_sink(debug_sink.iterator + cta * 256 + tid, checksum)
                if cutlass.const_expr(self.debug_mode not in ("load_only", "no_epilogue")):
                    if cutlass.const_expr(t.stage == 1):
                        stage_fc1(
                            acc,
                            up,
                            coords,
                            shared_h,
                            row_meta,
                            row,
                            end,
                            ntile * t.bn,
                            t.swapped,
                            trace0,
                            trace1,
                            trace_h,
                            self.debug,
                        )
                        cta_barrier(
                            watchdog_ptr,
                            t.watchdog,
                            cta,
                            tid,
                            6,
                            t.stage,
                            -1,
                            phase,
                            iteration,
                            expert,
                            ntile,
                            t.transaction_bytes,
                            math_only=True,
                        )
                        quantize(
                            shared_h, out, out_scales, row, scale_row, end, ntile, tid, t.rows, t.bn
                        )
                        if cutlass.const_expr(self.lora_rank > 0):
                            lora_partials(
                                shared_h,
                                (
                                    lora_z,
                                    lora_b0,
                                    lora_b1,
                                    lora_scale,
                                    lora_rows,
                                    lora_a2,
                                    lora_parts,
                                    lora_z2,
                                    lora_count,
                                ),
                                row,
                                end,
                                expert,
                                ntile * t.bn,
                                ntile,
                                tid,
                                t.rows,
                                t.bn,
                                self.lora_rank,
                                self.lora_rows,
                            )
                    else:
                        stage_fc2(
                            acc,
                            coords,
                            shared_h,
                            row,
                            end,
                            ntile * t.bn,
                            trace0,
                            self.debug,
                            out if cutlass.const_expr(self.combine == "rows") else None,
                        )
                        cta_barrier(
                            watchdog_ptr,
                            t.watchdog,
                            cta,
                            tid,
                            7,
                            t.stage,
                            -1,
                            phase,
                            iteration,
                            expert,
                            ntile,
                            t.transaction_bytes,
                            math_only=True,
                        )
                        if cutlass.const_expr(self.combine != "rows"):
                            scatter(shared_h, row_meta, out, row, end, ntile * t.bn, tid, t.rows)
                if cutlass.const_expr(t.alias_epilogue and self.debug_mode != "load_only"):
                    # Order generic epilogue writes before the next async TMA
                    # overwrite. Every math writer fences before releasing empty.
                    cute.arch.fence_proxy("async.shared", space="cta")
                # All math readers finish before reusing epilogue storage.
                cta_barrier(
                    watchdog_ptr,
                    t.watchdog,
                    cta,
                    tid,
                    8,
                    t.stage,
                    -1,
                    phase,
                    iteration,
                    expert,
                    ntile,
                    t.transaction_bytes,
                    math_only=True,
                )
                if cutlass.const_expr(t.alias_epilogue and self.debug_mode != "load_only"):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty + (sequence - 1) % t.stages)
                if cutlass.const_expr(self.debug_stamps):
                    last_end = timer()
                if tid == 0:
                    cute.arch.mbarrier_arrive(work_empty + record)
                iteration += 1
                record = iteration & 1
                wait_stage(
                    work_full + record,
                    (iteration // 2) & 1,
                    watchdog_ptr,
                    t.watchdog,
                    cta,
                    tid,
                    t.stage,
                    record,
                    iteration,
                    -1,
                    -1,
                    0,
                    -1,
                    site=11,
                )
            if cutlass.const_expr(self.debug_stamps):
                if tid % 32 == 0:
                    waits[tid // 32] = waited
                cta_barrier(
                    watchdog_ptr,
                    t.watchdog,
                    cta,
                    tid,
                    9,
                    t.stage,
                    -1,
                    0,
                    iteration,
                    -1,
                    -1,
                    0,
                    math_only=True,
                )
                if tid == 0:
                    total_wait = Int64(0)
                    for warp in cutlass.range_constexpr(8):
                        total_wait += waits[warp]
                    stamps[cta, 1] = last_end
                    # Sum of the eight warp leaders' accumulated full waits.
                    stamps[cta, 3] = total_wait
            stamp(
                watchdog_ptr,
                t.watchdog,
                cta,
                tid,
                9,
                t.stage,
                -1,
                0,
                iteration,
                -1,
                -1,
                t.transaction_bytes,
                -1,
            )
            # Only math signals completion, including for an empty CTA. The
            # producer may have published the terminal record before epilogue.
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_launch_dependents()
