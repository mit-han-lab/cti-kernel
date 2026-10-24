# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2025 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 QuACK team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 sm12x MoE and CUTLASS pipeline designs; rewritten.
# Unit-scale MXF8 follows our dense gated-FP8 design (studied in QuACK 0.6.5).
# See THIRD_PARTY_NOTICES. Only public PTX instructions are used.

"""Persistent grouped FC1+SiLU and FC2 kernels for SM120/121 (FC2 rows unweighted; the masked
combine applies the route weights).

A dedicated TMA warp streams weights; gather warps publish FP8 A through
mbarriers. FC2 consumers quantize h once into resident shared memory. Consumers
compute independent 128-K partials with unit-scale MXF8, then apply real scales.
Every tile has exactly one writer. The device reads the actual padded count;
allocation capacity never determines the persistent grid or iteration count.
"""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as hopper
import torch
from cutlass import BFloat16, Boolean, Float8E4M3FN, Float8E8M0FNU, Float32, Int8, Int32, Int64
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.memory import SmemAllocator
from cutlass.runtime import make_fake_stream
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils import blackwell_helpers, blockscaled_layout

from ._moe_block_fp8_reduce import masked_reduce

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()


def _unit_fragment(mma, tid, ptr, repetitions, is_a, shape):
    """Construct only the scale layout; fill registers with UE8M0 bits for 1."""
    layout_builder = blockscaled_layout.sm120_make_smem_layout_sfb
    partition = blackwell_helpers.partition_fragment_SFB
    if is_a:
        layout_builder = blockscaled_layout.sm120_make_smem_layout_sfa
        partition = blackwell_helpers.partition_fragment_SFA
    padded = (max(128, shape[0]), max(128, shape[1]), 128)
    layout = layout_builder(mma, padded, 32, 1)
    dummy = cute.make_tensor(cute.recast_ptr(ptr, dtype=Float8E8M0FNU), layout)
    registers = partition(dummy[None, None, 0], mma.get_slice(tid), tid)
    split = next(j for j in range(2, cute.rank(registers)) if cute.size(registers.shape[j:]) == 4)
    registers = cute.group_modes(registers, split, cute.rank(registers))
    registers = cute.group_modes(registers, 1, split)
    registers = cute.composition(registers, (None, cute.make_layout(repetitions), None))
    units = registers[None, None, 0]
    cute.recast_tensor(units, Int8).fill(127)
    return units


def _make_b_copy(mma, mm, warps, sliced):
    if sliced:
        # ldmatrix.x4 transfers two 8-column MMA N repetitions per warp.
        # Keep only that pair in the copy's value mode, exposing the other
        # pairs as a repetition mode that the sliced consumer can index.
        mma = cute.make_tiled_mma(
            warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU),
            cute.make_layout((4, warps // 4, 1)),
            permutation_mnk=(mm, 16 * (warps // 4), 32),
        )
    return cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), Float8E4M3FN), mma
    )


@cute.jit
def _divide_rn(a: Float32, b: Float32):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


@cute.jit
def _copy_arrive(barrier):
    # One deferred arrival per gather thread, covering its prior cp.async copies.
    # The ordinary arrival separately publishes scale/padding stores.
    cute.arch.inline_ptx(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [{$r0}];",
        read_only_args=[Int32(barrier.toint())],
    )


class GroupedKernel:
    def __init__(self, config, sms, *, fc1, top_k):
        self.align_bm = config.bm
        self.bm = config.bm if fc1 else min(config.bm, 64)
        self.width = config.bni if fc1 else config.bn
        self.halves = 2 if fc1 else 1
        self.stages, self.warps = config.stages, config.warps
        self.gather_warps = config.gather_warps if fc1 else 0
        self.swap, self.fc1, self.top_k = config.swap_ab, fc1, top_k
        self.resident = self.swap or not fc1
        self.slice_partial = fc1 and not self.swap and self.bm == 128 and self.width == 128
        self.grid = sms * config.ctas_per_sm
        self.mm, self.mn = (self.width, self.bm) if self.swap else (self.bm, self.width)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sa: cute.Tensor,
        w: cute.Tensor,
        sw: cute.Tensor,
        sorted_ids: cute.Tensor,
        experts: cute.Tensor,
        padded: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kblocks = a.shape[1] // 128
        self.aplanes = self.kblocks if self.resident else self.stages
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, Float8E4M3FN, 128), Float8E4M3FN
        )
        la = cute.tile_to_shape(atom, (self.bm, 128, self.aplanes), order=(0, 1, 2))
        lw = cute.tile_to_shape(atom, (self.width, 128, self.stages), order=(0, 1, 2))
        tma, gw = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w,
            cute.slice_(lw, (None, None, 0)),
            (self.width, 128),
        )
        wm = min(self.warps, 4, self.mm // 16)
        mma = cute.make_tiled_mma(
            warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU),
            cute.make_layout((wm, self.warps // wm, 1)),
            permutation_mnk=(self.mm, self.mn, 32),
        )

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[Int64, 2 * self.stages + 2]
            ascales: cute.struct.MemRange[Float32, self.bm * self.aplanes]
            wscales: cute.struct.MemRange[Float32, self.halves * self.stages]
            activation: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(la)], 128]
            weight: cute.struct.Align[
                cute.struct.MemRange[Float8E4M3FN, cute.cosize(lw) * self.halves], 128
            ]

        self.storage = Storage
        assert Storage.size_in_bytes() <= 99 * 1024, (
            "grouped MoE shape/preset exceeds 99 KiB shared memory"
        )
        # Capacity bounds launch size only. The persistent loop uses the device count.
        upper_blocks = cutlass.min(
            cute.ceil_div(sorted_ids.shape[0], self.bm),
            routing.shape[0] * (self.align_bm // self.bm),
        )
        upper = upper_blocks * (out.shape[1] // self.width)
        self.run(a, sa, tma, gw, sw, sorted_ids, experts, padded, routing, out, la, lw, mma).launch(
            grid=(cutlass.min(self.grid, upper), 1, 1),
            block=(32 * (self.warps + 1 + self.gather_warps), 1, 1),
            smem=Storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def gather(self, a, sa, ids, routing, shared, scales, block, kb, plane, gather_tid):
        # Flat V=16 views are essential: (1,16) is interpreted as 16 vector copies.
        for chunk in cutlass.range(cute.ceil_div(self.bm * 8, self.gather_warps * 32), unroll=1):
            index = gather_tid + chunk * self.gather_warps * 32
            if index < self.bm * 8:
                row, vector = index // 8, index % 8
                route = ids[block * self.bm + row]
                valid = (route >= 0) & (route < cute.size(routing))
                target = cute.local_tile(shared[None, None, plane], (1, 16), (row, vector))
                source_row = Int32(0)
                if valid:
                    source_row = route // self.top_k
                source = cute.local_tile(a, (1, 16), (source_row, kb * 8 + vector))
                atom = cute.make_copy_atom(cpasync.CopyG2SOp(), Float8E4M3FN, num_bits_per_copy=128)
                predicate = cute.make_rmem_tensor((16,), Boolean)
                predicate.fill(valid)
                # A false predicate emits cp.async with src-size=0 (zero fill),
                # keeping all writes to A in the same asynchronous proxy.
                cute.copy(atom, cute.coalesce(source), cute.coalesce(target), pred=predicate)
                if vector == 0:
                    scale = Float32(0)
                    if valid:
                        scale = sa[route // self.top_k, kb]
                    scales[row, plane] = scale

    @cute.jit
    def quantize_h(self, a, ids, routing, shared, scales, block, tid):
        # Sixteen lanes per 128-value group, eight BF16 values (16 bytes) per lane.
        # All consumers participate; no producer-side scalar load loop.
        for chunk in cutlass.range(
            cute.ceil_div(self.bm * self.kblocks * 16, self.warps * 32), unroll=1
        ):
            index = tid + chunk * self.warps * 32
            if index < self.bm * self.kblocks * 16:
                row = index // (self.kblocks * 16)
                kb, vector = index // 16 % self.kblocks, index % 16
                sorted_row = block * self.bm + row
                route = ids[sorted_row]
                values = cute.make_rmem_tensor((8,), BFloat16)
                values.fill(0)
                if route >= 0 and route < cute.size(routing):
                    source = cute.local_tile(a, (1, 8), (sorted_row, kb * 16 + vector))
                    cute.autovec_copy(cute.coalesce(source), values)
                peak, bad = Float32(1.0e-10), Int32(0)
                for j in cutlass.range_constexpr(8):
                    value = Float32(values[j])
                    peak = cutlass.max(peak, cute.math.abs(value))
                    bad |= Int32(not cute.math.isfinite(value))
                for delta in cutlass.range_constexpr(4):
                    peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, offset=1 << delta))
                    bad |= cute.arch.shuffle_sync_bfly(bad, offset=1 << delta)
                if bad != 0:
                    peak = Float32(float("nan"))
                if vector == 0:
                    scales[row, kb] = peak * Float32(1.0 / 448.0)
                factor = _divide_rn(Float32(448.0), peak)
                quantized = cute.make_rmem_tensor((8,), Float8E4M3FN)
                for j in cutlass.range_constexpr(8):
                    value = Float32(values[j]) * factor
                    value = cutlass.min(Float32(448), cutlass.max(Float32(-448), value))
                    if bad != 0:
                        value = Float32(float("nan"))
                    quantized[j] = value.to(Float8E4M3FN)
                target = cute.local_tile(shared[None, None, kb], (1, 8), (row, vector))
                cute.autovec_copy(quantized, cute.coalesce(target))

    @cute.jit
    def load_slice(
        self, copy_a, copy_b, pa, pb, p2, va, vb, kk, slot, ap, stage, half: cutlass.Constexpr
    ):
        if cutlass.const_expr(self.swap):
            if cutlass.const_expr(half == 0):
                cute.copy(copy_a, pa[None, None, kk, stage], va[None, None, slot])
            else:
                cute.copy(copy_a, p2[None, None, kk, stage], va[None, None, slot])
            cute.copy(copy_b, pb[None, None, kk, ap], vb[None, None, slot])
        else:
            cute.copy(copy_a, pa[None, None, kk, ap], va[None, None, slot])
            if cutlass.const_expr(half == 0):
                cute.copy(copy_b, pb[None, None, kk, stage], vb[None, None, slot])
            else:
                cute.copy(copy_b, p2[None, None, kk, stage], vb[None, None, slot])

    @cute.kernel
    def run(
        self,
        a: cute.Tensor,
        sa: cute.Tensor,
        tma: cute.CopyAtom,
        gw: cute.Tensor,
        sw: cute.Tensor,
        ids: cute.Tensor,
        experts: cute.Tensor,
        padded: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        la: cute.ComposedLayout,
        lw: cute.ComposedLayout,
        mma: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        storage = SmemAllocator().allocate(self.storage)
        full = storage.barriers.data_ptr()
        empty = full + self.stages
        afull, aempty = empty + self.stages, empty + self.stages + 1
        ascales = storage.ascales.get_tensor(
            cute.make_layout((self.bm, self.aplanes), stride=(1, self.bm))
        )
        wscales = storage.wscales.get_tensor(
            cute.make_layout((self.halves, self.stages), stride=(1, self.halves))
        )
        # TMA swizzles addresses, not buffer-relative offsets (base is only 128B aligned).
        activation = storage.activation.get_tensor(la.outer, swizzle=la.inner)
        weight = storage.weight.get_tensor(lw.outer, swizzle=lw.inner)
        second_weight = cute.make_tensor(
            cute.recast_ptr(
                storage.weight.data_ptr() + (self.halves - 1) * cute.cosize(lw), lw.inner
            ),
            lw.outer,
        )
        if wid == self.warps:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(self.stages):
                    # TMA warp publishes scales; streamed gather contributes both
                    # ordinary stores and deferred async arrivals, all 32 lanes/warp.
                    arrivals = 32 + (0 if self.resident else 64 * self.gather_warps)
                    cute.arch.mbarrier_init(full + s, arrivals)
                    cute.arch.mbarrier_init(empty + s, self.warps * 32)
                cute.arch.mbarrier_init(afull, max(1, 64 * self.gather_warps))
                cute.arch.mbarrier_init(aempty, self.warps * 32)
            cpasync.prefetch_descriptor(tma)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()  # Initialization only; never a per-tile CTA join.
        stage, phase, aphase = Int32(0), Int32(0), Int32(0)
        blocks = padded[0] // self.bm
        ntiles = out.shape[1] // self.width
        for tile in cutlass.range(bid, blocks * ntiles, grid, unroll=1):
            block, nt = tile % blocks, tile // blocks
            expert = experts[block // (self.align_bm // self.bm)]
            if expert >= 0 and expert < sw.shape[0]:
                if wid == self.warps:
                    weight_start = Int64(expert) * sw.shape[1] * 128 + nt * self.width
                    tile_w = cute.local_tile(
                        gw, (self.width, 128), (weight_start // self.width, None)
                    )
                    ws, wg = cpasync.tma_partition(
                        tma,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(weight, 0, 2),
                        cute.group_modes(tile_w, 0, 2),
                    )
                    if cutlass.const_expr(self.fc1):
                        tile_u = cute.local_tile(
                            gw,
                            (self.width, 128),
                            ((weight_start + out.shape[1]) // self.width, None),
                        )
                        us, ug = cpasync.tma_partition(
                            tma,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(second_weight, 0, 2),
                            cute.group_modes(tile_u, 0, 2),
                        )
                    for kb in cutlass.range(self.kblocks, unroll=1):
                        cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                        if lane < self.halves:
                            wscales[lane, stage] = sw[
                                expert, (nt * self.width + lane * out.shape[1]) // 128, kb
                            ]
                        if lane == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                full + stage, self.halves * self.width * 128
                            )
                        else:
                            cute.arch.mbarrier_arrive(full + stage)
                        cute.arch.sync_warp()
                        cute.copy(tma, wg[None, kb], ws[None, stage], tma_bar_ptr=full + stage)
                        if cutlass.const_expr(self.fc1):
                            cute.copy(tma, ug[None, kb], us[None, stage], tma_bar_ptr=full + stage)
                        stage += 1
                        if stage == self.stages:
                            stage = Int32(0)
                            phase ^= 1
                elif wid > self.warps:
                    if cutlass.const_expr(self.fc1):
                        gather_tid = tid - 32 * (self.warps + 1)
                        if cutlass.const_expr(self.resident):
                            cute.arch.mbarrier_wait(aempty, aphase ^ 1)
                            for kb in cutlass.range(self.kblocks, unroll=1):
                                self.gather(
                                    a,
                                    sa,
                                    ids,
                                    routing,
                                    activation,
                                    ascales,
                                    block,
                                    kb,
                                    kb,
                                    gather_tid,
                                )
                            _copy_arrive(afull)
                            cute.arch.mbarrier_arrive(afull)
                            aphase ^= 1
                        else:
                            for kb in cutlass.range(self.kblocks, unroll=1):
                                cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                                self.gather(
                                    a,
                                    sa,
                                    ids,
                                    routing,
                                    activation,
                                    ascales,
                                    block,
                                    kb,
                                    stage,
                                    gather_tid,
                                )
                                _copy_arrive(full + stage)
                                cute.arch.mbarrier_arrive(full + stage)
                                stage += 1
                                if stage == self.stages:
                                    stage = Int32(0)
                                    phase ^= 1
                else:
                    if cutlass.const_expr(self.fc1 and self.resident):
                        cute.arch.mbarrier_wait(afull, aphase)
                    if cutlass.const_expr(not self.fc1):
                        # Consumer-only joins: the independent TMA warp can run
                        # ahead through the weight ring during quantization/epilogue.
                        cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                        self.quantize_h(a, ids, routing, activation, ascales, block, tid)
                        cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                    left, right = activation, weight
                    if cutlass.const_expr(self.swap):
                        left, right = weight, activation
                    thread = mma.get_slice(tid)
                    ra = mma.make_fragment_A(thread.partition_A(left)[None, None, None, 0])
                    rb = mma.make_fragment_B(thread.partition_B(right)[None, None, None, 0])
                    ua = _unit_fragment(
                        mma, tid, left.iterator, cute.size(ra, mode=[1]), True, (self.mm, self.mn)
                    )
                    ub = _unit_fragment(
                        mma, tid, right.iterator, cute.size(rb, mode=[1]), False, (self.mm, self.mn)
                    )
                    copy_a = cute.make_tiled_copy_A(
                        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), Float8E4M3FN), mma
                    )
                    copy_b = _make_b_copy(mma, self.mm, self.warps, self.slice_partial)
                    ca, cb = copy_a.get_slice(tid), copy_b.get_slice(tid)
                    pa, pb = ca.partition_S(left), cb.partition_S(right)
                    if cutlass.const_expr(self.swap):
                        p2 = ca.partition_S(second_weight)
                    else:
                        p2 = cb.partition_S(second_weight)
                    va, vb = ca.retile(ra), cb.retile(rb)
                    shape = mma.partition_shape_C((self.mm, self.mn))
                    acc, up = (
                        cute.make_rmem_tensor(shape, Float32),
                        cute.make_rmem_tensor(shape, Float32),
                    )
                    partial_shape = (shape[0], shape[1], 2) if self.slice_partial else shape
                    partial = cute.make_rmem_tensor(partial_shape, Float32)
                    coords = thread.partition_C(cute.make_identity_tensor((self.mm, self.mn)))
                    acc.fill(0)
                    up.fill(0)
                    for kb in cutlass.range(self.kblocks, unroll=1):
                        cute.arch.mbarrier_wait(full + stage, phase)
                        ap = kb if self.resident else stage
                        for half in cutlass.range_constexpr(self.halves):
                            weight_scale = wscales[half, stage]
                            if cutlass.const_expr(self.swap):
                                factors = cute.make_rmem_tensor(
                                    (2 * cute.size(acc, mode=[2]),), Float32
                                )
                                for n in cutlass.range_constexpr(cute.size(acc, mode=[2])):
                                    for r in cutlass.range_constexpr(2):
                                        j = r + 4 * cute.size(acc, mode=[1]) * n
                                        factors[2 * n + r] = (
                                            ascales[coords[j][1], ap] * weight_scale
                                        )
                            else:
                                factors = cute.make_rmem_tensor(
                                    (2 * cute.size(acc, mode=[1]),), Float32
                                )
                                for m in cutlass.range_constexpr(cute.size(acc, mode=[1])):
                                    for r in cutlass.range_constexpr(2):
                                        j = 2 * r + 4 * m
                                        factors[2 * m + r] = (
                                            ascales[coords[j][0], ap] * weight_scale
                                        )
                            if cutlass.const_expr(self.slice_partial):
                                # The widest fused tile has two persistent C
                                # fragments. Compute one ldmatrix N pair at a
                                # time, keeping the partial fragment small.
                                pairs = cute.size(acc, mode=[2]) // 2
                                assert cute.size(pb, mode=[1]) == pairs
                                for n in cutlass.range_constexpr(pairs):
                                    small_b = cute.composition(
                                        cute.domain_offset((0, 2 * n, 0), rb),
                                        (None, cute.make_layout(2), None),
                                    )
                                    small_unit = cute.composition(
                                        cute.domain_offset((0, 2 * n), ub),
                                        (None, cute.make_layout(2)),
                                    )
                                    small_v = cb.retile(small_b)
                                    assert cute.size(small_v, mode=[1]) == 1
                                    assert cute.size(small_v) == cute.size(small_b)
                                    partial.fill(0)
                                    cute.copy(copy_a, pa[None, None, 0, ap], va[None, None, 0])
                                    if cutlass.const_expr(half == 0):
                                        cute.copy(
                                            copy_b, pb[None, n, 0, stage], small_v[None, 0, 0]
                                        )
                                    else:
                                        cute.copy(
                                            copy_b, p2[None, n, 0, stage], small_v[None, 0, 0]
                                        )
                                    for kk in cutlass.range_constexpr(4):
                                        if cutlass.const_expr(kk < 3):
                                            slot = (kk + 1) % 2
                                            cute.copy(
                                                copy_a,
                                                pa[None, None, kk + 1, ap],
                                                va[None, None, slot],
                                            )
                                            if cutlass.const_expr(half == 0):
                                                cute.copy(
                                                    copy_b,
                                                    pb[None, n, kk + 1, stage],
                                                    small_v[None, 0, slot],
                                                )
                                            else:
                                                cute.copy(
                                                    copy_b,
                                                    p2[None, n, kk + 1, stage],
                                                    small_v[None, 0, slot],
                                                )
                                        if cutlass.const_expr(
                                            kk == 2 and half == self.halves - 1 and n == pairs - 1
                                        ):
                                            cute.arch.mbarrier_arrive(empty + stage)
                                        cute.gemm(
                                            mma,
                                            partial,
                                            [ra[None, None, kk % 2], ua],
                                            [small_b[None, None, kk % 2], small_unit],
                                            partial,
                                        )
                                    for j in cutlass.range_constexpr(cute.size(partial)):
                                        dest = j + 8 * cute.size(acc, mode=[1]) * n
                                        f = 2 * (j // 4 % cute.size(acc, mode=[1])) + j // 2 % 2
                                        if cutlass.const_expr(half == 0):
                                            acc[dest] += partial[j] * factors[f]
                                        else:
                                            up[dest] += partial[j] * factors[f]
                            else:
                                partial.fill(0)
                                self.load_slice(
                                    copy_a, copy_b, pa, pb, p2, va, vb, 0, 0, ap, stage, half
                                )
                                for kk in cutlass.range_constexpr(4):
                                    if cutlass.const_expr(kk < 3):
                                        self.load_slice(
                                            copy_a,
                                            copy_b,
                                            pa,
                                            pb,
                                            p2,
                                            va,
                                            vb,
                                            kk + 1,
                                            (kk + 1) % 2,
                                            ap,
                                            stage,
                                            half,
                                        )
                                    if cutlass.const_expr(kk == 2 and half == self.halves - 1):
                                        cute.arch.mbarrier_arrive(empty + stage)
                                        if cutlass.const_expr(self.fc1 and self.resident):
                                            if kb == self.kblocks - 1:
                                                cute.arch.mbarrier_arrive(aempty)
                                    cute.gemm(
                                        mma,
                                        partial,
                                        [ra[None, None, kk % 2], ua],
                                        [rb[None, None, kk % 2], ub],
                                        partial,
                                    )
                                for j in cutlass.range_constexpr(cute.size(acc)):
                                    if cutlass.const_expr(self.swap):
                                        f = 2 * (j // (4 * cute.size(acc, mode=[1]))) + j % 2
                                    else:
                                        f = 2 * (j // 4 % cute.size(acc, mode=[1])) + j // 2 % 2
                                    if cutlass.const_expr(half == 0):
                                        acc[j] += partial[j] * factors[f]
                                    else:
                                        up[j] += partial[j] * factors[f]
                        stage += 1
                        if stage == self.stages:
                            stage = Int32(0)
                            phase ^= 1
                    if cutlass.const_expr(self.fc1 and self.resident):
                        aphase ^= 1
                    # Compute the full fragment before vector stores. Swap-AB
                    # pairs neighboring MMA rows with lane xor 4 (same token).
                    result = acc
                    if cutlass.const_expr(self.fc1):
                        for j in cutlass.range_constexpr(cute.size(acc)):
                            result[j] = (
                                _divide_rn(acc[j], Float32(1) + cute.math.exp(-acc[j])) * up[j]
                            )
                    stores = cute.size(acc) if self.swap else cute.size(acc) // 2
                    # Load every store's route before the first store, so the id loads overlap
                    # instead of each store waiting on its own.
                    routes = cute.make_rmem_tensor((stores,), Int32)
                    for j in cutlass.range_constexpr(stores):
                        index = j if self.swap else 2 * j
                        row = coords[index][1] if self.swap else coords[index][0]
                        routes[j] = ids[block * self.bm + row]
                    for j in cutlass.range_constexpr(stores):
                        index = j if self.swap else 2 * j
                        row = coords[index][1] if self.swap else coords[index][0]
                        col = coords[index][0] if self.swap else coords[index][1]
                        pair = cute.make_rmem_tensor((2,), BFloat16)
                        pair[0] = result[index].to(BFloat16)
                        if cutlass.const_expr(self.swap):
                            pair[1] = cute.arch.shuffle_sync_bfly(result[index], offset=4).to(
                                BFloat16
                            )
                        else:
                            pair[1] = result[index + 1].to(BFloat16)
                        sorted_row = block * self.bm + row
                        route = routes[j]
                        if route >= 0 and route < cute.size(routing) and col % 2 == 0:
                            target_row = sorted_row if self.fc1 else route
                            target = cute.local_tile(
                                out, (1, 2), (target_row, (nt * self.width + col) // 2)
                            )
                            cute.autovec_copy(pair, cute.coalesce(target))


def compile_kernel(config, sms, e, h, i, top_k, *, fc1, arch="sm_120a"):
    """Compile with symbolic row extents, without allocating tensors or launching."""
    fake = cute.runtime.make_fake_tensor
    rows, capacity, routes = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    k, n = (h, 2 * i) if fc1 else (i, h)
    dtype = Float8E4M3FN if fc1 else BFloat16
    a = fake(dtype, (rows, k), (k, 1), assumed_align=16)
    sa = fake(Float32, (cute.sym_int64(), k // 128), (k // 128, 1), assumed_align=4)
    w = fake(Float8E4M3FN, (e * n, k), (k, 1), assumed_align=16)
    sw = fake(
        Float32, (e, n // 128, k // 128), ((n // 128) * (k // 128), k // 128, 1), assumed_align=4
    )
    ids = fake(Int32, (capacity,), (1,), assumed_align=4)
    experts = fake(Int32, (cute.sym_int64(),), (1,), assumed_align=4)
    padded = fake(Int32, (1,), (1,), assumed_align=4)
    routing = fake(Float32, (routes,), (1,), assumed_align=4)
    out_width = i if fc1 else h
    out = fake(BFloat16, (cute.sym_int64(), out_width), (out_width, 1), assumed_align=16)
    return cute.compile(
        GroupedKernel(config, sms, fc1=fc1, top_k=top_k),
        a,
        sa,
        w,
        sw,
        ids,
        experts,
        padded,
        routing,
        out,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


def _functions(op, config):
    functions = []
    for fc1 in (True, False):
        key = (
            config,
            fc1,
            torch.bfloat16,
            op.device,
            op.device_class,
            op.e,
            op.h,
            op.i,
            op.top_k,
        )
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: cold kernels; call prepare outside capture")
            with _COMPILE_LOCK:
                if key not in _COMPILED:
                    major, minor, sms = op.device_class
                    _COMPILED[key] = compile_kernel(
                        config,
                        sms,
                        op.e,
                        op.h,
                        op.i,
                        op.top_k,
                        fc1=fc1,
                        arch=f"sm_{major}{minor}a",
                    )
        functions.append(_COMPILED[key])
    return tuple(functions)


def forward(op, x, weights, ids, config, *, return_intermediates=False):
    from .linear.block_fp8 import quantize_per_token_group_fp8
    from .moe_impl import moe_align_block_size

    fc1, fc2 = _functions(op, config)
    sorted_ids, experts, padded = moe_align_block_size(ids, config.bm, op.e)
    h = torch.empty((sorted_ids.numel(), op.i), dtype=x.dtype, device=x.device)
    down = torch.empty((ids.numel(), op.h), dtype=x.dtype, device=x.device)
    a, sa = quantize_per_token_group_fp8(x)
    routing = weights.view(-1)
    fc1(a, sa, op.w13.view(-1, op.h), op.s13, sorted_ids, experts, padded, routing, h)
    fc2(
        h,
        op.s2.view(-1, op.i // 128),
        op.w2.view(-1, op.i),
        op.s2,
        sorted_ids,
        experts,
        padded,
        routing,
        down,
    )
    out = torch.empty_like(x)
    masked_reduce(down, ids, out, weights)
    if return_intermediates:
        return out, h, down, sorted_ids, experts, padded
    return out
