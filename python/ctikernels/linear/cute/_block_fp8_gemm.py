# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright (c) 2025 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 QuACK team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: FlashInfer 0.7.1 blackwell_sm12x/kernel_moe_fp8_gemm.py, CUTLASS
# blackwell_geforce/dense_gemm.py and QuACK 0.6.5 gemm_sm120.py (unit-scale MXF8);
# hybrid Stream-K tail: CUTLASS examples/47_ampere_gemm_universal_streamk.

"""SM120/121 dense block-FP8 GEMM. Imported only by the qualified host provider.

A CTA owns a sequence of output tiles (or contiguous tail K ranges).
A producer warp stages TMA operands and FP32 scales;
4/8 consumer warps form a fresh FP32 partial for each 128-wide K block. The full
barrier counts every producer lane, making ordinary shared scale stores visible
alongside the TMA completion. The empty barrier counts every consumer thread.
Each CTA publishes its helper fragment as FIRST SK work, before or after DP,
and its owner fragment LAST. Owners reduce FP32 registers in fixed CTA order, then reset
the counter. No owner depends on work following another owner wait.
The default epilogue reuses operands as FP32 storage. The private epilogue uses
output-dtype storage and consumer-only barriers so the producer can run ahead.
The MXF8 atom uses constant UE8M0 unit scales, created once per CTA.
Diagnostic knockouts are accessible only through the development runner.
BM/BN describe the *internal* MMA orientation (swapped when requested).
"""

from math import ceil

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float8E4M3FN, Float8E8M0FNU, Float32, Int8, Int32, Int64
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils import LayoutEnum, SmemAllocator, blackwell_helpers, blockscaled_layout


def diagnostic_store(pointer, value):
    """Public PTX volatile shared store keeps diagnostic dependencies live."""
    cute.arch.inline_ptx(
        "st.volatile.shared.b32 [{$r0}], {$r1};",
        read_only_args=[Int32(pointer.toint()), value],
    )


@cute.jit
def load_progress(pointer):
    """Device-scope acquire observes the contributors' release publications."""
    return cute.arch.inline_ptx(
        "ld.acquire.gpu.global.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[Int64(pointer.toint())],
    )


@cute.jit
def reset_arrivals(pointer):
    cute.arch.inline_ptx(
        "st.release.gpu.global.b32 [{$r0}], {$r1};",
        read_only_args=[Int64(pointer.toint()), Int32(0)],
    )


@cute.jit
def arrive(pointer):
    # A release RMW extends the counter's release sequence for the owner acquire.
    cute.arch.inline_ptx(
        "{ .reg .u32 old; atom.release.gpu.global.add.u32 old, [{$r0}], 1; }",
        read_only_args=[Int64(pointer.toint())],
    )


@cute.jit
def load_tail_chunk(pointer, acc, chunk):
    """Only four temporary registers, born after the mainloop's last promotion."""
    # Bypass non-coherent per-SM L1 when reading another CTA's publication.
    words = cute.arch.inline_ptx(
        "ld.global.cg.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[Int64(pointer.toint())],
    )
    for e in cutlass.range_constexpr(4):
        acc[4 * chunk + e] = acc[4 * chunk + e] + words[e]


@cute.jit
def store_tail_chunk(pointer, acc, chunk):
    # Flatten the MMA C fragment in its register order, NOT output row order.
    # Adjacent consumer threads write adjacent aligned 16-byte vectors.
    cute.arch.inline_ptx(
        "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[
            Int64(pointer.toint()),
            acc[4 * chunk],
            acc[4 * chunk + 1],
            acc[4 * chunk + 2],
            acc[4 * chunk + 3],
        ],
    )


@cute.jit
def store_output(d, row, col, converted):
    target = cute.local_tile(d, (1, 4), (row, col // 4))
    cute.autovec_copy(converted, target)


def unit_scale_fragment(mma, tid, pointer, mn_atoms, operand, tile_shape=(128, 128, 128)):
    """Trace-time layout construction only; the borrowed pointer is never read.

    Scale layout helpers require 128-row/column granularity. Partition a padded
    descriptor (at least 128x128x128), then retain just this operand's MN atoms and one K
    slice. Every K=32 instruction can reuse that same unit-scale register slice.
    """
    layout_fn = (
        blockscaled_layout.sm120_make_smem_layout_sfa
        if operand == "a"
        else blockscaled_layout.sm120_make_smem_layout_sfb
    )
    partition_fn = (
        blackwell_helpers.partition_fragment_SFA
        if operand == "a"
        else blackwell_helpers.partition_fragment_SFB
    )
    layout = layout_fn(mma, tile_shape, 32, 1)
    descriptor = cute.make_tensor(cute.recast_ptr(pointer, dtype=Float8E8M0FNU), layout)
    fragment = partition_fn(descriptor[None, None, 0], mma.get_slice(tid), tid)
    # Padding can introduce extra MN modes. Locate the four K atoms from the
    # trailing shape, preserving at least one MN mode before grouping it.
    boundary = next(i for i in range(2, cute.rank(fragment)) if cute.size(fragment.shape[i:]) == 4)
    fragment = cute.group_modes(fragment, boundary, cute.rank(fragment))
    fragment = cute.group_modes(fragment, 1, boundary)
    fragment = cute.composition(fragment, (None, cute.make_layout(mn_atoms), None))
    result = fragment[None, None, 0]
    cute.recast_tensor(result, Int8).fill(127)  # UE8M0 bit encoding of 2**0, not numeric 127.
    return result


class Kernel:
    def __init__(self, config, sm_count, *, _dev_diagnostics=()):
        if not set(_dev_diagnostics) <= {"noscale", "noepi", "nomma"}:
            raise ValueError("unknown development knockout")
        self.noscale, self.noepi, self.nomma = (
            flag in _dev_diagnostics for flag in ("noscale", "noepi", "nomma")
        )
        self.scheduler = config.scheduler
        # All scheduler partitions use resident CTA slots, including tail fixup.
        self.sm_count = sm_count * config.ctas_per_sm
        self.tail_split_threshold_tiles = ceil(self.sm_count * config.tail_split_threshold)
        self.min_sk_iters = config.min_sk_iters
        self.sk_order, self.sk_waves = config.sk_order, config.sk_waves
        self.ctas_per_sm = config.ctas_per_sm
        self.bm, self.bn = config.bm, config.bn
        self.stages, self.warps = config.stages, config.warps
        self.producer_warps = config.producer_warps
        self.mx = config.mx
        self.pdl = config.pdl
        self.epi = config.epi
        self.smem_bytes = config.smem_bytes
        self.scale_rows = config.scale_rows
        self.epilogue_swizzle = config.epilogue_swizzle
        self.swap, self.raster, self.group = config.swap_ab, config.raster, config.group
        self.token_tile = self.bn if self.swap else self.bm
        self.weight_tile = self.bm if self.swap else self.bn

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w: cute.Tensor,
        sa: cute.Tensor,
        sw: cute.Tensor,
        d: cute.Tensor,
        workspace: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        # Both operands have contiguous K; a 128-byte swizzle also serves ldmatrix.
        layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, Float8E4M3FN, 128),
            Float8E4M3FN,
        )
        la = cute.tile_to_shape(layout_atom, (self.bm, 128, self.stages), order=(0, 1, 2))
        lb = cute.tile_to_shape(layout_atom, (self.bn, 128, self.stages), order=(0, 1, 2))
        left, right = a, w
        if cutlass.const_expr(self.swap):
            left, right = w, a
        ta, ga = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            left,
            cute.slice_(la, (None, None, 0)),
            (self.bm, 128),
        )
        tb, gb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            right,
            cute.slice_(lb, (None, None, 0)),
            (self.bn, 128),
        )
        warp_m = min(4, self.bm // 16)
        if cutlass.const_expr(self.mx):
            operation = warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU)
        else:
            operation = warp.MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32))
        mma = cute.make_tiled_mma(
            operation,
            cute.make_layout((warp_m, self.warps // warp_m, 1)),
            permutation_mnk=(self.bm, self.bn, 32),
        )

        epilogue_elements = self.bm * self.bn if self.epi == "private" else 0

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[Int64, 2 * self.stages]
            scales: cute.struct.MemRange[Float32, (self.scale_rows + 1) * self.stages]
            left: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(la)], 128]
            right: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(lb)], 128]
            epilogue: cute.struct.MemRange[d.element_type, epilogue_elements]
            diagnostic_sink: cute.struct.MemRange[Int32, 32 * self.warps if self.nomma else 0]

        self.storage = Storage
        assert Storage.size_in_bytes() == self.smem_bytes + (128 * self.warps if self.nomma else 0)
        assert Storage.size_in_bytes() <= 99 * 1024
        mt = cute.ceil_div(a.shape[0], self.token_tile)
        nt = cute.ceil_div(w.shape[0], self.weight_tile)
        self.run(ta, ga, tb, gb, sa, sw, d, workspace, la, lb, mma).launch(
            grid=(mt * nt if self.scheduler == "grid" else self.sm_count, 1, 1),
            block=(32 * (self.warps + self.producer_warps), 1, 1),
            max_number_threads=(32 * (self.warps + self.producer_warps), 1, 1),
            smem=Storage.size_in_bytes(),
            min_blocks_per_mp=self.ctas_per_sm,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def tile_coords(self, tile, mt, nt):
        if cutlass.const_expr(self.raster == "m"):
            tm, tn = tile % mt, tile // mt
        elif cutlass.const_expr(self.raster == "n"):
            tm, tn = tile // nt, tile % nt
        else:
            first_m = (tile // (self.group * nt)) * self.group
            height = cutlass.min(self.group, mt - first_m)
            offset = tile % (self.group * nt)
            tm, tn = first_m + offset % height, offset // height
        return tm, tn

    @cute.jit
    def activation_scale(self, scales, sa, row, token_start, m, kb, stage):
        value = Float32(0)
        if cutlass.const_expr(self.scale_rows == self.token_tile):
            value = scales[row, stage]
        else:
            if row < self.scale_rows:
                value = scales[row, stage]
            elif token_start + Int64(row) < m:
                value = sa[token_start + Int64(row), kb]
        return value

    @cute.jit
    def fixup(self, acc, workspace, tid, bid, tail_tile, first, helpers, k_end, k_blocks):
        """Consumers only; called after griddepcontrol.wait and the mainloop.

        A register chunk has four consecutive logical C-fragment values. The
        same MMA layout is used by helpers/owner, including swapped operands:
        offset = slot*BM*BN + (chunk*consumer_threads + tid)*4. No coordinate
        transpose, shared epilogue round-trip, or output-dtype rounding occurs.
        """
        if tail_tile >= 0 and (k_end < k_blocks or helpers > 0):
            flag = cute.recast_ptr(
                workspace.iterator + self.sm_count * self.bm * self.bn + tail_tile,
                dtype=Int32,
            )
            if k_end < k_blocks:
                for chunk in cutlass.range_constexpr(cute.size(acc) // 4):
                    offset = bid * self.bm * self.bn + (chunk * self.warps * 32 + tid) * 4
                    store_tail_chunk(workspace.iterator + offset, acc, chunk)
                # Every writer fences, then the consumer join hands publication
                # to lane 0. The producer does not participate in fixup barriers.
                cute.arch.inline_ptx("fence.acq_rel.gpu;")
                cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                if tid == 0:
                    arrive(flag)
            else:
                if tid == 0:
                    backoff = Int32(32)
                    while load_progress(flag) != helpers:
                        cute.arch.inline_ptx("nanosleep.u32 {$r0};", read_only_args=[backoff])
                        backoff = Int32(cutlass.min(backoff * 2, 256))
                # Propagate lane 0's acquire to every consumer reader.
                cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                # Keep the owner's sum in place, then add helpers in ascending
                # CTA id (= K order). Arrival order cannot change FP32 results.
                for slot in cutlass.range(first, first + helpers, unroll=1):
                    for chunk in cutlass.range_constexpr(cute.size(acc) // 4):
                        offset = slot * self.bm * self.bn + (chunk * self.warps * 32 + tid) * 4
                        load_tail_chunk(workspace.iterator + offset, acc, chunk)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                if tid == 0:
                    reset_arrivals(flag)

    @cute.jit
    def finish_tile(self, epilogue, d, tid, warp_id, token_start, weight_start):
        values = cute.make_rmem_tensor((1, 4), epilogue.element_type)
        converted = cute.make_rmem_tensor((1, 4), d.element_type)
        for v in cutlass.range_constexpr(cute.ceil_div(self.bm * self.bn, self.warps * 32 * 4)):
            index = (tid + v * self.warps * 32) * 4
            row, col = index // self.weight_tile, index % self.weight_tile
            shared_source = cute.local_tile(epilogue, (1, 4), (row, col // 4))
            if warp_id < self.warps and index < self.bm * self.bn:
                if cutlass.const_expr(self.noepi):
                    cute.autovec_copy(shared_source, values)
                    for e in cutlass.range_constexpr(cute.size(cute.recast_tensor(values, Int32))):
                        diagnostic_store(
                            cute.recast_ptr(shared_source.iterator, dtype=Int32) + e,
                            cute.recast_tensor(values, Int32)[e],
                        )
                else:
                    if token_start + row < d.shape[0]:
                        if cutlass.const_expr(self.epi == "private"):
                            # Four 16-bit values form one eight-byte shared read.
                            cute.autovec_copy(shared_source, converted)
                        else:
                            cute.autovec_copy(shared_source, values)
                            converted.store(values.load().to(d.element_type))
                        store_output(d, token_start + row, weight_start + col, converted)

    @cute.kernel
    def run(
        self,
        ta: cute.CopyAtom,
        ga: cute.Tensor,
        tb: cute.CopyAtom,
        gb: cute.Tensor,
        sa: cute.Tensor,
        sw: cute.Tensor,
        d: cute.Tensor,
        workspace: cute.Tensor | None,
        la: cute.ComposedLayout,
        lb: cute.ComposedLayout,
        mma: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bid, _, _ = cute.arch.block_idx()
        mt = cute.ceil_div(d.shape[0], self.token_tile)
        nt = cute.ceil_div(sw.shape[0] * 128, self.weight_tile)
        tiles = mt * nt
        dp_tiles, sk_tiles, sk_ctas = tiles, Int32(0), Int32(1)
        if cutlass.const_expr(self.scheduler == "tailsplit" and sw.shape[1] > 1):
            remainder = tiles % self.sm_count
            if remainder > 0 and remainder < self.tail_split_threshold_tiles:
                sk_tiles = Int32(remainder)
                sk_ctas = Int32(
                    cutlass.min(
                        self.sm_count, cutlass.max(1, sk_tiles * sw.shape[1] // self.min_sk_iters)
                    )
                )
                if cutlass.const_expr(self.sk_waves == 2):
                    if tiles >= self.sm_count:
                        sk_tiles = Int32(remainder + self.sm_count)
                        sk_ctas = Int32(self.sm_count)
                dp_tiles = tiles - sk_tiles
        step = tiles if self.scheduler == "grid" else self.sm_count
        dp_count = (dp_tiles + step - 1 - bid) // step
        # Clamp idle CTAs to an empty range; sk_ctas=1 also avoids fallback /0.
        sk_iterations = sk_tiles * sw.shape[1]
        sk_begin = sk_iterations * cutlass.min(bid, sk_ctas) // sk_ctas
        sk_end = sk_iterations * cutlass.min(bid + 1, sk_ctas) // sk_ctas
        unsplit_begin = cute.ceil_div(sk_begin, sw.shape[1])
        unsplit_end = sk_end // sw.shape[1]
        unsplit_count = cutlass.max(0, unsplit_end - unsplit_begin)
        helper_count, owner_count = Int32(0), Int32(0)
        if sk_begin < sk_end:
            if sk_end % sw.shape[1] != 0:
                helper_count = Int32(1)
            if sk_begin % sw.shape[1] != 0 and sk_end >= unsplit_begin * sw.shape[1]:
                owner_count = Int32(1)
        work_count = helper_count + dp_count + unsplit_count + owner_count
        # Contiguity proves at most one helper (right boundary) and one split
        # owner (left boundary); all interior tiles are unsplit. A range wholly
        # inside a tile is a helper unless it reaches the last K block: this
        # also covers many-way splits with sk_waves=1, using slot=helper CTA id.
        # Publish the helper as FIRST SK work, before or after DP. Active SK
        # leaves dp_tiles divisible by S, so every CTA has the same DP count.
        # DP/helpers never wait; owners depend only on helpers' first SK item,
        # never on another owner wait. With S resident CTAs there is no chain.
        dp_start, helper_start = helper_count, Int32(0)
        if cutlass.const_expr(self.sk_order == "dp_first"):
            dp_start, helper_start = Int32(0), dp_count

        storage = SmemAllocator().allocate(self.storage)
        full = storage.barriers.data_ptr()
        empty = full + self.stages
        scales = storage.scales.get_tensor(
            cute.make_layout(
                (self.scale_rows + 1, self.stages),
                stride=(1, self.scale_rows + 1),
            )
        )
        left = storage.left.get_tensor(la.outer, swizzle=la.inner)
        right = storage.right.get_tensor(lb.outer, swizzle=lb.inner)
        if warp_id == self.warps:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + s, self.producer_warps * 32)
                    cute.arch.mbarrier_init(empty + s, self.warps * 32)
            cpasync.prefetch_descriptor(ta)
            cpasync.prefetch_descriptor(tb)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        thread_mma = mma.get_slice(tid)
        ra = mma.make_fragment_A(thread_mma.partition_A(left)[None, None, None, 0])
        rb = mma.make_fragment_B(thread_mma.partition_B(right)[None, None, None, 0])
        if cutlass.const_expr(self.mx and not self.nomma):
            scale_tile = (max(128, self.bm), max(128, self.bn), 128)
            unit_a = unit_scale_fragment(
                mma, tid, left.iterator, cute.size(ra, mode=[1]), "a", scale_tile
            )
            unit_b = unit_scale_fragment(
                mma, tid, right.iterator, cute.size(rb, mode=[1]), "b", scale_tile
            )
        shape = mma.partition_shape_C((self.bm, self.bn))
        acc = cute.make_rmem_tensor(shape, Float32)
        partial = cute.make_rmem_tensor(shape, Float32)
        if cutlass.const_expr(not self.noscale):
            if cutlass.const_expr(self.swap):
                factors = cute.make_rmem_tensor((2 * cute.size(acc, mode=[2]),), Float32)
            else:
                factors = cute.make_rmem_tensor((2 * cute.size(acc, mode=[1]),), Float32)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), Float8E4M3FN), mma
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), Float8E4M3FN), mma
        )
        ca, cb = copy_a.get_slice(tid), copy_b.get_slice(tid)
        pa, pb = ca.partition_S(left), cb.partition_S(right)
        va, vb = ca.retile(ra), cb.retile(rb)
        if cutlass.const_expr(not self.swap):
            store_c = cute.make_tiled_copy_C(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    d.element_type if self.epi == "private" else Float32,
                    num_bits_per_copy=32 if self.epi == "private" else 64,
                ),
                mma,
            )
            store_thread = store_c.get_slice(tid)
        assert cute.size(acc, mode=[0]) == 4  # m16n8 FP32 C atom, used by the scale cache.
        # Each consumer's C values occur exactly once in a full tile slot.
        assert cute.size(acc) * self.warps * 32 == self.bm * self.bn
        coords = thread_mma.partition_C(cute.make_identity_tensor((self.bm, self.bn)))
        # Preserve contiguous four-element reads; spread MMA stores over banks.
        # Fixup finishes in FP32 registers before either epilogue. Private
        # storage rounds once, just as finish_tile does for alias storage.
        epi_layout = cute.make_composed_layout(
            cute.make_swizzle(*self.epilogue_swizzle),
            0,
            cute.make_layout((self.token_tile, self.weight_tile), stride=(self.weight_tile, 1)),
        )
        if cutlass.const_expr(self.epi == "private"):
            epilogue = cute.make_tensor(storage.epilogue.data_ptr(), epi_layout)
            output_acc = cute.make_rmem_tensor(shape, d.element_type)
        else:
            epilogue = cute.make_tensor(
                cute.recast_ptr(storage.left.data_ptr(), dtype=Float32),
                epi_layout,
            )
        stage, phase = Int32(0), Int32(0)
        for work in cutlass.range(work_count, unroll=1):
            tile = Int32(bid + (work - dp_start) * step)
            k_begin, k_end = Int32(0), Int32(sw.shape[1])
            tail_tile, first, helpers = Int32(-1), Int32(0), Int32(0)
            if cutlass.const_expr(self.scheduler == "tailsplit"):
                if work >= helper_start and work < helper_start + helper_count:
                    tail_tile = Int32(sk_end // sw.shape[1])
                elif work >= helper_count + dp_count:
                    tail_tile = Int32(unsplit_begin + work - helper_count - dp_count)
                    if work >= helper_count + dp_count + unsplit_count:
                        tail_tile = Int32(sk_begin // sw.shape[1])
                if tail_tile >= 0:
                    tile = Int32(dp_tiles + tail_tile)
                    k_begin = Int32(cutlass.max(sk_begin - tail_tile * sw.shape[1], 0))
                    k_end = Int32(cutlass.min(sk_end - tail_tile * sw.shape[1], sw.shape[1]))
                    first = Int32(
                        ((tail_tile * sw.shape[1] + 1) * sk_ctas - 1)
                        // cutlass.max(sk_iterations, 1)
                    )
                    if k_end == sw.shape[1]:
                        helpers = Int32(bid - first)
            tm, tn = self.tile_coords(tile, mt, nt)
            im, jn = Int64(tm), Int64(tn)
            if cutlass.const_expr(self.swap):
                im, jn = Int64(tn), Int64(tm)
            token_start = Int64(tm) * self.token_tile
            weight_start = Int64(tn) * self.weight_tile
            gl = cute.local_tile(ga, (self.bm, 128), (im, None))
            gr = cute.local_tile(gb, (self.bn, 128), (jn, None))
            sl, tl = cpasync.tma_partition(
                ta,
                0,
                cute.make_layout(1),
                cute.group_modes(left, 0, 2),
                cute.group_modes(gl, 0, 2),
            )
            sr, tr = cpasync.tma_partition(
                tb,
                0,
                cute.make_layout(1),
                cute.group_modes(right, 0, 2),
                cute.group_modes(gr, 0, 2),
            )
            if warp_id >= self.warps:
                loop_begin = k_begin
                if cutlass.const_expr(self.pdl):
                    if work == 0:
                        # First tile starts at stage=phase=0. Set the expectation
                        # ONCE for both operands; defer ALL producer arrivals until
                        # scales are stored below. Weight completion cannot release
                        # a full barrier, and every scale writer retains its release.
                        prefills = Int32(cutlass.min(self.stages, k_end - k_begin))
                        for s in cutlass.range(prefills, unroll=1):
                            cute.arch.mbarrier_wait(empty + s, 1)
                            if tid == self.warps * 32:
                                cute.arch.mbarrier_expect_tx(full + s, (self.bm + self.bn) * 128)
                            cute.arch.sync_warp()
                            if cutlass.const_expr(self.swap):
                                cute.copy(
                                    ta, tl[None, k_begin + s], sl[None, s], tma_bar_ptr=full + s
                                )
                            else:
                                cute.copy(
                                    tb, tr[None, k_begin + s], sr[None, s], tma_bar_ptr=full + s
                                )
                        cute.arch.griddepcontrol_wait()
                        for s in cutlass.range(prefills, unroll=1):
                            for r in cutlass.range_constexpr(cute.ceil_div(self.token_tile, 32)):
                                local_row = tid - self.warps * 32 + 32 * r
                                scale_row = token_start + Int64(local_row)
                                scale_value = Float32(0)
                                if local_row < self.scale_rows:
                                    if scale_row < d.shape[0]:
                                        scale_value = sa[scale_row, k_begin + s]
                                    scales[local_row, s] = scale_value
                            if tid == self.warps * 32:
                                scales[self.scale_rows, s] = sw[weight_start // 128, k_begin + s]
                            cute.arch.mbarrier_arrive(full + s)
                            cute.arch.sync_warp()
                            if cutlass.const_expr(self.swap):
                                cute.copy(
                                    tb, tr[None, k_begin + s], sr[None, s], tma_bar_ptr=full + s
                                )
                            else:
                                cute.copy(
                                    ta, tl[None, k_begin + s], sl[None, s], tma_bar_ptr=full + s
                                )
                            stage += 1
                            if stage == self.stages:
                                stage = Int32(0)
                                phase ^= 1
                        loop_begin = k_begin + prefills
                for kb in cutlass.range(loop_begin, k_end, unroll=1):
                    cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                    for r in cutlass.range_constexpr(
                        cute.ceil_div(self.token_tile, self.producer_warps * 32)
                    ):
                        local_row = tid - self.warps * 32 + self.producer_warps * 32 * r
                        scale_row = token_start + Int64(local_row)
                        scale_value = Float32(0)
                        if local_row < self.scale_rows:
                            if scale_row < d.shape[0]:
                                scale_value = sa[scale_row, kb]
                            scales[local_row, stage] = scale_value
                    if tid == self.warps * 32:
                        scales[self.scale_rows, stage] = sw[weight_start // 128, kb]
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, (self.bm + self.bn) * 128
                        )
                    else:
                        cute.arch.mbarrier_arrive(full + stage)
                    # Whichever lane the TMA atom elects must observe lane 0's
                    # transaction expectation before issuing either bulk copy.
                    if warp_id == self.warps:
                        cute.arch.sync_warp()
                        cute.copy(ta, tl[None, kb], sl[None, stage], tma_bar_ptr=full + stage)
                        cute.copy(tb, tr[None, kb], sr[None, stage], tma_bar_ptr=full + stage)
                    stage += 1
                    if stage == self.stages:
                        stage = Int32(0)
                        phase ^= 1
            else:
                if cutlass.const_expr(self.pdl):
                    if work == 0:
                        # Wait in EVERY consuming warp before activation reads,
                        # including the ctas_per_sm=2 global sa fallback. Together
                        # with the producer wait this also orders all d/workspace
                        # accesses (counters/partials included) in fixup. No
                        # workspace access precedes this wait; weight TMA may.
                        cute.arch.griddepcontrol_wait()
                acc.fill(0.0)
                for _kb in cutlass.range(k_begin, k_end, unroll=1):
                    cute.arch.mbarrier_wait(full + stage, phase)
                    # SM120 C fragments have atom values (column pair, row pair),
                    # followed by M/N repetitions. Load only distinct activation
                    # scales, then broadcast products across the other dimension.
                    if cutlass.const_expr(not self.noscale):
                        weight_scale = scales[self.scale_rows, stage]
                        if cutlass.const_expr(self.swap):
                            for n in cutlass.range_constexpr(cute.size(acc, mode=[2])):
                                for c in cutlass.range_constexpr(2):
                                    e = c + 4 * cute.size(acc, mode=[1]) * n
                                    factors[2 * n + c] = (
                                        self.activation_scale(
                                            scales,
                                            sa,
                                            coords[e][1],
                                            token_start,
                                            d.shape[0],
                                            _kb,
                                            stage,
                                        )
                                        * weight_scale
                                    )
                        else:
                            for m in cutlass.range_constexpr(cute.size(acc, mode=[1])):
                                for r in cutlass.range_constexpr(2):
                                    e = 2 * r + 4 * m
                                    factors[2 * m + r] = (
                                        self.activation_scale(
                                            scales,
                                            sa,
                                            coords[e][0],
                                            token_start,
                                            d.shape[0],
                                            _kb,
                                            stage,
                                        )
                                        * weight_scale
                                    )
                    partial.fill(0.0)
                    cute.copy(copy_a, pa[None, None, 0, stage], va[None, None, 0])
                    cute.copy(copy_b, pb[None, None, 0, stage], vb[None, None, 0])
                    # Two live K=32 register slots: prefetch next before current MMA.
                    # Slots 2/3 of the fragment are unused and eliminated by the compiler.
                    for kk in cutlass.range_constexpr(4):
                        if cutlass.const_expr(kk < 3):
                            nxt = (kk + 1) % 2
                            cute.copy(copy_a, pa[None, None, kk + 1, stage], va[None, None, nxt])
                            cute.copy(copy_b, pb[None, None, kk + 1, stage], vb[None, None, nxt])
                        if cutlass.const_expr(kk == 2):
                            # All shared reads, including scales, have completed.
                            # Producer can refill while the last MMAs/rescale execute.
                            cute.arch.mbarrier_arrive(empty + stage)
                        if cutlass.const_expr(self.nomma):
                            # Fold every loaded register into a per-thread volatile
                            # sink: no MMA, but ldmatrix/TMA cannot be dead-code removed.
                            packed_a = cute.recast_tensor(ra[None, None, kk % 2], Int32)
                            packed_b = cute.recast_tensor(rb[None, None, kk % 2], Int32)
                            checksum = Int32(0)
                            for e in cutlass.range_constexpr(cute.size(packed_a)):
                                checksum ^= packed_a[e]
                            for e in cutlass.range_constexpr(cute.size(packed_b)):
                                checksum ^= packed_b[e]
                            diagnostic_store(storage.diagnostic_sink.data_ptr() + tid, checksum)
                        elif cutlass.const_expr(self.mx):
                            cute.gemm(
                                mma,
                                partial,
                                [ra[None, None, kk % 2], unit_a],
                                [rb[None, None, kk % 2], unit_b],
                                partial,
                            )
                        else:
                            cute.gemm(
                                mma,
                                partial,
                                ra[None, None, kk % 2],
                                rb[None, None, kk % 2],
                                partial,
                            )
                    if cutlass.const_expr(self.noscale):
                        acc.store(acc.load() + partial.load())
                    else:
                        for e in cutlass.range_constexpr(cute.size(acc)):
                            if cutlass.const_expr(self.swap):
                                f = 2 * (e // (4 * cute.size(acc, mode=[1]))) + e % 2
                            else:
                                f = 2 * (e // 4 % cute.size(acc, mode=[1])) + e // 2 % 2
                            acc[e] = acc[e] + partial[e] * factors[f]
                    stage += 1
                    if stage == self.stages:
                        stage = Int32(0)
                        phase ^= 1
                if cutlass.const_expr(self.scheduler == "tailsplit" and not self.noepi):
                    self.fixup(
                        acc,
                        workspace,
                        tid,
                        bid,
                        tail_tile,
                        first,
                        helpers,
                        k_end,
                        Int32(sw.shape[1]),
                    )
            if cutlass.const_expr(self.epi == "private"):
                if cutlass.const_expr(self.pdl):
                    if work == work_count - 1:
                        cute.arch.griddepcontrol_launch_dependents()
                if warp_id < self.warps:
                    if k_end == sw.shape[1]:
                        # Finish previous shared reads before overwriting. The
                        # producer may run ahead, bounded by empty mbarriers.
                        cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                        output_acc.store(acc.load().to(d.element_type))
                        if cutlass.const_expr(self.swap):
                            for e in cutlass.range_constexpr(cute.size(acc)):
                                epilogue[coords[e][1], coords[e][0]] = output_acc[e]
                        else:
                            cute.copy(
                                store_c,
                                store_thread.retile(output_acc),
                                store_thread.partition_D(epilogue),
                            )
                        cute.arch.barrier(barrier_id=1, number_of_threads=self.warps * 32)
                        self.finish_tile(epilogue, d, tid, warp_id, token_start, weight_start)
            else:
                # Join producer and consumers before aliasing any operand stage.
                cute.arch.sync_threads()
                if cutlass.const_expr(self.pdl):
                    if work == work_count - 1:
                        cute.arch.griddepcontrol_launch_dependents()
                if k_end == sw.shape[1]:
                    if warp_id < self.warps:
                        if cutlass.const_expr(self.swap):
                            for e in cutlass.range_constexpr(cute.size(acc)):
                                epilogue[coords[e][1], coords[e][0]] = acc[e]
                        else:
                            cute.copy(
                                store_c,
                                store_thread.retile(acc),
                                store_thread.partition_D(epilogue),
                            )
                    cute.arch.sync_threads()
                    self.finish_tile(epilogue, d, tid, warp_id, token_start, weight_start)
                    # Alias generic shared accesses transition back to TMA.
                    if cutlass.const_expr(self.scheduler != "grid"):
                        cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            # Idle CTAs touch neither operands nor workspace.
            if work_count == 0:
                cute.arch.griddepcontrol_launch_dependents()
