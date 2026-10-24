# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# _moe_utils/moe_epilogue.py, _moe_utils/moe_activation.py).
# Local modifications: native gate/up, weighted FP32 activation, exact BF16 K32
# encoder, unique scale-word stores, and masked scalar BF16 atomics without weights.

"""After-S2R Q1 and unweighted BF16 output epilogues (lazy DSL import)."""

from functools import lru_cache


@lru_cache(maxsize=1)
def device_epilogues():
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    from ._quant import _bf16_encoder

    scale_from_peak, encode_bits = _bf16_encoder()

    @cute.jit
    def multiply(a, b):
        # Do not contract route weighting across the declared BF16 boundary.
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [a.ir_value(), b.ir_value()],
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def divide(a, b):
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [a.ir_value(), b.ir_value()],
                "div.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def weighted_swiglu(gate, up, weight):
        gate = cutlass.min(gate, Float32(10))
        up = cutlass.max(Float32(-10), cutlass.min(up, Float32(10)))
        # exp receives a nonpositive argument: extreme negative gates cannot
        # overflow the sigmoid denominator. Use rounded division, not rcp.approx.
        sigmoid = Float32(0)
        if gate < 0:
            ex = cute.math.exp(gate, fastmath=False)
            sigmoid = divide(ex, Float32(1) + ex)
        else:
            sigmoid = divide(Float32(1), Float32(1) + cute.math.exp(-gate, fastmath=False))
        return multiply(multiply(multiply(gate, sigmoid), up), weight)

    @cute.jit
    def lora_delta(z, b, slot, pair, expert, channel, scale, rank: cutlass.Constexpr):
        # scale * sum_r z[pair, slot, r] * B[expert, channel, r]: FP32, fixed order.
        total = Float32(0)
        for k in cutlass.range_constexpr(rank):
            total = total + z[pair, slot, k] * b[expert, channel, k].to(Float32)
        return multiply(scale, total)

    @cute.jit
    def lora_seed(
        acc,
        coords,
        row,
        end,
        channel,
        swapped: cutlass.Constexpr,
        lora,
        slot,
        expert,
        rank: cutlass.Constexpr,
        by_row: cutlass.Constexpr,
    ):
        # Seed an FP32 accumulator with the tile's LoRA delta before the K loop: its loads
        # overlap the first TMA stages and the MMA adds the base product on top.
        # lora = (z [P,slots,r], B [E,N,r], scale [1], row_pair); z is in pair order, or in
        # sorted-row order with by_row.
        z, b, scale, row_pair = lora
        for j in cutlass.range_constexpr(cute.size(acc)):
            rr, cc = coords[j]
            if cutlass.const_expr(swapped):
                rr, cc = cc, rr
            value = Float32(0)
            if row + rr < end:
                entry = row + rr
                if cutlass.const_expr(not by_row):
                    entry = row_pair[row + rr]
                value = lora_delta(z, b, slot, entry, expert, channel + cc, scale[0], rank)
            acc[j] = value

    @cute.jit
    def lora_mma_seed(
        mma,
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
        lora,
        slot,
        expert,
        rank: cutlass.Constexpr,
    ):
        # The same seed on tensor cores (non-swapped tiles, z in sorted-row order): a BF16
        # m16n8k16 tiled MMA sharing the W4A8 accumulator layout. z is split into three BF16
        # terms (hi, mid, lo: ~24 mantissa bits, as FP32); B is BF16 already. Two terms (~16
        # bits) measured 5x the scalar seed's delta error on GB10 via flipped BF16 h roundings.
        z, b, scale, _ = lora
        acc.fill(0)
        for k0 in cutlass.range_constexpr(0, rank, 16):
            for j in cutlass.range_constexpr(cute.size(a_hi)):
                m, k = a_coords[j]
                v = Float32(0)
                if (row + m < end) & (k0 + k < rank):
                    v = z[row + m, slot, k0 + k]
                hi = v.to(BFloat16)
                rest = v - hi.to(Float32)
                mid = rest.to(BFloat16)
                a_hi[j] = hi
                a_mid[j] = mid
                a_lo[j] = (rest - mid.to(Float32)).to(BFloat16)
            for j in cutlass.range_constexpr(cute.size(frag_b)):
                n, k = b_coords[j]
                w = BFloat16(0)
                if k0 + k < rank:
                    w = b[expert, channel + n, k0 + k]
                frag_b[j] = w
            cute.gemm(mma, acc, a_lo, frag_b, acc)
            cute.gemm(mma, acc, a_mid, frag_b, acc)
            cute.gemm(mma, acc, a_hi, frag_b, acc)
        for j in cutlass.range_constexpr(cute.size(acc)):
            acc[j] = multiply(scale[0], acc[j])

    @cute.jit
    def stage_fc1(
        gate,
        up,
        coords,
        shared,
        routes,
        row,
        end,
        channel,
        swapped: cutlass.Constexpr,
        trace_gate,
        trace_up,
        trace_h,
        debug: cutlass.Constexpr,
    ):
        for j in cutlass.range_constexpr(cute.size(gate)):
            rr, cc = coords[j]
            if cutlass.const_expr(swapped):
                rr, cc = cc, rr
            value = BFloat16(0)
            if row + rr < end:
                value = BFloat16(weighted_swiglu(gate[j], up[j], routes[row + rr]))
                if cutlass.const_expr(debug):
                    trace_gate[row + rr, channel + cc] = gate[j]
                    trace_up[row + rr, channel + cc] = up[j]
                    trace_h[row + rr, channel + cc] = value
            shared[rr, cc] = value

    @cute.jit
    def lora_partials(
        shared,
        lora,
        row,
        end,
        expert,
        channel,
        ntile,
        tid,
        rows: cutlass.Constexpr,
        bn: cutlass.Constexpr,
        rank: cutlass.Constexpr,
        by_row: cutlass.Constexpr,
    ):
        # Down-LoRA shrink of this tile's BF16 h (staged in shared memory): one FP32 partial
        # per (entry, N tile, rank), entries being pairs or, with by_row, sorted rows. The
        # thread that completes an (entry, rank) pair (an integer ticket, reset to zero after
        # use) sums its N tiles in a fixed order into z2.
        _, _, _, _, row_pair, a2, parts, z2, count = lora
        ntiles = cute.size(parts, mode=[1])
        # One thread per (row, group of 4 ranks): each 128-bit h chunk (shared) is read once
        # for its four A2 rows (global, contiguous in channels). The chunk order is rotated
        # by the row's pair id so a warp's rows hit different shared banks; the pair id, not
        # the row's position (which PREP's counters may vary), fixes every sum's order.
        load = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), BFloat16, num_bits_per_copy=128)
        hv = cute.make_rmem_tensor((8,), BFloat16)
        av = cute.make_rmem_tensor((8,), BFloat16)
        groups = rank // 4
        chunks = bn // 8
        for q in cutlass.range_constexpr((rows * groups + 255) // 256):
            index = q * 256 + tid
            if index < rows * groups:
                rr, k0 = index // groups, (index % groups) * 4
                if row + rr < end:
                    entry = row_pair[row + rr]
                    # Two FP32 chains per rank (ILP), combined in a fixed order.
                    sums = [Float32(0)] * 8
                    for c in cutlass.range_constexpr(chunks):
                        cc = (c + entry) % chunks * 8
                        cute.autovec_copy(
                            cute.make_tensor(shared.iterator + (rr * bn + cc), cute.make_layout(8)),
                            hv,
                        )
                        for kk in cutlass.range_constexpr(4):
                            offset = cute.crd2idx((expert, k0 + kk, channel + cc), a2.layout)
                            src = cute.make_tensor(
                                cute.make_ptr(
                                    BFloat16,
                                    a2.iterator.toint() + Int64(offset) * 2,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout(8),
                            )
                            cute.copy(load, src, av)
                            for j in cutlass.range_constexpr(8):
                                s = 2 * kk + j % 2
                                sums[s] = sums[s] + hv[j].to(Float32) * av[j].to(Float32)
                    pair = entry
                    if cutlass.const_expr(by_row):
                        pair = row + rr
                    for kk in cutlass.range_constexpr(4):
                        parts[pair, ntile, k0 + kk] = sums[2 * kk] + sums[2 * kk + 1]
                    # Publish the four partials, then take their tickets: the thread that
                    # completes an (entry, rank) pair sums its N tiles in order into z2.
                    cute.arch.fence_acq_rel_gpu()
                    for kk in cutlass.range_constexpr(4):
                        k = k0 + kk
                        ticket = cute.arch.atomic_add(
                            count.iterator + (pair * rank + k), Int32(1), sem="acq_rel", scope="gpu"
                        )
                        if ticket == ntiles - 1:
                            z = Float32(0)
                            for t in cutlass.range(ntiles):
                                z = z + parts[pair, t, k]
                            z2[pair, 0, k] = z
                            count[pair * rank + k] = Int32(0)

    @cute.jit
    def quantize(
        shared,
        q,
        scales,
        row,
        scale_row,
        end,
        channel_tile,
        tid,
        rows: cutlass.Constexpr,
        bn: cutlass.Constexpr,
    ):
        bits = cute.recast_tensor(shared, Uint16)
        lane, warp = tid % 32, tid // 32
        # One warp owns a row. Lane zero is the only scale writer per tile/row.
        # BN64 tiles 2j/2j+1 own bytes 0:2/2:4 of word j, respectively:
        # byte stores never read/modify/write the neighbour's half word.
        scale_bytes = cute.make_tensor(
            cute.recast_ptr(scales.iterator, dtype=Uint8),
            cute.make_layout(cute.size(scales) * 4),
        )
        for block in cutlass.range_constexpr(rows // 8):
            rr = block * 8 + warp
            word = Uint32(0)
            for group in cutlass.range_constexpr(bn // 32):
                raw = Int32(bits[rr, group * 32 + lane])
                peak = raw & 32767
                for bit in cutlass.range_constexpr(5):
                    peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, 1 << bit))
                scale = scale_from_peak(peak)
                word |= Uint32(scale) << (8 * group)
                if row + rr < end:
                    if cutlass.const_expr(bn == 64):
                        if lane == 0:
                            address = 4 * (
                                (channel_tile // 2) * cute.size(scales, mode=[1]) + scale_row + rr
                            )
                            scale_bytes[address + (channel_tile % 2) * 2 + group] = Uint8(scale)
                    q[row + rr, channel_tile * bn + group * 32 + lane] = encode_bits(
                        raw, peak, scale
                    )
            if cutlass.const_expr(bn == 128):
                if (lane == 0) & (row + rr < end):
                    scales[channel_tile, scale_row + rr] = Int32(word)

    @cute.jit
    def atomic_bf16(pointer, bits):
        # Public PTX scalar BF16 reduction requires only element alignment.
        # This also accepts contiguous out views whose base is offset by 2 B.
        llvm.inline_asm(
            None,
            [Int64(pointer.toint()).ir_value(), Uint16(bits).ir_value()],
            "red.global.add.noftz.bf16 [$0], $1;",
            "l,h",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def stage_fc2(
        acc,
        coords,
        shared,
        row,
        end,
        channel,
        trace_fc2,
        debug: cutlass.Constexpr,
        rows_out,
    ):
        # rows_out (BF16 [rows, H], the row combine's input) takes each route's row, rounded
        # once from FP32; otherwise the BF16 values are staged for the atomic scatter.
        for j in cutlass.range_constexpr(cute.size(acc)):
            rr, cc = coords[j]
            value = BFloat16(0)
            if row + rr < end:
                y = acc[j]
                value = BFloat16(y)
                if cutlass.const_expr(rows_out is not None):
                    rows_out[row + rr, channel + cc] = value
                if cutlass.const_expr(debug):
                    trace_fc2[row + rr, channel + cc] = acc[j]
            if cutlass.const_expr(rows_out is None):
                shared[rr, cc] = value

    @cute.jit
    def scatter(shared, tokens, out, row, end, channel, tid, rows: cutlass.Constexpr):
        bits = cute.recast_tensor(shared, Uint16)
        for block in cutlass.range_constexpr(rows * 128 // 256):
            index = block * 256 + tid
            rr, cc = index // 128, index % 128
            # Both the metadata read and atomic are predicated. Never load a
            # real token for a padded row and attempt to neutralize it with 0.
            if row + rr < end:
                token = Int64(tokens[row + rr])
                address = out.iterator + token * cute.size(out, mode=[1]) + channel + cc
                atomic_bf16(address, bits[rr, cc])

    return stage_fc1, quantize, stage_fc2, scatter, lora_seed, lora_mma_seed, lora_partials
