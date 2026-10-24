# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Warp-per-token dX epilogue with 128-bit row accesses and ordered slot sums."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr


class TokenBackward:
    def __init__(self, h, k, nt, shared, masked, splits):
        self.h, self.k, self.nt = h, k, nt
        self.shared, self.masked = shared, masked
        self.kp1, self.kpad = k + shared, 1 << (k - 1).bit_length()
        self.splits = splits

    @cute.jit
    def __call__(self, dxp, inv, dwpart, ws, sel, r, sg, dx, valid, stream: cuda.CUstream):
        self.kernel(dxp, inv, dwpart, ws, sel, r, sg, dx, valid).launch(
            grid=(cute.ceil_div(dx.shape[0], 4), self.splits, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load8(self, source, row, col):
        fragment = cute.make_rmem_tensor((8,), source.element_type)
        vector = cute.make_tensor(
            source.iterator + cute.assume(row.to(Int64) * self.h + col, divby=8),
            cute.make_layout((8,)),
        )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), source.element_type, num_bits_per_copy=128
        )
        cute.copy(atom, vector, fragment)
        return fragment

    @cute.kernel
    def kernel(self, dxp, inv, dwpart, ws, sel, r, sg, dx, valid):
        tid, _, _ = cute.arch.thread_idx()
        bid, split, _ = cute.arch.block_idx()
        lane, t = tid % 32, bid * 4 + tid // 32
        if t < dx.shape[0]:
            active = t < dx.shape[0]
            if const_expr(self.masked):
                # t is nonnegative and bounded by T, so this is equivalent to
                # comparing against clamp(valid, 0, T), without overflow.
                active = t < valid[0]
            pos, expert = Int32(-1), Int32(0)
            weight, dw, dz, ds = Float32(0), Float32(0), Float32(0), Float32(0)
            if active:
                if lane < self.kp1:
                    slot = t * self.kp1 + lane
                    pos = inv[slot]
                    weight = ws[slot]
                    if pos >= 0:
                        # Every slot walks tiles in exactly their original order.
                        for n in cutlass.range_constexpr(self.nt):
                            dw += dwpart[pos, n]
                if lane < self.k:
                    expert = sel[t, lane].to(Int32)
                product = weight * dw if lane < self.k else Float32(0)
                for step in cutlass.range_constexpr((self.kpad - 1).bit_length()):
                    product += cute.arch.shuffle_sync_bfly(product, self.kpad >> (step + 1))
                total = cute.arch.shuffle_sync(product, 0)
                dz = weight * (dw - total)
                if const_expr(self.shared):
                    gate_dw = cute.arch.shuffle_sync(dw, self.k)
                    gamma = cute.arch.shuffle_sync(weight, self.k)
                    ds = gamma * (Float32(1) - gamma) * gate_dw
            # Broadcast metadata before any per-lane H-tail predicate. All 32
            # lanes must participate, even when H is smaller than 256.
            positions = cute.make_rmem_tensor((self.kp1,), Int32)
            experts = cute.make_rmem_tensor((self.k,), Int32)
            derivs = cute.make_rmem_tensor((self.k,), Float32)
            for j in cutlass.range_constexpr(self.kp1):
                positions[j] = cute.arch.shuffle_sync(pos, j)
            for j in cutlass.range_constexpr(self.k):
                experts[j] = cute.arch.shuffle_sync(expert, j)
                derivs[j] = cute.arch.shuffle_sync(dz, j)
            for tile in range(cute.ceil_div(self.h, 256 * self.splits)):
                col = (tile * self.splits + split) * 256 + lane * 8
                if col < self.h:
                    acc = cute.make_rmem_tensor((8,), Float32)
                    acc.fill(0)
                    if active:
                        for j in cutlass.range_constexpr(self.kp1):
                            if positions[j] >= 0:
                                values = self.load8(dxp, positions[j], col)
                                acc.store(acc.load() + values.load().to(Float32))
                        # Keep router products separate from X until their tree
                        # reduction completes, as in the Triton token epilogue.
                        products = cute.make_rmem_tensor((8, self.kpad), Float32)
                        products.fill(0)
                        for j in cutlass.range_constexpr(self.k):
                            values = self.load8(r, experts[j], col)
                            for v in cutlass.range_constexpr(8):
                                products[v, j] = derivs[j] * values[v].to(Float32)
                        for step in cutlass.range_constexpr((self.kpad - 1).bit_length()):
                            for j in cutlass.range_constexpr(self.kpad >> (step + 1)):
                                for v in cutlass.range_constexpr(8):
                                    products[v, j] += products[v, j + (self.kpad >> (step + 1))]
                        for v in cutlass.range_constexpr(8):
                            acc[v] += products[v, 0]
                        if const_expr(self.shared):
                            values = self.load8(sg, Int32(0), col)
                            acc.store(acc.load() + ds * values.load().to(Float32))
                    result = cute.make_rmem_tensor((8,), BFloat16)
                    result.store(acc.load().to(BFloat16))
                    target = cute.make_tensor(
                        dx.iterator + cute.assume(t.to(Int64) * self.h + col, divby=8),
                        cute.make_layout((8,)),
                    )
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
                    )
                    cute.copy(atom, result, target)
