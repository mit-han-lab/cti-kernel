# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Original CuTe mHC kernels; no third-party implementation code.

Equations: DeepSeek-V4-Flash@60d8d707 inference/model.py:673-700 and
inference/kernel.py:372-440 (Sinkhorn axis order at 398-424). Split projection
and unrounded post statistics are also described in vLLM 0.31.0
model_executor/kernels/mhc/tilelang_kernels.py:741-755; this implementation adds
Gram normalization and a single-launch deterministic exchange.

Local idioms: collective/_host_exchange/cute/kernels.py:58-92 (vector IO),
collective/_two_node/cute/kernels.py:94-111 (publication), and
_cute_prepare.py:31-37 (PDL launch). Data pointers use an Int64 ABI; the counter
is a tensor argument so the launcher selects the current PyTorch CUDA stream.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.experimental import iket
from cutlass.runtime import make_fake_stream

from ...collective._host_exchange.cute.kernels import load_raw, put, store_raw
from ...collective._two_node.cute.kernels import _add_rn
from .._config import kernel_options


def _ptr(dtype, address, align=4):
    return cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _load(address):
    return cute.arch.load(_ptr(Float32, address), Float32)


def _store(address, value):
    cute.arch.store(_ptr(Float32, address), value)


def _load8(address):
    raw = cute.arch.load(_ptr(Int32, address, 16), ir.VectorType.get([4], Int32.mlir_type))
    return cute.TensorSSA(raw, (4,), Int32).bitcast(BFloat16).to(Float32)


def _store8(address, values):
    cute.arch.store(_ptr(Int32, address, 16), values.to(BFloat16).bitcast(Int32))


@cute.jit
def _load_shared4(pointer):
    # Keep each vector inside the token loop: hoisting all 24 rows holds 192
    # extra registers across residual calculation and spills on small prefill.
    values = cute.arch.inline_ptx(
        "ld.volatile.shared.v4.f32 { {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[Int32(pointer.toint())],
    )
    result = cute.make_rmem_tensor((4,), Float32)
    for k in cutlass.range_constexpr(4):
        result[k] = values[k]
    return result.load()


def _fp(a, b, op):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"{op}.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _fma(a, b, c):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(v).ir_value() for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _two_sum(a, b):
    s = _fp(a, b, "add")
    v = _fp(s, a, "sub")
    e = _fp(_fp(a, _fp(s, v, "sub"), "sub"), _fp(b, v, "sub"), "add")
    return s, e


@cute.jit
def _dot2(s, c, a, b):
    p = _fp(a, b, "mul")
    e = _fma(a, b, -p)
    hi, lo = _two_sum(s, p)
    return hi, _fp(c, _fp(lo, e, "add"), "add")


def _rounding_input(hi, lo):
    # Only a FP32 value exactly at a BF16 midpoint needs correction. Adjust
    # one FP32 ulp toward the low part; cvt.rn.bf16 then selects the right side.
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [hi.ir_value(), lo.ir_value()],
            "{ .reg .b32 bits, low, step; .reg .pred mid, nz, up, neg; "
            "mov.b32 bits, $1; and.b32 low, bits, 65535; "
            "setp.eq.u32 mid, low, 32768; setp.ne.f32 nz, $2, 0f00000000; "
            "and.pred mid, mid, nz; setp.gt.f32 up, $2, 0f00000000; "
            "setp.lt.f32 neg, $1, 0f00000000; xor.pred up, up, neg; "
            "selp.b32 step, 1, -1, up; @mid add.u32 bits, bits, step; mov.b32 $0, bits; }",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class Kernel:
    def __init__(self, cfg, plan, mode, decode, token_tile=16, *, allreduce=False):
        self.allreduce = allreduce
        self.h = cfg.hidden
        self.rms_eps, self.hc_eps, self.norm_eps = cfg.rms_eps, cfg.hc_eps, cfg.norm_eps
        self.post_mult, self.iters = cfg.post_mult, cfg.sinkhorn_iters
        self.size, self.grid, self.cap = plan
        self.mode, self.decode, self.tt = mode, decode, token_tile
        self.recompute = decode and self.cap > 32
        # Read once at construction; the host cache uses the same normalized knobs.
        (
            self.trace,
            self.trigger,
            self.fastdiv,
            self.chain,
            self.exchange,
            self.publish,
            self.fence,
            self.proj,
            self.ar_prefetch,
        ) = kernel_options()
        self.split_proj = not self.recompute and self.proj == "split"
        self.early = self.trigger != "0"
        self.dtype, self.bytes = Float32, 4
        # Padded G preserves 16-byte alignment for every workspace row.
        self.pitch = (self.grid + 3) // 4 * 4

    @cute.jit
    def __call__(
        self,
        x: Int64,
        r: Int64,
        p: Int64,
        c: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        w: Int64,
        ro: Int64,
        po: Int64,
        co: Int64,
        out: Int64,
        parts: Int64,
        counter: cute.Tensor,
        tokens: Int32,
        words: cute.Tensor,
        host: Int64,
        stride: Int64,
        rank: Int32,
        ysum: Int64,
        stream,
    ):
        if const_expr(self.mode == "post"):
            self.post_kernel(x, r, p, c, ro, tokens).launch(
                grid=[(tokens * (self.h // 8) * 4 + 127) // 128, 1, 1],
                block=[128, 1, 1],
                stream=stream,
                use_pdl=True,
            )
        elif const_expr(self.decode):
            self.decode_kernel(
                x,
                r,
                p,
                c,
                fn,
                scale,
                base,
                w,
                ro,
                po,
                co,
                out,
                parts,
                counter.iterator.toint(),
                tokens,
                words,
                host,
                stride,
                rank,
                ysum,
            ).launch(grid=[self.grid, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=True)
        else:
            self.prefill_kernel(x, r, p, c, fn, scale, base, w, ro, po, co, out, tokens).launch(
                grid=[(tokens + self.tt - 1) // self.tt, 1, 1],
                block=[256, 1, 1],
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def trace_push(self, name: cutlass.Constexpr):
        if const_expr(self.trace):
            iket.range_push(name)

    @cute.jit
    def trace_pop(self):
        if const_expr(self.trace):
            iket.range_pop()

    @cute.jit
    def stage_fn(
        self,
        fn,
        shared,
        h0,
        size: cutlass.Constexpr,
        tid,
        threads: cutlass.Constexpr,
        buffer=0,
        retain_loop: cutlass.Constexpr = False,
    ):
        # Original cp.async staging into vector-friendly shared layouts. Each copy
        # is 16 bytes; H%8 makes every vector wholly valid or wholly padding.
        for step in cutlass.range(96 * size // (4 * threads), unroll=1 if retain_loop else -1):
            v = step * threads + tid
            col = (v % (size // 4)) * 4
            row = v // (size // 4)
            j, m = row // 24, row % 24
            valid = h0 + col < self.h
            address = fn + Int64(((m * 4 + j) * self.h + h0 + col) * 4)
            if not valid:
                address = fn
            if const_expr(self.decode):
                # A warp's 4-float vectors are contiguous for each half of its
                # 8-column group: [m][half][group][stream][4].
                index = m * 4 * size + ((col % 8) // 4) * (2 * size)
                index += (col // 8) * 16 + j * 4
            else:
                # Projection owners differ in m: adjacent owners load adjacent
                # 16-byte vectors, [stream][column/4][m][4].
                index = ((j * (size // 4) + col // 4) * 24 + m) * 4
            dest = shared.iterator + buffer + index
            cute.arch.inline_ptx(
                "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
                read_only_args=[Int32(dest.toint()), address, Int32(16) if valid else Int32(0)],
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def stage_small(self, scale, base, small, tid):
        if tid < 3:
            small[tid] = _load(scale + Int64(tid * 4))
        if tid < 24:
            small[3 + tid] = _load(base + Int64(tid * 4))

    @cute.jit
    def residual(self, x, r, p, c, ro, t, h, j, store: cutlass.Constexpr = True):
        value = cute.make_rmem_tensor((8,), Float32)
        if const_expr(self.mode == "pre"):
            value.store(_load8(r + Int64((t * 4 * self.h + j * self.h + h) * 2)))
        else:
            old = cute.make_rmem_tensor((4, 8), Float32)
            coeff = cute.make_rmem_tensor((4,), Float32)
            for i in cutlass.range_constexpr(4):
                rv = _load8(r + Int64((t * 4 * self.h + i * self.h + h) * 2))
                coeff[i] = _load(c + Int64((t * 16 + i * 4 + j) * 4))
                for k in cutlass.range_constexpr(8):
                    old[i, k] = rv[k]
            xv = _load8(x + Int64((t * self.h + h) * 2))
            post = _load(p + Int64((t * 4 + j) * 4))
            rounded = cute.make_rmem_tensor((8,), Float32)
            for k in cutlass.range_constexpr(8):
                hi = _fp(post, xv[k], "mul")
                lo = _fma(post, xv[k], -hi)
                for i in cutlass.range_constexpr(4):
                    hi, lo = _dot2(hi, lo, coeff[i], old[i, k])
                hi, lo = _two_sum(hi, lo)
                value[k] = hi
                rounded[k] = _rounding_input(hi, lo)
            if const_expr(store):
                _store8(ro + Int64((t * 4 * self.h + j * self.h + h) * 2), rounded.load())
        return value

    @cute.jit
    def add_product(self, sums, comp, m, a, b):
        if const_expr(self.mode == "pre"):
            # Pre reads exact BF16 inputs: compensated FP32 dots keep its projection as
            # accurate as an FP32 GEMM (exit's inputs carry their own rounding).
            hi, lo = _dot2(sums[m], comp[m], a, b)
            sums[m], comp[m] = hi, lo
        else:
            sums[m] = _fma(a, b, sums[m])

    @cute.jit
    def lane_total(self, sums, comp, m):
        value = self.dtype(sums[m])
        if const_expr(self.mode == "pre"):
            value = _fp(value, Float32(comp[m]), "add")
        return value

    @cute.jit
    def add(self, a, b):
        value = _fp(a, b, "add")
        return value

    @cute.jit
    def round_output(self, value):
        hi = Float32(value)
        return hi

    @cute.jit
    def arith(self, a, b, op: cutlass.Constexpr):
        if const_expr(self.fastdiv and op == "div"):
            value = Float32(
                llvm.inline_asm(
                    Float32.mlir_type,
                    [Float32(a).ir_value(), Float32(b).ir_value()],
                    "div.full.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )
        else:
            value = _fp(a, b, op)
        return value

    @cute.jit
    def sigmoid(self, value):
        return self.arith(1.0, self.arith(1.0, cute.exp(-value, fastmath=False), "add"), "div")

    @cute.jit
    def inv_norm(self, ss, n: cutlass.Constexpr, eps: cutlass.Constexpr):
        mean = self.arith(self.arith(ss, n, "div"), eps, "add")
        root = cute.sqrt(mean, fastmath=False)
        return self.arith(1.0, root, "div")

    @cute.jit
    def row_sum(self, value):
        pair = self.arith(value, cute.arch.shuffle_sync_bfly(value, 1), "add")
        return self.arith(pair, cute.arch.shuffle_sync_bfly(pair, 2), "add")

    @cute.jit
    def col_sum(self, value):
        pair = self.arith(value, cute.arch.shuffle_sync_bfly(value, 4), "add")
        return self.arith(pair, cute.arch.shuffle_sync_bfly(pair, 8), "add")

    @cute.jit
    def gains_lane16(self, totals, offset, small, gains, goffset, po, token, write, valid):
        # Two tokens per warp, one matrix entry per lane. Every shuffle stays
        # inside its 16-lane token group; sums retain ascending axis order.
        # Whole warps always enter (full-mask shuffles); lanes of a missing
        # token compute on a clamped token and store nothing (valid=False).
        tid, _, _ = cute.arch.thread_idx()
        entry = tid % 16
        origin = (tid % 32) // 16 * 16
        col = entry % 4
        rms = self.inv_norm(totals[offset + 24], 4 * self.h, self.rms_eps)
        pre_logit = self.arith(
            self.arith(self.arith(totals[offset + col], rms, "mul"), small[0], "mul"),
            small[3 + col],
            "add",
        )
        pre = self.arith(self.sigmoid(pre_logit), self.hc_eps, "add")
        if (entry < 4) & valid:
            gains[goffset + col] = pre
            if write:
                post_logit = self.arith(
                    self.arith(self.arith(totals[offset + 4 + col], rms, "mul"), small[1], "mul"),
                    small[7 + col],
                    "add",
                )
                _store(
                    po + Int64((token * 4 + col) * 4),
                    Float32(self.arith(self.post_mult, self.sigmoid(post_logit), "mul")),
                )
        # Gather before divergence so all source lanes participate.
        pres = cute.make_rmem_tensor((4,), self.dtype)
        for k in cutlass.range_constexpr(4):
            pres[k] = cute.arch.shuffle_sync(pre, origin + k)
        if (entry == 0) & valid:
            ss = self.dtype(0.0)
            for j in cutlass.range_constexpr(4):
                for k in cutlass.range_constexpr(j, 4):
                    index = 25 + j * 4 - j * (j - 1) // 2 + k - j
                    factor = self.arith(pres[j], pres[k], "mul")
                    if const_expr(j != k):
                        factor = self.arith(factor, 2.0, "mul")
                    ss = _fma(factor, totals[offset + index], ss)
            gains[goffset + 4] = self.inv_norm(ss, self.h, self.norm_eps)
        return rms

    @cute.jit
    def sinkhorn_lane16(self, totals, offset, small, rms, co, token, write, valid):
        # The caller enters with a whole warp, including masked token tails.
        tid, _, _ = cute.arch.thread_idx()
        entry = tid % 16
        logit = self.arith(
            self.arith(self.arith(totals[offset + 8 + entry], rms, "mul"), small[2], "mul"),
            small[11 + entry],
            "add",
        )
        # Row (xor 1, 2) and column (xor 4, 8) butterflies: commutative pairs, so
        # every lane of a row/column gets the same bits.
        maximum = cutlass.max(logit, cute.arch.shuffle_sync_bfly(logit, 1))
        maximum = cutlass.max(maximum, cute.arch.shuffle_sync_bfly(maximum, 2))
        mat = cute.exp(self.arith(logit, maximum, "sub"), fastmath=False)
        denom = self.row_sum(mat)
        mat = self.arith(self.arith(mat, denom, "div"), self.hc_eps, "add")
        for iteration in range(self.iters):
            if iteration > 0:
                denom = self.row_sum(mat)
                mat = self.arith(mat, self.arith(denom, self.hc_eps, "add"), "div")
            denom = self.col_sum(mat)
            mat = self.arith(mat, self.arith(denom, self.hc_eps, "add"), "div")
        if valid:
            if write:
                _store(co + Int64((token * 16 + entry) * 4), Float32(mat))

    @cute.jit
    def gains(self, totals, offset, small, gains, goffset, po, token, write):
        # A single thread owns the pre/post entries and Gram quadratic form.
        rms = self.arith(
            1.0,
            cute.sqrt(
                self.arith(self.arith(totals[offset + 24], 4 * self.h, "div"), self.rms_eps, "add"),
                fastmath=False,
            ),
            "div",
        )
        logits = cute.make_rmem_tensor((8,), self.dtype)
        pre = cute.make_rmem_tensor((4,), self.dtype)
        for m in cutlass.range_constexpr(8):
            mix = self.arith(totals[offset + m], rms, "mul")
            si = min(m // 4, 2)
            logits[m] = self.arith(self.arith(mix, small[si], "mul"), small[3 + m], "add")
        for j in cutlass.range_constexpr(4):
            pre[j] = self.arith(self.sigmoid(logits[j]), self.hc_eps, "add")
            gains[goffset + j] = pre[j]
            if write:
                _store(
                    po + Int64((token * 4 + j) * 4),
                    Float32(self.arith(self.post_mult, self.sigmoid(logits[4 + j]), "mul")),
                )
        ss = self.dtype(0.0)
        for j in cutlass.range_constexpr(4):
            for k in cutlass.range_constexpr(j, 4):
                index = 25 + j * 4 - j * (j - 1) // 2 + k - j
                factor = self.arith(pre[j], pre[k], "mul")
                if const_expr(j != k):
                    factor = self.arith(factor, 2.0, "mul")
                ss = _fma(factor, totals[offset + index], ss)
        rstd = self.arith(
            1.0,
            cute.sqrt(
                self.arith(self.arith(ss, self.h, "div"), self.norm_eps, "add"), fastmath=False
            ),
            "div",
        )
        gains[goffset + 4] = rstd
        return rms

    @cute.jit
    def sinkhorn(self, totals, offset, small, rms, co, token, write):
        # Preserve the serial softmax and Sinkhorn addition order exactly.
        logits = cute.make_rmem_tensor((16,), self.dtype)
        for m in cutlass.range_constexpr(16):
            mix = self.arith(totals[offset + 8 + m], rms, "mul")
            logits[m] = self.arith(self.arith(mix, small[2], "mul"), small[11 + m], "add")
        mat = cute.make_rmem_tensor((4, 4), self.dtype)
        for i in cutlass.range_constexpr(4):
            maximum = logits[i * 4]
            for j in cutlass.range_constexpr(1, 4):
                maximum = cutlass.max(maximum, logits[i * 4 + j])
            denom = self.dtype(0.0)
            for j in cutlass.range_constexpr(4):
                mat[i, j] = cute.exp(self.arith(logits[i * 4 + j], maximum, "sub"), fastmath=False)
                denom = self.arith(denom, mat[i, j], "add")
            for j in cutlass.range_constexpr(4):
                mat[i, j] = self.arith(self.arith(mat[i, j], denom, "div"), self.hc_eps, "add")
        for iteration in range(self.iters):
            if iteration > 0:
                for i in cutlass.range_constexpr(4):
                    denom = self.dtype(0.0)
                    for j in cutlass.range_constexpr(4):
                        denom = self.arith(denom, mat[i, j], "add")
                    for j in cutlass.range_constexpr(4):
                        mat[i, j] = self.arith(
                            mat[i, j], self.arith(denom, self.hc_eps, "add"), "div"
                        )
            for j in cutlass.range_constexpr(4):
                denom = self.dtype(0.0)
                for i in cutlass.range_constexpr(4):
                    denom = self.arith(denom, mat[i, j], "add")
                for i in cutlass.range_constexpr(4):
                    mat[i, j] = self.arith(mat[i, j], self.arith(denom, self.hc_eps, "add"), "div")
        if write:
            for i in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(4):
                    _store(co + Int64((token * 16 + i * 4 + j) * 4), Float32(mat[i, j]))

    @cute.jit
    def coefficients_lane16(
        self, totals, offset, small, gains, goffset, po, co, token, write, valid
    ):
        rms = self.gains_lane16(totals, offset, small, gains, goffset, po, token, write, valid)
        self.sinkhorn_lane16(totals, offset, small, rms, co, token, write, valid)

    @cute.jit
    def coefficients(self, totals, offset, small, gains, goffset, po, co, token, write):
        rms = self.gains(totals, offset, small, gains, goffset, po, token, write)
        self.sinkhorn(totals, offset, small, rms, co, token, write)

    @cute.jit
    def decode_reduce_values(self, parts, totals, tokens, count: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        # One owner per value retains its ascending G chain. Load both values
        # before either chain so independent additions can overlap.
        for first in range(tid, tokens * 35, 128 * count):
            loaded = cute.make_rmem_tensor((count, self.pitch), self.dtype)
            loaded.fill(0.0)
            for g in cutlass.range_constexpr(0, self.pitch, 16 // self.bytes):
                if const_expr(g < self.grid):
                    for slot in cutlass.range_constexpr(count):
                        v = first + slot * 128
                        if v < tokens * 35:
                            addr = parts + Int64((v * self.pitch + g) * self.bytes)
                            # L2-only loads follow the acquire spin/RMW and CTA barrier.
                            raw = cute.arch.load(
                                _ptr(Int32, addr, 16),
                                ir.VectorType.get([4], Int32.mlir_type),
                                cop="cg",
                            )
                            vec = cute.TensorSSA(raw, (4,), Int32).bitcast(self.dtype)
                            for k in cutlass.range_constexpr(16 // self.bytes):
                                loaded[slot, g + k] = vec[k]
            values = cute.make_rmem_tensor((count,), self.dtype)
            values.fill(0.0)
            for g in cutlass.range_constexpr(self.grid):
                for slot in cutlass.range_constexpr(count):
                    values[slot] = self.add(values[slot], loaded[slot, g])
            for slot in cutlass.range_constexpr(count):
                v = first + slot * 128
                if v < tokens * 35:
                    totals[v] = values[slot]

    @cute.jit
    def decode_reduce(self, parts, totals, tokens):
        """Identical fixed-order totals for either exchange."""
        self.trace_push("mhc.partial_read")
        if tokens * 35 > 128:
            self.decode_reduce_values(parts, totals, tokens, 2)
        else:
            self.decode_reduce_values(parts, totals, tokens, 1)
        cute.arch.sync_threads()
        self.trace_pop()

    @cute.jit
    def decode_projection(self, weights, cache, parts, tokens, piece):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        j, group = lane % 4, lane // 4
        self.trace_push("mhc.projection")
        # Each warp keeps six rows (12 vectors = 48 FP32 registers) per piece.
        # Plain vector loads can be hoisted/reused across all token blocks.
        fv = cute.make_rmem_tensor((6, 8), Float32)
        for row in cutlass.range_constexpr(6):
            for half in cutlass.range_constexpr(2):
                fnptr = weights.iterator + (warp + row * 4) * 256 + half * 128 + group * 16 + j * 4
                raw = cute.arch.load(
                    cute.recast_ptr(fnptr, dtype=Int32), ir.VectorType.get([4], Int32.mlir_type)
                )
                vec = cute.TensorSSA(raw, (4,), Int32).bitcast(Float32)
                for k in cutlass.range_constexpr(4):
                    fv[row, half * 4 + k] = vec[k]
        for t0 in range(0, tokens, 4):
            sums = cute.make_rmem_tensor((24,), Float32)
            comp = cute.make_rmem_tensor((24,), Float32)
            sums.fill(0.0)
            comp.fill(0.0)
            for token in cutlass.range_constexpr(4):
                rv = cute.make_rmem_tensor((8,), Float32)
                rv.fill(0.0)
                if t0 + token < tokens:
                    for half in cutlass.range_constexpr(2):
                        ptr = (
                            cache.iterator
                            + ((t0 + token) * 4 + j) * self.size
                            + piece * 64
                            + group * 8
                            + half * 4
                        )
                        raw = cute.arch.load(
                            cute.recast_ptr(ptr, dtype=Int32),
                            ir.VectorType.get([4], Int32.mlir_type),
                        )
                        vec = cute.TensorSSA(raw, (4,), Int32).bitcast(Float32)
                        for k in cutlass.range_constexpr(4):
                            rv[half * 4 + k] = vec[k]
                    for row in cutlass.range_constexpr(6):
                        for half in cutlass.range_constexpr(2):
                            for k in cutlass.range_constexpr(4):
                                self.add_product(
                                    sums,
                                    comp,
                                    row * 4 + token,
                                    fv[row, half * 4 + k],
                                    rv[half * 4 + k],
                                )
            # Token masks never enclose shuffles: all 32 lanes reduce every slot.
            for row in cutlass.range_constexpr(6):
                for token in cutlass.range_constexpr(4):
                    value = self.lane_total(sums, comp, row * 4 + token)
                    for mask in (4, 8, 16, 1, 2):
                        value = self.add(value, cute.arch.shuffle_sync_bfly(value, mask))
                    if (lane == 0) & (t0 + token < tokens):
                        m = warp + row * 4
                        ptr = _ptr(
                            self.dtype,
                            parts
                            + Int64((((t0 + token) * 35 + m) * self.pitch + bid) * self.bytes),
                            self.bytes,
                        )
                        if piece > 0:
                            value = self.add(cute.arch.load(ptr, self.dtype), value)
                        cute.arch.store(ptr, value)
        self.trace_pop()

    @cute.jit
    def decode_gains(self, totals, small, gains, rms, po, tokens, write):
        tid, _, _ = cute.arch.thread_idx()
        if const_expr(self.chain == "lane16"):
            for first in range(tid // 32 * 2, tokens, 8):
                t = first + (tid % 32) // 16
                tc = cutlass.min(t, tokens - 1)
                value = self.gains_lane16(
                    totals, tc * 35, small, gains, tc * 5, po, tc, write, t < tokens
                )
                if (tid % 16 == 0) & (t < tokens):
                    rms[tc] = value
        else:
            for t in range(tid, tokens, 128):
                rms[t] = self.gains(totals, t * 35, small, gains, t * 5, po, t, write)
        cute.arch.sync_threads()

    @cute.jit
    def decode_sinkhorn(self, totals, small, rms, co, tokens, write):
        tid, _, _ = cute.arch.thread_idx()
        if const_expr(self.chain == "lane16"):
            for first in range(tid // 32 * 2, tokens, 8):
                t = first + (tid % 32) // 16
                tc = cutlass.min(t, tokens - 1)
                self.sinkhorn_lane16(totals, tc * 35, small, rms[tc], co, tc, write, t < tokens)
        else:
            for t in range(tid, tokens, 128):
                self.sinkhorn(totals, t * 35, small, rms[t], co, t, write)
        cute.arch.sync_threads()

    @cute.kernel
    def decode_kernel(
        self,
        x: Int64,
        r: Int64,
        p: Int64,
        c: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        w: Int64,
        ro: Int64,
        po: Int64,
        co: Int64,
        out: Int64,
        parts: Int64,
        counter: Int64,
        tokens: Int32,
        words: cute.Tensor,
        host: Int64,
        stride: Int64,
        rank: Int32,
        ysum: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        j, group = lane % 4, lane // 4
        size = self.size
        smem = cutlass.utils.SmemAllocator()
        weights = smem.allocate_tensor(Float32, cute.make_layout(96 * 64), 16)
        # All fn readers finish before this shared allocation becomes totals.
        # Output only reads cache/norm/gains (or recomputes in registers), and
        # never stages fn again. Keep totals and small live through late Sinkhorn;
        # neither may alias output scratch, including the recompute capacities.
        if const_expr(self.recompute):
            # Small-prefill tokens reuse fn without retaining all T residuals.
            # Totals outgrow the fn allocation at the largest capacity.
            totals = smem.allocate_tensor(self.dtype, cute.make_layout(self.cap * 35), 16)
        else:
            totals = cute.make_tensor(
                cute.recast_ptr(weights.iterator, dtype=self.dtype), cute.make_layout(self.cap * 35)
            )
            cache = smem.allocate_tensor(Float32, cute.make_layout(self.cap * 4 * size), 16)
        norm = smem.allocate_tensor(Float32, cute.make_layout(size), 16)
        small = smem.allocate_tensor(Float32, cute.make_layout(27), 16)
        gains = smem.allocate_tensor(self.dtype, cute.make_layout(self.cap * 5), 16)
        rms = smem.allocate_tensor(self.dtype, cute.make_layout(self.cap), 16)
        if const_expr(self.exchange == "last"):
            control = smem.allocate_tensor(Int64, cute.make_layout(2), 16)
        # pitch * bytes is a multiple of 16 in both modes. The host reserves
        # enough trailing space for 256 * 5 FP64 gains, including FP32 exit.
        gains_base = parts + Int64(self.cap * 35 * self.pitch * self.bytes)
        h0 = bid * size
        self.trace_push("mhc.prologue")
        self.stage_fn(fn, weights, h0, 64, tid, 128)
        self.stage_small(scale, base, small, tid)
        if tid < size // 8:
            nv = cute.make_rmem_tensor((8,), Float32)
            nv.fill(0.0)
            if h0 + tid * 8 < self.h:
                nv.store(_load8(w + Int64((h0 + tid * 8) * 2)))
            for k in cutlass.range_constexpr(8):
                norm[tid * 8 + k] = nv[k]
        self.trace_pop()
        self.trace_push("mhc.grid_wait")
        cute.arch.griddepcontrol_wait()
        self.trace_pop()
        # A successor's wait blocks until this entire grid completes and flushes,
        # even when triggered early. It may stage immutable fn while we compute,
        # but cannot touch activations/workspace/counter yet. The trigger releases
        # only after every CTA has executed it, hence all exchange CTAs are already
        # resident; together with the prepare-time occupancy bound this avoids
        # co-residency deadlock. No counter reset is needed between graph replays.
        # A later decode trigger keeps the successor's fn prefetch out of DRAM
        # while this grid's own activation loads are in flight.
        if const_expr(self.trigger == "1"):
            cute.arch.griddepcontrol_launch_dependents()
        if const_expr(self.allreduce):
            # B1 arrival requires all CTAs resident; the decode plan guarantees it.
            ar_control = smem.allocate_tensor(Int64, cute.make_layout(2), 8)
            word_base = words.iterator.toint()
            seq_ptr = _ptr(Int64, word_base, 8)
            if tid == 0:
                put(ar_control, 0, cute.arch.load(seq_ptr, Int64, sem="acquire", scope="gpu"))
                put(ar_control, 1, Int64(0))
            cute.arch.sync_threads()
            seq = ar_control[0]
            slot = seq & Int64(1)
            send = host + slot * stride
            recv = host + (slot + 2) * stride
            flags = host + 4 * stride
            doorbell = _ptr(Int64, flags + slot * 128, 8)
            rflag = _ptr(Int64, flags + (slot + 2) * 128, 8)
            abort = _ptr(Int64, flags + 512, 8)
            self.trace_push("mhc.ar_send")
            for v in range(tid, tokens * (size // 8), 128):
                t, h = v // (size // 8), h0 + (v % (size // 8)) * 8
                if h < self.h:
                    off = Int64((t * self.h + h) * 2)
                    store_raw(send + off, load_raw(x + off), cop="wt")
            cute.arch.fence_acq_rel_sys()
            cute.arch.sync_threads()
            if tid == 0:
                arrive = _ptr(Int64, word_base + 8, 8)
                ticket = cute.arch.atomic_add(arrive, Int64(1), sem="acq_rel", scope="gpu")
                if ticket == self.grid - 1:
                    cute.arch.store(arrive, Int64(0), sem="release", scope="gpu")
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.store(
                        doorbell,
                        (seq << 24) | Int64(tokens * self.h * 2),
                        sem="release",
                        scope="sys",
                    )
            self.trace_pop()
            self.trace_push("mhc.ar_wait")
            if const_expr(self.ar_prefetch):
                for v in range(tid, tokens * 4 * (size // 8), 128):
                    row, h = v // (size // 8), h0 + (v % (size // 8)) * 8
                    if h < self.h:
                        cute.arch.inline_ptx(
                            "prefetch.global.L2 [{$r0}];",
                            read_only_args=[r + Int64((row * self.h + h) * 2)],
                        )
            if tid == 0:
                seen = cute.arch.load(rflag, Int64, sem="acquire", scope="sys")
                failed = cute.arch.load(abort, Int64, sem="acquire", scope="sys")
                while (seen != seq) & (failed == 0):
                    seen = cute.arch.load(rflag, Int64, sem="acquire", scope="sys")
                    failed = cute.arch.load(abort, Int64, sem="acquire", scope="sys")
                put(ar_control, 1, failed)
                if failed != 0:
                    cute.arch.atomic_exch(
                        _ptr(Int64, word_base + 24, 8), Int64(1), sem="relaxed", scope="gpu"
                    )
            cute.arch.sync_threads()
            self.trace_pop()
            self.trace_push("mhc.ar_sum")
            for v in range(tid, tokens * (size // 8), 128):
                t, h = v // (size // 8), h0 + (v % (size // 8)) * 8
                if h < self.h:
                    off = Int64((t * self.h + h) * 2)
                    local = load_raw(x + off)
                    if ar_control[1] == 0:
                        a = cute.TensorSSA(local, (4,), Int32).bitcast(BFloat16).to(Float32)
                        b = cute.TensorSSA(load_raw(recv + off, cop="cv"), (4,), Int32)
                        peer = b.bitcast(BFloat16).to(Float32)
                        rank0, rank1 = a, peer
                        if rank != 0:
                            rank0, rank1 = peer, a
                        sums = cute.make_rmem_tensor((8,), Float32)
                        for k in cutlass.range_constexpr(8):
                            put(sums, k, _add_rn(rank0[k], rank1[k]))
                        store_raw(ysum + off, sums.load().to(BFloat16).bitcast(Int32))
                    else:
                        store_raw(ysum + off, local)
            # All recv reads finish before sequence advance. Each CTA only reads
            # its own ysum columns below, including the recompute pass.
            cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                finish = _ptr(Int64, word_base + 16, 8)
                ticket = cute.arch.atomic_add(finish, Int64(1), sem="acq_rel", scope="gpu")
                if ticket == self.grid - 1:
                    cute.arch.store(finish, Int64(0), sem="release", scope="gpu")
                    cute.arch.store(seq_ptr, seq + 1, sem="release", scope="gpu")
            x = ysum
            self.trace_pop()
        for piece in range(size // 64):
            if piece > 0:
                # Keep second-piece staging addresses out of the projection's
                # live registers. The first piece still stages before PDL wait.
                self.stage_fn(
                    fn, weights, h0 + piece * 64, 64, tid, 128, retain_loop=self.split_proj
                )
            self.trace_push("mhc.async_wait")
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self.trace_pop()
            for t in range(warp, tokens, 4):
                self.trace_push("mhc.residual_partials")
                h = h0 + piece * 64 + group * 8
                rv = cute.make_rmem_tensor((8,), Float32)
                rv.fill(0.0)
                if h < self.h:
                    loaded = self.residual(x, r, p, c, ro, t, h, j)
                    rv.store(loaded.load())
                self.trace_pop()
                self.trace_push("mhc.projection")
                sums = cute.make_rmem_tensor((29,), Float32)
                comp = cute.make_rmem_tensor((29,), Float32)
                sums.fill(0.0)
                comp.fill(0.0)
                if const_expr(not self.recompute):
                    for k in cutlass.range_constexpr(8):
                        cache[(t * 4 + j) * size + piece * 64 + group * 8 + k] = rv[k]
                if const_expr(not self.split_proj):
                    for m in cutlass.range_constexpr(24):
                        # Two aligned, bank-conflict-free 16-byte shared loads per
                        # lane: each half spans consecutive vectors across the warp.
                        for half in cutlass.range_constexpr(2):
                            fnptr = weights.iterator + m * 256 + half * 128 + group * 16 + j * 4
                            # Constrain load hoisting in the last-arriver variant too:
                            # otherwise the projection can spill before the exchange.
                            if const_expr(self.recompute or self.exchange == "last"):
                                fv = _load_shared4(fnptr)
                            else:
                                raw = cute.arch.load(
                                    cute.recast_ptr(fnptr, dtype=Int32),
                                    ir.VectorType.get([4], Int32.mlir_type),
                                )
                                fv = cute.TensorSSA(raw, (4,), Int32).bitcast(Float32)
                            for k in cutlass.range_constexpr(4):
                                self.add_product(sums, comp, m, fv[k], rv[half * 4 + k])
                self.trace_pop()
                self.trace_push("mhc.gram")
                for k in cutlass.range_constexpr(8):
                    self.add_product(sums, comp, 24, rv[k], rv[k])
                    for stream_index in cutlass.range_constexpr(4):
                        other = cute.arch.shuffle_sync(rv[k], group * 4 + stream_index)
                        self.add_product(sums, comp, 25 + stream_index, rv[k], other)
                self.trace_pop()
                self.trace_push("mhc.partial_stores")
                for m in cutlass.range_constexpr(29):
                    if const_expr(not self.split_proj or m >= 24):
                        value = self.lane_total(sums, comp, m)
                        for mask in (4, 8, 16):
                            value = self.add(value, cute.arch.shuffle_sync_bfly(value, mask))
                        if const_expr(m < 25):
                            for mask in (1, 2):
                                value = self.add(value, cute.arch.shuffle_sync_bfly(value, mask))
                            if lane == 0:
                                ptr = _ptr(
                                    self.dtype,
                                    parts
                                    + Int64(
                                        (t * 35 + m) * self.pitch * self.bytes + bid * self.bytes
                                    ),
                                    self.bytes,
                                )
                                if piece > 0:
                                    value = self.add(cute.arch.load(ptr, self.dtype), value)
                                cute.arch.store(ptr, value)
                        else:
                            if group == 0 and j <= m - 25:
                                index = 25 + j * 4 - j * (j - 1) // 2 + m - 25 - j
                                ptr = _ptr(
                                    self.dtype,
                                    parts
                                    + Int64(((t * 35 + index) * self.pitch + bid) * self.bytes),
                                    self.bytes,
                                )
                                if piece > 0:
                                    value = self.add(cute.arch.load(ptr, self.dtype), value)
                                cute.arch.store(ptr, value)
                self.trace_pop()
            cute.arch.sync_threads()
            if const_expr(self.trigger == "residual"):
                if piece == 0:
                    cute.arch.griddepcontrol_launch_dependents()
            if const_expr(self.split_proj):
                self.decode_projection(weights, cache, parts, tokens, piece)
                cute.arch.sync_threads()
        if const_expr(self.trigger == "exchange"):
            cute.arch.griddepcontrol_launch_dependents()
        self.trace_push("mhc.padding")
        if bid == self.grid - 1:
            for v in range(tid, tokens * 35, 128):
                for g in cutlass.range_constexpr(self.grid, self.pitch):
                    cute.arch.store(
                        _ptr(
                            self.dtype, parts + Int64((v * self.pitch + g) * self.bytes), self.bytes
                        ),
                        self.dtype(0.0),
                    )
        self.trace_pop()
        # Design reference: CUTLASS include/cutlass/barrier.h GenericBarrier
        # arrive_inc / wait_eq (https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/barrier.h).
        # CTA barrier -> one releasing thread; acquiring thread -> CTA barrier.
        # Our GPU acq_rel RMW/release store cumulatively publishes pre-barrier
        # writes from every CTA thread; the optional all-thread fence is redundant.
        if const_expr(self.exchange == "last"):
            self.trace_push("mhc.arrival")
            if const_expr(self.fence == "all"):
                cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                old = cute.arch.atomic_add(
                    _ptr(Int64, counter, 8), Int64(1), sem="acq_rel", scope="gpu"
                )
                control[0] = old // self.grid + 1
                control[1] = Int64(old % self.grid == self.grid - 1)
            cute.arch.sync_threads()
            self.trace_pop()
            epoch = control[0]
            # This predicate is CTA-uniform: every lane reaches each shuffle
            # in the reduction and lane16 coefficients, including odd T tails.
            if control[1] != 0:
                self.trace_push("mhc.last_gains")
                self.decode_reduce(parts, totals, tokens)
                self.decode_gains(totals, small, gains, rms, po, tokens, True)
                if const_expr(self.publish == "late"):
                    self.trace_push("mhc.last_sinkhorn")
                    self.decode_sinkhorn(totals, small, rms, co, tokens, True)
                    self.trace_pop()
                for v in range(tid, tokens * 5, 128):
                    cute.arch.store(
                        _ptr(self.dtype, gains_base + Int64(v * self.bytes), self.bytes), gains[v]
                    )
                if const_expr(self.fence == "all"):
                    cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_threads()
                if tid == 0:
                    cute.arch.store(_ptr(Int64, counter + 8, 8), epoch, sem="release", scope="gpu")
                self.trace_pop()
            else:
                self.trace_push("mhc.flag_wait")
                if tid == 0:
                    flag = _ptr(Int64, counter + 8, 8)
                    seen = cute.arch.load(flag, Int64, sem="acquire", scope="gpu")
                    while seen < epoch:
                        seen = cute.arch.load(flag, Int64, sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                for v in range(tid, tokens * 5, 128):
                    gains[v] = cute.arch.load(
                        _ptr(self.dtype, gains_base + Int64(v * self.bytes), self.bytes),
                        self.dtype,
                        cop="cg",
                    )
                cute.arch.sync_threads()
                self.trace_pop()
        else:
            self.trace_push("mhc.exchange")
            if const_expr(self.fence == "all"):
                cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            if tid == 0:
                arrival = _ptr(Int64, counter, 8)
                old = cute.arch.atomic_add(arrival, Int64(1), sem="acq_rel", scope="gpu")
                target = (old // self.grid + 1) * self.grid
                seen = cute.arch.load(arrival, Int64, sem="acquire", scope="gpu")
                while seen < target:
                    seen = cute.arch.load(arrival, Int64, sem="acquire", scope="gpu")
            cute.arch.sync_threads()
            self.trace_pop()
            self.decode_reduce(parts, totals, tokens)
            self.trace_push("mhc.coefficients")
            self.decode_gains(totals, small, gains, rms, po, tokens, bid == 0)
            self.decode_sinkhorn(totals, small, rms, co, tokens, bid == 0)
            self.trace_pop()
        self.trace_push("mhc.output")
        for v in range(tid, tokens * (size // 8), 128):
            t, col = v // (size // 8), (v % (size // 8)) * 8
            if h0 + col < self.h:
                values = cute.make_rmem_tensor((8,), Float32)
                if const_expr(self.recompute):
                    yv = cute.make_rmem_tensor((8,), Float32)
                    yv.fill(0.0)
                    for stream_index in cutlass.range_constexpr(4):
                        rv = self.residual(x, r, p, c, ro, t, h0 + col, stream_index, False)
                        for k in cutlass.range_constexpr(8):
                            yv[k] = _fma(gains[t * 5 + stream_index], rv[k], yv[k])
                    for k in cutlass.range_constexpr(8):
                        values[k] = _fp(_fp(yv[k], gains[t * 5 + 4], "mul"), norm[col + k], "mul")
                else:
                    for k in cutlass.range_constexpr(8):
                        y = self.dtype(0.0)
                        for stream_index in cutlass.range_constexpr(4):
                            a = gains[t * 5 + stream_index]
                            b = cache[(t * 4 + stream_index) * size + col + k]
                            y = _fma(a, b, y)
                        value = self.arith(
                            self.arith(y, gains[t * 5 + 4], "mul"), norm[col + k], "mul"
                        )
                        values[k] = self.round_output(value)
                _store8(out + Int64((t * self.h + h0 + col) * 2), values.load())

        self.trace_pop()
        if const_expr(self.exchange == "last" and self.publish == "early"):
            if control[1] != 0:
                # Reconverge after the output loop before full-mask shuffles.
                cute.arch.sync_threads()
                self.trace_push("mhc.last_sinkhorn")
                self.decode_sinkhorn(totals, small, rms, co, tokens, True)
                self.trace_pop()
        if const_expr(not self.early):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def prefill_kernel(
        self,
        x: Int64,
        r: Int64,
        p: Int64,
        c: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        w: Int64,
        ro: Int64,
        po: Int64,
        co: Int64,
        out: Int64,
        tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        chunk_size = 64 if const_expr(self.tt <= 4) else 32
        width = 256 // self.tt
        shards = max(1, width // 64)
        owners = width // shards
        smem = cutlass.utils.SmemAllocator()
        weights = smem.allocate_tensor(Float32, cute.make_layout(2 * 96 * chunk_size), 16)
        cache = smem.allocate_tensor(Float32, cute.make_layout(self.tt * 4 * chunk_size), 16)
        small = smem.allocate_tensor(Float32, cute.make_layout(27), 16)
        totals = smem.allocate_tensor(self.dtype, cute.make_layout(self.tt * 35), 16)
        if const_expr(shards > 1):
            shard_totals = smem.allocate_tensor(
                self.dtype, cute.make_layout(self.tt * shards * 35), 16
            )
        gains = smem.allocate_tensor(self.dtype, cute.make_layout(self.tt * 5), 16)
        self.trace_push("mhc.prologue")
        self.stage_fn(fn, weights, Int32(0), chunk_size, tid, 256)
        self.stage_small(scale, base, small, tid)
        self.trace_pop()
        self.trace_push("mhc.grid_wait")
        cute.arch.griddepcontrol_wait()
        self.trace_pop()
        # Same early-trigger contract as decode: successor activations still wait
        # for completion; only immutable weights may be prefetched concurrently.
        if const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # Small tiles split columns among independent projection owners, then
        # reduce their compensated totals in a fixed order once per token.
        slots = (35 + owners - 1) // owners
        local_t, owner = tid // width, tid % owners
        shard = (tid % width) // owners
        sums = cute.make_rmem_tensor((slots,), Float32)
        comp = cute.make_rmem_tensor((slots,), Float32)
        sums.fill(0.0)
        comp.fill(0.0)
        for chunk in range((self.h + chunk_size - 1) // chunk_size):
            self.trace_push("mhc.async_wait")
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self.trace_pop()
            if chunk + 1 < (self.h + chunk_size - 1) // chunk_size:
                self.stage_fn(
                    fn,
                    weights,
                    (chunk + 1) * chunk_size,
                    chunk_size,
                    tid,
                    256,
                    ((chunk + 1) % 2) * 96 * chunk_size,
                )
            self.trace_push("mhc.residual_partials")
            for v in range(tid, self.tt * (chunk_size // 2), 256):
                lt = v // (chunk_size // 2)
                j, col = (v % (chunk_size // 2)) // (chunk_size // 8), (v % (chunk_size // 8)) * 8
                t, h = bid * self.tt + lt, chunk * chunk_size + col
                rv = cute.make_rmem_tensor((8,), Float32)
                rv.fill(0.0)
                if t < tokens and h < self.h:
                    loaded = self.residual(x, r, p, c, ro, t, h, j)
                    rv.store(loaded.load())
                for k in cutlass.range_constexpr(8):
                    cache[(lt * 4 + j) * chunk_size + col + k] = rv[k]
            cute.arch.sync_threads()
            for slot in cutlass.range_constexpr(slots):
                m = owner + slot * owners
                if m < 24:
                    for j in cutlass.range_constexpr(4):
                        for vector in range(shard, chunk_size // 4, shards):
                            fnptr = weights.iterator + (chunk % 2) * 96 * chunk_size
                            fnptr += ((j * (chunk_size // 4) + vector) * 24 + m) * 4
                            raw = cute.arch.load(
                                cute.recast_ptr(fnptr, dtype=Int32),
                                ir.VectorType.get([4], Int32.mlir_type),
                            )
                            fv = cute.TensorSSA(raw, (4,), Int32).bitcast(Float32)
                            for k in cutlass.range_constexpr(4):
                                b = cache[(local_t * 4 + j) * chunk_size + vector * 4 + k]
                                self.add_product(sums, comp, slot, fv[k], b)
                elif m == 24:
                    for j in cutlass.range_constexpr(4):
                        for col in range(shard, chunk_size, shards):
                            a = cache[(local_t * 4 + j) * chunk_size + col]
                            self.add_product(sums, comp, slot, a, a)
                elif m < 35:
                    # Map the ten upper-triangle Gram entries without a lookup.
                    gj = Int32(0)
                    gk = m - 25
                    if m >= 29:
                        gj, gk = Int32(1), m - 28
                    if m >= 32:
                        gj, gk = Int32(2), m - 30
                    if m == 34:
                        gj, gk = Int32(3), Int32(3)
                    for col in range(shard, chunk_size, shards):
                        a = cache[(local_t * 4 + gj) * chunk_size + col]
                        b = cache[(local_t * 4 + gk) * chunk_size + col]
                        self.add_product(sums, comp, slot, a, b)
            cute.arch.sync_threads()
            self.trace_pop()
        self.trace_push("mhc.partial_stores")
        for slot in cutlass.range_constexpr(slots):
            m = owner + slot * owners
            if m < 35:
                if const_expr(shards > 1):
                    shard_totals[(local_t * shards + shard) * 35 + m] = self.lane_total(
                        sums, comp, slot
                    )
                else:
                    totals[local_t * 35 + m] = self.lane_total(sums, comp, slot)
        cute.arch.sync_threads()
        if const_expr(shards > 1):
            for v in range(tid, self.tt * 35, 256):
                value = self.dtype(0.0)
                for part in cutlass.range_constexpr(shards):
                    value = self.add(value, shard_totals[((v // 35) * shards + part) * 35 + v % 35])
                totals[v] = value
            cute.arch.sync_threads()
        self.trace_pop()
        self.trace_push("mhc.coefficients")
        if const_expr(self.chain == "lane16"):
            for first in range(tid // 32 * 2, self.tt, 16):
                lt = first + (tid % 32) // 16
                lc = cutlass.min(lt, self.tt - 1)
                self.coefficients_lane16(
                    totals,
                    lc * 35,
                    small,
                    gains,
                    lc * 5,
                    po,
                    co,
                    bid * self.tt + lc,
                    True,
                    (lt < self.tt) & (bid * self.tt + lt < tokens),
                )
        else:
            if tid < self.tt and bid * self.tt + tid < tokens:
                self.coefficients(
                    totals, tid * 35, small, gains, tid * 5, po, co, bid * self.tt + tid, True
                )
        cute.arch.sync_threads()
        self.trace_pop()
        self.trace_push("mhc.output")
        # Recompute the identical unrounded high part from L2-hot inputs.
        # A constexpr disables duplicate residual stores in this second pass.
        for v in range(tid, self.tt * (self.h // 8), 256):
            lt, h = v // (self.h // 8), (v % (self.h // 8)) * 8
            t = bid * self.tt + lt
            if t < tokens:
                y = cute.make_rmem_tensor((8,), self.dtype)
                y.fill(0.0)
                for j in cutlass.range_constexpr(4):
                    rv = self.residual(x, r, p, c, ro, t, h, j, False)
                    for k in cutlass.range_constexpr(8):
                        y[k] = _fma(gains[lt * 5 + j], rv[k], y[k])
                nv = _load8(w + Int64(h * 2))
                for k in cutlass.range_constexpr(8):
                    y[k] = self.arith(self.arith(y[k], gains[lt * 5 + 4], "mul"), nv[k], "mul")
                values = cute.make_rmem_tensor((8,), Float32)
                for k in cutlass.range_constexpr(8):
                    values[k] = self.round_output(y[k])
                _store8(out + Int64((t * self.h + h) * 2), values.load())

        self.trace_pop()
        if const_expr(not self.early):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def post_kernel(self, x: Int64, r: Int64, p: Int64, c: Int64, ro: Int64, tokens: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        v = bid * 128 + tid
        j, col = v % 4, v // 4
        t, h = col // (self.h // 8), (col % (self.h // 8)) * 8
        cute.arch.griddepcontrol_wait()
        if const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        if t < tokens:
            self.residual(x, r, p, c, ro, t, h, j)
        if const_expr(not self.early):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


def compile_kernel(
    cfg, plan, mode, decode, *, token_tile=16, cluster=False, allreduce=False, options=""
):
    """Compile a token-capacity bucket; runtime T can be smaller than capacity."""
    if allreduce and (
        mode != "exit" or not decode or cluster or plan[2] not in (4, 16, 32, 64, 128, 256)
    ):
        raise ValueError("allreduce requires exit mode and a grid decode bucket")
    if token_tile not in (1, 2, 4, 8, 16, 32):
        raise ValueError("token_tile must be 1, 2, 4, 8, 16 or 32")
    kernel_type = Kernel
    if cluster:
        from .cluster import ClusterKernel

        if mode != "exit" or plan[1] != 8 or plan[2] not in (1, 2, 4, 8):
            raise ValueError("cluster requires exit mode, eight CTAs and capacity 1/2/4/8")
        kernel_type = ClusterKernel
    return cute.compile(
        kernel_type(cfg, plan, mode, decode, token_tile, allreduce=allreduce),
        *[Int64(0) for _ in range(13)],
        cute.runtime.make_fake_tensor(Int64, (2,), stride=(1,), assumed_align=8),
        Int32(1),
        cute.runtime.make_fake_tensor(Int64, (4,), stride=(1,), assumed_align=8),
        Int64(0),
        Int64(0),
        Int32(0),
        Int64(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi {options}",
    )
