# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: SGLang 72730d6 / FLA chunked WY; QuACK 0.6.5 warp-MMA scheduling;
# independently implemented, no upstream kernel code copied.

"""Split preparation and persistent FP32 recurrence, with standalone normalization.

WY preserves FP32 convolution values through split-operand U preparation.
TK remains an optional formulation for comparison.
With o_split, recurrence exports chunk-start operands and unweighted residuals
for a separate output grid; final state remains owned by the recurrence CTA.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import T
from cutlass.utils import SmemAllocator


# Public PTX ISA only. The CTA barrier gathers all producer stores before the
# leader release, and distributes the leader acquire to every consumer thread.
def _release(ptr):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value()],
        "fence.acq_rel.gpu; st.release.gpu.global.u32 [$0], 1;",
        "l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _acquire(ptr):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint().ir_value()],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _sleep(delay):
    llvm.inline_asm(
        None,
        [Int32(delay).ir_value()],
        "nanosleep.u32 $0;",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _ready_wait(ptr):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        delay = Int32(32)
        while _acquire(ptr) == 0:
            _sleep(delay)
            delay = cutlass.min(delay * 2, Int32(256))
    cute.arch.sync_threads()


@cute.jit
def _ready_publish(ptr):
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        _release(ptr)


@cute.jit
def _exp(x):
    return cute.exp(x, fastmath=True)


@cute.jit
def _silu(x):
    t = _exp(-abs(x))
    s = Float32(1.0) / (Float32(1.0) + t)
    if x < 0.0:
        s = t * s
    return x * s


@cute.jit
def _sum_warp(x):
    for step in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> step)
    return x


@cute.jit
def _layout(m: cutlass.Constexpr, n: cutlass.Constexpr):
    return cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3), 0, cute.make_layout((m, n), stride=(n, 1))
    )


@cute.jit
def _transpose(t):
    return cute.composition(t, cute.make_layout((t.shape[1], t.shape[0]), stride=(t.shape[0], 1)))


@cute.jit
def _mma(warps: cutlass.Constexpr, tm: cutlass.Constexpr, tn: cutlass.Constexpr):
    return cute.make_tiled_mma(
        warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
        cute.make_layout((tm // 16, warps // (tm // 16), 1)),
        permutation_mnk=(tm, tn, 16),
    )


@cute.jit
def _gemm(
    mma,
    a,
    b,
    acc,
    ta: cutlass.Constexpr = False,
    tb: cutlass.Constexpr = False,
    group_threads: cutlass.Constexpr = 0,
):
    """Shared A[M,K], B[N,K]; ldmatrix feeds multi-warp MMA accumulators."""
    tid, _, _ = cute.arch.thread_idx()
    if const_expr(group_threads):
        tid = tid % group_threads
    thread = mma.get_slice(tid)
    ra = thread.make_fragment_A(thread.partition_A(a))
    rb = thread.make_fragment_B(thread.partition_B(b))
    ac = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=ta, num_matrices=4), BFloat16), mma
    )
    bc = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=tb, num_matrices=4), BFloat16), mma
    )
    at, bt = ac.get_slice(tid), bc.get_slice(tid)
    sa, sb = at.partition_S(a), bt.partition_S(b)
    da, db = at.retile(ra), bt.retile(rb)
    # Register lookahead: load k+1 before issuing the current MMA.
    cute.copy(ac, sa[None, None, 0], da[None, None, 0])
    cute.copy(bc, sb[None, None, 0], db[None, None, 0])
    for kk in cutlass.range_constexpr(cute.size(ra, mode=[2])):
        if const_expr(kk + 1 < cute.size(ra, mode=[2])):
            cute.copy(ac, sa[None, None, kk + 1], da[None, None, kk + 1])
            cute.copy(bc, sb[None, None, kk + 1], db[None, None, kk + 1])
        cute.gemm(mma, acc, ra[None, None, kk], rb[None, None, kk], acc)


@cute.jit
def _gmem(
    ptr,
    off,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    stride: cutlass.Constexpr,
    align: cutlass.Constexpr = 16,
):
    ptr = cute.make_ptr(
        ptr.dtype,
        ptr.toint() + Int64(off) * (ptr.dtype.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=align,
    )
    return cute.make_tensor(ptr, cute.make_layout((m, n), stride=(stride, 1)))


@cute.jit
def _copy(
    src,
    dst,
    threads: cutlass.Constexpr,
    asynchronous: cutlass.Constexpr = False,
    group_threads: cutlass.Constexpr = 0,
):
    tid, _, _ = cute.arch.thread_idx()
    if const_expr(group_threads):
        tid = tid % group_threads
    dtype = src.element_type
    width = 128 // dtype.width
    op = cute.nvgpu.CopyUniversalOp()
    if const_expr(asynchronous):
        op = cpasync.CopyG2SOp()
    atom = cute.make_copy_atom(op, dtype, num_bits_per_copy=128)
    for index in range(tid, cute.size(src) // width, threads):
        row, col = index // (src.shape[1] // width), (index % (src.shape[1] // width)) * width
        s = cute.make_tensor(
            (src.iterator + src.layout((row, col))).align(16), cute.make_layout(width)
        )
        d = cute.make_tensor(
            (dst.iterator + dst.layout((row, col))).align(16), cute.make_layout(width)
        )
        if const_expr(asynchronous):
            cute.copy(atom, s, d)
        else:
            r = cute.make_rmem_tensor((width,), dtype)
            cute.copy(atom, s, r)
            cute.copy(atom, r, d)


@cute.jit
def _load_vector(ptr, offset, frag):
    """128-bit loads for aligned product inputs; no assumption about their storage offset."""
    source = ptr + Int64(offset)
    if source.toint() % 16 == 0:
        tensor = cute.make_tensor(source.align(16), cute.make_layout(cute.size(frag)))
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), frag.element_type, num_bits_per_copy=128
            ),
            tensor,
            frag,
        )
    else:
        for i in cutlass.range_constexpr(cute.size(frag)):
            frag[i] = source[i]


@cute.jit
def _wait(remaining: cutlass.Constexpr = 0):
    cute.arch.cp_async_wait_group(remaining)
    cute.arch.sync_threads()


@cute.jit
def _publish(acc, coords, smem):
    for e in cutlass.range_constexpr(cute.size(acc)):
        c = coords[e]
        smem[c[0], c[1]] = acc[e].to(smem.element_type)


@cute.jit
def _state_operand(
    h, coords, shared, low: cutlass.Constexpr = False, prefetched: cutlass.Constexpr = False
):
    # H remains FP32 in registers. Repack the same tile for each precision term,
    # avoiding a second H allocation that would halve shared-memory occupancy.
    if const_expr(not prefetched):
        cute.arch.sync_threads()
    for e in cutlass.range_constexpr(cute.size(coords)):
        row, col = coords[e]
        value = h[e]
        if const_expr(low):
            value -= Float32(BFloat16(value))
        shared[row, col] = BFloat16(value)
    if const_expr(prefetched):
        # Caller has retired the previous H consumer. One barrier covers both
        # the new H operand and the completed asynchronous W/U/g copies.
        cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()


@cute.jit
def _owner(cu, chunk, count: Int32):
    seq, local, begin, end = Int32(-1), Int32(0), Int64(0), Int64(0)
    base = Int32(0)
    for s in range(count):
        lo, hi = Int64(cu[s]), Int64(cu[s + 1])
        nc = Int32((hi - lo + 63) // 64)
        if seq < 0 and chunk >= base and chunk < base + nc:
            seq, local, begin, end = s, chunk - base, lo, hi
        base = base + nc
    return seq, local, begin, end


@cute.jit
def _history_bits(
    x,
    history,
    token,
    begin,
    channel,
    slot,
    initial,
    stride: cutlass.Constexpr,
    dc: cutlass.Constexpr,
):
    value = BFloat16(0.0)
    if token >= begin:
        value = x[Int64(token) * stride + channel]
    elif slot >= 0 and initial != 0:
        value = history[(Int64(slot) * 3 + token - begin + 3) * dc + channel]
    return value


@cute.jit
def _inverse(lower, inverse, warps: cutlass.Constexpr):
    """FP32 right solve: one independent inverse row per warp, two columns/lane.

    Solve x (I+L) = e_row backwards. A pivot is a warp broadcast; all column
    updates are parallel and remain in registers. Only publication needs a CTA
    barrier, replacing the old 64 barriers and shared inverse reads.
    """
    tid, _, _ = cute.arch.thread_idx()
    lane, wid = tid % 32, tid // 32
    for row in range(wid, 64, warps):
        x0, x1 = Float32(0.0), Float32(0.0)
        if lane < row:
            x0 = -lower[row, lane]
        if lane + 32 < row:
            x1 = -lower[row, lane + 32]
        if lane == row:
            x0 = Float32(1.0)
        if lane + 32 == row:
            x1 = Float32(1.0)
        for step in range(row - 1):
            pivot = row - 1 - step
            value = x0
            if pivot >= 32:
                value = x1
            value = cute.arch.shuffle_sync(value, offset=pivot % 32)
            if lane < pivot:
                x0 -= value * lower[pivot, lane]
            if lane + 32 < pivot:
                x1 -= value * lower[pivot, lane + 32]
        inverse[row, lane], inverse[row, lane + 32] = x0, x1
    cute.arch.sync_threads()


@cute.jit
def _small_mma(mma, a, b, acc, bf16: cutlass.Constexpr = False):
    """16x16 product: four BF16 split terms or six multipart TF32 terms.

    BF16 mode approximates each operand with hi+lo (about 16 significant bits).
    TF32 mode keeps 11+11+remaining mantissa bits and omits products below FP32
    precision. Both accumulate in FP32; reassociation affects final rounding.
    """
    tid, _, _ = cute.arch.thread_idx()
    thread = mma.get_slice(tid % 32)
    pa, pb = thread.partition_A(a), thread.partition_B(b)
    if const_expr(bf16):
        a0, a1 = (thread.make_fragment_A(pa) for _ in range(2))
        b0, b1 = (thread.make_fragment_B(pb) for _ in range(2))
        for e in cutlass.range_constexpr(cute.size(pa)):
            val = Float32(pa[e])
            a0[e] = BFloat16(val)
            a1[e] = BFloat16(val - Float32(a0[e]))
        for e in cutlass.range_constexpr(cute.size(pb)):
            val = Float32(pb[e])
            b0[e] = BFloat16(val)
            b1[e] = BFloat16(val - Float32(b0[e]))
        cute.gemm(mma, acc, a1, b1, acc)
        cute.gemm(mma, acc, a1, b0, acc)
        cute.gemm(mma, acc, a0, b1, acc)
        cute.gemm(mma, acc, a0, b0, acc)
    else:
        a0, a1, a2 = (thread.make_fragment_A(pa) for _ in range(3))
        b0, b1, b2 = (thread.make_fragment_B(pb) for _ in range(3))
        for e in cutlass.range_constexpr(cute.size(pa)):
            val = Float32(pa[e])
            a0[e] = Int32(cute.arch.cvt_f32_tf32(val))
            rem = val - a0[e].bitcast(Float32)
            a1[e] = Int32(cute.arch.cvt_f32_tf32(rem))
            a2[e] = Int32(cute.arch.cvt_f32_tf32(rem - a1[e].bitcast(Float32)))
        for e in cutlass.range_constexpr(cute.size(pb)):
            val = Float32(pb[e])
            b0[e] = Int32(cute.arch.cvt_f32_tf32(val))
            rem = val - b0[e].bitcast(Float32)
            b1[e] = Int32(cute.arch.cvt_f32_tf32(rem))
            b2[e] = Int32(cute.arch.cvt_f32_tf32(rem - b1[e].bitcast(Float32)))
        for kk in cutlass.range_constexpr(cute.size(a0, mode=[2])):
            cute.gemm(mma, acc, a2[None, None, kk], b0[None, None, kk], acc)
            cute.gemm(mma, acc, a1[None, None, kk], b1[None, None, kk], acc)
            cute.gemm(mma, acc, a0[None, None, kk], b2[None, None, kk], acc)
            cute.gemm(mma, acc, a1[None, None, kk], b0[None, None, kk], acc)
            cute.gemm(mma, acc, a0[None, None, kk], b1[None, None, kk], acc)
            cute.gemm(mma, acc, a0[None, None, kk], b0[None, None, kk], acc)


@cute.jit
def _block(tensor, row, col, transpose: cutlass.Constexpr = False):
    layout = cute.make_layout((16, 16), stride=(64, 1))
    if const_expr(transpose):
        layout = cute.make_layout((16, 16), stride=(1, 64))
    return cute.make_tensor(tensor.iterator + row * 1024 + col * 16, layout)


@cute.jit
def _inverse_blocked(
    lower, inverse, workspace, warps: cutlass.Constexpr, bf16: cutlass.Constexpr = False
):
    """Four concurrent 16x16 diagonal solves; independent off-diagonal blocks by MMA."""
    tid, _, _ = cute.arch.thread_idx()
    lane, wid = tid % 32, tid // 32
    for e in range(tid, 4096, warps * 32):
        inverse[e // 64, e % 64] = Float32(0.0)
    cute.arch.sync_threads()
    for row in range(wid, 64, warps):
        base = row // 16 * 16
        x = Float32(0.0)
        if lane < row - base:
            x = -lower[row, base + lane]
        if lane == row - base:
            x = Float32(1.0)
        for step in range(row - base - 1):
            pivot = row - base - 1 - step
            value = cute.arch.shuffle_sync(x, offset=pivot)
            if lane < pivot:
                x -= value * lower[base + pivot, base + lane]
        if lane < 16:
            inverse[row, base + lane] = x
    cute.arch.sync_threads()
    mma = cute.make_tiled_mma(
        warp.MmaTF32Op((16, 8, 8)), cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 8)
    )
    if const_expr(bf16):
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
                    bf16,
                )
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                tmp[r, c] = -acc[e]
            cute.arch.sync_warp()
            acc.fill(0.0)
            _small_mma(mma, tmp, _block(inverse, j, j, True), acc, bf16)
            for e in cutlass.range_constexpr(cute.size(coords)):
                r, c = coords[e]
                inverse[i * 16 + r, j * 16 + c] = acc[e]
        cute.arch.sync_threads()


class Prefill:
    def __init__(self, hk, hv, stride, tune, phase="all", captures=0):
        self.captures = captures
        self.hk, self.hv, self.stride = hk, hv, stride
        self.phase = phase
        self.overlap = bool(tune["overlap"] and tune["stop_after"] == "all")
        self.prep_mode = bool(tune["prep_mode"] and tune["stop_after"] in ("solve", "wy", "all"))
        self.prefetch_depth = tune["prefetch_depth"]
        self.merge = bool(tune["merge_prepare"] and tune["stop_after"] in ("solve", "wy", "all"))
        self.kh_store = hv if self.merge else hk
        self.rec_ws = bool(tune["rec_ws"])
        self.rec_mode = tune["rec_mode"] if tune["form"] == "wy" else 0
        self.bv = tune["bv"]
        self.o_split = bool(tune["o_split"])
        self.bvo = tune["bvo"]
        self.ow = tune["output_warps"]
        self.pw, self.rw = tune["prepare_warps"], tune["recurrence_warps"]
        self.tm, self.tn = tune["mma_m"], tune["mma_n"]
        self.split = bool(tune["split_h"])
        self.tk = tune["form"] == "tk"
        self.t_store = tune["t_store"]
        self.r_split = bool(tune["r_split"])
        self.k_split = bool(tune["k_split"])
        self.v_fp32 = tune["v_store"] == "fp32"
        self.u_fp32 = tune["u_store"] == "fp32"
        self.ut_split = bool(tune["ut_split"])
        self.w_split = bool(tune["w_split"])
        self.o_bf16 = bool(tune["o_bf16"])
        self.update_split = bool(tune["update_split"])
        self.pre = tune["o_form"] == "pre"
        self.qp_split = bool(tune["qp_split"])
        # Prepare always builds precise QK for the pre form.
        self.aq_qk = self.pre or bool(tune["aq_qk"])
        self.aq_out = self.pre or bool(tune["aq_out"])
        self.solve_mma = tune["solve_mma"]
        self.stop = ("conv", "norm", "gram", "solve", "wy", "all").index(tune["stop_after"])
        self.dc = (2 * hk + hv) * 128

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
        nonempty = cutlass.min(count, x.shape[0])
        chunks = nonempty + (x.shape[0] - nonempty) // 64
        if x.shape[0] > 0:
            if const_expr(self.phase in ("all", "pipeline") and not (self.merge or self.prep_mode)):
                self.prepare_key(
                    x.iterator,
                    cw.iterator,
                    cu.iterator,
                    slots.iterator,
                    initial.iterator,
                    history.iterator,
                    q.iterator,
                    k.iterator,
                    kk.iterator,
                    qk.iterator,
                    kl.iterator,
                    hout.iterator,
                    count,
                ).launch(
                    grid=(self.hk * chunks, 1, 1),
                    block=(32 * self.pw, 1, 1),
                    stream=stream,
                    use_pdl=False,
                    preferred_smem_carveout=100,
                )
        if x.shape[0] > 0:
            if const_expr(self.phase in ("all", "pipeline") and self.stop >= 3):
                self.prepare_value(
                    x.iterator,
                    cw.iterator,
                    alog.iterator,
                    bias.iterator,
                    cu.iterator,
                    slots.iterator,
                    initial.iterator,
                    history.iterator,
                    k.iterator,
                    kl.iterator,
                    kk.iterator,
                    qk.iterator,
                    w.iterator,
                    u.iterator,
                    aq.iterator,
                    g.iterator,
                    hout.iterator,
                    q.iterator,
                    ready.iterator,
                    qp.iterator,
                    op.iterator,
                    count,
                ).launch(
                    grid=(
                        (self.hk if self.prep_mode else self.hv) * chunks,
                        1,
                        1,
                    ),
                    block=(32 * self.pw, 1, 1),
                    stream=stream,
                    use_pdl=self.overlap and not (self.merge or self.prep_mode),
                    preferred_smem_carveout=100,
                )
        if const_expr(self.phase in ("all", "pipeline") and self.stop == 5):
            self.recurrence(
                x.iterator,
                nw.iterator,
                cu.iterator,
                slots.iterator,
                initial.iterator,
                state.iterator,
                history.iterator,
                q.iterator,
                k.iterator,
                w.iterator,
                u.iterator,
                aq.iterator,
                g.iterator,
                hout.iterator,
                out.iterator,
                partial.iterator,
                y.iterator,
                kl.iterator,
                ready.iterator,
                qp.iterator,
                op.iterator,
                capture_offsets.iterator if self.captures else None,
                capture_rows.iterator if self.captures else None,
                hbuf.iterator,
                vbuf.iterator,
                x.shape[0],
            ).launch(
                grid=((128 // self.bv) * self.hv * count, 1, 1),
                block=(32 * self.rw, 1, 1),
                stream=stream,
                use_pdl=self.overlap,
                preferred_smem_carveout=100,
            )
        if x.shape[0] > 0:
            if const_expr(self.phase in ("all", "norm") and self.stop == 5):
                if const_expr(self.o_split):
                    self.output(
                        cu.iterator,
                        q.iterator,
                        aq.iterator,
                        g.iterator,
                        hbuf.iterator,
                        vbuf.iterator,
                        out.iterator,
                        partial.iterator,
                        count,
                    ).launch(
                        grid=(chunks, self.hv, 128 // self.bvo),
                        block=(32 * self.ow, 1, 1),
                        stream=stream,
                        use_pdl=False,
                        preferred_smem_carveout=100,
                    )
                self.norm(
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
                    capture_offsets.iterator if self.captures else None,
                    capture_rows.iterator if self.captures else None,
                ).launch(
                    grid=((x.shape[0] + 7) // 8, self.hv, 1),
                    block=(128, 1, 1),
                    stream=stream,
                    use_pdl=False,
                    preferred_smem_carveout=100,
                )

    @cute.jit
    def conv_prefetch(
        self, x, history, raw, start, begin, end, channel, slot, flag, columns: cutlass.Constexpr
    ):
        tid, _, _ = cute.arch.thread_idx()
        threads = 32 * self.pw
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128)
        # Input pointers/row strides can be only BF16 aligned. Branch on the actual
        # address, without copying/allocating or changing the graph's launch count.
        for v in range(tid, 67 * (columns // 8), threads):
            row, col = v // (columns // 8), (v % (columns // 8)) * 8
            token = start - 3 + row
            src = x + token * self.stride + channel + col
            valid = token >= begin and token < end
            if token < begin and slot >= 0 and flag != 0:
                src = history + (slot * 3 + token - begin + 3) * self.dc + channel + col
                valid = True
            if valid:
                if src.toint() % 16 == 0:
                    s = cute.make_tensor(_gmem(src, 0, 1, 8, 8).iterator, cute.make_layout(8))
                    d = cute.make_tensor(
                        (raw.iterator + raw.layout((row, col))).align(16), cute.make_layout(8)
                    )
                    cute.copy(atom, s, d)
                else:
                    for j in cutlass.range_constexpr(8):
                        raw[row, col + j] = src[j]
            else:
                for j in cutlass.range_constexpr(8):
                    raw[row, col + j] = BFloat16(0.0)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def conv_rows(
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
        extra=None,
        extra_full: cutlass.Constexpr = False,
        columns: cutlass.Constexpr = 128,
        prefetched: cutlass.Constexpr = False,
        extra_shared=None,
        wait_remaining: cutlass.Constexpr = 0,
    ):
        """Stage 67 projection rows once, then convolve eight adjacent channels/lane."""
        tid, _, _ = cute.arch.thread_idx()
        threads = 32 * self.pw
        if const_expr(not prefetched):
            self.conv_prefetch(x, history, raw, start, begin, end, channel, slot, flag, columns)
        weights = cute.make_rmem_tensor((32,), BFloat16)
        _load_vector(cw, (channel + (tid % (columns // 8)) * 8) * 4, weights)
        _wait(wait_remaining)
        vector_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        for v in range(tid, 64 * (columns // 8), threads):
            row, col = v // (columns // 8), (v % (columns // 8)) * 8
            values = cute.make_rmem_tensor((8,), Float32)
            samples = cute.make_rmem_tensor((8,), BFloat16)
            packed = cute.make_rmem_tensor((8,), BFloat16)
            values.fill(0.0)
            for tap in cutlass.range_constexpr(4):
                source = cute.make_tensor(
                    (raw.iterator + raw.layout((row + tap, col))).align(16), cute.make_layout(8)
                )
                cute.copy(vector_atom, source, samples)
                for j in cutlass.range_constexpr(8):
                    values[j] += Float32(samples[j]) * Float32(weights[j * 4 + tap])
            square = Float32(0.0)
            for j in cutlass.range_constexpr(8):
                val = values[j]
                if start + row >= end:
                    val = Float32(0.0)
                values[j] = _silu(val)
                square += values[j] * values[j]
            factor = Float32(1.0)
            if const_expr(normalize):
                # Sixteen adjacent lanes cover one 128-channel row.
                for step in cutlass.range_constexpr(4):
                    square += cute.arch.shuffle_sync_bfly(square, offset=1 << step)
                factor = cute.rsqrt(square + 1.0e-6)
            if const_expr(scale is not None):
                factor *= scale[row]
            for j in cutlass.range_constexpr(8):
                value = values[j] * factor
                packed[j] = BFloat16(value)
                if const_expr(extra_shared is not None):
                    extra_shared[row, col + j] = BFloat16(value - Float32(packed[j]))
                if const_expr(extra is not None):
                    if const_expr(extra_full):
                        extra[row * 128 + col + j] = value
                    else:
                        extra[row * 128 + col + j] = BFloat16(value - Float32(packed[j]))
            target = cute.make_tensor(
                (dst.iterator + dst.layout((row, col))).align(16), cute.make_layout(8)
            )
            cute.copy(vector_atom, packed, target)
        cute.arch.sync_threads()

    @cute.jit
    def history_stage(self, x, history, hout, begin, end, slot, flag, seq, channel):
        tid, _, _ = cute.arch.thread_idx()
        for e in range(tid, 3 * 16, 32 * self.pw):
            row, col = e // 16, (e % 16) * 8
            frag = cute.make_rmem_tensor((8,), BFloat16)
            for j in cutlass.range_constexpr(8):
                frag[j] = _history_bits(
                    x,
                    history,
                    end - 3 + row,
                    begin,
                    channel + col + j,
                    slot,
                    flag,
                    self.stride,
                    self.dc,
                )
            dst = _gmem(hout, (Int64(seq) * 3 + row) * self.dc + channel + col, 1, 8, 8)
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128),
                frag,
                cute.make_tensor(dst.iterator, cute.make_layout(8)),
            )

    @cute.jit
    def history_commit(self, history, hout, seq, slot, channel, threads: cutlass.Constexpr = 0):
        if const_expr(threads == 0):
            threads = 32 * self.rw
        if history.toint() % 16 == 0:
            _copy(
                _gmem(hout, Int64(seq) * 3 * self.dc + channel, 3, 128, self.dc),
                _gmem(history, slot * 3 * self.dc + channel, 3, 128, self.dc),
                threads,
            )
        else:
            tid, _, _ = cute.arch.thread_idx()
            for e in range(tid, 384, threads):
                row, col = e // 128, e % 128
                history[(slot * 3 + row) * self.dc + channel + col] = hout[
                    (Int64(seq) * 3 + row) * self.dc + channel + col
                ]

    @cute.jit
    def zero_history(self, history, slot, head, threads: cutlass.Constexpr):
        """Fresh empty row: V per value head, Q/K once per shared key head."""
        tid, _, _ = cute.arch.thread_idx()
        for e in range(tid, 384, threads):
            row, col = e // 128, e % 128
            history[(Int64(slot) * 3 + row) * self.dc + (2 * self.hk + head) * 128 + col] = (
                BFloat16(0.0)
            )
            if head % (self.hv // self.hk) == 0:
                kh = head // (self.hv // self.hk)
                for part in cutlass.range_constexpr(2):
                    history[
                        (Int64(slot) * 3 + row) * self.dc + (part * self.hk + kh) * 128 + col
                    ] = BFloat16(0.0)

    @cute.jit
    def capture_history(
        self,
        x,
        history,
        offsets,
        rows,
        initial,
        seq,
        slot,
        begin,
        end,
        head,
        threads: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        for capture in cutlass.range_constexpr(self.captures):
            p = offsets[Int64(seq) * self.captures + capture]
            target = Int64(rows[Int64(seq) * self.captures + capture])
            if target >= 0 and p > 0 and p <= end - begin:
                # Value channels are private to this head; only the first value
                # head of a group owns the shared q/k channels, as in commit.
                for part in cutlass.range_constexpr(3):
                    if part == 0 or head % (self.hv // self.hk) == 0:
                        channel = (2 * self.hk + head) * 128
                        if const_expr(part > 0):
                            channel = ((part - 1) * self.hk + head // (self.hv // self.hk)) * 128
                        for e in range(tid, 384, threads):
                            row, col = e // 128, e % 128
                            history[(target * 3 + row) * self.dc + channel + col] = _history_bits(
                                x,
                                history,
                                begin + p - 3 + row,
                                begin,
                                channel + col,
                                slot,
                                initial[seq],
                                self.stride,
                                self.dc,
                            )

    @cute.kernel
    def prepare_key(
        self,
        x: cute.Pointer,
        cw: cute.Pointer,
        cu: cute.Pointer,
        slots: cute.Pointer,
        initial: cute.Pointer,
        history: cute.Pointer,
        q: cute.Pointer,
        k: cute.Pointer,
        kk_out: cute.Pointer,
        qk_out: cute.Pointer,
        kl: cute.Pointer,
        hout: cute.Pointer,
        count: Int32,
    ):
        if const_expr(self.overlap):
            cute.arch.griddepcontrol_launch_dependents()
        bx, _, _ = cute.arch.block_idx()
        kh, chunk = Int32(bx) % self.hk, Int32(bx) // self.hk
        seq, local, begin, end = _owner(cu, chunk, count)
        if seq >= 0:
            alloc = SmemAllocator()
            raw = alloc.allocate_tensor(
                BFloat16, cute.make_layout((68, 128), stride=(128, 1)), byte_alignment=128
            )
            sk = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
            self.key_body(
                x,
                cw,
                slots,
                initial,
                history,
                q,
                k,
                kk_out,
                qk_out,
                kl,
                hout,
                raw,
                sk,
                chunk,
                kh,
                kh,
                seq,
                local,
                begin,
                end,
                True,
            )

    @cute.jit
    def key_body(
        self,
        x,
        cw,
        slots,
        initial,
        history,
        q,
        k,
        kk_out,
        qk_out,
        kl,
        hout,
        raw,
        sk,
        chunk,
        kh,
        storage_head,
        seq,
        local,
        begin,
        end,
        publish_history,
    ):
        tid, _, _ = cute.arch.thread_idx()
        threads = 32 * self.pw
        start = begin + Int64(local) * 64
        slot, flag = Int64(slots[seq]), initial[seq]
        sq = cute.make_tensor(raw.iterator, _layout(64, 128))
        qo = (Int64(chunk) * self.kh_store + storage_head) * 8192
        # Keep Q_hi contiguous for every precision mode: recurrence never needs
        # Q_lo or a precision-dependent stride. Prepare owns the second slab.
        grid_x, _, _ = cute.arch.grid_dim()
        chunks = Int32(grid_x) // (self.hv if self.merge else self.hk)
        qlo = Int64(chunks) * self.kh_store * 8192
        self.conv_rows(
            x,
            cw,
            history,
            raw,
            sk,
            start,
            begin,
            end,
            kh * 128,
            slot,
            flag,
            self.stop >= 1,
            extra=q + qlo + qo if self.aq_qk else None,
        )
        _copy(sk, _gmem(q, qo, 64, 128, 128), threads)
        cute.arch.sync_threads()
        self.conv_rows(
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
            self.stop >= 1,
            extra=kl + qo if self.k_split or self.aq_qk else None,
        )
        _copy(sk, _gmem(k, qo, 64, 128, 128), threads)
        if local == 0 and slot >= 0 and publish_history:
            self.history_stage(x, history, hout, begin, end, slot, flag, seq, kh * 128)
            self.history_stage(x, history, hout, begin, end, slot, flag, seq, (self.hk + kh) * 128)
        if const_expr(self.stop >= 2):
            # Raw projection staging is dead. Reuse it for Q, then FP32 products.
            cute.arch.sync_threads()
            _copy(_gmem(q, qo, 64, 128, 128), sq, threads, True)
            cute.arch.cp_async_commit_group()
            _wait()
            mma = _mma(self.pw, self.tm, self.tn)
            cg = mma.get_slice(tid).partition_C(cute.make_identity_tensor((64, 64)))
            kk = cute.make_rmem_tensor(cg.shape, Float32)
            qk = cute.make_rmem_tensor(cg.shape, Float32)
            kk.fill(0.0)
            qk.fill(0.0)
            _gemm(mma, sk, sk, kk)
            _gemm(mma, sq, sk, qk)
            if const_expr(self.k_split):
                cute.arch.sync_threads()
                _copy(_gmem(kl, qo, 64, 128, 128), sq, threads, True)
                cute.arch.cp_async_commit_group()
                _wait()
                _gemm(mma, sk, sq, kk)
                _gemm(mma, sq, sk, kk)
                _gemm(mma, sq, sq, kk)
                cute.arch.sync_threads()
                _copy(_gmem(q, qo, 64, 128, 128), sk, threads, True)
                cute.arch.cp_async_commit_group()
                _wait()
                if const_expr(not self.aq_qk):
                    _gemm(mma, sk, sq, qk)
            if const_expr(self.aq_qk):
                # Recover normalized Q/K before BF16 rounding, without changing KK.
                cute.arch.sync_threads()
                _copy(_gmem(q, qlo + qo, 64, 128, 128), sq, threads, True)
                _copy(_gmem(k, qo, 64, 128, 128), sk, threads, True)
                cute.arch.cp_async_commit_group()
                _wait()
                _gemm(mma, sq, sk, qk)
                cute.arch.sync_threads()
                _copy(_gmem(q, qo, 64, 128, 128), sq, threads, True)
                _copy(_gmem(kl, qo, 64, 128, 128), sk, threads, True)
                cute.arch.cp_async_commit_group()
                _wait()
                _gemm(mma, sq, sk, qk)
            cute.arch.sync_threads()
            product = cute.make_tensor(
                cute.recast_ptr(raw.iterator, dtype=Float32),
                cute.make_layout((64, 64), stride=(64, 1)),
            )
            _publish(kk, cg, product)
            cute.arch.sync_threads()
            _copy(product, _gmem(kk_out, qo // 2, 64, 64, 64), threads)
            cute.arch.sync_threads()
            _publish(qk, cg, product)
            cute.arch.sync_threads()
            _copy(product, _gmem(qk_out, qo // 2, 64, 64, 64), threads)

    @cute.jit
    def gates(self, x, alog, bias, sg, sb, start, end, head):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        # Two warps scan 32 rows each, then add the first half's total.
        if tid < 64:
            decay, beta = Float32(0.0), Float32(0.0)
            if start + tid < end:
                at = (start + tid) * self.stride + self.dc + self.hv * 128
                av = Float32(x[at + self.hv + head]) + Float32(bias[head])
                bv = Float32(x[at + head])
                t = _exp(-abs(av))
                lp = cute.log(1.0 + t)
                if t < 0.0001:
                    lp = t * (1.0 - 0.5 * t)
                sp = lp + cutlass.max(av, Float32(0.0))
                logsp = cute.log(sp)
                if av < -80.0:
                    logsp = av
                decay = cutlass.max(-_exp(Float32(alog[head]) + logsp), Float32(-512.0))
                bt = _exp(-abs(bv))
                beta = 1.0 / (1.0 + bt)
                if bv < 0.0:
                    beta *= bt
            for step in cutlass.range_constexpr(5):
                prev = cute.arch.shuffle_sync_up(decay, offset=1 << step)
                if lane >= (1 << step):
                    decay += prev
            sg[tid], sb[tid] = decay, beta
            # Only these two warps exchange the scan carry. Other warps join
            # once, at the final publication to the MMA consumers.
            cute.arch.barrier(barrier_id=1, number_of_threads=64)
            if tid >= 32:
                sg[tid] += sg[31]
        cute.arch.sync_threads()

    @cute.kernel
    def prepare_value(
        self,
        x: cute.Pointer,
        cw: cute.Pointer,
        alog: cute.Pointer,
        bias: cute.Pointer,
        cu: cute.Pointer,
        slots: cute.Pointer,
        initial: cute.Pointer,
        history: cute.Pointer,
        k: cute.Pointer,
        kl: cute.Pointer,
        kk_in: cute.Pointer,
        qk_in: cute.Pointer,
        w: cute.Pointer,
        u: cute.Pointer,
        aq: cute.Pointer,
        g: cute.Pointer,
        hout: cute.Pointer,
        q: cute.Pointer,
        ready: cute.Pointer,
        qp: cute.Pointer,
        op: cute.Pointer,
        count: Int32,
    ):
        if const_expr(self.overlap):
            cute.arch.griddepcontrol_launch_dependents()
            if const_expr(not (self.merge or self.prep_mode)):
                cute.arch.griddepcontrol_wait()
        bx, _, _ = cute.arch.block_idx()
        groups = self.hk if self.prep_mode else self.hv
        group, chunk = Int32(bx) % groups, Int32(bx) // groups
        tid, _, _ = cute.arch.thread_idx()
        threads = 32 * self.pw
        kh = group if self.prep_mode else group // (self.hv // self.hk)
        seq, local, begin, end = _owner(cu, chunk, count)
        if seq >= 0:
            start = begin + Int64(local) * 64
            slot, flag = Int64(slots[seq]), initial[seq]
            alloc = SmemAllocator()
            raw = alloc.allocate_tensor(
                BFloat16, cute.make_layout((68, 128), stride=(128, 1)), byte_alignment=128
            )
            rhs = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
            si = alloc.allocate_tensor(BFloat16, _layout(64, 64), byte_alignment=128)
            lower = cute.make_tensor(
                cute.recast_ptr(raw.iterator, dtype=Float32),
                cute.make_layout((64, 64), stride=(64, 1)),
            )
            inverse = cute.make_tensor(
                cute.recast_ptr(rhs.iterator, dtype=Float32),
                cute.make_layout((64, 64), stride=(64, 1)),
            )
            sg = alloc.allocate_tensor(Float32, cute.make_layout(64))
            sb = alloc.allocate_tensor(Float32, cute.make_layout(64))
            mma = _mma(self.pw, self.tm, self.tn)
            thread = mma.get_slice(tid)
            cg = thread.partition_C(cute.make_identity_tensor((64, 64)))
            if const_expr(self.prep_mode):
                self.key_body(
                    x,
                    cw,
                    slots,
                    initial,
                    history,
                    q,
                    k,
                    kk_in,
                    qk_in,
                    kl,
                    hout,
                    raw,
                    rhs,
                    chunk,
                    kh,
                    kh,
                    seq,
                    local,
                    begin,
                    end,
                    True,
                )
                # Global key products are CTA-local producers for every value head.
                cute.arch.sync_threads()
            for value_head in range(self.hv // self.hk if self.prep_mode else 1):
                head = group * (self.hv // self.hk) + value_head if self.prep_mode else group
                off = (Int64(chunk) * self.hv + head) * 8192
                goff = (Int64(chunk) * self.hv + head) * 64
                ko = (Int64(chunk) * self.kh_store + (head if self.merge else kh)) * 8192
                if const_expr(self.merge):
                    self.key_body(
                        x,
                        cw,
                        slots,
                        initial,
                        history,
                        q,
                        k,
                        kk_in,
                        qk_in,
                        kl,
                        hout,
                        raw,
                        rhs,
                        chunk,
                        kh,
                        head,
                        seq,
                        local,
                        begin,
                        end,
                        head % (self.hv // self.hk) == 0,
                    )
                    cute.arch.sync_threads()
                _copy(_gmem(kk_in, ko // 2, 64, 64, 64), lower, threads, True)
                _copy(_gmem(qk_in, ko // 2, 64, 64, 64), inverse, threads, True)
                cute.arch.cp_async_commit_group()
                self.gates(x, alog, bias, sg, sb, start, end, head)
                _copy(
                    cute.make_tensor(sg.iterator, cute.make_layout((1, 64), stride=(64, 1))),
                    _gmem(g, goff * (2 if self.tk else 1), 1, 64, 64),
                    threads,
                )
                if const_expr(self.tk):
                    _copy(
                        cute.make_tensor(sb.iterator, cute.make_layout((1, 64), stride=(64, 1))),
                        _gmem(g, goff * 2 + 64, 1, 64, 64),
                        threads,
                    )
                _wait()
                # Read both products before reusing either operand buffer.
                kk = cute.make_rmem_tensor(cg.shape, Float32)
                qk = cute.make_rmem_tensor(cg.shape, Float32)
                for e in cutlass.range_constexpr(cute.size(cg)):
                    i, j = cg[e]
                    kk[e], qk[e] = lower[i, j], inverse[i, j]
                cute.arch.sync_threads()
                for e in cutlass.range_constexpr(cute.size(cg)):
                    i, j = cg[e]
                    lv, av = Float32(0.0), Float32(0.0)
                    if i > j:
                        lv = kk[e] * sb[i] * _exp(sg[i] - sg[j])
                    if i >= j:
                        av = qk[e] * _exp(sg[i] - sg[j])
                    lower[i, j] = lv
                    si[i, j] = BFloat16(av)
                cute.arch.sync_threads()
                _copy(si, _gmem(aq, goff * 64 * (2 if self.aq_out else 1), 64, 64, 64), threads)
                if const_expr(self.aq_out):
                    cute.arch.sync_threads()
                    for e in cutlass.range_constexpr(cute.size(cg)):
                        i, j = cg[e]
                        av = Float32(0.0)
                        if i >= j:
                            av = qk[e] * _exp(sg[i] - sg[j])
                        si[i, j] = BFloat16(av - Float32(BFloat16(av)))
                    cute.arch.sync_threads()
                    _copy(si, _gmem(aq, goff * 128 + 4096, 64, 64, 64), threads)
                cute.arch.sync_threads()
                if const_expr(self.solve_mma):
                    _inverse_blocked(lower, inverse, si, self.pw, self.solve_mma == 2)
                else:
                    _inverse(lower, inverse, self.pw)
                if const_expr(self.tk):
                    # Publish T before reusing the inverse/raw storage for convolution.
                    if const_expr(self.t_store == "fp32"):
                        _copy(inverse, _gmem(w, goff * 64, 64, 64, 64), threads)
                    else:
                        for part in cutlass.range_constexpr(2 if self.t_store == "split" else 1):
                            for e in range(tid, 4096, threads):
                                value = inverse[e // 64, e % 64]
                                if const_expr(part == 1):
                                    value -= Float32(BFloat16(value))
                                si[e // 64, e % 64] = BFloat16(value)
                            cute.arch.sync_threads()
                            _copy(
                                si,
                                _gmem(
                                    w,
                                    goff * 64 * (2 if self.t_store == "split" else 1) + part * 4096,
                                    64,
                                    64,
                                    64,
                                ),
                                threads,
                            )
                            cute.arch.sync_threads()
                    cute.arch.sync_threads()
                    if const_expr(self.stop >= 4):
                        self.conv_rows(
                            x,
                            cw,
                            history,
                            raw,
                            rhs,
                            start,
                            begin,
                            end,
                            2 * self.hk * 128 + head * 128,
                            slot,
                            flag,
                            extra=u + off if self.v_fp32 else None,
                            extra_full=True,
                        )
                        if const_expr(not self.v_fp32):
                            _copy(rhs, _gmem(u, off, 64, 128, 128), threads)
                else:
                    # g has been published. Reuse sg for beta*exp(g), once per row.
                    if tid < 64:
                        sg[tid] = sb[tid] * _exp(sg[tid])
                    # Inverse hi lives in si; optional lo fits above 68x64 raw rows.
                    raw64 = cute.make_tensor(
                        raw.iterator, cute.make_layout((68, 64), stride=(64, 1))
                    )
                    tlo = cute.make_tensor(raw.iterator + 4352, _layout(64, 64))
                    for e in range(tid, 4096, threads):
                        value = inverse[e // 64, e % 64]
                        si[e // 64, e % 64] = BFloat16(value)
                        if const_expr(self.ut_split):
                            tlo[e // 64, e % 64] = BFloat16(value - Float32(BFloat16(value)))
                    cute.arch.sync_threads()
                    if const_expr(self.stop == 3):
                        _copy(si, _gmem(w, off * (1 + self.w_split), 64, 64, 128), threads)
                    if const_expr(self.stop >= 4):
                        # Two strips bound the live accumulator and fit V hi/lo in rhs.
                        rh = cute.make_tensor(rhs.iterator, _layout(64, 64))
                        rl = cute.make_tensor(rhs.iterator + 4096, _layout(64, 64))
                        uf = cute.make_tensor(
                            cute.recast_ptr(rhs.iterator, dtype=Float32),
                            cute.make_layout((64, 64), stride=(64, 1)),
                        )
                        wc = thread.partition_C(cute.make_identity_tensor((64, 64)))
                        wp = cute.make_rmem_tensor(wc.shape, Float32)
                        for part in range(2):
                            column = part * 64
                            _copy(_gmem(k, ko + column, 64, 64, 128), rh, threads, True)
                            cute.arch.cp_async_commit_group()
                            _wait()
                            for e in range(tid, 4096, threads):
                                i, j = e // 64, e % 64
                                rh[i, j] = BFloat16(Float32(rh[i, j]) * sg[i])
                            cute.arch.sync_threads()
                            # raw64 and Tlo are disjoint, including the padded row.
                            self.conv_prefetch(
                                x,
                                history,
                                raw64,
                                start,
                                begin,
                                end,
                                2 * self.hk * 128 + head * 128 + column,
                                slot,
                                flag,
                                64,
                            )
                            wp.fill(0.0)
                            _gemm(mma, si, _transpose(rh), wp, tb=True)
                            if const_expr(self.k_split):
                                cute.arch.sync_threads()
                                _copy(_gmem(kl, ko + column, 64, 64, 128), rh, threads, True)
                                cute.arch.cp_async_commit_group()
                                _wait()
                                for e in range(tid, 4096, threads):
                                    i, j = e // 64, e % 64
                                    rh[i, j] = BFloat16(Float32(rh[i, j]) * sg[i])
                                cute.arch.sync_threads()
                                _gemm(mma, si, _transpose(rh), wp, tb=True)
                            cute.arch.sync_threads()
                            _publish(wp, wc, rh)
                            if const_expr(self.w_split):
                                for e in cutlass.range_constexpr(cute.size(wc)):
                                    i, j = wc[e]
                                    rl[i, j] = BFloat16(wp[e] - Float32(BFloat16(wp[e])))
                            cute.arch.sync_threads()
                            _copy(
                                rh,
                                _gmem(w, off * (1 + self.w_split) + column, 64, 64, 128),
                                threads,
                            )
                            if const_expr(self.w_split):
                                _copy(rl, _gmem(w, off * 2 + 8192 + column, 64, 64, 128), threads)
                            cute.arch.sync_threads()
                            self.conv_rows(
                                x,
                                cw,
                                history,
                                raw64,
                                rh,
                                start,
                                begin,
                                end,
                                2 * self.hk * 128 + head * 128 + column,
                                slot,
                                flag,
                                scale=sb,
                                columns=64,
                                prefetched=True,
                                extra_shared=rl,
                            )
                            # beta*SiLU(conv(V)) is split in FP32, before any BF16 rounding.
                            wp.fill(0.0)
                            _gemm(mma, si, _transpose(rh), wp, tb=True)
                            _gemm(mma, si, _transpose(rl), wp, tb=True)
                            if const_expr(self.ut_split):
                                _gemm(mma, tlo, _transpose(rh), wp, tb=True)
                                _gemm(mma, tlo, _transpose(rl), wp, tb=True)
                            cute.arch.sync_threads()
                            if const_expr(self.u_fp32):
                                _publish(wp, wc, uf)
                                cute.arch.sync_threads()
                                _copy(uf, _gmem(u, off + column, 64, 64, 128), threads)
                            else:
                                _publish(wp, wc, rh)
                                cute.arch.sync_threads()
                                _copy(rh, _gmem(u, off + column, 64, 64, 128), threads)
                            cute.arch.sync_threads()
                if const_expr(self.pre and self.stop >= 4):
                    self.prepare_output(
                        q, w, u, aq, g, qp, op, raw, rhs, si, chunk, head, ko, off, mma, cg
                    )
                if const_expr(self.stop >= 4):
                    if local == 0 and slot >= 0:
                        self.history_stage(
                            x,
                            history,
                            hout,
                            begin,
                            end,
                            slot,
                            flag,
                            seq,
                            2 * self.hk * 128 + head * 128,
                        )
                if const_expr(self.overlap):
                    _ready_publish(ready + (Int64(chunk) * self.hv + head))
                cute.arch.sync_threads()

    @cute.jit
    def prepare_output(
        self, q, w, u, aq, g, qp, op, raw, rhs, sa, chunk, head, ko, off, mma, coords
    ):
        """Unscaled Q'=Q exp(g)-A W and O_pre=A U, after W/U publication.

        Reuse the solve/convolution tiles. A and FP32 U are split before MMA;
        W is already the BF16 operand consumed by the state recurrence.
        """
        tid, _, _ = cute.arch.thread_idx()
        threads = 32 * self.pw
        hi = cute.make_tensor(raw.iterator, _layout(64, 64))
        lo = cute.make_tensor(raw.iterator + 4096, _layout(64, 64))
        fp = cute.make_tensor(
            cute.recast_ptr(rhs.iterator, dtype=Float32), cute.make_layout((64, 64), stride=(64, 1))
        )
        grid_x, _, _ = cute.arch.grid_dim()
        chunks = Int32(grid_x) // (self.hk if self.prep_mode else self.hv)
        qlo = Int64(chunks) * self.kh_store * 8192
        acc = cute.make_rmem_tensor(coords.shape, Float32)
        for operand in cutlass.range_constexpr(2):
            for strip in range(2):
                column = strip * 64
                if const_expr(operand == 0):
                    _copy(_gmem(w, off + column, 64, 64, 128), hi, threads, True)
                elif const_expr(self.u_fp32):
                    _copy(_gmem(u, off + column, 64, 64, 128), fp, threads, True)
                else:
                    _copy(_gmem(u, off + column, 64, 64, 128), hi, threads, True)
                _copy(
                    _gmem(aq, (Int64(chunk) * self.hv + head) * 8192, 64, 64, 64), sa, threads, True
                )
                cute.arch.cp_async_commit_group()
                _wait()
                if const_expr(operand == 1 and self.u_fp32):
                    for e in range(tid, 4096, threads):
                        i, j = e // 64, e % 64
                        value = fp[i, j]
                        hi[i, j] = BFloat16(value)
                        lo[i, j] = BFloat16(value - Float32(BFloat16(value)))
                    cute.arch.sync_threads()
                acc.fill(0.0)
                _gemm(mma, sa, _transpose(hi), acc, tb=True)
                if const_expr(operand == 1 and self.u_fp32):
                    _gemm(mma, sa, _transpose(lo), acc, tb=True)
                cute.arch.sync_threads()
                _copy(
                    _gmem(aq, (Int64(chunk) * self.hv + head) * 8192 + 4096, 64, 64, 64),
                    sa,
                    threads,
                    True,
                )
                cute.arch.cp_async_commit_group()
                _wait()
                _gemm(mma, sa, _transpose(hi), acc, tb=True)
                if const_expr(operand == 1 and self.u_fp32):
                    _gemm(mma, sa, _transpose(lo), acc, tb=True)
                cute.arch.sync_threads()
                if const_expr(operand == 0):
                    for e in cutlass.range_constexpr(cute.size(coords)):
                        i, j = coords[e]
                        qi = ko + i * 128 + column + j
                        value = (Float32(q[qi]) + Float32(q[qi + qlo])) * _exp(
                            g[(Int64(chunk) * self.hv + head) * 64 + i]
                        ) - acc[e]
                        hi[i, j] = BFloat16(value)
                        if const_expr(self.qp_split):
                            lo[i, j] = BFloat16(value - Float32(BFloat16(value)))
                    cute.arch.sync_threads()
                    _copy(hi, _gmem(qp, off * (1 + self.qp_split) + column, 64, 64, 128), threads)
                    if const_expr(self.qp_split):
                        _copy(lo, _gmem(qp, off * 2 + 8192 + column, 64, 64, 128), threads)
                else:
                    _publish(acc, coords, fp)
                    cute.arch.sync_threads()
                    _copy(fp, _gmem(op, off + column, 64, 64, 128), threads)
                cute.arch.sync_threads()

    @cute.jit
    def prefetch(self, k, w, u, g, sw, sv, sg, sb, sf, chunk, head, tile):
        off = (chunk * self.hv + head) * 8192
        threads = 32 * self.rw
        if const_expr(self.tk):
            kh = head // (self.hv // self.hk)
            _copy(
                _gmem(
                    k, (chunk * self.kh_store + (head if self.merge else kh)) * 8192, 64, 128, 128
                ),
                sw,
                threads,
                True,
            )
            if const_expr(self.v_fp32):
                _copy(_gmem(u, off + tile * self.bv, 64, self.bv, 128), sf, threads, True)
            else:
                _copy(_gmem(u, off + tile * self.bv, 64, self.bv, 128), sv, threads, True)
            _copy(
                _gmem(g, (chunk * self.hv + head) * 128 + 64, 1, 64, 64),
                cute.make_tensor(sb.iterator, cute.make_layout((1, 64), stride=(64, 1))),
                threads,
                True,
            )
        else:
            _copy(_gmem(w, off * (1 + self.w_split), 64, 128, 128), sw, threads, True)
            if const_expr(self.u_fp32):
                _copy(_gmem(u, off + tile * self.bv, 64, self.bv, 128), sf, threads, True)
            else:
                _copy(_gmem(u, off + tile * self.bv, 64, self.bv, 128), sv, threads, True)
        _copy(
            _gmem(g, (chunk * self.hv + head) * (128 if self.tk else 64), 1, 64, 64),
            cute.make_tensor(sg.iterator, cute.make_layout((1, 64), stride=(64, 1))),
            threads,
            True,
        )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def residual(self, w, kl, chunk, head, qo, h, ch, sh, v, co, s0, sv, so, sg, sb, mma):
        threads = 32 * self.rw
        _state_operand(h, ch, sh, prefetched=True)
        v.fill(0.0)
        _gemm(mma, s0, sh, v)
        if const_expr(self.split):
            _state_operand(h, ch, sh, True)
            _gemm(mma, s0, sh, v)
        if const_expr(not self.tk and self.w_split):
            cute.arch.sync_threads()
            _copy(
                _gmem(w, (chunk * self.hv + head) * 16384 + 8192, 64, 128, 128),
                s0,
                threads,
                True,
            )
            cute.arch.cp_async_commit_group()
            _wait()
            _gemm(mma, s0, sh, v)
            if const_expr(self.split):
                _state_operand(h, ch, sh)
                _gemm(mma, s0, sh, v)
        if const_expr(self.tk and self.k_split):
            cute.arch.sync_threads()
            _copy(_gmem(kl, qo, 64, 128, 128), s0, threads, True)
            cute.arch.cp_async_commit_group()
            _wait()
            _gemm(mma, s0, sh, v)
            if const_expr(self.split):
                _state_operand(h, ch, sh)
                _gemm(mma, s0, sh, v)
        for e in cutlass.range_constexpr(cute.size(co)):
            row, col = co[e]
            if const_expr(self.tk):
                value = Float32(sv[row, col])
                if const_expr(self.v_fp32):
                    value = so[row, col]
                v[e] = sb[row] * (value - _exp(sg[row]) * v[e])
            else:
                value = Float32(sv[row, col])
                if const_expr(self.u_fp32):
                    value = so[row, col]
                v[e] = value - v[e]

    @cute.jit
    def output_barrier(self):
        if const_expr(self.rec_ws):
            cute.arch.barrier(barrier_id=3, number_of_threads=128)
        else:
            cute.arch.sync_threads()

    @cute.jit
    def intra_output(
        self, mma, aq, chunk, head, sa, sv, vlo, co, o, output_threads: cutlass.Constexpr = 0
    ):
        """Three BF16 products; stream A_lo through the retired A_hi tile.

        Swap the V residual registers with shared V_hi for the middle product.
        Synchronous low-plane copies leave next-chunk cp.async groups in flight.
        The specialized schedule calls this only from its four output warps.
        """
        group = 128 if self.rec_ws else 0
        _gemm(mma, sa, _transpose(sv), o, tb=True, group_threads=group)
        if const_expr(self.aq_out):
            self.output_barrier()
            for e in cutlass.range_constexpr(cute.size(co)):
                row, col = co[e]
                hi = Float32(sv[row, col])
                sv[row, col] = BFloat16(vlo[e])
                vlo[e] = hi
            self.output_barrier()
            _gemm(mma, sa, _transpose(sv), o, tb=True, group_threads=group)
            self.output_barrier()
            _publish(vlo, co, sv)
            _copy(
                _gmem(aq, (chunk * self.hv + head) * 8192 + 4096, 64, 64, 64),
                sa,
                output_threads if output_threads else (128 if self.rec_ws else 32 * self.rw),
                group_threads=group,
            )
            self.output_barrier()
            _gemm(mma, sa, _transpose(sv), o, tb=True, group_threads=group)

    @cute.jit
    def recurrence2_loop(
        self,
        q,
        k,
        w,
        u,
        aq,
        g,
        out,
        partial,
        ready,
        storage,
        h,
        ch,
        co,
        mma,
        mh,
        first,
        nc,
        head,
        kh,
        tile,
        begin,
        end,
        state,
        offsets,
        rows,
        seq,
        hbuf,
        vbuf,
    ):
        """Lifetime-buffered WY pipeline (no cross-warp ownership assumptions).

        Compact: two W planes; retired W becomes Q, A/O_pre, K, then output.
        Full: single W/Q planes, two A/O_pre and K planes. After WH and QH,
        W/Q and U/H retire: next operands issue while this chunk executes
        output addition (legacy A*r) and the state update. Only the loop-entry
        wait drains cp.async. In pre form q/aq here denote Q'/O_pre pointers;
        Q'_lo streams through Q'_hi storage before that tile retires.
        o_split exports exact BF16 H / FP32 residual before weighting. Compact
        replaces Q with K and leaves W/U/g prefetch in flight across the update;
        full omits Q/A banks and prefetches only W/K/U/g.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane, wid = tid % 32, tid // 32
        threads = 32 * self.rw
        alloc = SmemAllocator()
        sq = cute.make_tensor(storage.iterator, _layout(64, 128))
        if const_expr(self.prefetch_depth == 4):
            if const_expr(not self.o_split):
                sq = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
                ab = alloc.allocate_tensor(
                    Float32 if self.pre else BFloat16,
                    cute.make_layout(128 * self.bv if self.pre else 8192),
                    byte_alignment=128,
                )
            kb = alloc.allocate_tensor(BFloat16, cute.make_layout(16384), byte_alignment=128)
        sh = alloc.allocate_tensor(BFloat16, _layout(self.bv, 128), byte_alignment=128)
        sv = alloc.allocate_tensor(BFloat16, _layout(64, self.bv), byte_alignment=128)
        sg = alloc.allocate_tensor(Float32, cute.make_layout((1, 64), stride=(64, 1)))
        su = cute.make_tensor(
            cute.recast_ptr(sh.iterator, dtype=u.dtype),
            cute.make_layout((64, self.bv), stride=(self.bv, 1)),
        )
        qplanes = 1 + self.qp_split if self.pre else 1
        v = cute.make_rmem_tensor(co.shape, Float32)
        o = cute.make_rmem_tensor(co.shape, Float32)
        vlo = cute.make_rmem_tensor(co.shape, Float32)
        if nc > 0:
            if const_expr(self.overlap):
                _ready_wait(ready + (first * self.hv + head))
            sw = cute.make_tensor(storage.iterator, _layout(64, 128))
            _copy(_gmem(w, (first * self.hv + head) * 8192, 64, 128, 128), sw, threads, True)
            _copy(
                _gmem(u, (first * self.hv + head) * 8192 + tile * self.bv, 64, self.bv, 128),
                su,
                threads,
                True,
            )
            _copy(_gmem(g, (first * self.hv + head) * 64, 1, 64, 64), sg, threads, True)
            if const_expr(self.prefetch_depth == 4):
                qo = (first * self.kh_store + (head if self.merge else kh)) * 8192
                qoff = (first * self.hv + head) * 8192 if self.pre else qo
                if const_expr(not self.o_split):
                    _copy(_gmem(q, qoff * qplanes, 64, 128, 128), sq, threads, True)
                _copy(
                    _gmem(k, qo, 64, 128, 128),
                    cute.make_tensor(kb.iterator, _layout(64, 128)),
                    threads,
                    True,
                )
                if const_expr(not self.o_split):
                    _copy(
                        _gmem(
                            aq,
                            (first * self.hv + head)
                            * (8192 if self.pre else 4096 * (2 if self.aq_out else 1))
                            + (tile * self.bv if self.pre else 0),
                            64,
                            self.bv if self.pre else 64,
                            128 if self.pre else 64,
                        ),
                        cute.make_tensor(
                            ab.iterator,
                            cute.make_layout((64, self.bv), stride=(self.bv, 1))
                            if self.pre
                            else _layout(64, 64),
                        ),
                        threads,
                        True,
                    )
            cute.arch.cp_async_commit_group()
        for local in range(nc):
            chunk = first + local
            qo = (chunk * self.kh_store + (head if self.merge else kh)) * 8192
            qoff = (chunk * self.hv + head) * 8192 if self.pre else qo
            bank = local % 2
            sw = cute.make_tensor(storage.iterator, _layout(64, 128))
            if const_expr(self.prefetch_depth == 1):
                sw = cute.make_tensor((storage.iterator + bank * 8192).align(128), _layout(64, 128))
                next_w = cute.make_tensor(
                    (storage.iterator + (1 - bank) * 8192).align(128), _layout(64, 128)
                )
                sq = sw
                sa = cute.make_tensor(
                    cute.recast_ptr(sw.iterator, dtype=Float32) if self.pre else sw.iterator,
                    cute.make_layout((64, self.bv), stride=(self.bv, 1))
                    if self.pre
                    else _layout(64, 64),
                )
                sk = sw
            else:
                if const_expr(not self.o_split):
                    sa = cute.make_tensor(
                        (ab.iterator + bank * (64 * self.bv if self.pre else 4096)).align(128),
                        cute.make_layout((64, self.bv), stride=(self.bv, 1))
                        if self.pre
                        else _layout(64, 64),
                    )
                sk = cute.make_tensor((kb.iterator + bank * 8192).align(128), _layout(64, 128))
            _wait()
            # U must be in registers before its storage becomes the H operand.
            for e in cutlass.range_constexpr(cute.size(co)):
                row, col = co[e]
                v[e] = Float32(su[row, col])
            _state_operand(h, ch, sh)
            if const_expr(self.o_split):
                _copy(
                    sh,
                    _gmem(
                        hbuf,
                        ((chunk * self.hv + head) * 128 + tile * self.bv) * 128,
                        self.bv,
                        128,
                        128,
                    ),
                    threads,
                )
            o.fill(0.0)
            _gemm(mma, sw, sh, o)
            if const_expr(self.split):
                _state_operand(h, ch, sh, True)
                _gemm(mma, sw, sh, o)
            for e in cutlass.range_constexpr(cute.size(co)):
                v[e] -= o[e]
                if const_expr(self.aq_out and not self.pre):
                    vlo[e] = v[e] - Float32(BFloat16(v[e]))
            if const_expr(self.o_split):
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    vbuf[(chunk * self.hv + head) * 8192 + row * 128 + tile * self.bv + col] = v[e]
            if const_expr(self.prefetch_depth == 1):
                cute.arch.sync_threads()  # WH retires before Q (or split K) overwrites W.
                if const_expr(self.o_split):
                    # K replaces Q in the retired W plane. Keep next W in flight.
                    _copy(_gmem(k, qo, 64, 128, 128), sk, threads, True)
                else:
                    _copy(_gmem(q, qoff * qplanes, 64, 128, 128), sq, threads, True)
                cute.arch.cp_async_commit_group()
                # Commit next W after current Q/K so wait_group(1) leaves W
                # in flight. The other plane has exclusive ownership.
                if local + 1 < nc:
                    if const_expr(self.overlap):
                        _ready_wait(ready + ((chunk + 1) * self.hv + head))
                    _copy(
                        _gmem(w, ((chunk + 1) * self.hv + head) * 8192, 64, 128, 128),
                        next_w,
                        threads,
                        True,
                    )
                    cute.arch.cp_async_commit_group()
            if const_expr(not self.pre and not self.o_split):
                _publish(v, co, sv)
            if const_expr(self.split):
                _state_operand(h, ch, sh)
            if const_expr(self.prefetch_depth == 1):
                if local + 1 < nc:
                    _wait(1)
                else:
                    _wait()
            # Full schedule needs no barrier here: Q/H already resident and sv
            # is first consumed only after the QH retirement barrier below.
            if const_expr(not self.o_split):
                o.fill(0.0)
                _gemm(mma, sq, sh, o)
                if const_expr(self.split):
                    _state_operand(h, ch, sh, True)
                    _gemm(mma, sq, sh, o)
                if const_expr(self.pre and self.qp_split):
                    cute.arch.sync_threads()
                    # Synchronous copy preserves compact next-W lookahead.
                    _copy(_gmem(q, qoff * 2 + 8192, 64, 128, 128), sq, threads)
                    if const_expr(self.split):
                        _state_operand(h, ch, sh)
                    cute.arch.sync_threads()
                    _gemm(mma, sq, sh, o)
                    if const_expr(self.split):
                        _state_operand(h, ch, sh, True)
                        _gemm(mma, sq, sh, o)
            for e in cutlass.range_constexpr(cute.size(co)):
                row, col = co[e]
                if const_expr(not self.pre and not self.o_split):
                    o[e] *= _exp(sg[0, row])
                v[e] *= _exp(sg[0, 63] - sg[0, row])
            decay = _exp(sg[0, 63])
            for e in cutlass.range_constexpr(cute.size(ch)):
                h[e] *= decay
            # Retire H snapshot and g readers before next U/g overwrite.
            cute.arch.sync_threads()  # Also publishes fused unweighted sv.
            if const_expr(self.prefetch_depth == 1 and not self.o_split):
                _copy(
                    _gmem(
                        aq,
                        (chunk * self.hv + head)
                        * (8192 if self.pre else 4096 * (2 if self.aq_out else 1))
                        + (tile * self.bv if self.pre else 0),
                        64,
                        self.bv if self.pre else 64,
                        128 if self.pre else 64,
                    ),
                    sa,
                    threads,
                    True,
                )
                cute.arch.cp_async_commit_group()
            if local + 1 < nc:
                if const_expr(self.prefetch_depth == 4):
                    if const_expr(self.overlap):
                        _ready_wait(ready + ((chunk + 1) * self.hv + head))
                    next_qo = ((chunk + 1) * self.kh_store + (head if self.merge else kh)) * 8192
                    _copy(
                        _gmem(w, ((chunk + 1) * self.hv + head) * 8192, 64, 128, 128),
                        sw,
                        threads,
                        True,
                    )
                    if const_expr(not self.o_split):
                        next_qoff = ((chunk + 1) * self.hv + head) * 8192 if self.pre else next_qo
                        _copy(
                            _gmem(q, next_qoff * qplanes, 64, 128, 128),
                            sq,
                            threads,
                            True,
                        )
                    _copy(
                        _gmem(k, next_qo, 64, 128, 128),
                        cute.make_tensor(
                            (kb.iterator + (1 - bank) * 8192).align(128), _layout(64, 128)
                        ),
                        threads,
                        True,
                    )
                    if const_expr(not self.o_split):
                        _copy(
                            _gmem(
                                aq,
                                ((chunk + 1) * self.hv + head)
                                * (8192 if self.pre else 4096 * (2 if self.aq_out else 1))
                                + (tile * self.bv if self.pre else 0),
                                64,
                                self.bv if self.pre else 64,
                                128 if self.pre else 64,
                            ),
                            cute.make_tensor(
                                (
                                    ab.iterator + (1 - bank) * (64 * self.bv if self.pre else 4096)
                                ).align(128),
                                cute.make_layout((64, self.bv), stride=(self.bv, 1))
                                if self.pre
                                else _layout(64, 64),
                            ),
                            threads,
                            True,
                        )
                _copy(
                    _gmem(
                        u, ((chunk + 1) * self.hv + head) * 8192 + tile * self.bv, 64, self.bv, 128
                    ),
                    su,
                    threads,
                    True,
                )
                _copy(_gmem(g, ((chunk + 1) * self.hv + head) * 64, 1, 64, 64), sg, threads, True)
                cute.arch.cp_async_commit_group()
            if const_expr(not self.o_split):
                if const_expr(self.prefetch_depth == 1):
                    if local + 1 < nc:
                        _wait(1)  # Keep next U/g in flight during the output MMA.
                    else:
                        _wait()
                if const_expr(self.pre):
                    for e in cutlass.range_constexpr(cute.size(co)):
                        row, col = co[e]
                        o[e] += sa[row, col]
                else:
                    self.intra_output(mma, aq, chunk, head, sa, sv, vlo, co, o)
                for e in cutlass.range_constexpr(cute.size(co)):
                    o[e] *= 0.08838834764831845
                cute.arch.sync_threads()  # A/r retire before weighted r and K/output.
            _publish(v, co, sv)
            if const_expr(self.prefetch_depth == 1 and not self.o_split):
                _copy(_gmem(k, qo, 64, 128, 128), sk, threads, True)
                cute.arch.cp_async_commit_group()
                _wait()
            else:
                cute.arch.sync_threads()
            _gemm(mh, _transpose(sv), _transpose(sk), h, ta=True, tb=True)
            if const_expr(self.update_split):
                cute.arch.sync_threads()
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    sv[row, col] = BFloat16(v[e] - Float32(BFloat16(v[e])))
                cute.arch.sync_threads()
                _gemm(mh, _transpose(sv), _transpose(sk), h, ta=True, tb=True)
            if const_expr(self.captures):
                for capture in cutlass.range_constexpr(self.captures):
                    p = offsets[Int64(seq) * self.captures + capture]
                    target = Int64(rows[Int64(seq) * self.captures + capture])
                    if target >= 0 and p > 0 and p <= end - begin and p == (Int64(local) + 1) * 64:
                        for e in cutlass.range_constexpr(cute.size(ch)):
                            vi, ki = ch[e]
                            state[
                                ((target * self.hv + head) * 128 + tile * self.bv + vi) * 128 + ki
                            ] = h[e]
            if const_expr(not self.o_split):
                # Current A (full) or K (compact) is now free for FP32 output.
                so = cute.make_tensor(
                    cute.recast_ptr(sa.iterator, dtype=Float32),
                    cute.make_layout((64, self.bv), stride=(self.bv, 1)),
                )
                if const_expr(self.prefetch_depth == 1):
                    cute.arch.sync_threads()  # Only compact output aliases the K consumer.
                if const_expr(self.o_bf16):
                    for e in cutlass.range_constexpr(cute.size(co)):
                        o[e] = Float32(BFloat16(o[e]))
                _publish(o, co, so)
                cute.arch.sync_threads()
                for row in range(wid, 64, self.rw):
                    token = begin + Int64(local) * 64 + row
                    square = Float32(0.0)
                    for j in cutlass.range_constexpr((self.bv + 31) // 32):
                        if lane + j * 32 < self.bv:
                            value = so[row, lane + j * 32]
                            square += value * value
                    square = _sum_warp(square)
                    if token < end:
                        if lane == 0:
                            partial[(token * self.hv + head) * max(4, 128 // self.bv) + tile] = (
                                square
                            )
                            if const_expr(self.bv == 64):
                                partial[(token * self.hv + head) * 4 + tile + 2] = Float32(0.0)
                        width = 128 // out.dtype.width
                        if lane < self.bv // width:
                            frag = cute.make_rmem_tensor((width,), out.dtype)
                            for j in cutlass.range_constexpr(width):
                                frag[j] = so[row, lane * width + j].to(out.dtype)
                            dst = _gmem(
                                out,
                                (token * self.hv + head) * 128 + tile * self.bv + lane * width,
                                1,
                                width,
                                width,
                            )
                            cute.copy(
                                cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(), out.dtype, num_bits_per_copy=128
                                ),
                                frag,
                                cute.make_tensor(dst.iterator, cute.make_layout(width)),
                            )
            # All output readers belong to their row's warp. The next iteration's
            # entry barrier retires them before any plane can be overwritten.
        # Final K/A/output readers and outstanding copies retire before state save.
        cute.arch.sync_threads()

    @cute.jit
    def recurrence_ws_loop(
        self,
        q,
        k,
        w,
        u,
        aq,
        g,
        out,
        partial,
        ready,
        storage,
        h,
        ch,
        co,
        mma,
        mh,
        first,
        nc,
        head,
        kh,
        tile,
        begin,
        end,
    ):
        """Four chain warps and four output warps with distinct residual planes.

        Barrier 1 (256): H/input publication and residual publication.
        Barrier 2 (128): chain-only weighted residual reuse.
        Barrier 3 (128): output-only transpose before norm partials.
        The final CTA rendezvous retires all readers before next-chunk copies.
        No subgroup ever calls a CTA-wide barrier inside its divergent branch.
        """
        tid, _, _ = cute.arch.thread_idx()
        lane, wid = tid % 32, (tid % 128) // 32
        alloc = SmemAllocator()
        sw = cute.make_tensor(storage.iterator, _layout(64, 128))
        sq = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
        sk = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
        sa = alloc.allocate_tensor(BFloat16, _layout(64, 64), byte_alignment=128)
        sh = alloc.allocate_tensor(BFloat16, _layout(self.bv, 128), byte_alignment=128)
        sv = alloc.allocate_tensor(BFloat16, _layout(64, self.bv), byte_alignment=128)
        weighted = alloc.allocate_tensor(BFloat16, _layout(64, self.bv), byte_alignment=128)
        so = alloc.allocate_tensor(
            Float32, cute.make_layout((64, self.bv), stride=(self.bv, 1)), byte_alignment=128
        )
        su = cute.make_tensor(
            cute.recast_ptr(so.iterator, dtype=u.dtype),
            cute.make_layout((64, self.bv), stride=(self.bv, 1)),
        )
        sg = alloc.allocate_tensor(Float32, cute.make_layout((1, 64), stride=(64, 1)))
        v = cute.make_rmem_tensor(co.shape, Float32)
        o = cute.make_rmem_tensor(co.shape, Float32)
        vlo = cute.make_rmem_tensor(co.shape, Float32)
        for local in range(nc):
            chunk = first + local
            qo = (chunk * self.kh_store + (head if self.merge else kh)) * 8192
            if const_expr(self.overlap):
                _ready_wait(ready + (chunk * self.hv + head))
            _copy(_gmem(w, (chunk * self.hv + head) * 8192, 64, 128, 128), sw, 256, True)
            _copy(_gmem(q, qo, 64, 128, 128), sq, 256, True)
            _copy(_gmem(k, qo, 64, 128, 128), sk, 256, True)
            _copy(
                _gmem(aq, (chunk * self.hv + head) * 4096 * (2 if self.aq_out else 1), 64, 64, 64),
                sa,
                256,
                True,
            )
            _copy(
                _gmem(u, (chunk * self.hv + head) * 8192 + tile * self.bv, 64, self.bv, 128),
                su,
                256,
                True,
            )
            _copy(_gmem(g, (chunk * self.hv + head) * 64, 1, 64, 64), sg, 256, True)
            cute.arch.cp_async_commit_group()
            if tid < 128:
                _publish(h, ch, sh)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            o.fill(0.0)
            if tid < 128:
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    v[e] = Float32(su[row, col])
                _gemm(mma, sw, sh, o, group_threads=128)
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    v[e] -= o[e]
                    sv[row, col] = BFloat16(v[e])
                    if const_expr(self.aq_out):
                        so[row, col] = v[e] - Float32(BFloat16(v[e]))
                    v[e] *= _exp(sg[0, 63] - sg[0, row])
                    weighted[row, col] = BFloat16(v[e])
                for e in cutlass.range_constexpr(cute.size(ch)):
                    h[e] *= _exp(sg[0, 63])
            else:
                _gemm(mma, sq, sh, o, group_threads=128)
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    o[e] *= _exp(sg[0, row])
            # Both H consumers retire; both residual planes are now visible.
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            if tid < 128:
                _gemm(
                    mh, _transpose(weighted), _transpose(sk), h, ta=True, tb=True, group_threads=128
                )
                if const_expr(self.update_split):
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    for e in cutlass.range_constexpr(cute.size(co)):
                        row, col = co[e]
                        weighted[row, col] = BFloat16(v[e] - Float32(BFloat16(v[e])))
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    _gemm(
                        mh,
                        _transpose(weighted),
                        _transpose(sk),
                        h,
                        ta=True,
                        tb=True,
                        group_threads=128,
                    )
            else:
                if const_expr(self.aq_out):
                    for e in cutlass.range_constexpr(cute.size(co)):
                        row, col = co[e]
                        vlo[e] = so[row, col]
                self.intra_output(mma, aq, chunk, head, sa, sv, vlo, co, o)
                for e in cutlass.range_constexpr(cute.size(co)):
                    o[e] *= 0.08838834764831845
                    if const_expr(self.o_bf16):
                        o[e] = Float32(BFloat16(o[e]))
                _publish(o, co, so)
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                for row in range(wid, 64, 4):
                    token = begin + Int64(local) * 64 + row
                    square = Float32(0.0)
                    if lane < self.bv:
                        value = so[row, lane]
                        square = value * value
                    square = _sum_warp(square)
                    if token < end:
                        if lane == 0:
                            partial[(token * self.hv + head) * (128 // self.bv) + tile] = square
                        width = 128 // out.dtype.width
                        if lane < self.bv // width:
                            frag = cute.make_rmem_tensor((width,), out.dtype)
                            for j in cutlass.range_constexpr(width):
                                frag[j] = so[row, lane * width + j].to(out.dtype)
                            dst = _gmem(
                                out,
                                (token * self.hv + head) * 128 + tile * self.bv + lane * width,
                                1,
                                width,
                                width,
                            )
                            cute.copy(
                                cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(), out.dtype, num_bits_per_copy=128
                                ),
                                frag,
                                cute.make_tensor(dst.iterator, cute.make_layout(width)),
                            )
            cute.arch.barrier(barrier_id=1, number_of_threads=256)

    @cute.kernel
    def recurrence(
        self,
        x: cute.Pointer,
        nw: cute.Pointer,
        cu: cute.Pointer,
        slots: cute.Pointer,
        initial: cute.Pointer,
        state: cute.Pointer,
        history: cute.Pointer,
        q: cute.Pointer,
        k: cute.Pointer,
        w: cute.Pointer,
        u: cute.Pointer,
        aq: cute.Pointer,
        g: cute.Pointer,
        hout: cute.Pointer,
        out: cute.Pointer,
        partial: cute.Pointer,
        y: cute.Pointer,
        kl: cute.Pointer,
        ready: cute.Pointer,
        qp: cute.Pointer,
        op: cute.Pointer,
        offsets,
        rows,
        hbuf: cute.Pointer,
        vbuf: cute.Pointer,
        tokens: Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        tile = Int32(bx) % (128 // self.bv)
        head = (Int32(bx) // (128 // self.bv)) % self.hv
        seq = Int32(bx) // ((128 // self.bv) * self.hv)
        tid, _, _ = cute.arch.thread_idx()
        lane, wid = tid % 32, tid // 32
        threads = 32 * self.rw
        kh = head // (self.hv // self.hk)
        slot = Int64(slots[seq])
        begin, end = Int64(cu[seq]), Int64(cu[seq + 1])
        first = Int64(0)
        for s in range(seq):
            first += (Int64(cu[s + 1]) - Int64(cu[s]) + 63) // 64
        nc = Int32((end - begin + 63) // 64)
        alloc = SmemAllocator()
        s0 = alloc.allocate_tensor(
            BFloat16,
            _layout(
                64,
                256 if self.rec_mode == 2 and self.prefetch_depth == 1 and not self.rec_ws else 128,
            ),
            byte_alignment=128,
        )
        if const_expr(self.rec_mode != 2):
            s1 = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
            sh = alloc.allocate_tensor(BFloat16, _layout(self.bv, 128), byte_alignment=128)
            sv = alloc.allocate_tensor(BFloat16, _layout(64, self.bv), byte_alignment=128)
            sg = alloc.allocate_tensor(Float32, cute.make_layout(64))
            sums = alloc.allocate_tensor(Float32, cute.make_layout(64))
            if const_expr(self.tk):
                sb = alloc.allocate_tensor(Float32, cute.make_layout(64))
            if const_expr(self.rec_mode):
                output_a = alloc.allocate_tensor(BFloat16, _layout(64, 64), byte_alignment=128)
                output_v = alloc.allocate_tensor(BFloat16, _layout(64, self.bv), byte_alignment=128)
            # H operands retire after Q H. Its storage then holds r_lo during T r.
            srlo = cute.make_tensor(sh.iterator, _layout(64, self.bv))
            # s0 holds K/W, then T/A; s1 holds V/Q/T staging, K, then output.
            sa = cute.make_tensor(s0.iterator, _layout(64, 64))
            so = cute.make_tensor(
                cute.recast_ptr(s1.iterator, dtype=Float32),
                cute.make_layout((64, self.bv), stride=(self.bv, 1)),
            )
        compute_warps = 4 if self.rec_ws else self.rw
        mma = _mma(compute_warps, self.tm, self.tn)
        mh = mma
        if const_expr(self.bv == 16):
            mma = _mma(compute_warps, max(self.tm, compute_warps * 8), 16)
            mh = _mma(compute_warps, 16, max(self.tn, compute_warps * 8))
        compute_tid = tid % 128 if self.rec_ws else tid
        thread = mma.get_slice(compute_tid)
        ch = mh.get_slice(compute_tid).partition_C(cute.make_identity_tensor((self.bv, 128)))
        co = thread.partition_C(cute.make_identity_tensor((64, self.bv)))
        h = cute.make_rmem_tensor(ch.shape, Float32)
        v = cute.make_rmem_tensor(co.shape, Float32)
        o = cute.make_rmem_tensor(co.shape, Float32)
        vlo = cute.make_rmem_tensor(co.shape, Float32)
        h.fill(0.0)
        state_rows = min(32, self.bv)
        # Stage 32 state rows at a time through the idle operand buffer, using
        # 128-bit global transactions. Element-aligned pools retain a scalar path.
        state_tile = cute.make_tensor(
            cute.recast_ptr(s0.iterator, dtype=Float32),
            cute.make_layout((state_rows, 128), stride=(128, 1)),
        )
        if slot >= 0 and initial[seq] != 0:
            for part in cutlass.range_constexpr(self.bv // state_rows):
                off = ((slot * self.hv + head) * 128 + tile * self.bv + part * state_rows) * 128
                if state.toint() % 16 == 0:
                    _copy(_gmem(state, off, state_rows, 128, 128), state_tile, threads)
                else:
                    for e in range(tid, state_rows * 128, threads):
                        state_tile[e // 128, e % 128] = state[off + e]
                cute.arch.sync_threads()
                for e in cutlass.range_constexpr(cute.size(ch)):
                    vi, ki = ch[e]
                    if vi >= part * state_rows and vi < (part + 1) * state_rows:
                        h[e] = state_tile[vi - part * state_rows, ki]
                cute.arch.sync_threads()
        if const_expr(self.rec_ws):
            self.recurrence_ws_loop(
                q,
                k,
                w,
                u,
                aq,
                g,
                out,
                partial,
                ready,
                s0,
                h,
                ch,
                co,
                mma,
                mh,
                first,
                nc,
                head,
                kh,
                tile,
                begin,
                end,
            )
        elif const_expr(self.rec_mode == 2):
            self.recurrence2_loop(
                qp if self.pre else q,
                k,
                w,
                u,
                op if self.pre else aq,
                g,
                out,
                partial,
                ready,
                s0,
                h,
                ch,
                co,
                mma,
                mh,
                first,
                nc,
                head,
                kh,
                tile,
                begin,
                end,
                state,
                offsets,
                rows,
                seq,
                hbuf,
                vbuf,
            )
        else:
            if nc > 0:
                if const_expr(self.overlap):
                    _ready_wait(ready + (first * self.hv + head))
                self.prefetch(
                    k, w, u, g, s0, sv, sg, sb if self.tk else None, so, first, head, tile
                )
            for local in range(nc):
                chunk = first + local
                qo = (chunk * self.kh_store + (head if self.merge else kh)) * 8192
                if const_expr(self.rec_mode):
                    if local == 0:
                        self.residual(
                            w,
                            kl,
                            chunk,
                            head,
                            qo,
                            h,
                            ch,
                            sh,
                            v,
                            co,
                            s0,
                            sv,
                            so,
                            sg,
                            sb if self.tk else None,
                            mma,
                        )
                else:
                    self.residual(
                        w,
                        kl,
                        chunk,
                        head,
                        qo,
                        h,
                        ch,
                        sh,
                        v,
                        co,
                        s0,
                        sv,
                        so,
                        sg,
                        sb if self.tk else None,
                        mma,
                    )
                cute.arch.sync_threads()
                _publish(v, co, sv)
                _copy(_gmem(q, qo, 64, 128, 128), s1, threads, True)
                cute.arch.cp_async_commit_group()
                # Compute Q H before T r, so its H storage can hold r_lo.
                if const_expr(
                    self.split
                    and not ((self.tk and self.k_split) or (not self.tk and self.w_split))
                ):
                    _state_operand(h, ch, sh)
                _wait()
                o.fill(0.0)
                _gemm(mma, s1, sh, o)
                if const_expr(self.split):
                    _state_operand(h, ch, sh, True)
                    _gemm(mma, s1, sh, o)
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    o[e] *= _exp(sg[row])
                cute.arch.sync_threads()
                if const_expr(self.tk):
                    if const_expr(self.r_split):
                        for e in cutlass.range_constexpr(cute.size(co)):
                            row, col = co[e]
                            srlo[row, col] = BFloat16(v[e] - Float32(BFloat16(v[e])))
                    # Retired K and Q tiles hold T hi/lo and optional FP32 staging.
                    stlo = cute.make_tensor(s0.iterator + 4096, _layout(64, 64))
                    toff = (chunk * self.hv + head) * 4096
                    if const_expr(self.t_store == "fp32"):
                        stf = cute.make_tensor(
                            cute.recast_ptr(s1.iterator, dtype=Float32),
                            cute.make_layout((64, 64), stride=(64, 1)),
                        )
                        _copy(_gmem(w, toff, 64, 64, 64), stf, threads, True)
                        cute.arch.cp_async_commit_group()
                        _wait()
                        for e in range(tid, 4096, threads):
                            row, col = e // 64, e % 64
                            tvalue = stf[row, col]
                            hi = BFloat16(tvalue)
                            sa[row, col] = hi
                            stlo[row, col] = BFloat16(tvalue - Float32(hi))
                        cute.arch.sync_threads()
                    else:
                        _copy(
                            _gmem(w, toff * (2 if self.t_store == "split" else 1), 64, 64, 64),
                            sa,
                            threads,
                            True,
                        )
                        if const_expr(self.t_store == "split"):
                            _copy(_gmem(w, toff * 2 + 4096, 64, 64, 64), stlo, threads, True)
                        cute.arch.cp_async_commit_group()
                        _wait()
                    v.fill(0.0)
                    _gemm(mma, sa, _transpose(sv), v, tb=True)
                    if const_expr(self.r_split):
                        _gemm(mma, sa, _transpose(srlo), v, tb=True)
                    if const_expr(self.t_store != "bf16"):
                        _gemm(mma, stlo, _transpose(sv), v, tb=True)
                        if const_expr(self.r_split):
                            _gemm(mma, stlo, _transpose(srlo), v, tb=True)
                    cute.arch.sync_threads()
                    _publish(v, co, sv)
                    cute.arch.sync_threads()
                if const_expr(self.aq_out):
                    for e in cutlass.range_constexpr(cute.size(co)):
                        vlo[e] = v[e] - Float32(BFloat16(v[e]))
                if const_expr(self.rec_mode):
                    # Q H has consumed the old state. Preserve the unweighted BF16
                    # residual for output while the chain consumes its decayed value.
                    _copy(sv, output_v, threads)
                    _copy(
                        _gmem(
                            aq,
                            (chunk * self.hv + head) * 4096 * (2 if self.aq_out else 1),
                            64,
                            64,
                            64,
                        ),
                        output_a,
                        threads,
                        True,
                    )
                    _copy(_gmem(k, qo, 64, 128, 128), s1, threads, True)
                    cute.arch.cp_async_commit_group()
                else:
                    _copy(
                        _gmem(
                            aq,
                            (chunk * self.hv + head) * 4096 * (2 if self.aq_out else 1),
                            64,
                            64,
                            64,
                        ),
                        sa,
                        threads,
                        True,
                    )
                    cute.arch.cp_async_commit_group()
                    _wait()
                    # K fetch overlaps the intra-chunk output product.
                    _copy(_gmem(k, qo, 64, 128, 128), s1, threads, True)
                    cute.arch.cp_async_commit_group()
                    self.intra_output(mma, aq, chunk, head, sa, sv, vlo, co, o)
                    for e in cutlass.range_constexpr(cute.size(co)):
                        o[e] *= 0.08838834764831845
                # Weight residuals for the state update; avoid a second K tile/copy.
                cute.arch.sync_threads()
                for e in cutlass.range_constexpr(cute.size(co)):
                    row, col = co[e]
                    v[e] *= _exp(sg[63] - sg[row])
                    sv[row, col] = BFloat16(v[e])
                decay = _exp(sg[63])
                for e in cutlass.range_constexpr(cute.size(ch)):
                    h[e] *= decay
                _wait()
                _gemm(mh, _transpose(sv), _transpose(s1), h, ta=True, tb=True)
                if const_expr(self.update_split):
                    # Split only AFTER FP32 decay multiplication. Reuse the V tile
                    # for its residual; K low terms follow if requested.
                    cute.arch.sync_threads()
                    for e in cutlass.range_constexpr(cute.size(co)):
                        row, col = co[e]
                        sv[row, col] = BFloat16(v[e] - Float32(BFloat16(v[e])))
                    cute.arch.sync_threads()
                    _gemm(mh, _transpose(sv), _transpose(s1), h, ta=True, tb=True)
                if const_expr(self.k_split):
                    cute.arch.sync_threads()
                    _copy(_gmem(kl, qo, 64, 128, 128), s1, threads, True)
                    cute.arch.cp_async_commit_group()
                    _wait()
                    _gemm(mh, _transpose(sv), _transpose(s1), h, ta=True, tb=True)
                    if const_expr(self.update_split):
                        cute.arch.sync_threads()
                        _publish(v, co, sv)
                        cute.arch.sync_threads()
                        _gemm(mh, _transpose(sv), _transpose(s1), h, ta=True, tb=True)
                cute.arch.sync_threads()
                if const_expr(self.rec_mode):
                    if local + 1 < nc:
                        if const_expr(self.overlap):
                            _ready_wait(ready + ((chunk + 1) * self.hv + head))
                        self.prefetch(
                            k,
                            w,
                            u,
                            g,
                            s0,
                            sv,
                            sg,
                            sb if self.tk else None,
                            so,
                            chunk + 1,
                            head,
                            tile,
                        )
                        next_qo = (
                            (chunk + 1) * self.kh_store + (head if self.merge else kh)
                        ) * 8192
                        self.residual(
                            w,
                            kl,
                            chunk + 1,
                            head,
                            next_qo,
                            h,
                            ch,
                            sh,
                            v,
                            co,
                            s0,
                            sv,
                            so,
                            sg,
                            sb if self.tk else None,
                            mma,
                        )
                    self.intra_output(mma, aq, chunk, head, output_a, output_v, vlo, co, o)
                    for e in cutlass.range_constexpr(cute.size(co)):
                        o[e] *= 0.08838834764831845
                    if local + 1 < nc:
                        # All FP32 U reads must retire before so becomes output-c.
                        # Keep this AFTER the independent output MMA, not between
                        # the next WH and the current output instruction streams.
                        cute.arch.sync_threads()
                if const_expr(self.o_bf16):
                    for e in cutlass.range_constexpr(cute.size(co)):
                        o[e] = Float32(BFloat16(o[e]))
                _publish(o, co, so)
                cute.arch.sync_threads()
                for row in range(wid, 64, self.rw):
                    square = Float32(0.0)
                    for j in cutlass.range_constexpr((self.bv + 31) // 32):
                        if lane + j * 32 < self.bv:
                            out_value = so[row, lane + j * 32]
                            square += out_value * out_value
                    square = _sum_warp(square)
                    if lane == 0:
                        sums[row] = square
                for row in range(wid, 64, self.rw):
                    token = begin + Int64(local) * 64 + row
                    if token < end:
                        if lane == 0:
                            partial[(token * self.hv + head) * max(4, 128 // self.bv) + tile] = (
                                sums[row]
                            )
                            if const_expr(self.bv == 64):
                                partial[(token * self.hv + head) * 4 + tile + 2] = Float32(0.0)
                        width = 128 // out.dtype.width
                        if lane < self.bv // width:
                            frag = cute.make_rmem_tensor((width,), out.dtype)
                            for j in cutlass.range_constexpr(width):
                                frag[j] = so[row, lane * width + j].to(out.dtype)
                            dst = _gmem(
                                out,
                                (token * self.hv + head) * 128 + tile * self.bv + lane * width,
                                1,
                                width,
                                width,
                            )
                            cute.copy(
                                cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(), out.dtype, num_bits_per_copy=128
                                ),
                                frag,
                                cute.make_tensor(dst.iterator, cute.make_layout(width)),
                            )
                cute.arch.sync_threads()
                if const_expr(not self.rec_mode):
                    # s1's output is retired before staging next chunk's FP32 V there.
                    if local + 1 < nc:
                        if const_expr(self.overlap):
                            _ready_wait(ready + ((chunk + 1) * self.hv + head))
                        self.prefetch(
                            k,
                            w,
                            u,
                            g,
                            s0,
                            sv,
                            sg,
                            sb if self.tk else None,
                            so,
                            chunk + 1,
                            head,
                            tile,
                        )
        if const_expr(self.captures and not self.overlap):
            if tile == 0 and nc > 0:
                self.capture_history(
                    x, history, offsets, rows, initial, seq, slot, begin, end, head, threads
                )
                # Some threads export history elements other threads will commit.
                cute.arch.sync_threads()
        if slot >= 0 and (nc > 0 or initial[seq] == 0):
            for part in cutlass.range_constexpr(self.bv // state_rows):
                for e in cutlass.range_constexpr(cute.size(ch)):
                    vi, ki = ch[e]
                    if vi >= part * state_rows and vi < (part + 1) * state_rows:
                        if not self.rec_ws or tid < 128:
                            state_tile[vi - part * state_rows, ki] = h[e]
                cute.arch.sync_threads()
                off = ((slot * self.hv + head) * 128 + tile * self.bv + part * state_rows) * 128
                if state.toint() % 16 == 0:
                    _copy(state_tile, _gmem(state, off, state_rows, 128, 128), threads)
                else:
                    for e in range(tid, state_rows * 128, threads):
                        state[off + e] = state_tile[e // 128, e % 128]
                cute.arch.sync_threads()
            if tile == 0 and nc == 0 and initial[seq] == 0 and (not self.overlap or tokens == 0):
                self.zero_history(history, slot, head, threads)
            if const_expr(not self.overlap):
                if tile == 0 and nc > 0:
                    # K1 has completed globally; its staged history is safe to commit.
                    self.history_commit(history, hout, seq, slot, 2 * self.hk * 128 + head * 128)
                    if head % (self.hv // self.hk) == 0:
                        for part in cutlass.range_constexpr(2):
                            self.history_commit(
                                history, hout, seq, slot, (part * self.hk + kh) * 128
                            )
        if const_expr(self.overlap):
            # All threads wait only after their last stores. Norm may then commit
            # history: every prepare reader, including shared key heads, has retired.
            if tokens > 0:
                cute.arch.griddepcontrol_wait()

    @cute.kernel
    def output(
        self,
        cu: cute.Pointer,
        q: cute.Pointer,
        aq: cute.Pointer,
        g: cute.Pointer,
        hbuf: cute.Pointer,
        vbuf: cute.Pointer,
        out: cute.Pointer,
        partial: cute.Pointer,
        count: Int32,
    ):
        chunk_index, head, tile = cute.arch.block_idx()
        seq, local, begin, end = _owner(cu, chunk_index, count)
        if seq >= 0:
            chunk = Int64(chunk_index)
            tid, _, _ = cute.arch.thread_idx()
            lane, wid = tid % 32, tid // 32
            threads = 32 * self.ow
            kh = head // (self.hv // self.hk)
            qo = (chunk * self.kh_store + (head if self.merge else kh)) * 8192
            alloc = SmemAllocator()
            sq = alloc.allocate_tensor(BFloat16, _layout(64, 128), byte_alignment=128)
            sh = alloc.allocate_tensor(BFloat16, _layout(self.bvo, 128), byte_alignment=128)
            sv = alloc.allocate_tensor(BFloat16, _layout(64, self.bvo), byte_alignment=128)
            sa = alloc.allocate_tensor(BFloat16, _layout(64, 64), byte_alignment=128)
            sg = alloc.allocate_tensor(Float32, cute.make_layout((1, 64), stride=(64, 1)))
            # H's allocation holds FP32 residuals before H arrives, then output
            # after QH retires. All three views occupy exactly 256*BVO bytes.
            sf = cute.make_tensor(
                cute.recast_ptr(sh.iterator, dtype=Float32),
                cute.make_layout((64, self.bvo), stride=(self.bvo, 1)),
            )
            mma = _mma(self.ow, self.tm, self.tn)
            co = mma.get_slice(tid).partition_C(cute.make_identity_tensor((64, self.bvo)))
            o = cute.make_rmem_tensor(co.shape, Float32)
            vlo = cute.make_rmem_tensor(co.shape, Float32)
            _copy(
                _gmem(vbuf, (chunk * self.hv + head) * 8192 + tile * self.bvo, 64, self.bvo, 128),
                sf,
                threads,
                True,
            )
            _copy(_gmem(q, qo, 64, 128, 128), sq, threads, True)
            _copy(
                _gmem(aq, (chunk * self.hv + head) * 4096 * (1 + self.aq_out), 64, 64, 64),
                sa,
                threads,
                True,
            )
            _copy(_gmem(g, (chunk * self.hv + head) * 64, 1, 64, 64), sg, threads, True)
            cute.arch.cp_async_commit_group()
            _wait()
            for e in cutlass.range_constexpr(cute.size(co)):
                row, col = co[e]
                value = sf[row, col]
                sv[row, col] = BFloat16(value)
                if const_expr(self.aq_out):
                    vlo[e] = value - Float32(BFloat16(value))
            cute.arch.sync_threads()  # FP32 residual readers retire before H overwrites sf.
            _copy(
                _gmem(
                    hbuf,
                    ((chunk * self.hv + head) * 128 + tile * self.bvo) * 128,
                    self.bvo,
                    128,
                    128,
                ),
                sh,
                threads,
                True,
            )
            cute.arch.cp_async_commit_group()
            _wait()
            o.fill(0.0)
            _gemm(mma, sq, sh, o)
            for e in cutlass.range_constexpr(cute.size(co)):
                row, col = co[e]
                o[e] *= _exp(sg[0, row])
            self.intra_output(mma, aq, chunk, head, sa, sv, vlo, co, o, threads)
            for e in cutlass.range_constexpr(cute.size(co)):
                o[e] *= 0.08838834764831845
                if const_expr(self.o_bf16):
                    o[e] = Float32(BFloat16(o[e]))
            cute.arch.sync_threads()  # QH readers retire before shared H becomes output.
            _publish(o, co, sf)
            cute.arch.sync_threads()
            for row in range(wid, 64, self.ow):
                token = begin + Int64(local) * 64 + row
                # Keep the fused BV groups and butterfly order, independent of BVO.
                # BVO<BV is BV64/BVO32 only: norm reconstructs those two groups
                # directly from out once both CTAs have finished (ordinary launch).
                if const_expr(self.bvo >= self.bv):
                    for part in cutlass.range_constexpr(self.bvo // self.bv):
                        square = Float32(0.0)
                        for j in cutlass.range_constexpr((self.bv + 31) // 32):
                            if lane + j * 32 < self.bv:
                                value = sf[row, part * self.bv + lane + j * 32]
                                square += value * value
                        square = _sum_warp(square)
                        if token < end and lane == 0:
                            index = tile * (self.bvo // self.bv) + part
                            partial[(token * self.hv + head) * max(4, 128 // self.bv) + index] = (
                                square
                            )
                            if const_expr(self.bv == 64):
                                partial[(token * self.hv + head) * 4 + index + 2] = Float32(0.0)
                if token < end:
                    width = 128 // out.dtype.width
                    if lane < self.bvo // width:
                        frag = cute.make_rmem_tensor((width,), out.dtype)
                        for j in cutlass.range_constexpr(width):
                            frag[j] = sf[row, lane * width + j].to(out.dtype)
                        dst = _gmem(
                            out,
                            (token * self.hv + head) * 128 + tile * self.bvo + lane * width,
                            1,
                            width,
                            width,
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(), out.dtype, num_bits_per_copy=128
                            ),
                            frag,
                            cute.make_tensor(dst.iterator, cute.make_layout(width)),
                        )

    @cute.kernel
    def norm(
        self,
        x: cute.Pointer,
        nw: cute.Pointer,
        out: cute.Pointer,
        partial: cute.Pointer,
        y: cute.Pointer,
        tokens: Int32,
        count: Int32,
        cu: cute.Pointer,
        slots: cute.Pointer,
        history: cute.Pointer,
        hout: cute.Pointer,
        initial,
        offsets,
        rows,
    ):
        block, head, _ = cute.arch.block_idx()
        self.norm_rows(x, nw, out, partial, y, Int64(block) * 8, tokens, head)
        if const_expr(self.overlap):
            # Recurrence waited for prepare: every history reader has retired.
            # Stride over N so even T=N with fewer norm blocks commits every row.
            for seq in range(block, count, (Int64(tokens) + 7) // 8):
                slot = Int64(slots[seq])
                if const_expr(self.captures):
                    self.capture_history(
                        x,
                        history,
                        offsets,
                        rows,
                        initial,
                        seq,
                        slot,
                        Int64(cu[seq]),
                        Int64(cu[seq + 1]),
                        head,
                        128,
                    )
                    cute.arch.sync_threads()
                if slot >= 0 and cu[seq + 1] == cu[seq] and initial[seq] == 0:
                    self.zero_history(history, slot, head, 128)
                if slot >= 0 and cu[seq + 1] > cu[seq]:
                    self.history_commit(history, hout, seq, slot, (2 * self.hk + head) * 128, 128)
                    if head % (self.hv // self.hk) == 0:
                        kh = head // (self.hv // self.hk)
                        for part in cutlass.range_constexpr(2):
                            self.history_commit(
                                history, hout, seq, slot, (part * self.hk + kh) * 128, 128
                            )

    @cute.jit
    def norm_rows(self, x, nw, out, partial, y, start, tokens, head):
        tid, _, _ = cute.arch.thread_idx()
        token, lane = start + tid // 16, tid % 16
        total = Float32(0.0)
        if const_expr(not (self.o_split and self.bvo < self.bv)):
            if token < tokens and lane == 0:
                # Pad the two-part (BV64) case to a four-float scratch stride.
                parts = cute.make_rmem_tensor((max(4, 128 // self.bv),), Float32)
                _load_vector(
                    partial, (Int64(token) * self.hv + head) * max(4, 128 // self.bv), parts
                )
                for tile in cutlass.range_constexpr(128 // self.bv):
                    total += parts[tile]
        # Every lane participates, including the final partially valid warp.
        total = cute.arch.shuffle_sync(total, offset=(tid % 32) // 16 * 16)
        if const_expr(self.o_split and self.bvo < self.bv):
            # Emulate each BV64 warp's first butterfly in two half-warp
            # registers, then the remaining four steps within each row.
            # Invalid tail rows still participate in every shuffle.
            total = Float32(0.0)
            for tile in cutlass.range_constexpr(2):
                left, right = Float32(0.0), Float32(0.0)
                if token < tokens:
                    base = (Int64(token) * self.hv + head) * 128 + tile * 64 + lane
                    for j in cutlass.range_constexpr(2):
                        a = Float32(out[base + j * 32])
                        b = Float32(out[base + 16 + j * 32])
                        left += a * a
                        right += b * b
                square = left + right
                for step in cutlass.range_constexpr(4):
                    square += cute.arch.shuffle_sync_bfly(square, offset=8 >> step)
                total += square
        if token < tokens:
            inv = cute.rsqrt(total * (1.0 / 128.0) + 1.0e-6)
            frag = cute.make_rmem_tensor((8,), BFloat16)
            zvec = cute.make_rmem_tensor((8,), BFloat16)
            nvec = cute.make_rmem_tensor((8,), Float32)
            ovec = cute.make_rmem_tensor((8,), out.dtype)
            _load_vector(x, Int64(token) * self.stride + self.dc + head * 128 + lane * 8, zvec)
            _load_vector(nw, lane * 8, nvec)
            _load_vector(out, (Int64(token) * self.hv + head) * 128 + lane * 8, ovec)
            for j in cutlass.range_constexpr(8):
                frag[j] = BFloat16(Float32(ovec[j]) * inv * nvec[j] * _silu(Float32(zvec[j])))
            dst = _gmem(y, (Int64(token) * self.hv + head) * 128 + lane * 8, 1, 8, 8)
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128),
                frag,
                cute.make_tensor(dst.iterator, cute.make_layout(8)),
            )
