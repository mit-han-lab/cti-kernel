# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Ported from cti's CuTe RTX PRO 6000 pair solution (accepted tree 5d6a2ce7); protocol unchanged.
"""CuTe DSL kernels for the all-reduce + residual add + RMSNorm epilogue on
GPUs without peer access: the ranks exchange their partial sums through pinned host memory
that every rank maps, and each kernel sums, normalizes and stores in the same pass.

Protocol (after cti's earlier Triton version of this exchange; rewritten in CuTe DSL):

* A rank writes its partial rows into its own mailbox in host memory (16-byte write-through
  stores), then a CTA barrier, then one thread runs ``fence.acq_rel.sys`` (cumulative over
  the barrier) and stores the progress word ``(call tag << 24) | rows published``.
* A reader's one thread acquire-polls the peer's progress word until it covers the rows it
  needs (or shows the peer already in its next call), a CTA barrier, then every thread loads
  the peer's rows with ``ld.global.cv`` (never a cached line of another GPU's writes).
* The call tag: a counter array in device memory with one slot per SM. CTA ``b`` reads slot
  ``b`` at its start and at its end writes the new count to slots ``b, b + G, b + 2G, ...``,
  so every slot advances once per call whatever the grid ``G``: every CTA of a call sees the
  same tag, on every rank (same calls). Mailboxes are indexed by row and double-buffered by
  the tag's parity: a rank rewrites a region two calls later, after its peer has read it (the
  call in between completes only after the peer published that call's rows, which it does
  only after finishing the earlier call).
* h = residual + p_0 + p_1 + ... in FP32 in rank order on every rank, the sum of squares in a
  fixed order, rstd = 1 / sqrt(ss / H + eps), x = h * rstd * g with g = 1 + w
  (zero-centered) or w; both outputs rounded once to BF16, so every rank stores the same bits.
"""

# Preserve the accepted DSL loops, including their positional zip operations.
# ruff: noqa: B007, B905

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm

# Progress words sit this many bytes apart, so no two CTAs share a host cache line.
PROG_BYTES = 128
# Bits of a progress word below the call tag: the rows published.
TAG_SHIFT = 24
ROWS_MASK = (1 << TAG_SHIFT) - 1


def _i64():
    return ir.IntegerType.get_signless(64)


def globaltimer() -> Int64:
    return Int64(nvvm.read_ptx_sreg_globaltimer(_i64()))


def _ptr(dtype, addr, align):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def _v4():
    return ir.VectorType.get([4], Int32.mlir_type)


def load_raw(addr, cop=None):
    """Four 32-bit words (eight BF16) at ``addr``, one 16-byte load."""
    return cute.arch.load(_ptr(Int32, addr, 16), _v4(), cop=cop)


def store_raw(addr, value, cop=None):
    cute.arch.store(_ptr(Int32, addr, 16), value, cop=cop)


def to_f32(raw):
    """Four words of BF16 pairs -> an FP32 TensorSSA of 8 (element order kept)."""
    return cute.TensorSSA(raw, (4,), Int32).bitcast(BFloat16).to(Float32)


def store_bf16(addr, values):
    """Round eight FP32 values to BF16 once and store them as one 16-byte vector."""
    cute.arch.store(_ptr(Int32, addr, 16), values.to(BFloat16).bitcast(Int32))


def put(tensor, index, value):
    """``tensor[index] = value`` (a call, so a kernel's if/while regions do not rebind
    ``tensor``)."""
    tensor[index] = value


def peer_addr(host, safe, ok):
    """The address a lane reads a peer's vector from: ``host``, or for a lane past the end of
    the row (``ok`` false) ``safe``, a device address, so masked lanes never issue host reads
    (they would all read the same clamped vector over the link)."""
    if ok is None:
        return host
    return safe + Int64(ok) * (host - safe)


def both(a, b):
    """The conjunction of two lane / row predicates, either of which may be None (always)."""
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def covers(seen: Int64, tag: Int64, count: Int64):
    """Whether progress word ``seen`` shows ``count`` rows of call ``tag``, or the writer
    already in call ``tag + 1`` (it starts that call only after publishing all of call
    ``tag``; it cannot be further ahead, since its call ``tag + 2`` waits for this rank)."""
    seen_tag = seen >> TAG_SHIFT
    return ((seen_tag == tag) & ((seen & ROWS_MASK) >= count)) | (seen_tag == tag + 1)


def gains_of(wa, vecs, zero_centered):
    """The FP32 norm gains of a thread's vectors: ``1 + w`` (zero-centered) or ``w``."""
    if zero_centered:
        return [to_f32(load_raw(wa + off)) + Float32(1.0) for off, _ in vecs]
    return [to_f32(load_raw(wa + off)) for off, _ in vecs]


class _RowOps:
    """A CTA's view of one row: NT threads, 16-byte vectors v = tid + u * NT (u < VPT), the
    last partial step clamped to the row (loads) and masked (stores, sums)."""

    def __init__(self, H, NT):
        self.H, self.NT = H, NT
        self.NW = NT // 32
        self.NVEC = H // 8
        self.VPT = -(-self.NVEC // NT)

    def vectors(self, tidx):
        out = []
        for u in range(self.VPT):
            if (u + 1) * self.NT <= self.NVEC:
                out.append((Int64(tidx + u * self.NT) * 16, None))
            else:
                ok = tidx + u * self.NT < self.NVEC
                vc = cutlass.min(tidx + u * self.NT, self.NVEC - 1)
                out.append((Int64(vc) * 16, ok))
        return out


class FlagKernel:
    """R rows per CTA (calls of at most R rows per SM): CTA ``b`` pushes rows bR .. bR + R - 1
    to this rank's host mailbox (with ``pace``: row r not before ``r * pace_ns`` into the
    call, so this GPU's uploads enter the link at about its rate instead of queueing ahead of
    its own polls and downloads), publishes them with one fence and one progress word, waits
    for every peer's copy, sums in rank order, normalizes and stores in place."""

    def __init__(
        self, H, NT, world, rank, slots, rank_stride, par_stride, row_stride, pace, R=1,
        zero_centered=True,
    ):  # fmt: skip
        self.R = R
        self.zero_centered = zero_centered
        self.ops = _RowOps(H, NT)
        self.H, self.NT, self.world, self.rank, self.slots = H, NT, world, rank, slots
        self.rank_stride, self.par_stride, self.row_stride = rank_stride, par_stride, row_stride
        self.pace = pace

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mC: cute.Tensor,
        box: Int64,
        prog: Int64,
        rows: Int32,
        grid: Int32,
        eps: Float32,
        pace_ns: Int64,
        stream,
    ):
        # Programmatic dependent launch: the grid is launched while the preceding kernel
        # finishes and waits for it (griddepcontrol.wait) before touching memory, so only the
        # launch overlaps (measured 1-5% faster at every flag bucket).
        self.kernel(mX, mR, mW, mC, box, prog, rows, grid, eps, pace_ns).launch(
            grid=[grid, 1, 1], block=[self.NT, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mC: cute.Tensor,
        box: Int64,
        prog: Int64,
        rows: Int32,
        grid: Int32,
        eps: Float32,
        pace_ns: Int64,
    ):
        cute.arch.griddepcontrol_wait()
        ops = self.ops
        rank, world = self.rank, self.world
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        smem = cutlass.utils.SmemAllocator()
        R = self.R
        red = smem.allocate_tensor(Float32, cute.make_layout(R * ops.NW), 16)

        t_start = Int64(0)
        if const_expr(self.pace):
            t_start = globaltimer()
        tag = mC[bidx] + Int64(1)
        par = tag & Int64(1)
        xbase = mX.iterator.toint()
        rbase = mR.iterator.toint()
        xstride = Int64(mX.stride[0] * 2)
        rstride = Int64(mR.stride[0] * 2)
        vecs = ops.vectors(tidx)
        boxes = [box + Int64(q * self.rank_stride) + par * self.par_stride for q in range(world)]
        my_prog = _ptr(
            Int64, prog + Int64((rank * self.slots) * PROG_BYTES) + Int64(bidx) * PROG_BYTES, 8
        )

        def hold_until(until):
            """Spin on the global timer until it reaches ``until`` (ns)."""
            now = globaltimer()
            while now < until:
                now = globaltimer()

        def store_rows(base, values, row_ok, cop=None):
            for (off, ok), value in zip(vecs, values):
                pred = both(ok, row_ok)
                if const_expr(pred is None):
                    store_raw(base + off, value, cop=cop)
                else:
                    if pred:
                        store_raw(base + off, value, cop=cop)

        def normalize_store(hs, gains, xa, ra, i, row_ok):
            """Sum of squares (per thread in vector order, butterfly over the warp, warps in
            order), rstd, and the two BF16 stores of the row."""
            ss = Float32(0.0)
            for (off, ok), h in zip(vecs, hs):
                part = (h * h).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                if const_expr(ok is None):
                    ss = ss + part
                else:
                    ss = ss + part * Float32(ok)
            ss = cute.arch.warp_reduction_sum(ss)
            if lane == 0:
                put(red, i * ops.NW + warp, ss)
            cute.arch.sync_threads()
            total = Float32(0.0)
            for w in cutlass.range_constexpr(ops.NW):
                total = total + red[i * ops.NW + w]
            rstd = Float32(1.0) / cute.math.sqrt(total / Float32(ops.H) + eps)
            for (off, ok), h, g in zip(vecs, hs, gains):
                y = h * rstd * g
                pred = both(ok, row_ok)
                if const_expr(pred is None):
                    store_bf16(ra + off, h)
                    store_bf16(xa + off, y)
                else:
                    if pred:
                        store_bf16(ra + off, h)
                        store_bf16(xa + off, y)

        def push(count):
            """Upload this CTA's rows of this rank's partial sums, then publish ``count``."""
            for i in cutlass.range_constexpr(R):
                row, rc, row_ok = rows_of[i]
                src = xbase + Int64(rc) * xstride
                own = [load_raw(src + off) for off, _ in vecs]
                if const_expr(self.pace):
                    hold_until(t_start + Int64(row) * pace_ns)
                store_rows(boxes[rank] + Int64(row) * self.row_stride, own, row_ok, cop="wt")
            cute.arch.sync_threads()
            if tidx == 0:
                cute.arch.fence_acq_rel_sys()
                cute.arch.store(
                    my_prog, (tag << TAG_SHIFT) | Int64(count), sem="relaxed", scope="sys"
                )

        def wait(count):
            """One thread acquire-polls every peer's progress word; the barrier orders the
            other threads' later loads after its acquire."""
            if tidx == 0:
                for q in cutlass.range_constexpr(world):
                    if const_expr(q != rank):
                        peer = _ptr(
                            Int64,
                            prog + Int64((q * self.slots) * PROG_BYTES) + Int64(bidx) * PROG_BYTES,
                            8,
                        )
                        seen = cute.arch.load(peer, Int64, sem="acquire", scope="sys")
                        while not covers(seen, tag, Int64(count)):
                            seen = cute.arch.load(peer, Int64, sem="acquire", scope="sys")
            cute.arch.sync_threads()

        def reduce_row(i, gains):
            row, rc, row_ok = rows_of[i]
            xa = xbase + Int64(rc) * xstride
            ra = rbase + Int64(rc) * rstride
            res = [load_raw(ra + off) for off, _ in vecs]
            own = [load_raw(xa + off) for off, _ in vecs]
            hs = [to_f32(r) for r in res]
            for q in cutlass.range_constexpr(world):
                if const_expr(q == rank):
                    hs = [h + to_f32(o) for h, o in zip(hs, own)]
                else:
                    src = boxes[q] + Int64(row) * self.row_stride
                    hs = [
                        h
                        + to_f32(
                            load_raw(peer_addr(src + off, ra + off, both(ok, row_ok)), cop="cv")
                        )
                        for h, (off, ok) in zip(hs, vecs)
                    ]
            normalize_store(hs, gains, xa, ra, i, row_ok)

        # This CTA's rows: (row, the row clamped into the call for device loads, whether it
        # is in the call; None when it always is).
        rows_of = []
        for i in cutlass.range_constexpr(R):
            if const_expr(R == 1):
                rows_of.append((bidx, bidx, None))
            else:
                row = bidx * R + i
                rows_of.append((row, cutlass.min(row, rows - 1), row < rows))
        wa = mW.iterator.toint()
        push(1)
        gains = gains_of(wa, vecs, self.zero_centered)
        wait(1)
        for i in cutlass.range_constexpr(R):
            reduce_row(i, gains)
        # The call's tag to counter slots bidx, bidx + grid, ... (each slot once per call), one
        # slot per thread (a 1-CTA call would otherwise write all of them from one thread).
        j = bidx + tidx * grid
        while j < self.slots:
            put(mC, j, tag)
            j = j + self.NT * grid


class PCKernel:
    """Producer/consumer (calls of more than one row per SM): CTAs b < P only push rows
    p, p + P, ... (paced: row r not before r * pace_ns into the call) and publish after each;
    CTAs b >= P only wait, sum, normalize and store. Consumer c owns rows c, c + C, ... with C
    = K * P, all pushed by producer c % P, so it follows one progress word per peer, and one
    local word of that producer before it overwrites x (the producer must have read it)."""

    def __init__(
        self, H, NT, world, rank, slots, rank_stride, par_stride, row_stride, pace,
        zero_centered=True,
    ):  # fmt: skip
        self.zero_centered = zero_centered
        self.ops = _RowOps(H, NT)
        self.H, self.NT, self.world, self.rank, self.slots = H, NT, world, rank, slots
        self.rank_stride, self.par_stride, self.row_stride = rank_stride, par_stride, row_stride
        self.pace = pace

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mC: cute.Tensor,
        mL: cute.Tensor,
        box: Int64,
        prog: Int64,
        rows: Int32,
        producers: Int32,
        grid: Int32,
        eps: Float32,
        pace_ns: Int64,
        stream,
    ):
        self.kernel(mX, mR, mW, mC, mL, box, prog, rows, producers, grid, eps, pace_ns).launch(
            grid=[grid, 1, 1], block=[self.NT, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mC: cute.Tensor,
        mL: cute.Tensor,
        box: Int64,
        prog: Int64,
        rows: Int32,
        producers: Int32,
        grid: Int32,
        eps: Float32,
        pace_ns: Int64,
    ):
        ops = self.ops
        rank, world = self.rank, self.world
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(ops.NW), 16)

        t_start = Int64(0)
        if const_expr(self.pace):
            t_start = globaltimer()
        tag = mC[bidx] + Int64(1)
        par = tag & Int64(1)
        xbase = mX.iterator.toint()
        rbase = mR.iterator.toint()
        xstride = Int64(mX.stride[0] * 2)
        rstride = Int64(mR.stride[0] * 2)
        vecs = ops.vectors(tidx)
        boxes = [box + Int64(q * self.rank_stride) + par * self.par_stride for q in range(world)]
        P = producers
        if bidx < P:
            my_prog = _ptr(
                Int64, prog + Int64((rank * self.slots) * PROG_BYTES) + Int64(bidx) * PROG_BYTES, 8
            )
            n_my = (rows - bidx + P - 1) // P
            for i in range(n_my):
                row = bidx + i * P
                src = xbase + Int64(row) * xstride
                own = [load_raw(src + off) for off, _ in vecs]
                if const_expr(self.pace):
                    until = t_start + Int64(row) * pace_ns
                    now = globaltimer()
                    while now < until:
                        now = globaltimer()
                dst = boxes[rank] + Int64(row) * self.row_stride
                for (off, ok), value in zip(vecs, own):
                    if const_expr(ok is None):
                        store_raw(dst + off, value, cop="wt")
                    else:
                        if ok:
                            store_raw(dst + off, value, cop="wt")
                cute.arch.sync_threads()
                if tidx == 0:
                    word = (tag << TAG_SHIFT) | Int64(i + 1)
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.store(my_prog, word, sem="relaxed", scope="sys")
                    # Local progress: a strong store behind the fence (release pattern).
                    cute.arch.store(
                        _ptr(Int64, mL.iterator.toint() + Int64(bidx) * 8, 8),
                        word,
                        sem="relaxed",
                        scope="gpu",
                    )
        else:
            c = bidx - P
            C = grid - P
            p = c % P
            wa = mW.iterator.toint()
            gains = gains_of(wa, vecs, self.zero_centered)
            n_my = (rows - c + C - 1) // C
            for i in range(n_my):
                row = c + i * C
                need = Int64(row // P + 1)
                xa = xbase + Int64(row) * xstride
                ra = rbase + Int64(row) * rstride
                res = [load_raw(ra + off) for off, _ in vecs]
                own = [load_raw(xa + off) for off, _ in vecs]
                if tidx == 0:
                    for q in cutlass.range_constexpr(world):
                        if const_expr(q != rank):
                            peer = _ptr(
                                Int64,
                                prog + Int64((q * self.slots) * PROG_BYTES) + Int64(p) * PROG_BYTES,
                                8,
                            )
                            seen = cute.arch.load(peer, Int64, sem="acquire", scope="sys")
                            while not covers(seen, tag, need):
                                seen = cute.arch.load(peer, Int64, sem="acquire", scope="sys")
                cute.arch.sync_threads()
                hs = [to_f32(r) for r in res]
                for q in cutlass.range_constexpr(world):
                    if const_expr(q == rank):
                        hs = [h + to_f32(o) for h, o in zip(hs, own)]
                    else:
                        src = boxes[q] + Int64(row) * self.row_stride
                        hs = [
                            h + to_f32(load_raw(peer_addr(src + off, ra + off, ok), cop="cv"))
                            for h, (off, ok) in zip(hs, vecs)
                        ]
                ss = Float32(0.0)
                for (off, ok), h in zip(vecs, hs):
                    part = (h * h).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    if const_expr(ok is None):
                        ss = ss + part
                    else:
                        ss = ss + part * Float32(ok)
                ss = cute.arch.warp_reduction_sum(ss)
                if lane == 0:
                    put(red, warp, ss)
                # This rank's producer must have read x for the row before it is overwritten.
                if tidx == 0:
                    mine = _ptr(Int64, mL.iterator.toint() + Int64(p) * 8, 8)
                    lseen = cute.arch.load(mine, Int64, sem="acquire", scope="gpu")
                    while not covers(lseen, tag, need):
                        lseen = cute.arch.load(mine, Int64, sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                total = Float32(0.0)
                for w in cutlass.range_constexpr(ops.NW):
                    total = total + red[w]
                rstd = Float32(1.0) / cute.math.sqrt(total / Float32(ops.H) + eps)
                for (off, ok), h, g in zip(vecs, hs, gains):
                    y = h * rstd * g
                    if const_expr(ok is None):
                        store_bf16(ra + off, h)
                        store_bf16(xa + off, y)
                    else:
                        if ok:
                            store_bf16(ra + off, h)
                            store_bf16(xa + off, y)
                # ``red`` is rewritten by the next row.
                cute.arch.sync_threads()
        j = bidx + tidx * grid
        while j < self.slots:
            put(mC, j, tag)
            j = j + self.NT * grid


class CalibKernel:
    """The link's rate with every rank uploading and downloading at once, as the exchange
    runs: a handshake through host memory (each rank's CTA 0 publishes ``tag`` and waits for
    every peer's, then releases its other CTAs), then even CTAs store ``chunk`` bytes each to
    this rank's mailbox and odd CTAs load ``chunk`` bytes each of a peer's; ``mT`` gets the
    earliest start and the latest end (global timer, ns)."""

    def __init__(self, NT, world, rank):
        self.NT, self.world, self.rank = NT, world, rank

    @cute.jit
    def __call__(
        self,
        mT: cute.Tensor,
        own: Int64,
        peer: Int64,
        flags: Int64,
        chunk: Int64,
        tag: Int64,
        grid: Int32,
        stream,
    ):
        self.kernel(mT, own, peer, flags, chunk, tag).launch(
            grid=[grid, 1, 1], block=[self.NT, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self, mT: cute.Tensor, own: Int64, peer: Int64, flags: Int64, chunk: Int64, tag: Int64
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        go = _ptr(Int64, mT.iterator.toint() + 24, 8)
        if tidx == 0:
            if bidx == 0:
                mine = _ptr(Int64, flags + Int64(self.rank * PROG_BYTES), 8)
                cute.arch.store(mine, tag, sem="release", scope="sys")
                # A peer that never arrives (2 s) marks the run failed instead of hanging.
                deadline = globaltimer() + Int64(2_000_000_000)
                for q in cutlass.range_constexpr(self.world):
                    if const_expr(q != self.rank):
                        theirs = _ptr(Int64, flags + Int64(q * PROG_BYTES), 8)
                        seen = cute.arch.load(theirs, Int64, sem="acquire", scope="sys")
                        now = globaltimer()
                        while seen < tag and now < deadline:
                            seen = cute.arch.load(theirs, Int64, sem="acquire", scope="sys")
                            now = globaltimer()
                        if seen < tag:
                            put(mT, 5, Int64(1))
                cute.arch.store(go, tag, sem="release", scope="gpu")
            seen_go = cute.arch.load(go, Int64, sem="acquire", scope="gpu")
            while seen_go < tag:
                seen_go = cute.arch.load(go, Int64, sem="acquire", scope="gpu")
        cute.arch.sync_threads()
        t0 = globaltimer()
        half = bidx // 2
        steps = chunk // (self.NT * 16)
        acc = Int32(0)
        if bidx % 2 == 0:
            base = own + Int64(half) * chunk + Int64(tidx) * 16
            value = load_raw(mT.iterator.toint())
            for i in range(steps):
                store_raw(base + Int64(i) * (self.NT * 16), value, cop="wt")
        else:
            base = peer + Int64(half) * chunk + Int64(tidx) * 16
            for i in range(steps):
                words = cute.TensorSSA(
                    load_raw(base + Int64(i) * (self.NT * 16), cop="cv"), (4,), Int32
                )
                acc = acc + words.reduce(cute.ReductionOp.ADD, Int32(0), 0)
        cute.arch.sync_threads()
        if tidx == 0:
            cute.arch.fence_acq_rel_sys()
            t1 = globaltimer()
            cute.arch.atomic_min(_ptr(Int64, mT.iterator.toint(), 8), t0)
            cute.arch.atomic_max(_ptr(Int64, mT.iterator.toint() + 8, 8), t1)
        if acc == Int32(0x7FFFFFFF):
            put(mT, 4, Int64(acc))
