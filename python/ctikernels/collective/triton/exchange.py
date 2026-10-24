# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Exchange kernels: a producer epilogue publishes its partial tiles to every peer's inbox and
raises one flag per tile; the consumer sums the partials of each row in rank order (bitwise the
same on every rank), optionally adds the residual and applies RMSNorm, and rounds once.

Protocol:

* Descriptor: an int64 device tensor (``DESC_*`` indices) holding the call tag's address, the
  consumer's arrival counter, the shared abort word, this rank's error word, the slot size in
  bytes, the flags per writer, the spin timeout, the largest call (M * N) that publishes
  (larger ones go through NCCL), the largest call on the word protocol, the capacity (rows,
  columns; producers publish nothing beyond it), then one inbox base and one flag base per receiver, as this
  rank addresses them (host mailboxes or peer device memory).
* Calls of at most that many elements use self-validating 8-byte words (two values and the
  call tag), with no fence or flag: the consumer polls the data itself.
* Inbox of a receiver: ``[parity 2][peer W - 1][slot_bytes]``, a partial stored densely
  (``row * N + col``). Flags of a receiver: ``[peer W - 1][flag_cap]`` int32. A writer's peer
  index at receiver ``r`` is its rank, minus one above ``r``.
* A producer reads the call tag ``t`` (it never waits), stores its tile into every peer's inbox
  at slot ``(t & 1, rank)``, then a CTA barrier, ``fence.acq_rel.sys`` and the tile's flag = ``t``
  by one thread. Its own partial stays in the caller's output tensor.
* The consumer's program of a row polls the row's tile flags of every peer until each shows
  ``t`` or later (wrap-safe signed difference), then a fence and a CTA barrier, then reads the
  peer rows with ``ld.global.cv``. The last program to arrive (a device counter) advances the tag
  for the next call, so every rank's calls carry the same tags, eager or replayed.
* Slots double-buffer by tag parity: a writer stores call ``t + 2`` only after its consumer of
  ``t + 1`` saw this rank's ``t + 1`` flags, raised after this rank finished reading ``t``.
* A nonzero abort word (any rank's host may set it) or the timeout ends every spin; the program
  then sets this rank's error word and skips its stores.
"""

from __future__ import annotations

import triton
import triton.language as tl

DESC_TAG = tl.constexpr(0)
DESC_DONE = tl.constexpr(1)
DESC_ABORT = tl.constexpr(2)
DESC_ERROR = tl.constexpr(3)
DESC_SLOT_BYTES = tl.constexpr(4)
DESC_FLAG_CAP = tl.constexpr(5)
DESC_TIMEOUT = tl.constexpr(6)
DESC_PUBLISH_MAX = tl.constexpr(7)
DESC_LL_MAX = tl.constexpr(8)
DESC_MAX_ROWS = tl.constexpr(9)
DESC_MAX_COLS = tl.constexpr(10)
DESC_INBOX = tl.constexpr(11)  # then W inbox bases, then W flag bases


@triton.jit
def _release_flag(addr, tag):
    """One thread: fence.acq_rel.sys (cumulative over the preceding CTA barrier), then the flag."""
    tl.inline_asm_elementwise(
        """
        {
        .reg .pred p;
        .reg .u32 t;
        mov.u32 t, %tid.x;
        setp.eq.u32 p, t, 0;
        @p fence.acq_rel.sys;
        @p st.relaxed.sys.global.b32 [$1], $2;
        mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [addr, tag],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _acquire_fence():
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def publish(xch, values, row0, col0, M, N, tile, RANK: tl.constexpr, WORLD: tl.constexpr):
    """Epilogue hook: send this rank's partial tile ``values`` [BM, BN] (already in the output
    dtype) of rows ``row0 + i`` and columns ``col0 + j`` of the call's [M, N] output to every
    peer; out-of-range rows and columns are masked here.

    The caller still stores ``values`` into its own output, which the consumer reads as this
    rank's partial. ``tile`` is ``(row // TILE_M) * cdiv(N, TILE_N) + col // TILE_N`` for the
    ``TILE_M x TILE_N`` tiling the caller passes to the consumer (BM x BN, or rows of it);
    every tile is published exactly once per call. ``col0`` is a multiple of BN, ``N`` of 8.
    Call it at the end of the program.
    """
    BM: tl.constexpr = values.shape[0]
    BN: tl.constexpr = values.shape[1]
    N = tl.multiple_of(N, 8)
    # Calls the exchange routes to NCCL, or that exceed its capacity (the host's reduce then
    # raises on every rank alike), publish nothing.
    fits = (M <= tl.load(xch + DESC_MAX_ROWS)) & (N <= tl.load(xch + DESC_MAX_COLS))
    if fits & (M * N <= tl.load(xch + DESC_PUBLISH_MAX)):
        rows = (row0 + tl.arange(0, BM)).to(tl.int64)
        tag = tl.load(tl.load(xch + DESC_TAG).to(tl.pointer_type(tl.int32)))
        slot_bytes = tl.multiple_of(tl.load(xch + DESC_SLOT_BYTES), 4096)
        if M * N <= tl.load(xch + DESC_LL_MAX):
            # Two values and the tag in one 8-byte word: the word validates itself.
            pairs = tl.reshape(values.to(tl.int16, bitcast=True), [BM, BN // 2, 2])
            lo, hi = tl.split(pairs)
            low = (lo.to(tl.int32) & 0xFFFF) | (hi.to(tl.int32) << 16)
            word = (tag.to(tl.int64) << 32) | (low.to(tl.int64) & 0xFFFFFFFF)
            wcols = col0 // 2 + tl.arange(0, BN // 2)
            woffs = rows[:, None] * (N // 2) + wcols[None, :]
            wmask = (rows < M)[:, None] & (wcols < N // 2)[None, :]
            for r in tl.static_range(WORLD):
                if r != RANK:
                    peer = RANK - (RANK > r)
                    slot = ((tag & 1) * (WORLD - 1) + peer).to(tl.int64) * slot_bytes
                    dst = (tl.load(xch + DESC_INBOX + r) + slot).to(tl.pointer_type(tl.int64))
                    tl.store(tl.multiple_of(dst, 16) + woffs, word, mask=wmask)
        else:
            cols = col0 + tl.arange(0, BN)
            offs = rows[:, None] * N + cols[None, :]
            mask = (rows < M)[:, None] & (cols < N)[None, :]
            flag_cap = tl.load(xch + DESC_FLAG_CAP)
            if tile < flag_cap:
                for r in tl.static_range(WORLD):
                    if r != RANK:
                        peer = RANK - (RANK > r)
                        slot = ((tag & 1) * (WORLD - 1) + peer).to(tl.int64) * slot_bytes
                        inbox = tl.load(xch + DESC_INBOX + r) + slot
                        inbox = inbox.to(tl.pointer_type(values.dtype))
                        tl.store(tl.multiple_of(inbox, 16) + offs, values, mask=mask)
                tl.debug_barrier()
                for r in tl.static_range(WORLD):
                    if r != RANK:
                        peer = RANK - (RANK > r)
                        flag = tl.load(xch + DESC_INBOX + WORLD + r) + (peer * flag_cap + tile) * 4
                        _release_flag(flag, tag)


@triton.jit
def _stopped(polls, start, abort_ptr, timeout):
    """Every 64th poll: the abort word or the timeout."""
    stop = False
    if polls % 64 == 63:
        stop = tl.load(abort_ptr, volatile=True) != 0
        if timeout > 0:
            stop = stop | (_globaltimer() - start > timeout)
    return stop


@triton.jit
def _wait_tiles(flags, live, tag, abort_ptr, timeout):
    """Spin until every live flag shows ``tag`` or later; returns 1 on abort or timeout."""
    start = _globaltimer()
    failed = 0
    polls = 0
    seen = tl.load(flags, mask=live, other=tag, volatile=True)
    pending = tl.sum(((seen - tag) < 0).to(tl.int32), axis=0)
    while pending != 0:
        if _stopped(polls, start, abort_ptr, timeout):
            pending = 0
            failed = 1
        else:
            seen = tl.load(flags, mask=live, other=tag, volatile=True)
            pending = tl.sum(((seen - tag) < 0).to(tl.int32), axis=0)
        polls += 1
    return failed


@triton.jit
def _wait_words(words, live, tag, abort_ptr, timeout):
    """Spin until every live word carries ``tag``; returns the words' low halves (two values
    each) and 1 on abort or timeout."""
    start = _globaltimer()
    failed = 0
    polls = 0
    seen = tl.load(words, mask=live, other=0, volatile=True)
    pending = tl.sum((live & ((seen >> 32).to(tl.int32) != tag)).to(tl.int32), axis=0)
    while pending != 0:
        if _stopped(polls, start, abort_ptr, timeout):
            pending = 0
            failed = 1
        else:
            seen = tl.load(words, mask=live, other=0, volatile=True)
            pending = tl.sum((live & ((seen >> 32).to(tl.int32) != tag)).to(tl.int32), axis=0)
        polls += 1
    return seen.to(tl.int32), failed


@triton.jit
def _start(residual, offs, live, HAS_RESIDUAL: tl.constexpr, BLOCK_N: tl.constexpr):
    """The FP32 row sum's start: the residual, or zeros."""
    if HAS_RESIDUAL:
        h = tl.load(residual + offs, mask=live, other=0.0).to(tl.float32)
    else:
        h = tl.zeros((BLOCK_N,), dtype=tl.float32)
    return h


@triton.jit
def _release_gpu(addr, value):
    """One thread: fence.acq_rel.gpu (cumulative over the preceding CTA barrier), then the word."""
    tl.inline_asm_elementwise(
        """
        {
        .reg .pred p;
        .reg .u32 t;
        mov.u32 t, %tid.x;
        setp.eq.u32 p, t, 0;
        @p fence.acq_rel.gpu;
        @p st.relaxed.gpu.global.b32 [$1], $2;
        mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [addr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _row_total(sums, marks, row, part, ss, tag, abort_ptr, timeout, PARTS: tl.constexpr):
    """Sum of a row's per-part ``ss`` in part order, on each of its PARTS programs (all
    resident: the host keeps the grid within the SMs). Each part publishes its sum, then marks
    it with the call tag; tags only grow, so the marks need no reset. Returns the total and
    1 on abort or timeout."""
    tl.store(sums + row * PARTS + part, ss)
    tl.debug_barrier()
    _release_gpu(marks + row * PARTS + part, tag)
    parts = row * PARTS + tl.arange(0, PARTS)
    start = _globaltimer()
    failed = 0
    polls = 0
    seen = tl.load(marks + parts, volatile=True)
    pending = tl.sum((seen != tag).to(tl.int32), axis=0)
    while pending != 0:
        if _stopped(polls, start, abort_ptr, timeout):
            pending = 0
            failed = 1
        else:
            seen = tl.load(marks + parts, volatile=True)
            pending = tl.sum((seen != tag).to(tl.int32), axis=0)
        polls += 1
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    return tl.sum(tl.load(sums + parts, cache_modifier=".cg"), axis=0), failed


@triton.jit
def _unpack(low, dtype: tl.constexpr, BLOCK_N: tl.constexpr):
    """Words' low halves (two 16-bit values each) -> BLOCK_N values in order, as FP32."""
    lo = (low & 0xFFFF).to(tl.int16).to(dtype, bitcast=True)
    hi = ((low >> 16) & 0xFFFF).to(tl.int16).to(dtype, bitcast=True)
    return tl.reshape(tl.join(lo, hi), [BLOCK_N]).to(tl.float32)


# Width and strides stay unspecialized: specializing them makes Triton hand each thread several
# adjacent words, and volatile loads do not vectorize, so host reads lose coalescing (2.5x slower
# at M=1 on the RTX PRO 6000 pair).
@triton.jit(
    do_not_specialize=[
        "M",
        "N",
        "stride_own",
        "stride_out",
        "stride_res",
        "tile_m",
        "tile_n",
        "eps",
    ]
)
def _reduce_kernel(
    xch,
    own,
    out,
    residual,
    weight,
    sums,
    marks,
    M,
    N,
    stride_own,
    stride_out,
    stride_res,
    tile_m,
    tile_n,
    eps,
    CHUNK: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    NORM: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Program (row, part) owns CHUNK columns of a row: h = [residual] + p_0 + ... + p_{W-1}
    in FP32, rank order; then out = h (rounded once) or, with ``NORM``, out = h * rstd * gain
    and residual = h, the row's sum of squares totalled over its PARTS programs in part order."""
    row = tl.program_id(0)
    piece = tl.program_id(1)
    N = tl.multiple_of(N, 8)
    stride_own = tl.multiple_of(stride_own, 8)
    stride_out = tl.multiple_of(stride_out, 8)
    stride_res = tl.multiple_of(stride_res, 8)
    tag_ptr = tl.load(xch + DESC_TAG).to(tl.pointer_type(tl.int32))
    done_ptr = tl.load(xch + DESC_DONE).to(tl.pointer_type(tl.int32))
    tag = tl.load(tag_ptr)
    slot_bytes = tl.multiple_of(tl.load(xch + DESC_SLOT_BYTES), 4096)
    flag_cap = tl.load(xch + DESC_FLAG_CAP)
    abort_ptr = tl.load(xch + DESC_ABORT).to(tl.pointer_type(tl.int32))
    timeout = tl.load(xch + DESC_TIMEOUT)
    inbox = tl.multiple_of(tl.load(xch + DESC_INBOX + RANK), 4096)
    flag_base = tl.load(xch + DESC_INBOX + WORLD + RANK).to(tl.pointer_type(tl.int32))

    cols = piece * CHUNK + tl.arange(0, CHUNK)
    live = cols < N
    row64 = row.to(tl.int64)
    half = piece * (CHUNK // 2) + tl.arange(0, CHUNK // 2)
    ll = M * N <= tl.load(xch + DESC_LL_MAX)
    # The residual and the gain come from earlier kernels, complete before the producer
    # started: read them while the peers' data is in flight.
    h0 = _start(residual, row64 * stride_res + cols, live, HAS_RESIDUAL, CHUNK)
    if NORM:
        gain = tl.load(weight + cols, mask=live, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            gain += 1.0
    if WORLD == 2 and ll:
        # One peer: poll its words before waiting for the producer (with PDL, this kernel
        # starts while the producer drains); then sum in rank order.
        base = (tag & 1).to(tl.int64) * slot_bytes + inbox
        words = tl.multiple_of(base.to(tl.pointer_type(tl.int64)), 16) + row64 * (N // 2) + half
        low, failed = _wait_words(words, half * 2 < N, tag, abort_ptr, timeout)
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        h = h0
        mine = tl.load(own + row64 * stride_own + cols, mask=live, other=0.0).to(tl.float32)
        theirs = _unpack(low, own.dtype.element_ty, CHUNK)
        if RANK == 0:
            h += mine
            h += theirs
        else:
            h += theirs
            h += mine
    else:
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        h = h0
        failed = 0
        if ll:
            for w in tl.static_range(WORLD):
                if w == RANK:
                    mine = tl.load(own + row64 * stride_own + cols, mask=live, other=0.0)
                    h += mine.to(tl.float32)
                else:
                    peer = w - (w > RANK)
                    base = ((tag & 1) * (WORLD - 1) + peer).to(tl.int64) * slot_bytes + inbox
                    words = (
                        tl.multiple_of(base.to(tl.pointer_type(tl.int64)), 16)
                        + row64 * (N // 2)
                        + half
                    )
                    low, stop = _wait_words(words, half * 2 < N, tag, abort_ptr, timeout)
                    failed = failed | stop
                    h += _unpack(low, own.dtype.element_ty, CHUNK)
        else:
            n_tiles = tl.cdiv(N, tile_n)
            first = (row // tile_m).to(tl.int64) * n_tiles
            t = tl.arange(0, BLOCK_T)
            for w in tl.static_range(WORLD):
                if w != RANK:
                    flags = flag_base + (w - (w > RANK)) * flag_cap + first + t
                    failed = failed | _wait_tiles(flags, t < n_tiles, tag, abort_ptr, timeout)
            _acquire_fence()
            tl.debug_barrier()
            dense = row64 * N + cols
            for w in tl.static_range(WORLD):
                if w == RANK:
                    part = tl.load(own + row64 * stride_own + cols, mask=live, other=0.0)
                else:
                    peer = w - (w > RANK)
                    slot = ((tag & 1) * (WORLD - 1) + peer).to(tl.int64) * slot_bytes + inbox
                    part = tl.load(
                        tl.multiple_of(slot.to(tl.pointer_type(own.dtype.element_ty)), 16) + dense,
                        mask=live,
                        other=0.0,
                        cache_modifier=".cv",
                    )
                h += part.to(tl.float32)

    if NORM:
        # Every part marks its sum, failed or not, so its siblings never wait on it.
        ss = tl.sum(h * h, axis=0)
        if PARTS > 1:
            ss, stop = _row_total(sums, marks, row, piece, ss, tag, abort_ptr, timeout, PARTS)
            failed = failed | stop
    if failed == 0:
        if NORM:
            rstd = 1.0 / tl.sqrt_rn(ss / N + eps)
            tl.store(
                out + row64 * stride_out + cols,
                (h * rstd * gain).to(out.dtype.element_ty),
                mask=live,
            )
            tl.store(
                residual + row64 * stride_res + cols, h.to(residual.dtype.element_ty), mask=live
            )
        else:
            tl.store(out + row64 * stride_out + cols, h.to(out.dtype.element_ty), mask=live)
    else:
        error_ptr = tl.load(xch + DESC_ERROR).to(tl.pointer_type(tl.int32))
        tl.store(error_ptr, 1)

    # Every program read the tag before arriving; the last one advances it.
    if tl.atomic_add(done_ptr, 1, sem="acq_rel", scope="gpu") == M * PARTS - 1:
        tl.store(done_ptr, 0)
        tl.store(tag_ptr, tag + 1)


@triton.jit
def _skip_kernel(xch):
    """Move to the next call tag (a rejected call)."""
    tag_ptr = tl.load(xch + DESC_TAG).to(tl.pointer_type(tl.int32))
    tl.store(tag_ptr, tl.load(tag_ptr) + 1)


@triton.jit(do_not_specialize=["M", "N", "stride"])
def _publish_rows_kernel(
    xch,
    x,
    M,
    N,
    stride,
    BLOCK_N: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    """The row producer of ``all_reduce`` and ``add_rms_norm``: tile = one full row."""
    row = tl.program_id(0)
    stride = tl.multiple_of(stride, 8)
    cols = tl.arange(0, BLOCK_N)
    values = tl.load(x + row.to(tl.int64) * stride + cols, mask=cols < N, other=0.0)
    publish(xch, values[None, :], row, 0, M, N, row, RANK, WORLD)


def reduce_warps(block_n: int) -> int:
    return max(4, min(16, block_n // 256))


def launch_reduce(
    desc, own, out, residual, weight, eps, *, tile, norm, zero_centered, rank, world, pdl,
    parts, sums, marks,
) -> None:  # fmt: skip
    M, N = own.shape
    chunk = triton.next_power_of_2(N) // parts
    tile_m, tile_n = tile
    _reduce_kernel[(M, parts)](
        desc,
        own,
        out,
        residual if residual is not None else out,
        weight if weight is not None else out,
        sums,
        marks,
        M,
        N,
        own.stride(0),
        out.stride(0),
        residual.stride(0) if residual is not None else 0,
        tile_m,
        tile_n,
        float(eps),
        CHUNK=chunk,
        PARTS=parts,
        BLOCK_T=triton.next_power_of_2(triton.cdiv(N, tile_n)),
        HAS_RESIDUAL=residual is not None,
        NORM=norm,
        ZERO_CENTERED=zero_centered,
        RANK=rank,
        WORLD=world,
        USE_GDC=pdl,
        launch_pdl=pdl,
        num_warps=reduce_warps(chunk),
    )


def launch_publish_rows(desc, x, *, rank, world) -> None:
    M, N = x.shape
    block_n = triton.next_power_of_2(N)
    _publish_rows_kernel[(M,)](
        desc,
        x,
        M,
        N,
        x.stride(0),
        BLOCK_N=block_n,
        RANK=rank,
        WORLD=world,
        num_warps=reduce_warps(block_n),
    )


def launch_skip(desc) -> None:
    _skip_kernel[(1,)](desc)
