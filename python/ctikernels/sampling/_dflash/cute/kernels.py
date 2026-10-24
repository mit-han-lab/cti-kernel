# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe DSL 4.8.0 kernels, independently implemented from the K1 arithmetic ABI.

Two launches for vocabulary sampling: (row, chunk) partials, then row finalize.
Dense proposals add a position dimension to both grids. Fixed 4096-token chunks
contain four 1024-token ABI tiles; each 512-thread CTA handles two tiles at once.
Finalize scans chunk records sequentially and rereads only the selected chunk.
No vocabulary-sized intermediate writes. Tail and unaligned loads are scalar.
Public PTX ISA .rn instructions explicitly avoid FMA contraction and flush-to-zero in acceptance, residuals and CDF arithmetic.
Probability construction splits each (row, position) into the same 4096-token chunks.
Hardware qualification remains separate from offline compilation.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.utils import SmemAllocator

from ..reference.rng import PROPOSAL
from ._pdl import PDL


@cute.jit
def fp(a, b, op: cutlass.Constexpr):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"{op}.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def rcp_rn(b):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(b).ir_value()],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def fma_rn(a, b, c):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value(), Float32(c).ir_value()],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def row_divisor(b):
    """Per-row state for div_rn_by: (b, rcp.rn(b), whether b is in the fast-path range)."""
    return b, rcp_rn(b), (b >= Float32(2.0**-30)) & (b <= Float32(2.0**30))


@cute.jit
def div_rn_by(a, divisor):
    """Bitwise div.rn.f32(a, b) for a row-constant positive b.

    y = rcp.rn(b), then two Markstein corrections q += fma(-q, b, a) * y. With b in
    [2^-30, 2^30] and |a| in [2^-90, 2^90] there is no overflow or underflow, and the
    result is the correctly rounded quotient. Zero keeps its sign (a*y, y finite and
    nonzero for an in-range b); every other input (subnormal, huge, inf, NaN, or any a
    with an out-of-range b, e.g. a subnormal T whose rcp.rn is inf) uses div.rn.f32.
    """
    b, y, row_ok = divisor
    exponent = (float_bits(a) >> Uint32(23)) & Uint32(0xFF)
    result = Float32(0)
    if row_ok & (exponent >= Uint32(37)) & (exponent <= Uint32(216)):
        q = fp(a, y, "mul")
        q = fma_rn(fma_rn(-q, b, a), y, q)
        result = fma_rn(fma_rn(-q, b, a), y, q)
    elif row_ok & (a == Float32(0)):
        result = fp(a, y, "mul")
    else:
        result = fp(a, b, "div")
    return result


@cute.jit
def stable_exponent(x, maximum, scaled_max, tdiv):
    """Softmax exponent for T > 0: (x - max)/T, or x/T - max/T only where x - max overflows.

    x - max is exact for nearby logits, so the subtract-first form is used whenever it is
    finite. Only a spread above FLT_MAX (-inf from a finite x) falls back to scale-first
    with scaled_max = div.rn(max, T), which stays finite for large T; a masked x = -inf
    keeps -inf/T = -inf (probability 0). This is div_rn_by with
    the fallback inside its slow branch (-inf is never in the fast window), so the
    hot path is the same instruction sequence as div_rn_by(x - max).
    """
    d = fp(x, maximum, "sub")
    b, y, row_ok = tdiv
    exponent = (float_bits(d) >> Uint32(23)) & Uint32(0xFF)
    result = Float32(0)
    if row_ok & (exponent >= Uint32(37)) & (exponent <= Uint32(216)):
        q = fp(d, y, "mul")
        q = fma_rn(fma_rn(-q, b, d), y, q)
        result = fma_rn(fma_rn(-q, b, d), y, q)
    elif row_ok & (d == Float32(0)):
        result = fp(d, y, "mul")
    elif (d == Float32(float("-inf"))) & (x != Float32(float("-inf"))):
        result = fp(fp(x, b, "div"), scaled_max, "sub")
    else:
        result = fp(d, b, "div")
    return result


# If l - max overflows, the exact exponent is below -2^128/T; for T < 2^120 that is
# below -256 and exp gives 0 under either form (the rounded l/T - max/T stays below
# -255.9), so only rows with T >= HUGE_T can need the scale-first fallback.
HUGE_T = 2.0**120


@cute.jit
def row_exponent(x, maximum, scaled_max, tdiv, huge: cutlass.Constexpr):
    """Hot-loop exponent; callers unswitch once per row on T >= HUGE_T."""
    z = Float32(0)
    if cutlass.const_expr(huge):
        z = stable_exponent(x, maximum, scaled_max, tdiv)
    else:
        z = div_rn_by(fp(x, maximum, "sub"), tdiv)
    return z


@cute.jit
def philox(a, b, c, d, lo, hi):
    for _ in cutlass.range_constexpr(10):
        ac = Uint64(a) * Uint64(0xD2511F53)
        cc = Uint64(c) * Uint64(0xCD9E8D57)
        a, b, c, d = Uint32(cc >> 32) ^ b ^ lo, Uint32(cc), Uint32(ac >> 32) ^ d ^ hi, Uint32(ac)
        lo, hi = lo + Uint32(0x9E3779B9), hi + Uint32(0xBB67AE85)
    return a, b, c, d


@cute.jit
def uniform(key0, key1, cursor, domain, i):
    a, _, _, _ = philox(
        Uint32(cursor),
        Uint32(cursor >> 32),
        Uint32(domain),
        Uint32(i),
        Uint32(key0),
        Uint32(key1),
    )
    return fp(Float32(a >> 8), Float32(2.0**-24), "mul")


# Execution geometry; the ABI is the logical tile, not the CTA size. A CTA
# processes two adjacent tiles concurrently, then carries their totals in order.
THREADS = 512
VEC = 4
TILE_THREADS = 256
TILE = TILE_THREADS * VEC
WARPS = THREADS // 32
CHUNK = 4096
COMPACT_CAP = 1024
COMPACT_LOG = 10  # log2(COMPACT_CAP): the compact top-p bitonic sort
# Six levels, each [arrival counter, key, remaining, above, equal, full], uint64.
ROW_STATE_WORDS = 6 * 6
LEVEL_STATE_WORDS = 6
# Identity-filter probabilities: each CTA owns up to IDENTITY_GROUP chunks.
IDENTITY_GROUP = 8


def identity_group(rows, vocab, sm_count):
    """Host-static chunks per identity CTA.

    Measured on RTX 5090 and GB10 (B 1-4, Q 4-16, V 151936/248320): two chunks
    per CTA once there are at least four chunks per SM, else one; larger groups
    were slower everywhere.
    """
    chunks = rows * ((vocab + CHUNK - 1) // CHUNK)
    return 2 if chunks >= 4 * sm_count else 1


def greedy_group(rows, vocab, sm_count):
    """Host-static chunks per greedy_records CTA (all-greedy hint)."""
    return identity_group(rows, vocab, sm_count)


@cute.jit
def stage_records(source, staged, count: cutlass.Constexpr, tid):
    """Copy 32-bit record words cooperatively, preserving their exact bits."""
    for start in cutlass.range(0, count, THREADS, unroll_full=True):
        index = start + tid
        if index < count:
            staged[index] = source[index]
    cute.arch.sync_threads()


@cute.jit
def load4(w, base, v):
    values = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    values.fill(Float32(0))
    # Check the actual row address on device (including storage offsets). Never
    # promise 16-byte alignment to the provider's shape-only compilation cache.
    if ((w.toint() & 15) == 0) & (base + VEC <= v):
        aligned = cute.make_ptr(
            Float32, (w + base).toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        source = cute.make_tensor(aligned, cute.make_layout(VEC))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        cute.copy(atom, source, values)
    else:
        for j in cutlass.range_constexpr(VEC):
            if base + j < v:
                values[j] = w[base + j]
    return values


@cute.jit
def store4(w, base, v, values):
    """One 128-bit store when the device row address is aligned, else scalar stores."""
    if ((w.toint() & 15) == 0) & (base + VEC <= v):
        aligned = cute.make_ptr(
            Float32, (w + base).toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        target = cute.make_tensor(aligned, cute.make_layout(VEC))
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        cute.copy(atom, values, target)
    else:
        for j in cutlass.range_constexpr(VEC):
            if base + j < v:
                w[base + j] = values[j]


@cute.jit
def inclusive4(values):
    local = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    running = Float32(0)
    for j in cutlass.range_constexpr(VEC):
        running = fp(running, values[j], "add")
        local[j] = running
    return local


@cute.jit
def warp_scan(value, lane, steps: cutlass.Constexpr):
    # Hillis-Steele inclusive scan. Explicit lower clamp is required for shfl.up.
    for k in cutlass.range_constexpr(steps):
        other = cute.arch.shuffle_sync_up(value, offset=1 << k, mask_and_clamp=0)
        if lane >= (1 << k):
            value = fp(other, value, "add")
    return value


@cute.jit
def tile_scan(value, carry, tid, sums, prefixes):
    """Return (warp+lane exclusive prefix, preceding tiles, new row carry)."""
    lane, warp = tid % 32, tid // 32
    group = tid // TILE_THREADS
    scanned = warp_scan(value, lane, 5)
    before_lane = cute.arch.shuffle_sync_up(scanned, offset=1, mask_and_clamp=0)
    if lane == 0:
        before_lane = Float32(0)
    if lane == 31:
        sums[warp] = scanned
    cute.arch.sync_threads()
    if tid % TILE_THREADS < 32:
        warp_total = Float32(0)
        if lane < 8:
            warp_total = sums[group * 8 + lane]
        scanned_warps = warp_scan(warp_total, lane, 3)
        before_warp = cute.arch.shuffle_sync_up(scanned_warps, offset=1, mask_and_clamp=0)
        if lane == 0:
            before_warp = Float32(0)
        if lane < 8:
            prefixes[group * 8 + lane] = before_warp
        if lane == 7:
            sums[WARPS + group] = scanned_warps
    cute.arch.sync_threads()
    preceding = fp(prefixes[warp], before_lane, "add")
    tile_carry = carry
    next_carry = carry
    for k in cutlass.range_constexpr(THREADS // TILE_THREADS):
        if group == k:
            tile_carry = next_carry
        next_carry = fp(next_carry, sums[WARPS + k], "add")
    # All readers finish before the next tile (or the fallback p scan) reuses smem.
    cute.arch.sync_threads()
    return preceding, tile_carry, next_carry


@cute.jit
def reduce_choices(choice, last, tid, choices, lasts):
    lane, warp = tid % 32, tid // 32
    for k in cutlass.range_constexpr(5):
        choice = min(choice, cute.arch.shuffle_sync_bfly(choice, offset=1 << k))
        last = max(last, cute.arch.shuffle_sync_bfly(last, offset=1 << k))
    if lane == 0:
        choices[warp], lasts[warp] = choice, last
    cute.arch.sync_threads()
    if warp == 0:
        best, final = Int32(0x7FFFFFFF), Int32(-1)
        if lane < WARPS:
            best, final = choices[lane], lasts[lane]
        for k in cutlass.range_constexpr(5):
            best = min(best, cute.arch.shuffle_sync_bfly(best, offset=1 << k))
            final = max(final, cute.arch.shuffle_sync_bfly(final, offset=1 << k))
        if lane == 0:
            choices[0], lasts[0] = best, final
    cute.arch.sync_threads()
    result, final = choices[0], lasts[0]
    cute.arch.sync_threads()
    return result, final


@cute.jit
def masses(
    pvalues,
    q,
    sparse_q,
    base,
    v,
    chunk_start,
    subtract,
    sparse: cutlass.Constexpr,
    point: cutlass.Constexpr,
    draft,
):
    values = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    for j in cutlass.range_constexpr(VEC):
        values[j] = pvalues[j]
    if subtract:
        if cutlass.const_expr(point):
            # For t!=draft, p-0=p; retain the dense subtraction/clamp at draft.
            for j in cutlass.range_constexpr(VEC):
                if base + j == draft:
                    values[j] = max(fp(pvalues[j], Float32(1), "sub"), Float32(0))
        elif cutlass.const_expr(sparse):
            for j in cutlass.range_constexpr(VEC):
                values[j] = max(fp(pvalues[j], sparse_q[base + j - chunk_start], "sub"), Float32(0))
        else:
            qvalues = load4(q, base, v)
            for j in cutlass.range_constexpr(VEC):
                values[j] = max(fp(pvalues[j], qvalues[j], "sub"), Float32(0))
    return values


@cute.jit
def scatter_chunk(q, ids, sparse_q, row_position, chunk_start, tid):
    # Each candidate is added in original column order, before a single clamp.
    # Only IDs in this CTA's chunk touch shared memory. No atomics/reassociation.
    for start in range(0, CHUNK, THREADS):
        sparse_q[start + tid] = Float32(0)
    cute.arch.sync_threads()
    if tid == 0:
        for k in range(16):
            offset = row_position * 16 + k
            token = ids[offset] - chunk_start
            if (token >= 0) & (token < CHUNK):
                sparse_q[token] = fp(sparse_q[token], q[offset], "add")
    cute.arch.sync_threads()


@cute.jit
def acceptance_position(
    p,
    x,
    q,
    ids,
    seeds,
    cursors,
    b,
    slot,
    g: cutlass.Constexpr,
    v: cutlass.Constexpr,
    sparse: cutlass.Constexpr,
    point: cutlass.Constexpr,
    position,
):
    # Return g for an accepted/inactive position; RNG uses the full position.
    rejected = Int32(g)
    if position < g:
        token = x[Int64(b) * g + position]
        # Speculative work after an earlier rejection must not read invalid IDs.
        if (token >= 0) & (token < v):
            mass = Float32(0)
            if cutlass.const_expr(point):
                mass = Float32(1)
            elif cutlass.const_expr(sparse):
                for k in cutlass.range(16, unroll_full=True):
                    offset = (Int64(b) * g + position) * 16 + k
                    if ids[offset] == token:
                        mass = fp(mass, q[offset], "add")
            else:
                mass = q[(Int64(b) * g + position) * v + token]
            u = uniform(seeds[Int64(slot) * 2], seeds[Int64(slot) * 2 + 1], cursors[b], 2, position)
            if not (fp(u, mass, "mul") < p[(Int64(b) * (g + 1) + position) * v + token]):
                rejected = position
        else:
            rejected = position
    return rejected


@cute.jit
def acceptance(
    p,
    x,
    q,
    ids,
    seeds,
    cursors,
    b,
    slot,
    g: cutlass.Constexpr,
    v: cutlass.Constexpr,
    sparse: cutlass.Constexpr,
    point: cutlass.Constexpr,
):
    lane, _, _ = cute.arch.thread_idx()
    rejected = Int32(g)
    if cutlass.const_expr(g < 32):
        rejected = acceptance_position(
            p, x, q, ids, seeds, cursors, b, slot, g, v, sparse, point, lane
        )
    else:
        for position in cutlass.range(Int64(lane), g, 32):
            rejected = min(
                rejected,
                acceptance_position(
                    p, x, q, ids, seeds, cursors, b, slot, g, v, sparse, point, Int32(position)
                ),
            )
    for k in cutlass.range_constexpr(5):
        rejected = min(rejected, cute.arch.shuffle_sync_bfly(rejected, offset=1 << k))
    return rejected


@cute.jit
def warp_argmax(best, token_id):
    for k in cutlass.range_constexpr(5):
        value = cute.arch.shuffle_sync_bfly(best, offset=1 << k)
        token = cute.arch.shuffle_sync_bfly(token_id, offset=1 << k)
        if (value > best) | ((value == best) & (token < token_id)):
            best, token_id = value, token
    return best, token_id


@cute.jit
def argmax(w, v, chunk_start, tid, maxima, choices):
    best, token_id = Float32(float("-inf")), Int32(v)
    for start in range(0, CHUNK, THREADS * VEC):
        base = chunk_start + start + tid * VEC
        values = load4(w, base, v)
        for j in cutlass.range_constexpr(VEC):
            token = base + j
            if token < v:
                value = values[j]
                if (value > best) | ((value == best) & (token < token_id)):
                    best, token_id = value, token
    best, token_id = warp_argmax(best, token_id)
    lane, warp = tid % 32, tid // 32
    if lane == 0:
        maxima[warp], choices[warp] = best, token_id
    cute.arch.sync_threads()
    if warp == 0:
        score, winner = Float32(float("-inf")), Int32(v)
        if lane < WARPS:
            score, winner = maxima[lane], choices[lane]
        score, winner = warp_argmax(score, winner)
        if lane == 0:
            maxima[0], choices[0] = score, winner
    cute.arch.sync_threads()
    value, result = maxima[0], choices[0]
    cute.arch.sync_threads()
    return value, result


GREEDY_BATCH = 4


@cute.jit
def batch_argmax(
    rows, v, count, start, chunk_start, tid, maxima, choices, record_f, record_i, stride
):
    """argmax() for rows start..start+GREEDY_BATCH-1 (those < count) of one chunk.

    Same per-thread order and smallest-ID tie rule as argmax(); each row's CTA
    reduction is independent, so the rows share one barrier per stage. Lane 0 of
    warp k writes row start+k's (max, id) record at record + (start+k)*stride.
    """
    lane, warp = tid % 32, tid // 32
    loaded = cute.make_rmem_tensor(cute.make_layout(GREEDY_BATCH * 2 * VEC), Float32)
    loaded.fill(Float32(0))
    for k in cutlass.range_constexpr(GREEDY_BATCH):
        if start + k < count:
            for half in cutlass.range_constexpr(2):
                base = chunk_start + half * (THREADS * VEC) + tid * VEC
                values = load4(rows + Int64(start + k) * v, base, v)
                for j in cutlass.range_constexpr(VEC):
                    loaded[(k * 2 + half) * VEC + j] = values[j]
    for k in cutlass.range_constexpr(GREEDY_BATCH):
        best, token_id = Float32(float("-inf")), Int32(v)
        for half in cutlass.range_constexpr(2):
            base = chunk_start + half * (THREADS * VEC) + tid * VEC
            for j in cutlass.range_constexpr(VEC):
                token = base + j
                if token < v:
                    value = loaded[(k * 2 + half) * VEC + j]
                    if (value > best) | ((value == best) & (token < token_id)):
                        best, token_id = value, token
        best, token_id = warp_argmax(best, token_id)
        if lane == 0:
            maxima[k * WARPS + warp], choices[k * WARPS + warp] = best, token_id
    cute.arch.sync_threads()
    if warp < GREEDY_BATCH:
        if start + warp < count:
            score, winner = Float32(float("-inf")), Int32(v)
            if lane < WARPS:
                score, winner = maxima[warp * WARPS + lane], choices[warp * WARPS + lane]
            score, winner = warp_argmax(score, winner)
            if lane == 0:
                offset = Int64(start + warp) * stride
                record_f[offset], record_i[offset] = score, winner
    cute.arch.sync_threads()


class Chain:
    def __init__(self, gamma, vocab, mode, domain=PROPOSAL, scratch_positions=None, min_p=False):
        self.g, self.v, self.mode = gamma, vocab, mode
        self.domain = domain
        self.min_p = min_p
        assert not min_p or mode == "sample"
        self.scratch_positions = (
            max(2, gamma + 1) if scratch_positions is None else scratch_positions
        )
        self.n = (vocab + CHUNK - 1) // CHUNK
        # Greedy partials spread GREEDY_BATCH-position groups over grid y;
        # stochastic rows use only y = 0.
        self.batches = (gamma + GREEDY_BATCH) // GREEDY_BATCH

    def config_key(self):
        return (self.g, self.v, self.mode, self.domain, self.scratch_positions, self.min_p)

    @cute.jit
    def __call__(
        self,
        p: cute.Tensor,
        x: cute.Tensor,
        q: cute.Tensor,
        ids: cute.Tensor,
        temperatures: cute.Tensor,
        slots: cute.Tensor,
        seeds: cute.Tensor,
        cursors: cute.Tensor,
        work: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            p.iterator,
            x.iterator,
            q.iterator,
            ids.iterator,
            temperatures.iterator,
            slots.iterator,
            seeds.iterator,
            cursors.iterator,
            work.iterator,
            out.iterator,
            counts.iterator,
            False,
        ).launch(
            grid=(slots.shape[0], self.g if self.mode == "sample" else self.batches, self.n)
            if (self.g if self.mode == "sample" else self.batches) <= 65535
            else (slots.shape[0] * (self.g if self.mode == "sample" else self.batches), 1, self.n),
            block=(THREADS, 1, 1),
            use_pdl=PDL,
            stream=stream,
        )
        self.kernel(
            p.iterator,
            x.iterator,
            q.iterator,
            ids.iterator,
            temperatures.iterator,
            slots.iterator,
            seeds.iterator,
            cursors.iterator,
            work.iterator,
            out.iterator,
            counts.iterator,
            True,
        ).launch(
            grid=(slots.shape[0], self.g if self.mode == "sample" else 1, 1)
            if self.mode != "sample" or self.g <= 65535
            else (slots.shape[0] * self.g, 1, 1),
            block=(THREADS, 1, 1),
            use_pdl=PDL,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        p: cute.Pointer,
        x: cute.Pointer,
        q: cute.Pointer,
        ids: cute.Pointer,
        temperatures: cute.Pointer,
        slots: cute.Pointer,
        seeds: cute.Pointer,
        cursors: cute.Pointer,
        work: cute.Pointer,
        out: cute.Pointer,
        counts: cute.Pointer,
        finalize: cutlass.Constexpr,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, position, chunk = cute.arch.block_idx()
        positions: cutlass.Constexpr = (
            self.g if self.mode == "sample" else (1 if finalize else self.batches)
        )
        if cutlass.const_expr(positions > 65535):
            position = b % positions
            b = b // positions
        tid, _, _ = cute.arch.thread_idx()
        smem = SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout(WARPS + THREADS // TILE_THREADS))
        prefixes = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        choices = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        lasts = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        sparse_q = smem.allocate_tensor(
            Float32, cute.make_layout(CHUNK if self.mode == "sparse" else 1)
        )
        state = smem.allocate_tensor(Int32, cute.make_layout(4))
        draw_state = smem.allocate_tensor(Float32, cute.make_layout(2))
        if cutlass.const_expr(self.min_p):
            min_p_pivot = smem.allocate_tensor(Float32, cute.make_layout(1))
        batch_maxima = smem.allocate_tensor(Float32, cute.make_layout(GREEDY_BATCH * WARPS))
        batch_choices = smem.allocate_tensor(Int32, cute.make_layout(GREEDY_BATCH * WARPS))
        if cutlass.const_expr(finalize):
            # Greedy winners and stochastic records have disjoint lifetimes.
            # Q>32 stages winners in out, keeping shared storage independent of Q.
            staged = smem.allocate_tensor(
                Int32,
                cute.make_layout(
                    max(2 * self.n, 1)
                    if self.mode == "sample"
                    else max(4 * self.n, self.g + 1 if self.g < 32 else 1)
                ),
            )
            staged_f = cute.recast_ptr(staged.iterator, dtype=Float32)
        # Union of [B,Q,nchunks,2] words. Stochastic verify uses positions 0/1
        # for residual/p records; greedy uses all Q for max/id, sample uses gamma.
        wf = cute.make_ptr(Float32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        wi = cute.make_ptr(Int32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        row = Int64(b) * self.scratch_positions * self.n * 2
        slot = slots[b]
        if cutlass.const_expr(finalize):
            if cutlass.const_expr(self.mode == "sample"):
                if tid == 0:
                    out[Int64(b) * self.g + position] = Int32(-1)
            else:
                if cutlass.const_expr(self.g < 32):
                    if tid < self.g + 1:
                        out[Int64(b) * (self.g + 1) + tid] = Int32(-1)
                else:
                    for i in cutlass.range(Int64(tid), self.g + 1, THREADS):
                        out[Int64(b) * (self.g + 1) + i] = Int32(-1)
                if tid == 0:
                    counts[b] = Int32(0)
            # Complete output initialization before thread zero emits the prefix.
            cute.arch.sync_threads()
        # No input, workspace or RNG access for padding beyond the slot test.
        proceed = Int32(0)
        if slot >= 0:
            proceed = Int32(1)
            if cutlass.const_expr(self.mode != "sample" and not finalize):
                # Partial CTAs with y > 0 only serve greedy position groups.
                if position > 0:
                    if temperatures[b] != Float32(0):
                        proceed = Int32(0)
        if proceed != 0:
            greedy = temperatures[b] == Float32(0)
            if greedy:
                if cutlass.const_expr(finalize):
                    lane, warp = tid % 32, tid // 32
                    for start in cutlass.range(
                        0 if self.g < 32 else Int64(0),
                        1 if self.mode == "sample" else self.g + 1,
                        WARPS,
                        unroll_full=self.g < 32,
                    ):
                        i = start + warp
                        if i < (1 if self.mode == "sample" else self.g + 1):
                            pos = i
                            if cutlass.const_expr(self.mode == "sample"):
                                pos = position
                            best, winner = Float32(float("-inf")), Int32(self.v)
                            for base in cutlass.range(0, self.n, 32, unroll_full=True):
                                c = base + lane
                                if c < self.n:
                                    offset = row + (Int64(pos) * self.n + c) * 2
                                    value, token = wf[offset], wi[offset + 1]
                                    if (value > best) | ((value == best) & (token < winner)):
                                        best, winner = value, token
                            best, winner = warp_argmax(best, winner)
                            if lane == 0:
                                if cutlass.const_expr(self.mode != "sample" and self.g >= 32):
                                    out[Int64(b) * (self.g + 1) + i] = winner
                                else:
                                    staged[i] = winner
                    cute.arch.sync_threads()
                    if tid == 0:
                        a, continuing = Int32(0), Int32(1)
                        for i in cutlass.range(
                            1 if self.mode == "sample" else self.g + 1, unroll_full=self.g < 32
                        ):
                            if continuing == 1:
                                if cutlass.const_expr(self.mode != "sample" and self.g >= 32):
                                    winner = out[Int64(b) * (self.g + 1) + i]
                                else:
                                    winner = staged[i]
                                if cutlass.const_expr(self.mode == "sample"):
                                    out[Int64(b) * self.g + position] = winner
                                    continuing = Int32(0)
                                else:
                                    # Emit the mismatch winner too; later positions stay -1.
                                    out[Int64(b) * (self.g + 1) + i] = winner
                                    if i < self.g:
                                        if x[Int64(b) * self.g + i] == winner:
                                            a += 1
                                        else:
                                            continuing = Int32(0)
                            elif cutlass.const_expr(self.mode != "sample" and self.g >= 32):
                                out[Int64(b) * (self.g + 1) + i] = Int32(-1)
                        if cutlass.const_expr(self.mode != "sample"):
                            counts[b] = a
                elif cutlass.const_expr(self.mode == "sample"):
                    target = q + (Int64(b) * self.g + position) * self.v
                    best, winner = argmax(target, self.v, chunk * CHUNK, tid, sums, choices)
                    if tid == 0:
                        offset = row + (Int64(position) * self.n + chunk) * 2
                        wf[offset], wi[offset + 1] = best, winner
                else:
                    # Grid y selects GREEDY_BATCH positions: their loads are in
                    # flight together and their chunk reductions share one barrier.
                    batch_argmax(
                        p + Int64(b) * (self.g + 1) * self.v,
                        self.v,
                        self.g + 1,
                        position * GREEDY_BATCH,
                        chunk * CHUNK,
                        tid,
                        batch_maxima,
                        batch_choices,
                        wf + row + Int64(chunk) * 2,
                        wi + row + Int64(chunk) * 2 + 1,
                        self.n * 2,
                    )
            else:
                if cutlass.const_expr(self.mode == "sample" or self.g == 0):
                    if tid == 0:
                        state[0] = Int32(0)
                else:
                    if tid < 32:
                        accepted = acceptance(
                            p,
                            x,
                            q,
                            ids,
                            seeds,
                            cursors,
                            b,
                            slot,
                            self.g,
                            self.v,
                            self.mode == "sparse",
                            self.mode == "point",
                        )
                        if tid == 0:
                            state[0] = accepted
                cute.arch.sync_threads()
                a = state[0]
                target = p + (Int64(b) * (self.g + 1) + a) * self.v
                proposal = q
                subtract = a < self.g
                draft = Int32(-1)
                if cutlass.const_expr(self.mode == "point"):
                    if subtract:
                        draft = x[Int64(b) * self.g + a]
                record = row
                if cutlass.const_expr(self.mode == "sample"):
                    target = q + (Int64(b) * self.g + position) * self.v
                    subtract = False
                    record = row + Int64(position) * self.n * 2
                elif cutlass.const_expr(self.mode == "dense"):
                    proposal = q + (Int64(b) * self.g + a) * self.v
                pivot = Float32(0)
                if cutlass.const_expr(self.min_p):
                    # x/ids are borrowed FP32 min_p and final probability maxima.
                    # Each CTA reads completed records; partials never overwrite them.
                    if x[b] > Float32(0):
                        if tid < 32:
                            pmax = Float32(0)
                            for c in cutlass.range(tid, self.n, 32):
                                pmax = max(pmax, ids[(Int64(b) * self.g + position) * self.n + c])
                            for k in cutlass.range_constexpr(5):
                                pmax = max(pmax, cute.arch.shuffle_sync_bfly(pmax, offset=1 << k))
                            if tid == 0:
                                min_p_pivot[0] = fp(x[b], pmax, "mul")
                        cute.arch.sync_threads()
                        pivot = min_p_pivot[0]
                chunk_start = chunk * CHUNK
                if cutlass.const_expr(finalize):
                    stage_records(
                        wi + record, staged, self.n * (2 if self.mode == "sample" else 4), tid
                    )
                    if tid < 32:
                        last, plast = Int32(-1), Int32(-1)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            c = base + tid
                            if c < self.n:
                                last = max(last, staged[c * 2 + 1])
                                if cutlass.const_expr(self.mode != "sample"):
                                    plast = max(plast, staged[(self.n + c) * 2 + 1])
                        for k in cutlass.range_constexpr(5):
                            last = max(last, cute.arch.shuffle_sync_bfly(last, offset=1 << k))
                            plast = max(plast, cute.arch.shuffle_sync_bfly(plast, offset=1 << k))
                        if tid == 0:
                            state[2], state[3] = last, plast
                    cute.arch.sync_threads()
                    if tid == 0:
                        total, ptotal = Float32(0), Float32(0)
                        last, plast = state[2], state[3]
                        # Retain every add, including zeros, in ascending chunk order.
                        for c in cutlass.range(self.n, unroll_full=True):
                            total = fp(total, staged_f[c * 2], "add")
                            if cutlass.const_expr(self.mode != "sample"):
                                ptotal = fp(ptotal, staged_f[(self.n + c) * 2], "add")
                        fallback = Int32(0)
                        if subtract & (total == Float32(0)):
                            fallback, total, last = Int32(1), ptotal, plast
                        domain, draw_position = Int32(3), Int32(0)
                        if cutlass.const_expr(self.mode == "sample"):
                            domain, draw_position = Int32(self.domain), position
                        else:
                            if a == self.g:
                                domain = Int32(4)
                            counts[b] = a
                            for i in range(self.g):
                                if i < a:
                                    out[Int64(b) * (self.g + 1) + i] = x[Int64(b) * self.g + i]
                        u = uniform(
                            seeds[Int64(slot) * 2],
                            seeds[Int64(slot) * 2 + 1],
                            cursors[b],
                            domain,
                            draw_position,
                        )
                        threshold = fp(u, total, "mul")
                        carry, exclusive = Float32(0), Float32(0)
                        selected = Int32(-1)
                        for c in cutlass.range(self.n, unroll_full=True):
                            before = carry
                            carry = fp(carry, staged_f[(fallback * self.n + c) * 2], "add")
                            if (selected == -1) & (carry > threshold):
                                selected, exclusive = c, before
                        state[1], state[2], state[3] = selected, fallback, last
                        draw_state[0], draw_state[1] = threshold, exclusive
                    cute.arch.sync_threads()
                    chunk_start = state[1] * CHUNK
                    subtract = subtract & (state[2] == 0)
                # For a rounding spill no chunk is reread; use the whole-row last.
                selected = Int32(1)
                if cutlass.const_expr(finalize):
                    selected = Int32(state[1] >= 0)
                result = Int32(self.v)
                if selected == 1:
                    if cutlass.const_expr(self.mode == "sparse"):
                        if subtract:
                            scatter_chunk(q, ids, sparse_q, Int64(b) * self.g + a, chunk_start, tid)
                    total, ptotal = Float32(0), Float32(0)
                    carry, threshold = Float32(0), Float32(0)
                    if cutlass.const_expr(finalize):
                        threshold, carry = draw_state[0], draw_state[1]
                    choice, last, plast = Int32(self.v), Int32(-1), Int32(-1)
                    for start in range(0, CHUNK, THREADS * VEC):
                        base = chunk_start + start + tid * VEC
                        pv = load4(target, base, self.v)
                        values = masses(
                            pv,
                            proposal,
                            sparse_q,
                            base,
                            self.v,
                            chunk_start,
                            subtract,
                            self.mode == "sparse",
                            self.mode == "point",
                            draft,
                        )
                        if cutlass.const_expr(self.min_p):
                            for j in cutlass.range_constexpr(VEC):
                                if values[j] < pivot:
                                    values[j] = Float32(0)
                        local = inclusive4(values)
                        if cutlass.const_expr(finalize):
                            preceding, tile_carry, carry = tile_scan(
                                local[VEC - 1], carry, tid, sums, prefixes
                            )
                            for j in cutlass.range_constexpr(VEC):
                                if base + j < self.v:
                                    prefix = fp(tile_carry, fp(preceding, local[j], "add"), "add")
                                    if prefix > threshold:
                                        choice = min(choice, base + j)
                        else:
                            _, _, total = tile_scan(local[VEC - 1], total, tid, sums, prefixes)
                            if cutlass.const_expr(self.mode != "sample"):
                                if subtract:
                                    plocal = inclusive4(pv)
                                    _, _, ptotal = tile_scan(
                                        plocal[VEC - 1], ptotal, tid, sums, prefixes
                                    )
                                else:
                                    ptotal = total
                            for j in cutlass.range_constexpr(VEC):
                                if base + j < self.v:
                                    if values[j] > Float32(0):
                                        last = max(last, base + j)
                                    if pv[j] > Float32(0):
                                        plast = max(plast, base + j)
                    result, last = reduce_choices(choice, last, tid, choices, lasts)
                    if cutlass.const_expr(not finalize):
                        if cutlass.const_expr(self.mode != "sample"):
                            if subtract:
                                _, plast = reduce_choices(Int32(self.v), plast, tid, choices, lasts)
                            else:
                                plast = last
                        if tid == 0:
                            wf[record + chunk * 2], wi[record + chunk * 2 + 1] = total, last
                            if cutlass.const_expr(self.mode != "sample"):
                                wf[record + (self.n + chunk) * 2] = ptotal
                                wi[record + (self.n + chunk) * 2 + 1] = plast
                if cutlass.const_expr(finalize):
                    if tid == 0:
                        if result == self.v:
                            result = state[3]
                        if cutlass.const_expr(self.mode == "sample"):
                            out[Int64(b) * self.g + position] = result
                        else:
                            out[Int64(b) * (self.g + 1) + a] = result


class Candidates:
    def __init__(self, position):
        self.position = position

    def config_key(self):
        return (self.position,)

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        ids: cute.Tensor,
        temperatures: cute.Tensor,
        slots: cute.Tensor,
        seeds: cute.Tensor,
        cursors: cute.Tensor,
        out: cute.Tensor,
        oq: cute.Tensor,
        position_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            scores.iterator,
            ids.iterator,
            temperatures.iterator,
            slots.iterator,
            seeds.iterator,
            cursors.iterator,
            out.iterator,
            oq.iterator,
            position_tensor.iterator,
        ).launch(grid=(slots.shape[0], 1, 1), block=(32, 1, 1), use_pdl=PDL, stream=stream)

    @cute.kernel
    def kernel(
        self,
        scores: cute.Pointer,
        ids: cute.Pointer,
        temperatures: cute.Pointer,
        slots: cute.Pointer,
        seeds: cute.Pointer,
        cursors: cute.Pointer,
        out: cute.Pointer,
        oq: cute.Pointer,
        position_tensor: cute.Pointer,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            out[b] = Int32(-1)
            for k in range(16):
                oq[Int64(b) * 16 + k] = Float32(0)
            slot = slots[b]
            if slot >= 0:
                ss = cute.make_rmem_tensor(cute.make_layout(16), Float32)
                ii = cute.make_rmem_tensor(cute.make_layout(16), Int32)
                # Admission supplies unique ascending IDs with aligned scores.
                for k in cutlass.range_constexpr(16):
                    ii[k], ss[k] = ids[Int64(b) * 16 + k], scores[Int64(b) * 16 + k]
                best, winner = Float32(float("-inf")), Int32(0x7FFFFFFF)
                for k in cutlass.range_constexpr(16):
                    if ss[k] > best:
                        best, winner = ss[k], ii[k]
                temperature = temperatures[b]
                if temperature == Float32(0):
                    out[b] = winner
                else:
                    # Overflow-safe exponent, matching the reference FP32 order.
                    tdiv = row_divisor(temperature)
                    scaled_max = fp(best, temperature, "div")
                    total = Float32(0)
                    for k in cutlass.range_constexpr(16):
                        scaled = stable_exponent(ss[k], best, scaled_max, tdiv)
                        ss[k] = cute.exp(scaled, fastmath=False)
                        total = fp(total, ss[k], "add")
                    qtotal = Float32(0)
                    for k in cutlass.range_constexpr(16):
                        ss[k] = fp(ss[k], total, "div")
                        oq[Int64(b) * 16 + k] = ss[k]
                        qtotal = fp(qtotal, ss[k], "add")
                    if cutlass.const_expr(self.position is None):
                        position = position_tensor[0]
                    else:
                        position = Int32(self.position)
                    u = uniform(
                        seeds[Int64(slot) * 2], seeds[Int64(slot) * 2 + 1], cursors[b], 1, position
                    )
                    target = fp(u, qtotal, "mul")
                    prefix, chosen, last = Float32(0), Int32(-1), Int32(-1)
                    for k in cutlass.range_constexpr(16):
                        prefix = fp(prefix, ss[k], "add")
                        if (chosen == -1) & (prefix > target):
                            chosen = ii[k]
                        if ss[k] > Float32(0):
                            last = ii[k]
                    if chosen == -1:
                        chosen = last
                    out[b] = chosen


class Rng:
    """Private qualification entry point for raw Philox KATs and ABI vectors."""

    def config_key(self):
        return ()

    @cute.jit
    def __call__(
        self, counter: cute.Tensor, key: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(counter, key, out).launch(
            grid=(counter.shape[0], 1, 1), block=(32, 1, 1), use_pdl=PDL, stream=stream
        )

    @cute.kernel
    def kernel(self, counter: cute.Tensor, key: cute.Tensor, out: cute.Tensor):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            a, c, d, e = philox(
                Uint32(counter[b, 0]),
                Uint32(counter[b, 1]),
                Uint32(counter[b, 2]),
                Uint32(counter[b, 3]),
                Uint32(key[b, 0]),
                Uint32(key[b, 1]),
            )
            out[b, 0], out[b, 1], out[b, 2], out[b, 3] = Int64(a), Int64(c), Int64(d), Int64(e)


class Cursors:
    def config_key(self):
        return ()

    @cute.jit
    def __call__(
        self,
        anchors: cute.Tensor,
        prompts: cute.Tensor,
        slots: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(anchors, prompts, slots, out).launch(
            grid=((slots.shape[0] + 255) // 256, 1, 1),
            block=(256, 1, 1),
            use_pdl=PDL,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, anchors: cute.Tensor, prompts: cute.Tensor, slots: cute.Tensor, out: cute.Tensor
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = Int64(block) * 256 + tid
        if row < slots.shape[0]:
            slot = slots[row]
            cursor = Int64(0)
            if slot >= 0:
                cursor = anchors[row] - prompts[slot] + Int64(1)
            out[row] = cursor


@cute.jit
def probability_load4(w, base, v):
    """128-bit native loads, preserving the four-token arithmetic ownership.

    Adjacent lanes share eight BF16/FP16 elements via shuffles. Only the even
    lane loads; unaligned rows and the last incomplete pair use scalar loads.
    FP32 retains one 128-bit load per lane. Never change the Z reduction tree.
    """
    values = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    values.fill(Float32(0))
    if cutlass.const_expr(w.value_type.width == 16):
        pair_base = base - (base % 8)
        native = cute.make_rmem_tensor(cute.make_layout(8), w.value_type)
        native.fill(0)
        aligned_pair = (((w + pair_base).toint() & 15) == 0) & (pair_base + 8 <= v)
        if aligned_pair & (base % 8 == 0):
            aligned = cute.make_ptr(
                w.value_type,
                (w + pair_base).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            source = cute.make_tensor(aligned, cute.make_layout(8))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), w.value_type, num_bits_per_copy=128
            )
            cute.copy(atom, source, native)
        tid, _, _ = cute.arch.thread_idx()
        # All warp lanes execute the shuffles, including scalar-tail lanes.
        for j in cutlass.range_constexpr(VEC):
            upper = cute.arch.shuffle_sync(Float32(native[j + VEC]), (tid % 32) & ~1)
            if aligned_pair:
                values[j] = Float32(native[j])
                if base % 8 != 0:
                    values[j] = upper
            elif base + j < v:
                values[j] = Float32(w[base + j])
    else:
        values = load4(w, base, v)
    return values


@cute.jit
def float_bits(value):
    return Uint32(llvm.bitcast(T.i32(), Float32(value).ir_value()))


@cute.jit
def ordered_key(value):
    # IEEE +/- zero compare equal, so canonicalize before radix selection.
    if value == Float32(0):
        value = Float32(0)
    bits = float_bits(value)
    key = bits ^ Uint32(0x80000000)
    if (bits & Uint32(0x80000000)) != 0:
        key = ~bits
    return key


@cute.jit
def mass62(value, ceiling: cutlass.Constexpr = False):
    """Exact floor/ceil(value * 2^62); nonnegative finite FP32 <=1."""
    bits = float_bits(value)
    exponent = Int32((bits >> 23) & Uint32(255))
    mantissa = Uint64(bits & Uint32(0x7FFFFF))
    shift = exponent - 88
    if exponent != 0:
        mantissa = mantissa | Uint64(1 << 23)
    else:
        shift = Int32(-87)
    result = Uint64(0)
    if shift >= 0:
        result = mantissa << shift
    elif shift > -64:
        result = mantissa >> (-shift)
        if cutlass.const_expr(ceiling):
            if (mantissa & ((Uint64(1) << (-shift)) - Uint64(1))) != 0:
                result += Uint64(1)
    elif cutlass.const_expr(ceiling):
        if mantissa != 0:
            result = Uint64(1)
    return result


@cute.jit
def integer_scan(value, tid, scratch):
    """CTA exclusive prefix and total, exact uint64; all lanes participate."""
    lane, warp = tid % 32, tid // 32
    inclusive = Uint64(value)
    for k in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(inclusive, offset=1 << k, mask_and_clamp=0)
        if lane >= (1 << k):
            inclusive += other
    if lane == 31:
        scratch[warp] = inclusive
    cute.arch.sync_threads()
    if warp == 0:
        w = Uint64(0)
        if lane < WARPS:
            w = scratch[lane]
        for k in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(w, offset=1 << k, mask_and_clamp=0)
            if lane >= (1 << k):
                w += other
        if lane < WARPS:
            scratch[WARPS + lane] = w
    cute.arch.sync_threads()
    before = inclusive - Uint64(value)
    if warp > 0:
        before += scratch[WARPS + warp - 1]
    total = scratch[2 * WARPS - 1]
    cute.arch.sync_threads()
    return before, total


@cute.jit
def select_bin(hist, bins: cutlass.Constexpr, need, tid, scan, selected):
    """Descending-key histogram scan: selected bin, preceding mass/count, bucket, total."""
    local = Uint64(0)
    for j in cutlass.range_constexpr(bins // THREADS):
        local += Uint64(hist[tid * (bins // THREADS) + j])
    before, total = integer_scan(local, tid, scan)
    if tid == 0:
        selected[0], selected[1], selected[2] = Uint64(0), Uint64(0), Uint64(0)
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(bins // THREADS):
        index = tid * (bins // THREADS) + j
        count = Uint64(hist[index])
        if (before < need) & (before + count >= need):
            selected[0], selected[1], selected[2] = Uint64(index), before, count
        before += count
    cute.arch.sync_threads()
    index, preceding, count = Int32(selected[0]), selected[1], selected[2]
    cute.arch.sync_threads()
    return index, preceding, count, total


@cute.jit
def probability_reduce(value, tid, scratch, maximum: cutlass.Constexpr):
    lane, warp = tid % 32, tid // 32
    for k in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(value, offset=1 << k)
        if cutlass.const_expr(maximum):
            value = max(value, other)
        else:
            value = fp(value, other, "add")
    if lane == 0:
        scratch[warp] = value
    cute.arch.sync_threads()
    if warp == 0:
        value = Float32(0)
        if cutlass.const_expr(maximum):
            value = Float32(float("-inf"))
        if lane < WARPS:
            value = scratch[lane]
        for k in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(value, offset=1 << k)
            if cutlass.const_expr(maximum):
                value = max(value, other)
            else:
                value = fp(value, other, "add")
        if lane == 0:
            scratch[0] = value
    cute.arch.sync_threads()
    result = scratch[0]
    cute.arch.sync_threads()
    return result


@cute.jit
def load_histogram(ptr, dtype: cutlass.Constexpr):
    """L1-bypassing read of an integer bin completed by this grid's CTAs.

    Dispatch on the width: a recast shared view's element type need not be Uint64.
    """
    if cutlass.const_expr(dtype.width == 64):
        value = Uint64(
            llvm.inline_asm(
                T.i64(),
                [ptr.toint().ir_value()],
                "ld.global.cg.u64 $0, [$1];",
                "=l,l,~{memory}",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )
    else:
        value = Int32(
            llvm.inline_asm(
                T.i32(),
                [ptr.toint().ir_value()],
                "ld.global.cg.u32 $0, [$1];",
                "=r,l,~{memory}",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )
    return value


@cute.jit
def read_level(states, level: cutlass.Constexpr):
    """Completed prefix; callers only consume levels produced by a prior kernel."""
    record = states + level * LEVEL_STATE_WORDS
    return (
        Uint32(record[1]),
        Uint64(record[2]),
        Uint64(record[3]),
        Uint64(record[4]),
        Uint64(record[5]),
    )


@cute.jit
def resolve_level(
    hist,
    states,
    level: cutlass.Constexpr,
    need,
    chunks: cutlass.Constexpr,
    tid,
    local,
    scan,
    selected,
):
    """Arrive once per participating chunk; only the last CTA selects this level.

    Every flushing lane fences before the CTA barrier. The GPU acquire-release
    counter chains arrivals; its result is broadcast before the last CTA fences
    and stages completed bins through L1-bypassing loads. No CTA spins or waits
    for another CTA. Later kernels consume the five published output words.
    """
    record = states + level * LEVEL_STATE_WORDS
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()
    if tid == 0:
        selected[0] = cute.arch.atomic_add(record, Uint64(1), sem="acq_rel", scope="gpu")
    cute.arch.sync_threads()
    last = selected[0] == Uint64(chunks - 1)
    cute.arch.sync_threads()
    if last:
        cute.arch.fence_acq_rel_gpu()
        bins = 1024 if level == 2 else 2048
        shift = 21 if level == 0 else (10 if level == 1 else 0)
        key, remaining = Uint32(0), Uint64(need)
        above, equal, full = Uint64(0), Uint64(0), Uint64(0)
        if cutlass.const_expr(level > 0):
            key, remaining, above, equal, full = read_level(states, level - 1)
        for start in range(0, bins, THREADS):
            local[start + tid] = load_histogram(
                hist + level * 2048 + start + tid, local.element_type
            )
        cute.arch.sync_threads()
        index, preceding, equal, total = select_bin(
            local.iterator, bins, remaining, tid, scan, selected
        )
        if cutlass.const_expr(level == 0):
            full = total
        key |= Uint32(bins - 1 - index) << shift
        above += preceding
        remaining -= preceding
        if tid == 0:
            record[1], record[2], record[3] = Uint64(key), remaining, above
            record[4], record[5] = equal, full
            cute.arch.fence_acq_rel_gpu()


@cute.jit
def key_float(key):
    bits = ~key
    if (key & Uint32(0x80000000)) != 0:
        bits = key ^ Uint32(0x80000000)
    return Float32(llvm.bitcast(T.f32(), bits.ir_value()))


@cute.jit
def histogram_clear(hist, tid, bins: cutlass.Constexpr):
    for start in range(0, bins, THREADS):
        hist[start + tid] = hist.element_type(0)
    cute.arch.sync_threads()


@cute.jit
def histogram_flush(local, global_hist, tid, bins: cutlass.Constexpr):
    # Aggregate within each chunk before global atomics, including flat rows.
    cute.arch.sync_threads()
    for start in range(0, bins, THREADS):
        count = local[start + tid]
        if count != 0:
            cute.arch.atomic_add(global_hist + start + tid, count)


@cute.jit
def stage_probability_records(rec, staged, chunks: cutlass.Constexpr, tid):
    # Field 4 belongs to the later top-p phase and is not initialized yet.
    # Keep the five-word layout while copying only the four completed fields.
    layout = cute.make_layout((4, chunks), stride=(1, 5))
    stage_records(
        cute.make_tensor(rec, layout),
        cute.make_tensor(staged.iterator, layout),
        4 * chunks,
        tid,
    )


@cute.jit
def probability_record_reduce(
    rec, chunks: cutlass.Constexpr, chunk, tid, scratch, with_counts: cutlass.Constexpr
):
    """Warp-parallel maximum and exact preceding tie count; no FP sums."""
    recf = cute.recast_ptr(rec, dtype=Float32)
    scratch_i = cute.recast_ptr(scratch.iterator, dtype=Int32)
    if tid < 32:
        maximum, before = Float32(float("-inf")), Int32(0)
        for base in cutlass.range(0, chunks, 32, unroll_full=True):
            c = base + tid
            if c < chunks:
                maximum = max(maximum, recf[c * 5])
                if cutlass.const_expr(with_counts):
                    if c < chunk:
                        before += rec[c * 5 + 3]
        for k in cutlass.range_constexpr(5):
            maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1 << k))
            before += cute.arch.shuffle_sync_bfly(before, offset=1 << k)
        if tid == 0:
            scratch[0], scratch_i[2] = maximum, before
    cute.arch.sync_threads()
    maximum, before = scratch[0], scratch_i[2]
    cute.arch.sync_threads()
    return maximum, before


@cute.jit
def probability_state(
    states,
    rec,
    chunks: cutlass.Constexpr,
    chunk,
    k,
    temperature,
    v: cutlass.Constexpr,
    tid,
    scratch,
):
    """Recreate exactly P5's Z and preceding top-k tie count, without output reads."""
    tau, ties, equal = Uint32(0), Uint64(0), Uint64(0)
    if k < v:
        tau, ties, _, equal, _ = read_level(states, 2)
    smem = SmemAllocator()
    staged = smem.allocate_tensor(Int32, cute.make_layout(chunks * 5))
    stage_probability_records(rec, staged, chunks, tid)
    maximum, before = probability_record_reduce(staged.iterator, chunks, chunk, tid, scratch, True)
    recf = cute.recast_ptr(staged.iterator, dtype=Float32)
    if tid == 0:
        total = Float32(0)
        for c in cutlass.range(chunks, unroll_full=True):
            total = fp(total, recf[c * 5 + 2], "add")
        if k < v:
            scaled_max = fp(maximum, temperature, "div")
            tie_e = cute.exp(
                stable_exponent(key_float(tau), maximum, scaled_max, row_divisor(temperature)),
                fastmath=False,
            )
            total = fp(total, fp(Float32(ties), tie_e, "mul"), "add")
        scratch[0], scratch[1] = maximum, total
        # The count is an integer; don't round it through FP32 for large V.
        cute.recast_ptr(scratch.iterator, dtype=Int32)[2] = before
    cute.arch.sync_threads()
    maximum, total = scratch[0], scratch[1]
    before = Uint64(cute.recast_ptr(scratch.iterator, dtype=Int32)[2])
    cute.arch.sync_threads()
    return tau, ties, equal, maximum, total, before


@cute.jit
def chunk_reduce(
    partial, tid, scratch, results, maximum: cutlass.Constexpr, group: cutlass.Constexpr
):
    """probability_reduce for `group` independent chunks behind one barrier.

    Chunk k uses exactly probability_reduce's tree: lane butterflies, then warp
    k combines the sixteen warp values (missing lanes 0 or -inf) by butterflies.
    """
    lane, warp = tid % 32, tid // 32
    for k in cutlass.range_constexpr(group):
        value = partial[k]
        for s in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(value, offset=1 << s)
            if cutlass.const_expr(maximum):
                value = max(value, other)
            else:
                value = fp(value, other, "add")
        if lane == 0:
            scratch[k * WARPS + warp] = value
    cute.arch.sync_threads()
    if warp < group:
        value = Float32(0)
        if cutlass.const_expr(maximum):
            value = Float32(float("-inf"))
        if lane < WARPS:
            value = scratch[warp * WARPS + lane]
        for s in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(value, offset=1 << s)
            if cutlass.const_expr(maximum):
                value = max(value, other)
            else:
                value = fp(value, other, "add")
        if lane == 0:
            results[warp] = value
    cute.arch.sync_threads()


@cute.jit
def chunk_min(best, tid, choices, winners, group: cutlass.Constexpr):
    """Smallest candidate ID per chunk (reduce_choices' minimum), one barrier."""
    lane, warp = tid % 32, tid // 32
    for k in cutlass.range_constexpr(group):
        value = best[k]
        for s in cutlass.range_constexpr(5):
            value = min(value, cute.arch.shuffle_sync_bfly(value, offset=1 << s))
        if lane == 0:
            choices[k * WARPS + warp] = value
    cute.arch.sync_threads()
    if warp < group:
        value = Int32(0x7FFFFFFF)
        if lane < WARPS:
            value = choices[warp * WARPS + lane]
        for s in cutlass.range_constexpr(5):
            value = min(value, cute.arch.shuffle_sync_bfly(value, offset=1 << s))
        if lane == 0:
            winners[warp] = value
    cute.arch.sync_threads()


@cute.jit
def identity_row_max(recf, chunks: cutlass.Constexpr, tid, results, group: cutlass.Constexpr):
    """Row maximum of completed chunk records (order independent), once per CTA."""
    if tid < 32:
        maximum = Float32(float("-inf"))
        for base in cutlass.range(0, chunks, 32, unroll_full=True):
            if base + tid < chunks:
                maximum = max(maximum, recf[(base + tid) * 5])
        for k in cutlass.range_constexpr(5):
            maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1 << k))
        if tid == 0:
            results[group] = maximum
    cute.arch.sync_threads()
    return results[group]


@cute.jit
def warp_uniform(key):
    """Whether every lane of the (full) warp holds the same 32-bit key (vote.sync.all)."""
    first = cute.arch.shuffle_sync(key, 0)
    return Int32(
        cute.arch.inline_ptx(
            "{ .reg .pred same, all; setp.eq.u32 same, {$r0}, {$r1}; "
            "vote.sync.all.pred all, same, 0xffffffff; selp.u32 {$w0}, 1, 0, all; }",
            write_only_types=[Uint32],
            read_only_args=[Uint32(key), Uint32(first)],
        )
    )


@cute.jit
def histogram_count(hist, index, active):
    """hist[index] += 1 for active lanes; every warp lane must call it.

    A warp whose lanes all hit one bin (flat rows) adds its count once; otherwise
    each active lane adds 1. Integer sums: order-free.
    """
    key = Uint32(0xFFFFFFFF)
    if active:
        key = Uint32(index)
    if warp_uniform(key) != 0:
        if (cute.arch.lane_idx() == 0) & active:
            cute.arch.atomic_add(hist.iterator + index, Int32(32))
    elif active:
        cute.arch.atomic_add(hist.iterator + index, Int32(1))


# Mass histograms keep four 16-bit slices of each uint64 bin in native 32-bit shared
# atomics (sm_12x lowers 64-bit shared atomic adds to a CAS spin loop, and match.any /
# per-group redux.sync cost more than the adds). A bin's slice receives at most 4096
# addends below 2^16 per chunk (< 2^28); the flush recombines them modulo 2^64, exact
# because a bin's true mass (at most the chunk's probability mass) is below 2^63.
MASS_SLICES = 4
MASS_BINS = 2048


@cute.jit
def histogram_mass(hist, index, mass, active):
    """Sliced hist[index] += mass (uint64) for active lanes; every warp lane must call it.

    A warp whose lanes all hit one bin (flat rows) sums its masses first (the warp's
    probability mass is below 2^63) and one lane adds the nonzero slices; otherwise
    each active lane adds its nonzero slices.
    """
    key = Uint32(0xFFFFFFFF)
    value = Uint64(0)
    if active:
        key, value = Uint32(index), mass
    if warp_uniform(key) != 0:
        for s in cutlass.range_constexpr(5):
            low = cute.arch.shuffle_sync_bfly(Uint32(value), offset=1 << s)
            high = cute.arch.shuffle_sync_bfly(Uint32(value >> 32), offset=1 << s)
            value += (Uint64(high) << 32) | Uint64(low)
        if (cute.arch.lane_idx() == 0) & active:
            for slice_ in cutlass.range_constexpr(MASS_SLICES):
                part = Uint32((value >> (16 * slice_)) & Uint64(0xFFFF))
                if part != Uint32(0):
                    cute.arch.atomic_add(hist.iterator + slice_ * MASS_BINS + index, part)
    elif active:
        for slice_ in cutlass.range_constexpr(MASS_SLICES):
            part = Uint32((value >> (16 * slice_)) & Uint64(0xFFFF))
            if part != Uint32(0):
                cute.arch.atomic_add(hist.iterator + slice_ * MASS_BINS + index, part)


@cute.jit
def histogram_flush_mass(local, global_hist, tid, bins: cutlass.Constexpr):
    """histogram_flush for sliced mass bins: recombine, then one uint64 atomic per bin."""
    cute.arch.sync_threads()
    for start in range(0, bins, THREADS):
        index = start + tid
        total = Uint64(0)
        for slice_ in cutlass.range_constexpr(MASS_SLICES):
            total += Uint64(local[slice_ * MASS_BINS + index]) << (16 * slice_)
        if total != 0:
            cute.arch.atomic_add(global_hist + index, total)


@cute.jit
def large_state(states, rec, chunks: cutlass.Constexpr, chunk, top_k, b, use_k: cutlass.Constexpr,
                temperature, v: cutlass.Constexpr, tid, scratch):  # fmt: skip
    """P5's row state for one chunk (probability_state plus the per-row divisors).

    Returns (k, tau, kept ties, equal, maximum, scaled max, preceding ties, tdiv, zdiv).
    """
    k = Int32(v)
    if cutlass.const_expr(use_k):
        k = top_k[b]
    tau, ties, equal, maximum, total, before = probability_state(
        states, rec, chunks, chunk, k, temperature, v, tid, scratch
    )
    return (
        k,
        tau,
        ties,
        equal,
        maximum,
        fp(maximum, temperature, "div"),
        before,
        row_divisor(temperature),
        row_divisor(total),
    )


@cute.jit
def p5_tile(source, base, v: cutlass.Constexpr, large, tie_count, tid, scan):
    """One tile of P5's probabilities for a large row, bit for bit, and its tie count.

    Same loads, keys, smallest-ID top-k tie rule, exponent, exp and division as P5;
    every thread of the CTA calls it (the split-tie scan has barriers).
    """
    k, tau, kept_ties, equal, maximum, scaled_max, _, tdiv, zdiv = large
    values = probability_load4(source, base, v)
    keys = cute.make_rmem_tensor(cute.make_layout(VEC), Uint32)
    local_ties = Uint64(0)
    for j in cutlass.range_constexpr(VEC):
        keys[j] = ordered_key(values[j])
        if (base + j < v) & (k < v) & (keys[j] == tau):
            local_ties += Uint64(1)
    before, tile_ties = Uint64(0), Uint64(0)
    if (k < v) & (equal > kept_ties):
        before, tile_ties = integer_scan(local_ties, tid, scan)
    written = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    written.fill(Float32(0))
    for j in cutlass.range_constexpr(VEC):
        if base + j < v:
            is_tie = (k < v) & (keys[j] == tau)
            keep = (k == v) | (keys[j] > tau)
            keep = keep | (is_tie & ((equal == kept_ties) | (tie_count + before < kept_ties)))
            if is_tie:
                before += Uint64(1)
            prob = Float32(0)
            if keep:
                z = stable_exponent(values[j], maximum, scaled_max, tdiv)
                e = cute.exp(z, fastmath=False)
                prob = div_rn_by(e, zdiv)
            written[j] = prob
    return written, tile_ties


@cute.jit
def top_p_state(states, rec, chunk, p):
    """P9's row state: (full, threshold, tau, kept ties, split, scale, preceding ties)."""
    threshold = mass62(p, True)
    active = p < Float32(1)
    tau, need, above, equal, full = read_level(states, 0)
    if active & (full >= threshold):
        tau, need, above, equal, full = read_level(states, 2)
    fixed_sum, kept_ties, split = full, Uint64(0), Int32(0)
    if active & (full >= threshold):
        tau_value = Float32(llvm.bitcast(T.f32(), tau.ir_value()))
        unit = mass62(tau_value)
        kept_ties = (need + unit - Uint64(1)) // unit
        fixed_sum = above + kept_ties * unit
        split = Int32(equal > kept_ties * unit)
    scale = fp(Float32(fixed_sum), Float32(2.0**-62), "mul")
    before = Int32(0)
    if split != 0:
        for c in cutlass.range(chunk):
            before += rec[c * 5 + 4]
    return full, threshold, tau, kept_ties, split, scale, Uint64(before)


@cute.jit
def p9_tile(values, base, v: cutlass.Constexpr, top, tie_count, tid, scan):
    """One tile of P9's top-p filter and renormalization, bit for bit."""
    full, threshold, tau, kept_ties, split, scale, _ = top
    local, before, tile_ties = Uint64(0), Uint64(0), Uint64(0)
    if split != 0:
        for j in cutlass.range_constexpr(VEC):
            if (base + j < v) & (float_bits(values[j]) == tau):
                local += Uint64(1)
        before, tile_ties = integer_scan(local, tid, scan)
    written = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
    written.fill(Float32(0))
    for j in cutlass.range_constexpr(VEC):
        if base + j < v:
            key, value = float_bits(values[j]), Float32(0)
            keep = (
                (full < threshold)
                | (key > tau)
                | ((key == tau) & ((split == 0) | (tie_count + before < kept_ties)))
            )
            if key == tau:
                before += Uint64(1)
            if keep:
                value = fp(values[j], scale, "div")
            written[j] = value
    return written, tile_ties


class OrdinaryIdentity:
    """Identity-filter ordinary draw without an FP32 probability round trip.

    P1/P4 are Probabilities' identity phases (chunk max/argmax and Z records).
    D1 recomputes each chunk's p = div.rn(exp(z), Z) from the logits, bit for bit
    the values P5 would store, and sums them as Chain's sample partial does (its
    tile_scan carry: per-thread running sums of four tokens, the balanced warp and
    8-warp trees, then the four 1024-token tiles in order; computed here with
    butterflies behind one barrier for `group` chunks per CTA). D2 is Chain's
    sample finalize, rereading one chunk of logits. Min-p masks p below
    FP32(min_p * max p), where max p = div.rn(1, Z) exactly (the maximum's exponent
    is 0). T=0 rows take the P1 argmax records. Four launches. greedy_only (the
    greedy_records chunk group, else False) treats every live row as T=0 and
    launches only greedy_records and the finalize (two launches).
    """

    def __init__(
        self,
        vocab,
        batch,
        group,
        domain,
        min_p=False,
        greedy_only=False,
        looped=False,
        strided=False,
    ):
        self.v, self.n = vocab, (vocab + CHUNK - 1) // CHUNK
        self.strided = strided
        # looped=False: P4/D1 CTAs take one chunk group each (compile-time loop of 1,
        # the B=1-4 code). looped=True reads a runtime per-CTA group count.
        self.looped = looped
        self.domain, self.min_p, self.greedy_only = domain, min_p, greedy_only
        self.probabilities = Probabilities(
            1, vocab, "none", batch, group, strided=strided, unit_length=True
        )
        self.group = self.probabilities.group
        if greedy_only:  # greedy_records reduces a whole group in registers.
            self.group = max(1, min(greedy_only, self.n))
        self.groups = (self.n + self.group - 1) // self.group
        # P4 and D1 CTAs each take `loop` consecutive chunk groups (one row prologue).

    def config_key(self):
        return (
            self.v,
            self.group,
            self.domain,
            self.min_p,
            bool(self.greedy_only),
            self.looped,
            self.strided,
        )

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        temperatures: cute.Tensor,
        min_p: cute.Tensor,
        slots: cute.Tensor,
        seeds: cute.Tensor,
        cursors: cute.Tensor,
        workspace: cute.Tensor,
        work: cute.Tensor,
        out: cute.Tensor,
        loop: Int32,
        stream: cuda.CUstream,
    ):
        rows = Int64(slots.shape[0])
        looped = self.groups
        static = 1
        if cutlass.const_expr(self.looped):
            loop = cutlass.min(loop, self.groups)
            looped = (self.groups + loop - 1) // loop
            static = 0
        records = cute.recast_ptr(workspace.iterator + rows * 5120 * 12, dtype=Int32)
        maxima = cute.recast_ptr(records, dtype=Float32)
        target = cute.recast_ptr(work.iterator, dtype=Float32)  # P5 is not launched.
        if cutlass.const_expr(self.greedy_only):
            self.greedy_records(
                logits.iterator, slots.iterator, records, Int64(logits.stride[0])
            ).launch(
                grid=(rows, 1, self.groups),
                block=(THREADS, 1, 1),
                use_pdl=PDL,
                stream=stream,
            )
        for phase in cutlass.range_constexpr(0 if self.greedy_only else 2):
            self.probabilities.identity_kernel(
                logits.iterator,
                temperatures.iterator,
                slots.iterator,
                target,
                records,
                maxima,
                phase,
                1 if phase == 0 else static,
                Int32(1) if phase == 0 else loop,
                Int64(1),
                Int64(logits.stride[0]),
                Int64(0),
            ).launch(
                grid=(rows, 1, self.groups if phase == 0 else looped),
                block=(THREADS, 1, 1),
                use_pdl=PDL,
                stream=stream,
            )
        for finalize in cutlass.range_constexpr(1 if self.greedy_only else 0, 2):
            self.kernel(
                logits.iterator,
                temperatures.iterator,
                min_p.iterator,
                slots.iterator,
                seeds.iterator,
                cursors.iterator,
                records,
                work.iterator,
                out.iterator,
                finalize == 1,
                static,
                loop,
                Int64(logits.stride[0]),
            ).launch(
                grid=(rows, 1, 1 if finalize == 1 else looped),
                block=(THREADS, 1, 1),
                use_pdl=PDL,
                stream=stream,
            )

    @cute.kernel
    def greedy_records(
        self, logits: cute.Pointer, slots: cute.Pointer, records: cute.Pointer, stride_b: Int64
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        # greedy_only P1: one (max, smallest-ID argmax) record per `group` chunks.
        # The record lands in the group's first chunk; the group's other chunks get
        # (-inf, V), which never wins the finalize. The (value desc, ID asc) order is
        # total on non-NaN values, so the row token equals the per-chunk P1 records'.
        # Every lane loads eight consecutive tokens per chunk (one 128-bit load for
        # 16-bit logits on aligned rows).
        b, _, g = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group: cutlass.Constexpr = self.group
        smem = SmemAllocator()
        tops = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        winners = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        source = logits + Int64(b) * self.v
        if cutlass.const_expr(self.strided):
            source = logits + Int64(b) * stride_b
        rec = records + Int64(b) * self.n * 5
        recf = cute.recast_ptr(rec, dtype=Float32)
        if slots[b] >= 0:
            top, winner = Float32(float("-inf")), Int32(self.v)
            wide: cutlass.Constexpr = logits.value_type.width == 16
            for k in cutlass.range_constexpr(group):
                base = (g * group + k) * CHUNK + tid * (2 * VEC)
                native = cute.make_rmem_tensor(cute.make_layout(2 * VEC), logits.value_type)
                native.fill(0)
                if cutlass.const_expr(wide):
                    if (((source + base).toint() & 15) == 0) & (base + 2 * VEC <= self.v):
                        aligned = cute.make_ptr(
                            logits.value_type,
                            (source + base).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        atom = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), logits.value_type, num_bits_per_copy=128
                        )
                        cute.copy(
                            atom, cute.make_tensor(aligned, cute.make_layout(2 * VEC)), native
                        )
                    else:
                        for j in cutlass.range_constexpr(2 * VEC):
                            if base + j < self.v:
                                native[j] = source[base + j]
                    for j in cutlass.range_constexpr(2 * VEC):
                        z = Float32(native[j])
                        if base + j < self.v:
                            if (z > top) | ((z == top) & (base + j < winner)):
                                top, winner = z, base + j
                else:
                    for half in cutlass.range_constexpr(2):
                        loaded = load4(source, base + half * VEC, self.v)
                        for j in cutlass.range_constexpr(VEC):
                            z = loaded[j]
                            if base + half * VEC + j < self.v:
                                if (z > top) | ((z == top) & (base + half * VEC + j < winner)):
                                    top, winner = z, base + half * VEC + j
            top, winner = warp_argmax(top, winner)
            if lane == 0:
                tops[warp], winners[warp] = top, winner
            cute.arch.sync_threads()
            if warp == 0:
                top, winner = Float32(float("-inf")), Int32(self.v)
                if lane < WARPS:
                    top, winner = tops[lane], winners[lane]
                top, winner = warp_argmax(top, winner)
                if lane < group:
                    chunk = g * group + lane
                    if chunk < self.n:
                        if lane == 0:
                            recf[chunk * 5], rec[chunk * 5 + 1] = top, winner
                        else:
                            recf[chunk * 5], rec[chunk * 5 + 1] = (
                                Float32(float("-inf")),
                                Int32(self.v),
                            )

    @cute.jit
    def probabilities4(
        self, source, base, maximum, scaled_max, tdiv, zdiv, pivot, huge: cutlass.Constexpr
    ):
        """p for this thread's four tokens at base (0 outside the row or below pivot)."""
        loaded = probability_load4(source, base, self.v)
        values = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
        for j in cutlass.range_constexpr(VEC):
            z = row_exponent(loaded[j], maximum, scaled_max, tdiv, huge)
            e = cute.exp(z, fastmath=False)
            values[j] = Float32(0)
            if base + j < self.v:
                values[j] = div_rn_by(e, zdiv)
                if cutlass.const_expr(self.min_p):
                    if values[j] < pivot:
                        values[j] = Float32(0)
        return values

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        temperatures: cute.Pointer,
        min_p: cute.Pointer,
        slots: cute.Pointer,
        seeds: cute.Pointer,
        cursors: cute.Pointer,
        records: cute.Pointer,
        work: cute.Pointer,
        out: cute.Pointer,
        finalize: cutlass.Constexpr,
        static: cutlass.Constexpr,
        loop: Int32,
        stride_b: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, _, g = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group: cutlass.Constexpr = self.group
        smem = SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout(WARPS + THREADS // TILE_THREADS))
        prefixes = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        choices = smem.allocate_tensor(Int32, cute.make_layout(group * WARPS))
        lasts = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        halves = smem.allocate_tensor(Float32, cute.make_layout(group * 2 * WARPS))
        results = smem.allocate_tensor(Float32, cute.make_layout(4))
        state = smem.allocate_tensor(Int32, cute.make_layout(4))
        draw_state = smem.allocate_tensor(Float32, cute.make_layout(2))
        staged = smem.allocate_tensor(Float32, cute.make_layout(max(2 * self.n, 1)))
        staged_i = cute.recast_ptr(staged.iterator, dtype=Int32)
        best = cute.make_rmem_tensor(cute.make_layout(group), Int32)
        wf = cute.make_ptr(Float32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        wi = cute.make_ptr(Int32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        rec = records + Int64(b) * self.n * 5
        recf = cute.recast_ptr(rec, dtype=Float32)
        record = Int64(b) * self.n * 2
        source = logits + Int64(b) * self.v
        if cutlass.const_expr(self.strided):
            source = logits + Int64(b) * stride_b
        slot = slots[b]
        if cutlass.const_expr(finalize):
            if tid == 0:
                out[b] = Int32(-1)
            cute.arch.sync_threads()
        if slot >= 0:
            if cutlass.const_expr(self.greedy_only):
                temperature = Float32(0)  # Temperatures are not read.
            else:
                temperature = temperatures[b]
            if temperature == Float32(0):
                if cutlass.const_expr(finalize):
                    if tid < 32:
                        top, winner = Float32(float("-inf")), Int32(self.v)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            c = base + tid
                            if c < self.n:
                                value, token = recf[c * 5], rec[c * 5 + 1]
                                if (value > top) | ((value == top) & (token < winner)):
                                    top, winner = value, token
                        top, winner = warp_argmax(top, winner)
                        if tid == 0:
                            # An all -inf row has no winner; P5's all-zero row drew token 0.
                            if winner >= self.v:
                                winner = Int32(0)
                            out[b] = winner
            else:
                tdiv = row_divisor(temperature)
                maximum = identity_row_max(recf, self.n, tid, results, 0)
                for start in cutlass.range(0, self.n, THREADS, unroll_full=True):
                    if start + tid < self.n:
                        staged[start + tid] = recf[(start + tid) * 5 + 2]
                cute.arch.sync_threads()
                if tid == 0:
                    total = Float32(0)
                    for c in cutlass.range(self.n, unroll_full=True):
                        total = fp(total, staged[c], "add")
                    results[1] = total
                cute.arch.sync_threads()
                zdiv = row_divisor(results[1])
                scaled_max = fp(maximum, temperature, "div")
                pivot = Float32(0)
                if cutlass.const_expr(self.min_p):
                    if min_p[b] > Float32(0):
                        pivot = fp(min_p[b], div_rn_by(Float32(1), zdiv), "mul")
                if cutlass.const_expr(not finalize):
                    steps = loop
                    if cutlass.const_expr(static > 0):
                        steps = static  # compile-time count: the unlooped code
                    for step in cutlass.range(steps):
                        gg = g * steps + step
                        if gg < self.groups:
                            for huge in cutlass.range_constexpr(2):
                                use = temperature < Float32(HUGE_T)
                                if cutlass.const_expr(huge == 1):
                                    use = temperature >= Float32(HUGE_T)
                                if use:
                                    for k in cutlass.range_constexpr(group):
                                        last = Int32(-1)
                                        for half in cutlass.range_constexpr(2):
                                            base = (gg * group + k) * CHUNK + half * (THREADS * VEC)
                                            base += tid * VEC
                                            values = self.probabilities4(
                                                source, base, maximum, scaled_max, tdiv, zdiv, pivot,
                                                huge == 1,
                                            )  # fmt: skip
                                            running = Float32(0)
                                            for j in cutlass.range_constexpr(VEC):
                                                running = fp(running, values[j], "add")
                                                if values[j] > Float32(0):
                                                    last = max(last, base + j)
                                            for s in cutlass.range_constexpr(5):
                                                other = cute.arch.shuffle_sync_bfly(
                                                    running, offset=1 << s
                                                )
                                                running = fp(running, other, "add")
                                            if lane == 0:
                                                halves[(k * 2 + half) * WARPS + warp] = running
                                        best[k] = last
                            for k in cutlass.range_constexpr(group):
                                value = best[k]
                                for s in cutlass.range_constexpr(5):
                                    value = max(
                                        value, cute.arch.shuffle_sync_bfly(value, offset=1 << s)
                                    )
                                if lane == 0:
                                    choices[k * WARPS + warp] = value
                            cute.arch.sync_threads()
                            if tid < group:
                                c = gg * group + tid
                                if c < self.n:
                                    total = Float32(0)
                                    for half in cutlass.range_constexpr(2):
                                        for tile in cutlass.range_constexpr(2):
                                            w = (tid * 2 + half) * WARPS + tile * 8
                                            low = fp(fp(halves[w], halves[w + 1], "add"),
                                                     fp(halves[w + 2], halves[w + 3], "add"), "add")  # fmt: skip
                                            high = fp(fp(halves[w + 4], halves[w + 5], "add"),
                                                      fp(halves[w + 6], halves[w + 7], "add"), "add")  # fmt: skip
                                            total = fp(total, fp(low, high, "add"), "add")
                                    last = Int32(-1)
                                    for w in cutlass.range_constexpr(WARPS):
                                        last = max(last, choices[tid * WARPS + w])
                                    wf[record + c * 2], wi[record + c * 2 + 1] = total, last
                            # The next group rewrites halves/choices.
                            cute.arch.sync_threads()
                else:
                    stage_records(wi + record, staged_i, self.n * 2, tid)
                    if tid < 32:
                        last = Int32(-1)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            c = base + tid
                            if c < self.n:
                                last = max(last, staged_i[c * 2 + 1])
                        for s in cutlass.range_constexpr(5):
                            last = max(last, cute.arch.shuffle_sync_bfly(last, offset=1 << s))
                        if tid == 0:
                            state[2] = last
                    cute.arch.sync_threads()
                    if tid == 0:
                        total = Float32(0)
                        for c in cutlass.range(self.n, unroll_full=True):
                            total = fp(total, staged[c * 2], "add")
                        u = uniform(
                            seeds[Int64(slot) * 2],
                            seeds[Int64(slot) * 2 + 1],
                            cursors[b],
                            Int32(self.domain),
                            Int32(0),
                        )
                        threshold = fp(u, total, "mul")
                        carry, exclusive = Float32(0), Float32(0)
                        selected = Int32(-1)
                        for c in cutlass.range(self.n, unroll_full=True):
                            before = carry
                            carry = fp(carry, staged[c * 2], "add")
                            if (selected == -1) & (carry > threshold):
                                selected, exclusive = c, before
                        state[1] = selected
                        draw_state[0], draw_state[1] = threshold, exclusive
                    cute.arch.sync_threads()
                    chosen = state[1]
                    result = Int32(self.v)
                    # A rounding spill (no chunk crosses the threshold) takes the row's last.
                    if chosen >= 0:
                        threshold, carry = draw_state[0], draw_state[1]
                        choice = Int32(self.v)
                        for huge in cutlass.range_constexpr(2):
                            use = temperature < Float32(HUGE_T)
                            if cutlass.const_expr(huge == 1):
                                use = temperature >= Float32(HUGE_T)
                            if use:
                                for start in cutlass.range_constexpr(0, CHUNK, THREADS * VEC):
                                    base = chosen * CHUNK + start + tid * VEC
                                    values = self.probabilities4(
                                        source, base, maximum, scaled_max, tdiv, zdiv, pivot,
                                        huge == 1,
                                    )  # fmt: skip
                                    local = inclusive4(values)
                                    preceding, tile_carry, carry = tile_scan(
                                        local[VEC - 1], carry, tid, sums, prefixes
                                    )
                                    for j in cutlass.range_constexpr(VEC):
                                        if base + j < self.v:
                                            prefix = fp(
                                                tile_carry, fp(preceding, local[j], "add"), "add"
                                            )
                                            if prefix > threshold:
                                                choice = min(choice, base + j)
                        result, _ = reduce_choices(choice, Int32(-1), tid, choices, lasts)
                    if tid == 0:
                        if result == self.v:
                            result = state[2]
                        out[b] = result


class OrdinaryFiltered:
    """Filtered ordinary draw: Probabilities without large-row FP32 probabilities.

    Probabilities(materialize=False) stores only compact rows' probabilities; large
    rows' P5 values are recomputed from the logits (p5_tile) and filtered and
    renormalized as P9 would (p9_tile), so D1/D2 reduce exactly the values the
    materialized path writes, in Chain's sample order. Min-p uses P5's chunk
    maxima, divided by the top-p scale (division is monotone and the maximum is
    always kept). T=0 rows take the P1 argmax records.
    """

    def __init__(
        self, vocab, batch, filters, domain, min_p=False, compact_reserve=0, strided=False
    ):
        self.v, self.n = vocab, (vocab + CHUNK - 1) // CHUNK
        self.strided = strided
        self.domain, self.min_p = domain, min_p
        self.use_k = filters in ("auto", "both", "top_k")
        self.use_p = filters in ("auto", "both", "top_p")
        self.probabilities = Probabilities(
            1,
            vocab,
            filters,
            batch,
            1,
            output_maxima=min_p,
            materialize=False,
            compact_reserve=compact_reserve,
            strided=strided,
            unit_length=True,
        )

    def config_key(self):
        return (self.v, self.domain, self.min_p, self.probabilities.config_key())

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        temperatures: cute.Tensor,
        top_k: cute.Tensor,
        top_p: cute.Tensor,
        min_p: cute.Tensor,
        slots: cute.Tensor,
        seeds: cute.Tensor,
        cursors: cute.Tensor,
        probs: cute.Tensor,
        workspace: cute.Tensor,
        work: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = Int64(slots.shape[0])
        self.probabilities(logits, temperatures, top_k, top_p, slots, probs, workspace, stream)
        records = cute.recast_ptr(workspace.iterator + rows * 5120 * 12, dtype=Int32)
        compact_offset = rows * 5120 * 12 + ((rows * self.n * 20 + 255) // 256) * 256
        state_offset = compact_offset + ((rows * (2 * COMPACT_CAP + 1) * 4 + 255) // 256) * 256
        states = cute.recast_ptr(workspace.iterator + state_offset, dtype=Uint64)
        lists = cute.recast_ptr(workspace.iterator + compact_offset, dtype=Int32)
        maxima = cute.recast_ptr(records, dtype=Float32)
        if cutlass.const_expr(self.min_p):
            offset = state_offset + ((rows * ROW_STATE_WORDS * 8 + 255) // 256) * 256
            maxima = cute.recast_ptr(workspace.iterator + offset, dtype=Float32)
        for finalize in cutlass.range_constexpr(2):
            self.kernel(
                logits.iterator,
                temperatures.iterator,
                top_k.iterator,
                top_p.iterator,
                min_p.iterator,
                slots.iterator,
                seeds.iterator,
                cursors.iterator,
                lists,
                records,
                states,
                maxima,
                work.iterator,
                out.iterator,
                finalize == 1,
                Int64(logits.stride[0]),
            ).launch(
                grid=(rows, 1, 1 if finalize == 1 else self.n),
                block=(THREADS, 1, 1),
                use_pdl=PDL,
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        temperatures: cute.Pointer,
        top_k: cute.Pointer,
        top_p: cute.Pointer,
        min_p: cute.Pointer,
        slots: cute.Pointer,
        seeds: cute.Pointer,
        cursors: cute.Pointer,
        lists: cute.Pointer,
        records: cute.Pointer,
        states: cute.Pointer,
        maxima: cute.Pointer,
        work: cute.Pointer,
        out: cute.Pointer,
        finalize: cutlass.Constexpr,
        stride_b: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, _, chunk = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem = SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout(WARPS + THREADS // TILE_THREADS))
        prefixes = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        choices = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        lasts = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        scratch = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        scan = smem.allocate_tensor(Uint64, cute.make_layout(2 * WARPS))
        state = smem.allocate_tensor(Int32, cute.make_layout(4))
        draw_state = smem.allocate_tensor(Float32, cute.make_layout(3))
        staged = smem.allocate_tensor(Float32, cute.make_layout(max(2 * self.n, 1)))
        staged_i = cute.recast_ptr(staged.iterator, dtype=Int32)
        if cutlass.const_expr(self.use_k):
            # A compact row's chunk, rebuilt from its (token, probability) list.
            dense = smem.allocate_tensor(Float32, cute.make_layout(CHUNK), byte_alignment=16)
        wf = cute.make_ptr(Float32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        wi = cute.make_ptr(Int32, work.toint(), cute.AddressSpace.gmem, assumed_align=4)
        rec = records + Int64(b) * self.n * 5
        recf = cute.recast_ptr(rec, dtype=Float32)
        ks = states + Int64(b) * ROW_STATE_WORDS
        ps = ks + 3 * LEVEL_STATE_WORDS
        record = Int64(b) * self.n * 2
        source = logits + Int64(b) * self.v
        if cutlass.const_expr(self.strided):
            source = logits + Int64(b) * stride_b
        ids = lists + Int64(b) * (2 * COMPACT_CAP + 1)
        compact_values = cute.recast_ptr(ids + COMPACT_CAP, dtype=Float32)
        slot = slots[b]
        if cutlass.const_expr(finalize):
            if tid == 0:
                out[b] = Int32(-1)
            cute.arch.sync_threads()
        if slot >= 0:
            temperature = temperatures[b]
            if temperature == Float32(0):
                if cutlass.const_expr(finalize):
                    if tid < 32:
                        top, winner = Float32(float("-inf")), Int32(self.v)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            c = base + tid
                            if c < self.n:
                                value, token = recf[c * 5], rec[c * 5 + 1]
                                if (value > top) | ((value == top) & (token < winner)):
                                    top, winner = value, token
                        top, winner = warp_argmax(top, winner)
                        if tid == 0:
                            # An all -inf row has no winner; P5's all-zero row drew token 0.
                            if winner >= self.v:
                                winner = Int32(0)
                            out[b] = winner
            else:
                compact = False
                if cutlass.const_expr(self.use_k):
                    compact = (top_k[b] < self.v) & (top_k[b] <= COMPACT_CAP)
                p = Float32(1)
                if cutlass.const_expr(self.use_p):
                    p = top_p[b]
                    if compact:
                        p = Float32(1)
                c = chunk
                if cutlass.const_expr(finalize):
                    stage_records(wi + record, staged_i, self.n * 2, tid)
                    if tid < 32:
                        last = Int32(-1)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            cc = base + tid
                            if cc < self.n:
                                last = max(last, staged_i[cc * 2 + 1])
                        for k in cutlass.range_constexpr(5):
                            last = max(last, cute.arch.shuffle_sync_bfly(last, offset=1 << k))
                        if tid == 0:
                            state[2] = last
                    cute.arch.sync_threads()
                    if tid == 0:
                        total = Float32(0)
                        for cc in cutlass.range(self.n, unroll_full=True):
                            total = fp(total, staged[cc * 2], "add")
                        u = uniform(
                            seeds[Int64(slot) * 2],
                            seeds[Int64(slot) * 2 + 1],
                            cursors[b],
                            Int32(self.domain),
                            Int32(0),
                        )
                        threshold = fp(u, total, "mul")
                        carry, exclusive = Float32(0), Float32(0)
                        selected = Int32(-1)
                        for cc in cutlass.range(self.n, unroll_full=True):
                            before = carry
                            carry = fp(carry, staged[cc * 2], "add")
                            if (selected == -1) & (carry > threshold):
                                selected, exclusive = cc, before
                        state[1] = selected
                        draw_state[0], draw_state[1] = threshold, exclusive
                    cute.arch.sync_threads()
                    c = state[1]
                proceed = Int32(1)
                if cutlass.const_expr(finalize):
                    proceed = Int32(c >= 0)
                result = Int32(self.v)
                if proceed == 1:
                    large = large_state(
                        ks, rec, self.n, c, top_k, b, self.use_k, temperature, self.v, tid, scratch
                    )
                    tie_k = large[6]
                    top = top_p_state(ps, rec, c, p)
                    tie_p = top[6]
                    pivot = Float32(0)
                    if cutlass.const_expr(self.min_p):
                        if min_p[b] > Float32(0):
                            if tid < 32:
                                pmax = Float32(0)
                                if compact:
                                    # P5's compact scatter maxima: the list's maximum.
                                    for index in cutlass.range(tid, top_k[b], 32):
                                        pmax = max(pmax, compact_values[index])
                                else:
                                    for cc in cutlass.range(tid, self.n, 32):
                                        pmax = max(pmax, maxima[Int64(b) * self.n + cc])
                                for k in cutlass.range_constexpr(5):
                                    pmax = max(
                                        pmax, cute.arch.shuffle_sync_bfly(pmax, offset=1 << k)
                                    )
                                if tid == 0:
                                    if p < Float32(1):
                                        pmax = fp(pmax, top[5], "div")
                                    draw_state[2] = fp(min_p[b], pmax, "mul")
                            cute.arch.sync_threads()
                            pivot = draw_state[2]
                    total = Float32(0)
                    carry, threshold = Float32(0), Float32(0)
                    if cutlass.const_expr(finalize):
                        threshold, carry = draw_state[0], draw_state[1]
                    choice, last = Int32(self.v), Int32(-1)
                    values = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                    if cutlass.const_expr(self.use_k):
                        if compact:
                            # Exactly the FP32 row P5 would scatter: list values, else 0.
                            for start in cutlass.range_constexpr(0, CHUNK, THREADS):
                                dense[start + tid] = Float32(0)
                            cute.arch.sync_threads()
                            for start in cutlass.range_constexpr(0, COMPACT_CAP, THREADS):
                                index = start + tid
                                if index < top_k[b]:
                                    token = ids[index]
                                    if (token >= c * CHUNK) & (
                                        token < min((c + 1) * CHUNK, self.v)
                                    ):
                                        dense[token - c * CHUNK] = compact_values[index]
                            cute.arch.sync_threads()
                    for start in cutlass.range_constexpr(0, CHUNK, THREADS * VEC):
                        base = c * CHUNK + start + tid * VEC
                        if compact:
                            for j in cutlass.range_constexpr(VEC):
                                values[j] = Float32(0)
                                if cutlass.const_expr(self.use_k):
                                    values[j] = dense[start + tid * VEC + j]
                        else:
                            computed, ties = p5_tile(source, base, self.v, large, tie_k, tid, scan)
                            tie_k += ties
                            for j in cutlass.range_constexpr(VEC):
                                values[j] = computed[j]
                            if cutlass.const_expr(self.use_p):
                                if p < Float32(1):
                                    filtered, ties = p9_tile(
                                        values, base, self.v, top, tie_p, tid, scan
                                    )
                                    tie_p += ties
                                    for j in cutlass.range_constexpr(VEC):
                                        values[j] = filtered[j]
                        if cutlass.const_expr(self.min_p):
                            for j in cutlass.range_constexpr(VEC):
                                if values[j] < pivot:
                                    values[j] = Float32(0)
                        local = inclusive4(values)
                        if cutlass.const_expr(finalize):
                            preceding, tile_carry, carry = tile_scan(
                                local[VEC - 1], carry, tid, sums, prefixes
                            )
                            for j in cutlass.range_constexpr(VEC):
                                if base + j < self.v:
                                    prefix = fp(tile_carry, fp(preceding, local[j], "add"), "add")
                                    if prefix > threshold:
                                        choice = min(choice, base + j)
                        else:
                            _, _, total = tile_scan(local[VEC - 1], total, tid, sums, prefixes)
                            for j in cutlass.range_constexpr(VEC):
                                if base + j < self.v:
                                    if values[j] > Float32(0):
                                        last = max(last, base + j)
                    result, last = reduce_choices(choice, last, tid, choices, lasts)
                    if cutlass.const_expr(not finalize):
                        if tid == 0:
                            wf[record + chunk * 2], wi[record + chunk * 2 + 1] = total, last
                if cutlass.const_expr(finalize):
                    if tid == 0:
                        if result == self.v:
                            result = state[2]
                        out[b] = result


class Probabilities:
    """512 threads per chunk; compact k<=1024 adds one CTA per row.

    P0 clears count/mass histograms, arrival counters and compact lists; P1 max/argmax+count11;
    P2/P3 count radix; P4 unchanged Z_gt/tie records plus compact IDs; C resolves
    compact p/top-p; P5 writes compact zero/scatter, greedy/padding, or large
    FP32 probabilities plus mass11 for top-p. Large P6/P7/P8 read those FP32
    probabilities for mass radix/tie counts; P9 filters/renormalizes in place.
    Integer atomics only; append order never changes Z, top-p decisions or S.

    Fixed launches none/top_k/top_p/both/auto: 3/7/8/11/11. Compact top-k (with
    or without top-p): four logits passes plus zero/scatter. Large identity: 3
    logits passes; top-k: 5. Large top-p: 3 (identity k) or 5 logits passes,
    plus 3 FP32 probability reads (+1 split tie) and an in-place output rewrite.
    Compact rows skip large-path work and vice versa. P0 touches scratch only.
    """

    def __init__(
        self,
        length,
        vocab,
        filters="auto",
        batch=1,
        group=1,
        output_maxima=False,
        materialize=True,
        flat=False,
        compact_reserve=0,
        strided=False,
        unit_length=False,
    ):
        # Legacy length/batch arguments remain for standalone tuning callers;
        # all row geometry is read from tensors at runtime, never stored here.
        self.v = vocab
        # flat=True (only for L > 65535, grid.y's limit) puts B*L rows on grid.x;
        # otherwise grid = (B, L, chunks) as in the per-shape kernels.
        self.flat = flat
        # Dynamic shared memory requested for the compact CTA (0: its own usage).
        # One compact CTA per row runs a long serial pass; under PDL its CTAs are
        # placed while the previous phase drains and can pack two to an SM, which
        # slows both. Requesting more than half of an SM's shared memory keeps them
        # one per SM (measured: B=1-4 top-k/top-p 6-10% faster on RTX 5090).
        self.compact_reserve = compact_reserve
        # strided=False: logits rows are dense (row stride V, constant), the layout
        # every engine call uses and the per-shape kernels assumed; True reads the
        # runtime row strides. unit_length: rank-2 callers (L=1, compile time).
        self.strided, self.unit_length = strided, unit_length
        self.output_maxima = output_maxima
        # materialize=False (the ordinary draw): only compact rows store probabilities;
        # P6-P8 recompute large rows' P5 values from the logits and P9 is left to the draw.
        self.materialize = materialize
        self.n = (vocab + CHUNK - 1) // CHUNK
        self.compact_cap = COMPACT_CAP
        self.use_k = filters in ("auto", "both", "top_k")
        self.use_p = filters in ("auto", "both", "top_p")
        # filters="none" uses the identity path; group chunks per CTA is host static.
        self.identity = filters == "none"
        self.group = max(1, min(group, IDENTITY_GROUP, self.n))
        self.groups = (self.n + self.group - 1) // self.group

    def config_key(self):
        return (
            self.v,
            self.use_k,
            self.use_p,
            self.identity,
            self.group,
            self.output_maxima,
            self.materialize,
            self.compact_cap,
            self.flat,
            self.compact_reserve,
            self.strided,
            self.unit_length,
        )

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        temperatures: cute.Tensor,
        top_k: cute.Tensor,
        top_p: cute.Tensor,
        slots: cute.Tensor,
        out: cute.Tensor,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch = Int32(slots.shape[0])
        length = Int64(1)
        stride_l = Int64(0)
        if cutlass.const_expr(cute.rank(logits) == 3):
            length = Int64(logits.shape[1])
            stride_l = Int64(logits.stride[1])
        stride_b = Int64(logits.stride[0])
        rows = Int64(batch) * length
        counts = cute.recast_ptr(workspace.iterator, dtype=Int32)
        masses = cute.recast_ptr(workspace.iterator + rows * 5120 * 4, dtype=Uint64)
        records = cute.recast_ptr(workspace.iterator + rows * 5120 * 12, dtype=Int32)
        compact_offset = rows * 5120 * 12 + ((rows * self.n * 20 + 255) // 256) * 256
        lists = cute.recast_ptr(workspace.iterator + compact_offset, dtype=Int32)
        state_offset = compact_offset + ((rows * (2 * COMPACT_CAP + 1) * 4 + 255) // 256) * 256
        states = cute.recast_ptr(workspace.iterator + state_offset, dtype=Uint64)
        maxima = cute.recast_ptr(records, dtype=Float32)  # Unused alias in existing modes.
        if cutlass.const_expr(self.output_maxima):
            # Chain and ordinary sampling share min-p maxima after the row states.
            offset = state_offset + ((rows * ROW_STATE_WORDS * 8 + 255) // 256) * 256
            maxima = cute.recast_ptr(workspace.iterator + offset, dtype=Float32)
        if cutlass.const_expr(self.identity):
            for phase in cutlass.range_constexpr(3):
                self.identity_kernel(
                    logits.iterator,
                    temperatures.iterator,
                    slots.iterator,
                    out.iterator,
                    records,
                    maxima,
                    phase,
                    1,
                    Int32(1),
                    length,
                    stride_b,
                    stride_l,
                ).launch(
                    grid=(rows, 1, self.groups)
                    if cutlass.const_expr(self.flat)
                    else (batch, length, self.groups),
                    block=(THREADS, 1, 1),
                    use_pdl=PDL,
                    stream=stream,
                )
        else:
            if cutlass.const_expr(self.use_k or self.use_p):
                self.zero(counts, masses, lists, states, length).launch(
                    grid=(rows, 1, 10) if cutlass.const_expr(self.flat) else (batch, length, 10),
                    block=(THREADS, 1, 1),
                    use_pdl=PDL,
                    stream=stream,
                )
            for phase in cutlass.range_constexpr(1, 10):
                if cutlass.const_expr(
                    phase in (1, 4, 5)
                    or (self.use_k and phase in (2, 3))
                    or (self.use_p and phase in (6, 7, 8))
                    or (self.use_p and phase == 9 and self.materialize)
                ):
                    self.kernel(
                        logits.iterator,
                        temperatures.iterator,
                        top_k.iterator,
                        top_p.iterator,
                        slots.iterator,
                        out.iterator,
                        counts,
                        masses,
                        records,
                        lists,
                        states,
                        maxima,
                        phase,
                        length,
                        stride_b,
                        stride_l,
                    ).launch(
                        grid=(rows, 1, self.n)
                        if cutlass.const_expr(self.flat)
                        else (batch, length, self.n),
                        block=(THREADS, 1, 1),
                        use_pdl=PDL,
                        stream=stream,
                    )
                    if cutlass.const_expr(phase == 4 and self.use_k):
                        self.compact(
                            logits.iterator,
                            temperatures.iterator,
                            top_k.iterator,
                            top_p.iterator,
                            slots.iterator,
                            states,
                            records,
                            lists,
                            length,
                            stride_b,
                            stride_l,
                        ).launch(
                            grid=(rows, 1, 1)
                            if cutlass.const_expr(self.flat)
                            else (batch, length, 1),
                            block=(THREADS, 1, 1),
                            smem=self.compact_reserve if self.compact_reserve else None,
                            use_pdl=PDL,
                            stream=stream,
                        )

    @cute.jit
    def row_source(self, logits, b, position, row, stride_b, stride_l):
        if cutlass.const_expr(self.strided):
            return logits + Int64(b) * stride_b + Int64(position) * stride_l
        return logits + row * self.v

    @cute.kernel
    def identity_kernel(
        self,
        logits: cute.Pointer,
        temperatures: cute.Pointer,
        slots: cute.Pointer,
        out: cute.Pointer,
        records: cute.Pointer,
        maxima: cute.Pointer,
        phase: cutlass.Constexpr,
        static: cutlass.Constexpr,
        loop: Int32,
        length: Int64,
        stride_b: Int64,
        stride_l: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        # P1, P4 and P5 for identity filters, with `group` consecutive chunks per CTA
        # (P4: `loop` consecutive groups per CTA, sharing the row prologue).
        #
        # Per-chunk thread ownership, element order and reduction trees are those
        # of the per-chunk phases; only independent chunk reductions are batched
        # behind one barrier, and row state (max, Z) is staged once per CTA.
        b, position, g = cute.arch.block_idx()
        if cutlass.const_expr(self.flat):
            position = b % Int32(length)
            b = b // Int32(length)
        if cutlass.const_expr(self.unit_length):
            length = 1
        tid, _, _ = cute.arch.thread_idx()
        group: cutlass.Constexpr = self.group
        row = Int64(b) * length + position
        source, target = (
            self.row_source(logits, b, position, row, stride_b, stride_l),
            out + row * self.v,
        )
        rec = records + row * self.n * 5
        recf = cute.recast_ptr(rec, dtype=Float32)
        smem = SmemAllocator()
        scratch = smem.allocate_tensor(Float32, cute.make_layout(group * WARPS))
        choices = smem.allocate_tensor(Int32, cute.make_layout(group * WARPS))
        results = smem.allocate_tensor(Float32, cute.make_layout(group + 2))
        winners = smem.allocate_tensor(Int32, cute.make_layout(group + 1))
        if cutlass.const_expr(phase == 2):
            staged = smem.allocate_tensor(Float32, cute.make_layout(self.n))
        values = cute.make_rmem_tensor(cute.make_layout(group * 2 * VEC), Float32)
        partial = cute.make_rmem_tensor(cute.make_layout(group), Float32)
        if cutlass.const_expr(self.output_maxima and phase == 2):
            partial.fill(Float32(0))
        written = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
        if slots[b] == -1:
            if cutlass.const_expr(phase == 2):
                written.fill(Float32(0))
                for k in cutlass.range_constexpr(group):
                    for half in cutlass.range_constexpr(2):
                        base = (g * group + k) * CHUNK + half * (THREADS * VEC) + tid * VEC
                        store4(target, base, self.v, written)
        else:
            temperature = temperatures[b]
            tdiv = row_divisor(temperature)
            if cutlass.const_expr(phase == 0):
                best = cute.make_rmem_tensor(cute.make_layout(group), Int32)
                for k in cutlass.range_constexpr(group):
                    for half in cutlass.range_constexpr(2):
                        base = (g * group + k) * CHUNK + half * (THREADS * VEC) + tid * VEC
                        loaded = probability_load4(source, base, self.v)
                        for j in cutlass.range_constexpr(VEC):
                            values[(k * 2 + half) * VEC + j] = loaded[j]
                for k in cutlass.range_constexpr(group):
                    top, winner = Float32(float("-inf")), Int32(self.v)
                    for half in cutlass.range_constexpr(2):
                        base = (g * group + k) * CHUNK + half * (THREADS * VEC) + tid * VEC
                        for j in cutlass.range_constexpr(VEC):
                            z = values[(k * 2 + half) * VEC + j]
                            if base + j < self.v:
                                if (z > top) | ((z == top) & (base + j < winner)):
                                    top, winner = z, base + j
                    partial[k], best[k] = top, winner
                chunk_reduce(partial, tid, scratch, results, True, group)
                for k in cutlass.range_constexpr(group):
                    if partial[k] != results[k]:
                        best[k] = Int32(self.v)
                chunk_min(best, tid, choices, winners, group)
                if tid < group:
                    chunk = g * group + tid
                    if chunk < self.n:
                        recf[chunk * 5], rec[chunk * 5 + 1] = results[tid], winners[tid]
            elif cutlass.const_expr(phase == 1):
                if temperature > Float32(0):
                    maximum = identity_row_max(recf, self.n, tid, results, group)
                    scaled_max = fp(maximum, temperature, "div")
                    steps = loop
                    if cutlass.const_expr(static > 0):
                        steps = static  # compile-time count: the unlooped code
                    for step in cutlass.range(steps):
                        gg = g * steps + step
                        if gg < self.groups:
                            for k in cutlass.range_constexpr(group):
                                for half in cutlass.range_constexpr(2):
                                    base = (
                                        (gg * group + k) * CHUNK
                                        + half * (THREADS * VEC)
                                        + tid * VEC
                                    )
                                    loaded = probability_load4(source, base, self.v)
                                    for j in cutlass.range_constexpr(VEC):
                                        values[(k * 2 + half) * VEC + j] = loaded[j]
                            # Unswitch on T >= HUGE_T (row_exponent) outside the loop.
                            for huge in cutlass.range_constexpr(2):
                                use = temperature < Float32(HUGE_T)
                                if cutlass.const_expr(huge == 1):
                                    use = temperature >= Float32(HUGE_T)
                                if use:
                                    for k in cutlass.range_constexpr(group):
                                        total = Float32(0)
                                        for half in cutlass.range_constexpr(2):
                                            base = (gg * group + k) * CHUNK + half * (THREADS * VEC)
                                            base += tid * VEC
                                            for j in cutlass.range_constexpr(VEC):
                                                if base + j < self.v:
                                                    z = row_exponent(
                                                        values[(k * 2 + half) * VEC + j],
                                                        maximum,
                                                        scaled_max,
                                                        tdiv,
                                                        huge == 1,
                                                    )
                                                    e = cute.exp(z, fastmath=False)
                                                    total = fp(total, e, "add")
                                        partial[k] = total
                            chunk_reduce(partial, tid, scratch, results, False, group)
                            if tid < group:
                                chunk = gg * group + tid
                                if chunk < self.n:
                                    recf[chunk * 5 + 2] = results[tid]
            else:
                if temperature == Float32(0):
                    if tid < 32:
                        top, winner = Float32(float("-inf")), Int32(self.v)
                        for base in cutlass.range(0, self.n, 32, unroll_full=True):
                            c = base + tid
                            if c < self.n:
                                value, token = recf[c * 5], rec[c * 5 + 1]
                                if (value > top) | ((value == top) & (token < winner)):
                                    top, winner = value, token
                        top, winner = warp_argmax(top, winner)
                        if tid == 0:
                            winners[group] = winner
                    cute.arch.sync_threads()
                    winner = winners[group]
                    for k in cutlass.range_constexpr(group):
                        for half in cutlass.range_constexpr(2):
                            base = (g * group + k) * CHUNK + half * (THREADS * VEC) + tid * VEC
                            for j in cutlass.range_constexpr(VEC):
                                written[j] = Float32(base + j == winner)
                            store4(target, base, self.v, written)
                else:
                    maximum = identity_row_max(recf, self.n, tid, results, group)
                    for start in cutlass.range(0, self.n, THREADS, unroll_full=True):
                        if start + tid < self.n:
                            staged[start + tid] = recf[(start + tid) * 5 + 2]
                    cute.arch.sync_threads()
                    if tid == 0:
                        total = Float32(0)
                        for c in cutlass.range(self.n, unroll_full=True):
                            total = fp(total, staged[c], "add")
                        results[group + 1] = total
                    cute.arch.sync_threads()
                    zdiv = row_divisor(results[group + 1])
                    scaled_max = fp(maximum, temperature, "div")
                    for huge in cutlass.range_constexpr(2):
                        use = temperature < Float32(HUGE_T)
                        if cutlass.const_expr(huge == 1):
                            use = temperature >= Float32(HUGE_T)
                        if use:
                            for k in cutlass.range_constexpr(group):
                                for half in cutlass.range_constexpr(2):
                                    base = (g * group + k) * CHUNK + half * (THREADS * VEC)
                                    base += tid * VEC
                                    loaded = probability_load4(source, base, self.v)
                                    for j in cutlass.range_constexpr(VEC):
                                        z = row_exponent(
                                            loaded[j], maximum, scaled_max, tdiv, huge == 1
                                        )
                                        e = cute.exp(z, fastmath=False)
                                        written[j] = div_rn_by(e, zdiv)
                                        if cutlass.const_expr(self.output_maxima):
                                            if base + j < self.v:
                                                partial[k] = max(partial[k], written[j])
                                    store4(target, base, self.v, written)
                    if cutlass.const_expr(self.output_maxima):
                        chunk_reduce(partial, tid, scratch, results, True, group)
                        if tid < group:
                            c = g * group + tid
                            if c < self.n:
                                maxima[row * self.n + c] = results[tid]

    @cute.kernel
    def compact(
        self,
        logits: cute.Pointer,
        temperatures: cute.Pointer,
        top_k: cute.Pointer,
        top_p: cute.Pointer,
        slots: cute.Pointer,
        states: cute.Pointer,
        records: cute.Pointer,
        lists: cute.Pointer,
        length: Int64,
        stride_b: Int64,
        stride_l: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, position, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.flat):
            position = b % Int32(length)
            b = b // Int32(length)
        if cutlass.const_expr(self.unit_length):
            length = 1
        tid, _, _ = cute.arch.thread_idx()
        if slots[b] >= 0:
            temperature, k = temperatures[b], top_k[b]
            if (temperature > Float32(0)) & (k < self.v) & (k <= self.compact_cap):
                row = Int64(b) * length + position
                ids = lists + row * (2 * COMPACT_CAP + 1)
                result = cute.recast_ptr(ids + COMPACT_CAP, dtype=Float32)
                source = self.row_source(logits, b, position, row, stride_b, stride_l)
                smem = SmemAllocator()
                scan = smem.allocate_tensor(Uint64, cute.make_layout(2 * WARPS))
                scratch = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
                probs = smem.allocate_tensor(Float32, cute.make_layout(COMPACT_CAP))
                tokens = smem.allocate_tensor(Int32, cute.make_layout(COMPACT_CAP))
                sorted_keys = smem.allocate_tensor(
                    Uint64, cute.make_layout(COMPACT_CAP), byte_alignment=16
                )
                kept_slot = smem.allocate_tensor(Int32, cute.make_layout(COMPACT_CAP))
                _, _, _, maximum, total, _ = probability_state(
                    states + row * ROW_STATE_WORDS,
                    records + row * self.n * 5,
                    self.n,
                    Int32(0),
                    k,
                    temperature,
                    self.v,
                    tid,
                    scratch,
                )
                tdiv = row_divisor(temperature)
                scaled_max = fp(maximum, temperature, "div")
                tau, kept_ties, _, equal, _ = read_level(states + row * ROW_STATE_WORDS, 2)
                if equal > kept_ties:
                    # The kept_ties smallest-ID tokens at the k-th key, in ascending ID
                    # order, from P4's per-chunk tie counts; chunks without ties are
                    # skipped and the scan stops once the set is complete.
                    rec = records + row * self.n * 5
                    collected = Uint64(0)
                    for c in cutlass.range(self.n):
                        if (collected < kept_ties) & (rec[c * 5 + 3] != 0):
                            for start in cutlass.range_constexpr(0, CHUNK, THREADS * VEC):
                                base = c * CHUNK + start + tid * VEC
                                values = probability_load4(source, base, self.v)
                                local = Uint64(0)
                                for j in cutlass.range_constexpr(VEC):
                                    if (base + j < self.v) & (ordered_key(values[j]) == tau):
                                        local += Uint64(1)
                                before, tile_ties = integer_scan(local, tid, scan)
                                for j in cutlass.range_constexpr(VEC):
                                    if (base + j < self.v) & (ordered_key(values[j]) == tau):
                                        if collected + before < kept_ties:
                                            slot = Uint64(k) - kept_ties + collected + before
                                            ids[Int32(slot)] = Int32(base + j)
                                        before += Uint64(1)
                                collected += tile_ties
                    cute.arch.sync_threads()
                for start in cutlass.range_constexpr(0, COMPACT_CAP, THREADS):
                    index = start + tid
                    if index < k:
                        token = ids[index]
                        z = stable_exponent(Float32(source[token]), maximum, scaled_max, tdiv)
                        e = cute.exp(z, fastmath=False)
                        probs[index], tokens[index] = fp(e, total, "div"), token
                cute.arch.sync_threads()
                p = Float32(1)
                if cutlass.const_expr(self.use_p):
                    p = top_p[b]
                keep = cute.make_rmem_tensor(cute.make_layout(COMPACT_CAP // THREADS), Int32)
                for j in cutlass.range_constexpr(COMPACT_CAP // THREADS):
                    keep[j] = Int32(j * THREADS + tid < k)
                if cutlass.const_expr(self.use_p):
                    if p < Float32(1):
                        # A candidate's preceding mass sums the candidates before it in
                        # (p desc, ID asc) order: sort ((~bits(p)) << 31 | id << 10 | slot)
                        # ascending (bitonic, 1024 keys), take the exact integer exclusive
                        # prefix of the masses, and keep ranks whose prefix is below
                        # ceil(top_p 2^62). Independent of the atomic append order.
                        order = cute.make_tensor(
                            cute.recast_ptr(sorted_keys.iterator, dtype=Uint64),
                            cute.make_layout(COMPACT_CAP),
                        )
                        for j in cutlass.range_constexpr(COMPACT_CAP // THREADS):
                            index = j * THREADS + tid
                            key = Uint64(0xFFFFFFFFFFFFFFFF)
                            if index < k:
                                bits = Uint64(~float_bits(probs[index]))
                                key = (bits << 31) | (Uint64(tokens[index]) << 10) | Uint64(index)
                            order[index] = key
                        cute.arch.sync_threads()
                        for level in cutlass.range_constexpr(1, COMPACT_LOG + 1):
                            for step in cutlass.range_constexpr(level):
                                # One compare-exchange per thread: pair (i, i + stride).
                                stride: cutlass.Constexpr = 1 << (level - 1 - step)
                                i = (tid // stride) * (stride * 2) + tid % stride
                                first, second = order[i], order[i + stride]
                                if (first > second) == ((i & (1 << level)) == 0):
                                    order[i], order[i + stride] = second, first
                                cute.arch.sync_threads()
                        # Each thread owns ranks 2t and 2t+1.
                        masses_ = cute.make_rmem_tensor(cute.make_layout(2), Uint64)
                        for r in cutlass.range_constexpr(2):
                            rank = tid * 2 + r
                            masses_[r] = Uint64(0)
                            key = order[rank]
                            if key != Uint64(0xFFFFFFFFFFFFFFFF):
                                masses_[r] = mass62(probs[Int32(key & Uint64(1023))])
                        before, _ = integer_scan(masses_[0] + masses_[1], tid, scan)
                        threshold = mass62(p, True)
                        cute.arch.sync_threads()
                        for r in cutlass.range_constexpr(2):
                            rank = tid * 2 + r
                            key = order[rank]
                            if key != Uint64(0xFFFFFFFFFFFFFFFF):
                                kept_slot[Int32(key & Uint64(1023))] = Int32(before < threshold)
                            before += masses_[r]
                        cute.arch.sync_threads()
                        for j in cutlass.range_constexpr(COMPACT_CAP // THREADS):
                            index = j * THREADS + tid
                            if index < k:
                                keep[j] = kept_slot[index]
                local = Uint64(0)
                for j in cutlass.range_constexpr(COMPACT_CAP // THREADS):
                    index = j * THREADS + tid
                    if index < k:
                        if keep[j] != 0:
                            local += mass62(probs[index])
                _, fixed_sum = integer_scan(local, tid, scan)
                scale = fp(Float32(fixed_sum), Float32(2.0**-62), "mul")
                for j in cutlass.range_constexpr(COMPACT_CAP // THREADS):
                    index = j * THREADS + tid
                    if index < k:
                        value = Float32(0)
                        if keep[j] != 0:
                            value = probs[index]
                            if p < Float32(1):
                                value = fp(value, scale, "div")
                        result[index] = value

    @cute.kernel
    def zero(
        self,
        counts: cute.Pointer,
        masses: cute.Pointer,
        lists: cute.Pointer,
        states: cute.Pointer,
        length: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, position, tile = cute.arch.block_idx()
        if cutlass.const_expr(self.flat):
            position = b % Int32(length)
            b = b // Int32(length)
        if cutlass.const_expr(self.unit_length):
            length = 1
        tid, _, _ = cute.arch.thread_idx()
        row = Int64(b) * length + position
        index = row * 5120 + tile * THREADS + tid
        if (tile == 0) & (tid < 6):
            states[row * ROW_STATE_WORDS + tid * LEVEL_STATE_WORDS] = Uint64(0)
        if cutlass.const_expr(self.use_k):
            counts[index] = Int32(0)
            row = Int64(b) * length + position
            if tile < COMPACT_CAP // THREADS:
                lists[row * (2 * COMPACT_CAP + 1) + tile * THREADS + tid] = Int32(self.v)
            if (tile == 0) & (tid == 0):
                lists[row * (2 * COMPACT_CAP + 1) + 2 * COMPACT_CAP] = Int32(0)
        if cutlass.const_expr(self.use_p):
            masses[index] = Uint64(0)

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        temperatures: cute.Pointer,
        top_k: cute.Pointer,
        top_p: cute.Pointer,
        slots: cute.Pointer,
        out: cute.Pointer,
        counts: cute.Pointer,
        masses: cute.Pointer,
        records: cute.Pointer,
        lists: cute.Pointer,
        states: cute.Pointer,
        maxima: cute.Pointer,
        phase: cutlass.Constexpr,
        length: Int64,
        stride_b: Int64,
        stride_l: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, position, chunk = cute.arch.block_idx()
        if cutlass.const_expr(self.flat):
            position = b % Int32(length)
            b = b // Int32(length)
        if cutlass.const_expr(self.unit_length):
            length = 1
        tid, _, _ = cute.arch.thread_idx()
        row = Int64(b) * length + position
        source, target = (
            self.row_source(logits, b, position, row, stride_b, stride_l),
            out + row * self.v,
        )
        kh, ph = counts + row * 5120, masses + row * 5120
        ks = states + row * ROW_STATE_WORDS
        ps = ks + 3 * LEVEL_STATE_WORDS
        rec = records + row * self.n * 5
        recf = cute.recast_ptr(rec, dtype=Float32)
        ids = lists + row * (2 * COMPACT_CAP + 1)
        compact_values = cute.recast_ptr(ids + COMPACT_CAP, dtype=Float32)
        output_max = Float32(0)
        smem = SmemAllocator()
        scan = smem.allocate_tensor(Uint64, cute.make_layout(2 * WARPS))
        selected = smem.allocate_tensor(Uint64, cute.make_layout(3))
        scratch = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        choices = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        lasts = smem.allocate_tensor(Int32, cute.make_layout(WARPS))
        mass_phase: cutlass.Constexpr = phase in (5, 6, 7)
        hist = smem.allocate_tensor(
            Uint32 if mass_phase else Int32,
            cute.make_layout(
                MASS_SLICES * MASS_BINS if mass_phase else (2048 if phase in (1, 2, 3) else 1)
            ),
            byte_alignment=16,
        )
        if cutlass.const_expr(mass_phase):
            # resolve_level's last CTA stages the completed global uint64 bins here.
            hist64 = cute.make_tensor(
                cute.recast_ptr(hist.iterator, dtype=Uint64), cute.make_layout(2048)
            )
        if cutlass.const_expr(phase == 5):
            staged = smem.allocate_tensor(Int32, cute.make_layout(self.n * 5))
        if slots[b] == -1:
            if cutlass.const_expr(phase == 5 and self.materialize):
                for start in range(0, CHUNK, THREADS):
                    token = chunk * CHUNK + start + tid
                    if token < self.v:
                        target[token] = Float32(0)
        else:
            temperature = temperatures[b]
            tdiv = row_divisor(temperature)
            compact = False
            if cutlass.const_expr(self.use_k):
                compact = (
                    (temperature > Float32(0))
                    & (top_k[b] < self.v)
                    & (top_k[b] <= self.compact_cap)
                )
            if cutlass.const_expr(phase >= 5):
                if compact:
                    # Without materialization the draw reads compact rows' lists itself.
                    if cutlass.const_expr(phase == 5 and self.materialize):
                        for start in range(0, CHUNK, THREADS):
                            token = chunk * CHUNK + start + tid
                            if token < self.v:
                                target[token] = Float32(0)
                        cute.arch.sync_threads()
                        for start in range(0, COMPACT_CAP, THREADS):
                            index = start + tid
                            if index < top_k[b]:
                                token = ids[index]
                                if (token >= chunk * CHUNK) & (
                                    token < min((chunk + 1) * CHUNK, self.v)
                                ):
                                    value = compact_values[index]
                                    target[token] = value
                                    if cutlass.const_expr(self.output_maxima):
                                        output_max = max(output_max, value)
            if cutlass.const_expr(phase == 1):
                k = Int32(self.v)
                if cutlass.const_expr(self.use_k):
                    if temperature > Float32(0):
                        k = top_k[b]
                    if k < self.v:
                        histogram_clear(hist, tid, 2048)
                # Max, argmax and top-k keys use raw FP32 logits; T>0 only scales
                # (l - max) later, so l/T never overflows before stabilization.
                best, winner = Float32(float("-inf")), Int32(self.v)
                for start in range(0, CHUNK, THREADS * VEC):
                    base = chunk * CHUNK + start + tid * VEC
                    values = probability_load4(source, base, self.v)
                    for j in cutlass.range_constexpr(VEC):
                        if base + j < self.v:
                            z = values[j]
                            if (z > best) | ((z == best) & (base + j < winner)):
                                best, winner = z, base + j
                    if cutlass.const_expr(self.use_k):
                        if k < self.v:
                            for j in cutlass.range_constexpr(VEC):
                                digit = Int32(ordered_key(values[j]) >> 21)
                                histogram_count(hist, 2047 - digit, base + j < self.v)
                maximum = probability_reduce(best, tid, scratch, True)
                if best != maximum:
                    winner = Int32(self.v)
                winner, _ = reduce_choices(winner, Int32(-1), tid, choices, lasts)
                if tid == 0:
                    recf[chunk * 5], rec[chunk * 5 + 1] = maximum, winner
                if cutlass.const_expr(self.use_k):
                    if k < self.v:
                        histogram_flush(hist, kh, tid, 2048)
                        resolve_level(kh, ks, 0, Uint64(k), self.n, tid, hist, scan, selected)
            elif cutlass.const_expr(phase in (2, 3)):
                if temperature > Float32(0):
                    k = top_k[b]
                    if k < self.v:
                        prefix, _, _, _, _ = read_level(ks, phase - 2)
                        shift = 10 if phase == 2 else 0
                        bins = 2048 if phase == 2 else 1024
                        mask = 0xFFE00000 if phase == 2 else 0xFFFFFC00
                        histogram_clear(hist, tid, bins)
                        for start in range(0, CHUNK, THREADS * VEC):
                            base = chunk * CHUNK + start + tid * VEC
                            values = probability_load4(source, base, self.v)
                            for j in cutlass.range_constexpr(VEC):
                                key = ordered_key(values[j])
                                digit = Int32((key >> shift) & Uint32(bins - 1))
                                histogram_count(
                                    hist,
                                    bins - 1 - digit,
                                    (base + j < self.v) & ((key & Uint32(mask)) == prefix),
                                )
                        histogram_flush(hist, kh + (phase - 1) * 2048, tid, bins)
                        resolve_level(
                            kh, ks, phase - 1, Uint64(k), self.n, tid, hist, scan, selected
                        )
            elif cutlass.const_expr(phase in (4, 5)):
                if temperature == Float32(0):
                    if cutlass.const_expr(phase == 5 and self.materialize):
                        if tid < 32:
                            best, winner = Float32(float("-inf")), Int32(self.v)
                            for base in cutlass.range(0, self.n, 32, unroll_full=True):
                                c = base + tid
                                if c < self.n:
                                    value, token = recf[c * 5], rec[c * 5 + 1]
                                    if (value > best) | ((value == best) & (token < winner)):
                                        best, winner = value, token
                            best, winner = warp_argmax(best, winner)
                            if tid == 0:
                                choices[0] = winner
                        cute.arch.sync_threads()
                        winner = choices[0]
                        for start in range(0, CHUNK, THREADS):
                            token = chunk * CHUNK + start + tid
                            if token < self.v:
                                target[token] = Float32(token == winner)
                elif not compact or phase == 4:
                    k, p = Int32(self.v), Float32(1)
                    if cutlass.const_expr(self.use_k):
                        k = top_k[b]
                    if cutlass.const_expr(self.use_p):
                        p = top_p[b]
                    tau, kept_ties, equal = Uint32(0), Uint64(0), Uint64(0)
                    if cutlass.const_expr(self.use_k):
                        if k < self.v:
                            tau, kept_ties, _, equal, _ = read_level(ks, 2)
                    if cutlass.const_expr(phase == 5):
                        stage_probability_records(rec, staged, self.n, tid)
                        maximum, record_before = probability_record_reduce(
                            staged.iterator, self.n, chunk, tid, scratch, True
                        )
                        staged_f = cute.recast_ptr(staged.iterator, dtype=Float32)
                        if tid == 0:
                            total = Float32(0)
                            for c in cutlass.range(self.n, unroll_full=True):
                                total = fp(total, staged_f[c * 5 + 2], "add")
                            if k < self.v:
                                z = stable_exponent(
                                    key_float(tau),
                                    maximum,
                                    fp(maximum, temperature, "div"),
                                    tdiv,
                                )
                                tie_e = cute.exp(z, fastmath=False)
                                total = fp(total, fp(Float32(kept_ties), tie_e, "mul"), "add")
                            scratch[1], choices[0] = total, record_before
                    else:
                        # P4 only reads completed max records; other fields may
                        # be concurrently written by other P4 CTAs.
                        maximum, _ = probability_record_reduce(
                            rec, self.n, chunk, tid, scratch, False
                        )
                    cute.arch.sync_threads()
                    maximum = scratch[0]
                    scaled_max = fp(maximum, temperature, "div")
                    total, tie_count = Float32(0), Uint64(0)
                    if cutlass.const_expr(phase == 5):
                        total, tie_count = scratch[1], Uint64(choices[0])
                        zdiv = row_divisor(total)
                        if cutlass.const_expr(self.use_p):
                            if p < Float32(1):
                                histogram_clear(hist, tid, MASS_SLICES * MASS_BINS)
                    # Finish scratch readers before probability_reduce reuses it.
                    cute.arch.sync_threads()
                    for start in range(0, CHUNK, THREADS * VEC):
                        base = chunk * CHUNK + start + tid * VEC
                        values = probability_load4(source, base, self.v)
                        keys = cute.make_rmem_tensor(cute.make_layout(VEC), Uint32)
                        local_ties = Uint64(0)
                        for j in cutlass.range_constexpr(VEC):
                            keys[j] = ordered_key(values[j])
                            if (base + j < self.v) & (k < self.v) & (keys[j] == tau):
                                local_ties += Uint64(1)
                        before, tile_ties = Uint64(0), Uint64(0)
                        if cutlass.const_expr(phase == 5):
                            if (k < self.v) & (equal > kept_ties):
                                before, tile_ties = integer_scan(local_ties, tid, scan)
                        written = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                        written.fill(Float32(0))
                        for j in cutlass.range_constexpr(VEC):
                            if base + j < self.v:
                                is_tie = (k < self.v) & (keys[j] == tau)
                                keep = (k == self.v) | (keys[j] > tau)
                                if cutlass.const_expr(phase == 5):
                                    keep = keep | (
                                        is_tie
                                        & ((equal == kept_ties) | (tie_count + before < kept_ties))
                                    )
                                    if is_tie:
                                        before += Uint64(1)
                                if cutlass.const_expr(phase == 4):
                                    if compact:
                                        # A split tie set's smallest IDs go to the last
                                        # kept_ties slots in the compact kernel (in order).
                                        if keep | (is_tie & (equal == kept_ties)):
                                            index = cute.arch.atomic_add(
                                                ids + 2 * COMPACT_CAP, Int32(1)
                                            )
                                            ids[index] = Int32(base + j)
                                    if keep:
                                        z = stable_exponent(values[j], maximum, scaled_max, tdiv)
                                        e = cute.exp(z, fastmath=False)
                                        total = fp(total, e, "add")
                                else:
                                    prob = Float32(0)
                                    if keep:
                                        z = stable_exponent(values[j], maximum, scaled_max, tdiv)
                                        e = cute.exp(z, fastmath=False)
                                        prob = div_rn_by(e, zdiv)
                                    written[j] = prob
                                    if cutlass.const_expr(self.output_maxima):
                                        output_max = max(output_max, prob)
                        if cutlass.const_expr(phase == 5 and self.use_p):
                            if p < Float32(1):
                                # Out-of-row lanes hold 0 (no mass): they only join the vote.
                                for j in cutlass.range_constexpr(VEC):
                                    digit = Int32(float_bits(written[j]) >> 21)
                                    histogram_mass(
                                        hist, 2047 - digit, mass62(written[j]), base + j < self.v
                                    )
                        if cutlass.const_expr(phase == 5 and self.materialize):
                            store4(target, base, self.v, written)
                        if cutlass.const_expr(phase == 4):
                            tie_count += local_ties
                        else:
                            tie_count += tile_ties
                    if cutlass.const_expr(phase == 4):
                        total = probability_reduce(total, tid, scratch, False)
                        _, ties = integer_scan(tie_count, tid, scan)
                        if tid == 0:
                            recf[chunk * 5 + 2], rec[chunk * 5 + 3] = total, Int32(ties)
                    elif cutlass.const_expr(self.use_p):
                        if p < Float32(1):
                            histogram_flush_mass(hist, ph, tid, 2048)
                            resolve_level(
                                ph, ps, 0, mass62(p, True), self.n, tid, hist64, scan, selected
                            )
            else:
                if (temperature > Float32(0)) & (not compact):
                    p = top_p[b]
                    if p < Float32(1):
                        threshold = mass62(p, True)
                        # Below threshold, lower levels never participate or publish.
                        tau, need, above, equal, full = read_level(ps, 0)
                        if cutlass.const_expr(phase > 6):
                            if full >= threshold:
                                tau, need, above, equal, full = read_level(ps, min(phase - 6, 2))
                        if cutlass.const_expr(phase in (6, 7)):
                            if full >= threshold:
                                shift = 10 if phase == 6 else 0
                                bins = 2048 if phase == 6 else 1024
                                mask = 0xFFE00000 if phase == 6 else 0xFFFFFC00
                                if cutlass.const_expr(not self.materialize):
                                    large = large_state(
                                        ks, rec, self.n, chunk, top_k, b, self.use_k, temperature,
                                        self.v, tid, scratch,
                                    )  # fmt: skip
                                    tie_k = large[6]
                                histogram_clear(hist, tid, MASS_SLICES * MASS_BINS)
                                for start in range(0, CHUNK, THREADS * VEC):
                                    base = chunk * CHUNK + start + tid * VEC
                                    if cutlass.const_expr(self.materialize):
                                        values = load4(target, base, self.v)
                                    else:
                                        values, ties = p5_tile(
                                            source, base, self.v, large, tie_k, tid, scan
                                        )
                                        tie_k += ties
                                    for j in cutlass.range_constexpr(VEC):
                                        key = float_bits(values[j])
                                        digit = Int32((key >> shift) & Uint32(bins - 1))
                                        histogram_mass(
                                            hist,
                                            bins - 1 - digit,
                                            mass62(values[j]),
                                            (base + j < self.v) & ((key & Uint32(mask)) == tau),
                                        )
                                histogram_flush_mass(hist, ph + (phase - 5) * 2048, tid, bins)
                                resolve_level(
                                    ph,
                                    ps,
                                    phase - 5,
                                    threshold,
                                    self.n,
                                    tid,
                                    hist64,
                                    scan,
                                    selected,
                                )
                        else:
                            fixed_sum, kept_ties, split = full, Uint64(0), Int32(0)
                            if full >= threshold:
                                tau_value = Float32(llvm.bitcast(T.f32(), tau.ir_value()))
                                unit = mass62(tau_value)
                                kept_ties = (need + unit - Uint64(1)) // unit
                                fixed_sum = above + kept_ties * unit
                                split = Int32(equal > kept_ties * unit)
                            if cutlass.const_expr(phase == 8):
                                if split != 0:
                                    local = Uint64(0)
                                    if cutlass.const_expr(not self.materialize):
                                        large = large_state(
                                            ks, rec, self.n, chunk, top_k, b, self.use_k,
                                            temperature, self.v, tid, scratch,
                                        )  # fmt: skip
                                        tie_k = large[6]
                                    for start in range(0, CHUNK, THREADS * VEC):
                                        base = chunk * CHUNK + start + tid * VEC
                                        if cutlass.const_expr(self.materialize):
                                            values = load4(target, base, self.v)
                                        else:
                                            values, ties = p5_tile(
                                                source, base, self.v, large, tie_k, tid, scan
                                            )
                                            tie_k += ties
                                        for j in cutlass.range_constexpr(VEC):
                                            if (base + j < self.v) & (float_bits(values[j]) == tau):
                                                local += Uint64(1)
                                    _, tie_total = integer_scan(local, tid, scan)
                                    if tid == 0:
                                        rec[chunk * 5 + 4] = Int32(tie_total)
                            else:
                                scale = fp(Float32(fixed_sum), Float32(2.0**-62), "mul")
                                tie_count = Uint64(0)
                                if split != 0:
                                    if tid == 0:
                                        before = Int32(0)
                                        for c in range(chunk):
                                            before += rec[c * 5 + 4]
                                        choices[0] = before
                                    cute.arch.sync_threads()
                                    tie_count = Uint64(choices[0])
                                for start in range(0, CHUNK, THREADS * VEC):
                                    base = chunk * CHUNK + start + tid * VEC
                                    values = load4(target, base, self.v)
                                    local, before, tile_ties = Uint64(0), Uint64(0), Uint64(0)
                                    if split != 0:
                                        for j in cutlass.range_constexpr(VEC):
                                            if (base + j < self.v) & (float_bits(values[j]) == tau):
                                                local += Uint64(1)
                                        before, tile_ties = integer_scan(local, tid, scan)
                                    written = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                                    written.fill(Float32(0))
                                    for j in cutlass.range_constexpr(VEC):
                                        if base + j < self.v:
                                            key, value = float_bits(values[j]), Float32(0)
                                            keep = (
                                                (full < threshold)
                                                | (key > tau)
                                                | (
                                                    (key == tau)
                                                    & (
                                                        (split == 0)
                                                        | (tie_count + before < kept_ties)
                                                    )
                                                )
                                            )
                                            if key == tau:
                                                before += Uint64(1)
                                            if keep:
                                                value = fp(values[j], scale, "div")
                                            written[j] = value
                                            if cutlass.const_expr(self.output_maxima):
                                                output_max = max(output_max, value)
                                    store4(target, base, self.v, written)
                                    tie_count += tile_ties

            if cutlass.const_expr(self.output_maxima and phase in (5, 9)):
                write_max = temperature > Float32(0)
                if cutlass.const_expr(phase == 9):
                    write_max = write_max & (not compact) & (top_p[b] < Float32(1))
                if write_max:
                    # P1 and identity phase 0 leave raw max/argmax in record words
                    # 0/1; P4 uses 2/3 and P8 uses 4. All preserve 0/1, including
                    # compact and greedy. But top-p can discard the raw argmax
                    # when FP32 probabilities tie with a smaller ID. Thus reduce
                    # the exact values already being written, into separate scratch
                    # (overwriting the original records would race P5 readers).
                    # P5 covers compact/identity-p rows; P9 replaces large top-p
                    # maxima. Greedy/padding never read these records in the draw.
                    cute.arch.sync_threads()
                    output_max = probability_reduce(output_max, tid, scratch, True)
                    if tid == 0:
                        maxima[row * self.n + chunk] = output_max


class MinPRenormalize:
    """Two in-place chunk passes with probability_sum's exact FP32 reduction."""

    def __init__(self, length, vocab, batch, flat=False):
        self.v, self.flat, self.unit_length = vocab, flat, False
        self.n = (vocab + CHUNK - 1) // CHUNK

    def config_key(self):
        return (self.v, self.flat)

    @cute.jit
    def __call__(
        self,
        probs: cute.Tensor,
        temperatures: cute.Tensor,
        min_p: cute.Tensor,
        slots: cute.Tensor,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch = Int32(slots.shape[0])
        length = Int64(probs.shape[1])
        rows = Int64(batch) * length
        compact_offset = rows * 5120 * 12 + ((rows * self.n * 20 + 255) // 256) * 256
        state_offset = compact_offset + ((rows * (2 * COMPACT_CAP + 1) * 4 + 255) // 256) * 256
        maxima_offset = state_offset + ((rows * ROW_STATE_WORDS * 8 + 255) // 256) * 256
        partials_offset = maxima_offset + ((rows * self.n * 4 + 255) // 256) * 256
        maxima = cute.recast_ptr(workspace.iterator + maxima_offset, dtype=Float32)
        partials = cute.recast_ptr(workspace.iterator + partials_offset, dtype=Float32)
        for finalize in cutlass.range_constexpr(2):
            self.kernel(
                probs.iterator,
                temperatures.iterator,
                min_p.iterator,
                slots.iterator,
                maxima,
                partials,
                finalize == 1,
                length,
            ).launch(
                grid=(rows, 1, self.n)
                if cutlass.const_expr(self.flat)
                else (batch, length, self.n),
                block=(THREADS, 1, 1),
                use_pdl=PDL,
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        probs: cute.Pointer,
        temperatures: cute.Pointer,
        min_p: cute.Pointer,
        slots: cute.Pointer,
        maxima: cute.Pointer,
        partials: cute.Pointer,
        finalize: cutlass.Constexpr,
        length: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        b, position, chunk = cute.arch.block_idx()
        if cutlass.const_expr(self.flat):
            position = b % Int32(length)
            b = b // Int32(length)
        if cutlass.const_expr(self.unit_length):
            length = 1
        tid, _, _ = cute.arch.thread_idx()
        smem = SmemAllocator()
        scratch = smem.allocate_tensor(Float32, cute.make_layout(WARPS))
        state = smem.allocate_tensor(Float32, cute.make_layout(1))
        # Uniform branches skip padding/greedy before reading min_p or maxima.
        if slots[b] != -1:
            if temperatures[b] > Float32(0):
                if min_p[b] > Float32(0):
                    row = Int64(b) * length + position
                    target = probs + row * self.v
                    pmax = Float32(0)
                    if tid < self.n:
                        pmax = maxima[row * self.n + tid]
                    pmax = probability_reduce(pmax, tid, scratch, True)
                    pivot = fp(min_p[b], pmax, "mul")
                    total = Float32(0)
                    if cutlass.const_expr(finalize):
                        if tid == 0:
                            for c in cutlass.range(self.n):
                                total = fp(total, partials[row * self.n + c], "add")
                            state[0] = total
                        cute.arch.sync_threads()
                        total = state[0]
                    for start in cutlass.range(0, CHUNK, THREADS * VEC):
                        base = chunk * CHUNK + start + tid * VEC
                        values = load4(target, base, self.v)
                        for j in cutlass.range_constexpr(VEC):
                            value = Float32(0)
                            if values[j] >= pivot:
                                value = values[j]
                                if cutlass.const_expr(finalize):
                                    value = fp(value, total, "div")
                            if cutlass.const_expr(finalize):
                                values[j] = value
                            else:
                                total = fp(total, value, "add")
                        if cutlass.const_expr(finalize):
                            store4(target, base, self.v, values)
                    if cutlass.const_expr(not finalize):
                        total = probability_reduce(total, tid, scratch, False)
                        if tid == 0:
                            partials[row * self.n + chunk] = total


__all__ = ["Chain", "Candidates", "Rng", "Cursors", "Probabilities", "MinPRenormalize"]
