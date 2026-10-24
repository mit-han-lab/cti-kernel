# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""R5 kernels: no allocation, atomics or inter-CTA dependencies."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64

from ..op import P_NAMES, R_NAMES

THREADS = 128
MAX_BATCH = 4  # the unrolled small-batch path; larger B takes the generic path


class Prepare:
    """part: "full" (prepare_block) or "round" (prepare_round: no KV addressing;
    fixed_query_addresses is the Triton kernel). generic: the any-B kernel;
    otherwise the unrolled B<=4 kernel. Both are compiled per variant; the host
    picks by B."""

    def __init__(self, q, rope, k2, cursors, pdl, wait=True, part="full", generic=False):
        self.q, self.rope, self.k2, self.cursors = q, rope, k2, cursors
        self.pdl, self.wait, self.part, self.generic = pdl, wait, part, generic
        self.names = {"full": P_NAMES, "round": R_NAMES}[part]

    @cute.jit
    def __call__(
        self,
        t: tuple,
        m: Int64,
        guard: Int64,
        mask: Int32,
        n: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        named = dict(zip(self.names, t, strict=True))
        tok_stride, ci_stride = named["token_table"].stride[0], Int64(0)
        if cutlass.const_expr(self.part != "round"):
            ci_stride = named["cache_index_table"].stride[0]
        self.kernel(
            tuple(x.iterator for x in t),
            t[0].shape[0],
            m,
            guard,
            mask,
            tok_stride,
            ci_stride,
            n,
        ).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(
        self,
        t: tuple,
        batch: Int64,
        m: Int64,
        guard: Int64,
        mask: Int32,
        tok_stride: Int64,
        ci_stride: Int64,
        n: Int64,
    ):
        if cutlass.const_expr(self.pdl and self.wait):
            cute.arch.griddepcontrol_wait()
        address = self.part != "round"  # cache write indices and CSR
        cta, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        worker = Int64(cta * THREADS + tid)
        grid, _, _ = cute.arch.grid_dim()
        step = Int64(grid * THREADS)
        scalars = (m, guard, mask, tok_stride, ci_stride, n, worker, step, tid)
        if cutlass.const_expr(address):
            ptr = t[self.names.index("kv_indptr")]
            if (cta == 0) & (tid == 0):
                ptr[0] = 0
        if cutlass.const_expr(not self.generic):
            self.small(t, batch, cta, scalars)
        else:
            scan = ()
            if cutlass.const_expr(address):
                smem = cutlass.utils.SmemAllocator()
                scan = (
                    smem.allocate_tensor(Int64, cute.make_layout(THREADS), 16),
                    smem.allocate_tensor(Int64, cute.make_layout(THREADS), 16),
                    smem.allocate_tensor(Int64, cute.make_layout(THREADS), 16),
                    smem.allocate_tensor(Int32, cute.make_layout(THREADS), 16),
                    smem.allocate_tensor(Int32, cute.make_layout(THREADS // 32), 16),
                )
            self.any_batch(t, batch, cta, Int64(grid), scan, scalars)
        # No early dependent release: writes from every CTA must be complete.

    @cute.jit
    def small(self, t: tuple, batch: Int64, cta: Int32, scalars: tuple):
        """B<=4 (the host's choice): each CTA independently recomputes the prefixes from immutable
        inputs. The row loads are issued for all four slots up front (clamped to
        row B-1, always in bounds), so their latency is paid once rather than B
        times. Row b's small outputs belong to CTA b (the grid has >= 4 CTAs)."""
        v = dict(zip(self.names, t, strict=True))
        m = scalars[0]
        rows = v["request_indices"]
        base = Int64(0)
        rs, fs, ds = [], [], []
        for k in cutlass.range_constexpr(MAX_BATCH):
            rs.append(Int64(rows[min(Int64(k), batch - 1)]))
        for k in cutlass.range_constexpr(MAX_BATCH):
            fs.append(v["frontier_table"][rs[k]])
            ds.append(v["done_table"][rs[k]])
        for k in cutlass.range_constexpr(MAX_BATCH):
            if k < batch:
                dummy = ds[k] | (fs[k] < 0) | (fs[k] >= m)
                length = Int64(self.q)
                if not dummy:
                    length = fs[k] + self.q
                self.row(t, Int64(k), rs[k], fs[k], ds[k], dummy, length, base, cta == k, scalars)
                base += length

    @cute.jit
    def any_batch(
        self, t: tuple, batch: Int64, cta: Int32, grid: Int64, scan: tuple, scalars: tuple
    ):
        """Any B (the host uses it for B>4). CTA b % grid writes row b's small
        outputs. Without addressing there is no prefix. With addressing, every
        CTA scans all rows in chunks of THREADS (a block prefix over immutable
        inputs; no inter-CTA dependency); all CTAs then copy the chunk's CSR
        entries grid-stride. Lengths and offsets fit int32 (host bound
        B*(M+Q) <= 2^31-1; a non-dummy frontier is < M)."""
        v = dict(zip(self.names, t, strict=True))
        m, guard, _, _, ci_stride, n, worker, step, tid = scalars
        rows = v["request_indices"]
        if cutlass.const_expr(self.part == "round"):
            b = Int64(cta)
            while b < batch:
                r = Int64(rows[b])
                f, d = v["frontier_table"][r], v["done_table"][r]
                dummy = d | (f < 0) | (f >= m)
                self.row(t, b, r, f, d, dummy, Int64(0), Int64(0), True, scalars, False)
                b += grid
        else:
            cache, indices = v["cache_index_table"], v["kv_indices"]
            s_base, s_row, s_front, s_done, s_warp = scan
            warp, lane = tid // 32, tid % 32
            running = Int64(0)
            c = Int64(0)
            while c < batch:
                b = c + tid
                r, f, d = Int64(0), Int64(0), cutlass.Boolean(True)
                length = Int32(0)
                if b < batch:
                    r = Int64(rows[b])
                    f, d = v["frontier_table"][r], v["done_table"][r] != 0
                    length = Int32(self.q)
                    if not (d | (f < 0) | (f >= m)):
                        length = Int32(f + self.q)
                x = length
                for k in cutlass.range_constexpr(5):
                    y = cute.arch.shuffle_sync_up(x, offset=1 << k)
                    if lane >= (1 << k):
                        x = x + y
                if lane == 31:
                    s_warp[warp] = x
                cute.arch.sync_threads()
                before, total = Int32(0), Int32(0)
                for w in cutlass.range_constexpr(THREADS // 32):
                    value = s_warp[w]
                    if w < warp:
                        before = before + value
                    total = total + value
                s_base[tid] = running + Int64(before + x - length)
                s_row[tid], s_front[tid], s_done[tid] = r, f, Int32(d)
                cute.arch.sync_threads()
                count = min(Int64(THREADS), batch - c)
                # Small outputs: CTA b % grid owns row b; it walks only its rows.
                i = (Int64(cta) - c % grid + grid) % grid
                while i < count:
                    rb, fb, db = s_row[i], s_front[i], s_done[i] != 0
                    dummy = db | (fb < 0) | (fb >= m)
                    lb = Int64(self.q)
                    if not dummy:
                        lb = fb + self.q
                    self.row(t, c + i, rb, fb, db, dummy, lb, s_base[i], True, scalars, False)
                    i += grid
                # CSR: grid-stride over the chunk's flattened valid prefix; a
                # binary search over the chunk's offsets finds each entry's row.
                end = min(running + Int64(total), n)
                position = running + worker
                while position < end:
                    low, high = Int32(0), Int32(count)
                    while high - low > 1:
                        middle = (low + high) // 2
                        if s_base[middle] <= position:
                            low = middle
                        else:
                            high = middle
                    rb, fb, db = s_row[low], s_front[low], s_done[low] != 0
                    j = position - s_base[low]
                    g = guard + (c + Int64(low)) * self.q
                    index = Int32(g + j)
                    if not (db | (fb < 0) | (fb >= m)):
                        if j < m:
                            index = cache[rb * ci_stride + j]
                        else:
                            index = Int32(g + min(j - m, Int64(self.q - 1)))
                    indices[position] = index
                    position += step
                cute.arch.sync_threads()
                running += Int64(total)
                c += THREADS

    @cute.jit
    def row(
        self,
        t: tuple,
        b: Int64,
        r: Int64,
        f: Int64,
        d: cutlass.Boolean,
        dummy: cutlass.Boolean,
        length: Int64,
        base: Int64,
        owner: cutlass.Boolean,
        scalars: tuple,
        copy: cutlass.Constexpr = True,
    ):
        """Row b: its small outputs (owner CTA only) and, with addressing and
        copy, its share of the CSR segment [base, base + length)."""
        m, guard, mask, tok_stride, ci_stride, n, worker, step, tid = scalars
        v = dict(zip(self.names, t, strict=True))
        address = self.part != "round"
        table = v["token_table"]
        offsets, prompts = v["rope_offsets"], v["prompt_lengths"]
        slots, ids, positions = v["k2_slots"], v["input_ids"], v["positions"]
        rope, verify, cursors = v["rope_positions"], v["verify_rope_positions"], v["cursors"]
        if cutlass.const_expr(address):
            cache, writes = v["cache_index_table"], v["cache_write_indices"]
            ptr, indices = v["kv_indptr"], v["kv_indices"]
        start, rd = v["start_frontiers"], v["round_done"]
        g = guard + b * self.q
        if owner:
            if tid == 0:
                start[b], rd[b] = f, d
                if cutlass.const_expr(self.k2):
                    slots[b] = Int32(-1) if d else Int32(r)
                if cutlass.const_expr(self.cursors):
                    cursors[b] = Int64(Uint64(f) - Uint64(prompts[b]) + Uint64(1))
                if cutlass.const_expr(address):
                    ptr[b + 1] = Int32(base + length)
            if cutlass.const_expr(self.q <= THREADS):
                if tid < self.q:
                    position = Int64(tid)
                    k = b * self.q + position
                    logical = Int64(Uint64(f) + Uint64(position))
                    rotary = min(logical, m - 1)
                    positions[k], rope[k] = logical, rotary
                    if cutlass.const_expr(self.rope):
                        verify[k] = Int64(Uint64(rotary) + Uint64(offsets[b]))
                    token = mask
                    if position == 0:
                        column = min(max(f, Int64(0)), m - 1)
                        token = table[r * tok_stride + column]
                    ids[k] = token
                    if cutlass.const_expr(address):
                        index = Int32(g + position)
                        if not dummy:
                            column = f + position
                            if column < m:
                                index = cache[r * ci_stride + column]
                            else:
                                index = Int32(g + min(column - m, Int64(self.q - 1)))
                        writes[k] = index
            else:
                for position in cutlass.range(Int64(tid), self.q, THREADS):
                    k = b * self.q + position
                    logical = Int64(Uint64(f) + Uint64(position))
                    rotary = min(logical, m - 1)
                    positions[k], rope[k] = logical, rotary
                    if cutlass.const_expr(self.rope):
                        verify[k] = Int64(Uint64(rotary) + Uint64(offsets[b]))
                    token = mask
                    if position == 0:
                        column = min(max(f, Int64(0)), m - 1)
                        token = table[r * tok_stride + column]
                    ids[k] = token
                    if cutlass.const_expr(address):
                        index = Int32(g + position)
                        if not dummy:
                            column = f + position
                            if column < m:
                                index = cache[r * ci_stride + column]
                            else:
                                index = Int32(g + min(column - m, Int64(self.q - 1)))
                        writes[k] = index
        if cutlass.const_expr(address and copy):
            # Global flattened grid-stride, with the same worker residue in
            # each segment. Adjacent lanes copy adjacent columns within a row.
            j = (worker + step - base % step) % step
            bound = min(length, n - base)
            while j < bound:
                # Four neighboring worker residues have one vector owner. At
                # segment/iteration boundaries they may reach this group in
                # different iterations, but exactly its leading lane stores it.
                first = j - Int64(tid % 4)
                source = cache.toint() + (r * ci_stride + first) * 4
                destination = indices.toint() + (base + first) * 4
                vector = (not dummy) & (first >= 0) & (first + 3 < bound) & (first + 3 < m)
                vector = vector & (source % 16 == 0) & (destination % 16 == 0)
                if vector:
                    if tid % 4 == 0:
                        src = cute.make_tensor(
                            cute.make_ptr(Int32, source, cute.AddressSpace.gmem, assumed_align=16),
                            cute.make_layout(4),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                Int32, destination, cute.AddressSpace.gmem, assumed_align=16
                            ),
                            cute.make_layout(4),
                        )
                        values = cute.make_rmem_tensor(cute.make_layout(4), Int32)
                        load = cute.make_copy_atom(
                            cute.nvgpu.CopyG2ROp(), Int32, num_bits_per_copy=128
                        )
                        save = cute.make_copy_atom(
                            cute.nvgpu.CopyR2GOp(), Int32, num_bits_per_copy=128
                        )
                        cute.copy(load, src, values)
                        cute.copy(save, values, dst)
                else:
                    index = Int32(g + j)
                    if not dummy:
                        if j < m:
                            index = cache[r * ci_stride + j]
                        else:
                            index = Int32(g + min(j - m, Int64(self.q - 1)))
                    indices[base + j] = index
                j += step


class Commit:
    """One warp per row. generic: grid-stride over rows from any number of CTAs;
    otherwise one CTA with warp b on row b (B<=4). The host picks by B."""

    def __init__(self, q, s, mode, k2, pdl, wait=True, generic=False):
        self.q, self.s, self.mode, self.k2 = q, s, mode, k2
        self.pdl, self.wait, self.generic = pdl, wait, generic

    @cute.jit
    def __call__(self, t: tuple, m: Int64, blocks: Int32, stream: cuda.CUstream):
        self.kernel(
            tuple(x.iterator for x in t),
            t[0].shape[0],
            m,
            t[3].stride[0],
            t[5].stride[0],
            t[10].stride[0],
        ).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(
        self,
        t: tuple,
        batch: Int64,
        m: Int64,
        proposal_stride: Int64,
        token_stride: Int64,
        table_stride: Int64,
    ):
        if cutlass.const_expr(self.pdl and self.wait):
            cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        if cutlass.const_expr(not self.generic):
            b = tid // 32
            if b < batch:
                self.row(t, b, lane, m, proposal_stride, token_stride, table_stride)
        else:
            # Rows are independent: grid-stride over them, one warp per row.
            cta, _, _ = cute.arch.block_idx()
            grid, _, _ = cute.arch.grid_dim()
            b = Int64(cta * (THREADS // 32) + tid // 32)
            while b < batch:
                self.row(t, b, lane, m, proposal_stride, token_stride, table_stride)
                b += Int64(grid * (THREADS // 32))

    @cute.jit
    def row(
        self,
        t: tuple,
        b,  # Int32 (B<=4 kernel, as before) or Int64 (any-B kernel)
        lane: Int32,
        m: Int64,
        proposal_stride: Int64,
        token_stride: Int64,
        table_stride: Int64,
    ):
        (
            rows,
            start,
            rd,
            proposals,
            targets,
            tokens,
            accepted,
            ends,
            ignore,
            stops,
            table,
            fronts,
            dones,
            committed,
            record,
            lengths,
        ) = t
        r = Int64(rows[b])
        f, d = start[b], rd[b]
        if cutlass.const_expr(self.q <= 32):
            token = Int32(0)
            if lane < self.q:
                if cutlass.const_expr(self.mode == "greedy"):
                    token = Int32(targets[b * self.q + lane])
                    tokens[b * token_stride + lane] = token
                else:
                    token = tokens[b * token_stride + lane]
            # Register tokens are used below: never read this launch's writes.
            a = Int64(0)
            if cutlass.const_expr(self.mode == "sampled"):
                a = Int64(accepted[b])
            else:
                mismatch = Int32(self.q - 1)
                if lane < self.q - 1:
                    if proposals[b * proposal_stride + lane] != token:
                        mismatch = lane
                for k in cutlass.range_constexpr(5):
                    mismatch = min(mismatch, cute.arch.shuffle_sync_bfly(mismatch, offset=1 << k))
                a = Int64(mismatch)
            v, fin = Int64(0), cutlass.Boolean(True)
            if (not d) & (f >= 0) & (f < m):
                e = min(ends[b], m)
                budget = Int64(0)
                if e > f + 1:
                    budget = min(e - f - 1, Int64(self.q))
                limit = min(budget, a + 1)
                first = Int32(self.q)
                if (lane < self.q) & (Int64(lane) < limit) & (ignore[b] == 0):
                    # S=0 specializes away every stop load, including dummy.
                    for s in cutlass.range_constexpr(self.s):
                        if token == stops[s]:
                            first = lane
                for k in cutlass.range_constexpr(5):
                    first = min(first, cute.arch.shuffle_sync_bfly(first, offset=1 << k))
                v = Int64(first + 1) if first < self.q else limit
                fin = (first < self.q) | (f + v + 1 >= e)
                if (lane < self.q) & (Int64(lane) < v):
                    table[r * table_stride + f + 1 + lane] = token
        else:
            # Each lane scans its positions; only integer minima cross lanes.
            mismatch = Int32(self.q - 1)
            for position in cutlass.range(Int64(lane), self.q, 32):
                token = Int32(0)
                if cutlass.const_expr(self.mode == "greedy"):
                    token = Int32(targets[Int64(b) * self.q + position])
                    tokens[b * token_stride + position] = token
                else:
                    token = tokens[b * token_stride + position]
                if cutlass.const_expr(self.mode != "sampled"):
                    if position < self.q - 1:
                        if proposals[b * proposal_stride + position] != token:
                            mismatch = min(mismatch, Int32(position))
            a = Int64(0)
            if cutlass.const_expr(self.mode == "sampled"):
                a = Int64(accepted[b])
            else:
                for k in cutlass.range_constexpr(5):
                    mismatch = min(mismatch, cute.arch.shuffle_sync_bfly(mismatch, offset=1 << k))
                a = Int64(mismatch)
            v, fin = Int64(0), cutlass.Boolean(True)
            if (not d) & (f >= 0) & (f < m):
                e = min(ends[b], m)
                budget = Int64(0)
                if e > f + 1:
                    budget = min(e - f - 1, Int64(self.q))
                limit = min(budget, a + 1)
                first = Int32(self.q)
                if ignore[b] == 0:
                    for position in cutlass.range(Int64(lane), limit, 32):
                        if cutlass.const_expr(self.mode == "greedy"):
                            token = Int32(targets[Int64(b) * self.q + position])
                        else:
                            token = tokens[b * token_stride + position]
                        for s in cutlass.range_constexpr(self.s):
                            if token == stops[s]:
                                first = min(first, Int32(position))
                for k in cutlass.range_constexpr(5):
                    first = min(first, cute.arch.shuffle_sync_bfly(first, offset=1 << k))
                v = Int64(first) + 1 if first < self.q else limit
                fin = (first < self.q) | (f + v + 1 >= e)
                for position in cutlass.range(Int64(lane), v, 32):
                    if cutlass.const_expr(self.mode == "greedy"):
                        token = Int32(targets[Int64(b) * self.q + position])
                    else:
                        token = tokens[b * token_stride + position]
                    table[r * table_stride + f + 1 + position] = token
        if lane == 0:
            fronts[r], dones[r] = f + v, fin
            accepted[b] = Int32(-1) if d else Int32(a)
            committed[b], record[b] = Int32(v), Int32(fin)
            if cutlass.const_expr(self.k2):
                lengths[b] = Int32(v)
