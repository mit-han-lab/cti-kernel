# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Packed GQA split-KV attention with shared indices and an asynchronous KV ring.

Design reference: FlashAttention cute/flash_fwd.py, cute/ampere_helpers.py
The MMA accumulator's two adjacent columns become pairs in the next MMA's A
operand. Four lanes share each row's online softmax state.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.experimental import iket
from cutlass.utils import SmemAllocator

from ._geometry import counter_levels, merge_columns, merge_geometry

# Wide spread grids (B > 32) keep each lane's request unions in registers up to this
# many groups of 32 requests; larger batches reload them in runtime loops.
WIDE_CACHED_GROUPS = 8
# Smallest piece a wide spread grid cuts a request into (tree merge). RTX 5090 B=33-128,
# one long request of 2k keys: 256 -> 35-56 us, 1024 -> 47-67 us; 64-key pieces of a
# windowed 2k-key union cost 1.6x the per-request grid (merge-bound).
WIDE_MIN_CHUNK = 256


@cute.jit
def pointer(base, offset, align: cutlass.Constexpr = 16):
    return cute.make_ptr(
        base.dtype,
        base.toint() + Int64(offset) * (base.dtype.width // 8),
        base.memspace,
        assumed_align=align,
    )


@cute.jit
def async_row(dst, src, offset, valid):
    # PTX source-size zero fills the entire transaction without dereferencing src.
    count = Int32(0)
    if valid:
        count = Int32(16)
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[dst.toint().to(Int32), pointer(src, offset).toint(), count],
    )


class Attention:
    def __init__(
        self,
        query,
        group,
        rows,
        heads,
        causal,
        windowed: bool,
        splits,
        dtype,
        min_chunk=128,
        key_tile=32,
        key_groups=2,
        stages=3,
        chunk_cap=1024,
        page=1,
        pieces=False,
        sparse=False,
        short_union=0,
        inline_merge=False,
        distributed_merge=False,
        coop_pdl=True,
        head_dim=128,
        write_pages=True,
        iket=False,
        q_row_stride=None,
        table=False,
        early_trigger=False,
        spread=0,
        spread_budget=0,
    ):
        if head_dim not in (64, 128, 256):
            raise ValueError("DRAFT CuTe requires D in {64,128,256}")
        if table not in (False, "request", "row"):
            raise ValueError("table must be False, 'request' or 'row'")
        if table and (page != 1 or (splits > 1 and not inline_merge)):
            raise ValueError("table attention requires page_size=1 and an inline merge")
        # "row" metadata is indexed by execution row b, saving the dependent
        # request-indexed load before the table read.
        self.table, self.row_metadata = bool(table), table == "row"
        self.dim = head_dim
        self.q_row_stride = heads * group * head_dim if q_row_stride is None else q_row_stride
        self.iket = iket
        # D64/D128 use S<3,3,3>; D256 shifts its source bits with the row stride.
        # With D=256, S<3,3,3> survives ldmatrix partitioning as a composed
        # layout; CuTe 4.8 lowers its copies with aliased column/K-block addresses.
        self.swizzle_shift = 4 if head_dim == 256 else 3
        self.strip = min(128, head_dim)
        self.write_pages = write_pages
        if key_tile not in (32, 64):
            raise ValueError("DRAFT CuTe requires KEY_TILE in {32,64}")
        if chunk_cap < key_tile or chunk_cap % key_tile or min_chunk < 1:
            raise ValueError("DRAFT CuTe requires a positive MIN_CHUNK and tile-aligned CHUNK_CAP")
        if key_groups not in (1, 2) or stages not in (2, 3):
            raise ValueError("DRAFT CuTe requires KEY_GROUPS in {1,2} and STAGES in {2,3}")
        if type(short_union) is not int or not 0 <= short_union <= chunk_cap:
            raise ValueError("DRAFT CuTe requires 0 <= short_union <= chunk_cap")
        if spread and (
            spread < 2
            or table
            or not inline_merge
            or spread_budget % spread
            or not spread <= spread_budget <= splits
        ):
            raise ValueError("DRAFT CuTe spread needs B >= 2, CSR inputs and an inline merge")
        self.spread_budget = spread_budget
        # spread = B: grid x holds B * (per-request splits) CTAs per row tile, shared by
        # the requests in proportion to their runtime key unions (0 = per-request grid).
        self.spread = spread
        # Wide spread grids launch head-major (x = head and row tile): a request's head
        # CTAs run together and share its paged K/V rows, as the per-request grid's do.
        self.head_major = spread > 32 and splits <= 65535
        self.distributed_merge = distributed_merge
        self.early_trigger = early_trigger
        self.use_pdl = not distributed_merge or coop_pdl
        self.short_union, self.inline_merge = short_union, inline_merge
        self.query, self.group, self.rows, self.heads = query, group, rows, heads
        self.causal, self.windowed, self.splits, self.dtype = causal, windowed, splits, dtype
        self.rt = (query * group + rows - 1) // rows
        self.min_chunk, self.kt = min_chunk, key_tile
        self.groups, self.stages = key_groups, stages
        self.nk = key_tile // self.groups
        self.mma_threads = rows // 16 * 32
        self.threads = self.mma_threads * self.groups
        self.index_parts = (self.kt * (self.dim // 8) + self.threads - 1) // self.threads
        self.chunk_cap = chunk_cap
        self.pieces = pieces
        self.sparse = sparse
        self.page = page
        # Merge keeps this many partial rows in flight with the LSE loads.
        self.unroll = min(splits, 16)
        # The largest ring leaves no room for indices under the 99 KiB budget.
        self.v_stages = stages - 1 if key_tile == 64 and stages == 3 else stages
        self.delayed_v = self.v_stages != stages
        self.wait_groups = (stages - 2) * (2 if self.delayed_v else 1)
        self.kv_elements = self.kt * self.dim * self.stages
        self.combine_elements = rows * self.dim * 2 if key_groups == 2 else 0
        self.merge_fan_in, arena_bytes = merge_geometry(
            rows,
            key_tile,
            key_groups,
            stages,
            inline_merge and splits > 1 and not distributed_merge,
            self.dim,
            distributed=distributed_merge,
        )
        self.merge_cols = merge_columns(rows, self.dim)
        self.merge_levels = counter_levels(splits, self.merge_fan_in)
        self.counter_offsets = tuple(
            sum(self.merge_levels[:i]) for i in range(len(self.merge_levels))
        )
        self.smem_elements = arena_bytes // 2
        self.smem_bytes = arena_bytes + chunk_cap * 4
        self.merge_threads = min(128, self.threads)
        self.merge_capacity = min(
            16 * self.merge_threads // (self.dim // 4), arena_bytes // (4 * (self.dim + 4))
        )
        # Reuse the arena; small legacy rings split the FP32 epilogue columns.
        self.epilogue_cols = (
            self.dim if self.smem_elements * 2 >= rows * self.dim * 4 else self.dim // 2
        )
        if self.smem_bytes + 1024 > 99 * 1024:  # Include CuTe's static reserve.
            raise ValueError("DRAFT CuTe shared memory exceeds 99 KiB")
        self.scale = self.dim**-0.5 * 1.4426950408889634

    @cute.jit
    def trace_begin(self, name: cutlass.Constexpr):
        if const_expr(self.iket):
            iket.range_push(name)

    @cute.jit
    def trace_end(self):
        if const_expr(self.iket):
            iket.range_pop()

    @cute.jit
    def v_layout(self, transpose: cutlass.Constexpr = False):
        shape = (self.nk, self.dim, self.v_stages, self.groups)
        stride = (self.dim, 1, self.kt * self.dim, self.nk * self.dim)
        if const_expr(transpose):
            shape = (self.dim, self.nk, self.v_stages, self.groups)
            stride = (1, self.dim, self.kt * self.dim, self.nk * self.dim)
        return cute.make_composed_layout(
            cute.make_swizzle(3, 3, self.swizzle_shift), 0, cute.make_layout(shape, stride=stride)
        )

    @cute.jit
    def __call__(
        self,
        slots: cute.Tensor,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        work: cute.Tensor,
        counters: cute.Tensor,
        qp: cute.Tensor,
        pp: cute.Tensor,
        pi: cute.Tensor,
        last: cute.Tensor,
        fused: Int32,
        window: Int32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.rows // 16, 1, 1),
            # ldmatrix.x4 B-operand copies need a 16-wide N tile.
            permutation_mnk=(self.rows, 16, 16),
        )
        launch = self.kernel(
            slots,
            ptr,
            idx,
            q.iterator,
            k.iterator,
            v.iterator,
            out.iterator,
            work.iterator,
            counters,
            qp,
            pp,
            pi,
            last,
            fused,
            window,
            mma,
        )
        grid = (self.splits, self.heads * self.rt, 1 if self.spread else slots.shape[0])
        if const_expr(self.head_major):
            grid = (self.heads * self.rt, self.splits, 1)
        if const_expr(self.distributed_merge):
            launch.launch(
                grid=grid,
                block=(self.threads, 1, 1),
                stream=stream,
                cooperative=True,
                use_pdl=self.use_pdl,
            )
        else:
            launch.launch(
                grid=grid,
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def table_call(
        self,
        slots: cute.Tensor,
        frontier: cute.Tensor,
        done: cute.Tensor,
        idx: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        work: cute.Tensor,
        counters: cute.Tensor,
        stride: Int64,
        guard: Int32,
        maximum: Int32,
        window: Int32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.rows // 16, 1, 1),
            # ldmatrix.x4 B-operand copies need a 16-wide N tile.
            permutation_mnk=(self.rows, 16, 16),
        )
        launch = self.kernel(
            slots,
            frontier,
            idx,
            q.iterator,
            k.iterator,
            v.iterator,
            out.iterator,
            work.iterator,
            counters,
            done,
            done,
            done,
            done,
            Int32(0),
            window,
            mma,
            (stride, guard, maximum),
        )
        if const_expr(self.distributed_merge):
            launch.launch(
                grid=(self.splits, self.heads * self.rt, 1 if self.spread else slots.shape[0]),
                block=(self.threads, 1, 1),
                stream=stream,
                cooperative=True,
                use_pdl=self.use_pdl,
            )
        else:
            launch.launch(
                grid=(self.splits, self.heads * self.rt, 1 if self.spread else slots.shape[0]),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def publish(self, slots, ptr, idx, qp, pp, pi, last, b, cta, tid, per):
        # Prepare's CSR for request b, written by that request's CTAs (same values).
        # Page counts and their sum fit: pages <= tokens <= B*C < 2**31.
        queries, pages = Int32(0), Int32(0)
        for row in range(b):
            if slots[row] != -1:
                queries += self.query
                pages += Int32((Int64(ptr[row + 1] - ptr[row]) + self.page - 1) // self.page)
        start, length, count = Int32(0), Int32(0), Int32(0)
        if slots[b] != -1:
            start = ptr[b]
            length = ptr[b + 1] - start
            count = Int32((Int64(length) + self.page - 1) // self.page)
            if tid == 0 and cta == 0:
                last[b] = (length - 1) % self.page + 1
                qp[b + 1] = queries + self.query
        else:
            if tid == 0 and cta == 0:
                last[b] = 0
                qp[b + 1] = queries
        if tid == 0 and cta == 0:
            pp[b + 1] = pages + count
            if b == 0:
                pp[0], qp[0] = 0, 0
        if const_expr(self.write_pages):
            if const_expr(self.spread):
                # per = this request's CTAs per row tile (runtime): unit-step loop.
                stride = Int64(per) * self.heads * self.rt * self.threads
                first = Int64(cta) * self.threads + tid
                steps = max(Int64(0), (Int64(count) - first + stride - 1) // stride)
                for step in range(steps):
                    page = first + step * stride
                    pi[pages + page] = idx[start + page * self.page] // self.page
            else:
                stride = self.splits * self.heads * self.rt * self.threads
                for page in range(Int64(cta) * self.threads + tid, Int64(count), stride):
                    pi[pages + page] = idx[start + page * self.page] // self.page

    @cute.jit
    def count_index(self, batch, b):
        # Per-request nonempty split count, after the partials and their LSEs.
        return self.slot_count(batch) * self.rows * (self.dim + 1) + b

    def slot_count(self, batch):
        # Partial slots (rows * dim floats each): spread grids hold B in self.splits.
        if const_expr(self.spread):
            return Int64(self.heads) * self.rt * self.splits
        return Int64(batch) * self.heads * self.rt * self.splits

    def slot_base(self, b, head, rt, prefix):
        # First partial slot of (request, head, row tile); prefix = request's first CTA.
        if const_expr(self.spread):
            return (Int64(head) * self.rt + rt) * self.splits + prefix
        return ((Int64(b) * self.heads + head) * self.rt + rt) * self.splits

    @cute.jit
    def spread_count(self, union, chunk):
        # One CTA for inactive, empty and short requests (they still own output rows).
        count = max(Int32(1), (union + chunk - 1) // chunk)
        if const_expr(self.short_union > 0):
            if union <= self.short_union:
                count = Int32(1)
        return count

    @cute.jit
    def spread_schedule(self, slots, ptr, window, x, tid):
        """Map grid-x CTA x to (request, split) from the runtime key unions.

        The grid row holds S >= B * s CTAs (s: per-request splits). Chunk
        c = ceil(T / (B * s)) over the summed unions T gives uniform batches the
        per-request chunk ceil(u / s) and count, so their schedule is unchanged. If a
        ragged batch's counts exceed S, requests that fit one chunk keep one CTA and
        the others share the remaining CTAs, c = ceil(T_long / (S - ones)); if
        rounding still overflows, c = ceil(T / (S - B + 1)) fits since
        sum ceil(u_r / c) < T / c + B. Without the piece loop a chunk may not exceed
        the index buffer; then each request keeps its per-request split. Lane r owns
        request r (B <= 32): every warp derives the schedule with warp reductions and
        a scan. Int32 throughout: the CSR bounds the summed lengths below 2**31.
        """
        batch = const_expr(self.spread)
        if const_expr(batch <= 32):
            total = const_expr(self.splits)
            budget = const_expr(self.spread_budget)
            tile = self.kt
            lane = tid % 32
            lo, union = Int32(0), Int32(0)
            if lane < batch:
                flag, first, last = slots[lane], ptr[lane], ptr[lane + 1]
                length = last - first
                if const_expr(self.windowed):
                    lo = max(0, length - self.query - window + 1)
                if flag != -1:
                    union = length - lo
            sum_union = cute.arch.warp_redux_sync(union, "add")
            chunk = self.round_chunk(sum_union, budget)
            count = Int32(0)
            if lane < batch:
                count = self.spread_count(union, chunk)
            own = Int32(0)
            if cute.arch.warp_redux_sync(count, "add") > total:
                long_union, ones = Int32(0), Int32(0)
                if lane < batch:
                    if count > 1:
                        long_union = union
                    else:
                        ones = Int32(1)
                room = total - cute.arch.warp_redux_sync(ones, "add")
                chunk = self.round_chunk(cute.arch.warp_redux_sync(long_union, "add"), room)
                if lane < batch:
                    count = self.spread_count(union, chunk)
                if cute.arch.warp_redux_sync(count, "add") > total:
                    chunk = self.round_chunk(sum_union, const_expr(total - batch + 1))
                if const_expr(not self.pieces):
                    if chunk > self.chunk_cap:
                        own = Int32(1)
                if lane < batch:
                    count = self.spread_count(union, self.spread_piece(union, chunk, own))
            inclusive = count
            for step in cutlass.range_constexpr((batch - 1).bit_length()):
                value = cute.arch.shuffle_sync_up(inclusive, 1 << step)
                if lane >= (1 << step):
                    inclusive += value
            # redux.sync results are warp-uniform, which keeps the per-CTA index math on
            # the uniform datapath (shuffled values would not be). Prefixes are monotone.
            passed = Int32(0)
            if (lane < batch) & (inclusive <= x):
                passed = Int32(1)
            owner = cute.arch.warp_redux_sync(passed, "add")
            b, split, prefix = Int32(0), Int32(total), Int32(0)
            lo_b, chunk_b, count_b = Int32(0), Int32(0), Int32(1)
            if owner < batch:
                mine = Int32(0)
                if lane < owner:
                    mine = count
                prefix = cute.arch.warp_redux_sync(mine, "add")
                b, split = owner, x - prefix
                # The owner's lane loaded these words; reload them at a uniform address.
                length = ptr[owner + 1] - ptr[owner]
                if const_expr(self.windowed):
                    lo_b = max(0, length - self.query - window + 1)
                union = Int32(0)
                if slots[owner] != -1:
                    union = length - lo_b
                chunk_b = self.spread_piece(union, chunk, own)
                count_b = self.spread_count(union, chunk_b)
                if const_expr(self.short_union > 0):
                    if union <= self.short_union:
                        chunk_b = (union + tile - 1) // tile * tile
            # Otherwise idle: split = S and chunk 0 give an empty key range; b = 0 keeps
            # its loads in bounds.
            return b, split, prefix, lo_b, Int64(chunk_b), count_b

        else:
            return self.spread_schedule_wide(slots, ptr, window, x, tid)

    @cute.jit
    def spread_union(self, slots, ptr, window, request):
        # (key union, 1 if the request is active): zero for inactive and padding lanes.
        union, live = Int32(0), Int32(0)
        if request < self.spread:
            length = ptr[request + 1] - ptr[request]
            lo = Int32(0)
            if const_expr(self.windowed):
                lo = max(0, length - self.query - window + 1)
            if slots[request] != -1:
                union, live = length - lo, Int32(1)
        return union, live

    @cute.jit
    def spread_lane_count(self, union, request, chunk):
        # (CTAs, 1 if one CTA, union if more) for one lane; padding lanes take none.
        count, one, long = Int32(0), Int32(0), Int32(0)
        if request < self.spread:
            count = self.spread_count(union, chunk)
            if count > 1:
                long = union
            else:
                one = Int32(1)
        return count, one, long

    @cute.jit
    def spread_totals(self, slots, ptr, window, lane, chunk, unions):
        # Warp-uniform (CTAs, one-CTA requests, keys of the others). Cached: the lane's
        # block of requests is summed in registers first, so one redux per quantity.
        total, ones, long_union = Int32(0), Int32(0), Int32(0)
        if const_expr(unions is not None):
            first = lane * len(unions)
            for item in cutlass.range_constexpr(len(unions)):
                count, one, long = self.spread_lane_count(unions[item], first + item, chunk)
                total += count
                ones += one
                long_union += long
            total = cute.arch.warp_redux_sync(total, "add")
            ones = cute.arch.warp_redux_sync(ones, "add")
            long_union = cute.arch.warp_redux_sync(long_union, "add")
        else:
            for group in range(Int32((self.spread + 31) // 32)):
                request = group * 32 + lane
                group_union, _ = self.spread_union(slots, ptr, window, request)
                count, one, long = self.spread_lane_count(group_union, request, chunk)
                total += cute.arch.warp_redux_sync(count, "add")
                ones += cute.arch.warp_redux_sync(one, "add")
                long_union += cute.arch.warp_redux_sync(long, "add")
        return total, ones, long_union

    @cute.jit
    def spread_owner(self, union, request, lane, chunk, own, x, running):
        # One group's share of the owner search: (group total, owner lane, owner prefix).
        count = Int32(0)
        if request < self.spread:
            count = self.spread_count(union, self.spread_piece(union, chunk, own))
        group_total = cute.arch.warp_redux_sync(count, "add")
        owner, prefix = Int32(0), Int32(0)
        if (running <= x) & (x < running + group_total):
            inclusive = count
            for step in cutlass.range_constexpr(5):
                value = cute.arch.shuffle_sync_up(inclusive, 1 << step)
                if lane >= (1 << step):
                    inclusive += value
            passed = Int32(0)
            if (request < self.spread) & (inclusive <= x - running):
                passed = Int32(1)
            owner = cute.arch.warp_redux_sync(passed, "add")
            mine = Int32(0)
            if lane < owner:
                mine = count
            prefix = running + cute.arch.warp_redux_sync(mine, "add")
        return group_total, owner, prefix

    @cute.jit
    def wide_chunk(self, keys, parts):
        # Wide grids merge with the tree: no request is cut below WIDE_MIN_CHUNK keys.
        minimum = const_expr(WIDE_MIN_CHUNK if self.pieces else min(WIDE_MIN_CHUNK, self.chunk_cap))
        return max(self.round_chunk(keys, parts), Int32(minimum))

    @cute.jit
    def spread_schedule_wide(self, slots, ptr, window, x, tid):
        batch = const_expr(self.spread)
        total = const_expr(self.splits)
        groups = const_expr((batch + 31) // 32)
        per_request = const_expr(self.spread_budget // self.spread)
        lane = tid % 32
        b, split, prefix = Int32(0), Int32(total), Int32(0)
        lo_b, chunk_b, count_b = Int32(0), Int32(0), Int32(1)
        own = Int32(0)
        if const_expr(groups <= WIDE_CACHED_GROUPS):
            # Lane l owns requests l*G .. l*G+G-1 and loads them once, with this CTA's
            # own request for the one-CTA-per-request case: one memory round trip. Only
            # redux results feed the uniform chunk, prefix and owner math.
            first = lane * groups
            unions = []
            lane_union, lane_active = Int32(0), Int32(0)
            for item in cutlass.range_constexpr(groups):
                item_union, item_live = self.spread_union(slots, ptr, window, first + item)
                unions.append(item_union)
                lane_union += item_union
                lane_active += item_live
            mine_length, mine_flag = Int32(0), Int32(-1)
            if x < batch:
                mine_length = ptr[x + 1] - ptr[x]
                mine_flag = slots[x]
            sum_union = cute.arch.warp_redux_sync(lane_union, "add")
            active = cute.arch.warp_redux_sync(lane_active, "add")
            # Budget over active requests: inactive ones keep a CTA but no keys.
            chunk = self.wide_chunk(sum_union, max(active, 1) * per_request)
            count, ones, long_union = self.spread_totals(slots, ptr, window, lane, chunk, unions)
            if count == batch:
                # Every request takes one CTA: CTA x is request x.
                if x < batch:
                    b, split, prefix = x, Int32(0), x
                    if const_expr(self.windowed):
                        lo_b = max(0, mine_length - self.query - window + 1)
                    mine_union = Int32(0)
                    if mine_flag != -1:
                        mine_union = mine_length - lo_b
                    chunk_b = chunk
                    if const_expr(self.short_union > 0):
                        if mine_union <= self.short_union:
                            chunk_b = (mine_union + self.kt - 1) // self.kt * self.kt
            else:
                if count > total:
                    chunk = self.wide_chunk(long_union, total - ones)
                    count, ones, long_union = self.spread_totals(
                        slots, ptr, window, lane, chunk, unions
                    )
                    if count > total:
                        chunk = self.wide_chunk(sum_union, const_expr(total - batch + 1))
                    if const_expr(not self.pieces):
                        if chunk > self.chunk_cap:
                            own = Int32(1)
                lane_total = Int32(0)
                for item in cutlass.range_constexpr(groups):
                    if first + item < batch:
                        piece = self.spread_piece(unions[item], chunk, own)
                        lane_total += self.spread_count(unions[item], piece)
                inclusive = lane_total
                for step in cutlass.range_constexpr(5):
                    value = cute.arch.shuffle_sync_up(inclusive, 1 << step)
                    if lane >= (1 << step):
                        inclusive += value
                # Blocks are in request order, so the lanes ending at or before x are a
                # prefix; the owner lane's block is walked at uniform addresses.
                passed = Int32(0)
                if inclusive <= x:
                    passed = Int32(1)
                owner = cute.arch.warp_redux_sync(passed, "add")
                mine = Int32(0)
                if lane < owner:
                    mine = lane_total
                running = cute.arch.warp_redux_sync(mine, "add")
                if owner < 32:
                    found = Int32(0)
                    for item in cutlass.range_constexpr(groups):
                        request = owner * groups + item
                        item_union, _ = self.spread_union(slots, ptr, window, request)
                        item_count = Int32(0)
                        if request < batch:
                            piece = self.spread_piece(item_union, chunk, own)
                            item_count = self.spread_count(item_union, piece)
                        if (found == 0) & (x < running + item_count):
                            b, split, prefix = request, x - running, running
                            found = Int32(1)
                        running += item_count
                if split < total:
                    lo_b, chunk_b, count_b = self.spread_owner_words(
                        slots, ptr, window, b, chunk, own
                    )
        else:
            # Larger B: runtime group loops that reload (group g holds 32g..32g+31).
            sum_union, active = Int32(0), Int32(0)
            for group in range(Int32(groups)):
                group_union, group_live = self.spread_union(slots, ptr, window, group * 32 + lane)
                sum_union += cute.arch.warp_redux_sync(group_union, "add")
                active += cute.arch.warp_redux_sync(group_live, "add")
            chunk = self.wide_chunk(sum_union, max(active, 1) * per_request)
            count, ones, long_union = self.spread_totals(slots, ptr, window, lane, chunk, None)
            if count > total:
                chunk = self.wide_chunk(long_union, total - ones)
                count, ones, long_union = self.spread_totals(slots, ptr, window, lane, chunk, None)
                if count > total:
                    chunk = self.wide_chunk(sum_union, const_expr(total - batch + 1))
                if const_expr(not self.pieces):
                    if chunk > self.chunk_cap:
                        own = Int32(1)
            running = Int32(0)
            for group in range(Int32(groups)):
                request = group * 32 + lane
                search_union, _ = self.spread_union(slots, ptr, window, request)
                group_total, owner, start = self.spread_owner(
                    search_union, request, lane, chunk, own, x, running
                )
                if (running <= x) & (x < running + group_total):
                    prefix = start
                    b, split = group * 32 + owner, x - start
                running += group_total
            if split < total:
                lo_b, chunk_b, count_b = self.spread_owner_words(slots, ptr, window, b, chunk, own)
        return b, split, prefix, lo_b, Int64(chunk_b), count_b

    @cute.jit
    def spread_owner_words(self, slots, ptr, window, b, chunk, own):
        # The owner's (lo, chunk, count), reloaded at a uniform address.
        length = ptr[b + 1] - ptr[b]
        lo = Int32(0)
        if const_expr(self.windowed):
            lo = max(0, length - self.query - window + 1)
        union = Int32(0)
        if slots[b] != -1:
            union = length - lo
        chunk_b = self.spread_piece(union, chunk, own)
        count_b = self.spread_count(union, chunk_b)
        if const_expr(self.short_union > 0):
            if union <= self.short_union:
                chunk_b = (union + self.kt - 1) // self.kt * self.kt
        return lo, chunk_b, count_b

    @cute.jit
    def round_chunk(self, keys, parts):
        # ceil(keys / parts) rounded up to the key tile, at least MIN_CHUNK.
        chunk = ((keys + parts - 1) // parts + self.kt - 1) // self.kt * self.kt
        return max(chunk, Int32(self.min_chunk))

    @cute.jit
    def spread_piece(self, union, chunk, own):
        # own: the shared chunk would exceed the index buffer; keep per-request splits.
        piece = chunk
        if const_expr(not self.pieces):
            if own != 0:
                piece = self.round_chunk(union, const_expr(self.spread_budget // self.spread))
        return piece

    @cute.jit
    def split_bounds(self, length, window):
        lo = Int32(0)
        if const_expr(self.windowed):
            lo = max(0, length - self.query - window + 1)
        union = Int64(length) - lo
        chunk = max(
            ((union + self.splits - 1) // self.splits + self.kt - 1) // self.kt * self.kt,
            self.min_chunk,
        )
        count = Int32((union + chunk - 1) // chunk)
        if const_expr(self.short_union > 0):
            if union <= self.short_union:
                chunk = (union + self.kt - 1) // self.kt * self.kt
                count = Int32(1)
        # Rounded chunks may reach 2**31; only the split count is bounded by splits.
        return lo, chunk, count

    @cute.jit
    def walk(
        self,
        piece,
        pend,
        end,
        length,
        window,
        rt,
        warp,
        lane,
        group,
        head,
        tid,
        k,
        v,
        sk,
        sv,
        vt,
        indices,
        mma,
        the,
        aq,
        bk,
        bv,
        ck,
        cv,
        acc,
        maxima,
        sums,
    ):
        # A request no longer than W has every key within the window (|p-k| <= L-1 < W):
        # a CTA-uniform runtime branch takes the unwindowed walk, so W never selects code.
        if const_expr(self.windowed):
            if length <= window:
                self.walk_keys(
                    piece,
                    pend,
                    end,
                    length,
                    window,
                    rt,
                    warp,
                    lane,
                    group,
                    head,
                    tid,
                    k,
                    v,
                    sk,
                    sv,
                    vt,
                    indices,
                    mma,
                    the,
                    aq,
                    bk,
                    bv,
                    ck,
                    cv,
                    acc,
                    maxima,
                    sums,
                    windowed=False,
                )
            else:
                self.walk_keys(
                    piece,
                    pend,
                    end,
                    length,
                    window,
                    rt,
                    warp,
                    lane,
                    group,
                    head,
                    tid,
                    k,
                    v,
                    sk,
                    sv,
                    vt,
                    indices,
                    mma,
                    the,
                    aq,
                    bk,
                    bv,
                    ck,
                    cv,
                    acc,
                    maxima,
                    sums,
                    windowed=True,
                )
        else:
            self.walk_keys(
                piece,
                pend,
                end,
                length,
                window,
                rt,
                warp,
                lane,
                group,
                head,
                tid,
                k,
                v,
                sk,
                sv,
                vt,
                indices,
                mma,
                the,
                aq,
                bk,
                bv,
                ck,
                cv,
                acc,
                maxima,
                sums,
                windowed=False,
            )

    @cute.jit
    def walk_keys(
        self,
        piece,
        pend,
        end,
        length,
        window,
        rt,
        warp,
        lane,
        group,
        head,
        tid,
        k,
        v,
        sk,
        sv,
        vt,
        indices,
        mma,
        the,
        aq,
        bk,
        bv,
        ck,
        cv,
        acc,
        maxima,
        sums,
        windowed: cutlass.Constexpr,
    ):
        # Keys [piece, pend) of this split; the online softmax state carries over.
        for stage in cutlass.range_constexpr(self.stages - 1):
            self.gather(
                sk,
                sv,
                k,
                v,
                indices,
                stage * self.kt,
                Int32(pend - piece),
                head,
                tid,
                stage,
                copy_v=not self.delayed_v,
            )
            if const_expr(self.delayed_v):
                self.gather(
                    sk,
                    sv,
                    k,
                    v,
                    indices,
                    stage * self.kt,
                    Int32(pend - piece),
                    head,
                    tid,
                    stage,
                    copy_k=False,
                )
        cute.arch.cp_async_wait_group(self.wait_groups)
        cute.arch.barrier()
        buffer = Int32(0)
        # Piece-local offsets fit Int32 (at most chunk_cap plus ring lookahead).
        for local in range(Int32(0), Int32(pend - piece), self.kt):
            # Absolute Int64 keys, as in 0.1.12: Int32 request-relative distances
            # moved the CTA-uniform tile math off the uniform datapath (legacy
            # kt32 kernel: 30 vs 285 uniform instructions, 1-2% slower on RTX 5090).
            first = Int64(piece) + local
            # Commit even empty tail tiles so wait_group has a fixed distance.
            self.gather(
                sk,
                sv,
                k,
                v,
                indices,
                local + (self.stages - 1) * self.kt,
                Int32(pend - piece),
                head,
                tid,
                (buffer + self.stages - 1) % self.stages,
                copy_v=not self.delayed_v,
            )
            score = cute.make_rmem_tensor(the.partition_shape_C((self.rows, self.nk)), Float32)
            score.fill(0)
            cute.copy(ck, ck.partition_S(sk[None, None, buffer, group]), ck.retile(bk))
            for inner in cutlass.range_constexpr(self.dim // 16):
                cute.gemm(mma, score, aq[None, None, inner], bk[None, None, inner], score)
            for rr in cutlass.range_constexpr(2):
                packed = rt * self.rows + warp * 16 + lane // 4 + rr * 8
                position = Int64(length) - self.query + packed // self.group
                maximum = Float32(-float("inf"))
                for n in cutlass.range_constexpr(self.nk // 8):
                    for pair in cutlass.range_constexpr(2):
                        key = first + group * self.nk + n * 8 + lane % 4 * 2 + pair
                        valid = (key < end) & (packed < self.query * self.group)
                        if const_expr(windowed):
                            valid = valid & (position - key < window)
                            # key <= position (causal) already implies key - position < W.
                            if const_expr(not self.causal):
                                valid = valid & (key - position < window)
                        if const_expr(self.causal):
                            valid = valid & (key <= position)
                        value = Float32(-float("inf"))
                        if valid:
                            value = score[rr * 2 + pair, 0, n] * self.scale
                        score[rr * 2 + pair, 0, n] = value
                        maximum = max(maximum, value)
                maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1))
                maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 2))
                maximum = max(maximum, maxima[rr])
                safe_max = Float32(0)
                if maximum != -float("inf"):
                    safe_max = maximum
                rescale = cute.math.exp2(maxima[rr] - safe_max)
                total = Float32(0)
                for n in cutlass.range_constexpr(self.nk // 8):
                    for pair in cutlass.range_constexpr(2):
                        probability = cute.math.exp2(score[rr * 2 + pair, 0, n] - safe_max)
                        score[rr * 2 + pair, 0, n] = probability
                        total += probability
                total += cute.arch.shuffle_sync_bfly(total, 1)
                total += cute.arch.shuffle_sync_bfly(total, 2)
                sums[rr] = sums[rr] * rescale + total
                maxima[rr] = maximum
                for n in cutlass.range_constexpr(self.dim // 8):
                    for pair in cutlass.range_constexpr(2):
                        acc[rr * 2 + pair, 0, n] *= rescale
            prob = cute.make_rmem_tensor((8, 1, self.nk // 16), self.dtype)
            for inner in cutlass.range_constexpr(self.nk // 16):
                for value in cutlass.range_constexpr(8):
                    prob[value, 0, inner] = score[value % 4, 0, inner * 2 + value // 4].to(
                        self.dtype
                    )
            vbuffer = (local // self.kt) % self.v_stages
            cute.copy(cv, cv.partition_S(vt[None, None, vbuffer, group]), cv.retile(bv))
            if const_expr(self.delayed_v):
                # Every warp has read V before the shorter V ring reuses this slot.
                cute.arch.barrier()
                self.gather(
                    sk,
                    sv,
                    k,
                    v,
                    indices,
                    local + (self.stages - 1) * self.kt,
                    Int32(pend - piece),
                    head,
                    tid,
                    buffer,
                    copy_k=False,
                )
            for inner in cutlass.range_constexpr(self.nk // 16):
                cute.gemm(mma, acc, prob[None, None, inner], bv[None, None, inner], acc)
            cute.arch.cp_async_wait_group(self.wait_groups)
            cute.arch.barrier()
            buffer = (buffer + 1) % self.stages
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

    @cute.jit
    def stage_indices(self, indices, idx, start, first, last, tid, table_data=None):
        # One piece of at most chunk_cap token indices, piece-relative in shared memory.
        live = Int64(last)
        if const_expr(self.table):
            # Table keys below the live prefix copy like the CSR path; a per-key
            # copy/store branch cost 1-2% per layer at ctx 100 (paired chains).
            prefix, guard, maximum = table_data
            live = min(live, Int64(prefix))
        for part in cutlass.range_constexpr((self.chunk_cap + self.threads - 1) // self.threads):
            key = tid + part * self.threads
            if key < self.chunk_cap and Int64(first) + key < live:
                cute.arch.inline_ptx(
                    "cp.async.ca.shared.global [{$r0}], [{$r1}], 4;",
                    read_only_args=[
                        (indices.iterator + key).toint().to(Int32),
                        pointer(idx.iterator, Int64(start) + first + key, 4).toint(),
                    ],
                )
        if const_expr(self.table):
            # Guard keys: done rows and keys at or past M (CTA-uniform branch).
            if Int64(last) > live:
                for part in cutlass.range_constexpr(
                    (self.chunk_cap + self.threads - 1) // self.threads
                ):
                    key = tid + part * self.threads
                    j = Int64(first) + key
                    if key < self.chunk_cap and j < last and j >= live:
                        offset = Int32(j)
                        if prefix != 0:
                            offset = min(Int32(j) - maximum, self.query - 1)
                        indices[key] = guard + offset

    @cute.jit
    def gather(
        self,
        sk,
        sv,
        k,
        v,
        indices,
        first,
        end,
        head,
        tid,
        buffer,
        copy_k: cutlass.Constexpr = True,
        copy_v: cutlass.Constexpr = True,
    ):
        for part in cutlass.range_constexpr(self.index_parts):
            segment = tid + part * self.threads
            if segment < self.kt * (self.dim // 8):
                key, dim = segment // (self.dim // 8), segment % (self.dim // 8) * 8
                token = Int32(0)
                if first + key < end:
                    # Each D/8-lane segment broadcasts one shared-memory word.
                    token = indices[first + key]
                offset = (Int64(token) * self.heads + head) * self.dim + dim
                if const_expr(copy_k):
                    async_row(
                        sk.iterator + sk.layout((key % self.nk, dim, buffer, key // self.nk)),
                        k,
                        offset,
                        first + key < end,
                    )
                if const_expr(copy_v):
                    async_row(
                        sv.iterator
                        + sv.layout(
                            (key % self.nk, dim, (first // self.kt) % self.v_stages, key // self.nk)
                        ),
                        v,
                        offset,
                        first + key < end,
                    )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def store_tile(self, acc, arena, target, rt, tid, direct: cutlass.Constexpr):
        dtype = self.dtype if const_expr(direct) else Float32
        columns = const_expr(self.dim if direct else self.epilogue_cols)
        width = const_expr(8 if direct else 4)
        staged = cute.make_tensor(
            cute.recast_ptr(arena.iterator, dtype=dtype),
            cute.make_layout((self.rows, columns), stride=(columns, 1)),
        )
        load = cute.make_copy_atom(cute.nvgpu.CopyS2ROp(), dtype, num_bits_per_copy=128)
        store = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), dtype, num_bits_per_copy=128)
        fragment = cute.make_rmem_tensor((width,), dtype)
        row_step = const_expr(self.threads * width // columns)
        row0, dim0 = tid * width // columns, tid * width % columns
        source_base = pointer(staged.iterator, tid * width)
        if const_expr(direct):
            packed0 = rt * self.rows + row0
            target_base = pointer(
                target,
                (Int64(packed0 // self.group) * self.heads * self.group + packed0 % self.group)
                * self.dim
                + dim0,
            )
        else:
            target_base = pointer(target, row0 * self.dim + dim0)
        for half in cutlass.range_constexpr(self.dim // columns):
            if tid < self.mma_threads:
                for rr in cutlass.range_constexpr(2):
                    row = tid // 32 * 16 + tid % 32 // 4 + rr * 8
                    for n in cutlass.range_constexpr(columns // 8):
                        for pair in cutlass.range_constexpr(2):
                            dim = n * 8 + tid % 4 * 2 + pair
                            staged[row, dim] = acc[rr * 2 + pair, 0, half * (columns // 8) + n].to(
                                dtype
                            )
            cute.arch.barrier()
            # Consecutive lanes own aligned vectors; the workspace stays row-major.
            for part in cutlass.range_constexpr(self.rows * columns // (width * self.threads)):
                source = cute.make_tensor(
                    source_base + part * self.threads * width, cute.make_layout(width)
                )
                cute.copy(load, source, fragment)
                if const_expr(direct):
                    packed = rt * self.rows + row0 + part * row_step
                    if packed < self.query * self.group:
                        if const_expr(row_step % self.group == 0):
                            dest = cute.make_tensor(
                                target_base + part * row_step * self.heads * self.dim,
                                cute.make_layout(width),
                            )
                        else:
                            offset = (
                                Int64(packed // self.group) * self.heads * self.group
                                + packed % self.group
                            ) * self.dim + dim0
                            dest = cute.make_tensor(
                                pointer(target, offset), cute.make_layout(width)
                            )
                        cute.copy(store, fragment, dest)
                else:
                    dest = cute.make_tensor(
                        target_base + part * row_step * self.dim + half * columns,
                        cute.make_layout(width),
                    )
                    cute.copy(store, fragment, dest)
            if const_expr(half + 1 < self.dim // columns):
                cute.arch.barrier()

    @cute.jit
    def finish(self, work, out, counters, arena, batch, b, head, rt, tid, count, split, prefix):
        if const_expr(self.distributed_merge):
            self.finish_distributed(
                work, out, counters, arena, batch, b, head, rt, tid, count, split, prefix
            )
        else:
            self.finish_tree(
                work, out, counters, arena, batch, b, head, rt, tid, count, split, prefix
            )

    @cute.jit
    def finish_distributed(
        self, work, out, counters, arena, batch, b, head, rt, tid, count, split, prefix
    ):
        # Whole-CTA arrival wait includes thread zero's polling and the broadcast barrier.
        self.trace_begin("attn.merge_wait")
        # bar.sync gathers every partial/LSE store; thread zero's cumulative
        # GPU release publishes them all (no per-writer GPU fence is needed).
        cute.arch.barrier()
        counter = ((Int64(b) * self.heads + head) * self.rt + rt) * 2
        if tid == 0:
            address = pointer(counters.iterator, counter, 4).toint()
            flag = pointer(counters.iterator, counter + 1, 4).toint()
            # Acquire orders this generation snapshot before the arrival atomic.
            gen = cute.arch.inline_ptx(
                "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];",
                write_only_types=[Int32],
                read_only_args=[flag],
            )
            # Acquire collects prior arrivals so the last arriver can publish
            # every CTA's payload and proceed directly without polling.
            old = cute.arch.inline_ptx(
                "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], 1;",
                write_only_types=[Int32],
                read_only_args=[address],
            )
            if old == count - 1:
                # The next launch follows grid completion (with PDL it executes
                # griddepcontrol.wait before counters), so it cannot see stale arrival.
                cute.arch.inline_ptx(
                    "st.relaxed.gpu.global.u32 [{$r0}], 0;",
                    read_only_args=[address],
                )
                cute.arch.inline_ptx(
                    "{ .reg .u32 next; add.u32 next, {$r1}, 1; "
                    "st.release.gpu.global.u32 [{$r0}], next; }",
                    read_only_args=[flag, gen],
                )
            else:
                observed = gen
                polls = Int32(0)
                # Equality on 32-bit generations also handles wrap-around.
                while observed == gen:
                    if polls < 64:
                        polls += 1
                    else:
                        cute.arch.inline_ptx("nanosleep.u32 32;")
                    observed = cute.arch.inline_ptx(
                        "ld.acquire.gpu.global.u32 {$w0}, [{$r0}];",
                        write_only_types=[Int32],
                        read_only_args=[flag],
                    )
        # Cooperative launch guarantees grid co-residency. The acquire
        # and this barrier publish every split's payload to all merging threads.
        cute.arch.barrier()
        self.trace_end()
        self.merge_tile(work, out, arena, batch, b, head, rt, tid, count, split, prefix)

    @cute.jit
    def merge_tile(self, work, out, arena, batch, b, head, rt, tid, count, split, prefix):
        valid_rows = min(self.rows, self.query * self.group - rt * self.rows)
        row_begin = valid_rows * split // count
        row_end = valid_rows * (split + 1) // count
        owned = row_end - row_begin
        if owned > 0:
            # Flatten (source split, owned row, 16-byte vector) over the CTA.
            # The arena bounds each batch to <=16 vector loads/thread; full
            # batches have >=8. Small payloads fit entirely in a single batch.
            capacity = const_expr(self.merge_capacity)
            staged = cute.make_tensor(
                cute.recast_ptr(arena.iterator, dtype=Float32),
                cute.make_layout((self.dim, capacity), stride=(1, self.dim)),
            )
            stats = cute.make_tensor(
                staged.iterator + capacity * self.dim,
                cute.make_layout((4, capacity), stride=(1, 4)),
            )
            base = self.slot_base(b, head, rt, prefix) * self.rows
            lse_base = self.slot_count(batch) * self.rows * self.dim
            warps = const_expr(self.merge_threads // 32)
            row_parts = const_expr(((self.rows + 1) // 2 + warps - 1) // warps)
            result = cute.make_rmem_tensor((self.dim // 32, row_parts), Float32)
            maxima = cute.make_rmem_tensor((row_parts,), Float32)
            totals = cute.make_rmem_tensor((row_parts,), Float32)
            result.fill(0)
            maxima.fill(-float("inf"))
            totals.fill(0)
            batch_splits = capacity // owned
            for first in range(0, count, batch_splits):
                sources = min(batch_splits, count - first)
                payload_rows = sources * owned
                # Issue ALL partial and LSE copies before waiting or consuming.
                if tid < self.merge_threads:
                    for vector in range(tid, payload_rows * (self.dim // 4), self.merge_threads):
                        item, dim = vector // (self.dim // 4), vector % (self.dim // 4) * 4
                        source_split = first + item // owned
                        row = row_begin + item % owned
                        async_row(
                            staged.iterator + item * self.dim + dim,
                            work,
                            (base + source_split * self.rows + row) * self.dim + dim,
                            True,
                        )
                    for item in range(tid, payload_rows, self.merge_threads):
                        source_split = first + item // owned
                        row = row_begin + item % owned
                        # Aligned LSE vector; only this row's component is consumed.
                        async_row(
                            stats.iterator + item * 4,
                            work,
                            lse_base + base + source_split * self.rows + row // 4 * 4,
                            True,
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                for part in cutlass.range_constexpr(row_parts):
                    local_row = tid // 32 + part * warps
                    row = row_begin + local_row
                    if tid < self.merge_threads and local_row < owned:
                        maximum = maxima[part]
                        for inner in range(sources):
                            maximum = max(maximum, stats[row % 4, inner * owned + local_row])
                        safe_max = Float32(0)
                        if maximum != -float("inf"):
                            safe_max = maximum
                        rescale = cute.math.exp2(maxima[part] - safe_max)
                        totals[part] *= rescale
                        for d in cutlass.range_constexpr(self.dim // 32):
                            result[d, part] *= rescale
                        for inner in range(sources):
                            item = inner * owned + local_row
                            weight = cute.math.exp2(stats[row % 4, item] - safe_max)
                            totals[part] += weight
                            for d in cutlass.range_constexpr(self.dim // 32):
                                result[d, part] += (
                                    weight * staged[tid % 32 * (self.dim // 32) + d, item]
                                )
                        maxima[part] = maximum
                # Protect the shared payload until all row owners finish reading.
                cute.arch.barrier()
            for part in cutlass.range_constexpr(row_parts):
                local_row = tid // 32 + part * warps
                if tid < self.merge_threads and local_row < owned:
                    inverse = Float32(0)
                    if totals[part] > 0:
                        inverse = 1.0 / totals[part]
                    converted = cute.make_rmem_tensor((self.dim // 32,), self.dtype)
                    for d in cutlass.range_constexpr(self.dim // 32):
                        converted[d] = (result[d, part] * inverse).to(self.dtype)
                    packed = rt * self.rows + row_begin + local_row
                    offset = (
                        (Int64(b) * self.query + packed // self.group) * self.heads * self.group
                        + head * self.group
                        + packed % self.group
                    ) * self.dim + tid % 32 * (self.dim // 32)
                    target = cute.make_tensor(
                        pointer(out, offset, self.dim // 16), cute.make_layout(self.dim // 32)
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=self.dim // 2
                        ),
                        converted,
                        target,
                    )

    @cute.jit
    def finish_tree(
        self, work, out, counters, arena, batch, b, head, rt, tid, count, split, prefix
    ):
        # Attention's accumulator is dead. Every level reuses the same arena.
        winner = cute.make_tensor(
            cute.recast_ptr(arena.iterator, dtype=Int32)
            + self.merge_fan_in * self.rows * (self.merge_cols + 1),
            cute.make_layout(1),
        )
        tile = (Int64(b) * self.heads + head) * self.rt + rt
        active = Int32(1)
        node, remaining = Int32(split), Int32(count)
        for level in cutlass.range_constexpr(len(self.merge_levels)):
            if active != 0 and remaining > 1:
                group = node // self.merge_fan_in
                first = group * self.merge_fan_in
                sources = min(self.merge_fan_in, remaining - first)
                # Gather all partial/LSE stores before thread zero publishes them.
                cute.arch.barrier()
                if tid == 0:
                    address = pointer(
                        counters.iterator,
                        tile * sum(self.merge_levels) + self.counter_offsets[level] + group,
                        4,
                    ).toint()
                    # Public PTX: cumulative GPU release plus acquire of prior arrivals.
                    old = cute.arch.inline_ptx(
                        "atom.acq_rel.gpu.global.add.s32 {$w0}, [{$r0}], 1;",
                        write_only_types=[Int32],
                        read_only_args=[address],
                    )
                    winner[0] = Int32(old == sources - 1)
                    if old == sources - 1:
                        cute.arch.inline_ptx(
                            "st.relaxed.gpu.global.u32 [{$r0}], 0;",
                            read_only_args=[address],
                        )
                # Broadcast the winner and the acquired payload to the whole CTA.
                cute.arch.barrier()
                active = winner[0]
                if active != 0:
                    self.merge_group(
                        work,
                        out,
                        arena,
                        batch,
                        b,
                        head,
                        rt,
                        tid,
                        first,
                        sources,
                        remaining <= self.merge_fan_in,
                        prefix,
                        self.merge_fan_in**level,
                    )
                node = group
                remaining = (remaining + self.merge_fan_in - 1) // self.merge_fan_in
        # Non-winners leave immediately; there is no inter-CTA wait or polling.

    @cute.jit
    def merge_group(
        self,
        work,
        out,
        arena,
        batch,
        b,
        head,
        rt,
        tid,
        first,
        sources,
        final,
        prefix,
        stride: cutlass.Constexpr,
    ):
        for column in cutlass.range_constexpr(self.dim // self.merge_cols):
            self.merge_columns(
                work,
                out,
                arena,
                batch,
                b,
                head,
                rt,
                tid,
                first,
                sources,
                final,
                prefix,
                stride,
                column * self.merge_cols,
            )

    @cute.jit
    def merge_columns(
        self,
        work,
        out,
        arena,
        batch,
        b,
        head,
        rt,
        tid,
        first,
        sources,
        final,
        prefix,
        stride: cutlass.Constexpr,
        column: cutlass.Constexpr,
    ):
        staged = cute.make_tensor(
            cute.recast_ptr(arena.iterator, dtype=Float32),
            cute.make_layout(
                (self.merge_fan_in, self.rows, self.merge_cols),
                stride=(self.rows * self.merge_cols, self.merge_cols, 1),
            ),
        )
        stats = cute.make_tensor(
            staged.iterator + self.merge_fan_in * self.rows * self.merge_cols,
            cute.make_layout((self.merge_fan_in, self.rows), stride=(self.rows, 1)),
        )
        base = self.slot_base(b, head, rt, prefix) * self.rows
        lse_base = self.slot_count(batch) * self.rows * self.dim
        # All source partials AND packed LSE vectors issue before ONE wait.
        # cp.async.cg bypasses potentially stale L1 lines from the CTA epilogue.
        for vector in range(tid, sources * self.rows * (self.merge_cols // 4), self.threads):
            item, dim = vector // (self.merge_cols // 4), vector % (self.merge_cols // 4) * 4
            source, row = item // self.rows, item % self.rows
            async_row(
                staged.iterator + item * self.merge_cols + dim,
                work,
                (base + (first + source) * stride * self.rows + row) * self.dim + column + dim,
                True,
            )
        for vector in range(tid, sources * self.rows // 4, self.threads):
            source, row = vector * 4 // self.rows, vector * 4 % self.rows
            async_row(
                stats.iterator + vector * 4,
                work,
                lse_base + base + (first + source) * stride * self.rows + row,
                True,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        # Each lane owns four FP32 values within a row strip.
        # Store intermediate padded rows as well (later levels load full tiles).
        lanes = const_expr(self.strip // 4)
        strips = const_expr(self.merge_cols // self.strip)
        for item in range(tid // lanes, self.rows * strips, self.threads // lanes):
            row = item // strips
            dim = item % strips * self.strip + tid % lanes * 4
            maximum = Float32(-float("inf"))
            for source in range(sources):
                maximum = max(maximum, stats[source, row])
            safe_max = Float32(0)
            if maximum != -float("inf"):
                safe_max = maximum
            total = Float32(0)
            result = cute.make_rmem_tensor((4,), Float32)
            result.fill(0)
            for source in range(sources):
                weight = cute.math.exp2(stats[source, row] - safe_max)
                total += weight
                for d in cutlass.range_constexpr(4):
                    result[d] += weight * staged[source, row, dim + d]
            inverse = Float32(0)
            lse = Float32(-float("inf"))
            if total > 0:
                inverse = 1.0 / total
                lse = maximum + cute.math.log2(total)
            for d in cutlass.range_constexpr(4):
                result[d] *= inverse
            if final:
                packed = rt * self.rows + row
                if packed < self.query * self.group:
                    converted = cute.make_rmem_tensor((4,), self.dtype)
                    converted.store(result.load().to(self.dtype))
                    offset = (
                        (
                            (Int64(b) * self.query + packed // self.group) * self.heads * self.group
                            + head * self.group
                            + packed % self.group
                        )
                        * self.dim
                        + column
                        + dim
                    )
                    target = cute.make_tensor(pointer(out, offset, 8), cute.make_layout(4))
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=64
                        ),
                        converted,
                        target,
                    )
            else:
                dest = base + first * stride * self.rows + row
                target = cute.make_tensor(
                    pointer(work, dest * self.dim + column + dim), cute.make_layout(4)
                )
                cute.copy(
                    cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), Float32, num_bits_per_copy=128),
                    result,
                    target,
                )
                # Keep the original source weights until every column is consumed.
                # Only the first strip publishes the row's LSE.
                if const_expr(column + self.merge_cols == self.dim):
                    if dim == 0:
                        work[lse_base + dest] = lse
        # Retire every reader before the next level overwrites this shared arena.
        cute.arch.barrier()

    @cute.kernel
    def kernel(
        self,
        slots: cute.Tensor,
        ptr: cute.Tensor,
        idx: cute.Tensor,
        q: cute.Pointer,
        k: cute.Pointer,
        v: cute.Pointer,
        out: cute.Pointer,
        work: cute.Pointer,
        counters: cute.Tensor,
        qp: cute.Tensor,
        pp: cute.Tensor,
        pi: cute.Tensor,
        last: cute.Tensor,
        fused: Int32,
        window: Int32,
        mma: cute.TiledMma,
        table_data=None,
    ):
        self.trace_begin("attn.prologue")
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            if const_expr(self.early_trigger):
                # The dependent launches once every CTA here is resident and past
                # its wait; it still waits for this grid's completion before reads.
                cute.arch.griddepcontrol_launch_dependents()
        split, hr, b = cute.arch.block_idx()
        if const_expr(self.head_major):
            hr, split, b = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        # per: this request's CTAs per row tile; idle spread CTAs (split >= per) are not live.
        per, prefix, live = self.splits, 0, True
        spread_lo, spread_chunk = Int32(0), Int64(0)
        if const_expr(self.spread):
            b, split, prefix, spread_lo, spread_chunk, per = self.spread_schedule(
                slots, ptr, window, split, tid
            )
            live = split < per
        cta = Int64(hr) * per + split
        head, rt = hr // self.rt, hr % self.rt
        lane, warp = tid % 32, (tid % self.mma_threads) // 32
        group = tid // self.mma_threads
        inactive = False
        if const_expr(not self.table):
            inactive = slots[b] == -1
        if inactive:
            if const_expr(not self.table):
                if fused != 0:
                    if live:
                        self.publish(slots, ptr, idx, qp, pp, pi, last, b, cta, tid, per)
            self.trace_end()
            self.trace_begin("attn.partial")
            if const_expr(self.splits > 1 and not self.inline_merge):
                if split == 0 and hr == 0 and tid == 0:
                    work[self.count_index(slots.shape[0], b)] = Float32(0)
            if const_expr(self.splits == 1 or self.inline_merge):
                output_base = pointer(
                    out, (Int64(b) * self.query * self.heads + head) * self.group * self.dim
                )
                zeros = cute.make_rmem_tensor((8,), self.dtype)
                zeros.fill(0)
                for item in cutlass.range_constexpr(self.rows * (self.dim // 8) // self.threads):
                    element = (tid + item * self.threads) * 8
                    row, dim = rt * self.rows + element // self.dim, element % self.dim
                    if split == 0 and row < self.query * self.group:
                        offset = (
                            Int64(row // self.group) * self.heads * self.group + row % self.group
                        ) * self.dim + dim
                        target = cute.make_tensor(pointer(output_base, offset), cute.make_layout(8))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=128
                            ),
                            zeros,
                            target,
                        )
            self.trace_end()
        else:
            stage_data = None
            if const_expr(self.table):
                stride, guard, table_maximum = table_data
                request = slots[b]
                key = request
                if const_expr(self.row_metadata):
                    key = b
                frontier = ptr[key]
                length, prefix = Int32(self.query), Int32(0)
                # Narrow only after saturation; even INT64_MIN/MAX never enter arithmetic.
                if (qp[key] == 0) & (frontier >= 0) & (frontier < table_maximum):
                    length = Int32(frontier) + self.query
                    prefix = min(length, table_maximum)
                start = Int64(request) * stride
                stage_data = (prefix, guard + b * self.query, table_maximum)
            else:
                start = ptr[b]
                length = ptr[b + 1] - start
            lo, chunk, count = self.split_bounds(length, window)
            if const_expr(self.spread):
                lo, chunk, count = spread_lo, spread_chunk, per
            if const_expr(self.splits > 1 and not self.inline_merge):
                if split == 0 and hr == 0 and tid == 0:
                    work[self.count_index(slots.shape[0], b)] = Float32(count)
            begin = Int64(lo) + split * chunk
            end = min(length, begin + chunk)
            if begin < end:
                allocator = SmemAllocator()
                qlayout = cute.make_composed_layout(
                    cute.make_swizzle(3, 3, self.swizzle_shift),
                    0,
                    cute.make_layout((self.rows, self.dim), stride=(self.dim, 1)),
                )
                kvlayout = cute.make_composed_layout(
                    cute.make_swizzle(3, 3, self.swizzle_shift),
                    0,
                    cute.make_layout(
                        (self.nk, self.dim, self.stages, self.groups),
                        stride=(self.dim, 1, self.kt * self.dim, self.nk * self.dim),
                    ),
                )
                arena = allocator.allocate_tensor(
                    self.dtype, cute.make_layout(self.smem_elements), byte_alignment=128
                )
                sq = cute.make_tensor(arena.iterator, qlayout)
                sk = cute.make_tensor(arena.iterator, kvlayout)
                vlayout = self.v_layout()
                sv = cute.make_tensor(arena.iterator + self.kv_elements, vlayout)
                indices = allocator.allocate_tensor(
                    Int32, cute.make_layout(self.chunk_cap), byte_alignment=128
                )
                self.stage_indices(indices, idx, start, begin, end, tid, stage_data)
                for part in cutlass.range_constexpr(self.rows * (self.dim // 8) // self.threads):
                    segment = tid + part * self.threads
                    row, dim = segment // (self.dim // 8), segment % (self.dim // 8) * 8
                    packed = rt * self.rows + row
                    if const_expr(self.q_row_stride == self.heads * self.group * self.dim):
                        offset = (
                            (Int64(b) * self.query + packed // self.group) * self.heads * self.group
                            + head * self.group
                            + packed % self.group
                        ) * self.dim + dim
                    else:
                        offset = (
                            (Int64(b) * self.query + packed // self.group) * self.q_row_stride
                            + (head * self.group + packed % self.group) * self.dim
                            + dim
                        )
                    async_row(
                        sq.iterator + sq.layout((row, dim)),
                        q,
                        offset,
                        packed < self.query * self.group,
                    )
                cute.arch.cp_async_commit_group()
                # Publishing overlaps the index and Q loads instead of delaying them.
                if const_expr(not self.table):
                    if fused != 0:
                        self.publish(slots, ptr, idx, qp, pp, pi, last, b, cta, tid, per)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                the = mma.get_slice(tid % self.mma_threads)
                aq = the.make_fragment_A(the.partition_A(sq))
                bk = the.make_fragment_B(the.partition_B(sk[None, None, 0, group]))
                vt = cute.make_tensor(
                    sv.iterator,
                    self.v_layout(transpose=True),
                )
                bv = the.make_fragment_B(the.partition_B(vt[None, None, 0, group]))
                load = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
                )
                load_t = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
                )
                cq = cute.make_tiled_copy_A(load, mma).get_slice(tid % self.mma_threads)
                ck = cute.make_tiled_copy_B(load, mma).get_slice(tid % self.mma_threads)
                cv = cute.make_tiled_copy_B(load_t, mma).get_slice(tid % self.mma_threads)
                cute.copy(cq, cq.partition_S(sq), cq.retile(aq))
                # All groups own Q in registers before K/V reuse its shared storage.
                cute.arch.barrier()
                acc = cute.make_rmem_tensor(the.partition_shape_C((self.rows, self.dim)), Float32)
                acc.fill(0)
                maxima = cute.make_rmem_tensor((2,), Float32)
                sums = cute.make_rmem_tensor((2,), Float32)
                maxima.fill(-float("inf"))
                sums.fill(0)
                self.trace_end()
                self.trace_begin("attn.walk")
                if const_expr(self.pieces):
                    # A split walks its chunk in pieces of at most chunk_cap keys, so the
                    # split count follows occupancy, not capacity. Each piece re-stages
                    # its indices after the previous piece has drained the ring.
                    for piece in range(begin, end, self.chunk_cap):
                        pend = min(end, piece + self.chunk_cap)
                        if piece != begin:
                            self.stage_indices(indices, idx, start, piece, pend, tid, stage_data)
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.barrier()
                        self.walk(
                            piece,
                            pend,
                            end,
                            length,
                            window,
                            rt,
                            warp,
                            lane,
                            group,
                            head,
                            tid,
                            k,
                            v,
                            sk,
                            sv,
                            vt,
                            indices,
                            mma,
                            the,
                            aq,
                            bk,
                            bv,
                            ck,
                            cv,
                            acc,
                            maxima,
                            sums,
                        )
                else:
                    # One piece always covers the chunk (decided at construction).
                    self.walk(
                        begin,
                        end,
                        end,
                        length,
                        window,
                        rt,
                        warp,
                        lane,
                        group,
                        head,
                        tid,
                        k,
                        v,
                        sk,
                        sv,
                        vt,
                        indices,
                        mma,
                        the,
                        aq,
                        bk,
                        bv,
                        ck,
                        cv,
                        acc,
                        maxima,
                        sums,
                    )
                self.trace_end()
                self.trace_begin("attn.partial")
                if const_expr(self.groups == 2):
                    partial = cute.make_tensor(
                        cute.recast_ptr(sk.iterator, dtype=Float32),
                        # Adjacent lanes use adjacent banks during the warp-pair combine.
                        cute.make_layout(
                            (self.dim // 2, self.mma_threads), stride=(self.mma_threads, 1)
                        ),
                    )
                    stats = cute.make_tensor(
                        cute.recast_ptr(arena.iterator + self.combine_elements, dtype=Float32),
                        cute.make_layout((self.rows, 2), stride=(2, 1)),
                    )
                    if group == 1:
                        for rr in cutlass.range_constexpr(2):
                            row = warp * 16 + lane // 4 + rr * 8
                            if lane % 4 == 0:
                                stats[row, 0], stats[row, 1] = maxima[rr], sums[rr]
                            for n in cutlass.range_constexpr(self.dim // 8):
                                for pair in cutlass.range_constexpr(2):
                                    partial[
                                        rr * (self.dim // 4) + n * 2 + pair, tid % self.mma_threads
                                    ] = acc[rr * 2 + pair, 0, n]
                    cute.arch.barrier()
                    if group == 0:
                        for rr in cutlass.range_constexpr(2):
                            row = warp * 16 + lane // 4 + rr * 8
                            other_max, other_sum = stats[row, 0], stats[row, 1]
                            maximum = max(maxima[rr], other_max)
                            safe_max = Float32(0)
                            if maximum != -float("inf"):
                                safe_max = maximum
                            left = cute.math.exp2(maxima[rr] - safe_max)
                            right = cute.math.exp2(other_max - safe_max)
                            sums[rr] = sums[rr] * left + other_sum * right
                            maxima[rr] = maximum
                            for n in cutlass.range_constexpr(self.dim // 8):
                                for pair in cutlass.range_constexpr(2):
                                    acc[rr * 2 + pair, 0, n] = (
                                        acc[rr * 2 + pair, 0, n] * left
                                        + partial[
                                            rr * (self.dim // 4) + n * 2 + pair,
                                            tid % self.mma_threads,
                                        ]
                                        * right
                                    )
                base = (self.slot_base(b, head, rt, prefix) + split) * self.rows
                if group == 0:
                    for rr in cutlass.range_constexpr(2):
                        row = warp * 16 + lane // 4 + rr * 8
                        inverse = Float32(0)
                        lse = Float32(-float("inf"))
                        if sums[rr] > 0:
                            inverse = 1.0 / sums[rr]
                            lse = maxima[rr] + cute.math.log2(sums[rr])
                        for n in cutlass.range_constexpr(self.dim // 8):
                            for pair in cutlass.range_constexpr(2):
                                acc[rr * 2 + pair, 0, n] *= inverse
                        if const_expr(self.splits > 1):
                            if count > 1 and lane % 4 == 0:
                                lse_base = self.slot_count(slots.shape[0]) * self.rows * self.dim
                                work[lse_base + base + row] = lse
                if const_expr(self.groups == 2):
                    # All combine reads must finish before the epilogue reuses the arena.
                    cute.arch.barrier()
                if count == 1:
                    output_base = pointer(
                        out, (Int64(b) * self.query * self.heads + head) * self.group * self.dim
                    )
                    self.store_tile(acc, arena, output_base, rt, tid, direct=True)
                    self.trace_end()
                else:
                    if const_expr(self.splits > 1):
                        work_base = pointer(work, base * self.dim)
                        self.store_tile(acc, arena, work_base, rt, tid, direct=False)
                        self.trace_end()
                        if const_expr(self.inline_merge):
                            self.trace_begin("attn.merge")
                            self.finish(
                                work,
                                out,
                                counters,
                                arena,
                                slots.shape[0],
                                b,
                                head,
                                rt,
                                tid,
                                count,
                                split,
                                prefix,
                            )
                            self.trace_end()
                    else:
                        self.trace_end()
            else:
                if const_expr(not self.table):
                    if fused != 0:
                        if live:
                            self.publish(slots, ptr, idx, qp, pp, pi, last, b, cta, tid, per)
                self.trace_end()
        self.trace_begin("attn.epilogue")
        if const_expr(self.use_pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        self.trace_end()


class Merge(Attention):
    @cute.jit
    def __call__(
        self,
        slots: cute.Tensor,
        ptr: cute.Tensor,
        work: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel_merge(slots, ptr, work.iterator, out.iterator).launch(
            grid=(
                (Int64(slots.shape[0]) * self.heads * self.query * self.group + 3) // 4,
                self.dim // self.strip,
                1,
            ),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel_merge(
        self, slots: cute.Tensor, ptr: cute.Tensor, work: cute.Pointer, out: cute.Pointer
    ):
        self.trace_begin("merge.all")
        cute.arch.griddepcontrol_wait()
        tile, column, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        item = Int64(tile) * 4 + tid // 32
        packed = Int32(item % (self.query * self.group))
        head = Int32(item // (self.query * self.group) % self.heads)
        b = Int32(item // (self.query * self.group * self.heads))
        dim = lane * (self.strip // 32)
        if const_expr(self.dim > self.strip):
            dim += column * self.strip
        row, rt = packed % self.rows, packed // self.rows
        if b < slots.shape[0]:
            base = ((Int64(b) * self.heads + head) * self.rt + rt) * self.splits * self.rows + row
            lse_base = (
                Int64(slots.shape[0]) * self.heads * self.rt * self.splits * self.rows * self.dim
            )
            # Attention publishes the request's nonempty split count (0 when inactive),
            # so no request metadata is read here. Every load below is independent of
            # the count: stale entries past it are in-bounds workspace and get zero weight.
            count = work[self.count_index(slots.shape[0], b)]
            weights = cute.make_rmem_tensor(((self.splits + 31) // 32,), Float32)
            for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                split = chunk * 32 + lane
                value = Float32(-float("inf"))
                if split < self.splits:
                    value = work[lse_base + base + split * self.rows]
                weights[chunk] = value
            loaded = cute.make_rmem_tensor((self.strip // 32, self.unroll), Float32)
            loaded.fill(0)
            for inner in cutlass.range_constexpr(self.unroll):
                source = cute.make_tensor(
                    pointer(work, (base + inner * self.rows) * self.dim + dim, self.strip // 8),
                    cute.make_layout(self.strip // 32),
                )
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyG2ROp(), Float32, num_bits_per_copy=self.strip
                )
                if const_expr(self.sparse):
                    # Large capacities: most calls fill few of the splits, so wait for
                    # the count rather than read every stale partial.
                    if Float32(inner) < count:
                        cute.copy(atom, source, loaded[None, inner])
                else:
                    cute.copy(atom, source, loaded[None, inner])
            # Split zero owns the final output when only one split is nonempty.
            if count != 1:
                result = cute.make_rmem_tensor((self.strip // 32,), Float32)
                result.fill(0)
                if count > 1:
                    maximum = Float32(-float("inf"))
                    for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                        if Float32(chunk * 32 + lane) >= count:
                            weights[chunk] = Float32(-float("inf"))
                        maximum = max(maximum, weights[chunk])
                    for shift in cutlass.range_constexpr(5):
                        maximum = max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << shift))
                    safe_max = Float32(0)
                    if maximum != -float("inf"):
                        safe_max = maximum
                    total = Float32(0)
                    for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                        weights[chunk] = cute.math.exp2(weights[chunk] - safe_max)
                        total += weights[chunk]
                    for shift in cutlass.range_constexpr(5):
                        total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
                    for inner in cutlass.range_constexpr(self.unroll):
                        # Zero weights past the count; select so stale values never mix in.
                        weight = cute.arch.shuffle_sync(weights[inner // 32], inner % 32)
                        if Float32(inner) < count:
                            for d in cutlass.range_constexpr(self.strip // 32):
                                result[d] += loaded[d, inner] * weight
                    for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                        for first in range(
                            max(self.unroll, chunk * 32), min((chunk + 1) * 32, self.splits), 4
                        ):
                            if Float32(first) < count:
                                # Issue four independent vector loads before consuming any of them.
                                values = cute.make_rmem_tensor((self.strip // 32, 4), Float32)
                                values.fill(0)
                                for inner in cutlass.range_constexpr(4):
                                    split = first + inner
                                    if Float32(split) < count:
                                        source = cute.make_tensor(
                                            pointer(
                                                work,
                                                (base + split * self.rows) * self.dim + dim,
                                                self.strip // 8,
                                            ),
                                            cute.make_layout(self.strip // 32),
                                        )
                                        cute.copy(
                                            cute.make_copy_atom(
                                                cute.nvgpu.CopyG2ROp(),
                                                Float32,
                                                num_bits_per_copy=self.strip,
                                            ),
                                            source,
                                            values[None, inner],
                                        )
                                for inner in cutlass.range_constexpr(4):
                                    weight = cute.arch.shuffle_sync(
                                        weights[chunk], first - chunk * 32 + inner
                                    )
                                    for d in cutlass.range_constexpr(self.strip // 32):
                                        result[d] += values[d, inner] * weight
                    inverse = Float32(0)
                    if total > 0:
                        inverse = 1.0 / total
                    for d in cutlass.range_constexpr(self.strip // 32):
                        result[d] *= inverse
                converted = cute.make_rmem_tensor((self.strip // 32,), self.dtype)
                converted.store(result.load().to(self.dtype))
                offset = (
                    (Int64(b) * self.query + packed // self.group) * self.heads * self.group
                    + head * self.group
                    + packed % self.group
                ) * self.dim + dim
                target = cute.make_tensor(
                    pointer(out, offset, self.strip // 16), cute.make_layout(self.strip // 32)
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyR2GOp(), self.dtype, num_bits_per_copy=self.strip // 2
                    ),
                    converted,
                    target,
                )
        cute.arch.griddepcontrol_launch_dependents()
        self.trace_end()
