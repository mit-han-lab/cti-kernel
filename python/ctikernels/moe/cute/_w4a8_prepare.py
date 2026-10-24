# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CuTe route preparation and exact K32 Q0 scatter for fixed DSV4 geometry.

Design reference: FlashInfer 0.7.1 fused_moe/cute_dsl/blackwell_sm12x/moe_mxfp8_q0_route_triton.py

Small lists use two launches: stable ranks/reset, then chunked Q0/output reset.
Larger lists use reset, histogram, prefix, and fused assignment/Q0. Optional
separate assignment adds one launch but exposes more Q0 CTAs. No host reads of
route contents or device counters occur in prepare. Warm each specialization
before capture and use separate workspaces for concurrent calls/replays.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .._common import frozen, tensor, validate_quantized
from ._quant import _encoder
from ._runtime import require

# Local experts per rank: EP1 of DSV4 Flash (256) down to any EP split.
MAX_EXPERTS = 256


@dataclass(frozen=True)
class Config:
    small_rows: int = 256
    q0_chunk: int = 1024
    fuse_assignment: bool = True
    pdl: bool = True

    def __post_init__(self):
        if type(self.small_rows) is not int or not 0 <= self.small_rows <= 256:
            raise ValueError("small_rows must be an integer in [0,256]")
        if type(self.q0_chunk) is not int or self.q0_chunk not in (256, 512, 1024):
            raise ValueError("q0_chunk must be 256, 512 or 1024")
        if type(self.fuse_assignment) is not bool or type(self.pdl) is not bool:
            raise ValueError("fuse_assignment and pdl must be bool")


def carve(arena, layout):
    """Views of ``layout`` [(name, shape, dtype)] packed into the uint8 ``arena``, each at a
    256-byte boundary; with ``arena`` None, return the byte count instead."""
    views, offset = {}, 0
    for name, shape, dtype in layout:
        offset = -(-offset // 256) * 256
        size = dtype.itemsize
        for dim in shape:
            size *= dim
        if arena is not None:
            views[name] = arena[offset : offset + size].view(dtype).view(shape)
        offset += size
    return views if arena is not None else offset


class W4A8Workspace:
    """Allocate fixed capacity outside capture; mutable storage is stream-private.

    Only rows below num_rows are valid. Scale padding is reset to neutral bytes
    on every call; unused Q0/Q1/y data and forward row maps are undefined.
    slot_row [M,6] aliases pair_row; remote slots are reset to -1 every call.
    CPU allocation is supported for layout/reference tests. With ``arena`` (uint8, at
    least ``nbytes(m, combine)``), every tensor is a view into it.
    """

    @staticmethod
    def layout(m, combine, experts=128, inter=2048):
        rows = 6 * m
        scale_rows = (rows + 3 * experts) // 4 * 4
        i32, u8, f32 = torch.int32, torch.uint8, torch.float32
        return [
            ("q0", (rows, 4096), u8),
            ("q0_scales", (32, scale_rows), i32),
            ("q1", (rows, inter), u8),
            ("q1_scales", (inter // 128, scale_rows), i32),
            ("row_token", (rows,), i32),
            ("row_weight", (rows,), f32),
            ("row_pair", (rows,), i32),
            ("expert_offsets", (experts + 1,), i32),
            ("scale_offsets", (experts,), i32),
            ("num_rows", (1,), i32),
            ("pair_row", (rows,), i32),
            # Local expert per pair, written when PREP maps global ids.
            ("local_ids", (rows,), i32),
            # Per-route rows, each rounded once to BF16 (FI, SGLang and vLLM round per route
            # too); the slot-ordered combine sums them in FP32 and rounds once more.
            *([("y", (rows, 4096), torch.bfloat16)] if combine == "rows" else []),
            ("pair_scale", (rows,), i32),
            ("counts", (experts,), i32),
            ("cursor", (experts,), i32),
        ]

    @classmethod
    def nbytes(cls, m, combine="atomic", experts=128, inter=2048):
        return carve(None, cls.layout(m, combine, experts, inter))

    def __init__(self, m, device, *, combine="atomic", arena=None, experts=128, inter=2048):
        if type(m) is not int or m < 0:
            raise ValueError("m must be a nonnegative integer")
        if combine not in ("atomic", "rows"):
            raise ValueError("combine must be atomic or rows")
        self.combine = combine
        device = torch.device(device)
        if device.type == "cuda" and arena is None:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("allocate W4A8Workspace before capture")
        if type(experts) is not int or not 1 <= experts <= MAX_EXPERTS:
            raise ValueError(f"local experts must be 1..{MAX_EXPERTS}")
        if type(inter) is not int or not 128 <= inter <= 2048 or inter % 128:
            raise ValueError("intermediate size must be a multiple of 128 in 128..2048")
        self.m, self.experts, self.inter = m, experts, inter
        self.rows = 6 * m
        self.scale_rows = (self.rows + 3 * experts) // 4 * 4
        layout = self.layout(m, combine, experts, inter)
        if arena is None:
            arena = torch.empty(carve(None, layout), dtype=torch.uint8, device=device)
        elif arena.dtype != torch.uint8 or arena.numel() < carve(None, layout):
            raise ValueError("arena must be uint8 with at least nbytes(m, combine) elements")
        self.y = None
        self.__dict__.update(carve(arena, layout))
        self.device = arena.device
        # All route paths overwrite every pair, including remote slots (-1).
        self.slot_row = self.pair_row.view(m, 6)


@lru_cache(maxsize=1)
def _kernel_type():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint8, Uint16
    from cutlass.memory import SmemAllocator

    encode = _encoder()

    class PrepareKernel:
        def __init__(self, m, config, combine, prequantized, global_experts, experts, inter):
            self.prequantized = prequantized
            # > 0: ids are global and map through an expert map of this length.
            self.global_experts = global_experts
            self.m, self.experts, self.inter = m, experts, inter
            self.rows = 6 * m
            # Each expert's scale rows start at a multiple of 4: up to 3 padding rows each.
            self.scale_rows = (self.rows + 3 * experts) // 4 * 4
            self.config = config
            self.combine = combine
            self.small = self.rows <= config.small_rows
            self.fused = not self.small and config.fuse_assignment
            # Eight BF16 values per lane, four lanes per K32 group.
            self.threads = config.q0_chunk // 8
            # Every launch covers one thread per expert (and the closing offset).
            self.reset_blocks = max(
                (experts + 127) // 128, min(128, (48 * self.scale_rows + 127) // 128)
            )
            self.small_blocks = max(2, (self.rows + 127) // 128, (experts + 128) // 128)

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            input_q: cute.Tensor,
            input_s: cute.Tensor,
            ids: cute.Tensor,
            weights: cute.Tensor,
            q0: cute.Tensor,
            s0: cute.Tensor,
            s1: cute.Tensor,
            row_token: cute.Tensor,
            row_weight: cute.Tensor,
            row_pair: cute.Tensor,
            offsets: cute.Tensor,
            scale_offsets: cute.Tensor,
            num_rows: cute.Tensor,
            pair_row: cute.Tensor,
            pair_scale: cute.Tensor,
            counts: cute.Tensor,
            cursor: cute.Tensor,
            out: cute.Tensor,
            mapping: cute.Tensor,
            local: cute.Tensor,
            stream: cuda.CUstream,
        ):
            # Kernels after the first route pass read the mapped local ids.
            routes = local.iterator if self.global_experts > 0 else ids.iterator
            if cutlass.const_expr(self.small):
                self.small_routes(
                    ids.iterator,
                    weights.iterator,
                    row_token.iterator,
                    row_weight.iterator,
                    row_pair.iterator,
                    offsets.iterator,
                    scale_offsets.iterator,
                    num_rows.iterator,
                    pair_row.iterator,
                    pair_scale.iterator,
                    s0.iterator,
                    s1.iterator,
                    mapping.iterator,
                    local.iterator,
                ).launch(
                    grid=[self.small_blocks, 1, 1],
                    block=[128, 1, 1],
                    stream=stream,
                    use_pdl=self.config.pdl,
                )
            else:
                self.reset(counts.iterator, s0.iterator, s1.iterator).launch(
                    grid=[self.reset_blocks, 1, 1],
                    block=[128, 1, 1],
                    stream=stream,
                    use_pdl=self.config.pdl,
                )
                self.histogram(
                    ids.iterator, counts.iterator, mapping.iterator, local.iterator
                ).launch(
                    grid=[cute.ceil_div(self.rows, 128), 1, 1],
                    block=[128, 1, 1],
                    stream=stream,
                    use_pdl=self.config.pdl,
                )
                self.prefix(
                    counts.iterator,
                    offsets.iterator,
                    scale_offsets.iterator,
                    num_rows.iterator,
                    cursor.iterator,
                ).launch(
                    grid=[1, 1, 1],
                    block=[self.experts, 1, 1],
                    stream=stream,
                    use_pdl=self.config.pdl,
                )
                if cutlass.const_expr(not self.fused):
                    self.assign(
                        routes,
                        weights.iterator,
                        row_token.iterator,
                        row_weight.iterator,
                        row_pair.iterator,
                        offsets.iterator,
                        scale_offsets.iterator,
                        pair_row.iterator,
                        pair_scale.iterator,
                        cursor.iterator,
                    ).launch(
                        grid=[cute.ceil_div(self.rows, 128), 1, 1],
                        block=[128, 1, 1],
                        stream=stream,
                        use_pdl=self.config.pdl,
                    )
            if cutlass.const_expr(self.m > 0):
                chunks = 1 if self.fused else 4096 // self.config.q0_chunk
                self.quant_scatter(
                    x.iterator,
                    input_q.iterator,
                    input_s.iterator,
                    routes,
                    weights.iterator,
                    q0.iterator,
                    s0.iterator,
                    row_token.iterator,
                    row_weight.iterator,
                    row_pair.iterator,
                    offsets.iterator,
                    scale_offsets.iterator,
                    pair_row.iterator,
                    pair_scale.iterator,
                    cursor.iterator,
                    out.iterator,
                ).launch(
                    grid=[self.m, chunks, 1],
                    block=[self.threads, 1, 1],
                    stream=stream,
                    use_pdl=self.config.pdl,
                )

        @cute.jit
        def wait(self):
            if cutlass.const_expr(self.config.pdl):
                cute.arch.griddepcontrol_wait()

        @cute.jit
        def signal(self):
            if cutlass.const_expr(self.config.pdl):
                cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def expert_of(self, ids: cute.Pointer, mapping: cute.Pointer, pair: Int32):
            # A route's local expert: ids are local, or global ids mapped (else -1).
            expert = ids[pair]
            if cutlass.const_expr(self.global_experts > 0):
                g = expert
                expert = Int32(-1)
                if (g >= 0) & (g < self.global_experts):
                    expert = mapping[g]
            return expert

        @cute.jit
        def clear_scales(
            self, s0: cute.Pointer, s1: cute.Pointer, index: Int32, stride: cutlass.Constexpr
        ):
            for j in range(index, (32 + self.inter // 128) * self.scale_rows, stride):
                if j < 32 * self.scale_rows:
                    s0[j] = Int32(0x7F7F7F7F)
                else:
                    s1[j - 32 * self.scale_rows] = Int32(0x7F7F7F7F)

        @cute.kernel
        def small_routes(
            self,
            ids: cute.Pointer,
            weights: cute.Pointer,
            row_token: cute.Pointer,
            row_weight: cute.Pointer,
            row_pair: cute.Pointer,
            offsets: cute.Pointer,
            scale_offsets: cute.Pointer,
            num_rows: cute.Pointer,
            pair_row: cute.Pointer,
            pair_scale: cute.Pointer,
            s0: cute.Pointer,
            s1: cute.Pointer,
            mapping: cute.Pointer,
            local: cute.Pointer,
        ):
            self.wait()
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            index = block * 128 + thread
            self.clear_scales(s0, s1, index, self.small_blocks * 128)
            # Each prefix and pair rank is independent: no cross-CTA reads of
            # metadata written in this launch. Stable rank keeps duplicates.
            if index <= self.experts:
                before = Int32(0)
                for pair in range(self.rows):
                    expert = self.expert_of(ids, mapping, pair)
                    before += Int32((expert >= 0) & (expert < index))
                offsets[index] = before
                if index < self.experts:
                    scale_offsets[index] = ((before + 3 * index) // 4) * 4
                else:
                    num_rows[0] = before
            if index < self.rows:
                expert = self.expert_of(ids, mapping, index)
                if cutlass.const_expr(self.global_experts > 0):
                    local[index] = expert
                pair_row[index] = Int32(-1)
                pair_scale[index] = Int32(-1)
                if (expert >= 0) & (expert < self.experts):
                    before = Int32(0)
                    rank = Int32(0)
                    for pair in range(self.rows):
                        other = self.expert_of(ids, mapping, pair)
                        before += Int32((other >= 0) & (other < expert))
                        rank += Int32((other == expert) & (pair < index))
                    row = before + rank
                    pair_row[index] = row
                    pair_scale[index] = ((before + 3 * expert) // 4) * 4 + rank
                    row_token[row] = index // 6
                    row_weight[row] = weights[index]
                    row_pair[row] = index
            self.signal()

        @cute.kernel
        def reset(self, counts: cute.Pointer, s0: cute.Pointer, s1: cute.Pointer):
            self.wait()
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            index = block * 128 + thread
            if index < self.experts:
                counts[index] = Int32(0)
            self.clear_scales(s0, s1, index, self.reset_blocks * 128)
            self.signal()

        @cute.kernel
        def histogram(
            self,
            ids: cute.Pointer,
            counts: cute.Pointer,
            mapping: cute.Pointer,
            local: cute.Pointer,
        ):
            self.wait()
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            pair = block * 128 + thread
            if pair < self.rows:
                expert = self.expert_of(ids, mapping, pair)
                if cutlass.const_expr(self.global_experts > 0):
                    local[pair] = expert
                if (expert >= 0) & (expert < self.experts):
                    cute.arch.atomic_add(counts + expert, Int32(1), sem="relaxed", scope="gpu")
            self.signal()

        @cute.kernel
        def prefix(
            self,
            counts: cute.Pointer,
            offsets: cute.Pointer,
            scale_offsets: cute.Pointer,
            num_rows: cute.Pointer,
            cursor: cute.Pointer,
        ):
            self.wait()
            expert, _, _ = cute.arch.thread_idx()
            before = Int32(0)
            for j in range(expert):
                before += counts[j]
            offsets[expert] = before
            scale_offsets[expert] = ((before + 3 * expert) // 4) * 4
            cursor[expert] = before
            if expert == self.experts - 1:
                length = before + counts[expert]
                offsets[self.experts] = length
                num_rows[0] = length
            self.signal()

        @cute.jit
        def reserve(
            self,
            pair: Int32,
            ids: cute.Pointer,
            weights: cute.Pointer,
            row_token: cute.Pointer,
            row_weight: cute.Pointer,
            row_pair: cute.Pointer,
            offsets: cute.Pointer,
            scale_offsets: cute.Pointer,
            pair_row: cute.Pointer,
            pair_scale: cute.Pointer,
            cursor: cute.Pointer,
        ):
            expert = ids[pair]
            row = Int32(-1)
            scale = Int32(-1)
            if (expert >= 0) & (expert < self.experts):
                row = cute.arch.atomic_add(cursor + expert, Int32(1), sem="relaxed", scope="gpu")
                scale = scale_offsets[expert] + row - offsets[expert]
                row_token[row] = pair // 6
                row_weight[row] = weights[pair]
                row_pair[row] = pair
            pair_row[pair] = row
            pair_scale[pair] = scale
            return row, scale

        @cute.kernel
        def assign(
            self,
            ids: cute.Pointer,
            weights: cute.Pointer,
            row_token: cute.Pointer,
            row_weight: cute.Pointer,
            row_pair: cute.Pointer,
            offsets: cute.Pointer,
            scale_offsets: cute.Pointer,
            pair_row: cute.Pointer,
            pair_scale: cute.Pointer,
            cursor: cute.Pointer,
        ):
            self.wait()
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            pair = block * 128 + thread
            if pair < self.rows:
                self.reserve(
                    pair,
                    ids,
                    weights,
                    row_token,
                    row_weight,
                    row_pair,
                    offsets,
                    scale_offsets,
                    pair_row,
                    pair_scale,
                    cursor,
                )
            self.signal()

        @cute.kernel
        def quant_scatter(
            self,
            x: cute.Pointer,
            input_q: cute.Pointer,
            input_s: cute.Pointer,
            ids: cute.Pointer,
            weights: cute.Pointer,
            q0: cute.Pointer,
            s0: cute.Pointer,
            row_token: cute.Pointer,
            row_weight: cute.Pointer,
            row_pair: cute.Pointer,
            offsets: cute.Pointer,
            scale_offsets: cute.Pointer,
            pair_row: cute.Pointer,
            pair_scale: cute.Pointer,
            cursor: cute.Pointer,
            out: cute.Pointer,
        ):
            self.wait()
            thread, _, _ = cute.arch.thread_idx()
            token, chunk, _ = cute.arch.block_idx()
            allocator = SmemAllocator()
            destinations = allocator.allocate_tensor(Int32, cute.make_layout(6))
            scale_destinations = allocator.allocate_tensor(Int32, cute.make_layout(6))
            if thread < 6:
                pair = 6 * token + thread
                if cutlass.const_expr(self.fused):
                    row, scale_row = self.reserve(
                        pair,
                        ids,
                        weights,
                        row_token,
                        row_weight,
                        row_pair,
                        offsets,
                        scale_offsets,
                        pair_row,
                        pair_scale,
                        cursor,
                    )
                    destinations[thread] = row
                    scale_destinations[thread] = scale_row
                else:
                    destinations[thread] = pair_row[pair]
                    scale_destinations[thread] = pair_scale[pair]
            cute.arch.sync_threads()
            iterations = 4096 // self.config.q0_chunk if self.fused else 1
            for iteration in range(iterations):
                col = (chunk + iteration) * self.config.q0_chunk + thread * 8
                if cutlass.const_expr(self.prequantized):
                    codes = cute.make_rmem_tensor((8,), Uint8)
                    for j in cutlass.range_constexpr(8):
                        codes[j] = input_q[token * 4096 + col + j]
                else:
                    raw = cute.make_rmem_tensor((8,), Uint16)
                    for j in cutlass.range_constexpr(8):
                        raw[j] = x[token * 4096 + col + j]
                    codes, scale = encode(raw)
                if cutlass.const_expr(self.combine == "atomic"):
                    for j in cutlass.range_constexpr(8):
                        out[token * 4096 + col + j] = Uint16(0)
                # Sixteen lanes own four groups; exactly lane 0 of that
                # subgroup stores the complete scale word (no byte races).
                base = (thread % 32) // 16 * 16
                word = Int32(0)
                if cutlass.const_expr(self.prequantized):
                    # Only the word owner loads scales; no amax/scale shuffles.
                    # Byte loads also support element-aligned Quantized views.
                    if thread % 16 == 0:
                        for b in cutlass.range_constexpr(4):
                            word = word | (Int32(input_s[token * 128 + col // 32 + b]) << (8 * b))
                else:
                    for b in cutlass.range_constexpr(4):
                        word = word | (
                            cute.arch.shuffle_sync(scale, offset=base + 4 * b) << (8 * b)
                        )
                for slot in cutlass.range_constexpr(6):
                    row = destinations[slot]
                    if row >= 0:
                        for j in cutlass.range_constexpr(8):
                            q0[row * 4096 + col + j] = codes[j]
                        if thread % 16 == 0:
                            s0[(col // 128) * self.scale_rows + scale_destinations[slot]] = word
            # Every CTA signals, including all-remote tokens. Consumers wait
            # before reading maps, scales, codes, or output targeted by atomics.
            self.signal()

    return PrepareKernel


def prepare(x, ids, weights, ws, out, *, config=None, quantized=None, expert_map=None):
    """Build routes, quantize each token once and scatter local slots.

    ``ids`` are local expert ids, or with ``expert_map`` (int32 [E_global], global to local
    or -1) global ids that PREP maps itself, writing the local ids to ``ws.local_ids``.

    With quantized=, gather token-major bytes and pack scales without encoding.
    Zero output only for atomic combine; rows finalize overwrites every token.

    Config controls only launch geometry/PDL; quantization is always exact K32.
    Large-list row order is unspecified. All grids depend only on M and config.
    """
    frozen(x, ids, weights, out)
    if x.ndim != 2 or x.shape[1] != 4096:
        raise ValueError("prepare requires x [M,4096]")
    tensor(x, (len(x), 4096), torch.bfloat16, ws.device, "x")
    tensor(ids, (len(x), 6), torch.int32, ws.device, "ids")
    if expert_map is not None:
        if not len(expert_map):
            raise ValueError("expert_map must be nonempty")
        tensor(expert_map, (len(expert_map),), torch.int32, ws.device, "expert_map")
    tensor(weights, (len(x), 6), torch.float32, ws.device, "weights")
    tensor(out, x.shape, torch.bfloat16, ws.device, "out")
    if len(x) != ws.m:
        raise ValueError("workspace capacity must match M")
    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise TypeError("prepare config must be _w4a8_prepare.Config")
    if quantized is not None:
        validate_quantized(quantized, x)
        input_q, input_s = quantized.values.view(torch.uint8), quantized.scales
        for source in quantized:
            if out.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
                raise ValueError("out must not alias quantized inputs")
    else:
        # Unused, correctly typed dummies; no extra allocation or launch.
        input_q, input_s = ws.q0, ws.q0.view(torch.uint8)
    runtime = require(x.device)
    args = (
        x.view(torch.uint16),
        input_q,
        input_s,
        ids,
        weights,
        ws.q0,
        ws.q0_scales,
        ws.q1_scales,
        ws.row_token,
        ws.row_weight,
        ws.row_pair,
        ws.expert_offsets,
        ws.scale_offsets,
        ws.num_rows,
        ws.pair_row,
        ws.pair_scale,
        ws.counts,
        ws.cursor,
        out.view(torch.uint16),
        ws.counts if expert_map is None else expert_map,
        ws.local_ids,
    )
    prequantized = quantized is not None
    global_experts = 0 if expert_map is None else len(expert_map)
    shape = (ws.m, config, ws.combine, prequantized, global_experts, ws.experts, ws.inter)
    kernel = _kernel_type()(*shape)
    key = ("w4a8_prepare", *shape)
    runtime.compile(key, kernel, args)(*args)
