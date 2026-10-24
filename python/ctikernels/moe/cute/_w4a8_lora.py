# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DSV4 W4A8 routed experts with per-expert LoRA fused into the FC1/FC2 epilogues.

Stages, composed by the op layer for one call of any T: ``begin`` (PREP), the gate/up
shrink z13 = x A13 (``shrink_pairs``, the CuTe pair kernel, or the caller's own), ``finish``.
FC1 and FC2 seed their FP32 accumulators with
the LoRA deltas s (z13 B13) and s (z2 B2) before the K loop, so the gate/up delta enters before
the clamped weighted SwiGLU and the down delta before the route sum. FC1 also writes per-N-tile
partials of z2 = h A2 from its BF16 h tile; the N tile completing a (pair, rank) entry (an
integer ticket) sums them in a fixed order. Slot-ordered row combine (one BF16 rounding). No
floating-point atomics: deterministic. Warm each (M, rank) before capture.

Scratch (tickets, PREP rows, the FP32 row output, z13, partials, z2) is one zeroed byte
arena: the active ``GraphWorkspace`` slot ("moe_lora_w4a8", device, uint8), shared by every
layer's op; outside one, the op owns one arena per (M, rank).
"""

import torch

from ..._runtime.workspace import GraphWorkspace
from .._lora_common import scale_tensor
from .._lora_weights import FactorSlices
from . import _lora_provider
from . import _w4a8_fc1 as fc1
from . import _w4a8_fc2 as fc2
from . import _w4a8_finalize as finalize
from ._w4a8 import Config, W4A8Experts, tactic
from ._w4a8_prepare import W4A8Workspace, carve, prepare

TOP_K = 6


class W4A8LoRAExperts(W4A8Experts):
    """``W4A8Experts`` plus one adapter's six routed leaves per call (row combine only)."""

    def __init__(self, weights, *, config=None, scales=None):
        config = config or Config()
        if config.combine not in (None, "rows"):
            raise ValueError("W4A8 LoRA uses the deterministic row combine")
        rows = Config(**{**config.__dict__, "combine": "rows"})
        super().__init__(weights, config=rows, scales=scales)
        self._lora = {}  # (m, rank, by_row) -> (FC1, FC2, finalizers)
        self._arenas = {}  # (m, rank) -> private arena, outside a GraphWorkspace
        self._views = {}  # (m, rank) -> (arena pointer, ws, z13, parts, z2, count)
        self._tickets = {}  # arena pointer -> its ticket region (int32)

    def _layout(self, m, rank):
        p, ntiles = (
            TOP_K * m,
            self.inter // tactic(self.name, 1, self.smem_limit, self.config)[1].bn,
        )
        f32 = torch.float32
        return W4A8Workspace.layout(m, "rows", self.experts, self.inter) + [
            ("z13", (p, 2, rank), f32),
            # One FP32 partial of h A2 per FC1 N tile; FC1 sums them in order into z2.
            ("parts", (p, ntiles, rank), f32),
            ("z2", (p, 1, rank), f32),
        ]

    @staticmethod
    def _front(counters):
        # FC1's (pair, rank) tickets lead the arena: every call leaves them zero, and they
        # never overlap another call's scratch. Arenas are allocated zeroed.
        return -(-4 * counters // 256) * 256

    def nbytes(self, m, rank, counters):
        return self._front(counters) + carve(None, self._layout(m, rank))

    def workspace_bytes(self, max_rows, rank):
        """Bytes of the GraphWorkspace slot that serves calls of up to max_rows tokens at
        rank (one slot per device, shared by every layer); grows linearly in max_rows, the
        BF16 per-route output being 6 x 8 KiB per token."""
        return self.nbytes(max_rows, rank, TOP_K * max_rows * 128)

    def _arena(self, m, rank):
        """(arena, ticket count): a private arena per (M, rank), or the active slot's."""
        zeros = lambda n: torch.zeros(n, dtype=torch.uint8, device=self.device)
        group = GraphWorkspace.active()
        if group is None:
            key, counters = (m, rank), TOP_K * m * rank
            if key not in self._arenas:
                self._arenas[key] = zeros(self.nbytes(m, rank, counters))
            return self._arenas[key], counters
        slot = ("moe_lora_w4a8", self.device, torch.uint8)
        # Ticket room for every row count and rank the slot may serve.
        counters = TOP_K * group.max_rows * 128

        def size(rows, r):  # monotone in M: the slot's max_rows covers every call
            return (self.nbytes(rows, r, counters),)

        group.reserve(slot, (rank,), size, lambda counts: (zeros(counts[0]),))
        return group.take(slot, m, (self.nbytes(m, rank, counters),))[0], counters

    def _buffers(self, m, rank):
        arena, counters = self._arena(m, rank)
        cached = self._views.get((m, rank))
        if cached is None or cached[0] != arena.data_ptr():
            if cached is not None:  # the slot grew: drop views of the old arena
                self._views.clear()
                self._tickets.clear()
            front = self._front(counters)
            self._tickets[arena.data_ptr()] = arena[: 4 * counters].view(torch.int32)
            count = arena[: 4 * TOP_K * m * rank].view(torch.int32)
            views = carve(arena[front:], self._layout(m, rank))
            ws = W4A8Workspace(
                m,
                self.device,
                combine="rows",
                arena=arena[front:],
                experts=self.experts,
                inter=self.inter,
            )
            cached = (arena.data_ptr(), ws, views["z13"], views["parts"], views["z2"], count)
            self._views[(m, rank)] = cached
        return cached[1:]

    def reset_scratch(self):
        """Re-zero the FC1 tickets, e.g. after a memory-saver resume: they must be zero on
        entry. Eager only."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("reset_scratch is eager only")
        for tickets in self._tickets.values():
            tickets.zero_()

    def _compile_lora(self, m, rank, ws, buffers, out, by_row):
        key = (m, rank, by_row)
        if key in self._lora:
            return self._lora[key]
        z13, parts, z2, count = buffers
        selected = tactic(self.name, m, self.smem_limit, self.config)
        e = self.w13.shape[0]
        scale = scale_tensor(1.0, self.device)
        dummy = {
            name: torch.zeros(shape, dtype=torch.bfloat16, device=self.device)
            for name, shape in (
                ("gate_b", (e, self.inter, rank)),
                ("up_b", (e, self.inter, rank)),
                ("down_a", (e, rank, self.inter)),
                ("down_b", (e, 4096, rank)),
            )
        }
        first = fc1.compile_fc1(
            self.runtime,
            selected[1],
            self.grid,
            self.smem_limit,
            ws,
            self.w13,
            self.s13,
            pdl=self.config.pdl,
            lora=(z13, dummy["gate_b"], dummy["up_b"], scale, dummy["down_a"], parts, z2, count),
            up_first=self.config.up_first,
            lora_rows=by_row,
        )
        second = fc2.compile_fc2(
            self.runtime,
            selected[2],
            self.grid,
            self.smem_limit,
            ws,
            self.w2,
            self.s2,
            ws.y,
            pdl=self.config.pdl,
            combine="rows",
            lora=(z2, dummy["down_b"], scale),
            lora_rows=by_row,
        )
        finalizers = {
            aligned: finalize.compile_finalize(
                self.runtime, ws, out, pdl=self.config.pdl, aligned=aligned
            )
            for aligned in (False, True)
        }
        self._lora[key] = first, second, finalizers
        return self._lora[key]

    def begin(self, x, ids, weights, rank, out, *, expert_map=None, by_row=False):
        """Route preparation for one call of M >= 1 tokens: returns the call's scratch
        (ws, z13, parts, z2, tickets, launches). With ``by_row`` the caller's shrink writes
        z13 in PREP's sorted-row order (else pair order), and FC1 seeds the gate/up deltas on
        tensor cores. ``ids`` are local (-1 or out of range: no local route), or global with
        ``expert_map`` (PREP maps them into ws.local_ids)."""
        m = x.shape[0]
        if m < 1:
            raise ValueError("a W4A8 LoRA call has at least one token")
        if (m, rank, by_row) not in self._lora and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm W4A8 LoRA (M, rank) before capture")
        with torch.cuda.device(self.device):
            ws, *buffers = self._buffers(m, rank)
            launches = self._compile_lora(m, rank, ws, buffers, out, by_row)
            prepare(
                x, ids, weights, ws, out, config=self.config.prepare_config, expert_map=expert_map
            )
        return (ws, *buffers, launches)

    def shrink_pairs(self, scratch, x, ids, gate_a, up_a, *, expert_map=None):
        """The gate/up shrink z13 = x A13 in pair order (the CuTe pair kernel; small T)."""
        ws, z13 = scratch[0], scratch[1]
        local = ids if expert_map is None else ws.local_ids.view(x.shape[0], TOP_K)
        with torch.cuda.device(self.device):
            _lora_provider.shrink_(z13, x, FactorSlices((gate_a, up_a)), local, TOP_K)

    def finish(self, scratch, gate_b, up_b, down_a, down_b, scale, out):
        """FC1 and FC2 with the LoRA deltas, then the row combine into ``out``; z13 is ready."""
        ws, z13, parts, z2, count, (first, second, finalizers) = scratch
        scale = scale_tensor(scale, out.device)
        with torch.cuda.device(self.device):
            lora13 = (z13, gate_b, up_b, scale, down_a, parts, z2, count)
            first(*fc1.arguments(ws, self.w13, self.s13, lora=lora13))
            second(
                *fc2.arguments(ws, self.w2, self.s2, out, combine="rows", lora=(z2, down_b, scale))
            )
            finalizers[out.data_ptr() % 16 == 0](*finalize.arguments(ws, out))
        return out
