# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# moe_mxfp8_mxfp4_fc1_act_q1.py, moe_mxfp8_mxfp4_fc2_finalize.py,
# _moe_utils/heuristic.py).
# Local modifications: fixed native checkpoint ABI, exact resource-checked device
# tables, CTI runtime, lazy PREP dependency, explicit warm-up and workspace ownership.

"""DSV4 W4A8 provider with BF16 atomic or deterministic row combine.

An instance owns mutable workspaces. Serialize its calls and graph replays;
use separate instances for overlapping streams/in-flight graphs. Frozen packed
checkpoint weights may be shared between instances. Warm workspace(M) outside
capture and provide caller-owned out during capture.
"""

from dataclasses import dataclass

import torch

from .._common import (
    ExpertWeights,
    Geometry,
    Quantized,
    frozen,
    tensor,
    validate_quantized,
    validate_weights,
)
from . import _w4a8_fc1 as fc1
from . import _w4a8_fc2 as fc2
from . import _w4a8_finalize as finalize
from ._runtime import require
from ._w4a8_debug import validate_debug
from ._w4a8_layout import choose_tile
from ._w4a8_schedule import ORDERS


@dataclass(frozen=True)
class Config:
    combine: str | None = None
    fc1_rows: int | None = None
    fc2_rows: int | None = None
    fc1_stages: int | None = None
    fc2_stages: int | None = None
    grid_multiplier: int = 1
    pdl: bool = True
    # Passed unchanged to the separately owned preparation module.
    prepare_config: object = None
    watchdog: bool = False
    order: str | None = None
    # L2 prefetch run in K tiles (None: the device default in tactic()).
    prefetch: int | None = None
    # FC1 rows up then gate (the canonical MXFP4Experts layout); False: gate then up.
    up_first: bool = False
    # Explicit diagnostics only; outputs may be invalid. Dispatch never sets these.
    debug_mode: str | None = None
    debug_stamps: bool = False

    def __post_init__(self):
        validate_debug(self.debug_mode, self.debug_stamps)
        if self.combine not in (None, "atomic", "rows"):
            raise ValueError("combine must be atomic or rows")
        if self.fc1_rows not in (None, 8, 32, 64, 128) or self.fc2_rows not in (None, 32, 64, 128):
            raise ValueError("FC1 rows must be 8/32/64/128; FC2 rows must be 32/64/128")
        if self.order is not None and self.order not in ORDERS:
            raise ValueError("unsupported schedule order")
        if self.fc1_stages not in (None, 1, 2, 3, 4) or self.fc2_stages not in (None, 1, 2, 3, 4):
            raise ValueError("stage counts must be 1/2/3/4")
        if type(self.grid_multiplier) is not int or self.grid_multiplier not in (1, 2):
            raise ValueError("grid_multiplier must be 1 or 2")
        if not isinstance(self.watchdog, bool):
            raise TypeError("watchdog must be bool")
        if not isinstance(self.pdl, bool):
            raise TypeError("pdl must be bool")


# GB10 rows/combine choices follow the large-M epilogue diagnosis. The shared table's
# 2048 < M <= 4096 bucket takes 32-row FC1 tiles: at ~72-96 routes per local expert (EP2)
# 64-row tiles leave the last one partly empty (LoRA r16 EP2, cold, same-node A/B vs 64/64:
# M=3072 RTX 5090 1827 vs 1980 us, PRO 6000 1845 vs 1926; M=4096 and 6144 within noise;
# M=2048 and 8192 favor 64/64). Buckets separate the requested capacity threshold R/E<=32 at
# M=682/683; observed local counts never affect host dispatch. Device architecture alone
# is not a performance key. The last bucket covers any M: one call reads every expert
# weight once at any T.
UNBOUNDED = 1 << 31
TACTICS = dict.fromkeys(
    ("GB10", "5090", "PRO 6000", "generic"),
    tuple(
        (*entry, "atomic")
        for entry in (
            (32, 8, 32),
            (128, 8, 32),
            (512, 8, 32),
            (682, 8, 32),
            (1024, 64, 64),
            (2048, 64, 64),
            (4096, 32, 64),
            (UNBOUNDED, 64, 64),
        )
    ),
)
TACTICS["GB10"] = (
    (32, 8, 32, "atomic"),
    (128, 8, 32, "atomic"),
    (512, 8, 32, "atomic"),
    (1024, 64, 64, "rows"),
    (2048, 64, 64, "rows"),
    (UNBOUNDED, 64, 64, "rows"),
)


def combine_mode(name, m, config=None):
    """Select the combine contract from the device/M bucket or explicit override."""
    config = config or Config()
    label = next((key for key in TACTICS if key != "generic" and key in name), "generic")
    return config.combine or next(entry[3] for entry in TACTICS[label] if m <= entry[0])


def tactic(name, m, smem_limit, config=None):
    """Metadata-only selection. Return (bucket, FC1 tile, FC2 tile)."""
    if not isinstance(m, int) or m < 1:
        raise ValueError("W4A8 capacity M must be a positive integer")
    config = config or Config()
    label = next((key for key in TACTICS if key != "generic" and key in name), "generic")
    smem_limit //= config.grid_multiplier
    bucket, r1, r2, _ = next(entry for entry in TACTICS[label] if m <= entry[0])
    order = config.order or ("expert_ntile_mtile" if bucket > 1024 else "expert_mtile_ntile")
    bn = 128
    # L2 prefetch of canonical weight row runs, K tiles ahead: off by default. RTX 5090 and
    # PRO 6000 (GDDR7) are faster without (5090 M=1 FC1 21.4 vs 49.1 us, M=4096 1057 vs
    # 1556 us; PRO M=2048 862 vs 938 us at span 3).
    prefetch = 0
    if label == "GB10":
        # GB10 (LPDDR5X): span 3 recovers about half of the 64-B-row TMA cost (8, 16 lose).
        prefetch = 3
        # Measured 2026-10-16 (GB10, M=4096 cold): N-tile-major order with BN=128 lets the
        # concurrent M tiles of an expert share each weight box in L2 (FC1
        # 6.63 -> 5.37 ms, pipeline 10.45 -> 9.73 ms cold).
        order = config.order or ("expert_ntile_mtile" if m > 512 else "expert_mtile_ntile")
    if config.prefetch is not None:
        prefetch = config.prefetch
    return (
        bucket,
        choose_tile(
            1,
            config.fc1_rows or r1,
            smem_limit,
            config.fc1_stages,
            bn=bn,
            watchdog=config.watchdog,
            order=order,
            prefetch=prefetch,
        ),
        choose_tile(
            2,
            config.fc2_rows or r2,
            smem_limit,
            config.fc2_stages,
            watchdog=config.watchdog,
            order=order,
            prefetch=prefetch,
        ),
    )


class W4A8Experts:
    arithmetic = "cute_w4a8"
    path_variant = "w4a8"

    def __init__(self, weights: ExpertWeights, *, config=None, scales=None):
        """``scales``: (w13, w2) scale words already in FlashInfer's 128x4 interleave, int32
        [E, N, K/128] (e.g. the FlashInfer provider's copy), shared instead of built here."""
        # Any local expert count up to DSV4 Flash's 256 (EP1), and any intermediate size I
        # (multiple of 128, <= 2048: 2048 for EP, 2048 / tp for expert TP).
        self.inter = weights.w13.shape[1] // 2 if weights.w13.ndim == 3 else 0
        if not 128 <= self.inter <= 2048 or self.inter % 128:
            raise ValueError("W4A8 experts need I a multiple of 128 in 128..2048")
        validate_weights(weights, Geometry(intermediate=self.inter))
        self.experts = weights.w13.shape[0]
        self.runtime = require(weights.w13.device)
        self.config = config or Config()
        self.device = weights.w13.device
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("construct W4A8Experts outside capture")
            from ._w4a8_model import interleave_weight_scales

            # The canonical weights are read as they are (no copy). Scales are read in
            # FlashInfer's 128x4 interleave: the caller's copy (e.g. FlashInfer's own), else
            # built here once.
            self.w13, self.w2 = weights.w13, weights.w2
            if scales is None:
                scales = tuple(
                    interleave_weight_scales(t) for t in (weights.w13_scales, weights.w2_scales)
                )
            for words, raw in zip(scales, (weights.w13_scales, weights.w2_scales), strict=True):
                shape = (*raw.shape[:2], raw.shape[2] // 4)
                tensor(words, shape, torch.int32, self.device, "interleaved scales")
            self.s13, self.s2 = scales
            props = torch.cuda.get_device_properties(self.device)
            self.grid = props.multi_processor_count * self.config.grid_multiplier
            self.smem_limit = props.shared_memory_per_block_optin
            self.name = props.name
        self.weights = weights
        self._workspaces = {}
        self._launches = {}
        self._finalizers = {}
        self._tactics = {}
        self.debug_buffers = {}

    def workspace(self, m):
        """Allocate and warm PREP/FC1/FC2/finalize for M; return the private PREP workspace.

        With debug_stamps, debug_buffers[M] holds FC1/FC2 int64 [grid, 4]
        buffers: first-work start, last-work end, producer empty-wait total,
        sum of eight consumer warp leaders' full-wait totals, all in ns.
        Replays overwrite records; idle CTAs write zeros.
        """
        if m in self._workspaces:
            return self._workspaces[m]
        selected = tactic(self.name, m, self.smem_limit, self.config)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "call Dsv4RoutedExperts.prepare(M, route_kind) or "
                    "W4A8Experts.workspace(M) before capture"
                )
            from ._w4a8_prepare import W4A8Workspace, prepare

            ws = W4A8Workspace(
                m,
                device=self.device,
                combine=combine_mode(self.name, m, self.config),
                experts=self.experts,
                inter=self.inter,
            )
            x = torch.zeros((m, 4096), dtype=torch.bfloat16, device=self.device)
            ids = torch.full((m, 6), -1, dtype=torch.int32, device=self.device)
            routes = torch.zeros((m, 6), dtype=torch.float32, device=self.device)
            out = torch.empty_like(x)
            prepare(x, ids, routes, ws, out, config=self.config.prepare_config)
            # Warm the gather-only PREP specialization as well, for graph callers.
            q = Quantized(
                torch.zeros_like(x, dtype=torch.float8_e4m3fn),
                torch.full((m, 128), 127, device=self.device, dtype=torch.uint8),
            )
            prepare(x, ids, routes, ws, out, config=self.config.prepare_config, quantized=q)
            stamps = (None, None)
            if self.config.debug_stamps:
                stamps = tuple(
                    torch.empty((self.grid, 4), dtype=torch.int64, device=self.device)
                    for _ in range(2)
                )
            first = fc1.compile_fc1(
                self.runtime,
                selected[1],
                self.grid,
                self.smem_limit,
                ws,
                self.w13,
                self.s13,
                pdl=self.config.pdl,
                debug_mode=self.config.debug_mode,
                stamps=stamps[0],
                up_first=self.config.up_first,
            )
            second = fc2.compile_fc2(
                self.runtime,
                selected[2],
                self.grid,
                self.smem_limit,
                ws,
                self.w2,
                self.s2,
                out,
                pdl=self.config.pdl,
                debug_mode=self.config.debug_mode,
                stamps=stamps[1],
                combine=ws.combine,
            )
            first(*fc1.arguments(ws, self.w13, self.s13, stamps=stamps[0]))
            second(*fc2.arguments(ws, self.w2, self.s2, out, stamps=stamps[1], combine=ws.combine))
            if ws.combine == "rows":
                self._finalizers[m] = {
                    aligned: finalize.compile_finalize(
                        self.runtime, ws, out, pdl=self.config.pdl, aligned=aligned
                    )
                    for aligned in (False, True)
                }
                self._finalizers[m][True](*finalize.arguments(ws, out))
        self.debug_buffers[m] = stamps
        self._workspaces[m] = ws
        self._launches[m] = first, second
        self._tactics[m] = selected
        return ws

    def __call__(self, x, local_ids, weights, *, out=None, quantized=None, expert_map=None):
        """``local_ids`` [M, 6] int32 are local expert ids (-1 or out of range: remote), or
        global ids with ``expert_map`` (int32 [E_global], global to local or -1), which PREP
        maps itself. Warm each M (and mapping) eagerly before capture."""
        frozen(x, local_ids, weights, out, *self.weights.tensors())
        if x.ndim != 2:
            raise ValueError("x must have shape [M,4096]")
        m = x.shape[0]
        tensor(x, (m, 4096), torch.bfloat16, self.device, "x")
        if quantized is not None:
            validate_quantized(quantized, x)
        tensor(local_ids, (m, 6), torch.int32, self.device, "local_ids")
        tensor(weights, (m, 6), torch.float32, self.device, "weights")
        if out is not None:
            tensor(out, (m, 4096), torch.bfloat16, self.device, "out")
            inputs = (
                x,
                local_ids,
                weights,
                *self.weights.tensors(),
                self.w13,
                self.w2,
                self.s13,
                self.s2,
            )
            if quantized is not None:
                inputs += tuple(quantized)
            if any(
                out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs
            ):
                raise ValueError("out must not alias inputs or prepared weights")
        with torch.cuda.device(self.device):
            if out is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("supply out during W4A8 capture")
                out = torch.empty_like(x)
            ws = self.workspace(m)
            # Output must also be disjoint from mutable preparation/GEMM storage.
            if any(
                out.untyped_storage().data_ptr() == getattr(ws, name).untyped_storage().data_ptr()
                for name in (
                    "q0",
                    "q0_scales",
                    "q1",
                    "q1_scales",
                    "row_token",
                    "row_weight",
                    "row_pair",
                    "expert_offsets",
                    "scale_offsets",
                    "num_rows",
                    "pair_row",
                    "pair_scale",
                    "counts",
                    "cursor",
                    "y",
                )
                if getattr(ws, name) is not None
            ):
                raise ValueError("out must not alias the workspace")
            from ._w4a8_prepare import prepare

            prepare(
                x,
                local_ids,
                weights,
                ws,
                out,
                config=self.config.prepare_config,
                quantized=quantized,
                expert_map=expert_map,
            )
            first, second = self._launches[m]
            stamps = self.debug_buffers[m]
            first(*fc1.arguments(ws, self.w13, self.s13, stamps=stamps[0]))
            second(*fc2.arguments(ws, self.w2, self.s2, out, stamps=stamps[1], combine=ws.combine))
            if ws.combine == "rows":
                self._finalizers[m][out.data_ptr() % 16 == 0](*finalize.arguments(ws, out))
        return out
