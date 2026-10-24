# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Borrowed replay storage and pure-host byte descriptions (contract §§3, 8)."""

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class Region:
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    initial_value: int | None = None
    alignment: int = 256
    lifetime: str = "layer"

    @property
    def payload_bytes(self) -> int:
        return (
            prod(self.shape)
            * {
                torch.bfloat16: 2,
                torch.float32: 4,
                torch.int64: 8,
                torch.int32: 4,
                torch.bool: 1,
                torch.uint8: 1,
            }[self.dtype]
        )

    @property
    def aligned_bytes(self) -> int:
        return (self.payload_bytes + 255) // 256 * 256


def _valid_q(q):
    """Any positive verify length, including the anchor-only Q=1 case."""
    return isinstance(q, int) and not isinstance(q, bool) and q >= 1


def _valid_window(window):
    return type(window) is int and window >= 2 and window & (window - 1) == 0


def regions(entries):
    result, offset = {}, 0
    for name, shape, dtype, initial in entries:
        lifetime = (
            "transaction"
            if name.startswith("pending.")
            else "invocation"
            if name in ("status", "sections", "arrivals", "controls")
            else "consumer"
            if name == "out"
            else "layer"
        )
        region = Region(shape, dtype, offset, initial, lifetime=lifetime)
        result[name] = region
        offset += region.aligned_bytes
    return result


@dataclass(frozen=True)
class ReplayState:
    """Borrowed dense tensors; metadata/epoch must initially be zero (§§3–4)."""

    checkpoint: torch.Tensor
    conv_history: torch.Tensor
    ring: torch.Tensor
    metadata: torch.Tensor
    pool_epoch: torch.Tensor
    snapshot_metadata: torch.Tensor


@dataclass(frozen=True)
class ReplayLayers:
    """Validated borrowed layer table. Rebind after replacing any tensor.

    The tensors (including the pointer table) must retain their storage, shape and
    strides for the binding's lifetime. The caller owns stream ordering and unique
    destination slots, just as for single-layer handoff.
    """

    owner: object
    configuration: tuple
    states: tuple[ReplayState, ...]
    canonical_states: tuple[torch.Tensor, ...]
    histories: tuple[torch.Tensor, ...]
    table: torch.Tensor
    table_flat: torch.Tensor
    batch_size: int
    num_slots: int
    num_snapshot_rows: int
    copies_checkpoint: bool


@dataclass(frozen=True)
class ReplayPending:
    """One independently owned transaction block; metadata initially zero (§5)."""

    raw: torch.Tensor
    metadata: torch.Tensor


def replay_sizes(
    *,
    num_k_heads: int,
    num_v_heads: int,
    key_dim: int = 128,
    value_dim: int = 128,
    conv_width: int = 4,
    window: int = 8,
    verify_length: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    num_slots: int = 1,
    num_snapshot_rows: int = 1,
    batch_size: int = 1,
    immediate: bool = False,
):
    """Per-layer regions: payload versus ABI allocation bytes, contract §8.

    Alignment is applied to each *whole region*, never each row. ``ring`` includes
    L raw records and three history seeds; authoritative history is counted separately.
    ``immediate=True`` omits ring payload while retaining its region descriptor.
    Defaults describe one live slot, one canonical snapshot and one pending row.
    This function performs only host arithmetic and never allocates a tensor.
    """
    from .._contract import Geometry

    g = Geometry(num_k_heads, num_v_heads)
    if (key_dim, value_dim, conv_width) != (128, 128, 4) or dtype != torch.bfloat16:
        raise ValueError("portable replay requires K=V=128, conv_width=4 and BF16 records")
    if not _valid_window(window) or not _valid_q(verify_length):
        raise ValueError("window must be a power of two >= 2 and verify_length an integer >= 1")
    if any(type(n) is not int or n < 0 for n in (num_slots, num_snapshot_rows, batch_size)):
        raise ValueError("pool and batch counts must be nonnegative integers")
    r, s, b = num_slots, num_snapshot_rows, batch_size
    dc, w, hv = g.conv_dim, g.conv_dim + 2 * num_v_heads, num_v_heads
    return regions(
        [
            ("checkpoint", (r + s, hv, 128, 128), torch.float32, None),
            ("conv_history", (r + s, 3, dc), dtype, None),
            ("ring", (r, 0 if immediate else window + 3, w), dtype, None),
            ("metadata", (r, 8), torch.int64, 0),
            ("pool_epoch", (1,), torch.int64, 0),
            ("snapshot_metadata", (s, 4), torch.int64, 0),
            ("pending.raw", (b, verify_length, w), dtype, None),
            ("pending.metadata", (b, 8), torch.int64, 0),
        ]
    )
