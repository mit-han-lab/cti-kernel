# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Row copies across a pool's state tensors (prefix-hit state restore)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .._reroute import engine_checks

__all__ = ["RowCopy"]


class RowCopy:
    """Copies whole rows of a fixed set of tensors, ``tensor[targets[i]] = tensor[sources[i]]``
    for every tensor, bit for bit, in one launch on CUDA.

    Each tensor's dim 0 indexes rows, each row (``tensor[r]``) is contiguous and rows do not
    overlap (``stride(0)`` at least the row size); tensors may differ in row shape, dtype, row
    stride and row count. The row table is a small device tensor allocated at construction that
    holds the tensors' addresses: the tensors may live in a region whose memory is released and
    restored at the same addresses (for example torch_memory_saver), the table may not; rebuild
    RowCopy if the tensors move, and keep it alive while CUDA graphs that captured it replay.

    Index -1 skips that copy, and so does an index out of a tensor's row range, per tensor
    (tensors with fewer rows skip copies the others make). Valid targets must be distinct and
    disjoint from valid sources (otherwise the result is undefined). With ``CTI_ENGINE_CHECKS``
    on, a call checks this (and that every index is -1 or below the largest row count) with a
    host synchronization and raises ValueError; the CPU provider always checks. Calls otherwise
    never synchronize with the host and can be captured in a CUDA graph after one eager call
    (or ``warm_up``) of the same index dtype; capturing a cold dtype raises RuntimeError.
    """

    def __init__(self, tensors: Sequence[torch.Tensor]):
        tensors = tuple(tensors)
        if not tensors:
            raise ValueError("RowCopy needs at least one tensor")
        device = tensors[0].device
        for t in tensors:
            if t.device != device:
                raise ValueError("RowCopy tensors must share one device")
            if t.dim() < 1 or t.shape[0] < 1:
                raise ValueError("RowCopy tensors need at least one row")
            if t.requires_grad:
                raise ValueError("RowCopy tensors must not require grad")
            if not t[0].is_contiguous():
                raise ValueError("each row of a RowCopy tensor must be contiguous")
            if t.shape[0] > 1 and t.stride(0) < t[0].numel():
                raise ValueError("RowCopy tensor rows must not overlap (stride(0) below the row)")
        self.device = device
        if device.type == "cuda":
            from .triton import Provider
        elif device.type == "cpu":
            from .torch import Provider
        else:
            raise ValueError(f"RowCopy supports CPU and CUDA tensors, not {device.type}")
        self._provider = Provider(tensors)
        self.provider = self._provider.name
        self._rows = max(t.shape[0] for t in tensors)

    def warm_up(self, index_dtypes=(torch.int64,)) -> None:
        """Compile the launch for each index dtype (call before CUDA graph capture)."""
        self._provider.warm_up(tuple(index_dtypes))

    def __call__(self, sources: torch.Tensor, targets: torch.Tensor) -> None:
        """Copy row ``sources[i]`` to row ``targets[i]`` of every tensor, on the current stream.
        ``sources`` and ``targets`` are contiguous 1-D int32 or int64 tensors of one length on
        the tensors' device."""
        if sources.dim() != 1 or sources.shape != targets.shape:
            raise ValueError("sources and targets must be 1-D and of one length")
        if sources.dtype not in (torch.int32, torch.int64) or targets.dtype != sources.dtype:
            raise ValueError("sources and targets must both be int32 or both int64")
        if sources.device != self.device or targets.device != self.device:
            raise ValueError("sources and targets must be on the tensors' device")
        if sources.stride(0) != 1 or targets.stride(0) != 1:
            raise ValueError("sources and targets must be contiguous (unit stride)")
        if sources.numel():
            if self.device.type == "cpu" or engine_checks():
                check_indices(sources, targets, self._rows)
            self._provider(sources, targets)


def check_indices(sources: torch.Tensor, targets: torch.Tensor, rows: int) -> None:
    """Raise ValueError unless every index is -1 or in [0, rows), the valid targets are
    distinct and no valid source is also a valid target (a host synchronization on CUDA)."""
    for name, index in (("sources", sources), ("targets", targets)):
        if bool(((index != -1) & ((index < 0) | (index >= rows))).any()):
            raise ValueError(f"RowCopy {name} must be -1 or in [0, {rows})")
    valid_sources, valid_targets = sources[sources >= 0], targets[targets >= 0]
    if valid_targets.unique().numel() != valid_targets.numel():
        raise ValueError("RowCopy valid targets must be distinct")
    if bool(torch.isin(valid_sources, valid_targets).any()):
        raise ValueError("RowCopy valid sources and targets must be disjoint")
