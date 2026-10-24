# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Kernel scratch shared by one group of CUDA graphs."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Hashable

import torch

_ACTIVE: contextvars.ContextVar[GraphWorkspace | None] = contextvars.ContextVar(
    "graph_workspace", default=None
)


def _fits(work, counts):
    return all(t.numel() >= n for t, n in zip(work, counts, strict=True))


class GraphWorkspace:
    """Kernel scratch shared by one group of CUDA graphs.

    Contract: the graphs captured inside one workspace replay serially on one
    stream, never concurrently, and the workspace outlives them: graphs hold no
    reference, so ``close()`` only after every graph captured in it is released.

    ``max_rows`` bounds the rows of every captured call in the group. Enter the
    workspace around the group's eager warm-ups and captures, in any order;
    nesting it is a no-op and entering another one raises.

    Consumers (kernel ops) find it with :meth:`active` and own one slot each,
    keyed ``(consumer, device, dtype)``. An eager call inside the workspace
    :meth:`reserve`\\ s its call shape: the slot is (re)allocated eagerly, as
    ordinary allocator memory outside any graph pool, to cover that shape at
    every row count up to ``max_rows``. A call :meth:`take`\\ s its slot's
    tensors; the first take during a capture freezes every slot, after which a
    take that needs more raises instead of growing. An eager take before that sees the current,
    possibly later reallocated, tensors: take again on every call. Slots persist
    across replays and the workspace adds no graph nodes, so kernels must leave
    state such as counters as they found it.
    """

    def __init__(self, max_rows: int):
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._sizes: dict[tuple, tuple[int, ...]] = {}  # (slot, call) -> counts
        self._work: dict[tuple, tuple[torch.Tensor, ...]] = {}  # slot -> tensors
        self._frozen = False
        self._closed = False
        self._token = None
        self._depth = 0

    @staticmethod
    def active() -> GraphWorkspace | None:
        """The workspace entered in this context, if any."""
        return _ACTIVE.get()

    def __enter__(self) -> GraphWorkspace:
        active = _ACTIVE.get()
        if active is not None and active is not self:
            raise RuntimeError("another GraphWorkspace is already active")
        if self._closed:
            raise RuntimeError("GraphWorkspace is closed")
        if active is None:
            self._token = _ACTIVE.set(self)
        self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth == 0:
            token, self._token = self._token, None
            _ACTIVE.reset(token)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(t.numel() * t.element_size() for w in self._work.values() for t in w)

    def close(self) -> None:
        """Release every slot; no graph captured inside may replay afterwards."""
        with self._lock:
            self._closed = True
            self._sizes.clear()
            self._work.clear()

    def reserve(
        self,
        slot: tuple,
        call: Hashable,
        size: Callable[..., tuple[int, ...]],
        allocate: Callable[[tuple[int, ...]], tuple[torch.Tensor, ...]],
    ) -> None:
        """Cover call shape ``call`` in ``slot`` before the first capture (a no-op after it).

        ``size(max_rows, *call)`` returns per-tensor element counts covering every
        row count up to ``max_rows``; ``allocate(counts)`` builds zero-initialized
        (where needed) tensors. After the first capture this is a no-op.
        """
        with self._lock:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                self._frozen = True  # never allocate inside a capture
            if self._frozen or self._closed:
                return
            key = (slot, call)
            if key not in self._sizes:
                self._sizes[key] = size(self.max_rows, *call)
            counts = self._sizes[key]
            work = self._work.get(slot)
            if work is None or not _fits(work, counts):
                if work is not None:
                    counts = tuple(max(t.numel(), n) for t, n in zip(work, counts, strict=True))
                # Shared by the group: never an inference tensor of the reserving call.
                with torch.inference_mode(False):
                    self._work[slot] = allocate(counts)

    def take(self, slot: tuple, rows: int, counts: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """Return ``slot``'s tensors; a take during capture freezes the workspace."""
        with self._lock:
            if self._closed:
                raise RuntimeError("GraphWorkspace is closed")
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                self._frozen = True
            if rows > self.max_rows:
                raise RuntimeError(
                    f"graph workspace slot {slot}: {rows} rows exceed max_rows={self.max_rows}"
                )
            work = self._work.get(slot)
            if work is None:
                raise RuntimeError(
                    f"graph workspace slot {slot} was not reserved before the first capture; "
                    "run the group's eager warm-up inside the workspace"
                )
            if not _fits(work, counts):
                raise RuntimeError(
                    f"graph workspace slot {slot} too small at {rows} rows: needs {counts}, "
                    f"reserved {tuple(t.numel() for t in work)}; warm up every call shape "
                    "inside the workspace before the first capture"
                )
            return work
