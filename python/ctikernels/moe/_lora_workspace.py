# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Grow-only scratch arenas, private to each device and current CUDA stream.

Carved views may only be used within one call on that stream. Calls on the same
stream are ordered by the stream; different streams have independent arenas.
A call captured into a CUDA graph on a stream without a large enough arena
borrows the device's largest one (``torch.cuda.graph`` captures on a private
side stream after the warm-up calls): replays and eager calls that share an
arena must not run concurrently, as with any engine's resident MoE workspace.
Callers must serialize host-side submission of whole calls to the same stream;
the lock protects bookkeeping, not interleaved kernel submissions.
"""

import math
import threading
from dataclasses import dataclass

import torch

ENABLED = True  # Private test switch for the routed composition.
_ALIGNMENT = 256
_CHUNK = 2 * 1024 * 1024
_LOCK = threading.RLock()
_ARENAS = {}
_RETIRED = []  # Captured pointers must remain valid for the process lifetime.


@dataclass
class _Arena:
    buffer: torch.Tensor
    captured: bool = False


def _key(device):
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        return (stream.device, stream.cuda_stream)
    return (device, None)


def _capturing(device):
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def carve(device, specs):
    """Return contiguous, disjoint 256-byte-aligned views for (shape, dtype) specs.

    Growth during capture uses ordinary per-call allocations instead of replacing
    the arena. Warm up the largest batch on the capture stream before capture.
    CPU devices are supported for testing the allocator's bookkeeping.
    """
    device = torch.device(device)
    specs = list(specs)
    offsets, sizes, need = [], [], 0
    for shape, dtype in specs:
        if any(d < 0 for d in shape):
            raise ValueError("workspace dimensions must be nonnegative")
        size = math.prod(shape) * dtype.itemsize
        need = -(-need // _ALIGNMENT) * _ALIGNMENT
        offsets.append(need)
        sizes.append(size)
        need += size
    with _LOCK:
        key = _key(device)
        capturing = _capturing(device)
        arena = _ARENAS.get(key)
        if capturing and (arena is None or need > arena.buffer.numel()):
            # torch.cuda.graph captures on its own side stream: borrow the largest
            # arena the device's warm-up calls grew (capture is ordered after them).
            borrowed = [
                a for k, a in _ARENAS.items() if k[0] == key[0] and a.buffer.numel() >= need
            ]
            if borrowed:
                arena = max(borrowed, key=lambda a: a.buffer.numel())
                key = next(k for k, a in _ARENAS.items() if a is arena)
        old_size = 0 if arena is None else arena.buffer.numel()
        if arena is None or need > old_size:
            if capturing:
                return [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs]
            capacity = -(-max(need, (5 * old_size + 3) // 4, 1) // _CHUNK) * _CHUNK
            # CUDA allocations are already aligned. CPU allocations can start at
            # a 64-byte boundary; reserve enough slack to align the usable buffer.
            slack = _ALIGNMENT - 1 if device.type == "cpu" else 0
            buffer = torch.empty(capacity + slack, dtype=torch.uint8, device=device)
            start = -buffer.data_ptr() % _ALIGNMENT
            buffer = buffer.narrow(0, start, capacity)
            if arena is not None and arena.captured:
                _RETIRED.append(arena)
            arena = _ARENAS[key] = _Arena(buffer)
        arena.captured |= capturing
        return [
            arena.buffer.narrow(0, offset, size).view(dtype).view(shape)
            for (shape, dtype), offset, size in zip(specs, offsets, sizes, strict=True)
        ]


def release():
    """Drop uncaptured arenas; captured and retired buffers remain alive."""
    with _LOCK:
        for key in list(_ARENAS):
            if not _ARENAS[key].captured:
                del _ARENAS[key]
