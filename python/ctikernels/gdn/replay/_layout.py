# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host-only borrowed layout descriptors and exact byte-region alias checks."""

from dataclasses import dataclass
from heapq import merge
from math import gcd


def span(tensor):
    """Physical element span, relative to the view's existing storage offset."""
    return (
        0
        if not tensor.numel()
        else 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))
    )


def physical(tensor):
    return tensor.as_strided((span(tensor),), (1,))


def dense_rows(name, tensor):
    """One independent outer pitch, with nonoverlapping dense inner rows."""
    width = 1
    for n, s in reversed(list(zip(tensor.shape[1:], tensor.stride()[1:], strict=True))):
        if s != width:
            raise ValueError(f"{name} requires dense inner layout")
        width *= n
    if tensor.stride(0) < width:
        raise ValueError(f"{name} requires nonoverlapping rows")
    if tensor.stride(0) * tensor.element_size() % 16:
        raise ValueError(f"{name} requires 16-byte slot pitch")


def feature_rows(name, tensor):
    """Dense features, independently pitched tokens and batches."""
    width = tensor.shape[-1]
    if tensor.stride(-1) != 1 or tensor.stride(-2) < width:
        raise ValueError(f"{name} requires nonoverlapping dense feature rows")
    if tensor.ndim == 3 and tensor.stride(0) < ((tensor.shape[1] - 1) * tensor.stride(1) + width):
        raise ValueError(f"{name} requires nonoverlapping batches")


def _intervals(tensor):
    """Sorted occupied byte runs; dense payload dimensions collapse to one run.

    Zero strides repeat the same addresses. Merging handles arbitrary control
    strides as well as monotonically pitched pools, without reading tensor values.
    """
    width = tensor.element_size()
    dimensions = []
    for stride, count in sorted(
        (s * width, n) for n, s in zip(tensor.shape, tensor.stride(), strict=True) if n > 1 and s
    ):
        if not dimensions and stride <= width:
            width += (count - 1) * stride
        else:
            dimensions.append((stride, count))

    def runs(level, base):
        if level < 0:
            yield base, base + width
        else:
            stride, count = dimensions[level]
            yield from merge(*(runs(level - 1, base + i * stride) for i in range(count)))

    return runs(len(dimensions) - 1, tensor.data_ptr())


def overlaps(a, b):
    if a.device != b.device or not a.numel() or not b.numel():
        return False
    if a.data_ptr() >= b.data_ptr() + span(b) * b.element_size() or (
        b.data_ptr() >= a.data_ptr() + span(a) * a.element_size()
    ):
        return False
    left, right = iter(_intervals(a)), iter(_intervals(b))
    x, y = next(left, None), next(right, None)
    while x is not None and y is not None:
        if x[0] < y[1] and y[0] < x[1]:
            return True
        if x[1] <= y[0]:
            x = next(left, None)
        else:
            y = next(right, None)
    return False


def overlapping_pairs(tensors, *, overlap=None, exclude=None):
    """Exact aliases after an O(n log n + candidates) byte-extent sweep."""
    overlap = overlaps if overlap is None else overlap
    extents = []
    for i, tensor in enumerate(tensors):
        if tensor.numel():
            start = tensor.data_ptr()
            device = (tensor.device.type, tensor.device.index or 0)
            extents.append((device, start, start + span(tensor) * tensor.element_size(), i))
    active, max_end, previous = [], 0, None
    for device, start, end, i in sorted(extents):
        if device != previous or start >= max_end:
            active = []
            max_end = end
        else:
            active = [(stop, j) for stop, j in active if stop > start]
        for _, j in active:
            if exclude is not None and exclude(min(i, j), max(i, j)):
                continue
            if overlap(tensors[j], tensors[i]):
                yield min(i, j), max(i, j)
        active.append((end, i))
        max_end = max(max_end, end)
        previous = device


@dataclass(frozen=True)
class Layout:
    """Element pitches and vector-load alignment, all specialization constants."""

    checkpoint: int
    history: int
    canonical: int
    source_history: int
    indices: int = 1
    pair: int = 1
    positions: int = 1
    accepted: int = 1
    snapshots: int = 1
    offsets: int = 1
    valid: int = 1
    token: int = 0
    out_batch: int = 0
    out_token: int = 0
    x_alignment: int = 16
    checkpoint_alignment: int = 256
    history_alignment: int = 256
    out_alignment: int = 16


def layout(g, state, kw):
    def pitch(name, default=1, axis=0):
        t = kw.get(name)
        return default if t is None else t.stride(axis)

    x = kw.get("projected")
    out = kw.get("out")
    pairs = kw.get("pairs")
    alignment = 16
    if x is not None:
        alignment = gcd(16, x.data_ptr())
        for stride in x.stride()[:-1]:
            alignment = gcd(alignment, stride * x.element_size())
    return Layout(
        state.checkpoint.stride(0),
        state.conv_history.stride(0),
        pitch("canonical_state", g.v_heads * 16384),
        pitch("history", 3 * g.conv_dim),
        pitch("slot_indices") if pairs is None else pairs.stride(0),
        1 if pairs is None else pairs.stride(1),
        pitch("positions", pitch("next_positions")),
        pitch("accepted_lengths"),
        pitch("snapshot_rows"),
        pitch("offsets", pitch("snapshot_offsets")),
        pitch("snapshot_valid", axis=1 if kw.get("status_slot") is not None else 0),
        x.stride(1) if x is not None and x.ndim == 3 else g.proj_width,
        out.stride(0) if out is not None else g.value_width,
        out.stride(1) if out is not None and out.ndim == 3 else g.value_width,
        alignment,
        gcd(256, state.checkpoint.data_ptr()),
        gcd(256, state.conv_history.data_ptr()),
        16 if out is None else gcd(16, out.data_ptr()),
    )
