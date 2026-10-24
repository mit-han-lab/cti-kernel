# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host phase bindings; address checks retain no invocation tensors."""

from bisect import bisect_left
from dataclasses import fields

from ._layout import span


def intervals(spans):
    """Merge occupied extents for logarithmic candidate queries."""
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return tuple(a for a, _ in merged), tuple(b for _, b in merged)


def possible_overlap(persistent, writable, dynamic):
    """Conservative extent test, O(k log n + k log k).

    Read/read aliases are allowed. Pitched holes are resolved by the full API
    validator on a positive result, preserving exact overlap errors and exemptions.
    """
    all_end = write_end = -1
    for start, end, write in sorted(dynamic):
        if end <= start:
            continue
        starts, ends = persistent if write else writable
        index = bisect_left(starts, end) - 1
        if index >= 0 and ends[index] > start:
            return True
        if (all_end if write else write_end) > start:
            return True
        all_end = max(all_end, end)
        if write:
            write_end = max(write_end, end)
    return False


def descriptor(tensor):
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


class HostPlan:
    def __init__(self, state, weights, plan, per_call, pointers, config=None):
        self.state, self.weights, self.plan = state, weights, plan
        self.config = config
        self.tensors = tuple(getattr(state, f.name) for f in fields(state)) + weights
        self.pointers = tuple(t.data_ptr() for t in self.tensors)
        self.descriptors = tuple(descriptor(t) for t in self.tensors)
        spans = [(t.data_ptr(), t.data_ptr() + span(t) * t.element_size()) for t in self.tensors]
        self.persistent = intervals(spans)
        self.writable = intervals(spans[: -len(weights)])
        writes = {"out", "pending.raw", "pending.metadata", "snapshot_valid", "arrivals"}
        self.extents = tuple(
            (span(t) * t.element_size(), name in writes) for name, t in per_call.items()
        )
        self.last_pointers = pointers

    def unchanged(self):
        # set_/resize_/transpose_ can change storage or layout on the same object.
        return all(
            t.data_ptr() == pointer and descriptor(t) == desc
            for t, pointer, desc in zip(self.tensors, self.pointers, self.descriptors, strict=True)
        )

    def addresses_valid(self, pointers):
        if pointers == self.last_pointers:
            return True
        if possible_overlap(
            self.persistent,
            self.writable,
            [(p, p + size, write) for p, (size, write) in zip(pointers, self.extents, strict=True)],
        ):
            return False
        self.last_pointers = pointers
        return True
