# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host-only checks for borrowed inference buffers."""

from collections.abc import Iterable

import torch


def reject_grad(*tensors: torch.Tensor) -> None:
    if any(t.requires_grad for t in tensors):
        raise ValueError("inference operations do not support tensors requiring gradients")


def _byte_span(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return (
        1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))
    ) * tensor.element_size()


def disjoint(writable: Iterable[torch.Tensor], readonly: Iterable[torch.Tensor] = ()) -> None:
    """Check byte intervals, allowing disjoint views of one allocation."""
    writable = tuple(t for t in writable if t.numel())
    readonly = tuple(t for t in readonly if t.numel())
    for i, left in enumerate(writable):
        begin, end = left.data_ptr(), left.data_ptr() + _byte_span(left)
        for right in writable[i + 1 :] + readonly:
            other = right.data_ptr()
            if left.device == right.device and begin < other + _byte_span(right) and other < end:
                raise ValueError("writable buffers must not overlap each other or read-only inputs")


def byte_span(t: torch.Tensor) -> tuple[int, int]:
    """Conservative [begin, end) byte range from sizes and strides.

    Covers every byte a strided view can touch, including inter-row gaps, so
    an interval test on spans never misses an overlap (it may report one).
    """
    begin = t.data_ptr()
    if t.numel() == 0:
        return begin, begin
    low = high = 0
    for size, stride in zip(t.shape, t.stride(), strict=True):
        extent = (size - 1) * stride
        if extent < 0:
            low += extent
        else:
            high += extent
    item = t.element_size()
    return begin + low * item, begin + (high + 1) * item


def spans_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    (a0, a1), (b0, b1) = byte_span(a), byte_span(b)
    return a0 < b1 and b0 < a1


def internal_overlap(t: torch.Tensor) -> bool:
    """True if two elements of t may share memory (zero-stride, self-overlapping).

    Exact for zero strides; a layout the nesting test cannot prove disjoint
    counts as overlap.
    """
    dims = sorted(
        ((abs(stride), size) for size, stride in zip(t.shape, t.stride(), strict=True) if size > 1),
    )
    reach = 0
    for stride, size in dims:
        if stride == 0 or stride <= reach:
            return True
        reach += (size - 1) * stride
    return False


def copy_plan(dst: torch.Tensor, src: torch.Tensor):
    """2D byte-copy plan for equal-shape, equal-dtype, non-negative-stride views.

    Returns (width, height, dst_pitch, src_pitch, starts): one 2D copy of
    `height` rows of `width` bytes per (dst_offset, src_offset) in starts.
    Dims are coalesced jointly, so dense views take one copy. Single-row
    copies carry pitch = width, and every plan satisfies width <= both
    pitches (the CUDA 2D-copy rule); a plan that cannot raises ValueError.
    """
    item = dst.element_size()
    dims = [
        (size, d, s)
        for size, d, s in zip(dst.shape, dst.stride(), src.stride(), strict=True)
        if size != 1
    ]
    merged = []
    for size, d, s in dims:
        if merged and merged[-1][1] == d * size and merged[-1][2] == s * size:
            merged[-1] = (merged[-1][0] * size, d, s)
        else:
            merged.append((size, d, s))
    width = item
    if merged and merged[-1][1] == 1 and merged[-1][2] == 1:
        width *= merged.pop()[0]
    height, dst_pitch, src_pitch = 1, width, width
    if merged:
        height, d, s = merged.pop()
        dst_pitch, src_pitch = d * item, s * item
    if height > 1 and min(dst_pitch, src_pitch) < width:
        # A broadcast (zero or short stride) row: copy rows one by one.
        merged.append((height, dst_pitch // item, src_pitch // item))
        height = 1
    if height == 1:
        dst_pitch = src_pitch = width
    if width > dst_pitch or width > src_pitch or min(dst_pitch, src_pitch) < 0:
        raise ValueError(f"invalid copy plan: width {width}, pitches {dst_pitch}/{src_pitch}")
    starts = [(0, 0)]
    for size, d, s in merged:
        if d < 0 or s < 0:
            raise ValueError("raw copies need non-negative strides")
        starts = [(a + i * d * item, b + i * s * item) for a, b in starts for i in range(size)]
    return width, height, dst_pitch, src_pitch, starts


class CopyPlan:
    """A validated raw copy; run() only issues the planned copies."""

    __slots__ = ("cuda", "device", "dst", "src", "width", "height", "pitches", "starts")

    def __init__(self, dst: torch.Tensor, src: torch.Tensor):
        width, height, dst_pitch, src_pitch, starts = (
            copy_plan(dst, src) if dst.numel() else (0, 0, 0, 0, [])
        )
        self.cuda, self.device = dst.device.type == "cuda", dst.device
        self.dst, self.src = dst.data_ptr(), src.data_ptr()
        self.width, self.height, self.pitches = width, height, (dst_pitch, src_pitch)
        self.starts = starts

    def run(self) -> None:
        if not self.cuda:
            import ctypes

            for a, b in self.starts:
                for row in range(self.height):
                    ctypes.memmove(
                        self.dst + a + row * self.pitches[0],
                        self.src + b + row * self.pitches[1],
                        self.width,
                    )
            return
        from cuda.bindings import runtime

        stream = torch.cuda.current_stream(self.device).cuda_stream
        kind = runtime.cudaMemcpyKind.cudaMemcpyDefault
        for a, b in self.starts:
            if self.height == 1:
                (error,) = runtime.cudaMemcpyAsync(
                    self.dst + a, self.src + b, self.width, kind, stream
                )
            else:
                (error,) = runtime.cudaMemcpy2DAsync(
                    self.dst + a,
                    self.pitches[0],
                    self.src + b,
                    self.pitches[1],
                    self.width,
                    self.height,
                    kind,
                    stream,
                )
            if int(error) != 0:
                raise RuntimeError(f"raw device copy failed: {error}")


def plan_copy(dst: torch.Tensor, src: torch.Tensor) -> CopyPlan:
    """Plan and validate src -> dst before any collective is issued or recorded."""
    return CopyPlan(dst, src)


def raw_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy src into dst as raw device memory on the current stream.

    Bypasses torch tensor rules (inference mode, autograd, version counters).
    The caller has already checked dtype, shape, device and that dst has no
    internal overlap. CUDA copies are graph-capturable memcpy nodes. Copies
    that follow a collective are planned with plan_copy before it.
    """
    plan_copy(dst, src).run()
