# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Host-only scratch layouts and aligned views; returned outputs never use these views."""

from math import prod

import torch


class Layout:
    def __init__(self):
        self.slots = {}
        self.alignments = {}
        self.nbytes = 0

    def add(self, name, shape, dtype=torch.float32, *, alignment=16):
        size = prod(shape) * dtype.itemsize
        self.slots[name] = (self.nbytes, size, shape, dtype)
        self.alignments[name] = alignment
        # The caller promises only 16-byte alignment, including for child views.
        self.nbytes += (size + 15) // 16 * 16 + alignment - 16

    def include(self, name, layout):
        self.add(name, (layout.nbytes // 4,))

    def bind(self, workspace, device, *operands):
        if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.float32:
            raise ValueError(
                "workspace must be a contiguous, aligned FP32 CUDA buffer of workspace_bytes"
            )
        if (
            not workspace.is_cuda
            or workspace.device != device
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 16
            or workspace.numel() * 4 < self.nbytes
            or workspace.requires_grad
        ):
            raise ValueError(
                "workspace must be a contiguous, aligned FP32 CUDA buffer of workspace_bytes"
            )
        pointer = workspace.untyped_storage().data_ptr()
        if pointer and any(
            isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() == pointer
            for t in operands
        ):
            raise ValueError("workspace must not share storage with inputs, weights, scales or out")
        return Views(self, workspace)


class Views:
    def __init__(self, layout, workspace):
        self.layout = layout
        self.storage = workspace.view(-1).view(torch.uint8)

    def get(self, name, dtype=None):
        offset, size, shape, stored_dtype = self.layout.slots[name]
        offset += -(self.storage.data_ptr() + offset) % self.layout.alignments[name]
        return self.storage[offset : offset + size].view(dtype or stored_dtype).view(shape)

    def child(self, name, layout):
        return Views(layout, self.get(name))

    def flatten(self, x):
        try:
            return x.view(-1, x.shape[-1])
        except RuntimeError:
            rows = self.get("x", x.dtype)
            rows.view(x.shape).copy_(x)
            return rows


def maximum(max_m, requirement):
    if type(max_m) is not int or max_m < 1:
        raise ValueError("max_m must be positive")
    # Split-K and provider switches are non-monotonic. Inspect every M; this is
    # metadata only, with no dependency imports, compilation or CUDA work.
    return max(requirement(m) for m in range(1, max_m + 1))
