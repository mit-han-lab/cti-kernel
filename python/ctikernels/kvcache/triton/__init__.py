# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""One Triton launch copies rows of every tensor in a fixed table."""

import torch
import triton

from ._kernels import copy_rows

BLOCK = 1024
_MAX_COPIES = 65535  # grid dimension 1


class Provider:
    name = "triton"

    def __init__(self, tensors):
        table, first = [], 0
        for t in tensors:
            stride, row = t.stride(0) * t.element_size(), t[0].numel() * t.element_size()
            # Per tensor: the widest word every address, row stride and row size divides; its
            # class id is _kernels.VEC16 (16 bytes, moved as 8-byte units) .. WORD1 (1 byte).
            size = next(
                s for s in (16, 8, 4, 2, 1) if all(v % s == 0 for v in (t.data_ptr(), stride, row))
            )
            word, unit = 5 - size.bit_length(), min(size, 8)
            table.append((t.data_ptr(), stride // unit, row // unit, t.shape[0], word, first))
            first += triton.cdiv(row // unit, BLOCK)
        self.tensors = tensors  # the table addresses their storage
        self.table = torch.tensor(table, dtype=torch.int64).to(tensors[0].device)
        self.count = len(tensors)
        self.log = (self.count - 1).bit_length()
        self.chunks = first
        if self.chunks >= 2**31:
            raise ValueError("RowCopy tensors span more than 2**31 - 1 chunks")
        self.warm = set()

    def _launch(self, sources, targets):
        copies = sources.numel()
        if copies > _MAX_COPIES:
            raise ValueError(f"RowCopy copies at most {_MAX_COPIES} rows per call")
        if sources.dtype not in self.warm and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"RowCopy would compile its {sources.dtype} launch inside a CUDA graph capture; "
                "call warm_up (or one eager call) with this index dtype first"
            )
        with torch.cuda.device(sources.device):
            copy_rows[(self.chunks, copies)](
                self.table, sources, targets, self.count, LOG=self.log, BLOCK=BLOCK
            )
        self.warm.add(sources.dtype)

    def warm_up(self, index_dtypes):
        for dtype in index_dtypes:
            rows = torch.full((1,), -1, dtype=dtype, device=self.table.device)
            self._launch(rows, rows)

    def __call__(self, sources, targets):
        self._launch(sources, targets)
