# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Greedy LM-head row reduction, public as ``ctikernels.sampling.RowArgmax``.

Implementation notes: CTAs run position-major in chain mode. For Q > 32 with logits larger
than L2 on integrated devices, each later row streams through one CTA that stops once an
earlier mismatch is known; discrete devices use multi-wave splits.
"""

import torch


class RowArgmax:
    """First argmax of every logits row; a CPU reference or the CUDA CuTe provider by device.

    ``op(logits, out, workspace, *, proposals=None, block_size=None)`` takes logits
    bf16/fp16/fp32 ``[R, V]`` with unit column stride and row stride >= V, out int32/int64
    contiguous ``[R]``, and a contiguous 8-byte-aligned uint8 workspace of at least
    ``workspace_size(R)`` (24*R) bytes; none may alias. NaN beats every number; ties (signed
    zeros included) take the first index, as in torch.argmax.

    Zero the workspace once at allocation; every completed call leaves it zero. Calls sharing
    it must be ordered on a stream (or by events), never concurrent. A CUDA call allocates no
    device memory, never synchronizes the host and launches one kernel. Before graph capture,
    ``warm_up`` each logits/out/proposals dtype at the largest batch: that compiles every
    variant any batch of 1..B requests (same Q and V) can select. R, V, strides, Q and the
    split count are runtime values.

    Chain mode (greedy verify): proposals int32/int64 ``[B, Q-1]`` with unit inner stride
    unless empty, block_size Q >= 1, logits rows ordered b*Q+i, R = B*Q < 2^31. Rows up to and
    including each request's first i with ``out[b*Q+i] != proposals[b, i]`` (the last row when
    every proposal matches) hold the first argmax. A later row holds it or -1 (rows after a
    known mismatch skip their reads); the defined rows never depend on that timing. Q=1
    computes the bonus argmax only.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            from .cute.argmax import Provider
        elif self.device.type == "cpu":
            from .reference.argmax import Provider
        else:
            raise ValueError("RowArgmax supports CPU and CUDA")
        self._provider = Provider(self.device)

    @staticmethod
    def workspace_size(row_capacity):
        """Bytes for capacity rows: uint64 key, uint32 counter, uint32 skip count per row,
        plus two uint32 chain words per request (at most one request per row)."""
        if (
            not isinstance(row_capacity, int)
            or isinstance(row_capacity, bool)
            or not 0 <= row_capacity < 2**31
        ):
            raise ValueError("row capacity must be a nonnegative int32")
        return 24 * row_capacity

    @property
    def provider(self):
        return "cute" if self.device.type == "cuda" else "reference"

    def _check(self, logits, out, workspace, proposals=None, block_size=None):
        if (proposals is None) != (block_size is None):
            raise ValueError("chain mode needs both proposals and block_size")
        tensors = [("logits", logits), ("out", out), ("workspace", workspace)]
        if proposals is not None:
            tensors.append(("proposals", proposals))
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.requires_grad:
                raise ValueError(f"{name} requires grad")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on the operation device")
        if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("logits must be bf16, fp16 or fp32")
        if logits.ndim != 2 or not 1 <= logits.shape[1] < 2**31:
            raise ValueError("logits must be [R,V] with 1 <= V < 2^31")
        rows, vocab = logits.shape
        if rows >= 2**31:
            raise ValueError("B*Q (logits rows) must be <= 2147483647 (int32 row indices)")
        if logits.stride(1) != 1 or logits.stride(0) < vocab:
            raise ValueError("logits must have contiguous columns and nonoverlapping rows")
        if out.dtype not in (torch.int32, torch.int64):
            raise TypeError("out must be int32 or int64")
        if out.shape != (rows,) or out.stride() != (1,):
            raise ValueError("out must be contiguous [R]")
        if workspace.dtype != torch.uint8:
            raise TypeError("workspace must be uint8")
        if workspace.ndim != 1 or not workspace.is_contiguous():
            raise ValueError("workspace must be a contiguous vector")
        if workspace.numel() < self.workspace_size(rows) or workspace.data_ptr() % 8:
            raise ValueError("workspace is too small or misaligned")
        if proposals is not None:
            if (
                not isinstance(block_size, int)
                or isinstance(block_size, bool)
                or not 1 <= block_size < 2**31
            ):
                raise ValueError(
                    "block_size must be an int >= 1 and <= 2147483647 (int32 positions)"
                )
            if rows % block_size:
                raise ValueError("logits rows must be a multiple of block_size")
            if proposals.dtype not in (torch.int32, torch.int64):
                raise TypeError("proposals must be int32 or int64")
            if proposals.shape != (rows // block_size, block_size - 1) or (
                proposals.numel() and proposals.stride(1) != 1
            ):
                raise ValueError("proposals must be [R/Q, Q-1] with unit inner stride")
        # Include row padding in the conservative input address span.
        spans = [
            (
                logits.data_ptr(),
                ((rows - 1) * logits.stride(0) + vocab) * logits.element_size() if rows else 0,
            ),
            (out.data_ptr(), out.numel() * out.element_size()),
            (workspace.data_ptr(), workspace.numel()),
        ]
        if proposals is not None and proposals.numel():
            spans.append(
                (
                    proposals.data_ptr(),
                    ((proposals.shape[0] - 1) * proposals.stride(0) + proposals.shape[1])
                    * proposals.element_size(),
                )
            )
        for i, (base, size) in enumerate(spans):
            for other, extent in spans[i + 1 :]:
                if size and extent and base < other + extent and other < base + size:
                    raise ValueError("logits, out, workspace and proposals must not alias")

    def __call__(self, logits, out, workspace, *, proposals=None, block_size=None):
        self._check(logits, out, workspace, proposals, block_size)
        if logits.shape[0]:
            if proposals is None:
                self._provider.run(logits, out, workspace)
            else:
                self._provider.run(logits, out, workspace, proposals, block_size)

    def warm_up(self, logits, out, workspace, *, proposals=None, block_size=None):
        """Validate and compile this variant without launching or touching storage."""
        self._check(logits, out, workspace, proposals, block_size)
        if logits.shape[0]:
            if proposals is None:
                self._provider.warm_up(logits, out)
            else:
                self._provider.warm_up(logits, out, proposals, block_size)
