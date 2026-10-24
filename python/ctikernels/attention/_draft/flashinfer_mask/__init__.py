# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FA2 DRAFT with a device-packed symmetric/causal mask.

ABI studied in FlashInfer data/include/flashinfer/attention/{variants,prefill,
mask}.cuh and quantization/packbits.py, identical mask layout on 0.6.11.post3,
0.7.0.post1 and 0.7.1.
"""

import torch

from ..._flashinfer import FlashInferFixedQueryAttention
from .. import flashinfer
from ..flashinfer import check_support as check_support


def retained_capacity(*, max_kv_tokens, window, query_length, page_size, **kwargs):
    if window is None:
        return max_kv_tokens
    return min(max_kv_tokens, window - 1 + query_length + page_size - 1)


def mask_layout(*, batch_size, query_length, max_kv_tokens, page_size, window, **kwargs):
    """Return (bytes/request, aligned mask bytes, aligned int32 indptr bytes).

    The earliest visible key is L-Q-W+1. Whole-page trimming retains at most
    W-1+Q+P-1 keys, independently of C. The per-tile scheduling bound W-1+tile
    does not bound this CSR when Q exceeds the native query tile.
    """
    span = retained_capacity(
        max_kv_tokens=max_kv_tokens,
        window=window,
        query_length=query_length,
        page_size=page_size,
    )
    stride = (query_length * span + 7) // 8
    size = batch_size * stride
    return stride, (size + 255) // 256 * 256, flashinfer.fixed_query_bytes(batch_size)


def check_capacity(*, causal=False, **configuration):
    # The schedule is the same for either mask; CUSTOM supplies causality and
    # the right bound. Retain the existing native schedule admission contract.
    native = configuration | {"max_kv_tokens": retained_capacity(**configuration), "window": None}
    flashinfer.check_capacity(causal=True, **native)
    stride, _, _ = mask_layout(**configuration)
    # Native mask_indptr stores byte offsets, NOT bit offsets. Q*span can
    # exceed int32: FlashInfer and our packer both widen bit arithmetic to 64.
    if configuration["batch_size"] * stride > 2**31 - 1:
        raise NotImplementedError("FlashInfer custom-mask byte indptr exceeds signed int32")


def workspace_size(**configuration):
    check_capacity(**configuration)
    floating, integer = FlashInferFixedQueryAttention.workspace_size(
        **{
            key: configuration[key]
            for key in (
                "batch_size",
                "query_length",
                "num_qo_heads",
                "num_kv_heads",
                "head_dim",
                "device",
            )
        }
    )
    _, mask_bytes, indptr_bytes = mask_layout(**configuration)
    return floating, integer + flashinfer.fixed_query_bytes(
        configuration["batch_size"]
    ) + mask_bytes + indptr_bytes


class Provider(flashinfer.Provider):
    """Reuse FA2's plan/run; prepare a trimmed CSR then its custom mask."""

    _trim_window = True

    # Base prepare rechecks native scheduling geometry. Mask storage was checked
    # at construction; its capacity does not depend on the native query tile.
    def _check_capacity(self, **configuration):
        configuration["window"] = None if self._trim_window_left < 0 else self._trim_window_left + 1
        check_capacity(**configuration)

    def __init__(self, float_workspace, int_workspace, **kwargs):
        batch = kwargs["last_page_lengths"].numel()
        window_left = kwargs["window_left"]
        stride, mask_bytes, indptr_bytes = mask_layout(
            batch_size=batch,
            query_length=kwargs["query_length"],
            max_kv_tokens=kwargs["prepare_inputs"][-1],
            page_size=kwargs["page_size"],
            window=None if window_left < 0 else window_left + 1,
        )
        base_bytes = int_workspace.numel() - mask_bytes - indptr_bytes
        self._trim_window_left = window_left
        # CUSTOM runs every retained key; disable the native per-tile window
        # restriction. The mask supplies both window bounds and causality.
        kwargs["window_left"] = -1
        super().__init__(float_workspace, int_workspace.narrow(0, 0, base_bytes), **kwargs)
        from flashinfer.utils import MaskMode

        self._mask_mode = MaskMode.CUSTOM.value
        self._mask_stride = stride
        self._custom_mask = int_workspace.narrow(0, base_bytes, batch * stride)
        self._mask_indptr = int_workspace.narrow(0, base_bytes + mask_bytes, (batch + 1) * 4).view(
            torch.int32
        )

    def prepare(self):
        super().prepare()
        from .prepare import pack_mask

        tiles = (self._mask_stride + 255) // 256
        pack_mask[(self._batch_size * tiles,)](
            self._kv_indptr,
            self._native._paged_kv_last_page_len_buf,
            self._custom_mask,
            self._mask_indptr,
            BATCH=self._batch_size,
            QUERY=self._query_length,
            PAGE=self._page_size,
            STRIDE=self._mask_stride,
            BYTE_TILES=tiles,
            CAUSAL=self._causal,
            WINDOWED=self._trim_window_left >= 0,
            WINDOW_LEFT=self._trim_window_left,
        )
