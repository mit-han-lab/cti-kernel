# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DSV4 CUDA glue built on first use through tvm-ffi: the deterministic row top-512
(``csrc/jit/dsv4_topk.cu``: ties to the lower index, entries in a fixed order, -1 after
min(length, 512), and optionally the entry ids too) and the in-place inverse rotary (``csrc/jit/dsv4_rope.cu``)."""

from __future__ import annotations

import functools


@functools.cache
def _module():
    from ...utils import load_jit

    return load_jit(
        "dsv4_glue",
        cuda_files=["dsv4_topk.cu", "dsv4_rope.cu"],
        cuda_wrappers=[("topk", "Dsv4TopK::run"), ("inverse_rope", "Dsv4InverseRope::run")],
    )


def topk_rows(scores, lengths, page_table, row_to_batch, out, entries, *, page_rows):
    batch = row_to_batch if row_to_batch is not None else lengths[:0]
    _module().topk(
        scores, lengths, page_table, batch, out, out[:0] if entries is None else entries, page_rows
    )
    return out


def inverse_rope(o, table, positions):
    _module().inverse_rope(o, table, positions)
    return o


__all__ = ["inverse_rope", "topk_rows"]
