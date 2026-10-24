# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: FlashInfer's fp4_paged_mqa_logits (a CuTe DSL port of DeepGEMM's sm100 FP4 paged
# MQA logits) is the called library.

"""FlashInfer FP4 paged MQA logits over the DSV4 index pages (compute capability 10.x): the
pages are the kernel's fused FP4 KV blocks as they are (64 rows, data then UE8M0 scales)."""

from __future__ import annotations

import functools

import torch

# Releases that ship fp4_paged_mqa_logits (0.7.0.post1 is the core pin).
QUALIFIED_VERSIONS = ("0.7.0.post1", "0.7.1")


@functools.cache
def _library():
    """``(fp4_paged_mqa_logits, compute_paged_mqa_logits_schedule)``, or None."""
    import importlib.metadata

    try:
        if importlib.metadata.version("flashinfer-python") not in QUALIFIED_VERSIONS:
            return None
        from flashinfer.attn_scores import (
            compute_paged_mqa_logits_schedule,
            fp4_paged_mqa_logits,
        )
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return None
    return fp4_paged_mqa_logits, compute_paged_mqa_logits_schedule


def available(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] == 10 and _library() is not None


def schedule_buffer(device: torch.device) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.empty((sms + 1, 2), dtype=torch.int32, device=device)


def run(op, cache: torch.Tensor, out: torch.Tensor) -> None:
    """Scores of ``out.shape[0]`` queries whose transform, lengths and block table are in
    ``op``'s scratch; ``out`` has at least ``op.columns`` columns."""
    logits, schedule = _library()
    t = out.shape[0]
    lens = op._lens[:t]
    schedule(lens, out=op._schedule)
    logits(
        op._codes[:t].view(t, 1, 64, 64),
        op._scales[:t].view(torch.int32).view(t, 1, 64),
        cache.view(cache.shape[0], 64, 1, 68),
        op._weights[:t],
        lens,
        op._table[:t],
        op.max_entries,
        output_dtype=torch.float32,
        schedule_meta=op._schedule,
        out=out,
    )
