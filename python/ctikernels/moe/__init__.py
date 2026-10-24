# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen MoE operations: BF16 and W4A8 (MXFP4 x MXFP8) fused experts with optional
expert-parallel routing, and the frozen MoE-block training op."""

from ._align import align_allocations, moe_align_block_size_local
from ._route import hash_route, route, to_local_topk_ids
from .inference import Experts, MoEBlock, MXFP4Experts, fused_experts, pack_mxfp4_experts
from .training import moe_block, moe_block_training_bytes

__all__ = [
    "Experts",
    "MXFP4Experts",
    "MoEBlock",
    "align_allocations",
    "fused_experts",
    "hash_route",
    "moe_align_block_size_local",
    "moe_block",
    "moe_block_training_bytes",
    "pack_mxfp4_experts",
    "route",
    "to_local_topk_ids",
]
