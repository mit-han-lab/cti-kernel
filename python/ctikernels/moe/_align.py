# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Route alignment: our Triton kernel at every route count (imported on first use)."""

import torch


def _validate_routes(topk_ids, expert_map):
    for name, tensor, ndim in (("topk_ids", topk_ids, 2), ("expert_map", expert_map, 1)):
        if tensor.ndim != ndim or tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous int32 with {ndim} dimensions")
    if expert_map.numel() == 0:
        raise ValueError("expert_map must contain at least one global expert")
    if not topk_ids.is_cuda or not expert_map.is_cuda or topk_ids.device != expert_map.device:
        raise ValueError("topk_ids and expert_map must be CUDA tensors on the same device")


def align_allocations(n, block_size, num_local_experts, *, capacity_experts=None,
                      invalid_bucket=False):  # fmt: skip
    """Bytes of each allocation ``moe_align_block_size_local`` holds for ``n`` routes, as
    requested (unrounded; the CUDA caching allocator holds each in 512-byte granules);
    routing-independent: sized for every expert's last block partial."""
    from .triton._slot_layout import align_allocations as triton_allocations

    return triton_allocations(
        n, block_size, num_local_experts, capacity_experts=capacity_experts,
        invalid_bucket=invalid_bucket,
    )  # fmt: skip


def moe_align_block_size_local(
    topk_ids: torch.Tensor,
    block_size: int,
    expert_map: torch.Tensor,
    num_local_experts: int,
    *,
    capacity_experts: int | None = None,
    identity: bool = False,
    invalid_bucket: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group LOCAL routes, padding each expert with sentinel ``topk_ids.numel()``.

    IDs are global, or -1 for padding; map entries are local indices or -1.
    Callers guarantee those value ranges. No remote route gets a slot or block.
    ``identity=True`` promises ``expert_map`` is ``arange(num_local_experts)`` (EP1), so the
    kernel skips its loads.
    ``invalid_bucket=True`` instead keeps -1 and out-of-range routes: they fill trailing
    blocks of expert -1 (zero rows from the grouped GEMM), the provider layout's semantics;
    it needs ``capacity_experts > num_local_experts``.
    Route order is kept inside each expert (bitwise deterministic); the kernels launch on
    torch's current CUDA stream.
    """
    if not isinstance(num_local_experts, int) or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    _validate_routes(topk_ids, expert_map)
    if invalid_bucket and (capacity_experts is None or capacity_experts <= num_local_experts):
        raise ValueError("invalid_bucket needs an extra bucket (capacity_experts)")
    from .triton._slot_layout import align_block_size

    return align_block_size(
        topk_ids,
        block_size,
        num_local_experts,
        None if identity else expert_map,
        capacity_experts=capacity_experts,
        invalid_bucket=invalid_bucket,
    )
