# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Torch models and debug inverses for the private W4A8 routed layouts.

These helpers may synchronize and allocate; they are never provider fallbacks.
Unrouted pairs have zero codes and neutral (127) scale bytes in debug views.
"""

from dataclasses import dataclass

import torch

from .._common import frozen, tensor


def _pack_bytes(scales):
    if scales.dtype != torch.uint8 or scales.shape[-1] % 4:
        raise ValueError("scales require uint8 and a multiple of four groups")
    groups = scales.reshape(*scales.shape[:-1], scales.shape[-1] // 4, 4).int()
    return groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16) | (groups[..., 3] << 24)


def _unpack_bytes(words):
    if words.dtype != torch.int32:
        raise ValueError("packed scales require int32")
    return torch.stack([(words >> (8 * b)) & 255 for b in range(4)], -1).flatten(-2).byte()


def pack_activation_scales(scales):
    """Pack uint8 [P,K/32] into contiguous int32 [K/128,P]."""
    if scales.ndim != 2:
        raise ValueError("activation scales require [P,K/32]")
    return _pack_bytes(scales).t().contiguous()


def unpack_activation_scales(words):
    """Inverse of pack_activation_scales, including padding rows."""
    if words.ndim != 2:
        raise ValueError("activation words require [K/128,P]")
    return _unpack_bytes(words.t())


def interleave_weight_scales(scales):
    """uint8 [E,N,K/32] into FlashInfer's 128x4 interleave (block_scale_interleave), viewed
    as int32 [E,N,K/128]: per expert [N/128, K/128, 32, 4] words, row n of a block at word
    4 * (n % 32) + (n % 128) // 32. N must be a multiple of 128 (no padding)."""
    frozen(scales)
    e, n, groups = scales.shape
    if n % 128 or groups % 4:
        raise ValueError("interleaved weight scales need N % 128 == 0 and K/32 % 4 == 0")
    blocks = scales.view(e, n // 128, 4, 32, groups // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return blocks.contiguous().view(torch.int32).view(e, n, groups // 4)


@dataclass
class RouteModel:
    expert_offsets: torch.Tensor
    scale_offsets: torch.Tensor
    num_rows: torch.Tensor
    row_token: torch.Tensor
    row_weight: torch.Tensor
    row_pair: torch.Tensor
    pair_row: torch.Tensor
    pair_scale: torch.Tensor


def routing_metadata(local_ids, weights):
    """Stable ascending pair order within experts, retaining duplicate slots.

    IDs outside [0,128) are masked defensively, just as in the device kernels.
    Unused row/pair maps contain -1; only rows below num_rows are meaningful.
    """
    m = len(local_ids)
    tensor(local_ids, (m, 6), torch.int32, weights.device, "local_ids")
    tensor(weights, (m, 6), torch.float32, local_ids.device, "weights")
    device = local_ids.device
    flat = local_ids.flatten()
    live = (flat >= 0) & (flat < 128)
    pairs = torch.arange(6 * m, device=device)[live]
    pairs = pairs[torch.argsort(flat[live], stable=True)]
    counts = torch.bincount(flat[live].long(), minlength=128).int()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).int()))
    scale_offsets = ((offsets[:-1] + 3 * torch.arange(128, device=device)) // 4 * 4).int()
    length = len(pairs)
    row_pair = torch.full((6 * m,), -1, dtype=torch.int32, device=device)
    row_pair[:length] = pairs.int()
    row_token = torch.where(row_pair >= 0, row_pair // 6, -1)
    row_weight = torch.zeros(6 * m, dtype=torch.float32, device=device)
    row_weight[:length] = weights.flatten()[pairs]
    pair_row = torch.full_like(row_pair, -1)
    pair_scale = torch.full_like(row_pair, -1)
    pair_row[pairs] = torch.arange(length, dtype=torch.int32, device=device)
    experts = flat[pairs].long()
    pair_scale[pairs] = scale_offsets[experts] + pair_row[pairs] - offsets[experts]
    return RouteModel(
        offsets,
        scale_offsets,
        offsets[-1:].clone(),
        row_token,
        row_weight,
        row_pair,
        pair_row,
        pair_scale,
    )


def unpack_q0(ws):
    """Return CPU codes [M,6,4096], scales [M,6,128] in original pair order.

    Explicit debug readback, unsuitable for capture. Does not depend on scratch
    pair maps or within-expert order; only the fixed public workspace ABI.
    """
    offsets = ws.expert_offsets.cpu().long()
    starts = ws.scale_offsets.cpu().long()
    length = int(ws.num_rows.cpu()[0])
    counts = offsets[1:] - offsets[:-1]
    experts = torch.repeat_interleave(torch.arange(128), counts)
    scale_rows = starts[experts] + torch.arange(length) - offsets[experts]
    pairs = ws.row_pair[:length].cpu().long()
    codes = torch.zeros((ws.rows, 4096), dtype=torch.uint8)
    scales = torch.full((ws.rows, 128), 127, dtype=torch.uint8)
    codes[pairs] = ws.q0[:length].cpu()
    scales[pairs] = unpack_activation_scales(ws.q0_scales.cpu())[scale_rows]
    return codes.reshape(ws.m, 6, 4096), scales.reshape(ws.m, 6, 128)
