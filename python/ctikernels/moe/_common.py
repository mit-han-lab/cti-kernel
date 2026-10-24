# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DSV4 ABI, metadata-only validation and native checkpoint layout helpers."""

from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class Geometry:
    hidden: int = 4096
    intermediate: int = 2048
    experts: int = 256
    vocab: int = 129280

    def __post_init__(self):
        if self.hidden <= 0 or self.intermediate <= 0 or self.hidden % 32 or self.intermediate % 32:
            raise ValueError("hidden and intermediate must be positive multiples of 32")
        if self.experts < 6 or self.vocab < 1:
            raise ValueError("at least six experts and a nonempty vocabulary required")


@dataclass(frozen=True)
class ExpertWeights:
    w13: torch.Tensor
    w13_scales: torch.Tensor
    w2: torch.Tensor
    w2_scales: torch.Tensor

    def tensors(self):
        return (self.w13, self.w13_scales, self.w2, self.w2_scales)


class RouterOutput(NamedTuple):
    local_ids: torch.Tensor
    weights: torch.Tensor
    global_ids: torch.Tensor


class Quantized(NamedTuple):
    values: torch.Tensor
    scales: torch.Tensor


def frozen(*tensors):
    # Must precede device work, including output allocation and provider preparation.
    if any(t is not None and t.requires_grad for t in tensors):
        raise ValueError("DSV4 operations are inference-only; requires_grad must be False")


def tensor(t, shape, dtype, device, name):
    if t.shape != torch.Size(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(t.shape)}")
    if t.dtype != dtype:
        raise TypeError(f"{name}: expected {dtype}, got {t.dtype}")
    if t.device != device or not t.is_contiguous():
        raise ValueError(f"{name}: expected contiguous tensor on {device}")


def validate_quantized(q, x):
    """Metadata only; the caller promises q quantizes exactly x's BF16 bytes."""
    if not isinstance(q, Quantized):
        raise TypeError("quantized must be a Quantized")
    frozen(q.values, q.scales)
    tensor(q.values, x.shape, torch.float8_e4m3fn, x.device, "quantized.values")
    tensor(q.scales, (x.shape[0], x.shape[1] // 32), torch.uint8, x.device, "quantized.scales")


def validate_weights(w, geometry):
    frozen(*w.tensors())
    if w.w13.ndim != 3:
        raise ValueError("w13 must have three dimensions")
    e = w.w13.shape[0]
    if not 1 <= e <= geometry.experts:
        raise ValueError("invalid local expert count")
    h, i = geometry.hidden, geometry.intermediate
    for t, shape, name in zip(
        w.tensors(),
        ((e, 2 * i, h // 2), (e, 2 * i, h // 32), (e, h, i // 2), (e, h, i // 32)),
        ("w13", "w13_scales", "w2", "w2_scales"),
        strict=True,
    ):
        tensor(t, shape, torch.uint8, w.w13.device, name)


def unpack_e2m1(packed):
    """Return raw nibble codes; low nibble is even K (negative zero preserved)."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed E2M1 must be uint8")
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def pack_e2m1(codes):
    """Pack raw codes in [0,15]; code range is a caller precondition."""
    if codes.dtype != torch.uint8 or codes.shape[-1] % 2:
        raise ValueError("E2M1 codes must be uint8 with an even last dimension")
    return (codes[..., 0::2] & 15) | ((codes[..., 1::2] & 15) << 4)


def encode_ue8m0(exponents):
    """Encode integer base-two exponents, clamping to the finite UE8M0 range."""
    return (exponents.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)


def decode_ue8m0(scales, dtype=torch.float32):
    if scales.dtype != torch.uint8:
        raise TypeError("UE8M0 storage must be uint8")
    values = torch.ldexp(torch.ones_like(scales, dtype=dtype), scales.to(torch.int32) - 127)
    return torch.where(scales == 255, float("nan"), values)


def decode_e2m1(packed, scales, dtype=torch.float32):
    codes = unpack_e2m1(packed).long()
    table = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=packed.device, dtype=dtype
    )
    values = table[codes & 7] * torch.where(codes < 8, 1.0, -1.0).to(dtype)
    return values * decode_ue8m0(scales, dtype).repeat_interleave(32, dim=-1)


def decode_mxfp8(q, dtype=torch.float32):
    return q.values.to(dtype) * decode_ue8m0(q.scales, dtype).repeat_interleave(32, dim=-1)


def interleave_scales_(scales, rows=None):
    """uint8 [E, N, K/32] MXFP4 weight scales into FlashInfer's 128x4 interleave
    (``block_scale_interleave``), in place, one expert at a time; ``rows`` first permutes each
    expert's rows. Per expert, row n and group g land at byte
    ((n // 128) * (K/128) + g // 4) * 512 + (n % 32) * 16 + (n % 128) // 32 * 4 + g % 4.
    N % 128 == 0 and K/32 % 4 == 0 make it a permutation (no padding)."""
    e, n, groups = scales.shape
    if n % 128 or groups % 4:
        raise ValueError("interleaved scales need N % 128 == 0 and K/32 % 4 == 0")
    for s in scales:
        src = s if rows is None else s.index_select(0, rows)
        s.copy_(src.view(n // 128, 4, 32, groups // 4, 4).permute(0, 3, 2, 1, 4).reshape(n, groups))
    return scales


def deinterleave_scales(scales):
    """A canonical [E, N, K/32] copy of scales in the 128x4 interleave."""
    e, n, groups = scales.shape
    blocks = scales.view(e, n // 128, groups // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
    return blocks.reshape(e, n, groups)
