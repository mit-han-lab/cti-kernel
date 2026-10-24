# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Test-only preparation, independent scale packing and componentwise oracle.

This helper may synchronize/read routes on the host. Production never imports it.
"""

from itertools import permutations
from types import SimpleNamespace

import torch

from ctikernels.moe._common import Quantized, decode_mxfp8, tensor
from ctikernels.moe.cute._w4a8_model import (
    RouteModel,
    pack_activation_scales,
    routing_metadata,
)
from ctikernels.moe.torch._w4a8_tiled_reference import quantize

from ._moe_w4a8_validation import TINY, U32, bf16_ulp


def pack_scales(scales):
    e, n, groups = scales.shape
    values = scales.reshape(e, n, groups // 4, 4).to(torch.int64)
    words = sum(values[..., byte] << (8 * byte) for byte in range(4))
    return words.permute(0, 2, 1).contiguous().to(torch.int32)


def unpack_scales(words):
    bytes_ = torch.stack([(words.to(torch.int64) >> (8 * b)) & 255 for b in range(4)], -1)
    return bytes_.permute(0, 2, 1, 3).flatten(-2).to(torch.uint8)


def workspace(m, k=4096, intermediate=2048, device="cpu"):
    r, p = 6 * m, (6 * m + 384) // 4 * 4

    def empty(shape, dtype):
        return torch.empty(shape, device=device, dtype=dtype)

    return SimpleNamespace(
        m=m,
        rows=r,
        scale_rows=p,
        q0=empty((r, k), torch.uint8),
        q0_scales=empty((k // 128, p), torch.int32),
        q1=empty((r, intermediate), torch.uint8),
        q1_scales=empty((intermediate // 128, p), torch.int32),
        row_token=empty((r,), torch.int32),
        row_weight=empty((r,), torch.float32),
        row_pair=empty((r,), torch.int32),
        expert_offsets=empty((129,), torch.int32),
        scale_offsets=empty((128,), torch.int32),
        num_rows=empty((1,), torch.int32),
        slot_row=empty((m, 6), torch.int32),
        y=empty((r, k), torch.bfloat16),
    )


def prepare(x, ids, weights, ws, out, *, config=None):
    del config
    out.zero_()
    ws.q0.zero_()
    ws.q1.zero_()
    ws.q0_scales.fill_(0x7F7F7F7F)
    ws.q1_scales.fill_(0x7F7F7F7F)
    # Stable expert sort retains duplicate slot identity.
    flat = ids.cpu().flatten().tolist()
    pairs = sorted((p for p, e in enumerate(flat) if 0 <= e < 128), key=flat.__getitem__)
    counts = [sum(e == expert for e in flat) for expert in range(128)]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    starts = [(offsets[e] + 3 * e) // 4 * 4 for e in range(128)]
    ws.expert_offsets.copy_(torch.tensor(offsets, device=x.device, dtype=torch.int32))
    ws.scale_offsets.copy_(torch.tensor(starts, device=x.device, dtype=torch.int32))
    ws.num_rows.fill_(len(pairs))
    ws.slot_row.fill_(-1)
    ws.slot_row.flatten()[torch.tensor(pairs, device=x.device, dtype=torch.long)] = torch.arange(
        len(pairs), device=x.device, dtype=torch.int32
    )
    q = quantize(x)
    words = pack_scales(q.scales.unsqueeze(0))[0]
    for expert in range(128):
        first, last = offsets[expert : expert + 2]
        if first == last:
            continue
        pair = torch.tensor(pairs[first:last], device=x.device, dtype=torch.long)
        token = pair // 6
        ws.q0[first:last] = q.values.view(torch.uint8)[token]
        ws.q0_scales[:, starts[expert] : starts[expert] + last - first] = words[:, token]
        ws.row_token[first:last] = token.int()
        ws.row_weight[first:last] = weights.flatten()[pair]
        ws.row_pair[first:last] = pair.int()


def row_quantized(ws, stage, expert):
    first, last = ws.expert_offsets[expert : expert + 2].cpu().tolist()
    start = int(ws.scale_offsets[expert])
    words = getattr(ws, f"q{stage}_scales")[:, start : start + last - first]
    scales = unpack_scales(words.unsqueeze(0))[0]
    values = getattr(ws, f"q{stage}")[first:last].view(torch.float8_e4m3fn)
    return Quantized(values, scales)


def activation_interval(gate, up, bg, bu, routes):
    gate, up = gate.clamp_max(10), up.clamp(-10, 10)
    silu = gate * torch.sigmoid(gate)
    v = silu * up
    bv = 11 * bg + silu.abs() * bu + 1.1 * bg * bu + 8 * U32 * v.abs().clamp_min(TINY)
    weight = routes.double()[:, None]
    h = weight * v
    bh = weight.abs() * bv + U32 * ((weight * v).abs() + weight.abs() * bv) + TINY
    return h, (h - bh).bfloat16().double(), (h + bh).bfloat16().double()


def _scale(peak):
    fraction, exponent = torch.frexp(peak)
    return (exponent - 9 + (fraction > 0.875).int() + 127).clamp(0, 254)


def q1_cells(oracle, low, high, actual=None):
    """Enumerate intersecting UE8M0/RNE cells, capped at one coarser E4M3 step.

    Endpoints have already passed BF16 RNE. Zero is a separate neutral-scale
    cell; byte 0 has no lower scale bound and byte 254 has no upper one.
    Return a QDQ radius for FC2 propagation and optionally check actual cells.
    """
    value = decode_mxfp8(oracle, torch.float64)
    minmag = torch.where((low <= 0) & (high >= 0), 0, torch.minimum(low.abs(), high.abs()))
    maxmag = torch.maximum(low.abs(), high.abs())
    group_lo = minmag.reshape(len(low), -1, 32).amax(-1)
    group_hi = maxmag.reshape(len(low), -1, 32).amax(-1)
    first = _scale(group_lo.clamp_min(2.0**-133))
    last = _scale(group_hi.clamp_min(2.0**-133))
    candidates = set(range(int(first.min()), int(last.max()) + 1)) | {127}
    radius = torch.zeros_like(value)
    admitted = torch.zeros_like(value, dtype=torch.bool)
    actual_value = None if actual is None else decode_mxfp8(actual, torch.float64)
    oracle_sf = torch.exp2(oracle.scales.double() - 127).repeat_interleave(32, -1)
    for byte in sorted(candidates):
        allowed = (first <= byte) & (byte <= last) & (group_hi > 0)
        zero = (group_lo == 0) & (byte == 127)
        scale = 2.0 ** (byte - 127)
        qlo = (low / scale).clamp(-448, 448).float().to(torch.float8_e4m3fn).double() * scale
        qhi = (high / scale).clamp(-448, 448).float().to(torch.float8_e4m3fn).double() * scale
        coarse = torch.maximum(torch.full_like(oracle_sf, scale), oracle_sf)
        mag = torch.maximum(value.abs(), torch.maximum(qlo.abs(), qhi.abs())) / coarse
        step = (
            torch.exp2(torch.floor(torch.log2(mag.clamp_min(2.0**-9))) - 3).clamp_min(2.0**-9)
            * coarse
        )
        distance = torch.maximum((qlo - value).abs(), (qhi - value).abs()).clamp_max(step)
        radius = torch.maximum(radius, torch.where(allowed.repeat_interleave(32, -1), distance, 0))
        radius = torch.maximum(radius, torch.where(zero.repeat_interleave(32, -1), value.abs(), 0))
        if actual is not None:
            scale_ok = (actual.scales == byte).repeat_interleave(32, -1)
            ordinary = (
                allowed.repeat_interleave(32, -1) & (actual_value >= qlo) & (actual_value <= qhi)
            )
            ordinary &= (actual_value - value).abs() <= step
            neutral = zero.repeat_interleave(32, -1) & (actual_value == 0)
            admitted |= scale_ok & (ordinary | neutral)
    if actual is not None:
        assert admitted.all(), "Q1 left intersecting scale/RNE cells"
    return radius


def atomic_bound(slots, radii):
    """Any ordering of <=6 BF16-narrowed slots and BF16 atomic additions."""
    c = radii + bf16_ulp(slots.abs() + radii)
    ub = 2.0**-8
    gamma = 6 * ub / (1 - 6 * ub)
    return c.sum(1) + gamma * (slots.abs() + c).sum(1) + 6 * 2.0**-134 / (1 - 6 * ub)


def rows_combine(slots):
    """Each slot rounded once to BF16, added in FP32 in slot order, then one BF16 RNE."""
    total = torch.zeros_like(slots[:, 0], dtype=torch.float32)
    for slot in range(6):
        total = total + slots[:, slot].bfloat16().float()
    return total.bfloat16()


def rows_bound(slots, radii):
    """Propagate FC1/Q1/FC2 uncertainty, each slot's BF16 rounding, FP32 slot adds and the
    final RNE."""
    narrowed = radii + bf16_ulp(slots.abs() + radii)
    gamma = 6 * U32 / (1 - 6 * U32)
    combine = gamma * (slots.abs() + narrowed).sum(1) + 6 * TINY
    before_final = narrowed.sum(1) + combine
    return before_final + bf16_ulp(slots.sum(1).abs() + before_final)


def assert_attainable(actual, slots):
    """Structural atomic check for small fixtures, independently per coordinate."""
    admitted = torch.zeros_like(actual, dtype=torch.bool)
    for order in permutations(range(slots.shape[1])):
        total = torch.zeros_like(actual, dtype=torch.bfloat16)
        for slot in order:
            total = (total.float() + slots[:, slot].bfloat16().float()).bfloat16()
        admitted |= actual == total
    assert admitted.all(), "atomic result cannot arise from any slot permutation"


def prepare_reference(x, local_ids, weights, ws=None, out=None):
    """CPU Q0 oracle, quantizing each source row once with reference.quantize."""
    from ctikernels.moe.cute._w4a8_prepare import W4A8Workspace

    if x.device.type != "cpu":
        raise ValueError("prepare_reference is a CPU model")
    tensor(x, (len(x), 4096), torch.bfloat16, x.device, "x")
    ws = W4A8Workspace(len(x), x.device) if ws is None else ws
    if ws.m != len(x) or ws.device != x.device:
        raise ValueError("workspace device/shape mismatch")
    route = routing_metadata(local_ids, weights)
    for name in RouteModel.__dataclass_fields__:
        getattr(ws, name).copy_(getattr(route, name))
    q = quantize(x)
    length = int(route.num_rows[0])
    pairs = route.row_pair[:length].long()
    ws.q0[:length] = q.values.view(torch.uint8)[pairs // 6]
    ws.q0_scales.fill_(0x7F7F7F7F)
    ws.q1_scales.fill_(0x7F7F7F7F)
    ws.q0_scales[:, route.pair_scale[pairs].long()] = pack_activation_scales(q.scales[pairs // 6])
    if out is not None:
        tensor(out, x.shape, torch.bfloat16, x.device, "out")
        if ws.combine == "atomic":
            out.zero_()
    return ws
