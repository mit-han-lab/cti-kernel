# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Shared GDN prefill cases and precision checks for the GPU suites: inputs, FP64-relative
errors and the no-worse-than-fla-core rule."""

from __future__ import annotations

import statistics

import torch

from ctikernels.gdn._contract import Geometry

from . import _gdn_reference as reference
from .test_gdn import (
    _capture,
    _fla_core,
    _poison,
    _pools,
    _prefill_case,
    _projected,
    _same,
    _split,
    _weights,
)

K = V = 128
GEOMETRIES = [(16, 32), (16, 48)]
TP2_GEOMETRIES = [(8, 16), (8, 24)]
SLOTS = 24
# Errors within FP32's smallest normal count as exact.
ABSOLUTE_FLOOR = 2.0**-126


def _rms_error(got: torch.Tensor, exact: torch.Tensor) -> float:
    return float((got.double() - exact.double()).pow(2).mean().sqrt())


def _max_error(got: torch.Tensor, exact: torch.Tensor) -> float:
    return float((got.double() - exact.double()).abs().max())


def _no_worse_than_fla(
    name: str, ours, fla, exact, largest_errors: list[tuple[float, float]] | None = None
) -> None:
    """Ours against FP64 no worse than fla-core's: no multiplicative margin. RMS is checked
    per seed; collected largest errors are compared as a 5-seed mean. The largest error is
    one element's error, dominated by which element's BF16 output rounding lands worst. Its
    standard deviation across seeds for y is 5-34% of its mean for fla-core (all three
    devices, 5 seeds x 9 configurations x 3 kinds x 2 geometries). Per seed, ours exceeded
    it in 4 of 810 GB10 runs (none on RTX 5090 / RTX PRO 6000), by at most 8%, on ``heavy``
    inputs only; the worst 5-seed-mean ratio was 0.81. Per-seed y RMS was 0.40-0.69x
    fla-core's and the states' at most 0.08x (rounded).
    The absolute floor is FP32's smallest normal, only covering both errors being zero.
    Without a collection, the largest error is also checked per seed. The largest error
    relative to each element, at max(|exact|, row RMS / 8), is not compared: near-zero outputs
    of ill-conditioned rows put either implementation's above the other's."""
    ours_rms, fla_rms = _rms_error(ours, exact), _rms_error(fla, exact)
    ours_max, fla_max = _max_error(ours, exact), _max_error(fla, exact)
    assert torch.isfinite(ours).all(), name
    assert ours_rms <= fla_rms + ABSOLUTE_FLOOR, (name, ours_rms, fla_rms)
    if largest_errors is None:
        assert ours_max <= fla_max + ABSOLUTE_FLOOR, (name, ours_max, fla_max)
    else:
        largest_errors.append((ours_max, fla_max))


def _largest_no_worse(name: str, errors: list[tuple[float, float]]) -> None:
    """The largest error (one element's) no worse than fla-core's as a seed mean, with the
    margin two standard errors of the per-seed ratio (the capture suite's rule): ours and
    fla-core round the same BF16 operands, so which element rounds worst varies by seed."""
    ours, fla = zip(*errors, strict=True)
    ratios = [o / max(f, ABSOLUTE_FLOOR) for o, f in errors]
    margin = 2 * statistics.stdev(ratios) / len(ratios) ** 0.5 if len(ratios) > 1 else 0.0
    assert sum(ours) <= sum(fla) * (1 + margin) + ABSOLUTE_FLOOR, (name, ours, fla)


def _fla_prefill(
    geometry: Geometry,
    projected,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    cu_seqlens,
    slot_indices,
    has_initial_state,
    state,
    conv_state,
) -> torch.Tensor:
    """fla-core's public chain (unmodified) on dense copies of the batch's pool rows: y, with
    the pools updated as the operation updates them."""
    fla = _fla_core()
    hk, hv, dc = geometry.k_heads, geometry.v_heads, geometry.conv_dim
    count = cu_seqlens.shape[0] - 1
    slots, flags = slot_indices.tolist(), has_initial_state.tolist()
    bounds = cu_seqlens.tolist()
    history = torch.zeros(count, 3, dc, dtype=conv_state.dtype, device=projected.device)
    initial = torch.zeros(count, hv, V, K, dtype=torch.float32, device=projected.device)
    for n, (slot, flag) in enumerate(zip(slots, flags, strict=True)):
        if slot >= 0 and flag:
            history[n], initial[n] = conv_state[slot], state[slot]
    x, z, b, a = _split(projected, hk, hv)
    window = torch.zeros(count, dc, 4, dtype=conv_state.dtype, device=projected.device)
    window[:, :, 1:] = history.transpose(1, 2)
    mixed, _ = fla.conv(
        x[None],
        weight=conv_weight,
        initial_state=window,
        activation="silu",
        cu_seqlens=cu_seqlens.clone(),
    )
    q, k, v = (part.contiguous() for part in mixed[0].split([hk * K, hk * K, hv * V], 1))
    o, final = fla.chunk(
        q.view(1, -1, hk, K),
        k.view(1, -1, hk, K),
        v.view(1, -1, hv, V),
        g=a.view(1, -1, hv),
        beta=b.view(1, -1, hv),
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        cu_seqlens=cu_seqlens.clone(),
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    y = fla.gated_norm(
        o.reshape(-1, V), z.reshape(-1, V), norm_weight, None, activation="silu", eps=1e-6
    )
    for n, (slot, flag) in enumerate(zip(slots, flags, strict=True)):
        length = bounds[n + 1] - bounds[n]
        if slot < 0 or (length == 0 and flag):
            continue
        state[slot] = final[n] if length else 0
        conv_state[slot] = torch.cat((history[n], x[bounds[n] : bounds[n + 1]]))[-3:]
    return y.view(projected.shape[0], hv * V).to(projected.dtype)


def _inputs(k_heads, v_heads, lengths, kind: str, seed: int):
    """Weights and projected rows: ``ordinary``, ``slow`` decays (dt_bias near -8: the state
    accumulates across every chunk) or ``heavy`` (pre-conv q | k | v columns scaled by 16)."""
    weights = _weights(
        k_heads, v_heads, seed=seed, dt_bias=(-8.0, -6.0) if kind == "slow" else (-2.0, 2.0)
    )
    projected = _projected(sum(lengths), k_heads, v_heads, seed=seed + 1)
    if kind == "heavy":
        conv_dim = 2 * k_heads * K + v_heads * V
        projected[:, :conv_dim] = (projected[:, :conv_dim].float() * 16).to(torch.bfloat16)
    return weights, projected


__all__ = [
    "ABSOLUTE_FLOOR",
    "GEOMETRIES",
    "SLOTS",
    "TP2_GEOMETRIES",
    "_capture",
    "_fla_prefill",
    "_inputs",
    "_largest_no_worse",
    "_max_error",
    "_no_worse_than_fla",
    "_poison",
    "_pools",
    "_prefill_case",
    "_projected",
    "_rms_error",
    "_same",
    "reference",
]
