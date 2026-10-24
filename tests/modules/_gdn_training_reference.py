# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/modules/conv/triton/ops.py`, `fla/modules/conv/triton/kernels.py`,
# `fla/ops/gated_delta_rule/gate.py`); see THIRD_PARTY_NOTICES.

"""PyTorch reference of ``ctikernels.gdn._fla_training`` (FP32 ops, no blocking): fla-core's
conv forward, the conv's input gradient and the gate backward's da. CPU tests exercise the
provider composition with this reference; GPU tests compare the kernels with fla-core."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ctikernels.gdn._contract import CONV, HISTORY


def sequence_ids(cu_seqlens: torch.Tensor, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token's sequence and position in it (int64 [T] each)."""
    token = torch.arange(tokens, device=cu_seqlens.device)
    sequence = torch.searchsorted(cu_seqlens[1:].long(), token, right=True)
    return sequence, token - cu_seqlens.long()[sequence]


def sequence_index(cu_seqlens: torch.Tensor, tokens: int) -> torch.Tensor:
    """Each token's sequence (int32 [T]), as ``_fla_training.sequence_index``."""
    return sequence_ids(cu_seqlens, tokens)[0].to(torch.int32)


def pre_activation(x: torch.Tensor, weight: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """The causal depthwise conv of x [T, Dc] within each sequence from a zero history, FP32
    products and sums, oldest tap first."""
    tokens = x.shape[0]
    _, position = sequence_ids(cu_seqlens, tokens)
    token = torch.arange(tokens, device=x.device)
    total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for j in range(CONV):
        back = HISTORY - j  # tap j multiplies x_{t - back}
        value = torch.where(
            (position >= back)[:, None], x[(token - back).clamp(min=0)].float(), 0.0
        )
        total += weight[:, j].float() * value
    return total


def conv_forward(
    x: torch.Tensor, weight: torch.Tensor, sequence: torch.Tensor, cu_seqlens: torch.Tensor,
    key_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """SiLU(conv(x)) in x's dtype, split into contiguous q, k and v."""
    pre = pre_activation(x, weight, cu_seqlens)
    y = (pre * torch.sigmoid(pre)).to(x.dtype)
    q, k, v = y.split([key_width, key_width, x.shape[1] - 2 * key_width], 1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def conv_backward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    sequence: torch.Tensor,
    cu_seqlens: torch.Tensor,
    d_mixed: tuple[torch.Tensor, ...],
    out: torch.Tensor,
) -> None:
    """The conv's input gradient into ``out`` [T, Dc]: SiLU's slope (fla-core's swish_bwd) at
    the BF16 pre-activation, then dx_s = sum_j w_j dc_{s+3-j} within s's sequence, from the
    tap of s itself, each term one rounded multiply-add (``torch.addcmul``)."""
    rows, dc = x.shape
    c = pre_activation(x, conv_weight, cu_seqlens).bfloat16().float()
    d = torch.cat(list(d_mixed), -1).float()
    s = torch.sigmoid(c)
    d = F.pad(d * s * (1 + c * (1 - s)), (0, 0, 0, HISTORY))
    ids = F.pad(sequence.long(), (0, HISTORY), value=-1)
    dx = torch.zeros(rows, dc, dtype=torch.float32, device=x.device)
    for j in reversed(range(CONV)):
        shift = HISTORY - j
        term = torch.where(
            (ids[shift : shift + rows] == sequence.long())[:, None], d[shift : shift + rows], 0.0
        )
        dx = torch.addcmul(dx, conv_weight[:, j].float().expand_as(term), term)
    out.copy_(dx)


def gate_gradient(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    dyg: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """da = -exp(A_log) (dyg sigmoid(a + dt_bias)) into ``out`` (fla-core's gdn_gate_bwd dg)."""
    g = a.float() + dt_bias.float()
    out.copy_((-torch.exp(A_log.float()) * (dyg.float() * torch.sigmoid(g))).to(out.dtype))
