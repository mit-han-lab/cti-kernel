# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The CPU reference: out of place, so FP64 gradcheck may run backward repeatedly."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IN_PLACE = False  # the backward returns fresh tensors


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def swiglu_backward_(
    grad_h: torch.Tensor,
    gate_up: torch.Tensor,
    grad_y_b: torch.Tensor,
    x_a: torch.Tensor,
    grads: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    rank, width = grad_y_b.shape[1], gate.shape[1]
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    grad_gate = grad_h * up * sigmoid * (1 + gate * (1 - sigmoid))
    grad_up = grad_h * silu
    down_a, gate_b, up_b = grads.view(3, -1).unbind()
    torch.mm(grad_y_b.t(), silu * up, out=down_a.view(rank, width)).mul_(scaling)
    torch.mm(grad_gate.t(), x_a[:, :rank], out=gate_b.view(width, rank)).mul_(scaling)
    torch.mm(grad_up.t(), x_a[:, rank:], out=up_b.view(width, rank)).mul_(scaling)
    return torch.cat((grad_gate, grad_up), dim=-1)
