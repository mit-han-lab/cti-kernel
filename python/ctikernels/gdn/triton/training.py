# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Our portable Triton GDN training (plain ``tl.dot``; compute capabilities 10.0 to 12.1): the chunked gated
delta rule forward and backward with its glue, no fla-core.

Forward, four launches: ``features`` (conv + SiLU, q/k L2 norm, beta), ``solve`` (gates,
gamma, T = (I + L)^-1, w and u), ``states`` (the sequential pass: chunk states, v' and the
output) and ``gated_norm``. Saved: q, k and their 1/norms, v, beta, gamma, T, o and the norm's 1/rms, and
the state checkpoints (w, the chunk states h and v'), so the backward does not run the
sequential forward pass again.

Backward, five launches: the gated norm's (``gdn._gated_norm``), ``state_backward`` (the
reverse sequential pass, with each chunk's own dv' term), ``dqkw`` (per key head, over its value
heads), ``wy_backward`` (per value head, with the gate and beta-sigmoid gradients into d
projected) and ``conv_norm_backward`` (the q/k L2 norms' and the conv's input gradients).

Fixed launch configurations, no autotuning; no atomics (every output has one writer and every
reduction a fixed order), so a call is bitwise deterministic. Grids depend only on T and the
sequence count; sequence boundaries are read on the device (no host synchronisation). The
algorithm and its gradients are ``gdn.torch.training``'s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .. import _gated_norm
from .._contract import Geometry, K
from ._train_chunks import chunk_index
from ._train_configs import configs
from ._train_conv import conv_norm_backward
from ._train_features import features
from ._train_intra import CHUNK, solve, wy_backward
from ._train_output import dqkw, gated_norm
from ._train_state import state_backward, states

SCALE = K**-0.5


@dataclass(frozen=True)
class Chunks:
    """A batch's sequences on the device (``_train_chunks.chunk_index``): ``cu_seqlens``
    [N + 1], each sequence's first entry ``offsets`` [N + 1], the chunk table [capacity, 2]
    (sequence, chunk within it; entries past the batch's chunks name a chunk past the last
    sequence's end) and each token's ``sequence`` [T], all int32."""

    cu_seqlens: torch.Tensor
    offsets: torch.Tensor
    table: torch.Tensor
    sequence: torch.Tensor

    @classmethod
    def of(cls, cu_seqlens: torch.Tensor, rows: int) -> Chunks:
        """One launch; ``cu_seqlens`` is copied (the caller may refill its buffer), and
        capacity bounds the chunks of any ``rows``-row batch."""
        return cls(*chunk_index(cu_seqlens, rows, CHUNK))

    @property
    def capacity(self) -> int:
        return self.table.shape[0]


def rule_forward(q, k, v, a, beta, A_log, dt_bias, z, norm_weight, chunks: Chunks, configs,
                 initial_state=None, final=False):  # fmt: skip
    """The chunked gated delta rule with the gated RMSNorm: o [T, Hv, V] BF16, y [T, Hv V]
    BF16 and the norm's 1/rms r [T Hv] FP32, the tensors ``rule_backward`` reads (gamma, T, w,
    the chunk states h and v') and the final states [N, Hv, V, K] FP32 (None unless ``final``)
    of normalised q, k [T, Hk, K] and v [T, Hv, V] BF16 (contiguous), the gate input a [T, Hv] (strided
    rows), beta [T, Hv] FP32, z [T, Hv V] (strided rows), the norm weight and optional initial
    states [N, Hv, V, K] FP32; ``configs`` from ``_train_configs.configs``."""
    cu, table = chunks.cu_seqlens, chunks.table
    gamma, A, w, u = solve(k, v, a, beta, A_log, dt_bias, table, cu, configs["solve"])
    o, h, v_new, ht = states(q, k, w, u, gamma, cu, chunks.offsets, chunks.capacity, SCALE,
                             configs["states"], initial_state, final)  # fmt: skip
    y, rstd = gated_norm(o, z, norm_weight, configs["gated_norm"])
    return o, y, rstd, (gamma, A, w, h, v_new), ht


def rule_backward(q, k, v, a, b, beta, saved, A_log, dt_bias, do, chunks: Chunks, da, db, configs,
                  initial=False, dht=None):  # fmt: skip
    """``rule_forward``'s gradients from do [T, Hv, V] BF16 (and the final states' dht FP32)
    and its ``saved`` tensors, by the normalised q and k: dq [T, Hk, K] FP32 and dk = dk_heads
    [T, Hk, K] FP32 plus the value heads' dk_terms [T, Hv, K] BF16 (summed per key head in head
    order), dv [T, Hv, V] BF16 and, with ``initial``, the initial states' dh0 FP32 (else None);
    the raw gate inputs' gradients go into da and db ([T, Hv], strided rows; beta =
    sigmoid(b)). Returns (dq, dk_heads, dk_terms, dv, dh0)."""
    cu, table, offsets, capacity = chunks.cu_seqlens, chunks.table, chunks.offsets, chunks.capacity
    gamma, A, w, h, v_new = saved
    dh, du, dh0 = state_backward(q, k, w, gamma, do, cu, offsets, capacity, SCALE,
                                 configs["state_backward"], dht, initial)  # fmt: skip
    dq, dk, dw, parts = dqkw(q, k, v_new, gamma, h, do, dh, du, table, cu, SCALE, configs["dqkw"])
    del dh
    dk_terms, dv = wy_backward(k, v, beta, gamma, A, dw, du, parts, a, b, A_log, dt_bias, da, db,
                               table, cu, configs["wy_backward"])  # fmt: skip
    return dq, dk, dk_terms, dv, dh0


class Training:
    """``_contract.TrainKernel`` on our Triton kernels, with the launch configurations of a
    device of compute capability ``capability``."""

    provider = "triton"

    def __init__(self, capability: tuple[int, int]) -> None:
        self.configs = configs(capability)

    def train_forward(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        rows, dc, vw = projected.shape[0], geometry.conv_dim, geometry.value_width
        chunks = Chunks.of(cu_seqlens, rows)
        q, k, v, q_rstd, k_rstd, beta = features(
            geometry,
            projected,
            conv_weight,
            chunks.sequence,
            chunks.cu_seqlens,
            self.configs["features"],
        )
        a = projected[:, dc + vw + geometry.v_heads :]
        o, y, rstd, rule, _ = rule_forward(q, k, v, a, beta, A_log, dt_bias,
                                           projected[:, dc : dc + vw], norm_weight, chunks,
                                           self.configs)  # fmt: skip
        saved = (projected, conv_weight, A_log, dt_bias, norm_weight, chunks.cu_seqlens,
                 chunks.offsets, chunks.table, chunks.sequence)  # fmt: skip
        return y, (*saved, q, k, v, q_rstd, k_rstd, beta, o, rstd, *rule)

    def training_bytes(self, geometry: Geometry, tokens: int, sequences: int) -> tuple[int, int]:
        """(saved, backward) payload bytes, from ``train_forward``'s save list and
        ``train_backward``'s allocations.

        Saved, per row: projected, q and k, v, o, w and v' (2 values each), the q/k 1/norms,
        beta, the norm's 1/rms and gamma (FP32), T (BF16 [Hv, 64]) and the sequence index;
        per chunk entry the chunk state h and the table; cu_seqlens and offsets. The backward
        peaks in ``dqkw`` (grad, do, dv', dw, FP32 dq and dk per key head, dg's key blocks, dh)
        or in ``wy_backward``, where the dk terms and dv replace dh."""
        hk, hv, values = geometry.k_heads, geometry.v_heads, geometry.value_width
        chunks = (tokens + (CHUNK - 1) * sequences) // CHUNK
        state = 2 * hv * K * K
        row = 2 * geometry.proj_width + 4 * geometry.key_width + 8 * values + 8 * hk
        row += (4 + 4 + 4 + 2 * CHUNK) * hv + 4
        saved = tokens * row + chunks * (state + 8) + 8 * (sequences + 1)
        parts = K // self.configs["dqkw"]["BK"]
        row = 2 * geometry.proj_width + 6 * values + 8 * geometry.key_width + 4 * hv * parts
        return saved, tokens * row + max(chunks * state, 4 * values * tokens)

    def train_backward(
        self, geometry: Geometry, saved: Sequence[torch.Tensor], dy: torch.Tensor
    ) -> torch.Tensor:
        projected, conv_weight, A_log, dt_bias, norm_weight = saved[:5]
        chunks = Chunks(*saved[5:9])
        q, k, v, q_rstd, k_rstd, beta, o, rstd = saved[9:17]
        rows, dc, hv = projected.shape[0], geometry.conv_dim, geometry.v_heads
        vw = geometry.value_width
        grad = torch.empty(rows, projected.shape[1], dtype=torch.bfloat16, device=dy.device)
        do = torch.empty_like(o)
        _gated_norm.gated_rms_norm_backward(
            o,
            projected[:, dc : dc + vw].unflatten(1, (hv, -1)),
            norm_weight,
            rstd,
            dy.view(o.shape),
            do,
            grad[:, dc : dc + vw].unflatten(1, (hv, -1)),
        )
        gates = dc + vw
        dq, dk, dk_terms, dv, _ = rule_backward(
            q, k, v, projected[:, gates + hv :], projected[:, gates : gates + hv], beta,
            saved[17:], A_log, dt_bias, do, chunks, grad[:, gates + hv :],
            grad[:, gates : gates + hv], self.configs,
        )  # fmt: skip
        del do
        conv_norm_backward(
            geometry, projected[:, :dc], conv_weight, chunks.sequence, chunks.cu_seqlens, q, k,
            q_rstd, k_rstd, dq, dk, dk_terms, dv, grad[:, :dc], self.configs["conv_backward"],
        )  # fmt: skip
        return grad


__all__ = ["Chunks", "Training", "rule_backward", "rule_forward"]
