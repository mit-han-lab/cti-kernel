# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/ops/gated_delta_rule/chunk.py`); see THIRD_PARTY_NOTICES.

"""CuTe DSL GDN training provider on sm_120 and sm_121 (RTX 5090, RTX PRO 6000,
DGX Spark GB10), from the ``cti-kernels[cute]`` extra.

``training(device)`` returns ``CuteTraining``, the whole forward/backward composition,
or None where CuTe does not run (the caller falls back to the Triton provider). ``load(device)``
resolves its ``ChunkKernels`` once per device and process: None off CUDA, without CUDA,
with ``CTI_GDN_CUTE=0``, or without nvidia-cutlass-dsl at ``VERSION``. Only a resolution
that can use the DSL imports it; fla-core is configured before importing its stages.

The composition uses the shared conv and gated-norm backward building blocks and
fla-core's non-chunk stages. ``ChunkKernels`` forms G with fla-core's gate kernel and
runs ``_cute_fwd`` / ``_cute_bwd`` for the chunk forward and backward, respectively.
Its backward returns dg before the reverse cumulative sum.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
import warnings
from collections.abc import Sequence
from functools import cache

import torch

from ._contract import EPS, Geometry, K, V
from ._fla import CHUNK, _chunk_table, chunk_capacity, saved_bytes

VARIABLE = "CTI_GDN_CUTE"
DISTRIBUTION = "nvidia-cutlass-dsl"
VERSION = "4.8.0"
CAPABILITIES = ((12, 0), (12, 1))
RCP_LN2 = 1.4426950408889634  # fla-core's gates are in the log2 domain
# StatesR: 32-wide V slices, 4 warps (48 KB; two CTAs per SM on 5090 / PRO 6000).
STATES_SHAPE = (32, 4)

_lock = threading.Lock()
_compiling = threading.Lock()  # cute.compile changes process-wide DSL state
_forwards: dict[torch.device, ChunkKernels | None] = {}


def load(device: torch.device) -> ChunkKernels | None:
    """The CuTe kernels on ``device``, or None (see the module docstring)."""
    if device not in _forwards:
        with _lock:
            if device not in _forwards:
                _forwards[device] = _resolve(device)
    return _forwards[device]


def _resolve(device: torch.device) -> ChunkKernels | None:
    if device.type != "cuda" or os.environ.get(VARIABLE) == "0":
        return None
    if not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_capability(device) not in CAPABILITIES:
        return None
    try:
        if importlib.metadata.version(DISTRIBUTION) != VERSION:
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        from . import _cute_bwd, _cute_fwd
    except ImportError as error:
        warnings.warn(
            f"{DISTRIBUTION} {VERSION} is installed but does not import ({error}); GDN training "
            "falls back to the Triton provider",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    # fla-core's gate kernel forms G; the provider configures fla-core before it is imported.
    from . import _fla

    _fla.load("GatedDeltaNetTraining", device)
    from ._fla_fixed import gdn_gate_chunk_cumsum

    return ChunkKernels(device, _cute_fwd, _cute_bwd, gdn_gate_chunk_cumsum)


def training(device: torch.device) -> CuteTraining | None:
    """The CuTe training provider on ``device``, or None for the portable fallback. It runs
    fla-core's non-chunk stages, so fla-core's checks run on every resolution (``_fla.load``)."""
    chunk = load(device)
    if chunk is None:
        return None
    from . import _fla

    _fla.load("GatedDeltaNetTraining", device)
    return CuteTraining(chunk)


class CuteTraining:
    """The complete ``_contract.TrainKernel`` composition with CuTe chunk kernels."""

    provider = "cute"

    def __init__(self, chunk: ChunkKernels) -> None:
        # load() configured fla-core before these imports.
        from fla.ops.common.gate import fused_beta_sigmoid_bwd, fused_beta_sigmoid_fwd

        from . import _fla_fixed, _fla_training, _gated_norm, _l2norm

        self._chunk = chunk
        self._l2norm = _l2norm.l2norm_forward
        self._l2norm_bwd = _l2norm.l2norm_backward
        self._beta_sigmoid = fused_beta_sigmoid_fwd
        self._beta_sigmoid_bwd = fused_beta_sigmoid_bwd
        self._gated_norm_fwd = _fla_fixed.layer_norm_gated_fwd
        self._local_cumsum = _fla_fixed.chunk_local_cumsum
        self._gated_norm_bwd = _gated_norm.gated_rms_norm_backward
        self._sequence_index = _fla_training.sequence_index
        self._conv_fwd = _fla_training.conv_forward
        self._conv_bwd = _fla_training.conv_backward
        self._gate_gradient = _fla_training.gate_gradient

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
        """y, and ``saved``: projected, the weights, the fresh ``cu_seqlens``, chunk table and
        sequence index, o with the norm's 1/rms, and what the CuTe chunked backward reads
        (the normalised q and k with their 1/norm, v, the chunk-local cumulative decay, b,
        beta and the WY matrix A)."""
        rows, dc, hk, hv = projected.shape[0], geometry.conv_dim, geometry.k_heads, geometry.v_heads
        vw = geometry.value_width
        cu = cu_seqlens.clone()
        table = _chunk_table(cu, rows)
        sequence = self._sequence_index(cu, rows)
        # fla-core's conv forward (bitwise), written straight into contiguous q, k and v.
        q, k, v = self._conv_fwd(projected[:, :dc], conv_weight, sequence, cu, geometry.key_width)
        q, q_rstd = self._l2norm(q.view(1, rows, hk, K))
        k, k_rstd = self._l2norm(k.view(1, rows, hk, K))
        v = v.view(1, rows, hv, V)
        # The gate in fla-core's kernel (as its autograd chain forms it): a stays an input, and
        # its gradient da is formed in the backward, never A_log's or dt_bias's.
        b, a = self._gates(geometry, projected)
        b, a = b.view(1, rows, hv), a.view(1, rows, hv)
        beta = self._beta_sigmoid(b)
        G, o, A = self._chunk(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=beta,
            scale=K**-0.5,
            cu_seqlens=cu,
            chunk_indices=table,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        o = o.to(q.dtype)
        # fla-core's gated RMSNorm forward kernel (the one its rms_norm_gated runs), keeping its
        # 1/rms per row for the backward, as its autograd function does.
        y, _, rstd, _ = self._gated_norm_fwd(
            x=o.reshape(rows * hv, V),
            g=projected[:, dc : dc + vw].reshape(rows * hv, V).contiguous(),
            weight=norm_weight,
            bias=None,
            activation="silu",
            eps=EPS,
            is_rms_norm=True,
        )
        saved = (projected, conv_weight, A_log, dt_bias, norm_weight, cu, table, sequence, o, rstd)
        return y.view(rows, hv * V), (*saved, q, q_rstd, k, k_rstd, v, G, b, beta, A)

    def train_backward(
        self, geometry: Geometry, saved: Sequence[torch.Tensor], dy: torch.Tensor
    ) -> torch.Tensor:
        """d projected [T, P] BF16 from ``train_forward``'s ``saved`` and dy [T, Hv V]: the
        gated norm's do and dz (``_gated_norm``); dq, dk, dv and dbeta from CuTe's chunk
        kernels, with fla-core's L2 norm and beta sigmoid
        backward; da from its decay gradient (``_fla_training.gate_gradient``); the conv's input
        gradient with SiLU's slope (``_fla_training.conv_backward``)."""
        projected, conv_weight, A_log, dt_bias, norm_weight, cu, table, sequence, o, rstd = saved[
            :10
        ]
        q, q_rstd, k, k_rstd, v, G, b, beta, A = saved[10:]
        rows, dc, hv = projected.shape[0], geometry.conv_dim, geometry.v_heads
        vw = geometry.value_width
        grad = torch.empty(rows, projected.shape[1], dtype=torch.bfloat16, device=projected.device)
        do = torch.empty(1, rows, hv, V, dtype=o.dtype, device=o.device)
        self._gated_norm_bwd(
            o[0],
            projected[:, dc : dc + vw].unflatten(1, (hv, V)),
            norm_weight,
            rstd,
            dy.reshape(rows, hv, V),
            do[0],
            grad[:, dc : dc + vw].unflatten(1, (hv, V)),
        )
        scale, index = K**-0.5, {"cu_seqlens": cu, "chunk_indices": table}
        dq, dk, dv, dbeta, dg = self._chunk.backward(
            q=q, k=k, v=v, beta=beta, A=A, g=G, do=do, scale=scale, **index
        )
        del do
        dg = self._local_cumsum(dg, chunk_size=CHUNK, reverse=True, **index)
        self._gate_gradient(
            projected[:, dc + vw + hv :],
            A_log,
            dt_bias,
            dg.view(rows, hv),
            grad[:, dc + vw + hv :],
        )
        dq = self._l2norm_bwd(q, q_rstd, dq).reshape(rows, -1)
        dk = self._l2norm_bwd(k, k_rstd, dk).reshape(rows, -1)
        grad[:, dc + vw : dc + vw + hv] = self._beta_sigmoid_bwd(b, dbeta).reshape(rows, hv)
        self._conv_bwd(
            projected[:, :dc],
            conv_weight,
            sequence,
            cu,
            (dq, dk, dv.reshape(rows, vw)),
            grad[:, :dc],
        )
        return grad

    def training_bytes(self, geometry: Geometry, tokens: int, sequences: int) -> tuple[int, int]:
        """(saved, backward): the backward peaks in GradQK, with grad, do, v', dv' and the per
        value head dq, dk and dw (2 values each per row), dg's two key halves (FP32), the chunk
        states h and dh and the sequences' launch order live."""
        hv = geometry.v_heads
        row = 2 * geometry.proj_width + 12 * geometry.value_width + 8 * hv
        states = 2 * chunk_capacity(tokens, sequences) * 2 * hv * K * V
        return saved_bytes(geometry, tokens, sequences), tokens * row + states + 4 * sequences

    @staticmethod
    def _gates(geometry: Geometry, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The b and a columns, contiguous (fla-core's gate kernels read dense rows)."""
        tail = projected[:, geometry.conv_dim + geometry.value_width :]
        return tail[:, : geometry.v_heads].contiguous(), tail[:, geometry.v_heads :].contiguous()


class ChunkKernels:
    """The CuTe forward (call) and backward on one device; kernels compile on first use per
    geometry."""

    def __init__(self, device: torch.device, kernels, backward_kernels, gate) -> None:
        major, minor = torch.cuda.get_device_capability(device)
        self._options = f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a"
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._kernels = kernels
        self._backward_kernels = backward_kernels
        self._gate = gate
        self._compiled: dict[tuple, object] = {}
        self._first = torch.zeros(1, dtype=torch.int32, device=device)
        # Cache stage lookups per provider, before building compile arguments and factories.
        self._prologue = cache(self._prologue)
        self._recurrence = cache(self._recurrence)
        self._wu = cache(self._wu)
        self._local_dv = cache(self._local_dv)
        self._states = cache(self._states)
        self._state_grads = cache(self._state_grads)
        self._qk_grads = cache(self._qk_grads)
        self._wy_grads = cache(self._wy_grads)

    def __call__(
        self,
        *,
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens,
        chunk_indices,
        A_log,
        dt_bias,
    ):
        _, rows, hk, _ = k.shape
        hv = v.shape[2]
        count = cu_seqlens.shape[0] - 1
        G = self._gate(
            g=g,
            A_log=A_log,
            chunk_size=CHUNK,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        w = k.new_empty(rows, hv * K)
        u = v.new_empty(rows, hv * V)
        A = k.new_empty(1, rows, hv, CHUNK)
        o = v.new_empty(1, rows, hv, V)
        keys, gates = k.view(rows, hk * K), G.view(rows, hv)
        self._prologue(hk, hv)(
            keys,
            v.view(rows, hv * V),
            gates,
            beta.view(rows, hv),
            cu_seqlens,
            chunk_indices,
            w,
            u,
            A.view(rows, hv * CHUNK),
            chunk_indices.shape[0],
        )
        bv = self._slice(hv, count)
        self._recurrence(hk, hv, bv)(
            w, u, q.view(rows, hk * K), keys, gates, cu_seqlens, o.view(rows, hv * V), scale, count
        )
        return G, o, A

    def backward(self, *, q, k, v, beta, A, g, do, scale, cu_seqlens, chunk_indices):
        """dq, dk [1, T, Hk, K] and dv [1, T, Hv, V] BF16, dbeta and dg [1, T, Hv] FP32 for the
        forward's call and its o's gradient ``do``; dg before the reverse cumulative sum."""
        _, rows, hk, _ = k.shape
        hv = v.shape[2]
        count = cu_seqlens.shape[0] - 1
        capacity = chunk_indices.shape[0]
        q2, k2, v2, do2 = (
            q.view(rows, hk * K),
            k.view(rows, hk * K),
            v.view(rows, hv * V),
            do.view(rows, hv * V),
        )
        A2, g2, b2 = A.view(rows, hv * CHUNK), g.view(rows, hv), beta.view(rows, hv)
        w = k.new_empty(rows, hv * K)
        u = v.new_empty(rows, hv * V)
        local = v.new_empty(rows, hv * V)
        h = k.new_empty(capacity * hv * K, V)
        v_new = v.new_empty(rows, hv * V)
        order = self._order(cu_seqlens, count)
        self._wu(hk, hv)(k2, v2, A2, g2, b2, cu_seqlens, chunk_indices, w, u, capacity)
        self._local_dv(hk, hv)(q2, k2, do2, g2, cu_seqlens, chunk_indices, local, scale, capacity)
        self._states(hk, hv, *STATES_SHAPE)(
            w, u, k2, g2, cu_seqlens, chunk_indices, order, h, v_new, count, capacity
        )
        del u
        dh = k.new_empty(capacity * hv * K, V)
        dv2 = v.new_empty(rows, hv * V)
        self._state_grads(hk, hv, self._grad_slice(hv))(
            q2, k2, w, do2, local, g2, cu_seqlens, chunk_indices, order, dh, dv2, scale, count,
            capacity,
        )  # fmt: skip
        del w, local
        dq_v = k.new_empty(rows, hv * K)
        dk_v = k.new_empty(rows, hv * K)
        dw = k.new_empty(rows, hv * K)
        dg_v = g.new_empty(2 * rows, hv)  # GradQK's dg terms, one block of rows per key half
        self._qk_grads(hk, hv)(
            q2, k2, do2, v_new, dv2, h, dh, g2, cu_seqlens, chunk_indices, dq_v, dk_v, dw, dg_v,
            scale, capacity,
        )  # fmt: skip
        del h, dh, v_new
        dq, dk = q.new_empty(1, rows, hk, K), k.new_empty(1, rows, hk, K)
        dv = v.new_empty(1, rows, hv, V)
        dbeta, dg = beta.new_empty(1, rows, hv), g.new_empty(1, rows, hv)
        self._wy_grads(hk, hv)(
            k2, v2, A2, dw, dv2, dq_v, dk_v, dg_v, b2, g2, cu_seqlens, chunk_indices,
            dq.view(rows, hk * K), dk.view(rows, hk * K), dv.view(rows, hv * V),
            dbeta.view(rows, hv), dg.view(rows, hv), capacity,
        )  # fmt: skip
        return dq, dk, dv, dbeta, dg

    def _order(self, cu_seqlens: torch.Tensor, count: int) -> torch.Tensor:
        """The sequences by length, longest first (the recurrences launch them in this order)."""
        if count == 1:
            return self._first
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        return torch.argsort(lengths, descending=True, stable=True).to(torch.int32)

    def _slice(self, hv: int, count: int) -> int:
        """The recurrences' V slice: narrow while one wave of CTAs holds every (slice, head,
        sequence)."""
        return 32 if (V // 32) * hv * count <= self._sms else 64

    def _prologue(self, hk: int, hv: int):
        # k, v, g, beta, cu_seqlens, the chunk table; w, u, A; the table's length
        arguments = [("bf16", hk * K), ("bf16", hv * V), ("f32", hv), ("f32", hv), ("i32",)]
        arguments += [("i32", 2), ("bf16", hv * K), ("bf16", hv * V), ("bf16", hv * CHUNK), "i32"]
        return self._compile(("prologue", hk, hv), lambda: self._kernels.IntraP(hk, hv), arguments)

    def _recurrence(self, hk: int, hv: int, bv: int):
        # w, u, q, k, g, cu_seqlens; o; scale and the sequence count
        arguments = [("bf16", hv * K), ("bf16", hv * V), ("bf16", hk * K), ("bf16", hk * K)]
        arguments += [("f32", hv), ("i32",), ("bf16", hv * V), "f32", "i32"]
        return self._compile(
            ("recurrence", hk, hv, bv), lambda: self._kernels.RecurS(hk, hv, bv), arguments
        )

    def _wu(self, hk: int, hv: int):
        # k, v, A, g, beta, cu_seqlens, the chunk table; w, u; table length
        arguments = [("bf16", hk * K), ("bf16", hv * V), ("bf16", hv * CHUNK), ("f32", hv)]
        arguments += [("f32", hv), ("i32",), ("i32", 2), ("bf16", hv * K), ("bf16", hv * V), "i32"]
        return self._compile(("wu", hk, hv), lambda: self._backward_kernels.WuB(hk, hv), arguments)

    def _local_dv(self, hk: int, hv: int):
        # q, k, dO, g, cu_seqlens, the chunk table; dv_local; scale, table length
        arguments = [("bf16", hk * K), ("bf16", hk * K), ("bf16", hv * V), ("f32", hv), ("i32",)]
        arguments += [("i32", 2), ("bf16", hv * V), "f32", "i32"]
        return self._compile(
            ("local_dv", hk, hv), lambda: self._backward_kernels.LocalDV(hk, hv), arguments
        )

    def _grad_slice(self, hv: int) -> int:
        """GradStatesD's V slice: 32 wide while one sequence's slices fit in a wave (the longest
        sequences launch first)."""
        return 32 if (V // 32) * hv <= self._sms else 64

    def _states(self, hk: int, hv: int, bv: int, warps: int):
        # w, u, k, g, cu_seqlens, the chunk table, the order; h, v_new; sequence count, table
        # length
        arguments = [("bf16", hv * K), ("bf16", hv * V), ("bf16", hk * K), ("f32", hv), ("i32",)]
        arguments += [("i32", 2), ("i32",), ("bf16", V), ("bf16", hv * V), "i32", "i32"]
        return self._compile(
            ("states", hk, hv, bv, warps),
            lambda: self._backward_kernels.StatesR(hk, hv, bv, warps),
            arguments,
        )

    def _state_grads(self, hk: int, hv: int, bv: int):
        # q, k, w, dO, dv_local, g, cu_seqlens, the chunk table, the order; dh, dv2; scale,
        # sequence count, table length
        arguments = [("bf16", hk * K), ("bf16", hk * K), ("bf16", hv * K), ("bf16", hv * V)]
        arguments += [("bf16", hv * V), ("f32", hv), ("i32",), ("i32", 2), ("i32",), ("bf16", V)]
        arguments += [("bf16", hv * V), "f32", "i32", "i32"]
        return self._compile(
            ("state_grads", hk, hv, bv),
            lambda: self._backward_kernels.GradStatesD(hk, hv, bv),
            arguments,
        )

    def _qk_grads(self, hk: int, hv: int):
        # q, k, dO, v_new, dv2, h, dh, g, cu_seqlens, chunk table; dq_v, dk_v, dw, dg_v; scale,
        # table length
        arguments = [("bf16", hk * K), ("bf16", hk * K), ("bf16", hv * V), ("bf16", hv * V)]
        arguments += [("bf16", hv * V), ("bf16", V), ("bf16", V), ("f32", hv), ("i32",)]
        arguments += [("i32", 2), ("bf16", hv * K), ("bf16", hv * K), ("bf16", hv * K)]
        arguments += [("f32", hv), "f32", "i32"]
        return self._compile(
            ("qk_grads", hk, hv), lambda: self._backward_kernels.GradQK(hk, hv), arguments
        )

    def _wy_grads(self, hk: int, hv: int):
        # k, v, A, dw, dv2, dq_v, dk_v, dg_v, beta, g, cu_seqlens, chunk table; dq, dk, dv,
        # dbeta, dg; table length
        arguments = [("bf16", hk * K), ("bf16", hv * V), ("bf16", hv * CHUNK), ("bf16", hv * K)]
        arguments += [("bf16", hv * V), ("bf16", hv * K), ("bf16", hv * K), ("f32", hv)]
        arguments += [("f32", hv), ("f32", hv), ("i32",), ("i32", 2), ("bf16", hk * K)]
        arguments += [("bf16", hk * K), ("bf16", hv * V), ("f32", hv), ("f32", hv), "i32"]
        return self._compile(
            ("wy_grads", hk, hv), lambda: self._backward_kernels.GradWY(hk, hv), arguments
        )

    def _compile(self, key: tuple, make, arguments: list):
        compiled = self._compiled.get(key)
        if compiled is None:
            with _compiling:
                compiled = self._compiled.get(key)
                if compiled is None:
                    compiled = _compile(make(), arguments, self._options)
                    self._compiled[key] = compiled
        return compiled


def _compile(kernel, arguments: list, options: str):
    """``kernel`` compiled for TVM-FFI calls with torch tensors, on torch's current stream.
    ``arguments``: (dtype, columns) for a row-major [rows, columns] tensor, (dtype,) for a
    vector, a bare dtype for a scalar; rows, lengths and scalars are given per call."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass.runtime import make_fake_stream  # 4.8 (4.6: cutlass.cute.runtime)

    types = {"bf16": BFloat16, "f32": Float32, "i32": Int32}
    fake = cute.runtime.make_fake_compact_tensor

    def argument(spec):
        if isinstance(spec, str):
            return types[spec](0)
        if len(spec) == 1:
            return fake(types[spec[0]], (cute.sym_int(),), assumed_align=4)
        shape = (cute.sym_int(), spec[1])
        return fake(types[spec[0]], shape, stride_order=(1, 0), assumed_align=16)

    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, *map(argument, arguments), stream, options=options)
