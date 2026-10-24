# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/ops/gated_delta_rule/chunk.py`); see THIRD_PARTY_NOTICES.

"""fla-core provider adapter for the slot pools and training forward/backward composition.

``fla-core==0.5.2`` is required: composition uses private chunk forward/backward helpers.
``FLA_DISABLE_BACKEND_DISPATCH=1`` prevents per-call selection of other backends.
``FLA_USE_FAST_OPS=0`` keeps exp/log precision; both variables are pinned before import.
Operations listed in ``WRONG_ON`` fail construction on the affected devices.
Precision: no worse than fla-core against FP64; bitwise where tested (sign of zero aside).
Training computes activation gradients only, with no weight gradients."""

from __future__ import annotations

import importlib.metadata
import threading
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from . import _fla_fixed, _l2norm
from ._contract import CONV, EPS, Geometry, K, V
from ._fla_fixed import DISPATCH_MODULE, DISPATCH_VARIABLE, FAST_OPS_MODULE, FAST_OPS_VARIABLE

DISTRIBUTION = "fla-core"
VERSION = "0.5.2"
# fla-core's chunk size (its kernels' default, and its chunk tables').
CHUNK = 64
# 1 / ln 2: fla-core's gate cumsum works in log2 (fla.ops.utils.constant.RCP_LN2).
RCP_LN2 = 1.4426950408889634

# Operations this fla-core version computes wrongly on a device, by compute capability; their
# construction raises there (no fallback). Measured on prefill (torch 2.13.0, Triton 3.7.1);
# training runs the same chunked forward.
_CHUNK_OUTPUT_ON_SM103 = (
    "its chunked output kernel (chunk_fwd_kernel_o) intermittently writes a wrong 16-row "
    "block of o on sm_103 (GB300: about 1 call in 100 on bit-identical inputs, through "
    "fla-core's public chunk_gated_delta_rule as well)"
)
WRONG_ON = {
    ("GatedDeltaNetPrefill", (10, 3)): _CHUNK_OUTPUT_ON_SM103,
    ("GatedDeltaNetTraining", (10, 3)): _CHUNK_OUTPUT_ON_SM103,
}

_lock = threading.Lock()
_provider: FlaCore | None = None


def describe(device: torch.device) -> str:
    """``device`` with its product name and compute capability, when CUDA is available."""
    if device.type == "cuda" and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        return f"{device} ({torch.cuda.get_device_name(device)}, sm_{major}{minor})"
    return str(device)


def load(operation: str, device: torch.device) -> FlaCore:
    """fla-core under the pinned version and configuration, imported once per process.
    Raises RuntimeError naming ``operation`` and ``device`` when it cannot run them."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = _import(operation, device)
    if device.type == "cuda" and torch.cuda.is_available():
        wrong = WRONG_ON.get((operation, torch.cuda.get_device_capability(device)))
        if wrong is not None:
            raise RuntimeError(
                f"{operation} on {describe(device)} is not supported by {DISTRIBUTION} "
                f"{VERSION}: {wrong}"
            )
    return _provider


def _import(operation: str, device: torch.device) -> FlaCore:
    needs = (
        f"{operation} on {describe(device)} needs {DISTRIBUTION}=={VERSION}, "
        f"the provider of the gdn extra (pip install 'cti-kernels[gdn]')"
    )
    try:
        installed = importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"{needs}; it is not installed") from error
    if installed != VERSION:
        raise RuntimeError(f"{needs}; found {DISTRIBUTION} {installed}")
    _fla_fixed.pin_fla()
    try:
        import fla.modules.conv.triton.ops as conv
        import fla.modules.fused_norm_gate as norm
        import fla.ops.backends as backends
        import fla.ops.utils.op as ops
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.common.gate import fused_beta_sigmoid_bwd, fused_beta_sigmoid_fwd
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
    except ImportError as error:
        raise RuntimeError(f"{needs}; importing it failed: {error}") from error
    if not backends._DISPATCH_DISABLED:
        raise RuntimeError(
            f"{operation} on {describe(device)}: fla-core's {DISPATCH_MODULE} was imported "
            f"before ctikernels.gdn with its backend dispatch on; export {DISPATCH_VARIABLE}=1 "
            "in the process environment"
        )
    # With the variable at 1, fla-core 0.5.2's op module binds exp to libdevice's fast_expf (and
    # log, log2 and tanh to their fast variants).
    if "fast_expf" in ops.exp.fn.__code__.co_names:
        raise RuntimeError(
            f"{operation} on {describe(device)}: fla-core's {FAST_OPS_MODULE} was imported "
            f"before ctikernels.gdn with its fast exp and log on; export {FAST_OPS_VARIABLE}=0 "
            "in the process environment"
        )
    return FlaCore(
        conv_update=conv.causal_conv1d_update,
        recurrent=fused_recurrent_gated_delta_rule,
        gate_cumsum=gdn_gate_chunk_cumsum,
        intra=chunk_gated_delta_rule_fwd_intra,
        state_kernel=chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        output=chunk_fwd_o,
        training_chunk=_fla_fixed.chunk_gated_delta_rule_fwd,
        l2norm=_l2norm.l2norm_forward,
        beta_sigmoid=fused_beta_sigmoid_fwd,
        gated_norm=norm.rms_norm_gated,
        gated_norm_fwd=_fla_fixed.layer_norm_gated_fwd,
        recompute_w_u=_fla_fixed.recompute_w_u_fwd,
        chunk_h=_fla_fixed.chunk_gated_delta_rule_fwd_h,
        dv_local=_fla_fixed.chunk_bwd_dv_local,
        dhu=_fla_fixed.chunk_gated_delta_rule_bwd_dhu,
        dqkwg=_fla_fixed.chunk_bwd_dqkwg,
        wy_bwd=_fla_fixed.prepare_wy_repr_bwd,
        l2norm_bwd=_l2norm.l2norm_backward,
        beta_sigmoid_bwd=fused_beta_sigmoid_bwd,
        local_cumsum=_fla_fixed.chunk_local_cumsum,
    )


def chunk_capacity(tokens: int, sequences: int) -> int:
    """The chunk table's entries for ``tokens`` rows of at most ``sequences`` sequences."""
    return (tokens + (CHUNK - 1) * sequences) // CHUNK


def saved_bytes(geometry: Geometry, tokens: int, sequences: int) -> int:
    """``FlaCore.train_forward``'s saved payload, which the CuTe provider's matches: per row
    projected, q and k, v and o, the q/k 1/norms, the norm's 1/rms, G, beta (FP32), b and A
    (BF16 [Hv, 64]) and the sequence index; the chunk table and cu_seqlens."""
    hk, hv = geometry.k_heads, geometry.v_heads
    row = 2 * geometry.proj_width + 4 * geometry.key_width + 4 * geometry.value_width
    row += 8 * hk + (4 + 4 + 4 + 2 + 2 * CHUNK) * hv + 4
    return tokens * row + 8 * chunk_capacity(tokens, sequences) + 4 * (sequences + 1)


class FlaCore:
    """fla-core's entry points, and the decode, prefill and training compositions over them:
    ``decode`` and ``prefill`` implement ``_contract.DecodeKernel`` and ``PrefillKernel``,
    ``train_forward`` and ``train_backward`` ``TrainKernel``. The training entries are its
    gated RMSNorm forward kernel (which returns the 1/rms the backward reads) and the pieces
    of its chunked backward."""

    def __init__(
        self,
        *,
        conv_update,
        recurrent,
        gate_cumsum,
        intra,
        state_kernel,
        output,
        training_chunk,
        l2norm,
        beta_sigmoid,
        gated_norm,
        gated_norm_fwd,
        recompute_w_u,
        chunk_h,
        dv_local,
        dhu,
        dqkwg,
        wy_bwd,
        l2norm_bwd,
        beta_sigmoid_bwd,
        local_cumsum,
    ) -> None:
        # Triton: loaded with the provider, not with this module.
        from . import _fla_glue, _fla_training, _gated_norm, _rows
        from .triton import _features, _launch, _norm

        self._conv_update = conv_update
        self._recurrent = recurrent
        self._gate_cumsum = gate_cumsum
        self._intra = intra
        self._state_kernel = state_kernel
        self._output = output
        self._training_chunk = training_chunk
        self._l2norm = l2norm
        self._beta_sigmoid = beta_sigmoid
        self._gated_norm = gated_norm
        self._gated_norm_fwd = gated_norm_fwd
        self._recompute_w_u = recompute_w_u
        self._chunk_h = chunk_h
        self._dv_local = dv_local
        self._dhu = dhu
        self._dqkwg = dqkwg
        self._wy_bwd = wy_bwd
        self._l2norm_bwd = l2norm_bwd
        self._beta_sigmoid_bwd = beta_sigmoid_bwd
        self._local_cumsum = local_cumsum
        self._gather, self._write = _rows.gather_rows, _rows.write_rows
        self._prepare, self._commit = _fla_glue.prepare, _fla_glue.commit
        self._features, self._prefill_norm = _features.features, _norm.gated_norm
        self._launch = _launch.Launcher()
        self._gated_norm_bwd = _gated_norm.gated_rms_norm_backward
        self._sequence_index = _fla_training.sequence_index
        self._conv_fwd = _fla_training.conv_forward
        self._conv_bwd = _fla_training.conv_backward
        self._gate_gradient = _fla_training.gate_gradient

    def decode(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, dc, hk, hv = (
            projected.shape[0],
            geometry.conv_dim,
            geometry.k_heads,
            geometry.v_heads,
        )
        keep = slot_indices >= 0
        x = projected[:, :dc]
        history = self._gather(conv_state, slot_indices, keep)
        # fla-core's conv cache is [N, Dc, 4], newest input last; the step shifts column 0 out
        # without reading it, so only columns 1..3 (the history) are filled.
        window = torch.empty(batch, dc, CONV, dtype=conv_state.dtype, device=x.device)
        window[:, :, 1:] = history.transpose(1, 2)
        mixed, _ = self._conv_update(x, window, weight=conv_weight, activation="silu")
        q, k, v = mixed.split([geometry.key_width, geometry.key_width, geometry.value_width], 1)
        b, a = self._gates(geometry, projected)
        o, final = self._recurrent(
            q.reshape(batch, 1, hk, K),
            k.reshape(batch, 1, hk, K),
            v.reshape(batch, 1, hv, V),
            g=a.reshape(batch, 1, hv),
            beta=b.reshape(batch, 1, hv),
            initial_state=self._gather(state, slot_indices, keep),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
        )
        y = self._norm(geometry, o, projected, norm_weight)
        y.masked_fill_(~keep[:, None], 0)
        self._write(state, slot_indices, final, keep)
        self._write(conv_state, slot_indices, torch.cat((history[:, 1:], x[:, None]), 1), keep)
        return y

    def prefill(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Our features kernel (``triton._features``), fla-core's gate, intra-chunk, state and
        output kernels on dense states (its state kernel launched directly with our chunk
        offsets), our gated RMSNorm and the slot glue (``_fla_glue``): 10 launches. No state
        capture. Conv, SiLU, beta and
        fla's kernels are bitwise the fla-core chain on the same inputs; the q/k L2 norm and
        the RMSNorm's sums are tier-1 bounded (no worse than fla-core against FP64)."""
        rows, count, hk, hv = (
            projected.shape[0],
            cu_seqlens.shape[0] - 1,
            geometry.k_heads,
            geometry.v_heads,
        )
        tables = (cu_seqlens, slot_indices, has_initial_state)
        if rows == 0:
            # Every sequence is empty: no final state is read.
            self._commit(geometry, projected, *tables, state, state, conv_state)
            return projected.new_empty((0, geometry.value_width))
        table, offsets, initial = self._prepare(geometry, rows, *tables, state)
        q, k, v, a, beta = (
            t[None]
            for t in self._features(
                geometry, projected, conv_weight, *tables, conv_state, self._launch
            )
        )
        index = {"cu_seqlens": cu_seqlens, "chunk_indices": table, "chunk_size": CHUNK}
        gates = self._gate_cumsum(g=a, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, **index)
        w, u, _ = self._intra(k=k, v=v, g=gates, beta=beta, **index)
        starts = torch.empty((1, table.shape[0], hv, V, K), dtype=k.dtype, device=k.device)
        final = torch.empty((count, hv, V, K), dtype=torch.float32, device=k.device)
        residual = torch.empty_like(u)
        self._state_kernel[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], count * hv)](
            k=k, v=u, w=w, v_new=residual, g=gates, gk=None, h=starts, h0=initial, ht=final,
            cu_seqlens=cu_seqlens, chunk_offsets=offsets, T=rows, H=hk, HV=hv, K=K, V=V,
            BT=CHUNK, STATE_V_FIRST=True,
        )  # fmt: skip
        o = self._output(
            q=q, k=k, v=residual, h=starts, g=gates, scale=K**-0.5, state_v_first=True, **index
        )
        y = self._prefill_norm(geometry, o[0], projected, norm_weight, self._launch)
        self._commit(geometry, projected, *tables, final, state, conv_state)
        return y

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
        sequence index, o with the norm's 1/rms, and what fla-core's chunked backward reads
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
        G, o, A, _, _, _ = self._training_chunk(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            chunk_indices=table,
            use_gate_in_kernel=True,
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
        gated norm's do and dz (``_gated_norm``); dq, dk, dv and dbeta from fla-core's backward
        kernels as ``chunk_gated_delta_rule_bwd`` runs them, with its L2 norm and beta sigmoid
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
        # fla-core 0.5.2's chunked backward (fla/ops/gated_delta_rule/chunk.py,
        # chunk_gated_delta_rule_bwd), piece by piece.
        scale, index = K**-0.5, {"cu_seqlens": cu, "chunk_indices": table}
        w, u = self._recompute_w_u(k=k, v=v, beta=beta, A=A, g=G, **index)
        h, v_new, _ = self._chunk_h(
            k=k, w=w, u=u, g=G, initial_state=None, output_final_state=False, **index
        )
        du = self._dv_local(q=q, k=k, g=G, do=do, scale=scale, **index)
        dh, _, du = self._dhu(
            q=q, k=k, w=w, g=G, h0=None, dht=None, do=do, dv=du, scale=scale, **index
        )
        dq, dk, dw, dg = self._dqkwg(
            q=q, k=k, v=v_new, w=w, g=G, h=h, dv=du, do=do, dh=dh, scale=scale, **index
        )
        del h, v_new, dh, do
        dk2, dv, dbeta, dg2 = self._wy_bwd(k=k, v=v, beta=beta, g=G, A=A, dw=dw, du=du, **index)
        del w, u, du, dw
        dk.add_(dk2)
        dg.add_(dg2)
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
        """(saved, backward): the backward peaks as ``chunk_bwd_dqkwg`` sums dq over each key
        head's value heads, with grad, do, w, u, v', dv' and the per value head dq, dk and dw
        (2 values each per row), the summed dq, dg's key blocks (at most K / 32, fla-core's
        narrowest tiling) and the chunk states h and dh live."""
        hv, values = geometry.v_heads, geometry.value_width
        row = 2 * geometry.proj_width + 16 * values + 2 * geometry.key_width + 4 * hv * (K // 32)
        states = 2 * chunk_capacity(tokens, sequences) * 2 * hv * K * V
        return saved_bytes(geometry, tokens, sequences), tokens * row + states

    @staticmethod
    def _gates(geometry: Geometry, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The b and a columns, contiguous (fla-core's gate kernels read dense rows)."""
        tail = projected[:, geometry.conv_dim + geometry.value_width :]
        return tail[:, : geometry.v_heads].contiguous(), tail[:, geometry.v_heads :].contiguous()

    def _norm(self, geometry, o, projected, norm_weight) -> torch.Tensor:
        rows, hv = projected.shape[0], geometry.v_heads
        z = projected[:, geometry.conv_dim : geometry.conv_dim + geometry.value_width]
        y = self._gated_norm(
            o.reshape(rows * hv, V),
            z.reshape(rows * hv, V),
            norm_weight,
            None,
            activation="silu",
            eps=EPS,
        )
        return y.view(rows, hv * V).to(projected.dtype)


def _chunk_table(cu_seqlens: torch.Tensor, tokens: int) -> torch.Tensor:
    """fla-core's varlen chunk table [capacity, 2] (sequence, chunk within it), from device
    ops only; entries past the batch's chunks name a chunk past the last sequence's end."""
    count = cu_seqlens.shape[0] - 1
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    offsets = F.pad(((lengths + CHUNK - 1) // CHUNK).cumsum(0), (1, 0))
    capacity = (tokens + (CHUNK - 1) * count) // CHUNK
    entry = torch.arange(capacity, device=cu_seqlens.device)
    sequence = torch.searchsorted(offsets[1:], entry, right=True).clamp(max=count - 1)
    return torch.stack((sequence, entry - offsets[sequence]), 1).to(cu_seqlens.dtype)


__all__ = [
    "CHUNK",
    "DISPATCH_VARIABLE",
    "DISTRIBUTION",
    "FAST_OPS_VARIABLE",
    "VERSION",
    "WRONG_ON",
    "FlaCore",
    "describe",
    "load",
]
