# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CuTe product dispatch for GDN prefill on RTX 5090, RTX PRO 6000 and GB10.

CuTe serves every valid per-rank Geometry, including empty sequences. Measured
tables select speed, never admission. :func:`load` returns the dispatch for a device
type where CuTe DSL is available, else None (the operation binds our Triton prefill).
The configuration depends only on the token count T, the sequence count N, the geometry,
the device type and its SM count, never on tensor values, so a call reads nothing on the
host.
"""

from __future__ import annotations

import os
import threading

import torch

from ctikernels._runtime.tables import DeviceClass, Dim, TunedOp

from ._contract import Geometry, HashedTuple
from .triton.prefill import Prefill

# Accepted device types (exact device name, compute capability).
RTX_5090 = "RTX 5090"
RTX_PRO_6000 = "RTX PRO 6000"
GB10 = "GB10"
DEVICE_TYPES = {
    ("NVIDIA GeForce RTX 5090", (12, 0)): RTX_5090,
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0)): RTX_PRO_6000,
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)): RTX_PRO_6000,
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", (12, 0)): RTX_PRO_6000,
    ("NVIDIA GB10", (12, 1)): GB10,
}

# Which implementation serves a call, by host-known T and N (never lengths, strides or
# alignment) per device class: v1 ("cute") unless the device's table measured another
# faster on that bucket (benchmark/kernel_tuning/gdn_prefill_compare.py, same-card A-B):
# - "port" runs ONE sequence only (N = 1): a packed batch could hide a sequence the port
#   does not qualify for, and batches sharing (T, N) differ in their length mix. Its RTX 5090
#   rows are the N = 1 bands where it beats v1 and the best market and passes the frozen
#   5-seed gate with max error <= 1.25x fla-core.
# - "triton" is our Triton prefill (triton/prefill.py), on its measured-faster buckets.
# Token edges are powers of two plus 1535, 3071 and 6143, so each port band starts at an
# edge; a bucket inherits the nearest smaller measured bucket of the same N and geometry.
PROVIDER = TunedOp(
    "gdn.prefill.provider",
    version=1,
    space={"provider": ("cute", "port", "triton")},
    dims=(
        Dim(
            "tokens",
            (
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                1535,
                2048,
                3071,
                4096,
                6143,
                8192,
                16384,
                32768,
                65536,
            ),
        ),  # fmt: skip
        Dim("sequences", (1, 2, 4, 8, 16, 32, 64, 128, 256)),
        Dim("k_heads"),
        Dim("v_heads"),
    ),
    default=lambda shape, device: {"provider": "cute"},
)
# The (Hk, Hv) pairs GP1 built and qualified the port at.
PORT_GEOMETRIES = frozenset({(8, 16), (8, 24), (16, 32), (16, 48)})
# Host fast-path call plans, shared by every operation with one dispatch policy (plan()).
_PLANS: dict = {}


def _load_port(hk, hv, device):
    """Optional CuTe port; an unavailable import or constructor retains v1 silently."""
    try:
        from .chunk_ws.prefill import PortPrefill

        return PortPrefill(hk, hv, device)
    except Exception:  # noqa: BLE001 - optional accelerator construction must retain v1.
        return None


def device_type(device: torch.device) -> str | None:
    """The accepted device type of ``device``, or None."""
    name = torch.cuda.get_device_name(device)
    return DEVICE_TYPES.get((name, torch.cuda.get_device_capability(device)))


def load(geometry: Geometry, device: torch.device) -> TritonPrefill | None:
    """The CuTe dispatch on an accepted device type with CuTe available, else None."""
    if not torch.cuda.is_available() or device_type(device) is None:
        return None
    kernel = TritonPrefill(device_type(device), device, geometry.v_heads, hk=geometry.k_heads)
    return kernel if kernel._cute is not None else None


class TritonPrefill:
    """``_contract.PrefillKernel`` for one device; :meth:`variant` names what its first call
    at a shape compiles besides the shape, so the capture guard can require it eagerly."""

    provider = "triton"

    def __init__(self, kind: str, device: torch.device, hv: int = 48, *, hk: int = 16) -> None:
        from . import _cute_prefill

        self.kind = kind
        self._hk, self._hv = hk, hv
        self._capability = torch.cuda.get_device_capability(device)
        # The environment selects a construction-time policy. Separate operations can
        # benchmark forced CuTe and disabled CuTe in the same process.
        self._cute_mode = os.environ.get("CTI_GDN_PREFILL_CUTE")
        self._cute = None if self._cute_mode == "0" else _cute_prefill.load(device, (hk, hv))
        self._port_mode = os.environ.get("CTI_GDN_PREFILL_PORT")
        self._port = None
        # GP1 qualified the port only at these per-rank geometries; others never bind it.
        if (
            self._cute is not None
            and self._port_mode != "0"
            and kind in (RTX_5090, RTX_PRO_6000)
            and self._capability == (12, 0)
            and (hk, hv) in PORT_GEOMETRIES
        ):
            self._port = _load_port(hk, hv, device)
        # The table's device class; without CuTe every bucket is v1 (no table is read).
        sms = getattr(self._cute, "sms", 0)
        self._device_class = DeviceClass(*self._capability, sms)
        # Our Triton prefill, bound when this device's table picks it for this geometry.
        self._triton = None
        rows = PROVIDER.table(self._device_class)
        if self._cute is not None and any(
            key[2:] == (hk, hv) and row["provider"] == "triton" for key, row in rows.items()
        ):
            self._triton = Prefill(device, Geometry(hk, hv))
        self.provider = "cute"
        self._lock = threading.Lock()
        # Host fast path (GatedDeltaNetPrefill): CuTe bound on a real CUDA device.
        self._geometry = Geometry(hk, hv)
        self.fast = self._cute is not None and getattr(self._cute, "fast", False)
        if self.fast:
            self._policy = _cute_prefill.policy_id(
                (
                    "dispatch",
                    self._cute._policy,
                    self._port is not None,
                    self._port_mode,
                    self._triton is not None,
                    self._device_class,
                    kind,
                    self._capability,
                )
            )

    def variant(self, projected, *tensors: torch.Tensor) -> tuple:
        """The row stride the kernels see (see :func:`_compiled_stride`): they compile per
        stride."""
        tokens, count = projected.shape[0], tensors[5].shape[0]
        implementation = self._call_variant(projected, count)
        if implementation == "port":
            return ("port", projected.stride(0), self._port.variants(tokens, count))
        if implementation == "triton":
            return self._triton.variant(projected)
        return (implementation, projected.stride(0), getattr(self._cute, "tune_key", ()))

    def selected_variant(self, tokens: int, sequences: int) -> str:
        """Shape policy (``PROVIDER``); a port choice additionally requires supports(projected)
        at call time. Captures do not affect the choice."""
        if self._port is not None and sequences > 0 and self._port_mode == "1":
            return "port"
        choice = PROVIDER.lookup(
            self._device_class,
            tokens=tokens,
            sequences=max(sequences, 1),
            k_heads=self._hk,
            v_heads=self._hv,
        )["provider"]
        if choice == "port" and (self._port is None or not sequences):
            return "cute"
        if choice == "triton" and self._triton is None:
            return "cute"
        return choice

    def _call_variant(self, projected, count):
        selected = self.selected_variant(projected.shape[0], count)
        if selected == "port" and not self._port.supports(projected):
            return "cute"
        return selected

    def plan(self, projected, count):
        """The host decisions of a call, shared by every operation with this policy: the
        implementation, its capture-guard entry and (CuTe) the launch plan. Keyed by the
        host shapes and the projected row stride and 16-byte alignment, all the decisions
        read (see :meth:`variant`, ``guard`` as GatedDeltaNetPrefill records it)."""
        stride = projected.stride(0)
        tokens = projected.shape[0]
        ident = (self._policy, tokens, count, stride, projected.data_ptr() % 16 == 0)
        plan = _PLANS.get(ident)
        if plan is not None:
            return plan
        implementation = self._call_variant(projected, count)
        if implementation == "triton":
            implementation, guard, launch = self._triton.plan(projected, count)
        elif implementation == "port":
            launch = None
            guard = HashedTuple(((), ("port", stride, self._port.variants(tokens, count))))
        else:
            launch = self._cute.plan(self._geometry, tokens, count, stride)
            guard = HashedTuple(((), ("cute", stride, self._cute.tune_key, launch.key)))
        plan = (implementation, guard, launch)
        with self._lock:
            if len(_PLANS) >= 4096:
                _PLANS.clear()
            return _PLANS.setdefault(ident, plan)

    def run(
        self,
        plan,
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
        out,
        capture_rows,
        capturing,
    ):
        """``__call__`` for checked inputs and their :meth:`plan`."""
        if plan[0] == "triton":
            return self._triton.run(
                plan, projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens,
                slot_indices, has_initial_state, state, conv_state, out, capture_rows, capturing,
            )  # fmt: skip
        if plan[0] == "port":
            return self._port(
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
                out=out,
                capture_rows=capture_rows,
            )
        return self._cute.run(
            plan[2],
            self._geometry,
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
            out,
            capture_rows,
            capturing,
        )

    def port_variants(self, max_tokens: int, max_sequences: int) -> frozenset:
        """The port variant tuples (see PortPrefill.variants) that calls of at most
        ``max_tokens`` rows and ``max_sequences`` sequences select, given supported
        projected rows; host only.

        Table choices are constant within a ``PROVIDER`` bucket (edge + 1 to the next edge).
        The port's plan depends on T only through its 64-token chunk count, and at N >= 2
        it is always the full variant. Every chunk start and bucket bound is therefore a
        sufficient set of probe points: between two of them nothing changes.
        """
        return frozenset(variant for variant, _ in self._port_probes(max_tokens, max_sequences))

    def port_calls(self, max_tokens: int, max_sequences: int) -> dict:
        """Each reachable port variant with one nonempty (T, N) that selects it, for priming
        its kernels at warm-up."""
        calls = {}
        for variant, (t, n) in self._port_probes(max_tokens, max_sequences):
            if n <= t:
                calls.setdefault(variant, (t, n))
        return calls

    def _port_probes(self, max_tokens, max_sequences):
        """(variant, (T, N)) at every probe point that selects the port (see port_variants)."""
        if self._port is None:
            return []
        token_edges, count_edges = (dim.edges for dim in PROVIDER.dims[:2])
        tokens = {0, max_tokens, *range(1, max_tokens + 1, 64)}
        tokens.update(edge + offset for edge in token_edges for offset in (0, 1))
        counts = {1, 2, max_sequences, *(edge + 1 for edge in count_edges)}
        return [
            (self._port.variants(t, n), (t, n))
            for t in sorted(tokens)
            if 0 <= t <= max_tokens
            for n in sorted(counts)
            if 1 <= n <= max_sequences and self.selected_variant(t, n) == "port"
        ]

    def warm_up_port(self, max_tokens, max_sequences, stride):
        """Compile the port executables these calls can reach (:meth:`port_variants`) at
        ``stride``; return the count and the capture-guard variants. A variant no such call
        selects is not compiled: e.g. at Hv=32 on 170 SMs every port-band T plans CP, so the
        full variant is never built."""
        if self._port is None or stride % 8:
            return 0, ()
        reachable = sorted(self.port_variants(max_tokens, max_sequences))
        if not reachable:
            return 0, ()
        names = tuple(dict.fromkeys(name for variants in reachable for name in variants))
        compiled = self._port.warm_up(stride, variants=names)
        return compiled, tuple(("port", stride, v) for v in reachable)

    def __call__(
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
        *,
        out: torch.Tensor | None = None,
        capture_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        count = slot_indices.shape[0]
        implementation = self._call_variant(projected, count)
        if implementation == "triton":
            return self._triton(
                geometry, projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens,
                slot_indices, has_initial_state, state, conv_state, out=out,
                capture_rows=capture_rows,
            )  # fmt: skip
        if implementation == "port":
            return self._port(
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
                out=out,
                capture_rows=capture_rows,
            )
        return self._cute(
            geometry,
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
            out=out,
            capture_rows=capture_rows,
        )


__all__ = ["TritonPrefill", "device_type", "load"]
