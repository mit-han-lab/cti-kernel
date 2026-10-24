# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Our portable Triton GDN prefill (plain ``tl.dot``): four launches. Admitted on NVIDIA
cc >= 8.9; bound by auto from cc 10.0; validated on sm103, sm120, sm121 (sm100 compile-only).
Kernels launch on the operation's device and the current stream.

1. ``_features``: conv + SiLU over [history ++ sequence], q/k L2 norm, a and beta;
2. ``_intra``: gates and their chunk cumulative sums, the WY/UT solve, w and u;
3. ``_state``: the state pass, per-chunk states for the output, in-place final states,
   the capture and conv histories (slot-indexed, no gather or scatter);
4. ``_output``: the output and the gated RMSNorm, all chunks in parallel.

Grids depend only on T and N (sequence boundaries are read on the device), so a CUDA graph
captured once follows new lengths, slots and flags on replay. Deterministic: no atomics.
Precision: FP32 accumulation; BF16 operands where fla-core rounds (q, k, v, w, u, the state
for products, v'), so errors against FP64 are of fla-core's order."""

from __future__ import annotations

import torch

from .._contract import Geometry, HashedTuple

# The lowest compute capability the kernels are admitted on.
MIN_CAPABILITY = (8, 9)


class Prefill:
    """``_contract.PrefillKernel`` with state capture (``capture_rows`` [N] int32, a runtime
    argument of the same kernels). Raises RuntimeError below MIN_CAPABILITY."""

    provider = "triton"
    # Host fast path (GatedDeltaNetPrefill._call_fast): plan once per layout, then run.
    fast = True
    # The compiled handles, shared by every operation (see _launch.Launcher).
    launcher = None

    def __init__(self, device: torch.device, geometry: Geometry) -> None:
        capability = torch.cuda.get_device_capability(device)
        if capability < MIN_CAPABILITY:
            raise RuntimeError(
                f"Triton GDN prefill needs compute capability >= 8.9, not {capability}"
            )
        self.geometry = geometry
        # Triton: loaded with the provider, not with this module.
        from . import _features, _intra, _launch, _output, _state

        if Prefill.launcher is None:
            Prefill.launcher = _launch.Launcher()
        self._launch = Prefill.launcher
        self._device = device

        self._features, self._intra = _features.features, _intra.intra
        self._state, self._output = _state.state, _output.output
        self._plans: dict = {}

    @staticmethod
    def variant(projected: torch.Tensor, *tensors) -> tuple:
        """What a call compiles: T, N and captures are not specialized on, the row stride and
        the rows' 16-byte alignment are."""
        return ("triton", projected.stride(0), projected.data_ptr() % 16 == 0)

    def warm_up(self, row_stride: int) -> int:
        """Compile the kernels of calls whose ``projected`` rows are ``row_stride`` apart and
        16-byte aligned (every T, N and capture choice shares them), on a two-token scratch
        call that touches no pool row; returns the number newly compiled.
        Pool, ``out`` and metadata tensors are assumed 16-byte aligned, as allocated."""
        g, before = self.geometry, len(self._launch)
        device = self._device
        if torch.cuda.current_device() != device.index:
            with torch.cuda.device(device):
                return self.warm_up(row_stride)
        rows = torch.zeros(2, row_stride, dtype=torch.bfloat16, device=device)
        projected = rows[:, : g.proj_width]
        weights = (
            torch.zeros(g.conv_dim, 4, dtype=torch.bfloat16, device=device),
            *(torch.zeros(g.v_heads, device=device) for _ in range(2)),
            torch.ones(128, device=device),
        )
        cu = torch.tensor([0, 2], dtype=torch.int32, device=device)
        slots = torch.full((1,), -1, dtype=torch.int32, device=device)
        flags = torch.zeros(1, dtype=torch.bool, device=device)
        state = torch.zeros(1, g.v_heads, 128, 128, device=device)
        conv_state = torch.zeros(1, 3, g.conv_dim, dtype=torch.bfloat16, device=device)
        self(g, projected, *weights, cu, slots, flags, state, conv_state)
        return len(self._launch) - before

    def plan(self, projected: torch.Tensor, count: int) -> tuple:
        """(implementation, capture-guard entry, None): the variant :meth:`variant` names, for
        any T, N and captures."""
        aligned = projected.data_ptr() % 16 == 0
        key = (projected.stride(0), aligned)
        plan = self._plans.get(key)
        if plan is None:
            variant = ("triton", projected.stride(0), aligned)
            plan = self._plans[key] = ("triton", HashedTuple(((), variant)), None)
        return plan

    def run(self, plan, projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens,
            slot_indices, has_initial_state, state, conv_state, out, capture_rows,
            capturing) -> torch.Tensor:  # fmt: skip
        """``__call__`` for checked inputs and their :meth:`plan`."""
        self._launch.capturing = capturing
        try:
            if torch.cuda.current_device() != self._device.index:
                with torch.cuda.device(self._device):
                    return self.run(plan, projected, conv_weight, A_log, dt_bias, norm_weight,
                                    cu_seqlens, slot_indices, has_initial_state, state,
                                    conv_state, out, capture_rows, capturing)  # fmt: skip
            return self(
                self.geometry, projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens,
                slot_indices, has_initial_state, state, conv_state, out=out,
                capture_rows=capture_rows,
            )  # fmt: skip
        finally:
            self._launch.capturing = False

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
        flags = (cu_seqlens, slot_indices, has_initial_state)
        q, k, v, a, beta = self._features(
            geometry, projected, conv_weight, *flags, conv_state, self._launch
        )
        gamma, w, u = self._intra(geometry, k, v, a, beta, A_log, dt_bias, cu_seqlens, self._launch)
        capacity = (q.shape[0] + 63 * (cu_seqlens.shape[0] - 1)) // 64  # chunk table entries
        hbuf = self._state(
            geometry, k, w, u, gamma, capacity, projected, *flags, state, conv_state,
            capture_rows, self._launch,
        )  # fmt: skip
        return self._output(
            geometry, q, k, hbuf, u, gamma, projected, norm_weight, cu_seqlens, self._launch,
            out,
        )  # fmt: skip


__all__ = ["MIN_CAPABILITY", "Prefill"]
