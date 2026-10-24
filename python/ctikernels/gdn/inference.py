# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Gated DeltaNet inference: decode and varlen chunked prefill over slot-indexed state pools.

Each operation covers everything between the frozen ``in_proj`` and ``out_proj`` GEMMs of a
Qwen3.5 linear-attention layer: the causal conv (width 4) with SiLU over the q | k | v
columns, the q/k L2 norm (epsilon 1e-6, q scaled by ``K ** -0.5``), the gates
``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``, the delta-rule
recurrence ``S_t = e^g S_{t-1} + k_t (beta (v_t - e^g S_{t-1}^T k_t))^T``, ``o_t = S_t^T q_t``
(value head h reads key head ``h // (Hv // Hk)``) and the gated RMSNorm
``y = norm_weight * o * rsqrt(mean(o^2) + 1e-6) * silu(z)``.

**Geometry.** Per rank, ``Hk`` key heads and ``Hv`` value heads (a multiple of ``Hk``), head
dimensions K = V = 128. ``conv_dim = 2 Hk K + Hv V`` and ``P = conv_dim + Hv V + 2 Hv``.
Qwen3.5-9B and 35B-A3B are (16, 32), Qwen3.5-27B is (16, 48).

**Tensors.** All on the operation's CUDA device.

- ``projected`` ``[rows, P]`` BF16, unit column stride: the merged ``in_proj`` rows,
  columns ``q (Hk K) | k (Hk K) | v (Hv V) | z (Hv V) | b (Hv) | a (Hv)``.
- Frozen weights: ``conv_weight`` ``[conv_dim, 4]`` BF16 (tap j multiplies x_{t-3+j}; the
  checkpoint's ``[conv_dim, 1, 4]`` viewed), ``A_log`` and ``dt_bias`` ``[Hv]`` FP32,
  ``norm_weight`` ``[V]`` FP32 (a plain multiplier), all contiguous. A weight that requires
  grad raises ValueError, in or out of grad mode, before anything else is checked or
  launched; no weight gradient exists. Triton decode also accepts BF16/FP16 ``A_log``,
  ``dt_bias`` and ``norm_weight`` and int64 slots (loads widen to FP32/int64).
- Pools, updated in place: ``state`` ``[S, Hv, V, K]`` FP32 with ``state[s, h, j, i] =
  S[i, j]``, and ``conv_state`` ``[S, 3, conv_dim]`` BF16, the slot's last three pre-conv
  q | k | v rows, oldest first, as bit copies. Both contiguous; a slot index is a row of
  both.
- Returns ``y`` ``[rows, Hv V]`` BF16, new storage unless prefill's ``out=`` is supplied.
  ``out`` must be contiguous, on the operation's device, and must not require gradients.
  Callers must ensure it does not alias inputs or pools. CuTe writes into it directly;
  fallbacks may copy into it.

Prefill optionally exports one intermediate state per sequence (SGLang's rule) with
``capture_rows`` int32 [N], contiguous on the op's device. For sequence n of length L in this
call, p = floor(L / 64) * 64 (``CAPTURE_GRID``) is computed on the device; if
``capture_rows[n] >= 0`` and p >= 64, ``state[row]`` receives the FP32 state the call carries
after its first p tokens and ``conv_state[row]`` the pre-conv rows p-3..p-1 (bit copies),
else nothing is written. p = L when L is a multiple of 64 (the capture equals the slot's new
state). Rows are read on the device only. Callers guarantee nonnegative rows are in bounds and
do not alias active slots or other capture rows (undefined). y, slot states and slot
histories are bitwise the same with and without captures. CuTe, the port and our Triton
prefill capture; fla-core does not.

Shape, dtype, stride and device violations raise TypeError or ValueError. Index values are
not read on the host: callers guarantee unique valid slots in ``[0, S)`` and well-formed
``cu_seqlens``. No call synchronises with the host.

**Limits.** Prefill T <= 2^31-1 because cu_seqlens is int32; pointer offsets stay int64. CuTe v1 folds N and the chunk capacity
``min(N,T) + (T-min(N,T)) // 64`` into grid.x, so each grid product (Hk*C, G*C,
tiles*Hv*N) must fit 2^31-1; N and C may exceed 65535. CuTe v1 and v2 require
Hv <= 65535; v2 also N*Hv*vsplit <= 2^31-1 (grid.x).
Decode's total CTA grid must fit
2^31-1. Reachable grid limits raise ValueError before launch, using host shapes
only; no per-sequence length or pool value is synchronized to the host.

**Implementations.** An operation resolves its provider at construction. Decode on every
CUDA device of compute capability 8.9 or newer, for every valid Geometry (positive Hk, Hv
divisible by Hk, K = V = 128), uses our Triton/Gluon decode (``provider == "triton"``;
qualified on RTX 5090, RTX PRO 6000, GB10 and GB300; unverified on device on sm100). On RTX 5090,
RTX PRO 6000 and GB10, prefill uses CuTe DSL 4.8.0 when installed (``provider == "cute"``). Admission follows
correctness; tables choose speed. GEOMETRY_TUNES and ROW_TUNES are keyed by
(capability, (Hk, Hv)); untuned geometries use our v1 DEFAULT_TUNE with aq_qk=1.
The four existing geometries retain their measured schedules and compile keys.
v2 is selected only by measured rows or explicit CTI_GDN_CUTE_TUNE.
``selected_variant(T, N)`` stays ``cute`` for empty and short calls too; there
is no in-call library fallback. With CuTe unavailable or disabled, and on every other
device of compute capability 8.9 and above, prefill binds our Triton prefill
(``provider == "triton"``, ``triton/prefill.py``: four launches, captures included);
below 8.9 it binds ``fla-core``.
On sm120 RTX 5090/PRO 6000 at (8, 16), (8, 24), (16, 32), (16, 48), host-shape dispatch
can select the FlashInfer-structure port (``selected_variant(T, N) == "port"``), with
context parallelism for long N=1 calls, instead of v1 (``"cute"``). Unsupported projected
alignment uses v1; captures do not affect the choice. ``CTI_GDN_PREFILL_PORT=0`` disables
the port at construction; ``=1`` bypasses the measured table wherever the port is bound and supported.
GB10 retains its CuTe policy.
``CTI_GDN_PREFILL_CUTE=0`` explicitly
disables CuTe at construction for fallback comparisons; ``=1`` binds it where supported (the
provider table still chooses per (T, N) bucket).
CuTe and Triton warmup is per compiled variant and input row stride; T and N are symbolic.
Decode elsewhere, and prefill below 8.9, use the external provider fla-core (``_fla``, the
``cti-kernels[gdn]`` extra);
without it, or with another version, construction raises RuntimeError naming the operation
and the device. Accepted per-GPU-type kernels are entries
of ``_decode_kernel`` and ``_prefill_kernel`` keyed by device type (exact device name and
compute capability), accepted on the geometries they list; they implement
``_contract.DecodeKernel`` or ``PrefillKernel`` and receive checked inputs.

**CUDA graphs.** Calls are capturable: pool rows are addressed on the device, so a graph
captured once follows new slot indices, padding and (prefill) lengths on replay. The
kernels compile (fla-core's also autotune) on their first eager call, which a capture cannot
contain: run each operation once eagerly at the shape (decode: batch size; fla-core prefill:
token and sequence counts) or CuTe/Triton compiled variant (``warm_up``) before capturing it. The fused decode
also compiles per input dtype, row stride of ``projected`` and alignment of its inputs, and the fused
prefill per row stride of ``projected``, so warm them with inputs laid out as the captured
ones. Capturing an unwarmed shape or variant raises RuntimeError. Pool and input storage
must stay in place across replays.
"""

from __future__ import annotations

import torch

from .._runtime.launch import prime
from ._contract import (
    CONV,
    HISTORY,
    DecodeKernel,
    Geometry,
    HashedTuple,
    K,
    PrefillKernel,
    V,
    check_cu_seqlens,
    check_decode,
    check_prefill,
    check_projected,
    check_tensor,
    cuda_device,
    overlaps,
    require_frozen,
    require_no_autograd,
)

# Where auto binds our Triton prefill (CuTe aside); fla-core serves below.
TRITON_PREFILL_CAPABILITY = (8, 9)


class _ReroutedKernel:
    """Attach per-operation provenance without mutating shared provider methods."""

    def __init__(self, kernel, provider):
        self._kernel = kernel
        self.provider = provider

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def _decode_kernel(
    geometry: Geometry, device: torch.device, batch_invariant: bool = False
) -> DecodeKernel:
    """Decode's implementation on ``device``: the fused Triton kernel from compute capability
    8.9, else fla-core (which has no batch-invariant mode)."""
    from . import _triton_decode

    kernel = _triton_decode.load(geometry, device, batch_invariant=batch_invariant)
    if kernel is not None:
        return kernel
    if batch_invariant:
        raise ValueError(
            f"{GatedDeltaNetDecode.__name__}: batch_invariant needs the Triton decode "
            "(compute capability 8.9 or newer, Triton's Gluon installed)"
        )
    from ctikernels._reroute import reroute

    from . import _fla

    provider = reroute(
        GatedDeltaNetDecode.__name__,
        "fla-core",
        "the Triton/Gluon decode is unavailable for this device",
    )
    return _ReroutedKernel(_fla.load(GatedDeltaNetDecode.__name__, device).decode, provider)


def _prefill_kernel(
    geometry: Geometry, device: torch.device, batch_invariant: bool = False
) -> PrefillKernel:
    """CuTe where accepted (sm120/sm121, including TP2), our Triton prefill on the rest of
    cc >= 8.9, fla-core below. ``batch_invariant`` binds our Triton prefill alone on
    cc >= 8.9 and raises below."""
    from . import _triton_prefill

    triton = torch.cuda.get_device_capability(device) >= TRITON_PREFILL_CAPABILITY
    if batch_invariant and not triton:
        raise ValueError(
            f"{GatedDeltaNetPrefill.__name__}: batch_invariant needs the Triton prefill "
            "(compute capability 8.9 or newer)"
        )
    kernel = None if batch_invariant else _triton_prefill.load(geometry, device)
    if kernel is not None:
        return kernel
    if triton:
        from .triton.prefill import Prefill

        return Prefill(device, geometry)
    from ctikernels._reroute import reroute

    from . import _fla

    provider = reroute(
        GatedDeltaNetPrefill.__name__,
        "fla-core",
        "the Triton prefill serves compute capability 8.9 and above",
    )
    return _ReroutedKernel(_fla.load(GatedDeltaNetPrefill.__name__, device).prefill, provider)


class _CaptureGuard:
    """Shapes an operation ran eagerly: capturing any other would compile or autotune inside
    the capture."""

    def __init__(self, operation: str, device: torch.device) -> None:
        self._operation = operation
        self._device = device
        self._eager: set[tuple[int, ...]] = set()

    def active(self) -> bool:
        """Whether the operation's device's current stream is capturing."""
        with torch.cuda.device(self._device):
            return torch.cuda.is_current_stream_capturing()

    def capturing(self, projected: torch.Tensor, shape: tuple[int, ...], variant=()) -> bool:
        """Whether this call is captured; raises if ``shape`` (with the kernel's ``variant``,
        what else its compilation depends on) never ran eagerly."""
        capturing = projected.is_cuda and self.active()
        if capturing and (shape, variant) not in self._eager:
            raise RuntimeError(
                f"{self._operation}: run shape {shape} once eagerly before capturing it "
                "(its provider kernels compile and autotune on their first call"
                + (", per input layout" if variant else "")
                + ")"
            )
        return capturing

    def ran_entry(self, entry: HashedTuple) -> None:
        self._eager.add(entry)

    def ran(self, shape: tuple[int, ...], variant=()) -> None:
        self._eager.add(HashedTuple((shape, variant)))

    def check(self, entry: HashedTuple) -> bool:
        """:meth:`capturing` for a precomputed ``(shape, variant)`` entry (host fast path)."""
        if torch.cuda.current_device() == self._device.index:
            capturing = torch.cuda.is_current_stream_capturing()
        else:
            with torch.cuda.device(self._device):
                capturing = torch.cuda.is_current_stream_capturing()
        if capturing and entry not in self._eager:
            shape, variant = entry
            raise RuntimeError(
                f"{self._operation}: run shape {shape} once eagerly before capturing it "
                "(its provider kernels compile and autotune on their first call"
                + (", per input layout" if variant else "")
                + ")"
            )
        return capturing


class GatedDeltaNetDecode:
    """One token for each of ``rows`` sequences, pools advanced in place.

    ``op(projected, conv_weight, A_log, dt_bias, norm_weight, slot_indices, state,
    conv_state) -> y`` with ``slot_indices`` int32 ``[rows]`` (or int64 on Triton).
    Row r continues from
    ``state[slot]`` and ``conv_state[slot]``; afterwards ``state[slot]`` holds the new state
    and ``conv_state[slot]`` its old rows 1, 2 and the row's pre-conv q | k | v columns.
    Slot -1 is padding: its ``y`` row is zero and no pool row is read or written. Other slots
    are unchanged. Optional keyword ``out`` borrows contiguous BF16 [rows, value_width]
    storage; Triton writes it directly, other providers copy into it. See the module
    docstring for the full contract.

    ``batch_invariant=True`` makes each row's ``y`` and new state bitwise independent of the
    batch it runs in (its size, padding and the other rows): the Triton decode then launches
    whole-head CTAs at every batch size, slower at small batches.
    """

    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        device: torch.device | str | int,
        batch_invariant: bool = False,
    ) -> None:
        name = type(self).__name__
        self._geometry = Geometry(num_k_heads, num_v_heads)
        self._device = cuda_device(name, device)
        self._kernel = _decode_kernel(self._geometry, self._device, batch_invariant=batch_invariant)
        self._guard = _CaptureGuard(name, self._device)

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def provider(self) -> str:
        """The implementation resolved at construction: ``"triton"`` (cti's fused kernel) or
        ``"fla-core"``."""
        return getattr(self._kernel, "provider", "fla-core")

    def reset_scratch(self, pool: torch.Tensor | None = None) -> None:
        """Restore the fused decode's exchange scratch of the state pool ``pool`` (of every pool
        when None) to its initial zeros, on the current stream. Call it, eagerly, after anything
        that may have lost the scratch's contents, e.g. a memory-saver pause/resume without
        backup; it is cheap (a few KiB per slot). A no-op for fla-core; refused under capture."""
        reset = getattr(self._kernel, "reset_scratch", None)
        if reset is None:
            return
        if self._guard.active():
            raise RuntimeError(f"{type(self).__name__}.reset_scratch cannot be captured")
        reset(pool)

    def __call__(
        self,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        _replay: bool = False,
    ) -> torch.Tensor:
        name = type(self).__name__
        weights = {
            "conv_weight": conv_weight,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "norm_weight": norm_weight,
        }
        triton = self.provider == "triton"
        # Weights are frozen on every route and in every grad mode. Only replay's product
        # route (inference-only) admits grad-marked pools/inputs, with grad mode disabled.
        require_frozen(name, weights)
        marked_ok = _replay and triton and not torch.is_grad_enabled()
        decode_check = check_decode
        if triton:
            from ._triton_decode import check_decode as decode_check

        batch = decode_check(
            self._geometry,
            self._device,
            projected,
            *weights.values(),
            slot_indices,
            state,
            conv_state,
        )
        if not marked_ok:
            require_no_autograd(name, projected, (state, conv_state))
        if out is not None:
            check_tensor(
                "out", out, torch.bfloat16, (batch, self._geometry.value_width), self._device
            )
            if torch.is_grad_enabled() and out.requires_grad:
                raise RuntimeError(f"{name} does not support autograd through out")
            # Head CTAs write out while others still read inputs: no partial aliasing.
            if any(
                overlaps(out, x)
                for x in (projected, *weights.values(), slot_indices, state, conv_state)
            ):
                raise ValueError("out must not overlap projected, weights, slots or pools")
        if batch == 0:
            return out if out is not None else projected.new_empty(0, self._geometry.value_width)
        shape = (batch,)
        variant = getattr(self._kernel, "variant", None)
        tensors = (*weights.values(), slot_indices, state, conv_state)
        variant = () if variant is None else variant(projected, *tensors)
        if out is not None and self.provider == "triton":
            variant = (*variant, out.data_ptr() % 16 == 0)
        capturing = self._guard.capturing(projected, shape, variant)
        y = self._kernel(
            self._geometry,
            projected,
            *weights.values(),
            slot_indices,
            state,
            conv_state,
            **({"out": out} if out is not None and self.provider == "triton" else {}),
        )
        if not capturing:
            self._guard.ran(shape, variant)
        if out is not None and y is not out:
            out.copy_(y)
            return out
        return y


class _PrimeScratch:
    """Zeroed stateless inputs of prefill calls up to ``tokens`` rows ``stride`` apart, for
    the one priming call per compiled variant (``_runtime.launch``); allocated at first use."""

    def __init__(self, geometry: Geometry, device: torch.device, tokens: int, stride: int):
        self._shape, self.device, self._tensors = (geometry, tokens, stride), device, None

    def _allocate(self) -> tuple:
        (geometry, tokens, stride), device = self._shape, self.device
        bf16, f32 = torch.bfloat16, torch.float32
        conv, hv = geometry.conv_dim, geometry.v_heads
        rows = torch.zeros(tokens, stride, dtype=bf16, device=device)[:, : geometry.proj_width]
        weights = (
            torch.zeros(conv, CONV, dtype=bf16, device=device),
            torch.zeros(hv, dtype=f32, device=device),
            torch.zeros(hv, dtype=f32, device=device),
            torch.ones(V, dtype=f32, device=device),
        )
        pools = (
            torch.zeros(1, hv, V, K, dtype=f32, device=device),
            torch.zeros(1, HISTORY, conv, dtype=bf16, device=device),
        )
        return rows, weights, pools

    def args(self, tokens: int, count: int, captures: int = 0) -> tuple[tuple, dict]:
        """(positional, keyword) arguments: one sequence per row but the last, which takes
        the rest, all stateless (slot -1); with ``captures``, disabled capture rows (-1). With
        more sequences than rows, the extra sequences are empty, so cu_seqlens never
        decreases."""
        if self._tensors is None:
            self._tensors = self._allocate()
        (rows, weights, pools), device = self._tensors, self.device
        ends = [*(min(end, tokens) for end in range(1, count)), tokens]
        assert all(b >= a for a, b in zip((0, *ends), ends, strict=False)), ends
        cu = torch.tensor([0, *ends], dtype=torch.int32, device=device)
        slots = torch.full((count,), -1, dtype=torch.int32, device=device)
        initial = torch.zeros(count, dtype=torch.bool, device=device)
        call = (rows[:tokens], *weights, cu, slots, initial, *pools)
        if not captures:
            return call, {}
        disabled = torch.full((count,), -1, dtype=torch.int32, device=device)
        return call, {"capture_rows": disabled}

    def run(self, op, geometry, tokens: int, count: int, captures: int = 0) -> None:
        """``op`` on scratch: a CuTe provider (taking ``geometry`` first) or the operation."""
        args, keywords = self.args(tokens, count, captures)
        if geometry is not None:
            args = (geometry, *args)
        op(*args, **keywords)


class GatedDeltaNetPrefill:
    """Varlen chunked prefill of N sequences packed in ``projected`` ``[T, P]``.

    ``op(projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens, slot_indices,
    has_initial_state, state, conv_state, out=None, capture_rows=None) -> y`` with
    ``cu_seqlens`` int32 ``[N + 1]`` (from
    0 to T, nondecreasing; CuTe permits empty sequences), ``slot_indices`` int32 ``[N]`` and
    ``has_initial_state`` bool ``[N]``. A sequence with a slot continues from its pool rows
    when ``has_initial_state`` is set (a later chunk of a chunked prefill), else starts from
    a zero state and zero history without reading its rows (they may hold anything, NaN
    included); either way its final state and the last three pre-conv rows of (history ++
    sequence) are written to its slot. Slot -1 is stateless (the log-prob pass): zero state
    and history, no pool I/O. Other slots are unchanged. See the module docstring for the
    full contract. On sm120 RTX 5090/PRO 6000 the port and v1 can both be bound;
    unsupported projected alignment routes to v1. Empty fresh sequences zero their state and
    all three history rows; empty continued sequences leave their rows untouched. T=0, N>0
    returns empty y after the required pool writes; N=0 returns without a launch after
    validating weights and tensors. CuTe skips zero-sized token grids and reuses its symbolic
    T/N specializations.

    ``capture_rows`` int32 ``[N]`` (contiguous, this device) exports one state per sequence
    at p = floor(L / 64) * 64 when its row is >= 0 and p >= ``CAPTURE_GRID``: ``state[row]``
    gets the FP32 state after the call's first p tokens (continuing from the slot when
    ``has_initial_state``; a slot -1 sequence captures from zero) and ``conv_state[row]``
    rows p-3..p-1 of ``projected``. Row -1, or p < 64, writes nothing; None captures nothing.
    p and the rows are read on the device only, so a captured graph replays with new rows and
    lengths. Nonnegative rows must be in the pools and may not alias any active slot or any
    other capture row (undefined). The capture is the state the call itself carries at p
    (bitwise; p = L gives the slot's new state), and y, slot states and slot histories are
    bitwise the same with and without captures. ``supports_state_capture`` is true for CuTe,
    the port and Triton; fla-core raises NotImplementedError before launch.

    ``batch_invariant=True`` makes each sequence's ``y`` rows, final state and history bitwise
    independent of the batch it runs in (T, N, padding and the other sequences): it binds our
    Triton prefill for every T and N (no per-bucket CuTe or port choice; compute capability
    8.9 or newer, ValueError below). By default the provider may change between buckets.
    """

    # The capture point of a sequence of L tokens is floor(L / CAPTURE_GRID) * CAPTURE_GRID.
    CAPTURE_GRID = 64

    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        device: torch.device | str | int,
        batch_invariant: bool = False,
    ) -> None:
        name = type(self).__name__
        self._geometry = Geometry(num_k_heads, num_v_heads)
        self._device = cuda_device(name, device)
        self._kernel = _prefill_kernel(
            self._geometry, self._device, batch_invariant=batch_invariant
        )
        self._guard = _CaptureGuard(name, self._device)
        # Host fast path (CuTe on a CUDA device): see _call_fast.
        self._fast = getattr(self._kernel, "fast", False) is True
        self._bound = None

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def provider(self) -> str:
        """Bound provider: ``cute``, ``triton`` or ``fla-core``."""
        return getattr(self._kernel, "provider", "fla-core")

    @property
    def supports_state_capture(self) -> bool:
        """Whether ``capture_rows`` is accepted: true for ``cute`` (with the port) and
        ``triton``; ``fla-core`` rejects captures before launch."""
        return self.provider in ("cute", "triton")

    def selected_variant(self, tokens: int, sequences: int) -> str:
        """Implementation chosen for host-known (T, N), without launching or reading CUDA.

        ``CTI_GDN_PREFILL_CUTE`` is read at construction: 0 disables CuTe, 1 binds it where
        supported; the device/shape policy (the provider table) chooses per (T, N) bucket. Construct separate
        operations to compare policies in one process. Warm each before graph capture.
        Captures do not affect the choice. A ``port`` answer assumes supported projected
        alignment; the actual call checks it and otherwise uses v1.
        """
        select = getattr(self._kernel, "selected_variant", None)
        if select is None:
            return self.provider
        return select(tokens, sequences)

    def warm_up(
        self,
        max_tokens: int,
        *,
        max_sequences: int | None = None,
        row_stride: int | None = None,
        port: bool = True,
    ) -> int:
        """Compile every CuTe or Triton variant of calls with at most ``max_tokens`` rows and
        ``max_sequences`` sequences (default ``max_tokens``), with or without captures,
        ``projected`` rows ``row_stride`` apart (default ``proj_width``) and 16-byte aligned,
        so that neither such a call nor its capture compiles. Idempotent; returns the number
        newly compiled. fla-core compiles per shape on first call and returns 0.
        Raises RuntimeError inside a CUDA graph capture.

        On the RTX 5090 the port serves long single sequences (see
        ``selected_variant``); only the port variants such calls reach are compiled.
        ``port=False`` skips them, for engines that never send such a sequence: the
        dispatch is unchanged, so a port call then compiles on its first eager run and
        raises inside a capture, as any unwarmed shape does."""
        for name, value, low in (
            ("max_tokens", max_tokens, 1),
            ("max_sequences", max_tokens if max_sequences is None else max_sequences, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}, not {value!r}")
        if not isinstance(port, bool):
            raise TypeError("port must be a bool")
        stride = self._geometry.proj_width if row_stride is None else row_stride
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride < self._geometry.proj_width
        ):
            raise ValueError(f"row_stride must be an integer >= {self._geometry.proj_width}")
        if self.provider == "triton":
            # Inside a capture this raises (the entry is never recorded): it would compile there.
            self._guard.check(HashedTuple(((), ("warm_up",))))
            compiled = self._kernel.warm_up(stride)
            projected = torch.empty(1, stride, dtype=torch.bfloat16, device=self._device)
            self._guard.ran_entry(self._kernel.plan(projected, 1)[1])
            return compiled
        cute = getattr(self._kernel, "_cute", None)
        if self.provider != "cute" or cute is None:
            return 0
        keys = cute.variant_keys(
            self._geometry,
            max_tokens,
            max_tokens if max_sequences is None else max_sequences,
            stride,
            # v1 remains reachable for any shape through projected misalignment, even when
            # the shape table always chooses the port.
        )
        compiled = cute.warm_up(keys)
        scratch = _PrimeScratch(self._geometry, self._device, max_tokens, stride)
        for key, (tokens, count) in keys.items():
            prime((cute, key), scratch.run, cute, self._geometry, tokens, count, key[6])
        if getattr(self, "_fast", False):
            # Size this stream's shared eager scratch for the largest call now, not mid-step.
            from . import _cute_prefill

            sequences = max_tokens if max_sequences is None else max_sequences
            sizes = [0]
            for n in {1, min(sequences, max_tokens)}:
                try:
                    sizes.append(cute.plan(self._geometry, max_tokens, n, stride).nbytes)
                except ValueError:
                    continue  # such calls raise before launching
            _cute_prefill.reserve_scratch(self._device, max(sizes))
        # The guard entries a warm eager call records (see __call__): one per variant, whose
        # executables with and without captures compiled together.
        base = ("cute", stride, getattr(cute, "tune_key", ()))
        for key in keys:
            if not key[6]:
                self._guard.ran((), (*base, key))
        triton = getattr(self._kernel, "_triton", None)
        if triton is not None:
            # The buckets the device table gives to our Triton prefill (T and N symbolic).
            compiled += triton.warm_up(stride)
            projected = torch.empty(1, stride, dtype=torch.bfloat16, device=self._device)
            self._guard.ran_entry(triton.plan(projected, 1)[1])
        warm_port = getattr(self._kernel, "warm_up_port", None) if port else None
        if warm_port is not None:
            count, variants = warm_port(
                max_tokens, max_tokens if max_sequences is None else max_sequences, stride
            )
            compiled += count
            for variant in variants:
                self._guard.ran((), variant)
            calls = self._kernel.port_calls(
                max_tokens, max_tokens if max_sequences is None else max_sequences
            )
            for variant, (tokens, count) in calls.items():
                prime(("gdn-port", self._kernel, stride, variant), scratch.run, self, None,
                      tokens, count)  # fmt: skip
        return compiled

    def __call__(
        self,
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
        if getattr(self, "_fast", False):
            return self._call_fast(
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
            )
        name = type(self).__name__
        weights = {
            "conv_weight": conv_weight,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "norm_weight": norm_weight,
        }
        require_frozen(name, weights)
        shape = check_prefill(
            self._geometry,
            self._device,
            projected,
            *weights.values(),
            cu_seqlens,
            slot_indices,
            has_initial_state,
            state,
            conv_state,
        )
        if capture_rows is not None:
            check_tensor("capture_rows", capture_rows, torch.int32, (shape[1],), self._device)
            if not self.supports_state_capture:
                raise NotImplementedError(
                    f"{name}: capture_rows needs the cute or triton provider, not {self.provider}"
                )
        require_no_autograd(name, projected, (state, conv_state))
        if out is not None:
            check_tensor(
                "out", out, torch.bfloat16, (shape[0], self._geometry.value_width), self._device
            )
            if out.requires_grad:
                raise ValueError(f"{name}: out must not require gradients")
        if shape[1] == 0:
            return out if out is not None else projected.new_empty((0, self._geometry.value_width))
        variant = getattr(self._kernel, "variant", None)
        tensors = (*weights.values(), cu_seqlens, slot_indices, has_initial_state, state)
        variant = () if variant is None else variant(projected, *tensors, conv_state)
        cute = getattr(self._kernel, "_cute", None)
        capture_variant = getattr(cute, "capture_variant", None)
        guard_shape = shape
        if variant and variant[0] in ("port", "triton"):
            guard_shape = ()
        if capture_variant is not None and variant and variant[0] == "cute":
            guard_shape = ()  # CuTe specializes on its variant; T and N are symbolic.
            variant = (
                *variant,
                capture_variant(self._geometry, projected, slot_indices.numel()),
            )
        capturing = self._guard.capturing(projected, guard_shape, variant)
        y = self._kernel(
            self._geometry,
            projected,
            *weights.values(),
            cu_seqlens,
            slot_indices,
            has_initial_state,
            state,
            conv_state,
            **({"capture_rows": capture_rows} if capture_rows is not None else {}),
            **(
                {"out": out}
                if out is not None and variant and variant[0] in ("cute", "port", "triton")
                else {}
            ),
        )
        if out is not None and y is not out:
            out.copy_(y)
            y = out
        if not capturing:
            self._guard.ran(guard_shape, variant)
        return y

    def _bind(self, *tensors: torch.Tensor) -> None:
        """Record weights and pools that passed check_prefill (see :meth:`_is_bound`)."""
        import weakref

        self._bound = tuple((weakref.ref(t), t.data_ptr(), t.shape, t.dtype) for t in tensors)

    def _is_bound(self, *tensors) -> bool:
        """Whether these are the (contiguous) weights and pools of the last fully checked
        call, still contiguous with the same storage pointer, shape and dtype, so their
        contract checks still hold: the device cannot change in place without the pointer,
        and contiguity with the shape fixes the strides."""
        bound = self._bound
        if bound is None:
            return False
        for (ref, pointer, shape, dtype), t in zip(bound, tensors, strict=True):
            if (
                ref() is not t
                or t.data_ptr() != pointer
                or t.shape != shape
                or t.dtype is not dtype
                or not t.is_contiguous()
            ):
                return False
        return True

    def _call_fast(
        self,
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
    ) -> torch.Tensor:
        """__call__ with CuTe bound: the same checks, errors and launches, with the weight and
        pool checks done once per bound set (identity, pointer and layout), and the host
        decisions shared through the kernel's plan (see _triton_prefill.TritonPrefill.plan)."""
        name = "GatedDeltaNetPrefill"
        if self._is_bound(conv_weight, A_log, dt_bias, norm_weight, state, conv_state):
            # The frozen rule runs on every call, before anything else.
            if (
                conv_weight.requires_grad
                or A_log.requires_grad
                or dt_bias.requires_grad
                or norm_weight.requires_grad
            ):
                require_frozen(
                    name,
                    {
                        "conv_weight": conv_weight,
                        "A_log": A_log,
                        "dt_bias": dt_bias,
                        "norm_weight": norm_weight,
                    },
                )
            device = self._device
            rows = check_projected(self._geometry, device, projected)
            count = check_cu_seqlens(device, cu_seqlens, rows, allow_empty=True)
            if rows > 2**31 - 1:
                raise ValueError("prefill T must fit int32 cu_seqlens (T <= 2^31-1)")
            check_tensor("slot_indices", slot_indices, torch.int32, (count,), device)
            check_tensor("has_initial_state", has_initial_state, torch.bool, (count,), device)
        else:
            require_frozen(
                name,
                {
                    "conv_weight": conv_weight,
                    "A_log": A_log,
                    "dt_bias": dt_bias,
                    "norm_weight": norm_weight,
                },
            )
            rows, count = check_prefill(
                self._geometry,
                self._device,
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
            )
            self._bind(conv_weight, A_log, dt_bias, norm_weight, state, conv_state)
        if capture_rows is not None:
            check_tensor("capture_rows", capture_rows, torch.int32, (count,), self._device)
        require_no_autograd(name, projected, (state, conv_state))
        if out is not None:
            check_tensor(
                "out", out, torch.bfloat16, (rows, self._geometry.value_width), self._device
            )
            if out.requires_grad:
                raise ValueError(f"{name}: out must not require gradients")
        if count == 0:
            return out if out is not None else projected.new_empty((0, self._geometry.value_width))
        kernel = self._kernel
        plan = kernel.plan(projected, count)
        capturing = self._guard.check(plan[1])
        y = kernel.run(
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
        )
        if out is not None and y is not out:
            out.copy_(y)
            y = out
        if not capturing:
            self._guard.ran_entry(plan[1])
        return y


__all__ = ["GatedDeltaNetDecode", "GatedDeltaNetPrefill"]
