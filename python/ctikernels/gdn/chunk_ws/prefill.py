# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py,
#   flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""Fused SM12x prefill with bounded single-sequence context parallelism.

M/N use our chunk math: gate before inverse, identity init / zero V, no upstream
T precompute. Segment zero's init is copied to W before final pool writes;
fixup uses FP32 IEEE and the planner uses our provisional constants.

Metadata values follow the prefill contract: nonnegative lengths partition
projected, slots are -1 or valid pool indices, and owned slots are unique.
Mutable workspaces are call-local; compiled executables are shared.
"""

from dataclasses import dataclass

import torch

from .. import _contract
from ._cp import cp_capacity, cp_plan
from ._launch import ChunkPrefillSm120, _compiling


@dataclass(frozen=True)
class PortOptions:
    producer_registers: int = 72  # full/seg; no-output MN caps this at 56
    iket: bool = False
    prefetch: int = 2  # z-load grouping only; raw K/Q/V use TMA.
    watchdog: bool = False
    cp: bool = True
    cp_segments: int | None = None
    pdl: bool = True
    remap_y: bool = False  # bank-friendly lane ownership; measure spill tradeoff
    q_lookahead: bool = False  # publish Q(c+1) after V(c), before y/K

    def __post_init__(self):
        if type(self.cp) is not bool:
            raise TypeError("cp must be a bool")
        if type(self.remap_y) is not bool:
            raise TypeError("remap_y must be a bool")
        if type(self.q_lookahead) is not bool:
            raise TypeError("q_lookahead must be a bool")
        if type(self.pdl) is not bool:
            raise TypeError("pdl must be a bool")
        if self.cp_segments is not None and (
            type(self.cp_segments) is not int or self.cp_segments < 1
        ):
            raise ValueError("cp_segments must be a positive integer or None")
        if type(self.producer_registers) is not int or self.producer_registers not in (
            32,
            40,
            48,
            56,
            64,
            72,
            80,
        ):
            raise ValueError("producer_registers must be one of 32, 40, 48, 56, 64, 72, 80")
        if type(self.prefetch) is not int or self.prefetch not in (2, 4, 8):
            raise ValueError("prefetch must be one of 2, 4, 8")
        if type(self.watchdog) is not bool:
            raise TypeError("watchdog must be a bool")
        if type(self.iket) is not bool:
            raise TypeError("iket must be a bool")

    @property
    def math_registers(self):
        return self.registers()[1]

    def registers(self, variant="full"):
        producer = min(self.producer_registers, 56) if variant == "mn" else self.producer_registers
        return producer, ((504 - producer) // 2 // 8) * 8


def compile_key(hk, hv, row_stride, options, variant="full"):
    """T/N/S are intentionally absent; projected's physical row stride is static."""
    return variant, hk, hv, row_stride, options


DEFAULT_OPTIONS = PortOptions()
# launch_history executables by (caller's device or None, hk, hv, row stride, compiler options), shared by
# every chunk variant and every PortPrefill; written under _compiling.
_COMMITS: dict = {}


def compile_prefill(
    hk,
    hv,
    row_stride,
    compiler_options,
    options=DEFAULT_OPTIONS,
    variant="full",
    *,
    device=None,
):
    """Compile one chunk variant and the shared history commit for ``compiler_options``.

    ``device`` (the runtime caller's CUDA device index) only scopes the shared history
    cache; this function never queries CUDA, so offline builds with fake tensors and an
    explicit ``--gpu-arch`` work without a visible GPU (they share the ``None`` scope).
    """
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int8, Int32, Uint8
    from cutlass.runtime import make_fake_stream

    from ._fused import FusedChunkKernel, launch_history

    if variant not in ("full", "mn", "seg"):
        raise ValueError("unknown fused variant")
    fake = cute.runtime.make_fake_tensor
    geo = _contract.Geometry(hk, hv)

    def vector(dtype, length=None, align=4):
        return fake(
            dtype, (cute.sym_int() if length is None else length,), (1,), assumed_align=align
        )

    if row_stride % 8:
        raise ValueError("projected TMA row stride must be a multiple of 8 BF16 elements")
    x = fake(BFloat16, (cute.sym_int(), geo.proj_width), (row_stride, 1), assumed_align=16)
    weight = fake(BFloat16, (geo.conv_dim, 4), (4, 1), assumed_align=2)
    alog, bias, norm = vector(Float32, hv), vector(Float32, hv), vector(Float32, 128)
    cu, slots, initial = vector(Int32), vector(Int32), vector(Int8, align=1)
    rows = vector(Int32)
    state = fake(
        Float32,
        (cute.sym_int(), hv, 128, 128),
        (hv * 128 * 128, 128 * 128, 128, 1),
        assumed_align=4,
    )
    history = fake(
        BFloat16,
        (cute.sym_int(), 3, geo.conv_dim),
        (3 * geo.conv_dim, geo.conv_dim, 1),
        assumed_align=2,
    )
    y = fake(BFloat16, (cute.sym_int(), hv * 128), (hv * 128, 1), assumed_align=16)
    maps = vector(Uint8, align=128)
    if variant == "full":
        workspace = state
    else:
        cap = cute.sym_int()
        workspace = fake(
            Float32,
            (3, cap, hv, 128, 128),
            (cap * hv * 16384, hv * 16384, 16384, 128, 1),
            assumed_align=4,
        )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    if options.iket:
        compiler_options += " iket"
    kernel = FusedChunkKernel(
        hk,
        hv,
        options.producer_registers,
        options.iket,
        options.prefetch,
        options.watchdog,
        variant,
        options.pdl,
        options.q_lookahead,
        options.remap_y,
    )
    chunk = cute.compile(
        kernel,
        x,
        weight,
        alog,
        bias,
        norm,
        cu,
        slots,
        initial,
        state,
        history,
        y,
        maps,
        workspace,
        Int32(0),
        Int32(0),
        rows,
        Int32(0),
        stream,
        options=compiler_options,
    )
    chunk.cp2_smem_bytes = kernel.shared_storage.size_in_bytes()
    # The history commit depends only on the geometry, the row stride and the compiler
    # options, never on the chunk variant: every variant at a row stride shares one.
    commit_key = (device, hk, hv, row_stride, compiler_options)
    commit = _COMMITS.get(commit_key)
    if commit is None:
        commit = _COMMITS[commit_key] = cute.compile(
            launch_history,
            x,
            cu,
            slots,
            initial,
            history,
            state,
            rows,
            Int32(0),
            stream,
            options=compiler_options,
        )
    return chunk, commit


class PortPrefill(ChunkPrefillSm120):
    """GatedDeltaNetPrefill contract, D=128, BF16 y, FP32 in-place recurrent state.

    Projected requires a 16-byte aligned base and row stride divisible by eight
    BF16 elements. Construct outside capture and warm each variant/row stride eagerly.
    CP defaults on for host-known N=1 when cp_plan predicts a gain; cp_segments is
    a bounded target override (1 disables). Warm variants before capture. Each call
    allocates descriptor and CP workspaces on its current stream, including in
    graph capture, without host synchronization.
    pdl overlaps CP launch prologues; disable for paired timing. CUDA graph
    capture of PDL launches requires CUDA 12.3 or newer.
    Rounding follows FlashInfer (and vLLM): an FP16 inverse and BF16 NewV and O operands.
    remap_y experiments with four-token warp ownership to remove sO bank conflicts;
    q_lookahead publishes next Q after current V, before output/K work. Both default
    off pending GPU timing; MN ignores both because it produces no output.
    ``capture_rows`` (int32 [N] or None) follows FlashInfer's prefill state checkpoints at
    one point, p = L // 64 * 64, computed on the device: when ``capture_rows[n] >= 0`` and
    p >= 64, the FP32 state after p tokens and the conv history at p (projected rows
    p-3..p-1) are written to that row of ``state``/``conv_state``; otherwise nothing.
    Rows must not alias a slot written by the same call. y and slots are unchanged.
    """

    def __init__(
        self,
        num_k_heads,
        num_v_heads,
        device,
        *,
        producer_registers=72,
        iket=False,
        prefetch=2,
        watchdog=False,
        cp=True,
        cp_segments=None,
        pdl=True,
        q_lookahead=False,
        remap_y=False,
    ):
        geometry = _contract.Geometry(num_k_heads, num_v_heads)
        self.port_options = PortOptions(
            producer_registers,
            iket,
            prefetch,
            watchdog,
            cp,
            cp_segments,
            pdl,
            remap_y,
            q_lookahead,
        )
        self.geometry = geometry
        device = _contract.cuda_device("PortPrefill", device)
        super().__init__(device, num_k_heads, num_v_heads)
        self.compiled = {}
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.cp_capacity = cp_capacity(self.hv, self.num_sms)
        self.cp_fixup = None

    def plan(self, tokens: int, sequences: int) -> tuple[int, int]:
        """Host-only dispatch plan, including an empty packed input."""
        if type(tokens) is not int or tokens < 0:
            raise ValueError("tokens must be a nonnegative integer")
        if type(sequences) is not int or sequences < 1:
            raise ValueError("sequences must be a positive integer")
        if tokens and sequences == 1 and self.port_options.cp:
            return cp_plan(tokens, self.hv, self.num_sms, self.port_options.cp_segments)
        return 1, tokens

    def variants(self, tokens: int, sequences: int) -> tuple[str, ...]:
        """Compiled chunk variants used by this host-known shape."""
        return ("mn", "seg") if self.plan(tokens, sequences)[0] >= 2 else ("full",)

    def supports(self, projected, out=None) -> bool:
        """Host-only direct TMA eligibility; __call__ copies to a misaligned out."""
        return (
            isinstance(projected, torch.Tensor)
            and projected.ndim == 2
            and projected.stride(1) == 1
            and projected.stride(0) % 8 == 0
            and projected.data_ptr() % 16 == 0
            and (out is None or (isinstance(out, torch.Tensor) and out.data_ptr() % 16 == 0))
        )

    def _ensure_compiled(self, row_stride, variants, capturing):
        count = 0
        with _compiling:
            if "mn" in variants and self.cp_fixup is None:
                if capturing:
                    raise RuntimeError("PortPrefill CP must run eagerly before CUDA graph capture")
                from ._cp_fixup import compile_fixup

                self.cp_fixup = compile_fixup(self.hv, self.options, self.port_options.pdl)
                count += 1
            for variant in variants:
                key = compile_key(self.hk, self.hv, row_stride, self.port_options, variant)
                if key not in self.compiled:
                    if capturing:
                        raise RuntimeError(
                            "PortPrefill must run eagerly before CUDA graph capture for this variant/row stride"
                        )
                    self.compiled[key] = compile_prefill(
                        self.hk,
                        self.hv,
                        row_stride,
                        self.options,
                        self.port_options,
                        variant,
                        device=self.device.index,
                    )
                    count += 1
        return count

    def warm_up(self, row_stride: int, *, cp: bool = False, variants=None) -> int:
        """Compile ``variants`` (default full, plus mn/seg when ``cp``), the fixup if
        ``mn`` is among them, and the history commit; return newly cached entries."""
        if variants is None:
            variants = ("full", "mn", "seg") if cp else ("full",)
        variants = tuple(variants)
        if not variants or not set(variants) <= {"full", "mn", "seg"}:
            raise ValueError("variants must name full, mn or seg")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PortPrefill warm_up is forbidden inside CUDA graph capture")
            if (
                type(row_stride) is not int
                or row_stride < self.geometry.proj_width
                or row_stride % 8
            ):
                raise ValueError("row_stride must cover projected and be a multiple of 8")
            return self._ensure_compiled(row_stride, variants, False)

    def __call__(
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
        out=None,
        capture_rows=None,
    ):
        _contract.require_frozen(
            "PortPrefill",
            {
                "conv_weight": conv_weight,
                "A_log": A_log,
                "dt_bias": dt_bias,
                "norm_weight": norm_weight,
            },
        )
        # Reuse the product tensor checks, but allow N > T (empty sequences).
        rows = _contract.check_projected(self.geometry, self.device, projected)
        _contract.check_weights(
            self.geometry, self.device, conv_weight, A_log, dt_bias, norm_weight
        )
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a tensor")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must describe at least one sequence")
        sequences = cu_seqlens.shape[0] - 1
        _contract.check_tensor("cu_seqlens", cu_seqlens, torch.int32, (sequences + 1,), self.device)
        _contract.check_tensor("slot_indices", slot_indices, torch.int32, (sequences,), self.device)
        _contract.check_tensor(
            "has_initial_state", has_initial_state, torch.bool, (sequences,), self.device
        )
        _contract.check_pools(self.geometry, self.device, state, conv_state)
        if capture_rows is not None:
            _contract.check_tensor(
                "capture_rows", capture_rows, torch.int32, (sequences,), self.device
            )
        if projected.stride(0) % 8:
            raise ValueError("projected TMA row stride must be a multiple of 8 BF16 elements")
        if projected.data_ptr() % 16:
            raise ValueError("projected TMA base must be 16-byte aligned")
        _contract.require_no_autograd("PortPrefill", projected, (state, conv_state))
        if out is not None:
            _contract.check_tensor("out", out, torch.bfloat16, (rows, self.hv * 128), self.device)
            if out.requires_grad:
                raise RuntimeError("PortPrefill out must not require gradients")
        segments, seg_len = self.plan(rows, sequences)
        variants = self.variants(rows, sequences)
        destination = out
        with torch.cuda.device(self.device):
            self._ensure_compiled(
                projected.stride(0), variants, torch.cuda.is_current_stream_capturing()
            )
            if out is None or out.data_ptr() % 16:
                out = torch.empty((rows, self.hv * 128), dtype=torch.bfloat16, device=self.device)
            tensormaps = torch.empty(self.nsmid * 128, dtype=torch.uint8, device=self.device)
            # The full kernel erases W; use the already-live state as its dummy argument.
            workspace = (
                torch.empty(
                    (3, segments, self.hv, 128, 128), dtype=torch.float32, device=self.device
                )
                if segments >= 2
                else state
            )
            initial = has_initial_state.view(torch.int8)
            # One binary for both modes: rows are read only when capturing, so the live
            # int32 [N] slots stand in for them otherwise.
            targets = slot_indices if capture_rows is None else capture_rows
            capture = int(capture_rows is not None)
            args = (
                projected,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
                cu_seqlens,
                slot_indices,
                initial,
                state,
                conv_state,
                out,
                tensormaps,
                workspace,
                seg_len if segments >= 2 else 0,
                segments if segments >= 2 else 0,
                targets,
                capture,
            )
            for variant in variants:
                key = compile_key(self.hk, self.hv, projected.stride(0), self.port_options, variant)
                chunk, commit = self.compiled[key]
                # TMA descriptors cannot encode a zero token dimension.
                if rows:
                    chunk(*args)
                if variant == "mn" and segments > 2:
                    self.cp_fixup(workspace, segments)
            commit(
                projected, cu_seqlens, slot_indices, initial, conv_state, state, targets, capture
            )
            if destination is not None and destination is not out:
                destination.copy_(out)
                out = destination
        return out
