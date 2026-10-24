# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original adapter to FlashInfer's public ``mm_fp4`` CUTLASS backend.

FlashInfer (Apache-2.0, https://github.com/flashinfer-ai/flashinfer) is an external
dependency; no implementation is copied here. Our quantizer already writes its
128x4 scale layout, with unshuffled packed E2M1 bytes. Only views are needed at call
time. CTI global scales are descales (reciprocals of FlashInfer quantizer scales),
so alpha is their product, computed on device on every call, including replay.
"""

from __future__ import annotations

import importlib.metadata
import threading
import warnings

import torch

VERSION = "0.6.11.post3"
# Explicit qualification allowlist; never infer compatibility from a newer version.
# 0.7.1 on GB10 still requires the first-replay and FP64 GPU release gates.
QUALIFIED = {
    "0.6.11.post3": frozenset({(10, 0), (10, 3), (11, 0), (12, 0)}),
    "0.7.1": frozenset({(12, 0), (12, 1)}),
    # The 0.1.13 release pin: RTX 5090 GPU gates passed (49 tests, 2026-10-16).
    "0.7.0.post1": frozenset({(12, 0), (12, 1)}),
}


def qualified(version: str, capability: tuple[int, int]) -> bool:
    return capability in QUALIFIED.get(version, ())


def supports_shape(n: int, k: int, capability: tuple[int, int]) -> bool:
    """CUTLASS dense alignment; M is unrestricted (empty calls launch nothing).

    FlashInfer 0.7.0.post1's mm_fp4 requires contiguous BF16/FP16 output and
    128x4 scales, already guaranteed by NVFP4Linear. The sm120/sm121 kernel's
    can_implement checks 32-element FP4 operands and 8-element output alignment;
    the older architecture binding additionally requires N divisible by 32.
    """
    n_alignment = 8 if capability in {(12, 0), (12, 1)} else 32
    return n > 0 and k > 0 and n % n_alignment == 0 and k % 32 == 0


# NVFP4Linear's mm_fp4 backend per compute capability (default "cutlass"). b12x, FlashInfer's
# sm120 CuTe DSL GEMM: on RTX 5090 / RTX PRO 6000 ~28% faster than CUTLASS at decode, at least
# equal at prefill, bitwise equal to it at the 27B shapes, no workspace (2026-10-21); on GB10
# 4-6% slower. It needs N % 8 == 0 (checked by ``supports_shape``); gated GEMMs stay CUTLASS.
LINEAR_BACKEND = {(12, 0): "b12x"}


def linear_backend(capability: tuple[int, int]) -> str:
    return LINEAR_BACKEND.get(tuple(capability), "cutlass")


# Rows from which the GEMM launches without PDL, per device name. RTX PRO 6000, 27B down
# (K=17408), back-to-back calls under CUDA graphs (2026-10-16): with PDL the GEMM cost
# 675 vs 641 us at M=4096 (neutral at M=2048); the RTX 5090 gains from PDL at M=4096.
PDL_OFF_ROWS = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": 4096}


def _scale_view(scale: torch.Tensor) -> torch.Tensor:
    # Preserve physical swizzling; the 2D shape is metadata for mm_fp4's M buckets.
    return scale.view(torch.uint8).view(scale.shape[0] * 128, scale.shape[1] * 4)


class Launcher:
    def __init__(self, pdl_off_rows: int | None = None, backend: str = "cutlass"):
        self.pdl_off_rows = pdl_off_rows
        self.backend = backend  # NVFP4Linear's; gated calls use CUTLASS

    def prepare(self, data: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor):
        from ..nvfp4 import blocked_scale

        return data.t(), _scale_view(blocked_scale(scale)).t(), global_scale

    def linear(self, a, weight, out: torch.Tensor, *, alpha=None, backend=None) -> None:
        from flashinfer import mm_fp4

        data, scale, global_scale = weight
        mm_fp4(
            a.data.contiguous(),
            data,
            _scale_view(a.scale),
            scale,
            alpha=a.global_scale * global_scale if alpha is None else alpha,
            out_dtype=out.dtype,
            out=out,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=backend or "cutlass",
            use_nvfp4=True,
            enable_pdl=self.pdl_off_rows is None or a.data.shape[0] < self.pdl_off_rows,
        )

    def prepare_gated(self, gate, up):
        # Shared scalar storage remains shared after in-place updates. Equal values
        # in independent tensors are insufficient: either can change on replay.
        shared = gate.global_scale.data_ptr() == up.global_scale.data_ptr()
        if not shared:
            return (self.prepare(*gate), self.prepare(*up)), False
        data = torch.stack((gate.data, up.data), dim=1).flatten(0, 1)
        scale = torch.stack((gate.scale, up.scale), dim=1).flatten(0, 1)
        return self.prepare(data, scale, gate.global_scale), True

    def prepare_interleaved(self, data, scale, gate_global, up_global):
        """Views of one interleaved gate/up copy (rows g0, u0, ...; ``scale`` blocked)
        when the globals share storage. Independent globals need two GEMMs over
        contiguous per-slot rows, so only then are the slots copied out (dispatch never
        selects FlashInfer for them; only provider="flashinfer" does)."""
        if gate_global.data_ptr() == up_global.data_ptr():
            return (data.t(), _scale_view(scale).t(), gate_global), True
        from ..nvfp4 import unblocked_scale

        rows, half = data.shape
        slots = data.view(rows // 2, 2, half)
        grid = unblocked_scale(scale, rows, half * 2).reshape(rows // 2, 2, half // 8)
        return (
            tuple(
                self.prepare(slots[:, i].contiguous(), grid[:, i].contiguous(), g)
                for i, g in enumerate((gate_global, up_global))
            ),
            False,
        )

    def gated(self, a, weights, out: torch.Tensor, *, alpha=None) -> None:
        prepared, shared = weights
        if shared:
            self.linear(a, prepared, out, alpha=alpha)
        else:
            # CUTLASS mm_fp4 expects contiguous outputs. Store [gate rows | up rows]
            # in the same allocation; the quantizer reads this layout directly.
            gate_out, up_out = out.view(2, out.shape[0], out.shape[1] // 2).unbind(0)
            self.linear(a, prepared[0], gate_out, alpha=alpha)
            # Read the independent up global on device on every call/replay, and
            # apply alpha in GEMM before its single BF16 rounding.
            self.linear(a, prepared[1], up_out)


_launchers: dict[torch.device, Launcher | None] = {}
_lock = threading.Lock()


def load(device: torch.device) -> Launcher | None:
    """Resolve/build once outside capture, without a kernel launch or tensor readback."""
    with _lock:
        if device not in _launchers:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("prewarm FlashInfer NVFP4 outside CUDA graph capture")
                    _launchers[device] = _resolve(device)
            else:
                _launchers[device] = None
        return _launchers[device]


def _resolve(device: torch.device) -> Launcher | None:
    if device.type != "cuda":
        return None
    capability = torch.cuda.get_device_capability(device)
    if not any(capability in caps for caps in QUALIFIED.values()):
        return None
    try:
        if not qualified(importlib.metadata.version("flashinfer-python"), capability):
            return None
        from flashinfer import mm_fp4
        from flashinfer.gemm.gemm_base import get_cutlass_fp4_gemm_module

        if not callable(mm_fp4):
            return None
        # The pinned package's loader verifies the CUDA toolchain / compiled module.
        # This imports/builds code but does not execute a GEMM or synchronize a tensor.
        with torch.cuda.device(device):
            get_cutlass_fp4_gemm_module(*capability)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - optional backend import/build can fail
        warnings.warn(
            "FlashInfer NVFP4 CUTLASS is unavailable; using the default provider",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return Launcher(
        PDL_OFF_ROWS.get(torch.cuda.get_device_name(device)), linear_backend(capability)
    )
