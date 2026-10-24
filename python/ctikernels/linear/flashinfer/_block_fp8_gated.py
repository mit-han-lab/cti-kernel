# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: FlashInfer's cute_dsl_sm12x_fc1_act_q1_fp8 is the called library.

"""FlashInfer CuTe DSL provider of the block-FP8 gated FC1 (no imported library until
construction): FlashInfer's SM12x grouped FC1 kernel (gate_up GEMM, SiLU x up and 1 x 128 FP8
quantization in one launch) called with a single expert.

FlashInfer's grouped operands differ from the checkpoint's: the weight is ``[E, 2I, K]`` with
the up rows first, its scales are FP32 ``[E, K / 128, 2I / 128]``, and activation and output
scales are MN-major ``[groups, M_pad]`` with ``M_pad`` the token count rounded up to 4. The
provider reorders a copy of the weight and its scales once, at construction.
"""

from __future__ import annotations

import importlib.metadata

import torch
from packaging.version import Version

GROUP = 128
# FlashInfer releases whose FC1 kernel was qualified here (on nvidia-cutlass-dsl 4.8.0).
FC1_MIN_VERSION = "0.7.1"
_ROW_ALIGN = 4  # FlashInfer pads each expert's MN-major scale rows to 16 bytes


def _flashinfer_fc1(name: str, capability: tuple[int, int], intermediate: int, k: int):
    if capability[0] != 12:
        raise RuntimeError(f"{name}: flashinfer-cute-dsl needs compute capability 12.x")
    if intermediate % GROUP or k % GROUP:
        raise ValueError(f"{name}: flashinfer-cute-dsl needs I and K multiples of 128")
    hint = (
        f"flashinfer-python {FC1_MIN_VERSION} or newer with nvidia-cutlass-dsl "
        "4.7 or newer (the block-fp8 extra pins an older FlashInfer)"
    )
    try:
        version = importlib.metadata.version("flashinfer-python")
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import cute_dsl_sm12x_fc1_act_q1_fp8
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(f"{name}: flashinfer-cute-dsl needs {hint}") from error
    if Version(version) < Version(FC1_MIN_VERSION):
        raise RuntimeError(f"{name}: flashinfer-cute-dsl is qualified on {hint}, found {version}")
    return cute_dsl_sm12x_fc1_act_q1_fp8


def padded_rows(m: int) -> int:
    return -(-m // _ROW_ALIGN) * _ROW_ALIGN


class FlashInferFc1ActQ1:
    name = "flashinfer-cute-dsl"

    def __init__(self, weight, scale, *, capability, operation="BlockFP8GateUpSiLU") -> None:
        two_i, k = weight.shape
        i = two_i // 2
        self._fc1 = _flashinfer_fc1(operation, capability, i, k)
        blocks = i // GROUP
        # One-time up-first copies (2I x K bytes and the small scale grid, transposed to FP32).
        self._weight = torch.cat([weight[i:], weight[:i]])[None].contiguous()
        scale = scale.to(torch.float32)
        self._scale = torch.cat([scale[blocks:], scale[:blocks]]).t()[None].contiguous()
        self._intermediate = i
        self._indptr: dict[int, torch.Tensor] = {}

    def indptr(self, m: int, device: torch.device) -> torch.Tensor:
        """The one-expert row offsets ``[0, m]`` (int32), filled on the device (no host copy)
        once per batch size: a batch size's first call must be eager, not under capture."""
        offsets = self._indptr.get(m)
        if offsets is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "flashinfer-cute-dsl: run each batch size once eagerly before capturing it"
                )
            offsets = torch.full((2,), m, dtype=torch.int32, device=device)
            offsets[:1].zero_()
            self._indptr[m] = offsets
        return offsets

    def run(
        self,
        q: torch.Tensor,
        scale_mn: torch.Tensor,
        out_q: torch.Tensor,
        out_scale_mn: torch.Tensor,
        *,
        indptr: torch.Tensor | None = None,
    ) -> None:
        """``out_q`` [M, I] E4M3 and ``out_scale_mn`` [I / 128, M_pad] FP32 from ``q`` [M, K]
        E4M3 (contiguous) with MN-major ``scale_mn`` [K / 128, M_pad] FP32."""
        # FlashInfer launches on the current device's stream: make it the operands' device.
        with torch.cuda.device(q.device):
            self._fc1(
                q,
                scale_mn,
                self._weight,
                self._scale,
                self.indptr(q.shape[0], q.device) if indptr is None else indptr,
                out_q=out_q,
                out_sf=out_scale_mn,
            )
