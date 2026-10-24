# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Launch policy and providers for the local block-FP8 Triton kernels."""

from __future__ import annotations

import torch

from ..._runtime.tables import DeviceClass
from ._block_fp8_tuning import GROUP, _config, _groups, tuned_config

__all__ = ["GROUP", "TritonFP8Gemm", "_config", "_groups", "quantizer_key", "tuned_config"]


def _int_key(value: int) -> tuple[bool, bool]:
    """Triton's default integer specialization (not the launch grid)."""
    return value == 1, value % 16 == 0


def quantizer_key(m, k, capability, sms):
    """Quantizer M classes: packed whole groups flatten M to M * groups.

    Include both flattened and general-row signatures. Layout representatives
    cover the remaining strides/pointers; grid dimensions are launch metadata.
    """
    from .._block_fp8_dispatch import quantizer_tile

    return (
        quantizer_tile(capability, sms, m, _groups(k)),
        _int_key(m * _groups(k) if k % GROUP == 0 else m),
        _int_key(m),
    )


class TritonFP8Gemm:
    name = "triton-fp8"
    minimum_capability = (8, 9)
    requirement = "compute capability 8.9 or newer"
    # Above this bound, quantization repeated by each column tile costs more than a launch.
    fused_quantization_rows = 16

    def __init__(self, weight, scale, *, capability, sms, operation="BlockFP8Linear") -> None:
        if capability < self.minimum_capability:
            raise RuntimeError(f"{operation}: {self.name} needs {self.requirement}")
        self._weight = weight
        self._scale = scale
        self._sms = sms
        self._capability = tuple(capability)
        self.warmed_lora = set()
        self._configs = {}  # m -> config (pure in m)

    def config_for(self, m):
        config = self._configs.get(m)
        if config is None:
            if len(self._configs) >= 4096:
                self._configs.clear()
            device = DeviceClass(*self._capability, self._sms)
            config = self._configs[m] = tuned_config(device, m, *self._weight.shape, "fp8")
        return config

    def dispatch_key(self, m, *, quantized=False):
        config = self.config_for(m)
        fused = not quantized and m <= self.fused_quantization_rows
        quantizer = (
            quantizer_key(m, self._weight.shape[1], self._capability, self._sms)
            if not quantized and not fused
            else None
        )
        # GEMM M; split-K stride_os and reduction numel both carry M * N.
        # All other scalar strides are fixed for packed inputs and the chosen
        # output layout. Grid sizes are not specialized kernel arguments.
        split = _int_key(m * self._weight.shape[0]) if config[2] > 1 else None
        return config, fused, _int_key(m), split, quantizer

    def lora_key(self, m, quantized, lora):
        return self.dispatch_key(m, quantized=quantized), lora.rank_tile, lora.low.dtype

    def run(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor | None,
        out: torch.Tensor,
        *,
        buffers=None,
        lora=None,
    ) -> None:
        from . import _block_fp8

        m, k = x.shape
        if lora is not None:
            lora.check(m, out)
            # A LoRA variant compiles on its first eager call; BlockFP8Linear's capture
            # check refuses one not seen here.
            self.warmed_lora.add(self.lora_key(m, x_scale is not None, lora))
        if x_scale is None and m > self.fused_quantization_rows:
            q = (
                torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
                if buffers is None
                else buffers.get("q")
            )
            x_scale = (
                torch.empty((m, _groups(k)), dtype=torch.float32, device=x.device)
                if buffers is None
                else buffers.get("scale")
            )
            _block_fp8.quantize(x, q, x_scale)
            x = q
        config = self.config_for(m)
        _block_fp8.gemm(
            x,
            x_scale,
            self._weight,
            self._scale,
            out,
            config=config,
            partial=buffers.get("partial") if buffers is not None and config[2] > 1 else None,
            lora=lora,
        )
