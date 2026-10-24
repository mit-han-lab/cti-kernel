# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Small-M fused FC1: per-operation/device scratch, symbolic M within a bucket.

Calls and graph replays sharing an instance MUST be stream-ordered. Two concurrent
streams/graphs need separate op instances, including separately ordered warmups.
Counters are initialized once, outside capture, and reset by the last CTA.
"""

import threading
from dataclasses import dataclass

import torch

from . import _block_fp8_act_quant as act_quant

_compiled: dict[tuple, object] = {}
_ready: set[tuple] = set()
_compiling = threading.Lock()


@dataclass(frozen=True)
class FusedConfig:
    """Explicit engine override; r/warps tune FMA, warps_n/warps_k tune MMA."""

    r: int = 4
    warps: int = 4
    engine: str = "fma"
    warps_n: int = 1
    warps_k: int = 4
    cta_k_split: int = 1

    def __post_init__(self):
        if self.engine not in ("fma", "mma"):
            raise ValueError("fused-fc1: engine must be 'fma' or 'mma'")
        if type(self.r) is not int or self.r not in (2, 4, 8, 16):
            raise ValueError("fused-fc1: R must be 2, 4, 8 or 16")
        if type(self.warps) is not int or self.warps not in (4, 8):
            raise ValueError("fused-fc1: W must be 4 or 8")
        if type(self.warps_n) is not int or self.warps_n not in (1, 2, 4):
            raise ValueError("fused-fc1: warps_n must be 1, 2 or 4")
        if type(self.warps_k) is not int or self.warps_k not in (1, 2, 4, 8):
            raise ValueError("fused-fc1: warps_k must be 1, 2, 4 or 8")
        if self.warps_n * self.warps_k > 16:
            raise ValueError("fused-fc1: MMA CTA must have at most 16 warps")
        if type(self.cta_k_split) is not int or self.cta_k_split != 1:
            raise ValueError("fused-fc1: cta_k_split must be 1 (CTA splitting is unsupported)")

    def shared_bytes(self, m_max):
        if self.engine == "mma":
            tokens = max(8, m_max)
            # Streaming x bytes/scales, then lane-contiguous C fragments and flag.
            return tokens * self.warps_k * 132 + tokens * 16 * self.warps_n * self.warps_k * 4 + 4
        return m_max * 2 * self.r * (4 * self.warps) * 4 + 4


# Per-capability, then (inclusive M ceiling, inclusive I ceiling, config).
# Ordered first match; the engine is part of each cell and the compile cache key.
# Optional (major, minor, SM count) entries precede the capability fallback,
# allowing e.g. RTX 5090 and RTX PRO 6000 to choose different engines at the same M.
CONFIGS = {
    capability: (
        (1, 1024, FusedConfig(4, 4)),
        (1, None, FusedConfig(8, 8)),
        (16, 1024, FusedConfig(engine="mma", warps_n=1, warps_k=4)),
        (16, None, FusedConfig(engine="mma", warps_n=2, warps_k=4)),
    )
    for capability in ((12, 0), (12, 1))
}


def config_for(capability, intermediate, m=1, *, sms=None):
    m_bucket(m)
    entries = CONFIGS.get((*capability, sms), ()) + CONFIGS[tuple(capability)]
    for m_ceiling, ceiling, config in entries:
        if m <= m_ceiling and (ceiling is None or intermediate <= ceiling):
            return config
    raise ValueError("fused-fc1: no configuration")


def m_bucket(m):
    if type(m) is not int or not 1 <= m <= 16:
        raise ValueError("fused-fc1: M must be in 1..16")
    return 1 << (m - 1).bit_length()


def compile_key(device, capability, i, k, dtype, m, config, limit, ue8m0, layout, debug=False):
    """Shared by dispatch warm-up and the fused provider's compile/readiness caches."""
    return device, capability, i, k, dtype, m_bucket(m), config, limit, ue8m0, layout, debug


def workspace_shapes(intermediate):
    if type(intermediate) is not int or intermediate <= 0 or intermediate % 128:
        raise ValueError("fused-fc1: I must be a positive multiple of 128")
    return (16, intermediate), (intermediate // 128,)


class FusedFC1:
    name = "cute-fp8-fused"

    def __init__(
        self,
        weight,
        scale,
        *,
        capability,
        sms=None,
        config=None,
        configs=None,
        swiglu_limit=None,
        scale_ue8m0=False,
        scale_layout="k",
    ):
        from ..block_fp8 import _require_frozen
        from ._block_fp8_provider import available

        _require_frozen(self.name, {"weight": weight, "weight_scale_inv": scale})
        self.i, self.k = weight.shape[0] // 2, weight.shape[1]
        workspace_shapes(self.i)
        if weight.shape[0] != 2 * self.i or self.k <= 0 or self.k % 128:
            raise ValueError("fused-fc1: need [2I,H] with I/H multiples of 128")
        if config is not None and not isinstance(config, FusedConfig):
            raise TypeError("fused_config must be FusedConfig")
        if not available(capability):
            raise RuntimeError("fused-fc1: needs sm120/sm121 and CuTe DSL 4.8.0")
        act_quant.check_options(swiglu_limit, scale_ue8m0, scale_layout)
        self.config = config
        self._configs = dict(configs or {})
        self.capability = tuple(capability)
        self.sms = sms
        self._device = weight.device
        self._source_weight, self._source_scale = weight, scale
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("fused-fc1: construct and warmup before capture")
            self._weight = weight.contiguous()
            if self._weight.data_ptr() % 16:
                self._weight = self._weight.clone()
            self._scale = scale.to(torch.float32).contiguous()
        self.limit, self.ue8m0, self.layout = swiglu_limit, scale_ue8m0, scale_layout
        self._scratch = None

    def _check_frozen(self):
        from ..block_fp8 import _require_frozen

        _require_frozen(
            self.name,
            {
                "weight": self._source_weight,
                "weight_scale_inv": self._source_scale,
            },
        )

    def _workspace(self):
        if self._scratch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("fused-fc1: allocate workspace in eager warmup before capture")
            hshape, cshape = workspace_shapes(self.i)
            self._scratch = (
                torch.empty(hshape, dtype=torch.bfloat16, device=self._device),
                torch.zeros(cshape, dtype=torch.int32, device=self._device),
            )
        return self._scratch

    def config_for(self, m):
        m_bucket(m)
        return (
            self.config
            or self._configs.get(m)
            or config_for(self.capability, self.i, m, sms=self.sms)
        )

    def _compile(self, dtype, bucket, debug, config):
        from ._block_fp8_fused import compile_kernel

        return compile_kernel(
            self.i,
            self.k,
            dtype,
            bucket,
            config,
            self.capability,
            self.limit,
            self.ue8m0,
            self.layout,
            debug,
        )

    def run(self, x, *, debug=False, buffers=None, outputs=None):
        """Debug additionally returns actual in-kernel xq/sx and BF16 gate_up.

        Debug is a separate compiled signature; production launches allocate no
        input-quantization or gate_up intermediates.
        """
        from ..block_fp8 import _check_activation, _require_inference

        self._check_frozen()
        _check_activation(self.name, x, self._device, self.k)
        _require_inference(self.name, x)
        if x.ndim != 2:
            raise ValueError("fused-fc1: x must be [M,H]")
        bucket = m_bucket(x.shape[0])
        config = self.config_for(x.shape[0])
        key = compile_key(
            self._device,
            self.capability,
            self.i,
            self.k,
            x.dtype,
            x.shape[0],
            config,
            self.limit,
            self.ue8m0,
            self.layout,
            debug,
        )
        with torch.cuda.device(self._device):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and key not in _ready:
                raise RuntimeError("fused-fc1: warmup this compiled signature before capture")
            scratch = (
                self._workspace()
                if buffers is None
                else (buffers.get("hidden"), buffers.get("counters").zero_())
            )
            fn = _compiled.get(key)
            if fn is None:
                with _compiling:
                    fn = _compiled.get(key)
                    if fn is None:
                        fn = self._compile(x.dtype, bucket, debug, config)
                        _compiled[key] = fn
            q, scale = (
                act_quant.allocate_outputs(x.shape[0], self.i, self._device, self.layout)
                if outputs is None
                else outputs
            )
            qx = sx = gate_up = None
            if debug:
                qx, sx = act_quant.allocate_outputs(x.shape[0], self.k, self._device, "k")
                gate_up = torch.empty(
                    (x.shape[0], 2 * self.i),
                    dtype=torch.bfloat16,
                    device=self._device,
                )
            fn(x, self._weight, self._scale, q, scale, *scratch, qx, sx, gate_up)
            _ready.add(key)
        return (q, scale, qx, sx, gate_up) if debug else (q, scale)
