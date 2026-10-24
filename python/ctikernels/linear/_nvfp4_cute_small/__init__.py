# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Experimental ``cute`` small-M provider; importing this package does not import CuTe.

Prepare weights and a Workspace outside capture, then prewarm each configuration. Calls
and graph replays sharing a workspace must be serialized. Each warmed call launches one
kernel. Global factors remain references to the caller's tensors, including gate/up's
distinct factors. This standalone provider is not registered in ``nvfp4.PROVIDERS``.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
from copy import copy
from dataclasses import dataclass

import torch

from ..nvfp4 import NVFP4Activation, NVFP4Weight, scale_shape, unblocked_scale

VERSION = "4.8.0"
PROVIDER = "cute"
PROBES = ("load_only", "no_prologue", "no_mma", "no_reduce")


@dataclass(frozen=True)
class Config:
    n_tile: int = 64
    k_tile: int = 512
    stages: int = 2
    split_k: int = 8
    warps: int = 3  # Legacy count: MMA consumers plus one producer; excludes qwarps.
    max_slice_k: int = 3072  # Ignored compatibility field; A uses a_stages Ktile slots.
    ctas_per_sm: int = 1
    probe: str | None = None
    reduce_mode: str = "global"
    qwarps: int = 1
    a_stages: int = 4
    pdl: bool = False

    def __post_init__(self):
        if any(
            type(v) is not int
            for key, v in vars(self).items()
            if key not in ("probe", "reduce_mode", "pdl")
        ):
            raise ValueError("configuration fields must be integers")
        if type(self.pdl) is not bool:
            raise ValueError("pdl must be a bool")
        if not 1 <= self.qwarps <= 4 or not 2 <= self.a_stages <= 8 or self.a_stages < self.qwarps:
            raise ValueError("qwarps must be 1..4 and a_stages 2..8, a_stages >= qwarps")
        if self.warps + self.qwarps > 8:
            raise ValueError("warps + qwarps must not exceed 8 (256 threads)")
        if self.reduce_mode not in ("global", "cluster"):
            raise ValueError("reduce_mode must be global or cluster")
        if self.reduce_mode == "cluster" and self.split_k not in (2, 4, 8):
            raise ValueError("cluster split_k must be 2/4/8 (portable cluster limit)")
        if self.probe is not None and self.probe not in PROBES:
            raise ValueError("unknown diagnostic probe")
        if self.n_tile not in (16, 32, 64, 128) or self.k_tile not in (128, 256, 512, 1024):
            raise ValueError("n_tile must be 16/32/64/128 and k_tile 128/256/512/1024")
        if not 1 <= self.stages <= 8 or self.split_k < 1 or self.warps not in (2, 3, 5):
            raise ValueError("stages must be 1..8, split_k positive, warps 2/3/5")
        if self.n_tile % (8 * (self.warps - 1)):
            raise ValueError("consumer warps must evenly divide the N tile")
        if self.ctas_per_sm not in (1, 2, 3, 4):
            raise ValueError("ctas_per_sm must be 1..4")
        self.validate(self.k_tile)

    @property
    def smem_limit(self):
        # Reserve at least 1 KiB per block from the 100 KiB SM budget.
        return (100 // self.ctas_per_sm - 1) * 1024

    def slice_k(self, k):
        return self.a_stages * self.k_tile

    def smem_bytes(self, k):
        # A storage is reused for the FP32 result after a CTA-wide mainloop join.
        operands = max(self.a_stages * 9 * self.k_tile, 64 * self.n_tile)
        operands += self.stages * self.n_tile * self.k_tile * 9 // 16
        return operands + 16 * (self.stages + self.a_stages) + 16

    def validate(self, k):
        if k <= 64 or k % 32:
            # K<=64 (a single K64 scale atom) stalled the B/SFB ring in RTX 5090 tests.
            raise ValueError("K must exceed 64 and be divisible by 32")
        if self.smem_bytes(k) > self.smem_limit:
            raise ValueError(
                "configuration exceeds the shared-memory budget; "
                "reduce stages, k_tile, n_tile or requested ctas_per_sm"
            )


_DEFAULT_CONFIG = Config()


def available(device=None) -> bool:
    try:
        if importlib.metadata.version("nvidia-cutlass-dsl") != VERSION:
            return False
    except importlib.metadata.PackageNotFoundError:
        return False
    device = torch.device("cuda" if device is None else device)
    return (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
    )


SHAPES = {
    "9b-linear": (4096, 12288, False),
    "9b-gated": (24576, 4096, True),
    "27b-linear": (5120, 17408, False),
    "27b-gated": (34816, 5120, True),
    "35b-shared-linear": (2048, 512, False),
    "35b-shared-gated": (1024, 2048, True),
}


def presets(device=None, *, gated=False, shape=None):
    """Three unmeasured starting points; shape defaults to the 9B projection."""
    configs = sweep(device, gated=gated, shape=shape)
    if shape is not None and shape.startswith("35b-shared"):
        return configs[0], configs[5], configs[10]
    return configs[0], configs[3], configs[4]


def sweep(device=None, *, gated=False, shape=None):
    """Unmeasured candidates for RTX 5090 (170 SMs), RTX PRO 6000 (188), GB10 (48)."""
    if shape is None:
        shape = "9b-gated" if gated else "9b-linear"
    if shape not in SHAPES or SHAPES[shape][2] != gated:
        raise ValueError("shape must match the selected epilogue")
    _, k, _ = SHAPES[shape]
    device = torch.device("cuda" if device is None else device)
    small = torch.cuda.get_device_capability(device) == (12, 1)
    configs = []

    def add(nt, kt, stages, split=1, warps=3, ctas=1, mode="global", qwarps=1, a_stages=4):
        try:
            cfg = Config(
                nt,
                kt,
                stages,
                split,
                warps,
                4 * kt,
                ctas,
                reduce_mode=mode,
                qwarps=qwarps,
                a_stages=a_stages,
            )
        except ValueError as exc:
            if "shared-memory budget" not in str(exc):
                raise
            return
        cfg.validate(k)
        configs.append(cfg)

    if shape.startswith("35b-shared"):
        for split in (2, 4, 8):
            for nt in (32, 64) if gated else (16, 32):
                for qw in (1, 2):
                    add(nt, 128, 2, split, ctas=2 if small else 3, mode="cluster", qwarps=qw)
    elif gated:
        for nt, stages, ctas in ((64, 4, 1), (128, 3, 1), (64, 5, 2), (128, 4, 2)):
            for qw in (1, 2, 4):
                for ast in (4, 6):
                    # Keep each requested residency; shrink Ktile to fit.
                    kt = 256 if small else 512
                    limit = (100 // ctas - 1) * 1024
                    while (
                        kt > 128
                        and (
                            max(ast * 9 * kt, 64 * nt)
                            + stages * nt * kt * 9 // 16
                            + 16 * (stages + ast)
                            + 16
                        )
                        > limit
                    ):
                        kt //= 2
                    add(
                        nt,
                        kt,
                        stages,
                        warps=5 if nt == 128 and qw < 4 else 3,
                        ctas=ctas,
                        qwarps=qw,
                        a_stages=ast,
                    )
    else:
        kt = 256 if small else 512
        for nt, tile, stages, split in (
            (64, kt, 4, 1),
            (32, 256, 4, 1),
            (32, 256, 4, 2),
            (32, kt, 3, 4),
        ):
            for qw in (1, 2, 4):
                for ast in (4, 6):
                    add(
                        nt,
                        tile,
                        stages,
                        split,
                        mode="global" if split == 1 else "cluster",
                        qwarps=qw,
                        a_stages=ast,
                    )
    return tuple(configs)


def parse_config(value):
    """Parse legacy numeric fields plus optional reduction mode, qN, aN and pdl."""
    parts = value.split(":")
    fields, options = [], {}
    suffix = False
    for part in parts:
        if part in ("global", "cluster"):
            key, option = "reduce_mode", part
        elif part == "pdl":
            key, option = "pdl", True
        elif part.startswith("q"):
            key, option = "qwarps", int(part[1:])
        elif part.startswith("a"):
            key, option = "a_stages", int(part[1:])
        else:
            if suffix:
                raise ValueError("numeric config fields must precede trailing options")
            fields.append(int(part))
            continue
        suffix = True
        if key in options:
            raise ValueError("duplicate config option")
        options[key] = option
    if len(fields) not in (5, 6, 7):
        raise ValueError(
            "configs: N:K:stages:split_k:warps[:max_slice_k[:ctas_per_sm]]"
            "[:global|cluster][:qN][:aN][:pdl]"
        )
    return Config(*fields, **options)


def _frozen(*tensors):
    if any(t.requires_grad for t in tensors):
        raise ValueError("cute NVFP4 weights and scales must not require grad")


def _scalar(tensor, device):
    _frozen(tensor)
    if tensor.device != device or tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError("global scale must be one FP32 element on the operand device")
    if not tensor.is_contiguous():
        raise ValueError("global scale must be contiguous")


@dataclass(frozen=True)
class Weight:
    data: torch.Tensor
    scale: torch.Tensor
    global0: torch.Tensor
    global1: torch.Tensor
    mode: int  # 0: scalar, 1: interleaved gate/up scalars, 2: per-column factors
    sources: tuple[torch.Tensor, ...]

    @property
    def shape(self):
        return self.data.shape[0], self.data.shape[1] * 2


def _tma_scales(scale, n, k):
    # Eight rows per 32-byte sector; each row contributes its four K64 scales.
    dense = unblocked_scale(scale, n, k) if scale.ndim != 2 else scale
    sectors = torch.zeros(
        ((n + 7) // 8 * 8, (k + 63) // 64 * 4), dtype=torch.uint8, device=scale.device
    )
    sectors[:n, : k // 16] = dense.view(torch.uint8)
    return sectors.reshape(-1, 8, (k + 63) // 64, 4).permute(0, 2, 1, 3).contiguous()


def prepare(weight: NVFP4Weight) -> Weight:
    """Prepare once, accepting checkpoint scales or already blocked scales.

    A vector global factor is accepted for already interleaved gated weights. Scalar
    factors and vectors are read directly on every launch; none are expanded/copied.
    """
    _frozen(*weight)
    data, scale, factor = weight
    if data.dtype != torch.uint8 or data.ndim != 2 or not data.is_contiguous():
        raise ValueError("weight data must be contiguous uint8 [N, K/2]")
    n, half = data.shape
    if n < 1 or half < 16 or half % 16 or data.data_ptr() % 16:
        raise ValueError("weight needs N > 0, K divisible by 32 and 16-byte alignment")
    if data.device.type != "cuda" or scale.device != data.device or factor.device != data.device:
        raise ValueError("all weight fields must share a CUDA device")
    if scale.dtype != torch.float8_e4m3fn or not scale.is_contiguous():
        raise ValueError("weight scales must be contiguous E4M3")
    if tuple(scale.shape) not in ((n, half // 8), scale_shape(n, half * 2)):
        raise ValueError("weight scales have the wrong shape")
    if factor.dtype != torch.float32 or factor.numel() not in (1, n) or not factor.is_contiguous():
        raise ValueError("weight global factor must be FP32 scalar or [N]")
    compact = _tma_scales(scale, n, half * 2)
    return Weight(data, compact, factor, factor, 0 if factor.numel() == 1 else 2, tuple(weight))


def prepare_gated(gate: NVFP4Weight, up: NVFP4Weight) -> Weight:
    """Interleave checkpoint rows into a new copy; preserve both mutable device global scales."""
    _frozen(*gate, *up)  # Before even the first conversion or allocation kernel.
    if gate.data.shape != up.data.shape or gate.data.device != up.data.device:
        raise ValueError("gate and up must have the same shape and device")
    if gate.scale.ndim != 2 or up.scale.ndim != 2 or gate.data.shape[0] % 32:
        raise ValueError("gated preparation needs checkpoint scales and I divisible by 32")
    _scalar(gate.global_scale, gate.data.device)
    _scalar(up.global_scale, gate.data.device)
    prepare(gate)
    prepare(up)
    n, half = gate.data.shape
    data = torch.stack((gate.data, up.data), 1).reshape(2 * n, half)
    scales = torch.stack((gate.scale, up.scale), 1).reshape(2 * n, half // 8)
    return prepare_interleaved(data, scales, gate.global_scale, up.global_scale)


def prepare_interleaved(
    data: torch.Tensor, scale: torch.Tensor, gate_global: torch.Tensor, up_global: torch.Tensor
) -> Weight:
    """Gated weight from rows already interleaved (g0, u0, g1, u1, ...): ``data`` is read in
    place (no copy); ``scale`` is the ``[2I, H/16]`` grid or its blocked layout, re-laid out
    once into the kernel's scale sectors. Only these tensors and the globals are retained."""
    _frozen(data, scale, gate_global, up_global)
    device = data.device
    _scalar(gate_global, device)
    _scalar(up_global, device)
    if data.dtype != torch.uint8 or data.ndim != 2 or not data.is_contiguous():
        raise ValueError("weight data must be contiguous uint8 [2I, H/2]")
    n, half = data.shape
    if n < 1 or n % 64 or half < 16 or half % 16 or data.data_ptr() % 16:
        raise ValueError("gated weight needs I divisible by 32, H by 32 and 16-byte alignment")
    if scale.device != device or scale.dtype != torch.float8_e4m3fn or not scale.is_contiguous():
        raise ValueError("weight scales must be contiguous E4M3 on the data device")
    if tuple(scale.shape) not in ((n, half // 8), scale_shape(n, half * 2)):
        raise ValueError("weight scales have the wrong shape")
    return Weight(
        data,
        _tma_scales(scale, n, half * 2),
        gate_global,
        up_global,
        1,
        (data, gate_global, up_global),
    )


class Workspace:
    """Persistent output, split-K scratch and optional quantization/FP32 debug output.

    Construct outside capture. Debug activation padding starts at zero and is never
    overwritten. The production path neither allocates nor initializes device storage.
    """

    def __init__(
        self,
        weight: Weight,
        rows: int,
        config=_DEFAULT_CONFIG,
        *,
        gated=False,
        debug=False,
        outputs=True,
        scratch=None,
    ):
        _frozen(*weight.sources, weight.data, weight.scale, weight.global0, weight.global1)
        if not 1 <= rows <= 32:
            raise ValueError("cute small-M supports 1..32 rows")
        n, k = weight.shape
        config.validate(k)
        if gated and config.n_tile < 32:
            raise ValueError("gated quantization requires n_tile >= 32")
        if gated and n % 64:
            raise ValueError("gated output width must be divisible by 32")
        self.signature = (rows, n, k, config, gated, debug, weight.mode, weight.data.device)
        self.tiles = ((n + 127) // 128 * 128 if gated else n) // config.n_tile
        if not gated:
            self.tiles = (n + config.n_tile - 1) // config.n_tile
        device = weight.data.device
        counters, partial = self.scratch_shapes(weight.shape, rows, config, gated=gated)
        if scratch is None:
            self.counters = torch.zeros(counters, device=device, dtype=torch.int32)
            self.partial = torch.empty(partial, device=device, dtype=torch.float32)
        else:
            # Shared, self-resetting scratch (zeroed counters, any partials) of at least this size.
            self.counters = scratch[0][: math.prod(counters)].view(counters)
            self.partial = scratch[1][: math.prod(partial)].view(partial)
        self.probe_counts = torch.empty(
            ((rows + 15) // 16, self.tiles, config.split_k)
            if config.probe == "load_only"
            else (0,),
            device=device,
            dtype=torch.int32,
        )
        width = n // 2 if gated else n
        self.out_shapes = ((rows, width // 2 if gated else width), scale_shape(rows, width))
        self.out_dtypes = (torch.uint8 if gated else torch.bfloat16, torch.float8_e4m3fn)
        # Without outputs every call must bind caller-owned ``out``. A plain linear never
        # writes its output scales but still passes their (small) buffer to the kernel.
        self.out, self.out_scale = (
            torch.empty(shape, device=device, dtype=dtype) if outputs or (i and not gated) else None
            for i, (shape, dtype) in enumerate(zip(self.out_shapes, self.out_dtypes, strict=True))
        )
        self.debug_data = torch.zeros(
            (rows, k // 2) if debug else (1,), device=device, dtype=torch.uint8
        )
        self.debug_scale = torch.zeros(
            scale_shape(rows, k) if debug else (1,), device=device, dtype=torch.float8_e4m3fn
        )
        self.projection = torch.empty(
            (rows, n) if debug else (1,), device=device, dtype=torch.float32
        )
        self.postact = torch.empty(
            (rows, n // 2) if debug and gated else (1,), device=device, dtype=torch.float32
        )
        self.debug_timeout = os.environ.get("CTI_CUTE_SMALL_DEBUG_TIMEOUT") == "1"
        self.errors = torch.zeros(7 if self.debug_timeout else 1, device=device, dtype=torch.int32)
        self.stamps = torch.empty((1,), device=device, dtype=torch.int64)

    @staticmethod
    def scratch_shapes(weight_shape, rows, config, *, gated=False):
        """(counters, partials) shapes; both empty unless split-K reduces globally."""
        n, _ = weight_shape
        tiles = ((n + 127) // 128 * 128 // config.n_tile) if gated else -(-n // config.n_tile)
        if not (config.split_k > 1 and config.reduce_mode == "global"):
            return (0,), (0,)
        return ((rows + 15) // 16, tiles), (tiles, config.split_k, rows, config.n_tile)

    def check_debug(self):
        """Synchronize/check debug waits; call explicitly after a captured replay.

        Errors are sticky. Discard this workspace after a timeout: split counters
        and outputs may be incomplete. This method is a no-op in production.
        """
        if self.debug_timeout:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("check_debug must run outside CUDA graph capture")
            claimed, kind, stage, phase, bn, split, bm = self.errors.cpu().tolist()
            if claimed:
                name = {
                    1: "full (B/SFB ready)",
                    2: "empty (B/SFB released)",
                    3: "A full (A/SFA ready)",
                    4: "A empty (A/SFA released)",
                }.get(kind, "unknown")
                raise RuntimeError(
                    f"cute small mbarrier timeout: {name}, stage={stage}, phase={phase}, "
                    f"block=({bn}, {split}, {bm}); discard this workspace"
                )


def input_aligned(x):
    """Host metadata only: every row must support unconditional 16-byte loads.

    Rows must not overlap: expanded (zero row stride) views go to another provider.
    """
    return x.data_ptr() % 16 == 0 and x.stride(0) % 8 == 0 and x.stride(0) >= x.shape[1]


def _run(x, weight, input_scale, output_scale, config, workspace, gated):
    if not isinstance(weight, Weight):
        raise TypeError("prepare the weight outside capture before calling cute")
    _frozen(*weight.sources, weight.data, weight.scale, weight.global0, weight.global1)
    _scalar(input_scale, weight.data.device)
    _scalar(output_scale, weight.data.device)
    if x.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("cute NVFP4 is inference-only")
    n, k = weight.shape
    if (
        x.ndim != 2
        or x.shape[1] != k
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.device != weight.data.device
        or x.stride(1) != 1
        or x.stride(0) < k
    ):
        raise ValueError(
            "input must be BF16/FP16 [M, K] with unit inner stride on the weight device"
        )
    if not input_aligned(x):
        raise ValueError("input must have 16-byte alignment and x.stride(0) divisible by 8")
    if workspace.out is None:
        raise ValueError("a workspace without outputs needs out")
    expected = (x.shape[0], n, k, config, gated, workspace.signature[5], weight.mode, x.device)
    if workspace.signature != expected:
        raise ValueError("workspace does not match input, weight, configuration or epilogue")
    if not available(x.device):
        raise RuntimeError("cute NVFP4 requires DSL 4.8.0 and sm120/sm121")
    from . import kernel

    kernel.launch(x, weight, input_scale, output_scale, config, workspace)


def _output_workspace(workspace, out, scale=None):
    """Bind caller-owned output storage without copying or changing cached scratch."""
    device = workspace.signature[-1]
    pairs = zip((out, scale), workspace.out_shapes, workspace.out_dtypes, strict=True)
    for actual, shape, dtype in pairs:
        if actual is None:
            continue
        _frozen(actual)
        if (
            actual.shape != shape
            or actual.dtype != dtype
            or actual.device != device
            or not actual.is_contiguous()
            or actual.data_ptr() % 16
        ):
            raise ValueError("out must match workspace shape, dtype, device and 16-byte alignment")
    bound = copy(workspace)
    bound.out = out
    if scale is not None:
        bound.out_scale = scale
    return bound


def linear(
    x, weight: Weight, input_scale, config=_DEFAULT_CONFIG, *, workspace: Workspace, out=None
):
    """Write BF16 output to ``out`` when supplied, otherwise to workspace storage."""
    target = workspace if out is None else _output_workspace(workspace, out)
    _run(x, weight, input_scale, input_scale, config, target, False)
    return target.out


def gated(
    x,
    weight: Weight,
    input_scale,
    output_scale,
    config=_DEFAULT_CONFIG,
    *,
    workspace: Workspace,
    out=None,
):
    """SiLU(gate)*up in FP32, with no intermediate BF16 rounding, then NVFP4."""
    target = workspace if out is None else _output_workspace(workspace, out.data, out.scale)
    _run(x, weight, input_scale, output_scale, config, target, True)
    return NVFP4Activation(target.out, target.out_scale, output_scale)
