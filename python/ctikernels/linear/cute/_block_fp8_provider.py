# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: SGLang 72730d6 tile policy (fp8_blockwise_scaled_mm_sm120).

"""Dense block-FP8 CuTe provider, with lazy DSL imports and symbolic-M compilation.

Warm up every configuration/output dtype on each capture stream before capture.
Tail scratch is per call (zeroed; the graph's pool under capture) unless supplied, e.g. from
make_workspace or a BlockFP8Linear workspace. No host synchronization is needed.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
from dataclasses import dataclass, replace
from math import ceil

import torch

from ctikernels._runtime.workspace import GraphWorkspace

from . import _block_fp8_act_quant as act_quant

VERSION = "4.8.0"
_compiled: dict[tuple, object] = {}
_compiling = threading.Lock()


@dataclass(frozen=True)
class Config:
    bm: int = 128
    bn: int = 128
    stages: int = 2
    warps: int = 8
    swap_ab: bool = False
    raster: str = "group"
    group: int = 8
    scheduler: str = "grid"
    mx: bool = True
    ctas_per_sm: int = 1
    tail_split_threshold: float = 0.5  # Fraction of resident CTA slots.
    pdl: bool = False
    epi: str = "alias"
    min_sk_iters: int = 4
    sk_order: str = "helper_first"
    sk_waves: int = 2

    def __post_init__(self):
        if self.bm not in (32, 64, 128) or self.bn not in (32, 64, 128):
            raise ValueError("cute-fp8: BM/BN must be 32, 64 or 128 (BK=128)")
        if self.warps not in (4, 8) or self.stages not in (2, 3, 4):
            raise ValueError("cute-fp8: need 4/8 MMA warps and 2/3/4 stages")
        if self.raster not in ("m", "n", "group") or self.group < 1:
            raise ValueError("cute-fp8: raster must be m, n or group; group must be positive")
        if self.scheduler not in ("grid", "persistent", "tailsplit"):
            raise ValueError("cute-fp8: unknown scheduler")
        if self.epi not in ("alias", "private"):
            raise ValueError("cute-fp8: epi must be alias or private")
        if self.sk_order not in ("helper_first", "dp_first"):
            raise ValueError("cute-fp8: sk_order must be helper_first or dp_first")
        if self.sk_waves not in (1, 2):
            raise ValueError("cute-fp8: sk_waves must be 1 or 2")
        # ldmatrix.x4 for B needs >=16 output columns per N warp.
        if self.bn // (self.warps // min(4, self.bm // 16)) < 16:
            raise ValueError("cute-fp8: tile too narrow for ldmatrix.x4")
        if self.ctas_per_sm not in (1, 2):
            raise ValueError("cute-fp8: ctas_per_sm must be 1 or 2")
        if not 0 < self.tail_split_threshold <= 1:
            raise ValueError("cute-fp8: tail_split_threshold must be in (0, 1]")
        if not isinstance(self.min_sk_iters, int) or self.min_sk_iters < 1:
            raise ValueError("cute-fp8: min_sk_iters must be a positive integer")
        if self.ctas_per_sm == 2 and (self.smem_bytes + 1024) * 2 > 100 * 1024:
            raise ValueError("cute-fp8: two CTAs exceed the 100 KiB SM budget")
        if self.smem_bytes > 99 * 1024:
            raise ValueError("cute-fp8: configuration exceeds 99 KiB shared memory")

    @property
    def smem_bytes(self):
        prefix = self.stages * (16 + 4 * (self.scale_rows + 1))
        operands = ((prefix + 127) // 128) * 128 + self.stages * (self.bm + self.bn) * 128
        # Both supported output dtypes are 16-bit; the private buffer follows d.
        return operands + (2 * self.bm * self.bn if self.epi == "private" else 0)

    @property
    def scale_rows(self):
        # Leave room for CUDA's 1 KiB/CTA reservation. For the 128-row
        # occupancy-2 tile, consumers fetch the last 16 FP32 scales directly.
        return min(self.token_tile, 112) if self.ctas_per_sm == 2 else self.token_tile

    @property
    def threads(self):
        return 32 * (self.warps + self.producer_warps)

    @property
    def register_budget(self):
        # Round down to the public 256-register warp allocation granularity.
        return min(255, (65536 // (self.ctas_per_sm * self.threads) // 8) * 8)

    @property
    def token_tile(self):
        return self.bn if self.swap_ab else self.bm

    @property
    def weight_tile(self):
        return self.bm if self.swap_ab else self.bn

    @property
    def producer_warps(self):
        return 1

    @property
    def epilogue_swizzle(self):
        # FP32 word offsets. Normal: row[1:0] -> bank[4:3],
        # paired 64-bit stores. Swap: row[2:1] -> bank[4:3], scalar stores.
        # With 16-bit elements, shift the destination bit up to retain the
        # same byte/bank permutation and contiguous eight-byte output reads.
        base = 4 if self.epi == "private" else 3
        return (2, base, self.weight_tile.bit_length() - 1 - base + int(self.swap_ab))


CONFIGS = {
    "swap128x64s2": Config(128, 64, 2, 4, True),
    "swap128x32s3": Config(128, 32, 3, 4, True),
    "normal64x128s2w4": Config(64, 128, 2, 4),
    "swap128x32s4": Config(128, 32, 4, 4, True),
    "swap128x32s4w8": Config(128, 32, 4, 8, True),
    "swap128x64s3": Config(128, 64, 3, 4, True),
    "swap128x64s3w8": Config(128, 64, 3, 8, True),
    "normal32x128s4": Config(32, 128, 4, 4),
    "normal64x128s2": Config(64, 128, 2, 4, ctas_per_sm=2),
    "normal128x64s2": Config(128, 64, 2, 4, ctas_per_sm=2),
    "normal64x128s3": Config(64, 128, 3, 4),
    "normal64x128s3w8": Config(64, 128, 3, 8),
    "normal128x128s2": Config(),
    "normal128x128s3": Config(stages=3),
    "normal128x128s2w8": Config(),
    "normal128x64s3": Config(128, 64, 3, 4),
    "normal64x128s4": Config(64, 128, 4, 4),
    # Two resident CTAs per SM, a 4-stage swap tile, and group-4 raster.
    "swap128x32s2c2": Config(128, 32, 2, 4, True, ctas_per_sm=2),
    "swap64x64s2c2": Config(64, 64, 2, 4, True, ctas_per_sm=2),
    "swap64x32s3c2": Config(64, 32, 3, 4, True, ctas_per_sm=2),
    "swap128x64s4": Config(128, 64, 4, 4, True),
    "swap128x32s4g4": Config(128, 32, 4, 4, True, group=4),
}
for _group in (4, 8, 16):
    CONFIGS[f"normal128x128s2w8g{_group}"] = Config(group=_group)
for _name, _config in tuple(CONFIGS.items()):
    for _scheduler in ("persistent", "tailsplit"):
        CONFIGS[f"{_name}-{_scheduler}"] = replace(_config, scheduler=_scheduler)
    # Keep the original residency policy; omit private variants that exceed
    # either the per-CTA limit or the two-resident-CTA budget.
    _private_bytes = _config.smem_bytes + 2 * _config.bm * _config.bn
    if _private_bytes <= 99 * 1024 and (
        _config.ctas_per_sm == 1 or 2 * (_private_bytes + 1024) <= 100 * 1024
    ):
        for _scheduler in ("persistent", "tailsplit"):
            CONFIGS[f"{_name}-{_scheduler}-ovl"] = replace(
                _config, scheduler=_scheduler, epi="private"
            )
            CONFIGS[f"{_name}-{_scheduler}-ovl-pdl"] = replace(
                _config, scheduler=_scheduler, epi="private", pdl=True
            )
CONFIGS = {
    f"{name}-{atom}": replace(config, mx=atom == "mx")
    for name, config in CONFIGS.items()
    for atom in ("mx", "plain")
}
CONFIGS.update(
    {
        f"{name}-pdl": replace(config, pdl=True)
        for name, config in tuple(CONFIGS.items())
        if config.epi == "alias"
    }
)
# Canonical PDL spelling puts all policy choices before the atom. Retain the
# original atom-pdl aliases for callers and saved tuning tables.
for _name, _config in tuple(CONFIGS.items()):
    if _name.endswith("-pdl"):
        _base, _atom = _name.removesuffix("-pdl").rsplit("-", 1)
        CONFIGS[f"{_base}-pdl-{_atom}"] = _config
# Stream-K also when the last wave is at least half full (-t75: below 75% of
# the resident slots, -t100: any partial wave).
for _name, _config in tuple(CONFIGS.items()):
    if _config.scheduler == "tailsplit":
        for _percent in (75, 100):
            CONFIGS[f"{_name}-t{_percent}"] = replace(_config, tail_split_threshold=_percent / 100)
        for _threshold in ("", "-t100"):
            for _order in ("", "-dpf"):
                for _waves in ("", "-w1"):
                    CONFIGS[f"{_name}{_threshold}{_order}{_waves}"] = replace(
                        _config,
                        tail_split_threshold=1.0 if _threshold else _config.tail_split_threshold,
                        sk_order="dp_first" if _order else "helper_first",
                        sk_waves=1 if _waves else 2,
                    )


@dataclass(frozen=True)
class TailSchedule:
    """CPU mirror of Kernel.run; tile indices precede the raster mapping."""

    slots: int
    k_blocks: int
    dp_tiles: int
    sk_tiles: int
    sk_ctas: int
    sk_order: str = "helper_first"


@dataclass(frozen=True)
class TailWork:
    tile: int
    k_begin: int
    k_end: int
    role: str  # helper, dp, unsplit, or owner (a split tile's last fragment)


def schedule_for(tiles: int, sms: int, k_blocks: int, config: Config) -> TailSchedule:
    """Hybrid DP + Stream-K partition, including persistent fallback."""
    if tiles < 0 or sms < 1 or k_blocks < 1:
        raise ValueError("cute-fp8: invalid schedule dimensions")
    slots = sms * config.ctas_per_sm
    remainder = tiles % slots
    sk_tiles = 0
    if (
        config.scheduler == "tailsplit"
        and k_blocks > 1
        and 0 < remainder < ceil(config.tail_split_threshold * slots)
    ):
        sk_tiles = remainder + slots if config.sk_waves == 2 and tiles >= slots else remainder
    sk_ctas = 0
    if sk_tiles:
        sk_ctas = (
            slots
            if config.sk_waves == 2 and tiles >= slots
            else min(slots, max(1, sk_tiles * k_blocks // config.min_sk_iters))
        )
    return TailSchedule(slots, k_blocks, tiles - sk_tiles, sk_tiles, sk_ctas, config.sk_order)


def tail_work(schedule: TailSchedule, bid: int) -> tuple[TailWork, ...]:
    """Execution order: helper/DP or DP/helper, then unsplit SK, split owner.

    A contiguous interval has at most two partial boundary tiles. Only its
    right boundary can lack the tile's last K block (helper); only its left
    boundary can contain the last block without the first (split owner).
    Interior tiles are unsplit. A range wholly inside one tile is a helper
    unless it reaches the tile's end, in which case it is an owner (or unsplit).
    Thus even a many-way split needs at most one partial slot and one owner
    wait per CTA. The helper is always FIRST SK work; DP tiles never wait and
    their count is equal across CTAs when SK is active (dp_tiles % slots == 0).
    In either order, no owner depends on work following another owner's wait.
    """
    if not 0 <= bid < schedule.slots:
        raise ValueError("cute-fp8: CTA outside resident grid")
    kb = schedule.k_blocks
    dp = [TailWork(t, 0, kb, "dp") for t in range(bid, schedule.dp_tiles, schedule.slots)]
    sk = []
    if bid < schedule.sk_ctas:
        iterations = schedule.sk_tiles * kb
        begin = bid * iterations // schedule.sk_ctas
        end = (bid + 1) * iterations // schedule.sk_ctas
        for tile in range(begin // kb, (end + kb - 1) // kb):
            lo, hi = max(0, begin - tile * kb), min(kb, end - tile * kb)
            role = "helper" if hi < kb else "owner" if lo else "unsplit"
            sk.append(TailWork(schedule.dp_tiles + tile, lo, hi, role))
    helpers = [w for w in sk if w.role == "helper"]
    return tuple(
        (dp + helpers if schedule.sk_order == "dp_first" else helpers + dp)
        + [w for w in sk if w.role == "unsplit"]
        + [w for w in sk if w.role == "owner"]
    )


def tail_contributors(schedule: TailSchedule, tile: int) -> tuple[int, ...]:
    """Helper CTA/slot ids for an SK tile, in increasing K order (owner excluded)."""
    tile -= schedule.dp_tiles
    if not 0 <= tile < schedule.sk_tiles:
        return ()
    iterations = schedule.sk_tiles * schedule.k_blocks
    first = ((tile * schedule.k_blocks + 1) * schedule.sk_ctas - 1) // iterations
    owner = ((tile + 1) * schedule.k_blocks * schedule.sk_ctas - 1) // iterations
    return tuple(range(first, owner))


def tail_workspace(config: Config, sms: int) -> tuple[int, int]:
    """FP32 partial words and Int32 counter words in a single flat allocation.

    Slot b is [chunk][consumer thread][4 floats], one full MMA tile per CTA.
    Counters follow the slots and are indexed by SK-relative tile id. Reserve
    2*S counters for symbolic M; each launch uses fewer than 2*S.
    """
    slots = sms * config.ctas_per_sm
    return slots * config.bm * config.bn, 2 * slots


def available(capability: tuple[int, int]) -> bool:
    """Probe dependency metadata and known hardware without importing the DSL/CUDA."""
    if tuple(capability) not in ((12, 0), (12, 1)) or os.environ.get("CTI_BLOCK_FP8_CUTE") == "0":
        return False
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


_launched: set[tuple] = set()


class CuteFP8Gemm:
    name = "cute-fp8"

    def __init__(self, weight, scale, *, capability, sms, operation="BlockFP8Linear"):
        from ..block_fp8 import _check_weight

        weight, scale = _check_weight(operation, weight, scale, ndim=2)
        n, k = weight.shape
        if n % 128 or k % 128:
            raise ValueError(f"{operation}: cute-fp8 needs N and K multiples of 128")
        if not weight.is_contiguous() or weight.data_ptr() % 16:
            raise ValueError(f"{operation}: cute-fp8 needs a packed, 16-byte aligned weight")
        if not available(capability):
            raise RuntimeError(
                f"{operation}: cute-fp8 needs sm120/sm121 and nvidia-cutlass-dsl=={VERSION}"
            )
        if sms <= 0:
            raise ValueError("cute-fp8: SM count must be positive")
        with torch.cuda.device(weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: construct and warmup before CUDA graph capture")
        self._weight, self._source_scale = weight, scale
        self._scale = scale.to(torch.float32).contiguous()
        self._device = weight.device
        self._capability, self._sms = tuple(capability), sms

    def _check_frozen(self):
        from ..block_fp8 import _require_frozen

        _require_frozen(self.name, {"weight": self._weight, "weight_scale_inv": self._source_scale})

    def _key(self, config, dtype, strided=False):
        # M, input strides, streams and allocation addresses never specialize the code; a
        # row-strided output (row stride != N) compiles its own dynamic-stride variant, so
        # the packed-output kernel keeps its static layout.
        return (
            self._device,
            *self._weight.shape,
            config,
            self._capability,
            self._sms,
            dtype,
            strided,
        )

    def _function(self, config, dtype, strided=False):
        key = self._key(config, dtype, strided)
        fn = _compiled.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: cold configuration; warmup before capture")
            with _compiling:
                fn = _compiled.get(key)
                if fn is None:
                    fn = self._compile(config, dtype, strided)
                    _compiled[key] = fn
        return fn

    def _compile(self, config, dtype, strided=False):
        import cutlass.cute as cute
        from cutlass import BFloat16, Float8E4M3FN, Float16, Float32
        from cutlass.runtime import make_fake_stream

        from ._block_fp8_gemm import Kernel

        n, k = self._weight.shape
        fake = cute.runtime.make_fake_tensor
        m = cute.sym_int64()
        a = fake(Float8E4M3FN, (m, k), (cute.sym_int64(divisibility=16), 1), assumed_align=16)
        w = fake(Float8E4M3FN, (n, k), (k, 1), assumed_align=16)
        # Arbitrary scale strides cover row-major and padded MN-major gated outputs.
        sa = fake(Float32, (m, k // 128), (cute.sym_int64(), cute.sym_int64()), assumed_align=4)
        sw = fake(Float32, (n // 128, k // 128), (k // 128, 1), assumed_align=4)
        out_type = BFloat16 if dtype == torch.bfloat16 else Float16
        out_stride = cute.sym_int64(divisibility=8) if strided else n
        d = fake(out_type, (m, n), (out_stride, 1), assumed_align=16)
        workspace = None
        if config.scheduler == "tailsplit":
            workspace = fake(
                Float32,
                (sum(tail_workspace(config, self._sms)),),
                (1,),
                assumed_align=16,
            )
        major, minor = self._capability
        return cute.compile(
            Kernel(config, self._sms),
            a,
            w,
            sa,
            sw,
            d,
            workspace,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}a",
        )

    def make_workspace(self, config_name):
        """Allocate partial slots and resettable counters once before capture.

        Only partial slots may be poisoned by tests; do not modify the counters.
        Separate concurrent graphs need separate buffers and ordered initialization.
        """
        config = CONFIGS[config_name]
        if config.scheduler != "tailsplit":
            return None
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: allocate workspace before capture")
            return torch.zeros(
                (sum(tail_workspace(config, self._sms)),),
                device=self._device,
                dtype=torch.float32,
            )

    def _workspace(self, config_name, supplied=None):
        config = CONFIGS[config_name]
        if config.scheduler != "tailsplit":
            if supplied is not None:
                raise ValueError("cute-fp8: workspace is only used by tailsplit")
            return None
        size = sum(tail_workspace(config, self._sms))
        group = GraphWorkspace.active() if supplied is None else None
        if group is not None:
            # In a graph workspace every operation shares one buffer per config: the
            # counters' offset depends on the config, and every split resets its flags.
            slot = (f"block_fp8_tailsplit/{config_name}", self._device, torch.float32)
            group.reserve(
                slot,
                (size,),
                lambda max_rows, words: (words,),
                lambda counts: (self.make_workspace(config_name),),
            )
            return group.take(slot, 0, (size,))[0]
        # Without supplied storage, zeroed per-call scratch (the graph's pool under capture);
        # the instance keeps none.
        workspace = supplied
        if workspace is None:
            with torch.cuda.device(self._device):
                workspace = torch.zeros((size,), device=self._device, dtype=torch.float32)
        if (
            workspace.device != self._device
            or workspace.dtype != torch.float32
            or workspace.shape != (size,)
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 16
            or workspace.requires_grad
        ):
            raise ValueError("cute-fp8: workspace needs aligned contiguous FP32 configured storage")
        return workspace

    def prepare(self, config_name, *, out_dtype=torch.bfloat16):
        """Compile and allocate scratch before capture; warmup also loads CUDA modules."""
        self._check_frozen()
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("cute-fp8: output must be BF16 or FP16")
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: prepare must run before capture")
            self._function(CONFIGS[config_name], out_dtype)
            self._workspace(config_name)

    def bind(self, config_name, out_dtype, *, strided=False, quantize=None):
        """``launch(a, a_scale, out)`` for one call class already run eagerly: the GEMM (after
        the CuTe 1x128 quantizer of BF16/FP16 ``a`` when ``quantize`` is its (dtype, vectorized)
        class) on the current device, without checks or a device context. ``out`` is 16-byte
        aligned, with row stride N unless ``strided``; ``a`` matches the class."""
        config = CONFIGS[config_name]
        gemm = self._function(config, out_dtype, strided)
        weight, scale, device = self._weight, self._scale, self._device
        k = weight.shape[1]
        words = sum(tail_workspace(config, self._sms)) if config.scheduler == "tailsplit" else 0
        quantizer = None
        if quantize is not None:
            quantizer = act_quant._compiled[(device, k, *quantize, False)]

        def launch(a, a_scale, out):
            if quantizer is not None:
                m = a.shape[0]
                q = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
                a_scale = torch.empty((m, k // 128), device=device, dtype=torch.float32)
                quantizer(a, q, a_scale)
                a = q
            scratch = torch.zeros((words,), device=device, dtype=torch.float32) if words else None
            gemm(a, weight, a_scale, scale, out, scratch)

        return launch

    def run_x(self, x, out, config_name):
        """Enqueue activation quantization and GEMM, with matched PDL launch policy."""
        self._check_frozen()
        if CONFIGS[config_name].pdl:
            from ._block_fp8_act_quant import quantize
        else:
            from ..block_fp8 import quantize_per_token_group_fp8 as quantize

        q, scale = quantize(x)
        self.run(q, scale, out, config_name)

    def run(self, q, q_scale, out, config_name, *, workspace=None, buffers=None):
        """Enqueue one dense GEMM; inputs are E4M3 [M,K] and FP32 [M,K/128]."""
        from ..block_fp8 import _require_inference

        self._check_frozen()
        _require_inference(self.name, q, q_scale, out)
        config = CONFIGS[config_name]
        if q.dtype != torch.float8_e4m3fn or q_scale.dtype != torch.float32:
            raise TypeError("cute-fp8: input must be E4M3 with FP32 scales")
        if out.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("cute-fp8: output must be BF16 or FP16")
        if any(t.device != self._device for t in (q, q_scale, out)):
            raise ValueError("cute-fp8: operands must share the weight's CUDA device")
        if q.ndim != 2:
            raise ValueError("cute-fp8: input must be [M,K]")
        m, k = q.shape
        n = self._weight.shape[0]
        if k != self._weight.shape[1] or q_scale.shape != (m, k // 128) or out.shape != (m, n):
            raise ValueError("cute-fp8: incompatible operand shapes")
        if out.stride(1) != 1 or out.stride(0) < n:
            raise ValueError("cute-fp8: output needs unit column stride and row stride >= N")
        if not m:
            return
        if q.stride(1) != 1 or q.stride(0) < k or q.stride(0) % 16 or q.data_ptr() % 16:
            raise ValueError("cute-fp8: input needs unit K stride and 16-byte aligned rows")
        target = out
        if out.data_ptr() % 16 or out.stride(0) * out.element_size() % 16:
            # Plain vector stores honor dynamic row strides, but need aligned rows.
            target = (
                torch.empty(out.shape, dtype=out.dtype, device=out.device)
                if buffers is None
                else buffers.get("staging", out.dtype)
            )
        with torch.cuda.device(self._device):
            # A first eager launch loads the module for the context (any stream, any
            # operation sharing the compiled kernel); capture must not load modules.
            strided = target.stride(0) != n
            key = self._key(config, out.dtype, strided)
            if torch.cuda.is_current_stream_capturing() and key not in _launched:
                raise RuntimeError("cute-fp8: call warmup eagerly before CUDA graph capture")
            fn = self._function(config, out.dtype, strided)
            if buffers is not None and config.scheduler == "tailsplit":
                workspace = buffers.get("partial")
                partial_words, _ = tail_workspace(config, self._sms)
                workspace[partial_words:].zero_()
            scratch = self._workspace(config_name, workspace)
            fn(q, self._weight, q_scale, self._scale, target, scratch)
            _launched.add(key)
            if target is not out:
                out.copy_(target)

    def warmup(self, rows, config_name, *, out_dtype=torch.bfloat16):
        """Compile, allocate and launch eagerly on this stream, without synchronization."""
        self._check_frozen()
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cute-fp8: warmup must run before capture")
            self.prepare(config_name, out_dtype=out_dtype)
            n, k = self._weight.shape
            for m in (rows,) if isinstance(rows, int) else rows:
                if m > 0:
                    out = torch.empty((m, n), device=self._device, dtype=out_dtype)
                    # Exercise both activation dtypes eagerly, including module loading.
                    # Padded/unaligned rows must additionally warm up run_x on that layout.
                    if CONFIGS[config_name].pdl:
                        for dtype in (torch.bfloat16, torch.float16):
                            x = torch.zeros((m, k), device=self._device, dtype=dtype)
                            self.run_x(x, out, config_name)
                    else:
                        q = torch.zeros((m, k), device=self._device, dtype=torch.float8_e4m3fn)
                        scale = torch.ones((m, k // 128), device=self._device, dtype=torch.float32)
                        self.run(q, scale, out, config_name)
