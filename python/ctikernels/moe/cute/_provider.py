# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shape-only dispatch and thread-safe, stream-independent CuTe compilation.

Warm up each shape/save mode before graph capture. Kernels use the TVM FFI
current-stream environment, so cached executables also work on other streams.
CTI_MOE_CUTE_CONFIG accepts kind:BMxBNxBKsSTAGESwWARPS[gWMxWN][m|n][t].
The optional trailing t enables TMA weight loads; gathered rows use cp.async.
The default grid is 2 x (warps/2), or 1 x warps for BM16; raster defaults to m.
BN32 is the optional N split for paired kernels. All kinds share BM. Saved
backward may use its own config; recompute uses gate_up's entire config. Only
BM/BN/BK/warp grid determine recomputation's MMA order; stages/raster do not.
CTI_MOE_CUTE_PDL=0 disables programmatic dependent launch (on by default).
Down/dX prefetch weights before waiting for the preceding GEMM's outputs.
"""

import importlib.metadata
import os
import re
import threading
from dataclasses import dataclass, replace
from functools import lru_cache

KINDS = ("gate_up", "down", "bwd_slot", "dx_pair")
_LOCK = threading.RLock()
_COMPILED = {}


@dataclass(frozen=True)
class Config:
    bm: int = 64
    bn: int = 64
    bk: int = 64
    stages: int = 3
    warps: int = 4
    raster: str = "m"
    wm: int = 0
    wn: int = 0
    tma: bool = False

    def __post_init__(self):
        if self.wm == self.wn == 0:
            object.__setattr__(self, "wm", 1 if self.bm == 16 else 2)
            object.__setattr__(self, "wn", self.warps // self.wm)
        if (
            self.bm not in (16, 32, 64, 128)
            or self.bn not in (32, 64, 128, 256)
            or self.bk not in (32, 64, 128)
            or self.stages not in range(2, 7)
            or self.warps not in (4, 8)
            or self.raster not in ("m", "n")
            or self.wm not in (1, 2, 4, 8)
            or self.wn not in (1, 2, 4, 8)
            or self.wm * self.wn != self.warps
            or self.bm % (16 * self.wm)
            or self.bn % (8 * self.wn)
        ):
            raise ValueError("Unsupported MoE CuTe tile/warp configuration")

    def workspace_bytes(self, kind, saved=False):
        # Saved backward loads only q's B operand; recompute also loads gate/up.
        paired = kind == "gate_up" or (kind == "bwd_slot" and not saved)
        pipeline = (self.bm + self.bn * (2 if paired else 1)) * self.bk * self.stages * 2
        reduction = self.bm * self.wn * 4 if kind == "bwd_slot" else 0
        epilogue = self.bm * self.bn * 2
        return max(pipeline, reduction, epilogue)

    def shared_bytes(self, kind, saved=False):
        if self.tma:
            # A and each B stage occupy whole 1024-byte units for all supported
            # tiles. Align the workspace to the full 128B-swizzle period, and
            # keep the barriers outside the scratch reused by the epilogue.
            return 1024 + self.workspace_bytes(kind, saved) + self.stages * 8
        return ((self.bm * 4 + 127) // 128) * 128 + self.workspace_bytes(kind, saved)

    def two_ctas_smem(self, kind, saved=False):
        """Shared-memory eligibility only; registers can still limit occupancy."""
        return self.shared_bytes(kind, saved) <= 49 * 1024

    def validate(self, kind, h, ni, nis, saved=False):
        if kind not in KINDS:
            raise ValueError(f"Unknown MoE CuTe kind: {kind!r}")
        paired = kind in ("gate_up", "bwd_slot")
        ndims = (ni, nis) if paired else (h,)
        kdims = (h,) if paired else (ni, nis)
        if (
            (paired and self.bn == 256)
            or any(d % self.bn for d in ndims)
            or any(d % self.bk for d in kdims)
            or self.shared_bytes(kind, saved) > 99 * 1024
        ):
            raise ValueError(f"MoE CuTe {kind} config {self} exceeds shape/shared-memory limits")

    def __str__(self):
        return (
            f"{self.bm}x{self.bn}x{self.bk}s{self.stages}w{self.warps}"
            f"g{self.wm}x{self.wn}{self.raster}{'t' if self.tma else ''}"
        )


def parse_config(value):
    m = re.fullmatch(
        r"(16|32|64|128)x(32|64|128|256)x(32|64|128)s([2-6])w([48])"
        r"(?:g([1248])x([1248]))?([mn]?)(t?)",
        value,
    )
    if m is None:
        raise ValueError(f"Invalid MoE CuTe config: {value!r}")
    return Config(
        *(int(x) for x in m.groups()[:5]), m[8] or "m", int(m[6] or 0), int(m[7] or 0), bool(m[9])
    )


def overrides(value=None):
    result = {}
    value = os.environ.get("CTI_MOE_CUTE_CONFIG", "") if value is None else value
    for entry in filter(None, value.split(",")):
        kind, spec = entry.split(":", 1)
        if kind not in KINDS or kind in result:
            raise ValueError(f"Invalid or duplicate MoE CuTe kind: {kind!r}")
        result[kind] = parse_config(spec)
    if len({c.bm for c in result.values()}) > 1:
        raise ValueError("All MoE CuTe kinds must share BM")
    # A backward-only sweep also tunes the forward used for recompute, when that config
    # fits gate_up's paired budget (saved-only configs may be wider).
    if "bwd_slot" in result and "gate_up" not in result:
        if result["bwd_slot"].shared_bytes("gate_up") <= 99 * 1024:
            result["gate_up"] = result["bwd_slot"]
    return result


# Tokens per chunk, inclusive upper bounds. BM uniquely identifies each bucket.
# Edit this table after measuring; these are starting points, not measured winners.
DISPATCH = {
    # Up to 512 tokens per chunk the GEMMs stream weights at the DRAM roofline (RTX 5090:
    # every 16-128 row tile within 3%); 16-row tiles waste the least MMA on padding.
    ((12, 0), 512): {
        "gate_up": "16x32x64s4w4g1x4",
        "down": "16x64x64s4w4g1x4",
        "bwd_slot": "16x64x64s4w4g1x4",
        "dx_pair": "16x64x64s4w4g1x4",
    },
    # Measured (RTX 5090, Qwen3.5 EP1 and Qwen3-MoE, 2048-8192 tokens): small tiles with
    # shallow rings beat 128-row / 8-warp tiles, which lose occupancy to registers. The
    # BK=32 four-stage down/dx_pair tiles were measured with the former FP32 atomic epilogue.
    ((12, 0), float("inf")): {
        "gate_up": "64x64x64s2w4g2x2",
        "down": "64x128x32s4w4g2x2",
        # 8192-token chunks (RTX 5090): 64x128 is 8% faster than 64x64 at both shapes.
        "bwd_slot": "64x128x32s4w4g2x2",
        "dx_pair": "64x128x32s4w4g2x2n",
    },
    # GB10 (measured, Qwen3.5 EP1 and Qwen3-MoE, 8-256 tokens): longer contiguous weight
    # segments stream better from LPDDR: K=128 gate/up steps, 128-wide N rows for the
    # N-major backward operands, N-fastest rasters (1-5% at 64 tokens).
    ((12, 1), 512): {
        "gate_up": "16x32x128s3w4g1x4",
        "down": "16x32x64s6w4g1x4n",
        "bwd_slot": "16x128x64s4w4g1x4n",
        "dx_pair": "16x128x64s4w4g1x4n",
    },
    # Measured (GB10): N-fastest rasters keep a row block's gathered rows and its
    # expert's weights in the small L2 across all N tiles (1.45-1.55x for down/dx_pair).
    ((12, 1), 4096): {
        "gate_up": "64x64x64s3w4g2x2n",
        "down": "64x128x64s2w4g2x2n",
        "bwd_slot": "64x128x32s4w4g2x2n",
        "dx_pair": "64x128x32s4w4g2x2n",
    },
    # Measured (GB10, 8192-token chunks): 128-row, 8-warp tiles raise Qwen3-MoE's
    # gate/up, down and dx_pair to 43-53% of the MMA peak (~9% less GEMM time);
    # Qwen3.5 is neutral. Saved bwd_slot loses 7% at BM128 but stays in the bucket.
    ((12, 1), float("inf")): {
        "gate_up": "128x64x64s3w8g4x2n",
        "down": "128x256x32s3w8g2x4n",
        "bwd_slot": "128x128x32s3w8g2x4n",
        "dx_pair": "128x256x32s3w8g2x4n",
    },
}

# A chunk size inside the bucket each BM identifies (BM128 only exists above 4096).
BUCKET_TOKENS = {16: 128, 32: 512, 64: 2048, 128: 8192}


@lru_cache(maxsize=1)
def available():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except importlib.metadata.PackageNotFoundError:
        return False


def configs(capability, tokens, ni, nis, h, override=None):
    override = overrides() if override is None else override
    # Overrides fix BM and therefore select the matching default bucket too.
    # This makes _config(lay) independent of mutable backend state.
    bm = next(iter(override.values())).bm if override else None
    if bm is not None:
        tokens = BUCKET_TOKENS[bm]
    bounds = sorted(bound for cap, bound in DISPATCH if cap == capability)
    defaults = DISPATCH[capability, next(bound for bound in bounds if tokens <= bound)]
    table = {}
    for kind in KINDS:
        c = parse_config(defaults[kind])
        if bm is not None and c.bm != bm:
            c = replace(c, bm=bm)
        # Small shards still use the bucket's BM and legal warp grid.
        dims = (ni, nis) if kind in ("gate_up", "bwd_slot") else (h,)
        if any(d % c.bn for d in dims):
            c = replace(c, bn=64)
        kdims = (h,) if kind in ("gate_up", "bwd_slot") else (ni, nis)
        if any(d % c.bk for d in kdims):
            c = replace(c, bk=64)
        if kind in override:
            c = override[kind]
        # The bwd_slot entry serves the saved backward only (recompute uses gate_up's).
        sv = kind == "bwd_slot"
        if kind not in override and c.shared_bytes(kind, saved=sv) > 99 * 1024:
            # A BM128 override also enlarges unswept kinds. Shorten only their
            # default rings, leaving explicit stage counts untouched.
            while c.stages > 2 and c.shared_bytes(kind, saved=sv) > 99 * 1024:
                c = replace(c, stages=c.stages - 1)
        c.validate(kind, h, ni, nis, saved=sv)
        table[kind] = c
    if len({c.bm for c in table.values()}) != 1:
        raise ValueError("All MoE CuTe kinds must share BM")
    # Recompute needs enough space for gate_up even if saved backward differs.
    table["gate_up"].validate("bwd_slot", h, ni, nis)
    return table


def supports(h, ni, nis):
    if not available() or h <= 0 or ni <= 0 or nis < 0 or any(d % 64 for d in (h, ni, nis)):
        return False
    try:
        for cap in ((12, 0), (12, 1)):
            for tokens in (1, 512, 2048, 8192):
                configs(cap, tokens, ni, nis, h)
    except ValueError:
        return False
    return True


def compile_kernel(
    kind,
    config,
    h,
    ni,
    nis,
    e,
    kp1,
    saved,
    has_shared,
    capability,
    pdl=None,
):
    """Compile using symbolic row counts; usable offline without a CUDA device."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass.runtime import make_fake_stream

    from ._gemm import GroupedGemm

    if pdl is None:
        pdl = capability in ((12, 0), (12, 1)) and os.environ.get("CTI_MOE_CUTE_PDL", "1") != "0"
    config.validate(kind, h, ni, nis, saved and kind == "bwd_slot")
    fake = cute.runtime.make_fake_tensor
    rows, tokens, blocks, slots = (cute.sym_int() for _ in range(4))
    nmax = max(ni, nis)

    def tensor(dtype, shape):
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
        return fake(dtype, shape, tuple(stride), assumed_align=16)

    a_shape = (
        (tokens, h)
        if kind in ("gate_up", "bwd_slot")
        else (rows, nmax * (2 if kind == "dx_pair" else 1))
    )
    w1 = kind in ("gate_up", "dx_pair")
    b_shape = (e, 2 * ni, h) if w1 else (e, h, ni)
    sb_shape = (2 * nis, h) if w1 else (h, nis)
    out_shape = (rows, nmax if kind == "gate_up" else 2 * nmax if kind == "bwd_slot" else h)
    fa = tensor(BFloat16, a_shape)
    fb = tensor(BFloat16, b_shape)
    fs = tensor(BFloat16, sb_shape) if has_shared else fa
    fake_out = tensor(BFloat16, out_shape)
    fw = tensor(Float32, (slots,)) if kind == "bwd_slot" else fake_out
    fg = (
        tensor(BFloat16, (rows, nmax))
        if kind == "gate_up" or (kind == "bwd_slot" and saved)
        else fake_out
        if kind == "bwd_slot"
        else fa
    )
    fd = (
        tensor(Float32, (rows, nmax // config.bn))
        if kind == "bwd_slot"
        else fake_out
        if kind == "gate_up"
        else fw
    )
    fx = tensor(BFloat16, (tokens, h)) if kind == "bwd_slot" else fa
    f13 = tensor(BFloat16, (e, 2 * ni, h)) if kind == "bwd_slot" else fb
    fs13 = (
        tensor(BFloat16, (2 * nis, h))
        if kind == "bwd_slot" and has_shared
        else fa
        if kind == "bwd_slot"
        else fs
    )
    args = (
        fa,
        fb,
        fs,
        tensor(Int32, (rows,)),
        tensor(Int32, (blocks,)),
        fw,
        fake_out,
        fg,
        fg,
        fd,
        fx,
        f13,
        fs13,
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    # Separate fake objects keep each tensor argument independent, even when
    # callers alias placeholders. Symbolic dimensions are still shared.
    args = tuple(tensor(v.element_type, v.shape) for v in args[:-1]) + args[-1:]
    return cute.compile(
        GroupedGemm(
            kind,
            config,
            h,
            ni,
            nis,
            e,
            kp1,
            saved,
            has_shared,
            pdl,
        ),
        *args,
        options=f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a",
    )


class CuteBackend:
    supports = staticmethod(supports)

    def __init__(self, device, ni, nis, h):
        import torch

        self.device_layout = os.environ.get("CTI_MOE_CUTE_LAYOUT", "1") != "0"
        self.device_token = os.environ.get("CTI_MOE_CUTE_TOKEN", "1") != "0"
        self.device = torch.device(device)
        self.capability = torch.cuda.get_device_capability(self.device)
        if self.capability not in ((12, 0), (12, 1)) or not supports(h, ni, nis):
            raise ValueError("MoE CuTe requires sm_120/121, DSL 4.8.0, and supported dimensions")
        self.h, self.ni, self.nis = h, ni, nis
        self.override = overrides()
        self.pdl = os.environ.get("CTI_MOE_CUTE_PDL", "1") != "0"

    def block_m(self, device, tokens, ni, nis):
        return configs(self.capability, tokens, ni, nis, self.h, self.override)["gate_up"].bm

    def bwd_token(self, dxp, inv, dwpart, wslot, sel, r, sg, dx, valid):
        from ._token import token_backward

        token_backward(
            dxp,
            inv,
            dwpart,
            wslot,
            sel,
            r,
            sg,
            dx,
            valid,
            capability=self.capability,
        )

    def dw_tiles(self, device, tokens, ni, nis, *, saved=False):
        table = configs(self.capability, tokens, ni, nis, self.h, self.override)
        return max(ni, nis) // table["bwd_slot" if saved else "gate_up"].bn

    def _config(self, kind, lay, saved=False):
        tokens = BUCKET_TOKENS[lay.block_m]
        table = configs(self.capability, tokens, self.ni, self.nis, self.h, self.override)
        c = table["gate_up" if kind == "bwd_slot" and not saved else kind]
        if lay.block_m != c.bm:
            raise ValueError("Slot layout BM does not match the CuTe backend")
        return c

    def _run(
        self,
        kind,
        lay,
        a,
        b,
        sb,
        ws,
        out,
        gs,
        us,
        dw,
        x,
        w13,
        s13,
        kp1=1,
        saved=False,
    ):
        import torch

        cfg = self._config(kind, lay, saved)
        key = (
            kind,
            cfg,
            self.h,
            self.ni,
            self.nis,
            b.shape[0],
            kp1,
            saved,
            sb is not None,
            self.capability,
            self.pdl,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up this MoE CuTe shape/save mode before CUDA graph capture"
                )
            with _LOCK:
                fn = _COMPILED.get(key)
                if fn is None:
                    fn = compile_kernel(*key)
                    _COMPILED[key] = fn
        # Reuse caller buffers for unused arguments, matching the fake signature.
        # There are no runtime allocations or stream cache keys.
        fn(
            a,
            b,
            sb if sb is not None else a,
            lay.sorted_slot,
            lay.block_expert,
            ws,
            out,
            gs,
            us,
            dw,
            x,
            w13,
            s13 if s13 is not None else a,
        )

    def gate_up(self, x, w13, s13, lay, a, gsave, usave, *, ni, nis, kp1, save):
        self._run(
            "gate_up",
            lay,
            x,
            w13,
            s13,
            a,
            a,
            gsave if save else a,
            usave if save else a,
            a,
            x,
            w13,
            s13,
            kp1,
            save,
        )

    def down(self, a, w2, s2, lay, y, *, ni, nis):
        """Write unweighted BF16 slot rows; the combine applies route weights."""
        self._run("down", lay, a, w2, s2, y, y, a, a, y, a, w2, s2)

    def bwd_slot(
        self, x, dy, w13, s13, w2, s2, lay, wslot, gsave, usave, gg, dwpart, *, ni, nis, kp1, saved
    ):
        self._run(
            "bwd_slot",
            lay,
            dy,
            w2,
            s2,
            wslot,
            gg,
            gsave if saved else gg,
            usave if saved else gg,
            dwpart,
            x,
            w13,
            s13,
            kp1,
            saved,
        )

    def dx_pair(self, gg, w13, s13, lay, dxp, *, ni, nis):
        self._run("dx_pair", lay, gg, w13, s13, dxp, dxp, gg, gg, dxp, gg, w13, s13)


def backend(device, ni, nis, h):
    return CuteBackend(device, ni, nis, h)
