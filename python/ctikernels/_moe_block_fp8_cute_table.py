# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Pure-data MoE policy. Entries are heuristic until GPU sweeps replace them.

``path`` selects independent grouped and decode implementations.
Grouped BM is the sorted-row tile; decode BM=8 denotes resident token slots.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    bm: int = 16
    bni: int = 64
    bn: int = 128
    stages: int = 4
    warps: int = 4
    swap_ab: bool = True
    a_source: str = "fp8"
    ctas_per_sm: int = 1
    path: str = "grouped"
    gather_warps: int = 4
    use_pdl: bool = False

    def __post_init__(self):
        if self.path == "decode":
            if self.bm != 8 or self.bni not in (16, 32) or self.bn not in (16, 32, 64):
                raise ValueError("decode needs BM=8, BNi=16/32 and BN=16/32/64")
            if (
                self.stages not in (2, 3, 4)
                or self.warps != 4
                or not self.swap_ab
                or self.a_source != "bf16"
                or self.ctas_per_sm not in (2, 4)
            ):
                raise ValueError("invalid decode pipeline configuration")
            return
        if self.bm not in (16, 32, 64, 128):
            raise ValueError("BM must be 16, 32, 64 or 128")
        if self.bni not in (32, 64, 128) or self.bn not in (64, 128):
            raise ValueError("BNi must be 32/64/128 and BN must be 64/128")
        if self.stages not in (2, 3, 4, 6, 8) or self.warps not in (2, 4, 8):
            raise ValueError("need 2/3/4/6/8 stages and 2/4/8 consumer warps")
        if self.swap_ab and self.bm not in (16, 32):
            raise ValueError("swap-AB supports BM=16/32")
        if self.a_source != "fp8" or self.path != "grouped":
            raise ValueError("unknown activation source or kernel path")
        if self.ctas_per_sm not in (1, 2):
            raise ValueError("ctas_per_sm must be 1 or 2")
        if self.gather_warps not in (1, 2, 4):
            raise ValueError("gather_warps must be 1, 2 or 4")
        for fc1, width in ((True, self.bni), (False, self.bn)):
            bm = self.bm if fc1 else min(self.bm, 64)
            m, n = (width, bm) if self.swap_ab else (bm, width)
            wm = min(self.warps, 4, m // 16)
            if n // (self.warps // wm) < 16:
                raise ValueError("tile too narrow for ldmatrix.x4")
        if max(self.smem_bytes(True), self.smem_bytes(False)) > 99 * 1024:
            raise ValueError("configuration exceeds 99 KiB shared memory")

    def smem_bytes(self, fc1):
        if self.path == "decode":
            # Target H=2048; actual resident activation storage is shape-dependent.
            width = self.bni if fc1 else self.bn
            prefix = 16 * self.stages + 4 * (128 if fc1 else 8 * self.stages)
            prefix += 4 * (2 if fc1 else 1) * self.stages
            return (
                ((prefix + 127) // 128) * 128
                + (16384 if fc1 else 1024 * self.stages)
                + self.stages * width * 128 * (2 if fc1 else 1)
            )
        width, halves = (self.bni, 2) if fc1 else (self.bn, 1)
        bm = self.bm if fc1 else min(self.bm, 64)
        # FC1 small-M tiles keep H=2048 resident; FC2 keeps I=512 resident.
        planes = (16 if fc1 else 4) if (self.swap_ab or not fc1) else self.stages
        prefix = 16 * (self.stages + 1) + 4 * (bm * planes + halves * self.stages)
        return (
            ((prefix + 127) // 128) * 128 + bm * 128 * planes + self.stages * halves * width * 128
        )


PRESETS = {
    "swap16": Config(),
    "swap16-n32": Config(bni=32, bn=64, stages=6, warps=2),
    "swap16-n32-s8": Config(bni=32, bn=64, stages=8, warps=2),
    "swap16-n32-s2": Config(bni=32, bn=64, stages=2, warps=2, ctas_per_sm=2),
    "swap32": Config(bm=32, bni=32),
    "swap32-n32-s3": Config(bm=32, bni=32, stages=3),
    "swap32-n64-s2": Config(bm=32, stages=2),
    "group64": Config(bm=64, bni=64, bn=128, stages=3, swap_ab=False),
    "group64-n128": Config(
        bm=64, bni=128, bn=128, stages=2, warps=8, swap_ab=False, gather_warps=2
    ),
    "group64-n64-s2": Config(bm=64, bni=64, bn=64, stages=2, swap_ab=False, ctas_per_sm=2),
    "group128": Config(bm=128, bni=128, bn=128, stages=2, warps=8, swap_ab=False, gather_warps=2),
    "group128-n64": Config(bm=128, bni=64, bn=64, stages=3, warps=8, swap_ab=False),
}
PRESETS.update(
    {
        "decode-n16": Config(
            path="decode", a_source="bf16", bm=8, bni=16, bn=16, stages=2, ctas_per_sm=4
        ),
        "decode-n32": Config(
            path="decode", a_source="bf16", bm=8, bni=32, bn=32, stages=3, ctas_per_sm=2
        ),
        "decode-n64": Config(
            path="decode", a_source="bf16", bm=8, bni=32, bn=64, stages=3, ctas_per_sm=2
        ),
        "decode-n32-pdl": Config(
            path="decode",
            a_source="bf16",
            bm=8,
            bni=32,
            bn=32,
            stages=3,
            ctas_per_sm=2,
            use_pdl=True,
        ),
    }
)

# (CC major, CC minor, SM count) -> (inclusive M limit, provider, preset).
# Only measured wins belong here. CuTe remains explicit opt-in until qualified.
TABLE = {}


def selection(device_class, num_tokens):
    if not isinstance(num_tokens, int) or num_tokens < 0:
        raise ValueError("num_tokens must be a nonnegative integer")
    rows = TABLE.get(device_class, ())
    for limit, provider, preset in rows:
        if num_tokens <= limit:
            return provider, preset
    return "triton-fp8", None


def heuristic(num_tokens):
    return "swap16" if num_tokens <= 4 else "swap32" if num_tokens <= 128 else "group64"


def valid_presets(num_tokens):
    if not isinstance(num_tokens, int) or not 0 <= num_tokens <= 4096:
        raise ValueError("cute-fp8 supports 0..4096 tokens")
    return tuple(
        name for name, config in PRESETS.items() if config.path != "decode" or num_tokens <= 8
    )
