# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Linear dispatch by hardware tuple, M bucket (M=1..64), N and K; missing rows use cuBLAS.

TUNED rows imply the cute provider and carry its configuration. Device classes
map exact hardware tuples to tables; inclusive M buckets preserve holes.
Every M in a bucket uses its upper-edge configuration for bitwise identical CuTe rows.
Every row has paired evidence in benchmark/kernel_tuning/results/linear_smallm.md
(M=1..8) or benchmark/kernel_tuning/results/linear_smallm_bxq.md (B x Q).

The heuristic and bounded candidate grids serve offline probes and explicit warm-up autotune.
"""

import math
import random
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache

QWEN_9B = (
    (12288, 4096),
    (64, 4096),
    (12352, 4096),
    (4096, 4096),
    (10240, 4096),
    (24576, 4096),
    (4096, 12288),
    (248320, 4096),
)
QWEN_SHAPES = QWEN_9B + (
    (16384, 5120),
    (96, 5120),
    (16480, 5120),
    (5120, 6144),
    (14336, 5120),
    (34816, 5120),
    (5120, 17408),
    (248320, 5120),
    (12288, 2048),
    (64, 2048),
    (12352, 2048),
    (2048, 4096),
    (9216, 2048),
    (1024, 2048),
    (2048, 512),
    (256, 2048),
    (1, 2048),
    (248320, 2048),
    # Separate logical Q/gate and KV projections.
    (8192, 4096),
    (1024, 4096),
    (12288, 5120),
    (1024, 5120),
    (8192, 2048),
    (512, 2048),
)


_DEVICE_CLASSES = {((12, 0), 170): "rtx5090", ((12, 0), 188): "pro6000", ((12, 1), 48): "gb10"}

GATED_SHAPES = ((4096, 4096, 10240), (5120, 6144, 14336), (5120, 3072, 7168))  # (N, K, gate_stride)


def device_class(capability: tuple[int, int], sms: int) -> str | None:
    """Measured table key from the hardware tuple (major, minor, SMs) only."""
    return _DEVICE_CLASSES.get((tuple(capability), sms))


@lru_cache(maxsize=None)
def _hardware(device):
    """(device class, capability, SMs) of a CUDA device; torch is imported on first query."""
    import torch

    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    return (
        device_class(capability, props.multi_processor_count),
        capability,
        props.multi_processor_count,
    )


# Config.cluster: two-launch reduction, or one launch through DSMEM / the L2 scratch.
NONE, DSMEM, L2 = 0, 1, 2


@dataclass(frozen=True)
class Config:
    """Small tuning space: rows/CTA, K tile, pipeline stages, deterministic K splits.

    ``cluster`` reduces the splits inside one launch: the split CTAs of an N tile form a
    thread-block cluster and the slice owners sum the partials in split order (the FP32
    order of the reduce kernel). DSMEM (1) exchanges partials through distributed shared
    memory and needs no scratch; L2 (2) writes them to the split-K scratch. NONE (0) is
    the two-launch reduction.
    """

    rows: int = 64
    tile_k: int = 128
    stages: int = 2
    splits: int = 1
    warps: int = 0
    variant: str = "smem"
    prefetch: int = 4
    cluster: int = 0

    def __post_init__(self):
        if self.warps == 0:
            object.__setattr__(self, "warps", min(4, self.rows // 16))
        if self.variant not in ("smem", "reg", "wide") or not 2 <= self.prefetch <= 8:
            raise ValueError("variant must be smem/reg/wide and prefetch in [2, 8]")
        if self.variant in ("reg", "wide"):
            if not 1 <= self.warps <= 8 or self.rows not in (16 * self.warps, 32 * self.warps):
                raise ValueError("reg requires 1..8 warps and 16 or 32 rows per warp")
        elif self.warps not in (1, 2, 4, 8) or self.rows % (16 * self.warps):
            raise ValueError("warps must divide rows/16 and be 1/2/4/8")
        if (self.variant == "smem" and self.rows not in (16, 32, 64, 128)) or self.tile_k not in (
            64,
            128,
            256,
            512,
        ):
            raise ValueError("rows must be 16/32/64/128 and tile_k 64/128/256/512")
        if self.stages not in (2, 3, 4, 5, 6) or not 1 <= self.splits <= 64:
            raise ValueError("stages must be in [2, 6] and splits in [1, 64]")
        if type(self.cluster) is bool:  # Records written as false/true.
            object.__setattr__(self, "cluster", int(self.cluster))
        if self.cluster not in (NONE, DSMEM, L2) or (
            self.cluster and (not 2 <= self.splits <= 16 or self.rows % self.splits)
        ):
            # Above 8 CTAs the cluster size is non-portable (sm_121 rejects it).
            raise ValueError("cluster is 0/1/2 and needs 2..16 splits dividing rows")
        if self.shared_bytes > 96 * 1024:
            raise ValueError("configuration exceeds the 96 KiB shared memory budget")

    @property
    def shared_bytes(self):
        return self.shared_bytes_for(1)

    @property
    def scratch_splits(self) -> int:
        """Splits written to global FP32 scratch (0: none, in-kernel or no reduction)."""
        return self.splits if self.splits > 1 and self.cluster != DSMEM else 0

    @property
    def kernels(self) -> int:
        """Launches per call: the two-launch reduction adds a reduce kernel."""
        return 2 if self.splits > 1 and self.cluster == NONE else 1

    def shared_bytes_for(self, mtiles: int, gated: bool = False) -> int:
        """Shared bytes for x and optional gate, each occupying 8*mtiles rows, plus the
        two FP32 [8*mtiles, rows] tiles (receive, staging) of a DSMEM reduction."""
        xrows = 8 * mtiles * (2 if gated else 1)
        dsmem = self.cluster == DSMEM
        partial = (2 * 8 * mtiles * self.rows + 4 * self.splits) * 4 + 16 if dsmem else 0
        if self.variant in ("reg", "wide"):
            return xrows * (self.tile_k + 8) * 2 + partial
        return self.stages * (self.rows + xrows) * (self.tile_k + 8) * 2 + partial


# BEGIN GENERATED SMALLM TABLE
TUNED = {
    "rtx5090": {
        (1, 2048): {
            (1, 1): Config(16, 128, 2, 8, 1, "smem", 4, 1),
        },
        (1024, 2048): {
            (1, 1): Config(32, 128, 2, 8, 2, "smem", 4, 1),
        },
        (124160, 5120): {
            (1, 1): Config(32, 128, 2, 8, 2, "smem", 4, 1),
        },
        (151936, 4096): {
            (1, 1): Config(64, 128, 3, 4, 4, "smem", 4, 1),
        },
        (2048, 512): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (2048, 5120): {
            (1, 1): Config(64, 128, 3, 4, 4, "smem", 4, 1),
        },
        (248320, 2048): {
            (1, 1): Config(64, 128, 3, 4, 4, "smem", 4, 1),
        },
        (248320, 5120): {
            (1, 1): Config(32, 128, 2, 8, 2, "smem", 4, 1),
        },
        (5120, 3072): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (5120, 4096): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (64, 2048): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
        },
        (9216, 2048): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (96, 5120): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
        },
        (128, 2048): {
            (1, 8): Config(16, 128, 2, 16, 1, "smem", 4, 1),
            (9, 32): Config(32, 128, 2, 16, 2, "smem", 4, 2),
            (33, 56): Config(16, 128, 2, 16, 1, "reg", 4),
        },
        (151936, 2048): {
            (1, 1): Config(128, 128, 2, 1, 4, "smem", 4),
        },
        (129280, 4096): {
            (1, 1): Config(128, 128, 2, 1, 8, "smem", 4),
        },
        (64, 4096): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
            (2, 16): Config(16, 128, 2, 32, 1, "reg", 4),
        },
        (2048, 4096): {
            (1, 1): Config(32, 128, 3, 4, 2, "smem", 4, 1),
            (2, 16): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (4096, 4096): {
            (1, 16): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (4096, 12288): {
            (3, 8): Config(16, 128, 2, 3, 1, "reg", 4),
            (9, 16): Config(128, 128, 2, 6, 8, "reg", 4),
            (17, 24): Config(128, 128, 2, 3, 8, "reg", 4),
            (41, 48): Config(128, 128, 2, 3, 8, "reg", 4),
        },
        (4096, 20480): {
            (9, 16): Config(64, 128, 3, 2, 4, "smem", 4),
        },
        (6144, 4096): {
            (2, 16): Config(64, 128, 3, 1, 4, "smem", 4),
            (17, 32): Config(64, 128, 2, 1, 4, "smem", 4),
        },
        (10240, 4096): {
            (17, 24): Config(64, 64, 2, 1, 4, "smem", 4),
        },
        (12352, 4096): {
            (17, 32): Config(64, 64, 2, 1, 4, "smem", 4),
        },
        (248320, 4096): {
            (1, 1): Config(128, 64, 3, 1, 4, "smem", 4),
        },
    },
    "pro6000": {
        (1, 2048): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
        },
        (124160, 5120): {
            (1, 1): Config(32, 128, 3, 4, 2, "smem", 4, 1),
        },
        (151936, 4096): {
            (1, 1): Config(32, 128, 3, 4, 2, "smem", 4, 1),
        },
        (2048, 4096): {
            (1, 1): Config(64, 128, 3, 4, 4, "smem", 4, 1),
        },
        (2048, 512): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (5120, 6144): {
            (1, 1): Config(64, 128, 3, 2, 4, "smem", 4, 1),
        },
        (64, 2048): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
        },
        (128, 2048): {
            (1, 8): Config(16, 128, 2, 16, 1, "smem", 4, 1),
            (9, 32): Config(32, 128, 2, 16, 2, "smem", 4, 2),
            (33, 56): Config(16, 128, 2, 16, 1, "reg", 4),
        },
        (129280, 4096): {
            (1, 1): Config(64, 128, 2, 1, 4, "smem", 4),
        },
        (64, 4096): {
            (1, 1): Config(16, 256, 2, 8, 1, "smem", 4, 1),
            (2, 16): Config(16, 128, 2, 32, 1, "reg", 4),
        },
        (96, 5120): {
            (1, 1): Config(32, 128, 2, 32, 2, "smem", 4),
            (2, 2): Config(16, 128, 2, 8, 1, "smem", 4, 1),
            (3, 4): Config(32, 128, 2, 32, 2, "smem", 4),
            (5, 8): Config(16, 256, 2, 32, 1, "smem", 4),
        },
        (2048, 5120): {
            (1, 1): Config(64, 128, 2, 4, 4, "smem", 4, 1),
            (2, 8): Config(64, 128, 2, 4, 4, "smem", 4),
            (9, 16): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (4096, 4096): {
            (2, 4): Config(64, 128, 2, 4, 4, "smem", 4, 1),
            (5, 8): Config(64, 128, 3, 2, 4, "smem", 4, 1),
        },
        (4096, 12288): {
            (17, 40): Config(128, 128, 2, 3, 8, "reg", 4),
            (41, 48): Config(64, 128, 2, 8, 4, "smem", 4),
        },
        (4096, 20480): {
            (33, 40): Config(64, 128, 3, 2, 4, "smem", 4),
            (41, 48): Config(64, 128, 2, 8, 4, "smem", 4),
            (57, 64): Config(64, 128, 2, 8, 4, "smem", 4),
        },
        (5120, 3072): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
            (5, 16): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (5120, 4096): {
            (2, 8): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (5120, 8704): {
            (1, 1): Config(64, 128, 2, 2, 4, "smem", 4, 1),
        },
        (6144, 4096): {
            (1, 2): Config(64, 128, 3, 1, 4, "smem", 4),
            (3, 4): Config(64, 128, 2, 1, 4, "smem", 4),
            (5, 32): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (6144, 5120): {
            (1, 4): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (7168, 5120): {
            (5, 8): Config(64, 128, 3, 1, 4, "smem", 4),
            (17, 32): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (8240, 5120): {
            (5, 24): Config(64, 128, 3, 1, 4, "smem", 4),
            (25, 32): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (10240, 4096): {
            (1, 1): Config(64, 128, 2, 1, 4, "smem", 4),
        },
        (14336, 5120): {
            (17, 24): Config(64, 128, 2, 1, 4, "smem", 4),
            (25, 32): Config(64, 64, 2, 1, 4, "smem", 4),
        },
        (16384, 5120): {
            (1, 2): Config(64, 128, 3, 1, 4, "smem", 4),
        },
        (16480, 5120): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
            (17, 24): Config(64, 128, 2, 1, 4, "smem", 4),
            (25, 32): Config(128, 128, 2, 1, 8, "smem", 4),
        },
        (17408, 5120): {
            (1, 1): Config(32, 128, 3, 4, 2, "smem", 4, 1),
            (17, 24): Config(128, 128, 2, 1, 4, "smem", 4),
            (25, 32): Config(128, 128, 2, 1, 8, "smem", 4),
        },
        (34816, 5120): {
            (1, 1): Config(64, 256, 2, 1, 4, "smem", 4),
        },
        (248320, 4096): {
            (1, 1): Config(64, 128, 3, 1, 4, "smem", 4),
        },
    },
    "gb10": {
        (1, 2048): {
            (1, 1): Config(16, 256, 3, 8, 1, "smem", 4, 1),
        },
        (1024, 2048): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (12288, 2048): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (12352, 2048): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (124160, 5120): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (151936, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (2048, 5120): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (248320, 2048): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (248320, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (248320, 5120): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (5120, 3072): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (5120, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (9216, 2048): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (128, 2048): {
            (2, 2): Config(16, 64, 2, 32, 1, "reg", 2),
            (3, 4): Config(32, 128, 2, 16, 2, "reg", 4),
            (5, 8): Config(16, 64, 2, 32, 1, "reg", 2),
            (9, 16): Config(16, 128, 2, 16, 1, "reg", 4),
        },
        (151936, 2048): {
            (1, 1): Config(64, 512, 2, 4, 4, "wide", 4, 1),
        },
        (129280, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (64, 4096): {
            (2, 8): Config(32, 128, 2, 32, 1, "reg", 4),
        },
        (4096, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (6144, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (10240, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
        (12288, 4096): {
            (1, 1): Config(64, 256, 2, 8, 4, "wide", 2, 1),
        },
    },
}
# END GENERATED SMALLM TABLE

# BEGIN GENERATED SMALLM F32 TABLE
TUNED_F32 = {
    "rtx5090": {
        (1024, 4096): {
            (1, 8): Config(64, 128, 2, 8, 4, "smem", 4),
            (9, 16): Config(64, 128, 3, 4, 4, "smem", 4),
        },
        (2048, 4096): {
            (2, 16): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (512, 4096): {
            (1, 16): Config(64, 128, 2, 32, 4, "reg", 4),
        },
    },
    "pro6000": {
        (1024, 4096): {
            (2, 8): Config(64, 128, 2, 16, 4, "reg", 4),
            (9, 16): Config(128, 128, 2, 16, 8, "reg", 4),
        },
        (512, 4096): {
            (2, 8): Config(16, 128, 2, 32, 1, "reg", 4),
            (9, 16): Config(64, 128, 2, 32, 4, "reg", 4),
        },
    },
    "gb10": {
        (1024, 4096): {
            (1, 1): Config(64, 512, 2, 8, 4, "wide", 4),
            (9, 16): Config(64, 512, 2, 8, 4, "wide", 4),
        },
        (2048, 4096): {
            (1, 1): Config(64, 256, 2, 16, 4, "wide", 2),
        },
        (512, 4096): {
            (2, 8): Config(64, 256, 2, 16, 4, "wide", 4),
        },
    },
}
# END GENERATED SMALLM F32 TABLE

# BEGIN GENERATED GATED TABLE
TUNED_GATED = {
    "rtx5090": {
        (4096, 4096, 10240): {
            (1, 4): Config(32, 128, 4, 1, 2, "smem", 4),
            (5, 8): Config(64, 128, 3, 1, 4, "smem", 4),
            (9, 16): Config(64, 64, 2, 4, 4, "smem", 4),
        },
    },
    "pro6000": {
        (4096, 4096, 10240): {
            (1, 1): Config(32, 64, 3, 2, 2, "smem", 4),
            (2, 4): Config(64, 64, 3, 4, 4, "smem", 4),
            (5, 8): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (5120, 3072, 7168): {
            (1, 1): Config(64, 64, 2, 4, 4, "smem", 4),
            (2, 2): Config(32, 128, 3, 1, 2, "smem", 4),
            (3, 16): Config(64, 64, 2, 4, 4, "smem", 4),
        },
        (5120, 6144, 14336): {
            (1, 4): Config(64, 128, 2, 4, 4, "smem", 4),
        },
    },
    "gb10": {
        (4096, 4096, 10240): {
            (1, 1): Config(16, 512, 3, 4, 1, "smem", 4),
        },
    },
}
# END GENERATED GATED TABLE


# Dispatch granularity. Every M in one bucket runs the same provider and
# configuration (one compiled kernel, fixed split-K order), so rows are bitwise
# identical across M in the bucket, including M padded to the bucket edge.
M_BUCKETS = (
    (1, 1),
    (2, 2),
    (3, 4),
    (5, 8),
    (9, 16),
    (17, 24),
    (25, 32),
    (33, 40),
    (41, 48),
    (49, 56),
    (57, 64),
)
_BUCKET_OF = {m: bucket for bucket in M_BUCKETS for m in range(bucket[0], bucket[1] + 1)}


MIN_MARGIN = 0.03


def qualifies(speedup: float, epsilon: float, margin: float = MIN_MARGIN) -> bool:
    """Shared CuTe-over-cuBLAS gate (offline table and autotune): strictly above 1 + max(margin, epsilon)."""
    return speedup > 1 + max(margin, epsilon)


QUALIFY_PAIRS = 20
# GB10 timings are noisier: >= 60 pairs and a bootstrap 95% CI of the median
# paired ratio whose lower bound is above one.
GB10_MIN_PAIRS = 60


def qualification_pairs(kind: str | None) -> int:
    """Minimum interleaved pairs for a qualifying record on this device class."""
    return GB10_MIN_PAIRS if kind == "gb10" else QUALIFY_PAIRS


def point_qualifies(kind: str | None, point: dict) -> bool:
    """Per-device gate of one paired record (offline table and autotune):
    ``qualifies`` on pairs >= ``qualification_pairs``, plus ci_low > 1 on GB10."""
    ci_low = point.get("ci_low")
    return (
        type(point.get("pairs")) is int
        and point["pairs"] >= qualification_pairs(kind)
        and qualifies(point["speedup"], point["epsilon"])
        and (
            kind != "gb10"
            or (
                isinstance(ci_low, (int, float))
                and not isinstance(ci_low, bool)
                and math.isfinite(ci_low)
                and ci_low > 1
            )
        )
    )


def _percentile(values, fraction):
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_median_ci(ratios, seed) -> tuple[float, float]:
    """95% percentile CI of the median paired ratio from 2000 resamples of whole
    pairs with a local Random(seed); the bounds are interpolated 2.5/97.5th
    percentiles. Identical samples and seed give identical bounds."""
    if not ratios or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("finite positive paired speedups required")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def m_bucket(m: int) -> tuple[int, int] | None:
    """Inclusive (lo, hi) M bucket for M=1..64, else None."""
    return _BUCKET_OF.get(m)


def tuned_config(kind: str | None, n: int, k: int, m: int, *, fp32: bool = False) -> Config | None:
    """Table row for the M bucket of ``m``; rows are looked up at the bucket edge."""
    bucket = _BUCKET_OF.get(m)
    if bucket is None:
        return None
    edge = bucket[1]
    for (lo, hi), config in (TUNED_F32 if fp32 else TUNED).get(kind, {}).get((n, k), {}).items():
        if lo <= edge <= hi:
            return config
    return None


def tuned_gated_config(kind: str | None, n: int, k: int, gate_stride: int, m: int) -> Config | None:
    """Gated table row at the M bucket edge; missing strides/buckets stay unfused."""
    bucket = _BUCKET_OF.get(m)
    if bucket is None:
        return None
    for (lo, hi), config in TUNED_GATED.get(kind, {}).get((n, k, gate_stride), {}).items():
        if lo <= bucket[1] <= hi:
            return config
    return None


def accepts(kind: str | None, n: int, k: int, m: int) -> bool:
    return tuned_config(kind, n, k, m) is not None


def config_for(kind: str | None, sms: int, n: int, k: int) -> Config:
    rows = 16 if n <= 128 else 64
    tile_k = 64 if kind == "gb10" else 128
    tiles = (n + rows - 1) // rows
    target = max(1, (3 * sms + tiles - 1) // tiles)
    splits = min(64, max(1, k // tile_k), target)
    # Powers of two limit the number of compiled variants.
    splits = 1 << (splits.bit_length() - 1)
    return Config(rows, tile_k, 2, splits)


def _gb10_splits(n, k, rows, tile_k, target_bytes, target_ctas):
    """Prefer 16..64 KiB slices, then grid occupancy, then the requested slice.

    Use the kernel's rounded chunk count, not K/splits: non-divisible splits
    otherwise overestimate occupancy. Tiny matrices cannot meet both targets;
    vocabulary heads naturally exceed the target grid. Never add empty splits.
    """
    tiles = (k + tile_k - 1) // tile_k
    ntiles = (n + rows - 1) // rows
    choices = []
    for splits in range(1, min(64, tiles) + 1):
        chunks = (tiles + splits - 1) // splits
        if (splits - 1) * chunks >= tiles:
            continue
        size = 2 * rows * tile_k * chunks
        rank = (
            max(0, 16 * 1024 - size, size - 64 * 1024),
            max(0, target_ctas - ntiles * splits),
            abs(size - target_bytes),
            splits,
        )
        choices.append((rank, splits))
    return min(choices)[1]


def _smem_configs(default: Config, n: int = 0, k: int = 0, kind: str | None = None) -> list[Config]:
    """Ten GB10 candidates; sm120 uses up to ten big-N or eight ordinary.

    All fit 96 KiB. Larger row tiles amortize x loads; deeper rings add prefetch.
    Rows=128/K=256 and rows=128/K=128/stages>=3 exceed the shared budget.
    """
    if kind == "gb10":
        if n <= 0 or k <= 0:
            raise ValueError("GB10 tuning requires positive N and K")
        # One measured control and nine streaming experiments. Among rings
        # of depth >=3, K=512 fits only at rows=16/stages=3 (74,880 B); depth four exceeds
        # our conservative 96 KiB cap, below sm121's 99 KiB opt-in limit.
        options = [Config(64, 256, 2, 1, 4)]
        for rows, tile, stages, kib, ctas in (
            (16, 128, 3, 32, 1536),
            (16, 128, 4, 16, 3072),
            (32, 128, 3, 64, 1024),
            (32, 128, 4, 32, 1536),
            (16, 256, 3, 32, 1536),
            (32, 256, 4, 64, 1024),
            (16, 128, 6, 32, 1536),
            (32, 128, 5, 32, 1536),
            (16, 512, 3, 64, 1024),
        ):
            splits = _gb10_splits(n, k, rows, tile, kib * 1024, ctas)
            options.append(Config(rows, tile, stages, splits))
        return options
    if n >= 16384:
        return list(
            dict.fromkeys(
                [
                    default,
                    Config(128, 128, 2, 1, 4),
                    Config(128, 128, 2, 1, 8),
                    Config(128, 128, 2, 2, 8),
                    Config(128, 64, 3, 1, 4),
                    Config(128, 64, 4, 1, 8),
                    Config(64, 128, 3, 1),
                    Config(64, 128, 4, 1),
                    Config(64, 256, 2, 1),
                    Config(64, 128, 4, 2),
                ]
            )
        )
    rows, splits = default.rows, default.splits
    options = [
        default,
        Config(rows, 64, 2, splits),
        Config(rows, 128, 3, splits),
        Config(rows, 128, 2, max(1, splits // 2)),
        Config(rows, 128, 2, min(64, splits * 2)),
        Config(32 if rows == 16 else 128, 128, 2, splits),
        Config(rows, 256, 2, splits),
        Config(rows, 64, 3, splits),
        Config(32, 64, 2, splits),
        Config(16, 256, 3, splits),
    ]
    return list(dict.fromkeys(options))[:8]


def parse_config(setting: str) -> Config:
    """Accept legacy CSV or rows,tile,stages,splits,warps,variant,prefetch[,cluster]."""
    fields = setting.split(",")
    if not 4 <= len(fields) <= 8:
        raise ValueError("configuration needs 4..8 fields")
    if len(fields) == 8 and fields[7] not in ("0", "1", "2"):
        raise ValueError("cluster must be 0, 1 or 2")
    return Config(*(value if i == 5 else int(value) for i, value in enumerate(fields)))


def tune_configs(default: Config, n: int = 0, k: int = 0, kind: str | None = None) -> list[Config]:
    """At most ten candidates, with six register experiments on every device.

    Target four CTA-warps/SM on sm120, 24 on GB10 (many small CTAs).
    Tiny K/N can have insufficient parallelism; never manufacture empty splits.
    Legacy callers without shape dimensions retain the smem-only grid.
    """
    smem = _smem_configs(default, n, k, kind)
    if n <= 0 or k <= 0:
        return smem
    options = [smem[i] for i in (0, 1, 5, 9)] if kind == "gb10" else smem[:4]
    sms = {"rtx5090": 170, "pro6000": 188, "gb10": 48}.get(kind, 188)
    shapes = (
        ((16, 1, 2), (16, 1, 4), (16, 1, 8), (32, 1, 4), (32, 2, 4), (64, 2, 8))
        if kind == "gb10"
        else ((16, 1, 4), (32, 1, 8), (64, 2, 4), (64, 4, 4), (128, 4, 8), (128, 8, 4))
    )
    for rows, warps, depth in shapes:
        tile = depth * 32
        tiles = (k + tile - 1) // tile
        ntiles = (n + rows - 1) // rows
        target = (24 if kind == "gb10" else 4) * sms
        choices = []
        for splits in range(1, min(64, tiles) + 1):
            chunks = (tiles + splits - 1) // splits
            if (splits - 1) * chunks < tiles:
                choices.append((max(0, target - ntiles * splits * warps), splits))
        splits = min(choices)[1]
        options.append(Config(rows, tile, 2, splits, warps, "reg", depth))
    return list(dict.fromkeys(options))


def bxq_configs(n: int, k: int, kind: str | None = None) -> list[Config]:
    """Offline B x Q candidates, adding 2x/4x splits without empty trailing work.

    The caller filters shared_bytes_for(ceil(M/8)) against the device limit.
    """
    if n <= 0 or k <= 0 or k % 8:
        raise ValueError("positive N/K and K divisible by 8 required")
    base = tune_configs(Config(), n, k, kind)
    extra = []
    for cfg in base:
        tiles = (k + cfg.tile_k - 1) // cfg.tile_k
        for multiplier in (2, 4):
            splits = cfg.splits * multiplier
            chunks = (tiles + splits - 1) // splits
            if splits <= 64 and (splits - 1) * chunks < tiles:
                extra.append(replace(cfg, splits=splits))
    return list(dict.fromkeys(base + extra))
