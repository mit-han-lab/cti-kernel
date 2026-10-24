# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Launch configs of the block-FP8 Triton GEMM: today's rule and its tuned tables (no Triton
import; the provider and the kernels module both read it)."""

from __future__ import annotations

from ..._runtime.tables import DeviceClass, Dim, TunedOp, pow2_edges

GROUP = 128
# Shared memory per block on sm120/121 (and sm89), the binding limit of the GEMM's operand ring.
_SMEM = 99 * 1024


def _groups(size: int) -> int:
    return -(-size // GROUP)


def _config(
    m: int, n: int, k: int, sms: int, capability: tuple[int, int] | None = None
) -> tuple[int, ...]:
    """(BLOCK_M, BLOCK_N, SPLIT_K, warps, stages[, GROUP_M]) of the Triton GEMM for ``m``
    rows, from RTX 5090 sweeps over the Qwen3.5-35B-A3B projection shapes: tiles shrink, then
    K splits, until the grid covers the SMs (small batches read each weight byte once either
    way); large batches tile M by 128. Compute capability 12.1 (GB10, whose Triton dots run
    on F16 tensor cores after an E4M3 conversion) has its own rule up to 512 rows, from GB10
    sweeps: narrow 32-column tiles in four stages and, from 17 rows of a wide weight, one
    64-row tile (each weight byte read once); from 65 rows wide weights take 64 x 128 tiles
    walked row tile first (``GROUP_M``), so that a weight tile's row tiles run together."""

    def tiles(block_m: int, block_n: int) -> int:
        return -(-m // block_m) * -(-n // block_n)

    def split(block_m: int, block_n: int) -> int:
        split_k = 1
        while (
            split_k < 8
            and 2 * split_k * 4 <= _groups(k)
            and split_k * tiles(block_m, block_n) < sms
        ):
            split_k *= 2
        return split_k

    if capability == (12, 1) and m <= 512:
        if m <= 16:
            block_m, block_n, stages = 16, 32, 4
        elif m <= 64:
            block_m, block_n, stages = 64 if n >= 4096 else 32, 32, 4
        elif n >= 4096:
            return 64, 128, split(64, 128), 4, 3, 32
        else:
            block_m = 64 if m > 128 or (k >= 4096 and m > 96) else 32
            block_n = 64 if k >= 4096 else 32
            stages = 4 if m <= 128 or k >= 4096 else 3
        return block_m, block_n, split(block_m, block_n), 4, stages
    if m > 512:
        block_m = 128 if tiles(128, 64) >= sms else 64
        return block_m, 64, 1, 4, 2 if k >= 2048 and block_m == 128 else 3
    if m <= 64:
        block_m = 16 if m <= 16 else 32
        block_n = 128 if n >= 4096 else 64 if n >= 2048 else 32
    else:
        block_m = 64 if m > 128 or tiles(64, 64) >= sms else 32
        block_n = 32 if n <= 1024 and tiles(block_m, 64) < sms else 64
    return block_m, block_n, split(block_m, block_n), 8 if block_n == 128 else 4, 3


def _smem(config, mode) -> int:
    """Operand ring bytes: stages x (BLOCK_M + BLOCK_N) rows of 128 K-columns."""
    width = 2 if mode == "bf16" else 1
    return config["stages"] * (config["block_m"] + config["block_n"]) * GROUP * width


def _default(shape, device: DeviceClass) -> dict:
    """Today's rule (``_config``), with stages, then BLOCK_N, cut until a BF16-mode ring fits."""
    block_m, block_n, split_k, warps, stages, *group_m = _config(
        shape["m"], shape["n"], shape["k"], device.sms, (device.major, device.minor)
    )
    config = {
        "block_m": block_m,
        "block_n": block_n,
        "split_k": split_k,
        "warps": warps,
        "stages": stages,
        "group_m": group_m[0] if group_m else 1,
    }
    while _smem(config, shape["mode"]) > _SMEM and config["stages"] > 2:
        config["stages"] -= 1
    while _smem(config, shape["mode"]) > _SMEM and config["block_n"] > 32:
        config["block_n"] //= 2
    return config


# The Triton GEMM (FP8 x block-FP8 with in-kernel or given 1x128 activation scales, or the
# BF16/FP16 mode of a 16-bit weight): per-device-class tuned tables over (M bucket, N, K, mode),
# today's rule elsewhere. Split K only up to one 128-column group per split.
TUNED = TunedOp(
    "linear.block_fp8.gemm.triton",
    version=1,
    space={
        "block_m": (16, 32, 64, 128),
        "block_n": (32, 64, 128, 256),
        "split_k": (1, 2, 4, 8),
        "warps": (4, 8),
        "stages": (2, 3, 4, 5),
        "group_m": (1, 8, 16, 32),
    },
    dims=(Dim("m", pow2_edges(1, 65536)), Dim("n"), Dim("k"), Dim("mode")),
    default=_default,
    valid=lambda config, shape: (
        _smem(config, shape["mode"]) <= _SMEM and config["split_k"] <= _groups(shape["k"])
    ),
)


def tuned_config(device: DeviceClass, m: int, n: int, k: int, mode: str) -> tuple[int, ...]:
    """(BLOCK_M, BLOCK_N, SPLIT_K, warps, stages[, GROUP_M]) of an ``m`` x ``n`` x ``k`` GEMM in
    ``mode`` ("fp8" or "bf16") on ``device``'s class. Every caller path reads this function;
    the tuner pins configs by replacing it."""
    c = TUNED.lookup(device, m=m, n=n, k=k, mode=mode)
    config = (c["block_m"], c["block_n"], c["split_k"], c["warps"], c["stages"])
    return config if c["group_m"] == 1 else (*config, c["group_m"])
