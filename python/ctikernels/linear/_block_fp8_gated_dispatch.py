# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Small-M CuTe fusion policy and measured FlashInfer FC1 overrides."""

from . import _block_fp8_dispatch as gemm_dispatch
from ._block_fp8_gated_table import SMALL_M
from ._block_fp8_gated_table import TABLE as FC1_TABLE
from .cute._block_fp8_fused_provider import FusedConfig

CUTE = "cute-act-quant"
FLASHINFER = "flashinfer-cute-dsl"
FUSED = "cute-fp8-fused"
DEVICES = {(12, 0, 170): "rtx5090", (12, 0, 188): "rtxpro6000", (12, 1, 48): "gb10"}
TABLE = {
    hardware: {(n // 2, k): runs for (device, n, k), runs in FC1_TABLE.items() if device == name}
    for hardware, name in DEVICES.items()
}

Buckets = tuple[tuple[int, str], ...]


def fused_supported(capability, i, k):
    return tuple(capability) in ((12, 0), (12, 1)) and i > 0 and k > 0 and not (i % 128 or k % 128)


def parse_config(spec: str) -> FusedConfig:
    """Parse a complete compact engine spec without importing a CUDA/DSL module."""
    if not isinstance(spec, str):
        raise ValueError("fused config spec must be a string")
    engine, sep, fields = spec.partition(":")
    names = {"fma": {"r": "r", "w": "warps"}, "mma": {"wn": "warps_n", "wk": "warps_k"}}
    if not sep or engine not in names:
        raise ValueError("expected fma:r=4,w=4 or mma:wn=1,wk=4")
    values = {}
    for field in fields.split(","):
        key, eq, value = field.partition("=")
        if not eq or key not in names[engine] or key in values or not value.isdecimal():
            raise ValueError("invalid or repeated fused config field")
        values[key] = int(value)
    if values.keys() != names[engine].keys():
        raise ValueError("both fused config fields are required")
    return FusedConfig(
        engine=engine, **{names[engine][key]: value for key, value in values.items()}
    )


def config_spec(config: FusedConfig) -> str:
    if config.engine == "fma":
        return f"fma:r={config.r},w={config.warps}"
    return f"mma:wn={config.warps_n},wk={config.warps_k}"


def parse_runs(runs):
    """Validate and parse a small-M row once, at operation construction."""
    if not runs:
        raise ValueError("small-M runs must not be empty")
    parsed, previous = [], 0
    for ceiling, name, spec in runs:
        if type(ceiling) is not int or ceiling <= previous:
            raise ValueError("small-M ceilings must be positive and strictly increasing")
        if name not in (CUTE, FLASHINFER, FUSED):
            raise ValueError("unknown small-M provider")
        if name != FUSED and spec is not None:
            raise ValueError("only fused providers accept a config spec")
        if name == FUSED and ceiling > 16:
            raise ValueError("fused small-M ceiling must be <=16")
        parsed.append((ceiling, name, parse_config(spec) if spec is not None else None))
        previous = ceiling
    return tuple(parsed)


def small_m_buckets(capability, sms, i, k):
    runs = SMALL_M.get((DEVICES.get((*capability, sms)), i, k))
    if runs is not None:
        return parse_runs(runs)
    return ((1, FUSED, None),) if fused_supported(capability, i, k) else ()


def buckets(capability: tuple[int, int], sms: int, i: int, k: int) -> Buckets | None:
    """The measured runs of a merged ``[2I, K]`` weight on this device, else None."""
    return FC1_TABLE.get((DEVICES.get((*capability, sms)), 2 * i, k))


def provider(table: Buckets, rows: int) -> str:
    """The provider of the run holding ``rows``."""
    for last, name in table:
        if rows <= last:
            return name
    return table[-1][1]


def resolved_provider(table, rows, gemm_provider, measured=None):
    measured_cute = measured is not None and gemm_dispatch.provider(measured, rows) == "cute-fp8"
    if provider(table, rows) == FLASHINFER and not (measured_cute and gemm_provider == "cute-fp8"):
        return FLASHINFER
    return f"{gemm_provider}+{CUTE}"
