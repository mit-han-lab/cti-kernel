# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""C3 SiLU x mul dispatch: measured per device class and M band."""

from dataclasses import dataclass

from ctikernels.linear._dispatch import device_class  # noqa: F401


@dataclass(frozen=True)
class Config:
    threads: int = 128
    vectors: int = 1
    pdl: bool = False
    # Elements per vector (16: two 128-bit accesses, 8/4/2: one).
    elems: int = 8
    trigger: str = "end"
    cache: str = "none"

    def __post_init__(self):
        if self.threads not in (64, 128, 256, 512) or self.vectors not in (1, 2, 4):
            raise ValueError("threads must be 64/128/256/512; vectors must be 1/2/4")
        if self.elems not in (2, 4, 8, 16):
            raise ValueError("elems must be 2/4/8/16")
        if not isinstance(self.pdl, bool):
            raise ValueError("pdl must be bool")

        if self.trigger not in ("end", "early"):
            raise ValueError("trigger must be end or early")
        if self.trigger == "early" and not self.pdl:
            raise ValueError("early trigger requires pdl=True")
        if self.cache not in ("none", "st_ef", "ld_el", "both"):
            raise ValueError("cache must be none, st_ef, ld_el or both")

    @property
    def name(self):
        return (
            f"cute_t{self.threads}_v{self.vectors}_e{self.elems}_p{int(self.pdl)}"
            f"_t{'e' if self.trigger == 'early' else 'n'}"
            + (f"_c{self.cache.replace('_', '')}" if self.cache != "none" else "")
        )


@dataclass(frozen=True)
class GeneralConfig:
    lanes: int

    @property
    def name(self):
        return f"cute_general_n{self.lanes}"


# Inclusive M bands from the paired C3 sweep (benchmark/kernel_tuning/results/norm_act.md):
# narrow accesses and more threads for decode, 16-byte accesses from DFlash sizes up. PDL
# (as SGLang launches) overlaps the launch with the previous kernel's tail.
# Other widths/M use the general CuTe kernel; the table selects speed, not admission.
_LARGE = 1 << 40


def _c(threads, vectors, elems, pdl=True, cache="none"):
    return Config(threads, vectors, pdl, elems, "early" if pdl else "end", cache)


# Paired sweeps against the real vLLM 0.31.0 Inductor kernel and SGLang 72730d6 JIT: the
# early PDL trigger wins every decode/DFlash row; prefill rows are at the DRAM floor.
_BANDS = {
    "rtx5090": {
        (0, 16): _c(128, 1, 2),
        (17, 64): _c(256, 1, 4),
        (65, 511): _c(128, 1, 8, pdl=False),
        (512, _LARGE): _c(256, 1, 8, pdl=False),
    },
    "pro6000": {
        (0, 16): _c(128, 1, 2),
        (17, 64): _c(256, 1, 4),
        (65, 511): _c(256, 2, 4),
        (512, _LARGE): _c(128, 2, 8),
    },
    "gb10": {
        (0, 4): _c(128, 1, 2),
        (5, 16): _c(256, 1, 8),
        (17, 64): _c(128, 1, 4),
        # M=256 sits at GB10's L2 edge (19 MB working set): over 8 fresh processes no
        # provider passes hot and cold; t64/e8 has the best worst-regime median. The
        # evict-first store variant was slower with the down-proj GEMM consumer.
        (65, 383): _c(64, 1, 8),
        (384, _LARGE): _c(256, 1, 2),
    },
}
TABLE = {
    kind: {(12288, dtype): bands for dtype in ("bfloat16", "float16")}
    for kind, bands in _BANDS.items()
}


def config_for(kind, width, dtype, m):
    for (lo, hi), config in TABLE.get(kind, {}).get((width, dtype), {}).items():
        if lo <= m <= hi and width % config.elems == 0:
            return config
    return None


def parse_config(setting):
    """``threads,vectors,pdl[,elems[,trigger[,cache]]]``.

    Trigger: 0=end, 1=early. Cache: 0=none, 1=st_ef, 2=ld_el, 3=both.
    """
    values = list(map(int, setting.split(",")))
    if len(values) not in (3, 4, 5, 6):
        raise ValueError("config is threads,vectors,pdl[,elems[,trigger[,cache]]]")
    values += [8, 0, 0][len(values) - 3 :]
    threads, vectors, pdl, elems, trigger, cache = values
    if any(value not in (0, 1) for value in (pdl, trigger)):
        raise ValueError("PDL and trigger must be 0 or 1")
    if cache not in (0, 1, 2, 3):
        raise ValueError("cache must be 0, 1, 2 or 3")
    return Config(
        threads,
        vectors,
        bool(pdl),
        elems,
        "early" if trigger else "end",
        ("none", "st_ef", "ld_el", "both")[cache],
    )


def tune_configs():
    return [
        Config(t, v, p, e, trigger)
        for t in (64, 128, 256)
        for v in (1, 2)
        for e in (2, 4, 8, 16)
        for p in (False, True)
        for trigger in (("end", "early") if p else ("end",))
        if not (v == 2 and e in (2, 16))
    ] + [
        Config(t, 1, True, e, "early", cache)
        for t in (64, 128, 256, 512)
        for e in (8, 4, 2)
        for cache in ("st_ef", "ld_el", "both")
    ]
