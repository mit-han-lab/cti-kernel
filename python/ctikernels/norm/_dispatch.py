# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""C4/C5 add + RMSNorm dispatch: measured per device class, weight kind and M band."""

from dataclasses import dataclass

from ctikernels.linear._dispatch import device_class  # noqa: F401


@dataclass(frozen=True)
class Config:
    """CuTe launch layout and PDL options.

    split > 1 is opt-in via AddRMSNorm(config=...) or CTI_NORM_CONFIG;
    it is never auto-dispatched.

    static_weight=True reads weight before griddepcontrol.wait and requires the
    public AddRMSNorm opt-in. Weight must never be written while the op can run:
    no writes between graph replays, between capture and replay, or by a preceding
    kernel in the same stream. Any writer requires static_weight=False.
    """

    # Total CTA threads; independent row groups share one CTA barrier.
    threads: int = 512
    rows: int = 1
    pdl: bool = False
    static_weight: bool = False
    trigger: str = "end"
    split: int = 1

    def __post_init__(self):
        if type(self.split) is not int or not 1 <= self.split <= 8:
            raise ValueError("split must be an integer in 1..8")
        if self.split > 1 and self.rows != 1:
            raise ValueError("split > 1 requires rows == 1")
        if self.threads == 64 and self.split == 1:
            raise ValueError("64 threads requires split > 1")
        if self.threads not in (64, 128, 256, 512, 1024) or self.rows not in (1, 2, 4):
            raise ValueError("threads must be 64/128/256/512/1024; rows must be 1/2/4")
        if self.threads // self.rows < 32:
            raise ValueError("each row needs at least 32 threads")
        if not isinstance(self.pdl, bool) or not isinstance(self.static_weight, bool):
            raise ValueError("pdl and static_weight must be bool")

        if self.trigger not in ("end", "early"):
            raise ValueError("trigger must be end or early")
        if self.trigger == "early" and not self.pdl:
            raise ValueError("early trigger requires pdl=True")

    def admits(self, hidden):
        """Bound staging registers to at most 16 eight-element chunks per thread."""
        # 32 chunks (H=8192 at 32 threads/row) spills ~3 KiB/thread on sm_120a.
        # Keep that width on >=64 threads/row; still offer 32-thread rows at H=4096.
        if hidden // 8 < self.split:
            return False
        if self.split == 1:
            return 0 < hidden <= (self.threads // self.rows) * 8 * 16
        chunks = hidden // 8
        per = (chunks + self.split - 1) // self.split
        return 0 < hidden and chunks >= self.split and per <= (self.threads // self.rows) * 16

    @property
    def name(self):
        return (
            f"cute_t{self.threads}_r{self.rows}_p{int(self.pdl)}_s{int(self.static_weight)}"
            f"_t{'e' if self.trigger == 'early' else 'n'}"
            + (f"_c{self.split}" if self.split > 1 else "")
        )


@dataclass(frozen=True)
class Entry:
    provider: str
    config: Config = Config()

    def __post_init__(self):
        if self.provider not in ("cute", "flashinfer", "triton"):
            raise ValueError("invalid norm table provider")


# Inclusive bands per (device, weight kind, static flag), from the paired sweeps in
# benchmark/kernel_tuning/results/norm_act.md.
_LARGE = 1 << 40
_FI, _TRITON = Entry("flashinfer"), Entry("triton")


def _cute(threads, rows=1, pdl=True, static_weight=False, trigger="early"):
    if not pdl:
        trigger = "end"
    return Entry("cute", Config(threads, rows, pdl, static_weight, trigger))


def _s(threads, rows=1, pdl=True):
    return _cute(threads, rows, pdl, static_weight=True)


# Decode/DFlash rows (M <= 255) of both weight conventions use the CuTe kernel with an
# early PDL trigger: it won every paired C4 sweep row (hot, engine and cold) against the
# real vLLM 0.31.0 and SGLang 72730d6 dispatch. Static tables prefetch the weight
# before griddepcontrol.wait (AddRMSNorm(static_weight=True) only).
_SMALL = {
    ("rtx5090", False): {(0, 4): _cute(512), (5, 255): _cute(256)},
    ("rtx5090", True): {(0, 255): _s(512)},
    ("pro6000", False): {(0, 16): _cute(512), (17, 255): _cute(256)},
    ("pro6000", True): {(0, 4): _s(512), (5, 255): _s(256)},
    ("gb10", False): {(0, 4): _cute(512), (5, 255): _cute(256)},
    ("gb10", True): {(0, 4): _s(512), (5, 255): _s(512, 2)},
}
# Prefill (M >= 256): standard weight keeps FlashInfer (not a C4 row). Zero-centered (C5)
# bands take the best worst-regime provider of the paired FlashInfer 0.7.0.post1 sweeps
# (the 0.1.12 pin); there FlashInfer runs the SGLang market's own CuTe kernel.
_PREFILL_ZERO = {
    ("rtx5090", False): {
        (256, 511): _cute(512, 2),
        (512, 1023): _FI,
        (1024, 2047): _cute(1024, 2, pdl=False),
        (2048, 4095): _cute(512, pdl=False),
        (4096, 8191): _cute(1024, 2, pdl=False),
        (8192, _LARGE): _TRITON,
    },
    ("rtx5090", True): {
        (256, 511): _s(512, 2),
        (512, 1023): _s(512),
        (1024, 2047): _s(128, 2),
        (2048, 4095): _s(512, pdl=False),
        (4096, 8191): _s(1024, 2),
        (8192, _LARGE): _TRITON,
    },
    ("pro6000", False): {
        (256, 511): _FI,
        (512, 2047): _cute(512),
        (2048, 4095): _cute(128, 2, pdl=False),
        (4096, 8191): _FI,
        (8192, _LARGE): _cute(128, 2, pdl=False),
    },
    ("pro6000", True): {
        (256, 511): _s(256),
        (512, 1023): _s(512),
        (1024, 4095): _s(128, 2),
        (4096, 8191): _FI,
        (8192, _LARGE): _s(128, 2),
    },
    ("gb10", False): {
        (256, 511): _cute(512),
        (512, 1023): _cute(128, 2, pdl=False),
        (1024, _LARGE): _TRITON,
    },
    ("gb10", True): {(256, 511): _s(512), (512, 1023): _s(128, 2), (1024, _LARGE): _TRITON},
}


def _bands(kind, zero, static):
    prefill = _PREFILL_ZERO[kind, static] if zero else {(256, _LARGE): _FI}
    return {**_SMALL[kind, static], **prefill}


_BANDS = {
    (kind, zero, static): _bands(kind, zero, static)
    for kind in ("rtx5090", "pro6000", "gb10")
    for zero in (False, True)
    for static in (False, True)
}


TABLE = {
    kind: {
        (4096, dtype, dtype, zero, static): _BANDS[kind, zero, static]
        for dtype in ("bfloat16", "float16")
        for zero in (False, True)
        for static in (False, True)
    }
    for kind in ("rtx5090", "pro6000", "gb10")
}


# FP32-residual bands (x FP16/BF16, residual and sum_out FP32), per (device, hidden,
# static flag): the best worst-regime provider of the paired --residual-dtype fp32
# sweeps in benchmark/kernel_tuning/results/norm_act.md (RMSNORM-W). Measured with the
# standard weight; the CuTe and Triton kernels differ for zero-centered weights only
# by the +1 on the scale, so both weight kinds and both x dtypes share the bands.
_FP32_BANDS = {
    ("gb10", 2048, False): {
        (0, 2): _cute(128),
        (3, 8): _cute(256),
        (9, 40): _cute(512),
        (41, 255): _cute(256, 2),
        (256, 511): _cute(256, 1, pdl=False),
        (512, _LARGE): _TRITON,
    },
    ("gb10", 2048, True): {
        (0, 2): _s(256),
        (3, 8): _s(256),
        (9, 40): _s(512),
        (41, 255): _s(512),
        (256, 511): _s(256),
        (512, _LARGE): _TRITON,
    },
    ("gb10", 4096, False): {
        (0, 2): _cute(512),
        (3, 8): _cute(512),
        (9, 40): _cute(512),
        (41, 255): _cute(256, 2),
        (256, 511): _TRITON,
        (512, _LARGE): _cute(128, 2, pdl=False),
    },
    ("gb10", 4096, True): {
        (0, 2): _s(512),
        (3, 8): _s(512),
        (9, 40): _s(512),
        (41, 255): _s(256, 2),
        (256, 511): _TRITON,
        (512, _LARGE): _s(256),
    },
    ("pro6000", 2048, False): {
        (0, 2): _cute(256),
        (3, 8): _cute(256),
        (9, 40): _cute(256),
        (41, 255): _cute(256),
        (256, 511): _cute(512),
        (512, _LARGE): _cute(512),
    },
    ("pro6000", 2048, True): {
        (0, 2): _s(256),
        (3, 8): _s(256),
        (9, 40): _s(256),
        (41, 255): _s(256),
        (256, 511): _s(512, 2),
        (512, _LARGE): _s(256),
    },
    ("pro6000", 4096, False): {
        (0, 2): _cute(512),
        (3, 8): _cute(512),
        (9, 40): _cute(256),
        (41, 255): _cute(512),
        (256, 511): _cute(512),
        (512, _LARGE): _cute(512),
    },
    ("pro6000", 4096, True): {
        (0, 2): _s(512),
        (3, 8): _s(512),
        (9, 40): _s(512),
        (41, 255): _s(256),
        (256, 511): _s(1024, 2),
        (512, _LARGE): _s(512),
    },
    ("rtx5090", 2048, False): {
        (0, 2): _cute(128),
        (3, 8): _cute(256),
        (9, 40): _cute(256),
        (41, 255): _cute(256),
        (256, 511): _cute(512),
        (512, _LARGE): _cute(256, 2, pdl=False),
    },
    ("rtx5090", 2048, True): {
        (0, 2): _s(128),
        (3, 8): _s(256),
        (9, 40): _s(256),
        (41, 255): _s(256),
        (256, 511): _s(1024, 2),
        (512, _LARGE): _TRITON,
    },
    ("rtx5090", 4096, False): {
        (0, 2): _cute(512),
        (3, 8): _cute(512),
        (9, 40): _TRITON,
        (41, 255): _cute(256),
        (256, 511): _cute(256, 1, pdl=False),
        (512, _LARGE): _cute(256, 2, pdl=False),
    },
    ("rtx5090", 4096, True): {
        (0, 2): _s(512),
        (3, 8): _s(512),
        (9, 40): _s(512),
        (41, 255): _s(512),
        (256, 511): _s(512),
        (512, _LARGE): _s(256),
    },
}


_FP32_RESIDUAL = {
    kind: {
        (hidden, dtype, "float32", zero, static): _FP32_BANDS[kind, hidden, static]
        for hidden in (2048, 4096)
        for dtype in ("bfloat16", "float16")
        for zero in (False, True)
        for static in (False, True)
    }
    for kind in ("rtx5090", "pro6000", "gb10")
}


def entry_for(kind, hidden, dtype, m, zero_centered, static_weight=False, residual_dtype=None):
    residual_dtype = dtype if residual_dtype is None else residual_dtype
    table = _FP32_RESIDUAL if residual_dtype != dtype else TABLE
    for (lo, hi), entry in (
        table.get(kind, {})
        .get((hidden, dtype, residual_dtype, zero_centered, static_weight), {})
        .items()
    ):
        if lo <= m <= hi:
            return entry
    return None


def parse_config(setting):
    """``threads,rows,pdl[,static_weight[,trigger[,split]]]``; trigger 0=end, 1=early."""
    values = list(map(int, setting.split(",")))
    if len(values) not in (3, 4, 5, 6):
        raise ValueError("config is threads,rows,pdl[,static_weight[,trigger[,split]]]")
    values += [0, 0, 1][len(values) - 3 :]
    threads, rows, pdl, static_weight, trigger, split = values
    if any(value not in (0, 1) for value in (pdl, static_weight, trigger)):
        raise ValueError("PDL, static_weight and trigger must be 0 or 1")
    return Config(
        threads, rows, bool(pdl), bool(static_weight), "early" if trigger else "end", split
    )


def tune_configs(hidden=4096):
    return [
        config
        for t in (128, 256, 512, 1024)
        for r in (1, 2, 4)
        for p in (False, True)
        for s in (False, True)
        for trigger in (("end", "early") if p else ("end",))
        if (config := Config(t, r, p, s, trigger)).admits(hidden)
    ]
