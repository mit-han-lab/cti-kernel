# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Optional CuTe activation quantization; importing this resolver does not import the DSL.

CuTe is the default on sm120/sm121 with DSL 4.8.0; elsewhere the frozen Triton quantizer
is the fallback. ``CTI_NVFP4_QUANT=triton`` forces the fallback and ``=cute`` forces CuTe:
a forced CuTe request fails on an unsupported installation instead of silently
invalidating an A/B measurement.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading

import torch

VARIABLE = "CTI_NVFP4_QUANT"
VERSION = "4.8.0"
_launchers = {}
_lock = threading.Lock()


# The whole-row loop is kept but unselected (register-limited at 512 threads).
_ROW_LOOP_MIN = 1 << 62


_TWO_GROUPS_MIN = 4_000_000


def _geometry(cols: int, rows: int, wide: bool = False) -> tuple[int, int, int, int, int]:
    """Threads, lanes per row, load groups, rows per CTA, row-loop selection.

    Zero row-loop selection preserves the qualified small-batch path. ``wide`` selects
    the sm120 large-batch geometry.
    """
    halves = cols // 8
    if wide and rows * cols >= _TWO_GROUPS_MIN:
        # RTX 5090 (2026-10-16): two loads in flight per lane from about 4M values,
        # e.g. K=17408 M=1024 10.9 vs 16.3 us; fewer values favour one group. GB10 not.
        return 128, halves, 2, max(1, 128 // halves), 0
    if rows <= _ROW_LOOP_MIN:
        # Tile path at every measured M: on GB10 it matched or beat FlashInfer 0.6.11
        # fp4_quantize at M=128..4096 and ran 1.5-6x faster than the row loop below.
        return 128, halves, 1, max(1, 128 // halves), 0
    return _row_loop_geometry(cols)


# RTX PRO 6000, 27B down (K=17408) quantize + FlashInfer mm_fp4 chains (2026-10-16):
# triggering dependents at entry let GEMM CTAs occupy SMs while quantizer CTAs were
# still unscheduled (M=32..256 lost up to 14% vs SGLang); at M>=512 the early trigger's
# GEMM prologue overlap wins (M=2048/4096: 1.017/1.008 early vs 0.967/0.984 late).
_EARLY_TRIGGER_MIN = 6_000_000


def _early_trigger(cols: int, rows: int) -> bool:
    """Trigger PDL dependents at kernel entry (True) or after each CTA's work (False)."""
    return rows * cols >= _EARLY_TRIGGER_MIN


# Device name -> ((cols, first row, last row), ...) launched without PDL: the quantizer
# then starts only after its predecessor completes, instead of waiting resident on SMs.
PDL_OFF: dict[str, tuple[tuple[int, int, int], ...]] = {}


def _pdl(cols: int, rows: int, device_name: str) -> bool:
    """Whether the quantizer launches as a PDL secondary (the default)."""
    return not any(cols == c and lo <= rows <= hi for c, lo, hi in PDL_OFF.get(device_name, ()))


def _row_edges(cols: int, device_name: str) -> set[int]:
    """Row counts at which ``_geometry``/``_early_trigger``/``_pdl`` may change for
    ``cols``; constant between consecutive edges."""
    edges = {-(-limit // cols) for limit in (_TWO_GROUPS_MIN, _EARLY_TRIGGER_MIN)}
    edges.add(_ROW_LOOP_MIN + 1)
    for c, lo, hi in PDL_OFF.get(device_name, ()):
        if c == cols:
            edges.update((lo, hi + 1))
    return edges


def _row_loop_geometry(cols: int) -> tuple[int, int, int, int, int]:
    """The unselected whole-row loop, kept with its contract tests."""
    halves = cols // 8
    # One CTA owns a whole row. Adjacent lanes stay paired across column turns,
    # including non-power-of-two warp counts and the final partial warp.
    threads = min((halves + 31) // 32 * 32, 512)
    return threads, threads, (halves + threads - 1) // threads, 1, 1


def _launch_bounds(threads: int) -> int:
    # sm120/sm121 support 1536 threads and 24 blocks per SM. An impossible
    # minnctapersm hint is ignored by ptxas; keep the 2048-thread grid separately.
    return min(2048 // threads, 24, 1536 // threads)


def available(device: torch.device) -> bool:
    if device.type != "cuda" or torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        return False
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def load(device: torch.device):
    choice = os.environ.get(VARIABLE, "")
    if choice not in ("", "cute", "triton"):
        raise ValueError(f"{VARIABLE} must be 'cute' or 'triton'")
    if choice == "triton":
        return None
    with _lock:
        if device not in _launchers:
            _launchers[device] = _resolve(device)
        launcher = _launchers[device]
    if launcher is None and choice == "cute":
        raise RuntimeError("CuTe NVFP4 quantization needs DSL 4.8.0 and sm120/sm121")
    return launcher


def _resolve(device):
    if not available(device):
        return None
    try:
        from .launcher import Launcher
    except ImportError:
        return None
    return Launcher(device)
