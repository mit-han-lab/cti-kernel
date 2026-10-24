# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: SGLang 72730d6 and flash-linear-attention's chunked WY delta rule;
# independently implemented, no upstream kernel code copied.

"""Lazy host adapter for pipelined 64-token CuTe prefill.

Short rows use serial split preparation, recurrence and norm/history: four kernels,
with no memset, flags, events, or side stream.

Importing this module does not import cutlass, initialize CUDA, or read device values.
The DSL implementation is in ``_cute_prefill_kernels``. Compile failures propagate (in
particular, a forced CuTe benchmark must never silently measure Triton).

Overlap uses call-local int32 flags reset by cuMemsetD32Async on the caller
stream, followed by one Programmatic Dependent Launch (PDL) chain. The first
prepare launches ordinarily, fully ordered after the reset and earlier work.
Every prepare CTA triggers dependents as its first statement. Split value
prepare launches with PDL, triggers, then waits for key prepare before reading
its outputs. Recurrence launches with PDL and acquires per-chunk ready flags.
It waits for prepare completion only as its last statement, after all stores;
the following ordinary output (when split) and norm launches are fully ordered.
Norm can then safely commit history.

PDL admits a dependent grid only after EVERY primary CTA has triggered or
exited. Thus any recurrence CTA's awaited flag belongs to a prepare CTA already
resident or finished. Prepare never waits on recurrence, so this construction
makes progress across concurrent calls, streams and graphs without admission
budgets. If PDL is not honored, ordinary serialization is still correct.
Graph capture records programmatic edges and resets the entire flag capacity
on every replay. No side streams, events, host synchronization or epochs are used.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import itertools
import os
import threading

import torch

from ._contract import Geometry, HashedTuple

VERSION = "4.8.0"
CHUNK = 64
BV = 64
V2_QUALIFIED_GEOMETRIES = frozenset({(16, 32), (16, 48), (8, 16), (8, 24)})
DEFAULT_TUNE = {
    "prepare_warps": 8,
    "recurrence_warps": 4,
    "bv": BV,
    "overlap": 1,
    "overlap_min_tokens": 512,
    "rec_ws": 0,
    "overlap_key": 1,
    "overlap_norm": 0,
    "merge_prepare": 0,
    "rec_mode": 2,
    "prep_mode": -1,  # automatic: grouped for overlapped Hv<=32, split otherwise
    "prefetch_depth": 0,
    "split_h": 0,
    "form": "wy",
    "t_store": "split",
    "r_split": 1,
    "k_split": 0,
    "v_store": "bf16",
    "u_store": "fp32",
    "ut_split": 0,
    "w_split": 0,
    "o_bf16": 0,
    "o_split": 0,
    "bvo": 64,
    "output_warps": 4,
    "update_split": 0,
    "aq_qk": 1,
    "aq_out": 0,
    "o_form": "legacy",
    "qp_split": 1,
    "solve_mma": 2,
    "stop_after": "all",
    "mma_m": 32,
    "mma_n": 32,
}


def resolve_tune(tune, hv, one_wave=False):
    """Compile-time schedule; unknown capacity uses the compact shared layout.

    Depth 4 stages every next-chunk operand after WH/QH retire. Depth 1
    starts W after WH, U/g after QH, and streams Q/A/K through retired W.
    The latter fits two BV32 CTAs per SM. Extra precision W/K terms and TK
    retain the established recurrence, including its serial fallback.
    """
    tune = tune.copy()
    validate_output_split(tune)
    if tune["prep_mode"] == -1:
        # Measured on RTX 5090 / RTX PRO 6000: the grouped prepare only pays off
        # for overlapped Hv=32 calls; split is faster serially and at Hv=48.
        # Generic policy: keep Hv <= 32 for untuned geometries too. Grouped
        # prepare loops over R (including R=8), reusing the same shared storage
        # each iteration; R does not increase its shared-memory footprint.
        tune["prep_mode"] = int(bool(tune["overlap"]) and hv <= 32)
    if tune["merge_prepare"]:
        tune["prep_mode"] = 0
    if tune["rec_mode"] == 2 and (tune["form"] != "wy" or tune["w_split"] or tune["k_split"]):
        tune["rec_mode"] = 1 if tune["form"] == "wy" else 0
    if tune["rec_ws"]:
        tune["recurrence_warps"] = 8
        tune["prefetch_depth"] = 1
    if tune["prefetch_depth"] == 0:
        tune["prefetch_depth"] = 4 if one_wave and tune["bv"] <= 32 else 1
    if tune["rec_mode"] == 2 and tune["prefetch_depth"] == 4 and tune["bv"] > 32:
        raise ValueError("prefetch_depth=4 requires bv<=32 (96 KiB budget)")
    return tune


# Construction defaults, keyed explicitly so measured rows can be edited independently.
# aq_qk is off at Hv=32: the plain BF16 Q K^T already passes the fla-core gate there,
# and the extra prepare products slow the overlapped Hv=32 rows (RTX PRO 6000, T=2048).
# Overlap defaults follow paired serial/overlap timings at N = 1, 2, 4 and T = 512..16k:
# on sm120 overlap wins only at Hv = 32 from 2048 tokens;
# on GB10 at Hv = 32 from 512 and at Hv = 48 from 2048.
GEOMETRY_TUNES = {
    # TP2 entries copy Hv=32/48 respectively; provisional until measured on GPUs.
    ((12, 0), (8, 16)): {"o_split": 0, "bv": 32, "aq_qk": 0, "overlap_min_tokens": 2048},
    ((12, 0), (8, 24)): {"o_split": 0, "bv": 32, "overlap": 0},
    ((12, 1), (8, 16)): {"o_split": 0, "bv": 32, "aq_qk": 0},
    ((12, 1), (8, 24)): {"o_split": 0, "bv": 64, "overlap_min_tokens": 2048},
    ((12, 0), (16, 32)): {"o_split": 0, "bv": 32, "aq_qk": 0, "overlap_min_tokens": 2048},
    ((12, 0), (16, 48)): {"o_split": 0, "bv": 32, "overlap": 0},
    ((12, 1), (16, 32)): {"o_split": 0, "bv": 32, "aq_qk": 0},
    ((12, 1), (16, 48)): {"o_split": 0, "bv": 64, "overlap_min_tokens": 2048},
}


# Measured row overrides from paired sweeps.
# First match wins: (min_tokens, max_tokens, [min_count,] max_count, overrides).
# None is unbounded; legacy four-tuples have no minimum count.
ROW_TUNES = {
    # sm120 (RTX 5090 and RTX PRO 6000 measured alike). o_split only pays off for
    # one sequence: equal-length batches (N >= 2) are throughput-bound and lose
    # 8-39%, and N alone cannot tell them from skewed batches, so N > 1 keeps the base.
    ((12, 0), (8, 16)): ((256, None, 1, {"bv": 16, "overlap": 0, "o_split": 1, "bvo": 128}),),
    ((12, 0), (8, 24)): (
        (512, 4096, 1, {"o_split": 1, "bvo": 64}),
        (4097, None, 1, {"o_split": 1, "bvo": 128}),
    ),
    ((12, 0), (16, 32)): ((512, 1024, 1, {"o_split": 1, "bvo": 64}),),
    # Hv=48 and every GB10 geometry: o_split never wins (GB10 TP1 loses 30-50%).
    # GB10 v2 rows (fused gated norm): each measured row has bootstrap 95% CI_low > 1
    # against both vLLM 0.31.0 and SGLang 72730d6 and against frozen v1 (60
    # interleaved pairs, X925-pinned, one quiet GB10). Single sequences of 8 value-head
    # groups use vsplit 2 (24 -> 48 CTAs); 8x16 N=4 at T=8192 also covers the skewed
    # [6000, 1500, 500, 192] row, where vsplit 2 wins and vsplit 1 only ties v1.
    ((12, 1), (16, 32)): ((256, None, None, {"v2": 1}),),
    ((12, 1), (16, 48)): ((256, None, None, {"v2": 1}),),
    ((12, 1), (8, 24)): (
        (1024, None, 1, 1, {"v2": 1, "vsplit": 2}),
        (512, None, 2, None, {"v2": 1}),
    ),
    ((12, 1), (8, 16)): (
        (4096, None, 2, 3, {"v2": 1}),
        (4096, None, 4, 4, {"v2": 1, "vsplit": 2}),
        (4096, None, 5, None, {"v2": 1}),
        # Every remaining (v1) row: two-plane BF16 Q K^T. Single-plane Q K^T lost to
        # fla-core on the five-seed mean y max error at [64] (0.0877 vs 0.0860) and
        # tied at [512] (0.1227 vs 0.1230); aq_qk=1 gives 0.053 and 0.069 for 2.3-7.5%
        # (GB10 sweep, T <= 2048). aq_out=1 as well costs up to 15%.
        (None, None, None, None, {"aq_qk": 1}),
    ),
}


def read_tune(value=None, capability=None, geometry=None):
    """Validated, construction-time knobs; never re-read during graph capture."""
    config = DEFAULT_TUNE | GEOMETRY_TUNES.get((capability, geometry), {})
    value = os.environ.get("CTI_GDN_CUTE_TUNE", "") if value is None else value
    seen = set()
    for item in filter(None, value.split(",")):
        parts = item.split("=")
        if len(parts) != 2:
            raise ValueError("CTI_GDN_CUTE_TUNE expects comma-separated k=v pairs")
        key = parts[0].strip().lower()
        if (
            key not in DEFAULT_TUNE
            and key not in ("aq_precise", "v2", "vsplit", "v2_debug", "v2_ldt", "v2_snorm")
        ) or key in seen:
            raise ValueError("unknown or duplicate CTI_GDN_CUTE_TUNE key")
        try:
            number = (
                parts[1].strip().lower()
                if key in ("stop_after", "form", "o_form", "t_store", "v_store", "u_store")
                else int(parts[1])
            )
        except ValueError:
            raise ValueError("CTI_GDN_CUTE_TUNE values must be integers") from None
        config[key] = number
        seen.add(key)
    return _validate_tune(config, capability, geometry, seen)


def _validate_tune(config, capability, geometry, seen):
    """Shared validation and canonicalization for explicit and row tunes."""
    allowed = {
        "v2": (0, 1),
        "v2_debug": (0, 1),
        "v2_ldt": (0, 1),
        # 1 keeps the separate norm kernel (FP32 o in DRAM); 0 fuses the gated norm.
        "v2_snorm": (0, 1),
        "vsplit": (1, 2, 4),
        "prepare_warps": (4, 8),
        "recurrence_warps": (2, 4, 8),
        "bv": (16, 32, 64),
        "overlap": (0, 1),
        "overlap_min_tokens": range(0, 2**31),
        "rec_ws": (0, 1),
        "overlap_key": (0, 1),
        # Removed: persistent norm waiters have no supported progress guarantee.
        "overlap_norm": (0,),
        "merge_prepare": (0, 1),
        "rec_mode": (0, 1, 2),
        "prep_mode": (-1, 0, 1),
        "prefetch_depth": (0, 1, 4),
        "split_h": (0, 1),
        "form": ("tk", "wy"),
        "t_store": ("split", "fp32", "bf16"),
        "r_split": (0, 1),
        "k_split": (0, 1),
        "v_store": ("bf16", "fp32"),
        "u_store": ("bf16", "fp32"),
        "ut_split": (0, 1),
        "w_split": (0, 1),
        "o_bf16": (0, 1),
        "o_split": (0, 1),
        "bvo": (32, 64, 128),
        "output_warps": (4, 8),
        "update_split": (0, 1),
        "aq_precise": (0, 1),
        "aq_qk": (0, 1),
        "aq_out": (0, 1),
        "o_form": ("legacy", "pre"),
        "qp_split": (0, 1),
        "solve_mma": (0, 1, 2),
        "stop_after": ("conv", "norm", "gram", "solve", "wy", "all"),
        "mma_m": (16, 32),
        "mma_n": (16, 32),
    }
    for key, value in config.items():
        if key not in allowed:
            raise ValueError("unknown CTI_GDN_CUTE_TUNE key")
        if value not in allowed[key]:
            raise ValueError(f"CTI_GDN_CUTE_TUNE {key} must be one of {allowed[key]}")
    # Compatibility alias supplies both knobs; explicit independent knobs win,
    # regardless of their order in the tuning string. Keep compile keys canonical.
    if "aq_precise" in seen:
        precise = config.pop("aq_precise")
        for key in ("aq_qk", "aq_out"):
            if key not in seen:
                config[key] = precise
    # Precise compact output keeps the two-CTA shared-memory budget at BV32.
    # Keep the measured legacy BV64 ablation and explicit user tuning available.
    if (
        (config["aq_out"] or config["o_form"] == "pre")
        and capability == (12, 1)
        and "bv" not in seen
    ):
        config["bv"] = 32
    # Omitted and explicit zero have the original compile key and scratch ABI.
    for key in ("v2_debug", "v2_ldt", "v2_snorm"):
        if config.get(key, 0) and not config.get("v2", 0):
            raise ValueError(f"{key}=1 requires v2=1")
        if not config.get(key, 0):
            config.pop(key, None)
    if config.get("v2", 0):
        if geometry is not None and geometry not in V2_QUALIFIED_GEOMETRIES:
            raise ValueError(
                "v2 is precision-qualified only at (16,32), (16,48), (8,16), (8,24); "
                "(1,8) measured RMS 1.9e-3 vs fla-core 1.2e-3"
            )
        if config.get("overlap") and "overlap" in seen:
            raise ValueError("v2 requires serial execution (overlap=0)")
        required = {
            "form": "wy",
            "o_split": 0,
            "o_form": "legacy",
            "stop_after": "all",
            "solve_mma": 2,
            "u_store": "fp32",
            "o_bf16": 0,
            "split_h": 0,
            "update_split": 0,
            "ut_split": 0,
            "w_split": 0,
            "k_split": 0,
            "aq_out": 0,
        }
        for name, expected in required.items():
            if config[name] != expected:
                raise ValueError(f"v2 phase 1 requires {name}={expected}")
        config.update(overlap=0, bv=32, vsplit=config.get("vsplit", 1))
        # Precision belongs to the head ratio in the fused constructor, not a tune.
        if geometry is not None:
            hk, hv = geometry
            Geometry(hk, hv)
            precise = int(hv // hk == 3)
            if "aq_qk" in seen and config["aq_qk"] != precise:
                raise ValueError("v2 requires the geometry's default aq_qk precision")
            config["aq_qk"] = precise
    else:
        # Even explicit v2=0 must retain the exact v1 tune/compile key.
        config.pop("v2", None)
        config.pop("vsplit", None)
    validate_output_split(config)
    warps_to_check = [config["prepare_warps"], config["recurrence_warps"]]
    if config["o_split"]:
        warps_to_check.append(config["output_warps"])
    for warps in warps_to_check:
        if config["mma_m"] * config["mma_n"] < warps * 16 * 8:
            raise ValueError("MMA tile must contain at least one m16n8 atom per warp")
    if config["rec_ws"] and (
        config["rec_mode"] != 2
        or config["form"] != "wy"
        or config["bv"] > 32
        or any(config[k] for k in ("split_h", "k_split", "w_split"))
    ):
        raise ValueError("rec_ws=1 requires rec_mode=2, WY, bv<=32 and no h/k/w split")
    if config["o_form"] == "pre" and (
        config["form"] != "wy"
        or config["rec_mode"] != 2
        or any(config[k] for k in ("rec_ws", "w_split", "k_split"))
    ):
        raise ValueError(
            "o_form=pre requires WY rec_mode=2 without rec_ws/w_split/k_split; use o_form=legacy"
        )
    return config


def validate_output_split(tune):
    """Reject fallbacks that would silently bypass the state/output split."""
    if tune["o_split"] and (
        tune["rec_mode"] != 2
        or tune["form"] != "wy"
        or tune["o_form"] != "legacy"
        or any(tune[k] for k in ("rec_ws", "split_h", "w_split", "k_split"))
    ):
        raise ValueError(
            "o_split=1 requires rec_mode=2, form=wy, o_form=legacy and "
            "rec_ws=split_h=w_split=k_split=0"
        )


def validate_captures(tune, captures):
    """The capturing executable (``captures`` = 1) supports the dispatched WY recurrence and
    buffer schedules; every product tune qualifies, CTI_GDN_CUTE_TUNE overrides may not."""
    if captures and (
        tune["rec_mode"] != 2
        or tune["form"] != "wy"
        or tune["o_form"] != "legacy"
        or tune["stop_after"] != "all"
        or any(tune[k] for k in ("rec_ws", "w_split", "k_split", "merge_prepare"))
    ):
        raise ValueError(
            "capture requires WY rec_mode=2, legacy output, no rec_ws/w/k split or merge"
        )


def capture_offsets(cu_seqlens):
    """Each sequence's capture point p = floor(L / 64) * 64, int64 [N], computed on the device
    (two launches, no host read), for the kernels' on-grid capture (p = 0 writes nothing)."""
    lengths = torch.empty(cu_seqlens.shape[0] - 1, dtype=torch.int64, device=cu_seqlens.device)
    torch.sub(cu_seqlens[1:], cu_seqlens[:-1], out=lengths)
    return lengths.bitwise_and_(-CHUNK)


def chunk_capacity(tokens, count):
    """Bound sum(ceil(L_i/64)), allowing empty sequences without reading lengths.

    At most min(N,T) sequences are nonempty; each first chunk costs one token
    and each additional chunk costs 64. The bound increases with that count.
    """
    nonempty = min(tokens, count)
    return nonempty + (tokens - nonempty) // CHUNK


def pipeline_variant(tokens, tune):
    """Two symbolic-T executables, selected entirely by host shape and tuning."""
    return (
        "serial"
        if tokens < tune["overlap_min_tokens"] or not tune["overlap"] or tune["stop_after"] != "all"
        else 0
    )


_compiling = threading.Lock()

# Scratch buffers in executable argument order (between the pools and y, then after y).
SCRATCH = (
    "q",
    "k",
    "w",
    "u",
    "aq",
    "g",
    "history_out",
    "raw_out",
    "partial",
    "kk",
    "qk",
    "kl",
    "flags",
    "qp",
    "op",
    "hbuf",
    "vbuf",
)


def scratch_layout(geometry, tune, tokens, count, launcher=None):
    """(elements, dtype) of every SCRATCH buffer of one call, from host shapes only.

    chunk_capacity bounds sum ceil(length / 64). Ownership is computed on device,
    including on graph replay; unused capacity is never read by K2/K3.
    """
    hk, hv = geometry.k_heads, geometry.v_heads
    f32, bf16 = torch.float32, torch.bfloat16
    chunks = 0 if tune.get("v2", 0) else chunk_capacity(tokens, count)
    kh_store = hv if tune["merge_prepare"] else hk
    pre = tune["o_form"] == "pre"
    qk_split = pre or tune["aq_qk"]
    aq_split = pre or tune["aq_out"]
    tk = tune["form"] == "tk"
    t_elements = CHUNK * CHUNK * (2 if tune["t_store"] == "split" else 1)
    # The fused v2 norm writes y directly: no FP32 o or norm partials in DRAM.
    unfused = not (tune.get("v2", 0) and not tune.get("v2_snorm", 0))
    overlapped = pipeline_variant(tokens, tune) != "serial" and not getattr(
        launcher, "launch_blocking", False
    )
    return (
        # Q_hi then Q_lo slabs; the recurrence always reads contiguous Q_hi.
        (chunks * kh_store * CHUNK * 128 * (1 + qk_split), bf16),
        (chunks * kh_store * CHUNK * 128, bf16),
        (
            chunks * hv * (t_elements if tk else CHUNK * 128 * (1 + tune["w_split"])),
            f32 if tk and tune["t_store"] == "fp32" else bf16,
        ),
        (
            chunks * hv * CHUNK * 128,
            f32 if tune["v_store" if tk else "u_store"] == "fp32" else bf16,
        ),
        (chunks * hv * CHUNK * CHUNK * (1 + aq_split), bf16),
        (chunks * hv * CHUNK * (2 if tk else 1), f32),
        (count * 3 * geometry.conv_dim, bf16),
        (tokens * hv * 128 * unfused, bf16 if tune["o_bf16"] else f32),
        # Preserve the recurrence BV reduction groups even when output uses BVO.
        (tokens * hv * max(4, 128 // tune["bv"]) * unfused, f32),
        (chunks * kh_store * CHUNK * CHUNK, f32),
        (chunks * kh_store * CHUNK * CHUNK, f32),
        (chunks * kh_store * CHUNK * 128 if tune["k_split"] or qk_split else 0, bf16),
        (chunks * hv if overlapped else 0, torch.int32),
        (chunks * hv * CHUNK * 128 * (1 + tune["qp_split"]) if pre else 0, bf16),
        (chunks * hv * CHUNK * 128 if pre else 0, f32),
        (chunks * hv * 128 * 128 if tune["o_split"] else 0, bf16),
        (chunks * hv * CHUNK * 128 if tune["o_split"] else 0, f32),
    )


# ---------------------------------------------------------------------------- host fast path
#
# Every GatedDeltaNetPrefill layer of an engine step calls with the same (T, N) on one stream.
# The host decisions of such a call (row tune, compile key, launch limits, scratch layout) are
# a pure function of the launcher's construction-time policy and (T, N, row stride), so they are computed once per process and shared by every layer (_PLANS, keyed
# by small integer policy ids). Eager calls carve their scratch from one buffer per (device,
# stream) (_Arena): the layers of a step run in stream order, so they can share it the way
# the caching allocator already hands one call's freed scratch to the next. Its views are
# converted to TVM-FFI tensors once per (arena, plan). Host threads may share a stream (and
# TVM-FFI launches release the GIL), so each call holds the arena's lock from taking its
# views until its last launch is enqueued. Graph captures, calls above
# ARENA_MAX_BYTES and opt-in diagnostics allocate their own scratch, as before.

ARENA_MAX_BYTES = 128 << 20
_ALIGN = 256
_PLAN_LIMIT = 4096
_POLICIES: dict = {}
_PLANS: dict = {}
_ARENAS: dict = {}
_EMPTY: dict = {}
_host_lock = threading.Lock()


_plan_ids = itertools.count()


def policy_id(policy):
    """A small integer for a hashable construction-time policy, shared across operations."""
    ident = _POLICIES.get(policy)
    if ident is None:
        with _host_lock:
            ident = _POLICIES.setdefault(policy, len(_POLICIES))
    return ident


def _ffi_available():
    try:
        import tvm_ffi
    except ImportError:
        return False
    return hasattr(tvm_ffi, "from_dlpack")


class _Plan:
    """Host decisions of one (policy, T, N, stride) call; no tensors. ``key`` compiles the
    call without captures, ``capture_key`` (None if the tune rejects captures) with them."""

    __slots__ = ("id", "key", "capture_key", "tune", "empty", "layout", "nbytes", "legacy")

    def __init__(self, ident, key, capture_key, tune, empty, layout, nbytes, legacy):
        self.id, self.key, self.capture_key, self.tune = ident, key, capture_key, tune
        self.empty, self.layout, self.nbytes, self.legacy = empty, layout, nbytes, legacy


class _Arena:
    """One eager scratch buffer per (device, stream), its per-plan argument views and the
    lock a call holds across its whole enqueue sequence."""

    __slots__ = ("buffer", "views", "lock")

    def __init__(self, buffer):
        self.buffer, self.views, self.lock = buffer, {}, threading.Lock()


def _empty(device, dtype):
    """Shared zero-size argument (a fresh empty tensor has no storage either)."""
    entry = _EMPTY.get((device, dtype))
    if entry is None:
        import tvm_ffi

        tensor = torch.empty(0, dtype=dtype, device=device)
        entry = _EMPTY[(device, dtype)] = (tensor, tvm_ffi.from_dlpack(tensor))
    return entry


def _carve(buffer, plan, device=None):
    """SCRATCH arguments of ``plan`` (and the flags tensor for its reset): TVM-FFI views of
    ``buffer``, or with no buffer one fresh allocation per nonempty buffer, as __call__."""
    views, flags = [], None
    for name, offset, numel, dtype in plan.layout:
        if numel == 0:
            tensor, converted = _empty(device if buffer is None else buffer.device, dtype)
        elif buffer is None:
            tensor = converted = torch.empty(numel, dtype=dtype, device=device)
        else:
            import tvm_ffi

            tensor = buffer[offset : offset + numel * dtype.itemsize].view(dtype)
            converted = tvm_ffi.from_dlpack(tensor)
        if name == "flags":
            flags = tensor
        views.append(converted)
    return tuple(views), flags


def _current_stream_id(device):
    try:
        return torch._C._cuda_getCurrentStream(device.index)[0]
    except (AttributeError, TypeError, IndexError):
        return torch.cuda.current_stream(device).stream_id


def _arena_views(device, plan):
    """This stream's arena lock and its pre-converted scratch views for ``plan``; grows the
    arena eagerly if needed. The caller holds the lock while it enqueues launches using them."""
    stream = (device.index, _current_stream_id(device))
    arena = _ARENAS.get(stream)
    if arena is not None:
        views = arena.views.get(plan.id)
        if views is not None:
            return arena.lock, views
    with _host_lock:
        arena = _ARENAS.get(stream)
        if arena is None or arena.buffer.numel() < plan.nbytes:
            # The old buffer returns to the caching allocator on this stream, after the work
            # already queued on it; its views (and their TVM-FFI handles) go with it.
            size = max(plan.nbytes, 0 if arena is None else arena.buffer.numel())
            arena = _ARENAS[stream] = _Arena(torch.empty(size, dtype=torch.uint8, device=device))
        if len(arena.views) >= _PLAN_LIMIT:
            arena.views.clear()
        views = arena.views[plan.id] = _carve(arena.buffer, plan)
        return arena.lock, views


def reserve_scratch(device, nbytes):
    """Grow the current stream's arena to ``nbytes`` (at most ARENA_MAX_BYTES), eagerly."""
    nbytes = min(nbytes, ARENA_MAX_BYTES)
    if nbytes <= 0:
        return
    stream = (device.index, _current_stream_id(device))
    with _host_lock:
        arena = _ARENAS.get(stream)
        if arena is None or arena.buffer.numel() < nbytes:
            _ARENAS[stream] = _Arena(torch.empty(nbytes, dtype=torch.uint8, device=device))


def reset_flags(flags, stream):
    """A capturable memset node, ordered before the first ordinary prepare launch."""
    from cuda.bindings import driver as cuda

    with torch.cuda.device(flags.device):
        (error,) = cuda.cuMemsetD32Async(flags.data_ptr(), 0, flags.numel(), stream.cuda_stream)
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CuTe ready flag memset failed: {error.name}")


def _footprint(record):
    """Conservative allocation vector: registers/partition, smem, threads, CTAs.

    Four register partitions, 256-register warp allocation. Charge ceil(warps/4)
    to EVERY partition, including two-warp CTAs, so no placement is assumed.
    Shared allocations round up to 256 bytes (conservative on SM12x).
    """
    warps = (record["threads"] + 31) // 32
    return (
        ((record["registers"] * 32 + 255) // 256) * 256 * ((warps + 3) // 4),
        ((record["smem_bytes"] + 255) // 256) * 256,
        ((warps + 3) // 4) * 128,
        1,
    )


def available() -> bool:
    """Whether the required distribution is installed; no CUDA/DSL initialization."""
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def select(
    capability, tokens: int, sequences: int, mode: str | None, hv=48, device_kind=None
) -> bool:
    """Host-shape policy, independent of lengths/slots stored in device tensors."""
    if mode == "0" or capability not in ((12, 0), (12, 1)):
        return False
    return tokens >= 0 and sequences >= 0 and (tokens == 0 or sequences > 0)


def load(device: torch.device, geometry=(16, 48)):
    if not available():
        return None
    try:
        from . import _cute_prefill_kernels
    except ImportError:
        return None
    return Launcher(device, _cute_prefill_kernels, geometry)


class Launcher:
    def __init__(self, device, kernels, geometry=(16, 48)):
        self.device = device
        self.kernels = kernels
        self.compiled = {}
        cc = torch.cuda.get_device_capability(device)
        value = os.environ.get("CTI_GDN_CUTE_TUNE", "")
        self.tune = read_tune(value, capability=cc, geometry=geometry)
        self.tune_key = tuple(sorted(self.tune.items()))
        rows = []
        for row in () if value else ROW_TUNES.get((cc, geometry), ()):
            if len(row) == 4:
                lo, hi, max_count, overrides = row
                min_count = None
            else:
                lo, hi, min_count, max_count, overrides = row
            tune = _validate_tune(self.tune | overrides, cc, geometry, overrides)
            rows.append((lo, hi, min_count, max_count, tune, tuple(sorted(tune.items()))))
        self.row_tunes = tuple(rows)
        self.options = f"--enable-tvm-ffi --gpu-arch sm_{cc[0]}{cc[1]}a"
        props = torch.cuda.get_device_properties(device)
        self.sms = props.multi_processor_count
        self.launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0")
        # Host fast path (see _Plan): real CUDA devices with TVM-FFI tensor arguments.
        self.fast = device.type == "cuda" and torch.cuda.is_available() and _ffi_available()
        self._policy = policy_id(
            (
                device.index,
                tuple(geometry),
                cc,
                self.tune_key,
                tuple((*row[:4], row[5]) for row in self.row_tunes),
                self.sms,
                self.launch_blocking,
            )
        )
        self._weights = None

    def _row_tune(self, tokens, count):
        """Select a prevalidated tune using host shapes only; never inspect tensor data.

        A count band selects one of finitely many tunes. N itself remains
        symbolic: only the tune key and the existing one_wave class specialize it.
        """
        for lo, hi, min_count, max_count, tune, tune_key in getattr(self, "row_tunes", ()):
            if (
                (lo is None or lo <= tokens)
                and (hi is None or tokens <= hi)
                and (min_count is None or min_count <= count)
                and (max_count is None or count <= max_count)
            ):
                return tune, tune_key
        return self.tune, self.tune_key

    def _effective_tune(self, hv, one_wave, key=None):
        """Resolve the host-only overlap choice, independent of compilation/resources."""
        tune = self.tune.copy() if key is None else dict(key[5])
        if (
            (key is not None and key[2] == "serial")
            or getattr(self, "launch_blocking", False)
            or tune["stop_after"] != "all"
        ):
            tune["overlap"] = 0
        return resolve_tune(tune, hv, one_wave)

    def compile_key(self, geometry, projected, count, captures=0):
        """Row tune and wave class specialize otherwise symbolic T and N; ``captures`` (0 or
        1) selects the capturing executable, compiled beside the other (:meth:`warm_up`)."""
        return self.variant_key(geometry, projected.shape[0], count, projected.stride(0), captures)

    def variant_key(self, geometry, tokens, count, stride, captures=0):
        """``compile_key`` from host shapes alone."""
        tune, tune_key = self._row_tune(tokens, count)
        if tune.get("v2", 0):
            validate_captures(tune, captures)
        one_wave = (
            False
            if tune.get("v2", 0)
            else count * geometry.v_heads * (128 // tune["bv"]) <= getattr(self, "sms", 0)
        )
        return (
            geometry.k_heads,
            geometry.v_heads,
            pipeline_variant(tokens, tune),
            one_wave,
            stride,
            tune_key,
            captures,
        )

    def variant_keys(self, geometry, max_tokens, max_sequences, stride, select=None):
        """Every compile key of calls with T <= max_tokens and N <= max_sequences, with and
        without captures, keyed as ``variant_key``. Keys are constant between the
        breakpoints of the row tunes, the overlap threshold and the wave class, so the
        start of each interval represents it. ``select(T, N)`` filters CuTe calls."""
        tunes = (self.tune, *(row[4] for row in getattr(self, "row_tunes", ())))
        tokens, counts = {1}, {1}
        for lo, hi, min_count, max_count, *_ in getattr(self, "row_tunes", ()):
            tokens.update(v for v in (lo, None if hi is None else hi + 1) if v is not None)
            counts.update(v for v in (min_count, None if max_count is None else max_count + 1) if v)
        for tune in tunes:
            tokens.add(tune["overlap_min_tokens"])
            if not tune.get("v2", 0):
                counts.add(getattr(self, "sms", 0) // (geometry.v_heads * (128 // tune["bv"])) + 1)
        tokens |= counts  # Retain the established nonempty-call representatives too.
        keys = {}
        for t in sorted(v for v in tokens if v <= max_tokens):
            for n in sorted(v for v in counts if v <= max_sequences):
                if select is not None and not select(t, n):
                    continue
                for captures in (0, 1):
                    try:
                        validate_captures(self._row_tune(t, n)[0], captures)
                        keys.setdefault(self.variant_key(geometry, t, n, stride, captures), (t, n))
                    except ValueError:
                        continue  # such calls raise before compiling
        return keys

    def warm_up(self, keys):
        """Compile ``keys`` not compiled yet, each with its capturing or non-capturing twin
        (:meth:`_compile_eagerly`); returns that count. Raises inside a capture, even when
        every key is already compiled."""
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CuTe prefill warm-up cannot run inside a CUDA graph capture")
        compiled = 0
        for key in keys:
            if key in self.compiled:
                continue
            before = len(self.compiled)
            self._compile_eagerly(key)
            compiled += len(self.compiled) - before
        return compiled

    def capture_variant(self, geometry, projected, count):
        """Warmup follows the exact compiled variant, including the wave class; the two
        executables of a variant (with and without captures) compile together."""
        return self.compile_key(geometry, projected, count)

    def _twin(self, key):
        """``key`` with the other capture choice, or None if its tune rejects captures."""
        twin = (*key[:-1], 1 - key[-1])
        try:
            validate_captures(dict(key[5]), twin[-1])
        except ValueError:
            return None
        return type(key)(twin) if isinstance(key, HashedTuple) else twin

    def __call__(
        self,
        geometry,
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        cu_seqlens,
        slot_indices,
        has_initial_state,
        state,
        conv_state,
        *,
        out=None,
        capture_rows=None,
    ):
        tokens, count = projected.shape[0], slot_indices.numel()
        hk, hv = geometry.k_heads, geometry.v_heads
        captures = int(capture_rows is not None)
        tune, _ = self._row_tune(tokens, count)
        validate_captures(tune, captures)
        if count == 0:
            return out if out is not None else projected.new_empty((0, hv * 128))
        key = self.compile_key(geometry, projected, count, captures)
        validate_launch(hk, hv, tokens, count, self._effective_tune(hv, key[3], key))
        sizes = dict(zip(SCRATCH, scratch_layout(geometry, tune, tokens, count, self), strict=True))

        def scratch(name):
            numel, dtype = sizes[name]
            return torch.empty(numel, dtype=dtype, device=projected.device)

        # Keep the tensor-only v1 ABI, but allocate no global prologue intermediates.
        kk, qk, q, k, w, u, kl, aq, qp, op = (
            scratch(name) for name in ("kk", "qk", "q", "k", "w", "u", "kl", "aq", "qp", "op")
        )
        self.last_debug = None
        if tune.get("v2_debug", 0):
            # Reuse the unused FP32 op argument so non-debug executable ABIs stay
            # unchanged. Zeroing is ordered on the caller stream, also on replay.
            self.last_debug = torch.empty(
                (count * hv * tune["vsplit"], 14, CHUNK, 128),
                dtype=torch.float32,
                device=projected.device,
            )
            self.last_debug.zero_()
            op = self.last_debug.view(-1)
        g = scratch("g")
        # Opt-in diagnostics retain the actual launch buffers, without copies or
        # extra device allocations. Normal calls keep their original lifetimes.
        if getattr(self, "retain_scratch", False):
            self.last_scratch = {"q": q, "k": k, "w": w, "u": u, "aq": aq, "g": g}
        history_out, raw_out, partial, hbuf, vbuf = (
            scratch(name) for name in ("history_out", "raw_out", "partial", "hbuf", "vbuf")
        )
        # The norm kernel stores 128-bit vectors: bind a caller output directly only when
        # its base address is 16-byte aligned; otherwise write a fresh buffer and copy.
        direct = out is not None and out.data_ptr() % 16 == 0
        y = (
            out
            if direct
            else torch.empty((tokens, hv * 128), dtype=torch.bfloat16, device=projected.device)
        )
        # Call-local workspace, zeroed on the caller stream before the PDL chain. A graph
        # records this reset on every replay, including unused ragged capacity.
        flags = scratch("flags")
        arguments = (
            projected,
            *(
                t.view(-1)
                for t in (
                    conv_weight,
                    A_log,
                    dt_bias,
                    norm_weight,
                    cu_seqlens,
                    slot_indices,
                    has_initial_state.view(torch.int8),
                    state,
                    conv_state,
                )
            ),
            q,
            k,
            w,
            u,
            aq,
            g,
            history_out,
            raw_out,
            partial,
            y.view(-1),
            kk,
            qk,
            kl,
            flags,
            qp,
            op,
            # Unused placeholders keep the zero-capture host ABI tensor-only.
            capture_offsets(cu_seqlens) if captures else cu_seqlens,
            capture_rows if captures else slot_indices,
            hbuf,
            vbuf,
        )
        # T and N affect runtime grids and scratch sizes; the selected row tune
        # and wave class determine the compiled schedule.
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = self._compile_eagerly(key)
        if isinstance(compiled, dict):
            main = torch.cuda.current_stream(self.device)
            if flags.numel():
                reset_flags(flags, main)
            compiled["pipeline"](*arguments)
            compiled["norm"](*arguments)
        else:
            compiled(*arguments)
        if out is not None and not direct:
            out.copy_(y)
            return out
        return y

    def _compile_eagerly(self, key):
        """Compile ``key`` and its twin (:meth:`_twin`): the capture guard records one entry
        for both, so a captured graph never compiles the other."""
        with _compiling, torch.cuda.device(self.device):
            # Defense in depth: the product guard normally rejects this before allocation.
            # Ask the bound device, not the current one.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CuTe prefill must run once eagerly before CUDA graph capture")
            for each in (key, self._twin(key)):
                if each is not None and each not in self.compiled:
                    self.compiled[each] = self._compile(each)
        return self.compiled[key]

    def plan(self, geometry, tokens, count, stride):
        """The shared host decisions of a call (fast path); raises what __call__ raises."""
        ident = (self._policy, tokens, count, stride)
        plan = _PLANS.get(ident)
        if plan is not None:
            return plan
        tune, _ = self._row_tune(tokens, count)
        if count == 0:
            plan = _Plan(next(_plan_ids), None, None, tune, True, (), 0, False)
        else:
            hk, hv = geometry.k_heads, geometry.v_heads
            key = HashedTuple(self.variant_key(geometry, tokens, count, stride))
            validate_launch(hk, hv, tokens, count, self._effective_tune(hv, key[3], key))
            layout, offset = [], 0
            for name, (numel, dtype) in zip(
                SCRATCH, scratch_layout(geometry, tune, tokens, count, self), strict=True
            ):
                layout.append((name, offset, numel, dtype))
                offset += -(-numel * dtype.itemsize // _ALIGN) * _ALIGN
            plan = _Plan(
                next(_plan_ids),
                key,
                self._twin(key),
                tune,
                False,
                tuple(layout),
                offset,
                bool(tune.get("v2_debug", 0)),
            )
        with _host_lock:
            if len(_PLANS) >= _PLAN_LIMIT:
                _PLANS.clear()
            return _PLANS.setdefault(ident, plan)

    def _weight_arguments(self, conv_weight, A_log, dt_bias, norm_weight):
        """TVM-FFI handles of the frozen weights, converted once per bound weight set.

        Holds the last weights called with (frozen model parameters) until others arrive.
        """
        cached = self._weights
        pointers = (
            conv_weight.data_ptr(),
            A_log.data_ptr(),
            dt_bias.data_ptr(),
            norm_weight.data_ptr(),
        )
        if (
            cached is not None
            and cached[0] is conv_weight
            and cached[1] is A_log
            and cached[2] is dt_bias
            and cached[3] is norm_weight
            and cached[4] == pointers
        ):
            return cached[5]
        import tvm_ffi

        converted = tuple(
            tvm_ffi.from_dlpack(t)
            for t in (conv_weight.view(-1), A_log.view(-1), dt_bias.view(-1), norm_weight.view(-1))
        )
        self._weights = (conv_weight, A_log, dt_bias, norm_weight, pointers, converted)
        return converted

    def run(
        self,
        plan,
        geometry,
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        cu_seqlens,
        slot_indices,
        has_initial_state,
        state,
        conv_state,
        out,
        capture_rows,
        capturing,
    ):
        """``__call__`` for checked inputs and a ``plan`` of this launcher, with the same
        launches and arguments. Eager calls use the stream's shared scratch (see _Arena)."""
        if capture_rows is not None and plan.capture_key is None and not plan.empty:
            validate_captures(plan.tune, 1)
        if plan.empty:
            return out if out is not None else projected.new_empty((0, geometry.value_width))
        if plan.legacy or getattr(self, "retain_scratch", False):
            return self(
                geometry,
                projected,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
                cu_seqlens,
                slot_indices,
                has_initial_state,
                state,
                conv_state,
                out=out,
                capture_rows=capture_rows,
            )
        key = plan.key if capture_rows is None else plan.capture_key
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = self._compile_eagerly(key)
        weights = self._weight_arguments(conv_weight, A_log, dt_bias, norm_weight)
        if capturing or plan.nbytes > ARENA_MAX_BYTES:
            # A graph keeps its own scratch (from its private pool), as do very large calls.
            lock, (scratch, flags) = contextlib.nullcontext(), _carve(None, plan, projected.device)
        else:
            lock, (scratch, flags) = _arena_views(self.device, plan)
        with lock:
            return self._enqueue(
                compiled,
                geometry,
                projected,
                weights,
                cu_seqlens,
                slot_indices,
                has_initial_state,
                state,
                conv_state,
                out,
                capture_rows,
                scratch,
                flags,
            )

    def _enqueue(
        self,
        compiled,
        geometry,
        projected,
        weights,
        cu_seqlens,
        slot_indices,
        has_initial_state,
        state,
        conv_state,
        out,
        capture_rows,
        scratch,
        flags,
    ):
        """Every launch (and the output copy) of one fast-path call, under its scratch lock."""
        self.last_debug = None
        tokens = projected.shape[0]
        direct = out is not None and out.data_ptr() % 16 == 0
        y = (
            out
            if direct
            else torch.empty(
                (tokens, geometry.value_width), dtype=torch.bfloat16, device=projected.device
            )
        )
        if capture_rows is not None:
            offsets, rows = capture_offsets(cu_seqlens), capture_rows
        else:
            # Never read without captures (compile-time): one shared zero-size argument.
            offsets = rows = _empty(projected.device, torch.int32)[1]
        arguments = (
            projected,
            *weights,
            cu_seqlens,
            slot_indices,
            has_initial_state.view(torch.int8),
            state.view(-1),
            conv_state.view(-1),
            *scratch[:9],
            y.view(-1),
            *scratch[9:15],
            offsets,
            rows,
            *scratch[15:],
        )
        if isinstance(compiled, dict):
            if flags.numel():
                reset_flags(flags, torch.cuda.current_stream(self.device))
            compiled["pipeline"](*arguments)
            compiled["norm"](*arguments)
        else:
            compiled(*arguments)
        if out is not None and not direct:
            out.copy_(y)
            return out
        return y

    def _compile(self, key):
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32, Int8, Int32, Int64
        from cutlass.runtime import make_fake_stream

        hk, hv, _, one_wave, stride, tune_key, captures = key
        tune = dict(tune_key)
        validate_captures(tune, captures)
        width = (2 * hk + 2 * hv) * 128 + 2 * hv
        fake = cute.runtime.make_fake_tensor

        def vector(dtype):
            # Only element alignment is assumed: strided/misaligned product inputs are legal.
            return fake(dtype, (cute.sym_int(),), (1,), assumed_align=dtype.width // 8)

        x = fake(BFloat16, (cute.sym_int(), width), (stride, 1), assumed_align=2)
        tk = tune["form"] == "tk"
        dtypes = (
            BFloat16,
            Float32,
            Float32,
            Float32,
            Int32,
            Int32,
            Int8,
            Float32,
            BFloat16,
            BFloat16,
            BFloat16,
            Float32 if tk and tune["t_store"] == "fp32" else BFloat16,
            Float32 if tune["v_store" if tk else "u_store"] == "fp32" else BFloat16,
            BFloat16,
            Float32,
            BFloat16,
            BFloat16 if tune["o_bf16"] else Float32,
            Float32,
            BFloat16,
            Float32,
            Float32,
            BFloat16,
            Int32,
            BFloat16,
            Float32,
        )
        args = (
            x,
            *(
                vector(dtype)
                for dtype in (*dtypes, Int64 if captures else Int32, Int32, BFloat16, Float32)
            ),
        )
        tune = self._effective_tune(hv, one_wave, key)

        def compile_phase(phase):
            if tune.get("v2", 0):
                from ._cute_prefill_v2_kernels import FusedPrefill

                kernel = FusedPrefill(hk, hv, stride, tune, captures)
            else:
                kernel = self.kernels.Prefill(hk, hv, stride, tune, phase, captures)
            options = self.options
            # Phase executables have identical signatures; avoid overwriting their
            # offline artifacts (CuTe truncates the generated filenames).
            if "--dump-dir=" in options:
                import re
                from pathlib import Path

                match = re.search(r"--dump-dir=(\S+)", options)
                directory = Path(match[1]) / phase
                directory.mkdir(parents=True, exist_ok=True)
                options = options.replace(match[0], f"--dump-dir={directory}")
            return cute.compile(
                kernel, *args, make_fake_stream(use_tvm_ffi_env_stream=True), options=options
            )

        if tune["overlap"]:
            return {phase: compile_phase(phase) for phase in ("pipeline", "norm")}
        return compile_phase("all")


def validate_launch(hk, hv, tokens, count, tune):
    """CUDA limits checked before allocation; folded grids preserve linear CTA order.

    At grid.x=2**31, recurrence alone needs at least 2**31 * 16 * 128 * 4
    = 16 TiB of state (BV>=16, distinct live slots). Prepare needs at least
    2**31 * 64 * 128 * 2 = 32 TiB per head/chunk scratch plane. These are
    unreachable for allocatable live batches on supported GPUs; host bounds
    still reject oversized shapes, including batches dominated by empty rows.
    Geometry does not bound Hv, so norm/commit retain the grid.y check.

    SGLang 72730d6 is narrower: chunk_fwd_o in
    sglang/kernels/ops/attention/fla/chunk_o.py uses (cdiv(V,BV), NT, B*H),
    putting total chunks NT on grid.y<=65535. chunk_gated_delta_rule_fwd_h
    in chunk_delta_h.py uses (cdiv(V,BV), N*H), limiting N*H on grid.y.
    """
    if hv > 65535:
        raise ValueError("CuTe prefill requires Hv <= 65535 (CUDA grid.y limit)")
    if tune.get("v2", 0):
        products = {"N*Hv*vsplit": count * hv * tune["vsplit"]}
    else:
        effective = resolve_tune(
            tune | {"overlap": 0} if pipeline_variant(tokens, tune) == "serial" else tune, hv
        )
        chunks = chunk_capacity(tokens, count)
        products = {
            "Hk*chunks": hk * chunks,
            "G*chunks": (hk if effective["prep_mode"] else hv) * chunks,
            "tiles*Hv*N": (128 // effective["bv"]) * hv * count,
        }
    for name, size in products.items():
        if size > 2**31 - 1:
            raise ValueError(f"CuTe prefill {name} exceeds CUDA grid.x limit (2**31-1)")


def launch_info(hk, hv, tokens, count, tune, one_wave=False):
    """Host launch geometry and allocated dynamic shared bytes; no device queries."""
    if count == 0:
        return []
    if tune.get("v2", 0):
        fused = {
            "kernel": "fused_v2",
            "grid": [count * hv * tune["vsplit"], 1, 1],
            "ctas": count * hv * tune["vsplit"],
            "threads": 256,
            "dynamic_smem_bytes": v2_smem_layout(hk, hv, tune["vsplit"])["bytes"],
        }
        if not tune.get("v2_snorm", 0) or tokens == 0:
            # Empty separate-norm calls also use commit, since their norm grid is empty.
            # Only the fused gated norm clusters V slices of one (sequence, head).
            fused["cluster"] = [1 if tune.get("v2_snorm", 0) else tune["vsplit"], 1, 1]
            commit = {
                "kernel": "commit",
                "grid": [count, hv, 1],
                "ctas": count * hv,
                "threads": 128,
                "dynamic_smem_bytes": 0,
            }
            return [fused, commit]
        return [
            fused,
            {
                "kernel": "norm",
                "grid": [(tokens + 7) // 8, hv, 1],
                "ctas": ((tokens + 7) // 8) * hv,
                "threads": 128,
                "dynamic_smem_bytes": 0,
            },
        ]
    chunks = chunk_capacity(tokens, count)
    stop = ("conv", "norm", "gram", "solve", "wy", "all").index(tune["stop_after"])
    tune = resolve_tune(tune, hv, one_wave)
    bv = tune["bv"]
    specs = []
    if not (tune["merge_prepare"] or tune["prep_mode"]) or stop < 3:
        specs.append(("prepare_key", (hk * chunks, 1, 1), 32 * tune["prepare_warps"], 33792))
    if stop >= 3:
        specs.append(
            (
                "prepare_value",
                ((hk if tune["prep_mode"] else hv) * chunks, 1, 1),
                32 * tune["prepare_warps"],
                42496,
            )
        )
    if stop == 5:
        smem = 32768 + 384 * bv + 512
        smem += 256 if tune["form"] == "tk" else 0
        if tune["rec_mode"] and tune["form"] == "wy":
            smem += 8192 + 128 * bv
        if tune["rec_mode"] == 2:
            smem = (81920 if tune["prefetch_depth"] == 4 else 32768) + 384 * bv + 256
        if tune["o_form"] == "pre" and tune["prefetch_depth"] == 4:
            smem += 512 * bv - 16384
        if tune["o_split"] and tune["prefetch_depth"] == 4:
            smem -= 32768  # Q and both A banks leave the recurrence.
        if tune["rec_ws"]:
            smem = 57344 + 768 * bv + 256
        specs.append(
            ("recurrence", ((128 // bv) * hv * count, 1, 1), 32 * tune["recurrence_warps"], smem)
        )
        if tune["o_split"]:
            # Q, H, V_hi, A_hi/lo, g; Q/H storage is reused for FP32 staging/output.
            output_smem = 24576 + 384 * tune["bvo"] + 256
            specs.append(
                ("output", (chunks, hv, 128 // tune["bvo"]), 32 * tune["output_warps"], output_smem)
            )
        specs.append(("norm", ((tokens + 7) // 8, hv, 1), 128, 0))

    return [
        {
            "kernel": name,
            "grid": list(grid),
            "ctas": grid[0] * grid[1] * grid[2],
            "threads": threads,
            "dynamic_smem_bytes": smem,
        }
        for name, grid, threads, smem in specs
        if all(grid)
    ]


def compile_info(
    directory, hk, hv, tokens, count, tune, reserved_shared_per_cta=1024, *, one_wave=False
):
    """Read compiler resources from dumped cubins, never disassemble instructions.

    Used only by offline compile/probe scripts, outside capture and timing. Dynamic
    bytes are the CuTe allocator footprint; cuobjdump supplies static bytes, registers
    and spills. Offline SM12x examples additionally reserve 1 KiB/CTA for the
    driver; runtime queries the actual device reservation. An unavailable resource
    tool is an error rather than an invented count.
    """
    import re
    import shutil
    import subprocess
    from pathlib import Path

    executable = shutil.which("cuobjdump")
    if executable is None:
        candidate = Path("/usr/local/cuda/bin/cuobjdump")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise RuntimeError("CuTe resource reporting requires cuobjdump on PATH")
    records = {item["kernel"]: item for item in launch_info(hk, hv, tokens, count, tune, one_wave)}
    for binary in Path(directory).rglob("*.cubin"):
        result = subprocess.run(
            [executable, "--dump-resource-usage", str(binary)],
            capture_output=True,
            text=True,
            check=True,
        )
        for symbol, fields in re.findall(r"Function\s+(\S+):\s*\n([^\n]+)", result.stdout):
            for name, record in records.items():
                if f"_cutlass_{name}_" in symbol:
                    values = {key: int(value) for key, value in re.findall(r"(\w+):(\d+)", fields)}
                    record.update(
                        registers=values["REG"],
                        static_smem_bytes=values["SHARED"],
                        reserved_smem_bytes=reserved_shared_per_cta,
                        smem_bytes=record["dynamic_smem_bytes"]
                        + values["SHARED"]
                        + reserved_shared_per_cta,
                        stack_bytes=values["STACK"],
                        local_bytes=values["LOCAL"],
                    )
    if any("registers" not in record for record in records.values()):
        raise RuntimeError("missing CuTe cubin resource data; compile with --keep-cubin --dump-dir")
    for record in records.values():
        record.update(resource_limits(record))
    return list(records.values())


def resource_limits(record):
    """Offline capacity estimate for SM12x at the 100 KiB shared carveout.

    Uses the same conservative allocation rounding as admission. Cubin static
    bytes and the separate driver reservation are included in smem_bytes.
    This is capacity, not a runtime occupancy query or a placement guarantee.
    """
    registers, shared, threads, _ = _footprint(record)
    register_limit = 16384 // registers if registers else 24
    shared_limit = 102400 // shared if shared else 24
    return {
        "register_ctas_per_sm": register_limit,
        "shared_ctas_per_sm_100k": shared_limit,
        "ctas_per_sm_100k": min(24, 1536 // threads, register_limit, shared_limit),
    }


def v2_smem_layout(hk, hv, vsplit):
    """Serialized phase-1 arena, byte offsets; no CUDA/DSL import.

    Q/K occupy [0,32768); A occupies [32768,40960) after QK.
    Low Q/K use [32768,65536) only through QK. G/J retire into L/A.
    Solve: L@40960, S@57344, workspace@73728 (8192 bytes).
    WY: raw V strips -> E -> W @40960; B_hi/B_lo -> U @57344.
    S_hi sits above B (at least 73728); g/beta occupy the final 512 bytes.
    E reuses beta for 64 row factors after U. After the chain retires Q,
    norm uses Q's first 2048 bytes for [warp,token] sums; output reuses U.
    The diagonal MMA solve uses the existing workspace (no extra buffers).
    """
    Geometry(hk, hv)
    if vsplit not in (1, 2, 4):
        raise ValueError("vsplit must be 1, 2 or 4")
    low = hv // hk == 3
    sh = max(73728, 57344 + 32768 // vsplit)
    steps = {
        "conv": 32768 + 32768 * low + 17408,
        "gram_qk": 65536 + 32768 * low,
        "mask": 57344,
        "solve": 81920,
        "wy": sh + 8192,
        "chain": 57344 + 32768 // vsplit,
        "output": 57344 + 32768 // vsplit,
    }
    arena = max(steps.values())
    return {
        "bytes": arena + 512,
        "arena": arena,
        "s_hi": sh,
        "norm_partials": 0,
        "steps": {name: size + 512 for name, size in steps.items()},
    }
