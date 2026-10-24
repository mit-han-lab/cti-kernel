# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Explicit warm-up decisions for table holes; global contexts serialize tuning.

By default the cache is consulted only inside an ``autotune(...)`` context, so a
cache file left on a node never changes routes. Contexts nest and restore state. A context is visible to all threads; a global
lock serializes keys and cache access, and a separate lock serializes contexts.
No CUDA inspection, version lookup or file access happens at import time.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, astuple, dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from . import _dispatch
from ._dispatch import (
    M_BUCKETS,
    TUNED,
    TUNED_GATED,
    Config,
    bootstrap_median_ci,
    bxq_configs,
    device_class,
    m_bucket,
    point_qualifies,
    qualification_pairs,
)

_LOCK = threading.RLock()
_CONTEXT_LOCK = threading.RLock()
_CONTEXT = None
ACTIVE = False  # The only extra check on a cached cuBLAS call.
_CACHES = {}
_WARNED = set()
# Cache file format; formats 1-4 predate per-point pair counts, the strict
# shared predicate, the full identity key or GB10 CI evidence and are ignored (retuned).
FORMAT = 5
REGIMES = ("cold", "engine", "hot")
# Qualification uses at least qualification_pairs(kind) interleaved pairs
# (QUALIFY_PAIRS, 60 on GB10 with a bootstrap CI); screening uses fewer.
SCREEN_PAIRS = 5


def mode(value):
    """'context' (default): cache only inside autotune(); 'cache': always; '0': never."""
    value = os.environ.get("CTI_LINEAR_AUTOTUNE", "context") if value is None else value
    if value not in ("0", "context", "cache"):
        raise ValueError("autotune must be '0', 'context' or 'cache'")
    return value


@lru_cache(maxsize=1)
def _driver_version():
    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
        if library.nvmlInit_v2() == 0:
            try:
                version = ctypes.create_string_buffer(80)
                if library.nvmlSystemGetDriverVersion(version, len(version)) == 0:
                    return version.value.decode("ascii")
            finally:
                library.nvmlShutdown()
    except (OSError, AttributeError, UnicodeError):
        pass
    try:
        library = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(version)) == 0:
            return f"cuda{version.value}"
    except (OSError, AttributeError):
        return "unknown"
    return "unknown"


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _loaded_cublaslt():
    """Paths of the cuBLASLt libraries mapped into this process (torch links them)."""
    try:
        with open("/proc/self/maps") as maps:
            return sorted({line.split()[-1] for line in maps if "/libcublasLt.so" in line})
    except OSError:
        return []


@lru_cache(maxsize=1)
def _cublas_version():
    """cublasLtGetVersion of the cuBLASLt torch loaded, else of torch's CUDA major
    soname, else that major's wheel metadata."""
    major = (torch.version.cuda or "").split(".")[0]
    versions = []
    for name in _loaded_cublaslt() or [f"libcublasLt.so.{major}"]:
        try:
            library = ctypes.CDLL(name)
            library.cublasLtGetVersion.restype = ctypes.c_size_t
            versions.append(str(library.cublasLtGetVersion()))
        except (OSError, AttributeError):
            continue
    if versions:
        return ",".join(versions)
    return _version(f"nvidia-cublas-cu{major}") if major else "unknown"


@lru_cache(maxsize=1)
def _fingerprint():
    """Kernel-variant fingerprint: hash of the CuTe kernel sources and Config definitions."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*(root / "cute").glob("*.py"), root / "_dispatch.py"]):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=None)
def _device_name(device):
    return torch.cuda.get_device_properties(device).name


def identity(device):
    """Hardware identity of a key: (major, minor, SMs, device name)."""
    _, capability, sms = _dispatch._hardware(device)
    return (*capability, sms, _device_name(device))


def _kind(hardware):
    return device_class(tuple(hardware[:2]), hardware[2])


def _prefix(hardware):
    return [
        list(hardware),
        _driver_version(),
        torch.__version__,
        torch.version.cuda,
        _cublas_version(),
        _version("nvidia-cutlass-dsl"),
        _version("cti-kernels"),
        _fingerprint(),
    ]


def cache_key(hardware, n, k, gate_stride, bucket):
    return json.dumps([*_prefix(hardware), n, k, gate_stride, list(bucket)], separators=(",", ":"))


def _path():
    return _context_path(_CONTEXT)


def _context_path(context):
    if context is not None and context.cache is not None:
        return Path(context.cache).expanduser()
    setting = os.environ.get("CTI_LINEAR_AUTOTUNE_CACHE")
    if setting:
        return Path(setting).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return root / "ctikernels" / "linear-autotune.json"


def _warn(path, message="unreadable or invalid; using empty cache"):
    if (path, message) not in _WARNED:
        _WARNED.add((path, message))
        warnings.warn(f"Linear autotune cache {message}", stacklevel=3)


def _finite(value):
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return False


def _valid(key, record):
    """Persisted record check on load: finite, positive timings, complete CuTe evidence."""
    try:
        parsed = json.loads(key)
        bucket = tuple(parsed[-1])
        kind = _kind(parsed[0])
        points = [p for regime in record["regimes"].values() for p in regime.values()]
        return (
            bucket in M_BUCKETS
            and record["provider"] in ("cute", "cublas")
            and _finite(record)
            and all(type(m) is int and m_bucket(m) is not None for m in record["measured_m"])
            and (record["config"] is None or isinstance(record["config"], dict))
            and bool(points)
            and all(
                type(p["pairs"]) is int
                and p["pairs"] > 0
                and p["speedup"] > 0
                and p["epsilon"] >= 0
                and p["ours_us"] > 0
                and p["cublas_us"] > 0
                for p in points
            )
            and (record["provider"] != "cute" or _valid_cute(record, bucket, kind))
        )
    except (AttributeError, LookupError, OverflowError, TypeError, ValueError):
        return False


def _read(path):
    try:
        data = json.loads(path.read_text())
        if data.get("format") in (1, 2, 3, 4):
            return {}  # Older evidence or identity keys: retune them.
        if data.get("format") != FORMAT or not isinstance(data.get("entries"), dict):
            raise ValueError("invalid cache format")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, AttributeError, TypeError, RecursionError):
        _warn(path)
        return {}
    # Drop corrupt, partial or no-longer-qualifying records; a later warm-up retunes them.
    entries = {key: record for key, record in data["entries"].items() if _valid(key, record)}
    if len(entries) != len(data["entries"]):
        _warn(path, "has invalid records; dropping them")
    return entries


def _entries(path=None):
    path = _path() if path is None else path
    if path not in _CACHES:
        _CACHES[path] = _read(path)
    return _CACHES[path]


def lookup(hardware, n, k, gate_stride, bucket, *, in_context=False):
    """Cached record; with in_context, None outside an autotune() context. The context
    check and the read of its cache happen under one lock, so an exiting context
    never falls through to the node-default cache."""
    with _LOCK:
        context = _CONTEXT
        if in_context and context is None:
            return None
        entries = _entries(_context_path(context))
        return deepcopy(entries.get(cache_key(hardware, n, k, gate_stride, bucket)))


def _valid_cute(record, bucket, kind):
    """A CuTe record is reusable only with complete evidence that passes the shared
    per-device predicate now: cold, engine and hot points at both bucket edges and
    at every measured M, every one passing ``point_qualifies`` (>= 20 pairs; on
    GB10 >= 60 pairs and bootstrap ci_low > 1)."""
    try:
        measured = {str(m) for m in (*bucket, *record["measured_m"])}
        regimes = record["regimes"]
        points = [p for regime in regimes.values() for p in regime.values()]
        return (
            record["correct"] is True
            and record["deterministic"] is True
            and set(bucket) <= set(record["measured_m"])
            and all(measured <= set(regimes[regime]) for regime in REGIMES)
            and all(point_qualifies(kind, p) for p in points)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def record_config(record, bucket, kind):
    """Only CuTe decisions with complete passing evidence (``_valid_cute``) dispatch."""
    if not record or record.get("provider") != "cute" or not _valid_cute(record, bucket, kind):
        return None
    try:
        return Config(**record["config"])
    except (KeyError, TypeError, ValueError):
        return None


def _write(key, record):
    """Re-read under flock; first writer wins, including a cuBLAS decision."""
    if _CONTEXT is None or torch.cuda.is_current_stream_capturing():
        return record
    path = _path()
    try:
        import fcntl
    except ImportError:
        fcntl = None
    temp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(path) + ".lock").open("a") as lock:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX)
            entries = _read(path)
            entries.setdefault(key, record)
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                temp = stream.name
                json.dump(
                    {"format": FORMAT, "entries": entries}, stream, sort_keys=True, allow_nan=False
                )
                stream.write("\n")
            os.replace(temp, path)
            temp = None
            _CACHES[path] = entries
            return entries[key]
    except (OSError, ValueError):
        warnings.warn(
            "Linear autotune cache write failed; keeping decision in memory", stacklevel=3
        )
        return _entries().setdefault(key, record)
    finally:
        if temp is not None:
            Path(temp).unlink(missing_ok=True)


@dataclass
class _Context:
    ms: tuple | None
    deadline: float
    top_k: int
    cache: str | None
    pairs: int
    devices: set = field(default_factory=set)  # Devices this context (or a nested one) tuned on.


def _release():
    """Return tuning's own memory after the outermost context that tuned exits.

    Weight copies, inputs, references and timing graphs die with their frames;
    empty_cache() hands their blocks and the graphs' pools back (live graph pools are
    untouched). cuBLAS workspaces stay: see ``release_cublas_workspaces``.
    """
    torch.cuda.empty_cache()


@contextmanager
def autotune(ms=None, budget_seconds=60.0, top_k=3, cache=None, pairs=20):
    """Tune table holes during eager warm-up; budget includes compile and timing.

    Global across threads, with serialized tuning and nested context restoration.
    ``cache`` overrides the cache path for this context. ``pairs`` sets the
    qualification pairs, raised to at least QUALIFY_PAIRS (20; 60 on GB10, which
    also needs bootstrap ci_low > 1); lower values only
    shorten the candidate screen (at most SCREEN_PAIRS). In-flight operations
    finish, but expired keys are discarded and remain uncached cuBLAS fallbacks
    that a later context can retune. The outermost exit, normal or by exception,
    returns the memory tuning used (``_release``); a cleanup failure only warns.
    """
    global _CONTEXT, ACTIVE
    if not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError("budget_seconds must be finite and nonnegative")
    if type(top_k) is not int or top_k < 1 or type(pairs) is not int or pairs < 1:
        raise ValueError("top_k and pairs must be positive integers")
    if ms is not None:
        ms = tuple(sorted(set(ms)))
        if any(type(m) is not int or m_bucket(m) is None for m in ms):
            raise ValueError("ms must contain integers in 1..64")
    with _CONTEXT_LOCK:
        with _LOCK:
            previous = _CONTEXT
            _CONTEXT = current = _Context(
                ms, time.monotonic() + budget_seconds, top_k, cache, pairs
            )
            ACTIVE = True
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            with _LOCK:
                _CONTEXT = previous
                ACTIVE = previous is not None
                if previous is not None:
                    previous.devices |= current.devices  # The outermost exit releases.
            if previous is None and current.devices:
                try:
                    _release()
                except Exception as exc:  # noqa: BLE001 -- never mask the body's outcome
                    try:
                        warnings.warn(f"Linear autotune cleanup failed: {exc!r}", stacklevel=3)
                    except Warning:
                        if not body_failed:  # Warnings-as-errors never replace the body's error.
                            raise


def autotune_entries(device=None):
    """Copy of loaded entries matching this device's hardware and software versions."""
    if not torch.cuda.is_available():
        return {}
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        return {}
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    prefix = _prefix(identity(device))
    with _LOCK:
        result = {}
        for key, record in _entries().items():
            try:
                if json.loads(key)[: len(prefix)] == prefix:
                    result[key] = deepcopy(record)
            except (ValueError, TypeError):
                continue
        return result


def rank_candidates(kind, sms, n, k, edge, gate_stride, top_k):
    """Table neighbourhood first, then the heuristic grid by roofline time.

    Neighbours are this device's measured table configs: the same (N, K) first,
    then shapes sharing N or K: those within 4x weight size before the rest,
    each ordered by M-bucket distance and then by weight-size ratio. They fill top_k - 1 slots (the single slot when top_k is
    1) if they keep at least a quarter of the target warps busy; the remaining slots
    take the heuristic grid by roofline time in us: W / occupied bandwidth +
    split traffic / bandwidth + 2 us per split-K reduction. Active warps =
    ceil(N/rows) * splits * warps, targeting 4/SM on sm120 and 24/SM on GB10;
    bandwidth is 1661/1543/199 GB/s (unknown: 1500). Config field order breaks ties.
    """
    from .cute import mtiles_for

    if sms <= 0 or top_k < 1 or m_bucket(edge) is None:
        raise ValueError("positive SMs/top_k and a supported bucket required")
    bandwidth = {"rtx5090": 1661, "pro6000": 1543, "gb10": 199}.get(kind, 1500) * 1000
    target = (24 if kind == "gb10" else 4) * sms
    index = {bucket: i for i, bucket in enumerate(M_BUCKETS)}
    here = index[m_bucket(edge)]

    def fits(cfg):
        tiles = math.ceil(k / cfg.tile_k)
        if (cfg.splits - 1) * math.ceil(tiles / cfg.splits) >= tiles:
            return False
        return cfg.shared_bytes_for(mtiles_for(edge), bool(gate_stride)) <= 99 * 1024

    def occupancy(cfg):
        return math.ceil(n / cfg.rows) * cfg.splits * cfg.warps / target

    def score(cfg):
        traffic = 2 * cfg.splits * edge * n * 4 if cfg.scratch_splits else 0
        time = 2 * n * k / (bandwidth * min(1, occupancy(cfg)))
        return time + traffic / bandwidth + (2 if cfg.kernels > 1 else 0)

    near = []
    for shape, ranges in (TUNED_GATED if gate_stride else TUNED).get(kind, {}).items():
        if shape[0] != n and shape[1] != k:
            continue
        size = abs(math.log2(shape[0] * shape[1] / (n * k)))
        for (lo, hi), cfg in ranges.items():
            first, last = index[m_bucket(lo)], index[m_bucket(hi)]
            gap = 0 if first <= here <= last else min(abs(here - first), abs(here - last))
            near.append(((shape[:2] != (n, k), size > 2, gap, size, astuple(cfg)), cfg))
    chosen = []
    for _, cfg in sorted(near):
        if len(chosen) >= max(1, top_k - 1):
            break
        if cfg not in chosen and fits(cfg) and occupancy(cfg) >= 0.25:
            chosen.append(cfg)
    grid = sorted(
        (score(cfg), astuple(cfg), cfg)
        for cfg in dict.fromkeys(bxq_configs(n, k, kind))
        if cfg not in chosen and fits(cfg)
    )
    return (chosen + [row[-1] for row in grid])[:top_k]


def _record(bucket):
    from .cute import mtiles_for

    return {
        "provider": "cublas",
        "config": None,
        "mtiles": mtiles_for(bucket[1]),
        "decided_by": "autotune",
        "reason": "no candidates",
        "min_speedup": None,
        "epsilon": None,
        "regimes": {},
        "candidates": [],
        "measured_m": [],
        "correct": False,
        "deterministic": False,
        "seconds": 0.0,
        "created": time.time(),
    }


def tune(weight, hardware, kind, n, k, gate_stride, bucket):
    """Tune one missing key, or skip without recording when capture/budget forbids it."""
    with _LOCK:
        if _CONTEXT is None or torch.cuda.is_current_stream_capturing():
            return None
        key = cache_key(hardware, n, k, gate_stride, bucket)
        if key in _entries():
            return deepcopy(_entries()[key])
        context = _CONTEXT
        members = (
            [] if context.ms is None else [m for m in context.ms if bucket[0] <= m <= bucket[1]]
        )
        if context.ms is not None and not members:
            return None
        if time.monotonic() >= context.deadline:
            return None
        record = _record(bucket)
        started = time.monotonic()
        try:
            configs = rank_candidates(
                kind, hardware[2], n, k, bucket[1], gate_stride, context.top_k
            )
            with torch.inference_mode():
                _measure(
                    weight,
                    hardware,
                    gate_stride,
                    bucket,
                    sorted({*bucket, *members}),
                    configs,
                    context.pairs,
                    record,
                )
        except _BudgetExpired:
            return None
        except Exception as exc:  # noqa: BLE001 -- failed warm-up must leave a cuBLAS decision
            record.update(provider="cublas", reason=f"error: {type(exc).__name__}")
        if time.monotonic() >= context.deadline:
            return None
        record["seconds"] = time.monotonic() - started
        points = [p for regime in record["regimes"].values() for p in regime.values()]
        if not points or any(p.get("pairs", 0) < qualification_pairs(kind) for p in points):
            # No 20-pair evidence (errors, failed correctness, no candidates):
            # cuBLAS for this process only; a later process retunes the key.
            return deepcopy(_entries().setdefault(key, record))
        return deepcopy(_write(key, record))


def _reference_and_condition(x, w):
    """FP64 result and sum(abs(x*w)); chunk W to bound vocabulary-head memory."""
    xd = x.double()
    values, conditions = [], []
    for part in w.split(1024):
        wd = part.double()
        values.append(xd @ wd.t())
        conditions.append(xd.abs() @ wd.abs().t())
    return torch.cat(values, dim=1), torch.cat(conditions, dim=1)


def _accuracy(y, baseline, ref, condition, k):
    """Compare BF16 output to cuBLAS using three independent error gates.

    Allow an extra half BF16 ULP beyond max(cuBLAS, half ULP), with a
    condition-scale floor for cancellation. Also bound condition-normalized
    error and allow 5% RMS slack. No additive RMS tolerance: an exact baseline
    must stay exact.
    """
    if k <= 0:
        raise ValueError("reduction length K must be positive")
    err, base = (y.double() - ref).abs(), (baseline.double() - ref).abs()
    denom = condition.clamp_min(torch.finfo(torch.float64).tiny)
    absolute, base_absolute = err.max().item(), base.max().item()
    normalized, base_normalized = (err / denom).max().item(), (base / denom).max().item()
    rms, base_rms = err.square().mean().sqrt().item(), base.square().mean().sqrt().item()
    fp32_slack = 2 * torch.finfo(torch.float32).eps
    # Random-walk accumulation model: independent rounding perturbations grow
    # as sqrt(K). Anchor the condition-scale budget at K=1024 and never shrink
    # it below two eps32. This is a statistical allowance, not a worst-case
    # forward-error bound; normalized error and RMS remain independent gates.
    accumulation_floor = fp32_slack * max(1.0, math.sqrt(k / 1024))
    ulp = torch.exp2(
        torch.floor(torch.log2(ref.abs().clamp_min(torch.finfo(torch.float64).tiny))) - 7
    )
    ulp = torch.maximum(ulp, denom * accumulation_floor)
    ulp_error, base_ulp_error = (err / ulp).max().item(), (base / ulp).max().item()
    ulp_limit = max(base_ulp_error, 0.5) + 0.5
    # A quarter of the upper bound on a BF16 ULP is 2^-9 * |ref|.
    # Normalize that allowance by sum|x*w| and retain two FP32 epsilons for
    # cancellation. Since |ref| <= sum|x*w|, the additive allowance is at most
    # 2^-9 + two epsilons, far below an O(1) missing-chunk/fragment error.
    normalized_slack = 2**-9 * (ref.abs() / denom).max().item() + fp32_slack
    normalized_limit = base_normalized * 1.5 + normalized_slack
    rms_limit = base_rms * 1.05
    ok = bool(torch.isfinite(y).all())
    ok = ok and ulp_error <= ulp_limit and rms <= rms_limit
    ok = ok and normalized <= normalized_limit
    return {
        "correct": ok,
        "ulp_condition_floor": accumulation_floor,
        "max_abs": absolute,
        "cublas_max_abs": base_absolute,
        "max_ulp_error": ulp_error,
        "cublas_max_ulp_error": base_ulp_error,
        "max_ulp_error_limit": ulp_limit,
        "max_normalized": normalized,
        "cublas_max_normalized": base_normalized,
        "max_normalized_limit": normalized_limit,
        "rms": rms,
        "cublas_rms": base_rms,
        "rms_limit": rms_limit,
    }


def _rotation(weight, regime):
    """Bound clone storage by L2; repeat whole patterns for long event samples."""
    n, k = weight.shape
    weight_bytes = n * k * 2
    copies = 0
    if regime == "hot":
        weights = [weight]
    else:
        l2 = torch.cuda.get_device_properties(weight.device).L2_cache_size
        if l2 <= 0:
            raise ValueError("L2 cache size unavailable")
        # At most 64 clones: tiny weights are launch-bound, not DRAM-bound.
        copies = 1 if weight_bytes >= 2 * l2 else min(64, max(2, math.ceil(3 * l2 / weight_bytes)))
        weights = [weight.clone() for _ in range(copies)]
        if regime == "engine":
            weights = [w for cold in weights for w in (cold, weight)]
    pattern = len(weights)
    # Keep complete rotations at the cap; one full pattern is the minimum.
    repeats = max(1, min(256 // pattern, math.ceil((512 << 20) / (pattern * weight_bytes))))
    return weights, copies, pattern * repeats


def _timed(graph, events, calls):
    start, end = events
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    value = start.elapsed_time(end) * 1000 / calls
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid event duration")
    return value


def _paired(left, right, pairs, calls):
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = ([], [])
    for pair in range(pairs):
        for side in (0, 1) if pair % 2 else (1, 0):
            _check_deadline()  # The budget bounds every timed launch.
            samples[side].append(_timed((left, right)[side], events, calls))
    return samples


def _p95(values):
    values = sorted(values)
    position = 0.95 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _graph(fn, weights, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(weights[0])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            output = fn(weights[i % len(weights)])
    return graph, output


def _benchmark(ours, baseline, regime, pairs, weight, *, calibrate=True):
    flags = torch.backends.cuda.matmul
    previous = flags.allow_bf16_reduced_precision_reduction
    try:
        flags.allow_bf16_reduced_precision_reduction = True
        return _benchmark_default(ours, baseline, regime, pairs, weight, calibrate)
    finally:
        flags.allow_bf16_reduced_precision_reduction = previous


def _benchmark_default(ours, baseline, regime, pairs, weight, calibrate):
    weights, copies, calls = _rotation(weight, regime)
    # All CuTe captures precede cuBLAS: a later capture clears cuBLAS workspaces.
    ours_graph, ours_output = _graph(ours, weights, calls)
    baseline_graph, baseline_output = _graph(baseline, weights, calls)
    a, b = _paired(ours_graph, baseline_graph, pairs, calls)
    ratios = [y / x for x, y in zip(a, b, strict=True)]
    ci_low, ci_high = bootstrap_median_ci(ratios, seed=f"{regime}:{pairs}:{calls}")
    epsilon = 0.0
    if calibrate:
        left, right = _paired(baseline_graph, baseline_graph, 20, calls)
        epsilon = _p95([abs(x / y - 1) for x, y in zip(left, right, strict=True)])
    return {
        "cublas_bf16_reduced_precision": True,
        "copies": copies,
        "calls_per_graph": calls,
        "speedup": statistics.median(ratios),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "epsilon": epsilon,
        "ours_us": statistics.median(a),
        "cublas_us": statistics.median(b),
    }


class _Trial:
    """One seeded input/reference; caller weight is borrowed read-only."""

    def __init__(self, weight, gate_stride, hi):
        self.weight = weight
        self.n, self.k = weight.shape
        gen = torch.Generator(device=weight.device).manual_seed(0)
        self.x = torch.randn(hi, self.k, device=weight.device, dtype=torch.bfloat16, generator=gen)
        self.gate = None
        self.a = self.x
        if gate_stride:
            self.gate = torch.randn(
                hi, gate_stride, device=weight.device, dtype=torch.bfloat16, generator=gen
            )[:, : self.k]
            self.a = (self.x.float() * self.gate.float().sigmoid()).bfloat16()
        self.ref, self.condition = _reference_and_condition(self.a, weight)

    def correct(self, output, m):
        flags = torch.backends.cuda.matmul
        previous = flags.allow_bf16_reduced_precision_reduction
        try:
            flags.allow_bf16_reduced_precision_reduction = False
            baseline = F.linear(self.a[:m], self.weight)
        finally:
            flags.allow_bf16_reduced_precision_reduction = previous
        return _accuracy(output, baseline, self.ref[:m], self.condition[:m], self.k)["correct"]

    def call(self, compiled, config, m):
        x, gate = self.x[:m], self.gate[:m] if self.gate is not None else None
        output = torch.empty((m, self.n), device=x.device, dtype=x.dtype)
        shape = (config.splits, m, self.n) if config.scratch_splits else (1, 1, 1)
        partial = torch.empty(shape, device=x.device, dtype=torch.float32)

        def run(weight=None):
            args = (x, self.weight if weight is None else weight, output, partial)
            if gate is not None:
                compiled(*args, gate)
            else:
                compiled(*args)
            return output

        return run

    def benchmark(self, call, m, regime, pairs, *, calibrate=True):
        # Gate cost is FREE to the comparator, conservatively beating any gate kernel.
        return _benchmark(
            call,
            lambda weight: F.linear(self.x[:m], weight),
            regime,
            pairs,
            self.weight,
            calibrate=calibrate,
        )


def _qualify(record, kind):
    points = [point for regime in record["regimes"].values() for point in regime.values()]
    record["min_speedup"] = min(point["speedup"] for point in points)
    record["epsilon"] = max(point["epsilon"] for point in points)
    if not record["correct"]:
        record["reason"] = "incorrect"
    elif not record["deterministic"]:
        record["reason"] = "not deterministic"
    elif any(p.get("pairs", 0) < qualification_pairs(kind) for p in points):
        record["reason"] = f"fewer than {qualification_pairs(kind)} pairs"
    elif not all(point_qualifies(kind, p) for p in points):
        record["reason"] = "below margin or noise"
    else:
        record.update(provider="cute", reason="qualified")


class _BudgetExpired(Exception):
    pass


def _check_deadline():
    if _CONTEXT is not None and time.monotonic() >= _CONTEXT.deadline:
        raise _BudgetExpired


def _measure(weight, hardware, gate_stride, bucket, ms, configs, pairs, record):
    """CUDA boundary: compile, screen, then qualify one winner in all regimes."""
    from .cute import get_compiled, mtiles_for

    record["cublas_bf16_reduced_precision"] = True
    kind = _kind(hardware)
    count = max(qualification_pairs(kind), pairs)
    _check_deadline()
    if _CONTEXT is not None:  # GPU tuning starts here; the exiting context releases.
        _CONTEXT.devices.add(weight.device)
    trial = _Trial(weight, gate_stride, bucket[1])
    best = None
    for config in configs:
        _check_deadline()
        candidate = {"config": asdict(config)}
        record["candidates"].append(candidate)
        try:
            compiled = get_compiled(
                weight.device,
                trial.n,
                trial.k,
                config,
                hardware[:2],
                False,
                mtiles_for(bucket[1]),
                gate_stride,
            )
            if compiled is None:
                raise RuntimeError("candidate unavailable")
        except Exception as exc:  # noqa: BLE001 -- skip a failed compiler candidate
            candidate["error"] = type(exc).__name__
            record["reason"] = f"error: {type(exc).__name__}"
            continue
        _check_deadline()
        try:
            call = trial.call(compiled, config, bucket[1])
            if not trial.correct(call(), bucket[1]):
                candidate["error"] = "incorrect"
                record["reason"] = "incorrect"
                continue
            _check_deadline()
            screen = trial.benchmark(
                call, bucket[1], "cold", min(SCREEN_PAIRS, pairs), calibrate=False
            )["speedup"]
            candidate["screen_speedup"] = screen
        except Exception as exc:
            candidate["error"] = type(exc).__name__
            raise
        if best is None or screen > best[0]:
            best = screen, config, compiled
    _check_deadline()
    if best is None:
        return
    screen, config, compiled = best
    record.update(config=asdict(config), reason="screen below parity")
    if screen < 1.0:
        # A cuBLAS decision is persisted only on full-protocol evidence: confirm
        # the screen loss with the qualification pair count at the upper edge.
        _check_deadline()
        point = trial.benchmark(trial.call(compiled, config, bucket[1]), bucket[1], "cold", count)
        point["pairs"] = count
        record["regimes"]["cold"] = {str(bucket[1]): point}
        if point["speedup"] < 1.0:
            return
    record.update(correct=True, deterministic=True)
    hi_call = trial.call(compiled, config, bucket[1])
    hi = hi_call().clone()
    for m in ms:
        _check_deadline()
        call = trial.call(compiled, config, m)
        output = call().clone()
        record["correct"] &= trial.correct(output, m)
        record["deterministic"] &= torch.equal(output.view(torch.int16), call().view(torch.int16))
        record["deterministic"] &= torch.equal(output.view(torch.int16), hi[:m].view(torch.int16))
        for regime in REGIMES:
            _check_deadline()
            point = trial.benchmark(call, m, regime, count)
            point["pairs"] = count
            record["regimes"].setdefault(regime, {})[str(m)] = point
        record["measured_m"].append(m)
    _check_deadline()
    _qualify(record, kind)
