# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Offline sweep of one TunedOp on the current card: search, keep rule, sensitivity.

Per bucket, a coordinate search from the op's default config finds the fastest config
(each candidate checked against the default's output and for bitwise repeatability).
Buckets that differ only along the op's inheritance axis are then walked in ascending
order: a bucket keeps its own row only when its best config beats the config it would
otherwise inherit (the last kept row, or the default) by at least ``KEEP`` in a paired
A-B-A on this card. Finally each bucket times its adjacent neighbours' best configs
(screen timing), which shows which dimensions move the best config and by how much.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

KEEP = 1.03  # minimum speedup over the inherited config for a bucket's own row
PASSES = 2
MOVE = 0.01  # a search step must gain this much, so ties and unused params stay at default


@dataclass
class Case:
    """One bucket on one card: ``run(config)`` returns a no-argument launcher whose call
    writes ``output()`` (after ``reset()`` for ops that accumulate in place);
    ``close(out, ref)`` accepts a candidate's output."""

    shape: dict
    run: Callable[[dict], Callable[[], object]]
    output: Callable[[], object]
    close: Callable[[object, object], bool]
    comparator: tuple[str, Callable[[], object]] | None = None
    reset: Callable[[], object] = lambda: None
    sol: Callable[[], object] | None = None  # useful-work benchmark.sol.sol.Call of the bucket


@dataclass
class Bucket:
    shape: dict
    key: tuple
    best: dict
    best_us: float
    default_us: float
    tried: list = field(default_factory=list)  # [config, us | None (invalid/rejected)]
    errors: dict = field(default_factory=dict)  # rejection reason -> count

    @property
    def all_rejected(self) -> bool:
        """Every candidate other than the default failed (often an adapter bug)."""
        return len(self.tried) > 1 and all(us is None for _, us in self.tried[1:])


def _freeze(config):
    return tuple(sorted(config.items()))


def search(op, device_class, case: Case, timer, clone, equal) -> Bucket:
    """Coordinate search over ``op.space`` from the default config."""
    default = op.default(device_class, **case.shape)
    times: dict[tuple, float | None] = {}
    errors: dict[str, int] = {}
    reference = None

    def measure(config):
        nonlocal reference
        key = _freeze(config)
        if key in times:
            return times[key]
        if not op.valid(config, case.shape):
            times[key] = None
            return None
        try:
            launch = case.run(config)
            case.reset()
            launch()
            first = clone(case.output())
            case.reset()
            launch()
            if not equal(first, case.output()):
                raise RuntimeError("not bitwise repeatable")
            if reference is None:
                reference = first
            elif not case.close(first, reference):
                raise RuntimeError("output differs from the default config")
            times[key] = statistics.median(timer(launch))
        except (
            Exception
        ) as error:  # A config that fails to build or run is skipped; not the default.
            if reference is None:
                raise
            reason = f"{type(error).__name__}: {str(error)[:120]}"
            errors[reason] = errors.get(reason, 0) + 1
            times[key] = None
        return times[key]

    best, best_us = dict(default), measure(default)
    default_us = best_us
    for _ in range(PASSES):
        for name in sorted(op.space):
            for value in op.space[name]:
                candidate = {**best, name: value}
                us = measure(candidate)
                if us is not None and us < best_us * (1 - MOVE):
                    best, best_us = candidate, us
    tried = [[dict(key), us] for key, us in times.items()]
    key = op.bucket(**case.shape)
    return Bucket(case.shape, key, best, best_us, default_us, tried, errors)


def shard(op, shapes: Sequence[dict], index: int, count: int) -> list[dict]:
    """Shapes of shard ``index`` of ``count``: whole inheritance groups, dealt round-robin
    in sorted group order, so the keep rule never spans shards."""

    def rest(shape):
        key = op.bucket(**shape)
        return json.dumps(_plain(key if op.axis is None else _drop(key, op.axis)))

    groups = sorted({rest(shape) for shape in shapes})
    mine = set(groups[index::count])
    return [shape for shape in shapes if rest(shape) in mine]


def keep_rows(op, device_class, buckets: Sequence[Bucket], aba) -> tuple[dict, list[dict]]:
    """Apply the keep rule along the inheritance axis.

    ``aba(bucket, a_config, b_config)`` returns the paired speedup of b over a (a/b).
    Returns (table rows, provenance rows).
    """
    axis = op.axis
    groups: dict[tuple, list[Bucket]] = {}
    for b in buckets:
        rest = b.key if axis is None else b.key[:axis] + b.key[axis + 1 :]
        groups.setdefault(rest, []).append(b)
    rows, provenance = {}, []
    for group in groups.values():
        group.sort(key=lambda b: 0 if axis is None else b.key[axis])
        inherited = None
        for b in group:
            parent = op.default(device_class, **b.shape) if inherited is None else inherited
            vs_parent = 1.0 if b.best == parent else aba(b, parent, b.best)
            kept = vs_parent >= KEEP
            if kept:
                rows[b.key] = b.best
                inherited = b.best
            served = b.best if kept else parent
            default = op.default(device_class, **b.shape)
            provenance.append(
                {
                    "shape": b.shape,
                    "bucket": _plain(b.key),
                    "config": served,
                    "kept": kept,
                    "speedup_vs_inherited": round(vs_parent, 4),
                    "speedup_vs_default": round(
                        1.0 if served == default else aba(b, default, served), 4
                    ),
                }
            )
    return rows, provenance


def imported(op, device_class, measured: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    """Table rows and provenance from external paired measurements.

    ``measured`` rows are {"shape", "config", "speedup"} with speedup = default time / config
    time on one card; every config a bucket may inherit must be measured at that bucket.
    The keep rule is the same as :func:`keep_rows`."""
    by_bucket: dict[tuple, dict] = {}
    for row in measured:
        shape = dict(row["shape"])
        entry = by_bucket.setdefault(op.bucket(**shape), {"shape": shape, "speedups": {}})
        entry["speedups"][_freeze(op.check(row["config"], shape))] = row["speedup"]
    buckets = []
    for key, entry in by_bucket.items():
        default = _freeze(op.default(device_class, **entry["shape"]))
        entry["speedups"].setdefault(default, 1.0)
        best = max(entry["speedups"], key=entry["speedups"].get)
        tried = [[dict(c), 1 / v] for c, v in entry["speedups"].items()]
        buckets.append(
            Bucket(entry["shape"], key, dict(best), 1 / entry["speedups"][best], 1.0, tried)
        )

    def aba(bucket, a, b):
        speedups = by_bucket[bucket.key]["speedups"]
        if _freeze(a) not in speedups:
            raise ValueError(f"{bucket.shape}: inherited config {a} was not measured here")
        return speedups[_freeze(b)] / speedups[_freeze(a)]

    return keep_rows(op, device_class, buckets, aba)


def neighbours(op, buckets: Sequence[Bucket]) -> list[tuple[Bucket, str, Bucket]]:
    """(bucket, differing dims, neighbour) for adjacent neighbours with another best config.

    Neighbours differ in one dim (nearest value on each side for an ordered dim) or in two
    unordered dims that co-vary, such as k+sizes of one projection."""
    names = [dim.name for dim in op.dims]
    ordered = {i for i, dim in enumerate(op.dims) if dim.edges is not None}
    values = {i: sorted({b.key[i] for b in buckets}) for i in ordered}
    found = []
    for b in buckets:
        nearest: dict[tuple, Bucket] = {}
        for other in buckets:
            diff = tuple(i for i, (x, y) in enumerate(zip(b.key, other.key, strict=True)) if x != y)
            if not diff or len(diff) > 2 or (len(diff) == 2 and set(diff) & ordered):
                continue
            if len(diff) == 1 and diff[0] in ordered:
                at = values[diff[0]].index(b.key[diff[0]])
                if abs(values[diff[0]].index(other.key[diff[0]]) - at) != 1:
                    continue
            nearest[(diff, json.dumps(_plain(other.key)))] = other
        for (diff, _), other in sorted(nearest.items(), key=lambda item: item[0]):
            if other.best != b.best:
                found.append((b, "+".join(names[i] for i in diff), other))
    return found


def sensitivity(op, buckets: Sequence[Bucket], probe) -> dict[str, dict]:
    """Per set of differing dims: adjacent bucket pairs, and for those whose best configs
    differ, the speedup of the bucket's own best over the neighbour's best on the bucket's
    shape (``probe(bucket, configs)`` -> times, screen timing, both configs in one call)."""
    report: dict[str, dict] = {}
    for b, dims, other in neighbours(op, buckets):
        own, theirs = probe(b, [b.best, other.best])
        entry = report.setdefault(dims, {"pairs": 0, "speedups": []})
        entry["pairs"] += 1
        if own and theirs:
            entry["speedups"].append(round(theirs / own, 4))
    return report


def summarize(reports: Sequence[Mapping[str, dict]]) -> dict[str, dict]:
    """Combine sensitivity reports: per dim set the adjacent pairs whose best configs differ,
    and the median / max speedup of the own best over the neighbour's."""
    total: dict[str, dict] = {}
    for report in reports:
        for name, entry in report.items():
            into = total.setdefault(name, {"pairs": 0, "speedups": []})
            into["pairs"] += entry["pairs"]
            into["speedups"] += entry["speedups"]
    return {
        name: {
            "pairs": entry["pairs"],
            "median_speedup": statistics.median(entry["speedups"]) if entry["speedups"] else 1.0,
            "max_speedup": max(entry["speedups"], default=1.0),
            "over_keep": sum(s >= KEEP for s in entry["speedups"]),
        }
        for name, entry in total.items()
    }


def provenance(op, device: Mapping, rows: list[dict], sensitivity_report: Mapping) -> str:
    head = {"op": op.name, "version": op.version, "keep": KEEP, **device}
    return json.dumps({**head, "sensitivity": sensitivity_report, "rows": rows}, indent=1) + "\n"


def _drop(key, index):
    return key[:index] + key[index + 1 :]


def _plain(value):
    return [_plain(v) for v in value] if isinstance(value, tuple) else value
