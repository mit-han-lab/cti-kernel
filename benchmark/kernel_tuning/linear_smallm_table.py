# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Generate measured dispatch or Markdown from RESULT logs (CPU only).

Probe candidates use median cold paired speedup >= 1 + margin (default 3%).
Engine-regime (alternating cold/hot weights) acceptance must pass at both bucket
edges; hot-L2 rows only veto on RTX. GB10 also requires hot-L2 evidence,
at least 60 pairs per record and a bootstrap 95% CI lower bound above one.
Acceptance gates whole M buckets: every measured M must qualify the same launch
configuration, and both bucket edges must be measured. Only rows whose cuBLAS
comparator ran with torch's default reduced-precision BF16 reduction (as the
market calls F.linear) qualify. Current retries
are resolved separately for cold/hot L2; the minimum across aliases and L2 modes
must pass, with cold evidence required. Missing evidence falls back to cuBLAS.
A key shared by model families (MERGED_KEYS) must qualify from each family's rows.
Candidate --bucketize selects the majority config, breaking ties at the highest M.
Legacy probes obtain device_class from their preceding probe_environment row.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import astuple, fields
from pathlib import Path

from ctikernels.linear._dispatch import GB10_MIN_PAIRS, Config, qualifies
from ctikernels.linear.cute import mtiles_for

# Support the documented direct-script invocation as well as module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEVICES = {"rtx5090": "RTX 5090", "pro6000": "RTX PRO 6000", "gb10": "GB10"}
BEGIN = "# BEGIN GENERATED SMALLM TABLE"
END = "# END GENERATED SMALLM TABLE"
GATED_BEGIN = "# BEGIN GENERATED GATED TABLE"
GATED_END = "# END GENERATED GATED TABLE"
MS = (1, 2, 3, 4, 6, 8)
READ_GBPS = {"rtx5090": 1681.1, "pro6000": 1533.1, "gb10": 244.6}


def read_logs(paths, *, optional=False, round_label="supplied round", source_offset=0):
    rows = []
    for source, path in enumerate(paths, source_offset):
        if optional and not path.exists():
            continue
        start = len(rows)
        if any(word in path.name.lower() for word in ("secret", "token")):
            raise ValueError("sensitive-name inputs are forbidden")
        devices = {}
        for line in path.read_text().splitlines():
            if not line.startswith("RESULT "):
                continue
            row = json.loads(line[7:])
            if row.get("kind") == "probe_environment":
                devices[row["N"], row["K"]] = row.get("device_class")
            if row.get("kind") == "smallm_probe" and not row.get("device_class"):
                row["device_class"] = devices.get((row.get("N"), row.get("K")))
            row.update(_source=source, _round=round_label)
            rows.append(row)
        mark_completion(rows[start:])
    return rows


MARKERS = (
    "completion",
    "probe_completion",
    "accept_completion",
    "gate_accept_completion",
    "gate_probe_completion",
)
RUN_ROWS = {"accept": "smallm_accept", "gate_accept": "gate_accept", "gate_probe": "gate_probe"}


def completion_kind(row):
    kind = row.get("kind", "market")
    if kind in ("gate_accept", "gate_probe"):
        return kind + "_completion"
    if kind == "smallm_probe":
        return "probe_completion"
    if kind in ("smallm_accept", "smallm_step", "smallm_round"):
        return "accept_completion"
    return "completion"


def finished(marker):
    return marker.get("kind") == "completion" or marker.get("complete") is True


def successful(marker):
    return finished(marker) and (
        marker.get("kind") == "completion" or marker.get("failures", 0) == 0
    )


def mark_completion(records):
    """Scope completion to each run within one (possibly concatenated) log.

    Rows stamped with a run_id complete only through that run's own begin and
    successful end marker. Legacy rows without a run_id belong to the segment
    closed by the next marker of their kind; a marker that reports a row count
    must match the accept rows of its segment, so a truncated run followed by a
    complete one fails both, and rows after the last marker never complete.
    ``_finished`` marks rows of runs that reached their end marker, failures
    included (reported); ``_complete`` additionally needs zero failures (qualifies).
    ``_run`` identifies the run so retries of a key resolve in time order.
    """
    begun, ended, counted = set(), {}, defaultdict(int)
    for row in records:
        run = row.get("run_id")
        if run is None:
            continue
        for prefix, row_kind in RUN_ROWS.items():
            key = prefix, run, row.get("out_dtype", "bf16")
            if row.get("kind") == prefix + "_begin":
                begun.add(key)
            elif row.get("kind") == prefix + "_completion":
                ended[key] = row
            elif row.get("kind") == row_kind:
                counted[key] += 1
    segments = defaultdict(list)
    for index, row in enumerate(records):
        kind = row.get("kind", "market")
        run = row.get("run_id")
        if kind in tuple(p + "_begin" for p in RUN_ROWS) or (run is None and kind in MARKERS):
            if run is None:
                segment = segments.pop((kind, row.get("out_dtype", "bf16")), [])
                for item in segment:
                    item["_run"] = ("segment", kind, index)
                accepts = sum(r.get("kind") in RUN_ROWS.values() for r in segment)
                counted_ok = row.get("rows", accepts) == accepts
                for item in segment:
                    item["_finished"] = finished(row) and counted_ok
                    item["_complete"] = successful(row) and counted_ok
            continue
        if run is not None:
            row["_run"] = ("run", run)
            key = (
                completion_kind(row).removesuffix("_completion"),
                run,
                row.get("out_dtype", "bf16"),
            )
            marker = ended.get(key)
            closed = (
                key in begun
                and marker is not None
                and marker.get("rows", counted[key]) == counted[key]
            )
            row["_finished"] = closed and finished(marker)
            row["_complete"] = closed and successful(marker)
            continue
        row["_finished"] = row["_complete"] = False
        segments[completion_kind(row), row.get("out_dtype", "bf16")].append(row)


def dtype_rows(rows, out_dtype):
    """Output dtype is part of launch identity; legacy evidence is BF16 only."""
    if out_dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported output dtype")
    return [row for row in rows if row.get("out_dtype", "bf16") == out_dtype]


def select_configs(rows, margin=0.03, *, out_dtype="bf16"):
    check_margin(margin)
    return _select_configs(dtype_rows(rows, out_dtype), margin)


def _select_configs(rows, margin):
    """Rank probe candidates, also allowing losers in measurement reports."""
    grouped = defaultdict(list)
    invalid = set()
    for row in rows:
        if row.get("kind") != "smallm_probe" or row.get("device_class") not in DEVICES:
            continue
        if not row.get("config") or row.get("M") is None:
            continue
        cfg = Config(**row["config"])
        n, k, m = row["N"], row["K"], row["M"]
        if not (type(n) is int and type(k) is int and type(m) is int):
            raise ValueError("integer N/K/M required")
        if n <= 0 or k <= 0 or k % 8 or not 1 <= m <= 64:
            raise ValueError("unsupported probe dimensions")
        key = row["device_class"], n, k, m, cfg
        if row.get("status") == "incorrect" or row.get("correct") is False:
            invalid.add(key)
        if row.get("status") != "ok" or row.get("correct") is not True:
            continue
        if row.get("mode", "cold") != "cold" or row.get("provider") != "cute":
            continue
        ratio, us = row["paired_speedup"], row["ours_us"]
        if not all(math.isfinite(v) and v > 0 for v in (ratio, us)):
            raise ValueError("finite positive timing and paired ratio required")
        grouped[key].append((ratio, us))
    best = {}
    for key, samples in grouped.items():
        if key in invalid:
            continue
        ratio = statistics.median(r for r, _ in samples)
        if ratio < 1 + margin:
            continue
        us = statistics.median(t for _, t in samples)
        *shape, cfg = key
        rank = (-ratio, us, astuple(cfg))
        shape = tuple(shape)
        if shape not in best or rank < best[shape][0]:
            best[shape] = rank, cfg
    table = {kind: {} for kind in DEVICES}
    for (kind, n, k, m), (_, cfg) in sorted(best.items()):
        ranges = table[kind].setdefault((n, k), {})
        if ranges:
            lo, hi = next(reversed(ranges))
            if hi + 1 == m and ranges[lo, hi] == cfg:
                del ranges[lo, hi]
                ranges[lo, m] = cfg
                continue
        ranges[m, m] = cfg
    return table


CONFIG_FIELDS = frozenset(f.name for f in fields(Config))


def launch_identity(row, gated=False):
    """(Config, M bucket) an acceptance row measured, or None without a full record."""
    launch = row.get("launch_config")
    if gated and (
        not isinstance(launch, dict)
        or launch.get("gate_stride") != row.get("gate_stride")
        or type(row.get("gate_stride")) is not int
    ):
        return None
    if not isinstance(launch, dict) or not isinstance(launch.get("config"), dict):
        return None
    # Every code-selecting field must be recorded; Config defaults never fill gaps.
    # The one exception is ``cluster``: records written before that field existed
    # measured the only reduction the kernel then had (cluster=False).
    if set(launch["config"]) == CONFIG_FIELDS - {"cluster"}:
        launch = dict(launch, config={**launch["config"], "cluster": 0})
    if set(launch["config"]) != CONFIG_FIELDS:
        return None
    if launch.get("out_dtype", row.get("out_dtype", "bf16")) != row.get("out_dtype", "bf16"):
        return None
    cfg = Config(**launch["config"])
    if any(getattr(cfg, name) != value for name, value in launch["config"].items()):
        return None
    if launch.get("mtiles") != mtiles_for(row["M"]) or launch.get("cta_threads") != 32 * cfg.warps:
        return None
    return cfg, launch["mtiles"]


RETRY_KEY = ("kind", "model", "name", "layout", "N", "K", "gate_stride", "M", "B", "Q", "l2")


def resolve_retries(entries, gated=False):
    """Retries resolved per key in run order: the one rule for every consumer.

    ``entries`` are one device's rows of one log, in run (file) order. Per
    model/projection/N/K/M (and layout/B/Q for totals) the latest finished
    run's row decides; a correctness FAIL counts even from an unfinished run,
    so it invalidates the key until a later finished run replaces it. Other
    rows of unfinished runs only stand in for keys no run has decided (callers
    check ``decided``). A correctness FAIL also turns the current CuTe rows of
    every projection alias at its N/K/M that measured the same configuration
    (every configuration without a launch record) into FAIL, until that alias
    passes again. Qualification, the per-row reports and the round totals all
    read these rows.
    """
    accept_kind, provider = ("gate_accept", "cute-gated") if gated else ("smallm_accept", "cute")
    latest = {}
    for row in entries:
        key = (
            row.get("out_dtype", "bf16"),
            *(row.get(field, "cold" if field == "l2" else None) for field in RETRY_KEY),
        )
        if row.get("correct") is False and row.get("kind") == accept_kind:
            failed = launch_identity(row, gated) if row.get("provider") == provider else None
            for other, kept in latest.items():
                if (
                    kept.get("kind") == accept_kind
                    and kept.get("provider") == provider
                    and kept.get("l2", "cold") == row.get("l2", "cold")
                    and shape_key(kept, gated) == shape_key(row, gated)
                    and kept["M"] == row["M"]
                    and kept.get("out_dtype", "bf16") == row.get("out_dtype", "bf16")
                    and (failed is None or launch_identity(kept, gated) == failed)
                ):
                    latest[other] = dict(kept, verdict="FAIL", _invalidated=True)
        if decided(row) or key not in latest or not decided(latest[key]):
            latest[key] = row
    return list(latest.values())


def decided(row):
    """A row of a finished run, or a correctness FAIL from any run."""
    return bool(row.get("_finished")) or row.get("correct") is False


def shape_key(row, gated=False):
    return (row["N"], row["K"], row["gate_stride"]) if gated else (row["N"], row["K"])


def resolved_acceptance(rows, gated=False):
    """Current acceptance rows per device, retries resolved (``resolve_retries``)."""
    if gated:
        grouped = defaultdict(list)
        for row in rows:
            if is_current_acceptance(row, True) and device_kind(row) is not None:
                grouped[device_kind(row)].append(row)
    else:
        grouped = latest_rows(rows, is_current_acceptance)
    return {kind: resolve_retries(entries, gated) for kind, entries in grouped.items()}


def accepted_speedups(rows):
    """Speedups of ``accepted_evidence`` (cold-qualified keys only)."""
    return accepted_evidence(rows)[0]


def accepted_evidence(rows, gated=False):
    """Current-round CuTe acceptance speedup per device/N/K/M/configuration.

    Reads the rows ``resolved_acceptance`` keeps, so a correctness FAIL
    invalidates its configuration (every configuration when the failing row
    lacks a launch record) and a non-PASS latest row drops an earlier PASS.
    The minimum over projection names sharing N/K is kept. Only ``--accept``
    rows qualify; ``--previous-accept`` rows are for reporting. A row without
    the measured launch configuration qualifies nothing.

    Returns (speedups, engine): ``engine`` holds the keys with a finished,
    correct, passing engine-regime (alternating cold/hot L2) record. GB10 keys
    additionally need hot evidence in that set, so both regimes are required at
    the bucket edges. Every GB10 record needs >=60 pairs and ci_low > 1.
    """
    ratios, blocked, cold, engine, hot_keys = {}, set(), set(), set(), set()
    for kind, entries in resolved_acceptance(rows, gated).items():
        for row in entries:
            if row.get("provider") != ("cute-gated" if gated else "cute") or not decided(row):
                continue
            shape = kind, *shape_key(row, gated), row["M"], row.get("out_dtype", "bf16")
            identity = launch_identity(row, gated)
            if row.get("correct") is False:
                blocked.add((*shape, identity))
                continue
            if identity is None:
                continue
            # Only a comparator timed as the market calls F.linear counts (torch's
            # default reduced-precision BF16 reduction changes cuBLAS's kernel).
            if (
                row.get(
                    "market_bf16_reduced_precision" if gated else "cublas_bf16_reduced_precision"
                )
                is not True
            ):
                continue
            key = (*shape, *identity)
            if row.get("l2", "cold") == "cold":
                cold.add(key)
            # Qualification comes from a complete cold run plus finished, passing
            # engine (alternating cold/hot) rows at the bucket edges; GB10 also
            # requires hot-L2 there. A finished hot/engine run's per-row FAILs do not
            # void its other rows.
            hot = row.get("l2", "cold") in ("hot", "engine")
            if not (
                (row.get("_complete") or (hot and row.get("_finished")))
                and row.get("correct") is True
                and row.get("verdict") == "PASS"
            ):
                blocked.add((*shape, identity))
                continue
            if row.get("l2") == "engine":
                engine.add(key)
            if row.get("l2") == "hot":
                hot_keys.add(key)
            ratio = row["paired_speedup"]
            if not (math.isfinite(ratio) and ratio > 0):
                raise ValueError("finite positive acceptance ratio required")
            # A gain inside the row's own self-pair noise is inconclusive; with the
            # margin test below this is the shared ``qualifies`` gate per row.
            if not qualifies(ratio, row.get("epsilon", 0.0), margin=0.0) or (
                kind == "gb10"
                and not (
                    row.get("pairs", 0) >= GB10_MIN_PAIRS
                    and isinstance(row.get("ci_low"), (int, float))
                    and math.isfinite(row["ci_low"])
                    and row["ci_low"] > 1
                )
            ):
                ratio = min(ratio, 1.0)
            ratios[key] = min(ratio, ratios.get(key, ratio))
    speedups = {
        key: ratio
        for key, ratio in ratios.items()
        if key in cold and (*key[:-2], None) not in blocked and (*key[:-2], key[-2:]) not in blocked
    }
    return speedups, {
        key for key in engine & speedups.keys() if key[0] != "gb10" or key in hot_keys
    }


def is_current_acceptance(row, gated=False):
    return (
        row.get("kind") == ("gate_accept" if gated else "smallm_accept")
        and row.get("_round") == "current round"
    )


# (out_dtype, N, K) keys that more than one model family's inventory contains.
# A shared key is one GEMM, so it gets one row; that row keeps CuTe only where
# every family's own engine-site evidence qualifies it. A new overlap must be
# declared here (tests compare this with the inventories).
MERGED_KEYS = {("bf16", 64, 4096): frozenset({"qwen", "dsv4"})}
DSV4_INVENTORIES = (("dsv4_bf16_shapes.json", "bf16"), ("dsv4_fp32_shapes.json", "fp32"))


def model_families():
    """Model name -> family, and (family, out_dtype) -> inventory (N, K) keys."""
    from benchmark.kernel_tuning.linear_smallm import builtin_shapes

    family, keys = {}, defaultdict(set)
    for model, shapes in builtin_shapes().items():
        family[model] = "qwen"
        keys["qwen", "bf16"] |= {(s["N"], s["K"]) for s in shapes}
    here = Path(__file__).resolve().parent
    for name, out_dtype in DSV4_INVENTORIES:
        for model, shapes in json.loads((here / name).read_text())["models"].items():
            family[model] = "dsv4"
            keys["dsv4", out_dtype] |= {(s["N"], s["K"]) for s in shapes}
    return family, keys


def shared_keys():
    """(out_dtype, N, K) -> families, for keys in more than one family's inventory."""
    _, keys = model_families()
    sites = defaultdict(set)
    for (fam, out_dtype), shapes in keys.items():
        for n, k in shapes:
            sites[out_dtype, n, k].add(fam)
    return {key: frozenset(fams) for key, fams in sites.items() if len(fams) > 1}


def _evidence(rows, gated):
    speedups, engine = accepted_evidence(rows, gated)
    measured = defaultdict(list)
    for kind, current in resolved_acceptance(rows, gated).items():
        for row in current:
            if decided(row):
                measured[(kind, *shape_key(row, gated), row["M"])].append(row)
    return speedups, engine, measured


def gate_by_acceptance(table, rows, margin=0.03, refused=None, *, gated=False, out_dtype="bf16"):
    """Keep a whole bucket only when every measured M qualifies its exact launch.

    Both edges of the bucket must be measured, cold and in the engine regime
    (GB10 also requires hot records); other buckets fall back. Record mismatched launch evidence in
    ``refused``. A key in ``MERGED_KEYS`` must also qualify from each listed
    model family's rows alone.
    """
    from ctikernels.linear._dispatch import M_BUCKETS

    check_margin(margin)
    rows = dtype_rows(rows, out_dtype)
    evidence = _evidence(rows, gated)
    family = model_families()[0] if not gated else {}
    by_family = {}
    if not gated:
        for fam in {f for fams in MERGED_KEYS.values() for f in fams}:
            subset = [r for r in rows if family.get(r.get("model")) == fam]
            by_family[fam] = _evidence(subset, gated)

    def bucket_ok(kind, shape, lo, hi, cfg, speedups, engine, measured, refused):
        found = [row for m in range(lo, hi + 1) for row in measured[(kind, *shape, m)]]
        # cuBLAS can change kernels inside a bucket (flag off: 24576x4096 at M=42/43 on
        # the RTX 5090): both bucket edges must be measured.
        valid = bool(measured[(kind, *shape, lo)]) and bool(measured[(kind, *shape, hi)])
        if not all((kind, *shape, m, out_dtype, cfg, mtiles_for(m)) in engine for m in (lo, hi)):
            valid = False
        for row in found:
            m = row["M"]
            identity = launch_identity(row, gated)
            if row.get("provider") != ("cute-gated" if gated else "cute") or identity != (
                cfg,
                mtiles_for(m),
            ):
                valid = False
                if refused is not None and identity is not None and identity[0] != cfg:
                    refusal = (kind, *shape, m, cfg)
                    if refusal not in refused:
                        refused.append(refusal)
            if not qualifies(
                speedups.get((kind, *shape, m, out_dtype, cfg, mtiles_for(m)), 0),
                0.0,
                margin,
            ):
                valid = False
        return valid

    kept = {}
    for kind, shapes in bucketize(table).items():
        for shape, ranges in shapes.items():
            sites = () if gated else MERGED_KEYS.get((out_dtype, *shape), ())
            for lo, hi in M_BUCKETS:
                cfg = next((c for (start, end), c in ranges.items() if start <= hi <= end), None)
                if cfg is None:
                    continue
                valid = bucket_ok(kind, shape, lo, hi, cfg, *evidence, refused)
                for fam in sorted(sites):
                    valid = valid and bucket_ok(kind, shape, lo, hi, cfg, *by_family[fam], None)
                if valid:
                    for m in range(lo, hi + 1):
                        kept[(kind, *shape, m)] = cfg
    return _compress(kept)


def _compress(entries):
    """{(kind, n, k, m): cfg} -> table, merging consecutive M with equal configs only."""
    table = {kind: {} for kind in DEVICES}
    for (kind, *shape, m), cfg in sorted(entries.items(), key=lambda item: item[0]):
        ranges = table[kind].setdefault(tuple(shape), {})
        if ranges:
            lo, hi = next(reversed(ranges))
            if hi + 1 == m and ranges[lo, hi] == cfg:
                del ranges[lo, hi]
                ranges[lo, m] = cfg
                continue
        ranges[m, m] = cfg
    return table


def merge_candidates(base, selected, rows=()):
    """New probe winners replace base entries; other base entries stay candidates.

    A probe may not include the committed config, so a missing winner is not
    evidence against it. The following acceptance round re-gates every entry.
    """
    entries = {}
    for source in (base, selected):
        for kind, shapes in source.items():
            for shape, ranges in shapes.items():
                for (lo, hi), cfg in ranges.items():
                    for m in range(lo, hi + 1):
                        entries[(kind, *shape, m)] = cfg
    return _compress(entries)


def bucketize(table):
    """Whole-M-bucket candidates from a per-M table (one config per bucket).

    A bucket with any covered M takes the config covering most of its M values
    (ties: the one covering the highest M); acceptance then gates the bucket.
    """
    from ctikernels.linear._dispatch import M_BUCKETS

    entries = {}
    for kind, shapes in table.items():
        for shape, ranges in shapes.items():
            covered = {m: cfg for (lo, hi), cfg in ranges.items() for m in range(lo, hi + 1)}
            for lo, hi in M_BUCKETS:
                configs = [(m, covered[m]) for m in range(lo, hi + 1) if m in covered]
                if not configs:
                    continue
                counts = defaultdict(list)
                for m, cfg in configs:
                    counts[cfg].append(m)
                cfg = max(counts, key=lambda c: (len(counts[c]), max(counts[c])))
                for m in range(lo, hi + 1):
                    entries[(kind, *shape, m)] = cfg
    return _compress(entries)


def check_margin(margin):
    if not (math.isfinite(margin) and margin >= 0.03):
        raise ValueError("--margin must be finite and >= 0.03")
    return margin


def python_literal(table):
    lines = ["{"]
    for kind, shapes in table.items():
        lines.append(f'    "{kind}": {{')
        for shape, ranges in sorted(shapes.items()):
            lines.append(f"        {shape}: {{")
            for bounds, cfg in ranges.items():
                # cluster=0 rows keep their seven-field literal.
                args = ", ".join(map(repr, astuple(cfg)[: None if cfg.cluster else -1]))
                lines.append(f"            {bounds}: Config({args}),")
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def _rewrite_measured(source, table, measured, name="TUNED"):
    """Replace only measured shape values; preserve every other source byte."""
    original = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    present = {
        (ast.literal_eval(kind), ast.literal_eval(shape))
        for kind, rows in zip(original.keys, original.values, strict=True)
        for shape in rows.keys
    }
    scoped = {}
    for kind, n, k in sorted(measured):
        rows = table.get(kind, {}).get((n, k), {})
        # Empty measured shapes only delete existing entries; they are never added.
        if rows or (kind, (n, k)) in present:
            scoped.setdefault(kind, {})[n, k] = rows
    if not scoped:
        return source
    replacement = python_literal(scoped)
    desired = ast.parse(replacement, mode="eval").body
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def bounds(node):
        return offsets[node.lineno - 1] + node.col_offset, offsets[
            node.end_lineno - 1
        ] + node.end_col_offset

    def entries(node):
        return {ast.literal_eval(k): v for k, v in zip(node.keys, node.values, strict=True)}

    edits = []

    def span(node, key):
        """Bytes of one `key: value,` entry, with its line's leading whitespace."""
        index = next(i for i, k in enumerate(node.keys) if ast.literal_eval(k) == key)
        start, end = bounds(node.keys[index])[0], bounds(node.values[index])[1]
        if source[end : end + 1] == b",":
            end += 1
        while start and source[start - 1 : start] == b" ":
            start -= 1
        if source[start - 1 : start] == b"\n":
            start -= 1
        return start, end, b""

    def patch(old, new, depth):
        existing = entries(old)
        for key, value in entries(new).items():
            text = ast.get_source_segment(replacement, value).encode()
            if depth == 1 and isinstance(value, ast.Dict) and not value.keys:
                # A measured shape with no qualifying bucket leaves the table.
                if key in existing:
                    edits.append(span(old, key))
                continue
            if key in existing:
                if depth == 0:
                    patch(existing[key], value, 1)
                else:
                    edits.append((*bounds(existing[key]), text))
            else:
                start, _ = bounds(old)
                addition = (
                    b"\n" + b" " * (4 * (depth + 1)) + repr(key).encode() + b": " + text + b","
                )
                edits.append((start + 1, start + 1, addition))

    patch(original, desired, 0)
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


def rewrite(path, table, *, gated=False, only_measured=None, out_dtype="bf16"):
    begin, end = (GATED_BEGIN, GATED_END) if gated else (BEGIN, END)
    name = "TUNED_GATED" if gated else "TUNED"
    if out_dtype == "fp32":
        if gated:
            raise ValueError("FP32 gated output is unsupported")
        begin, end = "# BEGIN GENERATED SMALLM F32 TABLE", "# END GENERATED SMALLM F32 TABLE"
        name = "TUNED_F32"
    source = path.read_text()
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("expected exactly one generated block")
    if only_measured is not None:
        path.write_bytes(_rewrite_measured(path.read_bytes(), table, only_measured, name))
        return
    before, rest = source.split(begin)
    _, after = rest.split(end)
    path.write_text(
        before + begin + "\n" + name + " = " + python_literal(table) + "\n" + end + after
    )


def device_kind(row):
    kind = row.get("device_class")
    if kind in DEVICES:
        return kind
    return next((k for k, label in DEVICES.items() if label in row.get("device", "")), None)


def cell(value):
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def speedup_cell(row, digits=4):
    """Paired speedup with its measured 95% interval for GB10, when present."""
    ratio = row.get("paired_speedup")
    if ratio is None:
        return cell(ratio)
    result = f"{ratio:.{digits}f}"
    if device_kind(row) == "gb10" and all(row.get(k) is not None for k in ("ci_low", "ci_high")):
        result += f" [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    return result


def grid(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        "",
    ]


def latest_rows(rows, predicate):
    """Last supplied run per device/L2 regime; previous acceptance is a fallback."""
    selected = {}
    for row in rows:
        kind = device_kind(row)
        if kind is None or not predicate(row):
            continue
        priority = (row.get("_round") != "previous round", row.get("_source", 0))
        key = kind, row.get("l2", "cold"), row.get("out_dtype", "bf16")
        if key not in selected or priority > selected[key][0]:
            selected[key] = priority, []
        if priority == selected[key][0]:
            selected[key][1].append(row)
    result = defaultdict(list)
    for (kind, _l2, _dtype), (_, entries) in selected.items():
        result[kind].extend(entries)
    return dict(result)


def markdown(rows):
    # Only public labels, built-in projection names and typed numeric fields are
    # rendered. Never echo paths, environment dictionaries or exception messages.
    from benchmark.kernel_tuning.linear_smallm import (
        builtin_shapes,
        round_estimates,
        step_estimates,
    )

    models = builtin_shapes()
    output = [Path(__file__).with_name("linear_smallm_report.md").read_text().rstrip(), ""]
    market = latest_rows(rows, lambda r: "us_median" in r and r.get("mode") == "cold")
    output += [
        "## Market measurements",
        "",
        "Cold, read-flush; Qwen3.5-9B. Each provider cell is **microseconds / utilization**.",
        "",
    ]
    for kind, label in DEVICES.items():
        entries = market.get(kind, [])
        output += [f"### {label}", ""]
        if not entries:
            output += ["Pending: no cold market RESULT rows supplied.", ""]
            continue
        indexed = {
            (r.get("N"), r.get("K"), r.get("M"), r.get("provider")): r
            for r in entries
            if r.get("model") == "qwen3.5-9b"
        }
        complete = all(r.get("_complete", False) for r in entries)
        errors = sum(r.get("status") == "error" for r in entries)
        pairs = sorted({r["pairs"] for r in entries if "pairs" in r})
        reps = sorted({r["reps_per_sample"] for r in entries if "reps_per_sample" in r})
        output += [
            f"Run completion marker: {complete}; error rows: {errors}; pairs: {pairs}; replays/sample: {reps}.",
            "",
        ]
        data = []
        for shape in models["qwen3.5-9b"]:
            n, k = shape["N"], shape["K"]
            for m in sorted(set(MS) | {r["M"] for r in entries if isinstance(r.get("M"), int)}):
                sol = 2 * (n * k + m * k + m * n) / (READ_GBPS[kind] * 1e3)
                times = []
                for provider in ("cublas", "cublaslt", "quack", "ours"):
                    row = indexed.get((n, k, m, provider), {})
                    us = row.get("us_median")
                    if row.get("status") == "ok" and us is not None:
                        value = f"{us:.3f} / {sol / us:.3f}"
                        if provider == "ours":
                            actual = row.get("actual_provider", "cute")
                            value += (
                                " (cuBLAS)"
                                if actual == "cublas"
                                else " (CuTe)"
                                if actual == "cute"
                                else " (custom)"
                            )
                    else:
                        value = {"error": "error; rerun pending", "skipped": "skipped"}.get(
                            row.get("status"), "pending"
                        )
                    times.append(value)
                data.append([shape["name"], n, k, m, *times, sol])
        output += grid(
            [
                "Projection",
                "N",
                "K",
                "M",
                "cuBLAS",
                "cuBLASLt",
                "QuACK 0.6.5",
                "Ours (dispatch)",
                "SOL us",
            ],
            data,
        )

    output += [
        "## Probe best configurations",
        "",
        "Best correct cold candidate per device/N/K/M across supplied probe observations, including losers to cuBLAS. Repeats use median paired speedup; any incorrect observation vetoes that candidate/M. These selection data are not independent acceptance. `Admit` means speedup >=1.03, not that this report rewrote dispatch. Config = (rows, tile K, stages, splits, warps, variant, prefetch).",
        "",
    ]
    probe_rows = [r for r in rows if r.get("kind") == "smallm_probe"]
    best = _select_configs(probe_rows, margin=-1)
    samples = defaultdict(list)
    for row in probe_rows:
        if (
            row.get("status") == "ok"
            and row.get("correct") is True
            and row.get("mode", "cold") == "cold"
            and row.get("provider") == "cute"
        ):
            samples[device_kind(row), row["N"], row["K"], row["M"], Config(**row["config"])].append(
                row
            )
    for kind, label in DEVICES.items():
        output += [f"### {label}", ""]
        if not best[kind]:
            output += ["Pending: no qualifying probe observations.", ""]
            continue
        runs = {
            r.get("_source", 0): r.get("_complete", False)
            for r in probe_rows
            if device_kind(r) == kind
        }
        output += [
            f"Supplied probe runs: {len(runs)}; runs with successful completion markers: {sum(runs.values())}.",
            "",
        ]
        for model, shapes in models.items():
            output += [f"#### {model}", ""]
            data = []
            # Deduplicate projection aliases within each model.
            for n, k in dict.fromkeys((s["N"], s["K"]) for s in shapes):
                configs = {
                    m: cfg
                    for (lo, hi), cfg in best[kind].get((n, k), {}).items()
                    for m in range(lo, hi + 1)
                }
                for m in sorted(set(range(1, 9)) | set(configs)):
                    cfg = configs.get(m)
                    if cfg is None:
                        data.append([n, k, m, "pending", None, None, None, "no"])
                        continue
                    observed = samples[kind, n, k, m, cfg]
                    ratio = statistics.median(r["paired_speedup"] for r in observed)
                    ours = statistics.median(r["ours_us"] for r in observed)
                    baseline = (
                        statistics.median(r["cublas_us"] for r in observed if "cublas_us" in r)
                        if any("cublas_us" in r for r in observed)
                        else None
                    )
                    data.append(
                        [
                            n,
                            k,
                            m,
                            astuple(cfg),
                            ours,
                            baseline,
                            ratio,
                            "yes" if ratio >= 1.03 else "no",
                        ]
                    )
            output += grid(
                ["N", "K", "M", "Config", "Ours us", "cuBLAS us", "Paired speedup", "Admit"], data
            )

    # Decided cold-L2 rows only, and totals recomputed from them: rows of
    # unfinished runs are not results, and logged step/round rows of hot re-pair
    # runs would mix regimes. Hot rows only veto table entries.
    acceptance = {}
    for kind, entries in latest_rows(rows, lambda r: r.get("kind") == "smallm_accept").items():
        cold = [
            dict(r, device=kind)
            for r in resolve_retries(entries)
            if r.get("l2", "cold") == "cold" and decided(r)
        ]
        acceptance[kind] = cold + step_estimates(cold, models) + round_estimates(cold, models)
    output += [
        "## Acceptance",
        "",
        "At least 20 interleaved pairs and equally many cuBLAS self-pairs per row; epsilon is linear p95 of abs(self ratio - 1). GB10 qualification requires at least 60 pairs and a bootstrap 95% CI lower bound > 1 in cold, engine and hot records at both bucket edges. PASS requires correctness and paired speedup >= 1-epsilon. Rows of unfinished runs are omitted. Previous-round rows do not qualify a newer table.",
        "",
    ]
    for kind, label in DEVICES.items():
        entries = [r for r in acceptance.get(kind, []) if r.get("kind") == "smallm_accept"]
        output += [f"### {label}", ""]
        if not entries:
            output += ["Pending: no finished acceptance RESULT rows supplied.", ""]
            continue
        count = sum(r.get("verdict") == "PASS" for r in entries)
        output += [
            f"{entries[0].get('_round', 'supplied round')}; {count}/{len(entries)} rows PASS; successful completion marker: {all(r.get('_complete', False) for r in entries)}.",
            "",
        ]
        data = []
        for row in entries:
            if row.get("model") not in models or row.get("name") not in {
                s["name"] for s in models[row["model"]]
            }:
                continue
            data.append(
                [
                    row["model"],
                    row["name"],
                    row["N"],
                    row["K"],
                    row["M"],
                    "CuTe" if row.get("provider") == "cute" else "cuBLAS",
                    *(
                        speedup_cell(row) if f == "paired_speedup" else row.get(f)
                        for f in (
                            "kernels_per_call",
                            "ours_us",
                            "cublas_us",
                            "paired_speedup",
                            "epsilon",
                        )
                    ),
                    "PASS" if row.get("verdict") == "PASS" else "FAIL",
                ]
            )
        output += grid(
            [
                "Model",
                "Projection",
                "N",
                "K",
                "M",
                "Provider",
                "Kernels/call",
                "Dispatched us",
                "cuBLAS us",
                "Paired speedup",
                "epsilon",
                "Verdict",
            ],
            data,
        )

    output += [
        "## Decode-step GEMM estimates",
        "",
        "Sum of isolated median GEMM times times calls/step; excludes attention, recurrent kernels, scheduling and communication. Fused and split GDN layouts are alternatives. Delta = step_ms(B) - step_ms(1). This is not measured engine latency.",
        "",
    ]
    for kind, label in DEVICES.items():
        entries = [r for r in acceptance.get(kind, []) if r.get("kind") == "smallm_step"]
        output += [f"### {label}", ""]
        if not entries:
            output += ["Pending: no decode-step RESULT rows supplied.", ""]
            continue
        output += [entries[0].get("_round", "supplied round") + ".", ""]
        indexed = {(r.get("model"), r.get("layout"), r.get("M")): r for r in entries}
        data = []
        for model in models:
            for layout in ("fused", "split"):
                for m in sorted(set(MS) | {r["M"] for r in entries if isinstance(r.get("M"), int)}):
                    row = indexed.get((model, layout, m), {})
                    data.append(
                        [
                            model,
                            layout,
                            m,
                            *(
                                row.get(f)
                                for f in (
                                    "calls_per_step",
                                    "ours_step_ms",
                                    "cublas_step_ms",
                                    "ours_delta_ms",
                                    "cublas_delta_ms",
                                    "acceptance_passed",
                                )
                            ),
                        ]
                    )
        output += grid(
            [
                "Model",
                "Layout",
                "B",
                "Calls/step",
                "Dispatched ms",
                "cuBLAS ms",
                "Dispatched delta ms",
                "cuBLAS delta ms",
                "All rows PASS",
            ],
            data,
        )
    output += [
        "## B x Q round GEMM estimates",
        "",
        "Isolated sums; DFlash context projections use B rows, block projections B*Q. Draft totals omit the shared target lm_head; all totals exclude non-GEMM work.",
        "",
    ]
    for kind, label in DEVICES.items():
        entries = [
            r
            for r in acceptance.get(kind, [])
            if r.get("kind") == "smallm_round" and r.get("model") in models
        ]
        output += [f"### {label}", ""]
        if not entries:
            output += ["Pending: no round RESULT rows supplied.", ""]
            continue
        output += grid(
            [
                "Model",
                "Layout",
                "B",
                "Q",
                "M",
                "Context rows",
                "Calls/round",
                "Dispatched ms",
                "cuBLAS ms",
                "All rows PASS",
            ],
            [
                [
                    r.get(key)
                    for key in (
                        "model",
                        "layout",
                        "B",
                        "Q",
                        "M",
                        "context_rows",
                        "calls_per_round",
                        "ours_round_ms",
                        "cublas_round_ms",
                        "acceptance_passed",
                    )
                ]
                for r in entries
            ],
        )
    return "\n".join(output).rstrip()


def dispatched_rows(rows, tuned):
    """Current acceptance rows resolved against a dispatch table.

    A CuTe row counts as dispatched only if ``tuned`` holds the configuration it
    measured; other rows run F.linear, so their time is the F.linear time.
    """
    result = []
    for kind, entries in resolved_acceptance(rows).items():
        for source in entries:
            # Undecided rows of an unfinished run are not acceptance evidence.
            if not decided(source):
                continue
            # One label per class so round sums never split across host variants.
            row = dict(source, device_class=kind, device=DEVICES[kind])
            ranges = tuned.get(kind, {}).get((row["N"], row["K"]), {})
            cfg = next((c for (lo, hi), c in ranges.items() if lo <= row["M"] <= hi), None)
            identity = launch_identity(row) if row.get("provider") == "cute" else None
            if identity is not None and identity[0] == cfg:
                row["dispatched"] = "CuTe"
            else:
                row["dispatched"] = "F.linear"
                if row.get("provider") == "cute":
                    row["measured_cute"] = row["paired_speedup"]
                    row["measured_cute_interval"] = speedup_cell(row, 3)
                    row["ours_us"], row["paired_speedup"] = row["cublas_us"], 1.0
                    row.pop("ci_low", None)
                    row.pop("ci_high", None)
            result.append(row)
    return result


def bxq_markdown(rows, tuned, inventory=None):
    """Every acceptance row (device x shape x M) and every B x Q round sum."""
    from benchmark.kernel_tuning.linear_smallm import round_estimates

    resolved = dispatched_rows(rows, tuned)
    output = [
        "# Small-M B x Q acceptance (SM2)",
        "",
        "Generated by `linear_smallm_table.py --bxq-markdown` from the current acceptance logs "
        "(at least 20 rotating-weight pairs per row; epsilon = p95 of F.linear self-pairs). "
        "GB10 requires at least 60 pairs and a bootstrap 95% CI lower bound > 1 in every "
        "cold, engine and hot record at both bucket edges. Comparator: "
        "F.linear (cuBLAS), the kernel vLLM 0.22.1 and SGLang dispatch for these BF16 linears on "
        "sm120/sm121, timed with torch's default reduced-precision BF16 reduction (as both "
        "stacks and the engine run it). CuTe is dispatched only where the measured "
        "configuration reached paired speedup >= 1.03 above its epsilon at every measured M "
        "of its M bucket, both bucket edges included; "
        "every other row runs F.linear (speedup 1 by "
        "construction); a row whose measured CuTe configuration is not in the table is "
        "listed as an F.linear fallback with the CuTe ratio it measured. SOL = 2(NK + MK + MN) bytes / read-only DRAM ceiling "
        f"({' / '.join(str(READ_GBPS[k]) for k in DEVICES)} GB/s); SOL % uses the dispatched time. "
        "GB10 (secondary) is measured at decode M only. Rows and round sums both use the "
        "cold-L2 (rotating-weight) timings, the regime the engine runs: a decode step or "
        "DFlash round streams every weight once, far beyond L2. Engine re-pairs "
        "(consecutive calls, weights alternating cold/hot in L2) must pass at both bucket "
        "edges; hot-L2 re-pairs are required for GB10 and only veto on RTX; neither is listed.",
        "",
    ]
    for kind, label in DEVICES.items():
        # One regime for rows and round sums: cold L2, as the engine runs.
        entries = [
            r for r in resolved if r["device_class"] == kind and r.get("l2", "cold") == "cold"
        ]
        output += [f"## {label}", ""]
        if not entries:
            output += ["Pending: no current acceptance rows.", ""]
            continue
        shapes = defaultdict(list)
        for row in entries:
            shapes[row["N"], row["K"], row["M"]].append(row)
        cute = [rs[0] for rs in shapes.values() if rs[0]["dispatched"] == "CuTe"]
        fails = sum(
            any(r.get("verdict") != "PASS" and "measured_cute" not in r for r in rs)
            for rs in shapes.values()
        )
        output += [
            f"{len(shapes)} rows, {fails} FAIL, {len(cute)} on CuTe"
            + (
                f" (speedup {min(r['paired_speedup'] for r in cute):.3f}-"
                f"{max(r['paired_speedup'] for r in cute):.3f})."
                if cute
                else "."
            ),
            "",
        ]
        data = []
        for (n, k, m), rs in sorted(shapes.items()):
            row = rs[0]
            names = sorted({f"{r['model']}:{r['name']}" for r in rs})
            data.append(
                [
                    n,
                    k,
                    m,
                    ", ".join(names),
                    row["dispatched"],
                    f"{row['ours_us']:.2f}",
                    f"{row['cublas_us']:.2f}",
                    speedup_cell(row, 3),
                    f"{row['epsilon']:.4f}",
                    f"{100 * row['sol_us'] / row['ours_us']:.1f}",
                    f"F.linear fallback (CuTe measured {row['measured_cute_interval']})"
                    if "measured_cute" in row
                    else "PASS"
                    if all(r.get("verdict") == "PASS" for r in rs)
                    else "FAIL",
                ]
            )
        output += grid(
            [
                "N",
                "K",
                "M",
                "Projections",
                "Dispatched",
                "Ours us",
                "F.linear us",
                "Speedup",
                "epsilon",
                "SOL %",
                "Verdict",
            ],
            data,
        )
        rounds = [r for r in round_estimates(entries, inventory) if r.get("layout") != "split"]
        if not rounds:
            continue
        output += [
            f"### {label}: GEMM time per DFlash round (ms, dispatched / F.linear)",
            "",
            "Isolated GEMM medians x calls; block projections use M = B x Q, context "
            "projections M = B; drafts exclude the shared lm_head; fused GDN layout.",
            "",
        ]
        data = []
        for r in sorted(rounds, key=lambda r: (r["model"], r["B"], r["Q"])):
            data.append(
                [
                    r["model"],
                    r["B"],
                    r["Q"],
                    r["M"],
                    f"{r['ours_round_ms']:.3f}",
                    f"{r['cublas_round_ms']:.3f}",
                    f"{r['cublas_round_ms'] / r['ours_round_ms']:.3f}",
                ]
            )
        output += grid(["Model", "B", "Q", "M", "Dispatched ms", "F.linear ms", "Ratio"], data)
    return "\n".join(output).rstrip() + "\n"


def select_gated_configs(rows, margin=0.03, notes=None):
    """Choose one configuration per bucket that wins at every observed (edge, L2 regime)."""
    from ctikernels.linear._dispatch import M_BUCKETS

    check_margin(margin)
    samples, observed, invalid = defaultdict(list), defaultdict(set), set()
    for row in rows:
        if row.get("kind") != "gate_probe" or device_kind(row) is None or not row.get("_finished"):
            continue
        m = row.get("M")
        bucket = next(((lo, hi) for lo, hi in M_BUCKETS if lo <= m <= hi), None)
        if bucket is None or m not in bucket:
            continue
        group = device_kind(row), *shape_key(row, True), bucket
        point = m, row.get("l2", "cold")
        observed[group].add(point)
        identity = launch_identity(row, True)
        if identity is None:
            continue
        cfg = identity[0]
        key = *group, cfg, point
        if row.get("correct") is False:
            invalid.add(key)
        if row.get("correct") is not True or row.get("provider") != "cute-gated":
            continue
        ratio = row.get("paired_speedup", row.get("ratio"))
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("finite positive probe ratio required")
        samples[key].append(ratio)
    best = {}
    for (*parts, cfg, _point), _values in samples.items():
        group = tuple(parts)
        keys = [(*group, cfg, point) for point in observed[group]]
        if any(key in invalid or not samples.get(key) for key in keys):
            continue
        ratios = [statistics.median(samples[key]) for key in keys]
        if min(ratios) < 1 + margin:
            continue
        rank = (-min(ratios), -statistics.mean(ratios), astuple(cfg))
        if group not in best or rank < best[group][0]:
            best[group] = rank, cfg
    entries = {}
    for (kind, *shape, bucket), (_, cfg) in best.items():
        edges = {m for m, _l2 in observed[(kind, *shape, bucket)]}
        if len(edges) == 1 and bucket[0] != bucket[1] and notes is not None:
            notes.append(
                f"{kind} shape={tuple(shape)} bucket={bucket}: only edge M={next(iter(edges))} probed"
            )
        for m in range(bucket[0], bucket[1] + 1):
            entries[(kind, *shape, m)] = cfg
    return _compress(entries)


def gated_markdown(rows):
    """Every measured device x shape x M row; unfused rows are the market path, untimed."""
    from benchmark.kernel_tuning.linear_gate import builtin_shapes

    inventory = {s["model"]: s for s in builtin_shapes().values()}
    output = [
        "# Gated o_proj acceptance",
        "",
        "Market is the faster of vLLM (Inductor x*sigmoid(gate) + F.linear) and SGLang "
        "(fused_sigmoid_mul + F.linear), F.linear with torch's default flags. Times, "
        "epsilon and SOL are cold-L2 (rotating weights); engine and hot columns are the "
        "paired speedups of the same configuration in those regimes. Rows without a "
        "fused table entry keep the market path (gate kernel + Linear) and are not timed. "
        "Unfinished runs are omitted; totals are isolated o_proj times, not engine latency.",
        "",
    ]
    resolved = resolved_acceptance(rows, gated=True)
    for kind, label in DEVICES.items():
        regimes = defaultdict(dict)
        for row in resolved.get(kind, []):
            if decided(row) and row.get("_finished") and row.get("model") in inventory:
                regimes[row["model"], row["M"]][row.get("l2", "cold")] = row
        output += [f"## {label}", ""]
        if not regimes:
            output += ["Pending: no finished acceptance rows.", ""]
            continue
        data, totals = [], defaultdict(lambda: [0, 0.0, 0.0])
        for (model, m), by_l2 in sorted(regimes.items()):
            cold = by_l2.get("cold")
            if cold is None:
                continue
            shape = inventory[model]
            if cold.get("verdict") == "FALLBACK" or "ours_us" not in cold:
                data.append([model, *shape_key(cold, True), m, "market path", *["—"] * 12])
                continue

            def same_launch(row, cold=cold):
                # Only the same shape, gate stride and configuration is the same row.
                return (shape_key(row, True), launch_identity(row, True)) == (
                    shape_key(cold, True),
                    launch_identity(cold, True),
                )

            def speedup(l2, by_l2=by_l2):
                row = by_l2.get(l2)
                if row is None or not same_launch(row):
                    return "—"
                return speedup_cell(row) if "paired_speedup" in row else "—"

            data.append(
                [
                    model,
                    *shape_key(cold, True),
                    m,
                    cold["provider"],
                    cold["ours_us"],
                    f"{cold['vllm_us']:.4f} ({cold['vllm_mode']})",
                    cold["sglang_us"],
                    cold["faster_comparator"],
                    speedup_cell(cold),
                    cold["epsilon"],
                    speedup("engine"),
                    speedup("hot"),
                    100 * cold["sol_us"] / cold["ours_us"],
                    100 * cold["sol_us"] / cold["market_us"],
                    cold["kernels_per_call"],
                    "/".join(
                        by_l2[l2]["verdict"] if same_launch(by_l2[l2]) else "other launch"
                        for l2 in ("cold", "engine", "hot")
                        if l2 in by_l2
                    ),
                ]
            )
            if m in (1, 2, 3, 4) and shape_key(cold, True) == shape_key(shape, True):
                count = shape["count_per_step"]
                totals[model, m] = [
                    count,
                    count * cold["ours_us"] / 1000,
                    count * cold["market_us"] / 1000,
                ]
        output += grid(
            [
                "Model",
                "N",
                "K",
                "Gate stride",
                "M",
                "Dispatched",
                "Ours us",
                "vLLM us (mode)",
                "SGLang us",
                "Faster",
                "Speedup cold",
                "epsilon",
                "Speedup engine",
                "Speedup hot",
                "SOL % ours",
                "SOL % market",
                "Kernels/call",
                "Verdict c/e/h",
            ],
            data,
        )
        if totals:
            output += ["### o_proj per-step totals (cold, fused rows)", ""]
            output += grid(
                ["Model", "M", "Calls/step", "Ours ms", "Market ms"],
                [[model, m, *values] for (model, m), values in sorted(totals.items())],
            )
    return "\n".join(output).rstrip() + "\n"


def gated_main(args, rows, ap):
    from ctikernels.linear import _dispatch

    try:
        check_margin(args.margin)
    except ValueError as exc:
        ap.error(str(exc))
    if args.markdown:
        print(gated_markdown(rows), end="")
        return 0
    if args.bxq_markdown:
        ap.error("use --gated --markdown")
    if args.candidate:
        if not any(row.get("kind") == "gate_probe" for row in rows):
            ap.error("--gated --candidates requires gate_probe rows")
        notes = []
        table = select_gated_configs(rows, args.margin, notes)
        for note in notes:
            print(note, file=sys.stderr)
    else:
        if not any(is_current_acceptance(row, True) for row in rows):
            ap.error("--gated requires current gate_accept rows (--accept)")
        refused = []
        table = gate_by_acceptance(
            getattr(_dispatch, "TUNED_GATED", {}), rows, args.margin, refused, gated=True
        )
        for kind, n, k, stride, m, cfg in refused:
            print(
                f"refused {kind} N={n} K={k} gate_stride={stride} M={m} {astuple(cfg)}: "
                "no current acceptance record measured this configuration",
                file=sys.stderr,
            )
    if args.rewrite:
        rewrite(args.rewrite, table, gated=True)
    else:
        print(python_literal(table))
    return 0


def fp32_markdown(rows):
    """FP32 evidence report without the BF16/Qwen-specific report template."""
    output = [
        "# Linear FP32 output",
        "",
        "BF16 inputs; comparator: `torch.mm(x, weight.t(), out_dtype=torch.float32)` "
        "with torch default flags. Accuracy uses FP64 references and the full-precision "
        "cuBLAS flag, with FP32 slack only (no BF16 output rounding).",
        "",
    ]
    for kind, label in DEVICES.items():
        entries = [r for r in resolved_acceptance(rows).get(kind, []) if decided(r)]
        output += [f"## {label}", ""]
        output += grid(
            ["N", "K", "M", "L2", "Output", "Provider", "us", "cuBLAS/us", "Correct"],
            [
                [
                    r["N"],
                    r["K"],
                    r["M"],
                    r.get("l2", "cold"),
                    "fp32",
                    "CuTe" if r.get("provider") == "cute" else "cuBLAS",
                    cell(r.get("ours_us")),
                    speedup_cell(r),
                    str(r.get("correct")),
                ]
                for r in entries
            ],
        )
        output += [""]
    return "\n".join(output)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--gated", action="store_true")
    for group in ("market", "probe", "accept", "previous-accept"):
        ap.add_argument("--" + group, nargs="*", default=[], type=Path)

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--rewrite", nargs="?", const="python/ctikernels/linear/_dispatch.py", type=Path
    )
    mode.add_argument("--markdown", action="store_true")
    mode.add_argument(
        "--bxq-markdown",
        action="store_true",
        help="every current acceptance row and B x Q round sum against the committed table",
    )
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument(
        "--candidate",
        "--candidates",
        action="store_true",
        help="select from --probe only, without acceptance; the result must then pass "
        "a shipped-dispatch --accept round and a re-gating --rewrite before commit",
    )
    ap.add_argument(
        "--bucketize", action="store_true", help="expand merged candidates to whole M buckets"
    )
    ap.add_argument(
        "--only-measured",
        action="store_true",
        help="rewrite only device/N/K pairs in the supplied probes or completed acceptance logs",
    )
    args = ap.parse_args(argv)
    if args.only_measured and (not args.rewrite or args.gated):
        ap.error("--only-measured requires an ungated --rewrite")
    if args.bucketize and not args.candidate:
        ap.error("--bucketize requires --candidate")
    rows = read_logs(args.logs)
    for group, label in (
        ("market", "current round"),
        ("probe", "supplied probes"),
        ("previous_accept", "previous round"),
        ("accept", "current round"),
    ):
        paths = getattr(args, group)
        rows.extend(
            read_logs(
                paths,
                optional=True,
                round_label=label,
                source_offset=1 + max((r["_source"] for r in rows), default=-1),
            )
        )
    if args.gated:
        return gated_main(args, rows, ap)
    dtypes = sorted(
        {
            row.get("out_dtype", "bf16")
            for row in rows
            if row.get("kind") in ("smallm_probe", "smallm_accept", "market") or "us_median" in row
        }
    ) or ["bf16"]
    for out_dtype in dtypes:
        _dtype_main(args, dtype_rows(rows, out_dtype), ap, out_dtype)
    return 0


def _dtype_main(args, rows, ap, out_dtype):
    measured = None
    if args.only_measured:
        measured = {
            (device_kind(row), row["N"], row["K"])
            for row in rows
            if device_kind(row) is not None
            and (
                row.get("kind") == "smallm_probe"
                if args.candidate
                else is_current_acceptance(row) and row.get("_finished")
            )
        }
    if out_dtype == "fp32" and (args.markdown or args.bxq_markdown):
        print(fp32_markdown(rows))
        return 0
    if args.markdown:
        print(f"# Output dtype: {out_dtype}\n\n" + markdown(rows))
        return 0
    if args.bxq_markdown:
        from ctikernels.linear._dispatch import TUNED, TUNED_F32

        base = TUNED_F32 if out_dtype == "fp32" else TUNED

        print(f"# Output dtype: {out_dtype}\n\n" + bxq_markdown(rows, base), end="")
        return 0
    try:
        check_margin(args.margin)
    except ValueError as exc:
        ap.error(str(exc))
    if args.candidate:
        if not any(row.get("kind") == "smallm_probe" for row in rows):
            ap.error("--candidate requires --probe RESULT rows")
        from ctikernels.linear._dispatch import TUNED, TUNED_F32

        base = TUNED_F32 if out_dtype == "fp32" else TUNED

        table = merge_candidates(base, select_configs(rows, args.margin, out_dtype=out_dtype), rows)
        if args.bucketize:
            table = bucketize(table)
        if args.rewrite:
            rewrite(args.rewrite, table, only_measured=measured, out_dtype=out_dtype)
        else:
            print(python_literal(table))
        return 0
    if not any(is_current_acceptance(row) for row in rows):
        ap.error("dispatch generation requires current acceptance RESULT rows (--accept)")
    if any(row.get("kind") == "smallm_probe" for row in rows):
        table = select_configs(rows, args.margin, out_dtype=out_dtype)
    else:
        # Re-gate the committed table against new acceptance without re-probing.
        from ctikernels.linear._dispatch import TUNED, TUNED_F32

        base = TUNED_F32 if out_dtype == "fp32" else TUNED

        table = base
    refused = []
    table = gate_by_acceptance(table, rows, args.margin, refused, out_dtype=out_dtype)
    for kind, n, k, m, cfg in refused:
        print(
            f"refused {kind} N={n} K={k} M={m} {astuple(cfg)}: "
            "no current acceptance record measured this configuration",
            file=sys.stderr,
        )
    if args.rewrite:
        rewrite(args.rewrite, table, only_measured=measured, out_dtype=out_dtype)
    else:
        print(python_literal(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
