# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only per-kernel GDN SOL.

Uses the current product's host dispatch/launch definitions without importing
ctikernels, torch or CUDA. Default tuning only; no capture or forced variants.
The shared benchmark.sol.sol helper and ceilings are vendored under
benchmark/sol (see its PROVENANCE.md); --helper-root PATH selects another
copy. Run --self-test for all 102 frozen rows.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICES = ("rtx5090", "rtx-pro-6000", "gb10")
DEVICE_NAMES = {"rtx5090": "RTX 5090", "rtx-pro-6000": "RTX PRO 6000", "gb10": "GB10"}
HELPER_ROOT = ROOT
MMA = "bf16_m16n8k16"


@lru_cache(maxsize=None)
def shared_helper(root):
    """Import the shared helper from the selected tree, including an existing package."""
    sys.path.insert(0, str(root))
    # Keep an external helper worktree read-only, including its bytecode caches.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import benchmark

        location = str(root / "benchmark")
        if location not in benchmark.__path__:
            benchmark.__path__.insert(0, location)
        import benchmark.sol.sol as sol
    finally:
        sys.dont_write_bytecode = previous

    if Path(sol.__file__).resolve() != (root / "benchmark/sol/sol.py").resolve():
        raise ValueError("another SOL helper is already imported; use a fresh process")
    return sol


def ceilings(device):
    return shared_helper(HELPER_ROOT).load_ceilings(
        device, directory=HELPER_ROOT / "benchmark/sol/ceilings"
    )


@lru_cache(maxsize=1)
def host_dispatch():
    """Extract only our pure host definitions, never execute the module's imports.

    The allowlist intentionally excludes read_tune and every device entry point.
    Reading the product source avoids a stale duplicated dispatch snapshot.
    """
    source = ROOT / "python/ctikernels/gdn/_cute_prefill.py"
    constants = {"CHUNK", "BV", "DEFAULT_TUNE", "GEOMETRY_TUNES"}
    functions = {
        "resolve_tune",
        "validate_output_split",
        "chunk_capacity",
        "pipeline_variant",
        "launch_info",
    }
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in constants
        )
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    if len(nodes) != len(constants) + len(functions):
        raise ValueError("product host dispatch contract changed")
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<GDN host dispatch>", "exec"), scope)
    return scope


K = V = 128
C = 64

SMALL = [  # (lengths, initial flags) - small batch first
    ([16], [False]),
    ([64], [False]),
    ([128], [True]),
    ([256], [False]),
    ([512], [False]),
    ([64] * 2, [True, False]),
    ([128] * 4, [True, False, True, False]),
    ([32, 96, 200, 184], [True, True, False, False]),
]

LONG = [
    ([1024], [False]),
    ([2048], [False]),
    ([4096], [False]),
    ([8192], [False]),
    ([4096] * 2, [False, True]),
    ([2048] * 4, [False] * 4),
    ([1024] * 8, [True, False] * 4),
    ([6000, 1500, 500, 192], [False] * 4),
    ([16384], [False]),
]


def rows(which="all"):
    indices = range(17) if which == "all" else range(8) if which == "small" else range(8, 17)
    return [
        {
            "hk": 16,
            "hv": hv,
            "case": i,
            "lengths": list((SMALL + LONG)[i][0]),
            "initial": list((SMALL + LONG)[i][1]),
        }
        for hv in (32, 48)
        for i in indices
    ]


def compulsory_formula(hk, hv, lengths, initial):
    t, n, i = sum(lengths), len(lengths), sum(initial)
    q = 2 * hk * K + hv * V
    return (
        2 * t * q
        + 4 * t * hv
        + 4 * t * hv * V
        + 6 * (i + n) * q
        + 4 * (i + n) * hv * K * V
        + 8 * q
        + 4 * (2 * hv + V)
        + 4 * (n + 1)
        + 5 * n
    )


def build_call(device, hk, hv, lengths, initial):
    """Return a sol.Call plus an auditable per-buffer/category byte ledger.

    Only default dispatched WY/legacy configs are supported, with positive lengths
    >=3 and one distinct state slot per sequence (the frozen parity contract).
    Requests are retained separately from unique producer/consumer transfers.
    CTA-local reloads and shared-head/V-tile rereads are served by L2 even when
    the full allocation exceeds cache. No device code is imported or executed.
    """
    if hk != 16 or hv not in (32, 48) or not lengths or len(lengths) != len(initial):
        raise ValueError("unsupported geometry or sequence contract")
    if any(type(x) is not int or x < 3 for x in lengths) or any(
        type(x) is not bool for x in initial
    ):
        raise ValueError("lengths must be integers >=3 and initial flags must be bools")
    sol = shared_helper(HELPER_ROOT)
    table = ceilings(device)
    t, n, ini = sum(lengths), len(lengths), sum(initial)
    nc = [(x + 63) // 64 for x in lengths]
    chunks = sum(nc)
    cap = n + (t - n) // 64
    cc = (12, 1) if device == "gb10" else (12, 0)
    host = host_dispatch()
    tune = host["DEFAULT_TUNE"] | host["GEOMETRY_TUNES"][(cc, (hk, hv))]
    tune["overlap"] = int(host["pipeline_variant"](t, tune) != "serial")
    one_wave = n * hv * (128 // tune["bv"]) <= table["sm_count"]
    tune = host["resolve_tune"](tune, hv, one_wave)
    overlap, grouped, aq_qk = bool(tune["overlap"]), bool(tune["prep_mode"]), tune["aq_qk"]
    tiles = 128 // tune["bv"]
    specs = host["launch_info"](hk, hv, t, n, tune, one_wave)
    if overlap:
        specs.insert(
            0,
            {
                "kernel": "flags_memset",
                "grid": None,
                "ctas": None,
                "threads": None,
                "dynamic_smem_bytes": None,
            },
        )
    ledger = {
        s["kernel"]: {
            "compulsory": Counter(),
            "compulsory_reads": Counter(),
            "compulsory_writes": Counter(),
            "repeated": Counter(),
            "intermediate_reads": Counter(),
            "intermediate_writes": Counter(),
            "useful_flops": 0,
            "padded_flops": 0,
            "executed_flops": 0,
        }
        for s in specs
    }
    key = "prepare_value" if grouped else "prepare_key"
    value, rec, norm = "prepare_value", "recurrence", "norm"

    def boundary(name, category, unique, total=None, *, write=False):
        ledger[name]["compulsory"][category] += unique
        direction = "compulsory_writes" if write else "compulsory_reads"
        ledger[name][direction][category] += unique
        ledger[name]["repeated"][category] += (unique if total is None else total) - unique

    def traffic(name, buffer, read=0, write=0):
        ledger[name]["intermediate_reads"][buffer] += read
        ledger[name]["intermediate_writes"][buffer] += write

    # Disjoint projected columns: Q/K in key, V+a/b in value, z in norm.
    # 3-row halos at every chunk boundary, plus final history staging reads.
    qchannels, vchannels = 2 * hk * K, hv * V
    for name, channels in ((key, qchannels), (value, vchannels)):
        boundary(
            name, "activations", 2 * t * channels, 2 * (t + 3 * (chunks - n) + 3 * n) * channels
        )
        boundary(name, "weights", 8 * channels, 8 * chunks * channels)
        boundary(name, "kv_state", 6 * ini * channels)
        traffic(name, "history_out", write=6 * n * channels)
    boundary(value, "activations", 4 * t * hv)
    boundary(value, "weights", 8 * hv, 8 * chunks * hv)
    boundary(norm, "activations", 2 * t * hv * V)
    boundary(norm, "weights", 4 * V, 4 * V * hv * ((t + 7) // 8))
    boundary(norm, "outputs", 2 * t * hv * V, write=True)
    boundary(rec, "kv_state", 4 * ini * hv * K * V)
    boundary(rec, "kv_state", 4 * n * hv * K * V, write=True)
    commit = norm if overlap else rec
    boundary(commit, "kv_state", 6 * n * (qchannels + vchannels), write=True)
    traffic(commit, "history_out", read=6 * n * (qchannels + vchannels))
    # Metadata is placed in activations (helper has no metadata category).
    metadata = 4 * (n + 1) + 5 * n
    first_prepare = key
    for s in specs:
        name = s["kernel"]
        if name.startswith("prepare"):
            heads = s["grid"][0] // cap
            total = heads * cap * 4 * (n + 1) + heads * chunks * 5
        elif name == rec:
            total = hv * tiles * sum(4 * (seq + 2) + 5 for seq in range(n))
        elif name == norm and overlap:
            total = hv * n * 12
        else:
            continue
        boundary(name, "activations", metadata if name == first_prepare else 0, total)

    # Allocation capacity differs from touched chunks on ragged rows.
    alloc = {
        "q": cap * hk * 8192 * 2 * (1 + aq_qk),
        "k": cap * hk * 8192 * 2,
        "kl": cap * hk * 8192 * 2 * aq_qk,
        "kk": cap * hk * 4096 * 4,
        "qk": cap * hk * 4096 * 4,
        "w": cap * hv * 8192 * 2,
        "u": cap * hv * 8192 * 4,
        "aq": cap * hv * 4096 * 2,
        "g": cap * hv * 64 * 4,
        "raw_out": t * hv * 128 * 4,
        "partial": t * hv * 4 * 4,
        "history_out": 6 * n * (qchannels + vchannels),
        "flags": cap * hv * 4 if overlap else 0,
    }
    qbytes = chunks * hk * 8192 * 2
    for b in ("q", "k"):
        traffic(key, b, write=qbytes * (1 + aq_qk if b == "q" else 1))
    # key_body reads Q_hi for Gram; precision correction reloads Q_hi/Q_lo/K_hi/K_lo.
    traffic(key, "q", read=qbytes * (1 + 2 * aq_qk))
    traffic(key, "k", read=qbytes * aq_qk)
    traffic(key, "kl", read=qbytes * aq_qk, write=qbytes * aq_qk)
    for b in ("kk", "qk"):
        traffic(key, b, write=chunks * hk * 4096 * 4)
        traffic(value, b, read=chunks * hv * 4096 * 4)
    traffic(value, "k", read=chunks * hv * 8192 * 2)
    for b, per_head in (("w", 8192 * 2), ("u", 8192 * 4), ("aq", 4096 * 2), ("g", 64 * 4)):
        traffic(value, b, write=chunks * hv * per_head)
        traffic(rec, b, read=chunks * hv * per_head * (1 if b == "u" else tiles))
    for b in ("q", "k"):
        traffic(rec, b, read=chunks * hv * tiles * 8192 * 2)
    for b in ("raw_out", "partial"):
        traffic(rec, b, write=alloc[b])
        traffic(norm, b, read=alloc[b])
    if overlap:
        traffic("flags_memset", "flags", write=alloc["flags"])
        traffic(value, "flags", write=chunks * hv * 4)
        traffic(
            rec, "flags", read=chunks * hv * tiles * 4
        )  # one successful acquire; retries excluded

    # Unique interfaces, not the sum of CTA requests or the whole-call allocation.
    # Q_lo/K_lo and grouped KK/QK reloads are local to their producer CTA.
    # Q_hi/K_hi are shared across value heads and V tiles.
    interfaces = {}

    def transfer(producer, consumer, buffer, size):
        if producer != consumer:
            interfaces.setdefault((producer, consumer), Counter())[buffer] += size

    transfer(key, value, "k", qbytes)
    for b in ("kk", "qk"):
        transfer(key, value, b, chunks * hk * 4096 * 4)
    for b in ("q", "k"):
        transfer(key, rec, b, qbytes)
    for b in ("w", "u", "aq", "g"):
        transfer(value, rec, b, ledger[value]["intermediate_writes"][b])
    for name, channels in ((key, qchannels), (value, vchannels)):
        transfer(name, commit, "history_out", 6 * n * channels)
    for b in ("raw_out", "partial"):
        transfer(rec, norm, b, alloc[b])
    if overlap:
        transfer(value, rec, "flags", chunks * hv * 4)

    edges = []
    consumers = Counter(b for buffers in interfaces.values() for b in buffers)
    for item in ledger.values():
        item["dram_intermediate_reads"] = Counter()
        item["dram_intermediate_writes"] = Counter()
    for (producer, consumer), buffers in interfaces.items():
        unique = sum(buffers.values())
        # PDL permits consumption before the next chunk is prepared. Model an
        # optimistic prefetch window per sequence, not all chunks in the call.
        streaming = overlap and producer == value and consumer == rec
        live_chunks = min(chunks, n * tune["prefetch_depth"]) if streaming else chunks
        working_set = unique * live_chunks / chunks
        # Even a serial interface larger than L2 need not spill its resident tail.
        # Retain up to L2; only the excess needs a producer write + consumer read.
        # For streaming, the cache budget is reused for each window. Prioritize
        # buffers shared by interfaces (K_hi), so its producer write is not doubled.
        budget = min(unique, table["l2_bytes"] * chunks // live_chunks)
        spilled = {}
        for b in sorted(buffers, key=lambda b: (-consumers[b], b)):
            retained = min(budget, buffers[b])
            budget -= retained
            spilled[b] = buffers[b] - retained
            ledger[consumer]["dram_intermediate_reads"][b] += spilled[b]
            writes = ledger[producer]["dram_intermediate_writes"]
            writes[b] = max(writes[b], spilled[b])
        edges.append(
            {
                "producer": producer,
                "consumer": consumer,
                "unique_bytes": dict(buffers),
                "working_set_bytes": working_set,
                "fits_l2": working_set <= table["l2_bytes"],
                "streaming": streaming,
                "live_chunks": live_chunks,
                "dram_bytes": spilled,
            }
        )

    cs = [min(64, length - offset) for length in lengths for offset in range(0, length, 64)]
    for c in cs:
        # Useful arithmetic counts mathematical work once (no correction planes).
        # Diagonal inverse blocks are scalar; only off-diagonal work is BF16 MMA.
        blocks = [min(16, c - offset) for offset in range(0, c, 16)]
        inverse = c * (c - 1) * (c - 2) // 3 - sum(b * (b - 1) * (b - 2) // 3 for b in blocks)
        ledger[key]["useful_flops"] += 2 * hk * K * c * c
        ledger[value]["useful_flops"] += hv * (c * (c + 1) * (K + V) + inverse)
        ledger[rec]["useful_flops"] += hv * (6 * c * K * V + c * (c + 1) * V)
        # 16 off-diagonal 16^3 products, each implemented with four BF16 planes.
        ledger[key]["padded_flops"] += hk * 4 * C * C * K
        ledger[value]["padded_flops"] += hv * (4 * C * C * K + 16 * 2 * 16**3)
        ledger[rec]["padded_flops"] += hv * (6 * C * K * V + 2 * C * C * V)
        ledger[key]["executed_flops"] += hk * 2 * (2 + 2 * aq_qk) * C * C * K
        ledger[value]["executed_flops"] += hv * (6 * C * C * K + 4 * 16 * 2 * 16**3)
        ledger[rec]["executed_flops"] += hv * (6 * C * K * V + 2 * C * C * V)

    kernels = []
    for s in specs:
        name = s["kernel"]
        item = ledger[name]
        cats = item["compulsory"] + item["repeated"]
        cats["intermediates"] = sum(item["intermediate_reads"].values()) + sum(
            item["intermediate_writes"].values()
        )
        resident = Counter(item["repeated"])
        resident["intermediates"] = (
            cats["intermediates"]
            - sum(item["dram_intermediate_reads"].values())
            - sum(item["dram_intermediate_writes"].values())
        )
        dram_reads = sum(item["compulsory_reads"].values()) + sum(
            item["dram_intermediate_reads"].values()
        )
        dram_writes = sum(item["compulsory_writes"].values()) + sum(
            item["dram_intermediate_writes"].values()
        )
        item["dram_read_bytes"], item["dram_write_bytes"] = dram_reads, dram_writes
        direction = "write" if not dram_reads else "read" if not dram_writes else "copy"
        kernels.append(
            sol.Kernel(
                name,
                dict(cats),
                flops={MMA: item["useful_flops"]},
                executed_flops={MMA: item["executed_flops"]},
                n_ctas=s["ctas"] or 1,
                cta_threads=s["threads"] or 128,
                cta_smem_bytes=s["dynamic_smem_bytes"] or 0,
                pdl=overlap and name in (value, rec) and name != first_prepare,
                dram_mode=direction,
                cache_state="app",
                l2_resident_bytes={cat: size for cat, size in resident.items() if size},
            )
        )
    compulsory = sum(sum(x["compulsory"].values()) for x in ledger.values())
    repeated = sum(sum(x["repeated"].values()) for x in ledger.values())
    assert compulsory == compulsory_formula(hk, hv, lengths, initial)
    assert all(x >= 0 for item in ledger.values() for x in item["repeated"].values())
    # Capacity condition includes all scratch, input/output, used state slots, weights, metadata.
    external_footprint = (
        2 * t * ((2 * hk + 2 * hv) * 128 + 2 * hv)
        + 2 * t * hv * 128
        + 4 * n * hv * K * V
        + 6 * n * (qchannels + vchannels)
        + 8 * (qchannels + vchannels)
        + 8 * hv
        + 4 * V
        + metadata
    )
    footprint = sum(alloc.values()) + external_footprint
    detail = {
        "tune": tune,
        "launches": specs,
        "ledger": ledger,
        "producer_consumer_interfaces": edges,
        "allocations_bytes": alloc,
        "actual_chunks": chunks,
        "capacity_chunks": cap,
        "longest_chain_chunks": max(nc),
        "padded_tokens": chunks * 64,
        "useful_tokens": t,
        "compulsory_bytes": compulsory,
        "repeated_external_bytes": repeated,
        "external_footprint_bytes": external_footprint,
        "cache_footprint_bytes": footprint,
        "l2_capacity_bytes": table.get("l2_bytes"),
        "fits_l2": bool(table.get("l2_bytes") and footprint <= table["l2_bytes"]),
    }
    return sol.Call(kernels, mode="pdl_chain" if overlap else "serial"), detail


def overlap_summary(result, overlap, chunk_floor=None):
    times = {x.name: x.t_ref_us for x in result.per_kernel}
    if chunk_floor is not None:
        times["recurrence"] = max(times["recurrence"], chunk_floor)
    serial = sum(times.values())
    prep = sum(v for k, v in times.items() if k.startswith("prepare"))
    others = sum(
        v for k, v in times.items() if k not in ("prepare_key", "prepare_value", "recurrence")
    )
    ideal = max(prep, times["recurrence"]) + others if overlap else serial
    # All launches still pay the measured incremental node floor. Never subtract PDL floors.
    floors = sum(x.terms_us["floor"] for x in result.per_kernel)
    # Shared DRAM/L2/tensor engines cannot each deliver full capacity to two grids at once.
    resource = (
        others
        + max(
            sum(
                x.terms_us[r]
                for x in result.per_kernel
                if x.name.startswith("prepare") or x.name == "recurrence"
            )
            for r in ("dram", "l2", "tensor")
        )
        if overlap
        else serial
    )
    return {
        "serial_sum_us": serial,
        "ideal_overlap_us": ideal,
        "shared_resource_overlap_us": max(ideal, resource, floors),
        "sum_node_floors_us": floors,
    }


def analyze(device, row, ours_us=None, bw_model="large", chunk_latency_us=None):
    call, detail = build_call(device, row["hk"], row["hv"], row["lengths"], row["initial"])
    sol = shared_helper(HELPER_ROOT)
    table = ceilings(device)
    result = dict(
        device=device,
        geom=f"{row['hk']}x{row['hv']}",
        **row,
        **detail,
        kernels=[asdict(k) for k in call.kernels],
        scenarios={},
    )
    for cache in ("dram_only", "l2_aware"):
        kernels = call.kernels
        if cache == "dram_only":
            # Diagnostic request-volume scenario, explicitly NOT a lower bound.
            kernels = [
                replace(
                    k,
                    cache_state="cold",
                    l2_resident_bytes={},
                    dram_mode="write" if k.name == "flags_memset" else "copy",
                )
                for k in kernels
            ]
        both = sol.estimate_both(sol.Call(kernels, call.mode), table, bw_model=bw_model)
        scenario = {}
        for kind, estimate in both.items():
            entry = estimate.to_dict()
            # Reference the calibrated row; full measurements remain in the source tables.
            entry["assumptions"] = [
                a for a in entry["assumptions"] if "CTA capacity evidence:" not in a
            ]
            for breakdown in entry["per_kernel"]:
                evidence = breakdown["cta_capacity_evidence"]
                if evidence:
                    index = next(
                        i
                        for i, e in enumerate(table["scheduling"]["cta_capacity"])
                        if all(
                            e.get(k) == evidence.get(k)
                            for k in ("threads", "smem_bytes", "regs_per_thread", "cta_capacity")
                        )
                    )
                    breakdown["cta_capacity_evidence"] = {
                        k: evidence[k]
                        for k in (
                            "source",
                            "threads",
                            "smem_bytes",
                            "regs_per_thread",
                            "cta_capacity",
                            "theoretical_residency_per_sm",
                            "measured_max_concurrent",
                            "q_saturation",
                        )
                    }
                    breakdown["cta_capacity_evidence"]["measurement_ref"] = (
                        f"benchmark/sol/ceilings/{device}.json#/scheduling/cta_capacity/{index}"
                    )
            entry.update(overlap_summary(estimate, bool(detail["tune"]["overlap"])))
            if chunk_latency_us is not None:
                entry["with_assumed_chunk_latency"] = overlap_summary(
                    estimate,
                    bool(detail["tune"]["overlap"]),
                    detail["longest_chain_chunks"] * chunk_latency_us,
                )
            if ours_us is not None:
                entry["serial_utilization"] = asdict(estimate.utilization(ours_us))
                entry["overlap_utilization"] = entry["shared_resource_overlap_us"] / ours_us
                entry["overlap_above_one"] = entry["overlap_utilization"] > 1
            scenario[kind] = entry
        result["scenarios"][cache] = scenario
    # Primary: unique transfers + capacity-limited L2 reuse, resource-aware overlap.
    primary = result["scenarios"]["l2_aware"]["implementation"]
    result.update(
        sol_us=primary["shared_resource_overlap_us"],
        primary="l2_aware.implementation.shared_resource_overlap_us",
        chunk_latency={
            "assumed_us": chunk_latency_us,
            "chain_floor_us": None
            if chunk_latency_us is None
            else detail["longest_chain_chunks"] * chunk_latency_us,
            "expression": f"{detail['longest_chain_chunks']} * latency_per_chunk_us",
            "included_in_primary": False,
        },
        ours_us=ours_us,
        utilization=None if ours_us is None else primary["shared_resource_overlap_us"] / ours_us,
    )
    return result


def read_parity(path, device):
    """Read only known fields; chatter, malformed and unmatched records are ignored.

    Last matching successful median wins, permitting appended reruns. An explicit
    initial flag mismatch is rejected; absent flags use the frozen case contract.
    """
    found = {}
    aliases = {"rtx5090": ("5090",), "rtx-pro-6000": ("6000",), "gb10": ("gb10",)}
    contract = {(f"{r['hk']}x{r['hv']}", r["case"]): r for r in rows()}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("provider_ours", "product") != "product":
            continue
        gpu = str(record.get("gpu", record.get("device", ""))).lower()
        if not any(x in gpu for x in aliases[device]):
            continue
        key = (record.get("geom"), record.get("case"))
        if (
            key not in contract
            or record.get("lengths", record.get("lens")) != contract[key]["lengths"]
        ):
            continue
        if "initial" in record and record["initial"] != contract[key]["initial"]:
            continue
        median = record.get("ours_median_us")
        if median is None:
            values = record.get("ours_us")
            if isinstance(values, list):
                median = (
                    statistics.median(values)
                    if values
                    and all(
                        isinstance(x, (int, float))
                        and not isinstance(x, bool)
                        and math.isfinite(x)
                        and x > 0
                        for x in values
                    )
                    else None
                )
            else:
                median = values
        if (
            isinstance(median, (int, float))
            and not isinstance(median, bool)
            and math.isfinite(median)
            and median > 0
        ):
            found[key] = median
    return found


def markdown(results):
    lines = [
        "SOL in microseconds; primary = producer/consumer L2 reuse, executed MMA + CTA waves, shared-resource overlap.",
        "Per-kernel values are implementation estimates. Serial sum and ideal overlap are separate scenarios in JSON.",
        "DRAM-only counts all requests and is a diagnostic, not a lower bound; whole-call fits L2 is informational only.",
        "",
        "| device | geom | case | lengths | fits L2 | per launch µs (limit) | DRAM serial | L2 serial | ideal overlap | SOL | ours | SOL/ours |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        l2 = r["scenarios"]["l2_aware"]["implementation"]
        dram = r["scenarios"]["dram_only"]["implementation"]
        per = ", ".join(
            f"{k['name']} {k['t_ref_us']:.3f} ({k['limiting_resource'].replace('tensor', 'mma')})"
            for k in l2["per_kernel"]
        )
        measured = "—" if r["ours_us"] is None else f"{r['ours_us']:.3f}"
        util = "—" if r["utilization"] is None else f"{100 * r['utilization']:.2f}%"
        lines.append(
            f"| {DEVICE_NAMES[r['device']]} | {r['geom']} | {r['case']} | {r['lengths']} | {r['fits_l2']} | {per} | {dram['serial_sum_us']:.3f} | {l2['serial_sum_us']:.3f} | {l2['ideal_overlap_us']:.3f} | {r['sol_us']:.3f} | {measured} | {util} |"
        )
    return "\n".join(lines) + "\n"


def profile_comparisons(directory):
    """Compare every saved default profile; retain only allowlisted fields.

    Instrumented kernel durations are validation evidence, never model inputs.
    A missing stage or an unknown default row fails rather than hiding coverage.
    """
    contract = {f"{r['hk']}x{r['hv']}:" + "+".join(map(str, r["lengths"])): r for r in rows()}
    comparisons = []
    for device, filename in (
        ("rtx5090", "rtx5090"),
        ("rtx-pro-6000", "rtx-pro-6000"),
        ("gb10", "gb10"),
    ):
        count = 0
        for line in (Path(directory) / f"{filename}.log").read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("provider") != "ours":
                continue
            if record.get("tune", "") != "":
                continue
            row = contract[record["row"]]
            result = analyze(device, row)
            for kernel in result["scenarios"]["l2_aware"]["implementation"]["per_kernel"]:
                name = kernel["name"]
                if name == "flags_memset":
                    continue  # The saved top-kernel lists exclude memops.
                matches = [v for k, v in record["top"] if f"_{name}_" in k]
                if len(matches) != 1:
                    raise ValueError("profile must contain exactly one entry per stage")
                measured = matches[0]
                if (
                    isinstance(measured, bool)
                    or not isinstance(measured, (int, float))
                    or not math.isfinite(measured)
                    or measured <= 0
                ):
                    raise ValueError("invalid profile duration")
                comparisons.append(
                    {
                        "device": device,
                        "geom": result["geom"],
                        "case": row["case"],
                        "kernel": name,
                        "sol_us": kernel["t_ref_us"],
                        "measured_us": measured,
                        "ratio": kernel["t_ref_us"] / measured,
                    }
                )
            count += 1
        if not count:
            raise ValueError("no default profiles for a required device")
    return comparisons


def profile_markdown(comparisons):
    lines = [
        "| device | geom | case | kernel | SOL us | measured us | SOL/measured |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in comparisons:
        lines.append(
            f"| {DEVICE_NAMES[r['device']]} | {r['geom']} | {r['case']} | {r['kernel']} "
            f"| {r['sol_us']:.3f} | {r['measured_us']:.2f} | {100 * r['ratio']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def self_test():
    """Check conservation, dispatch boundaries, scratch traffic and helper scenarios."""
    count = 0
    for device in DEVICES:
        for row in rows():
            call, detail = build_call(device, row["hk"], row["hv"], row["lengths"], row["initial"])
            external = sum(
                sum(k.dram_bytes.values()) - k.dram_bytes["intermediates"] for k in call.kernels
            )
            old_d = compulsory_formula(row["hk"], row["hv"], row["lengths"], row["initial"])
            assert (
                external - detail["repeated_external_bytes"] == detail["compulsory_bytes"] == old_d
            )
            names = [k.name for k in call.kernels]
            threshold = 512 if device == "gb10" and row["hv"] == 32 else 2048
            overlap = sum(row["lengths"]) >= threshold and not (
                device != "gb10" and row["hv"] == 48
            )
            assert ("flags_memset" in names) == overlap
            assert ("prepare_key" not in names) == (overlap and row["hv"] == 32)
            assert call.kernels[-1].name == "norm"
            assert detail["actual_chunks"] <= detail["capacity_chunks"]
            assert detail["padded_tokens"] >= sum(row["lengths"])
            for k in call.kernels:
                assert k.executed_flops[MMA] >= k.flops[MMA]
            result = analyze(device, row)
            for scenario in result["scenarios"].values():
                for est in scenario.values():
                    assert (
                        abs(est["t_ref_us"] - sum(x["t_ref_us"] for x in est["per_kernel"])) < 1e-7
                    )
                    assert est["shared_resource_overlap_us"] >= est["ideal_overlap_us"]
                    assert est["shared_resource_overlap_us"] >= est["sum_node_floors_us"]
            for k in call.kernels:
                assert 0 <= sum(k.l2_resident_bytes.values()) <= sum(k.dram_bytes.values())
                item = detail["ledger"][k.name]
                assert item["dram_read_bytes"] + item["dram_write_bytes"] == (
                    sum(k.dram_bytes.values()) - sum(k.l2_resident_bytes.values())
                )
                for direction in ("reads", "writes"):
                    assert all(
                        0 <= size <= item[f"intermediate_{direction}"][buffer]
                        for buffer, size in item[f"dram_intermediate_{direction}"].items()
                    )
            for edge in detail["producer_consumer_interfaces"]:
                size = sum(edge["unique_bytes"].values())
                cache_budget = (
                    detail["l2_capacity_bytes"] * detail["actual_chunks"] // edge["live_chunks"]
                )
                assert sum(edge["dram_bytes"].values()) == max(0, size - cache_budget)
                for buffer, size in edge["dram_bytes"].items():
                    assert (
                        detail["ledger"][edge["producer"]]["dram_intermediate_writes"][buffer]
                        >= size
                    )
                    assert (
                        detail["ledger"][edge["consumer"]]["dram_intermediate_reads"][buffer]
                        >= size
                    )
            count += 1
    call, detail = build_call("rtx5090", 16, 32, [32, 96, 200, 184], [True, True, False, False])
    assert (detail["actual_chunks"], detail["capacity_chunks"]) == (10, 11)
    # Physical buffer references: W/A/Q/K are reread by four V tiles on BV32.
    assert detail["ledger"]["recurrence"]["intermediate_reads"]["w"] == 10 * 32 * 8192 * 2 * 4
    assert "flags_memset" not in detail["ledger"]
    _, overlapping = build_call("gb10", 16, 32, [32, 96, 200, 184], [True, True, False, False])
    assert overlapping["ledger"]["flags_memset"]["intermediate_writes"]["flags"] == 11 * 32 * 4
    # Boundary transitions and one-wave prefetch selection are independent contracts.
    for device in DEVICES:
        for hv in (32, 48):
            threshold = 512 if device == "gb10" and hv == 32 else 2048
            for tokens in (threshold - 1, threshold):
                _, d = build_call(device, 16, hv, [tokens], [False])
                expected = tokens >= threshold and not (device != "gb10" and hv == 48)
                assert bool(d["tune"]["overlap"]) == expected
                assert bool(d["tune"]["prep_mode"]) == (expected and hv == 32)
    _, d = build_call("rtx-pro-6000", 16, 32, [2048], [False])
    assert d["tune"]["prefetch_depth"] == 4
    _, d = build_call("rtx-pro-6000", 16, 32, [2048] * 4, [False] * 4)
    assert d["tune"]["prefetch_depth"] == 1
    _, d = build_call("gb10", 16, 48, [2048], [False])
    assert d["tune"]["bv"] == 64 and d["tune"]["prefetch_depth"] == 1
    short = analyze("rtx5090", rows()[0], ours_us=100, chunk_latency_us=1)
    assert short["utilization"] == short["sol_us"] / 100
    assert short["chunk_latency"]["chain_floor_us"] == 1
    assert short["padded_tokens"] == 64
    assert short["fits_l2"]
    assert (
        short["scenarios"]["l2_aware"]["implementation"]["t_ref_us"]
        < short["scenarios"]["dram_only"]["implementation"]["t_ref_us"]
    )
    # Full precision median, mixed device logs, wrong contract, chatter, reruns.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "parity.log"
        record = {
            "gpu": "NVIDIA GeForce RTX 5090",
            "geom": "16x32",
            "case": 0,
            "lens": [16],
            "provider_ours": "product",
            "ours_median_us": 12.3456789,
            "ours_us": [12, 13, 14],
        }
        path.write_text(
            "# chatter\n"
            + json.dumps(record)
            + "\n"
            + json.dumps(record | {"gpu": "NVIDIA GB10", "ours_median_us": 99})
            + "\n"
            + json.dumps(record | {"lens": [17], "ours_median_us": 88})
            + "\n"
        )
        assert read_parity(path, "rtx5090") == {("16x32", 0): 12.3456789}
        assert read_parity(path, "gb10") == {("16x32", 0): 99}
        path.write_text(
            json.dumps(record | {"ours_median_us": None, "ours_us": [12, 13, 14]}) + "\n"
        )
        assert read_parity(path, "rtx5090") == {("16x32", 0): 13}
        record["lengths"] = record.pop("lens")
        path.write_text(json.dumps(record) + "\n")
        assert read_parity(path, "rtx5090") == {("16x32", 0): 12.3456789}
    # Both helper bandwidth modes must work with the shipped direction curves.
    analyze("gb10", rows()[0], bw_model="curve")
    print(
        f"PASS: {count} rows; compulsory D, dispatch, padding, traffic, cache, overlap, joins, latency, curves"
    )


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Invalid arguments; use --help.\n")


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument("--device", choices=(*DEVICES, "all"), default="all")
    parser.add_argument("--helper-root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="validate saved default kernel profiles (<device>.log) on CPU",
    )
    parser.add_argument("--rows", choices=("small", "long", "all"), default="all")
    parser.add_argument("--parity-log", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bw-model", choices=("large", "curve"), default="large")
    parser.add_argument(
        "--chunk-latency-us", type=float, help="uncalibrated sensitivity, reported separately"
    )
    args = parser.parse_args()
    global HELPER_ROOT
    HELPER_ROOT = args.helper_root.resolve()
    if args.self_test:
        self_test()
        return
    if args.profile_dir:
        comparisons = profile_comparisons(args.profile_dir)
        md = profile_markdown(comparisons)
        if args.markdown:
            args.markdown.write_text(md)
        print(md, end="")
        if any(r["ratio"] > 1 for r in comparisons):
            sys.exit("Kernel SOL exceeds a saved default profile; fix the ledger.")
        return
    if args.chunk_latency_us is not None and (
        not math.isfinite(args.chunk_latency_us) or args.chunk_latency_us < 0
    ):
        parser.error("invalid latency")
    results = []
    for device in DEVICES if args.device == "all" else (args.device,):
        measured = read_parity(args.parity_log, device) if args.parity_log else {}
        results.extend(
            analyze(
                device,
                row,
                measured.get((f"{row['hk']}x{row['hv']}", row["case"])),
                args.bw_model,
                args.chunk_latency_us,
            )
            for row in rows(args.rows)
        )
    payload = {
        "helper_commit": "7e92a13",
        "bw_model": args.bw_model,
        "source_sha256": hashlib.sha256(
            (ROOT / "python/ctikernels/gdn/_cute_prefill.py").read_bytes()
        ).hexdigest(),
        "helper_sha256": hashlib.sha256(
            (HELPER_ROOT / "benchmark/sol/sol.py").read_bytes()
        ).hexdigest(),
        "ceiling_sha256": {
            d: hashlib.sha256(
                (HELPER_ROOT / f"benchmark/sol/ceilings/{d}.json").read_bytes()
            ).hexdigest()
            for d in DEVICES
        },
        "rows": results,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    md = markdown(results)
    if args.markdown:
        args.markdown.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        # No exception/argv dumps: caller-provided paths may contain credentials.
        sys.exit("SOL accounting failed; check input contract and ceiling coverage.")
