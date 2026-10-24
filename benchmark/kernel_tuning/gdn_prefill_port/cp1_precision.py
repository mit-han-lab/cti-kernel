#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Preview the frozen five-seed product precision gate with FP32 torch glue + chunk_ws.

Needs CuTe DSL 4.8.0. Add --with-fi (flashinfer-python 0.7.0.post1) for
FlashInfer.
Defaults to the four CI core cases; --all selects the complete precision matrix.
This is a numerical preview, not product integration or a performance benchmark:
q/k/v and o are BF16, glue is FP32, and y is BF16.
History and untouched pool rows are checked. No alternate schedules are swept.
Exit status is 1 if any chunk_ws gate fails; FI is diagnostic only.
"""

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import torch

from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120

# As in cp1_bench.py: support both the source tree and a flat script copy.
reference_root = Path(__file__).resolve().parents[3]
if not (reference_root / "tests/gpu/_gdn_reference.py").is_file():
    reference_root = Path.cwd()
sys.path.insert(0, str(reference_root))
from tests.gpu import _gdn_reference as reference  # noqa: E402
from tests.gpu._gdn_prefill_cases import (  # noqa: E402
    ABSOLUTE_FLOOR,
    GEOMETRIES,
    _fla_prefill,
    _max_error,
    _no_worse_than_fla,
    _rms_error,
    _same,
)
from tests.gpu.test_gdn_cute_prefill import (  # noqa: E402
    _REFERENCE_CACHE,
    LENGTHS,
    _precision_inputs,
    _precision_reference,
)
from tests.modules._gdn_chunk_ws_reference import fi_state, flashinfer_kernel  # noqa: E402

KINDS = ("ordinary", "slow", "heavy")
CORE = ("16,32,ordinary", "16,48,heavy")


def parse_cases(values):
    """HK,HV,KIND expands to both core lengths; :L1+L2 selects one packed batch."""
    cases = []
    for value in values:
        triple, separator, packed = value.partition(":")
        try:
            hk, hv, kind = triple.split(",")
            hk, hv = int(hk), int(hv)
            lengths = [int(length) for length in packed.split("+")] if separator else None
        except ValueError as error:
            raise ValueError(f"invalid case {value!r}; use HK,HV,KIND[:L1+L2]") from error
        if (hk, hv) not in GEOMETRIES or kind not in KINDS:
            raise ValueError(f"case {value!r} requires geometry in {GEOMETRIES}, kind in {KINDS}")
        # _precision_inputs has metadata for at most four sequences.
        if lengths is not None and (not 1 <= len(lengths) <= 4 or min(lengths) <= 0):
            raise ValueError(f"case {value!r} requires one to four positive lengths")
        for batch in [lengths] if lengths is not None else [LENGTHS[0], LENGTHS[-1]]:
            case = (hk, hv, kind, tuple(batch))
            if case not in cases:
                cases.append(case)
    return cases


def prepare(x, weights, hk, meta, conv_pool):
    """FP32 contract prologue; update only owned history rows, keeping raw BF16 bits."""
    hv, dc = weights.A_log.numel(), weights.conv_weight.shape[0]
    cu, slots, initial = (tensor.tolist() for tensor in meta)
    mixed = torch.empty((x.shape[0], dc), dtype=torch.float32, device=x.device)
    taps = weights.conv_weight.float()
    for n, (slot, read) in enumerate(zip(slots, initial, strict=True)):
        start, end = cu[n : n + 2]
        history = conv_pool[slot] if slot >= 0 and read else x.new_zeros(reference.CONV - 1, dc)
        window = torch.cat((history, x[start:end, :dc]))
        # Match the reference's four-tap reduction, with FP32 operands. No TF32 conv.
        windows = window.float().unfold(0, reference.CONV, 1)
        convolved = (windows * taps).sum(-1)
        mixed[start:end] = convolved * convolved.sigmoid()
        if slot >= 0:
            conv_pool[slot].copy_(window[-(reference.CONV - 1) :])
    q, k, v = mixed.split([hk * reference.K, hk * reference.K, hv * reference.V], -1)
    q, k = q.reshape(-1, hk, reference.K), k.reshape(-1, hk, reference.K)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + reference.EPS)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + reference.EPS)
    # Keep Hk heads: the kernel maps value head h to key head h // (Hv / Hk).
    # q is NOT pre-scaled; the recurrence receives K**-0.5 separately.
    q, k, v = (
        tensor.to(torch.bfloat16).contiguous() for tensor in (q, k, v.reshape(-1, hv, reference.V))
    )
    b, a = x[:, dc + hv * reference.V :].float().split([hv, hv], -1)
    dt = a + weights.dt_bias.float()
    # Exact softplus formula used by _gdn_reference (no threshold approximation).
    g = -weights.A_log.float().exp() * torch.logaddexp(dt, torch.zeros_like(dt))
    return q, k, v, g.contiguous(), b.sigmoid().contiguous()


def finish(o, x, weights):
    """The product returns BF16 y; norm and SiLU arithmetic here remain FP32."""
    dc = weights.conv_weight.shape[0]
    z = x[:, dc : dc + o.shape[1] * reference.V].float().reshape_as(o)
    o = o.float()
    y = (
        weights.norm_weight.float()
        * o
        * torch.rsqrt(o.square().mean(-1, keepdim=True) + reference.EPS)
        * (z * z.sigmoid())
    )
    return y.flatten(1).to(torch.bfloat16)


def errors(got, exact):
    return {
        "rms_abs": _rms_error(got, exact),
        "max_abs": _max_error(got, exact),
        "finite": bool(torch.isfinite(got).all()),
    }


def seed_gate(name, got, fla, exact):
    # Supplying a collection defers the largest-error check to the five-seed mean.
    try:
        _no_worse_than_fla(name, got, fla, exact, [])
    except AssertionError:
        return False
    return True


@torch.inference_mode()
def measure_seed(op, case, seed_index, fi):
    """Return only Python scalars; all FP64 GPU tensors die at this seed boundary."""
    hk, hv, kind, lengths = case
    weights, x, meta, base, pools, written = _precision_inputs(
        hk, hv, list(lengths), kind, seed_index
    )
    # This helper calls reference.prefill on the unpoisoned base, selecting written rows.
    # Unlike pytest sweeps, this standalone run never reuses a seed: evict CPU cache too.
    try:
        exact = _precision_reference(
            hk, hv, list(lengths), kind, seed_index, x, weights, meta, base, written
        )
    finally:
        _REFERENCE_CACHE.clear()
    before = [pool.clone() for pool in pools]
    fla_pools = [pool.clone() for pool in base]
    q, k, v, g, beta = prepare(x, weights, hk, meta, pools[1])
    cu, slots, initial = meta
    fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
    shared = {"fla": (fy, fla_pools[0][written])}
    if fi is not None:
        # FI has no initial flag or -1 ABI. Map stateless sequences to private scratch
        # rows, zero fresh rows, and retain the same [Hv,V,K] state layout and scale.
        init_fi, indices = fi_state(before[0], slots.tolist(), initial.tolist())
        pool_fi = init_fi.clone()
        out_fi = torch.empty_like(v)
        fi(
            out_fi,
            pool_fi,
            q,
            k,
            v,
            init_fi,
            g.exp(),
            beta,
            cu,
            reference.K**-0.5,
            state_indices=indices,
        )
        shared["fi"] = (finish(out_fi, x, weights), pool_fi[written])
    shared_errors = {
        quantity: {
            provider: errors(values[index], exact[index]) for provider, values in shared.items()
        }
        for index, quantity in enumerate(("y", "state"))
    }
    if fi is not None:
        for index, quantity in enumerate(("y", "state")):
            shared_errors[quantity]["fi"]["rms_pass"] = seed_gate(
                quantity, shared["fi"][index], shared["fla"][index], exact[index]
            )
    untouched = [slot for slot in range(pools[0].shape[0]) if slot not in written]
    pool = before[0].clone()
    o = torch.empty_like(v)
    op(q, k, v, g, beta, o, pool, slots, initial, cu, reference.K**-0.5)
    ours = (finish(o, x, weights), pool[written])
    result = {"seed_index": seed_index, "seed": 4200 + 10000 * seed_index}
    for index, quantity in enumerate(("y", "state")):
        result[quantity] = {
            **shared_errors[quantity],
            "ours": errors(ours[index], exact[index]),
        }
        result[quantity]["ours"]["rms_pass"] = seed_gate(
            quantity, ours[index], shared["fla"][index], exact[index]
        )
    result["history_pass"] = _same(pools[1][written], exact[2])
    result["untouched_pass"] = {
        "state": _same(pool[untouched], before[0][untouched]),
        "history": _same(pools[1][untouched], before[1][untouched]),
    }
    return result


def ratio(numerator, denominator):
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def aggregate(seeds, quantity, provider):
    ours = [seed[quantity][provider] for seed in seeds]
    fla = [seed[quantity]["fla"] for seed in seeds]
    ours_mean = sum(row["max_abs"] for row in ours) / len(ours)
    fla_mean = sum(row["max_abs"] for row in fla) / len(fla)
    ratios = [ratio(a["rms_abs"], b["rms_abs"]) for a, b in zip(ours, fla, strict=True)]
    rms_pass = all(row["rms_pass"] for row in ours)
    largest_pass = ours_mean <= fla_mean + ABSOLUTE_FLOOR
    return {
        "mean_largest_abs": ours_mean,
        "fla_mean_largest_abs": fla_mean,
        "mean_largest_ratio_vs_fla": ratio(ours_mean, fla_mean),
        "max_seed_rms_ratio_vs_fla": math.nan
        if any(math.isnan(r) for r in ratios)
        else max(ratios),
        "rms_pass": rms_pass,
        "mean_largest_pass": largest_pass,
        "pass": rms_pass and largest_pass,
    }


def emit(prefix, record):
    # Keep failure rows valid JSON too: nonfinite errors/ratios are explicit strings.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    print(prefix + " " + json.dumps(clean(record), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cases",
        nargs="+",
        default=CORE,
        metavar="HK,HV,KIND[:L1+L2]",
        help="default: 16,32,ordinary 16,48,heavy; omitted lengths expand to [16] and [1024]",
    )
    selection.add_argument("--all", action="store_true", help="all geometries, kinds and 8 LENGTHS")
    parser.add_argument(
        "--with-fi", action="store_true", help="also compare FlashInfer 0.7.0.post1"
    )
    args = parser.parse_args()
    try:
        cases = (
            [
                (hk, hv, kind, tuple(lengths))
                for hk, hv in GEOMETRIES
                for kind in KINDS
                for lengths in LENGTHS
            ]
            if args.all
            else parse_cases(args.cases)
        )
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        parser.error("this precision preview requires an sm120/sm121 CUDA GPU")
    fi = None
    if args.with_fi:
        try:
            version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version != "0.7.0.post1":
            parser.error("--with-fi requires flashinfer-python 0.7.0.post1; use WITH_FI070=1")
        fi = flashinfer_kernel()
        if fi is None:
            parser.error("FlashInfer delta_rule_prefill_dsl is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    ops, rows = {}, []
    for case in cases:
        hk, hv, kind, lengths = case
        if (hk, hv) not in ops:
            ops[hk, hv] = ChunkPrefillSm120(device, hk, hv)
        seeds = [measure_seed(ops[hk, hv], case, seed, fi) for seed in range(5)]
        gates = {
            provider: {
                quantity: aggregate(seeds, quantity, provider) for quantity in ("y", "state")
            }
            for provider in (("ours", "fi") if fi is not None else ("ours",))
        }
        history_pass = all(seed["history_pass"] for seed in seeds)
        untouched_pass = all(all(seed["untouched_pass"].values()) for seed in seeds)
        row = {
            "hk": hk,
            "hv": hv,
            "kind": kind,
            "lengths": lengths,
            "seeds": seeds,
            "gate": gates,
            "history_pass": history_pass,
            "untouched_pass": untouched_pass,
            "pass": all(gate["pass"] for gate in gates["ours"].values())
            and history_pass
            and untouched_pass,
        }
        emit("PREC_ROW", row)
        rows.append(row)
    passed = sum(row["pass"] for row in rows)
    summary = {
        "cases": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "seeds_per_case": 5,
        "with_fi": fi is not None,
        "absolute_floor": ABSOLUTE_FLOOR,
        "glue": "torch FP32; q/k/v/o/y BF16; state FP32",
        "history_checked": True,
        "untouched_checked": True,
        "deviations": ["torch glue + chunk_ws replaces product implementation; no schedule sweep"],
        "ratio_convention": "0/0=1; nonfinite values are strings",
        "failed_cases": [
            {key: row[key] for key in ("hk", "hv", "kind", "lengths")}
            for row in rows
            if not row["pass"]
        ],
    }
    if fi is not None:
        summary["fi_precision_pass"] = all(
            gate["pass"] for row in rows for gate in row["gate"]["fi"].values()
        )
    emit("PREC_SUMMARY", summary)

    return 0 if all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
