#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Preview the frozen five-seed precision gate for fused PortPrefill and product CuTe v1.

Both operations receive raw projected inputs and independent poisoned pools. Compare
with the frozen FP64 reference and fla-core: per-seed RMS and five-seed mean of the
largest absolute error, using the frozen ABSOLUTE_FLOOR. History and untouched pool
rows must match exactly (including unchanged NaNs). No torch prologue or norm glue.
Defaults to the four CI core cases; --all selects the complete precision matrix.
--ulp selects the ULP acceptance rows (five public seeds per regime).
--all also runs those rows and exhaustive triage regimes and tiny packed batches.
ULP uses raw error against FP64 and spacing in y's BF16 / state's FP32 dtype at
the reference: both max ULP and count above 4 ULP must be no worse than fla-core
on every seed, independently for y and state. v1 is reported and gated as well.
Slot -1 sequences compute stateless output and participate in the y gate.
The precision contract (user_contract) is also reported: ratio of five-seed mean RMS <= 1.05 and
ratio of five-seed worst maximum error <= 1.25, separately from the frozen gate.
Exit status is 1 if either provider fails either rule; no GPU timing is performed.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from ctikernels.gdn import GatedDeltaNetPrefill
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.chunk_ws.prefill import PortOptions, PortPrefill

# Support both the source tree and a flat script copy.
reference_root = Path(__file__).resolve().parents[3]
if not (reference_root / "tests/gpu/_gdn_reference.py").is_file():
    reference_root = Path.cwd()
sys.path.insert(0, str(reference_root))
from tests.gpu import _gdn_reference as reference  # noqa: E402
from tests.gpu._gdn_prefill_cases import (  # noqa: E402
    ABSOLUTE_FLOOR,
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

if __package__:
    from . import _ulp_cases
else:
    # A flat copy must include this sibling alongside the script.
    import _ulp_cases

GEOMETRIES = ((16, 32), (16, 48), (8, 16), (8, 24))
KINDS = ("ordinary", "slow", "heavy")
CORE = ("16,32,ordinary", "16,48,heavy")


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        # Never echo arbitrary argument values in diagnostics.
        self.exit(2, "Invalid arguments or unavailable CUDA/CuTe; use --help.\n")


def parse_port_options(value):
    """Validate typed constructor options offline."""
    options = {}
    for item in value.split(",") if value.strip() else ():
        key, separator, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not separator or not raw or key in options:
            raise ValueError("expected distinct k=v port options")
        if key in ("producer_registers", "prefetch", "cp_segments"):
            options[key] = int(raw)
        elif key in ("iket", "watchdog", "cp", "q_lookahead", "remap_y"):
            if raw.lower() not in ("true", "false", "0", "1"):
                raise ValueError("boolean port options require true/false or 0/1")
            options[key] = raw.lower() in ("true", "1")
        else:
            raise ValueError("unknown port option")
    # PortOptions validates register counts, prefetch, and boolean option types.
    PortOptions(**options)
    return options


def parse_geoms(values):
    """Accept space- or comma-separated HKxHV selections, preserving order."""
    geoms = []
    for value in values:
        for item in value.split(","):
            hk, hv = map(int, item.split("x"))
            if (hk, hv) not in GEOMETRIES:
                raise ValueError("unsupported geometry")
            if (hk, hv) not in geoms:
                geoms.append((hk, hv))
    if not geoms:
        raise ValueError("select at least one geometry")
    return geoms


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
    mean_rms = sum(row["rms_abs"] for row in ours) / len(ours)
    fla_mean_rms = sum(row["rms_abs"] for row in fla) / len(fla)
    max_abs = max(row["max_abs"] for row in ours)
    fla_max_abs = max(row["max_abs"] for row in fla)
    finite = all(row["finite"] for row in ours + fla)
    rms_ratio = ratio(mean_rms, fla_mean_rms)
    max_ratio = ratio(max_abs, fla_max_abs)
    return {
        "user_contract": {
            "mean_rms_abs": mean_rms,
            "fla_mean_rms_abs": fla_mean_rms,
            "max_abs": max_abs,
            "fla_max_abs": fla_max_abs,
            "rms_ratio_vs_fla": rms_ratio,
            "max_ratio_vs_fla": max_ratio,
            "rms_pass": finite and rms_ratio <= 1.05,
            "max_pass": finite and max_ratio <= 1.25,
            "pass": finite and rms_ratio <= 1.05 and max_ratio <= 1.25,
        },
        "mean_largest_abs": ours_mean,
        "fla_mean_largest_abs": fla_mean,
        "mean_largest_ratio_vs_fla": ratio(ours_mean, fla_mean),
        "max_seed_rms_ratio_vs_fla": math.nan
        if any(math.isnan(r) for r in ratios)
        else max(ratios),
        "rms_pass": rms_pass,
        "mean_largest_pass": largest_pass,
        "pass": finite and rms_pass and largest_pass,
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


@torch.inference_mode()
def measure_seed(ops, case, seed_index):
    """Return Python scalars only; release FP64 tensors and the cache after each seed."""
    hk, hv, kind, lengths = case
    weights, x, meta, base, pools, written = _precision_inputs(
        hk, hv, list(lengths), kind, seed_index
    )
    try:
        # Frozen helper calls tests/gpu/_gdn_reference.py on the unpoisoned base.
        exact = _precision_reference(
            hk, hv, list(lengths), kind, seed_index, x, weights, meta, base, written
        )
    finally:
        _REFERENCE_CACHE.clear()
    owned = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    excluded = 0
    exact_y = exact[0][owned]
    fla_pools = [pool.clone() for pool in base]
    fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
    fla = (fy[owned], fla_pools[0][written])
    references = (exact_y, exact[1])
    fla_errors = [errors(value, ref) for value, ref in zip(fla, references, strict=True)]
    untouched = [slot for slot in range(pools[0].shape[0]) if slot not in written]
    results = {}
    for provider, op in ops.items():
        candidate_pools = [pool.clone() for pool in pools]
        y = op(x, *weights.args(), *meta, *candidate_pools)
        ours = (y[owned], candidate_pools[0][written])
        result = {
            "provider": provider,
            "seed_index": seed_index,
            "seed": 4200 + 10000 * seed_index,
            "compared_y_rows": int(owned.sum()),
            "excluded_slot_minus_one_rows": excluded,
        }
        for index, quantity in enumerate(("y", "state")):
            result[quantity] = {
                "fla": fla_errors[index],
                "ours": {
                    **errors(ours[index], references[index]),
                    "rms_pass": seed_gate(quantity, ours[index], fla[index], references[index]),
                },
            }
        result["history_pass"] = _same(candidate_pools[1][written], exact[2])
        result["untouched_pass"] = {
            name: _same(pool[untouched], before[untouched])
            for name, pool, before in zip(("state", "history"), candidate_pools, pools, strict=True)
        }
        results[provider] = result
    return results


@torch.inference_mode()
def measure_ulp_seed(ops, case, seed, device):
    arguments = _ulp_cases.make_inputs(case, seed, device)
    x, *rest = arguments
    weights = reference.Weights(*rest[:4])
    meta, pools = arguments[5:8], arguments[8:]
    written = meta[1][meta[1] >= 0].long()
    # The same frozen token-by-token oracle as the original precision gate.
    y, state, _ = reference.prefill(x, weights, case.hk, *meta, *pools)
    exact = (y, state[written])
    del state
    metrics = {}
    providers = {
        "fla": lambda *args: _fla_prefill(Geometry(case.hk, case.hv), *args),
        **ops,
    }
    for name, op in providers.items():
        candidate = [pool.clone() for pool in pools]
        got_y = op(*arguments[:8], *candidate)
        if got_y.dtype != torch.bfloat16 or candidate[0].dtype != torch.float32:
            raise ValueError("ULP gate requires BF16 y and FP32 state")
        got = (got_y, candidate[0][written])
        metrics[name] = {
            quantity: _ulp_cases.ulp_metrics(actual, expected)
            for quantity, actual, expected in zip(("y", "state"), got, exact, strict=True)
        }
    for name in ops:
        for quantity in ("y", "state"):
            metrics[name][quantity]["pass"] = _ulp_cases.ulp_pass(
                metrics[name][quantity], metrics["fla"][quantity]
            )
    return {"seed": seed, **metrics}


def run_ulp_cases(cases, get_ops, device):
    failures, passed = [], 0
    for case in cases:
        ops = get_ops(case.hk, case.hv, case.lengths)
        seeds = [measure_ulp_seed(ops, case, seed, device) for seed in _ulp_cases.SEEDS]
        gates = {
            provider: {
                quantity: all(seed[provider][quantity]["pass"] for seed in seeds)
                for quantity in ("y", "state")
            }
            for provider in ("port", "v1")
        }
        row_pass = all(all(gate.values()) for gate in gates.values())
        identity = {**asdict(case), "T": sum(case.lengths)}
        emit(
            "ULP_ROW",
            {
                **identity,
                "port_options": asdict(ops["port"].port_options),
                "seeds": seeds,
                "gate": gates,
                "pass": row_pass,
            },
        )
        passed += row_pass
        for seed in seeds:
            for provider in ("port", "v1"):
                quantities = [q for q in ("y", "state") if not seed[provider][q]["pass"]]
                if quantities:
                    failures.append(
                        {
                            **identity,
                            "seed": seed["seed"],
                            "provider": provider,
                            "quantities": quantities,
                        }
                    )
    emit(
        "ULP_SUMMARY",
        {
            "cases": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
            "pass": not failures,
            "failures": failures,
            "seeds_per_case": len(_ulp_cases.SEEDS),
            "metric": "abs(output - FP64 reference) / output-dtype ULP at reference",
            "dtypes": {"y": "bfloat16", "state": "float32"},
            "rule": "per seed and output: max_ulp <= fla.max_ulp AND gt4 <= fla.gt4",
            "deviations": [
                "Fixed public seeds drive the inputs.",
                "Raw output-dtype ULP replaces conditioned BF16 ULP and absolute floor.",
                "The frozen repository FP64 oracle is the reference.",
                "Composition-tier resampling is not part of this data-regime acceptance set.",
            ],
        },
    )
    return not failures


def main():
    parser = QuietParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cases",
        nargs="+",
        default=CORE,
        metavar="HK,HV,KIND[:L1+L2]",
        help="default: 16,32,ordinary 16,48,heavy; omitted lengths expand to [16] and [1024]",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="frozen matrix plus ULP acceptance rows, all triage regimes and tiny packed batches",
    )
    selection.add_argument("--ulp", action="store_true", help="ULP acceptance set only")
    parser.add_argument(
        "--ulp-lengths",
        nargs="+",
        type=int,
        default=[],
        metavar="T",
        help="also run the ULP rule on one-sequence [T] cases for --geoms (normal/heavy/robust)",
    )
    parser.add_argument(
        "--geoms",
        nargs="+",
        default=["16x32,16x48,8x16,8x24"],
        help="geometries for --all or --ulp (space/comma-separated HKxHV); default all four",
    )
    parser.add_argument(
        "--port-options",
        default="",
        metavar="k=v,...",
        help=(
            "PortPrefill options (default: none): producer_registers/prefetch/cp_segments are "
            "integers, iket/watchdog/cp/q_lookahead/remap_y use true/false or 0/1"
        ),
    )
    args = parser.parse_args()
    try:
        port_options = parse_port_options(args.port_options)
        geoms = parse_geoms(args.geoms)
        cases = (
            [
                (hk, hv, kind, tuple(lengths))
                for hk, hv in geoms
                for kind in KINDS
                for lengths in LENGTHS
            ]
            if args.all
            else ([] if args.ulp else parse_cases(args.cases))
        )
        ulp_cases = (
            _ulp_cases.exhaustive_cases(geoms, LENGTHS)
            if args.all
            else [case for case in _ulp_cases.acceptance_cases() if (case.hk, case.hv) in geoms]
            if args.ulp
            else []
        )
        if any(length <= 0 for length in args.ulp_lengths):
            raise ValueError("ULP lengths must be positive")
        ulp_cases += [
            _ulp_cases.ULPCase(hk, hv, kind, (length,))
            for hk, hv in geoms
            for length in args.ulp_lengths
            for kind in ("normal", "heavy", "robust")
        ]
        if args.ulp and not ulp_cases:
            raise ValueError("no acceptance cases for selected geometries")
    except (TypeError, ValueError):
        parser.error("invalid cases or port options")
    if not torch.cuda.is_available():
        parser.error("requires an sm120/sm121 CUDA GPU")
    # Match the frozen default CuTe schedule regardless of caller tuning variables.
    os.environ["CTI_GDN_PREFILL_CUTE"] = "1"
    os.environ["CTI_GDN_CUTE_TUNE"] = ""
    # The product now dispatches sm120 rows to the port; "v1" here is the CuTe v1 path.
    os.environ["CTI_GDN_PREFILL_PORT"] = "0"
    device = torch.device("cuda", torch.cuda.current_device())
    ops, rows = {}, []

    def get_ops(hk, hv, lengths):
        if (hk, hv) not in ops:
            v1 = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=device)
            if v1.provider != "cute":
                parser.error("v1 must resolve to CuTe")
            ops[hk, hv] = {
                "port": PortPrefill(num_k_heads=hk, num_v_heads=hv, device=device, **port_options),
                "v1": v1,
            }
        if ops[hk, hv]["v1"].selected_variant(sum(lengths), len(lengths)) != "cute":
            parser.error("v1 must select CuTe for this case")
        return ops[hk, hv]

    # Run the ULP gate first so --all produces its diagnostics promptly.
    ulp_ok = run_ulp_cases(ulp_cases, get_ops, device) if ulp_cases else True
    if args.ulp:
        return 0 if ulp_ok else 1
    for case in cases:
        hk, hv, kind, lengths = case
        get_ops(hk, hv, lengths)
        shared_seeds = [measure_seed(ops[hk, hv], case, seed) for seed in range(5)]
        for provider in ("port", "v1"):
            seeds = [seed[provider] for seed in shared_seeds]
            gates = {quantity: aggregate(seeds, quantity, "ours") for quantity in ("y", "state")}
            history_pass = all(seed["history_pass"] for seed in seeds)
            untouched_pass = all(all(seed["untouched_pass"].values()) for seed in seeds)
            excluded = seeds[0]["excluded_slot_minus_one_rows"]
            frozen_pass = all(gate["pass"] for gate in gates.values())
            contract_pass = all(gate["user_contract"]["pass"] for gate in gates.values())
            row = {
                "frozen_gate_pass": frozen_pass,
                "user_contract_pass": contract_pass,
                "provider": provider,
                "port_options": asdict(ops[hk, hv]["port"].port_options)
                if provider == "port"
                else None,
                "hk": hk,
                "hv": hv,
                "kind": kind,
                "lengths": lengths,
                "y_scope": "owned rows only" if excluded else "all rows",
                "excluded_slot_minus_one_rows": excluded,
                "seeds": seeds,
                "gate": {"ours": gates},
                "history_pass": history_pass,
                "untouched_pass": untouched_pass,
                "pass": frozen_pass and contract_pass and history_pass and untouched_pass,
            }
            emit("PREC_ROW", row)
            rows.append(row)
    for provider in ("port", "v1"):
        provider_rows = [row for row in rows if row["provider"] == provider]
        passed = sum(row["pass"] for row in provider_rows)
        emit(
            "PREC_SUMMARY",
            {
                "provider": provider,
                "port_options": provider_rows[0]["port_options"],
                "cases": len(provider_rows),
                "passed": passed,
                "failed": len(provider_rows) - passed,
                "pass": passed == len(provider_rows),
                "seeds_per_case": 5,
                "absolute_floor": ABSOLUTE_FLOOR,
                "user_contract_limits": {"rms_ratio_vs_fla": 1.05, "max_ratio_vs_fla": 1.25},
                "path": "fused PortPrefill"
                if provider == "port"
                else "product GatedDeltaNetPrefill (CuTe default)",
                "history_checked": True,
                "untouched_checked": True,
                "deviations": [],
                "ratio_convention": "0/0=1; nonfinite values are strings",
                "frozen_gate_failed_cases": [
                    {
                        **{key: row[key] for key in ("hk", "hv", "kind", "lengths")},
                        "quantities": [
                            q for q, gate in row["gate"]["ours"].items() if not gate["pass"]
                        ],
                    }
                    for row in provider_rows
                    if not row["frozen_gate_pass"]
                ],
                "user_contract_failed_cases": [
                    {
                        **{key: row[key] for key in ("hk", "hv", "kind", "lengths")},
                        "quantities": [
                            q
                            for q, gate in row["gate"]["ours"].items()
                            if not gate["user_contract"]["pass"]
                        ],
                    }
                    for row in provider_rows
                    if not row["user_contract_pass"]
                ],
                "failed_cases": [
                    {key: row[key] for key in ("hk", "hv", "kind", "lengths")}
                    for row in provider_rows
                    if not row["pass"]
                ],
            },
        )
    return 0 if ulp_ok and all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
