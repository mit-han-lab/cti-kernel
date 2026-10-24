#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Diagnostics: per adversarial family, port variants vs fla-core vs CuTe v1 in raw ULP.

Each row forces one adversarial family on every span (_ulp_cases.FORCE_FAMILY) of a
single-geometry case and prints max ULP / >4 counts of y and state for every provider.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ulp_cases
import cp2_precision as cp
import torch

FAMILIES = (
    "plain",
    "duplicate",
    "colinear",
    "opposite",
    "decay_one",
    "decay_underflow",
    "beta_zero",
    "beta_one",
    "scaled",
    "zero_keys",
)


@torch.inference_mode()
def measure(ops, case, seed, device):
    """Per provider and view: (max |err| / row RMS of the reference, rows over 4 BF16
    half-ULPs of row RMS, i.e. 2**-7 relative, worst row index)."""
    arguments = _ulp_cases.make_inputs(case, seed, device)
    x, *rest = arguments
    weights = cp.reference.Weights(*rest[:4])
    meta, pools = arguments[5:8], arguments[8:]
    written = meta[1][meta[1] >= 0].long()
    y, state, _ = cp.reference.prefill(x, weights, case.hk, *meta, *pools)
    exact = (y.double(), state[written].double())
    del state
    providers = {"fla": lambda *a: cp._fla_prefill(cp.Geometry(case.hk, case.hv), *a), **ops}
    out = {}
    for name, op in providers.items():
        candidate = [pool.clone() for pool in pools]
        got_y = op(*arguments[:8], *candidate)
        got = (got_y, candidate[0][written])
        out[name] = {}
        for q, a, e in zip(("y", "state"), got, exact, strict=True):
            if e.numel() == 0:
                out[name][q] = [0.0, 0, -1]
                continue
            err = (a.double() - e.to(a.device)).abs()
            rms = e.to(a.device).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
            rel = (err / rms).amax(-1).reshape(-1)
            rel = torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf")))
            worst = int(rel.argmax())
            flat_err = err.reshape(rel.numel(), -1)[worst]
            flat_ref = e.to(a.device).reshape(rel.numel(), -1)[worst]
            col = int(flat_err.argmax())
            out[name][q] = [
                float(rel.max()),
                int((rel > 2**-7).sum()),
                worst,
                float(flat_err[col]),
                float(rms.reshape(-1)[worst]),
                float(flat_ref[col]),
                float(a.double().reshape(rel.numel(), -1)[worst][col]),
                col,
            ]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="8,16,977")
    parser.add_argument("--families", default="all")
    parser.add_argument("--kind", default="adversarial")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-scale-state", action="store_true")
    parser.add_argument("--scale-part", default="all")
    parser.add_argument(
        "--variants",
        default="",
        help="'|'-separated list of port option strings (k=v,...); default: one default port",
    )
    args = parser.parse_args()
    hk, hv, *lengths = (int(v) for v in args.case.replace("+", ",").split(","))
    os.environ["CTI_GDN_PREFILL_CUTE"] = "1"
    os.environ["CTI_GDN_CUTE_TUNE"] = ""
    os.environ["CTI_GDN_PREFILL_PORT"] = "0"
    device = torch.device("cuda", torch.cuda.current_device())
    v1 = cp.GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=device)
    ops = {"v1": v1}
    for index, text in enumerate(args.variants.split("|")):
        options = cp.parse_port_options(text)
        ops[f"p{index}"] = cp.PortPrefill(num_k_heads=hk, num_v_heads=hv, device=device, **options)
        print("VARIANT", f"p{index}", text, flush=True)
    _ulp_cases.SCALE_STATE = not args.no_scale_state
    parts = [None if p == "all" else p for p in args.scale_part.split(",")]
    families = range(10) if args.families == "all" else [int(f) for f in args.families.split(",")]
    for kind in args.kind.split(","):
        case = _ulp_cases.ULPCase(hk, hv, kind, tuple(lengths))
        adversarial = kind in ("adversarial", "robust")
        runs = (
            ([None, *families] if args.families == "all" else families) if adversarial else [None]
        )
        for family in runs:
            _ulp_cases.FORCE_FAMILY = family
            name = kind if family is None else FAMILIES[family]
            for part in parts:
                _ulp_cases.SCALE_PART = part
                for seed in _ulp_cases.SEEDS[: args.seeds]:
                    m = measure(ops, case, seed, device)
                    row = {
                        p: {
                            q: [round(v, 4) if isinstance(v, float) else v for v in m[p][q]]
                            for q in ("y", "state")
                        }
                        for p in m
                    }
                    print(
                        "FAM",
                        name + ("" if part is None else "_" + part),
                        seed,
                        json.dumps(row),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
