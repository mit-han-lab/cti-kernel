# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Per-row SOL for K2 verify + commit rounds (gdn_verify_compare rows).

Uses the vendored shared helper benchmark/sol/sol.py and its compact ceilings
(see benchmark/sol/PROVENANCE.md). One round of the trusted fused eager path is
two PDL launches: verify (read state, history, Q projected rows; write Q outputs
and Q pending raw rows) then commit (read state, history, m pending rows; write
state and history), with m = accepted_per_row. Bytes are logical boundary
transfers; mixed read/write kernels use copy bandwidth. The recurrence is scalar
FP32 (no MMA), so there is no tensor term and the useful and implementation
estimates coincide; scalar compute, the sequential per-token dependence and
synchronization are outside the helper's model.

Regimes: "warm" matches gdn_verify_compare (the fixture restores state before
every sample): when a round's footprint fits in L2, every byte is served at the
L2 rate. "cold" sends every byte to DRAM. Both are reported; SOL is t_ref/t_meas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.sol import sol

DEVICES = {"5090": "rtx5090", "PRO 6000": "rtx-pro-6000", "GB10": "gb10"}


def device_key(name):
    for token, key in DEVICES.items():
        if token in name:
            return key
    raise ValueError(f"no SOL ceilings for device {name!r}")


def round_bytes(b, hk, hv, q, m):
    dc = (2 * hk + hv) * 128
    w = dc + 2 * hv
    proj = dc + hv * 128 + 2 * hv
    state = 4 * b * hv * 128 * 128
    history = 2 * 3 * b * dc
    weights = 8 * dc + 8 * hv + 512
    verify = {
        "kv_state": state + history,
        "weights": weights,
        "activations": 2 * b * q * proj,
        "outputs": 2 * b * q * hv * 128,
        "intermediates": 2 * b * q * w + 64 * b,
    }
    commit = {
        "kv_state": 2 * (state + history),
        "weights": weights,
        "intermediates": 2 * b * m * w + 64 * b,
    }
    return verify, commit


def round_call(b, hk, hv, q, m, regime, l2_bytes):
    verify, commit = round_bytes(b, hk, hv, q, m)
    footprint = sum(verify.values()) + sum(commit.values())
    warm = regime == "warm" and footprint <= l2_bytes
    kernels = [
        sol.Kernel(
            name,
            dram_bytes=dict(traffic),
            cache_state="warm" if warm else "cold",
            l2_resident_bytes=dict(traffic) if warm else {},
            dram_mode="copy",
            pdl=True,
        )
        for name, traffic in (("verify", verify), ("commit", commit))
    ]
    return sol.Call(kernels, mode="pdl_chain"), warm


def row_sol(device, b, hk, hv, q, m, measured_us):
    table = sol.load_ceilings(device)
    out = {"kernels_per_round": 2}
    for regime in ("warm", "cold"):
        call, warm = round_call(b, hk, hv, q, m, regime, table["l2_bytes"])
        result = sol.estimate(call, table, estimate_type="useful", bw_model="curve")
        out[regime] = {
            "t_ref_us": result.t_ref_us,
            "SOL": result.t_ref_us / measured_us,
            "limiting": result.limiting_resource,
            "l2_resident": warm,
        }
    return out


def helper_pin():
    root = Path(sol.__file__).resolve().parent
    return {
        "sol.py": hashlib.sha256((root / "sol.py").read_bytes()).hexdigest(),
        "source": "benchmark/sol/PROVENANCE.md",
    }


def rows(paths):
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if "q" in row and "b" in row and "median_us" in row:
                yield row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipts", nargs="+", help="gdn_verify_compare JSON-lines outputs")
    args = parser.parse_args(argv)
    print(json.dumps(helper_pin()))
    print(
        "| Device | (Hk,Hv) | Q | B | ours us | t_ref warm us | SOL warm | t_ref cold us | SOL cold |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(
        rows(args.receipts),
        key=lambda r: (r["device"], r["shape"]["hv"], r["q"], r["b"]),
    ):
        hk, hv = row["shape"]["hk"], row["shape"]["hv"]
        ours = row["median_us"]["ours"]
        s = row_sol(
            device_key(row["device"]), row["b"], hk, hv, row["q"], row["accepted_per_row"], ours
        )
        print(
            f"| {row['device']} | ({hk},{hv}) | {row['q']} | {row['b']} | {ours:.1f} | "
            f"{s['warm']['t_ref_us']:.2f} | {s['warm']['SOL']:.3f} | "
            f"{s['cold']['t_ref_us']:.2f} | {s['cold']['SOL']:.3f} |"
        )


if __name__ == "__main__":
    main()
