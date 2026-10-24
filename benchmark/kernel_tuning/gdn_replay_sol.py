# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Serial memory/node-floor reference, matching the shared SOL curve formula.

Reference: the shared SOL helper benchmark/sol/sol.py (commit 4522f5a675ea5fa133816961cb98cffdb9fd6148,
kept outside this repository). No helper code is vendored.
Scalar FLOPs/synchronization are outside that helper's tensor-core model.
"""

import json
import math
from pathlib import Path

from benchmark.kernel_tuning.gdn_replay_common import digest

SOL_REVISION = "4522f5a675ea5fa133816961cb98cffdb9fd6148"


def transfer(size, memory, mode):
    if size <= 0:
        return 0.0
    points = sorted(memory["small_transfer"], key=lambda p: p["bytes"])
    if not points:
        raise ValueError("SOL curve is not calibrated")
    if size <= points[0]["bytes"]:
        return points[0]["time_us"]
    for lower, upper in zip(points, points[1:], strict=False):
        if size <= upper["bytes"]:
            weight = math.log(size / lower["bytes"]) / math.log(upper["bytes"] / lower["bytes"])
            return lower["time_us"] * (upper["time_us"] / lower["time_us"]) ** weight
    return size / (1000 * memory[f"dram_{mode}_gbs"])


def report(provider, b, hk, hv, window, measured_us, kernels, ceilings, regime):
    if ceilings is None:
        return {
            "SOL": None,
            "reason": "not available: pass --ceilings with measured device JSON",
            "helper_revision": SOL_REVISION,
        }
    path = Path(ceilings)
    table = json.loads(path.read_text())
    c, dc = 4 * b * hv * 128 * 128, (2 * hk + hv) * 128
    w = dc + 2 * hv
    weights = 8 * dc + 8 * hv + 512
    activation = 2 * b * (dc + hv * 128 + 2 * hv)
    output = 2 * b * hv * 128
    history = 12 * b * dc
    floor = table["fixed_costs"]["graph_node_us"]
    times, resources = {}, {}
    for estimate in ("useful", "implementation"):
        total = 0.0
        limiting = {}
        for phase in range(window):
            replay = provider in ("cute", "triton")
            state_bytes = c * (1 + (int(phase == window - 1) if replay else 1))
            ring = 2 * b * w * (phase + 4) if replay else 0
            if estimate == "implementation" and replay:
                # Per-head raw replay loads, including redundant grouped q/k reads.
                ring = 2 * b * hv * (phase + 1) * (4 * 3 * 128 + 2)
            arithmetic = (
                state_bytes + weights + activation + output + ring + (0 if replay else history)
            )
            traffic = [arithmetic] + [0.0] * max(0, kernels - 1)
            if replay:
                # A one-launch step also carries the history/ring/metadata publication.
                traffic[-1] += history + 2 * b * w + 64 * b
            for size in traffic:
                dram = transfer(size, table["memory"], "copy") if regime == "cold" else 0.0
                l2 = size / (table["memory"]["l2_read_gbs"] * 1000) if regime == "warm" else 0.0
                terms = {"DRAM": dram, "L2": l2, "tensor_compute": 0.0, "node_floor": floor}
                winner = max(terms, key=terms.get)
                total += terms[winner]
                limiting[winner] = limiting.get(winner, 0) + 1
        times[estimate] = total / window
        resources[estimate] = limiting
    return {
        "SOL": times["implementation"] / measured_us,
        "useful_SOL": times["useful"] / measured_us,
        "reference_us": times,
        "limiting_resource": resources,
        "helper_revision": SOL_REVISION,
        "ceiling_sha256": digest(path),
        "assumptions": [
            "serial sum of per-kernel max(DRAM,L2,tensor,node_floor)",
            "curve interpolation; scalar compute and synchronization omitted",
            "memory traffic estimate; no tensor wave correction (no MMA)",
            "external traffic is a useful boundary lower bound, not audited implementation",
        ],
    }
