# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only V2P0 single-kernel SOL; no compilation or product imports.

This is an optimistic fused floor, not a runtime prediction. For vsplit > 1,
the modeled split-output design requires a separate norm kernel;
its launch, scratch and communication are deliberately outside this one-kernel
model. Scalar conv/norm/gates, SFU, shared-memory and synchronization costs are
unmodeled. No captures, CP, scratch, cross-replay cache reuse or stateless slots.

Bytes are logical transfers. DRAM covers unique boundary I/O; L2 covers only
additional requests, following benchmark/sol/sol.py. Mixed read/write bytes use sustained copy bandwidth.
Decimal GB/s, TFLOP/s, microseconds, and two FLOPs per MAC. The graph-node floor
is inside max, never added. Equal-work wave efficiency applies to compute only;
the longest ragged chain is a separate floor, not an equal-length wave schedule.
The WY ledger follows v1's precision scheme. SOL defaults to serialized prologue
plus on-chain products; the overlapped-prologue chain is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.kernel_tuning import gdn_prefill_v2_matrix as matrix
from benchmark.sol import sol

K = V = 128
CHUNK = 64
MMA = "bf16_m16n8k16"
VSPLITS = (1, 2, 4)
# WY operands follow _cute_prefill_kernels.py: U uses B_hi/B_lo; QK adds
# Q_lo K_hi + Q_hi K_lo for the Hv=48 family. The inverse base count already
# includes all four split terms of each blocked product (solve_mma=2).
PLANES = {
    "gram": 1,
    "qk": 1,
    "inverse": 1,
    "w": 1,
    "u": 2,
    "wh": 1,
    "qh": 1,
    "ar": 1,
    "state_update": 1,
}
ON_CHAIN = ("wh", "qh", "ar", "state_update")


@lru_cache(maxsize=3)
def ceilings(device):
    """Cache the compact vendored tables; the helper validates their measured rates."""
    return sol.load_ceilings(device)


def chunk_flops(hv, vsplit=1, planes=None):
    """Executed MMA FLOPs per full 64-token chunk of one CTA, by product.

    Gram, QK, inverse and W repeat in each slice. Products with a V dimension
    partition that dimension. Dense padded chunks are charged even for tails.
    The blocked unit-lower inverse costs exactly 16 block products of 16^3 MACs
    with four split terms each, or 64*16^3 MACs at planes=1. QK uses one plane
    for hv in (16, 32), three for (24, 48); optional TP4 hv=8/12 follows its
    respective family. A planes mapping overrides any subset after hv defaults.
    """
    if type(hv) is not int or hv not in {shape[1] for shape in matrix.GEOMETRIES}:
        raise ValueError("hv must belong to a supported geometry")
    if type(vsplit) is not int or vsplit not in VSPLITS:
        raise ValueError("vsplit must be 1, 2 or 4")
    precision = PLANES | {"qk": PLANES["qk"] * (3 if hv in (12, 24, 48) else 1)}
    precision.update({} if planes is None else planes)
    if set(precision) != set(PLANES) or any(
        type(value) is not int or value < 1 for value in precision.values()
    ):
        raise ValueError("planes must contain known products and positive integer multipliers")
    width = V // vsplit
    macs = {
        "gram": CHUNK * CHUNK * K,
        "qk": CHUNK * CHUNK * K,
        "inverse": 64 * 16**3,
        "w": CHUNK * K * CHUNK,
        "u": CHUNK * width * CHUNK,
        "wh": CHUNK * width * K,
        "qh": CHUNK * K * width,
        "ar": CHUNK * width * CHUNK,
        "state_update": width * K * CHUNK,
    }
    return {name: 2 * count * precision[name] for name, count in macs.items()}


def byte_ledger(hk, hv, lengths, initial, vsplit=1):
    """Return compulsory reads/writes and additional L2 requests by buffer.

    Q/K inputs and history repeat across Hv/Hk siblings and slices. V and z
    slices are disjoint; neither is multiplied by vsplit. a/b gates repeat
    across slices. Each nonfirst chunk rereads three projected halo rows;
    the first chunk reads history only with initial state. Final history rows
    are retained locally (short fresh sequences write zero padding).
    Weights and sequence metadata are loaded once per CTA and retained across
    its chunks. Shared Q/K history is written once by an ideal owner; arrival
    counters and the split norm protocol are outside this floor.
    """
    if (hk, hv) not in matrix.GEOMETRIES or not lengths or len(lengths) != len(initial):
        raise ValueError("unsupported geometry or sequence contract")
    if any(type(length) is not int or length < 1 for length in lengths) or any(
        type(flag) is not bool for flag in initial
    ):
        raise ValueError("lengths must be positive integers and initial flags must be bools")
    if type(vsplit) is not int or vsplit not in VSPLITS:
        raise ValueError("vsplit must be 1, 2 or 4")
    tokens, sequences, initialized = sum(lengths), len(lengths), sum(initial)
    chunks = sum((length + CHUNK - 1) // CHUNK for length in lengths)
    qk, value = 2 * hk * K, hv * V
    conv = qk + value
    shared_copies = hv // hk * vsplit
    metadata = 4 * (sequences + 1) + 5 * sequences
    reads = {
        "projected_qk": 2 * tokens * qk,
        "projected_v": 2 * tokens * value,
        "projected_z": 2 * tokens * value,
        "projected_ba": 4 * tokens * hv,
        "initial_state": 4 * initialized * hv * K * V,
        "initial_history": 6 * initialized * conv,
        "conv_weights": 8 * conv,
        "gate_weights": 8 * hv,
        "norm_weight": 4 * V,
        "metadata": metadata,
    }
    writes = {
        "final_state": 4 * sequences * hv * K * V,
        "final_history": 6 * sequences * conv,
        "y": 2 * tokens * value,
    }
    repeated = {
        "projected_qk": reads["projected_qk"] * (shared_copies - 1),
        "projected_ba": reads["projected_ba"] * (vsplit - 1),
        "projected_v": 0,
        "projected_z": 0,
        "qk_halo": 6 * (chunks - sequences) * qk * shared_copies,
        "v_halo": 6 * (chunks - sequences) * value,
        "initial_history": 6 * initialized * qk * (shared_copies - 1),
        "conv_weights": 8 * (sequences * (qk * shared_copies + value) - conv),
        "gate_weights": reads["gate_weights"] * (sequences * vsplit - 1),
        "norm_weight": reads["norm_weight"] * (sequences * hv - 1),
        # Per CTA: two cu_seqlens entries, one slot, one bool; shared endpoints
        # are already charged exactly once in the compulsory metadata above.
        "metadata": 13 * sequences * hv * vsplit - metadata,
    }
    return {"dram_reads": reads, "dram_writes": writes, "l2_repeated": repeated}


def analyze(device, row, vsplit=1, *, planes=None, ctas_per_sm=1):
    """Analyze one split. All t_* and sol_us fields are in microseconds.

    Wave efficiency = grid / (ceil(grid / capacity) * capacity), with capacity
    = SMs * ctas_per_sm. This is an assumed residency, not a measured v2 launch.
    t_chain = max(chunks per sequence) * CTA chunk FLOPs / (GPU MMA peak / SMs).
    This serialized-prologue form is the default SOL bound. t_chain_overlapped
    uses only WH + QH + AR + state_update, assuming prologue work fully overlaps;
    total compute still charges every product. ledger.chain reports per-chunk,
    per-CTA FLOPs grouped into prologue and on_chain products.
    Raising residency never grants each CTA another SM's full throughput.
    """
    if type(ctas_per_sm) is not int or ctas_per_sm < 1:
        raise ValueError("ctas_per_sm must be a positive integer")
    hk, hv, lengths, initial = (row[key] for key in ("hk", "hv", "lengths", "initial"))
    ledger = byte_ledger(hk, hv, lengths, initial, vsplit)
    products = chunk_flops(hv, vsplit, planes)
    ledger["chain"] = {
        "prologue": {name: count for name, count in products.items() if name not in ON_CHAIN},
        "on_chain": {name: products[name] for name in ON_CHAIN},
    }
    chunks = [(length + CHUNK - 1) // CHUNK for length in lengths]
    grid = len(lengths) * hv * vsplit
    flops_by_product = {name: count * sum(chunks) * hv * vsplit for name, count in products.items()}
    flops = sum(flops_by_product.values())
    dram = sum(ledger["dram_reads"].values()) + sum(ledger["dram_writes"].values())
    l2 = sum(ledger["l2_repeated"].values())
    table = ceilings(device)
    kernel = sol.Kernel(
        name="gdn_prefill_v2_fused",
        dram_bytes={"activations": dram},
        l2_bytes=l2,
        flops={MMA: flops},
        executed_flops={MMA: flops},
        n_ctas=grid,
        dram_mode="copy",
    )
    estimate = sol.estimate(
        sol.Call([kernel]),
        table,
        estimate_type="implementation",
        bw_model="large",
        residency=ctas_per_sm,
    ).per_kernel[0]
    per_sm_flops_per_us = table["tensor_core"][MMA]["tflops_gpu"] * 1e6 / table["sm_count"]
    terms = {
        "graph": estimate.terms_us["floor"],
        "dram": estimate.terms_us["dram"],
        "l2": estimate.terms_us["l2"],
        "compute": estimate.terms_us["tensor"],
        "chain": max(chunks) * sum(products.values()) / per_sm_flops_per_us,
    }
    bound = max(terms.values())
    # Equal-length underfilled grids have mathematically identical compute and
    # chain floors. Report both, even if floating-point summation differs slightly.
    binds = [name for name, time in terms.items() if math.isclose(time, bound, rel_tol=1e-12)]
    return {
        **row,
        "device": device,
        "vsplit": vsplit,
        "grid": grid,
        "ctas_per_sm": ctas_per_sm,
        "waves": estimate.waves,
        "wave_efficiency": estimate.wave_efficiency,
        "dram_bytes": dram,
        "l2_bytes": l2,
        "flops": flops,
        "flops_by_product": flops_by_product,
        "chunk_cta_flops": sum(products.values()),
        "ledger": ledger,
        **{f"t_{name}": time for name, time in terms.items()},
        "t_chain_overlapped": max(chunks)
        * sum(ledger["chain"]["on_chain"].values())
        / per_sm_flops_per_us,
        "sol_us": bound,
        "binds": binds,
    }


def analyze_row(device, row, *, planes=None, ctas_per_sm=1):
    """Report all three splits and the fastest; exact ties prefer smaller splits."""
    variants = [
        analyze(device, row, split, planes=planes, ctas_per_sm=ctas_per_sm) for split in VSPLITS
    ]
    best = min(variants, key=lambda result: (result["sol_us"], result["vsplit"]))
    return {"variants": variants, "best_vsplit": best["vsplit"], "best": best}


def self_test():
    matrix.self_test()
    checked = 0
    for device in matrix.DEVICES:
        for row in matrix.rows():
            results = analyze_row(device, row)["variants"]
            assert len({result["dram_bytes"] for result in results}) == 1
            assert all(math.isfinite(r["sol_us"]) and r["sol_us"] > 0 for r in results)
            for left, right in zip(results, results[1:], strict=False):
                assert left["t_chain"] > right["t_chain"]
                assert left["l2_bytes"] < right["l2_bytes"]
            checked += 1
    assert checked == 204
    return checked


def markdown(results, device):
    lines = [
        f"{matrix.DEVICES[device]['name']}: optimistic single-kernel SOL. Times in us.",
        "L2 bytes are additional requests. * marks best vsplit; graph floor is inside max.",
        "SOL uses t_chain (all products, serialized prologue) by default. "
        "t_chain_overlapped uses on-chain products only, with prologue fully overlapped.",
        "",
        "| geom | case: lengths | vsplit | DRAM B | L2 B | FLOPs | t_dram | t_l2 | t_compute | t_chain | t_chain_overlapped | t_graph | SOL us | binds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        for variant in result["variants"]:
            r = variant
            split = str(r["vsplit"]) + ("*" if r["vsplit"] == result["best_vsplit"] else "")
            times = " | ".join(
                f"{r[key]:.3f}"
                for key in (
                    "t_dram",
                    "t_l2",
                    "t_compute",
                    "t_chain",
                    "t_chain_overlapped",
                    "t_graph",
                    "sol_us",
                )
            )
            lines.append(
                f"| {r['hk']}x{r['hv']} | {r['case']}: {r['lengths']} | {split} | "
                f"{r['dram_bytes']} | {r['l2_bytes']} | {r['flops']:.3f} | {times} | "
                f"{' + '.join(r['binds'])} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=matrix.DEVICES)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        count = self_test()
        print(f"PASS: matrix and {count} device rows, {count * len(VSPLITS)} split variants")
        return
    if args.device is None:
        parser.error("--device is required unless --self-test is used")
    results = [analyze_row(args.device, row) for row in matrix.rows()]
    if args.markdown:
        print(markdown(results, args.device))
    else:
        print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
