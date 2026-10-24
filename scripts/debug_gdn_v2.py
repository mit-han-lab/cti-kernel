#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Dump chunk-zero GDN v2 intermediates against FP64, and final y/state against v1.

Run from a checkout with PYTHONPATH=python on a supported GPU runner. The default
seed and poisoned fresh pools reproduce test_parity[16-32-1-0-False].
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.gpu._gdn_v2_reference import chunk0_intermediates


def summarize(name, actual, reference, *, token_rows=None, token_cols=None, row_offset=0):
    """Summarize a 2-D tile; padding is excluded from all primary metrics."""
    actual, reference = actual.detach().double().cpu(), reference.detach().double().cpu()
    if actual.shape != reference.shape or actual.ndim != 2:
        raise ValueError(f"{name}: expected matching two-dimensional tiles")
    mask = torch.ones_like(actual, dtype=torch.bool)
    if token_rows is not None:
        mask[token_rows:] = False
    if token_cols is not None:
        mask[:, token_cols:] = False
    got, expected = actual[mask], reference[mask]
    nonfinite = int((~got.isfinite()).sum())
    # Undefined comparisons are never silently omitted from max error. The finite
    # subset is also printed so a partly corrupt tile still has a useful scale.
    finite = got.isfinite() & expected.isfinite()
    finite_diff = (
        float((got[finite] - expected[finite]).abs().max()) if finite.any() else float("nan")
    )
    difference = float((got - expected).abs().max()) if got.numel() else float("nan")
    ref_max = float(expected.abs().max()) if expected.numel() else float("nan")
    locations = (actual.isnan() & mask).nonzero()
    first_nan = (
        (int(locations[0, 0]) + row_offset, int(locations[0, 1])) if locations.numel() else None
    )
    padded = int((~actual.isfinite() & ~mask).sum())
    print(
        f"{name}: max_abs_diff={difference:.8g} ref_max_abs={ref_max:.8g} "
        f"nonfinite={nonfinite} first_nan(row,col)={first_nan} "
        f"pad nonfinite count={padded} finite_max_abs_diff={finite_diff:.8g}",
        flush=True,
    )


def report_chunk(dump, reference, length, vs):
    for slot, name in (
        (0, "Q_h"),
        (1, "K_h"),
        (2, "g/beta"),
        (3, "L"),
        (4, "A_h"),
        (5, "S"),
        (6, "W"),
        (7, "U"),
        (8, "r"),
        (9, "o"),
    ):
        if slot == 2:
            for row, gate in enumerate(("g", "beta")):
                summarize(
                    f"slot 2 row {row} {gate}",
                    dump[2, row : row + 1, :64],
                    reference[gate][None],
                    token_cols=length,
                    row_offset=row,
                )
            continue
        ref = reference[name]
        summarize(f"slot {slot} {name}", dump[slot, :, : ref.shape[1]], ref, token_rows=length)
    for half in range(2):
        rows = min(64, max(0, vs - half * 64))
        if rows:
            summarize(
                f"slot {10 + half} H1 v={half * 64}..{half * 64 + rows - 1}",
                dump[10 + half, :rows],
                reference["H1"][half * 64 : half * 64 + rows],
            )
        else:
            # The unused slot is zero-filled scratch, not a state or token tile.
            summarize(
                f"slot {10 + half} unused (zero)",
                dump[10 + half],
                torch.zeros_like(dump[10 + half]),
            )
    summarize(
        "slots 12+13 B_h+B_l vs FP64 beta*silu(conv(V))",
        dump[12, :, :vs].double() + dump[13, :, :vs].double(),
        reference["B"],
        token_rows=length,
    )


def v1_chunk0(scratch, tune, lengths, hk, hv, sequence, head, vsplit, part):
    """Decode prepare_key/prepare_value's global, unswizzled storage.

    Q_hi precedes an optional whole Q_lo slab; W/A hi+lo are interleaved
    per (chunk, head). U's 64-column stores have row stride 128, not 64.
    Chunk ownership is sequence-major, using ceil(length/64) per sequence;
    the allocation can include unused capacity after the last live chunk.
    """
    if tune["form"] != "wy" or tune["o_form"] != "legacy":
        raise ValueError("scratch comparison requires v1 WY legacy output")
    kh_store = hv if tune["merge_prepare"] else hk
    chunks = scratch["k"].numel() // (kh_store * 64 * 128)
    chunk = sum((n + 63) // 64 for n in lengths[:sequence])
    kh = head if tune["merge_prepare"] else head // (hv // hk)
    width = 128 // vsplit
    columns = slice(part * width, (part + 1) * width)
    qhi = scratch["q"][: chunks * kh_store * 64 * 128]
    return {
        "Q_h": qhi.view(chunks, kh_store, 64, 128)[chunk, kh],
        "K_h": scratch["k"].view(chunks, kh_store, 64, 128)[chunk, kh],
        "g": scratch["g"].view(chunks, hv, 64)[chunk, head],
        "W": scratch["w"].view(chunks, hv, 1 + tune["w_split"], 64, 128)[chunk, head, 0],
        "U": scratch["u"].view(chunks, hv, 64, 128)[chunk, head, :, columns],
        "A_h": scratch["aq"].view(chunks, hv, 1 + tune["aq_out"], 64, 64)[chunk, head, 0],
    }


def report_v1(dump, scratch, length):
    for slot, name in ((0, "Q_h"), (1, "K_h"), (4, "A_h"), (6, "W"), (7, "U")):
        ref = scratch[name]
        summarize(
            f"slot {slot} {name} vs v1 scratch",
            dump[slot, :, : ref.shape[1]],
            ref,
            token_rows=length,
        )
    summarize("slot 2 g vs v1 scratch", dump[2, :1, :64], scratch["g"][None], token_cols=length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("16x32", "16x48", "8x16", "8x24"), default="16x32")
    parser.add_argument("--vsplit", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16])
    parser.add_argument("--initial", action="store_true")
    parser.add_argument(
        "--tune-extra",
        default="",
        help="extra CTI_GDN_CUTE_TUNE pairs for both v2 runs, e.g. v2_ldt=1",
    )
    args = parser.parse_args()
    if any(length <= 0 for length in args.lengths):
        parser.error("--lengths must contain positive lengths")
    if not torch.cuda.is_available():
        parser.error("this script requires a GPU runner; the reference itself also supports CPU")

    # Keep GPU test/fixture imports out of --help and the CPU summary tests.
    from ctikernels.gdn import GatedDeltaNetPrefill
    from tests.gpu._gdn_prefill_cases import _inputs, _poison, _pools, _prefill_case

    hk, hv = (int(value) for value in args.geometry.split("x"))
    count = len(args.lengths)
    slots = [7] if count == 1 else [9, 2, -1, 5][:count] + list(range(10, 10 + max(0, count - 4)))
    meta = _prefill_case(args.lengths, slots, [args.initial] * count)
    weights, x = _inputs(hk, hv, args.lengths, "ordinary", 4200)
    base = _pools(hk, hv, seed=4202, slots=max(24, max(slots) + 1))
    read = {slot for slot in slots if slot >= 0 and args.initial}
    pools = [_poison(pool, read, "nan", 4203) for pool in base]
    os.environ["CTI_GDN_PREFILL_CUTE"] = "1"

    def construct(tune):
        os.environ["CTI_GDN_CUTE_TUNE"] = tune
        op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
        if op.provider != "cute" or op.selected_variant(sum(args.lengths), count) != "cute":
            raise RuntimeError("CuTe product path unavailable; refusing fallback diagnostics")
        return op

    v2 = construct(f"v2=1,vsplit={args.vsplit},v2_debug=1,{args.tune_extra}")
    plain = construct(f"v2=1,vsplit={args.vsplit},v2_debug=0,{args.tune_extra}")
    v1 = construct("v2=0,overlap=0")
    v1._kernel._cute.retain_scratch = True
    p2, p1, pp = ([pool.clone() for pool in pools] for _ in range(3))
    with torch.inference_mode():
        y2 = v2(x, *weights.args(), *meta, *p2)
        y1 = v1(x, *weights.args(), *meta, *p1)
        yp = plain(x, *weights.args(), *meta, *pp)
    torch.cuda.synchronize()
    dump = v2._kernel._cute.last_debug
    if dump is None:
        raise RuntimeError("v2_debug did not produce a dump")
    dump = dump.cpu()
    scratch = {name: tile.cpu() for name, tile in v1._kernel._cute.last_scratch.items()}
    if scratch["u"].dtype != torch.float32:
        raise RuntimeError("expected v1 FP32 U scratch")
    # FP64 runs on CPU here: runner GPUs often have low FP64 throughput. Transfer
    # immutable inputs once; never use either provider's mutated state/history.
    reference_args = tuple(t.cpu() for t in (x, *weights.args()[:3], *meta, *pools))
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    print("Chunk-zero reference: FP64 math without intermediate BF16 rounding; o is scaled.")
    for seq, length in enumerate(args.lengths):
        for head in range(hv):
            for part in range(args.vsplit):
                cta = (seq * hv + head) * args.vsplit + part
                print(f"CTA {cta}: sequence={seq} value_head={head} part={part}", flush=True)
                ref = chunk0_intermediates(
                    *reference_args,
                    k_heads=hk,
                    sequence=seq,
                    value_head=head,
                    vsplit=args.vsplit,
                    part=part,
                )
                report_chunk(dump[cta], ref, min(64, length), 128 // args.vsplit)
                baseline = v1_chunk0(
                    scratch,
                    v1._kernel._cute.tune,
                    args.lengths,
                    hk,
                    hv,
                    seq,
                    head,
                    args.vsplit,
                    part,
                )
                report_v1(dump[cta], baseline, min(64, length))
    print("Final product v2 versus v1 (all chunks):", flush=True)
    summarize("y", y2, y1)
    summarize("y non-debug v2 versus v1", yp, y1)
    summarize("y non-debug v2 versus debug v2", yp, y2)
    for slot in slots:
        if slot >= 0:
            for head in range(hv):
                summarize(
                    f"state slot={slot} value_head={head} [v,k]",
                    p2[0][slot, head],
                    p1[0][slot, head],
                )
                summarize(
                    f"state non-debug versus debug slot={slot} value_head={head} [v,k]",
                    pp[0][slot, head],
                    p2[0][slot, head],
                )


if __name__ == "__main__":
    main()
