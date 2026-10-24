# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Measure the block-FP8 activation quantizer's tiles (``_block_fp8_dispatch.QUANTIZER``).

``--sweep``: on the current device, time ``quantize_per_token_group_fp8``'s kernel at every
tile (rows per program, warps; contiguous activations run one group per kernel row, so rows
there are groups) and sgl-kernel's quantizer, over rows x K. The activation
stays in L2, as after the kernel producing it: each sample is one CUDA-graph replay of 20
calls on the same tensor. Every tile's output is checked bit for bit against sgl-kernel's.
One ``QSWEEP`` JSON line per (rows, K) with the median microseconds of each. ``--part i/n``
runs every n-th shape.

``--emit LOG...``: read ``QSWEEP`` lines (several runs average) and print ``QUANTIZER``
entries: per device, runs over ``units = rows * groups`` bands, each band taking the tile
with the least geometric-mean time relative to each shape's fastest (the previous band's
tile when within 1 %), and the table's speed against sgl-kernel's and the size rule's.

usage: python -m benchmark.kernel_tuning.block_fp8_quantizer --sweep [--part 0/2]
       python -m benchmark.kernel_tuning.block_fp8_quantizer --emit sweep-*.log
"""

from __future__ import annotations

import argparse
import json
import math

KS = (128, 256, 512, 1024, 2048, 4096)
ROWS = (1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
ROWS += (3072, 4096, 6144, 8192, 16384, 32768)
TILES = tuple(
    (bm, nw) for bm in (1, 2, 4, 8, 16, 32, 64) for nw in (1, 2, 4, 8, 16) if bm * 128 >= 32 * nw
)
BANDS = tuple(1 << e for e in range(4, 17))  # last units of each run; the last runs on


def sweep(part: int, parts: int, iterations: int) -> None:
    import torch

    from benchmark.kernel_tuning.block_fp8 import _median_us, _sgl_quantize
    from ctikernels.linear.triton import _block_fp8

    torch.manual_seed(0)
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    base = {"cc": [device.major, device.minor], "sms": device.multi_processor_count}
    shapes = [(m, k) for k in KS for m in ROWS if m * k <= 32768 * 512][part::parts]
    with torch.inference_mode():
        for m, k in shapes:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x[:, : k // 2] *= 100  # groups of different magnitudes
            q = torch.empty(m, k, device="cuda", dtype=torch.float8_e4m3fn)
            scale = torch.empty(m, k // 128, device="cuda", dtype=torch.float32)
            expected_q, expected_scale = _sgl_quantize(x)
            record = {**base, "M": m, "K": k}
            record["sgl"] = _median_us([lambda x=x: _sgl_quantize(x)] * 20, iterations)
            for tile in TILES:

                def run(tile=tile, x=x, q=q, scale=scale):
                    _block_fp8.quantize(x, q, scale, tile=tile)

                q.zero_()
                scale.zero_()
                run()
                if not (
                    torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
                    and torch.equal(scale, expected_scale)
                ):
                    raise AssertionError(f"tile {tile} changed the output at {m} x {k}")
                record["%dx%d" % tile] = _median_us([run] * 20, iterations)
            print("QSWEEP " + json.dumps(record), flush=True)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(map(math.log, values)) / len(values))


def _rule(m: int) -> str:
    return "%dx4" % (4 if m <= 64 else 8 if m <= 1024 else 16)


def emit(logs: list[str]) -> None:
    samples: dict[tuple, dict[tuple[int, int], list[dict]]] = {}
    for path in logs:
        with open(path) as handle:
            for line in handle:
                if line.startswith("QSWEEP "):
                    r = json.loads(line[len("QSWEEP ") :])
                    device = samples.setdefault((*r["cc"], r["sms"]), {})
                    device.setdefault((r["M"], r["K"]), []).append(r)
    names = ["%dx%d" % tile for tile in TILES]
    for device, shapes in sorted(samples.items()):
        # Per shape, each tile's time relative to the fastest, averaged over runs.
        relative = {}
        for shape, runs in shapes.items():
            mean = {n: _geomean([r[n] for r in runs]) for n in [*names, "sgl"]}
            fastest = min(mean[n] for n in names)
            relative[shape] = {n: mean[n] / fastest for n in [*names, "sgl"]}
        table: list[tuple[int, str]] = []
        low = 0
        for last in (*BANDS, 1 << 62):
            band = [s for s in relative if low < s[0] * s[1] // 128 <= last]
            low = last
            if not band:
                continue
            score = {n: _geomean([relative[s][n] for s in band]) for n in names}
            best = min(score, key=score.get)
            if table and score[table[-1][1]] <= 1.01 * score[best]:
                best = table[-1][1]
            if table and table[-1][1] == best:
                table[-1] = (last, best)
            else:
                table.append((last, best))
        # The last run holds every larger batch: name it by the largest units swept.
        table[-1] = (max(s[0] * s[1] // 128 for s in relative), table[-1][1])
        entries = ", ".join(f"({last}, ({n.replace('x', ', ')}))" for last, n in table)
        print(f"    {device}: ({entries}{',' if len(table) == 1 else ''}),")

        def pick(shape, table=table):
            units = shape[0] * shape[1] // 128
            return next((n for last, n in table if units <= last), table[-1][1])

        for low_m, high_m in ((1, 4), (8, 64), (128, 512), (768, 4096), (6144, 32768)):
            band = [s for s in relative if low_m <= s[0] <= high_m]
            if band:
                sgl = [relative[s]["sgl"] / relative[s][pick(s)] for s in band]
                rule = [relative[s][_rule(s[0])] / relative[s][pick(s)] for s in band]
                print(
                    f"      rows {low_m}-{high_m}: sgl/table {_geomean(sgl):.3f} "
                    f"(min {min(sgl):.3f}), rule/table {_geomean(rule):.3f} (min {min(rule):.3f})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--emit", nargs="+", metavar="LOG")
    args = parser.parse_args()
    if args.emit:
        emit(args.emit)
    if args.sweep:
        part, parts = (int(v) for v in args.part.split("/"))
        sweep(part, parts, args.iterations)


if __name__ == "__main__":
    main()
