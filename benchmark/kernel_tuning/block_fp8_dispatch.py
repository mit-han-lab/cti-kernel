# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Measure the dense block-FP8 dispatch table (``ctikernels.linear._block_fp8_dispatch``).

``--sweep``: on the current device, time ``BlockFP8Linear`` forced to ``triton-fp8`` and to
``flashinfer-cutlass`` (and, on compute capability 12.1, to ``cuda-sm12x``) at the
Qwen3.5-35B-A3B-FP8 projection shapes and their equal TP=2 and
TP=4 shards (N or K split; packed weights), over a batch-size grid. Each sample is one
CUDA-graph replay of a call per weight copy (copies exceed the L2 cache); one ``DSWEEP`` JSON
line per (shape, batch) with the median microseconds of each provider. ``--part i/n`` runs
every n-th shape.

``--emit LOG...``: read ``DSWEEP`` lines and print ``TABLE`` entries: per device and shape,
runs of consecutive grid batch sizes won by the same provider, each as ``(last_m,
provider)``, batches below the grid on ``triton-fp8`` (``--margin 0.02``: among providers
within 2% of the fastest, the previous batch's is kept, else the one fastest at the next
batch). A batch runs the provider of the first run whose ``last_m`` it does not exceed
(the last run's above the grid): FlashInfer's time is flat within each 128-row tile band and
the grid holds every multiple of 128 up to 1024, so the grid point at or above a batch has
FlashInfer's cost of that batch and Triton's cost of at least it.

usage: python -m benchmark.kernel_tuning.block_fp8_dispatch --sweep [--part 0/2] [--max-batch 512]
       python -m benchmark.kernel_tuning.block_fp8_dispatch --emit sweep-*.log [--margin 0.02]
(``--emit`` keeps the last line of each (device, shape, batch): a later partial sweep
replaces those batches of an earlier one.)
"""

from __future__ import annotations

import argparse
import json
from functools import partial

BATCHES = (32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
BATCHES += (1536, 2048, 3072, 4096)
PROVIDERS = ("triton-fp8", "flashinfer-cutlass", "cuda-sm12x")
CODES = {"triton-fp8": "_T", "flashinfer-cutlass": "_F", "cuda-sm12x": "_C"}
# The devices whose sweeps time the local CUDA GEMM (GB10); the others keep their two-way runs.
SM12X_CAPABILITIES = ((12, 1),)


def shapes() -> list[tuple[int, int]]:
    from benchmark.kernel_tuning.block_fp8 import DENSE_SHAPES

    found = []
    for n, k in DENSE_SHAPES.values():
        for tp in (1, 2, 4):
            for shape in ((n // tp, k), (n, k // tp)):
                if shape not in found and shape[0] % 128 == 0 and shape[1] % 128 == 0:
                    found.append(shape)
    return found


def sweep(part: int, parts: int, iterations: int, max_batch: int = BATCHES[-1]) -> None:
    import torch

    from benchmark.kernel_tuning.block_fp8 import _median_us, _random_weight
    from ctikernels.linear.block_fp8 import BlockFP8Linear

    torch.manual_seed(0)
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    base = {"cc": [device.major, device.minor], "sms": device.multi_processor_count}
    with torch.inference_mode():
        for n, k in shapes()[part::parts]:
            copies = max(2, -(-(512 << 20) // (n * k)))
            weights = [_random_weight(n, k) for _ in range(copies)]
            providers = (
                PROVIDERS if (device.major, device.minor) in SM12X_CAPABILITIES else PROVIDERS[:2]
            )
            ops = {p: [BlockFP8Linear(*w, provider=p) for w in weights] for p in providers}
            for m in (m for m in BATCHES if m <= max_batch):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                times = {
                    p: _median_us(
                        [lambda op=op, x=x: op(x) for op in path_ops],
                        iterations,
                        warmups=[
                            partial(op.warm_up, batch_sizes=[m], dtype=x.dtype) for op in path_ops
                        ],
                    )
                    for p, path_ops in ops.items()
                }
                print("DSWEEP " + json.dumps({**base, "N": n, "K": k, "M": m, **times}), flush=True)
            del ops, weights
            torch.cuda.empty_cache()


def emit(logs: list[str], margin: float = 0.0) -> None:
    rows: dict[tuple, dict[tuple[int, int], dict[int, dict[str, float]]]] = {}
    for path in logs:
        with open(path) as handle:
            for line in handle:
                if not line.startswith("DSWEEP "):
                    continue
                r = json.loads(line[len("DSWEEP ") :])
                times = {p: r[p] for p in PROVIDERS if p in r}
                device = (*r["cc"], r["sms"])
                rows.setdefault(device, {}).setdefault((r["N"], r["K"]), {})[r["M"]] = times
    for device, table in sorted(rows.items()):
        print(f"    {device}: {{")
        for shape, by_batch in sorted(table.items()):
            # Batches below the grid stay on the Triton GEMM (its fused small-batch path).
            buckets: list[tuple[int, str]] = [(BATCHES[0] - 1, PROVIDERS[0])]
            batches = sorted(by_batch)
            for i, m in enumerate(batches):
                times = by_batch[m]
                best = min(times.values())
                # Hysteresis: among the providers within ``margin`` of the fastest, keep the
                # previous batch's, else take the one fastest at the next batch (single-point
                # wins within the noise do not split runs).
                close = [p for p in times if times[p] <= (1 + margin) * best]
                if buckets[-1][1] in close:
                    name = buckets[-1][1]
                else:
                    ahead = by_batch[batches[i + 1]] if i + 1 < len(batches) else times
                    name = min(close, key=lambda p: (ahead.get(p, float("inf")), times[p]))
                if buckets[-1][1] == name:
                    buckets[-1] = (m, name)
                else:
                    buckets.append((m, name))
            entries = ", ".join(f"({m}, {CODES[n]})" for m, n in buckets)
            print(f"        {shape}: ({entries}{',' if len(buckets) == 1 else ''}),")
        print("    },")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-batch", type=int, default=BATCHES[-1], help="sweep batches up to")
    parser.add_argument("--emit", nargs="+", metavar="LOG")
    parser.add_argument("--margin", type=float, default=0.0, help="hysteresis of --emit")
    args = parser.parse_args()
    if args.emit:
        emit(args.emit, args.margin)
    if args.sweep:
        part, parts = (int(v) for v in args.part.split("/"))
        sweep(part, parts, args.iterations, args.max_batch)


if __name__ == "__main__":
    main()
