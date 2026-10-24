# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Warm-up autotune against the offline table: hide shapes, re-decide, compare.

Each selected (N, K) is removed from this device's table, so autotune treats it
as a hole. The warm-up then decides every bucket of the engine M set. One RESULT
JSON line per key reports the table decision, the autotune decision, its gate
numbers and the tuning time. Agreement on provider is the headline; a different
CuTe configuration is still an agreement if both qualify.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

QWEN35_9B = "4096:4096,4096:12288,12352:4096,10240:4096,24576:4096,6144:4096"


def emit(record):
    print("RESULT " + json.dumps(record, sort_keys=True), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default=QWEN35_9B, help="N:K,N:K")
    ap.add_argument("--m", default="1,2,4,8,16,24,32")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget-seconds", type=float, default=1500)
    args = ap.parse_args(argv)
    shapes = [tuple(int(v) for v in item.split(":")) for item in args.shapes.split(",")]
    ms = sorted({int(v) for v in args.m.split(",")})

    from ctikernels.linear import Linear, _autotune, autotune
    from ctikernels.linear import _dispatch as dispatch
    from ctikernels.linear._dispatch import _hardware

    device = torch.device("cuda", torch.cuda.current_device())
    kind, capability, sms = _hardware(device)
    emit(
        {
            "kind": "autotune_check_begin",
            "device": torch.cuda.get_device_name(),
            "device_class": kind,
            "hardware": [*capability, sms],
            "shapes": shapes,
            "ms": ms,
        }
    )
    table = dispatch.TUNED.get(kind, {})
    shown = {shape: dict(table.get(shape, {})) for shape in shapes}
    for shape in shapes:
        table.pop(shape, None)
    cache = Path(tempfile.mkdtemp()) / "autotune.json"
    started = time.monotonic()
    with (
        torch.inference_mode(),
        autotune(ms=ms, budget_seconds=args.budget_seconds, top_k=args.top_k, cache=str(cache)),
    ):
        for n, k in shapes:
            gen = torch.Generator(device="cuda").manual_seed(n + k)
            weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
            x = torch.randn(max(ms), k, dtype=torch.bfloat16, device="cuda", generator=gen)
            op = Linear(provider="auto", autotune="cache")
            for m in ms:
                begun = time.monotonic()
                op(x[:m], weight)
                torch.cuda.synchronize()
                seconds = time.monotonic() - begun
                bucket = dispatch.m_bucket(m)
                record = _autotune.lookup(_autotune.identity(device), n, k, 0, bucket)
                offline = None
                for (lo, hi), config in shown[n, k].items():
                    if lo <= bucket[1] <= hi:
                        offline = config
                tuned = _autotune.record_config(record, bucket)
                emit(
                    {
                        "kind": "autotune_check",
                        "device_class": kind,
                        "N": n,
                        "K": k,
                        "M": m,
                        "bucket": list(bucket),
                        "table_provider": "cute" if offline else "cublas",
                        "table_config": None if offline is None else offline.__dict__,
                        "autotune_provider": "cute" if tuned else "cublas",
                        "agree": bool(offline) == bool(tuned),
                        "same_config": offline == tuned,
                        "provider_after_call": op.provider,
                        "seconds": seconds,
                        "record": record,
                    }
                )
            del weight, x, op
    emit(
        {
            "kind": "autotune_check_end",
            "device_class": kind,
            "elapsed_seconds": time.monotonic() - started,
            "complete": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
