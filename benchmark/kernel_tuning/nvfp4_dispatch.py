# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Emit the NVFP4 dispatch tables (``ctikernels.linear._nvfp4_dispatch``) from benchmark logs.

``python -m benchmark.kernel_tuning.nvfp4_dispatch LOG...`` reads the ``BENCH`` lines of
``benchmark.kernel_tuning.nvfp4`` runs and prints the ``GATED`` entries: per device type and
shape, runs of consecutive batch sizes (powers of two) won by the same candidate, each as
``(last batch, path, configuration)``. The default (QuACK ``p128x128`` fused), then the previous
batch size's winner, keeps a batch unless another candidate is more than ``MARGIN`` faster, so
noise does not churn the table. Repeated measurements of a candidate take their median. Linear
has no table (FlashInfer for every M); its records are ignored.

With ``--candidates``, read parity-harness ``CAND`` records instead. Keep the current
dispatched plan within 2% of the fastest owned candidate, coalesce equal winners,
and print CuTe configurations as ``Cute(...)`` placeholders. SGLang is never eligible.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

from benchmark.kernel_tuning.nvfp4 import CONFIGS, SMALL

BASELINES = frozenset({"bf16", "ours", "sglang"})
MARGIN = 1.03
DEFAULT_CONFIG = "p128x128"
SHAPE_KEYS = {"gated": ("H", "I"), "linear": ("N", "K")}


def _literal(path: str, label: str) -> str:
    if path == "flashinfer" or label == "flashinfer":
        return "_F()"
    if path == "triton":
        splits, n, k, warps, stages, pre = SMALL[label]
        return f"_S({splits}, {n}, {k}, {warps}, {stages}, {pre})"
    tile_m, tile_n, pingpong, dynamic = CONFIGS[label]
    return f"_Q({tile_m}, {tile_n}, {pingpong}, {dynamic})"


def _winner(op: str, times: dict[tuple[str, str], float], previous) -> tuple[str, str]:
    """The fastest candidate, unless the default or the previous batch size's winner is within
    ``MARGIN`` of it (in that order)."""
    best = min(times, key=times.get)
    for keep in (("fused" if op == "gated" else "quack", DEFAULT_CONFIG), previous):
        if keep in times and times[keep] <= MARGIN * times[best]:
            return keep
    return best


def emit(logs: list[str]) -> None:
    samples: dict[tuple, dict[int, dict[tuple[str, str], list[float]]]] = {}
    for log in logs:
        with open(log) as handle:
            for line in handle:
                if not line.startswith("BENCH "):
                    continue
                r = json.loads(line[len("BENCH ") :])
                if r["op"] == "linear" or "us" not in r or r["path"] in BASELINES:
                    continue
                path = {"nvfp4": "quack"}.get(r["path"], r["path"])
                device = (r["device"], tuple(r["capability"]))
                shape = tuple(r[key] for key in SHAPE_KEYS[r["op"]])
                case = samples.setdefault((r["op"], device, shape), {}).setdefault(r["M"], {})
                case.setdefault((path, r["config"]), []).append(r["us"])
    for op in ("gated",):
        print(f"{op.upper()} = {{")
        devices = sorted({device for o, device, _ in samples if o == op})
        for device in devices:
            print(f"    {device!r}: {{")
            for (o, d, shape), cases in sorted(samples.items()):
                if (o, d) != (op, device):
                    continue
                runs: list[tuple[int, tuple[str, str]]] = []
                for m, times in sorted(cases.items()):
                    medians = {key: statistics.median(us) for key, us in times.items()}
                    winner = _winner(op, medians, runs[-1][1] if runs else None)
                    if runs and runs[-1][1] == winner:
                        runs[-1] = (m, winner)
                    else:
                        runs.append((m, winner))
                entries = []
                for m, (path, label) in runs:
                    literal = _literal(path, label)
                    if op == "gated":
                        name = {"fused": "FUSED", "unfused": "UNFUSED", "triton": "TRITON"}[path]
                        entries.append(f"({m}, {name}, {literal})")
                    else:
                        entries.append(f"({m}, {literal})")
                comma = "," if len(entries) == 1 else ""
                print(f"        {shape}: ({', '.join(entries)}{comma}),")
            print("    },")
        print("}")


def _candidate_literal(record):
    provider, config = record["provider"], record["config"]
    constructor = {"quack": "_Q", "triton": "_S", "flashinfer": "_F", "cute": "Cute"}[provider]
    if provider == "cute":
        from ctikernels.linear._nvfp4_dispatch import Cute

        if not isinstance(config, dict):
            config = Cute(*config)._asdict()
        else:
            config = dict(config)
        if config.pop("probe", None) is not None:
            raise ValueError("diagnostic CuTe probes cannot enter dispatch tables")
        plan = Cute(**config)
        plan.config()
        return f"Cute({', '.join(f'{key}={value!r}' for key, value in plan._asdict().items())})"
    values = config.values() if isinstance(config, dict) else config
    if any(type(value) not in (int, bool) for value in values):
        raise ValueError("candidate configuration fields must be integers or booleans")
    if isinstance(config, dict):
        if any(not key.isidentifier() for key in config):
            raise ValueError("candidate configuration field names must be identifiers")
        fields = ", ".join(f"{key}={value!r}" for key, value in config.items())
    else:
        fields = ", ".join(repr(value) for value in config)
    return f"{constructor}({fields})"


def emit_candidates(logs: list[str]) -> None:
    """Coalesce gated buckets with 2% incumbent hysteresis; Linear records are ignored."""
    samples, incumbents = {}, {}
    for log in logs:
        with open(log) as handle:
            for line in handle:
                if not line.startswith("CAND "):
                    continue
                record = json.loads(line[len("CAND ") :])
                if (
                    record["op"] == "linear"
                    or record.get("provider") not in {"quack", "triton", "flashinfer", "cute"}
                    or "median_us" not in record
                    or record.get("status", "ok") != "ok"
                ):
                    continue
                op = record["op"]
                device = (record["device"], tuple(record["capability"]))
                shape = tuple(record[key] for key in SHAPE_KEYS[op])
                key = (op, device, shape)
                path = record["path"] if op == "gated" else None
                if op == "gated" and path not in {"quack-fused", "quack-unfused", "triton", "cute"}:
                    raise ValueError(f"unsupported gated candidate path: {path!r}")
                candidate = (path, _candidate_literal(record))
                times = record.get("samples_us", [record["median_us"]])
                if not times or any(not math.isfinite(t) or t <= 0 for t in times):
                    raise ValueError("candidate timings must be positive and finite")
                case = samples.setdefault(key, {}).setdefault(record["M"], {})
                case.setdefault(candidate, []).extend(times)
                if record.get("current"):
                    incumbent_key = (key, record["M"])
                    previous = incumbents.setdefault(incumbent_key, candidate)
                    if previous != candidate:
                        raise ValueError("logs disagree on the currently dispatched plan")
    for op in ("gated",):
        print(f"{op.upper()} = {{")
        for device in sorted({device for o, device, _ in samples if o == op}):
            print(f"    {device!r}: {{")
            for key, cases in sorted(samples.items()):
                o, d, shape = key
                if (o, d) != (op, device):
                    continue
                runs = []
                for m, times in sorted(cases.items()):
                    medians = {candidate: statistics.median(us) for candidate, us in times.items()}
                    best = min(medians, key=medians.get)
                    current = incumbents.get((key, m))
                    if current is None:
                        raise ValueError(f"missing current dispatch measurement for {key!r}, M={m}")
                    if medians[current] <= 1.02 * medians[best]:
                        best = current
                    if runs and runs[-1][1] == best:
                        runs[-1] = (m, best)
                    else:
                        runs.append((m, best))
                entries = []
                for m, (path, literal) in runs:
                    if op == "gated":
                        path_literal = {
                            "quack-fused": "FUSED",
                            "quack-unfused": "UNFUSED",
                            "triton": "TRITON",
                            "cute": "'cute'",
                        }[path]
                        entries.append(f"({m}, {path_literal}, {literal})")
                    else:
                        entries.append(f"({m}, {literal})")
                comma = "," if len(entries) == 1 else ""
                print(f"        {shape}: ({', '.join(entries)}{comma}),")
            print("    },")
        print("}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", metavar="LOG")
    parser.add_argument(
        "--candidates", action="store_true", help="read CAND logs with 2% hysteresis"
    )
    args = parser.parse_args()
    (emit_candidates if args.candidates else emit)(args.logs)


if __name__ == "__main__":
    main()
