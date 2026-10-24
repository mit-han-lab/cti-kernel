# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Summarize IKET 4.8 decoded JSON: repeated ranges summed per warp, max per CTA.

Means/medians are over launch/CTA samples, never summed across parallel warps.
Time is reported in source timestamp ticks unless --ns-per-tick is supplied.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def range_records(document):
    """Read interned 4.8 tables, accepting expanded RangeSpan exports too."""
    strings = document.get("stringTable", [])
    locations = document.get("locationTable", [])

    def walk(node, launch=None):
        if isinstance(node, list):
            for value in node:
                yield from walk(value, launch)
        elif isinstance(node, dict):
            if "gridId" in node:
                launch = (node.get("contextId", 0), node["gridId"])
            start = node.get("startTs", node.get("start_ts"))
            end = node.get("endTs", node.get("end_ts"))
            if start is not None and end is not None:
                name = node.get("rangeName", node.get("range_name"))
                if "rangeNameIdx" in node:
                    name = strings[node["rangeNameIdx"]]
                if isinstance(name, str) and name.startswith("mhc."):
                    locs = node.get("warp_locs", [])
                    if "warpLocIdxs" in node:
                        locs = [locations[index] for index in node["warpLocIdxs"]]
                    if launch is None or not locs:
                        raise ValueError("mHC range has no launch/CTA identity")
                    if end < start:
                        raise ValueError("range timestamps run backwards")
                    # Some exports retain a location for both endpoints and
                    # internal events. Our push/pop ranges stay in one warp.
                    if any(location != locs[0] for location in locs):
                        raise ValueError("mHC range crosses warp locations")
                    loc = locs[0]
                    cta = loc.get("ctaId", loc.get("cta_id"))
                    if isinstance(cta, dict):
                        cta = [cta[axis] for axis in ("x", "y", "z")]
                    warp = loc.get("warpId", loc.get("warp_id"))
                    yield launch, tuple(cta), warp, name, start, end
                return
            for key, value in node.items():
                if key not in ("stringTable", "locationTable"):
                    yield from walk(value, launch)

    yield from walk(document)


def summarize(document, last_launches=None, *, by_cta=False):
    records = list(range_records(document))
    launches = list(dict.fromkeys(record[0] for record in records))
    if last_launches is not None:
        launches = launches[-last_launches:]
    allowed = set(launches)
    warps = defaultdict(float)
    for launch, cta, warp, name, start, end in records:
        if launch in allowed:
            warps[launch, cta, warp, name] += end - start
    ctas = defaultdict(float)
    for (launch, cta, _warp, name), duration in warps.items():
        ctas[launch, cta, name] = max(ctas[launch, cta, name], duration)
    by_phase = defaultdict(list)
    for (_launch, cta, name), duration in ctas.items():
        by_phase[(name, cta) if by_cta else name].append(duration)
    if not by_phase:
        raise ValueError("no mHC ranges: use decoded *.trace.json with CTI_MHC_IKET=1")
    return by_phase


def timeline(document, last_launches=None, *, ns_per_tick=1.0):
    """CTA envelopes relative to wait completion; assume globaltimer nanoseconds.

    First compute min/median/max over participating CTAs in each launch, then
    average those statistics over launches. Repeated ranges use their envelope,
    not their summed duration. Missing phases do not contribute zero samples.
    """
    records = list(range_records(document))
    launches = list(dict.fromkeys(record[0] for record in records))
    if last_launches is not None:
        launches = launches[-last_launches:]
    if not launches:
        raise ValueError("no mHC ranges: use decoded *.trace.json with CTI_MHC_IKET=1")
    allowed = set(launches)
    ctas = {}
    for launch, cta, _warp, name, start, end in records:
        if launch in allowed:
            key = launch, cta, name
            previous = ctas.get(key, (start, end))
            ctas[key] = min(start, previous[0]), max(end, previous[1])
    factor = ns_per_tick / 1000
    samples = defaultdict(list)
    boundaries = []
    for launch in launches:
        ranges = [
            (name, start, end) for (grid, _, name), (start, end) in ctas.items() if grid == launch
        ]
        waits = [end for name, _, end in ranges if name == "mhc.grid_wait"]
        t0 = min(waits) if waits else min(start for _, start, _ in ranges)
        phases = defaultdict(list)
        for name, start, end in ranges:
            phases[name].append(((start - t0) * factor, (end - t0) * factor))
        for name, spans in phases.items():
            stats = []
            for values in zip(*spans, strict=True):
                stats.append((min(values), statistics.median(values), max(values)))
            samples[name].append(stats)
        arrivals = [start for name, start, _ in ranges if name in ("mhc.arrival", "mhc.exchange")]
        finishes = [end for name, _, end in ranges if name in ("mhc.output", "mhc.last_sinkhorn")]
        boundaries.append(
            {
                "launch": launch,
                "t0": t0,
                "last_cta_arrival_us": (max(arrivals) - t0) * factor if arrivals else None,
                "kernel_end_us": (max(finishes) - t0) * factor if finishes else None,
            }
        )
    phases = {
        name: {
            "launches": len(values),
            **{
                edge: tuple(statistics.mean(value[i][j] for value in values) for j in range(3))
                for i, edge in enumerate(("start", "end"))
            },
        }
        for name, values in samples.items()
    }
    return phases, boundaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="CTA start/end min/median/max in us; assumes ns ticks unless --ns-per-tick is set",
    )
    parser.add_argument("--by-cta", action="store_true", help="separate each CTA across launches")
    parser.add_argument(
        "--last-launches",
        type=int,
        default=32,
        help="keep final N instrumented launches, excluding prepare/warmup",
    )
    parser.add_argument(
        "--ns-per-tick",
        type=float,
        default=None,
        help="set 1 for globaltimer timestamps; omit for raw ticks",
    )
    args = parser.parse_args()
    if args.last_launches < 1 or (args.ns_per_tick is not None and args.ns_per_tick <= 0):
        parser.error("last-launches and ns-per-tick must be positive")
    for index, path in enumerate(args.trace):
        document = json.loads(path.read_text())
        if args.timeline:
            phases, boundaries = timeline(
                document, args.last_launches, ns_per_tick=args.ns_per_tick or 1.0
            )
            print(f"trace={index} phase launches start_us[min median max] end_us[min median max]")
            for name, stats in sorted(phases.items()):
                values = " ".join(
                    f"{value:12.3f}" for edge in ("start", "end") for value in stats[edge]
                )
                print(f"{name:24s} {stats['launches']:7d} {values}")
            for boundary in boundaries:
                arrival, end = boundary["last_cta_arrival_us"], boundary["kernel_end_us"]
                arrival = "missing" if arrival is None else f"{arrival:.3f}"
                end = "missing" if end is None else f"{end:.3f}"
                print(
                    f"launch={boundary['launch']} last CTA arrival_us={arrival} kernel_end_us={end}"
                )
            continue
        phases = summarize(document, args.last_launches, by_cta=args.by_cta)
        factor = 1 if args.ns_per_tick is None else args.ns_per_tick / 1000
        unit = "ticks" if args.ns_per_tick is None else "us"
        print(f"trace={index} phase CTA_samples mean_{unit} median_{unit}")
        for key, samples in sorted(phases.items()):
            name = f"{key[0]} CTA={key[1]}" if args.by_cta else key
            print(
                f"{name:24s} {len(samples):7d} "
                f"{statistics.mean(samples) * factor:12.3f} "
                f"{statistics.median(samples) * factor:12.3f}"
            )


if __name__ == "__main__":
    main()
