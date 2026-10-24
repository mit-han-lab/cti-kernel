# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Probe each shared projection independently with interleaved CUDA-graph timings.

Run as a module from the repository root. Each call includes destination zeroing.
Feed the JSON lines to moe_lora_shared_table to generate measured dispatch choices.
"""

import argparse
import itertools
import json
import math
import random
import statistics
from contextlib import contextmanager


def dense_name(config):
    return "d" + ".".join(map(str, config))


def dense_candidates(rows, k, sms, default):
    """Stage one: vary shrink settings while retaining heuristic expand settings."""
    configs = [default]
    tiles = (k + 63) // 64
    for bm, sk, stages in itertools.product((16, 32, 64), (1, 2, 4, 8, 16, 32), (2, 3, 4)):
        if sk > tiles or (rows + bm - 1) // bm * sk > 32 * sms:
            continue
        if stages > 2 and (tiles + sk - 1) // sk < stages - 1:
            continue
        config = (bm, sk, default[2], default[3], stages)
        if config not in configs:
            configs.append(config)
    return configs


def expand_candidates(config):
    bm, sk, _, _, stages = config
    return [(bm, sk, ebm, bn, stages) for ebm, bn in itertools.product((16, 32, 64), (64, 128))]


@contextmanager
def force_mode(mode, config=None):
    from ctikernels.moe.cute import _lora_dense as _cute_dense
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    old_mode, old_config = _shared_policy.SHARED_MODE_OVERRIDE, dict(_cute_dense.DENSE_OVERRIDE)
    _shared_policy.SHARED_MODE_OVERRIDE = mode
    _cute_dense.DENSE_OVERRIDE.clear()
    if config is not None:
        _cute_dense.DENSE_OVERRIDE["config"] = config
    try:
        yield
    finally:
        _shared_policy.SHARED_MODE_OVERRIDE = old_mode
        _cute_dense.DENSE_OVERRIDE.clear()
        _cute_dense.DENSE_OVERRIDE.update(old_config)


class GraphSample:
    def __init__(self, calls):
        import torch

        self.count = len(calls)
        # Keep captured input/output pointers alive through their closures.
        self._calls = calls
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for call in calls:
                    call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for call in calls:
                call()
        self.graph.replay()
        torch.cuda.synchronize()
        self.repeats = max(1, math.ceil(2.0 / max(self.elapsed(1), 1e-6)))

    def elapsed(self, repeats):
        import torch

        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(repeats):
            self.graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def sample(self):
        return self.elapsed(self.repeats) * 1000 / (self.repeats * self.count)


def benchmark_row(rows, rank, proj, hidden, intermediate, sms, device, capability, seed):
    import torch

    from ctikernels.moe.cute import _lora_dense as _cute_dense
    from ctikernels.moe.lora import shared_lora_w2_, shared_lora_w13_

    k, n, slices = (hidden, intermediate, 2) if proj == "w13" else (intermediate, hidden, 1)
    projection = shared_lora_w13_ if proj == "w13" else shared_lora_w2_
    rng = random.Random(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape, scale=1.0):
        return (
            torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator) * scale
        ).contiguous()

    a = normal((slices, rank, k), k**-0.5)
    b = normal((slices, n, rank), 0.1 / math.sqrt(rank))
    factors = (a, b) if proj == "w13" else (a[0], b[0])
    pool = [normal((rows, k)) for _ in range(4)]
    expected = (
        torch.cat(
            [(pool[-1].float() @ ai.float().T) @ bi.float().T for ai, bi in zip(a, b, strict=True)],
            dim=-1,
        )
        * 0.5
    )
    with force_mode("dense"):
        default = _cute_dense.dense_config(rows, k, n, rank, slices, sms)

    def measure(candidates):
        graphs, times = {}, {}
        for name, mode, config in candidates:
            dst = torch.empty((rows, slices * n), device="cuda", dtype=torch.bfloat16)

            def run(x, dst=dst, mode=mode, config=config):
                dst.zero_()
                with force_mode(mode, config):
                    projection(dst, x, factors, scaling=0.5, provider="cute")

            calls = [lambda x=x, run=run: run(x) for x in pool]
            try:
                calls[-1]()
                err = float((dst.float() - expected).norm() / expected.norm().clamp_min(1e-30))
                if not math.isfinite(err) or err >= 1e-2:
                    times[name] = f"bad {err}"
                    continue
                graph = GraphSample(calls)
                err = float((dst.float() - expected).norm() / expected.norm().clamp_min(1e-30))
                if not math.isfinite(err) or err >= 1e-2:
                    times[name] = f"bad {err}"
                    continue
                graphs[name] = graph
            except Exception as error:  # noqa: BLE001 - probe failures are recorded per candidate
                times[name] = type(error).__name__
        samples = {name: [] for name in graphs}
        for _ in range(7):
            order = list(graphs)
            rng.shuffle(order)
            for name in order:
                if name in times:
                    continue
                try:
                    samples[name].append(graphs[name].sample())
                except Exception as error:  # noqa: BLE001 - continue probing other candidates
                    times[name] = type(error).__name__
        times.update(
            {
                name: statistics.median(values)
                for name, values in samples.items()
                if name not in times
            }
        )
        return times

    configs = dense_candidates(rows, k, sms, default)
    candidates = [(mode, mode, None) for mode in ("cute", "split", "library")]
    candidates += [(dense_name(config), "dense", config) for config in configs]
    times = measure(candidates)
    valid_dense = [config for config in configs if isinstance(times[dense_name(config)], float)]
    if valid_dense:
        best_shrink = min(valid_dense, key=lambda config: times[dense_name(config)])
        # Include the existing expand choice as an anchor in stage two.
        times.update(
            measure(
                [(dense_name(config), "dense", config) for config in expand_candidates(best_shrink)]
            )
        )
    valid = {name: time for name, time in times.items() if isinstance(time, float)}
    return {
        "device": device,
        "sms": sms,
        "capability": capability,
        "T": rows,
        "r": rank,
        "proj": proj,
        "K": k,
        "N": n,
        "default": list(default),
        "best": min(valid, key=valid.get) if valid else None,
        "times": times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--proj", default="w13,w2", help="comma-separated projections: w13,w2")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    projections = args.proj.split(",")
    if any(proj not in ("w13", "w2") for proj in projections):
        parser.error("proj must contain w13 and/or w2")
    if any(t < 1 for t in args.tokens) or any(r < 8 or r > 128 or r % 8 for r in args.ranks):
        parser.error("tokens must be positive; ranks must be multiples of 8 in [8,128]")
    if any(d <= 0 or d % 8 for d in (args.hidden, args.intermediate)):
        parser.error("hidden and intermediate must be positive multiples of 8")
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = (prop.major, prop.minor)
    if capability not in ((12, 0), (12, 1)):
        parser.error("sm_120 or sm_121 is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows, rank, proj in itertools.product(args.tokens, args.ranks, projections):
        result = benchmark_row(
            rows,
            rank,
            proj,
            args.hidden,
            args.intermediate,
            prop.multi_processor_count,
            prop.name,
            capability,
            args.seed + rows * 1009 + rank * 17,
        )
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
