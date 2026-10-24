# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Eager cold-weight exit chain for run-iket; no graph capture or timed sleeps."""

import argparse
import functools
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(1, 4, 128, 1024), default=1)
    parser.add_argument("--weight-sets", type=int, default=16)
    parser.add_argument("--calls", type=int, default=None)
    args = parser.parse_args()
    if os.environ.get("CTI_MHC_IKET") != "1":
        parser.error("set CTI_MHC_IKET=1 before starting run-iket")
    if args.weight_sets < 1 or (args.calls is not None and args.calls < 1):
        parser.error("weight-sets and calls must be positive")

    import torch

    from ctikernels.mhc import MhcExit
    from ctikernels.mhc._reference import MhcConfig, make_state, make_weights

    from .bench import rotation_count

    cfg = MhcConfig(hidden=4096)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    count, footprint = rotation_count(cfg, args.tokens, args.weight_sets, props.L2_cache_size)
    calls = args.calls or 2 * count
    if calls < count:
        parser.error(f"cold rotation requires at least {count} calls")
    with torch.inference_mode():
        op = MhcExit(
            "cuda", hidden=4096, rms_eps=cfg.rms_eps, hc_eps=cfg.hc_eps, norm_eps=cfg.norm_eps
        )
        op.prepare([args.tokens])
        keepalive, launches = [], []
        for i in range(count):
            state = make_state(cfg, args.tokens, 1709 + i, "cuda")
            weights = make_weights(cfg, 2026 + i, "cuda")
            inputs = (
                state["x"],
                state["R"],
                state["post"],
                state["comb"],
                weights["fn"],
                weights["scale"],
                weights["base"],
                weights["norm_weight"],
            )
            outputs = op(*inputs)  # Validate and warm outside the measured chain.
            key = op._compile_key("exit", args.tokens)
            counter = op._counter_views.get(key[1], op._counter_views[4])
            # Bind the already-validated ABI once. Avoid per-call Python shape
            # checking and allocation gaps in this deliberately eager probe.
            launches.append(
                functools.partial(
                    op._compiled[key],
                    *[value.data_ptr() for value in (*inputs, *outputs)],
                    op._partials.data_ptr(),
                    counter,
                    args.tokens,
                    op._dummy_words,
                    0,
                    0,
                    0,
                    0,
                )
            )
            keepalive.append((inputs, outputs))
        torch.cuda.synchronize()
        print(
            f"T={args.tokens} calls={calls} weight_sets={count} fn_plus_R_bytes={footprint}",
            flush=True,
        )
        for i in range(calls):
            launches[i % count]()
        torch.cuda.synchronize()
        op.close()


if __name__ == "__main__":
    main()
