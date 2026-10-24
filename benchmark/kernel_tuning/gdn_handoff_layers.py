# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""R3d host submission, event timing, and fused CUDA graph replay benchmark."""

import argparse
import json
from statistics import median
from time import perf_counter

import torch

from ctikernels.gdn import GatedDeltaNetReplay


def allocate(specs, device):
    return {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device=device)
        for name, spec in specs.items()
    }


def host_time(call, count):
    torch.cuda.synchronize()
    samples = []
    for _ in range(count):
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1e6)
    torch.cuda.synchronize()
    return median(samples)


def device_time(call, count):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", "--J", type=int, default=24)
    parser.add_argument("--geometry", type=int, nargs=2, default=(16, 32), metavar=("HK", "HV"))
    parser.add_argument("--slots", "--R", type=int, default=4)
    parser.add_argument("--active-rows", type=int, default=None)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--calls", "--N", type=int, default=200)
    parser.add_argument("--checks", choices=("strict", "trusted"), default="strict")
    parser.add_argument("--out-of-place", action="store_true")
    args = parser.parse_args(argv)
    active = args.slots if args.active_rows is None else args.active_rows
    if min(args.layers, args.slots, args.calls) < 1 or not 0 <= active <= args.slots:
        parser.error("positive layer/slot/call counts and 0 <= active rows <= slots required")
    hk, hv = args.geometry
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        window=args.window,
        device="cuda",
        provider="cute",
        checks=args.checks,
    )
    if op.provider["handoff_layers"] != "cute":
        parser.error("fused benchmark requires nvidia-cutlass-dsl 4.8.0")
    states = [
        op.create_state(
            buffers=allocate(op.state_spec(num_slots=args.slots, num_snapshot_rows=0), op.device)
        )
        for _ in range(args.layers)
    ]
    canonical = [s.checkpoint.clone() if args.out_of_place else s.checkpoint for s in states]
    histories = [s.conv_history.clone() if args.out_of_place else s.conv_history for s in states]
    for source, history in zip(canonical, histories, strict=True):
        source.normal_()
        history.normal_()
    table = allocate(op.layer_table_spec(num_layers=args.layers), op.device)["layer_table"]
    binding = op.bind_layers(
        states=states, table=table, canonical_states=canonical, histories=histories
    )
    controls = op.bind_graph_buffers(
        buffers=allocate(
            op.graph_buffer_spec(batch_size=args.slots, num_layers=args.layers), op.device
        )
    )
    status = controls.status(0)
    status.sequence.fill_(1)
    op.begin(status=status)
    slots = torch.arange(args.slots, dtype=torch.int32, device=op.device)
    slots[active:] = -1
    positions = torch.full((args.slots,), 1024, dtype=torch.int64, device=op.device)

    def per_layer():
        for j, state in enumerate(states):
            op.handoff(
                canonical_state=canonical[j],
                history=histories[j],
                next_positions=positions,
                slot_indices=slots,
                state=state,
                status=status.for_layer(j),
            )

    def fused():
        op.handoff_layers(
            binding,
            next_positions=positions,
            slot_indices=slots,
            status=status if args.checks == "strict" else None,
        )

    for call in (per_layer, fused):
        for _ in range(5):
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    for name, call in (("per_layer", per_layer), ("fused", fused)):
        result = {
            "mode": name,
            "layers": args.layers,
            "geometry": [hk, hv],
            "slots": args.slots,
            "active_rows": active,
            "checks": "strict" if name == "per_layer" else args.checks,
            "host_us": host_time(call, args.calls),
            "device_us": device_time(call, args.calls),
            "launches_per_call": 5 * args.layers
            if name == "per_layer"
            else 2
            if args.checks == "strict"
            else 1,
        }
        if name == "fused":
            result["graph_host_us"] = host_time(graph.replay, args.calls)
            result["graph_device_us"] = device_time(graph.replay, args.calls)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
