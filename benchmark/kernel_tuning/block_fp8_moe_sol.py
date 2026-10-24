# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Speed-of-light floors for Qwen routed block-FP8 MoE (default: Qwen3.5).

E=256, top_k=8, H=2048, INTERMEDIATE=512; R=M*top_k. E4M3 weights have shapes
w13=[E,1024,2048], w2=[E,2048,512]; FP32 scales are [E,8,16], [E,16,4].
Each expert has 3 MiB of weights plus 768 bytes of scales. Touched weights
are modeled cold: the benchmark rotates 8 routings over 768 MiB of experts,
larger than every device's L2. This is a cold-weight assumption, not a claim
that every benchmark access misses L2. At large M, each expert's weights
are assumed read once, with perfect reuse across its assigned routes.

Uniform routing draws 8 distinct experts per token, independently across
tokens: E_unique=256*(1-(1-8/256)**M). The benchmark's
torch.randn(M,256).softmax(-1).topk(8) is essentially uniform by symmetry,
so both routing models have the same expectation (ignoring numerical ties).

Decode: 2 kernels, x BF16 read once, h BF16 [R,INTERMEDIATE] write/read, out BF16 write.
Grouped: 3 kernels minimum (gate/up+activation, down, reduction), additionally
writing/reading BF16 [R,H] down outputs. Routing, quantization, extra kernels,
non-MMA arithmetic, padding and redundant loads are excluded. FLOPs count
only the three projections; unit-scale MXF8 MMA is an optimistic ceiling.
SOL=max(total_bytes/DRAM_read_Bps, FLOPs/MXF8_FLOPs)+kernels*graph_node_cost.
GB/s and TFLOPS are decimal; MiB is binary. No GPU or torch is required.
"""

import argparse
import json
import math

# Measured ceilings: benchmark/sol/ceilings/*.json.
# Tuple: DRAM read GB/s, unit-scale MXF8 m16n8k32 TFLOPS, graph node us,
# L2 MiB, SM count, plain E4M3 TFLOPS (if supplied), copy-style GB/s (if supplied).
DEVICES = {
    "RTX 5090": (1680.4, 1001.8, 0.391, 96, 170, 504.7, None),
    "RTX PRO 6000": (1533.0, 934.4, 0.484, 128, 188, 934.8, None),
    "GB10": (243.5, 247.9, 0.410, 24, 48, None, 219.3),
}
SHAPES = {"qwen3.5": (256, 2048, 512, 8), "qwen3": (128, 2048, 768, 8)}
BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
CHAINS = {"decode": 2, "grouped": 3}


def estimate(device, m, chain, shape="qwen3.5", ep=1):
    e, h, i, top_k = SHAPES[shape]
    if ep not in (1, 2):
        raise ValueError("EP must be 1 or 2")
    bandwidth, peak, node_us, *_ = DEVICES[device]
    routes = m * top_k / ep
    unique = -(e // ep) * math.expm1(m * math.log1p(-top_k / e))
    weight_scale_bytes = unique * (3 * i * h + 4 * (3 * i * h // 128**2))
    activation_bytes = 4 * m * h + 4 * routes * i
    if chain == "grouped":
        activation_bytes += 4 * routes * h
    flops = 6 * routes * i * h
    dram_us = (weight_scale_bytes + activation_bytes) / (bandwidth * 1e3)
    mma_us = flops / (peak * 1e6)
    launch_us = CHAINS[chain] * node_us
    return (
        unique,
        weight_scale_bytes,
        activation_bytes,
        flops,
        dram_us,
        mma_us,
        launch_us,
        max(dram_us, mma_us) + launch_us,
    )


def print_tables(shape="qwen3.5", ep=1):
    e, h, i, top_k = SHAPES[shape]
    print("# Block-FP8 routed MoE speed-of-light floors\n")
    print(
        f"{shape}: E_global={e}, E_local={e // ep}, EP={ep}, "
        f"top_k={top_k}, H={h}, I={i}; expected local R={top_k / ep:g}M.\n"
    )
    print("Ceilings: `benchmark/sol/ceilings/*.json`.\n")
    print(f"Unique local experts: {e // ep}*(1-(1-{top_k}/{e})^M), for uniform routing and")
    print("benchmark randn/softmax/topk routing. W+S includes FP32 scales.\n")
    print(f"Cold weights; local expert weights total {3 * h * i * (e // ep) / 2**20:g} MiB.")
    print("Cache misses are a modeling assumption. Each touched expert is")
    print("read once even at large M. Input x is read once; h is written/read;")
    print("out is written. Grouped also writes/reads per-route down outputs.\n")
    print("Minimum kernels: decode=2; grouped=3 (gate/up+activation, down, reduction).")
    print("Routing, quantization, extra launches, padding and non-MMA work excluded.")
    print("FLOPs=2*R*(2*INTERMEDIATE*H+H*INTERMEDIATE). SOL=max(DRAM us,MMA us)+launch us.\n")
    for device, (bandwidth, peak, node, l2, sms, plain, copy) in DEVICES.items():
        print(f"## {device}\n")
        print(f"DRAM read {bandwidth} GB/s; unit-scale MXF8 {peak} TFLOPS;")
        print(f"graph node {node:.3f} us; L2 {l2} MiB; {sms} SMs.")
        if plain is not None:
            print(f"Plain E4M3 {plain} TFLOPS (reference only).")
        if copy is not None:
            print(f"Copy-style bandwidth {copy} GB/s (reference only).")
        print(
            "\n| M | chain | unique | W+S MiB | act bytes | GFLOPs | DRAM us | MMA us | launch us | sol_us |"
        )
        print("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in BATCHES:
            for chain in CHAINS:
                unique, ws, act, flops, dram, mma, launch, sol = estimate(
                    device, m, chain, shape, ep
                )
                print(
                    f"| {m} | {chain} | {unique:.3f} | {ws / 2**20:.6f} | {act} | "
                    f"{flops / 1e9:.6f} | {dram:.3f} | {mma:.3f} | {launch:.3f} | {sol:.3f} |"
                )
        print()


def print_measured(path, shape="qwen3.5", ep=1):
    print("## Measured utilization\n")
    print("Utilization = sol_us / measured_us (unclamped); both chain assumptions shown.\n")
    print(
        "| device | M | label | measured us | decode sol_us | decode util | grouped sol_us | grouped util |"
    )
    print("|:---|---:|:---|---:|---:|---:|---:|---:|")
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                device = row["device"].removeprefix("NVIDIA ")
                m, us, label = row["M"], row["us"], row["label"]
                if (
                    device not in DEVICES
                    or type(m) is not int
                    or m <= 0
                    or type(us) not in (int, float)
                    or not math.isfinite(us)
                    or us <= 0
                    or not isinstance(label, str)
                ):
                    raise ValueError
                label = label.replace("|", "&#124;").replace("\n", " ").replace("\r", " ")
                decode, grouped = (estimate(device, m, chain, shape, ep)[-1] for chain in CHAINS)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                raise ValueError(f"Invalid measured record at line {line_number}") from None
            print(
                f"| {device} | {m} | {label} | {us:.3f} | {decode:.3f} | "
                f"{decode / us:.2%} | {grouped:.3f} | {grouped / us:.2%} |"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", metavar="FILE", help="JSONL: device, M, us, label")
    parser.add_argument("--shape", choices=SHAPES, default="qwen3.5")
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print_tables(args.shape, args.ep)
    if args.measured:
        try:
            print_measured(args.measured, args.shape, args.ep)
        except (OSError, ValueError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
