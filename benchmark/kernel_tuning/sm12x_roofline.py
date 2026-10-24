# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Roofline microbenchmarks of a compute capability 12.x GPU (GB10) for block-FP8 GEMMs.

One ``UB`` JSON line per measurement: device limits; mma.sync latency (one dependent chain)
and throughput per SM (E4M3 m16n8k32 with FP32 and FP16 accumulation, BF16 and FP16
m16n8k16) over warps per SM and independent accumulators; DRAM read/copy/write streaming
over 1 GiB; L2-resident reads over 8-64 MiB; shared-memory bandwidth by access stride (bank
conflicts) and ldmatrix; DRAM and L2 fills of shared memory by cp.async and by TMA bulk
copies. Builds ``sm12x_roofline.cu`` with ``torch.utils.cpp_extension`` on first use.

usage: python -m benchmark.kernel_tuning.sm12x_roofline
"""

from __future__ import annotations

import json
import os
import pathlib

import torch

HERE = pathlib.Path(__file__).resolve().parent
FLOP = {0: 2 * 16 * 8 * 32, 1: 2 * 16 * 8 * 32, 2: 2 * 16 * 8 * 16, 3: 2 * 16 * 8 * 16}
MMA = {0: "e4m3.f32acc.m16n8k32", 1: "e4m3.f16acc.m16n8k32", 2: "bf16.f32acc.m16n8k16"}
MMA[3] = "f16.f16acc.m16n8k16"
GIB = 1 << 30


def emit(**record) -> None:
    print("UB " + json.dumps(record), flush=True)


def build():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
    return load(
        f"sm12x_roofline_{arch}",
        [str(HERE / "sm12x_roofline.cu")],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
    )


def main() -> None:
    ub = build()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    emit(kind="props", name=props.name, sms=sms, l2=props.L2_cache_size,
         smem_per_sm=props.shared_memory_per_multiprocessor,
         threads_per_sm=props.max_threads_per_multi_processor,
         regs_per_sm=props.regs_per_multiprocessor)  # fmt: skip

    for kind in MMA:
        iters = 4096
        _, cycles = ub.mma_bench(kind, 1, 1, 1, iters)
        emit(kind="mma_latency", mma=MMA[kind], cycles_per_mma=cycles / iters)
        for warps in (4, 8, 16):
            for ilp in (4, 8):
                iters = 8192
                seconds, cycles = min(ub.mma_bench(kind, ilp, sms, warps, iters) for _ in range(3))
                emit(kind="mma_throughput", mma=MMA[kind], warps_per_sm=warps, ilp=ilp,
                     tflops=sms * warps * iters * ilp * FLOP[kind] / seconds / 1e12,
                     flop_per_clk_per_sm=warps * iters * ilp * FLOP[kind] / cycles,
                     clock_ghz=cycles / seconds / 1e9)  # fmt: skip

    src = torch.empty(GIB, dtype=torch.uint8, device="cuda").random_(0, 255)
    dst = torch.empty(GIB, dtype=torch.uint8, device="cuda")
    for mode, label in ((0, "read.cs"), (1, "read"), (2, "copy"), (3, "write")):
        best = max(
            GIB * (2 if mode == 2 else 1) / ub.mem_bench(src, dst, mode, sms * blocks, threads, 5)
            for blocks in (2, 8, 16)
            for threads in (256, 1024)
        )
        emit(kind="dram", op=label, gbs=best / 1e9)
    for mib in (8, 16, 24, 32, 64):
        part = src[: mib << 20]
        seconds = min(ub.mem_bench(part, dst, 1, sms * 4, 1024, 20) for _ in range(3))
        emit(kind="l2_read", mib=mib, gbs=(mib << 20) / seconds / 1e9)

    for stride in (1, 2, 4, 8, 16, 32):
        for v4 in (False, True):
            if v4 and stride > 8:
                continue
            iters, threads = 8192, 1024
            _, cycles = ub.smem_bench(stride, v4, 0, sms, threads, iters)
            emit(kind="smem", access=("v4" if v4 else "u32") + f" stride {stride}",
                 bytes_per_clk_per_sm=threads * iters * (16 if v4 else 4) / cycles)  # fmt: skip
    _, cycles = ub.smem_bench(1, False, 1, sms, 1024, 8192)
    emit(kind="smem", access="ldmatrix.x4", bytes_per_clk_per_sm=1024 * 8192 * 16 / cycles)

    for tma in (0, 1):
        path = "tma.bulk" if tma else "cp.async.cg"
        for blocks in (1, 2):
            seconds = ub.async_bench(src, tma, sms * blocks, 256, 4 // blocks if tma else 4, 3)
            if seconds > 0:
                emit(kind="dram_to_smem", path=path, blocks_per_sm=blocks, gbs=GIB / seconds / 1e9)
        for mib in (4, 8, 16):
            buffer = src[: mib << 20]
            emit(kind="l2_to_smem", path=path, mib=mib,
                 gbs=ub.l2_bench(buffer, tma, max(8, 512 // mib), sms, 256, 4) / 1e9)  # fmt: skip


if __name__ == "__main__":
    main()
