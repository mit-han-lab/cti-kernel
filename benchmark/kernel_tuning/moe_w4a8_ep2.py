# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DSV4 Flash routed MoE (+LoRA) as real EP2 across two GPUs (one process per GPU).

Every rank holds 128 of the 256 experts and the same tokens; it computes its local partial
(``expert_map``), then the exit all-reduce (NCCL) sums the partials. Rank 0 checks the result
against the single-node reference over all 256 experts, then both ranks time one layer end to
end (CUDA graphs, cold L2): ours (CuTe W4A8, fused LoRA) vs FlashInfer W4A8 (+ the separate
moe.lora stage). One JSON row per T on rank 0.

    python benchmark/kernel_tuning/moe_w4a8_ep2.py --rank R --master <rank-0 address> --ms 1 2048
"""

import argparse
import json

import torch
import torch.distributed as dist

from benchmark.kernel_tuning.moe_w4a8 import EXPERTS, routes, time_graph, weights
from benchmark.kernel_tuning.moe_w4a8_lora import lora_weights


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def layer_row(m, rank, cute, fi, full, lora, lora_full, expert_map):
    from ctikernels.moe import moe_align_block_size_local, to_local_topk_ids
    from ctikernels.moe.lora import moe_lora_w2_, moe_lora_w13_
    from ctikernels.moe.torch._w4a8_reference import mxfp4_experts

    rw, ids = routes(m)
    gen = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(m, 4096, device="cuda", generator=gen).to(torch.bfloat16)
    out_c, out_l, out_f = (torch.empty_like(x) for _ in range(3))
    local_ids = to_local_topk_ids(ids, expert_map)
    c13 = torch.zeros(m * 6, 4096, dtype=torch.bfloat16, device="cuda")
    c2 = torch.zeros(m * 6, 2048, dtype=torch.bfloat16, device="cuda")
    c3 = torch.zeros(m * 6, 4096, dtype=torch.bfloat16, device="cuda")

    def ours(out=out_c):
        cute(x, rw, ids, expert_map=expert_map, out=out)
        dist.all_reduce(out)

    def ours_lora(out=out_l):
        cute(x, rw, ids, expert_map=expert_map, out=out, lora=lora, scaling=2.0)
        dist.all_reduce(out)

    def flashinfer(out=out_f):
        fi(x, rw, ids, expert_map=expert_map, out=out)
        dist.all_reduce(out)

    local = lora.gate_a.shape[0]

    def flashinfer_lora_with(block_m):
        def flashinfer_lora(out=out_f):  # FlashInfer base + the separate moe.lora stage
            fi(x, rw, ids, expert_map=expert_map, out=out)
            # The engine aligns the local routes once per step (capture needs it passed in).
            aligned = moe_align_block_size_local(ids, block_m, expert_map, local)
            names = ("sorted_token_ids", "expert_ids", "num_tokens_post_padded")
            kw = dict(zip(names, aligned, strict=True), block_m=block_m, scaling=2.0, top_k=6)
            moe_lora_w13_(c13, x, lora, local_ids, **kw)
            moe_lora_w2_(c3, c2, lora, local_ids, rw, **kw)
            dist.all_reduce(out)

        return flashinfer_lora

    for fn in (ours, ours_lora, flashinfer):
        fn()
    torch.cuda.synchronize()
    row = {"M": m, "device": torch.cuda.get_device_name()}
    if rank == 0:
        kw = {"swiglu_limit": 10.0}
        ref = mxfp4_experts(x, *full, rw, ids, **kw)
        ref_lora = mxfp4_experts(x, *full, rw, ids, lora=lora_full.routed, scaling=2.0, **kw)
        row.update(
            rel_cute=rel(out_c, ref), rel_flashinfer=rel(out_f, ref),
            rel_cute_lora=rel(out_l, ref_lora),
        )  # fmt: skip
    # Strongest form of the separate stage: its fastest alignment block (rank 0 decides).
    blocks = (16, 32, 64)
    times = []
    for block_m in blocks:
        dist.barrier()
        times.append(time_graph(flashinfer_lora_with(block_m), cold=True, calls=5, samples=3))
    pick = torch.tensor([min(range(3), key=lambda j: times[j]["us"])], device="cuda")
    dist.broadcast(pick, 0)
    row["flashinfer_lora_block_m"] = blocks[int(pick)]
    timed = (("cute", ours), ("cute_lora", ours_lora), ("flashinfer", flashinfer))
    for name, fn in timed + (("flashinfer_lora", flashinfer_lora_with(blocks[int(pick)])),):
        dist.barrier()
        row[name + "_us"] = time_graph(fn, cold=True, calls=5, samples=7)["us"]
    return row


def main(argv=None):
    from ctikernels.moe import MXFP4Experts
    from ctikernels.moe.lora import MoELoRAWeights

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", type=int, default=29541)
    parser.add_argument("--ms", type=int, nargs="+", default=[1, 2048])
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args(argv)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{args.master}:{args.port}", world_size=2, rank=args.rank
    )
    torch.cuda.set_device(0)
    local = EXPERTS // 2
    lo, hi = args.rank * local, (args.rank + 1) * local
    full = weights(EXPERTS)  # the same seed on every rank: identical global weights
    w = [t[lo:hi].contiguous() for t in full]
    expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
    expert_map[lo:hi] = torch.arange(local, dtype=torch.int32, device="cuda")
    lora_full = lora_weights(EXPERTS, args.lora_rank)
    lora = MoELoRAWeights.separate(*(t[lo:hi].contiguous() for t in lora_full.routed))
    cute = MXFP4Experts(
        *[t.clone() for t in w], swiglu_limit=10.0, provider="cute", scales_layout="canonical"
    )
    fi = MXFP4Experts(
        *[t.clone() for t in w], swiglu_limit=10.0, provider="flashinfer", scales_layout="canonical"
    )
    for m in args.ms:
        result = layer_row(m, args.rank, cute, fi, full, lora, lora_full, expert_map)
        if args.rank == 0:
            print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
