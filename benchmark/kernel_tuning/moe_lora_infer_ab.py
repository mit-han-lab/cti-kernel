# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Cross-snapshot MoE inference A/B: the same deterministic cases on two source trees, compared bit
for bit. Run the ``run`` step with PYTHONPATH at each tree's ``python`` (this file may come from
either tree), then ``compare``:

  python moe_lora_infer_ab.py run OUT.pt
  python moe_lora_infer_ab.py compare BASE.pt NEW.pt

Cases: fused_experts and MoEBlock (BF16); fused_experts_lora inference on Triton (and CuTe on
sm_120/121) with per-projection gate/up adapters, distinct and equal A, with and without the
gated shared expert and EP-masked routes. On a tree with the shared-A layout the equal-A cases
also run as shared-A (one A, B = [B_gate; B_up]), keyed ``<case>:shared_a`` and compared against
the base's equal-A case.
"""

import sys

import torch


def rand(*shape, scale=0.05):
    return (torch.randn(shape, device="cuda") * scale).bfloat16()


def run(path):
    from ctikernels.moe import MoEBlock, fused_experts
    from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora

    shared_a = hasattr(MoELoRAWeights, "separate")
    separate = MoELoRAWeights.separate if shared_a else MoELoRAWeights
    cap = torch.cuda.get_device_capability()
    providers = ["triton"] + (["cute"] if cap in ((12, 0), (12, 1)) else [])
    print("shared_a", shared_a, "providers", providers, flush=True)
    results = {}
    e, h, i, si = 64, 1024, 256, 256
    for t in (1, 4, 24, 64, 300, 2048):
        torch.manual_seed(1000 + t)
        x, w13, w2 = rand(t, h, scale=1.0), rand(e, 2 * i, h), rand(e, h, i)
        ids = torch.randint(0, e, (t, 8), device="cuda", dtype=torch.int32)
        rw = torch.rand((t, 8), device="cuda")
        results[f"fused_experts:{t}"] = fused_experts(x, w13, w2, rw, ids)
        if t <= 300:
            block = (rand(e, h), w13, w2, rand(2 * si, h), rand(h, si), rand(1, h))
            results[f"moe_block:{t}"] = MoEBlock()(x, *block).clone()
        masked = torch.where(ids < e // 2, ids, -1)
        for rank in (16, 64):
            torch.manual_seed(2000 + 7 * t + rank)
            a, a_up, b = rand(e, rank, h), rand(e, rank, h), rand(2, e, i, rank)
            a2, b2 = rand(e, rank, i), rand(e, h, rank)
            shared = (rand(2, rank, h), rand(2, si, rank), rand(rank, si), rand(h, rank))
            shared_w = (rand(2 * si, h), rand(h, si), rand(1, h))
            for gated in (False, True):
                extra = shared if gated else ()
                for ep, route in (("ep1", ids), ("ep2", masked)):
                    for provider in providers:
                        kw = {"scaling": 0.5, "provider": provider}
                        if gated:
                            kw["shared_expert"] = shared_w
                        key = f"lora:{t}:{rank}:{int(gated)}:{ep}:{provider}"
                        for name, up in (("separate", a_up), ("equal_a", a)):
                            leaves = (a, up.clone(), b[0].clone(), b[1].clone(), a2, b2)
                            with torch.no_grad():
                                results[f"{key}:{name}"] = fused_experts_lora(
                                    x, w13, w2, rw, route, separate(*leaves, *extra), **kw
                                )
                        if shared_a:
                            with torch.no_grad():
                                results[f"{key}:equal_a:shared_a"] = fused_experts_lora(
                                    x,
                                    w13,
                                    w2,
                                    rw,
                                    route,
                                    MoELoRAWeights(a, b, a2, b2, *extra),
                                    **kw,
                                )
    torch.cuda.synchronize()
    torch.save({k: v.cpu() for k, v in results.items()}, path)
    print("saved", len(results), "cases to", path, flush=True)


def compare(base_path, new_path):
    base, new = torch.load(base_path), torch.load(new_path)
    compared = differences = 0
    for key in sorted(new):
        ref = base.get(key.removesuffix(":shared_a"))
        if ref is None:
            continue
        compared += 1
        if not torch.equal(ref.view(torch.int16), new[key].view(torch.int16)):
            differences += 1
            d = (ref.float() - new[key].float()).abs()
            print("DIFF", key, int((d > 0).sum()), float(d.max()))
    missing = set(base) - {key.removesuffix(":shared_a") for key in new}
    print("compared", compared, "differences", differences, "missing", sorted(missing))
    return differences


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
