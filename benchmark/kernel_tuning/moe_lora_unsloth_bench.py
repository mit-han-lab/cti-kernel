# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired MoE LoRA training-step timings: ours against Unsloth's MoE LoRA path.

One row is one projection group's forward plus backward to x, the routing weights
and every LoRA factor, with frozen BF16 base weights:

* routed: per-expert LoRA on the experts' gate_up and down weights (top-8);
* shared: the shared expert's dense MLP with LoRA on gate, up and down.

Unsloth is imported, never bundled. Install it in a separate environment, for
example ``unsloth==2026.10.6 unsloth_zoo==2026.10.6 transformers==5.17.0
peft==0.21.2`` (``import unsloth`` must precede transformers and PEFT). The
Unsloth side is the transformers expert / MLP module wrapped by PEFT the way
Unsloth's ``get_peft_model`` wraps MoE models (``target_parameters`` on the
expert weights, ``target_modules`` on the shared MLP) with Unsloth's patched
forwards. Every Unsloth candidate below is timed and the fastest is the bar:

* routed: ``UNSLOTH_MOE_BACKEND`` ``grouped_mm`` (Unsloth's default: its Triton grouped
  GEMMs up to measured rows-per-expert limits on sm_12x, ``torch._grouped_mm`` above),
  the same with ``UNSLOTH_MOE_GROUPED_TRITON=1`` (Triton at every size) and
  ``unsloth_triton``; ``--backends`` adds ``native_torch`` (a per-expert loop);
* shared: PEFT's LoRA MLP and ``unsloth.kernels.fast_lora.apply_lora_mlp_swiglu``;
* adapter precision: BF16 adapters, and FP32 adapters under BF16 autocast
  (Unsloth's default ``float32_mixed_precision`` for BF16 models).

Ours and every candidate run as CUDA graphs when they capture, otherwise with
events; runs interleave in a shuffled order per sample. Each JSON row has all
medians, the fastest candidate and the median paired Unsloth/ours ratio.
"""

import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
import statistics
import sys

import torch

MODELS = {"qwen3-30b-a3b": (128, 2048, 768, False), "qwen35-35b-a3b": (256, 2048, 512, True)}
# grouped_mm_triton: grouped_mm with UNSLOTH_MOE_GROUPED_TRITON=1 (Triton grouped GEMMs for
# every size instead of the measured per-architecture row limits).
BACKENDS = ("grouped_mm", "grouped_mm_triton", "unsloth_triton", "native_torch")
SHARED_PATHS = ("peft", "fast_lora")
PRECISIONS = ("bf16", "fp32_autocast")


def int_list(text):
    return tuple(int(v) for v in text.split(","))


def name_list(choices):
    def parse(text):
        values = tuple(v for v in text.split(",") if v)
        if not values or not set(values) <= set(choices):
            raise argparse.ArgumentTypeError(f"choose from {','.join(choices)}")
        return values

    return parse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen35-35b-a3b")
    parser.add_argument("--tokens", type=int_list, default=(1, 64, 512, 2048, 8192))
    parser.add_argument("--ranks", type=int_list, default=(16, 32, 64))
    parser.add_argument("--kinds", type=name_list(("routed", "shared")))
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--backends", type=name_list(BACKENDS), default=BACKENDS[:3])
    parser.add_argument("--shared-paths", type=name_list(SHARED_PATHS), default=SHARED_PATHS)
    parser.add_argument("--precisions", type=name_list(PRECISIONS), default=PRECISIONS)
    parser.add_argument("--graphs", choices=("auto", "off"), default="auto")
    args = parser.parse_args(argv)
    shared = MODELS[args.model][3]
    args.kinds = args.kinds or (("routed", "shared") if shared else ("routed",))
    if "shared" in args.kinds and not shared:
        parser.error("qwen3-30b-a3b has no shared expert")
    return args


def select_backend(name):
    """Point Unsloth's cached MoE backend choice at ``name`` and return its pick."""
    if name == "grouped_mm_triton":
        name = "grouped_mm"
        os.environ["UNSLOTH_MOE_GROUPED_TRITON"] = "1"
    else:
        os.environ.pop("UNSLOTH_MOE_GROUPED_TRITON", None)
    os.environ["UNSLOTH_MOE_BACKEND"] = name
    selected = None
    for module_name, module in list(sys.modules.items()):
        if "unsloth" not in module_name or module is None:
            continue
        select = module.__dict__.get("select_moe_backend")
        if select is not None and hasattr(select, "cache_clear"):
            select.cache_clear()
            selected = select()
    return selected


def unsloth_modules(model):
    if model == "qwen35-35b-a3b":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig as Config,
        )

        return Config, modeling.Qwen3_5MoeExperts, modeling.Qwen3_5MoeMLP
    from transformers.models.qwen3_moe import modeling_qwen3_moe as modeling
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as Config

    return Config, modeling.Qwen3MoeExperts, modeling.Qwen3MoeMLP


class Holder(torch.nn.Module):
    def __init__(self, name, module):
        super().__init__()
        setattr(self, name, module)


def wrap_peft(module, name, rank, precision, *, experts):
    from peft import LoraConfig, get_peft_model

    if experts:
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=[],
            target_parameters=[f"{name}.gate_up_proj", f"{name}.down_proj"],
        )
    else:
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
    model = get_peft_model(Holder(name, module), config)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    trainable = []
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in parameter_name:
                raise RuntimeError(f"unexpected trainable parameter {parameter_name}")
            with torch.no_grad():
                parameter.data = (torch.randn_like(parameter, dtype=torch.float32) * 0.02).to(dtype)
            trainable.append(parameter)
    return model, trainable


def run_row(args, kind, tokens, rank, Config, Experts, Mlp):
    from benchmark.kernel_tuning.moe_lora_backward_bench import make_runner, paired_times
    from ctikernels.moe.lora import MoELoRAWeights, fused_experts_lora, shared_expert_lora

    torch.manual_seed(71)
    e, h, i, _ = MODELS[args.model]
    config = Config(
        hidden_size=h,
        moe_intermediate_size=i,
        shared_expert_intermediate_size=i,
        num_experts=e,
        num_experts_per_tok=8,
        num_hidden_layers=1,
        hidden_act="silu",
    )

    def rand(*shape, scale=0.02):
        return (torch.randn(shape, device="cuda") * scale).bfloat16()

    x = rand(tokens, h, scale=0.3)
    dy = rand(tokens, h, scale=0.3)
    scale = torch.tensor([1.0], device="cuda")
    functions, checks, keep, restore = {}, {}, [], {}
    if kind == "routed":
        w13 = torch.nn.Parameter(rand(e, 2 * i, h, scale=h**-0.5), requires_grad=False)
        w2 = torch.nn.Parameter(rand(e, h, i, scale=i**-0.5), requires_grad=False)
        ids = torch.randint(e, (tokens, 8), device="cuda")
        rw = torch.rand((tokens, 8), device="cuda")
        # Shared-A gate_up (PEFT gate_up_proj), the trainable layout.
        factors = [
            rand(*shape).requires_grad_()
            for shape in ((e, rank, h), (2, e, i, rank), (e, rank, i), (e, h, rank))
        ]
        xo, rwo = x.clone().requires_grad_(), rw.clone().requires_grad_()
        ids32 = ids.to(torch.int32)

        def ours():
            y = fused_experts_lora(
                xo,
                w13,
                w2,
                rwo,
                ids32,
                MoELoRAWeights(*factors),
                scaling=scale,
            )
            return torch.autograd.grad(y, (xo, rwo, *factors), dy)

        functions["ours"] = ours
        for precision in args.precisions:
            module = Experts(config).cuda().bfloat16()
            module.gate_up_proj, module.down_proj = w13, w2
            model, trainable = wrap_peft(module, "experts", rank, precision, experts=True)
            keep.append(model)
            experts = model.base_model.model.experts
            xu = x.clone().requires_grad_()
            rwu = rw.bfloat16().requires_grad_()
            autocast = precision != "bf16"
            for backend in args.backends:

                def unsloth(
                    experts=experts, xu=xu, rwu=rwu, trainable=trainable, b=backend, a=autocast
                ):
                    select_backend(b)
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if a
                        else contextlib.nullcontext()
                    )
                    with context:
                        y = experts(xu, ids, rwu)
                    return torch.autograd.grad(y, (xu, rwu, *trainable), dy.to(y.dtype))

                label = f"unsloth:{backend}:{precision}"
                functions[label] = unsloth
                checks[f"selected:{backend}"] = select_backend(backend)
            with torch.no_grad():
                select_backend(args.backends[0])
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if autocast
                    else contextlib.nullcontext()
                )
                with context:
                    on = experts(x, ids, rw.bfloat16()).float()
                    with model.disable_adapter():
                        off = experts(x, ids, rw.bfloat16()).float()
            # The adapters are random and nonzero, so the LoRA branch must move the output.
            checks[f"lora_delta_rel:{precision}"] = float((on - off).norm() / off.norm())
    else:
        xo = x.clone().requires_grad_()
        # Every candidate and ours read one copy of the base weights: separate copies
        # would let candidates that share theirs hit L2 after each other (GB10 holds the
        # whole T=1 working set in L2), which biases interleaved timings.
        w13 = rand(2 * i, h, scale=h**-0.5)
        w2 = rand(h, i, scale=i**-0.5)
        for precision in args.precisions:
            mlp = Mlp(config, intermediate_size=i).cuda().bfloat16()
            for layer, weight in (
                (mlp.gate_proj, w13[:i]),
                (mlp.up_proj, w13[i:]),
                (mlp.down_proj, w2),
            ):
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            model, trainable = wrap_peft(mlp, "mlp", rank, precision, experts=False)
            keep.append(model)
            peft_mlp = model.base_model.model.mlp
            autocast = precision != "bf16"
            paths = {"peft": lambda m, v: m(v)}
            if "fast_lora" in args.shared_paths:
                from unsloth.kernels.fast_lora import apply_lora_mlp_swiglu

                # The fused LoRA MLP takes [batch, sequence, hidden].
                paths["fast_lora"] = lambda m, v: apply_lora_mlp_swiglu(m, v[None])[0]
            for path in args.shared_paths:
                # fast_lora's backward is in place by default (dX overwrites its input), so
                # every candidate gets its own input, restored from x before each warmup,
                # capture and timed run (untimed), as ours reads an untouched x.
                xu = x.clone().requires_grad_()

                def unsloth(m=peft_mlp, xu=xu, trainable=trainable, f=paths[path], a=autocast):
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if a
                        else contextlib.nullcontext()
                    )
                    with context:
                        y = f(m, xu)
                    return torch.autograd.grad(y, (xu, *trainable), dy.to(y.dtype))

                label = f"unsloth:{path}:{precision}"
                functions[label] = unsloth
                restore[label] = lambda xu=xu: xu.detach().copy_(x)
            if precision == args.precisions[0]:
                gate, up, down = peft_mlp.gate_proj, peft_mlp.up_proj, peft_mlp.down_proj
                a_of = lambda layer: layer.lora_A["default"].weight.detach().bfloat16()
                b_of = lambda layer: layer.lora_B["default"].weight.detach().bfloat16()
                factors = [
                    torch.stack([a_of(gate), a_of(up)]).requires_grad_(),
                    torch.stack([b_of(gate), b_of(up)]).requires_grad_(),
                    a_of(down).clone().requires_grad_(),
                    b_of(down).clone().requires_grad_(),
                ]

                def ours(factors=factors):
                    y = shared_expert_lora(xo, w13, w2, tuple(factors), scale, provider="cute")
                    return torch.autograd.grad(y, (xo, *factors), dy)

                functions["ours"] = ours
                with torch.no_grad():
                    reference = shared_expert_lora(
                        x, w13, w2, tuple(factors), scale, provider="cute"
                    )
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if autocast
                        else contextlib.nullcontext()
                    )
                    with context:
                        theirs = peft_mlp(x)
                # Same adapters on both sides: the forwards must agree.
                checks["forward_rel_vs_peft"] = float(
                    (theirs.float() - reference.float()).norm() / reference.float().norm()
                )
    runners, modes, failed = {}, {}, {}
    for name, fn in functions.items():
        try:
            runners[name], modes[name] = make_runner(
                fn, args.graphs == "auto", before=restore.get(name)
            )
        except (RuntimeError, TypeError, ValueError) as error:  # recorded, not fatal
            failed[name] = f"{type(error).__name__}: {str(error)[:300]}"
        torch.cuda.synchronize()
    measurements = paired_times(runners, args.samples, before=restore)
    times = {name: statistics.median(values) for name, values in measurements.items()}
    candidates = {name: t for name, t in times.items() if name.startswith("unsloth:")}
    best = min(candidates, key=candidates.get) if candidates else None
    row = {
        "model": args.model,
        "kind": kind,
        "T": tokens,
        "rank": rank,
        "ep": 1,
        "samples": args.samples,
        "time_ms": times,
        "timing": modes,
        "failed": failed,
        "checks": checks,
        "unsloth_best": best,
    }
    if best is not None:
        ratios = [u / o for u, o in zip(measurements[best], measurements["ours"], strict=True)]
        row["unsloth_over_ours"] = statistics.median(ratios)
    return row


def main(argv=None):
    args = parse_args(argv)
    import unsloth  # noqa: F401  # applies Unsloth's patches before transformers/PEFT use

    versions = {
        name: importlib.metadata.version(name)
        for name in ("unsloth", "unsloth_zoo", "transformers", "peft", "torch", "triton")
    }
    versions["device"] = torch.cuda.get_device_name()
    print("VERSIONS " + json.dumps(versions), flush=True)
    Config, Experts, Mlp = unsloth_modules(args.model)
    torch.backends.cuda.matmul.allow_tf32 = False
    for kind in args.kinds:
        for tokens in args.tokens:
            for rank in args.ranks:
                gc.collect()
                torch.cuda.empty_cache()
                row = run_row(args, kind, tokens, rank, Config, Experts, Mlp)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
