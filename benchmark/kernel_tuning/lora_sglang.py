# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Full-projection LoRA parity against SGLang 72730d6 and vLLM 0.31.0 kernels.

No market implementation is vendored. Importing this module does not touch CUDA.
The three-slice qkv_gate case is the requested logical [q+gate,k,v] layout.
The inspected engine uses physical [q,k,v,gate] with shared q/gate A; this
benchmark deliberately gives every logical slice an independent A, as it does
for all six GDN slices (the engine can share A across GDN q/k/v).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.lora import ProjectionCase, _error

# Source: Qwen3.5-{9B,27B} HF text_config and the engine's 35B-A3B
# PUBLISHED_CONFIG fixture. Fields: hidden_size, intermediate_size,
# num_attention_heads * head_dim (q), num_key_value_heads * head_dim (kv),
# linear_num_key_heads * linear_key_head_dim (gdn key),
# linear_num_value_heads * linear_value_head_dim (gdn value),
# linear_num_value_heads (b,a). Engine qwen3_5 / qwen3_5_moe model wiring.
# Qwen3.8-27B's engine config fixture has the same geometry as Qwen3.5-27B.
CASES = {
    "9b": (
        ProjectionCase("qkv_gate", 4096, (8192, 1024, 1024)),
        ProjectionCase("o", 4096, (4096,)),
        ProjectionCase("in_proj", 4096, (2048, 2048, 4096, 4096, 32, 32)),
        ProjectionCase("out_proj", 4096, (4096,)),
        ProjectionCase("gate_up", 4096, (12288, 12288)),
        ProjectionCase("down", 12288, (4096,)),
    ),
    "27b": (
        ProjectionCase("qkv_gate", 5120, (12288, 1024, 1024)),
        ProjectionCase("o", 6144, (5120,)),
        ProjectionCase("in_proj", 5120, (2048, 2048, 6144, 6144, 48, 48)),
        ProjectionCase("out_proj", 6144, (5120,)),
        ProjectionCase("gate_up", 5120, (17408, 17408)),
        ProjectionCase("down", 17408, (5120,)),
    ),
    "35b-a3b": (
        ProjectionCase("qkv_gate", 2048, (8192, 512, 512)),
        ProjectionCase("o", 4096, (2048,)),
        ProjectionCase("in_proj", 2048, (2048, 2048, 4096, 4096, 32, 32)),
        ProjectionCase("out_proj", 4096, (2048,)),
    ),
}
SCALING = 2.0
ALLOCATION_LIMIT = 6 * 1024**3
MARKET_VARIANTS = ("sgl_csgmv", "sgl_triton", "vllm_punica")
VARIANTS = (
    "ours_cute",
    "ours_prod",
    "ours_triton",
    "ours_plan_fbase",
    "ours_torch",
    *MARKET_VARIANTS,
)
COMPARISONS = tuple(
    (ours, market)
    for ours in ("ours_prod", "ours_triton", "ours_cute")
    for market in MARKET_VARIANTS
) + (("ours_torch", "sgl_csgmv"), ("ours_plan_fbase", "sgl_csgmv"))
DENSE = "sglang.kernels.ops.lora.dense"
KERNEL_MODULES = (
    "kernel_utils",
    "lora_tuning_config",
    "chunked_sgmv_shrink",
    "chunked_sgmv_expand",
    "sgemm_lora_a",
    "sgemm_lora_b",
    "qkv_lora_b",
    "gate_up_lora_b",
)


def _positive(raw):
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _integers(raw):
    try:
        return tuple(dict.fromkeys(_positive(v) for v in raw.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--vllm-cache", type=Path, default=Path("/var/tmp/cti-lora-vllm"))
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument(
        "--provider-table",
        action="store_true",
        help="print the production provider chosen for each row",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--model", choices=(*CASES, "all"), default="all")
    parser.add_argument("--ranks", type=_integers, default=(16, 64, 128))
    parser.add_argument("--tokens", type=_integers, default=(1, 2, 3, 4, 8, 16, 32, 64))
    parser.add_argument("--pairs", type=_positive, default=20)
    parser.add_argument("--cases", help="comma-separated case names from the selected models")
    parser.add_argument("--ceilings", type=Path)
    parser.add_argument("--sglang-max-chunk-size", type=int, choices=(16, 32, 64, 128), default=16)
    return parser


def selected_cases(args):
    models = tuple(CASES) if args.model == "all" else (args.model,)
    names = set(args.cases.split(",")) if args.cases else None
    available = {case.name for model in models for case in CASES[model]}
    if names is not None and not names <= available:
        raise ValueError("unknown case selection")
    return [
        (model, case)
        for model in models
        for case in CASES[model]
        if names is None or case.name in names
    ]


@dataclass
class BatchInfo:
    """Only the dense-kernel routing fields; no server/runtime dependencies."""

    bs: int
    num_segments: int
    max_len: int
    seg_lens: torch.Tensor
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    permutation: torch.Tensor
    use_cuda_graph: bool = True
    expected_tokens: int = 0
    req_seg_indptr: torch.Tensor | None = None
    req_weight_indices: torch.Tensor | None = None


class _BoolEnv:
    def __init__(self, name):
        self.name = name

    def get(self):
        return os.environ.get(self.name, "false").lower() in ("true", "1")


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("missing module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_sglang(python_dir):
    """Load only explicit files, never execute SGLang package initializers.

    environ and all dense helpers are tried as real modules. The server utils
    and LoRA utils import distributed/model/transformers code, so only their
    two utility functions and metadata type are shimmed. No registry shim is
    needed by this revision's dense kernels. Use in a dedicated benchmark process.
    """
    root = Path(python_dir) / "sglang"
    if not (root / "kernels/ops/lora/dense").is_dir():
        raise ValueError("sglang-python must contain the sglang package")
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        raise RuntimeError("load_sglang requires a process without existing SGLang imports")
    imports = {}
    # Empty search paths prevent accidental imports of the heavy server stack.
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.lora",
        DENSE,
    ):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    utils = ModuleType("sglang.srt.utils")
    utils.cached_triton_kernel = lambda key: lambda kernel: kernel
    utils.get_device_name = lambda device_id=0: torch.cuda.get_device_name(device_id)
    sys.modules[utils.__name__] = utils
    imports[utils.__name__] = "stub: identity cache wrapper; CUDA device name"
    lora_utils = ModuleType("sglang.srt.lora.utils")
    lora_utils.LoRABatchInfo = BatchInfo
    sys.modules[lora_utils.__name__] = lora_utils
    imports[lora_utils.__name__] = "stub: dense batch metadata"
    try:
        environ = _load_file("sglang.srt.environ", root / "srt/environ.py")
        imports[environ.__name__] = "real"
    except Exception as exc:  # noqa: BLE001 -- optional lightweight dependency
        environ = ModuleType("sglang.srt.environ")
        environ.envs = SimpleNamespace(
            **{
                name: _BoolEnv(name)
                for name in (
                    "SGLANG_ENABLE_LORA_CSGMV_SPLIT_K",
                    "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
                )
            }
        )
        sys.modules[environ.__name__] = environ
        imports[environ.__name__] = "stub after " + type(exc).__name__
    modules, errors = {}, {}
    for leaf in KERNEL_MODULES:
        try:
            modules[leaf] = _load_file(
                f"{DENSE}.{leaf}", root / f"kernels/ops/lora/dense/{leaf}.py"
            )
            imports[f"{DENSE}.{leaf}"] = "real"
        except Exception as exc:  # noqa: BLE001 -- preserve other comparators
            errors[leaf] = type(exc).__name__
            imports[f"{DENSE}.{leaf}"] = "failed: " + type(exc).__name__
    return SimpleNamespace(modules=modules, imports=imports, errors=errors, environ=environ)


def chunk_size(tokens, maximum=16):
    # Backend heuristic with its release default max_lora_chunk_size=16.
    suggested = 128 if tokens >= 256 else 32 if tokens >= 64 else 16
    return max(16, min(maximum, suggested))


def batch_info(tokens, rank, *, chunked, maximum=16, device="cuda", prefill=False):
    size = chunk_size(tokens, maximum) if chunked else tokens
    ends = list(range(size, tokens, size)) + [tokens]
    # Decode graph allocation has B segment slots for csgmv, even when only
    # ceil(B/chunk) are live. Triton has one merged segment for one adapter.
    slots = tokens if chunked else 1
    if chunked and prefill:
        # Match ChunkedSgmvLoRABackend.init_prefill_cuda_graph_batch_info:
        # one long request, at least 16 chunks plus one adapter group.
        slots = max(16, len(ends)) + 1
    indptr = [0, *ends, *([tokens] * (slots - len(ends)))]
    tensor = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
    return BatchInfo(
        bs=tokens if chunked and not prefill else 1,
        num_segments=len(ends),
        max_len=size,
        seg_lens=tensor([0] * slots if chunked else [tokens]),
        seg_indptr=tensor(indptr),
        weight_indices=tensor([0] * slots),
        lora_ranks=tensor([rank]),
        scalings=torch.tensor([SCALING], device=device),
        permutation=tensor(range(tokens)),
        expected_tokens=tokens,
        req_seg_indptr=tensor([0, tokens] if prefill else range(tokens + 1)) if chunked else None,
        req_weight_indices=tensor([0] if prefill else [0] * tokens) if chunked else None,
    )


def kernel_mapping(case):
    count = len(case.output_sizes)
    expand = (
        "sgemm_lora_b_fwd"
        if count == 1
        else (
            "gate_up_lora_b_fwd"
            if count == 2 and len(set(case.output_sizes)) == 1
            else "qkv_lora_b_fwd"
        )
    )
    return {
        "sgl_csgmv": ["chunked_sgmv_lora_shrink_forward", "chunked_sgmv_lora_expand_forward"],
        "sgl_triton": ["sgemm_lora_a_fwd", expand],
        "vllm_punica": [
            "vllm.lora.ops.triton_ops.lora_shrink",
            "vllm.lora.ops.triton_ops.lora_expand",
        ],
    }


def _sglang_call(sgl, variant, case, operands, info, offsets, offsets_cpu):
    x, weight, packed_a, packed_b, _ = operands
    output = F.linear(x, weight)
    count = len(case.output_sizes)
    modules = sgl.modules
    if variant == "sgl_csgmv":
        low = modules["chunked_sgmv_shrink"].chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=packed_a,
            batch_info=info,
            num_slices=count,
        )
        return modules["chunked_sgmv_expand"].chunked_sgmv_lora_expand_forward(
            x=low,
            weights=packed_b,
            batch_info=info,
            slice_offsets=offsets,
            max_slice_size=max(case.output_sizes),
            base_output=output,
            slice_offsets_cpu=offsets_cpu,
        )
    low = modules["sgemm_lora_a"].sgemm_lora_a_fwd(x, packed_a, info, stack_num=count)
    if count == 1:
        return modules["sgemm_lora_b"].sgemm_lora_b_fwd(low, packed_b, info, output)
    if count == 2 and case.output_sizes[0] == case.output_sizes[1]:
        return modules["gate_up_lora_b"].gate_up_lora_b_fwd(
            low,
            packed_b,
            info,
            case.output_sizes[0],
            output,
        )
    return modules["qkv_lora_b"].qkv_lora_b_fwd(
        low,
        packed_b,
        info,
        offsets,
        max(case.output_sizes),
        output,
        n_slices=count,
    )


def _operands(case, rank, tokens, template=None, *, dtype=torch.bfloat16):
    count, n, k = len(case.output_sizes), sum(case.output_sizes), case.input_size
    if template is None:
        tensors = [
            torch.randn(*shape, dtype=dtype, device="cuda") * scale
            for shape, scale in (
                ((tokens, k), 0.25),
                ((n, k), 0.02),
                ((1, count * rank, k), 0.02),
                ((1, n, rank), 0.02),
            )
        ]
    else:
        tensors = [value.clone() for value in template[:4]]
    x, weight, packed_a, packed_b = tensors
    # Contiguous views share storage: no second packed copy per comparator.
    factors = tuple(
        factor
        for a, b in zip(packed_a[0].split(rank), packed_b[0].split(case.output_sizes), strict=True)
        for factor in (a, b)
    )
    return x, weight, packed_a, packed_b, factors


@dataclass
class ProjectionCall:
    call: object
    base: object
    plan: object = None

    def __call__(self):
        return self.call()

    @property
    def base_provider(self):
        return self.plan.base_provider if self.plan is not None else "torch"


def _calls(variant, operands, case, sgl, info, offsets, offsets_cpu, vllm=None):
    from ctikernels.linear._lora_inference import SmallBatchLoRAInferencePlan
    from ctikernels.linear._lora_policy import _SMALL_BATCH_POLICY
    from ctikernels.linear.lora import (
        _forward_projection_inference,
        lora_projection,
        try_prepare_lora_inference_plan,
    )

    # Offline ablations intentionally bypass production admission to measure
    # the frozen kernels. These overrides never enter the production policy.
    class TritonPlan(SmallBatchLoRAInferencePlan):
        def _select_provider(self, num_tokens):
            return "triton"

    class TorchBasePlan(TritonPlan):
        def base(self, x, weight, bias=None):
            output = F.linear(x, weight, bias)
            self._base_provider = "torch"
            return output

    class CutePlan(SmallBatchLoRAInferencePlan):
        @property
        def provider(self):
            return "cute-segmented"

    from ctikernels.linear import lora_delta_factors_

    if variant == "ours_plan":  # Compatibility spelling for one report release.
        variant = "ours_triton"
    calls = []
    for operand in operands:
        x, weight, _, _, factors = operand
        plan = None
        base = partial(F.linear, x, weight)
        if variant == "ours_cute":
            plan = CutePlan(
                factors,
                input_size=case.input_size,
                output_sizes=case.output_sizes,
                scaling=SCALING,
                policy=_SMALL_BATCH_POLICY,
            )
            base = partial(plan.base, x, weight)
            scaling = torch.tensor(SCALING, device=x.device, dtype=torch.float32)

            def call(base=base, x=x, factors=factors, scaling=scaling):
                return lora_delta_factors_(
                    base(), x, factors, scaling=scaling, output_sizes=case.output_sizes
                )
        elif variant in ("ours_triton", "ours_plan_fbase"):
            factory = TorchBasePlan if variant == "ours_plan_fbase" else TritonPlan
            plan = factory(
                factors,
                input_size=case.input_size,
                output_sizes=case.output_sizes,
                scaling=SCALING,
                policy=_SMALL_BATCH_POLICY,
            )
            call = partial(plan.project, x, weight, None)
            base = partial(plan.base, x, weight)
        elif variant == "ours_prod":
            prepared = try_prepare_lora_inference_plan(
                factors,
                input_size=case.input_size,
                output_sizes=case.output_sizes,
                scaling=SCALING,
            )
            call = partial(
                lora_projection,
                x,
                weight,
                None,
                factors,
                scaling=SCALING,
                inference_plan=prepared,
            )
            if prepared is not None and prepared.can_project(
                x, weight, None, factors, scaling=SCALING
            ):
                plan = prepared
                base = partial(plan.base, x, weight)
        elif variant == "ours_torch":
            call = partial(_forward_projection_inference, x, weight, None, SCALING, factors)
        elif variant == "base_only":
            call = base
        elif variant == "vllm_punica":
            from benchmark.kernel_tuning._lora_vllm import PunicaLinear

            if vllm is None:
                raise RuntimeError("vLLM unavailable")
            call = PunicaLinear(vllm, x, weight, factors, SCALING)
            base = call.base
        else:
            call = partial(
                _sglang_call, sgl, variant, case, operand, info[variant], offsets, offsets_cpu
            )
        calls.append(ProjectionCall(call, base, plan))
    return calls


def _capture(calls):
    # Warm on a side stream, as required by PyTorch CUDA graph capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for call in calls:
                call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for call in calls:
            output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _sample(graph, copies):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(10):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / copies)
    return statistics.median(samples)


def _iqr(values):
    quartiles = statistics.quantiles(values, n=4)
    return quartiles[0], quartiles[2]


def compare(graphs, ours, sgl, copies, pairs):
    # Same rotation/reversal, three warm rounds and ten-replay samples as pairs.py.
    order = ((ours, 0), (sgl, 0), (ours, 1), (sgl, 1))
    samples = {key: [] for key in order}
    for _ in range(3):
        for name, index in order:
            _sample(graphs[name][index][0], copies)
    for round_index in range(pairs):
        rotation = round_index % 4
        rotated = order[rotation:] + order[:rotation]
        if round_index % 2:
            rotated = rotated[::-1]
        for key in rotated:
            samples[key].append(_sample(graphs[key[0]][key[1]][0], copies))
    return _pair_result(samples, ours, sgl, pairs), samples


def _pair_result(samples, ours, sgl, pairs):
    ratios = [b / a for a, b in zip(samples[ours, 0], samples[sgl, 0], strict=True)]
    self_pairs = [
        [a / b for a, b in zip(samples[name, 1], samples[name, 0], strict=True)]
        for name in (ours, sgl)
    ]
    eps = max(
        value
        for values in self_pairs
        for value in (
            abs(statistics.median(values) - 1),
            (_iqr(values)[1] - _iqr(values)[0]) / 2,
        )
    )
    ratio = statistics.median(ratios)
    result = {
        "ratio_sgl_ours": ratio,  # Compatibility with older SGLang-only records.
        "ratio_market_ours": ratio,
        "ratio_iqr": _iqr(ratios),
        "eps": eps,
        "pairs": pairs,
        "ours_us": statistics.median(samples[ours, 0] + samples[ours, 1]),
        "sgl_us": statistics.median(samples[sgl, 0] + samples[sgl, 1]),
        "market_us": statistics.median(samples[sgl, 0] + samples[sgl, 1]),
        "self_pairs": {
            name: {"median": statistics.median(v), "iqr": _iqr(v)}
            for name, v in zip((ours, sgl), self_pairs, strict=True)
        },
        "verdict": "faster" if ratio - 1 > eps else "slower" if 1 - ratio > eps else "tie",
    }
    return result


def sample_variants(graphs, copies, pairs):
    """One interleaved experiment, with two independent captures per variant.

    Reuse each measured sample across comparisons. Seven variants cost 14 samples
    per round versus the old three four-sample comparisons (12): +17%, rather
    than multiplying runtime by the number of pairwise verdicts. Rotation and
    reversal balance ordering; each round still contributes one paired ratio.
    """
    order = tuple((name, i) for i in range(2) for name in graphs if name != "base_only")
    samples = {key: [] for key in order}
    for round_index in range(-3, pairs):
        rotation = round_index % len(order) if order else 0
        rotated = order[rotation:] + order[:rotation]
        if round_index % 2:
            rotated = rotated[::-1]
        for key in rotated:
            value = _sample(graphs[key[0]][key[1]][0], copies)
            if round_index >= 0:
                samples[key].append(value)
    return samples


def sol_counts(case, rank, tokens):
    n, k, slices = sum(case.output_sizes), case.input_size, len(case.output_sizes)
    factor_elements = rank * (slices * k + n)
    operand_bytes = 2 * (k * n + factor_elements + tokens * k)
    return operand_bytes, operand_bytes + 2 * tokens * n, 2 * tokens * (k * n + factor_elements)


def emit(prefix, record):
    print(prefix + " " + json.dumps(record, allow_nan=False), flush=True)


def _failure(row, stage, error):
    # Exception messages can contain arbitrary environment values or paths.
    kind = type(error).__name__ if isinstance(error, BaseException) else error
    emit(
        "ERROR",
        {key: row[key] for key in ("model", "case", "rank", "tokens")}
        | {"stage": stage, "error": kind},
    )


def _correctness(output, expected):
    error = _error(output, expected)
    return {
        "ok": bool(error["valid"]),
        "max_abs_err": error["max_absolute"] if math.isfinite(error["max_absolute"]) else None,
        "rel_l2": error["relative_l2"] if math.isfinite(error["relative_l2"]) else None,
    }


def _relative_l2(error, reference):
    numerator = float(torch.linalg.vector_norm(error.double()))
    denominator = float(torch.linalg.vector_norm(reference.double()))
    return numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)


def delta_reference(x, factors, scaling=SCALING):
    return torch.cat(
        [
            scaling * (x.double() @ a.double().T @ b.double().T)
            for a, b in zip(factors[0::2], factors[1::2], strict=True)
        ],
        dim=-1,
    )


def delta_correctness(output, base_same, delta64, base64):
    """Gate LoRA error independently of the selected base GEMM's precision.

    BF16 addition rounds the final output, so subtraction cannot recover an
    exact delta. The 3% delta bound permits this rounding while preventing a
    large base from hiding a missing/incorrect LoRA update; 0.4% bounds output.
    Full-output and base-only FP64 errors are diagnostics, never gates.
    """
    expected = base_same.double() + delta64
    error = output.double() - expected
    full = _relative_l2(error, expected)
    delta = _relative_l2((output.double() - base_same.double()) - delta64, delta64)
    return {
        "ok": math.isfinite(full) and math.isfinite(delta) and full <= 4e-3 and delta <= 3e-2,
        "rel_l2": full if math.isfinite(full) else None,
        "delta_rel_l2": delta if math.isfinite(delta) else None,
        "base_precision": {
            "full_output": _correctness(output, base64 + delta64),
            "base_only": _correctness(base_same, base64),
            "gate": False,
        },
    }


def market_verdicts(row):
    available = [
        name
        for name in MARKET_VARIANTS
        if row["variants"].get(name, {}).get("ok") and "us" in row["variants"][name]
    ]
    faster = min(available, key=lambda name: row["variants"][name]["us"]) if available else None
    row["faster_market"] = faster
    row["market_complete"] = len(available) == len(MARKET_VARIANTS)
    for ours in ("ours_prod", "ours_triton", "ours_cute"):
        row["comparisons"][f"{ours}_vs_faster_market"] = dict(
            row["comparisons"].get(f"{ours}_vs_{faster}", {"verdict": "unavailable"}), market=faster
        )


def run_row(row, case, args, sgl, l2, ceilings, vllm=None):
    from ctikernels.linear._lora_policy import policy_for_hardware, select_provider

    rank, tokens = row["rank"], row["tokens"]
    props = torch.cuda.get_device_properties(0)
    policy = policy_for_hardware(
        device_name=props.name,
        capability=(props.major, props.minor),
        sms=props.multi_processor_count,
    )
    chosen = (
        None
        if policy is None
        else select_provider(
            policy,
            num_slices=len(case.output_sizes),
            rank=rank,
            num_tokens=tokens,
            k=case.input_size,
        )
    )
    operand_bytes, traffic, flops = sol_counts(case, rank, tokens)
    copies = max(2, math.ceil(4 * l2 / operand_bytes))
    mapping = kernel_mapping(case)
    row.update(
        ncopy=copies,
        operand_bytes=operand_bytes,
        rotation_bytes=copies * operand_bytes,
        bytes=traffic,
        flops=flops,
        market_kernels=mapping,
        sgl_kernels=mapping,
        variants={},
        comparisons={},
        policy_would_use_plan=chosen is not None,
        policy_provider=chosen or "torch",
        device_class=None if policy is None else policy.device_class,
        policy_scope="production hardware, availability and provider table",
        chunk_size=chunk_size(tokens, args.sglang_max_chunk_size),
    )
    if copies * operand_bytes > ALLOCATION_LIMIT - 1024**3:
        raise MemoryError("rotation cannot fit allocation budget")
    operands = [_operands(case, rank, tokens, dtype=getattr(torch, args.dtype))]
    operands.extend(_operands(case, rank, tokens, operands[0]) for _ in range(copies - 1))
    x, weight, _, _, factors = operands[0]
    delta64 = delta_reference(x, factors)
    base64 = F.linear(x.double(), weight.double())
    offsets_cpu = torch.tensor([0, *accumulate(case.output_sizes)], dtype=torch.int32)
    offsets = offsets_cpu.to("cuda")
    info = {
        variant: batch_info(
            tokens,
            rank,
            chunked=variant == "sgl_csgmv",
            maximum=args.sglang_max_chunk_size,
            prefill=tokens > 64,
        )
        for variant in ("sgl_csgmv", "sgl_triton")
    }
    graphs, owners = {}, {}
    for variant in (*VARIANTS, "base_only"):
        result = row["variants"][variant] = {"ok": False}
        if variant == "vllm_punica" and args.no_vllm:
            result["disabled"] = True
            continue
        try:
            calls = _calls(variant, operands, case, sgl, info, offsets, offsets_cpu, vllm)
            output = calls[0]()  # Eagerly compile the actual base bucket before capture.
            base_same = calls[0].base()
            check = partial(delta_correctness, base_same=base_same, delta64=delta64, base64=base64)
            if variant == "base_only":
                check = lambda out: {"ok": True, "base_precision": _correctness(out, base64)}
            result.update(check(output))
            result["base_provider"] = calls[0].base_provider
            result["provider"] = (
                calls[0].plan.provider
                if calls[0].plan is not None
                else "torch"
                if variant.startswith("ours_")
                else variant
            )
            if not result["ok"]:
                _failure(row, variant, "correctness")
                continue
            captured = []
            for _ in range(1 if variant == "base_only" else 2):
                captured.append(_capture(calls))
                checked = check(captured[-1][1])
                if not checked["ok"]:
                    result.update(checked)
                    raise ArithmeticError("graph correctness")
            graphs[variant], owners[variant] = captured, calls
        except Exception as exc:  # noqa: BLE001 -- preserve independent variants
            result.update(ok=False, error=type(exc).__name__)
            _failure(row, variant, exc)
    samples = sample_variants(graphs, copies, args.pairs)
    row["timing_protocol"] = {
        "pairs": args.pairs,
        "warm_rounds": 3,
        "replays_per_sample": 10,
        "samples_per_round": len(samples),
        "legacy_samples_per_round": 12,
    }
    for ours, market in COMPARISONS:
        row["comparisons"][f"{ours}_vs_{market}"] = (
            _pair_result(samples, ours, market, args.pairs)
            if (ours, 0) in samples and (market, 0) in samples
            else {"verdict": "unavailable"}
        )
    for variant in graphs:
        values = samples.get((variant, 0), []) + samples.get((variant, 1), [])
        if not values:
            values = [_sample(graphs[variant][0][0], copies) for _ in range(args.pairs)]
        row["variants"][variant]["us"] = statistics.median(values)
    market_verdicts(row)
    base = row["variants"]["base_only"].get("us")
    if base is not None:
        for variant in VARIANTS:
            if "us" in row["variants"][variant]:
                # F.linear subtraction is diagnostic when a variant uses a CuTe base.
                row["variants"][variant]["lora_only_us"] = row["variants"][variant]["us"] - base
    if ceilings:
        row["sol_us"] = max(traffic / ceilings[0], flops / ceilings[1]) * 1e6
        plan_us = row["variants"]["ours_triton"].get("us")
        if plan_us:
            row["sol_frac"] = row["sol_us"] / plan_us
    add_report_aliases(row)


def add_report_aliases(row):
    """One release of report compatibility, without timing the variant twice."""
    row["variants"]["ours_plan"] = row["variants"]["ours_triton"]
    for name, result in tuple(row["comparisons"].items()):
        if name.startswith("ours_triton_vs_"):
            row["comparisons"][name.replace("ours_triton_vs_", "ours_plan_vs_", 1)] = result


def emit_provider_table(row):
    emit(
        "PROVIDER",
        {
            key: row.get(key)
            for key in (
                "model",
                "case",
                "device_class",
                "rank",
                "tokens",
                "K",
                "output_sizes",
                "policy_provider",
            )
        },
    )


def config_files(sgl, cases, ranks):
    """Report the actual real loader's selected file, using its open boundary."""
    module = sgl.modules.get("lora_tuning_config")
    if module is None:
        return {"error": "config module unavailable"}
    selected = {}
    opened = []

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path).parent.name + "/" + Path(path).name)
        return open(path, *args, **kwargs)

    module.open = tracked_open
    # Avoid real module logs exposing runtime config-directory paths.
    logging.getLogger(module.__name__).setLevel(logging.ERROR)
    try:
        for _, case in cases:
            for rank in ranks:
                for kernel, width in (
                    ("shrink", case.input_size),
                    ("expand", sum(case.output_sizes)),
                ):
                    opened.clear()
                    key = f"{kernel}:K={width}:R={rank}:S={len(case.output_sizes)}"
                    if key not in selected:
                        try:
                            module.get_lora_configs(kernel, width, rank, len(case.output_sizes))
                            selected[key] = opened[-1] if opened else "default"
                        except Exception as exc:  # noqa: BLE001 -- let affected rows report failure
                            selected[key] = "error: " + type(exc).__name__
    finally:
        del module.open
    return selected


def x925_cpus(cpuinfo):
    cpus = set()
    for block in cpuinfo.strip().split("\n\n"):
        fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if fields.get("CPU part", "").lower() == "0xd85" and "processor" in fields:
            cpus.add(int(fields["processor"]))
    return cpus


def pin_host():
    """Set affinity before CUDA initialization, then restart so threads inherit it."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    current = os.sched_getaffinity(0)
    fast = x925_cpus(Path("/proc/cpuinfo").read_text())
    if fast and not current <= fast:
        allowed = current & fast
        if not allowed:
            raise RuntimeError("Cortex-X925 cores excluded by current affinity")
        os.sched_setaffinity(0, allowed)
        os.execv(
            sys.executable,
            [sys.executable, "-m", "benchmark.kernel_tuning.lora_sglang", *sys.argv[1:]],
        )
    return sorted(os.sched_getaffinity(0)) if fast else []


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pairs < 2:
        parser.error("--pairs must be at least 2 for IQR")
    try:
        cases = selected_cases(args)
    except ValueError:
        parser.error("--cases contains a name absent from the selected models")
    if max(args.tokens) > 8192:
        parser.error("--tokens must be at most 8192")
    pinned = pin_host()
    from benchmark.kernel_tuning._prologue_market import quiet_external

    vllm, vllm_info = None, {"disabled": args.no_vllm}
    if not args.no_vllm:
        from benchmark.kernel_tuning._lora_vllm import load_vllm

        try:
            with quiet_external():
                vllm = load_vllm(args.vllm_cache)
            vllm_info = vllm.info
        except Exception as exc:  # noqa: BLE001 -- retain SGLang comparison
            vllm_info = {"version": "0.31.0", "error": type(exc).__name__}
    sgl = load_sglang(args.sglang_python)
    import triton

    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0)
    if l2 <= 0:
        raise RuntimeError("CUDA device must report a positive L2_cache_size")
    torch.cuda.set_per_process_memory_fraction(min(1.0, ALLOCATION_LIMIT / props.total_memory))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    ceilings = None
    if args.ceilings:
        document = json.loads(args.ceilings.read_text())
        ceilings = (
            float(document["memory"]["dram_read_gbs"]) * 1e9,
            float(document["tensor_core"]["bf16_m16n8k16"]["tflops_gpu"]) * 1e12,
        )
        if not all(math.isfinite(v) and v > 0 for v in ceilings):
            parser.error("ceilings must be finite positive rates")
    emit(
        "HOST",
        {
            "pinned_cpus": pinned,
            "vllm": vllm_info,
            "gpu": props.name,
            "cc": torch.cuda.get_device_capability(),
            "sm": props.multi_processor_count,
            "l2_bytes": l2,
            "versions": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                "sglang_source": "72730d6 (runtime supplied)",
            },
            "imports": sgl.imports,
            "csgmv_configs": config_files(sgl, cases, args.ranks),
            "sglang_max_chunk_size": args.sglang_max_chunk_size,
            "csgmv_split_k": sgl.environ.envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.get(),
            "deterministic": sgl.environ.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get(),
            "allocation_limit_bytes": ALLOCATION_LIMIT,
            "qkv_gate_layout": "logical q+gate,k,v; engine physical q,k,v,gate",
        },
    )
    with torch.inference_mode():
        for model, case in cases:
            for rank in args.ranks:
                for tokens in args.tokens:
                    row = {
                        "model": model,
                        "case": case.name,
                        "rank": rank,
                        "tokens": tokens,
                        "K": case.input_size,
                        "output_sizes": case.output_sizes,
                        "N": sum(case.output_sizes),
                        "dtype": args.dtype,
                        "scaling": SCALING,
                        "tp": 1,
                    }
                    try:
                        run_row(row, case, args, sgl, l2, ceilings, vllm)
                    except Exception as exc:  # noqa: BLE001 -- continue the survey after row failures
                        row["error"] = type(exc).__name__
                        _failure(row, "row", exc)
                    if args.provider_table:
                        emit_provider_table(row)
                    emit("ROW", row)
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception as exc:  # noqa: BLE001 -- device failure must not stop reporting
                        _failure(row, "cleanup", exc)


if __name__ == "__main__":
    main()
