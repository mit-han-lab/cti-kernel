# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Profile the LoRA inference specialization on a target GPU.

This tool produces advisory, content-addressed evidence. It never changes release
selection policy; maintainers promote a result only after reviewing the operation and
whole-model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark._support import (
    REPO_ROOT,
    distribution,
    git_source_state,
    sha256_file,
    write_receipt,
)

_DEFAULT_CASES = (
    "qwen3-0.6b-qkv=1024:2048,1024,1024",
    "qwen3-0.6b-o=2048:1024",
    "qwen3-0.6b-gate-up=1024:3072,3072",
    "qwen3-0.6b-down=3072:1024",
    "qwen3-4b-qkv=2560:4096,1024,1024",
    "qwen3-4b-o=4096:2560",
    "qwen3-4b-gate-up=2560:9728,9728",
    "qwen3-4b-down=9728:2560",
)
_DEFAULT_RANKS = (16, 64, 128)
_DEFAULT_TOKEN_COUNTS = (1, 2, 3, 4, 8, 16, 32)
_DEFAULT_EXECUTION_MODES = ("cuda_graph", "eager")
_SCALING = 2.0
_ATOL = 1e-4
_RTOL = 4e-3
_RELATIVE_L2_LIMIT = 4e-3


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    input_size: int
    output_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("projection case name must not be empty")
        if self.input_size <= 0 or not self.output_sizes or min(self.output_sizes) <= 0:
            raise ValueError("projection dimensions must be positive")

    @classmethod
    def parse(cls, raw: str) -> ProjectionCase:
        """Parse ``NAME=INPUT_SIZE:OUTPUT_SIZE[,OUTPUT_SIZE...]``."""

        name, separator, shape = raw.partition("=")
        input_size, shape_separator, output_sizes = shape.partition(":")
        if not separator or not shape_separator:
            raise argparse.ArgumentTypeError(
                f"invalid case {raw!r}; expected NAME=INPUT_SIZE:OUTPUT_SIZE[,OUTPUT_SIZE...]"
            )
        try:
            return cls(
                name=name,
                input_size=int(input_size),
                output_sizes=tuple(int(value) for value in output_sizes.split(",")),
            )
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid projection case {raw!r}: {error}") from error


def classify_result(
    direct: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
    *,
    min_speedup_fraction: float,
    max_cv: float,
) -> str:
    """Classify measured evidence without changing release policy."""

    if (
        not math.isfinite(min_speedup_fraction)
        or not math.isfinite(max_cv)
        or not 0 <= min_speedup_fraction < 1
        or not 0 < max_cv < 1
    ):
        raise ValueError(
            "classification thresholds require min_speedup_fraction in [0, 1) and max_cv in (0, 1)"
        )
    direct_p50 = float(direct["p50"])
    candidate_p50 = float(candidate["p50"])
    direct_cv = float(direct["cv"])
    candidate_cv = float(candidate["cv"])
    if (
        not all(math.isfinite(value) for value in (direct_p50, candidate_p50))
        or direct_p50 <= 0
        or candidate_p50 <= 0
    ):
        raise ValueError("latency measurements must be finite and positive")
    if (
        not all(math.isfinite(value) for value in (direct_cv, candidate_cv))
        or direct_cv < 0
        or candidate_cv < 0
    ):
        raise ValueError("latency variation must be finite and non-negative")
    if max(direct_cv, candidate_cv) > max_cv:
        return "noisy"
    delta = candidate_p50 / direct_p50 - 1.0
    if delta <= -min_speedup_fraction:
        return "candidate_preferred"
    if delta >= min_speedup_fraction:
        return "direct_preferred"
    return "inconclusive"


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _seed(case: ProjectionCase, rank: int, num_tokens: int = 0) -> int:
    payload = f"{case.name}:{case.input_size}:{case.output_sizes}:{rank}:{num_tokens}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _make_factors(case: ProjectionCase, rank: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor
        for output_size in case.output_sizes
        for tensor in (
            torch.randn(rank, case.input_size, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(output_size, rank, device="cuda", dtype=torch.bfloat16) * 0.02,
        )
    )


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    factors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    x_fp32 = x.float()
    output = F.linear(x_fp32, weight.float())
    deltas = tuple(
        F.linear(F.linear(x_fp32, factors[index].float()), factors[index + 1].float()) * _SCALING
        for index in range(0, len(factors), 2)
    )
    return output + torch.cat(deltas, dim=-1)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = actual.double() - expected.double()
    reference = expected.double()
    error_sum = float(difference.square().sum())
    reference_sum = float(reference.square().sum())
    rms = math.sqrt(error_sum / difference.numel())
    reference_rms = math.sqrt(reference_sum / reference.numel())
    relative_l2 = (
        math.sqrt(error_sum / reference_sum)
        if reference_sum
        else (0.0 if not error_sum else math.inf)
    )
    allowed_rms = _ATOL + _RTOL * reference_rms
    return {
        "allowed_rms": allowed_rms,
        "max_absolute": float(difference.abs().max()),
        "relative_l2": relative_l2,
        "rms": rms,
        "valid": rms <= allowed_rms and relative_l2 <= _RELATIVE_L2_LIMIT,
    }


def _peak_allocated_bytes(operation: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return peak - baseline


def _capture_repeated(
    operation: Callable[[], torch.Tensor], repetitions: int
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            output = operation()
    return graph, output


def _timed_graph_pair(
    direct: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    warmups: int,
    measurements: int,
    repetitions: int,
) -> tuple[list[float], list[float], tuple[object, ...]]:
    direct_graph, direct_output = _capture_repeated(direct, repetitions)
    candidate_graph, candidate_output = _capture_repeated(candidate, repetitions)
    for _ in range(warmups):
        direct_graph.replay()
        candidate_graph.replay()
    torch.cuda.synchronize()

    recorded: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for index in range(measurements):
        ordered = (("direct", direct_graph), ("candidate", candidate_graph))
        if index % 2:
            ordered = tuple(reversed(ordered))
        for name, graph in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            recorded.append((name, start, end))
    torch.cuda.synchronize()

    direct_samples = []
    candidate_samples = []
    for name, start, end in recorded:
        samples = direct_samples if name == "direct" else candidate_samples
        samples.append(start.elapsed_time(end) / repetitions)
    owners = (direct_graph, direct_output, candidate_graph, candidate_output)
    return direct_samples, candidate_samples, owners


def _timed_eager_pair(
    direct: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    warmups: int,
    measurements: int,
    repetitions: int,
) -> tuple[list[float], list[float], tuple[object, ...]]:
    outputs: list[torch.Tensor] = []
    for _ in range(warmups):
        outputs[:] = (direct(), candidate())
    torch.cuda.synchronize()

    recorded: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    operations = {"direct": direct, "candidate": candidate}
    for index in range(measurements):
        ordered = ("direct", "candidate") if index % 2 == 0 else ("candidate", "direct")
        for name in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                outputs[:] = (operations[name](),)
            end.record()
            recorded.append((name, start, end))
    torch.cuda.synchronize()

    direct_samples = []
    candidate_samples = []
    for name, start, end in recorded:
        samples = direct_samples if name == "direct" else candidate_samples
        samples.append(start.elapsed_time(end) / repetitions)
    return direct_samples, candidate_samples, tuple(outputs)


def _stabilize_gpu(seconds: float) -> None:
    if seconds == 0:
        return
    left = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    output = None
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        output = left @ right
        torch.cuda.synchronize()
    del left, right, output
    torch.cuda.empty_cache()


def _fingerprint(
    *,
    source_paths: Sequence[Path],
) -> dict[str, object]:
    import triton

    revision, dirty = git_source_state()
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(source_paths)
    }
    return {
        "device": {
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "index": device_index,
            "multi_processor_count": properties.multi_processor_count,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        },
        "software": {
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "source": {
            "dirty": dirty,
            "files": files,
            "revision": revision,
        },
    }


@torch.inference_mode()
def run_profile(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA kernel profiling requires a CUDA device")

    from benchmark import _support
    from ctikernels.linear import _lora_inference
    from ctikernels.linear import lora as lora_operation
    from ctikernels.linear._lora_inference import SmallBatchLoRAInferencePlan
    from ctikernels.linear.lora import lora_projection

    cases = tuple(args.cases or map(ProjectionCase.parse, _DEFAULT_CASES))
    ranks = tuple(args.ranks or _DEFAULT_RANKS)
    token_counts = tuple(args.token_counts or _DEFAULT_TOKEN_COUNTS)
    execution_modes = tuple(args.execution_modes or _DEFAULT_EXECUTION_MODES)
    _stabilize_gpu(args.stabilize_seconds)

    results = []
    started = time.time()
    for case in cases:
        for rank in ranks:
            torch.manual_seed(_seed(case, rank))
            weight = (
                torch.randn(
                    sum(case.output_sizes),
                    case.input_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.02
            )
            factors = _make_factors(case, rank)
            plan = SmallBatchLoRAInferencePlan(
                factors,
                input_size=case.input_size,
                output_sizes=case.output_sizes,
                scaling=_SCALING,
            )
            for num_tokens in token_counts:
                torch.manual_seed(_seed(case, rank, num_tokens))
                x = (
                    torch.randn(
                        num_tokens,
                        case.input_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.25
                )

                direct = partial(
                    lora_projection,
                    x,
                    weight,
                    None,
                    factors,
                    scaling=_SCALING,
                )
                candidate = partial(plan.project, x, weight, None)

                expected = _reference(x, weight, factors)
                direct_error = _error(direct(), expected)
                candidate_error = _error(candidate(), expected)
                if not direct_error["valid"] or not candidate_error["valid"]:
                    raise AssertionError(
                        f"correctness failure for {case.name}, rank={rank}, "
                        f"num_tokens={num_tokens}: direct={direct_error}, "
                        f"candidate={candidate_error}"
                    )
                direct_peak = _peak_allocated_bytes(direct)
                candidate_peak = _peak_allocated_bytes(candidate)

                for execution_mode in execution_modes:
                    timer = (
                        _timed_graph_pair if execution_mode == "cuda_graph" else _timed_eager_pair
                    )
                    direct_samples, candidate_samples, owners = timer(
                        direct,
                        candidate,
                        warmups=args.warmups,
                        measurements=args.measurements,
                        repetitions=args.repetitions,
                    )
                    if not (len(direct_samples) == len(candidate_samples) == args.measurements):
                        raise RuntimeError(
                            "kernel profiler produced an unexpected number of samples: "
                            f"direct={len(direct_samples)}, "
                            f"candidate={len(candidate_samples)}, "
                            f"expected={args.measurements}"
                        )
                    direct_timing = distribution(direct_samples)
                    candidate_timing = distribution(candidate_samples)
                    delta = float(candidate_timing["p50"]) / float(direct_timing["p50"]) - 1
                    advice = classify_result(
                        direct_timing,
                        candidate_timing,
                        min_speedup_fraction=args.min_speedup_fraction,
                        max_cv=args.max_cv,
                    )
                    results.append(
                        {
                            "advice": advice,
                            "candidate": {
                                "correctness": candidate_error,
                                "peak_allocated_bytes": candidate_peak,
                                "timing_ms": candidate_timing,
                            },
                            "case": asdict(case),
                            "direct": {
                                "correctness": direct_error,
                                "peak_allocated_bytes": direct_peak,
                                "timing_ms": direct_timing,
                            },
                            "execution_mode": execution_mode,
                            "p50_delta_fraction": delta,
                            "rank": rank,
                            "token_count": num_tokens,
                        }
                    )
                    print(
                        case.name,
                        f"rank={rank}",
                        f"tokens={num_tokens}",
                        execution_mode,
                        f"direct={float(direct_timing['p50']):.6f}ms",
                        f"candidate={float(candidate_timing['p50']):.6f}ms",
                        f"delta={delta:+.2%}",
                        advice,
                        flush=True,
                    )
                    del owners
                del candidate, direct, expected, x
            del plan, factors, weight
            torch.cuda.empty_cache()

    advice_counts = Counter(str(result["advice"]) for result in results)
    receipt = {
        "config": {
            "atol": _ATOL,
            "cases": [asdict(case) for case in cases],
            "execution_modes": execution_modes,
            "max_cv": args.max_cv,
            "measurements": args.measurements,
            "min_speedup_fraction": args.min_speedup_fraction,
            "ranks": ranks,
            "relative_l2_limit": _RELATIVE_L2_LIMIT,
            "repetitions": args.repetitions,
            "rtol": _RTOL,
            "scaling": _SCALING,
            "stabilize_seconds": args.stabilize_seconds,
            "token_counts": token_counts,
            "warmups": args.warmups,
        },
        "duration_seconds": time.time() - started,
        "fingerprint": _fingerprint(
            source_paths=(
                Path(_support.__file__).resolve(),
                Path(_lora_inference.__file__).resolve(),
                Path(lora_operation.__file__).resolve(),
                Path(__file__).resolve(),
            ),
        ),
        "kind": "operation_local_kernel_profile",
        "operation": "lora_projection",
        "results": results,
        "schema_version": 1,
        "status": "advisory",
        "summary": {
            "advice_counts": dict(sorted(advice_counts.items())),
            "num_results": len(results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_receipt(args.output, receipt)
    return args.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", dest="cases", action="append", type=ProjectionCase.parse)
    parser.add_argument("--rank", dest="ranks", action="append", type=_positive_int)
    parser.add_argument("--token-count", dest="token_counts", action="append", type=_positive_int)
    parser.add_argument(
        "--execution-mode",
        dest="execution_modes",
        action="append",
        choices=("cuda_graph", "eager"),
    )
    parser.add_argument("--warmups", type=_positive_int, default=10)
    parser.add_argument("--measurements", type=_positive_int, default=30)
    parser.add_argument("--repetitions", type=_positive_int, default=20)
    parser.add_argument("--stabilize-seconds", type=float, default=1.0)
    parser.add_argument("--min-speedup-fraction", type=float, default=0.03)
    parser.add_argument("--max-cv", type=float, default=0.03)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not math.isfinite(args.stabilize_seconds) or args.stabilize_seconds < 0:
        parser.error("stabilize-seconds must be finite and non-negative")
    try:
        classify_result(
            {"p50": 1.0, "cv": 0.0},
            {"p50": 1.0, "cv": 0.0},
            min_speedup_fraction=args.min_speedup_fraction,
            max_cv=args.max_cv,
        )
    except ValueError as error:
        parser.error(str(error))
    output = run_profile(args)
    print(f"receipt: {output}", flush=True)
    device = torch.cuda.current_device()
    print(
        "reviewed-policy key: "
        f"device_name={torch.cuda.get_device_name(device)!r}, "
        f"capability={torch.cuda.get_device_capability(device)!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
