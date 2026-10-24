# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import argparse
import json
import statistics

import torch
from flashinfer.sampling import softmax


def _measure(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    enable_pdl: bool,
    repetitions: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        softmax(logits, temperatures, enable_pdl=enable_pdl)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def _run_case(
    batch_size: int,
    vocab_size: int,
    *,
    warmup: int,
    repetitions: int,
    rounds: int,
) -> dict[str, object]:
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    temperatures = torch.ones(batch_size, device="cuda")
    expected = softmax(logits, temperatures, enable_pdl=False)
    actual = softmax(logits, temperatures, enable_pdl=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    for _ in range(warmup):
        softmax(logits, temperatures, enable_pdl=False)
        softmax(logits, temperatures, enable_pdl=True)
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {"disabled": [], "enabled": []}
    for round_index in range(rounds):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for enable_pdl in order:
            key = "enabled" if enable_pdl else "disabled"
            samples[key].append(
                _measure(
                    logits,
                    temperatures,
                    enable_pdl=enable_pdl,
                    repetitions=repetitions,
                )
            )

    disabled = statistics.median(samples["disabled"])
    enabled = statistics.median(samples["enabled"])
    return {
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "disabled_ms_median": disabled,
        "enabled_ms_median": enabled,
        "enabled_vs_disabled_percent": (enabled / disabled - 1) * 100,
        "disabled_ms_samples": samples["disabled"],
        "enabled_ms_samples": samples["enabled"],
        "correctness": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FlashInfer softmax PDL on one GPU")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 32))
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20_260_630)
    args = parser.parse_args()
    if min(*args.batch_sizes, args.vocab_size, args.repetitions, args.rounds) <= 0:
        parser.error("batch sizes, vocabulary size, repetitions, and rounds must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    capability = torch.cuda.get_device_capability(0)
    if capability < (9, 0):
        parser.error(f"PDL requires SM90 or newer; found SM{capability[0]}{capability[1]}")

    torch.manual_seed(args.seed)
    cases = [
        _run_case(
            batch_size,
            args.vocab_size,
            warmup=args.warmup,
            repetitions=args.repetitions,
            rounds=args.rounds,
        )
        for batch_size in args.batch_sizes
    ]
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "capability": list(capability),
                "seed": args.seed,
                "warmup": args.warmup,
                "repetitions_per_round": args.repetitions,
                "rounds": args.rounds,
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
