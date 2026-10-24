# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._dispatch import device_class as classify_device

Provider = Literal["segmented", "triton"]

# Measured exceptions keyed by (device_class, num_slices, rank, token bucket, K),
# where the token bucket is the smallest measured edge >= num_tokens. Missing rows
# prefer the segmented delta (our Triton kernel pair). The
# table applies only to nondeterministic plans (the frozen Triton split-K uses FP
# atomics); deterministic plans, the default, always use the segmented delta.
# Admission is checked separately, including for rows that request the frozen
# Triton provider outside its support range.
TOKEN_EDGES = (1, 2, 3, 4, 8, 16, 32, 64)
# BEGIN GENERATED PROVIDER TABLE
PROVIDER_TABLE: dict[tuple[str, int, int, int, int], Provider] = {
    ("gb10", 1, 16, 64, 6144): "triton",
    ("gb10", 1, 64, 16, 4096): "triton",
    ("gb10", 1, 64, 16, 6144): "triton",
    ("gb10", 1, 64, 32, 4096): "triton",
    ("gb10", 1, 64, 32, 6144): "triton",
    ("gb10", 1, 64, 64, 4096): "triton",
    ("gb10", 1, 128, 16, 4096): "triton",
    ("gb10", 1, 128, 32, 4096): "triton",
    ("gb10", 2, 128, 1, 4096): "triton",
    ("gb10", 2, 128, 16, 4096): "triton",
    ("gb10", 6, 128, 1, 4096): "triton",
    ("gb10", 6, 128, 1, 5120): "triton",
    ("gb10", 6, 128, 32, 4096): "triton",
    ("pro6000", 1, 64, 32, 4096): "triton",
    ("pro6000", 1, 64, 32, 6144): "triton",
    ("pro6000", 1, 64, 64, 6144): "triton",
    ("pro6000", 2, 16, 64, 4096): "triton",
    ("pro6000", 2, 16, 64, 5120): "triton",
    ("pro6000", 2, 64, 32, 5120): "triton",
    ("pro6000", 2, 64, 64, 4096): "triton",
    ("pro6000", 2, 128, 32, 5120): "triton",
    ("pro6000", 6, 64, 64, 4096): "triton",
    ("pro6000", 6, 128, 16, 5120): "triton",
    ("rtx5090", 1, 16, 64, 6144): "triton",
    ("rtx5090", 1, 64, 8, 6144): "triton",
    ("rtx5090", 1, 64, 16, 6144): "triton",
    ("rtx5090", 1, 64, 32, 4096): "triton",
    ("rtx5090", 1, 64, 32, 6144): "triton",
    ("rtx5090", 1, 64, 64, 4096): "triton",
    ("rtx5090", 1, 64, 64, 6144): "triton",
    ("rtx5090", 2, 16, 32, 5120): "triton",
    ("rtx5090", 2, 16, 64, 4096): "triton",
    ("rtx5090", 2, 64, 32, 4096): "triton",
    ("rtx5090", 2, 64, 32, 5120): "triton",
    ("rtx5090", 2, 64, 64, 4096): "triton",
    ("rtx5090", 2, 128, 32, 4096): "triton",
    ("rtx5090", 2, 128, 32, 5120): "triton",
    ("rtx5090", 2, 128, 64, 4096): "triton",
    ("rtx5090", 3, 128, 32, 4096): "triton",
    ("rtx5090", 6, 64, 32, 5120): "triton",
    ("rtx5090", 6, 64, 64, 4096): "triton",
    ("rtx5090", 6, 64, 64, 5120): "triton",
    ("rtx5090", 6, 128, 1, 4096): "triton",
    ("rtx5090", 6, 128, 2, 4096): "triton",
    ("rtx5090", 6, 128, 3, 4096): "triton",
    ("rtx5090", 6, 128, 4, 4096): "triton",
    ("rtx5090", 6, 128, 8, 4096): "triton",
    ("rtx5090", 6, 128, 16, 4096): "triton",
    ("rtx5090", 6, 128, 32, 4096): "triton",
    ("rtx5090", 6, 128, 32, 5120): "triton",
}
# END GENERATED PROVIDER TABLE
_TRITON_HARDWARE = frozenset(
    {
        ("NVIDIA GeForce RTX 5090", (12, 0)),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)),
        ("NVIDIA GB10", (12, 1)),
    }
)


def segmented_provider(capability: tuple[int, int]) -> str | None:
    """Segmented-delta provider: our Triton on every cc >= 8.9 (below: none, the torch path)."""
    return "triton" if tuple(capability) >= (8, 9) else None


@dataclass(frozen=True, slots=True)
class SmallBatchLoRAPolicy:
    device_class: str = "other"
    triton_available: bool = False  # the frozen FP-atomic Triton provider
    segmented: str = "triton"  # segmented_provider() of the device


# Structural segmented-delta limits, also useful to offline consumers without a CUDA device.
_SMALL_BATCH_POLICY = SmallBatchLoRAPolicy()


def provider_for(
    device_class: str, num_slices: int, rank: int, num_tokens: int, k: int
) -> Provider:
    edge = next((e for e in TOKEN_EDGES if num_tokens <= e), None)
    if edge is None:
        return "segmented"
    return PROVIDER_TABLE.get((device_class, num_slices, rank, edge, k), "segmented")


def policy_for_hardware(
    *,
    device_name: str,
    capability: tuple[int, int],
    sms: int = 0,
) -> SmallBatchLoRAPolicy | None:
    segmented = segmented_provider(capability)
    if segmented is None:
        return None
    return SmallBatchLoRAPolicy(
        device_class=classify_device(capability, sms) or "other",
        triton_available=(device_name, tuple(capability)) in _TRITON_HARDWARE,
        segmented=segmented,
    )


def supports_projection(*, num_slices: int, rank: int) -> bool:
    # The segmented delta covers ranks 1..512, a superset of the frozen Triton range.
    return num_slices > 0 and 1 <= rank <= 512


def select_provider(
    policy: SmallBatchLoRAPolicy,
    *,
    num_slices: int,
    rank: int,
    num_tokens: int,
    k: int,
    deterministic: bool = True,
) -> Provider | None:
    """The segmented delta is deterministic on every route. Measured frozen-Triton
    rows apply only with ``deterministic=False``."""
    if num_tokens < 1 or not supports_projection(num_slices=num_slices, rank=rank):
        return None
    triton_admitted = (
        not deterministic and policy.triton_available and rank <= 128 and num_tokens <= 64
    )
    preferred = provider_for(policy.device_class, num_slices, rank, num_tokens, k)
    if triton_admitted and preferred == "triton":
        return "triton"
    return "segmented"


def prefer_small_batch_plan(
    policy: SmallBatchLoRAPolicy,
    *,
    num_slices: int,
    rank: int,
    num_tokens: int,
    k: int = 0,
    deterministic: bool = True,
) -> bool:
    return (
        select_provider(
            policy,
            num_slices=num_slices,
            rank=rank,
            num_tokens=num_tokens,
            k=k,
            deterministic=deterministic,
        )
        is not None
    )
