# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import pytest

from ctikernels.linear import _lora_policy as policy

_KNOWN = (
    ("NVIDIA GeForce RTX 5090", (12, 0), 170, "rtx5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0), 188, "pro6000"),
    ("NVIDIA GB10", (12, 1), 48, "gb10"),
)


@pytest.mark.parametrize("device_name,capability,sms,device_class", _KNOWN)
def test_device_class_matches_linear_dispatch(device_name, capability, sms, device_class):
    admitted = policy.policy_for_hardware(device_name=device_name, capability=capability, sms=sms)
    assert admitted.device_class == device_class
    assert admitted.triton_available


@pytest.mark.parametrize(
    "capability", [(8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1), (13, 0)]
)
@pytest.mark.parametrize("sms", [1, 48, 170, 188, 200])
def test_segmented_admission_from_cc89(capability, sms):
    from ctikernels.linear._dispatch import device_class

    admitted = policy.policy_for_hardware(device_name="any name", capability=capability, sms=sms)
    assert not admitted.triton_available
    assert admitted.device_class == (device_class(capability, sms) or "other")
    assert policy.select_provider(admitted, num_slices=3, rank=8, num_tokens=1, k=1024) == (
        "segmented"
    )


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6)])
def test_below_cc89_is_not_admitted(capability):
    assert policy.policy_for_hardware(device_name=_KNOWN[0][0], capability=capability) is None


@pytest.mark.parametrize("device_name,capability,sms,device_class", _KNOWN)
def test_frozen_triton_admission_is_exact_and_nondeterministic(
    device_name, capability, sms, device_class
):
    admitted = policy.policy_for_hardware(device_name=device_name, capability=capability, sms=sms)
    other_capability = (12, 1) if capability == (12, 0) else (12, 0)
    for name, cap in (("any name", capability), (device_name, other_capability)):
        assert not policy.policy_for_hardware(device_name=name, capability=cap).triton_available
    for rank, tokens, expected in [
        (1, 1, "segmented"),
        (129, 1, "segmented"),
        (128, 65, "segmented"),
        (513, 1, None),
        (0, 1, None),
        (1, 0, None),
    ]:
        for deterministic in (True, False):
            assert (
                policy.select_provider(
                    admitted,
                    num_slices=3,
                    rank=rank,
                    num_tokens=tokens,
                    k=1024,
                    deterministic=deterministic,
                )
                == expected
            )


@pytest.mark.parametrize("num_slices", [1, 2, 3, 6, 9])
@pytest.mark.parametrize("rank", [1, 8, 16, 64, 128, 129, 256, 512])
@pytest.mark.parametrize("num_tokens", [1, 3, 64, 65, 200, 1024, 100_000])
def test_segmented_ranges(num_slices, rank, num_tokens):
    admitted = policy._SMALL_BATCH_POLICY
    assert policy.supports_projection(num_slices=num_slices, rank=rank)
    assert policy.prefer_small_batch_plan(
        admitted, num_slices=num_slices, rank=rank, num_tokens=num_tokens
    )
    assert policy.provider_for("other", num_slices, rank, num_tokens, 13) == "segmented"


@pytest.mark.parametrize("slices,rank,tokens", [(0, 8, 1), (1, 0, 1), (1, 513, 1), (1, 8, 0)])
def test_invalid_ranges(slices, rank, tokens):
    assert not policy.prefer_small_batch_plan(
        policy._SMALL_BATCH_POLICY, num_slices=slices, rank=rank, num_tokens=tokens
    )


def test_provider_table_rows_stay_in_frozen_triton_range():
    for (device, slices, rank, tokens, k), provider in policy.PROVIDER_TABLE.items():
        assert device in ("rtx5090", "pro6000", "gb10") and provider == "triton"
        assert slices >= 1 and 1 <= rank <= 128 and tokens in policy.TOKEN_EDGES and k > 0


def test_provider_table_uses_token_buckets(monkeypatch):
    monkeypatch.setitem(policy.PROVIDER_TABLE, ("gb10", 1, 64, 8, 1000), "triton")
    assert [policy.provider_for("gb10", 1, 64, t, 1000) for t in (4, 5, 8, 9)] == [
        "segmented",
        "triton",
        "triton",
        "segmented",
    ]


def test_provider_table_selects_exact_row_without_broadening_triton(monkeypatch):
    key = ("rtx5090", 3, 128, 64, 1024)
    monkeypatch.setitem(policy.PROVIDER_TABLE, key, "triton")
    admitted = policy.SmallBatchLoRAPolicy("rtx5090", True)
    assert policy.provider_for(*key) == "triton"
    assert (
        policy.select_provider(
            admitted, num_slices=3, rank=128, num_tokens=64, k=1024, deterministic=False
        )
        == "triton"
    )
    assert (
        policy.select_provider(admitted, num_slices=3, rank=128, num_tokens=64, k=1024)
        == "segmented"
    )
    for changed in [
        ("gb10", 3, 128, 64, 1024),
        ("rtx5090", 2, 128, 64, 1024),
        ("rtx5090", 3, 64, 64, 1024),
        ("rtx5090", 3, 128, 32, 1024),
        ("rtx5090", 3, 128, 64, 2048),
    ]:
        assert policy.provider_for(*changed) == "segmented"
    for rank, tokens in [(256, 64), (128, 65)]:
        monkeypatch.setitem(policy.PROVIDER_TABLE, ("rtx5090", 3, rank, tokens, 1024), "triton")
        assert (
            policy.select_provider(
                admitted, num_slices=3, rank=rank, num_tokens=tokens, k=1024, deterministic=False
            )
            == "segmented"
        )
    assert (
        policy.select_provider(
            policy.SmallBatchLoRAPolicy("rtx5090", False),
            num_slices=3,
            rank=128,
            num_tokens=64,
            k=1024,
        )
        == "segmented"
    )


@pytest.mark.parametrize("key", sorted(policy.PROVIDER_TABLE))
def test_deterministic_default_routes_measured_triton_rows_to_segmented(key):
    device, slices, rank, tokens, k = key
    admitted = policy.SmallBatchLoRAPolicy(device, True)
    kwargs = {"num_slices": slices, "rank": rank, "num_tokens": tokens, "k": k}
    assert policy.select_provider(admitted, **kwargs) == "segmented"
    assert policy.select_provider(admitted, **kwargs, deterministic=False) == "triton"
