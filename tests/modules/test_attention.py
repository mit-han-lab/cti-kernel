# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from ctikernels.attention.varlen import CausalVarlenAttention, NoncausalVarlenAttention

_OPERATIONS = pytest.mark.parametrize(
    "operation_type",
    (CausalVarlenAttention, NoncausalVarlenAttention),
    ids=("causal", "noncausal"),
)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16)


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch, forward_impl: object) -> ModuleType:
    package = ModuleType("flash_attn")
    package.__path__ = []
    interface = ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_func = forward_impl
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)
    return interface


def _reject_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_provider(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail("invalid inputs reached the provider"),
    )


@_OPERATIONS
@pytest.mark.parametrize("missing", ("package", "interface", "entry-point"))
def test_constructor_fails_actionably_without_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: type,
    missing: str,
) -> None:
    interface = _install_fake_provider(monkeypatch, lambda: None)
    if missing == "package":
        monkeypatch.setitem(sys.modules, "flash_attn", None)
    if missing in ("package", "interface"):
        monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", None)
    else:
        monkeypatch.delattr(interface, "flash_attn_varlen_func")

    with pytest.raises(
        RuntimeError,
        match=rf"{operation_type.__name__} requires.*install a compatible build.*flash_attn_varlen_func",
    ):
        operation_type()


@_OPERATIONS
@pytest.mark.parametrize("entry_point", (None, object()), ids=("none", "noncallable"))
def test_constructor_fails_actionably_for_a_noncallable_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: type,
    entry_point: object,
) -> None:
    _install_fake_provider(monkeypatch, entry_point)

    with pytest.raises(RuntimeError, match=r"install a compatible build.*flash_attn_varlen_func"):
        operation_type()


@_OPERATIONS
def test_forward_uses_the_function_bound_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: type,
) -> None:
    provider_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_flash_attention(*args: object, **kwargs: object) -> object:
        provider_calls.append((args, kwargs))
        return args[0]

    interface = _install_fake_provider(monkeypatch, fake_flash_attention)
    operation = operation_type()
    # CPU tensors cannot pass the CUDA residency check; the checks are covered below.
    monkeypatch.setattr(
        "ctikernels.attention.varlen._validate_inputs", lambda *_args, **_kwargs: None
    )
    interface.flash_attn_varlen_func = lambda *args, **kwargs: pytest.fail(
        "provider lookup must not recur on the forward path"
    )

    q = torch.empty((3, 4, 8))
    k = torch.empty((3, 2, 8))
    v = torch.empty((3, 2, 8))
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    result = operation.forward(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_sequence_length=3,
    )

    assert result is q
    assert len(provider_calls) == 1
    args, kwargs = provider_calls[0]
    assert len(args) == 3
    assert args[0] is q
    assert args[1] is k
    assert args[2] is v
    assert set(kwargs) == {
        "causal",
        "softmax_scale",
        "cu_seqlens_k",
        "cu_seqlens_q",
        "deterministic",
        "dropout_p",
        "max_seqlen_k",
        "max_seqlen_q",
    }
    assert kwargs["cu_seqlens_q"] is cu_seqlens
    assert kwargs["cu_seqlens_k"] is cu_seqlens
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_k"] == 3
    assert kwargs["dropout_p"] == 0.0
    assert kwargs["causal"] is (operation_type is CausalVarlenAttention)
    assert kwargs["deterministic"] is False


@_OPERATIONS
@pytest.mark.parametrize("head_dim", (64, 72, 128))
def test_forward_rejects_host_inputs_after_metadata_checks(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: type,
    head_dim: int,
) -> None:
    _reject_provider_calls(monkeypatch)
    operation = operation_type()
    q, k, v = _bf16(5, 3, 4, head_dim).unbind(1)

    with pytest.raises(ValueError, match="must reside on CUDA"):
        operation.forward(
            q,
            k,
            v,
            cu_seqlens=torch.tensor([0, 2, 2, 5], dtype=torch.int32),
            max_sequence_length=3,
        )


# Each entry overrides valid CPU arguments and breaks exactly one metadata rule.
_INVALID_METADATA = (
    pytest.param({"q": _bf16(3, 32)}, ValueError, r"\[num_tokens, num_heads", id="rank"),
    pytest.param({"v": _bf16(3, 1, 8)}, ValueError, "identical shapes", id="key-value-shapes"),
    pytest.param({"q": _bf16(2, 4, 8)}, ValueError, "share the token count", id="token-count"),
    pytest.param({"q": _bf16(3, 4, 16)}, ValueError, "share the token count", id="head-dimension"),
    pytest.param({"q": _bf16(3, 0, 8)}, ValueError, "query-head count", id="no-query-heads"),
    pytest.param({"q": _bf16(3, 3, 8)}, ValueError, "query-head count", id="head-ratio"),
    pytest.param({"q": torch.empty((3, 4, 8))}, TypeError, "BF16", id="dtype"),
    pytest.param(
        {"cu_seqlens": torch.tensor([[0, 3]], dtype=torch.int32)},
        ValueError,
        "at least one sequence",
        id="boundaries-rank",
    ),
    pytest.param(
        {"cu_seqlens": torch.tensor([0], dtype=torch.int32)},
        ValueError,
        "at least one sequence",
        id="no-sequences",
    ),
    pytest.param(
        {"cu_seqlens": torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32)},
        ValueError,
        "more sequences than packed tokens",
        id="too-many-sequences",
    ),
    # FlashAttention's single-query GQA path would otherwise read row i as sequence i.
    pytest.param(
        {"cu_seqlens": torch.tensor([0, 1, 1, 2, 3], dtype=torch.int32), "max_sequence_length": 1},
        ValueError,
        "more sequences than packed tokens",
        id="single-token-sequences-with-empty",
    ),
    pytest.param({"cu_seqlens": torch.tensor([0, 3])}, TypeError, "int32", id="boundaries-dtype"),
    pytest.param(
        {"cu_seqlens": torch.tensor([0, 1, 3, 3], dtype=torch.int32)[::2]},
        ValueError,
        "cu_seqlens must be contiguous",
        id="boundaries-stride",
    ),
    pytest.param(
        {"k": _bf16(3, 8, 2).transpose(1, 2)},
        ValueError,
        "contiguous in the head dimension",
        id="head-stride",
    ),
    pytest.param({"max_sequence_length": True}, TypeError, "integer", id="length-bool"),
    pytest.param({"max_sequence_length": 3.0}, TypeError, "integer", id="length-float"),
    pytest.param({"max_sequence_length": 0}, ValueError, "token extent", id="length-zero"),
    pytest.param({"max_sequence_length": 4}, ValueError, "token extent", id="length-overflow"),
)


@_OPERATIONS
@pytest.mark.parametrize(("overrides", "error", "message"), _INVALID_METADATA)
def test_forward_rejects_invalid_metadata_before_device_checks(
    monkeypatch: pytest.MonkeyPatch,
    operation_type: type,
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    _reject_provider_calls(monkeypatch)
    operation = operation_type()
    inputs = {
        "q": _bf16(3, 4, 8),
        "k": _bf16(3, 2, 8),
        "v": _bf16(3, 2, 8),
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        "max_sequence_length": 3,
    } | overrides

    with pytest.raises(error, match=message):
        operation.forward(**inputs)


@_OPERATIONS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_asymmetric_varlen_forwarding_and_metadata(monkeypatch, operation_type, dtype):
    from unittest.mock import Mock

    provider = Mock(side_effect=lambda q, *args, **kwargs: q)
    _install_fake_provider(monkeypatch, provider)
    # Pretend residency only; retain real shapes, strides and dtypes for validation.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.empty(16, 8, 64, dtype=dtype)
    k = torch.empty(1024, 2, 64, dtype=dtype)
    cq = torch.tensor([0, 1, 16], dtype=torch.int32)
    ck = torch.tensor([0, 512, 1024], dtype=torch.int32)
    result = operation_type().forward(
        q,
        k,
        k,
        cu_seqlens=cq,
        max_sequence_length=15,
        cu_seqlens_k=ck,
        max_sequence_length_k=512,
        softmax_scale=0.07,
    )
    assert result is q
    actual = provider.call_args.kwargs
    assert actual["cu_seqlens_q"] is cq and actual["cu_seqlens_k"] is ck
    assert actual["max_seqlen_q"] == 15 and actual["max_seqlen_k"] == 512
    assert actual["softmax_scale"] == 0.07


@_OPERATIONS
@pytest.mark.parametrize(
    "overrides,error,match",
    [
        ({"max_sequence_length_k": None}, ValueError, "together"),
        ({"cu_seqlens_k": None}, ValueError, "together"),
        (
            {"cu_seqlens_k": torch.tensor([0, 1, 3], dtype=torch.int32)},
            ValueError,
            "sequence count",
        ),
        ({"cu_seqlens_k": torch.tensor([0, 3])}, TypeError, "int32"),
        ({"cu_seqlens_k": torch.zeros(4, dtype=torch.int32)[::2]}, ValueError, "contiguous"),
        ({"max_sequence_length_k": True}, TypeError, "integer"),
        ({"max_sequence_length_k": 4}, ValueError, "KV token extent"),
    ],
)
def test_asymmetric_metadata_errors(monkeypatch, operation_type, overrides, error, match):
    _reject_provider_calls(monkeypatch)
    inputs = {
        "q": _bf16(3, 4, 8),
        "k": _bf16(3, 2, 8),
        "v": _bf16(3, 2, 8),
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        "max_sequence_length": 3,
        "cu_seqlens_k": torch.tensor([0, 3], dtype=torch.int32),
        "max_sequence_length_k": 3,
    }
    with pytest.raises(error, match=match):
        operation_type().forward(**(inputs | overrides))
