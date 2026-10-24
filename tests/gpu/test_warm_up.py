# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""After warm-up, first calls and captures within the warmed bounds compile nothing (KW1, KW3)."""

from __future__ import annotations

import pytest
import torch

from ctikernels import radix
from ctikernels.attention import _prologue

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import _capture, _pools, _prefill_case
from ._gdn_prefill_cases import _inputs as _gdn_inputs
from .test_attention_prologue import _call, _inputs, _operation, _table
from .test_gdn_cute_prefill import _REASON as _GDN_REASON

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _no_compile(*args):
    raise AssertionError("compiled after warm-up")


# CI warms short serial rows; the sweep adds the overlap variants.
@pytest.mark.skipif(_GDN_REASON is not None, reason=_GDN_REASON or "")
@pytest.mark.parametrize(
    "max_tokens,max_sequences", ci_cases([(600, 2), (2560, 3)], core=[(600, 2)])
)
def test_gdn_prefill_first_calls_after_warm_up_compile_nothing(
    max_tokens, max_sequences, monkeypatch
):
    from ctikernels.gdn import GatedDeltaNetPrefill

    hk, hv = 16, 32
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute"
    assert op.warm_up(max_tokens, max_sequences=max_sequences) > 0
    assert op.warm_up(max_tokens, max_sequences=max_sequences) == 0
    launcher = op._kernel._cute
    monkeypatch.setattr(launcher, "_compile", _no_compile)
    tokens_list = sorted({1, 7, 128, min(1000, max_tokens), max_tokens})

    def call(tokens, count, captures, seed):
        lengths = [tokens - count + 1] + [1] * (count - 1)
        weights, x = _gdn_inputs(hk, hv, lengths, "ordinary", seed)
        meta = _prefill_case(lengths, list(range(count)), [False] * count)
        pools = _pools(hk, hv, seed=seed + 1)
        extra = {}
        if captures:
            extra = {"capture_rows": torch.full((count,), -1, dtype=torch.int32, device="cuda")}
        return lambda: op(x, *weights.args(), *meta, *pools, **extra)

    with torch.inference_mode():
        for tokens in tokens_list:
            for count in range(1, min(tokens, max_sequences) + 1):
                for captures in (0, 1):
                    call(tokens, count, captures, 9000 + tokens + count)()
        torch.cuda.synchronize()
        # A capture at a shape never run eagerly passes the guard: warm-up marked its variant.
        _capture(call(max(2, max_tokens // 2), 2, 1, 9500))
    torch.cuda.synchronize()


@pytest.mark.parametrize("variant", [(128, 128, False), (256, 64, True)], ids=["hd128", "hd256"])
def test_prologue_first_calls_after_warm_up_compile_nothing(variant, monkeypatch):
    q_heads, kv_heads = (32, 8) if variant[0] == 128 else (16, 4)
    op = _operation(variant, q_heads, kv_heads)
    if op.provider != "cute":
        pytest.skip("the CuTe prologue does not run on this device")
    max_tokens = 2048
    assert op.warm_up(max_tokens) > 0
    assert op.warm_up(max_tokens) == 0
    monkeypatch.setattr(op._kernel, "_compile", _no_compile)
    table = _table(variant)
    for tokens in (1, 7, 64, 128, 1000, max_tokens):
        data = _inputs(variant, tokens, q_heads, kv_heads, seed=tokens)
        _call(variant, op, data, table, q_heads, kv_heads)
    torch.cuda.synchronize()
    assert _prologue.load(op.device, variant) is op._kernel


def test_radix_warm_up_then_compare():
    radix.warm_up()
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    assert radix.fast_compare_key(a, torch.tensor([1, 2, 9], dtype=torch.int64)) == 2
