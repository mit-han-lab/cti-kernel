# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear._dispatch import TUNED, Config, tuned_config
from ctikernels.linear.cute import TABLES, VARIANTS, smallm_bf16_linear
from ctikernels.linear.cute.variants import DEVICE_KEYS, _operation


def test_registration_and_table_roundtrip():
    variant = VARIANTS["smallm-bf16"]
    assert callable(variant) and variant.call is smallm_bf16_linear
    assert variant.dtype == "bf16" and variant.max_m == 64
    assert set(TABLES) == {(12, 0, 170), (12, 0, 188), (12, 1, 48)}
    assert {kind: TABLES[key] for kind, key in DEVICE_KEYS.items()} == TUNED
    for kind, key in DEVICE_KEYS.items():
        for n, k in TUNED[kind]:
            for m in range(66):
                assert variant.supports(key, n, k, m) == (tuned_config(kind, n, k, m) is not None)
    assert not variant.supports((12, 0, 169), 4096, 4096, 1)
    assert not variant.supports((12, 0, 170), 7, 8, 1)
    assert not variant.supports((12, 0, 170), 12288, 4096, 8)


def test_function_fallback_out_and_policy(monkeypatch):
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "invalid")
    x, w = torch.randn(3, 8).bfloat16(), torch.randn(7, 8).bfloat16()
    out = torch.empty(3, 7, dtype=torch.bfloat16)
    assert smallm_bf16_linear(x, w, out=out) is out
    torch.testing.assert_close(out, F.linear(x, w))
    cfg = Config(16, 64, 2, 2)
    assert _operation(cfg).config == cfg and _operation(cfg)._override == "cute"
    assert _operation(None)._override == "auto"
    torch.testing.assert_close(VARIANTS["smallm-bf16"](x, w, config=cfg), F.linear(x, w))
    with pytest.raises(TypeError, match="Config"):
        smallm_bf16_linear(x, w, config="invalid")
    with pytest.raises(TypeError, match="BF16"):
        smallm_bf16_linear(x.float(), w.float())
    with pytest.raises(ValueError, match="requires_grad"):
        smallm_bf16_linear(x, w.requires_grad_())
    with pytest.raises(ValueError, match="requires_grad"):
        smallm_bf16_linear(x.requires_grad_(), w.detach())
