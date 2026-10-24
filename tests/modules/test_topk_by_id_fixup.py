# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""topk_by_id's signed-zero fix-up on the CPU interpreter: from a selection that ranks +0 above
-0 (as FlashInfer CUB does) to the rule that treats them as equal (smaller ID first)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _order_bits(x, fold):
    bits = ((x.float() + 0.0) if fold else x.float()).view(torch.int32)
    return (bits ^ ((bits >> 31) & 0x7FFFFFFF)).long()


def _select(x, k, fold):
    """Top k by (value, smaller ID); ``fold`` treats -0 as +0 (the rule), else +0 > -0 (CUB)."""
    ids = torch.arange(x.shape[-1])
    words = (_order_bits(x, fold) << 32) | (0xFFFFFFFF - ids)
    return 0xFFFFFFFF - (words.topk(k, sorted=False).values & 0xFFFFFFFF)


def _rows(width):
    x = torch.full((5, width), float("-inf"))
    x[0, 5:11], x[0, 100:111] = -0.0, 0.0  # 17 zeros for 16 slots, -0 at smaller IDs
    x[1] = torch.randn(width)  # boundary away from zero: untouched
    x[2, 50:60], x[2, 3:9], x[2, width - 20 :] = 2.0, -0.0, 0.0
    x[3, 7], x[3, 9] = float("nan"), float("inf")
    x[3, width - 30 :] = -0.0
    x[4] = -0.0
    x[4, 1500:1505] = 0.0
    return x


def _exercise():
    from ctikernels.sampling.triton._topk_by_id import zero_boundary_fixup

    for width in (4099, 2048):
        x = _rows(width)
        for k in (16, 5):
            cub = _select(x, k, fold=False)
            values = x.gather(1, cub).contiguous()
            ids = cub.contiguous()
            zero_boundary_fixup[(x.shape[0],)](
                x, values, ids, width, K=k, KP=1 << (k - 1).bit_length(), BLOCK=1024
            )
            expected = _select(x, k, fold=True)
            assert torch.equal(ids.sort(-1).values, expected.sort(-1).values), (width, k)


def test_fixup_on_cpu_interpreter():
    pytest.importorskip("triton")
    command = f"import importlib; importlib.import_module({__spec__.name!r})._exercise()"
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rule_differs_from_cub_order_at_signed_zeros():
    # The case the fix-up exists for: the two orders pick different zeros.
    x = _rows(4099)
    cub, rule = _select(x, 16, False)[0].sort().values, _select(x, 16, True)[0].sort().values
    assert not torch.equal(cub, rule)
