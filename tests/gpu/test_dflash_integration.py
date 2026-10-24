# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Attention integration checks independent of round bookkeeping."""

import os

import pytest
import torch

from ._dflash import Fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
EXTENDED = os.environ.get("CTIK_DFLASH_EXTENDED") == "1"
ATOL = RTOL = 2e-2


@pytest.fixture(autouse=True)
def provider_and_oracle_precision():
    pytest.importorskip("flashinfer")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize(
    "q,causal,window",
    [(8, False, 2048)] + ([(16, False, None), (16, True, 4096)] if EXTENDED else []),
)
def test_tp2_local_head_slice_matches_tp1_oracle(q, causal, window):
    full = Fixture(2, q, 1, 32, 8, causal, window, c=4160)
    full.stage([2049, 4097], [1, 0])
    expected = full.oracle()
    local = Fixture(2, q, 1, 16, 4, causal, window, c=4160)
    local.stage([2049, 4097], [1, 0])
    for rank in (0, 1):
        local.query.copy_(full.query[:, rank * 16 : (rank + 1) * 16])
        local.k.copy_(full.k[:, :, rank * 4 : (rank + 1) * 4])
        local.v.copy_(full.v[:, :, rank * 4 : (rank + 1) * 4])
        local.run()
        torch.testing.assert_close(
            local.out.float(), expected[:, rank * 16 : (rank + 1) * 16], atol=ATOL, rtol=RTOL
        )
        local.check_metadata()


@pytest.mark.parametrize("causal,window", [(False, None), (True, 4096), (False, 2048)])
def test_attention_poison_isolation_and_rng_preservation(causal, window):
    """Retain the attention half of the retired joint K0 helper test."""
    f = Fixture(4, 8, 1, 32, 8, causal, window, c=128)
    for slots in ([0, -1, 2, -1], [-1, 1, -1, 3], [-1] * 4):
        outputs = []
        for poison in (-(2**30), 2**30):
            f.stage([32, 64, 17, 1], slots)
            ptr, indices = [0], []
            for row, slot in enumerate(slots):
                a, b = f.cpu_token_ptr[row : row + 2].tolist()
                indices.extend([poison] * 7 if slot == -1 else f.cpu_token_idx[a:b].tolist())
                ptr.append(len(indices))
            f.cpu_token_ptr = torch.tensor(ptr, dtype=torch.int32)
            f.cpu_token_idx.fill_(poison)
            f.cpu_token_idx[: len(indices)] = torch.tensor(indices, dtype=torch.int32)
            f.token_ptr.copy_(f.cpu_token_ptr)
            f.token_idx.copy_(f.cpu_token_idx)
            rng = torch.cuda.get_rng_state()
            f.run()
            assert torch.equal(rng, torch.cuda.get_rng_state())
            torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
            outputs.append(f.out.clone())
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
