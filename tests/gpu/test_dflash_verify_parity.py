# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""VERIFY migration contract: FP32 accuracy and mutual atol=rtol=1e-2; reports ulp spread."""

import pytest
import torch

from ctikernels.attention.inference import FlashInferFixedQueryAttention

from ._dflash import Fixture
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("b,q", ci_cases([1, 2, 3, 4], [4, 8, 16], core=[(1, 4), (4, 16)]))
@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
def test_verify_fixed_query_parity(b, q, provider, monkeypatch, record_property):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    f = Fixture(b, q, 16, 32, 8, True, None, c=1024, provider=provider)
    geometry = {"query_length": q, "num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128}
    sizes = FlashInferFixedQueryAttention.workspace_size(
        batch_size=b, device=f.query.device, **geometry
    )
    work = [torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes]
    # FixedQuery admits only KV lengths >= Q and has no inactive-slot input.
    # Compare common admitted inputs first, then deactivate DRAFT rows and check
    # unchanged active results and zero inactive output against that same baseline.
    fixed_qp = torch.arange(b + 1, dtype=torch.int32, device="cuda") * q
    fixed = FlashInferFixedQueryAttention(
        *work,
        query_indptr=fixed_qp,
        kv_indptr=f.page_ptr,
        kv_indices=f.page_idx,
        last_page_lengths=f.last,
        page_size=16,
        dtype=torch.bfloat16,
        causal=True,
        **geometry,
    )
    fixed_out = torch.empty_like(f.out)
    identical, elements, entire_cases = 0, 0, 0
    max_abs, max_ulps = 0.0, 0.0
    for turn, length in enumerate((100, 400, 1000, 16, 17, 1024)):
        slots = list(range(b))
        lengths = [max(q, length - row * 7) for row in range(b)]
        f.stage([n - q for n in lengths], slots, turn)
        expected = f.oracle()
        f.run()
        fixed.prepare()
        fixed.run(q=f.query, paged_kv_cache=(f.k, f.v), out=fixed_out)
        a, other = f.out.float(), fixed_out.float()
        torch.testing.assert_close(a, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(other, expected, atol=0.01, rtol=0.01)
        magnitude = torch.maximum(a.abs(), other.abs()).to(torch.bfloat16)
        ulp = (
            torch.nextafter(magnitude, torch.full_like(magnitude, torch.inf)).float()
            - magnitude.float()
        )
        # Both kernels round softmax probabilities to BF16 against different running
        # maxima, so outputs differ by several BF16 ulps: the contract is mutual
        # closeness at half the reference tolerance, with the spread reported.
        torch.testing.assert_close(a, other, atol=1e-2, rtol=1e-2)
        diff = (a - other).abs()
        max_abs = max(max_abs, float(diff.max()))
        max_ulps = max(max_ulps, float((diff / ulp).max()))
        equal = f.out.view(torch.int16) == fixed_out.view(torch.int16)
        identical += int(equal.sum())
        elements += equal.numel()
        entire_cases += int(equal.all())
        f.slots[-1] = -1
        f.cpu_slots[-1] = -1
        f.out.fill_(torch.nan)
        f.run()
        inactive_expected = f.oracle()
        torch.testing.assert_close(f.out.float(), inactive_expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(f.out[-q:]) == 0
        torch.testing.assert_close(f.out[:-q].float(), other[:-q], atol=1e-2, rtol=1e-2)
    report = (
        f"{identical}/{elements} bitwise identical elements; {entire_cases}/6 identical cases; "
        f"max |a-b| {max_abs:.3g}, {max_ulps:.1f} BF16 ulps"
    )
    print(f"VERIFY {provider} B={b} Q={q}: {report}")
    record_property("verify_bitwise_parity", report)
