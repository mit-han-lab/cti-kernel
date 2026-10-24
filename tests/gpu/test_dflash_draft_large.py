# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Int64 boundary regression; select with -k int64_bounds for memcheck.

Only token/page indices are large (~8.5 GiB); K/V repeats 32 physical pages.
Run directly under compute-sanitizer --tool memcheck with pytest -s to see bytes.
"""

import pytest
import torch

from ctikernels.attention.draft import DraftAttention

from .._dflash_reference import draft_attention_reference

pytestmark = [
    pytest.mark.exhaustive,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("causal,window", [(True, 4096), (False, 2048)])
def test_cute_int64_bounds(causal, window, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    length, query, page, physical_pages = 2**31 - 1, 8, 16, 32
    hq, hk, dim = 16, 4, 128
    pages = (length + page - 1) // page
    geometry = {
        "query_length": query,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
        "page_size": page,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
        "provider": "cute",
    }
    sizes = DraftAttention.workspace_size(
        batch_size=1, max_kv_tokens=length, device=torch.device("cuda"), **geometry
    )
    # Exact persistent payload plus bounded fill scratch and a conservative 64 MiB
    # allowance for the small visible-key FP32 oracle and allocator alignment.
    tile = 1 << 20
    persistent = (length + pages + 8) * 4 + query * 8
    persistent += 2 * query * hq * dim * 2 + 2 * physical_pages * page * hk * dim * 2
    need = persistent + sum(sizes) + tile * 4 + (64 << 20)
    print(f"int64_bounds allocation budget: {need} bytes ({need / 2**30:.3f} GiB)", flush=True)
    assert need < 12 << 30
    free, _ = torch.cuda.mem_get_info()
    if free < need + (2 << 30):
        pytest.skip(f"int64_bounds needs {need} bytes + 2 GiB headroom; free={free} bytes")

    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    tokens = torch.empty(length, dtype=torch.int32, device="cuda")
    # Logical page j maps to physical page j % physical_pages, preserving offsets.
    pattern = torch.arange(tile, dtype=torch.int32, device="cuda")
    pattern.remainder_(physical_pages * page)
    for start in range(0, length, tile):
        count = min(tile, length - start)
        tokens[start : start + count].copy_(pattern[:count])
    del pattern
    slots, ptr = ints([0]), ints([0, length])
    positions = torch.arange(length - query, length, dtype=torch.int64, device="cuda")
    qp, pp, last = ints([-1, -1]), ints([-1, -1]), ints([-1])
    pi = torch.empty(pages, dtype=torch.int32, device="cuda")
    work = [torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes]
    op = DraftAttention(
        *work,
        request_indices=slots,
        positions=positions,
        token_indptr=ptr,
        token_indices=tokens,
        query_indptr=qp,
        page_indptr=pp,
        page_indices=pi,
        last_page_lengths=last,
        **geometry,
    )
    generator = torch.Generator(device="cuda").manual_seed(731)
    q = torch.randn(query, hq, dim, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(
        physical_pages, page, hk, dim, dtype=q.dtype, device=q.device, generator=generator
    )
    v = torch.randn(k.shape, dtype=q.dtype, device=q.device, generator=generator)
    out = torch.empty_like(q)
    lo = length - query - window + 1
    visible = torch.arange(lo, length, dtype=torch.int64, device="cuda")
    # The existing oracle infers the suffix base from positions and visible length.
    expected = draft_attention_reference(
        q,
        k,
        v,
        slots=slots,
        positions=positions,
        token_indptr=ints([0, length - lo]),
        token_indices=visible % (physical_pages * page),
        query_length=query,
        causal=causal,
        window=window,
    )
    for fused in (False, True):
        qp.fill_(-1)
        pp.fill_(-1)
        last.fill_(-1)
        if fused:
            op.prepare_and_run(q=q, paged_kv_cache=(k, v), out=out)
        else:
            op.prepare()
            op.run(q=q, paged_kv_cache=(k, v), out=out)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
        assert qp.tolist() == [0, query]
        assert pp.tolist() == [0, pages]
        assert last.tolist() == [page - 1]
        sample = torch.tensor([0, 1, pages // 2, pages - 1], device="cuda")
        torch.testing.assert_close(pi[sample].long(), sample % physical_pages, atol=0, rtol=0)
