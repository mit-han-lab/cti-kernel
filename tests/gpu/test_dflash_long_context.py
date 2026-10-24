# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Default-suite long-context DRAFT and target VERIFY, eager and changing graphs.

Each case keeps 256 MiB of random BF16 K/V in a fragmented, repeated page pool.
Logical token/page CSR still spans the full B*C capacity. The FP32 oracle uses
16384-key chunks (at most 32 MiB of scores for these geometries), or just the
visible suffix for windowed masks. Cold provider compilation is not timed here.
"""

import pytest
import torch

from ctikernels.attention.inference import DraftAttention, FlashInferFixedQueryAttention

from .._dflash_reference import chunked_draft_attention_reference
from ._dflash import SENTINEL
from .test_dflash_draft_window import capture, compare_replay

PAGE = 16
POOL_PAGES = 2048  # Plus one poisoned fragmentation page per usable page.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def long_context_environment(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


class LongContextCase:
    """Fixture protocol used by capture/compare_replay, with bounded physical K/V."""

    def __init__(self, batch, capacity, query, heads, kv_heads, dim, causal, window, provider):
        assert batch * capacity < 2**31, "token CSR must fit signed int32"
        self.b, self.c, self.q = batch, capacity, query
        self.causal, self.window = causal, window
        self.fixed = provider == "fixed-query"
        geometry = {
            "query_length": query,
            "num_qo_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": dim,
        }
        operation = FlashInferFixedQueryAttention if self.fixed else DraftAttention
        draft = {} if self.fixed else {"max_kv_tokens": capacity}
        options = {"page_size": PAGE, "dtype": torch.bfloat16, "causal": causal}
        if not self.fixed:
            options.update(window=window, provider=provider)
            draft.update(options)
        sizes = operation.workspace_size(
            batch_size=batch, device=torch.device("cuda"), **geometry, **draft
        )
        pages = batch * ((capacity + PAGE - 1) // PAGE)
        kv_bytes = 2 * (2 * POOL_PAGES) * PAGE * kv_heads * dim * 2
        metadata_bytes = 4 * (batch * capacity + pages + 5 * batch + 3) + 8 * batch * query
        io_bytes = 2 * batch * query * heads * dim * 2 + 512
        # Include graph/provider allocations, FP32 chunk scratch, and allocator
        # slack. Reserve another GiB for CUDA libraries and other live tests.
        budget = kv_bytes + metadata_bytes + io_bytes + sum(sizes) + (1 << 30)
        assert budget < 8 << 30
        free, _ = torch.cuda.mem_get_info()
        print(
            f"long_context {provider} B={batch} C={capacity}: "
            f"budget={budget / 2**30:.3f} GiB + 1 GiB headroom, free={free / 2**30:.3f} GiB",
            flush=True,
        )
        if free < budget + (1 << 30):
            pytest.skip(f"insufficient free CUDA memory: need {budget + (1 << 30)}, free={free}")

        def ints(count):
            return torch.empty(count, dtype=torch.int32, device="cuda")

        self.slots = ints(batch)
        self.positions = torch.empty(batch * query, dtype=torch.int64, device="cuda")
        self.token_ptr, self.token_idx = ints(batch + 1), ints(batch * capacity)
        self.query_ptr, self.page_ptr = ints(batch + 1), ints(batch + 1)
        self.page_idx, self.last = ints(pages), ints(batch)
        self.workspaces = [torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes]
        metadata = {"query_indptr": self.query_ptr, "last_page_lengths": self.last}
        if self.fixed:
            metadata.update(kv_indptr=self.page_ptr, kv_indices=self.page_idx)
        else:
            metadata.update(
                request_indices=self.slots,
                positions=self.positions,
                token_indptr=self.token_ptr,
                token_indices=self.token_idx,
                page_indptr=self.page_ptr,
                page_indices=self.page_idx,
            )
        # Valid explicit provider geometries must construct; never catch/skip errors.
        self.op = operation(*self.workspaces, **metadata, **geometry, **options)
        self.query = torch.empty(batch * query, heads, dim, dtype=torch.bfloat16, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(1739)
        shape = (2 * POOL_PAGES, PAGE, kv_heads, dim)
        self.k, self.v = (
            torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
            for _ in range(2)
        )
        self.k[::2].fill_(torch.nan)
        self.v[::2].fill_(torch.nan)
        self.out_backing = torch.full(
            (self.query.numel() + 256,), 123.0, dtype=torch.bfloat16, device="cuda"
        )
        self.out = self.out_backing[128:-128].view_as(self.query)

    def stage(self, lengths, turn):
        assert len(lengths) == self.b and all(self.q <= n <= self.c for n in lengths)
        self.cpu_slots = torch.arange(self.b, dtype=torch.int32)
        self.cpu_positions = torch.cat(
            [torch.arange(n - self.q, n, dtype=torch.int64) for n in lengths]
        )
        self.cpu_token_ptr = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        counts = [(n + PAGE - 1) // PAGE for n in lengths]
        self.expected_page_ptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        self.expected_query_ptr = torch.arange(self.b + 1, dtype=torch.int32) * self.q
        self.expected_last = torch.tensor([(n - 1) % PAGE + 1 for n in lengths], dtype=torch.int32)
        self.cpu_token_idx = torch.full((self.b * self.c,), SENTINEL, dtype=torch.int32)
        self.expected_pages = torch.full(self.page_idx.shape, SENTINEL, dtype=torch.int32)
        for row, (length, count) in enumerate(zip(lengths, counts, strict=True)):
            # An odd stride permutes the power-of-two pool; all selected pages
            # are odd, so a bad physical address encounters a poisoned gap.
            physical = torch.arange(count, dtype=torch.int32) * 31 + row * 97 + turn * 53
            physical = physical.remainder(POOL_PAGES) * 2 + 1
            begin, end = self.expected_page_ptr[row : row + 2].tolist()
            self.expected_pages[begin:end] = physical
            begin, end = self.cpu_token_ptr[row : row + 2].tolist()
            self.cpu_token_idx[begin:end] = (
                physical[:, None] * PAGE + torch.arange(PAGE, dtype=torch.int32)
            ).flatten()[:length]
        for actual, expected in self.metadata():
            actual.copy_(expected)
        if not self.fixed:
            # DRAFT must regenerate page/query metadata from token CSR each replay.
            for value in (self.query_ptr, self.page_ptr, self.page_idx, self.last):
                value.fill_(SENTINEL)
        self.query.normal_(generator=torch.Generator(device="cuda").manual_seed(1739 + turn))
        self.out.fill_(torch.nan)

    def metadata(self):
        return (
            (self.slots, self.cpu_slots),
            (self.positions, self.cpu_positions),
            (self.token_ptr, self.cpu_token_ptr),
            (self.token_idx, self.cpu_token_idx),
            (self.query_ptr, self.expected_query_ptr),
            (self.page_ptr, self.expected_page_ptr),
            (self.page_idx, self.expected_pages),
            (self.last, self.expected_last),
        )

    def run(self):
        self.op.prepare()
        return self.op.run(q=self.query, paged_kv_cache=(self.k, self.v), out=self.out)

    def oracle(self):
        return chunked_draft_attention_reference(
            self.query,
            self.k,
            self.v,
            slots=self.cpu_slots,
            positions=self.cpu_positions,
            token_indptr=self.cpu_token_ptr,
            token_indices=self.cpu_token_idx,
            query_length=self.q,
            causal=self.causal,
            window=self.window,
        )

    def check_metadata(self):
        for actual, expected in self.metadata():
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        assert (self.out_backing[:128] == 123).all()
        assert (self.out_backing[-128:] == 123).all()


def check_long_context(case):
    # Capture shorter, different lengths even for B=1; replay twice without
    # recapture. Every checked batch contains a full-capacity request.
    case.stage([case.q + 29 * row for row in range(case.b)], turn=0)
    graph = capture([case])
    lengths = [case.c, case.c // 2 + 13, 4099, case.q + 3][: case.b]
    case.stage(lengths, turn=1)
    compare_replay([case], graph)
    lengths = [case.c, case.c // 3 + 7, 2051, case.q][: case.b]
    case.stage(lengths[1:] + lengths[:1], turn=2)
    compare_replay([case], graph)


@pytest.mark.parametrize("capacity", [131072, 262144], ids=lambda c: f"c{c}")
@pytest.mark.parametrize("batch", [1, 2, 3, 4], ids=lambda b: f"b{b}")
@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
@pytest.mark.parametrize(
    "query,causal,window",
    [
        pytest.param(16, True, 4096, id="qwen35-causal-w4096-q16"),
        pytest.param(16, True, None, id="qwen35-causal-full-q16"),
        pytest.param(8, False, 2048, id="dflash2-noncausal-w2048-q8"),
        pytest.param(16, False, None, id="qwen3-noncausal-full-q16"),
    ],
)
def test_draft_long_context(capacity, batch, provider, query, causal, window):
    # Existing model-pair DRAFT rows use Hq/Hkv=32/8, hd128.
    with torch.inference_mode():
        case = LongContextCase(batch, capacity, query, 32, 8, 128, causal, window, provider)
        check_long_context(case)


@pytest.mark.parametrize("capacity", [131072, 262144], ids=lambda c: f"c{c}")
@pytest.mark.parametrize("batch", [1, 2, 3, 4], ids=lambda b: f"b{b}")
@pytest.mark.parametrize(
    "heads,kv_heads,dim",
    [pytest.param(16, 4, 256, id="qwen35-hd256"), pytest.param(32, 8, 128, id="qwen3-hd128")],
)
def test_fixed_query_verify_long_context(capacity, batch, heads, kv_heads, dim):
    # Same target geometries as test_fixed_query_attention's min-chunk rows.
    with torch.inference_mode():
        case = LongContextCase(batch, capacity, 16, heads, kv_heads, dim, True, None, "fixed-query")
        check_long_context(case)
