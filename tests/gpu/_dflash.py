# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Engine-like stable storage for DRAFT GPU qualification."""

import torch

from ctikernels.attention.inference import DraftAttention

from .._dflash_reference import draft_attention_reference

SENTINEL = -1234567


class Fixture:
    def __init__(
        self,
        b,
        q,
        p,
        hq,
        hk,
        causal,
        window,
        c=4160,
        *,
        provider=None,
        head_dim=128,
        page_indices=True,
        query_row_stride=None,
    ):
        self.b, self.q, self.p, self.c = b, q, p, c
        self.causal, self.window = causal, window
        self.pages_per_row = (c + p - 1) // p
        self.slots = torch.empty(b, dtype=torch.int32, device="cuda")
        self.positions = torch.empty(b * q, dtype=torch.int64, device="cuda")
        self.token_ptr = torch.empty(b + 1, dtype=torch.int32, device="cuda")
        self.token_idx = torch.empty(b * c, dtype=torch.int32, device="cuda")
        self.query_ptr = torch.empty(b + 1, dtype=torch.int32, device="cuda")
        self.page_ptr = torch.empty(b + 1, dtype=torch.int32, device="cuda")
        self.page_idx = (
            torch.full((b * self.pages_per_row,), SENTINEL, dtype=torch.int32, device="cuda")
            if page_indices
            else None
        )
        self.last = torch.empty(b, dtype=torch.int32, device="cuda")
        sizes = DraftAttention.workspace_size(
            batch_size=b,
            query_length=q,
            max_kv_tokens=c,
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=head_dim,
            page_size=p,
            dtype=torch.bfloat16,
            causal=causal,
            window=window,
            provider=provider,
            device=self.slots.device,
        )
        self.workspaces = [torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes]
        self.op = DraftAttention(
            *self.workspaces,
            request_indices=self.slots,
            positions=self.positions,
            token_indptr=self.token_ptr,
            token_indices=self.token_idx,
            query_indptr=self.query_ptr,
            page_indptr=self.page_ptr,
            page_indices=self.page_idx,
            last_page_lengths=self.last,
            query_length=q,
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=head_dim,
            page_size=p,
            dtype=torch.bfloat16,
            causal=causal,
            window=window,
            provider=provider,
            query_row_stride=query_row_stride,
        )
        generator = torch.Generator(device="cuda").manual_seed(1739)
        self.query = torch.randn(
            (b * q, hq, head_dim), dtype=torch.bfloat16, device="cuda", generator=generator
        )
        self.k = torch.randn(
            (2 * b * self.pages_per_row, p, hk, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        self.v = torch.randn(self.k.shape, dtype=torch.bfloat16, device="cuda", generator=generator)
        # Fragmentation gaps are never legal addresses.
        self.k[::2].fill_(float("nan"))
        self.v[::2].fill_(float("nan"))
        self.out_backing = torch.full(
            (self.query.numel() + 256,), 123.0, dtype=torch.bfloat16, device="cuda"
        )
        self.out = self.out_backing[128:-128].view_as(self.query)

    def stage(self, contexts, slots, turn=0, base=0):
        lengths = [
            context + self.q if slot != -1 else 0
            for context, slot in zip(contexts, slots, strict=True)
        ]
        token_ptr, page_ptr, query_ptr = [0], [0], [0]
        tokens, pages, positions, last = [], [], [], []
        for row, (length, slot) in enumerate(zip(lengths, slots, strict=True)):
            count = (length + self.p - 1) // self.p
            physical = (
                torch.arange(self.pages_per_row).flip(0).roll(turn) + row * self.pages_per_row
            ) * 2 + 1
            chosen = physical[:count]
            pages.extend(chosen.tolist())
            tokens.extend(
                (chosen[:, None] * self.p + torch.arange(self.p)[None, :])
                .flatten()[:length]
                .tolist()
            )
            token_ptr.append(token_ptr[-1] + length)
            page_ptr.append(page_ptr[-1] + count)
            query_ptr.append(query_ptr[-1] + (self.q if slot != -1 else 0))
            positions.extend(
                range(base + length - self.q, base + length) if slot != -1 else [-(2**63)] * self.q
            )
            last.append((length - 1) % self.p + 1 if slot != -1 else 0)
        self.cpu_slots = torch.tensor(slots, dtype=torch.int32)
        self.cpu_positions = torch.tensor(positions, dtype=torch.int64)
        self.cpu_token_ptr = torch.tensor(token_ptr, dtype=torch.int32)
        self.cpu_token_idx = torch.full((self.b * self.c,), SENTINEL, dtype=torch.int32)
        self.cpu_token_idx[: len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        self.expected_query_ptr = torch.tensor(query_ptr, dtype=torch.int32)
        self.expected_page_ptr = torch.tensor(page_ptr, dtype=torch.int32)
        if self.page_idx is not None:
            self.expected_pages = torch.full_like(self.page_idx.cpu(), SENTINEL)
            self.expected_pages[: len(pages)] = torch.tensor(pages, dtype=torch.int32)
        else:
            self.expected_pages = None
        self.expected_last = torch.tensor(last, dtype=torch.int32)
        self.slots.copy_(self.cpu_slots)
        self.positions.copy_(self.cpu_positions)
        self.token_ptr.copy_(self.cpu_token_ptr)
        self.token_idx.copy_(self.cpu_token_idx)
        if self.page_idx is not None:
            self.page_idx.fill_(SENTINEL)
        self.out.fill_(float("nan"))
        generator = torch.Generator(device="cuda").manual_seed(1739 + turn)
        self.query.normal_(generator=generator)
        for row, slot in enumerate(slots):
            if slot == -1:
                self.query[row * self.q : (row + 1) * self.q].fill_(float("nan"))

    def run(self):
        self.op.prepare()
        return self.op.run(q=self.query, paged_kv_cache=(self.k, self.v), out=self.out)

    def oracle(self):
        return draft_attention_reference(
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
        for actual, expected in (
            (self.query_ptr, self.expected_query_ptr),
            (self.page_ptr, self.expected_page_ptr),
            (self.page_idx, self.expected_pages),
            (self.last, self.expected_last),
            (self.token_idx, self.cpu_token_idx),
            (self.token_ptr, self.cpu_token_ptr),
            (self.positions, self.cpu_positions),
            (self.slots, self.cpu_slots),
        ):
            if actual is not None:
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        assert (self.out_backing[:128] == 123).all()
        assert (self.out_backing[-128:] == 123).all()
