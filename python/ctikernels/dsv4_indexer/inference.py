# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DeepSeek-V4 (Flash) KV compressor and sparse-attention indexer, inference only.

**Pages.** Every compressed pool reuses the request's 256-token page ids: entry ``e`` of a
request lives at row ``pages[slot, e // rows] * rows + e % rows`` with ``rows = 256 / ratio``.
- Latent entries (``dim = 512``, CSA ratio 4 and HCA ratio 128) use the latent page ABI of
  ``ctikernels.dsv4_attention``: per row 448 E4M3 (UE8M0 scale per 64) + 64 BF16 rotary
  channels = 576 bytes, then 8 footer bytes per row (7 exponents + 0) after the page's data,
  the page padded to a multiple of 576 bytes (``latent_page_bytes(rows)``).
- Index keys (``dim = 128``, ratio 4): 64-row pages of 4352 bytes, 64 bytes of E2M1 pairs per
  row (even channel in the low nibble) of the Hadamard-rotated key, then 4 UE8M0 bytes per row
  (one per 32 channels). This is DeepSeek-V4's own indexer numerics (FP4 with UE8M0 per 32).

**State.** A request's open pooling windows live in caller-owned rings indexed by the request's
state ``slot``: ``state_kv`` and ``state_score`` FP32 ``[slots, ring, coff, dim]``
(``state_shape``), absolute token ``p`` at ``p % ring`` (CSA: ``ring = 8``, ``coff = 2``; HCA:
``ring = 128``, ``coff = 1``). There is no length counter: a window's positions before the
call's ``start`` are read from the ring, later ones from the call's input, negative ones are
masked. A request starting at position 0 reads no ring row, so a new request needs no clearing
(a slot left dirty or NaN by an earlier request is never read). A continuation reads only ring
rows its own earlier chunks wrote: feed a slot's chunks in order, and snapshot/restore
``state_kv`` and ``state_score`` together with the request.

**Rounding.** Latent entries are written as ``ctikernels.attention.inference.dsv4_store_kv``
writes rows (K-d contract): the FP32 pooled entry's RMSNorm (eps 1e-6) times the BF16 weight and
the RoPE at the window start in FP32, then the page pack (FP8 non-rotary, BF16 rotary). Index keys
keep DeepSeek-V4's BF16 casts (pooled, normed, rotated), then Hadamard (BF16) and FP4.

**CUDA graphs.** No host reads; per-call scratch is preallocated at construction for
``max_batch`` requests and ``max_tokens`` tokens. Triton compiles per kernel variant on first
use: run one decode and one prefill call of each shape class eagerly before capture. Calls on
one instance must be stream-ordered (the HCA decode election counter lives in the instance).
"""

from __future__ import annotations

import torch

PAGE_TOKENS = 256
INDEX_ROW_BYTES = 68
INDEX_PAGE_BYTES = 4352
LATENT_ROW_BYTES = 584


def latent_page_bytes(rows: int) -> int:
    """Bytes of a latent page of ``rows`` rows: data, footers, padded to 576."""
    return -(-rows * LATENT_ROW_BYTES // 576) * 576


def _check(name, x, shape, dtype, device):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if x.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, not {x.dtype}")
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, not {tuple(x.shape)}")
    if x.device != device or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")
    if x.requires_grad:
        raise ValueError(f"{name} must not require grad (inference only)")


class DSV4Compressor:
    """One layer kind's compressor: ``ratio`` 4 (CSA, overlapping windows) with ``dim`` 512
    (latent) or 128 (index keys), or ``ratio`` 128 (HCA) with ``dim`` 512.

    ``op(kv, score, ...)`` takes the projected tokens of a ragged batch: ``kv`` and ``score``
    FP32 ``[T, coff * dim]`` (``x @ wkv.T`` and ``x @ wgate.T``; the position bias ``ape`` is
    added inside), request ``b`` owning rows ``offsets[b]:offsets[b + 1]`` at absolute
    positions ``starts[b] + i``. It writes every entry the batch completes to ``cache`` and
    returns ``rows``: int64 ``[B, capacity]``, the cache row of each completed entry
    in order then ``-1``. Window positions before ``starts[b]`` come from the slot's ring and
    the rest from this call (positions below 0 are masked): a request starting at 0 reads no
    ring row, and a continuation reads only the ring rows its own earlier chunks wrote, so the
    caller feeds a slot's chunks in order and snapshots/restores the ring with the request.
    ``request_ids`` (int32) must be distinct apart from padding: id ``-1`` marks a padding row
    (as in ``ctikernels.gdn``), which reads no state or page and writes nothing but rows ``-1``,
    whatever its offsets and start; ``pages`` int32 ``[slots, P]``; ``rope`` FP32 ``[positions, 64]`` the rotary
    table of the trailing 64 channels (adjacent pairs): 32 cosines, then 32 sines per position. The returned tensors are views of
    instance buffers, overwritten by the next call.
    """

    def __init__(self, *, ratio: int, dim: int, max_tokens: int, max_batch: int, device):
        if (ratio, dim) not in ((4, 512), (4, 128), (128, 512)):
            raise ValueError("expected (ratio, dim) in (4, 512), (4, 128), (128, 512)")
        if min(max_tokens, max_batch) < 1:
            raise ValueError("max_tokens and max_batch must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DSV4Compressor runs on a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.ratio, self.dim = ratio, dim
        self.coff, self.ring = (2, 8) if ratio == 4 else (1, 128)
        self.rows_per_page = PAGE_TOKENS // ratio
        self.page_bytes = INDEX_PAGE_BYTES if dim == 128 else latent_page_bytes(self.rows_per_page)
        self.max_tokens, self.max_batch = max_tokens, max_batch
        self.capacity = -(-max_tokens // ratio) + 1
        self.slots = torch.full(
            (max_batch, self.capacity), -1, dtype=torch.int64, device=self.device
        )
        pooled = (max_batch, self.capacity, dim) if ratio == 128 else (1,)
        self.pooled = torch.empty(pooled, dtype=torch.float32, device=self.device)
        self.tickets = torch.zeros(max_batch, dtype=torch.int32, device=self.device)

    def state_shape(self, slots: int) -> tuple[int, int, int, int]:
        """Shape of ``state_kv`` / ``state_score`` for ``slots`` request slots."""
        return (slots, self.ring, self.coff, self.dim)

    @torch.no_grad()
    def __call__(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        *,
        state_kv: torch.Tensor,
        state_score: torch.Tensor,
        request_ids: torch.Tensor,
        offsets: torch.Tensor,
        starts: torch.Tensor,
        pages: torch.Tensor,
        cache: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        t, b, dev = kv.shape[0], request_ids.numel(), self.device
        width = self.coff * self.dim
        if t > self.max_tokens or b > self.max_batch:
            raise ValueError("max_tokens or max_batch exceeded")
        slots = state_kv.shape[0]
        for name, x, shape, dtype in (
            ("kv", kv, (t, width), torch.float32),
            ("score", score, (t, width), torch.float32),
            ("state_kv", state_kv, self.state_shape(slots), torch.float32),
            ("state_score", state_score, self.state_shape(slots), torch.float32),
            ("request_ids", request_ids, (b,), torch.int32),
            ("offsets", offsets, (b + 1,), torch.int32),
            ("starts", starts, (b,), torch.int64),
            ("pages", pages, (slots, pages.shape[-1]), torch.int32),
            ("cache", cache, (cache.shape[0], self.page_bytes), torch.uint8),
            ("ape", ape, (self.ratio, width), torch.float32),
            ("norm_weight", norm_weight, (self.dim,), torch.bfloat16),
            ("rope", rope, (rope.shape[0], 64), torch.float32),
        ):
            _check(name, x, shape, dtype, dev)
        if b:
            from .triton import _compressor

            _compressor.run(self, kv, score, state_kv, state_score, request_ids, offsets, starts,
                            pages, cache, ape, norm_weight, rope)  # fmt: skip
        return self.slots[:b]


class DSV4Indexer:
    """CSA indexer logits over the FP4 index pages, for the top-k of
    ``ctikernels.attention.inference.dsv4_topk_rows``.

    ``op(q, weights, ...)`` takes ``q`` BF16 ``[T, 64, 128]`` (``q_latent @ wq_b.T``, before
    RoPE), ``weights`` BF16 ``[T, 64]`` (``x @ weights_proj.T``), ``positions`` int64 ``[T]``,
    ``request_ids`` int32 ``[T]`` (each query's state slot), ``pages`` int32 ``[slots, P]``
    with ``P >= op.table_columns``, ``cache`` uint8 ``[pages, 4352]`` and the rotary table
    ``rope`` (as the compressor's). It fills ``out`` FP32 ``[T, >= op.columns]`` (unit column
    stride) and returns ``out[:, :max_entries]``: ``[t, j]`` is entry ``j``'s score for query
    ``t`` for ``j < min((positions[t] + 1) // 4, max_entries)``; later columns are unspecified
    (select with those lengths).

    Scores, as the DeepSeek-V4 reference's math in FP32: the query is rotated, rounded to
    BF16, Hadamard-transformed (BF16) and FP4 round-tripped (UE8M0 per 32); each head's dot with
    the decoded FP4 key (exact products, FP32 sums), ``relu(dot) * weight * 128^-0.5 * 64^-0.5``
    summed over heads in FP32. This is more precise than the official kernel's BF16 casts of
    dot, weight, product and sum. Ties go to the lower entry in the selection (FlashInfer
    ``top_k_page_table_transform(..., tie_break=1, deterministic=True)``).

    Providers (``op.provider``): ``"triton"`` (default; our block-scaled ``tl.dot_scaled`` kernel, 128 or 256 entries per program, key loads overlapping the
    query transform by programmatic dependent launch on 9.0+), or ``"flashinfer"`` on request:
    FlashInfer's ``fp4_paged_mqa_logits`` (persistent block-scaled tcgen05 MMA) on compute
    capability 10.x. Both are bitwise run to run;
    they differ only in FP32 summation order. Three (FlashInfer) or two (Triton) launches, no
    host reads, scratch for ``max_queries`` preallocated. Run once eagerly before graph capture
    (Triton and the CuTe DSL kernel compile on first use).
    """

    def __init__(self, *, max_queries: int, max_entries: int, device, provider: str = "auto"):
        if min(max_queries, max_entries) < 1:
            raise ValueError("max_queries and max_entries must be positive")
        if provider not in ("auto", "triton", "flashinfer"):
            raise ValueError(f"unknown indexer provider {provider!r}")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DSV4Indexer runs on a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        from .flashinfer import _logits

        fi = _logits.available(self.device)
        if provider == "flashinfer" and not fi:
            raise RuntimeError(
                "the flashinfer indexer needs compute capability 10.x and FlashInfer"
            )
        # GB300 end to end (transform + logits, DSV4 decode B=1-4, 8k-128k context): Triton
        # 4.7-9.9 us vs FlashInfer 8.1-10.5 us (its schedule launch, no PDL), so auto is Triton.
        self.provider = "triton" if provider == "auto" else provider
        self.max_queries, self.max_entries = max_queries, max_entries
        # FlashInfer writes whole 256-entry splits and reads 128-entry blocks of the table.
        self.columns = -(-max_entries // 256) * 256
        self.table_columns = -(-max_entries // 128) * 2
        dev = self.device
        # The transformed queries (E2M1 pairs and UE8M0 bytes per head), scaled FP32 weights,
        # visible entry counts and, for FlashInfer, the gathered block table.
        self._codes = torch.empty((max_queries, 64, 64), dtype=torch.uint8, device=dev)
        self._scales = torch.empty((max_queries, 64, 4), dtype=torch.uint8, device=dev)
        self._weights = torch.empty((max_queries, 64), dtype=torch.float32, device=dev)
        self._lens = torch.empty(max_queries, dtype=torch.int32, device=dev)
        fi_table = (max_queries, self.table_columns) if self.provider == "flashinfer" else (1, 1)
        self._table = torch.empty(fi_table, dtype=torch.int32, device=dev)
        self._schedule = _logits.schedule_buffer(dev) if self.provider == "flashinfer" else None
        self._pdl = torch.cuda.get_device_capability(dev) >= (9, 0)
        self._sms = torch.cuda.get_device_properties(dev).multi_processor_count

    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        *,
        positions: torch.Tensor,
        request_ids: torch.Tensor,
        pages: torch.Tensor,
        cache: torch.Tensor,
        rope: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        t, dev = q.shape[0], self.device
        if t > self.max_queries:
            raise ValueError("max_queries exceeded")
        for name, x, shape, dtype in (
            ("q", q, (t, 64, 128), torch.bfloat16),
            ("weights", weights, (t, 64), torch.bfloat16),
            ("positions", positions, (t,), torch.int64),
            ("request_ids", request_ids, (t,), torch.int32),
            ("pages", pages, (pages.shape[0], pages.shape[-1]), torch.int32),
            ("cache", cache, (cache.shape[0], INDEX_PAGE_BYTES), torch.uint8),
            ("rope", rope, (rope.shape[0], 64), torch.float32),
        ):
            _check(name, x, shape, dtype, dev)
        if pages.shape[1] < self.table_columns:
            raise ValueError(f"pages must have at least {self.table_columns} columns")
        if not isinstance(out, torch.Tensor) or out.dtype != torch.float32:
            raise TypeError("out must be an FP32 tensor")
        if out.ndim != 2 or out.shape[0] != t or out.shape[1] < self.columns:
            raise ValueError(f"out must be [{t}, >= {self.columns}]")
        if out.device != dev or out.stride(-1) != 1:
            raise ValueError(f"out must be on {dev} with unit column stride")
        if t:
            from .triton import _indexer

            fi = self.provider == "flashinfer"
            _indexer.prepare(self, q, weights, positions, request_ids, pages, rope, fi)
            if fi:
                from .flashinfer import _logits

                _logits.run(self, cache, out)
            else:
                _indexer.run(self, request_ids, pages, cache, out)
        return out[:, : self.max_entries]


__all__ = [
    "DSV4Compressor",
    "DSV4Indexer",
    "INDEX_PAGE_BYTES",
    "INDEX_ROW_BYTES",
    "LATENT_ROW_BYTES",
    "PAGE_TOKENS",
    "latent_page_bytes",
]
