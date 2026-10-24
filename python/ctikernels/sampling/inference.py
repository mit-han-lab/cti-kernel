# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch

from ._categorical import CategoricalSampling
from ._chain import ChainSampling
from ._dflash import RequestLocalChainSampling
from ._request_local import filter_requests, sample_requests


class RequestLocalCategorical:
    """Sample independent request streams in one batched categorical draw.

    Host seeds and positions are uploaded together; regrouping requests cannot
    change their random streams. There is no RNG state readback or mutable buffer.
    Optional filters materialize top-k-then-top-p probabilities without modifying
    inputs, by K1's exact law (``filter_requests``): exactly k tokens with
    smaller-ID ties, fixed-point top-p mass, bitwise run-to-run reproducible.
    Filtering does not change the RNG stream or consume additional draws.
    """

    def __init__(self) -> None:
        self._sample = sample_requests
        self._filter = filter_requests

    def sample(
        self,
        probabilities: torch.Tensor,
        seeds: tuple[int, ...],
        positions: tuple[int, ...],
        *,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = probabilities.shape[0]
        if len(seeds) != batch_size or len(positions) != batch_size:
            raise ValueError("request-local RNG state must match the probability batch")
        if batch_size == 0:
            return probabilities.new_empty(0, dtype=torch.int32)
        if top_k is not None or top_p is not None:
            probabilities = self._filter(probabilities, top_k, top_p)
        return self._sample(probabilities, seeds, positions)


def topk_by_id(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest of each row of ``logits`` [R, V] (BF16, FP16 or FP32, contiguous) as
    FP32 values and int64 IDs ordered by ascending ID; ties go to the smaller ID and -0 equals
    +0 (returned as +0); NaN ranks above +inf. Equal, bitwise, to packing (value, ID) keys
    and taking their top ``k``. FlashInfer CUB's top-k (``tie_break=1``) selects; a fix-up
    kernel re-chooses the zeros of rows whose boundary lies at zero (CUB ranks +0 above -0).
    No host sync; CUDA-graph capturable after one eager call (FlashInfer's JIT)."""
    if logits.ndim != 2 or not logits.is_contiguous() or not logits.is_cuda:
        raise ValueError("logits must be a contiguous CUDA [R, V] tensor")
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("logits must be BF16, FP16 or FP32")
    rows, width = logits.shape
    if not 0 < k <= width:
        raise ValueError("0 < k <= V")
    if rows == 0:
        empty = logits.new_empty(0, k, dtype=torch.int64)
        return empty.float(), empty
    import flashinfer

    from .triton._topk_by_id import zero_boundary_fixup

    values, ids = flashinfer.top_k(logits, k, sorted=False, deterministic=False, tie_break=1)
    ids = ids.to(torch.int64).contiguous()
    zero_boundary_fixup[(rows,)](
        logits, values.contiguous(), ids, width, K=k, KP=1 << (k - 1).bit_length(), BLOCK=4096,
        num_warps=4,
    )  # fmt: skip
    ids = ids.sort(-1).values
    return logits.gather(1, ids).float() + 0.0, ids


__all__ = [
    "steer_logits_",
    "topk_by_id",
    "CategoricalSampling",
    "ChainSampling",
    "RequestLocalCategorical",
    "RequestLocalChainSampling",
]


_INTS = (torch.int32, torch.int64)


def steer_logits_(
    logits: torch.Tensor,
    rows: torch.Tensor,
    generated: torch.Tensor,
    *,
    min_tokens: torch.Tensor,
    stop_indptr: torch.Tensor,
    stop_ids: torch.Tensor,
    bias_indptr: torch.Tensor,
    bias_ids: torch.Tensor,
    bias_values: torch.Tensor,
) -> None:
    """Sparse logit bias and min_tokens stop masking in place, in vLLM 0.31 V2's order. For
    logits row ``r`` of request ``n = rows[r]``: first ``logits[r, t] += v`` (one FP32
    round-to-nearest add) for each bias entry ``k`` in ``bias_indptr[n]:bias_indptr[n + 1]``
    (``t = bias_ids[k]``, ``v = bias_values[k]``); then, if ``generated[r] < min_tokens[n]``,
    ``logits[r, t] = -inf`` for each ``t`` in ``stop_ids[stop_indptr[n]:stop_indptr[n + 1]]``,
    so a mask wins over a positive bias. Every other element keeps its bits, -0.0 included
    (a dense ``logits + bias`` turns an untouched -0.0 into +0.0).

    ``logits`` FP32 [R, V], unit column stride, non-overlapping rows (e.g. a 2-D view of
    [N, Q, V]); ``rows`` int32 [R]; ``generated`` int32/int64 [R], the outputs before the
    token row ``r`` samples; ``min_tokens`` int32/int64 [N]; ``*_indptr`` int32/int64 [N + 1];
    ``stop_ids``, ``bias_ids`` int32; ``bias_values`` FP32, as long as ``bias_ids``. Caller
    preconditions, never read on the host: ``0 <= rows < N``, non-decreasing indptrs within
    their lists, ids in ``[0, V)``, a request's bias ids unique. CUDA: one Triton launch, no
    host sync or allocation, CUDA-graph capturable (meant to run eagerly, skipped by the
    caller when no row steers). CPU: the torch form of the same law."""
    _check_steering(
        logits, rows, generated, min_tokens, stop_indptr, stop_ids, bias_indptr, bias_ids,
        bias_values,
    )  # fmt: skip
    if logits.shape[0] == 0:
        return
    args = (rows, generated, min_tokens, stop_indptr, stop_ids, bias_indptr, bias_ids, bias_values)
    if logits.is_cuda:
        from .triton._steer import launch

        launch(logits, *args)
    else:
        _steer_torch(logits, *args)


def _check_steering(
    logits, rows, generated, min_tokens, stop_indptr, stop_ids, bias_indptr, bias_ids, bias_values
):  # fmt: skip
    if logits.ndim != 2 or logits.dtype != torch.float32 or logits.stride(-1) != 1:
        raise ValueError("logits must be FP32 [R, V] with unit column stride")
    r, v = logits.shape
    if r > 1 and logits.stride(0) < v:
        raise ValueError("logits rows must not overlap")
    n = min_tokens.shape[0] if min_tokens.ndim == 1 else -1
    expected = (
        ("rows", rows, (torch.int32,), (r,)),
        ("generated", generated, _INTS, (r,)),
        ("min_tokens", min_tokens, _INTS, (n,)),
        ("stop_indptr", stop_indptr, _INTS, (n + 1,)),
        ("bias_indptr", bias_indptr, _INTS, (n + 1,)),
        ("stop_ids", stop_ids, (torch.int32,), (stop_ids.numel(),)),
        ("bias_ids", bias_ids, (torch.int32,), (bias_ids.numel(),)),
        ("bias_values", bias_values, (torch.float32,), (bias_ids.numel(),)),
    )
    for name, tensor, dtypes, shape in expected:
        if tensor.dtype not in dtypes or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape} of {dtypes}")
        if tensor.device != logits.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the logits' device")


def _csr_entries(indptr, rows):
    """(logits row, entry index) of every entry of every row's request."""
    lo = indptr[rows].long()
    counts = indptr[rows + 1].long() - lo
    row = torch.repeat_interleave(torch.arange(rows.numel(), device=rows.device), counts)
    first = (counts.cumsum(0) - counts)[row]
    return row, lo[row] + torch.arange(row.numel(), device=rows.device) - first


def _steer_torch(
    logits, rows, generated, min_tokens, stop_indptr, stop_ids, bias_indptr, bias_ids, bias_values
):  # fmt: skip
    rows = rows.long()
    row, entry = _csr_entries(bias_indptr, rows)
    column = bias_ids[entry].long()
    logits[row, column] = logits[row, column] + bias_values[entry]
    row, entry = _csr_entries(stop_indptr, rows)
    short = (generated < min_tokens[rows])[row]
    logits[row[short], stop_ids[entry[short]].long()] = float("-inf")
