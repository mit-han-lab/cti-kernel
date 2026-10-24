# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Borrowed-output DN3 logprobs.

Rank semantics follow vLLM 0.31.0 V2 count(value >= selected); no code copied.
Processed filtering follows this package's S1 reference law.
"""

import math

import torch
import torch.nn.functional as F


def _probability_sum(values):
    """Batched S1 4096-token butterfly trees, sequential chunk accumulation."""
    n, vocab = values.shape
    padded = F.pad(values, (0, (-vocab) % 4096))
    tiles = padded.reshape(n, -1, 2, 512, 4)
    lanes = torch.zeros_like(tiles[:, :, 0, :, 0])
    for tile in range(2):
        for component in range(4):
            lanes = lanes + tiles[:, :, tile, :, component]
    warps = lanes.reshape(n, -1, 16, 32)
    indices = torch.arange(32, device=values.device)
    for offset in (1, 2, 4, 8, 16):
        warps = warps + warps[..., indices ^ offset]
    totals = F.pad(warps[..., 0], (0, 16))
    for offset in (1, 2, 4, 8, 16):
        totals = totals + totals[..., indices ^ offset]
    total = torch.zeros((n, 1), dtype=torch.float32, device=values.device)
    for chunk in range(totals.shape[1]):
        total = total + totals[:, chunk, :1]
    return total


def _processed(logits, temperatures, top_k, top_p, min_p):
    """FP32 S1 support; preserve log-space tails without exponent underflow."""
    vocab = logits.shape[1]
    empty = torch.isneginf(logits.amax(-1, keepdim=True))
    # Normalize a harmless row, then restore empty support. In particular,
    # no NaN may reach the floating-mass to integer conversion below.
    logits = torch.where(empty, 0, logits)
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    positions = torch.arange(vocab, device=logits.device)[None, :].expand_as(order)
    raw_rank = torch.empty_like(order).scatter_(1, order, positions)
    keep = raw_rank < top_k[:, None]
    maximum = logits.amax(-1, keepdim=True)
    t = torch.where(temperatures == 0, 1, temperatures)[:, None]
    difference = logits - maximum
    z = torch.where(
        torch.isneginf(difference) & torch.isfinite(logits),
        logits / t - maximum / t,
        difference / t,
    )
    weights = torch.where(keep, z.exp(), 0)
    # Match S1's special top-k boundary-tie summation as well as its full row tree.
    tau = logits.gather(1, order.gather(1, (top_k.long() - 1)[:, None]))
    above = logits > tau
    ties = top_k[:, None] - above.sum(-1, keepdim=True)
    tau_z = z.gather(1, order.gather(1, (top_k.long() - 1)[:, None]))
    total = torch.where(
        top_k[:, None] < vocab,
        _probability_sum(torch.where(above, weights, 0)) + ties.float() * tau_z.exp(),
        _probability_sum(weights),
    )
    probs = weights / total
    p_order = torch.argsort(probs, dim=-1, descending=True, stable=True)
    # Multiplication by a power of two is exact in FP64 for every FP32 input;
    # conversion to int64 floors nonnegative masses, including subnormals.
    masses = (probs.gather(1, p_order).double() * (2**62)).long()
    preceding = masses.cumsum(-1) - masses
    threshold = torch.ceil(top_p.double() * (2**62)).long()[:, None]
    prefix = preceding < threshold
    prefix = prefix | (top_p[:, None] == 1)
    p_keep = torch.empty_like(prefix).scatter_(1, p_order, prefix)
    keep = keep & p_keep
    # S1 renormalizes using the retained integer mass before applying min-p.
    scale = torch.where(prefix, masses, 0).sum(-1, keepdim=True).float() * (2.0**-62)
    probs = torch.where(top_p[:, None] == 1, probs, probs / scale)
    probs = torch.where(keep, probs, 0)
    if min_p is not None:
        keep = keep & (probs >= min_p[:, None] * probs.amax(-1, keepdim=True))
    greedy = positions == logits.argmax(-1, keepdim=True)
    keep = torch.where(temperatures[:, None] == 0, greedy, keep)
    z = torch.where(temperatures[:, None] == 0, 0, z)
    z = torch.where(keep, z, -torch.inf)
    z = z - z.amax(-1, keepdim=True)
    logprobs = z - z.exp().sum(-1, keepdim=True).log()
    logprobs = torch.where(empty, -torch.inf, logprobs)
    keep = keep & torch.isfinite(logprobs)
    probs = torch.where(temperatures[:, None] == 0, 1, probs)
    return logprobs, torch.where(keep, probs, 0)


def _order_key(probs, logprobs):
    """One total processed order as a unique-per-tie int64 key.

    Descending FP32 probability, then descending processed logprob (dense rank,
    so equal logprobs share a code), then the caller's stable lower-ID rule.
    Nonnegative FP32 bit patterns are monotone, so the high word orders p.
    """
    vocab = logprobs.shape[1]
    ordered, order = logprobs.sort(dim=-1, descending=True)
    starts = F.pad(ordered[:, 1:] != ordered[:, :-1], (1, 0), value=True)
    dense = starts.cumsum(-1)
    codes = torch.empty_like(dense).scatter_(1, order, vocab - dense)
    bits = torch.where(probs > 0, probs, 0).contiguous().view(torch.int32).long()
    return (bits << 32) | codes


class TopLogprobs:
    """Capturable torch provider for raw or S1-processed requested rows.

    Values/ranges are caller admission preconditions, never read back by run().
    Penalties must already be applied to logits. Outputs must not alias inputs
    or each other. Torch owns temporary allocations; borrowed scratch is empty.
    """

    MAX_VOCAB = 2**31 - 1
    # One full-width selection and tie scan plus small stable candidate sorts.
    # Above 1024, use one stable full sort; this is a policy, not a measured crossover.
    SORT_THRESHOLD = 1024

    def __init__(self, device, vocab_size: int, max_top_k: int, *, row_chunk: int = 64):
        for name, value, low, high in (
            ("vocab_size", vocab_size, 1, self.MAX_VOCAB),
            ("max_top_k", max_top_k, 0, vocab_size),
            ("row_chunk", row_chunk, 1, self.MAX_VOCAB),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        self.vocab_size, self.max_top_k, self.row_chunk = vocab_size, max_top_k, row_chunk
        # Construct on the operation device before capture. Run only slices
        # these constants, including for the final, shorter chunk.
        self._scan_dtype = torch.int32 if row_chunk * vocab_size < 2**31 else torch.int64
        self._columns = torch.arange(max_top_k, dtype=torch.int32, device=self.device)[None, :]
        self._targets = self._columns.to(self._scan_dtype) + 1
        self._rows = torch.arange(row_chunk, dtype=torch.int32, device=self.device)
        self._row_offsets = (self._rows.to(torch.int64) * vocab_size)[:, None]
        self._row_last = self._row_offsets + vocab_size - 1
        self._previous_ends = (self._row_offsets[:, 0] - 1).clamp_min_(0)
        self._has_previous = (self._rows != 0).to(self._scan_dtype)
        # The out= overload of where requires tensors for both alternatives.
        self._zero_logprob = torch.zeros((), dtype=torch.float32, device=self.device)
        self._zero_rank = torch.zeros((), dtype=torch.int32, device=self.device)
        self._padding_id = torch.full((), -1, dtype=torch.int32, device=self.device)

    def _ordered_ids(self, values):
        """Exact descending value / ascending ID, including -inf boundary ties."""
        capacity = self.max_top_k
        if capacity > self.SORT_THRESHOLD:
            return torch.argsort(values, dim=-1, descending=True, stable=True)[:, :capacity]
        _, ids = torch.topk(values, capacity, dim=-1, sorted=False)
        # All strictly greater values are present. Sorting by ID first makes
        # the stable value sort exact, and supplies the boundary without amin.
        ids = ids.sort(dim=-1).values
        vals, order = values.gather(1, ids).sort(dim=-1, descending=True, stable=True)
        ids = ids.gather(1, order)
        boundary = vals[:, -1:]
        above = (vals > boundary).sum(-1, keepdim=True, dtype=self._scan_dtype)
        flat = (values == boundary).reshape(-1).cumsum(0, dtype=self._scan_dtype)
        # Search directly for the tie needed by each output column. Strictly
        # greater slots use a dummy first-tie target and retain their topk ID.
        targets = (self._targets - above).clamp_min_(1)
        n = values.shape[0]
        if n > 1:
            base = flat.index_select(0, self._previous_ends[:n])
            base.mul_(self._has_previous[:n])
            targets.add_(base[:, None])
        ties = torch.searchsorted(flat, targets)
        # NaN/garbage padding can have no ties; searches may leave that row.
        # Clamp to both row bounds BEFORE subtracting the flattened offset.
        if n > 1:
            offsets = self._row_offsets[:n]
            ties.clamp_(min=offsets, max=self._row_last[:n]).sub_(offsets)
        else:
            ties.clamp_(0, self.vocab_size - 1)
        return torch.where(self._columns < above, ids, ties)

    @property
    def provider(self) -> str:
        return "triton" if self.device.type == "cuda" else "torch"

    def workspace_size(self, row_capacity) -> int:
        if (
            isinstance(row_capacity, bool)
            or not isinstance(row_capacity, int)
            or not 0 <= row_capacity <= self.MAX_VOCAB
        ):
            raise ValueError("row_capacity must be an integer in [0, 2**31-1]")
        return 0

    def _check(self, name, tensor, dtype, shape, *, rows=False):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in dtype:
            raise TypeError(f"{name} must have dtype in {dtype}")
        if tensor.requires_grad:
            raise ValueError(f"{name} requires grad")
        if tensor.device != self.device or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has wrong device or shape; expected {shape} on {self.device}")
        if rows:
            if tensor.stride(1) != 1 or tensor.stride(0) <= 0:
                raise ValueError(f"{name} requires unit inner stride and positive row stride")
        elif not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def run(
        self,
        logits,
        *,
        row_indices,
        num_rows,
        selected_ids,
        top_k,
        mode="raw",
        values="logprobs",
        floor=None,
        temperatures=None,
        filter_top_k=None,
        top_p=None,
        min_p=None,
        probabilities=None,
        workspace=None,
        out_selected_logprobs,
        out_ranks,
        out_top_ids,
        out_top_logprobs,
    ) -> None:
        """Write all output slots; device counts/indices/parameters may change on replay."""
        if not isinstance(mode, str) or mode not in ("raw", "processed"):
            raise ValueError("mode must be raw or processed")
        if not isinstance(values, str) or values not in ("logprobs", "logits"):
            raise ValueError("values must be logprobs or logits")
        if floor is not None and (not isinstance(floor, float) or not math.isfinite(floor)):
            raise ValueError("floor must be a finite Python float or None")
        for name, tensor, ndim in (("logits", logits, 2), ("row_indices", row_indices, 1)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")
        n, vocab = row_indices.numel(), self.vocab_size
        self.workspace_size(n)
        self._check(
            "logits",
            logits,
            (torch.bfloat16, torch.float16, torch.float32),
            (logits.shape[0], vocab),
            rows=True,
        )
        if n and logits.shape[0] == 0:
            raise ValueError("logits must have at least one storage row for a nonempty capacity")
        specs = [
            ("row_indices", row_indices, torch.int32, (n,)),
            ("num_rows", num_rows, torch.int32, (1,)),
            ("selected_ids", selected_ids, torch.int32, (n,)),
            ("top_k", top_k, torch.int32, (n,)),
            ("out_selected_logprobs", out_selected_logprobs, torch.float32, (n,)),
            ("out_ranks", out_ranks, torch.int32, (n,)),
            ("out_top_ids", out_top_ids, torch.int32, (n, self.max_top_k)),
            ("out_top_logprobs", out_top_logprobs, torch.float32, (n, self.max_top_k)),
        ]
        if mode == "processed":
            for name, value in (
                ("temperatures", temperatures),
                ("filter_top_k", filter_top_k),
                ("top_p", top_p),
            ):
                if value is None:
                    raise ValueError(f"{name} is required for processed logits")
        for name, value, dtype in (
            ("temperatures", temperatures, torch.float32),
            ("filter_top_k", filter_top_k, torch.int32),
            ("top_p", top_p, torch.float32),
            ("min_p", min_p, torch.float32),
        ):
            if value is not None:
                specs.append((name, value, dtype, (n,)))
        if probabilities is not None:
            if not isinstance(probabilities, torch.Tensor):
                raise TypeError("probabilities must be a tensor")
            if probabilities.ndim != 2 or (n and probabilities.shape[0] == 0):
                raise ValueError("probabilities must have shape [R,V] with storage rows")
            self._check(
                "probabilities",
                probabilities,
                (torch.float32,),
                (probabilities.shape[0], vocab),
                rows=True,
            )
        if workspace is not None:
            if not isinstance(workspace, torch.Tensor):
                raise TypeError("workspace must be a tensor")
            specs.append(("workspace", workspace, torch.uint8, (workspace.numel(),)))
        for name, tensor, dtype, shape in specs:
            self._check(name, tensor, (dtype,), shape)
        # Metadata-only overlap checks include strided storage spans.
        tensors = [logits, *[s[1] for s in specs]]
        if probabilities is not None:
            tensors.append(probabilities)
        for dst in (out_selected_logprobs, out_ranks, out_top_ids, out_top_logprobs):
            if not dst.numel():
                continue
            lo, hi = dst.data_ptr(), dst.data_ptr() + dst.numel() * dst.element_size()
            if sum(src is dst for src in tensors) != 1:
                raise ValueError("outputs must not alias another argument")
            for src in tensors:
                if src is dst or not src.numel():
                    continue
                extent = 1 + sum(
                    (size - 1) * stride
                    for size, stride in zip(src.shape, src.stride(), strict=True)
                )
                if lo < src.data_ptr() + extent * src.element_size() and src.data_ptr() < hi:
                    raise ValueError("outputs must not alias another argument")
        if self.device.type == "cuda":
            from .triton import _top_logprobs, _top_logprobs_processed

            outputs = (out_selected_logprobs, out_ranks, out_top_ids, out_top_logprobs)
            for start in range(0, n, self.row_chunk):
                sl = slice(start, min(n, start + self.row_chunk))
                rows = (row_indices[sl], num_rows, start, selected_ids[sl], top_k[sl])
                chunk = tuple(tensor[sl] for tensor in outputs)
                if mode == "raw":
                    _top_logprobs.run_raw(logits, *rows, values, floor, chunk)
                else:
                    _top_logprobs_processed.run_processed(
                        logits,
                        *rows,
                        values,
                        floor,
                        temperatures[sl],
                        filter_top_k[sl],
                        top_p[sl],
                        None if min_p is None else min_p[sl],
                        probabilities,
                        chunk,
                    )
            return
        for start in range(0, n, self.row_chunk):
            end = min(n, start + self.row_chunk)
            sl = slice(start, end)
            active = (
                self._rows[: end - start] < (num_rows if start == 0 else num_rows - start)
            ) & (selected_ids[sl] != -1)
            rows = row_indices[sl].clamp(0, logits.shape[0] - 1)
            selected = selected_ids[sl].clamp(0, vocab - 1).long()[:, None]
            gathered = logits.index_select(0, rows)
            empty = torch.isneginf(gathered.amax(-1, keepdim=True))
            scores = gathered
            if mode == "raw":
                logprobs = F.log_softmax(
                    torch.where(empty, 0, gathered), dim=-1, dtype=torch.float32
                )
                logprobs = torch.where(empty, -torch.inf, logprobs)
            else:
                mp = None if min_p is None else torch.where(active, min_p[sl], 0)
                # Only B converts floating masses to integers. Padding may
                # read garbage from any storage row, so sanitize first.
                gathered = torch.where(active[:, None], gathered.float(), 0)
                logprobs, probs = _processed(
                    gathered,
                    torch.where(active, temperatures[sl], 1),
                    torch.where(active, filter_top_k[sl], vocab),
                    torch.where(active, top_p[sl], 1),
                    mp,
                )
                tie_logprobs = logprobs
                if probabilities is not None:
                    rows = row_indices[sl].clamp(0, probabilities.shape[0] - 1)
                    probs = probabilities.index_select(0, rows)
                    probs = torch.where(active[:, None] & ~empty, probs, 0)
                    if mp is not None:
                        probs = torch.where(
                            probs >= mp[:, None] * probs.amax(-1, keepdim=True), probs, 0
                        )
                    total = probs.sum(-1, keepdim=True)
                    empty = empty | (total == 0)
                    normalizer = torch.where(empty, 1, total).log()
                # Both sources rank the same (p, processed logprob, ID) keys.
                tie_logprobs = torch.where(empty, -torch.inf, tie_logprobs)
                scores = _order_key(probs, tie_logprobs)
            if values == "logits":
                output_values = gathered.float()
                if mode == "processed":
                    t = torch.where(temperatures[sl] == 0, 1, temperatures[sl])[:, None]
                    support = probs > 0 if probabilities is not None else torch.isfinite(logprobs)
                    output_values = torch.where(support, output_values / t, -torch.inf)
                selected_lp = output_values.gather(1, selected)
            elif mode == "processed" and probabilities is not None:
                selected_lp = probs.gather(1, selected).log() - normalizer
            else:
                output_values = logprobs
                selected_lp = output_values.gather(1, selected)
            if floor is not None:
                selected_lp = torch.where(torch.isneginf(selected_lp), floor, selected_lp)
            selected_score = scores.gather(1, selected)
            ranks = (scores >= selected_score).sum(-1, dtype=torch.int32)
            torch.where(
                active, selected_lp[:, 0], self._zero_logprob, out=out_selected_logprobs[sl]
            )
            torch.where(active, ranks, self._zero_rank, out=out_ranks[sl])
            if self.max_top_k == 0:
                continue
            ids = self._ordered_ids(scores)
            if values == "logprobs" and mode == "processed" and probabilities is not None:
                top_lp = probs.gather(1, ids).log() - normalizer
            else:
                top_lp = output_values.gather(1, ids)
            if floor is not None:
                top_lp = torch.where(torch.isneginf(top_lp), floor, top_lp)
            live_columns = active[:, None] & (self._columns < top_k[sl, None])
            torch.where(live_columns, ids.to(torch.int32), self._padding_id, out=out_top_ids[sl])
            torch.where(live_columns, top_lp, self._zero_logprob, out=out_top_logprobs[sl])

    def warm_up(
        self,
        row_capacities,
        dtypes=(torch.bfloat16, torch.float32),
        modes=("raw", "processed"),
        values=("logprobs", "logits"),
    ) -> None:
        """Exercise value modes, both sources, min-p and the optional floor."""
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm_up must run outside CUDA capture")
        dtypes, modes, values = tuple(dtypes), tuple(modes), tuple(values)
        for n in row_capacities:
            self.workspace_size(n)

            def full(value, dtype=torch.float32, n=n):
                return torch.full((n,), value, dtype=dtype, device=self.device)

            args = {
                "row_indices": torch.arange(n, dtype=torch.int32, device=self.device),
                "num_rows": torch.tensor([n], dtype=torch.int32, device=self.device),
                "selected_ids": full(0, torch.int32),
                "top_k": full(self.max_top_k, torch.int32),
                "out_selected_logprobs": full(0),
                "out_ranks": full(0, torch.int32),
                "out_top_ids": torch.empty(
                    (n, self.max_top_k), dtype=torch.int32, device=self.device
                ),
                "out_top_logprobs": torch.empty(
                    (n, self.max_top_k), dtype=torch.float32, device=self.device
                ),
            }
            for dtype in dtypes:
                logits = torch.zeros((max(n, 1), self.vocab_size), dtype=dtype, device=self.device)
                for value_mode in values:
                    for floor in (None, -9999.0):
                        for mode in modes:
                            if mode == "processed":
                                for use_probs in (False, True):
                                    for mp in (None, full(0)):
                                        self.run(
                                            logits,
                                            **args,
                                            mode=mode,
                                            values=value_mode,
                                            floor=floor,
                                            temperatures=full(1),
                                            filter_top_k=full(self.vocab_size, torch.int32),
                                            top_p=full(1),
                                            min_p=mp,
                                            probabilities=torch.full_like(
                                                logits, 1 / self.vocab_size, dtype=torch.float32
                                            )
                                            if use_probs
                                            else None,
                                        )
                            else:
                                self.run(logits, **args, mode=mode, values=value_mode, floor=floor)
