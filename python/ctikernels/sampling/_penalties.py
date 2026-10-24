# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DN4 borrowed-output penalties; semantics follow vLLM 0.31.0 V2; no code copied.

FP32 conversion, rounded reciprocal then multiply for positive logits (multiply
by repetition otherwise), frequency product then subtraction, presence product
then subtraction. Each arithmetic operation rounds separately; no fusion/FMA.
Invalid vocabulary IDs are ignored. Metadata-only validation and shape-static
loops are CUDA-graph capturable. The whole token-table width L is scanned,
independent of actual lengths: cost scales with L for this torch provider.
"""

import torch

_INTS = (torch.int32, torch.int64)
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def _span(tensor):
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))
    return tensor.data_ptr(), tensor.data_ptr() + extent * tensor.element_size()


def _overlapping_rows(tensor):
    """Exact metadata-only check for up to two positive leading strides."""
    if not tensor.numel():
        return False
    width = tensor.shape[-1]
    dimensions = [
        (size, stride)
        for size, stride in zip(tensor.shape[:-1], tensor.stride()[:-1], strict=True)
        if size > 1
    ]
    if any(stride < width for _, stride in dimensions):
        return True
    if len(dimensions) < 2:
        return False
    # Same-sign row displacements cannot collide; for opposite signs, the
    # nearest bounded integer multiple of the other stride is sufficient.
    (short, step), (long, other) = sorted(dimensions)
    for delta in range(1, short):
        multiple = min(long - 1, max(1, delta * step // other))
        for candidate in (multiple, min(long - 1, multiple + 1)):
            if abs(delta * step - candidate * other) < width:
                return True
    return False


class LogitPenalties:
    """Apply engine-owned PenaltyRows to [N,V] or [N,Q,V] logits.

    rows, prompt_lengths, lengths are int32/int64[N]; presence, frequency,
    repetition are FP32[N]. Valid row indices, 0<=P<=C<=L, finite presence and
    frequency, and finite repetition>0 are caller preconditions, never read on
    the host. Counts (including drafts) must fit int32. Out is borrowed FP32;
    only an exact FP32 logits alias is allowed. All output elements are written.

    element_budget bounds live temporary tensor elements, excluding inputs,
    out and allocator/graph-pool caching. Two V-wide int32 count arrays are
    kept per request. A broadcast base pass handles positions together, followed by gathered draft fix-ups.
    Requests, positions, columns and draft comparisons are chunked by budget;
    at least 3*V+56 elements are required, including saved source logits for
    in-place fix-ups. No N*Q*V count allocation is needed.
    """

    def __init__(self, token_table, vocab_size, *, element_budget=2**24):
        if type(vocab_size) is not int or not 1 <= vocab_size < 2**31:
            raise ValueError("vocab_size must be an integer in [1, 2**31-1]")
        if type(element_budget) is not int or element_budget < 3 * vocab_size + 56:
            raise ValueError("element_budget must be an integer >= 3*vocab_size+56")
        if not isinstance(token_table, torch.Tensor):
            raise TypeError("token_table must be a tensor")
        if token_table.ndim != 2:
            raise ValueError("token_table must have shape [T,L]")
        self.device = token_table.device
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("token_table must be on CPU or CUDA")
        self._check("token_table", token_table, _INTS, tuple(token_table.shape))
        self.token_table, self.vocab_size = token_table, vocab_size
        self.element_budget = element_budget
        # Budget at least one gathered column even for the smallest budget.
        self.row_chunk = max(1, element_budget // (12 * vocab_size + 44))
        self.column_chunk = max(
            1,
            min(
                token_table.shape[1],
                (element_budget - 3 * vocab_size - 32) // 12,
            ),
        )
        self._columns = torch.arange(self.column_chunk, device=self.device, dtype=torch.int64)
        self._offsets = torch.arange(self.row_chunk, device=self.device, dtype=torch.int64)[:, None]
        self._offsets.mul_(2 * vocab_size)

    @property
    def provider(self) -> str:
        return "triton" if self.device.type == "cuda" else "torch"

    def _check(self, name, tensor, dtypes, shape):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in dtypes:
            raise TypeError(f"{name} must have dtype in {dtypes}")
        if tensor.requires_grad:
            raise ValueError(f"{name} requires grad")
        if tensor.device != self.device or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has wrong device or shape; expected {shape} on {self.device}")
        if any(s <= 0 for s in tensor.stride()):
            raise ValueError(f"{name} requires positive strides")

    def _count_columns(self, rows, sl, start, stop, counts):
        ids = self.token_table[:, start:stop].index_select(0, rows.rows[sl]).long()
        columns = self._columns[: stop - start] + start
        valid = (ids >= 0) & (ids < self.vocab_size)
        is_output = columns[None, :] >= rows.prompt_lengths[sl, None]
        in_context = columns[None, :] < rows.lengths[sl, None]
        index = ids.clamp(0, self.vocab_size - 1)
        index.add_(is_output.long() * self.vocab_size)
        index.add_(self._offsets[: ids.shape[0]])
        counts.view(-1).scatter_add_(0, index.reshape(-1), (valid & in_context).int().reshape(-1))

    def _arithmetic(self, x, rows, sl, prompt, counts, target):
        scale = torch.where((prompt > 0) | (counts > 0), rows.repetition[sl, None, None], 1.0)
        reciprocal = torch.ones_like(scale) / scale
        value = x * torch.where(x > 0, reciprocal, scale)
        del scale, reciprocal
        frequency = rows.frequency[sl, None, None] * counts.float()
        value = value - frequency
        del frequency
        presence = rows.presence[sl, None, None] * (counts > 0).float()
        value = value - presence
        # Preserve signed zero even if the caller supplies negative-zero
        # coefficients. This is also safe for infinities and maximum FP32.
        identity = (rows.presence[sl] == 0) & (rows.frequency[sl] == 0) & (rows.repetition[sl] == 1)
        torch.where(identity[:, None, None], x, value, out=target)

    def _draft_fixup(self, saved, rows, sl, drafts, first, last, base, out, spare, full_source):
        positions = torch.arange(first, last, device=self.device)
        size, group, d = base.shape[0], last - first, drafts.shape[1]
        # Reserve sixteen elements per target and eight per prefix comparison.
        # On tiny budgets both target and source draft slots split statically.
        targets = min(d, max(1, spare // (group * (16 + 8 * d))))
        sources = min(d, max(1, (spare - 16 * group * targets) // (8 * group * targets)))
        for start in range(0, d, targets):
            stop = min(start + targets, d)
            ids = drafts[:, start:stop].long().clamp(0, self.vocab_size - 1)
            index = ids[:, None, :].expand(size, group, stop - start)
            counts = base[:, 1].gather(1, ids)[:, None, :].expand_as(index).clone()
            for col in range(0, d, sources):
                end = min(col + sources, d)
                source = drafts[:, col:end]
                slots = torch.arange(col, end, device=self.device)
                # Compare original IDs with valid target columns: invalid source
                # IDs cannot match, even when an invalid target was clamped.
                equal = ids[:, :, None] == source[:, None, :]
                preceding = slots[None, :] < positions[:, None]
                counts.add_((equal[:, None] & preceding[None, :, None]).sum(-1, dtype=torch.int32))
                del source, slots, equal, preceding
            prompt = base[:, 0].gather(1, ids)[:, None, :]
            x = saved.gather(2, index) if full_source else saved[:, :, start:stop]
            value = torch.empty_like(x)
            self._arithmetic(x, rows, sl, prompt, counts, value)
            # Duplicate (including clamped invalid) targets get identical bits.
            out.scatter_(2, index, value)
            del ids, index, counts, prompt, x, value

    def _positions(self, logits, rows, sl, drafts, first, last, base, out, backup):
        group = last - first
        source, target = logits[sl, first:last], out[sl, first:last]
        saved = None
        full_source = drafts is not None and self.vocab_size <= 2 * drafts.shape[1]
        if drafts is not None:
            # Save ALL fix-up inputs for this position group before any base
            # writes. A full source copy is smaller for very small vocabularies.
            if full_source:
                saved = source.clone() if source.dtype == torch.float32 else source.float()
            else:
                ids = drafts.long().clamp(0, self.vocab_size - 1)
                saved = source.gather(2, ids[:, None, :].expand(-1, group, -1))
                del ids
                saved = saved.float()
        spare = self.element_budget // base.shape[0] - 2 * self.vocab_size - group * backup - 32
        width = min(self.vocab_size, max(1, spare // (4 * group + 2)))
        for col in range(0, self.vocab_size, width):
            columns = slice(col, col + width)
            self._arithmetic(
                source[:, :, columns].float(),
                rows,
                sl,
                base[:, 0, None, columns],
                base[:, 1, None, columns],
                target[:, :, columns],
            )
        if drafts is not None:
            self._draft_fixup(
                saved,
                rows,
                sl,
                drafts,
                first,
                last,
                base,
                target,
                spare,
                full_source,
            )

    def apply(self, logits, rows, lookahead=None, *, out) -> None:
        """Write all penalties with no host reads, using shape-static chunks."""
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a tensor")
        if logits.ndim not in (2, 3) or logits.shape[-1] != self.vocab_size:
            raise ValueError("logits must have shape [N,V] or [N,Q,V]")
        n, q = logits.shape[0], 1 if logits.ndim == 2 else logits.shape[1]
        if q < 1 or self.token_table.shape[1] + q - 1 >= 2**31:
            raise ValueError("Q must be positive and L+Q-1 must fit int32 counts")
        inputs = [self.token_table, logits]
        for name in ("rows", "prompt_lengths", "lengths", "presence", "frequency", "repetition"):
            tensor = getattr(rows, name)
            self._check(
                name,
                tensor,
                _INTS if name in ("rows", "prompt_lengths", "lengths") else (torch.float32,),
                (n,),
            )
            inputs.append(tensor)
        self._check("logits", logits, _FLOATS, tuple(logits.shape))
        self._check("out", out, (torch.float32,), tuple(logits.shape))
        for name, tensor in (("logits", logits), ("out", out)):
            if tensor.stride(-1) != 1:
                raise ValueError(f"{name} requires unit inner stride")
        if _overlapping_rows(out):
            raise ValueError("out has overlapping elements")
        if lookahead is None:
            if q > 1:
                raise ValueError("lookahead is required for Q>1")
        else:
            self._check("lookahead", lookahead, _INTS, (n, q - 1))
            inputs.append(lookahead)
        if out.numel():
            lo, hi = _span(out)
            for src in inputs:
                if not src.numel():
                    continue
                exact = (
                    src is logits
                    and src.dtype == torch.float32
                    and src.data_ptr() == out.data_ptr()
                    and src.stride() == out.stride()
                )
                if exact:
                    continue
                slo, shi = _span(src)
                if lo < shi and slo < hi:
                    raise ValueError("out must not overlap another input except exact FP32 logits")
        # Backup storage is at most V elements for one position on tiny
        # vocabularies, or two draft-width buffers during gather/conversion.
        drafts = lookahead if q > 1 else None
        backup = min(self.vocab_size, 2 * (q - 1)) if drafts is not None else 0
        # Four position-wide arrays plus two broadcast coefficient arrays
        # cover the base arithmetic; fix-ups need at least 24 per position.
        per_position = max(4 * self.vocab_size, 24) + backup
        row_chunk = min(
            self.row_chunk,
            max(1, self.element_budget // (4 * self.vocab_size + 32 + q * per_position)),
        )
        if logits.ndim == 2:
            logits, out = logits[:, None, :], out[:, None, :]
        if self.device.type == "cuda":
            from .triton import _penalties as triton_penalties

            triton_penalties.apply(self.token_table, self.vocab_size, logits, rows, drafts, out)
            return
        for start in range(0, n, row_chunk):
            stop = min(n, start + row_chunk)
            sl = slice(start, stop)
            budget = self.element_budget // (stop - start)
            column_chunk = min(self.column_chunk, (budget - 3 * self.vocab_size - 32) // 12)
            counts = torch.zeros(
                (stop - start, 2, self.vocab_size), dtype=torch.int32, device=self.device
            )
            for col in range(0, self.token_table.shape[1], column_chunk):
                self._count_columns(
                    rows, sl, col, min(col + column_chunk, self.token_table.shape[1]), counts
                )
            positions = max(1, (budget - 4 * self.vocab_size - 32) // per_position)
            for first in range(0, q, positions):
                self._positions(
                    logits,
                    rows,
                    sl,
                    None if drafts is None else drafts[sl],
                    first,
                    min(first + positions, q),
                    counts,
                    out,
                    backup,
                )
            del counts
