# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host metadata validation and device-local provider dispatch. No device reads."""

from functools import lru_cache
from itertools import product
from typing import NamedTuple

import torch

from ctikernels._storage import disjoint, reject_grad

# Ordering is also the compiled kernel ABI. Optionals borrow unused nonempty inputs.
P_NAMES = (
    "request_indices",
    "frontier_table",
    "done_table",
    "token_table",
    "cache_index_table",
    "rope_offsets",
    "prompt_lengths",
    "start_frontiers",
    "round_done",
    "k2_slots",
    "input_ids",
    "positions",
    "rope_positions",
    "verify_rope_positions",
    "cache_write_indices",
    "kv_indptr",
    "kv_indices",
    "cursors",
)
# prepare_round: P without KV addressing. fixed_query_addresses (Triton): the rest.
R_NAMES = (
    "request_indices",
    "frontier_table",
    "done_table",
    "token_table",
    "rope_offsets",
    "prompt_lengths",
    "start_frontiers",
    "round_done",
    "k2_slots",
    "input_ids",
    "positions",
    "rope_positions",
    "verify_rope_positions",
    "cursors",
)
A_NAMES = (
    "request_indices",
    "start_frontiers",
    "round_done",
    "cache_index_table",
    "cache_write_indices",
    "kv_indptr",
    "kv_indices",
)
C_NAMES = (
    "request_indices",
    "start_frontiers",
    "round_done",
    "proposals",
    "target_ids",
    "tokens",
    "accepted",
    "output_ends",
    "ignore_eos",
    "stop_token_ids",
    "token_table",
    "frontier_table",
    "done_table",
    "committed",
    "record_done",
    "k2_lengths",
)
I64 = {
    "frontier_table",
    "start_frontiers",
    "rope_offsets",
    "prompt_lengths",
    "positions",
    "rope_positions",
    "verify_rope_positions",
    "cursors",
    "target_ids",
    "output_ends",
}
BOOL = {"done_table", "round_done"}
TABLES = {"token_table", "cache_index_table", "tokens", "proposals"}


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _shape(q, m):
    _integer("block_size", q, 1, 2**31 - 1)
    _integer("max_seq_len", m, 1, 2**31 - 1)


def _geometry(batch, q, m):
    _integer("batch size", batch, 0, 2**31 - 1)
    _shape(q, m)
    if batch * (m + q) > 2**31 - 1:
        raise ValueError("B*(max_seq_len + block_size) must be <= 2147483647 (int32 CSR offsets)")


def _span(t):
    # disjoint() accepts contiguous intervals. Include inter-row gaps in the
    # interval of a strided table; otherwise an alias in the last row is missed.
    n = (
        0
        if t.numel() == 0
        else 1 + sum((s - 1) * d for s, d in zip(t.shape, t.stride(), strict=True))
    )
    return t.as_strided((n,), (1,)) if t.ndim == 2 else t


def _validate(t, names, shapes, writable):
    optional = {
        "rope_offsets",
        "prompt_lengths",
        "k2_slots",
        "verify_rope_positions",
        "cursors",
        "proposals",
        "target_ids",
        "k2_lengths",
    }
    if any(t[n] is None for n in names if n not in optional):
        raise ValueError("required buffers must be tensors, not None")
    present = {n: t[n] for n in names if t[n] is not None}
    if any(not isinstance(v, torch.Tensor) for v in present.values()):
        raise TypeError("all buffers must be tensors")
    reject_grad(*present.values())
    device = t["request_indices"].device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("round bookkeeping supports CPU and CUDA")
    for n, v in present.items():
        dtype = torch.int64 if n in I64 else torch.bool if n in BOOL else torch.int32
        if v.dtype != dtype:
            raise ValueError(f"{n} must have dtype {dtype}")
        if v.device != device:
            raise ValueError("all buffers must be on the same device")
        shape = shapes[n]
        if v.ndim != len(shape) or any(
            s is not None and s != a for s, a in zip(shape, v.shape, strict=True)
        ):
            raise ValueError(f"{n} must have shape {shape}")
        if n in TABLES:
            if v.numel() and (v.stride(1) != 1 or v.stride(0) < v.shape[1]):
                raise ValueError(f"{n} must be row-major with unit inner stride")
        elif not v.is_contiguous():
            raise ValueError(f"{n} must be contiguous")
    outputs = tuple(v for n, v in present.items() if n in writable)
    inputs = tuple(v for n, v in present.items() if n not in writable)
    try:
        disjoint(map(_span, outputs), map(_span, inputs))
    except ValueError:
        # A bounding interval can overlap even when every addressed row is
        # disjoint (e.g. record fields in row-stride gaps). Refine only that
        # uncommon case; the usual separate-buffer path stays constant-size.
        def segments(values):
            for value in values:
                if value.ndim == 2:
                    yield from value.unbind(0)
                else:
                    yield value

        disjoint(segments(outputs), segments(inputs))


def _tables(t, m, cache=False):
    rows = t["frontier_table"].numel()
    if rows < 1 or t["done_table"].numel() != rows:
        raise ValueError("frontier_table and done_table must have the same nonzero length")
    for n in ("token_table", "cache_index_table") if cache else ("token_table",):
        if t[n].shape[0] < rows or t[n].shape[1] < m:
            raise ValueError(f"{n} must cover the state rows and max_seq_len columns")


class _Cuda:
    """CuTe prepare_block, prepare_round and commit_block; Triton fixed_query_addresses."""

    def __init__(self, device):
        from .cute import Provider

        self.device, self.cute, self._addresses = device, Provider(device), None
        self.prepare, self.prepare_round = self.cute.prepare, self.cute.prepare_round
        self.commit = self.cute.commit

    @property
    def triton(self):
        if self._addresses is None:
            from .triton import Addresses

            self._addresses = Addresses(self.device)
            self._addresses._wait = self.cute._wait
        return self._addresses

    @property
    def _wait(self):
        return self.cute._wait

    @_wait.setter
    def _wait(self, value):
        # Private test-only mutation (no PDL wait), applied to both providers.
        self.cute._wait = value
        if self._addresses is not None:
            self._addresses._wait = value

    def warm_up(self, q, s, use_pdl, variants):
        addresses = RoundVariant("fixed_query_addresses")
        self.cute.warm_up(q, s, use_pdl, tuple(v for v in variants if v != addresses))
        if addresses in variants:
            self.triton.warm_up(q, use_pdl)

    def addresses(self, t):
        self.triton(t)


@lru_cache(maxsize=None)
def provider(device):
    if device.type == "cpu":
        from .reference import Provider

        return Provider(device)
    return _Cuda(device)


def prepare_block(
    request_indices,
    frontier_table,
    done_table,
    token_table,
    cache_index_table,
    *,
    block_size,
    max_seq_len,
    guard_start,
    mask_token_id,
    start_frontiers,
    round_done,
    input_ids,
    positions,
    rope_positions,
    cache_write_indices,
    kv_indptr,
    kv_indices,
    rope_offsets=None,
    prompt_lengths=None,
    k2_slots=None,
    verify_rope_positions=None,
    cursors=None,
    window=None,
    use_pdl=False,
):
    """Write one round's fixed-Q inputs and only the valid CSR prefix (R5 §2–3).

    Any B >= 0 (B=0 is a no-op: no launch, nothing written); Q >= 1 and
    B*(max_seq_len + Q) <= 2147483647 (int32 CSR offsets).
    guard_start + B*Q <= 2147483647 (int32 cache indices).
    All outputs are borrowed; returns None. ``window`` is reserved. CUDA uses
    one launch on the current stream. Warm all variants before serving/capture.
    request_indices must contain distinct valid request rows; -1 is not padding.
    """
    t = locals()
    if window is not None:
        raise NotImplementedError("windowed round bookkeeping is not implemented")
    b, q, m = request_indices.numel(), block_size, max_seq_len
    _geometry(b, q, m)
    _integer("guard_start", guard_start, 0, 2**31 - 1 - b * q)
    _integer("mask_token_id", mask_token_id, -(2**31), 2**31 - 1)
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    if (rope_offsets is None) != (verify_rope_positions is None):
        raise ValueError("verify_rope_positions iff rope_offsets")
    if (prompt_lengths is None) != (cursors is None):
        raise ValueError("cursors iff prompt_lengths")
    shapes = _round_shapes(b, q, P_NAMES)
    _validate(t, P_NAMES, shapes, set(P_NAMES[7:]))
    _tables(t, m, cache=True)
    if b:
        provider(request_indices.device).prepare(t)


def _round_shapes(b, q, names):
    shapes = dict.fromkeys(names, (b,))
    shapes.update(dict.fromkeys(("frontier_table", "done_table", "kv_indices"), (None,)))
    shapes.update(dict.fromkeys(("token_table", "cache_index_table"), (None, None)))
    shapes.update(
        dict.fromkeys(
            (
                "input_ids",
                "positions",
                "rope_positions",
                "verify_rope_positions",
                "cache_write_indices",
            ),
            (b * q,),
        )
    )
    shapes["kv_indptr"] = (b + 1,)
    return {n: shapes[n] for n in names}


def prepare_round(
    request_indices,
    frontier_table,
    done_table,
    token_table,
    *,
    block_size,
    max_seq_len,
    mask_token_id,
    start_frontiers,
    round_done,
    input_ids,
    positions,
    rope_positions,
    rope_offsets=None,
    verify_rope_positions=None,
    prompt_lengths=None,
    cursors=None,
    k2_slots=None,
    use_pdl=False,
):
    """prepare_block without KV addressing: no cache_index_table, guard,
    cache_write_indices or CSR. Every other output equals prepare_block's.

    Pair with fixed_query_addresses (which reads start_frontiers/round_done)
    when the fixed-query attention backend owns the KV addressing. One CUDA
    launch on the current stream; warm_up before capture.
    """
    t = locals()
    b, q, m = request_indices.numel(), block_size, max_seq_len
    _geometry(b, q, m)
    _integer("mask_token_id", mask_token_id, -(2**31), 2**31 - 1)
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    if (rope_offsets is None) != (verify_rope_positions is None):
        raise ValueError("verify_rope_positions iff rope_offsets")
    if (prompt_lengths is None) != (cursors is None):
        raise ValueError("cursors iff prompt_lengths")
    _validate(t, R_NAMES, _round_shapes(b, q, R_NAMES), set(R_NAMES[6:]))
    _tables(t, m)
    if b:
        provider(request_indices.device).prepare_round(t)


def fixed_query_addresses(
    request_indices,
    start_frontiers,
    round_done,
    cache_index_table,
    *,
    block_size,
    max_seq_len,
    guard_start,
    cache_write_indices,
    kv_indptr,
    kv_indices,
    use_pdl=False,
):
    """Cache write indices and the valid CSR prefix for one fixed-Q round.

    Reads the round state that prepare_round (or prepare_block) saved, never
    the live tables, with prepare_block's addressing rules: a row is a dummy
    if done or its frontier is outside [0, max_seq_len); its guard is
    g = guard_start + b*Q. request_indices must be valid cache_index_table
    rows. Same bounds as prepare_block. One CUDA launch (a Triton kernel; any
    cc >= 8.0 device); warm_up before capture.
    """
    t = locals()
    b, q, m = request_indices.numel(), block_size, max_seq_len
    _geometry(b, q, m)
    _integer("guard_start", guard_start, 0, 2**31 - 1 - b * q)
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    _validate(t, A_NAMES, _round_shapes(b, q, A_NAMES), set(A_NAMES[4:]))
    if cache_index_table.shape[0] < 1 or cache_index_table.shape[1] < m:
        raise ValueError("cache_index_table must cover max_seq_len columns")
    if b:
        provider(request_indices.device).addresses(t)


def commit_block(
    request_indices,
    start_frontiers,
    round_done,
    *,
    tokens,
    accepted,
    output_ends,
    ignore_eos,
    stop_token_ids,
    token_table,
    frontier_table,
    done_table,
    committed,
    record_done,
    block_size,
    max_seq_len,
    proposals=None,
    target_ids=None,
    k2_lengths=None,
    use_pdl=False,
):
    """Commit a verified round (R5 §3.4), using P's saved state, never live state.

    Any B >= 0 (B=0 is a no-op); Q >= 1 and
    B*(max_seq_len + Q) <= 2147483647 (int32 CSR/count ABI).
    Tensor/table offsets use int64. Q=1 has zero drafts and commits only the bonus.
    proposals+target_ids: greedy; proposals alone: greedy-prefilled; neither:
    sampled (tokens read-only, accepted RMW). Raw acceptance is not budget-clipped.

    Sampled-mode precondition: ``accepted[b]`` (from K1) is in [-1, Q-1]; the
    engine guarantees it. Outside that range v = a+1 < 0 may result: no
    out-of-bounds access and no token writes, but the frontier moves back. No
    clamp and no host check of device values (bitwise parity with the oracle).
    """
    t = locals()
    b, q, m = request_indices.numel(), block_size, max_seq_len
    _geometry(b, q, m)
    if target_ids is not None and proposals is None:
        raise ValueError("target_ids requires proposals")
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    shapes = dict.fromkeys(C_NAMES, (b,))
    shapes.update(dict.fromkeys(("frontier_table", "done_table", "stop_token_ids"), (None,)))
    shapes.update(
        proposals=(b, q - 1), tokens=(b, q), target_ids=(b * q,), token_table=(None, None)
    )
    writable = {
        "accepted",
        "token_table",
        "frontier_table",
        "done_table",
        "committed",
        "record_done",
        "k2_lengths",
    }
    if target_ids is not None:
        writable.add("tokens")
    _validate(t, C_NAMES, shapes, writable)
    _tables(t, m)
    if b:
        provider(request_indices.device).commit(t)


class RoundVariant(NamedTuple):
    """One compiled round-bookkeeping specialization, as declared to warm_up.

    ``kind`` names the op: "prepare_block" (fused), "prepare_round" and
    "fixed_query_addresses" (split pair) or "commit_block". ``rope``,
    ``cursors`` and ``k2`` say whether rope_offsets/verify_rope_positions,
    prompt_lengths/cursors and k2_slots (prepare) or k2_lengths (commit) are
    given. ``mode`` is commit_block's "greedy", "prefilled" or "sampled".
    fixed_query_addresses has no options; on CUDA it is a Triton kernel (the
    others are CuTe), warmed per Q and PDL. Q, S and PDL come from warm_up.
    """

    kind: str
    rope: bool = False
    k2: bool = False
    cursors: bool = False
    mode: str | None = None


PREPARE_KINDS = ("prepare_block", "prepare_round")
COMMIT_MODES = ("greedy", "prefilled", "sampled")
ALL_VARIANTS = (
    tuple(
        RoundVariant(kind, rope, k2, cursors)
        for kind in PREPARE_KINDS
        for rope, k2, cursors in product((False, True), repeat=3)
    )
    + (RoundVariant("fixed_query_addresses"),)
    + tuple(
        RoundVariant("commit_block", k2=k2, mode=m) for m in COMMIT_MODES for k2 in (False, True)
    )
)


def _variant(value):
    if not isinstance(value, RoundVariant):
        raise ValueError(f"variants must hold RoundVariant values, got {value!r}")
    kind, rope, k2, cursors, mode = value
    if any(type(flag) is not bool for flag in (rope, k2, cursors)):
        raise ValueError(f"{value!r}: rope, k2 and cursors must be bool")
    if kind in PREPARE_KINDS:
        ok = mode is None
    elif kind == "fixed_query_addresses":
        ok = mode is None and not (rope or k2 or cursors)
    elif kind == "commit_block":
        ok = mode in COMMIT_MODES and not (rope or cursors)
    else:
        ok = False
    if not ok:
        raise ValueError(f"invalid round variant {value!r}")
    return value


def warm_up(
    block_size,
    max_seq_len,
    num_stop_tokens,
    *,
    device,
    use_pdl=False,
    variants=None,
):
    """Compile the declared round-bookkeeping variants; allocates no device buffers.

    ``variants`` is an iterable of RoundVariant; None declares ALL_VARIANTS
    (fused and split prepare × M-RoPE × K2 × cursors, plus every commit mode ×
    K2). Returns the declared variants, deduplicated in ALL_VARIANTS order;
    each is compiled for this Q/S/PDL and device when the call returns.
    Capturing an op whose variant was not declared raises and names it.

    Call once per Q/S/PDL choice and device at bind, including graph-disabled
    runners; calls accumulate. B (any B >= 0), M, widths, strides, capacities
    and scalar IDs are runtime values: one warm_up covers every batch size.
    """
    _shape(block_size, max_seq_len)
    _integer("num_stop_tokens", num_stop_tokens, 0, 2**31 - 1)
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    declared = ALL_VARIANTS if variants is None else {_variant(v) for v in variants}
    declared = tuple(v for v in ALL_VARIANTS if v in declared)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("round bookkeeping supports CPU and CUDA")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    provider(device).warm_up(block_size, num_stop_tokens, use_pdl, declared)
    return declared


def compile_count():
    """Number of round-bookkeeping specializations compiled in this process:
    CuTe kernels plus Triton addresses binaries (CPU adds zero)."""
    from .cute import compile_count as cute
    from .triton import compile_count as triton

    return cute() + triton()
