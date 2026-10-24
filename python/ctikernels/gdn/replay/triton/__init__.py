# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton launch adapter; borrowed tensors only (plus one op-owned trusted status word),
current stream, no host value reads."""

from dataclasses import astuple

import torch
import triton

from ...._runtime.workspace import GraphWorkspace
from .._layout import layout
from . import _handoff
from . import _kernels as k
from . import _status as control
from ._status import clear_poison, invalidate, lifecycle, recovery_finish

__all__ = ["clear_poison", "handoff_layers", "invalidate", "lifecycle", "recovery_finish", "run"]


class _Trusted:
    """A trusted call's status: a zero code and poison word no kernel writes (only the
    skipped guard and validation write status), so every row runs as under a clean status."""

    def __init__(self, words):
        self.code, self.poison = words[0:1], words[1:2]


def _trusted(op):
    words = getattr(op, "_triton_trusted_words", None)
    if words is None:
        words = op._triton_trusted_words = torch.zeros(2, dtype=torch.int32, device=op.device)
    return _Trusted(words)


def run(op, mode, *, state, status, weights=None, step_impl=None, **kw):
    # Without a status the call is trusted (valid, uniquely mapped rows): no guard, no
    # validation, the same arithmetic and publication.
    trusted = status is None
    handle = _trusted(op) if trusted else status
    selector = kw.get("status_slot")
    slot_control, slot_args = {}, {}
    if selector is not None:
        pending = kw["pending"]
        depth = pending.raw.shape[0] if trusted else handle.sequence.numel()
        slot_control = {"SLOT": selector, "HAS_SLOT": True, "SLOT_DEPTH": depth}
        slot_args = dict(
            **slot_control,
            SD_STATUS=0 if trusted else handle.records.stride(1) // 4,
            SD_PR=pending.raw.stride(0),
            SD_PM=pending.metadata.stride(0),
            SD_VALID=kw["snapshot_valid"].stride(0) if kw.get("snapshot_valid") is not None else 0,
        )
    if not trusted:
        control.guard[(1,)](
            handle.record,
            handle.sequence,
            handle.poison,
            **slot_control,
            **(
                {"RD": handle.records.stride(1), "SQD": handle.sequence.stride(0)}
                if selector is not None
                else {}
            ),
        )
    status = handle.code
    s = state
    r, ns = s.metadata.shape[0], s.snapshot_metadata.shape[0]
    if mode == "reset":
        k.advance_epoch[(1,)](s.pool_epoch, status, handle.poison)
        k.reset_metadata[(max(1, triton.cdiv(max(r * 8, ns * 4), 256)),)](
            s.metadata,
            s.pool_epoch,
            s.snapshot_metadata,
            status,
            handle.poison,
            kw["weight_revision"],
            r,
            ns,
            256,
        )
        for p in kw["pending_blocks"]:
            k.reset_pending[(max(1, triton.cdiv(p.metadata.numel(), 256)),)](
                p.metadata, status, handle.poison, p.metadata.numel(), 256
            )
        return

    def arg(name):
        value = kw.get(name)
        return status if value is None else value

    lt = astuple(layout(op.geometry, s, kw))
    indices = arg("pairs") if "pairs" in kw else arg("slot_indices")
    batch = indices.shape[0]
    if not batch:
        return
    pending = kw.get("pending")
    pr, pm = (pending.raw, pending.metadata) if pending is not None else (status, status)
    pb = pending.raw.shape[int(selector is not None)] if pending is not None else 0
    capture = kw.get("snapshot_rows") is not None
    pos = arg("positions") if "positions" in kw else arg("next_positions")
    offsets = arg("offsets") if mode == "export_snapshot" else arg("snapshot_offsets")
    inplace = mode == "handoff" and (
        kw["canonical_state"].data_ptr() == s.checkpoint.data_ptr()
        or kw["history"].data_ptr() == s.conv_history.data_ptr()
    )
    g, window, q = op.geometry, op.window, op.verify_length
    if not trusted:
        k.validate[(batch,)](
            indices,
            pos,
            s.metadata,
            s.pool_epoch,
            s.snapshot_metadata,
            pm,
            arg("accepted_lengths"),
            arg("snapshot_rows"),
            offsets,
            status,
            handle.poison,
            r,
            ns,
            window,
            q,
            pb,
            mode,
            capture,
            inplace,
            LT=lt,
            **slot_args,
        )
        control.propagate[(1,)](
            status,
            handle.poison,
            **slot_control,
            **({"SD_STATUS": slot_args["SD_STATUS"]} if slot_args else {}),
        )
    x = kw.get("projected")
    xs = x.stride(0) if x is not None else 0
    if mode == "bridge_step":
        # Validation has completed for the entire batch. Materialize into the
        # borrowed canonical pool, then run the immediate provider and handoff.
        k.arithmetic[(batch, g.v_heads, _tiles("flush"))](
            s.checkpoint,
            s.ring,
            s.metadata,
            indices,
            status,
            pr,
            pm,
            *weights,
            status,
            status,
            status,
            status,
            status,
            handle.poison,
            status,
            r,
            g.k_heads,
            g.v_heads,
            window,
            q,
            0,
            0,
            "flush",
            False,
            LT=lt,
            BV=128 // _tiles("flush"),
            num_warps=_warps("flush"),
            enable_fp_fusion=False,
        )
        k.publish[(batch,)](
            s.checkpoint,
            s.conv_history,
            s.ring,
            s.metadata,
            s.pool_epoch,
            s.snapshot_metadata,
            indices,
            pos,
            status,
            pr,
            pm,
            status,
            status,
            status,
            status,
            status,
            handle.poison,
            status,
            r,
            g.k_heads,
            g.v_heads,
            window,
            q,
            0,
            0,
            "flush",
            False,
            triton.next_power_of_2(g.conv_dim + 2 * g.v_heads),
            LT=lt,
            num_warps=8,
        )
    prenorm = None
    if mode == "step" and step_impl is not None:
        step_impl(op, state=state, status=None if trusted else handle, weights=weights, **kw)
    elif mode not in ("reject", "release"):
        if mode == "verify" and _tiles(mode, q) > 1:
            prenorm = _prenorm(s.checkpoint, batch * q, g.v_heads)
        k.arithmetic[(batch, g.v_heads, _tiles(mode, q))](
            s.checkpoint,
            s.ring,
            s.metadata,
            indices,
            arg("projected"),
            pr,
            pm,
            *(weights or (status,) * 4),
            arg("accepted_lengths"),
            arg("snapshot_rows"),
            offsets,
            arg("out"),
            status,
            handle.poison,
            arg("canonical_state"),
            r,
            g.k_heads,
            g.v_heads,
            window,
            q,
            pb,
            xs,
            mode,
            capture,
            LT=lt,
            **slot_args,
            BV=128 // _tiles(mode, q),
            PRENORM=prenorm,
            num_warps=_warps(mode, q),
            enable_fp_fusion=False,
        )
    width = triton.next_power_of_2(g.conv_dim + 2 * g.v_heads)
    chunks = _PUBLISH_CHUNKS if mode == "verify" else 1
    k.publish[(batch, chunks)](
        s.checkpoint,
        s.conv_history,
        s.ring,
        s.metadata,
        s.pool_epoch,
        s.snapshot_metadata,
        indices,
        pos,
        arg("projected"),
        pr,
        pm,
        arg("accepted_lengths"),
        arg("snapshot_rows"),
        offsets,
        arg("snapshot_valid"),
        status,
        handle.poison,
        arg("history"),
        r,
        g.k_heads,
        g.v_heads,
        window,
        q,
        pb,
        xs,
        mode,
        capture,
        width // chunks,
        LT=lt,
        **slot_args,
        RINGLESS=s.ring.shape[1] == 0,
        CHUNKS=chunks,
        **(
            {
                "PRENORM": prenorm,
                "OUT": arg("out"),
                "NW": weights[3],
                "QP": triton.next_power_of_2(q),
            }
            if mode == "verify" and prenorm is not None
            else {}
        ),
        num_warps=8 if chunks == 1 else 4,
    )


# Warps per arithmetic program: the per-token chain is reduction-latency bound, and on GB10's
# anchor (Qwen3.5-9B, Q=8, B=1) 4 warps run it in 56% of 8 warps' time. Split programs hold
# fewer rows and use fewer warps.
_WARPS = 4
_SPLIT_WARPS = 4
_VERIFY_WARPS = 4


def _warps(mode, q=0):
    if _tiles(mode, q) == 1:
        return _WARPS
    return _VERIFY_WARPS if mode == "verify" else _SPLIT_WARPS


# Column blocks of verify's pending-record copy (one program per row was a serial tail).
_PUBLISH_CHUNKS = 32
# Longest verify that splits its V rows (pre-norm scratch: B * Q * hv * 512 bytes).
_SPLIT_VERIFY_Q = 64
# Value rows per program in modes without an output (the output's gated norm spans all 128).
_STATE_ROWS = 32


def _tiles(mode, q=0):
    """V-row programs per head: output-free modes always split; verify splits up to
    _SPLIT_VERIFY_Q tokens (its FP32 pre-norm scratch grows with Q); step never does."""
    if mode in ("step", "bridge_step") or (mode == "verify" and q > _SPLIT_VERIFY_Q):
        return 1
    return 128 // _STATE_ROWS


def _prenorm(x, tokens, heads):
    """FP32 [tokens, heads, 128] pre-norm verify outputs: the active GraphWorkspace's slot,
    otherwise call-owned (a capture outside a workspace takes it from the graph's pool)."""
    counts = (tokens * heads * 128,)
    group = GraphWorkspace.active()
    if group is None:
        return torch.empty(counts, dtype=torch.float32, device=x.device)
    slot = ("gdn_replay_prenorm", x.device)
    group.reserve(
        slot,
        (heads,),
        lambda rows, h: (rows * h * 128,),
        lambda n: (torch.empty(n, dtype=torch.float32, device=x.device),),
    )
    return group.take(slot, tokens, counts)[0]


# Elements per program step; programs per (row, layer) cover about four steps of the larger copy.
_HANDOFF_BLOCK = 1024


def handoff_layers(op, layers, positions, indices, status):
    """Every bound layer's handoff in two launches (strict control, then payload)."""
    strict = status is not None
    j, b = len(layers.states), layers.batch_size
    pitches = tuple(
        (s.checkpoint.stride(0), s.conv_history.stride(0), c.stride(0), h.stride(0))
        for s, c, h in zip(layers.states, layers.canonical_states, layers.histories, strict=True)
    )
    cache = op.__dict__.setdefault("_triton_handoff_pitches", {})
    key = (layers.table.data_ptr(), pitches)
    table = cache.get(key)
    if table is None:  # never under capture: the op admits only warmed table/stride signatures
        table = cache[key] = torch.tensor(pitches, dtype=torch.int64).to(op.device)
    g = op.geometry
    records = status.records if strict else layers.table_flat
    rs = status.records.stride(0) if strict else 0
    if strict:
        _handoff.control[(1,)](
            layers.table_flat, positions, indices, records, status.sequence, status.poison,
            b, indices.stride(0), positions.stride(0),
            SLOTS=layers.num_slots, LAYERS=j, RS=rs, BLOCK=128, num_warps=4,
        )  # fmt: skip
    if not b:
        return
    work = max(3 * g.conv_dim, g.v_heads * 128 * 128 if layers.copies_checkpoint else 0)
    chunks = max(1, triton.cdiv(work, 4 * _HANDOFF_BLOCK))
    _handoff.payload[(chunks, b, j)](
        layers.table_flat, table, positions, indices, records,
        status.sequence if strict else positions, indices.stride(0), positions.stride(0),
        SLOTS=layers.num_slots, WINDOW=op.window, DC=g.conv_dim, HV=g.v_heads, RS=rs,
        STRICT=strict, RINGLESS=all(s.ring.shape[1] == 0 for s in layers.states),
        BLOCK=_HANDOFF_BLOCK, num_warps=4,
    )  # fmt: skip
