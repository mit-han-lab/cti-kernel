# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Pure torch CPU oracle for replay contract §§3–8; never used for CUDA dispatch.

Original token recurrence from the contract. FP64 starts from exact supplied values;
FP32 provides a second arithmetic mode. Host loops/value reads are reference-only.
"""

from dataclasses import fields, replace

import torch

from ._status import lifecycle

__all__ = ["lifecycle", "run"]


def token(g, weights, matrix, history, raw, z=None):
    """One immediate update, in matrix.dtype, without storage-boundary rounding."""
    cw, alog, bias, norm = (w.to(matrix.dtype) for w in weights)
    dc, hk, hv = g.conv_dim, g.k_heads, g.v_heads
    window = torch.cat((history, raw[:dc][None])).to(matrix.dtype)
    c = (window.T * cw).sum(-1)
    c = c * torch.sigmoid(c)
    q, k, v = c.split((hk * 128, hk * 128, hv * 128))
    q, k = q.reshape(hk, 128), k.reshape(hk, 128)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = (a.repeat_interleave(hv // hk, 0) for a in (q, k))
    b, a = raw[dc:].to(matrix.dtype).split(hv)
    alpha = torch.exp(-torch.exp(alog) * torch.logaddexp(a + bias, torch.zeros_like(a)))
    d = torch.sigmoid(b)[:, None] * (
        v.reshape(hv, 128) - alpha[:, None] * (matrix * k[:, None, :]).sum(-1)
    )
    matrix = alpha[:, None, None] * matrix + d[:, :, None] * k[:, None, :]
    history = window[1:].to(history.dtype)
    y = None
    if z is not None:
        o = (matrix * q[:, None, :]).sum(-1)
        z = z.to(matrix.dtype).reshape(hv, 128)
        y = (
            norm * o * torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6) * z * torch.sigmoid(z)
        ).flatten()
    return matrix, history, y


def raw_row(g, projected):
    return torch.cat((projected[: g.conv_dim], projected[g.conv_dim + g.value_width :]))


def compose(op, state, slot, weights):
    """Replay checkpoint seeds plus committed raw rows; source is never changed."""
    meta = state.metadata[slot]
    n, cursor = int(meta[3]), int(meta[2])
    ring = state.ring[slot]
    start = cursor - n
    history = torch.stack(
        [ring[(start - 3 + i) % (op.window + 3), : op.geometry.conv_dim] for i in range(3)]
    )
    matrix = state.checkpoint[slot].to(op.reference_dtype).clone()
    for j in range(n):
        matrix, history, _ = token(
            op.geometry, weights, matrix, history, ring[(start + j) % (op.window + 3)]
        )
    return matrix, history


def _canonical(op, state, slot, matrix, history, position, *, reuse=False):
    state.checkpoint[slot].copy_(matrix)
    state.conv_history[slot].copy_(history)
    meta = state.metadata[slot]
    meta[0] = meta[1] = position
    meta[2:4] = 0
    meta[4] = 1
    if reuse:
        meta[5] += 1
    meta[7] = state.pool_epoch[0]
    state.ring[slot, -3:, : op.geometry.conv_dim].copy_(history)
    state.ring[slot, -3:, op.geometry.conv_dim :] = 0


def _snapshot(state, target, matrix, history, position, source):
    state.checkpoint[target].copy_(matrix)
    state.conv_history[target].copy_(history)
    sm = state.snapshot_metadata[target - state.metadata.shape[0]]
    sm[0], sm[1], sm[2], sm[3] = position, source[6], source[7], 1


def _slotted_run(op, mode, *, state, status, status_slot, weights, **kw):
    """Select/scatter with tensor indices; never extract the invocation slot.

    The CPU oracle already reads recurrence controls on the host. Its transaction
    runs on private copies so an invalid device selector cannot alter any payload.
    """
    from .._storage import ReplayPending

    pending = kw["pending"]
    depth = pending.raw.shape[0]
    good = ((status_slot >= 0) & (status_slot < depth)).reshape(())
    index = status_slot.to(torch.int64).clamp(0, depth - 1)
    selected = replace(
        status,
        records=status.records.index_select(1, index).squeeze(1),
        validity=status.validity.index_select(1, index).squeeze(1),
        sequence=status.sequence.index_select(0, index),
        poison=status.poison.clone(),
        slotted=False,
    )
    staged = replace(state, **{f.name: getattr(state, f.name).clone() for f in fields(state)})
    local = dict(kw)
    local["pending"] = ReplayPending(
        pending.raw.index_select(0, index).squeeze(0),
        pending.metadata.index_select(0, index).squeeze(0),
    )
    for name in ("snapshot_valid", "arrivals"):
        if kw.get(name) is not None:
            local[name] = kw[name].index_select(0, index).squeeze(0)
    if kw.get("out") is not None:
        local["out"] = kw["out"].clone()
    run(op, mode, state=staged, status=selected, weights=weights, **local)

    def copy(target, value):
        target.copy_(torch.where(good, value, target))

    def scatter(target, value, axis=0):
        previous = target.index_select(axis, index)
        target.index_copy_(axis, index, torch.where(good, value.unsqueeze(axis), previous))

    for f in fields(state):
        copy(getattr(state, f.name), getattr(staged, f.name))
    for f in fields(pending):
        scatter(getattr(pending, f.name), getattr(local["pending"], f.name))
    scatter(status.records, selected.records, 1)
    scatter(status.validity, selected.validity, 1)
    status.poison.copy_(torch.where(good, selected.poison, status.poison | 64))
    for name in ("snapshot_valid", "arrivals"):
        if kw.get(name) is not None:
            scatter(kw[name], local[name])
    if kw.get("out") is not None:
        copy(kw["out"], local["out"])


def run(op, mode, *, state, status, weights=None, status_slot=None, **kw):
    if status_slot is not None:
        _slotted_run(
            op, mode, state=state, status=status, weights=weights, status_slot=status_slot, **kw
        )
        return
    from ._status import guard

    handle = status
    if not guard(handle):
        # As on the device: a guard failure or an earlier poison still propagates
        # this record's whole code (including the dependency bit 32) into poison,
        # except for reset and empty batches, which never reach propagation.
        indices = kw.get("slot_indices", kw.get("pairs"))
        if mode != "reset" and indices is not None and indices.shape[0]:
            handle.poison[0] |= handle.code[0]
        if kw.get("out") is not None:
            kw["out"].zero_()
        return
    status = handle.code
    if mode == "reset":
        if int(status[0]):
            return
        state.pool_epoch.add_(1)
        state.metadata[:, :5] = 0
        state.metadata[:, 5] += 1
        state.metadata[:, 6] = kw["weight_revision"]
        state.metadata[:, 7] = state.pool_epoch[0]
        state.snapshot_metadata.zero_()
        for p in kw["pending_blocks"]:
            p.metadata.zero_()
        return
    slots, pairs = kw.get("slot_indices"), kw.get("pairs")
    entries = slots.tolist() if slots is not None else pairs[:, 0].tolist()
    r = state.metadata.shape[0]
    total = state.checkpoint.shape[0]
    pending = kw.get("pending")
    tasks, error = [], 0
    for b, source in enumerate(entries):
        target = int(pairs[b, 1]) if pairs is not None else source
        if source == -1 or target == -1:
            continue
        slot = target if mode == "import_snapshot" else source
        if not 0 <= slot < r:
            error |= 1
            continue
        meta = state.metadata[slot]
        if mode == "import_snapshot":
            if not r <= source < total:
                error |= 16
                continue
            sm = state.snapshot_metadata[source - r]
            if int(sm[3]) != 1 or sm[1] != meta[6] or sm[2] != state.pool_epoch[0]:
                error |= 16
        elif mode not in ("initialize", "handoff", "release"):
            if (
                int(meta[4]) != 1
                or meta[7] != state.pool_epoch[0]
                or not 0 <= int(meta[3]) < op.window
                or int(meta[1] - meta[0]) != int(meta[3])
                or not 0 <= int(meta[2]) < op.window + 3
            ):
                error |= 1
        if mode in ("initialize", "handoff"):
            if int(kw["next_positions"][b]) < 0:
                error |= 2
            if (
                mode == "handoff"
                and any(
                    kw[name].data_ptr() == getattr(state, dest).data_ptr()
                    for name, dest in (
                        ("canonical_state", "checkpoint"),
                        ("history", "conv_history"),
                    )
                )
                and slot != b
            ):
                error |= 1
        if mode in ("step", "bridge_step", "verify"):
            p = kw["positions"][b].reshape(-1)
            if any(
                int(p[j]) != int(meta[1]) + j or int(p[j]) >= 2**63 - 1 for j in range(p.numel())
            ):
                error |= 2
        pb, m, k, snap = b, 0, 0, -1
        if mode == "export_snapshot":
            snap = target
            if not r <= snap < total:
                error |= 16
            if pending is not None:
                matches = (pending.metadata[:, 0] == slot).nonzero().flatten().tolist()
                # Empty/stale descriptors do not count as a transaction match.
                matches = [j for j in matches if int(pending.metadata[j, 5]) in (1, 2)]
                if len(matches) != 1:
                    error |= 8
                    continue
                pb = matches[0]
        if mode in ("commit", "reject") or (mode == "export_snapshot" and pending is not None):
            pm = pending.metadata[pb]
            if (
                int(pm[0]) != slot
                or pm[1] != meta[5]
                or pm[2] != meta[0]
                or pm[3] != meta[1]
                or int(pm[4]) != op.verify_length
                or int(pm[5]) not in (1, 2)
                or pm[7] != state.pool_epoch[0]
            ):
                error |= 8
            if mode != "reject":
                m = int(kw["accepted_lengths"][pb])
                if not 0 <= m <= op.verify_length:
                    error |= 4
                if int(pm[5]) == 2 and int(pm[6]) != m:
                    error |= 8
                if int(meta[1]) > 2**63 - 1 - max(m, 0):
                    error |= 2
        if mode == "commit" and kw.get("snapshot_rows") is not None:
            snap = int(kw["snapshot_rows"][b])
            k = int(kw["snapshot_offsets"][b])
        if mode == "export_snapshot" and pending is not None:
            k = int(kw["offsets"][b])
        if snap != -1:
            if not r <= snap < total:
                error |= 16
            if k < 0:
                error |= 4
        tasks.append((b, slot, source, pb, m, k, snap))
    status[0] |= error
    # Propagation ORs the accumulated record code, as the device does.
    if entries:
        handle.poison[0] |= status[0]
    if int(status[0]):
        if kw.get("out") is not None:
            kw["out"].zero_()
        # Poison suppresses validity writes too, matching the CUDA stage. A stale
        # true bit is never permission to publish a failed sequence.
        return
    if kw.get("out") is not None:
        kw["out"].zero_()
    if kw.get("snapshot_valid") is not None:
        kw["snapshot_valid"].zero_()
    for b, slot, source, pb, m, k, snap in tasks:
        meta = state.metadata[slot]
        if mode == "release":
            meta[2:5] = 0
            meta[5] += 1
        elif mode == "initialize":
            _canonical(
                op,
                state,
                slot,
                torch.zeros_like(state.checkpoint[slot]),
                torch.zeros_like(state.conv_history[slot]),
                kw["next_positions"][b],
                reuse=True,
            )
        elif mode == "handoff":
            _canonical(
                op,
                state,
                slot,
                kw["canonical_state"][b],
                kw["history"][b].clone(),
                kw["next_positions"][b],
                reuse=int(meta[4]) != 1 or meta[7] != state.pool_epoch[0],
            )
        elif mode == "import_snapshot":
            _canonical(
                op,
                state,
                slot,
                state.checkpoint[source],
                state.conv_history[source],
                state.snapshot_metadata[source - r, 0],
                reuse=True,
            )
        elif mode == "reject":
            pending.metadata[pb, 5] = 4
        else:
            matrix, history = compose(op, state, slot, weights)
            pos = int(meta[1])
            if mode == "bridge_step":
                # Materialize before canonical compute; preserve the FP32 boundary.
                state.checkpoint[slot].copy_(matrix)
                meta[0], meta[3] = pos, 0
                matrix = state.checkpoint[slot].to(op.reference_dtype).clone()
            if mode in ("step", "bridge_step", "verify"):
                rows = kw["projected"][b] if mode == "verify" else kw["projected"][b : b + 1]
                for j, projected in enumerate(rows):
                    raw = raw_row(op.geometry, projected)
                    matrix, history, y = token(
                        op.geometry,
                        weights,
                        matrix,
                        history,
                        raw,
                        projected[
                            op.geometry.conv_dim : op.geometry.conv_dim + op.geometry.value_width
                        ],
                    )
                    (kw["out"][b, j] if mode == "verify" else kw["out"][b]).copy_(y)
                    if mode == "verify":
                        pending.raw[b, j].copy_(raw)
                if mode == "verify":
                    pending.metadata[b].copy_(
                        torch.tensor(
                            [
                                slot,
                                int(meta[5]),
                                int(meta[0]),
                                pos,
                                op.verify_length,
                                1,
                                -1,
                                int(meta[7]),
                            ],
                            dtype=torch.int64,
                        )
                    )
                elif mode == "bridge_step":
                    _canonical(op, state, slot, matrix, history, pos + 1)
                else:
                    state.ring[slot, int(meta[2])].copy_(raw)
                    state.conv_history[slot].copy_(history)
                    meta[1] += 1
                    meta[2] = (meta[2] + 1) % (op.window + 3)
                    meta[3] += 1
                    if int(meta[3]) == op.window:
                        state.checkpoint[slot].copy_(matrix)
                        meta[0] = meta[1]
                        meta[3] = 0
            elif mode == "flush":
                state.checkpoint[slot].copy_(matrix)
                meta[0], meta[3] = meta[1], 0
            elif mode in ("commit", "export_snapshot"):
                end = m if mode == "commit" else k if pending is not None and k <= m else 0
                capture = snap != -1 and k <= m
                if capture and k == 0:
                    _snapshot(state, snap, matrix, history, pos, meta)
                for j in range(end):
                    matrix, history, _ = token(
                        op.geometry, weights, matrix, history, pending.raw[pb, j]
                    )
                    if capture and j + 1 == k:
                        _snapshot(state, snap, matrix, history, pos + k, meta)
                if capture:
                    kw["snapshot_valid"][b] = True
                if mode == "commit":
                    _canonical(op, state, slot, matrix, history, pos + m)
                    pending.metadata[pb, 5:7] = torch.tensor([3, m])
                elif pending is not None:
                    pending.metadata[pb, 5:7] = torch.tensor([2, m])
