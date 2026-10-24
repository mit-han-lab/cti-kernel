# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Small deterministic CPU-generated replay fixtures, shared by CPU/GPU tests."""

from dataclasses import fields

import torch

from ctikernels.gdn import GatedDeltaNetReplay, ReplayPending, ReplayState


def case(*, device="cpu", hk=1, hv=4, window=4, q=8, batch=2, oracle=False, provider=None):
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        window=window,
        verify_length=q,
        device=device,
        checks="strict",
        provider=provider,
    )
    g = op.geometry
    gen = torch.Generator().manual_seed(23)
    weights = (
        torch.randn(g.conv_dim, 4, generator=gen).to(torch.bfloat16) * 0.25,
        torch.randn(hv, generator=gen) * 0.1 - 1,
        torch.randn(hv, generator=gen) * 0.1,
        torch.randn(128, generator=gen) * 0.1 + 1,
    )
    weights = tuple(w.to(device) for w in weights)
    buffers = {}
    for name, spec in op.state_spec(num_slots=batch + 2, num_snapshot_rows=batch + 3).items():
        dtype = torch.float64 if oracle and name == "checkpoint" else spec.dtype
        buffers[name] = torch.zeros(spec.shape, dtype=dtype, device=device)
    state = op.create_state(buffers=buffers)
    pending = op.create_pending(
        raw=torch.zeros(batch, q, g.conv_dim + 2 * hv, dtype=torch.bfloat16, device=device),
        metadata=torch.zeros(batch, 8, dtype=torch.int64, device=device),
    )
    controls = graph_buffers(op, batch=batch)
    status = controls.status(0)
    status.sequence.fill_(1)
    op.begin(status=status, layer_weights=(weights,))
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    positions = torch.full((batch,), 2**33 + 7, dtype=torch.int64, device=device)
    op.initialize(slot_indices=slots, next_positions=positions, state=state, status=status)
    return op, state, pending, weights, slots, positions, status


def projected(op, batch, *, length=None, seed=47):
    shape = (
        (batch, op.geometry.proj_width)
        if length is None
        else (batch, length, op.geometry.proj_width)
    )
    return (
        (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.3)
        .to(torch.bfloat16)
        .to(op.device)
    )


def clone(obj, device=None):
    values = [
        getattr(obj, f.name).clone().to(device or getattr(obj, f.name).device) for f in fields(obj)
    ]
    return (ReplayState if isinstance(obj, ReplayState) else ReplayPending)(*values)


def live_ring(state):
    """Ring with dead rows zeroed: only n>0 slots' 3 seeds + n records are defined (§4).

    Providers may leave different bytes in dead rows (CuTe reseeds an empty ring
    from authoritative history; the portable providers do not).
    """
    ring = state.ring
    if ring.shape[1] == 0:
        return ring
    meta = state.metadata[: ring.shape[0]].to(ring.device)
    n, cursor, rows = meta[:, 3:4], meta[:, 2:3], ring.shape[1]
    offset = (torch.arange(rows, device=ring.device) - cursor + n + 3) % rows
    live = (offset < n + 3) & (n > 0)
    return torch.where(live[..., None], ring, torch.zeros_like(ring))


def same(a, b):
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name == "ring" and isinstance(a, ReplayState):
            x, y = live_ring(a), live_ring(b)
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), f.name


def decode_steps(op, state, weights, slots, pos, status, count, *, seed=100):
    out = torch.empty(
        slots.numel(), op.geometry.value_width, dtype=torch.bfloat16, device=op.device
    )
    for i in range(count):
        x = projected(op, slots.numel(), seed=seed + i)
        op.step(x, *weights, slot_indices=slots, positions=pos, state=state, out=out, status=status)
        pos.add_(1)
    return out


def verify(op, state, pending, weights, slots, pos, status, *, seed=200):
    x = projected(op, slots.numel(), length=op.verify_length, seed=seed)
    out = torch.empty(
        slots.numel(),
        op.verify_length,
        op.geometry.value_width,
        dtype=torch.bfloat16,
        device=op.device,
    )
    positions = pos[:, None] + torch.arange(op.verify_length, device=op.device)
    op.verify(
        x,
        *weights,
        slot_indices=slots,
        positions=positions,
        state=state,
        pending=pending,
        out=out,
        status=status,
    )
    return x, out


def graph_buffers(op, *, batch=1, depth=2, layers=1):
    buffers = {}
    for name, spec in op.graph_buffer_spec(
        batch_size=batch, depth=depth, num_layers=layers
    ).items():
        buffers[name] = torch.full(
            spec.shape, spec.initial_value, dtype=spec.dtype, device=op.device
        )
    return op.bind_graph_buffers(buffers=buffers)


def finish(op, status):
    """Tests only: serving already has a completed asynchronous host mirror."""
    op.end(status=status)
    result = op.read_status_copy(
        *status.copy_to_host(), expected_sequence=int(status.sequence.cpu()[0])
    )
    op.settle(status=status, result=result)
    return result


def restart(op, status):
    result = finish(op, status)
    status.sequence.add_(1)
    op.begin(status=status)
    return result


def zero_rows(op, state, pending, weights, slots, pos, status, *, capturable=False):
    """Every per-row phase on B=0 rows (views of the live controls): no work, no error.
    ``capturable`` keeps only the phases ``op`` admits under CUDA graph capture."""
    g, q, none = op.geometry, op.verify_length, slots[:0]
    pending = op.create_pending(raw=pending.raw[:0], metadata=pending.metadata[:0])

    def work(phase):
        return {
            name: torch.full(
                spec.shape, spec.initial_value or 0, dtype=spec.dtype, device=op.device
            )
            for name, spec in op.workspace_spec(phase, batch_size=0).items()
            if name in ("arrivals", "controls", "sections")
        }

    calls = {
        "step": lambda: op.step(
            projected(op, 0), *weights, slot_indices=none, positions=pos[:0], state=state,
            out=torch.empty(0, g.value_width, dtype=torch.bfloat16, device=op.device),
            status=status, workspace=work("step"),
        ),
        "verify": lambda: op.verify(
            projected(op, 0, length=q), *weights, slot_indices=none,
            positions=torch.zeros(0, q, dtype=torch.int64, device=op.device), state=state,
            pending=pending,
            out=torch.empty(0, q, g.value_width, dtype=torch.bfloat16, device=op.device),
            status=status,
        ),
        "commit": lambda: op.commit(
            *weights, slot_indices=none,
            accepted_lengths=torch.zeros(0, dtype=torch.int32, device=op.device),
            state=state, pending=pending, status=status, workspace=work("commit"),
        ),
        "reject": lambda: op.reject(slot_indices=none, state=state, pending=pending, status=status),
        "flush": lambda: op.flush(
            *weights, slot_indices=none, state=state, status=status, workspace=work("flush")
        ),
        "handoff": lambda: op.handoff(
            canonical_state=torch.zeros(0, g.v_heads, 128, 128, device=op.device),
            history=torch.zeros(0, 3, g.conv_dim, dtype=torch.bfloat16, device=op.device),
            next_positions=pos[:0], slot_indices=none, state=state, status=status,
        ),
    }  # fmt: skip
    for phase, call in calls.items():
        if not capturable or phase in op.capabilities["capture"]:
            call()
