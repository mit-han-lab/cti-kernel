# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Phase microtimings, with restoration outside the event interval.

Explicit flush is outside graphs by ABI. Decode, verify, commit, reject and the
verify/accept/commit transaction use graphs. Each trial owns a fresh sequence;
status settlement occurs only after completion, outside the measured interval.
"""

from dataclasses import fields

import torch

from benchmark.kernel_tuning.gdn_replay_common import Replay
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs


def phase_times(provider, hk, hv, batch, window, budget, *, cute_config=None):
    if provider == "cute" and cute_config is None:
        from ctikernels.gdn.replay.cute import select

        cute_config = select(torch.cuda.get_device_name(), batch, geometry=(hk, hv), window=window)
    weights, x, initial, history = case_inputs(
        hk, hv, batch, batch, window + 16, "ordinary", 23, "cuda"
    )
    weights = weights.args()
    results = {}
    for n in (
        (0,) if cute_config is not None and cute_config.immediate else (0, 1, window - 1, window)
    ):
        results.update(
            _at(
                provider,
                hk,
                hv,
                batch,
                window,
                budget,
                weights,
                x,
                initial,
                history,
                n,
                cute_config,
            )
        )
    return results


def _at(provider, hk, hv, batch, window, budget, weights, x, initial, history, n, cute_config):
    results = {}
    runner = Replay(
        provider,
        hk,
        hv,
        batch,
        window,
        16,
        weights,
        initial,
        history,
        torch.zeros(batch, dtype=torch.int64, device="cuda"),
        cute_config=cute_config,
    )
    for t in range(min(n, window - 1)):
        runner.step(x[t])
    drafts = x[-16:].transpose(0, 1).contiguous().view(batch, -1).view(batch, 16, -1)
    if n == window:
        # Internal full-window fixture, never exposed at an ordinary public boundary.
        g = runner.op.geometry
        raw_row = torch.cat(
            (x[window - 1, :, : g.conv_dim], x[window - 1, :, g.conv_dim + g.value_width :]), -1
        )
        runner.state.ring[:, window - 1].copy_(raw_row)
        runner.state.conv_history[:batch, :2].copy_(runner.state.conv_history[:batch, 1:].clone())
        runner.state.conv_history[:batch, 2].copy_(x[window - 1, :, : g.conv_dim])
        runner.state.metadata[:, 1].add_(1)
        runner.state.metadata[:, 2].fill_(window)
        runner.state.metadata[:, 3].fill_(window)
        runner.pos.add_(1)
    else:
        runner.verify(drafts)
    accepted = torch.tensor(
        [(1, 8, 16, 0)[r % 4] for r in range(batch)], dtype=torch.int32, device="cuda"
    )
    vp = runner.pos[:, None] + torch.arange(16, device="cuda")
    saved = {f.name: getattr(runner.state, f.name).clone() for f in fields(runner.state)}
    pm = runner.pending.metadata.clone()
    raw = runner.pending.raw.clone()
    original_pos = runner.pos.clone()
    op, status = runner.op, runner.status
    op.end(status=status)
    sequence = 1
    result = op.read_status_copy(*status.copy_to_host(), expected_sequence=sequence)
    op.settle(status=status, result=result)

    def restore():
        for name, value in saved.items():
            getattr(runner.state, name).copy_(value)
        runner.pending.metadata.copy_(pm)
        runner.pending.raw.copy_(raw)
        runner.pos.copy_(original_pos)

    def operation(phase):
        op.begin(status=status, layer_weights=(weights,))
        if phase == "step":
            op.step(
                x[n],
                *weights,
                slot_indices=runner.slots,
                positions=runner.pos,
                state=runner.state,
                out=runner.out,
                status=status,
                workspace=runner.workspace,
            )
        elif phase == "flush":
            if n == window:
                _full_flush(runner)
            else:
                op.flush(
                    *weights,
                    slot_indices=runner.slots,
                    state=runner.state,
                    status=status,
                    workspace=runner.phase_workspace,
                )
        else:
            if phase in ("verify", "transaction"):
                op.verify(
                    drafts,
                    *weights,
                    slot_indices=runner.slots,
                    positions=vp,
                    state=runner.state,
                    pending=runner.pending,
                    out=runner.verify_out,
                    status=status,
                )
            if phase == "transaction":
                # Fixed device acceptance histogram; the copy is part of the boundary.
                runner.controls.buffers["accepted_lengths"][0].copy_(accepted)
            if phase in ("commit", "transaction"):
                lengths = (
                    runner.controls.buffers["accepted_lengths"][0]
                    if phase == "transaction"
                    else accepted
                )
                op.commit(
                    *weights,
                    slot_indices=runner.slots,
                    accepted_lengths=lengths,
                    state=runner.state,
                    pending=runner.pending,
                    status=status,
                    workspace=runner.phase_workspace,
                )
            if phase == "reject":
                op.reject(
                    slot_indices=runner.slots,
                    state=runner.state,
                    pending=runner.pending,
                    status=status,
                )
        op.end(status=status)

    phases = (
        ("flush",)
        if n == window
        else ("step", "flush", "verify", "commit", "reject", "transaction")
    )
    for phase in phases:
        budget.check()
        restore()
        sequence += 1
        status.sequence.fill_(sequence)
        operation(phase)
        result = op.read_status_copy(*status.copy_to_host(), expected_sequence=sequence)
        if not result.ok:
            raise RuntimeError("phase warmup failed")
        op.settle(status=status, result=result)
        restore()
        graph = None
        if phase != "flush":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                operation(phase)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(20):
            budget.check()
            restore()
            sequence += 1
            status.sequence.fill_(sequence)
            start.record()
            if graph is None:
                operation(phase)
            else:
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
            result = op.read_status_copy(*status.copy_to_host(), expected_sequence=sequence)
            if not result.ok:
                raise RuntimeError("phase trial failed")
            op.settle(status=status, result=result)
        key = f"{phase}/n={n}"
        ordered = sorted(samples)
        results[key] = {
            "samples_us": samples,
            "median_us": (ordered[9] + ordered[10]) / 2,
            "p95_us": ordered[18],
            "execution": "eager" if graph is None else "graph",
            "includes_begin_end": True,
        }
        if phase == "transaction":
            count = int(accepted.sum().cpu())
            results[key]["accepted_inputs"] = count
            results[key]["us_per_accepted_input"] = (ordered[9] + ordered[10]) / 2 / count
    return results


def _full_flush(runner):
    """Benchmark-only internal n=L entry; phase status is already begun."""
    import triton

    from ctikernels.gdn.replay.triton import _kernels as k
    from ctikernels.gdn.replay.triton import _status as control

    s, op, handle = runner.state, runner.op, runner.status
    code = handle.code
    control.guard[(1,)](handle.record, handle.sequence, handle.poison)
    shared = {
        "C": s.checkpoint,
        "RING": s.ring,
        "M": s.metadata,
        "INDICES": runner.slots,
        "X": code,
        "PR": code,
        "PM": code,
        "ACCEPT": code,
        "SR": code,
        "OFF": code,
        "STATUS": code,
        "POISON": handle.poison,
        "R": runner.r,
        "HK": op.geometry.k_heads,
        "HV": op.geometry.v_heads,
        "L": op.window,
        "Q": op.verify_length,
        "PB": 0,
        "XS": 0,
        "MODE": "flush",
        "CAPTURE": False,
    }
    cw, al, bias, nw = runner.weights
    k.arithmetic[(runner.b, op.geometry.v_heads)](
        **shared,
        CW=cw,
        AL=al,
        BIAS=bias,
        NW=nw,
        OUT=code,
        CANON=code,
        num_warps=8,
        enable_fp_fusion=False,
    )
    k.publish[(runner.b,)](
        **shared,
        H=s.conv_history,
        E=s.pool_epoch,
        SM=s.snapshot_metadata,
        POS=runner.pos,
        VALID=code,
        HISTORY=code,
        BLOCK=triton.next_power_of_2(op.geometry.conv_dim + 2 * op.geometry.v_heads),
        num_warps=8,
    )
