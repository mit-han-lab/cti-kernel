# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The tensor-parallel contract of the LoRA training ops (ctikernels.linear.training), checked on
any process group: each rank runs its shard, applies the documented reductions, and must match
the unsharded op."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ctikernels.linear.training import DenseProduct, lora_projection_training
from ctikernels.mlp.training import gated_mlp_lora

SCALING = 0.75


def _run(op, x, factors, grad):
    x = x.clone().requires_grad_()
    leaves = [f.clone().requires_grad_() for f in factors]
    y = op(x, leaves)
    y.backward(grad)
    return y.detach(), x.grad, [f.grad for f in leaves]


def _reduce(*tensors):
    for tensor in tensors:
        dist.all_reduce(tensor)


def _same_on_every_rank(*tensors):
    for tensor in tensors:
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        assert all(torch.equal(g, gathered[0]) for g in gathered), "ranks differ"


def check(
    device: torch.device, dtype: torch.dtype, exchange=None, rows: int = 5, **tolerance
) -> None:
    """Without ``exchange`` the caller-side reductions; with it, the op's own (row-chunked and
    overlapped from ``REDUCE_CHUNKS``'s row threshold on)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator().manual_seed(0)

    def tensor(*shape):  # unit-scale outputs: fan-in normalized
        values = torch.randn(*shape, generator=generator, dtype=torch.float64) * shape[-1] ** -0.5
        return values.to(device, dtype)

    hidden, width, out, r = 32, 64, 48, 4
    x, grad_hidden, grad_out = tensor(rows, hidden), tensor(rows, hidden), tensor(rows, out)
    w_gate, w_up, w_down, w_col, w_row = (
        tensor(width, hidden), tensor(width, hidden), tensor(hidden, width),
        tensor(out, hidden), tensor(hidden, out),
    )  # fmt: skip
    mlp = [
        tensor(*s)
        for s in ((r, hidden), (width, r), (r, hidden), (width, r), (r, width), (hidden, r))
    ]
    col = [tensor(r, hidden), tensor(out, r)]
    row = [tensor(r, out), tensor(hidden, r)]
    part = slice(rank * width // world, (rank + 1) * width // world)
    out_part = slice(rank * out // world, (rank + 1) * out // world)

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, **tolerance)

    # Gated MLP: gate_up column-parallel, down row-parallel over the intermediate width.
    def gated(w_gate, w_up, w_down, exchange=None):
        gate_up, down = DenseProduct(torch.cat((w_gate, w_up))), DenseProduct(w_down)
        return lambda x, f: gated_mlp_lora(
            x, gate_up, down, f[:4], f[4:], scaling=SCALING, exchange=exchange
        )

    full = _run(gated(w_gate, w_up, w_down), x, mlp, grad_hidden)
    local = [mlp[0], mlp[1][part], mlp[2], mlp[3][part], mlp[4][:, part], mlp[5]]
    y, dx, grads = _run(
        gated(w_gate[part], w_up[part], w_down[:, part], exchange), x, local, grad_hidden
    )
    if exchange is None:
        _reduce(y, dx)  # partial: output, x
    else:
        _same_on_every_rank(y, dx)
    _reduce(grads[0], grads[2], grads[5])  # partial: replicated A_g/A_u/B_d
    close(y, full[0])
    close(dx, full[1])
    expected = [
        full[2][0],
        full[2][1][part],
        full[2][2],
        full[2][3][part],
        full[2][4][:, part],
        full[2][5],
    ]
    for actual, want in zip(grads, expected, strict=True):
        close(actual, want)

    def projection(weight, reduce="input_grad", exchange=None, bias=None):
        return lambda x, f: lora_projection_training(
            x, DenseProduct(weight, bias), f, scaling=SCALING, exchange=exchange, reduce=reduce
        )

    # Column-parallel: W's output rows sharded, x replicated, A replicated, B sharded.
    full = _run(projection(w_col), x, col, grad_out)
    y, dx, (grad_a, grad_b) = _run(
        projection(w_col[out_part], "input_grad", exchange),
        x,
        [col[0], col[1][out_part]],
        grad_out[:, out_part],
    )
    if exchange is None:
        _reduce(dx)  # partial: x
    else:
        _same_on_every_rank(dx)
    _reduce(grad_a)  # partial: replicated A
    close(y, full[0][:, out_part])
    close(dx, full[1])
    close(grad_a, full[2][0])
    close(grad_b, full[2][1][out_part])

    # Row-parallel: W's input columns sharded, x sharded, A sharded, B replicated; the output
    # gradient is the reduced output's, replicated.
    x_row = tensor(rows, out)
    # A replicated bias joins the output once: the op adds it after its own reduction.
    bias = tensor(hidden) if exchange is not None else None
    full = _run(projection(w_row, bias=bias), x_row, row, grad_hidden)
    y, dx, (grad_a, grad_b) = _run(
        projection(w_row[:, out_part], "output", exchange, bias),
        x_row[:, out_part],
        [row[0][:, out_part], row[1]],
        grad_hidden,
    )
    if exchange is None:
        _reduce(y)  # partial: output
    else:
        _same_on_every_rank(y)
    _reduce(grad_b)  # partial: replicated B
    close(y, full[0])
    close(dx, full[1][:, out_part])
    close(grad_a, full[2][0][:, out_part])
    close(grad_b, full[2][1])
