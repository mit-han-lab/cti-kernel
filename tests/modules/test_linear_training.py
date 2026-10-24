# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from ctikernels.linear.training import DenseProduct, ProcessGroupExchange, lora_projection_training

from .. import _tensor_parallel_lora as tensor_parallel_lora

SCALING = 0.75


def _leaf(*shape: int, dtype=torch.float64) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype).requires_grad_()


def _reference(x, weight, bias, factors):
    deltas = [x @ a.t() @ b.t() for a, b in zip(factors[::2], factors[1::2], strict=True)]
    return F.linear(x, weight, bias) + SCALING * torch.cat(deltas, -1)


@pytest.mark.parametrize("rows", [0, 5])
def test_lora_projection_training(rows: int) -> None:
    torch.manual_seed(0)
    x, weight, bias = _leaf(rows, 6), torch.randn(7, 6, dtype=torch.float64), torch.randn(7)
    bias = bias.double()
    factors = (_leaf(2, 6), _leaf(3, 2), _leaf(2, 6), _leaf(4, 2))
    base = DenseProduct(weight, bias)

    def op(x, *factors):
        return lora_projection_training(x, base, factors, scaling=SCALING)

    if rows:
        assert torch.autograd.gradcheck(op, (x, *factors))
    with torch.no_grad():
        detached = op(x, *factors)
    y = op(x.reshape(1, rows, 6), *factors)
    torch.testing.assert_close(y.reshape(rows, 7), _reference(x, weight, bias, factors))
    assert torch.equal(detached, y.reshape(rows, 7))
    y.sum().backward()
    assert x.grad.shape == x.shape
    if not rows:
        assert all(not f.grad.any() for f in factors)


def _tp_rank(rank: int, port: int, failures) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        group = ProcessGroupExchange()
        for rows, exchange in ((5, None), (5, group), (2048, group)):
            tensor_parallel_lora.check(torch.device("cpu"), torch.float64, exchange, rows)
    except Exception as error:  # noqa: BLE001 - reported to the parent
        failures.put(f"rank {rank}: {error!r}")
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_partial_sums() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    failures = mp.get_context("spawn").SimpleQueue()
    mp.start_processes(_tp_rank, args=(port, failures), nprocs=2, start_method="spawn")
    assert failures.empty(), failures.get()
