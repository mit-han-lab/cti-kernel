# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Exchange on the CPU: transport selection, the communicator and local paths, validation."""

import pytest
import torch

from ctikernels.collective.inference import Exchange


class _Communicator:
    def __init__(self, world):
        self.world, self.calls = world, []

    def all_reduce(self, x, op):
        self.calls.append((tuple(x.shape), op))
        x.mul_(self.world)  # every rank holds the same partial here


def _norm_stub(monkeypatch, calls):
    def add_rms_norm_(x, residual, weight, eps, *, zero_centered):
        calls.append((eps, zero_centered))
        residual.add_(x)
        x.copy_(residual * 2)
        return x, residual

    monkeypatch.setattr("ctikernels.collective._common.add_rms_norm_", add_rms_norm_)


def make(**kwargs):
    args = {"rank": 0, "world_size": 1, "cpu_group": None, "device": "cpu",
            "max_tokens": 8, "max_hidden": 16}  # fmt: skip
    return Exchange(**(args | kwargs))


@pytest.mark.parametrize("transport", ["local", "nccl"])
def test_unfused_paths(monkeypatch, transport):
    calls = []
    _norm_stub(monkeypatch, calls)
    comm = _Communicator(2) if transport == "nccl" else None
    world = 2 if transport == "nccl" else 1
    xch = make(world_size=world, transport=transport, communicator=comm)
    assert xch.transport == transport and xch.epilogue is None
    assert xch.host_bytes == xch.device_bytes == 0
    x = torch.ones(3, 16)
    assert xch.all_reduce(x) is x and torch.equal(x, torch.full((3, 16), float(world)))
    out, res = xch.add_rms_norm(torch.ones(3, 16), torch.ones(3, 16), torch.ones(16), 1e-5, False)
    assert calls == [(1e-5, False)] and torch.equal(res, torch.full((3, 16), world + 1.0))
    partial, residual, dst = torch.ones(2, 16), torch.full((2, 16), 3.0), torch.empty(2, 16)
    xch.reduce(partial, tile=(16, 16), out=dst, residual=residual)
    assert torch.equal(dst, torch.full((2, 16), world + 3.0))
    if comm is not None:
        assert [shape for shape, _ in comm.calls] == [(3, 16), (3, 16), (2, 16)]
    xch.close()
    with pytest.raises(RuntimeError, match="after close"):
        xch.all_reduce(x)


def test_zero_rows_do_nothing():
    comm = _Communicator(2)
    xch = make(world_size=2, transport="nccl", communicator=comm)
    empty = torch.empty(0, 16)
    xch.all_reduce(empty)
    xch.reduce(empty, tile=(16, 16))
    assert comm.calls == []


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"rank": 2, "world_size": 2}, ValueError),
        ({"max_tokens": 0}, ValueError),
        ({"transport": "ipc"}, ValueError),
        ({"world_size": 2, "transport": "nccl"}, ValueError),
        ({"world_size": 2, "transport": "local"}, ValueError),
        ({"dtype": torch.float32}, TypeError),
    ],
)
def test_rejects(kwargs, error):
    with pytest.raises(error):
        make(**kwargs)


def test_reduce_shapes():
    xch = make()
    with pytest.raises(ValueError, match="same shape"):
        xch.reduce(torch.ones(2, 16), tile=(1, 16), out=torch.ones(3, 16))
    with pytest.raises(ValueError, match="needs the residual"):
        xch.reduce(torch.ones(2, 16), tile=(1, 16), norm=(torch.ones(16), 1e-6, True))


@pytest.mark.parametrize("transport", ["host", "peer"])
def test_routed_calls_reduce_across_ranks(monkeypatch, transport):
    """Calls above nccl_above on a publishing transport go through the communicator."""
    _norm_stub(monkeypatch, [])
    comm = _Communicator(2)
    xch = Exchange.__new__(Exchange)
    vars(xch).update(
        rank=0, world=2, device=torch.device("cpu"), max_tokens=8, max_hidden=16,
        dtype=torch.float32, communicator=comm, transport=transport, nccl_above=16,
        host_bytes=0, device_bytes=0, _host=None, _peer=None, _closed=False,
    )  # fmt: skip
    x = torch.ones(4, 16)
    assert torch.equal(xch.all_reduce(x), torch.full((4, 16), 2.0))
    out, res = xch.add_rms_norm(torch.ones(4, 16), torch.ones(4, 16), torch.ones(16), 1e-6)
    assert torch.equal(res, torch.full((4, 16), 3.0))
    xch.reduce(torch.ones(4, 16), tile=(16, 16))
    assert len(comm.calls) == 3


def test_cross_host_group_rejects_shared_memory(monkeypatch):
    monkeypatch.setattr("ctikernels.collective._common._one_host", lambda world, group: False)
    with pytest.raises(ValueError, match="one host"):
        make(world_size=2, transport="host", communicator=_Communicator(2))
    xch = make(world_size=2, communicator=_Communicator(2))
    assert xch.transport == "nccl"
