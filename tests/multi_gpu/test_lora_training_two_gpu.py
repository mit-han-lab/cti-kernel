# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The LoRA training ops' tensor-parallel contract on two GPUs (NCCL reductions)."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ctikernels.linear.training import CommStreamExchange, ProcessGroupExchange

from .. import _tensor_parallel_lora as tensor_parallel_lora

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")


class _Synchronous:
    """A blocking exchange (all_reduce only), as ctikernels.collective's Exchange is."""

    def __init__(self, group):
        self.group, self.world = group, group.world

    def all_reduce(self, x):
        return self.group.all_reduce(x)


def _rank(rank: int, port: int, failures) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    try:
        group = ProcessGroupExchange()
        streamed = CommStreamExchange(_Synchronous(group))
        cases = ((5, None), (5, group), (2048, group), (2048, streamed))
        for rows, exchange in cases:
            tensor_parallel_lora.check(
                torch.device("cuda", rank), torch.float32, exchange, rows, rtol=1e-4, atol=1e-4
            )
    except Exception as error:  # noqa: BLE001 - reported to the parent
        failures.put(f"rank {rank}: {error!r}")
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_partial_sums() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    failures = mp.get_context("spawn").SimpleQueue()
    mp.start_processes(_rank, args=(port, failures), nprocs=2, start_method="spawn")
    assert failures.empty(), failures.get()
