# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""A null NCCL window registration falls back to the in-place path (one rank)."""

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_null_window_registration_runs_in_place(tmp_path):
    if dist.is_initialized():
        pytest.skip("needs its own single-rank CPU group")
    from ctikernels import init_pynccl

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        comm = init_pynccl(
            tp_rank=0,
            tp_size=1,
            tp_cpu_group=dist.group.WORLD,
            max_size_bytes=1 << 20,
            _force_window=True,
            _test_null_window=True,
        )
        assert comm.window_mode() == "in_place"
        assert "registration returned no window" in comm.window_reason()
        assert comm.window_bytes() == 0
        assert not comm.get_buffer()
        for dtype in (torch.bfloat16, torch.float32, torch.int32):
            t = torch.arange(1000, device="cuda").to(dtype)
            expected = t.clone()
            comm.all_reduce(t, "max")
            torch.cuda.synchronize()
            assert torch.equal(t, expected)
        del comm
    finally:
        dist.destroy_process_group()
