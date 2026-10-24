# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Subprocess isolation for the concurrent-stream forward-progress GPU test."""

import os

import torch

from ._dflash import Fixture


def main():
    mode = os.environ["CTI_DRAFT_MERGE"]
    torch.backends.cuda.matmul.allow_tf32 = False
    for key, value in {
        "INLINE": 1,
        "SHORT": 64,
        "SPLITS": 4 if mode == "distributed" else 65,
        "ROWS": 32,
        "KEY_TILE": 64,
        "MIN_CHUNK": 64,
        "KEY_GROUPS": 2,
        "STAGES": 2,
    }.items():
        os.environ[f"CTI_DRAFT_{key}"] = str(value)
    batch = 1 if mode == "distributed" else 4
    fixtures = [
        Fixture(batch, 9, 16, 16, 4, causal, None, c=8192, provider="cute")
        for causal in (False, True)
    ]
    assert (
        fixtures[0].op._implementation._counters.data_ptr()
        != fixtures[1].op._implementation._counters.data_ptr()
    )
    streams = [torch.cuda.Stream() for _ in range(3)]
    graphs, expected = [], []
    for f, stream in zip(fixtures, streams, strict=False):
        assert f.op._implementation.merge_mode == mode
        f.stage([7000] * batch, [0] if batch == 1 else [0, -1, 2, 3])
        expected.append(f.oracle())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            f.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            f.op.prepare_and_run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
        graphs.append(graph)
    # Independent large GEMMs keep most SMs busy while both attention grids run.
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(a)
    load = torch.cuda.CUDAGraph()
    streams[2].wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(streams[2]):
        torch.mm(a, a, out=result)
    streams[2].synchronize()
    with torch.cuda.graph(load, stream=streams[2]):
        for _ in range(256):
            torch.mm(a, a, out=result)
    torch.cuda.synchronize()
    for _ in range(3):
        with torch.cuda.stream(streams[2]):
            load.replay()
        for stream, graph in zip(streams, graphs, strict=False):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for f, reference in zip(fixtures, expected, strict=True):
        torch.testing.assert_close(f.out.float(), reference, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(f.op._implementation._arrivals) == 0


if __name__ == "__main__":
    main()
