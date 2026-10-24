# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Device alpha, row strides and capture for the FP32 accumulation epilogue."""

import pytest
import torch

from ctikernels._alpha_gemm import WORKSPACE_BYTES, alpha_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _check(out, a, b, alpha):
    reference = (a.float().t() @ b.float() * alpha).to(out.dtype)
    # One output-dtype ULP at max|ref|, applied to every element. Unlike an
    # elementwise relative bound this stays meaningful at cancellation zeros.
    peak = reference.abs().max()
    ulp = (torch.nextafter(peak, torch.full_like(peak, torch.inf)) - peak).double()
    assert bool(((out.double() - reference.double()).abs() <= ulp).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(37, 1023, 96), (53, 97, 31)])
@pytest.mark.parametrize("strided", [False, True])
def test_alpha_gemm_device_scalar_and_graph(dtype, shape, strided):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(812)
        k, m, n = shape
        # Products and sums fit in FP32 exactly: 12 + 5 + ceil(log2(K))
        # significant bits <= 23. This isolates epilogue rounding from GEMM
        # reduction order; A still has enough significant bits to expose TF32.
        a = (
            torch.randint(1024, 4096, (k, m + 5 if strided else m), device="cuda").float() / 4096
        ).to(dtype)
        a = a[:, 1 : m + 1] if strided else a
        b = (torch.randint(1, 32, (k, n + 3), device="cuda").float() / 32).to(dtype)[:, :n]
        out = torch.empty(m, n, device="cuda", dtype=dtype)
        alpha = torch.tensor(0.375, device="cuda")
        workspace = torch.empty(WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                alpha_gemm(a, b, alpha, out=out, workspace=workspace)
            _check(out, a, b, alpha)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            alpha_gemm(a, b, alpha, out=out, workspace=workspace)
        for value in (0.125, -0.75, 0.0, 0.375):
            alpha.fill_(value)
            graph.replay()
            _check(out, a, b, alpha)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
