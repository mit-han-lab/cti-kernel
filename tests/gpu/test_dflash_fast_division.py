# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""div_rn_by (rcp.rn + two Markstein corrections) must equal div.rn.f32 bitwise."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _kernel():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32

    from ctikernels.sampling._dflash.cute.kernels import div_rn_by, fp, row_divisor

    @cute.kernel
    def kernel(a: cute.Tensor, b: cute.Tensor, fast: cute.Tensor, exact: cute.Tensor, n: Int32):
        bx, by, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        index = bx * 256 + tid
        if index < n:
            divisor = row_divisor(b[by])
            value = a[index]
            fast[by * n + index] = div_rn_by(value, divisor)
            exact[by * n + index] = fp(value, b[by], "div")

    @cute.jit
    def launch(a: cute.Tensor, b: cute.Tensor, fast: cute.Tensor, exact: cute.Tensor, n: Int32):
        kernel(a, b, fast, exact, n).launch(
            grid=((n + 255) // 256, b.shape[0], 1), block=(256, 1, 1)
        )

    return launch, cutlass, Float32


def _run(a, b):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    launch, _, _ = _kernel()
    n = a.numel()
    fast = torch.empty(b.numel() * n, dtype=torch.float32, device="cuda")
    exact = torch.empty_like(fast)
    args = [from_dlpack(t) for t in (a, b, fast, exact)]
    compiled = cute.compile(launch, *args, n)
    compiled(*args, n)
    torch.cuda.synchronize()
    return fast.view(torch.int32), exact.view(torch.int32)


def _divisors():
    generator = torch.Generator().manual_seed(7)
    mantissa = torch.randint(0, 2**23, (48,), generator=generator, dtype=torch.int32)
    exponent = torch.randint(127 - 30, 127 + 30, (48,), generator=generator, dtype=torch.int32)
    random = ((exponent << 23) | mantissa).view(torch.float32)
    # Temperatures, softmax totals, all-ones mantissas and range edges.
    special = torch.tensor(
        [
            0.7,
            0.6,
            1.0,
            1.5,
            2.0,
            0.05,
            3.0,
            248320.0,
            151936.0,
            1.0000001,
            2.0**-30,
            2.0**30,
            2.0**-31,
            2.0**31,
            (2**24 - 1) * 2.0**-23,
            (2**24 - 1) * 2.0**-40,
            1e-38,
            2.0**-126,
            3 * 2.0**-126,
            2.0**-127,
            2.0**-140,
            2.0**-149,
            2.0**127,
            3.4e38,
            0.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([random, special]).cuda()


@pytest.mark.parametrize("exponent_band", [(1, 37), (37, 100), (100, 160), (160, 217), (217, 255)])
def test_all_mantissas_against_div_rn(exponent_band):
    lo, hi = exponent_band
    # Every mantissa for a spread of exponents, both signs, plus zeros/inf/NaN.
    mantissa = torch.arange(0, 2**23, 7, dtype=torch.int32)
    exponents = torch.arange(lo, hi, max(1, (hi - lo) // 6), dtype=torch.int32)
    bits = (exponents[:, None] << 23 | mantissa[None, :]).reshape(-1)
    extra = torch.tensor([0, 1, 2**23 - 1, 0x7F800000, 0x7FC00000], dtype=torch.int32)
    bits = torch.cat([bits, extra])
    values = torch.cat([bits, bits | torch.tensor(-(2**31), dtype=torch.int32)]).view(torch.float32)
    values = values.cuda()
    # A few divisors per launch bounds the memory to about 1 GB.
    for divisors in _divisors().split(8):
        fast, exact = _run(values, divisors)
        nan = torch.isnan(exact.view(torch.float32))
        assert torch.equal(torch.isnan(fast.view(torch.float32)), nan)
        assert torch.equal(fast[~nan], exact[~nan])
        del fast, exact, nan
