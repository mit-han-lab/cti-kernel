# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Token epilogue parity and CUDA-graph replay against the Triton backend."""

import pytest
import torch
import triton

from ctikernels.moe.cute._provider import available
from ctikernels.moe.cute._token import token_backward
from ctikernels.moe.triton._kernels import _bwd_token_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def capability():
    cap = torch.cuda.get_device_capability()
    if cap not in ((12, 0), (12, 1)) or not available():
        pytest.skip("sm_120/121 and CuTe DSL 4.8.0 required")
    return cap


def inputs(t, k, shared, *, h=2048, nt=16, wide=False):
    torch.manual_seed(t + k)
    e, kp1 = 256, k + shared
    rows = t * kp1
    inv = torch.randperm(rows, device="cuda").to(torch.int32).reshape(t, kp1)
    sel = torch.randint(e, (t + 1, k), device="cuda", dtype=torch.int64 if wide else torch.int32)[
        1:
    ]
    # A global expert map marks remote slots; these router rows must still
    # participate in the selected-softmax derivative.
    mapping = torch.arange(e, device="cuda", dtype=torch.int32)
    mapping[::3] = -1
    inv[:, :k].masked_fill_(mapping[sel.long()] < 0, -1)
    dw = torch.randn(rows, nt, device="cuda") / nt**0.5
    ws = torch.empty(t + 1, kp1, device="cuda")[1:]
    ws[:, :k] = torch.softmax(torch.randn(t, k, device="cuda"), -1)
    if shared:
        ws[:, k] = torch.rand(t, device="cuda")
    dxp = torch.randn(rows, h, device="cuda").bfloat16()
    r = torch.randn(e, h, device="cuda").bfloat16()
    sg = torch.randn(1, h, device="cuda").bfloat16() if shared else None
    return dxp, inv.flatten(), dw, ws.flatten(), sel, r, sg


def reference(args, valid):
    dxp, inv, dw, ws, sel, r, sg = args
    t, k = sel.shape
    h = r.shape[1]
    out = torch.empty(t, h, dtype=torch.bfloat16, device="cuda")
    _bwd_token_kernel[(t, h // 64)](
        dxp,
        inv,
        dw,
        ws,
        sel,
        r,
        r if sg is None else sg,
        out,
        inv if valid is None else valid,
        t,
        h,
        k,
        k + (sg is not None),
        dw.shape[1],
        triton.next_power_of_2(k),
        valid is not None,
        64,
        HAS_SHARED=sg is not None,
        num_warps=4,
    )
    return out


def check(got, want, valid):
    assert torch.isfinite(got).all() and torch.isfinite(want).all()
    error = (got.float() - want.float()).abs()
    assert error.norm() < 1e-3 * want.float().norm().clamp_min(1e-30)
    # Max-absolute error at the reference's magnitude complements the norm
    # bound. Per-element relative bounds are unstable near cancellation when
    # the two compilers contract/reduce router products differently.
    magnitude = want.abs().max()
    ulp = (
        torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))).float()
        - magnitude.float()
    )
    assert error.max() <= 2 * ulp
    if valid is not None:
        n = max(0, min(len(got), valid.item()))
        assert not got[n:].count_nonzero() and not torch.signbit(got[n:]).any()


@pytest.mark.parametrize("t", (1, 7, 64, 1000, 4097))
@pytest.mark.parametrize("k", (2, 8))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("masked", (False, True))
def test_token(capability, t, k, shared, masked):
    args = inputs(t, k, shared)
    valid = torch.tensor(t // 2, dtype=torch.int32, device="cuda") if masked else None
    if masked:
        dxp, inv, dw, ws, sel, _, _ = args
        n = t // 2
        dead = inv.reshape(t, k + shared)[n:]
        rows = dead[dead >= 0].long()
        dw[rows] = float("nan")
        dxp[rows] = float("nan")
        ws.reshape(t, k + shared)[n:] = float("nan")
        sel[n:] = -1  # Invalid addresses must not be dereferenced by dead warps.
        dead.fill_(-1)
    got = torch.empty(t, 2048, dtype=torch.bfloat16, device="cuda")
    token_backward(*args, got, valid, capability=capability)
    check(got, reference(args, valid), valid)


@pytest.mark.parametrize("k", range(1, 9))
@pytest.mark.parametrize("h", (64, 192, 320))
def test_tail_and_index_dtype(capability, k, h):
    args = inputs(7, k, True, h=h, nt=3, wide=True)
    got = torch.empty(7, h, dtype=torch.bfloat16, device="cuda")
    token_backward(*args, got, None, capability=capability)
    check(got, reference(args, None), None)


@pytest.mark.parametrize("shared", (False, True))
def test_graph_replay(capability, shared):
    args = inputs(7, 8, shared)
    got = torch.empty(7, 2048, dtype=torch.bfloat16, device="cuda")
    valid = torch.tensor(7, dtype=torch.int32, device="cuda")

    def run():
        token_backward(*args, got, valid, capability=capability)

    # Compile on the current stream, capture on another stream: cache entries
    # must use TVM FFI's environment stream, not the original compilation stream.
    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    for n in (0, 3, 7, -(2**31), 2**31 - 1):
        valid.fill_(n)
        args[0].normal_()
        args[2].normal_()
        graph.replay()
        check(got, reference(args, valid), valid)


@pytest.mark.parametrize("shared", (False, True))
def test_remote_selected_row_contributes(capability, shared):
    args = inputs(1, 2, shared, h=64, nt=3)
    dxp, inv, dw, ws, sel, r, sg = args
    dxp.zero_()
    inv.fill_(-1)
    inv[0] = 0
    dw.fill_(float("nan"))
    dw[0] = torch.tensor([4, -1, -1], device="cuda")
    ws[:2] = 0.5
    sel[0, 0], sel[0, 1] = 0, 1
    r[0].fill_(2)
    r[1].fill_(6)
    if shared:
        inv[2] = 1
        dw[1] = torch.tensor([8, -2, -2], device="cuda")
        ws[2] = 0.5
        sg.fill_(3)
    got = torch.empty(1, 64, dtype=torch.bfloat16, device="cuda")
    token_backward(*args, got, None, capability=capability)
    # Local dw=2, remote dw=0: dz=(0.5,-0.5), router=-2. Shared adds 3.
    assert torch.equal(got, torch.full_like(got, 1 if shared else -2))
    assert torch.equal(got, reference(args, None))


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("n", (0, -(2**31)))
def test_all_nan_masked(capability, shared, n):
    args = inputs(7, 8, shared)
    for tensor in args:
        if tensor is not None:
            tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
    valid = torch.tensor(n, dtype=torch.int32, device="cuda")
    got = torch.empty(7, 2048, dtype=torch.bfloat16, device="cuda")
    token_backward(*args, got, valid, capability=capability)
    assert torch.equal(got, torch.zeros_like(got))
    assert not torch.signbit(got).any()
