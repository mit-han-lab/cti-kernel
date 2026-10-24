# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU algebra checks for the changed solve/update; these do not execute GPU code."""

import pytest
import torch


def _tf32_rna(x):
    # PTX cvt.rna.tf32.f32: retain ten fraction bits, ties away from zero.
    bits = x.contiguous().view(torch.int32)
    return ((bits + 4096) & -8192).view(torch.float32)


def _parts(x):
    hi = _tf32_rna(x)
    mid = _tf32_rna(x - hi)
    return hi, mid, _tf32_rna(x - hi - mid)


def _product(a, b, accumulator):
    aa, bb = _parts(a), _parts(b)
    # Same term and K-tile order as the six-MMA warp product. CPU matmul's
    # reduction order can differ from hardware, so this checks the algebra/error bound.
    for start in (0, 8):
        for i, j in ((2, 0), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0)):
            accumulator = accumulator + aa[i][:, start : start + 8] @ bb[j][start : start + 8]
    return accumulator


@pytest.mark.parametrize("bf16_solve", [False, True])
@pytest.mark.parametrize("decay", [0.0, 0.01, 0.5])
@pytest.mark.parametrize("length", [1, 17, 63, 64])
def test_blocked_inverse_against_fp64(decay, length, bf16_solve):
    product = (lambda a, b, c: c + _split_product(a, b)) if bf16_solve else _product
    generator = torch.Generator().manual_seed(811 + length)
    k = torch.randn(64, 128, generator=generator)
    k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().float()
    k[length:] = 0
    beta = torch.rand(64, generator=generator)
    g = -torch.arange(64).float() * decay
    lower = torch.tril((k @ k.T) * beta[:, None] * (g[:, None] - g[None, :]).exp(), -1)
    inverse = torch.zeros_like(lower)
    for row in range(64):
        base = row // 16 * 16
        x = -lower[row, base : base + 16].clone()
        x[row - base] = 1
        for pivot in range(row - base - 1, 0, -1):
            x[:pivot] -= x[pivot] * lower[base + pivot, base : base + pivot]
        inverse[row, base : base + 16] = x
    for distance in (1, 2, 3):
        for j in range(4 - distance):
            i = j + distance
            rows, cols = slice(i * 16, (i + 1) * 16), slice(j * 16, (j + 1) * 16)
            acc = torch.zeros(16, 16)
            for mid in range(j + 1, i + 1):
                inner = slice(mid * 16, (mid + 1) * 16)
                acc = product(inverse[rows, inner], lower[inner, cols], acc)
            inverse[rows, cols] = product(-acc, inverse[cols, cols], torch.zeros(16, 16))
    triangular = torch.eye(64, dtype=torch.float64) + lower.double()
    exact = torch.linalg.solve_triangular(
        triangular, torch.eye(64, dtype=torch.float64), upper=False
    )
    tolerance = 8e-6 if bf16_solve else 2e-7
    assert (inverse.double() - exact).abs().max() < tolerance
    assert (triangular @ inverse.double() - torch.eye(64)).abs().max() < tolerance


def test_split_update_keeps_residual_beyond_bf16():
    generator = torch.Generator().manual_seed(812)
    k = torch.randn(64, 128, generator=generator).bfloat16().float()
    weighted_v = torch.randn(64, 32, generator=generator) * torch.linspace(1, 0.01, 64)[:, None]
    hi = weighted_v.bfloat16().float()
    lo = (weighted_v - hi).bfloat16().float()
    exact = weighted_v.double().T @ k.double()
    baseline = hi.T @ k
    improved = baseline + lo.T @ k
    assert (improved.double() - exact).square().mean() < (
        baseline.double() - exact
    ).square().mean() / 10000


def _bf16(x):
    return x.bfloat16().float()


def _split_product(a, b, split_a=True, split_b=True):
    aa = [_bf16(a)]
    bb = [_bf16(b)]
    if split_a:
        aa.append(_bf16(a - aa[0]))
    if split_b:
        bb.append(_bf16(b - bb[0]))
    out = aa[0] @ bb[0]
    for i, lhs in enumerate(aa):
        for j, rhs in enumerate(bb):
            if i or j:
                out = out + lhs @ rhs
    return out


def _chunk_state(k, v, beta, log_decay, initial, *, form, v_fp32=False):
    """CPU numerical model of the operand roundings, independent of CuTe layouts."""
    state = initial.clone()
    for start in range(0, len(k), 64):
        length = min(64, len(k) - start)
        kc = torch.zeros(64, k.shape[1])
        vc = torch.zeros(64, v.shape[1])
        bc, gc = torch.zeros(64), torch.zeros(64)
        kc[:length], vc[:length] = k[start : start + length], v[start : start + length]
        bc[:length], gc[:length] = beta[start : start + length], log_decay[start : start + length]
        gc = gc.cumsum(0)
        effective_k = _bf16(kc) if form == "wy" else _bf16(kc) + _bf16(kc - _bf16(kc))
        gram = effective_k @ effective_k.T
        lower = torch.tril(gram * bc[:, None] * (gc[:, None] - gc[None, :]).exp(), -1)
        inverse = torch.linalg.solve_triangular(torch.eye(64) + lower, torch.eye(64), upper=False)
        if form == "wy":
            w = _bf16(_bf16(inverse) @ _bf16(kc * (bc * gc.exp())[:, None]))
            u = _bf16(_bf16(inverse) @ _bf16(vc * bc[:, None]))
            new_v = u - w @ _bf16(state)
            state = gc[-1].exp() * state + _bf16(kc).T @ _bf16(new_v * (gc[-1] - gc).exp()[:, None])
        else:
            kh = _split_product(kc, state)
            residual = bc[:, None] * ((vc if v_fp32 else _bf16(vc)) - gc.exp()[:, None] * kh)
            new_v = _split_product(inverse, residual)
            # Applying decay before splitting is essential; never weight rounded V.
            weighted_v = new_v * (gc[-1] - gc).exp()[:, None]
            state = gc[-1].exp() * state + _split_product(kc.T, weighted_v)
    return state


@pytest.mark.parametrize("length", [1, 65, 257])
@pytest.mark.parametrize("decay", [0.001, 0.2])
def test_tk_roundings_improve_state_against_serial_fp64(length, decay):
    """Cancellation, slow decay, carried state, partial chunks, and all split terms."""
    errors = {"wy": [], "tk": [], "fp32_v": []}
    for seed in range(5):
        generator = torch.Generator().manual_seed(9301 + seed)
        k = torch.randn(length, 128, generator=generator)
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        v = torch.randn(length, 16, generator=generator)
        beta = torch.randn(length, generator=generator).sigmoid()
        logs = -decay * torch.rand(length, generator=generator)
        initial = 0.2 * torch.randn(128, 16, generator=generator)
        exact = initial.double()
        for row in range(length):
            exact = logs[row].double().exp() * exact
            delta = beta[row].double() * (v[row].double() - k[row].double() @ exact)
            exact = exact + k[row].double()[:, None] * delta[None, :]
        for name in errors:
            got = _chunk_state(
                k,
                v,
                beta,
                logs,
                initial,
                form="wy" if name == "wy" else "tk",
                v_fp32=name == "fp32_v",
            )
            errors[name].append(float((got.double() - exact).abs().max()))
    assert sum(errors["tk"]) < 0.8 * sum(errors["wy"]), errors
    assert sum(errors["fp32_v"]) < 0.02 * sum(errors["wy"]), errors


@pytest.mark.parametrize("length", [1, 63, 65, 257])
@pytest.mark.parametrize("ut_split", [False, True])
def test_wy_fp32_u_preserves_conv_values(length, ut_split):
    """Isolate the observed V rounding source, including beta and tail padding.

    A BF16-rounded convolution cannot recover its residual by splitting later.
    U uses all selected cross terms and FP32 storage, unlike the BF16 ablation.
    """
    generator = torch.Generator().manual_seed(9401 + length)
    errors = {"new": 0.0, "rounded_v": 0.0, "rounded_u": 0.0}
    for start in range(0, length, 64):
        n = min(length - start, 64)
        v = torch.nn.functional.silu(torch.randn(64, 128, generator=generator))
        beta = torch.rand(64, generator=generator)
        beta[n:] = 0
        k = torch.nn.functional.normalize(torch.randn(64, 128, generator=generator), dim=-1)
        lower = torch.tril((k @ k.T) * beta[:, None], -1)
        t = torch.linalg.solve_triangular(torch.eye(64) + lower, torch.eye(64), upper=False)
        # Keep T operand precision identical to isolate V and U rounding effects.
        effective_t = _bf16(t) + (_bf16(t - _bf16(t)) if ut_split else 0)
        exact = effective_t.double() @ (beta[:, None] * v).double()
        new = _split_product(t, beta[:, None] * v, split_a=ut_split)
        rounded_v = _split_product(t, beta[:, None] * _bf16(v), split_a=ut_split)
        for name, value in (("new", new), ("rounded_v", rounded_v), ("rounded_u", _bf16(new))):
            errors[name] += float((value.double() - exact).square().sum())
    assert errors["new"] < errors["rounded_v"] / 10000, errors
    assert errors["new"] < errors["rounded_u"] / 10000, errors


@pytest.mark.parametrize("valid", [1, 17, 63, 64])
@pytest.mark.parametrize("decay", [0.0, 0.01, 0.5])
@pytest.mark.parametrize("bv", [16, 32, 64])
def test_pre_output_equals_legacy_in_fp64(valid, decay, bv):
    """Check the WY identity, masking and scale for arbitrary incoming state."""
    gen = torch.Generator().manual_seed(901 + valid + bv)
    q, k, w = (
        torch.randn(64, 128, generator=gen, dtype=torch.float64) / 128**0.5 for _ in range(3)
    )
    u = torch.randn(64, bv, generator=gen, dtype=torch.float64)
    h = torch.randn(128, bv, generator=gen, dtype=torch.float64)
    q[valid:] = k[valid:] = w[valid:] = 0
    u[valid:] = 0
    g = -torch.arange(64, dtype=torch.float64).clamp(max=valid - 1) * decay
    a = torch.tril((q @ k.T) * (g[:, None] - g[None, :]).exp())
    qe = q * g.exp()[:, None]
    legacy = (qe @ h + a @ (u - w @ h)) / 128**0.5
    qp, op = qe - a @ w, a @ u
    pre = (qp @ h + op) / 128**0.5
    torch.testing.assert_close(pre, legacy, atol=2e-14, rtol=2e-13)
    assert torch.count_nonzero(pre[valid:]) == 0


@pytest.mark.parametrize("qp_split", [0, 1])
def test_pre_output_split_operands_against_fp64(qp_split):
    """Model prepare rounding, then compare to the same stored WY operands."""
    gen = torch.Generator().manual_seed(713)
    q, k = (torch.randn(64, 128, generator=gen) / 128**0.5 for _ in range(2))
    w = (torch.randn(64, 128, generator=gen) * 0.1).bfloat16().float()
    u = torch.randn(64, 32, generator=gen)
    h = torch.randn(128, 32, generator=gen).bfloat16().float()
    g = -torch.arange(64) * 0.01

    def parts(x):
        hi = x.bfloat16().float()
        return hi, (x - hi).bfloat16().float()

    def product(a, b):
        ah, al = parts(a)
        bh, bl = parts(b)
        return ((ah @ bh + ah @ bl) + al @ bh) + al @ bl

    qh, ql = parts(q)
    kh, kl = parts(k)
    a = torch.tril(((qh @ kh.T + ql @ kh.T) + qh @ kl.T) * (g[:, None] - g[None, :]).exp())
    qp = (qh + ql) * g.exp()[:, None] - product(a, w)
    op = product(a, u)
    ph, pl = parts(qp)
    actual = (ph @ h + (pl @ h if qp_split else 0) + op) / 128**0.5
    exact_a = torch.tril(
        (q.double() @ k.double().T) * (g.double()[:, None] - g.double()[None, :]).exp()
    )
    exact = (
        (q.double() * g.double().exp()[:, None]) @ h.double()
        + exact_a @ (u.double() - w.double() @ h.double())
    ) / 128**0.5
    relative_rms = (actual.double() - exact).square().mean().sqrt() / exact.square().mean().sqrt()
    assert relative_rms < (3e-5 if qp_split else 3e-3)
