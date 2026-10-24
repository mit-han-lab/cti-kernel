# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Direct expert GEMM qualification, independent FP32 oracles and padding sentinels."""

import pytest
import torch

from ctikernels.moe.cute._provider import CuteBackend, available, configs

from ..modules._moe_layout_reference import build_slot_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.is_available() and torch.cuda.get_device_capability() not in ((12, 0), (12, 1)))
    or not available(),
    reason="sm_120/121 and nvidia-cutlass-dsl 4.8.0 required",
)


@pytest.fixture(autouse=True)
def fp32_reference():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _close(got, want):
    assert torch.isfinite(got).all()
    error = (got.float() - want.float()).norm()
    scale = want.float().norm().clamp_min(1e-20)
    assert error / scale < 1e-2, f"relative norm error {(error / scale).item():.6g}"


def _case(t, h, ni, nis, e, override="", monkeypatch=None, all_masked=False):
    if monkeypatch is not None:
        monkeypatch.setenv("CTI_MOE_CUTE_CONFIG", override)
    torch.manual_seed(171 + t)
    dev = torch.device("cuda")
    be = CuteBackend(dev, ni, nis, h)
    kp1 = 9 if nis else 8
    # Leave half the local experts empty and mask some slots as remote.
    se = torch.randint(0, max(e // 2, 1), (t, kp1), device="cpu")
    se.reshape(-1)[::11] = e + 1
    if nis:
        se[:, -1] = e
    if all_masked:
        se.fill_(e + 1)
    bm = be.block_m(dev, t, ni, nis)
    lay = build_slot_layout(se.reshape(-1).cuda(), e, bm)
    slots = lay.sorted_slot.cpu().long()
    experts = lay.block_expert.cpu().repeat_interleave(bm)
    nmax = max(ni, nis)
    nt = be.dw_tiles(dev, t, ni, nis)
    npad = slots.numel()

    def rand(*shape, scale=1.0):
        return (torch.randn(shape, device=dev) * scale).bfloat16()

    def output(width, dtype=torch.bfloat16):
        return torch.full((npad, width), float("nan"), dtype=dtype, device=dev)

    x, dy = rand(t, h), rand(t, h)
    w13, w2 = rand(e, 2 * ni, h, scale=h**-0.5), rand(e, h, ni, scale=ni**-0.5)
    s13 = rand(2 * nis, h, scale=h**-0.5) if nis else None
    s2 = rand(h, nis, scale=nis**-0.5) if nis else None
    ws = torch.rand(t * kp1, device=dev)
    a, gs, us = output(nmax), output(nmax), output(nmax)
    y, dx = output(h), output(h)
    gg, dw = output(2 * nmax), output(nt, torch.float32)
    # Each GEMM gets independent operands so one kernel cannot hide another's error.
    ain, gin = rand(npad, nmax), rand(npad, 2 * nmax)
    invalid = (slots < 0).to(dev)
    ain[invalid], gin[invalid] = float("nan"), float("nan")
    be.gate_up(x, w13, s13, lay, a, gs, us, ni=ni, nis=nis, kp1=kp1, save=True)
    be.down(ain, w2, s2, lay, y, ni=ni, nis=nis)
    be.dx_pair(gin, w13, s13, lay, dx, ni=ni, nis=nis)
    be.bwd_slot(
        x, dy, w13, s13, w2, s2, lay, ws, gs, us, gg, dw, ni=ni, nis=nis, kp1=kp1, saved=False
    )
    nt_saved = be.dw_tiles(dev, t, ni, nis, saved=True)
    gg_saved, dw_saved = output(2 * nmax), output(nt_saved, torch.float32)
    be.bwd_slot(
        x,
        dy,
        w13,
        s13,
        w2,
        s2,
        lay,
        ws,
        gs,
        us,
        gg_saved,
        dw_saved,
        ni=ni,
        nis=nis,
        kp1=kp1,
        saved=True,
    )
    # Save=False also must work with None activation buffers.
    a_unsaved = output(nmax)
    be.gate_up(x, w13, s13, lay, a_unsaved, None, None, ni=ni, nis=nis, kp1=kp1, save=False)
    for tensor in (a, a_unsaved, gs, us, y, dx, gg, dw, gg_saved, dw_saved):
        assert torch.isnan(tensor[invalid]).all(), "padding was overwritten"

    # Reference calculations use gathered rows and only live expert widths.
    for expert in range(e + bool(nis)):
        rr_cpu = torch.where((experts == expert) & (slots >= 0))[0]
        if not rr_cpu.numel():
            continue
        rr, ss = rr_cpu.to(dev), slots[rr_cpu].to(dev)
        tok = ss // kp1
        n = nis if expert == e else ni
        ww13 = s13 if expert == e else w13[expert]
        ww2 = s2 if expert == e else w2[expert]
        g = (x[tok].float() @ ww13[:n].float().T).bfloat16()
        u = (x[tok].float() @ ww13[n:].float().T).bfloat16()
        # Unsloth's rounding: BF16 SiLU, then BF16 SiLU*up.
        ab = ((g.float() * torch.sigmoid(g.float())).bfloat16().float() * u.float()).bfloat16()
        _close(gs[rr, :n], g)
        _close(us[rr, :n], u)
        _close(a[rr, :n], ab)
        # On the kernel's own BF16 gate/up the activation is torch's exact F.silu, bit for bit.
        silu = torch.nn.functional.silu(gs[rr, :n]) * us[rr, :n]
        assert torch.equal(a[rr, :n], silu)
        assert torch.equal(a[rr, :n], a_unsaved[rr, :n])
        _close(y[rr], (ain[rr, :n].float() @ ww2.float().T).bfloat16())
        gcat = torch.cat((gin[rr, :n], gin[rr, nmax : nmax + n]), dim=1).float()
        _close(dx[rr], (gcat @ ww13.float()).bfloat16())
        q = dy[tok].float() @ ww2.float()
        # Use forward's saved BF16 values to isolate backward math from tiny
        # accumulation-order differences at forward rounding boundaries.
        gf, uf = gs[rr, :n].float(), us[rr, :n].float()
        sig = torch.sigmoid(gf)
        sb = (gf * sig).bfloat16().float()
        af = (sb * uf).bfloat16().float()
        da = q * ws[ss, None]
        dg = (da * uf * sig * (1 + gf * (1 - sig))).bfloat16()
        du = (da * sb).bfloat16()
        _close(gg[rr, :n], dg)
        _close(gg[rr, nmax : nmax + n], du)
        width = nmax // nt
        part = (q * af).reshape(-1, n // width, width).sum(dim=-1)
        _close(dw[rr, : n // width], part)
        # Forward and recompute must reproduce the same rounded g/u. Identical
        # MMA/config and epilogue math should yield identical saved/recompute outputs.
        for sl in (slice(0, n), slice(nmax, nmax + n)):
            assert torch.equal(gg[rr, sl], gg_saved[rr, sl])
        saved_width = nmax // nt_saved
        saved_part = (q * af).reshape(-1, n // saved_width, saved_width).sum(dim=-1)
        _close(dw_saved[rr, : n // saved_width], saved_part)
        if be._config("bwd_slot", lay, True) == be._config("bwd_slot", lay, False):
            assert torch.equal(dw[rr, : n // width], dw_saved[rr, : n // width])
        assert torch.isnan(dw_saved[rr, n // saved_width :]).all()
        for tensor in (a, gs, us):
            assert torch.isnan(tensor[rr, n:]).all()
        assert torch.isnan(dw[rr, n // width :]).all()
        assert torch.isnan(gg[rr, n:nmax]).all()
        assert torch.isnan(gg[rr, nmax + n :]).all()

    # A cached executable must use the current stream, including graph capture.
    expected_y, expected_dw = y.clone(), dw.clone()
    y.fill_(float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            be.gate_up(x, w13, s13, lay, a, gs, us, ni=ni, nis=nis, kp1=kp1, save=True)
            be.down(ain, w2, s2, lay, y, ni=ni, nis=nis)
            be.dx_pair(gin, w13, s13, lay, dx, ni=ni, nis=nis)
            be.bwd_slot(
                x,
                dy,
                w13,
                s13,
                w2,
                s2,
                lay,
                ws,
                gs,
                us,
                gg,
                dw,
                ni=ni,
                nis=nis,
                kp1=kp1,
                saved=False,
            )
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0, equal_nan=True)
    for tensor in (a, gs, us, y, dx, gg, dw):
        assert torch.isnan(tensor[invalid]).all()


# Representative geometries; the dispatch test below covers all three BM buckets.
@pytest.mark.parametrize(
    "t,h,ni,nis,e",
    [
        (1, 256, 128, 64, 8),
        (7, 256, 128, 128, 8),
        (64, 2048, 512, 512, 256),
        (300, 2048, 512, 256, 256),
        pytest.param(2048, 2048, 768, 0, 128, marks=pytest.mark.exhaustive),
    ],
)
def test_grouped_gemms(t, h, ni, nis, e, monkeypatch):
    _case(t, h, ni, nis, e, monkeypatch=monkeypatch)


# Literal lists also drive the offline compiler audit (no GPU test import).
DEFAULT_CONFIGS = [
    "16x32x64s6w4g1x4n",
    "64x128x32s4w8g2x4",
    "64x64x64s3w8g4x2",
    "16x32x128s3w4g1x4t",
    "16x64x64s4w4g1x4t",
    "64x128x32s4w4g2x2nt",
]
EXHAUSTIVE_CONFIGS = [
    "32x64x64s2w4",
    "32x64x64s3w4",
    "64x64x64s2w4",
    "64x64x64s3w4",
    "128x64x32s4w8",
    "32x128x64s2w4n",
    "128x128x64s2w8",
    "16x32x128s4w4g1x4",
    "16x64x64s5w4g1x4",
    "32x64x32s6w4g2x2",
]
SINGLE_B_CONFIGS = ["32x256x32s3w8g2x4n", "64x256x32s3w8g4x2"]


@pytest.mark.parametrize(
    "spec",
    DEFAULT_CONFIGS + [pytest.param(s, marks=pytest.mark.exhaustive) for s in EXHAUSTIVE_CONFIGS],
)
def test_configs(spec, monkeypatch):
    # The same override also exercises K-major down and MN-major dx B layouts.
    override = ",".join(f"{kind}:{spec}" for kind in ("gate_up", "down", "bwd_slot", "dx_pair"))
    _case(7, 256, 128, 128, 8, override, monkeypatch)


@pytest.mark.parametrize("nis", [0, 64])
def test_tma_expert_descriptors(nis, monkeypatch):
    # Exercise missing shared descriptors and different routed/shared strides.
    spec = "16x64x64s4w4g1x4t"
    override = ",".join(f"{kind}:{spec}" for kind in ("gate_up", "down", "bwd_slot", "dx_pair"))
    _case(7, 256, 128, nis, 8, override, monkeypatch)


def test_tma_ring_phases(monkeypatch):
    # Eight gate/q iterations over three slots leave phases (1,1,0).
    # Shared down/dx use shorter K loops than routed experts in the same grid.
    spec = "16x32x128s3w4g1x4t"
    override = ",".join(f"{kind}:{spec}" for kind in ("gate_up", "down", "bwd_slot", "dx_pair"))
    _case(7, 1024, 512, 256, 8, override, monkeypatch)


@pytest.mark.parametrize("spec", SINGLE_B_CONFIGS)
def test_single_b_configs(spec, monkeypatch):
    _case(7, 256, 128, 128, 8, f"down:{spec},dx_pair:{spec}", monkeypatch)


def test_saved_independent_config(monkeypatch):
    _case(7, 256, 128, 64, 8, "gate_up:16x64x64s5w4g1x4,bwd_slot:16x32x32s6w4g1x4n", monkeypatch)


@pytest.mark.parametrize("cap", [(12, 0), (12, 1)])
@pytest.mark.parametrize("tokens", [7, 300, 513])
def test_dispatch_table(cap, tokens, monkeypatch):
    # Exercise both architecture tables on either GPU, including BN128 down/dx.
    table = configs(cap, tokens, 128, 128, 256, {})
    override = ",".join(f"{kind}:{cfg}" for kind, cfg in table.items())
    _case(tokens, 256, 128, 128, 8, override, monkeypatch)


@pytest.mark.parametrize("nis", [0, 128])
def test_all_masked(nis, monkeypatch):
    _case(7, 256, 128, nis, 8, monkeypatch=monkeypatch, all_masked=True)


@pytest.mark.parametrize("spec", ["32x64x64s4w4", "32x64x64s4w4t"])
def test_short_k_pipeline(spec, monkeypatch):
    # Shared down has one K tile with a three-group prologue. Out-of-range
    # prefetches must zero-fill without touching an invalid global address.
    override = ",".join(f"{kind}:{spec}" for kind in ("gate_up", "down", "bwd_slot", "dx_pair"))
    _case(7, 256, 128, 64, 8, override, monkeypatch)


@pytest.mark.exhaustive
@pytest.mark.parametrize("t", [1, 7, 64, 300, 2048])
@pytest.mark.parametrize(
    "geometry", [(2048, 512, 512, 256), (2048, 512, 256, 256), (2048, 768, 0, 128)]
)
@pytest.mark.parametrize("spec", ["32x64x64s2w4", "64x64x64s3w4", "128x128x64s2w8"])
def test_full_sweep(t, geometry, spec, monkeypatch):
    override = ",".join(f"{kind}:{spec}" for kind in ("gate_up", "down", "bwd_slot", "dx_pair"))
    _case(t, *geometry, override, monkeypatch)


@pytest.mark.parametrize(
    "spec", ["16x32x128s4w4g1x4", "64x64x64s3w8g4x2", "16x32x128s3w4g1x4t", "64x128x32s4w4g2x2nt"]
)
@pytest.mark.parametrize("kind", ["dx_pair", "bwd_slot"])
def test_mn_major_basis(kind, spec, monkeypatch):
    """One-hot K rows detect ldmatrix.trans fragment permutations exactly."""
    monkeypatch.setenv("CTI_MOE_CUTE_CONFIG", f"{kind}:{spec}")
    h, n, e, t = 256, 128, 2, 64
    be = CuteBackend(torch.device("cuda"), n, 0, h)
    bm = be.block_m(torch.device("cuda"), t, n, 0)
    lay = build_slot_layout(torch.zeros(t, dtype=torch.int64, device="cuda"), e, bm)
    rows = lay.sorted_slot.numel()
    # Distinct integer values remain exactly representable in BF16.
    pattern = (
        (
            torch.arange(2 * n * h, device="cuda").reshape(2 * n, h) * 13
            + torch.arange(h, device="cuda")
        )
        % 127
        - 63
    ).bfloat16()
    w13 = pattern[None].repeat(e, 1, 1)
    gs = torch.ones(rows, n, dtype=torch.bfloat16, device="cuda")
    us = torch.ones_like(gs)
    ws = torch.ones(t, device="cuda")
    # Sweep all K positions, including the gate/up boundary and all swizzle atoms.
    for base in range(0, 2 * n if kind == "dx_pair" else h, t):
        indices = torch.arange(t, device="cuda") + base
        if kind == "dx_pair":
            gg = torch.zeros(rows, 2 * n, dtype=torch.bfloat16, device="cuda")
            rr = lay.inv.long()
            gg[rr, indices] = 1
            out = torch.full((rows, h), float("nan"), dtype=torch.bfloat16, device="cuda")
            be.dx_pair(gg, w13, None, lay, out, ni=n, nis=0)
            assert torch.equal(out[rr], pattern[indices])
        else:
            dy = torch.zeros(t, h, dtype=torch.bfloat16, device="cuda")
            dy[torch.arange(t, device="cuda"), indices] = 1
            w2 = pattern[:n].T.contiguous()[None].repeat(e, 1, 1)
            out = torch.full((rows, 2 * n), float("nan"), dtype=torch.bfloat16, device="cuda")
            dw = torch.full(
                (rows, be.dw_tiles(torch.device("cuda"), t, n, 0, saved=True)),
                float("nan"),
                device="cuda",
            )
            be.bwd_slot(
                dy,
                dy,
                w13,
                None,
                w2,
                None,
                lay,
                ws,
                gs,
                us,
                out,
                dw,
                ni=n,
                nis=0,
                kp1=1,
                saved=True,
            )
            q = w2[0, indices].float()
            s = torch.sigmoid(torch.tensor(1.0, device="cuda"))
            _close(out[lay.inv.long(), n:], (q * s).bfloat16())
        assert torch.isnan(out[lay.sorted_slot < 0]).all()
