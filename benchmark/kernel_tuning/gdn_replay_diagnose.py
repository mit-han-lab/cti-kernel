# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU arithmetic attribution, not an emulation of NVIDIA transcendental instructions.

The two reduction trees deliberately model layout changes. Only GPU receipts can
attribute the exact historical 1.02x ratio. Raw replay with identical arithmetic
must be bitwise equal to immediate writeback on the same backend.
"""

import argparse
import json

import torch

from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs


def tree(x, alternate=False):
    if alternate:
        x = x.reshape(*x.shape[:-1], 8, 16).transpose(-1, -2).flatten(-2)
    while x.shape[-1] > 1:
        x = x[..., ::2] + x[..., 1::2]
    return x[..., 0]


def fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def token(x, weights, hk, state, history, *, alternate=False, normalized=False):
    """Product's scalar-normalized recurrence, with explicit FP32 FMA rounding."""
    cw, al, bias, nw = weights
    hv, dc = al.numel(), cw.shape[0]
    dtype = state.dtype
    taps = torch.cat((history, x[:, None, :dc]), 1).double()
    mixed = (taps * cw.double().T).sum(1).to(dtype)
    mixed = mixed * torch.sigmoid(mixed)
    q, k, v = mixed.split((hk * 128, hk * 128, hv * 128), -1)
    q = q.reshape(-1, hk, 128).repeat_interleave(hv // hk, 1)
    k = k.reshape(-1, hk, 128).repeat_interleave(hv // hk, 1)
    v = v.reshape(-1, hv, 128)
    reduce = lambda a: tree(a, alternate)
    rq = torch.rsqrt(reduce(q * q) + 1e-6) * 128**-0.5
    rk = torch.rsqrt(reduce(k * k) + 1e-6)
    z, b, a = x[:, dc:].to(dtype).split((hv * 128, hv, hv), -1)
    g = a + bias.to(dtype)
    alpha = torch.exp(-al.to(dtype).exp() * (g.clamp_min(0) + torch.log1p((-g.abs()).exp())))
    beta = torch.sigmoid(b)
    if normalized or dtype == torch.float64:
        k, q = k * rk[..., None], q * rq[..., None]
        decayed = alpha[..., None, None] * state
        delta = beta[..., None] * (v - reduce(decayed * k[..., None, :]))
        new = decayed + delta[..., None] * k[..., None, :]
        o = reduce(new * q[..., None, :])
    else:
        sk = reduce(state * k[..., None, :]) * rk[..., None]
        sq = reduce(state * q[..., None, :]) * rq[..., None]
        delta = beta[..., None] * fma(-alpha[..., None], sk, v)
        kq = reduce(k * q) * rq * rk
        o = fma(delta, kq[..., None], alpha[..., None] * sq)
        new = fma(
            (delta * rk[..., None])[..., None], k[..., None, :], alpha[..., None, None] * state
        )
    y = nw.to(dtype) * o * torch.rsqrt(reduce(o * o)[..., None] / 128 + 1e-6)
    y *= (z * torch.sigmoid(z)).reshape(-1, hv, 128)
    return y.to(torch.bfloat16), new, torch.cat((history[:, 1:], x[:, None, :dc]), 1)


def run(hk, hv, batch, seed, kind, steps=35, window=8):
    weights, xs, initial, hist = case_inputs(hk, hv, batch, batch, steps, kind, seed, "cpu")
    weights = weights.args()
    paths = ("product_order", "changed_tree", "normalized_key", "raw_replay")
    states = {p: initial.clone() for p in paths}
    histories = {p: hist.clone() for p in paths}
    exact, eh = initial.double(), hist.clone()
    checkpoint, seeds = initial.clone(), hist.clone()
    squared = {p: [0.0, 0.0] for p in paths}
    maxima = {p: [0.0, 0.0] for p in paths}
    for t, x in enumerate(xs):
        # Keep unrounded FP64 output for the judge (token returns BF16 for parity).
        _, exact, eh = token(x, weights, hk, exact, eh)
        ys = {}
        for p in paths:
            if p == "raw_replay":
                s, h = checkpoint.clone(), seeds.clone()
                for j in range(t - t % window, t + 1):
                    y, s, h = token(xs[j], weights, hk, s, h)
                if (t + 1) % window == 0:
                    checkpoint, seeds = s.clone(), h.clone()
            else:
                y, s, h = token(
                    x,
                    weights,
                    hk,
                    states[p],
                    histories[p],
                    alternate=p == "changed_tree",
                    normalized=p == "normalized_key",
                )
            states[p], histories[p], ys[p] = s, h, y
        # Independent pinned FP64 token reference for output, without BF16 cast.
        from tests.gpu._gdn_reference import Weights, decode

        if t == 0:
            ref_s, ref_h = initial.double(), hist.clone()
        ref_y, ref_s, ref_h = decode(x, Weights(*weights), hk, torch.arange(batch), ref_s, ref_h)
        for p in paths:
            for i, (got, truth) in enumerate(
                ((ys[p].reshape(batch, -1), ref_y), (states[p], ref_s))
            ):
                e = (got.double() - truth).abs()
                squared[p][i] += e.square().mean().item()
                maxima[p][i] = max(maxima[p][i], e.max().item())
        assert torch.equal(states["product_order"], states["raw_replay"])
        assert torch.equal(ys["product_order"], ys["raw_replay"])
    result = {}
    for p in paths:
        result[p] = {
            name: (squared[p][i] / squared["product_order"][i]) ** 0.5
            if squared["product_order"][i]
            else 1.0
            for i, name in enumerate(("output_rms_ratio", "state_rms_ratio"))
        }
        result[p].update(
            {
                name: maxima[p][i] / maxima["product_order"][i]
                if maxima["product_order"][i]
                else 1.0
                for i, name in enumerate(("output_max_ratio", "state_max_ratio"))
            }
        )
    return {
        "shape": [hk, hv],
        "B": batch,
        "seed": seed,
        "input_class": kind,
        "steps": steps,
        "L": window,
        "paths": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()
    torch.set_num_threads(1)
    worst = {}
    for hk, hv in ((16, 32), (16, 48)):
        for b in (1, 4):
            for kind in ("ordinary", "heavy", "zero_keys"):
                for seed in (11, 23, 47, 89, 131):
                    row = run(hk, hv, b, seed, kind, args.steps)
                    print(json.dumps(row), flush=True)
                    for path, metrics in row["paths"].items():
                        for metric, value in metrics.items():
                            key = path + "/" + metric
                            if value > worst.get(key, {}).get("ratio", -1):
                                worst[key] = {"ratio": value, "case": [hk, hv, b, seed, kind]}
    print(json.dumps({"type": "summary", "worst": worst}), flush=True)


if __name__ == "__main__":
    main()
