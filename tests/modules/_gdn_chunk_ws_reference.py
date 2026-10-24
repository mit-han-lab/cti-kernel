# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Independent FP64 recurrence and input builders shared by CP1 tests/benchmark."""

import importlib.metadata

import torch
import torch.nn.functional as F


def inputs(hk, hv, lengths, *, slow=False, seed=0, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    t = sum(lengths)

    def randn(*shape):
        return torch.randn(shape, device=device, generator=generator)

    q = F.normalize(randn(t, hk, 128), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(t, hk, 128), dim=-1).to(torch.bfloat16)
    v = randn(t, hv, 128).to(torch.bfloat16)
    g = -F.softplus(randn(t, hv)).clamp(min=1e-4, max=3.0)
    if slow:
        g *= 1e-3
    beta = randn(t, hv).sigmoid()
    pool = randn(len(lengths) + 3, hv, 128, 128) * 0.05
    cu = torch.tensor([0, *lengths], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return q, k, v, g, beta, pool, cu


@torch.no_grad()
def reference(q, k, v, g, beta, pool, lengths, slots, initial, scale, heads=None):
    """State is [V,K]: S*=exp(g); d=beta*(v-S@k); S+=d@k.T; o=scale*S@q.

    Metadata is supplied as host lists; no tensor metadata reads inside the loop.
    Returns output for selected heads and one final state per sequence (also -1).
    """
    hv, hk = v.shape[1], q.shape[1]
    heads = list(range(hv)) if heads is None else list(heads)
    kh = [head * hk // hv for head in heads]
    q64, k64 = q[:, kh].double(), k[:, kh].double()
    v64, g64, b64 = v[:, heads].double(), g[:, heads].double(), beta[:, heads].double()
    out = torch.empty((q.shape[0], len(heads), 128), device=q.device, dtype=torch.float64)
    states = []
    start = 0
    for length, slot, read in zip(lengths, slots, initial, strict=True):
        state = (
            pool[slot, heads].double().clone()
            if slot >= 0 and read
            else torch.zeros((len(heads), 128, 128), device=q.device, dtype=torch.float64)
        )
        for t in range(start, start + length):
            state.mul_(g64[t].exp()[:, None, None])
            pred = (state @ k64[t, :, :, None]).squeeze(-1)
            delta = b64[t, :, None] * (v64[t] - pred)
            state.add_(delta[:, :, None] * k64[t, :, None, :])
            out[t] = scale * (state @ q64[t, :, :, None]).squeeze(-1)
        states.append(state)
        start += length
    return out, torch.stack(states)


def rms_rel(actual, expected):
    diff = actual.double() - expected.double()
    return (
        diff.square().mean().sqrt() / expected.double().square().mean().sqrt().clamp_min(1e-30)
    ).item()


def flashinfer_kernel():
    """Only missing/other versions disable comparison; installed-kernel errors propagate."""
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not version.startswith("0.7."):
        return None
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import delta_rule_prefill_dsl

    return delta_rule_prefill_dsl


def fi_state(pool, slots, initial):
    """Map unowned sequences to private scratch rows; FI has no -1/initial mask ABI."""
    state = pool.clone()
    indices = []
    unused = [i for i in range(pool.shape[0]) if i not in slots]
    for slot, read in zip(slots, initial, strict=True):
        target = slot if slot >= 0 else unused.pop()
        indices.append(target)
        if slot < 0 or not read:
            state[target].zero_()
    return state, torch.tensor(indices, dtype=torch.int32, device=pool.device)
