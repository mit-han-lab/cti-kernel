# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Linear numerics vs FP64: signed error, M invariance and perturbation propagation.

Per Qwen3.5-9B decode shape and M=1..8 (lm_head M=1), compares CuTe (table
config, else the probe heuristic), F.linear and shipped dispatch against an FP64
reference, in BF16 ULPs of the reference: mean error along sign(ref) (a
magnitude bias), RMS, correctly-rounded fraction and the share of misrounded
outputs that moved toward zero. Then checks rows of M<8 against M=8 (bitwise),
and pushes the same small input perturbation through an 8-layer residual MLP
stack with each GEMM backend (CuTe, cuBLAS, FP32-accumulate + one rounding),
reporting log-prob differences. Results: RESULT JSON lines on stdout.
Summary: benchmark/kernel_tuning/results/linear_smallm_numerics.md.
"""

import json
import math

import torch
import torch.nn.functional as F

from ctikernels.linear._dispatch import config_for, device_class, tuned_config
from ctikernels.linear.inference import Linear

dev = torch.device("cuda")
# Set in main(): no CUDA activity at import time.
props = kind = None
SHAPES = {
    "gdn_in": (12352, 4096),
    "o_out": (4096, 4096),
    "qkv_gate": (10240, 4096),
    "gate_up": (24576, 4096),
    "down": (4096, 12288),
    "ba": (64, 4096),
    "lm_head": (248320, 4096),
}


def emit(**r):
    print("RESULT " + json.dumps(r), flush=True)


def ulp_of(y64):
    a = y64.abs().clamp_min(1e-30).float()
    _, e = torch.frexp(a)
    return torch.ldexp(torch.ones_like(a), e - 8).double()


def stats(y, y64):
    yr = y64.float().to(torch.bfloat16)
    u = ulp_of(y64)
    e = (y.double() - y64) / u
    s = torch.sign(y64)
    mag = e * s
    mism = y != yr
    tz = ((y.float().abs() < yr.float().abs()) & mism).sum().item()
    n = e.numel()
    return {
        "mean_signed_ulp": e.mean().item(),
        "mean_mag_ulp": mag.mean().item(),
        "sem_ulp": (e.std() / math.sqrt(n)).item(),
        "rms_ulp": e.pow(2).mean().sqrt().item(),
        "max_ulp": e.abs().max().item(),
        "correctly_rounded": 1 - mism.float().mean().item(),
        "mismatch_toward_zero": tz / max(1, mism.sum().item()),
        "mean_signed_rel": ((y.double() - y64) / y64.abs().mean()).mean().item(),
    }


def act(kindname, m, k, g):
    if kindname == "swiglu":
        a = torch.randn(m, k, generator=g, device=dev) * 1.5
        b = torch.randn(m, k, generator=g, device=dev) * 1.5
        return (F.silu(a) * b).bfloat16()
    gamma = 1 + 0.5 * torch.randn(k, generator=g, device=dev).abs()
    idx = torch.randperm(k, generator=g, device=dev)[: k // 100]
    gamma[idx] *= 10
    x = torch.randn(m, k, generator=g, device=dev) * gamma
    if kindname == "rmsnorm_shifted":
        x = x + 0.3
    return x.bfloat16()


def forced_config(n, k, m):
    return tuned_config(kind, n, k, m) or config_for(kind, props.multi_processor_count, n, k)


def per_shape():
    g = torch.Generator(device=dev).manual_seed(1234)
    for name, (n, k) in SHAPES.items():
        w = (torch.randn(n, k, generator=g, device=dev) * 0.02).bfloat16()
        w64 = w.double()
        ms = [1] if name == "lm_head" else list(range(1, 9))
        for actname in ("swiglu",) if name == "down" else ("rmsnorm", "rmsnorm_shifted"):
            # Same 8 rows for every M: row r of M=m is row r of the 8-row batch.
            xs = [act(actname, 8, k, g) for _ in range(4)]
            for m in ms:
                cfg = forced_config(n, k, m)
                cute = Linear(config=cfg, provider="cute")
                auto = Linear(provider="auto")
                acc = {"cute": [], "cublas": [], "auto": []}
                for x8 in xs:
                    x = x8[:m].contiguous()
                    y64 = x.double() @ w64.T
                    outs = {"cute": cute(x, w), "cublas": F.linear(x, w), "auto": auto(x, w)}
                    for key, y in outs.items():
                        acc[key].append((y, y64))
                torch.cuda.synchronize()
                for key in acc:
                    y = torch.cat([a for a, _ in acc[key]])
                    y64 = torch.cat([b for _, b in acc[key]])
                    emit(
                        kind="err",
                        device=kind,
                        shape=name,
                        n=n,
                        k=k,
                        m=m,
                        act=actname,
                        impl=key,
                        config=str(cfg)
                        if key == "cute"
                        else (auto._provider if key == "auto" else "cublas"),
                        **stats(y, y64),
                    )
        del w, w64
        torch.cuda.empty_cache()


def cublas_m_invariance():
    g = torch.Generator(device=dev).manual_seed(7)
    for name, (n, k) in SHAPES.items():
        if name == "lm_head":
            continue
        w = (torch.randn(n, k, generator=g, device=dev) * 0.02).bfloat16()
        x8 = act("rmsnorm", 8, k, g)
        y8 = F.linear(x8, w)
        cfg8 = forced_config(n, k, 8)
        c8 = Linear(config=cfg8, provider="cute")(x8, w)
        diffs = {}
        for m in range(1, 8):
            diffs[m] = {
                "cublas_rows_equal_m8": bool(torch.equal(F.linear(x8[:m].contiguous(), w), y8[:m])),
                "cute_rows_equal_m8_same_cfg": bool(
                    torch.equal(
                        Linear(config=cfg8, provider="cute")(x8[:m].contiguous(), w), c8[:m]
                    )
                ),
            }
        emit(kind="minv", device=kind, shape=name, rows=diffs)


def chain():
    """Residual MLP stack; a padded/unpadded-like perturbation enters row 0; GEMM backend varies."""
    L, H, INTER, V, M = 8, 4096, 12288, 32768, 8
    g = torch.Generator(device=dev).manual_seed(99)
    gu = [(torch.randn(2 * INTER, H, generator=g, device=dev) * 0.02).bfloat16() for _ in range(L)]
    dn = [
        (torch.randn(H, INTER, generator=g, device=dev) * 0.02 / math.sqrt(2 * L)).bfloat16()
        for _ in range(L)
    ]
    oo = [
        (torch.randn(H, H, generator=g, device=dev) * 0.02 / math.sqrt(2 * L)).bfloat16()
        for _ in range(L)
    ]
    head = (torch.randn(V, H, generator=g, device=dev) * 0.02).bfloat16()
    norm = lambda h: (
        h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()

    def backend(name):
        if name == "cublas":
            return lambda x, w: F.linear(x, w)
        if name == "rn_fp32":
            return lambda x, w: (x.float() @ w.float().T).bfloat16()
        ops = {}

        def f(x, w):
            key = tuple(w.shape)
            if key not in ops:
                ops[key] = Linear(
                    config=forced_config(w.shape[0], w.shape[1], x.shape[0]), provider="cute"
                )
            return ops[key](x, w)

        return f

    def run(lin, h):
        h = h.clone()
        for layer in range(L):
            a = norm(h)
            h = h + lin(a, oo[layer]).float()
            t = lin(norm(h), gu[layer])
            h = (
                h
                + lin(
                    (F.silu(t[:, :INTER].float()) * t[:, INTER:].float()).bfloat16(), dn[layer]
                ).float()
            )
        return torch.log_softmax(F.linear(norm(h), head).float(), -1)

    res = {}
    for b in ("cute", "cublas", "rn_fp32"):
        lin = backend(b)
        means, maxes = [], []
        for seed in range(16):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            h0 = torch.randn(M, H, generator=gg, device=dev)
            noise = (torch.rand(M, H, generator=gg, device=dev) < 0.02) * torch.sign(
                torch.randn(M, H, generator=gg, device=dev)
            )
            h1 = h0.bfloat16().float() + noise * h0.bfloat16().float().abs() * 2**-8
            d = (run(lin, h0.bfloat16().float()) - run(lin, h1)).abs()
            means.append(d.mean().item())
            maxes.append(d.max().item())
        res[b] = {
            "mean_logprob_diff": sum(means) / len(means),
            "max_logprob_diff": max(maxes),
            "per_seed_mean": means,
        }
    emit(kind="chain", device=kind, **res)


def main():
    global props, kind
    # Full-precision cuBLAS baseline: the question is CuTe's own rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    props = torch.cuda.get_device_properties(0)
    kind = device_class((props.major, props.minor), props.multi_processor_count)
    emit(kind="device", device=kind, name=props.name)
    per_shape()
    cublas_m_invariance()
    chain()
    emit(kind="done")


if __name__ == "__main__":
    main()
