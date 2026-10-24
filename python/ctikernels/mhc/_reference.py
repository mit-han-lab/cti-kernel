# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-importable mHC exit, with explicit rounding boundaries."""

import math
from dataclasses import dataclass

import torch


def round_bf16(value):
    """Round FP64 directly to BF16, ties to even, without FP32 double rounding.

    PyTorch's FP64 conversion goes through FP32. Only FP32 values exactly at
    a BF16 midpoint can choose the wrong neighbor. Restore the discarded sticky
    information by moving one FP32 ulp toward the original before conversion.
    Nonfinite values and signed zero retain PyTorch's conversion semantics.
    """
    if value.dtype != torch.float64:
        return value.bfloat16()
    intermediate = value.float()
    midpoint = (intermediate.view(torch.int32) & 0xFFFF) == 0x8000
    back = intermediate.double()
    adjust = midpoint & (back != value) & torch.isfinite(value)
    direction = torch.where(value > back, float("inf"), -float("inf")).float()
    sticky = torch.nextafter(intermediate, direction)
    return torch.where(adjust, sticky, intermediate).bfloat16()


@dataclass(frozen=True)
class MhcConfig:
    hidden: int = 4096
    mult: int = 4
    sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_eps: float = 1e-6
    norm_eps: float = 1e-6
    post_mult: float = 2.0

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hidden, self.mult, self.sinkhorn_iters)):
            raise ValueError("dimensions and Sinkhorn iterations must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.hc_eps, self.rms_eps, self.norm_eps)):
            raise ValueError("epsilons must be finite and positive")
        if not math.isfinite(self.post_mult):
            raise ValueError("post_mult must be finite")


def sinkhorn(logits, cfg):
    # Formula: DeepSeek-V4-Flash@60d8d707 inference/kernel.py:398-424.
    matrix = logits.softmax(dim=-1) + cfg.hc_eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.sinkhorn_iters - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + cfg.hc_eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + cfg.hc_eps)
    return matrix


def make_weights(cfg, seed=2026, device="cpu"):
    """Unit-variance mixes after RMS scaling; unit scale and small base offsets.

    fn has std 1/sqrt(mult*hidden), base std 0.1, norm weight 1 +/- 0.05.
    These are synthetic, nonsaturated weights, not checkpoint statistics.
    CPU generation makes seeds identical across ranks and device types.
    """
    gen = torch.Generator().manual_seed(seed)
    width = cfg.mult * cfg.hidden
    count = cfg.mult * (cfg.mult + 2)
    weights = {
        "fn": torch.randn(count, width, generator=gen) / math.sqrt(width),
        "scale": torch.ones(3),
        "base": torch.randn(count, generator=gen) * 0.1,
        "norm_weight": (1 + 0.05 * torch.randn(cfg.hidden, generator=gen)).bfloat16(),
    }
    return {key: value.to(device) for key, value in weights.items()}


def make_state(cfg, T, seed=1709, device="cpu"):
    if T < 1:
        raise ValueError("T must be positive")
    gen = torch.Generator().manual_seed(seed)
    x0, x1 = [
        (torch.randn(T, cfg.hidden, generator=gen) / math.sqrt(2)).bfloat16() for _ in range(2)
    ]
    state = {
        "x": (x0.float() + x1.float()).bfloat16(),
        "R": torch.randn(T, cfg.mult, cfg.hidden, generator=gen).bfloat16(),
        "post": cfg.post_mult * torch.randn(T, cfg.mult, generator=gen).sigmoid(),
        "comb": sinkhorn(torch.randn(T, cfg.mult, cfg.mult, generator=gen), cfg),
        "x0": x0,
        "x1": x1,
    }
    return {key: value.to(device) for key, value in state.items()}


def _exit(cfg, weights, x, R, post, comb, dtype, rounded, mode="exit"):
    # Equations/axis orientation: official inference/model.py:674-690.
    if mode == "pre":
        residual = R.to(dtype)
    else:
        x, R, post, comb = [value.to(dtype) for value in (x, R, post, comb)]
        post = post.reshape(-1, cfg.mult)
        residual = post[..., None] * x[:, None, :] + torch.einsum("tij,tih->tjh", comb, R)
    if rounded:
        residual = residual.bfloat16()
    if mode == "post":
        return {"residual": residual}
    flat = residual.to(dtype).flatten(1)
    mixes = (flat @ weights["fn"].to(dtype).T) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + cfg.rms_eps
    )
    m = cfg.mult
    scale, base = weights["scale"].to(dtype), weights["base"].to(dtype)
    pre = (mixes[:, :m] * scale[0] + base[:m]).sigmoid() + cfg.hc_eps
    post = cfg.post_mult * (mixes[:, m : 2 * m] * scale[1] + base[m : 2 * m]).sigmoid()
    comb = sinkhorn((mixes[:, 2 * m :] * scale[2] + base[2 * m :]).reshape(-1, m, m), cfg)
    layer_input = (pre[..., None] * residual.to(dtype)).sum(dim=1)
    if rounded:
        layer_input = layer_input.bfloat16()
    y = layer_input.to(dtype)
    out = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    out = out * weights["norm_weight"].to(dtype)
    if rounded:
        out = out.bfloat16()
    return {
        "residual": residual,
        "post": post,
        "comb": comb,
        "pre": pre,
        "layer_input": layer_input,
        "out": out,
    }


def exit_fp64(cfg, weights, x, R, post, comb):
    """All arithmetic FP64, starting from the same stored inputs; no BF16 rounding."""
    return _exit(cfg, weights, x, R, post, comb, torch.float64, False)


def exit_torch_bf16(cfg, weights, x, R, post, comb):
    """FP32 arithmetic with BF16 residual, collapsed input, and final output."""
    return _exit(cfg, weights, x, R, post, comb, torch.float32, True)


def output_dict(result):
    if isinstance(result, dict):
        return result
    return dict(zip(("residual", "post", "comb", "out"), result, strict=True))


def error_stats(candidate, oracle):
    """Max absolute and RMS error for each exposed output (no invented intermediates)."""
    metrics = {}
    for key, value in output_dict(candidate).items():
        if value.shape != oracle[key].shape:
            raise ValueError(f"{key}: output shape differs from oracle")
        error = value.detach().cpu().double() - oracle[key].detach().cpu().double()
        metrics[key] = {
            "max_abs": error.abs().max().item(),
            "rms": error.square().mean().sqrt().item(),
        }
    return metrics
