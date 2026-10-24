# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""In-place all-reduce + residual add + RMSNorm for a single-node MoE exit.

CuTe integration contract: one process per GPU (W=2), one instance per CPU
(gloo) group, constructed collectively outside graph capture after
``torch.cuda.set_device``. Rank 0 creates a /dev/shm file and broadcasts its
name; every rank maps it, barriers, and rank 0 unlinks it. Link calibration
then runs on every rank with barriers.

Each instance registers one pinned host region with cudaHostRegister(Portable |
Mapped): approximately W * 2 * rows_cap * round_up(2H, 128) plus progress words,
where rows_cap = min(8192, max(SMs, max_tokens)) for positive max_tokens.
H=2048, max_tokens=8192, W=2 needs 128 MiB of mailboxes (128 MiB + 60 KiB
including progress, alignment and reservation on 188 SMs). Caller tensors are
never registered or retained. x and residual need unit inner stride, a row
stride divisible by 8 elements and a 16-byte-aligned base; the weight must be
contiguous and 16-byte aligned. Other layouts raise ValueError: the exchange
never stages through device copies.

Each call fitting rows_cap is one launch on the current torch CUDA
stream, with no host synchronization, copy-engine work or NCCL. The flag
kernel uses programmatic dependent launch and waits for the preceding kernel
before touching memory; producer/consumer uses ordinary stream ordering.
Larger calls are sliced into complete exchanges. All ranks must issue the
same token counts in the same order, eager or replayed: a skipped call leaves
a peer spinning without a timeout, just like a collective.

CUDA graphs are supported, including slicing. Compile
first with ``prepare(token_counts)`` or an eager call per count. Device-memory
call tags support arbitrary replay counts mixed with eager calls, provided
every rank follows the same sequence. One instance can serve all MoE-exit and
row-parallel reductions on one stream; a concurrent stream needs its own.

``close()`` is local: synchronize this device, unregister, unmap. Each peer's
mapping keeps the region alive. Close on every rank after the last call any
peer may issue (for example after a shutdown CPU barrier); the process group
may already be destroyed. NCCL fallback rounds the reduction to BF16 before
the residual add; CuTe sums in FP32 and rounds once, so its error to an FP64
oracle does not exceed that composition's (``measured_floor``).
"""

from __future__ import annotations

import torch

from ctikernels.norm.inference import _require_mutable

from ._common import Epilogue, Exchange  # noqa: F401  (public: the group's exchange)


def _support_reason(name, capability, world_size, hidden, dtype, zero_centered, cute_available):
    if world_size != 2:
        return "CuTe host exchange requires world_size == 2"
    if capability != (12, 0) or "RTX PRO 6000" not in name:
        return "automatic CuTe dispatch requires an sm_120 RTX PRO 6000"
    if dtype != torch.bfloat16:
        return "CuTe host exchange requires bfloat16"
    if hidden not in {2048, 4096, 5120}:
        return "automatic CuTe dispatch requires hidden in {2048, 4096, 5120}"
    if not cute_available:
        return "CuTe host exchange requires cutlass (the cute extra)"
    return None


def _cute_available():
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def support(device, *, world_size, hidden, dtype, zero_centered) -> str | None:
    """Return why automatic CuTe dispatch is unavailable, or None when supported."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CuTe host exchange requires a CUDA device"
    args = (
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        world_size,
        hidden,
        dtype,
        zero_centered,
    )
    reason = _support_reason(*args, True)
    return reason if reason is not None else _support_reason(*args, _cute_available())


def reference(
    partials, residual, weight, eps, *, zero_centered=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-ordered FP32 accumulation, unrounded-sum normalization, one output cast.

    ``partials`` is [W, T, H]. Outputs have residual.dtype. Torch's sum of
    squares may differ from the CuTe reduction order in the last FP32 bits.
    """
    if partials.ndim != 3 or partials.shape[1:] != residual.shape:
        raise ValueError("partials must be [W, T, H] and residual [T, H]")
    if weight.shape != (residual.shape[-1],):
        raise ValueError("weight must be [H]")
    h = residual.float().clone()
    for partial in partials:
        h = h + partial.float()
    ss = (h * h).sum(dim=-1, keepdim=True)
    rstd = 1 / torch.sqrt(ss / h.shape[-1] + eps)
    gain = weight.float() + 1 if zero_centered else weight.float()
    return (h * rstd * gain).to(residual.dtype), h.to(residual.dtype)


def exact_reference(
    partials, residual, weight, eps, *, zero_centered=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 normalized output and residual sum, the oracle for the measured floor."""
    h = residual.double() + partials.double().sum(dim=0)
    rstd = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + eps)
    gain = weight.double() + 1 if zero_centered else weight.double()
    return h * rstd * gain, h


def market_composition(
    summed, residual, weight, eps, *, zero_centered=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused market numerics from an all-reduce result already rounded to the
    activation dtype: residual add in that dtype, then RMSNorm computed in FP32
    with an output cast. Inputs are not modified."""
    h = summed + residual
    hf = h.float()
    rstd = torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + eps)
    gain = weight.float() + 1 if zero_centered else weight.float()
    return (hf * rstd * gain).to(h.dtype), h


def measured_floor(ours, market, exact) -> tuple[list[str], dict]:
    """Gate ``ours`` against the market composition's error to the FP64 oracle.

    Each argument is an ``(x, residual)`` pair. For both outputs, our per-element
    maximum and RMS errors must not exceed the market's. Returns the failures and
    the measured errors.
    """
    failures, stats = [], {}
    for name, a, m, e in zip(("x", "residual"), ours, market, exact, strict=True):
        oracle = e.double().cpu()
        da, dm = (t.double().cpu() - oracle for t in (a, m))
        errors = {
            "max": (da.abs().max().item(), dm.abs().max().item()),
            "rms": (da.pow(2).mean().sqrt().item(), dm.pow(2).mean().sqrt().item()),
        }
        for kind, (mine, theirs) in errors.items():
            stats[f"{name}_{kind}"] = mine
            stats[f"{name}_{kind}_market"] = theirs
            if not mine <= theirs:
                failures.append(f"{name} {kind} error {mine:.3e} exceeds market {theirs:.3e}")
    return failures, stats


class AllReduceAddRMSNorm:
    """All-reduce partial sums + residual add + RMSNorm in place.

    Serves the MoE-exit / row-parallel epilogue of one tensor- or
    expert-parallel group on one node. A supplied NCCL communicator is borrowed;
    closing this object does not destroy it. ``max_tokens`` sizes mailboxes,
    and is not a limit on call size. All tensors must be on ``device``.
    """

    provider: str

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        cpu_group,
        device,
        hidden: int,
        max_tokens: int,
        eps: float,
        zero_centered: bool = True,
        dtype=torch.bfloat16,
        communicator=None,
        provider: str = "auto",
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be within a positive world_size")
        if hidden <= 0 or max_tokens < 0:
            raise ValueError("hidden must be positive and max_tokens nonnegative")
        if provider not in {"auto", "cute", "nccl", "local"}:
            raise ValueError(f"unknown collective provider: {provider!r}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.dtype = hidden, dtype
        self.eps, self.zero_centered = float(eps), zero_centered
        self.communicator = communicator
        self._exchange = None
        self._closed = False
        if provider == "auto":
            if world_size == 1:
                provider = "local"
            else:
                reason = support(
                    self.device,
                    world_size=world_size,
                    hidden=hidden,
                    dtype=dtype,
                    zero_centered=zero_centered,
                )
                provider = "cute" if reason is None else "nccl"
        if provider == "local" and world_size != 1:
            raise ValueError("local provider requires world_size == 1")
        if provider == "nccl" and communicator is None:
            raise ValueError("nccl provider requires a communicator")
        if provider == "cute":
            if (
                self.device.type != "cuda"
                or not torch.cuda.is_available()
                or torch.cuda.get_device_capability(self.device) not in {(12, 0), (12, 1)}
                or world_size != 2
                or dtype != torch.bfloat16
                or hidden % 8
                or not _cute_available()
            ):
                raise NotImplementedError(
                    "cute requires sm_120/sm_121, world_size=2, bfloat16, "
                    "hidden divisible by 8, and the cute extra"
                )
            from ._host_exchange.cute.exchange import HostExchange

            self._exchange = HostExchange(
                rank=rank,
                world_size=world_size,
                cpu_group=cpu_group,
                device=self.device,
                hidden=hidden,
                max_tokens=max_tokens,
                zero_centered=zero_centered,
            )
        self.provider = provider

    def _require_open(self):
        if self._closed:
            raise RuntimeError("AllReduceAddRMSNorm used after close()")

    def prepare(self, token_counts) -> None:
        """Compile all CuTe slice variants before capture; no-op for local/NCCL."""
        self._require_open()
        if self._exchange is not None:
            with torch.cuda.device(self.device):
                self._exchange.prepare(token_counts)

    def __call__(self, x, residual, weight) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_open()
        if weight.requires_grad:
            # The norm weight is frozen (inference): reject it before any device work.
            raise ValueError("AllReduceAddRMSNorm weight must not require gradients")
        _require_mutable("AllReduceAddRMSNorm", x, residual)
        if x.ndim != 2 or x.shape != residual.shape or x.shape[1] != self.hidden:
            raise ValueError("x and residual must have the same shape [T, hidden]")
        if weight.shape != (self.hidden,):
            raise ValueError("weight must have shape [hidden]")
        if any(t.device != self.device for t in (x, residual, weight)):
            raise ValueError("all tensors must be on the configured device")
        if x.dtype != self.dtype or residual.dtype != self.dtype:
            raise TypeError("x and residual must have the configured dtype")
        if self.provider == "cute" and weight.dtype != torch.bfloat16:
            raise TypeError("cute weight must have dtype bfloat16")
        if x.shape[0] == 0:
            return x, residual
        if self._exchange is not None:
            with torch.cuda.device(self.device):
                return self._exchange(x, residual, weight, self.eps)
        if self.provider == "nccl":
            self.communicator.all_reduce(x, "sum")
        from ctikernels.norm.inference import add_rms_norm_

        return add_rms_norm_(x, residual, weight, self.eps, zero_centered=self.zero_centered)

    def close(self) -> None:
        """Synchronize and release local exchange resources; idempotent."""
        if not self._closed:
            if self._exchange is not None:
                self._exchange.close()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
