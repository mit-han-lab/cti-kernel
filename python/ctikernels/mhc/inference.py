# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Four-stream mHC: one launch, no implicit fallback, one stream per instance.

Call prepare(token_counts) outside capture. In graphs all output buffers must be
supplied; eager calls may allocate omitted outputs. Inputs/outputs must be
contiguous, 16-byte aligned, disjoint tensors on the configured device. Weights
(fn, scale, base, norm_weight) must already be ready before a PDL chain starts;
only x/residual/post/comb may depend on its preceding kernel. Keep tensors alive
until their stream finishes. close() synchronizes before releasing workspace.

Unlike the market composition, projection, Gram statistics and collapse use the
unrounded FP32 post residual. Decode retains that FP32 residual; prefill
recomputes the same high part. Neither path rounds y before RMSNorm. Pre mode reads the
stored BF16 residual and then shares exit's FP32 arithmetic (its former FP64 chain ran ~4x
slower than exit on GB300's FP64 rate).
norm_weight is multiplied directly (not 1+w).

Collective exits (all_reduce=...) follow a same-calls contract: every rank
issues the same calls with identical tensor layouts (shapes, strides, storage
offsets and aliasing between arguments). Layout-only checks are therefore
rank-symmetric and run before any collective is issued or recorded: an output
with internal overlap (expanded or self-overlapping view) raises ValueError
there. Overlap between arguments uses conservative byte spans from sizes and
strides; an overlapping, misaligned or non-contiguous tensor is staged, never
rejected, once a collective may start. Every input is staged and every
copy-back planned and validated (width <= both pitches) before the collective;
after it only already-planned raw device copies (memcpys on the current
stream, graph-capturable) and kernel launches run. They bypass torch tensor
rules, so inference mode, grad mode or version counters, which are not layout
facts and may differ between ranks, cannot raise after a collective is recorded.
"""

import importlib.metadata
import math
import os

import torch

from .._storage import internal_overlap, plan_copy, raw_copy, spans_overlap
from ._config import kernel_options
from ._reference import MhcConfig, _exit


def _cute_available():
    try:
        import cutlass.cute  # noqa: F401

        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return False


def _policy(
    hidden,
    sms,
    shared_limit,
    capacity=32,
    *,
    shared_per_sm=None,
    regs_per_sm=65536,
    threads_per_sm=1024,
    slice_size=None,
):
    """Conservative co-residency bound, including 255 registers per thread.

    fn is staged in 64-column pieces and its allocation is reused for totals.
    This permits S=128/T=32 on devices with a 99 KiB per-block limit. Small
    token buckets prefer S=64 and allow two CTAs per SM when resources permit.
    """
    if capacity not in (4, 16, 32, 64, 128, 256):
        raise ValueError("decode capacity must be 4, 16, 32, 64, 128 or 256")
    if slice_size not in (None, 64, 128):
        raise ValueError("decode slice must be 64 or 128")
    if shared_per_sm is None:
        shared_per_sm = shared_limit
    sizes = (64, 128) if slice_size is None else (slice_size,)
    memory_ok = False
    for size in sizes:
        # Rounded allocations plus 2 KiB for static compiler shared memory.
        cache = capacity * 35 * 4 if capacity > 32 else capacity * 4 * size * 4
        shared = 96 * 64 * 4 + cache + size * 4
        shared += 128 + ((capacity * 5 * 4 + 15) // 16) * 16 + 2048
        shared += ((capacity * 4 + 15) // 16) * 16  # RMS retained until Sinkhorn.
        if shared > shared_limit:
            continue
        memory_ok = True
        resident = min(shared_per_sm // shared, regs_per_sm // (256 * 128), threads_per_sm // 128)
        grid = (hidden + size - 1) // size
        if grid <= sms * resident:
            return size, grid, capacity
    if not memory_ok:
        raise ValueError("insufficient shared memory for the mHC launch policy")
    raise ValueError("hidden requires more decode CTAs than can be co-resident")


def _cluster_plan(hidden, capacity):
    """Eight contiguous slices; warp-sized tails are masked inside the kernel."""
    return ((hidden + 511) // 512 * 64, 8, capacity)


def _prefill_tile(tokens, sms):
    """Largest tile up to 16 that supplies at least two CTAs per SM."""
    tile = 16
    while tile > 1 and (tokens + tile - 1) // tile < 2 * sms:
        tile //= 2
    return tile


def _device_plan(hidden, props, capacity=32, slice_size=None):
    return _policy(
        hidden,
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
        capacity,
        shared_per_sm=getattr(
            props, "shared_memory_per_multiprocessor", props.shared_memory_per_block_optin
        ),
        regs_per_sm=getattr(props, "regs_per_multiprocessor", 65536),
        threads_per_sm=getattr(props, "max_threads_per_multi_processor", 1024),
        slice_size=slice_size,
    )


def support(device, *, hidden, hc_mult=4, dtype=torch.bfloat16) -> str | None:
    """Return a reason when this device/shape is unsupported; no CUDA probe on CPU."""
    if hc_mult != 4:
        return "mHC requires hc_mult == 4"
    if type(hidden) is not int or hidden <= 0 or hidden % 8:
        return "hidden must be a positive integer divisible by 8"
    if dtype != torch.bfloat16:
        return "mHC requires bfloat16"
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "mHC requires a CUDA device"
    if torch.cuda.get_device_capability(device)[0] < 9:
        return "mHC requires sm_90 or newer (PDL)"
    if not _cute_available():
        return "mHC requires nvidia-cutlass-dsl 4.8.0 (the cute extra)"
    props = torch.cuda.get_device_properties(device)
    try:
        _device_plan(hidden, props)
    except ValueError as exc:
        return str(exc)
    return None


def _tensor(name, value, shape, dtype, device, layout=True):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if tuple(value.shape) not in shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on the configured device")
    if not layout:
        return
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    if value.requires_grad:
        raise ValueError(f"{name}: mHC is inference-only")


def _validate(mode, hidden, device, inputs, outputs, layout=True):
    r = inputs["residual"]
    if not isinstance(r, torch.Tensor) or r.ndim != 3 or r.shape[0] < 1:
        raise ValueError("residual must have shape [T, 4, hidden] with T > 0")
    t = r.shape[0]
    if t * hidden * 8 >= 2**31:
        raise ValueError("tensor byte indexing must fit int32")
    bf, fp = torch.bfloat16, torch.float32
    specs = {
        "residual": (((t, 4, hidden),), bf),
        "x": (((t, hidden),), bf),
        "post": (((t, 4), (t, 4, 1)), fp),
        "comb": (((t, 4, 4),), fp),
        "fn": (((24, 4 * hidden),), fp),
        "scale": (((3,),), fp),
        "base": (((24,),), fp),
        "norm_weight": (((hidden,),), bf),
        "residual_out": (((t, 4, hidden),), bf),
        "post_out": (((t, 4),), fp),
        "comb_out": (((t, 4, 4),), fp),
        "out": (((t, hidden),), bf),
    }
    for name, value in inputs.items():
        _tensor(name, value, *specs[name], device, layout)
    supplied = {name: value for name, value in outputs.items() if value is not None}
    for name, value in supplied.items():
        _tensor(name, value, *specs[name], device, layout)
        # Layout-only, hence rank-symmetric under the same-calls contract, and
        # checked before any collective: a self-overlapping output can neither
        # be written by the kernel nor be the target of a staging copy-back.
        if internal_overlap(value):
            raise ValueError(f"{name} must not have internal overlap (expanded view)")
    if not layout:
        return t, {name: specs[name] for name in outputs}
    # Byte spans allow disjoint views in one allocation; reject any overlap
    # with an output, including two outputs sharing storage.
    all_values = list(inputs.items()) + list(supplied.items())
    for index, (name, value) in enumerate(all_values):
        if name not in supplied:
            continue
        for other, tensor in all_values[:index]:
            if spans_overlap(value, tensor):
                raise ValueError(f"{name} must not overlap {other}")
    return t, {name: specs[name] for name in outputs}


def _misplaced(value):
    return not value.is_contiguous() or value.data_ptr() % 16 != 0


_CAPTURE_LAYOUT = "unfused capture needs contiguous, 16-byte aligned, non-overlapping tensors"


def _stage_layout(tokens, hidden):
    """Byte offsets of every exit tensor in one staging buffer (256 B aligned)."""
    sizes = {
        "x": tokens * hidden * 2,
        "x_sum": tokens * hidden * 2,
        "residual": tokens * 4 * hidden * 2,
        "post": tokens * 16,
        "comb": tokens * 64,
        "fn": 24 * 4 * hidden * 4,
        "scale": 12,
        "base": 96,
        "norm_weight": hidden * 2,
        "residual_out": tokens * 4 * hidden * 2,
        "post_out": tokens * 16,
        "comb_out": tokens * 64,
        "out": tokens * hidden * 2,
    }
    offsets, end = {}, 0
    for name, size in sizes.items():
        offsets[name] = end
        end += (size + 255) // 256 * 256
    return offsets, end


_WEIGHTS = ("fn", "scale", "base", "norm_weight")


def _frozen(weights):
    """Trainable weights are rejected before any collective is issued or recorded.

    Weights are fixed per module and identical across ranks under the
    same-calls contract, so this check is rank-symmetric.
    """
    for name in _WEIGHTS:
        value = weights.get(name)
        if isinstance(value, torch.Tensor) and value.requires_grad:
            raise ValueError(f"{name}: mHC is inference-only (weight requires grad)")


def _stage(inputs, outputs, view, absorb=False):
    """Absorb layout facts before a collective exit.

    Outputs with internal overlap were rejected by _validate before this, so
    every copy-back here is a raw device copy planned and validated here
    (plan_copy), before any collective; after it only CopyPlan.run() runs.
    Overlap uses conservative byte spans (a false positive costs one copy).

    Contiguity, 16-byte alignment, overlap and requires_grad can differ between
    ranks, so they never decide a route or an early rejection once a collective
    may start. Inputs with a misplaced layout are copied into aligned staging,
    outputs with a misplaced layout or overlapping an input are written to
    staging and copied back afterwards. Eagerly, facts staging cannot absorb
    (requires_grad, outputs overlapping each other) become an error raised
    after the collective has completed on both ranks. With absorb (capture,
    where nothing runs and a local raise would strand the peer at replay) they
    raise nothing: activations and outputs are used as raw device memory, and
    an output overlapping an earlier one is written to staging and copied back
    after it, so later outputs win on the shared bytes.
    view(name, like) returns an aligned contiguous buffer, or None when no
    prepared staging exists (capture without prepare).
    """
    error, staged_in, copy_back = None, dict(inputs), []
    for name, value in inputs.items():
        if value.requires_grad and error is None and not absorb:
            error = f"{name}: mHC is inference-only"
        if _misplaced(value):
            buffer = view(name, value)
            if buffer is None:
                error = error or f"{name}: {_CAPTURE_LAYOUT}"
                continue
            raw_copy(buffer, value)
            staged_in[name] = buffer
    staged_out = dict(outputs)
    supplied = [(name, value) for name, value in outputs.items() if value is not None]
    for index, (name, value) in enumerate(supplied):
        if value.requires_grad and error is None and not absorb:
            error = f"{name}: mHC is inference-only"
        shared = any(spans_overlap(value, other) for _, other in supplied[:index])
        if shared and not absorb:
            error = error or f"{name} must not overlap another output"
        if (
            _misplaced(value)
            or (shared and absorb)
            or any(spans_overlap(value, other) for other in inputs.values())
        ):
            buffer = view(name, value)
            if buffer is None:
                error = error or f"{name}: {_CAPTURE_LAYOUT}"
                continue
            staged_out[name] = buffer
            copy_back.append(plan_copy(value, buffer))
    return staged_in, staged_out, copy_back, error


def _reference(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    *,
    exact,
    rms_eps,
    hc_eps,
    norm_eps,
    post_mult,
    sinkhorn_iters,
    mode,
):
    if mode not in {"exit", "pre", "post"}:
        raise ValueError("mode must be exit, pre or post")
    cfg = MhcConfig(residual.shape[-1], 4, sinkhorn_iters, hc_eps, rms_eps, norm_eps, post_mult)
    weights = {"fn": fn, "scale": scale, "base": base, "norm_weight": norm_weight}
    result = _exit(
        cfg,
        weights,
        x,
        residual,
        post,
        comb,
        torch.float64 if exact else torch.float32,
        not exact,
        mode,
    )
    names = ("residual",) if mode == "post" else ("post", "comb", "out")
    if mode == "exit":
        names = ("residual", *names)
    return tuple(result[name] for name in names)


def reference(
    x,
    residual,
    post=None,
    comb=None,
    fn=None,
    scale=None,
    base=None,
    norm_weight=None,
    *,
    rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    post_mult=2.0,
    sinkhorn_iters=20,
    mode="exit",
):
    """Torch FP32 composition with BF16 residual/y/out, the market error floor.

    Returns a tuple in API output order (a one-element tuple for mode='post').
    For mode='pre' pass x=None; post/comb are unused. Disable TF32 for GPU floors.
    """
    return _reference(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm_weight,
        exact=False,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        post_mult=post_mult,
        sinkhorn_iters=sinkhorn_iters,
        mode=mode,
    )


def exact_reference(
    x,
    residual,
    post=None,
    comb=None,
    fn=None,
    scale=None,
    base=None,
    norm_weight=None,
    *,
    rms_eps=1e-6,
    hc_eps=1e-6,
    norm_eps=1e-6,
    post_mult=2.0,
    sinkhorn_iters=20,
    mode="exit",
):
    """Unrounded FP64 oracle from the stored inputs, in the same order as reference."""
    return _reference(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm_weight,
        exact=True,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        post_mult=post_mult,
        sinkhorn_iters=sinkhorn_iters,
        mode=mode,
    )


def measured_floor(ours, market, exact, *, slack=None):
    """Per-output max-abs and RMS gate against the market path's error (vs ``exact``); returns
    (failures, flat statistics). ``slack[name] = (factor, ulps)`` relaxes an output's bound to
    ``max(factor * market, ulps * FP32 ULP of max |exact|)`` (default strict: ``(1, 0)``)."""
    names = {
        1: ("residual_out",),
        3: ("post_out", "comb_out", "out"),
        4: ("residual_out", "post_out", "comb_out", "out"),
    }
    if len(ours) not in names or len(market) != len(ours) or len(exact) != len(ours):
        raise ValueError("expected matching exit (4), pre (3), or post (1) output tuples")
    failures, stats = [], {}
    for name, a, m, e in zip(names[len(ours)], ours, market, exact, strict=True):
        if a.shape != e.shape or m.shape != e.shape or e.numel() == 0:
            raise ValueError(f"{name}: nonempty output shapes must match")
        oracle = e.detach().cpu().double()
        da, dm = (v.detach().cpu().double() - oracle for v in (a, m))
        for kind, mine, theirs in (
            ("max", da.abs().max().item(), dm.abs().max().item()),
            ("rms", da.square().mean().sqrt().item(), dm.square().mean().sqrt().item()),
        ):
            stats[f"{name}_{kind}"] = mine
            stats[f"{name}_{kind}_market"] = theirs
            factor, ulps = (slack or {}).get(name, (1.0, 0))
            top = oracle.abs().max().float()
            ulp = (torch.nextafter(top, torch.tensor(math.inf)) - top).item()
            bound = max(theirs * factor, ulps * ulp)
            if not math.isfinite(mine) or not math.isfinite(theirs) or mine > bound:
                failures.append(f"{name} {kind} error {mine:.3e} exceeds market {bound:.3e}")
    return failures, stats


class MhcExit:
    """Prepared CuTe provider; one stream at a time per instance, including replay.

    Outputs supplied by callers must not alias inputs or each other. No tensors
    are retained except the private arrival counters and partial-sum workspace.
    Each mode/capacity bucket is compiled once; T remains a runtime argument.
    decode_slice optionally selects 64/128 columns. prefill_tile optionally
    selects 1/2/4/8/16/32 tokens and overrides small-prefill dispatch.
    Exit defaults to grid exchange through T=256, then adaptive prefill.
    CTI_MHC_DECODE=cluster selects cluster exchange for T<=8. Pre uses grid decode
    through T=32 and adaptive prefill above it; post is unchanged.

    all_reduce accepts a TwoNodeAllReduce provider and treats x as this rank's
    partial. Both ranks must issue the same calls in the same order on ordered
    streams. A fused grid decode replaces B1 all-reduce followed by exit in one
    launch and is bitwise equal to that composition. Other buckets/sizes use
    the provider followed by exit. prepare(counts, all_reduce=provider) compiles
    fused variants without launching a collective; the first fused eager call
    is a real collective that both ranks must issue. Capture requires prepare
    and supplied outputs; the B1 device sequence makes replay graph-safe.
    With all_reduce, only rank-symmetric facts (shape, dtype, configured device)
    raise before the collective. Local layout (contiguity, alignment, overlap)
    is copied through aligned staging sized at prepare (fused buckets: the
    provider's largest fused T; unfused buckets: each prepared T, used under
    capture; eager unfused calls stage into fresh buffers). A trainable weight
    raises before the collective (rank-symmetric: weights are per module), and
    prepare(weights=) rejects it before any capture. Eagerly, requires_grad
    activations/outputs and outputs overlapping each other raise after the
    collective; under capture they raise nothing (outputs overlapping an
    earlier one go through staging and are copied back after it).
    """

    def __init__(
        self,
        device,
        *,
        hidden,
        rms_eps,
        hc_eps,
        norm_eps,
        post_mult=2.0,
        sinkhorn_iters=20,
        decode_slice=None,
        prefill_tile=None,
    ):
        self.cfg = MhcConfig(hidden, 4, sinkhorn_iters, hc_eps, rms_eps, norm_eps, post_mult)
        reason = support(device, hidden=hidden)
        if reason is not None:
            raise ValueError(reason)
        if decode_slice not in (None, 64, 128):
            raise ValueError("decode_slice must be 64 or 128")
        if decode_slice is None and "CTI_MHC_DECODE_SLICE" in os.environ:
            decode_slice = int(os.environ["CTI_MHC_DECODE_SLICE"])
        if prefill_tile is None and "CTI_MHC_PREFILL_TILE" in os.environ:
            prefill_tile = int(os.environ["CTI_MHC_PREFILL_TILE"])
        if prefill_tile not in (None, 1, 2, 4, 8, 16, 32):
            raise ValueError("prefill_tile must be 1, 2, 4, 8, 16 or 32")
        self.decode = os.environ.get(
            "CTI_MHC_DECODE", "grid"
        )  # measured faster on GB10 (cluster: A/B only)
        if self.decode not in ("cluster", "grid"):
            raise ValueError("CTI_MHC_DECODE must be cluster or grid")
        self._sms = 48
        self.decode_slice, self.prefill_tile = decode_slice, prefill_tile
        self._plans = {}
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden = hidden
        self._compiled = {}
        self._counter = self._partials = self._ysum = self._dummy_words = None
        self._counter_views = {}
        # Collective-exit staging; replaced buffers stay alive for captured graphs.
        self._stage_buf, self._stage_cap, self._stage_old = None, 0, []
        self._closed = False
        self._plan = None

    def _require_open(self):
        if self._closed:
            raise RuntimeError("MhcExit used after close()")

    def prepare(self, token_counts, all_reduce=None, *, weights=None):
        """Warm ordinary modes; compile fused variants without collective launches.

        weights optionally maps fn/scale/base/norm_weight to the module's
        weights; a trainable one raises here, before any capture.
        """
        if weights is not None:
            _frozen(weights)
        self._prepare(token_counts, all_reduce, stage_unfused=True)

    def _prepare(self, token_counts, all_reduce, stage_unfused):
        self._require_open()
        counts = tuple(token_counts)
        if any(type(t) is not int or t < 1 or t * self.hidden * 8 >= 2**31 for t in counts):
            raise ValueError("token counts must be positive integers within int32 indexing")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare must run outside CUDA graph capture")
            if not counts:
                return
            if self._plan is None:
                p = torch.cuda.get_device_properties(self.device)
                self._sms = p.multi_processor_count
                self._plan = _device_plan(self.hidden, p)
                pitch = ((self.hidden + 63) // 64 + 3) // 4 * 4
                # A monotonic counter's epoch depends on G. Grid capacities
                # own separate counter/flag pairs; modes share the same epoch
                # and plan for a capacity. Cluster kernels never access them.
                self._counter = torch.zeros(12, dtype=torch.int64, device=self.device)
                self._dummy_words = self._counter.narrow(0, 0, 4)
                self._ysum = torch.empty(
                    (256, self.hidden), device=self.device, dtype=torch.bfloat16
                )
                self._counter_views = {
                    cap: self._counter.narrow(0, 2 * index, 2)
                    for index, cap in enumerate((4, 16, 32, 64, 128, 256))
                }
                # Modes are stream-ordered; pre needs only 32 token rows.
                # Each kernel puts gains after its cap * 35 * pitch partials;
                # reserve the largest partial region plus 256 * 5 gains (8-byte slots).
                self._partials = torch.empty(
                    256 * 35 * pitch + 256 * 5 * 2, device=self.device, dtype=torch.float32
                )
            # The exchange grid requires simultaneous residency. Check requested
            # capacities before warming any new kernel, also on later prepare calls.
            for t in counts:
                for mode in ("exit", "pre"):
                    cap = self._key(mode, t)[1]
                    if cap > 0 and cap not in self._plans:
                        p = torch.cuda.get_device_properties(self.device)
                        if cap > 1000:
                            plan = _cluster_plan(self.hidden, cap - 1000)
                            size, _, capacity = plan
                            shared = capacity * (4 * size + 8 * 35 + 70 + 5) * 4
                            shared += size * 4 + 128 + 2048
                            if shared > p.shared_memory_per_block_optin:
                                raise ValueError("insufficient shared memory for cluster decode")
                            self._plans[cap] = plan
                        else:
                            self._plans[cap] = _device_plan(self.hidden, p, cap, self.decode_slice)
            from ._cute.kernels import compile_kernel

            for t in counts:
                for mode in ("exit", "pre", "post"):
                    key = self._compile_key(mode, t)
                    if key not in self._compiled:
                        plan = self._plans.get(key[1], self._plan)
                        self._compiled[key] = compile_kernel(
                            self.cfg,
                            plan,
                            mode,
                            key[1] > 0,
                            token_tile=-key[1] if key[1] < 0 else 16,
                            cluster=key[1] > 1000,
                        )
                        self._warm(mode, t)
                if all_reduce is not None:
                    nbytes = t * self.hidden * 2
                    if self._fused(t, torch.bfloat16, all_reduce):
                        self._grow_stage(min(256, all_reduce.max_bytes // (2 * self.hidden)))
                    elif stage_unfused:
                        # Capture of unfused buckets stages local layout here.
                        self._grow_stage(t)
                    if nbytes <= all_reduce.max_bytes and nbytes % 16 == 0:
                        all_reduce.prepare([nbytes])
                    if self._fused(t, torch.bfloat16, all_reduce):
                        key = self._compile_key("exit", t, allreduce=True)
                        if key not in self._compiled:
                            self._compiled[key] = compile_kernel(
                                self.cfg, self._plans[key[1]], "exit", True, allreduce=True
                            )
                            # Warming this variant would issue a real collective.

    def _key(self, mode, tokens):
        cap = 0
        if mode == "exit" and tokens <= 8 and self.decode == "cluster":
            cap = 1000 + (1 if tokens == 1 else 2 if tokens == 2 else 4 if tokens <= 4 else 8)
        elif mode != "post" and tokens <= 32:
            cap = 4 if tokens <= 4 else 16 if tokens <= 16 else 32
        elif mode == "exit" and tokens <= 256 and self.prefill_tile is None:
            cap = 64 if tokens <= 64 else 128 if tokens <= 128 else 256
        elif mode != "post":
            cap = -(self.prefill_tile or _prefill_tile(tokens, self._sms))
        return mode, cap

    def _fused(self, tokens, dtype, all_reduce):
        # Agreed active, T, H, dtype and bucket are rank-symmetric facts.
        # Local storage offset/alignment never selects a different transport.
        nbytes = tokens * self.hidden * 2
        return (
            all_reduce is not None
            and all_reduce.active
            and self._key("exit", tokens)[1] in (4, 16, 32, 64, 128, 256)
            and dtype == torch.bfloat16
            and nbytes <= all_reduce.max_bytes
            and nbytes % 16 == 0
        )

    def _compile_key(self, mode, tokens, allreduce=False):
        # Match Kernel's construction-time environment reads, including existing
        # knobs. Changing a knob selects a distinct prepared variant; capture
        # requires that exact variant to have been prepared beforehand.
        return (*self._key(mode, tokens), allreduce, kernel_options())

    def _warm(self, mode, tokens):
        # Finite dummy data; all allocations/launches happen before capture.
        bf, fp = torch.bfloat16, torch.float32

        def zeros(shape, dtype):
            return torch.zeros(shape, device=self.device, dtype=dtype)

        r = zeros((tokens, 4, self.hidden), bf)
        post_args = {
            "x": zeros((tokens, self.hidden), bf),
            "post": zeros((tokens, 4), fp),
            "comb": zeros((tokens, 4, 4), fp),
        }
        weights = {
            "fn": zeros((24, 4 * self.hidden), fp),
            "scale": zeros((3,), fp),
            "base": zeros((24,), fp),
            "norm_weight": zeros((self.hidden,), bf),
        }
        if mode == "pre":
            self.pre(r, **weights)
        elif mode == "post":
            self.post(residual=r, **post_args)
        else:
            self(residual=r, **post_args, **weights)

    def _grow_stage(self, tokens):
        if tokens > self._stage_cap:
            if self._stage_buf is not None:
                self._stage_old.append(self._stage_buf)
            size = _stage_layout(tokens, self.hidden)[1]
            self._stage_buf = torch.empty(size, dtype=torch.uint8, device=self.device)
            self._stage_cap = tokens

    def _stage_view(self, tokens):
        if self._stage_cap < tokens:
            return lambda name, like: None
        offsets, _ = _stage_layout(self._stage_cap, self.hidden)

        def view(name, like):
            size = like.numel() * like.element_size()
            start = offsets[name]
            return self._stage_buf[start : start + size].view(like.dtype).view(like.shape)

        return view

    def _eager_view(self):
        return lambda name, like: torch.empty(like.shape, dtype=like.dtype, device=self.device)

    def _call(self, mode, inputs, outputs, all_reduce=None):
        self._require_open()
        # With a collective, only rank-symmetric facts (shape, dtype, configured
        # device) may raise before it; local layout is staged (_stage).
        collective = mode == "exit" and all_reduce is not None
        t, specs = _validate(mode, self.hidden, self.device, inputs, outputs, layout=not collective)
        if collective:
            _frozen(inputs)
        fused = mode == "exit" and self._fused(t, inputs["x"].dtype, all_reduce)
        if fused:
            if getattr(all_reduce, "_closed", False):
                raise RuntimeError("all-reduce provider is closed")
            if all_reduce.device != self.device:
                raise ValueError("all-reduce provider must be on the configured device")
        with torch.cuda.device(self.device):
            capture = torch.cuda.is_current_stream_capturing()
            if capture and any(v is None for v in outputs.values()):
                raise ValueError("all output buffers must be supplied during CUDA graph capture")
            if (
                self._plan is None
                or self._compile_key(mode, t, fused) not in self._compiled
                or (collective and (fused or capture) and self._stage_cap < t)
            ):
                if capture:
                    raise ValueError("call prepare(token_counts) before CUDA graph capture")
                self._prepare([t], all_reduce, stage_unfused=False)
            for name, value in outputs.items():
                if value is None:
                    shapes, dtype = specs[name]
                    outputs[name] = torch.empty(shapes[0], dtype=dtype, device=self.device)
            results, error, copy_back = dict(outputs), None, []
            if collective and not fused:
                # Every input is staged and every copy planned before the
                # collective: capture uses the staging prepared for t (no
                # allocation), eager fresh buffers. The sum lands in a dense
                # aligned slot (a misplaced x is copied there and reduced in
                # place; an in-place fallback returns its aligned input), so
                # nothing layout-dependent follows the collective. Local
                # errors raise after it, like the fused path.
                view = self._stage_view(t) if capture else self._eager_view()
                x = inputs["x"]
                rest = {name: value for name, value in inputs.items() if name != "x"}
                rest, outputs, copy_back, error = _stage(rest, outputs, view, capture)
                x_sum = view("x_sum", x)
                if x_sum is None:
                    raise ValueError(f"x: {_CAPTURE_LAYOUT}")
                if _misplaced(x):
                    raw_copy(x_sum, x)
                    x = x_sum
                inputs = {"x": all_reduce(x, x_sum), **rest}
                if error is not None:
                    raise ValueError(error)
            elif fused:
                view = self._stage_view(t)
                inputs, outputs, copy_back, error = _stage(inputs, outputs, view, capture)
            args = {**inputs, **outputs}
            order = (
                "x",
                "residual",
                "post",
                "comb",
                "fn",
                "scale",
                "base",
                "norm_weight",
                "residual_out",
                "post_out",
                "comb_out",
                "out",
            )
            pointers = [args[name].data_ptr() if name in args else 0 for name in order]
            key = self._compile_key(mode, t, fused)
            counter = self._counter_views.get(key[1], self._counter_views.get(4, self._counter))
            ar_args = (self._dummy_words, 0, 0, 0, 0)
            if fused:
                ar_args = (
                    all_reduce.words,
                    all_reduce.base,
                    all_reduce.layout[-1],
                    all_reduce.rank,
                    self._ysum.data_ptr(),
                )
            self._compiled[key](*pointers, self._partials.data_ptr(), counter, t, *ar_args)
            # Raw device copies: no torch tensor rule (inference mode,
            # autograd, version counters) can raise after the collective.
            for plan in copy_back:
                plan.run()
            if error is not None:
                raise ValueError(error)
        return tuple(results.values())

    def __call__(
        self,
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm_weight,
        *,
        residual_out=None,
        post_out=None,
        comb_out=None,
        out=None,
        all_reduce=None,
    ):
        return self._call(
            "exit",
            {
                "x": x,
                "residual": residual,
                "post": post,
                "comb": comb,
                "fn": fn,
                "scale": scale,
                "base": base,
                "norm_weight": norm_weight,
            },
            {"residual_out": residual_out, "post_out": post_out, "comb_out": comb_out, "out": out},
            all_reduce=all_reduce,
        )

    def pre(
        self, residual, fn, scale, base, norm_weight, *, post_out=None, comb_out=None, out=None
    ):
        return self._call(
            "pre",
            {
                "residual": residual,
                "fn": fn,
                "scale": scale,
                "base": base,
                "norm_weight": norm_weight,
            },
            {"post_out": post_out, "comb_out": comb_out, "out": out},
        )

    def post(self, x, residual, post, comb, *, residual_out=None):
        return self._call(
            "post",
            {"x": x, "residual": residual, "post": post, "comb": comb},
            {"residual_out": residual_out},
        )[0]

    def close(self):
        """Synchronize before releasing workspace. Do not replay graphs after close."""
        if not self._closed:
            if self._counter is not None:
                torch.cuda.synchronize(self.device)
            self._counter = self._partials = self._ysum = self._dummy_words = None
            self._stage_buf, self._stage_cap, self._stage_old = None, 0, []
            self._counter_views.clear()
            self._compiled.clear()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
