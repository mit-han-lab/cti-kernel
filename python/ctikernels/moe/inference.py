# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""BF16 frozen-expert composition shared by inference callers."""

from contextlib import nullcontext as _nullcontext
from itertools import pairwise

import torch

from ctikernels._runtime.workspace import GraphWorkspace

from .._runtime.launch import prime
from ._align import _validate_routes
from ._block import admits_shape, base_tiles, expert_tiles, hardware, validate_block
from ._common import ExpertWeights, deinterleave_scales, interleave_scales_
from ._configs import TUNED
from ._dispatch import _BLOCK_TABLE
from ._lora_common import device_info
from ._lora_weights import MoELoRAWeights, select_adapter
from .cute._lora_provider import supports_rank as _cute_lora_rank
from .cute._w4a8 import Config as W4A8Config
from .cute._w4a8_lora import W4A8LoRAExperts
from .lora import moe_lora_sorted_shrink_


def _experts(
    op,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_map: torch.Tensor | None = None,
    activation: str = "silu",
    out: torch.Tensor | None = None,
    config: dict | None = None,
    shared: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
    wait_stream=None,
) -> torch.Tensor:
    """Apply SwiGLU experts and combine only routes computed on this rank (vLLM's numerics:
    unweighted BF16 expert rows, FP32 FMA combine in route order, one rounding).

    ``w1`` is [E_local, 2I, H], ``w2`` is [E_local, H, I]. Global
    ``topk_ids`` index ``expert_map`` (local expert index, or -1 for remote).
    Routing IDs may themselves be -1 for graph padding when a map is supplied.
    The default output owns new storage. ``out`` may alias ``hidden_states``:
    reduction writes it only after both projections have finished reading input.
    This is inference only; weights must be frozen. Warm up before graph capture.
    """
    if w1.requires_grad or w2.requires_grad:
        raise ValueError("expert weights must be frozen (requires_grad=False)")
    if activation != "silu":
        raise ValueError("activation must be 'silu'")
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("expected hidden_states [T, H], w1 [E, 2I, H], w2 [E, H, I]")
    t, h = hidden_states.shape
    e, twice_i, wh = w1.shape
    i = twice_i // 2
    if not e or not h or not i or twice_i % 2 or wh != h or w2.shape != (e, h, i):
        raise ValueError("incompatible expert weight shapes")
    if topk_ids.ndim != 2 or topk_ids.shape[0] != t or topk_ids.shape[1] == 0:
        raise ValueError("topk_ids must have shape [T, top_k] with top_k > 0")
    if topk_weights.shape != topk_ids.shape or topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must be float32 with the same shape as topk_ids")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32")
    if out is not None and (out.shape != hidden_states.shape or out.requires_grad):
        raise ValueError("out must have shape [T, H] and not require gradients")
    tensors = [hidden_states, w1, w2] + ([] if out is None else [out])
    if any(x.dtype != torch.bfloat16 for x in tensors):
        raise ValueError("hidden_states, weights and out must be BF16")
    tensors += [topk_ids, topk_weights]
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("all tensors must be contiguous")
    global_experts = e if expert_map is None else expert_map.numel()
    base_tiles(t, global_experts, config)  # an invalid config fails before any device check
    if expert_map is not None:
        if e > 512:
            raise ValueError("expert_map alignment supports at most 512 local experts")
        _validate_routes(topk_ids, expert_map)
    if not hidden_states.is_cuda or any(x.device != hidden_states.device for x in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")
    up, down = expert_tiles(t, e, i, global_experts, hidden_states.device, config)
    op.provider = "triton"
    op.launch_config = {"up": dict(up), "down": dict(down)}
    if out is None:
        out = torch.empty_like(hidden_states)
    if t == 0:
        return out

    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.moe_impl import (
        fused_moe_kernel_triton,
        moe_align_block_size,
        moe_sum_reduce_triton,
    )

    top_k = topk_ids.shape[1]
    aligned = moe_align_block_size(topk_ids, up["BLOCK_SIZE_M"], e, expert_map=expert_map)
    aligned_down = (
        aligned
        if up["BLOCK_SIZE_M"] == down["BLOCK_SIZE_M"]
        else moe_align_block_size(topk_ids, down["BLOCK_SIZE_M"], e, expert_map=expert_map)
    )
    fc1, fc2, activated = op._intermediates(hidden_states, t, top_k, i, h)
    fused_moe_kernel_triton(
        hidden_states,
        w1,
        fc1,
        topk_weights,
        topk_ids,
        *aligned,
        False,
        top_k,
        up,
        torch.bfloat16,
    )
    silu_and_mul(fc1.view(t * top_k, 2 * i), out=activated)
    fused_moe_kernel_triton(
        activated,
        w2,
        fc2,
        topk_weights,
        topk_ids,
        *aligned_down,
        False,
        1,
        down,
        torch.bfloat16,
    )
    if wait_stream is not None:
        torch.cuda.current_stream(hidden_states.device).wait_stream(wait_stream)
    moe_sum_reduce_triton(
        fc2,
        out,
        topk_ids=topk_ids,
        expert_map=expert_map,
        topk_weights=topk_weights,
        shared=shared,
        shared_gate=shared_gate,
        valid_tokens=valid_tokens,
    )
    return out


def _allocate(spec, counts, device):
    """Tensors for (name, elements, dtype, fill) entries at ``counts`` elements: fill None is
    uninitialized, "arange" counts up, a tuple sets the leading values over zeros."""
    tensors = []
    for (_, _, dtype, fill), count in zip(spec, counts, strict=True):
        if fill is None:
            tensor = torch.empty(count, dtype=dtype, device=device)
        elif fill == "arange":
            tensor = torch.arange(count, dtype=dtype, device=device)
        elif isinstance(fill, tuple):
            tensor = torch.zeros(count, dtype=dtype, device=device)
            tensor[: len(fill)] = torch.tensor(fill, dtype=dtype)
        else:
            tensor = torch.full((count,), fill, dtype=dtype, device=device)
        tensors.append(tensor)
    return tuple(tensors)


class Experts:
    """Frozen routed SwiGLU experts; intermediate storage grows per device/stream.

    Warm-up buffers are reused during capture. If too small, capture allocates
    private graph-pool buffers without replacing the eager cache. A cached buffer
    that a capture used stays allocated for this object's lifetime, even after a
    larger eager call supersedes it, so a captured graph never touches freed memory.
    Serialize uses of a captured graph and eager calls sharing its warmed buffers.
    """

    def __init__(self, provider="auto", config=None):
        if provider not in ("auto", "triton"):
            raise ValueError("Experts provider must be auto or triton")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a dictionary or None")
        self.provider = provider
        self.launch_config = None
        self.config = config
        self._buffers = {}
        # Cached storage a captured graph may use: never freed while this object lives.
        self._graph_storage = {}

    def _intermediates(self, x, rows, top_k, inter, hidden):
        key = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
        sizes = (rows * top_k * 2 * inter, rows * top_k * hidden, rows * top_k * inter)
        # FC2 holds the unweighted routes in the input dtype (vLLM's C); the reduction weights
        # and combines them.
        dtypes = (x.dtype, x.dtype, x.dtype)
        previous = self._buffers.get(key, (None, None, None))
        buffers = tuple(
            old
            if old is not None and old.numel() >= size
            else torch.empty(size, dtype=dtype, device=x.device)
            for old, size, dtype in zip(previous, sizes, dtypes, strict=True)
        )
        if torch.cuda.is_current_stream_capturing():
            for old, new in zip(previous, buffers, strict=True):
                if new is old:
                    self._graph_storage[(new.data_ptr(), new.numel())] = new
        else:
            self._buffers[key] = buffers
        return (
            buffers[0][: sizes[0]].view(rows, top_k, 2 * inter),
            buffers[1][: sizes[1]].view(rows, top_k, hidden),
            buffers[2][: sizes[2]].view(rows * top_k, inter),
        )

    def __call__(
        self,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        *,
        expert_map=None,
        out=None,
        shared=None,
        shared_gate=None,
        valid_tokens=None,
        wait_stream=None,
    ):
        """Optional ``shared`` [T,H] and ``shared_gate`` logits [T,1] join the routed output in
        vLLM's BF16 op order (``r(r(r(sigmoid(gate)) * shared) + routed)``); ``valid_tokens``
        zeroes padded rows; ``wait_stream`` is a stream the reduction waits for (the one
        producing ``shared``)."""
        if w1.requires_grad or w2.requires_grad:
            raise ValueError("expert weights must be frozen (requires_grad=False)")
        with torch.cuda.device(hidden_states.device) if hidden_states.is_cuda else _nullcontext():
            return _experts(
                self,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_map=expert_map,
                out=out,
                config=self.config,
                shared=shared,
                shared_gate=shared_gate,
                valid_tokens=valid_tokens,
                wait_stream=wait_stream,
            )


def fused_experts(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    expert_map=None,
    activation="silu",
    out=None,
    config=None,
):
    """Functional Experts API. Retain an Experts object to reuse intermediate storage."""
    if w1.requires_grad or w2.requires_grad:
        raise ValueError("expert weights must be frozen (requires_grad=False)")
    if activation != "silu":
        raise ValueError("activation must be 'silu'")
    return Experts(config=config)(
        hidden_states, w1, w2, topk_weights, topk_ids, expert_map=expert_map, out=out
    )


class MoEBlock:
    """Router + top-8 + routed and gated shared experts, producing one EP partial.

    The caller performs the exit all-reduce. Shared weights are sliced along I
    across ranks; every rank computes its shared slice for every valid row.
    ``routing_ids`` exposes the latest routing record (padding is unspecified for
    triton_block, -1 for composed) and ``routing_weights`` its FP32 weights (None for a
    block-tree call longer than one chunk). Both providers route with the same arithmetic
    (``route``): a token's experts and weights do not depend on the row count or provider. ``kernels_per_call`` is exact for the block
    trees in steady state (excluding workspace initialization), and None for
    composition, whose library launch counts vary.

    Block-tree workspace. Inside an active ``GraphWorkspace``, calls of at most its
    ``max_rows`` rows use the scope's ("moe_block", device, dtype) slot, shared by every
    MoEBlock in the scope and sized at eager warm-up for ``max_rows``; every in-scope eager
    call of a shape the block tree admits reserves it, whatever provider its own row count
    takes. ``routing_ids`` then views the shared slot until the next block-tree call in the
    scope. Outside a scope (and for longer eager calls), the op keeps one workspace per device
    and block shape, sized by eager calls to the most rows seen; captures reuse it, and
    capturing more rows than that raises RuntimeError. A larger eager call replaces it,
    keeping the old one alive only if a graph captured it. Eager calls, captures and replays
    sharing a workspace must be stream-ordered.
    """

    def __init__(self, provider="auto"):
        if provider not in ("auto", "triton_block", "composed"):
            raise ValueError("MoEBlock provider must be auto, triton_block or composed")
        self._requested = provider
        self.provider = provider
        self.kernels_per_call = None
        self.routing_ids = None
        self.routing_weights = None
        self._experts = Experts()
        self._workspaces = {}
        self._identities = {}
        self._blocks = {}
        # Workspaces a graph captured before an eager call outgrew them: kept for the graphs.
        self._retired = []
        self._streams = {}

    def warm_up(
        self, max_rows, router_w, w_gate_up, w_down, shared_gate_up, shared_down,
        shared_gate_w, *, expert_map=None,
    ):  # fmt: skip
        """Run one call per kernel variant that calls of 1..``max_rows`` rows reach, on zeroed
        rows, eagerly, before capture (see :meth:`warm_rows`).

        The block's Triton kernels compile per tile configuration and argument
        specialization, and CUDA loads each at its first launch, which waits for all queued
        device work; without this a serving call of a new row count stalls (``_runtime.launch``).
        Also sizes the eager intermediates for ``max_rows``."""
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError(f"max_rows must be a positive integer, not {max_rows!r}")
        weights = (router_w, w_gate_up, w_down, shared_gate_up, shared_down, shared_gate_w)
        x = torch.zeros(max_rows, router_w.shape[1], dtype=router_w.dtype, device=router_w.device)
        shapes = tuple(tuple(w.shape) for w in weights)
        rows = self.warm_rows(max_rows, x.device, *w_gate_up.shape[:2])
        # Per op: its workspace, sized by the largest call first, shapes some kernel arguments.
        for t in [max_rows, *rows]:
            prime(("moe", self, x.device, shapes, self.decision(t, x.device), t), self, x[:t],
                  *weights, expert_map=expert_map)  # fmt: skip

    def warm_rows(self, max_rows, device, experts, twice_inter):
        """Row counts covering every kernel variant of calls up to ``max_rows`` rows: each
        block-tree count, and around every composed switch (the first composed count, each
        tuned tile table's nearest-key midpoint, each power of two the route and reduce
        kernels size by) the count, the next and the next multiple of 16 (Triton
        specializes integer arguments by divisibility by 16)."""
        kind = hardware(device)
        block = max(
            (
                maximum
                for maximum, provider in self._block_table(kind)
                if provider == "triton_block"
            ),
            default=0,
        )
        rows = set(range(1, min(block, max_rows) + 1))
        edges = {block + 1}
        power = 1
        while power <= max_rows:
            edges.add(power + 1)
            power *= 2
        tables = TUNED.get(kind, {}).get((experts, twice_inter // 2), {})
        for phase in ("up", "down"):
            keys = sorted(tables.get(phase, {}))
            edges.update((a + b) // 2 + 1 for a, b in pairwise(keys))
        for edge in edges:
            rows.update((edge, edge + 1, -(-edge // 16) * 16))
        return sorted((r for r in rows if r <= max_rows), reverse=True)

    def _block_table(self, kind):
        return _BLOCK_TABLE.get(kind, []) if self._requested == "auto" else (
            [(2**62, self._requested)] if self._requested == "triton_block" else [])  # fmt: skip

    def decision(self, T, device):
        """Shape-independent policy choice; actual calls also check weight admission.

        ``device`` may be a device-class string for offline inspection.
        """
        from ._dispatch import _BLOCK_TABLE

        if not isinstance(T, int) or T < 0:
            raise ValueError("T must be a nonnegative integer")
        if self._requested != "auto":
            return self._requested
        if device in ("rtx5090", "pro6000", "gb10", "unknown"):
            kind = device
        else:
            device = torch.device(device)
            kind = hardware(device) if device.type == "cuda" else None
        for maximum, provider in _BLOCK_TABLE.get(kind, []):
            if T <= maximum:
                return provider
        return "composed"

    def _block(self, device, kind, shape):
        """(backend, call) when the block tree admits this shape at some row count, else None;
        cached per device and row-independent shape, off the per-call path."""
        key = (
            device,
            shape["hidden"],
            shape["experts"],
            shape["inter"],
            shape["local_experts"],
            shape["shared_inter"],
            shape["dtype"],
        )
        if key not in self._blocks:
            block = None
            if (
                self._requested != "composed"
                and admits_shape(kind, **{**shape, "rows": 1})
                and (self._requested == "triton_block" or self.decision(1, kind) == "triton_block")
            ):
                from . import triton_block, triton_block_gb10

                backend = triton_block_gb10 if kind == "gb10" else triton_block
                call = (
                    backend.chunk(device),
                    shape["inter"],
                    shape["shared_inter"],
                    shape["local_experts"],
                )
                block = backend, call
            self._blocks[key] = block
        return self._blocks[key]

    def _identity(self, device, experts, capturing):
        """Identity expert map, made eagerly: a capture's fill would not run before eager use."""
        identity = self._identities.get((device, experts))
        if identity is None:
            if capturing:
                raise RuntimeError("MoEBlock: run an eager call of this shape before capture")
            identity = torch.arange(experts, dtype=torch.int32, device=device)
            self._identities[(device, experts)] = identity
        return identity

    @staticmethod
    def _spec(backend, rows, chunk, inter, shared_inter, local):
        return (
            *backend.workspace_spec(rows, chunk, inter, shared_inter, local),
            ("routing", rows * 8, torch.uint8, None),
            ("counts", rows + 1, torch.int32, "arange"),
        )

    @staticmethod
    def _bind(backend, spec, tensors, rows, call):
        tensors = dict(zip((name for name, *_ in spec), tensors, strict=True))
        workspace = backend.bind_workspace(tensors, rows, *call)
        workspace["routing"], workspace["counts"] = tensors["routing"], tensors["counts"]
        return workspace

    def _workspace(self, backend, device, rows, call, capturing):
        """The op's own workspace: sized by eager calls, reused by captures."""
        workspace = self._workspaces.get((device, *call))
        if workspace is None or workspace["shape"][0] < rows:
            if capturing:
                raise RuntimeError(
                    f"MoEBlock: capturing {rows} rows needs an eager call of at least {rows} "
                    "rows first (the triton_block workspace is sized eagerly)"
                )
            if workspace is not None and workspace["captured"]:
                self._retired.append(workspace)
            spec = self._spec(backend, rows, *call)
            tensors = _allocate(spec, tuple(n for _, n, *_ in spec), device)
            workspace = self._bind(backend, spec, tensors, rows, call)
            workspace["captured"] = False
            self._workspaces[(device, *call)] = workspace
        workspace["captured"] |= capturing
        return workspace

    def _scope_workspace(self, scope, backend, x, call):
        """The scope's shared slot, bound for ``scope.max_rows`` rows."""
        rows = scope.max_rows
        slot = ("moe_block", x.device, x.dtype)
        spec = self._spec(backend, rows, *call)
        tensors = scope.take(slot, x.shape[0], tuple(n for _, n, *_ in spec))
        return self._bind(backend, spec, tensors, rows, call)

    def _reserve(self, scope, backend, x, call):
        def size(max_rows, _backend, *call):
            return tuple(n for _, n, *_ in self._spec(backend, max_rows, *call))

        def allocate(counts):
            return _allocate(self._spec(backend, scope.max_rows, *call), counts, x.device)

        scope.reserve(("moe_block", x.device, x.dtype), (backend.__name__, *call), size, allocate)

    def __call__(
        self,
        x,
        router_w,
        w_gate_up,
        w_down,
        shared_gate_up,
        shared_down,
        shared_gate_w,
        *,
        expert_map=None,
        out=None,
        valid_tokens=None,
    ):
        # Validate frozen weights and metadata before querying CUDA or allocating anything.
        shape = validate_block(
            x,
            router_w,
            w_gate_up,
            w_down,
            shared_gate_up,
            shared_down,
            shared_gate_w,
            expert_map=expert_map,
            valid_tokens=valid_tokens,
            out=out,
        )
        kind = hardware(x.device)
        provider = self.decision(x.shape[0], kind or "unknown")
        scope = GraphWorkspace.active()
        capturing = torch.cuda.is_current_stream_capturing()
        block = self._block(x.device, kind, shape)
        if block is not None and scope is not None and not capturing:
            self._reserve(scope, block[0], x, block[1])
        if provider == "triton_block" and not admits_shape(kind, **shape):
            if self._requested == "triton_block":
                raise ValueError("triton_block does not admit this device or shape")
            provider = "composed"
        self.provider = provider
        with torch.cuda.device(x.device), torch.no_grad():
            if out is None:
                out = torch.empty_like(x)
            if x.shape[0] == 0:
                self.kernels_per_call = 0
                self.routing_ids = torch.empty((0, 8), dtype=torch.int32, device=x.device)
                self.routing_weights = torch.empty((0, 8), dtype=torch.float32, device=x.device)
                return out
            if provider == "triton_block":
                backend, call = block
                rows = x.shape[0]
                if scope is not None and (rows <= scope.max_rows or capturing):
                    workspace = self._scope_workspace(scope, backend, x, call)
                else:
                    workspace = self._workspace(backend, x.device, rows, call, capturing)
                if valid_tokens is None:
                    valid_tokens = workspace["counts"][rows : rows + 1]
                if expert_map is None:
                    expert_map = self._identity(x.device, router_w.shape[0], capturing)
                self.routing_ids = workspace["routing"][: rows * 8].view(rows, 8)
                self.routing_weights = backend.routing_weights(workspace, x.shape[0])
                backend.run_block(
                    x,
                    router_w,
                    shared_gate_w,
                    w_gate_up,
                    w_down,
                    shared_gate_up,
                    shared_down,
                    expert_map,
                    valid_tokens,
                    out,
                    self.routing_ids,
                    workspace,
                )
                self.kernels_per_call = 1 if kind == "gb10" else 2
            else:
                self._composed(
                    x,
                    router_w,
                    w_gate_up,
                    w_down,
                    shared_gate_up,
                    shared_down,
                    shared_gate_w,
                    expert_map,
                    valid_tokens,
                    out,
                    capturing,
                )
                self.kernels_per_call = None
        return out

    def _side_stream(self, device):
        stream = self._streams.get(device)
        if stream is None:
            stream = self._streams[device] = torch.cuda.Stream(device)
        return stream

    def _composed(
        self,
        x,
        router_w,
        w_gate_up,
        w_down,
        shared_gate_up,
        shared_down,
        shared_gate_w,
        expert_map,
        valid_tokens,
        out,
        capturing,
    ):
        """Router GEMM -> top-8 -> routed experts on the current stream, the shared expert and
        its gate logit on a side stream beside them (as SGLang's dual-stream block); the route
        reduction waits for the side stream and adds sigmoid(gate) * shared to its FP32 sum
        (one BF16 rounding per output). Without valid_tokens and expert_map nothing is masked
        and the routed align is the single-launch EP1 one."""
        import torch.nn.functional as F

        from ctikernels.activation.inference import silu_and_mul

        from ._route import route

        current = torch.cuda.current_stream(x.device)
        side = self._side_stream(x.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            shared = F.linear(silu_and_mul(F.linear(x, shared_gate_up)), shared_down)
            gate = F.linear(x, shared_gate_w)
        # The block tree's routing arithmetic at every row count (MB-R1: no M-dependence).
        weights, ids = route(x, router_w, 8, renormalize=True, valid_tokens=valid_tokens)
        if valid_tokens is not None and expert_map is None:
            # Padding rows route to -1, which only the expert_map align skips.
            expert_map = self._identity(x.device, router_w.shape[0], capturing)
        self.routing_ids = ids
        self.routing_weights = weights
        if not capturing:
            shared.record_stream(current)
            gate.record_stream(current)
        # The reduction waits for the side stream and adds sigmoid(gate) * shared in FP32.
        self._experts(
            x,
            w_gate_up,
            w_down,
            weights,
            ids,
            expert_map=expert_map,
            out=out,
            shared=shared,
            shared_gate=gate,
            valid_tokens=valid_tokens,
            wait_stream=side,
        )


def _mxfp4_tensor(name, t, dtypes, shape):
    if t.dtype not in dtypes or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtypes[0]} of shape {shape}")
    if t.requires_grad:
        raise ValueError(f"{name} must be frozen (requires_grad=False)")
    return t.view(torch.uint8)


_FP4 = (torch.uint8, torch.float4_e2m1fn_x2)
# W4A8 + LoRA: chunks of at least this many tokens shrink over PREP's expert-sorted rows
# (Triton); smaller ones use the CuTe pair kernel. Candidate for a tuned-table row.
SORTED_SHRINK_MIN_TOKENS = 64
_E8M0 = (torch.uint8, torch.float8_e8m0fnu)


def pack_mxfp4_experts(gate, gate_scale, up, up_scale, down, down_scale):
    """Stack checkpoint MXFP4 expert tensors into the canonical ``MXFP4Experts`` layout.

    Per expert stacks: ``gate``/``up`` [E, I, H/2] (E2M1 pairs, low nibble = lower K),
    scales [E, I, H/32] UE8M0; ``down`` [E, H, I/2], ``down_scale`` [E, H, I/32]. Canonical
    ("up_first"): w13/s13 rows are up (w3), then gate (w1); w2/s2 as in the checkpoint. Packing
    only concatenates and views bytes.
    """
    if gate.ndim != 3:
        raise ValueError("gate must be [E, I, H/2]")
    e, i, half_h = gate.shape
    h = 2 * half_h
    ops = [
        _mxfp4_tensor(n, t, d, s)
        for n, t, d, s in (
            ("gate", gate, _FP4, (e, i, h // 2)),
            ("gate_scale", gate_scale, _E8M0, (e, i, h // 32)),
            ("up", up, _FP4, (e, i, h // 2)),
            ("up_scale", up_scale, _E8M0, (e, i, h // 32)),
            ("down", down, _FP4, (e, h, i // 2)),
            ("down_scale", down_scale, _E8M0, (e, h, i // 32)),
        )
    ]
    g, gs, u, us, d, ds = ops
    return torch.cat((u, g), 1), torch.cat((us, gs), 1), d, ds


class MXFP4Experts:
    """Frozen routed SwiGLU experts, W4A8: MXFP4 weights x MXFP8 activations.

    Weights, canonical "up_first" layout (see ``pack_mxfp4_experts``): ``w13`` [E_local, 2I,
    H/2] (up rows, then gate), ``s13`` [E_local, 2I, H/32], ``w2`` [E_local, H, I/2], ``s2``
    [E_local, H, I/32]. H and I
    are multiples of 128. Activations are quantized per call to E4M3 with one UE8M0 scale per
    32 (smallest power of two mapping the absmax to <= 448). Per route: FP32 gate/up,
    optional ``swiglu_limit`` clamp (gate <= L, |up| <= L), SiLU(gate) * up * route weight,
    BF16 rounding, MXFP8 requantization, FP32 down projection; routes sum in slot order and
    round once to BF16. Only local routes contribute (expert parallel partial sums).
    Deterministic; graph-capturable after an eager warm-up of each shape (inside the
    ``GraphWorkspace`` when one is active).

    Providers: ``flashinfer`` (CUTLASS fused MoE on the sm120 family, TRT-LLM gen on the sm100
    family, offline-tuned tactics: its own MXFP8 quantization of the FC1 output, route weights
    applied at FC2 finalization) is ``auto``. ``triton`` (above, cc >= 8.9) only on request.
    ``cute`` (on request; sm_120/sm_121, DSV4 Flash geometry, up to 256 local experts): our CuTe
    W4A8 base with the deterministic row combine; ``out`` must not alias the input.

    One resident copy: the op takes over the four given tensors and every provider reads them in
    place (no device copies; ``op.workspace_bytes`` is 0). ``scales_layout`` (required) states
    what ``s13``/``s2`` hold: "canonical" (as above; construction rewrites them in place into
    FlashInfer's 128x4 interleave, ``_common.interleave_scales_``) or "interleaved" (already in
    it, e.g. tensors another op holds; read as they are).
    ``op.layout`` names the held layout: "up_first" (canonical weights, interleaved scales)
    everywhere except FlashInfer on the sm100 family, whose TRT-LLM gen kernels need w13/w2 rows
    and the scales' rows permuted ("trtllm_gen"). ``canonical_weights()`` returns canonical
    copies, made on demand. ``op.provider`` names the provider in use.
    """

    def __init__(self, w13, s13, w2, s2, *, scales_layout, swiglu_limit=None, provider="auto"):
        if scales_layout not in ("canonical", "interleaved"):
            raise ValueError('scales_layout must be "canonical" or "interleaved"')
        if provider not in ("auto", "triton", "flashinfer", "cute"):
            raise ValueError("MXFP4Experts provider must be auto, triton, flashinfer or cute")
        if w13.ndim != 3:
            raise ValueError("w13 must be [E_local, 2I, H/2]")
        e, twice_i, half_h = w13.shape
        i, h = twice_i // 2, 2 * half_h
        if not e or not i or not h or twice_i % 2 or i % 128 or h % 128:
            raise ValueError("E_local positive; H and I positive multiples of 128")
        if e > 512:
            raise ValueError("at most 512 local experts")
        weights = (
            _mxfp4_tensor("w13", w13, _FP4, (e, 2 * i, h // 2)),
            _mxfp4_tensor("s13", s13, _E8M0, (e, 2 * i, h // 32)),
            _mxfp4_tensor("w2", w2, _FP4, (e, h, i // 2)),
            _mxfp4_tensor("s2", s2, _E8M0, (e, h, i // 32)),
        )
        if any(t.device != w13.device for t in (s13, w2, s2)) or not w13.is_cuda:
            raise ValueError("weights must be CUDA tensors on one device")
        if swiglu_limit is not None and not swiglu_limit > 0:
            raise ValueError("swiglu_limit must be positive or None")
        self.swiglu_limit = swiglu_limit
        self.local_experts, self.hidden, self.inter, self.device = e, h, i, w13.device
        self._identity = torch.arange(e, dtype=torch.int32, device=w13.device)
        self._device = (hardware(w13.device), device_info(w13.device)[0])
        if provider == "auto":
            provider = "flashinfer"
        if provider == "flashinfer" and self._device[1][0] not in (10, 12):
            raise NotImplementedError("FlashInfer W4A8 experts need the sm100 or sm120 family")
        if provider == "triton" and self._device[1] < (8, 9):
            raise NotImplementedError("W4A8 Triton experts need cc >= 8.9")
        self.provider, self._weights, self._flashinfer = provider, weights, None
        self.workspace_bytes, self.layout, self._w4a8_lora, self._cute = 0, "up_first", None, None
        trtllm = provider == "flashinfer" and self._device[1][0] == 10
        for s in (weights[1], weights[3]):
            if scales_layout == "canonical" and not trtllm:
                interleave_scales_(s)
            elif scales_layout == "interleaved" and trtllm:
                # TRT-LLM gen permutes canonical rows, then interleaves them itself.
                for one in s.split(1):
                    one.copy_(deinterleave_scales(one))
        if provider == "cute":
            self._check_cute_geometry(6, "CuTe W4A8 experts")
            # One object for both paths: the LoRA object's no-LoRA call is the base kernel with
            # the deterministic row combine (the atomic one is not bitwise reproducible).
            self._cute = self._w4a8_lora = self._w4a8_lora_experts()
        if provider == "flashinfer":
            from .flashinfer._w4a8 import CutlassW4A8, TrtllmW4A8, tactics
            from .triton._w4a8 import local_routes

            kind = TrtllmW4A8 if trtllm else CutlassW4A8
            # Route mapping: one Triton glue launch.
            self._flashinfer = kind(*weights, swiglu_limit, local_routes, tactics(*self._device))
            self.layout = self._flashinfer.layout

    def _w4a8_lora_experts(self):
        # The held scales are already in the interleave the CuTe kernels read.
        _, s13, _, s2 = self._weights
        words = tuple(s.view(torch.int32) for s in (s13, s2))
        return W4A8LoRAExperts(
            ExpertWeights(*self._weights), config=W4A8Config(up_first=True), scales=words
        )

    def reset_scratch(self):
        """Restore the LoRA path's persistent scratch state (FC1 tickets, zero between calls)
        after its memory was paused or offloaded. Eager only; a no-op before any LoRA call."""
        if self._w4a8_lora is not None:
            self._w4a8_lora.reset_scratch()

    def _check_cute_geometry(self, top_k, what):
        if self._device[1] not in ((12, 0), (12, 1)):
            raise NotImplementedError(
                f"{what} run on sm_120/sm_121; the sm_100 family awaits its kernel"
            )
        geometry = (self.hidden, self.inter, top_k)
        # I: 2048 for EP; 2048 / tp per rank when the experts are split by expert TP.
        inter = 128 <= self.inter <= 2048 and self.inter % 128 == 0
        if (
            (self.hidden, top_k) != (4096, 6)
            or not inter
            or self.swiglu_limit != 10.0
            or self.local_experts > 256
        ):
            raise NotImplementedError(
                f"{what} support the DSV4 Flash geometry (H 4096, I a multiple of 128 up to "
                f"2048, top-6, up to 256 local experts, SwiGLU limit 10), not {geometry} with "
                f"{self.local_experts} local experts, limit {self.swiglu_limit}"
            )

    def canonical_weights(self):
        """(w13, s13, w2, s2) in the canonical "up_first" layout, for export or weight sync:
        copies made on demand (w13 and w2 are the held tensors themselves under "up_first")."""
        w13, s13, w2, s2 = self._weights
        if self.layout == "trtllm_gen":
            return self._flashinfer.canonical(w13, s13, w2, s2)
        return w13, deinterleave_scales(s13), w2, deinterleave_scales(s2)

    def __call__(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        expert_map=None,
        out=None,
        lora=None,
        scaling=1.0,
        adapter_id=0,
    ):
        """``hidden_states`` [T, H] BF16; ``topk_ids`` [T, k] int32 global expert ids (-1 =
        padding); ``topk_weights`` [T, k] FP32; ``expert_map`` [E_global] int32 maps global ids
        to local indices or -1 (remote); without it ids are local. ``out`` may alias the input.

        ``lora`` (``moe.lora.MoELoRAWeights``, the adapter slot ``adapter_id``, ``scaling``)
        adds per-expert LoRA inside the CuTe W4A8 kernel: the gate/up delta before the SwiGLU,
        the down delta before the route sum. It runs on sm_120/sm_121 for the DSV4 Flash
        geometry (H 4096, I a multiple of 128 up to 2048, top-6, up to 256 local experts, SwiGLU
        limit 10); ``out`` must not alias the input there."""
        x = hidden_states
        if x.ndim != 2 or x.shape[1] != self.hidden or x.dtype != torch.bfloat16:
            raise ValueError(f"hidden_states must be BF16 [T, {self.hidden}]")
        t = x.shape[0]
        if topk_ids.ndim != 2 or topk_ids.shape[0] != t or topk_ids.shape[1] == 0:
            raise ValueError("topk_ids must have shape [T, top_k] with top_k > 0")
        if topk_ids.dtype != torch.int32:
            raise ValueError("topk_ids must be int32")
        if topk_weights.shape != topk_ids.shape or topk_weights.dtype != torch.float32:
            raise ValueError("topk_weights must be float32 with the same shape as topk_ids")
        if out is not None and (out.shape != x.shape or out.dtype != x.dtype):
            raise ValueError("out must be BF16 [T, H]")
        tensors = [x, topk_ids, topk_weights] + ([] if out is None else [out])
        if expert_map is not None:
            _validate_routes(topk_ids, expert_map)
            tensors.append(expert_map)
        if any(not z.is_contiguous() or z.device != self.device for z in tensors):
            raise ValueError("inputs must be contiguous and on the weights' device")
        if x.requires_grad:
            raise ValueError("MXFP4Experts is inference only")
        if lora is not None:
            return self._lora_call(
                x, topk_weights, topk_ids, expert_map, out, lora, scaling, adapter_id
            )
        if out is None:
            out = torch.empty_like(x)
        if t == 0:
            return out
        if self._cute is not None:
            return self._cute_call(x, topk_weights, topk_ids, expert_map, out)
        mapping = self._identity if expert_map is None else expert_map
        if self._flashinfer is not None:
            with torch.cuda.device(x.device):
                return self._flashinfer(x, topk_weights, topk_ids, mapping, expert_map is None, out)
        from .triton._w4a8 import experts

        if self.layout != "up_first":
            raise RuntimeError(f"Triton W4A8 experts read the up_first layout, not {self.layout}")
        with torch.cuda.device(x.device):
            return experts(
                x,
                *self._weights,
                topk_weights,
                topk_ids,
                mapping,
                expert_map is None,
                self.swiglu_limit,
                out,
                self._device,
            )

    def _cute_call(self, x, topk_weights, topk_ids, expert_map, out):
        self._check_cute_geometry(topk_ids.shape[1], "CuTe W4A8 experts")
        if out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
            raise ValueError("out must not alias hidden_states with the cute provider")
        with torch.cuda.device(x.device):
            # One call for any T (every weight read once); global ids with the map (PREP maps
            # them), or local ids without one.
            self._cute(x, topk_ids, topk_weights, out=out, expert_map=expert_map)
        return out

    def _lora_call(self, x, topk_weights, topk_ids, expert_map, out, lora, scaling, adapter_id):
        if not isinstance(lora, MoELoRAWeights):
            raise ValueError("lora must be a moe.lora.MoELoRAWeights")
        self._check_cute_geometry(topk_ids.shape[1], "W4A8 MoE + LoRA")
        if out is not None and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
            raise ValueError("out must not alias hidden_states with LoRA")
        if out is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("supply out when capturing MXFP4Experts with LoRA")
            out = torch.empty_like(x)
        if x.shape[0] == 0:
            return out
        leaves = tuple(select_adapter(t, 3, adapter_id) for t in lora.routed)
        rank = leaves[0].shape[1]
        if not _cute_lora_rank(rank):
            raise NotImplementedError(
                f"W4A8 MoE + LoRA ranks: multiples of 8 in 8..128, not {rank}"
            )
        e, h, i = self.local_experts, self.hidden, self.inter
        shapes = ((e, rank, h),) * 2 + ((e, i, rank),) * 2 + ((e, rank, i), (e, h, rank))
        if any(t.shape != s or t.device != x.device for t, s in zip(leaves, shapes, strict=True)):
            raise ValueError(
                "LoRA leaves must be this rank's local [E_local, ...] factors, on x's device"
            )
        if self.layout != "up_first":
            raise RuntimeError(f"W4A8 LoRA experts read the up_first layout, not {self.layout}")
        if self._w4a8_lora is None:
            self._w4a8_lora = self._w4a8_lora_experts()
        with torch.cuda.device(x.device):
            self._lora_stages(x, topk_weights, topk_ids, expert_map, out, leaves, scaling)
        return out

    def _lora_stages(self, x, topk_weights, topk_ids, expert_map, out, leaves, scaling):
        # Composition of public stages for one call of any T (every weight read once): CuTe
        # PREP, a gate/up shrink (CuTe pair kernel, or Triton over the sorted rows), CuTe
        # FC1/FC2 with the LoRA deltas.
        cute, rank = self._w4a8_lora, leaves[0].shape[1]
        gate_a, up_a, gate_b, up_b, down_a, down_b = leaves
        by_row = x.shape[0] >= SORTED_SHRINK_MIN_TOKENS
        run = cute.begin(x, topk_ids, topk_weights, rank, out, expert_map=expert_map, by_row=by_row)
        if by_row:
            # Expert-grouped (Triton) over PREP's sorted rows, z13 in row order: each A tile
            # read once per row tile; FC1 then seeds on tensor cores.
            ws = run[0]
            moe_lora_sorted_shrink_(
                run[1], x, (gate_a, up_a), ws.row_token, None, ws.expert_offsets
            )
        else:
            cute.shrink_pairs(run, x, topk_ids, gate_a, up_a, expert_map=expert_map)
        cute.finish(run, gate_b, up_b, down_a, down_b, scaling, out)
