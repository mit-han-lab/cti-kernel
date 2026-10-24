# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-FP8 routed MoE, with independent CuTe and existing Triton providers.

Packed E4M3 w13 [E,2I,H], w2 [E,H,I]; multiplying BF16/FP32 128x128 scales;
contiguous BF16/FP16 x [M,H], FP32 routing weights and int32 ids [M,top_k]. H and I
must be positive multiples of 128. Ids must be in [-1,E); -1 contributes zero.
Values are a caller precondition: validation reads metadata only. Inference only.

Call-owned workspace (bytes), R=M*top_k, P=R+(E+1)*(BM-1): CuTe uses
2*P*I for sorted h, 2*R*H for down (masked rows are never read), 2*M*H for out,
4*P + 4*ceil(P/BM) + 4 + 4*(E+2) for alignment, and M*H + 4*M*(H/128)
for the grouped FC1 quantizer. Triton adds 4*R*I gate/up, 2*R*I activation,
M*H + 4*M*(H/128) and R*I + 4*R*(I/128) quantizers, besides alignment,
down and out. PyTorch's caching allocator owns all temporaries; no activation
or weight copies are retained.
With route_blocks, alignment becomes an op-owned constant cache of 4*R*BM+4
bytes per (device,R,BM). The fused CuTe activation omits the 2*R*I BF16/FP16 buffer.
Decode presets instead use only 2*R*I bytes for route h and 2*M*H for output,
with two kernel launches for M=1..8 and no alignment or reduction workspace.

Call prepare(batch_sizes, dtype=x.dtype) outside CUDA graph capture. It compiles and warms the
whole selected chain, including route constants and reduction. No call reads device scalars
on the host. Captured graphs retain their allocations and weight pointers.
With expert_map, E in alignment is the global count, weights use local ids,
and remote routes write zero without GEMM work.
"""

from __future__ import annotations

import importlib.metadata
import os
from functools import lru_cache
from types import SimpleNamespace

import torch

from . import _moe_block_fp8_cute_table as policy
from . import _moe_block_fp8_triton_table as triton_policy
from .linear.block_fp8_gated import silu_and_mul_quantize_fp8
from .moe_impl import _check_expert_map, block_fp8_moe_provider, check_shared, moe_shared_join_

MOE_BLOCK_FP8_PROVIDERS = ("cute-fp8", "triton-fp8")


@lru_cache(maxsize=1)
def _dsl_available():
    try:
        if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
            return False
        import cutlass.cute  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _select_provider(requested, device_class, num_tokens, available, enabled):
    capable = device_class[:2] in ((12, 0), (12, 1))
    if requested == "cute-fp8":
        if not capable or not available or not enabled:
            raise RuntimeError(
                "cute-fp8 needs sm120/121, nvidia-cutlass-dsl==4.8.0 and CTI_MOE_FP8_CUTE != 0"
            )
        return requested
    if requested == "triton-fp8":
        return requested
    chosen, _ = policy.selection(device_class, num_tokens)
    return chosen if capable and available and enabled else "triton-fp8"


def _tensor(name, t, dtype, shape=None):
    if not isinstance(t, torch.Tensor) or t.dtype not in (
        dtype if isinstance(dtype, tuple) else (dtype,)
    ):
        raise TypeError(f"BlockFP8MoE: {name} must be {dtype}")
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"BlockFP8MoE: {name} must have shape {tuple(shape)}")
    if not t.is_contiguous():
        raise ValueError(f"BlockFP8MoE: {name} must be contiguous")
    if t.requires_grad:
        raise ValueError(f"BlockFP8MoE: {name} must not require grad (inference only)")


def _check_weights(w13, s13, w2, s2, top_k):
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("BlockFP8MoE: top_k must be a positive integer")
    for name, t in (("w13", w13), ("w2", w2)):
        _tensor(name, t, torch.float8_e4m3fn)
        if t.ndim != 3:
            raise ValueError(f"BlockFP8MoE: {name} must be rank 3")
    e, twice_i, h = w13.shape
    i = twice_i // 2
    if e < 1 or h < 128 or i < 128 or h % 128 or i % 128 or twice_i != 2 * i:
        raise ValueError("BlockFP8MoE: E positive; H and I positive multiples of 128")
    _tensor("w2", w2, torch.float8_e4m3fn, (e, h, i))
    _tensor("s13", s13, (torch.bfloat16, torch.float32), (e, twice_i // 128, h // 128))
    _tensor("s2", s2, (torch.bfloat16, torch.float32), (e, h // 128, i // 128))
    if any(t.device != w13.device for t in (s13, w2, s2)):
        raise ValueError("BlockFP8MoE: all weights and scales must share a device")
    if any(t.data_ptr() % 16 for t in (w13, w2)):
        raise ValueError("BlockFP8MoE: weights must be 16-byte aligned")
    return e, h, i


def _check_inputs(x, weights, ids, *, device, hidden, top_k):
    if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
        raise TypeError(
            "BlockFP8MoE: FP32 hidden dtype gap: GEMM epilogue and activation need an "
            "FP32 compute type; x must be torch.bfloat16 or torch.float16"
        )
    _tensor("x", x, (torch.bfloat16, torch.float16))
    if x.ndim != 2 or x.shape[1] != hidden:
        raise ValueError(f"BlockFP8MoE: x must be [M, {hidden}]")
    _tensor("topk_weights", weights, torch.float32, (x.shape[0], top_k))
    _tensor("topk_ids", ids, torch.int32, (x.shape[0], top_k))
    if any(t.device != device for t in (x, weights, ids)):
        raise ValueError("BlockFP8MoE: inputs must share the weights' device")


def _require_bf16_api(name, x):
    if isinstance(x, torch.Tensor) and x.dtype != torch.bfloat16:
        raise TypeError(f"{name}: BF16 only; FP16 inference is supported by op(x, ...)")


def _apply_gate_up_hook(hook, gate_up):
    pointer, shape, device = gate_up.data_ptr(), gate_up.shape, gate_up.device
    result = hook(gate_up)
    _tensor("gate_up", gate_up, torch.bfloat16, shape)
    if gate_up.device != device or gate_up.data_ptr() != pointer:
        raise ValueError("gate_up_hook must not reallocate gate_up")
    if result is not None and result is not gate_up:
        raise ValueError("gate_up_hook must return None or the same gate_up tensor")


def _check_down_delta(delta, x, routes):
    if delta is not None:
        _tensor("down delta", delta, torch.bfloat16, (routes, x.shape[1]))
        if delta.device != x.device:
            raise ValueError("down delta must share the input device")
        if delta.data_ptr() % 16:
            raise ValueError("down delta must be 16-byte aligned")


def pack_block_fp8_experts(gate, gate_scale, up, up_scale, down, down_scale):
    """Pack checkpoint expert stacks, gate first, preserving multiplying scales.

    No dequantization, requantization, or scale inversion is performed.
    """
    _tensor("gate", gate, torch.float8_e4m3fn)
    if gate.ndim != 3:
        raise ValueError("gate must be [E, I, H]")
    e, i, h = gate.shape
    if e < 1 or min(i, h) < 128 or i % 128 or h % 128:
        raise ValueError("E positive; H and I positive multiples of 128")
    _tensor("up", up, gate.dtype, gate.shape)
    _tensor("down", down, gate.dtype, (e, h, i))
    for name, scale, shape in (
        ("gate_scale", gate_scale, (e, i // 128, h // 128)),
        ("up_scale", up_scale, (e, i // 128, h // 128)),
        ("down_scale", down_scale, (e, h // 128, i // 128)),
    ):
        _tensor(name, scale, (torch.bfloat16, torch.float32), shape)
    if gate_scale.dtype != up_scale.dtype:
        raise TypeError("gate_scale and up_scale must have the same dtype")
    if any(t.device != gate.device for t in (gate_scale, up, up_scale, down, down_scale)):
        raise ValueError("all checkpoint tensors must share a device")
    return torch.cat((gate, up), 1), torch.cat((gate_scale, up_scale), 1), down, down_scale


def _dx():
    # The frozen-expert backward loads with its optional CuTe/Triton providers on first use.
    from . import _moe_block_fp8_dx

    return _moe_block_fp8_dx


class BlockFP8MoE:
    def __init__(
        self,
        w13,
        s13,
        w2,
        s2,
        *,
        top_k=8,
        provider=None,
        expert_map=None,
        num_global_experts=None,
    ):
        if provider is not None and provider not in MOE_BLOCK_FP8_PROVIDERS:
            raise ValueError(f"provider must be one of {MOE_BLOCK_FP8_PROVIDERS}")
        self.e, self.h, self.i = _check_weights(w13, s13, w2, s2, top_k)
        self.device = w13.device
        self.num_global_experts = _check_expert_map(
            expert_map, self.device, self.e, num_global_experts
        )
        if expert_map is not None and num_global_experts not in (None, expert_map.numel()):
            raise ValueError("num_global_experts must equal expert_map.numel()")
        if expert_map is not None and provider == "cute-fp8":
            raise ValueError("cute-fp8 does not support expert_map yet")
        self.expert_map = expert_map
        self._cute_scales = None
        if self.device.type != "cuda":
            raise ValueError("BlockFP8MoE requires CUDA weights")
        cc = torch.cuda.get_device_capability(self.device)
        block_fp8_moe_provider(self.device)  # E4M3 tensor cores: compute capability 8.9+
        self.device_class = (
            *cc,
            torch.cuda.get_device_properties(self.device).multi_processor_count,
        )
        self.w13, self.s13, self.w2, self.s2 = w13, s13, w2, s2
        self.top_k, self.provider = top_k, provider
        self._requested_provider = provider
        self._available = cc in ((12, 0), (12, 1)) and _dsl_available()
        self._enabled = os.environ.get("CTI_MOE_FP8_CUTE") != "0"
        self._preset = None  # Development override; use set_config before prepare/capture.
        self._ready = set()
        self._route_blocks = {}
        self.provider = self.provider_for(1)

    def provider_for(self, num_tokens):
        policy.selection(self.device_class, num_tokens)  # Validate even for explicit providers.
        requested = getattr(self, "_requested_provider", self.provider)
        if getattr(self, "expert_map", None) is not None and requested is None:
            return "triton-fp8"
        return _select_provider(
            requested, self.device_class, num_tokens, self._available, self._enabled
        )

    def config_for(self, num_tokens):
        """Return Triton (FC1, FC2), including route_blocks/act, or a CuTe Config."""
        if self.provider_for(num_tokens) == "triton-fp8":
            return self._triton_config_for(num_tokens)
        policy.valid_presets(num_tokens)
        _, name = policy.selection(self.device_class, num_tokens)
        config = policy.PRESETS[self._preset or name or policy.heuristic(num_tokens)]
        if config.path == "decode" and (num_tokens > 8 or self.top_k > 8):
            raise ValueError("decode supports at most 8 tokens and top_k <= 8")
        return config

    def _triton_config_for(self, num_tokens):
        fc1, fc2 = triton_policy.selection(
            self.device_class, num_tokens, self.e, self.h, self.i, self.top_k
        )
        fc2["act"] = "cute-silu-quant" if self._available and self._enabled else "triton-silu-quant"
        return fc1, fc2

    def _provider_for_dtype(self, num_tokens, dtype):
        if dtype == torch.float16:
            if getattr(self, "_requested_provider", self.provider) == "cute-fp8":
                raise TypeError("BlockFP8MoE: cute-fp8 is BF16 only; use triton-fp8 for FP16")
            policy.selection(self.device_class, num_tokens)
            return "triton-fp8"
        return self.provider_for(num_tokens)

    def _config_for_provider(self, num_tokens, provider):
        # Public config_for/provider_for describe the BF16 policy. A future auto
        # CuTe row must still use the same dtype-independent Triton configs for FP16.
        if provider == "triton-fp8" and self.provider_for(num_tokens) != provider:
            return self._triton_config_for(num_tokens)
        return self.config_for(num_tokens)

    def set_config(self, preset):
        """Explicit CuTe development override for qualification and tuning."""
        if self.provider != "cute-fp8":
            raise ValueError("set_config requires provider='cute-fp8'")
        if preset not in policy.PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        self._preset = preset
        self._ready.clear()
        return self

    def __call__(self, x, topk_weights, topk_ids, *, shared=None, shared_gate=None):
        """Routed experts' output [M, H]; with ``shared`` [M, H] and its gate logits
        ``shared_gate`` [M, 1] (x's dtype), the gated shared expert joins it in vLLM's op order,
        ``r(r(r(sigmoid(gate)) * shared) + routed)``, as ``moe_sum_reduce_triton`` does."""
        _check_weights(self.w13, self.s13, self.w2, self.s2, self.top_k)
        _check_inputs(
            x, topk_weights, topk_ids, device=self.device, hidden=self.h, top_k=self.top_k
        )
        check_shared(x, shared, shared_gate)
        m = x.shape[0]
        provider = self._provider_for_dtype(m, x.dtype)
        if not m:
            return torch.empty_like(x)
        config = self._config_for_provider(m, provider)
        with torch.cuda.device(self.device):
            config_key = (
                tuple(tuple(sorted(c.items())) for c in config)
                if provider == "triton-fp8"
                else config
            )
            key = (m, x.dtype, provider, config_key)
            if torch.cuda.is_current_stream_capturing() and key not in self._ready:
                raise RuntimeError(
                    "BlockFP8MoE: call prepare([M], dtype=x.dtype) outside CUDA graph capture"
                )
            if provider == "triton-fp8":
                out = self._triton(
                    x, topk_weights, topk_ids, config, shared=shared, shared_gate=shared_gate
                )
            else:
                if config.path == "decode":
                    from ._moe_block_fp8_cute_decode import forward
                else:
                    from ._moe_block_fp8_cute import forward

                if self.expert_map is not None:
                    raise ValueError("cute-fp8 does not support expert_map yet")
                if self._cute_scales is None:
                    self._cute_scales = (self.s13.float(), self.s2.float())
                cute_op = SimpleNamespace(**vars(self))
                cute_op.s13, cute_op.s2 = self._cute_scales
                out = forward(cute_op, x, topk_weights, topk_ids, config)
                if shared is not None:
                    moe_shared_join_(out, shared, shared_gate)
            self._ready.add(key)
            return out

    def _route_layout(self, ids, bm):
        """One block per route; only column zero names a live output row.

        EM=R*BM gives exactly R M-blocks, including the final grouped grid
        remainder. Unmasked sorted-id loads end at EM-1; expert loads end at
        R-1. Padded=EM admits every block. The other BM-1 lanes contain R, so
        token<R masks A/scales, FC2 routing weights and C. Live A rows are
        route//top_k in FC1 and route in FC2. Each output route is written once
        per N tile, including -1/remote routes via the REMOTE zero path.
        """
        routes = ids.numel()
        key = (ids.device, routes, bm)
        if key not in self._route_blocks:
            if ids.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BlockFP8MoE: call prepare([M]) outside CUDA graph capture")
            sorted_ids = torch.full((routes, bm), routes, dtype=torch.int32, device=ids.device)
            sorted_ids[:, 0] = torch.arange(routes, dtype=torch.int32, device=ids.device)
            padded = torch.tensor([routes * bm], dtype=torch.int32, device=ids.device)
            self._route_blocks[key] = sorted_ids.view(-1), padded
        sorted_ids, padded = self._route_blocks[key]
        return sorted_ids, ids.view(-1), padded

    def forward_staged(
        self,
        x,
        topk_weights,
        topk_ids,
        *,
        gate_up_hook=None,
        down_hook=None,
        return_act=False,
        shared=None,
        shared_gate=None,
    ):
        """Expose route-order BF16 FC1 and activation caches for inference LoRA.

        gate_up_hook(cache) edits [M*top_k,2I] in place. down_hook(act) returns
        an unweighted BF16 [M*top_k,H] delta or None. return_act returns (out,act).
        Hooks must enqueue capture-safe work on the current stream.
        No-hook calls preserve the ordinary chain. ``shared`` and
        ``shared_gate`` join as in ``__call__``.
        """
        for name, hook in (("gate_up_hook", gate_up_hook), ("down_hook", down_hook)):
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} must be callable or None")
        if not isinstance(return_act, bool):
            raise TypeError("return_act must be bool")
        if gate_up_hook is None and down_hook is None and not return_act:
            return self(x, topk_weights, topk_ids, shared=shared, shared_gate=shared_gate)
        _require_bf16_api("forward_staged", x)
        _check_weights(self.w13, self.s13, self.w2, self.s2, self.top_k)
        _check_inputs(
            x, topk_weights, topk_ids, device=self.device, hidden=self.h, top_k=self.top_k
        )
        check_shared(x, shared, shared_gate)
        m = x.shape[0]
        provider = self.provider_for(m)
        if provider != "triton-fp8":
            raise NotImplementedError("forward_staged requires provider='triton-fp8'")
        if down_hook is not None and not self._available:
            raise RuntimeError("down_hook requires sm120/121 and nvidia-cutlass-dsl==4.8.0")
        need_act = return_act or down_hook is not None
        if not m:
            out = torch.empty_like(x)
            act = torch.empty((0, self.i), dtype=x.dtype, device=x.device)
            return (out, act) if return_act else out
        config = self.config_for(m)
        key = (
            m,
            x.dtype,
            provider,
            tuple(tuple(sorted(c.items())) for c in config),
            "staged",
            need_act,
            down_hook is not None,
        )
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing() and key not in self._ready:
                raise RuntimeError("BlockFP8MoE: call prepare([M], staged=True) outside capture")
            out = self._triton(
                x,
                topk_weights,
                topk_ids,
                config,
                gate_up_hook=gate_up_hook,
                down_hook=down_hook,
                return_act=return_act,
                shared=shared,
                shared_gate=shared_gate,
            )
            self._ready.add(key)
            return out

    def _triton(
        self,
        x,
        weights,
        ids,
        config,
        *,
        gate_up_hook=None,
        down_hook=None,
        return_act=False,
        shared=None,
        shared_gate=None,
    ):
        from .activation.inference import silu_and_mul
        from .linear.block_fp8 import quantize_per_token_group_fp8
        from .moe_impl import fused_moe_kernel_triton, moe_align_block_size, moe_sum_reduce_triton

        m = x.shape[0]
        fc1_config, fc2_config = (c.copy() for c in config)
        bm = fc1_config["BLOCK_SIZE_M"]
        if bm != fc2_config["BLOCK_SIZE_M"]:
            raise ValueError("FC1 and FC2 must share BLOCK_SIZE_M")
        route_blocks = fc1_config.pop("route_blocks", ids.numel() <= 64 and bm == 16)
        if fc2_config.pop("route_blocks", route_blocks) != route_blocks:
            raise ValueError("FC1 and FC2 must share route_blocks")
        # One fused SiLU*up + 1x128 quantize: CuTe on sm12x (BF16; FP16 keeps the split public
        # chain there, bit for bit), else our Triton kernel; "split" is the two public launches.
        act_kind = fc2_config.pop("act", "split")
        if act_kind == "cute-silu-quant" and x.dtype != torch.bfloat16:
            act_kind = "split"
        fused_act = act_kind != "split"
        need_act = return_act or down_hook is not None
        # A 256-column FC1 tile holds one gate group and its up group: the GEMM epilogue then
        # writes the quantized activation itself (the same bits), with no gate_up round trip.
        fc1_quant = (
            fused_act
            and not need_act
            and gate_up_hook is None
            and fc1_config["BLOCK_SIZE_N"] == 256
        )
        if route_blocks:
            sorted_ids, experts, padded = self._route_layout(ids, bm)
        else:
            sorted_ids, experts, padded = moe_align_block_size(ids, bm, self.num_global_experts)
        routes = m * self.top_k
        if fc1_quant:  # FC1 writes FC2's quantized input
            fc1_out = torch.empty(
                (m, self.top_k, self.i), dtype=torch.float8_e4m3fn, device=x.device
            )
            fc1_scale = torch.empty((routes, self.i // 128), dtype=torch.float32, device=x.device)
        else:
            fc1_out = gate_up = torch.empty(
                (m, self.top_k, 2 * self.i), dtype=x.dtype, device=x.device
            )
            fc1_scale = None
        act = (
            None
            if fused_act and not need_act
            else torch.empty((m * self.top_k, self.i), dtype=x.dtype, device=x.device)
        )
        down = torch.empty((m, self.top_k, self.h), dtype=x.dtype, device=x.device)
        for second, (source, target, weight, scale) in enumerate(
            ((x, fc1_out, self.w13, self.s13), (act, down, self.w2, self.s2))
        ):
            if second and gate_up_hook is not None:
                _apply_gate_up_hook(gate_up_hook, gate_up.view(-1, 2 * self.i))
            if second and fc1_quant:
                q, qs = fc1_out.view(routes, self.i), fc1_scale
            elif second and fused_act:
                q = torch.empty(
                    (m * self.top_k, self.i), dtype=torch.float8_e4m3fn, device=x.device
                )
                qs = torch.empty(
                    (m * self.top_k, self.i // 128), dtype=torch.float32, device=x.device
                )
                if act_kind == "cute-silu-quant":
                    from ._moe_block_fp8_cute_act import silu_mul_quant_fp8

                    silu_mul_quant_fp8(gate_up.view(-1, 2 * self.i), q, qs, act_out=act)
                else:
                    silu_and_mul_quantize_fp8(
                        gate_up.view(-1, 2 * self.i), outputs=(q, qs), act=act
                    )
            elif second:
                silu_and_mul(gate_up.view(-1, 2 * self.i), act)
                q, qs = quantize_per_token_group_fp8(source)
            else:
                q, qs = quantize_per_token_group_fp8(source)
            fused_moe_kernel_triton(
                q,
                weight,
                target,
                weights,
                ids,
                sorted_ids,
                experts,
                padded,
                False,  # FC2 rows stay unweighted: the reduction weights them (vLLM's finalize)
                1 if second else self.top_k,
                fc2_config if second else fc1_config,
                x.dtype,
                a_scale=qs,
                b_scale=scale,
                block_fp8_provider="triton-fp8",
                expert_map=self.expert_map,
                num_global_experts=self.num_global_experts,
                swiglu_quant_scale=None if second else fc1_scale,
            )
        out = torch.empty_like(x)
        if down_hook is not None:
            from ._moe_block_fp8_reduce import masked_reduce

            delta = down_hook(act)
            _check_down_delta(delta, x, ids.numel())
            masked_reduce(
                down.view(-1, self.h), ids, out, weights, expert_map=self.expert_map, delta=delta
            )
            if shared is not None:
                moe_shared_join_(out, shared, shared_gate)
        else:
            moe_sum_reduce_triton(
                down,
                out,
                topk_ids=ids,
                expert_map=self.expert_map,
                topk_weights=weights,
                shared=shared,
                shared_gate=shared_gate,
            )
        return (out, act) if return_act else out

    def dx(self, x, topk_weights, topk_ids, dy, *, saved=None, provider=None):
        """Frozen-expert backward: BF16 dX [M,H] and FP32 router dW [M,top_k].

        STE through both activation quantizers; ``provider``:
        "cute" (sm120/sm121), "triton" (every compute capability 8.9+) or None (CuTe on
        sm120/sm121, Triton elsewhere).
        """
        _require_bf16_api("dx()", x)
        return _dx().block_fp8_moe_dx(
            self, x, topk_weights, topk_ids, dy, saved=saved, provider=provider
        )

    def training_bytes(self, num_tokens):
        """``(saved, backward)`` bytes of these frozen experts in training, as the CUDA caching
        allocator holds them: this forward with ``dx`` as its backward over ``num_tokens`` rows.
        ``saved`` is what the forward keeps, the BF16 input included, with FP32 routing weights
        and int32 ids; ``backward`` the most ``dx`` holds at once, its outputs included (dX
        BF16, router dW FP32), past the output gradient. Routing does not enter: alignment
        buffers are sized for every expert's last block partial. The forward's transient peak
        is at most ``saved + backward``."""
        return _dx().training_bytes(self, num_tokens)

    def prepare_dx(self, batch_sizes, *, provider=None):
        """Warm the dX chain for each batch size outside CUDA graph capture."""
        _dx().prepare_dx(self, batch_sizes, provider=provider)

    def prepare(self, num_tokens_list, *, dtype=torch.bfloat16, staged=False):
        """Warm shapes for dtype (or a list/tuple of BF16/FP16 dtypes), with and without the
        gated shared expert.

        staged=True also warms BF16 activation export and both reductions.
        Application hooks must separately warm any kernels/workspace they own.
        """
        if not isinstance(staged, bool):
            raise TypeError("staged must be bool")
        dtypes = dtype if isinstance(dtype, (list, tuple)) else (dtype,)
        for dt in dtypes:
            if dt not in (torch.bfloat16, torch.float16):
                raise TypeError(
                    "prepare dtype must be torch.bfloat16 or torch.float16; FP32 hidden dtype gap"
                )
            if staged and dt != torch.bfloat16:
                raise TypeError(
                    "prepare(staged=True): BF16 only; FP16 inference is supported by op(x, ...)"
                )
            self._provider_for_dtype(0, dt)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BlockFP8MoE.prepare must run outside capture")
            for m in num_tokens_list:
                for dt in dtypes:
                    self._prepare_shape(m, dt, staged)

    def _prepare_shape(self, m, dtype, staged):
        provider = self._provider_for_dtype(m, dtype)
        self._config_for_provider(m, provider)
        x = torch.zeros((m, self.h), dtype=dtype, device=self.device)
        weights = torch.ones((m, self.top_k), dtype=torch.float32, device=self.device)
        ids = torch.zeros((m, self.top_k), dtype=torch.int32, device=self.device)
        gate = torch.zeros((m, 1), dtype=dtype, device=self.device)
        self(x, weights, ids)
        self(x, weights, ids, shared=x, shared_gate=gate)
        if staged:
            self.forward_staged(x, weights, ids, gate_up_hook=lambda gu: None)
            self.forward_staged(x, weights, ids, return_act=True)
            self.forward_staged(x, weights, ids, down_hook=lambda act: None)
            delta = torch.zeros((m * self.top_k, self.h), dtype=x.dtype, device=self.device)
            self.forward_staged(x, weights, ids, down_hook=lambda act, delta=delta: delta)
            self.forward_staged(
                x, weights, ids, down_hook=lambda act: None, shared=x, shared_gate=gate
            )
