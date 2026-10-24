# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FlashInfer fused MoE, MXFP8 activations x MXFP4 weights: CUTLASS on the sm120 family,
TRT-LLM gen on the sm100 family.

Numerics differ from the Triton provider: FlashInfer quantizes the FC1 output before applying
the route weight, which it applies at FC2 finalization. No copies are held: CUTLASS reads the
canonical "up_first" w13/w2 and the caller's scales already in the 128x4 interleave; TRT-LLM gen
permutes the caller's w13 and w2 rows and interleaves its permuted scales in place, one expert
at a time (load peak: one expert slice). Tactics come from a cache file tuned offline per device class
(``tactics/<class>.json``), loaded once into FlashInfer's autotuner at construction; a shape
missing there runs FlashInfer's default tactic. ``tune`` is set only by the offline tuning
run. A token with no local route gets zeros. Workspaces are allocated per call by the torch
allocator (the graph pool in capture).
"""

from contextlib import nullcontext
from importlib.resources import files

import torch

from .._common import deinterleave_scales, interleave_scales_


def _reorder_rows_(t, rows):
    """t[e] = t[e][rows] for every expert, in place."""
    for e in range(t.shape[0]):
        t[e].copy_(t[e].index_select(0, rows))
    return t


def tactics(device_class, capability):
    """The shipped tactic cache of a device class (else of its ``sm<cc>``), or None."""
    name = device_class or "sm{}{}".format(*capability)
    path = files(__package__) / "tactics" / f"{name}.json"
    return str(path) if path.is_file() else None


class CutlassW4A8:
    """``s13``/``s2`` in the 128x4 interleave (``interleave_scales_``), read in place."""

    def __init__(self, w13, s13, w2, s2, swiglu_limit, routes, tactics=None):
        from flashinfer.autotuner import AutoTuner

        e = w13.shape[0]
        # FlashInfer's SwiGLU reads the up rows first: the canonical order.
        self.layout = "up_first"
        self.w13 = w13.view(torch.int64)
        self.w2 = w2.view(torch.int64)
        self.s13 = s13.view(torch.int32)
        self.s2 = s2.view(torch.int32)
        ones = torch.ones(e, dtype=torch.float32, device=w13.device)
        limit = None
        if swiglu_limit is not None:
            limit = torch.full((e,), float(swiglu_limit), dtype=torch.float32, device=w13.device)
        self.experts, self.tactics, self.tune, self.routes = e, tactics, False, routes
        if tactics is not None:
            AutoTuner.get().load_configs(tactics)
        self.kwargs = {
            "fc1_expert_weights": self.w13,
            "fc2_expert_weights": self.w2,
            "output_dtype": torch.bfloat16,
            "quant_scales": [self.s13, ones, self.s2, ones],
            "swiglu_limit": limit,
            "use_mxfp8_act_scaling": True,
            # The fused finalize reduces routes with atomics (not bitwise reproducible).
            "use_fused_finalize": False,
        }

    def __call__(self, x, topk_weights, topk_ids, mapping, identity, out):
        from flashinfer import mxfp8_quantize
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe import cutlass_fused_moe

        ids, weights = self.routes(
            topk_weights, topk_ids, mapping, identity, self.experts, torch.float32
        )
        q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
        bucket = 1 << max(0, x.shape[0] - 1).bit_length()
        with autotune(True, cache=self.tactics) if self.tune else nullcontext():
            cutlass_fused_moe(
                input=q,
                input_sf=sf,
                token_selected_experts=ids,
                token_final_scales=weights,
                output=out,
                tune_max_num_tokens=bucket,
                **self.kwargs,
            )
        return out


class TrtllmW4A8:
    """TRT-LLM gen routed MoE: weights reordered to its gated-GEMM row order (up rows
    interleaved with gate rows) and shuffled for its 128-row epilogue tile; routes passed as
    (ids, BF16 weights). Scales: rows permuted alike, then the 128x4 interleave, in place."""

    def __init__(self, w13, s13, w2, s2, swiglu_limit, routes, tactics=None):
        from flashinfer.autotuner import AutoTuner
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        e, twice_i, half_h = w13.shape
        self.experts, self.inter = e, twice_i // 2
        self.tactics, self.tune, self.layout = tactics, False, "trtllm_gen"
        self.routes = routes
        cache, tile = {}, 128
        # FlashInfer's permutations index the [up; gate] order, the canonical one.
        rows = _maybe_get_cached_w3_w1_permute_indices(cache, w13[0], tile).to(w13.device)
        srows = _maybe_get_cached_w3_w1_permute_indices(cache, s13[0], tile, num_elts_per_sf=16)
        rows2 = get_w2_permute_indices_with_cache(cache, w2[0], tile).to(w2.device)
        srows2 = get_w2_permute_indices_with_cache(cache, s2[0], tile, num_elts_per_sf=16)
        self.w13 = _reorder_rows_(w13, rows)
        self.w2 = _reorder_rows_(w2, rows2)
        self.s13 = interleave_scales_(s13, srows.to(s13.device)).view(torch.float8_e4m3fn)
        self.s2 = interleave_scales_(s2, srows2.to(s2.device)).view(torch.float8_e4m3fn)
        self._inverse = tuple(r.argsort().to(w13.device) for r in (rows, srows, rows2, srows2))
        self.limit = None
        if swiglu_limit is not None:
            self.limit = torch.full((e,), float(swiglu_limit), device=w13.device)
        if tactics is not None:
            AutoTuner.get().load_configs(tactics)

    def __call__(self, x, topk_weights, topk_ids, mapping, identity, out):
        from flashinfer import mxfp8_quantize
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

        # Remote and padding routes go to an id outside [0, E_local) at weight 0.
        ids, weights = self.routes(
            topk_weights, topk_ids, mapping, identity, self.experts, torch.bfloat16
        )
        q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=32)
        bucket = 1 << max(0, x.shape[0] - 1).bit_length()
        with autotune(True, cache=self.tactics) if self.tune else nullcontext():
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=(ids, weights),
                routing_bias=None,
                hidden_states=q,
                hidden_states_scale=sf.view(torch.float8_e4m3fn).reshape(x.shape[0], -1),
                gemm1_weights=self.w13,
                gemm1_weights_scale=self.s13,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=self.limit,
                gemm2_weights=self.w2,
                gemm2_weights_scale=self.s2,
                gemm2_bias=None,
                output1_scale_scalar=None,
                output1_scale_gate_scalar=None,
                output2_scale_scalar=None,
                num_experts=2 * self.experts,
                top_k=topk_ids.shape[1],
                n_group=None,
                topk_group=None,
                intermediate_size=self.inter,
                local_expert_offset=0,
                local_num_experts=self.experts,
                routed_scaling_factor=None,
                routing_method_type=1,  # Renormalize; precomputed routes are used as given
                do_finalize=True,
                output=out,
                tune_max_num_tokens=bucket,
            )
        return out

    def canonical(self, w13, s13, w2, s2):
        """Canonical "up_first" copies of the held (permuted, interleaved) tensors."""
        s13, s2 = deinterleave_scales(s13), deinterleave_scales(s2)
        return tuple(
            torch.stack([t[e].index_select(0, inv) for e in range(t.shape[0])])
            for t, inv in zip((w13, s13, w2, s2), self._inverse, strict=True)
        )
