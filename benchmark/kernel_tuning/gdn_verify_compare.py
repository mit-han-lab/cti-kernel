# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired, one-layer speculative verify + accept microbenchmark (CUDA graphs).

Shell-free usage, in a Python session with this repository on sys.path::

    from benchmark.kernel_tuning.gdn_verify_compare import main

    main(
        [
            "--vllm-root",
            "checkouts/vllm-0.31.0",
            "--sglang-root",
            "checkouts/sglang-72730d6/python",
            "--q",
            "4",
            "8",
            "--batches",
            "1",
            "4",
            "--pairs",
            "20",
        ]
    )

Dispatch traced in the supplied source snapshots (paths relative to CLI roots):

* vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:
  QwenGatedDeltaNetAttention._forward_core, speculative branch, calls
  vllm/model_executor/layers/mamba/ops/causal_conv1d.py:causal_conv1d_update
  with num_accepted_tokens, then
  QwenGatedDeltaNetAttention.rearrange_mixed_qkv (loaded from that checkout)
  packs flattened Q/K/V via torch.cat and slices contiguous views for
  vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:
  fused_sigmoid_gating_delta_rule_update with 2-D ssm_state_indices and
  num_accepted_tokens. Gating and Q/K L2 normalization are fused there.
  _output_projection's gated RMSNorm uses
  vllm/third_party/flash_linear_attention/ops/layernorm_guard.py:layer_norm_fwd.
* sglang/srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward calls
  sglang/kernels/ops/attention/triton_gdn_fused_proj.py:
  fused_qkvzba_split_reshape_cat_contiguous to unpack projection outputs.
  sglang/srt/layers/attention/linear/gdn_backend.py:GDNAttnBackend.forward_extend
  target_verify calls sglang/kernels/ops/mamba/causal_conv1d_triton.py:
  causal_conv1d_update with intermediate_conv_window (NOT num_accept_tokens).
  linear/kernels/gdn_triton.py:TritonGDNKernel.target_verify calls
  sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:
  fused_sigmoid_gating_delta_rule_update with disable_state_update=True and
  intermediate_states_buffer. Qwen3.5 applies gated RMSNorm through
  sglang/kernels/ops/attention/fla/layernorm_gated.py:_layer_norm_fwd.
  hybrid_linear_attn_backend.py:update_mamba_state_after_mtp_verify calls
  sglang/kernels/ops/mamba/mamba_state_scatter_triton.py:
  scatter_mamba_states_after_mtp_verify; for this single-layer, contiguous BF16
  conv pool it launches fused_mamba_state_scatter_with_mask and
  fused_conv_window_scatter_multi. Both accept kernels are timed.

Configuration is explicit: vLLM VLLM_GDN_DECODE_KERNEL=triton, SGLang Triton
verify, linear drafts (DFlash or EAGLE topk=1), FP32 recurrent state, no radix
tracking, ReplaySSM, batch invariance or piecewise-prefill modes. These are NOT
claims about unspecified engine defaults: vLLM also has a CUDA fused MTP path
for Q<=8, and SGLang can select FlashInfer. Source hashes accompany every row.

--shape HK,HV supplies per-rank head counts (default 16,32); TP2 shapes
8,16 and 8,24 use the same 128-dimensional key/value heads.

The fixture is gdn_k2_round's seed 23 ordinary case, with --ours-window (default 8),
vector publication, trusted checks, fused publication and automatic value tile,
accepting Q//2. --ours-checks strict adds begin/end and status kernels;
--ours-fused off selects separate publication in trusted mode. Both settings,
the tile policy/resolved tile and PDL are recorded in every row and summary.
Positions, acceptance metadata, status sequence preparation, status copy/settle,
fixture restoration and graph capture are outside events. Each provider owns
a separate graph (ours spans a complete fold cycle; comparators stay one round).
--ours-commit-mode selects eager or delayed. Every sample restores the same canonical state (warm L2; no cache eviction). Twenty alternating
order pairs per comparator and twenty ours/ours self-pairs are the defaults.
Ratios are medians of paired comparator/ours timings; epsilon is the linearly
interpolated p95 of abs(self-pair ratio - 1). Speedup versus the faster comparator
is min(the two paired median ratios), with no significance clamping.

vLLM holds Q per-position SSM slots per row plus null slot zero. Its conv cache
has Q+2 columns. Acceptance is deferred: the next verify reads the previous
round's accepted-1 SSM slot and conv offset, then caches all Q new states. We
seed those selected locations with the SAME canonical fixture as ours. Its
graph thus includes steady-state acceptance, with no extra gather invented at
the end. SGLang holds B+1 persistent rows, B+1 intermediate rows with Q states,
and deduplicated Q+2-column conv windows; it commits this round at Q//2-1.

All providers start from the same BF16 [q,k,v,z,b,a] values and weights;
state is already [B,Hv,V,K], history [B,3,Dc]. Projection GEMMs (input and
output) are excluded for ALL providers. Buffers representing their outputs and
cache-layout initialization are prepared outside timing. Everything dispatched
between those GEMMs is inside the graph, with the following selected layouts:

* Ours consumes its projection layout directly. Trusted mode times verify +
  commit kernels only (including fused conv, gating, recurrence, norm and
  publication); precomputed positions are passed directly to verify.
* vLLM Qwen3.5 non-interleaved, all-spec batch: separate dense QKVZ/BA GEMM
  outputs; split/views, B/A contiguous copies, core-output zeroing, in-place
  conv on the strided QKV view, the original rearrange_mixed_qkv helper
  (flattening reshape copies, torch.cat and contiguous slice/views), recurrent
  update with deferred acceptance, core-output copy, z.reshape materialization
  and gated RMSNorm. The recurrent wrapper's contiguous calls are no-ops.
* SGLang Qwen3.5 unquantized BF16, no LoRA, <=1024 tokens: strided QKVZ/BA
  slices of one fused GEMM output; fused_qkvzba unpack into dense buffers,
  conv on the [B,Dc,Q] transpose (empty_like preserves its strides), split and
  reshape to strided Q/K/V accepted directly by Triton target_verify, gated
  RMSNorm and both post-verify scatter kernels. There is no Q/K/V packing
  copy on this selected route, and norm flattening is a view.

--assert-no-extra-copies defaults on: once in warmed, untimed execution it
counts materializing Tensor.contiguous calls by dispatch/leaf stage. Only
vLLM's two dispatch B/A copies are expected; all leaf stages must report zero.
Counts are saved per provider (null when disabled). This guard detects wrapper
layout regressions, not arbitrary allocations, reshape copies or kernel writes.
It is removed before CUDA graph capture. Serving compiler fusion outside the
custom op is not reproduced by this eager leaf harness.
BF16 conv and recurrent-output rounding in the engines differs from our fused
FP32 intermediates. The untimed check reports FP32 max absolute error over ALL
Q outputs, RMSE, finiteness and a configurable diagnostic allclose tolerance;
it never presents a numerical mismatch as a successful correctness check.
GPU validation is required; CPU import checks do not establish kernel parity.
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import fields
from functools import wraps
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from benchmark.kernel_tuning.gdn_decode_compare import _shape
from benchmark.kernel_tuning.gdn_verify_external import DISPATCH, SPTR, load_sglang, load_vllm

HK, HV, K = 16, 32, 128


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(comparisons, self_pairs):
    """Pure CPU reduction; preserve pairing rather than divide marginal medians."""
    ratios = {name: median(c / o for o, c in pairs) for name, pairs in comparisons.items()}
    ours = [t for pair in self_pairs for t in pair]
    ours += [o for pairs in comparisons.values() for o, _ in pairs]
    return {
        "median_us": {"ours": median(ours)}
        | {name: median(c for _, c in pairs) for name, pairs in comparisons.items()},
        "ratios_comparator_over_ours": ratios,
        "eps": percentile([abs(b / a - 1) for a, b in self_pairs], 0.95),
        "speedup_vs_faster_comparator": min(ratios.values()),
        "faster_comparator": min(ratios, key=ratios.get),
        "samples_us": comparisons | {"ours_self": self_pairs},
    }


class Restorable:
    def freeze(self, tensors):
        self.saved = [(tensor, tensor.clone()) for tensor in tensors]

    def prepare(self):
        for tensor, original in self.saved:
            tensor.copy_(original)

    def finish(self):
        pass


def fold_period(mode, q, window):
    """Rounds from n=0 back to n=0 at fixed acceptance max(1,Q//2)."""
    if mode == "eager":
        return 1
    n, rounds = 0, 0
    while True:
        rounds += 1
        n += max(1, q // 2)
        if n >= window:
            return rounds


class Ours(Restorable):
    def __init__(
        self,
        weights,
        drafts,
        initial,
        history,
        *,
        checks="trusted",
        fused=True,
        commit_mode="eager",
        window=8,
        rounds=None,
        hk=HK,
        hv=HV,
        ring=False,
        provider="cute",
    ):
        import torch

        import ctikernels.gdn.replay.cute._phase_launch as launch
        from benchmark.kernel_tuning.gdn_replay_common import Replay
        from ctikernels.gdn import GatedDeltaNetReplay

        launch.VECTOR_PUBLISH = True
        launch.PHASE_TILE = None
        launch.PHASE_WARPS = None
        launch.PHASE_MIN_BLOCKS = None
        launch.PDL = True
        launch.FUSED_PUBLISH = fused
        self.checks, self.provider = checks, provider
        b, q, _ = drafts.shape
        from ctikernels.gdn.replay.cute import select

        props = torch.cuda.get_device_properties(drafts.device)
        # (tile, warps, min CTAs per SM) chosen by the phase launch policy.
        self.effective_launch = launch.phase_launch(
            b, hv, props.multi_processor_count, q, select(props.name)
        )
        self.effective_tile = (
            "/".join(map(str, self.effective_launch)) if provider == "cute" else "triton"
        )
        self.runner = Replay(
            provider,
            hk,
            hv,
            b,
            window,
            q,
            weights.args(),
            initial,
            history,
            torch.zeros(b, dtype=torch.int64, device=drafts.device),
            # Size the state as the engine would: a ring only when commit may be delayed
            # (eager commit on an all-#25 device allocates no ring). ring=True keeps one
            # for eager too, so eager/delayed A/B pairs share the same storage.
            mode="delayed" if provider == "cute" and (ring or commit_mode != "eager") else "auto",
        )
        self.drafts = drafts
        # Keep the comparator's protocol explicit; trusted timings omit status
        # kernels and select fused or separate publication for A/B comparisons.
        self.runner.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device=drafts.device,
            window=window,
            verify_length=q,
            provider=provider,
            checks=checks,
            commit_mode=commit_mode,
        )
        self.commit_mode = self.runner.op.commit_mode_for(b)
        self.rounds = rounds or fold_period(self.commit_mode, q, window)
        # Host-precomputed positions: no position-update kernels inside capture.
        self.positions = [
            (torch.arange(q, dtype=torch.int64) + r * (max(1, q // 2)))
            .expand(b, q)
            .contiguous()
            .to(drafts.device)
            for r in range(self.rounds)
        ]
        self.outputs = [torch.empty_like(self.runner.verify_out) for _ in range(self.rounds)]
        # Compare the first round against the unchanged one-round comparators.
        self.output = self.outputs[0]
        self.accepted = torch.full((b,), max(1, q // 2), dtype=torch.int32, device=drafts.device)
        self.sequence = 1
        self.runner.op.end(status=self.runner.status)
        self.finish()
        self.freeze(
            [getattr(self.runner.state, f.name) for f in fields(self.runner.state)]
            + [self.runner.pending.raw, self.runner.pending.metadata]
        )

    def prepare(self):
        super().prepare()
        self.sequence += 1
        self.runner.status.sequence.fill_(self.sequence)

    def call(self):
        p = self.runner
        # A trusted Triton op takes no status: its fastest contract (no guard, no validation).
        status = None if self.provider == "triton" and self.checks == "trusted" else p.status
        if self.checks == "strict":
            p.op.begin(status=p.status, layer_weights=(p.weights,))
        for positions, output in zip(self.positions, self.outputs, strict=True):
            p.op.verify(
                self.drafts,
                *p.weights,
                slot_indices=p.slots,
                positions=positions,
                state=p.state,
                pending=p.pending,
                out=output,
                status=status,
            )
            p.op.commit(
                *p.weights,
                slot_indices=p.slots,
                accepted_lengths=self.accepted,
                state=p.state,
                pending=p.pending,
                status=status,
                workspace=p.phase_workspace
                if self.checks == "trusted" and status is not None
                else None,
            )
        if self.checks == "strict":
            p.op.end(status=p.status)

    def finish(self):
        if self.checks == "trusted" and self.sequence > 1:
            return
        p = self.runner
        result = p.op.read_status_copy(*p.status.copy_to_host(), expected_sequence=self.sequence)
        if not result.ok:
            raise RuntimeError(f"CuTe round failed: {result}")
        p.op.settle(status=p.status, result=result)


class External(Restorable):
    """Engine-shaped buffers; device follows the fixture, allowing CPU layout tests."""

    def __init__(self, provider, leaf, weights, drafts, initial, history, *, hk=HK, hv=HV):
        import torch

        self.hk, self.hv, self.dc = hk, hv, (2 * hk + hv) * K
        dc = self.dc
        self.provider, self.leaf, self.weights = provider, leaf, weights
        self.b, self.q, _ = drafts.shape
        b, q, device = self.b, self.q, drafts.device
        projection = drafts.view(b * q, -1)
        self.qkvz, self.ba = projection.split((dc + hv * K, 2 * hv), -1)
        if provider == "vllm":
            # Two separate projection GEMMs in Qwen3.5 forward_cuda.
            self.qkvz, self.ba = self.qkvz.contiguous(), self.ba.contiguous()
        # SGLang's unquantized BF16 small-batch GEMM returns these strided
        # slices of one fused output. Its unpack kernel accepts row strides.
        self.beta, self.alpha = self.ba.split(hv, -1)
        self.z = self.qkvz[:, dc:].view(b * q, hv, K)
        self.cu = torch.arange(b + 1, dtype=torch.int32, device=device) * q
        self.accepted = torch.full((b,), max(1, q // 2), dtype=torch.int32, device=device)
        self.steps = self.accepted - 1
        self.core = torch.empty(b * q, hv, K, dtype=drafts.dtype, device=device)
        self.geometry = SimpleNamespace(
            key_dim=hk * K, value_dim=hv * K, tp_size=1, head_k_dim=K, head_v_dim=K
        )
        if provider == "vllm":
            self.indices = torch.arange(1, b * q + 1, dtype=torch.int32, device=device).view(b, q)
            self.state = initial.new_zeros(b * q + 1, hv, K, K)
            selected = self.indices[:, max(1, q // 2) - 1].long()
            self.state[selected] = initial
            self.conv = history.new_zeros(b * q + 1, dc, q + 2)
            self.slots = self.indices[:, 0]
            self.conv[self.slots.long(), :, max(1, q // 2) - 1 : max(1, q // 2) + 2] = (
                history.transpose(1, 2)
            )
            self.freeze([self.state, self.conv, self.qkvz])
        else:
            self.slots = torch.arange(1, b + 1, dtype=torch.int32, device=device)
            self.state = initial.new_zeros(b + 1, hv, K, K)
            self.state[1:] = initial
            self.conv = history.new_zeros(b + 1, dc, 3)
            self.conv[1:] = history.transpose(1, 2)
            self.intermediate = initial.new_zeros(b + 1, q, hv, K, K)
            self.conv_storage = history.new_zeros(b + 1, dc, q + 2)
            s0, s1, s2 = self.conv_storage.stride()
            self.conv_windows = self.conv_storage.as_strided((b + 1, q, dc, 3), (s0, s2, s1, s2))
            # Default non-PP scratch rows are dense 0..B-1, independent of state slots.
            self.scratch = torch.arange(b, dtype=torch.int32, device=device)
            self.caches = SimpleNamespace(
                temporal=self.state.unsqueeze(0),
                intermediate_ssm=self.intermediate.unsqueeze(0),
                conv=[self.conv.unsqueeze(0)],
                intermediate_conv_window=[self.conv_windows.unsqueeze(0)],
            )
            self.freeze([self.state, self.conv])

    def call(self):
        hk, hv, dc = self.hk, self.hv, self.dc
        b, q, w, leaf = self.b, self.q, self.weights, self.leaf
        if self.provider == "vllm":
            mixed, z = self.qkvz.split((dc, hv * K), -1)
            z = z.reshape(b * q, hv, K)
            beta, a = self.ba.chunk(2, -1)
            # forward() zeroes the custom-op output and materializes a/b.
            self.core.zero_()
            beta, a = beta.contiguous(), a.contiguous()
            mixed = leaf.conv(
                mixed,
                self.conv,
                w.conv_weight,
                None,
                "silu",
                conv_state_indices=self.slots,
                num_accepted_tokens=self.accepted,
                query_start_loc=self.cu,
                max_query_len=q,
                validate_data=False,
            )
            query, key, value = leaf.rearrange(self.geometry, mixed)
        else:
            mixed, z, beta, a = leaf.unpack(self.qkvz, self.ba, hk, hv, K, K)
            mixed = (
                leaf.conv(
                    mixed.view(b, q, dc).transpose(1, 2),
                    self.conv,
                    w.conv_weight,
                    None,
                    "silu",
                    conv_state_indices=self.slots,
                    intermediate_conv_window=self.conv_windows,
                    intermediate_state_indices=self.scratch,
                )
                .transpose(1, 2)
                .view(b * q, dc)
            )
            query, key, value = mixed.split((hk * K, hk * K, hv * K), -1)
            query = query.view(1, b * q, hk, K)
            key = key.view(1, b * q, hk, K)
            value = value.view(1, b * q, hv, K)
        args = {
            "A_log": w.A_log,
            "dt_bias": w.dt_bias,
            "a": a,
            "b": beta,
            "q": query,
            "k": key,
            "v": value,
            "cu_seqlens": self.cu,
            "use_qk_l2norm_in_kernel": True,
        }
        if self.provider == "vllm":
            output, _ = leaf.recurrent(
                **args,
                initial_state=self.state,
                inplace_final_state=True,
                ssm_state_indices=self.indices,
                num_accepted_tokens=self.accepted,
            )
            self.core.copy_(output.squeeze(0))
            output = self.core
        else:
            output = leaf.recurrent(
                **args,
                initial_state_source=self.state,
                initial_state_indices=self.slots,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                is_kda=False,
                disable_state_update=True,
                intermediate_states_buffer=self.intermediate,
                intermediate_state_indices=self.scratch,
                cache_steps=q,
                retrieve_parent_token=None,
            )
        # Match the ordinary RMSNorm wrappers' flattening, including any z copy.
        normalized, _, _ = leaf.norm(
            output.reshape(-1, K),
            w.norm_weight,
            None,
            1e-6,
            z=z.reshape(-1, K),
            norm_before_gate=True,
            is_rms_norm=True,
        )
        self.output = normalized.view(b, q, hv * K)
        if self.provider == "sglang":
            leaf.scatter(self.caches, self.slots, self.steps, None, None)


def audit_contiguous_copies(provider):
    """Run once outside capture/events; count materializing Tensor.contiguous calls.

    No-op contiguous calls are recorded separately. Explicit engine packing
    (cat/unpack), reshape copies, output copy_ and acceptance writes are part of
    the dispatch, not extra contiguous copies. This is a wrapper/layout guard,
    not a census of CUDA allocations or arbitrary device copies.
    """
    import torch

    original = torch.Tensor.contiguous
    leaf = getattr(provider, "leaf", None)
    stage = "dispatch"
    counts = {"dispatch": {"calls": 0, "copies": 0}}

    def counted(tensor, *args, **kwargs):
        result = original(tensor, *args, **kwargs)
        counts[stage]["calls"] += 1
        counts[stage]["copies"] += int(result.data_ptr() != tensor.data_ptr())
        return result

    def instrument(name, fn):
        counts[name] = {"calls": 0, "copies": 0}

        @wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal stage
            previous, stage = stage, name
            try:
                return fn(*args, **kwargs)
            finally:
                stage = previous

        return wrapped

    try:
        if leaf is not None:
            provider.leaf = SimpleNamespace(
                **{name: instrument(name, fn) for name, fn in vars(leaf).items()}
            )
        torch.Tensor.contiguous = counted
        provider.call()
    finally:
        torch.Tensor.contiguous = original
        if leaf is not None:
            provider.leaf = leaf
    expected = 2 if getattr(provider, "provider", None) == "vllm" else 0
    extra = sum(
        max(0, count["copies"] - (expected if name == "dispatch" else 0))
        for name, count in counts.items()
    )
    result = {
        "scope": "materializing_Tensor.contiguous_calls",
        "stages": counts,
        "copies": sum(count["copies"] for count in counts.values()),
        "expected_dispatch_copies": expected,
        "extra_copies": extra,
    }
    if extra or counts["dispatch"]["copies"] != expected:
        raise AssertionError(f"Dispatch contiguous-copy audit failed: {result}")
    return result


class GraphRound:
    def __init__(self, provider, *, assert_no_extra_copies=True):
        import torch

        self.provider = provider
        self.copy_audit = None
        # Explicit side-stream warmup, including acceptance and norm variants.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for iteration in range(3):
                provider.prepare()
                if assert_no_extra_copies and iteration == 2:
                    self.copy_audit = audit_contiguous_copies(provider)
                else:
                    provider.call()
                provider.finish()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        provider.prepare()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            provider.call()
        # Capture records work; it does not execute a lifecycle to settle.
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            self.sample()

    def sample(self):
        self.provider.prepare()
        self.start.record()
        self.graph.replay()
        self.stop.record()
        self.stop.synchronize()
        us = self.start.elapsed_time(self.stop) * 1000
        self.provider.finish()
        if not math.isfinite(us) or us <= 0:
            raise RuntimeError("Invalid CUDA event duration")
        return us / getattr(self.provider, "rounds", 1)


def correctness(graphs, atol, rtol):
    import torch

    outputs = {}
    for name, graph in graphs.items():
        graph.sample()
        outputs[name] = graph.provider.output.float().clone()
    reference = outputs["ours"]
    checks = {}
    for name in ("vllm", "sglang"):
        output = outputs[name]
        difference = output - reference
        finite = bool(torch.isfinite(output).all() & torch.isfinite(reference).all())
        checks[name] = {
            "max_abs_diff_fp32": difference.abs().max().item() if finite else None,
            "rmse_fp32": difference.square().mean().sqrt().item() if finite else None,
            "finite": finite,
            "allclose": finite and torch.allclose(output, reference, atol=atol, rtol=rtol),
        }
    return {"atol": atol, "rtol": rtol, "providers": checks}


def measure(
    q,
    b,
    pairs,
    leaves,
    atol,
    rtol,
    *,
    ours_checks="trusted",
    ours_fused=True,
    ours_commit_mode="eager",
    ours_window=8,
    assert_no_extra_copies=True,
    hk=HK,
    hv=HV,
    ours_provider="cute",
):
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, x, initial, history = case_inputs(hk, hv, b, b, q + 1, "ordinary", 23, "cuda")
    drafts = x[-q:].transpose(0, 1).contiguous().view(b, q, -1)
    providers = {
        "ours": Ours(
            weights,
            drafts,
            initial,
            history,
            checks=ours_checks,
            fused=ours_fused,
            commit_mode=ours_commit_mode,
            window=ours_window,
            hk=hk,
            hv=hv,
            provider=ours_provider,
        )
    }
    providers.update(
        {
            name: External(name, leaf, weights, drafts, initial, history, hk=hk, hv=hv)
            for name, leaf in leaves.items()
        }
    )
    graphs = {
        name: GraphRound(provider, assert_no_extra_copies=assert_no_extra_copies)
        for name, provider in providers.items()
    }
    checks = correctness(graphs, atol, rtol)
    comparisons, self_pairs = {"vllm": [], "sglang": []}, []
    for i in range(pairs):
        # Alternate provider order as well as order within each pair.
        for name in ("vllm", "sglang") if i % 2 == 0 else ("sglang", "vllm"):
            order = ("ours", name) if i % 2 == 0 else (name, "ours")
            times = {key: graphs[key].sample() for key in order}
            comparisons[name].append((times["ours"], times[name]))
        self_pairs.append((graphs["ours"].sample(), graphs["ours"].sample()))
    return summarize(comparisons, self_pairs) | {
        "q": q,
        "b": b,
        "pairs": pairs,
        "accepted_per_row": max(1, q // 2),
        "config": {
            "shape": [hk, hv],
            "ours_provider": ours_provider,
            "ours_checks": ours_checks,
            "ours_fused": "on" if ours_fused else "off",
            "ours_window": ours_window,
            "ours_commit_mode": providers["ours"].commit_mode,
            "ours_rounds": providers["ours"].rounds,
            "ours_tile": "auto",
            "ours_effective_tile": providers["ours"].effective_tile,
            "ours_publication": "vector",
            "ours_pdl": True,
            "assert_no_extra_copies": assert_no_extra_copies,
        },
        "copy_audit": {name: graph.copy_audit for name, graph in graphs.items()},
        "correctness": checks,
        "correctness_ok": all(check["allclose"] for check in checks["providers"].values()),
    }


def source_hashes(roots):
    result = {}
    roots = roots | {"ours": Path(__file__).resolve().parents[2]}
    for provider, root in roots.items():
        paths = {entry.split(":", 1)[0] for entry in DISPATCH[provider]}
        if provider == "sglang":
            paths.add(SPTR)
        elif provider == "ours":
            paths.update(
                str(path.relative_to(root))
                for path in (root / "python/ctikernels/gdn/replay").rglob("*.py")
            )
            paths.update(
                {
                    "benchmark/kernel_tuning/gdn_verify_compare.py",
                    "benchmark/kernel_tuning/gdn_verify_external.py",
                    "benchmark/kernel_tuning/gdn_replay_inputs.py",
                }
            )
        result[provider] = {
            path: hashlib.sha256((Path(root).expanduser() / path).read_bytes()).hexdigest()
            for path in sorted(paths)
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--q", nargs="+", type=int, default=[4, 5, 6, 8, 9, 12, 16])
    parser.add_argument(
        "--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4]
    )
    parser.add_argument("--shape", type=_shape, default="16,32", metavar="HK,HV")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.1, help="Diagnostic BF16 output tolerance")
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--ours-commit-mode", choices=("eager", "delayed"), default="eager")
    parser.add_argument("--ours-window", type=int, default=8)
    parser.add_argument("--ours-checks", choices=("strict", "trusted"), default="trusted")
    parser.add_argument("--ours-provider", choices=("cute", "triton"), default="cute")
    parser.add_argument(
        "--ours-fused",
        choices=("on", "off"),
        default="on",
        help="Trusted CuTe publication fusion (use --ours-checks trusted)",
    )
    parser.add_argument(
        "--assert-no-extra-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit materializing contiguous calls once during untimed warmup",
    )
    args = parser.parse_args(argv)
    if args.ours_commit_mode == "delayed" and args.ours_checks != "trusted":
        parser.error("delayed commit requires --ours-checks trusted")
    if args.pairs < 2:
        parser.error("--pairs must be at least 2")
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    return args


def print_row(row):
    """Keep stdout JSONL machine-readable; put the per-row summary on stderr."""
    config = row["config"]
    print(json.dumps(row, allow_nan=False), flush=True)
    print(
        f"Q={row['q']} B={row['b']} "
        f"ours_checks={config['ours_checks']} ours_fused={config['ours_fused']} "
        f"ours_tile={config['ours_tile']} ({config['ours_effective_tile']}) "
        f"ours_pdl={config['ours_pdl']} "
        f"ours_commit_mode={config['ours_commit_mode']} ours_window={config['ours_window']} "
        f"K={config['ours_rounds']} "
        f"median_us={row['median_us']} copy_audit={row['copy_audit']}",
        file=sys.stderr,
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    hk, hv = args.shape
    roots = {"vllm": args.vllm_root, "sglang": args.sglang_root}
    hashes = source_hashes(roots)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarking; --help and module import are CPU-safe")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    leaves = {"vllm": load_vllm(args.vllm_root, sm), "sglang": load_sglang(args.sglang_root, sm)}
    metadata = {
        "device": torch.cuda.get_device_name(),
        "sm": sm,
        "dispatch": DISPATCH,
        "source_sha256": hashes,
        "cache_regime": "warm_l2_fixture_reset_before_each_sample",
        "timing": "cuda_graph_one_layer_round",
        "shape": {"hk": hk, "hv": hv, "k": K, "v": K, "conv_width": 4},
        "config": {
            "vllm_decode_kernel": "triton",
            "sglang_verify_backend": "triton",
            "draft_topk": 1,
            "state_dtype": "float32",
            "projection_dtype": "bfloat16",
            "vllm_projection_layout": "separate_qkvz_ba",
            "sglang_projection_layout": "fused_qkvzba_views",
            "pdl": sm >= 90,
        },
        "acceptance": {
            "vllm": "deferred selection fused into next verify",
            "sglang": "post_verify_scatter",
            "ours": "commit",
        },
    }
    with torch.inference_mode():
        for q in args.q:
            for b in args.batches:
                row = measure(
                    q,
                    b,
                    args.pairs,
                    leaves,
                    args.atol,
                    args.rtol,
                    hk=hk,
                    hv=hv,
                    ours_checks=args.ours_checks,
                    ours_fused=args.ours_fused == "on",
                    ours_commit_mode=args.ours_commit_mode,
                    ours_window=args.ours_window,
                    assert_no_extra_copies=args.assert_no_extra_copies,
                    ours_provider=args.ours_provider,
                )
                row["config"] = metadata["config"] | row["config"]
                print_row(metadata | row)


if __name__ == "__main__":
    main()
