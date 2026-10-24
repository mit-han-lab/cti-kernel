# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""K1.22 paired CUDA graph benchmark against pinned runtime source bundles.

Run from the repository root: PYTHONPATH=python python -m
benchmark.dflash_sampling.vllm_bench --comparator-root DIR [--quick].

vllm denotes Model Runner V2; vllm_v1 is informational for pipeline comparisons.

SGLang stochastic rows use the production CUDA JIT tree verifier, including
RNG, probability building and gather/commit. sglang_triton_proxy retains the
informational fixed-uniform Triton chain. G uses the native greedy kernel.

Each graph contains 20 executions. Twenty interleaved pairs and twenty comparator
self-pairs use bench.py's K1.22 statistics. Ratios >1 favor K1. `faster` selects
by comparator median; its stats contain the paired interval, not a
selection-adjusted confidence interval. Launches are measured per execution.
Verify starts with FP32 p: V2 is N/A, V1 uses its exponential-race bonus sampler
on p (no second filtering/softmax). V1 pipeline includes its bonus Sampler.

Default grid: both vocabularies for pipeline, V=248320 for verify (432 rows).
Quick changes shapes only, preserving R and all twenty statistical pairs.
Stdout is compact JSONL; provenance, scope and failures go to stderr. No GPU
imports/initialization at module import. GPU execution must be requested via CLI.
"""

import argparse
import contextlib
import dataclasses
import gc
import itertools
import json
import statistics
import sys
import traceback
from types import SimpleNamespace

import torch

from benchmark.dflash_round import bench as round_bench
from benchmark.dflash_sampling import bench, upstream
from ctikernels.sampling import RowArgmax
from ctikernels.sampling._dflash import RequestLocalChainSampling

R = 20
BATCHES = (1, 2, 3, 4)
QS = (4, 5, 6, 8, 9, 16)
# Selectable beyond the default sweep: Q>32 rows for the support-range gap P1.
EXTRA_QS = (32, 33, 64)
VOCABS = (151936, 248320)
MODES = ("G", "T", "KP")
# Selectable beyond the default sweep: KP plus min_p=0.05 (DN5); vLLM V2 only.
EXTRA_MODES = ("KPM",)
MIN_P = 0.05


def fixture(b, q, v, mode, case, device="cuda"):
    """Finite, unique maximum with a margin that underflows all other FP32 mass.

    Every row is a point mass, including the correction/bonus. This deliberately
    tests forced path costs; it is not a representative-logit distribution study.
    """
    logits = torch.full((b, q, v), -1024, dtype=torch.bfloat16, device=device)
    drafts = torch.arange(1, q, dtype=torch.int32, device=device).repeat(b, 1)
    winners = torch.arange(1, q + 1, dtype=torch.int64, device=device).repeat(b, 1)
    if case == "reject":
        winners[:, 0] = q + 1
    logits.scatter_(2, winners[:, :, None], 1024)
    p = torch.zeros((b, q, v), dtype=torch.float32, device=device)
    p.scatter_(2, winners[:, :, None], 1)
    temperature = {"G": 0.0, "T": 0.7, "KP": 0.6, "KPM": 0.6}[mode]
    kp = mode in ("KP", "KPM")
    return SimpleNamespace(
        b=b,
        q=q,
        v=v,
        mode=mode,
        case=case,
        logits=logits,
        p=p,
        drafts=drafts,
        temp=torch.full((b,), temperature, dtype=torch.float32, device=device),
        k=torch.full((b,), 20 if kp else v, dtype=torch.int32, device=device),
        top_p=torch.full((b,), 0.95 if kp else 1, dtype=torch.float32, device=device),
        rows=torch.arange(b, dtype=torch.int32, device=device),
        min_p=torch.full((b,), MIN_P, dtype=torch.float32, device=device)
        if mode == "KPM"
        else None,
    )


def ours_call(f, stage):
    op = RequestLocalChainSampling("cuda", f.q, f.v)
    p = torch.empty_like(f.p) if stage == "pipeline" else f.p
    workspace = torch.empty(
        max(op.workspace_size(f.b), op.workspace_size(f.b, "probabilities")),
        dtype=torch.uint8,
        device="cuda",
    )
    out = torch.empty((f.b, f.q), dtype=torch.int32, device="cuda")
    counts = torch.empty(f.b, dtype=torch.int32, device="cuda")
    common = {
        "request_indices": f.rows,
        "seeds": torch.tensor([[42, 0]], dtype=torch.uint32, device="cuda").repeat(f.b, 1),
        "cursors": torch.arange(f.b, dtype=torch.int64, device="cuda") + 2**32,
        "workspace": workspace,
    }

    def call():
        if stage == "pipeline":
            op.probabilities(
                f.logits,
                f.temp,
                f.k,
                f.top_p,
                request_indices=f.rows,
                workspace=workspace,
                out_probs=p,
                filters="both" if f.mode in ("KP", "KPM") else "none",
                min_p=f.min_p,
            )
        op.verify_point_mass(p, f.drafts, f.temp, **common, out_tokens=out, accepted_counts=counts)

    return call, lambda: (out, counts)


def ours_greedy_call(f, *, use_torch=False):
    """All-greedy engine dispatch at TP1: BF16 argmax + R5 greedy commit."""
    frontier, capacity = 4096, 8192
    state = round_bench.fixture(f.b, f.q, capacity, s=0, optional=False)
    state["frontier_table"].fill_(frontier)
    state["done_table"].zero_()
    # As in the R5 bench, C's destination tables do not feed its saved inputs.
    # Every execution overwrites the same outputs, without advancing the work.
    state["start_frontiers"].fill_(frontier)
    state["round_done"].zero_()
    state["output_ends"].fill_(capacity)
    state = round_bench.clone(state, f.logits.device)
    state["proposals"].copy_(f.drafts)
    logits = f.logits.view(f.b * f.q, f.v)
    target_ids = state["target_ids"]  # preallocated int64, just like engine argmax
    rows = state["request_indices"].long()
    round_bench.warm_up(f.q, capacity, 0, device=f.logits.device)
    if not use_torch:
        argmax = RowArgmax(f.logits.device)
        workspace = torch.zeros(
            argmax.workspace_size(f.b * f.q), dtype=torch.uint8, device=f.logits.device
        )
        chain = {"proposals": state["proposals"], "block_size": f.q}
        argmax.warm_up(logits, target_ids, workspace, **chain)

    def call():
        if use_torch:
            torch.argmax(logits, dim=-1, out=target_ids)
        else:
            # Chain mode: rows after a known first mismatch skip their reads.
            argmax(logits, target_ids, workspace, **chain)
        round_bench.commit(state)

    def outputs():
        # No stops/binding budgets: emitted length is raw accepted drafts + 1.
        # Compare actual table writes at F+1, ignoring the uncommitted suffix.
        if not torch.equal(state["committed"], state["accepted"] + 1):
            raise RuntimeError("Greedy committed length differs from accepted + 1")
        return state["token_table"][rows, frontier + 1 : frontier + 1 + f.q], state["accepted"]

    return call, outputs


def vllm_call(f, stage, src):
    """Full upstream V2 rejection entry, with persistent inputs built once.

    Upstream temporaries are allocated by its unchanged wrappers during graph
    capture and live in the graph pool on replay. All admission state, host
    mirrors, seeds, layouts and BF16 logits are allocated before capture.
    """
    import numpy as np  # vLLM V2 host state; GPU runs only

    if stage != "pipeline":
        raise ValueError("V2 has no probability-stage analogue")
    device = f.logits.device
    v2 = src.v2
    states = v2.SamplingStates(f.b, f.v)
    needs_processing = np.empty(f.b, dtype=bool)
    for i in range(f.b):
        needs_processing[i] = states.add_request(
            i,
            SimpleNamespace(
                temperature={"G": 0.0, "T": 0.7, "KP": 0.6, "KPM": 0.6}[f.mode],
                top_k=20 if f.mode in ("KP", "KPM") else -1,
                top_p=0.95 if f.mode in ("KP", "KPM") else 1.0,
                min_p=MIN_P if f.mode == "KPM" else 0.0,
                seed=42 + i,
                logprobs=None,
            ),
        )
    states.apply_staged_writes()
    req_states = SimpleNamespace(
        max_num_reqs=f.b,
        device=device,
        prefill_len=SimpleNamespace(gpu=torch.full((f.b,), 100, dtype=torch.int32, device=device)),
    )
    # Bypass server-only constructor dependencies. All methods in the timed
    # path are upstream, including the disabled thinking-budget fast return.
    sampler = v2.Sampler.__new__(v2.Sampler)
    sampler.sampling_states = states
    sampler.req_states = req_states
    sampler.needs_logits_processing = needs_processing
    sampler.logits_processors = []  # no penalties, bias, bad words or custom processors
    sampler.thinking_budget_state = v2.ThinkingBudgetState(req_states, None)
    sampler.logprobs_mode = "raw_logprobs"
    sampler.compute_nans = False
    sampler.use_fp64_gumbel = False
    rejection = v2.RejectionSampler(
        sampler,
        SimpleNamespace(
            num_speculative_tokens=f.q - 1,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
        device,
    )
    local_pos = torch.arange(f.q, dtype=torch.int32, device=device).repeat(f.b)
    input_ids = torch.zeros((f.b, f.q), dtype=torch.int32, device=device)
    input_ids[:, 1:] = f.drafts  # row zero is the anchor, drafts occupy rows 1..Q-1
    cumulative_np = np.arange(f.b + 1, dtype=np.int32) * f.q
    batch = SimpleNamespace(
        num_reqs=f.b,
        input_ids=input_ids.flatten(),
        positions=local_pos.to(torch.int64) + 100,
        logits_indices=torch.arange(f.b * f.q, dtype=torch.int64, device=device),
        idx_mapping=f.rows,
        idx_mapping_np=np.arange(f.b, dtype=np.int32),
        expanded_idx_mapping=f.rows.repeat_interleave(f.q),
        expanded_local_pos=local_pos,
        cu_num_logits=torch.tensor(cumulative_np, device=device),
        cu_num_logits_np=cumulative_np,
        seq_lens=torch.full((f.b,), 100 + f.q, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.full((f.b,), 100 + f.q, dtype=torch.int32),
    )
    logits = f.logits.view(-1, f.v)
    result = {}

    def call():
        result["out"] = rejection(logits, batch, draft_logits=None)

    def outputs():
        out = result["out"]
        # Normalize emitted-token count to accepted-draft count outside timing.
        return out.sampled_token_ids, out.num_sampled - 1

    return call, outputs


def vllm_v1_call(f, stage, src):
    g = f.q - 1
    cumulative = torch.arange(1, f.b + 1, dtype=torch.int32, device="cuda") * g
    target_indices = torch.tensor(
        [r * f.q + j for r in range(f.b) for j in range(g)], device="cuda"
    )
    bonus_indices = torch.arange(f.b, device="cuda") * f.q + g
    values = {
        "all_greedy": f.mode == "G",
        "all_random": f.mode != "G",
        "temperature": f.temp,
        "top_k": f.k if f.mode == "KP" else None,
        "top_p": f.top_p if f.mode == "KP" else None,
        "generators": {},
        "max_num_logprobs": None,
        "logprob_token_ids": None,
        "no_penalties": True,
        "allowed_token_ids_mask": None,
        "bad_words_token_ids": {},
        "output_token_ids": [[] for _ in range(f.b)],
        "spec_token_ids": [[] for _ in range(f.b)],
        "thinking_budget_state_holder": None,
        "logitsprocs": SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
    }
    metadata_type = dataclasses.make_dataclass(
        "SamplingMetadata", [(key, object) for key in values]
    )
    metadata = metadata_type(**values)
    spec = SimpleNamespace(
        max_spec_len=g,
        num_draft_tokens=[g] * f.b,
        cu_num_draft_tokens=cumulative,
        draft_token_ids=f.drafts.flatten(),
        bonus_logits_indices=bonus_indices,
        target_logits_indices=target_indices,
    )
    rejection = src.reject.RejectionSampler(src.sampler)
    result = {}
    # Splitting existing p is admission/layout setup for verify-only, outside
    # timing for all providers. Pipeline retains upstream gathers and conversions.
    targets = f.p[:, :g].reshape(-1, f.v).contiguous()
    bonus_p = f.p[:, g].contiguous()

    def call():
        if stage == "pipeline":
            result["out"] = rejection(
                spec, None, f.logits.view(-1, f.v), metadata
            ).sampled_token_ids
            return
        if f.mode == "G":
            bonus = bonus_p.argmax(-1).to(torch.int32).unsqueeze(-1)
            result["out"] = src.reject.rejection_sample(
                spec.draft_token_ids,
                spec.num_draft_tokens,
                g,
                cumulative,
                None,
                targets,
                bonus,
                metadata,
            )
            return
        # Given p, filtering and softmax are already complete. These are the
        # upstream RNG/recovered/rejection dispatchers, with no custom kernels.
        bonus = src.top.random_sample(bonus_p.clone(), {}).to(torch.int32).unsqueeze(-1)
        out = torch.full((f.b, f.q), -1, dtype=torch.int32, device="cuda")
        greedy = f.temp == 0
        uniforms = src.reject.generate_uniform_probs(
            f.b * g, spec.num_draft_tokens, {}, targets.device
        )
        recovered = src.reject.sample_recovered_tokens(
            g,
            spec.num_draft_tokens,
            cumulative,
            spec.draft_token_ids,
            None,
            targets,
            metadata,
            targets.device,
        )
        src.reject._rejection_random_sample(
            out,
            cumulative,
            spec.draft_token_ids,
            None,
            targets,
            bonus,
            recovered,
            uniforms,
            greedy,
            g,
            None,
        )
        result["out"] = out

    def outputs():
        out = result["out"]
        # Host parsing/count reconstruction is not included in the GPU stage.
        return out, (out != -1).sum(-1).to(torch.int32) - 1

    return call, outputs


def sglang_triton_proxy_call(f, stage, src):
    scope = (
        "verify_only"
        if stage == "pipeline" and f.mode == "KP" and not src.sgl_kp_available
        else stage
    )
    candidates = torch.zeros((f.b, f.q), dtype=torch.int64, device="cuda")
    candidates[:, 1:] = f.drafts
    counts = torch.empty(f.b, dtype=torch.int32, device="cuda")
    if f.mode == "G":
        out = torch.empty_like(candidates)
        commit, bonus = torch.empty_like(counts), torch.empty(f.b, dtype=torch.int64, device="cuda")
        prefix = torch.full((f.b,), 100, dtype=torch.int32, device="cuda")
        new_lengths = torch.empty_like(prefix)

        def greedy():
            target = (f.logits if stage == "pipeline" else f.p).argmax(-1)
            src.greedy._compute_dflash_accept_bonus_triton_unchecked(
                candidates,
                target,
                counts,
                commit,
                bonus,
                out,
                prefix,
                new_lengths,
            )

        return greedy, lambda: (out, counts), scope
    retrieve = torch.arange(f.b * f.q, dtype=torch.int64, device="cuda").view(f.b, f.q)
    next_token = torch.arange(1, f.q + 1, dtype=torch.int64, device="cuda").repeat(f.b, 1)
    next_token[:, -1] = -1
    sibling = torch.full_like(retrieve, -1)
    predicts = torch.empty(f.b * f.q, dtype=torch.int32, device="cuda")
    accept_index = torch.empty_like(retrieve)
    draft_p = torch.zeros_like(f.p)
    draft_p[:, :-1].scatter_(2, f.drafts.long().unsqueeze(-1), 1)
    info = SimpleNamespace(
        temperatures=f.temp[:, None],
        top_ks=f.k,
        top_ps=f.top_p,
        need_top_k_sampling=f.mode == "KP",
        need_top_p_sampling=f.mode == "KP",
    )
    # Fixed interior coins preserve the existing chain comparator convention.
    coins = torch.full((f.b, f.q), 0.5, dtype=torch.float32, device="cuda")
    final = torch.full((f.b,), 0.5, dtype=torch.float32, device="cuda")

    def call():
        p = f.p
        if scope == "pipeline":
            p = src.sgl.build_speculative_verify_target_probs(
                next_token_logits=f.logits.view(-1, f.v),
                sampling_info=info,
                draft_token_num=f.q,
                bs=f.b,
                max_top_k=20 if f.mode == "KP" else f.v,
                uniform_top_k_value=20 if f.mode == "KP" else f.v,
            )
        src.chain.chain_speculative_sampling_triton(
            predicts,
            accept_index,
            counts,
            candidates,
            retrieve,
            next_token,
            sibling,
            coins,
            final,
            p,
            draft_p,
            1.0,
            1.0,
            True,
            block_verification=False,
        )

    def outputs():
        # Gathering emitted tokens is an untimed output check.
        return predicts[accept_index.clamp(0, predicts.numel() - 1)], counts

    return call, outputs, scope


def sglang_call(f, stage, src):
    """Production DFlash stochastic wrapper; fixed coins are for checks only."""
    if f.mode == "G":
        return sglang_triton_proxy_call(f, stage, src)
    scope = (
        "verify_only"
        if stage == "pipeline" and f.mode == "KP" and not src.sgl_kp_available
        else stage
    )
    helpers = src.sgl_verify_factory(None if scope == "pipeline" else lambda **kw: f.p)
    candidates = torch.cat((torch.zeros_like(f.drafts[:, :1]), f.drafts), dim=1).long()
    info = SimpleNamespace(
        temperatures=f.temp[:, None],
        top_ks=f.k,
        top_ps=f.top_p,
        need_top_k_sampling=f.mode == "KP",
        need_top_p_sampling=f.mode == "KP",
    )
    result = {}

    def call(*, uniform_samples=None, uniform_samples_for_final_sampling=None):
        counts, bonus = helpers.verify(
            candidates=candidates,
            next_token_logits=f.logits.view(-1, f.v),
            sampling_info=info,
            max_top_k=20 if f.mode == "KP" else f.v,
            uniform_top_k_value=20 if f.mode == "KP" else f.v,
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        )
        result["out"], result["commit"] = helpers.commit(candidates, counts, bonus)
        result["counts"] = counts

    def outputs():
        if not torch.equal(result["commit"], result["counts"] + 1):
            raise RuntimeError("SGLang committed length differs from accepted + 1")
        return result["out"], result["counts"]

    return call, outputs, scope


def sglang_check_uniforms(f, accepted):
    """Match K1's seed/cursor/domain mapping outside timing, including final coin.

    K1 has separate correction/bonus domains; SGLang takes one final coin per
    request. The accepted count from K1 selects that request's final domain.
    """
    from ctikernels.sampling._dflash.reference import ACCEPT, BONUS, CORRECTION, uniform

    counts = accepted.cpu().tolist()
    coins = [[uniform(42, 2**32 + b, ACCEPT, i) for i in range(f.q)] for b in range(f.b)]
    final = [
        uniform(42, 2**32 + b, BONUS if count == f.q - 1 else CORRECTION, 0)
        for b, count in enumerate(counts)
    ]
    return {
        "uniform_samples": torch.tensor(coins, dtype=torch.float32, device=f.p.device),
        "uniform_samples_for_final_sampling": torch.tensor(
            final, dtype=torch.float32, device=f.p.device
        ),
    }


def capture_repeated(call):
    def repeated():
        for _ in range(R):
            call()

    return bench.capture(repeated)


def measure(left, right):
    # One event interval covers one graph containing R executions: host replay
    # latency cannot insert gaps between executions within that graph.
    def time(graph):
        return bench.time_graph(graph, repeats=1) / R

    self_pairs = [(time(right), time(right)) for _ in range(bench.PAIRS)]
    ours, theirs = [], []
    for i in range(bench.PAIRS):
        if i % 2:
            b, a = time(right), time(left)
        else:
            a, b = time(left), time(right)
        ours.append(a)
        theirs.append(b)
    return (
        ours,
        theirs,
        bench.paired_summary([b / a for a, b in zip(ours, theirs, strict=False)], self_pairs),
    )


def check(f, ours, theirs):
    """Every fixture row is a point mass, so tokens are deterministic in all
    modes: compare the accepted prefix plus the correction/bonus exactly."""
    a, ac = (t.cpu() for t in ours())
    b, bc = (t.cpu() for t in theirs())
    expected = 0 if f.case == "reject" else f.q - 1
    counts = torch.equal(ac, bc) and bool((ac == expected).all())
    visible = torch.arange(f.q)[None, :] <= ac[:, None]
    return "tokens+counts:ok" if counts and torch.equal(a[visible], b[visible]) else "FAIL"


def compact(value):
    if isinstance(value, float):
        return float(f"{value:.4g}")
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    return value


COMPARATOR_ORDER = ("vllm", "vllm_v1", "sglang", "sglang_triton_proxy")
STAT_ORDER = ("gm", "lo", "hi", "eps", "claim")


def availability(f, stage):
    """Missing timings are explicit and never eligible for fastest selection."""
    if f.mode == "KPM":
        return {
            "vllm": None if stage == "pipeline" else "no probability-stage analogue",
            "vllm_v1": "min-p rows compare V2 only",
            "sglang": "SGLang DFlash has no min-p",
            "sglang_triton_proxy": "SGLang DFlash has no min-p",
        }
    return {
        "vllm": None if stage == "pipeline" else "no probability-stage analogue",
        "vllm_v1": "grid budget" if stage == "pipeline" and f.v == 151936 else None,
        "sglang": None,
        "sglang_triton_proxy": "stochastic only" if f.mode == "G" else None,
    }


def summarize_row(f, stage, timings, summaries, launches, checks, scope, available=None):
    row = {
        "B": f.b,
        "Q": f.q,
        "V": f.v,
        "mode": f.mode,
        "case": f.case,
        "stage": stage,
        "runner": "v2",
        "ours_us": statistics.median(timings["ours"]),
    }
    for name, reason in (available or availability(f, stage)).items():
        item = {"runner": "v2" if name == "vllm" else "v1"} if name.startswith("vllm") else {}
        if reason:
            item["na"] = reason
        else:
            item["us"] = statistics.median(timings[name])
            item["stats"] = [summaries[name][key] for key in STAT_ORDER]
        row[name] = item
    if scope == "verify_only":
        for name in ("sglang", "sglang_triton_proxy"):
            row[name]["scope"] = scope
            if "stats" in row[name]:
                row[name]["stats"][-1] = False
    eligible = [
        name
        for name in COMPARATOR_ORDER
        if name in timings
        and name != "sglang_triton_proxy"
        and not (name == "sglang" and scope == "verify_only")
    ]
    primary = [name for name in eligible if stage != "pipeline" or name != "vllm_v1"]
    row["faster"] = min(primary, key=lambda name: row[name]["us"])
    row["faster_incl_v1"] = min(eligible, key=lambda name: row[name]["us"])
    # Compact fixed-order fields keep the full grid below 250 KB. Definitions
    # also go to setup stderr so stdout remains one record per fixture/stage.
    row["launches"] = [launches.get(name) for name in ("ours", *COMPARATOR_ORDER)]
    row["check"] = [checks.get(name) for name in COMPARATOR_ORDER]
    if stage == "pipeline" and f.mode == "G":
        row["ours_greedy"] = {
            "us": statistics.median(timings["ours_greedy"]),
            "vs": {
                name: [summary[key] for key in STAT_ORDER]
                for name, summary in summaries["ours_greedy"].items()
            },
            "launches": launches["ours_greedy"],
        }
        row["ours_greedy_torch"] = {
            "us": statistics.median(timings["ours_greedy_torch"]),
            # Torch/CuTe, paired directly; same CI/noise-floor claim as comparators.
            "stats": [summaries["ours_greedy_torch"][key] for key in STAT_ORDER],
            "launches": launches["ours_greedy_torch"],
        }
    return compact(row)


def run_row(f, stage, src):
    ours, our_outputs = ours_call(f, stage)
    calls, outputs = {"ours": ours}, {}
    available = availability(f, stage)
    if available["vllm"] is None:
        calls["vllm"], outputs["vllm"] = vllm_call(f, stage, src)
    if available["vllm_v1"] is None:
        v1_call, v1_outputs = vllm_v1_call(f, stage, src)
        try:
            v1_call()  # V1's bonus sampler imports FlashInfer lazily on first use.
        except ModuleNotFoundError as error:
            # V1 is informational only; a host without its optional dependency
            # keeps the primary V2/SGLang comparison and marks V1 unavailable.
            available["vllm_v1"] = f"unavailable: {error.name} not installed"
        else:
            calls["vllm_v1"], outputs["vllm_v1"] = v1_call, v1_outputs
    scope = stage
    if available["sglang"] is None:
        calls["sglang"], outputs["sglang"], scope = sglang_call(f, stage, src)
    if available["sglang_triton_proxy"] is None:
        calls["sglang_triton_proxy"], outputs["sglang_triton_proxy"], _ = sglang_triton_proxy_call(
            f, stage, src
        )
    greedy_outputs = {}
    if stage == "pipeline" and f.mode == "G":
        calls["ours_greedy"], greedy_outputs["ours_greedy"] = ours_greedy_call(f)
        calls["ours_greedy_torch"], greedy_outputs["ours_greedy_torch"] = ours_greedy_call(
            f, use_torch=True
        )

    def validate():
        checks = {name: check(f, our_outputs, other) for name, other in outputs.items()}
        # Check the actual eager/captured outputs first, so an untimed fixed-
        # coin call cannot hide a failure of the timed production sequence.
        if f.mode != "G" and checks.get("sglang", "FAIL") != "FAIL":
            calls["sglang"](**sglang_check_uniforms(f, our_outputs()[1]))
            checks["sglang"] = check(f, our_outputs, outputs["sglang"])
        for variant, result in greedy_outputs.items():
            greedy_checks = {
                name: check(f, result, other)
                for name, other in {"ours": our_outputs, **outputs}.items()
            }
            if "FAIL" in greedy_checks.values():
                raise RuntimeError(f"{variant} output validation failed: {greedy_checks}")
        if "FAIL" in checks.values():
            raise RuntimeError(f"Output validation failed: {checks}")
        return {
            name: "tc" if value == "tokens+counts:ok" else "c" for name, value in checks.items()
        }

    for call in calls.values():
        call()
    validate()
    launches = {name: bench.count_launches(call)["kernel_launches"] for name, call in calls.items()}
    graphs = {name: capture_repeated(call) for name, call in calls.items()}
    for graph in graphs.values():
        graph.replay()
    checks = validate()
    timings, summaries = {"ours": []}, {}
    for name in outputs:
        our_times, timings[name], summaries[name] = measure(graphs["ours"], graphs[name])
        timings["ours"].extend(our_times)
    if greedy_outputs:
        timings["ours_greedy"], summaries["ours_greedy"] = [], {}
        for name in outputs:
            our_times, _, summary = measure(graphs["ours_greedy"], graphs[name])
            timings["ours_greedy"].extend(our_times)
            summaries["ours_greedy"][name] = summary
        our_times, timings["ours_greedy_torch"], summaries["ours_greedy_torch"] = measure(
            graphs["ours_greedy"], graphs["ours_greedy_torch"]
        )
        timings["ours_greedy"].extend(our_times)
        validate()
    return summarize_row(f, stage, timings, summaries, launches, checks, scope, available)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-root", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, choices=BATCHES)
    parser.add_argument("--q", nargs="+", type=int, choices=QS + EXTRA_QS)
    parser.add_argument("--vocab", nargs="+", type=int, choices=VOCABS)
    parser.add_argument("--modes", nargs="+", choices=MODES + EXTRA_MODES)
    return parser.parse_args(argv)


def grid(args):
    for b, q, v, mode, case in itertools.product(
        args.batches or ((1,) if args.quick else BATCHES),
        args.q or ((4,) if args.quick else QS),
        args.vocab or ((248320,) if args.quick else VOCABS),
        args.modes or MODES,
        ("reject", "bonus"),
    ):
        yield b, q, v, mode, case, "pipeline"
        if v == 248320 and mode != "KPM":
            yield b, q, v, mode, case, "verify"


def main(argv=None):
    args = parse_args(argv)
    digest = upstream.validate_bundle(args.comparator_root)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for execution; importing this module is CPU-safe")
    with contextlib.redirect_stdout(sys.stderr):
        src = upstream.load(args.comparator_root)
        print(
            json.dumps(
                {
                    "event": "setup",
                    "manifest_sha256": digest,
                    "vllm_runner": "V2-primary; V1-informational",
                    "R": R,
                    "pairs": 20,
                    "self_pairs": 20,
                    "launch_order": ["ours", *COMPARATOR_ORDER],
                    "check_order": COMPARATOR_ORDER,
                    "check_codes": {"tc": "tokens+counts:ok", "c": "counts:ok"},
                    "stats_order": STAT_ORDER,
                    "ours_greedy": "G pipeline: CuTe RowArgmax + commit_block; vs uses stats_order",
                    "ours_greedy_torch": "torch.argmax + commit_block; stats=Torch/CuTe paired",
                    "vllm_v1_omission": "pipeline V=151936: grid budget",
                    "vllm_v1_bonus": "flashinfer-eligible" if src.flashinfer_enabled else "native",
                    "sglang": "greedy-native / production-CUDA-JIT-tree; timed torch RNG + gather/commit",
                    "sglang_triton_proxy": "informational; fixed uniforms; excluded from faster",
                    "sglang_checks": "K1-matched uniforms outside timing",
                    "timing": "graph-only; no eager fallback; no logprobs/penalties",
                }
            ),
            file=sys.stderr,
        )
    total_bytes = 0
    failures = 0
    for b, q, v, mode, case, stage in grid(args):
        with contextlib.redirect_stdout(sys.stderr):
            print(f"B={b} Q={q} V={v} {mode} {case} {stage}", file=sys.stderr, flush=True)
            try:
                f = fixture(b, q, v, mode, case)
                row = run_row(f, stage, src)
                del f
            except (RuntimeError, ImportError, OSError) as error:
                traceback.print_exc(file=sys.stderr)
                failures += 1
                row = {
                    "B": b,
                    "Q": q,
                    "V": v,
                    "mode": mode,
                    "case": case,
                    "stage": stage,
                    "error": type(error).__name__,
                    "timing": "unavailable; no eager fallback",
                }
            gc.collect()
        line = json.dumps(row, separators=(",", ":"))
        total_bytes += len(line) + 1
        print(line, flush=True)
    print(f"JSONL bytes: {total_bytes}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} rows failed; see stderr")


if __name__ == "__main__":
    main()
