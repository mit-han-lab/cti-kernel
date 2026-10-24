# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Runtime-source DFlash bookkeeping comparators: SGLang 72730d6 / vLLM 0.31.0.

Loads python/sglang/kernels/ops/{speculative/dflash,kvcache/kv_indices}.py
and vllm/v1/{spec_decode/utils,sample/rejection_sampler,worker/block_table}.py.
Launch sequences follow python/sglang/srt/speculative/dflash_worker_v2.py,
dflash_info.py, and vllm/v1/spec_decode/dflash.py / worker/gpu_model_runner.py.

Scope: dense attention, TP=1, ordinary RoPE, greedy, no grammar or logprobs;
SGLang static pool + FlashInfer (page=1), vLLM FlashAttention (page=16),
synchronous scheduler + padded drafter. Model work, logits/argmax, attention
scheduling, host work/transfers, and proposal copies retained by R5 are excluded.
These are GPU bookkeeping totals for the stated paths, not end-to-end latency
or a proof of the fastest possible framework/backend configuration.

configure() takes the directory containing sglang/ and vllm/. Only trusted
upstream source files should be supplied. No framework installation is needed.
build_all(..., comparator="vllm") captures one framework's ordered steps;
COMPARATORS[name].build_all(...) offers the same five-argument interface.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import importlib.abc
import importlib.util
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType

Step = tuple[str, Callable[[], None] | None, str]
_root: Path | None = None
_modules: dict[str, ModuleType] = {}

# This is a definition allowlist, not a replacement implementation. Bodies and
# Triton decorators are compiled from the supplied file with original line numbers.
_SOURCES = {
    "sgl_dflash": (
        "sglang/kernels/ops/speculative/dflash.py",
        (
            "_dflash_accept_bonus_contig_kernel",
            "_pick_num_warps",
            "_is_row_major_contiguous_2d",
            "_compute_dflash_accept_bonus_triton_unchecked",
            "_prepare_dflash_draft_block_contig_kernel",
            "_prepare_dflash_draft_block_unchecked",
        ),
    ),
    "sgl_kv": (
        "sglang/kernels/ops/kvcache/kv_indices.py",
        ("create_flashinfer_kv_indices_triton",),
    ),
    "vllm_utils": (
        "vllm/v1/spec_decode/utils.py",
        (
            "copy_and_expand_dflash_inputs_kernel",
            "eagle_prepare_next_token_padded_kernel",
            "eagle_prepare_inputs_padded_kernel",
        ),
    ),
    "vllm_reject": (
        "vllm/v1/sample/rejection_sampler.py",
        ("rejection_greedy_sample_kernel",),
    ),
    "vllm_slots": (
        "vllm/v1/worker/block_table.py",
        ("ComputeSlotMappingKernel.kernel",),
    ),
}


class _DefinitionsLoader(importlib.abc.Loader):
    """Import selected upstream definitions without importing their application.

    Default globals are torch, triton and tl; callers may supply explicit
    dependencies. Extracted static/class methods lose their binding decorator.
    No process-wide sys.modules shims for sglang/vllm are installed here.
    """

    def __init__(self, path: Path, names: tuple[str, ...], globals=None):
        self.path, self.names = path, names
        self.globals = globals or {}

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import torch
        import triton
        import triton.language as tl

        parsed = ast.parse(self.path.read_text(), filename=str(self.path))
        selected = []
        for qualified in self.names:
            body = parsed.body
            node = None
            for part in qualified.split("."):
                node = next((n for n in body if part in _definition_names(n)), None)
                if node is None:
                    raise ImportError(f"Missing upstream definition: {qualified}")
                body = getattr(node, "body", [])
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = [
                    d
                    for d in node.decorator_list
                    if not (isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod"))
                ]
            selected.append(node)
        module.__dict__.update(torch=torch, triton=triton, tl=tl)
        module.__dict__.update(self.globals)
        tree = ast.Module(body=selected, type_ignores=[])
        exec(
            compile(tree, str(self.path), "exec", flags=__future__.annotations.compiler_flag),
            module.__dict__,
        )


def _definition_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return (node.name,)
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    )
    return tuple(t.id for t in targets if isinstance(t, ast.Name))


def load_definitions(path, names, *, module_name, globals=None):
    """Load trusted runtime definitions/constants with explicit dependency globals.

    Imports are omitted; implementation bodies and source locations are intact.
    Callers own dependency adapters and must document any substituted helpers.
    """
    path = Path(path)
    loader = _DefinitionsLoader(path, tuple(names), globals)
    spec = importlib.util.spec_from_loader(module_name, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def configure(root: str) -> None:
    """Select staged sources, validating the five files without initializing CUDA."""
    global _root
    candidate = Path(root).expanduser().resolve()
    for relative, _ in _SOURCES.values():
        if not (candidate / relative).is_file():
            raise FileNotFoundError(f"Comparator root is missing {relative}")
    for module in _modules.values():
        sys.modules.pop(module.__name__, None)
    _modules.clear()
    _root = candidate


def load_sources() -> dict[str, ModuleType]:
    """CPU-safe import check; no JIT compilation or kernel execution occurs here."""
    if _root is None:
        raise RuntimeError("Call configure(comparator_root) before building comparators")
    for key, (relative, names) in _SOURCES.items():
        if key in _modules:
            continue
        loader = _DefinitionsLoader(_root / relative, names)
        name = f"_dflash_comparator_{key}"
        spec = importlib.util.spec_from_loader(name, loader, origin=str(_root / relative))
        if spec is None:
            raise ImportError(f"Cannot create source spec for {relative}")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(_root / relative)
        sys.modules[name] = module
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _modules[key] = module
    return dict(_modules)


def _validate(B, Q, F, M):
    # The upstream Triton kernels take Q as a runtime/power-of-two block size: no Q cap.
    if not (1 <= B <= 4 and 2 <= Q and F >= Q and M >= F + Q):
        raise ValueError("Require 1<=B<=4, 2<=Q, Q<=F and F+Q<=M")


def _fixtures(B, Q, device):
    """Deterministic ordinary IDs, with varied prefix acceptance across requests."""
    import torch

    accepted = [(Q - 1) // 2] if B == 1 else [(Q - 1) * b // (B - 1) for b in range(B)]
    proposals = [[100 + b * Q + j for j in range(Q - 1)] for b in range(B)]
    targets = [
        [value if j < accepted[b] else value + 1000 for j, value in enumerate(row)] + [2000 + b]
        for b, row in enumerate(proposals)
    ]
    return (
        accepted,
        torch.tensor(proposals, dtype=torch.int64, device=device),
        torch.tensor(targets, dtype=torch.int64, device=device),
    )


def _sglang(B, Q, F, M, device) -> list[Step]:
    import torch

    _validate(B, Q, F, M)
    modules = load_sources()
    dflash, kv = modules["sgl_dflash"], modules["sgl_kv"]
    _, proposals, target = _fixtures(B, Q, device)
    prefix = torch.full((B,), F, dtype=torch.int64, device=device)
    rows = torch.arange(B, dtype=torch.int64, device=device)
    table = torch.arange(B * M, dtype=torch.int32, device=device).reshape(B, M)
    anchors = torch.full((B,), 42, dtype=torch.int64, device=device)
    block_ids = torch.empty((B, Q), dtype=torch.int64, device=device)
    positions, cache_loc = torch.empty_like(block_ids), torch.empty_like(block_ids)
    # Draft-head assembly is outside R5 P/C; seed the resulting candidate buffer.
    candidates = torch.cat((anchors[:, None], proposals), dim=1)
    accept = torch.empty((B,), dtype=torch.int32, device=device)
    commit, bonus = torch.empty_like(accept), torch.empty_like(anchors)
    out, new_lengths = torch.empty_like(block_ids), torch.empty_like(prefix)

    def prepare():
        dflash._prepare_dflash_draft_block_unchecked(
            anchors, prefix, rows, table, block_ids, positions, cache_loc, 3
        )

    def make_csr():
        indices = torch.empty((B * (F + Q),), dtype=torch.int32, device=device)
        # Keep all graph-pool outputs live until this step's consumer is queued.
        scratch = {}

        def run():
            # dflash_info.py:159-196, shared by draft and target FlashInfer.
            qo = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=device)
            indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
            lengths = prefix + Q
            indptr[1:] = torch.cumsum(lengths, dim=0)
            kv.create_flashinfer_kv_indices_triton[(B,)](
                table, rows, lengths, indptr, None, indices, M, ENTRY_PAGE_SIZE=1
            )
            scratch.update(qo=qo, indptr=indptr, lengths=lengths)

        return run

    def verify():
        dflash._compute_dflash_accept_bonus_triton_unchecked(
            candidates, target, accept, commit, bonus, out, prefix, new_lengths
        )

    csr_note = (
        "FlashInfer static page=1: arange, zeros, add Q, cumsum(int64), "
        "copy/cast indptr, upstream KV gather grid=(B,); 6 expected GPU launches. "
        "Torch temporaries allocate in graph pool; provider plan excluded."
    )
    return [
        (
            "prepare:inputs",
            prepare,
            "1 upstream Triton launch; anchor/masks, positions AND cache locations.",
        ),
        ("prepare:cache_loc", None, "Included in prepare:inputs; no separate launch."),
        ("prepare:kv_indices:draft", make_csr(), csr_note),
        ("prepare:kv_indices", make_csr(), "Target verify. " + csr_note),
        (
            "commit:accept",
            verify,
            "1 upstream Triton launch; accept, bonus, output tokens, commit and new lengths.",
        ),
        (
            "commit:finish/lengths",
            None,
            "Lengths included in commit:accept; EOS/budget/finish on CPU, no GPU equivalent.",
        ),
        (
            "commit:token_table",
            None,
            "CPU request output list; GPU kernel emits a round buffer, no persistent token table.",
        ),
        (
            "prepare:rope/guards",
            None,
            "Ordinary positions already emitted; no R5 clamp, done-row guards or M-RoPE offsets.",
        ),
        (
            "prepare:allocation",
            None,
            "Scheduler reserves mapping before the round; excluded like R5 mapping updates.",
        ),
    ]


def _vllm(B, Q, F, M, device) -> list[Step]:
    import torch
    import triton

    _validate(B, Q, F, M)
    modules = load_sources()
    utils, rejection, slots = (modules["vllm_utils"], modules["vllm_reject"], modules["vllm_slots"])
    accepted, proposals, target = _fixtures(B, Q, device)
    page = 16
    # Match block-table token alignment (128), standard single-rank dense cache.
    width = triton.cdiv(M, 128) * (128 // page)
    table = torch.arange(B * width, dtype=torch.int32, device=device).reshape(B, width)
    query_start = torch.arange(B + 1, dtype=torch.int32, device=device) * Q
    cumulative_drafts = torch.arange(1, B + 1, dtype=torch.int32, device=device) * (Q - 1)
    prefix = torch.full((B,), F, dtype=torch.int32, device=device)
    scheduled = torch.full((B,), Q, dtype=torch.int32, device=device)
    req_indices = torch.arange(B, dtype=torch.int64, device=device).repeat_interleave(Q)
    offsets = torch.arange(Q, dtype=torch.int64, device=device).repeat(B)
    positions = torch.empty((B * Q,), dtype=torch.int64, device=device)
    seq_lens = torch.empty((B,), dtype=torch.int32, device=device)
    slot_mapping = torch.empty_like(positions)
    anchors = torch.full((B,), 42, dtype=torch.int32, device=device)
    # Pre-uploaded scheduler inputs: transfers and retained proposal copies excluded.
    verify_ids = torch.cat((anchors[:, None], proposals.to(torch.int32)), dim=1).flatten()
    logits_indices = torch.arange(B * Q, dtype=torch.int64, device=device)
    draft_indices = torch.tensor(
        [b * Q + j for b in range(B) for j in range(Q - 1)],
        dtype=torch.int64,
        device=device,
    )
    draft_ids = proposals.to(torch.int32).flatten()
    target_argmax = target[:, :-1].contiguous().flatten()
    bonus = target[:, -1].contiguous()
    output = torch.empty((B, Q), dtype=torch.int32, device=device)
    discarded = torch.zeros((B,), dtype=torch.bool, device=device)
    next_ids = anchors.clone()
    valid_counts = torch.tensor([a + 1 for a in accepted], dtype=torch.int32, device=device)
    rejected = torch.tensor([Q - a - 1 for a in accepted], dtype=torch.int32, device=device)
    sample_indices = torch.empty((B,), dtype=torch.int32, device=device)

    # Previous round's padded context has Q states, of which a+1 are valid.
    # Its last VALID position is F-1. F is total KV context, not Q context states.
    context_positions = torch.tensor(
        [F - a - 1 + j for a in accepted for j in range(Q)],
        dtype=torch.int64,
        device=device,
    )
    context_lengths = torch.tensor(
        [F + Q - a - 1 for a in accepted], dtype=torch.int32, device=device
    )
    draft_inputs = torch.empty((B * Q,), dtype=torch.int32, device=device)
    draft_positions = torch.empty_like(positions)
    context_pos_out = torch.empty_like(positions)
    context_slots, draft_slots = torch.empty_like(positions), torch.empty_like(positions)
    draft_sample_indices = torch.empty((B * (Q - 1),), dtype=torch.int32, device=device)
    arange = torch.arange(B + 1, dtype=torch.int32, device=device)
    scratch = {}

    def draft_prepare():
        # dflash.py:151-190; default padded drafter, Q context + Q query states.
        block = min(256, triton.next_power_of_2(2 * Q))
        utils.copy_and_expand_dflash_inputs_kernel[(B, triton.cdiv(2 * Q, block))](
            next_ids,
            context_positions,
            draft_inputs,
            context_pos_out,
            draft_positions,
            context_slots,
            draft_slots,
            draft_sample_indices,
            table,
            width,
            query_start,
            rejected,
            3,
            page,
            Q,
            Q - 1,
            B * Q,
            BLOCK_SIZE=block,
            HAS_NUM_REJECTED=True,
        )

    def draft_metadata():
        # Preserve the upstream allocations and three GPU arithmetic operations.
        scratch["draft_qo"] = arange * Q
        effective = context_lengths - rejected
        scratch["draft_lengths"] = effective + Q

    def target_prepare():
        # gpu_model_runner.py:2174-2181; no changes to scheduler source buffers.
        positions[:] = prefix[req_indices].to(torch.int64) + offsets
        seq_lens[:] = prefix + scheduled

    def target_slots():
        # block_table.py:543-552, including the extra padding CTA (empty here).
        slots.kernel[(B + 1,)](
            B * Q,
            B * Q,
            query_start,
            positions,
            table,
            width,
            page,
            slot_mapping,
            KV_CACHE_BLOCK_SIZE=page,
            BLOCKS_PER_KV_BLOCK=1,
            TOTAL_CP_WORLD_SIZE=1,
            TOTAL_CP_RANK=0,
            CP_KV_CACHE_INTERLEAVE_SIZE=1,
            PAD_ID=-1,
            BLOCK_SIZE=1024,
        )

    def target_metadata():
        # gpu_model_runner.py:2875-2876; CPU index construction/H2D excluded.
        selected = verify_ids[logits_indices]
        scratch["draft_ids"] = selected[draft_indices + 1]

    def verify():
        # rejection_sampler.py:443-481. Full()'s fill must repeat every round.
        output.fill_(-1)
        rejection.rejection_greedy_sample_kernel[(B,)](
            output,
            cumulative_drafts,
            scratch.get("draft_ids", draft_ids),
            target_argmax,
            bonus,
            None,
            Q - 1,
            None,
            None,
            SYNTHETIC_MODE=False,
        )

    def lengths():
        # llm_base_proposer.py:1084-1096, then 1122-1129; no CPU finish logic.
        utils.eagle_prepare_next_token_padded_kernel[(B,)](
            output,
            discarded,
            anchors,
            next_ids,
            valid_counts,
            32000,
            Q,
            B,
            Q,
            BLOCK_SIZE_TOKENS=triton.next_power_of_2(Q),
        )
        utils.eagle_prepare_inputs_padded_kernel[(B,)](
            cumulative_drafts, valid_counts, query_start, sample_indices, rejected, B
        )

    # A timing step must also work independently: initialize its input buffers
    # without launching upstream kernels. These stores are outside timed closures.
    positions.copy_(torch.arange(F, F + Q, dtype=torch.int64, device=device).repeat(B))
    seq_lens.fill_(F + Q)
    output.copy_(target)
    output.masked_fill_(torch.arange(Q, device=device)[None, :] >= valid_counts[:, None], -1)
    return [
        (
            "prepare:inputs",
            draft_prepare,
            "1 upstream Triton; draft IDs, context/query positions, cache slots, sample indices; page=16.",
        ),
        (
            "prepare:draft_lengths",
            draft_metadata,
            "3 torch ops: arange*Q, seq_lens-rejected, +Q; graph-pool temporaries.",
        ),
        (
            "prepare:target_inputs",
            target_prepare,
            "6 torch ops: gather, cast, add, copy positions; add, copy lengths; graph-pool temporaries.",
        ),
        (
            "prepare:cache_loc",
            target_slots,
            "1 upstream Triton target slot-mapping launch, grid=(B+1,); draft slots included in prepare:inputs.",
        ),
        (
            "prepare:verify_metadata",
            target_metadata,
            "3 torch ops: gather IDs, add index offset, gather drafts; graph-pool temporaries.",
        ),
        (
            "prepare:kv_indices",
            None,
            "FlashAttention consumes persistent page block table directly; no O(F) token-index rebuild.",
        ),
        (
            "commit:accept",
            verify,
            "Output fill + 1 upstream greedy rejection Triton; target/bonus argmax supplied, excluded as in R5.",
        ),
        (
            "commit:finish/lengths",
            lengths,
            "2 upstream Triton: valid count/next anchor, then rejected count/sample index; EOS/budget/finish on CPU.",
        ),
        (
            "commit:finish",
            None,
            "CPU scheduler check_stop truncates accepted output at EOS/stop/budget; no GPU done table.",
        ),
        (
            "commit:token_table",
            None,
            "Persistent token/output lists and frontier maintained on host; no R5 device token table.",
        ),
        (
            "prepare:rope/guards",
            None,
            "No R5 done-row guards or rope clamp in these DFlash helpers.",
        ),
        (
            "prepare:allocation",
            None,
            "Persistent block-table allocation/upload occurs in scheduler; outside timed GPU bookkeeping.",
        ),
    ]


COMPARATORS: dict[str, Callable[..., list[Step]]] = {"sglang": _sglang, "vllm": _vllm}


def build_all(B, Q, F, M, device, *, comparator: str = "sglang") -> Callable[[], None]:
    """Build one framework's steps once; replay all non-None steps in order.

    No synthetic 'total' step is inserted in the step list (that would cause
    double counting). The callable retains buffers and never advances F.
    """
    launches = [fn for _, fn, _ in COMPARATORS[comparator](B, Q, F, M, device) if fn]

    def run():
        for launch in launches:
            launch()

    return run


for _name, _builder in COMPARATORS.items():
    _builder.build_all = partial(build_all, comparator=_name)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-root", required=True)
    parser.add_argument(
        "--load-only", action="store_true", help="Import upstream definitions on CPU only"
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--frontier", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args()
    configure(args.comparator_root)
    load_sources()
    print("Loaded five upstream source files without framework package imports.")
    if args.load_only:
        print("CPU load check passed; CUDA was not queried or initialized.")
        return
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; upstream CPU load check passed.")
        return
    for name, build in COMPARATORS.items():
        steps = build(args.batch, args.block, args.frontier, args.capacity, "cuda")
        for step, fn, note in steps:
            print(f"{name:7s} {step:27s} {'RUN' if fn else 'N/A':3s} {note}")
            if fn:
                fn()
        torch.cuda.synchronize()
    print("CUDA eager self-check passed (graph capture/performance require runner validation).")


if __name__ == "__main__":
    _main()
