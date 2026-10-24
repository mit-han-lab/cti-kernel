# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Graph benchmark, JSONL stdout. Run with PYTHONPATH=python.

Examples: python benchmark/dflash_sampling/bench.py
          python benchmark/dflash_sampling/bench.py --sglang-path DIR [--sglang-path DIR]

B=1,2,3,4,8; dense gamma=15, sparse gamma=7; V=248320. Cases deliberately
fix first rejection or all-accepted bonus so path costs are interpretable.
Bytes are logical traffic estimates (including scratch passes), not measured
DRAM transactions. Cache hits and repeated sparse-ID loads make SOL a model,
not a hardware counter. No shared SOL helper exists in this repository.

The pinned SGLang 72730d6 chain_speculative_sampling_triton accepts saved dense
q with block_verification=False. Its target-only tree entry point has different
proposal semantics and is not used for a sampled-q comparison. If the chain
module cannot import, use FlashInfer's chain_speculative_sampling, clearly
labeled. Comparator timings include provider RNG and output handling; they do
not imply RNG ABI equivalence. SGLang receives fixed uniforms outside timing.
Comparators run in this process, preserving the harness's dependency activation.
Import provenance is logged once to stderr with a relative module file path.
Fixed-uniform output/count matches are informational and checked outside timing.
Point-mass comparisons feed SGLang explicit one-hot draft_probs with the same
fixed uniforms, matching target-only/greedy-draft usage; tokens/counts must match.
Twenty interleaved pairs and twenty self-pairs per case estimate timing noise.
Imports/JIT diagnostics go to stderr; stdout contains JSON measurement lines.
"""

import argparse
import contextlib
import functools
import importlib
import importlib.util
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

import torch

from ctikernels.sampling._dflash import RequestLocalChainSampling
from ctikernels.sampling._dflash.reference import ACCEPT, BONUS, CORRECTION, uniform

PEAK_GBS = {"RTX 5090": 1792, "RTX PRO 6000": 1792, "GB10": 273}
V = 248320
PAIRS = 20


def paired_summary(ratios, self_pairs):
    """K1.22: twenty paired log ratios and nearest-rank p95 self noise."""
    if len(ratios) != PAIRS or len(self_pairs) != PAIRS:
        raise ValueError("K1.22 requires twenty measured pairs and twenty self-pairs")
    epsilon = sorted(max(a / b, b / a) - 1 for a, b in self_pairs)[18]
    logs = [math.log(ratio) for ratio in ratios]
    mean = statistics.mean(logs)
    margin = 2.093024 * statistics.stdev(logs) / math.sqrt(PAIRS)
    lo, hi = math.exp(mean - margin), math.exp(mean + margin)
    return {"gm": math.exp(mean), "lo": lo, "hi": hi, "eps": epsilon, "claim": lo > 1 + epsilon}


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def peak_bandwidth(name):
    return next((bw for model, bw in PEAK_GBS.items() if model in name), None)


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def count_launches(call):
    """Count actual CUDA kernel events in one warmed eager call (outside timing).

    CUDAGraph has no public node-count API. Exclude CUDA copies/memsets: these
    are GPU activities but not kernel launches. No static API count substitutes
    for a missing CUPTI result.
    """
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            call()
            torch.cuda.synchronize()
        kernels = [
            event
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.lower().startswith(("memcpy", "memset"))
        ]
        if not kernels:
            raise RuntimeError("profiler returned no CUDA kernel events; check CUPTI availability")
        return {"kernel_launches": len(kernels), "launch_count_source": "torch.profiler eager call"}
    except (RuntimeError, AttributeError) as error:
        traceback.print_exc(file=sys.stderr)
        return {
            "kernel_launches": None,
            "launch_count_source": "unavailable",
            "launch_count_detail": str(error)[:500],
        }


def verify_traffic(b, g, sparse, case, *, point=False):
    """Logical upper estimate; no full-vocabulary scratch or second row pass."""
    chunks = (V + 4095) // 4096
    width = 8 if case == "rejection" and not (sparse or point) else 4
    visited = 1 if case == "rejection" else g
    acceptance_bytes = (chunks + 1) * visited * (72 if sparse else 4 if point else 8)
    # 16-byte records written/read, then one 4-byte sum per chunk for selection.
    partial_bytes = chunks * 36
    return b * (
        width * (V + 4096)
        + partial_bytes
        + acceptance_bytes
        + (chunks + 1) * 16
        + (g + 1) * 4
        + 4
        + ((chunks + 1) * 128 if sparse and case == "rejection" else 0)
        + ((chunks + 1) * 4 if point and case == "rejection" else 0)
    )


def time_graph(graph, repeats=50):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def make_case(b, sparse, case, *, point=False):
    g = 7 if sparse else 15
    op = RequestLocalChainSampling("cuda", g + 1, V)
    p = torch.full((b, g + 1, V), 1 / V, device="cuda")
    q = torch.full((b, g, V), 1 / V, device="cuda")
    x = torch.zeros(b, g, dtype=torch.int32, device="cuda")
    temp = torch.ones(b, device="cuda")
    common = {
        "request_indices": torch.arange(b, dtype=torch.int32, device="cuda"),
        "seeds": torch.tensor([[42, 0]], dtype=torch.uint32, device="cuda").repeat(b, 1),
        "cursors": torch.arange(b, dtype=torch.int64, device="cuda") + 2**32,
        "workspace": torch.empty(op.workspace_size(b), dtype=torch.uint8, device="cuda"),
    }
    out = torch.empty((b, g + 1), dtype=torch.int32, device="cuda")
    counts = torch.empty(b, dtype=torch.int32, device="cuda")
    if sparse:
        ids = torch.arange(16, dtype=torch.int32, device="cuda").repeat(b, g, 1)
        sq = torch.full((b, g, 16), 1 / 16, device="cuda")
        # Match sparse q for the accepted prefix; final bonus stays full vocabulary.
        p[:, :g].zero_()
        p[:, :g, :16] = 1 / 16
    if point:
        q.zero_()
        q[:, :, 0] = 1
        p[:, :g] = q
    if case == "rejection":
        p[:, 0] = 1 / (V - 1)
        p[:, 0, 0] = 0
    if point:

        def call():
            op.verify_point_mass(p, x, temp, **common, out_tokens=out, accepted_counts=counts)
    elif sparse:

        def call():
            op.verify_sparse(p, x, ids, sq, temp, **common, out_tokens=out, accepted_counts=counts)
    else:

        def call():
            op.verify_dense(p, x, q, temp, **common, out_tokens=out, accepted_counts=counts)

    return op, call, (p, q, x, temp, common, out, counts)


def measurements():
    name = torch.cuda.get_device_name()
    bw = peak_bandwidth(name)
    for b in (1, 2, 3, 4, 8):
        op = RequestLocalChainSampling("cuda", 8, V)
        scores = torch.arange(16, dtype=torch.float32, device="cuda").repeat(b, 1)
        ids = torch.arange(16, dtype=torch.int32, device="cuda").repeat(b, 1)
        temp = torch.ones(b, device="cuda")
        out, oq = (
            torch.empty(b, dtype=torch.int32, device="cuda"),
            torch.empty_like(scores),
        )
        common = {
            "request_indices": torch.arange(b, dtype=torch.int32, device="cuda"),
            "seeds": torch.tensor([[42, 0]], dtype=torch.uint32, device="cuda").repeat(b, 1),
            "cursors": torch.arange(b, dtype=torch.int64, device="cuda") + 2**32,
            "workspace": torch.empty(0, dtype=torch.uint8, device="cuda"),
        }

        def candidates(op=op, scores=scores, ids=ids, temp=temp, common=common, out=out, oq=oq):
            op.sample_candidates(scores, ids, temp, 0, **common, out_ids=out, out_q=oq)

        with contextlib.redirect_stdout(sys.stderr):
            graph = capture(candidates)
        report(
            graph,
            candidates,
            "sample_candidates",
            b,
            7,
            "temperature",
            b * (16 * 16 + 28),
            bw,
            name,
            op.workspace_layout(b, "candidates"),
        )
        for sparse, point in ((False, False), (True, False), (False, True)):
            for case in ("rejection", "bonus"):
                with contextlib.redirect_stdout(sys.stderr):
                    chain, call, _ = make_case(b, sparse, case, point=point)
                    graph = capture(call)
                g = chain.block_size - 1
                moved = verify_traffic(b, g, sparse, case, point=point)
                report(
                    graph,
                    call,
                    "verify_point_mass" if point else "verify_sparse" if sparse else "verify_dense",
                    b,
                    g,
                    case,
                    moved,
                    bw,
                    name,
                    chain.workspace_layout(b),
                )

        with contextlib.redirect_stdout(sys.stderr):
            chain, _, data = make_case(b, False, "bonus")
            _, q, x, temp, common, _, _ = data

            def proposals(chain=chain, q=q, temp=temp, common=common, x=x):
                chain.sample_dense_proposals(q, temp, **common, out_ids=x)

            graph = capture(proposals)
        chunks = (V + 4095) // 4096
        moved = b * 15 * (4 * (V + 4096) + chunks * 20 + 20)
        report(
            graph,
            proposals,
            "sample_dense_proposals",
            b,
            15,
            "temperature",
            moved,
            bw,
            name,
            chain.workspace_layout(b),
        )
    probability_measurements()


def probability_measurements():
    """Ideal traffic is one BF16 logits read plus one FP32 probabilities write."""
    if not hasattr(RequestLocalChainSampling, "probabilities"):
        return
    name = torch.cuda.get_device_name()
    bw = peak_bandwidth(name)
    length = 16
    seed = 42
    modes = (
        ("identity", 0.7, V, 1.0, "none"),
        ("top_k", 0.7, 50, 1.0, "top_k"),
        ("top_p", 0.7, V, 0.9, "top_p"),
        ("top_k+top_p", 0.7, 50, 0.9, "both"),
        ("top_k+top_p", 0.7, 50, 0.9, "auto"),
        ("production", 0.6, 20, 0.95, "both"),
        ("greedy", 0.0, V, 1.0, "none"),
    )
    for b in (1, 2, 3, 4, 8):
        op = RequestLocalChainSampling("cuda", length, V)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        logits = (
            torch.randn(b, length, V, device="cuda", generator=generator, dtype=torch.float32)
            .mul_(3)
            .to(torch.bfloat16)
        )
        out_probs = torch.empty((b, length, V), dtype=torch.float32, device="cuda")
        request_indices = torch.arange(b, dtype=torch.int32, device="cuda")
        workspace_bytes = op.workspace_size(b, "probabilities")
        workspace = torch.empty(max(4, workspace_bytes), dtype=torch.uint8, device="cuda")
        for case, temperature, k, p, hint in modes:
            temperatures = torch.full((b,), temperature, dtype=torch.float32, device="cuda")
            top_k = torch.full((b,), k, dtype=torch.int32, device="cuda")
            top_p = torch.full((b,), p, dtype=torch.float32, device="cuda")

            def call(
                op=op,
                logits=logits,
                temperatures=temperatures,
                top_k=top_k,
                top_p=top_p,
                request_indices=request_indices,
                workspace=workspace,
                out_probs=out_probs,
                hint=hint,
            ):
                op.probabilities(
                    logits,
                    temperatures,
                    top_k,
                    top_p,
                    request_indices=request_indices,
                    workspace=workspace,
                    out_probs=out_probs,
                    filters=hint,
                )

            with contextlib.redirect_stdout(sys.stderr):
                graph = capture(call)
                launches = count_launches(call)
            median = statistics.median(time_graph(graph) for _ in range(PAIRS))
            moved = logits.numel() * (logits.element_size() + out_probs.element_size())
            sol = moved / (bw * 1000) if bw else None
            emit(
                op="probabilities",
                provider="cute",
                batch=b,
                length=length,
                vocab=V,
                dtype="bfloat16",
                case=case,
                temperature=temperature,
                top_k=k,
                top_p=p,
                fixture_seed=seed,
                median_us=median,
                **launches,
                filters=hint,
                expected_kernel_launches=op.probability_launch_counts[hint],
                workspace_aligned_bytes=workspace_bytes,
                workspace_backing_bytes=workspace.numel(),
                ideal_bytes_estimate=moved,
                sol_us_estimate=sol,
                sol_fraction_estimate=sol / median if sol else None,
                peak_gbs=bw,
                device=name,
            )


def report(graph, call, op, b, gamma, case, moved, bw, device, workspace):
    with contextlib.redirect_stdout(sys.stderr):
        launches = count_launches(call)
    times = [time_graph(graph) for _ in range(PAIRS)]
    median = statistics.median(times)
    sol = moved / (bw * 1000) if bw else None
    emit(
        op=op,
        provider="cute",
        batch=b,
        gamma=gamma,
        vocab=V,
        case=case,
        median_us=median,
        **launches,
        expected_kernel_launches=RequestLocalChainSampling.launch_counts[op],
        workspace_payload_bytes=sum(r["payload_bytes"] for r in workspace["regions"]),
        workspace_aligned_bytes=workspace["total_bytes"],
        logical_bytes_estimate=moved,
        sol_us_estimate=sol,
        sol_fraction_estimate=sol / median if sol else None,
        peak_gbs=bw,
        device=device,
    )


def import_provenance(module, paths):
    """Identify the imported file without exposing an absolute directory."""
    filename = Path(module.__file__).resolve()
    roots = [(f"sglang-path[{i}]", path) for i, path in enumerate(paths)]
    roots += [(f"sys.path[{i}]", path) for i, path in enumerate(sys.path)]
    for source, root in roots:
        try:
            relative = filename.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return {
            "module": module.__name__,
            "module_file": relative.as_posix(),
            "relative_to": source,
        }
    return {
        "module": module.__name__,
        "module_file": filename.name,
        "relative_to": "basename-only",
    }


def load_reject_sampling(paths):
    """Load the pinned Triton file directly: it imports only triton, whereas the
    sglang package import closure pulls server-side dependencies."""
    relative = Path("sglang/kernels/ops/speculative/reject_sampling.py")
    for root in paths:
        candidate = Path(root) / relative
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("sglang_reject_sampling", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return importlib.import_module("sglang.kernels.ops.speculative.reject_sampling")


@functools.cache
def load_comparator(paths=()):
    """Import once in the harness interpreter; keep diagnostics off JSONL stdout."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module = load_reject_sampling(paths)
            chain = module.chain_speculative_sampling_triton
        label = "sglang-chain-triton"
    except (ImportError, AttributeError, OSError):
        traceback.print_exc(file=sys.stderr)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                module = importlib.import_module("flashinfer.sampling")
                chain = module.chain_speculative_sampling
            label = "flashinfer-chain-fallback"
        except (ImportError, AttributeError, OSError) as error:
            raise RuntimeError("neither chain provider imports: " + str(error)) from error
    print(
        json.dumps(
            {"event": "comparator_import", "comparator": label, **import_provenance(module, paths)},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return label, chain


def compare_outputs(ours, ours_counts, predicts, accept_index, counts):
    """Compare only emitted tokens, including correction/bonus, outside timing."""
    ours, ours_counts = ours.cpu(), ours_counts.cpu()
    predicts, accept_index, counts = predicts.cpu(), accept_index.cpu(), counts.cpu()
    positions = torch.arange(ours.shape[1])[None, :]
    visible = positions <= ours_counts[:, None]
    valid = (accept_index >= 0) & (accept_index < predicts.numel())
    # Clamp before indexing so a malformed comparator result reports a mismatch.
    tokens = predicts[accept_index.clamp(0, predicts.numel() - 1)]
    tokens_match = bool((valid & (positions <= counts[:, None]))[visible].all()) and torch.equal(
        tokens[visible], ours[visible]
    )
    counts_match = torch.equal(counts, ours_counts)
    rows = []
    for r in range(ours.shape[0]):
        n = int(ours_counts[r]) + 1
        mine = ours[r, :n].tolist()
        theirs = tokens[r, : int(counts[r]) + 1].tolist()
        if mine != theirs or int(counts[r]) != int(ours_counts[r]):
            rows.append(
                {
                    "row": r,
                    "ours": mine[-2:],
                    "theirs": theirs[-2:],
                    "counts": [n - 1, int(counts[r])],
                }
            )
    return {
        "mismatched_rows": rows[:8],
        "output_match": tokens_match and counts_match,
        "tokens_match": tokens_match,
        "counts_match": counts_match,
        "output_check": "fixed-uniform emitted prefix and accepted drafts",
    }


def comparator_call(b, case, data, paths=()):
    p, q, x, _, _, out, ours_counts = data
    g = x.shape[1]
    label, chain = load_comparator(paths)
    if label == "sglang-chain-triton":
        candidates = torch.zeros(b, g + 1, dtype=torch.int64, device="cuda")
        candidates[:, 1:] = x
        retrieve = torch.arange(b * (g + 1), dtype=torch.int64, device="cuda").reshape(b, g + 1)
        next_token = torch.arange(1, g + 2, dtype=torch.int64, device="cuda").repeat(b, 1)
        next_token[:, -1] = -1
        sibling = torch.full_like(retrieve, -1)
        predicts = torch.empty(b * (g + 1), dtype=torch.int32, device="cuda")
        accept_index = torch.empty_like(retrieve)
        counts = torch.empty(b, dtype=torch.int32, device="cuda")
        padded_q = torch.zeros_like(p)
        padded_q[:, :g] = q
        coins = torch.tensor(
            [[uniform(42, 2**32 + r, ACCEPT, i) for i in range(g + 1)] for r in range(b)],
            dtype=torch.float32,
            device="cuda",
        )
        final = torch.tensor(
            [
                uniform(42, 2**32 + r, CORRECTION if case == "rejection" else BONUS, 0)
                for r in range(b)
            ],
            dtype=torch.float32,
            device="cuda",
        )

        def call():
            chain(
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
                padded_q,
                1.0,
                1.0,
                True,
                block_verification=False,
            )

        def check():
            return compare_outputs(out, ours_counts, predicts, accept_index, counts)

        return label, call, check
    accepted = torch.zeros(b, dtype=torch.int32, device="cuda")
    emitted = torch.zeros_like(accepted)

    def call():
        # FlashInfer counters accumulate; reset is included in its measured graph.
        accepted.zero_()
        emitted.zero_()
        chain(q, x, p, accepted, emitted, deterministic=True, seed=42, offset=0)

    def check():
        return {
            "output_match": None,
            "tokens_match": None,
            "counts_match": None,
            "output_check": "not comparable: FlashInfer uses provider RNG, not fixed uniforms",
        }

    return label, call, check


def paired_worker(b, paths=(), *, point=False):
    if not torch.cuda.is_available():
        emit(comparator="N/A", reason="CUDA unavailable", batch=b)
        return
    operation = "verify_point_mass" if point else "verify_dense"
    for case in ("rejection", "bonus"):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                _, ours, data = make_case(b, False, case, point=point)
                label, theirs, check = comparator_call(b, case, data, paths)
                if point and label != "sglang-chain-triton":
                    raise RuntimeError("point-mass comparison requires SGLang fixed uniforms")
                left, right = capture(ours), capture(theirs)
                ours_launches, theirs_launches = count_launches(ours), count_launches(theirs)
                left.replay()
                right.replay()
                matches = check()
        except (ImportError, RuntimeError, OSError, AttributeError, TypeError) as error:
            traceback.print_exc(file=sys.stderr)
            emit(
                comparator="N/A",
                op=operation,
                reason=type(error).__name__,
                detail=str(error)[:500],
                batch=b,
                case=case,
            )
            continue
        if point and not matches["output_match"]:
            # Report, do not drop: adjacent-token differences can come from FP32
            # CDF summation order (chunked r2.1 versus a sequential cumsum).
            emit(record_type="output_mismatch", op=operation, batch=b, case=case, **matches)
        # Freeze comparator noise before candidate timings (nearest-rank p95).
        comparator_self_pairs = [(time_graph(right), time_graph(right)) for _ in range(PAIRS)]
        epsilon = sorted(max(a / b, b / a) - 1 for a, b in comparator_self_pairs)[18]
        ratios = []
        for pair in range(PAIRS):
            if pair % 2:
                theirs_us, ours_us = time_graph(right), time_graph(left)
            else:
                ours_us, theirs_us = time_graph(left), time_graph(right)
            self_a, self_b = time_graph(left), time_graph(left)
            their_self_a, their_self_b = comparator_self_pairs[pair]
            emit(
                comparator=label,
                op=operation,
                ours_launches=ours_launches,
                theirs_launches=theirs_launches,
                requested_sglang_revision="72730d6",
                revision_verification="in-process import provenance; revision not verified",
                **matches,
                batch=b,
                gamma=15,
                vocab=V,
                case=case,
                pair=pair,
                ours_us=ours_us,
                theirs_us=theirs_us,
                ratio=theirs_us / ours_us,
                their_self_a_us=their_self_a,
                their_self_b_us=their_self_b,
                their_epsilon_us=abs(their_self_a - their_self_b),
                self_a_us=self_a,
                self_b_us=self_b,
                epsilon_us=abs(self_a - self_b),
                epsilon_fraction=max(self_a / self_b, self_b / self_a) - 1,
                comparator_epsilon=epsilon,
            )
            ratios.append(theirs_us / ours_us)
        summary = paired_summary(ratios, comparator_self_pairs)
        emit(
            record_type="paired_summary",
            op=operation,
            **matches,
            contract="K1.22",
            comparator=label,
            requested_sglang_revision="72730d6",
            revision_verification="in-process import provenance; revision not verified",
            batch=b,
            gamma=15,
            vocab=V,
            case=case,
            pairs=PAIRS,
            ratio_definition="comparator_time / K1_time",
            geometric_mean_ratio=summary["gm"],
            min_ratio=min(ratios),
            max_ratio=max(ratios),
            ci_lower=summary["lo"],
            ci_upper=summary["hi"],
            ci_method="paired log-ratio t interval; sample sd (n-1)",
            ci_confidence=0.95,
            ci_t_critical=2.093024,
            epsilon=epsilon,
            speedup_claim=summary["claim"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sglang-path",
        action="append",
        default=[],
        metavar="DIR",
        help="append an import directory for pinned SGLang 72730d6 or FlashInfer (repeatable)",
    )
    parser.add_argument("--sglang-python", help="deprecated; errors with migration instructions")
    args = parser.parse_args()
    if args.sglang_python is not None:
        parser.error(
            "--sglang-python is deprecated: comparators now run in the current interpreter; "
            "use --sglang-path DIR (repeatable) to preserve the harness's CUTLASS activation"
        )
    paths = tuple(str(Path(path).resolve()) for path in args.sglang_path)
    if any(not Path(path).is_dir() for path in paths):
        parser.error("each --sglang-path must name an existing import directory")
    sys.path.extend(paths)
    emit(
        op="probabilities",
        comparator="N/A",
        reason="no equivalent deterministic truncation path in SGLang 72730d6 "
        "(FlashInfer top-k/top-p renorm uses different tie/threshold semantics, "
        "so not the same q law)",
    )
    if not torch.cuda.is_available():
        emit(status="N/A", reason="CUDA unavailable")
        return
    measurements()
    if paths:
        for b in (1, 2, 3, 4, 8):
            paired_worker(b, paths)
            paired_worker(b, paths, point=True)


if __name__ == "__main__":
    main()
