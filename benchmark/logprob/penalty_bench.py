# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DN4 warm-L2 CUDA-graph benchmark, including full table counting for ours.

Semantics follow vLLM 0.31.0 V2 and SGLang penaltylib; no code copied.
Market state maintenance is excluded. The opt-in SGLang callable is a cost
model, not SGLang's real dispatch (state copies at Q=1, eager repetition), so it
is excluded from the default providers and from the market denominator.
"""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.logprob.top_bench import capture
from ctikernels.sampling import LogitPenalties

PROVIDERS = ("ours", "ours_fp32_inplace", "vllm_v2", "sglang_penaltylib", "engine_torch")
DEFAULT_PROVIDERS = ("ours", "ours_fp32_inplace", "vllm_v2", "engine_torch")
MARKET = ("vllm_v2",)


def cases(table_width=32768, wide=False):
    for vocab in (151936, 248320):
        for b, q in (
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (1, 4),
            (1, 16),
            (2, 16),
            (4, 16),
            (4, 8),
            (1, 64),
        ):
            yield vocab, b, q, table_width
        if wide and table_width != 262144:
            for b in (1, 2, 3, 4):
                yield vocab, b, 1, 262144


def summarize(samples):
    medians = {name: statistics.median(times) for name, times in samples.items()}
    result = {"median_us": medians}
    market = [medians[name] for name in MARKET if name in medians]
    if "ours" in medians and market:
        result["ours_over_fastest_market"] = medians["ours"] / min(market)
    return result


def parity(ours, market):
    """Ordered IEEE bit distance; count differences, including signed zero."""
    a, b = (value.contiguous().view(torch.int32).long() for value in (ours, market))
    ordered_a = torch.where(a < 0, -2147483648 - a, a)
    ordered_b = torch.where(b < 0, -2147483648 - b, b)
    return {
        "max_ulp_difference": (ordered_a - ordered_b).abs().max().item(),
        "differing_elements": (a != b).sum().item(),
    }


def load_vllm(root):
    from benchmark.dflash_round.comparators import load_definitions

    return load_definitions(
        Path(root) / "vllm/v1/worker/gpu/sample/penalties.py",
        ("_penalties_kernel", "apply_penalties"),
        module_name="_dn4_vllm_v2_penalties",
        globals={"cdiv": lambda a, b: (a + b - 1) // b},
    )


def prepare_state(table, rows, vocab, drafts):
    """Prepare comparator state outside timing; CPU reads are intentional here."""
    b, q = rows.rows.numel(), drafts.shape[1] + 1
    prompt, output = [], []
    cpu_table = table.cpu()
    for n in range(b):
        slot, p, c = int(rows.rows[n]), int(rows.prompt_lengths[n]), int(rows.lengths[n])
        for target, tokens in ((prompt, cpu_table[slot, :p]), (output, cpu_table[slot, p:c])):
            ids = tokens.long()
            target.append(torch.bincount(ids[(ids >= 0) & (ids < vocab)], minlength=vocab))
    prompt = torch.stack(prompt).to(table.device)
    output = torch.stack(output).to(device=table.device, dtype=torch.int32)
    packed = torch.nn.functional.pad((prompt > 0).long(), (0, (-vocab) % 32)).reshape(b, -1, 32)
    shifts = torch.arange(32, device=table.device)
    packed = (packed << shifts).sum(-1).int()
    token_ids = torch.zeros((b, q), dtype=torch.int32, device=table.device)
    token_ids[:, 1:] = drafts
    # Position zero is the already-counted last input; V2 reads drafts at i+1.
    for n in range(b):
        length = int(rows.lengths[n])
        if length:
            token_ids[n, 0] = table[int(rows.rows[n]), length - 1]
    return SimpleNamespace(
        prompt_bin_mask=packed,
        output_bin_counts=output,
        expanded_idx_mapping=torch.arange(
            b, device=table.device, dtype=torch.int32
        ).repeat_interleave(q),
        expanded_local_pos=torch.arange(q, device=table.device, dtype=torch.int32).repeat(b),
        token_ids=token_ids.reshape(-1),
        frequency=output.float() * rows.frequency[:, None],
        presence=(output > 0).float() * rows.presence[:, None],
        repetition=torch.where(output > 0, rows.repetition[:, None], 1.0),
    )


def sglang_penaltylib(logits, state, q):
    """Prepared frequency/presence subtractions, then output-only repetition.

    penaltylib broadcasts base state to verify rows; no preceding drafts are
    counted. This models penaltylib apply, not DFlash verify's dropped repetition.
    """
    frequency, presence, repetition = (
        getattr(state, name).repeat_interleave(q, dim=0)
        for name in ("frequency", "presence", "repetition")
    )
    logits.sub_(frequency)
    logits.sub_(presence)
    logits.copy_(torch.where(logits > 0, logits / repetition, logits * repetition))
    return logits


def engine_torch(logits, table, rows, lookahead=None):
    """Semantic reimplementation of the engine's PenaltyRows torch path.

    Includes table counting and dense cumulative draft counts; positive seen
    logits use true division. Like that path, invalid IDs clamp to vocabulary
    edges, rather than being ignored. Returns FP32 [N,Q,V]. No code imported.
    """
    logits = logits.float()
    n, _, vocab = logits.shape
    tokens = table[rows.rows].long().clamp_(0, vocab - 1)
    positions = torch.arange(table.shape[1], device=table.device)
    context = positions[None, :] < rows.lengths[:, None]
    output = context & (positions[None, :] >= rows.prompt_lengths[:, None])
    counts = torch.zeros((n, vocab), device=logits.device, dtype=torch.float32)
    seen = torch.zeros_like(counts)
    counts.scatter_add_(1, tokens, output.float())
    seen.scatter_add_(1, tokens, context.float())
    counts, seen = counts[:, None], seen[:, None]
    if lookahead is not None and lookahead.shape[1]:
        ahead = torch.zeros_like(logits)
        ahead[:, 1:].scatter_(2, lookahead.long().clamp(0, vocab - 1)[..., None], 1.0)
        ahead = ahead.cumsum(1)
        counts, seen = counts + ahead, seen + ahead
    repetition = rows.repetition[:, None, None]
    penalized = torch.where(logits > 0, logits / repetition, logits * repetition)
    logits = torch.where(seen > 0, penalized, logits)
    logits = logits - rows.frequency[:, None, None] * counts
    return logits - rows.presence[:, None, None] * (counts > 0)


def output_parity(outputs):
    return {
        f"parity_{name}": parity(outputs["ours"], outputs[name])
        if {"ours", name} <= outputs.keys()
        else None
        for name in ("vllm_v2", "engine_torch")
    }


def measure_interleaved(callables, replays):
    graphs, outputs = {}, {}
    for name, call in callables.items():
        graphs[name], outputs[name] = capture(call)
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(replays)
        ]
        for name in graphs
    }
    for replay in range(replays):
        for name, graph in graphs.items():
            start, end = events[name][replay]
            start.record()
            graph.replay()
            end.record()
    torch.cuda.synchronize()
    samples = {
        name: [start.elapsed_time(end) * 1000 / 20 for start, end in pairs]
        for name, pairs in events.items()
    }
    return summarize(samples), output_parity(outputs)


def bench_row(vllm, vocab, b, q, table_width, replays, providers=DEFAULT_PROVIDERS):
    device = torch.device("cuda", torch.cuda.current_device())
    table = torch.randint(0, vocab, (b, table_width), dtype=torch.int32, device=device)
    rows = SimpleNamespace(
        rows=torch.arange(b, dtype=torch.int32, device=device),
        prompt_lengths=torch.full((b,), 2048, dtype=torch.int32, device=device),
        lengths=torch.full((b,), 2560, dtype=torch.int32, device=device),
        presence=torch.full((b,), 0.5, device=device),
        frequency=torch.full((b,), 0.3, device=device),
        repetition=torch.full((b,), 1.2, device=device),
    )
    source = torch.randn((b, q, vocab), dtype=torch.bfloat16, device=device)
    fp32 = source.float()
    drafts = torch.randint(0, vocab, (b, q - 1), dtype=torch.int32, device=device)
    state = prepare_state(table, rows, vocab, drafts)
    op = LogitPenalties(table, vocab)
    out, inplace, v2_out, sgl_out = (torch.empty_like(fp32) for _ in range(4))

    def ours():
        op.apply(source, rows, drafts, out=out)
        return out.reshape(b * q, vocab)

    def ours_inplace():
        inplace.copy_(fp32)
        op.apply(inplace, rows, drafts, out=inplace)
        return inplace.reshape(b * q, vocab)

    def v2():
        v2_out.copy_(fp32)
        flat = v2_out.reshape(b * q, vocab)
        vllm.apply_penalties(
            flat,
            state.expanded_idx_mapping,
            state.token_ids,
            state.expanded_local_pos,
            rows.repetition,
            rows.frequency,
            rows.presence,
            state.prompt_bin_mask,
            state.output_bin_counts,
        )
        return flat

    def sgl():
        sgl_out.copy_(fp32)
        return sglang_penaltylib(sgl_out.reshape(b * q, vocab), state, q)

    available = {
        "ours": ours,
        "ours_fp32_inplace": ours_inplace,
        "vllm_v2": v2,
        "sglang_penaltylib": sgl,
        "engine_torch": lambda: engine_torch(source, table, rows, drafts).reshape(b * q, vocab),
    }
    measured, check = measure_interleaved({name: available[name] for name in providers}, replays)
    return dict(
        device=torch.cuda.get_device_name(device),
        V=vocab,
        B=b,
        Q=q,
        R=b * q,
        dtype="bfloat16",
        table_width=table_width,
        prompt_length=2048,
        output_length=512,
        presence=0.5,
        frequency=0.3,
        repetition=1.2,
        provider=op.provider,
        executions_per_graph=20,
        replays=replays,
        l2="warm",
        market_state_maintenance="excluded; prepared prompt bitmask/output counts or dense penalty tensors",
        inplace_reset="FP32 source copy included for ours_fp32_inplace and both market providers on every execution",
        engine_torch_scope="full table counting, BF16-to-FP32 and dense cumulative drafts; true division",
        **check,
        **measured,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vllm-path", type=Path, help="Trusted vLLM 0.31.0 bundle root")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=DEFAULT_PROVIDERS)
    parser.add_argument("--table-width", type=int, default=32768)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.table_width < 2560 or args.replays < 30 or (args.limit is not None and args.limit < 1):
        parser.error("require table-width >= 2560, replays >= 30, limit >= 1")
    if "vllm_v2" in args.providers and args.vllm_path is None:
        parser.error("--vllm-path is required when timing vllm_v2")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no CPU timing")
    vllm = load_vllm(args.vllm_path) if "vllm_v2" in args.providers else None
    torch.manual_seed(1701)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as output:
        for index, case in enumerate(cases(args.table_width, args.wide)):
            if args.limit is not None and index >= args.limit:
                break
            row = bench_row(vllm, *case, args.replays, args.providers)
            line = json.dumps(row, sort_keys=True, allow_nan=False)
            times = " ".join(f"{name}={us:.1f}" for name, us in row["median_us"].items())
            checks = " ".join(
                f"{name}="
                + (
                    "n/a"
                    if row[name] is None
                    else f"{row[name]['differing_elements']}diff/{row[name]['max_ulp_difference']}ulp"
                )
                for name in ("parity_vllm_v2", "parity_engine_torch")
            )
            print(
                f"V={row['V']} B={row['B']} Q={row['Q']} L={row['table_width']}: {times} us {checks}",
                flush=True,
            )
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
