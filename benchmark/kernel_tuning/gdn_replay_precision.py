# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen r2.1 FP64 judge. JSONL stdout; runtime limits never imply qualification.

Default runs the complete requested matrix until --seconds expires. Select L/Q
and cases to shard receipts across jobs. --smoke retains the inequalities and all
20 self-pairs but shortens schedules; it can never qualify the frozen matrix.
"""

import argparse
import hashlib
import itertools
import math

import torch

from benchmark.kernel_tuning.gdn_replay_common import (
    Budget,
    BudgetExpired,
    Oracle,
    Product,
    Replay,
    emit,
    manifest,
)
from benchmark.kernel_tuning.gdn_replay_inputs import CLASSES, SEEDS, SHAPES, case_inputs

METRICS = ("rms", "max")


def ratio(a, b):
    """Exact zero rule, also used for self-pair calibration."""
    return torch.where(b > 0, a / b, torch.where(a == 0, 1.0, float("inf")))


def errors(actual, truth, *, check_finite=True):
    """Per batch row; caller separately identifies phase and observation position.

    Truth's LAST axis is V for output, K for state. Never flatten before ULP.
    """
    a, t = actual.double().reshape_as(truth), truth.double()
    if check_finite and not torch.isfinite(t).all():
        raise ValueError("nonfinite FP64 truth invalidates case")
    d = (a - t).abs()
    axis_rms = t.square().mean(-1, keepdim=True).sqrt()
    scale = torch.maximum(t.abs(), axis_rms / 8)
    exponent = torch.frexp(scale)[1]
    ulp = torch.pow(2.0, (exponent - 8).double()).clamp_min(2.0**-133)
    scaled = torch.where(d <= 2.0**-126, 0.0, d / ulp)
    dims = tuple(range(1, a.ndim))
    values = torch.stack((d.square().mean(dims).sqrt(), d.amax(dims), scaled.amax(dims)), -1)
    finite = (torch.isfinite(a) & torch.isfinite(t)).flatten(1).all(-1)
    return torch.where(finite[:, None], values, float("inf"))


def calibration_errors(actual, truth):
    """Batch independent replicas' FP64 metrics; calibration never uses ULP.

    Bound temporary storage to four states, even for B=4 and Hv=48. Keeping
    replica and batch axes separate preserves the frozen per-row inequalities.
    """
    rows = []
    for start in range(0, len(actual), 4):
        a = torch.stack(actual[start : start + 4]).double().reshape(-1, *truth.shape)
        d = a - truth.double()
        dims = tuple(range(2, d.ndim))
        finite = (torch.isfinite(a) & torch.isfinite(truth)).flatten(2).all(-1)
        rms, maximum = d.square().mean(dims).sqrt(), d.abs().amax(dims)
        values = torch.stack((rms, maximum, torch.zeros_like(rms)), -1)
        rows.append(torch.where(finite[..., None], values, float("inf")))
    return torch.cat(rows).reshape(20, 2, truth.shape[0], 3)


def calibrated(candidate, product, pairs):
    """pairs [20,2,B,3]; nearest rank 19/20, no invented noise floor."""
    a, b = pairs[:, 0, :, :2], pairs[:, 1, :, :2]
    invalid = ((a == 0) != (b == 0)).any(0) | (~torch.isfinite(a) | ~torch.isfinite(b)).any(0)
    eta = (ratio(a, b) - 1).abs().sort(dim=0).values[18]
    c, p = candidate[:, :2], product[:, :2]
    passed = (c <= p * (1 + eta)) & ~invalid & torch.isfinite(c) & torch.isfinite(p)
    return ratio(c, p), eta, passed, invalid


class Score:
    def __init__(self, kind):
        self.kind = kind
        self.worst = {}
        self.passed = True
        self.observations = 0
        self.pending = []

    def add(self, phase, position, candidate, baseline, pairs):
        rr, eta, ok, invalid = calibrated(candidate, baseline, pairs)
        payload = torch.stack(
            (
                rr,
                eta,
                ok.double(),
                invalid.double(),
                candidate[:, :2],
                baseline[:, :2],
                candidate[:, 2, None].expand_as(rr),
            ),
            -1,
        )
        self.pending.append((phase, position, payload))
        self.observations += candidate.shape[0]

    def drain(self):
        """One small D2H transfer per block, never one transfer per token/metric."""
        if not self.pending:
            return
        payloads = torch.stack([entry[2] for entry in self.pending]).cpu().tolist()
        for (phase, position, _), payload in zip(self.pending, payloads, strict=True):
            self._add_host(phase, position, payload)
        self.pending.clear()

    def _add_host(self, phase, position, payload):
        guard = 1 if self.kind == "ordinary" else 4
        for row in range(len(payload)):
            for i, metric in enumerate(METRICS):
                r, e, passed, bad, c, b, ulp = payload[row][i]
                self.passed &= bool(passed) and ulp <= guard
                key = phase.split("/")[0] + "_" + metric
                old = self.worst.get(key)
                if old is None or not math.isfinite(r) or r > old["ratio"]:
                    self.worst[key] = {
                        "ratio": r,
                        "eta": e,
                        "candidate_error": c,
                        "product_error": b,
                        "tuple": [phase, position, row],
                        "passed": bool(passed),
                        "invalid_calibration": bool(bad),
                        "max_ulp": ulp,
                    }


def projections(hk, hv, b, length, kind, seed, stream, fingerprint=None):
    # Each segment has its own frozen seed. Selecting other cases cannot change it.
    derived = seed + 104729 * stream
    cpu = case_inputs(hk, hv, b, b, length, kind, derived, "cpu")[1]
    if fingerprint is not None:
        fingerprint.update(cpu.contiguous().view(torch.uint8).numpy().tobytes())
    return cpu.cuda()


def run_schedule(
    hk,
    hv,
    batch,
    seed,
    kind,
    window,
    q,
    providers,
    budget,
    *,
    smoke=False,
    initial_position=0,
    chunked=False,
):
    weights, prefill, initial, history = case_inputs(hk, hv, batch, batch, 129, kind, seed, "cpu")
    fingerprint = hashlib.sha256()
    for value in (*weights.args(), prefill, initial, history):
        fingerprint.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    weights = type(weights)(*(v.cuda() for v in weights.args()))
    prefill, initial, history = prefill.cuda(), initial.cuda(), history.cuda()
    weights = weights.args()
    ref = Oracle(hk, weights, initial, history, graph=True)
    for t, x in enumerate(prefill):
        ref.step(x)
        if chunked and t + 1 in (1, 3, 6, 69):
            ref.state = ref.state.float().double()
    initial, history = ref.canonical().float().contiguous(), ref.history.clone()
    prefill_error = errors(initial, ref.canonical()).cpu().tolist()
    # All contenders, including the isolated FP64 judge, receive identical FP32 handoff.
    ref = Oracle(hk, weights, initial, history, graph=True)
    pos = torch.full((batch,), initial_position + 129, dtype=torch.int64, device="cuda")
    base = Product(hk, hv, weights, initial, history)
    # Independent storage and product launch policy for every self-pair, same B.
    calibration = [Product(hk, hv, weights, initial, history) for _ in range(40)]
    candidates = {
        # cute_strict: the CuTe provider with checks="strict" (guarded routes only).
        p: Replay(
            "cute" if p == "cute_strict" else p,
            hk,
            hv,
            batch,
            window,
            q,
            weights,
            initial,
            history,
            pos,
            checks="strict" if p == "cute_strict" else "trusted",
        )
        for p in providers
        if p != "product"
    }
    scores = {p: Score(kind) for p in providers}
    exact_checks = True
    budget.partial_scores = scores
    budget.partial_coverage = {
        "L": window,
        "Q": q,
        "initial_position": initial_position,
        "chunked": chunked,
        "complete": False,
    }
    obs = 0
    exact_pending = []

    def drain():
        nonlocal exact_checks
        for score in scores.values():
            score.drain()
        if exact_pending:
            exact_checks &= bool(torch.stack(exact_pending).all().cpu())
            exact_pending.clear()
        for r in candidates.values():
            r.check()

    def observe(label, truth_y, base_y, cal_y, ys, *, state=True):
        nonlocal obs, exact_checks
        if obs % 32 == 0:
            drain()
        try:
            budget.check()
        except BudgetExpired:
            drain()
            raise
        obs += 1
        budget.partial_coverage["observations"] = obs
        observations = [("output", truth_y, base_y, cal_y, ys)]
        if state:
            observations.append(
                (
                    "state",
                    ref.canonical(),
                    base.state.index_select(0, base.slots.long()),
                    [r.state.index_select(0, r.slots.long()) for r in calibration],
                    {p: r.materialize().cuda() for p, r in candidates.items()},
                )
            )
        for kind_of, truth, product, self_values, values in observations:
            exact = truth.reshape(batch, hv, 128) if kind_of == "output" else truth
            be = errors(product, exact, check_finite=False)
            ce = calibration_errors(self_values, exact)
            for p, score in scores.items():
                got = be if p == "product" else errors(values[p].cuda(), exact, check_finite=False)
                score.add(kind_of + "/" + label, obs, got, be, ce)
        if state:
            for r in candidates.values():
                meta = r.state.metadata[r.slots.long()]
                checks = torch.stack(
                    (
                        (meta[:, 1] == r.pos).all(),
                        (meta[:, 2] == r.expected_cursor).all(),
                        (meta[:, 3] == r.expected_n).all(),
                        (meta[:, 4:6] == 1).all(),
                        ((meta[:, 1] - meta[:, 0]) == meta[:, 3]).all(),
                        ((meta[:, 3] >= 0) & (meta[:, 3] < window)).all(),
                        (r.state.conv_history[r.slots.long()] == ref.history.to(r.device)).all(),
                    )
                )
                exact_pending.append(checks.all().cuda())

    def decode(x, label):
        truth = ref.step(x)
        by = base.step(x)
        cy = [r.step(x) for r in calibration]
        ys = {p: r.step(x) for p, r in candidates.items()}
        observe(label, truth, by, cy, ys)

    # Stagger each row by (5*r)%L using inactive rows; the frozen stream is per request.
    offsets = [(5 * r) % window for r in range(batch)]
    for t in range(max(offsets, default=0)):
        mask = torch.tensor([t < n for n in offsets], device="cuda")
        x = projections(hk, hv, batch, 1, kind, seed, 1 + t, fingerprint)[0]
        saved, saved_h = ref.state.clone(), ref.history.clone()
        truth = ref.step(x)
        ref.state = torch.where(mask[:, None, None, None], ref.state, saved)
        ref.history = torch.where(mask[:, None, None], ref.history, saved_h)
        truth = torch.where(mask[:, None, None], truth, 0.0)
        for r in [base, *calibration, *candidates.values()]:
            r.slots.copy_(torch.where(mask, torch.arange(batch, device="cuda"), -1).to(r.slots))
        by = base.step(x)
        cy = [r.step(x) for r in calibration]
        ys = {p: r.step(x) for p, r in candidates.items()}
        for r in candidates.values():
            r.pos.sub_((~mask).to(device=r.pos.device, dtype=r.pos.dtype))
        # Padding outputs are checked here; state checks resume with all slots active.
        observe("stagger", truth, by, cy, ys, state=False)
    for r in [base, *calibration, *candidates.values()]:
        r.slots.copy_(torch.arange(batch, dtype=torch.int32, device=r.slots.device))
    ordinary = 3 * window + 1 if smoke else 4096
    xs = projections(hk, hv, batch, ordinary, kind, seed, 100, fingerprint)
    for t, x in enumerate(xs):
        decode(x, f"decode:{t}")
        if t + 1 in (257, 1025, 3073):
            for r in candidates.values():
                r.flush()
    cycles = q + 1 if smoke else 256
    for j in range(cycles):
        # Engine-side physical slot permutation preserves each logical request stream.
        if j and batch > 1:
            new_slots = (torch.arange(batch, device="cuda") + j) % batch
            for r in [base, *calibration, *candidates.values()]:
                old = r.slots.long()
                new = new_slots.to(r.slots.device).long()
                arrays = (
                    [r.state, r.history]
                    if isinstance(r, Product)
                    else [r.state.checkpoint, r.state.conv_history, r.state.ring, r.state.metadata]
                )
                for array in arrays:
                    array.index_copy_(0, new, array.index_select(0, old))
                r.slots.copy_(new.to(torch.int32))
        xs = projections(hk, hv, batch, 2 * window, kind, seed, 1000 + 3 * j, fingerprint)
        for t in range(window + 1):
            decode(xs[t], f"cycle:{j}:before:{t}")
        for reject in (False, True):
            drafts = (
                projections(hk, hv, batch, q, kind, seed, 1001 + 3 * j + int(reject), fingerprint)
                .transpose(0, 1)
                .contiguous()
            )
            accepted_host = [0 if reject else (j + r) % (q + 1) for r in range(batch)]
            accepted = torch.tensor(
                accepted_host,
                dtype=torch.int32,
                device="cuda",
            )
            truth = ref.verify(drafts, accepted)
            by = base.verify(drafts, accepted)
            cy = [r.verify(drafts, accepted) for r in calibration]
            ys = {p: r.verify(drafts) for p, r in candidates.items()}
            for t in range(q):
                observe(
                    f"verify:{j}:{reject}:{t}",
                    truth[:, t],
                    by[:, t],
                    [y[:, t] for y in cy],
                    {p: y[:, t] for p, y in ys.items()},
                    state=False,
                )
            for r in candidates.values():
                offsets = torch.tensor(
                    [j % (m + 1) for m in accepted_host], dtype=torch.int64, device="cuda"
                )
                r.resolve(accepted, reject, offsets=None if reject else offsets)
            # A distinct committed-state observation after acceptance/reject.
            observe(
                f"resolve:{j}:{reject}",
                truth[:, -1],
                by[:, -1],
                [y[:, -1] for y in cy],
                {p: y[:, -1] for p, y in ys.items()},
            )
        for t in range(window - 1):
            decode(xs[window + 1 + t], f"cycle:{j}:after:{t}")
        if j % 4 == 3:
            for r in candidates.values():
                r.flush()
    drain()
    return scores, {
        "exact_checks": exact_checks,
        "observations": obs,
        "prefill_error": prefill_error,
        "full_schedule": not smoke,
        "L": window,
        "Q": q,
        "input_sha256": fingerprint.hexdigest(),
        "initial_position": initial_position,
        "chunked_reference_handoff": chunked,
        "product_launch_config": base.launch_config(),
        "missing_coverage": ["actual-prefill-provider-end-to-end", "whole-model"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--providers",
        nargs="+",
        choices=("torch_reference", "triton", "cute", "cute_strict", "product"),
        default=["torch_reference", "triton", "cute", "product"],
    )
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    p.add_argument("--shapes", nargs="+", choices=["16,32", "16,48"], default=["16,32", "16,48"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    p.add_argument("--windows", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--queries", nargs="+", type=int, default=[2, 8, 16, 32])
    p.add_argument("--positions", nargs="+", type=int, default=[0, 2**33 + 7])
    p.add_argument(
        "--handoffs", nargs="+", choices=["fresh", "chunked"], default=["fresh", "chunked"]
    )
    p.add_argument("--seconds", type=float, default=840)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    budget = Budget(args.seconds)
    if not torch.cuda.is_available():
        emit({"type": "summary", "qualified": False, "reason": "CUDA not available"})
        return 2
    torch.set_num_threads(1)
    info = manifest()
    worst, completed, truncated = {}, 0, False
    selected_passed = dict.fromkeys(args.providers, True)
    for shape, b, seed, kind in itertools.product(
        args.shapes, args.batches, args.seeds, args.classes
    ):
        hk, hv = map(int, shape.split(","))
        rows = {
            p: {
                "provider": p,
                "shape": [hk, hv],
                "B": b,
                "seed": seed,
                "input_class": kind,
                "passed": True,
                "configurations": [],
                "worst": {},
            }
            for p in args.providers
        }
        try:
            for window, q, start, handoff in itertools.product(
                args.windows, args.queries, args.positions, args.handoffs
            ):
                budget.partial_scores = None
                budget.check()
                scores, coverage = run_schedule(
                    hk,
                    hv,
                    b,
                    seed,
                    kind,
                    window,
                    q,
                    args.providers,
                    budget,
                    smoke=args.smoke,
                    initial_position=start,
                    chunked=handoff == "chunked",
                )
                for provider, score in scores.items():
                    row = rows[provider]
                    row["passed"] &= score.passed and coverage["exact_checks"]
                    row["configurations"].append(coverage)
                    for metric, value in score.worst.items():
                        measured = dict(
                            value, L=window, Q=q, initial_position=start, handoff=handoff
                        )
                        if measured["ratio"] > row["worst"].get(metric, {}).get("ratio", -1):
                            row["worst"][metric] = measured
        except BudgetExpired:
            truncated = True
            for provider, score in (getattr(budget, "partial_scores", None) or {}).items():
                rows[provider]["passed"] = False
                rows[provider]["configurations"].append(budget.partial_coverage)
                for metric, value in score.worst.items():
                    measured = dict(value, **budget.partial_coverage)
                    if measured["ratio"] > rows[provider]["worst"].get(metric, {}).get("ratio", -1):
                        rows[provider]["worst"][metric] = measured
        for provider, row in rows.items():
            selected_passed[provider] &= (
                row["passed"] and not truncated and info["baseline_pin_matches"]
            )
            row.update(
                kernel_gate_passed=bool(
                    row["passed"] and not truncated and info["baseline_pin_matches"]
                ),
                execution_mode="eager",
                type="precision",
                complete=not truncated,
                qualified=False,
                manifest=info,
                reason="kernel screening; full contract coverage still required",
            )
            emit(row)
            for metric, value in row["worst"].items():
                key = provider + "/" + metric
                if value["ratio"] > worst.get(key, {}).get("ratio", -1):
                    worst[key] = dict(value, case=[hk, hv, b, seed, kind])
        if truncated:
            break
        completed += 1
    emit(
        {
            "type": "summary",
            "qualified": False,
            "completed_cases": completed,
            "truncated": truncated,
            "smoke": args.smoke,
            "worst_unrounded_ratios": worst,
            "selected_cases_passed": selected_passed,
            "full_matrix_requested": (
                set(args.shapes) == {f"{hk},{hv}" for hk, hv in SHAPES}
                and set(args.batches) == {1, 4}
                and set(args.seeds) == set(SEEDS)
                and set(args.classes) == set(CLASSES)
                and set(args.windows) == {4, 8, 16}
                and set(args.queries) == {2, 8, 16, 32}
                and set(args.positions) == {0, 2**33 + 7}
                and set(args.handoffs) == {"fresh", "chunked"}
                and not args.smoke
            ),
        }
    )
    return 2 if truncated else 0 if all(selected_passed.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
