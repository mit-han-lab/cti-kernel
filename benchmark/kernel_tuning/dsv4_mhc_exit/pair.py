# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Two ranks on two nodes: AR, mHC, and their composition; CPU store coordination."""

import ctypes
import hashlib
import os
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist

from .bench import (
    arguments,
    block_calls,
    check,
    count_kernels,
    inputs,
    make_launcher,
    measure,
    prepare_provider,
    print_row,
    save_report,
    setup_row,
    validate,
)
from .market import failure_reason
from .reference import MhcConfig, output_dict


def digest(result):
    """Hash every output's dtype, shape, and exact bytes, including FP32 coefficients."""
    values = {"ar": result} if isinstance(result, torch.Tensor) else output_dict(result)
    hasher = hashlib.sha256()
    for key, value in sorted(values.items()):
        cpu_value = value.detach().contiguous().cpu()
        hasher.update(f"{key}:{cpu_value.dtype}:{tuple(cpu_value.shape)}".encode())
        # NumPy-free (the CPU CI env has none): the contiguous CPU bytes directly.
        hasher.update(
            ctypes.string_at(cpu_value.data_ptr(), cpu_value.numel() * cpu_value.element_size())
        )
    return hasher.hexdigest()


def rank_check(session, key, result):
    local = digest(result)
    same = session.exchange(key, local) == local
    session.agree(key + "/agree", same)
    return True


def ar_function(kind, op, nccl, state, rank):
    partial = state[f"x{rank}"]
    target = torch.empty_like(partial)
    if kind == "B1":

        def call():
            op(partial, target)
            return target
    else:
        target.copy_(partial)

        def call():
            nccl.all_reduce(target, "sum")
            return target

    return call, target


def composed_functions(ar_functions, mhc_functions):
    result = []
    for ar, mhc in zip(ar_functions, mhc_functions, strict=True):

        def call(x, R, post, comb, ar=ar, mhc=mhc):
            return mhc(ar(), R, post, comb)

        result.append(call)
    return result


def fused_functions(cfg, weights, states, ar, rank):
    from ctikernels.mhc import MhcExit

    op = MhcExit(
        "cuda",
        hidden=cfg.hidden,
        rms_eps=cfg.rms_eps,
        hc_eps=cfg.hc_eps,
        norm_eps=cfg.norm_eps,
        post_mult=cfg.post_mult,
        sinkhorn_iters=cfg.sinkhorn_iters,
    )
    tokens = states[0]["x"].shape[0]
    if not op._fused(tokens, torch.bfloat16, ar):
        raise ValueError(
            "fused row requires an active B1 provider and a fitting grid decode bucket"
        )
    op.prepare([tokens], all_reduce=ar)
    functions = []
    for w, state in zip(weights, states, strict=True):
        outputs = {
            "residual_out": torch.empty_like(state["R"]),
            "post_out": torch.empty_like(state["post"]),
            "comb_out": torch.empty_like(state["comb"]),
            "out": torch.empty_like(state["x"]),
        }

        def call(x, R, post, comb, w=w, state=state, outputs=outputs):
            return op(state[f"x{rank}"], R, post, comb, **w, **outputs, all_reduce=ar)

        functions.append(call)
    return functions


def random_correctness(session, cfg, weights, state, functions, op, nccl, rank, T):
    """Untimed random inputs test communication rounding and the entire exit."""
    results = {}
    for kind in ("B1", "NCCL"):
        ar, _ = ar_function(kind, op, nccl, state, rank)
        reduced = ar()
        bitwise = rank_check(session, f"random/{T}/{kind}/ar", reduced)
        session.agree(f"random/{T}/{kind}/expected", torch.equal(reduced, state["x"]))
        delta = reduced.cpu().double() - state["x"].cpu().double()
        results[kind] = {
            "bitwise_across_ranks": bitwise,
            "matches_bf16_sum": True,
            "errors": {
                "max_abs": delta.abs().max().item(),
                "rms": delta.square().mean().sqrt().item(),
            },
        }
        for name, fns in functions.items():
            output = fns[0](reduced, state["R"], state["post"], state["comb"])
            correct = check(cfg, weights[0], state, output)
            correct["bitwise_across_ranks"] = rank_check(
                session, f"random/{T}/{kind}/{name}", output
            )
            results[f"{kind}+{name}"] = correct
    return results


def run(session, op, nccl, args):
    # Reuse the established barrier, peer timing, and slower-rank policy.
    from benchmark.kernel_tuning.two_node_allreduce.bench import timed_sample

    cfg = MhcConfig()
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    l2 = max(l2, session.exchange("l2", l2))
    report = {
        "config": asdict(cfg),
        "calls": args.calls,
        "pairs": args.pairs,
        "seed": args.seed,
        "timing_inputs": "zero partials, random residuals; no reset copies inside timing",
        "results": [],
    }
    timer = lambda fn, repeats, calls: timed_sample(session, fn, repeats, calls)
    for T in args.rows:
        weights, states, regime = setup_row(cfg, T, args, l2)
        functions, statuses, metadata = {}, {}, {}
        for name in args.providers:
            provider, fns = prepare_provider(name, cfg, weights, states, args)
            if fns is not None:
                try:
                    # Compile/warm every set before either rank enters composed collectives.
                    for fn, state in zip(fns, states, strict=True):
                        fn(*inputs(state))
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 - optional providers report unavailable
                    provider.status = "unavailable: " + failure_reason(exc, "warmup")
                    fns = None
            peer_ok = session.exchange(f"available/{T}/{name}", fns is not None)
            statuses[name] = (
                provider.status if not fns or peer_ok else "unavailable: peer preparation failed"
            )
            metadata[name] = dict(provider.metadata)
            if fns is not None and peer_ok:
                functions[name] = fns
        random_checks = random_correctness(
            session, cfg, weights, states[0], functions, op, nccl, args.rank, T
        )
        fused, fused_status = None, "unavailable: cti provider not prepared"
        if "cti" in functions:
            try:
                fused = fused_functions(cfg, weights, states, op, args.rank)
                fused_status = "available"
            except Exception as exc:  # noqa: BLE001 - agree before collective launch
                fused_status = "unavailable: " + failure_reason(exc, "fused prepare")
        peer_ok = session.exchange(f"fused/available/{T}", fused is not None)
        if not peer_ok:
            fused, fused_status = None, "unavailable: peer fused preparation failed"
        if fused is not None:
            output = fused[0](*inputs(states[0]))
            correct = check(cfg, weights[0], states[0], output)
            correct["bitwise_across_ranks"] = rank_check(session, f"random/{T}/fused", output)
            separate = functions["cti"][0](
                op(states[0][f"x{args.rank}"]),
                states[0]["R"],
                states[0]["post"],
                states[0]["comb"],
            )
            same = digest(output) == digest(separate)
            session.agree(f"random/{T}/fused/composed", same)
            correct["bitwise_matches_composed"] = same
            random_checks["B1+mHC fused"] = correct
        # Match two_node_allreduce/bench.py:110-119: repeated in-place sums of
        # zero remain stable. Random correctness above uses real per-rank partials.
        for state in states:
            for key in ("x", "x0", "x1"):
                state[key].zero_()
        for mode in args.modes:
            launchers, rows, usable = {}, {}, {}
            for name in args.providers:
                rows[name] = {"status": statuses[name], "metadata": dict(metadata[name])}
                if name not in functions:
                    continue
                local_ok = True
                try:
                    launch = make_launcher(block_calls(functions[name], states, args.calls), mode)
                    output = launch()
                    index = (args.calls - 1) % len(states)
                    rows[name]["correctness"] = check(cfg, weights[index], states[index], output)
                    if args.count_kernels:
                        rows[name]["metadata"].update(count_kernels(launch, args.calls))
                except Exception as exc:  # noqa: BLE001 - optional providers report unavailable
                    rows[name]["status"] = "unavailable: " + failure_reason(exc, "capture/warmup")
                    local_ok = False
                peer_ok = session.exchange(f"mode/{T}/{mode}/{name}", local_ok)
                if local_ok and peer_ok:
                    launchers[name] = launch
                    usable[name] = functions[name]
                    rows[name]["correctness"]["bitwise_across_ranks"] = rank_check(
                        session, f"mhc/{T}/{mode}/{name}", output
                    )
                elif local_ok:
                    rows[name]["status"] = "unavailable: peer capture/warmup failed"
            ar_launchers, composed = {}, {}
            for kind in ("B1", "NCCL"):
                ar_functions = [
                    ar_function(kind, op, nccl, state, args.rank)[0] for state in states
                ]
                ar_wrappers = [(lambda *args, ar=ar: ar()) for ar in ar_functions]
                session.sync()
                ar_launchers[kind] = make_launcher(
                    block_calls(ar_wrappers, states, args.calls), mode
                )
                output = ar_launchers[kind]()
                rank_check(session, f"ar/{T}/{mode}/{kind}", output)
                session.agree(f"ar/zero/{T}/{mode}/{kind}", torch.count_nonzero(output).item() == 0)
                rows[kind] = {
                    "status": "available",
                    "correctness": {"bitwise_across_ranks": True},
                    "metadata": {"kernels_per_exit": None},
                }
                if args.count_kernels:
                    rows[kind]["metadata"].update(count_kernels(ar_launchers[kind], args.calls))
                for name, fns in usable.items():
                    label = f"{kind}+{name}"
                    session.sync()
                    composed[label] = make_launcher(
                        block_calls(composed_functions(ar_functions, fns), states, args.calls), mode
                    )
                    output = composed[label]()
                    index = (args.calls - 1) % len(states)
                    correct = check(cfg, weights[index], states[index], output)
                    correct["bitwise_across_ranks"] = rank_check(
                        session, f"composed/{T}/{mode}/{label}", output
                    )
                    rows[label] = {
                        "status": "available",
                        "correctness": correct,
                        "metadata": dict(metadata[name]),
                    }
                    rows[label]["metadata"]["mhc_kernels_per_exit"] = rows[label]["metadata"].pop(
                        "kernels_per_exit", None
                    )
                    rows[label]["metadata"]["kernels_per_exit"] = None
                    if args.count_kernels:
                        rows[label]["metadata"].update(count_kernels(composed[label], args.calls))
            label = "B1+mHC fused"
            rows[label] = {"status": fused_status}
            if fused is not None:
                session.sync()
                composed[label] = make_launcher(block_calls(fused, states, args.calls), mode)
                output = composed[label]()
                index = (args.calls - 1) % len(states)
                correct = check(cfg, weights[index], states[index], output)
                correct["bitwise_across_ranks"] = rank_check(session, f"fused/{T}/{mode}", output)
                rows[label] = {
                    "status": "available",
                    "correctness": correct,
                    "metadata": {"kernels_per_exit": 1, "backend": "CuTe DSL", "norm_fused": True},
                }
                if args.count_kernels:
                    rows[label]["metadata"].update(count_kernels(composed[label], args.calls))
            # Fused and composed providers share the same paired/interleaved samples.
            for group in (ar_launchers, launchers, composed):
                timings = measure(group, args, timer)
                for name, timing in timings.items():
                    timing["median_step_us"] = timing["median_us"] * args.calls
                    rows.setdefault(
                        name, {"status": "available", "correctness": {"bitwise_across_ranks": True}}
                    )["timing"] = timing
            for kind in ("B1", "NCCL"):
                for name in args.providers:
                    rows.setdefault(f"{kind}+{name}", {"status": rows[name]["status"]})
            report["results"].append(
                {
                    "T": T,
                    "bytes": T * cfg.hidden * 2,
                    "mode": mode,
                    **regime,
                    "random_correctness": random_checks,
                    "providers": rows,
                }
            )
            if args.rank == 0:
                for name, result in rows.items():
                    print_row(T, mode, name, result)
                save_report(report, args.out)
            del launchers, ar_launchers, composed
        op.check()
    return report


def main():
    parser = arguments(pair=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--store", required=True, help="rank-0 TCPStore address:port")
    args = parser.parse_args()
    validate(parser, args)
    if not torch.cuda.is_available():
        parser.error("two-node benchmark requires CUDA")
    from benchmark.kernel_tuning.two_node_allreduce.bench import Session, big_cores, proxy_core
    from ctikernels.collective.two_node import TwoNodeAllReduce
    from ctikernels.pynccl import init_pynccl

    core = proxy_core()
    cores = sorted(set(big_cores()) & os.sched_getaffinity(0) - {core})
    if not cores:
        parser.error("need an available big core besides the proxy core")
    os.sched_setaffinity(0, cores)
    os.environ.setdefault("NCCL_MAX_NCHANNELS", "2")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    address, port = args.store.rsplit(":", 1)
    store = dist.TCPStore(address, int(port), 2, args.rank == 0, timedelta(seconds=120))
    session = Session(args.rank, store)
    config = {
        key: getattr(args, key)
        for key in (
            "rows",
            "modes",
            "providers",
            "calls",
            "pairs",
            "seed",
            "weight_sets",
            "hot",
            "count_kernels",
        )
    }
    if session.exchange("config", config) != config:
        raise ValueError("benchmark rank configurations differ")
    dist.init_process_group(
        "gloo",
        store=dist.PrefixStore("bench/gloo", store),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    # Identical NCCL initialization to two_node_allreduce/bench.py:299-301.
    nccl = init_pynccl(
        tp_rank=args.rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=0
    )
    op = TwoNodeAllReduce(
        rank=args.rank, store=store, prefix="bench/provider", proxy_cpu=core, max_bytes=256 * 1024
    )
    try:
        op.prepare([T * 4096 * 2 for T in args.rows])
        session.agree("B1-active", op.active)
        op.arm()
        with torch.inference_mode():
            report = run(session, op, nccl, args)
        if args.rank == 0:
            save_report(report, args.out)
    finally:
        torch.cuda.synchronize()
        session.sync()
        op.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
