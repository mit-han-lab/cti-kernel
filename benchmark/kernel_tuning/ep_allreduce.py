# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired graph timing for host exchange and NCCL MoE-exit paths.

Run as ``python -m benchmark.kernel_tuning.ep_allreduce --out results.jsonl``.
The market comparison models SGLang 0.5.22 / vLLM 0.31.0 on a pair without
P2P: custom all-reduce requires P2P and FlashInfer PCIe IPC is opt-in, so the
ordinary fallback is direct PyNCCL all-reduce. No GPU work occurs at import.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import socket
import traceback
from contextlib import contextmanager
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark._support import _percentile
from ctikernels.collective.inference import (
    AllReduceAddRMSNorm,
    exact_reference,
    market_composition,
    measured_floor,
    reference,
)

CANDIDATES = (
    "cute",
    "cute_w8",
    "cute_w16",
    "nccl_ar",
    "nccl_ar_norm",
    "nccl_ar_cti_norm",
    "engine",
    "fi_pcie_ipc",
)
# The fused add-RMSNorm halves of the market bar; the faster one sets it per row.
NORM_CANDIDATES = ("nccl_ar_norm", "nccl_ar_cti_norm")
DEFAULT_TOKENS = "1,2,3,4,5,6,7,8,16,32,64,128,256,512,1024,2048,4096,8192"


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=_positive_int, default=2048)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--rounds", type=_positive_int, default=15)
    parser.add_argument("--calls", type=_positive_int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.tokens = list(dict.fromkeys(_positive_int(t) for t in args.tokens.split(",")))
    except (ValueError, argparse.ArgumentTypeError):
        parser.error("--tokens must be comma-separated positive integers")
    args.candidates = list(dict.fromkeys(args.candidates.split(",")))
    if any(c not in CANDIDATES for c in args.candidates):
        parser.error(f"--candidates must be comma-separated names from {CANDIDATES}")
    return args


def format_table(rows):
    lines = [
        "| H | T | CuTe µs | Market AR µs | Market AR + norm µs | AR / CuTe | AR + norm / CuTe |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    for row in rows:
        values = [row["hidden"], row["tokens"]]
        values += [
            number(row.get(key))
            for key in (
                "cute_us",
                "market_ar_us",
                "market_ar_norm_us",
                "speedup_vs_market_ar",
                "speedup_vs_market_ar_norm",
            )
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines)


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _loaded_nccl_version():
    """The NCCL the process resolves at run time (``torch.cuda.nccl.version()`` reports the
    version torch was built against, which a preloaded library does not change)."""
    import ctypes

    try:
        library = ctypes.CDLL("libnccl.so.2")
        code = ctypes.c_int()
        if library.ncclGetVersion(ctypes.byref(code)) != 0:
            return None
    except (AttributeError, OSError):
        return None
    major, rest = divmod(code.value, 10000)
    return [major, *divmod(rest, 100)]


def environment():
    """Read both visible devices before any per-process device selection."""
    return {
        "torch": torch.__version__,
        "nccl_built": torch.cuda.nccl.version(),
        "nccl_loaded": _loaded_nccl_version(),
        "flashinfer": _version("flashinfer-python"),
        "cutlass_dsl": _version("nvidia-cutlass-dsl"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
        "peer_access_0_to_1": torch.cuda.can_device_access_peer(0, 1),
    }


def _gather(value):
    values = [None, None]
    dist.all_gather_object(values, value)
    return values


def _signature_kwargs(function, values):
    """Bind known public parameter names; reject unrecognized required arguments."""
    kwargs = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is parameter.empty:
            raise TypeError(f"unsupported required PCIe IPC parameter: {name}")
    return kwargs


def probe_pcie_ipc(rank, hidden, max_tokens):
    """Probe the optional installed API; report failures instead of aborting timing.

    Releases without this workspace simply return unavailable. Parameter names
    are bound from the installed signature, so an incompatible API is reported
    explicitly rather than silently substituting a different collective.
    """
    workspace = None
    try:
        from flashinfer.comm import PcieIpcAllReduceWorkspace

        values = {
            "rank": rank,
            "tp_rank": rank,
            "world_size": 2,
            "tp_size": 2,
            "group": dist.group.WORLD,
            "cpu_group": dist.group.WORLD,
            "process_group": dist.group.WORLD,
            "device": torch.device("cuda", rank),
            "hidden": hidden,
            "hidden_dim": hidden,
            "hidden_size": hidden,
            "max_tokens": max_tokens,
            "max_token_num": max_tokens,
            "max_size_bytes": max_tokens * hidden * 2,
            "dtype": torch.bfloat16,
        }
        workspace = PcieIpcAllReduceWorkspace(
            **_signature_kwargs(PcieIpcAllReduceWorkspace, values)
        )
        # Some versions expose a shape predicate on the constructed workspace.
        supports = getattr(workspace, "supports", None)
        if supports is not None:
            supported = supports(**_signature_kwargs(supports, values))
            ok = supported[0] if isinstance(supported, tuple) else supported
            if not ok:
                raise RuntimeError(f"supports() returned {supported!r}")
        method = workspace.all_reduce
        # Resolve the call shape before entering capture, without a GPU launch.
        params = {"input", "inp", "x", "input_tensor", "out", "output", "output_tensor"}
        _signature_kwargs(method, values | dict.fromkeys(params))
        return workspace, {
            "status": "available",
            "signature": str(inspect.signature(PcieIpcAllReduceWorkspace)),
        }
    except Exception as error:  # noqa: BLE001 -- optional probe must never abort the comparison.
        return workspace, {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def _destroy_probe(workspace):
    if workspace is None:
        return
    try:
        close = getattr(workspace, "destroy", None) or getattr(workspace, "close", None)
        if close is not None:
            close()
    except Exception as error:  # noqa: BLE001 -- preserve probe diagnostics even if cleanup fails.
        print(json.dumps({"fi_pcie_ipc_cleanup": f"{type(error).__name__}: {error}"}), flush=True)


@contextmanager
def _flag_warps(warps):
    from ctikernels.collective._host_exchange.cute import exchange

    original = exchange.schedule

    def forced(*args, **kwargs):
        cfg = original(*args, **kwargs)
        return (cfg[0], warps, *cfg[2:]) if cfg[0] == exchange.FLAG else cfg

    if warps is not None:
        exchange.schedule = forced
    try:
        yield
    finally:
        exchange.schedule = original


class Candidate:
    def __init__(self, name, rank, args, direct, staged, ipc):
        self.name, self.op, self.ipc = name, None, ipc
        self.warps = {"cute_w8": 8, "cute_w16": 16}.get(name)
        self.label = name
        self.norm = None
        if name.startswith("cute"):
            self.op = AllReduceAddRMSNorm(
                rank=rank,
                world_size=2,
                cpu_group=dist.group.WORLD,
                device=f"cuda:{rank}",
                hidden=args.hidden,
                max_tokens=max(args.tokens),
                eps=1e-6,
                provider="cute",
            )
            with _flag_warps(self.warps):
                self.op.prepare(args.tokens)
        self.comm = staged if name == "engine" else direct
        if name in {"nccl_ar_cti_norm", "engine"}:
            from ctikernels.norm.inference import add_rms_norm_

            self.norm = lambda x, r, w: add_rms_norm_(x, r, w, 1e-6, zero_centered=True)
            self.label += " + ctikernels.add_rms_norm_"
        elif name == "nccl_ar_norm":
            # Without FlashInfer this candidate is reported unavailable, never substituted.
            from flashinfer.norm import gemma_fused_add_rmsnorm

            self.norm = lambda x, r, w: gemma_fused_add_rmsnorm(x, r, w, 1e-6)
            self.label = "nccl_ar + flashinfer.gemma_fused_add_rmsnorm"

    def call(self, x, residual, weight):
        if self.op is not None:
            with _flag_warps(self.warps):
                self.op(x, residual, weight)
        elif self.name == "fi_pcie_ipc":
            method = self.ipc.all_reduce
            values = {
                "input": x,
                "inp": x,
                "x": x,
                "input_tensor": x,
                "out": x,
                "output": x,
                "output_tensor": x,
            }
            result = method(**_signature_kwargs(method, values))
            if isinstance(result, torch.Tensor) and result.data_ptr() != x.data_ptr():
                x.copy_(result)
        else:
            self.comm.all_reduce(x, "sum")
            if self.norm is not None:
                self.norm(x, residual, weight)

    def close(self):
        if self.op is not None:
            self.op.close()


def _correct(candidate, rank, tokens, hidden, seed):
    g = torch.Generator().manual_seed(seed + rank)
    part = torch.randn(tokens, hidden, generator=g).bfloat16()
    parts = [torch.empty_like(part) for _ in range(2)]
    dist.all_gather(parts, part)
    g.manual_seed(seed + 100)
    residual = torch.randn(tokens, hidden, generator=g).bfloat16()
    weight = torch.randn(hidden, generator=g).bfloat16()
    ar_only = candidate.name in {"nccl_ar", "fi_pcie_ipc"}
    expected = reference(
        torch.stack(parts), torch.zeros_like(residual) if ar_only else residual, weight, 1e-6
    )
    x, r, w = part.cuda(), residual.cuda(), weight.cuda()
    floor = None
    if candidate.name.startswith("cute"):
        # Market composition on the same inputs: NCCL all-reduce, BF16 add, RMSNorm.
        summed = x.clone()
        candidate.comm.all_reduce(summed, "sum")
        market = market_composition(summed, r, w, 1e-6)
    candidate.call(x, r, w)
    x, r = x.cpu(), r.cpu()
    if ar_only:
        torch.testing.assert_close(x, expected[1], rtol=0, atol=0)
    elif candidate.name.startswith("cute"):
        torch.testing.assert_close(r, expected[1], rtol=0, atol=0)
        # Measured floor: error to the FP64 oracle at most the market composition's.
        exact = exact_reference(torch.stack(parts), residual, weight, 1e-6)
        failures, floor = measured_floor((x, r), market, exact)
        if failures:
            raise AssertionError("; ".join(failures))
    else:
        for actual, oracle in zip((x, r), expected, strict=True):
            torch.testing.assert_close(actual, oracle, rtol=0.02, atol=0.04)
    return floor


def _capture(candidate, tokens, args, distinct):
    # Zeros stay finite even when a graph repeatedly overwrites the same inputs.
    # Distinct buffers model successive layers; reuse is recorded when memory is tight.
    count = args.calls if distinct else 1
    buffers = [
        (
            torch.zeros(tokens, args.hidden, device="cuda", dtype=torch.bfloat16),
            torch.zeros(tokens, args.hidden, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(count)
    ]
    weight = torch.zeros(args.hidden, device="cuda", dtype=torch.bfloat16)
    candidate.call(*buffers[0], weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(args.calls):
            candidate.call(*buffers[index % count], weight)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, buffers, weight


def _worker(rank, port, args, env, queue):
    candidates, ipc = {}, None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(minutes=20),
        )
        from ctikernels import init_pynccl

        # Created after main sets debug environment for probe-only runs.
        direct = init_pynccl(
            tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=256
        )
        probe = {"status": "not requested"}
        if args.probe_only or "fi_pcie_ipc" in args.candidates:
            ipc, probe = probe_pcie_ipc(rank, args.hidden, max(args.tokens))
            reports = _gather(probe)
            if any(report["status"] != "available" for report in reports):
                probe = {"status": "unavailable", "ranks": reports}
        if args.probe_only:
            direct.all_reduce(torch.zeros(1, device="cuda", dtype=torch.bfloat16), "sum")
            torch.cuda.synchronize()
            if rank == 0:
                queue.put(
                    {
                        "environment": env,
                        "fi_pcie_ipc": probe,
                        "nccl_transport": "Read NCCL INFO lines containing 'via' in the raw process log.",
                    }
                )
        else:
            staged = None
            if "engine" in args.candidates:
                staged = init_pynccl(
                    tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=64 << 20
                )
            unavailable = {}
            for name in args.candidates:
                if name == "fi_pcie_ipc" and probe["status"] != "available":
                    unavailable[name] = probe
                    continue
                error = None
                try:
                    candidate = Candidate(name, rank, args, direct, staged, ipc)
                    candidates[name] = candidate
                except Exception as exc:  # noqa: BLE001 -- report candidate failure on both ranks.
                    error = f"{type(exc).__name__}: {exc}"
                errors = _gather(error)
                if any(errors):
                    unavailable[name] = {"status": "unavailable", "reasons": errors}
                    failed = candidates.pop(name, None)
                    if failed is not None:
                        failed.close()
            calibration = {
                name: {
                    "link_rate": c.op._exchange.link_rate,
                    "calibrated": c.op._exchange.calibrated,
                }
                for name, c in candidates.items()
                if c.op is not None
            }
            calibration = _gather(calibration)
            for tokens in args.tokens:
                graphs, results = {}, dict(unavailable)
                free = min(_gather(torch.cuda.mem_get_info()[0]))
                required = len(candidates) * args.calls * tokens * args.hidden * 4
                distinct = required < free * 0.5
                floors = {}
                for name, candidate in candidates.items():
                    error = None
                    try:
                        floors[name] = _correct(
                            candidate, rank, tokens, args.hidden, args.seed + tokens
                        )
                    except Exception as exc:  # noqa: BLE001 -- wrong candidates are never timed.
                        error = f"{type(exc).__name__}: {exc}"
                    errors = _gather(error)
                    if any(errors):
                        results[name] = {"status": "incorrect", "reasons": errors}
                        continue
                    error = None
                    try:
                        graphs[name] = _capture(candidate, tokens, args, distinct)
                    except Exception as exc:  # noqa: BLE001 -- optional capture support may be absent.
                        error = f"{type(exc).__name__}: {exc}"
                    errors = _gather(error)
                    if any(errors):
                        graphs.pop(name, None)
                        results[name] = {"status": "unavailable", "reasons": errors}
                samples = {name: [] for name in graphs}
                rng = random.Random(args.seed + tokens)
                for _ in range(args.rounds):
                    order = list(samples)
                    rng.shuffle(order)
                    for name in order:
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        dist.barrier()
                        elapsed, error = None, None
                        try:
                            start.record()
                            graphs[name][0].replay()
                            end.record()
                            end.synchronize()
                            elapsed = start.elapsed_time(end) * 1000 / args.calls
                        except Exception as exc:  # noqa: BLE001 -- a probe can fail at replay.
                            error = f"{type(exc).__name__}: {exc}"
                        readings = _gather((elapsed, error))
                        errors = [reading[1] for reading in readings]
                        if any(errors):
                            results[name] = {"status": "unavailable", "reasons": errors}
                            del samples[name]
                        else:
                            samples[name].append(max(reading[0] for reading in readings))
                for name, times in samples.items():
                    results[name] = {
                        "status": "ok",
                        "label": candidates[name].label,
                        "median_us": _percentile(times, 0.5),
                        "p20_us": _percentile(times, 0.2),
                        "p80_us": _percentile(times, 0.8),
                        "round_max_us": times,
                        "distinct_buffers": distinct,
                    }
                    if floors.get(name) is not None:
                        results[name]["error_floor"] = floors[name]

                def latency(name, results=results):
                    return results.get(name, {}).get("median_us")

                cute, market = latency("cute"), latency("nccl_ar")
                norms = [latency(name) for name in NORM_CANDIDATES if latency(name) is not None]
                market_norm = min(norms) if norms else None
                row = {
                    "hidden": args.hidden,
                    "tokens": tokens,
                    "rounds": args.rounds,
                    "calls": args.calls,
                    "seed": args.seed,
                    "environment": env,
                    "calibration": calibration,
                    "candidates": results,
                    "cute_us": cute,
                    "market_ar_us": market,
                    "market_ar_norm_us": market_norm,
                    "speedup_vs_market_ar": market / cute if market is not None and cute else None,
                    "speedup_vs_market_ar_norm": (
                        market_norm / cute if market_norm is not None and cute else None
                    ),
                }
                if rank == 0:
                    queue.put(row)
                del graphs
                torch.cuda.synchronize()
        dist.barrier()
        for candidate in candidates.values():
            candidate.close()
        _destroy_probe(ipc)
        dist.destroy_process_group()
        queue.put({"done": rank})
    except Exception:  # noqa: BLE001 -- return errors without forcibly ending peer processes.
        queue.put({"error": traceback.format_exc(), "rank": rank})
        queue.put({"done": rank})


def main(argv=None):
    args = parse_args(argv)
    if args.probe_only:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,ENV"
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This benchmark requires two CUDA GPUs")
    env = environment()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(rank, port, args, env, queue)) for rank in range(2)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, errors, done = [], [], 0
    with args.out.open("w") as output:
        for process in processes:
            process.start()
        while done < 2:
            row = queue.get()
            if "done" in row:
                done += 1
                continue
            if "error" in row:
                errors.append(row)
            elif "tokens" in row:
                rows.append(row)
            line = json.dumps(row)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()
        for process in processes:
            process.join()
    if errors or any(p.exitcode != 0 for p in processes):
        raise RuntimeError("A benchmark process failed; see the reported errors")
    if rows:
        print(format_table(rows))


if __name__ == "__main__":
    main()
