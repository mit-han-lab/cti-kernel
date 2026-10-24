# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired per-layer DFlash attention against both market dispatches.

One parent serializes measurements on one GPU through persistent JSONL workers.
The main worker hosts the candidate (DraftAttention automatic dispatch by default)
and vLLM's real vllm_flash_attn P=16 dispatch. The sglang worker prepends the
supplied FlashInfer 0.7.1 directory before imports, isolating it from the product's
FlashInfer. Both workers pin to available Cortex-X925 cores on GB10.

Each row allocates identical seed-0 BF16 Q/K/V in both workers, with n distinct
layer K/V buffers covering >= 2x L2, and captures one n-call graph per provider.
The last layer's graph output is checked against independent FP32 dense GQA
(atol=rtol=2e-2, TF32 disabled). After warmup, each reference gets exactly 20
reference/reference self-pairs. Epsilon is fixed at the nearest-rank p95 of
abs(a/b-1): the 19th sorted deviation. Each of >=20 rounds then measures one
candidate/vllm pair and one candidate/sglang pair, alternating which side runs
first by round parity. Every interval synchronizes, times --calls back-to-back
graph replays with one CUDA event pair, and reports elapsed/(calls*n) in us.

A reference passes when median(candidate/reference) <= 1+epsilon and both
correctness checks pass. faster_market names the reference with the lower
paired reference median; market_gate requires BOTH references to pass. An
omitted comparator makes its result and market_gate N/A, never PASS. All raw
samples, ratios, medians, dispatch metadata and correctness are emitted as JSONL.
Row errors emit ERROR and restart workers for the next row. Teardown releases
row graphs and buffers and empties the CUDA allocator cache.

Select one Q for the usual 48-row invocation (two shapes, four batches, three
contexts and two capacities); workers and imports persist across all rows::

    python -m benchmark.kernel_tuning.dflash_draft_layers_paired --queries 4 \
        --vllm-fa2 artifacts/_vllm_fa2_C.abi3.so --vllm-fa2-wheel vllm.whl \
        --sglang-flashinfer artifacts/flashinfer071

The full default query grid contains 336 rows. No torch or CUDA imports occur
in the parent or in the pure protocol helpers. stdout contains only JSONL;
worker library output, including native stdout, is redirected to stderr.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.kernel_tuning import dflash_draft_layers as layers

REFERENCES = ("vllm_fa2", "sglang")
MODULE = "benchmark.kernel_tuning.dflash_draft_layers_paired"
SELF_PAIRS = 20
CANDIDATES = ("auto", "cute", "cute_fused", "draft_fi", "table", "table_row", *layers.MIN_CHUNKS)


def epsilon(self_pairs):
    """Nearest-rank p95 for exactly twenty reference/reference pairs."""
    if len(self_pairs) != SELF_PAIRS:
        raise ValueError("epsilon requires exactly 20 self-pairs")
    return sorted(abs(a / b - 1) for a, b in self_pairs)[18]


def summarize_reference(self_pairs, samples, correct, *, tolerance=None):
    if tolerance is None:
        tolerance = epsilon(self_pairs)
    ratios = [candidate / reference for candidate, reference in samples]
    median = statistics.median(ratios)
    speed = "PASS" if median <= 1 + tolerance else "FAIL"
    return {
        "reference_self_pairs_us": self_pairs,
        "candidate_reference_pairs_us": samples,
        "epsilon": tolerance,
        "ratios": ratios,
        "median_ratio": median,
        "candidate_median_us": statistics.median(a for a, _ in samples),
        "reference_median_us": statistics.median(b for _, b in samples),
        "speed": speed,
        "status": "PASS" if speed == "PASS" and correct else "FAIL",
    }


def unavailable_reference(name):
    return {
        "reference_self_pairs_us": [],
        "candidate_reference_pairs_us": [],
        "epsilon": None,
        "ratios": [],
        "median_ratio": None,
        "candidate_median_us": None,
        "reference_median_us": None,
        "speed": "N/A",
        "status": "N/A",
        "reason": "--vllm-fa2 absent" if name == "vllm_fa2" else "--sglang-flashinfer absent",
    }


def market_decision(references, correctness):
    if any(references[name]["status"] == "N/A" for name in REFERENCES):
        return {"faster_market": None, "market_gate": "N/A"}
    faster = min(REFERENCES, key=lambda name: references[name]["reference_median_us"])
    passed = all(references[name]["status"] == "PASS" for name in REFERENCES) and all(
        correctness.get(name) == "PASS" for name in ("candidate", *REFERENCES)
    )
    return {"faster_market": faster, "market_gate": "PASS" if passed else "FAIL"}


def paired_timings(measure, available, *, pairs, calls, correctness):
    """Run serialized RPC intervals; store pairs in candidate/reference order."""
    calibration = {
        name: [[measure(name, calls), measure(name, calls)] for _ in range(SELF_PAIRS)]
        for name in available
    }
    # Fix both tolerances before any candidate/reference measurement.
    tolerances = {name: epsilon(values) for name, values in calibration.items()}
    samples = {name: [] for name in available}
    for index in range(pairs):
        for name in available:
            if index % 2:
                ref, candidate = measure(name, calls), measure("candidate", calls)
            else:
                candidate, ref = measure("candidate", calls), measure(name, calls)
            samples[name].append([candidate, ref])
    references = {
        name: (
            summarize_reference(
                calibration[name],
                samples[name],
                correctness["candidate"] == correctness[name] == "PASS",
                tolerance=tolerances[name],
            )
            if name in available
            else unavailable_reference(name)
        )
        for name in REFERENCES
    }
    return {"references": references, **market_decision(references, correctness)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", type=lambda value: value.split(","), default=["draft_noncausal", "q3_target"]
    )
    parser.add_argument("--batches", type=layers.csv_ints, default=[1, 2, 3, 4])
    parser.add_argument("--contexts", type=layers.csv_ints, default=[100, 400, 1000])
    parser.add_argument("--queries", type=layers.csv_ints, default=[4, 5, 6, 8, 9, 12, 16])
    parser.add_argument(
        "--capacities", default="exact,32768", help="comma-separated exact or token capacities"
    )
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--vllm-fa2", type=Path)
    parser.add_argument("--vllm-fa2-wheel", help="wheel name recorded in the header")
    parser.add_argument("--sglang-flashinfer", type=Path)
    parser.add_argument("--candidate", choices=CANDIDATES, default="auto")
    parser.add_argument("--worker", choices=("main", "sglang"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.capacities = [
            0 if item == "exact" else int(item) for item in args.capacities.split(",")
        ]
    except ValueError:
        parser.error("capacities must be a comma list of exact or positive integers")
    if (
        any(name not in layers.SHAPES for name in args.shapes)
        or min(args.batches) < 1
        or min(args.contexts) < 0
        or any(not 2 <= q <= 32 for q in args.queries)
        or any(
            cap < 0 or (cap and cap < max(args.contexts) + max(args.queries))
            for cap in args.capacities
        )
        or args.pairs < 20
        or args.calls < 1
        or args.warmup < 0
    ):
        parser.error(
            "require known shapes, B>=1, ctx>=0, Q=2..32, sufficient capacity, pairs>=20, calls>=1, warmup>=0"
        )
    if args.worker == "sglang" and args.sglang_flashinfer is None:
        parser.error("sglang worker requires --sglang-flashinfer")
    return args


def generate_rows(args):
    for Q in args.queries:
        for name in args.shapes:
            for B in args.batches:
                for ctx in args.contexts:
                    for cap in args.capacities:
                        yield {
                            "shape": name,
                            "B": B,
                            "ctx": ctx,
                            "Q": Q,
                            "capacity": cap or ctx + Q,
                        }


def artifact(args):
    digest = None
    if args.vllm_fa2 is not None:
        sha = hashlib.sha256()
        with args.vllm_fa2.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
    return {
        "so_sha256": digest,
        "wheel": Path(args.vllm_fa2_wheel).name if args.vllm_fa2_wheel else None,
    }


def worker_info(args):
    if args.worker == "sglang":
        sys.path.insert(0, str(args.sglang_flashinfer.resolve()))
    affinity = layers.pin_cores()
    import flashinfer
    import torch

    if args.worker == "sglang" and (
        flashinfer.__version__ != "0.7.1"
        or Path(flashinfer.__file__).resolve().parent
        != (args.sglang_flashinfer / "flashinfer").resolve()
    ):
        raise ValueError("SGLang requires the supplied FlashInfer 0.7.1 package")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    reported_l2 = getattr(props, "L2_cache_size", 0)
    try:
        version = importlib.metadata.version("cti-kernels")
    except importlib.metadata.PackageNotFoundError:
        version = None
    # Locate the package without importing product code in the SGLang worker.
    spec = importlib.util.find_spec("ctikernels")
    package_path = Path(spec.origin) if spec is not None and spec.origin is not None else None
    return {
        "device": props.name,
        "device_index": 0,
        "device_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "sms": props.multi_processor_count,
        "l2_bytes": reported_l2 or (96 << 20),
        "l2_source": "device" if reported_l2 else "conservative 96 MiB fallback",
        **affinity,
        "versions": {
            "torch": torch.__version__,
            "cti-kernels": version,
            "ctikernels_import_path": package_path.relative_to(
                package_path.parent.parent
            ).as_posix()
            if package_path
            else None,
            "flashinfer": flashinfer.__version__,
        },
    }


class WorkerRow:
    def __init__(self, row, args, l2):
        import torch

        self.graphs = {}
        self.retained = []
        self.geometry = layers.layer_geometry(row["shape"], row["B"], row["ctx"], row["Q"], l2)
        self.n = self.geometry["n"]
        q, kc, vc = layers.layer_buffers(row["B"], row["Q"], self.geometry)
        self.retained.extend([q, kc, vc])
        providers = (
            {"sglang": "sglang"} if args.worker == "sglang" else {"candidate": args.candidate}
        )
        if args.worker == "main" and args.vllm_fa2 is not None:
            providers["vllm_fa2"] = "vllm_fa2"
        options = SimpleNamespace(query=row["Q"], capacity=row["capacity"], vllm_fa2=args.vllm_fa2)
        self.report = {}
        expected = None
        for role, provider in providers.items():
            metadata = {"provider": provider, "op.provider": None, "cute_tuning": None}
            fn, output, resources = layers.build_provider(
                provider,
                q,
                kc,
                vc,
                row["B"],
                row["ctx"],
                self.geometry["lpad"],
                self.geometry["causal"],
                self.geometry["hq"],
                self.geometry["hk"],
                self.geometry["d"],
                metadata,
                options,
            )
            self.retained.extend([fn, output, resources])
            graph = layers.capture_graph(fn, self.n)
            self.graphs[role] = graph
            graph.replay()
            torch.cuda.synchronize()
            if expected is None:
                expected = layers.reference(
                    q,
                    kc[-1],
                    vc[-1],
                    row["B"],
                    self.geometry["lpad"],
                    row["ctx"] + row["Q"],
                    self.geometry["causal"],
                    row["Q"],
                    self.geometry["hq"],
                    self.geometry["hk"],
                    self.geometry["d"],
                )
            metadata["correct"] = (
                "PASS" if torch.allclose(output.float(), expected, atol=2e-2, rtol=2e-2) else "FAIL"
            )
            self.report[role] = metadata
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def warmup(self, provider, calls):
        import torch

        for _ in range(calls):
            self.graphs[provider].replay()
        torch.cuda.synchronize()

    def measure(self, provider, calls):
        import torch

        graph = self.graphs[provider]
        torch.cuda.synchronize()
        self.start.record()
        for _ in range(calls):
            graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / (calls * self.n)

    def close(self):
        import torch

        torch.cuda.synchronize()
        self.graphs.clear()
        self.retained.clear()
        gc.collect()
        torch.cuda.empty_cache()


def run_worker(args):
    # Preserve a JSON-only descriptor, then redirect Python and native stdout.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def reply(value):
            protocol.write(json.dumps(value, allow_nan=False) + "\n")
            protocol.flush()

        state = None
        try:
            info = worker_info(args)
            reply({"status": "READY", **info})
            import torch

            with torch.no_grad():
                for line in sys.stdin:
                    request = json.loads(line)
                    command = request["command"]
                    if command == "setup":
                        if state is not None:
                            raise RuntimeError("row must be torn down before setup")
                        state = WorkerRow(request["row"], args, info["l2_bytes"])
                        reply(
                            {"status": "OK", "providers": state.report, "geometry": state.geometry}
                        )
                    elif command == "measure":
                        reply(
                            {
                                "status": "OK",
                                "us": state.measure(request["provider"], request["calls"]),
                            }
                        )
                    elif command == "warmup":
                        state.warmup(request["provider"], request["calls"])
                        reply({"status": "OK"})
                    elif command == "teardown":
                        state.close()
                        state = None
                        reply({"status": "OK"})
                    else:
                        raise ValueError("unknown worker command")
            if state is not None:
                state.close()
        except Exception as exc:  # noqa: BLE001 - no paths or exception contents in receipts.
            reply({"status": "ERROR", "error_type": type(exc).__name__})
            return 1
    return 0


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, kind, args):
        command = [sys.executable, "-m", MODULE, "--worker", kind, "--candidate", args.candidate]
        if kind == "main" and args.vllm_fa2 is not None:
            command.extend(["--vllm-fa2", str(args.vllm_fa2)])
        if kind == "sglang":
            command.extend(["--sglang-flashinfer", str(args.sglang_flashinfer)])
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise WorkerError("worker closed its output")
        result = json.loads(line)
        if result["status"] == "ERROR":
            raise WorkerError("worker reported an error")
        return result

    def request(self, command, **fields):
        self.process.stdin.write(json.dumps({"command": command, **fields}) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        # EOF requests normal worker shutdown; never send a process signal.
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass  # An already-exited worker needs no EOF.
        self.process.wait()
        self.process.stdout.close()


def run_parent(args):
    workers, infos = {}, {}
    header_emitted = False
    failed = False
    generation = 0
    kinds = ["main"] + (["sglang"] if args.sglang_flashinfer is not None else [])
    available = [
        name
        for name in REFERENCES
        if (args.vllm_fa2 if name == "vllm_fa2" else args.sglang_flashinfer) is not None
    ]

    def stop_workers():
        for worker in workers.values():
            worker.close()
        workers.clear()

    try:
        for row_id, row in enumerate(generate_rows(args)):
            stage = "setup"
            result = {"type": "row", "row_id": row_id, **row, "candidate": args.candidate}
            try:
                if not workers:
                    generation += 1
                    infos.clear()
                    for kind in kinds:
                        worker = Worker(kind, args)
                        workers[kind] = worker
                        infos[kind] = worker.receive()
                    if "sglang" in infos and any(
                        infos["main"][key] != infos["sglang"][key]
                        for key in ("device", "device_uuid", "sms", "l2_bytes")
                    ):
                        raise WorkerError("workers must use the same GPU")
                if not header_emitted:
                    layers.emit(
                        {
                            "type": "header",
                            "workers": infos,
                            "vllm_fa2": artifact(args),
                            "candidate": args.candidate,
                            "pairs": args.pairs,
                            "reference_self_pairs": SELF_PAIRS,
                            "calls_per_interval": args.calls,
                            "warmup_replays": args.warmup,
                            "seed": 0,
                            "atol": 2e-2,
                            "rtol": 2e-2,
                            "timing": "CUDA events, us per layer call",
                            "pair_order": "candidate first on even rounds, reference first on odd rounds",
                        }
                    )
                    header_emitted = True
                providers = {}
                geometry = None
                for worker in workers.values():
                    prepared = worker.request("setup", row=row)
                    providers.update(prepared["providers"])
                    if geometry is not None and geometry != prepared["geometry"]:
                        raise WorkerError("worker layer geometry mismatch")
                    geometry = prepared["geometry"]
                correctness = {
                    name: providers[name]["correct"] if name in providers else "N/A"
                    for name in ("candidate", *REFERENCES)
                }
                result.update(
                    **geometry,
                    kvlen=row["ctx"] + row["Q"],
                    providers=providers,
                    correctness=correctness,
                )
                result["op.provider"] = providers["candidate"]["op.provider"]
                result["cute_tuning"] = providers["candidate"]["cute_tuning"]
                stage = "warmup"
                for name in ("candidate", *available):
                    workers["sglang" if name == "sglang" else "main"].request(
                        "warmup", provider=name, calls=args.warmup
                    )
                stage = "measurement"

                def measure(name, calls):
                    value = workers["sglang" if name == "sglang" else "main"].request(
                        "measure", provider=name, calls=calls
                    )["us"]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("timing must be finite and positive")
                    return value

                result.update(
                    paired_timings(
                        measure,
                        available,
                        pairs=args.pairs,
                        calls=args.calls,
                        correctness=correctness,
                    )
                )
                result["status"] = result["market_gate"]
                stage = "teardown"
                for worker in workers.values():
                    worker.request("teardown")
            except Exception as exc:  # noqa: BLE001 - continue rows with fresh worker processes.
                if not header_emitted:
                    layers.emit(
                        {
                            "type": "header",
                            "workers": infos,
                            "status": "ERROR",
                            "error_type": type(exc).__name__,
                        }
                    )
                    header_emitted = True
                result.update(
                    status="ERROR",
                    market_gate="ERROR",
                    error_stage=stage,
                    error_type=type(exc).__name__,
                )
                stop_workers()
            result["worker_generation"] = generation
            layers.emit(result)
            failed |= result["status"] != "PASS"
    finally:
        stop_workers()
    return int(failed)


def main(argv=None):
    args = parse_args(argv)
    return run_worker(args) if args.worker else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
