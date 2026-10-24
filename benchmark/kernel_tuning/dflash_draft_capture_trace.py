# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT attention capture/replay evidence; run as a module from the repo root."""

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

from tests.gpu._dflash import Fixture

MASKS = [(16, False, None), (16, True, 4096), (8, False, 2048), (4, True, None), (16, True, None)]
ALLOCATION = re.compile(
    r"^(cuda(?:Malloc|Free|HostAlloc)|cuMem(?:Alloc|Free|HostAlloc|Create|Map|Unmap|Release))"
)
SYNC = re.compile(
    r"^(cuda(?:StreamSynchronize|DeviceSynchronize|EventSynchronize|StreamQuery)"
    r"|cu(?:StreamSynchronize|CtxSynchronize|EventSynchronize|StreamQuery))"
)
COPY = re.compile(r"^(?:cuda|cu)Memcpy")
GRAPH_LAUNCH = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_v\d+)?(?:_ptsz|_ptds)?$")
MARKER = "dflash_round"


@contextlib.contextmanager
def diagnostic_stdout():
    """Keep package/JIT/native debug messages off the JSON Lines channel."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def allocation_state():
    stats = torch.cuda.memory_stats()
    keys = ("allocation.all.allocated", "allocated_bytes.all.allocated", "num_device_alloc")
    return {"memory_allocated": torch.cuda.memory_allocated(), **{k: stats[k] for k in keys}}


def profile_summary(events):
    """CPU APIs belong to the marker; async GPU activities span the whole profile."""
    # With GPU work in range the marker also appears as a gpu_user_annotation.
    spans = [
        e
        for e in events
        if e.get("name") == MARKER and e.get("ph") == "X" and e.get("cat") != "gpu_user_annotation"
    ]
    if len(spans) != 1:
        raise ValueError("expected one complete round marker")
    span = spans[0]
    start, end = span["ts"], span["ts"] + span["dur"]
    apis = Counter(
        e["name"]
        for e in events
        if e.get("ph") == "X"
        and start <= e.get("ts", -1) < end
        and (
            e.get("cat") in ("cuda_runtime", "cuda_driver")
            or re.match(r"^(?:cuda[A-Z]|cu[A-Z])", e.get("name", ""))
        )
    )
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    transfers = [e["name"] for e in gpu if e.get("cat") == "gpu_memcpy"]
    kernels = Counter(e["name"] for e in gpu if e.get("cat") == "kernel")
    return {
        "runtime_api_names": sorted(apis),
        "runtime_api_counts": dict(sorted(apis.items())),
        "allocation_apis": [name for name in apis if ALLOCATION.match(name)],
        "sync_apis": [name for name in apis if SYNC.match(name)],
        "memcpy_apis": [name for name in apis if COPY.match(name)],
        "cuda_activity_names": sorted({e["name"] for e in gpu}),
        "memcpy_activities": transfers,
        "kernel_counts": dict(sorted(kernels.items())),
        "kernel_count": sum(kernels.values()),
        "memset_count": sum(e.get("cat") == "gpu_memset" for e in gpu),
    }


# CUgraphNodeType values from the public CUDA driver API.
NODE_KINDS = (
    "KERNEL",
    "MEMCPY",
    "MEMSET",
    "HOST",
    "GRAPH",
    "EMPTY",
    "WAIT_EVENT",
    "EVENT_RECORD",
    "EXT_SEMAS_SIGNAL",
    "EXT_SEMAS_WAIT",
    "MEM_ALLOC",
    "MEM_FREE",
    "BATCH_MEM_OP",
    "CONDITIONAL",
)


def graph_node_counts(handle):
    """Count node kinds of a captured cudaGraph_t through the driver API."""
    import ctypes

    driver = ctypes.CDLL("libcuda.so.1")
    graph = ctypes.c_void_p(handle)
    count = ctypes.c_size_t(0)
    if driver.cuGraphGetNodes(graph, None, ctypes.byref(count)):
        raise RuntimeError("cuGraphGetNodes failed")
    nodes = (ctypes.c_void_p * count.value)()
    if count.value and driver.cuGraphGetNodes(graph, nodes, ctypes.byref(count)):
        raise RuntimeError("cuGraphGetNodes failed")
    counts = Counter()
    for node in nodes:
        kind = ctypes.c_int(-1)
        if driver.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)):
            raise RuntimeError("cuGraphNodeGetType failed")
        counts[NODE_KINDS[kind.value] if 0 <= kind.value < len(NODE_KINDS) else "UNKNOWN"] += 1
    return dict(sorted(counts.items()))


def _read_profile(profiler, directory, name):
    path = directory / name
    profiler.export_chrome_trace(str(path))
    return profile_summary(json.loads(path.read_text())["traceEvents"])


def trace_round(
    b, q, causal, window, *, capacity=4160, warmup=3, provider=None, fused=False
) -> dict:
    if b not in range(1, 5) or (q, causal, window) not in MASKS:
        raise ValueError("expected B1..4 and a supported attention mask")
    if capacity < q + 2 or warmup < 1:
        raise ValueError("capacity must exceed Q+1 and warmup must be positive")
    f = Fixture(b, q, 1, 32, 8, causal, window, c=capacity, provider=provider)
    call = f.op.prepare_and_run if fused else f.op.run

    def run():
        return call(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

    def stage(*args):
        f.stage(*args)
        if not fused:
            # Prepare-once metadata is outside the measured attention call.
            f.op.prepare()

    initial_slots = list(range(1, b + 1))
    if b > 1:
        initial_slots[-1] = -1
    initial = ([min(128, capacity - q)] * b, initial_slots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            stage(*initial)
            run()
    stream.synchronize()
    # Initialize profiler/CUPTI eagerly too; its teardown is outside measured scopes.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities):
        with torch.cuda.stream(stream):
            stage(*initial)
            run()
    stream.synchronize()
    # Keep the cudaGraph_t so its nodes can be enumerated after capture.
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with tempfile.TemporaryDirectory(prefix="dflash-trace-") as temporary:
        directory = Path(temporary)
        with torch.cuda.stream(stream):
            stage(*initial)
            stream.synchronize()
            with torch.profiler.profile(activities=activities) as profiler:
                graph.capture_begin()
                before_capture = allocation_state()
                try:
                    with torch.profiler.record_function(MARKER):
                        run()
                    after_capture = allocation_state()
                finally:
                    graph.capture_end()
        capture = _read_profile(profiler, directory, "capture.json")
        nodes = graph_node_counts(graph.raw_cuda_graph())
        graph.instantiate()
        # B1 cannot mix rows: active -> inactive -> reused active slot.
        replay_slots = list(range(2, b + 2))[::-1]
        if b > 1:
            replay_slots[0] = -1
        patterns = [replay_slots, [-1] * b, list(range(2, b + 2))]
        replays = []
        for turn, slots in enumerate(patterns, 1):
            contexts = [min(256 + turn + row, capacity - q) for row in range(b)]
            values = (contexts, slots, turn)
            with torch.cuda.stream(stream):
                stage(*values)
                stream.synchronize()
                before_replay = allocation_state()
                with torch.profiler.profile(activities=activities) as profiler:
                    with torch.profiler.record_function(MARKER):
                        graph.replay()
                stream.synchronize()
                after_replay = allocation_state()
            replay = _read_profile(profiler, directory, "replay.json")
            replay.update(
                slots=slots,
                contexts=contexts,
                memory_before=before_replay,
                memory_after=after_replay,
            )
            replay["flags"] = {
                "cuda_activity_present": replay["kernel_count"] > 0,
                "kernel_count_matches_graph": replay["kernel_count"] == nodes.get("KERNEL", 0),
                "no_memcpy_activity": not replay["memcpy_activities"],
                "only_graph_launch_apis": bool(replay["runtime_api_names"])
                and all(GRAPH_LAUNCH.fullmatch(name) for name in replay["runtime_api_names"]),
                "no_allocation_apis": not replay["allocation_apis"],
                "no_sync_apis": not replay["sync_apis"],
                "no_memcpy_apis": not replay["memcpy_apis"],
                "allocation_counters_unchanged": before_capture
                == after_capture
                == before_replay
                == after_replay,
            }
            replays.append(replay)
    flags = {
        "capture_apis_present": bool(capture["runtime_api_names"]),
        "capture_no_allocation_apis": not capture["allocation_apis"],
        "capture_no_sync_apis": not capture["sync_apis"],
        "capture_no_memcpy_apis": not capture["memcpy_apis"],
        "graph_kernel_nodes_present": nodes.get("KERNEL", 0) > 0,
        "graph_node_kinds_known": not nodes.get("UNKNOWN", 0),
        "graph_no_forbidden_nodes": not any(
            nodes.get(kind, 0) for kind in ("MEMCPY", "HOST", "MEM_ALLOC", "MEM_FREE", "GRAPH")
        ),
        "capture_allocation_counters_unchanged": before_capture == after_capture,
        "replays_pass": all(all(replay["flags"].values()) for replay in replays),
    }
    return {
        "type": "capture_trace",
        "b": b,
        "q": q,
        "causal": causal,
        "window": window,
        "capacity": capacity,
        "warmup": warmup,
        "fused": fused,
        "provider": f.op.provider,
        "scope": "DRAFT attention inside capture; one graph launch per replay",
        "initial_slots": initial_slots,
        "capture": capture,
        "graph_node_counts": nodes,
        "graph_memset_count": nodes.get("MEMSET", 0),
        "replays": replays,
        "memory_before_capture": before_capture,
        "memory_after_capture": after_capture,
        "flags": flags,
        "pass": all(flags.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=4160)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    passed = True
    for b in range(1, 5):
        for q, causal, window in MASKS:
            try:
                with diagnostic_stdout():
                    result = trace_round(
                        b,
                        q,
                        causal,
                        window,
                        capacity=args.capacity,
                        warmup=args.warmup,
                        fused=args.fused,
                    )
            except (RuntimeError, ValueError, OSError, ImportError, AssertionError) as error:
                result = {
                    "type": "capture_trace",
                    "b": b,
                    "q": q,
                    "causal": causal,
                    "window": window,
                    "error": f"{type(error).__name__}: {str(error)[:300]}",
                    "pass": False,
                }
            print(json.dumps(result), flush=True)
            passed &= result["pass"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
