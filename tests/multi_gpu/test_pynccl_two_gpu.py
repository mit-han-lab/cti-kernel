# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Two-rank pynccl dtype operations, mixed graphs, and byte broadcasts."""

import queue as std_queue
import socket
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
# Odd payload lengths: each is padded by one byte to view the buffer as 16-bit elements.
LENGTHS = (1, 3, 7, 255, 4097, 65537, 1048575)
REPLAYS = 5
WINDOW_BYTES = 1 << 20


class _RawCudaArray:
    """A CUDA array at an arbitrary address, including misaligned typed views."""

    def __init__(self, ptr, count, typestr):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (count,),
            "typestr": typestr,
            "version": 3,
        }


def _misaligned(ptr, count, device, typestr="<f2"):
    tensor = torch.as_tensor(_RawCudaArray(ptr, count, typestr), device=device)
    assert tensor.data_ptr() == ptr, "torch moved the misaligned view"
    return tensor


def _payload(owner, replay, length, padded):
    """The bytes rank `owner` broadcasts on `replay`: random, or all 0xFF (16-bit NaNs)."""
    if replay == 0:
        return torch.full((padded,), 0xFF, dtype=torch.uint8)
    generator = torch.Generator().manual_seed(owner * 1_000_003 + replay * 7919 + length)
    return torch.randint(0, 256, (padded,), generator=generator, dtype=torch.uint8)


def _rejections(comm, rank, errors):
    device = torch.device("cuda", rank)
    other = torch.device("cuda", 1 - rank)
    n = 64

    def f16(count, dev=device):
        return torch.zeros(count, dtype=torch.float16, device=dev)

    backing = torch.zeros(4 * n + 8, dtype=torch.uint8, device=device)
    odd = backing.data_ptr() + 1
    overlap = f16(2 * n)
    cases = {
        "dtype mismatch": (
            torch.zeros(2 * n, dtype=torch.bfloat16, device=device),
            f16(n),
            "dtype",
        ),
        "unsupported dtype": (
            torch.zeros(4 * n, dtype=torch.bool, device=device),
            torch.zeros(2 * n, dtype=torch.bool, device=device),
            "supported dtypes",
        ),
        "source on the other GPU": (f16(2 * n), f16(n, other), "communicator is on"),
        "destination on the other GPU": (f16(2 * n, other), f16(n), "communicator is on"),
        "destination too small": (f16(2 * n - 2), f16(n), "exactly world_size"),
        "destination too large": (f16(2 * n + 2), f16(n), "exactly world_size"),
        "non-contiguous source": (f16(2 * n), f16(2 * n)[::2], "contiguous"),
        "misaligned source": (f16(2 * n), _misaligned(odd, n, device), "aligned"),
        "misaligned destination": (_misaligned(odd, 2 * n, device), f16(n), "aligned"),
        "partial overlap": (overlap[: 2 * n], overlap[1 : n + 1], "overlaps"),
        "other rank's slice": (
            overlap[: 2 * n],
            overlap[(1 - rank) * n : (2 - rank) * n],
            "overlaps",
        ),
    }
    # Every rejection is raised before NCCL is called, so the ranks stay in step.
    for label, (dst, src, fragment) in cases.items():
        try:
            comm.all_gather(dst, src)
        except RuntimeError as error:
            if fragment not in str(error):
                errors.append(f"{label}: unexpected error {type(error).__name__}: {error}")
        else:
            errors.append(f"{label} was not rejected")
        _sync_collective(comm, errors, label)
    torch.cuda.synchronize()
    del backing


def _dtype_inputs(dtype, op, count, seed=0):
    if dtype == torch.float32:
        generator = torch.Generator().manual_seed(20261017 + seed)
        return tuple(torch.randn(count, generator=generator) for _ in range(2))
    if op == "prod":
        values = ((-3, 2, 0, -1, 4), (2, -2, 3, -4, 1))
    elif dtype == torch.int32 and op in ("max", "gather"):
        values = ((-(2**31), -17, 2**31 - 1, -5), (2**31 - 1, -3, -(2**31), -9))
    elif dtype == torch.int64:
        values = ((2**40 + seed, -(2**42), -9, 2**33), (2**39, -(2**41), 2**35, -7))
    else:
        values = ((-71, 13, -5, 100), (7, -31, 19, -200))
    return tuple(
        torch.tensor(row, dtype=dtype).repeat((count + len(row) - 1) // len(row))[:count]
        for row in values
    )


def _check_bits(got, want, label, errors):
    if not torch.equal(got.cpu().view(torch.uint8), want.contiguous().view(torch.uint8)):
        errors.append(f"{label}: result differs bitwise")


def _dtype_rejections(comm, rank, errors):
    device = torch.device("cuda", rank)
    supported = "supported dtypes"
    n = 37
    backing = torch.zeros(8 * n + 8, dtype=torch.uint8, device=device)
    misaligned = _misaligned(backing.data_ptr() + 2, n, device, "<f4")
    bad_destination = _misaligned(backing.data_ptr() + 2, 2 * n, device, "<f4")
    good = torch.zeros(n, dtype=torch.float32, device=device)
    output = torch.zeros(2 * n, dtype=torch.float32, device=device)
    cases = [
        (comm.all_reduce, (good, "unknown"), "Unknown reduce op"),
        (comm.all_reduce, (misaligned, "sum"), "4-byte aligned"),
        (comm.all_gather, (output, misaligned), "4-byte aligned"),
        (comm.all_gather, (bad_destination, good), "4-byte aligned"),
    ]
    for dtype in (torch.int8, torch.uint8, torch.int32, torch.int64):
        tensor = torch.ones(n, dtype=dtype, device=device)
        cases.append(
            (comm.all_reduce, (tensor, "avg"), "avg is not supported for int8/uint8/int32/int64")
        )
    for dtype in (torch.bool,):
        tensor = torch.ones(n, dtype=dtype, device=device)
        destination = torch.empty(2 * n, dtype=dtype, device=device)
        cases.extend(
            [
                (comm.all_reduce, (tensor, "sum"), supported),
                (comm.all_gather, (destination, tensor), supported),
            ]
        )
    for call, args, fragment in cases:
        try:
            call(*args)
        except RuntimeError as error:
            if fragment not in str(error):
                errors.append(f"{fragment}: unexpected error {error}")
        else:
            errors.append(f"{fragment}: call was not rejected")
        _sync_collective(comm, errors, fragment)
    torch.cuda.synchronize()


def _dtype_eager(comm, rank, errors):
    cases = (
        (torch.float32, "sum"),
        (torch.float32, "max"),
        (torch.float32, "min"),
        (torch.int32, "max"),
        (torch.int32, "sum"),
        (torch.int64, "max"),
        (torch.int64, "sum"),
        (torch.float32, "avg"),
        (torch.int32, "prod"),
    )
    for dtype, op in cases:
        itemsize = torch.empty((), dtype=dtype).element_size()
        # Exercise staging and fallback for every new dtype and operation.
        for count in (37, WINDOW_BYTES // itemsize + 17):
            a, b = _dtype_inputs(dtype, op, count)
            if op == "max":
                expected = torch.maximum(a, b)
            elif op == "min":
                expected = torch.minimum(a, b)
            elif op == "prod":
                expected = a * b
            else:
                expected = a + b
                if op == "avg":
                    expected /= 2
            # A sliced input also checks that the caller's data offset is honored.
            storage = torch.full((count + 2,), 73, dtype=dtype, device="cuda")
            tensor = storage[1:-1]
            tensor.copy_((a, b)[rank])
            comm.all_reduce(tensor, op)
            _check_bits(tensor, expected, f"{dtype} {op}, count={count}", errors)
            _check_bits(storage[[0, -1]], torch.full((2,), 73, dtype=dtype), "guards", errors)

    for dtype in (torch.float32, torch.int32, torch.int64):
        a, b = _dtype_inputs(dtype, "gather", 37)
        for inplace in (False, True):
            storage = torch.full((76,), 73, dtype=dtype, device="cuda")
            output = storage[1:-1]
            src = output[rank * 37 : (rank + 1) * 37] if inplace else torch.empty_like(output[:37])
            src.copy_((a, b)[rank])
            comm.all_gather(output, src)
            _check_bits(output, torch.cat((a, b)), f"{dtype} gather, inplace={inplace}", errors)
            _check_bits(storage[[0, -1]], torch.full((2,), 73, dtype=dtype), "guards", errors)


def _dtype_graph(comm, rank, errors):
    stream = torch.cuda.Stream()
    buffers = [
        torch.empty(37, dtype=dtype, device="cuda")
        for dtype in (torch.float32, torch.int32, torch.int64, torch.float32)
    ]
    output = torch.empty(74, dtype=torch.float32, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())

    def stage(replay):
        pairs = [
            _dtype_inputs(torch.float32, "sum", 37, replay),
            # Each rank writes its status; alternate which rank reports poison.
            tuple(
                torch.full((37,), replay + 1 if owner == replay % 2 else 0, dtype=torch.int32)
                for owner in range(2)
            ),
            _dtype_inputs(torch.int64, "max", 37, replay),
            _dtype_inputs(torch.float32, "gather", 37, replay + 1000),
        ]
        with torch.cuda.stream(stream):
            for buffer, pair in zip(buffers, pairs, strict=True):
                buffer.copy_(pair[rank])
            output.fill_(float("nan"))
        return [
            pairs[0][0] + pairs[0][1],
            torch.maximum(*pairs[1]),
            torch.maximum(*pairs[2]),
            torch.cat(pairs[3]),
        ]

    def collectives():
        comm.all_reduce(buffers[0], "sum")
        comm.all_reduce(buffers[1], "max")
        comm.all_reduce(buffers[2], "max")
        comm.all_gather(output, buffers[3])

    def verify(expected, label):
        stream.synchronize()
        for got, want in zip([*buffers[:3], output], expected, strict=True):
            _check_bits(got, want, f"{label}, {got.dtype}", errors)

    expected = stage(100)
    with torch.cuda.stream(stream):
        collectives()
    verify(expected, "graph warmup")
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        collectives()
    for replay in range(REPLAYS):
        expected = stage(replay)
        with torch.cuda.stream(stream):
            graph.replay()
        verify(expected, f"mixed graph replay {replay}")
        expected = stage(replay + 200)
        with torch.cuda.stream(stream):
            collectives()
        verify(expected, f"mixed eager after replay {replay}")
    del graph
    stream.synchronize()
    dist.barrier()


INTEGER_DTYPES = (torch.int8, torch.uint8, torch.int32, torch.int64)
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _sync_collective(comm, errors, label):
    tensor = torch.ones(4, dtype=torch.float32, device="cuda")
    comm.all_reduce(tensor, "sum")
    _check_bits(tensor, torch.full((4,), 2.0), f"after rejection {label}", errors)


def _exact_pair(dtype, count, replay):
    # Nonnegative even integers avoid uint8 overflow and make FP8 products,
    # sums and pre-scaled averages exactly representable, including e5m2.
    rows = ((0, 2, 4, 2), (4, 2, 0, 4))
    return tuple(
        torch.tensor(row, dtype=torch.float32)
        .roll(replay + rank)
        .repeat((count + 3) // 4)[:count]
        .to(dtype)
        for rank, row in enumerate(rows)
    )


def _exact_reduce(pair, dtype, op):
    # Torch CPU has no FP8 arithmetic: widen exactly, quantize intermediate
    # values to the wire dtype, and narrow the final result. Inputs are chosen
    # so each step is exact; avg follows NCCL's pre-multiply then sum semantics.
    work = tuple(t.float() if dtype in FP8_DTYPES else t for t in pair)
    if op == "avg":
        scaled = tuple((t * 0.5).to(dtype) for t in work)
        a, b = (t.float() for t in scaled) if dtype in FP8_DTYPES else scaled
        return (a + b).to(dtype)
    a, b = work
    result = {
        "sum": lambda: a + b,
        "prod": lambda: a * b,
        "max": lambda: torch.maximum(a, b),
        "min": lambda: torch.minimum(a, b),
    }[op]()
    return result.to(dtype)


def _market_matrix(comm, rank, errors, min_capability, *, only_all_reduce=False):
    from ctikernels.pynccl import PYNCCL_DTYPES, PYNCCL_REDUCE_OPS

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for dtype in PYNCCL_DTYPES:
        cases = []
        for op in PYNCCL_REDUCE_OPS:
            if op == "avg" and dtype in INTEGER_DTYPES:
                continue
            if dtype in FP8_DTYPES and min_capability < 90:
                continue
            cases.append(("all_reduce", op, False))
            if not only_all_reduce:
                cases.extend(("reduce_scatter", op, inplace) for inplace in (False, True))
        if not only_all_reduce:
            cases.extend(("all_gather", None, inplace) for inplace in (False, True))
            cases.extend(("broadcast", root, False) for root in (0, 1))
        for kind, op, inplace in cases:
            _market_case(comm, rank, errors, stream, dtype, kind, op, inplace)


def _market_case(comm, rank, errors, stream, dtype, kind, op, inplace):
    label = f"{comm.window_mode()} {dtype} {kind} op/root={op} inplace={inplace}"
    try:
        count = 18 if kind == "reduce_scatter" else 9
        # Nonzero storage offsets and guard elements exercise byte offsets.
        storage = torch.empty(count + 2, dtype=dtype, device="cuda")
        src = storage[1:-1]
        if kind == "all_gather":
            out_storage = torch.empty(2 * count + 2, dtype=dtype, device="cuda")
            dst = out_storage[1:-1]
            if inplace:
                src = dst[rank * count : (rank + 1) * count]
        elif kind == "reduce_scatter":
            dst = (
                src[rank * 9 : (rank + 1) * 9]
                if inplace
                else torch.empty(9, dtype=dtype, device="cuda")
            )
        else:
            dst = src

        def stage(replay):
            pair = _exact_pair(dtype, count, replay)
            with torch.cuda.stream(stream):
                storage.copy_(torch.full((count + 2,), 8, dtype=dtype))
                if kind == "all_gather":
                    out_storage.copy_(torch.full((2 * count + 2,), 8, dtype=dtype))
                src.copy_(pair[rank])
            if kind == "all_gather":
                return torch.cat(pair)
            if kind == "broadcast":
                return pair[op]
            reduced = _exact_reduce(pair, dtype, op)
            return reduced[rank * 9 : (rank + 1) * 9] if kind == "reduce_scatter" else reduced

        def call():
            if kind == "all_reduce":
                comm.all_reduce(src, op)
            elif kind == "reduce_scatter":
                comm.reduce_scatter(dst, src, op)
            elif kind == "all_gather":
                comm.all_gather(dst, src)
            else:
                comm.broadcast(src, op)

        def verify(want, phase):
            stream.synchronize()
            _check_bits(dst, want, f"{label} {phase}", errors)
            _check_bits(
                torch.cat((storage[:1].cpu(), storage[-1:].cpu())),
                torch.full((2,), 8, dtype=dtype),
                f"{label} guards",
                errors,
            )
            if kind == "all_gather":
                _check_bits(
                    torch.cat((out_storage[:1].cpu(), out_storage[-1:].cpu())),
                    torch.full((2,), 8, dtype=dtype),
                    f"{label} gather guards",
                    errors,
                )

        want = stage(0)
        with torch.cuda.stream(stream):
            call()
        verify(want, "eager")
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for replay in (1, 2):
            want = stage(replay)
            with torch.cuda.stream(stream):
                graph.replay()
            verify(want, f"graph replay={replay}")
        del graph
    except Exception as error:  # noqa: BLE001 -- report the precise runtime combination.
        errors.append(f"{label}: {type(error).__name__}: {error}")
    dist.barrier()


def _new_rejections(comm, rank, errors, min_capability):
    n = 8
    src = torch.zeros(2 * n, device="cuda")
    dst = torch.empty(n, device="cuda")
    boolean = torch.zeros(2 * n, dtype=torch.bool, device="cuda")
    backing = torch.empty(128, dtype=torch.uint8, device="cuda")
    misaligned = _misaligned(backing.data_ptr() + 1, n, torch.device("cuda", rank), "<f4")
    cases = [
        ("negative root", lambda: comm.broadcast(src, -1), "root"),
        ("large root", lambda: comm.broadcast(src, 2), "root"),
        ("broadcast bool", lambda: comm.broadcast(boolean, 0), "supported dtypes"),
        (
            "scatter bool",
            lambda: comm.reduce_scatter(boolean[:n], boolean, "sum"),
            "supported dtypes",
        ),
        (
            "scatter too small",
            lambda: comm.reduce_scatter(dst, src[:-1], "sum"),
            "exactly world_size",
        ),
        (
            "scatter too large",
            lambda: comm.reduce_scatter(dst[:-1], src, "sum"),
            "exactly world_size",
        ),
        ("scatter dtype", lambda: comm.reduce_scatter(dst.double(), src, "sum"), "dtype"),
        (
            "scatter partial overlap",
            lambda: comm.reduce_scatter(src[1 : n + 1], src, "sum"),
            "overlaps",
        ),
        (
            "scatter wrong chunk",
            lambda: comm.reduce_scatter(src[(1 - rank) * n : (2 - rank) * n], src, "sum"),
            "overlaps",
        ),
        ("scatter op", lambda: comm.reduce_scatter(dst, src, "unknown"), "Unknown reduce op"),
        ("broadcast alignment", lambda: comm.broadcast(misaligned, 0), "aligned"),
        ("scatter alignment", lambda: comm.reduce_scatter(misaligned, src, "sum"), "aligned"),
        (
            "scatter source alignment",
            lambda: comm.reduce_scatter(dst[:4], misaligned, "sum"),
            "aligned",
        ),
        ("broadcast contiguous", lambda: comm.broadcast(src[::2], 0), "contiguous"),
        ("scatter contiguous", lambda: comm.reduce_scatter(dst[::2], src[:n], "sum"), "contiguous"),
        ("broadcast CPU", lambda: comm.broadcast(torch.zeros(n), 0), "CUDA device"),
        ("scatter CPU", lambda: comm.reduce_scatter(torch.zeros(n), src, "sum"), "CUDA device"),
    ]
    other = torch.zeros(2 * n, device=torch.device("cuda", 1 - rank))
    cases.extend(
        [
            ("broadcast device", lambda: comm.broadcast(other, 0), "communicator is on"),
            (
                "scatter source device",
                lambda: comm.reduce_scatter(dst, other, "sum"),
                "communicator is on",
            ),
            (
                "scatter destination device",
                lambda: comm.reduce_scatter(other[:n], src, "sum"),
                "communicator is on",
            ),
        ]
    )
    for dtype in INTEGER_DTYPES:
        integer = torch.ones(2 * n, device="cuda", dtype=dtype)
        cases.append(
            (
                f"{dtype} scatter avg",
                lambda t=integer: comm.reduce_scatter(t[rank * n : (rank + 1) * n], t, "avg"),
                "avg is not supported",
            )
        )
    if min_capability < 90:
        for dtype in FP8_DTYPES:
            fp8 = torch.zeros(2 * n, device="cuda", dtype=dtype)
            for op in ("sum", "prod", "max", "min", "avg"):
                cases.extend(
                    [
                        (
                            f"{dtype} reduce {op} sm90",
                            lambda t=fp8, op=op: comm.all_reduce(t, op),
                            "capability >= 9.0",
                        ),
                        (
                            f"{dtype} scatter {op} sm90",
                            lambda t=fp8, op=op: comm.reduce_scatter(
                                t[rank * n : (rank + 1) * n], t, op
                            ),
                            "capability >= 9.0",
                        ),
                    ]
                )
    for label, call, fragment in cases:
        try:
            call()
        except RuntimeError as error:
            if fragment not in str(error):
                errors.append(f"{label}: unexpected rejection: {error}")
        else:
            errors.append(f"{label}: did not reject")
        _sync_collective(comm, errors, label)
    # Empty calls validate normally, then return before resolving a stream/NCCL.
    empty = torch.empty(0, device="cuda")
    comm.all_reduce(empty, "sum")
    comm.reduce_scatter(empty, empty, "sum")
    comm.all_gather(empty, empty)
    comm.broadcast(empty, 1)
    _sync_collective(comm, errors, "empty calls")


def _worker(rank, port, queue):
    errors, stats = [], {}
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(seconds=90),
        )

        from ctikernels import init_pynccl
        from ctikernels.pynccl import check_nccl_runtime_version

        stats["nccl_runtime"] = ".".join(map(str, check_nccl_runtime_version()))
        # The default: no staging window, every all-reduce in place.
        comm = init_pynccl(tp_rank=rank, tp_size=2, tp_cpu_group=dist.group.WORLD)
        if comm.device() != rank:
            errors.append(f"communicator device {comm.device()} != {rank}")
        summed = torch.full((1000,), rank + 1.0, dtype=torch.bfloat16, device="cuda")
        comm.all_reduce(summed, "sum")
        if not bool((summed == 3).all()):
            errors.append("in-place all_reduce without a window is wrong")

        capabilities = [None, None]
        major, minor = torch.cuda.get_device_capability()
        dist.all_gather_object(capabilities, major * 10 + minor)
        min_capability = min(capabilities)
        _market_matrix(comm, rank, errors, min_capability)
        _new_rejections(comm, rank, errors, min_capability)
        _rejections(comm, rank, errors)
        dist.barrier()

        windows = (
            (0, False, False),
            (WINDOW_BYTES, False, False),
            (WINDOW_BYTES, True, False),
            (WINDOW_BYTES, True, True),
        )
        for window_bytes, force, null_window in windows:
            dtype_comm = (
                comm
                if window_bytes == 0
                else init_pynccl(
                    tp_rank=rank,
                    tp_size=2,
                    tp_cpu_group=dist.group.WORLD,
                    max_size_bytes=window_bytes,
                    _force_window=force,
                    _test_null_window=null_window,
                )
            )
            window_errors = []
            if not window_bytes:
                expected_mode, expected_reason = "in_place", "max_size_bytes=0"
            elif force:
                expected_mode, expected_reason = "symmetric", "forced"
            else:
                p2p = [None, None]
                dist.all_gather_object(p2p, torch.cuda.can_device_access_peer(rank, 1 - rank))
                expected_mode = "symmetric" if all(p2p) else "in_place"
                expected_reason = "CUDA P2P probe passed" if all(p2p) else "no CUDA P2P"
            # A null registration (symmetric memory unsupported) runs in place.
            null_reason = "registration returned no window"
            if null_window or (
                expected_mode == "symmetric" and null_reason in dtype_comm.window_reason()
            ):
                expected_mode, expected_reason = "in_place", null_reason
            if (
                dtype_comm.window_mode() != expected_mode
                or expected_reason not in dtype_comm.window_reason()
            ):
                window_errors.append(
                    f"window decision: {dtype_comm.window_mode()}/{dtype_comm.window_reason()}"
                )
            expected_bytes = window_bytes if expected_mode == "symmetric" else 0
            if dtype_comm.window_bytes() != expected_bytes or bool(dtype_comm.get_buffer()) != bool(
                expected_bytes
            ):
                window_errors.append("window allocation disagrees with mode")
            if force:
                _market_matrix(
                    dtype_comm, rank, window_errors, min_capability, only_all_reduce=True
                )
            _dtype_rejections(dtype_comm, rank, window_errors)
            _dtype_eager(dtype_comm, rank, window_errors)
            _dtype_graph(dtype_comm, rank, window_errors)
            errors.extend(
                f"window={window_bytes}, forced={force}, null={null_window}: {error}"
                for error in window_errors
            )
            del dtype_comm
        stats["dtype_windows"] = len(windows)

        device = torch.device("cuda", rank)
        stream = torch.cuda.Stream()
        cases = []
        for index, length in enumerate(LENGTHS):
            padded = length + (length & 1)
            dtype = (torch.float16, torch.bfloat16)[index % 2]
            dst = torch.empty(2 * padded, dtype=torch.uint8, device=device)
            inplace = index % 3 == 2
            if inplace:
                src = dst[rank * padded : (rank + 1) * padded]
            else:
                src = torch.empty(padded, dtype=torch.uint8, device=device)
            cases.append((length, padded, dtype, src, dst, inplace))

        def stage(replay):
            with torch.cuda.stream(stream):
                for length, padded, _, src, dst, _ in cases:
                    dst.fill_(0xA5)
                    src.copy_(_payload(rank, replay, length, padded), non_blocking=False)

        def gather():
            for _, _, dtype, src, dst, _ in cases:
                comm.all_gather(dst.view(dtype), src.view(dtype))

        def verify(replay, label):
            stream.synchronize()
            for length, padded, dtype, _, dst, inplace in cases:
                gathered = dst.cpu()
                for root in range(2):
                    got = gathered[root * padded : root * padded + length]
                    want = _payload(root, replay, length, padded)[:length]
                    if not torch.equal(got, want):
                        errors.append(
                            f"{label}: root {root}, {length} bytes, {dtype}, inplace={inplace}"
                        )

        # Eager on the side stream, then one graph captured on it and replayed.
        with torch.cuda.stream(stream):
            stage(1)
            gather()
        verify(1, "eager")
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gather()
        for replay in range(REPLAYS):
            stage(replay)
            with torch.cuda.stream(stream):
                graph.replay()
            verify(replay, f"replay {replay}")
            # Eager calls interleave with replays on the same communicator.
            stage(replay + 100)
            with torch.cuda.stream(stream):
                gather()
            verify(replay + 100, f"eager after replay {replay}")
            dist.barrier()
        stats["graph_cases"] = len(cases)
        stats["replays"] = REPLAYS
        del graph
        torch.cuda.synchronize()
        dist.barrier()
    except Exception:  # noqa: BLE001 -- send worker failures to the parent without terminating peers.
        errors.append(traceback.format_exc())
    finally:
        queue.put((rank, errors, stats))
        if dist.is_initialized():
            dist.destroy_process_group()


def _collect(queue, processes):
    """Gather one result per worker; fail instead of blocking forever if a worker dies first."""
    results = []
    while len(results) < len(processes):
        try:
            results.append(queue.get(timeout=5))
        except std_queue.Empty:
            dead = [
                (rank, p.exitcode)
                for rank, p in enumerate(processes)
                if p.exitcode not in (None, 0)
            ]
            if dead:
                pytest.fail(f"worker exited before reporting (rank, exitcode): {dead}")
    return results


def test_two_gpu_pynccl_byte_broadcast():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(rank, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = _collect(queue, processes)
    for process in processes:
        process.join()
        assert process.exitcode == 0
    failures = []
    for rank, errors, stats in sorted(results):
        print({"rank": rank, **stats})
        failures.extend(f"rank {rank}: {error}" for error in errors)
    assert not failures, "\n".join(failures)
