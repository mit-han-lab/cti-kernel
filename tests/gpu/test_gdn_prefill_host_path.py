# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CuTe host fast path against the legacy path, including scratch ownership."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _cute_prefill, _triton_prefill

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import (
    GEOMETRIES,
    SLOTS,
    TP2_GEOMETRIES,
    _capture,
    _inputs,
    _poison,
    _pools,
    _prefill_case,
    _projected,
    _same,
)


def _reason():
    if not torch.cuda.is_available():
        return "CuTe prefill requires a CUDA GPU"
    if not _cute_prefill.available():
        return "CuTe prefill requires nvidia-cutlass-dsl[cu13]==4.8.0"
    if _triton_prefill.device_type(torch.device("cuda")) is None:
        return "CuTe prefill serves the accepted sm120/sm121 devices"
    if _cute_prefill.load(torch.device("cuda")) is None:
        return "nvidia-cutlass-dsl 4.8.0 or its CUDA libraries cannot import"
    return None


_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
LENGTHS = [[16], [1, 2, 61], [40, 38, 37, 36], [87], [0, 5, 0], [130, 1], [2048], [64, 64, 600]]


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")
    monkeypatch.delenv("CTI_GDN_PREFILL_CUTE", raising=False)
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)


def _op(hk=16, hv=32):
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute"
    assert op._fast is True
    return op


def _pair(hk=16, hv=32):
    fast, legacy = _op(hk, hv), _op(hk, hv)
    legacy._fast = False
    return fast, legacy


def _joint(x):
    # Engine-style capacity and a wider joint projection: neither slice owns storage.
    joint = torch.empty((x.shape[0] + 64, x.shape[1] + 8), dtype=x.dtype, device=x.device)
    projected = joint[: x.shape[0], : x.shape[1]]
    projected.copy_(x)
    return joint, projected


def _out(tokens, hv, mode="aligned"):
    if mode == "none":
        return None
    if mode == "misaligned":
        flat = torch.empty(tokens * hv * 128 + 1, dtype=torch.bfloat16, device="cuda")
        return flat[1 : 1 + tokens * hv * 128].view(tokens, hv * 128)
    return torch.empty((tokens + 64, hv * 128), dtype=torch.bfloat16, device="cuda")[:tokens]


def _metadata(lengths):
    slots = [7, 2, -1, 5][: len(lengths)]
    initial = [True, False, True, True][: len(lengths)]
    return _prefill_case(lengths, slots, initial), slots, initial


def _assert_pools(actual, expected):
    # _same views FP32/BF16 as int32/int16: NaN payloads and signed zeros count.
    for a, b in zip(actual, expected, strict=True):
        assert _same(a, b)


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES + TP2_GEOMETRIES, core=GEOMETRIES))
@pytest.mark.parametrize(
    "length_index", ci_cases(range(len(LENGTHS)), core=[(0,), (1,), (2,), (4,), (5,)])
)
def test_fast_path_bitwise_equals_legacy(hk, hv, length_index):
    _compare_twice(hk, hv, LENGTHS[length_index], "aligned")


@pytest.mark.parametrize("output", ["none", "misaligned"])
def test_fast_path_output_variants_bitwise_equal_legacy(output):
    _compare_twice(16, 32, [1, 2, 61], output)


def _compare_twice(hk, hv, lengths, output):
    fast, legacy = _pair(hk, hv)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 20100)
    _, projected = _joint(x)
    meta, slots, initial = _metadata(lengths)
    read = {s for s, flag in zip(slots, initial, strict=True) if s >= 0 and flag}
    pools = [_poison(p, read, "nan", 20102) for p in _pools(hk, hv, seed=20101, slots=SLOTS)]
    expected_pools = [p.clone() for p in pools]
    out, expected_out = (_out(sum(lengths), hv, output) for _ in range(2))
    with torch.inference_mode():
        for call in range(2):
            if call:
                # Fresh nonempty slots now continue from the first call's committed pools.
                meta[2].copy_(
                    torch.tensor(
                        [
                            flag or (length > 0 and slot >= 0)
                            for flag, length, slot in zip(initial, lengths, slots, strict=True)
                        ],
                        dtype=torch.bool,
                        device="cuda",
                    )
                )
            actual = fast(projected, *weights.args(), *meta, *pools, out=out)
            expected = legacy(projected, *weights.args(), *meta, *expected_pools, out=expected_out)
            if out is not None:
                assert actual is out and expected is expected_out
            assert _same(actual, expected)
            assert torch.isfinite(actual).all()
            _assert_pools(pools, expected_pools)


@pytest.mark.parametrize("hk,hv", [(16, 32), (8, 16)])
@pytest.mark.parametrize("lengths", [[130, 70], [64]])
def test_fast_path_captures_bitwise_equal_legacy(hk, hv, lengths):
    fast, legacy = _pair(hk, hv)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 20200)
    _, projected = _joint(x)
    meta, _, _ = _metadata(lengths)
    pools = _pools(hk, hv, seed=20201, slots=SLOTS)
    expected_pools = [p.clone() for p in pools]
    rows = torch.arange(12, 12 + len(lengths), dtype=torch.int32, device="cuda")
    if len(lengths) == 2:
        rows[-1] = -1
    exports = {"capture_rows": rows}
    with torch.inference_mode():
        actual = fast(projected, *weights.args(), *meta, *pools, **exports)
        expected = legacy(projected, *weights.args(), *meta, *expected_pools, **exports)
    assert _same(actual, expected)
    _assert_pools(pools, expected_pools)  # Includes every export and untouched spare row.


def _layer(seed, rounds):
    fast, legacy = _pair()
    weights, _ = _inputs(16, 32, rounds[0], "ordinary", seed)
    pools = _pools(16, 32, seed=seed + 1, slots=SLOTS)
    calls = []
    for index, lengths in enumerate(rounds):
        _, x = _joint(_projected(sum(lengths), 16, 32, seed=seed + 2 + index))
        meta, _, _ = _metadata(lengths)
        calls.append((x, *weights.args(), *meta))
    return SimpleNamespace(
        fast=fast,
        legacy=legacy,
        pools=pools,
        expected_pools=[p.clone() for p in pools],
        calls=calls,
    )


def test_layers_share_scratch_in_stream_order():
    rounds = [[40, 38, 37, 36], [87], [64, 33]]
    layers = [_layer(20300 + 10 * i, rounds) for i in range(4)]
    with torch.inference_mode():
        for index in range(3):
            for layer in layers:
                layer.y = layer.fast(*layer.calls[index], *layer.pools)
        # No equality assertions or synchronization in either round-robin launch sequence.
        for index in range(3):
            for layer in layers:
                layer.expected = layer.legacy(*layer.calls[index], *layer.expected_pools)
    for layer in layers:
        assert _same(layer.y, layer.expected)
        _assert_pools(layer.pools, layer.expected_pools)
    projected = layers[0].calls[0][0]
    plans = [layer.fast._kernel.plan(projected, 4) for layer in layers]
    assert all(plan is plans[0] for plan in plans)


def test_two_streams_have_separate_scratch():
    layers = [_layer(20400, [[40, 38, 37, 36]]), _layer(20410, [[87]])]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    current = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(current)
    with torch.inference_mode():
        for _ in range(5):
            for layer, stream in zip(layers, streams, strict=True):
                with torch.cuda.stream(stream):
                    layer.y = layer.fast(*layer.calls[0], *layer.pools)
        for stream in streams:
            stream.synchronize()
        for layer in layers:
            for _ in range(5):
                layer.expected = layer.legacy(*layer.calls[0], *layer.expected_pools)
    for layer in layers:
        assert _same(layer.y, layer.expected)
        _assert_pools(layer.pools, layer.expected_pools)
    device = layers[0].calls[0][0].device.index
    keys = [(device, stream.stream_id) for stream in streams]
    assert keys[0] != keys[1]
    assert all(key in _cute_prefill._ARENAS for key in keys)
    assert (
        _cute_prefill._ARENAS[keys[0]].buffer.data_ptr()
        != _cute_prefill._ARENAS[keys[1]].buffer.data_ptr()
    )


@pytest.mark.parametrize("tune", ["", "overlap=1,overlap_min_tokens=0"], ids=["serial", "overlap"])
def test_threads_on_one_stream_do_not_interleave_scratch(tune, monkeypatch):
    # TVM-FFI launches release the GIL, so without the arena lock two host threads on one
    # stream could enqueue A's pipeline, B's pipeline, then A's norm (which would read B's
    # intermediates). Each wrapped launch sequence sleeps with the GIL released, so another
    # thread enters its own sequence unless the call holds the scratch for its whole enqueue.
    if tune:
        monkeypatch.setenv("CTI_GDN_CUTE_TUNE", tune)
    layers = [_layer(20500, [[40, 38, 37, 36]]), _layer(20510, [[87, 64]])]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        # Compiles, and sizes this stream's arena for both plans, before the race (a call
        # that grows the arena gets a new buffer, which is safe but has its own lock).
        for layer in layers:
            layer.y = layer.fast(*layer.calls[0], *layer.pools)
    guard = threading.Lock()
    active = [0, 0]  # current, maximum

    def enter():
        with guard:
            active[0] += 1
            active[1] = max(active[1], active[0])

    def leave():
        with guard:
            active[0] -= 1

    def wrap(launch, first, last):
        def call(*args):
            if first:
                enter()
            launch(*args)
            time.sleep(0.02)
            if last:
                leave()

        return call

    for layer in layers:  # monkeypatch restores each launcher's compiled entry afterwards.
        compiled = layer.fast._kernel._cute.compiled
        projected, count = layer.calls[0][0], layer.calls[0][5].numel() - 1
        key = layer.fast._kernel.plan(projected, count)[2].key
        entry = compiled[key]
        assert isinstance(entry, dict) == bool(tune)
        if isinstance(entry, dict):
            entry = {
                "pipeline": wrap(entry["pipeline"], True, False),
                "norm": wrap(entry["norm"], False, True),
            }
        else:
            entry = wrap(entry, True, True)
        monkeypatch.setitem(compiled, key, entry)
    start, errors = threading.Barrier(len(layers)), []

    def run(layer):
        try:
            with torch.cuda.stream(stream), torch.inference_mode():
                start.wait()
                for _ in range(4):
                    layer.y = layer.fast(*layer.calls[0], *layer.pools)
        except Exception as error:  # noqa: BLE001 - reported on the main thread
            errors.append(error)

    threads = [threading.Thread(target=run, args=(layer,)) for layer in layers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stream.synchronize()
    assert not errors, errors
    with torch.inference_mode():
        for layer in layers:
            for _ in range(5):
                layer.expected = layer.legacy(*layer.calls[0], *layer.expected_pools)
    for layer in layers:
        assert _same(layer.y, layer.expected)
        _assert_pools(layer.pools, layer.expected_pools)
    assert active[1] == 1


def _compiled_count(op):
    return sum(len(c) if isinstance(c, dict) else 1 for c in op._kernel._cute.compiled.values())


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(16, 32)]))
def test_graph_capture_uses_private_scratch_and_matches_eager(hk, hv, monkeypatch):
    op, eager = _op(hk, hv), _op(hk, hv)
    weights, x = _inputs(hk, hv, [100, 28], "ordinary", 20500)
    joint, projected = _joint(x)
    meta = _prefill_case([100, 28], [7, 2], [True, False])
    pools = _pools(hk, hv, seed=20501, slots=SLOTS)
    out = _out(128, hv)
    private = []
    carve = _cute_prefill._carve

    def record_carve(buffer, plan, device=None):
        if buffer is None:  # Per-call allocation, never the stream's shared scratch.
            private.append(plan.id)
        return carve(buffer, plan, device)

    monkeypatch.setattr(_cute_prefill, "_carve", record_carve)
    with torch.inference_mode():
        op.warm_up(max_tokens=256, max_sequences=4, row_stride=joint.stride(0))
        eager.warm_up(max_tokens=256, max_sequences=4, row_stride=joint.stride(0))
        compiled = _compiled_count(op)
        torch.cuda.current_stream().synchronize()
        graph, captured = _capture(lambda: op(projected, *weights.args(), *meta, *pools, out=out))
        assert captured is out
        assert _compiled_count(op) == compiled
        assert len(private) == 1
        for index, (lengths, slots, initial) in enumerate(
            [
                ([64, 64], [9, 3], [False, True]),
                ([1, 127], [5, 7], [True, False]),
            ]
        ):
            if index:
                # Use this same op and caller stream between replays, with different scratch sizes.
                _, other_x = _joint(_projected(151, hk, hv, seed=20510))
                other_meta = _prefill_case([40, 38, 37, 36], [0, 1, 2, 3], [False] * 4)
                other_pools = _pools(hk, hv, seed=20511, slots=SLOTS)
                op(other_x, *weights.args(), *other_meta, *other_pools)
            new_meta = _prefill_case(lengths, slots, initial)
            new_pools = _pools(hk, hv, seed=20520 + index, slots=SLOTS)
            expected_pools = [p.clone() for p in new_pools]
            projected.copy_(_projected(128, hk, hv, seed=20530 + index))
            for dst, src in zip((*meta, *pools), (*new_meta, *new_pools), strict=True):
                dst.copy_(src)
            expected = eager(projected, *weights.args(), *new_meta, *expected_pools)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.current_stream().synchronize()
            assert _same(out, expected)
            _assert_pools(pools, expected_pools)
            assert _compiled_count(op) == compiled
        assert len(private) == 1  # Replay executes no host allocation path.


def test_bound_checks_still_raise(monkeypatch):
    fast, legacy = _pair()
    weights, x = _inputs(16, 32, [64], "ordinary", 20600)
    meta = _prefill_case([64], [7], [True])
    pools = _pools(16, 32, seed=20601, slots=SLOTS)
    expected_pools = [p.clone() for p in pools]
    with torch.inference_mode():
        actual = fast(x, *weights.args(), *meta, *pools)
        expected = legacy(x, *weights.args(), *meta, *expected_pools)
    assert _same(actual, expected)
    _assert_pools(pools, expected_pools)
    before = [p.clone() for p in pools]
    args = (x, *weights.args(), *meta)

    def no_launch(*args, **kwargs):
        pytest.fail("invalid bound arguments reached the launch path")

    with monkeypatch.context() as patch:
        patch.setattr(fast._kernel, "run", no_launch)
        # requires_grad can change without changing a bound tensor's identity or layout.
        for weight in weights.args():
            weight.requires_grad_(True)
            try:
                with pytest.raises(ValueError, match="frozen base weight"):
                    fast(*args, *pools)
            finally:
                weight.requires_grad_(False)
            _assert_pools(pools, before)
        with pytest.raises(TypeError, match="state"):
            fast(*args, pools[0].to(torch.bfloat16), pools[1])
        _assert_pools(pools, before)
        with pytest.raises(ValueError, match="conv_state"):
            fast(*args, pools[0], pools[1][:, :2])
        _assert_pools(pools, before)
        weight = weights.conv_weight
        shape = weight.shape
        saved = weight.clone()
        weight.resize_(weight.numel())
        try:
            with pytest.raises(ValueError, match="conv_weight"):
                fast(*args, *pools)
        finally:
            weight.resize_(shape).copy_(saved)
        _assert_pools(pools, before)
    with torch.inference_mode():
        actual = fast(*args, *pools)
        expected = legacy(*args, *expected_pools)
    assert _same(actual, expected)
    _assert_pools(pools, expected_pools)


def test_unwarmed_capture_still_refused():
    op = _op()
    weights, x = _inputs(16, 32, [64], "ordinary", 20700)
    meta = _prefill_case([64], [7], [True])
    pools = _pools(16, 32, seed=20701, slots=SLOTS)
    tail = (*weights.args(), *meta, *pools)
    with torch.inference_mode():
        op(x, *tail)
        _, unwarmed = _joint(x)  # New stride means a never-warmed compiled variant.
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="once eagerly"):
            with torch.cuda.graph(graph):
                op(unwarmed, *tail)
