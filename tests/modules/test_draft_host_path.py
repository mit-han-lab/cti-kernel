# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DraftAttention's bound per-call check accepts exactly what the full check accepts."""

import random
from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention.draft import DraftAttention

DTYPE = torch.bfloat16


def bound_operation(backing, row_stride=64):
    """B=2, Q=4, P=2, Hq/Hkv=4/2, D=16 on CPU; bound buffers live inside ``backing``."""
    op = DraftAttention.__new__(DraftAttention)
    op._device, op._dtype = torch.device("cpu"), DTYPE
    op._b, op._q, op._p, op._hq, op._hk, op._d = 2, 4, 2, 4, 2, 16
    op._query_row_stride = row_stride
    op._metadata = ()
    words = backing.view(torch.int32)
    op._readonly = (words[0:64], words[64:96])
    op._writable = (words[128:160], words[192:256], words[256:256])
    op._bind()
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    op._implementation = SimpleNamespace(run=launch, prepare_and_run=launch)
    return op, calls


def view(backing, offset, shape, stride):
    """A bf16 view at a byte offset of the uint8 backing."""
    return backing[offset:].view(DTYPE).as_strided(shape, stride)


def outcome(fn, *args):
    try:
        fn(*args)
    except (ValueError, AttributeError, TypeError) as error:
        return type(error), str(error)
    return None


def random_operands(rng, backing):
    size = backing.numel()

    slots = [8192, 16384, 24576, 32768]
    rng.shuffle(slots)

    def offset(nbytes):
        # Mostly separate 16-byte aligned slots; sometimes anywhere (overlap, misalignment).
        base = slots.pop() if rng.random() < 0.85 else rng.randrange(0, size - nbytes)
        if rng.random() < 0.05:
            base += 2
        return min(base, size - nbytes) & ~1

    rows, heads, dim = 8, 4, 16
    q_stride = rng.choice([(64, 16, 1), (64, 16, 1), (80, 16, 1), (64, 1, 4), (128, 16, 1)])
    q_shape = rng.choice([(rows, heads, dim), (rows, heads, dim), (rows - 1, heads, dim)])
    q_bytes = (1 + sum((n - 1) * s for n, s in zip(q_shape, q_stride, strict=True))) * 2
    q = view(backing, offset(q_bytes), q_shape, q_stride)
    out_shape = rng.choice([(rows, heads, dim), (rows, heads, dim), (rows, heads, 8)])
    out_stride = (out_shape[1] * out_shape[2], out_shape[2], 1)
    out = view(backing, offset(rows * heads * dim * 2), out_shape, out_stride)
    pages = rng.choice([1, 2, 5])
    tail = rng.choice([(2, 2, 16), (2, 2, 16), (2, 2, 16), (1, 2, 16), (2, 1, 16)])
    contiguous = (tail[0] * tail[1] * tail[2], tail[1] * tail[2], tail[2], 1)
    stride = contiguous if rng.random() < 0.85 else (contiguous[0] * 2, *contiguous[1:])
    nbytes = (pages - 1) * stride[0] * 2 + contiguous[0] * 2
    k = view(backing, offset(nbytes), (pages, *tail), stride)
    v_pages = pages if rng.random() < 0.9 else pages + 1
    v_bytes = (v_pages - 1) * stride[0] * 2 + contiguous[0] * 2
    v = view(backing, offset(v_bytes), (v_pages, *tail), stride)
    if rng.random() < 0.05:
        q = q.float()
    if rng.random() < 0.05:
        k = k.detach().clone().requires_grad_(False).float().to(DTYPE).requires_grad_()
    cache = (k, v) if rng.random() < 0.95 else [k, v]
    return q, cache, out


def run(op, q, cache, out):
    return op.run(q=q, paged_kv_cache=cache, out=out)


@pytest.mark.parametrize("seed", range(8))
def test_bound_check_matches_full_check(seed):
    rng = random.Random(seed)
    backing = torch.zeros(40960, dtype=torch.uint8)
    op, calls = bound_operation(backing)
    accepted = rejected = 0
    for _ in range(1000):
        q, cache, out = random_operands(rng, backing)
        expected = outcome(op._check_operands, q, cache, out)
        assert outcome(op._operands, q, cache, out) == expected
        assert outcome(run, op, q, cache, out) == expected
        accepted += expected is None
        rejected += expected is not None
    assert accepted > 20 and rejected > 100
    assert len(calls) == accepted


def test_plain_operands_skip_the_full_check(monkeypatch):
    backing = torch.zeros(4096, dtype=torch.uint8)
    op, calls = bound_operation(backing)
    monkeypatch.setattr(op, "_check_operands", lambda *args: pytest.fail("full check"))
    q = view(backing, 1024, (8, 4, 16), (64, 16, 1))
    out = view(backing, 2048, (8, 4, 16), (64, 16, 1))
    k = view(backing, 3072, (3, 2, 2, 16), (64, 32, 16, 1))
    v = view(backing, 3584, (3, 2, 2, 16), (64, 32, 16, 1))
    for _ in range(2):
        assert op.run(q=q, paged_kv_cache=(k, v), out=out) is out
    assert len(calls) == 2


def test_size_one_dimension_strides_take_the_full_check():
    # is_contiguous() ignores strides of size-1 dimensions; the bound compare does
    # not, so such views fall back to the full check and are still accepted.
    backing = torch.zeros(4096, dtype=torch.uint8)
    op, calls = bound_operation(backing)
    q = view(backing, 1024, (8, 4, 16), (64, 16, 1))
    out = view(backing, 2048, (8, 4, 16), (64, 16, 1))
    k = view(backing, 3072, (1, 2, 2, 16), (999, 32, 16, 1))
    v = view(backing, 3584, (1, 2, 2, 16), (64, 32, 16, 1))
    assert k.is_contiguous()
    assert op.run(q=q, paged_kv_cache=(k, v), out=out) is out
    assert len(calls) == 1


@pytest.mark.parametrize("where", ["readonly", "writable"])
def test_prepare_rechecks_bound_buffers(where):
    backing = torch.zeros(4096, dtype=torch.uint8)
    op, _ = bound_operation(backing)
    op._implementation.prepare = lambda: None
    op.prepare()
    # A bound buffer moved (set_) onto another one after construction.
    target = op._readonly[0] if where == "writable" else op._writable[0]
    moved = op._writable[0] if where == "writable" else op._readonly[1]
    moved.set_(target.untyped_storage(), target.storage_offset(), moved.shape)
    with pytest.raises(ValueError, match="overlap"):
        op.prepare()


def test_prepare_revalidates_metadata():
    backing = torch.zeros(4096, dtype=torch.uint8)
    op, _ = bound_operation(backing)
    op._implementation.prepare = lambda: None
    value = torch.zeros(4, dtype=torch.int32)
    op._metadata = ((value, (4,), torch.int32, "token_indptr"),)
    op.prepare()
    value.resize_(3)
    with pytest.raises(ValueError, match="token_indptr"):
        op.prepare()
