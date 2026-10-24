# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch
import torch.nn.functional as F

from ctikernels.moe import to_local_topk_ids
from ctikernels.moe._lora_weights import FactorSlices
from ctikernels.moe.lora import (
    MoELoRAWeights,
    fused_experts_lora,
    moe_lora_w2_,
    moe_lora_w13_,
    moe_lora_w13_act_,
    shared_expert_lora,
    shared_lora_w2_,
    shared_lora_w13_,
    shared_lora_w13_act_,
    validate_rank,
)
from ctikernels.moe.torch._lora_reference import oracle

from .._moe_lora_factors import leaves13, packed, shared_a


def _fake_reduce(source, out, *_, topk_weights=None, **__):
    """moe_sum_reduce_triton on CPU: the (weighted) route sum."""
    weights = 1.0 if topk_weights is None else topk_weights[..., None]
    return out.copy_((source.float() * weights).sum(1))


def factors(e=4, r=8, h=24, i=16, adapters=None, shared=False):
    prefix = () if adapters is None else (adapters,)
    rand = lambda *shape: (torch.randn(*prefix, *shape) * 0.1).bfloat16()
    kwargs = {}
    if shared:
        kwargs = {
            "as13": rand(2, r, h),
            "bs13": rand(2, 8, r),
            "as2": rand(r, 8),
            "bs2": rand(h, r),
        }
    return packed(rand(e, 2, r, h), rand(e, 2, i, r), rand(e, r, i), rand(e, h, r), **kwargs)


def patch_base(monkeypatch, modules, **kernels):
    """Stub the base MoE kernels in the modules that bound them."""
    for module in modules:
        for name, kernel in kernels.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, kernel)


@pytest.mark.parametrize("r", [1, 4, 12, 136, 1024])
def test_rank_valid(r):
    assert validate_rank(r) == r


@pytest.mark.parametrize("r", [0, -8, 8.0, True])
def test_rank_invalid(r):
    with pytest.raises(ValueError):
        validate_rank(r)


@pytest.mark.parametrize("adapters", [None, 2])
def test_separate_gate_up_leaves(adapters):
    w = factors(adapters=adapters, shared=True)
    leaves = [t.clone() for t in w.routed]
    lora = MoELoRAWeights.separate(*leaves, *w.shared)
    assert all(a is b for a, b in zip(lora.routed, leaves, strict=True))
    assert lora.rank == 8 and lora.shared == w.shared
    x = torch.randn(2, 24).bfloat16()
    ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    before = torch.randn(4, 32).bfloat16()
    out, ref = before.clone(), before.clone()
    adapter = {"adapter_id": 1} if adapters else {}
    moe_lora_w13_(out, x, lora, ids, scaling=0.5, top_k=2, **adapter)
    moe_lora_w13_(ref, x, w, ids, scaling=0.5, top_k=2, **adapter)
    assert torch.equal(out, ref)
    with pytest.raises(ValueError, match="up_a must have shape"):
        MoELoRAWeights.separate(leaves[0], leaves[1][..., :8].contiguous(), *leaves[2:])
    with pytest.raises(ValueError, match="gate and up A"):
        bad = (w.gate_a, w.up_a[..., :8].contiguous(), w.gate_b, w.up_b)
        moe_lora_w13_(out, x, bad, ids, scaling=0.5, top_k=2, **adapter)
    with pytest.raises(ValueError, match="gate_up_b must have shape|invalid factor dimensions"):
        moe_lora_w13_(out, x, (w.a13, w.b13), ids, scaling=0.5, top_k=2)
    with pytest.raises(ValueError, match="routed weights"):
        moe_lora_w13_(out, x, (w.a13,), ids, scaling=0.5, top_k=2)


@pytest.mark.parametrize("routing", ["mixed", "skewed", "masked", "empty"])
@pytest.mark.parametrize("rank", [8, 16, 32, 64, 96, 128])
def test_reference_independent_fp64(routing, rank):
    torch.manual_seed(123)
    w = factors(r=rank)
    ids = torch.tensor([[0, -1], [0, 2], [4, -9]], dtype=torch.int32)
    if routing == "skewed":
        ids.fill_(1)
    elif routing == "masked":
        ids.fill_(-1)
    elif routing == "empty":
        ids = ids[:0]
    x = torch.randn(ids.shape[0], 24).bfloat16()
    rw = torch.rand(ids.shape, dtype=torch.float32)
    for down in (False, True):
        inp = torch.randn(ids.numel(), 16).bfloat16() if down else x
        before = (torch.randn(ids.numel(), 24 if down else 32) * 0.02).bfloat16()
        a, b = (w.a2[:, None], w.b2[:, None]) if down else (w.a13, w.b13)
        expected, _, saved = oracle(
            before, inp, a, b, ids, scaling=0.7, top_k=2, routed_weights=rw if down else None
        )
        for zdtype in (torch.float32, torch.bfloat16):
            out = before.clone()
            z = torch.full((ids.numel(), 1 if down else 2, rank), -5, dtype=zdtype)
            if down:
                result = moe_lora_w2_(
                    out, inp, w, ids, rw, scaling=0.7, top_k=2, provider="torch", z_out=z
                )
            else:
                result = moe_lora_w13_(
                    out, inp, w, ids, scaling=0.7, top_k=2, provider="torch", z_out=z
                )
            assert result is out
            torch.testing.assert_close(out, expected, atol=0.001, rtol=0.008)
            valid = ((ids >= 0) & (ids < 4)).flatten()
            torch.testing.assert_close(
                z[valid].float(),
                saved[valid].float(),
                atol=0.01 if zdtype == torch.bfloat16 else 1e-6,
                rtol=0.004,
            )
            assert torch.equal(out[~valid], before[~valid])
            assert torch.all(z[~valid] == -5)


def test_adapter_selection_scale_and_zero():
    torch.manual_seed(5)
    w = factors(adapters=2, shared=True)
    ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int64)
    x = torch.randn(2, 24).bfloat16()
    before = torch.randn(4, 32).bfloat16()
    a, b = before.clone(), before.clone()
    moe_lora_w13_(a, x, w, ids, scaling=0.5, top_k=2, adapter_id=1)
    moe_lora_w13_(
        b,
        x,
        (w.gate_a[1], w.up_a[1], w.gate_b[1], w.up_b[1]),
        ids,
        scaling=torch.tensor([0.5]),
        top_k=2,
    )
    assert torch.equal(a, b)
    w.b13.zero_()
    w.b2.zero_()
    for op, inp, cache in (
        (moe_lora_w13_, x, before),
        (moe_lora_w2_, torch.randn(4, 16).bfloat16(), torch.randn(4, 24).bfloat16()),
    ):
        out = cache.clone()
        extra = (torch.rand(2, 2),) if op == moe_lora_w2_ else ()
        op(out, inp, w, ids, *extra, scaling=0.5, top_k=2, adapter_id=1)
        assert torch.equal(out, cache)
    for op in (moe_lora_w13_, moe_lora_w2_):
        extra = (torch.rand(2, 2),) if op == moe_lora_w2_ else ()
        with pytest.raises(NotImplementedError):
            op(before, x, w, ids, *extra, scaling=1.0, top_k=2, token_adapter_ids=ids)
    with pytest.raises(ValueError, match="adapter_id"):
        moe_lora_w13_(before, x, w, ids, scaling=1.0, top_k=2, adapter_id=2)


def test_local_id_mapping():
    ids = torch.tensor([[0, 1, 2, 3, -1, 9]], dtype=torch.int64)
    mapping = torch.tensor([2, -1, 0, 1], dtype=torch.int32)
    assert torch.equal(
        to_local_topk_ids(ids, mapping), torch.tensor([[2, -1, 0, 1, -1, -1]], dtype=torch.int32)
    )
    assert torch.all(to_local_topk_ids(ids, mapping[:0]) == -1)


def routed_oracle(x, w13, w2, ids, rw, w, scale):
    # Independent scalar pair loop; BF16 cache boundaries are part of the contract.
    out = torch.zeros(x.shape, dtype=torch.float64)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if not 0 <= e < w13.shape[0]:
                continue
            c1 = (w13[e].double() @ x[t].double()).bfloat16()
            if w is not None:
                for s in range(2):
                    n = w.b13.shape[-2]
                    delta = w.b13[e, s].double() @ (w.a13[e, s].double() @ x[t].double())
                    c1[s * n : (s + 1) * n] = (
                        c1[s * n : (s + 1) * n].double() + scale * delta
                    ).bfloat16()
            g, u = c1.float().chunk(2)
            c2 = (F.silu(g) * u).bfloat16()
            c3 = ((w2[e].double() @ c2.double()) * rw[t, k].double()).bfloat16()
            if w is not None:
                delta = w.b2[e].double() @ (w.a2[e].double() @ c2.double())
                c3 = (c3.double() + scale * rw[t, k].double() * delta).bfloat16()
            out[t] += c3.double()
    return out.bfloat16()


def test_compositions_and_shared():
    torch.manual_seed(62)
    w = factors(shared=True)
    x = torch.randn(3, 24).bfloat16()
    w13 = (torch.randn(4, 32, 24) * 0.1).bfloat16()
    w2 = (torch.randn(4, 24, 16) * 0.1).bfloat16()
    ids = torch.tensor([[0, -1], [2, 0], [9, 1]], dtype=torch.int32)
    rw = torch.rand(3, 2)
    for lora in (None, w):
        actual = fused_experts_lora(x, w13, w2, rw, ids, lora, scaling=0.5, provider="torch")
        expected = routed_oracle(x, w13, w2, ids, rw, lora, 0.5)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    w.b13.zero_()
    w.b2.zero_()
    assert torch.equal(
        fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5),
        fused_experts_lora(x, w13, w2, rw, ids, None, scaling=0.5),
    )
    sw13, sw2 = w13[0, :16].contiguous(), w2[0, :, :8].contiguous()
    c1 = (x.float() @ sw13.float().T).bfloat16()
    expected1, _, _ = oracle(c1, x, w.as13[None], w.bs13[None], None, scaling=0.5, top_k=1)
    shared_lora_w13_(c1, x, w, scaling=0.5)
    torch.testing.assert_close(c1, expected1)
    g, u = expected1.float().chunk(2, -1)
    c2 = (F.silu(g) * u).bfloat16()
    out = (c2.float() @ sw2.float().T).bfloat16()
    expected, _, _ = oracle(
        out, c2, w.as2[None, None], w.bs2[None, None], None, scaling=0.5, top_k=1
    )
    shared_lora_w2_(out, c2, w.shared, scaling=0.5)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(shared_expert_lora(x, sw13, sw2, w.shared, 0.5), expected)


def test_validation():
    w = factors()
    with pytest.raises(ValueError, match="rank"):
        packed(w.a13, w.b13, w.a2, w.b2, rank=16)
    with pytest.raises(ValueError, match="four shared"):
        packed(w.a13, w.b13, w.a2, w.b2, as13=torch.empty(2, 8, 24))
    out, x = torch.zeros(2, 32, dtype=torch.bfloat16), torch.zeros(1, 24, dtype=torch.bfloat16)
    ids = torch.zeros(1, 2, dtype=torch.int32)
    for kwargs in (
        {"scaling": 1},
        {"scaling": torch.ones(2)},
        {"provider": "bad"},
        {"adapter_id": 1},
        {"block_m": 8},
        {"z_out": torch.empty(2, 8)},
    ):
        options = {"scaling": 1.0, "top_k": 2}
        options.update(kwargs)
        with pytest.raises(ValueError):
            moe_lora_w13_(out, x, w, ids, **options)


def test_fp32_cache_only_for_routed_down():
    """FP32 is accepted only for the routed down cache, and not by an explicit cute."""
    w = factors()
    ids, rw = torch.tensor([[0, -1]], dtype=torch.int32), torch.rand(1, 2)
    x2 = (torch.randn(2, 16) * 0.1).bfloat16()
    cache = torch.randn(2, 24)
    expected = cache.double().clone()
    expected[0] += oracle(
        torch.zeros(2, 24, dtype=torch.bfloat16), x2, w.a2[:, None], w.b2[:, None], ids, scaling=0.5, top_k=2,
        routed_weights=rw,
    )[1][0]  # fmt: skip
    moe_lora_w2_(cache, x2, w, ids, rw, scaling=0.5, top_k=2)
    assert cache.dtype == torch.float32
    torch.testing.assert_close(cache.double(), expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="BF16 cache3"):
        moe_lora_w2_(cache, x2, w, ids, rw, scaling=0.5, top_k=2, provider="cute")
    with pytest.raises(ValueError, match="bfloat16"):
        moe_lora_w13_(torch.zeros(2, 32), torch.zeros(1, 24, dtype=torch.bfloat16), w, ids,
                      scaling=1.0, top_k=2)  # fmt: skip


def test_shared_expert_joins_the_gated_shared_mlp():
    """CPU: shared_expert adds sigmoid(x . w_gate) * shared_expert_lora(x) to the routed sum,
    rounded once; malformed shared weights raise."""
    torch.manual_seed(3)
    w = factors(shared=True)
    rand = lambda *shape: (torch.randn(*shape) * 0.1).bfloat16()
    x, w13, w2 = rand(3, 24), rand(4, 32, 24), rand(4, 24, 16)
    sgu, sd, sg = rand(16, 24), rand(24, 8), rand(1, 24)
    ids, rw = torch.tensor([[0, 3], [1, -1], [2, 2]], dtype=torch.int32), torch.rand(3, 2)
    routed = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5)
    shared = shared_expert_lora(x, sgu, sd, w, 0.5)
    expected = routed.float() + torch.sigmoid(x.float() @ sg.float().T) * shared.float()
    out = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5, shared_expert=(sgu, sd, sg))
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    for bad in ((sgu, sd), (sgu, sd, rand(2, 24)), (sgu, rand(24, 4), sg)):
        with pytest.raises(ValueError):
            fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5, shared_expert=bad)
    with pytest.raises(ValueError, match="frozen"):
        fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5,
                           shared_expert=(sgu, sd, sg.clone().requires_grad_()))  # fmt: skip


def test_zero_adapter_preserves_signed_zero_bits():
    w = factors()
    w.b13.zero_()
    x = torch.ones(1, 24, dtype=torch.bfloat16)
    cache = torch.full((2, 32), -0.0, dtype=torch.bfloat16)
    before = cache.clone()
    moe_lora_w13_(cache, x, w, torch.tensor([[0, -1]], dtype=torch.int32), scaling=1.0, top_k=2)
    assert torch.equal(cache.view(torch.int16), before.view(torch.int16))


@pytest.mark.parametrize(
    "shared,pairs,experts", [(True, 257, 1), (False, 1024, 256), (False, 32768, 128)]
)
def test_prefill_dispatch_metadata(monkeypatch, shared, pairs, experts):

    from ctikernels.moe.cute import _lora_provider as provider

    compiled, launched, alignments = [], [], []

    def compile_fake(*args, **kwargs):
        compiled.append(kwargs)
        return lambda *tensors: launched.append(tensors)

    def align(ids, bm, e):
        alignments.append((bm, e))
        return (
            torch.empty(pairs, dtype=torch.int32),
            torch.empty(pairs // bm, dtype=torch.int32),
            torch.tensor([pairs], dtype=torch.int32),
        )

    monkeypatch.setattr(provider, "_COMPILED", {})
    monkeypatch.setattr(provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(provider, "available", lambda: True)
    monkeypatch.setattr(provider, "compile_kernel", compile_fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(provider, "moe_align_block_size", align)
    top_k = 1 if shared else 8
    out = torch.empty(pairs, 128, dtype=torch.bfloat16)
    x = torch.empty(pairs // top_k, 128, dtype=torch.bfloat16)
    a = FactorSlices(torch.empty(2, experts, 16, 128, dtype=torch.bfloat16).unbind())
    b = FactorSlices(torch.empty(2, experts, 64, 16, dtype=torch.bfloat16).unbind())
    ids = None if shared else torch.zeros(pairs // top_k, top_k, dtype=torch.int32)
    provider.apply_(out, x, a, b, ids, torch.ones(1), top_k)
    # Shared rows use 16-row tiles below 2048 rows; routed rank-16 blocks use 16 rows.
    expected_bm = 16
    assert compiled[0]["block_m"] == expected_bm
    assert compiled[0]["shared"] == shared
    assert compiled[0]["cluster"] == 1
    assert alignments == ([] if shared else [(expected_bm, experts)])
    assert all(t is None for t in launched[0][-3:]) == shared


def test_block_dispatch_thresholds_and_single_chunk():
    from ctikernels.moe.cute._lora_provider import configuration, uses_block_path

    # Measured crossovers: rank <= 16 keeps the pair kernel up to 512 pairs (1024 on
    # small-SM parts); larger ranks switch at 128 on every part.
    assert not uses_block_path(512, 16, 170) and uses_block_path(513, 16, 170)
    assert not uses_block_path(128, 64, 170) and uses_block_path(129, 64, 170)
    assert not uses_block_path(1024, 16, 48) and uses_block_path(1025, 16, 48)
    assert not uses_block_path(128, 128, 48) and uses_block_path(129, 128, 48)
    for pairs in (513, 1024, 2048, 32768, 65536):
        for rank in (8, 16, 32, 64, 96, 128):
            for bm in (16, 32, 64):
                for k, n, slices in ((2048, 512, 2), (512, 2048, 1)):
                    cluster, tile, chunks = configuration(
                        pairs, slices, 132, k, n, aligned=True, block_m=bm, rank=rank
                    )
                    assert (cluster, tile, chunks) == (1, bm, 1)
    # Shared rows take blocks above 16 rows and split N to fill the SMs.
    assert configuration(16, 2, 132, 2048, 512, shared=True, block_m=16, rank=16)[1] == 0
    _, tile, chunks = configuration(17, 2, 132, 2048, 512, shared=True, block_m=16, rank=16)
    assert tile == 16 and chunks > 1


@pytest.mark.parametrize("base_bm", [32, 128])
def test_composition_reuses_or_rebuilds_alignment(monkeypatch, base_bm):

    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(_cute_provider, "available", lambda: True)
    alignments, base_calls, lora_calls = [], [], []

    def align(ids, bm, experts):
        schedule = (
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )
        alignments.append((bm, schedule))
        return schedule

    def base(x, w, out, rw, ids, sorted_ids, experts, padded, *args):
        base_calls.append(sorted_ids)
        out.zero_()

    def lora(out, *args, **kwargs):
        lora_calls.append(kwargs)
        return out

    # Exercise CUDA composition control flow using CPU storage and stubbed
    # kernels. No CUDA context, compiler or alignment extension is involved.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_ops, "moe_lora_w13_", lora)
    monkeypatch.setattr(_ops, "moe_lora_w2_", lora)
    monkeypatch.setattr(_ops, "_silu_and_mul", lambda c1: c1[..., : c1.shape[-1] // 2])
    patch_base(
        monkeypatch,
        (_ops, _cute_provider),
        moe_align_block_size=align,
        fused_moe_kernel_triton=base,
        moe_sum_reduce_triton=_fake_reduce,
    )
    w = factors()
    _ops.fused_experts_lora(
        torch.zeros(3, 24, dtype=torch.bfloat16),
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int32),
        w,
        scaling=1.0,
        provider="cute",
        base_config={"BLOCK_SIZE_M": base_bm},
    )
    assert [bm for bm, _ in alignments] == [base_bm, 16]
    # Launch grids use a trimmed view of each schedule (same storage, same prefix).
    assert all(schedule.data_ptr() == alignments[0][1][0].data_ptr() for schedule in base_calls)
    assert len(lora_calls) == 2
    for kwargs in lora_calls:
        assert kwargs["sorted_token_ids"].data_ptr() == alignments[-1][1][0].data_ptr()
        assert kwargs["block_m"] == alignments[-1][0]


def act_alignment(ids, experts, bm=16):
    """CPU schedule with an excluded remote bucket and pair-count padding."""
    blocks, block_experts = [], []
    for e in range(experts):
        pairs = torch.where(ids.flatten() == e)[0].to(torch.int32)
        padded = torch.full(((pairs.numel() + bm - 1) // bm * bm,), ids.numel(), dtype=torch.int32)
        padded[: pairs.numel()] = pairs
        blocks.append(padded)
        block_experts.extend([e] * (padded.numel() // bm))
    sorted_ids = torch.cat(blocks)
    return {
        "sorted_token_ids": sorted_ids,
        "expert_ids": torch.tensor(block_experts, dtype=torch.int32),
        "num_tokens_post_padded": torch.tensor([sorted_ids.numel()], dtype=torch.int32),
        "block_m": bm,
    }


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("zdtype", [torch.float32, torch.bfloat16])
def test_act_torch_fp32_oracle(shared, zero, zdtype):
    torch.manual_seed(11)
    t, top_k, rank = 7, 1 if shared else 2, 8
    w = factors(shared=True, adapters=2)
    a, b = (w.as13[1, None], w.bs13[1, None]) if shared else (w.a13[1], w.b13[1])
    if zero:
        b.zero_()
    n = b.shape[-2]
    x = torch.randn(t, 24).bfloat16()
    c1 = torch.randn(t * top_k, 2 * n).bfloat16()
    before = c1.clone()
    ids = torch.zeros(t, top_k, dtype=torch.int64)
    if not shared:
        ids[0] = torch.tensor([-1, 1 << 32])
        ids[1] = torch.tensor([2, 3])
    c2 = torch.full((t * top_k, n), -19, dtype=torch.bfloat16)
    expected = c2.clone()
    z = torch.full((t * top_k, 2, rank), -29, dtype=zdtype)
    _, delta, shrink = oracle(c1, x, a, b, ids, scaling=0.7, top_k=top_k, dtype=torch.float32)
    gate, up = (c1.float() + delta).chunk(2, -1)
    valid = ((ids >= 0) & (ids < a.shape[0])).flatten()
    expected[valid] = (F.silu(gate[valid]) * up[valid]).bfloat16()
    options = {"scaling": torch.tensor([0.7]), "adapter_id": 1, "provider": "torch", "z_out": z}
    if shared:
        result = shared_lora_w13_act_(c2, c1, x, w, **options)
    else:
        result = moe_lora_w13_act_(
            c2, c1, x, w, ids, top_k=top_k, **act_alignment(ids, 4), **options
        )
    assert result is c2
    torch.testing.assert_close(c2, expected, atol=0, rtol=0)
    assert torch.equal(c1.view(torch.int16), before.view(torch.int16))
    torch.testing.assert_close(z[valid], shrink[valid].to(zdtype), atol=1e-6, rtol=1e-6)
    assert torch.all(z[~valid] == -29)


def test_act_avoids_intermediate_bf16_rounding():
    # Exact small factors create a gate perturbation below half a BF16 ULP;
    # rounding gate first would erase it, changing the final BF16 product.
    a = torch.zeros(2, 8, 8, dtype=torch.bfloat16)
    b = torch.zeros(2, 8, 8, dtype=torch.bfloat16)
    a[0, 0, 0], b[0, :, 0] = 1, 0.003
    x = torch.ones(1, 8, dtype=torch.bfloat16)
    c1 = torch.ones(1, 16, dtype=torch.bfloat16)
    c1[:, 8:] = 1.015625
    c2 = torch.empty(1, 8, dtype=torch.bfloat16)
    shared_lora_w13_act_(c2, c1, x, (a, b), scaling=1.0, provider="torch")
    gate = c1[:, :8].float() + b[0, :, 0].float()
    expected = (F.silu(gate) * c1[:, 8:].float()).bfloat16()
    rounded = (F.silu(gate.bfloat16().float()) * c1[:, 8:].float()).bfloat16()
    assert torch.equal(c2, expected)
    assert not torch.equal(c2, rounded)


@pytest.mark.parametrize("bad", ["shape", "dtype", "noncontiguous", "alias", "missing", "none"])
def test_act_validation(bad):
    w = factors()
    x = torch.ones(2, 24, dtype=torch.bfloat16)
    c1 = torch.zeros(4, 32, dtype=torch.bfloat16)
    c2 = torch.zeros(4, 16, dtype=torch.bfloat16)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    options = act_alignment(ids, 4)
    if bad == "shape":
        c2 = c2[:2]
    elif bad == "dtype":
        c2 = c2.float()
    elif bad == "noncontiguous":
        c2 = torch.empty(16, 4, dtype=torch.bfloat16).T
    elif bad == "alias":
        c2 = c1.flatten()[:64].view(4, 16)
    elif bad == "missing":
        options["sorted_token_ids"] = None
    else:
        c2 = None
    with pytest.raises(ValueError):
        moe_lora_w13_act_(c2, c1, x, w, ids, scaling=1.0, top_k=2, provider="torch", **options)


@pytest.mark.parametrize("shared", [False, True])
def test_act_empty(shared):
    w = factors(shared=True)
    x = torch.empty(0, 24, dtype=torch.bfloat16)
    c1 = torch.empty(0, 16 if shared else 32, dtype=torch.bfloat16)
    c2 = torch.empty(0, c1.shape[1] // 2, dtype=torch.bfloat16)
    if shared:
        assert shared_lora_w13_act_(c2, c1, x, w, scaling=1.0) is c2
    else:
        ids = torch.empty(0, 2, dtype=torch.int32)
        assert (
            moe_lora_w13_act_(c2, c1, x, w, ids, scaling=1.0, top_k=2, **act_alignment(ids, 4))
            is c2
        )


# The rank-8 factors switch to aligned blocks above 512 pairs on a 132-SM part.
@pytest.mark.parametrize("pairs", [512, 514])
@pytest.mark.parametrize("provider_name", ["cute", "torch"])
def test_composition_activation_dispatch(monkeypatch, pairs, provider_name):

    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    # Exercise the fused activation entry point; split scheduling is tested separately.
    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", 0)
    calls = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_cute_provider, "available", lambda: True)
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(_ops, "moe_lora_w13_", lambda out, *a, **kw: calls.append("add"))
    monkeypatch.setattr(_ops, "moe_lora_w13_act_", lambda out, *a, **kw: calls.append("act"))
    monkeypatch.setattr(_ops, "moe_lora_w2_", lambda out, *a, **kw: None)

    def silu(c1):
        calls.append("silu")
        return torch.zeros(c1.shape[0], c1.shape[1] // 2, dtype=c1.dtype)

    monkeypatch.setattr(_ops, "_silu_and_mul", silu)
    patch_base(
        monkeypatch,
        (_ops, _cute_provider),
        moe_align_block_size=lambda ids, bm, e: tuple(
            act_alignment(ids, e, bm)[key]
            for key in ("sorted_token_ids", "expert_ids", "num_tokens_post_padded")
        ),
        fused_moe_kernel_triton=lambda x, w, out, *a: out.zero_(),
        moe_sum_reduce_triton=_fake_reduce,
    )
    w = factors()
    _ops.fused_experts_lora(
        torch.zeros(pairs // 2, 24, dtype=torch.bfloat16),
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(pairs // 2, 2),
        torch.zeros(pairs // 2, 2, dtype=torch.int32),
        w,
        scaling=1.0,
        provider=provider_name,
        overlap=False,
    )
    assert calls == (["act"] if pairs > 512 and provider_name != "torch" else ["add", "silu"])


@pytest.mark.parametrize("pairs", [16, 17])
def test_shared_composition_activation_dispatch(monkeypatch, pairs):
    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    calls = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(_ops, "shared_lora_w13_", lambda out, *a, **kw: calls.append("add"))

    def activate(c2, *args, **kwargs):
        calls.append("act")
        c2.zero_()

    monkeypatch.setattr(_ops, "shared_lora_w13_act_", activate)
    monkeypatch.setattr(_ops, "shared_lora_w2_", lambda out, *a, **kw: None)
    shared_expert_lora(
        torch.zeros(pairs, 24, dtype=torch.bfloat16),
        torch.zeros(16, 24, dtype=torch.bfloat16),
        torch.zeros(24, 8, dtype=torch.bfloat16),
        factors(shared=True),
        1.0,
        provider="cute",
    )
    assert calls == (["act"] if pairs > 16 else ["add"])


def test_act_zero_adapter_signed_zeros():
    w = factors(shared=True)
    w.bs13.zero_()
    x = torch.ones(1, 24, dtype=torch.bfloat16)
    c1 = torch.full((1, 16), -0.0, dtype=torch.bfloat16)
    c1[:, 8:] = 1
    c2 = torch.empty(1, 8, dtype=torch.bfloat16)
    shared_lora_w13_act_(c2, c1, x, w, scaling=1.0, provider="torch")
    assert torch.all(c2.view(torch.int16) == -32768)


@pytest.fixture
def overlap_cuda(monkeypatch):
    """CPU storage, real composition, and observable CUDA ordering primitives."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    calls, recorded, streams = [], [], []
    chunks = []
    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", 0)

    class Stream:
        def __init__(self, device=None):
            self.name = "side"
            streams.append(self)

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    main = Stream()
    main.name = "main"
    streams.clear()
    active = [main]

    class Event:
        def record(self, stream):
            assert active[0] is stream
            calls.append((stream.name, "record", self))

    @contextmanager
    def stream_context(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    def projection(out, x, a, b, ids, scale, top_k, rw, z, **kwargs):
        stage = kwargs["stage"]
        down = kwargs["down"]
        calls.append((active[0].name, stage, down))
        chunks.append((stage, down, kwargs["chunks"]))
        if stage != "full":
            assert z.dtype == torch.bfloat16
            assert z.shape == (out.shape[0], 1 if down else 2, a.shape[-2])
        if kwargs["activated"] is not None:
            kwargs["activated"].zero_()
        if stage == "shrink" and active[0] is not main:
            assert any(tensor is z and stream is active[0] for tensor, stream in recorded)

    def base(x, w, out, rw, ids, sorted_ids, experts, padded, weighted, top_k, *args):
        assert active[0] is main
        calls.append(("main", "base", top_k == 1))  # FC2 reads one row per pair
        out.zero_()

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda t, s: recorded.append((t, s)))
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: active[0])
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from ctikernels.moe import _lora_common

    monkeypatch.setattr(_lora_common, "_STREAMS", {})
    monkeypatch.setattr(_cute_provider, "available", lambda: True)
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(_cute_provider, "apply_", projection)
    monkeypatch.setattr(
        _ops, "_silu_and_mul", lambda c: torch.zeros(c.shape[0], c.shape[1] // 2).bfloat16()
    )
    patch_base(
        monkeypatch,
        (_ops, _cute_provider),
        moe_align_block_size=lambda ids, bm, e: tuple(
            act_alignment(ids, e, bm)[key]
            for key in ("sorted_token_ids", "expert_ids", "num_tokens_post_padded")
        ),
        fused_moe_kernel_triton=base,
        moe_sum_reduce_triton=_fake_reduce,
    )
    return SimpleNamespace(
        calls=calls, recorded=recorded, streams=streams, main=main, chunks=chunks
    )


@pytest.mark.parametrize("overlap", [None, False, True])
@pytest.mark.parametrize("provider_name", ["cute", "torch"])
@pytest.mark.parametrize("pairs", [512, 514])
def test_overlap_dispatch_and_event_order(overlap_cuda, overlap, provider_name, pairs):
    state = overlap_cuda
    x = torch.zeros(pairs // 2, 24, dtype=torch.bfloat16)
    ids = torch.zeros(pairs // 2, 2, dtype=torch.int32)
    scale = torch.tensor([0.5])
    fused_experts_lora(
        x,
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(pairs // 2, 2),
        ids,
        factors(),
        scaling=scale,
        provider=provider_name,
        overlap=overlap,
    )
    # Overlap is opt-in: only an explicit True forks the shrinks.
    enabled = pairs > 512 and overlap is True and provider_name != "torch"
    if not enabled:
        assert not state.streams and not state.recorded
        assert not any(call[1] in ("shrink", "expand", "record", "wait") for call in state.calls)
        return
    assert len(state.streams) == 1  # Both projections reuse the device stream.
    # Each fork/join uses distinct events and brackets its own GEMM.
    assert len(state.calls) == 14
    for offset, down in ((0, False), (7, True)):
        fork, wait, shrink, done, base, join, expand = state.calls[offset : offset + 7]
        assert fork[:2] == ("main", "record") and wait == ("side", "wait", fork[2])
        assert shrink == ("side", "shrink", down)
        assert done[:2] == ("side", "record") and join == ("main", "wait", done[2])
        assert base == ("main", "base", down) and expand == ("main", "expand", down)
        assert fork[2] is not done[2]
    assert len({call[2] for call in state.calls if call[1] == "record"}) == 4
    for tensor in (x, ids, scale):
        assert any(t.data_ptr() == tensor.data_ptr() for t, _ in state.recorded)


def test_overlap_stream_capture_warmup(overlap_cuda, monkeypatch):
    from ctikernels.moe import _lora_common as _cute_provider

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm up MoE LoRA overlap"):
        _cute_provider.overlap_stream(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    side = _cute_provider.overlap_stream(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert _cute_provider.overlap_stream(torch.device("cpu")) is side


@pytest.mark.parametrize(
    "stage,z_dtype,bm",
    [
        ("bad", None, 16),
        ("shrink", None, 16),
        ("expand", torch.float32, 16),
        ("expand", torch.bfloat16, 0),
        ("shrink", torch.float32, 16),
    ],
)
def test_split_compile_validation(stage, z_dtype, bm):
    from ctikernels.moe.cute._lora_provider import compile_kernel

    with pytest.raises(ValueError):
        compile_kernel(24, 16, 8, 2, 2, 4, (12, 0), stage=stage, z_dtype=z_dtype, block_m=bm)


@pytest.mark.parametrize("block13,block2", [(True, False), (False, True)])
def test_overlap_decided_per_projection(overlap_cuda, monkeypatch, block13, block2):
    from ctikernels.moe import lora as _ops

    monkeypatch.setattr(
        _ops,
        "_uses_block_projection",
        lambda *args, down=False, **kwargs: block2 if down else block13,
    )
    fused_experts_lora(
        torch.zeros(129, 24, dtype=torch.bfloat16),
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(129, 2),
        torch.zeros(129, 2, dtype=torch.int32),
        factors(),
        scaling=0.5,
        provider="cute",
        overlap=True,
    )
    stages = [call for call in overlap_cuda.calls if call[1] in ("shrink", "expand")]
    assert stages == [("side", "shrink", block2), ("main", "expand", block2)]


@pytest.mark.parametrize("overlap", [None, False, True])
def test_overlap_cpu_does_not_touch_cuda(monkeypatch, overlap):
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    def unexpected(*args, **kwargs):
        pytest.fail("CPU composition must not query CUDA or allocate streams")

    from ctikernels.moe import lora as _ops

    monkeypatch.setattr(_ops, "overlap_stream", unexpected)
    monkeypatch.setattr(_cute_provider, "device_info", unexpected)
    x = torch.ones(3, 24, dtype=torch.bfloat16)
    w13, w2 = torch.ones(4, 32, 24).bfloat16(), torch.ones(4, 24, 16).bfloat16()
    ids, rw, w = torch.zeros(3, 2, dtype=torch.int32), torch.ones(3, 2), factors()
    expected = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5, overlap=False)
    actual = fused_experts_lora(x, w13, w2, rw, ids, w, scaling=0.5, overlap=overlap)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("chunks", [None, 1, 2, 4, 8])
def test_split_stage_dispatch_cache(monkeypatch, chunks):
    from ctikernels.moe.cute import _lora_provider as provider

    compiled, launched = [], []

    def compile_fake(*args, **kwargs):
        compiled.append(kwargs)
        return lambda *args: launched.append(kwargs["stage"])

    monkeypatch.setattr(provider, "_COMPILED", {})
    monkeypatch.setattr(provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(provider, "available", lambda: True)
    monkeypatch.setattr(provider, "compile_kernel", compile_fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out, x = torch.empty(4, 32).bfloat16(), torch.empty(2, 24).bfloat16()
    w = factors()
    ids, z = torch.zeros(2, 2, dtype=torch.int32), torch.empty(4, 2, 8).bfloat16()
    # Activation forces the same BM on all stages even below the auto threshold.
    act = torch.empty(4, 16).bfloat16()
    for stage in ("full", "shrink", "expand", "shrink", "expand"):
        provider.apply_(
            out,
            x,
            FactorSlices((w.gate_a, w.up_a)),
            FactorSlices((w.gate_b, w.up_b)),
            ids,
            torch.ones(1),
            2,
            z_out=z,
            activated=act,
            stage=stage,
            chunks=chunks if stage == "expand" else None,
            **act_alignment(ids, 4),
        )
    assert [kw["stage"] for kw in compiled] == ["full", "shrink", "expand"]
    assert all(kw["block_m"] == 16 and kw["z_dtype"] == torch.bfloat16 for kw in compiled)
    assert [kw["chunks"] for kw in compiled] == [1, 1, chunks or 1]
    assert launched == ["full", "shrink", "expand", "shrink", "expand"]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    provider.apply_(
        out,
        x,
        FactorSlices((w.gate_a, w.up_a)),
        FactorSlices((w.gate_b, w.up_b)),
        ids,
        torch.ones(1),
        2,
        z_out=z,
        activated=act,
        stage="expand",
        chunks=chunks,
        **act_alignment(ids, 4),
    )
    with pytest.raises(RuntimeError, match="warm up this MoE LoRA shape"):
        provider.apply_(
            out,
            x,
            FactorSlices((w.gate_a, w.up_a)),
            FactorSlices((w.gate_b, w.up_b)),
            ids,
            torch.ones(1),
            2,
            z_out=z,
            activated=act,
            stage="expand",
            chunks=chunks,
            **act_alignment(ids, 4, bm=32),
        )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("empty", [False, True])
def test_base_weights_must_be_frozen(shared, which, empty, monkeypatch):
    from ctikernels.moe import lora as _ops

    x = torch.zeros(0 if empty else 2, 24, dtype=torch.bfloat16)
    base = [torch.zeros(16, 24, dtype=x.dtype), torch.zeros(24, 8, dtype=x.dtype)]
    names = ("w_gate_up", "w_down") if shared else ("w13", "w2")
    if not shared:
        base = [t.unsqueeze(0) for t in base]
    base[which].requires_grad_(True)

    def unexpected(*args, **kwargs):
        pytest.fail("base validation must precede projection or allocation")

    monkeypatch.setattr(_ops, "_uses_block_projection", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "empty_like", unexpected)
    # The decorated compositions already run under no_grad; an outer context
    # additionally confirms that validation reads the tensor attribute itself.
    with (
        torch.no_grad(),
        pytest.raises(
            ValueError, match=f"base expert weights are frozen; {names[which]} requires_grad"
        ),
    ):
        if shared:
            shared_expert_lora(x, *base, None, 0.5)
        else:
            ids = torch.zeros(x.shape[0], 1, dtype=torch.int32)
            fused_experts_lora(
                x, *base, torch.ones_like(ids, dtype=torch.float32), ids, None, scaling=0.5
            )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("which", range(4))
def test_grad_lora_factors_remain_accepted(shared, which):
    p = factors(shared=True)
    w = shared_a(p.a13[:, 0].contiguous(), p.b13, p.a2, p.b2, *p.shared)
    names = (
        ("as13", "bs13", "as2", "bs2") if shared else ("gate_up_a", "gate_up_b", "down_a", "down_b")
    )
    getattr(w, names[which]).requires_grad_(True)
    x = torch.randn(2, 24).bfloat16()
    if shared:
        result = shared_expert_lora(
            x,
            torch.zeros(16, 24, dtype=x.dtype),
            torch.zeros(24, 8, dtype=x.dtype),
            w.shared,
            0.5,
            provider="torch",
        )
    else:
        ids = torch.zeros(2, 1, dtype=torch.int32)
        result = fused_experts_lora(
            x,
            torch.zeros(4, 32, 24, dtype=x.dtype),
            torch.zeros(4, 24, 16, dtype=x.dtype),
            torch.ones(2, 1),
            ids,
            w,
            scaling=0.5,
            provider="torch",
        )
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert getattr(w, names[which]).requires_grad


@pytest.mark.parametrize(
    "capability,sms", [((12, 0), 170), ((12, 0), 188), ((12, 1), 48), ((12, 0), 100)]
)
@pytest.mark.parametrize("rows", [1, 16, 17, 128, 512, 2048, 4096])
@pytest.mark.parametrize("rank", [8, 64, 128])
def test_shared_dense_default_policy(monkeypatch, capability, sms, rows, rank):
    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", None)
    monkeypatch.setattr(_shared_policy, "SHARED_TABLE", {})
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: (capability, sms))
    expected = "cute" if rows <= 16 else "library" if rank >= 128 or rows >= 4096 else "dense"
    assert _ops._shared_choice(torch.empty(0), rows, rank, 2048, 512, 2) == (expected, None)


@pytest.mark.parametrize("mode", ["cute", "split", "library", "dense"])
@pytest.mark.parametrize(
    "capability,sms,device",
    [((12, 1), 48, "gb10"), ((12, 0), 180, "rtx_pro_6000"), ((12, 0), 179, "rtx_5090")],
)
def test_shared_policy_table_and_override(monkeypatch, mode, capability, sms, device):
    from ctikernels.moe.cute import _lora_shared_table as policy

    config = (32, 8, 16, 64, 2) if mode == "dense" else None
    entries = ((512, mode, config), (None, "library", None))
    monkeypatch.setattr(policy, "SHARED_MODE_OVERRIDE", None)
    monkeypatch.setattr(
        policy,
        "SHARED_TABLE",
        {
            device: {
                (2, 2048, 512, 64): entries,
                (2, 2048, 512, 128): ((None, "split", None),),
                (1, 2048, 512, 8): ((None, "cute", None),),
            }
        },
    )
    choose = lambda rows, rank, k=2048, n=512, slices=2: policy.shared_choice(
        rows, rank, k, n, slices, capability, sms
    )
    assert choose(512, 64) == (mode, config)  # Exact rank wins over rank 128.
    assert choose(512, 24) == (mode, config)  # Smallest matching larger rank.
    assert choose(512, 96) == ("split", None)
    assert choose(513, 64) == ("library", None)
    assert choose(10**9, 64) == ("library", None)
    assert choose(17, 64, k=1024) == ("dense", None)
    assert choose(17, 64, n=256) == ("dense", None)
    assert choose(17, 64, slices=1) == ("dense", None)
    assert choose(16, 256) == ("cute", None)
    assert choose(17, 256) == ("library", None)  # Untabulated rank >= 128.
    assert policy.shared_choice(17, 64, 2048, 512, 2, (9, 0), sms) == ("dense", None)
    monkeypatch.setattr(policy, "SHARED_MODE_OVERRIDE", mode)
    assert choose(10**9, 64) == (mode, None)
    monkeypatch.setattr(policy, "SHARED_MODE_OVERRIDE", "invalid")
    with pytest.raises(ValueError, match="shared mode"):
        choose(1, 8)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("zdtype", [None, torch.bfloat16, torch.float32])
def test_dense_dispatch_factor_views(monkeypatch, down, zdtype):
    from types import SimpleNamespace

    from ctikernels.moe.cute import _lora_provider as _cute_provider
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", None)
    config = (16, 1, 32, 64, 2)
    monkeypatch.setattr(
        _shared_policy,
        "SHARED_TABLE",
        {"rtx_5090": {(1 if down else 2, 24, 16, 8): ((None, "dense", config),)}},
    )
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 170))
    calls = []

    def apply(out, x, a, b, scale, **kwargs):
        calls.append((out, x, a, b, scale, kwargs))
        return out

    dense = SimpleNamespace(dense=SimpleNamespace(dense_apply_=apply))
    monkeypatch.setattr(_cute_provider, "kernels", lambda: dense)
    t, k, n, rank, slices = 17, 24, 16, 8, 1 if down else 2
    a = torch.zeros((rank, k) if down else (slices, rank, k), dtype=torch.bfloat16)
    b = torch.zeros((n, rank) if down else (slices, n, rank), dtype=torch.bfloat16)
    x, out = torch.zeros(t, k, dtype=a.dtype), torch.zeros(t, slices * n, dtype=a.dtype)
    z = None if zdtype is None else torch.empty(t, slices, rank, dtype=zdtype)
    scale = torch.tensor([0.7])
    fn = shared_lora_w2_ if down else shared_lora_w13_
    assert fn(out, x, (a, b), scaling=scale, provider="cute", z_out=z) is out
    assert len(calls) == 1
    call = calls[0]
    assert call[0] is out and call[1] is x
    assert call[2].shape == (slices * rank, k) and call[2].data_ptr() == a.data_ptr()
    assert call[3].shape == (slices, n, rank) and call[3].data_ptr() == b.data_ptr()
    assert call[4].data_ptr() == scale.data_ptr()
    assert call[5]["rank"] == rank and call[5]["slices"] == slices and call[5]["z_out"] is z
    assert call[5]["config"] == config


@pytest.mark.parametrize("t,rank,stage", [(15, 8, "full"), (17, 16, "expand")])
def test_dense_override_leaves_activation_dispatch(monkeypatch, t, rank, stage):
    from ctikernels.moe.cute import _lora_provider as _cute_provider
    from ctikernels.moe.cute import _lora_shared_table as _shared_policy

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_shared_policy, "SHARED_MODE_OVERRIDE", "dense")
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 170))
    calls = []
    monkeypatch.setattr(_cute_provider, "apply_", lambda *args, **kwargs: calls.append(kwargs))
    x = torch.zeros(t, 24, dtype=torch.bfloat16)
    c1, c2 = torch.zeros(t, 16, dtype=x.dtype), torch.zeros(t, 8, dtype=x.dtype)
    shared_lora_w13_act_(
        c2, c1, x, factors(r=rank, shared=True), scaling=torch.tensor([0.7]), provider="cute"
    )
    assert len(calls) == 1
    assert calls[0]["stage"] == stage and calls[0]["activated"] is c2


@pytest.mark.parametrize("rows", [16, 512, 8192])
@pytest.mark.parametrize("k", [256, 512, 2048, 4096])
@pytest.mark.parametrize("sms", [48, 170, 188])
def test_dense_probe_grid(rows, k, sms):
    import itertools

    from benchmark.kernel_tuning.moe_lora_shared_probe import dense_candidates, expand_candidates

    default = (32, 3, 32, 128, 2)
    configs = dense_candidates(rows, k, sms, default)
    tiles = (k + 63) // 64
    expected = {default} | {
        (bm, sk, 32, 128, stages)
        for bm, sk, stages in itertools.product((16, 32, 64), (1, 2, 4, 8, 16, 32), (2, 3, 4))
        if sk <= tiles
        and (rows + bm - 1) // bm * sk <= 32 * sms
        and (stages == 2 or (tiles + sk - 1) // sk >= stages - 1)
    }
    assert configs[0] == default
    assert len(configs) == len(set(configs))
    assert set(configs) == expected
    assert set(expand_candidates(default)) == {
        (32, 3, ebm, bn, 2) for ebm, bn in itertools.product((16, 32, 64), (64, 128))
    }


@pytest.fixture
def dense_host(monkeypatch):
    """Exercise host policy/allocation with only DSL decorators and types stubbed."""
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import ctikernels.moe.cute

    cute = SimpleNamespace(jit=lambda f: f, kernel=lambda f: f, Tensor=object, Pointer=object)
    driver = SimpleNamespace(CUstream=object)
    bindings = SimpleNamespace(driver=driver)
    modules = {
        "cuda": SimpleNamespace(bindings=bindings),
        "cuda.bindings": bindings,
        "cuda.bindings.driver": driver,
        "cutlass": SimpleNamespace(
            cute=cute, BFloat16=object, Float32=object, Int32=object, Int64=object, const_expr=bool
        ),
        "cutlass.cute": cute,
        "cutlass.runtime": SimpleNamespace(make_fake_stream=None),
        "cutlass.utils": SimpleNamespace(SmemAllocator=object),
        "ctikernels.moe.cute._lora_block": SimpleNamespace(
            **dict.fromkeys(("add_pair", "copy16", "load_a", "load_b", "mma", "store16"))
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(ctikernels.moe.cute.__file__).with_name("_lora_dense.py")
    spec = importlib.util.spec_from_file_location("ctikernels.moe.cute._dense_host_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "rows,k,n,rank,slices,sms,expected",
    [
        (512, 512, 2048, 64, 1, 170, (64, 8, 32, 128, 2)),
        (512, 2048, 512, 128, 2, 170, (32, 22, 32, 64, 2)),
        (1024, 2048, 512, 64, 2, 48, (64, 6, 32, 128, 3)),
        (128, 2048, 512, 16, 2, 170, (64, 32, 16, 64, 2)),
        (4096, 2048, 512, 16, 2, 48, (64, 2, 64, 128, 4)),
    ],
)
def test_dense_launch_policy(dense_host, rows, k, n, rank, slices, sms, expected):
    config = dense_host.dense_config(rows, k, n, rank, slices, sms)
    assert config == expected
    bm, sk, _, _, stages = config
    assert (
        dense_host.DenseShrink(k, rank * slices, bm, sk, stages=stages).smem_bytes + 1024
        <= 99 * 1024
    )


def test_dense_resource_guards_and_legacy_override(dense_host):
    dense_host.DENSE_OVERRIDE["config"] = (16, 8, 32, 128)
    assert dense_host.dense_config(512, 512, 2048, 64, 1, 170) == (16, 8, 32, 128, 2)
    with pytest.raises(ValueError, match="99 KiB"):
        dense_host.DenseShrink(2048, 256, 32, 8, stages=3)
    with pytest.raises(ValueError, match="64 FP32"):
        dense_host.DenseShrink(2048, 256, 64, 8)
    with pytest.raises(ValueError, match="stages"):
        dense_host.DenseShrink(512, 8, 16, 1, stages=5)


@pytest.mark.parametrize("split", [1, 8])
@pytest.mark.parametrize("legacy", [False, True])
def test_dense_compact_workspace_and_config_cache(dense_host, monkeypatch, split, legacy):
    provider = dense_host._lora_common
    monkeypatch.setattr(provider, "device_info", lambda device: ((12, 0), 170))
    monkeypatch.setattr(provider, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    allocations, launches, compiles = [], [], []
    empty, zeros = torch.empty, torch.zeros

    def allocate(kind, factory, *args, **kwargs):
        allocations.append(kind)
        return factory(*args, **kwargs)

    def shrink(x, a, ws):
        launches.append(ws)
        assert ws.shape == (17, 16) and ws.dtype == torch.float32
        if split > 1:
            assert torch.count_nonzero(ws) == 0
        ws.fill_(7)

    def expand(ws, b, out, scale, z):
        assert ws is launches[-1] and torch.all(ws == 7)

    def compile_kernel(*args):
        compiles.append(args)
        return shrink, expand

    x, a, b, out = [empty(shape) for shape in ((17, 512), (16, 512), (2, 64, 8), (17, 128))]
    scale = torch.ones(1)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: allocate("empty", empty, *a, **kw))
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: allocate("zeros", zeros, *a, **kw))
    monkeypatch.setattr(dense_host, "compile_dense", compile_kernel)
    config = (16, split, 32, 128)
    if not legacy:
        config += (3,)
    for _ in range(2):
        assert dense_host.dense_apply_(out, x, a, b, scale, rank=8, slices=2, config=config) is out
    assert allocations == ["empty" if split == 1 else "zeros"] * 2
    assert len(compiles) == 1 and compiles[0][5] == (*config[:4], 2 if legacy else 3)


@pytest.mark.parametrize("override", [False, True])
def test_dense_table_config_precedence(dense_host, monkeypatch, override):
    config, forced = (16, 1, 16, 64, 2), (32, 2, 32, 128, 3)
    if override:
        dense_host.DENSE_OVERRIDE["config"] = forced
    expected = forced if override else config
    assert dense_host.dense_config(512, 512, 2048, 8, 1, 170, config=config) == expected
    monkeypatch.setattr(dense_host._lora_common, "device_info", lambda device: ((12, 0), 170))
    monkeypatch.setattr(dense_host._lora_common, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiled = []

    def compile_kernel(*args):
        compiled.append(args[5])
        return lambda *args: None, lambda *args: None

    monkeypatch.setattr(dense_host, "compile_dense", compile_kernel)
    x, a, b, out = [torch.empty(shape) for shape in ((17, 512), (8, 512), (1, 64, 8), (17, 64))]
    dense_host.dense_apply_(out, x, a, b, torch.ones(1), rank=8, slices=1, config=config)
    assert compiled == [expected]


def test_shared_table_configs_support_smaller_ranks(dense_host):
    from ctikernels.moe.cute._lora_shared_table import SHARED_TABLE

    # Check future measured tables plus representative entries while it starts empty.
    tables = [
        SHARED_TABLE,
        {
            "gb10": {
                (2, 2048, 512, 128): ((None, "dense", (32, 8, 64, 128, 2)),),
                (1, 512, 2048, 128): ((None, "dense", (64, 4, 32, 64, 3)),),
                (1, 512, 2048, 32): ((None, "dense", (64, 1, 16, 128, 4)),),
            }
        },
    ]
    for table in tables:
        for shapes in table.values():
            for (slices, k, n, rank), entries in shapes.items():
                for _, mode, config in entries:
                    if mode != "dense" or config is None:
                        continue
                    bm, sk, ebm, bn, stages = config
                    for smaller in range(8, rank + 1, 8):
                        dense_host.DenseShrink(k, slices * smaller, bm, sk, stages=stages)
                        for save_z in (False, True):
                            dense_host.DenseExpand(n, smaller, slices, ebm, bn, save_z)


def test_shared_table_generator(tmp_path, capsys):
    import ast
    import json

    from benchmark.kernel_tuning.moe_lora_shared_table import build_table, main

    def record(rows, times, capability=(12, 0), sms=179, proj="w13", rank=64):
        return json.dumps(
            {
                "device": "test",
                "capability": capability,
                "sms": sms,
                "T": rows,
                "r": rank,
                "proj": proj,
                "K": 2048,
                "N": 512,
                "times": times,
                "best": "ignored",
            }
        )

    dense = "d32.8.16.64.2"
    lines = [
        "compiler diagnostic",
        "{}",
        "[]",
        record(16, {"cute": 2, dense: 3}),
        record(32, {"cute": 2, "split": "ValueError"}),
        record(64, {dense: 1, "cute": 2, "library": "bad 0.2"}),
        record(128, {dense: 1}),
        record(256, {"library": 1}),
        record(512, {"library": 1}),
        record(64, {"cute": 3}),
        record(1, {"split": 1}, capability=(12, 1), sms=48, proj="w2"),
        record(1, {"library": 1}, sms=180),
        record(1, {"cute": 1}, capability=(9, 0)),
        record(
            1,
            {
                "cute": float("nan"),
                dense: float("inf"),
                "split": -1,
                "library": True,
                "unknown": 0.1,
            },
            rank=8,
        ),
    ]
    table = build_table(lines)
    assert table == {
        "rtx_5090": {
            (2, 2048, 512, 64): (
                (32, "cute", None),
                (128, "dense", (32, 8, 16, 64, 2)),
                (None, "library", None),
            )
        },
        "rtx_pro_6000": {(2, 2048, 512, 64): ((None, "library", None),)},
        "gb10": {(1, 2048, 512, 64): ((None, "split", None),)},
    }
    assert build_table(reversed(lines)) == table
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text("\n".join(lines[:6]))
    second.write_text("\n".join(lines[6:]))
    policy = tmp_path / "policy.py"
    before, after = "# keep before\n", "# keep after\n"
    policy.write_text(
        before + "# BEGIN GENERATED SHARED TABLE\nSHARED_TABLE = {}\n"
        "# END GENERATED SHARED TABLE\n" + after
    )
    main([str(first), str(second)])
    literal = capsys.readouterr().out
    assert ast.literal_eval(literal.split(" = ", 1)[1]) == table
    assert "SHARED_TABLE = {}" in policy.read_text()
    args = [str(first), str(second), "--write", "--policy", str(policy)]
    main(args)
    written = policy.read_text()
    assert written.startswith(before) and written.endswith(after)
    assert literal in written
    main(args)
    assert policy.read_text() == written


def test_shared_policy_pro_falls_back_to_5090_table(monkeypatch):
    from ctikernels.moe.cute import _lora_shared_table as policy

    monkeypatch.setattr(policy, "SHARED_MODE_OVERRIDE", None)
    entries = ((None, "dense", (16, 4, 16, 64, 2)),)
    monkeypatch.setattr(policy, "SHARED_TABLE", {"rtx_5090": {(1, 512, 2048, 64): entries}})
    assert policy.shared_choice(512, 64, 512, 2048, 1, (12, 0), 188) == entries[0][1:]
    monkeypatch.setattr(
        policy,
        "SHARED_TABLE",
        {
            "rtx_5090": {(1, 512, 2048, 64): entries},
            "rtx_pro_6000": {(1, 512, 2048, 64): ((None, "library", None),)},
        },
    )
    assert policy.shared_choice(512, 64, 512, 2048, 1, (12, 0), 188) == ("library", None)
    assert policy.shared_choice(512, 64, 512, 2048, 1, (12, 1), 48) == ("dense", None)


def test_base_config_shared_lora_tiles():
    from ctikernels.moe.lora import base_config_for

    generic = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 3,
    }
    for tokens in (257, 382, 447, 512):
        for shared in (False, True):
            for gb10 in (False, True):
                assert base_config_for(tokens, shared_lora_tiles=shared, gb10=gb10) == generic
    for tokens in (33, 64):
        config = base_config_for(tokens, shared_lora_tiles=True)
        assert (config["BLOCK_SIZE_M"], config["num_warps"], config["num_stages"]) == (16, 4, 3)
        assert base_config_for(tokens)["BLOCK_SIZE_M"] == 32
        config = base_config_for(tokens, gb10=True)
        assert (config["BLOCK_SIZE_M"], config["num_warps"], config["num_stages"]) == (16, 8, 4)
        assert base_config_for(tokens, shared_lora_tiles=True, gb10=True) == config
    for tokens in (1, 32, 65, 256, 513, 8192):
        assert base_config_for(tokens, shared_lora_tiles=True) == base_config_for(tokens)
        assert base_config_for(tokens, gb10=True) == base_config_for(tokens)


@pytest.mark.parametrize(
    "num_global_experts,local_only,expected",
    [(None, False, 6), (8, False, 3), (4, False, 6), (8, True, 6)],
)
def test_lora_block_rows_count_expected_local_routes(
    monkeypatch, num_global_experts, local_only, expected
):

    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    monkeypatch.setattr(_cute_provider, "device_info", lambda device: ((12, 0), 132))
    monkeypatch.setattr(_cute_provider, "available", lambda: True)
    seen, split_seen = [], []
    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", None)
    monkeypatch.setattr(_cute_provider, "split_chunks", lambda *args: split_seen.append(args) or 4)

    def block_size(pairs, experts, rank=16):
        seen.append((pairs, experts, rank))
        return 16

    def align(ids, bm, experts):
        return (
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )

    # CPU storage with stubbed kernels, as in the alignment composition test.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_cute_provider, "block_size", block_size)
    monkeypatch.setattr(_ops, "moe_lora_w13_", lambda out, *args, **kwargs: out)
    monkeypatch.setattr(_ops, "moe_lora_w2_", lambda out, *args, **kwargs: out)
    monkeypatch.setattr(_ops, "_silu_and_mul", lambda c1: c1[..., : c1.shape[-1] // 2])
    patch_base(
        monkeypatch,
        (_ops, _cute_provider),
        moe_align_block_size=align,
        fused_moe_kernel_triton=lambda x, w, out, *args: out.zero_(),
        moe_sum_reduce_triton=_fake_reduce,
    )
    _ops.fused_experts_lora(
        torch.zeros(3, 24, dtype=torch.bfloat16),
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int32),
        factors(),
        scaling=1.0,
        provider="cute",
        base_config={"BLOCK_SIZE_M": 32},
        local_only=local_only,
        num_global_experts=num_global_experts,
    )
    assert seen == [(expected, 4, 8)]
    assert split_seen == [(expected, 8, 132)]


@pytest.mark.parametrize("bad", [3, 8.0, True])
def test_num_global_experts_validation(bad):
    w = factors()
    with pytest.raises(ValueError, match="num_global_experts"):
        fused_experts_lora(
            torch.zeros(3, 24, dtype=torch.bfloat16),
            torch.zeros(4, 32, 24, dtype=torch.bfloat16),
            torch.zeros(4, 24, 16, dtype=torch.bfloat16),
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32),
            w,
            scaling=1.0,
            num_global_experts=bad,
        )


def test_grid_capacity_bounds_padded_rows():
    from ctikernels.moe._lora_common import grid_capacity

    torch.manual_seed(3)
    for experts, pairs, block_m in ((256, 128, 16), (256, 256, 32), (8, 64, 16), (128, 7, 64)):
        ids = torch.randint(0, experts, (pairs,))
        counts = torch.bincount(ids, minlength=experts)
        padded = int(((counts + block_m - 1) // block_m * block_m).sum())
        buffer = torch.arange(pairs + (experts + 1) * (block_m - 1))
        trimmed = grid_capacity(buffer, pairs, block_m)
        assert padded <= trimmed.numel() <= buffer.numel()
        assert torch.equal(trimmed, buffer[: trimmed.numel()])


def _trainable_reference(x, a, b, ids, rw, top_k, scale, *, down):
    """FP32 autograd delta [pairs, slices*N] from canonical [E,S,r,K]/[E,S,N,r] factors."""
    e, slices = a.shape[:2]
    flat = ids.reshape(-1).long()
    rows = torch.arange(flat.numel()) if down else torch.arange(flat.numel()) // top_k
    safe = flat.clamp(0, e - 1)
    z = torch.einsum("pk,psrk->psr", x[rows].float(), a.float()[safe])
    delta = torch.einsum("psr,psnr->psn", z, b.float()[safe]).reshape(flat.numel(), -1)
    weight = ((flat >= 0) & (flat < e)).float()[:, None]
    if rw is not None:
        weight = weight * rw.reshape(-1, 1).float()
    return scale * weight * delta


@pytest.mark.parametrize("zero", [None, "a", "b"])
@pytest.mark.parametrize("kind", ["routed13", "routed2", "shared13", "shared2", "act"])
def test_low_level_trainable_factors_take_autograd_path(kind, zero):
    torch.manual_seed(3)
    t, top_k, e, r, h, i = 5, 2, 4, 8, 24, 16
    routed, down = kind in ("routed13", "routed2", "act"), kind.endswith("2")
    w = factors(e, r, h, i, shared=True)
    ids = torch.randint(0, e, (t, top_k), dtype=torch.int32) if routed else None
    if routed:
        ids[0, 0] = -1
        ids[1, 1] = e
    rw = torch.rand(t, top_k) if routed and down else None
    p = t * top_k if routed else t
    k, n, slices = (i if routed else 8, h, 1) if down else (h, i if routed else 8, 2)
    x = (torch.randn(p if down else t, k) * 0.3).bfloat16()
    if routed:
        a, b = (w.a2, w.b2) if down else (w.a13, w.b13)
    else:
        a, b = (w.as2, w.bs2) if down else (w.as13, w.bs13)
    a, b = a.clone(), b.clone()
    if zero is not None:
        (a if zero == "a" else b).zero_()
    a, b = a.requires_grad_(True), b.requires_grad_(True)
    a4 = a.unsqueeze(-3) if down else a
    b4 = b.unsqueeze(-3) if down else b
    if not routed:
        a4, b4 = a4[None], b4[None]
    before = (torch.randn(p, slices * n) * 0.01).bfloat16()
    cache = before.clone()
    out = None
    if kind == "act":
        out = torch.zeros(p, n, dtype=torch.bfloat16)
        moe_lora_w13_act_(
            out,
            cache,
            x,
            leaves13(a, b),
            ids,
            scaling=0.5,
            top_k=top_k,
            **act_alignment(ids, e),
        )
    elif kind == "routed13":
        moe_lora_w13_(cache, x, leaves13(a, b), ids, scaling=0.5, top_k=top_k)
    elif kind == "routed2":
        moe_lora_w2_(cache, x, (a, b), ids, rw, scaling=0.5, top_k=top_k)
    elif kind == "shared13":
        shared_lora_w13_(cache, x, (a, b), scaling=0.5)
    else:
        shared_lora_w2_(cache, x, (a, b), scaling=0.5)
    ref_ids = ids if routed else torch.zeros(t, 1, dtype=torch.int32)
    delta = _trainable_reference(x, a4, b4, ref_ids, rw, top_k if routed else 1, 0.5, down=down)
    expected = before.float() + delta
    if kind == "act":
        gate, up = expected.chunk(2, -1)
        expected = F.silu(gate) * up
        valid = ((ids >= 0) & (ids < e)).reshape(-1)
        expected = torch.where(valid[:, None], expected, 0)
        result = out
    else:
        result = cache
    assert result.requires_grad
    grad = torch.randn(result.shape)
    (result.float() * grad).sum().backward()
    ours = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None
    (expected * grad).sum().backward()
    torch.testing.assert_close(result.float(), expected.detach(), atol=2e-3, rtol=1e-2)
    for mine, ref in zip(ours, (a.grad, b.grad), strict=True):
        assert (mine - ref).norm() <= 2e-2 * ref.norm() + 1e-6
    # PEFT init (B = 0, A random): the update is zero, yet B must still learn.
    if zero == "b":
        assert ours[1].abs().sum() > 0


def _fp32_mlp(x, w13, w2, a13, b13, a2, b2, scale):
    """One expert's FP32 MLP with LoRA on gate/up/down (x [M,H])."""
    c1 = x @ w13.T + scale * torch.cat([(x @ a13[s].T) @ b13[s].T for s in range(2)], -1)
    gate, up = c1.chunk(2, -1)
    c2 = F.silu(gate) * up
    return c2 @ w2.T + scale * (c2 @ a2.T) @ b2.T


@pytest.mark.parametrize("zero", [None, "a", "b"])
@pytest.mark.parametrize("shared", [False, True])
def test_composition_gradients_match_fp32_autograd(shared, zero):
    torch.manual_seed(7)
    e, h, i, top_k, scale = 4, 24, 16, 2, 0.5
    w = factors(e, 8, h, i, shared=True)
    names = ("as13", "bs13", "as2", "bs2") if shared else ("a13", "b13", "a2", "b2")
    params = [getattr(w, name).clone() for name in names]
    if not shared:
        params[0] = params[0][:, 0].contiguous()  # training is shared-A: one A [E, r, H]
    if zero is not None:
        for idx in (0, 2) if zero == "a" else (1, 3):
            params[idx].zero_()
    params = [t.requires_grad_(True) for t in params]
    x = (torch.randn(6, h) * 0.3).bfloat16().requires_grad_(True)
    if shared:
        w13, w2 = (torch.randn(16, h) * 0.1).bfloat16(), (torch.randn(h, 8) * 0.1).bfloat16()
        out = shared_expert_lora(x, w13, w2, tuple(params), scale)
    else:
        w13 = (torch.randn(e, 2 * i, h) * 0.1).bfloat16()
        w2 = (torch.randn(e, h, i) * 0.1).bfloat16()
        ids = torch.tensor([[0, 1], [2, -1], [3, 3], [1, 4], [0, 2], [3, 1]], dtype=torch.int32)
        rw = torch.rand(6, top_k)
        out = fused_experts_lora(x, w13, w2, rw, ids, shared_a(*params), scaling=scale)
    grad = torch.randn(out.shape)
    (out.float() * grad).sum().backward()
    ours = [t.grad.clone() for t in (x, *params)]
    for t in (x, *params):
        t.grad = None
    xf, pf = x.float(), [t.float() for t in params]
    if shared:
        a13, b13, a2, b2 = pf
        ref = _fp32_mlp(xf, w13.float(), w2.float(), a13, b13, a2, b2, scale)
    else:
        ref = torch.zeros(6, h)
        for row in range(6):
            for j in range(top_k):
                ex = int(ids[row, j])
                if 0 <= ex < e:
                    a, b13, a2, b2 = (t[ex] for t in pf)
                    contrib = _fp32_mlp(
                        xf[row : row + 1],
                        w13[ex].float(),
                        w2[ex].float(),
                        torch.stack((a, a)),
                        b13,
                        a2,
                        b2,
                        scale,
                    )
                    ref = ref.index_add(0, torch.tensor([row]), rw[row, j] * contrib)
    (ref * grad).sum().backward()
    torch.testing.assert_close(out.float(), ref.detach(), atol=3e-3, rtol=2e-2)
    for mine, t in zip(ours, (x, *params), strict=True):
        assert (mine - t.grad).norm() <= 3e-2 * t.grad.norm() + 1e-6
    if zero == "b":
        assert ours[2].abs().sum() > 0 and ours[4].abs().sum() > 0


def test_shared_factors_in_weights_object_train():
    torch.manual_seed(8)
    w = factors(shared=True)
    shared = tuple(t.clone().requires_grad_(True) for t in w.shared)
    lora = packed(w.a13, w.b13, w.a2, w.b2, *shared)
    x = (torch.randn(6, 24) * 0.3).bfloat16()
    w13, w2 = (torch.randn(16, 24) * 0.1).bfloat16(), (torch.randn(24, 8) * 0.1).bfloat16()
    out = shared_expert_lora(x, w13, w2, lora, 0.5)
    ref = shared_expert_lora(x, w13, w2, shared, 0.5)
    assert out.requires_grad
    torch.testing.assert_close(out, ref)
    out.float().square().sum().backward()
    for t in shared:
        assert t.grad is not None and t.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["composition", "routed13", "routed2"])
def test_all_remote_routes_backprop_zero(kind):
    torch.manual_seed(9)
    e, top_k = 4, 2
    w = factors(e, shared=True)
    ids = torch.tensor([[-1, 4], [5, -1], [-1, -1]], dtype=torch.int32)
    rw = torch.rand(3, top_k).requires_grad_(True)
    if kind == "composition":
        leaves = (w.a13[:, 0], w.b13, w.a2, w.b2)
        params = [t.clone().requires_grad_(True) for t in leaves]
        x = (torch.randn(3, 24) * 0.3).bfloat16().requires_grad_(True)
        w13 = (torch.randn(e, 32, 24) * 0.1).bfloat16()
        w2 = (torch.randn(e, 24, 16) * 0.1).bfloat16()
        out = fused_experts_lora(x, w13, w2, rw, ids, shared_a(*params), scaling=0.5)
        assert torch.equal(out, torch.zeros_like(out))
    else:
        down = kind == "routed2"
        params = [
            t.clone().requires_grad_(True) for t in ((w.a2, w.b2) if down else (w.a13, w.b13))
        ]
        x = (torch.randn(6 if down else 3, 16 if down else 24) * 0.3).bfloat16()
        x.requires_grad_(True)
        out = torch.randn(6, 24 if down else 32).bfloat16()
        before = out.clone()
        if down:
            moe_lora_w2_(out, x, tuple(params), ids, rw, scaling=0.5, top_k=top_k)
        else:
            moe_lora_w13_(out, x, leaves13(*params), ids, scaling=0.5, top_k=top_k)
        assert torch.equal(out, before)
    assert out.requires_grad
    out.float().square().sum().backward()
    for t in (x, *params):
        assert t.grad is not None and torch.equal(t.grad, torch.zeros_like(t.grad))
    if kind != "routed13":
        assert rw.grad is not None and torch.equal(rw.grad, torch.zeros_like(rw.grad))


@pytest.mark.parametrize("trainable", ["x", "rw", "both", "factors"])
def test_all_remote_routes_base_only_backprop_zero(trainable):
    torch.manual_seed(10)
    e, top_k = 4, 2
    ids = torch.tensor([[-1, 4], [5, -1], [-1, -1]], dtype=torch.int32)
    x = (torch.randn(3, 24) * 0.3).bfloat16().requires_grad_(trainable in ("x", "both"))
    rw = torch.rand(3, top_k).requires_grad_(trainable in ("rw", "both"))
    w13 = (torch.randn(e, 32, 24) * 0.1).bfloat16()
    w2 = (torch.randn(e, 24, 16) * 0.1).bfloat16()
    lora, params = None, ()
    if trainable == "factors":
        w = factors(e)
        leaves = (w.a13[:, 0], w.b13, w.a2, w.b2)
        params = tuple(t.clone().requires_grad_(True) for t in leaves)
        lora = shared_a(*params)
    out = fused_experts_lora(x, w13, w2, rw, ids, lora, scaling=0.5)
    assert out.requires_grad
    assert torch.equal(out, torch.zeros_like(out))
    out.float().square().sum().backward()
    for t in (x, rw, *params):
        if t.requires_grad:
            assert t.grad is not None and torch.equal(t.grad, torch.zeros_like(t.grad))


@pytest.mark.parametrize("shared", [False, True])
def test_compositions_train_lora_factors(shared):
    torch.manual_seed(4)
    w = factors(shared=True)
    x = (torch.randn(6, 24) * 0.3).bfloat16().requires_grad_(True)
    if shared:
        lora = tuple(t.clone().requires_grad_(True) for t in w.shared)
        w13, w2 = (torch.randn(16, 24) * 0.1).bfloat16(), (torch.randn(24, 8) * 0.1).bfloat16()
        out = shared_expert_lora(x, w13, w2, lora, 0.5)
        params = lora
        with torch.no_grad():
            frozen = shared_expert_lora(x.detach(), w13, w2, tuple(t.detach() for t in lora), 0.5)
    else:
        leaves = (w.a13[:, 0], w.b13, w.a2, w.b2)
        params = tuple(t.clone().requires_grad_(True) for t in leaves)
        lora = shared_a(*params)
        w13, w2 = (
            (torch.randn(4, 32, 24) * 0.1).bfloat16(),
            (torch.randn(4, 24, 16) * 0.1).bfloat16(),
        )
        ids = torch.tensor([[0, 1], [2, -1], [3, 3], [1, 4], [0, 2], [3, 1]], dtype=torch.int32)
        rw = torch.rand(6, 2)
        out = fused_experts_lora(x, w13, w2, rw, ids, lora, scaling=0.5)
        with torch.no_grad():
            frozen = fused_experts_lora(
                x.detach(),
                w13,
                w2,
                rw,
                ids,
                shared_a(*(t.detach() for t in params)),
                scaling=0.5,
            )
    assert out.requires_grad
    torch.testing.assert_close(out.float(), frozen.float(), atol=2e-3, rtol=2e-2)
    out.float().square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_block_size_rows_per_expert():
    from ctikernels.moe.cute._lora_provider import block_size

    for rank in (32, 64, 96, 128):
        assert block_size(64 * 256, 256, rank) == 32
        assert block_size(64 * 256 - 1, 256, rank) == 16
    for rank in (8, 16):
        assert block_size(1 << 20, 256, rank) == 16


@pytest.mark.parametrize("rank", range(8, 129, 8))
@pytest.mark.parametrize("experts", [128, 64])
def test_qwen3_30b_factor_shapes(rank, experts):
    # Metadata-only construction avoids initializing large factors on the CPU.
    empty = lambda *shape: torch.empty(shape, dtype=torch.bfloat16)
    a13, b13 = empty(experts, 2, rank, 2048), empty(experts, 2, 768, rank)
    a2, b2 = empty(experts, rank, 768), empty(experts, 2048, rank)
    w = packed(a13, b13, a2, b2)
    assert w.rank == rank and w.shared is None
    assert w.gate_a.shape == (experts, rank, 2048)
    assert w.up_b.shape == (experts, 768, rank)
    with pytest.raises(ValueError, match="down_a must have shape"):
        packed(a13, b13, empty(experts, rank, 512), b2)
    with pytest.raises(ValueError, match="rank"):
        packed(a13, b13, a2, b2, rank=12)


@pytest.mark.parametrize("down", [False, True])
def test_qwen3_30b_ep2_projection_mask_and_shape_validation(down):
    e, rank, h, i, top_k = 64, 16, 2048, 768, 8
    k, n, slices = (i, h, 1) if down else (h, i, 2)
    a = torch.zeros(e, slices, rank, k, dtype=torch.bfloat16)
    b = torch.zeros(e, slices, n, rank, dtype=torch.bfloat16)
    # Only the last local expert contributes, so the EP boundary is observable.
    a[63, :, :, 0] = 1
    b[63] = 1
    mapping = torch.arange(128, dtype=torch.int32)
    mapping[64:] = -1
    global_ids = torch.tensor([[0, 63, 64, 127, -1, 128, 1, 62]], dtype=torch.int32)
    ids = to_local_topk_ids(global_ids, mapping)
    assert ids.tolist() == [[0, 63, -1, -1, -1, -1, 1, 62]]
    x = torch.ones(top_k if down else 1, k, dtype=torch.bfloat16)
    before = torch.ones(top_k, slices * n, dtype=torch.bfloat16)
    rw = torch.full((1, top_k), 0.25)
    op = moe_lora_w2_ if down else moe_lora_w13_
    factors = (a[:, 0], b[:, 0]) if down else leaves13(a, b)
    extra = (rw,) if down else ()
    actual = before.clone()
    op(actual, x, factors, ids, *extra, scaling=0.5, top_k=top_k, provider="torch")
    expected = before.clone()
    expected[1] += 0.5 * rank * (0.25 if down else 1)
    assert torch.equal(actual, expected)
    # Qwen3.5's old intermediate dimension must not silently pass validation.
    bad_x = torch.empty(x.shape[0], 512 if down else 2040, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="input/cache shapes"):
        op(actual, bad_x, factors, ids, *extra, scaling=0.5, top_k=top_k)
    bad_cache = torch.empty(top_k, 2040 if down else 1024, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="input/cache shapes"):
        op(bad_cache, x, factors, ids, *extra, scaling=0.5, top_k=top_k)


@pytest.mark.parametrize(
    "pairs,rank,sms,expected",
    [
        (0, 16, 48, 4),
        (1024, 16, 48, 4),
        (1025, 16, 132, 0),
        (1025, 32, 99, 0),
        (1025, 31, 100, 0),
        (1025, 32, 100, 4),
        (4096, 32, 100, 4),
        (4096, 128, 188, 4),
        (4097, 128, 188, 0),
        (1024, 128, 48, 4),
        (1025, 128, 48, 0),
    ],
)
def test_split_chunks_policy(pairs, rank, sms, expected):
    from ctikernels.moe.cute._lora_provider import split_chunks

    assert split_chunks(pairs, rank, sms) == expected


@pytest.mark.parametrize("entry", ["ops", "provider"])
@pytest.mark.parametrize(
    "stage,shared,chunks,message",
    [
        ("full", False, 4, "routed expand"),
        ("shrink", False, 4, "routed expand"),
        ("expand", True, 4, "routed expand"),
        ("expand", False, 0, "positive integer"),
        ("expand", False, -1, "positive integer"),
        ("expand", False, 1.5, "positive integer"),
        ("expand", False, True, "positive integer"),
    ],
)
def test_explicit_chunks_validation(entry, stage, shared, chunks, message):
    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    # Validation precedes tensor/device access, so both entry points run on CPU.
    ids = None if shared else torch.zeros(1, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match=message):
        if entry == "ops":
            _ops._apply(
                None,
                None,
                None,
                ids,
                scaling=1.0,
                top_k=2,
                shared=shared,
                stage=stage,
                chunks=chunks,
            )
        else:
            _cute_provider.apply_(None, None, None, None, ids, None, 2, stage=stage, chunks=chunks)


@pytest.mark.parametrize(
    "override,policy,expected",
    [(None, 4, 4), (None, 0, 0), (0, 4, 0), (1, 0, 1), (2, 0, 2), (4, 0, 4), (8, 0, 8)],
)
@pytest.mark.parametrize(
    "block13,block2,overlap",
    [(True, True, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_split_composition_dispatch(
    overlap_cuda, monkeypatch, override, policy, expected, block13, block2, overlap
):
    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    monkeypatch.setattr(_ops, "_SPLIT_ROUTED", override)
    monkeypatch.setattr(_cute_provider, "split_chunks", lambda *args: policy)
    monkeypatch.setattr(
        _ops,
        "_uses_block_projection",
        lambda *args, down=False, **kwargs: block2 if down else block13,
    )
    fused_experts_lora(
        torch.zeros(129, 24, dtype=torch.bfloat16),
        torch.zeros(4, 32, 24, dtype=torch.bfloat16),
        torch.zeros(4, 24, 16, dtype=torch.bfloat16),
        torch.ones(129, 2),
        torch.zeros(129, 2, dtype=torch.int32),
        factors(),
        scaling=0.5,
        provider="cute",
        overlap=overlap,
    )
    actual = [call for call in overlap_cuda.calls if call[1] in ("shrink", "expand")]
    projections = (False, True) if overlap or (expected and block13) else ()
    wanted = []
    for down in projections:
        if block2 if down else block13:
            wanted.extend(
                [("side" if overlap else "main", "shrink", down), ("main", "expand", down)]
            )
    assert actual == wanted
    for stage, _down, chunks in overlap_cuda.chunks:
        assert chunks == (expected if stage == "expand" and not overlap else None)
    if not overlap:
        assert not overlap_cuda.streams and not overlap_cuda.recorded


@pytest.mark.parametrize(
    "capability,rank,dsl,expected",
    [
        ((12, 0), 16, True, "triton"),
        ((12, 1), 128, True, "triton"),
        ((12, 0), None, False, "triton"),
        ((12, 1), 136, False, "triton"),
        ((10, 3), 8, False, "triton"),
        ((9, 0), 16, True, "triton"),
        ((8, 9), 1, True, "triton"),
        ((8, 0), 16, True, ValueError),
        ((7, 5), 16, True, ValueError),
    ],
)
def test_auto_provider_interval(monkeypatch, capability, rank, dsl, expected):
    """auto: Triton on every CUDA device of compute capability >= 8.9, inference and training
    (routes kept in FP32 to one rounding; the training forward is the inference output);
    never torch on CUDA, never decided by an installed library."""
    from ctikernels.moe import lora as _ops
    from ctikernels.moe.cute import _lora_provider as _cute_provider

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(_cute_provider, "device_info", lambda device: (capability, 132))
    monkeypatch.setattr(_cute_provider, "available", lambda: dsl)
    x = torch.zeros(1, 8)
    if isinstance(expected, str):
        assert _ops._resolve("auto", x, rank) == expected
        assert _ops.supported(torch.device("cuda"), torch.bfloat16)
    else:
        with pytest.raises(expected):
            _ops._resolve("auto", x, rank)
    assert _ops._resolve("torch", x, rank) == "torch"
    with pytest.raises(ValueError, match="provider must be"):
        _ops._resolve("bad", x, rank)


def test_caches_created_in_inference_mode_can_be_saved_for_backward(monkeypatch):
    """The cached scaling tensor and naive alignment, first built while serving under
    inference_mode, are later saved for backward by training."""
    from ctikernels.moe import _lora_common
    from ctikernels.moe import lora as lora_module

    monkeypatch.setattr(_lora_common, "_SCALES", {})
    monkeypatch.setattr(lora_module, "_NAIVE", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cpu = torch.device("cpu")
    with torch.inference_mode():
        scale = _lora_common.scale_tensor(0.7, cpu)
        sorted_ids, padded = lora_module._naive_alignment(8, 16, cpu)

    class Save(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, *tensors):
            ctx.save_for_backward(*tensors)
            return x * 2

        @staticmethod
        def backward(ctx, dy):
            return (dy * 2, *(None for _ in ctx.saved_tensors))

    x = torch.ones(3, requires_grad=True)
    Save.apply(x, scale.detach(), sorted_ids, padded).sum().backward()
    (x * scale.detach()).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), 2.7))
