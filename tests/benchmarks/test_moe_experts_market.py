# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contracts for Qwen3 routing, EP ownership, market adapters and accounting."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import moe_block_market as common
from benchmark.kernel_tuning import moe_experts_market as market
from benchmark.kernel_tuning.moe_triton_tune import expert_mapping, weight_layers


@pytest.mark.parametrize("ep", [1, 2, 4])
def test_moe_tune_contiguous_mapping(ep):
    maps = [expert_mapping(128 // ep, 128, rank) for rank in range(ep)]
    assert torch.stack([m >= 0 for m in maps]).sum(0).tolist() == [1] * 128
    for rank, mapping in enumerate(maps):
        assert mapping.dtype == torch.int32 and mapping.is_contiguous()
        assert torch.equal(mapping[mapping >= 0], torch.arange(128 // ep, dtype=torch.int32))
        expected = torch.arange(rank * (128 // ep), (rank + 1) * (128 // ep))
        assert torch.equal(torch.where(mapping >= 0)[0], expected)
    assert torch.equal(expert_mapping(256, 256, 0), torch.arange(256, dtype=torch.int32))


def test_moe_tune_weight_copies_exceed_l2():
    layer = 32 * 3 * 2048 * 768 * 2
    assert weight_layers(96 << 20, layer) == 2
    assert weight_layers(96 << 20, 4 * layer) == 1
    assert weight_layers(128 << 20, 1) == 4 * (128 << 20)


@pytest.mark.parametrize(
    "local,global_,rank", [(0, 128, 0), (64, 127, 0), (129, 128, 0), (32, 128, -1), (32, 128, 4)]
)
def test_moe_tune_invalid_mapping(local, global_, rank):
    with pytest.raises(ValueError):
        expert_mapping(local, global_, rank)


def test_qwen3_row_presets_and_cli():
    assert market.row_tokens("all") == list(market.TOKENS)
    assert market.row_tokens("decode") == [t for t in market.TOKENS if t <= 64]
    assert market.row_tokens("prefill") == [t for t in market.TOKENS if t >= 128]
    assert market.row_tokens("decode", "513,1,513") == [1, 513]
    args = market.parse_args([])
    assert args.tokens == list(market.TOKENS) and args.ep_map == "auto"
    assert args.rank is None and args.norms == [True]
    assert market.parse_args(["--check-only", "--norm-topk-prob", "false"]).norms == [True, False]
    assert market.parse_args(["--norm-topk-prob", "both"]).norms == [True, False]
    assert market.parse_args(["--norm-topk-prob", "false"]).norms == [False]
    for argv in (
        ["--ep", "4", "--rank", "4"],
        ["--t", "0"],
        ["--t", "bad"],
        ["--layers", "0"],
        ["--providers", "bad"],
    ):
        with pytest.raises(SystemExit):
            market.parse_args(argv)


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("rows", [0, 1, 9])
def test_qwen3_routing_cpu(norm, rows):
    gen = torch.Generator().manual_seed(71)
    x = torch.randn((rows, 32), generator=gen).bfloat16()
    router = torch.randn((128, 32), generator=gen).mul_(0.02).bfloat16()
    scales, ids = market.routing(x, router, norm)
    logits = torch.nn.functional.linear(x, router)
    probabilities = torch.softmax(logits.float(), dim=-1)
    selected, expected_ids = torch.topk(probabilities, 8, dim=-1)
    if norm:
        selected /= selected.sum(-1, keepdim=True)
    assert ids.dtype == torch.int32 and scales.dtype == torch.float32
    assert torch.equal(ids, expected_ids.int())
    torch.testing.assert_close(scales, selected, rtol=0, atol=0)
    if rows:
        assert (
            bool((scales.sum(-1) < 1).all())
            if not norm
            else torch.allclose(scales.sum(-1), torch.ones(rows))
        )
    # Router storage may be FP32, but logits must still have BF16 rounding.
    torch.testing.assert_close(market.routing(x, router.float(), norm)[0], scales, rtol=0, atol=0)


def test_qwen3_sol_counts_remote_padding_and_repeated_hits():
    mapping = expert_mapping(32, 128, 1)
    ids = torch.tensor([[32, 33, 32, 0, -1, 127, 63, 64], [-1] * 8], dtype=torch.int32)
    counts = market.work_counts(ids, mapping)
    assert counts == {
        "bytes": 3 * 3 * 2048 * 768 * 2 + 4 * 2 * 2048,
        "flops": 4 * 2 * 3 * 2048 * 768,
        "unique_local_experts_mean": 3,
        "local_routes_mean": 4,
    }
    remote = market.work_counts(torch.full_like(ids, -1), mapping)
    assert remote["flops"] == 0 and remote["bytes"] == 4 * 2 * 2048
    empty = market.work_counts(ids[:0], mapping)
    assert all(v == 0 for v in empty.values())
    roof = {"read_bytes_per_s": 1e12, "bf16_flops_per_s": 1e14}
    floor, notes = market.sol(
        [SimpleNamespace(ids=ids)], [SimpleNamespace(expert_map=mapping)], roof
    )
    assert floor == pytest.approx(max(counts["bytes"] / 1e12, counts["flops"] / 1e14) * 1e6)
    assert notes["local_routes_mean"] == 4


def test_qwen3_reference_ep_partials_cpu():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn((3, 16), generator=gen).bfloat16()
    up = torch.randn((128, 8, 16), generator=gen).bfloat16()
    down = torch.randn((128, 16, 4), generator=gen).bfloat16()
    router = torch.randn((128, 16), generator=gen).bfloat16()
    inputs = market.Inputs(x, *market.routing(x, router, False))
    w = market.Weights(router, up, down, expert_mapping(128, 128, 0))
    full = market.reference(inputs, w)
    for ep in (2, 4):
        partials = [
            market.reference(inputs, market.partition([w], ep, rank)[0]) for rank in range(ep)
        ]
        torch.testing.assert_close(sum(partials), full, rtol=1e-5, atol=1e-5)
    inputs.ids[0] = -1
    assert torch.count_nonzero(market.reference(inputs, w)[0]) == 0


@pytest.mark.parametrize(
    "name,rows,graph",
    [
        ("ours", 4096, True),
        ("sglang", 2048, True),
        ("sglang", 2049, False),
        ("vllm", 512, True),
        ("vllm", 513, False),
        ("vllm:graph", 16384, True),
    ],
)
def test_market_graph_limits(name, rows, graph):
    assert market.Experts(name, market.parse_args([]), 0, rows).graph_mode == graph


def test_sglang_adapter_maps_global_ids_and_filters():
    mapping = torch.full((128,), -1, dtype=torch.int32)
    mapping[96:] = torch.arange(32, dtype=torch.int32)

    def fused(x, up, down, scales, ids, **kwargs):
        assert torch.equal(ids, torch.tensor([[-1, 2]], dtype=torch.int32))
        assert kwargs["filter_expert"] and scales is inputs.scales
        return x

    sgl = SimpleNamespace(fused=SimpleNamespace(fused_experts_impl=fused))
    inputs = market.Inputs(
        torch.zeros(1, 4), torch.ones(1, 2), torch.tensor([[0, 98]], dtype=torch.int32)
    )
    block = market.Experts("sglang", market.parse_args(["--ep", "4"]), 3, 1, sgl=sgl)
    w = SimpleNamespace(gate_up=None, down=None, expert_map=mapping)
    assert block(inputs, w, (None,)) is inputs.x


def test_flashinfer_precomputed_adapter():
    inputs = market.Inputs(
        torch.zeros(1, 4), torch.ones(1, 2), torch.tensor([[0, 98]], dtype=torch.int32)
    )
    w = SimpleNamespace(up_gate=object(), down=object())
    calls = []
    fi = SimpleNamespace(
        fused=lambda **kwargs: calls.append(kwargs), activation=SimpleNamespace(Swiglu=object())
    )
    block = market.Experts("vllm", market.parse_args(["--ep", "4"]), 3, 1, fi=fi)
    slot = block.prepare(inputs)
    assert block(inputs, w, slot) is slot[0]
    call = calls[0]
    assert (
        call["token_selected_experts"] is inputs.ids and call["token_final_scales"] is inputs.scales
    )
    assert call["fc1_expert_weights"] is w.up_gate
    assert (call["ep_size"], call["ep_rank"], call["tp_size"]) == (4, 3, 1)


def test_shared_flashinfer_tuning_buckets(monkeypatch):
    calls = []

    @contextmanager
    def autotune(tune_mode=False, tuning_buckets=None):
        calls.append((tune_mode, tuning_buckets))
        yield

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    backend = object.__new__(common.FlashInferBackend)
    backend.tuned = False
    backend.buckets = lambda t: [1, 8, t]
    backend.autotune = autotune
    backend.tune_calls(SimpleNamespace(tokens=[1, 64]), [lambda: calls.append("expert")])
    assert calls == [(True, None), (False, [1, 8, 64]), "expert", "sync"]
    assert backend.tuned and backend.tuning_buckets == [1, 8, 64]
    backend.tune_calls(SimpleNamespace(tokens=[1, 64]), [lambda: pytest.fail("retuned")])


@pytest.mark.parametrize(
    "ep,mode,mapped",
    [(1, "auto", False), (1, "always", True), (2, "auto", True), (4, "auto", True)],
)
def test_ours_engine_map_policy(ep, mode, mapped):
    args = market.parse_args(["--ep", str(ep), "--ep-map", mode])
    block = market.Experts("ours", args, 0, 1)
    calls = []
    block.op = lambda *args, **kwargs: calls.append(kwargs)
    inputs = market.Inputs(
        torch.zeros(1, 4), torch.ones(1, 8), torch.zeros(1, 8, dtype=torch.int32)
    )
    w = SimpleNamespace(gate_up=None, down=None, expert_map=expert_mapping(128 // ep, 128, 0))
    block(inputs, w, block.prepare(inputs))
    assert calls[0]["expert_map"] is (w.expert_map if mapped else None)


def test_block_market_defaults_unchanged():
    args = common.parse_args([])
    assert args.tokens == list(range(1, 9)) and args.layers == 3
    assert args.providers == list(common.PROVIDERS)
    assert common.parse_args(["--rows", "prefill"]).layers == 2
    assert common.parse_args(["--ep", "2"]).rank is None
