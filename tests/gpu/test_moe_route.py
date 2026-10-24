# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""MB-R1: routing does not depend on the row count or provider; MB-R2: empty inputs."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from ctikernels.activation.inference import gelu_and_mul, silu_and_mul  # noqa: E402
from ctikernels.moe import Experts, MoEBlock, route  # noqa: E402
from ctikernels.moe._block import admits_shape, hardware  # noqa: E402
from ctikernels.moe._routing_check import classify_topk  # noqa: E402

CHUNKS = (1, 24, 64, 65, 256)
ROWS = 256


@pytest.fixture(scope="module")
def qwen35():
    """Qwen3.5-35B-A3B block weights (EP1): router, experts, shared expert and its gate."""
    pytest.importorskip("triton")
    gen = torch.Generator(device="cuda").manual_seed(31)
    opts = {"device": "cuda", "dtype": torch.bfloat16, "generator": gen}
    shapes = ((256, 2048), (256, 1024, 2048), (256, 2048, 512), (1024, 2048), (2048, 512))
    weights = [torch.randn(shape, **opts).mul_(0.02) for shape in shapes]
    weights.append(torch.randn((1, 2048), **opts).mul_(0.02))
    return tuple(weights)


def _bits(weights):
    return weights.contiguous().view(torch.int32)


def _chunked_block(op, x, weights, size):
    ids, wts, providers = [], [], set()
    for start in range(0, x.shape[0], size):
        op(x[start : start + size].contiguous(), *weights)
        providers.add(op.provider)
        assert op.routing_weights is not None
        ids.append(op.routing_ids.to(torch.int32).clone())
        wts.append(op.routing_weights.clone())
    return torch.cat(ids), torch.cat(wts), providers


def test_block_routing_is_bitwise_invariant_to_rows_and_provider(qwen35):
    """Same tokens chunked at M=1/24/64/65/256: identical expert ids and weights, whether the
    whole-block tree or the composed path routed them (the engine's prefix-hit case)."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn((ROWS, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    reference_w, reference_ids = route(x, qwen35[0], 8, renormalize=True)
    tie, mismatch = classify_topk(x, qwen35[0], reference_ids)
    assert mismatch == 0, (tie, mismatch)
    kind = hardware(x.device)
    seen = set()
    for size in CHUNKS:
        ids, wts, providers = _chunked_block(MoEBlock(), x, qwen35, size)
        seen |= providers
        assert torch.equal(ids, reference_ids), size
        assert torch.equal(_bits(wts), _bits(reference_w)), size
    for provider in ("triton_block", "composed"):
        if provider == "triton_block" and not admits_shape(
            kind,
            rows=64,
            hidden=2048,
            experts=256,
            inter=512,
            local_experts=256,
            shared_inter=512,
            dtype=torch.bfloat16,
        ):
            continue
        ids, wts, _ = _chunked_block(MoEBlock(provider), x, qwen35, 64)
        seen.add(provider)
        assert torch.equal(ids, reference_ids), provider
        assert torch.equal(_bits(wts), _bits(reference_w)), provider
    assert "composed" in seen


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("experts", [128, 256])
def test_route_is_bitwise_invariant_and_correct(experts, renormalize):
    """Qwen3-30B-A3B (E=128, norm_topk_prob both) and Qwen3.5 (E=256) routing at any M."""
    gen = torch.Generator(device="cuda").manual_seed(experts + renormalize)
    router = torch.randn((experts, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    router.mul_(0.02)
    x = torch.randn((ROWS, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    whole_w, whole_ids = route(x, router, 8, renormalize=renormalize)
    for size in CHUNKS:
        parts = [
            route(x[s : s + size], router, 8, renormalize=renormalize) for s in range(0, ROWS, size)
        ]
        assert torch.equal(torch.cat([p[1] for p in parts]), whole_ids), size
        assert torch.equal(_bits(torch.cat([p[0] for p in parts])), _bits(whole_w)), size
    tie, mismatch = classify_topk(x, router, whole_ids)
    assert mismatch == 0, (tie, mismatch)
    # Weights: the model's softmax (over all experts, renormalized over the selection when
    # asked) of BF16 logits. The kernel rounds its own FP32 sums, so a logit may sit one BF16
    # ulp from this FP64 reference (2**-8 relative): the tolerance covers that.
    logits = (x.double() @ router.double().T).bfloat16().double()
    probs = logits.softmax(-1).gather(1, whole_ids.long())
    if renormalize:
        probs /= probs.sum(-1, keepdim=True)
    torch.testing.assert_close(whole_w.double(), probs, rtol=2e-2, atol=1e-4)
    if renormalize:
        torch.testing.assert_close(whole_w.sum(-1), torch.ones(ROWS, device="cuda"))


@pytest.mark.parametrize(
    "experts,top_k,hidden,dtype",
    [
        (60, 4, 2048, torch.bfloat16),
        (64, 6, 1536, torch.bfloat16),
        (384, 8, 7168, torch.bfloat16),
        (8, 1, 1000, torch.bfloat16),
        (128, 8, 2048, torch.float16),
        (256, 8, 2048, torch.float32),
        (512, 10, 4096, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
def test_route_general_shapes(experts, top_k, hidden, dtype, renormalize):
    """SGLang's range: any expert count, top-k, hidden size and BF16/FP16/FP32 router."""
    gen = torch.Generator(device="cuda").manual_seed(experts * 7 + top_k)
    router = torch.randn((experts, hidden), device="cuda", dtype=dtype, generator=gen)
    router.mul_(0.02)
    x = torch.randn((300, hidden), device="cuda", dtype=dtype, generator=gen)
    whole_w, whole_ids = route(x, router, top_k, renormalize=renormalize)
    big = torch.cat([x] * 4)[:1025]  # the fused logit kernel, against split chunks
    big_w, big_ids = route(big, router, top_k, renormalize=renormalize)
    assert torch.equal(big_ids[:300], whole_ids) and torch.equal(_bits(big_w[:300]), _bits(whole_w))
    for size in (1, 64, 65):
        parts = [
            route(x[s : s + size], router, top_k, renormalize=renormalize)
            for s in range(0, 300, size)
        ]
        assert torch.equal(torch.cat([p[1] for p in parts]), whole_ids), size
        assert torch.equal(_bits(torch.cat([p[0] for p in parts])), _bits(whole_w)), size
    logits = x.double() @ router.double().T
    eps = {torch.bfloat16: 2.0**-7, torch.float16: 2.0**-10, torch.float32: 2.0**-20}[dtype]
    tol = eps * (logits.abs().amax(-1, keepdim=True) + 1)
    picked = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, whole_ids.long(), True)
    assert (picked.sum(-1) == top_k).all()
    # every selected expert is at least the best unselected one, up to logit rounding
    low = torch.where(picked, logits, float("inf")).amin(-1, keepdim=True)
    high = torch.where(picked, float("-inf"), logits).amax(-1, keepdim=True)
    assert (low >= high - 2 * tol).all()
    probs = logits.softmax(-1).gather(1, whole_ids.long())
    if renormalize:
        probs /= probs.sum(-1, keepdim=True)
    torch.testing.assert_close(whole_w.double(), probs, rtol=4 * eps + 1e-5, atol=1e-6)


def test_route_prefill_tiles_match_decode_tiles():
    """Prefill tiles (T > 64) and decode tiles (T <= 64) give the same bits for each token."""
    gen = torch.Generator(device="cuda").manual_seed(77)
    router = torch.randn((256, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    router.mul_(0.02)
    x = torch.randn((4096, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    whole_w, whole_ids = route(x, router, 8)
    parts = [route(x[s : s + 64], router, 8) for s in range(0, 4096, 64)]
    assert torch.equal(torch.cat([p[1] for p in parts]), whole_ids)
    assert torch.equal(_bits(torch.cat([p[0] for p in parts])), _bits(whole_w))


def test_route_padding_and_capture():
    router = torch.randn((128, 2048), device="cuda", dtype=torch.bfloat16).mul_(0.02)
    x = torch.randn((32, 2048), device="cuda", dtype=torch.bfloat16)
    valid = torch.tensor([20], device="cuda", dtype=torch.int32)
    full_w, full_ids = route(x, router, 8)
    w, ids = route(x, router, 8, valid_tokens=valid)
    assert torch.equal(ids[:20], full_ids[:20]) and (ids[20:] == -1).all()
    assert torch.equal(w[:20], full_w[:20]) and (w[20:] == 0).all()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route(x, router, 8, valid_tokens=valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cw, cids = route(x, router, 8, valid_tokens=valid)
    x.copy_(torch.randn_like(x))
    valid.fill_(32)
    graph.replay()
    w, ids = route(x, router, 8)
    assert torch.equal(cids, ids) and torch.equal(cw, w)


def test_zero_rows_everywhere(qwen35):
    """MB-R2: empty inputs return correctly shaped empty outputs and launch nothing."""
    for op in (silu_and_mul, gelu_and_mul):
        for shape in ((0, 1536), (0, 8, 1536)):
            x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            assert op(x).shape == (*shape[:-1], 768)
            out = torch.empty((*shape[:-1], 768), device="cuda", dtype=torch.bfloat16)
            assert op(x, out=out) is out
    router, up, down = qwen35[:3]
    x = torch.empty((0, 2048), device="cuda", dtype=torch.bfloat16)
    w, ids = route(x, router, 8)
    assert w.shape == (0, 8) and ids.shape == (0, 8)
    mapping = torch.full((256,), -1, device="cuda", dtype=torch.int32)
    mapping[:64] = torch.arange(64, device="cuda", dtype=torch.int32)
    valid = torch.zeros(1, device="cuda", dtype=torch.int32)
    for kwargs, local in (({}, 256), ({"expert_map": mapping}, 64)):
        out = Experts()(x, up[:local], down[:local], w, ids, **kwargs)
        assert out.shape == (0, 2048)
        out = Experts()(x, up[:local], down[:local], w, ids, valid_tokens=valid, **kwargs)
        assert out.shape == (0, 2048)
    for provider in ("auto", "composed", "triton_block"):
        op = MoEBlock(provider)
        try:
            out = op(x, *qwen35)
        except ValueError:
            assert provider == "triton_block"  # not admitted on this device
            continue
        assert out.shape == (0, 2048)
        assert op.routing_ids.shape == (0, 8) and op.routing_weights.shape == (0, 8)
    # A captured empty call records no kernel and replays.
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = MoEBlock("composed")(x, *qwen35)
        silu_and_mul(torch.empty((0, 1536), device="cuda", dtype=torch.bfloat16))
    graph.replay()
    assert out.shape == (0, 2048)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("row0", [(2.0**24, 0.0, 1.0, -(2.0**24)), (1.0, -(2.0**24), 2.0**24, 0.0)])
def test_route_keeps_partial_boundaries_at_every_t(dtype, row0):
    """Cancellation across slices: separate partials give expert 0 the logit 1, a running
    accumulator (either fold order) gives 0 < 0.5 of expert 1; every T must pick expert 0,
    in the split (T <= 1024) and fused (T > 1024) logit kernels and in the streamed top-k."""
    router = torch.zeros((4, 8), device="cuda", dtype=dtype)
    router[0, :4] = torch.tensor(row0)
    router[1, 0] = 0.5
    for t in (1, 64, 65, 1024, 1025, 4096):
        x = torch.ones((t, 8), device="cuda", dtype=dtype)
        for k in (1, 4):
            w, ids = route(x, router, k)
            assert (ids[:, 0] == 0).all(), (t, k)
        big = torch.zeros((5000, 8), device="cuda", dtype=dtype)
        big[:4] = router
        w, ids = route(x, big, 1)  # streamed selection
        assert (ids == 0).all() and (w == 1).all(), t


def _logit_families():
    """Every (dtype, hidden, tiles) family route() dispatches, with a row count selecting it."""
    from ctikernels.moe._route import ROUTER_SPLIT, _logit_tiles

    families = {}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for hidden in (8, 2048, 7168):
            bk = -(-hidden // ROUTER_SPLIT)
            for kind in (None, "gb10"):
                for t in (1, 64, 65, 256, 257, 512, 513, 2048, 2049):
                    for experts in (8, 128, 256):
                        tiles = _logit_tiles(t, experts, bk, dtype.itemsize, kind)
                        families.setdefault((dtype, hidden, tiles), (t, experts))
    return families


_FAMILIES = _logit_families()


def _zero_or_own_chain(ir):
    """Map each tt.dot to whether its accumulator is a zero constant, or a loop-carried value
    whose initial value is zero and whose yielded value is that dot's result (through
    branches that otherwise yield zero): no dot accumulates onto another chain's value."""
    import re

    values = {}  # SSA name -> defining text
    zeros = set()
    dots = []  # (result, accumulator, innermost loop frame)
    stack = []  # frames: dict(kind, args, inits, yields, results)
    for line in ir.splitlines():
        text = line.strip()
        define = re.match(r"(%[\w.$-]+)(?::(\d+))? = (.*)", text)
        if text.startswith("}"):
            frame = stack.pop() if stack else None
            if text.endswith("{") and frame is not None:  # } else {
                stack.append(frame)
            continue
        if define:
            name, count, body = define.groups()
            values[name] = body
            if re.match(r"arith\.constant dense<(0\.0+e\+00|-?0\.0+)>", body):
                zeros.add(name)
            convert = re.match(r"ttg\.convert_layout (%[\w.$#-]+)", body)
            if convert and convert.group(1) in zeros:
                zeros.add(name)
            dot = re.match(r"tt\.dot (%[\w.$#-]+), (%[\w.$#-]+), (%[\w.$#-]+)", body)
            if dot:
                loop = next((f for f in reversed(stack) if f["kind"] == "for"), None)
                dots.append((name, dot.group(3), loop))
        if text.endswith("{"):
            frame = {"kind": "other", "yields": []}
            if define and "scf.for" in define.group(3):
                pairs = re.findall(
                    r"(%[\w.$-]+) = (%[\w.$#-]+)", define.group(3).split("iter_args(")[1]
                )
                frame.update(kind="for", args=[a for a, _ in pairs], inits=[i for _, i in pairs])
            elif define and "scf.if" in define.group(3):
                frame.update(kind="if", result=define.group(1))
            stack.append(frame)
            if frame["kind"] == "if":
                values[define.group(1)] = frame
        elif text.startswith("scf.yield") and stack:
            stack[-1]["yields"].append(re.findall(r"%[\w.$#-]+", text.split(":")[0]))

    def origin(value, target, depth=0):
        """ "dot" (the target's chain), "zero" (a reset), or None. A select or branch must
        join the chain with a zero reset: two copies of the chain are not a reset."""
        if value == target:
            return "dot"
        if value in zeros:
            return "zero"
        name, _, index = value.partition("#")
        frame = values.get(name)
        select = re.match(r"arith\.select %\S+, (%[\w.$#-]+), (%[\w.$#-]+)", str(frame))
        if depth < 8 and select:
            parts = [origin(v, target, depth + 1) for v in select.groups()]
        elif depth < 8 and isinstance(frame, dict) and frame["yields"]:
            parts = [origin(y[int(index or 0)], target, depth + 1) for y in frame["yields"]]
        else:
            return None
        if None in parts or (len(set(parts)) == 1 and parts[0] == "dot"):
            return None
        return "dot" if "dot" in parts else "zero"

    verdicts = []
    for result, acc, loop in dots:
        if acc in zeros:
            verdicts.append((result, acc, True))
            continue
        ok = loop is not None and acc in loop["args"] and len(loop["yields"]) == 1
        if ok:
            i = loop["args"].index(acc)
            ok = loop["inits"][i] in zeros and origin(loop["yields"][0][i], result) == "dot"
        verdicts.append((result, acc, ok))
    return verdicts


@pytest.mark.parametrize("narrow", [False, True], ids=["fp32-out", "input-dtype-out"])
@pytest.mark.parametrize(
    "family", list(_FAMILIES), ids=lambda f: f"{str(f[0])[6:]}-h{f[1]}-{'_'.join(map(str, f[2]))}"
)
def test_route_logit_partials_never_fold_into_a_dot(family, narrow):
    """IR check of every dispatched logit family (tiles, split, flat, warps, stages, every
    dtype, single- and multi-chunk slices): each dot's accumulator is zero or its own chain."""
    import re

    import triton

    from ctikernels.moe._route import _ROUND, ROUTER_SPLIT
    from ctikernels.moe._route_kernels import _route_logits_kernel

    dtype, hidden, (bt, be, bkc, split, warps, stages, flat) = family
    t, experts = _FAMILIES[family]
    if narrow and (split or dtype == torch.float32):
        pytest.skip("split partials and FP32 inputs always store FP32 logits")
    x = torch.ones((t, hidden), device="cuda", dtype=dtype)
    router = torch.ones((experts, hidden), device="cuda", dtype=dtype)
    # Fused logits are stored in the input dtype with renormalize (route's specialisation).
    z = torch.empty(
        (ROUTER_SPLIT if split else 1, t, experts),
        device="cuda",
        dtype=dtype if narrow else torch.float32,
    )
    neb = triton.cdiv(experts, be)
    compiled = _route_logits_kernel[(neb * triton.cdiv(t, bt), ROUTER_SPLIT if split else 1)](
        x, router, z, t, hidden, experts, ROUTER_SPLIT, bt, be, bkc, split,
        dtype == torch.float32, _ROUND[dtype], flat, neb, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    # The add-into-dot fold is a TTIR combine; TTGIR still holds tt.dot below sm_100 (on
    # sm_100+ the MMA is ttng.tc_gen5_mma with a TMEM accumulator), so check both where present.
    ir = compiled.asm["ttir"]
    for level in [ir] + [compiled.asm["ttgir"]] * ("tt.dot " in compiled.asm["ttgir"]):
        verdicts = _zero_or_own_chain(level)
        assert verdicts and all(ok for _, _, ok in verdicts), verdicts
    if not split:
        assert ir.count("tt.elementwise_inline_asm") >= (1 if flat else ROUTER_SPLIT)
    assert torch.equal(z.sum(0) if split else z[0], torch.full_like(z[0], hidden))
    if narrow:
        element = "bf16" if dtype == torch.bfloat16 else "f16"
        assert re.search(rf"tt\.store[^\n]*!tt\.ptr<{element}>", ir), "no narrow logit store"


@pytest.mark.parametrize("experts,top_k", [(16, 16), (256, 8), (5000, 40)])
def test_route_nonfinite_logits(experts, top_k):
    """FP16 overflow (+-inf), NaN rows and top_k=E: valid, distinct ids in a fixed order."""
    gen = torch.Generator(device="cuda").manual_seed(experts)
    router = torch.randn((experts, 64), device="cuda", dtype=torch.float16, generator=gen)
    router[: experts // 2].mul_(3000.0)  # logits overflow FP16 to +-inf
    x = torch.randn((80, 64), device="cuda", dtype=torch.float16, generator=gen).mul_(30.0)
    x[3] = float("nan")
    x[5, 0] = float("inf")
    w, ids = route(x, router, top_k)
    assert ((ids >= 0) & (ids < experts)).all()
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()  # no expert twice
    for size in (1, 65):
        parts = [route(x[s : s + size], router, top_k) for s in range(0, 80, size)]
        assert torch.equal(torch.cat([p[1] for p in parts]), ids)
        assert torch.equal(_bits(torch.cat([p[0] for p in parts])), _bits(w))
    finite = torch.isfinite((x.float() @ router.float().T)).all(-1)
    assert finite.any() and (~finite).any()


def test_route_streams_very_many_experts():
    """More experts than a Triton tile holds (E > 2**20): argmax with ties to the lower id."""
    e = (1 << 20) + 1
    router = torch.zeros((e, 1), device="cuda", dtype=torch.bfloat16)
    router[e - 1] = 2.0
    router[7] = 2.0
    router[123456] = 1.0
    x = torch.ones((2, 1), device="cuda", dtype=torch.bfloat16)
    w, ids = route(x, router, 2)
    assert ids.tolist() == [[7, e - 1], [7, e - 1]]
    torch.testing.assert_close(w, torch.full((2, 2), 0.5, device="cuda"))


def test_block_padding_weights_follow_valid_prefix(qwen35):
    """Padded rows' weights are zero in every provider while the valid prefix changes, also
    on graph replay; valid rows match route."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn((64, 2048), device="cuda", dtype=torch.bfloat16, generator=gen)
    ref_w, ref_ids = route(x, qwen35[0], 8)
    for provider in ("triton_block", "composed"):
        if provider == "triton_block" and not admits_shape(
            hardware(x.device),
            rows=64,
            hidden=2048,
            experts=256,
            inter=512,
            local_experts=256,
            shared_inter=512,
            dtype=torch.bfloat16,
        ):
            continue
        op = MoEBlock(provider)
        valid = torch.tensor([64], device="cuda", dtype=torch.int32)
        op(x, *qwen35, valid_tokens=valid)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(x, *qwen35, valid_tokens=valid)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(x, *qwen35, valid_tokens=valid)
        for count in (64, 40, 0, 17):
            valid.fill_(count)
            graph.replay()
            torch.cuda.synchronize()
            w = op.routing_weights
            assert (w[count:] == 0).all(), (provider, count)
            assert torch.equal(_bits(w[:count]), _bits(ref_w[:count])), (provider, count)
            ids = op.routing_ids.to(torch.int32)
            assert torch.equal(ids[:count], ref_ids[:count]), (provider, count)


def _special_router(values, hidden=8, dtype=torch.bfloat16):
    # logit e = 1 * values[e] for the one-hot rows below: chosen bit patterns as logits
    router = torch.zeros((values.numel(), hidden), device="cuda", dtype=dtype)
    router[:, 0] = values.to(dtype)
    return router


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_route_special_values_agree_across_kernels(dtype):
    """Signed NaN payloads, +-inf, +-0 and subnormals: the same ids and weight bits from the
    split, fused and streamed paths at T=1/513/2049, and FP32 nonfinite selection."""
    bits = torch.tensor(
        [
            0x7FC00001,
            0xFFC00002,
            0x7F800000,
            0xFF800000,
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0x00800000,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    values = bits.view(torch.float32).cuda()
    values = torch.cat([values, torch.linspace(-2, 2, 52, device="cuda")])
    router = _special_router(values, dtype=dtype)
    for k in (8, 40):  # 40 > 32: streamed selection over the same logits
        results = []
        for t in (1, 513, 2049):  # split; fused on GB10; fused everywhere
            x = torch.zeros((t, 8), device="cuda", dtype=dtype)
            x[:, 0] = 1
            w, ids = route(x, router, k)
            assert ((ids >= 0) & (ids < values.numel())).all()
            assert (ids.sort(-1).values.diff(dim=-1) > 0).all()
            assert (ids == ids[:1]).all() and torch.equal(_bits(w), _bits(w[:1]).expand_as(w))
            results.append((ids[0], _bits(w[0])))
        for ids, w in results[1:]:
            assert torch.equal(ids, results[0][0]) and torch.equal(w, results[0][1]), k


def test_route_special_values_match_block_tree(qwen35):
    """BF16 special logits in the whole-block tree and in route select alike."""
    kind = hardware(torch.device("cuda", torch.cuda.current_device()))
    if not admits_shape(
        kind,
        rows=4,
        hidden=2048,
        experts=256,
        inter=512,
        local_experts=256,
        shared_inter=512,
        dtype=torch.bfloat16,
    ):
        pytest.skip("whole-block provider does not admit this device")
    bits = torch.tensor(
        [0x7FC1, 0xFFC2, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x8001], dtype=torch.int32
    ).to(torch.int16)
    special = bits.view(torch.bfloat16).float().cuda()
    values = torch.cat([special, torch.linspace(-3, 3, 248, device="cuda")])
    router = torch.zeros((256, 2048), device="cuda", dtype=torch.bfloat16)
    router[:, 0] = values.bfloat16()
    x = torch.zeros((4, 2048), device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1
    op = MoEBlock("triton_block")
    op(x, router, *qwen35[1:])
    w, ids = route(x, router, 8)
    assert torch.equal(op.routing_ids.to(torch.int32), ids)
    assert torch.equal(_bits(op.routing_weights), _bits(w))


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("experts,top_k", [(256, 64), (5000, 8)])
def test_route_streamed_paths(experts, top_k, renormalize):
    """k > 32 with E <= 4096 and E > 4096: streamed selection invariant across T=1024/1025,
    correct against FP64, and under a shrinking valid prefix in graph replay."""
    gen = torch.Generator(device="cuda").manual_seed(experts + top_k)
    router = torch.randn((experts, 256), device="cuda", dtype=torch.bfloat16, generator=gen)
    x = torch.randn((1025, 256), device="cuda", dtype=torch.bfloat16, generator=gen)
    w, ids = route(x, router, top_k, renormalize=renormalize)
    w2, ids2 = route(x[:1024], router, top_k, renormalize=renormalize)
    assert torch.equal(ids[:1024], ids2) and torch.equal(_bits(w[:1024]), _bits(w2))
    logits = (x.double() @ router.double().T).bfloat16().double()
    probs = logits.softmax(-1).gather(1, ids.long())
    if renormalize:
        probs /= probs.sum(-1, keepdim=True)
    torch.testing.assert_close(w.double(), probs, rtol=2e-2, atol=1e-5)
    xs = x[:64].contiguous()
    valid = torch.tensor([64], device="cuda", dtype=torch.int32)
    ref_w, ref_ids = route(xs, router, top_k, renormalize=renormalize)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route(xs, router, top_k, renormalize=renormalize, valid_tokens=valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cw, cids = route(xs, router, top_k, renormalize=renormalize, valid_tokens=valid)
    for count in (64, 33, 0, 5):
        valid.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cids[:count], ref_ids[:count]) and (cids[count:] == -1).all()
        assert torch.equal(_bits(cw[:count]), _bits(ref_w[:count])) and (cw[count:] == 0).all()


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_route_dispatch_boundaries(dtype, renormalize):
    """Every dispatched tile family (16x16 / 32x64 / 64x64 split, flat 64x128 fused) gives
    the same bits: rows routed at T=1, in 64-row chunks, and at T=64..2049 boundaries."""
    gen = torch.Generator(device="cuda").manual_seed(2049)
    router = torch.randn((128, 2048), device="cuda", dtype=dtype, generator=gen).mul_(0.02)
    x = torch.randn((2049, 2048), device="cuda", dtype=dtype, generator=gen)
    w, ids = route(x, router, 8, renormalize=renormalize)
    for t in (64, 65, 256, 257, 512, 513, 2048):
        wt, it = route(x[:t], router, 8, renormalize=renormalize)
        assert torch.equal(it, ids[:t]) and torch.equal(_bits(wt), _bits(w[:t])), t
    singles = [route(x[i : i + 1], router, 8, renormalize=renormalize) for i in range(0, 2049, 97)]
    assert torch.equal(torch.cat([p[1] for p in singles]), ids[::97])
    assert torch.equal(_bits(torch.cat([p[0] for p in singles])), _bits(w[::97]))


@pytest.mark.parametrize(
    "experts,top_k,rows",
    ((8, 2, 3), (128, 8, 40), (256, 8, 3000), (512, 10, 5), (8192, 4, 3), (64, 40, 6)),
)
@pytest.mark.parametrize("renormalize", (True, False))
def test_route_replay_and_probabilities(experts, top_k, rows, renormalize):
    """Replaying route's own selection reproduces its weights bitwise (every kernel path);
    the all-expert probabilities hold the unrenormalized weights at the selected experts."""
    gen = torch.Generator(device="cuda").manual_seed(experts + rows)
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16, generator=gen)
    router = torch.randn((experts, 256), device="cuda", dtype=torch.bfloat16, generator=gen)
    valid = torch.tensor([rows - 1], dtype=torch.int32, device="cuda")
    weights, ids = route(x, router, top_k, renormalize=renormalize, valid_tokens=valid)
    replay = ids.clamp_min(0).to(torch.uint8 if experts <= 256 else torch.int64)
    probs = None if renormalize else torch.full((rows, experts), torch.nan, device="cuda")
    again, same = route(
        x,
        router,
        top_k,
        renormalize=renormalize,
        valid_tokens=valid,
        selected=replay,
        probabilities=probs,
    )
    assert torch.equal(same, ids)
    assert torch.equal(_bits(again), _bits(weights))
    if probs is not None:
        assert torch.equal(probs[-1], torch.zeros_like(probs[-1]))
        picked = probs[:-1].gather(1, ids[:-1].long())
        assert torch.equal(_bits(picked), _bits(weights[:-1]))
        torch.testing.assert_close(probs[:-1].sum(1), torch.ones(rows - 1, device="cuda"))
    # A different given selection is weighted by the same logits: against this call's own
    # all-expert probabilities (bitwise without renormalize, renormalized in FP32 with it).
    other = torch.stack(
        [torch.randperm(experts, device="cuda", generator=gen)[:top_k] for _ in range(rows)]
    )
    probs_all = torch.empty((rows, experts), device="cuda")
    w_all, _ = route(x, router, top_k, renormalize=False, selected=other, probabilities=probs_all)
    picked = probs_all.gather(1, other)
    assert torch.equal(_bits(w_all), _bits(picked))
    if renormalize:
        w, _ = route(x, router, top_k, renormalize=True, selected=other)
        torch.testing.assert_close(w, picked / picked.sum(1, keepdim=True), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("experts,top_k,hidden", ((256, 6, 4096), (64, 8, 256), (300, 6, 520)))
@pytest.mark.parametrize("biased", (True, False))
@pytest.mark.parametrize("renormalize", (True, False))
def test_route_dsv4_scoring(experts, top_k, hidden, biased, renormalize):
    """DeepSeek-V4 scoring against the torch reference, invariant to the row count, and
    replayable bitwise."""
    from ..modules._moe_route_reference import bias_margin, dsv4_route

    gen = torch.Generator(device="cuda").manual_seed(experts + hidden)
    rows = 300
    x = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16, generator=gen)
    router = (
        torch.randn((experts, hidden), device="cuda", generator=gen) * hidden**-0.5
    ).bfloat16()
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.1 if biased else None
    valid = torch.tensor([rows - 3], dtype=torch.int32, device="cuda")
    kw = {"renormalize": renormalize, "scoring": "sqrtsoftplus", "bias": bias, "scale": 1.5}
    w, ids = route(x, router, top_k, valid_tokens=valid, **kw)
    want_w, want_ids = dsv4_route(
        x, router, top_k, bias, renormalize=renormalize, scale=1.5, valid=rows - 3
    )
    clear = bias_margin(x, router, top_k, bias) > 1e-4
    clear[rows - 3 :] = True
    assert torch.equal(ids[clear], want_ids[clear])
    keep = clear[:, None].expand_as(w)
    torch.testing.assert_close(w[keep], want_w[keep], rtol=1e-5, atol=1e-6)
    # The gate GEMM runs in cuBLAS above SCORE_GEMM_ROWS (its algorithm may change with the
    # row count): other row counts agree on clear rows and to FP32 rounding.
    for size in (1, 7, 64):
        part_w, part_ids = route(x[:size], router, top_k, **kw)
        keep = clear[:size]
        assert torch.equal(part_ids[keep], ids[:size][keep])
        torch.testing.assert_close(part_w[keep], w[:size][keep], rtol=1e-5, atol=1e-6)
    again, same = route(x, router, top_k, valid_tokens=valid, selected=ids.clamp_min(0), **kw)
    assert torch.equal(same, ids) and torch.equal(_bits(again), _bits(w))


@pytest.mark.parametrize("rows", (1, 5, 300))
def test_hash_route(rows):
    from ctikernels.moe import hash_route

    from ..modules._moe_route_reference import dsv4_hash_route

    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16, generator=gen)
    router = (torch.randn((256, 4096), device="cuda", generator=gen) * 4096**-0.5).bfloat16()
    table = torch.randint(256, (129280, 6), device="cuda", dtype=torch.int32, generator=gen)
    tokens = torch.randint(129280, (rows,), device="cuda", generator=gen)
    valid = torch.tensor([max(rows - 1, 1)], dtype=torch.int32, device="cuda")
    w, ids = hash_route(x, router, tokens, table, scale=1.5, valid_tokens=valid)
    want_w, want_ids = dsv4_hash_route(x, router, tokens, table, scale=1.5, valid=int(valid))
    assert torch.equal(ids, want_ids)
    torch.testing.assert_close(w, want_w, rtol=1e-5, atol=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gw, gids = hash_route(x, router, tokens, table, scale=1.5, valid_tokens=valid)
    graph.replay()
    assert torch.equal(gids, ids) and torch.equal(_bits(gw), _bits(w))


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("shape", ((0, 8), (3, 8), (4097, 6), (5,)))
def test_to_local_topk_ids(dtype, shape):
    """One launch, the torch contract on the CPU, also captured in a graph."""
    from ctikernels.moe import to_local_topk_ids

    gen = torch.Generator(device="cuda").manual_seed(sum(shape))
    ids = torch.randint(-1, 260, shape, device="cuda", generator=gen).to(dtype)
    mapping = torch.randint(-1, 32, (256,), device="cuda", generator=gen, dtype=torch.int32)
    want = to_local_topk_ids(ids.cpu(), mapping.cpu())
    assert torch.equal(to_local_topk_ids(ids, mapping).cpu(), want)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = to_local_topk_ids(ids, mapping)
    graph.replay()
    assert got.dtype == torch.int32 and torch.equal(got.cpu(), want)


@pytest.mark.parametrize("scoring", ("softmax", "sqrtsoftplus"))
def test_route_exact_ties_take_the_lower_id(scoring):
    """Bit-identical selection keys rank the lower expert id first (both scorings)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    router = torch.randn((256, 512), device="cuda", generator=gen).mul_(0.05).bfloat16()
    router[7] = router[3]
    router[200] = router[3]
    x = torch.randn((64, 512), device="cuda", generator=gen).bfloat16()
    kw = {}
    if scoring == "sqrtsoftplus":
        bias = torch.zeros(256, device="cuda")
        bias[[3, 7, 200]] = 10.0  # selected on every row
        kw = {"scoring": scoring, "bias": bias, "scale": 1.5}
    else:
        router[[3, 7, 200]] = (x.float().mean(0) * 50).bfloat16()  # the largest logits
    _, ids = route(x, router, 2, **kw)
    assert torch.equal(ids, torch.tensor([[3, 7]], device="cuda", dtype=torch.int32).expand(64, -1))


def test_route_dsv4_batch_invariant_at_every_row_count():
    """batch_invariant=True: sqrtsoftplus routing is bitwise row-count invariant past the
    cuBLAS threshold too."""
    from ctikernels.moe import _route

    gen = torch.Generator(device="cuda").manual_seed(11)
    rows = _route.SCORE_GEMM_ROWS + 300
    x = torch.randn((rows, 1024), device="cuda", dtype=torch.bfloat16, generator=gen)
    router = (torch.randn((256, 1024), device="cuda", generator=gen) * 1024**-0.5).bfloat16()
    bias = torch.randn(256, device="cuda", generator=gen) * 0.02
    kw = {"scoring": "sqrtsoftplus", "bias": bias, "scale": 1.5, "batch_invariant": True}
    w, ids = route(x, router, 6, **kw)
    for size in (1, 7, 64, _route.SCORE_GEMM_ROWS + 1):
        part_w, part_ids = route(x[:size], router, 6, **kw)
        assert torch.equal(part_ids, ids[:size])
        assert torch.equal(_bits(part_w), _bits(w[:size]))


@pytest.mark.parametrize("mode", ["renormalize", "probabilities", "replay"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("experts", [8, 64, 256, 512])
def test_route_is_vllm_topk_softmax(experts, dtype, mode):
    """Softmax routing over a power-of-two E <= 512 is vLLM 0.31's topk_softmax, bit for bit.
    Integer-valued inputs make every logit exact whatever the dot order, and tie often."""
    from ..modules._moe_reference import vllm_topk_softmax

    gen = torch.Generator(device="cuda").manual_seed(experts)
    ints = lambda *shape: torch.randint(-2, 3, shape, device="cuda", generator=gen)
    x = ints(300, 1024).to(dtype)
    router = (ints(experts, 1024) * 2.0**-4).to(dtype)
    logits = (x.float() @ router.float().T).to(dtype)  # exact sums, rounded once
    top_k = min(8, experts)
    want_w, want_ids, probs = vllm_topk_softmax(logits, top_k, renormalize=mode != "probabilities")
    if mode == "replay":
        selected = want_ids.flip(1).contiguous()
        want_w, _, _ = vllm_topk_softmax(logits, top_k, selected=selected)
        got_w, got_ids = route(x, router, top_k, selected=selected)
        want_ids = selected
    elif mode == "probabilities":
        got_p = torch.empty_like(probs)
        got_w, got_ids = route(x, router, top_k, renormalize=False, probabilities=got_p)
        assert torch.equal(_bits(got_p), _bits(probs))
    else:
        got_w, got_ids = route(x, router, top_k)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(_bits(got_w), _bits(want_w))


@pytest.mark.parametrize("mode", ["renormalize", "probabilities", "replay"])
@pytest.mark.parametrize("experts,top_k", [(60, 8), (384, 8), (384, 40), (1000, 64), (8192, 8)])
def test_route_any_e_is_the_vllm_formula(experts, top_k, mode):
    """Every other E (tiled and streamed kernels) routes by vLLM's formula in this module's
    summation order, bit for bit. Integer-valued inputs give exact logits and many ties."""
    from ..modules._moe_reference import vllm_topk_softmax

    gen = torch.Generator(device="cuda").manual_seed(experts + top_k)
    ints = lambda *shape: torch.randint(-2, 3, shape, device="cuda", generator=gen)
    x = ints(130, 512).bfloat16()
    router = (ints(experts, 512) * 2.0**-4).bfloat16()
    logits = (x.float() @ router.float().T).bfloat16()
    want_w, want_ids, probs = vllm_topk_softmax(logits, top_k, renormalize=mode != "probabilities")
    if mode == "replay":
        selected = want_ids.flip(1).contiguous()
        want_w, _, _ = vllm_topk_softmax(logits, top_k, selected=selected)
        got_w, got_ids = route(x, router, top_k, selected=selected)
        want_ids = selected
    elif mode == "probabilities":
        got_p = torch.empty_like(probs)
        got_w, got_ids = route(x, router, top_k, renormalize=False, probabilities=got_p)
        assert torch.equal(_bits(got_p), _bits(probs))
    else:
        got_w, got_ids = route(x, router, top_k)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(_bits(got_w), _bits(want_w))
