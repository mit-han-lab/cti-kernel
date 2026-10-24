# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2018-2020 Philippe Tillet
# SPDX-FileCopyrightText: Copyright 2020-2022 OpenAI
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2 (`python/minisgl/kernel/
# triton/fused_moe.py`), which builds on SGLang/vLLM's fused MoE and the Triton matmul tutorial;
# see THIRD_PARTY_NOTICES. Block-FP8 operands contributed by Waterpine.

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .rounding import round_to
from .silu import sigmoid, silu


@triton.jit
def shared_join(routed, shared, gate, OUT: tl.constexpr):
    """vLLM's gated shared-expert join (Qwen2-MoE), each step rounded to OUT:
    ``r(r(r(sigmoid(gate)) * shared) + r(routed))``; FP32 operands, an OUT result."""
    scale = round_to(sigmoid(gate), OUT)
    product = round_to(scale * shared, OUT)
    return (product + round_to(routed, OUT)).to(OUT)


@triton.jit
def moe_shared_join_kernel(
    out_ptr, out_stride, shared_ptr, shared_stride, gate_ptr, gate_stride, hidden_dim,
    BLOCK_DIM: tl.constexpr,
):  # fmt: skip
    """In place: one row's routed output ``out`` joins the gated shared expert."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask = cols < hidden_dim
    routed = tl.load(out_ptr + row * out_stride + cols, mask=mask, other=0.0).to(tl.float32)
    shared = tl.load(shared_ptr + row * shared_stride + cols, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + row * gate_stride).to(tl.float32)
    values = shared_join(routed, shared.to(tl.float32), gate, out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * out_stride + cols, values, mask=mask)


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _lora_columns(
    low_ptr,
    b_ptr,
    routes,
    route_mask,
    expert,
    cols,
    rank,
    stride_lm,
    stride_be,
    stride_bn,
    LORA_R: tl.constexpr,
):
    """FP32 [rows, cols] = low[routes, :rank] . B[expert, cols, :rank]^T. FP32 ``low`` enters as
    hi + lo 16-bit halves (two tensor-core dots, ~16 significant bits), B in its own dtype."""
    r = tl.arange(0, LORA_R)
    r_mask = r < rank
    low = tl.load(
        low_ptr + routes[:, None] * stride_lm + r[None, :],
        mask=route_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr + expert.to(tl.int64) * stride_be + cols[None, :].to(tl.int64) * stride_bn
        + r[:, None],
        mask=r_mask[:, None],
        other=0.0,
    )  # fmt: skip
    hi = low.to(b.dtype)
    lo = (low - hi.to(tl.float32)).to(b.dtype)
    return tl.dot(hi, b, out_dtype=tl.float32) + tl.dot(lo, b, out_dtype=tl.float32)


from ..collective.triton.exchange import publish  # noqa: E402


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    topk_ids_ptr,
    expert_map_ptr,
    weights_ptr,
    shared_ptr,
    shared_stride_0,
    gate_ptr,
    gate_stride_0,
    valid_ptr,
    MASKED: tl.constexpr,
    WEIGHTED: tl.constexpr,
    SHARED: tl.constexpr,
    VALID: tl.constexpr,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    GDC_EARLY: tl.constexpr = False,
    xch=None,
    EXCHANGE: tl.constexpr = False,
    XCH_RANK: tl.constexpr = 0,
    XCH_WORLD: tl.constexpr = 1,
):
    """``WEIGHTED``: unweighted rows combine as FlashInfer's finalize (vLLM's default):
    ``acc = fma(w_k, y_k, acc)`` in route order from +0, one rounding; padding routes (id -1, or
    unmapped under ``MASKED``) are skipped. Otherwise rows are summed in FP32.
    ``SHARED`` joins the gated shared expert in vLLM's output-dtype op order (Qwen2-MoE):
    ``out = r(r(r(sigmoid(gate)) * shared) + routed)``, each step rounded to the output dtype.
    ``EXCHANGE`` publishes every rounded output row block (tiles 1 x BLOCK_DIM) to the
    group's exchange (``collective.triton.exchange.publish``)."""
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    if USE_GDC:
        # Programmatic dependent launch: wait for the producers' writes before any load.
        tl.extra.cuda.gdc_wait()
        if GDC_EARLY:
            # Small grids let the next call's prologue launch now; it waits for our writes.
            tl.extra.cuda.gdc_launch_dependents()
    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)
    if VALID:
        valid = tl.load(valid_ptr)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            if MASKED:
                # Scalar loads once per route/row, shared across hidden columns.
                global_id = tl.load(topk_ids_ptr + token_index * topk_num + i)
                local_id = tl.load(expert_map_ptr + global_id, mask=global_id >= 0, other=-1)
                live = (global_id >= 0) & (local_id >= 0)
            elif WEIGHTED:
                live = tl.load(topk_ids_ptr + token_index * topk_num + i) >= 0
            else:
                live = True
            mask = (offs_dim < dim_end) & live
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=mask, other=0.0)
            if WEIGHTED:
                weight = tl.load(weights_ptr + token_index * topk_num + i)
                accumulator = tl.where(
                    live, tl.fma(weight, tmp.to(tl.float32), accumulator), accumulator
                )
            else:
                accumulator += tmp
        values = accumulator.to(output_ptr.dtype.element_ty)
        if SHARED:
            gate = tl.load(gate_ptr + token_index * gate_stride_0).to(tl.float32)
            shared = tl.load(
                shared_ptr + token_index * shared_stride_0 + offs_dim,
                mask=offs_dim < dim_end,
                other=0.0,
            ).to(tl.float32)
            values = shared_join(accumulator, shared, gate, output_ptr.dtype.element_ty)
        if VALID:
            values = tl.where(token_index < valid, values, 0.0).to(output_ptr.dtype.element_ty)
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(store_t_ptr, values, mask=offs_dim < dim_end)
        if EXCHANGE:
            tile = token_index * tl.cdiv(hidden_dim, BLOCK_DIM) + dim_block_id
            publish(
                xch, values[None, :], token_index, dim_start, token_num, hidden_dim, tile,
                XCH_RANK, XCH_WORLD,
            )  # fmt: skip


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_FP8: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    GDC_EARLY: tl.constexpr = False,
    expert_map_ptr=None,
    c_scale_ptr=None,
    stride_csm=0,
    SWIGLU_QUANT: tl.constexpr = False,
    act_ptr=None,
    stride_actm=0,
    ACT_OUT: tl.constexpr = False,
    lora_low_ptr=None,
    lora_gate_ptr=None,
    lora_up_ptr=None,
    lora_scale_ptr=None,
    lora_rank=0,
    stride_llm=0,
    stride_lls=0,
    stride_lbe=0,
    stride_lbn=0,
    LORA_R: tl.constexpr = 0,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    - BLOCK_FP8: A and B are E4M3 with multiplying scales a_scale (num_tokens or
        routes, ceil(K / 128)) and b_scale (E, ceil(N / 128), ceil(K / 128)); each
        BLOCK_SIZE_K step (a divisor of 128) is one FP32 dot scaled by its group's
        scales.

    - SWIGLU_QUANT (block FP8, BLOCK_SIZE_N = 256, B = gate rows then up rows, N = 2I): the
        tile gathers gate group pid_n (columns 0-127) and the matching up group (128-255)
        from B in place; the epilogue rounds both to compute_type, takes SiLU(gate) * up
        (the package activation's PTX), rounds it and writes its 1x128 E4M3 quantization to
        C [rows, I] and the FP32 scale to c_scale [rows, I / 128]: the bits of FC1, then
        ``linear.block_fp8_gated.silu_and_mul_quantize_fp8``, without the BF16 round trip.

    A block whose expert is -1 (a route to another rank's expert) writes zeros.
    Optional expert_map_ptr translates global block IDs to local weight/scale
    rows before this check; a missing pointer compiles out the translation.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    if USE_GDC:
        # Programmatic dependent launch: this grid may start while its producer drains;
        # every load below waits for the producer's writes. Dependents wait for ours; small
        # grids let them launch at once, large ones only as their CTAs finish (resident
        # waiting dependents would take SM slots from the remaining tiles).
        tl.extra.cuda.gdc_wait()
        if GDC_EARLY:
            tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    if SWIGLU_QUANT:
        column = tl.arange(0, BLOCK_SIZE_N)
        offs_bn = tl.where(column < 128, pid_n * 128 + column, N // 2 + pid_n * 128 + column - 128)
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_map_ptr is not None:
        off_experts = tl.load(expert_map_ptr + off_experts, mask=off_experts >= 0, other=-1).to(
            tl.int64
        )
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SWIGLU_QUANT and off_experts == -1:
        return  # FC2 writes zero rows for these routes and never reads their activation
    if off_experts == -1:
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            k_mask = offs_k < BLOCK_SIZE_K
        else:
            k_mask = offs_k < K - k * BLOCK_SIZE_K
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # We accumulate along the K dimension.
        if BLOCK_FP8:
            group = k * BLOCK_SIZE_K // 128
            a_scale = tl.load(
                a_scale_ptr + offs_token // top_k * stride_asm + group * stride_ask,
                mask=token_mask,
                other=0.0,
            )
            b_scale = tl.load(
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn // 128 * stride_bsn
                + group * stride_bsk
            ).to(tl.float32)
            accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale[None, :])
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if SWIGLU_QUANT:
        pair = tl.permute(tl.reshape(accumulator, (BLOCK_SIZE_M, 2, 128)), (0, 2, 1))
        gate, up = tl.split(pair)
        cols = pid_n * 128 + tl.arange(0, 128)
        if LORA_R > 0:  # gate/up += scale * low[route, s] . B_s[expert, cols]^T, in FP32
            scale = tl.load(lora_scale_ptr)
            gate += scale * _lora_columns(
                lora_low_ptr, lora_gate_ptr, offs_token, token_mask, off_experts, cols,
                lora_rank, stride_llm, stride_lbe, stride_lbn, LORA_R,
            )  # fmt: skip
            up += scale * _lora_columns(
                lora_low_ptr + stride_lls, lora_up_ptr, offs_token, token_mask, off_experts,
                cols, lora_rank, stride_llm, stride_lbe, stride_lbn, LORA_R,
            )  # fmt: skip
        g = gate.to(compute_type).to(tl.float32)
        u = up.to(compute_type).to(tl.float32)
        # SiLU(gate) * up as linear.block_fp8_gated computes it (SGLang's accurate form).
        v = silu(g) * u
        v = v.to(compute_type)
        if ACT_OUT:  # the rounded product, e.g. for a down-projection LoRA
            tl.store(
                act_ptr + stride_actm * offs_token[:, None] + cols[None, :], v,
                mask=token_mask[:, None],
            )  # fmt: skip
        v = v.to(tl.float32)
        amax = tl.reduce(tl.abs(v), 1, _max_nan)
        amax = tl.maximum(amax, 1e-10, propagate_nan=tl.PropagateNan.ALL)
        multiplier = libdevice.div_rn(tl.full(amax.shape, 448.0, tl.float32), amax)
        q = v * multiplier[:, None]
        q = tl.minimum(tl.maximum(q, -448.0, propagate_nan=tl.PropagateNan.ALL), 448.0,
                       propagate_nan=tl.PropagateNan.ALL)  # fmt: skip
        tl.store(
            c_ptr + stride_cm * offs_token[:, None] + stride_cn * cols[None, :],
            q.to(c_ptr.dtype.element_ty),
            mask=token_mask[:, None],
        )
        tl.store(c_scale_ptr + offs_token * stride_csm + pid_n, amax * (1.0 / 448.0),
                 mask=token_mask)  # fmt: skip
        return

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    # The output's dtype: FP32 C keeps weighted routes unrounded until the reduction.
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ----------------------------------------------------------------- frozen block-FP8 expert dX

_DX_BLOCK_M = 64  # sorted routes per program (the alignment block)
_DX_BLOCK_N = 64  # output columns per program


@triton.jit
def _expert(experts, mapping, block, MAPPED: tl.constexpr):
    expert = tl.load(experts + block)
    if MAPPED:
        expert = tl.where(expert >= 0, tl.load(mapping + tl.maximum(expert, 0)), -1)
    return expert


@triton.jit
def _fc2_backward_kernel(
    dy, w2, s2, routes, experts, padded, mapping, weights, gu, q, qs, dgdu, dw_partials,
    R, H: tl.constexpr, INTER: tl.constexpr, TOP_K: tl.constexpr, TILES: tl.constexpr,
    MAPPED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):  # fmt: skip
    block = tl.program_id(0)
    pid_n = tl.program_id(1)
    if block * BLOCK_M >= tl.load(padded):
        return
    expert = _expert(experts, mapping, block, MAPPED)
    if expert < 0:
        return
    route = tl.load(routes + block * BLOCK_M + tl.arange(0, BLOCK_M))
    valid = route < R
    route = tl.where(valid, route, 0).to(tl.int64)
    token = route // TOP_K
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, 128)
    weight = tl.load(weights + route, mask=valid, other=0.0)
    weighted = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    unweighted = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    base = w2 + expert.to(tl.int64) * H * INTER
    for kb in range(H // 128):
        rows = kb * 128 + k
        a = tl.load(dy + token[:, None] * H + rows[None, :], mask=valid[:, None], other=0.0)
        aw = (a.to(tl.float32) * weight[:, None]).to(tl.bfloat16)
        b = tl.load(base + rows[:, None] * INTER + n[None, :]).to(tl.bfloat16)
        factor = tl.load(s2 + expert.to(tl.int64) * (H // 128) * (INTER // 128) + kb * (INTER // 128)
                         + pid_n * BLOCK_N // 128).to(tl.float32)  # fmt: skip
        weighted += tl.dot(aw, b) * factor
        unweighted += tl.dot(a, b) * factor
    g = tl.load(gu + route[:, None] * (2 * INTER) + n[None, :], mask=valid[:, None], other=0.0)
    u = tl.load(
        gu + route[:, None] * (2 * INTER) + INTER + n[None, :], mask=valid[:, None], other=0.0
    )
    g, u = g.to(tl.float32), u.to(tl.float32)
    sig = sigmoid(g)  # torch's silu backward
    out = dgdu + route[:, None] * (2 * INTER) + n[None, :]
    tl.store(out, (weighted * u * sig * (1 + g * (1 - sig))).to(tl.bfloat16), valid[:, None])
    tl.store(out + INTER, (weighted * g * sig).to(tl.bfloat16), mask=valid[:, None])
    act = tl.load(q + route[:, None] * INTER + n[None, :], mask=valid[:, None], other=0.0)
    act = act.to(tl.float32) * tl.load(qs + route * (INTER // 128) + pid_n * BLOCK_N // 128,
                                       mask=valid, other=0.0)[:, None]  # fmt: skip
    tl.store(dw_partials + route * TILES + pid_n, tl.sum(act * unweighted, axis=1), mask=valid)


@triton.jit
def _grouped_dgrad_kernel(
    a_ptr, w, s, routes, experts, padded, mapping, out,
    R, stride_a, stride_out, N: tl.constexpr, K: tl.constexpr, ROW_DIVISOR: tl.constexpr,
    MAPPED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):  # fmt: skip
    """out[r] = a[r // ROW_DIVISOR] . W_hat[e_r] over sorted route blocks: each 128-row block
    of the E4M3 [N, K] weight is exact in BF16, its dot accumulates in FP32 and takes its
    scale in FP32; blocks are summed in order (one program per output tile)."""
    block = tl.program_id(0)
    pid_n = tl.program_id(1)
    if block * BLOCK_M >= tl.load(padded):
        return
    expert = _expert(experts, mapping, block, MAPPED)
    if expert < 0:
        return
    route = tl.load(routes + block * BLOCK_M + tl.arange(0, BLOCK_M))
    valid = route < R
    route = tl.where(valid, route, 0).to(tl.int64)
    row = route // ROW_DIVISOR
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    base = w + expert.to(tl.int64) * N * K
    for kb in range(N // 128):
        rows = kb * 128 + k
        a = tl.load(a_ptr + row[:, None] * stride_a + rows[None, :], mask=valid[:, None], other=0.0)
        b = tl.load(base + rows[:, None] * K + n[None, :]).to(a.dtype)
        factor = tl.load(s + expert.to(tl.int64) * (N // 128) * (K // 128)
                         + kb * (K // 128) + pid_n * BLOCK_N // 128).to(tl.float32)  # fmt: skip
        acc += tl.dot(a, b) * factor
    tl.store(out + route[:, None] * stride_out + n[None, :], acc.to(out.dtype.element_ty),
             mask=valid[:, None])  # fmt: skip


@triton.jit
def _local(ids, mapping, route, MAPPED: tl.constexpr):
    expert = tl.load(ids + route)
    if MAPPED:
        expert = tl.where(expert >= 0, tl.load(mapping + tl.maximum(expert, 0)), -1)
    return expert >= 0


@triton.jit
def _finish_kernel(
    route_dx, dw_partials, ids, mapping, dx, dw, M,
    H: tl.constexpr, TOP_K: tl.constexpr, TILES: tl.constexpr, MAPPED: tl.constexpr,
    BLOCK: tl.constexpr, TILES_P2: tl.constexpr,
):  # fmt: skip
    """dx[t] = BF16(sum of local routes' FP32 route_dx, in slot order); dw[r] = the sum of its
    tile partials in tile order, 0 for remote routes."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        route = token * TOP_K + slot
        local = _local(ids, mapping, route, MAPPED)
        total += tl.load(route_dx + route * H + cols, mask=mask & local, other=0.0)
    tl.store(dx + token * H + cols, total.to(tl.bfloat16), mask=mask)
    if tl.program_id(1) == 0:
        tiles = tl.arange(0, TILES_P2)
        for slot in tl.static_range(TOP_K):
            route = token * TOP_K + slot
            local = _local(ids, mapping, route, MAPPED)
            parts = tl.load(
                dw_partials + route * TILES + tiles, mask=(tiles < TILES) & local, other=0.0
            )
            tl.store(dw + route, tl.sum(parts, axis=0))


DX_BLOCK_M = _DX_BLOCK_M  # the route alignment block the caller sorts with


def block_fp8_moe_dgrad(a, w, s, topk_ids, layout, *, row_divisor, expert_map=None, out=None):
    """Grouped frozen block-FP8 dgrad: ``out[r] = a[r // row_divisor] . W_hat[e_r]`` for every
    route r = token * top_k + slot whose expert is on this rank.

    ``a`` BF16/FP16 [T or R, N] (unit column stride, any row stride); ``w`` E4M3 [E_local, N,
    K] with ``s`` [E_local, N / 128, K / 128] multiplying scales, N and K multiples of 128;
    ``layout`` = ``moe_impl.moe_align_block_size(topk_ids, DX_BLOCK_M, num_global_experts)``
    with ``expert_map`` (global -> local, -1 remote) when ids are global. ``out`` BF16/FP16/FP32
    [R, K] (unit column stride; new FP32 if None). Rows of remote or -1 routes are not
    written (no work). Deterministic: one program per output tile, 128-row blocks in order.
    The frozen-base dX of FC2 (a = dy, row_divisor = top_k) and FC1 (a = dgdu, 1)."""
    routes, experts, padded = layout
    r = topk_ids.numel()
    e, n, k = w.shape
    if out is None:
        out = torch.empty((r, k), dtype=torch.float32, device=a.device)
    mapped = expert_map is not None
    _grouped_dgrad_kernel[(routes.numel() // _DX_BLOCK_M, -(-k // _DX_BLOCK_N))](
        a, w, s, routes, experts, padded, expert_map if mapped else topk_ids, out,
        r, a.stride(0), out.stride(0), n, k, row_divisor, mapped, _DX_BLOCK_M, _DX_BLOCK_N,
        num_warps=4,
    )  # fmt: skip
    return out


def block_fp8_moe_dx(op, dy, topk_weights, topk_ids, gu, q, qs, layout):
    """Frozen block-FP8 expert dX and router dW (the ``_moe_block_fp8_dx`` contract): over
    routes sorted by expert, FC2's backward ``dact = BF16(dy * w_route) . W2_hat`` takes the
    SiLU backward on the saved BF16 gate/up and ``dW_route = sum(dequantized FP8 act * (dy .
    W2_hat))``; FC1's backward ``dgdu . W13_hat`` stays FP32 until the slot sum (slot order;
    dW partials in tile order: deterministic). Each 128-row block's exact BF16 dot of the raw
    E4M3 weights accumulates in FP32 and takes its scale in FP32; no weight copy.

    Returns (BF16 dX [M, H], FP32 router dW [M, top_k]) of ``op`` (a ``BlockFP8MoE``) from the
    forward's saved BF16 gate/up [R, 2I] and quantized activation (q, qs) [R, I];
    ``layout`` is ``moe_align_block_size(topk_ids, DX_BLOCK_M, num_global_experts)``."""
    m, r = dy.shape[0], topk_ids.numel()
    h, i, top_k = op.h, op.i, op.top_k
    routes, experts, padded = layout
    mapped = op.expert_map is not None
    mapping = op.expert_map if mapped else topk_ids
    tiles = i // _DX_BLOCK_N
    blocks = routes.numel() // _DX_BLOCK_M
    dgdu = torch.empty((r, 2 * i), dtype=torch.bfloat16, device=dy.device)
    dw_partials = torch.empty((r, tiles), dtype=torch.float32, device=dy.device)
    route_dx = torch.empty((r, h), dtype=torch.float32, device=dy.device)
    _fc2_backward_kernel[(blocks, tiles)](
        dy, op.w2, op.s2, routes, experts, padded,
        mapping, topk_weights.view(-1), gu, q, qs, dgdu, dw_partials,
        r, h, i, top_k, tiles, mapped, _DX_BLOCK_M, _DX_BLOCK_N, num_warps=4,
    )  # fmt: skip
    block_fp8_moe_dgrad(
        dgdu, op.w13, op.s13, topk_ids, layout, row_divisor=1, expert_map=op.expert_map,
        out=route_dx,
    )  # fmt: skip
    dx_out = torch.empty((m, h), dtype=torch.bfloat16, device=dy.device)
    dw = torch.empty((m, top_k), dtype=torch.float32, device=dy.device)
    block = 512
    _finish_kernel[(m, -(-h // block))](
        route_dx, dw_partials, topk_ids.view(-1),
        op.expert_map if mapped else topk_ids.view(-1), dx_out, dw.view(-1), m,
        h, top_k, tiles, mapped, block, triton.next_power_of_2(tiles), num_warps=4,
    )  # fmt: skip
    return dx_out, dw
