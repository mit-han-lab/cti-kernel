# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e; see THIRD_PARTY_NOTICES.
# `fla/ops/gated_delta_rule/chunk.py`, `fla/ops/gated_delta_rule/chunk_fwd.py`
# `fla/ops/gated_delta_rule/gate.py`, `fla/ops/common/chunk_o.py`, `fla/ops/utils/solve_tril.py`
# `fla/ops/gated_delta_rule/wy_fast.py`, `fla/ops/common/chunk_scaled_dot_kkt.py`
# `fla/ops/common/chunk_delta_h.py`, `fla/ops/utils/cumsum.py`, `fla/modules/fused_norm_gate.py`

"""Fixed launches for GDN training. Every fla-core import here follows ``pin_fla()``.

Only host orchestration is adapted here. FLA owns the JIT code, shape-derived tiles,
heuristics and arithmetic. No Autotuner.run, benchmarking or process-wide patching.
Varlen callers supply their device-built chunk table; missing tables are rejected to
avoid FLA's host-synchronizing table builder. K=V=128, BT=64 is the training contract.

Each config compiles without spills on sm_120/121, so no launch raises the CUDA stack
limit; where FLA's own candidates all spill (dhu, dqkwg, WY backward) the comment at the
entry says what changed.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from functools import wraps

import torch
import torch.nn.functional as F

DISPATCH_VARIABLE = "FLA_DISABLE_BACKEND_DISPATCH"
FAST_OPS_VARIABLE = "FLA_USE_FAST_OPS"
# The fla-core modules that read each variable, once, when they are imported.
DISPATCH_MODULE, FAST_OPS_MODULE = "fla.ops.backends", "fla.ops.utils.op"


def pin_fla() -> None:
    """Pin fla-core's backend dispatch off and its exact exp/log on, for its modules not yet
    imported. Call it before every fla-core import in ctikernels.gdn: whichever path imports
    fla-core first (the provider, a fixed launch, L2 norm) must not leave it configured
    otherwise. The top-level package may be loaded already without them (transformers
    imports it to look up kernels)."""
    if DISPATCH_MODULE not in sys.modules:
        os.environ[DISPATCH_VARIABLE] = "1"
    if FAST_OPS_MODULE not in sys.modules:
        os.environ[FAST_OPS_VARIABLE] = "0"


# Stages are explicit, including Triton's default 3 for configs that omit them.
# Shape-derived tiles (BK/BV of the host wrappers, NB) stay FLA's, except the dqkwg and WY
# backward entries below, which override them. Every config compiles with 0 spills on sm_120/121.
CONFIGS = {
    "l2norm_fwd_kernel": {"BT": 16, "num_warps": 4, "num_stages": 3},
    "l2norm_bwd_kernel": {"BT": 16, "num_warps": 4, "num_stages": 3},
    "gdn_gate_chunk_cumsum_scalar_kernel": {"num_warps": 4, "num_stages": 3},
    "chunk_scaled_dot_kkt_fwd_kernel": {"BK": 32, "num_warps": 4, "num_stages": 2},
    "merge_16x16_to_64x64_inverse_kernel": {"num_warps": 4, "num_stages": 2},
    "recompute_w_u_fwd_kernel": {"num_warps": 4, "num_stages": 2},
    # 2 warps as FLA on Blackwell (its tl.dot recurrence race note); BV=64 spills, BV=32 does not.
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {"BV": 32, "num_warps": 2, "num_stages": 2},
    "chunk_fwd_kernel_o": {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3},
    "chunk_bwd_kernel_dv_local": {"num_warps": 4, "num_stages": 2},
    # FLA's sm12x list (1 stage) spills 2-470 B, raising the stack limit; 2 stages do not.
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": {"BV": 32, "num_warps": 4, "num_stages": 2},
    # FLA derives BK=BV=64 on sm12x, and every warps/stages choice then spills (14-596 B).
    # BK=BV=32 is FLA's own derivation on smaller-shared-memory devices; 0 spills.
    "chunk_bwd_kernel_dqkwg": {"BK": 32, "BV": 32, "num_warps": 4, "num_stages": 2},
    # FLA restricts Blackwell to 2 warps / 4 stages (fla-org/flash-linear-attention#913), but
    # every 2-warp config spills (8-246 B) at some geometry. 4 warps, 1 stage (no pipelined
    # loads) with FLA's own tiles is spill-free on sm_120/121; tests/gpu/test_gdn_l2norm.py
    # stress-tests it (large grids, head ratios 1-3).
    "prepare_wy_repr_bwd_kernel": {"num_warps": 4, "num_stages": 1},
    "chunk_local_cumsum_scalar_kernel": {"num_warps": 4, "num_stages": 3},
    "layer_norm_gated_fwd_kernel": {"BT": 16, "num_warps": 4, "num_stages": 3},
    # Some 0.5.2 distributions fuse kkt/solve for BT=64. Preserve that chain when present.
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": {"BK": 32, "num_warps": 4, "num_stages": 3},
}


def _input_guard(function):
    """FLA's guard semantics for the three guarded host entries we adapt."""

    @wraps(function)
    def guarded(*args, **kwargs):
        tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
        device = tensors[0].device if tensors else None
        context = (
            torch.cuda.device(device)
            if device is not None and device.type == "cuda"
            else nullcontext()
        )
        with context:
            args = tuple(
                value.contiguous() if isinstance(value, torch.Tensor) else value for value in args
            )
            kwargs = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
            return function(*args, **kwargs)

    return guarded


def _launch(module, name, grid, **arguments):
    """Evaluate Heuristics and unwrap Autotuner; return the actual CompiledKernel.

    These FLA tuners have no pre_hook, reset_to_zero or restore_value obligations.
    Heuristics only inspect pointer presence; evaluate in the original wrapper order.
    """
    pin_fla()
    kernel = getattr(importlib.import_module(module), name)
    arguments = {**arguments, **CONFIGS[name]}
    # Heuristics and Autotuner both proxy arg_names; unwrap by wrapper-specific fields.
    while hasattr(kernel, "values") or hasattr(kernel, "configs"):
        if hasattr(kernel, "values"):
            for key, heuristic in kernel.values.items():
                arguments[key] = heuristic(arguments)
        kernel = kernel.fn
    return kernel[grid](**arguments)


def _tile_override(name, bk, bv):
    config = CONFIGS[name]
    return config.get("BK", bk), config.get("BV", bv)


def _chunks(cu, indices, tokens, size):
    if size != 64:
        raise ValueError("fixed GDN training requires chunk_size=64")
    if cu is not None:
        if indices is None:
            raise ValueError("varlen fixed launches require device-built chunk_indices")
        return len(indices)
    return (tokens + size - 1) // size


def _offsets(cu, size):
    # FLA prepare_chunk_offsets: int32 lengths -> padded int32 counts -> int64 cumsum.
    return F.pad((cu[1:] - cu[:-1] + size - 1) // size, (1, 0)).cumsum(-1)


def _tiles(k, v):
    import triton

    pin_fla()
    from fla.utils import check_shared_mem

    cap = (
        128
        if check_shared_mem("hopper", k.device.index)
        else (64 if check_shared_mem("ada", k.device.index) else 32)
    )
    return min(max(triton.next_power_of_2(k.shape[-1]), 16), cap), min(
        max(triton.next_power_of_2(v.shape[-1]), 16), cap
    )


@_input_guard
def gdn_gate_chunk_cumsum(
    g,
    A_log,
    chunk_size,
    scale=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
    output_dtype=torch.float,
):
    B, T, H = g.shape
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch(
        "fla.ops.gated_delta_rule.gate",
        "gdn_gate_chunk_cumsum_scalar_kernel",
        (nt, B * H),
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=chunk_size,
        REVERSE=False,
    )
    return o


@_input_guard
def chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    head_first=False,
    output_dtype=torch.float,
    chunk_indices=None,
    **kwargs,
):
    if g.ndim != 3 or (cu_seqlens is not None and g.shape[0] != 1):
        raise ValueError("GDN uses scalar gates with B=1 for varlen")
    B, H, T = g.shape if head_first else (g.shape[0], g.shape[2], g.shape[1])
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    o = torch.empty_like(g, dtype=output_dtype or g.dtype)
    _launch(
        "fla.ops.utils.cumsum",
        "chunk_local_cumsum_scalar_kernel",
        (nt, B * H),
        s=g,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return o


def chunk_scaled_dot_kkt_fwd(
    k,
    g=None,
    beta=None,
    cu_seqlens=None,
    chunk_size=64,
    output_dtype=torch.float32,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    HV = beta.shape[2]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    A = torch.empty(B, T, HV, chunk_size, device=k.device, dtype=output_dtype)
    _launch(
        "fla.ops.common.chunk_scaled_dot_kkt",
        "chunk_scaled_dot_kkt_fwd_kernel",
        (nt, B * HV),
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=chunk_size,
    )
    return A


@_input_guard
def solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    pin_fla()
    module = importlib.import_module("fla.ops.utils.solve_tril")
    B, T, H, BT = A.shape
    nt = _chunks(cu_seqlens, chunk_indices, T, BT)
    Ai = torch.zeros_like(A, dtype=A.dtype if output_dtype is None else output_dtype)
    # sm12x defaults to IEEE. Do not silently pick a precision from a multi-valued
    # autotune space: it has no unique fixed equivalent without GPU measurement.
    precisions = module.DOT_PRECISION_AUTOTUNE_LIST
    if len(precisions) != 1:
        raise ValueError("fixed solve_tril requires a single FLA DOT_PRECISION (default: ieee)")
    _launch(
        "fla.ops.utils.solve_tril",
        "merge_16x16_to_64x64_inverse_kernel",
        (nt, B * H),
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=module.IS_TMA_SUPPORTED,
        DOT_PRECISION=precisions[0],
    )
    return Ai


def recompute_w_u_fwd(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
    B, T, H, K = k.shape
    HV, V, BT = v.shape[2], v.shape[-1], A.shape[-1]
    nt = _chunks(cu_seqlens, chunk_indices, T, BT)
    w, u = k.new_empty(B, T, HV, K), torch.empty_like(v)
    _launch(
        "fla.ops.gated_delta_rule.wy_fast",
        "recompute_w_u_fwd_kernel",
        (nt, B * HV),
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
    )
    return w, u


def prepare_wy_repr_bwd(k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
    import triton

    pin_fla()
    from fla.utils import check_shared_mem

    B, T, H, K = k.shape
    HV, V, BT = v.shape[2], v.shape[-1], A.shape[-1]
    nt = _chunks(cu_seqlens, chunk_indices, T, BT)
    cap = 64 if check_shared_mem() else 32
    BK, BV = (min(max(triton.next_power_of_2(d), 16), cap) for d in (K, V))
    BK, BV = _tile_override("prepare_wy_repr_bwd_kernel", BK, BV)
    dk, dv, db = k.new_empty(B, T, HV, K), torch.empty_like(v), torch.empty_like(beta)
    dg = torch.empty_like(g) if g is not None else None
    _launch(
        "fla.ops.gated_delta_rule.wy_fast",
        "prepare_wy_repr_bwd_kernel",
        (nt, B * HV),
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


def chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    HV, V = u.shape[2:]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    n = B if cu_seqlens is None else len(cu_seqlens) - 1
    offsets = None if cu_seqlens is None else _offsets(cu_seqlens, chunk_size)
    dims = (V, K) if state_v_first else (K, V)
    h = k.new_empty(B, nt, HV, *dims)
    ht = k.new_zeros(n, HV, *dims, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None
    _launch(
        "fla.ops.common.chunk_delta_h",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        lambda meta: ((V + meta["BV"] - 1) // meta["BV"], n * HV),
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, ht


def chunk_gated_delta_rule_bwd_dhu(
    q,
    k,
    w,
    do,
    dv,
    g=None,
    gk=None,
    h0=None,
    dht=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    B, T, H, K = q.shape
    HV, V = do.shape[2:]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    n = B if cu_seqlens is None else len(cu_seqlens) - 1
    offsets = None if cu_seqlens is None else _offsets(cu_seqlens, chunk_size)
    dims = (V, K) if state_v_first else (K, V)
    dh = q.new_empty(B, nt, HV, *dims)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)
    _launch(
        "fla.ops.common.chunk_delta_h",
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
        lambda meta: ((V + meta["BV"] - 1) // meta["BV"], n * HV),
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=offsets,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        STATE_V_FIRST=state_v_first,
    )
    return dh, dh0, dv2


def chunk_fwd_o(
    q,
    k,
    v,
    h,
    g=None,
    g_gamma=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    o = torch.empty_like(v)
    _launch(
        "fla.ops.common.chunk_o",
        "chunk_fwd_kernel_o",
        lambda meta: ((V + meta["BV"] - 1) // meta["BV"], nt, B * HV),
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=K**-0.5 if scale is None else scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        STATE_V_FIRST=state_v_first,
    )
    return o


def chunk_bwd_dv_local(
    q,
    k,
    do,
    g=None,
    g_gamma=None,
    A=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    B, T, H, K = k.shape
    HV, V = do.shape[2:]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    BK, BV = _tiles(k, do)
    dv = torch.empty_like(do)
    _launch(
        "fla.ops.common.chunk_o",
        "chunk_bwd_kernel_dv_local",
        (nt, B * HV),
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_bwd_dqkwg(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=None,
    g=None,
    g_gamma=None,
    dv=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    pin_fla()
    from fla.utils import IS_NVIDIA_HOPPER, TRITON_ABOVE_3_4_0, TRITON_ABOVE_3_7_1

    if g is not None and IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1:
        raise RuntimeError("gated chunk_bwd_dqkwg on Hopper requires Triton >= 3.7.1")
    B, T, H, K = k.shape
    HV, V = v.shape[2:]
    nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
    BK, BV = _tile_override("chunk_bwd_kernel_dqkwg", *_tiles(k, v))
    nk = (K + BK - 1) // BK
    dq, dk = q.new_empty(B, T, HV, K), k.new_empty(B, T, HV, K)
    dg = torch.empty(nk, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None
    _launch(
        "fla.ops.common.chunk_o",
        "chunk_bwd_kernel_dqkwg",
        (nk, nt, B * HV),
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
    )
    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dq, dk, dw, dg.sum(0) if dg is not None else None


def layer_norm_gated_fwd(
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    residual=None,
    out_dtype=None,
    residual_dtype=None,
    is_rms_norm=False,
):
    import triton

    T, D = x.shape
    if D != 128:
        raise ValueError("fixed GDN gated norm requires D=128")
    if residual is not None:
        residual_dtype = residual.dtype
        assert residual.shape == (T, D)
    assert weight is None or weight.shape == (D,)
    assert bias is None or bias.shape == (D,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    residual_out = (
        torch.empty(T, D, device=x.device, dtype=residual_dtype)
        if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype)
        else None
    )
    mean = None if is_rms_norm else torch.empty(T, dtype=torch.float, device=x.device)
    rstd = torch.empty(T, dtype=torch.float, device=x.device)
    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
    _launch(
        "fla.modules.fused_norm_gate",
        "layer_norm_gated_fwd_kernel",
        lambda meta: ((T + meta["BT"] - 1) // meta["BT"],),
        x=x,
        g=g,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        T=T,
        D=D,
        BD=BD,
        NB=(T + 65535) // 65536,
        ACTIVATION=activation,
        IS_RMS_NORM=is_rms_norm,
    )
    return y, mean, rstd, residual_out if residual_out is not None else x


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    state_v_first=False,
    cu_seqlens=None,
    cp_context=None,
    chunk_indices=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    chunk_size=64,
):
    if cp_context is not None:
        raise ValueError("GDN training does not use FLA context parallelism")
    g_input = g if use_gate_in_kernel else None
    index = {"cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices}
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(g, A_log, chunk_size, 1.4426950408889634, dt_bias, **index)
    else:
        g = chunk_local_cumsum(g, chunk_size, scale=1.4426950408889634, **index)
    # Keep the installed FLA chain's algorithm: newer 0.5.2 sources fuse the inverse.
    pin_fla()
    chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    if hasattr(chunk, "chunk_gated_delta_rule_fwd_intra"):
        B, T, H, K = k.shape
        HV = beta.shape[2]
        nt = _chunks(cu_seqlens, chunk_indices, T, chunk_size)
        A = torch.zeros(B, T, HV, chunk_size, device=k.device, dtype=k.dtype)
        _launch(
            "fla.ops.gated_delta_rule.chunk_fwd",
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            (nt, B * HV),
            k=k,
            g=g,
            beta=beta,
            A=A,
            **index,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=chunk_size,
            BC=16,
        )
    else:
        A = chunk_scaled_dot_kkt_fwd(k, g, beta, chunk_size=chunk_size, **index)
        A = solve_tril(A, output_dtype=k.dtype, **index)
    w, u = recompute_w_u_fwd(k, v, beta, A, g, **index)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        **index,
    )
    o = chunk_fwd_o(
        q, k, v_new, h, g, scale=scale, state_v_first=state_v_first, chunk_size=chunk_size, **index
    )
    return g, o, A, final, initial_state, g_input
