# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own Qwen3.5-MoE decode-block Triton tree (98e1e538); CuTe rewrite pending.
"""Decode-block launch rules and op-owned workspace; CUDA imports are lazy."""

import torch

CTL_WORDS = 2048
G1D_STRIDE = 32
TL = 32

HIDDEN, EXPERTS, TOP_K = (2048, 256, 8)
CHUNK = 256
CHUNK_UNIFIED = 1024
CTAS_PER_SM = 2
ROUTER_SPLIT = 4
_SMS = {}
_UNIFIED = {}
WS_ORDER = (
    "ctl",
    "cnt",
    "g1d",
    "touched",
    "lists",
    "plog",
    "gam",
    "ploc",
    "pw",
    "need",
    "red",
    "act",
    "y",
)
STATE = ("ctl", "cnt", "g1d", "touched", "gam", "red")
SCRATCH = tuple((k for k in WS_ORDER if k not in STATE))


def _sm_count(device):
    key = device.index
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[key]


_RESIDENT = {}


def _resident(device, ctas, args, options):
    """The grid, at most the CTAs of this compiled kernel that the device holds at once
    (registers, shared memory and threads per SM from its properties): the kernel's static
    first wave needs every CTA resident. Compiled (not launched) on the first call of a
    configuration, then cached."""
    from .kernels import moe_block_kernel

    key = (device.index, ctas, tuple(sorted(options.items())))
    if key not in _RESIDENT:
        kernel = moe_block_kernel.warmup(*args, grid=(ctas,), **options)
        kernel._init_handles()
        props = torch.cuda.get_device_properties(device)
        warps = kernel.metadata.num_warps
        per_warp = -(-kernel.n_regs * 32 // 256) * 256
        by_regs = props.regs_per_multiprocessor // max(per_warp, 1) // warps
        by_smem = props.shared_memory_per_multiprocessor // (kernel.metadata.shared + 1024)
        by_threads = props.max_threads_per_multi_processor // (32 * warps)
        per_sm = min(by_regs, by_smem, by_threads)
        if per_sm < 1:
            raise RuntimeError("moe_block_kernel does not fit on one SM of this device")
        _RESIDENT[key] = min(ctas, per_sm * _sm_count(device))
    return _RESIDENT[key]


def _parts(width, shared_width, chunk):
    """name -> (elements, dtype, fill) for ``chunk`` rows; state parts carry their initial fill."""
    rows = chunk * TOP_K + chunk
    ia = max(width, shared_width)
    i32, f32, bf16 = (torch.int32, torch.float32, torch.bfloat16)
    return {
        "ctl": (CTL_WORDS, i32, 0),
        "cnt": (EXPERTS, i32, 0),
        "g1d": (EXPERTS * G1D_STRIDE, i32, 0),
        "touched": (EXPERTS * TL, i32, 0),
        "lists": (EXPERTS * chunk, i32, 0),
        "plog": (ROUTER_SPLIT * chunk * (EXPERTS + 16), f32, 0),
        "gam": (chunk, i32, -1),
        "ploc": (chunk * TOP_K, i32, 0),
        "pw": (chunk * TOP_K, f32, 0),
        "need": (chunk, i32, 0),
        "red": (chunk * (HIDDEN // 16), i32, 0),
        "act": (rows * ia, bf16, 0),
        "y": (rows * HIDDEN, f32, 0),
    }


def _unified(device):
    """Whether the GPU shares system DRAM."""
    key = device.index
    if key not in _UNIFIED:
        _UNIFIED[key] = bool(torch.cuda.get_device_properties(device).is_integrated)
    return _UNIFIED[key]


def _config(tokens, unified=False):
    """Preserved LPDDR rules: whole-row GEMVs for sparse experts, streaming MMA
    strips for larger groups, and evict-first hints on once-read weight tiles.
    """
    bt = 1 if tokens == 1 else 2
    cfg = {
        "BT_R": 16,
        "BE_R": 16,
        "KS": ROUTER_SPLIT,
        "BT_K": bt,
        "BM": 16,
        "BI": 16,
        "BK1": 128,
        "BH": 64,
        "BK2": 128,
        "BMS": 16,
        "num_warps": 4,
        "num_stages": 3,
    }
    if tokens <= 2:
        cfg.update(BI=16, BH=32, BK1=128, num_stages=4, PRE2=True)
    elif tokens <= 16:
        cfg.update(BI=8, BH=32, BK1=256, num_stages=4)
    else:
        cfg.update(BT_R=32, SPRE=True, BHS=32)
    if unified:
        cfg.update(BH=32, PRE2=True, EVF=True)
        if tokens <= 4:
            cfg.update(GV=1 if tokens == 1 else 2 if tokens == 2 else 4, GR=2, SGV=0)
        else:
            if tokens <= 16:
                cfg.update(BT_K=16, SBK=2)
            if tokens <= 128:
                cfg.update(GVD=4, GVS=True, GR=2, GVL=16)
            if tokens > 256:
                cfg.update(BMG=64, BMSG=64, BK1=256, num_stages=2)
            elif tokens > 16:
                cfg.update(BK1=512, num_stages=2)
    return cfg


def admits(*, device_class, **shape):
    from .._block import admits_shape

    return device_class == "gb10" and admits_shape(device_class, **shape)


def chunk(device):
    """Rows per launch; a workspace never holds more."""
    return CHUNK_UNIFIED if _unified(device) else CHUNK


def workspace_spec(rows, chunk, width, shared_width, local):
    """Tensors for calls of up to ``rows`` rows as (name, elements, dtype, fill): state sized
    for a whole chunk and initialized once (the kernel restores it before returning on every
    call), uninitialized scratch, and the shared-part pair."""
    state = _parts(width, shared_width, chunk)
    scratch = _parts(width, shared_width, min(rows, chunk))
    spec = tuple(
        (name, *state[name]) if name in STATE else (name, *scratch[name][:2], None)
        for name in WS_ORDER
    )
    return (*spec, ("shared_part", 2, torch.int32, (0, 1)))


def bind_workspace(tensors, rows, chunk, width, shared_width, local):
    """Workspace over ``tensors`` (name -> tensor, each at least its workspace_spec size)."""
    ws = {name: tensors[name] for name in (*WS_ORDER, "shared_part")}
    ws["gam"] = ws["gam"].view(torch.float32)
    ws["shape"] = (rows, width, shared_width, local)
    ws["device"] = tensors["ctl"].device
    return ws


def routing_weights(workspace, rows):
    """FP32 [rows, TOP_K] view of the latest call's routing weights (None past one chunk)."""
    pw = workspace["pw"]
    if rows * TOP_K > pw.numel():
        return None
    return pw[: rows * TOP_K].view(rows, TOP_K)


def run_block(
    x,
    router_w,
    shared_gate_w,
    w_gate_up,
    w_down,
    shared_gate_up,
    shared_down,
    expert_map,
    valid_tokens,
    out,
    routing_buf,
    workspace,
):
    from .._block import hardware, validate_block

    shape = validate_block(
        x,
        router_w,
        w_gate_up,
        w_down,
        shared_gate_up,
        shared_down,
        shared_gate_w,
        expert_map=expert_map,
        valid_tokens=valid_tokens,
        out=out,
    )
    if expert_map is None or valid_tokens is None or out is None:
        raise ValueError("run_block requires expert_map, valid_tokens and out tensors")
    expected = (shape["inter"], shape["shared_inter"], shape["local_experts"])
    capacity, *block = workspace.get("shape", (0,))
    if tuple(block) != expected or workspace.get("device") != x.device or capacity < x.shape[0]:
        raise ValueError("workspace must match this device and block shape and hold T rows")
    if not admits(device_class=hardware(x.device), **shape):
        raise ValueError("triton_block_gb10 does not admit this device or shape")
    if (
        routing_buf.shape != (x.shape[0], TOP_K)
        or routing_buf.dtype != torch.uint8
        or routing_buf.device != x.device
        or not routing_buf.is_contiguous()
    ):
        raise ValueError("routing_buf must be contiguous uint8 [T,8] on the input device")
    from .kernels import moe_block_kernel

    tokens = x.shape[0]
    local, width, shared_width = shape["local_experts"], shape["inter"], shape["shared_inter"]
    unified = _unified(x.device)
    chunk = CHUNK_UNIFIED if unified else CHUNK
    wsa = [workspace[k].data_ptr() for k in WS_ORDER]
    weights = [
        v.data_ptr()
        for v in (
            router_w,
            shared_gate_w,
            w_gate_up,
            w_down,
            shared_gate_up,
            shared_down,
            workspace["shared_part"],
            expert_map,
            valid_tokens,
        )
    ]
    with torch.cuda.device(x.device):
        ctas = _sm_count(x.device) * CTAS_PER_SM
        for row0 in range(0, tokens, chunk):
            rows = min(chunk, tokens - row0)
            cfg = _config(rows, unified)
            if unified and rows > 64:
                cfg.setdefault("LA", 1)
            options = dict(
                H=HIDDEN,
                E=EXPERTS,
                E_LOC=local,
                TOPK=TOP_K,
                I=width,
                SI=shared_width,
                IA=max(width, shared_width),
                EV=0,
                TRACE=0,
                **cfg,
            )
            args = [
                x[row0].data_ptr(),
                *weights,
                out[row0].data_ptr(),
                routing_buf[row0].data_ptr(),
                *wsa,
                workspace["ctl"].data_ptr(),
                rows,
                row0,
                tokens,
                ctas,
            ]
            args[-1] = _resident(x.device, ctas, args, options)
            moe_block_kernel[(args[-1],)](*args, **options)
    return out
