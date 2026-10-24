# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own Qwen3.5-MoE decode-block Triton tree (548eb81d); CuTe rewrite pending.
"""Decode-block launch rules and op-owned workspace; CUDA imports are lazy."""

import torch

CTL_WORDS = 192
G1D_STRIDE = 32
TL = 32

HIDDEN, EXPERTS, TOP_K = (2048, 256, 8)
CHUNK = 256
CTAS_PER_SM = 2
ROUTER_SPLIT = 4
_SMS = {}
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
EARLY_SHARED_DOWN_MAX_SMS = 176


def _early_shared_down(device):
    return _sm_count(device) <= EARLY_SHARED_DOWN_MAX_SMS


SMALL_T1_GRID = 320
T1_GRID_ALL = True
SMALL_ROWS_RULE = True
PRO_WC = True
EP2_T2_GRID = 340
EP2_T2_RULE = True
GEFORCE_EP2_WC = True
GEFORCE_EP2_WC_ROWS = 64
GEFORCE_POST = True
GEFORCE_POST2 = True
PRO_DAF = True
PRO_DAF10 = True
GEFORCE_DAF10 = True


def _small_rows_rule(device, rows, local, invariant):
    """(early_shared_down for the chunk's configuration, its grid size or None for the default, extra
    configuration keys)."""
    esd = _early_shared_down(device)
    if SMALL_ROWS_RULE and local == EXPERTS // 4 and (rows <= 2) and (not invariant):
        grid = SMALL_T1_GRID if rows == 1 and (T1_GRID_ALL or not esd) else None
        return (True, grid, {"DAF": 3} if PRO_DAF and (not esd) and (rows == 2) else {})
    if (
        SMALL_ROWS_RULE
        and EP2_T2_RULE
        and (not esd)
        and (local == EXPERTS // 2)
        and (rows == 2)
        and (not invariant)
    ):
        return (True, EP2_T2_GRID, {"CBR": True})
    if (
        SMALL_ROWS_RULE
        and PRO_DAF10
        and (not esd)
        and (local == EXPERTS // 4)
        and (rows == 3)
        and (not invariant)
    ):
        return (esd, None, {"DAF": 10})
    if (
        SMALL_ROWS_RULE
        and GEFORCE_DAF10
        and esd
        and (local == EXPERTS // 4)
        and (rows == 3)
        and (not invariant)
    ):
        return (esd, None, {"DAF": 10})
    return (esd, None, {})


def _sm_count(device):
    key = device.index
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[key]


INIT_PDL = True


def router_skip(device, rows, cfg):
    return _early_shared_down(device) and cfg["BT_R"] == 16 and (rows > 16)


def _words(n):
    return (n + 31) // 32 * 32


_LAYOUTS = {}


def _layout(rows, width, shared_width, local):
    """Word offsets of a call workspace's segments (cached per shape): the counted and polled
    state first (control words, per-expert counts, gate/up completion counts, plan lines,
    arrival counters: n_zero words cleared to 0; then the gamma sentinels: n_sent words set to
    -1 bits; both by router_kernel before each chunk's main kernel), then scratch that every
    launch writes before it reads (token lists, partial logits, slots, weights, arrival
    counts, activations, per-slot outputs)."""
    key = (rows, width, shared_width, local)
    lay = _LAYOUTS.get(key)
    if lay is None:
        ia = max(width, shared_width)
        arows = rows * TOP_K + rows
        nu = min(local, rows * TOP_K)
        segs = [
            ("ctl", CTL_WORDS, torch.int32),
            ("cnt", local, torch.int32),
            ("g1d", nu * G1D_STRIDE, torch.int32),
            ("touched", nu * TL, torch.int32),
            ("red", rows * (HIDDEN // 16), torch.int32),
            ("gam", rows, torch.float32),
            ("lists", local * rows, torch.int32),
            ("plog", ROUTER_SPLIT * rows * (EXPERTS + 16), torch.float32),
            ("ploc", rows * TOP_K, torch.int32),
            ("pw", rows * TOP_K, torch.float32),
            ("need", rows, torch.int32),
            ("act", (arows * ia + 1) // 2, torch.bfloat16),
            ("y", arows * HIDDEN, torch.float32),
        ]
        off, pos = ({}, 0)
        for name, n, _ in segs:
            off[name] = pos
            pos += _words(n)
        lay = _LAYOUTS[key] = {
            "segs": segs,
            "off": off,
            "total": pos,
            "n_zero": off["gam"],
            "n_sent": rows,
        }
    return lay


def _config(tokens, local=EXPERTS, invariant=False, early_shared_down=False):
    """Preserved row/device rules: preload down tiles while routing, narrow sparse
    gate/up tiles, and move routing weights/combine addresses ahead of dependent
    loads. Dynamic valid-row variants keep padded graph launches on these paths.
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
        "PEEK": True,
        "CB": tokens <= 64 and (local < EXPERTS or tokens <= 2),
        "num_warps": 4,
        "num_stages": 3,
    }
    mid_small = (
        tokens <= 2
        and local < EXPERTS
        and (early_shared_down or tokens == 2 or local > EXPERTS // 4)
    )
    if tokens <= 2 and (not mid_small):
        cfg.update(BI=16, BIN=8 if local < EXPERTS else 0, BH=32, BK1=128, num_stages=4, PRE2=True)
        if local == EXPERTS:
            cfg.update(SIGFIRST=True)
    elif tokens <= 16:
        cfg.update(BI=8, BH=32, BK1=256, num_stages=4)
        if local <= EXPERTS // 4 and tokens <= 16 or (local <= EXPERTS // 2 and tokens <= 12):
            cfg.update(SHWAIT=True, SPRE=True, SHDW=not early_shared_down)
        elif local <= EXPERTS // 2:
            cfg.update(SPRE=True, VSPRE=12, VSHW=12, SHDW=not early_shared_down)
        if local < EXPERTS and tokens > bt and (local <= EXPERTS // 4 or tokens <= 12):
            cfg.update(ONEB=True)
        if local < EXPERTS and tokens > 2:
            cfg.update(GCG=True)
    else:
        cfg.update(BT_R=16 if tokens <= 64 else 32, SPRE=True, BHS=32)
        if local <= EXPERTS // 2:
            cfg.update(BK1=256, num_stages=2)
        if EXPERTS // 4 < local <= EXPERTS // 2 and tokens <= 64:
            cfg.update(DPFT=16, DPFT_N=8)
            cfg.update(ONEB=True, ONI=True)
            if not early_shared_down:
                cfg.update(SIGFIRST=True)
            else:
                cfg.update(PSW=True, DPFT_PSW=True, PSW_V=16)
        if local <= EXPERTS // 4 and tokens <= 64:
            cfg.update(ONEB=True)
            cfg.update(VBIN=8, VBIN_MIN=2, VBIN_MAX=7, LRZ=True)
            if early_shared_down:
                cfg.update(VMIX_=2, VMIX_MIN=1, VMIX_MAX=1, NIG=True)
        if local <= EXPERTS // 4 and tokens > 64 and (not invariant):
            cfg.update(BI=8)
    if tokens == 1 and (not early_shared_down):
        if EXPERTS // 4 < local <= EXPERTS // 2:
            cfg.update(ONI=True)
        else:
            cfg.update(maxnreg=232)
    if local <= EXPERTS // 4 and tokens > 64 and (not invariant):
        cfg["SIGFIRST"] = True
    if local == EXPERTS and 9 <= tokens <= 16:
        cfg["DST"] = 3
    if early_shared_down and EXPERTS // 4 < local <= EXPERTS // 2 and (3 <= tokens <= 12):
        cfg["SIGFIRST"] = True
    if early_shared_down and local == EXPERTS // 4 and (tokens == 1) and (not invariant):
        cfg["SIGFIRST"] = True
        cfg["XB1"] = True
    if early_shared_down and local == EXPERTS // 4 and (tokens <= 2) and (not invariant):
        cfg["SDPF"] = True
    if early_shared_down and local == EXPERTS and (tokens == 1) and (not invariant):
        cfg.update(BH=16, BI=8)
    if tokens == 1 and (not early_shared_down) and (local < EXPERTS) and (not invariant):
        if cfg.get("SHWAIT") and cfg.get("SHDW", True):
            cfg.update(SDPFW=True, maxnreg=232)
        elif cfg.get("PRE2") or cfg.get("SPRE"):
            cfg["SDPF"] = True
    if early_shared_down and local == EXPERTS // 4 and (3 <= tokens <= 5) and (not invariant):
        cfg.update(PSW=True, PSW_NS=True)
    if early_shared_down and local == EXPERTS // 4 and (tokens == 2) and (not invariant):
        cfg["DST"] = 3
    if early_shared_down and local == EXPERTS // 4 and not 5 < tokens <= 16:
        cfg["CBR"] = True
    if not early_shared_down and PRO_WC and (tokens <= 64) and (not invariant):
        if local <= EXPERTS // 4 or (local < EXPERTS and tokens <= 16):
            cfg["CBR"] = True
    if (
        early_shared_down
        and GEFORCE_EP2_WC
        and (local == EXPERTS // 2)
        and (2 <= tokens <= GEFORCE_EP2_WC_ROWS)
        and (not invariant)
    ):
        cfg["CBR"] = True
    if early_shared_down and GEFORCE_POST and (not invariant) and (local < EXPERTS):
        if local == EXPERTS // 2 and tokens == 2:
            cfg.update(GCG=True, SDPF=True)
        elif local == EXPERTS // 2 and 17 <= tokens <= (48 if GEFORCE_POST2 else 32):
            cfg.update(DST=3, SDPF=True)
        elif local == EXPERTS // 2 and GEFORCE_POST2 and (49 <= tokens <= 64):
            cfg.update(DST=3)
    if (
        early_shared_down
        and GEFORCE_POST
        and (not invariant)
        and (local == EXPERTS)
        and (9 <= tokens <= 16)
    ):
        cfg["SIGFIRST"] = True
        if GEFORCE_POST2:
            cfg["GCG"] = True
    return cfg


def admits(*, device_class, **shape):
    from .._block import admits_shape

    return device_class in ("rtx5090", "pro6000") and admits_shape(device_class, **shape)


def chunk(device):
    """Rows per launch pair; a workspace never holds more."""
    return CHUNK


def workspace_spec(rows, chunk, width, shared_width, local):
    """Tensors for calls of up to ``rows`` rows as (name, elements, dtype, fill): uninitialized
    scratch, which router_kernel resets on every launch, and the shared-part pair."""
    lay = _layout(min(rows, chunk), width, shared_width, local)
    return (("buffer", lay["total"], torch.int32, None), ("shared_part", 2, torch.int32, (0, 1)))


def bind_workspace(tensors, rows, chunk, width, shared_width, local):
    """Workspace over ``tensors`` (name -> tensor, each at least its workspace_spec size)."""
    return {
        "buffer": tensors["buffer"],
        "layout": _layout(min(rows, chunk), width, shared_width, local),
        "shared_part": tensors["shared_part"],
        "shape": (rows, width, shared_width, local),
        "device": tensors["buffer"].device,
    }


def routing_weights(workspace, rows):
    """FP32 [rows, TOP_K] view of the latest call's routing weights (None past one chunk)."""
    if rows > CHUNK:
        return None
    off = workspace["layout"]["off"]["pw"]
    return workspace["buffer"][off : off + rows * TOP_K].view(torch.float32).view(rows, TOP_K)


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
    """Launch directly on the caller's current stream; workspace belongs to one op."""
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
        raise ValueError("triton_block does not admit this device or shape")
    if (
        routing_buf.shape != (x.shape[0], TOP_K)
        or routing_buf.dtype != torch.uint8
        or routing_buf.device != x.device
        or not routing_buf.is_contiguous()
    ):
        raise ValueError("routing_buf must be contiguous uint8 [T,8] on the input device")
    from .kernels import moe_block_kernel, router_kernel

    tokens = x.shape[0]
    local, width, shared_width = shape["local_experts"], shape["inter"], shape["shared_inter"]
    lay = workspace["layout"]
    base = workspace["buffer"].data_ptr()
    addr = {name: base + 4 * offset for name, offset in lay["off"].items()}
    wsa = [addr[k] for k in WS_ORDER]
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
        for row0 in range(0, tokens, CHUNK):
            rows = min(CHUNK, tokens - row0)
            esd, grid, extra = _small_rows_rule(x.device, rows, local, False)
            cfg = _config(rows, local, False, esd)
            cfg.update(extra)
            nctas = grid or _sm_count(x.device) * CTAS_PER_SM
            tail = [rows, row0, tokens, nctas]
            options = dict(
                H=HIDDEN,
                E=EXPERTS,
                E_LOC=local,
                TOPK=TOP_K,
                I=width,
                SI=shared_width,
                IA=max(width, shared_width),
                EV=0,
                **cfg,
            )
            nr = (EXPERTS // cfg["BE_R"] + 1) * cfg["KS"] * -(-rows // cfg["BT_R"])
            roptions = {
                "H": HIDDEN,
                "E": EXPERTS,
                "KS": cfg["KS"],
                "BT_R": cfg["BT_R"],
                "BE_R": cfg["BE_R"],
                "F32": cfg.get("F32", False),
                "num_warps": cfg["num_warps"],
                "num_stages": cfg["num_stages"],
                "PF": bool(
                    (cfg.get("SHWAIT", False) or cfg.get("VSHW", 0)) and cfg.get("SHDW", True)
                ),
                "SKIP": router_skip(x.device, rows, cfg),
                "LZ": cfg["BM"] if -(-rows // cfg["BT_K"]) > 1 else 0,
            }
            if cfg.get("DAF", 0) > 0:
                roptions.update(
                    YN=(rows * TOP_K + rows) * HIDDEN, YOFF=lay["off"]["y"] - lay["off"]["gam"]
                )
            router_kernel[(nr,)](
                x[row0].data_ptr(),
                router_w.data_ptr(),
                shared_gate_w.data_ptr(),
                valid_tokens.data_ptr(),
                addr["plog"],
                addr["ctl"],
                lay["n_zero"],
                addr["gam"],
                lay["n_sent"],
                *tail[:3],
                addr["lists"],
                local,
                PDL_IN=INIT_PDL,
                launch_pdl=INIT_PDL,
                **roptions,
            )
            moe_block_kernel[(nctas,)](
                x[row0].data_ptr(),
                *weights,
                out[row0].data_ptr(),
                routing_buf[row0].data_ptr(),
                *wsa,
                addr["ctl"],
                *tail,
                launch_pdl=True,
                **options,
            )
    return out
