# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Offline sweep of the Triton GDN training kernels' launch configurations on this GPU: each
kernel alone, on one training step's real operands, per candidate configuration. Prints one
``ROW`` JSON line per (kernel, configuration) with the median time, registers and spills; the
fastest spill-free row per kernel is what ``gdn/triton/_train_configs.py`` records.

    python benchmark/kernel_tuning/gdn_training_triton_sweep.py [--geometry 16,32] [--tokens 16384]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch

from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.triton import _train_conv as C
from ctikernels.gdn.triton import _train_features as F
from ctikernels.gdn.triton import _train_intra as I
from ctikernels.gdn.triton import _train_output as O
from ctikernels.gdn.triton import _train_state as S
from ctikernels.gdn.triton import training

KERNELS = {  # name -> (jit function, candidate space)
    "features": (F._features_kernel, {"BT": [16, 32, 64], "num_warps": [2, 4, 8]}),
    "solve": (I._solve_kernel, {"BK": [64, 128], "BW": [32, 64], "num_warps": [2, 4, 8]}),
    "states": (S._states_kernel, {"BV": [16, 32, 64], "num_warps": [2, 4, 8],
                                  "num_stages": [1, 2, 3]}),
    "gated_norm": (O._gated_norm_kernel, {"BT": [8, 16, 32], "num_warps": [4, 8]}),
    "state_backward": (S._state_bwd_kernel, {"BV": [16, 32, 64], "num_warps": [2, 4, 8],
                                             "num_stages": [1, 2, 3]}),
    "dqkw": (O._dqkw_kernel, {"BK": [32, 64], "BV": [32, 64], "num_warps": [4, 8],
                              "num_stages": [1, 2, 3]}),
    "wy_backward": (I._wy_bwd_kernel, {"BD": [32, 64], "num_warps": [4, 8],
                                       "num_stages": [1, 2]}),
    "conv_backward": (C._conv_norm_bwd_kernel, {"BT": [8, 16, 32], "num_warps": [2, 4, 8]}),
}  # fmt: skip


def _calls(geometry: Geometry, tokens: int, configs: dict, device):
    """name -> f(config) launching that kernel alone on a training step's operands."""
    g = torch.Generator(device=device).manual_seed(0)
    hv = geometry.v_heads
    projected = torch.randn(tokens, geometry.proj_width, device=device, generator=g).bfloat16()
    conv_weight = (0.5 * torch.randn(geometry.conv_dim, 4, device=device, generator=g)).bfloat16()
    A_log = torch.rand(hv, device=device, generator=g) * 2.8 - 0.7
    dt_bias = torch.rand(hv, device=device, generator=g) * 4 - 2
    norm_weight = 1 + 0.1 * torch.randn(128, device=device, generator=g)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    chunks = training.Chunks.of(cu, tokens)
    table, offsets, cap = chunks.table, chunks.offsets, chunks.capacity
    sequence = chunks.sequence
    dc, vw = geometry.conv_dim, geometry.value_width
    a, b = projected[:, dc + vw + hv :], projected[:, dc + vw : dc + vw + hv]
    q, k, v, qr, kr, beta = F.features(geometry, projected, conv_weight, sequence, cu,
                                       configs["features"])  # fmt: skip
    gamma, A, w, u = I.solve(k, v, a, beta, A_log, dt_bias, table, cu, configs["solve"])
    o, h, vn, _ = S.states(q, k, w, u, gamma, cu, offsets, cap, training.SCALE, configs["states"])
    norm = (projected[:, dc : dc + vw], norm_weight)
    do = torch.randn_like(o)
    dh, du, _ = S.state_backward(q, k, w, gamma, do, cu, offsets, cap, training.SCALE,
                                 configs["state_backward"])  # fmt: skip
    dq, dk, dw, parts = O.dqkw(q, k, vn, gamma, h, do, dh, du, table, cu, training.SCALE,
                               configs["dqkw"])  # fmt: skip
    grad = torch.empty_like(projected)
    dkt, dv = I.wy_backward(k, v, beta, gamma, A, dw, du, parts, a, b, A_log, dt_bias,
                           grad[:, dc + vw + hv :], grad[:, dc + vw : dc + vw + hv], table, cu,
                           configs["wy_backward"])  # fmt: skip
    s = training.SCALE
    return {
        "features": lambda c: F.features(geometry, projected, conv_weight, sequence, cu, c),
        "solve": lambda c: I.solve(k, v, a, beta, A_log, dt_bias, table, cu, c),
        "states": lambda c: S.states(q, k, w, u, gamma, cu, offsets, cap, training.SCALE, c),
        "gated_norm": lambda c: O.gated_norm(o, *norm, c),
        "state_backward": lambda c: S.state_backward(q, k, w, gamma, do, cu, offsets, cap, s, c),
        "dqkw": lambda c: O.dqkw(q, k, vn, gamma, h, do, dh, du, table, cu, s, c),
        "wy_backward": lambda c: I.wy_backward(
            k, v, beta, gamma, A, dw, du, parts, a, b, A_log, dt_bias,
            grad[:, dc + vw + hv :], grad[:, dc + vw : dc + vw + hv], table, cu, c),
        "conv_backward": lambda c: C.conv_norm_backward(
            geometry, projected[:, :dc], conv_weight, sequence, cu, q, k, qr, kr, dq, dk, dkt, dv,
            grad[:, :dc], c),
    }  # fmt: skip


def _compiled(fn):
    """The most recently compiled kernel of a jit function on the current device."""
    cache = fn.device_caches[torch.cuda.current_device()][0]
    return list(cache.values())[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="16,32")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--kernels", default=",".join(KERNELS))
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = Geometry(*(int(x) for x in args.geometry.split(",")))
    defaults = training.configs(torch.cuda.get_device_capability(device))
    calls = _calls(geometry, args.tokens, defaults, device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name in args.kernels.split(","):
        fn, space = KERNELS[name]
        keys = list(space)
        for values in itertools.product(*(space[k] for k in keys)):
            config = dict(zip(keys, values, strict=True))
            try:
                calls[name](config)
                torch.cuda.synchronize()
                kernel = _compiled(fn)
                times = []
                for _ in range(args.reps):
                    start.record()
                    calls[name](config)
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                row = {"ms": statistics.median(times), "regs": kernel.n_regs,
                       "spills": kernel.n_spills}  # fmt: skip
            except Exception as error:  # noqa: BLE001 (out of resources and the like)
                row = {"error": f"{type(error).__name__}: {str(error)[:120]}"}
            row = {"device": torch.cuda.get_device_name(device), "geometry": args.geometry,
                   "tokens": args.tokens, "kernel": name, "config": config, "default":
                   config == defaults[name], **row}  # fmt: skip
            print("ROW", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
