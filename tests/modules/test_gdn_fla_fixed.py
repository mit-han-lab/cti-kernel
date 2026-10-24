# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU contracts for fixed FLA launches: no FLA import or CUDA execution required."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _fla_fixed as fixed

# Independent inventory from FLA's BT=64 training call graph (including L2).
KERNELS = {
    "l2norm_fwd_kernel",
    "l2norm_bwd_kernel",
    "gdn_gate_chunk_cumsum_scalar_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "merge_16x16_to_64x64_inverse_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "chunk_bwd_kernel_dv_local",
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
    "chunk_bwd_kernel_dqkwg",
    "prepare_wy_repr_bwd_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "layer_norm_gated_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
}


def test_inventory_and_observed_spilling_warp_counts():
    assert set(fixed.CONFIGS) == KERNELS
    for config in fixed.CONFIGS.values():
        assert isinstance(config["num_warps"], int) and isinstance(config["num_stages"], int)
    # Pin the measured spill-free tiles, warps and stages together.
    for name, expected in {
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {
            "BV": 32,
            "num_warps": 2,
            "num_stages": 2,
        },
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": {
            "BV": 32,
            "num_warps": 4,
            "num_stages": 2,
        },
        "chunk_bwd_kernel_dqkwg": {"BK": 32, "BV": 32, "num_warps": 4, "num_stages": 2},
        "prepare_wy_repr_bwd_kernel": {"num_warps": 4, "num_stages": 1},
    }.items():
        assert fixed.CONFIGS[name] == expected
    # fwd_h retains FLA's Blackwell recurrence-race restriction. WY's wider launch
    # is qualified separately by the GPU stability test, with FLA's own tiles.
    assert fixed.CONFIGS["chunk_gated_delta_rule_fwd_kernel_h_blockdim64"]["num_warps"] == 2


def test_launch_unwraps_heuristics_and_autotuner_without_running_them(monkeypatch):
    calls, compiled = [], object()

    class Jit:
        arg_names = ["g", "T"]

        def __getitem__(self, grid):
            def launch(**args):
                calls.append((grid(args), args))
                return compiled

            return launch

    # Triton's Heuristics also has arg_names: that must NOT terminate unwrapping.
    jit = Jit()
    tuner = SimpleNamespace(fn=jit, configs=[object()], arg_names=jit.arg_names)
    heuristic = SimpleNamespace(
        fn=tuner, arg_names=jit.arg_names, values={"USE_G": lambda args: args["g"] is not None}
    )
    monkeypatch.setitem(
        sys.modules, "fake_fla_fixed", SimpleNamespace(recompute_w_u_fwd_kernel=heuristic)
    )
    for tokens in (1, 201, 1070):
        assert (
            fixed._launch(
                "fake_fla_fixed",
                "recompute_w_u_fwd_kernel",
                lambda args: (args["T"],),
                g=object(),
                T=tokens,
            )
            is compiled
        )
    assert [call[1]["T"] for call in calls] == [1, 201, 1070]
    assert all(args["USE_G"] and args["num_warps"] == 4 for _, args in calls)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("wy_shared_mem", [False, True])
def test_forward_backward_allocations_tiles_and_reductions(monkeypatch, fused, wy_shared_mem):
    """Run the host chain on CPU; the fake JIT checks outputs before filling them.

    Covers padded chunk capacity, grouped heads, zero initialization, all launch
    arguments, and the grouped dq/dk and split-dg reductions after the JIT launch.
    """
    calls = {}
    monkeypatch.setitem(
        sys.modules, "triton", SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    )
    utils = SimpleNamespace(
        check_shared_mem=lambda arch="none", tensor_idx=0: (
            wy_shared_mem if arch == "none" else arch == "ada"
        ),
        IS_NVIDIA_HOPPER=False,
        TRITON_ABOVE_3_4_0=True,
        TRITON_ABOVE_3_7_1=True,
    )
    monkeypatch.setitem(sys.modules, "fla.utils", utils)
    monkeypatch.setitem(
        sys.modules,
        "fla.ops.gated_delta_rule.chunk",
        SimpleNamespace(**({"chunk_gated_delta_rule_fwd_intra": object()} if fused else {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "fla.ops.utils.solve_tril",
        SimpleNamespace(DOT_PRECISION_AUTOTUNE_LIST=["ieee"], IS_TMA_SUPPORTED=False),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    outputs = {
        "gdn_gate_chunk_cumsum_scalar_kernel": ("o",),
        "chunk_scaled_dot_kkt_fwd_kernel": ("A",),
        "merge_16x16_to_64x64_inverse_kernel": ("Ai",),
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": ("A",),
        "recompute_w_u_fwd_kernel": ("w", "u"),
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ("h", "v_new", "ht"),
        "chunk_fwd_kernel_o": ("o",),
        "chunk_bwd_kernel_dv_local": ("dv",),
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": ("dh", "dh0", "dv2"),
        "chunk_bwd_kernel_dqkwg": ("dq", "dk", "dw", "dg"),
        "prepare_wy_repr_bwd_kernel": ("dk", "dv", "db", "dg"),
        "chunk_local_cumsum_scalar_kernel": ("o",),
        "layer_norm_gated_fwd_kernel": ("y", "rstd"),
    }

    def launch(module, name, grid, **args):
        calls[name] = args
        if name == "chunk_bwd_kernel_dqkwg":
            # _tiles derives 64 here; the override must also size the grid and dg.
            assert (args["BK"], args["BV"]) == (32, 32)
            nk = args["K"] // args["BK"]
            assert nk == 4
            assert grid == (nk, len(args["chunk_indices"]), args["B"] * args["HV"])
            assert args["dg"].shape == (nk, *args["g"].shape)
        for key in outputs[name]:
            value = args[key]
            if value is not None:
                if key == "Ai" or (key == "A" and "solve" in name):
                    assert torch.count_nonzero(value) == 0
                value.fill_(1)
        return object()

    monkeypatch.setattr(fixed, "_launch", launch)
    q = torch.ones(1, 201, 8, 128, dtype=torch.bfloat16)
    v = torch.ones(1, 201, 24, 128, dtype=torch.bfloat16)
    beta = torch.ones(1, 201, 24)
    cu = torch.tensor([0, 64, 194, 201], dtype=torch.int32)
    indices = torch.tensor([[0, 0], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]], dtype=torch.int32)
    index = {"cu_seqlens": cu, "chunk_indices": indices}
    G, o, A, ht, h0, g_input = fixed.chunk_gated_delta_rule_fwd(
        q,
        q,
        v,
        beta,
        beta,
        128**-0.5,
        None,
        False,
        use_gate_in_kernel=True,
        A_log=torch.ones(24),
        dt_bias=torch.ones(24),
        **index,
    )
    assert o.shape == v.shape and A.shape == (1, 201, 24, 64)
    assert A.dtype == q.dtype and G.dtype == torch.float32
    assert ht is h0 is None and g_input is beta
    w, u = fixed.recompute_w_u_fwd(q, v, beta, A, G, **index)
    h, vn, _ = fixed.chunk_gated_delta_rule_fwd_h(q, w, u, G, **index)
    assert h.shape == (1, 6, 24, 128, 128)
    local = fixed.chunk_bwd_dv_local(q, q, o, G, scale=128**-0.5, **index)
    dh, dh0, du = fixed.chunk_gated_delta_rule_bwd_dhu(q, q, w, o, local, G, **index)
    assert dh.shape == h.shape and dh0 is None
    dq, dk, dw, dg = fixed.chunk_bwd_dqkwg(q, q, vn, o, h, dh, w, G, dv=du, **index)
    assert dq.shape == dk.shape == q.shape and torch.all(dq == 3)
    assert torch.all(dg == 4)  # K=128, overridden BK=32: sum four partial gate gradients
    dk2, dv, db, dg2 = fixed.prepare_wy_repr_bwd(q, v, beta, A, dw, du, G, **index)
    assert torch.all(dk2 == 3) and dv.shape == v.shape and db.dtype == beta.dtype
    fixed.chunk_local_cumsum(dg + dg2, 64, reverse=True, **index)
    norm = fixed.layer_norm_gated_fwd(
        o.reshape(-1, 128), o.reshape(-1, 128), torch.ones(128), None, is_rms_norm=True
    )
    assert norm[1] is None and norm[2].dtype == torch.float32
    assert calls["layer_norm_gated_fwd_kernel"]["NB"] == 1
    for name, args in calls.items():
        if "chunk_offsets" in args:
            assert args["chunk_offsets"].tolist() == [0, 1, 4, 5]
            assert args["chunk_offsets"].dtype == torch.int64
        if "STATE_V_FIRST" in args:
            assert args["STATE_V_FIRST"] is False
        if "BK" in args:
            expected_tiles = {
                "chunk_bwd_kernel_dqkwg": (32, 32),
                # FLA derives both tiles from its shared-memory check, without overrides.
                "prepare_wy_repr_bwd_kernel": (64, 64) if wy_shared_mem else (32, 32),
            }
            assert (args["BK"], args["BV"]) == expected_tiles.get(name, (64, 64))
    if not fused:
        args = calls["merge_16x16_to_64x64_inverse_kernel"]
        assert args["DOT_PRECISION"] == "ieee" and not args["USE_TMA"]


def test_missing_varlen_table_cannot_fall_back_to_synchronizing_fla_builder():
    with pytest.raises(ValueError, match="chunk_indices"):
        fixed.chunk_local_cumsum(torch.empty(1, 10, 24), 64, cu_seqlens=torch.tensor([0, 10]))
    tree = ast.parse(Path(fixed.__file__).read_text())
    assert not any(
        isinstance(n, ast.Attribute) and n.attr in {"item", "tolist", "run"} for n in ast.walk(tree)
    )


def test_input_guard_preserves_flas_contiguity_semantics(monkeypatch):
    calls = []
    monkeypatch.setattr(fixed, "_launch", lambda *args, **kwargs: calls.append(kwargs))
    g = torch.ones(1, 24, 201).transpose(1, 2)
    assert not g.is_contiguous()
    cu = torch.tensor([0, 201, 999, 999], dtype=torch.int32)[::2]
    table = torch.zeros(4, 4, dtype=torch.int32)[:, ::2]
    fixed.gdn_gate_chunk_cumsum(g, torch.ones(48)[::2], 64, cu_seqlens=cu, chunk_indices=table)
    fixed.chunk_local_cumsum(g, 64, cu_seqlens=cu, chunk_indices=table)
    for args in calls:
        assert all(
            value.is_contiguous() for value in args.values() if isinstance(value, torch.Tensor)
        )


def test_cute_training_holds_fixed_entries_and_resolves_the_fixed_gate(monkeypatch):
    import ctikernels.gdn as gdn
    from ctikernels.gdn import _cute, _fla, _l2norm

    from ._gdn_provider import fake_fla, fake_kernels

    fake_fla(monkeypatch, dispatch_disabled=True)
    fake_kernels(monkeypatch)
    chunk = object()
    provider = _cute.CuteTraining(chunk)
    assert provider._chunk is chunk
    assert provider._gated_norm_fwd is fixed.layer_norm_gated_fwd
    assert provider._local_cumsum is fixed.chunk_local_cumsum
    assert provider._l2norm is _l2norm.l2norm_forward
    assert provider._l2norm_bwd is _l2norm.l2norm_backward
    monkeypatch.delenv(_cute.VARIABLE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(_cute.importlib.metadata, "version", lambda distribution: _cute.VERSION)
    monkeypatch.setattr(_fla, "load", lambda *args: object())
    for name in ("_cute_fwd", "_cute_bwd"):
        monkeypatch.setattr(gdn, name, object(), raising=False)
    monkeypatch.setattr(_cute, "ChunkKernels", lambda device, fwd, bwd, gate: gate)
    assert _cute._resolve(torch.device("cuda", 0)) is fixed.gdn_gate_chunk_cumsum


def test_fp64_reference_rebinds_a_copy_without_mutating_the_product(monkeypatch):
    import importlib

    from ..gpu._gdn_l2norm import _ORIGINALS, unmodified_fla_reference_op
    from ._gdn_provider import fake_fla

    fake_fla(monkeypatch, dispatch_disabled=True)
    monkeypatch.setitem(
        sys.modules, "fla.modules.l2norm", SimpleNamespace(l2norm_fwd=object(), l2norm_bwd=object())
    )
    kernel = SimpleNamespace(**dict.fromkeys(_ORIGINALS, object()))
    op = SimpleNamespace(provider="fla-core", _kernel=kernel)
    before = vars(kernel).copy()
    reference = unmodified_fla_reference_op(op)
    assert reference is not op and reference._kernel is not kernel
    assert vars(kernel) == before
    for attribute, (module, name) in _ORIGINALS.items():
        assert getattr(reference._kernel, attribute) is getattr(
            importlib.import_module(module), name
        )


def test_pinned_identity_context_restores_all_fla_references(monkeypatch):
    import importlib

    from ..gpu._gdn_l2norm import pinned_fla_reference
    from ._gdn_provider import fake_fla

    fake_fla(monkeypatch, dispatch_disabled=True)
    chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    norm = importlib.import_module("fla.modules.fused_norm_gate")
    names = (
        "l2norm_fwd",
        "l2norm_bwd",
        "chunk_gated_delta_rule_fwd",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_bwd_dv_local",
        "chunk_gated_delta_rule_bwd_dhu",
        "chunk_bwd_dqkwg",
        "prepare_wy_repr_bwd",
        "chunk_local_cumsum",
        "gdn_gate_chunk_cumsum",
    )
    for name in names:
        monkeypatch.setattr(chunk, name, object(), raising=False)
    before = {name: getattr(chunk, name) for name in names}
    norm_before = norm.layer_norm_gated_fwd
    with pinned_fla_reference():
        assert chunk.chunk_gated_delta_rule_fwd is fixed.chunk_gated_delta_rule_fwd
        assert norm.layer_norm_gated_fwd is fixed.layer_norm_gated_fwd
    assert before == {name: getattr(chunk, name) for name in names}
    assert norm.layer_norm_gated_fwd is norm_before
    with pinned_fla_reference(training=False):
        assert chunk.chunk_gated_delta_rule_fwd is before["chunk_gated_delta_rule_fwd"]
        assert norm.layer_norm_gated_fwd is norm_before
    assert before == {name: getattr(chunk, name) for name in names}
