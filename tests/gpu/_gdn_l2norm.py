# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Reference controls: pin ONLY bitwise identity checks; leave FP64 baselines unmodified."""

import copy
import importlib
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from ctikernels.gdn import _fla_fixed, _l2norm

# Provider attributes and FLA's original host entry points. Never patch the global provider.
_ORIGINALS = {
    "_training_chunk": ("fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule_fwd"),
    "_recompute_w_u": ("fla.ops.gated_delta_rule.wy_fast", "recompute_w_u_fwd"),
    "_chunk_h": ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_h"),
    "_dv_local": ("fla.ops.common.chunk_o", "chunk_bwd_dv_local"),
    "_dhu": ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_bwd_dhu"),
    "_dqkwg": ("fla.ops.common.chunk_o", "chunk_bwd_dqkwg"),
    "_wy_bwd": ("fla.ops.gated_delta_rule.wy_fast", "prepare_wy_repr_bwd"),
    "_local_cumsum": ("fla.ops.utils.cumsum", "chunk_local_cumsum"),
    "_gated_norm_fwd": ("fla.modules.fused_norm_gate", "layer_norm_gated_fwd"),
    "_l2norm": ("fla.modules.l2norm", "l2norm_fwd"),
    "_l2norm_bwd": ("fla.modules.l2norm", "l2norm_bwd"),
}


def unmodified_fla_reference_op(op):
    """Private copy of a fla-core op with ORIGINAL autotuned training and L2 entries.

    Used for FP64-relative precision, never for the bitwise identity assertions.
    """
    assert op.provider == "fla-core"
    _fla_fixed.pin_fla()
    reference = copy.copy(op)
    reference._kernel = copy.copy(op._kernel)
    for attribute, (module, name) in _ORIGINALS.items():
        setattr(reference._kernel, attribute, getattr(importlib.import_module(module), name))
    return reference


@contextmanager
def pinned_fla_reference(*, training=True):
    """ONLY for bitwise algorithm identity: FLA's public autograd chain, same configs.

    Keep this context through backward. Inference retains its original chunk launches,
    so its identity check passes training=False and pins only the existing fixed L2.
    """
    _fla_fixed.pin_fla()
    chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    with ExitStack() as stack:
        stack.enter_context(patch.object(chunk, "l2norm_fwd", _l2norm.l2norm_forward))
        stack.enter_context(patch.object(chunk, "l2norm_bwd", _l2norm.l2norm_backward))
        if training:
            for name in (
                "chunk_gated_delta_rule_fwd",
                "recompute_w_u_fwd",
                "chunk_gated_delta_rule_fwd_h",
                "chunk_bwd_dv_local",
                "chunk_gated_delta_rule_bwd_dhu",
                "chunk_bwd_dqkwg",
                "prepare_wy_repr_bwd",
                "chunk_local_cumsum",
                "gdn_gate_chunk_cumsum",
            ):
                stack.enter_context(patch.object(chunk, name, getattr(_fla_fixed, name)))
            norm = importlib.import_module("fla.modules.fused_norm_gate")
            stack.enter_context(
                patch.object(norm, "layer_norm_gated_fwd", _fla_fixed.layer_norm_gated_fwd)
            )
        yield
