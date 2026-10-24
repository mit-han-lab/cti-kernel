# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Independent FlashInfer norm provider, including the Gemma market kernel."""

from ctikernels._flashinfer_version import require_allowed


def add_rms_norm_(x, residual, weight, eps, *, zero_centered=False):
    if weight.requires_grad:
        raise ValueError("add_rms_norm_ requires a frozen weight")
    require_allowed()
    import flashinfer as norm

    fn = norm.gemma_fused_add_rmsnorm if zero_centered else norm.fused_add_rmsnorm
    # Preserve FlashInfer's default PDL choice (enabled on supported devices).
    # The fused norm ABI is 2-D. Views preserve caller storage and flattened M.
    hidden = x.shape[-1]
    m = x.numel() // hidden
    flat_x = x if x.ndim == 2 else x.view(m, hidden)
    flat_residual = residual if residual.ndim == 2 else residual.view(m, hidden)
    fn(flat_x, flat_residual, weight, eps)
    return x, residual
