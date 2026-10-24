# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Private RMS-norm launches; the public leaves own the gradient guards.

Grouped and residual-stream norms support both standard
(``weight``) and zero-centered (``1 + weight``) scales. The historical filename
is retained to avoid churn for private callers.

Grouped forward follows the Qwen3.5 grouped norm contributed by Waterpine.
Qwen3.5 arithmetic and flash-attention 47e91f1f7dd8a22649cee0dd9a182b69ffe782ef
(flash_attn/ops/triton/layer_norm.py) informed numerical behavior.

The grouped kernels normalize rows of any ``[..., groups, hidden]`` view whose prefix
collapses to two strides, such as Q/K heads inside a merged projection. The residual-stream
kernels serve every norm of a residual stream, in training and in inference: the add+norm and
the norm of a residual alone. Inference inputs have a dynamic row stride; training inputs
remain contiguous. An optional auxiliary sum output also has a dynamic row stride. Existing call modes remain runtime flags; the auxiliary output
is specialized away when absent.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The devices whose warp counts were measured; every other keeps the portable launch.
_QUALIFIED_HARDWARE = frozenset({("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0))})


@triton.jit
def _rms_norm_kernel(
    x,
    weight,
    output,
    num_groups,
    x_outer_stride,
    x_group_stride,
    output_outer_stride,
    output_group_stride,
    hidden_size: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    outer = row // num_groups
    group = row % num_groups
    if hidden_size <= BLOCK_SIZE:
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden_size

        x_offsets = outer * x_outer_stride + group * x_group_stride + columns
        values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
        normalized = values * tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
        output_offsets = outer * output_outer_stride + group * output_group_stride + columns
        tl.store(
            output + output_offsets,
            normalized * ((1.0 + weights) if ZERO_CENTERED else weights),
            mask=mask,
        )

    else:
        lanes = tl.arange(0, BLOCK_SIZE)
        square_sum = tl.full((), 0, tl.float32)
        for start in range(tl.cdiv(hidden_size, BLOCK_SIZE)):
            columns = start.to(tl.int64) * BLOCK_SIZE + lanes
            mask = columns < hidden_size
            offsets = outer * x_outer_stride + group * x_group_stride + columns
            values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
            square_sum += tl.sum(values * values, axis=0)
        inv_rms = tl.rsqrt(square_sum / hidden_size + eps)
        for start in range(tl.cdiv(hidden_size, BLOCK_SIZE)):
            columns = start.to(tl.int64) * BLOCK_SIZE + lanes
            mask = columns < hidden_size
            offsets = outer * x_outer_stride + group * x_group_stride + columns
            values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
            weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
            normalized = values * inv_rms
            output_offsets = outer * output_outer_stride + group * output_group_stride + columns
            tl.store(
                output + output_offsets,
                normalized * ((1.0 + weights) if ZERO_CENTERED else weights),
                mask=mask,
            )


@triton.jit
def _rms_norm_backward_kernel(
    x,
    weight,
    grad_output,
    grad_x,
    num_groups,
    x_outer_stride,
    x_group_stride,
    grad_outer_stride,
    grad_group_stride,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    outer = row // num_groups
    group = row % num_groups
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size

    x_offsets = outer * x_outer_stride + group * x_group_stride + columns
    grad_offsets = outer * grad_outer_stride + group * grad_group_stride + columns
    values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    upstream = tl.load(grad_output + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    # Recompute the inverse RMS rather than saving it; the input is read here anyway.
    scale = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    normalized = values * scale
    # The FP32 gradient reaching the normalized row is upstream * (1 + weight).
    scaled_upstream = upstream * (1.0 + weights)
    mean_projection = tl.sum(scaled_upstream * normalized, axis=0) / hidden_size
    grad = scale * (scaled_upstream - normalized * mean_projection)
    tl.store(grad_x + row * hidden_size + columns, grad, mask=mask)


@triton.jit(do_not_specialize=["has_x", "store_residual", "normalize_stored"])
def _residual_norm_kernel(
    x,
    residual,
    residual_out,
    weight,
    output,
    has_x,
    store_residual,
    normalize_stored,
    sum_out,
    sum_stride,
    x_stride,
    residual_stride,
    HAS_SUM: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    hidden_size: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Rows are strided: x and output share x_stride, residual and residual_out residual_stride.
    row = tl.program_id(0).to(tl.int64)
    if hidden_size <= BLOCK_SIZE:
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden_size
        offsets = row * x_stride + columns
        residual_offsets = row * residual_stride + columns

        values = tl.load(residual + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        if has_x:
            values += tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
            # The sum as stored: rounded in a narrow residual, exact in an FP32 one.
            stored = values.to(residual_out.dtype.element_ty)
            if store_residual:
                tl.store(residual_out + residual_offsets, stored, mask=mask)
            if HAS_SUM:
                tl.store(sum_out + row * sum_stride + columns, stored, mask=mask)
            # The norm reads the stored sum or, as FlashInfer's fused add does, the FP32 sum.
            if normalize_stored:
                values = stored.to(tl.float32)
        weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
        normalized = values * tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
        scale = (1.0 + weights) if ZERO_CENTERED else weights
        tl.store(output + offsets, normalized * scale, mask=mask)

    else:
        lanes = tl.arange(0, BLOCK_SIZE)
        square_sum = tl.full((), 0, tl.float32)
        for start in range(tl.cdiv(hidden_size, BLOCK_SIZE)):
            columns = start.to(tl.int64) * BLOCK_SIZE + lanes
            mask = columns < hidden_size
            values = tl.load(residual + row * residual_stride + columns, mask=mask, other=0.0).to(
                tl.float32
            )
            if has_x:
                values += tl.load(x + row * x_stride + columns, mask=mask, other=0.0).to(tl.float32)
                if normalize_stored:
                    values = values.to(residual_out.dtype.element_ty).to(tl.float32)
            square_sum += tl.sum(values * values, axis=0)
        inv_rms = tl.rsqrt(square_sum / hidden_size + eps)
        for start in range(tl.cdiv(hidden_size, BLOCK_SIZE)):
            columns = start.to(tl.int64) * BLOCK_SIZE + lanes
            mask = columns < hidden_size
            offsets = row * x_stride + columns
            residual_offsets = row * residual_stride + columns
            values = tl.load(residual + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            if has_x:
                values += tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
                stored = values.to(residual_out.dtype.element_ty)
                if store_residual:
                    tl.store(residual_out + residual_offsets, stored, mask=mask)
                if HAS_SUM:
                    tl.store(sum_out + row * sum_stride + columns, stored, mask=mask)
                if normalize_stored:
                    values = stored.to(tl.float32)
            weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
            normalized = values * inv_rms
            scale = (1.0 + weights) if ZERO_CENTERED else weights
            tl.store(output + offsets, normalized * scale, mask=mask)


@triton.jit(do_not_specialize=["saved_inputs", "has_grad_residual_out", "has_grad_x"])
def _residual_norm_backward_kernel(
    x,
    residual,
    weight,
    grad_output,
    grad_residual_out,
    grad_x,
    grad_residual,
    saved_inputs,
    has_grad_residual_out,
    has_grad_x,
    ZERO_CENTERED: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size
    offsets = row * hidden_size + columns

    values = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    if saved_inputs:
        # Rebuild the stored sum from its summands with the forward's add and rounding.
        values += tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        values = values.to(grad_residual.dtype.element_ty).to(tl.float32)
    upstream = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    # The grouped backward's formula, then the residual output's own gradient, all in FP32.
    scale = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    normalized = values * scale
    scaled_upstream = upstream * ((1.0 + weights) if ZERO_CENTERED else weights)
    mean_projection = tl.sum(scaled_upstream * normalized, axis=0) / hidden_size
    grad = scale * (scaled_upstream - normalized * mean_projection)
    if has_grad_residual_out:
        grad += tl.load(grad_residual_out + offsets, mask=mask, other=0.0).to(tl.float32)
    # Each gradient is that one FP32 value rounded once to its own dtype.
    tl.store(grad_residual + offsets, grad, mask=mask)
    if has_grad_x:
        tl.store(grad_x + offsets, grad, mask=mask)


def _validate(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not x.is_cuda or x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("expected CUDA input with a nonempty hidden dimension")
    if x.dtype not in _DTYPES or weight.dtype not in _DTYPES:
        raise TypeError("x and weight must use FP16, BF16 or FP32")
    if weight.shape != (x.shape[-1],) or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must be a contiguous [hidden] tensor on the input device")


def _validate_residual(x: torch.Tensor, residual: torch.Tensor) -> None:
    if residual.shape != x.shape or residual.device != x.device:
        raise ValueError("residual must match the input shape and device")
    if residual.dtype not in (x.dtype, torch.float32):
        raise TypeError("residual must use the input dtype or FP32")
    # Different element widths can make overlapping views reach another row.
    # Compare byte spans; disjoint contiguous views of one allocation are safe.
    if (
        residual.dtype != x.dtype
        and x.numel()
        and residual.data_ptr() < x.data_ptr() + x.numel() * x.element_size()
        and x.data_ptr() < residual.data_ptr() + residual.numel() * residual.element_size()
    ):
        raise ValueError("an FP32 residual must not share storage with overlapping input bytes")


def _validate_residual_rms_norm(
    residual: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> None:
    _validate(residual, weight)
    # The output takes the add+norm's input role, whose residual is that dtype or FP32.
    if out_dtype not in _DTYPES or residual.dtype not in (out_dtype, torch.float32):
        raise TypeError("residual must use out_dtype or FP32, with out_dtype FP16, BF16 or FP32")


def _require_residual_norm_layout(*tensors: torch.Tensor) -> None:
    # Offsets are row * hidden, and Triton keys a binary on pointer alignment as well.
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.data_ptr() % 16:
            raise ValueError("residual-stream norms need contiguous, 16-byte aligned tensors")


def _require_strided_residual_norm_layout(*tensors: torch.Tensor) -> None:
    # The inference add+norm takes one row stride per tensor, with unit hidden stride.
    for tensor in tensors:
        if _row_stride(tensor) is None:
            raise ValueError(
                "residual-stream norms need tensors whose rows collapse to "
                "one stride over a contiguous hidden dimension"
            )


def _row_stride(tensor: torch.Tensor) -> int | None:
    hidden = tensor.shape[-1]
    if tensor.stride(-1) != 1:
        return None
    if tensor.ndim < 2:
        return hidden
    try:
        rows = tensor.view(-1, hidden)
    except RuntimeError:
        return None
    if rows.stride(1) != 1 and hidden > 1:
        return None
    return rows.stride(0) if rows.shape[0] > 1 else hidden


def _rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """View ``[..., groups, hidden]`` as ``[outer, groups, hidden]`` without a copy."""

    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its hidden dimension")
    groups = 1
    for size in (*reversed(tensor.shape[:-1]), 1):
        groups *= size
        try:
            return tensor.view(-1, groups, tensor.shape[-1])
        except RuntimeError:
            continue
    raise ValueError(f"{name} prefix cannot be represented by two strides")


def _dense_rows(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` as the residual-stream kernels' rows, copied only when autograd hands over
    another layout."""

    if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _num_warps(
    hidden_size: int,
    *,
    backward: bool,
    device_name: str,
    capability: tuple[int, int],
) -> int:
    """``num_warps`` of a launch over rows of ``hidden_size`` on the named device.

    Qualified hardware, from hidden size 2048 on, takes the counts a sweep of 1 to 32 warps
    over 1 to 32768 rows measured best over all row counts: 16 forward and 8 backward. Any
    other device or hidden size keeps the portable count. The count never depends on the row
    count, so neither does a row's reduction, and its bits.
    """

    if hidden_size >= 2048 and (device_name, capability) in _QUALIFIED_HARDWARE:
        return 8 if backward else 16
    return 32 if hidden_size >= 4096 else 4


@functools.cache
def _hardware(device_index: int) -> tuple[str, tuple[int, int]]:
    # Asked once per device, so a launch pays one dict lookup for it.
    return torch.cuda.get_device_name(device_index), torch.cuda.get_device_capability(device_index)


def _launch_options(
    hidden_size: int, device_index: int, *, backward: bool
) -> dict[str, int | bool]:
    # FMA stays off so the FP32 math follows the eager formula before the final rounding.
    # Both kernel pairs take the warps of their direction, and their row reductions compile
    # to one layout; the GPU tests that compare the two bitwise guard it across Triton
    # upgrades.
    device_name, capability = _hardware(device_index)
    num_warps = _num_warps(
        hidden_size, backward=backward, device_name=device_name, capability=capability
    )
    return {"num_warps": num_warps, "enable_fp_fusion": False}


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = True,
) -> None:
    num_outer, num_groups, hidden_size = x.shape
    output = output.view(num_outer, num_groups, hidden_size)
    _rms_norm_kernel[(num_outer * num_groups,)](
        x,
        weight,
        output,
        num_groups,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        hidden_size,
        eps,
        min(65536, triton.next_power_of_2(hidden_size)),
        zero_centered,
        **_launch_options(hidden_size, x.get_device(), backward=False),
    )


def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    grad_x = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return grad_x
    x_rows = _rows(x, "x")
    num_outer, num_groups, hidden_size = x_rows.shape
    try:
        upstream = grad_output.view(x_rows.shape)
        if upstream.stride(-1) != 1:
            raise ValueError("strided hidden dimension")
    except (RuntimeError, ValueError):
        upstream = grad_output.contiguous().view(x_rows.shape)
    _rms_norm_backward_kernel[(num_outer * num_groups,)](
        x_rows,
        weight,
        upstream,
        grad_x,
        num_groups,
        x_rows.stride(0),
        x_rows.stride(1),
        upstream.stride(0),
        upstream.stride(1),
        hidden_size,
        eps,
        triton.next_power_of_2(hidden_size),
        **_launch_options(hidden_size, x.get_device(), backward=True),
    )
    return grad_x


def _launch_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    *,
    has_x: bool,
    store_residual: bool,
    normalize_stored: bool,
    sum_out: torch.Tensor | None = None,
    zero_centered: bool = True,
) -> None:
    hidden_size = residual.shape[-1]
    _residual_norm_kernel[(residual.numel() // hidden_size,)](
        x,
        residual,
        residual_out,
        weight,
        output,
        # Triton types a bool argument i1 and an int i32; ints keep every mode in one binary.
        int(has_x),
        int(store_residual),
        int(normalize_stored),
        residual_out if sum_out is None else sum_out,
        hidden_size if sum_out is None else sum_out.stride(0),
        _row_stride(x),
        _row_stride(residual),
        sum_out is not None,
        zero_centered,
        hidden_size,
        eps,
        min(65536, triton.next_power_of_2(hidden_size)),
        **_launch_options(hidden_size, residual.get_device(), backward=False),
    )


def _add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(normalized, summed)`` into new storage: the forward of :func:`training_add_rms_norm`,
    not the in-place inference :func:`add_rms_norm_`."""

    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    summed = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    if x.numel():
        # Training normalizes the sum as stored, which backward reads or rebuilds and which a
        # checkpoint segment ending in a torch add hands to the next one's residual_rms_norm.
        _launch_residual_norm(
            x,
            residual,
            summed,
            weight,
            output,
            eps,
            has_x=True,
            store_residual=True,
            normalize_stored=True,
            zero_centered=zero_centered,
        )
    return output, summed


def _residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
    *,
    zero_centered: bool = True,
) -> torch.Tensor:
    output = torch.empty(residual.shape, dtype=out_dtype, device=residual.device)
    if residual.numel():
        # Absent operands are stand-ins of their role's dtype: the output for x, the residual
        # for the sum it does not store.
        _launch_residual_norm(
            output,
            residual,
            residual,
            weight,
            output,
            eps,
            has_x=False,
            store_residual=False,
            normalize_stored=False,
            zero_centered=zero_centered,
        )
    return output


def _residual_norm_backward(
    saved_x: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_residual_out: torch.Tensor | None,
    eps: float,
    x_dtype: torch.dtype,
    *,
    write_grad_x: bool,
    zero_centered: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``(grad_x, grad_residual)``: the stored value's FP32 gradient rounded once to the
    residual's dtype and, with ``write_grad_x``, once to ``x_dtype``. ``saved_x`` means
    ``residual`` holds the input residual and the stored sum is rebuilt from the two."""

    grad_residual = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    grad_x = (
        torch.empty(residual.shape, dtype=x_dtype, device=residual.device) if write_grad_x else None
    )
    if residual.numel() == 0:
        return grad_x, grad_residual
    if grad_output is None:
        # Only the residual output's own gradient arrived; the norm adds nothing to it.
        grad_output = torch.zeros(residual.shape, dtype=x_dtype, device=residual.device)
    grad_output = _dense_rows(grad_output)
    hidden_size = residual.shape[-1]
    # Absent operands are stand-ins of their role's dtype, as in the forward.
    _residual_norm_backward_kernel[(residual.numel() // hidden_size,)](
        grad_output if saved_x is None else saved_x,
        residual,
        weight,
        grad_output,
        residual if grad_residual_out is None else _dense_rows(grad_residual_out),
        grad_output if grad_x is None else grad_x,
        grad_residual,
        int(saved_x is not None),
        int(grad_residual_out is not None),
        int(grad_x is not None),
        zero_centered,
        hidden_size,
        eps,
        triton.next_power_of_2(hidden_size),
        **_launch_options(hidden_size, residual.get_device(), backward=True),
    )
    return grad_x, grad_residual


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        return rms_norm(x, weight, eps, out_dtype=out_dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        x, weight = ctx.saved_tensors
        return _backward(x, weight, grad_output, ctx.eps), None, None, None


class _AddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        save_inputs: bool,
        zero_centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.zero_centered = zero_centered
        output, summed = _add_rms_norm(x, residual, weight, eps, zero_centered=zero_centered)
        # An unused output hands backward None rather than zeros, so its read is skipped.
        ctx.set_materialize_grads(False)
        ctx.eps, ctx.x_dtype = eps, x.dtype
        ctx.save_for_backward(*((x, residual) if save_inputs else (None, summed)), weight)
        return output, summed

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_output: torch.Tensor | None,
        grad_summed: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        saved_x, residual, weight = ctx.saved_tensors
        x_needs_grad, residual_needs_grad = ctx.needs_input_grad[:2]
        # A residual in x's dtype gives both inputs one gradient, returned as one tensor.
        separate = x_needs_grad and ctx.x_dtype != residual.dtype
        grad_x, grad_residual = _residual_norm_backward(
            saved_x,
            residual,
            weight,
            grad_output,
            grad_summed,
            ctx.eps,
            ctx.x_dtype,
            write_grad_x=separate,
            zero_centered=ctx.zero_centered,
        )
        if x_needs_grad and not separate:
            grad_x = grad_residual
        return grad_x, grad_residual if residual_needs_grad else None, None, None, None, None


class _ResidualRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out_dtype: torch.dtype,
        zero_centered: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.eps, ctx.out_dtype = eps, out_dtype
        ctx.zero_centered = zero_centered
        ctx.save_for_backward(residual, weight)
        # Autograd returns the unchanged residual as a view carrying this node, so the
        # gradient of its later uses joins the norm's in the one backward launch.
        return _residual_rms_norm(
            residual, weight, eps, out_dtype, zero_centered=zero_centered
        ), residual

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_output: torch.Tensor | None,
        grad_residual_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        residual, weight = ctx.saved_tensors
        _, grad = _residual_norm_backward(
            None,
            residual,
            weight,
            grad_output,
            grad_residual_out,
            ctx.eps,
            ctx.out_dtype,
            write_grad_x=False,
            zero_centered=ctx.zero_centered,
        )
        return grad, None, None, None, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
    zero_centered: bool = True,
) -> torch.Tensor:
    _validate(x, weight)
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if out_dtype not in _DTYPES:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    output = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if x.numel():
        _launch(_rows(x, "x"), weight, _rows(output, "output"), eps, zero_centered=zero_centered)
    return output


def rms_norm_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, *, zero_centered: bool = True
) -> torch.Tensor:
    _validate(x, weight)
    if x.numel():
        x_rows = _rows(x, "x")
        _launch(x_rows, weight, x_rows, eps, zero_centered=zero_centered)
    return x


def add_rms_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    store_residual: bool,
    normalize_stored: bool,
    sum_out: torch.Tensor | None = None,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(x, weight)
    _validate_residual(x, residual)
    if x.numel() == 0:
        return x, residual
    if residual.data_ptr() == x.data_ptr():
        raise ValueError("residual must be distinct from the input")
    _require_strided_residual_norm_layout(x, residual)
    _launch_residual_norm(
        x,
        residual,
        residual,
        weight,
        x,
        eps,
        has_x=True,
        store_residual=store_residual,
        normalize_stored=normalize_stored,
        sum_out=sum_out,
        zero_centered=zero_centered,
    )
    return x, residual


def residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype,
    zero_centered: bool = True,
) -> torch.Tensor:
    _validate_residual_rms_norm(residual, weight, out_dtype)
    _require_strided_residual_norm_layout(residual)
    return _residual_rms_norm(residual, weight, eps, out_dtype, zero_centered=zero_centered)


def training_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Out-of-place :func:`rms_norm` whose backward forms the input gradient only."""

    if torch.is_grad_enabled() and x.requires_grad:
        return _RMSNormFunction.apply(x, weight, eps, out_dtype)
    return rms_norm(x, weight, eps, out_dtype=out_dtype)


def training_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    save_inputs: bool,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Out-of-place :func:`add_rms_norm_` with ``normalize_stored`` set, whose backward is one
    launch for both inputs.

    Backward keeps the stored sum, or with ``save_inputs`` the two summands it rebuilds the
    sum from bit for bit."""

    _validate(x, weight)
    _validate_residual(x, residual)
    _require_residual_norm_layout(x, residual, weight)
    if torch.is_grad_enabled() and (x.requires_grad or residual.requires_grad):
        return _AddRMSNormFunction.apply(x, residual, weight, eps, save_inputs, zero_centered)
    return _add_rms_norm(x, residual, weight, eps, zero_centered=zero_centered)


def training_residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_dtype: torch.dtype,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`residual_rms_norm` and the residual passed through, whose gradient joins the
    norm's."""

    _validate_residual_rms_norm(residual, weight, out_dtype)
    _require_residual_norm_layout(residual, weight)
    if torch.is_grad_enabled() and residual.requires_grad:
        return _ResidualRMSNormFunction.apply(residual, weight, eps, out_dtype, zero_centered)
    return _residual_rms_norm(
        residual, weight, eps, out_dtype, zero_centered=zero_centered
    ), residual
