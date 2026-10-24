# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The attention prologue (``ctikernels.attention.inference.Prologue``) and its CuTe DSL
kernel's loader.

``load(device, variant)`` is the variant's launcher on ``device``, or None for another
hardware type, ``CTI_PROLOGUE_CUTE=0``, or nvidia-cutlass-dsl (the ``cti-kernels[cute]`` extra)
not at ``VERSION``. Resolution is once per device, variant and process; only a resolution
that can use nvidia-cutlass-dsl imports it.
"""

from __future__ import annotations

import importlib.metadata
import os
import threading
import warnings
import weakref

import torch

from .._runtime.launch import prime

VARIABLE = "CTI_PROLOGUE_CUTE"
DISTRIBUTION = "nvidia-cutlass-dsl"
VERSION = "4.8.0"
# Device identities: (name, capability).
_DEVICES = frozenset(
    {
        ("NVIDIA GeForce RTX 5090", (12, 0)),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)),
        ("NVIDIA GB10", (12, 1)),
    }
)
HEAD_DIM = 256
ROTARY_DIM = 64
_DEFAULT_VARIANT = (HEAD_DIM, ROTARY_DIM, True)
DEVICES = {_DEFAULT_VARIANT: _DEVICES, (128, 128, False): _DEVICES}
_VALIDATED_LIMIT = 256
_MAX_STRIDE = 2**31  # the kernel's strides are 32-bit
_DTYPES = (torch.bfloat16,)  # qkv and weights
# Cache dtypes: BF16, or FP8 stored by the kernel as bytes (the PTX cvt format name).
_CACHE_DTYPES = {torch.bfloat16: None, torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}


class Prologue:
    """Q/K RMSNorm, NeoX RoPE and paged KV write, with no backward.

    Supported ``(head_dim, rotary_dim, zero_centered)``: ``(256, 64, True)`` (Qwen3.5)
    and ``(128, 128, False)`` (Qwen3 / DFlash). Norm scales by ``1 + weight`` when
    zero-centered, otherwise by ``weight``.

    ``op(qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices)``
    returns normalized, rotated Q as a ``[tokens, num_q_heads, head_dim]`` view of ``qkv``.
    K in ``qkv`` is unspecified; V is unchanged and copied bit for bit. Only indexed cache
    slots change. All tensors are on ``device``, with unit inner stride and 16-byte alignment
    (integer tensors and scales aligned to their element size). ``dtype`` (qkv, weights) must
    be BF16 and ``cache_dtype`` BF16, ``float8_e4m3fn`` or ``float8_e5m2``; other dtypes raise
    TypeError at construction:

    - ``qkv``: BF16 ``[tokens, (num_q_heads + 2 * num_kv_heads) * head_dim]``, rows [Q | K | V].
    - ``q_weight``, ``k_weight``: contiguous BF16 ``[head_dim]``.
    - ``cos_sin_cache``: contiguous FP32 ``[max_position, rotary_dim]``, cosines then sines.
    - ``positions``: contiguous int64 ``[tokens]``, valid table positions.
    - ``cache_indices``: contiguous int32 ``[tokens]``, each a cache slot in ``[0, slots)``, or
      negative when ``masked``: a negative index (padding) skips that token's K and V cache
      writes, and its Q is still normalized and rotated. Without ``masked`` negative indices
      are not allowed (the CuTe kernel would write before the cache), so pad with a sink slot.
      Tokens sharing a slot leave that slot's contents undefined (any mix of the writers'
      rows); every other slot is exact.
    - ``k_cache``, ``v_cache``: ``cache_dtype`` ``[slots, num_kv_heads, head_dim]``, same
      shape/strides, each slot contiguous. qkv's row stride is a multiple of 8 below 2**31, the
      caches' slot stride a multiple of 16 bytes below 2**31 elements.
    - ``k_scale``, ``v_scale`` (keywords, FP8 caches only): FP32 one-element tensors, the
      per-tensor scales; None means 1. The cache holds K / k_scale (V / v_scale) rounded once
      to FP8, saturated to its finite range (vLLM's ``reshape_and_cache_flash`` FP8 rule). The
      CuTe kernel quantizes K's FP32 result; ``compose`` quantizes the BF16-rounded K.

    ``masked`` (construction) selects the CuTe kernel that skips negative slots; the default
    kernel has no such check. Floating inputs must not require grad. Metadata violations raise TypeError or ValueError;
    position and slot values are not checked on the host. Read-only ``provider`` is "cute"
    (FP32 math, one rounding per output) or "compose". Calls use the current stream without
    host synchronization and support capture after warm-up or eager compilation. A capture
    requiring compilation raises RuntimeError; warm up or run that token count eagerly first.
    """

    def __init__(
        self,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        eps: float,
        device: torch.device | str | int,
        head_dim: int = HEAD_DIM,
        rotary_dim: int = ROTARY_DIM,
        zero_centered: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_dtype: torch.dtype = torch.bfloat16,
        masked: bool = False,
        position_axes: tuple[int, ...] | None = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise TypeError(f"Prologue dtype must be torch.bfloat16, not {dtype!r}")
        if not isinstance(cache_dtype, torch.dtype) or cache_dtype not in _CACHE_DTYPES:
            raise TypeError(
                "Prologue cache_dtype must be torch.bfloat16, torch.float8_e4m3fn or "
                f"torch.float8_e5m2, not {cache_dtype!r}"
            )
        if not isinstance(masked, bool):
            raise TypeError(f"Prologue masked must be a bool, not {masked!r}")
        variant = (head_dim, rotary_dim, zero_centered)
        if variant not in DEVICES:
            raise ValueError(f"unsupported (head_dim, rotary_dim, zero_centered): {variant}")
        self._head_dim, self._rotary_dim, self._zero_centered = variant
        if position_axes is not None:
            position_axes = tuple(position_axes)
            if len(position_axes) != rotary_dim // 2 or not set(position_axes) <= {0, 1, 2}:
                raise ValueError(
                    f"position_axes must give each of the {rotary_dim // 2} rotary frequencies "
                    f"an axis in 0..2, not {position_axes!r}"
                )
        self.position_axes = position_axes
        if min(num_q_heads, num_kv_heads) <= 0:
            raise ValueError("head counts must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Prologue needs a CUDA device, not {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.num_q_heads, self.num_kv_heads, self.eps = num_q_heads, num_kv_heads, eps
        self.device = device
        self.cache_dtype = cache_dtype
        self.masked = masked
        self._cache = _CACHE_DTYPES[cache_dtype]
        # The scale of FP8 calls without one (no allocation per call).
        self._unit_scale = (
            None if self._cache is None else torch.ones(1, dtype=torch.float32, device=device)
        )
        self._kernel = load(device, variant)
        # Multi-axis ops keep the 1-D kernel for [T] positions and an axis kernel for [3, T].
        self._axis_kernel = (
            None if position_axes is None or self._kernel is None
            else load(device, variant, position_axes)
        )  # fmt: skip
        self._width = (num_q_heads + 2 * num_kv_heads) * head_dim
        self._q_width = num_q_heads * head_dim
        # Host fast path (see _check and _plan): tensors validated by the full check, and the
        # launch plan of each token count.
        self._bound = {}  # (role, id) -> (weakref, pointer, shape, dtype, strides)
        self._bound_caches = {}  # (k pointer, v pointer) -> (shape, strides, dtype)
        self._bound_rows = None  # (positions, cache_indices) of the last full check
        self._bound_scales = {}
        self._plans = {}  # tokens -> (compiled, Q programs, K programs, programs)
        self._axis_plans = {}  # the same for [3, T] positions
        # The scale arguments of BF16 calls, which the kernel does not read.
        self._unread_scales = (
            (torch.empty(1, dtype=torch.float32, device=device),) * 2
            if self._kernel is not None and self._cache is None
            else None
        )

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def rotary_dim(self) -> int:
        return self._rotary_dim

    @property
    def zero_centered(self) -> bool:
        return self._zero_centered

    @property
    def provider(self) -> str:
        return "cute" if self._kernel is not None else "compose"

    def warm_up(self, max_tokens: int) -> int:
        """Compile the CuTe configuration of every token count up to ``max_tokens``, so no
        such call or its capture compiles. Idempotent; returns the number newly compiled
        (0 for ``compose``). Raises RuntimeError inside a CUDA graph capture."""
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive integer, not {max_tokens!r}")
        if self._kernel is None:
            return 0
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prologue warm-up cannot run inside a CUDA graph capture")
        return sum(
            kernel.warm_up(
                max_tokens, self.num_q_heads, self.num_kv_heads, cache=self._cache,
                masked=self.masked,
            )
            for kernel in (self._kernel, self._axis_kernel)
            if kernel is not None
        )  # fmt: skip

    def __call__(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_indices: torch.Tensor,
        *,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self._check(
            qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
        )
        scales = self._scales(k_scale, v_scale)
        if tokens:
            if self._kernel is not None:
                if positions.dim() == 1:
                    plan = self._plans.get(tokens) or self._plan(tokens)
                else:  # [3, T]: the axis kernel reads them flat
                    plan = self._axis_plans.get(tokens) or self._plan(tokens, axes=True)
                    positions = positions.view(-1)
                compiled, q_programs, k_programs, programs = plan
                if scales is None:
                    scales = self._unread_scales
                else:
                    k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
                compiled(
                    qkv, q_weight, k_weight, cos_sin_cache, positions, cache_indices, k_cache,
                    v_cache, *scales, self.eps, q_programs, k_programs, programs,
                )  # fmt: skip
            else:
                _compose(
                    qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache,
                    cache_indices, self.eps, self.num_q_heads, self.num_kv_heads, self.head_dim,
                    self.zero_centered, scales, self.position_axes,
                )  # fmt: skip
        # qkv[:, :q_width].view(tokens, q_heads, head_dim), in one call (a one-token view
        # takes the contiguous row stride, as view does).
        q_width = self._q_width
        return qkv.as_strided(
            (tokens, self.num_q_heads, self.head_dim),
            (qkv.stride(0) if tokens > 1 else q_width, self.head_dim, 1),
        )

    def _plan(self, tokens: int, axes: bool = False) -> tuple:
        """The launch plan of ``tokens`` (compiling on first use outside a capture)."""
        kernel, plans = (
            (self._axis_kernel, self._axis_plans) if axes else (self._kernel, self._plans)
        )
        plan = kernel.plan(tokens, self.num_q_heads, self.num_kv_heads, self._cache, self.masked)
        if len(plans) >= _VALIDATED_LIMIT:
            plans.clear()
        plans[tokens] = plan
        return plan

    def _scales(self, k_scale, v_scale) -> tuple | None:
        """The (k, v) scale tensors of an FP8 call, None for BF16 caches."""
        if self._cache is None:
            if k_scale is not None or v_scale is not None:
                raise ValueError("k_scale and v_scale apply to FP8 caches only")
            return None
        return self._scale("k_scale", k_scale), self._scale("v_scale", v_scale)

    def _scale(self, name, scale) -> torch.Tensor:
        """``scale`` as a 1-element vector, checked fully on first use or a change (identity,
        pointer, shape, dtype), its grad rule every call."""
        if scale is None:
            return self._unit_scale
        bound = self._bound_scales
        entry = bound.get((name, id(scale)))
        if (
            entry is None
            or entry[0]() is not scale
            or scale.data_ptr() != entry[1]
            or scale.shape != entry[2]
            or scale.dtype is not torch.float32
            or scale.requires_grad
        ):
            self._check_scale(name, scale)
            if len(bound) >= _VALIDATED_LIMIT:
                bound.clear()
            bound[name, id(scale)] = (weakref.ref(scale), scale.data_ptr(), scale.shape)
        return scale.reshape(1)

    def _check_scale(self, name, scale) -> None:
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} must be a tensor or None, not {type(scale).__name__}")
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise TypeError(f"{name} must be a one-element torch.float32 tensor")
        if scale.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, not {scale.device}")
        if scale.requires_grad:
            raise ValueError(f"{name} must not require grad")
        if scale.data_ptr() % 4:
            raise ValueError(f"{name} must be 4-byte aligned")

    def _check(
        self, qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
    ) -> int:
        """The call's token count after its checks. Weights, table and caches that passed the
        full check (_check_full) are checked again only by identity per role (weights, table) or
        pointer (caches: new views per call), with shape, strides and dtype; positions and
        cache_indices by identity with the last fully checked pair; qkv in full. A CUDA pointer
        identifies its device (unified addressing), so a matching pointer keeps the device and
        alignment checks. Anything else runs the full check, which raises every error."""
        if (
            qkv.requires_grad
            or q_weight.requires_grad
            or k_weight.requires_grad
            or cos_sin_cache.requires_grad
            or k_cache.requires_grad
            or v_cache.requires_grad
        ):
            self._check_full(
                qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
            )
        shape, stride = qkv.shape, qkv.stride()
        tokens = shape[0] if len(shape) == 2 else -1
        bound, rows = self._bound, self._bound_rows
        if not (
            tokens >= 0
            and shape[1] == self._width
            and qkv.dtype is torch.bfloat16
            and stride[1] == 1
            and stride[0] % 8 == 0
            and stride[0] < _MAX_STRIDE
            and qkv.data_ptr() % 16 == 0
            and qkv.get_device() == self.device.index
            and _bound(bound, "q_weight", q_weight)
            and _bound(bound, "k_weight", k_weight)
            and _bound(bound, "cos_sin_cache", cos_sin_cache)
            and rows is not None
            and rows[0]() is positions
            and rows[1]() is cache_indices
            and positions.data_ptr() == rows[2]
            and cache_indices.data_ptr() == rows[3]
            and positions.shape == rows[4]
            and cache_indices.shape == rows[5]
            and rows[5][0] == tokens
            and positions.dtype is torch.int64
            and cache_indices.dtype is torch.int32
            and positions.stride(-1) == cache_indices.stride(0) == 1
            and k_cache.dtype is v_cache.dtype is self.cache_dtype
            and self._bound_caches.get((k_cache.data_ptr(), v_cache.data_ptr()))
            == (k_cache.shape, k_cache.stride(), k_cache.dtype)
            == (v_cache.shape, v_cache.stride(), v_cache.dtype)
        ):
            self._check_full(
                qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
            )
            self._bind(
                q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
            )
            tokens = qkv.shape[0]
        return tokens

    def _bind(self, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, indices):
        """Record tensors that passed the full check (see :meth:`_check`)."""
        bound = self._bound
        roles = ("q_weight", "k_weight", "cos_sin_cache")
        for role, tensor in zip(roles, (q_weight, k_weight, cos_sin_cache), strict=True):
            if len(bound) >= _VALIDATED_LIMIT:
                bound.clear()
            bound[role, id(tensor)] = (
                weakref.ref(tensor), tensor.data_ptr(), tensor.shape, tensor.dtype, tensor.stride()
            )  # fmt: skip
        if len(self._bound_caches) >= _VALIDATED_LIMIT:
            self._bound_caches.clear()
        self._bound_caches[k_cache.data_ptr(), v_cache.data_ptr()] = (
            k_cache.shape, k_cache.stride(), k_cache.dtype
        )  # fmt: skip
        self._bound_rows = (
            weakref.ref(positions), weakref.ref(indices), positions.data_ptr(),
            indices.data_ptr(), positions.shape, indices.shape,
        )  # fmt: skip

    def _check_full(
        self, qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices
    ) -> None:
        if q_weight.requires_grad or k_weight.requires_grad or cos_sin_cache.requires_grad:
            raise ValueError("Prologue's norm weights and cos/sin table must not require grad")
        if qkv.requires_grad or k_cache.requires_grad or v_cache.requires_grad:
            raise ValueError("Prologue writes qkv and the caches, which must not require grad")
        q_heads, kv_heads = self.num_q_heads, self.num_kv_heads
        tokens = qkv.shape[0] if qkv.dim() else -1
        bf16, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
        cache = self.cache_dtype
        # name: (tensor, dtype, rank, last dimension)
        expected = {
            "qkv": (qkv, bf16, 2, (q_heads + 2 * kv_heads) * self.head_dim),
            "q_weight": (q_weight, bf16, 1, self.head_dim),
            "k_weight": (k_weight, bf16, 1, self.head_dim),
            "cos_sin_cache": (cos_sin_cache, f32, 2, self.rotary_dim),
            "positions": (
                positions,
                i64,
                2 if self.position_axes is not None and positions.dim() == 2 else 1,
                tokens,
            ),
            "cache_indices": (cache_indices, i32, 1, tokens),
            "k_cache": (k_cache, cache, 3, self.head_dim),
            "v_cache": (v_cache, cache, 3, self.head_dim),
        }
        for name, (tensor, dtype, rank, size) in expected.items():
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must be {dtype}, not {tensor.dtype}")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}, not {tensor.device}")
            if tensor.dim() != rank or tensor.shape[-1] != size:
                raise ValueError(f"{name} has the wrong shape {tuple(tensor.shape)}")
            alignment = tensor.element_size() if dtype in (i32, i64) else 16
            if tensor.stride(-1) != 1 or tensor.data_ptr() % alignment:
                raise ValueError(f"{name} must be aligned with unit inner stride")
            if rank == 1 and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if not cos_sin_cache.is_contiguous():
            raise ValueError("cos_sin_cache must be contiguous")
        if positions.dim() == 2 and (positions.shape[0] != 3 or not positions.is_contiguous()):
            raise ValueError("multi-axis positions must be contiguous int64 [3, tokens]")
        if k_cache.shape != v_cache.shape or k_cache.stride() != v_cache.stride():
            raise ValueError("k_cache and v_cache must have one shape and layout")
        if k_cache.shape[1] != kv_heads or k_cache.stride(1) != self.head_dim:
            raise ValueError(
                f"caches must be [slots, {kv_heads}, {self.head_dim}], slots contiguous"
            )
        if qkv.stride(0) % 8 or qkv.stride(0) >= _MAX_STRIDE:
            raise ValueError("qkv's row stride must be a multiple of 8 below 2**31")
        slot_stride = k_cache.stride(0)
        if slot_stride * k_cache.element_size() % 16 or slot_stride >= _MAX_STRIDE:
            raise ValueError("the caches' slot stride must be a multiple of 16 bytes below 2**31")


def _bound(bound: dict, role: str, tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` passed the full check (see Prologue._check) in ``role``, unchanged.
    The role is part of the key, so a tensor checked as one argument never passes as another."""
    entry = bound.get((role, id(tensor)))
    return (
        entry is not None
        and entry[0]() is tensor
        and tensor.data_ptr() == entry[1]
        and tensor.shape == entry[2]
        and tensor.dtype is entry[3]
        and tensor.stride() == entry[4]
    )


def _compose(
    qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices, eps,
    q_heads, kv_heads, head_dim=HEAD_DIM, zero_centered=True, scales=None, position_axes=None,
) -> None:  # fmt: skip
    """The prologue from the norm, rotary and store operations, in that order; FP8 caches
    store K and V quantized from BF16."""
    from ..norm.inference import rms_norm_
    from ..rotary.inference import apply_rotary_
    from ..store import store_cache

    q, k, v = qkv.split([q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1)
    rms_norm_(q.view(-1, q_heads, head_dim), q_weight, eps, zero_centered=zero_centered)
    rms_norm_(k.view(-1, kv_heads, head_dim), k_weight, eps, zero_centered=zero_centered)
    apply_rotary_(q, k, cos_sin_cache, positions, head_dim=head_dim, position_axes=position_axes)
    if scales is not None:
        k, v = (_quantize(x, scale, k_cache.dtype) for x, scale in zip((k, v), scales, strict=True))
        k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    store_cache(k_cache, v_cache, cache_indices, k, v)


def _quantize(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x / scale`` rounded once to ``dtype`` with saturation to its finite range, as bytes."""
    limit = torch.finfo(dtype).max
    return (x.float() / scale).clamp_(-limit, limit).to(dtype).view(torch.uint8)


_lock = threading.Lock()
_compiling = threading.Lock()  # cute.compile changes process-wide DSL state
_launchers: dict[tuple, Launcher | None] = {}

# Decode batches run padded to CUDA-graph sizes (SGLang's list up to 64 requests), so a count up
# to 64 takes its padded size's configuration; larger calls their own.
_GRAPH_BUCKETS = (*range(1, 9), *range(10, 33, 2), *range(40, 65, 4))
# Calls moving at least this share of L2 load qkv evict-first and store Q evict-last.
_STREAM_SHARE = 1 / 7
# Head rows (tokens x Q + K + V heads, at the padded size) per configuration, as tuned on the
# RTX 5090: one head per warp in 2-warp programs from two tokens up to 144 rows, in 4-warp
# programs up to 384, in 8-warp programs up to 704 (832 when a token has at most 16 head
# rows); two heads per warp in 4-warp programs up to 4608. The hd128 variant starts with
# these same row thresholds and its actual byte traffic; GPU tuning may override the config.
_TWO_WARP_MAX_ROWS = 144
_FOUR_WARP_MAX_ROWS = 384
_EIGHT_WARP_MAX_ROWS = 704
_EIGHT_WARP_SLIM_MAX_ROWS = 832
_SLIM_ROWS_PER_TOKEN = 16
_TWO_HEADS_MAX_ROWS = 4608
# Larger streaming calls: 4 Q heads per warp in 8-warp programs below L2/5 moved, in 4-warp
# programs from L2/5, 2 heads per warp from L2/3.5 (DRAM-bound).
_STREAM_FOUR_WARPS_SHARE = 1 / 5
_STREAM_TWO_ROWS_SHARE = 1 / 3.5


def load(
    device: torch.device, variant: tuple = _DEFAULT_VARIANT, axes: tuple | None = None
) -> Launcher | None:
    """The variant's CuTe launcher on ``device`` (``axes``: per-frequency position axes, or
    None for 1-D positions), or None (see the module docstring)."""
    key = (device, variant, axes)
    if key not in _launchers:
        with _lock:
            if key not in _launchers:
                _launchers[key] = _resolve(device, variant, axes)
    return _launchers[key]


def _resolve(device: torch.device, variant: tuple, axes: tuple | None = None) -> Launcher | None:
    if device.type != "cuda" or os.environ.get(VARIABLE) == "0":
        return None
    try:
        if importlib.metadata.version(DISTRIBUTION) != VERSION:
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    hardware = torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)
    if hardware not in DEVICES[variant]:
        return None
    try:
        from . import _prologue_cute
    except ImportError as error:
        warnings.warn(
            f"{DISTRIBUTION} {VERSION} is installed but does not import ({error}); the attention "
            "prologue composes the norm, rotary and store operations",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return Launcher(device, _prologue_cute, variant, axes)


def _rows(target: int, heads: int) -> int:
    """The largest power of two that divides ``heads`` and is at most ``target``."""
    rows = 1
    while rows * 2 <= target and heads % (rows * 2) == 0:
        rows *= 2
    return rows


def config(
    tokens: int, q_heads: int, kv_heads: int, l2_bytes: int, head_dim: int = HEAD_DIM
) -> tuple:
    """(Q, K, V heads per warp, warps per program, streaming) for a call of ``tokens``."""
    padded = next((size for size in _GRAPH_BUCKETS if size >= tokens), tokens)
    per_token = q_heads + 2 * kv_heads
    rows = padded * per_token
    moved = 2 * tokens * per_token * head_dim * 2
    stream = moved >= _STREAM_SHARE * l2_bytes
    if rows <= _TWO_WARP_MAX_ROWS and padded > 1:
        return 1, 1, 1, 2, stream
    if rows <= _FOUR_WARP_MAX_ROWS:
        return 1, 1, 1, 4, stream
    if rows <= _EIGHT_WARP_MAX_ROWS or (
        rows <= _EIGHT_WARP_SLIM_MAX_ROWS and per_token <= _SLIM_ROWS_PER_TOKEN
    ):
        return 1, 1, 1, 8, stream
    two = _rows(2, q_heads), _rows(2, kv_heads), _rows(2, kv_heads)
    if rows <= _TWO_HEADS_MAX_ROWS or (stream and moved >= _STREAM_TWO_ROWS_SHARE * l2_bytes):
        return *two, 4, stream
    four = _rows(4, q_heads), _rows(2, kv_heads), _rows(2, kv_heads)
    if stream and moved >= _STREAM_FOUR_WARPS_SHARE * l2_bytes:
        return *four, 4, stream
    return *four, 8, stream


_default_config = config


class Launcher:
    """Launches the kernel on one device; each configuration compiles on its first call."""

    def __init__(
        self, device: torch.device, kernels, variant: tuple = _DEFAULT_VARIANT, axes=None
    ) -> None:
        self._variant, self._axes = variant, axes
        self._device = device
        arch = "sm_121a" if torch.cuda.get_device_capability(device) == (12, 1) else "sm_120a"
        self._options = f"--enable-tvm-ffi --gpu-arch {arch}"
        self._l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        self._kernels = kernels
        self._compiled: dict[tuple, object] = {}
        self._plans: dict[tuple, dict] = {}

    def __call__(
        self, qkv, q_weight, k_weight, cos_sin_cache, positions, k_cache, v_cache, cache_indices,
        eps, q_heads, kv_heads, *, scales=None, cache=None, config=None, masked=False,
    ) -> None:  # fmt: skip
        tokens = qkv.shape[0]
        if config is None:
            compiled, q_programs, k_programs, total_programs = self.plan(
                tokens, q_heads, kv_heads, cache, masked
            )
        else:
            compiled, q_programs, k_programs, total_programs = self._plan(
                tokens, q_heads, kv_heads, cache, masked, config
            )
        if cache is None:
            if scales is None:  # Direct tuning calls retain their original placeholder.
                scales = (cos_sin_cache.view(-1),) * 2
        else:
            k_cache, v_cache = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
        compiled(
            qkv, q_weight, k_weight, cos_sin_cache, positions, cache_indices, k_cache, v_cache,
            *scales, eps, q_programs, k_programs, total_programs,
        )  # fmt: skip

    def keys(self, max_tokens: int, q_heads: int, kv_heads: int, cache=None, masked=False) -> dict:
        """Each configuration key of calls with 1..max_tokens tokens, with its first count."""
        keys = {}
        for tokens in range(1, max_tokens + 1):
            config = _default_config(tokens, q_heads, kv_heads, self._l2_bytes, self._variant[0])
            key = (q_heads, kv_heads, *config, *self._variant, cache, masked, self._axes)
            keys.setdefault(key, tokens)
        return keys

    def warm_up(
        self, max_tokens: int, q_heads: int, kv_heads: int, cache=None, masked=False
    ) -> int:
        """Compile every configuration of calls up to ``max_tokens``; returns the new count."""
        before = len(self._compiled)
        for tokens in range(1, max_tokens + 1):
            self.plan(tokens, q_heads, kv_heads, cache, masked)
        return len(self._compiled) - before

    def plan(self, tokens, q_heads, kv_heads, cache=None, masked=False) -> tuple:
        """The compiled function and program counts for this token count."""
        geometry = (q_heads, kv_heads, cache, masked)
        plans = self._plans.get(geometry)
        if plans is None:
            plans = self._plans[geometry] = {}
        plan = plans.get(tokens)
        if plan is None:
            config = _default_config(tokens, q_heads, kv_heads, self._l2_bytes, self._variant[0])
            plan = self._plan(tokens, q_heads, kv_heads, cache, masked, config)
            plans[tokens] = plan
        return plan

    def _plan(self, tokens, q_heads, kv_heads, cache, masked, config) -> tuple:
        q_rows, k_rows, v_rows, warps, stream = config
        q_programs = -(-tokens * (q_heads // q_rows) // warps)
        k_programs = -(-tokens * (kv_heads // k_rows) // warps)
        v_programs = -(-tokens * (kv_heads // v_rows) // warps)
        key = (q_heads, kv_heads, *config, *self._variant, cache, masked, self._axes)
        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = self._compile(key)
        return compiled, q_programs, k_programs, q_programs + k_programs + v_programs

    def _compile(self, key: tuple):
        with _compiling:
            compiled = self._compiled.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prologue cannot compile during CUDA graph capture; call warm_up or run "
                        "that token count eagerly first"
                    )
                compiled = _compile(self._kernels.Kernel(*key), key[0], key[1], self._options)
                prime(compiled, self._load, key, compiled)
                self._compiled[key] = compiled
        return compiled

    def _load(self, key: tuple, compiled) -> None:
        """Launch ``compiled`` once on one-token scratch. CUDA loads a kernel lazily on its
        first launch, which takes milliseconds and stalls the host, so a configuration that
        was only compiled would pay that at its first serving call."""
        q_heads, kv_heads, q_rows, k_rows, v_rows, warps = key[:6]
        head_dim, rotary_dim, cache = self._variant[0], self._variant[1], key[10]
        device, bf16, f32 = self._device, torch.bfloat16, torch.float32
        qkv = torch.zeros(1, (q_heads + 2 * kv_heads) * head_dim, dtype=bf16, device=device)
        weight = torch.zeros(head_dim, dtype=bf16, device=device)
        table = torch.zeros(1, rotary_dim, dtype=f32, device=device)
        positions = torch.zeros(1 if self._axes is None else 3, dtype=torch.int64, device=device)
        slots = torch.zeros(1, dtype=torch.int32, device=device)
        caches = torch.zeros(2, 1, kv_heads, head_dim, dtype=bf16 if cache is None else torch.uint8,
                             device=device)  # fmt: skip
        scales = torch.ones(2, dtype=f32, device=device)
        q_programs = -(-(q_heads // q_rows) // warps)
        k_programs = -(-(kv_heads // k_rows) // warps)
        v_programs = -(-(kv_heads // v_rows) // warps)
        compiled(
            qkv, weight, weight, table, positions, slots, caches[0], caches[1], scales[:1],
            scales[1:], 0.0, q_programs, k_programs, q_programs + k_programs + v_programs,
        )  # fmt: skip


def _compile(kernel, q_heads: int, kv_heads: int, options: str):
    """``kernel`` compiled for TVM-FFI calls with torch tensors, on torch's current stream."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64, Uint8
    from cutlass.runtime import make_fake_stream

    head_dim, rotary_dim = kernel.head_dim, kernel.rotary_dim
    fake = cute.runtime.make_fake_tensor
    tokens = cute.sym_int()
    width = (q_heads + 2 * kv_heads) * head_dim
    qkv = fake(BFloat16, (tokens, width), (cute.sym_int(divisibility=8), 1), assumed_align=16)
    weight = fake(BFloat16, (head_dim,), (1,), assumed_align=16)
    table = fake(Float32, (cute.sym_int(), rotary_dim), (rotary_dim, 1), assumed_align=16)
    # Multi-axis positions arrive flat ([3, tokens] contiguous).
    positions = fake(Int64, (tokens if kernel.axes is None else cute.sym_int(),), (1,),
                     assumed_align=8)  # fmt: skip
    slots = fake(Int32, (tokens,), (1,), assumed_align=4)

    def cache():
        shape = (cute.sym_int(), kv_heads, head_dim)
        if kernel.cache is None:
            stride = (cute.sym_int(divisibility=8), head_dim, 1)
            return fake(BFloat16, shape, stride, assumed_align=16)
        stride = (cute.sym_int(divisibility=16), head_dim, 1)
        return fake(Uint8, shape, stride, assumed_align=16)

    scale = fake(Float32, (cute.sym_int(),), (1,), assumed_align=4)
    tensors = (qkv, weight, weight, table, positions, slots, cache(), cache(), scale, scale)
    scalars = (Float32(0.0), Int32(0), Int32(0), Int32(0))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, *tensors, *scalars, stream, options=options)
